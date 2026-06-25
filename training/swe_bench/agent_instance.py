"""OpenRLHF AgentInstanceBase implementation for SWE-bench.

Connects OpenRLHF's multi-turn agent loop to an execution environment.
Each instance manages one agent episode: reset() creates a sandbox,
step() executes tool calls, and the final step runs the SWE-bench
eval to produce a binary reward.

The execution backend is selected via the ``SWE_ENVIRONMENT`` env var:
  - ``k8s`` (default): K8s Pod sandbox via ``SWEBenchEnvironment``
  - ``local``: Sandlock-based local sandbox via ``LocalSWEBenchEnvironment``
"""

from __future__ import annotations

import json
import logging
import os

import torch
from openrlhf.utils.agent import AgentInstanceBase

try:
    from training.swe_bench.system_prompt import build_system_prompt
    from training.swe_bench.tool_parser import parse_tool_call
except ImportError:
    from system_prompt import build_system_prompt
    from tool_parser import parse_tool_call

logger = logging.getLogger(__name__)

class InfrastructureError(RuntimeError):
    """Raised when a rollout fails due to infrastructure, not model quality."""


_ENVIRONMENT = os.environ.get("SWE_ENVIRONMENT", "k8s")
_MAX_STEPS = int(os.environ.get("SWE_MAX_STEPS", "100"))
_MAX_OUTPUT_CHARS = int(os.environ.get("SWE_MAX_OUTPUT_CHARS", "8000"))
_IMAGE_REGISTRY = os.environ.get("SWE_IMAGE_REGISTRY", "")
_K8S_NAMESPACE = os.environ.get("SWE_K8S_NAMESPACE", "")
_SERVICE_ACCOUNT = os.environ.get("SWE_SERVICE_ACCOUNT", "swe-bench-training")


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... ({len(text) - limit} chars truncated) ...\n" + text[-half:]


def _format_tool_output(stdout: str, exit_code: int) -> str:
    truncated = _truncate(stdout)
    parts = [truncated]
    if exit_code != 0:
        parts.append(f"[exit code: {exit_code}]")
    return "\n".join(parts)


def _create_environment():
    if _ENVIRONMENT == "local":
        try:
            from training.swe_bench.cloudexe.local_environment import LocalSWEBenchEnvironment
        except ImportError:
            from cloudexe.local_environment import LocalSWEBenchEnvironment
        return LocalSWEBenchEnvironment(
            env_map_path=os.environ.get("SWE_ENV_MAP", "/root/swe-env-map.json"),
            repo_cache_dir=os.environ.get("SWE_REPO_CACHE", "/root/repo-cache"),
        )
    try:
        from training.swe_bench.environment import SWEBenchEnvironment
    except ImportError:
        from environment import SWEBenchEnvironment
    return SWEBenchEnvironment(
        image_registry=_IMAGE_REGISTRY,
        namespace=_K8S_NAMESPACE or None,
        service_account=_SERVICE_ACCOUNT,
    )


class SWEBenchAgentInstance(AgentInstanceBase):
    """One agent episode on a SWE-bench instance."""

    def __init__(self):
        self.env = _create_environment()
        self.step_count = 0
        self.max_steps = _MAX_STEPS
        self.instance_id = ""
        self._total_output_chars = 0
        self._nonzero_exits = 0

    async def reset(self, states: dict, **kwargs) -> dict:
        """Create the sandbox Pod and return the system prompt.

        OpenRLHF passes:
          states["observation"] = problem_statement (from dataset ``input`` field)
          states["label"] = instance metadata JSON (from dataset ``label`` field)
        """
        problem_statement = states["observation"]
        label = states.get("label", "{}")
        instance = json.loads(label) if isinstance(label, str) else label

        self.instance_id = instance.get("instance_id", "unknown")
        self.step_count = 0
        self._total_output_chars = 0
        self._nonzero_exits = 0

        await self.env.create(instance)

        prompt = build_system_prompt(problem_statement)
        return {"observation": prompt}

    async def step(self, state_dict: dict, **kwargs) -> dict:
        """Execute one tool call and return feedback + reward.

        OpenRLHF passes:
          state_dict["action_text"]      = model's latest generation
          state_dict["observation_text"] = full conversation so far
          state_dict["label"]            = ground truth label
        """
        action_text = state_dict.get("action_text", "")

        if self.step_count == 0 and "<think>" in action_text:
            logger.warning(
                f"[{self.instance_id}] Thinking mode appears to be ON — "
                f"model output contains <think> tags. "
                f"Ensure the dataset was built with enable_thinking=False."
            )

        tool_call = parse_tool_call(action_text)

        self.step_count += 1
        force_submit = self.step_count >= self.max_steps

        if tool_call.type == "submit" or force_submit:
            return await self._handle_submit()

        return await self._handle_bash(tool_call.content)

    async def _handle_bash(self, command: str) -> dict:
        try:
            stdout, exit_code = await self.env.execute(command)
        except Exception as e:
            stdout = f"Error executing command: {e}"
            exit_code = 1

        self._total_output_chars += len(stdout)
        if exit_code != 0:
            self._nonzero_exits += 1

        feedback = _format_tool_output(stdout, exit_code)

        return {
            "rewards": torch.tensor(0.0),
            "scores": torch.tensor(0.0),
            "environment_feedback": feedback,
            "done": False,
        }

    async def _handle_submit(self) -> dict:
        patch = ""
        try:
            patch = await self.env.get_patch()
            resolved, eval_output = await self.env.run_eval()
        except Exception as e:
            logger.error(f"[{self.instance_id}] Eval failed: {e}")
            raise InfrastructureError(
                f"[{self.instance_id}] Rollout discarded due to infrastructure failure: {e}"
            ) from e
        finally:
            await self.env.destroy()

        reward = 1.0 if resolved else 0.0

        patch_lines = patch.count("\n") if patch else 0
        files_changed = len(set(
            line.split(" b/", 1)[1]
            for line in patch.splitlines()
            if line.startswith("diff --git ")
        )) if patch else 0

        logger.info(
            f"[{self.instance_id}] Episode done: "
            f"{'RESOLVED' if resolved else 'NOT RESOLVED'} "
            f"in {self.step_count} steps, "
            f"{patch_lines} patch lines, {files_changed} files"
        )

        return {
            "rewards": torch.tensor(reward),
            "scores": torch.tensor(reward),
            "environment_feedback": "Patch submitted and evaluated.",
            "done": True,
            "extra_logs": {
                "resolved": reward,
                "steps": self.step_count,
                "patch_lines": patch_lines,
                "files_changed": files_changed,
                "empty_patch": float(not patch),
                "immediate_submit": float(self.step_count <= 1),
                "nonzero_exits": self._nonzero_exits,
                "total_output_chars": self._total_output_chars,
            },
        }
