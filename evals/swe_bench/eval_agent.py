"""RL agent evaluation script for SWE-bench.

Runs the RL training agent (same system prompt, same text-based ``<bash>``
tool protocol) on SWE-bench-format instances and reports resolve rates.
Drives the agent loop directly via vLLM's OpenAI-compatible chat API —
does NOT use OpenRLHF.

Submission is sentinel-based: the model echoes
``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` followed by the patch content.

Supports multiple runs per instance (for GRPO signal analysis), concurrent
episode execution, and checkpoint-based resumability.

Usage::

    python -m evals.swe_bench.eval_agent \\
        --dataset-path /tmp/swe_gym_lite.jsonl \\
        --vllm-url http://vllm-server:8000/v1 \\
        --model-name Qwen/Qwen3.5-9B \\
        --output-path /tmp/baseline_4x.jsonl \\
        --runs-per-instance 4 \\
        --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from training.swe_bench.environment import SWEBenchEnvironment
from training.swe_bench.observation import format_observation
from training.swe_bench.system_prompt import build_system_prompt, get_system_message
from training.swe_bench.tool_parser import check_submission_sentinel, parse_tool_call

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    vllm_url: str
    model_name: str
    max_len: int
    max_output_chars: int
    max_steps: int
    runs_per_instance: int
    concurrency: int
    temperature: float
    seed: int
    image_registry: str
    k8s_namespace: str
    service_account: str
    timeout: int


@dataclass
class EpisodeResult:
    resolved: bool
    steps: int
    total_tokens: int
    reason: str  # "submit" | "max_steps" | "token_limit" | "error"
    error: str | None = None
    wall_clock_seconds: float = 0.0


# -- Helpers ---------------------------------------------------------------

_MAX_FORMAT_ERRORS = 3

_FORMAT_NUDGE = (
    "Your response did not include a bash command. Every response must "
    "contain at least one <bash>command</bash> block. Please try again."
)


def _count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _load_instances(path: Path) -> list[dict]:
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            instance = json.loads(entry["label"])
            instances.append(instance)
    # Group by repo so sandbox pods reuse cached container images.
    instances.sort(key=lambda i: (i.get("repo", ""), i.get("instance_id", "")))
    return instances


def _load_completed(path: Path) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                completed.add(result["instance_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


# -- Core agent loop -------------------------------------------------------


async def _run_episode(
    instance: dict,
    client: AsyncOpenAI,
    tokenizer,
    config: EvalConfig,
) -> EpisodeResult:
    instance_id = instance["instance_id"]
    t0 = time.monotonic()

    env = SWEBenchEnvironment(
        image_registry=config.image_registry,
        namespace=config.k8s_namespace or None,
        service_account=config.service_account,
    )

    try:
        await env.create(instance)
    except Exception as e:
        logger.error(f"[{instance_id}] Pod creation failed: {e}")
        return EpisodeResult(
            resolved=False,
            steps=0,
            total_tokens=0,
            reason="error",
            error=f"Pod creation failed: {e}",
            wall_clock_seconds=time.monotonic() - t0,
        )

    system_msg = get_system_message()
    instance_prompt = build_system_prompt(instance["problem_statement"])
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": instance_prompt},
    ]
    total_tokens = (
        _count_tokens(system_msg, tokenizer)
        + _count_tokens(instance_prompt, tokenizer)
    )
    steps_completed = 0
    format_errors = 0

    try:
        for step in range(config.max_steps):
            if config.timeout and (time.monotonic() - t0) >= config.timeout:
                logger.info(
                    f"[{instance_id}] Timeout after {config.timeout}s "
                    f"at step {steps_completed}"
                )
                resolved, _ = await env.run_eval()
                return EpisodeResult(
                    resolved=resolved,
                    steps=steps_completed,
                    total_tokens=total_tokens,
                    reason="timeout",
                    wall_clock_seconds=time.monotonic() - t0,
                )

            remaining = config.max_len - total_tokens
            if remaining <= 0:
                resolved, _ = await env.run_eval()
                return EpisodeResult(
                    resolved=resolved,
                    steps=steps_completed,
                    total_tokens=total_tokens,
                    reason="token_limit",
                    wall_clock_seconds=time.monotonic() - t0,
                )

            response = await client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                max_tokens=min(4096, remaining),
                temperature=config.temperature,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )

            action_text = response.choices[0].message.content or ""
            total_tokens += _count_tokens(action_text, tokenizer)
            messages.append({"role": "assistant", "content": action_text})

            tool_call = parse_tool_call(action_text)
            steps_completed = step + 1

            if tool_call.type == "submit":
                resolved, _ = await env.run_eval()
                return EpisodeResult(
                    resolved=resolved,
                    steps=steps_completed,
                    total_tokens=total_tokens,
                    reason="submit",
                    wall_clock_seconds=time.monotonic() - t0,
                )

            if not tool_call.content.strip():
                format_errors += 1
                if format_errors <= _MAX_FORMAT_ERRORS:
                    messages.append({"role": "user", "content": _FORMAT_NUDGE})
                    total_tokens += _count_tokens(_FORMAT_NUDGE, tokenizer)
                    continue
                resolved, _ = await env.run_eval()
                return EpisodeResult(
                    resolved=resolved,
                    steps=steps_completed,
                    total_tokens=total_tokens,
                    reason="format_error",
                    wall_clock_seconds=time.monotonic() - t0,
                )

            format_errors = 0

            try:
                stdout, rc = await env.execute(tool_call.content)
            except Exception as e:
                stdout = f"Error executing command: {e}"
                rc = 1

            patch = check_submission_sentinel(stdout)
            if patch is not None:
                resolved, _ = await env.run_eval()
                return EpisodeResult(
                    resolved=resolved,
                    steps=steps_completed,
                    total_tokens=total_tokens,
                    reason="submit",
                    wall_clock_seconds=time.monotonic() - t0,
                )

            feedback = format_observation(stdout, rc, config.max_output_chars)
            total_tokens += _count_tokens(feedback, tokenizer)
            messages.append({"role": "user", "content": feedback})

            if step == config.max_steps - 1:
                resolved, _ = await env.run_eval()
                return EpisodeResult(
                    resolved=resolved,
                    steps=steps_completed,
                    total_tokens=total_tokens,
                    reason="max_steps",
                    wall_clock_seconds=time.monotonic() - t0,
                )

        resolved, _ = await env.run_eval()
        return EpisodeResult(
            resolved=resolved,
            steps=0,
            total_tokens=total_tokens,
            reason="max_steps",
            wall_clock_seconds=time.monotonic() - t0,
        )

    except Exception as e:
        logger.error(
            f"[{instance_id}] Episode error at step {steps_completed}: {e}"
        )
        return EpisodeResult(
            resolved=False,
            steps=steps_completed,
            total_tokens=total_tokens,
            reason="error",
            error=str(e),
            wall_clock_seconds=time.monotonic() - t0,
        )
    finally:
        await env.destroy()


# -- Instance-level orchestration ------------------------------------------


async def _run_instance(
    instance: dict,
    client: AsyncOpenAI,
    tokenizer,
    config: EvalConfig,
    semaphore: asyncio.Semaphore,
) -> dict:
    instance_id = instance["instance_id"]
    repo = instance.get("repo", "")

    async def _one(k: int) -> EpisodeResult:
        async with semaphore:
            logger.info(
                f"[{instance_id}] Starting run "
                f"{k + 1}/{config.runs_per_instance}"
            )
            result = await _run_episode(instance, client, tokenizer, config)
            logger.info(
                f"[{instance_id}] Run {k + 1}: "
                f"{'RESOLVED' if result.resolved else 'NOT RESOLVED'} "
                f"({result.reason}, {result.steps} steps, "
                f"{result.total_tokens} tokens, "
                f"{result.wall_clock_seconds:.0f}s)"
            )
            return result

    results = await asyncio.gather(
        *[_one(k) for k in range(config.runs_per_instance)]
    )

    runs = [asdict(r) for r in results]
    n = len(results)
    resolved_count = sum(1 for r in results if r.resolved)

    return {
        "instance_id": instance_id,
        "repo": repo,
        "runs": runs,
        "resolve_rate": resolved_count / n if n else 0.0,
        "avg_steps": sum(r.steps for r in results) / n if n else 0.0,
        "avg_tokens": sum(r.total_tokens for r in results) / n if n else 0.0,
    }


# -- Top-level evaluation -------------------------------------------------


async def run_eval(
    dataset_path: Path,
    output_path: Path,
    config: EvalConfig,
) -> None:
    instances = _load_instances(dataset_path)
    logger.info(f"Loaded {len(instances)} instances from {dataset_path}")

    completed = _load_completed(output_path)
    if completed:
        logger.info(f"Resuming: {len(completed)} instances already completed")

    remaining = [i for i in instances if i["instance_id"] not in completed]
    logger.info(
        f"Running {len(remaining)} instances "
        f"({config.runs_per_instance} runs each, "
        f"concurrency={config.concurrency})"
    )

    if not remaining:
        logger.info("All instances already completed")
        _print_summary(output_path, config)
        return

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    client = AsyncOpenAI(base_url=config.vllm_url, api_key="unused")
    semaphore = asyncio.Semaphore(config.concurrency)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        _run_instance(inst, client, tokenizer, config, semaphore)
        for inst in remaining
    ]

    done_count = len(completed)
    total_count = len(instances)

    for coro in asyncio.as_completed(tasks):
        result = await coro
        done_count += 1

        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        logger.info(
            f"[{done_count}/{total_count}] {result['instance_id']}: "
            f"resolve_rate={result['resolve_rate']:.0%}"
        )

    _print_summary(output_path, config)


def _print_summary(output_path: Path, config: EvalConfig) -> None:
    all_results = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_results.append(json.loads(line))

    total = len(all_results)
    if not total:
        return

    resolved_any = sum(1 for r in all_results if r["resolve_rate"] > 0)
    avg_rate = sum(r["resolve_rate"] for r in all_results) / total
    avg_steps = sum(r["avg_steps"] for r in all_results) / total
    avg_tokens = sum(r["avg_tokens"] for r in all_results) / total

    reason_counts: dict[str, int] = {}
    for r in all_results:
        for run in r["runs"]:
            reason = run.get("reason", "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print(f"\n{'=' * 60}")
    print("Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"Instances:          {total}")
    print(f"Runs per instance:  {config.runs_per_instance}")
    print(
        f"Resolved (any):     {resolved_any}/{total} "
        f"({resolved_any / total:.1%})"
    )
    print(f"Avg resolve rate:   {avg_rate:.1%}")
    print(f"Avg steps:          {avg_steps:.1f}")
    print(f"Avg tokens:         {avg_tokens:.0f}")
    print(f"Episode outcomes:   {reason_counts}")
    print(f"Results:            {output_path}")
    print(f"{'=' * 60}")


# -- CLI -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the RL agent on SWE-bench instances",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="JSONL dataset (output of build_dataset.py)",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="Model name in vLLM (default: Qwen/Qwen3.5-9B)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output JSONL path (doubles as checkpoint for resumability)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=16384,
        help="Token budget for conversation (default: 16384)",
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=10000,
        help="Observation char limit before head/tail truncation (default: 10000)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Agent step limit per episode (default: 100)",
    )
    parser.add_argument(
        "--runs-per-instance",
        type=int,
        default=1,
        help="Episodes per instance, K for GRPO analysis (default: 1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max concurrent episodes (default: 16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Wall-clock timeout per episode in seconds (default: 3600)",
    )
    parser.add_argument(
        "--image-registry",
        type=str,
        default=os.environ.get("SWE_IMAGE_REGISTRY", ""),
        help="Container image registry override",
    )
    parser.add_argument(
        "--k8s-namespace",
        type=str,
        default=os.environ.get("SWE_K8S_NAMESPACE", ""),
        help="K8s namespace for sandbox pods",
    )
    parser.add_argument(
        "--service-account",
        type=str,
        default=os.environ.get("SWE_SERVICE_ACCOUNT", "swe-bench-training"),
        help="K8s service account (default: swe-bench-training)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = EvalConfig(
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        max_len=args.max_len,
        max_output_chars=args.max_output_chars,
        max_steps=args.max_steps,
        runs_per_instance=args.runs_per_instance,
        concurrency=args.concurrency,
        temperature=args.temperature,
        seed=args.seed,
        image_registry=args.image_registry,
        k8s_namespace=args.k8s_namespace,
        service_account=args.service_account,
        timeout=args.timeout,
    )

    asyncio.run(run_eval(args.dataset_path, args.output_path, config))


if __name__ == "__main__":
    main()
