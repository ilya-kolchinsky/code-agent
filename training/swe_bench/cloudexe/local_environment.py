"""Local process-based environment for SWE-bench RL training on Cloudexe.

Replaces the K8s Pod sandbox (``environment.py``) with a Sandlock-based
local execution model.  Each agent episode gets:

  - A cloned conda env (from a pre-built base) for Python isolation
  - A fresh repo checkout at ``/testbed`` for filesystem isolation
  - Sandlock (Landlock + seccomp-bpf) for sandboxing agent commands

Prerequisites:
  - Pre-built conda envs and repo cache via ``setup_local_envs.py``
  - ``pip install sandlock`` (Python SDK)
  - Symlink /opt/miniconda3 -> /opt/miniconda (Cloudexe uses /opt/miniconda)

Lifecycle (same interface as SWEBenchEnvironment):
  env = LocalSWEBenchEnvironment(env_map_path, repo_cache_dir)
  await env.create(instance_dict)
  out, rc = await env.execute("ls")
  patch   = await env.get_patch()
  ok, log = await env.run_eval()
  await env.destroy()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import uuid
from functools import partial
from pathlib import Path
from subprocess import PIPE, TimeoutExpired

from swebench.harness.constants import DOCKER_PATCH, DOCKER_WORKDIR
from swebench.harness.test_spec.test_spec import make_test_spec

from evals.swe_bench.grader import grade_instance

logger = logging.getLogger(__name__)

_EXEC_TIMEOUT = int(os.environ.get("SWE_EXEC_TIMEOUT", "120"))
_EVAL_TIMEOUT = int(os.environ.get("SWE_EVAL_TIMEOUT", "600"))
_SANDLOCK_MEMORY = os.environ.get("SWE_SANDLOCK_MEMORY", "4G")
_MINICONDA_PATH = os.environ.get("SWE_MINICONDA_PATH", "/opt/miniconda3")
_CONDA_BIN = os.path.join(_MINICONDA_PATH, "bin", "conda")

_RC_PATTERN = re.compile(r"__RC_8372916__:(\d+)\n?")

_GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

_SANDLOCK_READABLE_PATHS = [
    "/usr",
    "/lib",
    "/lib64",
    "/bin",
    "/sbin",
    "/etc",
]

try:
    from sandlock import Sandbox as _Sandbox
    _HAS_SANDLOCK = True
except ImportError:
    _HAS_SANDLOCK = False


def _extract_install_commands(repo_script_list: list[str]) -> list[str]:
    """Extract project install commands from the repo setup script.

    The repo_script_list includes git clone, checkout, and cleanup steps
    followed by conda activation and project install.  We only need
    the install part since we handle repo setup ourselves.
    """
    install_lines = []
    past_conda_activate = False

    for line in repo_script_list:
        if "conda activate" in line and "source" not in line:
            past_conda_activate = True
            continue
        if past_conda_activate:
            if line.startswith("echo ") and "Current environment" in line:
                continue
            install_lines.append(line)

    return install_lines


class LocalSWEBenchEnvironment:
    """Sandlock-based sandbox for an agent episode on Cloudexe."""

    def __init__(
        self,
        env_map_path: str = "/root/swe-env-map.json",
        repo_cache_dir: str = "/root/repo-cache",
    ):
        self._env_map_path = env_map_path
        self._repo_cache_dir = Path(repo_cache_dir)
        self._env_map: dict[str, str] = {}

        self._episode_env_name: str | None = None
        self._episode_env_dir: str | None = None
        self._test_spec = None
        self._eval_script: str = ""
        self._instance_id: str = ""
        self._repo_url: str = ""
        self._base_commit: str = ""
        self._install_commands: list[str] = []

    def _load_env_map(self) -> dict[str, str]:
        if not self._env_map:
            with open(self._env_map_path) as f:
                self._env_map = json.load(f)
        return self._env_map

    async def create(self, instance: dict) -> None:
        """Set up a sandboxed environment for one agent episode."""
        self._instance_id = instance["instance_id"]
        repo = instance.get("repo", "")

        test_spec = make_test_spec(instance)
        self._test_spec = test_spec
        self._eval_script = test_spec.eval_script
        self._base_commit = instance.get("base_commit", "")

        env_map = self._load_env_map()
        base_env_name = env_map.get(test_spec.env_image_key)
        if not base_env_name:
            raise RuntimeError(
                f"No pre-built conda env for {test_spec.env_image_key}. "
                f"Run setup_local_envs.py first."
            )

        ep_id = uuid.uuid4().hex[:8]
        self._episode_env_name = f"ep-{ep_id}"

        self._install_commands = _extract_install_commands(test_spec.repo_script_list)

        loop = asyncio.get_event_loop()

        # Clone conda env
        logger.info(
            f"[{self._instance_id}] Cloning conda env {base_env_name} "
            f"-> {self._episode_env_name}"
        )
        await loop.run_in_executor(None, partial(self._clone_conda_env, base_env_name))

        # Set up repo at /testbed
        repo_org, repo_name = repo.split("/") if "/" in repo else ("", repo)
        cache_path = self._repo_cache_dir / repo_org / f"{repo_name}.git"
        self._repo_url = f"https://github.com/{repo}"

        logger.info(f"[{self._instance_id}] Setting up /testbed")
        await loop.run_in_executor(
            None, partial(self._setup_testbed, cache_path, instance)
        )

        # Run project install
        if self._install_commands:
            logger.info(f"[{self._instance_id}] Running project install")
            install_script = "\n".join(self._install_commands)
            await self._run_in_env(install_script, timeout=300, sandbox=False)

        logger.info(f"[{self._instance_id}] Environment ready")

    def _clone_conda_env(self, base_env_name: str) -> None:
        import subprocess

        result = subprocess.run(
            [
                _CONDA_BIN, "create", "--clone", base_env_name,
                "-n", self._episode_env_name, "-y", "-q",
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"conda clone failed: {result.stderr}"
            )

        envs_output = subprocess.run(
            [_CONDA_BIN, "info", "--envs"], capture_output=True, text=True
        )
        for line in envs_output.stdout.splitlines():
            if self._episode_env_name in line:
                parts = line.split()
                if len(parts) >= 2:
                    self._episode_env_dir = parts[-1]
                    break

    def _setup_testbed(self, cache_path: Path, instance: dict) -> None:
        import subprocess

        testbed = Path(DOCKER_WORKDIR)
        if testbed.exists():
            shutil.rmtree(testbed, ignore_errors=True)
        testbed.mkdir(parents=True, exist_ok=True)

        clone_cmd = ["git", "clone"]
        if cache_path.exists():
            clone_cmd.extend(["--reference", str(cache_path)])
        clone_cmd.extend([
            "--single-branch", self._repo_url, str(testbed),
        ])

        result = subprocess.run(
            clone_cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr}")

        commit = instance.get("base_commit", "")
        if commit:
            subprocess.run(
                ["git", "reset", "--hard", commit],
                cwd=str(testbed), capture_output=True, text=True, timeout=30,
            )

        subprocess.run(
            ["git", "remote", "remove", "origin"],
            cwd=str(testbed), capture_output=True, text=True,
        )
        subprocess.run(
            ["chmod", "-R", "777", str(testbed)],
            capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "config", "--global", "user.email", "setup@swebench.config"],
            capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "config", "--global", "user.name", "SWE-bench"],
            capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "commit", "--allow-empty", "-am", "SWE-bench"],
            cwd=str(testbed), capture_output=True, text=True,
        )

    def _build_sandbox(self) -> "_Sandbox | None":
        """Build a Sandlock Sandbox configured for this episode, or None."""
        if not _HAS_SANDLOCK:
            logger.warning("sandlock package not installed — running without sandbox")
            return None

        readable = [p for p in _SANDLOCK_READABLE_PATHS if os.path.exists(p)]
        if os.path.exists(_MINICONDA_PATH):
            readable.append(_MINICONDA_PATH)
        if self._repo_cache_dir.exists():
            readable.append(str(self._repo_cache_dir))

        writable = [DOCKER_WORKDIR, "/tmp"]
        if self._episode_env_dir:
            writable.append(self._episode_env_dir)

        return _Sandbox(
            fs_readable=readable,
            fs_writable=writable,
            max_memory=_SANDLOCK_MEMORY,
        )

    async def _run_in_env(
        self,
        command: str,
        timeout: int = _EXEC_TIMEOUT,
        sandbox: bool = True,
    ) -> tuple[str, int]:
        """Run a command with the episode's conda env activated.

        If sandbox=True, wraps in Sandlock for filesystem/resource isolation.
        If sandbox=False, runs directly (used for trusted setup steps).
        """
        sentinel = "__RC_8372916__"
        activate = (
            f"source {_MINICONDA_PATH}/bin/activate {self._episode_env_name}"
        )
        wrapped = f'{activate} && cd {DOCKER_WORKDIR} && ({command}); echo "{sentinel}:$?"'
        cmd_list = ["/bin/bash", "-c", wrapped]

        loop = asyncio.get_event_loop()

        sb = self._build_sandbox() if sandbox else None
        if sb is not None:
            try:
                result = await loop.run_in_executor(
                    None, partial(sb.run, cmd_list, timeout=timeout),
                )
                raw_output = result.stdout.decode("utf-8", errors="replace")
                if result.stderr:
                    raw_output += "\n" + result.stderr.decode("utf-8", errors="replace")
            except Exception as e:
                if "timeout" in str(e).lower():
                    return f"Command timed out after {timeout}s", -1
                raise
        else:
            try:
                proc = await loop.run_in_executor(
                    None, partial(_run_subprocess, cmd_list, timeout=timeout + 30),
                )
                raw_output = proc.stdout or ""
                if proc.stderr:
                    raw_output += "\n" + proc.stderr
            except TimeoutExpired:
                return f"Command timed out after {timeout}s", -1

        rc_match = _RC_PATTERN.search(raw_output)
        if rc_match:
            exit_code = int(rc_match.group(1))
            output = (
                raw_output[: rc_match.start()] + raw_output[rc_match.end():]
            ).strip()
        else:
            output = raw_output.strip()
            exit_code = -1

        return output, exit_code

    async def execute(
        self, command: str, timeout: int = _EXEC_TIMEOUT
    ) -> tuple[str, int]:
        """Execute a bash command in the sandbox."""
        return await self._run_in_env(command, timeout=timeout, sandbox=True)

    async def get_patch(self) -> str:
        """Extract the current diff from the repo."""
        output, _ = await self._run_in_env(
            f"cd {DOCKER_WORKDIR} && git diff",
            timeout=30,
            sandbox=True,
        )
        return output.strip()

    async def run_eval(self) -> tuple[bool, str]:
        """Run the SWE-bench eval script and grade the result."""
        patch = await self.get_patch()
        if not patch:
            return False, "empty patch"

        apply_chain = " || ".join(
            f'{cmd} "{DOCKER_PATCH}"' for cmd in _GIT_APPLY_CMDS
        )

        eval_script = self._eval_script.replace(
            "conda activate testbed",
            f"conda activate {self._episode_env_name}",
        )

        patch_escaped = patch.replace("'", "'\\''")

        eval_cmd = " && ".join([
            f"cd {DOCKER_WORKDIR}",
            "git checkout -- . 2>/dev/null || true",
            "git clean -fd 2>/dev/null || true",
            f"printf '%s\\n' '{patch_escaped}' > {DOCKER_PATCH}",
            f"( {apply_chain} )",
            f"cat > /tmp/eval.sh << 'EOF_EVAL_SCRIPT'\n{eval_script}\nEOF_EVAL_SCRIPT",
            "bash /tmp/eval.sh 2>&1",
        ])

        eval_output, _ = await self._run_in_env(
            eval_cmd, timeout=_EVAL_TIMEOUT, sandbox=True,
        )

        prediction = {
            "instance_id": self._instance_id,
            "model_patch": patch,
            "model_name_or_path": "rl-agent",
        }

        try:
            result = grade_instance(
                test_spec=self._test_spec,
                prediction=prediction,
                test_output=eval_output,
            )
            return result.resolved, eval_output
        except Exception as e:
            logger.warning(f"[{self._instance_id}] Grading failed: {e}")
            return False, eval_output

    async def destroy(self) -> None:
        """Clean up the episode environment."""
        loop = asyncio.get_event_loop()

        # Remove /testbed
        testbed = Path(DOCKER_WORKDIR)
        if testbed.exists():
            try:
                await loop.run_in_executor(
                    None,
                    partial(shutil.rmtree, testbed, ignore_errors=True),
                )
            except Exception as e:
                logger.warning(f"Failed to remove {testbed}: {e}")

        # Remove cloned conda env
        if self._episode_env_name:
            try:
                await loop.run_in_executor(
                    None,
                    partial(_remove_conda_env, self._episode_env_name),
                )
            except Exception as e:
                logger.warning(
                    f"Failed to remove conda env {self._episode_env_name}: {e}"
                )
            finally:
                self._episode_env_name = None
                self._episode_env_dir = None


def _run_subprocess(cmd: list[str], timeout: int = 150):
    import subprocess

    return subprocess.run(
        cmd,
        stdout=PIPE, stderr=PIPE,
        text=True, timeout=timeout,
    )


def _remove_conda_env(env_name: str) -> None:
    import subprocess

    subprocess.run(
        [_CONDA_BIN, "remove", "-n", env_name, "--all", "-y", "-q"],
        capture_output=True, text=True, timeout=120,
    )
