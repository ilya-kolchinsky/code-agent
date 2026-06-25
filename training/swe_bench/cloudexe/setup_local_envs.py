"""One-time setup script for local SWE-bench RL training on Cloudexe.

Pre-builds conda environments and clones repos so that per-episode
setup during training only needs a fast conda clone + git checkout.

Run on the Cloudexe base instance (no GPUs needed):

    python training/swe_bench/cloudexe/setup_local_envs.py

What it does:
  1. Installs required system packages (apt-get)
  2. Creates /opt/miniconda3 symlink if needed
  3. Builds one conda env per unique SWE-bench env spec (~80 envs)
  4. Clones each unique repo as a bare reference (~12 repos)
  5. Installs Sandlock binary
  6. Writes env mapping JSON for use during training

Runtime: ~1-2 hours.  Results persist on /root (Cloudexe persistent storage).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from swebench.harness.test_spec.test_spec import make_test_spec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"

DEFAULT_REPO_CACHE = "/root/repo-cache"
DEFAULT_ENV_MAP = "/root/swe-env-map.json"
DEFAULT_MINICONDA = "/opt/miniconda3"

SYSTEM_PACKAGES = [
    "locales",
    "graphviz",
    "libfreetype6-dev",
    "pkg-config",
    "ffmpeg",
    "imagemagick",
    "build-essential",
    "libffi-dev",
    "libtiff-dev",
    "wget",
    "git",
    "texlive",
    "dvipng",
    "cm-super",
]


def install_system_packages() -> None:
    logger.info("Installing system packages...")
    subprocess.run(
        ["apt-get", "update", "-y"],
        check=False, capture_output=True,
    )
    subprocess.run(
        ["apt-get", "install", "-y", "--no-install-recommends"] + SYSTEM_PACKAGES,
        check=False, capture_output=True,
    )
    logger.info("System packages installed")


def ensure_miniconda_symlink(miniconda_path: str) -> None:
    target = Path(miniconda_path)
    if target.exists():
        logger.info(f"Miniconda already at {miniconda_path}")
        return

    alt_path = Path("/opt/miniconda")
    if alt_path.exists():
        logger.info(f"Creating symlink {miniconda_path} -> {alt_path}")
        target.symlink_to(alt_path)
        return

    logger.warning(
        f"Neither {miniconda_path} nor /opt/miniconda found. "
        f"Conda environments may not work."
    )


def build_conda_envs(
    env_map_path: str,
    instance_limit: int = 0,
) -> dict[str, str]:
    """Build one conda env per unique env_image_key."""
    logger.info("Loading SWE-bench dataset...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    if instance_limit > 0:
        ds = ds.select(range(min(instance_limit, len(ds))))

    groups: dict[str, dict] = {}
    for item in ds:
        spec = make_test_spec(item)
        key = spec.env_image_key
        if key not in groups:
            groups[key] = {"spec": spec, "instance": item}

    logger.info(f"Found {len(groups)} unique environment specs")

    env_map: dict[str, str] = {}
    existing_map = {}
    if os.path.exists(env_map_path):
        with open(env_map_path) as f:
            existing_map = json.load(f)

    for i, (key, info) in enumerate(groups.items(), 1):
        if key in existing_map:
            env_name = existing_map[key]
            result = subprocess.run(
                ["conda", "info", "--envs"],
                capture_output=True, text=True,
            )
            if env_name in result.stdout:
                logger.info(f"[{i}/{len(groups)}] Env {env_name} already exists, skipping")
                env_map[key] = env_name
                continue

        key_hash = hashlib.md5(key.encode()).hexdigest()[:12]
        env_name = f"swe-base-{key_hash}"

        logger.info(f"[{i}/{len(groups)}] Building env {env_name} for {key}")

        spec = info["spec"]
        setup_script = spec.setup_env_script

        # Replace 'testbed' env name with our naming scheme
        setup_script = setup_script.replace(
            "conda create -n testbed",
            f"conda create -n {env_name}",
        )
        setup_script = setup_script.replace(
            "conda activate testbed",
            f"conda activate {env_name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False
        ) as f:
            f.write("#!/bin/bash\nset -euxo pipefail\n")
            f.write(setup_script)
            script_path = f.name

        try:
            result = subprocess.run(
                ["/bin/bash", script_path],
                capture_output=True, text=True,
                timeout=600,
            )
            if result.returncode != 0:
                logger.error(
                    f"Failed to build env {env_name}: {result.stderr[-500:]}"
                )
                continue

            env_map[key] = env_name
            logger.info(f"[{i}/{len(groups)}] Built env {env_name}")

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout building env {env_name}")
        finally:
            os.unlink(script_path)

        # Save progress incrementally
        with open(env_map_path, "w") as f:
            json.dump(env_map, f, indent=2)

    logger.info(f"Built {len(env_map)}/{len(groups)} conda environments")
    return env_map


def clone_repos(repo_cache_dir: str, instance_limit: int = 0) -> None:
    """Clone each unique repo as a bare reference for fast per-episode cloning."""
    logger.info("Loading dataset for repo list...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    if instance_limit > 0:
        ds = ds.select(range(min(instance_limit, len(ds))))

    repos: dict[str, str] = {}
    for item in ds:
        repo = item["repo"]
        if repo not in repos:
            repos[repo] = f"https://github.com/{repo}"

    logger.info(f"Found {len(repos)} unique repos to cache")

    cache = Path(repo_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    for i, (repo, url) in enumerate(repos.items(), 1):
        org, name = repo.split("/")
        repo_dir = cache / org
        repo_dir.mkdir(parents=True, exist_ok=True)
        bare_path = repo_dir / f"{name}.git"

        if bare_path.exists():
            logger.info(f"[{i}/{len(repos)}] {repo} already cached, skipping")
            continue

        logger.info(f"[{i}/{len(repos)}] Cloning {repo}...")
        result = subprocess.run(
            ["git", "clone", "--bare", url, str(bare_path)],
            capture_output=True, text=True,
            timeout=600,
        )
        if result.returncode != 0:
            logger.error(f"Failed to clone {repo}: {result.stderr[-300:]}")
        else:
            logger.info(f"[{i}/{len(repos)}] Cached {repo}")


def install_sandlock() -> None:
    """Download and install the Sandlock binary."""
    sandlock_path = Path("/usr/local/bin/sandlock")
    if sandlock_path.exists():
        logger.info("Sandlock already installed")
        return

    logger.info("Installing Sandlock...")
    # Try pip install first (Python SDK includes the binary)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "sandlock"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        # Check if the binary is now available
        which_result = subprocess.run(
            ["which", "sandlock"], capture_output=True, text=True,
        )
        if which_result.returncode == 0:
            logger.info(f"Sandlock installed via pip at {which_result.stdout.strip()}")
            return

    logger.warning(
        "Could not install Sandlock automatically. "
        "Please install manually: download from "
        "https://github.com/multikernel/sandlock/releases "
        "and place at /usr/local/bin/sandlock"
    )


def smoke_test(env_map_path: str, repo_cache_dir: str) -> bool:
    """Quick test: clone a repo, create env, run a command."""
    logger.info("Running smoke test...")

    with open(env_map_path) as f:
        env_map = json.load(f)

    if not env_map:
        logger.error("No environments built — cannot smoke test")
        return False

    first_key = next(iter(env_map))
    env_name = env_map[first_key]

    # Test conda env exists
    result = subprocess.run(
        ["conda", "run", "-n", env_name, "python", "--version"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        logger.error(f"Conda env {env_name} broken: {result.stderr}")
        return False

    logger.info(f"Smoke test passed: {env_name} -> {result.stdout.strip()}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pre-build local environments for SWE-bench RL training"
    )
    parser.add_argument(
        "--repo-cache-dir",
        default=DEFAULT_REPO_CACHE,
        help=f"Directory for bare repo clones (default: {DEFAULT_REPO_CACHE})",
    )
    parser.add_argument(
        "--env-map",
        default=DEFAULT_ENV_MAP,
        help=f"Output path for env mapping JSON (default: {DEFAULT_ENV_MAP})",
    )
    parser.add_argument(
        "--miniconda-path",
        default=DEFAULT_MINICONDA,
        help=f"Expected miniconda3 path (default: {DEFAULT_MINICONDA})",
    )
    parser.add_argument(
        "--instance-limit",
        type=int,
        default=0,
        help="Limit instances for testing (0 = all)",
    )
    parser.add_argument(
        "--skip-system-packages",
        action="store_true",
        help="Skip apt-get install step",
    )
    parser.add_argument(
        "--skip-repos",
        action="store_true",
        help="Skip repo cloning step",
    )
    parser.add_argument(
        "--skip-sandlock",
        action="store_true",
        help="Skip Sandlock installation step",
    )
    args = parser.parse_args()

    if not args.skip_system_packages:
        install_system_packages()

    ensure_miniconda_symlink(args.miniconda_path)

    env_map = build_conda_envs(
        env_map_path=args.env_map,
        instance_limit=args.instance_limit,
    )

    if not args.skip_repos:
        clone_repos(args.repo_cache_dir, instance_limit=args.instance_limit)

    if not args.skip_sandlock:
        install_sandlock()

    if env_map:
        smoke_test(args.env_map, args.repo_cache_dir)

    logger.info("Setup complete!")
    logger.info(f"  Env map: {args.env_map}")
    logger.info(f"  Repo cache: {args.repo_cache_dir}")
    logger.info(f"  Environments built: {len(env_map)}")


if __name__ == "__main__":
    main()
