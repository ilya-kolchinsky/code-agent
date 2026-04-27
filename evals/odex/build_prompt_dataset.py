"""Build prompted dataset for ODEX evaluation.

Uses the common prompt builder scaffolding with ODEX-specific callbacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evals.common.prompt_builder import build_prompted_dataset_main
from .prompt import build_odex_prompt


def _load_odex_dataset(dataset_path: Path) -> list[dict]:
    """Load ODEX dataset from JSONL file.

    Args:
        dataset_path: Path to ODEX JSONL dataset.

    Returns:
        List of ODEX instances (dicts with task_id, intent, test_list).
    """
    import json

    instances = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instance = json.loads(line)
            instances.append(instance)

    return instances


def _build_odex_prompt_callback(instance: dict, args: argparse.Namespace) -> str:
    """Callback to build a prompt for an ODEX instance.

    Args:
        instance: ODEX dataset instance.
        args: CLI arguments (includes --include-tests flag).

    Returns:
        Formatted prompt string.
    """
    return build_odex_prompt(
        instance=instance,
        include_tests=getattr(args, "include_tests", False),
    )


def _add_extra_args(parser: argparse.ArgumentParser) -> None:
    """Add ODEX-specific CLI arguments.

    Args:
        parser: Argument parser to extend.
    """
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test cases in the prompt",
    )


if __name__ == "__main__":
    build_prompted_dataset_main(
        dataset_loader=_load_odex_dataset,
        prompt_builder=_build_odex_prompt_callback,
        extra_args_fn=_add_extra_args,
        instance_id_key="task_id",
    )
