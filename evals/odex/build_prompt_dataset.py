"""Build the ODEX prompted dataset.

One-time prep step that loads ODEX dataset from HuggingFace and constructs
prompts. The output is a JSONL file where each line has task_id and text_inputs
fields. This file is uploaded to S3/MinIO and reused by all future Phase 1
inference runs.

This script does NOT require Ray, vLLM, or any cluster infrastructure.
It can be run locally or as a standalone K8s Job.

Usage:
    # Run locally
    python -m evals.odex.build_prompt_dataset \
        --dataset neulab/odex \
        --output /tmp/prompted_dataset.jsonl \
        --s3-output s3://odex/prompts/prompted_dataset.jsonl

    # Or just build locally without S3
    python -m evals.odex.build_prompt_dataset \
        --dataset neulab/odex \
        --output /tmp/prompted_dataset.jsonl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import load_dataset

from evals.common.prompt_builder import build_prompted_dataset_main
from evals.odex.prompt import create_prompt_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_odex_dataset(name: str, split: str) -> list[dict]:
    """Load ODEX dataset from HuggingFace."""
    dataset = load_dataset(name, split=split)
    return list(dataset)


def build_prompts(instances: list[dict], output_path: Path, **kwargs) -> Path:
    """Build ODEX prompts."""
    return create_prompt_dataset(
        instances=instances,
        output_path=output_path,
        include_tests=kwargs.get("include_tests", False),
    )


def add_odex_args(parser: argparse.ArgumentParser) -> None:
    """Add ODEX-specific arguments."""
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test cases in prompts (default: False)",
    )


def main():
    build_prompted_dataset_main(
        dataset_loader=load_odex_dataset,
        prompt_builder=build_prompts,
        default_dataset="neulab/odex",
        description="Build ODEX prompted dataset (one-time prep step)",
        extra_args_fn=add_odex_args,
    )


if __name__ == "__main__":
    main()
