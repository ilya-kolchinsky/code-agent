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
import json
import logging
from pathlib import Path
from urllib.request import urlopen

from evals.common.prompt_builder import build_prompted_dataset_main
from evals.odex.prompt import create_prompt_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_odex_dataset(name: str, split: str) -> list[dict]:
    """Load ODEX dataset from GitHub repository.

    Downloads en_test.jsonl directly from:
    https://github.com/zorazrw/odex/tree/master/data

    ODEX has duplicate task_ids (439 instances, 333 unique task_ids).
    We assign composite IDs: {task_id}_{index} for instances with the same task_id.

    Args:
        name: Dataset name (ignored, always uses zorazrw/odex)
        split: Dataset split (ignored, always uses test)

    Returns:
        List of ODEX instances with unique instance_id field.
    """
    # Always use en_test.jsonl from GitHub
    url = "https://raw.githubusercontent.com/zorazrw/odex/master/data/en_test.jsonl"

    logger.info(f"Downloading ODEX dataset from {url}")

    instances = []
    task_id_counts = {}

    with urlopen(url) as response:
        for line in response:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            instance = json.loads(line)

            # Track how many times we've seen this task_id
            task_id = instance.get("task_id")
            if task_id:
                count = task_id_counts.get(task_id, 0)
                # Assign composite instance_id: task_id_index
                instance["instance_id"] = f"{task_id}_{count}"
                task_id_counts[task_id] = count + 1

            instances.append(instance)

    logger.info(f"Loaded {len(instances)} instances from ODEX test set")
    unique_tasks = len(task_id_counts)
    logger.info(f"Dataset has {unique_tasks} unique task_ids, {len(instances)} total instances")
    return instances


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
