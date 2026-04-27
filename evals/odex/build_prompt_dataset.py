"""Build prompted dataset for ODEX evaluation.

Uses the common prompt builder scaffolding with ODEX-specific callbacks.

Usage:
    python -m evals.odex.build_prompt_dataset \
        --dataset-path /path/to/odex.jsonl \
        --output /tmp/prompted_dataset.jsonl \
        --s3-upload-uri s3://odex/prompts/prompted_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from evals.common.s3_storage import upload_file
from .prompt import create_prompt_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build ODEX prompted dataset (one-time prep step)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to ODEX dataset JSONL file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Local path to write the prompted dataset JSONL",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test cases in the prompt",
    )
    parser.add_argument(
        "--instance-limit",
        type=int,
        default=0,
        help="Max instances to process (0 = no limit)",
    )
    parser.add_argument(
        "--s3-upload-uri",
        type=str,
        default=None,
        help="S3 URI to upload the prompted dataset",
    )

    args = parser.parse_args()

    # Load ODEX dataset
    logger.info(f"Loading dataset from {args.dataset_path}")
    instances = []
    with open(args.dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instance = json.loads(line)
            instances.append(instance)

    logger.info(f"Loaded {len(instances)} instances")

    # Apply instance limit
    if args.instance_limit > 0:
        instances = instances[:args.instance_limit]
        logger.info(f"Limited to {len(instances)} instances")

    # Build prompted dataset
    output_path = create_prompt_dataset(
        instances=instances,
        output_path=Path(args.output_path),
        include_tests=args.include_tests,
    )

    logger.info(f"Prompted dataset written to {output_path}")

    # Upload to S3 if requested
    if args.s3_upload_uri:
        upload_file(str(output_path), args.s3_upload_uri)
        logger.info(f"Uploaded to {args.s3_upload_uri}")


if __name__ == "__main__":
    main()
