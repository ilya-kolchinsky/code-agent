"""Phase 1: Generate code solutions using Ray + vLLM.

Distributes ODEX instances across Ray workers that call vLLM endpoints
to generate Python code solutions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import ray

from evals.common.s3_storage import download_file, upload_file
from .code_worker import CodeWorker
from .prompt import load_prompt_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: Path) -> list[dict]:
    """Load ODEX dataset instances from JSONL.

    Args:
        dataset_path: Path to the ODEX dataset file.

    Returns:
        List of ODEX dataset instances.
    """
    instances = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instance = json.loads(line)
            instances.append(instance)

    logger.info(f"Loaded {len(instances)} instances from {dataset_path}")
    return instances


def load_existing_solutions(output_path: Path) -> set[str]:
    """Find task IDs that already have solutions on disk (resumability)."""
    completed = set()
    if not output_path.exists():
        return completed

    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                task_id = entry.get("task_id")
                if task_id:
                    completed.add(task_id)
            except json.JSONDecodeError:
                pass

    return completed


def _resolve_path(source: str, output_dir: Path, filename: str) -> Path:
    """Resolve a path that could be local or S3 URI.

    If source is an S3 URI, download to output_dir.
    Otherwise treat as a local path.
    """
    if source.startswith("s3://"):
        local_path = output_dir / filename
        logger.info(f"Downloading {source} to {local_path}")
        download_file(source, str(local_path))
        return local_path
    return Path(source)


def distribute_tasks(
    instances: list[dict],
    prompts: dict[str, str],
    vllm_endpoints: list[str],
    model_name_or_path: str,
    num_workers: int,
    timeout: int = 300,
) -> list[dict]:
    """Distribute ODEX instances across Ray workers for code generation.

    Args:
        instances: ODEX dataset instances.
        prompts: Map of task_id -> prompt text.
        vllm_endpoints: List of vLLM OpenAI-compatible endpoints.
        model_name_or_path: Model identifier.
        num_workers: Number of Ray workers to spawn.
        timeout: Request timeout in seconds.

    Returns:
        List of dicts with task_id, solution, full_output, error.
    """
    # Create Ray workers
    workers = [
        CodeWorker.remote(
            vllm_endpoints=vllm_endpoints,
            model_name_or_path=model_name_or_path,
            timeout=timeout,
        )
        for _ in range(num_workers)
    ]

    logger.info(f"Created {num_workers} CodeWorker actors")

    # Distribute instances across workers
    chunk_size = max(1, len(instances) // num_workers)
    chunks = [
        instances[i : i + chunk_size]
        for i in range(0, len(instances), chunk_size)
    ]

    # Round-robin remaining instances to workers if needed
    for i, worker_idx in enumerate(range(len(chunks), len(workers))):
        if i < len(instances):
            chunks.append([instances[len(chunks) * chunk_size + i]])

    logger.info(f"Split {len(instances)} instances into {len(chunks)} chunks")

    # Submit tasks to Ray workers
    futures = [
        worker.generate_solutions.remote(chunk, prompts)
        for worker, chunk in zip(workers, chunks)
        if chunk  # Skip empty chunks
    ]

    logger.info(f"Submitted {len(futures)} tasks to Ray workers")

    # Gather results
    all_results = []
    for future in futures:
        batch_results = ray.get(future)
        all_results.extend(batch_results)

    logger.info(f"Collected {len(all_results)} results")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Generate ODEX code solutions using Ray + vLLM"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path or S3 URI to ODEX dataset JSONL",
    )
    parser.add_argument(
        "--prompted-dataset-path",
        type=str,
        required=True,
        help="Path or S3 URI to prompted dataset JSONL (from build_prompt_dataset.py)",
    )
    parser.add_argument(
        "--vllm-endpoints",
        type=str,
        required=True,
        help="Comma-separated list of vLLM OpenAI-compatible endpoints",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Model identifier (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of Ray workers (default: 4)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to write code generation results JSONL",
    )
    parser.add_argument(
        "--s3-upload-uri",
        type=str,
        help="Optional S3 URI to upload results (e.g., s3://bucket/prefix/solutions.jsonl)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="vLLM request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--instance-limit",
        type=int,
        default=0,
        help="Max instances to evaluate (0 = no limit)",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default="auto",
        help="Ray cluster address (default: auto)",
    )

    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize Ray
    ray.init(address=args.ray_address, ignore_reinit_error=True)
    logger.info(f"Ray initialized: {ray.cluster_resources()}")

    # Resolve dataset and prompts paths (download from S3 if needed)
    dataset_path = _resolve_path(args.dataset_path, output_path.parent, "odex_dataset.jsonl")
    prompts_path = _resolve_path(args.prompted_dataset_path, output_path.parent, "prompted_dataset.jsonl")

    # Load dataset and prompts
    instances = load_dataset(dataset_path)
    logger.info(f"Loaded {len(instances)} instances")

    # Apply instance limit
    if args.instance_limit > 0:
        instances = instances[:args.instance_limit]
        logger.info(f"Limited to {len(instances)} instances")

    # Filter out already-completed instances (resumability)
    completed = load_existing_solutions(output_path)
    if completed:
        logger.info(f"Skipping {len(completed)} already-completed instances")
    pending = [inst for inst in instances if inst.get("task_id") not in completed]
    logger.info(f"Generating solutions for {len(pending)} instances")

    if not pending:
        logger.info("All instances already completed")
        if args.s3_upload_uri:
            upload_file(str(output_path), args.s3_upload_uri)
            logger.info(f"Uploaded existing results to {args.s3_upload_uri}")
        return

    # Load prompts
    prompts = load_prompt_dataset(prompts_path)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Check that we have prompts for all pending instances
    missing = [inst["task_id"] for inst in pending if inst.get("task_id") not in prompts]
    if missing:
        logger.warning(
            f"{len(missing)} instances have no prompt in the dataset. "
            f"First missing: {missing[:5]}"
        )

    # Parse vLLM endpoints
    vllm_endpoints = [ep.strip() for ep in args.vllm_endpoints.split(",")]
    logger.info(f"Using vLLM endpoints: {vllm_endpoints}")

    # Generate code solutions
    results = distribute_tasks(
        instances=pending,
        prompts=prompts,
        vllm_endpoints=vllm_endpoints,
        model_name_or_path=args.model_name_or_path,
        num_workers=args.num_workers,
        timeout=args.timeout,
    )

    # Append new results to output file
    with open(output_path, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info(f"Appended {len(results)} new results to {output_path}")

    # Optionally upload to S3
    if args.s3_upload_uri:
        upload_file(str(output_path), args.s3_upload_uri)
        logger.info(f"Uploaded results to {args.s3_upload_uri}")

    # Summary stats
    total = len(results)
    with_solution = sum(
        1 for r in results if r.get("solution") and r["solution"].strip()
    )
    with_error = sum(1 for r in results if r.get("error"))

    logger.info(
        f"Phase 1 complete: {total} new solutions generated, "
        f"{with_solution} with solutions, {with_error} errors"
    )


if __name__ == "__main__":
    main()
