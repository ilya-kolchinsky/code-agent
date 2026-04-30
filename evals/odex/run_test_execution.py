"""Phase 2: Execute code solutions and grade results.

Uses Ray + K8s Jobs to test generated code against ODEX test cases.
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import ray

from evals.common.s3_storage import download_file, upload_file
from .grader import aggregate_reports, TaskResult
from .test_worker import TestWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: Path) -> list[dict]:
    """Load ODEX dataset instances from JSONL.

    ODEX has duplicate task_ids (439 instances, 333 unique task_ids).
    We assign composite IDs: {task_id}_{index} for instances with the same task_id.

    Args:
        dataset_path: Path to the ODEX dataset file.

    Returns:
        List of ODEX dataset instances with test_list and instance_id.
    """
    instances = []
    task_id_counts = {}

    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
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

    logger.info(f"Loaded {len(instances)} instances from {dataset_path}")
    unique_tasks = len(task_id_counts)
    logger.info(f"Dataset has {unique_tasks} unique task_ids, {len(instances)} total instances")
    return instances


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


def load_solutions(solutions_path: Path) -> dict[str, str]:
    """Load generated code solutions from JSONL.

    Args:
        solutions_path: Path to Phase 1 output (code solutions).

    Returns:
        Dict mapping instance_id -> solution code.
    """
    solutions = {}
    with open(solutions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            instance_id = entry.get("instance_id")
            solution = entry.get("solution", "")
            if instance_id:
                solutions[instance_id] = solution

    logger.info(f"Loaded {len(solutions)} solutions from {solutions_path}")
    return solutions


def distribute_tests(
    instances: list[dict],
    solutions: dict[str, str],
    num_workers: int,
    run_id: str,
    k8s_namespace: Optional[str] = None,
    timeout: int = 300,
    service_account: Optional[str] = None,
    image: str = "python:3.11-slim",
) -> list[TaskResult]:
    """Distribute test execution across Ray workers.

    Args:
        instances: ODEX dataset instances with test_list.
        solutions: Map of task_id -> generated code.
        num_workers: Number of Ray workers to spawn.
        run_id: Unique run identifier.
        k8s_namespace: K8s namespace for Jobs.
        timeout: Per-task timeout in seconds.
        service_account: Optional K8s ServiceAccount.
        image: Python container image.

    Returns:
        List of TaskResult from grading.
    """
    # Create Ray workers
    workers = [
        TestWorker.remote(
            k8s_namespace=k8s_namespace,
            timeout=timeout,
            service_account=service_account,
            image=image,
        )
        for _ in range(num_workers)
    ]

    logger.info(f"Created {num_workers} TestWorker actors")

    # Distribute instances across workers
    # If fewer instances than workers, some workers will be idle
    chunk_size = max(1, len(instances) // num_workers)
    chunks = []
    for i in range(0, len(instances), chunk_size):
        chunks.append(instances[i : i + chunk_size])

    logger.info(f"Split {len(instances)} instances into {len(chunks)} chunks")

    # Submit tasks to Ray workers
    futures = [
        worker.execute_batch.remote(chunk, solutions, run_id)
        for worker, chunk in zip(workers, chunks)
        if chunk  # Skip empty chunks
    ]

    logger.info(f"Submitted {len(futures)} test execution tasks to Ray workers")

    # Gather results
    all_results = []
    for future in futures:
        batch_results = ray.get(future)
        all_results.extend(batch_results)

    logger.info(f"Collected {len(all_results)} task results")
    return all_results


def save_task_results(results: list[TaskResult], output_path: Path) -> None:
    """Save task-level results to JSONL.

    Args:
        results: List of TaskResult from grading.
        output_path: Path to write task results JSONL.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for result in results:
            # Convert TaskResult to dict
            entry = {
                "task_id": result.task_id,
                "passed": result.passed,
                "total_tests": result.total_tests,
                "passed_tests": result.passed_tests,
                "pass_rate": result.pass_rate,
                "solution_exists": result.solution_exists,
                "error": result.error,
                "test_outputs": result.test_outputs,
            }
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Saved {len(results)} task results to {output_path}")


def save_aggregate_report(
    results: list[TaskResult],
    output_path: Path,
) -> None:
    """Save aggregate report to JSON.

    Args:
        results: List of TaskResult from grading.
        output_path: Path to write aggregate report JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = aggregate_reports(results)

    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    logger.info(f"Saved aggregate report to {output_path}")
    logger.info(
        f"Results: {report.passed_tasks}/{report.total_tasks} passed "
        f"({report.pass_rate:.2%}), "
        f"avg test pass rate: {report.avg_test_pass_rate:.2%}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Execute ODEX code solutions and grade results"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path or S3 URI to ODEX dataset JSONL",
    )
    parser.add_argument(
        "--solutions-path",
        type=str,
        help="Path to Phase 1 solutions JSONL (local file)",
    )
    parser.add_argument(
        "--solutions-s3-uri",
        type=str,
        help="S3 URI to download solutions from (alternative to --solutions-path)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of Ray workers (default: 4)",
    )
    parser.add_argument(
        "--k8s-namespace",
        type=str,
        help="K8s namespace for test execution Jobs (auto-detected if omitted)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-task timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--service-account",
        type=str,
        help="K8s ServiceAccount for test Jobs",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="image-registry.openshift-image-registry.svc:5000/code-agent/odex-executor:latest",
        help="Python container image (default: odex-executor with numpy/pandas/requests/mock)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write results",
    )
    parser.add_argument(
        "--s3-upload-uri",
        type=str,
        help="Optional S3 URI prefix to upload results",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Unique run identifier (auto-generated if omitted)",
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

    # Validate solutions input
    if not args.solutions_path and not args.solutions_s3_uri:
        parser.error("Must provide either --solutions-path or --solutions-s3-uri")

    # Generate run_id if not provided
    run_id = args.run_id or str(uuid.uuid4())[:8]
    logger.info(f"Run ID: {run_id}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Ray
    ray.init(address=args.ray_address, ignore_reinit_error=True)
    logger.info(f"Ray initialized: {ray.cluster_resources()}")

    # Resolve dataset path (download from S3 if needed)
    dataset_path = _resolve_path(args.dataset_path, output_dir, "odex_dataset.jsonl")

    # Load dataset
    instances = load_dataset(dataset_path)
    logger.info(f"Loaded {len(instances)} instances")

    # Apply instance limit
    if args.instance_limit > 0:
        instances = instances[:args.instance_limit]
        logger.info(f"Limited to {len(instances)} instances")

    # Resolve solutions path (download from S3 if needed)
    if args.solutions_s3_uri:
        solutions_path = _resolve_path(args.solutions_s3_uri, output_dir, "solutions.jsonl")
    else:
        solutions_path = Path(args.solutions_path)

    solutions = load_solutions(solutions_path)

    # Execute tests and grade
    task_results = distribute_tests(
        instances=instances,
        solutions=solutions,
        num_workers=args.num_workers,
        run_id=run_id,
        k8s_namespace=args.k8s_namespace,
        timeout=args.timeout,
        service_account=args.service_account,
        image=args.image,
    )

    # Save results
    task_results_path = output_dir / "task_results.jsonl"
    aggregate_report_path = output_dir / "aggregate_report.json"

    save_task_results(task_results, task_results_path)
    save_aggregate_report(task_results, aggregate_report_path)

    # Optionally upload to S3
    if args.s3_upload_uri:
        s3_prefix = args.s3_upload_uri.rstrip("/")
        upload_file(str(task_results_path), f"{s3_prefix}/task_results.jsonl")
        upload_file(
            str(aggregate_report_path), f"{s3_prefix}/aggregate_report.json"
        )
        logger.info(f"Uploaded results to {s3_prefix}")

    logger.info("Phase 2 complete!")


if __name__ == "__main__":
    main()
