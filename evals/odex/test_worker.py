"""Phase 2 Ray worker: execute code solutions and run tests."""

from __future__ import annotations

import logging
from typing import Optional

import ray

from .executor import CodeExecutor, ExecutionResult
from .grader import grade_task, TaskResult

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class TestWorker:
    """Executes ODEX code solutions against test cases using K8s Jobs.

    Phase 2 worker that takes generated code and runs it against
    test cases in isolated K8s environments.
    """

    def __init__(
        self,
        k8s_namespace: Optional[str] = None,
        timeout: int = 300,
        service_account: Optional[str] = None,
        image: str = "python:3.11-slim",
    ):
        """Initialize the test worker.

        Args:
            k8s_namespace: K8s namespace. Auto-detected if not set.
            timeout: Per-task timeout in seconds (default 5 min).
            service_account: Optional K8s ServiceAccount.
            image: Python container image to use.
        """
        self.executor = CodeExecutor(
            k8s_namespace=k8s_namespace,
            timeout=timeout,
            service_account=service_account,
            image=image,
        )

    def execute_batch(
        self,
        instances: list[dict],
        solutions: dict[str, str],
        run_id: str,
    ) -> list[TaskResult]:
        """Execute a batch of code solutions against their test cases.

        Args:
            instances: List of ODEX dataset instances with test_list.
            solutions: Map of instance_id -> generated code solution.
            run_id: Unique run identifier for this batch.

        Returns:
            List of TaskResult with grading details.
        """
        results = []

        for instance in instances:
            # Get both task_id (original) and instance_id (composite)
            task_id = instance.get("task_id", "")
            instance_id = instance.get("instance_id", "")

            if not instance_id:
                logger.warning("Instance missing instance_id, skipping")
                continue

            # Look up solution using instance_id (composite key)
            solution = solutions.get(instance_id, "")

            # ODEX format: test assertions with entry_point
            test_assertions = instance.get("test", [])
            test_start = instance.get("test_start", "")
            entry_point = instance.get("entry_point", "")

            if not test_assertions:
                logger.warning(f"Instance {instance_id} has no test cases, skipping")
                task_result = grade_task(
                    task_id=instance_id,
                    solution=solution,
                    test_outputs=[],
                    error="No test cases available",
                )
                results.append(task_result)
                continue

            # Build ODEX test metadata
            test_metadata = {
                "test_start": test_start,
                "test_assertions": test_assertions,
                "entry_point": entry_point,
            }

            # Execute the solution against test cases
            # Use instance_id for K8s job naming
            exec_result: ExecutionResult = self.executor.execute_task(
                task_id=instance_id,
                run_id=run_id,
                solution_code=solution,
                test_metadata=test_metadata,
            )

            # Grade the execution result
            task_result = grade_task(
                task_id=instance_id,
                solution=solution,
                test_outputs=exec_result.outputs,
                error=exec_result.error,
            )

            results.append(task_result)

            status = "PASSED" if task_result.passed else "FAILED"
            logger.info(
                f"Instance {instance_id} (task {task_id}): {status} "
                f"({task_result.passed_tests}/{task_result.total_tests} tests)"
            )

        return results
