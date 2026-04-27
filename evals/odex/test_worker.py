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
            solutions: Map of task_id -> generated code solution.
            run_id: Unique run identifier for this batch.

        Returns:
            List of TaskResult with grading details.
        """
        results = []

        for instance in instances:
            task_id = instance.get("task_id", "")
            if not task_id:
                logger.warning("Instance missing task_id, skipping")
                continue

            solution = solutions.get(task_id, "")
            test_cases = instance.get("test_list", [])

            if not test_cases:
                logger.warning(f"Task {task_id} has no test cases, skipping")
                task_result = grade_task(
                    task_id=task_id,
                    solution=solution,
                    test_outputs=[],
                    error="No test cases available",
                )
                results.append(task_result)
                continue

            # Execute the solution against test cases
            exec_result: ExecutionResult = self.executor.execute_task(
                task_id=task_id,
                run_id=run_id,
                solution_code=solution,
                test_cases=test_cases,
            )

            # Grade the execution result
            task_result = grade_task(
                task_id=task_id,
                solution=solution,
                test_outputs=exec_result.outputs,
                error=exec_result.error,
            )

            results.append(task_result)

            status = "PASSED" if task_result.passed else "FAILED"
            logger.info(
                f"Task {task_id}: {status} "
                f"({task_result.passed_tests}/{task_result.total_tests} tests)"
            )

        return results
