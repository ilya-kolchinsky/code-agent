"""Grade ODEX evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evals.common.grader import BaseAggregateReport


@dataclass
class TaskResult:
    """Result of evaluating a single ODEX task."""

    task_id: str
    passed: bool
    total_tests: int
    passed_tests: int
    pass_rate: float
    solution_exists: bool
    error: str | None = None
    test_outputs: list[dict] | None = None


@dataclass
class AggregateReport(BaseAggregateReport):
    """Aggregate report across all evaluated ODEX tasks.

    Inherits common aggregation structure from BaseAggregateReport.
    Adds ODEX-specific metric: avg_test_pass_rate.
    """

    avg_test_pass_rate: float = 0.0

    # Aliases for base class fields (for backward compatibility)
    @property
    def total_tasks(self) -> int:
        return self.total

    @total_tasks.setter
    def total_tasks(self, value: int) -> None:
        self.total = value

    @property
    def passed_tasks(self) -> int:
        return self.passed

    @passed_tasks.setter
    def passed_tasks(self, value: int) -> None:
        self.passed = value

    @property
    def failed_tasks(self) -> int:
        return self.failed

    @failed_tasks.setter
    def failed_tasks(self, value: int) -> None:
        self.failed = value

    @property
    def error_tasks(self) -> int:
        return self.errors

    @error_tasks.setter
    def error_tasks(self, value: int) -> None:
        self.errors = value

    @property
    def empty_solution_tasks(self) -> int:
        return self.empty

    @empty_solution_tasks.setter
    def empty_solution_tasks(self, value: int) -> None:
        self.empty = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict with ODEX field names."""
        return {
            "total_tasks": self.total_tasks,
            "passed_tasks": self.passed_tasks,
            "failed_tasks": self.failed_tasks,
            "error_tasks": self.error_tasks,
            "empty_solution_tasks": self.empty_solution_tasks,
            "pass_rate": self.pass_rate,
            "avg_test_pass_rate": self.avg_test_pass_rate,
            "passed_ids": self.passed_ids,
            "failed_ids": self.failed_ids,
            "error_ids": self.error_ids,
        }


def grade_task(
    task_id: str,
    solution: str,
    test_outputs: list[dict],
    error: str | None = None,
) -> TaskResult:
    """Grade a single ODEX task from its execution results.

    Args:
        task_id: The task ID.
        solution: The generated code solution.
        test_outputs: List of test result dicts from executor.
        error: Optional error message from execution.

    Returns:
        TaskResult with grading details.
    """
    solution_exists = bool(solution and solution.strip())

    if error:
        return TaskResult(
            task_id=task_id,
            passed=False,
            total_tests=0,
            passed_tests=0,
            pass_rate=0.0,
            solution_exists=solution_exists,
            error=error,
            test_outputs=test_outputs,
        )

    if not test_outputs:
        return TaskResult(
            task_id=task_id,
            passed=False,
            total_tests=0,
            passed_tests=0,
            pass_rate=0.0,
            solution_exists=solution_exists,
            error="No test results available",
            test_outputs=[],
        )

    total_tests = len(test_outputs)
    passed_tests = sum(1 for t in test_outputs if t.get("passed", False))
    pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0

    # Task is considered passed if ALL tests pass
    passed = passed_tests == total_tests

    return TaskResult(
        task_id=task_id,
        passed=passed,
        total_tests=total_tests,
        passed_tests=passed_tests,
        pass_rate=pass_rate,
        solution_exists=solution_exists,
        test_outputs=test_outputs,
    )


def aggregate_reports(results: list[TaskResult]) -> AggregateReport:
    """Aggregate individual task results into a summary report.

    Args:
        results: List of TaskResult from grading individual tasks.

    Returns:
        AggregateReport with totals and pass rates.
    """
    report = AggregateReport(total=len(results))

    total_test_pass_rate = 0.0

    for result in results:
        total_test_pass_rate += result.pass_rate

        if result.error is not None:
            report.errors += 1
            report.failed += 1
            report.error_ids.append(result.task_id)
            report.failed_ids.append(result.task_id)
        elif not result.solution_exists:
            report.empty += 1
            report.failed += 1
            report.failed_ids.append(result.task_id)
        elif result.passed:
            report.passed += 1
            report.passed_ids.append(result.task_id)
        else:
            report.failed += 1
            report.failed_ids.append(result.task_id)

    # Compute derived metrics
    report.finalize()
    if report.total > 0:
        report.avg_test_pass_rate = total_test_pass_rate / report.total

    return report
