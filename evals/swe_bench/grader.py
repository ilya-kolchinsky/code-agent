"""Grade SWE-bench evaluation results.

Wraps swebench's grading logic to parse test output captured from pod logs,
grade individual instances, and aggregate results across a full evaluation run.
All operations are in-memory -- no PVC or file I/O required (except a temp file
needed internally by swebench's get_eval_report).

Adapted from https://github.com/MichaelClifford/swe-bench-on-kfp
"""

import inspect
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import TestSpec

_LOG_PATH_KWARG = (
    "log_path"
    if "log_path" in inspect.signature(get_eval_report).parameters
    else "test_log_path"
)

from evals.common.grader import BaseAggregateReport


@dataclass
class InstanceResult:
    """Result of evaluating a single SWE-bench instance."""

    instance_id: str
    resolved: bool
    patch_exists: bool
    patch_successfully_applied: bool
    error: str | None = None
    tests_status: dict[str, Any] | None = None


@dataclass
class AggregateReport(BaseAggregateReport):
    """Aggregate report across all evaluated SWE-bench instances."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict with SWE-bench field names."""
        return {
            "total_instances": self.total,
            "resolved_instances": self.passed,
            "unresolved_instances": self.failed,
            "error_instances": self.errors,
            "empty_patch_instances": self.empty,
            "resolve_rate": self.pass_rate,
            "resolved_ids": self.passed_ids,
            "unresolved_ids": self.failed_ids,
            "error_ids": self.error_ids,
        }


def grade_instance(
    test_spec: TestSpec,
    prediction: dict[str, str],
    test_output: str,
) -> InstanceResult:
    """Grade a single SWE-bench instance from its test output.

    Args:
        test_spec: The TestSpec for this instance.
        prediction: Dict with 'instance_id', 'model_name_or_path', and 'model_patch'.
        test_output: Raw test output text captured from pod logs.

    Returns:
        InstanceResult with grading details.
    """
    instance_id = prediction["instance_id"]
    temp_dir = None

    try:
        # get_eval_report's get_logs_eval() derives the repo from the log
        # path: Path(fp).parent.stem → repo.  Lowercase so the derived repo
        # matches the lowercase keys in MAP_REPO_TO_PARSER.
        temp_dir = Path(tempfile.mkdtemp()) / instance_id.lower()
        temp_dir.mkdir()
        temp_path = str(temp_dir / "test_output.txt")
        Path(temp_path).write_text(test_output)

        report = get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            include_tests_status=True,
            **{_LOG_PATH_KWARG: temp_path},
        )

        instance_report = report[instance_id]

        return InstanceResult(
            instance_id=instance_id,
            resolved=instance_report["resolved"],
            patch_exists=instance_report["patch_exists"],
            patch_successfully_applied=instance_report["patch_successfully_applied"],
            tests_status=instance_report.get("tests_status"),
        )

    except Exception as e:
        return InstanceResult(
            instance_id=instance_id,
            resolved=False,
            patch_exists=bool(prediction.get("model_patch")),
            patch_successfully_applied=False,
            error=str(e),
        )
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir.parent, ignore_errors=True)


def aggregate_reports(results: list[InstanceResult]) -> AggregateReport:
    """Aggregate individual instance results into a summary report.

    Args:
        results: List of InstanceResult from grading individual instances.

    Returns:
        AggregateReport with totals, resolve rate, and ID lists.
    """
    report = AggregateReport(total=len(results))

    for result in results:
        if result.error is not None:
            report.errors += 1
            report.failed += 1
            report.error_ids.append(result.instance_id)
            report.failed_ids.append(result.instance_id)
        elif not result.patch_exists:
            report.empty += 1
            report.failed += 1
            report.failed_ids.append(result.instance_id)
        elif result.resolved:
            report.passed += 1
            report.passed_ids.append(result.instance_id)
        else:
            report.failed += 1
            report.failed_ids.append(result.instance_id)

    report.finalize()

    return report
