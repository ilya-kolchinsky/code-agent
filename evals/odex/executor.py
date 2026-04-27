"""Code execution engine for ODEX evaluation.

Runs generated Python code against test cases in isolated environments.
Uses K8s Jobs for sandboxing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config

logger = logging.getLogger(__name__)

# How often to poll Job status (seconds)
_POLL_INTERVAL = 5


@dataclass
class ExecutionResult:
    """Result of executing code against test cases."""

    task_id: str
    succeeded: bool
    timed_out: bool
    outputs: list[dict]  # List of {input, expected, actual, passed}
    error: str | None = None


def _job_name(task_id: str, run_id: str) -> str:
    """Generate a unique K8s-safe Job name from task_id and run_id.

    K8s names must be lowercase, alphanumeric + hyphens, max 63 chars.
    """
    safe_id = str(task_id).lower().replace("_", "-").replace("/", "-")
    uid = hashlib.md5(f"{task_id}-{run_id}-{time.time()}".encode()).hexdigest()[:6]
    # Truncate safe_id first to preserve the uid suffix (uniqueness)
    max_base_len = 63 - len("odex--") - len(uid)
    safe_id = safe_id[:max_base_len].strip("-")
    return f"odex-{safe_id}-{uid}"


def _build_execution_script(solution_code: str, test_metadata: dict) -> str:
    """Build a Python script that executes the solution against ODEX test assertions.

    Args:
        solution_code: The generated Python code to test.
        test_metadata: Dict with 'test_start', 'test_assertions', 'entry_point'.

    Returns:
        Python script that runs the tests and outputs JSON results.
    """
    test_start = test_metadata.get("test_start", "")
    test_assertions = test_metadata.get("test_assertions", [])
    entry_point = test_metadata.get("entry_point", "")

    # Build test execution script for ODEX assertion-based tests
    script = f'''
import json
import sys
import traceback

# The generated solution
solution_code = """{solution_code}"""

# ODEX test metadata
test_start = """{test_start}"""
test_assertions = {json.dumps(test_assertions)}
entry_point = "{entry_point}"

results = []

try:
    # Execute the solution code to define the function
    exec_globals = {{}}
    exec(solution_code, exec_globals)

    # Get the entry point function
    if entry_point and entry_point in exec_globals:
        solution_func = exec_globals[entry_point]
    else:
        # Fallback: find any callable
        solution_func = None
        for name, obj in exec_globals.items():
            if callable(obj) and not name.startswith("_"):
                solution_func = obj
                break

    if solution_func is None:
        print(json.dumps({{"error": "No function found in solution code"}}))
        sys.exit(1)

    # Build and execute the check function for each assertion
    for i, assertion in enumerate(test_assertions):
        try:
            # Create check function with this assertion
            check_code = test_start + assertion
            check_globals = {{"candidate": solution_func}}
            exec(check_code, check_globals)

            # If we have a check function, call it
            if "check" in check_globals:
                check_func = check_globals["check"]
                check_func(solution_func)

            # If no exception was raised, test passed
            results.append({{
                "test_index": i,
                "assertion": assertion.strip(),
                "passed": True,
                "error": None,
            }})
        except AssertionError as e:
            results.append({{
                "test_index": i,
                "assertion": assertion.strip(),
                "passed": False,
                "error": f"AssertionError: {{str(e) or 'assertion failed'}}",
            }})
        except Exception as e:
            results.append({{
                "test_index": i,
                "assertion": assertion.strip(),
                "passed": False,
                "error": f"{{type(e).__name__}}: {{str(e)}}",
            }})

    print(json.dumps({{"results": results}}))

except Exception as e:
    print(json.dumps({{"error": f"Execution error: {{str(e)}}", "traceback": traceback.format_exc()}}))
    sys.exit(1)
'''
    return script


def _build_job_manifest(
    task_id: str,
    run_id: str,
    image: str,
    execution_script: str,
    namespace: str,
    timeout: int,
    service_account: Optional[str] = None,
) -> k8s_client.V1Job:
    """Build a K8s Job manifest for executing code tests.

    Args:
        task_id: ODEX task ID.
        run_id: Unique run identifier.
        image: Python container image.
        execution_script: Python script to execute.
        namespace: K8s namespace.
        timeout: Job timeout in seconds.
        service_account: Optional K8s ServiceAccount name.

    Returns:
        V1Job manifest.
    """
    job_name = _job_name(task_id, run_id)

    # Embed the script in the command
    command = [
        "/bin/bash",
        "-c",
        f"cat > /tmp/test_runner.py << 'EOF'\n{execution_script}\nEOF\n"
        f"python3 /tmp/test_runner.py"
    ]

    container = k8s_client.V1Container(
        name="executor",
        image=image,
        image_pull_policy="IfNotPresent",
        command=command,
        resources=k8s_client.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "1Gi"},
            limits={"cpu": "2", "memory": "2Gi"},
        ),
    )

    pod_spec = k8s_client.V1PodSpec(
        containers=[container],
        restart_policy="Never",
        service_account_name=service_account,
    )

    template = k8s_client.V1PodTemplateSpec(
        metadata=k8s_client.V1ObjectMeta(
            labels={
                "app": "odex-eval",
                "task-id": task_id[:63],
                "run-id": run_id[:63],
            },
        ),
        spec=pod_spec,
    )

    job_spec = k8s_client.V1JobSpec(
        template=template,
        backoff_limit=0,
        active_deadline_seconds=timeout,
        ttl_seconds_after_finished=300,
    )

    job = k8s_client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=k8s_client.V1ObjectMeta(
            name=job_name,
            namespace=namespace,
            labels={
                "app": "odex-eval",
                "task-id": task_id[:63],
                "run-id": run_id[:63],
            },
        ),
        spec=job_spec,
    )

    return job


def _detect_namespace() -> str:
    """Detect the current K8s namespace when running in-cluster."""
    ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    try:
        with open(ns_path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "default"


class CodeExecutor:
    """Manages K8s Jobs for executing ODEX code solutions."""

    def __init__(
        self,
        k8s_namespace: Optional[str] = None,
        timeout: int = 300,
        service_account: Optional[str] = None,
        image: str = "python:3.11-slim",
    ):
        """Initialize the executor.

        Args:
            k8s_namespace: K8s namespace. Auto-detected if not set.
            timeout: Per-task timeout in seconds (default 5 min).
            service_account: Optional K8s ServiceAccount.
            image: Python container image to use.
        """
        self.k8s_namespace = k8s_namespace or _detect_namespace()
        self.timeout = timeout
        self.service_account = service_account
        self.image = image

        # Load K8s config
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()

        self.batch_api = k8s_client.BatchV1Api()
        self.core_api = k8s_client.CoreV1Api()

    def create_job(
        self,
        task_id: str,
        run_id: str,
        solution_code: str,
        test_metadata: dict,
    ) -> str:
        """Create a K8s Job to execute code against test cases.

        Args:
            task_id: ODEX task ID.
            run_id: Unique run identifier.
            solution_code: The generated Python code.
            test_metadata: ODEX test metadata with assertions.

        Returns:
            The Job name.
        """
        execution_script = _build_execution_script(solution_code, test_metadata)

        job = _build_job_manifest(
            task_id=task_id,
            run_id=run_id,
            image=self.image,
            execution_script=execution_script,
            namespace=self.k8s_namespace,
            timeout=self.timeout,
            service_account=self.service_account,
        )

        self.batch_api.create_namespaced_job(
            namespace=self.k8s_namespace,
            body=job,
        )

        job_name = job.metadata.name
        logger.info(f"Created Job {job_name} for task {task_id}")
        return job_name

    def wait_for_job(self, job_name: str) -> tuple[bool, bool]:
        """Wait for a K8s Job to complete.

        Returns:
            Tuple of (succeeded: bool, timed_out: bool).
        """
        while True:
            job = self.batch_api.read_namespaced_job(
                name=job_name,
                namespace=self.k8s_namespace,
            )

            if job.status.succeeded and job.status.succeeded > 0:
                return True, False

            if job.status.failed and job.status.failed > 0:
                conditions = job.status.conditions or []
                timed_out = any(
                    c.type == "Failed" and c.reason == "DeadlineExceeded"
                    for c in conditions
                )
                return False, timed_out

            time.sleep(_POLL_INTERVAL)

    def get_pod_logs(self, job_name: str, retries: int = 5) -> str:
        """Get logs from the pod created by a Job.

        Returns:
            Pod log output as a string.
        """
        for attempt in range(retries):
            pods = self.core_api.list_namespaced_pod(
                namespace=self.k8s_namespace,
                label_selector=f"job-name={job_name}",
            )

            if not pods.items:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return ""

            pod_name = pods.items[0].metadata.name

            try:
                logs = self.core_api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.k8s_namespace,
                )
                if logs:
                    return logs
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return ""
            except k8s_client.ApiException as e:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                logger.error(f"Failed to read logs for pod {pod_name}: {e}")
                return ""

        return ""

    def delete_job(self, job_name: str) -> None:
        """Delete a Job and its pods."""
        try:
            self.batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.k8s_namespace,
                propagation_policy="Foreground",
            )
            logger.info(f"Deleted Job {job_name}")
        except k8s_client.ApiException as e:
            logger.warning(f"Failed to delete Job {job_name}: {e}")

    def execute_task(
        self,
        task_id: str,
        run_id: str,
        solution_code: str,
        test_metadata: dict,
    ) -> ExecutionResult:
        """Execute a solution against test cases.

        Creates a K8s Job, waits for completion, parses results.

        Args:
            task_id: ODEX task ID.
            run_id: Unique run identifier.
            solution_code: The generated Python code.
            test_metadata: ODEX test metadata with assertions.

        Returns:
            ExecutionResult with test outcomes.
        """
        job_name = None
        try:
            job_name = self.create_job(task_id, run_id, solution_code, test_metadata)
            succeeded, timed_out = self.wait_for_job(job_name)
            logs = self.get_pod_logs(job_name)

            if timed_out:
                return ExecutionResult(
                    task_id=task_id,
                    succeeded=False,
                    timed_out=True,
                    outputs=[],
                    error="Execution timed out",
                )

            if not logs:
                return ExecutionResult(
                    task_id=task_id,
                    succeeded=False,
                    timed_out=False,
                    outputs=[],
                    error="No output from execution",
                )

            # Parse JSON results from logs
            try:
                result_data = json.loads(logs.strip())
                if "error" in result_data:
                    return ExecutionResult(
                        task_id=task_id,
                        succeeded=False,
                        timed_out=False,
                        outputs=[],
                        error=result_data["error"],
                    )

                test_results = result_data.get("results", [])
                return ExecutionResult(
                    task_id=task_id,
                    succeeded=succeeded,
                    timed_out=False,
                    outputs=test_results,
                )

            except json.JSONDecodeError as e:
                return ExecutionResult(
                    task_id=task_id,
                    succeeded=False,
                    timed_out=False,
                    outputs=[],
                    error=f"Failed to parse execution output: {e}",
                )

        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            return ExecutionResult(
                task_id=task_id,
                succeeded=False,
                timed_out=False,
                outputs=[],
                error=str(e),
            )

        finally:
            if job_name:
                self.delete_job(job_name)
