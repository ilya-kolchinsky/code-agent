"""Phase 1 Ray worker: generate code solutions via vLLM.

Uses the generic InferenceWorker with ODEX-specific code extraction.
"""

from __future__ import annotations

import logging

import ray

from evals.common.inference_worker import InferenceWorker
from .prompt import extract_code_from_response

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class CodeWorker(InferenceWorker):
    """Generates code solutions for ODEX tasks via vLLM.

    Inherits from InferenceWorker and customizes for ODEX:
    - Uses task_id as the instance identifier
    - Extracts code from markdown fences
    - Returns predictions with ODEX-specific schema
    """

    def generate_solutions(
        self,
        instances: list[dict],
        prompts: dict[str, str],
    ) -> list[dict]:
        """Generate code solutions for a batch of ODEX instances.

        Args:
            instances: List of ODEX dataset instances.
            prompts: Map of task_id -> pre-built prompt text.

        Returns:
            List of dicts with keys: task_id, solution,
            full_output, model_name_or_path, error.
        """
        results = self.generate_batch(
            instances=instances,
            prompts=prompts,
            extract_fn=extract_code_from_response,
            instance_id_key="task_id",
        )

        # Rename 'prediction' to 'solution' for ODEX schema
        for result in results:
            result["solution"] = result.pop("prediction")
            result["task_id"] = result.pop("instance_id")

        return results
