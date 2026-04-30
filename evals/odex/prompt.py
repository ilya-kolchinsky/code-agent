"""ODEX prompt construction and code extraction."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def build_odex_prompt(instance: dict, include_tests: bool = False) -> str:
    """Build a prompt for ODEX code generation.

    ODEX is a code completion task - complete the function stub.

    Args:
        instance: ODEX dataset instance with 'prompt', 'intent', and 'test'.
        include_tests: Whether to include test cases in the prompt.

    Returns:
        Formatted prompt string.
    """
    # ODEX uses a completion format with a function stub
    function_stub = instance.get("prompt", "")
    intent = instance.get("intent", "")

    prompt_parts = [
        "Complete the following Python function to solve this task:",
        "",
        f"Task: {intent}",
        "",
        "```python",
        f"{function_stub}",
        "```",
        "",
        "Complete the function definition above by adding the return statement.",
        "Provide the COMPLETE function (including the def line) in your response.",
    ]

    if include_tests:
        test_assertions = instance.get("test", [])
        if test_assertions:
            prompt_parts.extend([
                "",
                "The function will be tested with these assertions:",
            ])
            for assertion in test_assertions[:3]:  # Show first 3 tests
                prompt_parts.append(f"  {assertion.strip()}")

    prompt_parts.extend([
        "",
        "Wrap your complete solution in ```python and ``` markers.",
    ])

    return "\n".join(prompt_parts)


def create_prompt_dataset(
    instances: list[dict],
    output_path: str | Path,
    include_tests: bool = False,
) -> Path:
    """Build a prompted dataset for ODEX.

    Args:
        instances: ODEX dataset instances (list of dicts).
        output_path: Path to write the prompted dataset JSONL.
        include_tests: Whether to include test cases in prompts.

    Returns:
        Path to the output JSONL file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Building ODEX prompted dataset: {len(instances)} instances, "
        f"include_tests={include_tests}"
    )

    with open(output_path, "w") as f:
        for instance in instances:
            # Use instance_id (composite ID) if available, fallback to task_id
            instance_id = instance.get("instance_id") or instance.get("task_id", "")
            task_id = instance.get("task_id", "")

            if not instance_id:
                logger.warning("Instance missing instance_id and task_id, skipping")
                continue

            prompt = build_odex_prompt(instance, include_tests=include_tests)

            entry = {
                "instance_id": instance_id,
                "task_id": task_id,  # Keep original task_id for reference
                "text_inputs": prompt,
            }
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Wrote prompted dataset to {output_path}")
    return output_path


def load_prompt_dataset(path: str | Path) -> dict[str, str]:
    """Load a prompted dataset into a map of instance_id -> prompt text.

    Args:
        path: Path to the JSONL file produced by create_prompt_dataset.

    Returns:
        Dict mapping instance_id to the prompt text (text_inputs field).
    """
    prompts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            instance_id = entry.get("instance_id")
            text_inputs = entry.get("text_inputs", "")
            if instance_id and text_inputs:
                prompts[instance_id] = text_inputs
    return prompts


def extract_code_from_response(response: str) -> str:
    """Extract Python code from a model response.

    Handles:
      - ```python ... ``` markdown code fences
      - ```...``` generic code fences
      - Raw Python code

    Args:
        response: Raw text from the LLM.

    Returns:
        The extracted Python code, or the full response if no code
        markers are found.
    """
    # Try to extract from ```python ... ``` fence
    python_fence_pattern = r"```python\s*\n(.*?)\n```"
    match = re.search(python_fence_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try to extract from generic ``` ... ``` fence
    generic_fence_pattern = r"```\s*\n(.*?)\n```"
    match = re.search(generic_fence_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return the full response if no fences found
    return response.strip()
