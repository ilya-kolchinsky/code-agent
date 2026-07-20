"""Build a training dataset for SWE-bench GRPO from HuggingFace.

Converts SWE-bench-format instances into the JSONL format that
OpenRLHF expects::

    {"input": "<chat_template_applied_prompt>", "label": "<instance_metadata_json>"}

The chat template is applied here with ``enable_thinking=False`` so that
OpenRLHF does not need to apply it again (do NOT pass
``--data.apply_chat_template`` to the training script).

Supports any HuggingFace dataset with the SWE-bench schema (instance_id,
problem_statement, test_patch, etc.).  Use ``--dataset`` and ``--split``
to select the source; default is SWE-bench Verified.

Usage:
    python build_dataset.py --output swe_bench_train.jsonl
    python build_dataset.py --output swe_bench_test.jsonl --instance-limit 16
    python build_dataset.py --output swe_bench_train.jsonl --model Qwen/Qwen3.5-9B
    python build_dataset.py --dataset SWE-Gym/SWE-Gym-Lite --output swe_gym_lite.jsonl
    python build_dataset.py --output swe_bench_250.jsonl --instance-limit 250 --shuffle --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"


def _auto_split(dataset_name: str) -> str:
    if dataset_name == "princeton-nlp/SWE-bench_Verified":
        return "test"
    return "train"


def build(
    output: Path,
    dataset_name: str = DEFAULT_DATASET,
    split: str | None = None,
    instance_limit: int = 0,
    shuffle: bool = False,
    seed: int = 42,
    model_name: str = DEFAULT_MODEL,
    enable_thinking: bool = False,
) -> None:
    resolved_split = split if split is not None else _auto_split(dataset_name)
    ds = load_dataset(dataset_name, split=resolved_split)

    if shuffle:
        ds = ds.shuffle(seed=seed)
    if instance_limit > 0:
        ds = ds.select(range(min(instance_limit, len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for row in ds:
            # Format as a chat message for the model
            messages = [{"role": "user", "content": row["problem_statement"]}]

            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

            # Include the full instance dict — make_test_spec() needs
            # test_patch, FAIL_TO_PASS, PASS_TO_PASS, etc.
            label = dict(row)
            entry = {
                "input": formatted,
                "label": json.dumps(label),
            }
            f.write(json.dumps(entry) + "\n")

    print(
        f"Wrote {len(ds)} instances to {output} "
        f"(dataset: {dataset_name}, split: {resolved_split}, "
        f"model: {model_name}, thinking: {'enabled' if enable_thinking else 'disabled'}"
        f"{', shuffled seed=' + str(seed) if shuffle else ''})"
    )


def main():
    parser = argparse.ArgumentParser(description="Build SWE-bench training dataset")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSONL path"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset identifier (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (default: 'test' for SWE-bench Verified, 'train' for others)",
    )
    parser.add_argument(
        "--instance-limit",
        type=int,
        default=0,
        help="Limit number of instances (0 = all)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Shuffle dataset before selecting instances",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model for chat template (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable model thinking/reasoning mode (default: disabled)",
    )
    args = parser.parse_args()
    build(
        args.output,
        dataset_name=args.dataset,
        split=args.split,
        instance_limit=args.instance_limit,
        shuffle=args.shuffle,
        seed=args.seed,
        model_name=args.model,
        enable_thinking=args.enable_thinking,
    )


if __name__ == "__main__":
    main()
