"""Select a fixed set of evaluation instance IDs from a HuggingFace dataset.

Loads the dataset, shuffles deterministically, takes the first N instance IDs,
and writes them to a text file (one ID per line).

Usage::

    python -m evals.swe_bench.select_eval_instances \
        --output evals/swe_bench/eval_instance_ids.txt \
        --count 250

    python -m evals.swe_bench.select_eval_instances \
        --dataset SWE-Gym/SWE-Gym-Lite --split train \
        --count 100 --seed 42 \
        --output /tmp/swe_gym_eval_ids.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"


def _auto_split(dataset_name: str) -> str:
    if dataset_name == "princeton-nlp/SWE-bench_Verified":
        return "test"
    return "train"


def main():
    parser = argparse.ArgumentParser(
        description="Select evaluation instance IDs from a HuggingFace dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output text file (one instance ID per line)",
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
        "--count",
        type=int,
        default=250,
        help="Number of instances to select (default: 250)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )

    args = parser.parse_args()
    split = args.split if args.split is not None else _auto_split(args.dataset)

    ds = load_dataset(args.dataset, split=split)
    ds = ds.shuffle(seed=args.seed)
    n = min(args.count, len(ds))
    ids = [ds[i]["instance_id"] for i in range(n)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for iid in ids:
            f.write(iid + "\n")

    print(f"Wrote {n} instance IDs to {args.output} (dataset: {args.dataset}, split: {split}, seed: {args.seed})")


if __name__ == "__main__":
    main()
