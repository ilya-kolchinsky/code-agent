"""Analyze eval_agent.py output and produce a statistical report.

Reads JSONL output from eval_agent.py and reports on:
- Resolve rates (overall, per-repo)
- Token usage distributions (mean, median, std, percentiles)
- Step count distributions
- Episode outcome breakdown (submit, token_limit, max_steps, error)
- Wall-clock time distributions
- Per-repo breakdown of all metrics

Usage::

    python -m evals.swe_bench.analyze_eval_results \
        --input /tmp/baseline_swe_bench_250.jsonl \
        --label "SWE-Bench Verified 1x Baseline"

    python -m evals.swe_bench.analyze_eval_results \
        --input /tmp/baseline_train_4x.jsonl \
        --label "Training Candidates 4x" \
        --grpo-analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunStats:
    resolved: list[bool] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    wall_clock: list[float] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _stats_line(values: list[float]) -> str:
    if not values:
        return "n=0"
    n = len(values)
    mean = sum(values) / n
    median = _percentile(values, 50)
    variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0
    std = math.sqrt(variance)
    p10 = _percentile(values, 10)
    p25 = _percentile(values, 25)
    p75 = _percentile(values, 75)
    p90 = _percentile(values, 90)
    return (
        f"n={n}  mean={mean:.1f}  median={median:.1f}  std={std:.1f}  "
        f"p10={p10:.1f}  p25={p25:.1f}  p75={p75:.1f}  p90={p90:.1f}  "
        f"min={min(values):.1f}  max={max(values):.1f}"
    )


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _record_run(stats: RunStats, run: dict) -> None:
    stats.resolved.append(run["resolved"])
    stats.steps.append(run["steps"])
    stats.tokens.append(run["total_tokens"])
    stats.wall_clock.append(run["wall_clock_seconds"])
    stats.reasons[run.get("reason", "unknown")] += 1


def collect_stats(results: list[dict]) -> RunStats:
    stats = RunStats()
    for r in results:
        for run in r["runs"]:
            _record_run(stats, run)
    return stats


def collect_per_repo(results: list[dict]) -> dict[str, RunStats]:
    repos: dict[str, RunStats] = defaultdict(RunStats)
    for r in results:
        repo = r.get("repo", "unknown")
        for run in r["runs"]:
            _record_run(repos[repo], run)
    return dict(repos)


def grpo_analysis(results: list[dict]) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("GRPO Signal Analysis (K-run variance)")
    lines.append("=" * 70)

    k_values = [len(r["runs"]) for r in results]
    if not k_values or max(k_values) <= 1:
        lines.append("Single-run data — GRPO analysis requires K>1 runs per instance.")
        return "\n".join(lines)

    k = max(k_values)
    lines.append(f"Runs per instance (K): {k}")
    lines.append("")

    always_fail = 0
    always_pass = 0
    mixed = 0
    mixed_details: dict[str, int] = defaultdict(int)

    for r in results:
        runs = r["runs"]
        n_resolved = sum(1 for run in runs if run["resolved"])
        n_total = len(runs)

        if n_resolved == 0:
            always_fail += 1
        elif n_resolved == n_total:
            always_pass += 1
        else:
            mixed += 1
            mixed_details[f"{n_resolved}/{n_total}"] += 1

    total = len(results)
    lines.append(f"Always fail (0/{k}):    {always_fail:4d}  ({always_fail/total:.1%})")
    lines.append(f"Always pass ({k}/{k}):    {always_pass:4d}  ({always_pass/total:.1%})")
    lines.append(f"Mixed outcome:         {mixed:4d}  ({mixed/total:.1%})")
    lines.append("")

    if mixed_details:
        lines.append("Mixed outcome breakdown:")
        for key in sorted(mixed_details.keys(), key=lambda k: int(k.split("/")[0])):
            lines.append(f"  {key}: {mixed_details[key]} instances")
        lines.append("")

    lines.append(f"Useful gradient signal: {mixed/total:.1%} of instances")
    lines.append(f"Wasted compute (0/{k}): {always_fail/total:.1%} of instances")
    if always_pass > 0:
        lines.append(
            f"Already solved ({k}/{k}): {always_pass/total:.1%} — "
            "these can be removed from training set"
        )

    if mixed > 0:
        mixed_instances = [
            r for r in results
            if 0 < sum(1 for run in r["runs"] if run["resolved"]) < len(r["runs"])
        ]
        mixed_tokens = [
            run["total_tokens"]
            for r in mixed_instances
            for run in r["runs"]
        ]
        mixed_steps = [
            run["steps"]
            for r in mixed_instances
            for run in r["runs"]
        ]
        lines.append("")
        lines.append("Metrics for mixed-outcome instances only:")
        lines.append(f"  Tokens: {_stats_line([float(t) for t in mixed_tokens])}")
        lines.append(f"  Steps:  {_stats_line([float(s) for s in mixed_steps])}")

    return "\n".join(lines)


def format_report(
    results: list[dict],
    label: str,
    do_grpo: bool = False,
) -> str:
    lines = []
    overall = collect_stats(results)
    per_repo = collect_per_repo(results)

    total_instances = len(results)
    total_runs = len(overall.resolved)
    resolved_runs = sum(1 for r in overall.resolved if r)
    resolved_instances = sum(1 for r in results if r["resolve_rate"] > 0)

    lines.append("=" * 70)
    lines.append(f"Evaluation Report: {label}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Instances:           {total_instances}")
    lines.append(f"Runs per instance:   {total_runs // total_instances if total_instances else 0}")
    lines.append(f"Total runs:          {total_runs}")
    lines.append("")

    lines.append("--- Resolve Rates ---")
    lines.append(
        f"Resolved (any run):  {resolved_instances}/{total_instances} "
        f"({resolved_instances/total_instances:.1%})"
    )
    lines.append(
        f"Resolved (per run):  {resolved_runs}/{total_runs} "
        f"({resolved_runs/total_runs:.1%})"
    )
    lines.append("")

    lines.append("--- Episode Outcomes ---")
    for reason, count in sorted(overall.reasons.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason:15s}  {count:4d}  ({count/total_runs:.1%})")
    lines.append("")

    non_error_runs = [
        run for r in results for run in r["runs"]
        if run.get("reason") != "error"
    ]
    non_error_tokens = [run["total_tokens"] for run in non_error_runs]
    non_error_steps = [run["steps"] for run in non_error_runs]
    non_error_wall = [run["wall_clock_seconds"] for run in non_error_runs]

    lines.append("--- Token Usage (excluding error runs) ---")
    lines.append(f"  {_stats_line([float(t) for t in non_error_tokens])}")
    lines.append("")

    lines.append("--- Step Count (excluding error runs) ---")
    lines.append(f"  {_stats_line([float(s) for s in non_error_steps])}")
    lines.append("")

    lines.append("--- Wall Clock Seconds (excluding error runs) ---")
    lines.append(f"  {_stats_line([float(w) for w in non_error_wall])}")
    lines.append("")

    resolved_tokens = [
        run["total_tokens"]
        for r in results for run in r["runs"] if run["resolved"]
    ]
    resolved_steps = [
        run["steps"]
        for r in results for run in r["runs"] if run["resolved"]
    ]
    if resolved_tokens:
        lines.append("--- Resolved Episodes Only ---")
        lines.append(f"  Tokens: {_stats_line([float(t) for t in resolved_tokens])}")
        lines.append(f"  Steps:  {_stats_line([float(s) for s in resolved_steps])}")
        lines.append("")

    unresolved_tokens = [
        run["total_tokens"]
        for r in results for run in r["runs"]
        if not run["resolved"] and run.get("reason") != "error"
    ]
    unresolved_steps = [
        run["steps"]
        for r in results for run in r["runs"]
        if not run["resolved"] and run.get("reason") != "error"
    ]
    if unresolved_tokens:
        lines.append("--- Unresolved Episodes Only (excluding errors) ---")
        lines.append(f"  Tokens: {_stats_line([float(t) for t in unresolved_tokens])}")
        lines.append(f"  Steps:  {_stats_line([float(s) for s in unresolved_steps])}")
        lines.append("")

    lines.append("--- Per-Repo Breakdown ---")
    lines.append(
        f"{'Repo':40s} {'Instances':>9s} {'Resolved':>8s} {'Rate':>6s} "
        f"{'AvgSteps':>8s} {'AvgTokens':>9s}"
    )
    lines.append("-" * 85)
    repo_instance_counts = defaultdict(int)
    for r in results:
        repo_instance_counts[r.get("repo", "unknown")] += 1

    for repo in sorted(per_repo.keys()):
        s = per_repo[repo]
        n_instances = repo_instance_counts[repo]
        n_resolved = sum(1 for r in s.resolved if r)
        n_runs = len(s.resolved)
        avg_steps = sum(s.steps) / n_runs if n_runs else 0
        avg_tokens = sum(s.tokens) / n_runs if n_runs else 0
        rate = n_resolved / n_runs if n_runs else 0
        lines.append(
            f"{repo:40s} {n_instances:9d} {n_resolved:8d} {rate:6.1%} "
            f"{avg_steps:8.1f} {avg_tokens:9.0f}"
        )
    lines.append("")

    if do_grpo:
        lines.append(grpo_analysis(results))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze eval_agent.py results and produce statistical report",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file (output of eval_agent.py)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="Evaluation Results",
        help="Label for the report header",
    )
    parser.add_argument(
        "--grpo-analysis",
        action="store_true",
        help="Include GRPO signal analysis (requires K>1 runs per instance)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    results = load_results(args.input)
    if not results:
        print("Error: no results found in input file", file=sys.stderr)
        sys.exit(1)

    report = format_report(results, args.label, args.grpo_analysis)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
