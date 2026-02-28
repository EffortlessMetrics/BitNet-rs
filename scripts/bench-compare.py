#!/usr/bin/env python3
"""Compare two Criterion benchmark JSON exports and detect regressions.

Usage:
    python scripts/bench-compare.py baseline.json current.json [--threshold 10]

Exits non-zero if any benchmark regresses more than --threshold percent.
Prints a Markdown table suitable for GitHub PR comments.
"""

import json
import sys
import argparse
from pathlib import Path


def load_estimates(path: Path) -> dict[str, float]:
    """Load Criterion estimates from a directory of benchmark JSON files.

    Criterion stores results in target/criterion/<name>/new/estimates.json.
    We also accept a flat JSON dict {name: nanoseconds} for simplicity.
    """
    results = {}

    if path.is_file():
        with open(path) as f:
            data = json.load(f)
        # Flat format: {"bench_name": ns_value, ...}
        if isinstance(data, dict):
            for name, val in data.items():
                if isinstance(val, (int, float)):
                    results[name] = float(val)
                elif isinstance(val, dict) and "point_estimate" in val:
                    results[name] = float(val["point_estimate"])
        return results

    # Directory format: walk Criterion output tree
    if path.is_dir():
        for estimates_file in path.rglob("estimates.json"):
            bench_name = estimates_file.parent.parent.name
            with open(estimates_file) as f:
                data = json.load(f)
            if "median" in data:
                results[bench_name] = float(data["median"]["point_estimate"])
            elif "mean" in data:
                results[bench_name] = float(data["mean"]["point_estimate"])
        return results

    print(f"Error: {path} is neither a file nor a directory", file=sys.stderr)
    sys.exit(2)


def compare(
    baseline: dict[str, float], current: dict[str, float], threshold: float
) -> tuple[list[dict], bool]:
    """Compare benchmarks and return (rows, has_regression)."""
    rows = []
    has_regression = False

    all_names = sorted(set(baseline.keys()) | set(current.keys()))
    for name in all_names:
        base_ns = baseline.get(name)
        curr_ns = current.get(name)

        if base_ns is None:
            rows.append(
                {
                    "name": name,
                    "baseline": "—",
                    "current": f"{curr_ns:.0f}",
                    "change": "NEW",
                    "status": "🆕",
                }
            )
            continue

        if curr_ns is None:
            rows.append(
                {
                    "name": name,
                    "baseline": f"{base_ns:.0f}",
                    "current": "—",
                    "change": "REMOVED",
                    "status": "🗑️",
                }
            )
            continue

        if base_ns == 0:
            pct = 0.0
        else:
            pct = ((curr_ns - base_ns) / base_ns) * 100.0

        if pct > threshold:
            status = "🔴"
            has_regression = True
        elif pct < -threshold:
            status = "🟢"
        else:
            status = "⚪"

        rows.append(
            {
                "name": name,
                "baseline": f"{base_ns:.0f}",
                "current": f"{curr_ns:.0f}",
                "change": f"{pct:+.1f}%",
                "status": status,
            }
        )

    return rows, has_regression


def format_markdown(rows: list[dict], threshold: float) -> str:
    """Format comparison as a GitHub Markdown table."""
    lines = []
    lines.append("## Benchmark Comparison")
    lines.append("")
    lines.append(f"Regression threshold: **{threshold}%**")
    lines.append("")
    lines.append("| Status | Benchmark | Baseline (ns) | Current (ns) | Change |")
    lines.append("|--------|-----------|---------------|--------------|--------|")

    for r in rows:
        lines.append(
            f"| {r['status']} | `{r['name']}` | {r['baseline']} | {r['current']} | {r['change']} |"
        )

    regressions = [r for r in rows if r["status"] == "🔴"]
    improvements = [r for r in rows if r["status"] == "🟢"]

    lines.append("")
    if regressions:
        lines.append(
            f"**{len(regressions)} regression(s) detected** exceeding {threshold}% threshold."
        )
    elif improvements:
        lines.append(f"✅ No regressions. {len(improvements)} improvement(s) detected.")
    else:
        lines.append("✅ No significant changes detected.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare Criterion benchmark results")
    parser.add_argument("baseline", type=Path, help="Baseline benchmark results")
    parser.add_argument("current", type=Path, help="Current benchmark results")
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold in percent (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write Markdown output to file (default: stdout)",
    )
    args = parser.parse_args()

    baseline = load_estimates(args.baseline)
    current = load_estimates(args.current)

    if not baseline:
        print("Warning: no baseline benchmarks found, skipping comparison")
        sys.exit(0)

    rows, has_regression = compare(baseline, current, args.threshold)
    md = format_markdown(rows, args.threshold)

    if args.output:
        args.output.write_text(md)
        print(f"Wrote comparison to {args.output}")
    else:
        print(md)

    if has_regression:
        print(f"\n::error::Benchmark regression detected (>{args.threshold}%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
