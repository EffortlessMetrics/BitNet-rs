#!/usr/bin/env python3
"""
compare_traces.py - Analyze trace files to identify first divergence point

Compares Rust and C++ trace files (Blake3 hashes and RMS values) to pinpoint
where activations first diverge during inference.

Usage:
    python scripts/compare_traces.py \\
        --rs-dir /tmp/bitnet-traces \\
        --cpp-dir /tmp/cpp-traces \\
        --output-json report.json \\
        --output-text report.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def colorize(text: str, color: str) -> str:
    """Add color to text if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.RESET}"
    return text


class TraceData:
    """Represents a single trace file."""

    def __init__(self, path: Path, data: dict):
        self.path = path
        self.name = data.get("name", "")
        self.shape = data.get("shape", [])
        self.blake3 = data.get("blake3", "")
        self.rms = data.get("rms", 0.0)
        self.num_elements = data.get("num_elements", 0)

    def __repr__(self):
        return f"TraceData({self.name}, blake3={self.blake3[:8]}..., rms={self.rms:.6f})"


class TraceComparison:
    """Compares two trace files."""

    def __init__(self, rs_trace: TraceData, cpp_trace: TraceData):
        self.rs = rs_trace
        self.cpp = cpp_trace
        self.name = rs_trace.name

        # Compute comparison metrics
        self.blake3_match = rs_trace.blake3 == cpp_trace.blake3
        self.shape_match = rs_trace.shape == cpp_trace.shape

        # Compute relative RMS difference
        max_rms = max(abs(rs_trace.rms), abs(cpp_trace.rms))
        if max_rms > 1e-10:
            self.rms_relative_diff = abs(rs_trace.rms - cpp_trace.rms) / max_rms
        else:
            self.rms_relative_diff = 0.0

        # Flag suspicious if hashes match but RMS differs significantly
        self.suspicious = self.blake3_match and self.rms_relative_diff > 0.01

        # Overall match status
        self.matches = self.blake3_match and self.shape_match

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "name": self.name,
            "blake3_match": self.blake3_match,
            "shape_match": self.shape_match,
            "rs_blake3": self.rs.blake3,
            "cpp_blake3": self.cpp.blake3,
            "rs_rms": self.rs.rms,
            "cpp_rms": self.cpp.rms,
            "rs_shape": self.rs.shape,
            "cpp_shape": self.cpp.shape,
            "rms_relative_diff": self.rms_relative_diff,
            "suspicious": self.suspicious,
            "matches": self.matches
        }


def load_trace_file(path: Path) -> Optional[TraceData]:
    """Load a single trace file."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return TraceData(path, data)
    except json.JSONDecodeError as e:
        print(f"{colorize('Warning:', Colors.YELLOW)} Malformed JSON in {path}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"{colorize('Warning:', Colors.YELLOW)} Failed to load {path}: {e}", file=sys.stderr)
        return None


def load_traces_from_dir(directory: Path) -> Dict[str, TraceData]:
    """Load all trace files from a directory."""
    traces = {}

    if not directory.exists():
        print(f"{colorize('Error:', Colors.RED)} Directory not found: {directory}", file=sys.stderr)
        return traces

    if not directory.is_dir():
        print(f"{colorize('Error:', Colors.RED)} Not a directory: {directory}", file=sys.stderr)
        return traces

    trace_files = list(directory.glob("*.trace"))

    if not trace_files:
        print(f"{colorize('Warning:', Colors.YELLOW)} No *.trace files found in {directory}", file=sys.stderr)

    for path in trace_files:
        trace = load_trace_file(path)
        if trace:
            # Use filename as key for matching
            key = path.name
            traces[key] = trace

    return traces


def parse_trace_order(name: str) -> Tuple[int, int, str]:
    """
    Parse trace name to extract ordering information.

    Returns: (token_idx, layer_idx, component_name)

    Examples:
        "t0_embed.trace" -> (0, -1, "embed")
        "t0_blk0_attn_norm.trace" -> (0, 0, "attn_norm")
        "t0_blk5_q_proj.trace" -> (0, 5, "q_proj")
        "t0_logits.trace" -> (0, 999999, "logits")
    """
    # Remove .trace extension
    base = name.replace(".trace", "")

    # Split by underscore
    parts = base.split("_")

    # Extract token index
    token_idx = 0
    if parts[0].startswith("t"):
        try:
            token_idx = int(parts[0][1:])
            parts = parts[1:]
        except ValueError:
            pass

    # Extract layer index
    layer_idx = -1
    if parts and parts[0].startswith("blk"):
        try:
            layer_idx = int(parts[0][3:])
            parts = parts[1:]
        except ValueError:
            pass

    # Special handling for embeddings and logits
    component = "_".join(parts)
    if component == "embed":
        layer_idx = -1
    elif component == "logits":
        layer_idx = 999999  # Sort logits last

    return (token_idx, layer_idx, component)


def match_and_compare_traces(
    rs_traces: Dict[str, TraceData],
    cpp_traces: Dict[str, TraceData]
) -> Tuple[List[TraceComparison], List[str]]:
    """
    Match traces by filename and compare them.

    Returns: (comparisons, missing_traces)
    """
    comparisons = []
    missing = []

    # Match traces by filename
    all_keys = set(rs_traces.keys()) | set(cpp_traces.keys())

    for key in all_keys:
        if key in rs_traces and key in cpp_traces:
            comparison = TraceComparison(rs_traces[key], cpp_traces[key])
            comparisons.append(comparison)
        elif key in rs_traces:
            missing.append(f"Missing in C++: {key}")
        else:
            missing.append(f"Missing in Rust: {key}")

    # Sort comparisons by order (token, layer, component)
    comparisons.sort(key=lambda c: parse_trace_order(c.name))

    return comparisons, missing


def find_first_divergence(comparisons: List[TraceComparison]) -> Optional[TraceComparison]:
    """Find the first trace where divergence occurs."""
    for comp in comparisons:
        if not comp.matches:
            return comp
    return None


def generate_json_report(
    comparisons: List[TraceComparison],
    missing: List[str],
    first_divergence: Optional[TraceComparison]
) -> dict:
    """Generate JSON report."""
    total_traces = len(comparisons) + len(missing)
    matched_traces = len(comparisons)
    diverged_traces = sum(1 for c in comparisons if not c.matches)

    report = {
        "summary": {
            "total_traces": total_traces,
            "matched_traces": matched_traces,
            "diverged_traces": diverged_traces,
            "missing_traces": len(missing)
        },
        "first_divergence": first_divergence.to_dict() if first_divergence else None,
        "all_comparisons": [c.to_dict() for c in comparisons],
        "missing_traces": missing
    }

    return report


def generate_text_report(
    comparisons: List[TraceComparison],
    missing: List[str],
    first_divergence: Optional[TraceComparison]
) -> str:
    """Generate human-readable text report."""
    lines = []

    lines.append("=" * 60)
    lines.append("  Trace Comparison Report")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    total_traces = len(comparisons) + len(missing)
    matched_traces = len(comparisons)
    diverged_traces = sum(1 for c in comparisons if not c.matches)

    lines.append(f"Total traces: {total_traces}")
    lines.append(f"Matched traces: {matched_traces}")
    lines.append(f"Diverged traces: {diverged_traces}")
    lines.append(f"Missing traces: {len(missing)}")
    lines.append("")

    # First divergence
    if first_divergence:
        lines.append(colorize("FIRST DIVERGENCE FOUND:", Colors.RED + Colors.BOLD))
        lines.append(f"  Name: {first_divergence.name}")
        lines.append(f"  RS Blake3:  {first_divergence.rs.blake3}")
        lines.append(f"  CPP Blake3: {first_divergence.cpp.blake3}")
        lines.append(f"  RS RMS:  {first_divergence.rs.rms:.6f}")
        lines.append(f"  CPP RMS: {first_divergence.cpp.rms:.6f}")
        lines.append(f"  Relative RMS diff: {first_divergence.rms_relative_diff * 100:.2f}%")
        lines.append("")
    else:
        lines.append(colorize("No divergences found - all traces match!", Colors.GREEN + Colors.BOLD))
        lines.append("")

    # Detailed comparison by layer
    lines.append("-" * 60)
    lines.append("Detailed Comparison:")
    lines.append("-" * 60)
    lines.append("")

    current_layer = None

    for comp in comparisons:
        token_idx, layer_idx, component = parse_trace_order(comp.name)

        # Print layer header
        if layer_idx != current_layer:
            current_layer = layer_idx
            if layer_idx == -1:
                lines.append(f"\n{colorize('Embeddings:', Colors.BLUE + Colors.BOLD)}")
            elif layer_idx == 999999:
                lines.append(f"\n{colorize('Logits:', Colors.BLUE + Colors.BOLD)}")
            else:
                lines.append(f"\n{colorize(f'Block {layer_idx}:', Colors.BLUE + Colors.BOLD)}")

        # Print component status
        if comp.matches:
            status = colorize("✓ MATCH", Colors.GREEN)
        else:
            status = colorize("✗ DIVERGED", Colors.RED)

        lines.append(f"  {component}: {status}")

        # Print details for diverged or suspicious traces
        if not comp.matches or comp.suspicious:
            lines.append(f"    RS:  blake3={comp.rs.blake3[:16]}... rms={comp.rs.rms:.6f}")
            lines.append(f"    CPP: blake3={comp.cpp.blake3[:16]}... rms={comp.cpp.rms:.6f}")
            lines.append(f"    Relative diff: {comp.rms_relative_diff * 100:.2f}%")

            if not comp.shape_match:
                lines.append(f"    Shape mismatch: RS={comp.rs.shape} vs CPP={comp.cpp.shape}")

            if comp.suspicious:
                lines.append(colorize("    ⚠ SUSPICIOUS: Hashes match but RMS differs", Colors.YELLOW))

    # Missing traces
    if missing:
        lines.append("")
        lines.append("-" * 60)
        lines.append(colorize("Missing Traces:", Colors.YELLOW + Colors.BOLD))
        lines.append("-" * 60)
        for m in missing:
            lines.append(f"  {m}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Rust and C++ trace files to identify divergence points",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare traces and output to console
    python scripts/compare_traces.py \\
        --rs-dir /tmp/bitnet-traces \\
        --cpp-dir /tmp/cpp-traces

    # Save JSON and text reports
    python scripts/compare_traces.py \\
        --rs-dir /tmp/bitnet-traces \\
        --cpp-dir /tmp/cpp-traces \\
        --output-json report.json \\
        --output-text report.txt
        """
    )

    parser.add_argument(
        "--rs-dir",
        type=Path,
        required=True,
        help="Directory containing Rust trace files"
    )

    parser.add_argument(
        "--cpp-dir",
        type=Path,
        required=True,
        help="Directory containing C++ trace files"
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output path for JSON report (optional)"
    )

    parser.add_argument(
        "--output-text",
        type=Path,
        help="Output path for text report (optional)"
    )

    args = parser.parse_args()

    # Load traces
    print(f"Loading Rust traces from {args.rs_dir}...")
    rs_traces = load_traces_from_dir(args.rs_dir)
    print(f"  Loaded {len(rs_traces)} Rust traces")

    print(f"Loading C++ traces from {args.cpp_dir}...")
    cpp_traces = load_traces_from_dir(args.cpp_dir)
    print(f"  Loaded {len(cpp_traces)} C++ traces")
    print()

    if not rs_traces and not cpp_traces:
        print(colorize("Error: No traces found in either directory", Colors.RED), file=sys.stderr)
        sys.exit(1)

    # Match and compare
    print("Matching and comparing traces...")
    comparisons, missing = match_and_compare_traces(rs_traces, cpp_traces)
    print(f"  Matched {len(comparisons)} trace pairs")
    print(f"  Found {len(missing)} missing traces")
    print()

    # Find first divergence
    first_divergence = find_first_divergence(comparisons)

    # Generate reports
    json_report = generate_json_report(comparisons, missing, first_divergence)
    text_report = generate_text_report(comparisons, missing, first_divergence)

    # Output JSON report
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(json_report, f, indent=2)
        print(f"JSON report written to {args.output_json}")

    # Output text report
    if args.output_text:
        with open(args.output_text, 'w') as f:
            # Strip ANSI codes for file output
            clean_text = text_report
            for color_code in [Colors.GREEN, Colors.RED, Colors.YELLOW, Colors.BLUE, Colors.BOLD, Colors.RESET]:
                clean_text = clean_text.replace(color_code, "")
            f.write(clean_text)
        print(f"Text report written to {args.output_text}")

    # Always print to console
    print()
    print(text_report)

    # Exit with error code if divergences found
    if first_divergence:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
