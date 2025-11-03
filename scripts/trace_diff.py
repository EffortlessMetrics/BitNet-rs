#!/usr/bin/env python3
"""
Compare trace files from two directories to find the first divergence.

Usage:
    python3 scripts/trace_diff.py /tmp/rust_traces /tmp/cpp_traces
"""
import json
import sys
import glob
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

def load_traces(directory: str) -> Dict[Tuple[int, int, str], dict]:
    """Load all trace files from directory into a dict keyed by (seq, layer, stage)."""
    traces = {}

    # Find all .trace and .jsonl files
    patterns = [
        os.path.join(directory, "*.trace"),
        os.path.join(directory, "*.jsonl"),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    record = json.loads(line)

                    # Extract key (seq, layer, stage)
                    seq = record.get('seq')
                    layer = record.get('layer')
                    stage = record.get('stage')

                    # Skip records without seq/layer/stage (backward compat)
                    if seq is None or layer is None or stage is None:
                        continue

                    key = (seq, layer, stage)
                    traces[key] = record
        except Exception as e:
            print(f"Warning: Failed to parse {filepath}: {e}", file=sys.stderr)

    return traces

def compare_traces(rust_dir: str, cpp_dir: str) -> Optional[Tuple[int, int, str]]:
    """
    Compare traces from two directories.

    Returns:
        None if all match, or (seq, layer, stage) of first divergence
    """
    print(f"Loading Rust traces from {rust_dir}...")
    rust_traces = load_traces(rust_dir)
    print(f"  Loaded {len(rust_traces)} Rust tracepoints")

    print(f"Loading C++ traces from {cpp_dir}...")
    cpp_traces = load_traces(cpp_dir)
    print(f"  Loaded {len(cpp_traces)} C++ tracepoints")
    print()

    # Get all keys (sorted for deterministic comparison)
    all_keys = sorted(set(rust_traces.keys()) | set(cpp_traces.keys()))

    for key in all_keys:
        seq, layer, stage = key

        # Check if key exists in both
        if key not in rust_traces:
            print(f"✗ Missing in Rust: seq={seq}, layer={layer}, stage={stage}")
            return key

        if key not in cpp_traces:
            print(f"✗ Missing in C++: seq={seq}, layer={layer}, stage={stage}")
            return key

        rust_rec = rust_traces[key]
        cpp_rec = cpp_traces[key]

        # Compare shapes
        if rust_rec.get('shape') != cpp_rec.get('shape'):
            print(f"✗ Shape mismatch at seq={seq}, layer={layer}, stage={stage}:")
            print(f"  Rust shape: {rust_rec.get('shape')}")
            print(f"  C++ shape:  {cpp_rec.get('shape')}")
            return key

        # Compare dtypes
        if rust_rec.get('dtype') != cpp_rec.get('dtype'):
            print(f"✗ Dtype mismatch at seq={seq}, layer={layer}, stage={stage}:")
            print(f"  Rust dtype: {rust_rec.get('dtype')}")
            print(f"  C++ dtype:  {cpp_rec.get('dtype')}")
            return key

        # Compare blake3 hashes
        rust_hash = rust_rec.get('blake3', '')
        cpp_hash = cpp_rec.get('blake3', '')

        if rust_hash != cpp_hash:
            print(f"✗ First divergence at seq={seq}, layer={layer}, stage={stage}:")
            print(f"  Rust blake3: {rust_hash[:16]}...")
            print(f"  C++ blake3:  {cpp_hash[:16]}...")
            print(f"  Rust stats:  rms={rust_rec.get('rms', 0.0):.6f}, num_elements={rust_rec.get('num_elements', 0)}")
            print(f"  C++ stats:   rms={cpp_rec.get('rms', 0.0):.6f}, num_elements={cpp_rec.get('num_elements', 0)}")
            return key

    print("✓ All tracepoints match")
    return None

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/trace_diff.py <rust_trace_dir> <cpp_trace_dir>")
        sys.exit(2)

    rust_dir = sys.argv[1]
    cpp_dir = sys.argv[2]

    if not os.path.isdir(rust_dir):
        print(f"Error: Rust trace directory not found: {rust_dir}")
        sys.exit(2)

    if not os.path.isdir(cpp_dir):
        print(f"Error: C++ trace directory not found: {cpp_dir}")
        sys.exit(2)

    divergence = compare_traces(rust_dir, cpp_dir)

    if divergence:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
