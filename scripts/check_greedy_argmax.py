#!/usr/bin/env python3
"""Compatibility wrapper for the Rust greedy-argmax invariant checker."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check greedy argmax invariant")
    parser.add_argument("json_file", help="Path to CLI JSON output")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Accepted for compatibility; Rust checker prints diagnostics on failure.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "--manifest-path",
            str(root / "Cargo.toml"),
            "-p",
            "bitnet-task",
            "--",
            "check-greedy-argmax",
            args.json_file,
        ],
        cwd=root,
        check=False,
    )
    return 7 if result.returncode != 0 else 0


if __name__ == "__main__":
    sys.exit(main())
