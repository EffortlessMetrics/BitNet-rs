#!/usr/bin/env python3
"""Compatibility wrapper for the Rust `xtask check-greedy-argmax` command."""

import os
import subprocess
import sys


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        "cargo",
        "run",
        "--locked",
        "--no-default-features",
        "-p",
        "xtask",
        "--",
        "check-greedy-argmax",
        *sys.argv[1:],
    ]
    return subprocess.call(cmd, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
