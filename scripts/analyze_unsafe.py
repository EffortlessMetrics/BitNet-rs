#!/usr/bin/env python3
"""Compatibility launcher for the Rust unsafe-code analyzer."""

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cmd = [
    "cargo",
    "run",
    "--quiet",
    "--locked",
    "--manifest-path",
    str(root / "Cargo.toml"),
    "-p",
    "bitnet-task",
    "--",
    "analyze-unsafe",
    *sys.argv[1:],
]
os.execvp(cmd[0], cmd)
