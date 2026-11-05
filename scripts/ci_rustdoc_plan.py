#!/usr/bin/env python3
"""Select workspace CPU Rustdoc without duplicating ordinary leaf-Rust checks."""

import argparse
import json


def rustdoc_required(event, paths, labels=()):
    # Main/manual/merge-queue and unknown events retain the full documentation proof.
    if event != "pull_request" or not paths or "full-ci" in labels:
        return True
    for path in paths:
        if path.startswith(("docs/tracking/", ".codex/campaigns/")):
            continue
        if (
            path.startswith("docs/")
            or path.endswith(".md")
            or path in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "src/lib.rs")
            or path.startswith(".cargo/")
            or path.endswith("/Cargo.toml")
            or path.endswith("/lib.rs")
            or path.startswith("src/api/")
            or "/src/api/" in path
            or path.startswith(("crates/bitnet-ffi/", "crates/bitnet-py/"))
            or path in (".github/workflows/ci-core.yml", ".github/workflows/docs-automation.yml", "scripts/ci_rustdoc_plan.py", "scripts/test_ci_rustdoc_plan.py", "scripts/docs_automation.sh")
        ):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--changed-files", default="")
    parser.add_argument("--labels-json", default="[]")
    args = parser.parse_args()
    labels = json.loads(args.labels_json)
    if labels is None:
        labels = []
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        parser.error("labels must be a JSON array of strings")
    paths = [path for path in args.changed_files.splitlines() if path]
    print(str(rustdoc_required(args.event, paths, labels)).lower())


if __name__ == "__main__":
    main()
