#!/usr/bin/env bash
# Rust-backed compatibility shim for adding --locked to cargo/cross commands.
# Usage:
#   scripts/fix-locked.sh .github/workflows/*.yml
#   scripts/fix-locked.sh --dry-run .github/workflows/*.yml
#   scripts/fix-locked.sh --check .github/workflows/*.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

exec cargo run --quiet --locked --manifest-path "$REPO_ROOT/tools/bitnet-task/Cargo.toml" -- fix-locked "$@"
