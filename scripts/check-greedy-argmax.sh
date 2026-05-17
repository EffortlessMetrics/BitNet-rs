#!/usr/bin/env bash
# Rust-backed greedy argmax invariant checker for BitNet CLI JSON receipts.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec cargo run --quiet --locked --manifest-path "$ROOT/Cargo.toml" -p bitnet-task -- check-greedy-argmax "$@"
