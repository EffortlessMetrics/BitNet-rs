#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
args=()

if [ $# -ge 1 ] && [ -n "${1:-}" ] && [[ "${1}" != -* ]]; then
    args+=(--model "$1")
    shift
fi

if [ $# -ge 1 ] && [ -n "${1:-}" ] && [[ "${1}" != -* ]]; then
    args+=(--tokenizer "$1")
    shift
fi

exec cargo run --quiet --locked --manifest-path "$ROOT/Cargo.toml" -p bitnet-task -- perf-phase1-quant-probe "${args[@]}" "$@"
