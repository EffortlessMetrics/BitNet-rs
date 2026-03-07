#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [ $# -eq 0 ]; then
    set -- master
fi
exec cargo run --quiet --locked --manifest-path "$ROOT/Cargo.toml" -p bitnet-task -- vendor-ggml-quants "$@"
