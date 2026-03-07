#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec cargo run --quiet --locked --manifest-path "$ROOT/Cargo.toml" -p bitnet-task -- show-quant-status "$@"
