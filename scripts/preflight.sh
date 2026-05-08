#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    eval "$(cargo run --quiet --locked --manifest-path "$ROOT/Cargo.toml" -p bitnet-task -- preflight --emit-env)"
    return $?
fi

exec cargo run --quiet --locked --manifest-path "$ROOT/Cargo.toml" -p bitnet-task -- preflight "$@"
