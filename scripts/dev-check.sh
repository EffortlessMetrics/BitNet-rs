#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CARGO_BIN="${CARGO:-cargo}"
FEATURES="${BITNET_DEV_FEATURES:-cpu}"
MODE="${1:-quick}"

usage() {
  cat <<'USAGE'
Usage: scripts/dev-check.sh [quick|full|fmt|clippy|test]

Fast local feedback for contributors. By default this checks the workspace
Cargo default-members with the CPU feature set, which avoids accidentally
running every opt-in crate while still catching common format, build, and unit
test issues.

Environment:
  CARGO                Cargo executable to use (default: cargo)
  BITNET_DEV_FEATURES  Feature set to pass with --no-default-features (default: cpu)
USAGE
}

run() {
  printf '\n+'
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  "$@"
}

case "$MODE" in
  quick)
    run "$CARGO_BIN" fmt --all -- --check
    run "$CARGO_BIN" check --locked --no-default-features --features "$FEATURES"
    run "$CARGO_BIN" test --locked --lib --no-default-features --features "$FEATURES"
    ;;
  full)
    run "$CARGO_BIN" fmt --all -- --check
    run "$CARGO_BIN" clippy --locked --workspace --all-targets --no-default-features --features "$FEATURES" -- -D warnings
    run "$CARGO_BIN" test --locked --workspace --no-default-features --features "$FEATURES"
    ;;
  fmt)
    run "$CARGO_BIN" fmt --all -- --check
    ;;
  clippy)
    run "$CARGO_BIN" clippy --locked --workspace --all-targets --no-default-features --features "$FEATURES" -- -D warnings
    ;;
  test)
    run "$CARGO_BIN" test --locked --lib --no-default-features --features "$FEATURES"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "error: unknown dev-check mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
