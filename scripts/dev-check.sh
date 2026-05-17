#!/usr/bin/env bash
# Common local development checks for bitnet-rs.
#
# The workspace intentionally has empty default features, so this wrapper keeps
# the CPU feature set and --locked flag consistent across fast local validation
# commands. Related one-step Cargo aliases live in .cargo/config.toml.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/dev-check.sh [COMMAND] [-- CARGO_ARGS...]

Commands:
  quick     Run rustfmt --check and cargo check for the CPU workspace (default)
  fmt       Run rustfmt in check mode
  check     Run cargo check for the CPU workspace
  clippy    Run cargo clippy for the CPU workspace with -D warnings
  test      Run CPU workspace tests, preferring cargo-nextest when installed
  all       Run fmt, check, clippy, and test
  help      Show this help text

Examples:
  scripts/dev-check.sh quick
  scripts/dev-check.sh clippy -- -p bitnet-cli
  scripts/dev-check.sh test -- -p bitnet-cli

Arguments after "--" are appended to cargo check/clippy/test commands before
feature flags and test-runner arguments. The fmt command ignores extra cargo
arguments.
USAGE
}

info() {
    printf '[dev-check] %s\n' "$*"
}

run_fmt() {
    info 'cargo fmt --all -- --check'
    cargo fmt --all -- --check
}

run_check() {
    local extra_args=("$@")
    info "cargo check ${extra_args[*]} --locked --workspace --all-targets --no-default-features --features cpu"
    cargo check "${extra_args[@]}" --locked --workspace --all-targets --no-default-features --features cpu
}

run_clippy() {
    local extra_args=("$@")
    info "cargo clippy ${extra_args[*]} --locked --workspace --all-targets --no-default-features --features cpu -- -D warnings"
    cargo clippy "${extra_args[@]}" --locked --workspace --all-targets --no-default-features --features cpu -- -D warnings
}

run_test() {
    local extra_args=("$@")

    if command -v cargo-nextest >/dev/null 2>&1; then
        info "cargo nextest run ${extra_args[*]} --locked --workspace --no-default-features --features cpu"
        cargo nextest run "${extra_args[@]}" --locked --workspace --no-default-features --features cpu
    else
        info "cargo test ${extra_args[*]} --locked --workspace --no-default-features --features cpu"
        cargo test "${extra_args[@]}" --locked --workspace --no-default-features --features cpu
    fi
}

command_name=${1:-quick}
if [[ $# -gt 0 ]]; then
    shift
fi

cargo_args=()
if [[ $# -gt 0 ]]; then
    if [[ $1 != '--' ]]; then
        printf 'error: unexpected argument %q; put cargo args after --\n\n' "$1" >&2
        usage >&2
        exit 2
    fi
    shift
    cargo_args=("$@")
fi

case "$command_name" in
    quick)
        run_fmt
        run_check "${cargo_args[@]}"
        ;;
    fmt)
        run_fmt
        ;;
    check)
        run_check "${cargo_args[@]}"
        ;;
    clippy)
        run_clippy "${cargo_args[@]}"
        ;;
    test)
        run_test "${cargo_args[@]}"
        ;;
    all)
        run_fmt
        run_check "${cargo_args[@]}"
        run_clippy "${cargo_args[@]}"
        run_test "${cargo_args[@]}"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        printf 'error: unknown command %q\n\n' "$command_name" >&2
        usage >&2
        exit 2
        ;;
esac
