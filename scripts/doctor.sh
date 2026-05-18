#!/usr/bin/env sh
# Lightweight local environment diagnostic for BitNet-rs contributors.
# Keep this script POSIX-sh compatible so it works before optional dev tooling is installed.

set -u

failures=0
warnings=0

info() {
  printf 'ℹ️  %s\n' "$*"
}

pass() {
  printf '✅ %s\n' "$*"
}

warn() {
  warnings=$((warnings + 1))
  printf '⚠️  %s\n' "$*"
}

fail() {
  failures=$((failures + 1))
  printf '❌ %s\n' "$*"
}

need_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    pass "found $1: $(command -v "$1")"
  else
    fail "missing required command: $1"
  fi
}

optional_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    pass "found optional $1: $(command -v "$1")"
  else
    warn "optional command not found: $1 ($2)"
  fi
}

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root" || exit 1

info "BitNet-rs developer environment doctor"
info "repo: $repo_root"
printf '\n'

need_cmd git
need_cmd cargo
need_cmd rustc

expected_toolchain=$(sed -n 's/^channel = "\([^"]*\)"/\1/p' rust-toolchain.toml | head -n 1)
if [ -z "$expected_toolchain" ]; then
  fail "could not read channel from rust-toolchain.toml"
elif command -v rustc >/dev/null 2>&1; then
  rustc_version=$(rustc --version 2>/dev/null || true)
  case "$rustc_version" in
    *" $expected_toolchain"*) pass "rustc matches rust-toolchain.toml ($rustc_version)" ;;
    *) fail "rustc does not match rust-toolchain.toml: expected $expected_toolchain, got ${rustc_version:-unknown}" ;;
  esac
fi

if command -v cargo >/dev/null 2>&1; then
  cargo_version=$(cargo --version 2>/dev/null || true)
  info "cargo: ${cargo_version:-unknown}"
fi

printf '\n'
info "Checking workspace metadata with Cargo.lock pinned dependencies"
if command -v cargo >/dev/null 2>&1; then
  if cargo metadata --locked --no-deps --format-version 1 >/dev/null; then
    pass "cargo metadata --locked --no-deps"
  else
    fail "cargo metadata --locked --no-deps failed"
  fi
fi

printf '\n'
info "Checking common optional developer tools"
optional_cmd cargo-nextest "recommended for fast workspace test runs"
optional_cmd cargo-watch "used by make watch"
optional_cmd cargo-audit "used by make audit"
optional_cmd cargo-outdated "used by make outdated"
optional_cmd cargo-bloat "used by make bloat"
optional_cmd tokei "used by make loc"
optional_cmd tree "used by make tree"
optional_cmd docker "used by make docker"

printf '\n'
info "Detected runtime feature default"
if command -v nvidia-smi >/dev/null 2>&1; then
  pass "nvidia-smi detected; Makefile will default FEATURES=gpu"
else
  os_name=$(uname -s 2>/dev/null | tr '[:upper:]' '[:lower:]')
  arch_name=$(uname -m 2>/dev/null)
  if [ "$os_name" = "darwin" ] && [ "$arch_name" = "arm64" ]; then
    pass "Apple Silicon detected; Makefile will default FEATURES=gpu"
  else
    pass "no default GPU signal detected; Makefile will default FEATURES=cpu"
  fi
fi

printf '\n'
if [ "$failures" -eq 0 ]; then
  if [ "$warnings" -eq 0 ]; then
    pass "doctor completed with no failures or warnings"
  else
    pass "doctor completed with no failures ($warnings warning(s))"
  fi
  exit 0
fi

fail "doctor found $failures failure(s) and $warnings warning(s)"
exit 1
