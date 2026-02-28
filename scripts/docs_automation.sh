#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[docs-automation] starting documentation checks"

if command -v markdownlint-cli2 >/dev/null 2>&1; then
  echo "[docs-automation] markdownlint"
  markdownlint-cli2 --config .markdownlint.jsonc \
    "**/*.md" \
    "!archive/**" \
    "!docs/archive/**" \
    "!target/**" \
    "!**/node_modules/**"
else
  echo "[docs-automation] markdownlint-cli2 is not installed; skipping markdown lint"
fi

if command -v lychee >/dev/null 2>&1; then
  echo "[docs-automation] lychee link checks"
  lychee --config .lychee.toml \
    "**/*.md" \
    "docs/**" \
    "README.md" \
    "CONTRIBUTING.md" \
    "CLAUDE.md"
else
  echo "[docs-automation] lychee is not installed; skipping link checks"
fi

echo "[docs-automation] rustdoc build"
RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -A warnings" cargo doc --locked --no-deps --workspace --no-default-features --features cpu

echo "[docs-automation] completed"
