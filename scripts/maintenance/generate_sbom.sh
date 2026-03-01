#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT_DIR/docs/sbom.json"

if ! command -v cargo-cyclonedx >/dev/null 2>&1; then
  echo "cargo-cyclonedx is required. Install with: cargo install cargo-cyclonedx --locked" >&2
  exit 1
fi

cargo cyclonedx --all --format json --override-filename sbom --output-directory "$ROOT_DIR/docs"
if [[ -f "$ROOT_DIR/docs/sbom.cdx.json" ]]; then
  mv "$ROOT_DIR/docs/sbom.cdx.json" "$OUT"
fi

echo "Wrote $OUT"
