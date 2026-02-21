#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-workspace}"  # workspace | bitnet-server-receipts

run() { echo "+ $*"; "$@"; }

if [[ "${MODE}" == "workspace" ]]; then
  echo "== Clean =="
  cargo clean

  echo "== Build & Test (strict code lints) =="
  RUSTFLAGS="-D warnings" \
    cargo build --locked --workspace --no-default-features --features cpu
  RUSTFLAGS="-D warnings" \
    cargo test  --locked --workspace --no-default-features --features cpu --lib

  echo "== Clippy (strict) =="
  cargo clippy --workspace --all-targets --no-default-features --features cpu \
    -- -D warnings

  echo "== Format check =="
  cargo fmt --all -- --check

  echo "== Docs (relaxed rustdoc) =="
  # Important: only relax rustdoc; keep code builds strict.
  RUSTDOCFLAGS="-A warnings" \
    cargo doc --locked --no-deps --workspace --no-default-features --features cpu

  echo "== MSRV check (1.89.0) =="
  rustup toolchain install 1.89.0 -q || true
  cargo +1.89.0 check --workspace --all-targets --locked \
    --no-default-features --features cpu

  echo "✅ All workspace checks passed."

elif [[ "${MODE}" == "bitnet-server-receipts" ]]; then
  echo "== bitnet-server: receipts validation sequence =="

  echo "Step 1: Baseline CPU check"
  RUSTC_WRAPPER="" RUSTFLAGS="-Dwarnings" \
    cargo +stable check -p bitnet-server --locked --no-default-features --features cpu

  echo "Step 2: Clippy (CPU only)"
  RUSTC_WRAPPER="" \
    cargo +stable clippy -p bitnet-server --all-targets --no-default-features --features cpu \
    -- -D warnings

  echo "Step 3: Format check"
  cargo +stable fmt --all -- --check

  echo "Step 4: Documentation"
  RUSTC_WRAPPER="" RUSTDOCFLAGS="-A warnings" \
    cargo +stable doc -p bitnet-server --locked --no-deps --no-default-features --features cpu

  echo "Step 5: MSRV (1.89.0)"
  RUSTC_WRAPPER="" \
    cargo +1.89.0 check -p bitnet-server --locked --no-default-features --features cpu

  echo "Step 6: Feature combo cpu,receipts"
  RUSTC_WRAPPER="" RUSTFLAGS="-Dwarnings" \
    cargo +stable check -p bitnet-server --locked --no-default-features --features "cpu,receipts"

  echo "Step 7: Feature combo cpu,receipts,tuning"
  RUSTC_WRAPPER="" RUSTFLAGS="-Dwarnings" \
    cargo +stable check -p bitnet-server --locked --no-default-features --features "cpu,receipts,tuning"

  echo "Step 8: Test happy path (receipts enabled)"
  RUSTC_WRAPPER="" \
    cargo +stable test -p bitnet-server --no-default-features --features "cpu,receipts,tuning" \
    -- emits_eviction_receipt_with_correct_payload

  echo "Step 9: Test guard path (receipts disabled)"
  RUSTC_WRAPPER="" \
    cargo +stable test -p bitnet-server --no-default-features --features "cpu,receipts" \
    -- does_not_emit_receipt_when_disabled

  echo "✅ All bitnet-server receipts checks passed."

else
  echo "Usage: $0 [workspace|bitnet-server-receipts]"
  echo ""
  echo "  workspace               - Full workspace validation (default)"
  echo "  bitnet-server-receipts  - Focused bitnet-server receipts validation"
  exit 1
fi
