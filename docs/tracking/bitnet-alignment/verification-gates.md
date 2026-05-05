# Verification Gates

## Gate 0 — Formatting only

Use for docs-only or tracking-only PRs.

```bash
cargo fmt --all -- --check
```

## Gate 1 — Focused crate

Use for single-crate code changes.

```bash
cargo fmt --all -- --check
cargo clippy --locked -p <crate> --all-targets --no-default-features --features cpu -- -D warnings
cargo test --locked -p <crate> --no-default-features --features cpu
```

## Gate 2 — Workspace CPU

Use for consolidation PRs.

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --no-default-features --features cpu -- -D warnings
cargo nextest run --locked --workspace --no-default-features --features cpu
```

## Gate 3 — Feature/grid

Use for feature lattice or public API changes.

```bash
cargo run --locked -p xtask -- grid-check
cargo nextest run --locked --workspace --no-default-features --features cpu,fixtures
```

## Gate 4 — Runtime proof

Use for inference/runtime changes.

```bash
BITNET_DISABLE_MINIMAL_LOADER=1 \
BITNET_STRICT_MODE=1 \
RUST_LOG=warn \
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "Answer with a single digit: 2+2=" \
  --max-tokens 1 \
  --temperature 0.0 \
  --greedy
```

## Rule

Do not claim a gate passed unless it was actually run.

## Local verification honesty

Record verification mechanically:

- passed:
- failed:
- blocked locally:
- not run:

On Windows, `cargo fmt --all -- --check` may fail with `os error 206` path-length
issues or newline-style issues before formatting starts. If that happens, record it
exactly, run narrower checks when useful, and rely on GitHub CI for the full
formatting gate. Do not solve local environment failures by formatting unrelated
files.
