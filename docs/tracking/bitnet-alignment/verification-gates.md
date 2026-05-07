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
cargo run --locked --no-default-features -p xtask -- grid-check
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

## CPU proof gates

Use these for the strict CPU path before server or GPU claims expand.

### Gate CPU-0 — Focused crate

```bash
cargo fmt --all -- --check
cargo clippy --locked -p <crate> --all-targets --no-default-features --features cpu -- -D warnings
cargo test --locked -p <crate> --no-default-features --features cpu
```

### Gate CPU-1 — Workspace CPU

```bash
cargo clippy --locked --workspace --all-targets --no-default-features --features cpu -- -D warnings
cargo nextest run --locked --workspace --no-default-features --features cpu
```

### Gate CPU-2 — Strict runtime proof

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
  --greedy \
  --json-out
```

### Gate CPU-3 — CPU receipt proof

Required receipt properties:

- `compute_path = real`
- `backend = cpu`
- loader mode is strict/enhanced, not compatibility fallback
- kernel IDs are non-empty and non-mock
- model path or hash is recorded
- tokens/sec or equivalent throughput metric is recorded

## Runtime proof prerequisites

Gate 4 and Gate CPU-2 are proof gates only when:

- server fake output has been fenced
- GGUF minimal fallback is explicit
- strict mode is enabled
- model and tokenizer paths are real and explicit
- output includes or can be tied to an honest receipt

A successful CLI run without these conditions is a smoke check, not runtime proof.

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
