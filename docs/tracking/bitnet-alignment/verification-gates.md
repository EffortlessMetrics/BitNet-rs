# Verification Gates

## Gate 0 — Formatting only

```bash
cargo fmt --all -- --check
```

## Gate 1 — Focused crate

```bash
cargo fmt --all -- --check
cargo clippy --locked -p <crate> --all-targets --no-default-features --features cpu -- -D warnings
cargo test --locked -p <crate> --no-default-features --features cpu
```

## Gate 2 — Workspace CPU

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --no-default-features --features cpu -- -D warnings
cargo nextest run --locked --workspace --no-default-features --features cpu
```

## Gate 3 — Feature/grid

```bash
cargo run --locked -p xtask -- grid-check
cargo nextest run --locked --workspace --no-default-features --features cpu,fixtures
```

## Gate 4 — Runtime proof

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

Do not claim a gate passed unless it was actually run.
