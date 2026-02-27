# bitnet-rs

[![CI](https://github.com/EffortlessMetrics/BitNet-rs/actions/workflows/ci-core.yml/badge.svg?branch=main)](https://github.com/EffortlessMetrics/BitNet-rs/actions/workflows/ci-core.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.92.0-blue.svg)](./rust-toolchain.toml)
[![Rust 2024](https://img.shields.io/badge/edition-2024-orange.svg)](./rust-toolchain.toml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](./LICENSE)

Rust inference engine for [BitNet](https://github.com/microsoft/BitNet) 1-bit large language models, with SIMD/CUDA acceleration and cross-validation against the C++ reference.

## Features

- **CPU inference** — AVX2/AVX-512/NEON SIMD kernels; I2_S BitNet32-F16 format at 10–20× QK256 scalar speed
- **GPU inference** — CUDA acceleration via the `gpu` feature (CUDA 12.x required)
- **Quantization formats** — I2_S BitNet32-F16, I2_S QK256 (GGML 256-element blocks), TL1, TL2, IQ2_S via FFI
- **Cross-validation** — per-token cosine similarity comparison against Microsoft's C++ reference (>0.99)
- **Honest-compute receipts** — schema v1.0.0 with 8 validation gates; `compute_path` must be `"real"`
- **Strict mode** — `BITNET_STRICT_MODE=1` rejects mock paths and suspicious LayerNorm weights (exit 8)
- **Chat templates** — raw, instruct, llama3-chat; auto-detected from GGUF metadata
- **SafeTensors → GGUF export** — `bitnet-st2gguf` preserves F16 LayerNorm weights
- **SRP microcrate architecture** — small, focused crates (`bitnet-logits`, `bitnet-sampling`, `bitnet-generation`, …) with zero breaking changes to existing public API

> **Current state (v0.1.0-qna-mvp):** QK256 uses scalar kernels (~0.1 tok/s on 2B models). For validation use `--max-tokens 4–16`. AVX2 dequantization foundation is merged; full ≥3× uplift is planned for v0.2.

## Quick Start

```bash
# 1. Download a model
cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf

# 2. Run inference  (always specify --no-default-features --features cpu|gpu)
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model  models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" --max-tokens 8

# 3. Interactive chat
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model  models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json
```

> Default features are **empty** by design — always pass `--no-default-features --features cpu` (or `gpu`).

## Status

| Feature                        | State | Notes |
|-------------------------------|-------|-------|
| CPU inference — I2_S BitNet32 | ✅    | Production path; 10–20× faster than QK256 scalar |
| CPU inference — I2_S QK256    | ✅    | Scalar kernels (~0.1 tok/s on 2B); AVX2 foundation merged |
| GPU inference — CUDA          | ⚠️   | Implemented; receipt validation pending |
| Interactive chat (REPL)       | ✅    | `/help`, `/clear`, `/metrics`, auto-template detection |
| Cross-validation vs C++       | ✅    | Cosine similarity > 0.99, per-token comparison |
| Honest-compute receipts       | ✅    | Schema v1.0.0, 8 validation gates |
| Strict mode                   | ✅    | Runtime guards prevent mock fallback |
| SafeTensors → GGUF export     | ✅    | `bitnet-st2gguf` with F16 LayerNorm preservation |
| Server / HTTP API             | 🚧    | Health endpoints wired; inference endpoints have TODOs |

## Architecture

Data flows top-to-bottom through the workspace:

```
bitnet-tokenizers ──────────────────────────────────────┐
                                                         │
bitnet-models  (GGUF loader, dual I2_S flavor detection) │
  └── bitnet-quantization  (I2_S / TL1 / TL2 / IQ2_S)  │
        └── bitnet-kernels (AVX2 / AVX-512 / NEON / CUDA)│
                                                         ▼
                        bitnet-inference  (autoregressive engine)
                          ├── bitnet-logits       (temperature / top-k / top-p)
                          ├── bitnet-sampling     (greedy, nucleus, repetition penalty)
                          ├── bitnet-generation   (decode loop, stop criteria)
                          ├── bitnet-prompt-templates  (raw / instruct / llama3-chat)
                          └── bitnet-receipts     (honest-compute receipt schema)
                                                         │
                                          ┌──────────────┴──────────────┐
                                     bitnet-cli                  bitnet-server
```

**SRP microcrates** (`bitnet-logits`, `bitnet-sampling`, `bitnet-generation`, `bitnet-engine-core`, `bitnet-device-probe`, `bitnet-gguf`, `bitnet-prompt-templates`, `bitnet-receipts`) keep coupling low and are re-exported from their original locations for zero breaking changes.

## Documentation

Organised by [Diátaxis](https://diataxis.fr/):

| Section | Contents |
|---------|----------|
| [**Tutorials**](docs/tutorials/) | Getting started, first inference, tokenizer discovery |
| [**How-to**](docs/howto/) | Install, run inference, export GGUF, cross-validate, validate models |
| [**Explanation**](docs/explanation/) | Architecture, quantization formats, dual-backend cross-val, feature flags |
| [**Reference**](docs/reference/) | CLI flags, environment variables, API, quantization support |

Key guides: [Quickstart](docs/quickstart.md) · [Environment variables](docs/environment-variables.md) · [GPU setup](docs/GPU_SETUP.md) · [C++ cross-validation](docs/howto/cpp-setup.md) · [Quantization support](docs/reference/quantization-support.md) · [Validation gates](docs/reference/validation-gates.md) · [Honest-compute receipts](docs/howto/receipt-verification.md) · [QK256 usage](docs/howto/use-qk256-models.md)

## Development

```bash
# Build
cargo build --no-default-features --features cpu           # CPU (development)
cargo build --no-default-features --features gpu           # GPU (requires CUDA 12.x)
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu,full-cli  # optimised release

# Nix (reproducible, identical to CI)
nix develop && nix build .#bitnet-cli && nix flake check

# Test
cargo nextest run --workspace --no-default-features --features cpu   # recommended (5 min timeout)
cargo nextest run --profile ci                                        # CI profile: 4 threads
BITNET_SKIP_SLOW_TESTS=1 cargo nextest run \
  --workspace --no-default-features --features cpu                    # skip slow QK256 scalar tests

# BDD grid compile-coverage check (runs cargo check for every grid cell)
cargo run -p xtask -- grid-check --cpu-only   # fast, suitable for PR CI
cargo run -p xtask -- grid-check --dry-run    # preview commands without running them

# Lint
cargo fmt --all && cargo clippy --all-targets --no-default-features --features cpu -- -D warnings
```

### Testing

The test suite has 3,520 passing tests (100% pass rate) across six complementary strategies:

| Strategy | Tooling | Purpose |
|----------|---------|---------|
| Unit / integration | `cargo nextest` | Per-crate correctness |
| Property-based | `proptest` — 230+ properties, 38 crates | Randomised invariants (quantisation round-trips, sampling reproducibility) |
| Snapshot | `insta` — 192 snapshots, 42 test files | Struct/API stability (breaks on unintended output changes) |
| BDD grid | `xtask grid-check` — 18 cells | Compile-coverage across all feature combinations |
| Fuzz | `cargo-fuzz` — 27 targets, nightly CI | Parser/kernel robustness against arbitrary inputs |
| E2E golden path | pinned token regression | Deterministic CPU inference (seed 42, token sequence `[140,459,459,459]`) |

See [docs/development/test-suite.md](docs/development/test-suite.md) for the full guide.

### Feature flags

| Flag | Purpose |
|------|---------|
| `cpu` | SIMD-optimised CPU inference (AVX2 / AVX-512 / NEON) |
| `gpu` | CUDA acceleration (preferred; requires CUDA 12.x) |
| `cuda` | Backward-compatible alias for `gpu` — prefer `gpu` in new code |
| `ffi` | C++ FFI bridge for cross-validation |
| `fixtures` | GGUF fixture-based integration tests (test-only) |
| `full-cli` | Enable all CLI subcommands |

Always use the unified GPU predicate in Rust code:
```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests welcome.

Before opening a PR, run:
```bash
cargo fmt --all && cargo clippy --all-targets --no-default-features --features cpu -- -D warnings
cargo nextest run --workspace --no-default-features --features cpu
```

Note: ~70 tests are intentionally `#[ignore]`-d (TDD scaffolding for tracked issues #254, #260, #469). This is expected MVP behaviour, not failures to fix.

## License

Dual-licensed under [MIT](LICENSE) and [Apache 2.0](LICENSE).
