# bitnet-rs

[![CI](https://github.com/EffortlessMetrics/BitNet-rs/actions/workflows/ci-core.yml/badge.svg?branch=main)](https://github.com/EffortlessMetrics/BitNet-rs/actions/workflows/ci-core.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.92.0-blue.svg)](./rust-toolchain.toml)
[![Rust 2024](https://img.shields.io/badge/edition-2024-orange.svg)](./rust-toolchain.toml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](./LICENSE)

High-performance Rust inference engine for 1-bit BitNet LLMs.

> [!WARNING]
> **Pre-alpha (v0.2.1-dev).** QK256 uses scalar kernels (~0.1 tok/s on 2B models). GPU backends are scaffolded but not validated. Significant correctness, performance, and validation work remains. Do not use in production.

## Quick Start

```bash
# Build (always specify features — defaults are empty)
cargo build --locked --no-default-features --features cpu

# Download a model
cargo run --locked -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf

# Run inference
RUST_LOG=warn cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" --max-tokens 8

# Interactive chat (auto-detects prompt template)
RUST_LOG=warn cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json
```

See [docs/quickstart.md](docs/quickstart.md) for the full getting-started guide.

## Status

| Feature | State | Notes |
|---------|-------|-------|
| CPU inference (I2_S BitNet32) | Working | Production path; SIMD-optimised |
| CPU inference (I2_S QK256) | Working | Scalar only (~0.1 tok/s on 2B); AVX2 foundation merged |
| Cross-validation vs C++ | Working | Per-token logits comparison framework |
| Honest-compute receipts | Working | Schema v1.0.0, 8 validation gates |
| Interactive chat (REPL) | Working | Auto-template detection, `/help`, `/clear`, `/metrics` |
| SafeTensors to GGUF export | Working | `bitnet-st2gguf` with F16 LayerNorm preservation |
| 59+ prompt templates | Working | LLaMA-3, Phi-4, Qwen, Gemma, Mistral, DeepSeek, etc. |
| GPU inference (CUDA) | Scaffold | Feature-gated; receipt validation pending |
| GPU backends (Metal/Vulkan/OpenCL/ROCm) | Scaffold | Stubs only; not validated |
| Server / HTTP API | Incomplete | Health endpoints wired; inference endpoints have TODOs |

## Architecture

```
bitnet-tokenizers ──────────────────────────────────────┐
                                                         │
bitnet-models  (GGUF loader, dual I2_S flavor detection) │
  └── bitnet-quantization  (I2_S / TL1 / TL2 / IQ2_S)  │
        └── bitnet-kernels (AVX2 / AVX-512 / NEON / CUDA)│
                                                         ▼
                        bitnet-inference  (autoregressive engine)
                          ├── bitnet-logits / bitnet-sampling / bitnet-generation
                          ├── bitnet-prompt-templates  (59+ variants)
                          └── bitnet-receipts     (honest-compute schema)
                                                         │
                                          ┌──────────────┴──────────────┐
                                     bitnet-cli                  bitnet-server
```

The workspace contains ~200 crates. See [docs/architecture-overview.md](docs/architecture-overview.md) for details.

## Building

```bash
cargo build --locked --no-default-features --features cpu           # CPU (development)
cargo build --locked --no-default-features --features gpu           # GPU (requires CUDA 12.x)
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --locked --release --no-default-features --features cpu,full-cli  # optimised release
```

### Feature flags

| Flag | Purpose |
|------|---------|
| `cpu` | SIMD-optimised CPU inference (AVX2/AVX-512/NEON) |
| `gpu` | GPU umbrella — CUDA backend |
| `full-cli` | Enable all CLI subcommands |
| `ffi` | C++ FFI bridge for cross-validation |
| `fixtures` | GGUF fixture-based integration tests (test-only) |

Nix: `nix develop && nix build .#bitnet-cli && nix flake check` — see [Nix guide](docs/kv-pool/NIX_FLAKE_USAGE.md).

## Testing

```bash
# Run all enabled tests (recommended)
cargo nextest run --locked --workspace --no-default-features --features cpu

# CI profile (4 threads, no retries)
cargo nextest run --locked --profile ci

# Lint
cargo fmt --all && cargo clippy --locked --all-targets --no-default-features --features cpu -- -D warnings
```

~58,700 test annotations spanning unit, property-based (proptest), snapshot (insta), fixture, fuzz (109 targets), and BDD categories. ~2,800 tests are intentionally `#[ignore]`-d with justification strings. See [docs/development/test-suite.md](docs/development/test-suite.md).

## Documentation

Organised by [Diataxis](https://diataxis.fr/):

| Section | Contents |
|---------|----------|
| [Tutorials](docs/tutorials/) | Getting started, first inference, tokenizer discovery |
| [How-to](docs/howto/) | Install, run inference, export GGUF, cross-validate, validate models |
| [Explanation](docs/explanation/) | Architecture, quantization formats, feature flags |
| [Reference](docs/reference/) | [Inference CLI](docs/reference/inference-cli-reference.md), [Cross-validation CLI](docs/reference/crossval-cli-reference.md), [environment variables](docs/environment-variables.md), [quantization](docs/reference/quantization-support.md) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Before opening a PR:

```bash
./ci/local.sh   # or: cargo fmt --all && cargo clippy --locked ... && cargo nextest run --locked ...
```

See [ROADMAP.md](ROADMAP.md) for project direction.

## License

Dual-licensed under [MIT](LICENSE) and [Apache 2.0](LICENSE).
