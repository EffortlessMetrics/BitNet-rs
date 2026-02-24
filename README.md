# BitNet.rs

[![CI](https://github.com/EffortlessMetrics/BitNet-rs/actions/workflows/ci-core.yml/badge.svg?branch=main)](https://github.com/EffortlessMetrics/BitNet-rs/actions/workflows/ci-core.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.92.0-blue.svg)](./rust-toolchain.toml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](./LICENSE)

Rust inference engine for 1-bit BitNet large language models — memory-safe, cross-validated against the C++ reference, with SIMD/CUDA acceleration.

## CLI Quickstart

```bash
# 1. Download a model
cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf

# 2. Run inference
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 8

# 3. Deterministic benchmark + receipt verification
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
  cargo run -p xtask -- benchmark \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokens 128
cargo run -p xtask -- verify-receipt

# 4. Interactive chat
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json
```

> **Always** specify `--no-default-features --features cpu|gpu` — default features are empty by design.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    bitnet-cli / bitnet-server           │
└────────────────────┬───────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   bitnet-inference  │  autoregressive engine
          │  ┌────────────────┐ │
          │  │ bitnet-sampling│ │  temperature / top-k / top-p
          │  │ bitnet-prompt- │ │  chat templates (raw/instruct/llama3)
          │  │   templates    │ │
          │  │ bitnet-receipts│ │  honest-compute receipts
          │  └────────────────┘ │
          └──────────┬──────────┘
                     │
     ┌───────────────▼─────────────────┐
     │          bitnet-models           │  GGUF loading, transformer
     │  ┌──────────────────────────┐   │
     │  │   bitnet-quantization    │   │  I2_S / TL1 / TL2 / IQ2_S
     │  │   bitnet-kernels (SIMD)  │   │  AVX2 / AVX-512 / NEON / CUDA
     │  └──────────────────────────┘   │
     └──────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │  bitnet-tokenizers  │  universal tokenizer + auto-discovery
          └─────────────────────┘
```

## Status (v0.1.0-qna-mvp)

| Feature                        | Status | Notes |
|-------------------------------|--------|-------|
| CPU inference — I2_S QK256    | ✅     | Scalar kernels (~0.1 tok/s on 2B); AVX2 foundation merged |
| CPU inference — I2_S BitNet32 | ✅     | Production path, 10-20× faster than QK256 scalar |
| GPU inference — CUDA          | ⚠️     | Implemented; receipt validation pending |
| Interactive chat (REPL)       | ✅     | `/help`, `/clear`, `/metrics`, auto-template detection |
| Cross-validation vs C++       | ✅     | Cosine similarity > 0.99, per-token comparison |
| Receipt / honest-compute      | ✅     | Schema v1.0.0, 8 validation gates |
| Strict mode                   | ✅     | Runtime guards prevent mock fallback |
| SafeTensors → GGUF export     | ✅     | `bitnet-st2gguf` with F16 LayerNorm preservation |
| Server / HTTP API             | 🚧     | Health endpoints wired; serving endpoints have TODOs |

## Build

```bash
# CPU (recommended for development)
cargo build --no-default-features --features cpu

# CPU — release + native SIMD
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu,full-cli

# GPU (requires CUDA 12.x)
cargo build --no-default-features --features gpu

# Nix (reproducible, identical to CI)
nix develop
nix build .#bitnet-cli
nix flake check
```

## Test

```bash
# All tests (nextest recommended — 5 min timeout)
cargo nextest run --workspace --no-default-features --features cpu

# CI profile (4 threads, no retries)
cargo nextest run --profile ci

# GGUF fixture tests
cargo test -p bitnet-models --test qk256_dual_flavor_tests --no-default-features --features fixtures

# Skip slow QK256 scalar tests
BITNET_SKIP_SLOW_TESTS=1 cargo nextest run \
  --workspace --no-default-features --features cpu
```

## Documentation

Organised by [Diátaxis](https://diataxis.fr/):

| Section | Contents |
|---------|----------|
| [**Tutorials**](docs/tutorials/) | Getting started, first inference, tokenizer discovery |
| [**How-to**](docs/how-to/) | Install, run inference, export GGUF, cross-validate, validate models |
| [**Explanation**](docs/explanation/) | Architecture, quantization formats, dual-backend, features |
| [**Reference**](docs/reference/) | CLI flags, environment variables, API, quantization support |

### Key guides

- [Quickstart](docs/quickstart.md)
- [Environment variables](docs/environment-variables.md)
- [GPU setup](docs/GPU_SETUP.md)
- [C++ cross-validation setup](docs/how-to/automatic-tokenizer-discovery.md)
- [Quantization support](docs/reference/quantization-support.md)
- [Validation gates](docs/reference/validation-gates.md)
- [QK256 Usage Guide](docs/howto/use-qk256-models.md) — GGML I2_S QK256 Format with 256-element blocks and `--strict-loader` validation
- [Dual I2_S Flavor Architecture](docs/explanation/i2s-dual-flavor.md) — how BitNet.rs differentiates between I2_S format variants

## Receipt Verification

BitNet.rs uses "honest-compute" receipts to verify real inference (no mock fallback).

```bash
# Run benchmark and write receipt
cargo run -p xtask -- benchmark \
  --model models/model.gguf --tokens 128

# Verify receipt against quality gates
cargo run -p xtask -- verify-receipt

# Strict mode — fail on suspicious LN weights (exit code 8)
BITNET_STRICT_MODE=1 cargo run -p xtask -- verify-receipt
```

Receipt JSON schema (v1.0.0):

```json
{
  "version": "1.0.0",
  "compute_path": "real",
  "kernels": ["i2s_cpu_avx2"],
  "tokens_per_sec": 0.1,
  "success": true
}
```

Key environment variables:

| Variable | Purpose |
|----------|---------|
| `BITNET_DETERMINISTIC` | Enable deterministic inference |
| `BITNET_SEED` | Random seed for reproducibility |
| `RAYON_NUM_THREADS` | Worker thread count (1 = single-threaded) |
| `BITNET_STRICT_MODE` | Fail on validation warnings |

Kernel ID hygiene: all kernel IDs must be non-empty strings ≤ 128 chars.
See [baselines/](baselines/) for reference receipts.



See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests welcome.

```bash
# Format + lint
cargo fmt --all && cargo clippy --all-targets --all-features -- -D warnings

# Run tests before pushing
cargo nextest run --workspace --no-default-features --features cpu
```

## License

Dual-licensed under [MIT](LICENSE) and [Apache 2.0](LICENSE).
