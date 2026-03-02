# bitnet-rs

[![CI](https://github.com/EffortlessMetrics/BitNet-rs/actions/workflows/ci-core.yml/badge.svg?branch=main)](https://github.com/EffortlessMetrics/BitNet-rs/actions/workflows/ci-core.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.92.0-blue.svg)](./rust-toolchain.toml)
[![Rust 2024](https://img.shields.io/badge/edition-2024-orange.svg)](./rust-toolchain.toml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](./LICENSE)

BitNet-rs is a high-performance Rust inference engine for 1-bit BitNet LLMs.

## Features

- **SIMD/CUDA/Metal/Vulkan kernels** — AVX2/AVX-512/NEON on CPU; CUDA (`gpu`), Metal (`metal`, macOS), Vulkan (`vulkan`), Intel Arc OpenCL (`opencl`) GPU backends
- **Multiple quantization formats** — I2_S BitNet32-F16, I2_S QK256 (GGML 256-element blocks), TL1, TL2, IQ2_S via FFI
- **Cross-validation** — per-token cosine-similarity comparison against Microsoft's C++ reference (>0.99)
- **Honest-compute receipts** — schema v1.0.0 with 8 validation gates; `compute_path` must be `"real"`
- **Chat templates** — 59+ template variants (LLaMA-3, Phi-4, Qwen, Gemma, Mistral, DeepSeek, and more); auto-detected from GGUF metadata or tokenizer path
- **SLM model support** — load and run Phi-4, Qwen, Gemma, Mistral, LLaMA, and SmolLM2 via SafeTensors ([quickstart guide](docs/slm-quickstart.md))
- **SafeTensors → GGUF export** — `bitnet-st2gguf` preserves F16 LayerNorm weights

> **v0.2.1-dev (pre-alpha):** QK256 uses scalar kernels (~0.1 tok/s on 2B models); use `--max-tokens 4–16` for validation. AVX2 dequantization is merged; ≥3× uplift planned. Significant correctness, performance, and validation work remains.

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
| GPU inference — CUDA          | ⚠️    | Implemented; receipt validation pending |
| GPU inference — Metal         | ⚠️    | Feature gate + kernel stubs; validation in progress |
| GPU inference — Vulkan        | ⚠️    | Runtime probing compiled; end-to-end validation pending |
| GPU inference — Intel oneAPI  | ⚠️    | Intel CPU/GPU feature gate; validation in progress |
| AMD ROCm detection            | ⚠️    | Device detection only; inference kernels not yet validated |
| GPU HAL — multi-backend       | 🔧    | `bitnet-gpu-hal`: OpenCL, Vulkan, Metal, ROCm backends; 10,000+ tests (scaffold; CPU-only validation) |
| Interactive chat (REPL)       | ✅    | `/help`, `/clear`, `/metrics`, auto-template detection |
| Cross-validation vs C++       | ✅    | Cosine similarity > 0.99, per-token comparison |
| Honest-compute receipts       | ✅    | Schema v1.0.0, 8 validation gates |
| Strict mode                   | ✅    | Runtime guards prevent mock fallback |
| SafeTensors → GGUF export     | ✅    | `bitnet-st2gguf` with F16 LayerNorm preservation |
| Server / HTTP API             | 🚧    | Health endpoints wired; inference endpoints have TODOs |

## GPU Multi-Backend Support

BitNet-rs supports inference on multiple GPU platforms:

| Backend | Feature Flag | Status | Hardware |
|---------|-------------|--------|----------|
| NVIDIA CUDA | `--features gpu` | ✅ Production | GeForce/Tesla/A100+ |
| Intel Arc (OpenCL) | `--features opencl` | 🔶 Alpha | Arc A770/A750 |
| AMD ROCm | `--features rocm` | 🧪 Experimental | Unvalidated target: RDNA3-class AMD GPUs |
| Vulkan | `--features vulkan` | 🧪 Experimental | Any Vulkan 1.3 GPU |
| Apple Metal | `--features metal` | 🧪 Experimental | M1/M2/M3+ |
| WebGPU | N/A (sub-crate only) | 🧪 Experimental | Browser/wgpu (`bitnet-wgpu`) |
| CPU (SIMD) | `--features cpu` | ✅ Production | x86-64/ARM64 |

### Quick Start (Intel Arc)

```bash
# Install Intel compute runtime (Ubuntu)
sudo apt install intel-opencl-icd clinfo

# Build with Intel GPU support
cargo build --release --no-default-features --features opencl,full-cli

# Run inference
cargo run -p bitnet-cli --no-default-features --features opencl,full-cli -- run \
  --model models/model.gguf --device opencl --prompt "Hello" --max-tokens 32
```

See [docs/INTEL_GPU_SETUP.md](docs/INTEL_GPU_SETUP.md) for detailed setup instructions.

### Device Selection

```bash
--device auto     # Auto-detect best available (default)
--device cpu      # Force CPU
--device cuda     # Force NVIDIA CUDA
--device opencl   # Force Intel OpenCL
--device vulkan   # Force Vulkan
```

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
                          ├── bitnet-prompt-templates  (59+ template variants; auto-detection)
                          └── bitnet-receipts     (honest-compute receipt schema)
                                                         │
                                          ┌──────────────┴──────────────┐
                                     bitnet-cli                  bitnet-server
```

**SRP microcrates** (`bitnet-logits`, `bitnet-sampling`, `bitnet-generation`, `bitnet-engine-core`, `bitnet-device-probe`, `bitnet-gguf`, `bitnet-prompt-templates`, `bitnet-receipts`) keep coupling low and are re-exported from their original locations for zero breaking changes.

### GPU Backend Crates

- `bitnet-opencl` — Intel GPU compute via OpenCL 3.0
- `bitnet-vulkan` — Cross-vendor Vulkan compute
- `bitnet-wgpu` / `bitnet-wgpu-runner` — WebGPU/WGSL compute shaders
- `bitnet-rocm` — AMD ROCm/HIP backend
- `bitnet-metal` — Apple Metal compute
- `bitnet-gpu-hal` — Unified Hardware Abstraction Layer (includes Level Zero backend module)

## Documentation

Organised by [Diátaxis](https://diataxis.fr/):

| Section | Contents |
|---------|----------|
| [**Tutorials**](docs/tutorials/) | Getting started, first inference, tokenizer discovery |
| [**How-to**](docs/howto/) | Install, run inference, export GGUF, cross-validate, validate models |
| [**Explanation**](docs/explanation/) | Architecture, quantization formats, dual-backend cross-val, feature flags |
| [**Reference**](docs/reference/) | CLI flags, environment variables, API, quantization support |

Key guides: [Quickstart](docs/quickstart.md) · [SLM models](docs/slm-quickstart.md) · [Environment variables](docs/environment-variables.md) · [GPU setup](docs/GPU_SETUP.md) · [Intel GPU setup](docs/INTEL_GPU_SETUP.md) · [C++ cross-validation](docs/howto/cpp-setup.md) · [Quantization support](docs/reference/quantization-support.md) · [Validation gates](docs/reference/validation-gates.md) · [Honest-compute receipts](docs/howto/receipt-verification.md) · [QK256 usage](docs/howto/use-qk256-models.md) · [macOS 26 Apple Silicon roadmap](docs/reference/macos-26-apple-silicon-roadmap.md)

## Building

```bash
cargo build --no-default-features --features cpu           # CPU (development)
cargo build --no-default-features --features gpu           # GPU (requires CUDA 12.x)
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu,full-cli  # optimised release

# Nix (reproducible, identical to CI)
nix develop && nix build .#bitnet-cli && nix flake check
```

### Feature flags

| Flag | Purpose |
|------|---------|
| `cpu` | SIMD-optimised CPU inference (AVX2 / AVX-512 / NEON) |
| `gpu` | Umbrella GPU feature — enables all compiled GPU backends |
| `cuda` | CUDA acceleration (preferred; requires CUDA 12.x); backward-compat alias for `gpu` |
| `metal` | Metal GPU backend (macOS/iOS Apple Silicon) |
| `vulkan` | Vulkan compute backend (cross-platform) |
| `oneapi` | Intel oneAPI (sub-crate feature in `bitnet-kernels`; use `opencl` for root-level Intel GPU) |
| `ffi` | C++ FFI bridge for cross-validation |
| `fixtures` | GGUF fixture-based integration tests (test-only) |
| `full-cli` | Enable all CLI subcommands |
| `rocm` | AMD ROCm detection (device probe; inference kernels not yet validated) |
| `npu` | NPU detection via `bitnet-device-probe` |
| `opencl` | Intel Arc OpenCL backend (experimental; `bitnet-opencl` crate) |

Always use the unified GPU predicate in Rust code:
```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
```

## Testing

```bash
# Run all enabled tests (recommended — 5-minute timeout)
cargo nextest run --workspace --no-default-features --features cpu

# CI profile (4 threads, no retries)
cargo nextest run --profile ci

# Skip slow QK256 scalar-kernel tests
BITNET_SKIP_SLOW_TESTS=1 cargo nextest run --workspace --no-default-features --features cpu

# BDD compile-coverage check
cargo run -p xtask -- grid-check

# Fixture-based integration tests
cargo test -p bitnet-models --test qk256_dual_flavor_tests --no-default-features --features fixtures

# Lint before pushing
cargo fmt --all && cargo clippy --all-targets --no-default-features --features cpu -- -D warnings
```

The suite has tens of thousands of tests spanning unit, property-based (proptest), snapshot (insta), fixture, fuzz (84 targets; 45 in nightly CI matrix), and BDD grid categories. ~1,050+ tests are intentionally `#[ignore]`-d — TDD scaffolds, resource-gated tests, slow tests, and crossval tests. See `#[ignore = "..."]` justification strings.

See [docs/development/test-suite.md](docs/development/test-suite.md) for full details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests welcome.

Before opening a PR, run:
```bash
cargo fmt --all && cargo clippy --all-targets --no-default-features --features cpu -- -D warnings
cargo nextest run --workspace --no-default-features --features cpu
```

Note: ~1,050+ tests are intentionally `#[ignore]`-d. This is expected — they are TDD scaffolds, resource-gated tests (model files, GPU hardware), slow tests, and crossval tests. See `#[ignore = "..."]` justification strings.

## License

Dual-licensed under [MIT](LICENSE) and [Apache 2.0](LICENSE).
