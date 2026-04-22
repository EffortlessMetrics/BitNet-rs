# bitnet-rs

[![CI](https://github.com/EffortlessMetrics/BitNet-rs/actions/workflows/ci-core.yml/badge.svg?branch=main)](https://github.com/EffortlessMetrics/BitNet-rs/actions/workflows/ci-core.yml)
[![Codecov](https://codecov.io/gh/EffortlessMetrics/BitNet-rs/graph/badge.svg?branch=main)](https://codecov.io/gh/EffortlessMetrics/BitNet-rs)
[![MSRV](https://img.shields.io/badge/MSRV-1.92.0-blue.svg)](./rust-toolchain.toml)
[![Rust 2024](https://img.shields.io/badge/edition-2024-orange.svg)](./rust-toolchain.toml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](./LICENSE)

Pre-alpha Rust inference engine and validation workspace for 1-bit BitNet LLMs.

> [!WARNING]
> **Pre-alpha. Do not use in production.**
>
> BitNet-rs is not yet a general-purpose local chat engine. The project currently focuses on loader, tokenizer, kernel, receipt, and hardware validation. Coherent BitNet answer quality is still under validation, so generated BitNet text should be treated as diagnostic output.

## What This Repo Is For

BitNet-rs is moving toward Rust-native BitNet inference, but the current repo is best understood as an inference-systems validation workspace. It is useful for contributors working on model loading, tokenization, quantization, kernel parity, hardware bring-up, receipts, and reproducible inference validation. It is not yet a polished end-user inference server.

What exists today:

- strict GGUF loading and tokenizer metadata checks
- I2_S / QK256 quantization and kernel infrastructure
- scalar, AVX2, AVX-512, NEON, CUDA, OpenCL, OpenVINO, Metal, and NPU validation work
- diagnostic answer-corpus and answer-parity receipts
- receipts that record hardware identity, runtime identity, fallback behavior, and kernel coverage
- dense SLM companion work used to validate the generation pipeline while BitNet model-artifact work continues

## Current Status

The repo has real inference infrastructure, but it does not yet provide supported coherent Rust BitNet local answers. The Microsoft BitNet.cpp reference path can answer the tiny suite with the official I2_S GGUF when the missing pre-tokenizer is supplied from Microsoft's tokenizer assets.

Backend receipts remain useful for selected-device execution, tokenizer and prompt diagnostics, fallback behavior, and kernel coverage. They are not, by themselves, evidence that the Rust-generated text is a supported answer. See the [model-artifact validation docs](docs/model-artifacts/ANSWER_ARTIFACT_GATE.md).

## Capability Matrix

| Area | State | What it means today |
|---|---|---|
| GGUF loading | Supported / hardening | Structural loading and metadata extraction are active work surfaces. |
| Tokenizer handling | Supported / hardening | Tokenizer metadata is checked strictly for answer-quality work. |
| I2_S BitNet32 CPU path | Diagnostic | CPU execution exists; coherent BitNet answer quality is still under validation. |
| I2_S QK256 CPU path | Diagnostic | Scalar, AVX2, and AVX-512 diagnostics have receipts; generated text quality is still under validation. |
| Scalar / SIMD parity | Diagnostic | Used for backend agreement checks and first-divergence debugging. |
| Dense SLM path | Early working | Companion/control path for generation-pipeline validation; not a BitNet quality result. |
| RTX 5070 Ti CUDA | Execution path validated / diagnostic | Packed BitNet CUDA has receipts through `CUDA-BITNET-009`; coherent CUDA answers and speed are not established. |
| Metal / OpenCL / OpenVINO / NPU | Probe / smoke | Hardware identity and narrow execution receipts exist; full BitNet answer quality is not established. |
| Cross-validation | Supported / hardening | Reference comparison infrastructure exists; model selection remains active work. |
| Honest-compute receipts | Supported | Receipts preserve backend, runtime, fallback, kernel, and timing metadata. |
| CLI run/chat | Diagnostic | Useful for exercising the pipeline; generated text is not yet a supported answer-quality surface. |
| Server / HTTP API | Incomplete | Health wiring exists; inference serving is not ready. |

## First Diagnostic Run

| Need | Start here |
|---|---|
| First token-generation walkthrough | [docs/tutorials/first-inference.md](docs/tutorials/first-inference.md) |
| Real GGUF model walkthrough | [docs/tutorials/real-gguf-model-inference.md](docs/tutorials/real-gguf-model-inference.md) |
| Model validation workflow | [docs/howto/validate-models.md](docs/howto/validate-models.md) |
| GGUF loading details | [docs/howto/gguf-model-validation-and-loading.md](docs/howto/gguf-model-validation-and-loading.md) |
| CLI flags and receipt options | [docs/reference/inference-cli-reference.md](docs/reference/inference-cli-reference.md) |

The commands below are a smoke path for contributors, not an answer-quality quickstart.

Build the CPU CLI:

```bash
cargo build --locked -p bitnet-cli --no-default-features --features cpu,full-cli
```

Download the official Microsoft BitNet GGUF:

```bash
cargo run --locked -p xtask --no-default-features -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf
```

Run a diagnostic CPU generation path:

```bash
RUST_LOG=warn cargo run --locked -p bitnet-cli \
  --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-new-tokens 8 \
  --strict-loader \
  --strict-tokenizer \
  --json-out target/bitnet/receipts/first-run.json
```

This exercises the model, tokenizer, generation, and receipt path. Treat the output as diagnostic evidence, not as a supported chat answer.

## Architecture

```text
bitnet-tokenizers --------------------------------------+
                                                        |
bitnet-models  (GGUF loader, I2_S detection, metadata)  |
  -> bitnet-quantization  (I2_S / TL1 / TL2 / IQ2_S)    |
        -> bitnet-kernels (scalar / AVX2 / AVX-512 / NEON / CUDA)
                                                        v
                        bitnet-inference  (autoregressive engine)
                          -> bitnet-logits
                          -> bitnet-sampling
                          -> bitnet-generation
                          -> bitnet-prompt-templates
                          -> bitnet-receipts
                                                        |
                                      +-----------------+----------------+
                                      v                                  v
                                  bitnet-cli                       bitnet-server
```

The workspace contains roughly 200 crates. See [docs/architecture-overview.md](docs/architecture-overview.md).

## Hardware Validation

Hardware validation is organized by platform so backend identity, runtime identity, fallback status, and receipt coverage stay explicit.

| Platform | Role |
|---|---|
| Intel 258V CPU | Lead BitNet CPU reference and AVX2 diagnostics. |
| i5-8250U CPU | Dense SLM CPU lead and low-power comparison. |
| Ryzen 9950X3D | AVX-512 support and high-performance CPU diagnostics. |
| RTX 5070 Ti | CUDA packed BitNet validation and future answer path. |
| Apple M4 | Metal, MPSGraph, and CPU/NEON validation. |
| Arc A770 | Discrete Intel GPU OpenCL/OpenVINO validation. |
| Arc 140V | Lunar Lake iGPU OpenCL/OpenVINO validation. |
| Intel NPU | OpenVINO NPU static-shape validation. |

See [docs/hardware/HARDWARE_MATRIX.md](docs/hardware/HARDWARE_MATRIX.md).

## Building

```bash
cargo build --locked --no-default-features --features cpu
cargo build --locked -p bitnet-cli --no-default-features --features cpu,full-cli
cargo build --locked --no-default-features --features gpu
```

Optimized CPU build:

```bash
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --locked --release -p bitnet-cli --no-default-features --features cpu,full-cli
```

### Feature Flags

| Flag | Purpose |
|---|---|
| `cpu` | CPU inference and diagnostics. |
| `cuda` | CUDA backend surface. |
| `gpu` | GPU umbrella feature for accelerator backends currently wired through the workspace. |
| `full-cli` | Full CLI command set. |
| `ffi` | C++ FFI bridge for cross-validation. |
| `fixtures` | GGUF fixture-based integration tests. |

Nix: `nix develop && nix build .#bitnet-cli && nix flake check` - see [Nix guide](docs/kv-pool/NIX_FLAKE_USAGE.md).

## Testing

```bash
cargo nextest run --locked --workspace --no-default-features --features cpu
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --no-default-features --features cpu -- -D warnings
```

The repository contains unit, property, snapshot, fixture, fuzz, BDD, receipt, and hardware-specific tests. Some tests are intentionally ignored with justification strings where hardware, model artifacts, or long-running evidence is required. See [docs/development/test-suite.md](docs/development/test-suite.md).

## Documentation

| Section | Contents |
|---|---|
| [docs/tutorials/](docs/tutorials/) | Getting started and first diagnostic runs. |
| [docs/howto/](docs/howto/) | Install, run, export, validate, and cross-check. |
| [docs/explanation/](docs/explanation/) | Architecture and design notes. |
| [docs/reference/](docs/reference/) | CLI, environment variables, quantization, and receipts. |
| [docs/model-artifacts/](docs/model-artifacts/) | Model artifact status and validation. |
| [docs/hardware/](docs/hardware/) | Hardware validation and benchmark protocol. |
| [docs/tracking/](docs/tracking/) | Campaign state and active work. |

## What We Are Working On

Near-term work is focused on:

1. matching the Microsoft BitNet.cpp reference path from Rust CPU
2. preserving the reference runner, tokenizer, pre-tokenizer, and prompt template chain
3. enriching backend-neutral answer diagnostics and first-divergence receipts
4. validating coherent BitNet answer quality against a deterministic corpus
5. validating strict CPU/CUDA answer parity after the Rust CPU path works
6. qualifying throughput after answer quality works

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Before opening a PR:

```bash
./ci/local.sh
```

New internal maintenance commands belong in `xtask`. `bitnet-task` exists only to preserve legacy `scripts/*.sh` entrypoints while that migration is in flight.

See [ROADMAP.md](ROADMAP.md) for project direction.

## License

Dual-licensed under [MIT](LICENSE) and [Apache 2.0](LICENSE).
