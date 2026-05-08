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
> BitNet-rs is not yet a general-purpose local chat engine. The project currently proves loader, tokenizer, kernel, receipt, and hardware-lane behavior under narrow claim boundaries. Coherent BitNet answer quality remains blocked on an `answer_ready` model artifact; until that shared gate passes, BitNet answer runs are diagnostic only.

## Current Boundary

The active blocker is model-artifact authority, not basic backend execution. The shared answer-artifact gate currently records no `answer_ready` BitNet artifact, and the official Microsoft I2_S GGUF is rejected for coherent local-answer claims because the deterministic prompt suite fails under the recorded reference evidence.

Backend receipts can still prove selected-device execution, tokenizer and prompt diagnostics, fallback behavior, and kernel coverage. They cannot prove coherent user answers until an artifact passes [docs/model-artifacts/ANSWER_ARTIFACT_GATE.md](docs/model-artifacts/ANSWER_ARTIFACT_GATE.md).

## What This Repo Is For

BitNet-rs is moving toward Rust-native BitNet inference, but the current repo is best understood as an inference-systems validation workspace. It is useful for contributors working on model loading, tokenization, quantization, kernel parity, hardware bring-up, receipts, and reproducible inference validation. It is not yet a polished end-user inference server.

Current proof surfaces include:

- strict GGUF loading and tokenizer authority checks
- I2_S / QK256 quantization and kernel infrastructure
- scalar, AVX2, AVX-512, NEON, CUDA, OpenCL, OpenVINO, Metal, and NPU proof lanes
- diagnostic answer-corpus and answer-parity receipts
- hardware identity, runtime identity, fallback, kernel, and claim-boundary receipts
- dense SLM companion lanes used to validate the generation pipeline while BitNet artifact authority remains blocked

## Capability Matrix

| Area | State | Current claim |
|---|---|---|
| GGUF loading | Supported / hardening | Structural loading and metadata extraction are active proof surfaces. |
| Tokenizer handling | Supported / hardening | Strict tokenizer authority is required for answer claims. |
| I2_S BitNet32 CPU path | Diagnostic | CPU execution exists; coherent BitNet answer quality is not claimed. |
| I2_S QK256 CPU path | Diagnostic | Scalar, AVX2, and AVX-512 diagnostic lanes are receipt-backed; answer quality depends on the artifact gate. |
| Scalar / SIMD parity | Diagnostic | Used for backend agreement checks and first-divergence evidence, not answer-readiness claims. |
| Dense SLM path | Early working | Companion/control lane for generation-pipeline validation; not a BitNet answer-quality claim. |
| RTX 5070 Ti CUDA | Execution proof complete / diagnostic | Packed BitNet CUDA proof is receipt-backed through `CUDA-BITNET-009`; coherent CUDA answers and speedup claims are still blocked by the answer-artifact gate. |
| Metal / OpenCL / OpenVINO / NPU | Probe / smoke lanes | Hardware identity and narrow execution receipts; no full BitNet answer-readiness claim. |
| Cross-validation | Supported / hardening | Reference comparison infrastructure exists; artifact authority remains gatekeeping. |
| Honest-compute receipts | Supported | Receipts preserve backend, runtime, fallback, kernel, timing, and claim boundaries. |
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
cargo run --locked -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf
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

This exercises the model, tokenizer, generation, and receipt path. It is not a coherent answer-quality claim until the shared answer-artifact gate has an `answer_ready` BitNet artifact.

## Claim Boundaries

Before the answer-artifact gate passes, BitNet-rs may claim:

- structural GGUF loading
- tokenizer and prompt-template diagnostics
- backend execution proof
- scalar/SIMD/CUDA diagnostic receipts
- hardware identity and fallback receipts
- diagnostic-only answer-corpus output

Before the answer-artifact gate passes, BitNet-rs must not claim:

- coherent BitNet local answers
- production inference readiness
- CUDA, Metal, OpenCL, OpenVINO, or NPU answer readiness
- server inference readiness
- speedup claims tied to generated answer quality

See [docs/model-artifacts/ANSWER_ARTIFACT_GATE.md](docs/model-artifacts/ANSWER_ARTIFACT_GATE.md).

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

## Hardware Lanes

Hardware validation is lane-based. Each lane must preserve hardware identity, runtime identity, selected backend identity, fallback status, proof stage, and claim boundary.

| Lane | Role |
|---|---|
| Intel 258V CPU | Lead BitNet CPU reference and AVX2 diagnostic lane. |
| i5-8250U CPU | Dense SLM CPU lead and low-power comparison lane. |
| Ryzen 9950X3D | AVX-512 support validator and high-performance CPU diagnostic lane. |
| RTX 5070 Ti | CUDA packed BitNet proof and answer-productization lane. |
| Apple M4 | Metal, MPSGraph, and CPU/NEON validation lane. |
| Arc A770 | Discrete Intel GPU OpenCL/OpenVINO lane. |
| Arc 140V | Lunar Lake iGPU OpenCL/OpenVINO lane. |
| Intel NPU | OpenVINO NPU static-shape proof lane. |

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

The repository contains unit, property, snapshot, fixture, fuzz, BDD, receipt, and hardware-lane tests. Some tests are intentionally ignored with justification strings where hardware, model artifacts, or long-running evidence is required. See [docs/development/test-suite.md](docs/development/test-suite.md).

## Documentation

| Section | Contents |
|---|---|
| [docs/tutorials/](docs/tutorials/) | Getting started and first diagnostic runs. |
| [docs/howto/](docs/howto/) | Install, run, export, validate, and cross-check. |
| [docs/explanation/](docs/explanation/) | Architecture and design notes. |
| [docs/reference/](docs/reference/) | CLI, environment variables, quantization, and receipts. |
| [docs/model-artifacts/](docs/model-artifacts/) | Answer-artifact gate and model authority. |
| [docs/hardware/](docs/hardware/) | Hardware proof lanes and benchmark protocol. |
| [docs/tracking/](docs/tracking/) | Campaign state and active work lanes. |

## Current Development Focus

Near-term work is focused on:

1. finding or producing an answer-ready BitNet artifact
2. recording reference-runner, tokenizer, pre-tokenizer, and prompt-template authority
3. enriching backend-neutral answer diagnostics and first-divergence receipts
4. proving coherent BitNet answer quality against a deterministic corpus
5. proving strict CPU/CUDA answer parity only after artifact authority passes
6. qualifying throughput only after answer quality is green

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Before opening a PR:

```bash
./ci/local.sh
```

New internal maintenance commands belong in `xtask`. `bitnet-task` exists only to preserve legacy `scripts/*.sh` entrypoints while that migration is in flight.

See [ROADMAP.md](ROADMAP.md) for project direction.

## License

Dual-licensed under [MIT](LICENSE) and [Apache 2.0](LICENSE).
