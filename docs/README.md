# BitNet-rs Documentation

BitNet-rs is a Rust implementation of BitNet — Microsoft's 1-bit weight neural network inference framework.

Documentation is organized using the [Diátaxis](https://diataxis.fr/) framework.

---

## [Tutorials](tutorials/) — learning by doing

Step-by-step guides for getting started.

- [Getting started](getting-started.md) — install, run your first model
- [Your first inference](tutorials/first-inference.md) — load a GGUF and generate tokens
- [Real GGUF model inference](tutorials/real-gguf-model-inference.md) — end-to-end inference walkthrough
- [Tokenizer auto-discovery](tutorials/tokenizer-auto-discovery.md) — automatic tokenizer detection

---

## [How-to guides](howto/) — solve specific problems

Task-oriented. These assume you know what you want to do.

| Guide | Purpose |
|-------|---------|
| [cpp-setup.md](howto/cpp-setup.md) | Set up C++ cross-validation reference |
| [export-clean-gguf.md](howto/export-clean-gguf.md) | Export safe clean GGUF from SafeTensors |
| [validate-models.md](howto/validate-models.md) | Run 3-stage model validation |
| [use-qk256-models.md](howto/use-qk256-models.md) | Load and run QK256 format models |
| [parity-playbook.md](howto/parity-playbook.md) | Verify Rust vs C++ numeric parity |
| [troubleshoot-intelligibility.md](howto/troubleshoot-intelligibility.md) | Debug incoherent model output |
| [deterministic-inference-setup.md](howto/deterministic-inference-setup.md) | Set up reproducible inference |
| [receipt-verification.md](howto/receipt-verification.md) | Verify inference receipts |
| [strict-mode-validation-workflows.md](howto/strict-mode-validation-workflows.md) | Use strict validation in CI |
| [automatic-tokenizer-discovery.md](howto/automatic-tokenizer-discovery.md) | Configure tokenizer auto-detection |
| [quantization-optimization-and-performance.md](howto/quantization-optimization-and-performance.md) | Optimize quantization performance |

---

## [Explanation](explanation/) — background and concepts

Understanding-oriented. These explain *why* things work the way they do.

| Topic | Description |
|-------|-------------|
| [adr/README.md](adr/README.md) | Architectural Decision Records |
| [architecture-overview.md](architecture-overview.md) | System components and design principles |
| [explanation/FEATURES.md](explanation/FEATURES.md) | Feature flag system |
| [explanation/dual-backend-crossval.md](explanation/dual-backend-crossval.md) | Dual-backend cross-validation design |
| [explanation/i2s-dual-flavor.md](explanation/i2s-dual-flavor.md) | I2_S quantization flavor auto-detection |
| [explanation/correction-policy.md](explanation/correction-policy.md) | Model-specific correction policies |
| [explanation/cpu-inference-architecture.md](explanation/cpu-inference-architecture.md) | CPU inference pipeline |
| [explanation/device-feature-detection.md](explanation/device-feature-detection.md) | Runtime device/capability detection |
| [explanation/backend-detection-and-device-selection-patterns.md](explanation/backend-detection-and-device-selection-patterns.md) | Backend selection patterns |
| [gpu-kernel-architecture.md](gpu-kernel-architecture.md) | CUDA kernel design |
| [tokenizer-architecture.md](tokenizer-architecture.md) | Universal tokenizer system |

---

## [Reference](reference/) — technical specifications

Information-oriented. Look up exact behaviors, formats, and APIs.

| Document | Contents |
|----------|---------|
| [reference/quantization-support.md](reference/quantization-support.md) | All supported quantization formats |
| [reference/validation-gates.md](reference/validation-gates.md) | Validation system gates and thresholds |
| [environment-variables.md](environment-variables.md) | All runtime configuration env vars |
| [reference/api-reference.md](reference/api-reference.md) | Public API contracts |
| [reference/strict-mode-api.md](reference/strict-mode-api.md) | Strict mode behavior |

---

## Development

| Document | Purpose |
|----------|---------|
| [development/build-commands.md](development/build-commands.md) | Build matrix and cargo commands |
| [development/test-suite.md](development/test-suite.md) | Test organization and CI lanes |
| [development/gpu-development.md](development/gpu-development.md) | CUDA development guide |
| [development/validation-framework.md](development/validation-framework.md) | Quality assurance pipeline |
| [development/xtask.md](development/xtask.md) | Developer tooling reference |
| [performance-benchmarking.md](performance-benchmarking.md) | Benchmarking setup and baselines |

---

## Archive

Historical sprint notes, issue analysis documents, and implementation plans are preserved in [`archive/`](archive/) but are not maintained.
