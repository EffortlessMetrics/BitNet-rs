# BitNet-rs Documentation

BitNet-rs is a pre-alpha Rust inference engine for BitNet-style 1-bit
language models. CPU inference is the primary validated path; GPU and
accelerator backends are documented as scaffolded or validation work unless a
platform page explicitly says otherwise.

Documentation is organized using the [Diátaxis](https://diataxis.fr/) framework:
tutorials for learning, how-to guides for tasks, explanation for design
context, and reference pages for exact behavior.

## Start here

| If you want to... | Read this first | Then check |
|---|---|---|
| Build the project locally | [development/build-commands.md](development/build-commands.md) | [development/test-suite.md](development/test-suite.md) |
| Run the CLI against a GGUF model | [tutorials/first-inference.md](tutorials/first-inference.md) | [howto/gguf-model-validation-and-loading.md](howto/gguf-model-validation-and-loading.md) |
| Understand the repository layout | [architecture-overview.md](architecture-overview.md) | [development/REPO_SURFACES.md](development/REPO_SURFACES.md) |
| Validate a model artifact | [model-artifacts/MODEL_COVERAGE_MATRIX.md](model-artifacts/MODEL_COVERAGE_MATRIX.md) | [howto/validate-models.md](howto/validate-models.md) |
| Debug tokenizer or answer-quality drift | [howto/troubleshoot-intelligibility.md](howto/troubleshoot-intelligibility.md) | [howto/parity-playbook.md](howto/parity-playbook.md) |
| Check hardware support status | [hardware/HARDWARE_MATRIX.md](hardware/HARDWARE_MATRIX.md) | [hardware/PROOF_STAGES.md](hardware/PROOF_STAGES.md) |

## Command conventions

Unless a page says otherwise, commands assume they are run from the repository
root. Default Cargo features are intentionally empty, so examples spell out the
runtime feature set explicitly:

```bash
cargo build --locked --no-default-features --features cpu
cargo test --locked --workspace --no-default-features --features cpu
```

Use `cpu,full-cli` when invoking CLI subcommands that are not part of the
minimal library surface:

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- --help
```

GPU examples document wiring and validation work. Treat them as platform notes
until the relevant hardware page records proof-stage evidence for your backend.

## Documentation maintenance checklist

When adding or updating docs, keep the following contract in sync:

1. Link new user-facing pages from this index or the nearest section index.
2. Mark diagnostic, scaffolded, or hardware-specific flows explicitly.
3. Prefer copy-pasteable commands with `--locked`, `--no-default-features`, and
   the intended feature list.
4. Put model, tokenizer, and receipt paths in examples so validation artifacts
   are easy to reproduce.
5. Move stale sprint notes and superseded plans into [`archive/`](archive/)
   instead of leaving them in the main navigation path.

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
| [api/README.md](api/README.md) | Generated API snapshots and contract baselines |
| [bitnet/BITNET_CPU_PATH_PLAN.md](bitnet/BITNET_CPU_PATH_PLAN.md) | CPU GGUF/tokenizer/layout/kernel roadmap and strict receipt contract |
| [specs/intel-lunar-lake-258v-buildout-plan.md](specs/intel-lunar-lake-258v-buildout-plan.md) | Lunar Lake 258V CPU validation, Arc 140V, NPU identity, platform probe, and receipt buildout plan |

---

## Development

| Document | Purpose |
|----------|---------|
| [development/build-commands.md](development/build-commands.md) | Build matrix and cargo commands |
| [development/CRATE_BOUNDARY_POLICY.md](development/CRATE_BOUNDARY_POLICY.md) | Rules for deciding when a design seam deserves a Cargo package boundary |
| [development/REPO_SURFACES.md](development/REPO_SURFACES.md) | Target public crate surface, internal module-family map, and collapse waves |
| [development/test-suite.md](development/test-suite.md) | Test organization and CI lanes |
| [development/gpu-development.md](development/gpu-development.md) | CUDA development guide |
| [development/validation-framework.md](development/validation-framework.md) | Quality assurance pipeline |
| [development/xtask.md](development/xtask.md) | Developer tooling reference |
| [performance-benchmarking.md](performance-benchmarking.md) | Benchmarking setup and baselines |

---

## Archive

Historical sprint notes, issue analysis documents, and implementation plans are preserved in [`archive/`](archive/) but are not maintained.
