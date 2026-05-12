# BitNet-rs Documentation

BitNet-rs is a pre-alpha Rust-native local model runtime and validation
workspace for efficient language models, including dense SLMs and BitNet / 1-bit
model families. The documentation follows the [Diátaxis](https://diataxis.fr/)
structure so readers can choose learning-oriented tutorials, task-oriented
how-to guides, explanation, or exact reference material.

> [!IMPORTANT]
> Generated BitNet text is still a diagnostic surface, not a supported
> answer-quality claim. Treat receipts, backend identity, tokenizer checks, and
> parity evidence as validation artifacts; they do not by themselves prove that a
> generated answer is production-ready.

## Choose Your Path

| If you want to... | Start with... | Then read... |
|---|---|---|
| Understand the project status | [Repository README](../README.md) | [Answer artifact gate](model-artifacts/ANSWER_ARTIFACT_GATE.md) |
| Run the first diagnostic generation path | [First inference tutorial](tutorials/first-inference.md) | [CLI reference](reference/inference-cli-reference.md) |
| Validate a real model artifact | [Real GGUF model inference](tutorials/real-gguf-model-inference.md) | [Model validation workflow](howto/validate-models.md) |
| Debug tokenizer or output quality issues | [Tokenizer auto-discovery](tutorials/tokenizer-auto-discovery.md) | [Intelligibility troubleshooting](howto/troubleshoot-intelligibility.md) |
| Work on Rust code | [Build commands](development/build-commands.md) | [Test suite guide](development/test-suite.md) |
| Inspect hardware/backend claims | [Hardware matrix](hardware/HARDWARE_MATRIX.md) | [Benchmark protocol](hardware/BENCHMARK_PROTOCOL.md) |
| Check active campaign state | [Tracking docs](tracking/) | [Generated tracking state](tracking/generated/) |

## Active Documentation Areas

### Tutorials — learning by doing

Step-by-step guides for first-time or infrequent workflows.

- [Getting started](getting-started.md) — install prerequisites and orient around the repo.
- [Your first inference](tutorials/first-inference.md) — load a GGUF and generate diagnostic tokens.
- [Real GGUF model inference](tutorials/real-gguf-model-inference.md) — end-to-end model walkthrough.
- [Tokenizer auto-discovery](tutorials/tokenizer-auto-discovery.md) — automatic tokenizer detection.

### How-to guides — solve a specific problem

Task-oriented docs for contributors who already know their goal.

| Guide | Purpose |
|---|---|
| [C++ setup](howto/cpp-setup.md) | Set up the C++ cross-validation reference. |
| [Export clean GGUF](howto/export-clean-gguf.md) | Export a safe clean GGUF from SafeTensors. |
| [Validate models](howto/validate-models.md) | Run model validation stages and interpret failures. |
| [Use QK256 models](howto/use-qk256-models.md) | Load and run QK256-format models. |
| [Parity playbook](howto/parity-playbook.md) | Verify Rust vs. reference numeric parity. |
| [Troubleshoot intelligibility](howto/troubleshoot-intelligibility.md) | Debug incoherent model output. |
| [Deterministic inference setup](howto/deterministic-inference-setup.md) | Configure reproducible inference runs. |
| [Receipt verification](howto/receipt-verification.md) | Inspect and verify inference receipts. |
| [Strict-mode workflows](howto/strict-mode-validation-workflows.md) | Use strict validation in local and CI workflows. |
| [Automatic tokenizer discovery](howto/automatic-tokenizer-discovery.md) | Configure tokenizer auto-detection. |
| [Quantization optimization](howto/quantization-optimization-and-performance.md) | Tune quantization performance and diagnostics. |

### Explanation — background and design context

Conceptual material for understanding why the system is structured as it is.

| Topic | Description |
|---|---|
| [Architecture overview](architecture-overview.md) | System components and design principles. |
| [Architectural decision records](adr/README.md) | Accepted design decisions and rationale. |
| [Feature flags](explanation/FEATURES.md) | Feature flag strategy and backend predicates. |
| [Dual-backend cross-validation](explanation/dual-backend-crossval.md) | Reference comparison design. |
| [I2_S dual flavor detection](explanation/i2s-dual-flavor.md) | I2_S quantization flavor auto-detection. |
| [Correction policy](explanation/correction-policy.md) | Model-specific correction policies. |
| [CPU inference architecture](explanation/cpu-inference-architecture.md) | CPU inference pipeline. |
| [Device feature detection](explanation/device-feature-detection.md) | Runtime device and capability detection. |
| [Backend selection patterns](explanation/backend-detection-and-device-selection-patterns.md) | Backend selection and fallback patterns. |
| [GPU kernel architecture](gpu-kernel-architecture.md) | CUDA kernel design notes. |
| [Tokenizer architecture](tokenizer-architecture.md) | Universal tokenizer system. |

### Reference — exact commands, schemas, and contracts

Information-oriented pages for looking up behavior rather than learning a flow.

| Document | Contents |
|---|---|
| [Inference CLI reference](reference/inference-cli-reference.md) | CLI flags and receipt options. |
| [Quantization support](reference/quantization-support.md) | Supported quantization formats. |
| [Validation gates](reference/validation-gates.md) | Validation gates and thresholds. |
| [Environment variables](environment-variables.md) | Runtime configuration environment variables. |
| [API reference](reference/api-reference.md) | Public API contracts. |
| [Strict-mode API](reference/strict-mode-api.md) | Strict-mode behavior. |
| [API snapshots](api/README.md) | Generated API snapshots and contract baselines. |
| [BitNet CPU path plan](bitnet/BITNET_CPU_PATH_PLAN.md) | CPU GGUF/tokenizer/layout/kernel roadmap and strict receipt contract. |
| [Lunar Lake 258V buildout plan](specs/intel-lunar-lake-258v-buildout-plan.md) | Intel 258V, Arc 140V, NPU, platform probe, and receipt plan. |

### Development

Contributor-oriented commands, policies, and validation references.

| Document | Purpose |
|---|---|
| [Build commands](development/build-commands.md) | Build matrix and Cargo commands. |
| [Crate boundary policy](development/CRATE_BOUNDARY_POLICY.md) | Rules for Cargo package boundaries. |
| [Repository surfaces](development/REPO_SURFACES.md) | Public crate surface and internal module-family map. |
| [Test suite](development/test-suite.md) | Test organization and CI lanes. |
| [GPU development](development/gpu-development.md) | CUDA development guide. |
| [Validation framework](development/validation-framework.md) | Quality assurance pipeline. |
| [xtask reference](development/xtask.md) | Developer tooling reference. |
| [Performance benchmarking](performance-benchmarking.md) | Benchmark setup and baselines. |

## Archive Policy

Historical sprint notes, issue analysis documents, and implementation plans are
preserved in [`docs/archive/`](archive/) and [`archive/`](../archive/) for
provenance. They are not maintained as current guidance and may refer to older
MSRVs, status claims, command names, or acceptance criteria. When an archived
page conflicts with an active page, prefer the active page.
