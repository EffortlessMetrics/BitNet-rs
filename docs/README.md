# BitNet-rs Documentation

BitNet-rs is a pre-alpha Rust-native local model runtime and validation
workspace for small language models, including BitNet / 1-bit model families and
dense SLMs. The documentation is organized with the
[Diátaxis](https://diataxis.fr/) model so readers can choose between tutorials,
task guides, conceptual explanations, and reference material.

> [!IMPORTANT]
> BitNet answer quality is still being validated. Treat `run` and `chat` output
> from BitNet artifacts as diagnostic evidence unless the relevant model,
> tokenizer, backend, and receipt gates say otherwise. Dense SLM paths can be
> valid local-answer lanes when their artifact-specific gates pass.

## Start Here

| If you want to... | Read this first | Then use |
| --- | --- | --- |
| Install the workspace and run a smoke check | [Getting started](getting-started.md) | [First inference](tutorials/first-inference.md) |
| Run an official GGUF through the CLI | [Real GGUF model inference](tutorials/real-gguf-model-inference.md) | [Inference CLI reference](reference/inference-cli-reference.md) |
| Validate whether a model artifact is acceptable | [Validate models](howto/validate-models.md) | [Answer artifact gate](model-artifacts/ANSWER_ARTIFACT_GATE.md) |
| Debug incoherent generated text | [Troubleshoot intelligibility](howto/troubleshoot-intelligibility.md) | [Parity playbook](howto/parity-playbook.md) |
| Work on kernels, quantization, or backend parity | [Architecture overview](architecture-overview.md) | [Quantization support](reference/quantization-support.md) |
| Bring up or verify hardware | [Hardware matrix](hardware/HARDWARE_MATRIX.md) | [Benchmark protocol](hardware/BENCHMARK_PROTOCOL.md) |
| Add code or run local validation | [Build commands](development/build-commands.md) | [Test suite](development/test-suite.md) |
| Understand badge and PR evidence boundaries | [Verification](VERIFICATION.md) | [RIPR evidence policy](RIPR_EVIDENCE_POLICY.md) |
| Understand project direction and proof sequencing | [Roadmap](../ROADMAP.md) | [Campaign trackers](tracking/campaigns/README.md) |

## Documentation Map

### [Tutorials](tutorials/) — learning by doing

Step-by-step paths for first-time or infrequent workflows.

- [Getting started](getting-started.md) — install prerequisites and run the basic project checks
- [Your first inference](tutorials/first-inference.md) — load a GGUF and generate diagnostic tokens
- [Real GGUF model inference](tutorials/real-gguf-model-inference.md) — end-to-end model walkthrough
- [Tokenizer auto-discovery](tutorials/tokenizer-auto-discovery.md) — automatic tokenizer detection

### [How-to guides](howto/) — solve specific problems

Task-oriented instructions for common contributor jobs.

| Guide | Purpose |
| --- | --- |
| [cpp-setup.md](howto/cpp-setup.md) | Set up the C++ cross-validation reference path. |
| [export-clean-gguf.md](howto/export-clean-gguf.md) | Export a safe clean GGUF from SafeTensors. |
| [validate-models.md](howto/validate-models.md) | Run staged model validation. |
| [gguf-model-validation-and-loading.md](howto/gguf-model-validation-and-loading.md) | Inspect GGUF metadata and loader behavior. |
| [use-qk256-models.md](howto/use-qk256-models.md) | Load and run QK256-format models. |
| [parity-playbook.md](howto/parity-playbook.md) | Verify Rust vs. reference numeric parity. |
| [troubleshoot-intelligibility.md](howto/troubleshoot-intelligibility.md) | Debug incoherent or suspicious model output. |
| [deterministic-inference-setup.md](howto/deterministic-inference-setup.md) | Set up reproducible inference runs. |
| [receipt-verification.md](howto/receipt-verification.md) | Verify inference receipts and evidence files. |
| [strict-mode-validation-workflows.md](howto/strict-mode-validation-workflows.md) | Use strict validation in local and CI lanes. |
| [automatic-tokenizer-discovery.md](howto/automatic-tokenizer-discovery.md) | Configure tokenizer auto-detection. |
| [quantization-optimization-and-performance.md](howto/quantization-optimization-and-performance.md) | Optimize quantization performance. |

### [Explanation](explanation/) — background and concepts

Understanding-oriented material that explains design choices and system shape.

| Topic | Description |
| --- | --- |
| [adr/README.md](adr/README.md) | Architectural Decision Records. |
| [architecture-overview.md](architecture-overview.md) | System components and design principles. |
| [explanation/FEATURES.md](explanation/FEATURES.md) | Feature flag model and expected cargo invocations. |
| [explanation/dual-backend-crossval.md](explanation/dual-backend-crossval.md) | Dual-backend cross-validation design. |
| [explanation/i2s-dual-flavor.md](explanation/i2s-dual-flavor.md) | I2_S quantization flavor auto-detection. |
| [explanation/correction-policy.md](explanation/correction-policy.md) | Model-specific correction policies. |
| [explanation/cpu-inference-architecture.md](explanation/cpu-inference-architecture.md) | CPU inference pipeline. |
| [explanation/device-feature-detection.md](explanation/device-feature-detection.md) | Runtime device and capability detection. |
| [explanation/backend-detection-and-device-selection-patterns.md](explanation/backend-detection-and-device-selection-patterns.md) | Backend selection patterns. |
| [gpu-kernel-architecture.md](gpu-kernel-architecture.md) | CUDA kernel design. |
| [tokenizer-architecture.md](tokenizer-architecture.md) | Universal tokenizer system. |

### [Reference](reference/) — technical specifications

Lookup material for exact behavior, formats, and APIs.

| Document | Contents |
| --- | --- |
| [reference/quantization-support.md](reference/quantization-support.md) | Supported quantization formats. |
| [reference/validation-gates.md](reference/validation-gates.md) | Validation gates and thresholds. |
| [environment-variables.md](environment-variables.md) | Runtime configuration environment variables. |
| [reference/api-reference.md](reference/api-reference.md) | Public API contracts. |
| [reference/inference-cli-reference.md](reference/inference-cli-reference.md) | CLI flags, generation options, and receipt paths. |
| [reference/strict-mode-api.md](reference/strict-mode-api.md) | Strict mode behavior. |
| [api/README.md](api/README.md) | Generated API snapshots and contract baselines. |
| [bitnet/BITNET_CPU_PATH_PLAN.md](bitnet/BITNET_CPU_PATH_PLAN.md) | CPU GGUF/tokenizer/layout/kernel roadmap and strict receipt contract. |
| [specs/intel-lunar-lake-258v-buildout-plan.md](specs/intel-lunar-lake-258v-buildout-plan.md) | Lunar Lake 258V CPU, Arc 140V, NPU, probe, and receipt buildout plan. |

### Hardware and model evidence

| Area | Entry point |
| --- | --- |
| Hardware platform status | [hardware/HARDWARE_MATRIX.md](hardware/HARDWARE_MATRIX.md) |
| Proof stages and benchmark rules | [hardware/PROOF_STAGES.md](hardware/PROOF_STAGES.md), [hardware/BENCHMARK_PROTOCOL.md](hardware/BENCHMARK_PROTOCOL.md) |
| Model artifact status | [model-artifacts/MODEL_COVERAGE_MATRIX.md](model-artifacts/MODEL_COVERAGE_MATRIX.md) |
| Answer quality gate | [model-artifacts/ANSWER_ARTIFACT_GATE.md](model-artifacts/ANSWER_ARTIFACT_GATE.md) |

### Source-of-truth and claim boundaries

| Area | Entry point |
| --- | --- |
| Why a proof lane exists | [proposals/README.md](proposals/README.md) |
| What must be true | [specs/README.md](specs/README.md) |
| Durable decisions | [adr/README.md](adr/README.md) |
| User-facing status and claim tiers | [status/README.md](status/README.md) |
| Active campaign work state | [tracking/TRACKER_MODEL.md](tracking/TRACKER_MODEL.md) |
| Roadmap | [../ROADMAP.md](../ROADMAP.md) |
| Proof-convergence plan | [../plans/proof-convergence/README.md](../plans/proof-convergence/README.md) |

### Development

| Document | Purpose |
| --- | --- |
| [development/build-commands.md](development/build-commands.md) | Build matrix and cargo commands. |
| [development/CRATE_BOUNDARY_POLICY.md](development/CRATE_BOUNDARY_POLICY.md) | Rules for deciding when a seam deserves a Cargo package boundary. |
| [development/REPO_SURFACES.md](development/REPO_SURFACES.md) | Target public crate surface and internal module-family map. |
| [development/test-suite.md](development/test-suite.md) | Test organization and CI lanes. |
| [development/gpu-development.md](development/gpu-development.md) | CUDA development guide. |
| [development/validation-framework.md](development/validation-framework.md) | Quality assurance pipeline. |
| [development/xtask.md](development/xtask.md) | Developer tooling reference. |
| [performance-benchmarking.md](performance-benchmarking.md) | Benchmarking setup and baselines. |

## Status Vocabulary

Documentation uses these words consistently:

- **Supported** means the path has an explicit contract and validation lane.
- **Diagnostic** means the path is useful for evidence collection but is not an
  answer-quality claim.
- **Probe / smoke** means the path verifies presence, identity, or a narrow
  execution slice only.
- **Scaffolded** means code or docs exist, but full validation is not complete.

When a page appears to disagree with an artifact gate, hardware matrix, or
receipt policy, prefer the gate/matrix/policy and update the stale page.

## Archive

Historical sprint notes, issue analysis documents, and implementation plans are
preserved in [`archive/`](archive/) but are not maintained. Use archive pages as
background only; do not treat them as current status unless a maintained doc
links to them for a specific reason.
