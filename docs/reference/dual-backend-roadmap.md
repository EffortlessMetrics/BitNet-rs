# Dual-Backend Support Implementation Roadmap

> **Last updated**: reflects implementation state after PRs #608–#689.
> Items marked ✅ are **done**; items marked 🔲 are **planned**.

---

## Implementation Status

### ✅ What's Implemented

| Capability | Location | PR |
|---|---|---|
| Library discovery (bitnet.cpp + llama.cpp) | `crossval/build.rs`, `crates/bitnet-sys/` | #608 |
| Feature lattice: `gpu` umbrella, `cuda = ["gpu"]` backward-compat alias | workspace `Cargo.toml` | #611 |
| Orthogonal runtime reporting: `gpu`/`cuda` never conflated | `crates/bitnet-runtime-feature-flags/src/lib.rs` | #611 |
| Kernel capability registry (single source of truth) | `crates/bitnet-common/src/kernel_registry.rs` | #611 |
| `xtask analyze-library` symbol inspection command | `xtask/src/analyze_library.rs` | #611 |
| `bitnet-device-probe` SRP microcrate | `crates/bitnet-device-probe/` | #629 |
| `bitnet-logits` SRP microcrate | `crates/bitnet-logits/` | #629 |
| `bitnet-generation` SRP microcrate | `crates/bitnet-generation/` | #629 |
| `bitnet-engine-core` SRP microcrate | `crates/bitnet-engine-core/` | #629 |
| `bitnet-gguf` SRP microcrate | `crates/bitnet-gguf/` | #629 |
| `KernelBackend`, `KernelCapabilities`, `SimdLevel` types | `crates/bitnet-common/src/kernel_registry.rs` | #611 |
| CodeRabbit path exclusions (archive, reports) | `.coderabbit.yaml` | #611 |
| GPU smoke workflow (compile-only PR lane) | `.github/workflows/gpu-smoke.yml` | #611 |
| Preflight backend availability checks | `xtask/` | #611 |
| `BITNET_GPU_FAKE`, `BITNET_STRICT_MODE` env guards | `crates/bitnet-device-probe/` | #609 |
| TL1/TL2 quantizer round-trip accuracy fix | `crates/bitnet-quantization/` | #641 |
| Runtime backend validation + enforcement wired into CLI/server | `crates/bitnet-cli/`, `crates/bitnet-server/` | #642 |
| `BackendCapabilities` snapshot: `requested=X detected=[…] selected=Y` | `crates/bitnet-kernels/src/device_features.rs` | #642 |
| CPU golden path E2E tests (5 deterministic, no model download) | `crates/bitnet-inference/tests/cpu_golden_path.rs` | #643 |
| SRP microcrates wired into CI matrix | `.github/workflows/ci-core.yml` | #644 |
| Security audit fixes (bytes, time CVEs; audit.toml accepted-risk) | `Cargo.lock`, `.cargo/audit.toml` | #645 |
| Nightly fuzz runs with artifact upload | `.github/workflows/fuzz-ci.yml` | #609 |
| Cross-validation scheduled lanes | `.github/workflows/crossval.yml` | #611 |
| Build-time symbol detection → rustc-cfg flags | `crates/bitnet-sys/build.rs` (feature: `symbol-analysis`) | #611 |
| Proptest for `bitnet-gguf` and `bitnet-generation` | `crates/bitnet-gguf/src/lib.rs`, `crates/bitnet-generation/src/lib.rs` | #649 |
| GGUF header libfuzzer fuzz target | `fuzz/fuzz_targets/gguf_header.rs` | #649 |
| Fuzz target registration completeness (all 11 targets in Cargo.toml + CI) | `fuzz/Cargo.toml`, `fuzz-ci.yml` | #649 |
| Proptest for `bitnet-device-probe` and `bitnet-engine-core` | `crates/bitnet-device-probe/src/lib.rs`, `crates/bitnet-engine-core/src/lib.rs` | #650 |
| Insta snapshot tests for 10 SRP microcrates (45 snapshots) | `crates/*/tests/snapshot_tests.rs` | #651 |
| AC6 hash stubs implemented; TC3/TC4/tc_ac6 unblocked (88 tests pass, 6 ignored) | `crossval/tests/tokenizer_authority_tests.rs`, `crossval/tests/fixtures/` | #653 |
| AC5 source detection stubs implemented; 2 remaining fixture tests unblocked | `crossval/tests/tokenizer_authority_tests.rs` | #654 |
| `Tokenizer::get_family_name()` trait method (llama3 detection) | `crates/bitnet-tokenizers/src/lib.rs` | #673 |
| Template detection + KV cache init tests unblocked | `crates/bitnet-inference/tests/template_detection.rs`, `kv_cache_validation.rs` | #673 |
| QK256 tolerance, CLI snapshot, simple_real_inference tests unblocked | `crates/bitnet-quantization/`, `crates/bitnet-cli/` | #674 |
| Receipts property test: mock-containing IDs excluded from strategy | `crates/bitnet-receipts/src/lib.rs` | #675 |
| Issue #159 resolved: `MockGgufFileBuilder` uses real `GgufWriter` | `crates/bitnet-models/tests/gguf_weight_loading_tests.rs` | #676 |
| Env-var race conditions eliminated: `serial+temp_env` across bitnet-kernels, bitnet-models, bitnet-trace, bitnet-runtime-profile-contract-core | `crates/bitnet-kernels/`, `crates/bitnet-models/`, `crates/bitnet-trace/`, `crates/bitnet-runtime-profile-contract-core/` | #678 |
| Flaky SIMD throughput test removed | `crates/bitnet-kernels/tests/` | #681 |
| Clippy warnings resolved workspace-wide (zero-warning CPU policy) | workspace | #682 |
| Property tests for `bitnet-logits` (13 properties) | `crates/bitnet-logits/tests/property_tests.rs` | #683 |
| Property tests for `bitnet-generation` (8 tests) | `crates/bitnet-generation/tests/property_tests.rs` | #683 |
| Property tests for `bitnet-engine-core` standalone suite (7 tests) | `crates/bitnet-engine-core/tests/property_tests.rs` | #683 |
| `ci-core.yml` paths filter includes doc-only files (`CLAUDE.md`, `CHANGELOG.md`, etc.) | `.github/workflows/ci-core.yml` | #684 |
| Tracing instrumentation in `TemplateType::detect()`; `#[traced_test]` log-capture unit tests | `crates/bitnet-prompt-templates/src/lib.rs` | #686 |
| 4 previously-ignored tests unblocked (kv_cache unique-layer-index, template_detection behavioral) | `crates/bitnet-inference/tests/` | #686 |
| `tracing-test 0.2.6` added to workspace dependencies | `Cargo.toml` | #686 |
| Fixture timeout fixed: >300s → ~1s (50k-vocab allocations replaced with 512-dim) | `crates/bitnet-quantization/tests/fixtures/models/qlinear_layer_data.rs` | #687 |
| Property tests for `bitnet-transformer` KVCache invariants (4 proptest properties) | `crates/bitnet-transformer/tests/property_tests.rs` | #689 |

### 🔲 What's Planned

1. **CUDA smoke lane** (self-hosted runner, nightly/manual)
   - Allocate device, run small inference, upload parity receipt
   - Blocked on having a CUDA-capable self-hosted runner

2. **Scheduled fuzz/crossval evidence expansion**
   - Nightly timeboxed fuzz runs with artifact upload (partially done via `fuzz-ci.yml`)
   - Real-model crossval producing receipts (gated on model download infrastructure)

3. **GgufTokenizer `real_vocab_size()` method**
   - Distinguish real vs padded vocab sizes in `bitnet-tokenizers`
   - Required for tokenizer fixture tests in `tokenizer_vocab_size.rs`

4. **Log-capture test infrastructure**
   - `tracing-test` integration for testing `warn!`/`debug!` emission in hot paths
   - Required for kv_cache_validation and qk256_tolerance log-format tests

---

## Feature Lattice Design

### Current (CUDA-first, non-CUDA-ready)

```toml
[features]
gpu = ["kernels", "inference", "tokenizers", "bitnet-kernels/gpu", ...]
cuda = ["gpu"]      # backward-compat alias — CUDA is the only GPU backend today
```

### Code Gating Rules

```rust
// CUDA-specific imports:
#[cfg(feature = "cuda")]
use cudarc::...;

// Generic "any GPU compiled":
#[cfg(feature = "gpu")]
pub fn gpu_dispatch() { ... }

use bitnet_device_probe::{gpu_compiled, gpu_available_runtime};
```

### Adding ROCm / Metal (future, additive only)

Adding a new GPU backend only requires new feature entries and new
backend-specific modules. Existing `#[cfg(feature = "gpu")]` code
continues to work without modification.

---

## Known Issues

| Issue | Affected Component | Status |
|---|---|---|
| Shape mismatch in layer-norm (#254) | `bitnet-inference` | In analysis |
| Mock elimination (#260) | `bitnet-inference` | Pending refactor |
| Tokenizer parity + FFI hygiene (#469) | `bitnet-tokenizers`, `crossval` | Active development |

---

## Related Documentation

- `docs/architecture-overview.md` — overall system design and crate relationships
- `docs/howto/cpp-setup.md` — building the C++ reference for cross-validation
- `docs/explanation/i2s-dual-flavor.md` — I2S QK256 vs BitNet32 format details
- `crates/bitnet-common/src/kernel_registry.rs` — canonical kernel/backend type definitions
