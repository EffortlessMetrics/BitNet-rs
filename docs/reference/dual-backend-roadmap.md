# Dual-Backend Support Implementation Roadmap

> **Last updated**: reflects implementation state after PRs #608–#733.
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
| Env-var race fixed in `bitnet-startup-contract-core`; 5 proptest properties added | `crates/bitnet-startup-contract-core/src/lib.rs`, `tests/property_tests.rs` | #691 |
| 6 proptest properties for `bitnet-runtime-feature-flags-core` (cuda⇒gpu, cpu⇒inference+kernels+tokenizers, feature_line prefix) | `crates/bitnet-runtime-feature-flags-core/tests/property_tests.rs` | #692 |
| 10 proptest properties for `bitnet-bdd-grid-core` (Display↔FromStr round-trips for all 3 enums; FeatureSet invariants) | `crates/bitnet-bdd-grid-core/tests/property_tests.rs` | #694 |
| Integration proptest coverage for 6 SRP microcrates: policy-core (7), scenarios-core (8), honest-compute (16), rope (11), validation (10), feature-contract (8) = ~45 new tests | `crates/*/tests/property_tests.rs` | #696 |
| Backend selection property + snapshot tests (13 tests, no feature gate): `BackendRequest::Display`, `summary()` format, `select_backend()` determinism, 2 insta snapshots | `crates/bitnet-common/tests/backend_selection_tests.rs` | #698 |
| GGUF `open()` file integration tests using 224-byte `mini.gguf` fixture (4 tests): happy path, snapshot, nonexistent error, bad-magic error | `crates/bitnet-gguf/tests/file_open_tests.rs` | #699 |
| `bitnet-device-probe` proptest integration tests (7 proptest): SIMD determinism, gpu_compiled stability, DeviceCapabilities invariants, BITNET_GPU_FAKE / BITNET_STRICT_MODE env-var semantics | `crates/bitnet-device-probe/tests/property_tests.rs` | #701 |
| `bitnet-transformer` KVCache snapshot tests (5 insta snapshots): initial state, post-append, post-clear, layer-initial-state, 3-append seq_len accumulation | `crates/bitnet-transformer/tests/snapshot_tests.rs` | #702 |
| `bitnet-receipts` snapshot tests (6 JSON snapshots) + `bitnet-sampling` snapshot tests (6 debug snapshots) + `bitnet-prompt-templates` snapshot tests (7 snaps): all template types + multi-turn history | `crates/bitnet-receipts/tests/snapshot_tests.rs`, `crates/bitnet-sampling/tests/snapshot_tests.rs`, `crates/bitnet-prompt-templates/tests/snapshot_tests.rs` | #703 |
| `bitnet-bdd-grid` proptest + snapshot tests (8 proptest: LazyLock stability, required∩forbidden disjoint, find/rows consistency, supports/violations semantics; 4 snapshots: grid summary, Unit/Local cell, EndToEnd/CI cell, cell count) | `crates/bitnet-bdd-grid/tests/property_tests.rs`, `crates/bitnet-bdd-grid/tests/snapshot_tests.rs` | #705 |
| `bitnet-trace` snapshot tests (4 JSON: minimal, optional-fields, decode-step, logits) + `bitnet-compat` snapshot tests (2: diagnose messages + count) + `bitnet-bdd-grid-core` snapshot tests (6: Scenario/Env/Feature Display strings, FeatureSet labels, BddCell debug, BddGrid::validate None) | `crates/bitnet-trace/tests/snapshot_tests.rs`, `crates/bitnet-compat/tests/snapshot_tests.rs`, `crates/bitnet-bdd-grid-core/tests/snapshot_tests.rs` | #706 |
| `bitnet-quantization` snapshot tests (5: QuantizationType Display/Debug, best_for_arch x86_64/aarch64, round_trip I2S/TL2) + `bitnet-cli` snapshot tests (5, `full-cli` gated: InferenceCommand defaults, help text for --max-tokens/--stop/sampling/--prompt-template aliases) | `crates/bitnet-quantization/tests/snapshot_tests.rs`, `crates/bitnet-cli/tests/snapshot_tests.rs` | #707 |
| `bitnet-runtime-feature-flags-core` (4 snaps: to_labels CPU/GPU+CUDA/empty, active feature set) + `bitnet-testing-scenarios-core` (4: scenario descriptions, timeout, log level, count) + `bitnet-startup-contract-core` (3: RuntimeComponent labels, summary tokens, is_compatible) + `bitnet-feature-contract` (4: consistent/inconsistent/empty) + `bitnet-testing-policy-core` (3: unit/local summary, active_profile_summary, CI snapshot) | `crates/bitnet-{runtime-feature-flags,testing-scenarios,startup-contract,feature-contract,testing-policy}-core/tests/snapshot_tests.rs` | #708 |
| `bitnet-inference` (7 snaps: GenerationConfig defaults/greedy/creative, InferenceConfig defaults, validation error) + `bitnet-kernels` (4: provider count/fallback/selection/name, cpu-gated) + `bitnet-models` (5: LoadConfig/ProductionLoadConfig defaults, DeviceStrategy Debug) + `bitnet-server` (6: ServerSettings host/port/timeouts, BatchEngineConfig, ConcurrencyConfig, DeviceConfig variants) | `crates/bitnet-{inference,kernels,models,server}/tests/snapshot_tests.rs` | #709 |
| `bitnet-runtime-context-core` (3: ActiveContext Unit/Local+Integration/Ci defaults, TestingScenario variants) + `bitnet-startup-contract-diagnostics` (3: profile_summary format, info count, warnings count) + `bitnet-startup-contract-guard` (3: RuntimeComponent labels, is_compatible, feature_line prefix) + `bitnet-runtime-feature-flags` (3: feature_line prefix+full, feature_labels count) + `bitnet-testing-scenarios-profile-core` (7: ConfigurationContext/ReportingProfile/FixtureProfile/ComparisonToleranceProfile/CrossValidationProfile/TestConfigProfile defaults, ReportFormat variants) | `crates/bitnet-{runtime-context-core,startup-contract-diagnostics,startup-contract-guard,runtime-feature-flags,testing-scenarios-profile-core}/tests/snapshot_tests.rs` | #710 |
| `bitnet-st2gguf` snapshot tests (6: `ConversionOptions` defaults/strict, `QuantizationType` I2_S/TL2 Display, `ConversionError` missing-file/unsupported-format messages) | `crates/bitnet-st2gguf/tests/snapshot_tests.rs` | #711 |
| Snapshot tests for 5 thin-wrapper/policy-facade crates (15 total): `bitnet-runtime-profile-contract-core` (4), `bitnet-testing-policy-contract` (3), `bitnet-testing-policy-interop` (3), `bitnet-testing-policy-tests` (3), `bitnet-testing-profile` (2) | `crates/bitnet-{runtime-profile-contract-core,testing-policy-contract,testing-policy-interop,testing-policy-tests,testing-profile}/tests/snapshot_tests.rs` | #712 |
| `bitnet-runtime-feature-flags` snapshot test name/value corrected to `--features cpu` reality | `crates/bitnet-runtime-feature-flags/tests/snapshot_tests.rs` | #715 |
| `docs/development/test-suite.md` updated: snapshot/property/fuzz sections, test counts 1935→2082+ | `docs/development/test-suite.md` | #716 |
| `CLAUDE.md` updated: test count 970+→2082+, snapshot/property test categories added | `CLAUDE.md` | #717 |
| `GgufTokenizer::real_vocab_size()` method — distinguishes real vs padded vocab sizes | `crates/bitnet-tokenizers/src/lib.rs`, `crates/bitnet-tokenizers/src/gguf_loader.rs` | #673 |
| Log-capture test infrastructure (`tracing-test 0.2.6`) and `#[traced_test]` unit tests | `crates/bitnet-prompt-templates/src/lib.rs` | #686 |
| Proptest for `bitnet-compat` (+5) and `bitnet-st2gguf` (+9); workspace: 3,345→3,359 tests | `crates/bitnet-compat/tests/property_tests.rs`, `crates/bitnet-st2gguf/tests/property_tests.rs` | #722 |
| Proptest for `bitnet-trace` (+6 JSON round-trip/hash/rms invariants) and `bitnet-kernels` (+8 provider/quantize invariants); proptest: 22→23 crates | `crates/bitnet-trace/tests/property_tests.rs`, `crates/bitnet-kernels/tests/property_tests.rs` | #724 |
| Proptest for `bitnet-server` (+11 BatchEngineConfig/SecurityValidator invariants); proptest: 23→24 crates | `crates/bitnet-server/tests/property_tests.rs` | #725 |
| Proptest for `bitnet-cli` (+9 CLI arg-parsing invariants); proptest: 24→26 crates; workspace: 3,359→3,384 tests | `crates/bitnet-cli/tests/property_tests.rs` | #726 |
| Fix GPU mode compilation in `bitnet-receipts` (missing optional `bitnet-kernels` dep; let-chain type inference fix) | `crates/bitnet-receipts/Cargo.toml`, `src/lib.rs` | #731 |
| Fix CLI tests: assert `.stderr(...)` not `.stdout(...)` for error messages logged via `tracing::error!` | `crates/bitnet-cli/tests/inspect_ln_stats.rs`, `validation_workflow.rs` | #730 |
| Proptest for `bitnet-common` (+17), `bitnet-tokenizers` (+9), `bitnet-inference` (+16); proptest: 26→29 crates; workspace: 3,384→3,426 tests | `crates/*/tests/property_tests.rs` | #732 |
| Proptest for `bitnet-quantization` (+12 math invariants) and `bitnet-models` (+8 predicate/re-export invariants) | `crates/bitnet-quantization/tests/property_tests.rs`, `crates/bitnet-models/tests/property_tests.rs` | #733 |
| docs: add `docs/tutorials/first-inference.md` (step-by-step inference guide); fix broken link in `docs/README.md`; expand docs README index with 11 previously unlisted guides | `docs/tutorials/first-inference.md`, `docs/README.md` | #735 |
| fix: config doctest called `with_max_new_tokens()` but method was renamed to `with_max_tokens()` | `crates/bitnet-inference/src/config.rs` | #736 |

### 🔲 What's Planned

1. **CUDA smoke lane** (self-hosted runner, nightly/manual)
   - Allocate device, run small inference, upload parity receipt
   - Blocked on having a CUDA-capable self-hosted runner

2. **Scheduled fuzz/crossval evidence expansion**
   - Nightly timeboxed fuzz runs with artifact upload (partially done via `fuzz-ci.yml`)
   - Real-model crossval producing receipts (gated on model download infrastructure)

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
