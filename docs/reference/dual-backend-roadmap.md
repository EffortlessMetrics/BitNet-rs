# Dual-Backend Support Implementation Roadmap

> **Last updated**: reflects implementation state after PRs #608–#997.
> Items marked ✅ are **done**; items marked 🔲 are **planned**.
> **Recent wave (PRs #978–#997)**: Metal backend (#992), Vulkan runtime probing (#993), Intel oneAPI
> backend (#986), AVX-512 kernel selection and hardening (#989), NEON kernel improvements (#988),
> OpenGL probing (#984), OpenCL/Vulkan CLI aliases (#985), ROCm availability field (#995),
> AVX-512 TL2 quantization kernels (#997), TL2 2-bit domain fix (#978), CPU golden path E2E
> infrastructure (#949).

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
| `bitnet-sampling` SRP microcrate: Phase 6 extraction ✅ DONE | `crates/bitnet-sampling/` | #629 |
| `bitnet-transformer` SRP microcrate: Phase 6 extraction ✅ DONE | `crates/bitnet-transformer/` | #629 |
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
| docs: update test counts and roadmap for PRs #733-#736; update CHANGELOG; fix duplicate `--no-default-features` in getting-started.md | `CLAUDE.md`, `docs/development/test-suite.md`, `CHANGELOG.md`, `docs/getting-started.md` | #737 |
| Proptest for `bitnet-sys` (+10 CompileTimeLibCapabilities invariants) and `bitnet-st-tools` (+8 is_ln_gamma properties); proptest: 30→32 crates; workspace: 3,446→3,464 tests | `crates/bitnet-sys/tests/property_tests.rs`, `crates/bitnet-st-tools/tests/property_tests.rs` | #738 |
| docs: update test counts and roadmap for PR #738; 3,464 tests; 32 proptest crates | `CLAUDE.md`, `docs/development/test-suite.md`, `CHANGELOG.md`, `docs/reference/dual-backend-roadmap.md` | #739 |
| Add fuzz targets for `bitnet-logits` (6 functions) and `bitnet-generation` (`check_stop`); proptest for `bitnet-runtime-context-core` (+8 tests: Display→FromStr round-trips, env-var default precedence); fuzz: 11→13 targets; proptest: 32→33 crates; workspace: 3,464→3,472 tests | `fuzz/fuzz_targets/logits_transforms.rs`, `fuzz/fuzz_targets/generation_stop_check.rs`, `crates/bitnet-runtime-context-core/tests/property_tests.rs` | #740 |
| chore: remove duplicate workspace members in Cargo.toml (bitnet-device-probe, -logits, -generation, -engine-core, -gguf listed twice) | `Cargo.toml` | #741 |
| docs: update test counts and roadmap for PRs #739-#741; 3,472 tests; 33 proptest crates; 13 fuzz targets | `CLAUDE.md`, `docs/development/test-suite.md`, `CHANGELOG.md`, `docs/reference/dual-backend-roadmap.md` | #742 |
| Proptest for `bitnet-runtime-profile-contract-core` (+7 properties: snapshot/violation coherence, feature-set completeness) and `bitnet-testing-policy-runtime` (+6 properties: GridCompatibility invariants); proptest: 33→35 crates; workspace: 3,472→3,485 tests | `crates/bitnet-runtime-profile-contract-core/tests/property_tests.rs`, `crates/bitnet-testing-policy-runtime/tests/property_tests.rs` | #743 |
| test(proptest): add property tests for `bitnet-testing-policy-tests` (+8 properties: PolicyDiagnostics coherence, `is_grid_compatible()` invariant); proptest: 35→36 crates; workspace: 3,485→3,493 tests | `crates/bitnet-testing-policy-tests/tests/property_tests.rs` | #745 |
| ci: add 19 BDD/policy/testing-infra crates to Build & Test matrix; these crates had test suites but were excluded from CI (run with `--features cpu` to satisfy profile snapshot requirements) | `.github/workflows/ci-core.yml` | #746 |
| fix: serialize `find_bitnet_lib_dirs` tests with `#[serial(bitnet_env)]` to prevent `BITNET_CROSSVAL_LIBDIR` race condition in xtask | `xtask/src/cpp_setup_auto.rs` | #748 |
| test(proptest): add 10 property + unit tests for `bitnet-test-support` (EnvGuard/EnvScope API semantics) | `crates/bitnet-test-support/tests/property_tests.rs` | #749 |
| test(proptest): add 17 property + unit tests for `bitnet-testing-scenarios-profile-core` (Default value invariants, fuzz-grade shape coverage across 5 structs) | `crates/bitnet-testing-scenarios-profile-core/tests/property_tests.rs` | #750 |
| Restrict model loading to configured directories via `BITNET_ALLOWED_MODEL_DIRECTORIES`; path canonicalization and directory containment check | `crates/bitnet-models/src/`, `crates/bitnet-server/src/` | #753 |
| Keyboard navigation (ArrowLeft/ArrowRight/Home/End) for WASM browser example tab list | `examples/wasm/` | #754 |
| Project renamed from BitNet.rs to BitNet-rs throughout (1,531 files, 6,281 occurrences) | workspace-wide | #755 |
| Harden model path validation: prevent symlink traversal and empty-string allowlist bypass | `crates/bitnet-models/src/`, `crates/bitnet-server/src/` | #756 |
| TL1 kernel fix: replaced `matmul_i2s` with `dequantize+matmul` pipeline; fixed 3 compounding bugs | `crates/bitnet-quantization/`, `crates/bitnet-kernels/` | #760 |
| TL2 kernel fix: replaced `matmul_i2s` with `dequantize+matmul` pipeline (same pattern as #760) | `crates/bitnet-quantization/`, `crates/bitnet-kernels/` | #761 |
| Proptest added to 6 infrastructure crates; proptest total: 44→50 crates | `crates/*/tests/property_tests.rs` | #762 |
| `bitnet-device-probe` microcrate: `CpuCapabilities`/`GpuCapabilities` types with proptest coverage | `crates/bitnet-device-probe/` | #765 |
| CPU golden-path E2E test with receipt invariants | `crates/bitnet-inference/tests/cpu_golden_path.rs` | #766 |
| test(gguf): expanded property tests, snapshot tests, and unit tests for bitnet-gguf — 33 → 49 tests | `crates/bitnet-gguf/` | #767 |
| test(srp-crates): expand proptest coverage for bitnet-logits, bitnet-generation, bitnet-engine-core | `crates/bitnet-logits/`, `crates/bitnet-generation/`, `crates/bitnet-engine-core/` | #768 |
| feat(inference): `BackendStartupSummary` — startup logs `requested=X detected=[…] selected=Y` | `crates/bitnet-inference/`, `crates/bitnet-server/` | #771 |
| ci: standalone `grid-check` job in `ci-core.yml` running `xtask grid-check --cpu-only` | `.github/workflows/ci-core.yml` | #772 |
| chore: CHANGELOG and CLAUDE.md docs update for PRs #765–#771 | `CHANGELOG.md`, `CLAUDE.md` | #773 |
| test(sampling): 7 new proptests for `bitnet-sampling` (top_k, repetition_penalty, temperature entropy, multi-step, reset) | `crates/bitnet-sampling/tests/property_tests.rs` | #774 |
| feat(ci): nightly scheduled fuzz workflow with corpus caching — 7 targets × 60 s, crash artifact upload | `.github/workflows/nightly-fuzz.yml` | #775 |
| feat(inference): `bitnet-logits` wired as dependency of `bitnet-inference`; duplicate logits math in `generation/sampling.rs` replaced | `crates/bitnet-inference/`, `crates/bitnet-logits/` | #776 |
| feat(ci): `gpu-smoke.yml` updated with weekly schedule and receipt artifact upload | `.github/workflows/gpu-smoke.yml` | #777 |
| refactor(quantization): dead code cleanup — removed unused `KernelProvider` imports and unused fields | `crates/bitnet-quantization/` | #779 |
| test(integration): 12 cross-crate SRP integration tests for logits/generation/prompt-templates/engine-core | `crates/bitnet-inference/tests/srp_integration_test.rs` | #781 |
| feat(fuzz): RoPE table generation fuzz target — `rope_table_gen.rs` using `arbitrary::Arbitrary`; verifies sin²+cos²≈1 invariant | `fuzz/fuzz_targets/rope_table_gen.rs` | #783 |
| test(transformer): 5 new KVCache/config property tests — shape invariants after N appends, layer independence, layer count, head divisibility, seq_len monotonicity | `crates/bitnet-transformer/tests/property_tests.rs` | #784 |
| test(tokenizers): 5 new property tests — BOS/EOS prepend, decode never panics, tokenize preserves words, config serde round-trip, EOS ID bounds | `crates/bitnet-tokenizers/tests/property_tests.rs` | #785 |
| docs(api): Examples/Errors/Panics sections added to bitnet-logits, bitnet-generation, bitnet-engine-core, bitnet-sampling, bitnet-device-probe public APIs | `crates/bitnet-{logits,generation,engine-core,sampling,device-probe}/src/lib.rs` | #786 |
| bench: 6 criterion benchmark functions in `srp_ops.rs` — logits pipeline, top-k (k=5/50), repetition penalty, argmax, RoPE build_tables, KV cache append | `benches/srp_ops.rs` | #787 |
| feat(fuzz): BPE tokenizer encode fuzz target — `tokenizer_encode.rs` with 4 exercise paths; fuzz total: 13→15 | `fuzz/fuzz_targets/tokenizer_encode.rs` | #788 |
| E2E golden-path reproducibility + pinned-output tests (2 deterministic, seed=42; tokens [140,459,459,459] pinned) | `crates/bitnet-inference/tests/e2e_cpu_golden_path.rs` | #790 |
| README modernization — Rust 2024 badge, Features list, architecture diagram, Feature flags table | `README.md` | #791 |
| feat(fuzz): BPE tokenizer encode fuzz target (re-create) — `fuzz/fuzz_targets/tokenizer_encode.rs` with 4 exercise paths; fuzz total remains 15 | `fuzz/fuzz_targets/tokenizer_encode.rs` | #792 |
| chore: docs update batch #790-#791 — Updated `CHANGELOG.md` and `CLAUDE.md` for PRs #790 and #791 | `CHANGELOG.md`, `CLAUDE.md` | #793 |
| chore: GitHub repo settings update — Updated `.github/settings.yml` description/topics; added `.github/settings.yml` to `ci-core.yml` path triggers | `.github/settings.yml`, `.github/workflows/ci-core.yml` | #794 |
| chore: release v0.1.1 — version bump 0.1.0 → 0.1.1 across workspace; `Cargo.lock` regenerated | `Cargo.toml` workspace members, `Cargo.lock` | #798 |
| feat(fuzz): add gguf_metadata_values fuzz target — new `fuzz/fuzz_targets/gguf_metadata_values.rs`; exercises arbitrary GGUF metadata value sequences for parser panic safety | `fuzz/fuzz_targets/gguf_metadata_values.rs` | #799 |
| test: add 21 proptest cases for bitnet-gguf (GgufValue, TensorInfo, GgufMetadataKv) and 7 for bitnet-sys (CompileTimeLibCapabilities summary) | `crates/bitnet-gguf/tests/property_tests.rs`, `crates/bitnet-sys/tests/property_tests.rs` | #800 |
| test: add `InferenceReceipt::to_json_string()` convenience method + snapshot test pinning receipt JSON output | `crates/bitnet-receipts/src/lib.rs`, `crates/bitnet-receipts/tests/snapshot_tests.rs` | #801 |
| test(integration): expand SRP cross-crate integration tests to 22 tests (10 new) — bitnet-logits→bitnet-sampling pipeline, bitnet-generation stop criteria, RoPE tables, bitnet-device-probe determinism, bitnet-engine-core session config | `crates/bitnet-inference/tests/srp_integration_test.rs` | #802 |
| ci: add `fuzz/**` to `ci-core.yml` path triggers so fuzz PRs receive required CI checks | `.github/workflows/ci-core.yml` | #803 |
| chore: fix stale MSRV cache key in compatibility workflow (1.89 → 1.92) — prevents incorrect CI cache hits | `.github/workflows/compatibility.yml` | #805 |
| test: add proptest coverage for bitnet-logits (temperature scaling, softmax invariants, top-k filtering, repetition penalty) and bitnet-generation (stop criteria, token accumulation, streaming order) | `crates/bitnet-logits/tests/property_tests.rs`, `crates/bitnet-generation/tests/property_tests.rs` | #806 |
| feat(fuzz): add tokenizer_encode_decode fuzz target covering BasicTokenizer, UniversalTokenizer, wrapper tokenizers, and HfTokenizer BPE round-trips | `fuzz/fuzz_targets/tokenizer_encode_decode.rs` | #807 |
| test: add 22 proptest cases for bitnet-quantization — TL1/TL2/I2_S round-trip bounded error, scale positivity, block alignment, and edge cases (all-zeros, alternating signs) | `crates/bitnet-quantization/tests/property_tests.rs` | #808 |
| feat(fuzz): safetensors parser fuzz targets — `safetensors_metadata.rs` (header path) and `safetensors_parser.rs` (full parse); `fuzz/Cargo.toml` updated | `fuzz/fuzz_targets/safetensors_metadata.rs`, `fuzz/fuzz_targets/safetensors_parser.rs` | #813 |
| test(bdd): 5 new BDD grid cells (Unit/Ci, Integration/Ci, Performance/Local, Development/Local, Smoke/Ci); snapshot count 8→13 | `crates/bitnet-bdd-grid/src/lib.rs` | #814 |
| test(proptest): 31 property + unit tests for `bitnet-ffi` — BitNetCConfig/BitNetCInferenceConfig round-trips, BitNetCError display, MemoryStats arithmetic, thread-local error state | `crates/bitnet-ffi/tests/property_tests.rs` | #815 |
| CI workflow fixes: push/PR triggers and `cargo-tests` smoke job added to property-tests and performance-tracking workflows | `.github/workflows/property-tests.yml`, `.github/workflows/performance-tracking.yml` | #879 |
| Additional property tests for `bitnet-server` (batch/concurrency/security config invariants) and `bitnet-ffi` (C API round-trips) | `crates/bitnet-server/tests/`, `crates/bitnet-ffi/tests/` | #880 |
| 10 previously-ignored tests activated: env-var guards converted from `#[ignore]` to active tests | workspace | #901 |
| 24 `bitnet-logits` property-based tests (softmax, log-softmax, repetition penalty, temperature, top-k, top-p, numerical stability) | `crates/bitnet-logits/tests/` | #902 |
| Fuzz targets: `sampling_no_panic`, `receipt_json_roundtrip`, `prompt_template_no_panic` (3 new targets; total grows to 18) | `fuzz/fuzz_targets/` | #903 |
| E2E golden-path tests: synthetic GGUF fixture (< 2 KB, no model download), 5 deterministic tests covering header parsing, metadata round-trip, full load, receipt invariants | `crates/bitnet-inference/tests/` | #904 |
| `xtask grid-check [--dry-run]` BDD compile-coverage gate: 18 cells including cpu, cpu+full-cli, cpu+crossval, cpu+gpu | `xtask/src/grid_check.rs` | #905 |
| 45 comprehensive tokenizer property-based tests: 40 proptest + 5 integration covering encode/decode round-trips, vocab consistency, Unicode safety, config JSON round-trip, auto-discovery, batch encoding | `crates/bitnet-tokenizers/tests/` | #906 |
| 43 server security & middleware tests: 7 CORS, 5 security header, 9 request validation, 4 model path, 3 config/auth, 4 IP extraction, 8 property, 1 health endpoint; uses `tower::ServiceExt::oneshot` | `crates/bitnet-server/tests/` | #907 |
| 72 comprehensive tests for `bitnet-compat` (diagnose/export/verify/stamp paths) and `bitnet-sys` (CompileTimeLibCapabilities, disabled-feature stubs) | `crates/bitnet-compat/tests/`, `crates/bitnet-sys/tests/` | #909 |
| 13 fewer ignored tests: `tokenization_smoke` converted to `CROSSVAL_GGUF` opt-in; AC3/AC6 acceptance tests gated on `BITNET_RUN_SLOW_TESTS` (1043→1030 ignored) | workspace | #910 |
| Comprehensive tests for `bitnet-runtime-feature-flags` (RuntimeFeatureFlags snapshots, CPU-always-true invariant, CUDA/GPU orthogonality, JSON round-trips) and `bitnet-feature-contract` | `crates/bitnet-runtime-feature-flags/tests/`, `crates/bitnet-feature-contract/tests/` | #911 |
| 25+ comprehensive tests for `bitnet-device-probe`: SIMD level detection ordering, GPU probe validation, DeviceCapabilities construction/equality, probe determinism and capability invariants | `crates/bitnet-device-probe/tests/` | #912 |
| 25+ extended integration and property tests for `bitnet-quantization`: I2_S, TL1, TL2, QK256 roundtrip accuracy, block-size invariants, edge-case inputs | `crates/bitnet-quantization/tests/` | #914 |
| 65 tests for `bitnet-validation` (LayerNorm bounds, policy keys, gate modes, RMS validation) and 56 tests for `bitnet-kernel-registry` (kernel dispatch, capability registry, SIMD selection) | `crates/bitnet-validation/tests/`, `crates/bitnet-kernel-registry/tests/` | #915 |
| 20+ comprehensive tests for `bitnet-trace`: TraceSink and TraceComparison APIs, JSON round-trips, hash invariants, decode-step tracing | `crates/bitnet-trace/tests/` | #916 |
| 36 tests for `bitnet-bdd-grid` (grid rows/columns, cell lookup, scenario validation) and 28 tests for `bitnet-testing-policy` (PolicyDiagnostics, GridCompatibility invariants) | `crates/bitnet-bdd-grid/tests/`, `crates/bitnet-testing-policy/tests/` | #917 |
| Wave 3 fuzz targets: `gguf_writer_roundtrip`, `tokenizer_encode_no_panic`, `validation_no_panic` | `fuzz/fuzz_targets/` | #919 |
| 43 tests for `bitnet-inference` (SamplingConfig, SamplingStrategy, GenerationConfig, InferenceReceipt, ModelInfo, ProductionInferenceEngine, streaming, error handling) and 43 tests for `bitnet-models` (ModelMetadata, QuantizationType, BitNetConfig, LoadConfig, ProductionModelLoader, tensor operations) | `crates/bitnet-inference/tests/`, `crates/bitnet-models/tests/` | #924 |
| **Ignored-test reduction wave 3**: justification strings added to all remaining `#[ignore]` markers; env-var guards for environment-sensitive tests | workspace-wide | #925 |
| **1000+ tests milestone** ✅: comprehensive README rewrite, updated `docs/development/test-suite.md` with current counts and xtask grid-check docs | `README.md`, `docs/development/test-suite.md` | #920 |
| 29 tests for `bitnet-transformer` (RoPE invariants, KVCache shape/seq-len properties) and 37 tests for `bitnet-honest-compute` (ComputeMode gating, receipt validation, honest-compute enforcement) | `crates/bitnet-transformer/tests/`, `crates/bitnet-honest-compute/tests/` | #921 |
| 26 extended tests for `bitnet-st2gguf` (conversion pipeline, quantization type handling, error variants) and 20 tests for `bitnet-server` (batch engine config, CORS, security middleware) | `crates/bitnet-st2gguf/tests/`, `crates/bitnet-server/tests/` | #922 |

### ✅ Phase 7: Test Coverage (DONE — 1500+ tests)

All Phase 7 test coverage targets have been met:
- **1500+ total tests** across workspace (snapshot, property, integration, fuzz, E2E); wave 4 fuzz targets (#937), insta snapshot tests (#938), and extended tokenizer/validation tests (#934) added in PRs #933–#942
- **Ignored-test reduction**: wave 3 (#925) added justification strings to all remaining `#[ignore]` markers and env-var guards for environment-sensitive tests
- **Phase 6 SRP microcrates**: bitnet-device-probe, bitnet-logits, bitnet-generation,
  bitnet-sampling, bitnet-transformer, bitnet-engine-core, bitnet-gguf — all ✅ DONE
- **Fuzz coverage**: 18+ fuzz targets across GGUF, tokenizer, quantization, generation,
  validation, receipts, and transformer paths
- **Wave 3 fuzz targets**: gguf_writer_roundtrip, tokenizer_encode_no_panic, validation_no_panic

### ✅ Phase 8: Multi-Backend GPU Support (DONE — PRs #978–#997)

All Phase 8 multi-backend targets have been completed:

| Capability | Location | PR |
|---|---|---|
| TL2 input quantization fix — stays in 2-bit domain | `crates/bitnet-quantization/` | #978 |
| CPU golden path E2E test infrastructure (5 deterministic tests, synthetic GGUF) | `crates/bitnet-inference/tests/cpu_golden_path.rs` | #949 |
| OpenGL probing in `DeviceProbe` | `crates/bitnet-device-probe/` | #984 |
| OpenCL/Vulkan CLI aliases (`--backend opencl`, `--backend vulkan`) | `crates/bitnet-cli/` | #985 |
| Intel oneAPI backend (`feature = "oneapi"`) — Intel CPU/GPU acceleration | `crates/bitnet-kernels/`, `crates/bitnet-device-probe/` | #986 |
| NEON kernel improvements — ARM NEON throughput and accuracy | `crates/bitnet-kernels/` | #988 |
| AVX-512 kernel selection and hardening | `crates/bitnet-kernels/` | #989 |
| Metal backend (`feature = "metal"`) — macOS/iOS GPU acceleration | `crates/bitnet-kernels/`, workspace | #992 |
| Vulkan runtime probing (`feature = "vulkan"`) — cross-platform GPU compute | `crates/bitnet-device-probe/`, `crates/bitnet-kernels/` | #993 |
| ROCm availability field in `DeviceProbe` | `crates/bitnet-device-probe/src/lib.rs` | #995 |
| AVX-512 TL2 quantization kernels | `crates/bitnet-kernels/` | #997 |

### 🔲 What's Planned

1. **Real-model crossval receipts** (gated on model download infrastructure)
   - Full crossval lane with C++ reference producing JSON receipts on nightly
   - Requires: `BITNET_CPP_DIR` provisioned on nightly runner

2. **Metal kernel implementations** — compute-shader BitNet GEMV for Apple Silicon
3. **Vulkan kernel implementations** — SPIR-V BitNet GEMV for cross-platform GPU
4. **ROCm/HIP kernel implementations** — AMD GPU acceleration

---

## Feature Lattice Design

### Current (multi-backend, `gpu` umbrella)

```toml
[features]
gpu    = ["kernels", "inference", "tokenizers", "bitnet-kernels/gpu", ...]
cuda   = ["gpu"]     # backward-compat alias — CUDA backend
metal  = ["gpu"]     # macOS/iOS Metal backend
vulkan = ["gpu"]     # cross-platform Vulkan compute backend
oneapi = ["gpu"]     # Intel oneAPI backend
```

### Code Gating Rules

```rust
// CUDA-specific imports:
#[cfg(feature = "cuda")]
use cudarc::...;

// Metal-specific code (macOS/iOS):
#[cfg(feature = "metal")]
pub fn metal_dispatch() { ... }

// Vulkan-specific code:
#[cfg(feature = "vulkan")]
pub fn vulkan_dispatch() { ... }

// Intel oneAPI:
#[cfg(feature = "oneapi")]
pub fn oneapi_dispatch() { ... }

// Generic "any GPU compiled" — works for all backends:
#[cfg(feature = "gpu")]
pub fn gpu_dispatch() { ... }

use bitnet_device_probe::{gpu_compiled, gpu_available_runtime};
```

### Additive backend model

Adding a new GPU backend only requires new feature entries and new
backend-specific modules. Existing `#[cfg(feature = "gpu")]` code
continues to work without modification across all backends.

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
