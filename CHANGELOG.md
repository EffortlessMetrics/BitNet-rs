# Changelog

All notable changes to bitnet-rs will be documented in this file.

## [Unreleased]

### Added
- `feat: add bitnet-logits property tests` — 4 new proptests for logits invariants in `crates/bitnet-logits/tests/logits_tests.rs` (#843)
- `feat(wasm): add copy-to-clipboard button for generated text in browser example` — clipboard UX improvement in WASM browser example (#809)
- `test: expand proptest coverage for compat, templates, and feature-flag crates` — 18 new proptests across bitnet-compat, bitnet-templates, and bitnet-runtime-feature-flags (#841)

### Fixed
- `fix(server): implement rate limiter cleanup to prevent memory leak` — `ConcurrencyManager::cleanup_rate_limiters` now properly cleans up idle entries to prevent unbounded memory growth (#810)

### Performance
- `perf(logits): optimize apply_top_p with sparse filtering` — skip zero-probability tokens before sorting, reducing unnecessary work in nucleus sampling (#811)
- `ci: increase fuzz build timeout to 60 minutes` — fuzz CI job timeout raised to 60 min, improved caching, added `RUSTFLAGS=-C debuginfo=0` to speed up fuzz builds (#840)
- `test: add numerical accuracy integration tests for bitnet-quantization` — 6 new integration tests (I2S dequantize, TL1 LUT, TL2 symmetry, round-trip accuracy, QK256 block size, zero vector) (#838)
- `test: add CPU golden path E2E validation tests` — 4 new E2E tests (stop token, receipt kernel IDs, schema version, max_tokens boundary) (#837)

## [v0.1.2] - 2026-02-27

Test coverage expansion wave (proptests, fuzz targets, BDD grid), reduced ignored tests to 77.

### Fixed
- `chore: fix stale MSRV cache key in compatibility workflow (1.89 → 1.92)` — prevents incorrect CI cache hits from cached 1.89 toolchain artifacts (#805)

### Added
- `test: expand proptest coverage for thin-coverage crates` — 15 new property tests across bitnet-bdd-grid, bitnet-honest-compute, bitnet-rope, and bitnet-trace (#834)
- `test: add snapshot tests for sampling, gguf, and receipts` — 5 new insta snapshots pinning GgufFileInfo display output and receipt schema version string for regression detection (#833)
- `test: audit and reduce ignored tests` — reduced ignored test count from 91 → 77 by enabling tests that are no longer blocked (#831)
- `test: expand server and CLI integration tests` — 18 new integration tests covering health endpoint, CORS, security validation, and CLI template parsing (#830)
- `feat(fuzz): add transformer_config and gguf_kv_read fuzz targets` — 2 new fuzz targets covering TransformerConfig deserialization and GGUF KV key/value read paths (#829)
- `test(bdd): expand BDD grid with new scenario cells` — 5 new BDD cells (18 total, was 13) in `crates/bitnet-bdd-grid/src/lib.rs` (#828)
- `test: proptest coverage for bitnet-common` — 9 new property tests in `crates/bitnet-common/tests/property_tests.rs` (MockTensor shapes, QuantizationType round-trips, error types, device consistency, KernelCapabilities ordering, warn_once deduplication) (#826)
- `test: expand proptest coverage for bitnet-sampling` — 7 new property tests in `crates/bitnet-sampling/tests/property_tests.rs` (temperature scaling, top-k/p filtering, repetition penalty, seed reproducibility, empty context, greedy determinism) (#825)
- `test: proptest coverage for bitnet-receipts` — 14 new proptests in `crates/bitnet-receipts/tests/property_tests.rs` (schema version, builder, JSON round-trips, kernel ID validation, honest compute gates, token counts) (#823)
- `test: proptest coverage for bitnet-validation` — 8 new proptests in `crates/bitnet-validation/tests/property_tests.rs` (LayerNorm bounds, error messages, policy keys, gate modes) (#822)
- `test: proptest coverage for bitnet-models` — 13 new proptests in `crates/bitnet-models/tests/property_tests.rs` (ModelConfig validation, GgufTensorType element sizes, GGUF magic bytes, path safety) (#821)
- `feat(fuzz): add quantization_input fuzz target` — new `fuzz/fuzz_targets/quantization_input.rs` covering I2S/TL1/TL2 dequantize, QK256 parsing, gemv_qk256_row, unpack/code_to_f32 (#820)
- `test: expand property test coverage for bitnet-tokenizers` — 8 new proptests in `crates/bitnet-tokenizers/tests/property_tests.rs` (token encoding bounds, special tokens, vocab size, encode-decode consistency, config validation, round-trip determinism) (#819)
- `feat(fuzz): add safetensors parser fuzz targets` — `fuzz/fuzz_targets/safetensors_metadata.rs` targets `SafeTensors::read_metadata()` header path; `fuzz/fuzz_targets/safetensors_parser.rs` replaced original stub; `fuzz/Cargo.toml` updated (#813)
- `test(bdd): expand BDD grid coverage` — 5 new BDD grid cells added in `crates/bitnet-bdd-grid/src/lib.rs` (Unit/Ci, Integration/Ci, Performance/Local, Development/Local, Smoke/Ci); grid snapshot count updated from 8 to 13 (#814)
- `test: add proptest coverage for bitnet-ffi` — `crates/bitnet-ffi/tests/property_tests.rs` with 31 tests (25 proptest + 6 unit): BitNetCConfig/BitNetCInferenceConfig round-trips, BitNetCError display invariants, MemoryStats arithmetic, thread-local error state (#815)
- `test: add proptest coverage for bitnet-logits and bitnet-generation` — temperature scaling, softmax invariants, top-k filtering, repetition penalty (bitnet-logits); stop criteria, token accumulation, streaming order (bitnet-generation) (#806)
- `feat(fuzz): add tokenizer_encode_decode fuzz target` — covers BasicTokenizer, UniversalTokenizer, wrapper tokenizers, and HfTokenizer BPE round-trips (#807)
- `test: add 22 proptest cases for bitnet-quantization` — TL1/TL2/I2_S round-trip bounded error, scale positivity, block alignment, and edge cases (all-zeros, alternating signs) (#808)

## [v0.1.1] - 2026-02-26

### Security
- Restrict model loading to configured directories via `BITNET_ALLOWED_MODEL_DIRECTORIES` (#753)
- Harden model path validation: prevent symlink traversal and empty-string allowlist bypass (#756)

### Changed
- Project renamed from BitNet.rs to BitNet-rs throughout (1,531 files, 6,281 occurrences) (#755)
- `refactor(quantization): dead code cleanup` — Removed unused `KernelProvider` imports and unused fields from `bitnet-quantization` (#779)

### Added
- `ci: add fuzz/** to ci-core.yml path triggers` — Added `fuzz/**` glob to the `paths` filter in `ci-core.yml` so fuzz PRs get required CI checks (#803)
- `test(integration): expand SRP cross-crate integration tests` — Expanded `srp_integration_test.rs` to 22 tests (10 new), covering bitnet-logits→bitnet-sampling pipeline, bitnet-generation stop criteria, RoPE tables, bitnet-device-probe determinism, bitnet-engine-core session config (#802)
- `test: add InferenceReceipt::to_json_string() convenience method + snapshot test` — Added `to_json_string()` on `InferenceReceipt`; snapshot test pins receipt JSON output for regression detection (#801)
- `test: add 21 proptest cases for bitnet-gguf and 7 for bitnet-sys` — `bitnet-gguf`: 21 properties covering `GgufValue`, `TensorInfo`, `GgufMetadataKv` round-trips and invariants; `bitnet-sys`: 7 properties for `CompileTimeLibCapabilities` summary logic (#800)
- `feat(fuzz): add gguf_metadata_values fuzz target` — New `fuzz/fuzz_targets/gguf_metadata_values.rs` for GGUF parser panic safety; exercises arbitrary metadata value sequences (#799)
- `chore: release v0.1.1` — Version bump 0.1.0 → 0.1.1 across workspace; `Cargo.lock` regenerated (#798)
- `chore: GitHub repo settings update` — Updated `.github/settings.yml` description and topics; added `.github/settings.yml` to `ci-core.yml` path triggers (#794)
- `chore: docs update batch #790-#791` — Updated `CHANGELOG.md` and `CLAUDE.md` for PRs #790 (E2E golden-path tests) and #791 (README modernization) (#793)
- `feat(fuzz): BPE tokenizer encode fuzz target (re-create)` — Recreated `fuzz/fuzz_targets/tokenizer_encode.rs` for BPE encode/decode paths with 4 exercise paths; fuzz total remains 15 (#792)
- `feat(fuzz): RoPE table generation fuzz target` — `fuzz/fuzz_targets/rope_table_gen.rs` using `arbitrary::Arbitrary`; verifies sin²+cos²≈1 (Pythagorean) invariant across arbitrary dimensions and base values (#783)
- `test(transformer): 5 new KVCache/config property tests` — shape invariants after N appends, layer independence, layer count validation, head divisibility check, seq_len monotonicity (#784)
- `test(tokenizers): 5 new property tests for encode/decode` — BOS/EOS prepend behaviour, decode never panics, tokenize preserves words, config serde round-trip, EOS ID bounds (#785)
- `docs(api): Examples/Errors/Panics sections for SRP crate APIs` — documentation improvements across `bitnet-logits`, `bitnet-generation`, `bitnet-engine-core`, `bitnet-sampling`, `bitnet-device-probe` (#786)
- `bench: criterion benchmarks for SRP ops (srp_ops.rs)` — 6 benchmark functions: logits pipeline, top-k at k=5/k=50, repetition penalty, argmax, RoPE build_tables, KV cache append (#787)
- `feat(fuzz): BPE tokenizer encode fuzz target` — `fuzz/fuzz_targets/tokenizer_encode.rs` with 4 exercise paths (empty, ASCII, Unicode, max-length boundary); fuzz total: 13→15 (#788)
- `test(e2e): reproducibility and pinned-output golden-path tests` — Added `crates/bitnet-inference/tests/e2e_cpu_golden_path.rs` with 2 deterministic E2E tests: `test_e2e_golden_path_reproducible` (seed=42, same seed gives identical tokens) and `test_e2e_golden_path_pinned_output` (pins greedy-argmax tokens [140,459,459,459] as regression guard); no model download required (#790)
- `docs: modernize README to well-designed FOSS format` — Rewrote README.md: added Rust 2024 edition badge, Features bullet list, new architecture diagram, Feature flags table; removed verbose receipt verification section; trimmed to ~90 lines net (#791)
- `ci: add BDD grid-check job to CI Core workflow` — Standalone `grid-check` job in `ci-core.yml` runs `xtask grid-check --cpu-only` in parallel with the build matrix (#772)
- `chore: docs update for PRs #765–#771` — CHANGELOG and CLAUDE.md updated to reflect merged PRs (#773)
- `test(sampling): expand proptest coverage for bitnet-sampling` — 7 new proptests covering top_k, repetition_penalty, temperature entropy, multi-step, and reset invariants (#774)
- `feat(ci): nightly scheduled fuzz workflow with corpus caching` — New `.github/workflows/nightly-fuzz.yml`; runs 7 fuzz targets for 60 s nightly, caches corpus per-target, uploads crash artifacts (#775)
- `feat(inference): wire bitnet-logits into bitnet-inference (SRP integration)` — Replaced duplicate logits math in `generation/sampling.rs` with `bitnet-logits` crate (#776)
- `feat(ci): CUDA smoke lane weekly schedule with receipt upload` — `gpu-smoke.yml` updated with weekly schedule and receipt artifact upload (#777)
- `feat(inference)`: BackendStartupSummary — startup logs `requested=X detected=[…] selected=Y` (#771)
- `test(srp-crates): expanded proptest coverage for bitnet-logits, bitnet-generation, bitnet-engine-core (#768)`
- `test(gguf): expanded property tests, snapshot tests, and unit tests for bitnet-gguf — 33 → 49 tests (#767)`
- **`bitnet-device-probe` microcrate — `CpuCapabilities`/`GpuCapabilities` with proptest coverage** (#765)
- **CPU golden-path E2E test with receipt invariants** (#766)
- **Property tests for 6 infrastructure crates** (PR #762): Extends proptest coverage to 6 additional crates. Proptest workspace total: **50 crates**.
- Keyboard navigation (ArrowLeft/ArrowRight/Home/End) for WASM browser example tab list (#754)
- **Property tests for `bitnet-test-support`** (PR #749): 10 new property + unit tests for `EnvGuard`/`EnvScope` API semantics — set/restore/remove round-trips, nested scope isolation, `model_path()`/`run_slow_tests()`/`run_e2e()` env helpers. Workspace total: **3,520 tests, all passing**. Proptest coverage spans **38 crates** (+2: `bitnet-test-support`, `bitnet-testing-scenarios-profile-core`).
- **Property tests for `bitnet-testing-scenarios-profile-core`** (PR #750): 17 new property + unit tests for Default value invariants across 5 structs — `FixtureProfile`, `CrossValidationProfile`, `ComparisonToleranceProfile`, `ReportingProfile`, `ResourceConstraints`. Includes fuzz-grade shape coverage for numeric fields, URL/path fields, and nested struct coherence.
- **CI env isolation fixes for `bitnet-runtime-context-core` and `bitnet-runtime-profile-contract-core`** (in PR #746): Tests that check `from_env_with_defaults()` Local default now use `temp_env::with_vars` to clear the `CI` env var so they pass in GitHub Actions. Snapshot tests for `active_context_default_fields` also isolated. `test_config_profile_defaults` snapshot uses `insta::with_settings!` filter to normalize the CPU-dependent `max_parallel_tests` field. Added `filters` feature to workspace insta definition.
- **CI coverage for BDD/policy/testing-infra crates** (PR #746): Added 19 BDD/policy/testing-infrastructure crates to the Build & Test matrix in `ci-core.yml`. These crates (including `bitnet-bdd-grid`, `bitnet-feature-contract`, `bitnet-testing-policy-core`, `bitnet-runtime-profile-contract-core`, etc.) had full test suites but were excluded from CI. All tests run with `--no-default-features --features cpu` to satisfy profile snapshot requirements.
- **Property tests for `bitnet-testing-policy-tests`** (PR #745): 8 new property/unit tests for `PolicyDiagnostics` invariants — `from_context_is_deterministic`, `is_grid_compatible_coherent_with_violations` (key: `is_grid_compatible() ↔ violations().is_some_and(|m,f| m.is_empty() && f.is_empty())`), `summary_never_panics_and_contains_scenario`, `feature_contract_consistent_does_not_panic`, `diagnostics_for_context_matches_from_context`, `profile_config_does_not_panic`, plus 2 unit tests (unit/local and e2e/ci). Workspace total: **3,493 tests, all passing**. Proptest coverage spans **36 crates**.

### Fixed
- **TL2 kernel fix** (PR #761): Replaced `matmul_i2s` with a `dequantize+matmul` pipeline (same pattern as TL1 fix in #760), eliminating 3 compounding bugs that caused incorrect TL2 quantization results.
- **TL1 kernel fix** (PR #760): Replaced `matmul_i2s` with a `dequantize+matmul` pipeline, fixing 3 compounding bugs that caused incorrect TL1 quantization results.
- **xtask `find_bitnet_lib_dirs` race condition** (PR #748): `test_find_bitnet_lib_dirs_both_tiers` was failing intermittently because `test_find_bitnet_lib_dirs_env_override` used a nested-function anti-pattern for `#[serial(bitnet_env)]` — the inner function's attribute doesn't actually serialize the outer `#[test]`. Fixed by adding `#[serial(bitnet_env)]` directly to all 5 related tests and flattening the nested-function pattern.


- **Docs update for PRs #739-#741** (PR #742): Updated test counts to 3,472, fuzz targets to 13, proptest crates to 33 across `CLAUDE.md`, `docs/development/test-suite.md`, `CHANGELOG.md`, `docs/reference/dual-backend-roadmap.md`.
- **Fuzz targets for `bitnet-logits` and `bitnet-generation`; proptest for `bitnet-runtime-context-core`** (PR #740): 2 new fuzz targets (bringing total to **13**) + 8 new property tests. `logits_transforms` fuzzes all 6 public functions in `bitnet-logits` with arbitrary f32 data including NaN/infinity — invariants: no panic, `argmax` returns valid index, `apply_top_k` count ≤ len, `softmax_in_place` outputs are non-negative and finite. `generation_stop_check` fuzzes `check_stop()` in `bitnet-generation` with arbitrary `StopCriteria`, token IDs, and decoded tail text — invariant: never panics. `bitnet-runtime-context-core` (+8 proptest/unit): `TestingScenario`/`ExecutionEnvironment` Display→FromStr round-trips (all variants), `from_env_with_defaults` default precedence (serial+temp-env isolation), parse-error-not-panic on garbage strings. Workspace total: **3,472 tests, all passing**. Proptest coverage spans **33 crates**.
- **Property tests for `bitnet-sys` and `bitnet-st-tools`** (PR #738): 18 new proptest properties across 2 previously uncovered crates. `bitnet-sys` (+10): `CompileTimeLibCapabilities` implication invariants (`has_cuda ⇒ available`, `has_bitnet_shim ⇒ available`), `summary()` canonical key presence/determinism, clone equality, and `cpp=available`/`cpp=unavailable` token correctness. `bitnet-st-tools` (+8): `is_ln_gamma()` fast-path (`non-.weight` suffix always returns `false`), known LN name matching, projection name rejection, determinism, and no-panic on arbitrary Unicode input. Workspace total: **3,464 tests, all passing**. Proptest coverage spans **32 crates**.
- **First-inference tutorial** (PR #735): Adds `docs/tutorials/first-inference.md` — a step-by-step guide covering model loading, greedy vs. temperature sampling, interactive chat, and performance notes. Fixes the broken link in `docs/README.md` and expands the Diataxis navigation index with 11 previously-unlisted guide links.
- **Property tests for `bitnet-common`, `bitnet-tokenizers`, `bitnet-inference`** (PR #732): 42 new proptest properties. `bitnet-common` (+17): Device ordering, JSON round-trips, `BackendSelection::select_backend(Cuda, cpu_only_caps)` always returns Err. `bitnet-tokenizers` (+9): `TokenizerConfig` field bounds, `BasicTokenizer` round-trip. `bitnet-inference` (+16): `GenerationConfig` builder invariants, stop-token semantics, `validate()` correctness. Workspace total: **3,426 tests, all passing**. Proptest coverage spans **29 crates**.
- **Property tests for `bitnet-quantization` and `bitnet-models`** (PR #733): 20 new proptest properties. `bitnet-quantization` (+12): `qk256_tolerance_bytes` monotonicity/floor/proportionality, `calculate_scale` positive-finite, `pack_2bit_values`/`unpack_2bit_values` round-trip, `quantize_value`/`dequantize_value` order-preservation. `bitnet-models` (+8): `is_layernorm_weight`/`is_projection_weight` mutual exclusivity, re-export identity. Workspace total: **3,446 tests, all passing**. Proptest coverage spans **30 crates**.
- **Property tests for `bitnet-trace`, `bitnet-kernels`, `bitnet-server`, and `bitnet-cli`** (PRs #724–#726): Extends proptest coverage to the four remaining high-value crates — `bitnet-trace` (+6 tests: JSON round-trip, name non-empty, `num_elements` = shape product, blake3 = 64 hex chars, rms ≥ 0.0, optional fields omitted when None); `bitnet-kernels` (+8 tests: always-has-providers, non-empty provider names, stable selection, CPU kernel available, TL1/I2S quantize output sizes, `gpu_compiled()` constancy); `bitnet-server` (+11 tests: `BatchEngineConfig` max_batch_size/concurrent batches positive, `RequestPriority` transitive ordering, `BatchRequest` builder invariants, `SecurityConfig` field bounds, `SecurityValidator` prompt length enforcement, temperature/top_p range validation); `bitnet-cli` (+9 tests: max_tokens positive, all three aliases equivalent, temperature/repetition_penalty parse range, top_k/top_p option handling, greedy flag, u64 seed precision). Workspace total: **3,384 tests, all passing**. Proptest coverage now spans **26 crates**.

### Fixed
- **Config doctest calling renamed method** (PR #736): `bitnet-inference/src/config.rs` module-level doctest called `with_max_new_tokens()` which was renamed to `with_max_tokens()`. All 16 doctests now pass.
- **CLI tests asserting stderr vs stdout** (PR #730): Error messages logged via `tracing::error!` go to stderr (when using `tracing-subscriber`), not stdout. `inspect_ln_stats.rs` and `validation_workflow.rs` now use `.stderr(...)` assertions.
- **Workspace snapshot tests (4 tests across 3 crates)** (PR #720): Replaced exact-count/exact-value snapshot assertions with presence checks in `bitnet-runtime-feature-flags` (`feature_labels_count_with_cpu_feature`, `feature_line_format_stable`), `bitnet-startup-contract-core` (`cli_component_observe_is_compatible_or_has_state`), and `bitnet-testing-policy-kit` (`active_feature_labels_returns_list`). Cargo feature unification in workspace builds activates extra features (`fixtures`, `reporting`, `trend`, `quantization`) via other crates, making context-dependent exact-match snapshots fail. Full workspace run now: **3,359 passed, 0 failed, 462 skipped**.
- **Duplicate workspace members in `Cargo.toml`** (PR #741): `bitnet-device-probe`, `bitnet-logits`, `bitnet-generation`, `bitnet-engine-core`, and `bitnet-gguf` were each listed twice in `[workspace.members]`. Cargo deduplicates silently, so there was no functional impact, but the duplicates were confusing. Removed the redundant entries from the phase-6 block at the bottom of the list.

### Documentation
- **Dual-Backend Roadmap update** (PR #719): Marks PRs #711–#717 as implemented in the roadmap tracking table; adds retrospective rows for previously-implemented but un-tracked items (Phase 6 SRP microcrates, BDD grid runner, CPU golden path, GPU smoke lane).

### Added
- **CLAUDE.md Test Count and Category Update** (PR #717): Updates test count from 970+ to 2,082+, skipped count from ~466 to ~462, and adds snapshot tests (37 files, 200+ assertions) and property tests (20 files, 100+ properties) to the Working Test Categories section.
- **`docs/development/test-suite.md` Snapshot/Property/Fuzz Update** (PR #716): Adds dedicated sections for Snapshot Tests (insta, 37 files, 200+ assertions, run/review/update commands), Property Tests (proptest, 20 files, PROPTEST_CASES env, key invariants), and Fuzz Testing (11 targets table, corpus location, nightly CI schedule); updates test counts and category table.
- **Snapshot Tests for 5 Thin Wrapper/Policy-Facade Crates** (PR #712): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for — `bitnet-runtime-profile-contract-core` (4 snapshots: canonical grid row count, active profile summary format, Unit/Local has cell, Unit/Local no GPU required), `bitnet-testing-policy-contract` (3 snapshots: `PolicyContract::detect()`, feature consistency, drift_check semantics), `bitnet-testing-policy-interop` (3 snapshots: `Environment` alias, grid row count, validate helper), `bitnet-testing-policy-tests` (3 snapshots: `GridScenario`/`GridEnvironment` aliases, `PolicyDiagnostics` cell presence), `bitnet-testing-profile` (2 snapshots: identity conversion helpers, grid row count). Adds 15 snapshot tests total.
- **`bitnet-st2gguf` Snapshot Tests** (PR #711): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for `bitnet-st2gguf` — 6 snapshots: `ConversionOptions` defaults, `ConversionOptions` strict mode, `QuantizationType` I2_S format string, `QuantizationType` TL2 format string, `ConversionError` missing file message, `ConversionError` unsupported format message. Catches silent default drift in the SafeTensors→GGUF conversion pipeline.
- **Markdownlint Workflow npm Resilience** (PR #714): Adds `|| echo ...` fallback on npm install step and `command -v markdownlint-cli2 &&` guard on the lint invocation so transient npm 403 errors no longer fail documentation PRs.
- **Snapshot Tests for 5 Runtime/Policy Microcrates** (PR #710): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for — `bitnet-runtime-context-core` (3 snapshots: `ActiveContext` scenario/environment Display strings for Unit/Local and Integration/Ci defaults, all TestingScenario variants), `bitnet-startup-contract-diagnostics` (3 snapshots: `StartupContractReport::profile_summary()` format, `info` line count, `warnings` count for compatible contract), `bitnet-startup-contract-guard` (3 snapshots: `RuntimeComponent` label strings, `is_compatible()` with Observe policy, `feature_line` prefix invariant), `bitnet-runtime-feature-flags` (3 snapshots: `feature_line()` prefix + full output, `feature_labels()` count without features), `bitnet-testing-scenarios-profile-core` (7 snapshots: `ConfigurationContext::default()`, `ReportingProfile` default formats, `FixtureProfile`/`ComparisonToleranceProfile`/`CrossValidationProfile`/`TestConfigProfile` defaults, `ReportFormat` variants Debug strings). Adds 19 snapshot tests total.
- **Snapshot Tests for `bitnet-inference`, `bitnet-kernels`, `bitnet-models`, `bitnet-server`** (PR #709): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for four high-value large crates — `bitnet-inference` (7 snapshots: `GenerationConfig` default fields, greedy/creative constructors, `InferenceConfig` defaults, validation error message), `bitnet-kernels` (4 snapshots: `KernelManager` provider count, fallback presence, selection returns Ok, name non-empty, behind `cpu` feature), `bitnet-models` (5 snapshots: `LoadConfig` defaults, `ProductionLoadConfig` defaults + max size, `DeviceStrategy` Debug formats), `bitnet-server` (6 snapshots: `ServerSettings` host/port/timeouts/device, `BatchEngineConfig` batch size, `ConcurrencyConfig` max requests, `DeviceConfig` variants). Adds 22 snapshot tests total.
- **Snapshot Tests for 5 BDD Infrastructure Microcrates** (PR #708): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for — `bitnet-runtime-feature-flags-core` (4 snapshots: `to_labels()` for CPU/GPU+CUDA/empty, active feature set labels), `bitnet-testing-scenarios-core` (4 snapshots: all scenario descriptions, Unit timeout, CI log level, scenario count), `bitnet-startup-contract-core` (3 snapshots: `RuntimeComponent.label()` for all variants, summary structure tokens, `is_compatible()` value), `bitnet-feature-contract` (4 snapshots: consistent/inconsistent contract snapshots, empty input), `bitnet-testing-policy-core` (3 snapshots: unit/local summary format, `active_profile_summary()`, integration/CI snapshot). Adds 18 snapshot tests total.
- **Snapshot Tests for `bitnet-quantization` and `bitnet-cli`** (PR #707): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for two high-value crates — `bitnet-quantization` (5 snapshots: `QuantizationType` Display/Debug format strings, `best_for_arch()` platform-specific selection for x86_64/aarch64, and `validate_round_trip()` boolean results for I2S and TL2 with known inputs), `bitnet-cli` (5 snapshots gated behind `full-cli`: `InferenceCommand` field defaults, help text sections for `--max-tokens`+aliases, `--stop`+aliases, sampling flags, and `--prompt-template`). Catches silent default drift and alias removal regressions.
- **Snapshot Tests for `bitnet-trace`, `bitnet-compat`, `bitnet-bdd-grid-core`** (PR #706): Adds `insta` to dev-deps and creates `snapshot_tests.rs` for three more microcrates — `bitnet-trace` (4 JSON snapshots pinning `TraceRecord` serialization format: minimal, with optional fields, decode-step, logits), `bitnet-compat` (2 snapshots pinning `GgufCompatibilityFixer::diagnose()` diagnostic message strings and issue count for a minimal GGUF), `bitnet-bdd-grid-core` (6 snapshots pinning `TestingScenario`/`ExecutionEnvironment`/`BitnetFeature` Display strings, `FeatureSet::labels()`, `BddCell` debug format, `BddGrid::validate()` None return). Adds 12 snapshot tests total across the three crates.
- **`bitnet-bdd-grid` Proptest and Snapshot Tests** (PR #705): Adds `proptest` and `insta` dev-deps to `bitnet-bdd-grid` and creates `property_tests.rs` (8 tests: 5 proptest properties for LazyLock stability, required∩forbidden always disjoint, `find()` consistent with `rows()`, `supports()`/`violations()` semantics, all intents non-empty; 3 unit tests: cell count == 8, Unit/Local present, Smoke/PreProduction present) and `snapshot_tests.rs` (4 snapshots pinning the full grid summary, Unit/Local cell features, EndToEnd/CI cell features, and total cell count).
- **Snapshot Tests for `bitnet-receipts`, `bitnet-sampling`, `bitnet-prompt-templates`** (PR #703): Adds dedicated `snapshot_tests.rs` files for three key SRP microcrates — `bitnet-receipts` (6 insta JSON snapshots: cpu_real_receipt, mock_receipt, receipt_with_backend_summary, receipt_with_model_info, receipt_with_performance, receipt_with_test_results), `bitnet-sampling` (6 insta debug snapshots: default/greedy/creative config, greedy_sample output, seeded sample seed=42, repetition penalty before/after context), `bitnet-prompt-templates` (7 insta snapshots: raw_simple, instruct_no_system, instruct_with_system_prompt, llama3_single_turn, llama3_with_system, instruct_multi_turn, llama3_multi_turn). Adds 19 snapshot tests total; previously only 1 snapshot existed in receipts and 2 in prompt-templates.
- **`bitnet-device-probe` Proptest Integration Tests** (PR #701): `crates/bitnet-device-probe/tests/property_tests.rs` (7 proptest tests) — covers `simd_level_display_never_empty` / `simd_level_consistent_across_calls` (determinism invariants), `gpu_compiled_is_stable` (compile-time constant must not vary), `device_capabilities_cpu_rust_always_true`, `device_capabilities_cuda_compiled_matches_gpu_compiled`, `gpu_fake_cuda_returns_true` / `gpu_fake_none_returns_false` (BITNET_GPU_FAKE env-var override semantics), and BITNET_STRICT_MODE interaction tests. Completes proptest coverage for all SRP microcrates in the workspace.
- **`bitnet-transformer` KVCache Snapshot Tests** (PR #702): `crates/bitnet-transformer/tests/snapshot_tests.rs` (5 insta snapshots) — pins `KVCache` and `LayerKVCache` structural invariants: initial state (num_layers, seq_len=0, max_seq_len, n_kv_heads), post-append (only target layer seq_len increments), post-clear (all layers reset), layer-initial-state, and 3-append seq_len accumulation. First snapshot coverage for this crate.
- **GGUF `open()` File Integration Tests** (PR #699): `crates/bitnet-gguf/tests/file_open_tests.rs` (4 tests) — exercises the real mmap-backed `open()` path using the committed 224-byte `tests/models/mini.gguf` synthetic fixture; covers happy-path (version=3, tensor_count=0, metadata_count=4), insta snapshot, nonexistent-path error, and wrong-magic-bytes error. No model download required.
- **Backend Selection Property + Snapshot Tests** (PR #698): `crates/bitnet-common/tests/backend_selection_tests.rs` (13 tests, no feature gate) — 5 proptest properties for `BackendRequest::Display` invariants, `summary()` format contracts, `select_backend()` determinism; 6 unit tests for specific selection scenarios; 2 insta snapshots pinning the summary string and `BackendSelectionResult` Debug format
- **Property Tests for 6 SRP Microcrates** (PR #696): Integration-level proptest coverage for `bitnet-testing-policy-core` (7 tests: PolicySnapshot invariants), `bitnet-testing-scenarios-core` (8 tests: Display↔FromStr round-trips + CI contract), `bitnet-honest-compute` (11 property + 5 unit: mock detection, classification, kernel ID hygiene), `bitnet-rope` (7 property + 4 unit: Pythagorean identity, dimension invariants, error paths), `bitnet-validation` (10 property + 4 unit: LN name detection, ruleset bounds), `bitnet-feature-contract` (5 property + 3 unit: set-diff semantics, consistency invariants) — adds ~45 tests
- **Property Tests for `bitnet-bdd-grid-core`** (PR #694): 10 proptest properties — `TestingScenario`/`ExecutionEnvironment`/`BitnetFeature` Display→FromStr round-trips (all variants lossless); `FeatureSet` insert→contains, superset satisfaction, labels completeness; plus 4 unit error-case tests
- **Property Tests for `bitnet-runtime-feature-flags-core`** (PR #692): 6 proptest properties verifying `FeatureActivation → FeatureSet` conversion invariants — `cuda ⇒ gpu`, `cpu ⇒ inference+kernels+tokenizers`, `feature_line` prefix, default activation empty labels, and 2 unit stability tests
- **Property Tests for `bitnet-startup-contract-core`** (PR #691): 4 proptest properties verifying `ProfileContract` invariants — context round-trips, summary non-empty, `Observe` policy never fails `enforce()`, `is_compatible()` consistent with `state()`; plus 1 unit stability test (5 total)
- **Property Tests for `bitnet-transformer`** (PR #689): 4 proptest properties covering KVCache invariants — seq_len tracks N append operations correctly, clear/reset semantics, overflow (capacity) rejection, and initial zero state
- **Tracing Instrumentation for Template Detection** (PR #686): `bitnet-prompt-templates::TemplateType::detect()` now emits `debug!` on each branch (Llama3Chat, Instruct, Raw) and `warn!` on the Raw fallback; in-crate `#[traced_test]` unit tests verify log capture
- **`tracing-test` 0.2.6 Added to Workspace** (PR #686): Shared dev-dep for crates that test tracing output

### Fixed
- **`bitnet-runtime-feature-flags` Snapshot Test Names and Values** (PR #715): Renamed `feature_labels_without_features_is_empty` → `feature_labels_count_with_cpu_feature` (the old name was incorrect — tests always run with `--features cpu`); updated snapshot values to reflect 4 compiled features (`cpu`, `inference`, `kernels`, `tokenizers`).
- **Env-var race in `bitnet-startup-contract-core`** (PR #691): Replaced `unsafe { env::set_var/remove_var }` with `temp_env::with_var + #[serial(bitnet_env)]` in `evaluate_preserves_context_overrides` test
- **4 Previously-Ignored Tests Unblocked** (PR #686): `test_once_per_layer_warning_guards`, `test_kv_cache_warning_message_format` (unique layer indices avoid `Once` state collisions), `test_detection_logs_decision`, `test_fallback_logs_warning` (converted to behavioral assertions; log coverage in emitting crate)
- **Fixture Timeout: 5-min → ~1s** (PR #687): Reduced huge synthetic fixture allocations in `bitnet-quantization` — vocab fixtures (50257→512), large projection (1024×4096→512×1024), GGUF model layers (realistic dims→2×32) reduced fixture allocation from >2GB to <5MB total

### Added (prior)
- **Property Tests for `bitnet-logits`** (PR #683): 13 proptest properties verifying softmax sum-to-one / non-negativity / argmax-preservation, temperature scaling semantics (T=1.0 identity, argmax preservation), top-k filtering (≤k elements, k=0/k≥len no-ops), argmax correctness, and repetition penalty semantics (1.0 no-op, reduces positives, worsens negatives)
- **Property Tests for `bitnet-generation`** (PR #683): 8 tests (5 property + 3 unit) covering `check_stop` priority order (token-ID > EOS > max-tokens > stop-string), `max_tokens=0` disabling budget, determinism, and stop-string boundary matching
- **Property Tests for `bitnet-engine-core` standalone suite** (PR #683): 7 tests (5 property + 2 unit) in `tests/property_tests.rs` for `SessionConfig` JSON round-trips, `BackendInfo` JSON round-trips, `SessionMetrics` non-negativity, and default values
- **`Tokenizer::get_family_name()` trait method** (PR #673): Returns family hint (`"llama3"`, `"default"`, etc.) based on special tokens; used for auto-template detection
- **Property Tests for `bitnet-device-probe`** (PR #650): 4 proptest properties verifying GPU compiled idempotency, SIMD level determinism, CPU always available, and cuda_compiled consistency with `gpu_compiled()`
- **Property Tests for `bitnet-engine-core`** (PR #650): 3 proptest properties verifying `SessionConfig` JSON round-trips, `BackendInfo` JSON round-trips, and `SessionMetrics` non-negativity invariants
- **Property Tests for `bitnet-validation`** (PR #650): 7 proptest properties for `rules.rs` (detect_rules name invariants, envelope bounds, proj_rms consistency) and `names.rs` (is_ln_gamma keyword recognition, suffix requirement, determinism)
- **Property Tests for `bitnet-sampling`** (PR #650): 5 proptest properties for greedy argmax, index bounds, softmax distribution validity, top-k finite count, temperature=0 greedy equivalence
- **Property Tests for `bitnet-rope`** (PR #650): 4 proptest properties for valid input success, shape invariants, sin²+cos²=1 trig identity, odd-dim rejection
- **Property Tests for `bitnet-receipts`** (PR #650): 4 proptest properties for schema validation, real compute_path, valid kernel ID acceptance, empty kernel ID rejection
- **Property Tests for `bitnet-common/kernel_registry`** (PR #650): 4 proptest properties for no-duplicate backends, best_available reachability, CUDA preference, requires_gpu semantics
- **Property Tests for `bitnet-prompt-templates`** (PR #650): 4 proptest properties for user text inclusion, Raw identity, Instruct suffix, non-Raw stop sequences
- **Property Tests for `bitnet-tokenizers`** (PR #650): 4 proptest properties for BasicTokenizer ASCII round-trip, byte-count invariant, empty input, empty slice decode
- **Property Tests for `bitnet-honest-compute`** (PR #650): 4 proptest properties for valid ID acceptance, too-long ID rejection, compute_path validation, classify_compute_path correctness
- **Docs Archive Cleanup** (PR #650): 136 stale planning/sprint/spec documents moved to `docs/archive/`; `docs/explanation/` reduced to 13 user-facing Diataxis docs
- **Libfuzzer Crash Artifacts Gitignored**: Added `fuzz/crash-*`, `fuzz/slow-unit-*`, `fuzz/leak-*`, `fuzz/timeout-*` patterns to `.gitignore` to keep fuzz crash files out of the repo
- **Runtime Backend Selection** (PR #642): `BackendCapabilities` snapshot at CLI/server startup producing `requested=X detected=[…] selected=Y` log line and receipt field
- **CPU Golden Path E2E Tests** (PR #643): 5 deterministic end-to-end tests in `bitnet-inference` always running in PR CI without model download
- **SRP Microcrates Wired Into CI** (PR #644): `bitnet-logits`, `bitnet-gguf`, `bitnet-generation`, `bitnet-device-probe`, `bitnet-engine-core` added to CI test matrix
- **QK256 (GGML I2_S) Pure-Rust Support** (PR #640): GGUF loader detects and stores QK256 tensors; pure-Rust `gemv_qk256()` kernel; dual I2_S flavor detection; 17 comprehensive tests; no FFI required
- **Property Tests for SRP Microcrates** (PR #649): proptest suites for `bitnet-gguf` (header parsing, magic validation, arbitrary-byte safety) and `bitnet-generation` (stop criteria, max-token budget, stop-string matching)
- **GGUF Header Fuzz Target** (PR #649, `fuzz/fuzz_targets/gguf_header.rs`): dedicated libfuzzer target exercising `bitnet-gguf`'s lightweight header parser
- **Fuzz Target Registration** (PR #649): `architecture_detection`, `tl_lut_helper`, `tokenizer_discovery`, `vocab_size_extraction` fuzz targets were present but not registered in `fuzz/Cargo.toml`; now properly wired

### Fixed
- **Flaky SIMD throughput test** (PR #681): Removed brittle `throughput.is_finite()` assertion in `bitnet-kernels` quantized matmul throughput test; avoids intermittent failures on CI runners where elapsed time rounds to 0
- **Clippy warnings workspace-wide** (PR #682): Resolved all `clippy --all-targets --features cpu` warnings; zero-warning policy enforced for CPU feature gate
- **`ci-core.yml` paths filter excludes documentation-only PRs** (PR #684): Added `CLAUDE.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.github/copilot-instructions.md` to the `paths:` trigger so docs-only PRs correctly run required status checks instead of being permanently blocked
- **Env-var race conditions eliminated across workspace** (PR #678): Replaced bare `unsafe { env::set_var/remove_var }` with `temp_env::with_var`/`with_vars`/`with_var_unset` + `#[serial(bitnet_env)]` in `bitnet-kernels` (3 tests in `issue_260_feature_gated_tests.rs`), `bitnet-models` (`test_iq2s_backend_selection`), `bitnet-trace` (5 integration tests + 1 unit test), and `bitnet-runtime-profile-contract-core` (4 tests); eliminates flaky test failures from cross-test env var races in parallel nextest runs
- **Template detection + KV cache init tests unblocked** (PR #673): 3 tests previously failing due to missing `get_family_name()` and wrong KV cache test logic now pass without `#[ignore]`
- **QK256, CLI, and simple inference tests unblocked** (PR #674): 9 TDD scaffold tests (`test_qk256_tolerance_*`, `test_help_text_snapshot`, `test_simple_real_inference`, etc.) activated by fixing stub implementations
- **Receipts property test mock filter** (PR #675): `proptest` strategy for valid kernel IDs excluded "mock"-containing strings that are rejected by honest-compute policy (`0.99f32 as f64` precision issue also fixed)
- **Issue #159 resolved — `MockGgufFileBuilder` writes real GGUFs** (PR #676): Replaced `b"mock_gguf_content"` placeholder with `GgufWriter`-based synthetic GGUF containing real transformer layer tensors; 4 previously ignored tests (`test_ac1_complete_transformer_weight_parsing_cpu`, `test_ac2_i2s/tl1/tl2_quantization_accuracy_cpu`) now pass
- **TL1/TL2 Quantizer Round-Trip Accuracy** (PR #641): LUT offset mismatch in `pack_2bit_values` caused clipping of codes 2–3; dequantize now maps ternary inputs within `[-1, 1]`
- **Security Audit** (PR #645): Updated `bytes` to 1.11.1 (RUSTSEC-2026-0007) and `time` to 0.3.47 (RUSTSEC-2026-0009); added documented accepted-risk entries for gix-date, rsa, bincode advisories; fixed LGPL false-positive in GPL license check; fixed supply-chain Cargo.lock verification command
- **GGUF Tokenizer Test Fixtures** (PR #648): Repaired three malformed GGUF fixtures (`kv_count` off-by-one, corrupted magic); rebuilt `llama3-with-hf-tokenizer.gguf` with correct 5-KV HF tokenizer metadata; added `.gitignore` exceptions so fixtures reach CI; fixed `get_u32_metadata` in `bitnet-models` to also accept `GgufValue::I32` (as generated by llama.cpp)
- **Gitignore Fix** (PR #649): `fuzz/` was globally gitignored, preventing fuzz source files from being tracked; replaced with specific artifact-only ignores for `fuzz/target/`, `fuzz/artifacts/`, `fuzz/coverage/`, `fuzz/corpus/`
- **Env-var race condition hardening** (PR #678): Replaced `unsafe { env::set_var / remove_var }` call sites without serial+RAII cleanup with `temp_env::with_var` + `#[serial(bitnet_env)]` where practical; duplicate `blake3` dev-dep removed; GPU cfg predicates unified to `any(feature = "gpu", feature = "cuda")`

### Documentation
- **CLAUDE.md + copilot-instructions.md accuracy refresh** (PR #680): Removed stale "Active Blockers" references to non-existent/closed issues (#254, #260, #439, #469); updated MSRV to 1.92.0; updated test counts (3,097 passing, 466 skipped); simplified ignored-test categorization to reflect actual reasons (model-gated, CUDA, slow, crossval, TDD scaffold); updated architecture doc with SRP microcrate section

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0-rc.0] - 2025-10-17

### Added

- **Pure-Rust GGUF Tokenizer** ([feat/crossval-parity-harness](https://github.com/EffortlessSteven/BitNet-rs/tree/feat/crossval-parity-harness)):
  - Load tokenizers directly from GGUF metadata (SPM protobuf, BPE vocab+merges)
  - No external tokenizer files required for self-contained models
  - Auto-detection from `tokenizer.ggml.model` metadata
- **BPE ByteLevel Prefix Space Fix**:
  - Enable `add_prefix_space=true` for both BPE pre-tokenizer **and** decoder
  - Ensures consistent " What" tokenization with leading space marker
  - Proper handling of GPT-2 style Ġ (U+0120) prefix markers
- **BPE Piece-to-GGUF-ID Remapping**:
  - Maps HuggingFace token IDs to authoritative GGUF vocabulary IDs via `HashMap<String, u32>`
  - Prevents ID drift from HuggingFace's internal ID assignment
  - Uses GGUF `tokenizer.ggml.tokens` array as source of truth (index = token ID)
  - Model-aware token IDs: BitNet GGUF has "ĠWhat" at position 3639 (not universal 3923)
- **Receipt-Based Provenance** (crossval):
  - Tokenizer metadata: `merges_count` (BPE), `tokenizer_blob_sha256` (SPM)
  - Environment metadata: `target_cpu`, `cpu_features`, `libc`, `rayon_threads`, `seed`
  - C++ metadata: `llama_cpp_commit` (from BITNET_CPP_DIR)
  - Prompt hash: `blake3` for formatted prompt verification
  - Timeout receipts with diagnostic data (120s guard)
- **Model-Aware Golden Token Tests**:
  - Split fixtures: `golden_tokens_gpt2.json`, `golden_tokens_llama.json`, `golden_tokens_llama3.json`
  - Auto-select based on `tokenizer.ggml.model` from GGUF
  - Exact-match validation locks in BPE and SPM behavior
- **Optional tok-debug Diagnostics**:
  - Feature-gated `--features tok-debug` for piece→ID diagnostics
  - Dumps first 8 tokens: `hf_id`, `piece`, `gguf_id`
- **LLaMA-3 Chat Prompt Support**:
  - Multi-prompt support via `CROSSVAL_PROMPT_SET=math|chat|all`
  - Auto-detect `parse_special=true` for `<|start_header_id|>`, `<|eot_id|>`
  - Proper EOT vs EOS handling
- **CI Workflows**:
  - `parity-proof.yml`: Fast PR gate with receipt artifact upload
  - `nightly-parity-matrix.yml`: Prompt+quant matrix with dated archiving

### Fixed

- **BPE Token ID Mapping**:
  - Rust tokenizer now uses GGUF vocabulary IDs instead of HuggingFace internal IDs
  - Fixes token ID mismatches by looking up piece strings in GGUF `tokenizer.ggml.tokens` array
  - Handles space normalization: both ` What` and `ĠWhat` forms correctly mapped
- **FFI Memory Safety**:
  - Hardened batch lifecycle management in C++ shim
  - Two-call tokenization pattern (preflight + allocation)
  - Explicit safety contract documentation for FFI boundary
- **Deterministic Inference**:
  - Seeded runs with `BITNET_SEED` + `RAYON_NUM_THREADS=1`
  - Reproducible tokenization and generation
- **SPM Blob Reproducibility**:
  - SHA256 fingerprinting of SentencePiece protobuf
  - Validates tokenizer integrity across runs

### Documentation

- Added `docs/releases/v0.10.0-rc.0-summary.md`: Comprehensive release guide
- Updated `CLAUDE.md`: Document tok-debug feature and golden token tests
- CI workflow examples for parity validation

### Testing

- Parity test timeout increased: 60s → 120s for 2B+ models
- Golden token tests: 8 test cases across 3 tokenizer families
- FFI lifecycle test: 100x create/drop cycles (no crashes)

### Changed

- **Prompt Template Auto-Detection Default** ([PR #467](https://github.com/EffortlessMetrics/BitNet-rs/pull/467)):
  - Changed auto-detection fallback from `raw` to `instruct` template for better out-of-box experience with instruction-tuned models
  - Auto-detection now logs template selection at `info` level for better visibility
  - Use `--prompt-template raw` to explicitly request raw completion behavior if needed
- **Kernel Recorder Receipt Improvements** ([PR #467](https://github.com/EffortlessMetrics/BitNet-rs/pull/467)):
  - Kernel recorder now resets before each inference turn to track per-turn kernel usage
  - Receipt kernel lists are now sorted, deduplicated, and capped at 32 entries to prevent bloat
  - Documents that receipts track coarse kernel classes (e.g., `i2s_gemv`, `tl1_lut_q`) not individual calls
- **Weight Mapper Code Quality Improvements** ([#209](https://github.com/EffortlessSteven/BitNet-rs/pull/209)):
  - **Enhanced Pick Helper Function**: Refactored `pick` helper in `weight_mapper.rs` to return both key and tensor, eliminating code duplication
  - **Streamlined Tensor Alias Resolution**: Simplified embedding and lm_head tensor alias handling with improved maintainability
  - **Code Structure Optimization**: Enhanced internal helper functions while maintaining all existing functionality and backward compatibility
  - **Comprehensive Validation**: All 62 tests in bitnet-models package pass, ensuring no functional regressions
- **Cargo Configuration Cleanup** ([#113](https://github.com/EffortlessSteven/BitNet-rs/pull/113)):
  - Remove tool-generated metadata files (`.crates.toml`, `.crates2.json`) from version control
  - Commit `Cargo.lock` files for reproducible builds across environments
  - Standardize GPU feature aliases in cargo config to use `gpu` instead of `cuda`

### Added

- **Token-Level Stop Sequences** ([PR #467](https://github.com/EffortlessMetrics/BitNet-rs/pull/467)):
  - Added `stop_token_ids` field to `GenerationConfig` for fast token-level stop checking
  - Avoids expensive string decoding for common stop tokens like LLaMA-3's `<|eot_id|>`
  - Falls back to string-based stop sequence matching for compatibility
- **Enhanced CLI Help Text Testing** ([PR #467](https://github.com/EffortlessMetrics/BitNet-rs/pull/467)):
  - Help footer tests now use direct CLI builder instead of subprocess for faster, more reliable testing
  - Exposed `build_cli()` function in `bitnet-cli` for external test use

- **CPU Forward Pass with Autoregressive Generation** ([#462](https://github.com/EffortlessSteven/BitNet-rs/issues/462)):
  - **CPU Forward Pass**: Complete autoregressive generation from BOS token through logits with device-aware CPU inference
  - **CLI Inference**: `bitnet-cli infer` command with CPU backend support and deterministic inference
  - **TL LUT Helper**: Safe `bitnet_kernels::tl_lut::lut_index()` with checked arithmetic, overflow detection, and 100% mutation testing coverage
  - **Receipt CPU Validation**: Honest compute validation with CPU quantized kernel enforcement (i2s_*, tl1_*, tl2_*) and silent CPU fallback detection
  - **Enhanced Testing**: 91% overall mutation testing score (31/31 tests passing, 20/22 mutants killed)
- **WebAssembly (WASM) Compatibility Improvements** ([#170](https://github.com/EffortlessSteven/BitNet-rs/pull/170)):
  - **Enhanced WASM Build Compatibility**: Avoiding native dependency conflicts for seamless WebAssembly compilation
  - **Updated Tokenizers Configuration**: Added `unstable_wasm` feature for proper WebAssembly support alongside existing features
  - **Fixed Workspace Dependency Management**: Consistent dependency versions across all WASM-related crates
  - **Improved Browser Compatibility**: Proper feature gating and dependency management for browser environments
  - **SIMD Intrinsic Compatibility**: Fixed AVX2 intrinsics for WebAssembly target compatibility using portable alternatives
  - **Zero Breaking Changes**: All improvements maintain full backward compatibility with existing builds
- **Enhanced GGUF Tokenizer with Optimized Byte Mapping** ([#171](https://github.com/EffortlessSteven/BitNet-rs/pull/171)):
  - **O(1) Byte Lookup Performance**: Replaced HashMap with `byte_to_id[256]` array for optimized tokenization performance
  - **Improved UTF-8 Handling**: Enhanced byte buffer management in decode operations for robust text processing
  - **BOS Token Support**: Added BOS token support to BasicTokenizer with vocab boundary checks and enhanced safety
  - **Critical SPM Compilation Fix**: Resolved compilation error in SentencePiece tokenizer that prevented `spm` feature from working
  - **Enhanced token_to_piece Functionality**: Direct byte lookup for improved token-to-text conversion performance
  - **Comprehensive Test Coverage**: Added unit tests for BOS token handling, vocab overflow protection, and enhanced tokenizer functionality
  - **Backward Compatibility**: All enhancements maintain full API compatibility with existing tokenizer implementations
- **GPU Infrastructure Foundation - CUDA Context and Module Access** ([#199](https://github.com/EffortlessSteven/BitNet-rs/pull/199)):
  - **Public CUDA Infrastructure Access**: Exposed CUDA context and module through public `context()` and `module()` accessor methods
  - **Custom Kernel Loading Foundation**: Enables loading of specialized PTX kernels for domain-specific GPU operations
  - **Advanced Memory Management Support**: CUDA context access enables custom memory pools and advanced GPU memory operations
  - **Device-Aware Launch Optimization**: Integrated `calculate_optimal_launch_params()` in matrix multiplication operations replacing hardcoded 16x16 block sizes
  - **Dead Code Elimination**: Removed `#[allow(dead_code)]` attributes from infrastructure fields, ensuring active utilization
  - **GPU Infrastructure Sequence Foundation**: First phase of three-part GPU enhancement sequence (#199 → #202 → #206)
  - **Enhanced Testing Framework**: Added comprehensive GPU infrastructure tests for context access, module loading, and optimal launch parameters
  - **Production-Ready Architecture**: Maintains backward compatibility while enabling advanced GPU programming capabilities
  - **Multi-Stream Coordination Support**: Foundation for overlapped execution and advanced GPU orchestration
  - **Performance Monitoring Integration**: Enhanced GPU operations tracking with device-specific optimization metrics
- **Enhanced GGUF Tensor Alignment Validation** ([#210](https://github.com/EffortlessSteven/BitNet-rs/pull/210)):
  - **Comprehensive Tensor Offset Validation**: All tensor offsets are validated against `general.alignment` to ensure proper memory alignment
  - **Data Section Boundary Validation**: Validates that the tensor data section starts at properly aligned boundaries
  - **Metadata Consistency Checks**: Verifies that n_dims field matches actual dimensions array length preventing parsing corruption
  - **Enhanced Error Messages**: Detailed error messages include tensor names, offsets, and alignment requirements for easier debugging
  - **Memory Safety Improvements**: Prevents out-of-bounds access with robust bounds checking and detailed tensor information
  - **Malformed GGUF Detection**: Detects corrupted or non-standard GGUF files with comprehensive validation before processing
  - **Backward Compatibility**: Enhanced validation maintains full compatibility with existing valid GGUF files
  - **Performance Impact**: Negligible performance impact while significantly improving reliability and error detection
- **Enhanced SIMD Kernels with Optimized Memory Access** ([#174](https://github.com/EffortlessSteven/BitNet-rs/pull/174)):
  - **Improved Memory Operations**: Refactored SIMD store operations in I2S quantization using cleaner `_mm_storeu_si64` and `_mm_loadu_si64` intrinsics
  - **Cross-Platform Compatibility**: Added 7 comprehensive SIMD compatibility tests ensuring consistent behavior across x86_64 and ARM64 architectures
  - **Performance Validation**: Implemented SIMD/scalar parity validation for all quantization types with comprehensive accuracy testing
  - **Architecture-Specific Testing**: Enhanced data alignment scenario testing for robust SIMD operations across different memory layouts
  - **Benchmark Infrastructure**: Added 9 specialized benchmark suites for performance comparison between SIMD implementations
  - **Microbenchmark Framework**: Comprehensive performance baseline validation with automated SIMD optimization verification
  - **Enhanced Error Handling**: Improved cross-architecture support with proper CPU feature detection and graceful scalar fallback
- **Comprehensive NaN-Safe Sampling Pipeline** ([#184](https://github.com/EffortlessSteven/BitNet-rs/pull/184)):
  - **Automatic NaN Sanitization**: Converts NaN logits to negative infinity for predictable behavior
  - **Enhanced Top-K Filtering**: Pre-filters NaN values and uses safe partial_cmp() with fallback to Ordering::Equal
  - **Robust Top-P Filtering**: Sanitizes logits before probability calculation with graceful edge case handling
  - **Safe Sorting Operations**: Prevents panics from NaN comparisons with deterministic tie-breaking
  - **Comprehensive Test Coverage**: New tests for `test_top_k_filter_with_nan`, `test_top_p_filter_with_nan`, and `test_sample_with_nan_logits`
  - **Production Reliability**: Prevents runtime crashes from model output anomalies while maintaining streaming inference
- **Enhanced Prefill Functionality for Batch Inference** ([#187](https://github.com/EffortlessSteven/BitNet-rs/pull/187)):
  - **Explicit Prefill Integration**: Added `engine.prefill()` method for explicit cache warming and latency measurement in batch inference operations
  - **Structured Performance Metrics**: Enhanced `TimingMetrics` with separate measurement for prefill, decode, tokenization, and total inference time
  - **Comprehensive Throughput Calculations**: New `ThroughputMetrics` with tokens-per-second for prefill, decode, and end-to-end performance
  - **Production-Ready Error Handling**: Robust error handling for empty tokens, invalid tokens, and context length exceeded scenarios
  - **Comprehensive Test Coverage**: 13 specialized tests (8 unit + 5 integration) covering batch prefill operations, performance consistency, and error recovery
  - **Enhanced CLI Integration**: Updated inference commands with prefill timing support and structured JSON metrics output
  - **Mock Testing Infrastructure**: Comprehensive mock model and tokenizer with realistic timing for accurate performance validation
  - **Documentation Enhancement**: Detailed inline documentation with usage examples, performance benefits, and troubleshooting guides
  - **Backward Compatibility**: Zero breaking changes to existing API while adding enhanced functionality
- **PrefillEngine Trait Abstraction** ([#139](https://github.com/EffortlessSteven/BitNet-rs/pull/139)):
  - **Clean Dependency Injection**: Added PrefillEngine trait to enable proper mocking in CLI inference tests
  - **Async Support**: Trait methods support async/await pattern matching InferenceEngine API
  - **Test Infrastructure**: MockEngine implementation for isolated unit testing of inference pipelines
  - **Backward Compatible**: InferenceEngine implements PrefillEngine with existing functionality preserved
  - **Enhanced Testability**: Enables comprehensive unit testing of CLI batch inference without external dependencies
- **Production-Ready Streaming Inference** ([#182](https://github.com/EffortlessSteven/BitNet-rs/pull/182)):
  - Real async streaming implementation using `GenerationStream` with futures and `StreamExt::next()`
  - Enhanced NaN-safe sampling operations with hardened floating-point comparisons in `top_k_filter` and `top_p_filter`
  - Accurate performance metrics collection during streaming with proper prefill execution via `engine.eval_ids()`
  - Integration tests enabled by default (removed feature gates) for comprehensive test coverage
  - Futures crate dependency reintroduced for async stream processing with `StreamExt::next()` support
  - Robust error handling with `.unwrap_or(std::cmp::Ordering::Equal)` for NaN-resilient sorting operations
- **Device-Aware GPU Quantization Support**:
  - Enhanced I2S, TL1, and TL2 quantizers with automatic GPU acceleration
  - Device-aware dequantization with intelligent CPU fallback
  - GPU memory optimization and mixed precision support
  - Comprehensive GPU vs CPU validation tests with configurable tolerance
  - Proper feature gating with `#[cfg(feature = "gpu")]` for CPU-only builds
- **Device Memory Tracking Infrastructure** ([#185](https://github.com/EffortlessSteven/BitNet-rs/pull/185)):
  - Real-time host memory monitoring via `memory-stats` crate for process-specific usage
  - System memory tracking via `sysinfo` crate with optimized refresh calls
  - Thread-safe memory statistics with Arc<Mutex<DeviceStatsInternal>> protection
  - Memory efficiency metrics and usage percentage calculations in DeviceStats
  - Comprehensive test coverage for memory tracking functionality
  - Integration with device-aware quantization for automatic memory monitoring
- **Enhanced CUDA Kernel Infrastructure**:
  - Improved CUDA kernel provider with better error handling
  - Memory management optimization with automatic leak detection
  - Performance monitoring with built-in kernel execution timing
  - Device information extraction and capability detection
  - Advanced validation framework with numerical accuracy testing
- **Documentation Quality Improvements**:
  - Fixed HTML tag warnings in API documentation
  - Enhanced quantization documentation with device-aware capabilities
  - Updated CLAUDE.md with comprehensive GPU validation commands
  - Improved inline documentation with proper backtick formatting
  - Fixed broken intra-doc links and reference formatting
  - Added comprehensive project analysis in `GOALS_VS_REALITY_ANALYSIS.md` with goal-by-goal assessment ([#152](https://github.com/EffortlessSteven/BitNet-rs/pull/152))
- **Teacher-Forcing Scoring and Perplexity Calculation** ([#134](https://github.com/EffortlessSteven/BitNet-rs/pull/134)):
  - New `score` CLI command with real teacher-forcing evaluation using inference engine
  - Device selection support (`--device cpu|cuda|metal|auto`) with automatic fallback
  - Batch processing (`--batch-size`) for improved throughput on large datasets
  - JSON output with detailed metrics (tokens, mean_nll, perplexity, latency)
  - External tokenizer support (`--tokenizer`) and token limit controls (`--max-tokens`)
  - Comprehensive CLI documentation with usage examples and troubleshooting guides
- **Python Binding Environment Documentation** ([#134](https://github.com/EffortlessSteven/BitNet-rs/pull/134)):
  - New `PYTHON_BINDING_ENVIRONMENT.md` documenting PyO3 linking requirements
  - Workspace configuration explanation for optional Python binding tests
  - System package installation instructions for Ubuntu/Debian, CentOS/RHEL/Fedora, and macOS
  - CI/CD recommendations for environment-dependent testing
- **Streaming Token ID Support** ([#107](https://github.com/EffortlessSteven/BitNet-rs/pull/107)):
  - Enhanced `StreamResponse` struct with `token_ids: Vec<u32>` field for real-time token ID access
  - Server-Sent Events (SSE) streaming endpoint with JSON token metadata at `/v1/stream`
  - Token-by-token streaming with configurable buffering and error handling
  - Comprehensive test coverage for streaming functionality and token ID accuracy
  - Updated examples and documentation to demonstrate token ID streaming usage
- **2D Convolution Operations Support** ([#165](https://github.com/EffortlessSteven/BitNet-rs/pull/165)):
  - Complete 2D convolution implementation with `conv2d` and `conv2d_quantized` functions
  - Support for NCHW input format and OIHW weight format with configurable parameters
  - Stride, padding, and dilation operations for flexible convolution configurations
  - Quantized convolution with I2S, TL1, and TL2 quantization types
  - On-the-fly dequantization with per-channel scaling factors
  - PyTorch reference testing framework for numerical correctness validation
  - Comprehensive unit tests covering basic functionality, edge cases, and error handling
  - Integration with existing bitnet-rs kernel architecture and error handling patterns
- **GGUF Validation API**:
  - Fast 24-byte header-only validation without loading full model
  - Production-ready parser with typed errors and non-exhaustive enums
  - `compat-check` CLI command with stable exit codes for CI automation
  - `--strict` flag for enforcing version and sanity checks
  - Early validation in engine before heavy memory allocations
  - JSON output for programmatic validation scripts
- **GGUF KV Metadata Reader**:
  - Read and inspect GGUF key-value metadata without full model loading
  - Support for all GGUF value types (except arrays, deferred)
  - `--show-kv` flag in CLI to display model metadata
  - `--kv-limit` flag to control number of displayed KV pairs
  - JSON output includes metadata when `--show-kv` is used
  - Safety limits to prevent excessive memory usage

### Fixed

- **Code Quality and Security Improvements**:
  - Fixed critical PyO3 security vulnerability (RUSTSEC-2025-0020)
  - Resolved 45+ clippy warnings across workspace for better code quality
  - Updated dependencies (atty→is-terminal, removed wee_alloc)
  - Enhanced type safety and error handling documentation
  - Resolved all clippy warnings across the codebase with proper type improvements
  - Enhanced kernel validation system with improved error handling and performance metrics
  - Fixed FFI bridge test tolerance calculations for accurate migration recommendations
  - Improved universal tokenizer documentation and error handling
  - Enhanced model loading with better GGUF handling and error propagation
  - Standardized code formatting and documentation strings throughout
- **bitnet-server Build Issues**:
  - Restored Git metadata support using vergen-gix v1.x
  - Moved runtime dependencies from build-dependencies to correct section
  - Made health endpoint robust with option_env! for graceful fallbacks
- **Memory Safety and Environment Variable Handling** ([#181](https://github.com/EffortlessSteven/BitNet-rs/pull/181)):
  - Enhanced BitNetTensor with proper device tracking and memory leak prevention
  - Replaced unsafe `Box::leak()` with safe `OnceLock<Vec<f32>>` caching for host data
  - Safe type conversion using `bytemuck::cast_slice` instead of manual transmutation
  - Removed redundant `DeviceType` enum in favor of unified `Device` enum
  - Rust 2024 compliance: marked environment variable manipulations as `unsafe`
  - Improved Clone trait implementation for BitNetTensor with proper data handling
- **IQ2_S FFI Layout Enhancement and Parity Testing** ([#142](https://github.com/EffortlessSteven/BitNet-rs/pull/142)):
  - Enhanced `BlockIq2S` struct with perfect GGML `block_iq2_s` layout compatibility (82 bytes)
  - Added compile-time size and alignment assertions for layout parity verification
  - Enabled previously ignored `iq2s_rust_matches_ffi` parity test with precise element-by-element comparison
  - Replaced hardcoded block size with compile-time `size_of::<BlockIq2S>()` for consistency
  - Zero API breaking changes, maintaining full backward compatibility
- **IQ2_S Quantization Layout Alignment** ([#132](https://github.com/EffortlessSteven/BitNet-rs/pull/132)):
  - Updated IQ2_S block layout from 66B to 82B to match GGML specification exactly
  - Corrected QMAP values from `[-2,-1,0,1]` to `[-2,-1,1,2]` eliminating zero mapping
  - Updated test expectations to match new quantization mapping pattern
  - Ensures bit-exact compatibility between Rust and GGML backends
  - Fixed unsafe code warnings with proper unsafe blocks for Rust 2024 compliance
- **Security Vulnerability Resolution** ([#107](https://github.com/EffortlessSteven/BitNet-rs/pull/107)):
  - Updated PyO3 from v0.21.2 to v0.25.1 to resolve CVE-2024-9979 buffer overflow vulnerability
  - Updated related Python binding dependencies (numpy, pyo3-async-runtimes) for compatibility
  - Enhanced security posture of Python bindings and server components
- **FFI Safety and Validation Improvements**:
  - Enhanced FFI functions with `unsafe fn` signatures for Rust 2024 safety compliance
  - Fixed clippy warnings in test infrastructure and removed unneeded unit expressions
  - Added proper unsafe blocks for raw pointer operations in C API layer
  - Maintained full API compatibility with existing C clients while improving memory safety
- **CUDA Device Information Querying** (PR #102):
  - Real device property querying using cudarc's CUdevice_attribute API
  - Comprehensive device information extraction (compute capability, memory, multiprocessor count)
  - Enhanced error handling for device query failures
  - Enhanced test coverage for device information validation

### Enhanced

- **Weight Mapper Model Compatibility Validation** ([#144](https://github.com/EffortlessSteven/BitNet-rs/pull/144)):
  - Enhanced `validate_model_compatibility()` to use weight mapper for GGUF tensor validation
  - Replaced TODO placeholder with actual GGUF parsing and tensor name mapping
  - Added detection of unmapped tensors with detailed error reporting and debugging metrics
  - Comprehensive test coverage with fixture support for both success and corruption scenarios
  - Improved fallback handling in universal tokenizer with SpmTokenizer typo fixes
- **GPU Kernel Refactoring** (PR #108):
  - **CUDA Implementation**: Enhanced cudarc 0.17 API compatibility with performance tracking and error handling
  - **Memory Management**: Implemented OptimizedMemoryPool with device-specific allocation, caching, leak detection, and device_id() access method
  - **Mixed Precision Infrastructure**: Added PrecisionMode support for FP16/BF16 operations on modern GPUs
  - **Comprehensive Validation**: GpuValidator with numerical accuracy, performance benchmarking, and memory health checks
  - **FFI Bridge Improvements**: Enhanced C++ kernel integration with feature gating and performance comparison tools
  - **Device Information**: Detailed CudaDeviceInfo with compute capability, memory, and precision support detection
  - **Launch Parameter Optimization**: Dynamic kernel configuration based on device capabilities and workload characteristics
- **Documentation Synchronization** (Post-PR #113):
  - Updated all documentation to use standardized `gpu` feature flag instead of `cuda`
  - Maintained backward compatibility by documenting `cuda` as an alias for `gpu`
  - Synchronized build commands across CLAUDE.md, README.md, FEATURES.md, and GPU setup guides
  - Updated cargo aliases in `.cargo/config.toml` to use `gpu` feature consistently
  - Enhanced lockfile tracking for reproducible builds (Cargo.lock now versioned)
  - Enhanced Diátaxis framework compliance with clearer tutorial/reference categorization

  - Updated all documentation to use standardized `gpu` feature flag instead of `cuda`
  - Maintained backward compatibility by documenting `cuda` as an alias for `gpu`
  - Synchronized build commands across CLAUDE.md, README.md, FEATURES.md, and GPU setup guides
  - Updated cargo aliases in `.cargo/config.toml` to use `gpu` feature consistently
  - Enhanced lockfile tracking for reproducible builds (Cargo.lock now versioned)
  - Enhanced Diátaxis framework compliance with clearer tutorial/reference categorization
- **CI/Docker Git Metadata Support**:
  - Added Git metadata injection in GitHub Actions CI
  - Updated Dockerfile with VCS build args for metadata without .git
  - Added docker-build.sh script for easy builds with Git metadata
  - Added OCI standard labels for container registries
  - Environment variable overrides for deterministic builds

## [0.3.0] - 2025-01-03

### Added (0.3.0)

- **IQ2_S Quantization Support**:
  - Native Rust implementation with optimized dequantization
  - FFI backend via GGML for compatibility
  - Comprehensive unit tests and validation scripts
  - Backend parity testing between FFI and native implementations
- **Enhanced Test Suite**:
  - Feature-gated test configuration system
  - Improved fixture management with conditional compilation
  - Comprehensive integration test coverage
  - CI-friendly reporting with multiple output formats
- **Comprehensive CI Validation Framework**:
  - 8-gate acceptance system with JSON-driven detection
  - Distinct exit codes (0-10) for precise CI triage
  - Performance ratio gates with baseline comparisons
  - Deterministic execution environment (SEED=42, THREADS=1)
  - Portable memory profiling with GNU time/gtime
- **Score/Perplexity Subcommand**:
  - Teacher-forcing perplexity calculation skeleton
  - JSON output with tokenizer origin tracking
  - Support for external SentencePiece models
  - Ready for logits API integration
- **Strict Mode Enforcement**:
  - Zero unmapped tensors requirement
  - SentencePiece tokenizer validation
  - BOS token policy enforcement
  - Deterministic tie-breaking (lowest ID)
- Cross-validation framework for numerical accuracy testing
- Performance benchmarking suite with automated regression detection

### Fixed (0.3.0)

- Model loading edge cases and error handling improvements
- Memory management optimizations for large models
- Cross-platform compatibility improvements

### Changed (0.3.0)

- Improved API ergonomics and error messages
- Enhanced documentation with more examples
- Streamlined build process and dependency management

## [0.2.0] - 2024-12-15

### Added (0.2.0)

- Basic quantization support (I2_S, TL1, TL2)
- GGUF format compatibility
- Python bindings with PyO3
- C API for llama.cpp compatibility
- Streaming inference capabilities
- Initial CUDA support

### Fixed (0.2.0)

- Memory safety improvements
- Performance optimizations
- Cross-validation accuracy

## [0.1.0] - 2024-11-01

### Added (0.1.0)

- Initial release
- Basic BitNet model loading and inference
- CPU-only quantization support
- Core API design and architecture
