# #[ignore] Tests Audit Report

## Executive Summary

This audit comprehensively catalogs all **240 #[ignore] tests** in the BitNet-rs codebase, categorizing them into:

1. **Properly Classified (56 tests, 23.3%)**: Tests with explicit issue references
2. **Unclassified (184 tests, 76.7%)**: Tests without issue references - **Primary focus of this audit**
3. **Resolved Issues (0 tests)**: Tests blocking issues that have been fixed

### Key Finding

**29+ unclassified tests lack any tracking reference**, making it difficult to understand their status, priority, and resolution criteria. This audit recommends systematic reclassification and tracking.

---

## Part 1: Properly Classified Tests (56 tests)

### Issue #254: Shape Mismatch in Layer-Norm (10 tests)

**Status**: In analysis phase  
**Impact**: Blocks real inference tests; affects multiple architectures

| File | Test | Reason |
|------|------|--------|
| `crates/bitnet-inference/tests/test_real_vs_mock_comparison.rs:16` | test_real_inference_vs_mock | Shape mismatch in layer-norm - needs investigation |
| `crates/bitnet-inference/tests/issue_254_ac4_receipt_generation.rs:22` | test_receipt_generation_framework | Receipt generation unimplemented |
| `crates/bitnet-inference/tests/test_real_inference.rs:21` | test_real_inference_path | Shape mismatch in layer-norm |
| `crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs:24-290` | test_greedy_determinism_* (7 tests) | TDD placeholder - Deterministic inference (50-100+ forward passes) |

**Recommended Action**: Link all 7 deterministic generation tests to #254; create subtask for shape mismatch root cause.

---

### Issue #260: Mock Elimination Not Complete (11 tests)

**Status**: Awaiting refactoring  
**Impact**: Prevents full transition to real inference paths (~15 blocked end-to-end tests)

| File | Test | Reason |
|------|------|--------|
| `crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs:179` | test_cpu_simd_kernel_integration | TDD placeholder - quantized_matmul not yet implemented |
| `crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs:314` | test_tl2_optimization_avx | TDD placeholder - TL2 4096-entry table unimplemented |
| `crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs:566-780` | (4 FFI scaffold tests) | TDD scaffold - CppQuantizationBridge/FfiMemoryManager implementation needed |
| `crates/bitnet-common/tests/issue_260_strict_mode_tests.rs:112-189` | test_strict_mode_* (2 tests) | TDD placeholder - Strict mode validation behavior unimplemented |
| `crates/bitnet-inference/tests/issue_260_mock_elimination_inference_tests.rs:524-1038` | (5 perf/CI detector tests) | TDD placeholder - CI mock detector, perf regression, benchmarks unimplemented |

**Recommended Action**: Create milestone to unblock all #260 tests by implementing missing components.

---

### Other Issues (35 tests)

| Issue | Count | Key Tests |
|-------|-------|-----------|
| #159 (GGUF loader) | 20 | gguf_weight_loading_*.rs - TDD placeholders for real implementation |
| #248 (NN scaffolding) | 8 | neural_network_test_scaffolding.rs - Quantized linear, attention, generation |
| #432 (GPU flaky) | 3 | gpu/tests.rs - CUDA context cleanup issues |
| #159, #462, other | 4 | Miscellaneous quantization and kernel tests |

---

## Part 2: Unclassified Tests (184 tests) - DETAILED AUDIT

### Category 1: Requires Model/Fixture (29 tests)

**Problem**: Tests depend on external model files not in repository.  
**Current Solution**: Tests are marked #[ignore] and manually run with `BITNET_GGUF` env var.

| Subcategory | Count | Files | Recommendation |
|-------------|-------|-------|-----------------|
| Requires GGUF model | 17 | bitnet-cli tests, bitnet-inference greedy_decode, template tests | Gate on CI-only or create fixture provisioning |
| Requires CLI binary | 4 | intelligibility_smoke.rs | Add CI job for integration testing |
| Requires model lock file | 2 | crossval/smoke.rs | Document lock file generation process |
| Requires model download | 6 | documentation_validation.rs, xtask tests | Add to CI model provisioning pipeline |

**Sample Tests**:
- `crates/bitnet-cli/tests/intelligibility_smoke.rs:424` - "requires model file and CLI binary"
- `crates/bitnet-inference/tests/greedy_decode_parity.rs:214-485` - "requires model file" (5 tests)
- `xtask/tests/documentation_validation.rs:224,359` - "requires model download and execution"

**Recommended Action**:
```yaml
# Create shared fixture strategy:
Option A: Gate on environment - use BITNET_GGUF to enable
Option B: CI-only - tag with #[cfg(test_enable_model_fixtures)]
Option C: Download in CI - add model provisioning step to .github/workflows/
```

---

### Category 2: Fixture Needed (13 tests)

**Problem**: Tests require specific test fixtures that don't exist or need infrastructure setup.

| Fixture Type | Count | Key Tests |
|--------------|-------|-----------|
| Receipt fixtures | 2 | ci_parity_smoke_test.rs:202,231 |
| FFI fixtures | 3 | ffi_build_tests.rs - "Requires FFI implementation fixture not yet available" |
| Log capture mechanism | 2 | tokenizer_vocab_size.rs:109, kv_cache_validation.rs:162,269 |
| Tokenizer fixtures | 2 | tokenizer_vocab_size.rs:48,212 |
| C++ reference + GGUF | 3 | test_ac5_production_readiness.rs |
| Other | 1 | sp_roundtrip.rs |

**Recommended Action**:
- Create `tests/fixtures/` directory structure
- Document fixture provisioning in `CONTRIBUTING.md`
- Add fixture generation scripts for FFI, C++, and GGUF

---

### Category 3: Network/Auth Dependent (15 tests)

**Problem**: Tests require internet access, authentication tokens, or external services.

| Dependency | Count | Files | CI Status |
|------------|-------|-------|-----------|
| HF_TOKEN (Hugging Face) | 4 | xtask/tests/ci_integration_tests.rs, tokenizer_subcommand_tests.rs | Gated on CI secret |
| Network access | 7 | xtask/tests - download, network calls | Requires internet access |
| GitHub API | 1 | tests/issue_465_ci_gates_tests.rs | Requires GH branch protection access |
| cargo binary | 1 | tests/readme_examples.rs | Requires cargo in PATH |

**Sample Tests**:
- `xtask/tests/ci_integration_tests.rs:29` - "Requires HF_TOKEN secret in CI"
- `xtask/tests/tokenizer_subcommand_tests.rs:26,78,159` - "Requires network access"
- `tests/issue_465_ci_gates_tests.rs:28` - "Requires GitHub API access"

**Recommended Action**:
- Tag with `#[cfg(test_requires_network)]` or `#[cfg(ci)]`
- Document which CI secrets are needed
- Provide mock implementations for local development

---

### Category 4: GPU/CUDA Environment Required (10 tests)

**Problem**: Tests require CUDA installation and GPU hardware.

| File | Tests | Current Status |
|------|-------|-----------------|
| `crates/bitnet-kernels/src/gpu/cuda.rs:539,582` | 2 | Manual GPU testing |
| `crates/bitnet-kernels/src/gpu/benchmark.rs:299` | 1 | Benchmark (GPU required) |
| `crates/bitnet-kernels/src/gpu/validation.rs:562` | 1 | Validator tests |
| `crates/bitnet-kernels/tests/gpu_integration.rs:20,166,183,225` | 4 | Integration tests |
| `crates/bitnet-kernels/tests/gpu_quantization.rs:139,179,265,317,349` | 5 | Quantization tests |

**Recommended Action**:
- Replace #[ignore] with proper feature gate: `#[cfg(any(feature = "gpu", feature = "cuda"))]`
- Move GPU tests to separate job in CI
- Use `skip_if_no_cuda()` helper instead of #[ignore]

---

### Category 5: Performance/Timeout Issues (6 tests)

**Problem**: Tests are slow, hang, or timing-sensitive.

| Test | Issue | Duration |
|------|-------|----------|
| `crates/bitnet-kernels/tests/cpu_simd_receipts.rs:111,155` | Hanging test - investigating | Unknown |
| `crates/bitnet-inference/tests/issue_254_ac4_receipt_generation.rs:100` | Slow integration path | ~300s |
| `crates/bitnet-inference/tests/ac3_autoregressive_generation.rs:207` | Slow: 25 model generations | Unknown |
| `crates/bitnet-inference/tests/ac3_autoregressive_generation.rs:314` | Slow: 52 model generations | Unknown |
| `crates/bitnet-inference/tests/ac3_autoregressive_generation.rs:414` | Slow: 76 model generations | Unknown |
| `crates/bitnet-server/tests/concurrent_load_tests.rs:313` | Flaky perf test in CI | Timing-sensitive |

**Recommended Action**:
```rust
// For hanging tests:
#[test]
#[timeout(5000)] // 5-second timeout
#[ignore] // Issue #XXX: Investigate timeout
fn test_simd_vector_add() { /* ... */ }

// For slow tests:
#[test]
#[ignore = "Slow test (~300s): run manually with RUN_SLOW_TESTS=1"]
fn test_slow_integration() { /* ... */ }
```

---

### Category 6: Work in Progress (WIP) (10 tests)

**Problem**: Tests are scaffolding for incomplete features.

| File | Count | Feature |
|------|-------|---------|
| `crates/bitnet-inference/tests/full_engine_compilation_test.rs` | 8 | Full-engine implementation (WIP) |
| `crates/bitnet-inference/tests/type_exports_test.rs:102` | 1 | Full-engine exports |
| `crates/bitnet-inference/tests/ac9_comprehensive_integration_testing.rs:22` | 1 | AC9 comprehensive integration |

**Sample**:
```rust
#[ignore = "WIP: full-engine implementation in progress"]
fn test_full_inference_engine() { /* ... */ }
```

**Recommended Action**:
- Link to feature branch or create tracking issue
- Add target completion date
- Convert to unit tests where possible

---

### Category 7: Implementation Pending (18 tests)

**Problem**: Tests have placeholder implementations with no clear blocking issue.

| File | Tests | Pattern |
|------|-------|---------|
| `crates/bitnet-inference/tests/template_detection.rs` | 7 | "implementation pending: verify detection affects formatting", etc. |
| `crates/bitnet-inference/tests/ac1_quantized_linear_layers.rs:142,236` | 2 | "TODO: Update to use QuantizedLinear::new_tl1/tl2()" |
| `crates/bitnet-cli/tests/cli_args_aliases.rs:89,206` | 2 | "implementation pending: verify help text" |
| Other | 7 | Various pending features |

**Recommended Action**:
- Create issues for each pending implementation
- Link #[ignore] to issue numbers
- Set priority and assignment

---

### Category 8: Integration Tests (9 tests)

**Problem**: Tests are marked integration-only but lack clear CI placement.

| Test | Scope | CI Job? |
|------|-------|---------|
| `crossval/tests/parity_receipts.rs:488` | Filesystem access | Unknown |
| `xtask/tests/ci_parity_smoke_test.rs:52,90` | Parity smoke script | GitHub Actions |
| `xtask/tests/ci_parity_smoke_test.rs:128` | Strict mode implementation | GitHub Actions |
| `crates/bitnet-inference/tests/kv_cache_validation.rs:230,296` | KVCache/attention integration | Not yet |

**Recommended Action**:
- Clarify scope: unit test vs. integration test vs. e2e
- Create separate `tests/integration/` directory structure
- Document which CI jobs run which tests

---

### Category 9: Miscellaneous/Vague (71 tests)

**Problem**: 71 tests with unclear or non-standard ignore reasons.

**Examples**:
- `xtask/src/main.rs:5564` - Just `#[ignore]` with no reason
- `crossval/tests/smoke.rs:35` - "requires CROSSVAL_GGUF and C++ libraries" (unclear requirement)
- `crates/bitnet-kernels/tests/issue_462_tl_lut_tests.rs:153` - "Requires QuantizedLinear TL1/TL2 integration with lut_index helper"
- `crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs:566` - "TDD scaffold: requires CppQuantizationBridge and FfiQuantizer implementation"
- Multiple `issue_254_ac*_deterministic_generation.rs` tests with just `#[ignore]` and documentation in comments

**Recommended Action**: Create comprehensive audit with one-on-one owner review.

---

## Part 3: Resolved Issues

### Issue #439: Feature Gate Consistency - ✅ RESOLVED (PR #475)

**Status**: Merged to main

- GPU/CPU feature predicates unified
- All device selection and fallback tests validated
- No remaining #[ignore] tests linked to #439

**Legacy Impact**: This resolution enables removal of some device-conditional tests that were previously marked #[ignore].

---

## Part 4: Audit Recommendations

### Immediate Actions (Priority: HIGH)

1. **Create Issue Tracking System**
   ```
   All #[ignore] tests must have one of:
   - Issue reference (#254, #260, etc.)
   - CI gate explanation (@ci-team review required)
   - Fixture dependency documented (with provisioning plan)
   - Explicit owner assignment
   ```

2. **Fix 29 Unclassified Model/Fixture Tests**
   - Assign to Test Infra team
   - Create shared model fixture strategy
   - Document in CONTRIBUTING.md

3. **Eliminate 71 "Miscellaneous" Ignores**
   - Audit each one with original author
   - Add issue references or move to proper category
   - Target: 0 miscellaneous ignores

### Short-Term (1-2 sprints)

4. **Replace Timeout-Based #[ignore] with Proper Gates**
   - GPU tests: Use `#[cfg(any(feature = "gpu", feature = "cuda"))]`
   - Network tests: Use `#[cfg(test_requires_network)]` or gating
   - Model tests: Gate on environment variable or CI-only

5. **Create Test Configuration File**
   ```yaml
   # .cargo/test-config.yml
   categories:
     model_fixtures:
       gate: BITNET_GGUF
       ci_enabled: true
       owner: test-infra
     gpu_tests:
       gate: "feature=gpu"
       ci_enabled: true
       owner: gpu-team
     slow_tests:
       gate: RUN_SLOW_TESTS=1
       ci_enabled: false
       owner: engineering
   ```

### Long-Term (Post-MVP)

6. **Reduce #[ignore] Count by 50%**
   - Implement missing fixtures
   - Resolve blocking issues (#254, #260, etc.)
   - Convert scaffolding tests to real implementations

7. **Establish #[ignore] Policy**
   - Maximum lifecycle: 2 sprints without blocking issue reference
   - Regular audits: Monthly categorization sweep
   - Test triage: Weekly review of new ignores

---

## Appendix: Quick Reference Tables

### Tests by Owner (Recommended Assignment)

| Owner | Category | Count | Priority |
|-------|----------|-------|----------|
| Test Infra | Fixtures, CI gating | 35 | HIGH |
| GPU Team | CUDA/GPU tests | 10 | MEDIUM |
| Feature Owners | Implementation pending | 18 | MEDIUM |
| Performance Team | Slow/timeout tests | 6 | HIGH |
| CI/Automation | Network/auth tests | 15 | MEDIUM |
| Issue #254 Owner | Shape mismatch blocking | 10 | HIGH |
| Issue #260 Owner | Mock elimination blocking | 11 | HIGH |

### Tests by Crate

| Crate | Count | Key Issues |
|-------|-------|-----------|
| `xtask` | 35+ | Network, CI integration, model download |
| `bitnet-cli` | 25+ | Model fixtures, implementation pending |
| `bitnet-inference` | 60+ | Model fixtures, WIP features, slow tests |
| `bitnet-kernels` | 30+ | GPU/CUDA, performance, flaky |
| `bitnet-models` | 25+ | Fixtures, property tests, weight loading |
| Other | 15+ | Misc. tests and utilities |

---

## Checklist for Implementation

- [ ] Create GitHub issue per unclassified test category
- [ ] Assign owners to each category
- [ ] Document fixture provisioning strategy
- [ ] Replace #[ignore] timeouts with proper feature gates
- [ ] Add to CI: model fixture setup and network gate variables
- [ ] Monthly audit: Categorize new #[ignore] tests within 1 week of merge
- [ ] Quarterly review: Reduce #[ignore] count by 10% per quarter (target: <120 by end of year)

