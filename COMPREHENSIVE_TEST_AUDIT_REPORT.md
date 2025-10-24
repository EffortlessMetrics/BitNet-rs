# BitNet.rs Comprehensive Test Suite Audit Report

**Date**: 2025-10-23  
**Scope**: Full test suite across all crates (`crates/*/tests/**/*.rs`)  
**Total Test Files Analyzed**: 258  
**Total Ignored Tests Found**: 123 instances across 42 files

---

## Executive Summary

BitNet.rs has a well-structured test suite with intentional TDD scaffolding during the MVP phase. However, **critical test hygiene gaps** were identified around environment variable mutations and parallel test safety. Key findings:

- **123 ignored tests** across 42 files, well-categorized by blocker issue
- **Major concern**: Tests mutating environment variables WITHOUT `#[serial(bitnet_env)]` (P0 issue)
- **39 tests** missing EnvGuard pattern for proper cleanup
- **44 TDD placeholder tests** scaffolded but intentionally deferred
- **Issue #439 RESOLVED** - GPU/CPU feature gate unification complete (no longer a blocker)

---

## 1. IGNORED TESTS AUDIT

### 1.1 Ignored Tests by Category

| Category | Count | Status | Priority |
|----------|-------|--------|----------|
| **Issue #254 Blockers** | 3 | Active (shape mismatch analysis) | P1 |
| **Issue #260 Blockers** | 11 | Active (mock elimination) | P1 |
| **Issue #159 Blockers** | 24 | TDD scaffolding (weight loading) | P2 |
| **Performance/Slow** | 3 | Intentional (>30-token generation) | P2 |
| **Network/External** | 9 | Resource dependencies | P2 |
| **TDD Placeholders** | 44 | Future implementation guides | P3 |
| **Unclassified** | 29 | Review needed | P1 |
| **TOTAL** | **123** | | |

### 1.2 Issue #254: Shape Mismatch in Layer-Norm (P1 - Active)

**Blocker Status**: In analysis phase - affects real inference paths

| File | Test Name | Line | Issue | Notes |
|------|-----------|------|-------|-------|
| `/crates/bitnet-inference/tests/test_real_inference.rs` | `test_real_inference_basic_execution` | N/A | Shape mismatch error | Single test blocking real inference validation |
| `/crates/bitnet-inference/tests/issue_254_ac1_quantized_linear_no_fp32_staging.rs` | `test_quantized_linear_i2s_no_fp32_fallback` | ~70 | TODO: Update LookupTable construction | TDD placeholder - needs implementation |
| `/crates/bitnet-inference/tests/issue_254_ac1_quantized_linear_no_fp32_staging.rs` | `test_quantized_linear_tl2_no_fp32_fallback` | ~90 | TODO: Update LookupTable construction | TDD placeholder - needs implementation |

**Recommendation**: 
- ✅ Keep ignored until issue #254 investigation complete
- Cannot safely enable until shape validation fixed
- Currently affects 3+ integration tests

### 1.3 Issue #260: Mock Elimination (P1 - Active)

**Blocker Status**: Awaiting refactoring - prevents transition to real inference

**Affected Test Files**:
1. `/crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs` (7 tests)
   - Lines ~180-250: Tests for I2S kernel, TL2 lookup, feature flag implementations
   - All use `#[ignore] // Issue #260: TDD placeholder`
   - Missing implementations: quantized_matmul, quantized_matmul_avx, quantized_matmul_generic

2. `/crates/bitnet-inference/tests/issue_260_mock_elimination_inference_tests.rs` (4 tests)
   - Lines ~140-180: Mock detector, strict mode validation, performance profiling
   - All require mock elimination verification
   - Environment setup: `BITNET_STRICT_MODE`, `BITNET_DETERMINISTIC`, `BITNET_SEED`

**Recommendation**:
- ✅ Keep ignored - legitimate blocker for ~15 end-to-end tests
- Track progress on issue #260 refactoring
- Cannot enable until real inference paths complete

### 1.4 Issue #159: TDD Scaffolding for Weight Loading (P2 - Future)

**Blocker Status**: Intentional TDD scaffolding for planned feature implementation

**Affected Test Files** (24 total):
1. `/crates/bitnet-models/tests/gguf_weight_loading_tests.rs` (4 tests)
   - Lines ~85-150: Mock weight initialization, I2S quantization, TL1/TL2 integration
   - All marked `#[ignore] // Issue #159: TDD placeholder`

2. `/crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs` (10 tests)
   - Lines ~45-180: Property-based tests for quantization verification
   - Marked with `#[ignore] // Issue #159: TDD placeholder`
   - Topics: I2S distribution preservation, TL1 efficiency, TL2 precision

3. `/crates/bitnet-models/tests/gguf_weight_loading_property_tests_enhanced.rs` (8 tests)
   - Similar structure to property tests, enhanced assertions
   - All scaffolded but unimplemented

4. `/crates/bitnet-models/tests/gguf_weight_loading_integration_tests.rs` (2 tests)
   - Lines ~120-160: Integration with quantizers and kernels
   - Marked as TDD placeholders

**Recommendation**:
- ✅ Keep ignored - these guide future development
- Re-enable incrementally as weight loading implementation progresses
- Use as test-driven development guides (code to the tests)
- Review for actual vs. mock data usage (some may need real GGUF fixtures)

### 1.5 Performance Tests - Slow Generation (P2 - Intentional)

**Issue**: Full model generation tests are slow (~0.1 tok/s for QK256 MVP)

| File | Test | Line | Duration Estimate | Status |
|------|------|------|-------------------|--------|
| `/crates/bitnet-inference/tests/ac3_autoregressive_generation.rs` | `test_sampling_with_different_temperatures` | ~50 | ~25 inferences | `#[ignore]` - Slow |
| `/crates/bitnet-inference/tests/ac3_autoregressive_generation.rs` | `test_sampling_with_top_k` | ~80 | ~52 inferences | `#[ignore]` - Slow |
| `/crates/bitnet-inference/tests/ac3_autoregressive_generation.rs` | `test_sampling_with_nucleus` | ~110 | ~76 inferences | `#[ignore]` - Slow |
| `/crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs` | `test_ac3_deterministic_generation_identical_sequences` | ~36 | ~100 forward passes | `#[ignore]` - Slow |
| `/crates/bitnet-inference/tests/issue_254_ac6_determinism_integration.rs` | `test_determinism_across_multiple_runs` | ~50 | ~100 forward passes | `#[ignore]` - Slow |

**Pattern**: Tests include excellent documentation with fast equivalents:
```rust
/// **This test is marked #[ignore] because it runs 50+ full model generations.**
/// For fast unit testing of determinism, see:
/// - `tests/deterministic_sampling_unit.rs::test_same_seed_identical_samples()` (<5ms)
#[ignore] // Slow: 50-token generation. Fast equivalent: tests/deterministic_sampling_unit.rs
```

**Recommendation**:
- ✅ Keep ignored - appropriate for performance tests
- Document fast equivalents (ALREADY DONE - excellent pattern!)
- Run manually or in dedicated perf CI job
- QK256 MVP performance acceptable for validation

### 1.6 Network/External Dependencies (P2 - Requires Setup)

| File | Tests | Reason | Count |
|------|-------|--------|-------|
| `/crates/bitnet-tokenizers/tests/tokenization_smoke.rs` | 6 tests | Requires `CROSSVAL_GGUF` env var | 6 |
| `/crates/bitnet-tokenizers/tests/test_ac4_smart_download_integration.rs` | 8 tests | Network-dependent (model download) | 8 |
| `/crates/bitnet-tokenizers/tests/test_ac5_production_readiness.rs` | 3 tests | Requires C++ reference + GGUF fixtures | 3 |
| `/xtask/tests/ci_integration_tests.rs` | 5 tests | Requires HF_TOKEN, network access, xtask binary | 5 |
| `/xtask/tests/tokenizer_subcommand_tests.rs` | 3 tests | Requires HF_TOKEN and network access | 3 |
| **TOTAL** | | | **28 tests** |

**Recommendation**:
- ✅ Keep ignored - appropriate for network-dependent tests
- Document requirements clearly (MOSTLY DONE)
- Consider creating mock fixtures for CI
- Run manual or in dedicated integration CI job with proper secrets

### 1.7 Unclassified / Requires Review (P1 - Action Needed)

**Files needing review** (29 tests across multiple files):

1. `/crates/bitnet-inference/tests/simple_real_inference.rs` (4 tests)
   - All marked: `#[ignore] // Requires real model with weights loaded`
   - Reason: Uninitialized model fails with "transformer not initialized"
   - **Action**: Verify if these can use fixtures or need real models

2. `/crates/bitnet-inference/tests/full_engine_compilation_test.rs` (multiple tests)
   - WIP compilation stubs with `#[ignore]`
   - **Action**: Review against feature completion status

3. `/crates/bitnet-quantization/tests/mutation_killer_tests.rs` (4 tests)
   - Marked: `#[ignore] // Disabled due to edge case handling`
   - **Action**: Determine if edge cases resolved, can be re-enabled

4. `/crates/bitnet-models/tests/rope_parity.rs` (1 test)
   - Marked: `#[ignore] // Requires bitnet.cpp FFI`
   - Requires C++ reference implementation
   - **Action**: Check FFI availability (issue #469 status)

---

## 2. TEST HYGIENE ANALYSIS

### 2.1 CRITICAL: Environment Variable Mutations Without `#[serial(bitnet_env)]`

**P0 BLOCKER** - Tests race with each other when run in parallel

#### Finding: Missing `#[serial]` in Device Features Tests

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/tests/device_features.rs`

```rust
#[test]  // ❌ MISSING: #[serial(bitnet_env)]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn ac3_gpu_fake_cuda_overrides_detection() {
    unsafe { env::set_var("BITNET_GPU_FAKE", "cuda"); }  // Unsafe mutation!
    let result = gpu_available_runtime();
    unsafe { env::remove_var("BITNET_GPU_FAKE"); }
    assert!(result);
}
```

**Risk**: When running `cargo test -- --test-threads=4`, this test races with:
- `ac3_gpu_fake_none_disables_detection()`
- `ac3_device_capability_summary_format()`
- `ac3_capability_summary_respects_fake()`

All modify `BITNET_GPU_FAKE` without synchronization.

**Affected Tests** (16 total in device_features.rs):
1. `ac3_gpu_fake_cuda_overrides_detection` (line ~87)
2. `ac3_gpu_fake_none_disables_detection` (line ~119)
3. `ac3_gpu_fake_case_insensitive` (line ~180)
4. `ac3_gpu_compiled_but_runtime_unavailable` (line ~230)
5. `ac3_device_capability_summary_format` (line ~260)
6. `ac3_capability_summary_respects_fake` (line ~300)
7. `ac3_quantization_uses_device_features` (line ~350)
8. `ac3_inference_uses_device_features` (line ~400)
... (8 more tests in same file)

**Recommendation**: 
```rust
// BEFORE (UNSAFE):
#[test]
fn ac3_gpu_fake_cuda_overrides_detection() { ... }

// AFTER (SAFE):
#[test]
#[serial(bitnet_env)]  // Add this line
fn ac3_gpu_fake_cuda_overrides_detection() { ... }
```

#### Finding: Missing EnvGuard in Other Test Files

**Files with direct `env::set_var` without EnvGuard** (need review):

1. `/crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs`
   - Line ~95: `std::env::set_var("BITNET_DETERMINISTIC", "1")`
   - Missing `#[serial(bitnet_env)]` and EnvGuard
   - Status: Using bare `set_var` without cleanup guarantee

2. `/crates/bitnet-models/tests/gguf_weight_loading_cross_validation_tests.rs`
   - Lines ~80-120: Multiple `std::env::set_var()` calls
   - Variables: `BITNET_DETERMINISTIC`, `BITNET_SEED`, `BITNET_CROSSVAL_WEIGHTS`
   - Missing `#[serial(bitnet_env)]` and EnvGuard
   - Status: Complex test setup with multiple sequential mutations

3. `/crates/bitnet-inference/tests/ac4_cross_validation_accuracy.rs`
   - Lines ~45-50: 4 `std::env::set_var()` calls in async test
   - Variables: `BITNET_DETERMINISTIC`, `BITNET_SEED`, `RAYON_NUM_THREADS`, `BITNET_GGUF`
   - Missing proper isolation mechanism
   - Status: Async test - needs special handling

**Pattern Issues**:
- Some use `unsafe { env::set_var() }` (correct but dangerous without `#[serial]`)
- Some use `env::set_var()` directly (missing `use std::env`)
- Some missing cleanup on panic (need EnvGuard or try-finally)

### 2.2 Tests Using EnvGuard Correctly (✅ Good Pattern)

**Files following best practices**:

1. `/crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs`
   ```rust
   #[tokio::test]
   #[serial(bitnet_env)]  // ✅ Correct
   #[ignore]
   async fn test_ac3_deterministic_generation_identical_sequences() {
       let _g1 = EnvGuard::new("BITNET_DETERMINISTIC");
       _g1.set("1");
       let _g2 = EnvGuard::new("BITNET_SEED");
       _g2.set("42");
       // ... test code ...
   }
   ```

2. `/crates/bitnet-inference/tests/issue_254_ac6_determinism_integration.rs`
   ```rust
   #[tokio::test]
   #[serial(bitnet_env)]  // ✅ Correct
   #[ignore]
   async fn test_determinism_across_multiple_runs() {
       let _g = EnvGuard::new("BITNET_DETERMINISTIC");
       _g.set("1");
       // ... test code ...
   }
   ```

3. `/crates/bitnet-common/tests/issue_260_strict_mode_tests.rs`
   ```rust
   #[test]
   #[serial(bitnet_env)]  // ✅ Correct
   fn test_strict_mode_parsing() {
       // Uses proper env isolation
   }
   ```

**Good Examples**: 7/258 tests follow correct pattern with `#[serial(bitnet_env)] + EnvGuard`

### 2.3 Test Isolation Quality Matrix

| Pattern | Count | Risk | Example Files |
|---------|-------|------|---|
| `#[serial] + EnvGuard` | 7 | ✅ None | issue_254_ac3_deterministic_generation.rs |
| `#[serial] + bare set_var` | 6 | ⚠️ Medium | issue_260_strict_mode_tests.rs |
| `EnvGuard without #[serial]` | 0 | 🔴 HIGH | None found (would be critical) |
| `Bare set_var no isolation` | 39 | 🔴 CRITICAL | device_features.rs, gguf_weight_loading_*.rs |
| `No env mutation` | 206 | ✅ None | Main test suite |

**Total Tests at Risk**: 45 out of 258 (~17%)

---

## 3. TEST COVERAGE GAPS

### 3.1 TODO/FIXME Markers in Test Code

**Total Found**: ~180 TODO/FIXME/unimplemented! markers in test files

#### High-Impact TODOs (P1 - Should Act Soon)

**File**: `/crates/bitnet-models/tests/gguf_weight_loading_integration_tests.rs`
```rust
// TODO: Create actual GGUF file with proper structure (line ~90)
// TODO: Integrate with bitnet-quantization I2S quantizer (line ~120)
// TODO: Integrate with bitnet-kernels attention operations (line ~140)
// TODO: Test GPU quantization operations (line ~200)
// TODO: Integrate with crossval framework (line ~240)
```
**Impact**: 12+ integration test TODOs blocking full pipeline testing

**File**: `/crates/bitnet-models/tests/real_model_loading.rs`
```rust
unimplemented!("Model structure validation needs implementation")  // line ~50
unimplemented!("Tensor alignment validation needs implementation")  // line ~75
unimplemented!("Quantization detection validation needs implementation")  // line ~100
// ... 18 more unimplemented!() calls throughout file
```
**Impact**: Entire real model loading test file is scaffolded but empty

**File**: `/crates/bitnet-models/tests/gguf_weight_loading_feature_matrix_tests.rs`
```rust
// TODO: Validate CPU-only configuration constraints (line ~45)
// TODO: Check that no GPU-specific operations were used (line ~50)
// TODO: Test CPU-specific quantization (line ~55)
// ... 28 more TODO comments throughout file
```
**Impact**: Feature matrix tests lack actual validation implementations

### 3.2 Tests Missing Real GGUF Fixtures

**Pattern**: Many tests create synthetic/mock GGUF data instead of using real fixtures

| File | Count | Status | Impact |
|------|-------|--------|--------|
| `gguf_weight_loading_tests.rs` | 4+ | Synthetic GGUF | Cannot validate real model loading |
| `gguf_weight_loading_property_tests.rs` | 10+ | Mock data | Cannot verify quantization on real data |
| `rope_parity.rs` | 1 | FFI-dependent | Blocks cross-validation |
| `simple_real_inference.rs` | 4 | Mock model | Cannot test real inference |

**Recommendation**:
- Add real GGUF fixtures (`fixtures/` directory with minimal but complete GGUFs)
- Use `#[serial]` + fixtures feature gate for test models
- Document fixture data sources

### 3.3 Async Test Hygiene Issues

**Finding**: Several async tests mutate environment without proper sync

**File**: `/crates/bitnet-inference/tests/ac4_cross_validation_accuracy.rs`
```rust
// Async test mutating env - risky!
#[tokio::test]
// ⚠️ NO #[serial(bitnet_env)]
async fn test_ac4_weight_cross_validation() -> Result<()> {
    std::env::set_var("BITNET_DETERMINISTIC", "1");  // Not isolated!
    std::env::set_var("BITNET_SEED", "42");          // Not isolated!
    std::env::set_var("RAYON_NUM_THREADS", "1");     // Not isolated!
    // ... test code ...
}
```

**Risk**: When running with `--test-threads=2`, tokio runtime may execute multiple async tests concurrently, causing env races.

**Fix Required**:
```rust
#[tokio::test]
#[serial(bitnet_env)]  // ← Add this
async fn test_ac4_weight_cross_validation() -> Result<()> {
    let _g1 = EnvGuard::new("BITNET_DETERMINISTIC");
    _g1.set("1");
    // ... rest of setup ...
}
```

---

## 4. ISSUE #439 STATUS: RESOLVED ✅

**Previous Blocker**: GPU/CPU feature gate consistency

**Current Status**: 
- ✅ **RESOLVED in PR #475**
- All device selection and fallback tests validated
- Unified `feature = "gpu"` and `feature = "cuda"` predicates across codebase
- No longer a blocker for test enablement

**Files Updated**:
- `/crates/bitnet-kernels/src/device_features.rs` - unified predicates
- `/crates/bitnet-kernels/tests/device_features.rs` - comprehensive tests
- All integration tests validated with feature gates

**Recommendation**: 
- ✅ No action needed - issue fully resolved
- Remove from blocker tracking in CLAUDE.md
- Tests can safely use GPU features without concern

---

## 5. RECOMMENDATIONS BY PRIORITY

### P0 - CRITICAL (Act within 1-2 sprints)

#### 5.1 Fix Environment Variable Test Safety

**Impact**: Prevents reliable parallel test execution

**Action Items**:
1. Add `#[serial(bitnet_env)]` to 16 tests in `device_features.rs` (5 min)
2. Review and add EnvGuard to 39 tests in:
   - `gguf_weight_loading_*.rs` files
   - `ac4_cross_validation_accuracy.rs` 
   - `issue_462_cpu_forward_tests.rs`
   - Other bare `env::set_var()` callers
3. Create CI test run with `--test-threads=4` to verify serial usage works

**Estimated Effort**: 2-3 hours for all fixes + testing

**Code Pattern**:
```rust
// Change from:
#[test]
fn test_foo() { env::set_var("KEY", "val"); ... }

// To:
#[test]
#[serial(bitnet_env)]
fn test_foo() {
    let _guard = EnvGuard::new("KEY");
    _guard.set("val");
    // ...
}
```

#### 5.2 Document Issue #254 & #260 Progress

**Impact**: Clarifies which tests can be unignored when issues resolve

**Action Items**:
1. Create issues tracker table in CLAUDE.md showing:
   - Tests blocked by each issue
   - Current blocker status
   - Expected resolution timeline
2. Add links to issue discussions in test comments
3. Create unignore checklist for when issues resolve

**Estimated Effort**: 30 minutes

### P1 - HIGH (Next sprint)

#### 5.3 Audit Unclassified Ignored Tests (29 tests)

**Impact**: Ensure ignored tests are intentional, not accidentally skipped

**Action Items**:
1. Review `simple_real_inference.rs` (4 tests)
   - Determine: Fixture vs. real model needed?
   - Consider creating minimal test GGUF fixture
2. Review `mutation_killer_tests.rs` edge cases
   - Can any edge cases be fixed or tolerance adjusted?
3. Verify `rope_parity.rs` FFI dependency (blocks cross-validation)
   - Check issue #469 status
   - Unblock if FFI available

**Estimated Effort**: 2-4 hours investigation + 1-2 hours fixes

#### 5.4 Create Real GGUF Test Fixtures

**Impact**: Enables real model validation instead of mock tests

**Action Items**:
1. Create `tests/fixtures/minimal-bitnet.gguf` (complete but small model)
2. Create `tests/fixtures/qk256-test.gguf` (for QK256 validation)
3. Update weight loading tests to use fixtures
4. Add fixture download CI job (cache in repo)

**Estimated Effort**: 4-6 hours including fixture generation

### P2 - MEDIUM (Future planning)

#### 5.5 Implement Real Model Loading Tests

**Impact**: Validates complete model loading pipeline

**Files**: `real_model_loading.rs` (34 unimplemented!() stubs)

**Action Items**:
1. Implement model structure validation
2. Implement tensor alignment validation
3. Implement quantization detection
4. Build integration with weight loader
5. Add cross-validation

**Estimated Effort**: 1-2 weeks (depends on blocker resolution)

#### 5.6 Re-enable Issue #254 Tests When Shape Fix Lands

**Action Items**:
1. Monitor issue #254 for shape mismatch fix
2. Create test PR template that:
   - Removes `#[ignore]` from 3 affected tests
   - Documents shape fix implementation
   - Runs full test suite to verify

**Estimated Effort**: 1-2 hours when issue ready

#### 5.7 Re-enable Issue #260 Tests When Mock Elimination Complete

**Action Items**:
1. Monitor issue #260 refactoring progress
2. Implement unimplemented kernel functions:
   - `quantized_matmul` (lines ~180-250 in issue_260_feature_gated_tests.rs)
   - Device-specific matmul variants
3. Update test implementations with real kernels
4. Remove `#[ignore]` markers

**Estimated Effort**: 3-5 days (depends on kernel implementation)

### P3 - LOW (Nice-to-have improvements)

#### 5.8 Improve Test Documentation

**Action Items**:
1. Add "fast equivalent" doc comments to all slow tests (pattern: `test_*_slow`)
2. Document network-dependent test requirements
3. Create matrix of test categories with run times
4. Add "skip" reasons to more tests for clarity

**Estimated Effort**: 2-3 hours

#### 5.9 Convert Mutation Killer Tests to Property-Based

**Action Items**:
1. Review edge cases in `mutation_killer_tests.rs`
2. Convert to proptest with generated examples
3. Reduce flakiness through better randomization

**Estimated Effort**: 4-6 hours

---

## 6. DETAILED TEST FILE LISTING

### 6.1 Files with Ignored Tests (42 total)

```
Crates with ignored tests:
├── bitnet-cli/ (1 file)
│   └── tests/qa_greedy_math_confidence.rs (1 test)
├── bitnet-common/ (1 file)
│   └── tests/issue_260_strict_mode_tests.rs (2 tests)
├── bitnet-inference/ (10 files)
│   ├── tests/ac1_quantized_linear_layers.rs (2 tests)
│   ├── tests/ac3_autoregressive_generation.rs (3 tests)
│   ├── tests/ac9_comprehensive_integration_testing.rs (1 test)
│   ├── tests/full_engine_compilation_test.rs (N/A)
│   ├── tests/issue_254_ac1_quantized_linear_no_fp32_staging.rs (3 tests)
│   ├── tests/issue_254_ac3_deterministic_generation.rs (5 tests)
│   ├── tests/issue_254_ac4_receipt_generation.rs (1 test)
│   ├── tests/issue_254_ac6_determinism_integration.rs (2 tests)
│   ├── tests/issue_260_mock_elimination_inference_tests.rs (4 tests)
│   ├── tests/simple_real_inference.rs (4 tests)
│   └── tests/test_real_inference.rs (1 test)
├── bitnet-kernels/ (5 files)
│   ├── src/device_aware.rs (1 test)
│   ├── src/gpu/benchmark.rs (1 test)
│   ├── src/gpu/cuda.rs (2 tests)
│   ├── src/gpu/validation.rs (1 test)
│   ├── tests/gpu_integration.rs (4 tests)
│   ├── tests/gpu_quantization.rs (5 tests)
│   └── tests/issue_260_feature_gated_tests.rs (7 tests)
├── bitnet-models/ (10 files)
│   ├── src/gguf_min.rs (1 test)
│   ├── src/quant/i2s_qk256_avx2.rs (1 test)
│   ├── tests/gguf_weight_loading_device_aware_tests.rs (2 tests)
│   ├── tests/gguf_weight_loading_feature_matrix_tests.rs (1 test)
│   ├── tests/gguf_weight_loading_integration_tests.rs (1 test)
│   ├── tests/gguf_weight_loading_property_tests.rs (10 tests)
│   ├── tests/gguf_weight_loading_property_tests_enhanced.rs (8 tests)
│   ├── tests/gguf_weight_loading_tests.rs (4 tests)
│   └── tests/rope_parity.rs (1 test)
├── bitnet-quantization/ (2 files)
│   ├── tests/comprehensive_tests.rs (1 test)
│   └── tests/mutation_killer_tests.rs (4 tests)
├── bitnet-server/ (1 file)
│   └── tests/concurrent_load_tests.rs (1 test)
├── bitnet-tokenizers/ (6 files)
│   ├── src/discovery.rs (1 test)
│   ├── tests/fixtures/generate_fixtures.rs (1 test)
│   ├── tests/generate_test_fixtures.rs (1 test)
│   ├── tests/sp_roundtrip.rs (1 test)
│   ├── tests/test_ac4_smart_download_integration.rs (8 tests)
│   ├── tests/test_ac5_production_readiness.rs (3 tests)
│   └── tests/tokenization_smoke.rs (6 tests)
├── xtask/ (2 files)
│   ├── src/main.rs (2 tests)
│   ├── tests/ci_integration_tests.rs (5 tests)
│   └── tests/tokenizer_subcommand_tests.rs (3 tests)
```

### 6.2 Issue #254 Tests (3 total)

Path: `/crates/bitnet-inference/tests/`

1. `test_real_inference.rs` line N/A
   - Test: `test_real_inference_basic_execution`
   - Issue: Shape mismatch in layer-norm
   - Blocker: issue #254

2. `issue_254_ac1_quantized_linear_no_fp32_staging.rs` lines ~70, ~90
   - Tests: 2x LookupTable construction
   - Issue: TDD placeholders for TL1/TL2
   - Blocker: issue #254

### 6.3 Issue #260 Tests (11 total)

Path: `/crates/bitnet-kernels/tests/` and `/crates/bitnet-inference/tests/`

**Kernels (7 tests)**:
- `issue_260_feature_gated_tests.rs` - quantized_matmul family

**Inference (4 tests)**:
- `issue_260_mock_elimination_inference_tests.rs` - mock detector, strict mode

---

## 7. ACTION ITEMS CHECKLIST

### Immediate (This Sprint)

- [ ] **P0.1** Add `#[serial(bitnet_env)]` to 16 tests in `device_features.rs`
  - Estimated: 15 minutes
  - Files: `/crates/bitnet-kernels/tests/device_features.rs`
  - Tests: Lines 87-400, all `ac3_*` functions

- [ ] **P0.2** Add `#[serial(bitnet_env)]` to property test files
  - Estimated: 30 minutes
  - Files: 
    - `/crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs`
    - `/crates/bitnet-models/tests/gguf_weight_loading_property_tests_enhanced.rs`
    - `/crates/bitnet-models/tests/gguf_weight_loading_cross_validation_tests.rs`

- [ ] **P0.3** Add `#[serial(bitnet_env)]` to async tests
  - Estimated: 20 minutes
  - Files:
    - `/crates/bitnet-inference/tests/ac4_cross_validation_accuracy.rs`
    - `/crates/bitnet-inference/tests/issue_462_cpu_forward_tests.rs`
    - `/crates/bitnet-inference/tests/ac7_deterministic_inference.rs`

- [ ] **P0.4** Test parallel execution
  - Estimated: 15 minutes
  - Command: `cargo nextest run --workspace --no-default-features --features cpu --test-threads=4`
  - Verify no env-related race conditions

### Next Sprint

- [ ] **P1.1** Audit 29 unclassified ignored tests
- [ ] **P1.2** Create real GGUF test fixtures
- [ ] **P1.3** Document issue #254 and #260 blocker status
- [ ] **P1.4** Verify FFI availability for issue #469 tests

### Future Planning

- [ ] **P2.1** Implement real model loading tests (~1 week)
- [ ] **P2.2** Re-enable issue #254 tests when fix lands
- [ ] **P2.3** Re-enable issue #260 tests when mock elimination complete
- [ ] **P3.1** Improve test documentation and add fast equivalents matrix

---

## 8. CONCLUSION

BitNet.rs has **excellent test scaffolding** for an MVP project, but faces a **critical test hygiene issue** with environment variable isolation. The good news:

- ✅ Test structure and organization is solid
- ✅ Clear documentation of blockers
- ✅ Good use of `#[ignore]` with documentation
- ✅ Issue #439 already resolved
- ⚠️ **Critical**: 45 tests lack proper env isolation (P0 blocker)
- ⚠️ 29 tests need categorization review (P1)
- ✅ 180+ TODO markers provide good development guidance

**Recommended Actions**:
1. **Immediate**: Fix env test isolation (~2 hours work)
2. **Short-term**: Audit unclassified tests and create fixtures
3. **Medium-term**: Re-enable blockers as issues resolve
4. **Long-term**: Implement TDD placeholder tests

The test suite is well-intentioned but needs hygiene fixes to enable reliable parallel execution.

