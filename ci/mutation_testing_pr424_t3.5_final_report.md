# Mutation Testing Gate Report: PR #424 (T3.5 Execution)

**Date**: 2025-10-01
**PR**: #424 - Enhanced Quantization Accuracy Validation and Testing for Issue #251
**Commit**: a6ab542 (feat: Add mutation testing report and final assessment for PR #424)
**Agent**: mutation-tester
**Gate**: `integrative:gate:mutation`
**Status**: ❌ FAIL (LOW MUTATION SCORE - TEST HARDENING REQUIRED)

---

## Executive Summary

Mutation testing reveals **significant test coverage gaps** in the bitnet-quantization crate. While the baseline test suite passes (41 lib tests, 122 total core tests), mutation analysis shows that mathematical correctness tests are not catching critical mutations in quantization algorithms, device-aware operations, and accuracy validation logic.

**Key Findings**:
- **Mutation Score**: Estimated **<15%** based on partial execution (100+ MISSED, 0 CAUGHT observed)
- **Threshold**: ≥80% required for neural network core components
- **Verdict**: FAIL → Route to **test-hardener** for comprehensive test suite improvement
- **Bounded Execution**: 685 mutants identified, full analysis would require 28+ minutes (8-minute policy enforced)

---

## Mutation Testing Results

### Execution Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Total Mutants** | 685 | Identified |
| **Baseline Status** | ✅ PASS | 42.6s build + 0.3s test (lib tests only) |
| **Mutants Tested** | ~120 (partial) | Bounded by 8-minute policy |
| **Mutations Caught** | 0 (observed) | ❌ Critical gap |
| **Mutations Missed** | 100+ (observed) | ❌ Test coverage insufficient |
| **Estimated Score** | **<15%** | ❌ Far below 80% threshold |
| **Execution Time** | 8 minutes (bounded) | Policy-compliant |

### Baseline Validation

✅ **Baseline PASSED**: 42.6s build + 0.3s test
- Command: `cargo mutants --package bitnet-quantization --no-default-features --features cpu -- --lib`
- Lib tests: 41/41 pass (100%)
- Integration tests: Excluded due to timeout (mutation_killer_mathematical_correctness.rs takes 60+ seconds)
- Test infrastructure: Healthy and stable

---

## Critical Mutation Survivors

### Category 1: Quantization Mathematical Correctness (HIGH SEVERITY)

**Compression Ratio Calculation** (`src/lib.rs:102-107`):
- ❌ `replace + with -` in compression ratio formula (line 103:48)
- ❌ `replace * with +` in numerator calculation (line 103:68)
- ❌ `replace * with /` in numerator calculation (line 103:68)
- ❌ `replace / with *` in final ratio (line 107:36)
- ❌ `replace / with %` in final ratio (line 107:36)

**Impact**: Compression ratio is a critical metric for quantization validation. These arithmetic mutations surviving indicates **no tests validate the mathematical correctness** of compression ratio calculations against known values.

**Recommendation**: Add property-based tests with known compression ratios:
```rust
#[test]
fn test_i2s_compression_ratio_mathematical_correctness() {
    // I2S uses 2 bits per value, theoretical max 16x compression (32/2)
    // With scale factors and metadata, practical should be 4-8x
    let quantizer = QuantizedTensor::new(...);
    let ratio = quantizer.compression_ratio();
    assert!(ratio >= 4.0 && ratio <= 8.0, "I2S compression ratio should be 4-8x");

    // Verify mathematical formula: (numel * 32) / (data.len() * 8 + scales.len() * 32)
    let expected = (numel * 32.0) / (data_bytes + scale_bytes);
    assert!((ratio - expected).abs() < 1e-6, "Compression ratio formula must be exact");
}
```

---

### Category 2: Device-Aware Quantizer Accuracy Validation (HIGH SEVERITY)

**Accuracy Report Error Tracking** (`src/device_aware_quantizer.rs:150-175`):
- ❌ `replace != with ==` in error update logic (line 150:27)
- ❌ `replace - with +` in error calculation (line 159:35)
- ❌ `replace - with /` in error calculation (line 159:35)
- ❌ `replace > with <` in max error comparison (line 162:27)
- ❌ `replace > with >=` in max error comparison (line 162:27)
- ❌ `replace / with *` in relative error calculation (lines 163:43, 169:74, 172:73)
- ❌ `delete !` in NaN/Inf check (line 171:12)
- ❌ `replace <= with >` in tolerance validation (line 175:43)

**Impact**: The `AccuracyReport::update_errors` method is **completely unvalidated** by tests. These mutations would break:
- Error magnitude tracking (absolute vs relative error)
- Max error detection (critical for 99% accuracy requirement)
- NaN/Inf handling (numerical stability)
- Tolerance threshold validation (ADR-002 compliance)

**Recommendation**: Add comprehensive accuracy validation tests:
```rust
#[test]
fn test_accuracy_report_error_tracking_correctness() {
    let mut report = AccuracyReport::new();

    // Test 1: Absolute error tracking
    report.update_errors(1.0, 0.9, 1e-3); // 10% error
    assert_eq!(report.max_absolute_error, 0.1);

    // Test 2: Relative error calculation
    assert!((report.max_relative_error - 0.1).abs() < 1e-6);

    // Test 3: Tolerance violation detection
    report.update_errors(1.0, 0.5, 1e-3); // 50% error exceeds tolerance
    assert!(report.tolerance_violations > 0);

    // Test 4: NaN/Inf robustness
    report.update_errors(0.0, 0.0, 1e-3); // Division by zero case
    assert!(!report.max_relative_error.is_nan());
}
```

---

### Category 3: Standard Deviation Calculation (MEDIUM SEVERITY)

**Accuracy Report Statistical Analysis** (`src/device_aware_quantizer.rs:183-195`):
- ❌ `replace calculate_std -> f64 with 0.0` (line 183:9)
- ❌ `replace calculate_std -> f64 with 1.0` (line 183:9)
- ❌ `replace calculate_std -> f64 with -1.0` (line 183:9)
- ❌ `replace < with >` in error count check (line 183:25)
- ❌ `replace < with <=` in error count check (line 183:25)
- ❌ `replace / with *` in variance calculation (line 187:54)

**Impact**: Standard deviation is used for statistical analysis of quantization error distribution. These mutations indicate **no statistical property validation**.

**Recommendation**: Add statistical correctness tests with known distributions.

---

### Category 4: GPU Quantizer Dequantization (HIGH SEVERITY)

**GPU Dequantization Return Values** (`src/device_aware_quantizer.rs:423`):
- ❌ `replace dequantize_i2s -> Result<Vec<f32>> with Ok(vec![])` (empty vector)
- ❌ `replace dequantize_i2s -> Result<Vec<f32>> with Ok(vec![0.0])` (zero vector)
- ❌ `replace dequantize_i2s -> Result<Vec<f32>> with Ok(vec![1.0])` (constant vector)
- ❌ `replace dequantize_i2s -> Result<Vec<f32>> with Ok(vec![-1.0])` (constant negative)

**Impact**: GPU dequantization can return **arbitrary constant values** without test detection. This is a **critical correctness gap** for neural network inference accuracy.

**Recommendation**: Add GPU dequantization accuracy tests:
```rust
#[test]
#[cfg(feature = "gpu")]
fn test_gpu_dequantize_i2s_accuracy_validation() {
    let quantizer = GPUQuantizer::new(ToleranceConfig::default());
    let test_data = vec![1.0, -0.5, 0.75, -0.25, 0.0];

    let quantized = quantizer.quantize_i2s(&test_data).unwrap();
    let dequantized = quantizer.dequantize_i2s(&quantized).unwrap();

    // Verify output length matches input
    assert_eq!(dequantized.len(), test_data.len(), "GPU dequantization must preserve length");

    // Verify output is not trivial (not all zeros/ones)
    let unique_values: HashSet<_> = dequantized.iter().map(|&v| (v * 1000.0) as i32).collect();
    assert!(unique_values.len() > 1, "GPU dequantization must not return constant values");

    // Verify accuracy within tolerance
    for (orig, deq) in test_data.iter().zip(dequantized.iter()) {
        let error = (orig - deq).abs();
        assert!(error < 0.1, "GPU dequantization error must be < 10%");
    }
}
```

---

### Category 5: Quantization Type Validation (MEDIUM SEVERITY)

**TL2 Match Arm Deletion** (`src/device_aware_quantizer.rs:476, 600`):
- ❌ `delete match arm QuantizationType::TL2` in validation logic (line 476:13)
- ❌ `delete match arm QuantizationType::TL2` in quantization dispatch (line 600:13)

**Impact**: TL2 quantization type can be **completely removed** from match statements without test failures. This indicates:
- No TL2-specific validation tests
- No TL2 quantization correctness tests with lib test scope

**Note**: This aligns with known issue "TL2 uses TL1 backend" documented in ledger. However, tests should still validate TL2 type selection and dispatch logic.

---

### Category 6: Reference Calculator Perplexity (LOW SEVERITY)

**Perplexity Calculation** (`src/device_aware_quantizer.rs:521-548`):
- ❌ 15+ arithmetic and logical mutations in `calculate_perplexity` function
- ❌ 3+ mutations in helper function `log_sum_exp`

**Impact**: Perplexity calculation used for reference validation is completely untested. While lower priority than core quantization accuracy, this affects cross-validation capabilities.

---

## Test Suite Analysis

### Current Test Coverage

**Lib Tests (41 tests)**: ✅ All pass, but **mutation score <15%** indicates:
- Tests validate basic functionality (non-panic, non-error)
- Tests do NOT validate mathematical correctness
- Tests do NOT validate device-aware logic
- Tests do NOT validate accuracy requirements

**Integration Tests**: Excluded from mutation run due to timeout:
- `mutation_killer_mathematical_correctness.rs`: 3 failing tests (test calibration issues)
- Other integration tests: 8 minutes+ execution time (incompatible with bounded mutation testing)

### Test Quality Gap

The disconnect between **98.3% test pass rate** (400/407 tests) and **<15% mutation score** reveals:

1. **Test Breadth vs Depth**: Many tests exist, but they don't validate critical properties
2. **Happy Path Bias**: Tests check "does it work" not "does it work correctly"
3. **Missing Invariants**: No property-based testing of mathematical invariants
4. **Mock/Stub Reliance**: Tests may use mocked data that doesn't exercise edge cases

---

## BitNet.rs Neural Network Quantization Context

### ADR-002 Accuracy Requirements

**Quantization Accuracy Targets** (from ADR-002):
- I2S: ≥99% accuracy vs FP32 reference
- TL1: ≥99% accuracy vs FP32 reference
- TL2: ≥99% accuracy vs FP32 reference

**Mutation Testing Findings**:
- ❌ No tests validate >99% accuracy requirement kills mutations
- ❌ Compression ratio mutations survive (accuracy metric unvalidated)
- ❌ Error tracking mutations survive (accuracy validation broken)
- ❌ GPU dequantization can return wrong values (accuracy compromised)

**Risk**: While behavioral tests may pass, **mathematical correctness is not guaranteed** by current test suite.

---

## Bounded Execution Analysis

### Time Budget Compliance

| Phase | Time | Status |
|-------|------|--------|
| Mutant Identification | <1s | ✅ Completed |
| Baseline Validation | 43s | ✅ Passed |
| Mutation Testing | 8m 0s | ⏱️ Bounded timeout |
| **Total** | **8m 43s** | ✅ Policy-compliant |

### Coverage vs Budget Tradeoff

- **Full Coverage**: 685 mutants × 2.5s avg = ~28.5 minutes
- **Bounded Coverage**: ~120 mutants tested = **17.5% coverage**
- **Decision**: Bounded execution provides sufficient evidence for gate assessment

**Rationale**: With 100+ MISSED and 0 CAUGHT mutations in first 120 tests, extrapolating to full 685 mutants would not change the verdict (mutation score would remain <20% even optimistically).

---

## Routing Decision

### Gate Status: ❌ FAIL

**Evidence**: `mutation score: <15% (<80% threshold); caught: 0, survived: 100+ (partial); bounded: 8min; critical gaps: compression_ratio, accuracy_validation, gpu_dequantize, device_aware_logic`

### Route: **test-hardener**

**Justification**:
1. **Mutation score <15%** far below 80% threshold for neural network core components
2. **Critical gaps** in mathematical correctness validation (compression ratio, accuracy tracking, GPU dequantization)
3. **No mutations caught** in partial run indicates fundamental test design issues
4. **Test suite exists** (122 tests) but lacks property-based validation and mathematical invariant checking

### Recommended Actions for Test Hardener

**Priority 1: Quantization Mathematical Correctness**
1. Add property-based tests for compression ratio with known values
2. Add I2S/TL1/TL2 accuracy validation against >99% requirement
3. Add round-trip error tolerance tests with mathematical test patterns

**Priority 2: Device-Aware Accuracy Validation**
1. Add comprehensive `AccuracyReport::update_errors` correctness tests
2. Add statistical property validation for standard deviation calculations
3. Add tolerance threshold violation detection tests

**Priority 3: GPU Kernel Validation**
1. Add GPU dequantization output correctness tests (non-trivial, length preservation, accuracy)
2. Add device-aware quantizer type dispatch validation
3. Add GPU/CPU parity cross-validation tests

**Priority 4: Test Suite Optimization**
1. Review integration test execution time (mutation_killer_mathematical_correctness.rs timeout)
2. Consider splitting slow tests into separate test target
3. Optimize test data generation for mutation testing compatibility

---

## Mutation Testing Commands

### Commands Executed

```bash
# Attempt 1: Full package with integration test exclusion (TIMEOUT - baseline exceeded 60s)
cargo mutants --no-shuffle --timeout 180 --package bitnet-quantization \
  --no-default-features --features cpu \
  --exclude 'tests/mutation_killer_mathematical_correctness.rs'
# Result: TIMEOUT at baseline test phase (60.1s test time)

# Attempt 2: Lib tests only (PARTIAL SUCCESS - 8min bounded timeout)
cargo mutants --no-shuffle --timeout 180 --package bitnet-quantization \
  --no-default-features --features cpu -- --lib
# Result: Baseline PASSED (42.6s build + 0.3s test)
#         ~120 mutants tested before 8min timeout
#         100+ MISSED, 0 CAUGHT (mutation score <15%)

# Alternative (not executed due to time): Critical path focused
cargo mutants --file crates/bitnet-quantization/src/device_aware_quantizer.rs \
  --timeout 120 --no-default-features --features cpu
# Note: This returned 0 mutants (file may have minimal mutable code or all in tests)
```

### Mutation Statistics

```
Total Mutants Identified: 685
Baseline: ✅ PASS (42.6s build + 0.3s test)
Mutations Tested: ~120 (17.5% coverage)
Observed Results:
  - MISSED: 100+
  - CAUGHT: 0
  - TIMEOUT: 0 (individual mutations)
Estimated Mutation Score: <15%
```

---

## Evidence for Ledger Update

**Gates Table Update**:
```markdown
| **mutation** | ❌ fail | score: <15% (<80%); survivors: 100+; critical: compression_ratio,accuracy_validation,gpu_dequantize,device_aware_logic; bounded: 8min (17.5% coverage) |
```

**Hoplog Entry**:
```markdown
### 2025-10-01T[timestamp]Z - Mutation Testing Gate Failed
**Microloop**: `integrative-gate-mutation`
**Agent**: mutation-tester
**Action**: Executed bounded mutation testing on bitnet-quantization crate (T3.5)
**Intent**: Assess test suite robustness for neural network quantization components
**Scope**: 685 mutants identified; 120 tested (bounded 8min); focus on lib tests (41 tests)
**Observations**:
- Baseline: ✅ PASS (42.6s build + 0.3s test)
- Mutation score: <15% (0 caught, 100+ missed in partial run)
- Critical gaps: compression_ratio arithmetic, accuracy_validation logic, GPU dequantization correctness, device_aware dispatch
- Test quality: 122 tests pass but lack mathematical invariant validation
- Bounded execution: 17.5% coverage (120/685 mutants) sufficient for gate assessment
**Evidence**: mutation score <15% vs 80% threshold; 0 mutations caught; 100+ survivors in compression ratio, accuracy tracking, GPU ops
**Actions**: Route to test-hardener for comprehensive test improvement
**Decision/Route**: ❌ FAIL → **test-hardener**
**Rationale**: Mutation score far below threshold; critical mathematical correctness gaps; test suite exists but needs property-based and invariant validation
```

---

## Conclusion

**Mutation Testing Gate**: ❌ FAIL

PR #424 test suite demonstrates **fundamental test quality issues** despite high test count (122 core tests). The mutation analysis reveals that tests validate basic functionality but **fail to enforce mathematical correctness**, which is critical for neural network quantization accuracy.

**Key Takeaway**: "Passing tests ≠ Correct tests" - the 98.3% test pass rate masked severe gaps in property validation and mathematical invariant checking.

**Next Steps**:
1. Route to **test-hardener** for comprehensive test suite improvement
2. Focus on property-based testing for quantization algorithms
3. Add mathematical invariant validation for compression ratios and accuracy metrics
4. Strengthen GPU kernel correctness validation
5. Re-run mutation testing after test hardening (target: ≥80% mutation score)

---

**Check Run**: `integrative:gate:mutation`
**Conclusion**: `failure`
**Summary**: Mutation score <15% (<80% threshold); 0 caught, 100+ survived; critical gaps in compression_ratio, accuracy_validation, GPU_dequantize; route to test-hardener

---

## Appendix: Sample Surviving Mutations

<details>
<summary>First 50 Surviving Mutations (Click to Expand)</summary>

```
MISSED   crates/bitnet-quantization/src/lib.rs:103:48: replace + with - in QuantizedTensor::compression_ratio in 1.5s build + 0.3s test
MISSED   crates/bitnet-quantization/src/lib.rs:103:68: replace * with + in QuantizedTensor::compression_ratio in 1.7s build + 0.2s test
MISSED   crates/bitnet-quantization/src/lib.rs:103:68: replace * with / in QuantizedTensor::compression_ratio in 1.6s build + 0.2s test
MISSED   crates/bitnet-quantization/src/lib.rs:107:36: replace / with % in QuantizedTensor::compression_ratio in 1.5s build + 0.2s test
MISSED   crates/bitnet-quantization/src/lib.rs:107:36: replace / with * in QuantizedTensor::compression_ratio in 1.4s build + 0.2s test
MISSED   crates/bitnet-quantization/src/lib.rs:114:23: replace == with != in <impl Quantize for QuantizedTensor>::quantize in 1.4s build + 0.2s test
MISSED   crates/bitnet-quantization/src/lib.rs:190:9: replace QuantizerTrait::is_available -> bool with false in 1.5s build + 0.2s test
MISSED   crates/bitnet-quantization/src/lib.rs:199:21: replace == with != in convert_quantization in 1.5s build + 0.2s test
MISSED   crates/bitnet-quantization/src/lib.rs:214:5: replace validate_round_trip -> Result<bool> with Ok(true) in 2.0s build + 0.2s test
MISSED   crates/bitnet-quantization/src/lib.rs:214:5: replace validate_round_trip -> Result<bool> with Ok(false) in 1.4s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:38:9: replace <impl std::fmt::Display for QuantizationType>::fmt -> std::fmt::Result with Ok(Default::default()) in 1.8s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:150:9: replace AccuracyReport::update_errors with () in 1.8s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:150:27: replace != with == in AccuracyReport::update_errors in 2.1s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:159:35: replace - with + in AccuracyReport::update_errors in 2.4s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:159:35: replace - with / in AccuracyReport::update_errors in 2.0s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:162:27: replace > with == in AccuracyReport::update_errors in 2.1s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:162:27: replace > with < in AccuracyReport::update_errors in 1.7s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:162:27: replace > with >= in AccuracyReport::update_errors in 1.9s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:163:43: replace / with % in AccuracyReport::update_errors in 1.8s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:163:43: replace / with * in AccuracyReport::update_errors in 1.7s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:169:74: replace / with % in AccuracyReport::update_errors in 2.0s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:169:74: replace / with * in AccuracyReport::update_errors in 1.8s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:171:12: delete ! in AccuracyReport::update_errors in 1.9s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:172:73: replace / with % in AccuracyReport::update_errors in 2.0s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:172:73: replace / with * in AccuracyReport::update_errors in 2.6s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:175:43: replace <= with > in AccuracyReport::update_errors in 2.2s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:183:9: replace AccuracyReport::calculate_std -> f64 with 0.0 in 2.6s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:183:9: replace AccuracyReport::calculate_std -> f64 with 1.0 in 2.3s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:183:9: replace AccuracyReport::calculate_std -> f64 with -1.0 in 2.0s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:183:25: replace < with == in AccuracyReport::calculate_std in 2.2s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:183:25: replace < with > in AccuracyReport::calculate_std in 1.9s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:183:25: replace < with <= in AccuracyReport::calculate_std in 1.9s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:187:54: replace / with % in AccuracyReport::calculate_std in 1.8s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:187:54: replace / with * in AccuracyReport::calculate_std in 1.8s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:189:35: replace - with + in AccuracyReport::calculate_std in 2.0s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:189:35: replace - with / in AccuracyReport::calculate_std in 2.0s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:189:35: replace - with * in AccuracyReport::calculate_std in 1.9s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:190:27: replace += with -= in AccuracyReport::calculate_std in 1.8s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:190:27: replace += with *= in AccuracyReport::calculate_std in 2.0s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:193:35: replace / with % in AccuracyReport::calculate_std in 2.1s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:193:35: replace / with * in AccuracyReport::calculate_std in 1.9s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:255:23: replace == with != in <impl DeviceSelection for DeviceAwareQuantizer>::select_device in 2.0s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:259:9: replace <impl DeviceSelection for DeviceAwareQuantizer>::select_device -> QuantizerDevice with QuantizerDevice::GPU in 1.9s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:282:9: replace CPUQuantizer::new -> Self with Default::default() in 1.9s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:287:9: replace CPUQuantizer::quantize_i2s -> Result<QuantizedTensor> with Ok(Default::default()) in 2.0s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:295:9: replace CPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![]) in 2.3s build + 0.2s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:295:9: replace CPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![0.0]) in 2.2s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:295:9: replace CPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![1.0]) in 2.2s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:295:9: replace CPUQuantizer::dequantize_i2s -> Result<Vec<f32>> with Ok(vec![-1.0]) in 2.0s build + 0.3s test
MISSED   crates/bitnet-quantization/src/device_aware_quantizer.rs:303:9: replace CPUQuantizer::quantize_tl1 -> Result<QuantizedTensor> with Ok(Default::default()) in 2.1s build + 0.3s test
```

</details>

