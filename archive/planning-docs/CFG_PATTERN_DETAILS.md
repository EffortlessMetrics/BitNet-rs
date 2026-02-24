# Conditional Compilation Pattern Details

## Overview

This document provides detailed information about all `#[cfg(...)]` patterns found in test files across the BitNet.rs workspace.

## Summary Statistics

- **Total test files**: 459
- **Files with cfg gates**: ~45
- **Test functions with cfg gates**: ~51 (across 18 files with ALL tests gated)
- **Module-level cfg gates**: 1 (cfg(false))

## Files with ALL Tests Behind a Single Cfg Gate

### 1. `crates/bitnet-kernels/tests/issue_462_tl_lut_tests.rs`
**Gate**: Complex target_arch combinations
**Test count**: 13 tests
**Pattern**: 
```rust
#[cfg(target_arch = "x86_64")]
#[test]
fn test_name() { ... }
```

### 2. `crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs`
**Gate**: `feature = "cpu"`
**Test count**: 17 tests
**Pattern**:
```rust
#[cfg(feature = "cpu")]
#[test]
fn test_name() { ... }
```

### 3. `crates/bitnet-models/tests/gguf_weight_loading_property_tests_enhanced.rs`
**Gate**: `feature = "cpu"`
**Test count**: 7 tests

### 4. `crates/bitnet-models/tests/embedding_transpose_normalization.rs`
**Gate**: `feature = "cpu"`
**Test count**: 9 tests

### 5. `crates/bitnet-inference/tests/full_engine_compilation_test.rs`
**Gate**: `feature = "full-engine"`
**Test count**: 8 tests
**Note**: Tests compile-time features of full engine

### 6. `crates/bitnet-inference/tests/quantization_accuracy_strict_test.rs`
**Gate**: `feature = "inference"`
**Test count**: 9 tests

### 7. `crates/bitnet-tokenizers/tests/test_ac1_embedded_tokenizer_support.rs`
**Gate**: `feature = "cpu"`
**Test count**: 10 tests
**Issue**: Should run with `--features cpu` but shows 0 tests

### 8. `crates/bitnet-tokenizers/tests/test_ac2_model_architecture_detection.rs`
**Gate**: `feature = "cpu"`
**Test count**: 16 tests
**Issue**: Should run with `--features cpu` but shows 0 tests

### 9. `crates/bitnet-tokenizers/tests/test_ac3_vocabulary_size_resolution.rs`
**Gate**: `feature = "cpu"`
**Test count**: 16 tests
**Issue**: Should run with `--features cpu` but shows 0 tests

### 10. `crates/bitnet-tokenizers/tests/test_ac5_production_readiness.rs`
**Gate**: `feature = "cpu"`
**Test count**: 15 tests
**Issue**: Should run with `--features cpu` but shows 0 tests

### 11. `crates/bitnet-cli/tests/issue_462_cli_inference_tests.rs`
**Gate**: `feature = "cpu"`
**Test count**: 4 tests

### 12. `crates/bitnet-quantization/tests/tl_packing_correctness.rs`
**Gate**: `feature = "inference"`
**Test count**: 21 tests

### 13. `crates/bitnet-quantization/tests/feature_gate_consistency.rs`
**Gate**: `feature = "inference"`
**Test count**: 4 tests

### 14. `crates/bitnet-quantization/tests/device_tests.rs`
**Gate**: `feature = "gpu"`
**Test count**: 1 test

### 15. `crates/bitnet-ggml-ffi/tests/iq2s_link.rs`
**Gate**: `feature = "iq2s-ffi"`
**Test count**: 2 tests

### 16. `crates/bitnet-server/tests/ac02_concurrent_requests.rs`
**Gate**: `feature = "opentelemetry"`
**Test count**: 6 tests

### 17. `crates/bitnet-server/tests/dependencies_test.rs`
**Gate**: `feature = "opentelemetry"`
**Test count**: 4 tests

### 18. `crates/bitnet-tokenizers/tests/universal_tokenizer_integration.rs` (Module-level)
**Gate**: `#![cfg(false)]` at module level
**Test count**: 7 tests
**Status**: Entire module disabled - TDD scaffold

## Feature Gate Frequency Analysis

### By Feature
```
feature = "cpu"            →  10 tests  (should already run!)
feature = "gpu"            →  12 tests  (CUDA required)
feature = "inference"      →  20 tests  (optional feature)
feature = "opentelemetry"  →  12 tests  (optional metrics)
feature = "full-engine"    →   8 tests  (optional feature)
feature = "ffi"            →   2 tests  (FFI optional)
feature = "simd"           →   2 tests  (SIMD optional)
feature = "iq2s-ffi"       →   1 test   (FFI optional)
target_arch = "x86_64"     →   1 test   (arch-specific)
target_arch = "aarch64"    →   1 test   (arch-specific)
target_arch = "wasm32"     →   2 tests  (arch-specific)
```

### By Crate
```
bitnet-kernels/tests      →  13 tests (mostly arch-gated)
bitnet-models/tests       →  33 tests (mostly CPU-gated)
bitnet-inference/tests    →  17 tests (feature-gated)
bitnet-tokenizers/tests   →  57 tests (mostly CPU-gated + 7 module-disabled)
bitnet-quantization/tests →  26 tests (feature-gated)
bitnet-cli/tests          →   4 tests (CPU-gated)
bitnet-server/tests       →  10 tests (OT-gated)
bitnet-ggml-ffi/tests     →   2 tests (FFI-gated)
```

## Complex Cfg Patterns Found

### Pattern 1: Feature Combinations
```rust
#[cfg(all(feature = "inference", feature = "crossval"))]
fn test_name() { ... }
```
**Crates using**: bitnet-inference, bitnet-tokenizers
**Impact**: Tests require multiple features enabled

### Pattern 2: Any Alternative
```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn test_name() { ... }
```
**Crates using**: bitnet-inference, bitnet-kernels
**Impact**: Tests run if EITHER feature is enabled

### Pattern 3: Negation
```rust
#[cfg(not(feature = "gpu"))]
fn test_name() { ... }
```
**Crates using**: bitnet-common, bitnet-inference
**Impact**: Tests run when GPU feature is NOT enabled

### Pattern 4: Complex Combinations
```rust
#[cfg(all(feature = "gpu", not(feature = "strict")))]
fn test_name() { ... }
```
**Crates using**: bitnet-kernels
**Impact**: Tests run with GPU enabled AND strict mode disabled

## Module-Level Cfg Patterns

### Complete Module Disablement
**File**: `crates/bitnet-tokenizers/tests/universal_tokenizer_integration.rs`
```rust
#![cfg(false)]
#![allow(dead_code, unused_variables, unused_imports)]

#[test]
fn test_1() { ... }  // Never compiles

#[test]
fn test_2() { ... }  // Never compiles
```
**Reason**: TDD scaffold - module intentionally disabled
**Impact**: 7 tests completely unreachable
**To enable**: Change `#![cfg(false)]` to `#![cfg(true)]` or remove

## Architecture-Specific Patterns

### x86_64 Only
```rust
#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_simd() { ... }
```
**Crates**: bitnet-kernels, bitnet-quantization
**Count**: 1-9 tests depending on feature combination

### ARM64 Only
```rust
#[cfg(target_arch = "aarch64")]
#[test]
fn test_neon_simd() { ... }
```
**Crates**: bitnet-kernels, bitnet-quantization
**Count**: 1-9 tests depending on feature combination

### WebAssembly Only
```rust
#[cfg(target_arch = "wasm32")]
#[test]
fn test_wasm_compat() { ... }
```
**Crates**: bitnet-wasm
**Count**: 2 tests

## Miscellaneous Patterns

### Debug Assertions Only
```rust
#[cfg(debug_assertions)]
#[test]
fn test_slow_verification() { ... }
```
**Purpose**: Extra validation in debug builds only
**Count**: 1 test

### Unix Only
```rust
#[cfg(unix)]
#[test]
fn test_unix_specific() { ... }
```
**Purpose**: Tests requiring Unix-like systems
**Count**: 1 test

## Impact Assessment by Category

| Category | Tests | Status | Priority |
|---|---|---|---|
| Module disabled (cfg(false)) | 7 | TDD scaffold | Convert to #[ignore] |
| CPU-feature gated (should run!) | 51 | Investigation needed | High |
| GPU-feature gated | 12 | By design (OK) | None |
| Architecture-gated | ~20 | By design (OK) | None |
| Other optional features | ~30 | By design (OK) | None |
| Tests in tests/ crate | ~1,750 | Not auto-discovered | Remove autotests = false |

## Recommendations for Pattern Improvements

### 1. Replace `#![cfg(false)]` with `#[ignore]`
**Current**:
```rust
#![cfg(false)]
#[test]
fn test_foo() { unimplemented!() }
```

**Better**:
```rust
#[test]
#[ignore = "TDD scaffold for universal tokenizer - Issue #XXX"]
fn test_foo() { unimplemented!() }
```

**Benefit**: Tests remain visible and can be selectively run with `--include-ignored`

### 2. Group Feature-Gated Tests
**Current**: Each test individually gated
**Better**: Gate entire module with #![cfg(...)] if all tests share the same gate

```rust
#![cfg(feature = "inference")]

#[test]
fn test_1() { ... }

#[test]
fn test_2() { ... }
```

### 3. Document Cfg Requirements
**Add**: Comments explaining why cfg gate is needed

```rust
/// Test quantization accuracy with real inference pipeline
/// Requires: feature = "inference" (includes full model loading)
#[cfg(feature = "inference")]
#[test]
fn test_quantization_accuracy_with_inference() {
    // ...
}
```

### 4. Use Consistent Architecture Patterns
**Current**: Various patterns (x86_64, target_arch = "x86_64", etc.)
**Better**: Standardize to `#[cfg(target_arch = "x86_64")]`

## Files Recommended for Audit

1. `crates/bitnet-tokenizers/tests/test_ac*.rs` - Why don't CPU tests run?
2. `crates/bitnet-tokenizers/tests/universal_tokenizer_integration.rs` - Convert cfg(false)
3. `tests/Cargo.toml` - Remove autotests = false

## Related Issues

- #254: Shape mismatch in layer-norm (affects inference tests)
- #260: Mock elimination not complete
- #439: Feature gate consistency
- #469: Tokenizer parity and FFI build hygiene

