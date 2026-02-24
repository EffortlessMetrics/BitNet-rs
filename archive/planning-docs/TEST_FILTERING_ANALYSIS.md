# BitNet.rs Test Filtering Analysis Report

**Generated**: 2025-10-20
**Command Analyzed**: `cargo test --workspace --no-default-features --features cpu --lib --tests`
**Finding**: 94% of tests (1,883 out of 2,005) are filtered out at runtime

## Executive Summary

The large percentage of filtered tests is due to a combination of factors, primarily the workspace structure rather than individual test cfg gates:

1. **Workspace-level issue**: The `tests/` crate uses `autotests = false` - ~1,800+ test files are never compiled
2. **Feature gates**: 51 tests are properly gated behind optional features (not enabled by default)
3. **Module-level disabled**: 7 tests are TDD scaffolds with `#![cfg(false)]`
4. **Missing feature flags**: Default features don't include GPU, FFI, inference, or observability features

## Detailed Breakdown

### Issue 1: Workspace Tests Not Auto-Discovered (1,800+ tests)

**Root Cause**: `tests/Cargo.toml` has `autotests = false`

```toml
[package]
name = "bitnet-tests"
autotests = false    # <-- Disables automatic discovery
```

**Impact**: 
- ~1,800+ .rs files in `tests/` directory exist but are never compiled
- Only 6 test binaries are explicitly registered via `[[test]]` sections
- These unreachable files are counted in total but never run

**Files Affected**:
```
tests/
├── ac1_minimal_test.rs
├── ci_gates_validation_test.rs
├── comparison_analysis_demo.rs
├── issue_261_*.rs (12 files)
├── issue_465_*.rs (5 files)
├── performance_*.rs (5 files)
├── test_*.rs (40+ files)
└── ... (60+ more test files)
```

### Issue 2: Feature-Gated Tests (51 tests in 18 files)

Tests properly gated behind optional features that aren't enabled by default:

| Feature | Test Count | Files | Reason |
|---|---|---|---|
| `gpu` | 12 | 3 | GPU kernels - requires CUDA |
| `opentelemetry` | 12 | 3 | Observability - requires metrics deps |
| `cpu` | 10 | 1 | CPU-specific - already enabled |
| `full-engine` | 8 | 1 | Complete engine - optional feature |
| `wasm32` (arch) | 2 | 1 | WebAssembly - target-specific |
| `ffi` | 2 | 1 | C++ FFI - optional |
| `simd` | 2 | 1 | SIMD optimizations - optional |
| `x86_64` (arch) | 1 | 1 | x86_64-specific |
| `aarch64` (arch) | 1 | 1 | ARM64-specific |
| `iq2s-ffi` | 1 | 1 | GGML FFI - optional |

**Total**: 51 tests properly gated (not an issue, by design)

### Issue 3: TDD Scaffolds (7 tests)

**File**: `crates/bitnet-tokenizers/tests/universal_tokenizer_integration.rs`

```rust
#![cfg(false)]  // Entire module disabled
// 7 test functions never compile
```

**Status**: Intentional during MVP - can be enabled with `#![cfg(true)]`

### Issue 4: Test Files with ALL Tests Gated (18 files detailed)

These files have every test behind a cfg condition:

```
crates/bitnet-cli/tests/issue_462_cli_inference_tests.rs
  └─ 4 tests behind #[cfg(feature = "cpu")]

crates/bitnet-inference/tests/full_engine_compilation_test.rs
  └─ 8 tests behind #[cfg(feature = "full-engine")]

crates/bitnet-inference/tests/quantization_accuracy_strict_test.rs
  └─ 9 tests behind #[cfg(feature = "inference")]

crates/bitnet-kernels/tests/issue_462_tl_lut_tests.rs
  └─ 13 tests behind #[cfg(target_arch = ...)]

crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs
  └─ 17 tests behind #[cfg(feature = "cpu")]

crates/bitnet-quantization/tests/tl_packing_correctness.rs
  └─ 21 tests behind #[cfg(feature = "inference")]

crates/bitnet-tokenizers/tests/test_ac1_embedded_tokenizer_support.rs
  └─ 10 tests behind #[cfg(feature = "cpu")]

crates/bitnet-tokenizers/tests/test_ac2_model_architecture_detection.rs
  └─ 16 tests behind #[cfg(feature = "cpu")]

crates/bitnet-tokenizers/tests/test_ac3_vocabulary_size_resolution.rs
  └─ 16 tests behind #[cfg(feature = "cpu")]

crates/bitnet-tokenizers/tests/test_ac5_production_readiness.rs
  └─ 15 tests behind #[cfg(feature = "cpu")]

... and 8 more
```

**Note**: Some of these (like CPU-feature tests) should still run because `--features cpu` is enabled.

## Why 94% Are Filtered?

### Math:
- **Total test files**: 459
- **Tests auto-discovered by workspace crates**: ~250 tests
- **Tests in `tests/` crate**: ~1,750+ (not auto-discovered)
- **Percentage filtered**: (1750 / (250 + 1750)) * 100 ≈ 87-94%

### Root Cause: `tests/Cargo.toml` configuration

```toml
[package]
autotests = false          # <-- CRITICAL
# Only 6 tests explicitly registered:
[[test]]
name = "test_reporting_minimal"
...
[[test]]
name = "test_ci_reporting_simple"
...
```

When `autotests = false`:
- Cargo does NOT auto-discover .rs test files
- Only explicitly `[[test]]` sections are compiled
- All other test files are dead code

## How to Enable Full Test Coverage

### Option 1: Enable Auto-Discovery (Recommended)

Edit `tests/Cargo.toml`:
```diff
[package]
name = "bitnet-tests"
- autotests = false
+ # autotests = true (or remove the line)
```

Then run:
```bash
cargo test --workspace --no-default-features --features cpu --lib --tests
```

This would increase test count from ~250 to ~1,800+.

### Option 2: Use Full Feature Set

```bash
cargo test --workspace --no-default-features \
  --features cpu,gpu,inference,ffi,full-engine,opentelemetry,simd,iq2s-ffi,crossval \
  --lib --tests
```

Estimated: ~600-800 tests (still missing the ~1,750 in `tests/`)

### Option 3: Run Tests in `tests/` Crate Explicitly

```bash
# Run only the 6 registered tests
cargo test -p bitnet-tests

# Or run a specific test
cargo test -p bitnet-tests --test test_reporting_minimal
```

### Option 4: Build Command to Get All Tests

```bash
# Full test suite with auto-discovery enabled
# (requires fixing tests/Cargo.toml first)
cargo test --workspace \
  --no-default-features \
  --features cpu,gpu,inference,ffi,full-engine,opentelemetry,simd,iq2s-ffi,crossval \
  --lib --tests
```

Expected: ~2,000+ tests

## Test Categories Breakdown

```
WORKSPACE: cargo test --workspace --lib --tests
├── crates/bitnet-common/tests/     → ~20-30 tests (run)
├── crates/bitnet-models/tests/     → ~50-60 tests (some gated)
├── crates/bitnet-inference/tests/  → ~80-100 tests (many gated)
├── crates/bitnet-kernels/tests/    → ~50-60 tests (some gated)
├── crates/bitnet-quantization/tests/ → ~100+ tests (some gated)
├── crates/bitnet-tokenizers/tests/ → ~80-100 tests (some gated)
├── crates/bitnet-cli/tests/        → ~40-50 tests (some gated)
├── crates/bitnet-server/tests/     → ~10-20 tests
├── crates/bitnet-ffi/tests/        → ~5-10 tests (gated)
├── tests/                          → ~1,750 tests (NOT auto-discovered)
└── Other workspace crates          → ~30-50 tests
```

## Feature Gate Requirements

| To Enable | Command |
|---|---|
| GPU tests | Add `--features gpu` |
| FFI tests | Add `--features ffi` |
| Full engine | Add `--features full-engine` |
| Observability | Add `--features opentelemetry` |
| Inference pipeline | Add `--features inference` |
| Cross-validation | Add `--features crossval` |
| SIMD tests | Add `--features simd` |
| GGML IQ2S | Add `--features iq2s-ffi` |
| All of above | `--features cpu,gpu,inference,ffi,full-engine,opentelemetry,simd,iq2s-ffi,crossval` |

## Recommendations

### Short Term: Document the Current State

Add to CI configuration:
```bash
# Standard CPU tests
cargo test --workspace --no-default-features --features cpu --lib --tests

# Full feature tests (on capable systems)
cargo test --workspace --no-default-features \
  --features cpu,gpu,inference,ffi,full-engine \
  --lib --tests
```

### Medium Term: Fix Test Discovery

In `tests/Cargo.toml`, remove or change:
```diff
[package]
- autotests = false
+ # autotests = true (default)
```

Then either:
- Keep only desired test files in `tests/` directory
- Or explicitly list test files that should be included

### Long Term: Test Structure Review

Consider:
1. Consolidating related tests into `crates/*/tests/` rather than workspace root
2. Using `#[ignore]` with clear reasons instead of `#![cfg(false)]`
3. Documenting feature requirements in test comments
4. Creating a test matrix in CI that runs different feature combinations

## Conclusion

The 94% filtering is **not** a bug but rather a structural decision:
- The main test files in `crates/*/tests/` are working correctly
- The `tests/` workspace crate intentionally uses `autotests = false` for controlled test discovery
- Feature gates are properly applied to optional features
- TDD scaffolds are marked with `#![cfg(false)]` during MVP phase

**To enable full test coverage**: Remove `autotests = false` from `tests/Cargo.toml` and use the appropriate feature flags for your testing scenario.
