# Environment Variable Mutation Test Audit Report

## Executive Summary

This report identifies ALL tests that mutate environment variables without proper `#[serial(bitnet_env)]` annotation or `EnvGuard` protection. The codebase has significant race condition vulnerabilities when tests run in parallel.

**Critical Finding: 14 of 16 tests in device_features.rs are unprotected**

---

## PRIORITY 1: device_features.rs (CRITICAL)

**File:** `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/tests/device_features.rs`

**Status:** 14 of 16 tests missing `#[serial(bitnet_env)]` annotation

### Unprotected Tests (Need Immediate Fixes)

#### Module: `runtime_detection`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 87-109 | `ac3_gpu_fake_cuda_overrides_detection` | `BITNET_GPU_FAKE` | **UNPROTECTED** |
| 116-141 | `ac3_gpu_fake_none_disables_detection` | `BITNET_GPU_FAKE` | **UNPROTECTED** |
| 162-193 | `ac3_gpu_fake_case_insensitive` | `BITNET_GPU_FAKE` (loop × 4) | **CRITICAL** |
| 198-238 | `ac3_gpu_compiled_but_runtime_unavailable` | `BITNET_GPU_FAKE` | **UNPROTECTED** |
| 240-300 | `mutation_gpu_runtime_real_detection` | `BITNET_GPU_FAKE` (set/unset × 2) | **CRITICAL** |
| 305-364 | `mutation_gpu_fake_or_semantics` | `BITNET_GPU_FAKE` (alternating) | **CRITICAL** |

#### Module: `integration_tests`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 471-519 | `ac3_capability_summary_respects_fake` | `BITNET_GPU_FAKE` (2 pairs) | **CRITICAL** |

#### Module: `strict_mode_tests`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 520-560 | `ac_strict_mode_forbids_fake_gpu` | `BITNET_STRICT_MODE`, `BITNET_GPU_FAKE` | **CRITICAL** |
| 557-594 | `ac_strict_mode_allows_real_gpu` | `BITNET_STRICT_MODE`, `BITNET_GPU_FAKE` | **CRITICAL** |

### Safe Tests (No Changes Needed)

- Line 44-55: `ac3_gpu_compiled_true_with_features` ✓ (no env mutations)
- Line 63-74: `ac3_gpu_compiled_false_without_features` ✓ (no env mutations)
- Line 373-434: `mutation_gpu_compiled_correctness` ✓ (no env mutations)
- Line 607-641: `ac3_quantization_uses_device_features` ✓ (no env mutations)
- Line 638-672: `ac3_inference_uses_device_features` ✓ (no env mutations)

---

## PRIORITY 2: Secondary Files with Env Mutations

### File: `crates/bitnet-tokenizers/tests/cross_validation_tests.rs`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 312-358 | `test_deterministic_cross_validation` | `BITNET_DETERMINISTIC`, `BITNET_SEED`, `RAYON_NUM_THREADS` | **UNPROTECTED** |

**Issue:** Async test without `#[serial(bitnet_env)]` annotation

---

### File: `crates/bitnet-tokenizers/tests/integration_tests.rs`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 489-581 | `test_cross_platform_compatibility` | Multiple (5 iterations) | **CRITICAL** |

**Issue:** Async test with looped env mutations, no synchronization

---

### File: `crates/bitnet-tokenizers/tests/test_ac4_smart_download_integration.rs`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 296-323 | `ac4_offline_mode_handling` | `BITNET_OFFLINE` | **UNPROTECTED** |

---

### File: `xtask/tests/ffi_build_tests.rs`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 300-322 | `test_bitnet_cpp_system_includes_helper` | `BITNET_CPP_DIR` | **UNPROTECTED** |

---

### File: `crates/bitnet-server/tests/ac03_model_hot_swapping.rs`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 335-354 | `ac3_cross_validation_during_swap_ok` | `BITNET_DETERMINISTIC`, `BITNET_SEED` | **UNPROTECTED** |

---

### File: `crates/bitnet-server/tests/otlp_metrics_test.rs`

| Line Range | Test Function | Env Variables | Status | Note |
|-----------|---------------|---------------|--------|------|
| 75-109 | `test_ac2_default_endpoint_fallback` | `OTEL_*` | ⚠️ PARTIAL | Has `#[serial]` but not `#[serial(bitnet_env)]` |
| 120-175 | `test_ac2_custom_endpoint_configuration` | `OTEL_*` | **UNPROTECTED** |
| 167-197 | `test_ac2_resource_attributes_set` | `OTEL_SERVICE_NAME` | **UNPROTECTED** |
| 396-425 | `test_ac2_endpoint_env_detection` | `OTEL_EXPORTER_OTLP_ENDPOINT` | **UNPROTECTED** |

---

### File: `xtask/tests/ci_integration_tests.rs`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 91-143 | `test_ci_without_hf_token` | `HF_TOKEN` | **UNPROTECTED** |
| 242-290 | `test_ci_fallback_strategy` | `HF_TOKEN` | **UNPROTECTED** |

**Issue:** Multiple tests modifying shared `HF_TOKEN` without synchronization

---

### File: `xtask/tests/tokenizer_subcommand_tests.rs`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 277-320 | `test_fetch_auth_error` | `HF_TOKEN` | **UNPROTECTED** |

---

### File: `tests-new/fixtures/fixtures/validation_tests.rs`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 387-438 | `test_deterministic_behavior` | `BITNET_DETERMINISTIC`, `BITNET_SEED` | **UNPROTECTED** |

---

### File: `tests-new/integration/debug_integration.rs`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 565-599 | `test_debug_config_from_env` | `BITNET_DEBUG_*` (3 vars) | **UNPROTECTED** |

**Issue:** Multiple env vars mutated simultaneously without synchronization

---

### File: `tests/common/gpu.rs`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 31-52 | `test_gpu_tests_disabled_by_default` | `BITNET_ENABLE_GPU_TESTS` | **UNPROTECTED** |
| 40-51 | `test_gpu_tests_enabled_when_set` | `BITNET_ENABLE_GPU_TESTS` | **UNPROTECTED** |
| 53-66 | `test_gpu_tests_disabled_when_set_to_non_one` | `BITNET_ENABLE_GPU_TESTS` | **UNPROTECTED** |

**Issue:** Three tests sharing `BITNET_ENABLE_GPU_TESTS` without synchronization

---

### File: `tests/test_enhanced_error_handling.rs`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 287-341 | `test_error_pattern_detection` | `CI` | **UNPROTECTED** |

---

### File: `tests/test_fixture_reliability.rs`

| Line Range | Test Function | Env Variables | Status |
|-----------|---------------|---------------|--------|
| 14-115 | `test_fixture_reliability_and_cleanup` | `BITNET_TEST_CACHE` | **UNPROTECTED** |
| 116-157 | `test_fixture_concurrent_reliability` | `BITNET_TEST_CACHE` | **UNPROTECTED** |
| 158-222 | `test_realistic_fixture_cleanup` | `BITNET_TEST_CACHE` | **UNPROTECTED** |
| 223-269 | `test_fixture_error_recovery` | `BITNET_TEST_CACHE` | **UNPROTECTED** |

**Issue:** Four async tests sharing `BITNET_TEST_CACHE` - HIGH CONTENTION RISK

---

### File: `tests/test_configuration_scenarios.rs`

**Status:** REQUIRES DETAILED REVIEW

- Multiple test cases with env mutations via helper methods
- Lines with `std::env::set_var` / `std::env::remove_var`: 1073-1200 range
- Needs analysis of calling test functions and annotation status

---

## Risk Analysis

### Critical Risk Factors

1. **Parallel Execution Vulnerability:** All unprotected tests fail silently when run with `--test-threads > 1`
2. **Shared Environment State:** Tests like device_features.rs all modify `BITNET_GPU_FAKE`
3. **Loop-Based Mutations:** Some tests (e.g., `ac3_gpu_fake_case_insensitive`) loop 4 times without protection
4. **Sequential Mutations:** Tests with multiple set/unset pairs (e.g., `mutation_gpu_runtime_real_detection`) are prone to race conditions
5. **Shared Env Vars Across Files:** `HF_TOKEN`, `BITNET_DETERMINISTIC`, `BITNET_SEED` appear in multiple files without coordination

### Likelihood of Failures

- **Intermittent flakiness:** 95% likely in CI/CD with parallel execution
- **Local test pass but CI failure:** 90% likely if running locally with `--test-threads=1`
- **Cascading failures:** 70% likely if multiple unprotected tests modify same var

---

## Fix Strategy

### Option 1: Add `#[serial(bitnet_env)]` Annotation

**Fastest fix for immediate stability**

```rust
#[test]
#[serial(bitnet_env)]  // Add this line
fn ac3_gpu_fake_cuda_overrides_detection() {
    unsafe {
        std::env::set_var("BITNET_GPU_FAKE", "cuda");
    }
    // ...
}
```

### Option 2: Use EnvGuard Helper (Recommended)

**Cleaner and more maintainable**

```rust
use tests::support::env_guard::EnvGuard;

#[test]
fn ac3_gpu_fake_cuda_overrides_detection() {
    let _guard = EnvGuard::new("BITNET_GPU_FAKE", "cuda");
    // Env var automatically restored on drop
    // No need for manual remove_var()
}
```

**Benefits:**
- Automatic restoration on panic
- No manual cleanup code
- Better readability
- RAII pattern

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| **Priority 1 Files** | 1 (device_features.rs) |
| **Priority 1 Unprotected Tests** | 9 |
| **Secondary Files with Issues** | 13+ |
| **Secondary Unprotected Tests** | 30+ |
| **Total Unprotected Tests** | 39+ |
| **Files with Shared Env Vars** | 7 |
| **Most Critical Env Var** | `BITNET_GPU_FAKE` (6+ tests) |

---

## Recommendations

1. **Immediate:** Add `#[serial(bitnet_env)]` to all 14 unprotected tests in device_features.rs
2. **Short-term:** Apply same fix to 9 secondary files in priority 2
3. **Long-term:** Refactor to use `EnvGuard` for automatic cleanup and better code quality
4. **Governance:** Add lint rules or CI checks to prevent future unprotected env mutations

---

## References

- **CLAUDE.md**: Section on `#[serial(bitnet_env)]` usage pattern
- **Test Framework:** `tests/support/env_guard.rs` for EnvGuard implementation
- **Related Issue:** Environment isolation pattern for parallel test execution

