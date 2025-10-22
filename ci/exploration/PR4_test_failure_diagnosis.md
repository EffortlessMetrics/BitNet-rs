# PR4 Failing Inference Test Diagnosis Report

**Date**: 2025-10-22  
**Test**: `test_strict_mode_enforcer_validates_fallback`  
**Location**: `crates/bitnet-inference/tests/strict_mode_runtime_guards.rs:312-335`  
**Status**: CURRENTLY MARKED #[ignore] - FLAKY (environment pollution)

---

## Executive Summary

The test `test_strict_mode_enforcer_validates_fallback` is intentionally marked as `#[ignore]` with a note: "FLAKY: Environment variable pollution in workspace runs - passes in isolation with --test-threads=1". This is a **correctness issue**, not a schema/API mismatch.

When the test runs:
1. **In isolation**: PASSES ✓
2. **With parallel tests (test-threads > 1)**: FLAKY (intermittent failures due to global state)
3. **In workspace runs**: May fail due to concurrent tests modifying `BITNET_STRICT_MODE` env var

The root cause is **thread-unsafe global state mutation** combined with test isolation issues.

---

## Root Cause Analysis

### Problem 1: Global OnceLock State Pollution

**Location**: `crates/bitnet-common/src/strict_mode.rs:11`

```rust
static STRICT_MODE_CONFIG: OnceLock<StrictModeConfig> = OnceLock::new();

impl StrictModeConfig {
    pub fn from_env() -> Self {
        let enabled = env::var("BITNET_STRICT_MODE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        // ... creates config based on environment
    }
}

impl StrictModeEnforcer {
    pub fn new() -> Self {
        Self::with_config(None)
    }
    
    pub fn with_config(config: Option<StrictModeConfig>) -> Self {
        let config = config
            .unwrap_or_else(|| 
                STRICT_MODE_CONFIG
                    .get_or_init(StrictModeConfig::from_env)
                    .clone()
            );
        Self { config }
    }
    
    // GOOD: Bypasses OnceLock for testing
    pub fn new_fresh() -> Self {
        let config = StrictModeConfig::from_env();
        Self { config }
    }
}
```

**The Issue**:
- `OnceLock::get_or_init()` caches the config **the first time it's called**
- Once cached, subsequent calls return the cached value **regardless of environment changes**
- Test modifies `BITNET_STRICT_MODE=1` via `env::set_var()`
- But OnceLock may still return the cached config from a previous test

### Problem 2: Environment Variable Mutation in Tests

**Location**: `crates/bitnet-inference/tests/strict_mode_runtime_guards.rs:22-49`

```rust
fn with_strict_mode<F, R>(enabled: bool, test: F) -> R
where
    F: FnOnce() -> R,
{
    let key = "BITNET_STRICT_MODE";
    let old_value = env::var(key).ok();

    unsafe {
        if enabled {
            env::set_var(key, "1");  // Modifies GLOBAL state
        } else {
            env::remove_var(key);     // Modifies GLOBAL state
        }
    }

    let result = test();

    // Attempts to restore - but may fail if other tests run concurrently
    unsafe {
        match old_value {
            Some(val) => env::set_var(key, val),
            None => env::remove_var(key),
        }
    }

    result
}
```

**The Issue**:
- `env::set_var()` is **not thread-safe** for concurrent tests
- Test 1 (thread A): Sets `BITNET_STRICT_MODE=1`
- Test 2 (thread B): Meanwhile, checks `BITNET_STRICT_MODE` → sees Test 1's value (RACE CONDITION)
- Test 1: Restores original value
- Test 2: Now sees wrong environment state

### Problem 3: Assertion Expects Specific Error Format

**Location**: `crates/bitnet-inference/tests/strict_mode_runtime_guards.rs:329-333`

```rust
if let Err(BitNetError::StrictMode(msg)) = result {
    assert!(msg.contains("FP32 fallback"), "Error should mention FP32 fallback: {}", msg);
    assert!(msg.contains("128"), "Error should include dimensions: {}", msg);
    assert!(msg.contains("256"), "Error should include dimensions: {}", msg);
}
```

**Potential Issue**: If the error format changes, assertions fail.

Current error message format (`strict_mode.rs:151-154`):
```rust
Err(BitNetError::StrictMode(format!(
    "Strict mode: FP32 fallback rejected - qtype={:?}, device={:?}, layer_dims={:?}, reason={}",
    quantization_type, device, layer_dimensions, fallback_reason
)))
```

Message would be: `"Strict mode: FP32 fallback rejected - qtype=I2S, device=Cpu, layer_dims=[128, 256], reason=test_kernel_unavailable"`

This **DOES contain** the required strings:
- ✓ "FP32 fallback" (literal match)
- ✓ "128" (in array repr)
- ✓ "256" (in array repr)

---

## Current vs Expected Receipt Structure

### Current InferenceReceipt Schema (v1.0.0)

**Location**: `crates/bitnet-inference/src/receipts.rs:169-223`

```rust
pub struct InferenceReceipt {
    pub schema_version: String,        // "1.0.0" ✓
    pub timestamp: String,              // ISO 8601 ✓
    pub compute_path: String,           // "real" | "mock" ✓
    pub backend: String,                // "cpu" | "cuda" | "metal" ✓
    pub kernels: Vec<String>,           // ["i2s_gemv", ...] ✓
    pub deterministic: bool,            // true/false ✓
    pub environment: HashMap<String, String>,  // BITNET_*, RAYON_*, system info ✓
    pub model_info: ModelInfo,          // Config metadata ✓
    pub test_results: TestResults,      // Pass/fail/skipped counts ✓
    pub performance_baseline: PerformanceBaseline,  // TPS, latency, memory ✓
    pub cross_validation: Option<CrossValidation>, // Deprecated ✓
    pub parity: Option<ParityMetadata>, // C++ comparison metrics ✓
    pub corrections: Vec<CorrectionRecord>, // LayerNorm fixes ✓
}
```

### Fields Validated by Receipt Verification

**From**: `crates/bitnet-inference/src/receipts.rs:325-396`

| Field | Validation | Required |
|-------|-----------|----------|
| `schema_version` | Must equal "1.0.0" | YES |
| `compute_path` | Must be "real" (not "mock") | YES |
| `kernels` | Non-empty, no empty strings, no "mock" (case-insensitive), ≤128 chars each, ≤10K total | YES |
| `backend` | Free-form string ("cpu"/"cuda"/"metal") - not validated | NO |
| `test_results.failed` | Must be 0 | YES |
| `test_results.accuracy_tests` | If present, all metrics must pass (MSE ≤ tolerance) | NO |
| `test_results.determinism_tests` | If deterministic=true, must have identical_sequences=true | NO |
| `corrections` | May be present; gated by env var in CI | NO |

### Example Receipt from ci/inference.json

```json
{
  "backend": "cpu",
  "compute_path": "real",
  "deterministic": true,
  "environment": {
    "BITNET_VERSION": "0.1.0",
    "OS": "linux-x86_64",
    "RUST_VERSION": "rustc 1.92.0"
  },
  "kernels": ["mock_inference"],        // ❌ PROBLEM: Contains "mock"
  "model": {
    "path": "/path/to/model.gguf"
  },
  "schema_version": "1.0.0",
  "timestamp": "2025-10-16T21:20:38.669561345+00:00",
  "tokens_generated": 4,
  "tokens_per_second": 200.0,
  "tokens_requested": 4
}
```

**Note**: This receipt has `compute_path="real"` but `kernels=["mock_inference"]` - this is **INVALID** and would fail `validate_kernel_ids()`.

---

## Two Solutions

### Solution A: Fix Environment Variable Race Condition (RECOMMENDED)

**Approach**: Eliminate global mutable state and use test-local config instead of OnceLock.

#### A.1: Add Test-Friendly API

**File**: `crates/bitnet-common/src/strict_mode.rs`

```rust
#[cfg(test)]
impl StrictModeEnforcer {
    /// Create enforcer with forced configuration (test-only)
    /// This bypasses OnceLock entirely, ensuring test isolation
    pub fn new_test_with_config(enabled: bool) -> Self {
        let config = StrictModeConfig {
            enabled,
            fail_on_mock: enabled,
            require_quantization: enabled,
            enforce_quantized_inference: enabled,
            validate_performance: enabled,
            ci_enhanced_mode: false,
            log_all_validations: false,
            fail_fast_on_any_mock: false,
        };
        Self { config }
    }
}
```

#### A.2: Update Test to Use Direct Config

**File**: `crates/bitnet-inference/tests/strict_mode_runtime_guards.rs`

**Before** (lines 312-335):
```rust
#[test]
#[ignore] // FLAKY: Environment variable pollution in workspace runs
fn test_strict_mode_enforcer_validates_fallback() {
    with_strict_mode(true, || {
        let enforcer = bitnet_common::strict_mode::StrictModeEnforcer::new_fresh();
        // ... rest of test
    });
}
```

**After**:
```rust
#[test]
fn test_strict_mode_enforcer_validates_fallback() {
    // Use test-only API that doesn't depend on environment or global state
    let enforcer = bitnet_common::strict_mode::StrictModeEnforcer::new_test_with_config(true);

    let result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &[128, 256],
        "test_kernel_unavailable",
    );

    // In strict mode, fallback validation should fail
    assert!(result.is_err(), "Strict mode should reject fallback");

    if let Err(BitNetError::StrictMode(msg)) = result {
        assert!(msg.contains("FP32 fallback"), "Error should mention FP32 fallback: {}", msg);
        assert!(msg.contains("128"), "Error should include dimensions: {}", msg);
        assert!(msg.contains("256"), "Error should include dimensions: {}", msg);
    }
}
```

#### A.3: Remove Unsafe with_strict_mode Helper

```rust
// DELETE this entire helper function - no longer needed:
fn with_strict_mode<F, R>(enabled: bool, test: F) -> R
where
    F: FnOnce() -> R,
{
    // ... 27 lines of unsafe env mutation
}
```

### Benefits of Solution A:
- **✓ No flakiness**: No shared global state → test isolation guaranteed
- **✓ Simpler code**: 27 lines of unsafe code removed
- **✓ Thread-safe**: Each test gets its own StrictModeEnforcer with explicit config
- **✓ Fast**: No environment variable syscalls, just direct struct creation
- **✓ Deterministic**: Same behavior regardless of test execution order or parallelism

### Implementation Complexity: **Low** (30 min)

---

### Solution B: Quarantine with Tracking Issue (ALTERNATIVE)

**Approach**: Keep test ignored but document tracking issue properly.

#### B.1: Document Flakiness Issue

**File**: `GitHub Issues`

Create issue: "Issue #XXX: Fix environment variable pollution in strict_mode tests"

```markdown
## Problem
The test `test_strict_mode_enforcer_validates_fallback` is flaky when run in parallel.

## Root Cause
1. OnceLock caches StrictModeConfig on first access
2. Tests use unsafe env::set_var() to modify BITNET_STRICT_MODE
3. Concurrent tests see stale OnceLock values → assertions fail

## Evidence
- Passes in isolation: `cargo test --test-threads=1`
- Flaky in workspace: `cargo test` (parallel mode)
- All other 11 tests in suite pass reliably

## Solution
Replace environment-based config with explicit test API:
- Add StrictModeEnforcer::new_test_with_config(bool)
- Update test to use direct config instead of env vars
- Remove unsafe with_strict_mode() helper
```

#### B.2: Update Test Ignore Annotation

**File**: `crates/bitnet-inference/tests/strict_mode_runtime_guards.rs:314`

```rust
#[test]
#[ignore] // Issue #XXX: Environment variable pollution - test passes in isolation only
fn test_strict_mode_enforcer_validates_fallback() {
    // ... test code unchanged
}
```

#### B.3: Add to Blockers List

**File**: `CLAUDE.md`

Add to "Active Blockers" section:
```markdown
### Issue #XXX: Environment Variable Pollution in Strict Mode Tests
**Status**: Quarantined until refactoring
**Impact**: 1 test ignored; 11 others pass
**Workaround**: Run with --test-threads=1
**Fix**: Use explicit test config API instead of unsafe env mutation
```

### Benefits of Solution B:
- **✓ No code changes required**: Stable, already passing all other tests
- **✓ Documented**: Issue tracked in GitHub with root cause analysis
- **✓ Low risk**: Doesn't modify any production code

### Benefits of Solution A Over B:
- **✓ Unblocks test**: Test runs reliably as part of normal `cargo test`
- **✓ Improves code**: Removes 27 lines of unsafe code
- **✓ Better design**: Uses explicit dependency injection
- **✓ Future-proof**: Pattern reusable for other tests

---

## Recommendation: **SOLUTION A** (Fix Environment Variable Race)

**Rationale**:
1. **Highest value**: Fixes the flakiness instead of working around it
2. **Low effort**: Only requires adding ~15 lines of code
3. **Improves codebase**: Eliminates unsafe global state mutation
4. **Consistent pattern**: Matches Rust best practices for dependency injection
5. **Better test isolation**: Each test completely independent

---

## Step-by-Step Implementation Plan for Solution A

### Phase 1: Add Test Config API (5 minutes)

**File**: `crates/bitnet-common/src/strict_mode.rs`

**Location**: After the existing `impl StrictModeEnforcer` block (around line 224)

```rust
#[cfg(test)]
impl StrictModeEnforcer {
    /// Create enforcer with explicit configuration (test-only, no OnceLock)
    ///
    /// # Arguments
    /// * `enabled` - Whether strict mode is enabled
    ///
    /// # Example
    /// ```
    /// let enforcer = StrictModeEnforcer::new_test_with_config(true);
    /// assert!(enforcer.is_enabled());
    /// ```
    pub fn new_test_with_config(enabled: bool) -> Self {
        Self {
            config: StrictModeConfig {
                enabled,
                fail_on_mock: enabled,
                require_quantization: enabled,
                enforce_quantized_inference: enabled,
                validate_performance: enabled,
                ci_enhanced_mode: false,
                log_all_validations: false,
                fail_fast_on_any_mock: false,
            },
        }
    }
}
```

### Phase 2: Update Test (10 minutes)

**File**: `crates/bitnet-inference/tests/strict_mode_runtime_guards.rs`

**Step 1**: Remove `with_strict_mode` helper function (lines 22-49)

**Step 2**: Update `test_strict_mode_config_from_env()` to not use helper (lines 292-310)

```rust
#[test]
fn test_strict_mode_config_from_env() {
    // Test that environment reading works correctly
    // (This test still needs unsafe because it validates env::var reading)
    
    // Test with strict mode enabled
    unsafe {
        std::env::set_var("BITNET_STRICT_MODE", "1");
        let config = bitnet_common::strict_mode::StrictModeConfig::from_env();
        assert!(config.enabled);
        assert!(config.require_quantization);
        assert!(config.enforce_quantized_inference);
        std::env::remove_var("BITNET_STRICT_MODE");
    }

    // Test with strict mode disabled
    unsafe {
        std::env::remove_var("BITNET_STRICT_MODE");
        let config = bitnet_common::strict_mode::StrictModeConfig::from_env();
        assert!(!config.enabled);
    }
}
```

**Step 3**: Remove #[ignore] and update `test_strict_mode_enforcer_validates_fallback()` (lines 312-335)

```rust
#[test]
fn test_strict_mode_enforcer_validates_fallback() {
    // Use test config API instead of unsafe environment mutation
    let enforcer = bitnet_common::strict_mode::StrictModeEnforcer::new_test_with_config(true);

    let result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &[128, 256],
        "test_kernel_unavailable",
    );

    // In strict mode, fallback validation should fail
    assert!(result.is_err(), "Strict mode should reject fallback");

    if let Err(BitNetError::StrictMode(msg)) = result {
        assert!(msg.contains("FP32 fallback"), "Error should mention FP32 fallback: {}", msg);
        assert!(msg.contains("128"), "Error should include dimensions: {}", msg);
        assert!(msg.contains("256"), "Error should include dimensions: {}", msg);
    }
}
```

**Step 4**: Update `test_non_strict_mode_skips_validation()` (lines 338-358)

```rust
#[test]
fn test_non_strict_mode_skips_validation() {
    let enforcer = bitnet_common::strict_mode::StrictModeEnforcer::new_test_with_config(false);

    let result = enforcer.validate_quantization_fallback(
        QuantizationType::I2S,
        Device::Cpu,
        &[128, 256],
        "test_kernel_unavailable",
    );

    assert!(
        result.is_ok(),
        "validate_quantization_fallback should return Ok in non-strict mode"
    );
}
```

**Step 5**: Update other tests that use `with_strict_mode()`:

- Lines 124-136: `test_strict_blocks_fp32_fallback_i2s()` - Remove `with_strict_mode` wrapper, add #[ignore] if async test needs environment setup
- Lines 142-170: `test_strict_mode_tl1_quantization()` - Same
- Lines 175-204: `test_strict_mode_tl2_quantization()` - Same
- Lines 210-222: `test_non_strict_allows_fallback()` - Same
- Lines 230-265: `test_error_message_includes_layer_info()` - Same
- Lines 275-288: `test_attention_projection_validation()` - Same
- Lines 362-388: `test_strict_mode_end_to_end()` - Same

### Phase 3: Verify (10 minutes)

```bash
# Build
cargo build -p bitnet-common --no-default-features --features cpu

# Run all tests in isolation
cargo test -p bitnet-inference --no-default-features --features cpu \
  --test strict_mode_runtime_guards -- --test-threads=1

# Run tests in parallel (should pass now)
cargo test -p bitnet-inference --no-default-features --features cpu \
  --test strict_mode_runtime_guards -- --test-threads=4

# Run entire test suite
cargo test --workspace --no-default-features --features cpu

# Check linting
cargo clippy -p bitnet-common --all-targets -- -D warnings
cargo fmt -p bitnet-common --check
```

### Phase 4: Commit

```bash
git add -A
git commit -m "fix(strict_mode): eliminate environment variable pollution in tests

- Add StrictModeEnforcer::new_test_with_config() for test isolation
- Remove unsafe with_strict_mode() helper function
- Re-enable test_strict_mode_enforcer_validates_fallback (was #[ignore])
- Update all strict_mode tests to use explicit config instead of env vars
- Fixes Issue #XXX: Race condition in parallel test execution

This change:
- Improves test reliability (no flakiness in parallel runs)
- Removes 27 lines of unsafe code
- Follows Rust best practices (explicit dependency injection)
- Makes tests thread-safe and deterministic"
```

---

## Validation Checklist

- [ ] `cargo build -p bitnet-common --no-default-features --features cpu` (succeeds)
- [ ] `cargo test -p bitnet-inference --no-default-features --features cpu --test strict_mode_runtime_guards` (all 12 tests pass)
- [ ] `cargo test --workspace --no-default-features --features cpu` (no regressions)
- [ ] `cargo clippy -p bitnet-common --all-targets -- -D warnings` (no warnings)
- [ ] `cargo fmt --all --check` (formatted correctly)
- [ ] Test runs reliably in parallel (no flakiness detected across 10 runs)
- [ ] Document removed in CLAUDE.md "Known Issues" section

---

## Files to Modify Summary

| File | Lines | Change | Risk |
|------|-------|--------|------|
| `crates/bitnet-common/src/strict_mode.rs` | +15 | Add test config API | LOW |
| `crates/bitnet-inference/tests/strict_mode_runtime_guards.rs` | -50, +30 | Remove helper, update tests | LOW |
| `CLAUDE.md` | -5 | Remove from known issues | LOW |

**Total LOC Delta**: ~-25 lines (net reduction)  
**Total Risk**: LOW  
**Estimated Effort**: 25 minutes  
**Expected Outcome**: 100% test pass rate in all modes

---

## References

- Test file: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/strict_mode_runtime_guards.rs`
- Strict mode module: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-common/src/strict_mode.rs`
- Receipt validation: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/receipts.rs`
- Issue context: `Issue #465` (CPU path followup), `Issue #260` (Mock elimination)

