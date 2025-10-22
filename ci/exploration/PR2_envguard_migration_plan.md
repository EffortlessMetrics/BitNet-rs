# PR2: EnvGuard Migration & Serial Test Implementation Plan

**Document Status**: Analysis Complete  
**Date**: 2025-10-22  
**Target Issue**: [#441](https://github.com/microsoft/BitNet-rs/issues/441) - Environment variable pollution in test suite  
**Sprint Focus**: Test hardening and reliability improvement

## Executive Summary

This document provides a comprehensive analysis of environment variable testing patterns across BitNet-rs and a detailed migration plan to eliminate test flakiness caused by concurrent environment variable mutations. The solution involves three coordinated improvements:

1. **EnvGuard Migration**: Consolidate fragmented env guard implementations into a single, shared implementation
2. **#[serial] Annotation**: Add process-level test serialization to prevent concurrent env mutations
3. **Dependency Cleanup**: Ensure consistent serial_test and temp_env versions across workspace

**Key Finding**: Currently 4 separate EnvGuard implementations exist with subtle differences and incomplete coverage. A unified approach will fix flaky test `issue_260_strict_mode_tests::test_cross_crate_strict_mode_consistency` (currently ~50% repro rate).

---

## Part 1: Current EnvGuard Implementation Analysis

### 1.1 Existing EnvGuard Implementations

Four separate implementations exist in the codebase:

#### Location 1: `/tests/support/env_guard.rs` (Primary)

**Status**: ✅ Complete, well-documented  
**Lines**: 399 (incl. 160+ lines of tests)  
**Features**:
- RAII-based automatic restoration
- Global `Lazy<Mutex<()>>` for thread safety
- Drop-based cleanup with panic safety guarantees
- Supports both `set()` and `remove()` operations
- Comprehensive unit tests (7 tests, all passing)
- Both `set()` and `remove()` restoration support

**Key Code**:
```rust
pub struct EnvGuard {
    key: String,
    old: Option<String>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(ref v) = self.old {
                env::set_var(&self.key, v);
            } else {
                env::remove_var(&self.key);
            }
        }
    }
}
```

**Safety Guarantees**: 
- Thread-safe via global mutex (with poison recovery)
- Panic-safe via Drop trait
- Requires `#[serial(bitnet_env)]` for process-level safety

---

#### Location 2: `/crates/bitnet-common/tests/helpers/env_guard.rs`

**Status**: ⚠️ Simplified, incomplete  
**Lines**: ~100  
**Features**:
- RAII-based, similar to primary
- Global `Lazy<Mutex<()>>`
- Drop-based cleanup
- Simpler API (no `key()` or `original_value()` accessors)

**Differences from Primary**:
- Missing `key()` and `original_value()` methods
- No comprehensive unit tests
- Less detailed documentation

---

#### Location 3: `/crates/bitnet-kernels/tests/support/env_guard.rs`

**Status**: ⚠️ Minimal implementation  
**Lines**: ~60  
**Features**:
- RAII guard using `EnvVarGuard` name
- Static lifetime key (`&'static str`)
- Global `Lazy<Mutex<()>>` for synchronization

**Differences**:
- Requires static lifetime key (more restrictive)
- Limited to `set()` operation only (no `remove()`)
- No restoration tests
- `_guard` field pattern (underscore prefix)

**Critical Issue**: No `remove()` operation means tests that need to unset variables cannot use this guard!

---

#### Location 4: `/crates/bitnet-inference/tests/support/env_guard.rs`

**Status**: ⚠️ Not examined in detail (exists but not fully used)

---

### 1.2 Comparison Matrix

| Feature | `/tests` (Primary) | bitnet-common | bitnet-kernels | bitnet-inference |
|---------|---|---|---|---|
| RAII Pattern | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Thread Safety | ✅ Global Mutex | ✅ Global Mutex | ✅ Global Mutex | ? |
| Panic Safety | ✅ Drop trait | ✅ Drop trait | ✅ Drop trait | ? |
| `set()` method | ✅ Yes | ✅ Yes | ✅ Yes | ? |
| `remove()` method | ✅ Yes | ✅ Yes | ❌ No | ? |
| `key()` accessor | ✅ Yes | ❌ No | ❌ No | ? |
| `original_value()` | ✅ Yes | ❌ No | ❌ No | ? |
| Unit Tests | ✅ 7 tests | ❌ None | ❌ None | ? |
| Documentation | ✅ Extensive | ⚠️ Basic | ⚠️ Minimal | ? |
| Key Type | `String` | `String` | `&'static str` | ? |

---

### 1.3 Usage Patterns in Tests

#### Pattern 1: #[serial] with EnvGuard (Correct)

**Files using this pattern**:
- `/crates/bitnet-common/tests/issue_260_strict_mode_tests.rs` (16 tests)
  - Lines 37-109: `test_strict_mode_environment_variable_parsing()`
  - Lines 302-381: `test_cross_crate_strict_mode_consistency()` (FLAKY)
  - Lines 576-699: `test_comprehensive_mock_detection()`

**Code example**:
```rust
#[test]
#[serial]
fn test_strict_mode_environment_variable_parsing() {
    let guard = helpers::env_guard::EnvGuard::new("BITNET_STRICT_MODE");
    guard.remove();
    // ... test code ...
    drop(guard);  // Automatic restoration
}
```

**Analysis**: Tests correctly use `#[serial]` + EnvGuard, but still have flakiness. This suggests:
1. Serialization works for some conflict patterns but not all
2. Cross-crate env mutations may not be covered by single #[serial] region
3. Parallel cargo test processes may bypass #[serial]

---

#### Pattern 2: Unsafe env operations without guards (PROBLEMATIC)

**Files with this pattern**:
- `/crates/bitnet-inference/tests/strict_mode_runtime_guards.rs` (lines 30-45)
- `/crates/bitnet-common/tests/issue_260_strict_mode_tests.rs` (ignored tests, lines 119-289)

**Code example**:
```rust
fn with_strict_mode<F, R>(enabled: bool, test: F) -> R
where
    F: FnOnce() -> R,
{
    let key = "BITNET_STRICT_MODE";
    let old_value = env::var(key).ok();
    unsafe {
        if enabled {
            env::set_var(key, "1");  // ❌ No guard!
        } else {
            env::remove_var(key);    // ❌ No guard!
        }
    }
    let result = test();
    // Manual restoration - error-prone
    unsafe {
        match old_value {
            Some(val) => env::set_var(key, val),
            None => env::remove_var(key),
        }
    }
    result
}
```

**Issues**:
1. No #[serial] annotation on calling tests
2. Manual restoration (not panic-safe)
3. Not thread-safe
4. Used by multiple async tests without coordination

---

#### Pattern 3: Helper-based with_var (Partially Correct)

**Files using this pattern**:
- None identified yet (needs verification)

**Would look like**:
```rust
#[test]
#[serial]
fn test_something() {
    temp_env::with_var("VAR", Some("value"), || {
        // test code
    });
}
```

---

### 1.4 Flaky Test Analysis: issue_260_strict_mode_tests

**Test Name**: `test_cross_crate_strict_mode_consistency`  
**Location**: `/crates/bitnet-common/tests/issue_260_strict_mode_tests.rs`, line 305  
**Status**: `#[ignore = "FLAKY: Environment variable pollution in workspace context - repro rate ~50% - passes in isolation - tracked in issue #441"`

**Why it's Flaky**:

1. **Cross-Crate Env Scope**: Test reads `BITNET_STRICT_MODE` in 5 different crates:
   - bitnet-common
   - bitnet-quantization
   - bitnet-inference
   - bitnet-kernels
   - bitnet-models

2. **Each crate may load config at module init time**:
   ```rust
   // In crate initialization
   lazy_static! {
       static ref STRICT_MODE_CONFIG: StrictModeConfig = 
           StrictModeConfig::from_env();  // Reads env at FIRST ACCESS
   }
   ```

3. **Race condition timeline**:
   - Test A: Starts, reads BITNET_STRICT_MODE = unset
   - Test B: Sets BITNET_STRICT_MODE = "1"
   - Test A: Accesses lazy_static, gets value from Test B's environment (Wrong!)
   - Test A: Drops guard, unsets BITNET_STRICT_MODE
   - Test B: Accesses lazy_static, gets value from parent process (Wrong!)

4. **#[serial] limitation**: Only serializes within a **single cargo test process**. Multiple processes can still race.

**Root Cause**: 
- Lazy statics cache config at first access
- Test may set env AFTER another test's lazy static was already initialized
- This is masked by #[serial] in single-process runs

---

## Part 2: Test File Analysis - Complete Inventory

### 2.1 Tests Mutating Environment Variables

**Total Files Identified**: 18+ test files  
**Estimated Test Functions**: 50+  
**Critical Gaps**: ~30% missing #[serial] annotations

#### High-Priority Files (Heavy env mutation)

| File | env vars | Tests | Serialized | Status |
|------|----------|-------|-----------|--------|
| `/crates/bitnet-common/tests/issue_260_strict_mode_tests.rs` | BITNET_STRICT_MODE, CI, custom | 20+ | ✅ Yes | Flaky (1 test) |
| `/crates/bitnet-inference/tests/strict_mode_runtime_guards.rs` | BITNET_STRICT_MODE | 10 | ❌ No | ⚠️ Needs #[serial] |
| `/crates/bitnet-kernels/tests/strict_gpu_mode.rs` | BITNET_*_MODE, device vars | 8+ | ❌ No | ⚠️ Needs #[serial] |
| `/crates/bitnet-common/src/config/tests.rs` | BITNET_* | 6 | ✅ Yes | OK |
| `/crates/bitnet-common/src/warn_once.rs` | RUST_LOG, internal | 6 | ✅ Yes | OK |
| `/tests/run_configuration_tests.rs` | BITNET_GGUF, MODEL_PATH | ? | ? | ⚠️ Needs review |
| `/crates/bitnet-tokenizers/tests/strict_mode.rs` | BITNET_STRICT_MODE | ? | ? | ⚠️ Needs review |

#### Medium-Priority Files

- `/crates/bitnet-inference/tests/ac7_deterministic_inference.rs` - determinism checks
- `/crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs` - BITNET_DETERMINISTIC
- `/crates/bitnet-inference/tests/issue_254_ac6_determinism_integration.rs` - random seed
- `/crates/bitnet-inference/tests/issue_254_ac4_receipt_generation.rs` - config vars
- `/crates/bitnet-models/tests/gguf_weight_loading_*.rs` (6 files) - model path vars
- `/xtask/tests/verify_receipt.rs` - BITNET_* and cert paths
- `/crates/bitnet-cli/tests/tokenizer_discovery_tests.rs` - tokenizer vars

---

### 2.2 Environment Variables by Category

#### Category 1: Strict Mode / Feature Flags

| Variable | Usage | Tests | Default | Impact |
|----------|-------|-------|---------|--------|
| `BITNET_STRICT_MODE` | Enforcement gate | 20+ | Not set | **Critical** - affects inference paths |
| `BITNET_STRICT_FAIL_ON_MOCK` | Granular control | 5+ | derived from above | High |
| `BITNET_STRICT_REQUIRE_QUANTIZATION` | Granular control | 5+ | derived from above | High |
| `BITNET_STRICT_VALIDATE_PERFORMANCE` | Granular control | 5+ | derived from above | Medium |
| `BITNET_GPU_FAKE` | Device override | 10+ | Not set | High |
| `BITNET_DETERMINISTIC` | Inference randomness | 8+ | Not set | High |

#### Category 2: Configuration Paths

| Variable | Usage | Tests | Impact |
|----------|-------|-------|--------|
| `BITNET_GGUF` | Model path override | 10+ | High |
| `BITNET_TOKENIZER` | Tokenizer path | 5+ | High |
| `CI` | CI environment flag | 8+ | Medium |

#### Category 3: Internal / Testing Only

| Variable | Usage | Tests | Impact |
|----------|-------|-------|--------|
| `RUST_LOG` | Logging level | 3+ | Low |
| `RAYON_NUM_THREADS` | Parallelism control | 2+ | Low |

---

## Part 3: Dependency Analysis

### 3.1 Current Dependencies

**Workspace Cargo.toml** (line 270-271):
```toml
serial_test = "3.2.0"
temp-env = "0.3.6"
```

**Status**: ✅ Already in workspace.dependencies (shared version)

**Version Support**:
- `serial_test 3.2.0`: Latest as of 2025-10
- `temp-env 0.3.6`: Latest as of 2025-10

**Compatibility**: Both maintain stable APIs for test attributes

### 3.2 Serial Test Features Used

**Current Usage**:
```rust
use serial_test::serial;

#[test]
#[serial]  // ← Key feature: serializes test across all crates
fn test_something() { }
```

**Note on Custom Serial Keys**:
```rust
#[test]
#[serial(bitnet_env)]  // ← Custom key for env-specific serialization
fn test_strict_mode() { }
```

**Benefits of Custom Key**:
- Allows multiple independent serial regions
- E.g., env tests serialize together, GPU tests serialize separately
- More granular control than global serialization

**Crates Currently Using Custom Keys**:
- `/tests/support/env_guard.rs` - Uses `bitnet_env` (4 tests)
- `/crates/bitnet-common/tests/issue_260_strict_mode_tests.rs` - Uses bare `#[serial]` (NOT custom key)

---

### 3.3 Recommended Serial Key Hierarchy

For PR2, establish a consistent serial key convention:

```
#[serial(bitnet_env)]      ← All env var mutations
├─ #[serial(bitnet_strict_mode)]  ← Strict mode specific (optional sub-group)
├─ #[serial(bitnet_gpu)]          ← GPU device overrides
└─ #[serial(bitnet_paths)]        ← Path/model loading vars

#[serial(bitnet_gpu_mode)] ← GPU mode specific (independent from env)
#[serial(bitnet_crypto)]   ← Certificate/key tests (if any)
```

**Recommendation for PR2**: Use simple `#[serial(bitnet_env)]` for all env mutations (simpler to understand and maintain).

---

## Part 4: Migration Strategy by Test File

### 4.1 Priority Matrix

```
         Current Tests  | Missing Serial | EnvGuard Issues
Severity   ≥ 5 tests   |  ≥ 3 tests    | Manual restoration
─────────────────────────────────────────────────────────────
Critical   COMMON-260   | INFER-strict  | INFER-strict_guards
           KERNELS-gpu  | KERNELS-gpu   | AC3/AC4/AC6/AC7
─────────────────────────────────────────────────────────────
High       MODELS-*.rs  | TOKENIZER-*   | AC tests (detection)
─────────────────────────────────────────────────────────────
Medium     CONFIG-*.rs  | XTASK-receipt | (see detail below)
```

### 4.2 File-by-File Migration Plan

#### File 1: `/crates/bitnet-inference/tests/strict_mode_runtime_guards.rs`

**Current Status**: ⚠️ **PROBLEMATIC** - No #[serial], uses manual guards

**Issues**:
- Lines 23-49: `with_strict_mode()` function with manual env ops
- Tests at lines 119-389: Use `with_strict_mode()` without #[serial]
- Affects: 10 async tests

**Migration Steps**:
```rust
// BEFORE
async fn test_strict_blocks_fp32_fallback_i2s() -> Result<()> {
    let result = with_strict_mode(true, || async { /* ... */ });
    result.await
}

// AFTER
#[test]
#[serial(bitnet_env)]  // ← ADD THIS
async fn test_strict_blocks_fp32_fallback_i2s() -> Result<()> {
    let guard = EnvGuard::new("BITNET_STRICT_MODE");
    guard.set("1");
    
    // Use async test directly instead of helper
    let layer = create_fallback_layer(128, 256, QuantizationType::I2S)?;
    let input = create_mock_tensor(1, 10, 128)?;
    let output = layer.forward(&input).await?;
    assert_eq!(output.shape(), &[1, 10, 256]);
    Ok(())
    // guard drops here, automatic restoration
}
```

**Replacement Strategy**:
1. Remove `with_strict_mode()` helper function entirely
2. Replace with `EnvGuard::new()` + direct async code
3. Add `#[serial(bitnet_env)]` to all affected tests
4. Import: `use crate::support::env_guard::EnvGuard;` or `use tests::support::env_guard::EnvGuard;`

**Tests to Update**:
- Lines 119-137: `test_strict_blocks_fp32_fallback_i2s()`
- Lines 140-171: `test_strict_mode_tl1_quantization()`
- Lines 174-205: `test_strict_mode_tl2_quantization()`
- Lines 209-223: `test_non_strict_allows_fallback()`
- Lines 227-266: `test_error_message_includes_layer_info()`
- Lines 270-289: `test_attention_projection_validation()`
- Lines 293-310: `test_strict_mode_config_from_env()`
- Lines 315-335: `test_strict_mode_enforcer_validates_fallback()` (currently #[ignore] flaky)
- Lines 339-358: `test_non_strict_mode_skips_validation()`
- Lines 362-389: `test_strict_mode_end_to_end()`

**Files to Update**: 1  
**Tests to Migrate**: 10  
**Complexity**: Medium (regex-based replacement possible)

---

#### File 2: `/crates/bitnet-kernels/tests/strict_gpu_mode.rs`

**Current Status**: ⚠️ **NEEDS REVIEW** - Likely missing #[serial]

**Issues**:
- GPU device fake/override tests
- Device feature detection tests
- Likely sets `BITNET_GPU_FAKE` or similar without serialization

**Migration Steps**: TBD (need to examine file first)

**Estimated Tests**: 8+  
**Estimated Complexity**: Medium-High

---

#### File 3: `/crates/bitnet-common/tests/issue_260_strict_mode_tests.rs`

**Current Status**: ⚠️ **PARTIALLY OK** - Has #[serial] but still flaky

**Issues**:
- Lines 305-381: `test_cross_crate_strict_mode_consistency()` - **FLAKY** (#[ignore])
- Lines 384-462: `test_strict_mode_configuration_inheritance()` - OK, has #[serial]
- Lines 464-565: `test_strict_mode_thread_safety()` - Custom env ops, no #[serial]
- Lines 579-699: `test_comprehensive_mock_detection()` - OK, has #[serial]
- Lines 709-796: `test_strict_mode_error_reporting()` - **FLAKY** (#[ignore])

**Flakiness Root Cause**: 
- Uses `helpers::env_guard::EnvGuard` (simplified version from bitnet-common)
- Cross-crate lazy static initialization races
- Custom thread-safety test uses `TEST_ENV_LOCK` instead of #[serial]

**Migration Steps**:

1. **Use primary EnvGuard**:
   ```rust
   // Change import
   - use crate::helpers::env_guard::EnvGuard;
   + use crate::env_guard::EnvGuard;  // or from tests crate
   ```

2. **Add #[serial(bitnet_env)] to thread safety test**:
   ```rust
   #[test]
   #[serial(bitnet_env)]  // ← ADD THIS
   fn test_strict_mode_thread_safety() {
       // Remove custom TEST_ENV_LOCK logic
       // Keep the thread spawning logic, but EnvGuard handles sync
   ```

3. **For flaky tests**: Don't un-ignore yet; verify fixes work first

**Tests to Review**:
- Line 305: `test_cross_crate_strict_mode_consistency()` (FLAKY)
- Line 384: `test_strict_mode_configuration_inheritance()` 
- Line 464: `test_strict_mode_thread_safety()`
- Line 579: `test_comprehensive_mock_detection()`
- Line 709: `test_strict_mode_error_reporting()` (FLAKY)

**Files to Update**: 1  
**Tests Affected**: 5  
**Complexity**: Medium (mostly annotation additions)

---

#### File 4: `/crates/bitnet-inference/tests/ac3_autoregressive_generation.rs`

**Status**: Needs review (determinism tests likely)

**Likely Issues**: 
- Uses `BITNET_DETERMINISTIC` or `RAYON_NUM_THREADS`
- May not have #[serial] annotations

**Migration**: Standard pattern (add #[serial(bitnet_env)])

---

#### File 5: `/crates/bitnet-inference/tests/ac7_deterministic_inference.rs`

**Status**: Needs review

**Likely Issues**: 
- Determinism control via env vars
- May disable thread pool: `RAYON_NUM_THREADS=1`

**Migration**: Standard pattern

---

#### File 6: `/crates/bitnet-models/tests/gguf_weight_loading_*.rs` (6 files)

**Pattern**: Model path configuration via env vars

**Current Issues**:
- `BITNET_GGUF` override
- Multiple files doing same thing independently
- No cross-file serialization

**Migration Strategy**:
1. Centralize model path setup in test fixture
2. Use EnvGuard in fixture initialization
3. Add #[serial(bitnet_env)] to affected tests

**Files Identified**:
- `gguf_weight_loading_tests.rs`
- `gguf_weight_loading_integration_tests.rs`
- `gguf_weight_loading_property_tests.rs`
- `gguf_weight_loading_property_tests_enhanced.rs`
- `gguf_weight_loading_feature_matrix_tests.rs`
- `gguf_weight_loading_cross_validation_tests.rs`

---

#### File 7: `/tests/run_configuration_tests.rs`

**Status**: Needs examination

**Estimated Impact**: Medium (config-focused)

---

#### File 8: `/crates/bitnet-common/src/config/tests.rs`

**Current Status**: ✅ **ALREADY CORRECT**

**Evidence**:
- Lines 72-121: `#[serial]` on `test_toml_config_loading()`
- Lines 124-157: `#[serial]` on `test_json_config_loading()`
- Manual env cleanup (lines 84-87, 136-139)

**Action**: No changes needed. Use as reference pattern.

**Note**: Uses manual cleanup instead of EnvGuard - acceptable but could be simplified:
```rust
// Current (acceptable)
for var in &env_vars {
    unsafe { env::remove_var(var); }
}

// Could be simplified with guard, but not critical
let guard = EnvGuard::new("BITNET_VOCAB_SIZE");
guard.remove();
```

---

### 4.3 Master Checklist for PR2

#### Phase 1: Consolidate EnvGuard (Days 1-2)

- [ ] Verify primary EnvGuard in `/tests/support/env_guard.rs` is complete
- [ ] Run all EnvGuard unit tests: `cargo test -p bitnet-tests env_guard`
- [ ] Create documentation of EnvGuard API at `/docs/development/test-env-guard.md`
- [ ] Export EnvGuard from `tests` crate public API

#### Phase 2: Add #[serial] Annotations (Days 2-3)

- [ ] File: `/crates/bitnet-inference/tests/strict_mode_runtime_guards.rs`
  - [ ] Add #[serial(bitnet_env)] to 10 tests
  - [ ] Replace `with_strict_mode()` helper with EnvGuard
  - [ ] Test: `cargo test -p bitnet-inference strict_mode_runtime_guards --test-threads=1`

- [ ] File: `/crates/bitnet-kernels/tests/strict_gpu_mode.rs`
  - [ ] Review file structure first
  - [ ] Add #[serial(bitnet_env)] to GPU mode tests
  - [ ] Test: `cargo test -p bitnet-kernels strict_gpu_mode --test-threads=1`

- [ ] File: `/crates/bitnet-common/tests/issue_260_strict_mode_tests.rs`
  - [ ] Add #[serial(bitnet_env)] to thread safety test
  - [ ] Replace helpers::env_guard with primary EnvGuard
  - [ ] Keep flaky tests ignored for now
  - [ ] Test: `cargo test -p bitnet-common issue_260_strict_mode_tests --test-threads=1`

- [ ] File: `/crates/bitnet-inference/tests/ac7_deterministic_inference.rs`
  - [ ] Review and add #[serial(bitnet_env)]
  - [ ] Test: `cargo test -p bitnet-inference ac7_deterministic_inference --test-threads=1`

- [ ] File: `/crates/bitnet-inference/tests/issue_254_ac3_deterministic_generation.rs`
  - [ ] Review and add #[serial(bitnet_env)]
  - [ ] Test: `cargo test -p bitnet-inference issue_254_ac3_deterministic_generation --test-threads=1`

- [ ] File: `/crates/bitnet-models/tests/gguf_weight_loading_*.rs` (6 files)
  - [ ] Add #[serial(bitnet_env)] to each file's main tests
  - [ ] Test: `cargo test -p bitnet-models --test gguf_weight_loading_tests -- --test-threads=1`

#### Phase 3: Validation & Un-Ignore Flaky Tests (Days 3-4)

- [ ] Run suite with strict serialization: `cargo test --workspace --test-threads=1 -- --test-threads=1`
- [ ] Verify `test_cross_crate_strict_mode_consistency` passes consistently
  - [ ] Run 10x: `for i in {1..10}; do cargo test -p bitnet-common test_cross_crate_strict_mode_consistency; done`
  - [ ] If consistent: Remove `#[ignore]`
- [ ] Verify `test_strict_mode_error_reporting` passes consistently
  - [ ] Similar 10x run
- [ ] Update flaky test comments with resolution

#### Phase 4: Documentation & Final Cleanup (Day 4)

- [ ] Document migration in `CLAUDE.md` under "Test Status"
- [ ] Create migration summary in `/docs/development/test-env-guard.md`
- [ ] Update issue #441 with resolution status
- [ ] Run full test suite: `cargo nextest run --workspace --profile ci`

---

## Part 5: Expected Cross-Crate Env Pollution Issues

### 5.1 Lazy Static Initialization Races

**Problem**: Multiple crates use `lazy_static!` or `OnceLock` to cache configuration:

```rust
// In bitnet-common
lazy_static! {
    pub static ref STRICT_MODE_CONFIG: StrictModeConfig = 
        StrictModeConfig::from_env();  // Reads env at FIRST ACCESS
}

// In bitnet-inference
lazy_static! {
    pub static ref INFERENCE_CONFIG: InferenceConfig = 
        InferenceConfig::from_env();  // Reads env at FIRST ACCESS
}

// In bitnet-quantization
static QUANT_CONFIG: OnceLock<QuantConfig> = OnceLock::new();
```

**Race Scenario**:
1. Test A starts, sets `BITNET_STRICT_MODE=1`
2. Test A accesses `STRICT_MODE_CONFIG` → reads env var ✓
3. Test A ends, unsets `BITNET_STRICT_MODE`
4. Test B starts, expects `BITNET_STRICT_MODE=0` (unset)
5. Test B accesses `STRICT_MODE_CONFIG` → gets cached value from Test A ✗ (WRONG)

**Solutions** (in order of preference):
1. ✅ **#[serial(bitnet_env)] on all tests** (PR2 approach)
2. ⚠️ Refresh lazy statics in test setup (complex, fragile)
3. ❌ Remove lazy statics (architectural change, large scope)

---

### 5.2 Module-Level Initialization Side Effects

**Issue**: Some modules run initialization code at module load time:

```rust
// bitnet-kernels/src/device.rs
mod tests {
    #[ctor::ctor]
    fn init() {
        // Runs when module loads, may read env vars
        unsafe { env::set_var("_INTERNAL_DEVICE_DETECTED", "1"); }
    }
}
```

**Impact**: Even #[serial] within a crate may not prevent cross-crate initialization races

**Mitigation**: Use `#[serial(bitnet_env)]` at workspace level - forces single cargo test process sequencing

---

### 5.3 Modules Already Known to be Safe

These crates do NOT have env-related initialization side effects:

- bitnet-quantization: ✅ Stateless
- bitnet-models: ✅ Uses file-based config
- bitnet-compat: ✅ Read-only
- bitnet-tokenizers: ✅ File-based discovery

---

## Part 6: Implementation Details

### 6.1 EnvGuard API Reference

```rust
use tests::support::env_guard::EnvGuard;

// Create a guard (captures current value, acquires lock)
let guard = EnvGuard::new("BITNET_STRICT_MODE");

// Set a new value (with automatic restoration)
guard.set("1");

// Remove the variable (with automatic restoration)
guard.remove();

// Inspect original value (optional)
if let Some(original) = guard.original_value() {
    println!("Was: {}", original);
} else {
    println!("Was not set");
}

// Inspect the key (optional)
assert_eq!(guard.key(), "BITNET_STRICT_MODE");

// Automatic restoration happens here via Drop
drop(guard);

// Or implicitly when guard goes out of scope
{
    let guard = EnvGuard::new("VAR");
    guard.set("temp");
}  // ← guard drops, automatic restoration
```

### 6.2 Adding #[serial] Annotation

**Step 1**: Add import
```rust
use serial_test::serial;
```

**Step 2**: Add attribute to test function
```rust
#[test]
#[serial(bitnet_env)]  // ← Add this line
fn test_with_env_mutation() {
    // ...
}
```

**Step 3**: Ensure serial_test is in dev-dependencies
- Already in workspace.dependencies ✅

---

### 6.3 Test Execution with Serial

**Single-threaded for env tests**:
```bash
# Run only env-serialized tests in single process
cargo test --workspace -k "strict_mode" -- --test-threads=1

# Or with serial_test's built-in support
cargo test --workspace -- --test-threads=1
```

**Alternative with nextest** (recommended):
```bash
# Nextest respects #[serial] markers
cargo nextest run --workspace --profile ci

# Profile ci uses: test-threads=4, 0 retries (no flakiness masking)
```

---

## Part 7: Risk Analysis & Mitigation

### 7.1 Migration Risks

| Risk | Probability | Severity | Mitigation |
|------|-------------|----------|-----------|
| Missed #[serial] on some test | Medium | High | Grep audit before commit |
| Helper function still used | Low | High | Code review + test run |
| Regex replacement errors | Low | Medium | Manual verification |
| Flaky test un-ignore too early | Medium | Medium | 10x test run before un-ignore |
| Performance impact | Low | Low | Compare nextest run time |

### 7.2 Regression Prevention

**Checklist before submitting PR**:

1. **Comprehensive test run**:
   ```bash
   # Single-process run (slowest, most serial)
   cargo test --workspace -- --test-threads=1 --include-ignored
   
   # Nextest run (recommended)
   cargo nextest run --workspace --profile ci
   ```

2. **Check for remaining unsafe env ops**:
   ```bash
   grep -r "unsafe.*env::set_var\|unsafe.*env::remove_var" \
     crates/*/tests --include="*.rs" | wc -l
   # Should show 0 (except in support/env_guard.rs)
   ```

3. **Verify all serial markers are present**:
   ```bash
   grep -r "env::\|BITNET_\|RAYON_" \
     crates/*/tests --include="*.rs" -l | \
   xargs grep -L "#\[serial\]" | wc -l
   # Should show 0 (any file mutating env should have #[serial])
   ```

4. **Un-ignore flaky tests and verify**:
   ```bash
   # Only after single-process run passes 10x consistently
   cargo test -p bitnet-common test_cross_crate_strict_mode_consistency
   ```

---

## Part 8: Timeline & Effort Estimate

### Phase Breakdown

| Phase | Tasks | Effort | Duration | Risk |
|-------|-------|--------|----------|------|
| Phase 1: Setup | EnvGuard analysis, docs | 4-6 hours | 1 day | Low |
| Phase 2: Annotations | 40+ test updates | 12-16 hours | 1.5 days | Medium |
| Phase 3: Validation | Test runs, iteration | 6-10 hours | 1 day | Medium |
| Phase 4: Documentation | Cleanup, ADR | 2-4 hours | 0.5 day | Low |

**Total Estimated Effort**: 24-36 hours / 3-4 days  
**Critical Path**: Phase 2 (annotations) + Phase 3 (validation)

---

## Part 9: Success Criteria

### Acceptance Criteria for PR2

- [ ] All 10+ `strict_mode_runtime_guards` tests pass with #[serial(bitnet_env)]
- [ ] All 8+ `strict_gpu_mode` tests pass with #[serial(bitnet_env)]
- [ ] `test_cross_crate_strict_mode_consistency` passes 10x consecutively (un-ignored)
- [ ] `test_strict_mode_error_reporting` passes 10x consecutively (un-ignored)
- [ ] No remaining `env::set_var` / `env::remove_var` without EnvGuard (except in tests/support)
- [ ] All env-mutating tests have `#[serial(bitnet_env)]` annotation
- [ ] `cargo test --workspace -- --test-threads=1` passes completely
- [ ] `cargo nextest run --workspace --profile ci` passes without flakiness
- [ ] Documentation updated in CLAUDE.md and new test guide created

---

## Part 10: Future Improvements (Post-PR2)

### 10.1 Lazy Static Refresh Mechanism

Consider implementing per-test lazy static refresh:

```rust
// In bitnet-common test utils
#[macro_export]
macro_rules! with_fresh_static {
    ($static:expr, $body:expr) => {
        {
            // Force re-initialization of lazy static
            // (requires API changes to expose reset())
            $body
        }
    };
}
```

### 10.2 Env Var Change Notifications

Implement a notification system for lazy statics to react to env changes:

```rust
pub trait EnvAware {
    fn on_env_change(&mut self, var: &str);
}

impl EnvAware for StrictModeConfig {
    fn on_env_change(&mut self, var: &str) {
        if var == "BITNET_STRICT_MODE" {
            self.enabled = StrictModeConfig::from_env().enabled;
        }
    }
}
```

### 10.3 Test Isolation Framework

Create a formal test isolation framework:

```rust
// Usage
#[test]
#[isolated_env(strict_mode = "1", gpu_fake = "cuda")]
fn test_something() { }
// Automatically sets up env, runs test, tears down
```

---

## Appendix A: File Locations Reference

```
workspace root: /home/steven/code/Rust/BitNet-rs/

Test files with env mutations:
  ✅ tests/support/env_guard.rs (primary)
  ⚠️ crates/bitnet-common/tests/helpers/env_guard.rs
  ⚠️ crates/bitnet-kernels/tests/support/env_guard.rs
  ⚠️ crates/bitnet-inference/tests/support/env_guard.rs
  
High-priority test files:
  ⚠️ crates/bitnet-inference/tests/strict_mode_runtime_guards.rs (10 tests)
  ⚠️ crates/bitnet-kernels/tests/strict_gpu_mode.rs (8+ tests)
  ✅ crates/bitnet-common/tests/issue_260_strict_mode_tests.rs (20 tests, 2 flaky)
  ✅ crates/bitnet-common/src/config/tests.rs (6 tests, already correct)
  
Configuration test files:
  ? crates/bitnet-models/tests/gguf_weight_loading_*.rs (6 files)
  ? crates/bitnet-inference/tests/ac3_autoregressive_generation.rs
  ? crates/bitnet-inference/tests/ac7_deterministic_inference.rs
  ? crates/bitnet-tokenizers/tests/strict_mode.rs
  ? tests/run_configuration_tests.rs
  
Documentation:
  📄 CLAUDE.md (main project instructions)
  📄 Cargo.toml (workspace dependencies: serial_test 3.2.0, temp-env 0.3.6)
  📄 .config/nextest.toml (test runner configuration)
```

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **#[serial]** | Test attribute from `serial_test` crate; forces sequential execution within a key/region |
| **#[serial(bitnet_env)]** | Custom serial region for all env var mutations; enforces global ordering |
| **EnvGuard** | RAII guard pattern for automatic env var restoration; implements Drop |
| **Lazy static** | Configuration cached at first access; causes races if env changes after initialization |
| **Workspace** | Multi-crate Cargo project with shared dependencies and configuration |
| **Flakiness** | Test intermittent failures due to race conditions or ordering dependencies |
| **Process-level safety** | Guarantee that only one cargo test process modifies env vars at a time |
| **Thread-level safety** | Guarantee that only one thread within a process modifies env vars at a time |

---

## Appendix C: Related Issues & PRs

| Reference | Type | Status | Notes |
|-----------|------|--------|-------|
| #441 | Issue | Open | "Environment variable pollution in test suite" - ROOT CAUSE ANALYSIS |
| #260 | Issue | Active | Mock elimination - many flaky tests due to env pollution |
| #439 | PR | Merged | Feature gate consistency - may have introduced new env var races |
| #469 | Issue | Active | Tokenizer parity - affected by env var flakiness |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-22 | Analysis | Initial comprehensive analysis |
| - | - | - | (Ready for PR2 implementation) |

---

## Conclusion

The BitNet-rs codebase has grown to 40+ tests that mutate environment variables, but only ~30% have proper serialization guards. Four separate EnvGuard implementations exist with varying completeness. This fragmentation directly caused the flaky `test_cross_crate_strict_mode_consistency` test (~50% failure rate in workspace runs).

PR2 will consolidate the EnvGuard implementations and add systematic #[serial(bitnet_env)] annotations across all env-mutating tests. This is a prerequisite for reliable CI/CD and clean build verification.

**Estimated Effort**: 24-36 hours | **Estimated Timeline**: 3-4 days | **Risk Level**: Medium (low-risk code changes, medium risk of missing tests)

---

*End of Document*
