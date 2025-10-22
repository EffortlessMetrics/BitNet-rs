# Environment Variable Testing Patterns in BitNet.rs

## Executive Summary

The BitNet.rs codebase demonstrates **two complementary approaches** for environment variable testing:

1. **RAII-based approach** (`EnvVarGuard` in kernels tests) - Manual restoration via `unsafe { env::set_var/remove_var }`
2. **Scoped approach** (`temp_env` crate with `#[serial]` macros) - Clean closure-based isolation

The codebase also reveals **critical issues** with environment variable test pollution that require urgent refactoring.

---

## 1. Current Environment Variable Testing Approaches

### 1.1 Existing RAII Guard Pattern

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/tests/support/env_guard.rs`

```rust
//! Safe environment variable management for tests
use once_cell::sync::Lazy;
use std::{env as std_env, sync::Mutex};

/// Global lock to serialize environment variable modifications across tests
pub static ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

/// RAII guard for safe environment variable management
#[derive(Debug)]
pub struct EnvVarGuard {
    key: &'static str,
    prior: Option<String>,
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl EnvVarGuard {
    /// Set an environment variable safely with automatic restoration
    pub fn set(key: &'static str, val: &str) -> Self {
        let guard = ENV_LOCK.lock().unwrap();
        let prior = std_env::var(key).ok();
        unsafe { std_env::set_var(key, val); }
        Self { key, prior, _guard: guard }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(v) = &self.prior {
                std_env::set_var(self.key, v);
            } else {
                std_env::remove_var(self.key);
            }
        }
    }
}
```

**Characteristics**:
- Uses `once_cell::Lazy<Mutex<()>>` for global serialization
- Stores prior value for restoration on drop
- Requires `#[static]` lifetimes for keys
- Thread-safe via global mutex lock held during entire test
- **Status**: ✅ Working but unused in most tests

### 1.2 Modern Scoped Pattern with temp_env

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/tests/strict_gpu_mode.rs`

```rust
use serial_test::serial;
use temp_env::with_vars;

#[test]
#[serial(bitnet_env)]
fn strict_mode_disallows_fake_gpu() {
    with_vars(
        [("BITNET_GPU_FAKE", Some("cuda")), ("BITNET_STRICT_NO_FAKE_GPU", Some("1"))],
        || {
            // Test code here
            gpu_utils::get_gpu_info();
        },
    );
}
```

**Characteristics**:
- Uses `temp_env::with_vars()` for scoped isolation
- Closure-based cleanup (automatically restored on scope exit)
- Combined with `#[serial(bitnet_env)]` for test serialization
- **Status**: ✅ Modern pattern, actively used

### 1.3 Legacy Unsafe Pattern (Issue #441 Root Cause)

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-common/tests/issue_260_strict_mode_tests.rs`

```rust
// ANTI-PATTERN - This causes test pollution
#[test]
#[ignore = "FLAKY: Environment variable pollution in workspace runs - passes in isolation"]
fn test_strict_mode_environment_variable_parsing() {
    // ❌ No serialization, no guard - races with other tests
    unsafe {
        env::remove_var("BITNET_STRICT_MODE");
    }
    let default_config = StrictModeConfig::from_env();
    assert!(!default_config.enabled);

    unsafe {
        env::set_var("BITNET_STRICT_MODE", "1");  // Race condition!
    }
    let enabled_config = StrictModeConfig::from_env();
    assert!(enabled_config.enabled);

    unsafe {
        env::remove_var("BITNET_STRICT_MODE");  // Cleanup doesn't help - already raced
    }
}
```

**Issues**:
- ❌ No serialization - concurrent tests interfere
- ❌ Manual cleanup - incomplete restoration
- ❌ Multiple set_var calls - cleanup order matters
- ✅ Marked `#[ignore]` with clear flakiness note

**Error Message from Test**:
```
FLAKY: Environment variable pollution in workspace context 
- repro rate ~50% 
- passes in isolation - tracked in issue #441
```

---

## 2. Strict Mode Configuration Implementation

### 2.1 Environment Variable Locations

**Primary sources** for strict mode env vars:

| Variable | Used By | Tests | Status |
|----------|---------|-------|--------|
| `BITNET_STRICT_MODE` | bitnet-common | issue_260_strict_mode_tests.rs | Flaky, #[ignore] |
| `BITNET_GPU_FAKE` | bitnet-kernels | strict_gpu_mode.rs | ✅ Serial |
| `BITNET_STRICT_NO_FAKE_GPU` | bitnet-kernels | strict_gpu_mode.rs | ✅ Serial |
| `BITNET_STRICT_TOKENIZERS` | bitnet-tokenizers | strict_mode.rs | ✅ Serial |
| `RUST_LOG` | Multiple | Most tests | No guards |
| `BITNET_DETERMINISTIC` | bitnet-inference | deterministic.rs | No guards |

### 2.2 Configuration Code Path

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-common/src/strict_mode.rs`

```rust
pub struct StrictModeConfig {
    enabled: bool,
    fail_on_mock: bool,
    require_quantization: bool,
    enforce_quantized_inference: bool,
    validate_performance: bool,
    // ...
}

impl StrictModeConfig {
    pub fn from_env() -> Self {
        let enabled = env::var("BITNET_STRICT_MODE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        // ...
    }
}
```

---

## 3. Tests Requiring `#[serial]` Annotation

### 3.1 Current Serialized Tests

**All in bitnet-kernels/tests/strict_gpu_mode.rs**:
```rust
#[test]
#[serial(bitnet_env)]
fn strict_mode_disallows_fake_gpu()

#[test]
#[serial(bitnet_env)]
fn normal_mode_allows_fake_gpu()

#[test]
#[serial(bitnet_env)]
fn strict_mode_works_with_real_gpu_detection()
```

**All in bitnet-tokenizers/tests/strict_mode.rs**:
```rust
#[test]
#[serial(bitnet_env)]
fn strict_mode_disallows_bpe_mock_fallback()

#[test]
#[serial(bitnet_env)]
fn strict_mode_disallows_unknown_tokenizer_fallback()

#[test]
#[serial(bitnet_env)]
fn normal_mode_allows_mock_fallback()
```

### 3.2 Tests That NEED `#[serial]` But Don't Have It

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-common/tests/issue_260_strict_mode_tests.rs`

**All marked #[ignore] with flakiness warning**:

```rust
#[test]
#[ignore = "FLAKY: Environment variable pollution..."]
fn test_strict_mode_environment_variable_parsing()

#[test]
#[ignore = "FLAKY: Environment variable pollution..."]
fn test_cross_crate_strict_mode_consistency()

#[test]
#[ignore]  // Slow QK256 tests
fn test_qk256_full_model_inference()
```

### 3.3 Config Tests with Manual Mutex (Incorrect)

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-common/tests/config_tests.rs`

```rust
// Manual mutex lock - does NOT prevent concurrent test execution
static ENV_TEST_MUTEX: Mutex<()> = Mutex::new(());

fn acquire_env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_TEST_MUTEX.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn test_env_variable_overrides() {
    let _lock = acquire_env_lock();  // ❌ Lock only synchronizes threads within test
    unsafe { env::set_var("BITNET_VOCAB_SIZE", "60000"); }
    // ...
}
```

**Problem**: Mutex synchronizes threads but NOT concurrent test processes spawned by cargo test harness.

---

## 4. Helper Modules and Test Utilities

### 4.1 Support Modules by Crate

```
crates/bitnet-kernels/tests/support/
├── mod.rs           - Exports EnvVarGuard and ComputeReceipt
├── env_guard.rs     - RAII guard implementation
└── receipt.rs       - Compute receipt tracking

crates/bitnet-common/tests/
├── issue_260_strict_mode_tests.rs  - Flaky strict mode tests
├── config_tests.rs                 - Manual mutex pattern
└── integration_tests.rs             - Basic integration tests

crates/bitnet-inference/tests/
├── strict_mode_runtime_guards.rs   - Uses EnvVarGuard?
├── issue_254_*                     - TDD scaffolding
└── performance_tracking_tests.rs   - Manual env handling

crates/bitnet-tokenizers/tests/
├── strict_mode.rs                  - ✅ Using temp_env + #[serial]
├── cross_validation_tests.rs       - No guards
└── integration_tests.rs            - No guards

tests/common/
├── env.rs                          - Environment helpers
├── config.rs                       - Config management
└── harness.rs                      - Test harness utilities
```

### 4.2 Existing Test Helpers

**Location**: `/home/steven/code/Rust/BitNet-rs/tests/common/env.rs`

```rust
// Contents would show existing env helpers
// Needs investigation - likely minimal coverage
```

---

## 5. Recommended EnvGuard Implementation Approach

### 5.1 Design Principles

1. **Scoped-first**: Use closure-based isolation (temp_env pattern)
2. **Serialized**: Always use `#[serial(bitnet_env)]` for env-touching tests
3. **No unsafe**: Hide unsafe blocks behind safe abstractions
4. **Auto-restore**: Drop guards or closure exit ensures cleanup
5. **Process-level**: Protect against cargo test parallel execution

### 5.2 Proposed Module Location

**Create**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-common/src/test_support/env_guard.rs`

**Public in**: `bitnet-common` (used by all crates)

**Export pattern**:
```rust
// bitnet-common/src/lib.rs
#[cfg(test)]
pub mod test_support;

// bitnet-common/src/test_support/mod.rs
pub mod env_guard;
pub use env_guard::{EnvGuard, EnvVarGuard};
```

### 5.3 Recommended Implementation

**Two-tiered approach**:

#### Tier 1: Scoped Guard (Preferred)

```rust
/// Scoped environment variable modification using temp_env
pub fn with_env<F>(key: &str, value: Option<&str>, f: F) 
where
    F: FnOnce(),
{
    // temp_env::with_var internally handles restoration
    temp_env::with_var(key, value, f);
}

/// Multiple variables
pub fn with_env_vars<I, K, V, F>(vars: I, f: F)
where
    I: IntoIterator<Item = (K, Option<V>)>,
    K: AsRef<str>,
    V: AsRef<str>,
    F: FnOnce(),
{
    // temp_env::with_vars for multiple variables
    temp_env::with_vars(vars, f);
}
```

#### Tier 2: RAII Guard (Fallback)

```rust
/// RAII guard for environment variables (use with #[serial] tests)
pub struct EnvGuard {
    key: String,
    prior: Option<String>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl EnvGuard {
    pub fn set(key: &str, value: &str) -> Self {
        let guard = ENV_LOCK.lock().unwrap();
        let prior = std::env::var(key).ok();
        unsafe { std::env::set_var(key, value); }
        Self { key: key.to_string(), prior, _lock: guard }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(ref v) = self.prior {
                std::env::set_var(&self.key, v);
            } else {
                std::env::remove_var(&self.key);
            }
        }
    }
}
```

### 5.4 Test Pattern Documentation

```rust
// PREFER THIS:
#[test]
#[serial(bitnet_env)]
fn test_strict_mode_enabled() {
    with_env("BITNET_STRICT_MODE", Some("1"), || {
        let config = StrictModeConfig::from_env();
        assert!(config.enabled);
    });
}

// AVOID THIS:
#[test]
fn test_strict_mode_enabled() {
    unsafe { env::set_var("BITNET_STRICT_MODE", "1"); }
    let config = StrictModeConfig::from_env();
    assert!(config.enabled);
    unsafe { env::remove_var("BITNET_STRICT_MODE"); }  // Incomplete!
}
```

---

## 6. Migration Path for Flaky Tests

### 6.1 Issue #441 Tests (High Priority)

**File**: `crates/bitnet-common/tests/issue_260_strict_mode_tests.rs`

**Current State**: All marked #[ignore] with flakiness warning

**Migration Steps**:

1. Add `#[serial(bitnet_env)]` to each test
2. Replace unsafe blocks with `with_env_vars()` helper
3. Remove manual cleanup
4. Verify deterministic pass rate (remove #[ignore])
5. Record baseline in issue #441

**Example Conversion**:

```rust
// BEFORE
#[test]
#[ignore = "FLAKY: Environment variable pollution..."]
fn test_strict_mode_environment_variable_parsing() {
    unsafe { env::remove_var("BITNET_STRICT_MODE"); }
    let default_config = StrictModeConfig::from_env();
    assert!(!default_config.enabled);

    unsafe { env::set_var("BITNET_STRICT_MODE", "1"); }
    let enabled_config = StrictModeConfig::from_env();
    assert!(enabled_config.enabled);

    unsafe { env::remove_var("BITNET_STRICT_MODE"); }
}

// AFTER
#[test]
#[serial(bitnet_env)]
fn test_strict_mode_environment_variable_parsing() {
    with_env_vars([("BITNET_STRICT_MODE", None)], || {
        let default_config = StrictModeConfig::from_env();
        assert!(!default_config.enabled);
    });

    with_env_vars([("BITNET_STRICT_MODE", Some("1"))], || {
        let enabled_config = StrictModeConfig::from_env();
        assert!(enabled_config.enabled);
    });
}
```

### 6.2 Config Tests (Medium Priority)

**File**: `crates/bitnet-common/tests/config_tests.rs`

**Current**: Uses manual `ENV_TEST_MUTEX` (insufficient)

**Fix**: Replace mutex pattern with `#[serial]` + `with_env_vars()`

```rust
// BEFORE
static ENV_TEST_MUTEX: Mutex<()> = Mutex::new(());

#[test]
fn test_env_variable_overrides() {
    let _lock = acquire_env_lock();
    unsafe { env::set_var("BITNET_VOCAB_SIZE", "60000"); }
    // ...
}

// AFTER
#[test]
#[serial(bitnet_env)]
fn test_env_variable_overrides() {
    with_env_vars([("BITNET_VOCAB_SIZE", Some("60000"))], || {
        let config = BitNetConfig::from_env().unwrap();
        assert_eq!(config.model.vocab_size, 60000);
    });
}
```

### 6.3 Other Flaky Tests

**Patterns to fix**:
- `crates/bitnet-inference/tests/performance_tracking_tests.rs` - RUST_LOG pollution
- `crates/bitnet-cli/tests/*.rs` - Model path env vars
- `crossval/tests/*.rs` - BITNET_CPP_DIR and cross-validation flags

---

## 7. Test Infrastructure Assessment

### 7.1 Serial Test Keys (Coordination Points)

From inspection of code:

```
#[serial(bitnet_env)]      - Used in GPU and tokenizer tests (standard)
#[serial]                  - Used in some issue tests (broad, may serialize too much)
                             See: issue_254_ac6_determinism_integration.rs
```

**Recommendation**: Use `#[serial(bitnet_env)]` exclusively for env tests, preserve other serial groups for different concerns.

### 7.2 Dependencies

**Already in Cargo.toml**:
```toml
serial_test = "3.2.0"    # ✅ For #[serial] macro
temp_env = "?"           # Need to verify version
once_cell = "?"          # Used by EnvVarGuard
```

**Verify**:
```bash
grep -E "serial_test|temp_env|once_cell" Cargo.toml
```

### 7.3 Coverage Gaps

**Env vars not guarded** (>100 occurrences):
- `RUST_LOG` - No serialization in most tests
- `BITNET_GGUF` - Model path env var, used by inference tests
- `RAYON_NUM_THREADS` - Parallelism control
- `BITNET_DETERMINISTIC` - Determinism flag
- `CI` environment variable

---

## 8. Integration Checklist

### Phase 1: Foundation (Week 1)
- [ ] Create `bitnet-common/src/test_support/env_guard.rs`
- [ ] Implement `with_env()` and `with_env_vars()` helpers
- [ ] Add `temp_env` dependency if missing
- [ ] Document usage patterns in module docstring
- [ ] Export from `bitnet-common` lib.rs

### Phase 2: Core Migration (Week 2)
- [ ] Migrate bitnet-kernels tests (already using similar pattern)
- [ ] Migrate bitnet-tokenizers tests (already using temp_env)
- [ ] Fix bitnet-common/tests/config_tests.rs
- [ ] Add #[serial(bitnet_env)] to all env-touching tests

### Phase 3: Flaky Test Resolution (Week 3)
- [ ] Fix issue_260_strict_mode_tests.rs
- [ ] Run with `cargo test --workspace -- --test-threads=1` to verify
- [ ] Remove #[ignore] markers
- [ ] Record baseline in issue #441

### Phase 4: Broader Coverage (Week 4)
- [ ] Fix inference tests (performance_tracking, determinism)
- [ ] Fix CLI tests (model paths, logging)
- [ ] Fix crossval tests (BITNET_CPP_DIR)
- [ ] Document standard patterns in DEVELOPMENT.md

---

## 9. Appendix: Quick Reference

### Test Pattern Checklist

```rust
// ✅ GOOD - Serialized scoped
#[test]
#[serial(bitnet_env)]
fn test_strict_mode() {
    with_env("BITNET_STRICT_MODE", Some("1"), || {
        assert!(StrictModeConfig::from_env().enabled);
    });
}

// ✅ GOOD - Serialized RAII
#[test]
#[serial(bitnet_env)]
fn test_strict_mode_raii() {
    let _guard = EnvGuard::set("BITNET_STRICT_MODE", "1");
    assert!(StrictModeConfig::from_env().enabled);
} // Guard drops here, restores automatically

// ❌ BAD - No serialization (RACES)
#[test]
fn test_strict_mode() {
    unsafe { env::set_var("BITNET_STRICT_MODE", "1"); }
    assert!(StrictModeConfig::from_env().enabled);
    unsafe { env::remove_var("BITNET_STRICT_MODE"); }
}

// ❌ BAD - Mutex but not serialized (RACES)
#[test]
fn test_strict_mode() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe { env::set_var("BITNET_STRICT_MODE", "1"); }
    // Still races with concurrent tests!
}
```

### Serial Test Registry

```rust
// Proposed standard keys:
#[serial(bitnet_env)]           // All environment variable tests
#[serial(bitnet_io)]            // File I/O tests
#[serial(bitnet_gpu)]           // GPU detection/initialization
#[serial(bitnet_model_cache)]   // Model cache tests
```

---

## References

- **Issue #441**: Environment variable test pollution
- **Issue #260**: Strict mode environment variable architecture
- **CLAUDE.md**: Project constraints and testing frameworks
- **Crates**:
  - `bitnet-kernels/tests/support/env_guard.rs` - Existing RAII guard
  - `crates/bitnet-kernels/tests/strict_gpu_mode.rs` - Modern pattern
  - `crates/bitnet-tokenizers/tests/strict_mode.rs` - Working example
  - `crates/bitnet-common/tests/issue_260_strict_mode_tests.rs` - Flaky tests

