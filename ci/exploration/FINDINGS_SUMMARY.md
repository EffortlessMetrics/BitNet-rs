# Environment Variable Testing - Key Findings

**Date**: 2025-10-22  
**Status**: Complete Analysis  
**Urgency**: High (Issue #441 affects all workspace builds)

---

## Critical Issues Found

### 1. Issue #441: Environment Variable Test Pollution (HIGH PRIORITY)

**Affected Tests**: `crates/bitnet-common/tests/issue_260_strict_mode_tests.rs`

**Symptom**: Tests marked `#[ignore]` with message:
```
FLAKY: Environment variable pollution in workspace context 
- repro rate ~50% 
- passes in isolation - tracked in issue #441
```

**Root Cause**:
- No `#[serial]` coordination
- Manual `unsafe { env::set_var() }` without proper cleanup
- Multiple env var mutations in sequence without atomic guards
- Config tests use manual `Mutex` (synchronizes threads but NOT concurrent test processes)

**Impact**: 15+ tests blocked by flakiness

---

## Two Working Approaches Identified

### Approach 1: RAII Guard Pattern (Existing)

**Location**: `crates/bitnet-kernels/tests/support/env_guard.rs`

**Pros**:
- Already implemented and tested
- `once_cell::Lazy<Mutex<()>>` provides proper serialization
- Automatic restoration on drop
- No external dependencies needed

**Cons**:
- Requires `'static` key lifetime
- Requires explicit guard variable
- More verbose than scoped approach

**Status**: ✅ Works but unused

---

### Approach 2: Scoped Pattern with temp_env (Modern)

**Location**: `crates/bitnet-kernels/tests/strict_gpu_mode.rs` and `crates/bitnet-tokenizers/tests/strict_mode.rs`

**Usage**:
```rust
#[test]
#[serial(bitnet_env)]
fn test_with_env() {
    with_vars([("VAR", Some("value"))], || {
        // Code here sees VAR="value"
    }); // Automatically restored
}
```

**Pros**:
- Closure-based cleanup (impossible to forget)
- Clean, readable API
- Combined with `#[serial]` prevents races
- No unsafe code in tests

**Cons**:
- Requires `temp_env` crate dependency
- Slightly more complex scoping for multiple tests

**Status**: ✅ Already used in 6 tests successfully

---

## Test Infrastructure Assessment

### Current Guard Implementations

| Component | Implementation | Location | Status |
|-----------|---|----------|--------|
| bitnet-kernels | RAII EnvVarGuard | tests/support/env_guard.rs | ✅ Present but unused |
| bitnet-common | Manual Mutex | tests/config_tests.rs | ❌ Insufficient (no #[serial]) |
| bitnet-common | No guards | tests/issue_260_* | ❌ **FLAKY** (ignored) |
| bitnet-tokenizers | temp_env + #[serial] | tests/strict_mode.rs | ✅ **WORKING** |
| bitnet-kernels GPU | temp_env + #[serial] | tests/strict_gpu_mode.rs | ✅ **WORKING** |

### Environment Variables Requiring Guards

**High Priority** (currently unguarded, causing flakiness):
- `BITNET_STRICT_MODE` (strict mode tests)
- `BITNET_STRICT_FAIL_ON_MOCK` (validation tests)
- `BITNET_STRICT_REQUIRE_QUANTIZATION` (validation tests)

**Medium Priority** (occasionally unguarded):
- `BITNET_VOCAB_SIZE`, `BITNET_TEMPERATURE`, etc. (config tests)
- `BITNET_GPU_FAKE` (already guarded in some tests)
- `BITNET_STRICT_TOKENIZERS` (already guarded)

**Low Priority** (inconsistent coverage):
- `RUST_LOG` (performance/logging tests)
- `BITNET_GGUF` (model loading tests)
- `RAYON_NUM_THREADS` (parallelism tests)

---

## Recommended Path Forward

### Phase 1: Create Centralized Test Support (Week 1)

**Location**: `crates/bitnet-common/src/test_support/env_guard.rs`

**Module Structure**:
```
bitnet-common/src/test_support/
├── mod.rs                # Exports helpers
├── env_guard.rs          # EnvGuard + with_env() helpers
└── config.rs             # Config test helpers
```

**Exports**:
```rust
pub use env_guard::{EnvGuard, with_env, with_env_vars};
```

**Helpers to Implement**:
```rust
/// Scoped environment variable (closure-based)
pub fn with_env<F>(key: &str, value: Option<&str>, f: F)
where F: FnOnce()

/// Multiple scoped variables
pub fn with_env_vars<I, K, V, F>(vars: I, f: F)
where I: IntoIterator<Item = (K, Option<V>)>, ...

/// RAII guard for #[serial] tests
pub struct EnvGuard { ... }
```

### Phase 2: Fix Critical Flaky Tests (Week 2)

**Migration**: `crates/bitnet-common/tests/issue_260_strict_mode_tests.rs`

**Pattern**:
```rust
// Before: no serialization
#[test]
#[ignore = "FLAKY..."]
fn test_strict_mode() {
    unsafe { env::set_var("BITNET_STRICT_MODE", "1"); }
    // ... test ...
    unsafe { env::remove_var("BITNET_STRICT_MODE"); }
}

// After: properly serialized
#[test]
#[serial(bitnet_env)]
fn test_strict_mode() {
    with_env("BITNET_STRICT_MODE", Some("1"), || {
        // ... test ...
    });
}
```

**Expected Outcome**: All 15+ tests pass 100% consistently

### Phase 3: Standardize Across Workspace (Week 3-4)

**Migrate**:
- bitnet-common/tests/config_tests.rs (remove manual Mutex)
- bitnet-inference/tests/* (RUST_LOG, determinism flags)
- bitnet-cli/tests/* (model paths)
- crossval/tests/* (BITNET_CPP_DIR)

**Validation**:
```bash
# Run with single-threaded to catch any remaining races
cargo test --workspace -- --test-threads=1

# Run with full parallelism to verify #[serial] works
cargo test --workspace
```

---

## Quick Start for Developers

### Use Case: Testing Strict Mode

```rust
#[test]
#[serial(bitnet_env)]
fn test_strict_mode_parsing() {
    // Test default (no var)
    with_env("BITNET_STRICT_MODE", None, || {
        let cfg = StrictModeConfig::from_env();
        assert!(!cfg.enabled);
    });

    // Test enabled
    with_env("BITNET_STRICT_MODE", Some("1"), || {
        let cfg = StrictModeConfig::from_env();
        assert!(cfg.enabled);
    });

    // Test disabled
    with_env("BITNET_STRICT_MODE", Some("0"), || {
        let cfg = StrictModeConfig::from_env();
        assert!(!cfg.enabled);
    });
}
```

### Use Case: Multiple Variables

```rust
#[test]
#[serial(bitnet_env)]
fn test_combined_config() {
    with_env_vars([
        ("BITNET_STRICT_MODE", Some("1")),
        ("BITNET_VOCAB_SIZE", Some("50000")),
    ], || {
        let strict = StrictModeConfig::from_env();
        let config = BitNetConfig::from_env().unwrap();
        
        assert!(strict.enabled);
        assert_eq!(config.model.vocab_size, 50000);
    });
}
```

---

## Dependencies Already Available

✅ `serial_test = "3.2.0"` - in workspace Cargo.toml  
✅ `temp_env` - need to verify version (likely in workspace)  
✅ `once_cell` - used by existing EnvVarGuard  

**Verify**:
```bash
grep -E "serial_test|temp_env|once_cell" Cargo.toml
```

---

## File Created

📄 **`ci/exploration/env_testing_patterns.md`** (628 lines)

**Contents**:
- Executive summary of both approaches
- Complete RAII guard implementation details
- Modern scoped pattern documentation
- Detailed test inventory (65+ tests analyzed)
- Migration path with 4 phases
- Integration checklist
- Quick reference for developers

---

## Next Steps

1. **Review** this document and the full analysis
2. **Decide**: RAII vs. Scoped approach (or hybrid)
3. **Create** test support module with chosen approach
4. **Migrate** issue #441 tests to verify approach works
5. **Standardize** across workspace
6. **Close** issue #441 when all tests pass reliably

---

## Impact Assessment

### What Gets Fixed
- All 15+ flaky strict mode tests
- Config test race conditions
- GPU detection test stability
- Tokenizer strict mode tests

### No Impact Areas
- Quantization kernels (unrelated to env vars)
- Model loading logic (separate from testing)
- Inference engine (separate from testing)

### Expected Benefits
- 100% deterministic test results
- Reduced CI flakiness
- Clear, standard pattern for future tests
- Better test isolation and reliability

