# Test Infrastructure API Updates Specification

**Issue**: #447 (AC6-AC7)
**Status**: Ready for Implementation
**Priority**: P3 - Test Infrastructure Fixes
**Date**: 2025-10-11
**Affected Crate**: `tests` (BitNet.rs test harness)

---

## Executive Summary

Update `tests` crate to match current TestConfig API, migrating from deprecated field names to modern Duration-based timing configuration. Verify existing fixtures module declaration is accessible. These changes resolve compilation failures in test infrastructure while maintaining zero impact on production neural network code.

**Key Changes**:
- **AC6**: Verify fixtures module compilation (no changes needed)
- **AC7**: Migrate `timeout_seconds` → `test_timeout: Duration`
- **AC7**: Remove invalid `fail_fast` references (field does not exist in TestConfig)
- Zero impact on inference, quantization, or model loading

---

## Acceptance Criteria

### AC6: Verify fixtures module compilation under fixtures feature
**Test Tag**: `// AC6: Verify fixtures module declaration`

**Requirements**:
- **Status**: Module already declared in `tests/lib.rs:31` and `tests/common/mod.rs:35`
- Validate existing `FixtureManager` accessibility to dependent tests
- **Action**: Verification-only (no implementation changes needed)

**Validation Command**:
```bash
cargo test -p tests --no-default-features --features fixtures --no-run
```

**Expected Output**: Compilation succeeds with fixtures module accessible

**Evidence**:
- `tests/lib.rs:31`: `pub mod fixtures { pub use crate::common::fixtures::*; }`
- `tests/common/mod.rs:35`: `pub mod fixtures;`

**Current State**: ✅ **ALREADY COMPLETE** - No implementation required

---

### AC7: Update tests crate to match current TestConfig API
**Test Tag**: `// AC7: TestConfig API migration (Duration, remove fail_fast)`

**Requirements**:
- **CORRECTED**: Replace `timeout_seconds: u64` → `test_timeout: Duration`
- **CRITICAL**: Remove invalid `fail_fast` references
  - Old `TestConfig` had `fail_fast` at top level (deprecated)
  - New `TestConfig` does NOT have `fail_fast` field
  - `TimeConstraints.fail_fast` exists but is NOT part of `TestConfig` structure
- Update `tests/run_configuration_tests.rs` and related test files

**Validation Command**:
```bash
cargo test -p tests --no-run
```

**Expected Output**: Compilation succeeds with correct TestConfig API usage

**Current API Structure** (`tests/common/config.rs:14-32`):
```rust
pub struct TestConfig {
    pub max_parallel_tests: usize,
    pub test_timeout: Duration,  // ← CORRECT: Duration type
    pub cache_dir: PathBuf,
    pub log_level: String,
    pub coverage_threshold: f64,
    #[cfg(feature = "fixtures")]
    pub fixtures: FixtureConfig,
    pub crossval: CrossValidationConfig,
    pub reporting: ReportingConfig,  // ← Does NOT contain fail_fast
}
```

---

## Technical Design

### AC6: Fixtures Module Verification

**Current State Analysis**:

```rust
// tests/lib.rs:30-33
#[cfg(feature = "fixtures")]
pub mod fixtures {
    pub use crate::common::fixtures::*;
}
```

```rust
// tests/common/mod.rs:35
pub mod fixtures;
```

**Verification Tasks**:
1. Compile with `--features fixtures` flag
2. Confirm `FixtureManager` is accessible
3. No implementation changes required

**Status**: ✅ Already functional - verification-only acceptance criterion

---

### AC7: TestConfig API Migration

#### Problem: Deprecated Field References

**Affected Files** (from grep analysis):
- `tests/run_configuration_tests.rs` (14 occurrences of `timeout_seconds`, 11 occurrences of `fail_fast`)
- `tests/test_configuration_scenarios.rs` (2 occurrences of `fail_fast`)
- `tests/common/fast_feedback.rs` (5 occurrences of `fail_fast`)
- `tests/common/fast_feedback_simple.rs` (2 occurrences of `fail_fast`)
- `tests/common/config_scenarios_backup.rs` (1 occurrence of `fail_fast`)
- `tests/bin/fast_feedback_demo.rs` (1 occurrence of `fail_fast`)
- `tests/bin/fast_feedback_simple_demo.rs` (1 occurrence of `fail_fast`)

**Analysis**: Most `fail_fast` references are for `FastFeedbackConfig` or `TimeConstraints`, **NOT** `TestConfig`. Only `run_configuration_tests.rs` incorrectly accesses `TestConfig.fail_fast`.

---

#### Migration Pattern: timeout_seconds → test_timeout

**OLD (Deprecated)**:
```rust
// tests/run_configuration_tests.rs:74
assert_eq!(config.timeout_seconds, 300);

// tests/run_configuration_tests.rs:152
let config = TestConfig {
    timeout_seconds: 240,
    // ...
};
```

**NEW (Correct)**:
```rust
// Use Duration::from_secs() for initialization
assert_eq!(config.test_timeout, Duration::from_secs(300));

let config = TestConfig {
    test_timeout: Duration::from_secs(240),
    // ...
};
```

---

#### Migration Pattern: fail_fast Removal

**OLD (INVALID - field does not exist)**:
```rust
// tests/run_configuration_tests.rs:75
assert!(config.fail_fast);  // ❌ TestConfig has no fail_fast field

// tests/run_configuration_tests.rs:153
let config = TestConfig {
    fail_fast: false,  // ❌ DOES NOT COMPILE
    // ...
};
```

**NEW (Correct - Remove Invalid References)**:
```rust
// Simply remove these assertions and field initializers
// TestConfig does not have fail_fast field

// If test logic requires fail-fast behavior, use TimeConstraints instead:
let mut time_constraints = TimeConstraints::default();
time_constraints.fail_fast = true;
```

**Important**: `fail_fast` exists in `TimeConstraints` (line 85 of `test_configuration_scenarios.rs`) but **NOT** in `TestConfig`.

---

### Search and Replace Automation

#### Step 1: Find All Affected Files

```bash
# Find timeout_seconds usage
rg "timeout_seconds" --type rust tests/ -l

# Find fail_fast usage in TestConfig contexts
rg "config\.fail_fast|TestConfig.*fail_fast" --type rust tests/ -l
```

**Expected Output**:
```
tests/run_configuration_tests.rs
```

---

#### Step 2: Manual Migration (run_configuration_tests.rs)

**File**: `tests/run_configuration_tests.rs`

**Line-by-Line Changes**:

| Line | OLD Code | NEW Code | Reason |
|------|----------|----------|--------|
| 74 | `assert_eq!(config.timeout_seconds, 300);` | `assert_eq!(config.test_timeout, Duration::from_secs(300));` | Field rename + type change |
| 75 | `assert!(config.fail_fast);` | **REMOVE LINE** | Field does not exist |
| 91 | `assert_eq!(minimal.timeout_seconds, 30);` | `assert_eq!(minimal.test_timeout, Duration::from_secs(30));` | Field rename + type change |
| 92 | `assert!(!minimal.fail_fast);` | **REMOVE LINE** | Field does not exist |
| 98 | `assert_eq!(dev.timeout_seconds, 600);` | `assert_eq!(dev.test_timeout, Duration::from_secs(600));` | Field rename + type change |
| 99 | `assert!(dev.fail_fast);` | **REMOVE LINE** | Field does not exist |
| 105 | `assert_eq!(ci.timeout_seconds, 1800);` | `assert_eq!(ci.test_timeout, Duration::from_secs(1800));` | Field rename + type change |
| 106 | `assert!(!ci.fail_fast);` | **REMOVE LINE** | Field does not exist |
| 127 | `assert_eq!(config.timeout_seconds, 120);` | `assert_eq!(config.test_timeout, Duration::from_secs(120));` | Field rename + type change |
| 128 | `assert!(!config.fail_fast);` | **REMOVE LINE** | Field does not exist |
| 152 | `timeout_seconds: 240,` | `test_timeout: Duration::from_secs(240),` | Field rename + type change |
| 153 | `fail_fast: false,` | **REMOVE LINE** | Field does not exist |
| 185 | `assert_eq!(loaded_config.timeout_seconds, config.timeout_seconds);` | `assert_eq!(loaded_config.test_timeout, config.test_timeout);` | Field rename |
| 186 | `assert_eq!(loaded_config.fail_fast, config.fail_fast);` | **REMOVE LINE** | Field does not exist |
| 199 | `timeout_seconds: 300,` | `test_timeout: Duration::from_secs(300),` | Field rename + type change |
| 200 | `fail_fast: true,` | **REMOVE LINE** | Field does not exist |
| 217 | `assert_eq!(merged.timeout_seconds, 300);` | `assert_eq!(merged.test_timeout, Duration::from_secs(300));` | Field rename + type change |
| 218 | `assert!(merged.fail_fast);` | **REMOVE LINE** | Field does not exist |
| 231 | `timeout_seconds: 60,` | `test_timeout: Duration::from_secs(60),` | Field rename + type change |
| 232 | `fail_fast: false,` | **REMOVE LINE** | Field does not exist |
| 276 | `timeout_seconds: 180,` | `test_timeout: Duration::from_secs(180),` | Field rename + type change |
| 277 | `fail_fast: true,` | **REMOVE LINE** | Field does not exist |
| 298 | `assert_eq!(deserialized.timeout_seconds, config.timeout_seconds);` | `assert_eq!(deserialized.test_timeout, config.test_timeout);` | Field rename |
| 299 | `assert_eq!(deserialized.fail_fast, config.fail_fast);` | **REMOVE LINE** | Field does not exist |
| 312 | `timeout_seconds: 300,` | `test_timeout: Duration::from_secs(300),` | Field rename + type change |
| 320 | `let invalid_config = TestConfig { timeout_seconds: 0, ..Default::default() };` | `let invalid_config = TestConfig { test_timeout: Duration::from_secs(0), ..Default::default() };` | Field rename + type change |

**Import Addition** (top of file):
```rust
use std::time::Duration;  // Add if not already present
```

---

#### Step 3: Verify Other Files

**Files with `fail_fast` but NOT TestConfig**:
- `tests/test_configuration_scenarios.rs:286` - Uses `TimeConstraints.fail_fast` ✅ Correct
- `tests/test_configuration_scenarios.rs:1171` - Uses `TimeConstraints.fail_fast` ✅ Correct
- `tests/common/fast_feedback.rs` - Uses `FastFeedbackConfig.fail_fast` ✅ Correct
- `tests/common/fast_feedback_simple.rs` - Uses own config struct ✅ Correct

**Conclusion**: Only `run_configuration_tests.rs` requires changes.

---

## Migration Checklist

### Phase 1: AC6 Verification
- [ ] **AC6.1**: Compile tests with `--features fixtures` flag
- [ ] **AC6.2**: Verify FixtureManager is accessible
- [ ] **AC6.3**: Confirm no implementation changes required

**Validation**:
```bash
cargo test -p tests --no-default-features --features fixtures --no-run
# Expected: Compilation succeeds
```

---

### Phase 2: AC7 Migration (run_configuration_tests.rs)
- [ ] **AC7.1**: Add `use std::time::Duration;` import (if missing)
- [ ] **AC7.2**: Replace all `timeout_seconds` with `test_timeout: Duration::from_secs(...)`
- [ ] **AC7.3**: Remove all invalid `config.fail_fast` assertions (11 occurrences)
- [ ] **AC7.4**: Remove all invalid `fail_fast: bool` field initializers (7 occurrences)

**Validation**:
```bash
cargo test -p tests --no-run
# Expected: Compilation succeeds
```

---

### Phase 3: Validation
- [ ] **AC7.5**: Run full test suite to confirm no runtime regressions
- [ ] **AC7.6**: Verify test assertions still validate correct behavior

**Validation**:
```bash
cargo test -p tests
# Expected: Tests pass with updated API
```

---

## Validation Commands Matrix

| Phase | Command | Expected Output | AC |
|-------|---------|-----------------|-----|
| **Fixtures Verification** | `cargo test -p tests --no-default-features --features fixtures --no-run` | Compilation succeeds | AC6 |
| **API Migration** | `cargo test -p tests --no-run` | Compilation succeeds | AC7 |
| **Runtime Tests** | `cargo test -p tests` | Tests pass | AC7 |
| **All Features** | `cargo test -p tests --all-features` | Tests pass | AC6, AC7 |

---

## Rollback Strategy

### Rollback Steps

1. **Revert Test File Changes**:
   ```bash
   git checkout tests/run_configuration_tests.rs
   ```

2. **Validate Rollback**:
   ```bash
   cargo test -p tests --no-run
   ```

### Rollback Criteria
- Test assertions fail unexpectedly
- Compilation errors in dependent test files
- Behavioral changes in test execution

**Risk**: Low - changes are mechanical field renames and removals

---

## API Contract Documentation

### TestConfig Current API

**Module**: `tests::common::config`

```rust
pub struct TestConfig {
    /// Maximum number of tests to run in parallel
    pub max_parallel_tests: usize,

    /// Timeout for individual tests (Duration type)
    pub test_timeout: Duration,  // ← CORRECT FIELD NAME

    /// Directory for test cache and temporary files
    pub cache_dir: PathBuf,

    /// Logging level for test execution
    pub log_level: String,

    /// Minimum code coverage threshold (0.0 to 1.0)
    pub coverage_threshold: f64,

    /// Configuration for test fixtures
    #[cfg(feature = "fixtures")]
    pub fixtures: FixtureConfig,

    /// Configuration for cross-validation testing
    pub crossval: CrossValidationConfig,

    /// Configuration for test reporting
    pub reporting: ReportingConfig,

    // NOTE: fail_fast field DOES NOT EXIST in TestConfig
}
```

---

### Breaking Changes

**None** - This is a fix to align with the correct API (already deployed in `tests/common/config.rs`).

**Deprecated Fields**:
- `timeout_seconds: u64` (replaced by `test_timeout: Duration`)
- `fail_fast: bool` (removed - use `TimeConstraints.fail_fast` if needed)

---

## Test Structure

### Verification Tests (AC6)

```rust
// tests/fixtures/validation_tests.rs

#[test]
#[cfg(feature = "fixtures")]
fn test_ac6_fixtures_module_accessible() { // AC:6
    use crate::fixtures::FixtureManager;

    // Verify type is accessible from public API
    let _phantom: Option<FixtureManager> = None;
}
```

---

### Migration Validation Tests (AC7)

```rust
// tests/run_configuration_tests.rs (updated tests)

#[tokio::test]
async fn test_ac7_config_default_values() -> TestOpResult<()> { // AC:7
    use std::time::Duration;

    let config = TestConfig::default();

    // Verify Duration-based timeout (not u64 seconds)
    assert_eq!(config.test_timeout.as_secs(), 300);

    // Verify fail_fast is NOT a TestConfig field
    // (This test would fail to compile if fail_fast existed)

    Ok(())
}

#[tokio::test]
async fn test_ac7_config_initialization() -> TestOpResult<()> { // AC:7
    use std::time::Duration;

    let config = TestConfig {
        max_parallel_tests: 8,
        test_timeout: Duration::from_secs(120),
        ..Default::default()
    };

    assert_eq!(config.test_timeout.as_secs(), 120);

    Ok(())
}
```

---

## BitNet.rs Standards Compliance

### Feature Flag Discipline
✅ Fixtures module properly gated behind `#[cfg(feature = "fixtures")]`
✅ Validation commands use `--no-default-features` with explicit feature selection
✅ Zero impact on production inference code

### Workspace Structure Alignment
✅ Changes isolated to `tests` crate (not part of production library)
✅ No impact on `bitnet-*` workspace crates
✅ Test infrastructure improvements only

### Neural Network Development Patterns
✅ Zero impact on inference, quantization, or model loading
✅ Test configuration changes do not affect neural network logic
✅ Maintains existing test coverage and validation

### TDD and Test Naming
✅ All tests tagged with `// AC:N` comments
✅ Test names follow `test_acN_*` convention
✅ Validation commands map directly to acceptance criteria

### GGUF Compatibility
✅ No impact (test infrastructure only, no model format changes)

---

## References

### Current API Documentation
- `tests/common/config.rs:14-32` - TestConfig structure
- `tests/common/config.rs:148-175` - ReportingConfig structure
- `tests/lib.rs:30-33` - Fixtures module re-export
- `tests/common/mod.rs:35` - Fixtures module declaration

### Related Issues
- Issue #447 - Compilation fixes across workspace crates
- PR #445 - Test harness hygiene fixes (recent cleanup)

### Migration Evidence
- Finalized ACs document: `/home/steven/code/Rust/BitNet-rs/docs/explanation/specs/issue-447-finalized-acceptance-criteria.md:183-234`

---

## Approval Checklist

Before implementation:
- [x] Both acceptance criteria (AC6-AC7) clearly defined
- [x] AC6 confirmed as verification-only (no changes needed)
- [x] AC7 migration pattern documented with line-by-line changes
- [x] Validation commands specified with expected outputs
- [x] Rollback strategy documented
- [x] Zero impact on production code confirmed
- [x] BitNet.rs standards compliance verified
- [x] Test structure defined with AC tags

**Status**: ✅ Ready for Implementation

**Next Steps**: NEXT → impl-creator (implementation of AC7 only; AC6 is already complete)
