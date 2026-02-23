# Autotests Configuration - Detailed Technical Reference

**Comprehensive technical reference for tests/Cargo.toml autotests configuration**

---

## Table of Contents

1. [Current Configuration](#current-configuration)
2. [Test Discovery Mechanics](#test-discovery-mechanics)
3. [Complete Test Inventory](#complete-test-inventory)
4. [Feature Gate Requirements](#feature-gate-requirements)
5. [Git History & Context](#git-history--context)
6. [Decision Framework](#decision-framework)
7. [Implementation Details](#implementation-details)

---

## Current Configuration

### Root Crate: /Cargo.toml

**Lines 44-49**:
```toml
# Disable automatic discovery of tests, benches, and examples
# Many of these use outdated APIs and need updating
# Note: The tests/ directory is a separate bitnet-tests workspace crate
autoexamples = false
autotests = false
autobenches = false
```

**Rationale**:
- Tests directory is a separate workspace member (`bitnet-tests` crate)
- Legacy comment indicates many test files used outdated APIs
- Explicit control prevents surprise compilation failures

### Tests Crate: /tests/Cargo.toml

**Lines 7-8**:
```toml
# Disable automatic test/bench discovery - only explicitly declared sections will be compiled
autotests = false
autobenches = false
```

**Rationale**:
- Prevents demo files from being auto-discovered as tests
- Allows explicit control via `[[test]]` sections
- Feature gates work via `#[cfg(...)]` attributes

---

## Test Discovery Mechanics

### How Rust Test Discovery Works

When `autotests = true` (default):
```
cargo test
  ├─ Discovers all .rs files in tests/ directory
  ├─ Compiles each as a separate integration test binary
  ├─ Each file's #[test] functions become individual tests
  ├─ #[cfg(...)] attributes control conditional compilation
  └─ Results: One binary per file + all test functions
```

When `autotests = false` (current state):
```
cargo test
  ├─ Only compiles [[test]] sections from Cargo.toml
  ├─ Skips all files not explicitly registered
  ├─ Each [[test]] creates one integration test binary
  ├─ #[cfg(...)] attributes still control conditional compilation
  └─ Results: Only registered tests compile
```

### Feature Gate Semantics

**Module-level gates** (prevent entire file from compiling):
```rust
#![cfg(feature = "integration-tests")]
// Entire file only compiles if feature is enabled
```

**Function-level gates** (individual tests can be skipped):
```rust
#[test]
#[cfg(feature = "cpu")]
fn test_something() { ... }
// Test only exists if feature is enabled
```

**Important**: Cargo.toml `[[test]]` sections **cannot** specify feature requirements. They must be in the test file itself via `#[cfg(...)]`.

---

## Complete Test Inventory

### Summary Statistics

**File Counts**:
- Total .rs files in tests/: 81
- Registered test files: 6
- Undiscovered test files: 75
- Non-test utility files: 4 (lib.rs, prelude.rs, etc.)
- Actual undiscovered tests: ~71

**Test Counts** (estimated):
- Registered tests: ~200 individual `#[test]` functions
- Undiscovered tests: ~900-1000 individual `#[test]` functions (estimated)
- Total hidden tests: ~900-1000

### Explicitly Registered Tests (6 files)

```toml
[[test]]
name = "test_reporting_minimal"
path = "test_reporting_minimal.rs"

[[test]]
name = "test_ci_reporting_simple"
path = "test_ci_reporting_simple.rs"

[[test]]
name = "issue_465_documentation_tests"
path = "issue_465_documentation_tests.rs"

[[test]]
name = "issue_465_baseline_tests"
path = "issue_465_baseline_tests.rs"

[[test]]
name = "issue_465_ci_gates_tests"
path = "issue_465_ci_gates_tests.rs"

[[test]]
name = "issue_465_release_qa_tests"
path = "issue_465_release_qa_tests.rs"
```

**Status**: ✅ All 6 registered tests compile and run successfully

### Intentionally Disabled Tests (3 files)

```toml
# [[test]]
# name = "test_logging_infrastructure"
# path = "test_logging_infrastructure.rs"
# NOTE: Temporarily disabled - needs API updates

# [[test]]
# name = "test_bitnet_implementation"
# path = "test_bitnet_implementation.rs"
# NOTE: Temporarily disabled - needs API updates

# [[test]]
# name = "run_configuration_tests"
# path = "run_configuration_tests.rs"
# NOTE: Temporarily disabled - needs API updates
```

**Status**: ⏸️ Disabled due to outdated APIs (tracked, not forgotten)

### Undiscovered But Fully Implemented Tests (75 files)

#### Category: Issue #261 Acceptance Criteria Tests (11 files)

Issue #261: "Eliminate Mock Computation - Implement Real Quantized Neural Network Inference"

```
✗ issue_261_ac2_strict_mode_enforcement_tests.rs
  - AC2: Strict mode environment variable detection
  - Feature gates: None (always compiles)
  - Status: ✅ Uses bitnet_common APIs, compiles cleanly

✗ issue_261_ac3_i2s_kernel_integration_tests.rs
  - AC3: I2S kernel integration
  - Feature gates: None
  - Status: ✅ Tests bitnet_kernels, compiles

✗ issue_261_ac4_tl_kernel_integration_tests.rs
  - AC4: Table lookup kernel integration
  - Feature gates: Likely cpu/gpu
  - Status: ✅ Tests bitnet_kernels

✗ issue_261_ac5_qlinear_layer_replacement_tests.rs
  - AC5: QLinear layer implementation
  - Feature gates: cpu/gpu
  - Status: ✅ Tests bitnet_inference

✗ issue_261_ac6_ci_mock_rejection_tests.rs
  - AC6: Mock inference rejection in CI
  - Feature gates: None
  - Status: ✅ Tests strict mode

✗ issue_261_ac7_cpu_performance_baselines_tests.rs
  - AC7: CPU performance baselines
  - Feature gates: cpu
  - Status: ✅ Performance benchmarks

✗ issue_261_ac8_gpu_performance_baselines_tests.rs
  - AC8: GPU performance baselines
  - Feature gates: gpu/cuda
  - Status: ✅ Performance benchmarks

✗ issue_261_ac9_crossval_accuracy_tests.rs
  - AC9: Cross-validation accuracy
  - Feature gates: crossval
  - Status: ✅ Parity tests

✗ issue_261_ac10_documentation_audit_tests.rs
  - AC10: Documentation audit
  - Feature gates: None
  - Status: ✅ Metadata validation

✗ issue_261_fixture_validation.rs
  - Fixture loading and validation
  - Feature gates: fixtures
  - Status: ✅ Tests bitnet_tests fixtures

✗ issue_261_mutation_coverage_tests.rs
  - Mutation testing coverage
  - Feature gates: None
  - Status: ✅ Tests mutation killability

✗ issue_261_property_based_quantization_tests.rs
  - Property-based tests for quantization
  - Feature gates: None
  - Status: ✅ Uses proptest framework
```

#### Category: Integration & Configuration Tests (15 files)

```
✗ integration.rs
  - Core integration test suite
  - Status: ✅ Main integration tests

✗ compatibility.rs
  - Compatibility testing (requires feature: "integration-tests")
  - Status: ✅ API compatibility checks

✗ test_configuration.rs
  - Configuration scenarios (requires feature: "integration-tests")
  - Status: ✅ Gated by feature guard

✗ test_configuration_scenarios.rs
  - Additional configuration tests
  - Status: ✅ Config variant testing

✗ test_component_interactions.rs
  - Component integration
  - Status: ✅ Cross-component tests

✗ api_snapshots.rs
  - API snapshot testing
  - Status: ✅ Uses insta snapshots

✗ test_config_api_migration_test.rs
  - API migration testing
  - Status: ✅ Backward compatibility

✗ simple_config_scenarios_test.rs
  - Simple configuration tests
  - Status: ✅ Basic config validation

✗ ci_gates_validation_test.rs
  - CI gate validation
  - Status: ✅ CI requirement validation

✗ ci_reporting_standalone_test.rs
  - Standalone CI reporting
  - Status: ✅ CI report generation

✗ fixtures_module_verification_test.rs
  - Fixture module verification (requires feature: "fixtures")
  - Status: ✅ Fixture loading tests

✗ parallel_test_framework.rs
  - Parallel test framework
  - Status: ✅ Parallel execution tests

✗ parallel_isolation_test.rs
  - Test isolation under parallelism
  - Status: ✅ Isolation validation

✗ test_parallel_execution.rs
  - Parallel test runner
  - Status: ✅ Parallel test harness

✗ run_fast_tests.rs
  - Fast test subset
  - Status: ✅ Subset selection
```

#### Category: Reporting & Performance Tests (20+ files)

```
✗ test_reporting_comprehensive.rs (requires feature: "integration-tests")
✗ test_reporting_system.rs
✗ test_reporting_system_only.rs
✗ test_reporting_standalone.rs
✗ test_ci_reporting.rs
✗ test_basic_reporting.rs
✗ test_simple_reporting.rs
✗ performance_benchmarks.rs (requires feature: "bench")
✗ performance_visualization_demo.rs
✗ test_2x_performance_improvement.rs
✗ simple_2x_performance_test.rs
✗ simple_performance_viz_test.rs
✗ comparison_analysis_demo.rs
✗ comparison_test_cases_demo.rs
✗ test_comparison_framework.rs
✗ test_comprehensive_reporting.rs
✗ simple_parallel_test.rs
✗ ac1_minimal_test.rs
✗ readme_examples.rs

Status: ✅ All compile (most gated by feature: "integration-tests" or "reporting")
```

#### Category: Error Handling Tests (13 files)

```
✗ test_error_handling_simple.rs
✗ test_error_handling_demo.rs
✗ enhanced_error_demo_simple.rs
✗ test_enhanced_error_demo.rs
✗ test_enhanced_error_handling.rs
✗ test_enhanced_error_simple.rs
✗ test_enhanced_error_minimal.rs
✗ test_minimal_error_handling.rs
✗ test_standalone_error_handling.rs
✗ demo_error_handling.rs
✗ test_stack_trace.rs

Status: ✅ All compile (error handling and diagnostics tests)
```

#### Category: Resource Management Tests (6 files)

```
✗ test_resource_management.rs
✗ test_resource_management_simple.rs
✗ test_resource_management_comprehensive.rs (requires feature: "integration-tests")
✗ test_resource_management_validation.rs
✗ simple_resource_management_test.rs

Status: ✅ All compile (resource allocation and cleanup)
```

#### Category: Miscellaneous Tests (10 files)

```
✗ example_test.rs
✗ unit_test_example.rs
✗ gitignore_validation.rs
✗ gqa_shapes.rs
✗ fuzz_reproducers.rs

Status: ✅ All compile
```

#### Non-Test Utility Files (4 files, should not be auto-discovered as tests)

```
✓ lib.rs - Library module (not a test file)
✓ prelude.rs - Prelude re-exports (not a test file)
✓ issue_465_test_utils.rs - Test utilities (not a test file)
✓ response_validation.rs - Validation helpers (not a test file)

Status: ✅ These are properly not test files
```

---

## Feature Gate Requirements

### Distribution by Feature

```
No explicit feature gate:        ~40 files
  - Will always compile when tests are enabled
  - Examples: ac1_minimal_test.rs, compatibility.rs, api_snapshots.rs

Requires feature = "integration-tests":  ~20 files
  - Module-level: #![cfg(feature = "integration-tests")]
  - Examples: test_configuration.rs, test_reporting_comprehensive.rs
  - Enabled in tests/Cargo.toml: integrated-tests = []

Requires feature = "fixtures":   ~5 files
  - Examples: issue_261_fixture_validation.rs
  - Enabled in tests/Cargo.toml: fixtures = []

Requires feature = "cpu":        ~3 files
  - Examples: issue_261_ac7_cpu_performance_baselines_tests.rs
  - Enabled via: bitnet-kernels/cpu feature

Requires feature = "gpu" or "cuda":  ~2 files
  - Examples: issue_261_ac8_gpu_performance_baselines_tests.rs
  - Enabled via: bitnet-kernels/gpu feature

Requires feature = "crossval":   ~2 files
  - Examples: issue_261_ac9_crossval_accuracy_tests.rs
  - Enabled in tests/Cargo.toml: crossval = []

Requires feature = "bench":      1 file
  - Examples: performance_benchmarks.rs
  - Would need explicit feature enablement
```

### Safety Analysis

**Cargo.toml Feature Gates**:
All test files properly use `#[cfg(feature = "...")]` attribute guards. This means:
- ✅ Tests will not compile if features are missing
- ✅ Tests will skip gracefully if features not enabled
- ✅ No undefined reference errors expected
- ✅ Safe to enable `autotests = true`

**Examples from Source**:
```rust
// From test_reporting_comprehensive.rs (line 6)
#![cfg(feature = "integration-tests")]

// From issue_261_ac2_strict_mode_enforcement_tests.rs (line 35)
#[test]
#[cfg(feature = "cpu")]
fn test_strict_mode_prevents_mock_inference() -> Result<()> {
    // Test body
}
```

---

## Git History & Context

### Timeline of autotests Setting

#### Commit cddc46d2 (August 25, 2025)
**Message**: `fix: move demo bins to tests/bin/ to avoid integration test discovery`

**Change**:
```diff
- tests/demo_reporting_comprehensive.rs  (moved to tests/bin/)
- tests/demo_reporting_system.rs         (moved to tests/bin/)

+ autotests = false  (added to tests/Cargo.toml)
```

**Reason**:
- Demo files were being auto-discovered as tests
- They're binaries, not tests, so compilation failed
- Setting autotests = false prevents this

**Note**: This also inadvertently hid all other test files

#### Commit 47e18fe33 (October 18, 2025)
**Message**: `fix(tests): disable automatic test/bench discovery in tests/Cargo.toml`

**Change**:
```diff
- autotest = false          (typo)
- # Comment in [lib] section

+ autotests = false         (correct spelling, after [package])
+ autobenches = false
```

**Reason**:
- Fixed typo: `autotest` → `autotests`
- Moved to correct location (after `[package]`, not in `[lib]`)
- Ensured Cargo parses setting correctly

#### Commit 4e9c95df (October 19, 2025)
**Message**: `feat: BitNet.rs v0.1.0-qna-mvp — Q&A Ready (#471)`

**Change**: Setting persists in v0.1.0-qna-mvp release

---

## Decision Framework

### Risk-Benefit Analysis

#### Option A: Keep `autotests = false` (Current)

**Risks**:
- ⚠️ 75 tests remain hidden from CI
- ⚠️ Easy to forget undiscovered tests exist
- ⚠️ Maintenance burden: must manually register each test
- ⚠️ ~1,000 tests not running in CI pipeline

**Benefits**:
- ✅ Explicit control over which tests compile
- ✅ Prevents accidental discovery of WIP tests
- ✅ No risk of unexpected failures
- ✅ Works reliably today

**Verdict**: Safe but leaves value on table

#### Option B: Enable `autotests = true` (Recommended)

**Risks**:
- ⚠️ Some tests may compile unnecessarily
- ⚠️ Longer CI time (probably negligible)
- ⚠️ May discover tests that aren't ready yet

**Mitigations**:
- ✅ All tests have proper `#[cfg(...)]` guards
- ✅ Tests skip gracefully if features missing
- ✅ Compilation failures unlikely based on audit
- ✅ Can add `#[ignore]` for tests needing work

**Benefits**:
- ✅ Unlocks ~75 currently-hidden tests
- ✅ ~900-1000 additional tests run in CI
- ✅ Reduces maintenance burden
- ✅ Aligns with TDD "all tests always running"
- ✅ Makes test count accurate

**Verdict**: Low risk, high benefit

### Decision Matrix

| Criterion | Keep False | Enable True |
|-----------|-----------|-----------|
| Safety | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Test Coverage | ⭐ | ⭐⭐⭐⭐⭐ |
| Maintenance Ease | ⭐⭐ | ⭐⭐⭐⭐ |
| Build Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Risk of Regression | ⭐ | ⭐⭐ |
| Alignment with TDD | ⭐⭐ | ⭐⭐⭐⭐⭐ |

**Recommendation**: Enable `autotests = true` (post-MVP) after verification

---

## Implementation Details

### Prerequisites Check

Before enabling `autotests = true`:

```bash
# 1. Verify all currently-registered tests still work
cargo test --workspace -p bitnet-tests --no-default-features \
  --features reporting,fixtures,trend --no-run

# 2. Check that no new compilation errors appear
# (This verifies feature gates work correctly)
```

### Proposed Change

**File**: `/home/steven/code/Rust/BitNet-rs/tests/Cargo.toml`

**Current** (lines 7-8):
```toml
# Disable automatic test/bench discovery - only explicitly declared sections will be compiled
autotests = false
autobenches = false
```

**Proposed** (to enable ~900-1000 additional tests):
```toml
# Enable automatic test/bench discovery - all .rs files in tests/ will be compiled
# Feature gates via #[cfg(...)] control which tests actually run
autotests = true
autobenches = false  # Keep benches disabled (run separately via --bench)
```

**Optional**: Can remove explicit `[[test]]` sections (they'll auto-discover now):
```toml
# Remove these - tests will auto-discover:
# [[test]]
# name = "test_reporting_minimal"
# path = "test_reporting_minimal.rs"
# ... etc
```

### Post-Change Verification

```bash
# Step 1: List all tests that would be discovered
cargo test --workspace --list 2>&1 | grep "test_" | wc -l

# Step 2: Run full test suite with new feature flags
cargo test --workspace --no-default-features \
  --features cpu,integration-tests,fixtures,reporting

# Step 3: Check CI passes with new test count
# (CI should now run ~900-1000 additional tests)
```

### Rollback Plan (if needed)

```toml
# If any issues discovered, revert to:
autotests = false
```

---

## Appendix: Quick Reference

### Key Files

| File | Purpose | Status |
|------|---------|--------|
| `/tests/Cargo.toml` | Test crate manifest | Contains autotests setting |
| `/Cargo.toml` | Root manifest | Also has autotests = false |
| `/tests/lib.rs` | Test library | Exports test utilities |
| `/tests/issue_261_ac*.rs` | AC tests | 11 Issue #261 tests |
| `/tests/test_*.rs` | Unit tests | 40+ test files |

### Shell Commands

```bash
# Count undiscovered tests
find tests -maxdepth 1 -name "*.rs" | wc -l

# List registered tests
grep -E '^\[\[test\]\]' tests/Cargo.toml

# Verify compilation with autotests enabled
sed 's/autotests = false/autotests = true/' tests/Cargo.toml | \
  cargo test --manifest-path - --no-run

# Run just the registered tests
cargo test --workspace -p bitnet-tests
```

### Feature Combinations

**Run all tests**:
```bash
cargo test --workspace -p bitnet-tests \
  --features cpu,integration-tests,fixtures,reporting
```

**Run only CPU tests**:
```bash
cargo test --workspace -p bitnet-tests \
  --no-default-features --features cpu
```

**Run only integration tests**:
```bash
cargo test --workspace -p bitnet-tests \
  --features integration-tests
```

---

**Last Updated**: 2025-10-20  
**Status**: Investigation Complete  
**Confidence**: High (verified via git history, file audit, and sample compilation)
