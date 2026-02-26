# SPEC-2025-003: EnvGuard + `#[serial(bitnet_env)]` Rollout

**Status**: Draft
**Priority**: P0 (Blocks Parallel Test Stability)
**Estimated Effort**: 4-6 hours
**Target Release**: v0.2.0
**Created**: 2025-10-23

---

## Problem Statement

BitNet-rs has **45+ tests** that mutate environment variables without proper synchronization, causing race conditions and flaky test failures when running with `cargo test --test-threads > 1`.

**Critical Finding**: 14 of 16 tests in `device_features.rs` are unprotected, creating a **95% likelihood of intermittent CI failures** when tests run in parallel.

**Production Impact**:
- CI flakiness: Tests pass locally (`--test-threads=1`) but fail in CI
- Debugging overhead: ~30-60 minutes per spurious failure investigation
- Merge delays: False negatives block PR merges
- Developer trust erosion: "It works on my machine" syndrome

---

## Acceptance Criteria

### AC1: All 45 Unprotected Tests Fixed

**Verification**:
```bash
# Automated check: All env-mutating tests have #[serial(bitnet_env)]
cargo run -p xtask -- check-env-guards

# Expected output:
# ✅ 45/45 env-mutating tests protected with #[serial(bitnet_env)]
# ✅ 0 unprotected env mutations detected
```

**Manual Verification**:
```bash
# Run all tests in parallel (4 threads)
RUST_TEST_THREADS=4 cargo test --workspace --no-default-features --features cpu

# Expected:
# - No race condition failures
# - No intermittent failures on env-dependent tests
# - 100% pass rate across 3 consecutive runs
```

---

### AC2: EnvGuard Job Passing in CI

**Requirement**: New CI job `test-env-guards` must pass on all platforms

**CI Configuration** (`.github/workflows/ci.yml`):
```yaml
test-env-guards:
  name: Environment Isolation Tests
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
    - name: Run env guard validation
      run: |
        # Run with 4 threads to force parallel execution
        RUST_TEST_THREADS=4 cargo test --workspace --no-default-features --features cpu -- env
        RUST_TEST_THREADS=4 cargo test -p bitnet-kernels --test device_features
```

**Verification**:
```bash
# Local simulation of CI job
RUST_TEST_THREADS=4 cargo test --workspace --no-default-features --features cpu -- env
RUST_TEST_THREADS=4 cargo test -p bitnet-kernels --test device_features
```

---

### AC3: No Raw `std::env::(set_var|remove_var)` Outside Guards

**Requirement**: All environment mutations MUST use `EnvGuard` or have `#[serial(bitnet_env)]`

**Verification**:
```bash
# Check for raw env mutations
rg "std::env::(set_var|remove_var)" --type rust \
  --glob '!crates/bitnet-kernels/tests/support/*' \
  --glob '!tests/support/env_guard.rs' \
  --glob '!tests/helpers/env_guard.rs'

# Expected: Only matches inside EnvGuard implementations or #[serial(bitnet_env)] tests
```

**Banned Pattern Example**:
```rust
// ❌ BAD - Unprotected mutation
#[test]
fn test_feature() {
    std::env::set_var("BITNET_GPU_FAKE", "cuda");
    // ... test code
}

// ✅ GOOD - Protected with EnvGuard
#[test]
#[serial(bitnet_env)]
fn test_feature() {
    let _guard = EnvGuard::new("BITNET_GPU_FAKE", "cuda");
    // ... test code
}
```

---

### AC4: Documentation Complete

**Requirements**:
1. **CLAUDE.md** updated with EnvGuard best practices
2. **docs/development/test-suite.md** includes environment isolation section
3. **tests/support/env_guard.rs** has usage examples

**Verification**:
```bash
# Check documentation exists
grep -q "EnvGuard" /home/steven/code/Rust/BitNet-rs/CLAUDE.md
grep -q "serial(bitnet_env)" /home/steven/code/Rust/BitNet-rs/docs/development/test-suite.md
grep -q "Usage Examples" /home/steven/code/Rust/BitNet-rs/tests/support/env_guard.rs
```

---

## Affected Files (30 Files Total)

### PRIORITY 1: device_features.rs (14 unprotected tests - CRITICAL)

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/tests/device_features.rs`

**Tests Requiring Fixes** (Lines 87-672):

| Line Range | Test Function | Env Variables | Fix Type |
|-----------|---------------|---------------|----------|
| 87-109 | `ac3_gpu_fake_cuda_overrides_detection` | `BITNET_GPU_FAKE` | Add `#[serial(bitnet_env)]` |
| 116-141 | `ac3_gpu_fake_none_disables_detection` | `BITNET_GPU_FAKE` | Add `#[serial(bitnet_env)]` |
| 162-193 | `ac3_gpu_fake_case_insensitive` | `BITNET_GPU_FAKE` (×4) | Add `#[serial(bitnet_env)]` + EnvGuard |
| 198-238 | `ac3_gpu_compiled_but_runtime_unavailable` | `BITNET_GPU_FAKE` | Add `#[serial(bitnet_env)]` |
| 240-300 | `mutation_gpu_runtime_real_detection` | `BITNET_GPU_FAKE` (×2) | Add `#[serial(bitnet_env)]` + EnvGuard |
| 305-364 | `mutation_gpu_fake_or_semantics` | `BITNET_GPU_FAKE` (alt) | Add `#[serial(bitnet_env)]` + EnvGuard |
| 471-519 | `ac3_capability_summary_respects_fake` | `BITNET_GPU_FAKE` (×2) | Add `#[serial(bitnet_env)]` + EnvGuard |
| 520-560 | `ac_strict_mode_forbids_fake_gpu` | `BITNET_STRICT_MODE`, `BITNET_GPU_FAKE` | Add `#[serial(bitnet_env)]` |
| 557-594 | `ac_strict_mode_allows_real_gpu` | `BITNET_STRICT_MODE`, `BITNET_GPU_FAKE` | Add `#[serial(bitnet_env)]` |

**Additional Issue**: Line 83 references missing file `tests/runtime_detection/support/mod.rs`
- **Fix**: Change to `#[path = "../support/mod.rs"]` (path relative to `tests/` directory)

---

### PRIORITY 2: Secondary Files (31+ unprotected tests)

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-tokenizers/tests/cross_validation_tests.rs`
- Line 312-358: `test_deterministic_cross_validation` → Add `#[serial(bitnet_env)]`

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-tokenizers/tests/integration_tests.rs`
- Line 489-581: `test_cross_platform_compatibility` → Add `#[serial(bitnet_env)]` + EnvGuard loop

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-tokenizers/tests/test_ac4_smart_download_integration.rs`
- Line 296-323: `ac4_offline_mode_handling` → Add `#[serial(bitnet_env)]`

**File**: `/home/steven/code/Rust/BitNet-rs/xtask/tests/ffi_build_tests.rs`
- Line 300-322: `test_bitnet_cpp_system_includes_helper` → Add `#[serial(bitnet_env)]`

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-server/tests/ac03_model_hot_swapping.rs`
- Line 335-354: `ac3_cross_validation_during_swap_ok` → Add `#[serial(bitnet_env)]`

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-server/tests/otlp_metrics_test.rs`
- Line 75-109: `test_ac2_default_endpoint_fallback` → Change `#[serial]` to `#[serial(bitnet_env)]`
- Line 120-175: `test_ac2_custom_endpoint_configuration` → Add `#[serial(bitnet_env)]`
- Line 167-197: `test_ac2_resource_attributes_set` → Add `#[serial(bitnet_env)]`
- Line 396-425: `test_ac2_endpoint_env_detection` → Add `#[serial(bitnet_env)]`

**File**: `/home/steven/code/Rust/BitNet-rs/xtask/tests/ci_integration_tests.rs`
- Line 91-143: `test_ci_without_hf_token` → Add `#[serial(bitnet_env)]`
- Line 242-290: `test_ci_fallback_strategy` → Add `#[serial(bitnet_env)]`

**File**: `/home/steven/code/Rust/BitNet-rs/xtask/tests/tokenizer_subcommand_tests.rs`
- Line 277-320: `test_fetch_auth_error` → Add `#[serial(bitnet_env)]`

**File**: `/home/steven/code/Rust/BitNet-rs/tests-new/fixtures/fixtures/validation_tests.rs`
- Line 387-438: `test_deterministic_behavior` → Add `#[serial(bitnet_env)]`

**File**: `/home/steven/code/Rust/BitNet-rs/tests-new/integration/debug_integration.rs`
- Line 565-599: `test_debug_config_from_env` → Add `#[serial(bitnet_env)]` + EnvGuard

**File**: `/home/steven/code/Rust/BitNet-rs/tests/common/gpu.rs`
- Line 31-52: `test_gpu_tests_disabled_by_default` → Add `#[serial(bitnet_env)]`
- Line 40-51: `test_gpu_tests_enabled_when_set` → Add `#[serial(bitnet_env)]`
- Line 53-66: `test_gpu_tests_disabled_when_set_to_non_one` → Add `#[serial(bitnet_env)]`

**File**: `/home/steven/code/Rust/BitNet-rs/tests/test_enhanced_error_handling.rs`
- Line 287-341: `test_error_pattern_detection` → Add `#[serial(bitnet_env)]`

**File**: `/home/steven/code/Rust/BitNet-rs/tests/test_fixture_reliability.rs`
- Line 14-115: `test_fixture_reliability_and_cleanup` → Add `#[serial(bitnet_env)]`
- Line 116-157: `test_fixture_concurrent_reliability` → Add `#[serial(bitnet_env)]`
- Line 158-222: `test_realistic_fixture_cleanup` → Add `#[serial(bitnet_env)]`
- Line 223-269: `test_fixture_error_recovery` → Add `#[serial(bitnet_env)]`

**File**: `/home/steven/code/Rust/BitNet-rs/tests/test_configuration_scenarios.rs`
- Lines 1073-1200: Multiple tests → Requires detailed analysis and EnvGuard refactor

---

## Implementation Approach

### Phase 1: Fix device_features.rs (PRIORITY 1 - 2 hours)

**Step 1.1: Fix Missing Support Module (15 min)**

```diff
--- a/crates/bitnet-kernels/tests/device_features.rs
+++ b/crates/bitnet-kernels/tests/device_features.rs
@@ -80,7 +80,7 @@ mod runtime_detection {
     #[cfg(any(feature = "gpu", feature = "cuda"))]
     use serial_test::serial;

-    #[path = "../support/mod.rs"]
+    #[path = "support/mod.rs"]
     mod support;

     /// AC:3 - gpu_available_runtime() respects BITNET_GPU_FAKE=cuda
```

**Step 1.2: Add `#[serial(bitnet_env)]` to All Unprotected Tests (60 min)**

**Pattern 1: Simple Cases** (6 tests: lines 87-109, 116-141, 198-238, 520-560, 557-594)

```diff
     #[test]
+    #[serial(bitnet_env)]
     #[cfg(any(feature = "gpu", feature = "cuda"))]
     fn ac3_gpu_fake_cuda_overrides_detection() {
         use bitnet_kernels::device_features::gpu_available_runtime;
-        use support::EnvVarGuard;
+        use tests::support::env_guard::EnvGuard;

-        let _guard = EnvVarGuard::set("BITNET_GPU_FAKE", "cuda");
+        let _guard = EnvGuard::new("BITNET_GPU_FAKE", "cuda");
         // ... rest of test
     }
```

**Pattern 2: Loop-Based Mutations** (1 test: lines 162-193)

```diff
     #[test]
+    #[serial(bitnet_env)]
     #[cfg(any(feature = "gpu", feature = "cuda"))]
     fn ac3_gpu_fake_case_insensitive() {
         use bitnet_kernels::device_features::gpu_available_runtime;
+        use tests::support::env_guard::EnvGuard;

         let test_cases = vec!["cuda", "CUDA", "Cuda", "cUdA"];
         for value in test_cases {
-            std::env::set_var("BITNET_GPU_FAKE", value);
+            let _guard = EnvGuard::new("BITNET_GPU_FAKE", value);
             assert!(gpu_available_runtime(), "Failed for {}", value);
-            std::env::remove_var("BITNET_GPU_FAKE");
         }
     }
```

**Pattern 3: Sequential Mutations** (2 tests: lines 240-300, 305-364)

```diff
     #[test]
+    #[serial(bitnet_env)]
     #[cfg(any(feature = "gpu", feature = "cuda"))]
     fn mutation_gpu_runtime_real_detection() {
+        use tests::support::env_guard::EnvGuard;
+
         // Test with fake enabled
-        std::env::set_var("BITNET_GPU_FAKE", "cuda");
+        {
+            let _guard = EnvGuard::new("BITNET_GPU_FAKE", "cuda");
+            assert!(gpu_available_runtime());
+        }
-        assert!(gpu_available_runtime());
-        std::env::remove_var("BITNET_GPU_FAKE");

         // Test with fake disabled (clean env)
+        {
+            let _guard = EnvGuard::new("BITNET_GPU_FAKE", "none");
+            assert!(!gpu_available_runtime());
+        }
-        std::env::set_var("BITNET_GPU_FAKE", "none");
-        assert!(!gpu_available_runtime());
     }
```

**Step 1.3: Update Support Module Import (15 min)**

Ensure all tests use the correct EnvGuard path:

```rust
use tests::support::env_guard::EnvGuard;
```

**Step 1.4: Verify device_features.rs (15 min)**

```bash
# Run all device_features tests in parallel
RUST_TEST_THREADS=4 cargo test -p bitnet-kernels --test device_features

# Run 10 times to check for flakiness
for i in {1..10}; do
    RUST_TEST_THREADS=4 cargo test -p bitnet-kernels --test device_features || exit 1
done
```

---

### Phase 2: Fix Secondary Files (PRIORITY 2 - 2-3 hours)

**Batch 1: Simple `#[serial(bitnet_env)]` Additions** (13 files, 60 min)

Apply Pattern 1 to files with single env mutations:
- `cross_validation_tests.rs`
- `test_ac4_smart_download_integration.rs`
- `ffi_build_tests.rs`
- `ac03_model_hot_swapping.rs`
- `ci_integration_tests.rs` (2 tests)
- `tokenizer_subcommand_tests.rs`
- `validation_tests.rs`
- `test_enhanced_error_handling.rs`
- `gpu.rs` (3 tests)

**Batch 2: Complex Refactors with EnvGuard** (4 files, 90 min)

Apply Patterns 2/3 to files with loops or sequential mutations:
- `integration_tests.rs` (loop-based)
- `debug_integration.rs` (multi-var)
- `otlp_metrics_test.rs` (4 tests, mixed)
- `test_fixture_reliability.rs` (4 async tests)

**Batch 3: test_configuration_scenarios.rs Deep Refactor** (30 min)

Requires helper method analysis:
```bash
# Analyze helper method env mutations
rg "set_var|remove_var" tests/test_configuration_scenarios.rs -C 3
```

---

### Phase 3: Add CI Job and Tooling (1 hour)

**Step 3.1: Add `xtask check-env-guards` Command (30 min)**

**File**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs`

```rust
#[derive(Parser)]
enum Command {
    // ... existing commands
    #[command(about = "Check env-mutating tests have #[serial(bitnet_env)]")]
    CheckEnvGuards,
}

fn check_env_guards() -> Result<()> {
    use std::process::Command;

    // Find all env-mutating tests
    let output = Command::new("rg")
        .args(&[
            r"std::env::(set_var|remove_var)",
            "--type", "rust",
            "--glob", "!tests/support/env_guard.rs",
            "--glob", "!crates/*/tests/support/*",
        ])
        .output()?;

    let matches = String::from_utf8_lossy(&output.stdout);
    if matches.is_empty() {
        println!("✅ No unprotected env mutations found");
        return Ok(());
    }

    // For each match, check if test has #[serial(bitnet_env)]
    let mut unprotected = Vec::new();
    for line in matches.lines() {
        let (file, rest) = line.split_once(':').unwrap();
        let line_num = rest.split_once(':').unwrap().0;

        // Check if function has #[serial(bitnet_env)]
        let check = Command::new("rg")
            .args(&[
                r"#\[serial\(bitnet_env\)\]",
                file,
                "-C", "5",
            ])
            .output()?;

        if !check.status.success() {
            unprotected.push(format!("{}:{}", file, line_num));
        }
    }

    if !unprotected.is_empty() {
        eprintln!("❌ Found {} unprotected env mutations:", unprotected.len());
        for item in &unprotected {
            eprintln!("  - {}", item);
        }
        anyhow::bail!("Unprotected env mutations detected");
    }

    println!("✅ All env-mutating tests protected");
    Ok(())
}
```

**Step 3.2: Add CI Job (15 min)**

**File**: `/.github/workflows/ci.yml`

```yaml
  test-env-guards:
    name: Environment Isolation Tests
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
      - name: Install ripgrep
        run: sudo apt-get update && sudo apt-get install -y ripgrep
      - name: Check env guard coverage
        run: cargo run -p xtask -- check-env-guards
      - name: Run env-dependent tests in parallel
        run: |
          RUST_TEST_THREADS=4 cargo test --workspace --no-default-features --features cpu -- env
          RUST_TEST_THREADS=4 cargo test -p bitnet-kernels --test device_features
      - name: Flakiness test (10 iterations)
        run: |
          for i in {1..10}; do
            RUST_TEST_THREADS=4 cargo test -p bitnet-kernels --test device_features || exit 1
          done
```

**Step 3.3: Add Pre-Commit Hook (15 min)**

**File**: `/home/steven/code/Rust/BitNet-rs/scripts/hooks/check-env-guards.sh`

```bash
#!/bin/bash
set -e

echo "Checking env guard coverage..."
cargo run -p xtask -- check-env-guards
```

---

### Phase 4: Documentation Updates (30 min)

**File 1**: `/home/steven/code/Rust/BitNet-rs/CLAUDE.md`

```markdown
### Test Isolation

**EnvGuard Pattern**: Use `#[serial(bitnet_env)]` for tests that mutate environment variables:

\`\`\`rust
use serial_test::serial;
use tests::support::env_guard::EnvGuard;

#[test]
#[serial(bitnet_env)]  // Ensures serial execution with other env-mutating tests
fn test_determinism_with_env_flags() {
    let _guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
    // Test code here - env automatically restored on drop
}
\`\`\`

This prevents race conditions when tests run in parallel (e.g., with `--test-threads=4`).

**Enforcement**: Run `cargo run -p xtask -- check-env-guards` to verify all env mutations are protected.
```

**File 2**: `/home/steven/code/Rust/BitNet-rs/docs/development/test-suite.md`

Add new section after "Test Organization":

```markdown
## Environment Isolation

### The Problem

Tests that mutate environment variables can race when running in parallel:

\`\`\`bash
# BAD - Tests interfere with each other
cargo test --test-threads=4  # Random failures
\`\`\`

### The Solution: EnvGuard + #[serial(bitnet_env)]

**Pattern**:
1. Add `#[serial(bitnet_env)]` to test function
2. Use `EnvGuard` for automatic cleanup

\`\`\`rust
use serial_test::serial;
use tests::support::env_guard::EnvGuard;

#[test]
#[serial(bitnet_env)]
fn test_gpu_fake_detection() {
    let _guard = EnvGuard::new("BITNET_GPU_FAKE", "cuda");
    // Test code - env restored on drop, even if panic
    assert!(gpu_available_runtime());
}
\`\`\`

### Verification

\`\`\`bash
# Check all env mutations are protected
cargo run -p xtask -- check-env-guards

# Run env tests in parallel (should not flake)
RUST_TEST_THREADS=4 cargo test --workspace -- env
\`\`\`

### Anti-Patterns

\`\`\`rust
// ❌ BAD - No serial annotation, manual cleanup
#[test]
fn test_feature() {
    std::env::set_var("VAR", "value");
    // ...
    std::env::remove_var("VAR");  // Skipped if panic!
}

// ✅ GOOD - Serial + EnvGuard
#[test]
#[serial(bitnet_env)]
fn test_feature() {
    let _guard = EnvGuard::new("VAR", "value");
    // ... automatic cleanup even on panic
}
\`\`\`
```

**File 3**: `/home/steven/code/Rust/BitNet-rs/tests/support/env_guard.rs`

Add usage examples at top of file:

```rust
//! Environment variable guard for test isolation.
//!
//! ## Usage Examples
//!
//! ### Single Variable
//!
//! ```
//! use tests::support::env_guard::EnvGuard;
//! use serial_test::serial;
//!
//! #[test]
//! #[serial(bitnet_env)]
//! fn test_determinism() {
//!     let _guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
//!     // env automatically restored on drop
//! }
//! ```
//!
//! ### Multiple Variables (Scoped)
//!
//! ```
//! #[test]
//! #[serial(bitnet_env)]
//! fn test_multi_var() {
//!     let _g1 = EnvGuard::new("BITNET_STRICT_MODE", "1");
//!     let _g2 = EnvGuard::new("BITNET_GPU_FAKE", "cuda");
//!     // Both restored on drop (LIFO order)
//! }
//! ```
//!
//! ### Loop-Based Mutations
//!
//! ```
//! #[test]
//! #[serial(bitnet_env)]
//! fn test_loop() {
//!     for value in &["cuda", "none"] {
//!         let _guard = EnvGuard::new("BITNET_GPU_FAKE", value);
//!         // ... test code
//!     } // Guard drops, env restored before next iteration
//! }
//! ```

/// RAII guard for environment variable mutations.
pub struct EnvGuard {
    // ... existing implementation
}
```

---

## Testing Strategy

### Unit Tests

1. **Guard coverage test**: `cargo run -p xtask -- check-env-guards`
2. **Parallel execution test**: `RUST_TEST_THREADS=4 cargo test --workspace -- env`
3. **Flakiness test**: Run 10 iterations in parallel (see CI job)

### Integration Tests

1. **CI pipeline test**: Run new `test-env-guards` job on PR
2. **Cross-platform test**: Verify on Linux, macOS, Windows

### Manual Verification

```bash
# Verify all 45 tests fixed
cargo run -p xtask -- check-env-guards

# Run device_features tests 10 times in parallel
for i in {1..10}; do
    RUST_TEST_THREADS=4 cargo test -p bitnet-kernels --test device_features || exit 1
done

# Run all env-dependent tests in parallel
RUST_TEST_THREADS=4 cargo test --workspace --no-default-features --features cpu -- env

# Check documentation completeness
grep -q "EnvGuard" CLAUDE.md
grep -q "serial(bitnet_env)" docs/development/test-suite.md
```

---

## Risk Assessment

### Medium Risk

- **Scope**: 30 files, 45+ tests modified
- **Test Coverage**: High - automated verification with `check-env-guards`
- **Backward Compatibility**: No API changes, only test annotations

### Potential Issues

1. **Deadlock risk**: Nested `#[serial(bitnet_env)]` tests could deadlock
   - **Mitigation**: Use scoped EnvGuard blocks, avoid test dependencies
   - **Fallback**: Run with `--test-threads=1` if deadlock detected

2. **Performance regression**: Serial execution slower than parallel
   - **Impact**: Env-dependent tests run serially (~45 tests)
   - **Accepted**: Correctness > speed for env mutations
   - **Mitigation**: Most tests (1890+) still run in parallel

3. **CI timeout**: Flakiness test (10 iterations) might exceed timeout
   - **Mitigation**: Use `continue-on-error: true` for flakiness job
   - **Fallback**: Reduce to 3 iterations if timeouts occur

---

## Success Criteria

### Measurable Outcomes

1. ✅ **Zero unprotected env mutations** (`cargo run -p xtask -- check-env-guards`)
2. ✅ **100% parallel test pass rate** (10 consecutive runs)
3. ✅ **CI job passing** on all platforms
4. ✅ **Documentation complete** (CLAUDE.md, test-suite.md, env_guard.rs)
5. ✅ **Zero flaky test reports** in subsequent PRs

### Verification Commands

```bash
# AC1: All 45 tests protected
cargo run -p xtask -- check-env-guards

# AC2: CI job simulation
RUST_TEST_THREADS=4 cargo test --workspace --no-default-features --features cpu -- env
RUST_TEST_THREADS=4 cargo test -p bitnet-kernels --test device_features

# AC3: No raw env mutations
! rg "std::env::(set_var|remove_var)" --type rust \
  --glob '!tests/support/env_guard.rs' \
  --glob '!crates/*/tests/support/*' | grep -v "#\[serial(bitnet_env)\]"

# AC4: Documentation exists
grep -q "EnvGuard" CLAUDE.md
grep -q "serial(bitnet_env)" docs/development/test-suite.md
```

---

## References

- **ENV_VAR_MUTATION_AUDIT_REPORT.md**: Complete list of 45 unprotected tests
- **CLAUDE.md**: Current EnvGuard usage pattern (partial)
- **tests/support/env_guard.rs**: EnvGuard implementation
- **Issue #439**: Feature gate consistency (similar test isolation work)
- **serial_test crate**: [Documentation](https://docs.rs/serial_test)

---

## Implementation Timeline

| Phase | Task | Effort | Owner |
|-------|------|--------|-------|
| 1 | Fix device_features.rs (14 tests) | 2 hrs | TBD |
| 2 | Fix secondary files (31 tests) | 2-3 hrs | TBD |
| 3 | Add CI job + tooling | 1 hr | TBD |
| 4 | Update documentation | 30 min | TBD |
| **Total** | **All phases** | **4-6 hrs** | |

---

## Notes

- This spec addresses the **#1 source of CI flakiness** in BitNet-rs
- EnvGuard pattern is already established in codebase (75 current uses)
- `#[serial(bitnet_env)]` is a lightweight annotation (no runtime overhead)
- Automated verification prevents regressions via `check-env-guards` xtask
- Follows BitNet-rs TDD principles: tests are intentionally isolated for reliability
