# GitHub Issues for P0 Tasks - Ready for Submission

**Generated**: 2025-10-23
**Related Branch**: `feat/comprehensive-integration-qk256-envguard-receipts-strict-avx2`
**Related PR**: #475

---

## Issue #1: Harden build.rs: Remove unwraps, Add cargo:warning Fallbacks

**Labels**: `P0`, `build-system`, `hygiene`, `good-first-issue`
**Milestone**: v0.2.0
**Estimated Effort**: 2-3 hours

### Problem Statement

BitNet.rs build scripts (`build.rs`) contain unhygienic patterns that cause silent failures in minimal container environments:

1. **Missing `cargo:warning=` directives**: `bitnet-ggml-ffi/build.rs` uses `eprintln!()` instead of `println!("cargo:warning=...")`, making warnings invisible
2. **Silent CI failures**: When `VENDORED_GGML_COMMIT` is missing in CI, builds succeed locally but fail silently in Docker containers
3. **Inconsistent error handling**: `bitnet-kernels/build.rs` uses proper `cargo:warning=` patterns, but `bitnet-ggml-ffi/build.rs` does not

**Production Impact**: Builds succeed locally but fail silently in Docker containers or CI environments without clear diagnostics.

### Acceptance Criteria

- [ ] **AC1**: No `unwrap()` calls in build scripts
  ```bash
  ! grep -n "unwrap()" crates/bitnet-kernels/build.rs crates/bitnet-ggml-ffi/build.rs
  ```

- [ ] **AC2**: `cargo:warning=` directives visible during build
  ```bash
  rm crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT
  cargo build -p bitnet-ggml-ffi 2>&1 | grep "cargo:warning=Failed to read VENDORED_GGML_COMMIT"
  git checkout crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT
  ```

- [ ] **AC3**: Builds succeed without `$HOME` environment variable
  ```bash
  env -u HOME cargo build --no-default-features --features cpu -p bitnet-kernels
  ```

- [ ] **AC4**: CI panics on missing critical markers
  ```bash
  # Should panic with actionable message
  ! CI=1 cargo build -p bitnet-ggml-ffi 2>&1 | grep "VENDORED_GGML_COMMIT is 'unknown' in CI"
  ```

### Affected Files

**Primary** (require modification):
1. `/home/steven/code/Rust/BitNet-rs/crates/bitnet-ggml-ffi/build.rs` (lines 8-18)
   - Replace `eprintln!()` with `println!("cargo:warning=...")`

2. `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/build.rs` (lines 11-21)
   - No changes needed (reference example)

**Secondary** (documentation):
3. `/home/steven/code/Rust/BitNet-rs/docs/development/build-commands.md`
   - Add "Build Script Hygiene" section

4. `/home/steven/code/Rust/BitNet-rs/CLAUDE.md`
   - Update troubleshooting guide

### Implementation Steps

**Step 1**: Fix `eprintln!()` → `println!()` in `bitnet-ggml-ffi/build.rs`

```diff
--- a/crates/bitnet-ggml-ffi/build.rs
+++ b/crates/bitnet-ggml-ffi/build.rs
@@ -7,13 +7,13 @@
         let marker = Path::new("csrc/VENDORED_GGML_COMMIT");
         let commit = fs::read_to_string(marker).unwrap_or_else(|e| {
-            eprintln!(
+            println!(
                 "cargo:warning=Failed to read VENDORED_GGML_COMMIT marker at '{}': {}",
                 marker.display(),
                 e
             );
-            eprintln!(
-                "cargo:warning=Using 'unknown' as fallback. Run 'cargo xtask vendor-ggml' to fix."
+            println!(
+                "cargo:warning=Using 'unknown' as fallback. Run 'cargo xtask vendor-ggml' to fix."
             );
             "unknown".into()
         });
```

**Step 2**: Add verification tests in `xtask/tests/build_script_hygiene_tests.rs`

**Step 3**: Update documentation

### Testing Verification

```bash
# Test 1: No unwrap() calls
! grep -n "unwrap()" crates/*/build.rs

# Test 2: Build without HOME
env -u HOME cargo build --no-default-features --features cpu -p bitnet-kernels

# Test 3: Warning visibility (local mode)
mv crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT{,.bak}
cargo build -p bitnet-ggml-ffi 2>&1 | grep "cargo:warning="
mv crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT{.bak,}

# Test 4: CI panic (should fail)
! CI=1 cargo build -p bitnet-ggml-ffi
```

### References

- **Specification**: `/home/steven/code/Rust/BitNet-rs/docs/explanation/specs/SPEC-2025-002-build-script-hygiene-hardening.md`
- **Cargo Book**: [Build Scripts - Warning Directives](https://doc.rust-lang.org/cargo/reference/build-scripts.html#cargo-warning)
- **Related**: Issue #439 (Feature gate consistency)

---

## Issue #2: Roll Out EnvGuard + #[serial(bitnet_env)] Across Env Tests

**Labels**: `P0`, `test-infrastructure`, `flakiness`, `parallel-safety`
**Milestone**: v0.2.0
**Estimated Effort**: 4-6 hours

### Problem Statement

BitNet.rs has **45+ tests** that mutate environment variables without proper synchronization, causing race conditions when running with `cargo test --test-threads > 1`.

**Critical Finding**: 14 of 16 tests in `device_features.rs` are unprotected, creating a **95% likelihood of intermittent CI failures** in parallel execution.

**Impact**:
- CI flakiness: Tests pass locally (`--test-threads=1`) but fail randomly in CI
- Debugging overhead: ~30-60 minutes per spurious failure investigation
- Merge delays: False negatives block PR merges

### Acceptance Criteria

- [ ] **AC1**: All 45 unprotected tests fixed
  ```bash
  cargo run -p xtask -- check-env-guards
  # Expected: ✅ 45/45 env-mutating tests protected
  ```

- [ ] **AC2**: EnvGuard CI job passing
  ```bash
  RUST_TEST_THREADS=4 cargo test --workspace --no-default-features --features cpu -- env
  RUST_TEST_THREADS=4 cargo test -p bitnet-kernels --test device_features
  ```

- [ ] **AC3**: No raw `std::env::(set_var|remove_var)` outside guards
  ```bash
  rg "std::env::(set_var|remove_var)" --type rust \
    --glob '!tests/support/env_guard.rs' | grep -v "#\[serial(bitnet_env)\]"
  # Expected: No matches
  ```

- [ ] **AC4**: Documentation complete
  ```bash
  grep -q "EnvGuard" CLAUDE.md
  grep -q "serial(bitnet_env)" docs/development/test-suite.md
  ```

### Affected Files (30 Total)

**PRIORITY 1** - `device_features.rs` (14 unprotected tests - CRITICAL):

| Line Range | Test Function | Env Variables | Fix |
|-----------|---------------|---------------|-----|
| 87-109 | `ac3_gpu_fake_cuda_overrides_detection` | `BITNET_GPU_FAKE` | Add `#[serial(bitnet_env)]` |
| 116-141 | `ac3_gpu_fake_none_disables_detection` | `BITNET_GPU_FAKE` | Add `#[serial(bitnet_env)]` |
| 162-193 | `ac3_gpu_fake_case_insensitive` | `BITNET_GPU_FAKE` (×4 loop) | Add `#[serial(bitnet_env)]` + EnvGuard |
| 198-238 | `ac3_gpu_compiled_but_runtime_unavailable` | `BITNET_GPU_FAKE` | Add `#[serial(bitnet_env)]` |
| 240-300 | `mutation_gpu_runtime_real_detection` | `BITNET_GPU_FAKE` (×2) | Add `#[serial(bitnet_env)]` + EnvGuard |
| 305-364 | `mutation_gpu_fake_or_semantics` | `BITNET_GPU_FAKE` (alt) | Add `#[serial(bitnet_env)]` + EnvGuard |
| 471-519 | `ac3_capability_summary_respects_fake` | `BITNET_GPU_FAKE` (×2) | Add `#[serial(bitnet_env)]` + EnvGuard |
| 520-560 | `ac_strict_mode_forbids_fake_gpu` | `BITNET_STRICT_MODE`, `BITNET_GPU_FAKE` | Add `#[serial(bitnet_env)]` |
| 557-594 | `ac_strict_mode_allows_real_gpu` | `BITNET_STRICT_MODE`, `BITNET_GPU_FAKE` | Add `#[serial(bitnet_env)]` |

**Additional**: Line 83 has incorrect path `#[path = "../support/mod.rs"]` → should be `#[path = "support/mod.rs"]`

**PRIORITY 2** - Secondary files (31+ tests):
- `crates/bitnet-tokenizers/tests/cross_validation_tests.rs` (1 test)
- `crates/bitnet-tokenizers/tests/integration_tests.rs` (1 test)
- `crates/bitnet-tokenizers/tests/test_ac4_smart_download_integration.rs` (1 test)
- `xtask/tests/ffi_build_tests.rs` (1 test)
- `crates/bitnet-server/tests/ac03_model_hot_swapping.rs` (1 test)
- `crates/bitnet-server/tests/otlp_metrics_test.rs` (4 tests)
- `xtask/tests/ci_integration_tests.rs` (2 tests)
- `xtask/tests/tokenizer_subcommand_tests.rs` (1 test)
- `tests-new/fixtures/fixtures/validation_tests.rs` (1 test)
- `tests-new/integration/debug_integration.rs` (1 test)
- `tests/common/gpu.rs` (3 tests)
- `tests/test_enhanced_error_handling.rs` (1 test)
- `tests/test_fixture_reliability.rs` (4 tests)
- `tests/test_configuration_scenarios.rs` (multiple tests - needs analysis)

**Full List**: See `/home/steven/code/Rust/BitNet-rs/ENV_VAR_MUTATION_AUDIT_REPORT.md`

### Implementation Steps

**Phase 1**: Fix `device_features.rs` (2 hours)
1. Fix missing support module path (line 83)
2. Add `#[serial(bitnet_env)]` to 14 unprotected tests
3. Convert raw env mutations to EnvGuard pattern

**Phase 2**: Fix secondary files (2-3 hours)
1. Batch 1: Simple `#[serial(bitnet_env)]` additions (13 files)
2. Batch 2: Complex refactors with EnvGuard (4 files)
3. Batch 3: `test_configuration_scenarios.rs` deep refactor

**Phase 3**: Add CI job + tooling (1 hour)
1. Implement `cargo run -p xtask -- check-env-guards`
2. Add `test-env-guards` CI job
3. Add pre-commit hook

**Phase 4**: Update documentation (30 min)
1. Update CLAUDE.md with EnvGuard best practices
2. Add section to `docs/development/test-suite.md`
3. Add usage examples to `tests/support/env_guard.rs`

### Example Fixes

**Pattern 1: Simple Case**
```diff
     #[test]
+    #[serial(bitnet_env)]
     fn test_gpu_fake() {
-        std::env::set_var("BITNET_GPU_FAKE", "cuda");
+        let _guard = EnvGuard::new("BITNET_GPU_FAKE", "cuda");
         // ... test code
-        std::env::remove_var("BITNET_GPU_FAKE");
     }
```

**Pattern 2: Loop-Based**
```diff
     #[test]
+    #[serial(bitnet_env)]
     fn test_case_insensitive() {
+        use tests::support::env_guard::EnvGuard;
+
         for value in &["cuda", "CUDA"] {
-            std::env::set_var("BITNET_GPU_FAKE", value);
+            let _guard = EnvGuard::new("BITNET_GPU_FAKE", value);
             assert!(gpu_available_runtime());
-            std::env::remove_var("BITNET_GPU_FAKE");
         }
     }
```

### Testing Verification

```bash
# Verify all 45 tests protected
cargo run -p xtask -- check-env-guards

# Run in parallel (should not flake)
for i in {1..10}; do
    RUST_TEST_THREADS=4 cargo test -p bitnet-kernels --test device_features || exit 1
done

# Simulate CI job
RUST_TEST_THREADS=4 cargo test --workspace --no-default-features --features cpu -- env
```

### References

- **Specification**: `/home/steven/code/Rust/BitNet-rs/docs/explanation/specs/SPEC-2025-003-envguard-serial-rollout.md`
- **Audit Report**: `/home/steven/code/Rust/BitNet-rs/ENV_VAR_MUTATION_AUDIT_REPORT.md`
- **EnvGuard Implementation**: `/home/steven/code/Rust/BitNet-rs/tests/support/env_guard.rs`
- **serial_test crate**: [Documentation](https://docs.rs/serial_test)

---

## Issue #3: Investigate and Fix PR #475 All-Features CI Failures

**Labels**: `P0`, `ci`, `feature-gates`, `blocker`
**Milestone**: v0.2.0
**Estimated Effort**: 1-2 hours

### Problem Statement

PR #475 achieves **100% test pass rate** with `--features cpu` (1935/1935 tests passing), but encounters compilation failures with `--all-features`:

```
error: couldn't read `crates/bitnet-kernels/tests/runtime_detection/support/mod.rs`:
       No such file or directory (os error 2)
  --> crates/bitnet-kernels/tests/device_features.rs:83:5
   |
83 |     mod support;
   |     ^^^^^^^^^^^^
```

**Root Cause**: Incorrect module path in `device_features.rs` - references non-existent `runtime_detection/support/mod.rs` subdirectory.

**Impact**: Blocks PR #475 merge; affects CI jobs using `--all-features` (doctest, potential all-features validation jobs).

### Acceptance Criteria

- [ ] **AC1**: Compilation succeeds with `--all-features`
  ```bash
  cargo check --workspace --all-features
  ```

- [ ] **AC2**: Clippy clean with `--all-features`
  ```bash
  cargo clippy --workspace --all-features --all-targets -- -D warnings
  ```

- [ ] **AC3**: Tests compile with `--all-features`
  ```bash
  cargo test --workspace --all-features --no-run
  ```

- [ ] **AC4**: Doctest passes (CI job simulation)
  ```bash
  cargo test --doc --workspace --all-features
  ```

### Root Cause Analysis

**Current Code** (line 82-83):
```rust
#[cfg(test)]
mod runtime_detection {
    #[path = "../support/mod.rs"]  // ❌ WRONG
    mod support;
}
```

**Problem**: Path tries to load `crates/bitnet-kernels/tests/runtime_detection/support/mod.rs` (does not exist)

**Actual Location**: `crates/bitnet-kernels/tests/support/mod.rs` (exists)

**Why Only With `--all-features`**: The `runtime_detection` module contains GPU-specific tests that are only compiled when GPU features are enabled.

### Affected Files

**Primary** (require fix):
1. `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/tests/device_features.rs`
   - **Line 82-83**: Fix module path

**Secondary** (verification):
2. `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/tests/support/mod.rs`
   - Verify file exists and is accessible

3. `/home/steven/code/Rust/BitNet-rs/.github/workflows/ci.yml`
   - **Line 161**: Verify doctest job configuration
   - Consider adding `clippy-all-features` and `test-all-features` jobs

### Implementation Steps

**Step 1**: Fix module path (5 minutes)

```diff
--- a/crates/bitnet-kernels/tests/device_features.rs
+++ b/crates/bitnet-kernels/tests/device_features.rs
@@ -79,7 +79,7 @@ mod runtime_detection {
     #[cfg(any(feature = "gpu", feature = "cuda"))]
     use serial_test::serial;

-    #[path = "../support/mod.rs"]
+    #[path = "support/mod.rs"]
     mod support;

     /// AC:3 - gpu_available_runtime() respects BITNET_GPU_FAKE=cuda
```

**Step 2**: Verify compilation (10 minutes)

```bash
# Test with --features cpu (should still pass)
cargo check --workspace --no-default-features --features cpu

# Test with --all-features (should now pass)
cargo check --workspace --all-features

# Clippy with --all-features
cargo clippy --workspace --all-features --all-targets -- -D warnings

# Doctest (matches CI job)
cargo test --doc --workspace --all-features
```

**Step 3**: CI configuration audit (15 minutes)

Check if additional CI jobs are needed:
- `clippy-all-features`: Comprehensive lint validation
- `test-all-features`: Full feature matrix testing

**Step 4**: Feature gate consistency check (30 minutes)

Validate hypothesis about GPU feature triggering:
```bash
# Test with only GPU feature
cargo check -p bitnet-kernels --tests --no-default-features --features gpu

# Test with only CPU feature
cargo check -p bitnet-kernels --tests --no-default-features --features cpu

# Test with both features
cargo check -p bitnet-kernels --tests --no-default-features --features cpu,gpu
```

### Testing Verification

```bash
# Pre-fix: Confirm error exists
! cargo check --workspace --all-features 2>&1 | grep "couldn't read.*runtime_detection/support"

# Post-fix: All checks pass
cargo check --workspace --all-features
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace --all-features --no-run
cargo test --doc --workspace --all-features

# No similar errors in other files
! rg '#\[path = "\.\./support' --type rust | grep -v "device_features.rs"
```

### Follow-Up Actions

**Immediate** (this PR):
1. Fix module path in `device_features.rs`
2. Verify compilation with `--all-features`
3. Update CLAUDE.md with `--all-features` testing guidance

**Short-Term** (next PR):
1. Add `clippy-all-features` CI job
2. Add `test-all-features` CI job
3. Add `cargo check --all-features` to pre-commit hooks

**Long-Term** (v0.2.0):
1. Add `cargo xtask check-feature-matrix` command
2. Document feature gate best practices
3. Add feature gate consistency lints

### References

- **Specification**: `/home/steven/code/Rust/BitNet-rs/docs/explanation/specs/SPEC-2025-004-all-features-ci-failure-investigation.md`
- **PR #475**: Comprehensive integration PR
- **Related**: Issue #439 (Feature gate unification)
- **Cargo Book**: [Feature Resolution](https://doc.rust-lang.org/cargo/reference/features.html)

---

## Summary

**Total Issues**: 3
**Total Estimated Effort**: 7-11 hours
**Priority**: All P0 (Blocks v0.2.0)
**Dependencies**: Issue #3 blocks PR #475 merge; Issues #1-2 are independent

**Recommended Order**:
1. **Issue #3** (1-2 hours) - Immediate blocker for PR #475
2. **Issue #2** (4-6 hours) - Eliminates CI flakiness
3. **Issue #1** (2-3 hours) - Hardens build reliability

All specifications are available in `/home/steven/code/Rust/BitNet-rs/docs/explanation/specs/` for detailed implementation guidance.
