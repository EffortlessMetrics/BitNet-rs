# SPEC-2025-004: All-Features CI Failure Investigation

**Status**: Draft
**Priority**: P0 (Blocks Merge - PR #475)
**Estimated Effort**: 1-2 hours (Investigation + Fix)
**Target Release**: v0.2.0
**Created**: 2025-10-23

---

## Problem Statement

PR #475 encounters compilation failures when using `--all-features` flag, specifically:

```
error: couldn't read `crates/bitnet-kernels/tests/runtime_detection/support/mod.rs`:
       No such file or directory (os error 2)
  --> crates/bitnet-kernels/tests/device_features.rs:83:5
   |
83 |     mod support;
   |     ^^^^^^^^^^^^
```

**Root Cause**: Incorrect module path in `device_features.rs` test file. The path references a non-existent `runtime_detection/support/mod.rs` subdirectory.

**Impact**:
- Blocks PR #475 merge (1935/1935 tests passing with `--features cpu`, but fails with `--all-features`)
- Affects CI jobs that run with full feature set
- Prevents comprehensive feature gate validation

---

## Acceptance Criteria

### AC1: Compilation Succeeds with `--all-features`

**Verification**:
```bash
# Must compile without errors
cargo check --workspace --all-features

# Expected output:
# Finished `dev` profile [unoptimized + debuginfo] target(s) in X.XXs
```

---

### AC2: Clippy Clean with `--all-features`

**Verification**:
```bash
# Must pass with zero warnings
cargo clippy --workspace --all-features --all-targets -- -D warnings

# Expected output:
# Finished `dev` profile [unoptimized + debuginfo] target(s) in X.XXs
```

---

### AC3: Tests Pass with `--all-features`

**Verification**:
```bash
# Must pass all enabled tests
cargo test --workspace --all-features --no-run

# Expected output:
# Finished `test` profile [unoptimized + debuginfo] target(s) in X.XXs
```

---

### AC4: CI Jobs Pass on PR #475

**CI Jobs to Verify**:
1. **Doctest** (`test --doc --workspace --all-features`) - Line 161 in `.github/workflows/ci.yml`
2. **Clippy all-features** (if exists) - Currently not defined, but should be added
3. **Test all-features** (if exists) - Currently not defined, but should be added

**Verification**:
```bash
# Simulate doctest job (from .github/workflows/ci.yml:161)
cargo test --doc --workspace --all-features

# Simulate potential clippy-all-features job
cargo clippy --workspace --all-features --all-targets -- -D warnings

# Simulate potential test-all-features job
cargo test --workspace --all-features --no-run
```

---

## Root Cause Analysis

### Issue 1: Incorrect Module Path in device_features.rs

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/tests/device_features.rs`
**Line**: 82-83

**Current Code**:
```rust
#[cfg(test)]
mod runtime_detection {
    // ...
    #[path = "../support/mod.rs"]  // ❌ WRONG - references parent of tests/
    mod support;
}
```

**Problem**: The `#[path = "../support/mod.rs"]` directive tries to load:
- `crates/bitnet-kernels/tests/runtime_detection/support/mod.rs` (DOES NOT EXIST)

**Actual File Location**:
- `crates/bitnet-kernels/tests/support/mod.rs` (EXISTS)

**Correct Code**:
```rust
#[cfg(test)]
mod runtime_detection {
    // ...
    #[path = "support/mod.rs"]  // ✅ CORRECT - relative to tests/ directory
    mod support;
}
```

**Why This Matters**:
- Test modules are rooted at `crates/bitnet-kernels/tests/`
- `runtime_detection` is a submodule, not a subdirectory
- Path should be relative to the test root, not the submodule

---

### Issue 2: All-Features vs Feature-Gated Builds

**Hypothesis**: The error only manifests with `--all-features` because:

1. **With `--features cpu`**: Only CPU-related device features compiled
2. **With `--all-features`**: All conditional compilation paths activated, including GPU+CPU tests
3. **Path resolution**: The incorrect path is within a module that may have additional `#[cfg(...)]` gates

**Investigation Needed**:
```bash
# Check if runtime_detection module has feature gates
grep -A 10 "mod runtime_detection" crates/bitnet-kernels/tests/device_features.rs

# Expected finding:
# #[cfg(test)]
# mod runtime_detection {
#     #[cfg(any(feature = "gpu", feature = "cuda"))]
#     use serial_test::serial;
#
#     #[path = "../support/mod.rs"]  // <- ERROR HERE
#     mod support;
# }
```

**Explanation**: The module might only compile when both:
- `test` cfg is active (always true for `cargo test`)
- `gpu` or `cuda` features are enabled (only true with `--all-features`)

---

## Affected Files

### PRIMARY: Files Requiring Fixes

1. **`/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/tests/device_features.rs`**
   - **Line 82-83**: Fix module path
   - **Change**: `#[path = "../support/mod.rs"]` → `#[path = "support/mod.rs"]`

### SECONDARY: Files Requiring Verification

2. **`/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/tests/support/mod.rs`**
   - **Status**: Verify file exists and is accessible
   - **Expected**: File should contain `EnvVarGuard` implementation

3. **`/home/steven/code/Rust/BitNet-rs/.github/workflows/ci.yml`**
   - **Line 161**: Verify doctest job uses `--all-features`
   - **Action**: Check if additional all-features jobs are needed

---

## Implementation Approach

### Step 1: Fix Module Path (5 minutes)

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/tests/device_features.rs`

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

**Rationale**:
- Test modules in `crates/foo/tests/` are rooted at the `tests/` directory
- Relative paths should be from the root, not from inline module declarations
- `support/mod.rs` correctly references `crates/bitnet-kernels/tests/support/mod.rs`

---

### Step 2: Verify Compilation (10 minutes)

```bash
# Test 1: Check with --features cpu (should still pass)
cargo check --workspace --no-default-features --features cpu

# Test 2: Check with --all-features (should now pass)
cargo check --workspace --all-features

# Test 3: Clippy with --all-features (should be clean)
cargo clippy --workspace --all-features --all-targets -- -D warnings

# Test 4: Test compilation with --all-features
cargo test --workspace --all-features --no-run

# Test 5: Doctest (matches CI job)
cargo test --doc --workspace --all-features
```

---

### Step 3: CI Configuration Audit (15 minutes)

**Goal**: Ensure CI validates all feature combinations

**Current State** (`.github/workflows/ci.yml`):
- Line 161: `cargo test --doc --workspace --all-features` (doctest job)
- Line 109: `cargo clippy --workspace --no-default-features --features cpu ...` (only CPU)
- Line 123: `cargo nextest run --workspace --no-default-features --features cpu` (only CPU)

**Missing Jobs**:
1. **clippy-all-features**: Comprehensive lint validation
2. **test-all-features**: Full feature matrix testing

**Recommendation**: Add new CI jobs (optional, but recommended)

```yaml
  clippy-all-features:
    name: Clippy (All Features)
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
      - name: Run clippy with all features
        run: cargo clippy --workspace --all-features --all-targets -- -D warnings

  test-all-features-compile:
    name: Test Compilation (All Features)
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
      - name: Check all features compile
        run: cargo test --workspace --all-features --no-run
```

**Note**: These jobs ensure feature gate consistency across all combinations.

---

### Step 4: Feature Gate Consistency Check (30 minutes)

**Investigation**: Determine why error only appears with `--all-features`

```bash
# Check feature gates in device_features.rs
rg "#\[cfg\(" crates/bitnet-kernels/tests/device_features.rs

# Expected findings:
# - #[cfg(test)] - always active for tests
# - #[cfg(any(feature = "gpu", feature = "cuda"))] - GPU-related tests
# - Other conditional compilation directives
```

**Hypothesis Validation**:

1. **Test with only GPU feature**:
   ```bash
   cargo check -p bitnet-kernels --tests --no-default-features --features gpu
   ```
   - **Expected**: Error appears (confirms GPU feature triggers the path)

2. **Test with only CPU feature**:
   ```bash
   cargo check -p bitnet-kernels --tests --no-default-features --features cpu
   ```
   - **Expected**: No error (confirms CPU-only builds skip the module)

3. **Test with both features**:
   ```bash
   cargo check -p bitnet-kernels --tests --no-default-features --features cpu,gpu
   ```
   - **Expected**: Error appears (confirms both features together trigger it)

**Conclusion**: The `runtime_detection` module likely contains GPU-specific tests that are only compiled when GPU features are enabled.

---

## Testing Strategy

### Pre-Fix Validation

```bash
# Confirm error exists
! cargo check --workspace --all-features 2>&1 | grep "couldn't read.*runtime_detection/support/mod.rs"

# Expected: Error found
```

### Post-Fix Validation

```bash
# AC1: Compilation succeeds
cargo check --workspace --all-features

# AC2: Clippy clean
cargo clippy --workspace --all-features --all-targets -- -D warnings

# AC3: Tests compile
cargo test --workspace --all-features --no-run

# AC4: Doctest passes (CI simulation)
cargo test --doc --workspace --all-features

# Feature-specific validation
cargo check -p bitnet-kernels --tests --no-default-features --features cpu
cargo check -p bitnet-kernels --tests --no-default-features --features gpu
cargo check -p bitnet-kernels --tests --no-default-features --features cpu,gpu
```

### Regression Prevention

**Add to `.github/workflows/ci.yml`** (recommended):

```yaml
  verify-all-features:
    name: Verify All Features Compile
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
      - name: Check all features
        run: cargo check --workspace --all-features
      - name: Clippy all features
        run: cargo clippy --workspace --all-features --all-targets -- -D warnings
      - name: Test all features compile
        run: cargo test --workspace --all-features --no-run
```

---

## Risk Assessment

### Low Risk

- **Scope**: Single-line fix in test file
- **Backward Compatibility**: No API changes, only test infrastructure
- **Rollback**: Trivial revert if issues discovered

### Potential Issues

1. **Other incorrect paths**: Similar errors might exist in other test files
   - **Mitigation**: Comprehensive grep for `#[path = "../support/`
   - **Verification**: `rg '#\[path = "\.\./support' --type rust`

2. **Feature gate inconsistencies**: Other modules might have similar issues
   - **Mitigation**: Add `clippy-all-features` CI job
   - **Detection**: Regular CI runs with full feature matrix

3. **Doctest failures**: Fix might expose other all-features issues
   - **Mitigation**: Run doctests locally before merge
   - **Fallback**: Document and file follow-up issues

---

## Success Criteria

### Measurable Outcomes

1. ✅ **Compilation succeeds** with `--all-features` (exit code 0)
2. ✅ **Clippy clean** with `--all-features` (zero warnings)
3. ✅ **Tests compile** with `--all-features` (no-run succeeds)
4. ✅ **Doctest passes** with `--all-features` (CI job succeeds)
5. ✅ **No similar errors** in other test files (grep verification)

### Verification Commands

```bash
# AC1: Compilation
cargo check --workspace --all-features

# AC2: Clippy
cargo clippy --workspace --all-features --all-targets -- -D warnings

# AC3: Test compilation
cargo test --workspace --all-features --no-run

# AC4: Doctest
cargo test --doc --workspace --all-features

# Additional: No similar errors
! rg '#\[path = "\.\./support' --type rust | grep -v "^crates/bitnet-kernels/tests/device_features.rs"
```

---

## Root Cause Deep Dive

### Why Did This Pass Before?

**Timeline Analysis**:

1. **PR #475 Branch**: `feat/comprehensive-integration-qk256-envguard-receipts-strict-avx2`
   - Includes EnvGuard rollout changes
   - Modified `device_features.rs` to use EnvGuard
   - Introduced incorrect module path during refactoring

2. **CI Configuration**:
   - Primary test jobs use `--no-default-features --features cpu`
   - Doctest job uses `--all-features` but may not have been running on this branch

3. **Local Development**:
   - Developers typically test with `--features cpu` or `--features gpu`
   - `--all-features` rarely used locally (not in quick reference commands)

**Lesson Learned**: Add `--all-features` to regular development workflow

---

## Follow-Up Actions

### Immediate (This PR)

1. ✅ Fix module path in `device_features.rs`
2. ✅ Verify compilation with `--all-features`
3. ✅ Update CLAUDE.md with `--all-features` testing guidance

### Short-Term (Next PR)

1. Add `clippy-all-features` CI job
2. Add `test-all-features` CI job
3. Add `cargo check --all-features` to pre-commit hooks

### Long-Term (v0.2.0)

1. Add `cargo xtask check-feature-matrix` command
2. Document feature gate best practices
3. Add feature gate consistency lints

---

## References

- **PR #475**: Comprehensive integration PR with 100% test pass rate (CPU features)
- **Issue #439**: Feature gate unification (similar consistency work)
- **CLAUDE.md**: Quick reference commands (needs `--all-features` addition)
- **Cargo Book**: [Feature Resolution](https://doc.rust-lang.org/cargo/reference/features.html)

---

## Implementation Timeline

| Task | Effort | Owner |
|------|--------|-------|
| Step 1: Fix module path | 5 min | TBD |
| Step 2: Verify compilation | 10 min | TBD |
| Step 3: CI configuration audit | 15 min | TBD |
| Step 4: Feature gate consistency check | 30 min | TBD |
| Documentation updates | 15 min | TBD |
| **Total** | **1-2 hours** | |

---

## Notes

- This issue is a **blocker** for PR #475 merge (1935/1935 tests passing, but CI fails)
- The fix is **trivial** (single line change), but investigation is important
- Adding `--all-features` CI jobs prevents future regressions
- This highlights the importance of testing **all feature combinations**, not just primary configurations
- The error message is **clear and actionable** ("No such file or directory"), making investigation straightforward
- Root cause is **refactoring error**, not a fundamental design issue

---

## Related Work

- **SPEC-2025-003**: EnvGuard rollout (introduces changes to `device_features.rs`)
- **ENV_VAR_MUTATION_AUDIT_REPORT.md**: Identifies 14 unprotected tests in `device_features.rs`
- **PR_475_FINAL_SUCCESS_REPORT.md**: Documents 100% test pass rate with `--features cpu`

This spec completes the investigation phase and provides a clear implementation path for fixing the all-features CI failures.
