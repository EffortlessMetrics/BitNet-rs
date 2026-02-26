# SPEC-2025-002: Build Script Hygiene Hardening

**Status**: Draft
**Priority**: P0 (Blocks Production Reliability)
**Estimated Effort**: 2-3 hours
**Target Release**: v0.2.0
**Created**: 2025-10-23

---

## Problem Statement

BitNet-rs build scripts (`build.rs`) contain unhygienic patterns that can cause silent failures in minimal container environments and CI/CD pipelines:

1. **Missing `HOME` environment variable handling**: The `bitnet-ggml-ffi/build.rs` script reads `csrc/VENDORED_GGML_COMMIT` using `.unwrap_or_else()` with only `eprintln!()` warnings, not `cargo:warning=` directives
2. **Silent CI failures**: When `VENDORED_GGML_COMMIT` is missing in CI, the build script panics instead of emitting visible warnings
3. **Inconsistent error handling**: `bitnet-kernels/build.rs` uses proper `cargo:warning=` for `HOME` fallback, but `bitnet-ggml-ffi/build.rs` does not

**Production Impact**: Builds succeed locally but fail silently in Docker containers or CI environments without clear diagnostics.

---

## Acceptance Criteria

### AC1: Remove All `unwrap()` Calls from Build Scripts

**Verification**:
```bash
# Must return empty (exit code 1 from grep)
grep -n "unwrap()" crates/bitnet-kernels/build.rs crates/bitnet-ggml-ffi/build.rs
```

**Expected Output**: No matches found

---

### AC2: `cargo:warning=` Fallback Directives Visible

**Requirement**: All error conditions emit `cargo:warning=` directives visible during `cargo build`

**Test Case**:
```bash
# Remove VENDORED_GGML_COMMIT marker
rm crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT

# Build with warnings visible
cargo build --no-default-features --features cpu -p bitnet-ggml-ffi 2>&1 | grep -i "warning"

# Expected output:
# warning: Failed to read VENDORED_GGML_COMMIT marker at 'csrc/VENDORED_GGML_COMMIT': No such file or directory
# warning: Using 'unknown' as fallback. Run 'cargo xtask vendor-ggml' to fix.
```

---

### AC3: Builds Succeed Without `$HOME` Environment Variable

**Test Case**:
```bash
# Unset HOME and build
env -u HOME cargo build --no-default-features --features cpu -p bitnet-kernels 2>&1

# Expected:
# - Build completes successfully
# - Warning emitted: "cargo:warning=HOME not set; falling back to /tmp for C++ artifact cache (build.rs)"
# - Exit code: 0
```

---

### AC4: CI Panic on Missing Critical Markers

**Requirement**: In `CI=1` environment, missing `VENDORED_GGML_COMMIT` must panic with actionable message

**Test Case**:
```bash
# Remove marker and build in CI mode
rm crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT
CI=1 cargo build --no-default-features --features cpu -p bitnet-ggml-ffi 2>&1

# Expected:
# thread 'main' panicked at crates/bitnet-ggml-ffi/build.rs:XX:YY:
# VENDORED_GGML_COMMIT is 'unknown' in CI.
# Run: cargo xtask vendor-ggml --commit <sha>
# Or set crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT
```

---

## Affected Files

### Primary Files (Require Modification)

1. **`/home/steven/code/Rust/BitNet-rs/crates/bitnet-ggml-ffi/build.rs`**
   - **Lines 8-18**: Replace `eprintln!()` with `println!("cargo:warning=...")`
   - **Current**:
     ```rust
     let commit = fs::read_to_string(marker).unwrap_or_else(|e| {
         eprintln!(
             "cargo:warning=Failed to read VENDORED_GGML_COMMIT marker at '{}': {}",
             marker.display(),
             e
         );
         eprintln!(
             "cargo:warning=Using 'unknown' as fallback. Run 'cargo xtask vendor-ggml' to fix."
         );
         "unknown".into()
     });
     ```
   - **Fixed**:
     ```rust
     let commit = fs::read_to_string(marker).unwrap_or_else(|e| {
         println!(
             "cargo:warning=Failed to read VENDORED_GGML_COMMIT marker at '{}': {}",
             marker.display(),
             e
         );
         println!(
             "cargo:warning=Using 'unknown' as fallback. Run 'cargo xtask vendor-ggml' to fix."
         );
         "unknown".into()
     });
     ```

2. **`/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/build.rs`**
   - **Lines 11-21**: Already follows correct pattern (no changes needed, reference example)
   - **Good Example**:
     ```rust
     fn get_home_dir() -> PathBuf {
         if let Some(home) = env_var("HOME") {
             return PathBuf::from(home);
         }
         println!("cargo:warning=HOME not set; falling back to /tmp for C++ artifact cache (build.rs)");
         PathBuf::from("/tmp")
     }
     ```

### Secondary Files (Documentation)

3. **`/home/steven/code/Rust/BitNet-rs/docs/development/build-commands.md`**
   - Add section: "Build Script Hygiene and Error Handling"
   - Document `cargo:warning=` directive usage
   - Document `CI=1` panic behavior for missing critical markers

4. **`/home/steven/code/Rust/BitNet-rs/CLAUDE.md`**
   - Update "Build script hygiene" section (if exists)
   - Add troubleshooting entry for `VENDORED_GGML_COMMIT` errors

---

## Implementation Approach

### Step 1: Replace `eprintln!()` with `println!("cargo:warning=...")`

**Rationale**: Cargo only recognizes `println!()` directives from build scripts, not `eprintln!()`

**Change**:
```diff
--- a/crates/bitnet-ggml-ffi/build.rs
+++ b/crates/bitnet-ggml-ffi/build.rs
@@ -7,13 +7,13 @@
         // Inject vendored commit into the crate's env
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

---

### Step 2: Add Verification Tests

**New Test File**: `/home/steven/code/Rust/BitNet-rs/xtask/tests/build_script_hygiene_tests.rs`

```rust
#[test]
fn test_no_unwrap_in_build_scripts() {
    use std::process::Command;

    let output = Command::new("grep")
        .args(&["-n", "unwrap()", "crates/bitnet-kernels/build.rs", "crates/bitnet-ggml-ffi/build.rs"])
        .output()
        .expect("Failed to run grep");

    assert!(
        !output.status.success(),
        "Found unwrap() calls in build scripts:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn test_cargo_warnings_visible() {
    use std::process::Command;
    use std::fs;

    // Temporarily rename marker file
    let marker = "crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT";
    let backup = format!("{}.bak", marker);

    if fs::metadata(marker).is_ok() {
        fs::rename(marker, &backup).expect("Failed to backup marker");
    }

    let output = Command::new("cargo")
        .args(&["build", "--no-default-features", "--features", "cpu", "-p", "bitnet-ggml-ffi"])
        .output()
        .expect("Failed to run cargo build");

    // Restore marker
    if fs::metadata(&backup).is_ok() {
        fs::rename(&backup, marker).expect("Failed to restore marker");
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("cargo:warning=Failed to read VENDORED_GGML_COMMIT"),
        "Expected cargo:warning directive not found in build output:\n{}",
        stderr
    );
}

#[test]
#[cfg_attr(not(feature = "ci_tests"), ignore)]
fn test_ci_panic_on_missing_marker() {
    use std::process::Command;
    use std::fs;

    // Temporarily rename marker file
    let marker = "crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT";
    let backup = format!("{}.bak", marker);

    if fs::metadata(marker).is_ok() {
        fs::rename(marker, &backup).expect("Failed to backup marker");
    }

    let output = Command::new("cargo")
        .env("CI", "1")
        .args(&["build", "--no-default-features", "--features", "cpu", "-p", "bitnet-ggml-ffi"])
        .output()
        .expect("Failed to run cargo build");

    // Restore marker
    if fs::metadata(&backup).is_ok() {
        fs::rename(&backup, marker).expect("Failed to restore marker");
    }

    assert!(
        !output.status.success(),
        "Build should have panicked in CI mode with missing marker"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("VENDORED_GGML_COMMIT is 'unknown' in CI"),
        "Expected CI panic message not found:\n{}",
        stderr
    );
}
```

---

### Step 3: Update Documentation

**File**: `/home/steven/code/Rust/BitNet-rs/docs/development/build-commands.md`

Add new section:

```markdown
## Build Script Hygiene

BitNet-rs build scripts follow strict hygiene rules to ensure reliable builds in all environments:

### Cargo Warning Directives

All non-fatal build errors MUST use `println!("cargo:warning=...")`, not `eprintln!()`:

```rust
// ✅ GOOD - Warning visible in cargo output
println!("cargo:warning=HOME not set; falling back to /tmp");

// ❌ BAD - Warning only visible in stderr, not cargo diagnostics
eprintln!("cargo:warning=HOME not set; falling back to /tmp");
```

### CI Panic Behavior

In `CI=1` environments, build scripts panic on missing critical markers:

```bash
# CI build with missing VENDORED_GGML_COMMIT
CI=1 cargo build --features cpu -p bitnet-ggml-ffi

# Panics with actionable message:
# VENDORED_GGML_COMMIT is 'unknown' in CI.
# Run: cargo xtask vendor-ggml --commit <sha>
```

### No `unwrap()` Policy

Build scripts MUST NOT use `.unwrap()` or `.expect()` without graceful fallbacks:

```rust
// ✅ GOOD - Graceful fallback with warning
let commit = fs::read_to_string(marker).unwrap_or_else(|e| {
    println!("cargo:warning=Failed to read marker: {}", e);
    "unknown".into()
});

// ❌ BAD - Panics without context
let commit = fs::read_to_string(marker).unwrap();
```
```

---

## Testing Strategy

### Unit Tests

1. **Grep-based validation**: No `unwrap()` calls in build scripts
2. **Warning visibility test**: `cargo:warning=` directives appear in build output
3. **CI panic test**: Missing markers trigger panic with actionable message

### Integration Tests

1. **Minimal container test**: Build succeeds in `FROM scratch` Docker image
2. **CI pipeline test**: Build succeeds in GitHub Actions with `CI=1`
3. **Offline build test**: Build succeeds without network access

### Manual Verification

```bash
# Test 1: Build without HOME
env -u HOME cargo build --no-default-features --features cpu -p bitnet-kernels

# Test 2: Build with missing marker (local mode)
mv crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT{,.bak}
cargo build --no-default-features --features cpu -p bitnet-ggml-ffi 2>&1 | grep warning
mv crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT{.bak,}

# Test 3: Build with missing marker (CI mode - should panic)
mv crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT{,.bak}
! CI=1 cargo build --no-default-features --features cpu -p bitnet-ggml-ffi
mv crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT{.bak,}

# Test 4: Verify no unwrap() calls
! grep -n "unwrap()" crates/*/build.rs
```

---

## Risk Assessment

### Low Risk

- **Scope**: Only 2 files modified (1 fix + 1 test addition)
- **Backward Compatibility**: No API changes, only error handling improvements
- **Rollback**: Easy revert if issues discovered

### Potential Issues

1. **CI flakiness**: If `VENDORED_GGML_COMMIT` is dynamically generated in CI, panic behavior might cause false failures
   - **Mitigation**: Ensure `cargo xtask vendor-ggml` runs before build in CI
   - **Fallback**: Add `CI_ALLOW_UNKNOWN_COMMIT=1` escape hatch if needed

2. **Warning noise**: Developers without C++ setup might see warnings on every build
   - **Mitigation**: Clear documentation on how to suppress (run `cargo xtask vendor-ggml`)
   - **Accepted**: Better than silent failures

---

## Success Criteria

### Measurable Outcomes

1. ✅ **Zero `unwrap()` calls** in `build.rs` files (grep verification)
2. ✅ **100% warning visibility** in `cargo build` output (manual test)
3. ✅ **CI panic on missing markers** (automated test in `ci_tests` feature)
4. ✅ **Docker build success** without `$HOME` environment variable
5. ✅ **Documentation complete** with hygiene rules and troubleshooting

### Verification Commands

```bash
# AC1: No unwrap() calls
! grep -rn "unwrap()" crates/bitnet-kernels/build.rs crates/bitnet-ggml-ffi/build.rs

# AC2: Warnings visible
rm crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT
cargo build -p bitnet-ggml-ffi 2>&1 | grep "cargo:warning=Failed to read VENDORED_GGML_COMMIT"
git checkout crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT

# AC3: Build without HOME
env -u HOME cargo build --no-default-features --features cpu -p bitnet-kernels

# AC4: CI panic
! CI=1 cargo build -p bitnet-ggml-ffi 2>&1 | grep "VENDORED_GGML_COMMIT is 'unknown' in CI"
```

---

## References

- **CLAUDE.md**: Section on build script hygiene
- **Related Issue**: #439 (Feature gate consistency - similar hygiene improvements)
- **Cargo Book**: [Build Scripts - Warning Directives](https://doc.rust-lang.org/cargo/reference/build-scripts.html#cargo-warning)
- **BitNet-rs Contracts**: "Never modify GGUF in-place", "Always specify features"

---

## Implementation Timeline

| Task | Effort | Owner |
|------|--------|-------|
| Step 1: Fix `eprintln!()` → `println!()` | 15 min | TBD |
| Step 2: Add verification tests | 60 min | TBD |
| Step 3: Update documentation | 30 min | TBD |
| Step 4: Manual verification | 30 min | TBD |
| **Total** | **2-3 hours** | |

---

## Notes

- This spec follows BitNet-rs TDD practices: tests before implementation
- Warning directives are production-critical for container deployments
- CI panic behavior prevents silent failures in automated pipelines
