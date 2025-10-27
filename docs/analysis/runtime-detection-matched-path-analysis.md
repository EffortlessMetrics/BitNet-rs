# Analysis: Runtime Detection with Matched Path Tracking

**Document ID**: `runtime-detection-matched-path-analysis`  
**Date**: 2025-10-27  
**Status**: Complete  
**Focus**: Feasibility and design for exposing matched path information from runtime detection

## Executive Summary

Runtime detection in BitNet.rs currently returns only a boolean indicating whether libraries were found. To support enhanced diagnostics and CI-safe warning features, the detection functions must return **matched path information** along with availability status.

**Current Return Type** (test helper):
```rust
Result<(bool, Option<PathBuf>), String>
```

**Target Return Type** (already implemented):
```rust
Result<(bool, Option<PathBuf>), String>
```

This analysis examines:
1. **Existing implementations** and their current signatures
2. **Matched path information** already available in detection
3. **Integration impact** on production code
4. **Backward compatibility** strategy
5. **Implementation roadmap**

**Key Finding**: The enhancement is complete in test helpers. Matched paths are already being located and returned during detection. Production code (preflight.rs) simply needs to use this information.

---

## Section 1: Current State Analysis

### 1.1 Test Helper Implementation (`tests/support/backend_helpers.rs`)

**Current Function** (lines 581-652):
```rust
pub fn detect_backend_runtime(
    backend: CppBackend,
) -> Result<(bool, Option<std::path::PathBuf>), String>
```

**Status**: ✅ ALREADY RETURNS MATCHED PATH!

The test helper function **already implements the target signature**. It returns `Ok((true, Some(path)))` when libraries are found:

```rust
// Lines 633-647
for dir in candidates {
    if !dir.exists() { continue; }

    let all_found = needs.iter().all(|stem| {
        exts.iter().any(|ext| {
            let lib_name = format_lib_name_ext(stem, ext);
            dir.join(&lib_name).exists()
        })
    });

    if all_found {
        return Ok((true, Some(dir)));  // ← Returns matched path
    }
}

Ok((false, None))  // ← No path when not found
```

### 1.2 Supporting Functions

| Function | Location | Returns | Status |
|----------|----------|---------|--------|
| `get_library_search_paths()` | lines 605-616 | `Vec<PathBuf>` | ✅ Complete |
| `format_lib_name_ext()` | lines 666-672 | `String` | ✅ Complete |
| `is_ci()` | lines 230-237 | `bool` | ✅ Complete |
| `emit_stale_build_warning()` | lines 96-101 | `()` | ✅ Complete |
| `emit_verbose_stale_warning()` | lines 116-175 | `()` | ✅ Complete |
| `format_ci_stale_skip_diagnostic()` | lines 189-221 | `String` | ✅ Complete |

### 1.3 Production Code (Preflight)

**Current Function** (xtask/src/crossval/preflight.rs lines 519-695):
```rust
pub fn preflight_backend_libs(backend: CppBackend, verbose: bool) -> Result<()>
```

**Current Behavior**:
- Checks build-time constants (lines 552-555)
- Prints verbose diagnostics if requested
- Returns error if backend unavailable
- **Does NOT call runtime detection**

**Issue**: Production code doesn't leverage the matched path information already available in test helpers.

---

## Section 2: Three-Tier Search Path Analysis

Runtime detection implements a **3-tier priority search hierarchy** matching build-time logic:

### 2.1 Priority 1: Explicit Global Override
- **Variable**: `BITNET_CROSSVAL_LIBDIR`
- **Purpose**: Force detection to specific directory
- **Location**: lines 587-589

### 2.2 Priority 2: Backend-Specific RPATH Overrides
- **Variables**: `CROSSVAL_RPATH_BITNET`, `CROSSVAL_RPATH_LLAMA`
- **Feature**: Supports colon-separated paths (lines 367-373)
- **Purpose**: Backend-specific search paths

### 2.3 Priority 3: Backend Installation Root
- **Variables**: `BITNET_CPP_DIR`, `LLAMA_CPP_DIR`
- **Subdirectories Searched**: `build`, `build/bin`, `build/lib`
- **Location**: lines 605-616

### 2.4 Priority 4: Default Cache Fallback
- **Defaults**: `~/.cache/bitnet_cpp`, `~/.cache/llama_cpp`
- **Location**: lines 611-616

### 2.5 Platform-Specific Library Names

| Backend | Linux | macOS | Windows |
|---------|-------|-------|---------|
| BitNet | `libbitnet.so` | `libbitnet.dylib` | `bitnet.dll` |
| Llama | `libllama.so` + `libggml.so` | `.dylib` | `.dll` |

**Key Constraint**: For llama.cpp, **both** `libllama` and `libggml` must be present

---

## Section 3: Integration Points

### 3.1 What Already Exists

**Test Helper Usage** (lines 148-225 in runtime_detection_warning_tests.rs):
```rust
#[test]
fn test_detect_backend_runtime_returns_matched_path_bitnet() {
    let (found, path) = detect_backend_runtime(CppBackend::BitNet).unwrap();
    assert!(found);
    assert_eq!(path.unwrap(), expected_dir);
}
```

**Usage Pattern**: Tests already handle tuple return type correctly

### 3.2 What Needs Integration

**Preflight.rs** (lines 519-695):
- Currently: Only checks build-time constants
- Needed: Call runtime detection as Priority 2 fallback
- Needed: Use matched path for warning messages
- Needed: Branch by CI environment

**Integration Pattern**:
```rust
if let Ok((found, path)) = detect_backend_runtime(backend) {
    if found {
        if is_ci() {
            eprintln!("{}", format_ci_stale_skip_diagnostic(backend, path.as_deref()));
            std::process::exit(0);
        } else {
            emit_stale_build_warning(backend);
            return Ok(());
        }
    }
}
```

---

## Section 4: Dev vs CI Semantics

### 4.1 CI Mode (when `is_ci()` returns true)

**Detected Platforms**:
- GitHub Actions (`GITHUB_ACTIONS`)
- GitLab CI (`GITLAB_CI`)
- Jenkins (`JENKINS_HOME`)
- CircleCI (`CIRCLECI`)
- Generic CI (`CI`, `BITNET_TEST_NO_REPAIR`)

**Behavior**:
1. Runtime detection finds libraries ✓
2. Build-time constant is false (stale) ✓
3. **Action**: Skip test immediately
4. **Exit Code**: 0 (success/skip, not failure)
5. **Message**: CI diagnostic explaining stale build

**Rationale**: Ensures deterministic CI behavior

### 4.2 Dev Mode (when `is_ci()` returns false)

**Behavior**:
1. Runtime detection finds libraries ✓
2. Build-time constant is false (stale) ✓
3. **Action**: Emit warning and continue
4. **Exit Code**: Test proceeds normally
5. **Message**: Single-line warning + rebuild command

**Rationale**: Developer convenience without blocking tests

---

## Section 5: Minimal Code Changes for Production

### 5.1 Single File to Modify

**File**: `xtask/src/crossval/preflight.rs`

**Changes**:
1. Add import (1 line)
2. Add Priority 2 detection call (5-10 lines)
3. Add CI branching (10-15 lines)
4. **Total**: ~30 lines of code

### 5.2 Code Pattern

```rust
// Add to imports
use tests::support::backend_helpers::{
    detect_backend_runtime,
    emit_stale_build_warning,
    format_ci_stale_skip_diagnostic,
};

// In preflight_backend_libs function, after Priority 1:
if let Ok((found, matched_path)) = detect_backend_runtime(backend) {
    if found {
        if is_ci() {
            let diagnostic = format_ci_stale_skip_diagnostic(backend, matched_path.as_deref());
            eprintln!("{}", diagnostic);
            std::process::exit(0);
        } else {
            emit_stale_build_warning(backend);
            return Ok(());
        }
    }
}
```

---

## Section 6: Functions Already Implemented

| Function | Status | Location |
|----------|--------|----------|
| `detect_backend_runtime()` | ✅ Complete | backend_helpers.rs:581 |
| `get_library_search_paths()` | ✅ Complete | backend_helpers.rs:605 |
| `is_ci()` | ✅ Complete | preflight.rs:411 + backend_helpers.rs:230 |
| `emit_stale_build_warning()` | ✅ Complete | backend_helpers.rs:96 |
| `emit_verbose_stale_warning()` | ✅ Complete | backend_helpers.rs:116 |
| `format_ci_stale_skip_diagnostic()` | ✅ Complete | backend_helpers.rs:189 |

**All required functions exist and work correctly.**

---

## Section 7: Integration Checklist

- [x] Runtime detection returns matched path: `Ok((bool, Option<PathBuf>))`
- [x] Test helper functions exist and are complete
- [x] CI detection logic implemented and tested
- [x] Warning emission functions exist
- [x] Environment variable contracts documented
- [ ] Preflight.rs imports and calls runtime detection
- [ ] Matched path passed to warning functions
- [ ] CI mode skips with exit code 0
- [ ] Dev mode emits warning and continues
- [ ] All tests passing

---

## Section 8: Why This Works

### 8.1 Matched Path Already Captured

The detection function iterates through candidate directories checking if all required libraries exist. When found, it returns the actual directory path (not just "found=true").

### 8.2 No New APIs Needed

All existing functions in the specification are already implemented:
- Path-returning runtime detection: ✅
- CI environment detection: ✅
- Warning emission with path: ✅
- CI skip diagnostic formatting: ✅

### 8.3 Backward Compatible

Only change needed in production code is in preflight.rs, which doesn't currently call runtime detection anyway. No existing callers to break.

### 8.4 Test-Ready

The test scaffold (runtime_detection_warning_tests.rs) has 29 test cases ready, with proper environment isolation via EnvGuard and #[serial(bitnet_env)] markers.

---

## Section 9: Next Steps

### Phase 1: Integration (1-2 hours)
1. Modify xtask/src/crossval/preflight.rs (~30 lines)
2. Run existing test suite
3. Verify no regressions

### Phase 2: Testing (2-3 hours)
1. Implement test cases from runtime_detection_warning_tests.rs
2. Verify CI mode skips with correct exit code
3. Verify dev mode continues with warning
4. Test both BitNet and llama.cpp backends

### Phase 3: Documentation (1-2 hours)
1. Update CLAUDE.md with matched path info
2. Update howto/cpp-setup.md with stale build troubleshooting
3. Add environment variable reference

---

## Summary

**Current State**: Runtime detection with matched path is **fully implemented** in test helpers.

**Gap**: Production code (preflight.rs) doesn't use this information yet.

**Solution**: 30 lines of code to integrate runtime detection into preflight.rs, using matched path for diagnostics.

**Risk Level**: Low (using existing tested code, minimal changes, CI-safe)

**Effort**: 4-6 hours total (integration + testing + docs)

**Impact**: Clear diagnostic messages for stale build scenarios, CI-safe behavior, improved developer experience.

