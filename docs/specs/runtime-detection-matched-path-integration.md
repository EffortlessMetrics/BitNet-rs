# Runtime Detection Matched Path Integration Specification

**Status**: Implementation Ready
**Version**: 1.0
**Feature**: Runtime Detection with Matched Path Tracking
**Priority**: P0-2 (Critical Developer Experience)
**Scope**: `xtask/src/crossval/preflight.rs`, `tests/support/backend_helpers.rs`

---

## 1. Executive Summary

### 1.1 Problem Statement

**Current State**: Runtime detection in `backend_helpers.rs` returns `Ok((bool, Option<PathBuf>))` with matched library path, but production code in `preflight.rs` only uses build-time constants and ignores runtime detection.

**Gap**: When libraries are installed **after** xtask build (stale build scenario), production code has no awareness of runtime-detected paths, leading to:
- Generic error messages without library location context
- No matched path shown in verbose diagnostics
- Inconsistent warning output between dev/CI modes
- Missed opportunity for actionable troubleshooting guidance

**Desired State**: Integration of matched path tracking from runtime detection into preflight diagnostics, providing clear library location context in warnings and CI skip messages.

### 1.2 Key Insight: This is INTEGRATION, Not Development

**Critical Finding**: The target functionality **already exists** in `tests/support/backend_helpers.rs`:
- `detect_backend_runtime()` → `Result<(bool, Option<PathBuf>), String>` ✅ Complete
- 3-tier search path logic (BITNET_CROSSVAL_LIBDIR → RPATH → cache) ✅ Complete
- Platform-specific library name formatting ✅ Complete
- CI detection (`is_ci()`) ✅ Complete
- Warning emission functions (`emit_stale_build_warning()`, `format_ci_stale_skip_diagnostic()`) ✅ Complete

**Task**: Wire existing runtime detection into preflight.rs (~30 lines of integration code).

### 1.3 Goals

| AC | Description | Effort | Testable |
|----|-------------|--------|----------|
| **AC1** | Import `detect_backend_runtime()` into preflight.rs | 1 line | Compile check |
| **AC2** | Dev mode warning shows matched path | 5 lines | String assertion |
| **AC3** | CI mode skip shows matched path in diagnostic | 5 lines | String assertion |
| **AC4** | Verbose diagnostics include matched path + library list | 10 lines | Output validation |
| **AC5** | Preserve existing behavior where path not needed | 5 lines | Regression test |
| **AC6** | Environment variable priority honored (BITNET_CROSSVAL_LIBDIR → RPATH → cache) | 0 lines (already implemented) | Unit test |
| **AC7** | Platform-specific library extensions handled (so/dylib/dll) | 0 lines (already implemented) | Unit test |

**Total Estimate**: 4-6 hours (integration 1-2h, testing 2-3h, documentation 1-2h)

---

## 2. Technical Architecture

### 2.1 Existing Runtime Detection Flow (backend_helpers.rs)

```rust
// Location: tests/support/backend_helpers.rs:581-652
pub fn detect_backend_runtime(
    backend: CppBackend,
) -> Result<(bool, Option<std::path::PathBuf>), String> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    // Priority 1: Explicit global override
    if let Ok(p) = std::env::var("BITNET_CROSSVAL_LIBDIR") {
        candidates.push(p.into());
    }

    // Priority 2: Granular backend-specific overrides
    match backend {
        CppBackend::BitNet => {
            if let Ok(p) = std::env::var("CROSSVAL_RPATH_BITNET") {
                candidates.push(p.into());
            }
        }
        CppBackend::Llama => {
            if let Ok(p) = std::env::var("CROSSVAL_RPATH_LLAMA") {
                candidates.push(p.into());
            }
        }
    }

    // Priority 3: Backend home directory + subdirectories
    let home_var = match backend {
        CppBackend::BitNet => "BITNET_CPP_DIR",
        CppBackend::Llama => "LLAMA_CPP_DIR",
    };

    if let Ok(root) = std::env::var(home_var) {
        let root_path = std::path::Path::new(&root);
        for sub in ["build", "build/bin", "build/lib"] {
            candidates.push(root_path.join(sub));
        }
    }

    // Check each candidate and return FIRST MATCH with path
    for dir in candidates {
        if !dir.exists() { continue; }

        let all_found = needs.iter().all(|stem| {
            exts.iter().any(|ext| {
                let lib_name = format_lib_name_ext(stem, ext);
                dir.join(&lib_name).exists()
            })
        });

        if all_found {
            return Ok((true, Some(dir))); // ← Returns matched path!
        }
    }

    Ok((false, None))
}
```

**Key Properties**:
- ✅ Returns matched directory path when libraries found
- ✅ Returns `None` when no match found
- ✅ Handles platform-specific extensions (so/dylib/dll)
- ✅ Respects environment variable priority order
- ✅ Checks all required libraries for backend (libbitnet OR libllama+libggml)

### 2.2 Production Code Integration Point (preflight.rs)

**Current Implementation** (lines 549-695):
```rust
pub fn preflight_backend_libs(backend: CppBackend, verbose: bool) -> Result<()> {
    // Check build-time detection from crossval crate
    let has_libs = match backend {
        CppBackend::BitNet => HAS_BITNET,
        CppBackend::Llama => HAS_LLAMA,
    };

    if !has_libs {
        // ❌ PROBLEM: No runtime detection fallback here
        // Just prints error and bails
        bail!("Backend '{}' libraries NOT FOUND", backend.name());
    }

    // Success path...
    Ok(())
}
```

**Proposed Integration** (Priority 2 detection):
```rust
pub fn preflight_backend_libs(backend: CppBackend, verbose: bool) -> Result<()> {
    // Priority 1: Build-time detection (authoritative)
    let has_libs = match backend {
        CppBackend::BitNet => HAS_BITNET,
        CppBackend::Llama => HAS_LLAMA,
    };

    // ✅ NEW: Priority 2 runtime detection (stale build fallback)
    if !has_libs {
        if let Ok((runtime_available, matched_path)) = detect_backend_runtime(backend) {
            if runtime_available {
                // STALE BUILD SCENARIO: Libraries found at runtime but not build-time

                if is_ci() {
                    // AC3: CI mode - skip with matched path diagnostic
                    let skip_msg = format_ci_stale_skip_diagnostic(backend, matched_path.as_deref());
                    eprintln!("{}", skip_msg);
                    std::process::exit(0); // Exit code 0 = skip (not failure)
                } else {
                    // AC2: Dev mode - warn with matched path and continue
                    let verbose_flag = std::env::var("VERBOSE").is_ok();
                    if let Some(path) = matched_path {
                        emit_stale_build_warning(backend, &path, verbose_flag);
                    }
                    return Ok(()); // Continue execution
                }
            }
        }

        // No runtime detection - original error path
        bail!("Backend '{}' libraries NOT FOUND", backend.name());
    }

    // Success path (build-time available)
    if verbose {
        print_verbose_success_diagnostics(backend);
    }
    Ok(())
}
```

**Integration Cost**: ~30 lines of code (15 lines logic + 15 lines error handling)

### 2.3 Stale Build Warning Functions (Already Implemented)

**Location**: `tests/support/backend_helpers.rs:68-221`

**AC2: Standard Warning (Dev Mode)**:
```rust
// Lines 96-101
pub fn emit_standard_stale_warning(backend: CppBackend) {
    eprintln!(
        "⚠️  STALE BUILD: {} found at runtime but not at build time. Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all",
        backend.name()
    );
}
```

**AC4: Verbose Warning (Dev Mode + Matched Path)**:
```rust
// Lines 116-175
pub fn emit_verbose_stale_warning(backend: CppBackend, matched_path: &std::path::Path) {
    eprintln!("Runtime Detection Results:");
    eprintln!("  Matched path: {}", matched_path.display());

    // AC4: List libraries found in matched path
    if let Ok(entries) = std::fs::read_dir(matched_path) {
        let mut libs = Vec::new();
        for entry in entries.flatten() {
            if let Some(name) = entry.path().file_name().and_then(|n| n.to_str()) {
                if name.starts_with("lib") && name.ends_with(".so") {
                    libs.push(name.to_string());
                }
            }
        }
        if !libs.is_empty() {
            eprintln!("  Libraries found: {}", libs.join(", "));
        }
    }
}
```

**AC3: CI Skip Diagnostic**:
```rust
// Lines 189-221
pub fn format_ci_stale_skip_diagnostic(
    backend: CppBackend,
    matched_path: Option<&std::path::Path>,
) -> String {
    let mut msg = String::new();
    msg.push_str("⊘ Test skipped: {} not available (CI mode)\n");

    if let Some(path) = matched_path {
        msg.push_str(&format!("Runtime found libraries at: {}\n", path.display()));
        msg.push_str("But xtask was built before libraries were installed.\n\n");
    }

    msg.push_str("In CI mode:\n");
    msg.push_str("  • Build-time detection is the source of truth\n");
    msg.push_str("  • Runtime fallback is DISABLED for determinism\n");
    msg
}
```

### 2.4 CI Detection Logic (Already Implemented)

**Location**: `tests/support/backend_helpers.rs:230-237`

```rust
pub fn is_ci() -> bool {
    std::env::var_os("CI").is_some()
        || std::env::var_os("GITHUB_ACTIONS").is_some()
        || std::env::var_os("JENKINS_HOME").is_some()
        || std::env::var_os("GITLAB_CI").is_some()
        || std::env::var_os("CIRCLECI").is_some()
        || std::env::var_os("BITNET_TEST_NO_REPAIR").is_some()
}
```

**Detected Platforms**:
- GitHub Actions (`GITHUB_ACTIONS`, `CI`)
- GitLab CI (`GITLAB_CI`, `CI`)
- Jenkins (`JENKINS_HOME`)
- CircleCI (`CIRCLECI`)
- Manual override (`BITNET_TEST_NO_REPAIR`)

---

## 3. Acceptance Criteria (Detailed)

### AC1: Import Existing Function

**Requirement**: Make `detect_backend_runtime()` available to preflight.rs

**Implementation**:
```rust
// xtask/src/crossval/preflight.rs (add to imports)
use tests::support::backend_helpers::{
    detect_backend_runtime,
    emit_stale_build_warning,
    format_ci_stale_skip_diagnostic,
    is_ci,
};
```

**Test**: Compile check passes

### AC2: Dev Mode Warning with Path

**Requirement**: Show matched path in stale build warning (dev mode only)

**Expected Output** (standard mode):
```text
⚠️  STALE BUILD: bitnet.cpp found at runtime but not at build time.
Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all
```

**Expected Output** (verbose mode, `VERBOSE=1`):
```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  STALE BUILD DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Backend 'bitnet.cpp' found at runtime but not at xtask build time.

Runtime Detection Results:
  Matched path: /home/user/.cache/bitnet_cpp/build
  Libraries found: libbitnet.so

Build-Time Detection State:
  HAS_BITNET = false (stale)

Fix:
  cargo clean -p crossval && cargo build -p xtask --features crossval-all
```

**Test Strategy**:
```rust
#[test]
fn test_dev_mode_warning_shows_matched_path() {
    let output = capture_stderr(|| {
        emit_verbose_stale_warning(CppBackend::BitNet, Path::new("/tmp/libs"));
    });

    assert!(output.contains("Matched path: /tmp/libs"));
}
```

### AC3: CI Mode Skip with Path

**Requirement**: Show matched path in CI skip diagnostic, exit with code 0 (skip, not failure)

**Expected Output**:
```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⊘ Test skipped: bitnet.cpp not available (CI mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CI mode detected (CI=1 or BITNET_TEST_NO_REPAIR=1).
Runtime detection found libraries but build-time constants are stale.

Runtime found libraries at: /home/runner/.cache/bitnet_cpp/build
But xtask was built before libraries were installed.

In CI mode:
  • Build-time detection is the source of truth
  • Runtime fallback is DISABLED for determinism
  • xtask must be rebuilt to detect libraries

Setup Instructions:
  1. Install backend:
     eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
  2. Rebuild xtask:
     cargo clean -p crossval && cargo build -p xtask --features crossval-all
  3. Re-run CI job
```

**Test Strategy**:
```rust
#[test]
#[serial(bitnet_env)]
fn test_ci_mode_skip_shows_matched_path() {
    let _guard = EnvGuard::new("CI", "1");

    let diagnostic = format_ci_stale_skip_diagnostic(
        CppBackend::BitNet,
        Some(Path::new("/tmp/libs"))
    );

    assert!(diagnostic.contains("Runtime found libraries at: /tmp/libs"));
    assert!(diagnostic.contains("CI mode"));
    assert!(diagnostic.contains("Build-time detection is the source of truth"));
}
```

### AC4: Verbose Diagnostics

**Requirement**: Multi-line format with matched path + library listing

**Expected Elements**:
- Separator lines (visual clarity)
- Backend name and detection status
- Matched path with `path.display()`
- Library listing (filtering .so/.dylib/.dll files)
- Build-time state (HAS_BITNET/HAS_LLAMA = false)
- Fix instructions (rebuild command)

**Test Strategy**:
```rust
#[test]
fn test_verbose_warning_lists_libraries() {
    let temp_dir = tempfile::TempDir::new().unwrap();

    // Create mock libraries
    std::fs::write(temp_dir.path().join("libbitnet.so"), b"").unwrap();
    std::fs::write(temp_dir.path().join("libllama.so"), b"").unwrap();

    let output = capture_stderr(|| {
        emit_verbose_stale_warning(CppBackend::BitNet, temp_dir.path());
    });

    assert!(output.contains("libbitnet.so"));
    assert!(output.contains("Matched path:"));
}
```

### AC5: Backward Compatibility

**Requirement**: Preserve existing behavior where matched path not needed (build-time available)

**Test Strategy**:
```rust
#[test]
fn test_build_time_available_skips_runtime_detection() {
    // Simulate HAS_BITNET = true
    let result = preflight_backend_libs(CppBackend::BitNet, false);

    // Should succeed immediately without calling detect_backend_runtime
    assert!(result.is_ok());
}
```

### AC6: Environment Variable Priority

**Requirement**: Honor 3-tier search path priority (already implemented, validate)

**Priority Order**:
1. `BITNET_CROSSVAL_LIBDIR` (global override)
2. `CROSSVAL_RPATH_BITNET` / `CROSSVAL_RPATH_LLAMA` (granular)
3. `BITNET_CPP_DIR/build`, `LLAMA_CPP_DIR/build` (standard)

**Test Strategy**:
```rust
#[test]
#[serial(bitnet_env)]
fn test_env_priority_global_override_wins() {
    let temp_global = tempfile::TempDir::new().unwrap();
    let temp_rpath = tempfile::TempDir::new().unwrap();

    // Create libs in both locations
    create_mock_backend_libs(&temp_global, CppBackend::BitNet).unwrap();
    create_mock_backend_libs(&temp_rpath, CppBackend::BitNet).unwrap();

    let _guard1 = EnvGuard::new("BITNET_CROSSVAL_LIBDIR", temp_global.path().to_str().unwrap());
    let _guard2 = EnvGuard::new("CROSSVAL_RPATH_BITNET", temp_rpath.path().to_str().unwrap());

    let (found, path) = detect_backend_runtime(CppBackend::BitNet).unwrap();
    assert!(found);
    assert_eq!(path.unwrap(), temp_global.path());  // Global wins
}
```

### AC7: Platform-Specific Library Extensions

**Requirement**: Handle .so (Linux), .dylib (macOS), .dll (Windows) (already implemented, validate)

**Test Strategy**:
```rust
#[test]
fn test_platform_specific_extensions() {
    #[cfg(target_os = "linux")]
    assert_eq!(format_lib_name("bitnet"), "libbitnet.so");

    #[cfg(target_os = "macos")]
    assert_eq!(format_lib_name("bitnet"), "libbitnet.dylib");

    #[cfg(target_os = "windows")]
    assert_eq!(format_lib_name("bitnet"), "bitnet.dll");
}
```

---

## 4. Implementation Plan

### Phase 1: Integration (1-2 hours, ~30 lines)

**Tasks**:
1. Add imports to `xtask/src/crossval/preflight.rs`
2. Add Priority 2 runtime detection block in `preflight_backend_libs()`
3. Wire matched path to warning emission functions
4. Add CI branching (skip vs warning)
5. Verify compile passes

**Deliverables**:
- Modified `preflight.rs` with runtime detection integration
- Build passes: `cargo build -p xtask --features crossval-all`

### Phase 2: Testing (2-3 hours, 29 test cases)

**Test Scaffolding** (already exists in `tests/support/runtime_detection_warning_tests.rs`):
- 29 test functions with scaffolding
- EnvGuard for environment isolation
- `#[serial(bitnet_env)]` markers for safety
- Mock library creation helpers

**Test Implementation Tasks**:
1. **AC1 Tests** (1 test): Compile check
2. **AC2 Tests** (6 tests): Dev mode warnings
   - Standard warning format
   - Verbose warning format
   - Matched path display
   - Library listing
   - Deduplication (Once flag)
   - VERBOSE env var handling
3. **AC3 Tests** (4 tests): CI mode skip
   - Skip diagnostic format
   - Matched path inclusion
   - Exit code 0 verification
   - CI detection accuracy
4. **AC4 Tests** (3 tests): Verbose diagnostics
   - Multi-line format
   - Library listing
   - Build-time state display
5. **AC5 Tests** (2 tests): Backward compatibility
   - Build-time available path
   - No runtime detection called
6. **AC6 Tests** (3 tests): Environment priority
   - Global override priority
   - Granular override priority
   - Standard directory fallback
7. **AC7 Tests** (3 tests): Platform extensions
   - Linux .so
   - macOS .dylib
   - Windows .dll (conditional)

**Test Execution**:
```bash
# Run new tests
cargo test -p tests --test runtime_detection_warning_tests \
  --no-default-features --features cpu -- --nocapture

# Run with nextest (recommended)
cargo nextest run -p tests --test runtime_detection_warning_tests \
  --no-default-features --features cpu
```

### Phase 3: Documentation (1-2 hours)

**Updates Required**:
1. **CLAUDE.md**: Add runtime detection + matched path section
2. **docs/howto/cpp-setup.md**: Add stale build troubleshooting with matched path examples
3. **docs/environment-variables.md**: Document BITNET_CROSSVAL_LIBDIR, CROSSVAL_RPATH_*
4. **Inline documentation**: Update function docstrings in preflight.rs

**Documentation Deliverables**:
- Clear explanation of stale build scenario
- Examples showing matched path in warnings
- CI vs dev mode behavior differences
- Environment variable priority order
- Troubleshooting guide with matched path diagnostics

---

## 5. Test Strategy

### 5.1 Unit Tests (Pure Logic)

**Goal**: Test individual functions in isolation

**Coverage**:
- `format_ci_stale_skip_diagnostic()` - string formatting
- `emit_standard_stale_warning()` - warning format
- `emit_verbose_stale_warning()` - verbose format
- `detect_backend_runtime()` - matched path return value
- `is_ci()` - environment detection

**Example**:
```rust
#[test]
fn test_format_ci_skip_includes_matched_path() {
    let diagnostic = format_ci_stale_skip_diagnostic(
        CppBackend::BitNet,
        Some(Path::new("/tmp/libs"))
    );

    assert!(diagnostic.contains("Runtime found libraries at: /tmp/libs"));
    assert!(diagnostic.contains("CI mode"));
}
```

### 5.2 Integration Tests (End-to-End)

**Goal**: Test full preflight flow with runtime detection

**Coverage**:
- Build-time available → no runtime detection called
- Build-time unavailable + runtime available + dev mode → warning + continue
- Build-time unavailable + runtime available + CI mode → skip + exit 0
- Build-time unavailable + runtime unavailable → error

**Example**:
```rust
#[test]
#[serial(bitnet_env)]
fn test_stale_build_dev_mode_continues() {
    let temp = create_mock_backend_libs(CppBackend::BitNet).unwrap();
    let _guard = EnvGuard::new("BITNET_CROSSVAL_LIBDIR", temp.path().to_str().unwrap());

    // Simulate HAS_BITNET = false (stale)
    // Runtime detection should find libs and emit warning

    let result = preflight_backend_libs(CppBackend::BitNet, true);
    assert!(result.is_ok());  // Should continue in dev mode
}
```

### 5.3 Edge Cases

**Coverage**:
- Permission errors reading directory
- Empty RPATH variables
- Missing subdirectories
- Corrupted library files
- Symlink resolution
- Network filesystem delays

**Example**:
```rust
#[test]
#[serial(bitnet_env)]
fn test_permission_denied_graceful_fallback() {
    let temp = tempfile::TempDir::new().unwrap();

    // Remove read permissions
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(temp.path()).unwrap().permissions();
        perms.set_mode(0o000);
        std::fs::set_permissions(temp.path(), perms).unwrap();
    }

    let _guard = EnvGuard::new("BITNET_CROSSVAL_LIBDIR", temp.path().to_str().unwrap());

    let (found, path) = detect_backend_runtime(CppBackend::BitNet).unwrap();
    assert!(!found);  // Should gracefully skip unreadable directory
    assert!(path.is_none());
}
```

---

## 6. Environment Variables Reference

| Variable | Purpose | Priority | Example |
|----------|---------|----------|---------|
| `BITNET_CROSSVAL_LIBDIR` | Global library directory override | 1 (highest) | `/custom/libs` |
| `CROSSVAL_RPATH_BITNET` | BitNet-specific library path (colon-separated) | 2 | `/opt/bitnet/lib:/usr/local/lib` |
| `CROSSVAL_RPATH_LLAMA` | Llama-specific library path (colon-separated) | 2 | `/opt/llama/lib` |
| `BITNET_CPP_DIR` | BitNet installation root (searches subdirs) | 3 | `~/.cache/bitnet_cpp` |
| `LLAMA_CPP_DIR` | Llama installation root (searches subdirs) | 3 | `~/.cache/llama_cpp` |
| `VERBOSE` | Enable verbose diagnostic output | N/A | `1` |
| `CI` | Detect CI environment (GitHub Actions, etc.) | N/A | `true` |
| `BITNET_TEST_NO_REPAIR` | Manual CI mode override | N/A | `1` |

**Subdirectories Searched** (Priority 3):
- `{BITNET_CPP_DIR}/build`
- `{BITNET_CPP_DIR}/build/bin`
- `{BITNET_CPP_DIR}/build/lib`

**Platform-Specific Library Names**:
- Linux: `libbitnet.so`, `libllama.so`, `libggml.so`
- macOS: `libbitnet.dylib`, `libllama.dylib`, `libggml.dylib`
- Windows: `bitnet.dll`, `llama.dll`, `ggml.dll`

---

## 7. Dev vs CI Semantics

### 7.1 Dev Mode (Interactive Environment)

**Detected When**: `is_ci()` returns `false`

**Behavior**:
1. Runtime detection finds libraries ✓
2. Build-time constant is false (stale) ✓
3. **Action**: Emit warning and **continue**
4. **Exit Code**: Test proceeds normally
5. **Message**: Single-line warning (standard) or multi-line (verbose)

**Rationale**: Developer convenience - don't block workflow, just inform about stale build

**Example**:
```bash
$ cargo run -p xtask -- preflight --backend bitnet --verbose
⚠️  STALE BUILD: bitnet.cpp found at runtime but not at build time.
Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all

# Test continues...
✓ Backend 'bitnet.cpp' libraries found
```

### 7.2 CI Mode (Automated Environment)

**Detected When**: `is_ci()` returns `true` (any of):
- `GITHUB_ACTIONS` set
- `GITLAB_CI` set
- `JENKINS_HOME` set
- `CIRCLECI` set
- `CI` set
- `BITNET_TEST_NO_REPAIR` set

**Behavior**:
1. Runtime detection finds libraries ✓
2. Build-time constant is false (stale) ✓
3. **Action**: Skip test immediately
4. **Exit Code**: 0 (success/skip, **not** failure)
5. **Message**: Multi-line CI diagnostic with setup instructions

**Rationale**: Deterministic CI behavior - build-time detection is the source of truth

**Example**:
```bash
$ CI=1 cargo run -p xtask -- preflight --backend bitnet --verbose
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⊘ Test skipped: bitnet.cpp not available (CI mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CI mode detected (CI=1 or BITNET_TEST_NO_REPAIR=1).
Runtime detection found libraries but build-time constants are stale.

Runtime found libraries at: /home/runner/.cache/bitnet_cpp/build

In CI mode:
  • Build-time detection is the source of truth
  • Runtime fallback is DISABLED for determinism

# Exit code 0
```

---

## 8. Success Criteria

### 8.1 Functional Requirements

- ✅ AC1: `detect_backend_runtime()` imported and callable from preflight.rs
- ✅ AC2: Dev mode warning displays matched path (standard + verbose)
- ✅ AC3: CI mode skip displays matched path in diagnostic
- ✅ AC4: Verbose diagnostics include matched path + library listing
- ✅ AC5: Build-time available path preserves original behavior (no runtime detection)
- ✅ AC6: Environment variable priority honored (3-tier search)
- ✅ AC7: Platform-specific library extensions handled correctly

### 8.2 Quality Requirements

- ✅ All 29 test cases pass (unit + integration + edge cases)
- ✅ EnvGuard prevents test pollution across parallel runs
- ✅ `#[serial(bitnet_env)]` markers prevent race conditions
- ✅ No regressions in existing preflight behavior
- ✅ Matched path display is UTF-8 safe (`path.display()`)
- ✅ CI detection is accurate (6 environment variables checked)

### 8.3 Documentation Requirements

- ✅ CLAUDE.md updated with runtime detection section
- ✅ cpp-setup.md includes stale build troubleshooting
- ✅ environment-variables.md documents all search variables
- ✅ Inline docstrings updated in preflight.rs

---

## 9. Risk Assessment

### 9.1 Low Risk (Integration)

**Reason**: Target functionality already exists and is tested in `backend_helpers.rs`

**Mitigation**: This is pure integration work - no new algorithms or complex logic

### 9.2 Potential Issues

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Import path resolution | Low | Low | Use `tests::support::backend_helpers::` explicit path |
| CI detection false negatives | Medium | Medium | Test all 6 CI environment variables |
| Path display encoding | Low | Medium | Always use `path.display()` for UTF-8 safety |
| Test environment pollution | Low | High | Use EnvGuard + `#[serial(bitnet_env)]` |
| Backward compatibility break | Low | High | Test build-time available path separately |

### 9.3 Rollback Plan

If integration causes issues:
1. Revert preflight.rs changes (restore original Priority 1 only)
2. Keep runtime detection in backend_helpers.rs (no changes needed)
3. Re-test with original behavior
4. Address issues incrementally with feature flags

---

## 10. Implementation Checklist

- [ ] **Phase 1: Integration** (1-2 hours)
  - [ ] Add imports to preflight.rs
  - [ ] Add Priority 2 runtime detection block
  - [ ] Wire matched path to warning functions
  - [ ] Add CI branching logic
  - [ ] Verify `cargo build -p xtask --features crossval-all` passes

- [ ] **Phase 2: Testing** (2-3 hours)
  - [ ] Implement AC1 test (compile check)
  - [ ] Implement AC2 tests (6 dev mode warning tests)
  - [ ] Implement AC3 tests (4 CI mode skip tests)
  - [ ] Implement AC4 tests (3 verbose diagnostic tests)
  - [ ] Implement AC5 tests (2 backward compatibility tests)
  - [ ] Implement AC6 tests (3 environment priority tests)
  - [ ] Implement AC7 tests (3 platform extension tests)
  - [ ] Verify all tests pass: `cargo nextest run --test runtime_detection_warning_tests`

- [ ] **Phase 3: Documentation** (1-2 hours)
  - [ ] Update CLAUDE.md with runtime detection section
  - [ ] Update cpp-setup.md with stale build troubleshooting
  - [ ] Update environment-variables.md with search path variables
  - [ ] Update inline docstrings in preflight.rs

- [ ] **Phase 4: Validation**
  - [ ] Run full test suite: `cargo test --workspace --no-default-features --features cpu`
  - [ ] Run nextest CI profile: `cargo nextest run --profile ci`
  - [ ] Manual test: Simulate stale build scenario in dev mode
  - [ ] Manual test: Simulate stale build scenario in CI mode
  - [ ] Verify no regressions in preflight behavior

---

## 11. Examples

### Example 1: Dev Mode Warning (Standard)

**Scenario**: Developer installed bitnet.cpp after building xtask

**Command**:
```bash
$ cargo run -p xtask -- preflight --backend bitnet
```

**Output**:
```text
⚠️  STALE BUILD: bitnet.cpp found at runtime but not at build time.
Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all

✓ Backend 'bitnet.cpp' libraries found
```

**Exit Code**: 0 (continues normally)

### Example 2: Dev Mode Warning (Verbose)

**Scenario**: Developer wants detailed diagnostics

**Command**:
```bash
$ VERBOSE=1 cargo run -p xtask -- preflight --backend bitnet --verbose
```

**Output**:
```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  STALE BUILD DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Backend 'bitnet.cpp' found at runtime but not at xtask build time.

This happens when:
  1. You built xtask
  2. Then installed bitnet.cpp libraries later
  3. xtask binary still contains old detection constants

Runtime Detection Results:
  Matched path: /home/user/.cache/bitnet_cpp/build
  Libraries found: libbitnet.so

Build-Time Detection State:
  HAS_BITNET = false (stale)

Fix:
  cargo clean -p crossval && cargo build -p xtask --features crossval-all

Then re-run your test.
```

**Exit Code**: 0 (continues normally)

### Example 3: CI Mode Skip

**Scenario**: CI job runs before xtask rebuild

**Command**:
```bash
$ CI=1 cargo run -p xtask -- preflight --backend bitnet --verbose
```

**Output**:
```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⊘ Test skipped: bitnet.cpp not available (CI mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CI mode detected (CI=1 or BITNET_TEST_NO_REPAIR=1).
Runtime detection found libraries but build-time constants are stale.

Runtime found libraries at: /home/runner/.cache/bitnet_cpp/build
But xtask was built before libraries were installed.

In CI mode:
  • Build-time detection is the source of truth
  • Runtime fallback is DISABLED for determinism
  • xtask must be rebuilt to detect libraries

Setup Instructions:
  1. Install backend:
     eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
  2. Rebuild xtask:
     cargo clean -p crossval && cargo build -p xtask --features crossval-all
  3. Re-run CI job
```

**Exit Code**: 0 (skip, not failure)

---

## 12. Appendix: Existing Code References

### A. Runtime Detection Implementation

**File**: `tests/support/backend_helpers.rs`
**Lines**: 581-652
**Function**: `detect_backend_runtime()`
**Status**: ✅ Complete, tested, production-ready

### B. Warning Emission Functions

**File**: `tests/support/backend_helpers.rs`
**Lines**: 68-221
**Functions**:
- `emit_standard_stale_warning()` - Line 96
- `emit_verbose_stale_warning()` - Line 116
- `format_ci_stale_skip_diagnostic()` - Line 189
- `is_ci()` - Line 230

**Status**: ✅ Complete, tested, production-ready

### C. Test Scaffolding

**File**: `tests/support/runtime_detection_warning_tests.rs`
**Test Count**: 29 functions
**Coverage**: AC1-AC7 with environment isolation
**Status**: Scaffolded, ready for implementation

### D. Production Integration Point

**File**: `xtask/src/crossval/preflight.rs`
**Function**: `preflight_backend_libs()` (line 549)
**Required Changes**: ~30 lines (Priority 2 detection block)
**Status**: Integration needed

---

## 13. Conclusion

This specification defines a **low-risk integration task** that wires existing, tested runtime detection functionality into production preflight code. The matched path information is already captured by `detect_backend_runtime()` - we simply need to:

1. Call the function from preflight.rs (5 lines)
2. Pass matched path to warning functions (10 lines)
3. Branch on CI detection (15 lines)

**Total Implementation**: ~30 lines of integration code + 29 test case implementations + documentation updates.

**Key Success Factor**: All required functions already exist and work correctly. This is pure integration work with clear acceptance criteria and comprehensive test coverage.
