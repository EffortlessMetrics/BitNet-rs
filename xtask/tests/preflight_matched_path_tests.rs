//! Integration tests for preflight matched path integration
//!
//! **Tests specification**: `docs/specs/runtime-detection-matched-path-integration.md`
//!
//! # Test Coverage
//!
//! This test suite validates the integration of runtime detection with matched path
//! tracking into preflight.rs. Tests verify that when libraries are installed AFTER
//! xtask build (stale build scenario), the production preflight code:
//!
//! 1. Calls runtime detection as fallback (Priority 2)
//! 2. Emits warnings with matched library paths in dev mode
//! 3. Skips with diagnostics showing matched paths in CI mode
//! 4. Preserves backward compatibility (Priority 1 still fast path)
//!
//! ## Acceptance Criteria Coverage
//!
//! - AC1: Import detect_backend_runtime() into preflight.rs → Compile check
//! - AC2: Dev mode warning shows matched path → String assertion
//! - AC3: CI mode skip shows matched path in diagnostic → String assertion
//! - AC4: Verbose diagnostics include matched path + library list → Output validation
//! - AC5: Preserve existing behavior where path not needed → Regression test
//! - AC6: Environment variable priority honored → Unit test (reuse backend_helpers tests)
//! - AC7: Platform-specific library extensions → Unit test (reuse backend_helpers tests)
//!
//! # Test Organization
//!
//! - **Category A**: Warning Format with Matched Path (8 tests) → AC2, AC4
//! - **Category B**: Preflight Integration with Runtime Detection (12 tests) → AC1, AC2, AC3, AC5
//! - **Category C**: Edge Cases for Integration (9 tests)
//!
//! **Total Tests**: 29 tests (many reuse existing backend_helpers test infrastructure)
//!
//! # Test Status: RED Phase (TDD)
//!
//! These tests are scaffolded to FAIL initially (marked with `todo!()`). They validate
//! the planned integration of matched path tracking from runtime detection into
//! production preflight code.
//!
//! # Note: Function Already Exists
//!
//! The `detect_backend_runtime()` function is ALREADY IMPLEMENTED in
//! `tests/support/backend_helpers.rs` and returns `Result<(bool, Option<PathBuf>), String>`.
//! These tests verify the **integration** of this existing function into preflight.rs.

#[allow(unused_imports)]
use bitnet_crossval::backend::CppBackend;

// ============================================================================
// Category A: Warning Format Tests with Matched Path (AC2, AC4)
// ============================================================================

/// Tests spec: runtime-detection-matched-path-integration.md#AC2
/// Validates: emit_stale_build_warning includes matched path in standard mode
#[test]
#[cfg(all(test, feature = "crossval-all"))]
fn test_emit_stale_warning_includes_matched_path() {
    // TODO: AC2 - Verify standard warning format includes:
    // 1. "⚠️  STALE BUILD: {backend} found at runtime but not at build time"
    // 2. Rebuild command: "cargo clean -p crossval && cargo build -p xtask --features crossval-all"
    // 3. Does NOT include matched path (that's verbose mode only)
    //
    // Expected behavior:
    // - Standard warning is concise (one line)
    // - Matched path shown only in verbose mode
    // - Use tests::support::backend_helpers::emit_standard_stale_warning
    todo!("AC2: Verify standard warning emitted (without matched path - that's verbose mode)");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC4
/// Validates: emit_verbose_stale_warning shows matched path and library listing
#[test]
#[cfg(all(test, feature = "crossval-all"))]
fn test_emit_verbose_warning_shows_library_listing() {
    // TODO: AC4 - Create temp directory with mock libraries and verify:
    // 1. Verbose warning contains separator lines
    // 2. Contains "⚠️  STALE BUILD DETECTION" header
    // 3. Contains "Matched path: {path}"
    // 4. Contains "Libraries found: libbitnet.so" (or platform equivalent)
    // 5. Contains explanation sections:
    //    - "This happens when:"
    //    - "Why rebuild is needed:"
    //    - "Runtime Detection Results:"
    //    - "Build-Time Detection State:"
    //    - "Fix:"
    //
    // Use tests::support::backend_helpers::emit_verbose_stale_warning with temp path
    // Capture stderr to validate output format
    todo!("AC4: Verify verbose warning shows matched path and lists libraries");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC3
/// Validates: format_ci_stale_skip_diagnostic includes matched path in CI skip
#[test]
#[cfg(all(test, feature = "crossval-all"))]
fn test_emit_ci_skip_diagnostic_shows_matched_path() {
    // TODO: AC3 - Call format_ci_stale_skip_diagnostic with matched path and verify:
    // 1. Contains "⊘ Test skipped: {backend} not available (CI mode)"
    // 2. Contains "Runtime found libraries at: {path}"
    // 3. Contains "But xtask was built before libraries were installed"
    // 4. Contains setup instructions (3 steps)
    // 5. Contains CI mode explanation
    //
    // Expected:
    // - Multi-line diagnostic format
    // - Matched path prominently displayed
    // - Clear actionable instructions
    //
    // Use tests::support::backend_helpers::format_ci_stale_skip_diagnostic
    todo!("AC3: Verify CI skip diagnostic shows matched path with setup instructions");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC2, AC4
/// Validates: Warning format differentiates between BitNet and Llama backends
#[test]
#[cfg(all(test, feature = "crossval-all"))]
fn test_warning_format_bitnet_backend() {
    // TODO: AC2 - Verify BitNet-specific warning format:
    // 1. Backend name "bitnet.cpp" in warning
    // 2. HAS_BITNET constant mentioned in verbose mode
    // 3. Correct library names (libbitnet.so on Linux)
    //
    // Use CppBackend::BitNet parameter
    todo!("AC2: Verify BitNet backend name in warning messages");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC2, AC4
/// Validates: Warning format differentiates between BitNet and Llama backends
#[test]
#[cfg(all(test, feature = "crossval-all"))]
fn test_warning_format_llama_backend() {
    // TODO: AC2 - Verify Llama-specific warning format:
    // 1. Backend name "llama.cpp" in warning
    // 2. HAS_LLAMA constant mentioned in verbose mode
    // 3. Correct library names (libllama.so, libggml.so on Linux)
    //
    // Use CppBackend::Llama parameter
    todo!("AC2: Verify Llama backend name in warning messages");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC4
/// Validates: Verbose warning multi-line format with all sections
#[test]
#[cfg(all(test, feature = "crossval-all"))]
fn test_verbose_warning_multi_line_format() {
    // TODO: AC4 - Verify verbose warning structure:
    // 1. Separator lines (━━━━━━...)
    // 2. Header section
    // 3. Timeline section ("This happens when:")
    // 4. Rationale section ("Why rebuild is needed:")
    // 5. Runtime results section (matched path, libraries)
    // 6. Build-time state section
    // 7. Fix section (rebuild command)
    //
    // Use tempfile::tempdir() to create mock library directory
    // Call emit_verbose_stale_warning and capture stderr
    todo!("AC4: Verify verbose warning has complete multi-line structure");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC3
/// Validates: CI diagnostic includes detailed setup instructions
#[test]
#[cfg(all(test, feature = "crossval-all"))]
fn test_ci_diagnostic_includes_setup_instructions() {
    // TODO: AC3 - Verify CI diagnostic setup instructions:
    // 1. Step 1: "eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\""
    // 2. Step 2: "cargo clean -p crossval && cargo build -p xtask --features crossval-all"
    // 3. Step 3: "Re-run CI job"
    // 4. Explanation of CI determinism requirement
    //
    // Call format_ci_stale_skip_diagnostic and validate string content
    todo!("AC3: Verify CI diagnostic has complete setup instructions");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC2, AC4
/// Validates: Warning deduplication per backend using std::sync::Once
#[test]
#[cfg(all(test, feature = "crossval-all"))]
fn test_warning_deduplication_per_backend() {
    // TODO: AC2 - Verify warning deduplication:
    // 1. Call emit_stale_build_warning(BitNet) twice
    // 2. First call: warning emitted
    // 3. Second call: no output (deduplicated via Once)
    // 4. Call emit_stale_build_warning(Llama) once
    // 5. Llama warning emitted (separate Once guard)
    //
    // Expected:
    // - Per-backend deduplication using static Once flags
    // - BitNet and Llama have separate Once guards
    //
    // Note: This test validates the Once usage in backend_helpers.rs
    // The implementation already exists in emit_stale_build_warning()
    todo!("AC2: Verify warning deduplication per backend using std::sync::Once");
}

// ============================================================================
// Category B: Preflight Integration Tests (AC1, AC2, AC3, AC5)
// ============================================================================

/// Tests spec: runtime-detection-matched-path-integration.md#AC1, AC2
/// Validates: Preflight dev mode emits warning with matched path
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
fn test_preflight_dev_mode_emits_warning_with_path() {
    // TODO: AC1, AC2 - Integration test for dev mode:
    // 1. Create temp directory with mock BitNet libraries
    // 2. Set BITNET_CPP_DIR to temp path
    // 3. Clear CI environment variables (ensure dev mode)
    // 4. Call preflight_backend_libs(CppBackend::BitNet, false)
    //    (assuming HAS_BITNET=false for stale build scenario)
    // 5. Verify:
    //    - Function returns Ok(())
    //    - Warning emitted to stderr
    //    - Warning contains backend name
    //    - Test continues (does not exit or panic)
    //
    // Expected behavior:
    // - Dev mode allows test to proceed with warning
    // - Runtime detection fallback kicks in when build-time unavailable
    //
    // Use tests::support::env_guard::EnvGuard for environment isolation
    // Use tests::support::backend_helpers::create_mock_backend_libs for mock libraries
    todo!("AC1, AC2: Verify preflight dev mode emits warning and continues with matched path");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC1, AC3
/// Validates: Preflight CI mode skips with diagnostic showing matched path
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
#[ignore] // Requires subprocess execution to capture exit code
fn test_preflight_ci_mode_skips_with_diagnostic() {
    // TODO: AC1, AC3 - Integration test for CI mode:
    // 1. Create temp directory with mock BitNet libraries
    // 2. Set BITNET_CPP_DIR to temp path
    // 3. Set CI=1 (enable CI mode)
    // 4. Run preflight_backend_libs as subprocess (to capture exit code)
    // 5. Verify:
    //    - Exit code 0 (skip, not failure)
    //    - Stderr contains CI skip diagnostic
    //    - Diagnostic shows matched library path
    //    - Diagnostic includes setup instructions
    //
    // Expected behavior:
    // - CI mode skips test immediately with exit(0)
    // - No runtime override allowed in CI (build-time is source of truth)
    //
    // This test requires subprocess execution - marked #[ignore]
    // Run manually with: cargo test --ignored
    todo!(
        "AC1, AC3: Verify preflight CI mode skips with matched path diagnostic (requires subprocess)"
    );
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC1, AC4
/// Validates: Preflight verbose mode shows matched path in diagnostics
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
fn test_preflight_verbose_shows_matched_path() {
    // TODO: AC1, AC4 - Verify verbose diagnostics integration:
    // 1. Create temp directory with mock libraries
    // 2. Set BITNET_CPP_DIR and VERBOSE=1
    // 3. Clear CI variables (dev mode)
    // 4. Call preflight_backend_libs(CppBackend::BitNet, true)
    // 5. Capture stderr and verify:
    //    - Contains "⚠️  STALE BUILD DETECTION"
    //    - Contains "Matched path: {temp_path}"
    //    - Contains library listing
    //    - Contains rebuild instructions
    //    - Function returns Ok(())
    //
    // Use tests::support::backend_helpers::emit_verbose_stale_warning
    todo!("AC1, AC4: Verify preflight verbose mode shows matched path and library listing");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC1, AC5
/// Validates: Preflight no warning when build-time detection succeeds
#[test]
#[cfg(all(test, feature = "crossval-all"))]
fn test_preflight_no_warning_when_build_time_available() {
    // TODO: AC1, AC5 - Verify backward compatibility (Priority 1 fast path):
    // 1. When HAS_BITNET=true at compile time:
    //    - preflight_backend_libs returns Ok(()) immediately
    //    - No runtime detection called
    //    - No warning emitted
    //    - Fast path performance maintained
    //
    // Expected:
    // - Build-time detection (Priority 1) takes precedence
    // - Runtime detection (Priority 2) only used when build-time fails
    //
    // Note: This test validates existing behavior preservation
    // Cannot test HAS_BITNET=true without recompiling crossval crate
    // Validated through code review and regression testing
    todo!(
        "AC1, AC5: Verify build-time available path preserves existing behavior (no runtime check)"
    );
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC1, AC6
/// Validates: Matched path from BITNET_CROSSVAL_LIBDIR (Priority 1 env)
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
fn test_preflight_matched_path_in_crossval_libdir() {
    // TODO: AC1, AC6 - Verify environment variable priority:
    // 1. Create temp directory with mock libraries
    // 2. Set BITNET_CROSSVAL_LIBDIR to temp path (Priority 1)
    // 3. Set different CROSSVAL_RPATH_BITNET (Priority 2 - should be ignored)
    // 4. Call detect_backend_runtime(CppBackend::BitNet)
    // 5. Verify matched path equals BITNET_CROSSVAL_LIBDIR
    //
    // Expected:
    // - BITNET_CROSSVAL_LIBDIR (Priority 1) wins over granular overrides
    // - Matched path returned matches Priority 1 env var
    //
    // Use tests::support::backend_helpers::detect_backend_runtime
    // Use tests::support::env_guard::EnvGuard for isolation
    todo!("AC1, AC6: Verify matched path from BITNET_CROSSVAL_LIBDIR (highest priority)");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC1, AC6
/// Validates: Matched path from CROSSVAL_RPATH_BITNET (Priority 2 env)
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
fn test_preflight_matched_path_in_rpath() {
    // TODO: AC1, AC6 - Verify granular RPATH override:
    // 1. Create temp directory with mock libraries
    // 2. Clear BITNET_CROSSVAL_LIBDIR (no Priority 1)
    // 3. Set CROSSVAL_RPATH_BITNET to temp path (Priority 2)
    // 4. Call detect_backend_runtime(CppBackend::BitNet)
    // 5. Verify matched path equals CROSSVAL_RPATH_BITNET
    //
    // Expected:
    // - CROSSVAL_RPATH_BITNET (Priority 2) used when Priority 1 unavailable
    // - Matched path returned matches Priority 2 env var
    //
    // Use tests::support::backend_helpers::detect_backend_runtime
    todo!("AC1, AC6: Verify matched path from CROSSVAL_RPATH_BITNET (Priority 2)");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC1, AC6
/// Validates: Matched path from BITNET_CPP_DIR/build subdirectory (Priority 3)
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
fn test_preflight_matched_path_in_cpp_dir_subdir() {
    // TODO: AC1, AC6 - Verify standard directory fallback:
    // 1. Create temp directory structure: {root}/build/{libs}
    // 2. Clear BITNET_CROSSVAL_LIBDIR and CROSSVAL_RPATH_BITNET
    // 3. Set BITNET_CPP_DIR to temp root
    // 4. Call detect_backend_runtime(CppBackend::BitNet)
    // 5. Verify matched path equals {root}/build
    //
    // Expected:
    // - BITNET_CPP_DIR/build (Priority 3) used when Priority 1 & 2 unavailable
    // - Subdirectory search: build, build/bin, build/lib
    // - First match returned
    //
    // Use tests::support::backend_helpers::detect_backend_runtime
    todo!("AC1, AC6: Verify matched path from BITNET_CPP_DIR/build subdirectory (Priority 3)");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC1, AC2
/// Validates: Multiple backends have separate warnings (BitNet + Llama)
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
fn test_preflight_multiple_backends_separate_warnings() {
    // TODO: AC1, AC2 - Verify per-backend warning deduplication:
    // 1. Create temp directories for both BitNet and Llama libraries
    // 2. Set environment variables for both backends
    // 3. Call ensure_backend_or_skip(CppBackend::BitNet) - first warning
    // 4. Call ensure_backend_or_skip(CppBackend::BitNet) - deduplicated
    // 5. Call ensure_backend_or_skip(CppBackend::Llama) - second warning
    // 6. Verify:
    //    - BitNet warning emitted once
    //    - Llama warning emitted once
    //    - Separate std::sync::Once guards per backend
    //
    // Use tests::support::backend_helpers::ensure_backend_or_skip
    todo!("AC1, AC2: Verify multiple backends have separate warning deduplication");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC1, AC2
/// Validates: Warning deduplication across multiple calls (std::sync::Once)
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
fn test_preflight_deduplication_per_backend() {
    // TODO: AC1, AC2 - Verify Once-based deduplication:
    // 1. Create temp directory with mock libraries
    // 2. Set BITNET_CPP_DIR
    // 3. Call ensure_backend_or_skip(CppBackend::BitNet) 3 times
    // 4. Capture stderr for all calls
    // 5. Verify:
    //    - First call: warning present
    //    - Second call: no warning (deduplicated)
    //    - Third call: no warning (deduplicated)
    //
    // Expected:
    // - std::sync::Once ensures warning printed exactly once
    // - Subsequent calls are no-ops
    //
    // Use tests::support::backend_helpers::ensure_backend_or_skip
    todo!("AC1, AC2: Verify warning deduplication using std::sync::Once");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC1, AC5
/// Validates: Backward compatibility - no path needed when build-time succeeds
#[test]
#[cfg(all(test, feature = "crossval-all"))]
fn test_preflight_backward_compat_no_path_needed() {
    // TODO: AC1, AC5 - Verify backward compatibility:
    // 1. When HAS_BITNET=true (build-time detection succeeded):
    //    - preflight_backend_libs returns Ok(()) immediately
    //    - No matched path lookup needed
    //    - No environment variable checks
    //    - Fast path maintained
    //
    // Expected:
    // - Existing behavior preserved (no regression)
    // - Runtime detection only used as fallback
    //
    // Note: Cannot test without recompiling crossval with HAS_BITNET=true
    // Validated through code review and existing test suite
    todo!(
        "AC1, AC5: Verify backward compatibility - no matched path needed when build-time succeeds"
    );
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC1, AC2, AC3
/// Validates: Preflight returns Ok on runtime override in dev mode
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
fn test_preflight_returns_ok_on_runtime_override_dev() {
    // TODO: AC1, AC2, AC3 - Verify dev mode runtime override:
    // 1. Create temp directory with mock libraries (stale build simulation)
    // 2. Set BITNET_CPP_DIR to temp path
    // 3. Clear CI variables (ensure dev mode)
    // 4. Call ensure_backend_or_skip(CppBackend::BitNet)
    //    (assuming HAS_BITNET=false)
    // 5. Verify:
    //    - Function returns (does not panic)
    //    - Warning emitted to stderr
    //    - Test continues execution
    //
    // Expected:
    // - Dev mode allows runtime override with warning
    // - Test proceeds even though build-time detection failed
    //
    // Use tests::support::backend_helpers::ensure_backend_or_skip
    todo!("AC1, AC2, AC3: Verify preflight returns Ok on runtime override in dev mode");
}

/// Tests spec: runtime-detection-matched-path-integration.md#AC1, AC3
/// Validates: Preflight exits on runtime override in CI mode
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
#[ignore] // Requires subprocess to capture exit code
fn test_preflight_exits_on_runtime_override_ci() {
    // TODO: AC1, AC3 - Verify CI mode exit behavior:
    // 1. Create temp directory with mock libraries (stale build simulation)
    // 2. Set BITNET_CPP_DIR and CI=1
    // 3. Run ensure_backend_or_skip as subprocess
    // 4. Verify:
    //    - Exit code 0 (skip, not failure)
    //    - Stderr contains CI skip diagnostic
    //    - Diagnostic shows matched path
    //
    // Expected:
    // - CI mode does not allow runtime override
    // - Build-time detection is source of truth (determinism)
    // - Test skipped with exit(0)
    //
    // This test requires subprocess execution - marked #[ignore]
    todo!("AC1, AC3: Verify preflight exits on runtime override in CI mode (requires subprocess)");
}

// ============================================================================
// Category C: Edge Case Tests
// ============================================================================

/// Tests spec: runtime-detection-matched-path-integration.md
/// Validates: Empty RPATH environment variable is handled gracefully
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
fn test_detect_backend_empty_rpath_env() {
    // TODO: Edge case - Empty RPATH:
    // 1. Set CROSSVAL_RPATH_BITNET="" (empty string)
    // 2. Call detect_backend_runtime(CppBackend::BitNet)
    // 3. Verify:
    //    - No panic or error
    //    - Empty string ignored in search path
    //    - Falls back to next priority (BITNET_CPP_DIR)
    //
    // Expected:
    // - Empty environment variables handled gracefully
    // - No candidate added to search list for empty strings
    //
    // Use tests::support::backend_helpers::detect_backend_runtime
    // Use tests::support::env_guard::EnvGuard
    todo!("Edge case: Empty RPATH environment variable handled gracefully");
}

/// Tests spec: runtime-detection-matched-path-integration.md
/// Validates: Missing subdirectories are skipped during search
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
fn test_detect_backend_missing_subdirs() {
    // TODO: Edge case - Missing subdirectories:
    // 1. Create temp directory without build/ subdirectory
    // 2. Set BITNET_CPP_DIR to temp path
    // 3. Call detect_backend_runtime(CppBackend::BitNet)
    // 4. Verify:
    //    - No panic on missing subdirectories
    //    - Returns (false, None) when no libraries found
    //
    // Expected:
    // - Non-existent paths skipped gracefully
    // - Search continues to next candidate
    //
    // Use tests::support::backend_helpers::detect_backend_runtime
    todo!("Edge case: Missing subdirectories are skipped during search");
}

/// Tests spec: runtime-detection-matched-path-integration.md
/// Validates: Partial library set detection fails correctly
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
fn test_detect_backend_partial_library_set() {
    // TODO: Edge case - Partial library set (Llama backend):
    // 1. Create temp directory with only libllama.so (missing libggml.so)
    // 2. Set LLAMA_CPP_DIR to temp path
    // 3. Call detect_backend_runtime(CppBackend::Llama)
    // 4. Verify:
    //    - Returns (false, None)
    //    - All required libraries must be present
    //
    // Expected:
    // - Llama backend requires BOTH libllama and libggml
    // - Partial matches rejected
    //
    // Use tests::support::backend_helpers::detect_backend_runtime
    todo!("Edge case: Partial library set detection fails correctly (requires all libs)");
}

/// Tests spec: runtime-detection-matched-path-integration.md
/// Validates: Symlink resolution follows links to actual library locations
#[test]
#[cfg(all(test, feature = "crossval-all", unix))]
#[serial_test::serial(bitnet_env)]
#[ignore] // Requires symlink support (platform-specific)
fn test_detect_backend_symlink_resolution() {
    // TODO: Edge case - Symlink resolution:
    // 1. Create temp directory with real libraries
    // 2. Create symlink to temp directory
    // 3. Set BITNET_CPP_DIR to symlink path
    // 4. Call detect_backend_runtime(CppBackend::BitNet)
    // 5. Verify:
    //    - Symlinks followed correctly
    //    - Libraries found in symlinked target
    //    - Matched path may be symlink or target (implementation-defined)
    //
    // Expected:
    // - std::path::Path::exists() follows symlinks by default
    // - Detection works through symlinked directories
    //
    // This test is Unix-specific and marked #[ignore]
    todo!("Edge case: Symlink resolution follows links to library locations (Unix only)");
}

/// Tests spec: runtime-detection-matched-path-integration.md
/// Validates: Permission errors during directory read are handled gracefully
#[test]
#[cfg(all(test, feature = "crossval-all", unix))]
#[serial_test::serial(bitnet_env)]
#[ignore] // Requires permission manipulation (may fail in CI)
fn test_detect_backend_permission_error() {
    // TODO: Edge case - Permission errors:
    // 1. Create temp directory and remove read permissions
    // 2. Set BITNET_CPP_DIR to restricted directory
    // 3. Call detect_backend_runtime(CppBackend::BitNet)
    // 4. Verify:
    //    - No panic on permission denied
    //    - Returns (false, None) or Err with descriptive message
    //    - Continues to next search candidate
    //
    // Expected:
    // - Permission errors handled gracefully
    // - Descriptive error message if Err returned
    //
    // This test requires permission manipulation - marked #[ignore]
    todo!("Edge case: Permission errors during directory read handled gracefully (Unix only)");
}

/// Tests spec: runtime-detection-matched-path-integration.md
/// Validates: Non-existent paths in environment variables are skipped
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
fn test_detect_backend_nonexistent_path() {
    // TODO: Edge case - Non-existent paths:
    // 1. Set BITNET_CPP_DIR to non-existent path (/tmp/nonexistent-12345)
    // 2. Call detect_backend_runtime(CppBackend::BitNet)
    // 3. Verify:
    //    - No panic
    //    - Returns (false, None)
    //    - Non-existent paths skipped
    //
    // Expected:
    // - Path existence check before library search
    // - Non-existent paths skipped silently
    //
    // Use tests::support::backend_helpers::detect_backend_runtime
    todo!("Edge case: Non-existent paths in environment variables are skipped");
}

/// Tests spec: runtime-detection-matched-path-integration.md
/// Validates: Multiple search paths checked in priority order
#[test]
#[cfg(all(test, feature = "crossval-all"))]
#[serial_test::serial(bitnet_env)]
fn test_detect_backend_multiple_search_paths() {
    // TODO: Edge case - Multiple search paths:
    // 1. Create 3 temp directories (priority 1, 2, 3)
    // 2. Place libraries in priority 2 directory only
    // 3. Set all environment variables
    // 4. Call detect_backend_runtime(CppBackend::BitNet)
    // 5. Verify:
    //    - Returns (true, Some(priority_2_path))
    //    - First match wins (priority order)
    //
    // Expected:
    // - Search stops at first match
    // - Priority order respected (BITNET_CROSSVAL_LIBDIR > CROSSVAL_RPATH_BITNET > BITNET_CPP_DIR)
    //
    // Use tests::support::backend_helpers::detect_backend_runtime
    todo!("Edge case: Multiple search paths checked in priority order (first match wins)");
}

/// Tests spec: runtime-detection-matched-path-integration.md
/// Validates: Windows .dll extension handled correctly
#[test]
#[cfg(all(test, feature = "crossval-all", target_os = "windows"))]
#[serial_test::serial(bitnet_env)]
fn test_detect_backend_windows_dll_extensions() {
    // TODO: Edge case - Windows DLL extensions:
    // 1. Create temp directory with bitnet.dll (no "lib" prefix)
    // 2. Set BITNET_CPP_DIR to temp path
    // 3. Call detect_backend_runtime(CppBackend::BitNet)
    // 4. Verify:
    //    - Returns (true, Some(matched_path))
    //    - Windows naming conventions respected (no lib prefix)
    //
    // Expected:
    // - Windows uses "bitnet.dll" not "libbitnet.dll"
    // - Platform-specific naming handled by format_lib_name_ext
    //
    // This test is Windows-specific (cfg(target_os = "windows"))
    todo!("Edge case: Windows .dll extension handled correctly (no lib prefix)");
}

/// Tests spec: runtime-detection-matched-path-integration.md
/// Validates: macOS .dylib extension handled correctly
#[test]
#[cfg(all(test, feature = "crossval-all", target_os = "macos"))]
#[serial_test::serial(bitnet_env)]
fn test_detect_backend_macos_dylib_extensions() {
    // TODO: Edge case - macOS dylib extensions:
    // 1. Create temp directory with libbitnet.dylib
    // 2. Set BITNET_CPP_DIR to temp path
    // 3. Call detect_backend_runtime(CppBackend::BitNet)
    // 4. Verify:
    //    - Returns (true, Some(matched_path))
    //    - macOS naming conventions respected (lib prefix + .dylib)
    //
    // Expected:
    // - macOS uses "libbitnet.dylib" not "libbitnet.so"
    // - Platform-specific extension handled by format_lib_name_ext
    //
    // This test is macOS-specific (cfg(target_os = "macos"))
    todo!("Edge case: macOS .dylib extension handled correctly");
}
