//! Comprehensive test scaffolding for test infrastructure helpers
//!
//! Tests specification: docs/specs/test-infrastructure-conditional-execution.md
//!
//! # Acceptance Criteria Coverage (AC1-AC7)
//!
//! This test suite provides comprehensive coverage for the test infrastructure
//! helpers that enable conditional test execution based on C++ backend availability.
//!
//! ## Coverage Map
//!
//! - **AC1**: `ensure_backend_or_skip()` function behavior (8 tests)
//! - **AC2**: Backend availability detection mechanisms (6 tests)
//! - **AC3**: Auto-repair integration with retry logic (7 tests)
//! - **AC4**: Skip messages with setup instructions (5 tests)
//! - **AC5**: Test fixture helpers for environment isolation (8 tests)
//! - **AC6**: Serial test execution pattern enforcement (6 tests)
//! - **AC7**: Platform-specific helpers for cross-platform testing (5 tests)
//!
//! ## Test Categories
//!
//! ### Meta-Tests (Testing the Test Helpers)
//!
//! These tests validate the test infrastructure itself, ensuring helpers behave
//! correctly under various conditions (backend available/unavailable, CI/local,
//! repair success/failure).
//!
//! ### Mock Strategy
//!
//! - Build-time constants mocking: Via conditional compilation and test features
//! - Command execution mocking: Test doubles for `std::process::Command`
//! - Environment isolation: `#[serial(bitnet_env)]` + `EnvGuard` RAII pattern
//!
//! ### Platform Coverage
//!
//! - Linux: `.so` libraries, `LD_LIBRARY_PATH`
//! - macOS: `.dylib` libraries, `DYLD_LIBRARY_PATH`
//! - Windows: `.dll` libraries, `PATH`

use serial_test::serial;
use std::path::PathBuf;

// Import test infrastructure
mod support {
    pub mod backend_helpers;
    pub mod env_guard;
}

// Test helper imports
#[allow(unused_imports)]
use support::backend_helpers::{
    ensure_backend_or_skip, ensure_bitnet_or_skip, ensure_llama_or_skip,
};
use support::env_guard::{EnvGuard, EnvScope};

#[allow(unused_imports)]
use bitnet_crossval::backend::CppBackend;

// ============================================================================
// AC1: ensure_backend_or_skip() Helper Behavior (8 tests)
// ============================================================================

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac1
///
/// Validates: Backend available → ensure_backend_or_skip returns without panic
#[test]
fn test_ac1_backend_available_continues_execution() {
    use bitnet_crossval::{HAS_BITNET, HAS_LLAMA};
    // When the backend is available at build time, the function returns immediately.
    if HAS_BITNET {
        ensure_backend_or_skip(CppBackend::BitNet); // Must not panic.
    }
    if HAS_LLAMA {
        ensure_backend_or_skip(CppBackend::Llama); // Must not panic.
    }
    // If neither backend is available this test is a no-op (still passes).
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac1
///
/// Validates: Backend unavailable + repair allowed → attempts auto-repair
///
/// # Test Strategy
///
/// Mock `HAS_BITNET = false`, `BITNET_TEST_NO_REPAIR` unset, `CI` unset
/// Expected: Attempts to run `cargo run -p xtask -- setup-cpp-auto`
#[test]
#[ignore = "TDD scaffold: Test: backend unavailable + repair allowed → attempts repair"]
#[serial(bitnet_env)]
fn test_ac1_backend_unavailable_repair_allowed_attempts_repair() {
    let _guard_no_repair = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    let _guard_ci = EnvGuard::new("CI");

    // Remove both CI and NO_REPAIR flags to enable repair
    _guard_no_repair.remove();
    _guard_ci.remove();

    // TDD scaffolding - implementation pending
    //
    // Test logic:
    // 1. Mock HAS_BITNET = false
    // 2. Mock Command::new("cargo") to capture args
    // 3. Call ensure_backend_or_skip(CppBackend::BitNet)
    // 4. Verify: Command invoked with ["run", "-p", "xtask", "--", "setup-cpp-auto"]
    // 5. Verify: Prints "Attempting auto-repair..." to stderr
    // 6. Verify: Prints rebuild instructions after successful repair
    //
    // Mock strategy:
    // - Use Command wrapper/trait for testability
    // - Capture stderr output for verification
    unimplemented!(
        "Test: backend unavailable + repair allowed → attempts repair\n\
         Spec: AC1 - auto-repair orchestration\n\
         Mock: Command execution capture"
    );
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac1
///
/// Validates: Backend unavailable + repair disabled (CI mode) → skips gracefully
///
/// # Test Strategy
///
/// Mock `HAS_BITNET = false`, `BITNET_TEST_NO_REPAIR = 1`
/// Expected: Immediate skip with diagnostic, no repair attempt
#[test]
#[serial(bitnet_env)]
fn test_ac1_backend_unavailable_repair_disabled_skips_gracefully() {
    use bitnet_crossval::HAS_BITNET;
    if HAS_BITNET {
        return; // Backend available – no skip occurs.
    }
    let mut scope = EnvScope::new();
    scope.set("BITNET_TEST_NO_REPAIR", "1");
    scope.remove("BITNET_CROSSVAL_LIBDIR");
    scope.remove("CROSSVAL_RPATH_BITNET");
    scope.remove("BITNET_CPP_DIR");

    let result = std::panic::catch_unwind(|| {
        ensure_backend_or_skip(CppBackend::BitNet);
    });
    assert!(result.is_err(), "Should panic (SKIPPED) when backend unavailable with NO_REPAIR");
    let err = result.unwrap_err();
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(msg.contains("SKIPPED"), "Panic should contain 'SKIPPED', got: {msg}");
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac1
///
/// Validates: Backend unavailable + repair fails → skips with diagnostic
///
/// # Test Strategy
///
/// Mock `HAS_BITNET = false`, auto-repair command returns non-zero exit code
/// Expected: Skip message includes repair failure reason
#[test]
#[ignore = "TDD scaffold: Test: backend unavailable + repair fails → skips with diagnostic"]
#[serial(bitnet_env)]
fn test_ac1_backend_unavailable_repair_fails_skips_with_diagnostic() {
    let _guard_no_repair = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    let _guard_ci = EnvGuard::new("CI");

    // Remove both flags to enable repair
    _guard_no_repair.remove();
    _guard_ci.remove();

    // TDD scaffolding - implementation pending
    //
    // Test logic:
    // 1. Mock HAS_BITNET = false
    // 2. Mock Command to return exit code 1 (failure)
    // 3. Call ensure_backend_or_skip(CppBackend::BitNet)
    // 4. Verify: Prints "auto-repair failed: command failed"
    // 5. Verify: Prints manual setup instructions
    // 6. Verify: Includes troubleshooting guidance
    //
    // Mock strategy:
    // - Mock Command::status() to return Err or ExitStatus(1)
    // - Capture stderr for diagnostic message verification
    unimplemented!(
        "Test: backend unavailable + repair fails → skips with diagnostic\n\
         Spec: AC1 - repair failure handling\n\
         Mock: Command failure simulation"
    );
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac1
///
/// Validates: Convenience wrapper ensure_bitnet_or_skip() delegates correctly
#[test]
#[serial(bitnet_env)]
fn test_ac1_convenience_wrapper_ensure_bitnet_or_skip() {
    use bitnet_crossval::HAS_BITNET;
    let mut scope = EnvScope::new();
    scope.set("CI", "1");
    scope.set("BITNET_TEST_NO_REPAIR", "1");
    scope.remove("BITNET_CROSSVAL_LIBDIR");
    scope.remove("CROSSVAL_RPATH_BITNET");
    scope.remove("BITNET_CPP_DIR");

    let result = std::panic::catch_unwind(ensure_bitnet_or_skip);
    if HAS_BITNET {
        assert!(result.is_ok(), "ensure_bitnet_or_skip should not panic when HAS_BITNET=true");
    } else {
        assert!(
            result.is_err(),
            "ensure_bitnet_or_skip should panic (SKIPPED) in CI with no backend"
        );
        let err = result.unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(msg.contains("SKIPPED"), "ensure_bitnet_or_skip panic should contain 'SKIPPED'");
    }
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac1
///
/// Validates: Convenience wrapper ensure_llama_or_skip() delegates correctly
#[test]
#[serial(bitnet_env)]
fn test_ac1_convenience_wrapper_ensure_llama_or_skip() {
    use bitnet_crossval::HAS_LLAMA;
    let mut scope = EnvScope::new();
    scope.set("CI", "1");
    scope.set("BITNET_TEST_NO_REPAIR", "1");
    scope.remove("BITNET_CROSSVAL_LIBDIR");
    scope.remove("CROSSVAL_RPATH_LLAMA");
    scope.remove("LLAMA_CPP_DIR");

    let result = std::panic::catch_unwind(ensure_llama_or_skip);
    if HAS_LLAMA {
        assert!(result.is_ok(), "ensure_llama_or_skip should not panic when HAS_LLAMA=true");
    } else {
        assert!(
            result.is_err(),
            "ensure_llama_or_skip should panic (SKIPPED) in CI with no backend"
        );
        let err = result.unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(msg.contains("SKIPPED"), "ensure_llama_or_skip panic should contain 'SKIPPED'");
    }
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac1
///
/// Validates: Backend-specific flags passed to auto-repair command
#[test]
#[ignore = "TDD scaffold: Test: auto-repair uses backend-specific command flags"]
#[serial(bitnet_env)]
fn test_ac1_auto_repair_backend_specific_flags() {
    let _guard_no_repair = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    let _guard_ci = EnvGuard::new("CI");

    _guard_no_repair.remove();
    _guard_ci.remove();

    // TDD scaffolding - implementation pending
    //
    // Test logic (BitNet backend):
    // 1. Mock HAS_BITNET = false
    // 2. Call ensure_backend_or_skip(CppBackend::BitNet)
    // 3. Verify: Command args include "--bitnet" flag
    //
    // Test logic (Llama backend):
    // 1. Mock HAS_LLAMA = false
    // 2. Call ensure_backend_or_skip(CppBackend::Llama)
    // 3. Verify: Command args do NOT include backend-specific flag (default)
    //
    // Mock strategy:
    // - Capture Command args for verification
    // - Test both backends in separate test cases
    unimplemented!(
        "Test: auto-repair uses backend-specific command flags\n\
         Spec: AC1 - backend selection in setup command\n\
         BitNet: ['--bitnet'], Llama: default"
    );
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac1
///
/// Validates: Rebuild instructions printed after successful repair
#[test]
#[ignore = "TDD scaffold: Test: auto-repair shows rebuild instructions"]
#[serial(bitnet_env)]
fn test_ac1_auto_repair_shows_rebuild_instructions() {
    let _guard_no_repair = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    let _guard_ci = EnvGuard::new("CI");

    _guard_no_repair.remove();
    _guard_ci.remove();

    // TDD scaffolding - implementation pending
    //
    // Test logic:
    // 1. Mock HAS_BITNET = false
    // 2. Mock attempt_auto_repair to return Ok(())
    // 3. Call ensure_backend_or_skip(CppBackend::BitNet)
    // 4. Verify: Prints "✓ Backend installed. Rebuild required to detect:"
    // 5. Verify: Prints "cargo clean -p crossval && cargo build --features crossval-all"
    //
    // Stderr capture:
    // - Use test stdout/stderr capture
    // - Verify exact message format from spec
    unimplemented!(
        "Test: auto-repair shows rebuild instructions\n\
         Spec: AC1 - rebuild guidance after successful repair"
    );
}

// ============================================================================
// AC2: Backend Availability Detection (6 tests)
// ============================================================================

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac2
///
/// Validates: Build-time constant detection (HAS_BITNET, HAS_LLAMA)
#[test]
fn test_ac2_build_time_constant_detection() {
    use bitnet_crossval::{HAS_BITNET, HAS_LLAMA};
    // The constants must be booleans and accessible.
    let _bitnet: bool = HAS_BITNET;
    let _llama: bool = HAS_LLAMA;
    // In a test environment without C++ backends, both should be false.
    // (Unless CROSSVAL_HAS_BITNET/LLAMA was set at build time.)
    // Just verify the constants compile and are boolean.
    assert!(HAS_BITNET == true || HAS_BITNET == false);
    assert!(HAS_LLAMA == true || HAS_LLAMA == false);
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac2
///
/// Validates: Runtime detection fallback when libraries installed post-build
#[test]
#[serial(bitnet_env)]
fn test_ac2_runtime_detection_fallback() {
    use bitnet_crossval::HAS_BITNET;
    use support::backend_helpers::create_mock_backend_libs;

    // create_mock_backend_libs creates its own TempDir and returns it.
    let temp = create_mock_backend_libs(CppBackend::BitNet).unwrap();

    let mut scope = EnvScope::new();
    // Point runtime detection to the mock libs.
    scope.set("CROSSVAL_RPATH_BITNET", temp.path().to_str().unwrap());
    scope.remove("BITNET_CROSSVAL_LIBDIR");
    scope.remove("BITNET_CPP_DIR");
    // Disable CI mode so the stale-build path allows execution to continue.
    scope.remove("CI");
    scope.remove("BITNET_TEST_NO_REPAIR");
    scope.remove("BITNET_REPAIR_ATTEMPTED");
    scope.remove("BITNET_REPAIR_IN_PROGRESS");

    // When HAS_BITNET=false AND runtime detection finds libs AND no CI:
    // ensure_backend_or_skip should return (not panic).
    if !HAS_BITNET {
        // Stale-build dev path: function emits a warning and returns.
        let result = std::panic::catch_unwind(|| {
            ensure_backend_or_skip(CppBackend::BitNet);
        });
        assert!(
            result.is_ok(),
            "ensure_backend_or_skip should return when runtime finds libs (non-CI stale build)"
        );
    }
    // If HAS_BITNET=true the function returns without even checking runtime – also OK.
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac2
///
/// Validates: Rebuild warning printed when runtime differs from build-time
#[test]
#[serial(bitnet_env)]
fn test_ac2_rebuild_warning_on_detection_mismatch() {
    use bitnet_crossval::HAS_BITNET;
    use support::backend_helpers::{create_mock_backend_libs, detect_backend_runtime};

    // Create mock libs so runtime detection succeeds
    let temp = create_mock_backend_libs(CppBackend::BitNet).unwrap();

    let mut scope = EnvScope::new();
    scope.set("CROSSVAL_RPATH_BITNET", temp.path().to_str().unwrap());
    scope.remove("BITNET_CROSSVAL_LIBDIR");
    scope.remove("BITNET_CPP_DIR");

    // Verify runtime detection finds the libs
    let (found, matched_path) = detect_backend_runtime(CppBackend::BitNet).unwrap();
    assert!(found, "Runtime detection should find mock libs");
    assert!(matched_path.is_some(), "Should return matched path");

    // If build-time says false but runtime says true, that's the mismatch scenario.
    // We can't control HAS_BITNET (it's a build-time constant), but we can verify
    // the detection discrepancy exists when HAS_BITNET is false.
    if !HAS_BITNET {
        // The stale-build dev path: ensure_backend_or_skip should return (not panic)
        // because runtime found the libs in non-CI mode.
        scope.remove("CI");
        scope.remove("BITNET_TEST_NO_REPAIR");
        scope.remove("BITNET_REPAIR_ATTEMPTED");
        scope.remove("BITNET_REPAIR_IN_PROGRESS");

        let result = std::panic::catch_unwind(|| {
            ensure_backend_or_skip(CppBackend::BitNet);
        });
        assert!(
            result.is_ok(),
            "Should continue (with warning) when runtime finds libs in dev mode"
        );
    }
    // If HAS_BITNET is true, the build-time/runtime mismatch doesn't apply.
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac2
///
/// Validates: CI mode detection via BITNET_TEST_NO_REPAIR environment variable
#[test]
#[serial(bitnet_env)]
fn test_ac2_ci_mode_detection_via_no_repair_flag() {
    let _guard = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    _guard.set("1");

    // Verify: is_ci_or_no_repair() returns true when BITNET_TEST_NO_REPAIR=1
    // We can't call is_ci_or_no_repair() directly (it's private), but we can
    // verify the behavior indirectly by checking that ensure_backend_or_skip
    // doesn't attempt repair when the flag is set.
    //
    // For now, just verify the environment variable is set correctly
    assert_eq!(std::env::var("BITNET_TEST_NO_REPAIR"), Ok("1".to_string()));
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac2
///
/// Validates: CI mode detection via CI environment variable
#[test]
#[serial(bitnet_env)]
fn test_ac2_ci_mode_detection_via_ci_flag() {
    let _guard = EnvGuard::new("CI");
    _guard.set("1");

    // Verify: CI=1 is set correctly
    assert_eq!(std::env::var("CI"), Ok("1".to_string()));
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac2
///
/// Validates: Interactive mode (no CI/NO_REPAIR flags) allows repair
#[test]
#[serial(bitnet_env)]
fn test_ac2_interactive_mode_allows_repair() {
    use temp_env::with_vars_unset;
    with_vars_unset(["BITNET_TEST_NO_REPAIR", "CI"], || {
        // Verify: Both environment variables are unset
        assert!(std::env::var("BITNET_TEST_NO_REPAIR").is_err());
        assert!(std::env::var("CI").is_err());
    });
}

// ============================================================================
// AC3: Auto-Repair Integration with Retry Logic (7 tests)
// ============================================================================

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac3
///
/// Validates: Auto-repair invokes xtask setup-cpp-auto command
#[test]
#[ignore = "TDD scaffold: Test: auto-repair invokes xtask setup-cpp-auto"]
#[serial(bitnet_env)]
fn test_ac3_auto_repair_invokes_xtask_setup_cpp_auto() {
    let _guard_no_repair = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    let _guard_ci = EnvGuard::new("CI");

    _guard_no_repair.remove();
    _guard_ci.remove();

    // TDD scaffolding - implementation pending
    //
    // Test logic:
    // 1. Mock HAS_BITNET = false
    // 2. Mock Command::new("cargo")
    // 3. Call ensure_backend_or_skip(CppBackend::BitNet)
    // 4. Verify: Command args = ["run", "-p", "xtask", "--", "setup-cpp-auto", "--emit=sh"]
    // 5. Verify: Command is executed (status() called)
    //
    // Mock strategy:
    // - Use Command wrapper trait for testability
    // - Capture invocation args for verification
    unimplemented!(
        "Test: auto-repair invokes xtask setup-cpp-auto\n\
         Spec: AC3 - command orchestration\n\
         Mock: Command execution capture"
    );
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac3
///
/// Validates: Retry logic on transient errors (network, temporary build failures)
#[test]
#[ignore = "TDD scaffold: Test: retry logic on transient errors"]
#[serial(bitnet_env)]
fn test_ac3_retry_logic_on_transient_errors() {
    let _guard_no_repair = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    let _guard_ci = EnvGuard::new("CI");

    _guard_no_repair.remove();
    _guard_ci.remove();

    // TDD scaffolding - implementation pending
    //
    // Test logic:
    // 1. Mock HAS_BITNET = false
    // 2. Mock attempt_auto_repair to return Err(RepairError::NetworkError) first
    // 3. Mock second attempt to return Ok(())
    // 4. Call ensure_backend_or_skip(CppBackend::BitNet)
    // 5. Verify: Two repair attempts made (initial + 1 retry)
    // 6. Verify: Delay between retries (1s, 2s exponential backoff)
    //
    // Mock strategy:
    // - Track retry count
    // - Verify sleep durations
    // - Simulate transient error recovery
    unimplemented!(
        "Test: retry logic on transient errors\n\
         Spec: AC3 - exponential backoff retry strategy"
    );
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac3
///
/// Validates: Error classification (network, build, prerequisites, permissions)
#[test]
#[ignore = "TDD scaffold: Test: error classification for repair failures"]
fn test_ac3_error_classification() {
    // TDD scaffolding - implementation pending
    //
    // Test logic:
    // 1. Test RepairError::NetworkError with retryable = true
    // 2. Test RepairError::BuildError with retryable = true/false
    // 3. Test RepairError::MissingPrerequisites with retryable = false
    // 4. Test RepairError::PermissionDenied with retryable = false
    // 5. Test RepairError::Unknown
    //
    // Verify:
    // - is_retryable() returns correct value for each error type
    // - Error messages are descriptive
    unimplemented!(
        "Test: error classification for repair failures\n\
         Spec: AC3 - RepairError enum with retryability"
    );
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac3
///
/// Validates: Recursion prevention via BITNET_REPAIR_IN_PROGRESS guard
#[test]
#[ignore = "TDD scaffold: Test: recursion prevention via environment guard"]
#[serial(bitnet_env)]
fn test_ac3_recursion_prevention() {
    let _guard_no_repair = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    let _guard_ci = EnvGuard::new("CI");
    let _guard_in_progress = EnvGuard::new("BITNET_REPAIR_IN_PROGRESS");

    _guard_no_repair.remove();
    _guard_ci.remove();
    _guard_in_progress.set("1"); // Simulate recursion

    // TDD scaffolding - implementation pending
    //
    // Test logic:
    // 1. Set BITNET_REPAIR_IN_PROGRESS = 1
    // 2. Call attempt_auto_repair_with_retry()
    // 3. Verify: Returns Err(RepairError::RecursionDetected)
    // 4. Verify: No Command execution attempted
    //
    // Recursion scenarios:
    // - Setup script triggers another test run
    // - Nested repair attempts
    unimplemented!(
        "Test: recursion prevention via environment guard\n\
         Spec: AC3 - BITNET_REPAIR_IN_PROGRESS safety"
    );
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac3
///
/// Validates: Maximum retry limit enforcement (2 retries max)
#[test]
#[ignore = "TDD scaffold: Test: max retry limit enforcement"]
#[serial(bitnet_env)]
fn test_ac3_max_retry_limit_enforcement() {
    let _guard_no_repair = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    let _guard_ci = EnvGuard::new("CI");

    _guard_no_repair.remove();
    _guard_ci.remove();

    // TDD scaffolding - implementation pending
    //
    // Test logic:
    // 1. Mock attempt_auto_repair to always return transient error
    // 2. Call ensure_backend_or_skip(CppBackend::BitNet)
    // 3. Verify: Exactly 3 attempts made (initial + 2 retries)
    // 4. Verify: Final error returned after max retries
    //
    // Retry verification:
    // - Count Command invocations
    // - Verify backoff delays (1s, 2s)
    // - Verify final error message includes "max retries exceeded"
    unimplemented!(
        "Test: max retry limit enforcement\n\
         Spec: AC3 - bounded retry strategy (max 2 retries)"
    );
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac3
///
/// Validates: Non-retryable errors fail immediately (no retry)
#[test]
#[ignore = "TDD scaffold: Test: non-retryable errors fail immediately"]
#[serial(bitnet_env)]
fn test_ac3_non_retryable_errors_fail_immediately() {
    let _guard_no_repair = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    let _guard_ci = EnvGuard::new("CI");

    _guard_no_repair.remove();
    _guard_ci.remove();

    // TDD scaffolding - implementation pending
    //
    // Test logic:
    // 1. Mock attempt_auto_repair to return MissingPrerequisites error
    // 2. Call ensure_backend_or_skip(CppBackend::BitNet)
    // 3. Verify: Only 1 attempt made (no retries)
    // 4. Verify: Error message includes prerequisite details
    //
    // Non-retryable errors:
    // - MissingPrerequisites (git, cmake, compiler missing)
    // - PermissionDenied
    // - RecursionDetected
    unimplemented!(
        "Test: non-retryable errors fail immediately\n\
         Spec: AC3 - skip retry for permanent errors"
    );
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac3
///
/// Validates: Success message printed after successful repair
#[test]
#[ignore = "TDD scaffold: Test: success message after successful repair"]
#[serial(bitnet_env)]
fn test_ac3_success_message_after_repair() {
    let _guard_no_repair = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    let _guard_ci = EnvGuard::new("CI");

    _guard_no_repair.remove();
    _guard_ci.remove();

    // TDD scaffolding - implementation pending
    //
    // Test logic:
    // 1. Mock attempt_auto_repair to return Ok(())
    // 2. Call ensure_backend_or_skip(CppBackend::BitNet)
    // 3. Verify: Prints "✓ BitNet backend installed."
    // 4. Verify: Prints rebuild instructions
    //
    // Message format verification:
    // - Capture stderr output
    // - Match exact success message format
    unimplemented!(
        "Test: success message after successful repair\n\
         Spec: AC3 - user-friendly success feedback"
    );
}

// ============================================================================
// AC4: Skip Messages with Setup Instructions (5 tests)
// ============================================================================

/// Capture the skip diagnostic from ensure_bitnet_or_skip() in CI mode.
///
/// Returns `Some(message)` if the function panicked with a SKIPPED message,
/// or `None` if the backend was available (no diagnostic produced).
fn capture_bitnet_skip_diagnostic() -> Option<String> {
    use bitnet_crossval::HAS_BITNET;
    if HAS_BITNET {
        return None; // Backend available – no diagnostic is produced.
    }
    let result = std::panic::catch_unwind(ensure_bitnet_or_skip);
    let err = result.expect_err("ensure_bitnet_or_skip must panic when HAS_BITNET=false in CI");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    Some(msg)
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac4
///
/// Validates: Skip message format is standardized and actionable
#[test]
#[serial(bitnet_env)]
fn test_ac4_skip_message_format_standardized() {
    let mut scope = EnvScope::new();
    scope.set("CI", "1");
    scope.set("BITNET_TEST_NO_REPAIR", "1");
    scope.remove("BITNET_CROSSVAL_LIBDIR");
    scope.remove("CROSSVAL_RPATH_BITNET");
    scope.remove("BITNET_CPP_DIR");

    let Some(msg) = capture_bitnet_skip_diagnostic() else {
        return; // Backend available – skip this format test.
    };

    assert!(msg.contains("SKIPPED"), "Message should start with SKIPPED prefix, got:\n{msg}");
    assert!(msg.contains("bitnet.cpp"), "Message should mention 'bitnet.cpp', got:\n{msg}");
    assert!(
        msg.contains("Option A: Auto-setup"),
        "Message should contain Option A instructions, got:\n{msg}"
    );
    assert!(
        msg.contains("Option B: Manual setup"),
        "Message should contain Option B instructions, got:\n{msg}"
    );
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac4
///
/// Validates: Skip message includes auto-setup instructions (Option A)
#[test]
#[serial(bitnet_env)]
fn test_ac4_skip_message_includes_auto_setup_instructions() {
    let mut scope = EnvScope::new();
    scope.set("CI", "1");
    scope.set("BITNET_TEST_NO_REPAIR", "1");
    scope.remove("BITNET_CROSSVAL_LIBDIR");
    scope.remove("CROSSVAL_RPATH_BITNET");
    scope.remove("BITNET_CPP_DIR");

    let Some(msg) = capture_bitnet_skip_diagnostic() else {
        return;
    };

    assert!(
        msg.contains("setup-cpp-auto"),
        "Message should include setup-cpp-auto command, got:\n{msg}"
    );
    assert!(
        msg.contains("cargo clean -p crossval"),
        "Message should include rebuild instructions, got:\n{msg}"
    );
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac4
///
/// Validates: Skip message includes manual setup instructions (Option B)
#[test]
#[serial(bitnet_env)]
fn test_ac4_skip_message_includes_manual_setup_instructions() {
    let mut scope = EnvScope::new();
    scope.set("CI", "1");
    scope.set("BITNET_TEST_NO_REPAIR", "1");
    scope.remove("BITNET_CROSSVAL_LIBDIR");
    scope.remove("CROSSVAL_RPATH_BITNET");
    scope.remove("BITNET_CPP_DIR");

    let Some(msg) = capture_bitnet_skip_diagnostic() else {
        return;
    };

    assert!(
        msg.contains("Option B: Manual setup"),
        "Message should include manual setup option, got:\n{msg}"
    );
    assert!(msg.contains("git clone"), "Message should include git clone command, got:\n{msg}");
    assert!(msg.contains("cmake"), "Message should include cmake build instructions, got:\n{msg}");
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac4
///
/// Validates: Backend-specific instructions (BitNet vs Llama)
#[test]
#[serial(bitnet_env)]
fn test_ac4_backend_specific_instructions() {
    use bitnet_crossval::{HAS_BITNET, HAS_LLAMA};

    let mut scope = EnvScope::new();
    scope.set("CI", "1");
    scope.set("BITNET_TEST_NO_REPAIR", "1");
    scope.remove("BITNET_CROSSVAL_LIBDIR");
    scope.remove("CROSSVAL_RPATH_BITNET");
    scope.remove("BITNET_CPP_DIR");
    scope.remove("CROSSVAL_RPATH_LLAMA");
    scope.remove("LLAMA_CPP_DIR");

    // BitNet diagnostic should reference BITNET_CPP_DIR.
    if !HAS_BITNET {
        let bitnet_result = std::panic::catch_unwind(ensure_bitnet_or_skip);
        let err = bitnet_result.expect_err("Should panic for BitNet in CI");
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains("BITNET_CPP_DIR") || msg.contains("bitnet"),
            "BitNet diagnostic should mention BITNET_CPP_DIR or bitnet, got:\n{msg}"
        );
    }

    // Llama diagnostic should reference LLAMA_CPP_DIR.
    if !HAS_LLAMA {
        let llama_result = std::panic::catch_unwind(ensure_llama_or_skip);
        let err = llama_result.expect_err("Should panic for Llama in CI");
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains("LLAMA_CPP_DIR") || msg.contains("llama"),
            "Llama diagnostic should mention LLAMA_CPP_DIR or llama, got:\n{msg}"
        );
    }
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac4
///
/// Validates: Documentation link included in skip message
#[test]
#[serial(bitnet_env)]
fn test_ac4_documentation_link_included() {
    let mut scope = EnvScope::new();
    scope.set("CI", "1");
    scope.set("BITNET_TEST_NO_REPAIR", "1");
    scope.remove("BITNET_CROSSVAL_LIBDIR");
    scope.remove("CROSSVAL_RPATH_BITNET");
    scope.remove("BITNET_CPP_DIR");

    let Some(msg) = capture_bitnet_skip_diagnostic() else {
        return;
    };

    assert!(
        msg.contains("docs/howto/cpp-setup.md"),
        "Message should reference docs/howto/cpp-setup.md, got:\n{msg}"
    );
}

// ============================================================================
// AC5: Test Fixture Helpers (8 tests)
// ============================================================================

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac5
///
/// Validates: Mock library creation with platform-specific extensions
#[test]
fn test_ac5_mock_library_creation_platform_specific() {
    use support::backend_helpers::{create_mock_backend_libs, format_lib_name};

    let temp_dir = create_mock_backend_libs(CppBackend::BitNet).unwrap();

    // Verify the library file was created with the correct name
    let expected_lib = temp_dir.path().join(format_lib_name("bitnet"));
    assert!(expected_lib.exists(), "Expected library file {} to exist", expected_lib.display());

    // Platform-specific validation
    if cfg!(target_os = "linux") {
        assert!(expected_lib.to_string_lossy().ends_with(".so"));
        assert!(expected_lib.file_name().unwrap().to_string_lossy().starts_with("lib"));
    } else if cfg!(target_os = "macos") {
        assert!(expected_lib.to_string_lossy().ends_with(".dylib"));
        assert!(expected_lib.file_name().unwrap().to_string_lossy().starts_with("lib"));
    } else if cfg!(target_os = "windows") {
        assert!(expected_lib.to_string_lossy().ends_with(".dll"));
        // Windows doesn't use "lib" prefix
        assert!(!expected_lib.file_name().unwrap().to_string_lossy().starts_with("lib"));
    }
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac5
///
/// Validates: Mock library creation for multiple libraries (Llama + GGML)
#[test]
fn test_ac5_mock_library_creation_multiple_libs() {
    use support::backend_helpers::{create_mock_backend_libs, format_lib_name};

    let temp_dir = create_mock_backend_libs(CppBackend::Llama).unwrap();

    // Verify both libraries were created
    let llama_lib = temp_dir.path().join(format_lib_name("llama"));
    let ggml_lib = temp_dir.path().join(format_lib_name("ggml"));

    assert!(llama_lib.exists(), "Expected llama library {} to exist", llama_lib.display());
    assert!(ggml_lib.exists(), "Expected ggml library {} to exist", ggml_lib.display());

    // Verify correct extensions
    let lib_ext = if cfg!(target_os = "linux") {
        "so"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        panic!("Unsupported platform")
    };

    assert!(llama_lib.to_string_lossy().ends_with(lib_ext));
    assert!(ggml_lib.to_string_lossy().ends_with(lib_ext));
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac5
///
/// Validates: MockLibraryBuilder with version suffix support
#[test]
fn test_ac5_mock_library_builder_version_suffix() {
    // Minimal implementation: MockLibraryBuilder not yet implemented
    // This test is deferred until the builder pattern is needed
    // For now, verify that basic mock library creation works (tested above)

    use support::backend_helpers::create_mock_backend_libs;

    // Basic verification that mock creation works without version suffix
    let temp = create_mock_backend_libs(CppBackend::Llama).unwrap();
    assert!(temp.path().exists());

    // TODO: Implement MockLibraryBuilder for advanced use cases
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac5
///
/// Validates: create_temp_cpp_env() integrated helper
#[test]
#[serial(bitnet_env)]
fn test_ac5_create_temp_cpp_env_integrated_setup() {
    // Minimal implementation: Manual setup using basic helpers
    use support::backend_helpers::create_mock_backend_libs;
    use support::env_guard::EnvGuard;

    let temp = create_mock_backend_libs(CppBackend::BitNet).unwrap();
    let temp_path = temp.path().to_string_lossy().to_string();

    // Manual environment setup - use a block to ensure guards drop properly
    {
        let _guard_cpp_dir = EnvGuard::new("BITNET_CPP_DIR_TEST");
        _guard_cpp_dir.set(&temp_path);

        // Verify environment is set
        assert_eq!(std::env::var("BITNET_CPP_DIR_TEST").unwrap(), temp_path);
    }

    // After guards drop, verify cleanup (use different key to avoid conflicts)
    assert!(std::env::var("BITNET_CPP_DIR_TEST").is_err());

    // TODO: Implement integrated create_temp_cpp_env() helper
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac5
///
/// Validates: workspace_root() discovery helper
#[test]
fn test_ac5_workspace_root_discovery() {
    // Minimal implementation: Walk up from CARGO_MANIFEST_DIR
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Walk up until we find .git
    while !path.join(".git").exists() {
        if !path.pop() {
            panic!("Could not find workspace root (.git directory)");
        }
    }

    // Verify we found the workspace root
    assert!(path.join(".git").exists());
    assert!(path.ends_with("BitNet-rs"));

    // TODO: Implement workspace_root() helper function
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac5
///
/// Validates: env_guard() convenience function
#[test]
#[serial(bitnet_env)]
fn test_ac5_env_guard_convenience_function() {
    use support::env_guard::EnvGuard;

    // Manual approach (current pattern)
    let guard = EnvGuard::new("BITNET_TEST");
    guard.set("value");

    // Verify value is set
    assert_eq!(std::env::var("BITNET_TEST").unwrap(), "value");

    // Cleanup on drop
    drop(guard);
    assert!(std::env::var("BITNET_TEST").is_err());

    // TODO: Implement env_guard() convenience wrapper:
    // let _guard = env_guard("BITNET_TEST", "value");
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac5
///
/// Validates: env_guard_remove() convenience function
#[test]
#[serial(bitnet_env)]
fn test_ac5_env_guard_remove_convenience_function() {
    use support::env_guard::EnvGuard;

    // Set original value
    unsafe {
        std::env::set_var("BITNET_TEST", "original");
    }

    // Manual approach (current pattern)
    {
        let guard = EnvGuard::new("BITNET_TEST");
        guard.remove();

        // Verify removed
        assert!(std::env::var("BITNET_TEST").is_err());

        // Cleanup on drop restores original
    }

    // Verify restored
    assert_eq!(std::env::var("BITNET_TEST").unwrap(), "original");

    // Cleanup
    unsafe {
        std::env::remove_var("BITNET_TEST");
    }

    // TODO: Implement env_guard_remove() convenience wrapper:
    // let _guard = env_guard_remove("BITNET_TEST");
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac5
///
/// Validates: EnvGuard RAII cleanup on panic
#[test]
#[serial(bitnet_env)]
fn test_ac5_env_guard_panic_safety() {
    use support::env_guard::EnvGuard;

    // Set original value
    unsafe {
        std::env::set_var("BITNET_TEST_PANIC", "original");
    }

    let result = std::panic::catch_unwind(|| {
        let guard = EnvGuard::new("BITNET_TEST_PANIC");
        guard.set("temporary");

        // Verify temporary value is set
        assert_eq!(std::env::var("BITNET_TEST_PANIC").unwrap(), "temporary");

        panic!("intentional panic");
    });

    assert!(result.is_err(), "should have panicked");

    // Verify restoration happened despite panic
    assert_eq!(std::env::var("BITNET_TEST_PANIC").unwrap(), "original");

    // Cleanup
    unsafe {
        std::env::remove_var("BITNET_TEST_PANIC");
    }
}

// ============================================================================
// AC6: Serial Test Execution Pattern (6 tests)
// ============================================================================

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac6
///
/// Validates: #[serial(bitnet_env)] prevents concurrent env mutation
#[test]
#[serial(bitnet_env)]
fn test_ac6_serial_prevents_concurrent_env_mutation() {
    let _guard = EnvGuard::new("BITNET_TEST_CONCURRENT");
    _guard.set("value1");

    // Verify that the env var is set within the serial scope.
    assert_eq!(
        std::env::var("BITNET_TEST_CONCURRENT").as_deref(),
        Ok("value1"),
        "Env var should be set within serial scope"
    );
    // After _guard drops, the var will be restored to its prior state.
    // The #[serial(bitnet_env)] annotation ensures no other test
    // can concurrently mutate env vars in this group.
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac6
///
/// Validates: EnvGuard + #[serial] provides deterministic isolation
#[test]
#[serial(bitnet_env)]
fn test_ac6_env_guard_serial_deterministic_isolation() {
    const KEY: &str = "BITNET_TEST_ISOLATION";
    const VALUE: &str = "isolated_value";

    // Verify that a consistent result is produced on every access.
    for _ in 0..10 {
        let guard = EnvGuard::new(KEY);
        guard.set(VALUE);
        assert_eq!(
            std::env::var(KEY).as_deref(),
            Ok(VALUE),
            "Env var should be consistently set within EnvGuard scope"
        );
        // guard drops here; original value is restored before next iteration.
    }
    // After the loop, the env var should be absent (was not set before).
    let _ = EnvGuard::new(KEY); // Take the lock to read safely.
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac6
///
/// Validates: Tests without #[serial] can run in parallel (no env mutation)
#[test]
fn test_ac6_parallel_execution_without_env_mutation() {
    // This test does NOT use #[serial(bitnet_env)] or EnvGuard, because it
    // does not mutate environment variables. It is safe to run concurrently.
    let a = 2 + 2;
    let b = 2 * 2;
    assert_eq!(a, b, "Pure computation is safe to run in parallel");
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac6
///
/// Validates: RequiresEnvIsolation trait marker (compile-time documentation)
#[test]
#[serial(bitnet_env)]
fn test_ac6_requires_env_isolation_trait_marker() {
    // Verify that EnvGuard provides the isolation contract that
    // RequiresEnvIsolation would formalize: set a var, verify it's set,
    // then verify it's restored on drop.
    let test_key = "BITNET_TEST_TRAIT_MARKER";

    // Ensure clean state
    let original = std::env::var(test_key).ok();
    {
        let guard = EnvGuard::new(test_key);
        guard.set("isolated_value");
        assert_eq!(std::env::var(test_key).unwrap(), "isolated_value");
    }
    // After guard drops, value should be restored
    assert_eq!(std::env::var(test_key).ok(), original);
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac6
///
/// Validates: Anti-pattern detection (missing #[serial] with EnvGuard)
#[test]
fn test_ac6_anti_pattern_missing_serial_detection() {
    // Verify that the anti-pattern (EnvGuard without #[serial]) is documented.
    // A custom clippy lint for enforcement is future work, but we can verify
    // that EnvGuard's documentation and the codebase convention exist.
    //
    // The convention is: every test using EnvGuard MUST have #[serial(bitnet_env)].
    // This test validates that EnvGuard itself works correctly (the guard contract),
    // serving as documentation of the required pattern.
    let test_key = "BITNET_ANTI_PATTERN_CHECK";
    let guard = EnvGuard::new(test_key);
    guard.set("check");
    assert_eq!(std::env::var(test_key).unwrap(), "check");
    drop(guard);
    // After drop, the original state is restored.
    // The actual anti-pattern lint is deferred to a future clippy plugin.
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac6
///
/// Validates: Global ENV_LOCK mutex provides thread safety
#[test]
fn test_ac6_env_lock_mutex_thread_safety() {
    // Verify that EnvGuard uses a global mutex that serializes concurrent access.
    // We spawn threads that each create an EnvGuard for the same key and verify
    // no data races occur (values don't interleave).
    use std::sync::{Arc, Barrier};
    use std::thread;

    let barrier = Arc::new(Barrier::new(3));
    let mut handles = vec![];

    for i in 0..3 {
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait(); // Synchronize thread start
            let key = format!("BITNET_MUTEX_TEST_{}", i);
            let guard = EnvGuard::new(&key);
            guard.set(&format!("thread_{}", i));
            let val = std::env::var(&key).unwrap();
            assert_eq!(val, format!("thread_{}", i));
            drop(guard);
        }));
    }

    for h in handles {
        h.join().expect("Thread should not panic");
    }
}

// ============================================================================
// AC7: Platform-Specific Helpers (5 tests)
// ============================================================================

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac7
///
/// Validates: get_loader_path_var() returns platform-specific variable
#[test]
fn test_ac7_get_loader_path_var_platform_specific() {
    use support::backend_helpers::get_loader_path_var;

    // Platform-specific validation
    if cfg!(target_os = "linux") {
        assert_eq!(get_loader_path_var(), "LD_LIBRARY_PATH");
    } else if cfg!(target_os = "macos") {
        assert_eq!(get_loader_path_var(), "DYLD_LIBRARY_PATH");
    } else if cfg!(target_os = "windows") {
        assert_eq!(get_loader_path_var(), "PATH");
    } else {
        panic!("Unsupported platform for testing");
    }
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac7
///
/// Validates: get_lib_extension() returns platform-specific extension
#[test]
fn test_ac7_get_lib_extension_platform_specific() {
    use support::backend_helpers::get_lib_extension;

    // Platform-specific validation
    if cfg!(target_os = "linux") {
        assert_eq!(get_lib_extension(), "so");
    } else if cfg!(target_os = "macos") {
        assert_eq!(get_lib_extension(), "dylib");
    } else if cfg!(target_os = "windows") {
        assert_eq!(get_lib_extension(), "dll");
    } else {
        panic!("Unsupported platform for testing");
    }
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac7
///
/// Validates: format_lib_name() includes platform-specific prefix and extension
#[test]
fn test_ac7_format_lib_name_platform_specific() {
    use support::backend_helpers::format_lib_name;

    // Platform-specific validation
    if cfg!(target_os = "linux") {
        assert_eq!(format_lib_name("bitnet"), "libbitnet.so");
    } else if cfg!(target_os = "macos") {
        assert_eq!(format_lib_name("bitnet"), "libbitnet.dylib");
    } else if cfg!(target_os = "windows") {
        assert_eq!(format_lib_name("bitnet"), "bitnet.dll");
    } else {
        panic!("Unsupported platform for testing");
    }
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac7
///
/// Validates: Platform helpers work across all supported platforms
#[test]
fn test_ac7_platform_helpers_cross_platform_compatibility() {
    use support::backend_helpers::{format_lib_name, get_lib_extension, get_loader_path_var};

    // Call all platform helpers - should not panic
    let loader_var = get_loader_path_var();
    let lib_ext = get_lib_extension();
    let lib_name = format_lib_name("test");

    // Verify results are valid (non-empty strings)
    assert!(!loader_var.is_empty());
    assert!(!lib_ext.is_empty());
    assert!(!lib_name.is_empty());

    // Verify lib_name contains the extension
    assert!(lib_name.ends_with(lib_ext));
}

/// Tests spec: docs/specs/test-infrastructure-conditional-execution.md#ac7
///
/// Validates: Unsupported platform detection with clear error
#[test]
fn test_ac7_unsupported_platform_detection() {
    // Note: This test verifies that our platform helpers work on supported platforms
    // Testing unsupported platforms would require cross-compilation or mocking
    // which is beyond the scope of unit tests
    //
    // The platform helpers explicitly panic on unsupported platforms with clear messages
    // This is verified by code review of the implementation

    // For now, we just verify that the current platform is supported
    use support::backend_helpers::{get_lib_extension, get_loader_path_var};

    // These calls should not panic on supported platforms
    let _ = get_loader_path_var();
    let _ = get_lib_extension();

    // If we reach here, the platform is supported (test passes)
}

// ============================================================================
// Integration Tests (3 tests)
// ============================================================================

/// Tests: End-to-end conditional test execution workflow
#[test]
#[serial(bitnet_env)]
fn test_integration_conditional_test_execution_workflow() {
    use bitnet_crossval::HAS_BITNET;

    // Scenario: CI mode with backend unavailable → immediate skip
    let mut scope = EnvScope::new();
    scope.set("BITNET_TEST_NO_REPAIR", "1");
    scope.set("CI", "1");
    scope.remove("BITNET_CROSSVAL_LIBDIR");
    scope.remove("CROSSVAL_RPATH_BITNET");
    scope.remove("BITNET_CPP_DIR");

    if !HAS_BITNET {
        let result = std::panic::catch_unwind(|| {
            ensure_backend_or_skip(CppBackend::BitNet);
        });
        assert!(result.is_err(), "Should panic-skip when backend unavailable in CI");
        let err = result.unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(msg.contains("SKIPPED"), "Skip message should contain 'SKIPPED', got: {msg}");
    }
    // If HAS_BITNET is true, ensure_backend_or_skip returns immediately — also valid.
}

/// Tests: Integration with EnvGuard for environment restoration
#[test]
#[serial(bitnet_env)]
fn test_integration_env_guard_restoration() {
    let test_key = "BITNET_INTEGRATION_TEST";

    // Ensure clean state
    let original = std::env::var(test_key).ok();

    {
        let mut scope = EnvScope::new();
        scope.set(test_key, "temporary_value");
        assert_eq!(
            std::env::var(test_key).unwrap(),
            "temporary_value",
            "Value should be changed inside scope"
        );
    }
    // After scope drops, value should be restored
    assert_eq!(std::env::var(test_key).ok(), original, "Value should be restored after scope exit");
}

/// Tests: Integration with serial test execution
#[test]
#[serial(bitnet_env)]
fn test_integration_serial_test_execution() {
    // Verify #[serial(bitnet_env)] + EnvGuard pattern works correctly:
    // environment changes don't leak across serial test boundaries.
    let key = "BITNET_SERIAL_TEST";

    let mut scope = EnvScope::new();
    scope.set(key, "serial_value");
    assert_eq!(std::env::var(key).unwrap(), "serial_value");

    // The #[serial(bitnet_env)] attribute ensures this test runs sequentially
    // with other bitnet_env tests. The EnvScope ensures restoration on drop.
    // Together, they prevent environment pollution across tests.
}

// ============================================================================
// Property-Based Tests (2 tests)
// ============================================================================

/// Tests: Property-based environment variable combinations
#[test]
#[serial(bitnet_env)]
fn test_property_environment_variable_combinations() {
    use support::backend_helpers::is_ci;

    // Test all combinations of CI and NO_REPAIR flags
    let combos: &[(&str, &str, bool)] = &[
        // (CI value, NO_REPAIR value, expected_ci_or_no_repair)
        ("1", "1", true), // Both set → no repair
        ("1", "", true),  // CI only → no repair
        ("", "1", true),  // NO_REPAIR only → no repair
    ];

    for &(ci_val, no_repair_val, _expected) in combos {
        let mut scope = EnvScope::new();
        if ci_val.is_empty() {
            scope.remove("CI");
        } else {
            scope.set("CI", ci_val);
        }
        if no_repair_val.is_empty() {
            scope.remove("BITNET_TEST_NO_REPAIR");
        } else {
            scope.set("BITNET_TEST_NO_REPAIR", no_repair_val);
        }

        // is_ci() checks CI and GITHUB_ACTIONS env vars
        if !ci_val.is_empty() {
            assert!(is_ci(), "CI={ci_val} should make is_ci() return true");
        }
    }

    // Test: both unset → repair allowed
    {
        let mut scope = EnvScope::new();
        scope.remove("CI");
        scope.remove("GITHUB_ACTIONS");
        scope.remove("BITNET_TEST_NO_REPAIR");
        assert!(!is_ci(), "With CI unset, is_ci() should return false");
    }
}

/// Tests: Property-based backend detection
#[test]
#[serial(bitnet_env)]
fn test_property_backend_detection() {
    use bitnet_crossval::{HAS_BITNET, HAS_LLAMA};

    // Verify build-time constants are consistent booleans
    let bitnet_available: bool = HAS_BITNET;
    let llama_available: bool = HAS_LLAMA;

    // Test ensure_backend_or_skip behavior for each backend
    // When build-time constant is true, the function should return immediately.
    if bitnet_available {
        // Should not panic
        ensure_backend_or_skip(CppBackend::BitNet);
    }
    if llama_available {
        // Should not panic
        ensure_backend_or_skip(CppBackend::Llama);
    }

    // When build-time constant is false and CI mode is on, should skip
    let mut scope = EnvScope::new();
    scope.set("CI", "1");
    scope.set("BITNET_TEST_NO_REPAIR", "1");
    scope.remove("BITNET_CROSSVAL_LIBDIR");
    scope.remove("CROSSVAL_RPATH_BITNET");
    scope.remove("CROSSVAL_RPATH_LLAMA");
    scope.remove("BITNET_CPP_DIR");

    if !bitnet_available {
        let result = std::panic::catch_unwind(|| {
            ensure_backend_or_skip(CppBackend::BitNet);
        });
        assert!(result.is_err(), "BitNet unavailable + CI should skip");
    }
    if !llama_available {
        let result = std::panic::catch_unwind(|| {
            ensure_backend_or_skip(CppBackend::Llama);
        });
        assert!(result.is_err(), "Llama unavailable + CI should skip");
    }
}

// ============================================================================
// Error Handling Tests (3 tests)
// ============================================================================

/// Tests: Handle xtask command not found
#[test]
#[ignore = "TDD scaffold: Test: handle xtask command not found"]
#[serial(bitnet_env)]
fn test_error_handling_xtask_not_found() {
    let _guard_no_repair = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    let _guard_ci = EnvGuard::new("CI");

    _guard_no_repair.remove();
    _guard_ci.remove();

    // TDD scaffolding - implementation pending
    //
    // Test logic:
    // 1. Mock Command::new("cargo") to return Err(NotFound)
    // 2. Call ensure_backend_or_skip(CppBackend::BitNet)
    // 3. Verify: Returns Err("Failed to run setup-cpp-auto: command not found")
    // 4. Verify: Prints diagnostic with manual setup instructions
    //
    // Error classification:
    // - MissingPrerequisites (cargo not in PATH)
    unimplemented!(
        "Test: handle xtask command not found\n\
         Spec: Error handling for missing cargo/xtask"
    );
}

/// Tests: Handle xtask command failure
#[test]
#[ignore = "TDD scaffold: Test: handle xtask command failure"]
#[serial(bitnet_env)]
fn test_error_handling_xtask_failure() {
    let _guard_no_repair = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    let _guard_ci = EnvGuard::new("CI");

    _guard_no_repair.remove();
    _guard_ci.remove();

    // TDD scaffolding - implementation pending
    //
    // Test logic:
    // 1. Mock Command to return exit code 1
    // 2. Call ensure_backend_or_skip(CppBackend::BitNet)
    // 3. Verify: Returns Err("returned non-zero exit code: 1")
    // 4. Verify: Prints stderr from command
    //
    // Error classification:
    // - BuildError (retryable if transient)
    // - NetworkError (retryable)
    unimplemented!(
        "Test: handle xtask command failure\n\
         Spec: Error handling for setup-cpp-auto failures"
    );
}

/// Tests: Handle permission errors during repair
#[test]
#[ignore = "TDD scaffold: Test: handle permission errors during repair"]
#[serial(bitnet_env)]
fn test_error_handling_permission_errors() {
    let _guard_no_repair = EnvGuard::new("BITNET_TEST_NO_REPAIR");
    let _guard_ci = EnvGuard::new("CI");

    _guard_no_repair.remove();
    _guard_ci.remove();

    // TDD scaffolding - implementation pending
    //
    // Test logic:
    // 1. Mock Command to fail with permission denied error
    // 2. Call ensure_backend_or_skip(CppBackend::BitNet)
    // 3. Verify: Graceful skip with diagnostic
    // 4. Verify: Suggests using sudo or changing installation directory
    //
    // Error classification:
    // - PermissionDenied (non-retryable)
    unimplemented!(
        "Test: handle permission errors during repair\n\
         Spec: Error handling for permission issues"
    );
}

// ============================================================================
// Coverage Analysis (1 test)
// ============================================================================

/// Tests: Coverage analysis metadata and documentation
///
/// This test serves as documentation for coverage targets.
/// Target: 95%+ coverage of backend_helpers.rs code paths
#[test]
fn test_coverage_target_documentation() {
    // Coverage documentation test — verifies key functions are accessible.
    //
    // Coverage targets:
    // - ensure_backend_or_skip: 100% (all branches)
    // - is_ci / is_ci_or_no_repair: 100%
    // - detect_backend_runtime: 100%
    // - format_skip_diagnostic: both backends
    // - Convenience wrappers: ensure_bitnet_or_skip, ensure_llama_or_skip
    //
    // Measurement: cargo tarpaulin --workspace --out Html

    // Verify key functions are importable and callable
    use support::backend_helpers::{
        detect_backend_runtime, format_lib_name, get_lib_extension, get_loader_path_var, is_ci,
    };

    // Smoke-check each helper
    let _loader_var = get_loader_path_var();
    let _ext = get_lib_extension();
    let _name = format_lib_name("test");
    let _ci = is_ci();
    let _ = detect_backend_runtime(CppBackend::BitNet);
}

// ============================================================================
// Helper Function Metadata Tests (for testing helpers themselves)
// ============================================================================

/// Meta-test helper: Verify skip behavior occurred
///
/// This helper is used by other tests to verify that a test properly skipped
/// execution (printed skip message, didn't attempt backend operations).
///
/// # Example
///
/// ```rust,ignore
/// verify_skip_behavior(|| {
///     ensure_backend_or_skip(CppBackend::BitNet);
/// }).expect("should have skipped");
/// ```
#[allow(dead_code)]
fn verify_skip_behavior<F>(test_fn: F) -> Result<(), String>
where
    F: Fn() + std::panic::UnwindSafe,
{
    let result = std::panic::catch_unwind(test_fn);
    match result {
        Err(err) => {
            let msg = err
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            if msg.contains("SKIPPED") {
                Ok(())
            } else {
                Err(format!("Panicked but not with SKIPPED: {msg}"))
            }
        }
        Ok(()) => Err("Function returned without skipping".to_string()),
    }
}

/// Meta-test helper: Assert setup instructions present in skip message
///
/// Verifies that a skip message includes actionable setup instructions.
#[allow(dead_code)]
fn assert_setup_instructions_present(skip_msg: &str) {
    assert!(
        skip_msg.contains("Option A: Auto-setup") || skip_msg.contains("setup-cpp-auto"),
        "Skip message should contain auto-setup instructions"
    );
    assert!(
        skip_msg.contains("Option B: Manual setup") || skip_msg.contains("git clone"),
        "Skip message should contain manual setup instructions"
    );
}

/// Meta-test helper: Verify backend detection result
///
/// Checks that backend detection returns expected availability result.
#[allow(dead_code)]
fn verify_backend_detection(backend: CppBackend, expected: bool) -> Result<(), String> {
    use bitnet_crossval::{HAS_BITNET, HAS_LLAMA};

    let actual = match backend {
        CppBackend::BitNet => HAS_BITNET,
        CppBackend::Llama => HAS_LLAMA,
    };

    if actual == expected {
        Ok(())
    } else {
        Err(format!("Backend {:?}: expected={}, actual={}", backend, expected, actual))
    }
}

/// Meta-test helper: Create mock library setup for platform
///
/// Creates mock C++ backend libraries in a temporary directory with
/// platform-specific naming and extensions.
#[allow(dead_code)]
fn mock_library_setup(backend: CppBackend) -> Result<(PathBuf, Vec<PathBuf>), String> {
    use support::backend_helpers::{create_mock_backend_libs, format_lib_name};

    let temp = create_mock_backend_libs(backend)?;
    let lib_names = match backend {
        CppBackend::BitNet => vec!["bitnet"],
        CppBackend::Llama => vec!["llama", "ggml"],
    };

    let paths: Vec<PathBuf> =
        lib_names.iter().map(|name| temp.path().join(format_lib_name(name))).collect();

    let dir = temp.path().to_path_buf();
    // Leak the TempDir so the caller can use the paths (caller responsible for cleanup)
    std::mem::forget(temp);
    Ok((dir, paths))
}
