//! Comprehensive TDD test scaffolding for robust re-exec with cargo run fallback
//!
//! **Specification**: docs/specs/reexec-cargo-fallback-robust.md (AC1-AC7)
//!
//! This test suite validates the two-tier re-execution mechanism:
//! - Fast path: exec() with current_exe() path (Unix only, zero overhead)
//! - Fallback path: cargo run when binary unavailable (all platforms)
//!
//! The robust re-exec implementation ensures automatic C++ backend installation
//! works reliably even when the rebuilt binary is temporarily unavailable due
//! to race conditions or filesystem inconsistencies.
//!
//! **Architecture**:
//! ```text
//! reexec_current_command(original_args)
//!   ├─ Unix Fast Path:
//!   │  ├─ Get current_exe()
//!   │  ├─ Check Path::exists()
//!   │  ├─ Log diagnostics
//!   │  ├─ Try exec() (never returns on success)
//!   │  └─ On ENOENT → Fall to fallback
//!   │
//!   └─ Fallback Path (all platforms):
//!      ├─ Build cargo run command
//!      ├─ Add -p xtask --features crossval-all
//!      ├─ Add original arguments
//!      ├─ Set BITNET_REPAIR_PARENT=1
//!      ├─ Spawn child process
//!      └─ Exit with child's code
//! ```
//!
//! **Test Strategy**:
//! - Feature-gated with `#[cfg(feature = "crossval-all")]`
//! - Serial execution with `#[serial(bitnet_env)]` for env-mutating tests
//! - TDD scaffolding: Tests compile but fail with `todo!()` until implementation
//! - Platform coverage: Unix (exec) and Windows (spawn)
//! - Mock subprocess invocations to avoid actual re-exec during tests
//!
//! **Traceability**: Each test references its acceptance criterion (AC1-AC7)
//! for easy spec-to-test mapping and coverage verification.

#![cfg(feature = "crossval-all")]

use serial_test::serial;

// ============================================================================
// Unit Tests: Core Functionality
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// Tests feature spec: reexec-cargo-fallback-robust.md#AC3
    /// AC:AC3 - All CLI arguments preserved across re-exec
    ///
    /// **Given**: Original command with multiple CLI arguments
    /// **When**: reexec_current_command() is invoked
    /// **Then**: All arguments (except program name) are passed to fallback
    ///
    /// **Implementation Note**: Arguments are preserved via:
    /// 1. Fast path: cmd.args(original_args) for exec()
    /// 2. Fallback path: cmd.args(&original_args[1..]) for cargo run
    ///    (Skip first arg since cargo run adds program name)
    ///
    /// **Related**: FR3 - Preserve all CLI arguments across re-exec
    #[test]
    #[serial(bitnet_env)]
    fn test_reexec_preserves_arguments() {
        // **Test Strategy**:
        // 1. Create original_args with complex CLI arguments
        // 2. Mock reexec_current_command() to capture Command built
        // 3. Verify fast path: args match original_args exactly
        // 4. Verify fallback path: args match original_args[1..] (skip program name)
        //
        // **Example Arguments**:
        // original_args = ["xtask", "preflight", "--backend", "bitnet", "--repair=auto"]
        //
        // Fast path Command should have:
        //   args: ["xtask", "preflight", "--backend", "bitnet", "--repair=auto"]
        //
        // Fallback path Command should have:
        //   cargo run -p xtask --features crossval-all -- preflight --backend bitnet --repair=auto
        //   (Note: "xtask" program name is skipped, cargo run adds it)
        //
        // **Mock Strategy**:
        // - Can't test exec() directly (would replace process)
        // - Instead, test argument construction logic separately
        // - Integration test will validate end-to-end behavior

        todo!("AC3: Implement after reexec_current_command() is available");

        // Pseudo-implementation:
        // let original_args = vec![
        //     "xtask".to_string(),
        //     "preflight".to_string(),
        //     "--backend".to_string(),
        //     "bitnet".to_string(),
        //     "--repair=auto".to_string(),
        // ];
        //
        // // Mock: Capture Command built by reexec_current_command
        // let fallback_cmd = build_fallback_command(&original_args);
        //
        // // Verify fallback command structure
        // assert_eq!(fallback_cmd.get_program(), "cargo");
        // assert_eq!(fallback_cmd.get_args(), &[
        //     "run",
        //     "-p",
        //     "xtask",
        //     "--features",
        //     "crossval-all",
        //     "--",
        //     "preflight",
        //     "--backend",
        //     "bitnet",
        //     "--repair=auto",
        // ]);
    }

    /// Tests feature spec: reexec-cargo-fallback-robust.md#AC4
    /// AC:AC4 - BITNET_REPAIR_PARENT guard prevents infinite loops
    ///
    /// **Given**: Re-execution triggered by auto-repair
    /// **When**: Child process checks BITNET_REPAIR_PARENT
    /// **Then**: Guard is set to "1", preventing recursive repair
    ///
    /// **Implementation Note**: The guard is set in both paths:
    /// - Fast path: cmd.env("BITNET_REPAIR_PARENT", "1") before exec()
    /// - Fallback path: cmd.env("BITNET_REPAIR_PARENT", "1") before spawn()
    ///
    /// **Related**: FR4 - Preserve BITNET_REPAIR_PARENT environment variable
    #[test]
    #[serial(bitnet_env)]
    fn test_reexec_guard_prevents_infinite_loop() {
        // **Test Strategy**:
        // 1. Verify BITNET_REPAIR_PARENT=1 is set in Command environment
        // 2. Verify is_repair_parent() returns true when guard is set
        // 3. Verify repair logic is skipped when guard is detected
        //
        // **Recursion Prevention Flow**:
        // Parent process (initial repair):
        //   1. Detects backend missing
        //   2. Invokes setup-cpp-auto
        //   3. Rebuilds xtask
        //   4. Calls reexec_current_command() with BITNET_REPAIR_PARENT=1
        //
        // Child process (re-executed):
        //   1. Checks is_repair_parent() → returns true
        //   2. Skips repair logic
        //   3. Only validates backend detection
        //   4. Reports success or failure
        //   5. Never triggers recursive repair
        //
        // **Mock Strategy**:
        // - Mock is_repair_parent() check
        // - Verify repair flow is skipped when guard is set

        todo!("AC4: Implement after is_repair_parent() is available");

        // Pseudo-implementation:
        // use tests::support::env_guard::EnvGuard;
        //
        // // Test with guard set (child process scenario)
        // {
        //     let _guard = EnvGuard::new("BITNET_REPAIR_PARENT");
        //     _guard.set("1");
        //
        //     assert!(is_repair_parent(), "Guard should be detected");
        //
        //     // Mock: Verify repair logic is skipped
        //     let should_repair = !is_repair_parent();
        //     assert!(!should_repair, "Repair should be skipped when guard is set");
        // }
        //
        // // Test without guard (parent process scenario)
        // {
        //     let _guard = EnvGuard::new("BITNET_REPAIR_PARENT");
        //     _guard.remove();
        //
        //     assert!(!is_repair_parent(), "Guard should not be detected");
        //
        //     // Mock: Verify repair logic can proceed
        //     let should_repair = !is_repair_parent();
        //     assert!(should_repair, "Repair should proceed without guard");
        // }
    }

    /// Tests feature spec: reexec-cargo-fallback-robust.md#AC7
    /// AC:AC7 - Exit code propagated correctly from spawned process
    ///
    /// **Given**: Fallback path spawns cargo run child process
    /// **When**: Child exits with specific code (0, 1, or custom)
    /// **Then**: Parent exits with same code
    ///
    /// **Implementation Note**: Exit code propagation:
    /// ```rust
    /// match cmd.status() {
    ///     Ok(status) => {
    ///         let code = status.code().unwrap_or(1);
    ///         eprintln!("[reexec] Fallback child exited with code: {}", code);
    ///         std::process::exit(code);  // AC7: Propagate exit code
    ///     }
    ///     Err(e) => Err(RepairError::Unknown { ... })
    /// }
    /// ```
    ///
    /// **Related**: FR7 - Exit with correct code on fallback spawn failure
    #[test]
    #[serial(bitnet_env)]
    fn test_reexec_exit_code_propagation() {
        // **Test Strategy**:
        // 1. Mock cmd.status() to return ExitStatus with specific code
        // 2. Verify parent calls std::process::exit(code)
        // 3. Test multiple exit codes:
        //    - 0 (success)
        //    - 1 (generic failure)
        //    - 3 (network failure, retryable)
        //    - 4 (build failure)
        //
        // **Challenge**: Can't directly test exit() in unit test (terminates process)
        //
        // **Solutions**:
        // a) Integration test: Spawn subprocess and check its exit code
        // b) Extract exit code logic into testable function:
        //    fn get_exit_code(status: ExitStatus) -> i32
        // c) Mock exit() call verification (track that exit was called with code)

        todo!("AC7: Implement after reexec_current_command() is available");

        // Pseudo-implementation:
        // use std::process::ExitStatus;
        //
        // // Mock ExitStatus with specific code
        // fn mock_exit_status(code: i32) -> ExitStatus {
        //     // Platform-specific: ExitStatus construction is tricky
        //     // Use Command::new("true").status() and modify
        //     unimplemented!("Requires platform-specific ExitStatus mocking")
        // }
        //
        // // Test exit code 0 (success)
        // let status = mock_exit_status(0);
        // let code = status.code().unwrap_or(1);
        // assert_eq!(code, 0, "Success code should be 0");
        //
        // // Test exit code 3 (network failure)
        // let status = mock_exit_status(3);
        // let code = status.code().unwrap_or(1);
        // assert_eq!(code, 3, "Network failure code should be 3");
        //
        // // Test None case (signal termination on Unix)
        // let code = None.unwrap_or(1);
        // assert_eq!(code, 1, "Signal termination should default to 1");
    }

    /// Tests feature spec: reexec-cargo-fallback-robust.md#AC5
    /// AC:AC5 - Diagnostic logging shows resolved path + existence
    ///
    /// **Given**: Re-execution triggered by auto-repair
    /// **When**: Fast path is attempted (Unix)
    /// **Then**: Diagnostic logs show:
    ///   - [reexec] Fast path: /path/to/xtask
    ///   - [reexec] Binary exists: YES/NO
    ///   - [reexec] Attempting exec()... OR Binary doesn't exist, skipping exec()
    ///   - [reexec] Fast path failed: <error> (if exec fails)
    ///
    /// **When**: Fallback path is used
    /// **Then**: Diagnostic logs show:
    ///   - [reexec] Trying cargo run fallback...
    ///   - [reexec] Fallback command: cargo run -p xtask --features crossval-all -- [args]
    ///   - [reexec] Fallback child exited with code: <code>
    ///
    /// **Related**: FR5 - Log diagnostic information before exec attempts
    #[test]
    #[serial(bitnet_env)]
    fn test_reexec_diagnostic_logging() {
        // **Test Strategy**:
        // 1. Capture stderr output during reexec_current_command()
        // 2. Verify diagnostic messages are present
        // 3. Verify message format matches specification
        //
        // **Diagnostic Output Examples**:
        //
        // Fast path success (Unix):
        //   [repair] Re-executing with updated detection...
        //   [reexec] Fast path: /home/user/BitNet-rs/target/debug/xtask
        //   [reexec] Binary exists: YES
        //   [reexec] Attempting exec()...
        //   (process replaced, no further output)
        //
        // Fallback success (Unix, binary missing):
        //   [repair] Re-executing with updated detection...
        //   [reexec] Fast path: /home/user/BitNet-rs/target/debug/xtask
        //   [reexec] Binary exists: NO
        //   [reexec] Binary doesn't exist, skipping exec()
        //   [reexec] Trying cargo run fallback...
        //   [reexec] Fallback command: cargo run -p xtask --features crossval-all -- ["preflight", "--backend", "bitnet"]
        //   [reexec] Fallback child exited with code: 0
        //
        // **Mock Strategy**:
        // - Redirect stderr to buffer
        // - Call reexec_current_command() in subprocess
        // - Verify stderr contains expected diagnostic messages

        todo!("AC5: Implement after reexec_current_command() is available");

        // Pseudo-implementation:
        // use std::io::Write;
        // use std::sync::Mutex;
        //
        // // Capture stderr output
        // let stderr_buffer = Arc::new(Mutex::new(Vec::new()));
        // let stderr_clone = stderr_buffer.clone();
        //
        // // Mock stderr writer
        // struct MockStderr(Arc<Mutex<Vec<u8>>>);
        // impl Write for MockStderr {
        //     fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        //         self.0.lock().unwrap().write(buf)
        //     }
        //     fn flush(&mut self) -> std::io::Result<()> {
        //         self.0.lock().unwrap().flush()
        //     }
        // }
        //
        // // Call reexec_current_command() with mock stderr
        // // (Requires refactoring to accept custom stderr)
        //
        // // Verify diagnostic messages
        // let stderr_output = String::from_utf8(stderr_buffer.lock().unwrap().clone()).unwrap();
        // assert!(stderr_output.contains("[repair] Re-executing with updated detection..."));
        // assert!(stderr_output.contains("[reexec] Fast path:") || stderr_output.contains("[reexec] Trying cargo run fallback..."));
    }
}

// ============================================================================
// Integration Tests: End-to-End Scenarios
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Tests feature spec: reexec-cargo-fallback-robust.md#AC1
    /// AC:AC1 - Fast path uses exec() when binary exists
    ///
    /// **Given**: Unix platform with valid xtask binary
    /// **When**: reexec_current_command() is called
    /// **Then**: Fast path attempts exec() (process replacement)
    ///
    /// **Platform Note**: This test is Unix-only due to exec() availability
    ///
    /// **Related**: FR1 - Try exec() with current_exe() path first (fast path)
    #[test]
    #[cfg(unix)]
    #[ignore = "TODO: Requires subprocess harness to avoid replacing test process"]
    #[serial(bitnet_env)]
    fn test_reexec_fast_path_with_existing_binary() {
        // **Test Strategy**:
        // 1. Build xtask binary in known location
        // 2. Verify binary exists and is executable
        // 3. Fork subprocess to test exec() behavior
        // 4. Verify parent process is replaced (never returns from exec)
        //
        // **Challenge**: exec() replaces current process
        // **Solution**: Use fork() or subprocess wrapper to isolate test
        //
        // **Subprocess Harness**:
        // ```bash
        // # Parent test process
        // cargo build -p xtask --features crossval-all
        // test_binary=$(find target/debug -name xtask -type f)
        //
        // # Spawn child to test exec
        // child_pid=$(fork_test_process)
        //   # Child process:
        //   original_args = ["xtask", "preflight", "--backend", "bitnet"]
        //   reexec_current_command(&original_args);
        //   # If exec() succeeds, this line never reached
        //   unreachable!("exec() should have replaced process")
        //
        // # Parent verifies child was replaced by exec
        // wait_for_child(child_pid)
        // assert_child_replaced_successfully()
        // ```

        todo!("AC1: Implement with subprocess harness to test exec() behavior");

        // Pseudo-implementation:
        // use std::os::unix::process::CommandExt;
        // use std::process::Stdio;
        //
        // // Build xtask binary
        // let build_status = Command::new("cargo")
        //     .args(&["build", "-p", "xtask", "--features", "crossval-all"])
        //     .status()
        //     .expect("Failed to build xtask");
        // assert!(build_status.success(), "Build should succeed");
        //
        // // Find xtask binary
        // let xtask_binary = PathBuf::from("target/debug/xtask");
        // assert!(xtask_binary.exists(), "xtask binary should exist after build");
        //
        // // Fork subprocess to test exec
        // // (Requires unsafe fork() or Command spawn with verification)
        // let mut child = Command::new(&xtask_binary)
        //     .arg("preflight")
        //     .arg("--backend")
        //     .arg("bitnet")
        //     .env("BITNET_REPAIR_PARENT", "1")  // Prevent repair in test
        //     .stdout(Stdio::piped())
        //     .stderr(Stdio::piped())
        //     .spawn()
        //     .expect("Failed to spawn test subprocess");
        //
        // // Verify child process runs successfully (exec succeeded)
        // let output = child.wait_with_output().expect("Failed to wait for child");
        // assert!(output.status.success(), "Child should exit successfully after exec");
    }

    /// Tests feature spec: reexec-cargo-fallback-robust.md#AC2
    /// AC:AC2 - Fallback activation when current_exe() fails
    ///
    /// **Given**: Binary path unavailable (current_exe() fails or binary missing)
    /// **When**: reexec_current_command() is called
    /// **Then**: Fallback path uses cargo run
    ///
    /// **Simulation**: Delete binary after rebuild to trigger fallback
    ///
    /// **Related**: FR2 - Fall back to cargo run when binary unavailable
    #[test]
    #[ignore = "TODO: Requires subprocess harness to test fallback without terminating test"]
    #[serial(bitnet_env)]
    fn test_reexec_fallback_when_current_exe_fails() {
        // **Test Strategy**:
        // 1. Build xtask binary
        // 2. Simulate missing binary (delete or move after current_exe())
        // 3. Call reexec_current_command() in subprocess
        // 4. Verify cargo run fallback is used
        // 5. Verify child process exits successfully
        //
        // **Fallback Trigger Conditions**:
        // - current_exe() returns Err
        // - current_exe() succeeds but path.exists() returns false
        // - exec() fails with ENOENT (No such file or directory)
        //
        // **Diagnostic Output Expected**:
        //   [reexec] Fast path: /path/to/xtask
        //   [reexec] Binary exists: NO
        //   [reexec] Binary doesn't exist, skipping exec()
        //   [reexec] Trying cargo run fallback...
        //   [reexec] Fallback command: cargo run -p xtask --features crossval-all -- [args]
        //   [reexec] Fallback child exited with code: 0

        todo!("AC2: Implement with subprocess harness to test fallback path");

        // Pseudo-implementation:
        // use std::fs;
        //
        // // Build xtask binary
        // let build_status = Command::new("cargo")
        //     .args(&["build", "-p", "xtask", "--features", "crossval-all"])
        //     .status()
        //     .expect("Failed to build xtask");
        // assert!(build_status.success());
        //
        // // Find and verify binary exists
        // let xtask_binary = PathBuf::from("target/debug/xtask");
        // assert!(xtask_binary.exists(), "Binary should exist after build");
        //
        // // Move binary to simulate unavailability
        // let backup_path = PathBuf::from("target/debug/xtask.backup");
        // fs::rename(&xtask_binary, &backup_path)
        //     .expect("Failed to move binary");
        //
        // // Verify binary no longer exists
        // assert!(!xtask_binary.exists(), "Binary should be unavailable");
        //
        // // Spawn subprocess that will trigger fallback
        // let mut child = Command::new("cargo")
        //     .args(&[
        //         "run",
        //         "-p",
        //         "xtask",
        //         "--features",
        //         "crossval-all",
        //         "--",
        //         "preflight",
        //         "--backend",
        //         "bitnet",
        //     ])
        //     .env("BITNET_REPAIR_PARENT", "1")  // Prevent repair
        //     .stdout(Stdio::piped())
        //     .stderr(Stdio::piped())
        //     .spawn()
        //     .expect("Failed to spawn cargo run");
        //
        // // Verify fallback succeeds
        // let output = child.wait_with_output().expect("Failed to wait for child");
        // assert!(output.status.success(), "Fallback should succeed");
        //
        // // Restore binary for other tests
        // fs::rename(&backup_path, &xtask_binary)
        //     .expect("Failed to restore binary");
    }

    /// Tests feature spec: reexec-cargo-fallback-robust.md#AC4
    /// AC:AC4 - Recursion prevention (re-exec child detection)
    ///
    /// **Given**: BITNET_REPAIR_PARENT=1 is set by parent
    /// **When**: Child process starts and checks is_repair_parent()
    /// **Then**: Child skips repair and only validates detection
    ///
    /// **Flow**:
    /// ```
    /// Parent (initial repair):
    ///   1. Detect backend missing
    ///   2. Invoke setup-cpp-auto
    ///   3. Rebuild xtask
    ///   4. Call reexec_current_command() with BITNET_REPAIR_PARENT=1
    ///
    /// Child (re-executed):
    ///   1. Check is_repair_parent() → true
    ///   2. Skip repair logic
    ///   3. Only validate backend detection
    ///   4. Exit with success (0) or failure (1-6)
    /// ```
    ///
    /// **Related**: FR4 - Preserve BITNET_REPAIR_PARENT environment variable
    #[test]
    #[ignore = "TODO: Requires end-to-end repair workflow test"]
    #[serial(bitnet_env)]
    fn test_reexec_recursion_guard_prevents_loops() {
        // **Test Strategy**:
        // 1. Set BITNET_REPAIR_PARENT=1 in environment
        // 2. Call preflight with --repair=auto
        // 3. Verify repair logic is skipped (no setup-cpp-auto invocation)
        // 4. Verify only validation is performed
        // 5. Verify process exits without recursive repair
        //
        // **Mock Strategy**:
        // - Mock is_repair_parent() to return true
        // - Mock repair workflow to track invocations
        // - Assert repair is never called when guard is set

        todo!("AC4: Implement after is_repair_parent() and repair workflow are integrated");

        // Pseudo-implementation:
        // use tests::support::env_guard::EnvGuard;
        //
        // // Set recursion guard
        // let _guard = EnvGuard::new("BITNET_REPAIR_PARENT");
        // _guard.set("1");
        //
        // // Mock: Track repair invocations
        // let repair_called = Arc::new(Mutex::new(false));
        // let repair_called_clone = repair_called.clone();
        //
        // // Mock repair function
        // fn mock_repair(repair_called: Arc<Mutex<bool>>) -> Result<(), RepairError> {
        //     *repair_called.lock().unwrap() = true;
        //     Ok(())
        // }
        //
        // // Call preflight with guard set
        // let result = preflight_with_repair(
        //     CppBackend::Bitnet,
        //     RepairMode::Auto,
        //     || mock_repair(repair_called_clone),
        // );
        //
        // // Verify repair was not called
        // assert!(!*repair_called.lock().unwrap(), "Repair should be skipped when guard is set");
        //
        // // Verify validation still performed
        // assert!(result.is_ok() || result.is_err(), "Validation should run");
    }

    /// Tests feature spec: reexec-cargo-fallback-robust.md#AC1-AC7
    /// End-to-end integration: repair → rebuild → re-exec → detection flow
    ///
    /// **Given**: Backend is missing (triggers auto-repair)
    /// **When**: User runs `preflight --backend bitnet --repair=auto`
    /// **Then**: Complete repair workflow succeeds:
    ///   1. Detect backend missing
    ///   2. Invoke setup-cpp-auto (download + build C++)
    ///   3. Rebuild xtask (to detect new libraries)
    ///   4. Re-exec with BITNET_REPAIR_PARENT=1
    ///   5. Child validates detection succeeds
    ///   6. Exit with code 0
    ///
    /// **This test validates ALL acceptance criteria in integration**:
    /// - AC1: Fast path tried first (Unix)
    /// - AC2: Fallback activated if needed
    /// - AC3: Arguments preserved
    /// - AC4: Recursion guard prevents loops
    /// - AC5: Diagnostic logging present
    /// - AC6: Windows uses cargo run consistently
    /// - AC7: Exit code propagated correctly
    #[test]
    #[ignore = "TODO: Requires full repair workflow implementation"]
    #[serial(bitnet_env)]
    fn test_reexec_end_to_end_repair_flow() {
        // **Test Strategy**:
        // 1. Clean C++ backend directories (simulate missing)
        // 2. Run preflight --backend bitnet --repair=auto
        // 3. Verify setup-cpp-auto is invoked
        // 4. Verify xtask rebuild succeeds
        // 5. Verify re-exec occurs with BITNET_REPAIR_PARENT=1
        // 6. Verify child process validates detection
        // 7. Verify exit code is 0 (success)
        // 8. Verify no recursive repair (guard works)
        //
        // **Challenges**:
        // - Requires network access (git clone)
        // - Requires build tools (cmake, ninja)
        // - Takes significant time (multi-minute)
        // - May fail in CI without proper setup
        //
        // **Solution**: Mark as #[ignore] by default, run manually or in
        // dedicated integration test CI job with proper environment.

        todo!("AC1-AC7: Implement comprehensive end-to-end test with full repair workflow");

        // Pseudo-implementation:
        // use std::fs;
        // use std::path::Path;
        // use tests::support::env_guard::EnvGuard;
        //
        // // Clean backend directories (simulate missing)
        // let cpp_dir = dirs::home_dir()
        //     .unwrap()
        //     .join(".cache")
        //     .join("bitnet_cpp");
        // if cpp_dir.exists() {
        //     fs::remove_dir_all(&cpp_dir)
        //         .expect("Failed to clean C++ backend directory");
        // }
        //
        // // Run preflight with auto-repair
        // let mut child = Command::new("cargo")
        //     .args(&[
        //         "run",
        //         "-p",
        //         "xtask",
        //         "--features",
        //         "crossval-all",
        //         "--",
        //         "preflight",
        //         "--backend",
        //         "bitnet",
        //         "--repair=auto",
        //         "--verbose",
        //     ])
        //     .stdout(Stdio::piped())
        //     .stderr(Stdio::piped())
        //     .spawn()
        //     .expect("Failed to spawn preflight");
        //
        // // Wait for completion
        // let output = child.wait_with_output()
        //     .expect("Failed to wait for preflight");
        //
        // // Verify success
        // assert!(output.status.success(), "Preflight should succeed after repair");
        //
        // // Verify diagnostic output contains repair markers
        // let stderr = String::from_utf8_lossy(&output.stderr);
        // assert!(stderr.contains("[repair]"), "Should show repair activity");
        // assert!(stderr.contains("[reexec]"), "Should show re-exec activity");
        // assert!(stderr.contains("BITNET_REPAIR_PARENT"), "Should show recursion guard");
        //
        // // Verify backend is now available
        // let validation_result = Command::new("cargo")
        //     .args(&[
        //         "run",
        //         "-p",
        //         "xtask",
        //         "--features",
        //         "crossval-all",
        //         "--",
        //         "preflight",
        //         "--backend",
        //         "bitnet",
        //     ])
        //     .status()
        //     .expect("Failed to run validation");
        // assert!(validation_result.success(), "Backend should be available after repair");
    }

    /// Tests feature spec: reexec-cargo-fallback-robust.md#AC6
    /// AC:AC6 - Windows uses spawn() pattern consistently
    ///
    /// **Given**: Windows platform (no exec() available)
    /// **When**: reexec_current_command() is called
    /// **Then**: Always uses cargo run fallback (spawn + wait + exit)
    ///
    /// **Platform Behavior**:
    /// - Unix: Fast path (exec) → Fallback (spawn)
    /// - Windows: Fallback only (spawn)
    ///
    /// **Related**: FR6 - Handle both Unix (exec) and Windows (spawn) platforms
    #[test]
    #[cfg(windows)]
    #[ignore = "TODO: Requires Windows CI environment"]
    #[serial(bitnet_env)]
    fn test_reexec_windows_uses_spawn_consistently() {
        // **Test Strategy**:
        // 1. Verify no exec() path is compiled on Windows (#[cfg(unix)])
        // 2. Call reexec_current_command() on Windows
        // 3. Verify cargo run fallback is always used
        // 4. Verify child spawns successfully
        // 5. Verify parent exits with child's code
        //
        // **Windows-Specific Considerations**:
        // - No process replacement (exec not available)
        // - cmd.status() always used (not exec())
        // - Exit code propagation via std::process::exit()
        //
        // **Diagnostic Output Expected**:
        //   [repair] Re-executing with updated detection...
        //   [reexec] Trying cargo run fallback...
        //   [reexec] Fallback command: cargo run -p xtask --features crossval-all -- [args]
        //   [reexec] Fallback child exited with code: 0

        todo!("AC6: Implement Windows-specific spawn test");

        // Pseudo-implementation:
        // use std::process::Stdio;
        //
        // // Build xtask on Windows
        // let build_status = Command::new("cargo")
        //     .args(&["build", "-p", "xtask", "--features", "crossval-all"])
        //     .status()
        //     .expect("Failed to build xtask");
        // assert!(build_status.success());
        //
        // // Spawn subprocess to test Windows behavior
        // let mut child = Command::new("cargo")
        //     .args(&[
        //         "run",
        //         "-p",
        //         "xtask",
        //         "--features",
        //         "crossval-all",
        //         "--",
        //         "preflight",
        //         "--backend",
        //         "bitnet",
        //     ])
        //     .env("BITNET_REPAIR_PARENT", "1")  // Prevent repair
        //     .stdout(Stdio::piped())
        //     .stderr(Stdio::piped())
        //     .spawn()
        //     .expect("Failed to spawn on Windows");
        //
        // // Verify child runs successfully
        // let output = child.wait_with_output()
        //     .expect("Failed to wait for child");
        // assert!(output.status.success(), "Windows spawn should succeed");
        //
        // // Verify diagnostic output
        // let stderr = String::from_utf8_lossy(&output.stderr);
        // assert!(stderr.contains("[reexec] Trying cargo run fallback..."),
        //         "Windows should always use fallback");
        // assert!(!stderr.contains("Fast path"),
        //         "Windows should not attempt fast path");
    }
}

// ============================================================================
// Helper Functions and Mock Infrastructure
// ============================================================================

#[cfg(test)]
mod test_helpers {

    /// Mock helper: Build command arguments for fallback path
    ///
    /// Extracts argument building logic for testability without triggering
    /// actual subprocess spawn or exec.
    ///
    /// **Arguments**:
    /// - `original_args`: Command-line arguments from env::args()
    ///
    /// **Returns**:
    /// Vector of arguments for cargo run command:
    /// ["run", "-p", "xtask", "--features", "crossval-all", "--", <original_args[1..]>]
    fn build_fallback_args(original_args: &[String]) -> Vec<String> {
        let mut args = vec![
            "run".to_string(),
            "-p".to_string(),
            "xtask".to_string(),
            "--features".to_string(),
            "crossval-all".to_string(),
            "--".to_string(),
        ];

        // Skip program name (first arg), preserve rest
        if original_args.len() > 1 {
            args.extend_from_slice(&original_args[1..]);
        }

        args
    }

    #[test]
    fn test_build_fallback_args_helper() {
        // Test with typical preflight command
        let original_args = vec![
            "xtask".to_string(),
            "preflight".to_string(),
            "--backend".to_string(),
            "bitnet".to_string(),
            "--repair=auto".to_string(),
        ];

        let fallback_args = build_fallback_args(&original_args);

        assert_eq!(
            fallback_args,
            vec![
                "run",
                "-p",
                "xtask",
                "--features",
                "crossval-all",
                "--",
                "preflight",
                "--backend",
                "bitnet",
                "--repair=auto",
            ]
        );
    }

    #[test]
    fn test_build_fallback_args_empty() {
        // Test with only program name (no additional args)
        let original_args = vec!["xtask".to_string()];
        let fallback_args = build_fallback_args(&original_args);

        assert_eq!(fallback_args, vec!["run", "-p", "xtask", "--features", "crossval-all", "--",]);
    }

    #[test]
    fn test_build_fallback_args_with_flags() {
        // Test with complex flag combinations
        let original_args = vec![
            "xtask".to_string(),
            "crossval-per-token".to_string(),
            "--model".to_string(),
            "model.gguf".to_string(),
            "--tokenizer".to_string(),
            "tokenizer.json".to_string(),
            "--prompt".to_string(),
            "What is 2+2?".to_string(),
            "--max-tokens".to_string(),
            "4".to_string(),
            "--verbose".to_string(),
        ];

        let fallback_args = build_fallback_args(&original_args);

        assert_eq!(
            fallback_args,
            vec![
                "run",
                "-p",
                "xtask",
                "--features",
                "crossval-all",
                "--",
                "crossval-per-token",
                "--model",
                "model.gguf",
                "--tokenizer",
                "tokenizer.json",
                "--prompt",
                "What is 2+2?",
                "--max-tokens",
                "4",
                "--verbose",
            ]
        );
    }
}
