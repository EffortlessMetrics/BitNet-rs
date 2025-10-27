//! Integration tests for repair flow with environment propagation (env-export-before-rebuild-deterministic.md)
//!
//! **Specification**: docs/specs/env-export-before-rebuild-deterministic.md (Version 1.0.0)
//!
//! This test suite validates the complete integration of environment export parsing
//! and application within the auto-repair workflow. These tests ensure that exports
//! from `setup-cpp-auto` are captured, parsed, and applied before rebuilding xtask.
//!
//! **Acceptance Criteria Coverage (2 tests)**:
//! - AC3: Integration with Repair Flow (2 tests)
//!   - `attempt_repair_once()` returns stdout containing exports
//!   - `preflight_with_auto_repair()` applies exports before rebuild
//!
//! **Test Strategy**:
//! - Feature-gated with `#[cfg(feature = "crossval-all")]`
//! - Serial execution with `#[serial(bitnet_env)]` for environment isolation
//! - Uses EnvGuard for automatic cleanup
//! - TDD scaffolding: Tests compile but fail with `todo!()` until implementation
//! - Spec references: `/// Tests spec: env-export-before-rebuild-deterministic.md#AC3`
//!
//! **Traceability**: Each test validates the integration points in the repair workflow
//! where environment exports are captured, parsed, and applied.

#![cfg(feature = "crossval-all")]

use serial_test::serial;
use std::env;

// ============================================================================
// Test Helpers - Local EnvGuard Implementation
// ============================================================================

/// RAII guard for environment variable management
///
/// Automatically restores environment variable state on drop, ensuring test isolation.
#[allow(dead_code)]
struct EnvGuard {
    key: String,
    old: Option<String>,
}

#[allow(dead_code)]
impl EnvGuard {
    /// Create a new environment variable guard, capturing current state
    fn new(key: &str) -> Self {
        let old = env::var(key).ok();
        Self { key: key.to_string(), old }
    }

    /// Clear the environment variable and return guard
    fn clear(key: &str) -> Self {
        let guard = Self::new(key);
        unsafe {
            env::remove_var(key);
        }
        guard
    }

    /// Set the environment variable to a new value
    fn set(&self, val: &str) {
        unsafe {
            env::set_var(&self.key, val);
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(ref v) = self.old {
                env::set_var(&self.key, v);
            } else {
                env::remove_var(&self.key);
            }
        }
    }
}

#[cfg(test)]
mod repair_integration_tests {
    use super::*;

    // TODO: Import functions once implemented
    // use crate::crossval::preflight::{attempt_repair_once, preflight_with_auto_repair, CppBackend, RepairMode};

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC3
    /// AC:AC3 - attempt_repair_once() captures and returns stdout
    ///
    /// **Given**: Successful `setup-cpp-auto` execution
    /// **When**: Calling `attempt_repair_once(backend, verbose)`
    /// **Then**: Function returns stdout containing shell export statements
    ///
    /// **Test steps**:
    /// 1. Mock or run actual `setup-cpp-auto --emit=sh`
    /// 2. Call `attempt_repair_once(CppBackend::BitNet, true)`
    /// 3. Verify returned String contains export statements
    /// 4. Verify export format matches expected shell syntax
    ///
    /// **Expected behavior**:
    /// - Function returns `Result<String, RepairError>`
    /// - String contains: `export BITNET_CPP_DIR="/path"`
    /// - String contains: `export LD_LIBRARY_PATH="/path:..."`
    /// - String may contain non-export lines (echo statements) that should be skipped
    ///
    /// **Coverage**: Validates gap fix for stdout capture (previously discarded)
    #[test]
    #[serial(bitnet_env)]
    fn test_attempt_repair_once_captures_stdout() {
        let _guard = EnvGuard::new("BITNET_CPP_DIR");

        // TODO: Test stdout capture from setup-cpp-auto
        // Setup:
        //   - Mock setup-cpp-auto command or use test fixture
        //   - Ensure command outputs export statements to stdout
        // Execute:
        //   - Call attempt_repair_once(CppBackend::BitNet, verbose: true)
        // Assert:
        //   - Result is Ok(String)
        //   - String is non-empty
        //   - String contains "export BITNET_CPP_DIR="
        //   - String contains "export LD_LIBRARY_PATH=" or "export DYLD_LIBRARY_PATH="
        // Note: This validates the return type change from Result<(), RepairError>
        //       to Result<String, RepairError> as specified in the spec
        todo!("AC3: Implement stdout capture test for attempt_repair_once");
    }

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC3
    /// AC:AC3 - preflight_with_auto_repair() applies env before rebuild
    ///
    /// **Given**: Auto-repair flow triggered by missing backend
    /// **When**: Running `preflight_with_auto_repair(backend, verbose, repair_mode)`
    /// **Then**: Environment variables parsed from stdout and applied before rebuild
    ///
    /// **Test steps**:
    /// 1. Trigger auto-repair workflow (backend not found)
    /// 2. Capture repair flow execution
    /// 3. Verify environment parsing occurred
    /// 4. Verify rebuild_xtask_with_env() called with parsed exports
    /// 5. Verify cargo subprocess received environment variables
    ///
    /// **Expected behavior**:
    /// - `setup-cpp-auto` stdout captured (not discarded)
    /// - `parse_sh_exports()` called on stdout
    /// - `apply_env_exports()` sets variables in current process
    /// - `rebuild_xtask_with_env()` passes exports to cargo
    /// - Cargo build.rs sees BITNET_CPP_DIR in environment
    ///
    /// **Coverage**: End-to-end integration of all three repair gaps:
    /// - Gap 1: Stdout capture (not discarded)
    /// - Gap 2: Environment propagation to cargo
    /// - Gap 3: Parsing function exists and is called
    #[test]
    #[serial(bitnet_env)]
    fn test_preflight_with_auto_repair_applies_env_before_rebuild() {
        let _guard1 = EnvGuard::new("BITNET_CPP_DIR");
        let _guard2 = EnvGuard::new("LD_LIBRARY_PATH");

        // TODO: Test complete repair flow with environment propagation
        // Setup:
        //   - Clear BITNET_CPP_DIR to trigger repair mode
        //   - Mock or use real setup-cpp-auto with known output
        // Execute:
        //   - Call preflight_with_auto_repair(
        //         backend: CppBackend::BitNet,
        //         verbose: true,
        //         repair_mode: RepairMode::Auto
        //     )
        // Assert:
        //   - Result is Ok (repair succeeded)
        //   - BITNET_CPP_DIR is now set in environment
        //   - LD_LIBRARY_PATH contains expected path
        //   - Verbose output shows: "[preflight] Rebuilding xtask with environment variables..."
        //   - Verbose output shows applied variables (BITNET_CPP_DIR, LD_LIBRARY_PATH)
        // Note: This is the key integration test validating the complete fix
        //       for the environment export propagation gap
        todo!("AC3: Implement preflight auto-repair environment application test");
    }
}
