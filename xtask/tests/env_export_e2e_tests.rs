//! End-to-end tests for deterministic auto-repair with environment propagation
//!
//! **Specification**: docs/specs/env-export-before-rebuild-deterministic.md (Version 1.0.0)
//!
//! This test suite validates the complete repair → rebuild → re-exec cycle with
//! environment variable propagation, ensuring deterministic detection of C++ libraries
//! after successful auto-repair.
//!
//! **Acceptance Criteria Coverage (2 tests)**:
//! - AC4: Persistent Detection After Re-exec (2 tests)
//!   - Full repair → rebuild → re-exec flow succeeds
//!   - Re-exec child reports HAS_BITNET=true (no BITNET_STUB mode)
//!
//! **Test Strategy**:
//! - Feature-gated with `#[cfg(feature = "crossval-all")]`
//! - Serial execution with `#[serial(bitnet_env)]` for environment isolation
//! - **MARKED #[ignore]**: Requires real setup-cpp-auto and cargo build
//! - Manual execution only: `cargo test --features crossval-all -- --ignored`
//! - Uses EnvGuard for automatic cleanup
//! - TDD scaffolding: Tests compile but fail with `todo!()` until implementation
//! - Spec references: `/// Tests spec: env-export-before-rebuild-deterministic.md#AC4`
//!
//! **Traceability**: Each test validates the complete auto-repair cycle produces
//! a correctly-built xtask binary with HAS_BITNET=true constant.

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
mod e2e_determinism_tests {
    use super::*;

    // TODO: Import functions once implemented
    // use crate::crossval::preflight::{preflight_with_auto_repair, CppBackend, RepairMode};
    // use std::process::Command;

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC4
    /// AC:AC4 - Deterministic detection after repair (bitnet.cpp)
    ///
    /// **Given**: Clean state with no C++ libraries initially detected
    /// **When**: Running complete auto-repair cycle (repair → rebuild → re-exec)
    /// **Then**: Re-exec child reports HAS_BITNET=true and "✓ bitnet.cpp AVAILABLE"
    ///
    /// **Test steps**:
    /// 1. Clean environment (remove BITNET_CPP_DIR if set)
    /// 2. Run `preflight_with_auto_repair(CppBackend::BitNet, verbose, RepairMode::Auto)`
    /// 3. Verify repair succeeded (exit code 0)
    /// 4. Verify rebuild succeeded with environment variables applied
    /// 5. Verify re-exec child detects backend as available
    /// 6. Check HAS_BITNET constant in rebuilt binary
    /// 7. Verify no "BITNET_STUB mode" warnings in cargo output
    ///
    /// **Expected behavior**:
    /// - Repair phase: `setup-cpp-auto` installs libraries, outputs exports
    /// - Parse phase: Exports captured and parsed from stdout
    /// - Rebuild phase: Cargo receives BITNET_CPP_DIR environment
    /// - Build.rs detection: crossval/build.rs finds libraries via BITNET_CPP_DIR
    /// - Re-exec child: Reports "✓ bitnet.cpp AVAILABLE (detected after repair)"
    /// - Cargo output: Contains "✓ BITNET_FULL: BitNet.cpp and llama.cpp libraries found"
    /// - No warnings: No "BITNET_STUB mode" or "Set BITNET_CPP_DIR" messages
    ///
    /// **Success metrics**:
    /// - Zero BITNET_STUB warnings after successful repair
    /// - HAS_BITNET=true in rebuilt xtask binary
    /// - Deterministic: Same result on repeated runs
    ///
    /// **Coverage**: Validates complete fix for environment export propagation gap
    /// and deterministic auto-repair workflow (Issue #P0-1).
    #[test]
    #[ignore = "Requires real setup-cpp-auto and cargo build - manual execution only"]
    #[serial(bitnet_env)]
    fn test_deterministic_available_after_repair_bitnet() {
        let _guard = EnvGuard::new("BITNET_CPP_DIR");

        // TODO: Test complete repair cycle for bitnet.cpp backend
        // Setup:
        //   - Remove BITNET_CPP_DIR from environment (clean state)
        //   - Ensure bitnet.cpp not initially detected
        // Execute:
        //   - Run preflight_with_auto_repair(
        //         backend: CppBackend::BitNet,
        //         verbose: true,
        //         repair_mode: RepairMode::Auto
        //     )
        // Assert (Repair phase):
        //   - Result is Ok (no error)
        //   - setup-cpp-auto executed successfully
        //   - Stdout contains export statements
        // Assert (Rebuild phase):
        //   - rebuild_xtask_with_env() called with parsed exports
        //   - Cargo build output contains "✓ BITNET_FULL"
        //   - No "BITNET_STUB mode" warning in cargo output
        // Assert (Re-exec phase):
        //   - Re-exec child reports "✓ bitnet.cpp AVAILABLE"
        //   - HAS_BITNET constant = true in rebuilt binary
        //   - Backend status shows "available (auto-repaired)"
        // Assert (Determinism):
        //   - Run preflight again (should now show available, no repair)
        //   - Same result on repeated runs
        // Note: This is the PRIMARY validation for deterministic auto-repair
        //       fixing the environment export propagation gap
        todo!("AC4: Implement deterministic bitnet.cpp repair E2E test");
    }

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC4
    /// AC:AC4 - Deterministic detection after repair (llama.cpp)
    ///
    /// **Given**: Clean state with no llama.cpp libraries initially detected
    /// **When**: Running complete auto-repair cycle (repair → rebuild → re-exec)
    /// **Then**: Re-exec child reports HAS_LLAMA=true and "✓ llama.cpp AVAILABLE"
    ///
    /// **Test steps**:
    /// 1. Clean environment (remove BITNET_CROSSVAL_LIBDIR if set)
    /// 2. Run `preflight_with_auto_repair(CppBackend::Llama, verbose, RepairMode::Auto)`
    /// 3. Verify repair succeeded (exit code 0)
    /// 4. Verify rebuild succeeded with environment variables applied
    /// 5. Verify re-exec child detects llama.cpp as available
    /// 6. Check HAS_LLAMA constant in rebuilt binary
    /// 7. Verify no "BITNET_STUB mode" warnings for llama backend
    ///
    /// **Expected behavior**:
    /// - Repair phase: `setup-cpp-auto` installs llama.cpp, outputs exports
    /// - Parse phase: Exports captured (BITNET_CROSSVAL_LIBDIR)
    /// - Rebuild phase: Cargo receives library paths
    /// - Build.rs detection: crossval/build.rs finds libllama.so, libggml.so
    /// - Re-exec child: Reports "✓ llama.cpp AVAILABLE"
    /// - Cargo output: Contains "Linked libraries: llama, ggml"
    ///
    /// **Success metrics**:
    /// - HAS_LLAMA=true in rebuilt xtask binary
    /// - llama.cpp backend shows as available
    /// - Deterministic: Same result on repeated runs
    ///
    /// **Coverage**: Validates dual-backend support (bitnet.cpp + llama.cpp)
    /// with correct environment variable handling for both backends.
    #[test]
    #[ignore = "Requires real setup-cpp-auto and cargo build - manual execution only"]
    #[serial(bitnet_env)]
    fn test_deterministic_available_after_repair_llama() {
        let _guard1 = EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
        let _guard2 = EnvGuard::new("LD_LIBRARY_PATH");

        // TODO: Test complete repair cycle for llama.cpp backend
        // Setup:
        //   - Remove BITNET_CROSSVAL_LIBDIR from environment (clean state)
        //   - Ensure llama.cpp not initially detected
        // Execute:
        //   - Run preflight_with_auto_repair(
        //         backend: CppBackend::Llama,
        //         verbose: true,
        //         repair_mode: RepairMode::Auto
        //     )
        // Assert (Repair phase):
        //   - Result is Ok (no error)
        //   - setup-cpp-auto executed for llama backend
        //   - Stdout contains BITNET_CROSSVAL_LIBDIR export
        // Assert (Rebuild phase):
        //   - rebuild_xtask_with_env() receives llama library paths
        //   - Cargo build output shows llama.cpp detection
        //   - build.rs finds libllama.so and libggml.so
        // Assert (Re-exec phase):
        //   - Re-exec child reports "✓ llama.cpp AVAILABLE"
        //   - HAS_LLAMA constant = true in rebuilt binary
        //   - Backend status shows "available (auto-repaired)"
        // Assert (Determinism):
        //   - Second preflight run shows available (no re-repair)
        //   - Environment variables persist across runs
        // Note: This validates dual-backend auto-repair capability with
        //       correct environment variable handling for llama.cpp
        todo!("AC4: Implement deterministic llama.cpp repair E2E test");
    }
}
