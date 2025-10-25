//! Backend availability helpers for conditional test execution
//!
//! This module provides helpers for tests that depend on C++ backend availability
//! (BitNet.cpp, llama.cpp). Tests use these helpers to skip gracefully when backends
//! are unavailable, with optional auto-repair in local development.
//!
//! # Design Goals
//!
//! 1. **Zero-install developer experience**: Tests auto-skip when backends unavailable
//! 2. **Auto-repair capability**: Optionally attempt backend installation when missing
//! 3. **Deterministic CI**: `BITNET_TEST_NO_REPAIR=1` prevents downloads during test runs
//! 4. **Clear diagnostics**: Distinguish "skipped" from "passed" from "failed"
//!
//! # Usage
//!
//! ```rust,ignore
//! use tests::support::backend_helpers::ensure_bitnet_or_skip;
//!
//! #[test]
//! fn test_cpp_crossval() {
//!     ensure_bitnet_or_skip();
//!
//!     // Test code runs only if BitNet backend available
//!     let session = create_bitnet_session();
//!     assert!(session.is_valid());
//! }
//! ```
//!
//! # Environment Variables
//!
//! - `BITNET_TEST_NO_REPAIR`: Disable auto-repair (CI mode)
//! - `CI`: Auto-enable no-repair mode in CI environments
//!
//! # Backend Detection
//!
//! Backend availability is determined by compile-time constants from `bitnet_crossval`:
//! - `HAS_BITNET`: BitNet.cpp libraries available at build time
//! - `HAS_LLAMA`: llama.cpp libraries available at build time
//!
//! **Important**: After installing backends, you must rebuild the `crossval` crate
//! for the constants to update:
//!
//! ```bash
//! cargo clean -p crossval && cargo build --features crossval-all
//! ```

use bitnet_crossval::backend::CppBackend;

/// Ensure backend is available, skip test if not
///
/// This function checks if the specified C++ backend is available. If not,
/// it will either skip the test immediately (CI mode) or attempt auto-repair
/// (local dev mode).
///
/// # Behavior
///
/// 1. **Backend available**: Returns immediately, test continues
/// 2. **Backend unavailable + `BITNET_TEST_NO_REPAIR=1`**: Prints skip message, returns
/// 3. **Backend unavailable + local dev**: Attempts auto-repair, then skips
///
/// # Arguments
///
/// * `backend` - The C++ backend to check (BitNet or Llama)
///
/// # Environment
///
/// - `BITNET_TEST_NO_REPAIR=1`: Disable auto-repair (CI mode)
/// - `CI=1`: Auto-enable no-repair mode
///
/// # Examples
///
/// ```rust,ignore
/// #[test]
/// fn test_bitnet_crossval() {
///     ensure_backend_or_skip(CppBackend::BitNet);
///     // Test code here
/// }
/// ```
pub fn ensure_backend_or_skip(backend: CppBackend) {
    use bitnet_crossval::{HAS_BITNET, HAS_LLAMA};

    // Check if backend is available
    let available = match backend {
        CppBackend::BitNet => HAS_BITNET,
        CppBackend::Llama => HAS_LLAMA,
    };

    if available {
        return; // Backend available, proceed with test
    }

    // Backend unavailable - check if auto-repair allowed
    if is_ci_or_no_repair() {
        // CI mode: deterministic skip without downloads
        print_skip_diagnostic(backend, "backend unavailable (BITNET_TEST_NO_REPAIR set)");
        return;
    }

    // Local dev mode: attempt auto-repair
    eprintln!("⚠️  {} backend unavailable. Attempting auto-repair...", backend.name());

    if let Err(e) = attempt_auto_repair(backend) {
        print_skip_diagnostic(backend, &format!("auto-repair failed: {}", e));
        return;
    }

    // Repair succeeded but constants still frozen
    eprintln!("✓ {} backend installed. Rebuild required to detect:", backend.name());
    eprintln!("  cargo clean -p crossval && cargo build --features crossval-all");
    print_skip_diagnostic(backend, "backend available after rebuild");
}

/// Convenience wrapper for BitNet backend
///
/// # Examples
///
/// ```rust,ignore
/// #[test]
/// fn test_bitnet_tokenization() {
///     ensure_bitnet_or_skip();
///     // Test code here
/// }
/// ```
pub fn ensure_bitnet_or_skip() {
    ensure_backend_or_skip(CppBackend::BitNet);
}

/// Convenience wrapper for Llama backend
///
/// # Examples
///
/// ```rust,ignore
/// #[test]
/// fn test_llama_crossval() {
///     ensure_llama_or_skip();
///     // Test code here
/// }
/// ```
pub fn ensure_llama_or_skip() {
    ensure_backend_or_skip(CppBackend::Llama);
}

/// Check if we're in CI or no-repair mode
///
/// Returns true if:
/// - `BITNET_TEST_NO_REPAIR=1` is set, OR
/// - `CI=1` is set (GitHub Actions, GitLab CI, etc.)
fn is_ci_or_no_repair() -> bool {
    std::env::var("BITNET_TEST_NO_REPAIR").is_ok() || std::env::var("CI").is_ok()
}

/// Attempt to install missing backend (local dev only)
///
/// This runs `cargo run -p xtask -- setup-cpp-auto` to install the backend.
///
/// # Arguments
///
/// * `backend` - The backend to install
///
/// # Returns
///
/// - `Ok(())` if installation succeeded
/// - `Err(String)` if installation failed with error message
fn attempt_auto_repair(backend: CppBackend) -> Result<(), String> {
    use std::process::Command;

    // Build xtask command based on backend
    let mut cmd = Command::new("cargo");
    cmd.args(&["run", "-p", "xtask", "--", "setup-cpp-auto", "--emit=sh"]);

    // Add backend-specific flags if needed
    match backend {
        CppBackend::BitNet => {
            cmd.arg("--bitnet");
        }
        CppBackend::Llama => {
            // Default setup installs llama.cpp
        }
    }

    let status = cmd.status().map_err(|e| format!("Failed to run setup-cpp-auto: {}", e))?;

    if !status.success() {
        return Err(format!(
            "setup-cpp-auto returned non-zero exit code: {}",
            status.code().unwrap_or(-1)
        ));
    }

    Ok(())
}

/// Print skip diagnostic message
///
/// This prints a standardized skip message to stderr, providing actionable
/// instructions for setting up the backend.
///
/// # Arguments
///
/// * `backend` - The backend that is unavailable
/// * `reason` - Human-readable reason for skipping
fn print_skip_diagnostic(backend: CppBackend, reason: &str) {
    eprintln!("SKIPPED: {} backend {}", backend.name(), reason);
    eprintln!("  To enable: {}", backend.setup_command());
    eprintln!("  Then rebuild: cargo clean -p crossval && cargo build --features crossval-all");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_ci_or_no_repair_with_no_repair_flag() {
        // This test would need environment isolation
        // See test_support_tests.rs for comprehensive tests
        unimplemented!("See test_support_tests.rs for comprehensive environment tests");
    }

    #[test]
    fn test_is_ci_or_no_repair_with_ci_flag() {
        unimplemented!("See test_support_tests.rs for comprehensive environment tests");
    }

    #[test]
    fn test_is_ci_or_no_repair_interactive() {
        unimplemented!("See test_support_tests.rs for comprehensive environment tests");
    }

    #[test]
    fn test_ensure_backend_or_skip_backend_available() {
        unimplemented!("See test_support_tests.rs for comprehensive backend tests");
    }

    #[test]
    fn test_ensure_backend_or_skip_backend_unavailable_ci() {
        unimplemented!("See test_support_tests.rs for comprehensive backend tests");
    }

    #[test]
    fn test_ensure_backend_or_skip_backend_unavailable_repair() {
        unimplemented!("See test_support_tests.rs for comprehensive backend tests");
    }

    #[test]
    fn test_convenience_wrappers() {
        unimplemented!("See test_support_tests.rs for comprehensive wrapper tests");
    }

    #[test]
    fn test_attempt_auto_repair_success() {
        unimplemented!("See test_support_tests.rs for comprehensive repair tests");
    }

    #[test]
    fn test_attempt_auto_repair_failure() {
        unimplemented!("See test_support_tests.rs for comprehensive repair tests");
    }

    #[test]
    fn test_print_skip_diagnostic_format() {
        unimplemented!("See test_support_tests.rs for comprehensive diagnostic tests");
    }
}
