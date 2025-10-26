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

#[allow(unused_imports)]
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
#[allow(dead_code)]
pub fn ensure_backend_or_skip(backend: CppBackend) {
    use bitnet_crossval::{HAS_BITNET, HAS_LLAMA};

    // Check build-time constant first
    let build_time_available = match backend {
        CppBackend::BitNet => HAS_BITNET,
        CppBackend::Llama => HAS_LLAMA,
    };

    if build_time_available {
        return; // Backend available at build time, continue test
    }

    // Check runtime detection as fallback (in case libraries installed post-build)
    if let Ok(runtime_available) = detect_backend_runtime(backend) {
        if runtime_available {
            print_rebuild_warning(backend);
            return; // Backend available at runtime, warn about rebuild but continue
        }
    }

    // Backend unavailable - check if we should attempt repair
    if !is_ci_or_no_repair() {
        eprintln!("Attempting auto-repair for {:?} backend...", backend);
        if let Ok(()) = attempt_auto_repair(backend) {
            eprintln!("✓ {:?} backend installed.", backend);
            print_rebuild_instructions(backend);
            return;
        } else {
            eprintln!("Auto-repair failed.");
        }
    }

    // Skip test with diagnostic
    print_skip_diagnostic(backend, None);
    panic!("SKIPPED: {:?} backend unavailable", backend);
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
#[allow(dead_code)]
pub fn ensure_llama_or_skip() {
    ensure_backend_or_skip(CppBackend::Llama);
}

/// Check if we're in CI or no-repair mode
///
/// Returns true if:
/// - `BITNET_TEST_NO_REPAIR=1` is set, OR
/// - `CI=1` is set (GitHub Actions, GitLab CI, etc.)
#[allow(dead_code)]
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
#[allow(dead_code)]
fn attempt_auto_repair(backend: CppBackend) -> Result<(), String> {
    use std::process::Command;

    // Build xtask command based on backend
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "-p", "xtask", "--", "setup-cpp-auto", "--emit=sh"]);

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
/// * `context` - Optional context string (e.g., "CI mode", "auto-repair failed")
#[allow(dead_code)]
fn print_skip_diagnostic(backend: CppBackend, context: Option<&str>) {
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    match backend {
        CppBackend::BitNet => eprintln!("⊘ Test skipped: bitnet.cpp not available"),
        CppBackend::Llama => eprintln!("⊘ Test skipped: llama.cpp not available"),
    }
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if let Some(ctx) = context {
        eprintln!("\nContext: {}", ctx);
    }

    eprintln!("\nThis test requires the {:?} C++ reference backend.", backend);

    eprintln!("\nSetup Instructions:");
    eprintln!("──────────────────────────────────────────────────────────");

    eprintln!("\n  Option A: Auto-setup (recommended)\n");
    eprintln!("    1. Install backend:");
    eprintln!("       eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\"");
    eprintln!("\n    2. Rebuild xtask to update detection:");
    eprintln!("       cargo clean -p crossval && cargo build -p xtask --features crossval-all");
    eprintln!("\n    3. Re-run tests:");
    eprintln!("       cargo test --workspace --no-default-features --features cpu");

    eprintln!("\n  Option B: Manual setup (advanced)\n");
    match backend {
        CppBackend::BitNet => {
            eprintln!("    1. Clone and build BitNet.cpp:");
            eprintln!(
                "       git clone https://github.com/microsoft/BitNet.git ~/.cache/bitnet_cpp"
            );
            eprintln!("       cd ~/.cache/bitnet_cpp && mkdir build && cd build");
            eprintln!("       cmake .. && cmake --build .");
            eprintln!("\n    2. Set environment variables:");
            eprintln!("       export BITNET_CPP_DIR=~/.cache/bitnet_cpp");
            eprintln!(
                "       export LD_LIBRARY_PATH=~/.cache/bitnet_cpp/build/bin:$LD_LIBRARY_PATH"
            );
        }
        CppBackend::Llama => {
            eprintln!("    1. Clone and build llama.cpp:");
            eprintln!(
                "       git clone https://github.com/ggerganov/llama.cpp.git ~/.cache/llama_cpp"
            );
            eprintln!("       cd ~/.cache/llama_cpp && mkdir build && cd build");
            eprintln!("       cmake .. && cmake --build .");
            eprintln!("\n    2. Set environment variables:");
            eprintln!("       export LLAMA_CPP_DIR=~/.cache/llama_cpp");
            eprintln!("       export LD_LIBRARY_PATH=~/.cache/llama_cpp/build:$LD_LIBRARY_PATH");
        }
    }
    eprintln!("\n    3. Rebuild and re-run tests (steps 2-3 from Option A)");

    eprintln!("\nDocumentation: docs/howto/cpp-setup.md");
    eprintln!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

/// Detect backend availability at runtime
///
/// This provides a fallback detection mechanism when libraries were installed
/// after xtask was built (avoiding full rebuild requirement).
///
/// # Arguments
///
/// * `backend` - The backend to detect
///
/// # Returns
///
/// - `Ok(true)` if libraries found via dynamic loader or environment variables
/// - `Ok(false)` if libraries not found
/// - `Err(String)` if detection failed
#[allow(dead_code)]
fn detect_backend_runtime(backend: CppBackend) -> Result<bool, String> {
    // Check environment variable first
    let env_var = match backend {
        CppBackend::BitNet => "BITNET_CPP_DIR",
        CppBackend::Llama => "LLAMA_CPP_DIR",
    };

    if let Ok(dir) = std::env::var(env_var) {
        let path = std::path::Path::new(&dir);
        if path.exists() {
            return Ok(true);
        }
    }

    // Could add more sophisticated runtime detection here
    // (e.g., checking LD_LIBRARY_PATH, searching common install locations)
    // For now, just rely on environment variable
    Ok(false)
}

/// Print rebuild warning when runtime detection differs from build-time
///
/// # Arguments
///
/// * `backend` - The backend that was detected at runtime
#[allow(dead_code)]
fn print_rebuild_warning(backend: CppBackend) {
    eprintln!("⚠️  Backend libraries found at runtime but not at build time.");
    eprintln!("    Rebuild xtask to update detection:");
    eprintln!("    cargo clean -p crossval && cargo build -p xtask --features crossval-all");
    eprintln!("    ({:?} backend)", backend);
}

/// Print rebuild instructions after successful repair
///
/// # Arguments
///
/// * `backend` - The backend that was installed
#[allow(dead_code)]
fn print_rebuild_instructions(backend: CppBackend) {
    eprintln!("✓ Backend installed. Rebuild required to detect:");
    eprintln!("  cargo clean -p crossval && cargo build --features crossval-all");
    eprintln!("  ({:?} backend)", backend);
}

// ============================================================================
// Platform-Specific Helpers (AC7)
// ============================================================================

/// Get platform-specific dynamic loader path variable name
///
/// # Returns
/// - `"LD_LIBRARY_PATH"` on Linux
/// - `"DYLD_LIBRARY_PATH"` on macOS
/// - `"PATH"` on Windows
///
/// # Example
/// ```rust
/// let loader_var = get_loader_path_var();
/// env::set_var(loader_var, "/custom/lib");
/// ```
pub fn get_loader_path_var() -> &'static str {
    if cfg!(target_os = "linux") {
        "LD_LIBRARY_PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else if cfg!(target_os = "windows") {
        "PATH"
    } else {
        panic!("Unsupported platform: {}", std::env::consts::OS)
    }
}

/// Get platform-specific shared library extension
///
/// # Returns
/// - `"so"` on Linux
/// - `"dylib"` on macOS
/// - `"dll"` on Windows
pub fn get_lib_extension() -> &'static str {
    if cfg!(target_os = "linux") {
        "so"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        panic!("Unsupported platform: {}", std::env::consts::OS)
    }
}

/// Format library name with platform-specific prefix/extension
///
/// # Arguments
/// * `stem` - Library name stem (e.g., "bitnet", "llama")
///
/// # Returns
/// - `"libbitnet.so"` on Linux
/// - `"libbitnet.dylib"` on macOS
/// - `"bitnet.dll"` on Windows
///
/// # Example
/// ```rust
/// let lib_name = format_lib_name("bitnet");
/// assert_eq!(lib_name, "libbitnet.so"); // Linux
/// ```
pub fn format_lib_name(stem: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{}.dll", stem)
    } else if cfg!(target_os = "macos") {
        format!("lib{}.dylib", stem)
    } else {
        format!("lib{}.so", stem)
    }
}

// ============================================================================
// Mock Library Creation Helpers (AC5)
// ============================================================================

/// Create mock C++ backend libraries for testing
///
/// This creates empty shared library files with correct platform-specific
/// extensions and naming conventions.
///
/// # Arguments
/// * `backend` - The backend to mock (BitNet or Llama)
///
/// # Returns
/// - `Ok(TempDir)` - Temporary directory containing mock libraries
/// - `Err(String)` - Error message if creation failed
///
/// # Platform-Specific Behavior
/// - Linux: Creates `libbitnet.so`, `libllama.so`, `libggml.so`
/// - macOS: Creates `libbitnet.dylib`, `libllama.dylib`, `libggml.dylib`
/// - Windows: Creates `bitnet.dll`, `llama.dll`, `ggml.dll`
///
/// # Example
/// ```rust,ignore
/// use tempfile::TempDir;
///
/// #[test]
/// fn test_library_discovery() {
///     let temp = create_mock_backend_libs(CppBackend::BitNet).unwrap();
///     assert!(temp.path().join(format_lib_name("bitnet")).exists());
/// }
/// ```
#[allow(dead_code)]
pub fn create_mock_backend_libs(backend: CppBackend) -> Result<tempfile::TempDir, String> {
    use std::fs::File;
    use tempfile::TempDir;

    let temp = TempDir::new().map_err(|e| format!("Failed to create temp dir: {}", e))?;

    let lib_names = match backend {
        CppBackend::BitNet => vec!["bitnet"],
        CppBackend::Llama => vec!["llama", "ggml"],
    };

    for name in lib_names {
        let lib_path = temp.path().join(format_lib_name(name));

        // Create empty file
        File::create(&lib_path)
            .map_err(|e| format!("Failed to create mock library {}: {}", lib_path.display(), e))?;

        // Set executable permissions on Unix platforms
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&lib_path)
                .map_err(|e| format!("Failed to get metadata: {}", e))?
                .permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&lib_path, perms)
                .map_err(|e| format!("Failed to set permissions: {}", e))?;
        }
    }

    Ok(temp)
}

#[cfg(test)]
mod tests {
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
