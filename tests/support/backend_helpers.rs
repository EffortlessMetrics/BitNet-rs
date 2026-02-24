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
//! use bitnet_tests::support::backend_helpers::ensure_bitnet_or_skip;
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

// ============================================================================
// Stale Build Warning Functions (AC1-AC7)
// ============================================================================

/// Emit stale build warning when runtime detection succeeds but build-time constants are false
///
/// This function provides user-facing warnings when libraries are installed after xtask build.
/// It uses std::sync::Once for deduplication to ensure warnings appear only once per backend.
///
/// # Arguments
///
/// * `backend` - The C++ backend detected at runtime
/// * `matched_path` - The directory path where libraries were found
/// * `verbose` - If true, emit detailed diagnostic output
///
/// # Environment Variables
///
/// * `VERBOSE` - If set to "1", enables verbose diagnostic output
fn emit_stale_build_warning(backend: CppBackend, matched_path: &std::path::Path, verbose: bool) {
    use std::sync::Once;

    // Per-backend deduplication using static Once flags
    static BITNET_WARNING_EMITTED: Once = Once::new();
    static LLAMA_WARNING_EMITTED: Once = Once::new();

    let once_flag = match backend {
        CppBackend::BitNet => &BITNET_WARNING_EMITTED,
        CppBackend::Llama => &LLAMA_WARNING_EMITTED,
    };

    once_flag.call_once(|| {
        if verbose {
            emit_verbose_stale_warning(backend, matched_path);
        } else {
            emit_standard_stale_warning(backend);
        }
    });
}

/// Emit standard one-line stale build warning
///
/// Format: Single-line with warning symbol, backend name, and rebuild command
///
/// # Arguments
///
/// * `backend` - The C++ backend detected at runtime
pub fn emit_standard_stale_warning(backend: CppBackend) {
    eprintln!(
        "⚠️  STALE BUILD: {} found at runtime but not at build time. Rebuild required: cargo clean -p crossval && cargo build -p xtask --features crossval-all",
        backend.name()
    );
}

/// Emit verbose multi-line stale build diagnostic
///
/// Provides comprehensive diagnostic information including:
/// - What happened (libraries found at runtime but not build-time)
/// - Why rebuild is needed (build-time detection is baked into binary)
/// - Runtime detection results (matched path, libraries found)
/// - Build-time state (HAS_BITNET/HAS_LLAMA = false)
/// - Fix instructions (rebuild command)
///
/// # Arguments
///
/// * `backend` - The C++ backend detected at runtime
/// * `matched_path` - The directory path where libraries were found
pub fn emit_verbose_stale_warning(backend: CppBackend, matched_path: &std::path::Path) {
    const SEPARATOR: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

    eprintln!("{}", SEPARATOR);
    eprintln!("⚠️  STALE BUILD DETECTION");
    eprintln!("{}", SEPARATOR);
    eprintln!();
    eprintln!("Backend '{}' found at runtime but not at xtask build time.", backend.name());
    eprintln!();
    eprintln!("This happens when:");
    eprintln!("  1. You built xtask");
    eprintln!("  2. Then installed {} libraries later", backend.name());
    eprintln!("  3. xtask binary still contains old detection constants");
    eprintln!();
    eprintln!("Why rebuild is needed:");
    eprintln!("  • Library detection runs at BUILD time (not runtime)");
    eprintln!("  • Results are baked into the xtask binary as constants");
    eprintln!("  • Runtime detection is a fallback for developer convenience");
    eprintln!("  • Rebuild refreshes the constants to match filesystem reality");
    eprintln!();
    eprintln!("Runtime Detection Results:");
    eprintln!("  Matched path: {}", matched_path.display());

    // List libraries found in matched path
    if let Ok(entries) = std::fs::read_dir(matched_path) {
        let mut libs = Vec::new();
        for entry in entries.flatten() {
            if let Some(name) = entry.path().file_name().and_then(|n| n.to_str())
                && name.starts_with("lib")
                && (name.ends_with(".so") || name.ends_with(".dylib") || name.ends_with(".a"))
            {
                libs.push(name.to_string());
                #[cfg(target_os = "windows")]
                if name.ends_with(".dll") {
                    libs.push(name.to_string());
                }
            }
        }
        if !libs.is_empty() {
            eprintln!("  Libraries found: {}", libs.join(", "));
        }
    }

    eprintln!();
    eprintln!("Build-Time Detection State:");
    eprintln!(
        "  HAS_{} = false (stale)",
        match backend {
            CppBackend::BitNet => "BITNET",
            CppBackend::Llama => "LLAMA",
        }
    );

    eprintln!();
    eprintln!("Fix:");
    eprintln!("  cargo clean -p crossval && cargo build -p xtask --features crossval-all");
    eprintln!();
    eprintln!("Then re-run your test.");
}

/// Format CI-mode skip message when runtime detects libraries but build-time constants are stale
///
/// Provides clear diagnostic explaining why test is skipped in CI mode and setup instructions.
///
/// # Arguments
///
/// * `backend` - The backend that is unavailable at build-time
/// * `matched_path` - Optional matched path where runtime found libraries
///
/// # Returns
///
/// Formatted skip diagnostic message
pub fn format_ci_stale_skip_diagnostic(
    backend: CppBackend,
    matched_path: Option<&std::path::Path>,
) -> String {
    const SEPARATOR: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

    let mut msg = String::new();
    msg.push_str(&format!("{}\n", SEPARATOR));
    msg.push_str(&format!("⊘ Test skipped: {} not available (CI mode)\n", backend_name(backend)));
    msg.push_str(&format!("{}\n\n", SEPARATOR));

    msg.push_str("CI mode detected (CI=1 or BITNET_TEST_NO_REPAIR=1).\n");
    msg.push_str("Runtime detection found libraries but build-time constants are stale.\n\n");

    if let Some(path) = matched_path {
        msg.push_str(&format!("Runtime found libraries at: {}\n", path.display()));
        msg.push_str("But xtask was built before libraries were installed.\n\n");
    }

    msg.push_str("In CI mode:\n");
    msg.push_str("  • Build-time detection is the source of truth\n");
    msg.push_str("  • Runtime fallback is DISABLED for determinism\n");
    msg.push_str("  • xtask must be rebuilt to detect libraries\n\n");

    msg.push_str("Setup Instructions:\n");
    msg.push_str("  1. Install backend:\n");
    msg.push_str("     eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\"\n");
    msg.push_str("  2. Rebuild xtask:\n");
    msg.push_str("     cargo clean -p crossval && cargo build -p xtask --features crossval-all\n");
    msg.push_str("  3. Re-run CI job\n");

    msg
}

/// Check if running in CI environment (for CI-aware behavior)
///
/// Checks standard CI environment variables used by major CI/CD platforms.
///
/// # Returns
///
/// `true` if any CI environment variable is set, `false` otherwise
pub fn is_ci() -> bool {
    std::env::var_os("CI").is_some()
        || std::env::var_os("GITHUB_ACTIONS").is_some()
        || std::env::var_os("JENKINS_HOME").is_some()
        || std::env::var_os("GITLAB_CI").is_some()
        || std::env::var_os("CIRCLECI").is_some()
        || std::env::var_os("BITNET_TEST_NO_REPAIR").is_some()
}

// ============================================================================
// Backend Availability Checks
// ============================================================================

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
/// - `BITNET_REPAIR_ATTEMPTED=1`: Prevent multiple repair attempts
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

    // Check build-time constant first (Priority 1)
    let build_time_available = match backend {
        CppBackend::BitNet => HAS_BITNET,
        CppBackend::Llama => HAS_LLAMA,
    };

    if build_time_available {
        return; // Backend available at build time, continue test
    }

    // Check runtime detection as fallback (Priority 2)
    if let Ok((runtime_available, matched_path)) = detect_backend_runtime(backend)
        && runtime_available
        && is_ci()
    {
        // STALE BUILD SCENARIO + CI: respect build-time constants only (no runtime override)
        let skip_msg = format_ci_stale_skip_diagnostic(backend, matched_path.as_deref());
        panic!("SKIPPED: {}", skip_msg);
    } else if let Ok((runtime_available, matched_path)) = detect_backend_runtime(backend)
        && runtime_available
    {
        // STALE BUILD SCENARIO + DEV: allow test to proceed with warning
        let verbose = std::env::var("VERBOSE").is_ok();
        if let Some(path) = matched_path {
            emit_stale_build_warning(backend, &path, verbose);
        }
        return; // Continue execution
    }

    // CI mode: skip immediately (Priority 3)
    if is_ci_or_no_repair() {
        panic!("SKIPPED: {}", format_skip_diagnostic(backend, None));
    }

    // Check if repair already attempted (Priority 4)
    if std::env::var_os("BITNET_REPAIR_ATTEMPTED").is_some() {
        panic!("SKIPPED: {}", format_skip_diagnostic(backend, Some("repair already attempted")));
    }

    // Dev mode: attempt auto-repair (Priority 5)
    unsafe {
        std::env::set_var("BITNET_REPAIR_ATTEMPTED", "1");
    }
    eprintln!("Attempting auto-repair for {} backend...", backend_name(backend));

    match auto_repair_with_rebuild(backend) {
        Ok(()) => {
            eprintln!("✅ {} backend installed and configured.", backend_name(backend));
            // Backend now available, test can continue
        }
        Err(e) => {
            eprintln!("❌ Auto-repair failed: {}", e);
            panic!("SKIPPED: {}", format_skip_diagnostic(backend, Some(&e.to_string())));
        }
    }
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
#[allow(dead_code)]
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

/// Auto-repair with rebuild (AC1-AC2 implementation)
///
/// Performs full auto-repair cycle:
/// 1. Check recursion guard
/// 2. Run setup-cpp-auto
/// 3. Rebuild xtask (optional - may require manual step)
/// 4. Verify backend available
///
/// # Arguments
///
/// * `backend` - The backend to repair
///
/// # Returns
///
/// - `Ok(())` if repair successful and backend now available
/// - `Err(String)` if repair failed with error message
#[allow(dead_code)]
fn auto_repair_with_rebuild(backend: CppBackend) -> Result<(), String> {
    // Recursion prevention - check if already in repair
    if std::env::var_os("BITNET_REPAIR_IN_PROGRESS").is_some() {
        return Err("Recursion detected: repair already in progress".to_string());
    }

    // Set recursion guard
    unsafe {
        std::env::set_var("BITNET_REPAIR_IN_PROGRESS", "1");
    }

    // Attempt repair
    let result = attempt_auto_repair(backend);

    // Clear recursion guard
    unsafe {
        std::env::remove_var("BITNET_REPAIR_IN_PROGRESS");
    }

    // Return result
    result
}

/// Get backend name as string
fn backend_name(backend: CppBackend) -> &'static str {
    match backend {
        CppBackend::BitNet => "bitnet.cpp",
        CppBackend::Llama => "llama.cpp",
    }
}

/// Format skip diagnostic message without printing
///
/// Returns a formatted skip message string with setup instructions.
///
/// # Arguments
///
/// * `backend` - The backend that is unavailable
/// * `error_context` - Optional error context string
///
/// # Returns
///
/// Formatted skip diagnostic message
#[allow(dead_code)]
fn format_skip_diagnostic(backend: CppBackend, error_context: Option<&str>) -> String {
    let mut msg = String::new();
    msg.push_str("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    msg.push_str(&format!("⊘ Test skipped: {} not available\n", backend_name(backend)));
    msg.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    if let Some(ctx) = error_context {
        msg.push_str(&format!("\nContext: {}\n", ctx));
    }

    msg.push_str(&format!("\nThis test requires the {:?} C++ reference backend.\n", backend));

    msg.push_str("\nSetup Instructions:\n");
    msg.push_str("──────────────────────────────────────────────────────────\n");

    msg.push_str("\n  Option A: Auto-setup (recommended)\n\n");
    msg.push_str("    1. Install backend:\n");
    msg.push_str("       eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\"\n");
    msg.push_str("\n    2. Rebuild xtask to update detection:\n");
    msg.push_str(
        "       cargo clean -p crossval && cargo build -p xtask --features crossval-all\n",
    );
    msg.push_str("\n    3. Re-run tests:\n");
    msg.push_str("       cargo test --workspace --no-default-features --features cpu\n");

    msg.push_str("\n  Option B: Manual setup (advanced)\n\n");
    match backend {
        CppBackend::BitNet => {
            msg.push_str("    1. Clone and build BitNet.cpp:\n");
            msg.push_str(
                "       git clone https://github.com/microsoft/BitNet.git ~/.cache/bitnet_cpp\n",
            );
            msg.push_str("       cd ~/.cache/bitnet_cpp && mkdir build && cd build\n");
            msg.push_str("       cmake .. && cmake --build .\n");
            msg.push_str("\n    2. Set environment variables:\n");
            msg.push_str("       export BITNET_CPP_DIR=~/.cache/bitnet_cpp\n");
            msg.push_str(
                "       export LD_LIBRARY_PATH=~/.cache/bitnet_cpp/build/bin:$LD_LIBRARY_PATH\n",
            );
        }
        CppBackend::Llama => {
            msg.push_str("    1. Clone and build llama.cpp:\n");
            msg.push_str(
                "       git clone https://github.com/ggerganov/llama.cpp.git ~/.cache/llama_cpp\n",
            );
            msg.push_str("       cd ~/.cache/llama_cpp && mkdir build && cd build\n");
            msg.push_str("       cmake .. && cmake --build .\n");
            msg.push_str("\n    2. Set environment variables:\n");
            msg.push_str("       export LLAMA_CPP_DIR=~/.cache/llama_cpp\n");
            msg.push_str(
                "       export LD_LIBRARY_PATH=~/.cache/llama_cpp/build:$LD_LIBRARY_PATH\n",
            );
        }
    }
    msg.push_str("\n    3. Rebuild and re-run tests (steps 2-3 from Option A)\n");

    msg.push_str("\nDocumentation: docs/howto/cpp-setup.md\n");
    msg.push_str("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    msg
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
    eprint!("{}", format_skip_diagnostic(backend, context));
}

/// Runtime backend detection fallback (post-install, pre-rebuild)
///
/// This function provides runtime detection when libraries are installed after xtask build.
/// It checks multiple sources in priority order:
///
/// 1. `BITNET_CROSSVAL_LIBDIR` (explicit override)
/// 2. Backend-specific granular overrides (`CROSSVAL_RPATH_BITNET`, `CROSSVAL_RPATH_LLAMA`)
/// 3. Backend home dir + subdirectories (`BITNET_CPP_DIR/build`, `LLAMA_CPP_DIR/build`, etc.)
///
/// # Arguments
///
/// * `backend` - The C++ backend to detect (BitNet or Llama)
///
/// # Returns
///
/// * `Ok((true, Some(path)))` - Backend libraries found at runtime, with matched path
/// * `Ok((false, None))` - Backend libraries not found
/// * `Err(String)` - Error during detection
///
/// # Platform-Specific Library Extensions
///
/// - Linux: `.so`
/// - macOS: `.dylib`
/// - Windows: `.dll`
pub fn detect_backend_runtime(
    backend: CppBackend,
) -> Result<(bool, Option<std::path::PathBuf>), String> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    // Priority 1: BITNET_CROSSVAL_LIBDIR (explicit override)
    if let Ok(p) = std::env::var("BITNET_CROSSVAL_LIBDIR") {
        for part in p.split(if cfg!(windows) { ';' } else { ':' }) {
            candidates.push(part.into());
        }
    }

    // Priority 2: Granular overrides (backend-specific), colon-separated on Unix
    match backend {
        CppBackend::BitNet => {
            if let Ok(p) = std::env::var("CROSSVAL_RPATH_BITNET") {
                for part in p.split(if cfg!(windows) { ';' } else { ':' }) {
                    candidates.push(part.into());
                }
            }
        }
        CppBackend::Llama => {
            if let Ok(p) = std::env::var("CROSSVAL_RPATH_LLAMA") {
                for part in p.split(if cfg!(windows) { ';' } else { ':' }) {
                    candidates.push(part.into());
                }
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

    // Check for required library filenames per platform
    let exts = if cfg!(target_os = "windows") {
        vec!["dll"]
    } else if cfg!(target_os = "macos") {
        vec!["dylib"]
    } else {
        vec!["so"]
    };

    let needs: &[&str] = match backend {
        CppBackend::BitNet => &["bitnet"],
        CppBackend::Llama => &["llama", "ggml"],
    };

    // Check each candidate directory and return first match with path
    for dir in candidates {
        if !dir.exists() {
            continue;
        }

        // Check if all required libraries are present
        let all_found = needs.iter().all(|stem| {
            exts.iter().any(|ext| {
                let lib_name = format_lib_name_ext(stem, ext);
                dir.join(&lib_name).exists()
            })
        });

        if all_found {
            return Ok((true, Some(dir)));
        }
    }

    Ok((false, None))
}

/// Format library name with specific extension (helper for runtime detection)
///
/// # Arguments
///
/// * `stem` - Library name stem (e.g., "bitnet", "llama")
/// * `ext` - File extension (e.g., "so", "dylib", "dll")
///
/// # Returns
///
/// Formatted library name:
/// - Windows: `{stem}.{ext}` (e.g., "bitnet.dll")
/// - Unix: `lib{stem}.{ext}` (e.g., "libbitnet.so")
fn format_lib_name_ext(stem: &str, ext: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{}.{}", stem, ext)
    } else {
        format!("lib{}.{}", stem, ext)
    }
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
/// use bitnet_tests::support::backend_helpers::get_loader_path_var;
/// use std::env;
///
/// let loader_var = get_loader_path_var();
/// unsafe {
///     env::set_var(loader_var, "/custom/lib");
/// }
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

/// Get backend library filenames for current platform
///
/// Returns the list of library filenames required for the specified backend,
/// using platform-specific naming conventions.
///
/// # Arguments
///
/// * `backend` - The C++ backend (BitNet or Llama)
///
/// # Returns
///
/// Vector of library filenames with platform-specific extensions and prefixes:
/// - Linux: `["libbitnet.so"]` or `["libllama.so", "libggml.so"]`
/// - macOS: `["libbitnet.dylib"]` or `["libllama.dylib", "libggml.dylib"]`
/// - Windows: `["bitnet.dll"]` or `["llama.dll", "ggml.dll"]`
///
/// # Examples
///
/// ```rust,ignore
/// use bitnet_tests::support::backend_helpers::get_backend_lib_names;
/// use xtask::crossval::backend::CppBackend;
///
/// let bitnet_libs = get_backend_lib_names(CppBackend::BitNet);
/// assert_eq!(bitnet_libs.len(), 1);
/// assert!(bitnet_libs[0].starts_with("lib") || cfg!(target_os = "windows"));
///
/// let llama_libs = get_backend_lib_names(CppBackend::Llama);
/// assert_eq!(llama_libs.len(), 2); // libllama + libggml
/// ```
#[allow(dead_code)]
pub fn get_backend_lib_names(backend: CppBackend) -> Vec<String> {
    let stems = match backend {
        CppBackend::BitNet => vec!["bitnet"],
        CppBackend::Llama => vec!["llama", "ggml"],
    };

    stems.iter().map(|stem| format_lib_name(stem)).collect()
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
/// use bitnet_tests::support::backend_helpers::format_lib_name;
///
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

/// Create mock C++ libraries in specified directory (flexible version for AC2/AC7/AC8 tests)
///
/// This helper provides more control over library creation for testing precedence
/// and environment variable behavior. Unlike `create_mock_backend_libs()`, this
/// allows specifying the target directory and optionally skipping file creation
/// for negative tests.
///
/// # Arguments
///
/// * `dir` - Directory to create libraries in
/// * `backend` - Backend type (determines which libraries to create)
/// * `create_files` - If true, creates actual files; if false, only creates directory structure
///
/// # Returns
///
/// * `Ok(())` - Libraries created successfully
/// * `Err(std::io::Error)` - File creation or directory error
///
/// # Platform-Specific Behavior
///
/// - Linux: Creates `libbitnet.so`, `libllama.so`, `libggml.so`
/// - macOS: Creates `libbitnet.dylib`, `libllama.dylib`, `libggml.dylib`
/// - Windows: Creates `bitnet.dll`, `llama.dll`, `ggml.dll`
///
/// # Examples
///
/// ```rust,ignore
/// use tempfile::TempDir;
/// use std::path::Path;
/// use bitnet_tests::support::backend_helpers::create_mock_cpp_libs;
/// use xtask::crossval::backend::CppBackend;
///
/// #[test]
/// fn test_precedence_with_mocks() {
///     let temp = TempDir::new().unwrap();
///     let high_priority = temp.path().join("high");
///     let low_priority = temp.path().join("low");
///
///     std::fs::create_dir_all(&high_priority).unwrap();
///     std::fs::create_dir_all(&low_priority).unwrap();
///
///     // Create libs in both directories
///     create_mock_cpp_libs(&high_priority, CppBackend::BitNet, true).unwrap();
///     create_mock_cpp_libs(&low_priority, CppBackend::BitNet, true).unwrap();
///
///     // Test environment variable precedence...
/// }
/// ```
#[allow(dead_code)]
pub fn create_mock_cpp_libs(
    dir: &std::path::Path,
    backend: CppBackend,
    create_files: bool,
) -> Result<(), std::io::Error> {
    use std::fs::{File, create_dir_all};

    // Ensure directory exists
    create_dir_all(dir)?;

    if !create_files {
        return Ok(()); // Directory structure only
    }

    // Get library names for backend
    let lib_names = get_backend_lib_names(backend);

    for lib_name in lib_names {
        let lib_path = dir.join(&lib_name);

        // Create empty file
        File::create(&lib_path)?;

        // Set executable permissions on Unix platforms
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&lib_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&lib_path, perms)?;
        }
    }

    Ok(())
}

// ============================================================================
// Test Assertion Helpers (AC2/AC7/AC8)
// ============================================================================

/// Assert that backend runtime detection returns expected results
///
/// This helper simplifies test assertions for backend availability and matched path
/// validation. It calls `detect_backend_runtime()` and asserts on both availability
/// and matched path.
///
/// # Arguments
///
/// * `backend` - The C++ backend to detect
/// * `expected_available` - Expected availability boolean
/// * `expected_path` - Expected matched path (None if backend unavailable)
///
/// # Panics
///
/// Panics if:
/// - `detect_backend_runtime()` returns an error
/// - Availability doesn't match `expected_available`
/// - Matched path doesn't match `expected_path` (canonical comparison)
///
/// # Examples
///
/// ```rust,ignore
/// use tempfile::TempDir;
/// use bitnet_tests::support::backend_helpers::{assert_backend_runtime, create_mock_cpp_libs};
/// use bitnet_tests::support::env_guard::EnvGuard;
/// use xtask::crossval::backend::CppBackend;
///
/// #[test]
/// #[serial_test::serial(bitnet_env)]
/// fn test_backend_detection() {
///     let temp = TempDir::new().unwrap();
///     create_mock_cpp_libs(temp.path(), CppBackend::BitNet, true).unwrap();
///
///     let _guard = EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
///     _guard.set(temp.path().to_str().unwrap());
///
///     // Assert backend is available at temp path
///     assert_backend_runtime(
///         CppBackend::BitNet,
///         true,
///         Some(temp.path())
///     );
/// }
/// ```
#[allow(dead_code)]
pub fn assert_backend_runtime(
    backend: CppBackend,
    expected_available: bool,
    expected_path: Option<&std::path::Path>,
) {
    let result = detect_backend_runtime(backend);

    assert!(
        result.is_ok(),
        "detect_backend_runtime({:?}) should not error: {:?}",
        backend,
        result.unwrap_err()
    );

    let (available, matched_path) = result.unwrap();

    assert_eq!(
        available, expected_available,
        "Backend {:?} availability mismatch: expected {}, got {}",
        backend, expected_available, available
    );

    match (matched_path, expected_path) {
        (Some(actual), Some(expected)) => {
            let actual_canonical = actual.canonicalize().unwrap_or_else(|_| {
                panic!("Failed to canonicalize actual path: {}", actual.display())
            });
            let expected_canonical = expected.canonicalize().unwrap_or_else(|_| {
                panic!("Failed to canonicalize expected path: {}", expected.display())
            });

            assert_eq!(
                actual_canonical,
                expected_canonical,
                "Matched path mismatch: expected {}, got {}",
                expected.display(),
                actual.display()
            );
        }
        (None, None) => {
            // Both None - match
        }
        (Some(actual), None) => {
            panic!("Unexpected matched path: got Some({}), expected None", actual.display());
        }
        (None, Some(expected)) => {
            panic!("Missing matched path: got None, expected Some({})", expected.display());
        }
    }
}

// ============================================================================
// Environment Setup Helpers (AC7/AC8 - Precedence Tests)
// ============================================================================

/// Setup environment variables for precedence testing
///
/// This helper configures multiple environment variables in priority order
/// for testing environment variable precedence in runtime detection.
///
/// # Environment Variable Priority Order (from highest to lowest)
///
/// 1. `BITNET_CROSSVAL_LIBDIR` - Explicit override (highest precedence)
/// 2. `BITNET_CPP_DIR` or `LLAMA_CPP_DIR` - Backend home directory (medium precedence)
/// 3. `CROSSVAL_RPATH_BITNET` or `CROSSVAL_RPATH_LLAMA` - Granular RPATH (lowest precedence)
///
/// # Arguments
///
/// * `crossval_libdir` - Optional path for BITNET_CROSSVAL_LIBDIR (Priority 1)
/// * `cpp_dir` - Optional path for BITNET_CPP_DIR/LLAMA_CPP_DIR (Priority 2)
/// * `rpath` - Optional path for CROSSVAL_RPATH_BITNET/LLAMA (Priority 3)
///
/// # Returns
///
/// Vector of `EnvGuard` instances for automatic cleanup. The guards MUST be kept
/// alive for the duration of the test to maintain environment state.
///
/// # Examples
///
/// ```rust,ignore
/// use tempfile::TempDir;
/// use bitnet_tests::support::backend_helpers::{setup_precedence_test, create_mock_cpp_libs};
/// use bitnet_tests::support::env_guard::EnvGuard;
/// use xtask::crossval::backend::CppBackend;
///
/// #[test]
/// #[serial_test::serial(bitnet_env)]
/// fn test_crossval_libdir_precedence() {
///     let high = TempDir::new().unwrap();
///     let low = TempDir::new().unwrap();
///
///     create_mock_cpp_libs(high.path(), CppBackend::BitNet, true).unwrap();
///     create_mock_cpp_libs(low.path(), CppBackend::BitNet, true).unwrap();
///
///     // Setup: BITNET_CROSSVAL_LIBDIR (high) vs CROSSVAL_RPATH_BITNET (low)
///     let _guards = setup_precedence_test(
///         Some(high.path()),  // Priority 1
///         None,               // Priority 2 (skip)
///         Some(low.path())    // Priority 3
///     );
///
///     // Runtime detection should use high priority path
///     let (available, matched_path) = detect_backend_runtime(CppBackend::BitNet).unwrap();
///     assert!(available);
///     assert_eq!(matched_path.unwrap().canonicalize().unwrap(),
///                high.path().canonicalize().unwrap());
/// }
/// ```
#[allow(dead_code)]
pub fn setup_precedence_test(
    crossval_libdir: Option<&std::path::Path>,
    cpp_dir: Option<&std::path::Path>,
    rpath: Option<&std::path::Path>,
) -> Vec<super::env_guard::EnvGuard> {
    let mut guards = Vec::new();

    // Priority 1: BITNET_CROSSVAL_LIBDIR (explicit override)
    if let Some(path) = crossval_libdir {
        let guard = super::env_guard::EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
        guard.set(path.to_str().expect("Path must be valid UTF-8"));
        guards.push(guard);
    }

    // Priority 2: BITNET_CPP_DIR or LLAMA_CPP_DIR (backend home)
    // Note: Caller should specify which backend via separate parameter if needed
    // For now, we set BITNET_CPP_DIR as the default
    if let Some(path) = cpp_dir {
        let guard = super::env_guard::EnvGuard::new("BITNET_CPP_DIR");
        guard.set(path.to_str().expect("Path must be valid UTF-8"));
        guards.push(guard);
    }

    // Priority 3: CROSSVAL_RPATH_BITNET or CROSSVAL_RPATH_LLAMA (granular RPATH)
    // Default to CROSSVAL_RPATH_BITNET
    if let Some(path) = rpath {
        let guard = super::env_guard::EnvGuard::new("CROSSVAL_RPATH_BITNET");
        guard.set(path.to_str().expect("Path must be valid UTF-8"));
        guards.push(guard);
    }

    guards
}

/// Setup environment variables for backend-specific precedence testing
///
/// This is a specialized version of `setup_precedence_test()` that allows
/// specifying the backend type explicitly, ensuring correct environment
/// variable names (BITNET_CPP_DIR vs LLAMA_CPP_DIR, etc.).
///
/// # Arguments
///
/// * `backend` - The C++ backend type (BitNet or Llama)
/// * `crossval_libdir` - Optional path for BITNET_CROSSVAL_LIBDIR (Priority 1)
/// * `cpp_dir` - Optional path for {BITNET|LLAMA}_CPP_DIR (Priority 2)
/// * `rpath` - Optional path for CROSSVAL_RPATH_{BITNET|LLAMA} (Priority 3)
///
/// # Returns
///
/// Vector of `EnvGuard` instances for automatic cleanup
///
/// # Examples
///
/// ```rust,ignore
/// use bitnet_tests::support::backend_helpers::setup_backend_precedence_test;
/// use xtask::crossval::backend::CppBackend;
///
/// #[test]
/// #[serial_test::serial(bitnet_env)]
/// fn test_llama_precedence() {
///     let high = TempDir::new().unwrap();
///     let low = TempDir::new().unwrap();
///
///     create_mock_cpp_libs(high.path(), CppBackend::Llama, true).unwrap();
///     create_mock_cpp_libs(low.path(), CppBackend::Llama, true).unwrap();
///
///     // Setup with correct Llama environment variable names
///     let _guards = setup_backend_precedence_test(
///         CppBackend::Llama,
///         Some(high.path()),
///         None,
///         Some(low.path())
///     );
///
///     // Runtime detection uses LLAMA_CPP_DIR, CROSSVAL_RPATH_LLAMA
///     let (available, matched_path) = detect_backend_runtime(CppBackend::Llama).unwrap();
///     assert!(available);
/// }
/// ```
#[allow(dead_code)] // Test scaffolding - used in future tests
pub fn setup_backend_precedence_test(
    backend: CppBackend,
    crossval_libdir: Option<&std::path::Path>,
    cpp_dir: Option<&std::path::Path>,
    rpath: Option<&std::path::Path>,
) -> Vec<super::env_guard::EnvGuard> {
    let mut guards = Vec::new();

    // Priority 1: BITNET_CROSSVAL_LIBDIR (explicit override - shared for both backends)
    if let Some(path) = crossval_libdir {
        let guard = super::env_guard::EnvGuard::new("BITNET_CROSSVAL_LIBDIR");
        guard.set(path.to_str().expect("Path must be valid UTF-8"));
        guards.push(guard);
    }

    // Priority 2: Backend-specific home directory
    if let Some(path) = cpp_dir {
        let env_var = match backend {
            CppBackend::BitNet => "BITNET_CPP_DIR",
            CppBackend::Llama => "LLAMA_CPP_DIR",
        };
        let guard = super::env_guard::EnvGuard::new(env_var);
        guard.set(path.to_str().expect("Path must be valid UTF-8"));
        guards.push(guard);
    }

    // Priority 3: Backend-specific granular RPATH
    if let Some(path) = rpath {
        let env_var = match backend {
            CppBackend::BitNet => "CROSSVAL_RPATH_BITNET",
            CppBackend::Llama => "CROSSVAL_RPATH_LLAMA",
        };
        let guard = super::env_guard::EnvGuard::new(env_var);
        guard.set(path.to_str().expect("Path must be valid UTF-8"));
        guards.push(guard);
    }

    guards
}

/// Create temporary directory for mock libraries (convenience wrapper)
///
/// This is a thin wrapper around `tempfile::tempdir()` for consistency
/// with other helper functions.
///
/// # Returns
///
/// A `TempDir` instance that auto-cleans on drop
///
/// # Panics
///
/// Panics if temporary directory creation fails
///
/// # Examples
///
/// ```rust,ignore
/// use bitnet_tests::support::backend_helpers::{create_temp_lib_dir, create_mock_cpp_libs};
/// use xtask::crossval::backend::CppBackend;
///
/// #[test]
/// fn test_with_temp_libs() {
///     let temp = create_temp_lib_dir();
///     create_mock_cpp_libs(temp.path(), CppBackend::BitNet, true).unwrap();
///
///     // Test code here...
///     // TempDir auto-cleaned on drop
/// }
/// ```
#[allow(dead_code)] // Test scaffolding - used in future tests
pub fn create_temp_lib_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("Failed to create temporary directory")
}

// ============================================================================
// Test Data Constants (AC2/AC7/AC8)
// ============================================================================

/// Mock library file size for testing (1KB)
///
/// This constant is used to create mock library files with a realistic size
/// for testing file detection and validation.
#[cfg(test)]
#[allow(dead_code)] // Test scaffolding - used in future tests
pub const MOCK_BITNET_LIB_SIZE: u64 = 1024;

/// Mock library file size for testing (2KB)
///
/// This constant is used for Llama libraries to differentiate from BitNet
/// in tests that validate library-specific behavior.
#[cfg(test)]
#[allow(dead_code)] // Test scaffolding - used in future tests
pub const MOCK_LLAMA_LIB_SIZE: u64 = 2048;

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "TDD scaffold: see test_support_tests.rs for comprehensive environment tests"]
    fn test_is_ci_or_no_repair_with_no_repair_flag() {
        unimplemented!("See test_support_tests.rs for comprehensive environment tests");
    }

    #[test]
    #[ignore = "TDD scaffold: see test_support_tests.rs for comprehensive environment tests"]
    fn test_is_ci_or_no_repair_with_ci_flag() {
        unimplemented!("See test_support_tests.rs for comprehensive environment tests");
    }

    #[test]
    #[ignore = "TDD scaffold: see test_support_tests.rs for comprehensive environment tests"]
    fn test_is_ci_or_no_repair_interactive() {
        unimplemented!("See test_support_tests.rs for comprehensive environment tests");
    }

    #[test]
    #[ignore = "TDD scaffold: see test_support_tests.rs for comprehensive backend tests"]
    fn test_ensure_backend_or_skip_backend_available() {
        unimplemented!("See test_support_tests.rs for comprehensive backend tests");
    }

    #[test]
    #[ignore = "TDD scaffold: see test_support_tests.rs for comprehensive backend tests"]
    fn test_ensure_backend_or_skip_backend_unavailable_ci() {
        unimplemented!("See test_support_tests.rs for comprehensive backend tests");
    }

    #[test]
    #[ignore = "TDD scaffold: see test_support_tests.rs for comprehensive backend tests"]
    fn test_ensure_backend_or_skip_backend_unavailable_repair() {
        unimplemented!("See test_support_tests.rs for comprehensive backend tests");
    }

    #[test]
    #[ignore = "TDD scaffold: see test_support_tests.rs for comprehensive wrapper tests"]
    fn test_convenience_wrappers() {
        unimplemented!("See test_support_tests.rs for comprehensive wrapper tests");
    }

    #[test]
    #[ignore = "TDD scaffold: see test_support_tests.rs for comprehensive repair tests"]
    fn test_attempt_auto_repair_success() {
        unimplemented!("See test_support_tests.rs for comprehensive repair tests");
    }

    #[test]
    #[ignore = "TDD scaffold: see test_support_tests.rs for comprehensive repair tests"]
    fn test_attempt_auto_repair_failure() {
        unimplemented!("See test_support_tests.rs for comprehensive repair tests");
    }

    #[test]
    #[ignore = "TDD scaffold: see test_support_tests.rs for comprehensive diagnostic tests"]
    fn test_print_skip_diagnostic_format() {
        unimplemented!("See test_support_tests.rs for comprehensive diagnostic tests");
    }
}
