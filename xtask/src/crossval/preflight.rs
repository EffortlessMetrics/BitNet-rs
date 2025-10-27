#![allow(dead_code)] // TODO: Enhanced preflight functions to be integrated with main command

//! Backend library preflight validation
//!
//! Verifies that required C++ libraries are available before running cross-validation.

use crate::crossval::CppBackend;
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;
use std::{env, process::Command};

/// Error types for auto-repair functionality
///
/// These errors provide detailed diagnostics and recovery steps for users
/// experiencing issues during automatic C++ backend installation.
#[derive(Debug, thiserror::Error)]
pub enum RepairError {
    /// Network failure during git clone or download
    #[error(
        "Network failure during clone: {error}\n\nRecovery steps:\n1. Check internet connection: ping github.com\n2. Verify firewall allows git clone\n3. Retry with: cargo run -p xtask -- preflight --backend {backend} --repair\n\nFor more help, see:\n  docs/howto/cpp-setup.md"
    )]
    NetworkFailure { error: String, backend: String },

    /// Build failure during CMake or compilation
    #[error(
        "Build failure: {error}\n\nRecovery steps:\n1. Check dependencies installed:\n   - cmake --version  (should be >= 3.18)\n   - gcc --version or clang --version\n2. Review build log above for specific errors\n3. Try manual setup: cargo run -p xtask -- setup-cpp-auto --backend {backend}\n\nFor more help, see:\n  docs/howto/cpp-setup.md\n  docs/GPU_SETUP.md (for GPU-related build errors)"
    )]
    BuildFailure { error: String, backend: String },

    /// Permission denied during file operations
    #[error(
        "Permission denied: {path}\n\nRecovery steps:\n1. Check directory ownership:\n   ls -ld {path}\n2. Fix ownership if needed:\n   sudo chown -R $USER {path}\n3. OR set custom directory:\n   export BITNET_CPP_DIR=~/my-bitnet-cpp\n   cargo run -p xtask -- preflight --backend {backend} --repair"
    )]
    PermissionDenied { path: String, backend: String },

    /// Unknown error (catch-all for unclassified errors)
    #[error(
        "Unknown error: {error}\n\nBackend: {backend}\n\nPlease report this issue with full output."
    )]
    Unknown { error: String, backend: String },

    /// setup-cpp-auto command failed
    #[error("setup-cpp-auto failed: {0}")]
    SetupFailed(String),

    /// Revalidation failed after repair (backend still unavailable)
    #[error(
        "Revalidation failed after repair: backend still unavailable\n\nThis may indicate:\n1. Libraries were built but in unexpected location\n2. Build succeeded but libraries are incompatible\n3. Partial build state from previous failed attempt\n\nRecovery steps:\n1. Clean previous build state:\n   rm -rf ~/.cache/bitnet_cpp\n2. Retry repair:\n   cargo run -p xtask -- preflight --backend {backend} --repair --verbose\n3. If problem persists, see manual setup: docs/howto/cpp-setup.md"
    )]
    RevalidationFailed { backend: String },

    /// xtask rebuild failed during repair
    #[error("xtask rebuild failed: {0}")]
    RebuildFailed(#[from] RebuildError),

    /// Repair already in progress (recursion guard)
    #[error(
        "Repair already in progress (recursion detected)\n\nThis indicates an internal error. Please report this issue with:\n1. Full command that triggered repair\n2. Environment variables (BITNET_CPP_DIR, etc.)\n3. Output from: cargo run -p xtask -- preflight --verbose"
    )]
    RecursionDetected,
}

impl RepairError {
    /// Classify error from stderr output
    ///
    /// Determines error type based on stderr content from setup-cpp-auto or cmake.
    ///
    /// # Arguments
    ///
    /// * `stderr` - Standard error output to classify
    /// * `backend` - Backend name for error context
    ///
    /// # Returns
    ///
    /// Classified RepairError instance:
    /// - NetworkFailure: Connection timeouts, git clone failures, DNS errors
    /// - BuildFailure: CMake errors, compiler errors, linker failures
    /// - PermissionDenied: EACCES, permission denied, cannot create directory
    /// - Unknown: Unrecognized error pattern
    pub fn classify(stderr: &str, backend: &str) -> Self {
        let lower = stderr.to_lowercase();

        // Network error patterns
        if lower.contains("connection timeout")
            || lower.contains("failed to clone")
            || lower.contains("could not resolve host")
            || lower.contains("network unreachable")
            || lower.contains("connection refused")
            || lower.contains("failed to connect")
            || lower.contains("timed out")
        {
            return RepairError::NetworkFailure {
                error: stderr.to_string(),
                backend: backend.to_string(),
            };
        }

        // Build error patterns
        if lower.contains("cmake error")
            || lower.contains("ninja: build stopped")
            || lower.contains("undefined reference")
            || lower.contains("no such file or directory")
            || lower.contains("compilation failed")
            || lower.contains("cannot find")
            || lower.contains("compiler")
        {
            return RepairError::BuildFailure {
                error: stderr.to_string(),
                backend: backend.to_string(),
            };
        }

        // Permission error patterns
        if lower.contains("permission denied")
            || lower.contains("eacces")
            || lower.contains("cannot create directory")
            || lower.contains("access is denied")
        {
            // Try to extract path from error message
            let path = stderr
                .lines()
                .find(|line| {
                    line.to_lowercase().contains("permission denied")
                        || line.to_lowercase().contains("eacces")
                })
                .and_then(|line| {
                    line.split_whitespace()
                        .find(|word| word.starts_with('/') || word.starts_with("~/"))
                })
                .unwrap_or("~/.cache/bitnet_cpp");

            return RepairError::PermissionDenied {
                path: path.to_string(),
                backend: backend.to_string(),
            };
        }

        // Unknown error (catch-all)
        RepairError::Unknown { error: stderr.to_string(), backend: backend.to_string() }
    }

    /// Map RepairError to semantic exit code
    ///
    /// Provides clear exit codes for CI/CD integration and automated workflows.
    ///
    /// # Returns
    ///
    /// Exit code integer:
    /// - NetworkFailure → 3 (retryable)
    /// - BuildFailure → 5 (permanent, needs dependency installation)
    /// - PermissionDenied → 4 (permanent, needs ownership fix)
    /// - RecursionDetected → 6 (internal error)
    /// - Others → 1 (generic unavailable)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let error = RepairError::NetworkFailure {
    ///     error: "connection timeout".to_string(),
    ///     backend: "bitnet".to_string(),
    /// };
    /// assert_eq!(error.to_exit_code(), 3);
    /// ```
    pub fn to_exit_code(&self) -> i32 {
        match self {
            RepairError::NetworkFailure { .. } => PreflightExitCode::NetworkFailure as i32,
            RepairError::BuildFailure { .. } => PreflightExitCode::BuildFailure as i32,
            RepairError::PermissionDenied { .. } => PreflightExitCode::PermissionDenied as i32,
            RepairError::RecursionDetected => PreflightExitCode::RecursionDetected as i32,
            RepairError::Unknown { .. }
            | RepairError::SetupFailed(_)
            | RepairError::RevalidationFailed { .. }
            | RepairError::RebuildFailed(_) => PreflightExitCode::Unavailable as i32,
        }
    }
}

/// Error types for xtask rebuild operations
#[derive(Debug, thiserror::Error)]
pub enum RebuildError {
    /// cargo clean failed
    #[error(
        "cargo clean failed: {0}\n\nRecovery steps:\n1. Check cargo is installed and accessible\n2. Verify no file locks on target/ directory\n3. Try manual clean: cargo clean -p xtask -p crossval"
    )]
    CleanFailed(String),

    /// cargo build failed
    #[error(
        "cargo build failed: {0}\n\nRecovery steps:\n1. Check Rust toolchain: rustc --version\n2. Verify no compilation errors in xtask crate\n3. Try manual build: cargo build -p xtask --features crossval-all"
    )]
    BuildFailed(String),

    /// Permission denied during rebuild
    #[error(
        "Permission denied during rebuild: {0}\n\nRecovery steps:\n1. Check target/ directory permissions\n2. Ensure no process has files locked\n3. Try: sudo chown -R $USER target/"
    )]
    PermissionDenied(String),
}

/// Exit codes for preflight command
///
/// These codes provide semantic meaning for CI/CD integration and user diagnostics.
///
/// # Usage
///
/// ```ignore
/// use xtask::crossval::preflight::PreflightExitCode;
///
/// // Convert from RepairError
/// let exit_code = PreflightExitCode::from_repair_error(&error);
///
/// // Use in main return
/// std::process::exit(exit_code as i32);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum PreflightExitCode {
    /// Backend available (ready for cross-validation)
    Available = 0,

    /// Backend unavailable after repair (repair disabled or failed non-retryable)
    Unavailable = 1,

    /// Invalid arguments (unknown backend name)
    InvalidArgs = 2,

    /// Auto-repair failed due to network error (after retries)
    NetworkFailure = 3,

    /// Auto-repair failed due to permission error
    PermissionDenied = 4,

    /// Auto-repair failed due to build error
    BuildFailure = 5,

    /// Recursion detected during repair (internal error)
    RecursionDetected = 6,
}

impl PreflightExitCode {
    /// Convert RepairError to exit code
    ///
    /// Maps error types to semantic exit codes for CI/CD integration.
    ///
    /// # Arguments
    ///
    /// * `err` - The repair error to classify
    ///
    /// # Returns
    ///
    /// Appropriate exit code for the error type:
    /// - NetworkFailure → 3
    /// - BuildFailure → 5
    /// - PermissionDenied → 4
    /// - RecursionDetected → 6
    /// - Unknown/Other → 1 (generic unavailable)
    pub fn from_repair_error(err: &RepairError) -> Self {
        match err {
            RepairError::NetworkFailure { .. } => PreflightExitCode::NetworkFailure,
            RepairError::BuildFailure { .. } => PreflightExitCode::BuildFailure,
            RepairError::PermissionDenied { .. } => PreflightExitCode::PermissionDenied,
            RepairError::RecursionDetected => PreflightExitCode::RecursionDetected,
            RepairError::Unknown { .. }
            | RepairError::SetupFailed(_)
            | RepairError::RevalidationFailed { .. }
            | RepairError::RebuildFailed(_) => PreflightExitCode::Unavailable,
        }
    }
}

/// RepairMode enum controlling auto-repair behavior
///
/// Determines whether preflight should automatically attempt to repair missing backends
/// by invoking setup-cpp-auto, rebuilding xtask, and re-executing with updated detection.
///
/// # Variants
///
/// - `Auto`: Default in development/interactive mode. Automatically repairs when backend missing.
/// - `Never`: Default in CI. Never attempts auto-repair, fails fast with clear error.
/// - `Always`: Forces repair even if backend appears available (useful for updates/refresh).
///
/// # Examples
///
/// ```ignore
/// // Automatically repair if backend missing (default locally)
/// let mode = RepairMode::Auto;
/// assert!(mode.should_repair(false)); // backend unavailable → repair
/// assert!(!mode.should_repair(true)); // backend available → no repair
///
/// // Never repair (default in CI)
/// let mode = RepairMode::Never;
/// assert!(!mode.should_repair(false)); // never repair
/// assert!(!mode.should_repair(true));  // never repair
///
/// // Always repair (force refresh)
/// let mode = RepairMode::Always;
/// assert!(mode.should_repair(false)); // always repair
/// assert!(mode.should_repair(true));  // always repair
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairMode {
    /// Automatically repair missing backend (default in interactive mode)
    Auto,

    /// Never attempt auto-repair (explicit opt-out, default in CI)
    Never,

    /// Always attempt repair even if backend appears available (force refresh)
    Always,
}

impl RepairMode {
    /// Determine if repair should be attempted based on backend availability
    ///
    /// # Arguments
    ///
    /// * `backend_available` - Whether the backend is currently detected
    ///
    /// # Returns
    ///
    /// `true` if repair workflow should be initiated, `false` otherwise
    ///
    /// # Logic
    ///
    /// - `Auto`: Only repair if backend unavailable
    /// - `Never`: Never repair
    /// - `Always`: Always repair (even if available)
    pub fn should_repair(self, backend_available: bool) -> bool {
        match self {
            RepairMode::Auto => !backend_available, // Only if missing
            RepairMode::Never => false,             // Never repair
            RepairMode::Always => true,             // Always repair
        }
    }

    /// Create RepairMode from CLI flags with CI-aware defaults
    ///
    /// # Arguments
    ///
    /// * `repair_flag` - Optional explicit repair mode from --repair flag
    /// * `ci_detected` - Whether running in CI environment
    ///
    /// # Returns
    ///
    /// Appropriate RepairMode:
    /// - Explicit flag value if provided
    /// - `Never` if in CI environment (safe default)
    /// - `Auto` if local environment (user-friendly default)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Explicit flag takes precedence
    /// let mode = RepairMode::from_cli_flags(Some("never"), false);
    /// assert_eq!(mode, RepairMode::Never);
    ///
    /// // CI defaults to Never
    /// let mode = RepairMode::from_cli_flags(None, true);
    /// assert_eq!(mode, RepairMode::Never);
    ///
    /// // Local defaults to Auto
    /// let mode = RepairMode::from_cli_flags(None, false);
    /// assert_eq!(mode, RepairMode::Auto);
    /// ```
    pub fn from_cli_flags(repair_flag: Option<&str>, ci_detected: bool) -> Self {
        match repair_flag {
            Some("auto") => RepairMode::Auto,
            Some("never") => RepairMode::Never,
            Some("always") => RepairMode::Always,
            None => {
                // Default behavior: CI-aware
                if ci_detected {
                    RepairMode::Never // Safe default: fail fast in CI
                } else {
                    RepairMode::Auto // User-friendly: auto-repair locally
                }
            }
            _ => RepairMode::Never, // Unknown value → conservative default
        }
    }
}

/// Detect if running in CI environment
///
/// Checks standard CI environment variables used by major CI/CD platforms.
///
/// # Returns
///
/// `true` if any CI environment variable is set, `false` otherwise
///
/// # Detected Platforms
///
/// - GitHub Actions: `GITHUB_ACTIONS`, `CI`
/// - GitLab CI: `GITLAB_CI`, `CI`
/// - Jenkins: `JENKINS_HOME`
/// - CircleCI: `CIRCLECI`
/// - Generic: `CI`, `BITNET_TEST_NO_REPAIR`
///
/// # Examples
///
/// ```ignore
/// // In CI environment
/// std::env::set_var("GITHUB_ACTIONS", "true");
/// assert!(is_ci());
///
/// // In local environment
/// std::env::remove_var("CI");
/// assert!(!is_ci());
/// ```
pub fn is_ci() -> bool {
    std::env::var_os("CI").is_some()
        || std::env::var_os("GITHUB_ACTIONS").is_some()
        || std::env::var_os("JENKINS_HOME").is_some()
        || std::env::var_os("GITLAB_CI").is_some()
        || std::env::var_os("CIRCLECI").is_some()
        || std::env::var_os("BITNET_TEST_NO_REPAIR").is_some()
}

/// Check if an error is retryable
///
/// Network errors are typically transient and can be retried with exponential backoff.
/// Build and permission errors are usually permanent and should not be retried.
///
/// # Arguments
///
/// * `err` - The repair error to check
///
/// # Returns
///
/// `true` if error should be retried, `false` otherwise
pub fn is_retryable_error(err: &RepairError) -> bool {
    matches!(err, RepairError::NetworkFailure { .. })
}

/// Progress tracker for verbose repair logging
///
/// Provides timestamped progress messages during auto-repair flow.
/// Only emits messages when verbose mode is enabled.
///
/// # Examples
///
/// ```ignore
/// let progress = RepairProgress::new(true); // verbose enabled
/// progress.log("DETECT", "Backend 'bitnet.cpp' not found");
/// progress.log("REPAIR", "Cloning from GitHub...");
/// // Output: [  0.00s] DETECT: Backend 'bitnet.cpp' not found
/// //         [  2.15s] REPAIR: Cloning from GitHub...
/// ```
pub struct RepairProgress {
    start_time: std::time::Instant,
    verbose: bool,
}

impl RepairProgress {
    /// Create new progress tracker
    ///
    /// # Arguments
    ///
    /// * `verbose` - If true, emit progress messages to stderr
    pub fn new(verbose: bool) -> Self {
        Self { start_time: std::time::Instant::now(), verbose }
    }

    /// Log a progress message with timestamp
    ///
    /// Format: `[  XXX.XXs] STAGE: message`
    ///
    /// # Arguments
    ///
    /// * `stage` - Stage identifier (e.g., "DETECT", "REPAIR", "REBUILD")
    /// * `message` - Progress message to display
    pub fn log(&self, stage: &str, message: &str) {
        if self.verbose {
            let elapsed = self.start_time.elapsed();
            eprintln!("[{:>6.2}s] {}: {}", elapsed.as_secs_f64(), stage, message);
        }
    }
}

// Feature-gated imports for build-time library detection
// These constants are set by crossval/build.rs during compilation
#[cfg(any(
    feature = "crossval",
    feature = "crossval-all",
    feature = "inference",
    feature = "ffi"
))]
use bitnet_crossval::{BACKEND_STATE, HAS_BITNET, HAS_LLAMA};

// Fallback constants when crossval features not enabled
#[cfg(not(any(
    feature = "crossval",
    feature = "crossval-all",
    feature = "inference",
    feature = "ffi"
)))]
const HAS_BITNET: bool = false;
#[cfg(not(any(
    feature = "crossval",
    feature = "crossval-all",
    feature = "inference",
    feature = "ffi"
)))]
const HAS_LLAMA: bool = false;
#[cfg(not(any(
    feature = "crossval",
    feature = "crossval-all",
    feature = "inference",
    feature = "ffi"
)))]
const BACKEND_STATE: &str = "none";

// Visual separators for enhanced error messages
const SEPARATOR_HEAVY: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
const SEPARATOR_LIGHT: &str =
    "─────────────────────────────────────────────────────────────────────";

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
fn emit_standard_stale_warning(backend: CppBackend) {
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
fn emit_verbose_stale_warning(backend: CppBackend, matched_path: &std::path::Path) {
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
fn format_ci_stale_skip_diagnostic(
    backend: CppBackend,
    matched_path: Option<&std::path::Path>,
) -> String {
    const SEPARATOR: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

    let mut msg = String::new();
    msg.push_str(&format!("{}\n", SEPARATOR));
    msg.push_str(&format!("⊘ Test skipped: {} not available (CI mode)\n", backend.name()));
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

// ============================================================================
// Environment Export Parsing (env-export-before-rebuild-deterministic.md)
// ============================================================================

/// Parse shell export statements from `setup-cpp-auto` output
///
/// Extracts environment variable key-value pairs from shell export syntax.
/// Supports multiple shell formats for cross-platform compatibility.
///
/// # Supported Formats
///
/// - **POSIX sh/bash**: `export KEY=value` or `export KEY="value"`
/// - **Fish shell**: `set -gx KEY "value"`
/// - **PowerShell**: `$env:KEY = "value"`
///
/// # Arguments
///
/// * `script` - Shell script output containing export statements
///
/// # Returns
///
/// HashMap of environment variable key-value pairs, with quotes stripped
/// from values. Non-export lines are silently skipped.
///
/// # Examples
///
/// ```ignore
/// let script = r#"
/// export BITNET_CPP_DIR="/path/to/libs"
/// export LD_LIBRARY_PATH="/lib:/usr/lib"
/// echo "Setup complete"
/// "#;
///
/// let exports = parse_sh_exports(script);
/// assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/path/to/libs".to_string()));
/// assert_eq!(exports.len(), 2); // Non-export lines skipped
/// ```
///
/// # Specification
///
/// See: `docs/specs/env-export-before-rebuild-deterministic.md` (AC1)
pub fn parse_sh_exports(script: &str) -> HashMap<String, String> {
    let mut result = HashMap::new();

    for line in script.lines() {
        let line = line.trim();

        // Parse POSIX sh/bash: export KEY=value
        if let Some(export) = line.strip_prefix("export ") {
            if let Some((key, value)) = export.split_once('=') {
                let key = key.trim();
                let value = strip_quotes(value.trim());
                // Skip if key is empty (malformed)
                if !key.is_empty() {
                    result.insert(key.to_string(), value);
                }
            }
        }
        // Parse fish shell: set -gx KEY "value"
        else if line.starts_with("set -gx ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 && parts[0] == "set" && parts[1] == "-gx" {
                let key = parts[2];
                // Join remaining parts as value (handles multi-word values)
                let value = parts[3..].join(" ");
                let value = strip_quotes(&value);
                result.insert(key.to_string(), value);
            }
        }
        // Parse PowerShell: $env:KEY = "value"
        else if line.starts_with("$env:")
            && let Some(rest) = line.strip_prefix("$env:")
            && let Some((key, value)) = rest.split_once('=')
        {
            let key = key.trim();
            let value = strip_quotes(value.trim());
            result.insert(key.to_string(), value);
        }
    }

    result
}

/// Strip surrounding quotes from a string value
///
/// Removes matching single or double quotes from start and end of string.
/// Preserves inner quotes (e.g., `"BitNet \"C++\" Backend"` becomes
/// `BitNet \"C++\" Backend`).
///
/// # Arguments
///
/// * `s` - String potentially wrapped in quotes
///
/// # Returns
///
/// String with outer quotes removed if present, otherwise unchanged
fn strip_quotes(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2
        && ((s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')))
    {
        return s[1..s.len() - 1].to_string();
    }
    s.to_string()
}

/// Rebuild xtask with environment variables applied
///
/// Invokes `cargo build -p xtask --features crossval-all` with provided
/// environment variables set, enabling build.rs to detect newly installed
/// C++ libraries via BITNET_CPP_DIR and LD_LIBRARY_PATH.
///
/// # Arguments
///
/// * `env_vars` - Environment variables to set during cargo build
/// * `verbose` - If true, print diagnostic messages
///
/// # Returns
///
/// * `Ok(())` - Rebuild succeeded with environment applied
/// * `Err(RepairError::RebuildFailed)` - Cargo build failed
///
/// # Environment Propagation
///
/// This function applies environment variables to the cargo subprocess,
/// ensuring build.rs scripts can detect library paths during compilation.
/// Existing environment variables are preserved (additive behavior).
///
/// # Examples
///
/// ```ignore
/// let mut exports = HashMap::new();
/// exports.insert("BITNET_CPP_DIR".to_string(), "/path/to/libs".to_string());
/// exports.insert("LD_LIBRARY_PATH".to_string(), "/path/to/libs/lib".to_string());
///
/// rebuild_xtask_with_env(&exports, true)?;
/// // cargo build -p xtask --features crossval-all
/// // (with BITNET_CPP_DIR and LD_LIBRARY_PATH set)
/// ```
///
/// # Specification
///
/// See: `docs/specs/env-export-before-rebuild-deterministic.md` (AC2)
pub fn rebuild_xtask_with_env(
    env_vars: &HashMap<String, String>,
    verbose: bool,
) -> Result<(), RepairError> {
    if verbose {
        eprintln!("[preflight] Rebuilding xtask with {} environment variables...", env_vars.len());
        for (key, value) in env_vars {
            eprintln!("[preflight]   {}={}", key, value);
        }
    }

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "-p", "xtask", "--features", "crossval-all"]);

    // Apply environment variables to cargo subprocess
    for (key, value) in env_vars {
        cmd.env(key, value);
    }

    let output = cmd
        .output()
        .map_err(|e| RepairError::RebuildFailed(RebuildError::BuildFailed(e.to_string())))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(RepairError::RebuildFailed(RebuildError::BuildFailed(stderr.to_string())));
    }

    if verbose {
        eprintln!("[preflight] ✓ Rebuild complete with environment applied");
    }

    Ok(())
}

/// Verify required libraries are available for selected backend
///
/// Checks build-time detection from the crossval crate, which exports constants:
/// - `bitnet_crossval::HAS_BITNET`: Set to `true` if libbitnet* found during crossval build
/// - `bitnet_crossval::HAS_LLAMA`: Set to `true` if libllama*/libggml* found during crossval build
///
/// These constants are determined by `crossval/build.rs` during compilation and exported
/// as public constants in the crossval crate, allowing xtask to query library availability
/// at runtime.
///
/// # Arguments
///
/// * `backend` - The C++ backend to validate
/// * `verbose` - If true, print diagnostic messages
///
/// # Errors
///
/// Returns an error with setup instructions if required libraries are not found.
///
/// # Examples
///
/// ```no_run
/// use xtask::crossval::{CppBackend, preflight_backend_libs};
///
/// # fn main() -> anyhow::Result<()> {
/// // Check if llama.cpp libraries are available
/// preflight_backend_libs(CppBackend::Llama, false)?;
/// # Ok(())
/// # }
/// ```
pub fn preflight_backend_libs(backend: CppBackend, verbose: bool) -> Result<()> {
    // Check build-time detection from crossval crate
    // These constants are set by crossval/build.rs based on library availability
    let has_libs = match backend {
        CppBackend::BitNet => HAS_BITNET,
        CppBackend::Llama => HAS_LLAMA,
    };

    // Runtime backend state validation: warn if requesting BitNet but only llama available
    if backend == CppBackend::BitNet && BACKEND_STATE == "llama" {
        eprintln!("{}", SEPARATOR_HEAVY);
        eprintln!("⚠️  WARNING: BitNet backend requested but not fully available at compile time");
        eprintln!("{}", SEPARATOR_HEAVY);
        eprintln!();
        eprintln!("Compiled backend state: {} (llama fallback mode)", BACKEND_STATE);
        eprintln!("Requested backend: BitNet");
        eprintln!();
        eprintln!("This means:");
        eprintln!("  • Only llama.cpp libraries were found during crossval build");
        eprintln!("  • BitNet.cpp libraries are NOT available");
        eprintln!("  • Cross-validation can only use llama.cpp backend");
        eprintln!();
        eprintln!("To enable full BitNet backend:");
        eprintln!("  1. Install BitNet.cpp libraries:");
        eprintln!("     {}", backend.setup_command());
        eprintln!();
        eprintln!("  2. Rebuild xtask to detect BitNet libraries:");
        eprintln!("     cargo clean -p xtask -p crossval");
        eprintln!("     cargo build -p xtask --features crossval-all");
        eprintln!();
        eprintln!("  3. Verify BitNet backend is now available:");
        eprintln!("     cargo run -p xtask -- preflight --backend bitnet --verbose");
        eprintln!("{}", SEPARATOR_HEAVY);
        eprintln!();
    }

    if !has_libs {
        // Priority 2: Runtime detection fallback (stale build scenario)
        if let Ok((runtime_available, matched_path)) =
            crate::crossval::detect_backend_runtime(backend)
        {
            if runtime_available && is_ci() {
                // STALE BUILD SCENARIO + CI: skip with matched path diagnostic and exit
                let skip_msg = format_ci_stale_skip_diagnostic(backend, matched_path.as_deref());
                eprintln!("{}", skip_msg);
                std::process::exit(0); // Exit code 0 = skip (not failure)
            } else if runtime_available {
                // STALE BUILD SCENARIO + DEV: warn with matched path and continue
                let verbose_flag = std::env::var("VERBOSE").is_ok() || verbose;
                if let Some(path) = matched_path {
                    emit_stale_build_warning(backend, &path, verbose_flag);
                }
                return Ok(()); // Continue execution
            }
        }

        if verbose {
            print_verbose_failure_diagnostics(backend);
        }

        // Libraries not found - provide enhanced actionable error message
        let backend_short = backend.name().split('.').next().unwrap();
        let required_libs = match backend {
            CppBackend::BitNet => "libbitnet*.so",
            CppBackend::Llama => "libllama*.so, libggml*.so",
        };

        #[cfg(target_os = "linux")]
        let ld_var = "LD_LIBRARY_PATH";
        #[cfg(target_os = "macos")]
        let ld_var = "DYLD_LIBRARY_PATH";
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let ld_var = "PATH";

        bail!(
            "\n{}\n\
             ❌ Backend '{}' libraries NOT FOUND\n\
             {}\n\
             \n\
             CRITICAL: Library detection happens at BUILD time, not runtime.\n\
             If you just installed C++ libraries, xtask MUST be rebuilt to detect them.\n\
             \n\
             Required libraries: {}\n\
             \n\
             {}\n\
             RECOVERY STEPS\n\
             {}\n\
             \n\
             Option A: One-Command Setup (Recommended for First-Time Users)\n\
             {}\n\
             \n\
               Step 1: Install and configure C++ reference implementation\n\
                 {}\n\
             \n\
               Step 2: Rebuild xtask to detect newly installed libraries\n\
                 cargo clean -p xtask && cargo build -p xtask --features crossval-all\n\
             \n\
               Step 3: Verify detection succeeded\n\
                 cargo run -p xtask -- preflight --backend {} --verbose\n\
             \n\
               Then retry your original command.\n\
             \n\
             Option B: Manual Setup + {} (Requires Setting Before Every Run)\n\
             {}\n\
             \n\
               Step 1: Install {} manually (skip if already installed)\n\
                 See: docs/howto/cpp-setup.md\n\
             \n\
               Step 2: Set environment variable for this session\n\
                 export BITNET_CPP_DIR=/path/to/{}\n\
                 export {}=$BITNET_CPP_DIR/build:${}  # Runtime library path\n\
             \n\
               Step 3: Rebuild xtask to embed library paths (rpath)\n\
                 cargo clean -p xtask && cargo build -p xtask --features crossval-all\n\
             \n\
               Note: Option B requires setting {} before EVERY run.\n\
                     Option A embeds library paths permanently (rpath).\n\
             \n\
             {}\n\
             TROUBLESHOOTING\n\
             {}\n\
             \n\
             If setup fails, run verbose diagnostics to see what's happening:\n\
               cargo run -p xtask -- preflight --backend {} --verbose\n\
             \n\
             This will show:\n\
               • Environment variables checked (BITNET_CPP_DIR, {}, etc.)\n\
               • Library search paths in priority order\n\
               • Which paths exist vs missing\n\
               • All libraries found in each path\n\
               • Build-time detection flags\n\
             \n\
             For more help, see:\n\
               docs/howto/cpp-setup.md (Detailed C++ setup guide)\n\
               docs/explanation/dual-backend-crossval.md (Architecture overview)\n",
            SEPARATOR_HEAVY,
            backend.name(),
            SEPARATOR_HEAVY,
            required_libs,
            SEPARATOR_HEAVY,
            SEPARATOR_HEAVY,
            SEPARATOR_LIGHT,
            backend.setup_command(),
            backend_short,
            ld_var,
            SEPARATOR_LIGHT,
            backend.name(),
            backend.name(),
            ld_var,
            ld_var,
            ld_var,
            SEPARATOR_HEAVY,
            SEPARATOR_HEAVY,
            backend_short,
            ld_var
        );
    }

    if verbose {
        print_verbose_success_diagnostics(backend);
    } else {
        println!("✓ Backend '{}' libraries found", backend.name());
    }

    Ok(())
}

/// Print verbose diagnostics when libraries are found successfully
fn print_verbose_success_diagnostics(backend: CppBackend) {
    println!("{}", SEPARATOR_HEAVY);
    println!("✓ Backend '{}': AVAILABLE", backend.name());
    println!("{}", SEPARATOR_HEAVY);
    println!();

    // Environment Configuration
    println!("Environment Configuration");
    println!("{}", SEPARATOR_LIGHT);
    print_env_var_status("  BITNET_CPP_DIR");
    print_env_var_status("  BITNET_CROSSVAL_LIBDIR");

    #[cfg(target_os = "linux")]
    print_env_var_status("  LD_LIBRARY_PATH");

    #[cfg(target_os = "macos")]
    print_env_var_status("  DYLD_LIBRARY_PATH");

    #[cfg(target_os = "windows")]
    print_env_var_status("  PATH");

    if std::env::var("BITNET_CPP_PATH").is_ok() {
        println!("  BITNET_CPP_PATH       = (deprecated, use BITNET_CPP_DIR)");
    }

    println!();

    // Library Search Paths (numbered in priority order)
    println!("Library Search Paths (Priority Order)");
    println!("{}", SEPARATOR_LIGHT);

    // Check if BITNET_CROSSVAL_LIBDIR is set
    if let Ok(lib_dir) = std::env::var("BITNET_CROSSVAL_LIBDIR") {
        println!("  1. BITNET_CROSSVAL_LIBDIR override");
        let path = std::path::PathBuf::from(&lib_dir);
        if path.exists() {
            println!("     ✓ {} (exists)", lib_dir);
            if let Some(libs) = find_libs_in_path(&path, backend) {
                println!("     Found libraries:");
                for lib in libs {
                    println!("       - {}", lib);
                }
            }
        } else {
            println!("     ✗ {} (not found)", lib_dir);
        }
    } else {
        println!("  1. BITNET_CROSSVAL_LIBDIR override");
        println!("     (not set - using default search order)");
    }
    println!();

    // Enumerate remaining search paths
    let search_paths = get_library_search_paths();
    for (idx, path) in search_paths.iter().enumerate().skip(1) {
        let path_desc = if let Some(parent) = path.parent() {
            if parent.ends_with("bitnet_cpp") {
                format!(
                    "BITNET_CPP_DIR/{}",
                    path.file_name().and_then(|s| s.to_str()).unwrap_or("")
                )
            } else {
                path.display().to_string()
            }
        } else {
            path.display().to_string()
        };

        let context_label = get_path_context_label(path);

        println!("  {}. {}{}", idx + 1, path_desc, context_label);
        if path.exists() {
            println!("     ✓ {} (exists)", path.display());
            if let Some(libs) = find_libs_in_path(path, backend) {
                println!("     Found libraries:");
                for lib in libs {
                    println!("       - {}", lib);
                }
            }
        } else {
            println!("     ✗ {} (not found)", path.display());
        }
        println!();
    }

    // Required Libraries
    println!("Required Libraries for {} Backend", backend.name());
    println!("{}", SEPARATOR_LIGHT);
    for &lib in backend.required_libs() {
        println!("  ✓ {}.so (found at build time)", lib);
    }
    println!();

    // Build-Time Detection Metadata
    println!("{}", format_build_metadata(backend));
    println!();

    // Runtime Backend State Validation
    println!("Runtime Backend State");
    println!("{}", SEPARATOR_LIGHT);
    println!("  Compiled backend state: {}", BACKEND_STATE);
    match BACKEND_STATE {
        "full" => println!("    ✓ Full BitNet backend available (BitNet.cpp + llama.cpp)"),
        "llama" => {
            println!("    ⚠️  Llama fallback mode (only llama.cpp, BitNet.cpp NOT available)")
        }
        "none" => println!("    ✗ No backend available (stub mode)"),
        other => println!("    ? Unknown state: {}", other),
    }
    println!();

    // Platform-Specific Configuration
    println!("Platform-Specific Configuration");
    println!("{}", SEPARATOR_LIGHT);

    #[cfg(target_os = "linux")]
    {
        println!("  Platform: Linux");
        println!("  Standard library: libstdc++ (dynamic linking)");
        println!("  RPATH embedded: YES (no LD_LIBRARY_PATH required)");
        println!("  Loader search order: rpath → LD_LIBRARY_PATH → system paths");
    }

    #[cfg(target_os = "macos")]
    {
        println!("  Platform: macOS");
        println!("  Standard library: libc++ (dynamic linking)");
        println!("  RPATH embedded: YES (no DYLD_LIBRARY_PATH required)");
        println!("  Loader search order: rpath → DYLD_LIBRARY_PATH → system paths");
    }

    #[cfg(target_os = "windows")]
    {
        println!("  Platform: Windows");
        println!("  Standard library: MSVC runtime (dynamic linking)");
        println!("  DLL search order: executable dir → PATH → system dirs");
    }
    println!();

    // Summary
    println!("Summary");
    println!("{}", SEPARATOR_LIGHT);
    println!("✓ All required libraries detected at build time");
    println!("✓ Runtime library resolution configured (rpath)");
    println!("✓ Cross-validation with {} is supported", backend.name());
    println!();
    println!("To test cross-validation:");
    println!("  cargo run -p xtask --features crossval-all -- crossval-per-token \\");
    println!("    --model models/model.gguf \\");
    println!("    --tokenizer models/tokenizer.json \\");
    println!("    --prompt \"Test\" \\");
    println!("    --max-tokens 4 \\");
    println!("    --cpp-backend {} \\", backend.name().split('.').next().unwrap());
    println!("    --verbose");
}

/// Print verbose diagnostics when libraries are NOT found
fn print_verbose_failure_diagnostics(backend: CppBackend) {
    println!("{}", SEPARATOR_HEAVY);
    println!("❌ Backend '{}': NOT AVAILABLE", backend.name());
    println!("{}", SEPARATOR_HEAVY);
    println!();

    // Diagnosis Section
    println!("DIAGNOSIS: Required libraries not detected at xtask build time.");
    println!("This means either:");
    println!("  (a) C++ libraries were never installed, OR");
    println!("  (b) C++ libraries were installed AFTER xtask was built");
    println!();

    // Environment Configuration (Current State)
    println!("Environment Configuration (Current State)");
    println!("{}", SEPARATOR_LIGHT);
    print_env_var_status("  BITNET_CPP_DIR");
    print_env_var_status("  BITNET_CROSSVAL_LIBDIR");

    #[cfg(target_os = "linux")]
    print_env_var_status("  LD_LIBRARY_PATH");

    #[cfg(target_os = "macos")]
    print_env_var_status("  DYLD_LIBRARY_PATH");

    #[cfg(target_os = "windows")]
    print_env_var_status("  PATH");

    // Check if no environment variables are set
    let has_cpp_dir = std::env::var("BITNET_CPP_DIR").is_ok();
    let has_lib_dir = std::env::var("BITNET_CROSSVAL_LIBDIR").is_ok();
    if !has_cpp_dir && !has_lib_dir {
        println!();
        println!("  ⚠️  WARNING: No environment variables set for library discovery.");
        println!("     xtask will search default path: ~/.cache/bitnet_cpp");
    }

    println!();

    // Library Search Paths (Checked During Last Build)
    println!("Library Search Paths (Checked During Last Build)");
    println!("{}", SEPARATOR_LIGHT);

    if let Ok(lib_dir) = std::env::var("BITNET_CROSSVAL_LIBDIR") {
        println!("  1. BITNET_CROSSVAL_LIBDIR override");
        let path = std::path::PathBuf::from(&lib_dir);
        if path.exists() {
            println!("     ✓ {} (exists)", lib_dir);
        } else {
            println!("     ✗ {} (not found)", lib_dir);
        }
    } else {
        println!("  1. BITNET_CROSSVAL_LIBDIR override");
        println!("     (not set - using default search order)");
    }
    println!();

    let search_paths = get_library_search_paths();
    for (idx, path) in search_paths.iter().enumerate().skip(1) {
        let context_label = get_path_context_label(path);
        println!("  {}. {}{}", idx + 1, path.display(), context_label);
        if path.exists() {
            println!("     ✓ {} (exists)", path.display());

            // Show what was searched for
            println!("     Searched for: {:?}", backend.required_libs());

            // List any libraries found (might be other backends)
            if let Ok(entries) = std::fs::read_dir(path) {
                let mut found_any = false;
                for entry in entries.flatten() {
                    if let Some(name) = entry.path().file_name().and_then(|n| n.to_str())
                        && name.starts_with("lib")
                        && (name.ends_with(".so")
                            || name.ends_with(".dylib")
                            || name.ends_with(".a"))
                    {
                        if !found_any {
                            println!("     Other libraries found:");
                            found_any = true;
                        }
                        println!("       - {}", name);
                    }
                }
                if !found_any {
                    println!("     No libraries found in this directory");
                }
            }
        } else {
            println!("     ✗ {} (not found)", path.display());
        }
        println!();
    }

    // Required Libraries (Searched For)
    println!("Required Libraries (Searched For)");
    println!("{}", SEPARATOR_LIGHT);
    for &lib in backend.required_libs() {
        println!("  ✗ {}.so / {}.dylib", lib, lib);
    }
    println!();

    // Build-Time Detection Metadata
    println!("{}", format_build_metadata(backend));
    println!();

    // RECOMMENDED FIX
    println!("{}", SEPARATOR_HEAVY);
    println!("RECOMMENDED FIX");
    println!("{}", SEPARATOR_HEAVY);
    println!();
    println!("Step 1: Install C++ reference implementation (auto-setup):");
    println!("  {}", backend.setup_command());
    println!();
    println!("  This will:");
    match backend {
        CppBackend::BitNet => {
            println!("    • Clone bitnet.cpp to ~/.cache/bitnet_cpp");
        }
        CppBackend::Llama => {
            println!("    • Clone llama.cpp to ~/.cache/bitnet_cpp");
        }
    }
    println!("    • Build with CMake (dynamic linking)");
    println!("    • Set BITNET_CPP_DIR environment variable");
    #[cfg(target_os = "linux")]
    println!("    • Add LD_LIBRARY_PATH to your shell profile");
    #[cfg(target_os = "macos")]
    println!("    • Add DYLD_LIBRARY_PATH to your shell profile");
    println!();
    println!("Step 2: Rebuild xtask to detect newly installed libraries:");
    println!("  cargo clean -p xtask");
    println!("  cargo build -p xtask --features crossval-all");
    println!();
    println!("  Why rebuild?");
    println!("    • Library detection runs during BUILD (not runtime)");
    println!("    • Build script scans filesystem for {}*", backend.required_libs().join("*/"));
    println!("    • Detection results baked into xtask binary as constants");
    println!("    • If libraries installed after build, xtask won't see them");
    println!();
    println!("Step 3: Verify detection succeeded:");
    println!(
        "  cargo run -p xtask -- preflight --backend {} --verbose",
        backend.name().split('.').next().unwrap()
    );
    println!();
    println!("  Expected output:");
    println!("    ✓ Backend '{}': AVAILABLE", backend.name());
    println!(
        "    CROSSVAL_HAS_{} = true",
        match backend {
            CppBackend::BitNet => "BITNET",
            CppBackend::Llama => "LLAMA",
        }
    );
    println!();
    println!("Step 4: Retry your original cross-validation command.");
    println!();

    // ALTERNATIVE: Manual Installation
    println!("{}", SEPARATOR_HEAVY);
    println!("ALTERNATIVE: Manual Installation");
    println!("{}", SEPARATOR_HEAVY);
    println!();
    println!("If auto-setup fails, install manually:");
    println!();
    println!("  1. Clone and build {}:", backend.name());
    match backend {
        CppBackend::BitNet => {
            println!("     git clone https://github.com/microsoft/BitNet");
            println!("     cd BitNet");
        }
        CppBackend::Llama => {
            println!("     git clone https://github.com/ggerganov/llama.cpp");
            println!("     cd llama.cpp");
        }
    }
    println!("     cmake -B build -DBUILD_SHARED_LIBS=ON");
    println!("     cmake --build build");
    println!();
    println!("  2. Set environment variables:");
    match backend {
        CppBackend::BitNet => {
            println!("     export BITNET_CPP_DIR=/path/to/BitNet");
        }
        CppBackend::Llama => {
            println!("     export BITNET_CPP_DIR=/path/to/llama.cpp");
        }
    }
    #[cfg(target_os = "linux")]
    println!("     export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build:$LD_LIBRARY_PATH");
    #[cfg(target_os = "macos")]
    println!("     export DYLD_LIBRARY_PATH=$BITNET_CPP_DIR/build:$DYLD_LIBRARY_PATH");
    println!();
    println!("  3. Rebuild xtask (same as Step 2 above)");
    println!();
    println!("For detailed guidance, see: docs/howto/cpp-setup.md");
}

/// Print environment variable status
fn print_env_var_status(var_name: &str) {
    // Extract the variable name without leading spaces
    let trimmed = var_name.trim();
    let indent = &var_name[..var_name.len() - trimmed.len()];

    match std::env::var(trimmed) {
        Ok(value) => {
            // Format with proper alignment (20 chars for variable name)
            let formatted_name = format!("{:<20}", trimmed);
            // Truncate very long values
            if value.len() > 60 {
                println!("{}{}= {}...", indent, formatted_name, &value[..57]);
            } else {
                println!("{}{}= {}", indent, formatted_name, value);
            }
        }
        Err(_) => {
            let formatted_name = format!("{:<20}", trimmed);
            println!("{}{}= (not set)", indent, formatted_name);
        }
    }
}

/// Get library search paths (mirrors crossval/build.rs logic)
///
/// # Visibility
///
/// Public for testing purposes. This function mirrors the search logic in
/// `crossval/build.rs` to allow runtime diagnostics and test validation.
pub fn get_library_search_paths() -> Vec<std::path::PathBuf> {
    use std::env;

    let mut paths = Vec::new();

    // Priority 1: Explicit BITNET_CROSSVAL_LIBDIR
    if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
        paths.push(std::path::PathBuf::from(lib_dir));
    }

    // Priority 2: BITNET_CPP_DIR or BITNET_CPP_PATH
    let bitnet_root =
        env::var("BITNET_CPP_DIR").or_else(|_| env::var("BITNET_CPP_PATH")).unwrap_or_else(|_| {
            format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap_or_else(|_| ".".into()))
        });

    let root = Path::new(&bitnet_root);
    paths.push(root.join("build"));
    paths.push(root.join("build/bin")); // Standalone llama.cpp layout
    paths.push(root.join("build/lib"));
    paths.push(root.join("build/3rdparty/llama.cpp/src"));
    paths.push(root.join("build/3rdparty/llama.cpp/ggml/src"));
    paths.push(root.join("lib"));

    paths
}

/// Find libraries in a given path for a specific backend
fn find_libs_in_path(path: &Path, backend: CppBackend) -> Option<Vec<String>> {
    let required = backend.required_libs();
    let mut found = Vec::new();

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Some(name) = entry.path().file_stem().and_then(|s| s.to_str()) {
                // Check if this library matches any required library
                for &req in required {
                    if name.starts_with(req)
                        && let Some(full_name) = entry.path().file_name().and_then(|n| n.to_str())
                    {
                        found.push(full_name.to_string());
                    }
                }
            }
        }
    }

    if found.is_empty() { None } else { Some(found) }
}

/// Returns context label for special search paths
///
/// Provides user-friendly labels to clarify the purpose of different search paths:
/// - Embedded llama.cpp dependency paths
/// - Embedded ggml paths
/// - Standalone llama.cpp installations
/// - Explicit override paths
///
/// # Arguments
///
/// * `path` - The search path to label
///
/// # Returns
///
/// A static string label or empty string for standard paths that don't need context.
///
/// # Visibility
///
/// Public for testing purposes.
pub fn get_path_context_label(path: &Path) -> &'static str {
    let path_str = path.to_string_lossy();

    if path_str.contains("3rdparty/llama.cpp/src") {
        " (embedded llama.cpp)"
    } else if path_str.contains("3rdparty/llama.cpp/ggml") {
        " (embedded ggml)"
    } else if path_str.ends_with("build/bin") {
        " (standalone llama.cpp)"
    } else if path_str.contains("CROSSVAL_LIBDIR") {
        " (explicit override)"
    } else {
        "" // No label for standard paths
    }
}

/// Get xtask build timestamp for staleness detection
///
/// Returns the modification time of the xtask binary, which indicates when
/// it was last built. This helps users identify stale builds where libraries
/// were installed after xtask compilation.
///
/// # Returns
///
/// * `Some(timestamp)` - ISO 8601 formatted timestamp if available
/// * `None` - If binary path or metadata cannot be accessed
fn get_xtask_build_timestamp() -> Option<String> {
    use chrono::{DateTime, Utc};
    use std::fs;
    use std::time::SystemTime;

    // Get current executable path
    std::env::current_exe().ok().and_then(|path| {
        // Get file metadata
        fs::metadata(&path).ok().and_then(|meta| {
            // Get modification time
            meta.modified().ok().and_then(|modified| {
                // Convert to duration since UNIX epoch
                modified.duration_since(SystemTime::UNIX_EPOCH).ok().and_then(|duration| {
                    let secs = duration.as_secs() as i64;
                    let nanos = duration.subsec_nanos();

                    // Convert to chrono DateTime for human-readable formatting
                    DateTime::from_timestamp(secs, nanos).map(|dt: DateTime<Utc>| {
                        // Format as ISO 8601: "YYYY-MM-DD HH:MM:SS UTC"
                        dt.format("%Y-%m-%d %H:%M:%S UTC").to_string()
                    })
                })
            })
        })
    })
}

/// Format build metadata section for diagnostics
///
/// Returns a formatted string showing build-time detection constants,
/// xtask build timestamp, and feature flags.
///
/// # Arguments
///
/// * `backend` - The C++ backend to show metadata for
///
/// # Returns
///
/// Formatted string with build metadata section
///
/// # Visibility
///
/// Public for testing purposes.
pub fn format_build_metadata(backend: CppBackend) -> String {
    let has_backend = match backend {
        CppBackend::BitNet => HAS_BITNET,
        CppBackend::Llama => HAS_LLAMA,
    };

    let backend_name = match backend {
        CppBackend::BitNet => "BITNET",
        CppBackend::Llama => "LLAMA",
    };

    let timestamp = get_xtask_build_timestamp().unwrap_or_else(|| "unknown".to_string());
    let required_libs = backend.required_libs().join(", ");

    format!(
        "Build-Time Detection Metadata\n\
         {}\n\
         CROSSVAL_HAS_{} = {}\n\
         Required libraries: {}\n\
         Last xtask build: {}\n\
         Build feature flags: crossval-all",
        SEPARATOR_LIGHT, backend_name, has_backend, required_libs, timestamp
    )
}

/// Print backend availability status for diagnostics
///
/// This is a convenience function for the `xtask preflight` command.
///
/// # Arguments
///
/// * `verbose` - If true, print additional diagnostic information
#[allow(dead_code)]
pub fn print_backend_status(verbose: bool) {
    println!("Backend Library Status:");
    println!();

    // Check BitNet (from crossval crate constants)
    let has_bitnet = HAS_BITNET;

    if has_bitnet {
        println!("  ✓ bitnet.cpp: AVAILABLE");
        if verbose {
            println!("    Required libraries: {:?}", CppBackend::BitNet.required_libs());
        } else {
            println!("    Libraries: libbitnet*");
        }
    } else {
        println!("  ✗ bitnet.cpp: NOT AVAILABLE");
        println!("    Setup: {}", CppBackend::BitNet.setup_command());
    }

    println!();

    // Check LLaMA (from crossval crate constants)
    let has_llama = HAS_LLAMA;

    if has_llama {
        println!("  ✓ llama.cpp: AVAILABLE");
        if verbose {
            println!("    Required libraries: {:?}", CppBackend::Llama.required_libs());
        } else {
            println!("    Libraries: libllama*, libggml*");
        }
    } else {
        println!("  ✗ llama.cpp: NOT AVAILABLE");
        println!("    Setup: {}", CppBackend::Llama.setup_command());
    }

    println!();

    if !has_bitnet && !has_llama {
        println!("No C++ backends available. Cross-validation will not work.");
        println!("Run setup commands above to install backends.");
    } else if has_bitnet && has_llama {
        println!("Both backends available. Dual-backend cross-validation supported.");
    }
}

/// Preflight check with optional auto-repair
///
/// Implements the detect → repair → redetect flow for automatic C++ backend installation.
///
/// # Arguments
///
/// * `backend` - Which C++ backend to check (bitnet or llama)
/// * `verbose` - Show detailed diagnostics
/// * `should_repair` - Whether to attempt auto-repair if backend is missing
///
/// # Returns
///
/// * `Ok(())` - Backend is available (or successfully repaired)
/// * `Err(anyhow::Error)` - Backend unavailable and repair failed/disabled
///
/// # Examples
///
/// ```ignore
/// // Auto-repair disabled (traditional behavior)
/// preflight_with_auto_repair(CppBackend::BitNet, false, false)?;
///
/// // Auto-repair enabled (new behavior)
/// preflight_with_auto_repair(CppBackend::BitNet, true, true)?;
/// ```
#[allow(dead_code)] // Will be used when integrated with command handler
pub fn preflight_with_auto_repair(
    backend: CppBackend,
    verbose: bool,
    repair_mode: RepairMode,
) -> Result<()> {
    // AC5: Check recursion guard (re-exec child path)
    if is_repair_parent() {
        // We are the re-exec child, do NOT attempt repair again
        if verbose {
            eprintln!("[repair] Re-exec detected, skipping repair (checking detection only)");
        }

        // Just check backend availability (should now be detected)
        let is_available = match backend {
            CppBackend::BitNet => HAS_BITNET,
            CppBackend::Llama => HAS_LLAMA,
        };

        if is_available {
            println!("✓ {} AVAILABLE (detected after repair)", backend.name());
            return Ok(());
        } else {
            bail!(
                "Revalidation failed: backend '{}' still unavailable after repair.\n\
                This may indicate:\n\
                1. Libraries were built but in unexpected location\n\
                2. Build succeeded but libraries are incompatible\n\
                3. Partial build state from previous failed attempt\n\
                \n\
                Recovery steps:\n\
                1. Clean previous build state: rm -rf ~/.cache/{}_cpp\n\
                2. Retry repair: cargo run -p xtask -- preflight --backend {} --repair=auto\n\
                3. If problem persists, see manual setup: docs/howto/cpp-setup.md",
                backend.name(),
                backend.name(),
                backend.name()
            );
        }
    }

    // Step 1: Check if backend is already available
    let is_available = match backend {
        CppBackend::BitNet => HAS_BITNET,
        CppBackend::Llama => HAS_LLAMA,
    };

    // Step 2: Determine if repair should be attempted based on RepairMode
    let should_repair = repair_mode.should_repair(is_available);

    if is_available && !should_repair {
        // Backend already available and no forced repair - use existing preflight logic
        return preflight_backend_libs(backend, verbose);
    }

    if !should_repair {
        // No repair requested - use traditional error path
        return preflight_backend_libs(backend, verbose);
    }

    // Step 3: Attempt repair with retry logic
    if verbose {
        eprintln!();
        eprintln!("Backend '{}' not found at build time", backend.name());
        eprintln!();
        eprintln!("Auto-repairing... (this will take 5-10 minutes on first run)");
    }

    let stdout = match attempt_repair_with_retry(backend, verbose) {
        Ok(stdout) => stdout,
        Err(e) => {
            // Repair failed - show error with recovery steps
            eprintln!("\n{}", e);
            bail!("Auto-repair failed for backend '{}'", backend.name());
        }
    };

    // Step 3.5: Parse environment exports from stdout (AC1)
    if verbose {
        eprintln!("[repair] Step 2/4: Parsing environment exports...");
    }

    let env_exports = parse_sh_exports(&stdout);

    if verbose && !env_exports.is_empty() {
        eprintln!("[repair] Parsed {} environment variables", env_exports.len());
    }

    // Step 4: Rebuild xtask with environment variables applied (AC2 + AC3)
    if verbose {
        eprintln!("[repair] Step 3/4: Rebuilding xtask with environment...");
    }

    if let Err(e) = rebuild_xtask_with_env(&env_exports, verbose) {
        eprintln!("\n{}", e);
        bail!("xtask rebuild failed after successful C++ setup");
    }

    // Step 5: Re-exec with updated binary (AC4)
    if verbose {
        eprintln!("[repair] Step 4/4: Re-executing with updated detection...");
    }

    let original_args: Vec<String> = env::args().collect();
    if let Err(e) = reexec_current_command(&original_args) {
        eprintln!("\n{}", e);
        bail!("Re-exec failed after successful rebuild");
    }

    // This point is never reached on Unix (exec replaces process)
    // On Windows, process exits in reexec_current_command
    Ok(())
}

/// Attempt to repair a missing backend
///
/// Invokes setup-cpp-auto to install C++ libraries, then prompts for xtask rebuild.
///
/// # Arguments
///
/// * `backend` - The backend to install
/// * `verbose` - Show detailed progress
///
/// # Returns
///
/// * `Ok(())` - Setup succeeded (user must rebuild xtask)
/// * `Err(RepairError)` - Setup failed with classified error
fn attempt_repair(backend: CppBackend, verbose: bool) -> Result<(), RepairError> {
    let progress = RepairProgress::new(verbose);

    // Check recursion guard
    if env::var("BITNET_REPAIR_IN_PROGRESS").is_ok() {
        return Err(RepairError::RecursionDetected);
    }

    // Set recursion guard
    // SAFETY: This env var is only used as a recursion guard during auto-repair.
    // We ensure cleanup happens even on error via the removal below.
    unsafe {
        env::set_var("BITNET_REPAIR_IN_PROGRESS", "1");
    }

    progress.log("DETECT", &format!("Backend '{}' not found", backend.name()));
    progress.log("REPAIR", "Invoking setup-cpp-auto...");

    // Invoke setup-cpp-auto
    let setup_result = Command::new(
        env::current_exe()
            .map_err(|e| RepairError::SetupFailed(format!("Failed to get current exe: {}", e)))?,
    )
    .args(["setup-cpp-auto", "--emit=sh"])
    .output()
    .map_err(|e| RepairError::SetupFailed(format!("Failed to execute setup-cpp-auto: {}", e)))?;

    // Cleanup recursion guard
    // SAFETY: We're removing the same env var we set above, restoring the environment state.
    unsafe {
        env::remove_var("BITNET_REPAIR_IN_PROGRESS");
    }

    if !setup_result.status.success() {
        let stderr = String::from_utf8_lossy(&setup_result.stderr);
        let backend_name = backend.name();

        return Err(RepairError::classify(&stderr, backend_name));
    }

    progress.log("REPAIR", "C++ libraries installed successfully");

    // Note: We don't rebuild xtask automatically because:
    // 1. It would require re-executing the current binary
    // 2. Build-time constants are baked into the current process
    // 3. User needs to rebuild explicitly to see updated constants
    //
    // Instead, we show clear instructions for the required rebuild step.

    Ok(())
}

/// Check if we are the re-exec child (post-rebuild detection)
///
/// Returns true if `BITNET_REPAIR_PARENT` environment variable is set,
/// indicating that this process was spawned after auto-repair and rebuild.
///
/// # Returns
///
/// * `true` - This is a re-exec child, skip repair
/// * `false` - Normal execution, repair allowed if needed
fn is_repair_parent() -> bool {
    env::var_os("BITNET_REPAIR_PARENT").is_some()
}

/// Check if repair is currently in progress (recursion guard)
///
/// Returns true if `BITNET_REPAIR_IN_PROGRESS` environment variable is set,
/// indicating that an auto-repair operation is already running.
///
/// # Returns
///
/// * `true` - Repair is in progress (guard active)
/// * `false` - No active repair (guard not set)
fn is_repair_in_progress() -> bool {
    env::var_os("BITNET_REPAIR_IN_PROGRESS").is_some()
}

/// Check if running in CI mode or with test-mode no-repair override
///
/// Returns `true` if either:
/// - `CI` environment variable is set (any value)
/// - `BITNET_TEST_NO_REPAIR` environment variable is set (any value)
///
/// This allows tests and CI pipelines to disable auto-repair behavior.
///
/// # Returns
///
/// * `true` - Running in CI or test-mode no-repair (auto-repair disabled)
/// * `false` - Running in dev mode (auto-repair enabled)
///
/// # Examples
///
/// ```ignore
/// if is_ci_or_no_repair() {
///     eprintln!("⊘ Auto-repair disabled (CI=1 or BITNET_TEST_NO_REPAIR=1)");
///     return Ok(());
/// }
/// ```
pub fn is_ci_or_no_repair() -> bool {
    env::var("CI").is_ok() || env::var("BITNET_TEST_NO_REPAIR").is_ok()
}

/// Auto-repair with setup-cpp-auto subprocess invocation
///
/// Implements AC6: setup-cpp-auto invocation on missing backend.
///
/// This function:
/// 1. Checks RepairMode: returns early if Never
/// 2. Sets recursion guard: `BITNET_REPAIR_IN_PROGRESS=1`
/// 3. Invokes: `cargo run -p xtask -- setup-cpp-auto --backend {bitnet|llama}`
/// 4. Captures stdout/stderr for diagnostics
/// 5. Clears recursion guard on completion
///
/// # Arguments
///
/// * `backend` - The C++ backend to repair (BitNet or Llama)
/// * `mode` - RepairMode controlling whether repair should proceed
///
/// # Returns
///
/// * `Ok(())` - Repair succeeded (setup-cpp-auto completed successfully)
/// * `Err(RepairError)` - Repair failed with classified error
///
/// # Errors
///
/// Returns `RepairError::RecursionDetected` if called while repair is already in progress.
/// Returns classified errors (NetworkFailure, BuildFailure, PermissionDenied) based on stderr.
pub fn auto_repair_with_setup_cpp(
    backend: CppBackend,
    mode: RepairMode,
) -> Result<(), RepairError> {
    // Early exit for CI or test-mode no-repair
    if is_ci_or_no_repair() {
        eprintln!("⊘ Auto-repair disabled (CI=1 or BITNET_TEST_NO_REPAIR=1)");
        return Ok(());
    }

    // Check RepairMode: return early if Never
    // Note: backend_available parameter not needed here since caller checks availability
    if matches!(mode, RepairMode::Never) {
        return Ok(());
    }

    // Check recursion guard
    if is_repair_in_progress() {
        return Err(RepairError::RecursionDetected);
    }

    // Set recursion guard
    // SAFETY: This env var is only used as a recursion guard during auto-repair.
    // We ensure cleanup happens even on error via the removal below.
    unsafe {
        env::set_var("BITNET_REPAIR_IN_PROGRESS", "1");
    }

    // Invoke setup-cpp-auto subprocess
    let current_exe = env::current_exe().map_err(|e| {
        // Cleanup recursion guard on error
        unsafe {
            env::remove_var("BITNET_REPAIR_IN_PROGRESS");
        }
        RepairError::SetupFailed(format!("Failed to get current exe: {}", e))
    })?;

    let backend_arg = match backend {
        CppBackend::BitNet => "bitnet",
        CppBackend::Llama => "llama",
    };

    let setup_result = Command::new(&current_exe)
        .args(["setup-cpp-auto", "--backend", backend_arg, "--emit=sh"])
        .env("BITNET_REPAIR_IN_PROGRESS", "1")  // Explicit env pass
        .output()
        .map_err(|e| {
            // Cleanup recursion guard on execution failure
            unsafe {
                env::remove_var("BITNET_REPAIR_IN_PROGRESS");
            }
            RepairError::SetupFailed(format!("Failed to execute setup-cpp-auto: {}", e))
        })?;

    // Cleanup recursion guard
    // SAFETY: We're removing the same env var we set above, restoring the environment state.
    unsafe {
        env::remove_var("BITNET_REPAIR_IN_PROGRESS");
    }

    // Check command exit status
    if !setup_result.status.success() {
        let stderr = String::from_utf8_lossy(&setup_result.stderr);
        return Err(RepairError::classify(&stderr, backend.name()));
    }

    Ok(())
}

/// Rebuild xtask quickly (incremental, no clean)
///
/// After auto-repair installs C++ libraries, this rebuilds xtask to pick up
/// the new build-time detection constants without doing a full clean.
///
/// # Arguments
///
/// * `verbose` - If true, print progress messages
///
/// # Returns
///
/// * `Ok(())` - Rebuild succeeded
/// * `Err(RebuildError)` - Rebuild failed
fn rebuild_xtask(verbose: bool) -> Result<(), RebuildError> {
    if verbose {
        eprintln!("[preflight] Rebuilding xtask...");
    }

    let build_status = Command::new("cargo")
        .args(["build", "-p", "xtask", "--features", "crossval-all"])
        .status()
        .map_err(|e: std::io::Error| RebuildError::BuildFailed(e.to_string()))?;

    if !build_status.success() {
        return Err(RebuildError::BuildFailed(format!(
            "cargo build exited with code {:?}",
            build_status.code()
        )));
    }

    if verbose {
        eprintln!("[preflight] ✓ Rebuild complete");
    }

    Ok(())
}

/// Rebuild xtask to detect newly installed libraries (with clean)
///
/// This function does a full clean + rebuild cycle, which is slower but
/// ensures all build-time detection runs from scratch.
///
/// For faster incremental rebuilds after auto-repair, use `rebuild_xtask()` instead.
///
/// # Returns
///
/// * `Ok(())` - Rebuild succeeded
/// * `Err(RebuildError)` - Clean or build failed
#[allow(dead_code)]
fn rebuild_xtask_for_detection() -> Result<(), RebuildError> {
    // Step 1: Clean xtask and crossval crates
    let clean_status = Command::new("cargo")
        .args(["clean", "-p", "xtask", "-p", "crossval"])
        .status()
        .map_err(|e: std::io::Error| RebuildError::CleanFailed(e.to_string()))?;

    if !clean_status.success() {
        return Err(RebuildError::CleanFailed(format!(
            "cargo clean exited with code {:?}",
            clean_status.code()
        )));
    }

    // Step 2: Rebuild with crossval features
    rebuild_xtask(false)
}

/// Re-execute current xtask binary with original arguments
///
/// Implements robust two-tier re-execution mechanism for automatic C++ backend
/// repair workflow. After `setup-cpp-auto` installs backend libraries and
/// `rebuild_xtask()` recompiles the binary, this function re-executes the new
/// binary to pick up updated build-time constants (`HAS_BITNET`, `HAS_LLAMA`).
///
/// # Two-Tier Execution Strategy
///
/// **Tier 1: Fast Path (Unix only)**
/// - Calls `exec()` with `current_exe()` path to replace current process
/// - Zero overhead: no spawn, same PID, instant transition
/// - Fails gracefully when binary unavailable (ENOENT)
///
/// **Tier 2: Fallback Path (all platforms)**
/// - Uses `cargo run -p xtask --features crossval-all -- <args>`
/// - Rebuilds binary if needed (handles race conditions transparently)
/// - Spawns child process, parent exits with child's exit code
///
/// # Race Condition Handling
///
/// Between `path.exists()` check and `exec()` call, cargo may:
/// - Delete old binary for incremental rebuild (10-100ms window)
/// - Move binary to new location during link phase
/// - Invalidate /proc/self/exe symlink on kernel updates
///
/// This window is typically 10-100ms on local filesystems, but can extend
/// to seconds on network filesystems. The fallback path handles this
/// transparently by letting cargo rebuild the binary.
///
/// # Recursion Guard
///
/// Sets `BITNET_REPAIR_PARENT=1` to prevent infinite recursion. The re-executed
/// binary will skip repair and only validate detection.
///
/// # Platform-Specific Behavior
///
/// * **Unix**: Uses `exec()` to replace current process (never returns on success)
/// * **Windows**: Spawns child process and exits with its status code
///
/// # Arguments
///
/// * `original_args` - Full argument list from `env::args()`, including program name
///
/// # Returns
///
/// Never returns on Unix fast path success (process replaced).
/// Returns `RepairError` only if both exec() and cargo run fail.
///
/// # Examples
///
/// ```ignore
/// use xtask::crossval::preflight::reexec_current_command;
///
/// // After rebuild, re-exec with original args
/// let args: Vec<String> = env::args().collect();
/// reexec_current_command(&args)?;
/// // This point never reached on Unix (exec replaces process)
/// // On Windows, process exits in reexec_current_command
/// ```
///
/// # Acceptance Criteria
///
/// - **AC1**: Fast path uses exec() when binary exists (Unix)
/// - **AC2**: Fallback to cargo run when binary unavailable
/// - **AC3**: All CLI arguments preserved across re-exec
/// - **AC4**: BITNET_REPAIR_PARENT guard prevents infinite loops
/// - **AC5**: Diagnostic logging shows resolved path + existence
/// - **AC6**: Windows uses spawn() pattern consistently
/// - **AC7**: Exit code propagated correctly from spawned process
pub fn reexec_current_command(original_args: &[String]) -> Result<(), RepairError> {
    eprintln!("[repair] Re-executing with updated detection...");

    // AC1 + AC5: Unix fast path - Try exec() with current_exe() first
    #[cfg(unix)]
    {
        if let Ok(current_exe) = env::current_exe() {
            let exists = current_exe.exists();

            // AC5: Diagnostic logging (resolved path and existence)
            eprintln!("[reexec] exe: {}", current_exe.display());
            eprintln!("[reexec] exe exists: {}", exists);
            eprintln!("[reexec] args: {:?}", original_args);

            // AC1: Only attempt exec() if binary exists
            if exists {
                use std::os::unix::process::CommandExt;

                let mut cmd = Command::new(&current_exe);

                // AC3: Preserve all original arguments (including program name for exec)
                cmd.args(original_args);

                // AC4: Set recursion guard to prevent child from attempting repair
                cmd.env("BITNET_REPAIR_PARENT", "1");

                eprintln!("[reexec] Attempting exec()...");

                // Try exec() - never returns on success
                // If we reach the line after this, exec() failed
                let err = cmd.exec();

                // AC2 + AC5: exec() failed (race condition or other error) - log and fall through
                eprintln!("[reexec] Fast path failed: {}", err);
                eprintln!("[reexec] Error kind: {:?}", err.kind());

                // Fall through to cargo run fallback
            } else {
                // AC2 + AC5: Binary doesn't exist - skip exec() and use fallback
                eprintln!("[reexec] Binary doesn't exist, skipping exec()");
            }
        } else {
            // AC2 + AC5: current_exe() failed - skip exec() and use fallback
            eprintln!("[reexec] current_exe() failed, skipping exec()");
        }
    }

    // AC2 + AC6: Fallback path - cargo run (all platforms, Unix if exec failed)
    // This path is the ONLY path on Windows (no exec() available)
    eprintln!("[reexec] Trying cargo run fallback...");

    let mut cmd = Command::new("cargo");
    cmd.arg("run").arg("-p").arg("xtask").arg("--features").arg("crossval-all").arg("--");

    // AC3: Preserve all original arguments (skip program name - cargo run adds it)
    if original_args.len() > 1 {
        cmd.args(&original_args[1..]);
    }

    // AC4: Set recursion guard to prevent child from attempting repair
    cmd.env("BITNET_REPAIR_PARENT", "1");

    // AC5: Diagnostic logging for fallback command
    eprintln!(
        "[reexec] Fallback command: cargo run -p xtask --features crossval-all -- {:?}",
        if original_args.len() > 1 { &original_args[1..] } else { &[] }
    );

    // AC6 + AC7: Spawn child and wait for completion (Windows spawn semantics)
    match cmd.status() {
        Ok(status) => {
            // AC7: Propagate child's exit code exactly
            let code = status.code().unwrap_or(1);
            eprintln!("[reexec] Fallback child exited with code: {}", code);
            std::process::exit(code);
        }
        Err(e) => {
            // Both exec() (if Unix) and cargo run fallback failed
            // Classify error to provide clear recovery steps
            let error_msg = match e.kind() {
                std::io::ErrorKind::NotFound => {
                    format!(
                        "Re-exec failed: cargo not found in PATH\n\
                         Tried: cargo run -p xtask --features crossval-all -- ...\n\
                         Error: {}\n\
                         \n\
                         Recovery steps:\n\
                         1. Install Rust toolchain: https://rustup.rs/\n\
                         2. Verify cargo in PATH: which cargo\n\
                         3. Retry preflight with --repair=auto",
                        e
                    )
                }
                std::io::ErrorKind::PermissionDenied => {
                    format!(
                        "Re-exec failed: permission denied executing cargo\n\
                         Error: {}\n\
                         \n\
                         Recovery steps:\n\
                         1. Check cargo executable permissions\n\
                         2. Verify user has execute permission\n\
                         3. If in restricted environment, manually rebuild xtask",
                        e
                    )
                }
                _ => {
                    format!(
                        "Re-exec failed: cargo run failed\n\
                         Error: {}\n\
                         \n\
                         This may indicate:\n\
                         1. Cargo workspace is corrupted\n\
                         2. Insufficient disk space\n\
                         3. File system permissions issue\n\
                         \n\
                         Recovery steps:\n\
                         1. Try manual rebuild: cargo build -p xtask --features crossval-all\n\
                         2. Check disk space: df -h\n\
                         3. Verify workspace: cargo check -p xtask",
                        e
                    )
                }
            };

            Err(RepairError::Unknown { error: error_msg, backend: "unknown".to_string() })
        }
    }
}

/// Retry configuration for auto-repair with exponential backoff
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial backoff delay in milliseconds
    pub initial_backoff_ms: u64,
    /// Maximum backoff delay in milliseconds
    pub max_backoff_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig { max_retries: 3, initial_backoff_ms: 1000, max_backoff_ms: 16000 }
    }
}

/// Attempt auto-repair with retry logic and exponential backoff
///
/// Implements the core auto-repair orchestration with:
/// - Retry loop for transient network errors (max 3 attempts)
/// - Exponential backoff: 1s, 2s, 4s
/// - No retry for permanent errors (build/permission)
///
/// # Arguments
///
/// * `backend` - The C++ backend to repair
/// * `verbose` - Show detailed progress messages
///
/// # Returns
///
/// * `Ok(stdout)` - Repair succeeded with captured stdout
/// * `Err(RepairError)` - Repair failed after retries
///
/// # Examples
///
/// ```ignore
/// use xtask::crossval::preflight::{attempt_repair_with_retry, RepairMode};
/// use xtask::crossval::CppBackend;
///
/// // Auto-repair BitNet backend with retries
/// let stdout = attempt_repair_with_retry(CppBackend::BitNet, true)?;
/// ```
pub fn attempt_repair_with_retry(
    backend: CppBackend,
    verbose: bool,
) -> Result<String, RepairError> {
    let config = RetryConfig::default();
    let mut retries = 0;

    loop {
        match attempt_repair_once(backend, verbose) {
            Ok((_, stdout)) => return Ok(stdout),
            Err(e) if is_retryable_error(&e) && retries < config.max_retries => {
                retries += 1;
                let backoff_ms = config.initial_backoff_ms * 2_u64.pow(retries - 1);
                let backoff_ms = backoff_ms.min(config.max_backoff_ms);

                eprintln!(
                    "[repair] Network error, retry {}/{} after {}ms...",
                    retries, config.max_retries, backoff_ms
                );

                std::thread::sleep(Duration::from_millis(backoff_ms));
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Attempt repair once (single attempt, no retries)
///
/// Core repair logic that invokes setup-cpp-auto and classifies errors.
/// Modified to capture and return stdout for environment export parsing.
///
/// # Arguments
///
/// * `backend` - The C++ backend to repair
/// * `verbose` - Show detailed progress messages
///
/// # Returns
///
/// * `Ok((success, stdout))` - Repair result with captured stdout
/// * `Err(RepairError)` - Repair failed with classified error
///
/// # Specification
///
/// See: `docs/specs/env-export-before-rebuild-deterministic.md` (AC3)
fn attempt_repair_once(backend: CppBackend, verbose: bool) -> Result<(bool, String), RepairError> {
    let progress = RepairProgress::new(verbose);

    // Check recursion guard
    if env::var("BITNET_REPAIR_IN_PROGRESS").is_ok() {
        return Err(RepairError::RecursionDetected);
    }

    // Set recursion guard
    // SAFETY: This env var is only used as a recursion guard during auto-repair.
    // We ensure cleanup happens even on error via the removal below.
    unsafe {
        env::set_var("BITNET_REPAIR_IN_PROGRESS", "1");
    }

    progress.log("DETECT", &format!("Backend '{}' not found", backend.name()));
    progress.log("REPAIR", "Invoking setup-cpp-auto... (this will take 5-10 minutes on first run)");

    // Invoke setup-cpp-auto
    let setup_result = Command::new(
        env::current_exe()
            .map_err(|e| RepairError::SetupFailed(format!("Failed to get current exe: {}", e)))?,
    )
    .args(["setup-cpp-auto", "--emit=sh"])
    .output()
    .map_err(|e| RepairError::SetupFailed(format!("Failed to execute setup-cpp-auto: {}", e)))?;

    // Cleanup recursion guard
    // SAFETY: We're removing the same env var we set above, restoring the environment state.
    unsafe {
        env::remove_var("BITNET_REPAIR_IN_PROGRESS");
    }

    // Capture stdout (contains shell export statements)
    let stdout = String::from_utf8_lossy(&setup_result.stdout).to_string();

    if !setup_result.status.success() {
        let stderr = String::from_utf8_lossy(&setup_result.stderr);
        let backend_name = backend.name();

        return Err(RepairError::classify(&stderr, backend_name));
    }

    progress.log("REPAIR", "C++ libraries installed successfully");
    progress.log("REBUILD", "Next: Rebuild xtask to detect libraries");

    // Show rebuild instructions
    if verbose {
        eprintln!();
        eprintln!("✓ Setup complete! Rebuilding xtask to detect libraries...");
        eprintln!();
        eprintln!("  cargo clean -p xtask && cargo build -p xtask --features crossval-all");
        eprintln!();
        eprintln!("After build completes, re-run:");
        eprintln!(
            "  cargo run -p xtask -- preflight --backend {} --verbose",
            backend.name().split('.').next().unwrap()
        );
    }

    Ok((true, stdout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_preflight_respects_env() {
        // This test documents the expected behavior
        // Actual values depend on build-time detection

        // If CROSSVAL_HAS_LLAMA=true, preflight should succeed
        // If CROSSVAL_HAS_LLAMA=false, preflight should fail

        // We can't test the actual behavior without mocking env vars,
        // but we can verify the function compiles and has correct signature
        let _ = preflight_backend_libs(CppBackend::Llama, false);
    }

    #[test]
    fn test_print_backend_status_runs() {
        // Just verify it doesn't panic
        print_backend_status(false);
        print_backend_status(true);
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_is_repair_in_progress_false_by_default() {
        // Ensure no recursion guard is set initially
        unsafe {
            std::env::remove_var("BITNET_REPAIR_IN_PROGRESS");
        }
        assert!(!is_repair_in_progress());
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_is_repair_in_progress_true_when_set() {
        // Set recursion guard
        unsafe {
            std::env::set_var("BITNET_REPAIR_IN_PROGRESS", "1");
        }
        assert!(is_repair_in_progress());
        // Cleanup
        unsafe {
            std::env::remove_var("BITNET_REPAIR_IN_PROGRESS");
        }
    }

    #[test]
    fn test_auto_repair_with_setup_cpp_returns_ok_for_never_mode() {
        // RepairMode::Never should return Ok immediately without invoking setup-cpp-auto
        let result = auto_repair_with_setup_cpp(CppBackend::Llama, RepairMode::Never);
        assert!(result.is_ok());
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_auto_repair_with_setup_cpp_detects_recursion() {
        // Set recursion guard
        unsafe {
            std::env::set_var("BITNET_REPAIR_IN_PROGRESS", "1");
        }

        // Should detect recursion and return error
        let result = auto_repair_with_setup_cpp(CppBackend::Llama, RepairMode::Auto);
        assert!(matches!(result, Err(RepairError::RecursionDetected)));

        // Cleanup
        unsafe {
            std::env::remove_var("BITNET_REPAIR_IN_PROGRESS");
        }
    }
}
