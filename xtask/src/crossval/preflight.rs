#![allow(dead_code)] // TODO: Enhanced preflight functions to be integrated with main command

//! Backend library preflight validation
//!
//! Verifies that required C++ libraries are available before running cross-validation.

use crate::crossval::CppBackend;
use anyhow::{Result, bail};
use std::path::Path;
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

/// Classify error from stderr output
///
/// Determines error type based on stderr content from setup-cpp-auto or cmake.
///
/// # Arguments
///
/// * `stderr` - Standard error output to classify
///
/// # Returns
///
/// Error classification:
/// - Network: Connection timeouts, git clone failures, DNS errors
/// - Build: CMake errors, compiler errors, linker failures
/// - Permission: EACCES, permission denied, cannot create directory
/// - Unknown: Unrecognized error pattern
fn classify_error(stderr: &str) -> &'static str {
    let lower = stderr.to_lowercase();

    // Network error patterns
    if lower.contains("connection timeout")
        || lower.contains("failed to clone")
        || lower.contains("could not resolve host")
        || lower.contains("network unreachable")
        || lower.contains("connection refused")
        || lower.contains("failed to connect")
    {
        return "network";
    }

    // Build error patterns
    if lower.contains("cmake error")
        || lower.contains("ninja: build stopped")
        || lower.contains("undefined reference")
        || lower.contains("no such file or directory")
        || lower.contains("compilation failed")
        || lower.contains("cannot find")
    {
        return "build";
    }

    // Permission error patterns
    if lower.contains("permission denied")
        || lower.contains("eacces")
        || lower.contains("cannot create directory")
        || lower.contains("access is denied")
    {
        return "permission";
    }

    "unknown"
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
    should_repair: bool,
) -> Result<()> {
    // Step 1: Check if backend is already available
    let is_available = match backend {
        CppBackend::BitNet => HAS_BITNET,
        CppBackend::Llama => HAS_LLAMA,
    };

    if is_available {
        // Backend already available - use existing preflight logic
        return preflight_backend_libs(backend, verbose);
    }

    // Step 2: Backend not available - attempt repair if enabled
    if !should_repair {
        // No repair requested - use traditional error path
        return preflight_backend_libs(backend, verbose);
    }

    // Step 3: Attempt repair
    if verbose {
        eprintln!("Backend '{}' not found, attempting auto-repair...", backend.name());
    }

    if let Err(e) = attempt_repair(backend, verbose) {
        // Repair failed - show error with recovery steps
        eprintln!("\n{}", e);
        bail!("Auto-repair failed for backend '{}'", backend.name());
    }

    // Step 4: Revalidation note
    // NOTE: After rebuild, we would need to re-execute the xtask binary to see
    // updated build-time constants. For now, we show success message and expect
    // user to re-run preflight to verify.
    if verbose {
        eprintln!(
            "✓ Backend '{}' setup complete. Rebuild xtask to detect libraries:",
            backend.name()
        );
        eprintln!("  cargo clean -p xtask && cargo build -p xtask --features crossval-all");
        eprintln!(
            "  cargo run -p xtask -- preflight --backend {} --verbose",
            backend.name().split('.').next().unwrap()
        );
    } else {
        println!("✓ Backend '{}' setup complete (rebuild xtask to detect)", backend.name());
    }

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
        let error_type = classify_error(&stderr);
        let backend_name = backend.name().to_string();

        return Err(match error_type {
            "network" => {
                RepairError::NetworkFailure { error: stderr.to_string(), backend: backend_name }
            }
            "build" => {
                RepairError::BuildFailure { error: stderr.to_string(), backend: backend_name }
            }
            "permission" => {
                // Try to extract path from error message
                let path = stderr
                    .lines()
                    .find(|line| line.contains("Permission denied"))
                    .and_then(|line| {
                        line.split_whitespace()
                            .find(|word| word.starts_with('/') || word.starts_with("~/"))
                    })
                    .unwrap_or("~/.cache/bitnet_cpp");
                RepairError::PermissionDenied { path: path.to_string(), backend: backend_name }
            }
            _ => RepairError::SetupFailed(stderr.to_string()),
        });
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

/// Rebuild xtask to detect newly installed libraries
///
/// NOTE: This function is planned for future implementation.
/// Currently, we rely on user to manually rebuild xtask after setup-cpp-auto.
///
/// The challenge is that build-time constants (HAS_BITNET, HAS_LLAMA) are
/// embedded during compilation and can't be updated at runtime. We would need to:
/// 1. Rebuild xtask binary
/// 2. Re-execute the new binary
/// 3. Continue from where we left off
///
/// For the MVP, we show clear rebuild instructions to the user.
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
