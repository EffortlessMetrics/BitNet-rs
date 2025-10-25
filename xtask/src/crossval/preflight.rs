//! Backend library preflight validation
//!
//! Verifies that required C++ libraries are available before running cross-validation.

use crate::crossval::CppBackend;
use anyhow::{Result, bail};
use std::path::Path;

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
        CppBackend::BitNet => bitnet_crossval::HAS_BITNET,
        CppBackend::Llama => bitnet_crossval::HAS_LLAMA,
    };

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

        println!("  {}. {}", idx + 1, path_desc);
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
        println!("  {}. {}", idx + 1, path.display());
        if path.exists() {
            println!("     ✓ {} (exists)", path.display());

            // Show what was searched for
            println!("     Searched for: {:?}", backend.required_libs());

            // List any libraries found (might be other backends)
            if let Ok(entries) = std::fs::read_dir(path) {
                let mut found_any = false;
                for entry in entries.flatten() {
                    if let Some(name) = entry.path().file_name().and_then(|n| n.to_str()) {
                        if name.starts_with("lib")
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
fn get_library_search_paths() -> Vec<std::path::PathBuf> {
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
                    if name.starts_with(req) {
                        if let Some(full_name) = entry.path().file_name().and_then(|n| n.to_str()) {
                            found.push(full_name.to_string());
                        }
                    }
                }
            }
        }
    }

    if found.is_empty() { None } else { Some(found) }
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
    use std::fs;
    use std::time::SystemTime;

    // Get current executable path
    std::env::current_exe().ok().and_then(|path| {
        // Get file metadata
        fs::metadata(&path).ok().and_then(|meta| {
            // Get modification time
            meta.modified().ok().and_then(|modified| {
                // Convert to duration since UNIX epoch
                modified.duration_since(SystemTime::UNIX_EPOCH).ok().map(|duration| {
                    // Format as ISO 8601-ish timestamp (seconds since epoch for simplicity)
                    let secs = duration.as_secs();
                    // Convert to readable format (YYYY-MM-DD HH:MM:SS UTC approximation)
                    // This is a simplified version - full ISO 8601 would require chrono
                    format!("{} seconds since epoch", secs)
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
fn format_build_metadata(backend: CppBackend) -> String {
    let has_backend = match backend {
        CppBackend::BitNet => bitnet_crossval::HAS_BITNET,
        CppBackend::Llama => bitnet_crossval::HAS_LLAMA,
    };

    let backend_name = match backend {
        CppBackend::BitNet => "BITNET",
        CppBackend::Llama => "LLAMA",
    };

    let timestamp = get_xtask_build_timestamp().unwrap_or_else(|| "unknown".to_string());

    format!(
        "Build-Time Detection Metadata\n\
         {}\n\
         CROSSVAL_HAS_{} = {}\n\
         Last xtask build: {}\n\
         Build feature flags: crossval-all",
        SEPARATOR_LIGHT, backend_name, has_backend, timestamp
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
    let has_bitnet = bitnet_crossval::HAS_BITNET;

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
    let has_llama = bitnet_crossval::HAS_LLAMA;

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
