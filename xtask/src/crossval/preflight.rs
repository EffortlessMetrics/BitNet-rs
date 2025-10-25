//! Backend library preflight validation
//!
//! Verifies that required C++ libraries are available before running cross-validation.

use crate::crossval::CppBackend;
use anyhow::{Result, bail};
use std::path::Path;

/// Verify required libraries are available for selected backend
///
/// Checks compile-time detection from build.rs environment variables:
/// - CROSSVAL_HAS_BITNET: Set to "true" if libbitnet* found during build
/// - CROSSVAL_HAS_LLAMA: Set to "true" if libllama*/libggml* found during build
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
    // Check compile-time detection from build.rs
    let has_libs = match backend {
        CppBackend::BitNet => {
            // Emitted by crossval/build.rs as CROSSVAL_HAS_BITNET=true/false
            option_env!("CROSSVAL_HAS_BITNET").map(|v| v == "true").unwrap_or(false)
        }
        CppBackend::Llama => {
            // Emitted by crossval/build.rs as CROSSVAL_HAS_LLAMA=true/false
            option_env!("CROSSVAL_HAS_LLAMA").map(|v| v == "true").unwrap_or(false)
        }
    };

    if !has_libs {
        if verbose {
            print_verbose_failure_diagnostics(backend);
        }

        // Libraries not found - provide actionable error message
        bail!(
            "Backend '{}' selected but required libraries not found.\n\
             \n\
             Setup instructions:\n\
             1. Install C++ reference implementation:\n\
                {}\n\
             \n\
             2. Verify libraries are loaded:\n\
                cargo run -p xtask -- preflight --backend {}\n\
             \n\
             3. Rebuild xtask to detect libraries:\n\
                cargo clean -p xtask && cargo build -p xtask --features crossval-all\n\
             \n\
             Required libraries: {:?}\n\
             \n\
             Note: Library detection happens at BUILD time. If you just installed\n\
             the C++ reference, you must rebuild xtask for detection to work.",
            backend.name(),
            backend.setup_command(),
            backend.name(),
            backend.required_libs()
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
    println!("✓ Backend '{}' libraries: AVAILABLE", backend.name());
    println!();

    // Environment variables checked
    println!("Environment Variables:");
    print_env_var_status("BITNET_CPP_DIR");
    print_env_var_status("BITNET_CPP_PATH");
    print_env_var_status("BITNET_CROSSVAL_LIBDIR");

    #[cfg(target_os = "linux")]
    print_env_var_status("LD_LIBRARY_PATH");

    #[cfg(target_os = "macos")]
    print_env_var_status("DYLD_LIBRARY_PATH");

    #[cfg(target_os = "windows")]
    print_env_var_status("PATH");

    println!();

    // Library search paths (from build.rs logic)
    println!("Library Search Paths:");
    let search_paths = get_library_search_paths();
    for path in &search_paths {
        if path.exists() {
            println!("  ✓ {} (exists)", path.display());

            // List libraries found in this path
            if let Some(libs) = find_libs_in_path(path, backend) {
                for lib in libs {
                    println!("    - {}", lib);
                }
            }
        } else {
            println!("  ✗ {} (not found)", path.display());
        }
    }

    println!();

    // Required libraries
    println!("Required Libraries: {:?}", backend.required_libs());
    println!("Status: All required libraries detected at build time");

    println!();

    // Build flags that would be set
    println!("Build Configuration:");
    match backend {
        CppBackend::BitNet => {
            println!("  CROSSVAL_HAS_BITNET=true");
            println!("  Linked: libbitnet (dynamic)");
        }
        CppBackend::Llama => {
            println!("  CROSSVAL_HAS_LLAMA=true");
            println!("  Linked: libllama, libggml (dynamic)");
        }
    }

    #[cfg(target_os = "linux")]
    println!("  Standard library: libstdc++ (dynamic)");

    #[cfg(target_os = "macos")]
    println!("  Standard library: libc++ (dynamic)");
}

/// Print verbose diagnostics when libraries are NOT found
fn print_verbose_failure_diagnostics(backend: CppBackend) {
    println!("❌ Backend '{}' libraries: NOT AVAILABLE", backend.name());
    println!();

    // Environment variables checked
    println!("Environment Variables:");
    print_env_var_status("BITNET_CPP_DIR");
    print_env_var_status("BITNET_CPP_PATH");
    print_env_var_status("BITNET_CROSSVAL_LIBDIR");

    #[cfg(target_os = "linux")]
    print_env_var_status("LD_LIBRARY_PATH");

    #[cfg(target_os = "macos")]
    print_env_var_status("DYLD_LIBRARY_PATH");

    #[cfg(target_os = "windows")]
    print_env_var_status("PATH");

    println!();

    // Library search paths
    println!("Library Search Paths:");
    let search_paths = get_library_search_paths();

    if search_paths.is_empty() {
        println!("  (no paths configured - set BITNET_CPP_DIR or BITNET_CROSSVAL_LIBDIR)");
    } else {
        for path in &search_paths {
            if path.exists() {
                println!("  ✓ {} (exists)", path.display());

                // Show what was searched for
                println!("    Searched for: {:?}", backend.required_libs());

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
                                    println!("    Found (other libraries):");
                                    found_any = true;
                                }
                                println!("      - {}", name);
                            }
                        }
                    }
                    if !found_any {
                        println!("    No libraries found in this directory");
                    }
                }
            } else {
                println!("  ✗ {} (not found)", path.display());
            }
        }
    }

    println!();

    // Required libraries
    println!("Searched for:");
    for lib in backend.required_libs() {
        println!("  - {}.so / {}.dylib / {}.a", lib, lib, lib);
    }

    println!();

    // Next steps
    println!("Next Steps:");
    println!("  1. Run setup command:");
    println!("     {}", backend.setup_command());
    println!();
    println!("  2. Or manually install and set environment:");
    match backend {
        CppBackend::BitNet => {
            println!("     export BITNET_CPP_DIR=/path/to/bitnet.cpp");
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
    println!("  3. Rebuild xtask to detect libraries:");
    println!("     cargo clean -p xtask && cargo build -p xtask --features crossval-all");
    println!();
    println!("  4. Re-run preflight:");
    println!(
        "     cargo run -p xtask --features crossval-all -- preflight --backend {} --verbose",
        backend.name().split('.').next().unwrap()
    );
}

/// Print environment variable status
fn print_env_var_status(var_name: &str) {
    match std::env::var(var_name) {
        Ok(value) => {
            // Truncate very long values
            if value.len() > 80 {
                println!("  {} = {}...", var_name, &value[..77]);
            } else {
                println!("  {} = {}", var_name, value);
            }
        }
        Err(_) => println!("  {} = (not set)", var_name),
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

    // Check BitNet
    let has_bitnet = option_env!("CROSSVAL_HAS_BITNET").map(|v| v == "true").unwrap_or(false);

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

    // Check LLaMA
    let has_llama = option_env!("CROSSVAL_HAS_LLAMA").map(|v| v == "true").unwrap_or(false);

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
