//! Backend library preflight validation
//!
//! Verifies that required C++ libraries are available before running cross-validation.

use crate::crossval::CppBackend;
use anyhow::{Result, bail};

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
        println!("✓ Backend '{}' libraries found", backend.name());
        println!("  Required: {:?}", backend.required_libs());
    }

    Ok(())
}

/// Print backend availability status for diagnostics
///
/// This is a convenience function for the `xtask preflight` command.
#[allow(dead_code)] // Reserved for future preflight command
pub fn print_backend_status() {
    println!("Backend Library Status:");
    println!();

    // Check BitNet
    let has_bitnet = option_env!("CROSSVAL_HAS_BITNET").map(|v| v == "true").unwrap_or(false);

    if has_bitnet {
        println!("  ✓ bitnet.cpp: AVAILABLE");
        println!("    Libraries: libbitnet*");
    } else {
        println!("  ✗ bitnet.cpp: NOT AVAILABLE");
        println!("    Setup: {}", CppBackend::BitNet.setup_command());
    }

    println!();

    // Check LLaMA
    let has_llama = option_env!("CROSSVAL_HAS_LLAMA").map(|v| v == "true").unwrap_or(false);

    if has_llama {
        println!("  ✓ llama.cpp: AVAILABLE");
        println!("    Libraries: libllama*, libggml*");
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
        print_backend_status();
    }
}
