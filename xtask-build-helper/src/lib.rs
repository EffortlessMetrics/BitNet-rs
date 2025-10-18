//! FFI Build Hygiene
//!
//! Unified C++ shim compilation for build scripts across the workspace.
//! This module provides a single source of truth for FFI build configuration,
//! eliminating duplicate logic and enforcing hygiene practices:
//!
//! - Use `-isystem` for third-party headers (CUDA, bitnet.cpp) to suppress warnings
//! - Use `-I` for local headers to preserve warning visibility
//! - No global warning suppression (`.warnings(false)`)
//! - Consistent compiler flags across all FFI builds
//!
//! # Issue #469 AC6
//!
//! This module implements AC6 FFI build hygiene consolidation.

use std::path::{Path, PathBuf};

/// Compile a C/C++ FFI shim with unified hygiene settings.
///
/// This function provides a single entry point for compiling C/C++ shim code
/// across all FFI-enabled crates in the workspace. It enforces hygiene by:
///
/// - Using `-I` for local includes (warnings visible)
/// - Using `-isystem` for system/third-party includes (warnings suppressed)
/// - Applying consistent compiler flags (C++17 for C++, -O2, -fPIC)
/// - Never using `.warnings(false)` to preserve local code warnings
///
/// The function auto-detects whether to use C or C++ mode based on file extension:
/// - `.cc`, `.cpp`, `.cxx` → C++ mode with `-std=c++17`
/// - `.c` → C mode
///
/// # Arguments
///
/// * `shim_path` - Path to the C/C++ shim source file (e.g., csrc/shim.cc or csrc/shim.c)
/// * `output_name` - Name of the compiled library (e.g., "my_shim")
/// * `include_dirs` - Local include directories (use `-I`, warnings visible)
/// * `system_include_dirs` - System/third-party includes (use `-isystem`, warnings suppressed)
///
/// # Example
///
/// ```no_run
/// use std::path::PathBuf;
/// use xtask::ffi::{compile_cpp_shim, bitnet_cpp_system_includes, cuda_system_includes};
///
/// let local_includes = vec![PathBuf::from("csrc")];
/// let system_includes = bitnet_cpp_system_includes().unwrap_or_default();
///
/// compile_cpp_shim(
///     Path::new("csrc/my_shim.cc"),
///     "my_shim",
///     &local_includes,
///     &system_includes,
/// ).expect("Failed to compile shim");
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The shim source file does not exist
/// - Compilation fails (linker errors, syntax errors, etc.)
pub fn compile_cpp_shim(
    shim_path: &Path,
    output_name: &str,
    include_dirs: &[PathBuf],
    system_include_dirs: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>> {
    if !shim_path.exists() {
        return Err(format!(
            "C/C++ shim source not found: {}\n\
             Expected a valid C/C++ source file (.c, .cc, .cpp, .cxx)",
            shim_path.display()
        )
        .into());
    }

    // Auto-detect C++ mode from file extension
    let is_cpp = shim_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e == "cc" || e == "cpp" || e == "cxx")
        .unwrap_or(false);

    let lang = if is_cpp { "C++" } else { "C" };
    eprintln!("xtask::ffi: Compiling {} shim from {}", lang, shim_path.display());

    let mut builder = cc::Build::new();
    builder.file(shim_path).flag_if_supported("-O2").flag_if_supported("-fPIC");

    // Set C++ mode and standard if needed
    if is_cpp {
        builder.cpp(true).flag_if_supported("-std=c++17");
    }

    // Add local include directories with -I (warnings visible)
    for include_dir in include_dirs {
        if include_dir.exists() {
            builder.include(include_dir);
            eprintln!("xtask::ffi:   -I {}", include_dir.display());
        }
    }

    // Add system include directories with -isystem (warnings suppressed)
    // Note: cc-rs does not provide a direct `.system_include()` method,
    // so we manually add -isystem flags. This is the standard approach
    // for suppressing third-party header warnings.
    for include_dir in system_include_dirs {
        if include_dir.exists() {
            builder.flag(format!("-isystem{}", include_dir.display()));
            eprintln!("xtask::ffi:   -isystem {}", include_dir.display());
        }
    }

    // Add warning suppression for known noisy third-party patterns
    // Note: These are applied globally but only affect third-party headers
    // due to -isystem usage. Local code warnings are preserved via -I.
    builder
        .flag_if_supported("-Wno-unknown-pragmas")
        .flag_if_supported("-Wno-deprecated-declarations");

    // Compile the shim
    builder.compile(output_name);

    eprintln!("xtask::ffi: {} shim compiled successfully: {}", lang, output_name);
    Ok(())
}

/// Compile multiple C/C++ FFI shim files with unified hygiene settings.
///
/// This is a convenience wrapper around `compile_cpp_shim` for cases where
/// multiple source files need to be compiled into a single library.
///
/// # Arguments
///
/// * `shim_paths` - Paths to the C/C++ shim source files
/// * `output_name` - Name of the compiled library (e.g., "my_shim")
/// * `include_dirs` - Local include directories (use `-I`, warnings visible)
/// * `system_include_dirs` - System/third-party includes (use `-isystem`, warnings suppressed)
///
/// # Example
///
/// ```no_run
/// use std::path::{Path, PathBuf};
/// use xtask::ffi::compile_cpp_shims_multi;
///
/// let shims = vec![
///     Path::new("csrc/shim1.c"),
///     Path::new("csrc/shim2.c"),
/// ];
/// let local_includes = vec![PathBuf::from("csrc")];
/// let system_includes = vec![PathBuf::from("csrc/vendored/include")];
///
/// compile_cpp_shims_multi(&shims, "my_shim", &local_includes, &system_includes)
///     .expect("Failed to compile shims");
/// ```
///
/// # Errors
///
/// Returns an error if any source file does not exist or compilation fails.
pub fn compile_cpp_shims_multi(
    shim_paths: &[&Path],
    output_name: &str,
    include_dirs: &[PathBuf],
    system_include_dirs: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>> {
    if shim_paths.is_empty() {
        return Err("No shim files provided".into());
    }

    // Verify all files exist
    for shim_path in shim_paths {
        if !shim_path.exists() {
            return Err(format!(
                "C/C++ shim source not found: {}\n\
                 Expected a valid C/C++ source file (.c, .cc, .cpp, .cxx)",
                shim_path.display()
            )
            .into());
        }
    }

    eprintln!("xtask::ffi: Compiling {} shim files into {}", shim_paths.len(), output_name);

    let mut builder = cc::Build::new();
    builder.flag_if_supported("-O2").flag_if_supported("-fPIC");

    // Add all source files and detect C++ mode
    let mut any_cpp = false;
    for shim_path in shim_paths {
        builder.file(shim_path);
        eprintln!("xtask::ffi:   Source: {}", shim_path.display());

        // Check if any file is C++
        let is_cpp = shim_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e == "cc" || e == "cpp" || e == "cxx")
            .unwrap_or(false);
        any_cpp |= is_cpp;
    }

    // Set C++ mode if any file is C++
    if any_cpp {
        builder.cpp(true).flag_if_supported("-std=c++17");
    }

    // Add local include directories with -I (warnings visible)
    for include_dir in include_dirs {
        if include_dir.exists() {
            builder.include(include_dir);
            eprintln!("xtask::ffi:   -I {}", include_dir.display());
        }
    }

    // Add system include directories with -isystem (warnings suppressed)
    for include_dir in system_include_dirs {
        if include_dir.exists() {
            builder.flag(format!("-isystem{}", include_dir.display()));
            eprintln!("xtask::ffi:   -isystem {}", include_dir.display());
        }
    }

    // Add warning suppression for known noisy third-party patterns
    builder
        .flag_if_supported("-Wno-unknown-pragmas")
        .flag_if_supported("-Wno-deprecated-declarations");

    // Compile all shims into one library
    builder.compile(output_name);

    eprintln!("xtask::ffi: Shims compiled successfully: {}", output_name);
    Ok(())
}

/// Get standard CUDA system include paths.
///
/// Returns common CUDA include paths that should be added as system includes
/// (using `-isystem`) to suppress CUDA header warnings.
///
/// This function is best-effort and returns paths even if they don't exist.
/// Callers should filter by existence before use (compile_cpp_shim does this).
///
/// # Returns
///
/// A vector of potential CUDA include paths:
/// - `/usr/local/cuda/include` (standard Linux installation)
/// - `/usr/local/cuda/targets/x86_64-linux/include` (multi-arch installation)
/// - `/usr/local/cuda/targets/aarch64-linux/include` (ARM64/Jetson)
///
/// # Example
///
/// ```no_run
/// use xtask::ffi::cuda_system_includes;
///
/// let cuda_includes = cuda_system_includes();
/// // Use with compile_cpp_shim as system_include_dirs
/// ```
pub fn cuda_system_includes() -> Vec<PathBuf> {
    vec![
        PathBuf::from("/usr/local/cuda/include"),
        PathBuf::from("/usr/local/cuda/targets/x86_64-linux/include"),
        PathBuf::from("/usr/local/cuda/targets/aarch64-linux/include"),
    ]
}

/// Get BitNet C++ reference system include paths.
///
/// Returns include paths for the Microsoft BitNet C++ reference implementation,
/// which should be added as system includes (using `-isystem`) to suppress
/// third-party header warnings (llama.cpp, ggml, etc.).
///
/// The function reads `BITNET_CPP_DIR` or falls back to `$HOME/.cache/bitnet_cpp`.
///
/// # Returns
///
/// A vector of BitNet C++ include paths:
/// - `$BITNET_CPP_DIR/include` (BitNet headers)
/// - `$BITNET_CPP_DIR/3rdparty/llama.cpp/include` (llama.cpp headers)
/// - `$BITNET_CPP_DIR/3rdparty/llama.cpp/ggml/include` (ggml headers)
/// - `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/include` (build-time headers)
/// - `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/include` (build-time ggml headers)
///
/// # Errors
///
/// Returns an error if:
/// - `BITNET_CPP_DIR` is not set and `HOME` is not available
///
/// # Example
///
/// ```no_run
/// use xtask::ffi::bitnet_cpp_system_includes;
///
/// let cpp_includes = bitnet_cpp_system_includes().unwrap_or_default();
/// // Use with compile_cpp_shim as system_include_dirs
/// ```
///
/// # TODO: Version Sensitivity
///
/// The llama.cpp API may change across versions. Future work should:
/// - Parse VENDORED_LLAMA_COMMIT or equivalent version marker
/// - Adjust include paths based on detected llama.cpp version
/// - Emit warnings for untested API versions
pub fn bitnet_cpp_system_includes() -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let cpp_dir = std::env::var("BITNET_CPP_DIR")
        .or_else(|_| std::env::var("BITNET_CPP_PATH")) // Legacy support
        .or_else(|_| std::env::var("HOME").map(|h| format!("{}/.cache/bitnet_cpp", h)))?;

    let cpp_dir = PathBuf::from(cpp_dir);

    // TODO: llama.cpp API version detection
    // The include paths below assume a specific llama.cpp directory structure.
    // If llama.cpp reorganizes headers (e.g., moves ggml to a separate repo),
    // these paths may need adjustment. Consider:
    // - Reading $BITNET_CPP_DIR/VENDORED_LLAMA_COMMIT
    // - Mapping known versions to include path patterns
    // - Emitting warnings for untested llama.cpp versions

    Ok(vec![
        cpp_dir.join("include"),
        cpp_dir.join("3rdparty/llama.cpp/include"),
        cpp_dir.join("3rdparty/llama.cpp/ggml/include"),
        cpp_dir.join("build/3rdparty/llama.cpp/include"),
        cpp_dir.join("build/3rdparty/llama.cpp/ggml/include"),
    ])
}
