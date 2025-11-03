// ============================================================================
// Backend State Enum - Three-state library detection
// ============================================================================
// Specification: docs/specs/bitnet-buildrs-detection-enhancement.md Section 3.2
//
// Distinguishes three library availability states:
// - FullBitNet: BitNet.cpp libraries found (llama optional)
// - LlamaFallback: Only llama.cpp libraries found, BitNet missing
// - Unavailable: No libraries found (STUB mode)
//
// Critical: Fixes Gap 1 where line 145 conflated "found_bitnet || found_llama"
// as "BITNET_AVAILABLE", misleading users when only llama.cpp is available.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum BackendState {
    FullBitNet,
    LlamaFallback,
    Unavailable,
}

impl BackendState {
    #[allow(dead_code)]
    fn as_str(&self) -> &str {
        match self {
            BackendState::FullBitNet => "full",
            BackendState::LlamaFallback => "llama",
            BackendState::Unavailable => "none",
        }
    }

    #[allow(dead_code)]
    fn is_available(&self) -> bool {
        !matches!(self, BackendState::Unavailable)
    }
}

/// Determine backend availability state based on library detection
///
/// Three-state logic (Specification Section 2.1):
/// - FullBitNet: BitNet.cpp libraries found (llama optional)
/// - LlamaFallback: Only llama.cpp libraries found, BitNet missing
/// - Unavailable: No libraries found
#[allow(dead_code)]
fn determine_backend_state(_found_bitnet: bool, _found_llama: bool) -> BackendState {
    match (_found_bitnet, _found_llama) {
        (true, _) => BackendState::FullBitNet, // BitNet found (llama irrelevant)
        (false, true) => BackendState::LlamaFallback, // Only llama found
        (false, false) => BackendState::Unavailable, // Nothing found
    }
}

/// Build three-tier search path hierarchy for library detection
///
/// Specification Section 2.2 - Three-Tier Search Path Priority
///
/// Returns: (primary_paths, embedded_paths, fallback_paths)
/// - Tier 1 (PRIMARY): BitNet.cpp-specific locations (3 paths)
/// - Tier 2 (EMBEDDED): Embedded llama.cpp locations (2 paths)
/// - Tier 3 (FALLBACK): Generic fallback locations (2 paths)
///
/// Critical: Adds missing path "build/3rdparty/llama.cpp/build/bin" (Gap 2 fix)
#[allow(dead_code)]
fn build_search_path_tiers(
    _bitnet_root: &str,
) -> (Vec<std::path::PathBuf>, Vec<std::path::PathBuf>, Vec<std::path::PathBuf>) {
    use std::path::Path;

    let root = Path::new(_bitnet_root);

    // Tier 1: PRIMARY BitNet.cpp locations (checked first)
    let primary_paths = vec![
        root.join("build/3rdparty/llama.cpp/build/bin"), // NEW: Embedded llama.cpp CMake output (Gap 2 fix)
        root.join("build/lib"),                          // Top-level CMake lib output
        root.join("build/bin"),                          // Top-level CMake bin output
    ];

    // Tier 2: EMBEDDED llama.cpp locations
    let embedded_paths = vec![
        root.join("build/3rdparty/llama.cpp/src"), // Llama library source output
        root.join("build/3rdparty/llama.cpp/ggml/src"), // GGML library source output
    ];

    // Tier 3: FALLBACK locations (last resort)
    let fallback_paths = vec![
        root.join("build"), // Top-level build root
        root.join("lib"),   // Install prefix lib directory
    ];

    (primary_paths, embedded_paths, fallback_paths)
}

/// Format RPATH string with colon-separated paths
///
/// Specification Section 3.2 - RPATH Format
///
/// Takes library directories and formats them for RPATH emission.
/// Priority order: primary → embedded → fallback (runtime loader searches in order)
#[allow(dead_code)]
fn format_rpath(_library_dirs: &[std::path::PathBuf]) -> String {
    _library_dirs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(":")
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(have_cpp)");
    println!("cargo:rustc-check-cfg=cfg(have_bitnet_full)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/bitnet_cpp_wrapper.c");
    println!("cargo:rerun-if-changed=src/bitnet_cpp_wrapper.cc");

    // Export environment metadata for parity receipts
    // Get rustc version
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    if let Ok(output) = std::process::Command::new(&rustc).arg("--version").output() {
        if let Ok(version) = String::from_utf8(output.stdout) {
            println!("cargo:rustc-env=RUSTC_VERSION={}", version.trim());
        }
    } else {
        println!("cargo:rustc-env=RUSTC_VERSION=unknown");
    }

    // Get target triple
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=TARGET={}", target);

    // Compile FFI wrapper if either llama-ffi or bitnet-ffi is enabled
    #[cfg(any(feature = "llama-ffi", feature = "bitnet-ffi"))]
    compile_ffi();
}

#[cfg(any(feature = "llama-ffi", feature = "bitnet-ffi"))]
fn compile_ffi() {
    use std::{env, path::Path};

    // Check if BITNET_CPP_DIR is set and do preliminary availability check
    let bitnet_cpp_dir_set = env::var("BITNET_CPP_DIR").is_ok();

    // Get the bitnet root directory for header checks
    // Support deprecated BITNET_CPP_PATH for backward compatibility
    let bitnet_root =
        env::var("BITNET_CPP_DIR").or_else(|_| {
            if let Ok(path) = env::var("BITNET_CPP_PATH") {
                println!(
                    "cargo:warning=crossval: BITNET_CPP_PATH is deprecated. Use BITNET_CPP_DIR instead."
                );
                Ok(path)
            } else {
                Err(std::env::VarError::NotPresent)
            }
        }).unwrap_or_else(|_| {
            format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap_or_else(|_| ".".into()))
        });

    // Check for essential header files to verify BitNet.cpp installation
    let has_headers = Path::new(&bitnet_root).join("include").exists()
        || Path::new(&bitnet_root).join("src").exists();

    // Preliminary availability check (will be refined after library search)
    let preliminary_available = bitnet_cpp_dir_set && has_headers;

    // Fallback: Compile legacy C wrapper if it exists (for backward compatibility)
    let c_wrapper_path = Path::new("src/bitnet_cpp_wrapper.c");
    if c_wrapper_path.exists() {
        cc::Build::new().file(c_wrapper_path).compile("bitnet_cpp_wrapper_c");
        println!("cargo:rustc-link-lib=static=bitnet_cpp_wrapper_c");
    }

    // Build multi-tier library search paths (Specification Section 2.2)
    let mut possible_lib_dirs = Vec::new();

    // Priority 0: Explicit BITNET_CROSSVAL_LIBDIR override (highest priority)
    // Environment variable: BITNET_CROSSVAL_LIBDIR
    // Purpose: Explicit library directory override for crossval (highest priority)
    // Use case: When C++ libraries are in a non-standard location
    // Example: BITNET_CROSSVAL_LIBDIR=/opt/custom/lib cargo build -p crossval --features llama-ffi
    if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
        possible_lib_dirs.push(Path::new(&lib_dir).to_path_buf());
    } else {
        // Priority 1-3: Build three-tier search path hierarchy
        // Specification Section 2.2 - fixes Gap 2 (missing path)
        let (primary_paths, embedded_paths, fallback_paths) = build_search_path_tiers(&bitnet_root);

        // Add in priority order: primary → embedded → fallback
        // This ensures BitNet-specific paths are checked before generic fallbacks
        possible_lib_dirs.extend(primary_paths);
        possible_lib_dirs.extend(embedded_paths);
        possible_lib_dirs.extend(fallback_paths);
    }

    // Track what we find
    let mut found_bitnet = false;
    let mut found_llama = false;
    let mut all_found_libs = Vec::new();
    let mut rpath_dirs = Vec::new();

    // Search directories
    for lib_dir in &possible_lib_dirs {
        if !lib_dir.exists() {
            continue;
        }

        println!("cargo:rustc-link-search=native={}", lib_dir.display());

        // Track if this directory contains any libraries (for RPATH emission)
        let mut dir_has_libs = false;

        // Scan for library files
        if let Ok(entries) = std::fs::read_dir(lib_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_stem().and_then(|s| s.to_str())
                    && let Some(ext) = path.extension()
                    && (ext == "so" || ext == "dylib" || ext == "a")
                {
                    // Detect BitNet libraries
                    if name.starts_with("libbitnet") {
                        let lib_name = name.strip_prefix("lib").unwrap_or(name);
                        println!("cargo:rustc-link-lib=dylib={}", lib_name);
                        all_found_libs.push(lib_name.to_string());
                        found_bitnet = true;
                        dir_has_libs = true;
                    }

                    // Detect LLaMA/GGML libraries with explicit linking
                    // Note: Explicit linking ensures clarity vs pattern matching
                    if name.starts_with("libllama") {
                        let lib_name = name.strip_prefix("lib").unwrap_or(name);
                        println!("cargo:rustc-link-lib=dylib={}", lib_name);
                        all_found_libs.push(lib_name.to_string());
                        found_llama = true;
                        dir_has_libs = true;
                    } else if name.starts_with("libggml") {
                        let lib_name = name.strip_prefix("lib").unwrap_or(name);
                        println!("cargo:rustc-link-lib=dylib={}", lib_name);
                        all_found_libs.push(lib_name.to_string());
                        // ggml implies llama support
                        found_llama = true;
                        dir_has_libs = true;
                    }
                }
            }
        }

        // Only add directory to RPATH if it contains relevant libraries
        if dir_has_libs {
            rpath_dirs.push(lib_dir.clone());
        }
    }

    // Emit single colon-separated RPATH entry for runtime library resolution (Linux/macOS)
    // This eliminates the need for LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
    // Only emit RPATH if libraries were actually found (not in STUB mode)
    if !rpath_dirs.is_empty() && (found_bitnet || found_llama) {
        let rpath: String = rpath_dirs
            .iter()
            .map(|p: &std::path::PathBuf| p.display().to_string())
            .collect::<Vec<_>>()
            .join(":");

        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath);

        #[cfg(target_os = "macos")]
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath);
    }

    // Determine backend availability state (Specification Section 4.2 - fixes Gap 1)
    // CRITICAL FIX: Replace simple bool conflation with three-state logic
    // Old (INCORRECT): let bitnet_available = preliminary_available && (found_bitnet || found_llama);
    // New (CORRECT): Three-state BackendState enum properly distinguishes:
    //   - FullBitNet: BitNet.cpp libraries found (requires headers)
    //   - LlamaFallback: Only llama.cpp libraries found (no headers needed)
    //   - Unavailable: No libraries found
    //
    // Fixed logic: LlamaFallback doesn't require preliminary_available (headers)
    // Only FullBitNet requires headers for BitNet.cpp
    let backend_state = match (found_bitnet, found_llama, preliminary_available) {
        (true, _, true) => BackendState::FullBitNet, // BitNet found + headers
        (false, true, _) => BackendState::LlamaFallback, // Only llama (no headers needed)
        _ => BackendState::Unavailable,              // Nothing found or BitNet without headers
    };

    let bitnet_available = backend_state.is_available();

    // Compile the C++ wrapper with the correct define based on actual availability
    let cc_wrapper_path = Path::new("src/bitnet_cpp_wrapper.cc");
    if cc_wrapper_path.exists() {
        let mut build = cc::Build::new();
        build.file(cc_wrapper_path).cpp(true).flag_if_supported("-std=c++17");

        // Add include paths if available
        if bitnet_available {
            // Standalone llama.cpp include paths (preferred)
            let llama_standalone_include = Path::new(&bitnet_root).join("include");
            if llama_standalone_include.exists() {
                build.include(&llama_standalone_include);
            }
            let ggml_standalone_include = Path::new(&bitnet_root).join("ggml/include");
            if ggml_standalone_include.exists() {
                build.include(&ggml_standalone_include);
            }

            // BitNet.cpp include paths
            let bitnet_include_dir = Path::new(&bitnet_root).join("include");
            if bitnet_include_dir.exists() {
                build.include(&bitnet_include_dir);
            }
            let src_dir = Path::new(&bitnet_root).join("src");
            if src_dir.exists() {
                build.include(&src_dir);
            }
            // Add llama.cpp include path for llama.h (embedded in BitNet.cpp)
            let llama_include = Path::new(&bitnet_root).join("3rdparty/llama.cpp/include");
            if llama_include.exists() {
                build.include(&llama_include);
            }
            // Add ggml include path (for ggml.h dependency in BitNet.cpp)
            let ggml_include = Path::new(&bitnet_root).join("3rdparty/llama.cpp/ggml/include");
            if ggml_include.exists() {
                build.include(&ggml_include);
            }
        }

        // Set the correct compilation mode based on actual library availability
        if bitnet_available {
            build.define("BITNET_AVAILABLE", None);
            println!("cargo:warning=crossval: Compiling C++ wrapper in BITNET_AVAILABLE mode");
        } else {
            build.define("BITNET_STUB", None);
            println!(
                "cargo:warning=crossval: Compiling C++ wrapper in BITNET_STUB mode (no libraries found)"
            );
        }

        build.compile("bitnet_cpp_wrapper_cc");
        println!("cargo:rustc-link-lib=static=bitnet_cpp_wrapper_cc");
    }

    // Emit build-time environment variables for runtime detection (Specification Section 3.1)
    println!("cargo:rustc-env=CROSSVAL_HAS_BITNET={}", found_bitnet);
    println!("cargo:rustc-env=CROSSVAL_HAS_LLAMA={}", found_llama);

    // NEW: Emit three-state backend status (Specification Section 3.1)
    println!("cargo:rustc-env=CROSSVAL_BACKEND_STATE={}", backend_state.as_str());

    // NEW: Emit RPATH for consumer (xtask/build.rs) - colon-separated paths (Specification Section 3.2)
    // Enhancement: Deduplicate and canonicalize paths before emission
    // Only emit if libraries were actually found (not in STUB mode)
    if !rpath_dirs.is_empty() && (found_bitnet || found_llama) {
        use std::collections::BTreeSet;

        // Deduplicate paths using BTreeSet (preserves sorted order for stability)
        let mut unique_rpath_dirs = BTreeSet::new();
        for dir in &rpath_dirs {
            // Canonicalize if possible, fallback to original path
            let canonical = dir.canonicalize().unwrap_or_else(|_| dir.clone());
            unique_rpath_dirs.insert(canonical.display().to_string());
        }

        let rpath_str: String = unique_rpath_dirs.iter().cloned().collect::<Vec<_>>().join(":");
        println!("cargo:rustc-env=CROSSVAL_RPATH_BITNET={}", rpath_str);

        // Emit diagnostic for transparency (shows number of unique library paths)
        println!(
            "cargo:warning=crossval: BitNet RPATH includes {} unique library paths",
            unique_rpath_dirs.len()
        );
    }

    // Only emit cfg(have_cpp) when libraries are actually found (not in STUB mode)
    // This ensures cfg(have_cpp) reliably indicates real C++ backend availability
    if backend_state.is_available() {
        println!("cargo:rustc-cfg=have_cpp");
    }

    // NEW: Emit cfg(have_bitnet_full) only when full BitNet.cpp backend available (Specification Section 3.1)
    if matches!(backend_state, BackendState::FullBitNet) {
        println!("cargo:rustc-cfg=have_bitnet_full");
    }

    // Emit enhanced diagnostic messages based on backend state (Specification Section 3.3 - fixes Gap 3)
    match backend_state {
        BackendState::FullBitNet => {
            println!(
                "cargo:warning=crossval: ✓ BITNET_FULL: BitNet.cpp and llama.cpp libraries found"
            );
            println!("cargo:warning=crossval: Backend: full");
            println!("cargo:warning=crossval: Linked libraries: {}", all_found_libs.join(", "));
            println!("cargo:warning=crossval: Headers found in: {}", bitnet_root);
        }
        BackendState::LlamaFallback => {
            println!(
                "cargo:warning=crossval: ⚠ LLAMA_FALLBACK: LLaMA.cpp libraries found, BitNet.cpp NOT found"
            );
            println!("cargo:warning=crossval: Backend: llama (fallback)");
            println!("cargo:warning=crossval: Linked libraries: {}", all_found_libs.join(", "));
            println!(
                "cargo:warning=crossval: BitNet backend unavailable - only llama.cpp cross-validation supported"
            );
            println!(
                "cargo:warning=crossval: To enable full BitNet.cpp: check git submodule status, rebuild with CMake"
            );
        }
        BackendState::Unavailable => {
            println!("cargo:warning=crossval: ✗ BITNET_STUB mode: No C++ libraries found");
            println!("cargo:warning=crossval: Backend: none");
            if !bitnet_cpp_dir_set {
                println!(
                    "cargo:warning=crossval: Set BITNET_CPP_DIR to enable C++ backend integration"
                );
                println!(
                    "cargo:warning=crossval: Or run: eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\""
                );
            } else if !has_headers {
                println!(
                    "cargo:warning=crossval: BITNET_CPP_DIR set but no headers found in: {}",
                    bitnet_root
                );
            } else {
                println!("cargo:warning=crossval: Headers found but no libraries detected");
                println!(
                    "cargo:warning=crossval: Check that BitNet.cpp is built in: {}",
                    bitnet_root
                );
            }
        }
    }

    // Link C++ standard library if we found any libraries
    if found_bitnet || found_llama {
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-lib=dylib=stdc++");

        #[cfg(target_os = "macos")]
        println!("cargo:rustc-link-lib=dylib=c++");
    }
}
