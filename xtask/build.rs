fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BITNET_CROSSVAL_LIBDIR");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_DIR");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_PATH"); // Deprecated but still watched
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR"); // Standalone llama.cpp support
    println!("cargo:rerun-if-env-changed=CROSSVAL_RPATH_BITNET");
    println!("cargo:rerun-if-env-changed=CROSSVAL_RPATH_LLAMA");

    // Emit RPATH for xtask runtime loader stability (optional but recommended)
    //
    // Why this is needed:
    //   - xtask binary may invoke crossval functions that dynamically link to C++ backends
    //   - Without RPATH, users must remember to export LD_LIBRARY_PATH before running xtask
    //   - With RPATH, the loader finds libraries automatically
    //
    // When this applies:
    //   - Only when BITNET_CROSSVAL_LIBDIR is set (explicit library directory)
    //   - Only on Linux/macOS (Windows uses PATH for DLL search)
    //
    // See also:
    //   - crossval/build.rs for FFI wrapper compilation
    //   - docs/howto/cpp-setup.md for C++ reference setup
    #[cfg(any(feature = "crossval", feature = "crossval-all", feature = "ffi"))]
    embed_crossval_rpath();
}

#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "ffi"))]
fn embed_crossval_rpath() {
    use std::{env, path::Path};

    // Priority 1: Legacy single-directory override (backward compatibility)
    if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
        let lib_path = Path::new(&lib_dir);
        if lib_path.exists() {
            // Emit warning if user also set new variables
            if env::var("CROSSVAL_RPATH_BITNET").is_ok() || env::var("CROSSVAL_RPATH_LLAMA").is_ok()
            {
                println!(
                    "cargo:warning=xtask: Both BITNET_CROSSVAL_LIBDIR and CROSSVAL_RPATH_* set. \
                     Using BITNET_CROSSVAL_LIBDIR (takes precedence)."
                );
            }
            emit_rpath(lib_path.display().to_string());
            return;
        } else {
            println!(
                "cargo:warning=xtask: BITNET_CROSSVAL_LIBDIR set but directory does not exist: {}",
                lib_dir
            );
        }
    }

    // Priority 2: Read granular environment variables and merge
    let mut rpath_candidates: Vec<String> = Vec::new();

    if let Ok(bitnet_path) = env::var("CROSSVAL_RPATH_BITNET") {
        let path = Path::new(&bitnet_path);
        if path.exists() {
            rpath_candidates.push(bitnet_path);
        } else {
            println!(
                "cargo:warning=xtask: CROSSVAL_RPATH_BITNET set but directory does not exist: {}",
                bitnet_path
            );
        }
    }

    if let Ok(llama_path) = env::var("CROSSVAL_RPATH_LLAMA") {
        let path = Path::new(&llama_path);
        if path.exists() {
            rpath_candidates.push(llama_path);
        } else {
            println!(
                "cargo:warning=xtask: CROSSVAL_RPATH_LLAMA set but directory does not exist: {}",
                llama_path
            );
        }
    }

    // Merge and deduplicate if we have candidates
    if !rpath_candidates.is_empty() {
        // Import merge function from our library
        // Note: We need to use include! or paste the function here since build.rs
        // can't easily depend on the library being built
        let rpath_refs: Vec<&str> = rpath_candidates.iter().map(|s| s.as_str()).collect();
        let merged = merge_and_deduplicate(&rpath_refs);
        emit_rpath(merged);
        return;
    }

    // Priority 3: Fallback to BITNET_CPP_DIR and LLAMA_CPP_DIR auto-discovery
    // Support both backends with merged RPATH (AC14-AC17)
    let mut all_candidate_dirs: Vec<std::path::PathBuf> = Vec::new();

    // Priority 3a: BitNet.cpp auto-discovery
    // Support deprecated BITNET_CPP_PATH for backward compatibility
    let bitnet_cpp_dir = env::var("BITNET_CPP_DIR").or_else(|_| {
        // Deprecated: BITNET_CPP_PATH fallback
        if let Ok(path) = env::var("BITNET_CPP_PATH") {
            println!(
                "cargo:warning=xtask: BITNET_CPP_PATH is deprecated. Use BITNET_CPP_DIR instead."
            );
            Ok(path)
        } else {
            Err(std::env::VarError::NotPresent)
        }
    });

    if let Ok(cpp_dir) = bitnet_cpp_dir {
        use std::path::PathBuf;
        let cpp_path = Path::new(&cpp_dir);

        // Build multi-tier search paths matching crossval/build.rs (Tier 1 + Tier 2)
        // This ensures xtask finds the same libraries that crossval FFI links against

        // Tier 1: PRIMARY BitNet.cpp locations (checked first)
        let thirdparty_bin = cpp_path.join("build/3rdparty/llama.cpp/build/bin");
        if thirdparty_bin.exists() {
            all_candidate_dirs.push(thirdparty_bin);
        }

        let build_lib = cpp_path.join("build/lib");
        if build_lib.exists() {
            all_candidate_dirs.push(build_lib);
        }

        let build_bin = cpp_path.join("build/bin");
        if build_bin.exists() {
            all_candidate_dirs.push(build_bin);
        }

        // Tier 2: EMBEDDED llama.cpp locations
        let thirdparty_src = cpp_path.join("build/3rdparty/llama.cpp/src");
        if thirdparty_src.exists() {
            all_candidate_dirs.push(thirdparty_src);
        }

        let thirdparty_ggml = cpp_path.join("build/3rdparty/llama.cpp/ggml/src");
        if thirdparty_ggml.exists() {
            all_candidate_dirs.push(thirdparty_ggml);
        }

        // Tier 3: FALLBACK locations (last resort)
        let build_dir = cpp_path.join("build");
        if build_dir.exists() {
            all_candidate_dirs.push(build_dir);
        }
    }

    // Priority 3b: Standalone llama.cpp auto-discovery (AC10-AC12, AC14-AC17)
    if let Ok(llama_dir) = env::var("LLAMA_CPP_DIR") {
        use std::path::PathBuf;
        let llama_path = Path::new(&llama_dir);

        // Tier 1: PRIMARY llama.cpp locations (standalone build structure)
        // llama.cpp uses different build layout than BitNet.cpp:
        //   - build/            ← libllama.so, libggml.so (top-level)
        //   - build/bin/        ← CMake bin output
        //   - build/lib/        ← CMake lib output
        let build_top = llama_path.join("build");
        if build_top.exists() {
            all_candidate_dirs.push(build_top);
        }

        let build_bin = llama_path.join("build/bin");
        if build_bin.exists() {
            all_candidate_dirs.push(build_bin);
        }

        let build_lib = llama_path.join("build/lib");
        if build_lib.exists() {
            all_candidate_dirs.push(build_lib);
        }

        // Tier 2: Alternative llama.cpp build locations
        let src_dir = llama_path.join("src");
        if src_dir.exists() {
            all_candidate_dirs.push(src_dir);
        }

        let ggml_src = llama_path.join("ggml/src");
        if ggml_src.exists() {
            all_candidate_dirs.push(ggml_src);
        }
    }

    // Merge all discovered candidate directories into single RPATH (AC16)
    // Deduplication and ordering handled by merge_and_deduplicate (AC17)
    if !all_candidate_dirs.is_empty() {
        let rpath_refs: Vec<&str> = all_candidate_dirs
            .iter()
            .map(|p| p.to_str().unwrap_or(""))
            .filter(|s| !s.is_empty())
            .collect();
        let merged = merge_and_deduplicate(&rpath_refs);
        emit_rpath(merged);
        return; // Early return after successful RPATH emission
    }

    // Priority 4: Default installation paths (when no env vars set)
    // Check both default BitNet.cpp and llama.cpp installation locations
    if let Ok(home) = env::var("HOME") {
        use std::path::PathBuf;
        let mut default_candidate_dirs: Vec<PathBuf> = Vec::new();

        // Priority 4a: Default BitNet.cpp installation ($HOME/.cache/bitnet_cpp)
        // Matches crossval/build.rs default behavior
        let default_bitnet = PathBuf::from(&home).join(".cache/bitnet_cpp");

        // Check default BitNet.cpp installation paths
        let thirdparty_bin = default_bitnet.join("build/3rdparty/llama.cpp/build/bin");
        if thirdparty_bin.exists() {
            default_candidate_dirs.push(thirdparty_bin);
        }

        let build_lib = default_bitnet.join("build/lib");
        if build_lib.exists() {
            default_candidate_dirs.push(build_lib);
        }

        let build_bin = default_bitnet.join("build/bin");
        if build_bin.exists() {
            default_candidate_dirs.push(build_bin);
        }

        // Vendored llama.cpp in default BitNet.cpp location
        let llama_vendored = default_bitnet.join("build/3rdparty/llama.cpp/src");
        if llama_vendored.exists() {
            default_candidate_dirs.push(llama_vendored);
        }

        let ggml_vendored = default_bitnet.join("build/3rdparty/llama.cpp/ggml/src");
        if ggml_vendored.exists() {
            default_candidate_dirs.push(ggml_vendored);
        }

        // Priority 4b: Default llama.cpp installation ($HOME/.cache/llama_cpp)
        // Standalone llama.cpp default location
        let default_llama = PathBuf::from(&home).join(".cache/llama_cpp");

        let llama_build = default_llama.join("build");
        if llama_build.exists() {
            default_candidate_dirs.push(llama_build);
        }

        let llama_build_bin = default_llama.join("build/bin");
        if llama_build_bin.exists() {
            default_candidate_dirs.push(llama_build_bin);
        }

        let llama_build_lib = default_llama.join("build/lib");
        if llama_build_lib.exists() {
            default_candidate_dirs.push(llama_build_lib);
        }

        // Emit RPATH if any default installation found
        if !default_candidate_dirs.is_empty() {
            let rpath_refs: Vec<&str> = default_candidate_dirs
                .iter()
                .map(|p| p.to_str().unwrap_or(""))
                .filter(|s| !s.is_empty())
                .collect();
            let merged = merge_and_deduplicate(&rpath_refs);
            emit_rpath(merged);
        }
    }

    // No library directory found - this is fine, xtask will work without crossval features
    // User can run setup-cpp-auto to bootstrap environment later
}

#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "ffi"))]
fn merge_and_deduplicate(paths: &[&str]) -> String {
    use std::collections::HashSet;
    use std::path::PathBuf;

    const MAX_RPATH_LENGTH: usize = 4096; // Conservative limit for linker

    let mut seen = HashSet::new();
    let mut merged = Vec::new();

    for path_str in paths {
        let path = PathBuf::from(path_str);

        // Canonicalize to resolve symlinks and normalize paths
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "cargo:warning=xtask: Failed to canonicalize path {}: {}. Skipping.",
                    path.display(),
                    e
                );
                continue; // Skip invalid paths
            }
        };

        // Deduplicate using canonical path
        if seen.insert(canonical.clone()) {
            merged.push(canonical);
        }
    }

    // Join with colon separator (POSIX RPATH syntax)
    let result = merged.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(":");

    // Sanity check: RPATH length limit
    if result.len() > MAX_RPATH_LENGTH {
        panic!(
            "Merged RPATH exceeds maximum length ({} > {}). \
             Please use BITNET_CROSSVAL_LIBDIR to specify a single directory, \
             or reduce the number of library paths.",
            result.len(),
            MAX_RPATH_LENGTH
        );
    }

    result
}

#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "ffi"))]
fn emit_rpath(rpath: String) {
    // Emit link search directive (compile-time)
    // Note: rustc-link-search does NOT support colon-separated paths,
    // so we emit the first path only for link-time resolution.
    // The full merged path is used in rustc-link-arg for runtime RPATH.
    let first_path = rpath.split(':').next().unwrap_or(&rpath);
    println!("cargo:rustc-link-search=native={}", first_path);

    // Emit RPATH for runtime library resolution
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath);
        println!("cargo:warning=xtask: Embedded merged RPATH for runtime loader: {}", rpath);
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath);
        println!("cargo:warning=xtask: Embedded merged RPATH for runtime loader: {}", rpath);
    }

    #[cfg(target_os = "windows")]
    {
        println!(
            "cargo:warning=xtask: Windows detected - RPATH not applicable. \
             Ensure libraries are in PATH or use setup-cpp-auto. \
             Merged paths: {}",
            rpath
        );
    }
}
