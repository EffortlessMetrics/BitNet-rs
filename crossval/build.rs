fn main() {
    println!("cargo:rustc-check-cfg=cfg(have_cpp)");
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
    let bitnet_root =
        env::var("BITNET_CPP_DIR").or_else(|_| env::var("BITNET_CPP_PATH")).unwrap_or_else(|_| {
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

    // Build multi-tier library search paths
    let mut possible_lib_dirs = Vec::new();

    // Priority 1: Explicit BITNET_CROSSVAL_LIBDIR
    if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
        possible_lib_dirs.push(Path::new(&lib_dir).to_path_buf());
    }

    // Priority 2: BITNET_CPP_DIR (already retrieved above for header checks)

    // Add potential lib directories
    // llama.cpp standalone build puts libraries in build/bin/
    possible_lib_dirs.push(Path::new(&bitnet_root).join("build/bin"));
    possible_lib_dirs.push(Path::new(&bitnet_root).join("build"));
    possible_lib_dirs.push(Path::new(&bitnet_root).join("build/lib"));
    // BitNet.cpp embedded llama.cpp paths
    possible_lib_dirs.push(Path::new(&bitnet_root).join("build/3rdparty/llama.cpp/src"));
    possible_lib_dirs.push(Path::new(&bitnet_root).join("build/3rdparty/llama.cpp/ggml/src"));
    possible_lib_dirs.push(Path::new(&bitnet_root).join("lib"));

    // Track what we find
    let mut found_bitnet = false;
    let mut found_llama = false;
    let mut all_found_libs = Vec::new();

    // Search directories
    for lib_dir in &possible_lib_dirs {
        if !lib_dir.exists() {
            continue;
        }

        println!("cargo:rustc-link-search=native={}", lib_dir.display());

        // Add RPATH for runtime library resolution (Linux/macOS)
        // This eliminates the need for LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

        #[cfg(target_os = "macos")]
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

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
                    }

                    // Detect LLaMA/GGML libraries
                    if name.starts_with("libllama") || name.starts_with("libggml") {
                        let lib_name = name.strip_prefix("lib").unwrap_or(name);
                        println!("cargo:rustc-link-lib=dylib={}", lib_name);
                        all_found_libs.push(lib_name.to_string());
                        found_llama = true;
                    }
                }
            }
        }
    }

    // Determine if BitNet.cpp is truly available (headers + libraries)
    let bitnet_available = preliminary_available && (found_bitnet || found_llama);

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

    // Emit build-time environment variables for runtime detection
    println!("cargo:rustc-env=CROSSVAL_HAS_BITNET={}", found_bitnet);
    println!("cargo:rustc-env=CROSSVAL_HAS_LLAMA={}", found_llama);

    // We always have the wrapper now (cfg for conditional compilation)
    println!("cargo:rustc-cfg=have_cpp");

    // Emit diagnostic messages
    if bitnet_available {
        if found_bitnet && found_llama {
            println!(
                "cargo:warning=crossval: ✓ BITNET_AVAILABLE: Both bitnet.cpp and llama.cpp libraries found"
            );
            println!("cargo:warning=crossval: Dual-backend cross-validation supported");
        } else if found_bitnet {
            println!("cargo:warning=crossval: ✓ BITNET_AVAILABLE: BitNet.cpp libraries found");
            println!("cargo:warning=crossval: BitNet parity validation supported");
        } else if found_llama {
            println!("cargo:warning=crossval: ✓ BITNET_AVAILABLE: LLaMA.cpp libraries found");
            println!("cargo:warning=crossval: LLaMA parity validation supported");
        }
        println!("cargo:warning=crossval: Linked libraries: {}", all_found_libs.join(", "));
        println!("cargo:warning=crossval: Headers found in: {}", bitnet_root);
    } else {
        println!("cargo:warning=crossval: ✗ BITNET_STUB mode: No C++ libraries found");
        if !bitnet_cpp_dir_set {
            println!(
                "cargo:warning=crossval: Set BITNET_CPP_DIR to enable C++ backend integration"
            );
        } else if !has_headers {
            println!(
                "cargo:warning=crossval: BITNET_CPP_DIR set but no headers found in: {}",
                bitnet_root
            );
        } else {
            println!("cargo:warning=crossval: Headers found but no libraries detected");
            println!("cargo:warning=crossval: Check that BitNet.cpp is built in: {}", bitnet_root);
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
