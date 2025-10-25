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

    // Only compile if ffi feature is enabled
    #[cfg(feature = "ffi")]
    compile_ffi();
}

#[cfg(feature = "ffi")]
fn compile_ffi() {
    use std::{env, path::Path};

    // Check if BITNET_CPP_DIR is set to determine compilation mode
    let bitnet_available = env::var("BITNET_CPP_DIR").is_ok();

    // Compile C++ wrapper (.cc file)
    let cc_wrapper_path = Path::new("src/bitnet_cpp_wrapper.cc");
    if cc_wrapper_path.exists() {
        let mut build = cc::Build::new();
        build.file(cc_wrapper_path).cpp(true).flag_if_supported("-std=c++17");

        // Set compilation mode based on BITNET_CPP_DIR availability
        if bitnet_available {
            build.define("BITNET_AVAILABLE", None);
            println!("cargo:warning=crossval: Compiling C++ wrapper in AVAILABLE mode");
        } else {
            build.define("BITNET_STUB", None);
            println!(
                "cargo:warning=crossval: Compiling C++ wrapper in STUB mode (set BITNET_CPP_DIR for real integration)"
            );
        }

        build.compile("bitnet_cpp_wrapper_cc");

        // Emit link directive so tests can find the wrapper
        println!("cargo:rustc-link-lib=static=bitnet_cpp_wrapper_cc");
    }

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

    // Priority 2: BITNET_CPP_DIR (if set)
    let bitnet_root =
        env::var("BITNET_CPP_DIR").or_else(|_| env::var("BITNET_CPP_PATH")).unwrap_or_else(|_| {
            format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap_or_else(|_| ".".into()))
        });

    // Add potential lib directories
    possible_lib_dirs.push(Path::new(&bitnet_root).join("build"));
    possible_lib_dirs.push(Path::new(&bitnet_root).join("build/lib"));
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

    // Emit build-time environment variables for runtime detection
    println!("cargo:rustc-env=CROSSVAL_HAS_BITNET={}", found_bitnet);
    println!("cargo:rustc-env=CROSSVAL_HAS_LLAMA={}", found_llama);

    // We always have the wrapper now (cfg for conditional compilation)
    println!("cargo:rustc-cfg=have_cpp");

    // Emit diagnostic messages
    if found_bitnet && found_llama {
        println!(
            "cargo:warning=crossval: Both bitnet.cpp and llama.cpp libraries found (dual-backend support)"
        );
        println!("cargo:warning=crossval: Linked libraries: {}", all_found_libs.join(", "));
    } else if found_bitnet {
        println!(
            "cargo:warning=crossval: Found bitnet.cpp libraries only (BitNet parity supported)"
        );
        println!("cargo:warning=crossval: Linked libraries: {}", all_found_libs.join(", "));
    } else if found_llama {
        println!("cargo:warning=crossval: Found llama.cpp libraries only (LLaMA parity supported)");
        println!("cargo:warning=crossval: Linked libraries: {}", all_found_libs.join(", "));
    } else {
        println!("cargo:warning=crossval: No C++ libraries found (crossval will use mock/stub)");
        println!("cargo:warning=crossval: Set BITNET_CPP_DIR to enable C++ backend integration");
    }

    // Link C++ standard library if we found any libraries
    if found_bitnet || found_llama {
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-lib=dylib=stdc++");

        #[cfg(target_os = "macos")]
        println!("cargo:rustc-link-lib=dylib=c++");
    }
}
