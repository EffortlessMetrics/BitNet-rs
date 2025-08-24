fn main() {
    println!("cargo:rustc-check-cfg=cfg(have_cpp)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/bitnet_cpp_wrapper.c");

    // Only compile if ffi feature is enabled
    #[cfg(feature = "ffi")]
    compile_ffi();
}

#[cfg(feature = "ffi")]
fn compile_ffi() {
    use std::{env, path::Path};

    // Compile our C wrapper
    cc::Build::new().file("src/bitnet_cpp_wrapper.c").compile("bitnet_cpp_wrapper");

    let root = env::var("BITNET_CPP_DIR")
        .or_else(|_| env::var("BITNET_CPP_PATH")) // Legacy support
        .unwrap_or_else(|_| format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap()));

    // Try multiple possible library locations
    let possible_lib_dirs = [
        Path::new(&root).join("build").join("lib"),
        Path::new(&root).join("build"),
        Path::new(&root).join("lib"),
    ];

    // We always have the wrapper now
    println!("cargo:rustc-cfg=have_cpp");

    // Link to the actual BitNet library if available
    let mut found_lib_dir = None;
    for lib_dir in &possible_lib_dirs {
        if lib_dir.exists() {
            found_lib_dir = Some(lib_dir);
            break;
        }
    }

    if let Some(lib) = found_lib_dir {
        println!("cargo:rustc-link-search=native={}", lib.display());

        // Look for any available libraries (more flexible naming)
        let lib_files =
            std::fs::read_dir(lib).unwrap_or_else(|_| panic!("Could not read lib directory"));
        let mut found_libs = false;

        for entry in lib_files.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                // Look for common library patterns
                if (name.starts_with("libbitnet")
                    || name.starts_with("libllama")
                    || name.starts_with("libggml"))
                    && (path
                        .extension()
                        .map_or(false, |ext| ext == "so" || ext == "dylib" || ext == "a"))
                {
                    let lib_name = name.strip_prefix("lib").unwrap_or(name);
                    println!("cargo:rustc-link-lib=dylib={}", lib_name);
                    found_libs = true;
                }
            }
        }

        if found_libs {
            println!("cargo:warning=bitnet-crossval: C++ libraries found and linked from {}", root);

            // Also link C++ standard library
            #[cfg(target_os = "linux")]
            println!("cargo:rustc-link-lib=dylib=stdc++");
            #[cfg(target_os = "macos")]
            println!("cargo:rustc-link-lib=dylib=c++");
        } else {
            println!(
                "cargo:warning=bitnet-crossval: Using mock C wrapper (no recognized libraries found)"
            );
        }
    } else {
        println!(
            "cargo:warning=bitnet-crossval: Using mock C wrapper (no library directories found)"
        );
    }
}
