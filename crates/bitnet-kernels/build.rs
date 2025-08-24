use std::{env, path::Path};

fn main() {
    // Tell rustc that `cfg(have_cpp)` is a known conditional
    println!("cargo:rustc-check-cfg=cfg(have_cpp)");

    // Always allow re-run if this file changes
    println!("cargo:rerun-if-changed=build.rs");

    // CUDA configuration
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        // Add CUDA library paths
        println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=/usr/local/cuda/lib64/stubs");
        println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-search=/usr/lib64");

        // Try NVIDIA Jetson paths for ARM64
        println!("cargo:rustc-link-search=/usr/local/cuda/targets/aarch64-linux/lib");
        println!("cargo:rustc-link-search=/usr/local/cuda/targets/aarch64-linux/lib/stubs");

        // Common paths for CUDA installations
        println!("cargo:rustc-link-search=/usr/local/cuda/targets/x86_64-linux");
        println!("cargo:rustc-link-search=/usr/local/cuda/targets/x86_64-linux/lib");
        println!("cargo:rustc-link-search=/usr/local/cuda/targets/x86_64-linux/lib/stubs");

        // Link CUDA libraries
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=nvrtc");
        println!("cargo:rustc-link-lib=curand");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
    }

    // Only do FFI detection work if the crate feature "ffi" is enabled
    let ffi_enabled = env::var_os("CARGO_FEATURE_FFI").is_some();
    if !ffi_enabled {
        // No link lines or warnings in non-FFI builds
        return;
    }

    // Where the C++ artifacts live. Allow override via env.
    let root = env::var("BITNET_CPP_PATH")
        .unwrap_or_else(|_| format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap()));

    let inc = Path::new(&root).join("include");
    let lib = Path::new(&root).join("build").join("lib");

    // Check for the presence of C++ headers and libraries
    let have_header = inc.join("ggml-bitnet.h").exists();
    let have_static = lib.join("libbitnet.a").exists() || lib.join("libbitnet_static.a").exists();
    let have_shared = lib.join("libbitnet.so").exists() || lib.join("libbitnet.dylib").exists();

    // Also check for individual component libraries
    let have_components = lib.join("libggml.a").exists() || lib.join("libggml.so").exists();

    if have_header && (have_static || have_shared || have_components) {
        // We found the C++ library!
        println!("cargo:rustc-cfg=have_cpp");
        println!("cargo:rustc-link-search=native={}", lib.display());

        // Link the libraries we found
        if have_static {
            if lib.join("libbitnet.a").exists() {
                println!("cargo:rustc-link-lib=static=bitnet");
            } else if lib.join("libbitnet_static.a").exists() {
                println!("cargo:rustc-link-lib=static=bitnet_static");
            }
        } else if have_shared {
            println!("cargo:rustc-link-lib=dylib=bitnet");
        }

        // If we have component libraries, link them too
        if lib.join("libggml.a").exists() {
            println!("cargo:rustc-link-lib=static=ggml");
        }
        if lib.join("libllama.a").exists() {
            println!("cargo:rustc-link-lib=static=llama");
        }

        eprintln!("bitnet-kernels: FFI bridge enabled with C++ library from {}", root);
    } else {
        // DO NOT emit link lines. Build the crate with FFI stubs instead.
        eprintln!(
            "bitnet-kernels: FFI enabled but C++ library not found at {}; using stub implementation",
            root
        );
        eprintln!(
            "bitnet-kernels: To enable the real FFI bridge, build the C++ library with `cargo xtask fetch-cpp` or set BITNET_CPP_PATH"
        );
    }
}
