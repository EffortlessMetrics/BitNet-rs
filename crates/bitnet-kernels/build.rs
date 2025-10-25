use std::{
    env,
    path::{Path, PathBuf},
};

/// Safe environment variable access with fallback
fn env_var(name: &str) -> Option<String> {
    env::var_os(name).map(|v| v.to_string_lossy().into_owned())
}

/// Get HOME directory with safe fallbacks for minimal container environments
fn get_home_dir() -> PathBuf {
    // Try HOME environment variable first
    if let Some(home) = env_var("HOME") {
        return PathBuf::from(home);
    }

    // Fallback to /tmp with warning for minimal Docker containers
    println!("cargo:warning=HOME not set; falling back to /tmp for C++ artifact cache (build.rs)");
    PathBuf::from("/tmp")
}

fn main() {
    // Tell rustc that `cfg(have_cpp)` is a known conditional
    println!("cargo:rustc-check-cfg=cfg(have_cpp)");

    // Always allow re-run if this file changes
    println!("cargo:rerun-if-changed=build.rs");

    // Unified GPU detection: honor both "gpu" and legacy "cuda" features for back-compat.
    // This ensures the build script recognizes GPU builds regardless of which feature is enabled.
    // See Issue #439 for unified predicate approach.
    let gpu =
        env::var_os("CARGO_FEATURE_GPU").is_some() || env::var_os("CARGO_FEATURE_CUDA").is_some();

    if gpu {
        // Emit build-time cfg flag for unified GPU detection
        println!("cargo:rustc-cfg=bitnet_build_gpu");

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
    // Check BITNET_CPP_DIR first for consistency with other build.rs files (crossval, bitnet-sys)
    let root = env_var("BITNET_CPP_DIR")
        .or_else(|| env_var("BITNET_CPP_PATH"))
        .unwrap_or_else(|| format!("{}/.cache/bitnet_cpp", get_home_dir().display()));

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
