//! Build script for compiling C++ kernels during FFI bridge transition
//!
//! This build script compiles existing C++ kernel implementations when the
//! ffi-bridge feature is enabled, allowing for gradual migration from C++
//! to native Rust implementations.

#[cfg(feature = "ffi-bridge")]
fn main() {
    use std::env;
    use std::path::PathBuf;

    println!("cargo:rerun-if-changed=../../src/ggml-bitnet-lut.cpp");
    println!("cargo:rerun-if-changed=../../src/ggml-bitnet-mad.cpp");
    println!("cargo:rerun-if-changed=../../include/ggml-bitnet.h");
    println!("cargo:rerun-if-changed=build.rs");

    // Get the workspace root directory
    let workspace_root = env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let cpp_src_dir = workspace_root.join("src");
    let cpp_include_dir = workspace_root.join("include");

    // Check if C++ source files exist
    let lut_cpp = cpp_src_dir.join("ggml-bitnet-lut.cpp");
    let mad_cpp = cpp_src_dir.join("ggml-bitnet-mad.cpp");
    let header = cpp_include_dir.join("ggml-bitnet.h");

    if !lut_cpp.exists() || !mad_cpp.exists() || !header.exists() {
        println!("cargo:warning=C++ kernel sources not found, FFI bridge will not be functional");
        println!("cargo:warning=Expected files:");
        println!("cargo:warning=  {}", lut_cpp.display());
        println!("cargo:warning=  {}", mad_cpp.display());
        println!("cargo:warning=  {}", header.display());
        return;
    }

    // Configure C++ compilation
    let mut build = cc::Build::new();

    build
        .cpp(true)
        .std("c++17")
        .include(&cpp_include_dir)
        .file(&lut_cpp)
        .file(&mad_cpp)
        .file("src/ffi/cpp_bridge.cpp"); // Our bridge implementation

    // Platform-specific optimizations
    if cfg!(target_arch = "x86_64") {
        build.flag("-mavx2").flag("-mfma").flag("-mf16c");

        // Enable AVX-512 if available
        if env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default().contains("avx512f") {
            build.flag("-mavx512f");
        }
    } else if cfg!(target_arch = "aarch64") {
        // ARM64 optimizations
        build.flag("-march=armv8-a+simd");
    }

    // Debug/Release configuration
    if env::var("PROFILE").unwrap() == "release" {
        build.opt_level(3).flag("-DNDEBUG").flag("-ffast-math");
    } else {
        build.opt_level(0).debug(true).flag("-DDEBUG");
    }

    // Compile the C++ code
    build.compile("bitnet_cpp_kernels");

    // Link against math library on Unix
    if cfg!(unix) {
        println!("cargo:rustc-link-lib=m");
    }

    // Generate bindings if bindgen is available
    #[cfg(feature = "ffi-bridge")]
    generate_bindings(&cpp_include_dir);
}

#[cfg(feature = "ffi-bridge")]
fn generate_bindings(include_dir: &std::path::Path) {
    use std::env;
    use std::path::PathBuf;

    let bindings = ::bindgen::Builder::default()
        .header(include_dir.join("ggml-bitnet.h").to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .parse_callbacks(Box::new(::bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs")).expect("Couldn't write bindings!");
}

#[cfg(not(feature = "ffi-bridge"))]
fn main() {
    // Do nothing when FFI bridge is not enabled
}
