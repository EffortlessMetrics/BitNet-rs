//! Build script for bitnet-sys crate
//!
//! This script links against the Microsoft BitNet C++ implementation when
//! the crossval feature is enabled. It fails fast if dependencies are missing.

use std::env;
use std::path::PathBuf;

fn main() {
    // If the crate is compiled without `--features bitnet-sys/ffi`,
    // skip all native build steps so the workspace remains green.
    if std::env::var("CARGO_FEATURE_FFI").is_err() {
        println!("cargo:warning=bitnet-sys: 'ffi' feature not enabled; skipping native build");
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_DIR");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_PATH"); // Legacy support

    #[cfg(feature = "ffi")]
    {
        // When crossval feature is enabled, we REQUIRE the C++ implementation
        let cpp_dir = env::var("BITNET_CPP_DIR")
            .or_else(|_| env::var("BITNET_CPP_PATH")) // Try legacy env var
            .map(PathBuf::from)
            .expect(
                "BITNET_CPP_DIR must be set to the Microsoft BitNet repository root.\n\
                 Run: ./ci/fetch_bitnet_cpp.sh\n\
                 Then: export BITNET_CPP_DIR=$HOME/.cache/bitnet_cpp"
            );

        if !cpp_dir.exists() {
            panic!(
                "bitnet-sys: BITNET_CPP_DIR points to non-existent path: {}\n\
                 Run: ./ci/fetch_bitnet_cpp.sh",
                cpp_dir.display()
            );
        }

        // Verify the C++ implementation is built
        let build_dir = cpp_dir.join("build");
        if !build_dir.exists() {
            panic!(
                "bitnet-sys: BitNet C++ not built. Build directory missing: {}\n\
                 Run: ./ci/fetch_bitnet_cpp.sh",
                build_dir.display()
            );
        }

        println!("cargo:warning=bitnet-sys: Building with cross-validation support");
        println!("cargo:warning=bitnet-sys: Using BitNet C++ from: {}", cpp_dir.display());

        // Link against the C++ implementation - fail on error
        link_cpp_implementation(&cpp_dir).expect("Failed to link Microsoft BitNet C++ libraries");

        // Generate bindings - fail on error
        generate_bindings(&cpp_dir)
            .expect("Failed to generate FFI bindings from Microsoft BitNet headers");
    }

    #[cfg(not(feature = "ffi"))]
    {
        // When ffi is disabled, create minimal bindings
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        std::fs::write(
            out_path.join("bindings.rs"),
            "// Bindings disabled - ffi feature not enabled\n",
        )
        .unwrap();
    }
}

#[cfg(feature = "ffi")]
fn link_cpp_implementation(cpp_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let build_dir = cpp_dir.join("build");

    // Library search paths - order matters!
    let lib_search_paths = [
        build_dir.join("3rdparty/llama.cpp/src"),
        build_dir.join("3rdparty/llama.cpp/ggml/src"),
        build_dir.join("3rdparty/llama.cpp"),
        build_dir.join("lib"),
        build_dir.clone(),
    ];

    // Add all existing search paths
    let mut found_any = false;
    for path in &lib_search_paths {
        if path.exists() {
            println!("cargo:rustc-link-search=native={}", path.display());

            // Add RPATH for runtime library resolution (Linux/macOS)
            // This eliminates the need for LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
            }

            found_any = true;
        }
    }

    if !found_any {
        return Err("No library directories found. Is BitNet C++ built?".into());
    }

    // Helper to check if a library exists
    fn lib_present(dir: &std::path::Path, name: &str) -> bool {
        let so = dir.join(format!("lib{}.so", name));
        let dylib = dir.join(format!("lib{}.dylib", name));
        let a = dir.join(format!("lib{}.a", name));
        so.exists() || dylib.exists() || a.exists()
    }

    // Link the main llama library (required)
    println!("cargo:rustc-link-lib=dylib=llama");

    // Only link ggml if it exists as a separate library
    // (it might be statically linked into llama)
    if lib_search_paths.iter().any(|dir| lib_present(dir, "ggml")) {
        println!("cargo:rustc-link-lib=dylib=ggml");
    }

    // Platform-specific runtime dependencies
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=m");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-lib=dylib=msvcrt");
    }

    Ok(())
}

#[cfg(feature = "ffi")]
fn generate_bindings(cpp_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // We'll generate bindings from llama.h which is the main C API
    let mut llama_h = cpp_dir.join("3rdparty/llama.cpp/include/llama.h");
    if !llama_h.exists() {
        // Try alternate location
        llama_h = cpp_dir.join("include/llama.h");
        if !llama_h.exists() {
            return Err(format!(
                "llama.h not found. Expected at: {}\n\
                 Is the Microsoft BitNet repository complete?",
                llama_h.display()
            )
            .into());
        }
    }

    // Also check for BitNet-specific headers
    let bitnet_h = cpp_dir.join("include/ggml-bitnet.h");
    let use_bitnet = bitnet_h.exists();

    println!("cargo:warning=bitnet-sys: Generating bindings from {}", llama_h.display());
    if use_bitnet {
        println!(
            "cargo:warning=bitnet-sys: Also including BitNet-specific APIs from {}",
            bitnet_h.display()
        );
    }

    let mut builder = bindgen::Builder::default()
        .header(llama_h.to_string_lossy())
        // Include paths
        .clang_arg(format!("-I{}", cpp_dir.join("3rdparty/llama.cpp/include").display()))
        .clang_arg(format!("-I{}", cpp_dir.join("3rdparty/llama.cpp/ggml/include").display()))
        .clang_arg(format!("-I{}", cpp_dir.join("include").display()))
        // Main llama.cpp C API
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*")
        // GGML types we might need
        .allowlist_type("ggml_.*")
        .allowlist_var("GGML_.*")
        // Standard options
        .derive_debug(true)
        .derive_default(true)
        // Note: derive_copy automatically includes Clone
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    // Add BitNet-specific header if available
    if use_bitnet {
        builder = builder
            .header(bitnet_h.to_string_lossy())
            .allowlist_function("ggml_bitnet_.*")
            .allowlist_function("ggml_qgemm_lut")
            .allowlist_function("ggml_preprocessor");
    }

    let bindings = builder.generate().map_err(|e| format!("bindgen failed: {}", e))?;

    let out_path = PathBuf::from(env::var("OUT_DIR")?);
    bindings.write_to_file(out_path.join("bindings.rs"))?;

    println!("cargo:warning=bitnet-sys: Generated C++ bindings successfully");
    Ok(())
}
