//! Build script for bitnet-sys crate
//!
//! This script links against the Microsoft BitNet C++ implementation when
//! the crossval feature is enabled. It fails fast if dependencies are missing.

use std::env;
#[allow(unused_imports)]
use std::path::Path;
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
        // When crossval feature is enabled, try to find the C++ implementation
        let cpp_dir = env::var("BITNET_CPP_DIR")
            .or_else(|_| env::var("BITNET_CPP_PATH")) // Try legacy env var
            .or_else(|_| env::var("HOME").map(|h| format!("{}/.cache/bitnet_cpp", h)))
            .map(PathBuf::from)
            .unwrap_or_default();

        if cpp_dir.as_os_str().is_empty() || !cpp_dir.exists() {
            println!("cargo:warning=BITNET_CPP_DIR not set/invalid; building without C++ bridge");
            println!("cargo:rustc-cfg=bitnet_cpp_unavailable");
            
            // Create minimal bindings file
            let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
            std::fs::write(
                out_path.join("bindings.rs"),
                "// C++ bridge unavailable - BITNET_CPP_DIR not set\n",
            )
            .unwrap();
            return;
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
fn link_cpp_implementation(cpp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
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
fn generate_bindings(cpp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // We'll generate bindings from llama.h which is the main C API
    // Try multiple possible locations for llama.h in the Microsoft BitNet repo
    let possible_llama_locations = [
        cpp_dir.join("3rdparty/llama.cpp/include/llama.h"),
        cpp_dir.join("include/llama.h"),
        cpp_dir.join("src/llama.h"),
        cpp_dir.join("llama.h"),
    ];
    
    let mut llama_h = None;
    for location in &possible_llama_locations {
        if location.exists() {
            llama_h = Some(location.clone());
            break;
        }
    }
    
    let llama_h = llama_h.ok_or_else(|| {
        format!(
            "llama.h not found in any expected location:\n{}\n\
             Is the Microsoft BitNet repository complete?",
            possible_llama_locations.iter()
                .map(|p| format!("  - {}", p.display()))
                .collect::<Vec<_>>()
                .join("\n")
        )
    })?;

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
        .header(llama_h.to_string_lossy());
    
    // Add include paths - check which ones exist
    let possible_include_paths = [
        cpp_dir.join("3rdparty/llama.cpp/include"),
        cpp_dir.join("3rdparty/llama.cpp/ggml/include"),
        cpp_dir.join("include"),
        cpp_dir.join("src"),
        cpp_dir.clone(),
    ];
    
    for include_path in &possible_include_paths {
        if include_path.exists() {
            builder = builder.clang_arg(format!("-I{}", include_path.display()));
        }
    }
    
    builder = builder
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
