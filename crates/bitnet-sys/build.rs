//! Build script for bitnet-sys crate
//!
//! This script links against the Microsoft BitNet C++ implementation when
//! the `ffi` feature is enabled. It fails fast if dependencies are missing.

use std::env;
use std::path::Path;
#[cfg(feature = "ffi")]
use std::path::PathBuf;

fn main() {
    // Always declare cfg keys so rustc doesn't warn about unknown cfg values.
    println!("cargo::rustc-check-cfg=cfg(bitnet_cpp_available)");
    println!("cargo::rustc-check-cfg=cfg(bitnet_cpp_has_cuda)");
    println!("cargo::rustc-check-cfg=cfg(bitnet_cpp_has_bitnet_shim)");

    // If the crate is compiled without `--features bitnet-sys/ffi`,
    // skip all native build steps so the workspace remains green.
    if env::var("CARGO_FEATURE_FFI").is_err() {
        // No native build needed when FFI is disabled, but still ensure the build
        // script reruns if relevant inputs change.
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-env-changed=BITNET_CPP_DIR");
        println!("cargo:rerun-if-env-changed=BITNET_CPP_PATH");
        println!("cargo::rustc-check-cfg=cfg(bitnet_sys_stub)");
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_DIR");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_PATH"); // Legacy support
    // Declare the cfg key so rustc doesn't warn about unknown cfg values
    println!("cargo::rustc-check-cfg=cfg(bitnet_sys_stub)");

    #[cfg(feature = "ffi")]
    {
        // When the ffi feature is enabled, try to find the C++ implementation.
        // Empty BITNET_CPP_DIR ("") is treated as unset (stub mode).
        let explicit_dir = env::var("BITNET_CPP_DIR")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| env::var("BITNET_CPP_PATH").ok().filter(|s| !s.is_empty()));

        let cpp_dir = explicit_dir.map(PathBuf::from).or_else(|| {
            env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(format!("{}/.cache/bitnet_cpp", h)))
                .filter(|d| d.exists())
        });

        // If no C++ directory is available, enter stub mode: compile succeeds but
        // runtime calls will return errors. Emit cfg flag so lib.rs can gate real
        // bindings/wrappers behind #[cfg(not(bitnet_sys_stub))].
        let cpp_dir = match cpp_dir {
            Some(d) if d.exists() => d,
            _ => {
                let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set by cargo");
                std::fs::write(
                    Path::new(&out_dir).join("bindings.rs"),
                    "// stub mode: C++ libraries not available\n",
                )
                .expect("bitnet-sys: failed to write stub bindings.rs");
                println!("cargo:rustc-cfg=bitnet_sys_stub");
                eprintln!(
                    "bitnet-sys: STUB mode — compiling without C++ libraries. \
                     Set BITNET_CPP_DIR to enable real cross-validation."
                );
                return;
            }
        };

        // Verify the C++ implementation is built
        let build_dir = cpp_dir.join("build");
        if !build_dir.exists() {
            panic!(
                "bitnet-sys: BitNet C++ not built. Build directory missing: {}\n\
                 Run: ./ci/fetch_bitnet_cpp.sh",
                build_dir.display()
            );
        }

        eprintln!("bitnet-sys: Building with cross-validation support");
        eprintln!("bitnet-sys: Using BitNet C++ from: {}", cpp_dir.display());

        // Symbol analysis: emit cfg flags for detected capabilities.
        // Only runs when the `symbol-analysis` feature is explicitly requested.
        if std::env::var("CARGO_FEATURE_SYMBOL_ANALYSIS").is_ok() {
            run_symbol_analysis(&cpp_dir);
        }

        // Link against the C++ implementation - fail on error
        link_cpp_implementation(&cpp_dir).expect("Failed to link Microsoft BitNet C++ libraries");

        // Compile the C++ shim
        compile_cpp_shim(&cpp_dir).expect("Failed to compile C++ shim");

        // Generate bindings - fail on error
        generate_bindings(&cpp_dir)
            .expect("Failed to generate FFI bindings from Microsoft BitNet headers");
    }
}

#[cfg(feature = "ffi")]
fn link_cpp_implementation(cpp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let build_dir = cpp_dir.join("build");

    // Library search paths - order matters!
    // Support both BitNet.cpp (with 3rdparty/llama.cpp) and standalone llama.cpp
    let lib_search_paths = [
        // Standalone llama.cpp paths
        build_dir.join("bin"), // llama.cpp puts libraries in build/bin/
        // BitNet.cpp embedded llama.cpp paths
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
        println!("cargo:rustc-link-lib=dylib=gomp");
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
fn compile_cpp_shim(cpp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;

    let shim_cc = PathBuf::from("csrc/bitnet_c_shim.cc");

    if !shim_cc.exists() {
        return Err(format!(
            "C++ shim source not found: {}\n\
             Expected location: crates/bitnet-sys/csrc/bitnet_c_shim.cc",
            shim_cc.display()
        )
        .into());
    }

    eprintln!("bitnet-sys: Compiling C++ shim from {}", shim_cc.display());

    let build_dir = cpp_dir.join("build");

    // Local include directories (use -I, warnings visible)
    let local_includes = vec![
        PathBuf::from("include"), // Local bitnet_c.h
    ];

    // System include directories (use -isystem, warnings suppressed)
    // TODO: llama.cpp API version detection
    // These paths assume a specific llama.cpp directory structure.
    // If llama.cpp reorganizes headers, paths may need adjustment.
    let system_includes = vec![
        // Standalone llama.cpp paths
        cpp_dir.join("include"),      // llama.h
        cpp_dir.join("ggml/include"), // ggml.h
        // BitNet.cpp embedded llama.cpp paths
        cpp_dir.join("3rdparty/llama.cpp/include"),
        cpp_dir.join("3rdparty/llama.cpp/ggml/include"),
        cpp_dir.join("src"),
        build_dir.join("3rdparty/llama.cpp/include"),
        build_dir.join("3rdparty/llama.cpp/ggml/include"),
    ];

    // Use unified compile_cpp_shim from xtask-build-helper
    xtask_build_helper::compile_cpp_shim(
        &shim_cc,
        "bitnet_c_shim",
        &local_includes,
        &system_includes,
    )?;

    eprintln!("bitnet-sys: C++ shim compiled successfully");
    Ok(())
}

#[cfg(feature = "ffi")]
fn generate_bindings(cpp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // We'll generate bindings from both our custom bitnet_c.h wrapper and llama.h
    let bitnet_c_h = PathBuf::from("include/bitnet_c.h");

    if !bitnet_c_h.exists() {
        return Err(format!(
            "bitnet_c.h not found at: {}\n\
             Expected location: crates/bitnet-sys/include/bitnet_c.h",
            bitnet_c_h.display()
        )
        .into());
    }

    // Try to find llama.h as well
    let possible_llama_locations = [
        // Standalone llama.cpp path (most common)
        cpp_dir.join("include/llama.h"),
        // BitNet.cpp embedded llama.cpp paths
        cpp_dir.join("3rdparty/llama.cpp/include/llama.h"),
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
            possible_llama_locations
                .iter()
                .map(|p| format!("  - {}", p.display()))
                .collect::<Vec<_>>()
                .join("\n")
        )
    })?;

    eprintln!(
        "bitnet-sys: Generating bindings from {} and {}",
        bitnet_c_h.display(),
        llama_h.display()
    );

    let mut builder = bindgen::Builder::default()
        .header(bitnet_c_h.to_string_lossy())
        .header(llama_h.to_string_lossy());

    // Add include paths - check which ones exist
    let possible_include_paths = [
        // Standalone llama.cpp paths
        cpp_dir.join("include"),
        cpp_dir.join("ggml/include"),
        // BitNet.cpp embedded llama.cpp paths
        cpp_dir.join("3rdparty/llama.cpp/include"),
        cpp_dir.join("3rdparty/llama.cpp/ggml/include"),
        cpp_dir.join("src"),
        cpp_dir.to_path_buf(),
    ];

    for include_path in &possible_include_paths {
        if include_path.exists() {
            builder = builder.clang_arg(format!("-I{}", include_path.display()));
        }
    }

    builder = builder
        // BitNet C wrapper API
        .allowlist_function("bitnet_.*")
        .allowlist_type("bitnet_.*")
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
        // For Rust 2024 compatibility
        .raw_line("// Auto-generated bindings - DO NOT EDIT")
        // Note: derive_copy automatically includes Clone
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    let bindings = builder.generate().map_err(|e| format!("bindgen failed: {}", e))?;

    let out_path = PathBuf::from(env::var("OUT_DIR")?);
    let bindings_path = out_path.join("bindings.rs");

    // Write initial bindings
    bindings.write_to_file(&bindings_path)?;

    // Post-process to add unsafe to extern blocks for Rust 2024
    // Only add unsafe if not already present (bindgen 0.69.5+ already adds it)
    let bindings_content = std::fs::read_to_string(&bindings_path)?;
    let fixed_content = bindings_content.replace("extern \"C\" {", "unsafe extern \"C\" {");
    // Remove duplicate unsafe keywords that may have been created
    let fixed_content =
        fixed_content.replace("unsafe unsafe extern \"C\" {", "unsafe extern \"C\" {");
    std::fs::write(&bindings_path, fixed_content)?;

    eprintln!("bitnet-sys: Generated C++ bindings successfully");
    Ok(())
}

/// Inspect shared libraries in `cpp_dir/build/` using `nm` or `objdump` and
/// emit `cargo:rustc-cfg` flags for detected capabilities.
///
/// Always emits `bitnet_cpp_available`. Conditionally emits `bitnet_cpp_has_cuda`
/// and `bitnet_cpp_has_bitnet_shim` based on symbol presence.
#[allow(dead_code)]
fn run_symbol_analysis(cpp_dir: &Path) {
    let build_dir = cpp_dir.join("build");
    if !build_dir.exists() {
        return;
    }

    // Walk build dir up to depth 3, collect .so and .dylib files.
    let mut libraries: Vec<std::path::PathBuf> = Vec::new();
    collect_libraries(&build_dir, 0, 3, &mut libraries);

    if libraries.is_empty() {
        eprintln!(
            "bitnet-sys: symbol-analysis: no shared libraries found in {}",
            build_dir.display()
        );
        // cpp_dir exists but no analyzable libs found; do NOT emit bitnet_cpp_available.
        return;
    }

    let mut has_cuda = false;
    let mut has_shim = false;

    for lib in &libraries {
        eprintln!("bitnet-sys: symbol-analysis: inspecting {}", lib.display());
        if let Some((cuda, shim)) = analyze_library_symbols(lib) {
            if cuda {
                has_cuda = true;
            }
            if shim {
                has_shim = true;
            }
        }
    }

    println!("cargo:rustc-cfg=bitnet_cpp_available");
    if has_cuda {
        eprintln!("bitnet-sys: symbol-analysis: CUDA symbols detected");
        println!("cargo:rustc-cfg=bitnet_cpp_has_cuda");
    }
    if has_shim {
        eprintln!("bitnet-sys: symbol-analysis: BitNet shim symbols detected");
        println!("cargo:rustc-cfg=bitnet_cpp_has_bitnet_shim");
    }
}

/// Recursively collect `.so` and `.dylib` files up to `max_depth`.
#[allow(dead_code)]
fn collect_libraries(
    dir: &Path,
    depth: usize,
    max_depth: usize,
    out: &mut Vec<std::path::PathBuf>,
) {
    if depth > max_depth {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_libraries(&path, depth + 1, max_depth, out);
        } else if let Some(ext) = path.extension()
            && (ext == "so" || ext == "dylib" || ext == "dll")
        {
            out.push(path);
        }
    }
}

/// Run `nm --dynamic --defined-only` (falling back to `objdump -T`) on `lib`
/// and return `(has_cuda, has_bitnet_shim)`. Returns `None` if both tools fail.
#[allow(dead_code)]
fn analyze_library_symbols(lib: &Path) -> Option<(bool, bool)> {
    let output = std::process::Command::new("nm")
        .args(["--dynamic", "--defined-only"])
        .arg(lib)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| o.stdout)
        .or_else(|| {
            std::process::Command::new("objdump")
                .args(["-T"])
                .arg(lib)
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| o.stdout)
        })?;

    let text = String::from_utf8_lossy(&output);
    let has_cuda = text.lines().any(|l| {
        let l = l.to_ascii_lowercase();
        l.contains("cuda") || l.contains("cublas") || l.contains("cudarc")
    });
    let has_shim = text.lines().any(|l| {
        l.contains("bitnet_eval")
            || l.contains("bitnet_init")
            || l.contains("bitnet_create_context")
    });
    Some((has_cuda, has_shim))
}
