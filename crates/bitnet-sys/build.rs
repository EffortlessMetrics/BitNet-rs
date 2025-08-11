//! Build script for bitnet-sys crate
//!
//! This script handles conditional compilation of C++ bindings when the
//! crossval feature is enabled, with helpful error messages when dependencies
//! are missing.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_PATH");
    println!("cargo:rerun-if-feature-changed=crossval");
    
    // Only build C++ bindings if crossval feature is enabled
    if !cfg!(feature = "crossval") {
        println!("cargo:warning=bitnet-sys: crossval feature disabled, skipping C++ bindings");
        return;
    }
    
    println!("cargo:warning=bitnet-sys: Building with cross-validation support");
    
    // Check if we have access to the C++ implementation
    let cpp_path = env::var("BITNET_CPP_PATH")
        .unwrap_or_else(|_| {
            // Default path where ci/fetch_bitnet_cpp.sh places the code
            let home = env::var("HOME")
                .or_else(|_| env::var("USERPROFILE"))
                .expect("Could not determine home directory");
            format!("{}/.cache/bitnet_cpp", home)
        });
    
    let cpp_path = PathBuf::from(cpp_path);
    
    if !cpp_path.exists() {
        print_cpp_not_found_error(&cpp_path);
        return;
    }
    
    // Check for clang availability
    if !has_clang() {
        print_clang_not_found_error();
        return;
    }
    
    // Set up include and library paths
    setup_cpp_paths(&cpp_path);
    
    // Generate bindings if possible
    if let Err(e) = generate_bindings(&cpp_path) {
        println!("cargo:warning=bitnet-sys: Failed to generate bindings: {}", e);
        println!("cargo:warning=bitnet-sys: Cross-validation will be limited");
        // Create a stub bindings file so compilation doesn't fail
        create_stub_bindings();
    }
}

fn print_cpp_not_found_error(cpp_path: &PathBuf) {
    println!("cargo:warning=bitnet-sys: BitNet C++ implementation not found at {:?}", cpp_path);
    println!("cargo:warning=bitnet-sys: ");
    println!("cargo:warning=bitnet-sys: To enable cross-validation:");
    println!("cargo:warning=bitnet-sys:   1. Run: ./ci/fetch_bitnet_cpp.sh");
    println!("cargo:warning=bitnet-sys:   2. Or set BITNET_CPP_PATH to your BitNet.cpp location");
    println!("cargo:warning=bitnet-sys:   3. Then rebuild with: cargo build --features crossval");
    println!("cargo:warning=bitnet-sys: ");
    println!("cargo:warning=bitnet-sys: Cross-validation features will be disabled");
}

fn print_clang_not_found_error() {
    println!("cargo:warning=bitnet-sys: clang not found - cannot generate C++ bindings");
    println!("cargo:warning=bitnet-sys: ");
    println!("cargo:warning=bitnet-sys: To install clang:");
    
    if cfg!(target_os = "linux") {
        println!("cargo:warning=bitnet-sys:   Ubuntu/Debian: sudo apt install clang libclang-dev");
        println!("cargo:warning=bitnet-sys:   RHEL/CentOS: sudo yum install clang clang-devel");
        println!("cargo:warning=bitnet-sys:   Arch: sudo pacman -S clang");
    } else if cfg!(target_os = "macos") {
        println!("cargo:warning=bitnet-sys:   macOS: xcode-select --install");
        println!("cargo:warning=bitnet-sys:   Or: brew install llvm");
    } else if cfg!(target_os = "windows") {
        println!("cargo:warning=bitnet-sys:   Windows: Install LLVM from https://llvm.org/");
        println!("cargo:warning=bitnet-sys:   Or: winget install LLVM.LLVM");
    }
    
    println!("cargo:warning=bitnet-sys: ");
    println!("cargo:warning=bitnet-sys: Cross-validation features will be disabled");
}

fn has_clang() -> bool {
    Command::new("clang")
        .arg("--version")
        .output()
        .is_ok()
}

fn setup_cpp_paths(cpp_path: &PathBuf) {
    // Set up include paths
    let include_path = cpp_path.join("include");
    if include_path.exists() {
        println!("cargo:include={}", include_path.display());
        println!("cargo:rustc-env=BITNET_CPP_INCLUDE_PATH={}", include_path.display());
    }
    
    // Look for built libraries
    let lib_paths = [
        cpp_path.join("build").join("lib"),
        cpp_path.join("build").join("install").join("lib"),
        cpp_path.join("lib"),
    ];
    
    for lib_path in &lib_paths {
        if lib_path.exists() {
            println!("cargo:rustc-link-search=native={}", lib_path.display());
            println!("cargo:rustc-env=BITNET_CPP_LIB_PATH={}", lib_path.display());
            
            // Try to link common library names
            let lib_names = ["bitnet", "bitnet_cpp", "BitNet"];
            for lib_name in &lib_names {
                // Check if static library exists
                let static_lib = lib_path.join(format!("lib{}.a", lib_name));
                let static_lib_win = lib_path.join(format!("{}.lib", lib_name));
                
                if static_lib.exists() || static_lib_win.exists() {
                    println!("cargo:rustc-link-lib=static={}", lib_name);
                    println!("cargo:warning=bitnet-sys: Linked static library: {}", lib_name);
                    break;
                }
                
                // Check if shared library exists
                let shared_lib = lib_path.join(format!("lib{}.so", lib_name));
                let shared_lib_mac = lib_path.join(format!("lib{}.dylib", lib_name));
                let shared_lib_win = lib_path.join(format!("{}.dll", lib_name));
                
                if shared_lib.exists() || shared_lib_mac.exists() || shared_lib_win.exists() {
                    println!("cargo:rustc-link-lib=dylib={}", lib_name);
                    println!("cargo:warning=bitnet-sys: Linked shared library: {}", lib_name);
                    break;
                }
            }
            break;
        }
    }
}

#[cfg(feature = "crossval")]
fn generate_bindings(cpp_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let header_paths = [
        cpp_path.join("include").join("ggml-bitnet.h"),
        cpp_path.join("include").join("bitnet.h"),
        cpp_path.join("include").join("bitnet.hpp"),
        cpp_path.join("bitnet.h"),
        cpp_path.join("src").join("bitnet.h"),
    ];
    
    let header_path = header_paths
        .iter()
        .find(|p| p.exists())
        .ok_or("No BitNet header file found")?;
    
    println!("cargo:warning=bitnet-sys: Generating bindings from {:?}", header_path);
    
    let bindings = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        .clang_arg(format!("-I{}", cpp_path.join("include").display()))
        .clang_arg(format!("-I{}", cpp_path.join("3rdparty/llama.cpp/ggml/include").display()))
        .clang_arg(format!("-I{}", cpp_path.display()))
        // Generate bindings for GGML BitNet functions
        .allowlist_function("ggml_bitnet_.*")
        .allowlist_function("ggml_qgemm_lut")
        .allowlist_function("ggml_preprocessor")
        .allowlist_type("bitnet_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("GGML_.*")
        // Generate safe wrappers
        .derive_debug(true)
        .derive_default(true)
        .derive_copy(true)
        // Handle C++ specifics
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()?;
    
    let out_path = PathBuf::from(env::var("OUT_DIR")?);
    bindings.write_to_file(out_path.join("bindings.rs"))?;
    
    println!("cargo:warning=bitnet-sys: Generated C++ bindings successfully");
    Ok(())
}

#[cfg(not(feature = "crossval"))]
fn generate_bindings(_cpp_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Create empty bindings file when feature is disabled
    let out_path = PathBuf::from(env::var("OUT_DIR")?);
    std::fs::write(
        out_path.join("bindings.rs"),
        "// Bindings disabled - crossval feature not enabled\n",
    )?;
    Ok(())
}

fn create_stub_bindings() {
    let out_path = env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_path);
    let content = r#"
// Stub bindings - real C++ library not available
// These are placeholder types to allow compilation

#[repr(C)]
pub struct ggml_tensor {
    _private: [u8; 0],
}

#[repr(C)]
pub struct bitnet_tensor_extra {
    pub lut_scales_size: i32,
    pub BK: i32,
    pub n_tile_num: i32,
    pub qweights: *mut u8,
    pub scales: *mut f32,
}

// Placeholder functions
extern "C" {
    pub fn ggml_bitnet_init();
    pub fn ggml_bitnet_free();
}
"#;
    std::fs::write(out_path.join("bindings.rs"), content)
        .expect("Failed to write stub bindings");
    println!("cargo:warning=bitnet-sys: Created stub bindings for compilation");
}