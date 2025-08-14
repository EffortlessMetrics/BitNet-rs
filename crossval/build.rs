//! Build script for cross-validation crate
//!
//! This script handles conditional compilation of C++ bindings when the
//! crossval feature is enabled.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_DIR");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_PATH");

    // Only build C++ bindings if crossval feature is enabled
    if !cfg!(feature = "crossval") {
        return;
    }

    println!("cargo:warning=Building with cross-validation support");

    // Check if we have access to the C++ implementation
    // Support both BITNET_CPP_DIR and BITNET_CPP_PATH for consistency
    let cpp_path = env::var("BITNET_CPP_DIR")
        .or_else(|_| env::var("BITNET_CPP_PATH"))
        .unwrap_or_else(|_| {
            // Default path where ci/fetch_bitnet_cpp.sh places the code
            let home = env::var("HOME")
                .or_else(|_| env::var("USERPROFILE"))
                .expect("Could not determine home directory");
            format!("{}/.cache/bitnet_cpp", home)
        });

    let cpp_path = PathBuf::from(cpp_path);

    if !cpp_path.exists() {
        println!(
            "cargo:warning=BitNet C++ implementation not found at {:?}",
            cpp_path
        );
        println!("cargo:warning=Run ci/fetch_bitnet_cpp.sh to download it");
        println!("cargo:warning=Cross-validation will be disabled");
        return;
    }

    // Set up include paths
    let include_path = cpp_path.join("include");
    if include_path.exists() {
        println!("cargo:include={}", include_path.display());
    }

    // Look for built libraries
    let lib_path = cpp_path.join("build").join("lib");
    if lib_path.exists() {
        println!("cargo:rustc-link-search=native={}", lib_path.display());
        println!("cargo:rustc-link-lib=static=bitnet");
    } else {
        println!(
            "cargo:warning=BitNet C++ library not found at {:?}",
            lib_path
        );
        println!("cargo:warning=Make sure to build the C++ implementation first");
    }

    // Generate bindings if bindgen is available
    #[cfg(feature = "crossval")]
    generate_bindings(&cpp_path);
}

#[cfg(feature = "crossval")]
fn generate_bindings(cpp_path: &PathBuf) {
    let header_path = cpp_path.join("include").join("bitnet.h");

    if !header_path.exists() {
        println!("cargo:warning=BitNet header not found at {:?}", header_path);
        return;
    }

    // Check if clang is available
    if !has_clang() {
        println!("cargo:warning=clang not found - cannot generate bindings");
        println!("cargo:warning=Install clang to enable cross-validation");
        println!("cargo:warning=  Ubuntu/Debian: apt install clang");
        println!("cargo:warning=  macOS: xcode-select --install");
        println!("cargo:warning=  Windows: Install LLVM from https://llvm.org/");
        return;
    }

    let bindings = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        .clang_arg(format!("-I{}", cpp_path.join("include").display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    println!("cargo:warning=Generated C++ bindings successfully");
}

#[cfg(feature = "crossval")]
fn has_clang() -> bool {
    std::process::Command::new("clang")
        .arg("--version")
        .output()
        .is_ok()
}

#[cfg(not(feature = "crossval"))]
fn generate_bindings(_cpp_path: &PathBuf) {
    // No-op when crossval feature is disabled
}
