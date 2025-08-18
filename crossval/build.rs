use std::{env, path::Path};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(have_cpp)");
    println!("cargo:rerun-if-changed=build.rs");
    
    let ffi_enabled = env::var_os("CARGO_FEATURE_FFI").is_some();
    if !ffi_enabled {
        return;
    }

    let root = env::var("BITNET_CPP_PATH")
        .unwrap_or_else(|_| format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap()));
    let inc = Path::new(&root).join("include");
    let lib = Path::new(&root).join("build").join("lib");

    let have_header = inc.join("bitnet.h").exists() || inc.join("ggml-bitnet.h").exists();
    let have_static = lib.join("libbitnet_cpp.a").exists() || lib.join("libbitnet.a").exists();
    let have_shared = lib.join("libbitnet_cpp.so").exists() || lib.join("libbitnet.so").exists();

    if have_header && (have_static || have_shared) {
        println!("cargo:rustc-cfg=have_cpp");
        println!("cargo:rustc-link-search=native={}", lib.display());
        if have_static {
            if lib.join("libbitnet_cpp.a").exists() {
                println!("cargo:rustc-link-lib=static=bitnet_cpp");
            } else {
                println!("cargo:rustc-link-lib=static=bitnet");
            }
        } else {
            if lib.join("libbitnet_cpp.so").exists() {
                println!("cargo:rustc-link-lib=dylib=bitnet_cpp");
            } else {
                println!("cargo:rustc-link-lib=dylib=bitnet");
            }
        }
        println!("cargo:warning=bitnet-crossval: C++ library found and linked from {}", root);
    } else {
        println!("cargo:warning=bitnet-crossval: ffi enabled but no C++ lib found; building with stubs (no link). Set BITNET_CPP_PATH to enable real crossval.");
        println!("cargo:warning=BitNet C++ library not found at {:?}", lib);
        println!("cargo:warning=Make sure to build the C++ implementation first");
        println!("cargo:warning=BitNet header not found at {:?}", inc.join("bitnet.h"));
    }
}