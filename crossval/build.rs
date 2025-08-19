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
    cc::Build::new()
        .file("src/bitnet_cpp_wrapper.c")
        .compile("bitnet_cpp_wrapper");

    let root = env::var("BITNET_CPP_PATH")
        .unwrap_or_else(|_| format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap()));
    let lib = Path::new(&root).join("build").join("lib");

    // We always have the wrapper now
    println!("cargo:rustc-cfg=have_cpp");
    
    // Link to the actual BitNet library if available
    if lib.exists() {
        println!("cargo:rustc-link-search=native={}", lib.display());
        
        // Link to libbitnet.so if it exists (for real implementation)
        if lib.join("libbitnet.so").exists() {
            println!("cargo:rustc-link-lib=dylib=bitnet");
            println!("cargo:warning=bitnet-crossval: C++ library found and linked from {}", root);
            
            // Also link C++ standard library
            #[cfg(target_os = "linux")]
            println!("cargo:rustc-link-lib=dylib=stdc++");
            #[cfg(target_os = "macos")]
            println!("cargo:rustc-link-lib=dylib=c++");
        } else {
            println!("cargo:warning=bitnet-crossval: Using mock C wrapper (real BitNet library not found)");
        }
    } else {
        println!("cargo:warning=bitnet-crossval: Using mock C wrapper (BitNet library path not found)");
    }
}