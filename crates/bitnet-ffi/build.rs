use std::env;
use std::path::PathBuf;

fn main() {
    // Generate pkg-config file
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Tell cargo to link with required libraries
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../include/llama_compat.h");
    
    // Copy header to output
    let include_dir = out_dir.join("include");
    std::fs::create_dir_all(&include_dir).unwrap();
    
    // Copy our compatibility header
    std::fs::copy(
        "../../include/llama_compat.h",
        include_dir.join("llama_compat.h")
    ).unwrap();
    
    // Generate pkg-config file for C/C++ users
    let pkg_config = format!(
        r#"prefix={}
exec_prefix=${{prefix}}
includedir=${{prefix}}/include
libdir=${{exec_prefix}}/lib

Name: bitnet-ffi
Description: BitNet.rs FFI bindings with llama.cpp compatibility
Version: {}
Cflags: -I${{includedir}}
Libs: -L${{libdir}} -lbitnet_ffi
"#,
        out_dir.display(),
        env!("CARGO_PKG_VERSION")
    );
    
    std::fs::write(out_dir.join("bitnet.pc"), pkg_config).unwrap();
    
    // Print installation instructions
    if env::var("PROFILE").unwrap() == "release" {
        eprintln!("╭─────────────────────────────────────────────────────────────╮");
        eprintln!("│ BitNet.rs FFI Library Built Successfully!                  │");
        eprintln!("├─────────────────────────────────────────────────────────────┤");
        eprintln!("│ To use with C/C++:                                         │");
        eprintln!("│                                                             │");
        eprintln!("│ 1. Copy header:                                            │");
        eprintln!("│    cp include/llama_compat.h /usr/local/include/           │");
        eprintln!("│                                                             │");
        eprintln!("│ 2. Copy library:                                           │");
        eprintln!("│    cp target/release/libbitnet_ffi.a /usr/local/lib/       │");
        eprintln!("│                                                             │");
        eprintln!("│ 3. Compile your code:                                      │");
        eprintln!("│    gcc your_code.c -lbitnet_ffi -o your_app               │");
        eprintln!("│                                                             │");
        eprintln!("│ That's it! Your llama.cpp code now uses BitNet.rs!        │");
        eprintln!("╰─────────────────────────────────────────────────────────────╯");
    }
}