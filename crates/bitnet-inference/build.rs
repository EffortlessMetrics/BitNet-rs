//! Build script for bitnet-inference crate
//!
//! Detects whether bitnet-sys is in stub mode (no C++ libraries available)
//! and propagates the `bitnet_sys_stub` cfg flag to this crate, mirroring
//! the detection in `bitnet-sys/build.rs`.

fn main() {
    // Declare the cfg key so rustc doesn't warn about unknown cfg values
    // when #[cfg(not(bitnet_sys_stub))] gates are used in this crate.
    println!("cargo::rustc-check-cfg=cfg(bitnet_sys_stub)");

    println!("cargo:rerun-if-env-changed=BITNET_CPP_DIR");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    // Only detect stub mode when FFI feature is enabled.
    if std::env::var("CARGO_FEATURE_FFI").is_err() {
        return;
    }

    // Mirror the stub detection from bitnet-sys/build.rs so both crates agree.
    let explicit_dir = std::env::var("BITNET_CPP_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::env::var("BITNET_CPP_PATH")
                .ok()
                .filter(|s| !s.is_empty())
        });

    let cpp_dir = explicit_dir
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| std::path::PathBuf::from(format!("{}/.cache/bitnet_cpp", h)))
                .filter(|d| d.exists())
        });

    let in_stub_mode = !matches!(cpp_dir, Some(d) if d.exists());

    if in_stub_mode {
        println!("cargo:rustc-cfg=bitnet_sys_stub");
    }
}
