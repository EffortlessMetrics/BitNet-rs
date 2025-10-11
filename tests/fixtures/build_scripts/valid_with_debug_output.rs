// Valid build.rs with debug output for troubleshooting
// Tests specification: docs/explanation/issue-439-spec.md#ac2-build-script-parity

fn main() {
    // Unified GPU detection with debug output
    let cargo_feature_gpu = std::env::var_os("CARGO_FEATURE_GPU").is_some();
    let cargo_feature_cuda = std::env::var_os("CARGO_FEATURE_CUDA").is_some();

    let gpu = cargo_feature_gpu || cargo_feature_cuda;

    // Debug output (only visible with --verbose builds)
    eprintln!("build.rs: CARGO_FEATURE_GPU={}", cargo_feature_gpu);
    eprintln!("build.rs: CARGO_FEATURE_CUDA={}", cargo_feature_cuda);
    eprintln!("build.rs: GPU compiled={}", gpu);

    if gpu {
        println!("cargo:rustc-cfg=bitnet_build_gpu");
        println!("cargo:rustc-cfg=gpu_compiled");
    }

    println!("cargo:rerun-if-changed=Cargo.toml");
}
