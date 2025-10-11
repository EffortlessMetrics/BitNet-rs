// Valid build.rs pattern for unified GPU feature detection
// Tests specification: docs/explanation/issue-439-spec.md#ac2-build-script-parity

fn main() {
    // Unified GPU detection: Check both feature="gpu" AND feature="cuda"
    let gpu = std::env::var_os("CARGO_FEATURE_GPU").is_some()
           || std::env::var_os("CARGO_FEATURE_CUDA").is_some();

    if gpu {
        println!("cargo:rustc-cfg=bitnet_build_gpu");
        println!("cargo:rustc-cfg=gpu_compiled");

        // Optional: Link CUDA libraries if available
        // This is a simplified example - real implementation may include
        // more sophisticated CUDA toolkit detection
    }

    // Rerun if Cargo.toml changes (for feature flag modifications)
    println!("cargo:rerun-if-changed=Cargo.toml");
}
