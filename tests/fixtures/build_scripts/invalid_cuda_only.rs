// ANTI-PATTERN: Only checks feature="cuda", ignores feature="gpu"
// This is INVALID and will fail Issue #439 AC2 validation tests

fn main() {
    // WRONG: Only checks CUDA feature, not GPU feature
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        println!("cargo:rustc-cfg=gpu_compiled");
    }

    // This pattern causes compile-time drift where:
    // - cargo build --features gpu → GPU NOT compiled
    // - cargo build --features cuda → GPU compiled
    //
    // Expected unified pattern:
    // let gpu = CARGO_FEATURE_GPU.is_some() || CARGO_FEATURE_CUDA.is_some();
}
