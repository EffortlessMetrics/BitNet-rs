// ANTI-PATTERN: Only checks feature="gpu", ignores feature="cuda" alias
// This is INVALID and will fail Issue #439 AC2 validation tests

fn main() {
    // WRONG: Only checks GPU feature, not CUDA alias
    if std::env::var_os("CARGO_FEATURE_GPU").is_some() {
        println!("cargo:rustc-cfg=gpu_compiled");
    }

    // This pattern breaks backward compatibility where users may still use:
    // cargo build --features cuda
    //
    // Expected unified pattern:
    // let gpu = CARGO_FEATURE_GPU.is_some() || CARGO_FEATURE_CUDA.is_some();
}
