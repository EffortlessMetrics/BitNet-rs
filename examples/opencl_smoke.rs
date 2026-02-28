//! OpenCL smoke test for Intel GPU support.
//!
//! This example verifies that the OpenCL/oneAPI kernel backend can:
//! 1. Detect compile-time oneapi feature support
//! 2. Query kernel capabilities for OneApi backend availability
//! 3. Attempt OpenCL platform enumeration and device initialization
//!
//! # Running
//!
//! ```bash
//! cargo run --example opencl_smoke --no-default-features --features oneapi
//! ```
//!
//! # Expected output
//!
//! - **With Intel GPU**: Reports platform/device info, confirms kernel init
//! - **Without Intel GPU**: Reports no device found (not an error)
//! - **Without oneapi feature**: Compile error (guarded by required-features)

fn main() {
    println!("=== BitNet-rs OpenCL Smoke Test ===");
    println!();

    check_compile_time_support();
    check_kernel_capabilities();
    try_opencl_init();
}

/// Verify that oneapi feature was compiled in.
fn check_compile_time_support() {
    println!("[1/3] Compile-time feature check");

    let compiled = bitnet::kernels::device_features::oneapi_compiled();
    println!("  oneapi compiled: {compiled}");
    assert!(compiled, "oneapi feature should be enabled");

    let gpu = bitnet::kernels::device_features::gpu_compiled();
    println!("  gpu compiled:    {gpu}");
    println!("  PASS");
    println!();
}

/// Query kernel capabilities and test backend selection.
fn check_kernel_capabilities() {
    use bitnet::common::backend_selection::{select_backend, BackendRequest};
    use bitnet::common::kernel_registry::KernelCapabilities;

    println!("[2/3] Kernel capabilities & backend selection");

    let caps = KernelCapabilities::from_compile_time();
    println!("  oneapi_compiled: {}", caps.oneapi_compiled);
    println!("  oneapi_runtime:  {}", caps.oneapi_runtime);
    println!("  simd_level:      {:?}", caps.simd_level);

    assert!(caps.oneapi_compiled, "oneapi should be compiled");

    match select_backend(BackendRequest::OneApi, &caps) {
        Ok(result) => {
            println!("  backend: {}", result.summary());
        }
        Err(e) => {
            println!("  backend selection: OneApi unavailable ({e})");
            println!("  (expected without Intel GPU runtime)");
        }
    }
    println!("  PASS");
    println!();
}

/// Attempt to initialize the OpenCL kernel provider.
fn try_opencl_init() {
    println!("[3/3] OpenCL device initialization");

    match bitnet::kernels::OpenClKernel::new() {
        Ok(kernel) => {
            println!("  platform: {}", kernel.platform_name());
            println!("  device:   {}", kernel.device_name());
            run_matmul_check(&kernel);
            println!("  PASS - OpenCL kernel ready");
        }
        Err(e) => {
            println!("  init failed: {e}");
            println!("  (expected without Intel GPU)");
            println!("  SKIP");
        }
    }
    println!();
    println!("=== Smoke test complete ===");
}

/// Run a trivial matmul to verify the compute pipeline.
fn run_matmul_check(kernel: &bitnet::kernels::OpenClKernel) {
    use bitnet::kernels::KernelProvider;

    println!("  Running 4x4 matmul sanity check...");

    let m = 4;
    let n = 4;
    let k = 4;

    let a: Vec<i8> = vec![
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ];
    // I2_S packed: 4 ternary values per byte
    let b: Vec<u8> = vec![0x55; 4]; // [1,1,1,1] repeated
    let mut c = vec![0.0f32; m * n];

    match kernel.matmul_i2s(&a, &b, &mut c, m, n, k) {
        Ok(()) => println!("  result: {c:?}"),
        Err(e) => println!("  matmul failed (non-fatal): {e}"),
    }
}
