#![cfg(feature = "oneapi")]
//! Tests for the OpenCL kernel provider under the `oneapi` feature.

use bitnet_kernels::KernelProvider;
use bitnet_kernels::gpu::opencl::OpenClKernel;

#[test]
fn opencl_kernel_new_does_not_panic() {
    // Should either succeed (Intel GPU present) or return error gracefully
    let result = OpenClKernel::new();
    match result {
        Ok(kernel) => {
            assert!(kernel.is_available());
            assert_eq!(kernel.name(), "opencl-intel");
            println!("OpenCL kernel available: {}", kernel.device_name());
        }
        Err(e) => {
            println!("OpenCL not available (expected in CI): {}", e);
        }
    }
}

#[test]
fn opencl_kernel_name_is_correct() {
    if let Ok(kernel) = OpenClKernel::new() {
        assert_eq!(kernel.name(), "opencl-intel");
    }
}

#[test]
fn opencl_matmul_small_if_available() {
    let kernel = match OpenClKernel::new() {
        Ok(k) => k,
        Err(_) => return, // Skip if no GPU
    };

    // Small 2x2 matmul smoke test — just verify it doesn't panic
    let a: Vec<i8> = vec![1, 0, 0, 1]; // 2x2 identity-ish
    let b: Vec<u8> = vec![0b01_01, 0b01_01]; // packed 1s
    let mut c = vec![0.0f32; 4]; // 2x2 output

    let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
    if let Err(e) = result {
        println!("matmul_i2s error (may be expected): {}", e);
    }
}

#[test]
fn oneapi_feature_compiles() {
    // If this test compiles and runs, the oneapi feature wiring is correct.
    let _ = std::mem::size_of::<OpenClKernel>();
}
