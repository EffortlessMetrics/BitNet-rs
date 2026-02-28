//! Tests for the OpenCL kernel stub under the `oneapi` feature.

use bitnet_kernels::gpu::opencl::OpenClKernel;

#[test]
fn opencl_kernel_new_returns_error() {
    let result = OpenClKernel::new();
    assert!(result.is_err(), "OpenClKernel::new() should return Err while unimplemented");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("not yet implemented"),
        "error should mention 'not yet implemented', got: {err_msg}"
    );
}

#[test]
fn oneapi_feature_compiles() {
    // If this test compiles and runs, the oneapi feature wiring is correct.
    assert_eq!(std::mem::size_of::<OpenClKernel>(), std::mem::size_of::<()>());
}
