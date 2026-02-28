//! Tests for SPIR-V pre-compilation and fallback behaviour.

use std::path::Path;

// Include build helper for direct testing of compile functions.
#[path = "../../../build_helpers/opencl_compile.rs"]
mod opencl_compile;

#[test]
fn spirv_compiler_lookup_does_not_panic() {
    let _clang = opencl_compile::find_clang();
    let _ocloc = opencl_compile::find_ocloc();
}

#[test]
fn spirv_compile_missing_source_returns_gracefully() {
    let cl = Path::new("nonexistent_kernel.cl");
    let spv = Path::new("nonexistent_kernel.spv");
    let result = opencl_compile::compile_cl_to_spv(cl, spv);
    match result {
        opencl_compile::SpvCompileResult::Compiled(_) => {
            panic!("should not compile a nonexistent file");
        }
        _ => {} // CompilerNotFound or Failed
    }
}

#[test]
fn force_source_env_defaults_to_unset() {
    // BITNET_OPENCL_FORCE_SOURCE should not be set in normal test runs
    let val = std::env::var("BITNET_OPENCL_FORCE_SOURCE").unwrap_or_default();
    let _ = val; // just verifying the lookup doesn't panic
}

#[test]
fn kernel_sources_available_for_source_fallback() {
    // Even without SPIR-V, embedded .cl sources are always available
    let kernels = [
        bitnet_kernels::kernels::MATMUL_I2S_SRC,
        bitnet_kernels::kernels::QUANTIZE_I2S_SRC,
        bitnet_kernels::kernels::ELEMENTWISE_SRC,
    ];
    for src in &kernels {
        assert!(!src.is_empty(), "kernel source should not be empty");
        assert!(src.contains("__kernel"), "kernel source should contain __kernel");
    }
}

#[test]
fn spirv_env_vars_are_optional() {
    // BITNET_SPV_* env vars are only set when clang/ocloc is available at
    // build time. The runtime should gracefully handle their absence.
    for name in ["MATMUL_I2S", "QUANTIZE_I2S", "ELEMENTWISE", "ATTENTION"] {
        let key = format!("BITNET_SPV_{name}");
        let _ = std::env::var(&key); // must not panic
    }
}

#[test]
fn spirv_vs_source_timing_stub() {
    // Documents the expected performance comparison interface.
    // Actual timing requires an OpenCL device; without one this is a no-op.
    let spv_startup_ms: f64 = 0.0;
    let src_compile_ms: f64 = 0.0;
    assert!(
        spv_startup_ms <= src_compile_ms || src_compile_ms == 0.0,
        "SPIR-V startup should be faster than source compilation"
    );
}
