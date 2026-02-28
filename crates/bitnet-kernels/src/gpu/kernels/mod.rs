//! OpenCL kernel source files for Intel Arc GPU compute.
//!
//! Kernel sources are embedded at compile time via `include_str!` and
//! compiled to OpenCL programs at runtime via `clCreateProgramWithSource`.

/// I2S matrix multiplication kernel source (naive, used as fallback).
pub const MATMUL_I2S_SRC: &str = include_str!("matmul_i2s.cl");

/// Tiled I2S matrix multiplication kernel source (Intel Arc optimized).
pub const MATMUL_I2S_TILED_SRC: &str = include_str!("matmul_i2s_tiled.cl");

/// I2S quantization kernel source.
pub const QUANTIZE_I2S_SRC: &str = include_str!("quantize_i2s.cl");

/// Element-wise operation kernels source.
pub const ELEMENTWISE_SRC: &str = include_str!("elementwise.cl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sources_are_not_empty() {
        assert!(!MATMUL_I2S_SRC.is_empty(), "matmul_i2s.cl should not be empty");
        assert!(!MATMUL_I2S_TILED_SRC.is_empty(), "matmul_i2s_tiled.cl should not be empty");
        assert!(!QUANTIZE_I2S_SRC.is_empty(), "quantize_i2s.cl should not be empty");
        assert!(!ELEMENTWISE_SRC.is_empty(), "elementwise.cl should not be empty");
    }

    #[test]
    fn kernel_sources_contain_kernel_keyword() {
        assert!(MATMUL_I2S_SRC.contains("__kernel"), "matmul_i2s.cl missing __kernel");
        assert!(
            MATMUL_I2S_TILED_SRC.contains("__kernel"),
            "matmul_i2s_tiled.cl missing __kernel"
        );
        assert!(QUANTIZE_I2S_SRC.contains("__kernel"), "quantize_i2s.cl missing __kernel");
        assert!(ELEMENTWISE_SRC.contains("__kernel"), "elementwise.cl missing __kernel");
    }

    #[test]
    fn matmul_kernel_has_correct_function_name() {
        assert!(MATMUL_I2S_SRC.contains("matmul_i2s"), "kernel function name mismatch");
    }

    #[test]
    fn quantize_kernel_has_correct_function_name() {
        assert!(QUANTIZE_I2S_SRC.contains("quantize_i2s"), "kernel function name mismatch");
    }

    #[test]
    fn elementwise_kernels_have_expected_functions() {
        assert!(ELEMENTWISE_SRC.contains("vec_add"), "missing vec_add kernel");
        assert!(ELEMENTWISE_SRC.contains("rms_norm"), "missing rms_norm kernel");
        assert!(ELEMENTWISE_SRC.contains("silu"), "missing silu kernel");
        assert!(ELEMENTWISE_SRC.contains("scale"), "missing scale kernel");
    }

    #[test]
    fn tiled_matmul_kernel_has_expected_functions() {
        assert!(
            MATMUL_I2S_TILED_SRC.contains("matmul_i2s_tiled"),
            "missing matmul_i2s_tiled kernel"
        );
        assert!(
            MATMUL_I2S_TILED_SRC.contains("matmul_i2s_tiled_vec4"),
            "missing matmul_i2s_tiled_vec4 kernel"
        );
    }

    #[test]
    fn tiled_matmul_kernel_has_configurable_tile_size() {
        assert!(MATMUL_I2S_TILED_SRC.contains("TILE_SIZE"), "missing TILE_SIZE define");
        assert!(MATMUL_I2S_TILED_SRC.contains("LOCAL_SIZE_X"), "missing LOCAL_SIZE_X define");
        assert!(MATMUL_I2S_TILED_SRC.contains("LOCAL_SIZE_Y"), "missing LOCAL_SIZE_Y define");
    }

    #[test]
    fn tiled_matmul_kernel_uses_local_memory() {
        assert!(
            MATMUL_I2S_TILED_SRC.contains("__local"),
            "tiled kernel should use __local memory"
        );
        assert!(
            MATMUL_I2S_TILED_SRC.contains("barrier(CLK_LOCAL_MEM_FENCE)"),
            "tiled kernel should have local memory barriers"
        );
    }

    #[test]
    fn tiled_matmul_kernel_uses_branchless_decode() {
        assert!(
            MATMUL_I2S_TILED_SRC.contains("decode_ternary"),
            "tiled kernel should use branchless decode_ternary helper"
        );
    }
}
