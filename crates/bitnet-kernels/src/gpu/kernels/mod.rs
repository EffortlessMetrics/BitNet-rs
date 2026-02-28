//! OpenCL kernel source files for Intel Arc GPU compute.
//!
//! Kernel sources are embedded at compile time via `include_str!` and
//! compiled to OpenCL programs at runtime via `clCreateProgramWithSource`.

/// I2S matrix multiplication kernel source.
pub const MATMUL_I2S_SRC: &str = include_str!("matmul_i2s.cl");

/// I2S quantization kernel source.
pub const QUANTIZE_I2S_SRC: &str = include_str!("quantize_i2s.cl");

/// Element-wise operation kernels source.
pub const ELEMENTWISE_SRC: &str = include_str!("elementwise.cl");
pub const EMBEDDING_SRC: &str = include_str!("embedding.cl");

/// Tiled I2S matrix multiplication kernel source (local-memory + float4).
pub const MATMUL_I2S_TILED_SRC: &str = include_str!("matmul_i2s_tiled.cl");

/// KV cache management kernels source.
pub const KV_CACHE_SRC: &str = include_str!("kv_cache.cl");
pub const LINEAR_SRC: &str = include_str!("linear.cl");

/// FP16 matrix multiplication kernel source (cl_khr_fp16).
pub const MATMUL_FP16_SRC: &str = include_str!("matmul_fp16.cl");

/// INT8 matrix multiplication kernel source.
pub const MATMUL_INT8_SRC: &str = include_str!("matmul_int8.cl");
pub const SILU_GATE_SRC: &str = include_str!("silu_gate.cl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sources_are_not_empty() {
        assert!(!MATMUL_I2S_SRC.is_empty(), "matmul_i2s.cl should not be empty");
        assert!(!QUANTIZE_I2S_SRC.is_empty(), "quantize_i2s.cl should not be empty");
        assert!(!ELEMENTWISE_SRC.is_empty(), "elementwise.cl should not be empty");
        assert!(!MATMUL_I2S_TILED_SRC.is_empty(), "matmul_i2s_tiled.cl should not be empty");
        assert!(!MATMUL_FP16_SRC.is_empty(), "matmul_fp16.cl should not be empty");
        assert!(!MATMUL_INT8_SRC.is_empty(), "matmul_int8.cl should not be empty");
    }

    #[test]
    fn kernel_sources_contain_kernel_keyword() {
        assert!(MATMUL_I2S_SRC.contains("__kernel"), "matmul_i2s.cl missing __kernel");
        assert!(QUANTIZE_I2S_SRC.contains("__kernel"), "quantize_i2s.cl missing __kernel");
        assert!(ELEMENTWISE_SRC.contains("__kernel"), "elementwise.cl missing __kernel");
        assert!(MATMUL_I2S_TILED_SRC.contains("__kernel"), "matmul_i2s_tiled.cl missing __kernel");
        assert!(MATMUL_FP16_SRC.contains("__kernel"), "matmul_fp16.cl missing __kernel");
        assert!(MATMUL_INT8_SRC.contains("__kernel"), "matmul_int8.cl missing __kernel");
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
    fn tiled_matmul_has_correct_function_names() {
        assert!(
            MATMUL_I2S_TILED_SRC.contains("matmul_i2s_tiled"),
            "missing matmul_i2s_tiled kernel"
        );
    }

    #[test]
    fn tiled_matmul_uses_local_memory() {
        assert!(MATMUL_I2S_TILED_SRC.contains("__local"), "tiled kernel should use local memory");
    }

    #[test]
    fn tiled_matmul_uses_work_groups() {
        assert!(
            MATMUL_I2S_TILED_SRC.contains("get_local_id"),
            "tiled kernel should use work-group local IDs"
        );
        assert!(MATMUL_I2S_TILED_SRC.contains("get_group_id"), "tiled kernel should use group IDs");
    }
}
