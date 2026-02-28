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

/// FP16 matrix multiplication kernel source (cl_khr_fp16).
pub const MATMUL_FP16_SRC: &str = include_str!("matmul_fp16.cl");

/// INT8 matrix multiplication kernel source.
pub const MATMUL_INT8_SRC: &str = include_str!("matmul_int8.cl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sources_are_not_empty() {
        assert!(!MATMUL_I2S_SRC.is_empty(), "matmul_i2s.cl should not be empty");
        assert!(!QUANTIZE_I2S_SRC.is_empty(), "quantize_i2s.cl should not be empty");
        assert!(!ELEMENTWISE_SRC.is_empty(), "elementwise.cl should not be empty");
        assert!(!MATMUL_FP16_SRC.is_empty(), "matmul_fp16.cl should not be empty");
        assert!(!MATMUL_INT8_SRC.is_empty(), "matmul_int8.cl should not be empty");
    }

    #[test]
    fn kernel_sources_contain_kernel_keyword() {
        assert!(MATMUL_I2S_SRC.contains("__kernel"), "matmul_i2s.cl missing __kernel");
        assert!(QUANTIZE_I2S_SRC.contains("__kernel"), "quantize_i2s.cl missing __kernel");
        assert!(ELEMENTWISE_SRC.contains("__kernel"), "elementwise.cl missing __kernel");
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
}
