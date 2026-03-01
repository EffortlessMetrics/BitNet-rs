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

/// Embedding lookup and output projection kernel source.
pub const EMBEDDING_SRC: &str = include_str!("embedding.cl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sources_are_not_empty() {
        assert!(!MATMUL_I2S_SRC.is_empty(), "matmul_i2s.cl should not be empty");
        assert!(!QUANTIZE_I2S_SRC.is_empty(), "quantize_i2s.cl should not be empty");
        assert!(!ELEMENTWISE_SRC.is_empty(), "elementwise.cl should not be empty");
    }

    #[test]
    fn kernel_sources_contain_kernel_keyword() {
        assert!(MATMUL_I2S_SRC.contains("__kernel"), "matmul_i2s.cl missing __kernel");
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
    fn embedding_source_is_not_empty() {
        assert!(!EMBEDDING_SRC.is_empty(), "embedding.cl should not be empty");
    }

    #[test]
    fn embedding_source_contains_kernel_keyword() {
        assert!(EMBEDDING_SRC.contains("__kernel"), "embedding.cl missing __kernel");
    }

    #[test]
    fn embedding_kernels_have_expected_functions() {
        assert!(EMBEDDING_SRC.contains("embedding_lookup"), "missing embedding_lookup kernel");
        assert!(
            EMBEDDING_SRC.contains("output_projection"),
            "missing output_projection kernel"
        );
        assert!(EMBEDDING_SRC.contains("embedding_rms_norm"), "missing embedding_rms_norm kernel");
        assert!(
            EMBEDDING_SRC.contains("add_position_embedding"),
            "missing add_position_embedding kernel"
        );
        assert!(
            EMBEDDING_SRC.contains("embedding_lookup_padded"),
            "missing embedding_lookup_padded kernel"
        );
    }
}
