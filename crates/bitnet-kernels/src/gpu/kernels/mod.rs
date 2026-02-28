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

/// QK256 quantization kernels (256-element block dequantize, matmul, scale apply).
pub const QK256_SRC: &str = include_str!("qk256.cl");

/// TL1 (Ternary Level 1) kernels (pack, unpack, matmul).
pub const TL1_SRC: &str = include_str!("tl1.cl");

/// TL2 (Ternary Level 2) kernels (dequantize, quantize, matmul with per-group scales).
pub const TL2_SRC: &str = include_str!("tl2.cl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sources_are_not_empty() {
        assert!(!MATMUL_I2S_SRC.is_empty(), "matmul_i2s.cl should not be empty");
        assert!(!QUANTIZE_I2S_SRC.is_empty(), "quantize_i2s.cl should not be empty");
        assert!(!ELEMENTWISE_SRC.is_empty(), "elementwise.cl should not be empty");
        assert!(!QK256_SRC.is_empty(), "qk256.cl should not be empty");
        assert!(!TL1_SRC.is_empty(), "tl1.cl should not be empty");
        assert!(!TL2_SRC.is_empty(), "tl2.cl should not be empty");
    }

    #[test]
    fn kernel_sources_contain_kernel_keyword() {
        assert!(MATMUL_I2S_SRC.contains("__kernel"), "matmul_i2s.cl missing __kernel");
        assert!(QUANTIZE_I2S_SRC.contains("__kernel"), "quantize_i2s.cl missing __kernel");
        assert!(ELEMENTWISE_SRC.contains("__kernel"), "elementwise.cl missing __kernel");
        assert!(QK256_SRC.contains("__kernel"), "qk256.cl missing __kernel");
        assert!(TL1_SRC.contains("__kernel"), "tl1.cl missing __kernel");
        assert!(TL2_SRC.contains("__kernel"), "tl2.cl missing __kernel");
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
    fn qk256_kernels_have_expected_functions() {
        assert!(QK256_SRC.contains("qk256_dequantize"), "missing qk256_dequantize kernel");
        assert!(QK256_SRC.contains("qk256_matmul"), "missing qk256_matmul kernel");
        assert!(QK256_SRC.contains("qk256_apply_scales"), "missing qk256_apply_scales kernel");
    }

    #[test]
    fn tl1_kernels_have_expected_functions() {
        assert!(TL1_SRC.contains("tl1_pack"), "missing tl1_pack kernel");
        assert!(TL1_SRC.contains("tl1_unpack"), "missing tl1_unpack kernel");
        assert!(TL1_SRC.contains("tl1_matmul"), "missing tl1_matmul kernel");
    }

    #[test]
    fn tl2_kernels_have_expected_functions() {
        assert!(TL2_SRC.contains("tl2_dequantize"), "missing tl2_dequantize kernel");
        assert!(TL2_SRC.contains("tl2_quantize"), "missing tl2_quantize kernel");
        assert!(TL2_SRC.contains("tl2_matmul"), "missing tl2_matmul kernel");
    }

    #[test]
    fn qk256_uses_256_element_blocks() {
        assert!(QK256_SRC.contains("256"), "QK256 should reference 256-element blocks");
        assert!(QK256_SRC.contains("64"), "QK256 should reference 64 bytes per block");
    }

    #[test]
    fn tl2_uses_per_group_scales() {
        assert!(TL2_SRC.contains("group_size"), "TL2 should use per-group scales");
        assert!(TL2_SRC.contains("group_scales"), "TL2 should have group_scales parameter");
    }
}
