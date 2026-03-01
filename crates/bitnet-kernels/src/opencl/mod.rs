//! OpenCL kernel sources for Intel Arc GPU acceleration.
//!
//! These kernels target Intel Xe-HPG architecture (Arc A770/A750)
//! and are compiled at runtime via the OpenCL driver.

/// Matrix multiplication kernels (naive, tiled, batched)
pub const MATMUL_CL: &str = include_str!("matmul.cl");

/// Softmax kernels (row-wise, with temperature)
pub const SOFTMAX_CL: &str = include_str!("softmax.cl");

/// Layer normalization kernels (LayerNorm, RMSNorm)
pub const LAYER_NORM_CL: &str = include_str!("layer_norm.cl");

/// Rotary Position Embedding (RoPE) kernel
pub const ROPE_CL: &str = include_str!("rope.cl");

/// Element-wise operations (add, mul, scale, SiLU, GELU, ReLU)
pub const ELEMENTWISE_CL: &str = include_str!("elementwise.cl");

/// Quantized operations (I2_S dequantize, quantized matvec)
pub const QUANTIZED_CL: &str = include_str!("quantized.cl");

/// All kernel sources combined
pub const ALL_KERNELS: &[(&str, &str)] = &[
    ("matmul", MATMUL_CL),
    ("softmax", SOFTMAX_CL),
    ("layer_norm", LAYER_NORM_CL),
    ("rope", ROPE_CL),
    ("elementwise", ELEMENTWISE_CL),
    ("quantized", QUANTIZED_CL),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sources_not_empty() {
        assert!(!MATMUL_CL.is_empty());
        assert!(!SOFTMAX_CL.is_empty());
        assert!(!LAYER_NORM_CL.is_empty());
        assert!(!ROPE_CL.is_empty());
        assert!(!ELEMENTWISE_CL.is_empty());
        assert!(!QUANTIZED_CL.is_empty());
    }

    #[test]
    fn all_kernels_has_correct_count() {
        assert_eq!(ALL_KERNELS.len(), 6);
    }

    #[test]
    fn kernel_sources_contain_kernel_keyword() {
        for (name, source) in ALL_KERNELS {
            assert!(
                source.contains("__kernel"),
                "Kernel source '{}' should contain __kernel",
                name
            );
        }
    }

    #[test]
    fn matmul_has_tiled_variant() {
        assert!(MATMUL_CL.contains("matmul_tiled"));
        assert!(MATMUL_CL.contains("matmul_naive"));
        assert!(MATMUL_CL.contains("matmul_batched"));
    }

    #[test]
    fn softmax_has_temperature() {
        assert!(SOFTMAX_CL.contains("softmax_with_temperature"));
        assert!(SOFTMAX_CL.contains("softmax_row"));
    }

    #[test]
    fn layer_norm_has_rms() {
        assert!(LAYER_NORM_CL.contains("layer_norm"));
        assert!(LAYER_NORM_CL.contains("rms_norm"));
    }

    #[test]
    fn rope_has_forward() {
        assert!(ROPE_CL.contains("rope_forward"));
        assert!(ROPE_CL.contains("theta_base"));
    }

    #[test]
    fn elementwise_has_all_ops() {
        for op in &["add", "mul", "scale", "silu", "gelu", "relu"] {
            assert!(ELEMENTWISE_CL.contains(op), "elementwise.cl should contain '{}'", op);
        }
    }

    #[test]
    fn quantized_has_i2s_ops() {
        assert!(QUANTIZED_CL.contains("dequantize_i2s"));
        assert!(QUANTIZED_CL.contains("i2s_matvec"));
    }

    #[test]
    fn kernel_names_are_unique() {
        let mut names: Vec<&str> = Vec::new();
        for (name, _) in ALL_KERNELS {
            assert!(!names.contains(name), "Duplicate kernel name: {}", name);
            names.push(name);
        }
    }
}
