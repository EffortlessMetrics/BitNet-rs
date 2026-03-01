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

/// Scaled dot-product attention kernel source (scores, softmax, weighted sum).
pub const ATTENTION_SRC: &str = include_str!("attention.cl");

/// Tiled GEMM and quantized GEMV kernels optimized for Intel Arc A770 Xe-HPG.
pub const TILED_MATMUL_SRC: &str = include_str!("tiled_matmul.cl");

/// Rotary Position Embedding kernel source (real-time and cached variants).
pub const ROPE_SRC: &str = include_str!("rope.cl");

/// RMSNorm and LayerNorm normalization kernels with tree reduction.
pub const NORMALIZATION_SRC: &str = include_str!("normalization.cl");

/// Activation function kernels (SiLU, GELU, ReLU, fused SiLU*up, softmax).
pub const ACTIVATIONS_SRC: &str = include_str!("activations.cl");

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
        assert!(EMBEDDING_SRC.contains("output_projection"), "missing output_projection kernel");
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

    #[test]
    fn attention_source_is_not_empty() {
        assert!(!ATTENTION_SRC.is_empty(), "attention.cl should not be empty");
    }

    #[test]
    fn attention_source_contains_kernel_keyword() {
        assert!(ATTENTION_SRC.contains("__kernel"), "attention.cl missing __kernel");
    }

    #[test]
    fn attention_kernels_have_expected_functions() {
        assert!(
            ATTENTION_SRC.contains("attention_scores"),
            "missing attention_scores kernel"
        );
        assert!(
            ATTENTION_SRC.contains("attention_softmax"),
            "missing attention_softmax kernel"
        );
        assert!(
            ATTENTION_SRC.contains("attention_weighted_sum"),
            "missing attention_weighted_sum kernel"
        );
    }

    #[test]
    fn activations_source_is_not_empty() {
        assert!(!ACTIVATIONS_SRC.is_empty(), "activations.cl should not be empty");
    }

    #[test]
    fn activations_source_contains_kernel_keyword() {
        assert!(
            ACTIVATIONS_SRC.contains("__kernel"),
            "activations.cl missing __kernel"
        );
    }

    #[test]
    fn activations_kernels_have_expected_functions() {
        assert!(ACTIVATIONS_SRC.contains("silu"), "missing silu kernel");
        assert!(ACTIVATIONS_SRC.contains("silu_mul"), "missing silu_mul kernel");
        assert!(ACTIVATIONS_SRC.contains("gelu"), "missing gelu kernel");
        assert!(ACTIVATIONS_SRC.contains("relu"), "missing relu kernel");
        assert!(
            ACTIVATIONS_SRC.contains("elementwise_add"),
            "missing elementwise_add kernel"
        );
        assert!(
            ACTIVATIONS_SRC.contains("elementwise_mul"),
            "missing elementwise_mul kernel"
        );
        assert!(ACTIVATIONS_SRC.contains("scale"), "missing scale kernel");
        assert!(
            ACTIVATIONS_SRC.contains("softmax_full"),
            "missing softmax_full kernel"
        );
    }

    #[test]
    fn rope_source_is_not_empty() {
        assert!(!ROPE_SRC.is_empty(), "rope.cl should not be empty");
    }

    #[test]
    fn rope_source_contains_kernel_keyword() {
        assert!(ROPE_SRC.contains("__kernel"), "rope.cl missing __kernel");
    }

    #[test]
    fn rope_kernels_have_expected_functions() {
        assert!(ROPE_SRC.contains("rope_apply"), "missing rope_apply kernel");
        assert!(
            ROPE_SRC.contains("rope_apply_cached"),
            "missing rope_apply_cached kernel"
        );
    }

    #[test]
    fn rope_kernel_has_required_parameters() {
        assert!(ROPE_SRC.contains("theta_base"), "rope_apply missing theta_base param");
        assert!(
            ROPE_SRC.contains("position_offset"),
            "rope kernels missing position_offset param"
        );
        assert!(ROPE_SRC.contains("cos_cache"), "rope_apply_cached missing cos_cache");
        assert!(ROPE_SRC.contains("sin_cache"), "rope_apply_cached missing sin_cache");
    }

    #[test]
    fn normalization_source_is_not_empty() {
        assert!(!NORMALIZATION_SRC.is_empty(), "normalization.cl should not be empty");
    }

    #[test]
    fn normalization_source_contains_kernel_keyword() {
        assert!(
            NORMALIZATION_SRC.contains("__kernel"),
            "normalization.cl missing __kernel"
        );
    }

    #[test]
    fn normalization_kernels_have_expected_functions() {
        assert!(NORMALIZATION_SRC.contains("rmsnorm"), "missing rmsnorm kernel");
        assert!(NORMALIZATION_SRC.contains("layernorm"), "missing layernorm kernel");
    }

    #[test]
    fn tiled_matmul_source_is_not_empty() {
        assert!(!TILED_MATMUL_SRC.is_empty(), "tiled_matmul.cl should not be empty");
    }

    #[test]
    fn tiled_matmul_source_contains_kernel_keyword() {
        assert!(TILED_MATMUL_SRC.contains("__kernel"), "tiled_matmul.cl missing __kernel");
    }

    #[test]
    fn tiled_matmul_kernels_have_expected_functions() {
        assert!(
            TILED_MATMUL_SRC.contains("tiled_matmul_f32"),
            "missing tiled_matmul_f32 kernel"
        );
        assert!(
            TILED_MATMUL_SRC.contains("quantized_gemv_i2s"),
            "missing quantized_gemv_i2s kernel"
        );
    }

    #[test]
    fn tiled_matmul_uses_local_memory() {
        assert!(
            TILED_MATMUL_SRC.contains("__local float*"),
            "tiled_matmul_f32 should use local memory tiles"
        );
        assert!(
            TILED_MATMUL_SRC.contains("barrier(CLK_LOCAL_MEM_FENCE)"),
            "tiled kernel should use local memory barriers"
        );
    }
}
