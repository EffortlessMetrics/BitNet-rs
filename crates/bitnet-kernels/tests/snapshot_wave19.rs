//! Wave 19 snapshot tests for CUDA kernel configurations and GPU data
//! structures in `bitnet-kernels`.
//!
//! Pins the Debug representations of CUDA kernel config structs,
//! GPU validation/mixed-precision types, and related enums so that
//! unintentional changes are caught at review time.

// =========================================================================
// Section 1 — CUDA attention configs
// =========================================================================

use bitnet_kernels::cuda::attention::{AttentionConfig, AttentionKernelConfig};

#[test]
fn attention_kernel_config_default_shape() {
    let cfg = AttentionKernelConfig::for_shape(8, 64, 32, 32, true).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn attention_kernel_config_asymmetric_seq() {
    let cfg = AttentionKernelConfig::for_shape(4, 128, 16, 64, false).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn attention_config_causal() {
    let cfg = AttentionConfig::new(8, 64, 32, true).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn attention_config_non_causal_custom_scale() {
    let cfg = AttentionConfig::new(4, 128, 16, false).unwrap().with_scale(0.05);
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 2 — CUDA matmul configs
// =========================================================================

use bitnet_kernels::cuda::matmul::{GemmConfig, MatmulConfig, MatmulDtype};

#[test]
fn matmul_config_default_debug() {
    let cfg = MatmulConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn matmul_config_f16_batched() {
    let cfg = MatmulConfig::default()
        .with_dtype(MatmulDtype::F16)
        .with_batch_size(4)
        .unwrap()
        .with_transpose(true, false);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn gemm_config_basic() {
    let cfg = GemmConfig::new(256, 512, 128);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn gemm_config_with_scalars_and_transpose() {
    let cfg = GemmConfig::new(64, 64, 64).with_scalars(2.0, 0.5).with_transpose(true, true);
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 3 — CUDA softmax configs
// =========================================================================

use bitnet_kernels::cuda::softmax::{BatchedSoftmaxConfig, SoftmaxConfig};

#[test]
fn softmax_config_default_shape() {
    let cfg = SoftmaxConfig::for_shape(512, 8).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn softmax_config_causal_log() {
    let cfg = SoftmaxConfig::for_shape(128, 4)
        .unwrap()
        .with_causal_mask()
        .with_log_softmax()
        .with_in_place();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn batched_softmax_config_debug() {
    let cfg = BatchedSoftmaxConfig::new(2, 8, 64).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn batched_softmax_causal_with_temperature() {
    let cfg = BatchedSoftmaxConfig::new(4, 16, 128)
        .unwrap()
        .with_temperature(0.7)
        .unwrap()
        .with_causal_mask()
        .with_log_softmax();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 4 — CUDA fusion configs
// =========================================================================

use bitnet_kernels::cuda::fusion::{
    FusedElementwiseLaunchConfig, FusedMatmulLaunchConfig, FusedOp, FusionConfig,
};

#[test]
fn fusion_config_default_debug() {
    let cfg = FusionConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn fused_matmul_launch_config_debug() {
    let cfg = FusedMatmulLaunchConfig::new(1024, 256).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn fused_elementwise_launch_config_debug() {
    let cfg = FusedElementwiseLaunchConfig::new(2048).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn fused_op_all_variants_debug() {
    let variants: Vec<FusedOp> = vec![
        FusedOp::RmsNormLinear,
        FusedOp::GeluLinear,
        FusedOp::SoftmaxMask,
        FusedOp::AddNormalize,
        FusedOp::ScaleAndAdd,
    ];
    insta::assert_debug_snapshot!(variants);
}

// =========================================================================
// Section 5 — CUDA memory pool configs
// =========================================================================

use bitnet_kernels::cuda::memory_pool::MemoryPoolConfig as CudaMemoryPoolConfig;

#[test]
fn cuda_memory_pool_config_default_debug() {
    let cfg = CudaMemoryPoolConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 6 — CUDA KV cache configs
// =========================================================================

use bitnet_kernels::cuda::kv_cache::{CacheDtype, KvCacheConfig};

#[test]
fn kv_cache_config_f32_debug() {
    let cfg = KvCacheConfig::new(12, 8, 64, 2048, CacheDtype::F32).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn kv_cache_config_f16_debug() {
    let cfg = KvCacheConfig::new(24, 16, 128, 4096, CacheDtype::F16).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cache_dtype_all_variants_debug() {
    let variants: Vec<CacheDtype> = vec![CacheDtype::F32, CacheDtype::F16, CacheDtype::Bf16];
    insta::assert_debug_snapshot!(variants);
}

// =========================================================================
// Section 7 — CUDA activation configs
// =========================================================================

use bitnet_kernels::cuda::activations::{ActivationConfig, ActivationType, SiluGateConfig};

#[test]
fn activation_config_silu_debug() {
    let cfg = ActivationConfig::new(4096, ActivationType::SiLU).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn activation_config_gelu_debug() {
    let cfg = ActivationConfig::new(1024, ActivationType::GELU).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn silu_gate_config_debug() {
    let cfg = SiluGateConfig::new(2048).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 8 — CUDA batch norm configs
// =========================================================================

use bitnet_kernels::cuda::batch_norm::{
    BatchNormConfig as CudaBatchNormCfg, BatchNormKernel, CudaBatchNormConfig,
};

#[test]
fn cuda_batch_norm_config_default_debug() {
    let cfg = CudaBatchNormCfg::new(64).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cuda_batch_norm_config_custom_debug() {
    let cfg = CudaBatchNormCfg::new(128).unwrap().with_eps(1e-3).with_momentum(0.2).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cuda_batch_norm_launch_config_debug() {
    let cfg = CudaBatchNormConfig::new(256, 1e-5, 0.1).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn batch_norm_kernel_debug() {
    let kernel = BatchNormKernel::new(32).unwrap();
    insta::assert_debug_snapshot!(kernel);
}

// =========================================================================
// Section 9 — CUDA quantize / quantized_matmul configs
// =========================================================================

use bitnet_kernels::cuda::quantize::{QuantMethod, QuantizeConfig};
use bitnet_kernels::cuda::quantized_matmul::I2sMatmulConfig;

#[test]
fn quantize_config_default_debug() {
    let cfg = QuantizeConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn quant_method_all_variants_debug() {
    let variants: Vec<QuantMethod> = vec![
        QuantMethod::AbsMax,
        QuantMethod::MinMax,
        QuantMethod::Symmetric,
        QuantMethod::Percentile(95),
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn i2s_matmul_config_default_debug() {
    let cfg = I2sMatmulConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 10 — CUDA RoPE, RMSNorm, LayerNorm configs
// =========================================================================

use bitnet_kernels::cuda::layernorm::LayerNormConfig as CudaLayerNormConfig;
use bitnet_kernels::cuda::rmsnorm::RmsNormConfig;
use bitnet_kernels::cuda::rope::RopeConfig;

#[test]
fn rope_config_default_shape_debug() {
    let cfg = RopeConfig::for_shape(64, 8, 128).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn rope_config_with_options_debug() {
    let cfg = RopeConfig::for_shape(128, 16, 256)
        .unwrap()
        .with_base(500_000.0)
        .with_scaling_factor(2.0)
        .with_position_offset(32)
        .with_interleaved(true);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn rmsnorm_config_debug() {
    let cfg = RmsNormConfig::for_shape(2048, 16).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn rmsnorm_config_custom_eps_debug() {
    let cfg = RmsNormConfig::for_shape(768, 4).unwrap().with_eps(1e-8);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cuda_layernorm_config_default_debug() {
    let cfg = CudaLayerNormConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cuda_layernorm_config_with_shape_debug() {
    let cfg = CudaLayerNormConfig::new(1e-5, true).unwrap().with_normalized_shape(vec![768]);
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 11 — CUDA embedding configs
// =========================================================================

use bitnet_kernels::cuda::embedding::{EmbeddingKernelConfig, PositionEmbeddingConfig};

#[test]
fn embedding_kernel_config_debug() {
    let cfg = EmbeddingKernelConfig::new(32000, 2048, 128).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn embedding_kernel_config_with_padding_debug() {
    let cfg = EmbeddingKernelConfig::new(50257, 768, 64).unwrap().with_padding_idx(0);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn position_embedding_config_debug() {
    let cfg = PositionEmbeddingConfig::new(2048, 768, 128).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 12 — CUDA gating / elementwise / conv1d / pooling configs
// =========================================================================

use bitnet_kernels::cuda::elementwise::{ElementwiseConfig, ElementwiseOp};
use bitnet_kernels::cuda::gating::{GatingConfig, GatingType};
use bitnet_kernels::cuda::pooling::{
    AdaptivePool2dConfig, CudaPoolType, Pool2dConfig, PoolingConfig,
};

#[test]
fn gating_config_swiglu_debug() {
    let cfg = GatingConfig::new(4096, GatingType::SwiGLU).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn gating_type_all_variants_debug() {
    let variants: Vec<GatingType> = vec![GatingType::SwiGLU, GatingType::GeGLU, GatingType::ReGLU];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn elementwise_config_add_debug() {
    let cfg = ElementwiseConfig::new(8192, ElementwiseOp::Add).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn elementwise_config_clamp_debug() {
    let cfg = ElementwiseConfig::new(1024, ElementwiseOp::Clamp)
        .unwrap()
        .with_clamp_bounds(-1.0, 1.0)
        .unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn pooling_config_max_debug() {
    let cfg = PoolingConfig::new(CudaPoolType::Max, 256, 3, 2, 1).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn pool2d_config_debug() {
    let cfg = Pool2dConfig::new(2, 64, 28, 28, 3, 3, 1, 1, 1, 1).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn adaptive_pool2d_config_debug() {
    let cfg = AdaptivePool2dConfig::new(1, 512, 14, 14, 7, 7).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 13 — CUDA QK256 GEMV config
// =========================================================================

use bitnet_kernels::cuda::qk256_gemv::Qk256GemvConfig;

#[test]
fn qk256_gemv_config_default_debug() {
    let cfg = Qk256GemvConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 14 — GPU validation / mixed precision configs (gpu feature only)
// =========================================================================

#[cfg(any(feature = "gpu", feature = "cuda"))]
mod gpu_configs {
    use bitnet_kernels::gpu::mixed_precision::{MixedPrecisionMetrics, PrecisionMode};
    use bitnet_kernels::gpu::validation::ValidationConfig;

    #[test]
    fn validation_config_default_debug() {
        let cfg = ValidationConfig::default();
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn mixed_precision_metrics_default_debug() {
        let metrics = MixedPrecisionMetrics::default();
        insta::assert_debug_snapshot!(metrics);
    }

    #[test]
    fn precision_mode_all_variants_debug() {
        let variants: Vec<PrecisionMode> = vec![
            PrecisionMode::FP32,
            PrecisionMode::FP16,
            PrecisionMode::BF16,
            PrecisionMode::Auto,
        ];
        insta::assert_debug_snapshot!(variants);
    }

    // =========================================================================
    // Section 15 — GPU memory optimization config
    // =========================================================================

    use bitnet_kernels::gpu::memory_optimization::{
        MemoryPoolConfig as GpuMemoryPoolConfig, MemoryStats as GpuMemoryStats,
    };

    #[test]
    fn gpu_memory_pool_config_default_debug() {
        let cfg = GpuMemoryPoolConfig::default();
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn gpu_memory_stats_default_debug() {
        let stats = GpuMemoryStats::default();
        insta::assert_debug_snapshot!(stats);
    }
}

// =========================================================================
// Section 16 — CPU FFN config (kernel-adjacent)
// =========================================================================

use bitnet_kernels::cpu::ffn::{FfnActivation, FfnConfig};

#[test]
fn ffn_config_silu_debug() {
    let cfg = FfnConfig::new(2048, 5504, FfnActivation::SiLU).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn ffn_activation_all_variants_debug() {
    let variants: Vec<FfnActivation> =
        vec![FfnActivation::GeLU, FfnActivation::SiLU, FfnActivation::ReLU];
    insta::assert_debug_snapshot!(variants);
}

// =========================================================================
// Section 17 — CPU loss reduction enum
// =========================================================================

use bitnet_kernels::cpu::loss::LossReduction;

#[test]
fn loss_reduction_all_variants_debug() {
    let variants: Vec<LossReduction> =
        vec![LossReduction::None, LossReduction::Mean, LossReduction::Sum];
    insta::assert_debug_snapshot!(variants);
}
