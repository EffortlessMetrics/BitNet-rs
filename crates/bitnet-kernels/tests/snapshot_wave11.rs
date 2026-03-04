//! Wave 11 snapshot tests for `bitnet-kernels` — CPU kernel configs,
//! CUDA kernel source strings, KernelProvider dispatch, and config
//! struct Debug output.
//!
//! Pins the Debug representations and construction defaults of config
//! structs that were not covered in waves 5–6 so that unintentional
//! changes are caught at review time.

// =========================================================================
// Section 1 — CPU config struct Debug snapshots
// =========================================================================

// ── TransposeConfig ────────────────────────────────────────────────


use bitnet_kernels::cpu::batch_norm::BatchNormConfig;

#[test]
fn batch_norm_config_default_debug() {
    let cfg = BatchNormConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn batch_norm_config_custom_debug() {
    let cfg = BatchNormConfig { num_features: 64, eps: 1e-3, momentum: 0.2, training: true };
    insta::assert_debug_snapshot!(cfg);
}

// ── LayerNormConfig ────────────────────────────────────────────────

use bitnet_kernels::cpu::layer_norm::{GroupNormConfig, LayerNormConfig};

#[test]
fn layer_norm_config_default_debug() {
    let cfg = LayerNormConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn layer_norm_config_custom_debug() {
    let cfg =
        LayerNormConfig { normalized_shape: vec![64, 128], eps: 1e-6, elementwise_affine: false };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn group_norm_config_debug() {
    let cfg = GroupNormConfig::new(8, 64, 256);
    insta::assert_debug_snapshot!(cfg);
}

// ── EmbeddingConfig ────────────────────────────────────────────────

use bitnet_kernels::cpu::embedding::EmbeddingConfig;

#[test]
fn embedding_config_debug() {
    let cfg = EmbeddingConfig { vocab_size: 32000, embedding_dim: 768, padding_idx: Some(0) };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn embedding_config_no_padding_debug() {
    let cfg = EmbeddingConfig { vocab_size: 50257, embedding_dim: 1024, padding_idx: None };
    insta::assert_debug_snapshot!(cfg);
}

// ── CpuEmbeddingConfig ────────────────────────────────────────────

use bitnet_kernels::cpu::CpuEmbeddingConfig;

#[test]
fn cpu_embedding_config_debug() {
    let cfg = CpuEmbeddingConfig::new(32000, 768).with_padding_idx(0).with_max_norm(1.0);
    insta::assert_debug_snapshot!(cfg);
}

// ── Conv2dConfig ───────────────────────────────────────────────────

use bitnet_kernels::cpu::Conv2dConfig;

#[test]
fn conv2d_config_simple_debug() {
    let cfg = Conv2dConfig::new(3, 64, 3);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn conv2d_config_custom_debug() {
    let cfg = Conv2dConfig {
        in_channels: 64,
        out_channels: 128,
        kernel_h: 5,
        kernel_w: 5,
        stride_h: 2,
        stride_w: 2,
        padding_h: 2,
        padding_w: 2,
        dilation_h: 1,
        dilation_w: 1,
        groups: 4,
    };
    insta::assert_debug_snapshot!(cfg);
}

// ── SimdMatmulConfig ──────────────────────────────────────────────

use bitnet_kernels::cpu::simd_matmul::SimdMatmulConfig;

#[test]
fn simd_matmul_config_default_debug() {
    let cfg = SimdMatmulConfig::new(32, 64, 128);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn simd_matmul_config_with_transpose_debug() {
    let cfg = SimdMatmulConfig {
        m: 16,
        n: 32,
        k: 64,
        alpha: 0.5,
        beta: 1.0,
        transpose_a: true,
        transpose_b: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

// ── FusionConfig ──────────────────────────────────────────────────

use bitnet_kernels::cpu::fusion::{FusedOp, FusionConfig};

#[test]
fn fusion_config_default_debug() {
    let cfg = FusionConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn fusion_config_disabled_debug() {
    let cfg = FusionConfig::disabled();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn fused_op_display_all_variants() {
    let ops = [
        FusedOp::RmsNormLinear,
        FusedOp::GeluLinear,
        FusedOp::SoftmaxMask,
        FusedOp::AddNormalize,
        FusedOp::ScaleAndAdd,
    ];
    let displays: Vec<String> = ops.iter().map(|o| o.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

// ── KvCacheConfig ─────────────────────────────────────────────────

use bitnet_kernels::cpu::kv_cache::{KvCacheConfig, KvDtype};

#[test]
fn kv_cache_config_debug() {
    let cfg = KvCacheConfig {
        num_layers: 32,
        num_heads: 8,
        head_dim: 64,
        max_seq_len: 2048,
        dtype: KvDtype::F32,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn kv_dtype_all_variants_debug() {
    let dtypes = [KvDtype::F32, KvDtype::F16, KvDtype::Bf16];
    let debugs: Vec<String> = dtypes.iter().map(|d| format!("{d:?}")).collect();
    insta::assert_debug_snapshot!(debugs);
}

// ── GqaConfig ─────────────────────────────────────────────────────

use bitnet_kernels::cpu::attention::{CpuAttentionConfig, GqaConfig};

#[test]
fn gqa_config_debug() {
    let cfg = GqaConfig {
        num_q_heads: 32,
        num_kv_heads: 8,
        head_dim: 64,
        seq_len: 512,
        causal: true,
        scale: None,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cpu_attention_config_debug() {
    let cfg = CpuAttentionConfig {
        batch_size: 4,
        num_heads: 8,
        seq_len: 256,
        head_dim: 64,
        scale: None,
        causal_mask: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cpu_attention_config_resolved_scale() {
    let cfg = CpuAttentionConfig {
        batch_size: 1,
        num_heads: 1,
        seq_len: 1,
        head_dim: 64,
        scale: None,
        causal_mask: false,
    };
    insta::assert_snapshot!(format!("{:.6}", cfg.resolved_scale()));
}

// ── ScatterGatherConfig / ScatterReduce ───────────────────────────

use bitnet_kernels::cpu::scatter_gather::{ScatterGatherConfig, ScatterReduce};

#[test]
fn scatter_gather_config_default_debug() {
    let cfg = ScatterGatherConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn scatter_reduce_all_variants_debug() {
    let modes = [
        ScatterReduce::Assign,
        ScatterReduce::Add,
        ScatterReduce::Max,
        ScatterReduce::Min,
        ScatterReduce::Mul,
    ];
    let debugs: Vec<String> = modes.iter().map(|m| format!("{m:?}")).collect();
    insta::assert_debug_snapshot!(debugs);
}

// ── ActivationType ────────────────────────────────────────────────

use bitnet_kernels::cpu::ActivationType;

#[test]
fn activation_type_all_variants_debug() {
    let types = [
        ActivationType::ReLU,
        ActivationType::LeakyReLU(0.01),
        ActivationType::GELU,
        ActivationType::GELUTanh,
        ActivationType::SiLU,
        ActivationType::Swish(1.0),
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::HardSigmoid,
        ActivationType::HardSwish,
        ActivationType::Mish,
        ActivationType::Softplus,
        ActivationType::ELU(1.0),
        ActivationType::SELU,
        ActivationType::QuickGELU,
    ];
    let debugs: Vec<String> = types.iter().map(|t| format!("{t:?}")).collect();
    insta::assert_debug_snapshot!(debugs);
}

// ── LossReduction ─────────────────────────────────────────────────

use bitnet_kernels::cpu::LossReduction;

#[test]
fn loss_reduction_all_variants_debug() {
    let modes = [LossReduction::None, LossReduction::Mean, LossReduction::Sum];
    let debugs: Vec<String> = modes.iter().map(|m| format!("{m:?}")).collect();
    insta::assert_debug_snapshot!(debugs);
}

// =========================================================================
// Section 2 — KernelManager dispatch
// =========================================================================

use bitnet_kernels::KernelManager;

#[test]
fn kernel_manager_selected_provider_name() {
    let mgr = KernelManager::new();
    let name = mgr.selected_provider_name();
    // The name depends on the host CPU; filter the specific provider.
    insta::with_settings!({filters => vec![
        (r"(?:AVX-?512|AVX2|SSE4\.?2|NEON|Fallback|neon|avx2|avx512)", "[PROVIDER]"),
    ]}, {
        insta::assert_snapshot!(format!("has_name={}", name.is_some()));
    });
}

#[test]
fn kernel_manager_select_best_succeeds() {
    let mgr = KernelManager::new();
    let provider = mgr.select_best();
    assert!(provider.is_ok());
    let name = provider.unwrap().name();
    insta::with_settings!({filters => vec![
        (r"(?:AVX-?512|AVX2|SSE4\.?2|NEON|Fallback|neon|avx2|avx512)", "[PROVIDER]"),
    ]}, {
        insta::assert_snapshot!(format!("provider_name={name}"));
    });
}

// =========================================================================
// Section 3 — CUDA kernel source strings (gpu feature only)
// =========================================================================

#[cfg(any(feature = "gpu", feature = "cuda"))]
mod cuda_kernel_sources {
    #[test]
    fn activation_kernel_src_snapshot() {
        insta::assert_snapshot!(bitnet_kernels::cuda::activations::ACTIVATION_KERNEL_SRC);
    }

    #[test]
    fn layernorm_kernel_src_snapshot() {
        insta::assert_snapshot!(bitnet_kernels::cuda::layernorm::LAYERNORM_KERNEL_SRC);
    }

    #[test]
    fn transpose_2d_kernel_src_snapshot() {
        insta::assert_snapshot!(bitnet_kernels::cuda::transpose::TRANSPOSE_2D_KERNEL_SRC);
    }

    #[test]
    fn transpose_nd_kernel_src_snapshot() {
        insta::assert_snapshot!(bitnet_kernels::cuda::transpose::TRANSPOSE_ND_KERNEL_SRC);
    }

    #[test]
    fn elementwise_binary_kernel_src_snapshot() {
        insta::assert_snapshot!(bitnet_kernels::cuda::elementwise::ELEMENTWISE_BINARY_KERNEL_SRC);
    }

    #[test]
    fn elementwise_unary_kernel_src_snapshot() {
        insta::assert_snapshot!(bitnet_kernels::cuda::elementwise::ELEMENTWISE_UNARY_KERNEL_SRC);
    }

    #[test]
    fn fusion_kernel_src_snapshot() {
        insta::assert_snapshot!(bitnet_kernels::cuda::fusion::FUSION_KERNEL_SRC);
    }

    #[test]
    fn rope_forward_kernel_src_snapshot() {
        insta::assert_snapshot!(bitnet_kernels::cuda::rope::ROPE_FORWARD_KERNEL_SRC);
    }

    #[test]
    fn rope_backward_kernel_src_snapshot() {
        insta::assert_snapshot!(bitnet_kernels::cuda::rope::ROPE_BACKWARD_KERNEL_SRC);
    }

    #[test]
    fn batch_norm_train_kernel_src_snapshot() {
        insta::assert_snapshot!(bitnet_kernels::cuda::batch_norm::BATCH_NORM_TRAIN_KERNEL_SRC);
    }

    #[test]
    fn batch_norm_inference_kernel_src_snapshot() {
        insta::assert_snapshot!(bitnet_kernels::cuda::batch_norm::BATCH_NORM_INFERENCE_KERNEL_SRC);
    }

    #[test]
    fn attention_kernel_src_snapshot() {
        insta::assert_snapshot!(bitnet_kernels::cuda::attention::ATTENTION_KERNEL_SRC);
    }
}
