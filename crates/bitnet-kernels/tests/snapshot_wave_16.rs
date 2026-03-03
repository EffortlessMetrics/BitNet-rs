//! Snapshot wave 16 — 67 insta tests for kernel output stability.
//!
//! Covers: config defaults, enum Debug/Display formatting, kernel output
//! shapes, error messages, metric calculations, and state transitions.

// ── Config default snapshots ───────────────────────────────────────

#[test]
fn snap_conv2d_config_default() {
    let config = bitnet_kernels::cpu::Conv2dConfig::default();
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_layer_norm_config_default() {
    let config = bitnet_kernels::cpu::LayerNormConfig::default();
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_linear_config_default() {
    let config = bitnet_kernels::cpu::LinearConfig::default();
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_batch_norm_config_default() {
    let config = bitnet_kernels::cpu::BatchNormConfig::default();
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_fusion_config_default() {
    let config = bitnet_kernels::cpu::fusion::FusionConfig::default();
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_fusion_config_disabled() {
    let config = bitnet_kernels::cpu::fusion::FusionConfig::disabled();
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_scatter_gather_config_default() {
    let config = bitnet_kernels::cpu::scatter_gather::ScatterGatherConfig::default();
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_device_capability_matrix_default() {
    let matrix = bitnet_kernels::capability_matrix::DeviceCapabilityMatrix::default();
    insta::assert_debug_snapshot!(matrix);
}

// ── Config constructor snapshots ───────────────────────────────────

#[test]
fn snap_conv2d_config_3x3() {
    let config = bitnet_kernels::cpu::Conv2dConfig::new(3, 16, 3);
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_layer_norm_config_hidden_768() {
    let config = bitnet_kernels::cpu::LayerNormConfig::new(vec![768]);
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_group_norm_config_8g_32c() {
    let config = bitnet_kernels::cpu::layer_norm::GroupNormConfig::new(8, 32, 64);
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_batch_norm_config_64_features() {
    let config = bitnet_kernels::cpu::batch_norm::BatchNormConfig::new(64);
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_attention_config_mha() {
    let config = bitnet_kernels::cpu::AttentionConfig {
        num_heads: 8,
        head_dim: 64,
        seq_len: 128,
        causal: true,
        use_alibi: false,
        scale: None,
    };
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_gqa_config() {
    let config = bitnet_kernels::cpu::GqaConfig {
        num_q_heads: 32,
        num_kv_heads: 8,
        head_dim: 64,
        seq_len: 512,
        causal: true,
        scale: None,
    };
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_cpu_attention_config() {
    let config = bitnet_kernels::cpu::attention::CpuAttentionConfig {
        batch_size: 2,
        num_heads: 4,
        seq_len: 16,
        head_dim: 64,
        scale: None,
        causal_mask: true,
    };
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_rope_config_default() {
    let config = bitnet_kernels::cpu::rope::RopeConfig::new(64, 2048);
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_rope_config_custom_base() {
    let config = bitnet_kernels::cpu::rope::RopeConfig::new(128, 4096)
        .with_base(500_000.0)
        .with_scaling_factor(2.0);
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_kv_cache_config() {
    let config = bitnet_kernels::cpu::kv_cache::KvCacheConfig {
        num_layers: 24,
        num_heads: 8,
        head_dim: 64,
        max_seq_len: 2048,
        dtype: bitnet_kernels::cpu::kv_cache::KvDtype::F32,
    };
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_embedding_config() {
    let config = bitnet_kernels::cpu::embedding::EmbeddingConfig {
        vocab_size: 32000,
        embedding_dim: 768,
        padding_idx: Some(0),
    };
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_cpu_embedding_config_with_options() {
    let config = bitnet_kernels::cpu::CpuEmbeddingConfig::new(50000, 512)
        .with_padding_idx(0)
        .with_max_norm(1.0);
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_ffn_config_gelu() {
    let config =
        bitnet_kernels::cpu::FfnConfig::new(768, 3072, bitnet_kernels::cpu::FfnActivation::GeLU)
            .unwrap();
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_pool_config_max() {
    let config = bitnet_kernels::cpu::PoolConfig {
        pool_type: bitnet_kernels::cpu::PoolType::Max,
        kernel_size: 3,
        stride: 2,
        padding: 1,
        dilation: 1,
        ceil_mode: false,
    };
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_reduction_config_sum() {
    let config = bitnet_kernels::reduction::ReductionConfig::new(
        256,
        4,
        bitnet_kernels::reduction::ReductionOp::Sum,
    )
    .unwrap();
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_shaped_reduction_config_axis1_keepdim() {
    let config = bitnet_kernels::shaped_reduction::ShapedReductionConfig::new(
        bitnet_kernels::reduction::ReductionOp::Mean,
        Some(1),
        true,
    );
    insta::assert_debug_snapshot!(config);
}

#[test]
fn snap_shaped_reduction_config_global() {
    let config = bitnet_kernels::shaped_reduction::ShapedReductionConfig::global(
        bitnet_kernels::reduction::ReductionOp::L2Norm,
    );
    insta::assert_debug_snapshot!(config);
}

// ── Enum variant Debug snapshots ───────────────────────────────────

#[test]
fn snap_pool_type_all_variants() {
    let variants = vec![
        bitnet_kernels::cpu::PoolType::Max,
        bitnet_kernels::cpu::PoolType::Average,
        bitnet_kernels::cpu::PoolType::GlobalMax,
        bitnet_kernels::cpu::PoolType::GlobalAverage,
        bitnet_kernels::cpu::PoolType::AvgPoolCountIncludePad,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn snap_reduction_axis_variants() {
    let variants = vec![
        bitnet_kernels::cpu::reduction::ReductionAxis::Row,
        bitnet_kernels::cpu::reduction::ReductionAxis::Column,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn snap_reduction_op_all_variants() {
    let variants = vec![
        bitnet_kernels::reduction::ReductionOp::Sum,
        bitnet_kernels::reduction::ReductionOp::Max,
        bitnet_kernels::reduction::ReductionOp::Min,
        bitnet_kernels::reduction::ReductionOp::Mean,
        bitnet_kernels::reduction::ReductionOp::L2Norm,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn snap_loss_reduction_variants() {
    let variants = vec![
        bitnet_kernels::cpu::LossReduction::None,
        bitnet_kernels::cpu::LossReduction::Mean,
        bitnet_kernels::cpu::LossReduction::Sum,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn snap_gating_type_variants() {
    let variants = vec![
        bitnet_kernels::cpu::GatingType::SwiGLU,
        bitnet_kernels::cpu::GatingType::GeGLU,
        bitnet_kernels::cpu::GatingType::ReGLU,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn snap_ffn_activation_variants() {
    let variants = vec![
        bitnet_kernels::cpu::FfnActivation::GeLU,
        bitnet_kernels::cpu::FfnActivation::SiLU,
        bitnet_kernels::cpu::FfnActivation::ReLU,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn snap_kv_dtype_variants() {
    let variants = vec![
        bitnet_kernels::cpu::kv_cache::KvDtype::F32,
        bitnet_kernels::cpu::kv_cache::KvDtype::F16,
        bitnet_kernels::cpu::kv_cache::KvDtype::Bf16,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn snap_scatter_reduce_variants() {
    let variants = vec![
        bitnet_kernels::cpu::scatter_gather::ScatterReduce::Assign,
        bitnet_kernels::cpu::scatter_gather::ScatterReduce::Add,
        bitnet_kernels::cpu::scatter_gather::ScatterReduce::Max,
        bitnet_kernels::cpu::scatter_gather::ScatterReduce::Min,
        bitnet_kernels::cpu::scatter_gather::ScatterReduce::Mul,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn snap_fused_op_variants() {
    let variants = vec![
        bitnet_kernels::cpu::fusion::FusedOp::RmsNormLinear,
        bitnet_kernels::cpu::fusion::FusedOp::GeluLinear,
        bitnet_kernels::cpu::fusion::FusedOp::SoftmaxMask,
        bitnet_kernels::cpu::fusion::FusedOp::AddNormalize,
        bitnet_kernels::cpu::fusion::FusedOp::ScaleAndAdd,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn snap_activation_type_variants() {
    let variants: Vec<bitnet_kernels::cpu::ActivationType> = vec![
        bitnet_kernels::cpu::ActivationType::ReLU,
        bitnet_kernels::cpu::ActivationType::LeakyReLU(0.01),
        bitnet_kernels::cpu::ActivationType::GELU,
        bitnet_kernels::cpu::ActivationType::GELUTanh,
        bitnet_kernels::cpu::ActivationType::SiLU,
        bitnet_kernels::cpu::ActivationType::Swish(1.0),
        bitnet_kernels::cpu::ActivationType::Sigmoid,
        bitnet_kernels::cpu::ActivationType::Tanh,
        bitnet_kernels::cpu::ActivationType::HardSigmoid,
        bitnet_kernels::cpu::ActivationType::HardSwish,
        bitnet_kernels::cpu::ActivationType::Mish,
        bitnet_kernels::cpu::ActivationType::Softplus,
        bitnet_kernels::cpu::ActivationType::ELU(1.0),
        bitnet_kernels::cpu::ActivationType::SELU,
        bitnet_kernels::cpu::ActivationType::QuickGELU,
    ];
    insta::assert_debug_snapshot!(variants);
}

// ── Display formatting snapshots ───────────────────────────────────

#[test]
fn snap_device_class_display_all() {
    use bitnet_kernels::capability_matrix::DeviceClass;
    let display: Vec<String> = DeviceClass::ALL.iter().map(|d| d.to_string()).collect();
    insta::assert_debug_snapshot!(display);
}

#[test]
fn snap_operation_category_display_all() {
    use bitnet_kernels::capability_matrix::OperationCategory;
    let display: Vec<String> = OperationCategory::ALL.iter().map(|o| o.to_string()).collect();
    insta::assert_debug_snapshot!(display);
}

#[test]
fn snap_precision_support_display_all() {
    use bitnet_kernels::capability_matrix::PrecisionSupport;
    let display: Vec<String> = PrecisionSupport::ALL.iter().map(|p| p.to_string()).collect();
    insta::assert_debug_snapshot!(display);
}

#[test]
fn snap_support_level_display_variants() {
    use bitnet_kernels::capability_matrix::SupportLevel;
    let variants = vec![
        SupportLevel::Full(0.95).to_string(),
        SupportLevel::Partial("no FP16 accumulate".into()).to_string(),
        SupportLevel::Emulated.to_string(),
        SupportLevel::Unsupported.to_string(),
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn snap_fused_op_display() {
    use bitnet_kernels::cpu::fusion::FusedOp;
    let display: Vec<String> = vec![
        FusedOp::RmsNormLinear.to_string(),
        FusedOp::GeluLinear.to_string(),
        FusedOp::SoftmaxMask.to_string(),
        FusedOp::AddNormalize.to_string(),
        FusedOp::ScaleAndAdd.to_string(),
    ];
    insta::assert_debug_snapshot!(display);
}

#[test]
fn snap_scatter_mode_display() {
    use bitnet_kernels::scatter_gather::ScatterMode;
    let variants = vec![
        format!("{:?}", ScatterMode::Assign),
        format!("{:?}", ScatterMode::Add),
        format!("{:?}", ScatterMode::Max),
        format!("{:?}", ScatterMode::Min),
    ];
    insta::assert_debug_snapshot!(variants);
}

// ── Error message formatting snapshots ─────────────────────────────

#[test]
fn snap_fusion_error_dimension_mismatch() {
    let err =
        bitnet_kernels::cpu::fusion::FusionError::DimensionMismatch { expected: 512, got: 768 };
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snap_fusion_error_invalid_config() {
    let err =
        bitnet_kernels::cpu::fusion::FusionError::InvalidConfig("min_fusion_size is 0".into());
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snap_fusion_error_empty_input() {
    let err = bitnet_kernels::cpu::fusion::FusionError::EmptyInput;
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snap_ffn_config_zero_dim_error() {
    let err =
        bitnet_kernels::cpu::FfnConfig::new(0, 3072, bitnet_kernels::cpu::FfnActivation::GeLU)
            .unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snap_reduction_config_zero_dim_error() {
    let err = bitnet_kernels::reduction::ReductionConfig::new(
        0,
        4,
        bitnet_kernels::reduction::ReductionOp::Sum,
    )
    .unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snap_reduction_config_zero_reductions_error() {
    let err = bitnet_kernels::reduction::ReductionConfig::new(
        256,
        0,
        bitnet_kernels::reduction::ReductionOp::Sum,
    )
    .unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snap_pool_config_zero_kernel_error() {
    let config = bitnet_kernels::cpu::PoolConfig {
        pool_type: bitnet_kernels::cpu::PoolType::Max,
        kernel_size: 0,
        stride: 1,
        padding: 0,
        dilation: 1,
        ceil_mode: false,
    };
    let err = config.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snap_pool_config_zero_stride_error() {
    let config = bitnet_kernels::cpu::PoolConfig {
        pool_type: bitnet_kernels::cpu::PoolType::Average,
        kernel_size: 3,
        stride: 0,
        padding: 0,
        dilation: 1,
        ceil_mode: false,
    };
    let err = config.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snap_attention_config_zero_heads_error() {
    let config = bitnet_kernels::cpu::AttentionConfig {
        num_heads: 0,
        head_dim: 64,
        seq_len: 128,
        causal: false,
        use_alibi: false,
        scale: None,
    };
    let err = config.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snap_cpu_attention_zero_batch_error() {
    let config = bitnet_kernels::cpu::attention::CpuAttentionConfig {
        batch_size: 0,
        num_heads: 4,
        seq_len: 16,
        head_dim: 64,
        scale: None,
        causal_mask: false,
    };
    let err = config.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

// ── Kernel output shape snapshots ──────────────────────────────────

#[test]
fn snap_conv2d_output_size_basic() {
    let out = bitnet_kernels::cpu::compute_output_size(28, 3, 1, 0, 1);
    insta::assert_snapshot!(format!("input=28, kernel=3, stride=1, pad=0, dil=1 -> {out}"));
}

#[test]
fn snap_conv2d_output_size_with_padding() {
    let out = bitnet_kernels::cpu::compute_output_size(28, 3, 1, 1, 1);
    insta::assert_snapshot!(format!("input=28, kernel=3, stride=1, pad=1, dil=1 -> {out}"));
}

#[test]
fn snap_conv2d_output_size_stride2() {
    let out = bitnet_kernels::cpu::compute_output_size(32, 3, 2, 1, 1);
    insta::assert_snapshot!(format!("input=32, kernel=3, stride=2, pad=1, dil=1 -> {out}"));
}

#[test]
fn snap_conv2d_output_size_dilated() {
    let out = bitnet_kernels::cpu::compute_output_size(28, 3, 1, 0, 2);
    insta::assert_snapshot!(format!("input=28, kernel=3, stride=1, pad=0, dil=2 -> {out}"));
}

#[test]
fn snap_reduction_output_shape_axis0() {
    let config = bitnet_kernels::shaped_reduction::ShapedReductionConfig::new(
        bitnet_kernels::reduction::ReductionOp::Sum,
        Some(0),
        false,
    );
    let shape = bitnet_kernels::shaped_reduction::reduction_output_shape(&[4, 8, 16], &config);
    insta::assert_debug_snapshot!(shape);
}

#[test]
fn snap_reduction_output_shape_axis1_keepdim() {
    let config = bitnet_kernels::shaped_reduction::ShapedReductionConfig::new(
        bitnet_kernels::reduction::ReductionOp::Mean,
        Some(1),
        true,
    );
    let shape = bitnet_kernels::shaped_reduction::reduction_output_shape(&[4, 8, 16], &config);
    insta::assert_debug_snapshot!(shape);
}

#[test]
fn snap_reduction_output_shape_global() {
    let config = bitnet_kernels::shaped_reduction::ShapedReductionConfig::global(
        bitnet_kernels::reduction::ReductionOp::Max,
    );
    let shape = bitnet_kernels::shaped_reduction::reduction_output_shape(&[4, 8, 16], &config);
    insta::assert_debug_snapshot!(shape);
}

#[test]
fn snap_reduction_output_shape_global_keepdim() {
    let config = bitnet_kernels::shaped_reduction::ShapedReductionConfig::new(
        bitnet_kernels::reduction::ReductionOp::Sum,
        None,
        true,
    );
    let shape = bitnet_kernels::shaped_reduction::reduction_output_shape(&[2, 3, 4], &config);
    insta::assert_debug_snapshot!(shape);
}

// ── Metric / value computation snapshots ───────────────────────────

#[test]
fn snap_attention_resolved_scale_default() {
    let config = bitnet_kernels::cpu::AttentionConfig {
        num_heads: 8,
        head_dim: 64,
        seq_len: 128,
        causal: true,
        use_alibi: false,
        scale: None,
    };
    insta::assert_snapshot!(format!("{:.6}", config.resolved_scale()));
}

#[test]
fn snap_attention_resolved_scale_explicit() {
    let config = bitnet_kernels::cpu::AttentionConfig {
        num_heads: 8,
        head_dim: 64,
        seq_len: 128,
        causal: false,
        use_alibi: false,
        scale: Some(0.05),
    };
    insta::assert_snapshot!(format!("{:.6}", config.resolved_scale()));
}

#[test]
fn snap_scatter_reduce_identities() {
    use bitnet_kernels::cpu::scatter_gather::ScatterReduce;
    let identities: Vec<(String, String)> = vec![
        ScatterReduce::Assign,
        ScatterReduce::Add,
        ScatterReduce::Max,
        ScatterReduce::Min,
        ScatterReduce::Mul,
    ]
    .into_iter()
    .map(|r| (format!("{r:?}"), format!("{}", r.identity())))
    .collect();
    insta::assert_debug_snapshot!(identities);
}

#[test]
fn snap_scatter_mode_identities() {
    use bitnet_kernels::scatter_gather::ScatterMode;
    let identities: Vec<(String, String)> =
        vec![ScatterMode::Assign, ScatterMode::Add, ScatterMode::Max, ScatterMode::Min]
            .into_iter()
            .map(|m| (format!("{m:?}"), format!("{}", m.identity())))
            .collect();
    insta::assert_debug_snapshot!(identities);
}

#[test]
fn snap_kv_dtype_element_bytes() {
    use bitnet_kernels::cpu::kv_cache::KvDtype;
    let sizes: Vec<(String, usize)> = vec![KvDtype::F32, KvDtype::F16, KvDtype::Bf16]
        .into_iter()
        .map(|d| (format!("{d:?}"), d.element_bytes()))
        .collect();
    insta::assert_debug_snapshot!(sizes);
}

#[test]
fn snap_quantization_error_metrics() {
    let err = bitnet_kernels::cpu::quantize::QuantizationError {
        mse: 0.00042,
        max_abs_error: 0.031,
        snr: 33.8,
    };
    insta::assert_debug_snapshot!(err);
}

// ── Kernel output value snapshots ──────────────────────────────────

#[test]
fn snap_activation_relu_output() {
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let output =
        bitnet_kernels::cpu::apply_activation(&input, bitnet_kernels::cpu::ActivationType::ReLU);
    insta::assert_debug_snapshot!(output);
}

#[test]
fn snap_activation_silu_output() {
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let output =
        bitnet_kernels::cpu::apply_activation(&input, bitnet_kernels::cpu::ActivationType::SiLU);
    insta::assert_debug_snapshot!(output);
}

#[test]
fn snap_activation_gelu_output() {
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let output =
        bitnet_kernels::cpu::apply_activation(&input, bitnet_kernels::cpu::ActivationType::GELU);
    insta::assert_debug_snapshot!(output);
}

#[test]
fn snap_pooling_max_output() {
    let input = vec![1.0, 3.0, 2.0, 5.0, 4.0, 1.0, 3.0, 2.0];
    let config = bitnet_kernels::cpu::PoolConfig {
        pool_type: bitnet_kernels::cpu::PoolType::Max,
        kernel_size: 3,
        stride: 1,
        padding: 0,
        dilation: 1,
        ceil_mode: false,
    };
    let output = bitnet_kernels::cpu::PoolingKernel::apply(&input, &config).unwrap();
    insta::assert_debug_snapshot!(output);
}

#[test]
fn snap_pooling_global_avg_output() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let config = bitnet_kernels::cpu::PoolConfig {
        pool_type: bitnet_kernels::cpu::PoolType::GlobalAverage,
        kernel_size: 0,
        stride: 0,
        padding: 0,
        dilation: 1,
        ceil_mode: false,
    };
    let output = bitnet_kernels::cpu::PoolingKernel::apply(&input, &config).unwrap();
    insta::assert_debug_snapshot!(output);
}

#[test]
fn snap_reduction_sum_1d() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = bitnet_kernels::cpu::reduction::ReductionKernel::sum(&input).unwrap();
    insta::assert_snapshot!(format!("{result:.1}"));
}

#[test]
fn snap_reduction_mean_1d() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = bitnet_kernels::cpu::reduction::ReductionKernel::mean(&input).unwrap();
    insta::assert_snapshot!(format!("{result:.1}"));
}

#[test]
fn snap_reduction_max_with_index() {
    let input = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    let result = bitnet_kernels::cpu::reduction::ReductionKernel::max(&input).unwrap();
    insta::assert_debug_snapshot!(result);
}

#[test]
fn snap_reduction_min_with_index() {
    let input = vec![3.0, 1.0, 4.0, 0.5, 2.0];
    let result = bitnet_kernels::cpu::reduction::ReductionKernel::min(&input).unwrap();
    insta::assert_debug_snapshot!(result);
}

// ── State transition snapshots ─────────────────────────────────────

#[test]
fn snap_kv_cache_initial_state() {
    let config = bitnet_kernels::cpu::kv_cache::KvCacheConfig {
        num_layers: 2,
        num_heads: 4,
        head_dim: 8,
        max_seq_len: 16,
        dtype: bitnet_kernels::cpu::kv_cache::KvDtype::F32,
    };
    let cache = bitnet_kernels::cpu::kv_cache::KvCache::new(config).unwrap();
    insta::assert_snapshot!(format!(
        "layers={}, seq_len_per_layer={:?}",
        cache.blocks.len(),
        cache.blocks.iter().map(|b| b.seq_len).collect::<Vec<_>>()
    ));
}

#[test]
fn snap_quantize_symmetric_i8_small() {
    let input = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let (quantized, scale) = bitnet_kernels::cpu::quantize::quantize_symmetric_i8(&input, 8);
    insta::assert_debug_snapshot!((quantized, format!("{scale:.6}")));
}

#[test]
fn snap_quantize_symmetric_i8_zeros() {
    let input = vec![0.0, 0.0, 0.0, 0.0];
    let (quantized, scale) = bitnet_kernels::cpu::quantize::quantize_symmetric_i8(&input, 8);
    insta::assert_debug_snapshot!((quantized, format!("{scale:.1}")));
}

// ── Capability matrix profile snapshots ────────────────────────────

#[test]
fn snap_capability_matrix_builtin_profile_count() {
    let matrix = bitnet_kernels::capability_matrix::DeviceCapabilityMatrix::with_builtin_profiles();
    let names: Vec<&str> = matrix.profiles().iter().map(|p| p.name.as_str()).collect();
    insta::assert_debug_snapshot!(names);
}

#[test]
fn snap_compatibility_report_cpu_simd() {
    use bitnet_kernels::capability_matrix::*;
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    let profile = matrix.profile_for_class(DeviceClass::CpuSimd).unwrap();
    let required = vec![
        (OperationCategory::MatrixOps, PrecisionSupport::FP32),
        (OperationCategory::NormOps, PrecisionSupport::FP32),
        (OperationCategory::ActivationOps, PrecisionSupport::FP32),
    ];
    let report = CompatibilityReport::generate(profile, &required);
    insta::assert_snapshot!(report.summary());
}

#[test]
fn snap_compatibility_report_cpu_scalar_binary() {
    use bitnet_kernels::capability_matrix::*;
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    let profile = matrix.profile_for_class(DeviceClass::CpuScalar).unwrap();
    let required = vec![
        (OperationCategory::QuantizedOps, PrecisionSupport::Binary),
        (OperationCategory::AttentionOps, PrecisionSupport::FP32),
    ];
    let report = CompatibilityReport::generate(profile, &required);
    insta::assert_snapshot!(report.to_string());
}
