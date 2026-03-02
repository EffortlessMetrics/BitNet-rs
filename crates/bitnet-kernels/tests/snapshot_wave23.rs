//! Wave 23 snapshot tests — kernel configuration regression tests.
//!
//! Pins the Debug representations of OpenCL GQA, graph compiler, continuous
//! batching, prefix cache, numerical stability, mixed precision, token
//! generation, elementwise, reductions, matmul variants, quantized I2S,
//! KV cache, FFN, and softmax variant structs so that unintentional changes
//! are caught at review time.

// =========================================================================
// Section 1 — OpenCL GQA configs
// =========================================================================

use bitnet_kernels::opencl_gqa::{AttentionType, GqaConfig, GqaError, GqaStats, HeadGrouping};

#[test]
fn gqa_attention_type_all_variants() {
    let types: Vec<AttentionType> =
        vec![AttentionType::Mha, AttentionType::Gqa, AttentionType::Mqa];
    insta::assert_debug_snapshot!(types);
}

#[test]
fn gqa_config_standard_mha() {
    let cfg = GqaConfig::new(32, 32, 128, 4096).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn gqa_config_grouped_query() {
    let cfg = GqaConfig::new(32, 8, 128, 4096).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn gqa_config_multi_query() {
    let cfg = GqaConfig::new(32, 1, 128, 4096).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn gqa_error_all_variants() {
    let errors: Vec<GqaError> = vec![
        GqaError::ZeroDimension("head_dim".into()),
        GqaError::UnevenGrouping { num_q_heads: 32, num_kv_heads: 5 },
        GqaError::TooManyKvHeads { num_q_heads: 8, num_kv_heads: 16 },
        GqaError::BufferMismatch { expected: 4096, actual: 2048 },
        GqaError::SequenceTooLong { seq_len: 8192, max_seq_len: 4096 },
    ];
    insta::assert_debug_snapshot!(errors);
}

#[test]
fn gqa_head_grouping() {
    let cfg = GqaConfig::new(8, 2, 64, 2048).unwrap();
    let grouping = HeadGrouping::from_config(&cfg);
    insta::assert_debug_snapshot!(grouping);
}

#[test]
fn gqa_stats_compute() {
    let cfg = GqaConfig::new(32, 8, 128, 4096).unwrap();
    let stats = GqaStats::compute(&cfg, 512);
    insta::assert_debug_snapshot!(stats);
}

// =========================================================================
// Section 2 — OpenCL graph compiler configs
// =========================================================================

use bitnet_kernels::opencl_graph_compiler::{
    CompileError, DType, MemoryPlan, OpType, OptimizationPass, a770_fusion_patterns, cpu_add_node,
    create_compute_graph,
};
use std::collections::HashMap;

#[test]
fn graph_op_type_all_variants() {
    let ops: Vec<OpType> = vec![
        OpType::MatMul,
        OpType::Add,
        OpType::Softmax,
        OpType::RmsNorm,
        OpType::RoPE,
        OpType::SiLU,
        OpType::Mul,
        OpType::Transpose,
        OpType::Reshape,
        OpType::Concat,
        OpType::Split,
        OpType::Quantize,
        OpType::Dequantize,
    ];
    insta::assert_debug_snapshot!(ops);
}

#[test]
fn graph_dtype_all_variants() {
    let types: Vec<DType> = vec![DType::F32, DType::F16, DType::I8, DType::Ternary];
    insta::assert_debug_snapshot!(types);
}

#[test]
fn graph_two_node_chain() {
    let mut graph = create_compute_graph("test_graph");
    let n0 = cpu_add_node(&mut graph, OpType::MatMul, vec![], vec![1, 32, 2048]);
    let _n1 = cpu_add_node(&mut graph, OpType::RmsNorm, vec![n0], vec![1, 32, 2048]);
    insta::assert_debug_snapshot!(graph);
}

#[test]
fn graph_a770_fusion_patterns() {
    let patterns = a770_fusion_patterns();
    insta::assert_debug_snapshot!(patterns);
}

#[test]
fn graph_optimization_pass_all_variants() {
    let passes: Vec<OptimizationPass> = vec![
        OptimizationPass::DeadCodeElimination,
        OptimizationPass::ConstantFolding,
        OptimizationPass::OperatorFusion(vec![]),
        OptimizationPass::MemoryPlanning,
        OptimizationPass::LayoutOptimization,
    ];
    insta::assert_debug_snapshot!(passes);
}

#[test]
fn graph_memory_plan() {
    let plan = MemoryPlan {
        buffer_sizes: vec![4096, 8192, 2048],
        buffer_assignments: HashMap::from([(0, 0), (1, 1), (2, 0)]),
        peak_memory: 12288,
        reuse_count: 1,
    };
    // Sort buffer_assignments for deterministic output (HashMap order is random)
    let mut sorted: Vec<_> = plan.buffer_assignments.iter().collect();
    sorted.sort_by_key(|(k, _)| *k);
    insta::assert_debug_snapshot!((&plan.buffer_sizes, sorted, plan.peak_memory, plan.reuse_count));
}

#[test]
fn graph_compile_error_all_variants() {
    let errors: Vec<CompileError> = vec![
        CompileError::CyclicGraph,
        CompileError::ShapeMismatch {
            node_id: 5,
            expected: vec![1, 32, 128],
            got: vec![1, 32, 64],
        },
        CompileError::UnsupportedFusion("triple softmax".into()),
        CompileError::InvalidGraph("disconnected subgraph".into()),
    ];
    insta::assert_debug_snapshot!(errors);
}

// =========================================================================
// Section 3 — OpenCL continuous batching configs
// =========================================================================

use bitnet_kernels::opencl_continuous_batch::{
    BatchError, ContinuousBatchConfig, ContinuousBatchStats, IterationBatch, PreemptionPolicy,
    SlotManager,
};

#[test]
fn batch_config_snapshot() {
    let cfg =
        ContinuousBatchConfig { max_batch_size: 8, max_seq_len: 4096, iteration_timeout_ms: 500 };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn batch_slot_manager_empty() {
    let cfg =
        ContinuousBatchConfig { max_batch_size: 4, max_seq_len: 2048, iteration_timeout_ms: 100 };
    let mgr = SlotManager::new(cfg);
    insta::assert_debug_snapshot!(mgr);
}

#[test]
fn batch_error_all_variants() {
    let errors: Vec<BatchError> = vec![
        BatchError::BatchFull,
        BatchError::SlotNotFound(3),
        BatchError::RequestNotFound(42),
        BatchError::PreemptionFailed,
        BatchError::ConfigError("max_tokens must be > 0".into()),
    ];
    insta::assert_debug_snapshot!(errors);
}

#[test]
fn batch_preemption_policy_all_variants() {
    let policies: Vec<PreemptionPolicy> = vec![
        PreemptionPolicy::Disabled,
        PreemptionPolicy::LowestPriority,
        PreemptionPolicy::ShortestGeneration,
    ];
    insta::assert_debug_snapshot!(policies);
}

#[test]
fn batch_iteration_batch_default() {
    let batch = IterationBatch::default();
    insta::assert_debug_snapshot!(batch);
}

#[test]
fn batch_stats_default() {
    let stats = ContinuousBatchStats::default();
    insta::assert_debug_snapshot!(stats);
}

// =========================================================================
// Section 4 — OpenCL prefix cache configs
// =========================================================================

use bitnet_kernels::opencl_prefix_cache::{
    EvictionPolicy, PrefixCacheConfig, PrefixCacheError, PrefixTree, SharedKvRef, SystemPromptPin,
};

#[test]
fn prefix_cache_config_default() {
    let cfg = PrefixCacheConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn prefix_cache_error_all_variants() {
    let errors: Vec<PrefixCacheError> = vec![
        PrefixCacheError::CacheFull { max_entries: 1024 },
        PrefixCacheError::PrefixTooLong { len: 10000, max: 4096 },
        PrefixCacheError::RefcountNonZero { entry_id: 3 },
        PrefixCacheError::EntryNotFound { entry_id: 99 },
        PrefixCacheError::DuplicatePin,
    ];
    insta::assert_debug_snapshot!(errors);
}

#[test]
fn prefix_eviction_policy() {
    let policy = EvictionPolicy::Lru;
    insta::assert_debug_snapshot!(policy);
}

#[test]
fn prefix_shared_kv_ref() {
    let kv_ref = SharedKvRef { entry_id: 42, token_len: 128 };
    insta::assert_debug_snapshot!(kv_ref);
}

#[test]
fn prefix_system_prompt_pin() {
    let pin = SystemPromptPin { tokens: vec![1, 2, 3, 128000], entry_id: 0 };
    insta::assert_debug_snapshot!(pin);
}

#[test]
fn prefix_stats_empty_tree() {
    let cfg = PrefixCacheConfig::default();
    let tree = PrefixTree::new(cfg);
    insta::assert_debug_snapshot!(tree.stats());
}

// =========================================================================
// Section 5 — OpenCL numerical stability configs
// =========================================================================

use bitnet_kernels::opencl_numerical_stability::{InfDetector, NanDetector, StabilityConfig};

#[test]
fn stability_config_default() {
    let cfg = StabilityConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn nan_detector_clean() {
    let data = vec![1.0, 2.0, 3.0];
    let detector = NanDetector::scan(&data);
    insta::assert_debug_snapshot!(detector);
}

#[test]
fn nan_detector_with_nans() {
    let data = vec![1.0, f32::NAN, 3.0, f32::NAN];
    let detector = NanDetector::scan(&data);
    insta::assert_snapshot!(format!("{:?}", detector));
}

#[test]
fn inf_detector_clean() {
    let data = vec![1.0, 2.0, 3.0];
    let detector = InfDetector::scan(&data);
    insta::assert_debug_snapshot!(detector);
}

#[test]
fn inf_detector_with_infs() {
    let data = vec![f32::INFINITY, 2.0, f32::NEG_INFINITY];
    let detector = InfDetector::scan(&data);
    insta::assert_snapshot!(format!("{:?}", detector));
}

// =========================================================================
// Section 6 — OpenCL mixed precision configs
// =========================================================================

use bitnet_kernels::opencl_mixed_precision::{
    CastOp, LayerKind, MixedPrecisionMatmul, Precision, PrecisionPolicy, RoundingMode,
};

#[test]
fn precision_all_variants() {
    let variants: Vec<Precision> = vec![
        Precision::F32,
        Precision::F16,
        Precision::BF16,
        Precision::I8,
        Precision::I4,
        Precision::I2,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn layer_kind_all_variants() {
    let kinds: Vec<LayerKind> = vec![
        LayerKind::Attention,
        LayerKind::FeedForward,
        LayerKind::Embedding,
        LayerKind::LayerNorm,
        LayerKind::Output,
    ];
    insta::assert_debug_snapshot!(kinds);
}

#[test]
fn rounding_mode_all_variants() {
    let modes: Vec<RoundingMode> =
        vec![RoundingMode::NearestEven, RoundingMode::Truncate, RoundingMode::Stochastic];
    insta::assert_debug_snapshot!(modes);
}

#[test]
fn precision_policy_default() {
    let policy = PrecisionPolicy::default();
    insta::assert_debug_snapshot!(policy);
}

#[test]
fn precision_policy_uniform() {
    let policy = PrecisionPolicy::uniform(Precision::F16);
    insta::assert_debug_snapshot!(policy);
}

#[test]
fn cast_op_f32_to_f16() {
    let op = CastOp::new(Precision::F32, Precision::F16);
    insta::assert_debug_snapshot!(op);
}

#[test]
fn mixed_precision_matmul_i8_i2() {
    let mm = MixedPrecisionMatmul::new(Precision::I8, Precision::I2);
    insta::assert_debug_snapshot!(mm);
}

// =========================================================================
// Section 7 — OpenCL token generation configs
// =========================================================================

use bitnet_kernels::opencl_token_gen::{
    GenerationConfig, GenerationError, GenerationState, GenerationStats, SamplingMethod, StopReason,
};

#[test]
fn generation_config_default() {
    let cfg = GenerationConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn sampling_method_all_variants() {
    let methods: Vec<SamplingMethod> = vec![
        SamplingMethod::Greedy,
        SamplingMethod::Temperature(0.7),
        SamplingMethod::TopK(50),
        SamplingMethod::TopP(0.95),
        SamplingMethod::TopKP(50, 0.95),
    ];
    insta::assert_debug_snapshot!(methods);
}

#[test]
fn stop_reason_all_variants() {
    let reasons: Vec<StopReason> = vec![
        StopReason::MaxTokens,
        StopReason::StopToken(128009),
        StopReason::EndOfSequence,
        StopReason::Error("numerical overflow".into()),
    ];
    insta::assert_debug_snapshot!(reasons);
}

#[test]
fn generation_error_all_variants() {
    let errors: Vec<GenerationError> = vec![
        GenerationError::InvalidConfig("temperature must be > 0".into()),
        GenerationError::EmptyPrompt,
        GenerationError::VocabTooSmall,
        GenerationError::NumericalError("logits contain NaN".into()),
        GenerationError::MaxTokensExceeded,
    ];
    insta::assert_debug_snapshot!(errors);
}

#[test]
fn generation_state_default() {
    let state = GenerationState::default();
    insta::assert_debug_snapshot!(state);
}

#[test]
fn generation_stats_snapshot() {
    let stats = GenerationStats {
        total_tokens: 64,
        prefill_time_us: 12000,
        decode_time_us: 48000,
        tokens_per_second: 1333.3,
    };
    insta::assert_debug_snapshot!(stats);
}

// =========================================================================
// Section 8 — OpenCL elementwise configs
// =========================================================================

use bitnet_kernels::opencl_elementwise::{BroadcastRule, ElemOp, ElemwiseError};

#[test]
fn elem_op_all_variants() {
    let ops: Vec<ElemOp> = vec![
        ElemOp::Add,
        ElemOp::Sub,
        ElemOp::Mul,
        ElemOp::Div,
        ElemOp::Scale,
        ElemOp::Residual,
        ElemOp::Max,
        ElemOp::Min,
        ElemOp::Abs,
    ];
    insta::assert_debug_snapshot!(ops);
}

#[test]
fn elemwise_error_all_variants() {
    let errors: Vec<ElemwiseError> = vec![
        ElemwiseError::ShapeMismatch { a: vec![2, 3], b: vec![4, 5] },
        ElemwiseError::ZeroDimension,
        ElemwiseError::DivisionByZero,
        ElemwiseError::InvalidClampBounds { min: 1.0, max: 0.5 },
        ElemwiseError::DataShapeMismatch { expected: 100, actual: 50 },
    ];
    insta::assert_debug_snapshot!(errors);
}

#[test]
fn broadcast_rule_same_shape() {
    let rule = BroadcastRule::new(&[2, 3], &[2, 3]).unwrap();
    insta::assert_debug_snapshot!(rule);
}

// =========================================================================
// Section 9 — OpenCL reduction configs
// =========================================================================

use bitnet_kernels::opencl_reductions::{ReduceConfig, ReduceDtype, ReduceOp};

#[test]
fn reduce_op_all_variants() {
    let ops: Vec<ReduceOp> = vec![
        ReduceOp::Sum,
        ReduceOp::Max,
        ReduceOp::Min,
        ReduceOp::Mean,
        ReduceOp::Variance,
        ReduceOp::ArgMax,
        ReduceOp::ArgMin,
        ReduceOp::Prod,
        ReduceOp::LogSumExp,
    ];
    insta::assert_debug_snapshot!(ops);
}

#[test]
fn reduce_dtype_all_variants() {
    let dtypes: Vec<ReduceDtype> = vec![ReduceDtype::F32, ReduceDtype::F16, ReduceDtype::I32];
    insta::assert_debug_snapshot!(dtypes);
}

#[test]
fn reduce_config_global_sum() {
    let cfg = ReduceConfig::global(ReduceOp::Sum);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn reduce_config_axis_keepdims() {
    let cfg = ReduceConfig::new(ReduceOp::Mean)
        .with_axis(1)
        .with_keepdims(true)
        .with_dtype(ReduceDtype::F16);
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 10 — OpenCL matmul variant configs
// =========================================================================

use bitnet_kernels::opencl_matmul_variants::{
    MatmulConfig as OclMatmulConfig, MatmulError, MatmulStrategy,
};

#[test]
fn matmul_strategy_all_variants() {
    let strategies: Vec<MatmulStrategy> = vec![
        MatmulStrategy::Naive,
        MatmulStrategy::Tiled,
        MatmulStrategy::Vectorized,
        MatmulStrategy::SubgroupTiled,
        MatmulStrategy::BatchedGemm,
    ];
    insta::assert_debug_snapshot!(strategies);
}

#[test]
fn matmul_config_basic() {
    let cfg = OclMatmulConfig::new(1024, 2048, 512);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn matmul_config_with_tiles_and_transpose() {
    let cfg =
        OclMatmulConfig::new(512, 512, 256).with_tiles(32, 32, 16).with_transpose(false, true);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn matmul_error_all_variants() {
    let errors: Vec<MatmulError> = vec![
        MatmulError::DimensionMismatch { expected: 1024, got: 512, dim: "k" },
        MatmulError::InvalidTileSize { tile: 64, dim: 32, name: "m" },
        MatmulError::EmptyMatrix,
        MatmulError::BatchSizeMismatch { expected: 8, got: 4 },
    ];
    insta::assert_debug_snapshot!(errors);
}

// =========================================================================
// Section 11 — OpenCL quantized I2S configs
// =========================================================================

use bitnet_kernels::opencl_quantized::{I2sBlockLayout, I2sPackedFormat, I2sScaleFormat};

#[test]
fn i2s_packed_format_default() {
    let fmt = I2sPackedFormat::default();
    insta::assert_debug_snapshot!(fmt);
}

#[test]
fn i2s_scale_format_all_variants() {
    let formats: Vec<I2sScaleFormat> = vec![I2sScaleFormat::Fp32, I2sScaleFormat::Fp16];
    insta::assert_debug_snapshot!(formats);
}

#[test]
fn i2s_block_layout_all_variants() {
    let layouts: Vec<I2sBlockLayout> =
        vec![I2sBlockLayout::BitNet32F16, I2sBlockLayout::Qk256, I2sBlockLayout::Custom(128)];
    insta::assert_debug_snapshot!(layouts);
}

// =========================================================================
// Section 12 — OpenCL KV cache configs
// =========================================================================

use bitnet_kernels::opencl_kv_cache::{KvCacheConfig as OclKvCacheConfig, KvCacheError};

#[test]
fn kv_cache_error_all_variants() {
    let errors: Vec<KvCacheError> = vec![
        KvCacheError::LayerOutOfBounds { requested: 24, available: 12 },
        KvCacheError::CacheFull { max_len: 4096 },
        KvCacheError::DimensionMismatch { expected: 2048, got: 1024 },
    ];
    insta::assert_debug_snapshot!(errors);
}

#[test]
fn kv_cache_config_snapshot() {
    let cfg = OclKvCacheConfig {
        max_seq_len: 4096,
        num_heads: 32,
        head_dim: 128,
        num_layers: 24,
        dtype_bytes: 4,
    };
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 13 — OpenCL FFN configs
// =========================================================================

use bitnet_kernels::opencl_ffn::{ActivationType as OclActivationType, FfnConfig as OclFfnConfig};

#[test]
fn ocl_activation_type_all_variants() {
    let variants: Vec<OclActivationType> = vec![
        OclActivationType::SiLU,
        OclActivationType::GELU,
        OclActivationType::GELUApprox,
        OclActivationType::ReLU,
        OclActivationType::Swish,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn ocl_ffn_config_snapshot() {
    let cfg = OclFfnConfig::new(2048, 5632, OclActivationType::SiLU);
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 14 — OpenCL softmax variant configs
// =========================================================================

use bitnet_kernels::opencl_softmax_variants::SoftmaxConfig as OclSoftmaxConfig;

#[test]
fn ocl_softmax_config_default() {
    let cfg = OclSoftmaxConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn ocl_softmax_config_with_topk_and_temp() {
    let cfg = OclSoftmaxConfig::new().with_temperature(0.7).with_top_k(50).with_axis(1);
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 15 — CPU gating configs
// =========================================================================

use bitnet_kernels::cpu::gating::GatingType as CpuGatingType;

#[test]
fn cpu_gating_type_all_variants() {
    let variants: Vec<CpuGatingType> =
        vec![CpuGatingType::SwiGLU, CpuGatingType::GeGLU, CpuGatingType::ReGLU];
    insta::assert_debug_snapshot!(variants);
}
