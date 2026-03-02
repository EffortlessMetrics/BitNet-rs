//! Wave 22 snapshot tests — kernel configuration regression tests.
//!
//! Pins the Debug representations of CUDA stream management, sparse matrix,
//! graph execution, fused attention, profiling, dequantization, cooperative
//! groups, CPU pipeline/tensor parallelism, capability matrix, and OpenCL
//! autotuner / telemetry / registry / cache structs so that unintentional
//! changes are caught at review time.

// =========================================================================
// Section 1 — CUDA stream management configs
// =========================================================================

use bitnet_kernels::cuda::stream_mgmt::{
    DefaultStreamBehavior, PipelineStageKind, ScheduleStrategy, StreamConfig, StreamOp,
    StreamPriority, StreamScheduler,
};

#[test]
fn stream_config_default() {
    let cfg = StreamConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn stream_config_high_priority_profiling() {
    let cfg = StreamConfig {
        num_streams: 8,
        priority: StreamPriority::High,
        default_stream_behavior: DefaultStreamBehavior::Legacy,
        enable_profiling: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn stream_priority_all_variants() {
    let variants: Vec<StreamPriority> =
        vec![StreamPriority::Low, StreamPriority::Normal, StreamPriority::High];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn default_stream_behavior_all_variants() {
    let variants: Vec<DefaultStreamBehavior> =
        vec![DefaultStreamBehavior::Legacy, DefaultStreamBehavior::PerThread];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn pipeline_stage_kind_all_variants() {
    let variants: Vec<PipelineStageKind> = vec![
        PipelineStageKind::HostToDevice,
        PipelineStageKind::Compute,
        PipelineStageKind::DeviceToHost,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn stream_op_basic() {
    let op = StreamOp::new("matmul_kernel", 1000);
    insta::assert_debug_snapshot!(op);
}

#[test]
fn stream_op_with_dependency() {
    let op = StreamOp::new("softmax_kernel", 500).with_dependency(42);
    insta::assert_debug_snapshot!(op);
}

#[test]
fn schedule_strategy_all_variants() {
    let variants: Vec<ScheduleStrategy> = vec![
        ScheduleStrategy::RoundRobin,
        ScheduleStrategy::LeastLoaded,
        ScheduleStrategy::PriorityBased,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn stream_scheduler_round_robin() {
    let scheduler = StreamScheduler::new(ScheduleStrategy::RoundRobin);
    insta::assert_debug_snapshot!(scheduler);
}

// =========================================================================
// Section 2 — CUDA sparse matrix configs
// =========================================================================

use bitnet_kernels::cuda::sparse::{ElementwiseSpOp, SparseConfig, SparseFormat};

#[test]
fn sparse_format_all_variants() {
    let formats: Vec<SparseFormat> = vec![
        SparseFormat::CSR,
        SparseFormat::CSC,
        SparseFormat::COO,
        SparseFormat::BSR,
        SparseFormat::Block,
    ];
    insta::assert_debug_snapshot!(formats);
}

#[test]
fn sparse_config_csr_basic() {
    let cfg = SparseConfig::new(SparseFormat::CSR, 1024, 512).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn sparse_config_bsr_with_block_size() {
    let cfg = SparseConfig::new(SparseFormat::BSR, 2048, 2048)
        .unwrap()
        .with_block_size(64)
        .unwrap()
        .with_threshold(0.01);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn elementwise_sp_op_all_variants() {
    let ops: Vec<ElementwiseSpOp> =
        vec![ElementwiseSpOp::Add, ElementwiseSpOp::Sub, ElementwiseSpOp::Mul];
    insta::assert_debug_snapshot!(ops);
}

// =========================================================================
// Section 3 — CUDA graph execution configs
// =========================================================================

use bitnet_kernels::cuda::graph_exec::{
    CaptureState, GraphNode, LayerGraphConfig, MultiStreamConfig, NodeKind, OptimizeStats,
};

#[test]
fn node_kind_kernel() {
    let kind = NodeKind::Kernel { name: "gemm_i2s".into(), grid: [128, 1, 1], block: [256, 1, 1] };
    insta::assert_debug_snapshot!(kind);
}

#[test]
fn node_kind_all_simple_variants() {
    let kinds: Vec<NodeKind> = vec![
        NodeKind::MemCopy { bytes: 4096 },
        NodeKind::MemSet { bytes: 1024 },
        NodeKind::HostCallback { label: "sync_point".into() },
        NodeKind::Barrier,
        NodeKind::Empty,
    ];
    insta::assert_debug_snapshot!(kinds);
}

#[test]
fn graph_node_kernel_with_params() {
    let node = GraphNode::kernel("rmsnorm_fwd", [64, 1, 1], [256, 1, 1])
        .with_param("eps", 1e-6)
        .on_stream(0);
    insta::assert_debug_snapshot!(node);
}

#[test]
fn graph_node_memcopy() {
    let node = GraphNode::memcopy(65536).on_stream(1);
    insta::assert_debug_snapshot!(node);
}

#[test]
fn capture_state_all_variants() {
    let states: Vec<CaptureState> =
        vec![CaptureState::Idle, CaptureState::Capturing, CaptureState::Complete];
    insta::assert_debug_snapshot!(states);
}

#[test]
fn layer_graph_config_default() {
    let cfg = LayerGraphConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn multi_stream_config_default() {
    let cfg = MultiStreamConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn optimize_stats_default() {
    let stats = OptimizeStats::default();
    insta::assert_debug_snapshot!(stats);
}

// =========================================================================
// Section 4 — CUDA fused attention configs
// =========================================================================

use bitnet_kernels::cuda::fused_attention::{
    AttentionMetrics, AttentionPattern, FusedAttentionConfig, FusedAttentionError,
};

#[test]
fn fused_attention_config_basic() {
    let cfg = FusedAttentionConfig::new(64, 8, 8, 2048).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn fused_attention_config_gqa_flash() {
    let cfg = FusedAttentionConfig::new(128, 32, 8, 4096)
        .unwrap()
        .with_causal(true)
        .with_flash_attention(true);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn attention_pattern_all_variants() {
    let patterns: Vec<AttentionPattern> = vec![
        AttentionPattern::Causal,
        AttentionPattern::Full,
        AttentionPattern::SlidingWindow { window_size: 256 },
        AttentionPattern::Sparse { block_size: 64 },
    ];
    insta::assert_debug_snapshot!(patterns);
}

#[test]
fn fused_attention_error_display() {
    let errors: Vec<String> = vec![
        format!("{:?}", FusedAttentionError::InvalidConfig("head_dim must be > 0".into())),
        format!(
            "{:?}",
            FusedAttentionError::ShapeMismatch {
                expected: "[1,8,64,64]".into(),
                actual: "[1,8,64,32]".into(),
            }
        ),
        format!("{:?}", FusedAttentionError::SequenceTooLong { seq_len: 8192, max_seq_len: 4096 }),
        format!("{:?}", FusedAttentionError::InvalidGqaRatio { num_heads: 8, num_kv_heads: 3 }),
    ];
    insta::assert_debug_snapshot!(errors);
}

#[test]
fn attention_metrics_compute() {
    let m = AttentionMetrics::compute(8, 128, 128, 64);
    insta::assert_debug_snapshot!(m);
}

#[test]
fn attention_metrics_compute_gqa() {
    let m = AttentionMetrics::compute_gqa(32, 8, 256, 256, 128);
    insta::assert_debug_snapshot!(m);
}

// =========================================================================
// Section 5 — CUDA dequantization configs
// =========================================================================

use bitnet_kernels::cuda::dequant::{DequantConfig, DequantPrecision, QuantBitWidth, ScaleMode};

#[test]
fn dequant_config_default() {
    let cfg = DequantConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn dequant_precision_all_variants() {
    let variants: Vec<DequantPrecision> = vec![DequantPrecision::F32, DequantPrecision::F16];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn quant_bit_width_all_variants() {
    let variants: Vec<QuantBitWidth> =
        vec![QuantBitWidth::Int2, QuantBitWidth::Int4, QuantBitWidth::Int8];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn scale_mode_all_variants() {
    let variants: Vec<ScaleMode> =
        vec![ScaleMode::Uniform, ScaleMode::PerBlock, ScaleMode::PerChannel];
    insta::assert_debug_snapshot!(variants);
}

// =========================================================================
// Section 6 — CUDA loss configs
// =========================================================================

use bitnet_kernels::cuda::loss::{LossConfig, LossReduction as CudaLossReduction};

#[test]
fn cuda_loss_config_basic() {
    let cfg = LossConfig::new(64, 32000).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cuda_loss_reduction_all_variants() {
    let variants: Vec<CudaLossReduction> =
        vec![CudaLossReduction::Mean, CudaLossReduction::Sum, CudaLossReduction::None];
    insta::assert_debug_snapshot!(variants);
}

// =========================================================================
// Section 7 — CUDA cooperative groups configs
// =========================================================================

use bitnet_kernels::cuda::cooperative_groups::{
    CoalescedGroup, CooperativeGroupConfig, CooperativeReduceOp, GridGroup, ThreadBlockGroup,
};

#[test]
fn cooperative_group_config_default() {
    let cfg = CooperativeGroupConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cooperative_reduce_op_all_variants() {
    let ops: Vec<CooperativeReduceOp> = vec![
        CooperativeReduceOp::Sum,
        CooperativeReduceOp::Max,
        CooperativeReduceOp::Min,
        CooperativeReduceOp::Product,
    ];
    insta::assert_debug_snapshot!(ops);
}

#[test]
fn thread_block_group_debug() {
    let g = ThreadBlockGroup::new(256, 0).unwrap();
    insta::assert_debug_snapshot!(g);
}

#[test]
fn grid_group_debug() {
    let g = GridGroup::new(256, 128).unwrap();
    insta::assert_debug_snapshot!(g);
}

#[test]
fn coalesced_group_debug() {
    let g = CoalescedGroup::new(0xFFFF_FFFF).unwrap();
    insta::assert_debug_snapshot!(g);
}

// =========================================================================
// Section 8 — CUDA warp ops config
// =========================================================================

use bitnet_kernels::cuda::warp_ops::WarpConfig;

#[test]
fn warp_config_default() {
    let cfg = WarpConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 9 — CUDA profiling configs
// =========================================================================

use bitnet_kernels::cuda::profiling::{
    Bottleneck, BottleneckAnalyzer, OccupancyCalculator, OccupancyLimiter,
};

#[test]
fn occupancy_calculator_sm80() {
    let calc = OccupancyCalculator::new(2048, 32, 163_840, 65536);
    insta::assert_debug_snapshot!(calc);
}

#[test]
fn occupancy_limiter_all_variants() {
    let limiters: Vec<OccupancyLimiter> = vec![
        OccupancyLimiter::Threads,
        OccupancyLimiter::Blocks,
        OccupancyLimiter::SharedMemory,
        OccupancyLimiter::Registers,
    ];
    insta::assert_debug_snapshot!(limiters);
}

#[test]
fn bottleneck_all_variants() {
    let variants: Vec<Bottleneck> = vec![
        Bottleneck::MemoryBound,
        Bottleneck::ComputeBound,
        Bottleneck::LatencyBound,
        Bottleneck::Unknown,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn bottleneck_analyzer_default() {
    let analyzer = BottleneckAnalyzer::new();
    insta::assert_debug_snapshot!(analyzer);
}

// =========================================================================
// Section 10 — CPU pipeline parallel configs
// =========================================================================

use bitnet_kernels::cpu::pipeline_parallel::{PipelineConfig, PipelineSchedule, PipelineStage};

#[test]
fn pipeline_schedule_all_variants() {
    let variants: Vec<PipelineSchedule> = vec![
        PipelineSchedule::Sequential,
        PipelineSchedule::GPipe,
        PipelineSchedule::PipeDream,
        PipelineSchedule::Interleaved,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn pipeline_stage_with_affinity() {
    let stage = PipelineStage::new(0, 8).with_affinity(2);
    insta::assert_debug_snapshot!(stage);
}

#[test]
fn pipeline_config_two_stage_gpipe() {
    let cfg = PipelineConfig::new(
        vec![PipelineStage::new(0, 16), PipelineStage::new(16, 32)],
        4,
        PipelineSchedule::GPipe,
    );
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 11 — CPU tensor parallel configs
// =========================================================================

use bitnet_kernels::cpu::tensor_parallel::{
    CommBackend, ShardingStrategy, TensorParallelConfig, TensorParallelError, TensorParallelMetrics,
};

#[test]
fn comm_backend_all_variants() {
    let variants: Vec<CommBackend> = vec![CommBackend::InProcess, CommBackend::SharedMemory];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn tensor_parallel_config_debug() {
    let cfg = TensorParallelConfig {
        num_ranks: 4,
        rank_id: 0,
        comm_backend: CommBackend::InProcess,
        overlap_compute_comm: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn sharding_strategy_all_variants() {
    let variants: Vec<ShardingStrategy> = vec![
        ShardingStrategy::ColumnParallel,
        ShardingStrategy::RowParallel,
        ShardingStrategy::Custom { splits: vec![128, 256, 128] },
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn tensor_parallel_error_variants() {
    let errors: Vec<TensorParallelError> = vec![
        TensorParallelError::InvalidConfig("num_ranks must be > 0".into()),
        TensorParallelError::UnevenSharding { tensor_len: 100, num_shards: 3 },
        TensorParallelError::ShardIndexOutOfBounds { index: 5, total: 4 },
    ];
    insta::assert_debug_snapshot!(errors);
}

#[test]
fn tensor_parallel_metrics_default() {
    let m = TensorParallelMetrics::default();
    insta::assert_debug_snapshot!(m);
}

// =========================================================================
// Section 12 — CPU cache-aware matmul configs
// =========================================================================

use bitnet_kernels::cpu::cache_matmul::{CacheConfig, TilingStrategy};

#[test]
fn cache_config_conservative() {
    let cfg = CacheConfig::conservative();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cache_config_custom_sizes() {
    let cfg = CacheConfig::with_sizes(32768, 262144, 8388608, 64);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn tiling_strategy_debug() {
    let ts = TilingStrategy { block_m: 64, block_n: 128, block_k: 32 };
    insta::assert_debug_snapshot!(ts);
}

// =========================================================================
// Section 13 — CPU pooling configs
// =========================================================================

use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType};

#[test]
fn pool_type_all_variants() {
    let types: Vec<PoolType> = vec![
        PoolType::Max,
        PoolType::Average,
        PoolType::GlobalMax,
        PoolType::GlobalAverage,
        PoolType::AvgPoolCountIncludePad,
        PoolType::Lp(2.0),
        PoolType::Adaptive,
    ];
    insta::assert_debug_snapshot!(types);
}

#[test]
fn pool_config_default() {
    let cfg = PoolConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn pool_config_max_pool_3x3() {
    let cfg = PoolConfig::new(PoolType::Max, 3, 2, 1);
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 14 — CPU linear config
// =========================================================================

use bitnet_kernels::cpu::linear::LinearConfig;

#[test]
fn linear_config_basic() {
    let cfg = LinearConfig::new(1, 768, 3072).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn linear_config_with_bias() {
    let cfg = LinearConfig::new(4, 2048, 5504).unwrap().with_bias(true);
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 15 — Capability matrix configs
// =========================================================================

use bitnet_kernels::capability_matrix::{
    CapabilityEntry, DeviceClass, OperationCategory, PrecisionSupport, SupportLevel,
};

#[test]
fn device_class_all_variants() {
    let classes: Vec<DeviceClass> = vec![
        DeviceClass::IntelArc,
        DeviceClass::IntelXe,
        DeviceClass::NvidiaCuda,
        DeviceClass::AmdRocm,
        DeviceClass::AppleMetal,
        DeviceClass::CpuSimd,
        DeviceClass::CpuScalar,
        DeviceClass::WebGpu,
    ];
    insta::assert_debug_snapshot!(classes);
}

#[test]
fn support_level_all_variants() {
    let levels: Vec<SupportLevel> = vec![
        SupportLevel::Full(0.95),
        SupportLevel::Partial("requires fallback for large matrices".into()),
        SupportLevel::Emulated,
        SupportLevel::Unsupported,
    ];
    insta::assert_debug_snapshot!(levels);
}

#[test]
fn capability_entry_full_support() {
    let entry = CapabilityEntry::new(
        OperationCategory::MatrixOps,
        PrecisionSupport::FP32,
        SupportLevel::Full(1.0),
    );
    insta::assert_debug_snapshot!(entry);
}

// =========================================================================
// Section 16 — OpenCL autotuner configs
// =========================================================================

use bitnet_kernels::opencl_autotuner::{
    BenchmarkResult as TuningBenchmarkResult, ParamSet, SearchStrategy, TuningCacheKey, TuningParam,
};

#[test]
fn tuning_param_debug() {
    let p = TuningParam::new("workgroup_size", 32, 512, 32, 256);
    insta::assert_debug_snapshot!(p);
}

#[test]
fn search_strategy_all_variants() {
    let variants: Vec<SearchStrategy> = vec![
        SearchStrategy::Exhaustive,
        SearchStrategy::RandomSample(100),
        SearchStrategy::SimulatedAnnealing {
            initial_temp: 1.0,
            cooling_rate: 0.95,
            iterations: 500,
        },
        SearchStrategy::BayesianOpt { evaluations: 50 },
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn tuning_benchmark_result_debug() {
    let r = TuningBenchmarkResult::new(
        ParamSet(vec![("wg_size".into(), 256), ("tile_m".into(), 16)]),
        42.5,
        120.0,
        800.0,
    );
    insta::assert_debug_snapshot!(r);
}

#[test]
fn tuning_cache_key_debug() {
    let key = TuningCacheKey::new("gemm_f32", "Intel Arc A770", vec![1024, 1024, 1024]);
    insta::assert_debug_snapshot!(key);
}

// =========================================================================
// Section 17 — OpenCL telemetry configs
// =========================================================================

use bitnet_kernels::opencl_telemetry::{KernelMetrics, Metric, MetricKind, TelemetryConfig};

#[test]
fn metric_kind_all_variants() {
    let kinds: Vec<MetricKind> =
        vec![MetricKind::Counter, MetricKind::Gauge, MetricKind::Histogram, MetricKind::Timer];
    insta::assert_debug_snapshot!(kinds);
}

#[test]
fn metric_counter_debug() {
    let m = Metric::new("kernel_dispatches", MetricKind::Counter, 42.0);
    // Filter the timestamp which varies per run (spans multiple lines in Debug)
    insta::with_settings!({filters => vec![
        (r"tv_sec: \d+", "tv_sec: [FILTERED]"),
        (r"tv_nsec: \d+", "tv_nsec: [FILTERED]"),
    ]}, {
        insta::assert_debug_snapshot!(m);
    });
}

#[test]
fn kernel_metrics_fresh() {
    let m = KernelMetrics::new("gemm_i2s");
    insta::assert_debug_snapshot!(m);
}

#[test]
fn telemetry_config_default() {
    let cfg = TelemetryConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 18 — OpenCL kernel registry configs
// =========================================================================

use bitnet_kernels::opencl_registry::{DeviceConstraints, KernelOp, KernelVariant};

#[test]
fn kernel_op_all_variants() {
    let ops: Vec<KernelOp> = vec![
        KernelOp::MatMul,
        KernelOp::MatVec,
        KernelOp::Softmax,
        KernelOp::RmsNorm,
        KernelOp::LayerNorm,
        KernelOp::RoPE,
        KernelOp::Attention,
        KernelOp::SiLU,
        KernelOp::GELU,
        KernelOp::ReLU,
        KernelOp::ElementwiseAdd,
        KernelOp::ElementwiseMul,
        KernelOp::Scale,
        KernelOp::Embedding,
        KernelOp::Dequantize,
        KernelOp::KvCacheAppend,
    ];
    insta::assert_debug_snapshot!(ops);
}

#[test]
fn kernel_variant_all_variants() {
    let variants: Vec<KernelVariant> = vec![
        KernelVariant::OpenClScalar,
        KernelVariant::OpenClTiled,
        KernelVariant::OpenClVectorized,
        KernelVariant::CpuSimd,
        KernelVariant::CpuScalar,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn device_constraints_a770() {
    let dc = DeviceConstraints::a770_defaults();
    insta::assert_debug_snapshot!(dc);
}

// =========================================================================
// Section 19 — OpenCL cache configs
// =========================================================================

use bitnet_kernels::opencl_cache::{
    CacheConfig as OclCacheConfig, CacheEvictionStrategy, CachePolicy, CacheStats as OclCacheStats,
};

#[test]
fn cache_policy_all_variants() {
    let policies: Vec<CachePolicy> = vec![
        CachePolicy::NoCache,
        CachePolicy::MemoryOnly,
        CachePolicy::DiskOnly,
        CachePolicy::MemoryAndDisk,
    ];
    insta::assert_debug_snapshot!(policies);
}

#[test]
fn cache_eviction_strategy_all_variants() {
    let strategies: Vec<CacheEvictionStrategy> = vec![
        CacheEvictionStrategy::Lru,
        CacheEvictionStrategy::Lfu,
        CacheEvictionStrategy::Fifo,
        CacheEvictionStrategy::SizeWeighted,
    ];
    insta::assert_debug_snapshot!(strategies);
}

#[test]
fn ocl_cache_config_default() {
    let cfg = OclCacheConfig::default();
    // Filter the cache directory path which varies per system
    insta::with_settings!({filters => vec![
        (r#"cache_dir: ".*""#, r#"cache_dir: "[CACHE_DIR]""#),
    ]}, {
        insta::assert_debug_snapshot!(cfg);
    });
}

#[test]
fn ocl_cache_stats_default() {
    let stats = OclCacheStats::default();
    insta::assert_debug_snapshot!(stats);
}
