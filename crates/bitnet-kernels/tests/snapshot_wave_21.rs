//! Wave 21 snapshot tests for CUDA and CPU kernel configurations in
//! `bitnet-kernels`.
//!
//! Pins the Debug representations of profiling, stream management,
//! graph execution, cooperative groups, warp primitives, scatter/gather,
//! batch normalization, loss functions, and tensor parallel configs.

use std::time::Duration;

// =========================================================================
// Section 1 — CUDA profiling configs
// =========================================================================

use bitnet_kernels::cuda::profiling::{
    BandwidthMetrics, Bottleneck, BottleneckAnalyzer, ComputeMetrics, KernelProfile,
    MemoryEventKind, OccupancyCalculator, OccupancyLimiter, ProfileAccumulator,
};

#[test]
fn occupancy_calculator_default_debug() {
    let calc = OccupancyCalculator::default();
    insta::assert_debug_snapshot!(calc);
}

#[test]
fn occupancy_result_thread_limited() {
    let calc = OccupancyCalculator::default();
    let result = calc.calculate(256, 0, 32);
    insta::assert_debug_snapshot!(result);
}

#[test]
fn occupancy_result_shared_mem_limited() {
    let calc = OccupancyCalculator::new(2048, 32, 49152, 65536);
    let result = calc.calculate(128, 32768, 16);
    insta::assert_debug_snapshot!(result);
}

#[test]
fn bandwidth_metrics_debug() {
    let bw = BandwidthMetrics::new(1024 * 1024, 512 * 1024, Duration::from_micros(100), 900e9);
    insta::assert_debug_snapshot!(bw);
}

#[test]
fn compute_metrics_debug() {
    let cm = ComputeMetrics::new(1_000_000_000, Duration::from_millis(1), 312e12);
    insta::assert_debug_snapshot!(cm);
}

#[test]
fn kernel_profile_minimal_debug() {
    let p = KernelProfile::new("matmul_f32", Duration::from_micros(42));
    insta::assert_debug_snapshot!(p);
}

#[test]
fn kernel_profile_with_grid_block_debug() {
    let p = KernelProfile::new("rmsnorm", Duration::from_micros(8))
        .with_grid_dim(128, 1, 1)
        .with_block_dim(256, 1, 1);
    insta::assert_debug_snapshot!(p);
}

#[test]
fn bottleneck_analyzer_default_debug() {
    let ba = BottleneckAnalyzer::default();
    insta::assert_debug_snapshot!(ba);
}

#[test]
fn bottleneck_all_variants_debug() {
    let variants = vec![
        Bottleneck::MemoryBound,
        Bottleneck::ComputeBound,
        Bottleneck::LatencyBound,
        Bottleneck::Unknown,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn occupancy_limiter_all_variants_debug() {
    let variants = vec![
        OccupancyLimiter::Threads,
        OccupancyLimiter::Blocks,
        OccupancyLimiter::SharedMemory,
        OccupancyLimiter::Registers,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn memory_event_kind_variants_debug() {
    let variants = vec![MemoryEventKind::Allocate, MemoryEventKind::Deallocate];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn profile_accumulator_empty_debug() {
    let acc = ProfileAccumulator::new();
    insta::assert_debug_snapshot!(acc);
}

// =========================================================================
// Section 2 — CUDA stream management configs
// =========================================================================

use bitnet_kernels::cuda::stream_mgmt::{
    DefaultStreamBehavior, PipelineStage, PipelineStageKind, StreamConfig, StreamPriority,
};

#[test]
fn stream_config_default_debug() {
    let cfg = StreamConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn stream_priority_all_variants_debug() {
    let variants = vec![StreamPriority::Low, StreamPriority::Normal, StreamPriority::High];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn default_stream_behavior_variants_debug() {
    let variants = vec![DefaultStreamBehavior::Legacy, DefaultStreamBehavior::PerThread];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn pipeline_stage_h2d_debug() {
    let stage = PipelineStage::new(PipelineStageKind::HostToDevice, "weights_upload", 4096);
    insta::assert_debug_snapshot!(stage);
}

#[test]
fn pipeline_stage_compute_debug() {
    let stage = PipelineStage::new(PipelineStageKind::Compute, "matmul_kernel", 65536);
    insta::assert_debug_snapshot!(stage);
}

#[test]
fn pipeline_stage_kind_all_variants_debug() {
    let variants = vec![
        PipelineStageKind::HostToDevice,
        PipelineStageKind::Compute,
        PipelineStageKind::DeviceToHost,
    ];
    insta::assert_debug_snapshot!(variants);
}

// =========================================================================
// Section 3 — CUDA graph execution configs
// =========================================================================

use bitnet_kernels::cuda::graph_exec::{LayerGraphConfig, MultiStreamConfig, OptimizeStats};

#[test]
fn layer_graph_config_default_debug() {
    let cfg = LayerGraphConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn multi_stream_config_default_debug() {
    let cfg = MultiStreamConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn optimize_stats_default_debug() {
    let stats = OptimizeStats::default();
    insta::assert_debug_snapshot!(stats);
}

// =========================================================================
// Section 4 — CUDA cooperative groups configs
// =========================================================================

use bitnet_kernels::cuda::cooperative_groups::{
    CoalescedGroup, CooperativeGroupConfig, CooperativeReduceOp, GridGroup, ThreadBlockGroup,
};

#[test]
fn cooperative_group_config_default_debug() {
    let cfg = CooperativeGroupConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cooperative_group_config_with_grid_sync_debug() {
    let cfg = CooperativeGroupConfig::new(512).unwrap().with_grid_sync();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cooperative_group_config_with_cluster_debug() {
    let cfg = CooperativeGroupConfig::new(256).unwrap().with_cluster(4).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cooperative_reduce_op_all_variants_debug() {
    let variants = vec![
        CooperativeReduceOp::Sum,
        CooperativeReduceOp::Max,
        CooperativeReduceOp::Min,
        CooperativeReduceOp::Product,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn thread_block_group_debug() {
    let tbg = ThreadBlockGroup::new(256, 0).unwrap();
    insta::assert_debug_snapshot!(tbg);
}

#[test]
fn grid_group_debug() {
    let gg = GridGroup::new(256, 128).unwrap();
    insta::assert_debug_snapshot!(gg);
}

#[test]
fn coalesced_group_full_mask_debug() {
    let cg = CoalescedGroup::new(0xFFFF_FFFF).unwrap();
    insta::assert_debug_snapshot!(cg);
}

#[test]
fn coalesced_group_partial_mask_debug() {
    let cg = CoalescedGroup::new(0x0000_00FF).unwrap();
    insta::assert_debug_snapshot!(cg);
}

// =========================================================================
// Section 5 — CUDA warp operations configs
// =========================================================================

use bitnet_kernels::cuda::warp_ops::WarpConfig;

#[test]
fn warp_config_default_debug() {
    let cfg = WarpConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 6 — CPU scatter/gather configs
// =========================================================================

use bitnet_kernels::cpu::scatter_gather::{ScatterGatherConfig, ScatterReduce};

#[test]
fn scatter_gather_config_default_debug() {
    let cfg = ScatterGatherConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn scatter_gather_config_with_add_debug() {
    let cfg = ScatterGatherConfig::with_reduce(ScatterReduce::Add);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn scatter_reduce_all_variants_debug() {
    let variants = vec![
        ScatterReduce::Assign,
        ScatterReduce::Add,
        ScatterReduce::Max,
        ScatterReduce::Min,
        ScatterReduce::Mul,
    ];
    insta::assert_debug_snapshot!(variants);
}

// =========================================================================
// Section 7 — CPU batch norm configs
// =========================================================================

use bitnet_kernels::cpu::batch_norm::BatchNormConfig;
use bitnet_kernels::cpu::batch_normalization::{SimdBatchNormConfig, SimdBatchNormState};

#[test]
fn batch_norm_config_default_debug() {
    let cfg = BatchNormConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn batch_norm_config_training_debug() {
    let cfg = BatchNormConfig { num_features: 256, eps: 1e-5, momentum: 0.1, training: true };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn simd_batch_norm_config_default_debug() {
    let cfg = SimdBatchNormConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn simd_batch_norm_state_initial_debug() {
    let state = SimdBatchNormState::new(4);
    insta::assert_debug_snapshot!(state);
}

// =========================================================================
// Section 8 — CPU loss function types
// =========================================================================

use bitnet_kernels::cpu::loss::LossReduction;

#[test]
fn loss_reduction_all_variants_debug() {
    let variants = vec![LossReduction::None, LossReduction::Mean, LossReduction::Sum];
    insta::assert_debug_snapshot!(variants);
}

// =========================================================================
// Section 9 — CPU tensor parallel configs
// =========================================================================

use bitnet_kernels::cpu::tensor_parallel::{
    CommBackend, ShardingStrategy, TensorParallelConfig, TensorParallelError,
    TensorParallelMetrics, TensorShard,
};

#[test]
fn tensor_parallel_config_2rank_debug() {
    let cfg = TensorParallelConfig {
        num_ranks: 2,
        rank_id: 0,
        comm_backend: CommBackend::InProcess,
        overlap_compute_comm: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn tensor_parallel_config_4rank_overlap_debug() {
    let cfg = TensorParallelConfig {
        num_ranks: 4,
        rank_id: 1,
        comm_backend: CommBackend::SharedMemory,
        overlap_compute_comm: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn comm_backend_all_variants_debug() {
    let variants = vec![CommBackend::InProcess, CommBackend::SharedMemory];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn sharding_strategy_column_debug() {
    let s = ShardingStrategy::ColumnParallel;
    insta::assert_debug_snapshot!(s);
}

#[test]
fn sharding_strategy_row_debug() {
    let s = ShardingStrategy::RowParallel;
    insta::assert_debug_snapshot!(s);
}

#[test]
fn sharding_strategy_custom_debug() {
    let s = ShardingStrategy::Custom { splits: vec![128, 256, 128] };
    insta::assert_debug_snapshot!(s);
}

#[test]
fn tensor_shard_debug() {
    let shard =
        TensorShard { data: vec![1.0, 2.0, 3.0, 4.0], rank_id: 0, shard_index: 0, total_shards: 2 };
    insta::assert_debug_snapshot!(shard);
}

#[test]
fn tensor_parallel_metrics_default_debug() {
    let m = TensorParallelMetrics::default();
    insta::assert_debug_snapshot!(m);
}

#[test]
fn tensor_parallel_error_uneven_debug() {
    let e = TensorParallelError::UnevenSharding { tensor_len: 100, num_shards: 3 };
    insta::assert_debug_snapshot!(e);
}

#[test]
fn tensor_parallel_error_oob_debug() {
    let e = TensorParallelError::ShardIndexOutOfBounds { index: 5, total: 4 };
    insta::assert_debug_snapshot!(e);
}
