//! Wave 26 snapshot tests for `bitnet-kernels` — CUDA attention masks,
//! dequantization configs, loss configs, stream management, profiling
//! structs, FFN configs, perf tracker, and GPU benchmark/spirv types.
//!
//! Pins Debug/Display output so unintentional changes are caught at review.

use std::time::Duration;

// =========================================================================
// Section 1 — CUDA attention mask configs
// =========================================================================

use bitnet_kernels::cuda::attention_mask::{
    AlibiConfig, AttentionMaskConfig, BlockSparseConfig, PrefixMaskConfig, SlidingWindowConfig,
};

#[test]
fn w26_attention_mask_config_debug() {
    let cfg = AttentionMaskConfig::new(512);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_attention_mask_config_small() {
    let cfg = AttentionMaskConfig::new(8);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_sliding_window_config_debug() {
    let cfg = SlidingWindowConfig::new(1024, 128).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_sliding_window_config_small() {
    let cfg = SlidingWindowConfig::new(32, 8).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_block_sparse_config_debug() {
    let cfg = BlockSparseConfig::new(2048, 64).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_alibi_config_debug() {
    let cfg = AlibiConfig::new(512, 8).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_alibi_config_many_heads() {
    let cfg = AlibiConfig::new(2048, 32).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_prefix_mask_config_debug() {
    let cfg = PrefixMaskConfig::new(256, 64).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 2 — CUDA dequantization configs
// =========================================================================

use bitnet_kernels::cuda::dequant::{DequantConfig, DequantPrecision, QuantBitWidth, ScaleMode};

#[test]
fn w26_dequant_config_default() {
    let cfg = DequantConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_dequant_precision_f32() {
    insta::assert_debug_snapshot!(DequantPrecision::F32);
}

#[test]
fn w26_dequant_precision_f16() {
    insta::assert_debug_snapshot!(DequantPrecision::F16);
}

#[test]
fn w26_quant_bit_width_all() {
    let widths = vec![QuantBitWidth::Int2, QuantBitWidth::Int4, QuantBitWidth::Int8];
    insta::assert_debug_snapshot!(widths);
}

#[test]
fn w26_scale_mode_all() {
    let modes = vec![ScaleMode::Uniform, ScaleMode::PerBlock, ScaleMode::PerChannel];
    insta::assert_debug_snapshot!(modes);
}

#[test]
fn w26_dequant_config_int4_f16_per_channel() {
    let cfg = DequantConfig {
        bit_width: QuantBitWidth::Int4,
        precision: DequantPrecision::F16,
        block_size: 128,
        scale_mode: ScaleMode::PerChannel,
    };
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 3 — CUDA loss configs
// =========================================================================

use bitnet_kernels::cuda::loss::{LossConfig, LossReduction};

#[test]
fn w26_loss_reduction_all() {
    let reductions = vec![LossReduction::Mean, LossReduction::Sum, LossReduction::None];
    insta::assert_debug_snapshot!(reductions);
}

#[test]
fn w26_loss_config_classification() {
    let cfg = LossConfig::new(64, 32000).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_loss_config_with_sum() {
    let mut cfg = LossConfig::new(128, 50257).unwrap();
    cfg.reduction = LossReduction::Sum;
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 4 — CUDA FFN configs
// =========================================================================

use bitnet_kernels::cuda::ffn::{FfnActivationType, FfnConfig, QuantBits, SparseFfnConfig};

#[test]
fn w26_ffn_activation_all() {
    let acts = vec![FfnActivationType::SiLU, FfnActivationType::GELU, FfnActivationType::ReLU];
    insta::assert_debug_snapshot!(acts);
}

#[test]
fn w26_quant_bits_all() {
    let bits = vec![QuantBits::Int2, QuantBits::Int4];
    insta::assert_debug_snapshot!(bits);
}

#[test]
fn w26_ffn_config_llama() {
    let cfg = FfnConfig::new(1, 4096, 11008, FfnActivationType::SiLU).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_ffn_config_gelu_small() {
    let cfg = FfnConfig::new(8, 768, 3072, FfnActivationType::GELU).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_sparse_ffn_config_debug() {
    let base = FfnConfig::new(1, 4096, 11008, FfnActivationType::SiLU).unwrap();
    let cfg = SparseFfnConfig::new(base, 2).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 5 — CUDA stream management
// =========================================================================

use bitnet_kernels::cuda::stream_mgmt::{
    DefaultStreamBehavior, PipelineStage, PipelineStageKind, ScheduleStrategy, StreamConfig,
    StreamPriority,
};

#[test]
fn w26_stream_priority_all() {
    let priorities = vec![StreamPriority::Low, StreamPriority::Normal, StreamPriority::High];
    insta::assert_debug_snapshot!(priorities);
}

#[test]
fn w26_stream_config_default() {
    let cfg = StreamConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_stream_config_high_perf() {
    let cfg = StreamConfig {
        num_streams: 16,
        priority: StreamPriority::High,
        default_stream_behavior: DefaultStreamBehavior::Legacy,
        enable_profiling: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_default_stream_behavior_all() {
    let behaviors = vec![DefaultStreamBehavior::Legacy, DefaultStreamBehavior::PerThread];
    insta::assert_debug_snapshot!(behaviors);
}

#[test]
fn w26_schedule_strategy_all() {
    let strategies = vec![
        ScheduleStrategy::RoundRobin,
        ScheduleStrategy::LeastLoaded,
        ScheduleStrategy::PriorityBased,
    ];
    insta::assert_debug_snapshot!(strategies);
}

#[test]
fn w26_pipeline_stage_kind_all() {
    let kinds = vec![
        PipelineStageKind::HostToDevice,
        PipelineStageKind::Compute,
        PipelineStageKind::DeviceToHost,
    ];
    insta::assert_debug_snapshot!(kinds);
}

#[test]
fn w26_pipeline_stage_compute() {
    let stage = PipelineStage::new(PipelineStageKind::Compute, "matmul_i2s", 5000);
    insta::assert_debug_snapshot!(stage);
}

#[test]
fn w26_pipeline_stage_transfer() {
    let stage = PipelineStage::new(PipelineStageKind::HostToDevice, "weights_upload", 1200);
    insta::assert_debug_snapshot!(stage);
}

// =========================================================================
// Section 6 — CUDA profiling structs
// =========================================================================

use bitnet_kernels::cuda::profiling::{
    AggregateStats, BandwidthMetrics, Bottleneck, ComputeMetrics, OccupancyLimiter, OccupancyResult,
};

#[test]
fn w26_bottleneck_all_display() {
    let bottlenecks = vec![
        Bottleneck::MemoryBound,
        Bottleneck::ComputeBound,
        Bottleneck::LatencyBound,
        Bottleneck::Unknown,
    ];
    let displays: Vec<String> = bottlenecks.iter().map(|b| b.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

#[test]
fn w26_occupancy_limiter_all_display() {
    let limiters = vec![
        OccupancyLimiter::Threads,
        OccupancyLimiter::Blocks,
        OccupancyLimiter::SharedMemory,
        OccupancyLimiter::Registers,
    ];
    let displays: Vec<String> = limiters.iter().map(|l| l.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

#[test]
fn w26_occupancy_result_high() {
    let result = OccupancyResult {
        theoretical: 0.95,
        active_warps: 48,
        max_warps: 64,
        limiter: OccupancyLimiter::Registers,
    };
    insta::assert_debug_snapshot!(result);
}

#[test]
fn w26_occupancy_result_low() {
    let result = OccupancyResult {
        theoretical: 0.25,
        active_warps: 8,
        max_warps: 64,
        limiter: OccupancyLimiter::SharedMemory,
    };
    insta::assert_debug_snapshot!(result);
}

#[test]
fn w26_bandwidth_metrics_debug() {
    let m = BandwidthMetrics::new(1_048_576, 524_288, Duration::from_micros(100), 900e9);
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w26_compute_metrics_debug() {
    let m = ComputeMetrics::new(2_000_000_000, Duration::from_millis(10), 15.0e12);
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w26_aggregate_stats_debug() {
    let stats = AggregateStats {
        count: 100,
        mean: Duration::from_micros(250),
        min: Duration::from_micros(180),
        max: Duration::from_micros(420),
        stddev_secs: 0.000045,
    };
    insta::assert_debug_snapshot!(stats);
}

// =========================================================================
// Section 7 — Perf tracker
// =========================================================================

use bitnet_kernels::perf_tracker::{KernelTiming, PerfTracker};

#[test]
fn w26_kernel_timing_basic() {
    let t = KernelTiming::new("gemm_i2s", Duration::from_micros(350), 1024 * 1024);
    insta::assert_debug_snapshot!(t);
}

#[test]
fn w26_kernel_timing_with_flops() {
    let t = KernelTiming::new("matmul_f32", Duration::from_millis(5), 4096 * 4096)
        .with_flops(137_438_953_472);
    insta::assert_debug_snapshot!(t);
}

#[test]
fn w26_perf_tracker_empty() {
    let tracker = PerfTracker::new();
    insta::assert_debug_snapshot!(tracker);
}

#[test]
fn w26_perf_tracker_with_records() {
    let mut tracker = PerfTracker::new();
    tracker.record(KernelTiming::new("softmax", Duration::from_micros(50), 32768));
    tracker.record(KernelTiming::new("rope", Duration::from_micros(30), 8192));
    tracker.record(
        KernelTiming::new("gemm_i2s", Duration::from_micros(800), 1048576)
            .with_flops(2_147_483_648),
    );
    insta::assert_debug_snapshot!(tracker);
}

// =========================================================================
// Section 8 — GPU benchmark / spirv / debug types (feature-gated)
// =========================================================================

#[cfg(any(feature = "gpu", feature = "cuda", feature = "oneapi"))]
mod gpu_snapshot_tests {
    use bitnet_kernels::gpu::benchmark::BenchmarkConfig;
    use bitnet_kernels::gpu::debug_layer::{Mismatch, Tolerance, ValidationReport};
    use bitnet_kernels::gpu::spirv_cache::{CacheStatsSnapshot, SpirvCacheConfig};
    use bitnet_kernels::gpu::validation::PerformanceResult;

    #[test]
    fn w26_benchmark_config_default() {
        let cfg = BenchmarkConfig::default();
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn w26_performance_result_debug() {
        let result = PerformanceResult {
            dimensions: (256, 256, 256),
            cpu_time_ms: 12.5,
            gpu_time_ms: 0.8,
            speedup: 15.625,
            gflops: 41.9,
        };
        insta::assert_debug_snapshot!(result);
    }

    #[test]
    fn w26_performance_result_small_matrix() {
        let result = PerformanceResult {
            dimensions: (64, 64, 64),
            cpu_time_ms: 0.3,
            gpu_time_ms: 0.5,
            speedup: 0.6,
            gflops: 1.05,
        };
        insta::assert_debug_snapshot!(result);
    }

    #[test]
    fn w26_tolerance_default() {
        let t = Tolerance::default();
        insta::assert_debug_snapshot!(t);
    }

    #[test]
    fn w26_mismatch_debug() {
        let m =
            Mismatch { index: 42, expected: 1.0, got: 1.0001, abs_diff: 0.0001, rel_diff: 0.0001 };
        insta::assert_debug_snapshot!(m);
    }

    #[test]
    fn w26_validation_report_pass() {
        let report = ValidationReport {
            operation: "layer_norm".into(),
            tolerance: Tolerance::default(),
            mismatches: vec![],
            total_elements: 1024,
        };
        insta::assert_debug_snapshot!(report);
    }

    #[test]
    fn w26_validation_report_with_mismatches() {
        let report = ValidationReport {
            operation: "matmul_i2s".into(),
            tolerance: Tolerance { abs_epsilon: 1e-3, rel_epsilon: 1e-2 },
            mismatches: vec![
                Mismatch { index: 7, expected: 0.5, got: 0.512, abs_diff: 0.012, rel_diff: 0.024 },
                Mismatch { index: 99, expected: -1.0, got: -0.98, abs_diff: 0.02, rel_diff: 0.02 },
            ],
            total_elements: 4096,
        };
        insta::assert_debug_snapshot!(report);
    }

    #[test]
    fn w26_spirv_cache_config_custom() {
        let cfg = SpirvCacheConfig {
            cache_dir: std::path::PathBuf::from("/tmp/spirv_cache"),
            max_cache_size_mb: 512,
            device_fingerprint: "0x10de:0x2684:535.129.03".into(),
        };
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn w26_cache_stats_snapshot_empty() {
        let snap = CacheStatsSnapshot { hits: 0, misses: 0, total_size_bytes: 0, entry_count: 0 };
        insta::assert_debug_snapshot!(snap);
    }

    #[test]
    fn w26_cache_stats_snapshot_active() {
        let snap = CacheStatsSnapshot {
            hits: 1500,
            misses: 200,
            total_size_bytes: 4_194_304,
            entry_count: 128,
        };
        insta::assert_debug_snapshot!(snap);
    }
}

// =========================================================================
// Section 9 — Additional CUDA memory pool and KV cache tests
// =========================================================================

use bitnet_kernels::cuda::memory_pool::{MemoryPoolConfig, MemoryStats};

#[test]
fn w26_cuda_memory_pool_config_custom() {
    let cfg = MemoryPoolConfig {
        initial_size: 2 * 1024 * 1024,
        max_size: 512 * 1024 * 1024,
        block_size: 512,
        alignment: 128,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w26_cuda_memory_stats_fragmented() {
    let stats = MemoryStats {
        total: 1_073_741_824,
        used: 536_870_912,
        free: 536_870_912,
        fragmentation: 0.42,
        num_allocations: 256,
        num_free_blocks: 64,
    };
    insta::assert_debug_snapshot!(stats);
}

#[test]
fn w26_cuda_memory_stats_full() {
    let stats = MemoryStats {
        total: 268_435_456,
        used: 268_435_456,
        free: 0,
        fragmentation: 0.0,
        num_allocations: 1,
        num_free_blocks: 0,
    };
    insta::assert_debug_snapshot!(stats);
}

use bitnet_kernels::cuda::kv_cache::CacheStats;

#[test]
fn w26_kv_cache_stats_debug() {
    let stats = CacheStats {
        entries_per_layer: vec![128, 128, 128, 128],
        memory_bytes: 16_777_216,
        hit_rate: 0.95,
        avg_access_time_us: 0.8,
    };
    insta::assert_debug_snapshot!(stats);
}
