//! Wave 20 snapshot tests — CUDA and CPU kernel configuration coverage.
//!
//! Pins Debug output of config/state structs across OpenCL profiling,
//! OpenCL pipeline, OpenCL/flash attention, CUDA conv1d, CPU tiling,
//! CPU RoPE, and additional CUDA config variants not covered in wave 19.

use std::collections::HashMap;

// =========================================================================
// Section 1 — OpenCL profiling structs
// =========================================================================

use bitnet_kernels::opencl_profiling::{
    KernelProfile, KernelStats, ProfilingReport, ProfilingSession, SessionSummary,
};

#[test]
fn profiling_session_empty() {
    let session = ProfilingSession::new();
    insta::assert_debug_snapshot!(session.summary());
}

#[test]
fn profiling_session_with_records() {
    let mut session = ProfilingSession::new();
    session.record(KernelProfile {
        kernel_name: "matmul_f32".into(),
        global_work_size: vec![1024, 1024],
        local_work_size: vec![16, 16],
        queued_ns: 1_000,
        submit_ns: 2_000,
        start_ns: 3_000,
        end_ns: 13_000,
    });
    session.record(KernelProfile {
        kernel_name: "softmax_f32".into(),
        global_work_size: vec![512],
        local_work_size: vec![64],
        queued_ns: 14_000,
        submit_ns: 15_000,
        start_ns: 16_000,
        end_ns: 21_000,
    });
    insta::assert_debug_snapshot!(session.summary());
}

#[test]
fn kernel_profile_debug() {
    let profile = KernelProfile {
        kernel_name: "rope_forward".into(),
        global_work_size: vec![256, 8],
        local_work_size: vec![32, 1],
        queued_ns: 100,
        submit_ns: 200,
        start_ns: 500,
        end_ns: 1_500,
    };
    insta::assert_debug_snapshot!(profile);
}

#[test]
fn profiling_report_empty() {
    let summary = SessionSummary {
        total_kernels: 0,
        total_gpu_time_ms: 0.0,
        avg_kernel_time_us: 0.0,
        kernel_breakdown: HashMap::new(),
    };
    let report = ProfilingReport::new(summary);
    insta::assert_snapshot!(report.to_table());
}

#[test]
fn profiling_report_with_data() {
    let mut breakdown = HashMap::new();
    breakdown.insert(
        "gemm_i2s".into(),
        KernelStats {
            count: 10,
            total_us: 500.0,
            min_us: 30.0,
            max_us: 80.0,
            avg_us: 50.0,
            std_dev_us: 12.5,
        },
    );
    let summary = SessionSummary {
        total_kernels: 10,
        total_gpu_time_ms: 0.5,
        avg_kernel_time_us: 50.0,
        kernel_breakdown: breakdown,
    };
    let report = ProfilingReport::new(summary);
    insta::assert_snapshot!(report.to_table());
}

// =========================================================================
// Section 2 — OpenCL pipeline configs
// =========================================================================

use bitnet_kernels::opencl_pipeline::{
    PipelineConfig, PipelineExecution, PipelineStage, StageResult,
};

#[test]
fn pipeline_config_small_model() {
    let cfg = PipelineConfig {
        num_layers: 12,
        hidden_dim: 768,
        num_heads: 12,
        head_dim: 64,
        intermediate_dim: 3072,
        vocab_size: 32000,
        max_seq_len: 2048,
        use_gpu: false,
        fallback_to_cpu: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn pipeline_config_large_model() {
    let cfg = PipelineConfig {
        num_layers: 32,
        hidden_dim: 4096,
        num_heads: 32,
        head_dim: 128,
        intermediate_dim: 11008,
        vocab_size: 128256,
        max_seq_len: 8192,
        use_gpu: true,
        fallback_to_cpu: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn pipeline_stage_all_variants() {
    let stages: Vec<PipelineStage> = vec![
        PipelineStage::Embedding,
        PipelineStage::RmsNorm,
        PipelineStage::Attention,
        PipelineStage::FeedForward,
        PipelineStage::FinalNorm,
        PipelineStage::LogitProjection,
    ];
    insta::assert_debug_snapshot!(stages);
}

#[test]
fn pipeline_stage_display() {
    let names: Vec<String> = vec![
        PipelineStage::Embedding,
        PipelineStage::RmsNorm,
        PipelineStage::Attention,
        PipelineStage::FeedForward,
        PipelineStage::FinalNorm,
        PipelineStage::LogitProjection,
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();
    insta::assert_debug_snapshot!(names);
}

#[test]
fn stage_result_cpu_debug() {
    let result = StageResult {
        stage: PipelineStage::Attention,
        output_shape: vec![1, 128, 768],
        execution_time_ns: 5_000_000,
        used_gpu: false,
        fallback_triggered: false,
    };
    insta::assert_debug_snapshot!(result);
}

#[test]
fn stage_result_gpu_fallback() {
    let result = StageResult {
        stage: PipelineStage::FeedForward,
        output_shape: vec![1, 128, 3072],
        execution_time_ns: 2_000_000,
        used_gpu: false,
        fallback_triggered: true,
    };
    insta::assert_debug_snapshot!(result);
}

#[test]
fn pipeline_execution_debug() {
    let exec = PipelineExecution {
        stages: vec![
            StageResult {
                stage: PipelineStage::Embedding,
                output_shape: vec![1, 128, 768],
                execution_time_ns: 1_000_000,
                used_gpu: false,
                fallback_triggered: false,
            },
            StageResult {
                stage: PipelineStage::Attention,
                output_shape: vec![1, 128, 768],
                execution_time_ns: 8_000_000,
                used_gpu: false,
                fallback_triggered: false,
            },
        ],
        total_time_ns: 9_000_000,
        tokens_generated: 1,
    };
    insta::assert_debug_snapshot!(exec);
}

// =========================================================================
// Section 3 — OpenCL attention score / mask structs
// =========================================================================

use bitnet_kernels::opencl_attention::AttentionScores;

#[test]
fn attention_scores_small() {
    let scores = AttentionScores {
        weights: vec![0.5, 0.3, 0.2, 0.1, 0.6, 0.3, 0.0, 0.0, 1.0],
        seq_len: 3,
        kv_len: 3,
    };
    insta::assert_debug_snapshot!(scores);
}

#[test]
fn attention_mask_causal_with_offset() {
    let mask = AttentionMask::causal(4, 6, 2);
    insta::assert_debug_snapshot!(mask);
}

#[test]
fn attention_mask_none_rectangular() {
    let mask = AttentionMask::none(2, 8);
    insta::assert_debug_snapshot!(mask);
}

// Additional CUDA softmax variant
use bitnet_kernels::cuda::softmax::SoftmaxConfig;

#[test]
fn softmax_config_with_temperature() {
    let cfg = SoftmaxConfig::for_shape(256, 16).unwrap().with_temperature(0.5).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 4 — OpenCL attention configs
// =========================================================================

use bitnet_kernels::opencl_attention::{
    AttentionConfig as OclAttentionConfig, AttentionMask,
    FlashAttentionConfig as OclTiledFlashConfig, KVCacheEntry,
};

#[test]
fn ocl_attention_config_mha() {
    let cfg = OclAttentionConfig::new(8, 64, 2048, true).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn ocl_attention_config_gqa() {
    let cfg = OclAttentionConfig::new_gqa(32, 8, 128, 4096, false).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn ocl_tiled_flash_config_default() {
    let cfg = OclTiledFlashConfig::default_opencl();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn ocl_tiled_flash_config_custom() {
    let cfg = OclTiledFlashConfig::new(64, 64, false).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn attention_mask_causal() {
    let mask = AttentionMask::causal(4, 4, 0);
    insta::assert_debug_snapshot!(mask);
}

#[test]
fn attention_mask_none() {
    let mask = AttentionMask::none(3, 5);
    insta::assert_debug_snapshot!(mask);
}

#[test]
fn kv_cache_entry_empty() {
    let entry = KVCacheEntry::new(4, 4);
    insta::assert_debug_snapshot!(entry);
}
// =========================================================================
// Section 5 — CUDA Conv1d configs
// =========================================================================

use bitnet_kernels::cuda::conv1d::{Conv1dConfig, PaddingMode};

#[test]
fn conv1d_config_basic() {
    let cfg = Conv1dConfig {
        in_channels: 256,
        out_channels: 512,
        kernel_size: 3,
        stride: 1,
        padding: PaddingMode::Zero(1),
        dilation: 1,
        groups: 1,
        bias: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn conv1d_config_depthwise_same_padding() {
    let cfg = Conv1dConfig {
        in_channels: 128,
        out_channels: 128,
        kernel_size: 5,
        stride: 2,
        padding: PaddingMode::Same,
        dilation: 1,
        groups: 128,
        bias: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn conv1d_config_dilated() {
    let cfg = Conv1dConfig {
        in_channels: 64,
        out_channels: 64,
        kernel_size: 3,
        stride: 1,
        padding: PaddingMode::Zero(2),
        dilation: 2,
        groups: 1,
        bias: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn padding_mode_variants() {
    let modes: Vec<PaddingMode> =
        vec![PaddingMode::Zero(0), PaddingMode::Zero(3), PaddingMode::Same];
    insta::assert_debug_snapshot!(modes);
}

// =========================================================================
// Section 6 — CUDA KV cache stats
// =========================================================================

use bitnet_kernels::cuda::kv_cache::CacheStats;

#[test]
fn cache_stats_initial() {
    let stats = CacheStats {
        entries_per_layer: vec![0; 12],
        memory_bytes: 0,
        hit_rate: 1.0,
        avg_access_time_us: 0.0,
    };
    insta::assert_debug_snapshot!(stats);
}

#[test]
fn cache_stats_populated() {
    let stats = CacheStats {
        entries_per_layer: vec![128, 128, 127, 128, 126, 128],
        memory_bytes: 25_165_824,
        hit_rate: 0.985,
        avg_access_time_us: 0.42,
    };
    insta::assert_debug_snapshot!(stats);
}

// =========================================================================
// Section 7 — CPU SIMD matmul / tiling configs
// =========================================================================

use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, TileConfig};

#[test]
fn tile_config_default() {
    let cfg = TileConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn tile_config_small() {
    let cfg = TileConfig::SMALL;
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn tile_config_large() {
    let cfg = TileConfig::LARGE;
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn tile_config_custom() {
    let cfg = TileConfig::new(48, 96, 32);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn simd_matmul_config_basic() {
    let cfg = SimdMatmulConfig::new(256, 512, 128);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn simd_matmul_config_transposed() {
    let cfg = SimdMatmulConfig {
        m: 64,
        n: 64,
        k: 64,
        alpha: 2.0,
        beta: 0.5,
        transpose_a: true,
        transpose_b: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 8 — CPU RoPE config
// =========================================================================

use bitnet_kernels::cpu::rope::RopeConfig as CpuRopeConfig;

#[test]
fn cpu_rope_config_default() {
    let cfg = CpuRopeConfig::new(64, 2048);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cpu_rope_config_custom_base() {
    let cfg = CpuRopeConfig::new(128, 8192).with_base(500_000.0);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn cpu_rope_config_with_scaling() {
    let cfg = CpuRopeConfig::new(64, 4096).with_base(10_000.0).with_scaling_factor(4.0);
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 9 — Additional CUDA elementwise op variants
// =========================================================================

use bitnet_kernels::cuda::elementwise::{ElementwiseConfig, ElementwiseOp};

#[test]
fn elementwise_op_all_variants() {
    let ops: Vec<ElementwiseOp> = vec![
        ElementwiseOp::Add,
        ElementwiseOp::Mul,
        ElementwiseOp::Sub,
        ElementwiseOp::Div,
        ElementwiseOp::FusedAddMul,
        ElementwiseOp::Relu,
        ElementwiseOp::Gelu,
        ElementwiseOp::Silu,
        ElementwiseOp::Sigmoid,
        ElementwiseOp::Tanh,
        ElementwiseOp::Clamp,
    ];
    insta::assert_debug_snapshot!(ops);
}

#[test]
fn elementwise_config_sub() {
    let cfg = ElementwiseConfig::new(2048, ElementwiseOp::Sub).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn elementwise_config_silu() {
    let cfg = ElementwiseConfig::new(4096, ElementwiseOp::Silu).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 10 — Additional CUDA quantize config variants
// =========================================================================

use bitnet_kernels::cuda::quantize::{QuantMethod, QuantizeConfig};

#[test]
fn quantize_config_minmax() {
    let cfg = QuantizeConfig { block_size: 64, method: QuantMethod::MinMax };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn quantize_config_percentile() {
    let cfg = QuantizeConfig { block_size: 128, method: QuantMethod::Percentile(99) };
    insta::assert_debug_snapshot!(cfg);
}
