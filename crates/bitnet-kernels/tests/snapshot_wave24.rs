//! Wave 24 snapshot tests — CPU kernel configs, pipeline-parallel, and
//! computed kernel outputs at fixed inputs.
//!
//! Covers: PipelineSchedule, PipelineStage (cpu::pipeline_parallel),
//! PipelineConfig, Conv2DParams, softmax/matmul/rope computed outputs,
//! i2s_matmul outputs, reduction kernel outputs, embedding outputs,
//! activation function outputs, and shaped reduction configs.

use bitnet_kernels::cpu::pipeline_parallel::{PipelineConfig, PipelineSchedule, PipelineStage};

// =========================================================================
// Section 1 — CPU pipeline-parallel config snapshots (9 tests)
// =========================================================================

#[test]
fn w24_pipeline_schedule_all_variants() {
    let variants: Vec<PipelineSchedule> = vec![
        PipelineSchedule::Sequential,
        PipelineSchedule::GPipe,
        PipelineSchedule::PipeDream,
        PipelineSchedule::Interleaved,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn w24_pipeline_schedule_display() {
    let displays: Vec<String> = vec![
        PipelineSchedule::Sequential,
        PipelineSchedule::GPipe,
        PipelineSchedule::PipeDream,
        PipelineSchedule::Interleaved,
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();
    insta::assert_debug_snapshot!(displays);
}

#[test]
fn w24_pipeline_stage_simple() {
    let stage = PipelineStage::new(0, 6);
    insta::assert_debug_snapshot!(stage);
}

#[test]
fn w24_pipeline_stage_with_affinity() {
    let stage = PipelineStage::new(6, 12).with_affinity(3);
    insta::assert_debug_snapshot!(stage);
}

#[test]
fn w24_pipeline_stage_num_layers() {
    let stage = PipelineStage::new(4, 16);
    insta::assert_snapshot!(format!("num_layers={}", stage.num_layers()));
}

#[test]
fn w24_pipeline_config_two_stage_gpipe() {
    let cfg = PipelineConfig::new(
        vec![PipelineStage::new(0, 16), PipelineStage::new(16, 32)],
        4,
        PipelineSchedule::GPipe,
    );
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_pipeline_config_four_stage_pipedream() {
    let cfg = PipelineConfig::new(
        vec![
            PipelineStage::new(0, 8).with_affinity(0),
            PipelineStage::new(8, 16).with_affinity(1),
            PipelineStage::new(16, 24).with_affinity(2),
            PipelineStage::new(24, 32).with_affinity(3),
        ],
        2,
        PipelineSchedule::PipeDream,
    );
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_pipeline_config_num_stages() {
    let cfg = PipelineConfig::new(
        vec![PipelineStage::new(0, 12), PipelineStage::new(12, 24), PipelineStage::new(24, 32)],
        8,
        PipelineSchedule::Interleaved,
    );
    insta::assert_snapshot!(format!("num_stages={}", cfg.num_stages()));
}

#[test]
fn w24_pipeline_config_validate_ok() {
    let cfg = PipelineConfig::new(vec![PipelineStage::new(0, 16)], 1, PipelineSchedule::Sequential);
    assert!(cfg.validate().is_ok());
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 2 — Convolution params (2 tests)
// =========================================================================

use bitnet_kernels::convolution::Conv2DParams;

#[test]
fn w24_conv2d_params_default_3x3() {
    let params = Conv2DParams { stride: (1, 1), padding: (1, 1), dilation: (1, 1) };
    insta::assert_debug_snapshot!(params);
}

#[test]
fn w24_conv2d_params_dilated_5x5() {
    let params = Conv2DParams { stride: (2, 2), padding: (2, 2), dilation: (2, 2) };
    insta::assert_debug_snapshot!(params);
}

// =========================================================================
// Section 3 — Batched softmax computed output snapshots (2 tests)
// =========================================================================

use bitnet_kernels::cpu::batched_softmax;

#[test]
fn w24_batched_softmax_single_row() {
    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let output = batched_softmax(&input, 1, 4).unwrap();
    insta::assert_debug_snapshot!(output);
}

#[test]
fn w24_batched_softmax_two_rows() {
    let input = vec![1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0];
    let output = batched_softmax(&input, 2, 3).unwrap();
    insta::assert_debug_snapshot!(output);
}

// =========================================================================
// Section 3b — Conv2dConfig and BatchNormConfig (3 tests)
// =========================================================================

use bitnet_kernels::cpu::batch_norm::BatchNormConfig;
use bitnet_kernels::cpu::conv2d::Conv2dConfig;

#[test]
fn w24_conv2d_config_3x3() {
    let cfg = Conv2dConfig {
        in_channels: 64,
        out_channels: 128,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        padding_h: 1,
        padding_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 1,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_batch_norm_config_default() {
    let cfg = BatchNormConfig::new(256);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_batch_norm_config_custom() {
    let cfg = BatchNormConfig { num_features: 512, eps: 1e-3, momentum: 0.05, training: true };
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 4 — RoPE frequency table and config snapshots (3 tests)
// =========================================================================

use bitnet_kernels::cpu::rope::{RopeConfig, compute_frequencies};

#[test]
fn w24_rope_config_default() {
    let cfg = RopeConfig::new(8, 64);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_rope_config_custom_base() {
    let cfg = RopeConfig::new(16, 128).with_base(500_000.0);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_rope_frequencies_dim8() {
    let cfg = RopeConfig::new(8, 64);
    let freqs = compute_frequencies(&cfg);
    let rounded: Vec<f32> = freqs.iter().map(|v| (v * 1e6).round() / 1e6).collect();
    insta::assert_debug_snapshot!(rounded);
}

// =========================================================================
// Section 5 — i2s_matmul computed output (3 tests)
// =========================================================================

use bitnet_kernels::cpu::quantized_matmul::{i2s_matmul_f32, pack_i2s};

#[test]
fn w24_pack_i2s_ternary_values() {
    let packed: Vec<u8> = vec![
        pack_i2s([1, 0, -1, 1]),
        pack_i2s([-1, 0, 1, -1]),
        pack_i2s([0, 0, 1, -1]),
        pack_i2s([1, 0, -1, 0]),
    ];
    insta::assert_debug_snapshot!(packed);
}

#[test]
fn w24_i2s_matmul_identity_4x4() {
    // 4×4 identity in ternary with block_size=4
    let weights: Vec<i8> = vec![
        1, 0, 0, 0, // row 0
        0, 1, 0, 0, // row 1
        0, 0, 1, 0, // row 2
        0, 0, 0, 1, // row 3
    ];
    let packed: Vec<u8> = weights.chunks(4).map(|c| pack_i2s([c[0], c[1], c[2], c[3]])).collect();
    let activations = vec![3.0_f32, 7.0, 2.0, 5.0];
    let scales = vec![1.0_f32; 4]; // one scale per block
    let mut out = vec![0.0_f32; 4];
    i2s_matmul_f32(&activations, &packed, &scales, &mut out, 1, 4, 4, 4).unwrap();
    insta::assert_debug_snapshot!(out);
}

#[test]
fn w24_i2s_matmul_mixed_4x4() {
    let weights: Vec<i8> = vec![
        1, -1, 0, 1, // row 0
        0, 1, -1, 0, // row 1
        -1, 0, 1, 1, // row 2
        1, 1, -1, -1, // row 3
    ];
    let packed: Vec<u8> = weights.chunks(4).map(|c| pack_i2s([c[0], c[1], c[2], c[3]])).collect();
    let activations = vec![1.0_f32, 2.0, 3.0, 4.0];
    let scales = vec![0.5_f32; 4];
    let mut out = vec![0.0_f32; 4];
    i2s_matmul_f32(&activations, &packed, &scales, &mut out, 1, 4, 4, 4).unwrap();
    insta::assert_debug_snapshot!(out);
}

// =========================================================================
// Section 6 — Reduction kernel outputs (4 tests)
// =========================================================================

use bitnet_kernels::cpu::reduction::{ReductionAxis, ReductionKernel};

#[test]
fn w24_reduction_axis_variants() {
    let axes: Vec<ReductionAxis> = vec![ReductionAxis::Row, ReductionAxis::Column];
    insta::assert_debug_snapshot!(axes);
}

#[test]
fn w24_reduction_sum_flat() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = ReductionKernel::sum(&data);
    insta::assert_debug_snapshot!(result);
}

#[test]
fn w24_reduction_sum_axis_row() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = ReductionKernel::sum_axis(&data, 2, 3, ReductionAxis::Row);
    insta::assert_debug_snapshot!(result);
}

#[test]
fn w24_reduction_max_flat() {
    let data = vec![1.0_f32, 5.0, 3.0, 4.0, 2.0, 6.0];
    let result = ReductionKernel::max(&data);
    insta::assert_debug_snapshot!(result);
}

// =========================================================================
// Section 7 — Activation function outputs (2 tests)
// =========================================================================

use bitnet_kernels::cpu::activations::{gelu_vec, silu_vec};

#[test]
fn w24_gelu_known_inputs() {
    let input = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let output = gelu_vec(&input);
    insta::assert_debug_snapshot!(output);
}

#[test]
fn w24_silu_known_inputs() {
    let input = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let output = silu_vec(&input);
    insta::assert_debug_snapshot!(output);
}

// =========================================================================
// Section 8 — Embedding lookup outputs (2 tests)
// =========================================================================

use bitnet_kernels::cpu::embedding::{embedding_lookup_batched, positional_encoding};

#[test]
fn w24_embedding_lookup_batched_output() {
    let table = vec![
        0.1_f32, 0.2, 0.3, // token 0
        0.4, 0.5, 0.6, // token 1
        0.7, 0.8, 0.9, // token 2
    ];
    let indices: &[&[u32]] = &[&[2, 0, 1]];
    let result = embedding_lookup_batched(&table, indices, 3, 3);
    insta::assert_debug_snapshot!(result);
}

#[test]
fn w24_positional_encoding_4x4() {
    let pe = positional_encoding(4, 4, 10000.0);
    let rounded: Vec<f32> = pe.iter().map(|v| (v * 10000.0).round() / 10000.0).collect();
    insta::assert_debug_snapshot!(rounded);
}

// =========================================================================
// Section 9 — Shaped reduction config (3 tests)
// =========================================================================

use bitnet_kernels::reduction::ReductionOp;
use bitnet_kernels::shaped_reduction::ShapedReductionConfig;

#[test]
fn w24_shaped_reduction_config_global_sum() {
    let cfg = ShapedReductionConfig::global(ReductionOp::Sum);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_shaped_reduction_config_axis0_mean() {
    let cfg = ShapedReductionConfig::new(ReductionOp::Mean, Some(0), false);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_shaped_reduction_config_axis1_keepdim() {
    let cfg = ShapedReductionConfig::new(ReductionOp::Max, Some(1), true);
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 10 — Pipeline validation errors (2 tests)
// =========================================================================

#[test]
fn w24_pipeline_config_validate_empty_stages() {
    let cfg = PipelineConfig::new(vec![], 1, PipelineSchedule::GPipe);
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn w24_pipeline_config_validate_zero_micro_batch() {
    let cfg = PipelineConfig::new(vec![PipelineStage::new(0, 8)], 0, PipelineSchedule::GPipe);
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}
