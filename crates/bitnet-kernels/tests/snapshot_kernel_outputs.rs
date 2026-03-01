//! Wave-4 snapshot tests for kernel *outputs* (numerical results).
//!
//! Unlike the API-surface snapshot tests in `snapshot_tests.rs` and
//! `snapshot_wave5.rs`, these tests pin the actual computed values
//! produced by the CPU kernels so that accidental numerical regressions
//! are caught at review time.

use bitnet_kernels::cpu::{conv1d, embedding, quantized_matmul, rope, softmax};
use bitnet_kernels::reduction::{self, ReductionOp};

// ── helpers ────────────────────────────────────────────────────────

/// Format an f32 slice to 6 decimal places for stable snapshots.
fn fmt_f32(v: &[f32]) -> String {
    let items: Vec<String> =
        v.iter().map(|x| format!("{x:.6}")).collect();
    format!("[{}]", items.join(", "))
}

// ── softmax ────────────────────────────────────────────────────────

#[test]
fn softmax_uniform_input() {
    let input = vec![1.0, 1.0, 1.0, 1.0];
    let out = softmax::softmax(&input, 1.0).unwrap();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn softmax_ascending_input() {
    let input = vec![0.0, 1.0, 2.0, 3.0];
    let out = softmax::softmax(&input, 1.0).unwrap();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn softmax_with_temperature() {
    let input = vec![0.0, 1.0, 2.0, 3.0];
    let out = softmax::softmax(&input, 0.5).unwrap();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn softmax_batch_two_rows() {
    // Two rows of length 3
    let input = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
    let out = softmax::softmax_batch(&input, 3, 1.0).unwrap();
    insta::assert_snapshot!(fmt_f32(&out));
}

// ── conv1d ─────────────────────────────────────────────────────────

#[test]
fn conv1d_simple_no_padding() {
    let config = conv1d::Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 3,
        stride: 1,
        padding: conv1d::PaddingMode::Zero(0),
        dilation: 1,
        groups: 1,
        bias: false,
    };
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // [1, 5]
    let weight = vec![1.0, 0.0, -1.0]; // [1, 1, 3]
    let out = conv1d::conv1d_forward(&input, &weight, None, &config)
        .unwrap();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn conv1d_with_bias_and_padding() {
    let config = conv1d::Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 3,
        stride: 1,
        padding: conv1d::PaddingMode::Zero(1),
        dilation: 1,
        groups: 1,
        bias: true,
    };
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![0.5, 0.5, 0.5];
    let bias = vec![0.1];
    let out = conv1d::conv1d_forward(
        &input,
        &weight,
        Some(&bias),
        &config,
    )
    .unwrap();
    insta::assert_snapshot!(fmt_f32(&out));
}

// ── embedding lookup ───────────────────────────────────────────────

#[test]
fn embedding_lookup_basic() {
    // vocab_size=4, embedding_dim=3
    #[rustfmt::skip]
    let table: Vec<f32> = vec![
        0.1, 0.2, 0.3,   // token 0
        0.4, 0.5, 0.6,   // token 1
        0.7, 0.8, 0.9,   // token 2
        1.0, 1.1, 1.2,   // token 3
    ];
    let indices = vec![2, 0, 3];
    let out = embedding::embedding_lookup(&table, &indices, 3).unwrap();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn embedding_accumulate_weighted() {
    #[rustfmt::skip]
    let table: Vec<f32> = vec![
        1.0, 0.0,   // token 0
        0.0, 1.0,   // token 1
        0.5, 0.5,   // token 2
    ];
    let indices = vec![0, 1, 2];
    let weights = vec![0.5, 0.3, 0.2];
    let out = embedding::embedding_accumulate(
        &table, &indices, &weights, 2,
    )
    .unwrap();
    insta::assert_snapshot!(fmt_f32(&out));
}

// ── RoPE rotation ──────────────────────────────────────────────────

#[test]
fn rope_frequencies_position_zero() {
    let config = rope::RopeConfig::new(4, 2);
    let freqs = rope::compute_frequencies(&config);
    // Position 0: cos(0)=1, sin(0)=0 for all dim pairs
    insta::assert_snapshot!(fmt_f32(&freqs[..4]));
}

#[test]
fn rope_apply_single_position() {
    let config = rope::RopeConfig::new(4, 4);
    let freqs = rope::compute_frequencies(&config);
    let mut data = vec![1.0, 0.0, 1.0, 0.0];
    rope::apply_rope(&mut data, 1, 4, &freqs);
    insta::assert_snapshot!(fmt_f32(&data));
}

#[test]
fn rope_round_trip_position_zero() {
    // At position 0 angle=0 so rotation is identity.
    let config = rope::RopeConfig::new(4, 2);
    let freqs = rope::compute_frequencies(&config);
    let mut data = vec![3.0, 4.0, 5.0, 6.0];
    rope::apply_rope(&mut data, 0, 4, &freqs);
    insta::assert_snapshot!(fmt_f32(&data));
}

// ── quantized matmul ───────────────────────────────────────────────

#[test]
fn i2s_matmul_identity_weights() {
    // 1×4 activations, 4×2 weights (all +1), block_size=4
    let activations = vec![1.0, 2.0, 3.0, 4.0];
    let weights_packed = vec![
        quantized_matmul::pack_i2s([1, 1, 1, 1]), // col 0
        quantized_matmul::pack_i2s([1, 1, 1, 1]), // col 1
    ];
    let scales = vec![1.0, 1.0]; // one scale per col × 1 block
    let mut out = vec![0.0f32; 2];
    quantized_matmul::i2s_matmul_f32(
        &activations,
        &weights_packed,
        &scales,
        &mut out,
        1, 2, 4, 4,
    )
    .unwrap();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn i2s_matmul_mixed_ternary() {
    // 1×4 activations, 4×1 weight col [+1, -1, 0, +1], scale=2.0
    let activations = vec![1.0, 2.0, 3.0, 4.0];
    let weights_packed =
        vec![quantized_matmul::pack_i2s([1, -1, 0, 1])];
    let scales = vec![2.0];
    let mut out = vec![0.0f32; 1];
    quantized_matmul::i2s_matmul_f32(
        &activations,
        &weights_packed,
        &scales,
        &mut out,
        1, 1, 4, 4,
    )
    .unwrap();
    // (1*1 + 2*(-1) + 3*0 + 4*1) * 2.0 = (1-2+0+4)*2 = 6.0
    insta::assert_snapshot!(fmt_f32(&out));
}

// ── reduction ──────────────────────────────────────────────────────

#[test]
fn reduction_row_wise_sum_and_max() {
    let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
    let sums =
        reduction::reduce_rows_f32(&matrix, 2, 3, ReductionOp::Sum)
            .unwrap();
    let maxes =
        reduction::reduce_rows_f32(&matrix, 2, 3, ReductionOp::Max)
            .unwrap();
    insta::assert_snapshot!(format!(
        "sums={} maxes={}",
        fmt_f32(&sums),
        fmt_f32(&maxes),
    ));
}
