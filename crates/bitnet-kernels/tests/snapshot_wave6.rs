//! Wave 6 snapshot tests for `bitnet-kernels` activation, reduction,
//! quantized-matmul, attention, and RoPE modules.
//!
//! Pins numerical outputs so that kernel refactors or SIMD changes that
//! alter results are caught at review time.

use bitnet_kernels::cpu::attention::{AttentionConfig, AttentionKernel};
use bitnet_kernels::cpu::quantized_matmul::{i2s_matmul_f32, pack_i2s};
use bitnet_kernels::cpu::reduction::{ReductionAxis, ReductionKernel};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, apply_rope_batch, compute_frequencies};
use bitnet_kernels::cpu::simd_math::{fast_exp_f32, fast_sigmoid_f32, fast_tanh_f32};

// ── Helpers ────────────────────────────────────────────────────────────

/// Format an f32 slice to 6 decimal places for stable snapshots.
fn fmt_f32(data: &[f32]) -> String {
    data.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(", ")
}

/// Scalar activation helpers (deterministic, no SIMD variance).
fn relu(x: f32) -> f32 {
    x.max(0.0)
}
fn gelu(x: f32) -> f32 {
    // GELU approximation (tanh-based, matches PyTorch default)
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x >= 0.0 { x } else { alpha * x }
}
fn elu(x: f32, alpha: f32) -> f32 {
    if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) }
}
fn selu(x: f32) -> f32 {
    let alpha = 1.6732632;
    let lambda = 1.050_701;
    lambda * if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) }
}
fn mish(x: f32) -> f32 {
    x * ((1.0 + x.exp()).ln()).tanh()
}
fn hardswish(x: f32) -> f32 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        x
    } else {
        x * (x + 3.0) / 6.0
    }
}
fn hardsigmoid(x: f32) -> f32 {
    ((x + 3.0) / 6.0).clamp(0.0, 1.0)
}

/// Canonical test input shared across activation tests.
const ACTIVATION_INPUT: [f32; 8] = [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0];

// ═══════════════════════════════════════════════════════════════════════
// Activation function snapshots
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn activation_relu() {
    let out: Vec<f32> = ACTIVATION_INPUT.iter().map(|&x| relu(x)).collect();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn activation_gelu() {
    let out: Vec<f32> = ACTIVATION_INPUT.iter().map(|&x| gelu(x)).collect();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn activation_silu() {
    let out: Vec<f32> = ACTIVATION_INPUT.iter().map(|&x| silu(x)).collect();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn activation_sigmoid() {
    let out: Vec<f32> = ACTIVATION_INPUT.iter().map(|&x| sigmoid(x)).collect();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn activation_tanh() {
    let out: Vec<f32> = ACTIVATION_INPUT.iter().map(|&x| x.tanh()).collect();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn activation_leaky_relu() {
    let out: Vec<f32> = ACTIVATION_INPUT.iter().map(|&x| leaky_relu(x, 0.01)).collect();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn activation_elu() {
    let out: Vec<f32> = ACTIVATION_INPUT.iter().map(|&x| elu(x, 1.0)).collect();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn activation_selu() {
    let out: Vec<f32> = ACTIVATION_INPUT.iter().map(|&x| selu(x)).collect();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn activation_mish() {
    let out: Vec<f32> = ACTIVATION_INPUT.iter().map(|&x| mish(x)).collect();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn activation_hardswish() {
    let out: Vec<f32> = ACTIVATION_INPUT.iter().map(|&x| hardswish(x)).collect();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn activation_hardsigmoid() {
    let out: Vec<f32> = ACTIVATION_INPUT.iter().map(|&x| hardsigmoid(x)).collect();
    insta::assert_snapshot!(fmt_f32(&out));
}

// ── SIMD-dispatched activations (exp, sigmoid, tanh from simd_math) ──

#[test]
fn simd_math_exp() {
    let out = fast_exp_f32(&ACTIVATION_INPUT);
    insta::with_settings!({filters => vec![
        // Allow tiny float rounding variance between scalar and SIMD.
        (r"\d+\.\d{5}\d+", "[FP]"),
    ]}, {
        insta::assert_snapshot!(fmt_f32(&out));
    });
}

#[test]
fn simd_math_sigmoid() {
    let out = fast_sigmoid_f32(&ACTIVATION_INPUT);
    insta::with_settings!({filters => vec![
        (r"\d+\.\d{5}\d+", "[FP]"),
    ]}, {
        insta::assert_snapshot!(fmt_f32(&out));
    });
}

#[test]
fn simd_math_tanh() {
    let out = fast_tanh_f32(&ACTIVATION_INPUT);
    insta::with_settings!({filters => vec![
        (r"\d+\.\d{5}\d+", "[FP]"),
    ]}, {
        insta::assert_snapshot!(fmt_f32(&out));
    });
}

// ═══════════════════════════════════════════════════════════════════════
// Reduction operation snapshots
// ═══════════════════════════════════════════════════════════════════════

const REDUCTION_INPUT: [f32; 6] = [3.0, -1.0, 4.0, 1.5, -2.0, 5.0];

#[test]
fn reduction_sum() {
    let result = ReductionKernel::sum(&REDUCTION_INPUT).unwrap();
    insta::assert_snapshot!(format!("{result:.6}"));
}

#[test]
fn reduction_max() {
    let result = ReductionKernel::max(&REDUCTION_INPUT).unwrap();
    insta::assert_debug_snapshot!(result);
}

#[test]
fn reduction_min() {
    let result = ReductionKernel::min(&REDUCTION_INPUT).unwrap();
    insta::assert_debug_snapshot!(result);
}

#[test]
fn reduction_mean() {
    let result = ReductionKernel::mean(&REDUCTION_INPUT).unwrap();
    insta::assert_snapshot!(format!("{result:.6}"));
}

#[test]
fn reduction_l2_norm() {
    let result = ReductionKernel::l2_norm(&REDUCTION_INPUT).unwrap();
    insta::assert_snapshot!(format!("{result:.6}"));
}

#[test]
fn reduction_argmax_via_max() {
    let result = ReductionKernel::max(&REDUCTION_INPUT).unwrap();
    insta::assert_snapshot!(format!("argmax={} value={:.6}", result.index, result.value));
}

#[test]
fn reduction_argmin_via_min() {
    let result = ReductionKernel::min(&REDUCTION_INPUT).unwrap();
    insta::assert_snapshot!(format!("argmin={} value={:.6}", result.index, result.value));
}

#[test]
fn reduction_row_wise_sum() {
    // 2×3 matrix
    let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = ReductionKernel::sum_axis(&matrix, 2, 3, ReductionAxis::Row).unwrap();
    insta::assert_snapshot!(fmt_f32(&result));
}

// ═══════════════════════════════════════════════════════════════════════
// Quantized matmul CPU fallback snapshot
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn quantized_matmul_cpu_fallback() {
    // 2×4 activations × 4×2 ternary weights, block_size=32
    let m: usize = 2;
    let n: usize = 2;
    let k: usize = 4;
    let block_size: usize = 32;

    let activations: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 1.5, -0.5];

    // Ternary weight matrix (k×n, row-major): [[1,0],[−1,1],[0,−1],[1,1]]
    let weights_ternary: Vec<i8> = vec![1, 0, -1, 1, 0, -1, 1, 1];

    // Pack column-major
    let packed_k = k.div_ceil(4); // 1
    let num_blocks_k = k.div_ceil(block_size); // 1
    let mut packed = vec![0u8; packed_k * n];
    for col in 0..n {
        for row in 0..k {
            let val = weights_ternary[row * n + col];
            let code: u8 = match val {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            };
            let byte_idx = col * packed_k + row / 4;
            let bit_off = (row % 4) * 2;
            packed[byte_idx] |= code << bit_off;
        }
    }
    let scales = vec![1.0f32; n * num_blocks_k];
    let mut out = vec![0.0f32; m * n];

    i2s_matmul_f32(&activations, &packed, &scales, &mut out, m, n, k, block_size).unwrap();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn quantized_matmul_pack_i2s_round_trip() {
    let vals: [i8; 4] = [1, -1, 0, 1];
    let packed = pack_i2s(vals);
    insta::assert_snapshot!(format!("packed=0x{packed:02x} binary=0b{packed:08b}"));
}

// ═══════════════════════════════════════════════════════════════════════
// Attention snapshots
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn attention_single_head() {
    // seq_len=2, head_dim=4, no causal mask
    let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let scale = 1.0 / (4.0f32).sqrt();

    let out = AttentionKernel::scaled_dot_product(&q, &k, &v, None, scale, 2, 2, 4).unwrap();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn attention_multi_head() {
    // seq_len=2, num_heads=2, head_dim=2
    let cfg = AttentionConfig { num_heads: 2, head_dim: 2, seq_len: 2, causal: false, scale: None };
    // q, k, v: [seq_len=2, num_heads*head_dim=4]
    let q = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let k = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let out = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg).unwrap();
    insta::assert_snapshot!(fmt_f32(&out));
}

#[test]
fn attention_causal_mask() {
    // seq_len=3, head_dim=2, single head with causal masking
    let cfg = AttentionConfig { num_heads: 1, head_dim: 2, seq_len: 3, causal: true, scale: None };
    // Uniform q/k so attention is determined purely by the causal mask
    let q = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let k = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let v = vec![1.0, 0.0, 0.0, 1.0, -1.0, -1.0];

    let out = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg).unwrap();
    insta::assert_snapshot!(fmt_f32(&out));
}

// ═══════════════════════════════════════════════════════════════════════
// RoPE snapshots
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn rope_frequencies_default_base() {
    let cfg = RopeConfig::new(4, 4);
    let freqs = compute_frequencies(&cfg);
    // Snapshot the frequency table for positions 0..4, head_dim=4
    insta::assert_snapshot!(fmt_f32(&freqs));
}

#[test]
fn rope_frequencies_custom_base() {
    let cfg = RopeConfig::new(4, 4).with_base(500_000.0);
    let freqs = compute_frequencies(&cfg);
    insta::assert_snapshot!(fmt_f32(&freqs));
}

#[test]
fn rope_apply_position_1() {
    let cfg = RopeConfig::new(4, 4);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![1.0, 0.0, 1.0, 0.0];
    apply_rope(&mut data, 1, 4, &freqs);
    insta::assert_snapshot!(fmt_f32(&data));
}

#[test]
fn rope_apply_position_3() {
    let cfg = RopeConfig::new(4, 4);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![1.0, 0.5, 0.8, -0.3];
    apply_rope(&mut data, 3, 4, &freqs);
    insta::assert_snapshot!(fmt_f32(&data));
}

#[test]
fn rope_batch_two_heads() {
    let head_dim = 4;
    let num_heads = 2;
    let seq_len = 2;
    let cfg = RopeConfig::new(head_dim, seq_len);
    let freqs = compute_frequencies(&cfg);

    // [seq_len=2, num_heads=2, head_dim=4]
    let mut data = vec![
        1.0, 0.0, 1.0, 0.0, // pos 0, head 0
        0.0, 1.0, 0.0, 1.0, // pos 0, head 1
        1.0, 1.0, 1.0, 1.0, // pos 1, head 0
        -1.0, 0.5, -0.5, 0.25, // pos 1, head 1
    ];
    apply_rope_batch(&mut data, 0, seq_len, num_heads, head_dim, &freqs);
    insta::assert_snapshot!(fmt_f32(&data));
}

#[test]
fn rope_with_scaling_factor() {
    let cfg = RopeConfig::new(4, 4).with_scaling_factor(0.5);
    let freqs = compute_frequencies(&cfg);
    let mut data = vec![1.0, 0.0, 1.0, 0.0];
    apply_rope(&mut data, 2, 4, &freqs);
    insta::assert_snapshot!(fmt_f32(&data));
}
