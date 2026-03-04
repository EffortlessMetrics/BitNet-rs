//! NEON-optimized quantized feed-forward network kernels for Apple Silicon.
//!
//! Provides fused quantized FFN primitives (SwiGLU, gated MLP, residual add)
//! that operate on `i8` quantized weights with `f32` scales, using NEON SIMD
//! intrinsics on `aarch64` with scalar fallbacks on other architectures.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use std::arch::aarch64::*;

// ── Scalar helpers ──────────────────────────────────────────────────────

#[inline(always)]
fn scalar_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn scalar_silu(x: f32) -> f32 {
    x * scalar_sigmoid(x)
}

// ── NEON fast exp / sigmoid / silu ──────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_fast_exp(x: float32x4_t) -> float32x4_t {
    let min_val = unsafe { vdupq_n_f32(-88.0) };
    let max_val = unsafe { vdupq_n_f32(88.0) };
    let x = unsafe { vmaxq_f32(vminq_f32(x, max_val), min_val) };

    let log2e = unsafe { vdupq_n_f32(std::f32::consts::LOG2_E) };
    let ln2 = unsafe { vdupq_n_f32(std::f32::consts::LN_2) };

    let t = unsafe { vmulq_f32(x, log2e) };
    let n = unsafe { vrndmq_f32(t) };
    let f = unsafe { vmlsq_f32(x, n, ln2) };

    let c1 = unsafe { vdupq_n_f32(1.0) };
    let c2 = unsafe { vdupq_n_f32(0.5) };
    let c3 = unsafe { vdupq_n_f32(1.0 / 6.0) };
    let c4 = unsafe { vdupq_n_f32(1.0 / 24.0) };

    let mut p = unsafe { vmlaq_f32(c3, f, c4) };
    p = unsafe { vmlaq_f32(c2, f, p) };
    p = unsafe { vmlaq_f32(c1, f, p) };
    p = unsafe { vmlaq_f32(c1, f, p) };

    let n_i = unsafe { vcvtq_s32_f32(n) };
    let bias = unsafe { vdupq_n_s32(127) };
    let pow2n = unsafe { vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n_i, bias), 23)) };

    unsafe { vmulq_f32(p, pow2n) }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_sigmoid(x: float32x4_t) -> float32x4_t {
    let one = unsafe { vdupq_n_f32(1.0) };
    let neg_x = unsafe { vnegq_f32(x) };
    let exp_neg = unsafe { neon_fast_exp(neg_x) };
    let denom = unsafe { vaddq_f32(one, exp_neg) };
    unsafe { vdivq_f32(one, denom) }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_silu(x: float32x4_t) -> float32x4_t {
    let sig = unsafe { neon_sigmoid(x) };
    unsafe { vmulq_f32(x, sig) }
}

// ── quantized_linear_neon ───────────────────────────────────────────────

/// Quantized linear layer: `output[i] = sum_j(input[j] * weights[i*in_dim+j]) * scale`
///
/// `weights` is row-major `[out_dim, in_dim]` stored as `i8`.
pub fn quantized_linear_neon(
    input: &[f32],
    weights: &[i8],
    scale: f32,
    output: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    assert!(input.len() >= in_dim);
    assert!(weights.len() >= out_dim * in_dim);
    assert!(output.len() >= out_dim);

    #[cfg(target_arch = "aarch64")]
    {
        neon_quantized_linear_impl(input, weights, scale, output, in_dim, out_dim);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_quantized_linear(input, weights, scale, output, in_dim, out_dim);
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_quantized_linear_impl(
    input: &[f32],
    weights: &[i8],
    scale: f32,
    output: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    let chunks = in_dim / 4;
    let remainder = in_dim % 4;

    for row in 0..out_dim {
        let row_off = row * in_dim;
        let mut acc = unsafe { vdupq_n_f32(0.0) };

        for c in 0..chunks {
            let base = c * 4;
            let inp = unsafe { vld1q_f32(input.as_ptr().add(base)) };
            // Convert 4 i8 weights to f32
            let w0 = weights[row_off + base] as f32;
            let w1 = weights[row_off + base + 1] as f32;
            let w2 = weights[row_off + base + 2] as f32;
            let w3 = weights[row_off + base + 3] as f32;
            let wv = unsafe {
                let mut tmp = [0.0f32; 4];
                tmp[0] = w0;
                tmp[1] = w1;
                tmp[2] = w2;
                tmp[3] = w3;
                vld1q_f32(tmp.as_ptr())
            };
            acc = unsafe { vfmaq_f32(acc, inp, wv) };
        }

        // Horizontal sum
        let mut sum: f32 = unsafe {
            let pair = vpaddq_f32(acc, acc);
            vgetq_lane_f32(vpaddq_f32(pair, pair), 0)
        };

        // Scalar tail
        for t in 0..remainder {
            let idx = chunks * 4 + t;
            sum += input[idx] * weights[row_off + idx] as f32;
        }

        output[row] = sum * scale;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn scalar_quantized_linear(
    input: &[f32],
    weights: &[i8],
    scale: f32,
    output: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    for row in 0..out_dim {
        let row_off = row * in_dim;
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            sum += input[j] * weights[row_off + j] as f32;
        }
        output[row] = sum * scale;
    }
}

// Reference scalar implementation used by tests on all platforms.
#[allow(dead_code)]
fn scalar_quantized_linear_ref(
    input: &[f32],
    weights: &[i8],
    scale: f32,
    output: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    for row in 0..out_dim {
        let row_off = row * in_dim;
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            sum += input[j] * weights[row_off + j] as f32;
        }
        output[row] = sum * scale;
    }
}

// ── quantized_swiglu_neon ───────────────────────────────────────────────

/// Fused SwiGLU activation: `output[i] = silu(gate[i]) * up[i]`
pub fn quantized_swiglu_neon(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let len = gate.len().min(up.len()).min(output.len());

    #[cfg(target_arch = "aarch64")]
    {
        neon_swiglu_impl(gate, up, output, len);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_swiglu(gate, up, output, len);
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_swiglu_impl(gate: &[f32], up: &[f32], output: &mut [f32], len: usize) {
    let chunks = len / 4;
    let remainder = len % 4;

    for c in 0..chunks {
        let base = c * 4;
        unsafe {
            let g = vld1q_f32(gate.as_ptr().add(base));
            let u = vld1q_f32(up.as_ptr().add(base));
            let silu_g = neon_silu(g);
            let result = vmulq_f32(silu_g, u);
            vst1q_f32(output.as_mut_ptr().add(base), result);
        }
    }

    for t in 0..remainder {
        let idx = chunks * 4 + t;
        output[idx] = scalar_silu(gate[idx]) * up[idx];
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn scalar_swiglu(gate: &[f32], up: &[f32], output: &mut [f32], len: usize) {
    for i in 0..len {
        output[i] = scalar_silu(gate[i]) * up[i];
    }
}

#[allow(dead_code)]
fn scalar_swiglu_ref(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let len = gate.len().min(up.len()).min(output.len());
    for i in 0..len {
        output[i] = scalar_silu(gate[i]) * up[i];
    }
}

// ── quantized_ffn_forward_neon ──────────────────────────────────────────

/// Full FFN forward pass with SwiGLU: gate projection → up projection → SwiGLU → down projection.
///
/// `w_gate`: `[intermediate_dim, hidden_dim]` row-major i8
/// `w_up`:   `[intermediate_dim, hidden_dim]` row-major i8
/// `w_down`: `[hidden_dim, intermediate_dim]` row-major i8
#[allow(clippy::too_many_arguments)]
pub fn quantized_ffn_forward_neon(
    input: &[f32],
    w_gate: &[i8],
    w_up: &[i8],
    w_down: &[i8],
    scale_gate: f32,
    scale_up: f32,
    scale_down: f32,
    output: &mut [f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) {
    assert!(input.len() >= hidden_dim);
    assert!(w_gate.len() >= intermediate_dim * hidden_dim);
    assert!(w_up.len() >= intermediate_dim * hidden_dim);
    assert!(w_down.len() >= hidden_dim * intermediate_dim);
    assert!(output.len() >= hidden_dim);

    let mut gate_proj = vec![0.0f32; intermediate_dim];
    let mut up_proj = vec![0.0f32; intermediate_dim];
    let mut swiglu_out = vec![0.0f32; intermediate_dim];

    quantized_linear_neon(input, w_gate, scale_gate, &mut gate_proj, hidden_dim, intermediate_dim);
    quantized_linear_neon(input, w_up, scale_up, &mut up_proj, hidden_dim, intermediate_dim);
    quantized_swiglu_neon(&gate_proj, &up_proj, &mut swiglu_out);
    quantized_linear_neon(&swiglu_out, w_down, scale_down, output, intermediate_dim, hidden_dim);
}

// ── quantized_gated_mlp_neon ────────────────────────────────────────────

/// Gated MLP: w1 (gate) projection → w3 (up) projection → SwiGLU → w2 (down) projection.
///
/// Follows the common naming convention where w1=gate, w3=up, w2=down.
#[allow(clippy::too_many_arguments)]
pub fn quantized_gated_mlp_neon(
    input: &[f32],
    w1: &[i8],
    w2: &[i8],
    w3: &[i8],
    s1: f32,
    s2: f32,
    s3: f32,
    output: &mut [f32],
    dim: usize,
    inter_dim: usize,
) {
    // w1 = gate [inter_dim, dim], w3 = up [inter_dim, dim], w2 = down [dim, inter_dim]
    quantized_ffn_forward_neon(input, w1, w3, w2, s1, s3, s2, output, dim, inter_dim);
}

// ── quantized_ffn_residual_neon ─────────────────────────────────────────

/// Residual connection: `output[i] = input[i] + alpha * ffn_output[i]`
pub fn quantized_ffn_residual_neon(
    input: &[f32],
    ffn_output: &[f32],
    output: &mut [f32],
    alpha: f32,
) {
    let len = input.len().min(ffn_output.len()).min(output.len());

    #[cfg(target_arch = "aarch64")]
    {
        neon_residual_impl(input, ffn_output, output, alpha, len);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_residual(input, ffn_output, output, alpha, len);
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_residual_impl(
    input: &[f32],
    ffn_output: &[f32],
    output: &mut [f32],
    alpha: f32,
    len: usize,
) {
    let chunks = len / 4;
    let remainder = len % 4;
    let alpha_v = unsafe { vdupq_n_f32(alpha) };

    for c in 0..chunks {
        let base = c * 4;
        unsafe {
            let inp = vld1q_f32(input.as_ptr().add(base));
            let ffn = vld1q_f32(ffn_output.as_ptr().add(base));
            let scaled = vmulq_f32(ffn, alpha_v);
            let result = vaddq_f32(inp, scaled);
            vst1q_f32(output.as_mut_ptr().add(base), result);
        }
    }

    for t in 0..remainder {
        let idx = chunks * 4 + t;
        output[idx] = input[idx] + alpha * ffn_output[idx];
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn scalar_residual(input: &[f32], ffn_output: &[f32], output: &mut [f32], alpha: f32, len: usize) {
    for i in 0..len {
        output[i] = input[i] + alpha * ffn_output[i];
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    const EPS: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn assert_slices_approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(approx_eq(*x, *y, tol), "index {i}: {x} vs {y} (diff={})", (x - y).abs());
        }
    }

    // ── quantized_linear_neon tests ─────────────────────────────────

    #[test]
    fn test_linear_identity_weights() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights: Vec<i8> = vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        let mut output = vec![0.0f32; 4];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 4, 4);
        assert_slices_approx(&output, &[1.0, 2.0, 3.0, 4.0], EPS);
    }

    #[test]
    fn test_linear_all_ones() {
        let input = vec![1.0; 8];
        let weights = vec![1i8; 8 * 4];
        let mut output = vec![0.0f32; 4];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 8, 4);
        for v in &output {
            assert!(approx_eq(*v, 8.0, EPS));
        }
    }

    #[test]
    fn test_linear_scale_factor() {
        let input = vec![1.0; 4];
        let weights = vec![1i8; 4 * 2];
        let mut output = vec![0.0f32; 2];
        quantized_linear_neon(&input, &weights, 0.5, &mut output, 4, 2);
        for v in &output {
            assert!(approx_eq(*v, 2.0, EPS));
        }
    }

    #[test]
    fn test_linear_negative_weights() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![-1i8; 4];
        let mut output = vec![0.0f32; 1];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 4, 1);
        assert!(approx_eq(output[0], -10.0, EPS));
    }

    #[test]
    fn test_linear_zero_weights() {
        let input = vec![5.0; 4];
        let weights = vec![0i8; 4 * 2];
        let mut output = vec![99.0f32; 2];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 4, 2);
        for v in &output {
            assert!(approx_eq(*v, 0.0, EPS));
        }
    }

    #[test]
    fn test_linear_zero_input() {
        let input = vec![0.0f32; 4];
        let weights = vec![1i8; 4 * 2];
        let mut output = vec![99.0f32; 2];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 4, 2);
        for v in &output {
            assert!(approx_eq(*v, 0.0, EPS));
        }
    }

    #[test]
    fn test_linear_single_element() {
        let input = vec![3.0f32];
        let weights = vec![2i8];
        let mut output = vec![0.0f32; 1];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 1, 1);
        assert!(approx_eq(output[0], 6.0, EPS));
    }

    #[test]
    fn test_linear_non_multiple_of_4() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights: Vec<i8> = vec![1, 1, 1, 1, 1];
        let mut output = vec![0.0f32; 1];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 5, 1);
        assert!(approx_eq(output[0], 15.0, EPS));
    }

    #[test]
    fn test_linear_matches_scalar() {
        let input = vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.5, 0.0, -2.0];
        let weights: Vec<i8> = vec![1, -1, 0, 1, -1, 1, 1, 0, 0, 1, -1, 1, -1, 0, 1, -1];
        let mut neon_out = vec![0.0f32; 2];
        let mut scalar_out = vec![0.0f32; 2];
        quantized_linear_neon(&input, &weights, 0.75, &mut neon_out, 8, 2);
        scalar_quantized_linear_ref(&input, &weights, 0.75, &mut scalar_out, 8, 2);
        assert_slices_approx(&neon_out, &scalar_out, EPS);
    }

    #[test]
    fn test_linear_large_dim() {
        let in_dim = 64;
        let out_dim = 16;
        let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
        let weights: Vec<i8> = (0..out_dim * in_dim).map(|i| ((i % 3) as i8 - 1)).collect();
        let mut neon_out = vec![0.0f32; out_dim];
        let mut scalar_out = vec![0.0f32; out_dim];
        quantized_linear_neon(&input, &weights, 1.0, &mut neon_out, in_dim, out_dim);
        scalar_quantized_linear_ref(&input, &weights, 1.0, &mut scalar_out, in_dim, out_dim);
        assert_slices_approx(&neon_out, &scalar_out, EPS);
    }

    #[test]
    fn test_linear_zero_scale() {
        let input = vec![1.0; 4];
        let weights = vec![1i8; 4];
        let mut output = vec![99.0f32; 1];
        quantized_linear_neon(&input, &weights, 0.0, &mut output, 4, 1);
        assert!(approx_eq(output[0], 0.0, EPS));
    }

    #[test]
    fn test_linear_negative_scale() {
        let input = vec![1.0; 4];
        let weights = vec![1i8; 4];
        let mut output = vec![0.0f32; 1];
        quantized_linear_neon(&input, &weights, -2.0, &mut output, 4, 1);
        assert!(approx_eq(output[0], -8.0, EPS));
    }

    #[test]
    fn test_linear_mixed_signs() {
        let input = vec![1.0, -1.0, 1.0, -1.0];
        let weights: Vec<i8> = vec![1, 1, -1, -1];
        let mut output = vec![0.0f32; 1];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 4, 1);
        // 1*1 + (-1)*1 + 1*(-1) + (-1)*(-1) = 1 - 1 - 1 + 1 = 0
        assert!(approx_eq(output[0], 0.0, EPS));
    }

    #[test]
    fn test_linear_dim_7() {
        let input = vec![1.0; 7];
        let weights = vec![1i8; 7 * 3];
        let mut output = vec![0.0f32; 3];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 7, 3);
        for v in &output {
            assert!(approx_eq(*v, 7.0, EPS));
        }
    }

    #[test]
    fn test_linear_max_weight_values() {
        let input = vec![1.0; 4];
        let weights = vec![127i8; 4];
        let mut output = vec![0.0f32; 1];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 4, 1);
        assert!(approx_eq(output[0], 508.0, EPS));
    }

    #[test]
    fn test_linear_min_weight_values() {
        let input = vec![1.0; 4];
        let weights = vec![-128i8; 4];
        let mut output = vec![0.0f32; 1];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 4, 1);
        assert!(approx_eq(output[0], -512.0, EPS));
    }

    // ── quantized_swiglu_neon tests ─────────────────────────────────

    #[test]
    fn test_swiglu_zeros() {
        let gate = vec![0.0f32; 8];
        let up = vec![1.0f32; 8];
        let mut output = vec![0.0f32; 8];
        quantized_swiglu_neon(&gate, &up, &mut output);
        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        for v in &output {
            assert!(approx_eq(*v, 0.0, EPS));
        }
    }

    #[test]
    fn test_swiglu_positive_gate() {
        let gate = vec![2.0f32; 4];
        let up = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        quantized_swiglu_neon(&gate, &up, &mut output);
        let expected = scalar_silu(2.0);
        for v in &output {
            assert!(approx_eq(*v, expected, EPS));
        }
    }

    #[test]
    fn test_swiglu_negative_gate() {
        let gate = vec![-3.0f32; 4];
        let up = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        quantized_swiglu_neon(&gate, &up, &mut output);
        let expected = scalar_silu(-3.0);
        for v in &output {
            assert!(approx_eq(*v, expected, EPS));
        }
    }

    #[test]
    fn test_swiglu_up_scaling() {
        let gate = vec![1.0f32; 4];
        let up = vec![3.0f32; 4];
        let mut output = vec![0.0f32; 4];
        quantized_swiglu_neon(&gate, &up, &mut output);
        let expected = scalar_silu(1.0) * 3.0;
        for v in &output {
            assert!(approx_eq(*v, expected, EPS));
        }
    }

    #[test]
    fn test_swiglu_matches_scalar() {
        let gate = vec![0.5, -1.0, 2.0, -0.3, 0.7, -2.0, 1.5, 0.0];
        let up = vec![1.0, 2.0, -1.0, 0.5, -0.5, 3.0, -2.0, 1.0];
        let mut neon_out = vec![0.0f32; 8];
        let mut scalar_out = vec![0.0f32; 8];
        quantized_swiglu_neon(&gate, &up, &mut neon_out);
        scalar_swiglu_ref(&gate, &up, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, EPS);
    }

    #[test]
    fn test_swiglu_non_multiple_of_4() {
        let gate = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let up = vec![1.0; 5];
        let mut output = vec![0.0f32; 5];
        quantized_swiglu_neon(&gate, &up, &mut output);
        for (i, &g) in gate.iter().enumerate() {
            assert!(approx_eq(output[i], scalar_silu(g), EPS));
        }
    }

    #[test]
    fn test_swiglu_single_element() {
        let gate = vec![1.5f32];
        let up = vec![2.0f32];
        let mut output = vec![0.0f32; 1];
        quantized_swiglu_neon(&gate, &up, &mut output);
        assert!(approx_eq(output[0], scalar_silu(1.5) * 2.0, EPS));
    }

    #[test]
    fn test_swiglu_large_positive() {
        let gate = vec![10.0f32; 4];
        let up = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        quantized_swiglu_neon(&gate, &up, &mut output);
        // silu(10) ≈ 10.0 (sigmoid(10) ≈ 1.0)
        for v in &output {
            assert!(approx_eq(*v, scalar_silu(10.0), EPS));
        }
    }

    #[test]
    fn test_swiglu_large_negative() {
        let gate = vec![-10.0f32; 4];
        let up = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        quantized_swiglu_neon(&gate, &up, &mut output);
        // silu(-10) ≈ 0.0
        for v in &output {
            assert!(approx_eq(*v, scalar_silu(-10.0), EPS));
        }
    }

    #[test]
    fn test_swiglu_both_negative() {
        let gate = vec![-1.0f32; 4];
        let up = vec![-2.0f32; 4];
        let mut output = vec![0.0f32; 4];
        quantized_swiglu_neon(&gate, &up, &mut output);
        let expected = scalar_silu(-1.0) * (-2.0);
        for v in &output {
            assert!(approx_eq(*v, expected, EPS));
        }
    }

    #[test]
    fn test_swiglu_symmetry() {
        let gate_pos = vec![2.0f32; 4];
        let gate_neg = vec![-2.0f32; 4];
        let up = vec![1.0f32; 4];
        let mut out_pos = vec![0.0f32; 4];
        let mut out_neg = vec![0.0f32; 4];
        quantized_swiglu_neon(&gate_pos, &up, &mut out_pos);
        quantized_swiglu_neon(&gate_neg, &up, &mut out_neg);
        // silu is NOT symmetric but we can check relationship
        // silu(x) + silu(-x) = x*sig(x) + (-x)*sig(-x) = x*(sig(x) - sig(-x)) = x*(2*sig(x) - 1)
        // Just verify they have expected signs
        assert!(out_pos[0] > 0.0);
        assert!(out_neg[0] < 0.0);
    }

    #[test]
    fn test_swiglu_zero_up() {
        let gate = vec![5.0f32; 4];
        let up = vec![0.0f32; 4];
        let mut output = vec![99.0f32; 4];
        quantized_swiglu_neon(&gate, &up, &mut output);
        for v in &output {
            assert!(approx_eq(*v, 0.0, EPS));
        }
    }

    #[test]
    fn test_swiglu_16_elements() {
        let gate: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 2.0).collect();
        let up: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let mut neon_out = vec![0.0f32; 16];
        let mut scalar_out = vec![0.0f32; 16];
        quantized_swiglu_neon(&gate, &up, &mut neon_out);
        scalar_swiglu_ref(&gate, &up, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, EPS);
    }

    // ── quantized_ffn_residual_neon tests ───────────────────────────

    #[test]
    fn test_residual_alpha_one() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let ffn = vec![0.5, 0.5, 0.5, 0.5];
        let mut output = vec![0.0f32; 4];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 1.0);
        assert_slices_approx(&output, &[1.5, 2.5, 3.5, 4.5], EPS);
    }

    #[test]
    fn test_residual_alpha_zero() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let ffn = vec![10.0; 4];
        let mut output = vec![0.0f32; 4];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 0.0);
        assert_slices_approx(&output, &input, EPS);
    }

    #[test]
    fn test_residual_alpha_half() {
        let input = vec![2.0; 4];
        let ffn = vec![4.0; 4];
        let mut output = vec![0.0f32; 4];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 0.5);
        // 2.0 + 0.5 * 4.0 = 4.0
        for v in &output {
            assert!(approx_eq(*v, 4.0, EPS));
        }
    }

    #[test]
    fn test_residual_negative_alpha() {
        let input = vec![3.0; 4];
        let ffn = vec![1.0; 4];
        let mut output = vec![0.0f32; 4];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, -1.0);
        for v in &output {
            assert!(approx_eq(*v, 2.0, EPS));
        }
    }

    #[test]
    fn test_residual_non_multiple_of_4() {
        let input = vec![1.0; 7];
        let ffn = vec![2.0; 7];
        let mut output = vec![0.0f32; 7];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 1.0);
        for v in &output {
            assert!(approx_eq(*v, 3.0, EPS));
        }
    }

    #[test]
    fn test_residual_zeros() {
        let input = vec![0.0; 8];
        let ffn = vec![0.0; 8];
        let mut output = vec![99.0f32; 8];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 1.0);
        for v in &output {
            assert!(approx_eq(*v, 0.0, EPS));
        }
    }

    #[test]
    fn test_residual_single_element() {
        let input = vec![5.0f32];
        let ffn = vec![3.0f32];
        let mut output = vec![0.0f32; 1];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 2.0);
        assert!(approx_eq(output[0], 11.0, EPS));
    }

    #[test]
    fn test_residual_negative_values() {
        let input = vec![-1.0, -2.0, -3.0, -4.0];
        let ffn = vec![-1.0, -1.0, -1.0, -1.0];
        let mut output = vec![0.0f32; 4];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 1.0);
        assert_slices_approx(&output, &[-2.0, -3.0, -4.0, -5.0], EPS);
    }

    #[test]
    fn test_residual_large_alpha() {
        let input = vec![1.0; 4];
        let ffn = vec![0.001; 4];
        let mut output = vec![0.0f32; 4];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 1000.0);
        for v in &output {
            assert!(approx_eq(*v, 2.0, EPS));
        }
    }

    #[test]
    fn test_residual_16_elements() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let ffn: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0f32; 16];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 1.0);
        for i in 0..16 {
            let expected = input[i] + ffn[i];
            assert!(approx_eq(output[i], expected, EPS));
        }
    }

    #[test]
    fn test_residual_in_place_semantics() {
        // Verify output doesn't depend on its initial value
        let input = vec![1.0; 4];
        let ffn = vec![2.0; 4];
        let mut out1 = vec![0.0f32; 4];
        let mut out2 = vec![999.0f32; 4];
        quantized_ffn_residual_neon(&input, &ffn, &mut out1, 1.0);
        quantized_ffn_residual_neon(&input, &ffn, &mut out2, 1.0);
        assert_slices_approx(&out1, &out2, EPS);
    }

    #[test]
    fn test_residual_dim_5() {
        let input = vec![1.0; 5];
        let ffn = vec![1.0; 5];
        let mut output = vec![0.0f32; 5];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 0.5);
        for v in &output {
            assert!(approx_eq(*v, 1.5, EPS));
        }
    }

    // ── quantized_ffn_forward_neon tests ────────────────────────────

    #[test]
    fn test_ffn_forward_zero_input() {
        let hidden = 4;
        let inter = 4;
        let input = vec![0.0f32; hidden];
        let w_gate = vec![1i8; inter * hidden];
        let w_up = vec![1i8; inter * hidden];
        let w_down = vec![1i8; hidden * inter];
        let mut output = vec![99.0f32; hidden];
        quantized_ffn_forward_neon(
            &input,
            &w_gate,
            &w_up,
            &w_down,
            1.0,
            1.0,
            1.0,
            &mut output,
            hidden,
            inter,
        );
        for v in &output {
            assert!(approx_eq(*v, 0.0, EPS));
        }
    }

    #[test]
    fn test_ffn_forward_zero_weights() {
        let hidden = 4;
        let inter = 4;
        let input = vec![1.0f32; hidden];
        let w_gate = vec![0i8; inter * hidden];
        let w_up = vec![0i8; inter * hidden];
        let w_down = vec![0i8; hidden * inter];
        let mut output = vec![99.0f32; hidden];
        quantized_ffn_forward_neon(
            &input,
            &w_gate,
            &w_up,
            &w_down,
            1.0,
            1.0,
            1.0,
            &mut output,
            hidden,
            inter,
        );
        for v in &output {
            assert!(approx_eq(*v, 0.0, EPS));
        }
    }

    #[test]
    fn test_ffn_forward_scale_propagation() {
        let hidden = 4;
        let inter = 4;
        let input = vec![1.0f32; hidden];
        let w_gate = vec![1i8; inter * hidden];
        let w_up = vec![1i8; inter * hidden];
        let w_down = vec![1i8; hidden * inter];
        let mut out1 = vec![0.0f32; hidden];
        let mut out2 = vec![0.0f32; hidden];
        quantized_ffn_forward_neon(
            &input, &w_gate, &w_up, &w_down, 1.0, 1.0, 1.0, &mut out1, hidden, inter,
        );
        quantized_ffn_forward_neon(
            &input, &w_gate, &w_up, &w_down, 0.0, 1.0, 1.0, &mut out2, hidden, inter,
        );
        // When gate scale is 0, gate projection is all zeros, silu(0)=0 → output all zeros
        for v in &out2 {
            assert!(approx_eq(*v, 0.0, EPS));
        }
    }

    #[test]
    fn test_ffn_forward_negative_weights() {
        let hidden = 4;
        let inter = 4;
        let input = vec![1.0f32; hidden];
        let w_gate = vec![-1i8; inter * hidden];
        let w_up = vec![1i8; inter * hidden];
        let w_down = vec![1i8; hidden * inter];
        let mut output = vec![0.0f32; hidden];
        quantized_ffn_forward_neon(
            &input,
            &w_gate,
            &w_up,
            &w_down,
            1.0,
            1.0,
            1.0,
            &mut output,
            hidden,
            inter,
        );
        // gate projection = -4.0 per row, silu(-4.0) ≈ small negative
        // Verify computation completes and produces finite values
        for v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ffn_forward_dims_8_16() {
        let hidden = 8;
        let inter = 16;
        let input: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.1).collect();
        let w_gate: Vec<i8> = (0..inter * hidden).map(|i| ((i % 3) as i8 - 1)).collect();
        let w_up: Vec<i8> = (0..inter * hidden).map(|i| ((i % 5) as i8 - 2)).collect();
        let w_down: Vec<i8> = (0..hidden * inter).map(|i| ((i % 3) as i8 - 1)).collect();
        let mut output = vec![0.0f32; hidden];
        quantized_ffn_forward_neon(
            &input,
            &w_gate,
            &w_up,
            &w_down,
            0.5,
            0.5,
            0.5,
            &mut output,
            hidden,
            inter,
        );
        for v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ffn_forward_matches_manual() {
        // Manually compute for tiny dims: hidden=2, inter=2
        let hidden = 2;
        let inter = 2;
        let input = vec![1.0, 2.0];
        let w_gate: Vec<i8> = vec![1, 0, 0, 1]; // [[1,0],[0,1]]
        let w_up: Vec<i8> = vec![1, 1, 1, 1]; // [[1,1],[1,1]]
        let w_down: Vec<i8> = vec![1, 0, 0, 1]; // [[1,0],[0,1]]
        let scale = 1.0;

        // gate_proj = [1*1+0*2, 0*1+1*2] = [1.0, 2.0]
        // up_proj   = [1*1+1*2, 1*1+1*2] = [3.0, 3.0]
        // swiglu    = [silu(1.0)*3.0, silu(2.0)*3.0]
        let swiglu0 = scalar_silu(1.0) * 3.0;
        let swiglu1 = scalar_silu(2.0) * 3.0;
        // down      = [1*sw0+0*sw1, 0*sw0+1*sw1] = [sw0, sw1]

        let mut output = vec![0.0f32; hidden];
        quantized_ffn_forward_neon(
            &input,
            &w_gate,
            &w_up,
            &w_down,
            scale,
            scale,
            scale,
            &mut output,
            hidden,
            inter,
        );
        assert!(approx_eq(output[0], swiglu0, EPS));
        assert!(approx_eq(output[1], swiglu1, EPS));
    }

    #[test]
    fn test_ffn_forward_down_scale_zero() {
        let hidden = 4;
        let inter = 4;
        let input = vec![1.0f32; hidden];
        let w_gate = vec![1i8; inter * hidden];
        let w_up = vec![1i8; inter * hidden];
        let w_down = vec![1i8; hidden * inter];
        let mut output = vec![99.0f32; hidden];
        quantized_ffn_forward_neon(
            &input,
            &w_gate,
            &w_up,
            &w_down,
            1.0,
            1.0,
            0.0,
            &mut output,
            hidden,
            inter,
        );
        for v in &output {
            assert!(approx_eq(*v, 0.0, EPS));
        }
    }

    #[test]
    fn test_ffn_forward_non_square() {
        let hidden = 4;
        let inter = 8;
        let input = vec![1.0f32; hidden];
        let w_gate = vec![1i8; inter * hidden];
        let w_up = vec![1i8; inter * hidden];
        let w_down = vec![1i8; hidden * inter];
        let mut output = vec![0.0f32; hidden];
        quantized_ffn_forward_neon(
            &input,
            &w_gate,
            &w_up,
            &w_down,
            1.0,
            1.0,
            1.0,
            &mut output,
            hidden,
            inter,
        );
        for v in &output {
            assert!(v.is_finite());
        }
    }

    // ── quantized_gated_mlp_neon tests ──────────────────────────────

    #[test]
    fn test_gated_mlp_zero_input() {
        let dim = 4;
        let inter = 4;
        let input = vec![0.0f32; dim];
        let w1 = vec![1i8; inter * dim];
        let w2 = vec![1i8; dim * inter];
        let w3 = vec![1i8; inter * dim];
        let mut output = vec![99.0f32; dim];
        quantized_gated_mlp_neon(&input, &w1, &w2, &w3, 1.0, 1.0, 1.0, &mut output, dim, inter);
        for v in &output {
            assert!(approx_eq(*v, 0.0, EPS));
        }
    }

    #[test]
    fn test_gated_mlp_matches_ffn() {
        let dim = 4;
        let inter = 8;
        let input: Vec<f32> = vec![0.5, -1.0, 2.0, 0.3];
        let w1: Vec<i8> = (0..inter * dim).map(|i| ((i % 3) as i8 - 1)).collect();
        let w2: Vec<i8> = (0..dim * inter).map(|i| ((i % 3) as i8 - 1)).collect();
        let w3: Vec<i8> = (0..inter * dim).map(|i| ((i % 5) as i8 - 2)).collect();

        let mut mlp_out = vec![0.0f32; dim];
        let mut ffn_out = vec![0.0f32; dim];

        quantized_gated_mlp_neon(&input, &w1, &w2, &w3, 0.5, 0.5, 0.5, &mut mlp_out, dim, inter);
        // gated_mlp maps: w1=gate, w3=up, w2=down
        // ffn_forward expects: w_gate, w_up, w_down, scale_gate, scale_up, scale_down
        quantized_ffn_forward_neon(&input, &w1, &w3, &w2, 0.5, 0.5, 0.5, &mut ffn_out, dim, inter);
        assert_slices_approx(&mlp_out, &ffn_out, EPS);
    }

    #[test]
    fn test_gated_mlp_scale_propagation() {
        let dim = 4;
        let inter = 4;
        let input = vec![1.0f32; dim];
        let w1 = vec![1i8; inter * dim];
        let w2 = vec![1i8; dim * inter];
        let w3 = vec![1i8; inter * dim];
        let mut output = vec![0.0f32; dim];
        quantized_gated_mlp_neon(&input, &w1, &w2, &w3, 0.0, 1.0, 1.0, &mut output, dim, inter);
        // s1=0 → gate projection zeros → silu(0)=0 → output zeros
        for v in &output {
            assert!(approx_eq(*v, 0.0, EPS));
        }
    }

    #[test]
    fn test_gated_mlp_dim_8_16() {
        let dim = 8;
        let inter = 16;
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
        let w1: Vec<i8> = (0..inter * dim).map(|i| ((i % 3) as i8 - 1)).collect();
        let w2: Vec<i8> = (0..dim * inter).map(|i| ((i % 3) as i8 - 1)).collect();
        let w3: Vec<i8> = (0..inter * dim).map(|i| ((i % 5) as i8 - 2)).collect();
        let mut output = vec![0.0f32; dim];
        quantized_gated_mlp_neon(&input, &w1, &w2, &w3, 1.0, 1.0, 1.0, &mut output, dim, inter);
        for v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_gated_mlp_negative_scales() {
        let dim = 4;
        let inter = 4;
        let input = vec![1.0f32; dim];
        let w1 = vec![1i8; inter * dim];
        let w2 = vec![1i8; dim * inter];
        let w3 = vec![1i8; inter * dim];
        let mut output = vec![0.0f32; dim];
        quantized_gated_mlp_neon(&input, &w1, &w2, &w3, -1.0, 1.0, 1.0, &mut output, dim, inter);
        for v in &output {
            assert!(v.is_finite());
        }
    }

    // ── Scalar helper tests ─────────────────────────────────────────

    #[test]
    fn test_scalar_sigmoid_zero() {
        assert!(approx_eq(scalar_sigmoid(0.0), 0.5, EPS));
    }

    #[test]
    fn test_scalar_sigmoid_large_positive() {
        assert!(approx_eq(scalar_sigmoid(20.0), 1.0, EPS));
    }

    #[test]
    fn test_scalar_sigmoid_large_negative() {
        assert!(approx_eq(scalar_sigmoid(-20.0), 0.0, EPS));
    }

    #[test]
    fn test_scalar_silu_zero() {
        assert!(approx_eq(scalar_silu(0.0), 0.0, EPS));
    }

    #[test]
    fn test_scalar_silu_positive() {
        let val = scalar_silu(1.0);
        assert!(val > 0.0);
        assert!(val < 1.0);
    }

    #[test]
    fn test_scalar_silu_negative() {
        let val = scalar_silu(-1.0);
        assert!(val < 0.0);
    }

    // ── Cross-function integration tests ────────────────────────────

    #[test]
    fn test_ffn_then_residual() {
        let hidden = 4;
        let inter = 4;
        let input = vec![1.0f32; hidden];
        let w_gate = vec![1i8; inter * hidden];
        let w_up = vec![1i8; inter * hidden];
        let w_down = vec![1i8; hidden * inter];
        let mut ffn_out = vec![0.0f32; hidden];
        quantized_ffn_forward_neon(
            &input,
            &w_gate,
            &w_up,
            &w_down,
            1.0,
            1.0,
            1.0,
            &mut ffn_out,
            hidden,
            inter,
        );
        let mut final_out = vec![0.0f32; hidden];
        quantized_ffn_residual_neon(&input, &ffn_out, &mut final_out, 1.0);
        for v in &final_out {
            assert!(v.is_finite());
            // Should be input + ffn_out, so > input since silu produces positive for positive gate
            assert!(*v >= input[0]);
        }
    }

    #[test]
    fn test_linear_then_swiglu() {
        let dim = 4;
        let inter = 4;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let w_gate = vec![1i8; inter * dim];
        let w_up = vec![1i8; inter * dim];
        let mut gate_proj = vec![0.0f32; inter];
        let mut up_proj = vec![0.0f32; inter];
        let mut swiglu_out = vec![0.0f32; inter];

        quantized_linear_neon(&input, &w_gate, 1.0, &mut gate_proj, dim, inter);
        quantized_linear_neon(&input, &w_up, 1.0, &mut up_proj, dim, inter);
        quantized_swiglu_neon(&gate_proj, &up_proj, &mut swiglu_out);

        for v in &swiglu_out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_double_residual() {
        let input = vec![1.0; 8];
        let ffn1 = vec![0.5; 8];
        let ffn2 = vec![0.25; 8];
        let mut mid = vec![0.0f32; 8];
        let mut out = vec![0.0f32; 8];
        quantized_ffn_residual_neon(&input, &ffn1, &mut mid, 1.0);
        quantized_ffn_residual_neon(&mid, &ffn2, &mut out, 1.0);
        for v in &out {
            assert!(approx_eq(*v, 1.75, EPS));
        }
    }

    #[test]
    fn test_gated_mlp_then_residual() {
        let dim = 4;
        let inter = 8;
        let input = vec![1.0f32; dim];
        let w1: Vec<i8> = (0..inter * dim).map(|i| ((i % 3) as i8 - 1)).collect();
        let w2: Vec<i8> = (0..dim * inter).map(|i| ((i % 3) as i8 - 1)).collect();
        let w3: Vec<i8> = (0..inter * dim).map(|i| ((i % 5) as i8 - 2)).collect();
        let mut mlp_out = vec![0.0f32; dim];
        quantized_gated_mlp_neon(&input, &w1, &w2, &w3, 1.0, 1.0, 1.0, &mut mlp_out, dim, inter);
        let mut final_out = vec![0.0f32; dim];
        quantized_ffn_residual_neon(&input, &mlp_out, &mut final_out, 0.5);
        for v in &final_out {
            assert!(v.is_finite());
        }
    }

    // ── Edge case tests ─────────────────────────────────────────────

    #[test]
    fn test_linear_large_output_dim() {
        let in_dim = 4;
        let out_dim = 32;
        let input = vec![1.0; in_dim];
        let weights = vec![1i8; out_dim * in_dim];
        let mut output = vec![0.0f32; out_dim];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, in_dim, out_dim);
        for v in &output {
            assert!(approx_eq(*v, 4.0, EPS));
        }
    }

    #[test]
    fn test_linear_dim_1() {
        let input = vec![7.0f32];
        let weights = vec![3i8];
        let mut output = vec![0.0f32; 1];
        quantized_linear_neon(&input, &weights, 2.0, &mut output, 1, 1);
        assert!(approx_eq(output[0], 42.0, EPS));
    }

    #[test]
    fn test_swiglu_dim_3() {
        let gate = vec![1.0, 2.0, 3.0];
        let up = vec![1.0, 1.0, 1.0];
        let mut output = vec![0.0f32; 3];
        quantized_swiglu_neon(&gate, &up, &mut output);
        for (i, &g) in gate.iter().enumerate() {
            assert!(approx_eq(output[i], scalar_silu(g), EPS));
        }
    }

    #[test]
    fn test_residual_dim_3() {
        let input = vec![1.0, 2.0, 3.0];
        let ffn = vec![0.1, 0.2, 0.3];
        let mut output = vec![0.0f32; 3];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 1.0);
        assert_slices_approx(&output, &[1.1, 2.2, 3.3], EPS);
    }

    #[test]
    fn test_ffn_forward_dim_1() {
        let input = vec![1.0f32];
        let w_gate = vec![1i8];
        let w_up = vec![1i8];
        let w_down = vec![1i8];
        let mut output = vec![0.0f32; 1];
        quantized_ffn_forward_neon(
            &input,
            &w_gate,
            &w_up,
            &w_down,
            1.0,
            1.0,
            1.0,
            &mut output,
            1,
            1,
        );
        // gate_proj=[1.0], up_proj=[1.0], swiglu=[silu(1.0)*1.0], down=[silu(1.0)]
        assert!(approx_eq(output[0], scalar_silu(1.0), EPS));
    }

    #[test]
    fn test_linear_alternating_weights() {
        let input = vec![1.0; 8];
        let weights: Vec<i8> = (0..8).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let mut output = vec![0.0f32; 1];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 8, 1);
        // sum = 1-1+1-1+1-1+1-1 = 0
        assert!(approx_eq(output[0], 0.0, EPS));
    }

    #[test]
    fn test_swiglu_gate_one_up_zero() {
        let gate = vec![1.0f32; 4];
        let up = vec![0.0f32; 4];
        let mut output = vec![99.0f32; 4];
        quantized_swiglu_neon(&gate, &up, &mut output);
        for v in &output {
            assert!(approx_eq(*v, 0.0, EPS));
        }
    }

    #[test]
    fn test_residual_with_mixed_signs() {
        let input = vec![1.0, -1.0, 2.0, -2.0];
        let ffn = vec![-1.0, 1.0, -2.0, 2.0];
        let mut output = vec![0.0f32; 4];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 1.0);
        assert_slices_approx(&output, &[0.0, 0.0, 0.0, 0.0], EPS);
    }

    // ── Determinism tests ───────────────────────────────────────────

    #[test]
    fn test_linear_deterministic() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let weights: Vec<i8> = (0..16 * 4).map(|i| ((i % 3) as i8 - 1)).collect();
        let mut out1 = vec![0.0f32; 4];
        let mut out2 = vec![0.0f32; 4];
        quantized_linear_neon(&input, &weights, 0.5, &mut out1, 16, 4);
        quantized_linear_neon(&input, &weights, 0.5, &mut out2, 16, 4);
        assert_slices_approx(&out1, &out2, 0.0);
    }

    #[test]
    fn test_swiglu_deterministic() {
        let gate: Vec<f32> = (0..12).map(|i| (i as f32) * 0.3 - 1.5).collect();
        let up: Vec<f32> = (0..12).map(|i| (i as f32) * 0.2 + 0.1).collect();
        let mut out1 = vec![0.0f32; 12];
        let mut out2 = vec![0.0f32; 12];
        quantized_swiglu_neon(&gate, &up, &mut out1);
        quantized_swiglu_neon(&gate, &up, &mut out2);
        assert_slices_approx(&out1, &out2, 0.0);
    }

    #[test]
    fn test_residual_deterministic() {
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let ffn: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5).collect();
        let mut out1 = vec![0.0f32; 12];
        let mut out2 = vec![0.0f32; 12];
        quantized_ffn_residual_neon(&input, &ffn, &mut out1, 0.7);
        quantized_ffn_residual_neon(&input, &ffn, &mut out2, 0.7);
        assert_slices_approx(&out1, &out2, 0.0);
    }

    #[test]
    fn test_ffn_forward_deterministic() {
        let hidden = 4;
        let inter = 8;
        let input: Vec<f32> = vec![0.5, -1.0, 2.0, 0.3];
        let w_gate: Vec<i8> = (0..inter * hidden).map(|i| ((i % 3) as i8 - 1)).collect();
        let w_up: Vec<i8> = (0..inter * hidden).map(|i| ((i % 5) as i8 - 2)).collect();
        let w_down: Vec<i8> = (0..hidden * inter).map(|i| ((i % 3) as i8 - 1)).collect();
        let mut out1 = vec![0.0f32; hidden];
        let mut out2 = vec![0.0f32; hidden];
        quantized_ffn_forward_neon(
            &input, &w_gate, &w_up, &w_down, 0.5, 0.5, 0.5, &mut out1, hidden, inter,
        );
        quantized_ffn_forward_neon(
            &input, &w_gate, &w_up, &w_down, 0.5, 0.5, 0.5, &mut out2, hidden, inter,
        );
        assert_slices_approx(&out1, &out2, 0.0);
    }

    // ── Additional coverage tests ───────────────────────────────────

    #[test]
    fn test_linear_128_dim() {
        let in_dim = 128;
        let out_dim = 4;
        let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();
        let weights: Vec<i8> = (0..out_dim * in_dim).map(|i| ((i % 3) as i8 - 1)).collect();
        let mut neon_out = vec![0.0f32; out_dim];
        let mut scalar_out = vec![0.0f32; out_dim];
        quantized_linear_neon(&input, &weights, 1.0, &mut neon_out, in_dim, out_dim);
        scalar_quantized_linear_ref(&input, &weights, 1.0, &mut scalar_out, in_dim, out_dim);
        assert_slices_approx(&neon_out, &scalar_out, EPS);
    }

    #[test]
    fn test_swiglu_32_elements() {
        let gate: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let up: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let mut neon_out = vec![0.0f32; 32];
        let mut scalar_out = vec![0.0f32; 32];
        quantized_swiglu_neon(&gate, &up, &mut neon_out);
        scalar_swiglu_ref(&gate, &up, &mut scalar_out);
        assert_slices_approx(&neon_out, &scalar_out, EPS);
    }

    #[test]
    fn test_residual_32_elements() {
        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let ffn: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0f32; 32];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 2.0);
        for i in 0..32 {
            let expected = input[i] + 2.0 * ffn[i];
            assert!(approx_eq(output[i], expected, EPS));
        }
    }

    #[test]
    fn test_linear_fractional_input() {
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let weights = vec![1i8; 8];
        let mut output = vec![0.0f32; 1];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 8, 1);
        let expected: f32 = input.iter().sum();
        assert!(approx_eq(output[0], expected, EPS));
    }

    #[test]
    fn test_linear_rectangular_tall() {
        // More outputs than inputs
        let in_dim = 2;
        let out_dim = 8;
        let input = vec![1.0, 1.0];
        let weights: Vec<i8> = (0..out_dim * in_dim).map(|i| ((i % 3) as i8 - 1)).collect();
        let mut output = vec![0.0f32; out_dim];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, in_dim, out_dim);
        let mut scalar_out = vec![0.0f32; out_dim];
        scalar_quantized_linear_ref(&input, &weights, 1.0, &mut scalar_out, in_dim, out_dim);
        assert_slices_approx(&output, &scalar_out, EPS);
    }

    #[test]
    fn test_swiglu_near_zero_gate() {
        let gate = vec![1e-6f32; 4];
        let up = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        quantized_swiglu_neon(&gate, &up, &mut output);
        // silu(~0) ≈ 0
        for v in &output {
            assert!(v.abs() < 0.01);
        }
    }

    #[test]
    fn test_residual_large_values() {
        let input = vec![1e6f32; 4];
        let ffn = vec![1e6f32; 4];
        let mut output = vec![0.0f32; 4];
        quantized_ffn_residual_neon(&input, &ffn, &mut output, 1.0);
        for v in &output {
            assert!(approx_eq(*v, 2e6, 1.0));
        }
    }

    #[test]
    fn test_ffn_forward_small_scales() {
        let hidden = 4;
        let inter = 4;
        let input = vec![1.0f32; hidden];
        let w_gate = vec![1i8; inter * hidden];
        let w_up = vec![1i8; inter * hidden];
        let w_down = vec![1i8; hidden * inter];
        let mut output = vec![0.0f32; hidden];
        quantized_ffn_forward_neon(
            &input,
            &w_gate,
            &w_up,
            &w_down,
            0.001,
            0.001,
            0.001,
            &mut output,
            hidden,
            inter,
        );
        // With very small scales, output should be near zero
        for v in &output {
            assert!(v.abs() < 0.1);
        }
    }

    #[test]
    fn test_gated_mlp_all_zero_weights() {
        let dim = 4;
        let inter = 4;
        let input = vec![5.0f32; dim];
        let w1 = vec![0i8; inter * dim];
        let w2 = vec![0i8; dim * inter];
        let w3 = vec![0i8; inter * dim];
        let mut output = vec![99.0f32; dim];
        quantized_gated_mlp_neon(&input, &w1, &w2, &w3, 1.0, 1.0, 1.0, &mut output, dim, inter);
        for v in &output {
            assert!(approx_eq(*v, 0.0, EPS));
        }
    }

    #[test]
    fn test_linear_output_overwrites() {
        let input = vec![1.0; 4];
        let weights = vec![1i8; 4 * 2];
        let mut output = vec![12345.0f32; 2];
        quantized_linear_neon(&input, &weights, 1.0, &mut output, 4, 2);
        // Output should be overwritten, not accumulated
        for v in &output {
            assert!(approx_eq(*v, 4.0, EPS));
        }
    }

    #[test]
    fn test_swiglu_mismatched_shorter_output() {
        // Output shorter than gate/up — should only fill output.len()
        let gate = vec![1.0; 8];
        let up = vec![1.0; 8];
        let mut output = vec![0.0f32; 4];
        quantized_swiglu_neon(&gate, &up, &mut output);
        assert_eq!(output.len(), 4);
        for v in &output {
            assert!(approx_eq(*v, scalar_silu(1.0), EPS));
        }
    }
}
