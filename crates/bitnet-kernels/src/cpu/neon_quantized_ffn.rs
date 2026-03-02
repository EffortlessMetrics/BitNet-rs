//! ARM NEON optimized quantized feed-forward network kernel for Apple Silicon.
//!
//! Provides NEON-accelerated FFN operations for ternary-quantized (I2_S)
//! weights, including quantized linear projection, SiLU activation, and a
//! full gated FFN block (gate + up + down projections).
//!
//! I2_S encoding (2 bits per value, 4 values per byte, LSB-first):
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1
//! - `0b10` → unused (treated as 0)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

// ── I2_S decode helper ─────────────────────────────────────────────────

/// Decode a single 2-bit I2_S code to its signed float value.
#[inline(always)]
fn decode_i2s(bits: u8) -> f32 {
    match bits & 0x03 {
        0b01 => 1.0,
        0b11 => -1.0,
        _ => 0.0, // 0b00 = 0, 0b10 = unused → 0
    }
}

// ── Quantized linear layer ─────────────────────────────────────────────

/// Compute a quantized linear projection: `Y = X · Wᵀ · diag(scales) + bias`.
///
/// Performs a ternary matrix multiplication where the weight matrix is
/// I2_S packed (2 bits per value, 4 values per byte, LSB-first), using
/// NEON FMA intrinsics for the inner dot products.
///
/// # Arguments
///
/// - `input`: row-major input matrix, shape `[m, k]`
/// - `weights_i2`: I2_S packed weight matrix, row-major, shape
///   `[n, ceil(k/4)]` bytes (each output row is `ceil(k/4)` bytes)
/// - `scales`: per-output-row scale factors, length `n`
/// - `bias`: optional per-output-row bias, length `n`
/// - `m`: number of input rows (batch size)
/// - `n`: number of output columns
/// - `k`: inner dimension (input columns / weight columns before packing)
///
/// # Returns
///
/// Flat `Vec<f32>` of shape `[m, n]` in row-major order.
pub fn neon_quantized_linear(
    input: &[f32],
    weights_i2: &[u8],
    scales: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert!(input.len() >= m * k, "input too small: need {}, got {}", m * k, input.len());
    let packed_k = k.div_ceil(4);
    assert!(
        weights_i2.len() >= n * packed_k,
        "weights too small: need {}, got {}",
        n * packed_k,
        weights_i2.len()
    );
    assert!(scales.len() >= n, "scales too small: need {n}, got {}", scales.len());
    if let Some(b) = bias {
        assert!(b.len() >= n, "bias too small: need {n}, got {}", b.len());
    }

    let mut output = vec![0.0f32; m * n];

    for row in 0..m {
        let x = &input[row * k..(row * k + k)];
        for col in 0..n {
            let w_start = col * packed_k;
            let w_bytes = &weights_i2[w_start..w_start + packed_k];
            let dot = neon_i2s_dot(w_bytes, x, k);
            let val = dot * scales[col] + bias.map_or(0.0, |b| b[col]);
            output[row * n + col] = val;
        }
    }

    output
}

/// Inner dot product between I2_S packed weights and f32 activations
/// using NEON FMA.
#[inline]
fn neon_i2s_dot(packed: &[u8], activations: &[f32], k: usize) -> f32 {
    // Dequantize into a temporary buffer, then NEON dot-product.
    let full_bytes = k / 4;
    let remainder = k % 4;

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: target_feature gating is handled by parent module cfg.
        // NEON is always available on aarch64.
        unsafe { neon_i2s_dot_inner(packed, activations, full_bytes, remainder) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_i2s_dot(packed, activations, full_bytes, remainder)
    }
}

/// NEON-accelerated I2_S dot product kernel.
///
/// # Safety
///
/// Requires aarch64 NEON support (always available on aarch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_i2s_dot_inner(
    packed: &[u8],
    activations: &[f32],
    full_bytes: usize,
    remainder: usize,
) -> f32 {
    let lut: [f32; 4] = [0.0, 1.0, 0.0, -1.0];
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..full_bytes {
        let byte = packed[i];
        let c0 = (byte & 0x03) as usize;
        let c1 = ((byte >> 2) & 0x03) as usize;
        let c2 = ((byte >> 4) & 0x03) as usize;
        let c3 = ((byte >> 6) & 0x03) as usize;

        let w = [lut[c0], lut[c1], lut[c2], lut[c3]];
        let vw = vld1q_f32(w.as_ptr());
        let va = vld1q_f32(activations.as_ptr().add(i * LANES));
        acc = vfmaq_f32(acc, vw, va);
    }

    let mut sum = vaddvq_f32(acc);

    // Scalar tail for remaining elements
    if remainder > 0 && full_bytes < packed.len() {
        let byte = packed[full_bytes];
        for j in 0..remainder {
            let bits = (byte >> (j * 2)) & 0x03;
            sum += decode_i2s(bits) * activations[full_bytes * LANES + j];
        }
    }

    sum
}

/// Scalar fallback for I2_S dot product (non-aarch64 builds).
#[cfg(not(target_arch = "aarch64"))]
fn scalar_i2s_dot(packed: &[u8], activations: &[f32], full_bytes: usize, remainder: usize) -> f32 {
    let mut sum = 0.0f32;

    for i in 0..full_bytes {
        let byte = packed[i];
        for j in 0..4 {
            let bits = (byte >> (j * 2)) & 0x03;
            sum += decode_i2s(bits) * activations[i * 4 + j];
        }
    }

    if remainder > 0 && full_bytes < packed.len() {
        let byte = packed[full_bytes];
        for j in 0..remainder {
            let bits = (byte >> (j * 2)) & 0x03;
            sum += decode_i2s(bits) * activations[full_bytes * 4 + j];
        }
    }

    sum
}

// ── SiLU / swish activation ────────────────────────────────────────────

/// Apply SiLU (Sigmoid Linear Unit) activation in-place using NEON.
///
/// SiLU(x) = x · σ(x) = x / (1 + exp(−x))
///
/// Processes 4 elements at a time via NEON, with scalar fallback for the
/// tail.
pub fn neon_silu_activation(input: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_silu_inner(input) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_silu(input);
    }
}

/// NEON-accelerated SiLU kernel.
///
/// # Safety
///
/// Requires aarch64 NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_silu_inner(input: &mut [f32]) {
    let n = input.len();
    let chunks = n / LANES;
    let ptr = input.as_mut_ptr();

    let one = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let offset = i * LANES;
        let vx = vld1q_f32(ptr.add(offset));
        let neg_x = vnegq_f32(vx);
        // σ(x) ≈ 1 / (1 + exp(−x)) — use scalar per-lane for exp
        let lane_vals = [
            vgetq_lane_f32::<0>(vx),
            vgetq_lane_f32::<1>(vx),
            vgetq_lane_f32::<2>(vx),
            vgetq_lane_f32::<3>(vx),
        ];
        let neg_vals = [
            vgetq_lane_f32::<0>(neg_x),
            vgetq_lane_f32::<1>(neg_x),
            vgetq_lane_f32::<2>(neg_x),
            vgetq_lane_f32::<3>(neg_x),
        ];
        let sigmoid = [
            1.0 / (1.0 + neg_vals[0].exp()),
            1.0 / (1.0 + neg_vals[1].exp()),
            1.0 / (1.0 + neg_vals[2].exp()),
            1.0 / (1.0 + neg_vals[3].exp()),
        ];
        let vs = vld1q_f32(sigmoid.as_ptr());
        let result = vmulq_f32(vx, vs);
        vst1q_f32(ptr.add(offset), result);
    }

    // Scalar tail
    for i in (chunks * LANES)..n {
        let x = input[i];
        input[i] = x / (1.0 + (-x).exp());
    }
}

/// Scalar SiLU fallback for non-aarch64 builds.
#[cfg(not(target_arch = "aarch64"))]
fn scalar_silu(input: &mut [f32]) {
    for x in input.iter_mut() {
        *x = *x / (1.0 + (-*x).exp());
    }
}

// ── Gated FFN block ────────────────────────────────────────────────────

/// Full gated feed-forward network block with ternary-quantized weights.
///
/// Computes: `down(silu(gate(x)) * up(x))` where each projection is a
/// quantized linear layer with I2_S packed weights and per-row scales.
///
/// This implements the SwiGLU-style FFN used in LLaMA / BitNet:
///
/// 1. `gate_out = gate_weights · x` (quantized linear, `[intermediate_dim]`)
/// 2. `up_out = up_weights · x` (quantized linear, `[intermediate_dim]`)
/// 3. `hidden = SiLU(gate_out) ⊙ up_out` (element-wise)
/// 4. `output = down_weights · hidden` (quantized linear, `[hidden_dim]`)
///
/// # Arguments
///
/// - `input`: flat f32 slice of length `hidden_dim`
/// - `gate_weights`, `up_weights`: I2_S packed, shape
///   `[intermediate_dim, ceil(hidden_dim/4)]` bytes each
/// - `down_weights`: I2_S packed, shape
///   `[hidden_dim, ceil(intermediate_dim/4)]` bytes
/// - `gate_scales`, `up_scales`: length `intermediate_dim`
/// - `down_scales`: length `hidden_dim`
/// - `hidden_dim`: model hidden dimension
/// - `intermediate_dim`: FFN intermediate dimension
///
/// # Returns
///
/// `Vec<f32>` of length `hidden_dim`.
pub fn neon_gated_ffn(
    input: &[f32],
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    gate_scales: &[f32],
    up_scales: &[f32],
    down_scales: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
    // 1. Gate projection: [1, hidden_dim] × [intermediate_dim, hidden_dim]ᵀ
    let mut gate_out = neon_quantized_linear(
        input,
        gate_weights,
        gate_scales,
        None,
        1,
        intermediate_dim,
        hidden_dim,
    );

    // 2. Up projection: [1, hidden_dim] × [intermediate_dim, hidden_dim]ᵀ
    let up_out =
        neon_quantized_linear(input, up_weights, up_scales, None, 1, intermediate_dim, hidden_dim);

    // 3. SiLU(gate) ⊙ up (element-wise)
    neon_silu_activation(&mut gate_out);
    elementwise_mul_inplace(&mut gate_out, &up_out);

    // 4. Down projection: [1, intermediate_dim] × [hidden_dim, intermediate_dim]ᵀ
    neon_quantized_linear(
        &gate_out,
        down_weights,
        down_scales,
        None,
        1,
        hidden_dim,
        intermediate_dim,
    )
}

/// Element-wise multiply `a *= b` with NEON acceleration.
fn elementwise_mul_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_elementwise_mul(a, b) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for (x, y) in a.iter_mut().zip(b.iter()) {
            *x *= *y;
        }
    }
}

/// NEON element-wise multiply kernel.
///
/// # Safety
///
/// Requires aarch64 NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_elementwise_mul(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    let chunks = n / LANES;

    for i in 0..chunks {
        let offset = i * LANES;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        let vr = vmulq_f32(va, vb);
        vst1q_f32(a.as_mut_ptr().add(offset), vr);
    }

    for i in (chunks * LANES)..n {
        a[i] *= b[i];
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    /// Helper: encode a slice of ternary values {-1, 0, +1} into I2_S
    /// packed bytes (4 values per byte, LSB-first).
    fn encode_i2s(values: &[i8]) -> Vec<u8> {
        let mut packed = Vec::with_capacity(values.len().div_ceil(4));
        for chunk in values.chunks(4) {
            let mut byte = 0u8;
            for (j, &v) in chunk.iter().enumerate() {
                let bits: u8 = match v {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                byte |= bits << (j * 2);
            }
            packed.push(byte);
        }
        packed
    }

    #[test]
    fn test_quantized_linear_identity() {
        // 4×4 "identity-like" ternary weights: diagonal = +1, rest = 0
        let w_row0: Vec<i8> = vec![1, 0, 0, 0];
        let w_row1: Vec<i8> = vec![0, 1, 0, 0];
        let w_row2: Vec<i8> = vec![0, 0, 1, 0];
        let w_row3: Vec<i8> = vec![0, 0, 0, 1];

        let mut weights = Vec::new();
        for row in [&w_row0, &w_row1, &w_row2, &w_row3] {
            weights.extend(encode_i2s(row));
        }

        let scales = vec![1.0; 4];
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let out = neon_quantized_linear(&input, &weights, &scales, None, 1, 4, 4);

        assert_eq!(out.len(), 4);
        for i in 0..4 {
            assert!(
                (out[i] - input[i]).abs() < 1e-6,
                "mismatch at {i}: expected {}, got {}",
                input[i],
                out[i]
            );
        }
    }

    #[test]
    fn test_silu_activation() {
        let mut values = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let expected: Vec<f32> = values.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

        neon_silu_activation(&mut values);

        for (i, (&got, &exp)) in values.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "SiLU mismatch at {i}: expected {exp}, got {got}");
        }
    }

    #[test]
    fn test_gated_ffn_smoke() {
        let hidden_dim = 4;
        let intermediate_dim = 8;

        // All-ones gate and up weights → known intermediate values
        let ones_row: Vec<i8> = vec![1; hidden_dim];
        let gate_weights: Vec<u8> =
            (0..intermediate_dim).flat_map(|_| encode_i2s(&ones_row)).collect();
        let up_weights = gate_weights.clone();

        let down_row: Vec<i8> = vec![1; intermediate_dim];
        let down_weights: Vec<u8> = (0..hidden_dim).flat_map(|_| encode_i2s(&down_row)).collect();

        let gate_scales = vec![1.0; intermediate_dim];
        let up_scales = vec![1.0; intermediate_dim];
        let down_scales = vec![1.0; hidden_dim];

        let input = vec![1.0; hidden_dim];
        let out = neon_gated_ffn(
            &input,
            &gate_weights,
            &up_weights,
            &down_weights,
            &gate_scales,
            &up_scales,
            &down_scales,
            hidden_dim,
            intermediate_dim,
        );

        assert_eq!(out.len(), hidden_dim);
        // gate(x) = sum of input = 4.0 for each intermediate neuron
        // SiLU(4.0) = 4.0 / (1 + exp(-4)) ≈ 3.928
        // up(x) = 4.0
        // hidden = SiLU(4.0) * 4.0 ≈ 15.713
        // down(hidden) = sum of 8 intermediates ≈ 125.70
        let silu_4 = 4.0_f32 / (1.0 + (-4.0_f32).exp());
        let expected_intermediate = silu_4 * 4.0;
        let expected_out = expected_intermediate * intermediate_dim as f32;
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - expected_out).abs() < 0.1,
                "FFN mismatch at {i}: expected ~{expected_out}, got {v}"
            );
        }
    }

    #[test]
    fn test_quantized_linear_zeros() {
        // All-zero weights → output should be zero (plus bias if any)
        let k = 8;
        let n = 4;
        let packed_k = k.div_ceil(4);
        let weights = vec![0u8; n * packed_k]; // all 0b00 → weight = 0
        let scales = vec![1.0; n];
        let input = vec![42.0; k];

        // Without bias
        let out = neon_quantized_linear(&input, &weights, &scales, None, 1, n, k);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.abs() < 1e-6, "expected 0 at {i}, got {v}");
        }

        // With bias
        let bias = vec![1.0, 2.0, 3.0, 4.0];
        let out = neon_quantized_linear(&input, &weights, &scales, Some(&bias), 1, n, k);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - bias[i]).abs() < 1e-6, "expected {} at {i}, got {v}", bias[i]);
        }
    }
}
