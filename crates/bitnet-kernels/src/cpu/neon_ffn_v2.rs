//! NEON-optimized feed-forward network (FFN/MLP) v2 kernels for Apple Silicon.
//!
//! Provides six operations commonly found in transformer FFN layers:
//!
//! 1. **Standard FFN** — two-layer projection with GELU activation
//! 2. **SwiGLU FFN** — gated FFN used in LLaMA-style architectures
//! 3. **GELU activation** — Gaussian Error Linear Unit (fast approximation)
//! 4. **SiLU activation** — Sigmoid Linear Unit (swish)
//! 5. **Fused residual FFN** — FFN with residual add in a single pass
//! 6. **Quantized FFN** — FFN with 2-bit (I2_S) packed weights
//!
//! Each operation has an `unsafe fn neon_*` implementation using NEON intrinsics,
//! a `fn scalar_*` fallback, and a public dispatcher that selects at runtime.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Constants ───────────────────────────────────────────────────────

const SQRT_2_OVER_PI: f32 = 0.797_884_6;
const GELU_COEFF: f32 = 0.044_715;

// ── Scalar helpers ──────────────────────────────────────────────────

#[inline(always)]
fn scalar_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn scalar_gelu(x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    0.5 * x * (1.0 + inner.tanh())
}

#[inline(always)]
fn scalar_silu(x: f32) -> f32 {
    x * scalar_sigmoid(x)
}

/// Decode a single 2-bit I2_S code: 0→0, 1→+1, 3→−1.
#[inline(always)]
fn decode_i2s(bits: u8) -> f32 {
    match bits & 0x03 {
        0b01 => 1.0,
        0b11 => -1.0,
        _ => 0.0,
    }
}

// ── Scalar matmul helper ────────────────────────────────────────────

/// Scalar row × column dot product: `sum(a[j] * b[row * cols + j])`.
#[inline]
fn scalar_matvec_row(input: &[f32], weight_row: &[f32]) -> f32 {
    debug_assert_eq!(input.len(), weight_row.len());
    input
        .iter()
        .zip(weight_row.iter())
        .map(|(&a, &b)| a * b)
        .sum()
}

// ── 1. Standard FFN forward ─────────────────────────────────────────

/// NEON-accelerated standard FFN: `output = activation(input × W1) × W2`.
///
/// Weight layouts: W1 `[intermediate × hidden]`, W2 `[hidden × intermediate]`.
///
/// # Safety
///
/// Caller must ensure NEON is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_ffn_forward_f32(
    input: &[f32],
    w1: &[f32],
    w2: &[f32],
    output: &mut [f32],
    hidden: usize,
    intermediate: usize,
) {
    assert_eq!(input.len(), hidden);
    assert_eq!(w1.len(), intermediate * hidden);
    assert_eq!(w2.len(), hidden * intermediate);
    assert_eq!(output.len(), hidden);

    // First projection: hidden → intermediate with GELU activation
    let mut mid = vec![0.0f32; intermediate];
    for r in 0..intermediate {
        let row_start = r * hidden;
        let mut acc = vdupq_n_f32(0.0);
        let chunks = hidden / 4;
        let rem = hidden % 4;

        for c in 0..chunks {
            let off = c * 4;
            let a = vld1q_f32(input.as_ptr().add(off));
            let b = vld1q_f32(w1.as_ptr().add(row_start + off));
            acc = vfmaq_f32(acc, a, b);
        }
        let mut sum = vaddvq_f32(acc);
        let tail = chunks * 4;
        for j in 0..rem {
            sum += input[tail + j] * w1[row_start + tail + j];
        }
        mid[r] = scalar_gelu(sum);
    }

    // Second projection: intermediate → hidden
    for r in 0..hidden {
        let row_start = r * intermediate;
        let mut acc = vdupq_n_f32(0.0);
        let chunks = intermediate / 4;
        let rem = intermediate % 4;

        for c in 0..chunks {
            let off = c * 4;
            let a = vld1q_f32(mid.as_ptr().add(off));
            let b = vld1q_f32(w2.as_ptr().add(row_start + off));
            acc = vfmaq_f32(acc, a, b);
        }
        let mut sum = vaddvq_f32(acc);
        let tail = chunks * 4;
        for j in 0..rem {
            sum += mid[tail + j] * w2[row_start + tail + j];
        }
        output[r] = sum;
    }
}

/// Scalar fallback for standard FFN.
pub fn scalar_ffn_forward_f32(
    input: &[f32],
    w1: &[f32],
    w2: &[f32],
    output: &mut [f32],
    hidden: usize,
    intermediate: usize,
) {
    assert_eq!(input.len(), hidden);
    assert_eq!(w1.len(), intermediate * hidden);
    assert_eq!(w2.len(), hidden * intermediate);
    assert_eq!(output.len(), hidden);

    let mut mid = vec![0.0f32; intermediate];
    for r in 0..intermediate {
        let row = &w1[r * hidden..(r + 1) * hidden];
        mid[r] = scalar_gelu(scalar_matvec_row(input, row));
    }
    for r in 0..hidden {
        let row = &w2[r * intermediate..(r + 1) * intermediate];
        output[r] = scalar_matvec_row(&mid, row);
    }
}

/// Public dispatcher for standard FFN forward.
pub fn ffn_forward_f32(
    input: &[f32],
    w1: &[f32],
    w2: &[f32],
    output: &mut [f32],
    hidden: usize,
    intermediate: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_ffn_forward_f32(input, w1, w2, output, hidden, intermediate);
            }
            return;
        }
    }
    scalar_ffn_forward_f32(input, w1, w2, output, hidden, intermediate);
}

// ── 2. SwiGLU FFN ──────────────────────────────────────────────────

/// NEON-accelerated SwiGLU FFN (LLaMA style):
/// `output = (SiLU(input × Wg) ⊙ (input × Wup)) × Wdown`.
///
/// # Safety
///
/// Caller must ensure NEON is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_swiglu_ffn_f32(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &mut [f32],
    hidden: usize,
    intermediate: usize,
) {
    assert_eq!(input.len(), hidden);
    assert_eq!(w_gate.len(), intermediate * hidden);
    assert_eq!(w_up.len(), intermediate * hidden);
    assert_eq!(w_down.len(), hidden * intermediate);
    assert_eq!(output.len(), hidden);

    let mut gate_proj = vec![0.0f32; intermediate];
    let mut up_proj = vec![0.0f32; intermediate];

    // Compute gate and up projections
    for r in 0..intermediate {
        let row_start = r * hidden;
        let mut acc_g = vdupq_n_f32(0.0);
        let mut acc_u = vdupq_n_f32(0.0);
        let chunks = hidden / 4;
        let rem = hidden % 4;

        for c in 0..chunks {
            let off = c * 4;
            let inp = vld1q_f32(input.as_ptr().add(off));
            let g = vld1q_f32(w_gate.as_ptr().add(row_start + off));
            let u = vld1q_f32(w_up.as_ptr().add(row_start + off));
            acc_g = vfmaq_f32(acc_g, inp, g);
            acc_u = vfmaq_f32(acc_u, inp, u);
        }
        let mut sum_g = vaddvq_f32(acc_g);
        let mut sum_u = vaddvq_f32(acc_u);
        let tail = chunks * 4;
        for j in 0..rem {
            sum_g += input[tail + j] * w_gate[row_start + tail + j];
            sum_u += input[tail + j] * w_up[row_start + tail + j];
        }
        gate_proj[r] = scalar_silu(sum_g);
        up_proj[r] = sum_u;
    }

    // Element-wise gate ⊙ up using NEON
    let mut mid = vec![0.0f32; intermediate];
    let chunks = intermediate / 4;
    let rem = intermediate % 4;
    for c in 0..chunks {
        let off = c * 4;
        let g = vld1q_f32(gate_proj.as_ptr().add(off));
        let u = vld1q_f32(up_proj.as_ptr().add(off));
        vst1q_f32(mid.as_mut_ptr().add(off), vmulq_f32(g, u));
    }
    let tail = chunks * 4;
    for j in 0..rem {
        mid[tail + j] = gate_proj[tail + j] * up_proj[tail + j];
    }

    // Down projection: intermediate → hidden
    for r in 0..hidden {
        let row_start = r * intermediate;
        let mut acc = vdupq_n_f32(0.0);
        let chunks = intermediate / 4;
        let rem = intermediate % 4;

        for c in 0..chunks {
            let off = c * 4;
            let a = vld1q_f32(mid.as_ptr().add(off));
            let b = vld1q_f32(w_down.as_ptr().add(row_start + off));
            acc = vfmaq_f32(acc, a, b);
        }
        let mut sum = vaddvq_f32(acc);
        let tail = chunks * 4;
        for j in 0..rem {
            sum += mid[tail + j] * w_down[row_start + tail + j];
        }
        output[r] = sum;
    }
}

/// Scalar fallback for SwiGLU FFN.
pub fn scalar_swiglu_ffn_f32(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &mut [f32],
    hidden: usize,
    intermediate: usize,
) {
    assert_eq!(input.len(), hidden);
    assert_eq!(w_gate.len(), intermediate * hidden);
    assert_eq!(w_up.len(), intermediate * hidden);
    assert_eq!(w_down.len(), hidden * intermediate);
    assert_eq!(output.len(), hidden);

    let mut mid = vec![0.0f32; intermediate];
    for r in 0..intermediate {
        let g_row = &w_gate[r * hidden..(r + 1) * hidden];
        let u_row = &w_up[r * hidden..(r + 1) * hidden];
        let gate_val = scalar_silu(scalar_matvec_row(input, g_row));
        let up_val = scalar_matvec_row(input, u_row);
        mid[r] = gate_val * up_val;
    }
    for r in 0..hidden {
        let row = &w_down[r * intermediate..(r + 1) * intermediate];
        output[r] = scalar_matvec_row(&mid, row);
    }
}

/// Public dispatcher for SwiGLU FFN.
pub fn swiglu_ffn_f32(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &mut [f32],
    hidden: usize,
    intermediate: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_swiglu_ffn_f32(
                    input, w_gate, w_up, w_down, output, hidden, intermediate,
                );
            }
            return;
        }
    }
    scalar_swiglu_ffn_f32(input, w_gate, w_up, w_down, output, hidden, intermediate);
}

// ── 3. GELU activation ─────────────────────────────────────────────

/// NEON-accelerated GELU: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
///
/// # Safety
///
/// Caller must ensure NEON is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gelu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / 4;
    let rem = n % 4;

    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let coeff = vdupq_n_f32(GELU_COEFF);
    let sqrt2pi = vdupq_n_f32(SQRT_2_OVER_PI);

    for i in 0..chunks {
        let off = i * 4;
        let x = vld1q_f32(input.as_ptr().add(off));
        let x3 = vmulq_f32(vmulq_f32(x, x), x);
        let inner = vmulq_f32(sqrt2pi, vfmaq_f32(x, coeff, x3));
        // tanh via scalar for each lane
        let mut inner_arr = [0.0f32; 4];
        vst1q_f32(inner_arr.as_mut_ptr(), inner);
        let tanh_arr = [
            inner_arr[0].tanh(),
            inner_arr[1].tanh(),
            inner_arr[2].tanh(),
            inner_arr[3].tanh(),
        ];
        let tanh_v = vld1q_f32(tanh_arr.as_ptr());
        let result = vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, tanh_v)));
        vst1q_f32(output.as_mut_ptr().add(off), result);
    }

    let tail = chunks * 4;
    for j in 0..rem {
        output[tail + j] = scalar_gelu(input[tail + j]);
    }
}

/// Scalar fallback for GELU.
pub fn scalar_gelu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = scalar_gelu(*x);
    }
}

/// Public dispatcher for GELU activation.
pub fn gelu_f32(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_gelu_f32(input, output);
            }
            return;
        }
    }
    scalar_gelu_f32(input, output);
}

// ── 4. SiLU activation ─────────────────────────────────────────────

/// NEON-accelerated SiLU: `x * sigmoid(x)`.
///
/// Computes sigmoid in scalar, then uses NEON for the final multiply.
///
/// # Safety
///
/// Caller must ensure NEON is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_silu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();

    // Compute sigmoid into output buffer (scalar)
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = scalar_sigmoid(*x);
    }

    // Multiply x * sigmoid(x) using NEON
    let chunks = n / 4;
    let rem = n % 4;

    for i in 0..chunks {
        let off = i * 4;
        let x_vec = vld1q_f32(input.as_ptr().add(off));
        let sig_vec = vld1q_f32(output.as_ptr().add(off));
        vst1q_f32(output.as_mut_ptr().add(off), vmulq_f32(x_vec, sig_vec));
    }

    let tail = chunks * 4;
    for j in 0..rem {
        let idx = tail + j;
        output[idx] *= input[idx];
    }
}

/// Scalar fallback for SiLU.
pub fn scalar_silu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = scalar_silu(*x);
    }
}

/// Public dispatcher for SiLU activation.
pub fn silu_f32(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_silu_f32(input, output);
            }
            return;
        }
    }
    scalar_silu_f32(input, output);
}

// ── 5. Fused residual FFN ──────────────────────────────────────────

/// NEON-accelerated fused residual FFN:
/// `output = residual + activation(input × W1) × W2`.
///
/// # Safety
///
/// Caller must ensure NEON is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_residual_ffn_f32(
    input: &[f32],
    residual: &[f32],
    w1: &[f32],
    w2: &[f32],
    output: &mut [f32],
    hidden: usize,
    intermediate: usize,
) {
    assert_eq!(input.len(), hidden);
    assert_eq!(residual.len(), hidden);
    assert_eq!(w1.len(), intermediate * hidden);
    assert_eq!(w2.len(), hidden * intermediate);
    assert_eq!(output.len(), hidden);

    // FFN into output
    neon_ffn_forward_f32(input, w1, w2, output, hidden, intermediate);

    // Fused residual add using NEON
    let chunks = hidden / 4;
    let rem = hidden % 4;
    for c in 0..chunks {
        let off = c * 4;
        let ffn_v = vld1q_f32(output.as_ptr().add(off));
        let res_v = vld1q_f32(residual.as_ptr().add(off));
        vst1q_f32(output.as_mut_ptr().add(off), vaddq_f32(ffn_v, res_v));
    }
    let tail = chunks * 4;
    for j in 0..rem {
        output[tail + j] += residual[tail + j];
    }
}

/// Scalar fallback for fused residual FFN.
pub fn scalar_fused_residual_ffn_f32(
    input: &[f32],
    residual: &[f32],
    w1: &[f32],
    w2: &[f32],
    output: &mut [f32],
    hidden: usize,
    intermediate: usize,
) {
    scalar_ffn_forward_f32(input, w1, w2, output, hidden, intermediate);
    for (o, r) in output.iter_mut().zip(residual.iter()) {
        *o += r;
    }
}

/// Public dispatcher for fused residual FFN.
pub fn fused_residual_ffn_f32(
    input: &[f32],
    residual: &[f32],
    w1: &[f32],
    w2: &[f32],
    output: &mut [f32],
    hidden: usize,
    intermediate: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_fused_residual_ffn_f32(
                    input, residual, w1, w2, output, hidden, intermediate,
                );
            }
            return;
        }
    }
    scalar_fused_residual_ffn_f32(
        input, residual, w1, w2, output, hidden, intermediate,
    );
}

// ── 6. Quantized FFN (I2_S 2-bit weights) ──────────────────────────

/// Dequantize a packed I2_S byte to 4 f32 values.
#[inline(always)]
fn unpack_i2s_byte(byte: u8) -> [f32; 4] {
    [
        decode_i2s(byte),
        decode_i2s(byte >> 2),
        decode_i2s(byte >> 4),
        decode_i2s(byte >> 6),
    ]
}

/// Scalar dot product of input with a packed I2_S weight row, scaled.
fn scalar_i2s_dot(input: &[f32], packed: &[u8], scale: f32) -> f32 {
    let n = input.len();
    let full_bytes = n / 4;
    let rem = n % 4;
    let mut sum = 0.0f32;

    for b in 0..full_bytes {
        let vals = unpack_i2s_byte(packed[b]);
        let off = b * 4;
        sum += input[off] * vals[0]
            + input[off + 1] * vals[1]
            + input[off + 2] * vals[2]
            + input[off + 3] * vals[3];
    }
    if rem > 0 {
        let last_byte = packed[full_bytes];
        let vals = unpack_i2s_byte(last_byte);
        let off = full_bytes * 4;
        for j in 0..rem {
            sum += input[off + j] * vals[j];
        }
    }
    sum * scale
}

/// NEON-accelerated FFN with 2-bit quantized weights.
///
/// Weight packing: 4 values per byte, LSB-first. Each row of W1 has
/// `ceil(hidden/4)` bytes; each row of W2 has `ceil(intermediate/4)` bytes.
/// Scales: one per row.
///
/// # Safety
///
/// Caller must ensure NEON is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_quantized_ffn_i2_f32(
    input: &[f32],
    w1_packed: &[u8],
    w1_scales: &[f32],
    w2_packed: &[u8],
    w2_scales: &[f32],
    output: &mut [f32],
    hidden: usize,
    intermediate: usize,
) {
    let w1_row_bytes = (hidden + 3) / 4;
    let w2_row_bytes = (intermediate + 3) / 4;
    assert_eq!(input.len(), hidden);
    assert_eq!(w1_packed.len(), intermediate * w1_row_bytes);
    assert_eq!(w1_scales.len(), intermediate);
    assert_eq!(w2_packed.len(), hidden * w2_row_bytes);
    assert_eq!(w2_scales.len(), hidden);
    assert_eq!(output.len(), hidden);

    // First projection with GELU
    let mut mid = vec![0.0f32; intermediate];
    for r in 0..intermediate {
        let row_packed = &w1_packed[r * w1_row_bytes..(r + 1) * w1_row_bytes];

        // NEON dot product with on-the-fly dequantization
        let lut: [f32; 4] = [0.0, 1.0, 0.0, -1.0];
        let mut acc = vdupq_n_f32(0.0);
        let full_bytes = hidden / 4;
        let rem = hidden % 4;

        for b in 0..full_bytes {
            let byte = row_packed[b];
            let vals = [
                lut[(byte & 0x03) as usize],
                lut[((byte >> 2) & 0x03) as usize],
                lut[((byte >> 4) & 0x03) as usize],
                lut[((byte >> 6) & 0x03) as usize],
            ];
            let w_vec = vld1q_f32(vals.as_ptr());
            let i_vec = vld1q_f32(input.as_ptr().add(b * 4));
            acc = vfmaq_f32(acc, i_vec, w_vec);
        }
        let mut sum = vaddvq_f32(acc);
        if rem > 0 {
            let byte = row_packed[full_bytes];
            let vals = unpack_i2s_byte(byte);
            let off = full_bytes * 4;
            for j in 0..rem {
                sum += input[off + j] * vals[j];
            }
        }
        mid[r] = scalar_gelu(sum * w1_scales[r]);
    }

    // Second projection
    for r in 0..hidden {
        let row_packed = &w2_packed[r * w2_row_bytes..(r + 1) * w2_row_bytes];

        let lut: [f32; 4] = [0.0, 1.0, 0.0, -1.0];
        let mut acc = vdupq_n_f32(0.0);
        let full_bytes = intermediate / 4;
        let rem = intermediate % 4;

        for b in 0..full_bytes {
            let byte = row_packed[b];
            let vals = [
                lut[(byte & 0x03) as usize],
                lut[((byte >> 2) & 0x03) as usize],
                lut[((byte >> 4) & 0x03) as usize],
                lut[((byte >> 6) & 0x03) as usize],
            ];
            let w_vec = vld1q_f32(vals.as_ptr());
            let m_vec = vld1q_f32(mid.as_ptr().add(b * 4));
            acc = vfmaq_f32(acc, m_vec, w_vec);
        }
        let mut sum = vaddvq_f32(acc);
        if rem > 0 {
            let byte = row_packed[full_bytes];
            let vals = unpack_i2s_byte(byte);
            let off = full_bytes * 4;
            for j in 0..rem {
                sum += mid[off + j] * vals[j];
            }
        }
        output[r] = sum * w2_scales[r];
    }
}

/// Scalar fallback for quantized FFN.
pub fn scalar_quantized_ffn_i2_f32(
    input: &[f32],
    w1_packed: &[u8],
    w1_scales: &[f32],
    w2_packed: &[u8],
    w2_scales: &[f32],
    output: &mut [f32],
    hidden: usize,
    intermediate: usize,
) {
    let w1_row_bytes = (hidden + 3) / 4;
    let w2_row_bytes = (intermediate + 3) / 4;
    assert_eq!(input.len(), hidden);
    assert_eq!(w1_packed.len(), intermediate * w1_row_bytes);
    assert_eq!(w1_scales.len(), intermediate);
    assert_eq!(w2_packed.len(), hidden * w2_row_bytes);
    assert_eq!(w2_scales.len(), hidden);
    assert_eq!(output.len(), hidden);

    let mut mid = vec![0.0f32; intermediate];
    for r in 0..intermediate {
        let row = &w1_packed[r * w1_row_bytes..(r + 1) * w1_row_bytes];
        mid[r] = scalar_gelu(scalar_i2s_dot(input, row, w1_scales[r]));
    }
    for r in 0..hidden {
        let row = &w2_packed[r * w2_row_bytes..(r + 1) * w2_row_bytes];
        output[r] = scalar_i2s_dot(&mid, row, w2_scales[r]);
    }
}

/// Public dispatcher for quantized FFN.
pub fn quantized_ffn_i2_f32(
    input: &[f32],
    w1_packed: &[u8],
    w1_scales: &[f32],
    w2_packed: &[u8],
    w2_scales: &[f32],
    output: &mut [f32],
    hidden: usize,
    intermediate: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_quantized_ffn_i2_f32(
                    input, w1_packed, w1_scales, w2_packed, w2_scales, output,
                    hidden, intermediate,
                );
            }
            return;
        }
    }
    scalar_quantized_ffn_i2_f32(
        input, w1_packed, w1_scales, w2_packed, w2_scales, output, hidden,
        intermediate,
    );
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // f64 reference implementations for precision comparison
    fn ref_gelu_f64(x: f64) -> f64 {
        let x3 = x * x * x;
        let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x3);
        0.5 * x * (1.0 + inner.tanh())
    }

    fn ref_silu_f64(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    fn ref_sigmoid_f64(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    // Helper: pack I2_S values into bytes
    fn pack_i2s(values: &[i8]) -> Vec<u8> {
        let mut packed = Vec::new();
        for chunk in values.chunks(4) {
            let mut byte = 0u8;
            for (j, &v) in chunk.iter().enumerate() {
                let bits: u8 = match v {
                    0 => 0b00,
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

    // Helper: identity-like weight matrix (scaled for test stability)
    fn identity_weights(rows: usize, cols: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; rows * cols];
        let n = rows.min(cols);
        for i in 0..n {
            w[i * cols + i] = 1.0;
        }
        w
    }

    const TOL: f32 = 1e-4;

    // ── GELU tests ──────────────────────────────────────────────────

    #[test]
    fn test_gelu_basic() {
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5];
        let mut output = vec![0.0; input.len()];
        gelu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            let expected = ref_gelu_f64(x as f64) as f32;
            assert!((o - expected).abs() < TOL, "gelu({x}): got {o}, expected {expected}");
        }
    }

    #[test]
    fn test_gelu_vs_f64_reference() {
        let values: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0.0; values.len()];
        gelu_f32(&values, &mut output);
        for (&x, &o) in values.iter().zip(output.iter()) {
            let expected = ref_gelu_f64(x as f64) as f32;
            assert!(
                (o - expected).abs() < TOL,
                "gelu({x}): got {o}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_gelu_empty() {
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        gelu_f32(&input, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn test_gelu_single_element() {
        let input = vec![1.5];
        let mut output = vec![0.0];
        gelu_f32(&input, &mut output);
        let expected = ref_gelu_f64(1.5) as f32;
        assert!((output[0] - expected).abs() < TOL);
    }

    #[test]
    fn test_gelu_all_zeros() {
        let input = vec![0.0; 17];
        let mut output = vec![1.0; 17];
        gelu_f32(&input, &mut output);
        for &o in &output {
            assert_eq!(o, 0.0);
        }
    }

    #[test]
    fn test_gelu_monotonic() {
        let values: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0; values.len()];
        gelu_f32(&values, &mut output);
        for i in 1..output.len() {
            assert!(
                output[i] >= output[i - 1],
                "GELU not monotonic at index {i}: {} < {}",
                output[i],
                output[i - 1]
            );
        }
    }

    #[test]
    fn test_gelu_positive_for_positive_input() {
        let input: Vec<f32> = (1..=50).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0; input.len()];
        gelu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(o > 0.0, "gelu({x}) should be positive, got {o}");
        }
    }

    #[test]
    fn test_gelu_negative_bounded() {
        // GELU of very negative inputs should be close to 0
        let input = vec![-10.0, -5.0, -3.0];
        let mut output = vec![0.0; 3];
        gelu_f32(&input, &mut output);
        for &o in &output {
            assert!(o.abs() < 0.1, "gelu of large negative should be near 0, got {o}");
        }
    }

    #[test]
    fn test_gelu_odd_length() {
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let mut output = vec![0.0; 7];
        gelu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            let expected = ref_gelu_f64(x as f64) as f32;
            assert!((o - expected).abs() < TOL);
        }
    }

    #[test]
    fn test_gelu_scalar_matches_dispatcher() {
        let input: Vec<f32> = (-5..=5).map(|i| i as f32).collect();
        let mut out_scalar = vec![0.0; input.len()];
        let mut out_dispatch = vec![0.0; input.len()];
        scalar_gelu_f32(&input, &mut out_scalar);
        gelu_f32(&input, &mut out_dispatch);
        for i in 0..input.len() {
            assert!(
                (out_scalar[i] - out_dispatch[i]).abs() < 1e-6,
                "Mismatch at {i}"
            );
        }
    }

    // ── SiLU tests ──────────────────────────────────────────────────

    #[test]
    fn test_silu_basic() {
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5];
        let mut output = vec![0.0; input.len()];
        silu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            let expected = ref_silu_f64(x as f64) as f32;
            assert!((o - expected).abs() < TOL, "silu({x}): got {o}, expected {expected}");
        }
    }

    #[test]
    fn test_silu_vs_f64_reference() {
        let values: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0.0; values.len()];
        silu_f32(&values, &mut output);
        for (&x, &o) in values.iter().zip(output.iter()) {
            let expected = ref_silu_f64(x as f64) as f32;
            assert!(
                (o - expected).abs() < TOL,
                "silu({x}): got {o}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_silu_empty() {
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        silu_f32(&input, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn test_silu_single_element() {
        let input = vec![2.0];
        let mut output = vec![0.0];
        silu_f32(&input, &mut output);
        let expected = ref_silu_f64(2.0) as f32;
        assert!((output[0] - expected).abs() < TOL);
    }

    #[test]
    fn test_silu_all_zeros() {
        let input = vec![0.0; 13];
        let mut output = vec![1.0; 13];
        silu_f32(&input, &mut output);
        for &o in &output {
            assert_eq!(o, 0.0, "silu(0) should be 0");
        }
    }

    #[test]
    fn test_silu_at_zero() {
        // SiLU(0) = 0 * sigmoid(0) = 0
        let input = vec![0.0];
        let mut output = vec![99.0];
        silu_f32(&input, &mut output);
        assert_eq!(output[0], 0.0);
    }

    #[test]
    fn test_silu_positive_for_positive() {
        let input: Vec<f32> = (1..=20).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0.0; input.len()];
        silu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(o > 0.0, "silu({x}) should be positive, got {o}");
        }
    }

    #[test]
    fn test_silu_bounded_below() {
        // SiLU minimum is approximately -0.278 at x ≈ -1.278
        let input: Vec<f32> = (-100..=0).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0; input.len()];
        silu_f32(&input, &mut output);
        for &o in &output {
            assert!(o >= -0.4, "silu should be bounded below, got {o}");
        }
    }

    #[test]
    fn test_silu_odd_length() {
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mut output = vec![0.0; 5];
        silu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            let expected = ref_silu_f64(x as f64) as f32;
            assert!((o - expected).abs() < TOL);
        }
    }

    #[test]
    fn test_silu_scalar_matches_dispatcher() {
        let input: Vec<f32> = (-5..=5).map(|i| i as f32).collect();
        let mut out_scalar = vec![0.0; input.len()];
        let mut out_dispatch = vec![0.0; input.len()];
        scalar_silu_f32(&input, &mut out_scalar);
        silu_f32(&input, &mut out_dispatch);
        for i in 0..input.len() {
            assert!(
                (out_scalar[i] - out_dispatch[i]).abs() < 1e-6,
                "Mismatch at {i}"
            );
        }
    }

    // ── Standard FFN tests ──────────────────────────────────────────

    #[test]
    fn test_ffn_forward_identity_weights() {
        // With identity W1 and W2, output ≈ gelu(input) projected back
        let h = 4;
        let inter = 4;
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let w1 = identity_weights(inter, h);
        let w2 = identity_weights(h, inter);
        let mut output = vec![0.0; h];
        ffn_forward_f32(&input, &w1, &w2, &mut output, h, inter);

        // Each element goes through gelu then identity
        for i in 0..h {
            let expected = ref_gelu_f64(input[i] as f64) as f32;
            assert!(
                (output[i] - expected).abs() < TOL,
                "ffn output[{i}] = {}, expected {}",
                output[i],
                expected
            );
        }
    }

    #[test]
    fn test_ffn_forward_all_zeros_input() {
        let h = 4;
        let inter = 8;
        let input = vec![0.0; h];
        let w1 = vec![1.0; inter * h];
        let w2 = vec![1.0; h * inter];
        let mut output = vec![99.0; h];
        ffn_forward_f32(&input, &w1, &w2, &mut output, h, inter);
        for &o in &output {
            assert!(o.abs() < 1e-6, "FFN of zeros should be ~0, got {o}");
        }
    }

    #[test]
    fn test_ffn_forward_correctness() {
        let h = 2;
        let inter = 3;
        let input = vec![1.0, 2.0];
        // W1: 3x2, W2: 2x3
        let w1 = vec![0.5, 0.3, -0.1, 0.4, 0.2, -0.5];
        let w2 = vec![0.1, -0.2, 0.3, 0.4, 0.1, -0.3];
        let mut output = vec![0.0; h];
        ffn_forward_f32(&input, &w1, &w2, &mut output, h, inter);

        // Compute reference in f64
        let mut mid = vec![0.0f64; inter];
        for r in 0..inter {
            let mut sum = 0.0f64;
            for c in 0..h {
                sum += input[c] as f64 * w1[r * h + c] as f64;
            }
            mid[r] = ref_gelu_f64(sum);
        }
        let mut expected = vec![0.0f64; h];
        for r in 0..h {
            for c in 0..inter {
                expected[r] += mid[c] * w2[r * inter + c] as f64;
            }
        }

        for i in 0..h {
            assert!(
                (output[i] - expected[i] as f32).abs() < TOL,
                "ffn[{i}]: got {}, expected {}",
                output[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_ffn_forward_various_sizes() {
        for &(h, inter) in &[(1, 1), (3, 5), (8, 4), (7, 9), (16, 32)] {
            let input: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let w1 = vec![0.1; inter * h];
            let w2 = vec![0.1; h * inter];
            let mut output = vec![0.0; h];
            ffn_forward_f32(&input, &w1, &w2, &mut output, h, inter);
            // Just verify no panic/NaN
            for &o in &output {
                assert!(o.is_finite(), "FFN output NaN for h={h}, inter={inter}");
            }
        }
    }

    #[test]
    fn test_ffn_scalar_matches_dispatcher() {
        let h = 4;
        let inter = 6;
        let input: Vec<f32> = (0..h).map(|i| i as f32 * 0.3).collect();
        let w1: Vec<f32> = (0..inter * h).map(|i| (i as f32 * 0.01) - 0.1).collect();
        let w2: Vec<f32> = (0..h * inter).map(|i| (i as f32 * 0.01) - 0.05).collect();
        let mut out_scalar = vec![0.0; h];
        let mut out_dispatch = vec![0.0; h];
        scalar_ffn_forward_f32(&input, &w1, &w2, &mut out_scalar, h, inter);
        ffn_forward_f32(&input, &w1, &w2, &mut out_dispatch, h, inter);
        for i in 0..h {
            assert!(
                (out_scalar[i] - out_dispatch[i]).abs() < 1e-5,
                "Mismatch at {i}: scalar={}, dispatch={}",
                out_scalar[i],
                out_dispatch[i]
            );
        }
    }

    // ── SwiGLU FFN tests ────────────────────────────────────────────

    #[test]
    fn test_swiglu_basic() {
        let h = 4;
        let inter = 4;
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let w_gate = identity_weights(inter, h);
        let w_up = identity_weights(inter, h);
        let w_down = identity_weights(h, inter);
        let mut output = vec![0.0; h];
        swiglu_ffn_f32(&input, &w_gate, &w_up, &w_down, &mut output, h, inter);
        // With identity: output[i] = silu(input[i]) * input[i]
        for i in 0..h {
            let expected = ref_silu_f64(input[i] as f64) as f32 * input[i];
            assert!(
                (output[i] - expected).abs() < TOL,
                "swiglu[{i}]: got {}, expected {}",
                output[i],
                expected
            );
        }
    }

    #[test]
    fn test_swiglu_gating_mechanism() {
        // If gate output is 0 (silu(0)=0), output should be ~0
        let h = 2;
        let inter = 2;
        let input = vec![0.0, 0.0];
        let w_gate = vec![1.0; inter * h];
        let w_up = vec![1.0; inter * h];
        let w_down = vec![1.0; h * inter];
        let mut output = vec![99.0; h];
        swiglu_ffn_f32(&input, &w_gate, &w_up, &w_down, &mut output, h, inter);
        for &o in &output {
            assert!(o.abs() < 1e-6, "SwiGLU of zeros should gate to ~0, got {o}");
        }
    }

    #[test]
    fn test_swiglu_correctness() {
        let h = 2;
        let inter = 3;
        let input = vec![1.0, 2.0];
        let w_gate = vec![0.5, 0.3, -0.1, 0.4, 0.2, -0.5];
        let w_up = vec![0.1, -0.2, 0.3, 0.4, 0.1, -0.3];
        let w_down = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.1];
        let mut output = vec![0.0; h];
        swiglu_ffn_f32(&input, &w_gate, &w_up, &w_down, &mut output, h, inter);

        // f64 reference
        let mut mid = vec![0.0f64; inter];
        for r in 0..inter {
            let mut sum_g = 0.0f64;
            let mut sum_u = 0.0f64;
            for c in 0..h {
                sum_g += input[c] as f64 * w_gate[r * h + c] as f64;
                sum_u += input[c] as f64 * w_up[r * h + c] as f64;
            }
            mid[r] = ref_silu_f64(sum_g) * sum_u;
        }
        let mut expected = vec![0.0f64; h];
        for r in 0..h {
            for c in 0..inter {
                expected[r] += mid[c] * w_down[r * inter + c] as f64;
            }
        }

        for i in 0..h {
            assert!(
                (output[i] - expected[i] as f32).abs() < TOL,
                "swiglu[{i}]: got {}, expected {}",
                output[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_swiglu_all_zeros_input() {
        let h = 3;
        let inter = 4;
        let input = vec![0.0; h];
        let w_gate = vec![1.0; inter * h];
        let w_up = vec![1.0; inter * h];
        let w_down = vec![1.0; h * inter];
        let mut output = vec![99.0; h];
        swiglu_ffn_f32(&input, &w_gate, &w_up, &w_down, &mut output, h, inter);
        for &o in &output {
            assert!(o.abs() < 1e-6);
        }
    }

    #[test]
    fn test_swiglu_various_sizes() {
        for &(h, inter) in &[(1, 1), (3, 5), (8, 4), (7, 9)] {
            let input: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let w_g = vec![0.1; inter * h];
            let w_u = vec![0.1; inter * h];
            let w_d = vec![0.1; h * inter];
            let mut output = vec![0.0; h];
            swiglu_ffn_f32(&input, &w_g, &w_u, &w_d, &mut output, h, inter);
            for &o in &output {
                assert!(o.is_finite());
            }
        }
    }

    #[test]
    fn test_swiglu_scalar_matches_dispatcher() {
        let h = 4;
        let inter = 6;
        let input: Vec<f32> = (0..h).map(|i| i as f32 * 0.3).collect();
        let w_g: Vec<f32> = (0..inter * h).map(|i| (i as f32 * 0.01) - 0.1).collect();
        let w_u: Vec<f32> = (0..inter * h).map(|i| (i as f32 * 0.02) - 0.05).collect();
        let w_d: Vec<f32> = (0..h * inter).map(|i| (i as f32 * 0.01) + 0.01).collect();
        let mut out_scalar = vec![0.0; h];
        let mut out_dispatch = vec![0.0; h];
        scalar_swiglu_ffn_f32(&input, &w_g, &w_u, &w_d, &mut out_scalar, h, inter);
        swiglu_ffn_f32(&input, &w_g, &w_u, &w_d, &mut out_dispatch, h, inter);
        for i in 0..h {
            assert!(
                (out_scalar[i] - out_dispatch[i]).abs() < 1e-5,
                "Mismatch at {i}"
            );
        }
    }

    // ── Residual FFN tests ──────────────────────────────────────────

    #[test]
    fn test_residual_ffn_adds_correctly() {
        let h = 4;
        let inter = 4;
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let residual = vec![10.0, 20.0, 30.0, 40.0];
        let w1 = identity_weights(inter, h);
        let w2 = identity_weights(h, inter);
        let mut output = vec![0.0; h];
        fused_residual_ffn_f32(&input, &residual, &w1, &w2, &mut output, h, inter);

        // Reference: FFN(input) + residual
        let mut ffn_only = vec![0.0; h];
        ffn_forward_f32(&input, &w1, &w2, &mut ffn_only, h, inter);
        for i in 0..h {
            let expected = ffn_only[i] + residual[i];
            assert!(
                (output[i] - expected).abs() < TOL,
                "residual_ffn[{i}]: got {}, expected {}",
                output[i],
                expected
            );
        }
    }

    #[test]
    fn test_residual_ffn_zero_input() {
        let h = 3;
        let inter = 4;
        let input = vec![0.0; h];
        let residual = vec![1.0, 2.0, 3.0];
        let w1 = vec![0.1; inter * h];
        let w2 = vec![0.1; h * inter];
        let mut output = vec![0.0; h];
        fused_residual_ffn_f32(&input, &residual, &w1, &w2, &mut output, h, inter);
        // FFN(0) ≈ 0, so output ≈ residual
        for i in 0..h {
            assert!(
                (output[i] - residual[i]).abs() < 0.01,
                "residual[{i}]: got {}, expected ~{}",
                output[i],
                residual[i]
            );
        }
    }

    #[test]
    fn test_residual_ffn_zero_residual() {
        let h = 4;
        let inter = 4;
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let residual = vec![0.0; h];
        let w1 = identity_weights(inter, h);
        let w2 = identity_weights(h, inter);
        let mut output = vec![0.0; h];
        let mut ffn_only = vec![0.0; h];
        fused_residual_ffn_f32(&input, &residual, &w1, &w2, &mut output, h, inter);
        ffn_forward_f32(&input, &w1, &w2, &mut ffn_only, h, inter);
        for i in 0..h {
            assert!(
                (output[i] - ffn_only[i]).abs() < 1e-6,
                "With zero residual, output should equal FFN"
            );
        }
    }

    #[test]
    fn test_residual_ffn_scalar_matches_dispatcher() {
        let h = 4;
        let inter = 6;
        let input: Vec<f32> = (0..h).map(|i| i as f32 * 0.3).collect();
        let residual: Vec<f32> = (0..h).map(|i| i as f32 * 1.5).collect();
        let w1: Vec<f32> = (0..inter * h).map(|i| (i as f32 * 0.01) - 0.1).collect();
        let w2: Vec<f32> = (0..h * inter).map(|i| (i as f32 * 0.01) - 0.05).collect();
        let mut out_scalar = vec![0.0; h];
        let mut out_dispatch = vec![0.0; h];
        scalar_fused_residual_ffn_f32(
            &input, &residual, &w1, &w2, &mut out_scalar, h, inter,
        );
        fused_residual_ffn_f32(
            &input, &residual, &w1, &w2, &mut out_dispatch, h, inter,
        );
        for i in 0..h {
            assert!(
                (out_scalar[i] - out_dispatch[i]).abs() < 1e-5,
                "Mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_residual_ffn_various_sizes() {
        for &(h, inter) in &[(1, 1), (3, 5), (8, 4), (7, 9)] {
            let input: Vec<f32> = (0..h).map(|i| i as f32 * 0.1).collect();
            let residual: Vec<f32> = (0..h).map(|i| i as f32).collect();
            let w1 = vec![0.1; inter * h];
            let w2 = vec![0.1; h * inter];
            let mut output = vec![0.0; h];
            fused_residual_ffn_f32(&input, &residual, &w1, &w2, &mut output, h, inter);
            for &o in &output {
                assert!(o.is_finite());
            }
        }
    }

    // ── Quantized FFN tests ─────────────────────────────────────────

    #[test]
    fn test_i2s_decode() {
        assert_eq!(decode_i2s(0b00), 0.0);
        assert_eq!(decode_i2s(0b01), 1.0);
        assert_eq!(decode_i2s(0b11), -1.0);
        assert_eq!(decode_i2s(0b10), 0.0);
    }

    #[test]
    fn test_i2s_pack_unpack() {
        let values = vec![1, -1, 0, 1];
        let packed = pack_i2s(&values);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_i2s_byte(packed[0]);
        assert_eq!(unpacked[0], 1.0);
        assert_eq!(unpacked[1], -1.0);
        assert_eq!(unpacked[2], 0.0);
        assert_eq!(unpacked[3], 1.0);
    }

    #[test]
    fn test_quantized_ffn_identity_scale() {
        // 4×4 identity in I2_S with scale=1
        let h = 4;
        let inter = 4;
        let input = vec![1.0, 0.5, -0.5, 0.0];

        // Identity: diagonal = +1 (0b01), off-diagonal = 0 (0b00)
        let w1_row_bytes = 1; // 4/4 = 1 byte per row
        let mut w1_packed = vec![0u8; inter * w1_row_bytes];
        // Row 0: [1,0,0,0] → byte = 0b00_00_00_01
        w1_packed[0] = 0b00_00_00_01;
        // Row 1: [0,1,0,0] → byte = 0b00_00_01_00
        w1_packed[1] = 0b00_00_01_00;
        // Row 2: [0,0,1,0] → byte = 0b00_01_00_00
        w1_packed[2] = 0b00_01_00_00;
        // Row 3: [0,0,0,1] → byte = 0b01_00_00_00
        w1_packed[3] = 0b01_00_00_00;

        let w1_scales = vec![1.0; inter];
        let w2_packed = w1_packed.clone();
        let w2_scales = vec![1.0; h];
        let mut output = vec![0.0; h];

        quantized_ffn_i2_f32(
            &input,
            &w1_packed,
            &w1_scales,
            &w2_packed,
            &w2_scales,
            &mut output,
            h,
            inter,
        );

        // Output should be gelu(input) through identity
        for i in 0..h {
            let expected = ref_gelu_f64(input[i] as f64) as f32;
            assert!(
                (output[i] - expected).abs() < TOL,
                "qffn[{i}]: got {}, expected {}",
                output[i],
                expected
            );
        }
    }

    #[test]
    fn test_quantized_ffn_scales() {
        let h = 4;
        let inter = 4;
        let input = vec![1.0, 0.0, 0.0, 0.0];

        // W1 row 0 has +1 at position 0 with scale 2.0
        let w1_row_bytes = 1;
        let mut w1_packed = vec![0u8; inter * w1_row_bytes];
        w1_packed[0] = 0b00_00_00_01; // first row: [1,0,0,0]
        let w1_scales = vec![2.0, 0.0, 0.0, 0.0];

        // W2: identity with scale 1
        let w2_packed = {
            let mut p = vec![0u8; h * w1_row_bytes];
            p[0] = 0b00_00_00_01;
            p[1] = 0b00_00_01_00;
            p[2] = 0b00_01_00_00;
            p[3] = 0b01_00_00_00;
            p
        };
        let w2_scales = vec![1.0; h];
        let mut output = vec![0.0; h];

        quantized_ffn_i2_f32(
            &input,
            &w1_packed,
            &w1_scales,
            &w2_packed,
            &w2_scales,
            &mut output,
            h,
            inter,
        );

        // Mid[0] = gelu(1.0 * 2.0) = gelu(2.0), rest = gelu(0) = 0
        let expected_0 = ref_gelu_f64(2.0) as f32;
        assert!(
            (output[0] - expected_0).abs() < TOL,
            "qffn scaled: got {}, expected {}",
            output[0],
            expected_0
        );
    }

    #[test]
    fn test_quantized_ffn_all_zeros_input() {
        let h = 4;
        let inter = 4;
        let input = vec![0.0; h];
        let w1_row_bytes = 1;
        let w1_packed = vec![0b01_01_01_01u8; inter * w1_row_bytes]; // all +1
        let w1_scales = vec![1.0; inter];
        let w2_packed = vec![0b01_01_01_01u8; h * ((inter + 3) / 4)];
        let w2_scales = vec![1.0; h];
        let mut output = vec![99.0; h];

        quantized_ffn_i2_f32(
            &input,
            &w1_packed,
            &w1_scales,
            &w2_packed,
            &w2_scales,
            &mut output,
            h,
            inter,
        );

        for &o in &output {
            assert!(o.abs() < 1e-6, "Quantized FFN of zeros should be ~0, got {o}");
        }
    }

    #[test]
    fn test_quantized_ffn_negative_weights() {
        let h = 4;
        let inter = 4;
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let w1_row_bytes = 1;
        // All -1 weights: 0b11 per position
        let w1_packed = vec![0b11_11_11_11u8; inter * w1_row_bytes];
        let w1_scales = vec![1.0; inter];
        let w2_row_bytes = 1;
        let mut w2_packed = vec![0u8; h * w2_row_bytes];
        // W2 identity
        w2_packed[0] = 0b00_00_00_01;
        w2_packed[1] = 0b00_00_01_00;
        w2_packed[2] = 0b00_01_00_00;
        w2_packed[3] = 0b01_00_00_00;
        let w2_scales = vec![1.0; h];
        let mut output = vec![0.0; h];

        quantized_ffn_i2_f32(
            &input,
            &w1_packed,
            &w1_scales,
            &w2_packed,
            &w2_scales,
            &mut output,
            h,
            inter,
        );

        // dot(input, all-neg-ones) = -4.0 per row
        let mid_val = ref_gelu_f64(-4.0) as f32;
        for &o in &output {
            assert!(
                (o - mid_val).abs() < TOL,
                "got {o}, expected {mid_val}"
            );
        }
    }

    #[test]
    fn test_quantized_ffn_scalar_matches_dispatcher() {
        let h = 4;
        let inter = 4;
        let input = vec![0.5, -0.3, 0.8, 0.1];
        let w1_row_bytes = 1;
        let w1_packed = vec![0b01_11_00_01u8; inter * w1_row_bytes];
        let w1_scales = vec![0.5; inter];
        let w2_packed = vec![0b11_01_00_11u8; h * ((inter + 3) / 4)];
        let w2_scales = vec![0.3; h];
        let mut out_scalar = vec![0.0; h];
        let mut out_dispatch = vec![0.0; h];
        scalar_quantized_ffn_i2_f32(
            &input,
            &w1_packed,
            &w1_scales,
            &w2_packed,
            &w2_scales,
            &mut out_scalar,
            h,
            inter,
        );
        quantized_ffn_i2_f32(
            &input,
            &w1_packed,
            &w1_scales,
            &w2_packed,
            &w2_scales,
            &mut out_dispatch,
            h,
            inter,
        );
        for i in 0..h {
            assert!(
                (out_scalar[i] - out_dispatch[i]).abs() < 1e-5,
                "Mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_quantized_ffn_non_aligned_size() {
        // hidden=5 is not a multiple of 4
        let h = 5;
        let inter = 3;
        let input: Vec<f32> = (0..h).map(|i| i as f32 * 0.2).collect();
        let w1_row_bytes = (h + 3) / 4; // 2 bytes per row
        let w1_packed = vec![0b01_01_01_01u8; inter * w1_row_bytes];
        let w1_scales = vec![1.0; inter];
        let w2_row_bytes = (inter + 3) / 4; // 1 byte per row
        let w2_packed = vec![0b01_01_01_01u8; h * w2_row_bytes];
        let w2_scales = vec![1.0; h];
        let mut output = vec![0.0; h];

        quantized_ffn_i2_f32(
            &input,
            &w1_packed,
            &w1_scales,
            &w2_packed,
            &w2_scales,
            &mut output,
            h,
            inter,
        );

        for &o in &output {
            assert!(o.is_finite(), "Quantized FFN should handle non-aligned sizes");
        }
    }

    // ── Cross-function consistency tests ────────────────────────────

    #[test]
    fn test_gelu_silu_differ() {
        let input = vec![1.0, -1.0, 0.5, 2.0];
        let mut gelu_out = vec![0.0; 4];
        let mut silu_out = vec![0.0; 4];
        gelu_f32(&input, &mut gelu_out);
        silu_f32(&input, &mut silu_out);
        // GELU and SiLU should give different results for non-zero
        let different = gelu_out
            .iter()
            .zip(silu_out.iter())
            .any(|(&g, &s)| (g - s).abs() > 1e-6);
        assert!(different, "GELU and SiLU should produce different outputs");
    }

    #[test]
    fn test_gelu_silu_agree_at_zero() {
        let input = vec![0.0];
        let mut gelu_out = vec![99.0];
        let mut silu_out = vec![99.0];
        gelu_f32(&input, &mut gelu_out);
        silu_f32(&input, &mut silu_out);
        assert_eq!(gelu_out[0], 0.0);
        assert_eq!(silu_out[0], 0.0);
    }

    #[test]
    fn test_ffn_vs_swiglu_differ() {
        let h = 4;
        let inter = 4;
        let input = vec![1.0, 0.5, -0.5, 0.2];
        let w = identity_weights(inter, h);
        let wd = identity_weights(h, inter);
        let mut ffn_out = vec![0.0; h];
        let mut swiglu_out = vec![0.0; h];
        ffn_forward_f32(&input, &w, &wd, &mut ffn_out, h, inter);
        swiglu_ffn_f32(&input, &w, &w, &wd, &mut swiglu_out, h, inter);
        let different = ffn_out
            .iter()
            .zip(swiglu_out.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-6);
        assert!(
            different,
            "Standard FFN and SwiGLU should produce different outputs"
        );
    }

    #[test]
    fn test_large_hidden_dim() {
        let h = 128;
        let inter = 64;
        let input: Vec<f32> = (0..h).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let w1: Vec<f32> = (0..inter * h).map(|i| (i as f32 * 0.001) - 0.3).collect();
        let w2: Vec<f32> = (0..h * inter).map(|i| (i as f32 * 0.001) - 0.2).collect();
        let mut output = vec![0.0; h];
        ffn_forward_f32(&input, &w1, &w2, &mut output, h, inter);
        for &o in &output {
            assert!(o.is_finite());
        }
    }

    #[test]
    fn test_gelu_large_positive() {
        let input = vec![10.0, 50.0, 100.0];
        let mut output = vec![0.0; 3];
        gelu_f32(&input, &mut output);
        // For large positive x, GELU ≈ x
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(
                (o - x).abs() / x.abs() < 0.01,
                "gelu({x}) ≈ {x}, got {o}"
            );
        }
    }

    #[test]
    fn test_silu_large_positive() {
        let input = vec![10.0, 50.0, 100.0];
        let mut output = vec![0.0; 3];
        silu_f32(&input, &mut output);
        // For large positive x, SiLU ≈ x
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(
                (o - x).abs() / x.abs() < 0.01,
                "silu({x}) ≈ {x}, got {o}"
            );
        }
    }

    #[test]
    fn test_silu_symmetry() {
        // SiLU is NOT symmetric, but silu(x) + silu(-x) should be interesting
        // Actually: silu(-x) = -x * sigmoid(-x) = -x * (1 - sigmoid(x))
        let input_pos = vec![1.0, 2.0, 3.0];
        let input_neg: Vec<f32> = input_pos.iter().map(|&x| -x).collect();
        let mut out_pos = vec![0.0; 3];
        let mut out_neg = vec![0.0; 3];
        silu_f32(&input_pos, &mut out_pos);
        silu_f32(&input_neg, &mut out_neg);
        // silu(x) + silu(-x) = x * sig(x) + (-x) * sig(-x) = x * (sig(x) - sig(-x))
        // = x * (2*sig(x) - 1)
        for i in 0..3 {
            let sum = out_pos[i] + out_neg[i];
            let x = input_pos[i];
            let expected = x * (2.0 * ref_sigmoid_f64(x as f64) as f32 - 1.0);
            assert!(
                (sum - expected).abs() < TOL,
                "silu symmetry check failed at x={x}"
            );
        }
    }

    #[test]
    fn test_residual_preserves_values() {
        // If FFN output is ~0 (zero input), residual should pass through
        let h = 4;
        let inter = 4;
        let input = vec![0.0; h];
        let residual = vec![42.0, -17.5, 100.0, 0.001];
        let w1 = vec![0.0; inter * h];
        let w2 = vec![0.0; h * inter];
        let mut output = vec![0.0; h];
        fused_residual_ffn_f32(&input, &residual, &w1, &w2, &mut output, h, inter);
        for i in 0..h {
            assert!(
                (output[i] - residual[i]).abs() < 1e-6,
                "Residual should pass through when FFN output is 0"
            );
        }
    }

    #[test]
    fn test_gelu_16_elements() {
        // Exactly 4 NEON chunks, no tail
        let input: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let mut output = vec![0.0; 16];
        gelu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            let expected = ref_gelu_f64(x as f64) as f32;
            assert!((o - expected).abs() < TOL);
        }
    }

    #[test]
    fn test_silu_16_elements() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let mut output = vec![0.0; 16];
        silu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            let expected = ref_silu_f64(x as f64) as f32;
            assert!((o - expected).abs() < TOL);
        }
    }

    #[test]
    fn test_quantized_all_zeros_weights() {
        let h = 4;
        let inter = 4;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let w1_row_bytes = 1;
        let w1_packed = vec![0u8; inter * w1_row_bytes]; // all 0b00 = 0
        let w1_scales = vec![1.0; inter];
        let w2_packed = vec![0u8; h * ((inter + 3) / 4)];
        let w2_scales = vec![1.0; h];
        let mut output = vec![99.0; h];
        quantized_ffn_i2_f32(
            &input,
            &w1_packed,
            &w1_scales,
            &w2_packed,
            &w2_scales,
            &mut output,
            h,
            inter,
        );
        for &o in &output {
            assert!(o.abs() < 1e-6, "All-zero weights should produce ~0 output");
        }
    }

    #[test]
    fn test_ffn_single_element() {
        let h = 1;
        let inter = 1;
        let input = vec![2.0];
        let w1 = vec![0.5];
        let w2 = vec![0.3];
        let mut output = vec![0.0];
        ffn_forward_f32(&input, &w1, &w2, &mut output, h, inter);
        let expected = ref_gelu_f64(2.0 * 0.5) as f32 * 0.3;
        assert!((output[0] - expected).abs() < TOL);
    }

    #[test]
    fn test_swiglu_single_element() {
        let h = 1;
        let inter = 1;
        let input = vec![2.0];
        let w_gate = vec![0.5];
        let w_up = vec![0.3];
        let w_down = vec![0.7];
        let mut output = vec![0.0];
        swiglu_ffn_f32(&input, &w_gate, &w_up, &w_down, &mut output, h, inter);
        let gate_val = ref_silu_f64(2.0 * 0.5) as f32;
        let up_val = 2.0 * 0.3;
        let expected = gate_val * up_val * 0.7;
        assert!((output[0] - expected).abs() < TOL);
    }

    #[test]
    fn test_gelu_two_elements() {
        let input = vec![0.5, -0.5];
        let mut output = vec![0.0; 2];
        gelu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            let expected = ref_gelu_f64(x as f64) as f32;
            assert!((o - expected).abs() < TOL);
        }
    }

    #[test]
    fn test_silu_two_elements() {
        let input = vec![0.5, -0.5];
        let mut output = vec![0.0; 2];
        silu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            let expected = ref_silu_f64(x as f64) as f32;
            assert!((o - expected).abs() < TOL);
        }
    }

    #[test]
    fn test_ffn_asymmetric_dimensions() {
        let h = 3;
        let inter = 11;
        let input: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.2).collect();
        let w1: Vec<f32> = (0..inter * h).map(|i| (i as f32).sin() * 0.1).collect();
        let w2: Vec<f32> = (0..h * inter).map(|i| (i as f32).cos() * 0.1).collect();
        let mut out_scalar = vec![0.0; h];
        let mut out_dispatch = vec![0.0; h];
        scalar_ffn_forward_f32(&input, &w1, &w2, &mut out_scalar, h, inter);
        ffn_forward_f32(&input, &w1, &w2, &mut out_dispatch, h, inter);
        for i in 0..h {
            assert!((out_scalar[i] - out_dispatch[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_quantized_ffn_mixed_values() {
        // Mix of +1, -1, 0 weights with non-trivial input
        let h = 4;
        let inter = 4;
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let w1_row_bytes = 1;
        // Row patterns: [+1,-1,0,+1], [-1,+1,+1,0], [0,0,+1,-1], [+1,+1,-1,-1]
        let w1_packed = vec![
            0b01_00_11_01, // +1, -1, 0, +1
            0b00_01_01_11, // -1, +1, +1, 0
            0b11_01_00_00, // 0, 0, +1, -1
            0b11_11_01_01, // +1, +1, -1, -1
        ];
        let w1_scales = vec![1.0; inter];
        let w2_packed = vec![
            0b00_00_00_01,
            0b00_00_01_00,
            0b00_01_00_00,
            0b01_00_00_00,
        ];
        let w2_scales = vec![1.0; h];
        let mut out_s = vec![0.0; h];
        let mut out_d = vec![0.0; h];
        scalar_quantized_ffn_i2_f32(
            &input, &w1_packed, &w1_scales, &w2_packed, &w2_scales,
            &mut out_s, h, inter,
        );
        quantized_ffn_i2_f32(
            &input, &w1_packed, &w1_scales, &w2_packed, &w2_scales,
            &mut out_d, h, inter,
        );
        for i in 0..h {
            assert!(
                (out_s[i] - out_d[i]).abs() < 1e-5,
                "Mixed quant mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_gelu_three_elements() {
        let input = vec![-1.0, 0.0, 1.0];
        let mut output = vec![0.0; 3];
        gelu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            let expected = ref_gelu_f64(x as f64) as f32;
            assert!((o - expected).abs() < TOL);
        }
    }
}
