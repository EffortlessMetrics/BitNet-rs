//! NEON-optimized mixed precision GEMM v2 for Apple Silicon.
//! Handles I2_S (2-bit) × F16 → F32 matrix multiplication with
//! micro-tiling and fused accumulation.
//!
//! This module provides six primary operations, each with a NEON fast path
//! and a portable scalar fallback:
//!
//! 1. `mixed_i2s_f16_gemm_v2` — I2_S × F16 → F32 with 4×4 micro-tiles
//! 2. `mixed_i2s_f32_gemm_v2` — I2_S × F32 → F32 with 4×4 micro-tiles
//! 3. `mixed_bf16_f32_gemm`   — BF16 × F32 → F32 accumulation
//! 4. `fused_quantize_gemm`   — quantize-then-multiply in one pass
//! 5. `mixed_precision_batch_gemm` — batched mixed-precision GEMM
//! 6. `asymmetric_quant_gemm` — asymmetric quantized GEMM with zero-point
//!
//! I2_S encoding (2 bits per value, 4 values per byte, LSB-first):
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1
//! - `0b10` → unused (treated as 0)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Constants ──────────────────────────────────────────────────────

/// I2_S f32 LUT: index by 2-bit code → f32 value.
const I2S_LUT: [f32; 4] = [0.0, 1.0, 0.0, -1.0];

/// Micro-tile dimensions for the v2 tiled kernels.
const MICRO_M: usize = 4;
const MICRO_N: usize = 4;

// ── Helpers ────────────────────────────────────────────────────────

/// Unpack one packed byte into 4 f32 values via the LUT.
#[inline(always)]
fn unpack_byte_f32(byte: u8) -> [f32; 4] {
    [
        I2S_LUT[(byte & 0x03) as usize],
        I2S_LUT[((byte >> 2) & 0x03) as usize],
        I2S_LUT[((byte >> 4) & 0x03) as usize],
        I2S_LUT[((byte >> 6) & 0x03) as usize],
    ]
}

/// Scalar f16→f32 conversion.
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((127 - 15 + 1) as i32 + e) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exp == 0x1F {
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }
    let f32_exp = exp + (127 - 15);
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
}

/// Scalar bf16→f32 conversion: upper 16 bits of f32.
#[inline(always)]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Scalar f32→bf16 truncation.
#[inline(always)]
fn f32_to_bf16(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

/// Quantise a single f32 to a symmetric 2-bit I2_S code.
///
/// Returns the 2-bit packed code (0b01 = +1, 0b11 = -1, 0b00 = 0).
#[inline(always)]
fn quantize_to_i2s(val: f32, inv_scale: f32) -> u8 {
    let q = (val * inv_scale).round() as i32;
    match q.clamp(-1, 1) {
        1 => 0b01,
        -1 => 0b11,
        _ => 0b00,
    }
}

// ── 1. mixed_i2s_f16_gemm_v2 ──────────────────────────────────────

/// NEON I2_S × F16 → F32 GEMM v2 with 4×4 micro-tiles.
///
/// `C[m×n] += scale · dequant(W)[m×k] · A[k×n]`
///
/// Weight matrix `W` is row-major I2_S packed (`m` rows, each row
/// `ceil(k/4)` bytes). `A` is row-major f16 (as `u16`) `[k×n]`.
/// `C` is row-major f32 `[m×n]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mixed_i2s_f16_gemm_v2_neon(
    weights: &[u8],
    activations: &[u16],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    let packed_k = k.div_ceil(4);

    let m_tiles = m / MICRO_M;
    let n_tiles = n / MICRO_N;

    // Full micro-tiles
    for mt in 0..m_tiles {
        for nt in 0..n_tiles {
            let row0 = mt * MICRO_M;
            let col0 = nt * MICRO_N;

            let mut acc = [vdupq_n_f32(0.0); MICRO_M];

            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;

                // Load 4 f16 activations for this k, cols col0..col0+4
                let a_arr = [
                    f16_to_f32(activations[kk * n + col0]),
                    f16_to_f32(activations[kk * n + col0 + 1]),
                    f16_to_f32(activations[kk * n + col0 + 2]),
                    f16_to_f32(activations[kk * n + col0 + 3]),
                ];
                let va = vld1q_f32(a_arr.as_ptr());

                for ti in 0..MICRO_M {
                    let r = row0 + ti;
                    let byte = weights[r * packed_k + byte_idx];
                    let w_val = I2S_LUT[((byte >> bit_off) & 0x03) as usize];
                    let vw = vdupq_n_f32(w_val);
                    acc[ti] = vfmaq_f32(acc[ti], vw, va);
                }
            }

            let vscale = vdupq_n_f32(scale);
            for ti in 0..MICRO_M {
                let off = (row0 + ti) * n + col0;
                let vc = vld1q_f32(output.as_ptr().add(off));
                let result = vfmaq_f32(vc, acc[ti], vscale);
                vst1q_f32(output.as_mut_ptr().add(off), result);
            }
        }
    }

    // Remainder: scalar fallback for edges
    let row_full = m_tiles * MICRO_M;
    let col_full = n_tiles * MICRO_N;

    // Edge columns for tiled rows
    for row in 0..row_full {
        let w_row = row * packed_k;
        for col in col_full..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;
                let w_val = I2S_LUT[((weights[w_row + byte_idx] >> bit_off) & 0x03) as usize];
                sum += w_val * f16_to_f32(activations[kk * n + col]);
            }
            output[row * n + col] += sum * scale;
        }
    }

    // Edge rows (all columns)
    for row in row_full..m {
        let w_row = row * packed_k;
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;
                let w_val = I2S_LUT[((weights[w_row + byte_idx] >> bit_off) & 0x03) as usize];
                sum += w_val * f16_to_f32(activations[kk * n + col]);
            }
            output[row * n + col] += sum * scale;
        }
    }
}

/// Scalar fallback for `mixed_i2s_f16_gemm_v2`.
fn mixed_i2s_f16_gemm_v2_scalar(
    weights: &[u8],
    activations: &[u16],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    let packed_k = k.div_ceil(4);
    for row in 0..m {
        let w_row = row * packed_k;
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;
                let w_val = I2S_LUT[((weights[w_row + byte_idx] >> bit_off) & 0x03) as usize];
                sum += w_val * f16_to_f32(activations[kk * n + col]);
            }
            output[row * n + col] += sum * scale;
        }
    }
}

/// Mixed I2_S × F16 → F32 GEMM v2 with runtime dispatch.
///
/// On aarch64 with NEON available, uses 4×4 micro-tiled NEON path.
/// Otherwise falls back to portable scalar implementation.
pub fn mixed_i2s_f16_gemm_v2(
    weights: &[u8],
    activations: &[u16],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    let packed_k = k.div_ceil(4);
    assert!(weights.len() >= m * packed_k, "weights too short");
    assert!(activations.len() >= k * n, "activations too short");
    assert!(output.len() >= m * n, "output too short");

    if m == 0 || k == 0 || n == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                mixed_i2s_f16_gemm_v2_neon(weights, activations, output, m, k, n, scale);
            }
            return;
        }
    }
    mixed_i2s_f16_gemm_v2_scalar(weights, activations, output, m, k, n, scale);
}

// ── 2. mixed_i2s_f32_gemm_v2 ──────────────────────────────────────

/// NEON I2_S × F32 → F32 GEMM v2 with 4×4 micro-tiles.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mixed_i2s_f32_gemm_v2_neon(
    weights: &[u8],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    let packed_k = k.div_ceil(4);

    let m_tiles = m / MICRO_M;
    let n_tiles = n / MICRO_N;

    for mt in 0..m_tiles {
        for nt in 0..n_tiles {
            let row0 = mt * MICRO_M;
            let col0 = nt * MICRO_N;

            let mut acc = [vdupq_n_f32(0.0); MICRO_M];

            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;

                let va = vld1q_f32(activations.as_ptr().add(kk * n + col0));

                for ti in 0..MICRO_M {
                    let r = row0 + ti;
                    let byte = weights[r * packed_k + byte_idx];
                    let w_val = I2S_LUT[((byte >> bit_off) & 0x03) as usize];
                    let vw = vdupq_n_f32(w_val);
                    acc[ti] = vfmaq_f32(acc[ti], vw, va);
                }
            }

            let vscale = vdupq_n_f32(scale);
            for ti in 0..MICRO_M {
                let off = (row0 + ti) * n + col0;
                let vc = vld1q_f32(output.as_ptr().add(off));
                let result = vfmaq_f32(vc, acc[ti], vscale);
                vst1q_f32(output.as_mut_ptr().add(off), result);
            }
        }
    }

    // Remainder rows/cols via scalar
    let row_full = m_tiles * MICRO_M;
    let col_full = n_tiles * MICRO_N;

    for row in 0..row_full {
        let w_row = row * packed_k;
        for col in col_full..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;
                let w_val = I2S_LUT[((weights[w_row + byte_idx] >> bit_off) & 0x03) as usize];
                sum += w_val * activations[kk * n + col];
            }
            output[row * n + col] += sum * scale;
        }
    }

    for row in row_full..m {
        let w_row = row * packed_k;
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;
                let w_val = I2S_LUT[((weights[w_row + byte_idx] >> bit_off) & 0x03) as usize];
                sum += w_val * activations[kk * n + col];
            }
            output[row * n + col] += sum * scale;
        }
    }
}

/// Scalar fallback for `mixed_i2s_f32_gemm_v2`.
fn mixed_i2s_f32_gemm_v2_scalar(
    weights: &[u8],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    let packed_k = k.div_ceil(4);
    for row in 0..m {
        let w_row = row * packed_k;
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;
                let w_val = I2S_LUT[((weights[w_row + byte_idx] >> bit_off) & 0x03) as usize];
                sum += w_val * activations[kk * n + col];
            }
            output[row * n + col] += sum * scale;
        }
    }
}

/// Mixed I2_S × F32 → F32 GEMM v2 with runtime dispatch.
pub fn mixed_i2s_f32_gemm_v2(
    weights: &[u8],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    let packed_k = k.div_ceil(4);
    assert!(weights.len() >= m * packed_k, "weights too short");
    assert!(activations.len() >= k * n, "activations too short");
    assert!(output.len() >= m * n, "output too short");

    if m == 0 || k == 0 || n == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                mixed_i2s_f32_gemm_v2_neon(weights, activations, output, m, k, n, scale);
            }
            return;
        }
    }
    mixed_i2s_f32_gemm_v2_scalar(weights, activations, output, m, k, n, scale);
}

// ── 3. mixed_bf16_f32_gemm ─────────────────────────────────────────

/// NEON BF16 × F32 → F32 GEMM.
///
/// `C[m×n] += A_bf16[m×k] · B_f32[k×n]`
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mixed_bf16_f32_gemm_neon(
    a_bf16: &[u16],
    b_f32: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    for row in 0..m {
        let n_full = n / 4;
        for col4 in 0..n_full {
            let col0 = col4 * 4;
            let mut acc = vdupq_n_f32(0.0);

            for kk in 0..k {
                let a_val = bf16_to_f32(a_bf16[row * k + kk]);
                let va = vdupq_n_f32(a_val);
                let vb = vld1q_f32(b_f32.as_ptr().add(kk * n + col0));
                acc = vfmaq_f32(acc, va, vb);
            }

            let off = row * n + col0;
            let vc = vld1q_f32(output.as_ptr().add(off));
            vst1q_f32(output.as_mut_ptr().add(off), vaddq_f32(vc, acc));
        }

        // Remainder columns
        for col in (n_full * 4)..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += bf16_to_f32(a_bf16[row * k + kk]) * b_f32[kk * n + col];
            }
            output[row * n + col] += sum;
        }
    }
}

/// Scalar fallback for `mixed_bf16_f32_gemm`.
fn mixed_bf16_f32_gemm_scalar(
    a_bf16: &[u16],
    b_f32: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += bf16_to_f32(a_bf16[row * k + kk]) * b_f32[kk * n + col];
            }
            output[row * n + col] += sum;
        }
    }
}

/// Mixed BF16 × F32 → F32 GEMM with runtime dispatch.
pub fn mixed_bf16_f32_gemm(
    a_bf16: &[u16],
    b_f32: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    assert!(a_bf16.len() >= m * k, "a_bf16 too short");
    assert!(b_f32.len() >= k * n, "b_f32 too short");
    assert!(output.len() >= m * n, "output too short");

    if m == 0 || k == 0 || n == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                mixed_bf16_f32_gemm_neon(a_bf16, b_f32, output, m, k, n);
            }
            return;
        }
    }
    mixed_bf16_f32_gemm_scalar(a_bf16, b_f32, output, m, k, n);
}

// ── 4. fused_quantize_gemm ─────────────────────────────────────────

/// NEON fused quantize-then-GEMM.
///
/// Quantises `weights_f32[m×k]` to I2_S on the fly and multiplies by
/// `activations[k×n]`, accumulating into `output[m×n]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fused_quantize_gemm_neon(
    weights_f32: &[f32],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    quant_scale: f32,
) {
    let inv_scale = if quant_scale.abs() > f32::EPSILON {
        1.0 / quant_scale
    } else {
        0.0
    };

    for row in 0..m {
        let n_full = n / 4;
        for col4 in 0..n_full {
            let col0 = col4 * 4;
            let mut acc = vdupq_n_f32(0.0);

            for kk in 0..k {
                let raw = weights_f32[row * k + kk];
                let q = (raw * inv_scale).round().clamp(-1.0, 1.0);
                let w_val = q * quant_scale;
                let vw = vdupq_n_f32(w_val);
                let va = vld1q_f32(activations.as_ptr().add(kk * n + col0));
                acc = vfmaq_f32(acc, vw, va);
            }

            let off = row * n + col0;
            let vc = vld1q_f32(output.as_ptr().add(off));
            vst1q_f32(output.as_mut_ptr().add(off), vaddq_f32(vc, acc));
        }

        for col in (n_full * 4)..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let raw = weights_f32[row * k + kk];
                let q = (raw * inv_scale).round().clamp(-1.0, 1.0);
                sum += (q * quant_scale) * activations[kk * n + col];
            }
            output[row * n + col] += sum;
        }
    }
}

/// Scalar fallback for `fused_quantize_gemm`.
fn fused_quantize_gemm_scalar(
    weights_f32: &[f32],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    quant_scale: f32,
) {
    let inv_scale = if quant_scale.abs() > f32::EPSILON {
        1.0 / quant_scale
    } else {
        0.0
    };

    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let raw = weights_f32[row * k + kk];
                let q = (raw * inv_scale).round().clamp(-1.0, 1.0);
                sum += (q * quant_scale) * activations[kk * n + col];
            }
            output[row * n + col] += sum;
        }
    }
}

/// Fused quantize-then-GEMM with runtime dispatch.
///
/// Quantises `weights_f32` to symmetric ternary I2_S on the fly using
/// `quant_scale`, then computes the GEMM against `activations`.
pub fn fused_quantize_gemm(
    weights_f32: &[f32],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    quant_scale: f32,
) {
    assert!(weights_f32.len() >= m * k, "weights_f32 too short");
    assert!(activations.len() >= k * n, "activations too short");
    assert!(output.len() >= m * n, "output too short");

    if m == 0 || k == 0 || n == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                fused_quantize_gemm_neon(
                    weights_f32, activations, output, m, k, n, quant_scale,
                );
            }
            return;
        }
    }
    fused_quantize_gemm_scalar(weights_f32, activations, output, m, k, n, quant_scale);
}

// ── 5. mixed_precision_batch_gemm ──────────────────────────────────

/// NEON batched mixed-precision GEMM.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mixed_precision_batch_gemm_neon(
    weights: &[u8],
    activations_batch: &[u16],
    output_batch: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    batch_size: usize,
    scale: f32,
) {
    let a_stride = k * n;
    let c_stride = m * n;

    for b in 0..batch_size {
        let a_off = b * a_stride;
        let c_off = b * c_stride;
        mixed_i2s_f16_gemm_v2_neon(
            weights,
            &activations_batch[a_off..a_off + a_stride],
            &mut output_batch[c_off..c_off + c_stride],
            m,
            k,
            n,
            scale,
        );
    }
}

/// Scalar fallback for `mixed_precision_batch_gemm`.
fn mixed_precision_batch_gemm_scalar(
    weights: &[u8],
    activations_batch: &[u16],
    output_batch: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    batch_size: usize,
    scale: f32,
) {
    let a_stride = k * n;
    let c_stride = m * n;

    for b in 0..batch_size {
        let a_off = b * a_stride;
        let c_off = b * c_stride;
        mixed_i2s_f16_gemm_v2_scalar(
            weights,
            &activations_batch[a_off..a_off + a_stride],
            &mut output_batch[c_off..c_off + c_stride],
            m,
            k,
            n,
            scale,
        );
    }
}

/// Batched mixed-precision I2_S × F16 → F32 GEMM with runtime dispatch.
///
/// Computes `C_b[m×n] += scale · dequant(W)[m×k] · A_b[k×n]` for
/// `batch_size` independent F16 activation matrices sharing the same
/// packed I2_S weight matrix.
pub fn mixed_precision_batch_gemm(
    weights: &[u8],
    activations_batch: &[u16],
    output_batch: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    batch_size: usize,
    scale: f32,
) {
    let packed_k = k.div_ceil(4);
    let a_stride = k * n;
    let c_stride = m * n;
    assert!(weights.len() >= m * packed_k, "weights too short");
    assert!(
        activations_batch.len() >= batch_size * a_stride,
        "activations_batch too short"
    );
    assert!(
        output_batch.len() >= batch_size * c_stride,
        "output_batch too short"
    );

    if m == 0 || k == 0 || n == 0 || batch_size == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                mixed_precision_batch_gemm_neon(
                    weights,
                    activations_batch,
                    output_batch,
                    m,
                    k,
                    n,
                    batch_size,
                    scale,
                );
            }
            return;
        }
    }
    mixed_precision_batch_gemm_scalar(
        weights,
        activations_batch,
        output_batch,
        m,
        k,
        n,
        batch_size,
        scale,
    );
}

// ── 6. asymmetric_quant_gemm ───────────────────────────────────────

/// NEON asymmetric quantized GEMM with zero-point.
///
/// `C[m×n] += scale · (dequant(W)[m×k] - zero_point) · A[k×n]`
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn asymmetric_quant_gemm_neon(
    weights: &[u8],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
    zero_point: f32,
) {
    let packed_k = k.div_ceil(4);
    let vzp = vdupq_n_f32(zero_point);

    for row in 0..m {
        let w_row = row * packed_k;
        let n_full = n / 4;

        for col4 in 0..n_full {
            let col0 = col4 * 4;
            let mut acc = vdupq_n_f32(0.0);

            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;
                let byte = weights[w_row + byte_idx];
                let w_val = I2S_LUT[((byte >> bit_off) & 0x03) as usize];
                let vw = vsubq_f32(vdupq_n_f32(w_val), vzp);
                let va = vld1q_f32(activations.as_ptr().add(kk * n + col0));
                acc = vfmaq_f32(acc, vw, va);
            }

            let off = row * n + col0;
            let vc = vld1q_f32(output.as_ptr().add(off));
            let vscale = vdupq_n_f32(scale);
            vst1q_f32(
                output.as_mut_ptr().add(off),
                vfmaq_f32(vc, acc, vscale),
            );
        }

        for col in (n_full * 4)..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;
                let byte = weights[w_row + byte_idx];
                let w_val = I2S_LUT[((byte >> bit_off) & 0x03) as usize];
                sum += (w_val - zero_point) * activations[kk * n + col];
            }
            output[row * n + col] += sum * scale;
        }
    }
}

/// Scalar fallback for `asymmetric_quant_gemm`.
fn asymmetric_quant_gemm_scalar(
    weights: &[u8],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
    zero_point: f32,
) {
    let packed_k = k.div_ceil(4);
    for row in 0..m {
        let w_row = row * packed_k;
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;
                let byte = weights[w_row + byte_idx];
                let w_val = I2S_LUT[((byte >> bit_off) & 0x03) as usize];
                sum += (w_val - zero_point) * activations[kk * n + col];
            }
            output[row * n + col] += sum * scale;
        }
    }
}

/// Asymmetric quantized I2_S GEMM with zero-point and runtime dispatch.
///
/// Computes `C[m×n] += scale · (dequant(W)[m×k] - zero_point) · A[k×n]`.
/// The zero-point correction enables asymmetric quantization schemes.
pub fn asymmetric_quant_gemm(
    weights: &[u8],
    activations: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
    zero_point: f32,
) {
    let packed_k = k.div_ceil(4);
    assert!(weights.len() >= m * packed_k, "weights too short");
    assert!(activations.len() >= k * n, "activations too short");
    assert!(output.len() >= m * n, "output too short");

    if m == 0 || k == 0 || n == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                asymmetric_quant_gemm_neon(
                    weights, activations, output, m, k, n, scale, zero_point,
                );
            }
            return;
        }
    }
    asymmetric_quant_gemm_scalar(weights, activations, output, m, k, n, scale, zero_point);
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test helpers ───────────────────────────────────────────────

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (diff {}, tol {tol})",
                (x - y).abs()
            );
        }
    }

    /// Pack signed weights (−1, 0, +1) into I2_S row-major bytes.
    fn pack_i2s(weights: &[i8], m: usize, k: usize) -> Vec<u8> {
        let packed_k = k.div_ceil(4);
        let mut packed = vec![0u8; m * packed_k];
        for row in 0..m {
            for kk in 0..k {
                let code = match weights[row * k + kk] {
                    1 => 0b01u8,
                    -1 => 0b11u8,
                    _ => 0b00u8,
                };
                let byte_idx = kk / 4;
                let bit_off = (kk % 4) * 2;
                packed[row * packed_k + byte_idx] |= code << bit_off;
            }
        }
        packed
    }

    /// Naive scalar GEMM: C[m×n] += scale · W[m×k] · A[k×n].
    fn naive_i2s_gemm(weights: &[i8], activations: &[f32], m: usize, k: usize, n: usize, scale: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += weights[row * k + kk] as f32 * activations[kk * n + col];
                }
                out[row * n + col] = sum * scale;
            }
        }
        out
    }

    /// Naive GEMM for f32 × f32.
    fn naive_f32_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[row * k + kk] * b[kk * n + col];
                }
                out[row * n + col] = sum;
            }
        }
        out
    }

    /// Convert f32 slice to f16 u16 bit-patterns.
    fn f32_slice_to_f16(vals: &[f32]) -> Vec<u16> {
        vals.iter().map(|&v| f32_to_f16_bits(v)).collect()
    }

    /// Scalar f32→f16 conversion for test data.
    fn f32_to_f16_bits(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;
        if exp == 0 {
            return (sign << 15) as u16;
        }
        if exp == 0xFF {
            return ((sign << 15) | (0x1F << 10) | (mant >> 13)) as u16;
        }
        let new_exp = exp - 127 + 15;
        if new_exp <= 0 {
            return (sign << 15) as u16;
        }
        if new_exp >= 0x1F {
            return ((sign << 15) | (0x1F << 10)) as u16;
        }
        ((sign << 15) | ((new_exp as u32) << 10) | (mant >> 13)) as u16
    }

    /// Convert f32 slice to bf16 u16 bit-patterns.
    fn f32_slice_to_bf16(vals: &[f32]) -> Vec<u16> {
        vals.iter().map(|&v| f32_to_bf16(v)).collect()
    }

    // ══════════════════════════════════════════════════════════════
    // 1. mixed_i2s_f16_gemm_v2 tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_i2s_f16_v2_identity_2x2() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let packed = pack_i2s(&w, 2, 2);
        let a_f32 = [1.0f32, 2.0, 3.0, 4.0];
        let a_f16 = f32_slice_to_f16(&a_f32);
        let mut out = vec![0.0f32; 4];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out, 2, 2, 2, 1.0);
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0], 0.01);
    }

    #[test]
    fn test_i2s_f16_v2_scale() {
        let w: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_i2s(&w, 2, 2);
        let a_f32 = [1.0f32, 1.0, 1.0, 1.0];
        let a_f16 = f32_slice_to_f16(&a_f32);
        let mut out = vec![0.0f32; 4];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out, 2, 2, 2, 0.5);
        assert_close(&out, &[1.0, 1.0, 1.0, 1.0], 0.01);
    }

    #[test]
    fn test_i2s_f16_v2_negative_weights() {
        let w: Vec<i8> = vec![-1, -1, -1, -1];
        let packed = pack_i2s(&w, 2, 2);
        let a_f32 = [1.0f32, 2.0, 3.0, 4.0];
        let a_f16 = f32_slice_to_f16(&a_f32);
        let mut out = vec![0.0f32; 4];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out, 2, 2, 2, 1.0);
        assert_close(&out, &[-4.0, -6.0, -4.0, -6.0], 0.01);
    }

    #[test]
    fn test_i2s_f16_v2_zero_weights() {
        let w = vec![0i8; 6];
        let packed = pack_i2s(&w, 2, 3);
        let a_f16 = f32_slice_to_f16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut out = vec![0.0f32; 4];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out, 2, 3, 2, 1.0);
        assert_close(&out, &[0.0, 0.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn test_i2s_f16_v2_1x1() {
        let w: Vec<i8> = vec![1];
        let packed = pack_i2s(&w, 1, 1);
        let a_f16 = f32_slice_to_f16(&[3.0]);
        let mut out = vec![0.0f32; 1];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out, 1, 1, 1, 2.0);
        assert_close(&out, &[6.0], 0.01);
    }

    #[test]
    fn test_i2s_f16_v2_empty_m() {
        let packed = pack_i2s(&[], 0, 4);
        let a_f16 = f32_slice_to_f16(&[0.0; 16]);
        let mut out = vec![0.0f32; 0];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out, 0, 4, 4, 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_i2s_f16_v2_empty_k() {
        let mut out = vec![0.0f32; 4];
        mixed_i2s_f16_gemm_v2(&[], &[], &mut out, 2, 0, 2, 1.0);
        assert_close(&out, &[0.0; 4], 1e-6);
    }

    #[test]
    fn test_i2s_f16_v2_empty_n() {
        let packed = pack_i2s(&[0; 8], 2, 4);
        let mut out = vec![0.0f32; 0];
        mixed_i2s_f16_gemm_v2(&packed, &[], &mut out, 2, 4, 0, 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_i2s_f16_v2_correctness_3x5x4() {
        let m = 3;
        let k = 5;
        let n = 4;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0][i % 3]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let expected = naive_i2s_gemm(&w, &a, m, k, n, 1.0);
        let packed = pack_i2s(&w, m, k);
        let a_f16 = f32_slice_to_f16(&a);
        let mut out = vec![0.0f32; m * n];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out, m, k, n, 1.0);
        assert_close(&out, &expected, 0.02);
    }

    #[test]
    fn test_i2s_f16_v2_correctness_4x8x4() {
        let m = 4;
        let k = 8;
        let n = 4;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();
        let expected = naive_i2s_gemm(&w, &a, m, k, n, 0.5);
        let packed = pack_i2s(&w, m, k);
        let a_f16 = f32_slice_to_f16(&a);
        let mut out = vec![0.0f32; m * n];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out, m, k, n, 0.5);
        assert_close(&out, &expected, 0.02);
    }

    #[test]
    fn test_i2s_f16_v2_large_16x32x8() {
        let m = 16;
        let k = 32;
        let n = 8;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0, 1, -1][i % 5]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.03).sin()).collect();
        let expected = naive_i2s_gemm(&w, &a, m, k, n, 1.0);
        let packed = pack_i2s(&w, m, k);
        let a_f16 = f32_slice_to_f16(&a);
        let mut out = vec![0.0f32; m * n];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out, m, k, n, 1.0);
        assert_close(&out, &expected, 0.05);
    }

    #[test]
    fn test_i2s_f16_v2_accumulates() {
        let w: Vec<i8> = vec![1, 1];
        let packed = pack_i2s(&w, 1, 2);
        let a_f16 = f32_slice_to_f16(&[1.0, 1.0]);
        let mut out = vec![10.0f32; 1];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out, 1, 2, 1, 1.0);
        assert_close(&out, &[12.0], 0.01);
    }

    #[test]
    fn test_i2s_f16_v2_neon_vs_scalar() {
        let m = 5;
        let k = 7;
        let n = 3;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0][i % 3]).collect();
        let packed = pack_i2s(&w, m, k);
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2 - 1.0).collect();
        let a_f16 = f32_slice_to_f16(&a);

        let mut out_scalar = vec![0.0f32; m * n];
        mixed_i2s_f16_gemm_v2_scalar(&packed, &a_f16, &mut out_scalar, m, k, n, 0.75);

        let mut out_dispatch = vec![0.0f32; m * n];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out_dispatch, m, k, n, 0.75);

        assert_close(&out_dispatch, &out_scalar, 1e-5);
    }

    // ══════════════════════════════════════════════════════════════
    // 2. mixed_i2s_f32_gemm_v2 tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_i2s_f32_v2_identity() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let packed = pack_i2s(&w, 2, 2);
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        mixed_i2s_f32_gemm_v2(&packed, &a, &mut out, 2, 2, 2, 1.0);
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_i2s_f32_v2_scale() {
        let w: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_i2s(&w, 2, 2);
        let a = [1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        mixed_i2s_f32_gemm_v2(&packed, &a, &mut out, 2, 2, 2, 3.0);
        assert_close(&out, &[6.0, 6.0, 6.0, 6.0], 1e-5);
    }

    #[test]
    fn test_i2s_f32_v2_zero_weights() {
        let w = vec![0i8; 8];
        let packed = pack_i2s(&w, 2, 4);
        let a = vec![1.0f32; 8];
        let mut out = vec![0.0f32; 4];
        mixed_i2s_f32_gemm_v2(&packed, &a, &mut out, 2, 4, 2, 1.0);
        assert_close(&out, &[0.0; 4], 1e-6);
    }

    #[test]
    fn test_i2s_f32_v2_1x1() {
        let w: Vec<i8> = vec![-1];
        let packed = pack_i2s(&w, 1, 1);
        let a = [5.0f32];
        let mut out = vec![0.0f32; 1];
        mixed_i2s_f32_gemm_v2(&packed, &a, &mut out, 1, 1, 1, 1.0);
        assert_close(&out, &[-5.0], 1e-6);
    }

    #[test]
    fn test_i2s_f32_v2_empty() {
        let packed = pack_i2s(&[], 0, 4);
        let mut out = vec![0.0f32; 0];
        mixed_i2s_f32_gemm_v2(&packed, &[0.0; 16], &mut out, 0, 4, 4, 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_i2s_f32_v2_correctness_3x7x5() {
        let m = 3;
        let k = 7;
        let n = 5;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0][i % 3]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let expected = naive_i2s_gemm(&w, &a, m, k, n, 0.5);
        let packed = pack_i2s(&w, m, k);
        let mut out = vec![0.0f32; m * n];
        mixed_i2s_f32_gemm_v2(&packed, &a, &mut out, m, k, n, 0.5);
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_i2s_f32_v2_exact_tile_4x4() {
        let m = 4;
        let k = 8;
        let n = 4;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let expected = naive_i2s_gemm(&w, &a, m, k, n, 1.0);
        let packed = pack_i2s(&w, m, k);
        let mut out = vec![0.0f32; m * n];
        mixed_i2s_f32_gemm_v2(&packed, &a, &mut out, m, k, n, 1.0);
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_i2s_f32_v2_remainder_tile_5x9x6() {
        let m = 5;
        let k = 9;
        let n = 6;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0, 1, -1][i % 5]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 - 0.3).collect();
        let expected = naive_i2s_gemm(&w, &a, m, k, n, 1.0);
        let packed = pack_i2s(&w, m, k);
        let mut out = vec![0.0f32; m * n];
        mixed_i2s_f32_gemm_v2(&packed, &a, &mut out, m, k, n, 1.0);
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_i2s_f32_v2_large_128x256() {
        let m = 8;
        let k = 128;
        let n = 16;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0, 1, -1, 0][i % 6]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.01).cos()).collect();
        let expected = naive_i2s_gemm(&w, &a, m, k, n, 1.0);
        let packed = pack_i2s(&w, m, k);
        let mut out = vec![0.0f32; m * n];
        mixed_i2s_f32_gemm_v2(&packed, &a, &mut out, m, k, n, 1.0);
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn test_i2s_f32_v2_accumulates() {
        let w: Vec<i8> = vec![1, 1];
        let packed = pack_i2s(&w, 1, 2);
        let a = [1.0f32, 1.0];
        let mut out = vec![5.0f32; 1];
        mixed_i2s_f32_gemm_v2(&packed, &a, &mut out, 1, 2, 1, 1.0);
        assert_close(&out, &[7.0], 1e-6);
    }

    #[test]
    fn test_i2s_f32_v2_neon_vs_scalar() {
        let m = 6;
        let k = 11;
        let n = 5;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0][i % 3]).collect();
        let packed = pack_i2s(&w, m, k);
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.15 - 0.5).collect();

        let mut out_scalar = vec![0.0f32; m * n];
        mixed_i2s_f32_gemm_v2_scalar(&packed, &a, &mut out_scalar, m, k, n, 1.0);

        let mut out_dispatch = vec![0.0f32; m * n];
        mixed_i2s_f32_gemm_v2(&packed, &a, &mut out_dispatch, m, k, n, 1.0);

        assert_close(&out_dispatch, &out_scalar, 1e-5);
    }

    #[test]
    fn test_i2s_f32_v2_vs_f16_v2() {
        let m = 4;
        let k = 8;
        let n = 4;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let packed = pack_i2s(&w, m, k);
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let a_f16 = f32_slice_to_f16(&a);

        let mut out_f32 = vec![0.0f32; m * n];
        mixed_i2s_f32_gemm_v2(&packed, &a, &mut out_f32, m, k, n, 1.0);

        let mut out_f16 = vec![0.0f32; m * n];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out_f16, m, k, n, 1.0);

        // f16 has reduced precision so use a wider tolerance
        assert_close(&out_f16, &out_f32, 0.05);
    }

    // ══════════════════════════════════════════════════════════════
    // 3. mixed_bf16_f32_gemm tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_bf16_f32_identity() {
        let a_f32 = [1.0f32, 0.0, 0.0, 1.0];
        let a_bf16 = f32_slice_to_bf16(&a_f32);
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        mixed_bf16_f32_gemm(&a_bf16, &b, &mut out, 2, 2, 2);
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0], 0.01);
    }

    #[test]
    fn test_bf16_f32_ones() {
        let a_f32 = [1.0f32; 4];
        let a_bf16 = f32_slice_to_bf16(&a_f32);
        let b = [1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        mixed_bf16_f32_gemm(&a_bf16, &b, &mut out, 2, 2, 2);
        assert_close(&out, &[2.0, 2.0, 2.0, 2.0], 0.01);
    }

    #[test]
    fn test_bf16_f32_1x1() {
        let a_bf16 = f32_slice_to_bf16(&[3.0]);
        let b = [4.0f32];
        let mut out = vec![0.0f32; 1];
        mixed_bf16_f32_gemm(&a_bf16, &b, &mut out, 1, 1, 1);
        assert_close(&out, &[12.0], 0.1);
    }

    #[test]
    fn test_bf16_f32_empty() {
        let a_bf16 = f32_slice_to_bf16(&[0.0; 16]);
        let mut out = vec![0.0f32; 0];
        mixed_bf16_f32_gemm(&a_bf16, &[0.0; 16], &mut out, 0, 4, 4);
        assert!(out.is_empty());
    }

    #[test]
    fn test_bf16_f32_correctness_3x4x2() {
        let m = 3;
        let k = 4;
        let n = 2;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.3).collect();
        let expected = naive_f32_gemm(&a, &b, m, k, n);
        let a_bf16 = f32_slice_to_bf16(&a);
        let mut out = vec![0.0f32; m * n];
        mixed_bf16_f32_gemm(&a_bf16, &b, &mut out, m, k, n);
        // bf16 has 7-bit mantissa → ~1% precision
        assert_close(&out, &expected, 0.15);
    }

    #[test]
    fn test_bf16_f32_accumulates() {
        let a_bf16 = f32_slice_to_bf16(&[1.0]);
        let b = [2.0f32];
        let mut out = vec![10.0f32; 1];
        mixed_bf16_f32_gemm(&a_bf16, &b, &mut out, 1, 1, 1);
        assert_close(&out, &[12.0], 0.1);
    }

    #[test]
    fn test_bf16_f32_neon_vs_scalar() {
        let m = 3;
        let k = 5;
        let n = 4;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2 - 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let a_bf16 = f32_slice_to_bf16(&a);

        let mut out_scalar = vec![0.0f32; m * n];
        mixed_bf16_f32_gemm_scalar(&a_bf16, &b, &mut out_scalar, m, k, n);

        let mut out_dispatch = vec![0.0f32; m * n];
        mixed_bf16_f32_gemm(&a_bf16, &b, &mut out_dispatch, m, k, n);

        assert_close(&out_dispatch, &out_scalar, 1e-5);
    }

    #[test]
    fn test_bf16_f32_large_8x16x8() {
        let m = 8;
        let k = 16;
        let n = 8;
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.1).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.07).cos()).collect();
        let expected = naive_f32_gemm(&a, &b, m, k, n);
        let a_bf16 = f32_slice_to_bf16(&a);
        let mut out = vec![0.0f32; m * n];
        mixed_bf16_f32_gemm(&a_bf16, &b, &mut out, m, k, n);
        assert_close(&out, &expected, 0.25);
    }

    #[test]
    fn test_bf16_conversion_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 100.0, -0.125];
        for &v in &values {
            let bf = f32_to_bf16(v);
            let back = bf16_to_f32(bf);
            // bf16 truncation loses lower 16 bits of mantissa
            assert!((v - back).abs() <= v.abs() * 0.01 + 1e-6, "bf16 roundtrip failed for {v}: got {back}");
        }
    }

    // ══════════════════════════════════════════════════════════════
    // 4. fused_quantize_gemm tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_fused_quant_identity() {
        // Weights [1,0; 0,1] with scale=1 → identity
        let w = [1.0f32, 0.0, 0.0, 1.0];
        let a = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 4];
        fused_quantize_gemm(&w, &a, &mut out, 2, 2, 2, 1.0);
        assert_close(&out, &[5.0, 6.0, 7.0, 8.0], 0.01);
    }

    #[test]
    fn test_fused_quant_scale() {
        // m=1, k=2, n=1: W=[0.8, 0.8] → quantize to [1,1], A=[3.0, 3.0]
        let w = [0.8f32, 0.8];
        let a = [3.0f32, 3.0];
        let mut out = vec![0.0f32; 1];
        fused_quantize_gemm(&w, &a, &mut out, 1, 2, 1, 1.0);
        // 1*3 + 1*3 = 6
        assert_close(&out, &[6.0], 0.01);
    }

    #[test]
    fn test_fused_quant_negative() {
        // m=1, k=2, n=1
        let w = [-0.8f32, -0.8];
        let a = [2.0f32, 2.0];
        let mut out = vec![0.0f32; 1];
        fused_quantize_gemm(&w, &a, &mut out, 1, 2, 1, 1.0);
        // -1*2 + -1*2 = -4
        assert_close(&out, &[-4.0], 0.01);
    }

    #[test]
    fn test_fused_quant_small_weights_clamp_to_zero() {
        // m=1, k=2, n=1: weights near zero should quantize to 0
        let w = [0.1f32, -0.1];
        let a = [100.0f32, 100.0];
        let mut out = vec![0.0f32; 1];
        fused_quantize_gemm(&w, &a, &mut out, 1, 2, 1, 1.0);
        assert_close(&out, &[0.0], 1.0);
    }

    #[test]
    fn test_fused_quant_empty() {
        let mut out = vec![0.0f32; 0];
        fused_quantize_gemm(&[0.0; 16], &[0.0; 16], &mut out, 0, 4, 4, 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_fused_quant_1x1() {
        let w = [1.0f32];
        let a = [7.0f32];
        let mut out = vec![0.0f32; 1];
        fused_quantize_gemm(&w, &a, &mut out, 1, 1, 1, 1.0);
        assert_close(&out, &[7.0], 0.01);
    }

    #[test]
    fn test_fused_quant_accumulates() {
        let w = [1.0f32];
        let a = [3.0f32];
        let mut out = vec![10.0f32; 1];
        fused_quantize_gemm(&w, &a, &mut out, 1, 1, 1, 1.0);
        assert_close(&out, &[13.0], 0.01);
    }

    #[test]
    fn test_fused_quant_neon_vs_scalar() {
        let m = 3;
        let k = 4;
        let n = 3;
        let w: Vec<f32> = (0..m * k).map(|i| [0.9, -0.9, 0.0, 0.9][i % 4]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();

        let mut out_scalar = vec![0.0f32; m * n];
        fused_quantize_gemm_scalar(&w, &a, &mut out_scalar, m, k, n, 1.0);

        let mut out_dispatch = vec![0.0f32; m * n];
        fused_quantize_gemm(&w, &a, &mut out_dispatch, m, k, n, 1.0);

        assert_close(&out_dispatch, &out_scalar, 1e-5);
    }

    #[test]
    fn test_fused_quant_correctness_4x8x4() {
        let m = 4;
        let k = 8;
        let n = 4;
        let w: Vec<f32> = (0..m * k).map(|i| [0.9, -0.9, 0.1, 0.9][i % 4]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1 - 1.0).collect();

        let mut out_fused = vec![0.0f32; m * n];
        fused_quantize_gemm(&w, &a, &mut out_fused, m, k, n, 1.0);

        // Manually quantize and compute
        let quant_w: Vec<i8> = w.iter().map(|&v| {
            let q = v.round().clamp(-1.0, 1.0) as i32;
            q as i8
        }).collect();
        let expected = naive_i2s_gemm(&quant_w, &a, m, k, n, 1.0);
        assert_close(&out_fused, &expected, 0.1);
    }

    // ══════════════════════════════════════════════════════════════
    // 5. mixed_precision_batch_gemm tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_gemm_single() {
        let m = 2;
        let k = 4;
        let n = 2;
        let w: Vec<i8> = vec![1, -1, 0, 1, 0, 1, -1, 0];
        let packed = pack_i2s(&w, m, k);
        let a_f32 = vec![1.0f32; k * n];
        let a_f16 = f32_slice_to_f16(&a_f32);
        let mut out = vec![0.0f32; m * n];
        mixed_precision_batch_gemm(&packed, &a_f16, &mut out, m, k, n, 1, 1.0);
        let expected = naive_i2s_gemm(&w, &a_f32, m, k, n, 1.0);
        assert_close(&out, &expected, 0.02);
    }

    #[test]
    fn test_batch_gemm_multi() {
        let m = 2;
        let k = 4;
        let n = 2;
        let batch = 3;
        let w: Vec<i8> = vec![1, -1, 0, 1, 0, 1, -1, 0];
        let packed = pack_i2s(&w, m, k);

        let mut a_batch_f32 = vec![0.0f32; batch * k * n];
        for b in 0..batch {
            for i in 0..(k * n) {
                a_batch_f32[b * k * n + i] = (b as f32 + 1.0) * (i as f32 + 1.0) * 0.1;
            }
        }
        let a_batch_f16 = f32_slice_to_f16(&a_batch_f32);

        let mut out = vec![0.0f32; batch * m * n];
        mixed_precision_batch_gemm(&packed, &a_batch_f16, &mut out, m, k, n, batch, 1.0);

        for b in 0..batch {
            let a_slice = &a_batch_f32[b * k * n..(b + 1) * k * n];
            let expected = naive_i2s_gemm(&w, a_slice, m, k, n, 1.0);
            let out_slice = &out[b * m * n..(b + 1) * m * n];
            assert_close(out_slice, &expected, 0.05);
        }
    }

    #[test]
    fn test_batch_gemm_empty_batch() {
        let w = pack_i2s(&[1i8, 0, 0, 1], 2, 2);
        let mut out = vec![0.0f32; 0];
        mixed_precision_batch_gemm(&w, &[], &mut out, 2, 2, 2, 0, 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_batch_gemm_neon_vs_scalar() {
        let m = 3;
        let k = 5;
        let n = 2;
        let batch = 2;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0][i % 3]).collect();
        let packed = pack_i2s(&w, m, k);
        let a_f32: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.2).collect();
        let a_f16 = f32_slice_to_f16(&a_f32);

        let mut out_scalar = vec![0.0f32; batch * m * n];
        mixed_precision_batch_gemm_scalar(&packed, &a_f16, &mut out_scalar, m, k, n, batch, 0.5);

        let mut out_dispatch = vec![0.0f32; batch * m * n];
        mixed_precision_batch_gemm(&packed, &a_f16, &mut out_dispatch, m, k, n, batch, 0.5);

        assert_close(&out_dispatch, &out_scalar, 1e-5);
    }

    #[test]
    fn test_batch_gemm_scale() {
        let m = 1;
        let k = 2;
        let n = 1;
        let w: Vec<i8> = vec![1, 1];
        let packed = pack_i2s(&w, m, k);
        let a_f16 = f32_slice_to_f16(&[1.0, 1.0]);
        let mut out = vec![0.0f32; 1];
        mixed_precision_batch_gemm(&packed, &a_f16, &mut out, m, k, n, 1, 3.0);
        assert_close(&out, &[6.0], 0.02);
    }

    // ══════════════════════════════════════════════════════════════
    // 6. asymmetric_quant_gemm tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_asymmetric_zero_zp() {
        // With zero_point=0, should match symmetric GEMM
        let m = 2;
        let k = 4;
        let n = 2;
        let w: Vec<i8> = vec![1, -1, 0, 1, 0, 1, -1, 0];
        let packed = pack_i2s(&w, m, k);
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();

        let mut out_sym = vec![0.0f32; m * n];
        mixed_i2s_f32_gemm_v2(&packed, &a, &mut out_sym, m, k, n, 1.0);

        let mut out_asym = vec![0.0f32; m * n];
        asymmetric_quant_gemm(&packed, &a, &mut out_asym, m, k, n, 1.0, 0.0);

        assert_close(&out_asym, &out_sym, 1e-5);
    }

    #[test]
    fn test_asymmetric_nonzero_zp() {
        // W = [1], A = [2], scale=1, zp=0.5
        // result = (1 - 0.5) * 2 = 1.0
        let w: Vec<i8> = vec![1];
        let packed = pack_i2s(&w, 1, 1);
        let a = [2.0f32];
        let mut out = vec![0.0f32; 1];
        asymmetric_quant_gemm(&packed, &a, &mut out, 1, 1, 1, 1.0, 0.5);
        assert_close(&out, &[1.0], 1e-5);
    }

    #[test]
    fn test_asymmetric_scale_and_zp() {
        let w: Vec<i8> = vec![1, -1];
        let packed = pack_i2s(&w, 1, 2);
        let a = [1.0f32, 1.0];
        // (1-0.25)*1 + (-1-0.25)*1 = 0.75 - 1.25 = -0.5
        // * scale 2.0 = -1.0
        let mut out = vec![0.0f32; 1];
        asymmetric_quant_gemm(&packed, &a, &mut out, 1, 2, 1, 2.0, 0.25);
        assert_close(&out, &[-1.0], 1e-5);
    }

    #[test]
    fn test_asymmetric_empty() {
        let packed = pack_i2s(&[], 0, 4);
        let mut out = vec![0.0f32; 0];
        asymmetric_quant_gemm(&packed, &[0.0; 16], &mut out, 0, 4, 4, 1.0, 0.0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_asymmetric_1x1() {
        let w: Vec<i8> = vec![-1];
        let packed = pack_i2s(&w, 1, 1);
        let a = [3.0f32];
        // (-1 - 0.0) * 3 * 1.0 = -3
        let mut out = vec![0.0f32; 1];
        asymmetric_quant_gemm(&packed, &a, &mut out, 1, 1, 1, 1.0, 0.0);
        assert_close(&out, &[-3.0], 1e-5);
    }

    #[test]
    fn test_asymmetric_accumulates() {
        let w: Vec<i8> = vec![1];
        let packed = pack_i2s(&w, 1, 1);
        let a = [2.0f32];
        let mut out = vec![10.0f32; 1];
        asymmetric_quant_gemm(&packed, &a, &mut out, 1, 1, 1, 1.0, 0.0);
        assert_close(&out, &[12.0], 1e-5);
    }

    #[test]
    fn test_asymmetric_correctness_3x5x4() {
        let m = 3;
        let k = 5;
        let n = 4;
        let zp = 0.1f32;
        let scale = 0.5;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0][i % 3]).collect();
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2 - 0.5).collect();
        let packed = pack_i2s(&w, m, k);

        // Compute expected: (W - zp) * A * scale
        let mut expected = vec![0.0f32; m * n];
        let packed_k = k.div_ceil(4);
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    let byte_idx = kk / 4;
                    let bit_off = (kk % 4) * 2;
                    let byte = packed[row * packed_k + byte_idx];
                    let w_val = I2S_LUT[((byte >> bit_off) & 0x03) as usize];
                    sum += (w_val - zp) * a[kk * n + col];
                }
                expected[row * n + col] = sum * scale;
            }
        }

        let mut out = vec![0.0f32; m * n];
        asymmetric_quant_gemm(&packed, &a, &mut out, m, k, n, scale, zp);
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_asymmetric_neon_vs_scalar() {
        let m = 4;
        let k = 6;
        let n = 3;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let packed = pack_i2s(&w, m, k);
        let a: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.3 - 1.0).collect();

        let mut out_scalar = vec![0.0f32; m * n];
        asymmetric_quant_gemm_scalar(&packed, &a, &mut out_scalar, m, k, n, 0.5, 0.1);

        let mut out_dispatch = vec![0.0f32; m * n];
        asymmetric_quant_gemm(&packed, &a, &mut out_dispatch, m, k, n, 0.5, 0.1);

        assert_close(&out_dispatch, &out_scalar, 1e-5);
    }

    #[test]
    fn test_asymmetric_large_zp() {
        // Large zero-point should shift all weights negative
        let w: Vec<i8> = vec![1, 1, 1, 1];
        let packed = pack_i2s(&w, 2, 2);
        let a = [1.0f32; 4];
        // (1 - 10) * 1 + (1 - 10) * 1 = -18 per element
        let mut out = vec![0.0f32; 4];
        asymmetric_quant_gemm(&packed, &a, &mut out, 2, 2, 2, 1.0, 10.0);
        assert_close(&out, &[-18.0, -18.0, -18.0, -18.0], 1e-4);
    }

    // ══════════════════════════════════════════════════════════════
    // Cross-cutting tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_f16_to_f32_special_values() {
        // +0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // -0
        assert_eq!(f16_to_f32(0x8000), -0.0);
        // +1
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // -1
        assert!((f16_to_f32(0xBC00) + 1.0).abs() < 1e-6);
        // +Inf
        assert!(f16_to_f32(0x7C00).is_infinite());
        // NaN
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn test_bf16_to_f32_special_values() {
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
        assert!(bf16_to_f32(0x7F80).is_infinite());
        assert!(bf16_to_f32(0x7FC0).is_nan());
    }

    #[test]
    fn test_i2s_lut_values() {
        assert_eq!(I2S_LUT[0b00], 0.0);
        assert_eq!(I2S_LUT[0b01], 1.0);
        assert_eq!(I2S_LUT[0b10], 0.0);
        assert_eq!(I2S_LUT[0b11], -1.0);
    }

    #[test]
    fn test_unpack_byte() {
        // byte = 0b11_01_00_01 = 0xD1 → [+1, 0, +1, -1]
        let vals = unpack_byte_f32(0b11_01_00_01);
        assert_eq!(vals, [1.0, 0.0, 1.0, -1.0]);
    }

    #[test]
    fn test_unpack_byte_zeros() {
        let vals = unpack_byte_f32(0x00);
        assert_eq!(vals, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_unpack_byte_all_positive() {
        // 0b01_01_01_01 = 0x55
        let vals = unpack_byte_f32(0x55);
        assert_eq!(vals, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_unpack_byte_all_negative() {
        // 0b11_11_11_11 = 0xFF
        let vals = unpack_byte_f32(0xFF);
        assert_eq!(vals, [-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_quantize_to_i2s_values() {
        assert_eq!(quantize_to_i2s(0.9, 1.0), 0b01);
        assert_eq!(quantize_to_i2s(-0.9, 1.0), 0b11);
        assert_eq!(quantize_to_i2s(0.1, 1.0), 0b00);
        assert_eq!(quantize_to_i2s(0.5, 1.0), 0b01); // rounds to 1
        assert_eq!(quantize_to_i2s(-0.5, 1.0), 0b11); // rounds to -1
    }

    #[test]
    fn test_no_catastrophic_cancellation_f16() {
        // Large values that could cause cancellation
        let m = 2;
        let k = 4;
        let n = 1;
        let w: Vec<i8> = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let packed = pack_i2s(&w, m, k);
        let a_f32 = [100.0f32, 100.0, 100.0, 100.0];
        let a_f16 = f32_slice_to_f16(&a_f32);
        let mut out = vec![0.0f32; m * n];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out, m, k, n, 1.0);
        // 1*100 + (-1)*100 + 1*100 + (-1)*100 = 0
        assert_close(&out, &[0.0, 0.0], 1.0);
    }

    #[test]
    fn test_no_catastrophic_cancellation_f32() {
        let m = 2;
        let k = 4;
        let n = 1;
        let w: Vec<i8> = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let packed = pack_i2s(&w, m, k);
        let a = [1000.0f32, 1000.0, 1000.0, 1000.0];
        let mut out = vec![0.0f32; m * n];
        mixed_i2s_f32_gemm_v2(&packed, &a, &mut out, m, k, n, 1.0);
        assert_close(&out, &[0.0, 0.0], 1e-3);
    }

    #[test]
    fn test_mixed_precision_consistency() {
        // Same computation via f32, f16, and bf16 paths should
        // agree within precision bounds.
        let m = 2;
        let k = 4;
        let n = 2;
        let w: Vec<i8> = vec![1, 0, -1, 1, -1, 1, 0, -1];
        let packed = pack_i2s(&w, m, k);
        let a_f32: Vec<f32> = vec![1.0, 2.0, 0.5, 1.5, 3.0, 0.25, 2.0, 1.0];

        let mut out_f32 = vec![0.0f32; m * n];
        mixed_i2s_f32_gemm_v2(&packed, &a_f32, &mut out_f32, m, k, n, 1.0);

        let a_f16 = f32_slice_to_f16(&a_f32);
        let mut out_f16 = vec![0.0f32; m * n];
        mixed_i2s_f16_gemm_v2(&packed, &a_f16, &mut out_f16, m, k, n, 1.0);

        // f16 is less precise, but should be within 5%
        assert_close(&out_f16, &out_f32, 0.1);
    }
}
