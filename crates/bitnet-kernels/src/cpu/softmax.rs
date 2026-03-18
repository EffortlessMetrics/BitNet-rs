//! Optimized CPU softmax with SIMD intrinsics and numerical stability.
//!
//! Provides standard, in-place, log, temperature-scaled, masked, top-K, online
//! (streaming), and batched softmax variants.  On x86-64 with AVX2, the hot
//! loops (find-max, exp-sum, normalize) are vectorized 8-wide; a scalar
//! fallback handles all other targets and tail elements.
#![allow(unsafe_op_in_unsafe_fn)]

use bitnet_common::{BitNetError, KernelError, Result};

// ── SIMD imports ────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
#[allow(clippy::wildcard_imports)]
use std::arch::aarch64::*;

// ── AVX2 helpers ────────────────────────────────────────────────────────

/// 8-wide AVX2 horizontal max → scalar.
///
/// # Safety
/// Caller must ensure AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax_avx2(v: __m256) -> f32 {
    // swap high/low 128-bit lanes and take element-wise max
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let m128 = _mm_max_ps(lo128, hi128);
    // reduce 4→2→1
    let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1));
    _mm_cvtss_f32(m32)
}

/// 8-wide AVX2 horizontal sum → scalar.
///
/// # Safety
/// Caller must ensure AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(lo128, hi128);
    let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    let s32 = _mm_add_ss(s64, _mm_shuffle_ps(s64, s64, 1));
    _mm_cvtss_f32(s32)
}

/// Fast scalar exp with clamping to avoid inf/NaN.
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    x.clamp(-88.0, 88.0).exp()
}

/// Vectorized exp(x) for 8×f32 using AVX2 (Cephes-style polynomial).
///
/// Uses the identity exp(x) = 2^(x * log2(e)) and splits into integer
/// and fractional parts.  The fractional part is approximated with a
/// degree-5 minimax polynomial (Cephes coefficients).
///
/// Accuracy: max relative error < 2e-7 over [-88, 88].
///
/// # Safety
/// Caller must ensure AVX2 + FMA are available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::excessive_precision)] // Cody-Waite constants need exact bit patterns
#[inline]
unsafe fn exp_avx2(x: __m256) -> __m256 {
    // Clamp to avoid overflow/underflow.
    let lo = _mm256_set1_ps(-88.376_26_f32);
    let hi = _mm256_set1_ps(88.376_26_f32);
    let x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);

    // Compute t = x * log2(e) and split into integer n and fraction f.
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let t = _mm256_mul_ps(x, log2e);
    let n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // f = x - n * ln(2)  (Cody-Waite range reduction for precision)
    let ln2_hi = _mm256_set1_ps(0.693_145_751_953_125_f32);
    let ln2_lo = _mm256_set1_ps(1.428_606_765_330_187_e-6_f32);
    let f = _mm256_sub_ps(_mm256_sub_ps(x, _mm256_mul_ps(n, ln2_hi)), _mm256_mul_ps(n, ln2_lo));

    // Polynomial approximation of exp(f) - 1 on [-ln2/2, ln2/2].
    // Coefficients from Cephes (degree 5 minimax).
    let c5 = _mm256_set1_ps(1.987_569_1e-4);
    let c4 = _mm256_set1_ps(1.398_199_9e-3);
    let c3 = _mm256_set1_ps(8.333_452e-3);
    let c2 = _mm256_set1_ps(4.166_579_6e-2);
    let c1 = _mm256_set1_ps(1.666_666_6e-1);
    let c0 = _mm256_set1_ps(5.000_000_2e-1);
    let one = _mm256_set1_ps(1.0);

    // Horner's method: p = ((((c5*f + c4)*f + c3)*f + c2)*f + c1)*f + c0
    let mut p = _mm256_fmadd_ps(c5, f, c4);
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(f, f), _mm256_add_ps(f, one));

    // Reconstruct: exp(x) = p * 2^n via exponent bit manipulation.
    let ni = _mm256_cvtps_epi32(n);
    let pow2n =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
    _mm256_mul_ps(p, pow2n)
}

// ── Core scalar implementation ──────────────────────────────────────────

/// Find max of a slice (scalar).
fn scalar_max(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Numerically-stable softmax written to `output` (scalar path).
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn softmax_scalar(input: &[f32], output: &mut [f32]) {
    if input.is_empty() {
        return;
    }
    let max_val = scalar_max(input);
    let mut sum = 0.0f32;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        let e = fast_exp(x - max_val);
        *o = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for o in output.iter_mut() {
            *o *= inv;
        }
    }
}

// ── NEON helpers ────────────────────────────────────────────────────────

/// 4-wide NEON horizontal max → scalar.
///
/// # Safety
/// Caller must ensure this is called on an aarch64 target.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn hmax_neon(v: float32x4_t) -> f32 {
    let pair = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
    let pair = vpmax_f32(pair, pair);
    vget_lane_f32(pair, 0)
}

/// 4-wide NEON horizontal sum → scalar.
///
/// # Safety
/// Caller must ensure this is called on an aarch64 target.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn hsum_neon(v: float32x4_t) -> f32 {
    let pair = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
    let pair = vpadd_f32(pair, pair);
    vget_lane_f32(pair, 0)
}

/// Vectorized exp(x) for 4×f32 using NEON (Cephes-style polynomial).
///
/// Uses the same identity and coefficients as [`exp_avx2`] but operates
/// on 4-wide `float32x4_t` vectors.
///
/// Accuracy: max relative error < 2e-7 over [-88, 88].
///
/// # Safety
/// Caller must ensure this is called on an aarch64 target.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::excessive_precision)]
#[inline]
unsafe fn exp_neon(x: float32x4_t) -> float32x4_t {
    // Clamp to avoid overflow/underflow.
    let lo = vdupq_n_f32(-88.376_26_f32);
    let hi = vdupq_n_f32(88.376_26_f32);
    let x = vminq_f32(vmaxq_f32(x, lo), hi);

    // Compute t = x * log2(e) and split into integer n and fraction f.
    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let t = vmulq_f32(x, log2e);
    let n = vrndnq_f32(t);

    // f = x - n * ln(2)  (Cody-Waite range reduction for precision)
    let ln2_hi = vdupq_n_f32(0.693_145_751_953_125_f32);
    let ln2_lo = vdupq_n_f32(1.428_606_765_330_187_e-6_f32);
    let f = vsubq_f32(vsubq_f32(x, vmulq_f32(n, ln2_hi)), vmulq_f32(n, ln2_lo));

    // Polynomial approximation of exp(f) - 1 on [-ln2/2, ln2/2].
    // Coefficients from Cephes (degree 5 minimax).
    let c5 = vdupq_n_f32(1.987_569_1e-4);
    let c4 = vdupq_n_f32(1.398_199_9e-3);
    let c3 = vdupq_n_f32(8.333_452e-3);
    let c2 = vdupq_n_f32(4.166_579_6e-2);
    let c1 = vdupq_n_f32(1.666_666_6e-1);
    let c0 = vdupq_n_f32(5.000_000_2e-1);
    let one = vdupq_n_f32(1.0);

    // Horner's method: p = ((((c5*f + c4)*f + c3)*f + c2)*f + c1)*f + c0
    let mut p = vfmaq_f32(c4, c5, f);
    p = vfmaq_f32(c3, p, f);
    p = vfmaq_f32(c2, p, f);
    p = vfmaq_f32(c1, p, f);
    p = vfmaq_f32(c0, p, f);
    p = vfmaq_f32(vaddq_f32(f, one), p, vmulq_f32(f, f));

    // Reconstruct: exp(x) = p * 2^n via exponent bit manipulation.
    let ni = vcvtq_s32_f32(n);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23));
    vmulq_f32(p, pow2n)
}

// ── NEON softmax core ──────────────────────────────────────────────────

/// Numerically-stable softmax using NEON intrinsics.
///
/// # Safety
/// Caller must ensure this is called on an aarch64 target.
#[cfg(target_arch = "aarch64")]
unsafe fn softmax_neon(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    if n == 0 {
        return;
    }

    // ── pass 1: find max ────────────────────────────────────────────────
    let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
    let chunks = n / 4;
    let inp = input.as_ptr();

    for i in 0..chunks {
        let v = vld1q_f32(inp.add(i * 4));
        vmax = vmaxq_f32(vmax, v);
    }
    let mut max_val = hmax_neon(vmax);
    for i in (chunks * 4)..n {
        max_val = max_val.max(*inp.add(i));
    }

    // ── pass 2: exp(x - max) and accumulate sum ─────────────────────────
    let vmax_bc = vdupq_n_f32(max_val);
    let mut vsum = vdupq_n_f32(0.0);
    let outp = output.as_mut_ptr();

    for i in 0..chunks {
        let v = vld1q_f32(inp.add(i * 4));
        let shifted = vsubq_f32(v, vmax_bc);
        let exp_v = exp_neon(shifted);
        vst1q_f32(outp.add(i * 4), exp_v);
        vsum = vaddq_f32(vsum, exp_v);
    }
    let mut sum_exp = hsum_neon(vsum);
    for i in (chunks * 4)..n {
        let e = fast_exp(*inp.add(i) - max_val);
        *outp.add(i) = e;
        sum_exp += e;
    }

    // ── pass 3: normalize ───────────────────────────────────────────────
    if sum_exp > 0.0 {
        let inv = vdupq_n_f32(1.0 / sum_exp);
        for i in 0..chunks {
            let v = vld1q_f32(outp.add(i * 4));
            vst1q_f32(outp.add(i * 4), vmulq_f32(v, inv));
        }
        let inv_s = 1.0 / sum_exp;
        for i in (chunks * 4)..n {
            *outp.add(i) *= inv_s;
        }
    }
}

/// In-place numerically-stable softmax using NEON intrinsics.
///
/// # Safety
/// Caller must ensure this is called on an aarch64 target.
#[cfg(target_arch = "aarch64")]
unsafe fn softmax_neon_inplace(data: &mut [f32]) {
    let n = data.len();
    if n == 0 {
        return;
    }

    // ── pass 1: find max ────────────────────────────────────────────────
    let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
    let chunks = n / 4;
    let ptr = data.as_mut_ptr();

    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        vmax = vmaxq_f32(vmax, v);
    }
    let mut max_val = hmax_neon(vmax);
    for i in (chunks * 4)..n {
        max_val = max_val.max(*ptr.add(i));
    }

    // ── pass 2: exp(x - max) in-place and accumulate sum ────────────────
    let vmax_bc = vdupq_n_f32(max_val);
    let mut vsum = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        let shifted = vsubq_f32(v, vmax_bc);
        let exp_v = exp_neon(shifted);
        vst1q_f32(ptr.add(i * 4), exp_v);
        vsum = vaddq_f32(vsum, exp_v);
    }
    let mut sum_exp = hsum_neon(vsum);
    for i in (chunks * 4)..n {
        let e = fast_exp(*ptr.add(i) - max_val);
        *ptr.add(i) = e;
        sum_exp += e;
    }

    // ── pass 3: normalize ───────────────────────────────────────────────
    if sum_exp > 0.0 {
        let inv = vdupq_n_f32(1.0 / sum_exp);
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            vst1q_f32(ptr.add(i * 4), vmulq_f32(v, inv));
        }
        let inv_s = 1.0 / sum_exp;
        for i in (chunks * 4)..n {
            *ptr.add(i) *= inv_s;
        }
    }
}

// ── AVX2 softmax core ──────────────────────────────────────────────────

/// Numerically-stable softmax using AVX2 intrinsics.
///
/// # Safety
/// Caller must ensure AVX2 + FMA are available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn softmax_avx2(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    if n == 0 {
        return;
    }

    // ── pass 1: find max ────────────────────────────────────────────────
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    let chunks = n / 8;
    let inp = input.as_ptr();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(inp.add(i * 8));
        vmax = _mm256_max_ps(vmax, v);
    }
    let mut max_val = hmax_avx2(vmax);
    for i in (chunks * 8)..n {
        max_val = max_val.max(*inp.add(i));
    }

    // ── pass 2: exp(x - max) and accumulate sum ─────────────────────────
    let vmax_bc = _mm256_set1_ps(max_val);
    let mut vsum = _mm256_setzero_ps();
    let outp = output.as_mut_ptr();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(inp.add(i * 8));
        let shifted = _mm256_sub_ps(v, vmax_bc);
        let exp_v = exp_avx2(shifted);
        _mm256_storeu_ps(outp.add(i * 8), exp_v);
        vsum = _mm256_add_ps(vsum, exp_v);
    }
    let mut sum_exp = hsum_avx2(vsum);
    for i in (chunks * 8)..n {
        let e = fast_exp(*inp.add(i) - max_val);
        *outp.add(i) = e;
        sum_exp += e;
    }

    // ── pass 3: normalize ───────────────────────────────────────────────
    if sum_exp > 0.0 {
        let inv = _mm256_set1_ps(1.0 / sum_exp);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(outp.add(i * 8));
            _mm256_storeu_ps(outp.add(i * 8), _mm256_mul_ps(v, inv));
        }
        let inv_s = 1.0 / sum_exp;
        for i in (chunks * 8)..n {
            *outp.add(i) *= inv_s;
        }
    }
}

/// In-place numerically-stable softmax using AVX2 intrinsics.
///
/// Avoids the allocation in `softmax_avx2` by reading and writing
/// the same buffer (each element is read once then overwritten).
///
/// # Safety
/// Caller must ensure AVX2 + FMA are available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn softmax_avx2_inplace(data: &mut [f32]) {
    let n = data.len();
    if n == 0 {
        return;
    }

    // ── pass 1: find max ────────────────────────────────────────────────
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    let chunks = n / 8;
    let ptr = data.as_mut_ptr();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        vmax = _mm256_max_ps(vmax, v);
    }
    let mut max_val = hmax_avx2(vmax);
    for i in (chunks * 8)..n {
        max_val = max_val.max(*ptr.add(i));
    }

    // ── pass 2: exp(x - max) in-place and accumulate sum ────────────────
    let vmax_bc = _mm256_set1_ps(max_val);
    let mut vsum = _mm256_setzero_ps();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        let shifted = _mm256_sub_ps(v, vmax_bc);
        let exp_v = exp_avx2(shifted);
        _mm256_storeu_ps(ptr.add(i * 8), exp_v);
        vsum = _mm256_add_ps(vsum, exp_v);
    }
    let mut sum_exp = hsum_avx2(vsum);
    for i in (chunks * 8)..n {
        let e = fast_exp(*ptr.add(i) - max_val);
        *ptr.add(i) = e;
        sum_exp += e;
    }

    // ── pass 3: normalize ───────────────────────────────────────────────
    if sum_exp > 0.0 {
        let inv = _mm256_set1_ps(1.0 / sum_exp);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(ptr.add(i * 8));
            _mm256_storeu_ps(ptr.add(i * 8), _mm256_mul_ps(v, inv));
        }
        let inv_s = 1.0 / sum_exp;
        for i in (chunks * 8)..n {
            *ptr.add(i) *= inv_s;
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────

/// Numerically-stable softmax over `input`, written to `output`.
///
/// Uses AVX2 when available on x86-64, otherwise falls back to scalar.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] when `input.len() != output.len()`.
pub fn softmax_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("input/output length mismatch: {} vs {}", input.len(), output.len()),
        }));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detection above guarantees AVX2 + FMA.
            unsafe { softmax_avx2(input, output) };
            return Ok(());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64.
        unsafe { softmax_neon(input, output) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    softmax_scalar(input, output);

    Ok(())
}

/// In-place numerically-stable softmax.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] when `data` is empty with no
/// meaningful softmax (this implementation silently succeeds on empty input
/// for ergonomic use).
pub fn softmax_f32_inplace(data: &mut [f32]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detection above guarantees AVX2 + FMA.
            unsafe { softmax_avx2_inplace(data) };
            return Ok(());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64.
        unsafe { softmax_neon_inplace(data) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let max_val = scalar_max(data);
        let mut sum = 0.0f32;
        for x in data.iter_mut() {
            let e = fast_exp(*x - max_val);
            *x = e;
            sum += e;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for x in data.iter_mut() {
                *x *= inv;
            }
        }
    }
    Ok(())
}

/// AVX2+FMA numerically-stable log-softmax.
///
/// `log_softmax(x)_i = x_i - max - log(Σ exp(x_j - max))`
///
/// Uses vectorized exp and reduce for the sum-of-exp pass.
///
/// # Safety
/// Caller must ensure AVX2 + FMA are available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn log_softmax_avx2(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    if n == 0 {
        return;
    }

    // ── pass 1: find max ────────────────────────────────────────────────
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    let chunks = n / 8;
    let inp = input.as_ptr();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(inp.add(i * 8));
        vmax = _mm256_max_ps(vmax, v);
    }
    let mut max_val = hmax_avx2(vmax);
    for i in (chunks * 8)..n {
        max_val = max_val.max(*inp.add(i));
    }

    // ── pass 2: sum exp(x - max) ────────────────────────────────────────
    let vmax_bc = _mm256_set1_ps(max_val);
    let mut vsum = _mm256_setzero_ps();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(inp.add(i * 8));
        let shifted = _mm256_sub_ps(v, vmax_bc);
        let exp_v = exp_avx2(shifted);
        vsum = _mm256_add_ps(vsum, exp_v);
    }
    let mut sum_exp = hsum_avx2(vsum);
    for i in (chunks * 8)..n {
        sum_exp += fast_exp(*inp.add(i) - max_val);
    }

    // ── pass 3: output[i] = input[i] - log_sum_exp ─────────────────────
    let log_sum_exp = max_val + sum_exp.ln();
    let vlse = _mm256_set1_ps(log_sum_exp);
    let outp = output.as_mut_ptr();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(inp.add(i * 8));
        _mm256_storeu_ps(outp.add(i * 8), _mm256_sub_ps(v, vlse));
    }
    for i in (chunks * 8)..n {
        *outp.add(i) = *inp.add(i) - log_sum_exp;
    }
}

/// Numerically-stable log-softmax: `log_softmax(x)_i = x_i - max - log(Σ exp(x_j - max))`.
///
/// On x86-64 with AVX2+FMA, uses vectorized exp and SIMD reduce.
/// Falls back to scalar on all other targets.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on length mismatch.
pub fn log_softmax_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("input/output length mismatch: {} vs {}", input.len(), output.len()),
        }));
    }
    if input.is_empty() {
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified above; lengths checked.
            unsafe {
                log_softmax_avx2(input, output);
            }
            return Ok(());
        }
    }

    let max_val = scalar_max(input);
    let mut sum_exp = 0.0f32;
    for &x in input {
        sum_exp += fast_exp(x - max_val);
    }
    let log_sum_exp = max_val + sum_exp.ln();
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x - log_sum_exp;
    }
    Ok(())
}

/// Temperature-scaled softmax: `softmax(x / temperature)`.
///
/// When `temperature` is very close to zero (< 1e-7), the output is a
/// one-hot vector at the argmax position.  Very large temperatures produce
/// a near-uniform distribution.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on length mismatch or
/// negative temperature.
pub fn softmax_with_temperature(input: &[f32], output: &mut [f32], temperature: f32) -> Result<()> {
    if input.len() != output.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("input/output length mismatch: {} vs {}", input.len(), output.len()),
        }));
    }
    if temperature < 0.0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("temperature must be non-negative, got {temperature}"),
        }));
    }
    if input.is_empty() {
        return Ok(());
    }

    // Near-zero temperature → one-hot at argmax.
    if temperature < 1e-7 {
        output.fill(0.0);
        let argmax = input
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        output[argmax] = 1.0;
        return Ok(());
    }

    // Scale into output buffer, then softmax in-place (avoids a Vec allocation).
    let inv_temp = 1.0 / temperature;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x * inv_temp;
    }
    softmax_f32_inplace(output)
}

/// Masked softmax: positions where `mask[i]` is `false` are set to 0 in the
/// output; remaining positions receive a valid softmax distribution that sums
/// to ~1.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on length mismatch.
pub fn softmax_with_mask(input: &[f32], output: &mut [f32], mask: &[bool]) -> Result<()> {
    if input.len() != output.len() || input.len() != mask.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "length mismatch: input={}, output={}, mask={}",
                input.len(),
                output.len(),
                mask.len()
            ),
        }));
    }
    if input.is_empty() {
        return Ok(());
    }

    // Write masked values directly into output, avoiding a temporary Vec.
    for ((&x, &m), o) in input.iter().zip(mask.iter()).zip(output.iter_mut()) {
        *o = if m { x } else { f32::NEG_INFINITY };
    }

    softmax_f32_inplace(output)?;

    // Ensure masked positions are exactly 0 (exp(-inf) may give tiny values).
    for (o, &m) in output.iter_mut().zip(mask.iter()) {
        if !m {
            *o = 0.0;
        }
    }
    Ok(())
}

/// Top-K softmax: computes softmax only over the `k` largest elements,
/// setting all others to 0.  The surviving values sum to ~1.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on length mismatch or `k == 0`.
pub fn softmax_topk(input: &[f32], output: &mut [f32], k: usize) -> Result<()> {
    if input.len() != output.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("input/output length mismatch: {} vs {}", input.len(), output.len()),
        }));
    }
    if k == 0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "k must be > 0".to_string(),
        }));
    }
    if input.is_empty() {
        return Ok(());
    }

    let effective_k = k.min(input.len());

    // Find the k-th largest value via partial sort on indices.
    let mut indices: Vec<usize> = (0..input.len()).collect();
    indices.select_nth_unstable_by(effective_k.saturating_sub(1), |&a, &b| {
        input[b].partial_cmp(&input[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Fill output with NEG_INFINITY, then copy top-k values into their positions.
    // This avoids allocating a separate bool mask Vec.
    output.fill(f32::NEG_INFINITY);
    for &idx in &indices[..effective_k] {
        output[idx] = input[idx];
    }

    softmax_f32_inplace(output)?;

    // Ensure non-top-k positions are exactly 0 (exp(-inf) may give tiny values).
    for o in output.iter_mut() {
        if *o < f32::MIN_POSITIVE {
            *o = 0.0;
        }
    }
    Ok(())
}

/// Online (streaming) softmax — single-pass numerically-stable algorithm.
///
/// Maintains a running max and a correction factor so the full output can be
/// produced in one scan without a separate max-finding pass.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on length mismatch.
pub fn softmax_online(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("input/output length mismatch: {} vs {}", input.len(), output.len()),
        }));
    }
    if input.is_empty() {
        return Ok(());
    }

    // Single-pass: track running max and denominator.
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0f32;

    for &x in input {
        if x > running_max {
            // Rescale accumulated sum to new max.
            running_sum *= fast_exp(running_max - x);
            running_max = x;
        }
        running_sum += fast_exp(x - running_max);
    }

    // Write normalized output.
    let log_sum = running_max + running_sum.ln();
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = fast_exp(x - log_sum);
    }
    Ok(())
}

/// Batched softmax: applies softmax independently to each row of a
/// `[batch_size, seq_len]` tensor stored contiguously.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] when `input.len() !=
/// batch_size * seq_len` or `output.len() != input.len()`.
pub fn batched_softmax_opt(
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    seq_len: usize,
) -> Result<()> {
    let total = batch_size * seq_len;
    if input.len() != total {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "input length {} != batch_size({}) * seq_len({})",
                input.len(),
                batch_size,
                seq_len
            ),
        }));
    }
    if output.len() != total {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "output length {} != batch_size({}) * seq_len({})",
                output.len(),
                batch_size,
                seq_len
            ),
        }));
    }

    for bi in 0..batch_size {
        let off = bi * seq_len;
        let row_in = &input[off..off + seq_len];
        let row_out = &mut output[off..off + seq_len];
        softmax_f32(row_in, row_out)?;
    }
    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert all values sum to ~1.
    fn assert_sums_to_one(v: &[f32], tol: f32) {
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < tol, "expected sum ≈ 1.0, got {s} (delta {})", (s - 1.0).abs());
    }

    /// Helper: assert all values are non-negative.
    fn assert_non_negative(v: &[f32]) {
        for (i, &x) in v.iter().enumerate() {
            assert!(x >= 0.0, "output[{i}] = {x} < 0");
        }
    }

    // ── softmax_f32 basic ───────────────────────────────────────────────

    #[test]
    fn test_softmax_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-6);
        assert_non_negative(&output);
        // Values should be monotonically increasing.
        for i in 1..output.len() {
            assert!(output[i] > output[i - 1]);
        }
    }

    #[test]
    fn test_softmax_single_element() {
        let input = [42.0];
        let mut output = [0.0; 1];
        softmax_f32(&input, &mut output).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_softmax_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        softmax_f32(&input, &mut output).unwrap();
    }

    #[test]
    fn test_softmax_all_same() {
        let input = [5.0; 8];
        let mut output = [0.0; 8];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-6);
        for &x in &output {
            assert!((x - 0.125).abs() < 1e-6, "expected uniform 1/8, got {x}");
        }
    }

    #[test]
    fn test_softmax_length_mismatch() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 3];
        assert!(softmax_f32(&input, &mut output).is_err());
    }

    // ── Numerical stability ─────────────────────────────────────────────

    #[test]
    fn test_softmax_large_positive() {
        let input = [1000.0, 1001.0, 1002.0];
        let mut output = [0.0; 3];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
        assert_non_negative(&output);
        for &x in &output {
            assert!(x.is_finite(), "non-finite output with large inputs");
        }
    }

    #[test]
    fn test_softmax_large_negative() {
        let input = [-1000.0, -999.0, -998.0];
        let mut output = [0.0; 3];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
        assert_non_negative(&output);
    }

    #[test]
    fn test_softmax_mixed_extreme() {
        let input = [-1000.0, 0.0, 1000.0];
        let mut output = [0.0; 3];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
        // The largest should dominate.
        assert!(output[2] > 0.99);
    }

    #[test]
    fn test_softmax_stability_no_nan() {
        let input = [f32::MAX / 2.0, f32::MAX / 2.0];
        let mut output = [0.0; 2];
        softmax_f32(&input, &mut output).unwrap();
        for &x in &output {
            assert!(!x.is_nan(), "NaN in output");
            assert!(x.is_finite(), "non-finite output");
        }
    }

    // ── In-place ────────────────────────────────────────────────────────

    #[test]
    fn test_softmax_inplace_basic() {
        let mut data = vec![1.0, 2.0, 3.0];
        softmax_f32_inplace(&mut data).unwrap();
        assert_sums_to_one(&data, 1e-6);
    }

    #[test]
    fn test_softmax_inplace_empty() {
        let mut data: Vec<f32> = vec![];
        softmax_f32_inplace(&mut data).unwrap();
    }

    #[test]
    fn test_softmax_inplace_matches_out_of_place() {
        let input = vec![0.5, -1.0, 2.0, 0.0, 1.5];
        let mut out1 = [0.0; 5];
        softmax_f32(&input, &mut out1).unwrap();
        let mut out2 = input.clone();
        softmax_f32_inplace(&mut out2).unwrap();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-7, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_softmax_inplace_single() {
        let mut data = [99.0];
        softmax_f32_inplace(&mut data).unwrap();
        assert!((data[0] - 1.0).abs() < 1e-7);
    }

    // ── log_softmax ─────────────────────────────────────────────────────

    #[test]
    fn test_log_softmax_basic() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 3];
        log_softmax_f32(&input, &mut output).unwrap();
        // All log-softmax values should be <= 0.
        for &x in &output {
            assert!(x <= 0.0, "log_softmax value {x} > 0");
        }
        // exp(log_softmax) should sum to 1.
        let exp_sum: f32 = output.iter().map(|&x| x.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_identity() {
        // log(softmax(x)) should equal log_softmax(x) for the same input.
        let input = [0.5, -1.0, 2.0, 0.0];
        let mut sm = [0.0; 4];
        softmax_f32(&input, &mut sm).unwrap();
        let log_sm: Vec<f32> = sm.iter().map(|&x| x.ln()).collect();

        let mut lsm = [0.0; 4];
        log_softmax_f32(&input, &mut lsm).unwrap();

        for (a, b) in log_sm.iter().zip(lsm.iter()) {
            assert!((a - b).abs() < 1e-5, "log(softmax) vs log_softmax mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_log_softmax_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        log_softmax_f32(&input, &mut output).unwrap();
    }

    #[test]
    fn test_log_softmax_length_mismatch() {
        let input = [1.0];
        let mut output = [0.0; 2];
        assert!(log_softmax_f32(&input, &mut output).is_err());
    }

    #[test]
    fn test_log_softmax_single() {
        let input = [5.0];
        let mut output = [0.0; 1];
        log_softmax_f32(&input, &mut output).unwrap();
        assert!((output[0] - 0.0).abs() < 1e-7, "log_softmax of single element should be 0");
    }

    #[test]
    fn test_log_softmax_large_values() {
        let input = [1000.0, 1001.0, 1002.0];
        let mut output = [0.0; 3];
        log_softmax_f32(&input, &mut output).unwrap();
        for &x in &output {
            assert!(x.is_finite(), "non-finite log_softmax with large inputs");
        }
        let exp_sum: f32 = output.iter().map(|&x| x.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_log_softmax_avx2_sized() {
        // 32 elements: exercises the AVX2 8-wide path (4 full chunks).
        let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.3 - 4.0).collect();
        let mut output = [0.0f32; 32];
        log_softmax_f32(&input, &mut output).unwrap();

        // All outputs must be ≤ 0.
        for (i, &x) in output.iter().enumerate() {
            assert!(x <= 0.0, "log_softmax[{i}] = {x} > 0");
            assert!(x.is_finite(), "log_softmax[{i}] is not finite");
        }
        // exp(log_softmax) should sum to 1.
        let exp_sum: f32 = output.iter().map(|&x| x.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-4, "sum = {exp_sum}");
    }

    #[test]
    fn test_log_softmax_matches_log_of_softmax() {
        // Verify: log(softmax(x)) ≈ log_softmax(x) for a large vector.
        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut sm = [0.0f32; 64];
        let mut lsm = [0.0f32; 64];
        softmax_f32(&input, &mut sm).unwrap();
        log_softmax_f32(&input, &mut lsm).unwrap();
        for (i, (&s, &l)) in sm.iter().zip(lsm.iter()).enumerate() {
            let log_s = s.ln();
            assert!(
                (log_s - l).abs() < 1e-4,
                "mismatch at [{i}]: log(softmax)={log_s} vs log_softmax={l}"
            );
        }
    }

    // ── Temperature ─────────────────────────────────────────────────────

    #[test]
    fn test_temperature_one_is_identity() {
        let input = [1.0, 2.0, 3.0];
        let mut out_t1 = [0.0; 3];
        softmax_with_temperature(&input, &mut out_t1, 1.0).unwrap();
        let mut out_plain = [0.0; 3];
        softmax_f32(&input, &mut out_plain).unwrap();
        for (a, b) in out_t1.iter().zip(out_plain.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_temperature_zero_is_argmax() {
        let input = [1.0, 5.0, 3.0, 2.0];
        let mut output = [0.0; 4];
        softmax_with_temperature(&input, &mut output, 0.0).unwrap();
        assert!((output[1] - 1.0).abs() < 1e-7, "argmax should be 1.0");
        assert!((output[0]).abs() < 1e-7);
        assert!((output[2]).abs() < 1e-7);
        assert!((output[3]).abs() < 1e-7);
    }

    #[test]
    fn test_temperature_high_approaches_uniform() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];
        softmax_with_temperature(&input, &mut output, 1e6).unwrap();
        let expected = 1.0 / 4.0;
        for &x in &output {
            assert!((x - expected).abs() < 1e-3, "expected near-uniform with high temp, got {x}");
        }
    }

    #[test]
    fn test_temperature_low_sharpens() {
        let input = [1.0, 2.0, 3.0];
        let mut out_sharp = [0.0; 3];
        softmax_with_temperature(&input, &mut out_sharp, 0.1).unwrap();
        // The max element should be very dominant.
        assert!(out_sharp[2] > 0.99);
    }

    #[test]
    fn test_temperature_negative_error() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 2];
        assert!(softmax_with_temperature(&input, &mut output, -1.0).is_err());
    }

    #[test]
    fn test_temperature_length_mismatch() {
        let input = [1.0];
        let mut output = [0.0; 2];
        assert!(softmax_with_temperature(&input, &mut output, 1.0).is_err());
    }

    #[test]
    fn test_temperature_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        softmax_with_temperature(&input, &mut output, 1.0).unwrap();
    }

    // ── Masked softmax ──────────────────────────────────────────────────

    #[test]
    fn test_masked_softmax_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [true, false, true, false];
        let mut output = [0.0; 4];
        softmax_with_mask(&input, &mut output, &mask).unwrap();

        assert!((output[1]).abs() < 1e-7, "masked position should be 0");
        assert!((output[3]).abs() < 1e-7, "masked position should be 0");
        // Unmasked should sum to ~1.
        let unmasked_sum: f32 = output.iter().sum();
        assert!((unmasked_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_masked_softmax_all_masked() {
        let input = [1.0, 2.0, 3.0];
        let mask = [false, false, false];
        let mut output = [0.0; 3];
        softmax_with_mask(&input, &mut output, &mask).unwrap();
        for &x in &output {
            assert!((x).abs() < 1e-7, "all-masked output should be 0");
        }
    }

    #[test]
    fn test_masked_softmax_none_masked() {
        let input = [1.0, 2.0, 3.0];
        let mask = [true, true, true];
        let mut out_masked = [0.0; 3];
        softmax_with_mask(&input, &mut out_masked, &mask).unwrap();
        let mut out_plain = [0.0; 3];
        softmax_f32(&input, &mut out_plain).unwrap();
        for (a, b) in out_masked.iter().zip(out_plain.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_masked_softmax_single_unmasked() {
        let input = [1.0, 2.0, 3.0];
        let mask = [false, true, false];
        let mut output = [0.0; 3];
        softmax_with_mask(&input, &mut output, &mask).unwrap();
        assert!((output[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_masked_softmax_length_mismatch() {
        let input = [1.0, 2.0];
        let mask = [true];
        let mut output = [0.0; 2];
        assert!(softmax_with_mask(&input, &mut output, &mask).is_err());
    }

    #[test]
    fn test_masked_softmax_empty() {
        let input: [f32; 0] = [];
        let mask: [bool; 0] = [];
        let mut output: Vec<f32> = vec![];
        softmax_with_mask(&input, &mut output, &mask).unwrap();
    }

    // ── Top-K softmax ───────────────────────────────────────────────────

    #[test]
    fn test_topk_basic() {
        let input = [1.0, 4.0, 2.0, 3.0, 5.0];
        let mut output = [0.0; 5];
        softmax_topk(&input, &mut output, 2).unwrap();

        // Only 2 non-zero values.
        let nonzero: Vec<_> = output.iter().filter(|&&x| x > 1e-9).collect();
        assert_eq!(nonzero.len(), 2);

        // The top-2 indices should be 1 (value 4) and 4 (value 5).
        assert!(output[4] > 1e-9);
        assert!(output[1] > 1e-9);

        // Non-zero values sum to ~1.
        let s: f32 = output.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_topk_k_equals_n() {
        let input = [1.0, 2.0, 3.0];
        let mut out_topk = [0.0; 3];
        softmax_topk(&input, &mut out_topk, 3).unwrap();
        let mut out_plain = [0.0; 3];
        softmax_f32(&input, &mut out_plain).unwrap();
        for (a, b) in out_topk.iter().zip(out_plain.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_topk_k_greater_than_n() {
        let input = [1.0, 2.0];
        let mut out_topk = [0.0; 2];
        softmax_topk(&input, &mut out_topk, 10).unwrap();
        assert_sums_to_one(&out_topk, 1e-6);
    }

    #[test]
    fn test_topk_k_is_one() {
        let input = [1.0, 5.0, 3.0];
        let mut output = [0.0; 3];
        softmax_topk(&input, &mut output, 1).unwrap();
        assert!((output[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_topk_k_zero_error() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 2];
        assert!(softmax_topk(&input, &mut output, 0).is_err());
    }

    #[test]
    fn test_topk_length_mismatch() {
        let input = [1.0];
        let mut output = [0.0; 2];
        assert!(softmax_topk(&input, &mut output, 1).is_err());
    }

    #[test]
    fn test_topk_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        softmax_topk(&input, &mut output, 1).unwrap();
    }

    // ── Online softmax ──────────────────────────────────────────────────

    #[test]
    fn test_online_matches_standard() {
        let input = [0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 3.0, -2.0];
        let mut out_std = vec![0.0; input.len()];
        softmax_f32(&input, &mut out_std).unwrap();
        let mut out_online = vec![0.0; input.len()];
        softmax_online(&input, &mut out_online).unwrap();
        for (a, b) in out_std.iter().zip(out_online.iter()) {
            assert!((a - b).abs() < 1e-5, "standard vs online mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_online_sums_to_one() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0; 5];
        softmax_online(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
    }

    #[test]
    fn test_online_stability_large() {
        let input = [1000.0, 1001.0, 1002.0];
        let mut output = [0.0; 3];
        softmax_online(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-4);
        for &x in &output {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn test_online_single() {
        let input = [42.0];
        let mut output = [0.0; 1];
        softmax_online(&input, &mut output).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_online_empty() {
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        softmax_online(&input, &mut output).unwrap();
    }

    #[test]
    fn test_online_length_mismatch() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 3];
        assert!(softmax_online(&input, &mut output).is_err());
    }

    // ── Batched softmax ─────────────────────────────────────────────────

    #[test]
    fn test_batched_basic() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0; 6];
        batched_softmax_opt(&input, &mut output, 2, 3).unwrap();

        // Each row should sum to ~1.
        assert_sums_to_one(&output[0..3], 1e-6);
        assert_sums_to_one(&output[3..6], 1e-6);
    }

    #[test]
    fn test_batched_matches_individual() {
        let row1 = [0.5, -1.0, 2.0];
        let row2 = [3.0, 1.0, -0.5];
        let input: Vec<f32> = row1.iter().chain(row2.iter()).copied().collect();
        let mut out_batched = [0.0; 6];
        batched_softmax_opt(&input, &mut out_batched, 2, 3).unwrap();

        let mut out_r1 = [0.0; 3];
        softmax_f32(&row1, &mut out_r1).unwrap();
        let mut out_r2 = [0.0; 3];
        softmax_f32(&row2, &mut out_r2).unwrap();

        for (a, b) in out_batched[0..3].iter().zip(out_r1.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
        for (a, b) in out_batched[3..6].iter().zip(out_r2.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_batched_single_row() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 3];
        batched_softmax_opt(&input, &mut output, 1, 3).unwrap();
        assert_sums_to_one(&output, 1e-6);
    }

    #[test]
    fn test_batched_dim_mismatch_input() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 3];
        assert!(batched_softmax_opt(&input, &mut output, 2, 3).is_err());
    }

    #[test]
    fn test_batched_dim_mismatch_output() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0; 3];
        assert!(batched_softmax_opt(&input, &mut output, 2, 3).is_err());
    }

    // ── SIMD vs scalar equivalence ──────────────────────────────────────

    #[test]
    fn test_avx2_vs_scalar_small() {
        let input = [0.1, -0.2, 0.3, -0.4, 0.5];
        let mut out_api = [0.0; 5];
        softmax_f32(&input, &mut out_api).unwrap();
        let mut out_scalar = [0.0; 5];
        softmax_scalar(&input, &mut out_scalar);
        for (a, b) in out_api.iter().zip(out_scalar.iter()) {
            assert!((a - b).abs() < 1e-6, "API vs scalar mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_avx2_vs_scalar_exact_8() {
        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.5 - 2.0).collect();
        let mut out_api = [0.0; 8];
        softmax_f32(&input, &mut out_api).unwrap();
        let mut out_scalar = [0.0; 8];
        softmax_scalar(&input, &mut out_scalar);
        for (a, b) in out_api.iter().zip(out_scalar.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_avx2_vs_scalar_16() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.3).collect();
        let mut out_api = [0.0; 16];
        softmax_f32(&input, &mut out_api).unwrap();
        let mut out_scalar = [0.0; 16];
        softmax_scalar(&input, &mut out_scalar);
        for (a, b) in out_api.iter().zip(out_scalar.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_avx2_vs_scalar_17_non_aligned() {
        let input: Vec<f32> = (0..17).map(|i| i as f32 * 0.1).collect();
        let mut out_api = [0.0; 17];
        softmax_f32(&input, &mut out_api).unwrap();
        let mut out_scalar = [0.0; 17];
        softmax_scalar(&input, &mut out_scalar);
        for (a, b) in out_api.iter().zip(out_scalar.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_avx2_vs_scalar_large() {
        let input: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.01).sin()).collect();
        let mut out_api = [0.0; 1024];
        softmax_f32(&input, &mut out_api).unwrap();
        let mut out_scalar = [0.0; 1024];
        softmax_scalar(&input, &mut out_scalar);
        for (i, (a, b)) in out_api.iter().zip(out_scalar.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "mismatch at [{i}]: API={a} scalar={b}");
        }
    }

    #[test]
    fn test_avx2_vs_scalar_size_7() {
        let input = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0];
        let mut out_api = [0.0; 7];
        softmax_f32(&input, &mut out_api).unwrap();
        let mut out_scalar = [0.0; 7];
        softmax_scalar(&input, &mut out_scalar);
        for (a, b) in out_api.iter().zip(out_scalar.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_avx2_vs_scalar_size_9() {
        let input: Vec<f32> = (0..9).map(|i| i as f32 - 4.0).collect();
        let mut out_api = [0.0; 9];
        softmax_f32(&input, &mut out_api).unwrap();
        let mut out_scalar = [0.0; 9];
        softmax_scalar(&input, &mut out_scalar);
        for (a, b) in out_api.iter().zip(out_scalar.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_softmax_all_zeros() {
        let input = [0.0; 4];
        let mut output = [0.0; 4];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-6);
        for &x in &output {
            assert!((x - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_very_negative() {
        let input = [-100.0, -200.0, -300.0];
        let mut output = [0.0; 3];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
        // The least negative should dominate.
        assert!(output[0] > 0.99, "least negative should dominate");
    }

    #[test]
    fn test_softmax_nan_input() {
        let input = [1.0, f32::NAN, 3.0];
        let mut output = [0.0; 3];
        // We don't guarantee specific behavior but it shouldn't panic.
        let _ = softmax_f32(&input, &mut output);
    }

    #[test]
    fn test_softmax_inf_input() {
        let input = [1.0, f32::INFINITY, 3.0];
        let mut output = [0.0; 3];
        let _ = softmax_f32(&input, &mut output);
        // Should not panic.
    }

    #[test]
    fn test_softmax_neg_inf_input() {
        let input = [1.0, f32::NEG_INFINITY, 3.0];
        let mut output = [0.0; 3];
        softmax_f32(&input, &mut output).unwrap();
        assert!((output[1]).abs() < 1e-7, "NEG_INFINITY position should be ~0");
    }

    #[test]
    fn test_softmax_monotonicity() {
        // For strictly increasing input, output should be strictly increasing.
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut output = [0.0; 10];
        softmax_f32(&input, &mut output).unwrap();
        for i in 1..10 {
            assert!(output[i] > output[i - 1], "monotonicity violated at {i}");
        }
    }

    #[test]
    fn test_softmax_two_elements() {
        let input = [0.0, 0.0];
        let mut output = [0.0; 2];
        softmax_f32(&input, &mut output).unwrap();
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_symmetry() {
        let input = [1.0, 2.0, 1.0];
        let mut output = [0.0; 3];
        softmax_f32(&input, &mut output).unwrap();
        assert!(
            (output[0] - output[2]).abs() < 1e-7,
            "symmetric inputs should give symmetric outputs"
        );
    }

    // ── Additional temperature tests ────────────────────────────────────

    #[test]
    fn test_temperature_sums_to_one() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        for &temp in &[0.1, 0.5, 1.0, 2.0, 10.0] {
            let mut output = [0.0; 5];
            softmax_with_temperature(&input, &mut output, temp).unwrap();
            assert_sums_to_one(&output, 1e-5);
        }
    }

    #[test]
    fn test_temperature_preserves_order() {
        let input = [1.0, 3.0, 2.0];
        let mut output = [0.0; 3];
        softmax_with_temperature(&input, &mut output, 0.5).unwrap();
        assert!(output[1] > output[2]);
        assert!(output[2] > output[0]);
    }

    // ── Additional masked tests ─────────────────────────────────────────

    #[test]
    fn test_masked_preserves_relative_order() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [true, false, true, true];
        let mut output = [0.0; 4];
        softmax_with_mask(&input, &mut output, &mask).unwrap();
        assert!(output[3] > output[2]);
        assert!(output[2] > output[0]);
    }

    // ── Additional online tests ─────────────────────────────────────────

    #[test]
    fn test_online_matches_standard_large() {
        let input: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.1).sin()).collect();
        let mut out_std = [0.0; 256];
        softmax_f32(&input, &mut out_std).unwrap();
        let mut out_online = [0.0; 256];
        softmax_online(&input, &mut out_online).unwrap();
        for (i, (a, b)) in out_std.iter().zip(out_online.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "mismatch at [{i}]: std={a} online={b}");
        }
    }

    #[test]
    fn test_online_all_same() {
        let input = [3.0; 16];
        let mut output = [0.0; 16];
        softmax_online(&input, &mut output).unwrap();
        let expected = 1.0 / 16.0;
        for &x in &output {
            assert!((x - expected).abs() < 1e-5);
        }
    }

    // ── Additional batched tests ────────────────────────────────────────

    #[test]
    fn test_batched_many_rows() {
        let batch = 10;
        let seq = 32;
        let input: Vec<f32> = (0..(batch * seq)).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut output = vec![0.0; batch * seq];
        batched_softmax_opt(&input, &mut output, batch, seq).unwrap();
        for bi in 0..batch {
            let row = &output[bi * seq..(bi + 1) * seq];
            assert_sums_to_one(row, 1e-5);
            assert_non_negative(row);
        }
    }

    #[test]
    fn test_batched_single_element_rows() {
        let input = [5.0, 10.0, -3.0];
        let mut output = [0.0; 3];
        batched_softmax_opt(&input, &mut output, 3, 1).unwrap();
        for &x in &output {
            assert!((x - 1.0).abs() < 1e-7);
        }
    }

    // ── Cross-variant consistency ───────────────────────────────────────

    #[test]
    fn test_all_variants_agree_on_basic_input() {
        let input = [0.5, -1.0, 2.0, 0.0, 1.5];
        let n = input.len();

        let mut out_std = vec![0.0; n];
        softmax_f32(&input, &mut out_std).unwrap();

        let mut out_inplace = input.to_vec();
        softmax_f32_inplace(&mut out_inplace).unwrap();

        let mut out_online = vec![0.0; n];
        softmax_online(&input, &mut out_online).unwrap();

        let mut out_temp = vec![0.0; n];
        softmax_with_temperature(&input, &mut out_temp, 1.0).unwrap();

        let mut out_batch = vec![0.0; n];
        batched_softmax_opt(&input, &mut out_batch, 1, n).unwrap();

        let mask = vec![true; n];
        let mut out_mask = vec![0.0; n];
        softmax_with_mask(&input, &mut out_mask, &mask).unwrap();

        let mut out_topk = vec![0.0; n];
        softmax_topk(&input, &mut out_topk, n).unwrap();

        for i in 0..n {
            let ref_val = out_std[i];
            let tol = 1e-5;
            assert!((out_inplace[i] - ref_val).abs() < tol, "inplace mismatch at {i}");
            assert!((out_online[i] - ref_val).abs() < tol, "online mismatch at {i}");
            assert!((out_temp[i] - ref_val).abs() < tol, "temp=1 mismatch at {i}");
            assert!((out_batch[i] - ref_val).abs() < tol, "batch mismatch at {i}");
            assert!((out_mask[i] - ref_val).abs() < tol, "mask mismatch at {i}");
            assert!((out_topk[i] - ref_val).abs() < tol, "topk mismatch at {i}");
        }
    }

    #[test]
    fn test_log_softmax_vs_manual() {
        let input = [2.0, 1.0, 0.1];
        let mut sm = [0.0; 3];
        softmax_f32(&input, &mut sm).unwrap();
        let manual_log: Vec<f32> = sm.iter().map(|&x| x.ln()).collect();

        let mut lsm = [0.0; 3];
        log_softmax_f32(&input, &mut lsm).unwrap();

        for (a, b) in manual_log.iter().zip(lsm.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // ── Stress / larger sizes ───────────────────────────────────────────

    #[test]
    fn test_softmax_size_33() {
        let input: Vec<f32> = (0..33).map(|i| i as f32 * 0.2 - 3.0).collect();
        let mut output = [0.0; 33];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
    }

    #[test]
    fn test_softmax_size_63() {
        let input: Vec<f32> = (0..63).map(|i| (i as f32).sin()).collect();
        let mut output = [0.0; 63];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
    }

    #[test]
    fn test_softmax_size_64() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32).cos()).collect();
        let mut output = [0.0; 64];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
    }

    #[test]
    fn test_softmax_size_65() {
        let input: Vec<f32> = (0..65).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut output = [0.0; 65];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
    }

    #[test]
    fn test_softmax_size_128() {
        let input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.05 - 3.0).collect();
        let mut output = [0.0; 128];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
    }

    #[test]
    fn test_softmax_size_255() {
        let input: Vec<f32> = (0..255).map(|i| (i as f32 * 0.02).sin()).collect();
        let mut output = [0.0; 255];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
    }

    #[test]
    fn test_softmax_size_256() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.02).cos()).collect();
        let mut output = [0.0; 256];
        softmax_f32(&input, &mut output).unwrap();
        assert_sums_to_one(&output, 1e-5);
    }

    // ── NEON parity ─────────────────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn softmax_neon_matches_scalar() {
        for &size in &[1, 3, 4, 7, 8, 15, 16, 31, 32, 64, 128, 255, 256] {
            let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.3 - 5.0).collect();

            // Scalar reference
            let mut expected = vec![0.0f32; size];
            softmax_scalar(&input, &mut expected);

            // NEON path
            let mut got = vec![0.0f32; size];
            unsafe { softmax_neon(&input, &mut got) };

            for i in 0..size {
                assert!(
                    (got[i] - expected[i]).abs() < 1e-6,
                    "softmax_neon mismatch at index {i} for size {size}: got {} expected {}",
                    got[i],
                    expected[i],
                );
            }

            // In-place variant
            let mut inplace = input.clone();
            unsafe { softmax_neon_inplace(&mut inplace) };

            for i in 0..size {
                assert!(
                    (inplace[i] - expected[i]).abs() < 1e-6,
                    "softmax_neon_inplace mismatch at index {i} for size {size}: got {} expected {}",
                    inplace[i],
                    expected[i],
                );
            }
        }
    }
}
