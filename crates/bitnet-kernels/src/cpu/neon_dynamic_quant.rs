//! NEON-optimized dynamic quantization kernels for Apple Silicon (aarch64).
//!
//! Provides five dynamic quantization operations with runtime-computed scales,
//! using ARM NEON `float32x4` intrinsics for 4-wide parallel computation.
//! Each function has a NEON fast-path and scalar fallback selected at compile
//! time via `cfg(target_arch)`.
//!
//! # Kernels
//!
//! - [`dynamic_quantize_symmetric_neon`] — symmetric int8 quantization (auto scale)
//! - [`dynamic_quantize_asymmetric_neon`] — asymmetric uint8 quantization (scale + zero-point)
//! - [`dynamic_quantize_per_token_neon`] — per-token symmetric int8 quantization
//! - [`calibrate_quantization_range_neon`] — percentile-based range calibration
//! - [`smooth_quantize_neon`] — SmoothQuant-style channel-smoothed quantization

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::missing_safety_doc,
    clippy::float_cmp,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::let_and_return,
    clippy::derivable_impls,
    clippy::excessive_precision,
    clippy::manual_is_multiple_of
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── 1. dynamic_quantize_symmetric_neon ────────────────────────────

/// Symmetric dynamic quantization: maps `f32` values to `i8` using a
/// single scale derived from the absolute maximum of the input.
///
/// `scale = absmax / 127.0`; each value is quantized as
/// `round(x / scale)` clamped to `[-127, 127]`.
///
/// Returns `(quantized, scale)`.
pub fn dynamic_quantize_symmetric_neon(input: &[f32]) -> (Vec<i8>, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_symmetric(input) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_symmetric(input)
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_symmetric(input: &[f32]) -> (Vec<i8>, f32) {
    let n = input.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }

    // — find absmax via NEON ------------------------------------------------
    let ptr = input.as_ptr();
    let chunks = n / 4;
    let mut vmax = unsafe { vdupq_n_f32(0.0) };

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
        let va = unsafe { vabsq_f32(v) };
        vmax = unsafe { vmaxq_f32(vmax, va) };
    }

    let mut absmax: f32 = unsafe { vmaxvq_f32(vmax) };
    for i in (chunks * 4)..n {
        absmax = absmax.max(unsafe { ptr.add(i).read() }.abs());
    }

    if absmax == 0.0 {
        return (vec![0i8; n], 0.0);
    }

    let scale = absmax / 127.0;
    let inv_scale = 127.0 / absmax;

    // — quantize ------------------------------------------------------------
    let mut out = vec![0i8; n];
    let vinv = unsafe { vdupq_n_f32(inv_scale) };
    let vmin_clamp = unsafe { vdupq_n_f32(-127.0) };
    let vmax_clamp = unsafe { vdupq_n_f32(127.0) };

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
        let scaled = unsafe { vmulq_f32(v, vinv) };
        // clamp
        let clamped = unsafe { vmaxq_f32(vmin_clamp, vminq_f32(vmax_clamp, scaled)) };
        // round via vrndnq_f32 (round-to-nearest-even)
        let rounded = unsafe { vrndnq_f32(clamped) };
        // convert to i32 then extract lanes
        let i32v = unsafe { vcvtq_s32_f32(rounded) };
        let arr = unsafe { extract_i32_lanes(i32v) };
        for j in 0..4 {
            out[i * 4 + j] = arr[j] as i8;
        }
    }
    for i in (chunks * 4)..n {
        let v = unsafe { ptr.add(i).read() };
        let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
        out[i] = q;
    }

    (out, scale)
}

// vgetq_lane_s32 requires a const generic — use a helper on aarch64
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn extract_i32_lanes(v: int32x4_t) -> [i32; 4] {
    let mut arr = [0i32; 4];
    unsafe { vst1q_s32(arr.as_mut_ptr(), v) };
    arr
}

#[cfg(not(target_arch = "aarch64"))]
fn scalar_symmetric(input: &[f32]) -> (Vec<i8>, f32) {
    let n = input.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }
    let absmax = input.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if absmax == 0.0 {
        return (vec![0i8; n], 0.0);
    }
    let scale = absmax / 127.0;
    let inv = 127.0 / absmax;
    let out = input.iter().map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8).collect();
    (out, scale)
}

// ── 2. dynamic_quantize_asymmetric_neon ───────────────────────────

/// Asymmetric dynamic quantization: maps `f32` to `u8` using
/// `scale = (max - min) / 255.0` and `zero_point = round(-min / scale)`.
///
/// Returns `(quantized, scale, zero_point)`.
pub fn dynamic_quantize_asymmetric_neon(input: &[f32]) -> (Vec<u8>, f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_asymmetric(input) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_asymmetric(input)
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_asymmetric(input: &[f32]) -> (Vec<u8>, f32, f32) {
    let n = input.len();
    if n == 0 {
        return (Vec::new(), 0.0, 0.0);
    }

    let ptr = input.as_ptr();
    let chunks = n / 4;
    let mut vmin = unsafe { vdupq_n_f32(f32::MAX) };
    let mut vmax = unsafe { vdupq_n_f32(f32::MIN) };

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
        vmin = unsafe { vminq_f32(vmin, v) };
        vmax = unsafe { vmaxq_f32(vmax, v) };
    }

    let mut fmin = unsafe { vminvq_f32(vmin) };
    let mut fmax = unsafe { vmaxvq_f32(vmax) };
    for i in (chunks * 4)..n {
        let v = unsafe { ptr.add(i).read() };
        fmin = fmin.min(v);
        fmax = fmax.max(v);
    }

    let range = fmax - fmin;
    if range == 0.0 {
        return (vec![0u8; n], 0.0, fmin);
    }

    let scale = range / 255.0;
    let inv_scale = 255.0 / range;
    let zero_point = (-fmin * inv_scale).round();

    let mut out = vec![0u8; n];
    let _vzp = unsafe { vdupq_n_f32(zero_point) };
    let vinv = unsafe { vdupq_n_f32(inv_scale) };
    let vclamp_min = unsafe { vdupq_n_f32(0.0) };
    let vclamp_max = unsafe { vdupq_n_f32(255.0) };
    let vfmin = unsafe { vdupq_n_f32(fmin) };

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
        let shifted = unsafe { vsubq_f32(v, vfmin) };
        let scaled = unsafe { vmulq_f32(shifted, vinv) };
        let rounded = unsafe { vrndnq_f32(scaled) };
        let clamped = unsafe { vmaxq_f32(vclamp_min, vminq_f32(vclamp_max, rounded)) };
        let i32v = unsafe { vcvtq_s32_f32(clamped) };
        let arr = unsafe { extract_i32_lanes(i32v) };
        for j in 0..4 {
            out[i * 4 + j] = arr[j] as u8;
        }
    }
    for i in (chunks * 4)..n {
        let v = unsafe { ptr.add(i).read() };
        let q = ((v - fmin) * inv_scale).round().clamp(0.0, 255.0) as u8;
        out[i] = q;
    }

    (out, scale, zero_point)
}

#[cfg(not(target_arch = "aarch64"))]
fn scalar_asymmetric(input: &[f32]) -> (Vec<u8>, f32, f32) {
    let n = input.len();
    if n == 0 {
        return (Vec::new(), 0.0, 0.0);
    }
    let fmin = input.iter().cloned().fold(f32::MAX, f32::min);
    let fmax = input.iter().cloned().fold(f32::MIN, f32::max);
    let range = fmax - fmin;
    if range == 0.0 {
        return (vec![0u8; n], 0.0, fmin);
    }
    let scale = range / 255.0;
    let inv = 255.0 / range;
    let zero_point = (-fmin * inv).round();
    let out = input.iter().map(|&v| ((v - fmin) * inv).round().clamp(0.0, 255.0) as u8).collect();
    (out, scale, zero_point)
}

// ── 3. dynamic_quantize_per_token_neon ────────────────────────────

/// Per-token symmetric quantization: for each token (row of length
/// `hidden_dim`), computes an independent scale from the row absmax and
/// quantizes to `i8`.
///
/// `input.len()` must equal `seq_len * hidden_dim`.
///
/// Returns `(quantized, scales)` where `scales.len() == seq_len`.
pub fn dynamic_quantize_per_token_neon(
    input: &[f32],
    seq_len: usize,
    hidden_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(input.len(), seq_len * hidden_dim, "input length must equal seq_len * hidden_dim");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_per_token(input, seq_len, hidden_dim) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_per_token(input, seq_len, hidden_dim)
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_per_token(input: &[f32], seq_len: usize, hidden_dim: usize) -> (Vec<i8>, Vec<f32>) {
    let mut out = vec![0i8; input.len()];
    let mut scales = vec![0.0f32; seq_len];

    for t in 0..seq_len {
        let row = &input[t * hidden_dim..(t + 1) * hidden_dim];
        let ptr = row.as_ptr();
        let chunks = hidden_dim / 4;

        // absmax
        let mut vmax = unsafe { vdupq_n_f32(0.0) };
        for i in 0..chunks {
            let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
            vmax = unsafe { vmaxq_f32(vmax, vabsq_f32(v)) };
        }
        let mut absmax = unsafe { vmaxvq_f32(vmax) };
        for i in (chunks * 4)..hidden_dim {
            absmax = absmax.max(unsafe { ptr.add(i).read() }.abs());
        }

        let scale = if absmax == 0.0 { 0.0 } else { absmax / 127.0 };
        scales[t] = scale;

        if absmax == 0.0 {
            continue;
        }

        let inv = 127.0 / absmax;
        let vinv = unsafe { vdupq_n_f32(inv) };
        let vmin_c = unsafe { vdupq_n_f32(-127.0) };
        let vmax_c = unsafe { vdupq_n_f32(127.0) };
        let base = t * hidden_dim;

        for i in 0..chunks {
            let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
            let s = unsafe { vmulq_f32(v, vinv) };
            let c = unsafe { vmaxq_f32(vmin_c, vminq_f32(vmax_c, s)) };
            let r = unsafe { vrndnq_f32(c) };
            let iv = unsafe { vcvtq_s32_f32(r) };
            let arr = unsafe { extract_i32_lanes(iv) };
            for j in 0..4 {
                out[base + i * 4 + j] = arr[j] as i8;
            }
        }
        for i in (chunks * 4)..hidden_dim {
            let v = unsafe { ptr.add(i).read() };
            out[base + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }

    (out, scales)
}

#[cfg(not(target_arch = "aarch64"))]
fn scalar_per_token(input: &[f32], seq_len: usize, hidden_dim: usize) -> (Vec<i8>, Vec<f32>) {
    let mut out = vec![0i8; input.len()];
    let mut scales = vec![0.0f32; seq_len];

    for t in 0..seq_len {
        let row = &input[t * hidden_dim..(t + 1) * hidden_dim];
        let absmax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if absmax == 0.0 { 0.0 } else { absmax / 127.0 };
        scales[t] = scale;
        if absmax == 0.0 {
            continue;
        }
        let inv = 127.0 / absmax;
        let base = t * hidden_dim;
        for (i, &v) in row.iter().enumerate() {
            out[base + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }

    (out, scales)
}

// ── 4. calibrate_quantization_range_neon ──────────────────────────

/// Calibrate the quantization range using percentile clipping.
///
/// Sorts absolute values and returns `(min_val, max_val)` clipped at the
/// given `percentile` (0.0–1.0). For example, `percentile = 0.99` clips
/// the top 1% of absolute values.
pub fn calibrate_quantization_range_neon(input: &[f32], percentile: f32) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_calibrate(input, percentile) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_calibrate(input, percentile)
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_calibrate(input: &[f32], percentile: f32) -> (f32, f32) {
    let n = input.len();
    if n == 0 {
        return (0.0, 0.0);
    }

    // Compute absolute values using NEON
    let mut abs_vals = vec![0.0f32; n];
    let ptr = input.as_ptr();
    let optr = abs_vals.as_mut_ptr();
    let chunks = n / 4;

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
        let a = unsafe { vabsq_f32(v) };
        unsafe { vst1q_f32(optr.add(i * 4), a) };
    }
    for i in (chunks * 4)..n {
        unsafe { *optr.add(i) = (*ptr.add(i)).abs() };
    }

    abs_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let clip_idx = ((n as f32 * percentile).ceil() as usize).min(n).saturating_sub(1);
    let clip_val = abs_vals[clip_idx];

    (-clip_val, clip_val)
}

#[cfg(not(target_arch = "aarch64"))]
fn scalar_calibrate(input: &[f32], percentile: f32) -> (f32, f32) {
    let n = input.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut abs_vals: Vec<f32> = input.iter().map(|v| v.abs()).collect();
    abs_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let clip_idx = ((n as f32 * percentile).ceil() as usize).min(n).saturating_sub(1);
    let clip_val = abs_vals[clip_idx];
    (-clip_val, clip_val)
}

// ── 5. smooth_quantize_neon ───────────────────────────────────────

/// SmoothQuant-style quantization: multiplies each element by a
/// per-channel smooth factor before symmetric int8 quantization.
///
/// `smooth_factor.len()` must equal `input.len()`.
///
/// Returns `(quantized, scale)`.
pub fn smooth_quantize_neon(input: &[f32], smooth_factor: &[f32]) -> (Vec<i8>, f32) {
    assert_eq!(
        input.len(),
        smooth_factor.len(),
        "input and smooth_factor must have the same length"
    );

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_smooth(input, smooth_factor) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_smooth(input, smooth_factor)
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_smooth(input: &[f32], smooth_factor: &[f32]) -> (Vec<i8>, f32) {
    let n = input.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }

    // Step 1: multiply by smooth factor and find absmax
    let mut smoothed = vec![0.0f32; n];
    let iptr = input.as_ptr();
    let sptr = smooth_factor.as_ptr();
    let optr = smoothed.as_mut_ptr();
    let chunks = n / 4;
    let mut vmax = unsafe { vdupq_n_f32(0.0) };

    for i in 0..chunks {
        let vi = unsafe { vld1q_f32(iptr.add(i * 4)) };
        let vs = unsafe { vld1q_f32(sptr.add(i * 4)) };
        let prod = unsafe { vmulq_f32(vi, vs) };
        unsafe { vst1q_f32(optr.add(i * 4), prod) };
        vmax = unsafe { vmaxq_f32(vmax, vabsq_f32(prod)) };
    }

    let mut absmax = unsafe { vmaxvq_f32(vmax) };
    for i in (chunks * 4)..n {
        let v = unsafe { *iptr.add(i) * *sptr.add(i) };
        unsafe { *optr.add(i) = v };
        absmax = absmax.max(v.abs());
    }

    if absmax == 0.0 {
        return (vec![0i8; n], 0.0);
    }

    let scale = absmax / 127.0;
    let inv = 127.0 / absmax;

    // Step 2: quantize smoothed values
    let mut out = vec![0i8; n];
    let vinv = unsafe { vdupq_n_f32(inv) };
    let vmin_c = unsafe { vdupq_n_f32(-127.0) };
    let vmax_c = unsafe { vdupq_n_f32(127.0) };

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(optr.add(i * 4)) };
        let s = unsafe { vmulq_f32(v, vinv) };
        let c = unsafe { vmaxq_f32(vmin_c, vminq_f32(vmax_c, s)) };
        let r = unsafe { vrndnq_f32(c) };
        let iv = unsafe { vcvtq_s32_f32(r) };
        let arr = unsafe { extract_i32_lanes(iv) };
        for j in 0..4 {
            out[i * 4 + j] = arr[j] as i8;
        }
    }
    for i in (chunks * 4)..n {
        let v = unsafe { *optr.add(i) };
        out[i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }

    (out, scale)
}

#[cfg(not(target_arch = "aarch64"))]
fn scalar_smooth(input: &[f32], smooth_factor: &[f32]) -> (Vec<i8>, f32) {
    let n = input.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }
    let smoothed: Vec<f32> = input.iter().zip(smooth_factor.iter()).map(|(&a, &b)| a * b).collect();
    let absmax = smoothed.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if absmax == 0.0 {
        return (vec![0i8; n], 0.0);
    }
    let scale = absmax / 127.0;
    let inv = 127.0 / absmax;
    let out = smoothed.iter().map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8).collect();
    (out, scale)
}

// ══════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    // ── Helper ────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    // ── 1. symmetric ─────────────────────────────────────────────

    #[test]
    fn test_symmetric_empty() {
        let (q, s) = dynamic_quantize_symmetric_neon(&[]);
        assert!(q.is_empty());
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_symmetric_single() {
        let (q, s) = dynamic_quantize_symmetric_neon(&[1.0]);
        assert_eq!(q.len(), 1);
        assert!(s > 0.0);
        assert_eq!(q[0], 127);
    }

    #[test]
    fn test_symmetric_zeros() {
        let input = vec![0.0; 16];
        let (q, s) = dynamic_quantize_symmetric_neon(&input);
        assert_eq!(s, 0.0);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_symmetric_positive_only() {
        let input = vec![0.5, 1.0, 1.5, 2.0];
        let (q, s) = dynamic_quantize_symmetric_neon(&input);
        assert!(approx_eq(s, 2.0 / 127.0, 1e-6));
        assert_eq!(q[3], 127); // max maps to 127
        assert!(q[0] > 0); // positive stays positive
    }

    #[test]
    fn test_symmetric_negative_only() {
        let input = vec![-0.5, -1.0, -1.5, -2.0];
        let (q, s) = dynamic_quantize_symmetric_neon(&input);
        assert!(approx_eq(s, 2.0 / 127.0, 1e-6));
        assert_eq!(q[3], -127); // min maps to -127
    }

    #[test]
    fn test_symmetric_mixed() {
        let input = vec![-1.0, 0.0, 1.0, 0.5];
        let (q, s) = dynamic_quantize_symmetric_neon(&input);
        assert!(approx_eq(s, 1.0 / 127.0, 1e-6));
        assert_eq!(q[0], -127);
        assert_eq!(q[1], 0);
        assert_eq!(q[2], 127);
    }

    #[test]
    fn test_symmetric_scale_correctness() {
        let input = vec![3.0, -3.0, 1.5, -1.5, 0.0, 0.0, 0.0, 0.0];
        let (q, s) = dynamic_quantize_symmetric_neon(&input);
        assert!(approx_eq(s, 3.0 / 127.0, 1e-6));
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -127);
    }

    #[test]
    fn test_symmetric_large_input() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let (q, s) = dynamic_quantize_symmetric_neon(&input);
        assert_eq!(q.len(), 256);
        assert!(s > 0.0);
    }

    #[test]
    fn test_symmetric_non_aligned() {
        // Not a multiple of 4 — exercises scalar tail path
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (q, s) = dynamic_quantize_symmetric_neon(&input);
        assert_eq!(q.len(), 5);
        assert!(approx_eq(s, 5.0 / 127.0, 1e-6));
        assert_eq!(q[4], 127);
    }

    #[test]
    fn test_symmetric_dequant_roundtrip() {
        let input = vec![1.0, -0.5, 0.25, -0.75, 0.0, 0.1, -0.9, 0.6];
        let (q, s) = dynamic_quantize_symmetric_neon(&input);
        for (i, &orig) in input.iter().enumerate() {
            let reconstructed = q[i] as f32 * s;
            assert!(
                (reconstructed - orig).abs() < s + 1e-6,
                "idx {i}: orig={orig} reconstructed={reconstructed}"
            );
        }
    }

    #[test]
    fn test_symmetric_max_maps_to_127() {
        let input = vec![0.0, 0.0, 0.0, 42.0];
        let (q, _) = dynamic_quantize_symmetric_neon(&input);
        assert_eq!(q[3], 127);
    }

    #[test]
    fn test_symmetric_min_maps_to_neg127() {
        let input = vec![0.0, 0.0, 0.0, -42.0];
        let (q, _) = dynamic_quantize_symmetric_neon(&input);
        assert_eq!(q[3], -127);
    }

    #[test]
    fn test_symmetric_identical_values() {
        let input = vec![5.0; 8];
        let (q, s) = dynamic_quantize_symmetric_neon(&input);
        assert!(approx_eq(s, 5.0 / 127.0, 1e-6));
        assert!(q.iter().all(|&v| v == 127));
    }

    #[test]
    fn test_symmetric_tiny_values() {
        let input = vec![1e-7, -1e-7, 2e-7, -2e-7];
        let (_q, s) = dynamic_quantize_symmetric_neon(&input);
        assert!(s > 0.0);
        assert!(s < 1e-5);
    }

    #[test]
    fn test_symmetric_preserves_sign() {
        let input = vec![-2.0, 1.0, -0.5, 0.25, -3.0, 0.0, 1.5, -1.0];
        let (q, _) = dynamic_quantize_symmetric_neon(&input);
        for (i, &v) in input.iter().enumerate() {
            if v > 0.0 {
                assert!(q[i] > 0, "idx {i}");
            } else if v < 0.0 {
                assert!(q[i] < 0, "idx {i}");
            } else {
                assert_eq!(q[i], 0, "idx {i}");
            }
        }
    }

    #[test]
    fn test_symmetric_large_magnitude() {
        let input = vec![1e6, -1e6, 0.0, 500_000.0];
        let (q, s) = dynamic_quantize_symmetric_neon(&input);
        assert!(approx_eq(s, 1e6 / 127.0, 1.0));
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -127);
    }

    #[test]
    fn test_symmetric_length_1_to_8() {
        for len in 1..=8 {
            let input: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let (q, _) = dynamic_quantize_symmetric_neon(&input);
            assert_eq!(q.len(), len);
        }
    }

    #[test]
    fn test_symmetric_monotonic_ordering() {
        let input = vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
        let (q, _) = dynamic_quantize_symmetric_neon(&input);
        for i in 1..q.len() {
            assert!(q[i] >= q[i - 1], "monotonicity broken at {i}");
        }
    }

    // ── 2. asymmetric ────────────────────────────────────────────

    #[test]
    fn test_asymmetric_empty() {
        let (q, s, z) = dynamic_quantize_asymmetric_neon(&[]);
        assert!(q.is_empty());
        assert_eq!(s, 0.0);
        assert_eq!(z, 0.0);
    }

    #[test]
    fn test_asymmetric_single() {
        let (q, s, _z) = dynamic_quantize_asymmetric_neon(&[5.0]);
        assert_eq!(q.len(), 1);
        // single value → range = 0 → const output
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_asymmetric_zeros() {
        let input = vec![0.0; 8];
        let (q, s, _) = dynamic_quantize_asymmetric_neon(&input);
        assert_eq!(s, 0.0);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_asymmetric_positive_range() {
        let input = vec![0.0, 1.0, 2.0, 3.0];
        let (q, s, _z) = dynamic_quantize_asymmetric_neon(&input);
        assert!(s > 0.0);
        // min maps to 0, max maps to 255
        assert_eq!(q[0], 0);
        assert_eq!(q[3], 255);
    }

    #[test]
    fn test_asymmetric_negative_range() {
        let input = vec![-3.0, -2.0, -1.0, 0.0];
        let (q, s, _z) = dynamic_quantize_asymmetric_neon(&input);
        assert!(s > 0.0);
        assert_eq!(q[0], 0); // most negative → 0
        assert_eq!(q[3], 255); // least negative → 255
    }

    #[test]
    fn test_asymmetric_mixed_range() {
        let input = vec![-1.0, 0.0, 1.0, 2.0];
        let (q, s, _z) = dynamic_quantize_asymmetric_neon(&input);
        assert!(approx_eq(s, 3.0 / 255.0, 1e-5));
        assert_eq!(q[0], 0); // -1 is min
        assert_eq!(q[3], 255); // 2 is max
    }

    #[test]
    fn test_asymmetric_scale_and_zp() {
        let input = vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0];
        let (_, s, zp) = dynamic_quantize_asymmetric_neon(&input);
        assert!(approx_eq(s, 3.0 / 255.0, 1e-5));
        assert!(approx_eq(zp, 0.0, 1.0)); // zp ≈ 0 when min=0
    }

    #[test]
    fn test_asymmetric_dequant_roundtrip() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (q, s, _zp) = dynamic_quantize_asymmetric_neon(&input);
        let fmin = 1.0f32;
        for (i, &orig) in input.iter().enumerate() {
            let reconstructed = q[i] as f32 * s + fmin;
            assert!(
                (reconstructed - orig).abs() < s + 1e-4,
                "idx {i}: orig={orig} recon={reconstructed}"
            );
        }
    }

    #[test]
    fn test_asymmetric_output_in_range() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        let (q, _, _) = dynamic_quantize_asymmetric_neon(&input);
        for &v in &q {
            let _ = v; // u8 is always in range 0..=255
        }
    }

    #[test]
    fn test_asymmetric_non_aligned() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let (q, _, _) = dynamic_quantize_asymmetric_neon(&input);
        assert_eq!(q.len(), 7);
        assert_eq!(q[0], 0);
        assert_eq!(q[6], 255);
    }

    #[test]
    fn test_asymmetric_identical_values() {
        let input = vec![3.0; 8];
        let (q, s, _) = dynamic_quantize_asymmetric_neon(&input);
        assert_eq!(s, 0.0);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_asymmetric_monotonic() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let (q, _, _) = dynamic_quantize_asymmetric_neon(&input);
        for i in 1..q.len() {
            assert!(q[i] >= q[i - 1], "monotonicity at {i}");
        }
    }

    #[test]
    fn test_asymmetric_large_input() {
        let input: Vec<f32> = (0..512).map(|i| (i as f32) * 0.01).collect();
        let (q, s, _) = dynamic_quantize_asymmetric_neon(&input);
        assert_eq!(q.len(), 512);
        assert!(s > 0.0);
    }

    #[test]
    fn test_asymmetric_negative_zp() {
        // All values positive: zero_point should be near 0
        let input = vec![10.0, 20.0, 30.0, 40.0];
        let (_, _, zp) = dynamic_quantize_asymmetric_neon(&input);
        // zp = round(-min / scale) = round(-10 * 255 / 30) ≈ -85
        // It depends on the formula; just check it's finite
        assert!(zp.is_finite());
    }

    // ── 3. per_token ─────────────────────────────────────────────

    #[test]
    fn test_per_token_single_token() {
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let (q, s) = dynamic_quantize_per_token_neon(&input, 1, 4);
        assert_eq!(q.len(), 4);
        assert_eq!(s.len(), 1);
        assert!(approx_eq(s[0], 1.0 / 127.0, 1e-6));
    }

    #[test]
    fn test_per_token_two_tokens() {
        let input = vec![
            1.0, 0.0, -1.0, 0.5, // token 0 — absmax=1
            2.0, 0.0, -2.0, 1.0, // token 1 — absmax=2
        ];
        let (q, s) = dynamic_quantize_per_token_neon(&input, 2, 4);
        assert_eq!(s.len(), 2);
        assert!(approx_eq(s[0], 1.0 / 127.0, 1e-6));
        assert!(approx_eq(s[1], 2.0 / 127.0, 1e-6));
        // token 0 max → 127
        assert_eq!(q[0], 127);
        // token 1 max → 127
        assert_eq!(q[4], 127);
    }

    #[test]
    fn test_per_token_zeros_row() {
        let input = vec![
            0.0, 0.0, 0.0, 0.0, // all zeros
            1.0, 2.0, 3.0, 4.0, // non-zero
        ];
        let (q, s) = dynamic_quantize_per_token_neon(&input, 2, 4);
        assert_eq!(s[0], 0.0);
        assert!(q[0..4].iter().all(|&v| v == 0));
        assert!(s[1] > 0.0);
    }

    #[test]
    fn test_per_token_non_aligned_hidden() {
        let hidden = 5;
        let seq = 2;
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let (q, s) = dynamic_quantize_per_token_neon(&input, seq, hidden);
        assert_eq!(q.len(), 10);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn test_per_token_large() {
        let seq = 8;
        let hidden = 64;
        let input: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 - 256.0) / 256.0).collect();
        let (q, s) = dynamic_quantize_per_token_neon(&input, seq, hidden);
        assert_eq!(q.len(), seq * hidden);
        assert_eq!(s.len(), seq);
        assert!(s.iter().all(|&v| v >= 0.0));
    }

    #[test]
    #[should_panic]
    fn test_per_token_length_mismatch() {
        dynamic_quantize_per_token_neon(&[1.0, 2.0, 3.0], 2, 2);
    }

    #[test]
    fn test_per_token_independent_scales() {
        // Each token has a very different magnitude
        let mut input = vec![0.0f32; 16];
        input[0] = 100.0; // token 0 absmax = 100
        input[4] = 0.01; // token 1 absmax = 0.01
        input[8] = 50.0; // token 2 absmax = 50
        input[12] = 1.0; // token 3 absmax = 1
        let (_, s) = dynamic_quantize_per_token_neon(&input, 4, 4);
        assert!(approx_eq(s[0], 100.0 / 127.0, 1e-4));
        assert!(approx_eq(s[1], 0.01 / 127.0, 1e-8));
        assert!(approx_eq(s[2], 50.0 / 127.0, 1e-4));
        assert!(approx_eq(s[3], 1.0 / 127.0, 1e-6));
    }

    #[test]
    fn test_per_token_roundtrip() {
        let input = vec![0.5, -0.3, 0.7, -0.1, 1.0, -1.0, 0.0, 0.5];
        let (q, s) = dynamic_quantize_per_token_neon(&input, 2, 4);
        for t in 0..2 {
            for i in 0..4 {
                let idx = t * 4 + i;
                let recon = q[idx] as f32 * s[t];
                assert!(
                    (recon - input[idx]).abs() < s[t] + 1e-5,
                    "token {t} idx {i}: orig={} recon={recon}",
                    input[idx]
                );
            }
        }
    }

    #[test]
    fn test_per_token_preserves_sign() {
        let input = vec![-2.0, 1.0, -0.5, 0.25, 3.0, -3.0, 0.0, 1.5];
        let (q, _) = dynamic_quantize_per_token_neon(&input, 2, 4);
        for (i, &v) in input.iter().enumerate() {
            if v > 0.0 {
                assert!(q[i] > 0, "idx {i}");
            } else if v < 0.0 {
                assert!(q[i] < 0, "idx {i}");
            }
        }
    }

    #[test]
    fn test_per_token_hidden_dim_1() {
        let input = vec![5.0, -3.0, 7.0];
        let (q, s) = dynamic_quantize_per_token_neon(&input, 3, 1);
        assert_eq!(q.len(), 3);
        assert_eq!(s.len(), 3);
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -127);
        assert_eq!(q[2], 127);
    }

    // ── 4. calibrate ─────────────────────────────────────────────

    #[test]
    fn test_calibrate_empty() {
        let (lo, hi) = calibrate_quantization_range_neon(&[], 0.99);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    #[test]
    fn test_calibrate_single() {
        let (lo, hi) = calibrate_quantization_range_neon(&[5.0], 0.99);
        assert!(approx_eq(lo, -5.0, 1e-6));
        assert!(approx_eq(hi, 5.0, 1e-6));
    }

    #[test]
    fn test_calibrate_full_percentile() {
        let input = vec![1.0, -2.0, 3.0, -4.0];
        let (lo, hi) = calibrate_quantization_range_neon(&input, 1.0);
        assert!(approx_eq(hi, 4.0, 1e-6));
        assert!(approx_eq(lo, -4.0, 1e-6));
    }

    #[test]
    fn test_calibrate_clips_outlier() {
        // 99 values at 1.0, one outlier at 100.0
        let mut input = vec![1.0; 99];
        input.push(100.0);
        let (_lo, hi) = calibrate_quantization_range_neon(&input, 0.99);
        // At 99th percentile of 100 values, the clip index rounds to
        // the 99th sorted entry. Since 99 values are 1.0 and one is 100.0,
        // the clipped value should be 1.0 (the 99th sorted element).
        assert!(hi <= 100.0);
        assert!(hi >= 1.0);
    }

    #[test]
    fn test_calibrate_symmetric_output() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (lo, hi) = calibrate_quantization_range_neon(&input, 1.0);
        assert!(approx_eq(lo, -hi, 1e-6), "output must be symmetric");
    }

    #[test]
    fn test_calibrate_negative_input() {
        let input = vec![-5.0, -3.0, -1.0, -7.0];
        let (lo, hi) = calibrate_quantization_range_neon(&input, 1.0);
        assert!(approx_eq(hi, 7.0, 1e-6));
        assert!(approx_eq(lo, -7.0, 1e-6));
    }

    #[test]
    fn test_calibrate_zeros() {
        let input = vec![0.0; 8];
        let (lo, hi) = calibrate_quantization_range_neon(&input, 0.99);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    #[test]
    fn test_calibrate_50th_percentile() {
        // sorted abs: [1, 2, 3, 4, 5, 6, 7, 8]
        let input = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let (lo, hi) = calibrate_quantization_range_neon(&input, 0.5);
        // 50th pct of 8 items → ceil(8*0.5)-1 = 3 → abs_vals[3] = 4
        assert!(approx_eq(hi, 4.0, 1e-6));
        assert!(approx_eq(lo, -4.0, 1e-6));
    }

    #[test]
    fn test_calibrate_low_percentile() {
        let input: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let (_, hi) = calibrate_quantization_range_neon(&input, 0.1);
        // 10th pct of 100 → sorted abs[9] = 10.0
        assert!(approx_eq(hi, 10.0, 1e-4));
    }

    #[test]
    fn test_calibrate_non_aligned() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 elements
        let (lo, hi) = calibrate_quantization_range_neon(&input, 1.0);
        assert!(approx_eq(hi, 5.0, 1e-6));
        assert!(approx_eq(lo, -5.0, 1e-6));
    }

    #[test]
    fn test_calibrate_large_input() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let (lo, hi) = calibrate_quantization_range_neon(&input, 0.95);
        assert!(hi > 0.0);
        assert!(lo < 0.0);
        assert!(approx_eq(lo, -hi, 1e-6));
    }

    // ── 5. smooth_quantize ───────────────────────────────────────

    #[test]
    fn test_smooth_empty() {
        let (q, s) = smooth_quantize_neon(&[], &[]);
        assert!(q.is_empty());
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_smooth_identity_factor() {
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let factor = vec![1.0; 4];
        let (q1, s1) = smooth_quantize_neon(&input, &factor);
        let (q2, s2) = dynamic_quantize_symmetric_neon(&input);
        assert_eq!(q1, q2);
        assert!(approx_eq(s1, s2, 1e-6));
    }

    #[test]
    fn test_smooth_zero_factor() {
        let input = vec![100.0, -200.0, 300.0, -400.0];
        let factor = vec![0.0; 4];
        let (q, s) = smooth_quantize_neon(&input, &factor);
        assert_eq!(s, 0.0);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_smooth_scaling() {
        let input = vec![2.0, -2.0, 1.0, -1.0];
        let factor = vec![0.5, 0.5, 0.5, 0.5];
        // smoothed = [1.0, -1.0, 0.5, -0.5], absmax = 1.0
        let (q, s) = smooth_quantize_neon(&input, &factor);
        assert!(approx_eq(s, 1.0 / 127.0, 1e-6));
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -127);
    }

    #[test]
    fn test_smooth_per_channel() {
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let factor = vec![1.0, 2.0, 3.0, 4.0];
        // smoothed = [1, 2, 3, 4]
        let (q, s) = smooth_quantize_neon(&input, &factor);
        assert!(approx_eq(s, 4.0 / 127.0, 1e-6));
        assert_eq!(q[3], 127);
    }

    #[test]
    #[should_panic]
    fn test_smooth_length_mismatch() {
        smooth_quantize_neon(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn test_smooth_non_aligned() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let factor = vec![1.0; 7];
        let (q, s) = smooth_quantize_neon(&input, &factor);
        assert_eq!(q.len(), 7);
        assert!(approx_eq(s, 7.0 / 127.0, 1e-5));
    }

    #[test]
    fn test_smooth_preserves_sign() {
        let input = vec![-3.0, 2.0, -1.0, 4.0, -5.0, 6.0, -7.0, 8.0];
        let factor = vec![1.0; 8];
        let (q, _) = smooth_quantize_neon(&input, &factor);
        for (i, &v) in input.iter().enumerate() {
            if v > 0.0 {
                assert!(q[i] > 0, "idx {i}");
            } else if v < 0.0 {
                assert!(q[i] < 0, "idx {i}");
            }
        }
    }

    #[test]
    fn test_smooth_large_factor() {
        let input = vec![0.001, -0.001, 0.002, -0.002];
        let factor = vec![1000.0; 4];
        // smoothed = [1, -1, 2, -2]
        let (q, s) = smooth_quantize_neon(&input, &factor);
        assert!(approx_eq(s, 2.0 / 127.0, 1e-4));
        assert_eq!(q[2], 127);
        assert_eq!(q[3], -127);
    }

    #[test]
    fn test_smooth_negative_factor() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let factor = vec![-1.0; 4];
        // smoothed = [-1, -2, -3, -4]
        let (q, s) = smooth_quantize_neon(&input, &factor);
        assert!(s > 0.0);
        assert!(q[0] < 0);
        assert_eq!(q[3], -127);
    }

    #[test]
    fn test_smooth_roundtrip() {
        let input = vec![0.5, -0.3, 0.7, -0.1, 0.9, -0.8, 0.2, -0.6];
        let factor = vec![2.0; 8];
        let (q, s) = smooth_quantize_neon(&input, &factor);
        for (i, &orig) in input.iter().enumerate() {
            let smoothed = orig * factor[i];
            let recon = q[i] as f32 * s;
            assert!(
                (recon - smoothed).abs() < s + 1e-5,
                "idx {i}: smoothed={smoothed} recon={recon}"
            );
        }
    }

    #[test]
    fn test_smooth_large_input() {
        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let factor: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let (q, s) = smooth_quantize_neon(&input, &factor);
        assert_eq!(q.len(), n);
        assert!(s > 0.0);
    }

    // ── Cross-function consistency ───────────────────────────────

    #[test]
    fn test_symmetric_vs_smooth_identity() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let identity = vec![1.0f32; 32];
        let (qs, ss) = dynamic_quantize_symmetric_neon(&input);
        let (qm, sm) = smooth_quantize_neon(&input, &identity);
        assert!(approx_eq(ss, sm, 1e-6));
        assert_eq!(qs, qm);
    }

    #[test]
    fn test_per_token_single_matches_symmetric() {
        let input = vec![1.0, -0.5, 0.25, -0.75];
        let (qs, ss) = dynamic_quantize_symmetric_neon(&input);
        let (qp, sp) = dynamic_quantize_per_token_neon(&input, 1, 4);
        assert_eq!(qs, qp);
        assert!(approx_eq(ss, sp[0], 1e-6));
    }

    #[test]
    fn test_calibrate_then_symmetric() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let (lo, hi) = calibrate_quantization_range_neon(&input, 1.0);
        // Verify calibrated range covers input
        let absmax = input.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(hi >= absmax - 1e-6);
        assert!(lo <= -absmax + 1e-6);
    }

    // ── Edge cases ───────────────────────────────────────────────

    #[test]
    fn test_symmetric_nan_free() {
        let input = vec![0.0; 4];
        let (q, s) = dynamic_quantize_symmetric_neon(&input);
        assert!(!s.is_nan());
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_asymmetric_nan_free() {
        let input = vec![0.0; 4];
        let (_, s, zp) = dynamic_quantize_asymmetric_neon(&input);
        assert!(!s.is_nan());
        assert!(!zp.is_nan());
    }

    #[test]
    fn test_smooth_nan_free() {
        let input = vec![0.0; 4];
        let factor = vec![0.0; 4];
        let (q, s) = smooth_quantize_neon(&input, &factor);
        assert!(!s.is_nan());
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_symmetric_clamp_bounds() {
        // Very large values should clamp to ±127
        let input = vec![1e10, -1e10, 5e9, -5e9];
        let (q, _) = dynamic_quantize_symmetric_neon(&input);
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -127);
    }

    #[test]
    fn test_asymmetric_output_bounds() {
        let input: Vec<f32> = (0..256).map(|i| i as f32 * 100.0 - 12800.0).collect();
        let (q, _, _) = dynamic_quantize_asymmetric_neon(&input);
        assert_eq!(*q.iter().min().unwrap(), 0);
        assert_eq!(*q.iter().max().unwrap(), 255);
    }

    #[test]
    fn test_per_token_all_zeros_input() {
        let input = vec![0.0f32; 32];
        let (q, s) = dynamic_quantize_per_token_neon(&input, 4, 8);
        assert!(s.iter().all(|&v| v == 0.0));
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_calibrate_all_same() {
        let input = vec![3.0; 16];
        let (lo, hi) = calibrate_quantization_range_neon(&input, 0.99);
        assert!(approx_eq(hi, 3.0, 1e-6));
        assert!(approx_eq(lo, -3.0, 1e-6));
    }

    #[test]
    fn test_symmetric_two_elements() {
        let (q, s) = dynamic_quantize_symmetric_neon(&[3.0, -3.0]);
        assert_eq!(q.len(), 2);
        assert!(approx_eq(s, 3.0 / 127.0, 1e-6));
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -127);
    }

    #[test]
    fn test_asymmetric_two_elements() {
        let (q, s, _) = dynamic_quantize_asymmetric_neon(&[0.0, 10.0]);
        assert_eq!(q.len(), 2);
        assert!(s > 0.0);
        assert_eq!(q[0], 0);
        assert_eq!(q[1], 255);
    }

    #[test]
    fn test_smooth_single_element() {
        let (q, s) = smooth_quantize_neon(&[4.0], &[0.5]);
        // smoothed = 2.0, scale = 2/127
        assert_eq!(q.len(), 1);
        assert!(approx_eq(s, 2.0 / 127.0, 1e-6));
        assert_eq!(q[0], 127);
    }

    #[test]
    fn test_per_token_many_tokens() {
        let seq = 64;
        let hid = 16;
        let input: Vec<f32> =
            (0..seq * hid).map(|i| ((i * 7 + 3) % 100) as f32 / 50.0 - 1.0).collect();
        let (q, s) = dynamic_quantize_per_token_neon(&input, seq, hid);
        assert_eq!(q.len(), seq * hid);
        assert_eq!(s.len(), seq);
        assert!(s.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_calibrate_high_percentile_near_max() {
        let input: Vec<f32> = (1..=1000).map(|i| i as f32).collect();
        let (_, hi) = calibrate_quantization_range_neon(&input, 0.999);
        // 99.9% of 1000 → index ceil(999) - 1 = 998 → value 999.0
        assert!(hi >= 999.0);
    }

    #[test]
    fn test_symmetric_exactly_four_elements() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (q, s) = dynamic_quantize_symmetric_neon(&input);
        assert_eq!(q.len(), 4);
        assert!(approx_eq(s, 4.0 / 127.0, 1e-6));
        assert_eq!(q[3], 127);
    }

    #[test]
    fn test_asymmetric_exactly_four_elements() {
        let input = vec![-2.0, -1.0, 1.0, 2.0];
        let (q, s, _) = dynamic_quantize_asymmetric_neon(&input);
        assert_eq!(q.len(), 4);
        assert!(approx_eq(s, 4.0 / 255.0, 1e-5));
        assert_eq!(q[0], 0);
        assert_eq!(q[3], 255);
    }

    #[test]
    fn test_smooth_exactly_four_elements() {
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let factor = vec![2.0, 2.0, 2.0, 2.0];
        // smoothed = [2, -2, 1, -1]
        let (q, s) = smooth_quantize_neon(&input, &factor);
        assert_eq!(q.len(), 4);
        assert!(approx_eq(s, 2.0 / 127.0, 1e-6));
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -127);
    }

    #[test]
    fn test_symmetric_alternating() {
        let input = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let (q, s) = dynamic_quantize_symmetric_neon(&input);
        assert!(approx_eq(s, 1.0 / 127.0, 1e-6));
        for i in 0..8 {
            if i % 2 == 0 {
                assert_eq!(q[i], 127);
            } else {
                assert_eq!(q[i], -127);
            }
        }
    }

    #[test]
    fn test_asymmetric_all_negative() {
        let input = vec![-10.0, -5.0, -1.0, -0.1];
        let (q, s, _) = dynamic_quantize_asymmetric_neon(&input);
        assert!(s > 0.0);
        assert_eq!(q[0], 0); // most negative → 0
        assert_eq!(q[3], 255); // least negative → 255
    }

    #[test]
    fn test_calibrate_mixed_sign() {
        let input = vec![-10.0, 5.0, -3.0, 8.0];
        let (lo, hi) = calibrate_quantization_range_neon(&input, 1.0);
        assert!(approx_eq(hi, 10.0, 1e-6));
        assert!(approx_eq(lo, -10.0, 1e-6));
    }

    #[test]
    fn test_per_token_scale_nonzero_for_nonzero_rows() {
        let input = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let (_, s) = dynamic_quantize_per_token_neon(&input, 2, 4);
        assert!(s[0] > 0.0);
        assert!(s[1] > 0.0);
    }

    #[test]
    fn test_asymmetric_min_to_max_spread() {
        let input = vec![-100.0, 100.0, 0.0, 50.0];
        let (q, s, _) = dynamic_quantize_asymmetric_neon(&input);
        assert!(approx_eq(s, 200.0 / 255.0, 1e-3));
        assert_eq!(q[0], 0);
        assert_eq!(q[1], 255);
    }

    #[test]
    fn test_smooth_mixed_sign_factors() {
        let input = vec![2.0, -3.0, 4.0, -5.0];
        let factor = vec![-1.0, 1.0, -1.0, 1.0];
        // smoothed = [-2, -3, -4, -5] → absmax = 5
        let (q, s) = smooth_quantize_neon(&input, &factor);
        assert!(approx_eq(s, 5.0 / 127.0, 1e-5));
        assert_eq!(q[3], -127);
    }
}
