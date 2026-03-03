//! ARM NEON optimized normalization kernels.
//!
//! On non-`aarch64` targets every function delegates to the scalar fallback.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── helpers ──────────────────────────────────────────────────────────

/// Horizontal sum of a NEON `float32x4_t`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vaddvq_f32_compat(v: float32x4_t) -> f32 {
    let pair = unsafe { vpadd_f32(vget_low_f32(v), vget_high_f32(v)) };
    unsafe { vget_lane_f32(vpadd_f32(pair, pair), 0) }
}

// ── NEON mean ────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub fn mean_f32_neon(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    unsafe {
        let n = data.len();
        let chunks = n / 4;
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            acc = vaddq_f32(acc, v);
        }
        let mut sum = vaddvq_f32_compat(acc);
        for &v in &data[chunks * 4..] {
            sum += v;
        }
        sum / n as f32
    }
}

// ── NEON mean of squares ─────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub fn mean_sq_f32_neon(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    unsafe {
        let n = data.len();
        let chunks = n / 4;
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            acc = vmlaq_f32(acc, v, v);
        }
        let mut sum = vaddvq_f32_compat(acc);
        for &v in &data[chunks * 4..] {
            sum += v * v;
        }
        sum / n as f32
    }
}

// ── NEON variance ────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub fn variance_f32_neon(data: &[f32], mean: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    unsafe {
        let n = data.len();
        let chunks = n / 4;
        let vmean = vdupq_n_f32(mean);
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            let diff = vsubq_f32(v, vmean);
            acc = vmlaq_f32(acc, diff, diff);
        }
        let mut sum = vaddvq_f32_compat(acc);
        for &v in &data[chunks * 4..] {
            let d = v - mean;
            sum += d * d;
        }
        sum / n as f32
    }
}

// ── NEON `LayerNorm` ─────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub fn layer_norm_f32_neon(data: &mut [f32], epsilon: f32) {
    if data.is_empty() {
        return;
    }
    let m = mean_f32_neon(data);
    let v = variance_f32_neon(data, m);
    let inv_std = 1.0 / (v + epsilon).sqrt();
    unsafe {
        let n = data.len();
        let chunks = n / 4;
        let vmean = vdupq_n_f32(m);
        let vscale = vdupq_n_f32(inv_std);
        let ptr = data.as_mut_ptr();
        for i in 0..chunks {
            let x = vld1q_f32(ptr.add(i * 4));
            let centered = vsubq_f32(x, vmean);
            let normed = vmulq_f32(centered, vscale);
            vst1q_f32(ptr.add(i * 4), normed);
        }
        for x in &mut data[chunks * 4..] {
            *x = (*x - m) * inv_std;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub fn layer_norm_affine_f32_neon(data: &mut [f32], gamma: &[f32], beta: &[f32], epsilon: f32) {
    if data.is_empty() {
        return;
    }
    let m = mean_f32_neon(data);
    let v = variance_f32_neon(data, m);
    let inv_std = 1.0 / (v + epsilon).sqrt();
    unsafe {
        let n = data.len();
        let chunks = n / 4;
        let vmean = vdupq_n_f32(m);
        let vscale = vdupq_n_f32(inv_std);
        let dptr = data.as_mut_ptr();
        let gptr = gamma.as_ptr();
        let bptr = beta.as_ptr();
        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(dptr.add(off));
            let g = vld1q_f32(gptr.add(off));
            let b = vld1q_f32(bptr.add(off));
            let centered = vsubq_f32(x, vmean);
            let normed = vmulq_f32(centered, vscale);
            let scaled = vmlaq_f32(b, g, normed);
            vst1q_f32(dptr.add(off), scaled);
        }
        for i in (chunks * 4)..n {
            data[i] = (gamma[i] * (data[i] - m)).mul_add(inv_std, beta[i]);
        }
    }
}

// ── NEON `RMSNorm` ───────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub fn rms_norm_f32_neon(data: &mut [f32], epsilon: f32) {
    if data.is_empty() {
        return;
    }
    let ms = mean_sq_f32_neon(data);
    let inv_rms = 1.0 / (ms + epsilon).sqrt();
    unsafe {
        let n = data.len();
        let chunks = n / 4;
        let vscale = vdupq_n_f32(inv_rms);
        let ptr = data.as_mut_ptr();
        for i in 0..chunks {
            let x = vld1q_f32(ptr.add(i * 4));
            let normed = vmulq_f32(x, vscale);
            vst1q_f32(ptr.add(i * 4), normed);
        }
        for x in &mut data[chunks * 4..] {
            *x *= inv_rms;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub fn rms_norm_scale_f32_neon(data: &mut [f32], gamma: &[f32], epsilon: f32) {
    if data.is_empty() {
        return;
    }
    let ms = mean_sq_f32_neon(data);
    let inv_rms = 1.0 / (ms + epsilon).sqrt();
    unsafe {
        let n = data.len();
        let chunks = n / 4;
        let vscale = vdupq_n_f32(inv_rms);
        let dptr = data.as_mut_ptr();
        let gptr = gamma.as_ptr();
        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(dptr.add(off));
            let g = vld1q_f32(gptr.add(off));
            let normed = vmulq_f32(x, vscale);
            let scaled = vmulq_f32(g, normed);
            vst1q_f32(dptr.add(off), scaled);
        }
        for i in (chunks * 4)..n {
            data[i] = gamma[i] * data[i] * inv_rms;
        }
    }
}

// ── NEON `GroupNorm` ─────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub fn group_norm_f32_neon(
    data: &mut [f32],
    num_groups: usize,
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
) {
    let total = data.len();
    if total == 0 || num_groups == 0 {
        return;
    }
    let cpg = total / num_groups;
    for g in 0..num_groups {
        let start = g * cpg;
        let end = start + cpg;
        let group = &data[start..end];
        let m = mean_f32_neon(group);
        let v = variance_f32_neon(group, m);
        let inv_std = 1.0 / (v + epsilon).sqrt();
        unsafe {
            let chunks = cpg / 4;
            let vmean = vdupq_n_f32(m);
            let vscale = vdupq_n_f32(inv_std);
            let dptr = data.as_mut_ptr().add(start);
            let gptr = gamma.as_ptr().add(start);
            let bptr = beta.as_ptr().add(start);
            for i in 0..chunks {
                let off = i * 4;
                let x = vld1q_f32(dptr.add(off));
                let gg = vld1q_f32(gptr.add(off));
                let bb = vld1q_f32(bptr.add(off));
                let centered = vsubq_f32(x, vmean);
                let normed = vmulq_f32(centered, vscale);
                let scaled = vmlaq_f32(bb, gg, normed);
                vst1q_f32(dptr.add(off), scaled);
            }
            for i in (chunks * 4)..cpg {
                let idx = start + i;
                data[idx] = (gamma[idx] * (data[idx] - m)).mul_add(inv_std, beta[idx]);
            }
        }
    }
}

// ── Dispatch functions (choose NEON or scalar) ───────────────────────

/// Select the best available `LayerNorm` implementation.
#[inline]
pub fn layer_norm_f32_dispatch(data: &mut [f32], epsilon: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        layer_norm_f32_neon(data, epsilon);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::scalar::layer_norm_f32(data, epsilon);
    }
}

/// `LayerNorm` with affine dispatch.
#[inline]
pub fn layer_norm_affine_f32_dispatch(data: &mut [f32], gamma: &[f32], beta: &[f32], epsilon: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        layer_norm_affine_f32_neon(data, gamma, beta, epsilon);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::scalar::layer_norm_affine_f32(data, gamma, beta, epsilon);
    }
}

/// `RMSNorm` dispatch.
#[inline]
pub fn rms_norm_f32_dispatch(data: &mut [f32], epsilon: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        rms_norm_f32_neon(data, epsilon);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::scalar::rms_norm_f32(data, epsilon);
    }
}

/// `RMSNorm` with scale dispatch.
#[inline]
pub fn rms_norm_scale_f32_dispatch(data: &mut [f32], gamma: &[f32], epsilon: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        rms_norm_scale_f32_neon(data, gamma, epsilon);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::scalar::rms_norm_scale_f32(data, gamma, epsilon);
    }
}

/// `GroupNorm` dispatch.
#[inline]
pub fn group_norm_f32_dispatch(
    data: &mut [f32],
    num_groups: usize,
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        group_norm_f32_neon(data, num_groups, gamma, beta, epsilon);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::scalar::group_norm_f32(data, num_groups, gamma, beta, epsilon);
    }
}
