//! ARM NEON accelerated softmax implementations.
//!
//! Each function processes 4 × f32 lanes at a time via NEON SIMD and falls
//! back to scalar code for the remaining tail elements.

use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vdupq_n_f32, vgetq_lane_f32, vld1q_f32, vmaxq_f32, vmulq_f32,
    vst1q_f32, vsubq_f32,
};

// ── helpers ────────────────────────────────────────────────────────────────

/// Horizontal max of a `float32x4_t`.
#[inline(always)]
unsafe fn hmax_f32x4(v: float32x4_t) -> f32 {
    unsafe {
        let a = vgetq_lane_f32(v, 0);
        let b = vgetq_lane_f32(v, 1);
        let c = vgetq_lane_f32(v, 2);
        let d = vgetq_lane_f32(v, 3);
        a.max(b).max(c.max(d))
    }
}

/// Horizontal sum of a `float32x4_t`.
#[inline(always)]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    unsafe {
        let a = vgetq_lane_f32(v, 0);
        let b = vgetq_lane_f32(v, 1);
        let c = vgetq_lane_f32(v, 2);
        let d = vgetq_lane_f32(v, 3);
        a + b + c + d
    }
}

/// Element-wise `exp` for a NEON lane (no NEON `exp` intrinsic – use scalar).
#[inline(always)]
unsafe fn exp_f32x4(v: float32x4_t) -> float32x4_t {
    unsafe {
        let a = vgetq_lane_f32(v, 0).exp();
        let b = vgetq_lane_f32(v, 1).exp();
        let c = vgetq_lane_f32(v, 2).exp();
        let d = vgetq_lane_f32(v, 3).exp();
        let arr: [f32; 4] = [a, b, c, d];
        vld1q_f32(arr.as_ptr())
    }
}

// ── public backend functions ───────────────────────────────────────────────

pub(crate) fn softmax_inplace(x: &mut [f32]) {
    // Safety: all NEON intrinsics require `target_arch = "aarch64"` which is
    // guaranteed by the `#[cfg]` gate on this module.
    unsafe { softmax_inplace_neon(x) }
}

pub(crate) fn log_softmax(x: &[f32]) -> Vec<f32> {
    unsafe { log_softmax_neon(x) }
}

pub(crate) fn temperature_softmax(x: &[f32], temperature: f32) -> Vec<f32> {
    let inv_temp = 1.0 / temperature;
    let mut scaled: Vec<f32> = unsafe {
        let mut out = vec![0.0_f32; x.len()];
        let chunks = x.len() / 4;
        let vt = vdupq_n_f32(inv_temp);
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(x.as_ptr().add(offset));
            let s = vmulq_f32(v, vt);
            vst1q_f32(out.as_mut_ptr().add(offset), s);
        }
        for i in (chunks * 4)..x.len() {
            out[i] = x[i] * inv_temp;
        }
        out
    };
    softmax_inplace(&mut scaled);
    scaled
}

pub(crate) fn online_softmax(x: &[f32]) -> Vec<f32> {
    // Online softmax doesn't lend itself to trivial NEON vectorisation of the
    // running-max update (it's inherently sequential). Delegate to scalar.
    crate::scalar::online_softmax(x)
}

// ── unsafe NEON kernels ────────────────────────────────────────────────────

unsafe fn softmax_inplace_neon(x: &mut [f32]) {
    let len = x.len();
    let chunks = len / 4;
    let tail = chunks * 4;

    // 1. Find max ───────────────────────────────────────────────────────────
    let mut vmax = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(x.as_ptr().add(i * 4)) };
        vmax = unsafe { vmaxq_f32(vmax, v) };
    }
    let mut max_val = unsafe { hmax_f32x4(vmax) };
    for &v in &x[tail..] {
        max_val = max_val.max(v);
    }

    // 2. exp(x - max) and accumulate sum ────────────────────────────────────
    let vmax_splat = unsafe { vdupq_n_f32(max_val) };
    let mut vsum = unsafe { vdupq_n_f32(0.0) };
    for i in 0..chunks {
        let offset = i * 4;
        let v = unsafe { vld1q_f32(x.as_ptr().add(offset)) };
        let shifted = unsafe { vsubq_f32(v, vmax_splat) };
        let e = unsafe { exp_f32x4(shifted) };
        unsafe { vst1q_f32(x.as_mut_ptr().add(offset), e) };
        vsum = unsafe { vaddq_f32(vsum, e) };
    }
    let mut sum = unsafe { hsum_f32x4(vsum) };
    for v in &mut x[tail..] {
        let e = (*v - max_val).exp();
        sum += e;
        *v = e;
    }

    // 3. normalise ──────────────────────────────────────────────────────────
    let inv = 1.0 / sum;
    let vinv = unsafe { vdupq_n_f32(inv) };
    for i in 0..chunks {
        let offset = i * 4;
        let v = unsafe { vld1q_f32(x.as_ptr().add(offset)) };
        let n = unsafe { vmulq_f32(v, vinv) };
        unsafe { vst1q_f32(x.as_mut_ptr().add(offset), n) };
    }
    for v in &mut x[tail..] {
        *v *= inv;
    }
}

unsafe fn log_softmax_neon(x: &[f32]) -> Vec<f32> {
    let len = x.len();
    let chunks = len / 4;
    let tail = chunks * 4;

    // 1. Find max
    let mut vmax = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(x.as_ptr().add(i * 4)) };
        vmax = unsafe { vmaxq_f32(vmax, v) };
    }
    let mut max_val = unsafe { hmax_f32x4(vmax) };
    for &v in &x[tail..] {
        max_val = max_val.max(v);
    }

    // 2. sum(exp(x - max))
    let vmax_splat = unsafe { vdupq_n_f32(max_val) };
    let mut vsum = unsafe { vdupq_n_f32(0.0) };
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(x.as_ptr().add(i * 4)) };
        let shifted = unsafe { vsubq_f32(v, vmax_splat) };
        let e = unsafe { exp_f32x4(shifted) };
        vsum = unsafe { vaddq_f32(vsum, e) };
    }
    let mut sum = unsafe { hsum_f32x4(vsum) };
    for &v in &x[tail..] {
        sum += (v - max_val).exp();
    }

    let log_sum_exp = sum.ln() + max_val;

    // 3. x_i - log_sum_exp
    let mut out = vec![0.0_f32; len];
    let vlse = unsafe { vdupq_n_f32(log_sum_exp) };
    for i in 0..chunks {
        let offset = i * 4;
        let v = unsafe { vld1q_f32(x.as_ptr().add(offset)) };
        let r = unsafe { vsubq_f32(v, vlse) };
        unsafe { vst1q_f32(out.as_mut_ptr().add(offset), r) };
    }
    for i in tail..len {
        out[i] = x[i] - log_sum_exp;
    }
    out
}
