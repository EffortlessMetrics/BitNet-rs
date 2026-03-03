//! ARM NEON vectorized elementwise operations for Apple Silicon.
//!
//! Provides `f32` elementwise arithmetic, comparison, selection, and reduction
//! operations using NEON intrinsics, with scalar fallbacks for non-aarch64
//! targets and remainder elements.

// ---------------------------------------------------------------------------
// NEON binary arithmetic (aarch64)
// ---------------------------------------------------------------------------

/// Elementwise addition with broadcasting: `out[i] = a[i] + b[i]`.
///
/// When `b.len() == 1` the single value is broadcast across all elements.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_vec_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, out.len());
    assert!(b.len() == n || b.len() == 1, "b must match a or be len 1");

    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    if b.len() == 1 {
        let vb = vdupq_n_f32(b[0]);
        for i in 0..chunks {
            let off = i * 4;
            unsafe {
                let va = vld1q_f32(a_ptr.add(off));
                vst1q_f32(o_ptr.add(off), vaddq_f32(va, vb));
            }
        }
        for i in (chunks * 4)..n {
            out[i] = a[i] + b[0];
        }
    } else {
        let b_ptr = b.as_ptr();
        for i in 0..chunks {
            let off = i * 4;
            unsafe {
                let va = vld1q_f32(a_ptr.add(off));
                let vb = vld1q_f32(b_ptr.add(off));
                vst1q_f32(o_ptr.add(off), vaddq_f32(va, vb));
            }
        }
        for i in (chunks * 4)..n {
            out[i] = a[i] + b[i];
        }
    }
}

/// Elementwise subtraction with broadcasting: `out[i] = a[i] - b[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_vec_sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, out.len());
    assert!(b.len() == n || b.len() == 1, "b must match a or be len 1");

    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    if b.len() == 1 {
        let vb = vdupq_n_f32(b[0]);
        for i in 0..chunks {
            let off = i * 4;
            unsafe {
                let va = vld1q_f32(a_ptr.add(off));
                vst1q_f32(o_ptr.add(off), vsubq_f32(va, vb));
            }
        }
        for i in (chunks * 4)..n {
            out[i] = a[i] - b[0];
        }
    } else {
        let b_ptr = b.as_ptr();
        for i in 0..chunks {
            let off = i * 4;
            unsafe {
                let va = vld1q_f32(a_ptr.add(off));
                let vb = vld1q_f32(b_ptr.add(off));
                vst1q_f32(o_ptr.add(off), vsubq_f32(va, vb));
            }
        }
        for i in (chunks * 4)..n {
            out[i] = a[i] - b[i];
        }
    }
}

/// Elementwise multiplication with broadcasting: `out[i] = a[i] * b[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_vec_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, out.len());
    assert!(b.len() == n || b.len() == 1, "b must match a or be len 1");

    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    if b.len() == 1 {
        let vb = vdupq_n_f32(b[0]);
        for i in 0..chunks {
            let off = i * 4;
            unsafe {
                let va = vld1q_f32(a_ptr.add(off));
                vst1q_f32(o_ptr.add(off), vmulq_f32(va, vb));
            }
        }
        for i in (chunks * 4)..n {
            out[i] = a[i] * b[0];
        }
    } else {
        let b_ptr = b.as_ptr();
        for i in 0..chunks {
            let off = i * 4;
            unsafe {
                let va = vld1q_f32(a_ptr.add(off));
                let vb = vld1q_f32(b_ptr.add(off));
                vst1q_f32(o_ptr.add(off), vmulq_f32(va, vb));
            }
        }
        for i in (chunks * 4)..n {
            out[i] = a[i] * b[i];
        }
    }
}

/// Elementwise division with broadcasting: `out[i] = a[i] / b[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_vec_div_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, out.len());
    assert!(b.len() == n || b.len() == 1, "b must match a or be len 1");

    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    if b.len() == 1 {
        let vb = vdupq_n_f32(b[0]);
        for i in 0..chunks {
            let off = i * 4;
            unsafe {
                let va = vld1q_f32(a_ptr.add(off));
                vst1q_f32(o_ptr.add(off), vdivq_f32(va, vb));
            }
        }
        for i in (chunks * 4)..n {
            out[i] = a[i] / b[0];
        }
    } else {
        let b_ptr = b.as_ptr();
        for i in 0..chunks {
            let off = i * 4;
            unsafe {
                let va = vld1q_f32(a_ptr.add(off));
                let vb = vld1q_f32(b_ptr.add(off));
                vst1q_f32(o_ptr.add(off), vdivq_f32(va, vb));
            }
        }
        for i in (chunks * 4)..n {
            out[i] = a[i] / b[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Fused multiply-add
// ---------------------------------------------------------------------------

/// Fused multiply-add: `out[i] = a[i] * b[i] + c[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fma_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, c.len());
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            let vc = vld1q_f32(c_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vfmaq_f32(vc, va, vb));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i].mul_add(b[i], c[i]);
    }
}

/// Fused multiply-add with scalar multiplier: `out[i] = a[i] * scale + b[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fma_scalar_f32(a: &[f32], scale: f32, b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;
    let vs = vdupq_n_f32(scale);

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vfmaq_f32(vb, va, vs));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i].mul_add(scale, b[i]);
    }
}

// ---------------------------------------------------------------------------
// Element-wise min / max / clamp
// ---------------------------------------------------------------------------

/// Elementwise minimum: `out[i] = min(a[i], b[i])`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_min_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vminq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i].min(b[i]);
    }
}

/// Elementwise maximum: `out[i] = max(a[i], b[i])`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_max_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vmaxq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i].max(b[i]);
    }
}

/// Elementwise clamp: `out[i] = clamp(a[i], lo, hi)`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_clamp_f32(a: &[f32], lo: f32, hi: f32, out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;
    let vlo = vdupq_n_f32(lo);
    let vhi = vdupq_n_f32(hi);

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let clamped = vminq_f32(vmaxq_f32(va, vlo), vhi);
            vst1q_f32(o_ptr.add(off), clamped);
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i].max(lo).min(hi);
    }
}

// ---------------------------------------------------------------------------
// Absolute value and sign
// ---------------------------------------------------------------------------

/// Elementwise absolute value: `out[i] = |a[i]|`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_abs_f32(a: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vabsq_f32(va));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i].abs();
    }
}

/// Elementwise sign: `out[i] = signum(a[i])` (-1, 0, or 1).
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sign_f32(a: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    let vzero = vdupq_n_f32(0.0);
    let vone = vdupq_n_f32(1.0);
    let vneg = vdupq_n_f32(-1.0);

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let pos_mask = vcgtq_f32(va, vzero);
            let neg_mask = vcltq_f32(va, vzero);
            let pos_part: float32x4_t =
                vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vone), pos_mask));
            let neg_part: float32x4_t =
                vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vneg), neg_mask));
            let result = vaddq_f32(pos_part, neg_part);
            vst1q_f32(o_ptr.add(off), result);
        }
    }
    for i in (chunks * 4)..n {
        out[i] = if a[i] > 0.0 {
            1.0
        } else if a[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
    }
}

/// Elementwise negate: `out[i] = -a[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_neg_f32(a: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vnegq_f32(va));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = -a[i];
    }
}

// ---------------------------------------------------------------------------
// Power, sqrt, reciprocal (approximate)
// ---------------------------------------------------------------------------

/// Elementwise square root: `out[i] = sqrt(a[i])`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sqrt_f32(a: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vsqrtq_f32(va));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i].sqrt();
    }
}

/// Approximate elementwise reciprocal: `out[i] ≈ 1 / a[i]`.
///
/// Uses a single Newton-Raphson refinement step on the NEON `vrecpeq_f32`
/// estimate for ~24-bit accuracy.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_recip_f32(a: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let est = vrecpeq_f32(va);
            let refined = vmulq_f32(est, vrecpsq_f32(va, est));
            vst1q_f32(o_ptr.add(off), refined);
        }
    }
    for i in (chunks * 4)..n {
        out[i] = 1.0 / a[i];
    }
}

/// Approximate elementwise reciprocal square root: `out[i] ≈ 1/sqrt(a[i])`.
///
/// Uses a single Newton-Raphson refinement step on the NEON `vrsqrteq_f32`
/// estimate.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rsqrt_f32(a: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let est = vrsqrteq_f32(va);
            let refined = vmulq_f32(est, vrsqrtsq_f32(va, vmulq_f32(est, est)));
            vst1q_f32(o_ptr.add(off), refined);
        }
    }
    for i in (chunks * 4)..n {
        out[i] = 1.0 / a[i].sqrt();
    }
}

/// Elementwise power: `out[i] = a[i].powf(exp)`.
///
/// Scalar implementation — no NEON intrinsic for general power, but the loop
/// is vectorised by LLVM when built with `-C target-cpu=native`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_pow_f32(a: &[f32], exp: f32, out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = a[i].powf(exp);
    }
}

/// Elementwise square: `out[i] = a[i] * a[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_square_f32(a: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vmulq_f32(va, va));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i] * a[i];
    }
}

// ---------------------------------------------------------------------------
// Element-wise comparison → mask vectors
// ---------------------------------------------------------------------------

/// Elementwise greater-than: `out[i] = if a[i] > b[i] { 1.0 } else { 0.0 }`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_cmpgt_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;
    let vone = vdupq_n_f32(1.0);
    let vzero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            let mask = vcgtq_f32(va, vb);
            vst1q_f32(o_ptr.add(off), vbslq_f32(mask, vone, vzero));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = if a[i] > b[i] { 1.0 } else { 0.0 };
    }
}

/// Elementwise less-than: `out[i] = if a[i] < b[i] { 1.0 } else { 0.0 }`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_cmplt_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;
    let vone = vdupq_n_f32(1.0);
    let vzero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            let mask = vcltq_f32(va, vb);
            vst1q_f32(o_ptr.add(off), vbslq_f32(mask, vone, vzero));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = if a[i] < b[i] { 1.0 } else { 0.0 };
    }
}

/// Elementwise equality: `out[i] = if a[i] == b[i] { 1.0 } else { 0.0 }`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_cmpeq_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;
    let vone = vdupq_n_f32(1.0);
    let vzero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            let mask = vceqq_f32(va, vb);
            vst1q_f32(o_ptr.add(off), vbslq_f32(mask, vone, vzero));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = if a[i] == b[i] { 1.0 } else { 0.0 };
    }
}

/// Elementwise not-equal: `out[i] = if a[i] != b[i] { 1.0 } else { 0.0 }`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_cmpne_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;
    let vone = vdupq_n_f32(1.0);
    let vzero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            let eq_mask = vceqq_f32(va, vb);
            // ne = NOT eq: select(eq_mask, 0.0, 1.0)
            vst1q_f32(o_ptr.add(off), vbslq_f32(eq_mask, vzero, vone));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = if a[i] != b[i] { 1.0 } else { 0.0 };
    }
}

/// Elementwise greater-than-or-equal comparison.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_cmpge_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;
    let vone = vdupq_n_f32(1.0);
    let vzero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            let mask = vcgeq_f32(va, vb);
            vst1q_f32(o_ptr.add(off), vbslq_f32(mask, vone, vzero));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = if a[i] >= b[i] { 1.0 } else { 0.0 };
    }
}

// ---------------------------------------------------------------------------
// Conditional select
// ---------------------------------------------------------------------------

/// Conditional select: `out[i] = if mask[i] != 0.0 { a[i] } else { b[i] }`.
///
/// A mask value of `1.0` (or any non-zero) selects from `a`; `0.0` selects
/// from `b`. This pairs naturally with the `neon_cmp*` functions above.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_select_f32(mask: &[f32], a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = mask.len();
    assert_eq!(n, a.len());
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());

    let m_ptr = mask.as_ptr();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let chunks = n / 4;
    let vzero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let vm = vld1q_f32(m_ptr.add(off));
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            // cond is all-ones where mask==0 → select b; else select a
            let cond = vceqq_f32(vm, vzero);
            vst1q_f32(o_ptr.add(off), vbslq_f32(cond, vb, va));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = if mask[i] != 0.0 { a[i] } else { b[i] };
    }
}

// ---------------------------------------------------------------------------
// Reduction operations
// ---------------------------------------------------------------------------

/// Horizontal sum across all elements.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_reduce_sum_f32(data: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = data.len();
    if n == 0 {
        return 0.0;
    }

    let ptr = data.as_ptr();
    let chunks = n / 4;
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(ptr.add(i * 4));
            acc = vaddq_f32(acc, v);
        }
    }

    let mut sum = vaddvq_f32(acc);
    for i in (chunks * 4)..n {
        sum += unsafe { *ptr.add(i) };
    }
    sum
}

/// Horizontal maximum across all elements.
///
/// Returns `f32::NEG_INFINITY` for empty slices.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_reduce_max_f32(data: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = data.len();
    if n == 0 {
        return f32::NEG_INFINITY;
    }

    let ptr = data.as_ptr();
    let chunks = n / 4;
    let mut acc = vdupq_n_f32(f32::NEG_INFINITY);

    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(ptr.add(i * 4));
            acc = vmaxq_f32(acc, v);
        }
    }

    let mut m = vmaxvq_f32(acc);
    for i in (chunks * 4)..n {
        let val = unsafe { *ptr.add(i) };
        if val > m {
            m = val;
        }
    }
    m
}

/// Horizontal minimum across all elements.
///
/// Returns `f32::INFINITY` for empty slices.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_reduce_min_f32(data: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = data.len();
    if n == 0 {
        return f32::INFINITY;
    }

    let ptr = data.as_ptr();
    let chunks = n / 4;
    let mut acc = vdupq_n_f32(f32::INFINITY);

    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(ptr.add(i * 4));
            acc = vminq_f32(acc, v);
        }
    }

    let mut m = vminvq_f32(acc);
    for i in (chunks * 4)..n {
        let val = unsafe { *ptr.add(i) };
        if val < m {
            m = val;
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Scalar fallbacks for non-aarch64 targets
// ---------------------------------------------------------------------------

/// Scalar fallback: elementwise add with broadcasting.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_vec_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    assert!(b.len() == n || b.len() == 1);
    if b.len() == 1 {
        for i in 0..n {
            out[i] = a[i] + b[0];
        }
    } else {
        for i in 0..n {
            out[i] = a[i] + b[i];
        }
    }
}

/// Scalar fallback: elementwise sub with broadcasting.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_vec_sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    assert!(b.len() == n || b.len() == 1);
    if b.len() == 1 {
        for i in 0..n {
            out[i] = a[i] - b[0];
        }
    } else {
        for i in 0..n {
            out[i] = a[i] - b[i];
        }
    }
}

/// Scalar fallback: elementwise mul with broadcasting.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_vec_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    assert!(b.len() == n || b.len() == 1);
    if b.len() == 1 {
        for i in 0..n {
            out[i] = a[i] * b[0];
        }
    } else {
        for i in 0..n {
            out[i] = a[i] * b[i];
        }
    }
}

/// Scalar fallback: elementwise div with broadcasting.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_vec_div_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    assert!(b.len() == n || b.len() == 1);
    if b.len() == 1 {
        for i in 0..n {
            out[i] = a[i] / b[0];
        }
    } else {
        for i in 0..n {
            out[i] = a[i] / b[i];
        }
    }
}

/// Scalar fallback: fused multiply-add.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fma_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, c.len());
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = a[i].mul_add(b[i], c[i]);
    }
}

/// Scalar fallback: fused multiply-add with scalar multiplier.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fma_scalar_f32(a: &[f32], scale: f32, b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = a[i].mul_add(scale, b[i]);
    }
}

/// Scalar fallback: elementwise min.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_min_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = a[i].min(b[i]);
    }
}

/// Scalar fallback: elementwise max.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_max_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = a[i].max(b[i]);
    }
}

/// Scalar fallback: elementwise clamp.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_clamp_f32(a: &[f32], lo: f32, hi: f32, out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = a[i].max(lo).min(hi);
    }
}

/// Scalar fallback: absolute value.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_abs_f32(a: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = a[i].abs();
    }
}

/// Scalar fallback: sign.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_sign_f32(a: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = if a[i] > 0.0 {
            1.0
        } else if a[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
    }
}

/// Scalar fallback: negate.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_neg_f32(a: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = -a[i];
    }
}

/// Scalar fallback: square root.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_sqrt_f32(a: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = a[i].sqrt();
    }
}

/// Scalar fallback: reciprocal.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_recip_f32(a: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = 1.0 / a[i];
    }
}

/// Scalar fallback: reciprocal square root.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_rsqrt_f32(a: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = 1.0 / a[i].sqrt();
    }
}

/// Scalar fallback: power.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_pow_f32(a: &[f32], exp: f32, out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = a[i].powf(exp);
    }
}

/// Scalar fallback: square.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_square_f32(a: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = a[i] * a[i];
    }
}

/// Scalar fallback: greater-than comparison.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_cmpgt_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = if a[i] > b[i] { 1.0 } else { 0.0 };
    }
}

/// Scalar fallback: less-than comparison.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_cmplt_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = if a[i] < b[i] { 1.0 } else { 0.0 };
    }
}

/// Scalar fallback: equality comparison.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_cmpeq_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = if a[i] == b[i] { 1.0 } else { 0.0 };
    }
}

/// Scalar fallback: not-equal comparison.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_cmpne_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = if a[i] != b[i] { 1.0 } else { 0.0 };
    }
}

/// Scalar fallback: greater-than-or-equal comparison.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_cmpge_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = if a[i] >= b[i] { 1.0 } else { 0.0 };
    }
}

/// Scalar fallback: conditional select.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_select_f32(mask: &[f32], a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = mask.len();
    assert_eq!(n, a.len());
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());
    for i in 0..n {
        out[i] = if mask[i] != 0.0 { a[i] } else { b[i] };
    }
}

/// Scalar fallback: horizontal sum.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_reduce_sum_f32(data: &[f32]) -> f32 {
    data.iter().sum()
}

/// Scalar fallback: horizontal max.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_reduce_max_f32(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Scalar fallback: horizontal min.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_reduce_min_f32(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::INFINITY, f32::min)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn assert_slices_approx(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(approx_eq(a, e, tol), "index {i}: got {a}, expected {e} (tol {tol})");
        }
    }

    // -----------------------------------------------------------------------
    // Vector add
    // -----------------------------------------------------------------------

    #[test]
    fn test_vec_add_exact() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_vec_add_f32(&a, &b, &mut out) };
        assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_vec_add_remainder() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_vec_add_f32(&a, &b, &mut out) };
        assert_eq!(out, [11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn test_vec_add_broadcast() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0f32];
        let mut out = [0.0f32; 5];
        unsafe { neon_vec_add_f32(&a, &b, &mut out) };
        assert_eq!(out, [11.0, 12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn test_vec_add_empty() {
        let a: &[f32] = &[];
        let b: &[f32] = &[];
        let mut out: Vec<f32> = vec![];
        unsafe { neon_vec_add_f32(a, b, &mut out) };
        assert!(out.is_empty());
    }

    // -----------------------------------------------------------------------
    // Vector sub
    // -----------------------------------------------------------------------

    #[test]
    fn test_vec_sub_exact() {
        let a = [10.0f32, 20.0, 30.0, 40.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_vec_sub_f32(&a, &b, &mut out) };
        assert_eq!(out, [9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn test_vec_sub_broadcast() {
        let a = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let b = [5.0f32];
        let mut out = [0.0f32; 5];
        unsafe { neon_vec_sub_f32(&a, &b, &mut out) };
        assert_eq!(out, [5.0, 15.0, 25.0, 35.0, 45.0]);
    }

    #[test]
    fn test_vec_sub_negative() {
        let a = [1.0f32, -2.0, 3.0, -4.0];
        let b = [-1.0f32, 2.0, -3.0, 4.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_vec_sub_f32(&a, &b, &mut out) };
        assert_eq!(out, [2.0, -4.0, 6.0, -8.0]);
    }

    // -----------------------------------------------------------------------
    // Vector mul
    // -----------------------------------------------------------------------

    #[test]
    fn test_vec_mul_exact() {
        let a = [2.0f32, 3.0, 4.0, 5.0];
        let b = [10.0f32, 10.0, 10.0, 10.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_vec_mul_f32(&a, &b, &mut out) };
        assert_eq!(out, [20.0, 30.0, 40.0, 50.0]);
    }

    #[test]
    fn test_vec_mul_broadcast() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [3.0f32];
        let mut out = [0.0f32; 5];
        unsafe { neon_vec_mul_f32(&a, &b, &mut out) };
        assert_eq!(out, [3.0, 6.0, 9.0, 12.0, 15.0]);
    }

    #[test]
    fn test_vec_mul_zeros() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.0f32, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_vec_mul_f32(&a, &b, &mut out) };
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // Vector div
    // -----------------------------------------------------------------------

    #[test]
    fn test_vec_div_exact() {
        let a = [10.0f32, 20.0, 30.0, 40.0];
        let b = [2.0f32, 5.0, 10.0, 8.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_vec_div_f32(&a, &b, &mut out) };
        assert_eq!(out, [5.0, 4.0, 3.0, 5.0]);
    }

    #[test]
    fn test_vec_div_broadcast() {
        let a = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let b = [10.0f32];
        let mut out = [0.0f32; 5];
        unsafe { neon_vec_div_f32(&a, &b, &mut out) };
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_vec_div_remainder() {
        let a = [6.0f32, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0];
        let b = [3.0f32, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0];
        let mut out = [0.0f32; 7];
        unsafe { neon_vec_div_f32(&a, &b, &mut out) };
        assert_eq!(out, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    // -----------------------------------------------------------------------
    // FMA
    // -----------------------------------------------------------------------

    #[test]
    fn test_fma_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0f32, 3.0, 4.0, 5.0, 6.0];
        let c = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_fma_f32(&a, &b, &c, &mut out) };
        assert_eq!(out, [12.0, 26.0, 42.0, 60.0, 80.0]);
    }

    #[test]
    fn test_fma_zeros() {
        let a = [0.0f32; 4];
        let b = [1.0f32; 4];
        let c = [5.0f32; 4];
        let mut out = [0.0f32; 4];
        unsafe { neon_fma_f32(&a, &b, &c, &mut out) };
        assert_eq!(out, [5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_fma_scalar_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_fma_scalar_f32(&a, 3.0, &b, &mut out) };
        // a*3 + b = [13, 26, 39, 52, 65]
        assert_eq!(out, [13.0, 26.0, 39.0, 52.0, 65.0]);
    }

    #[test]
    fn test_fma_scalar_zero_scale() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_fma_scalar_f32(&a, 0.0, &b, &mut out) };
        assert_eq!(out, [5.0, 6.0, 7.0, 8.0]);
    }

    // -----------------------------------------------------------------------
    // Min / Max / Clamp
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_basic() {
        let a = [1.0f32, 5.0, 3.0, 7.0, 2.0];
        let b = [4.0f32, 2.0, 6.0, 1.0, 8.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_min_f32(&a, &b, &mut out) };
        assert_eq!(out, [1.0, 2.0, 3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_min_negative() {
        let a = [-1.0f32, -5.0, 3.0, -7.0];
        let b = [-4.0f32, -2.0, -6.0, -1.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_min_f32(&a, &b, &mut out) };
        assert_eq!(out, [-4.0, -5.0, -6.0, -7.0]);
    }

    #[test]
    fn test_max_basic() {
        let a = [1.0f32, 5.0, 3.0, 7.0, 2.0];
        let b = [4.0f32, 2.0, 6.0, 1.0, 8.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_max_f32(&a, &b, &mut out) };
        assert_eq!(out, [4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_max_negative() {
        let a = [-1.0f32, -5.0, 3.0, -7.0];
        let b = [-4.0f32, -2.0, -6.0, -1.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_max_f32(&a, &b, &mut out) };
        assert_eq!(out, [-1.0, -2.0, 3.0, -1.0]);
    }

    #[test]
    fn test_clamp_basic() {
        let a = [-2.0f32, 0.5, 1.5, 3.0, 0.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_clamp_f32(&a, 0.0, 1.0, &mut out) };
        assert_eq!(out, [0.0, 0.5, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_clamp_wide_range() {
        let a = [100.0f32, -100.0, 50.0, -50.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_clamp_f32(&a, -10.0, 10.0, &mut out) };
        assert_eq!(out, [10.0, -10.0, 10.0, -10.0]);
    }

    #[test]
    fn test_clamp_all_within() {
        let a = [0.1f32, 0.5, 0.9, 0.3];
        let mut out = [0.0f32; 4];
        unsafe { neon_clamp_f32(&a, 0.0, 1.0, &mut out) };
        assert_eq!(out, [0.1, 0.5, 0.9, 0.3]);
    }

    // -----------------------------------------------------------------------
    // Abs / Sign / Neg
    // -----------------------------------------------------------------------

    #[test]
    fn test_abs_basic() {
        let a = [-1.0f32, 2.0, -3.0, 4.0, -5.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_abs_f32(&a, &mut out) };
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_abs_zeros() {
        let a = [0.0f32, -0.0, 0.0, -0.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_abs_f32(&a, &mut out) };
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_abs_all_positive() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_abs_f32(&a, &mut out) };
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sign_basic() {
        let a = [-3.0f32, 0.0, 5.0, -1.0, 0.0, 2.0];
        let mut out = [0.0f32; 6];
        unsafe { neon_sign_f32(&a, &mut out) };
        assert_eq!(out, [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sign_all_positive() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_sign_f32(&a, &mut out) };
        assert_eq!(out, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sign_all_negative() {
        let a = [-1.0f32, -2.0, -3.0, -4.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_sign_f32(&a, &mut out) };
        assert_eq!(out, [-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_neg_basic() {
        let a = [1.0f32, -2.0, 3.0, -4.0, 5.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_neg_f32(&a, &mut out) };
        assert_eq!(out, [-1.0, 2.0, -3.0, 4.0, -5.0]);
    }

    #[test]
    fn test_neg_zeros() {
        let a = [0.0f32, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_neg_f32(&a, &mut out) };
        for &v in &out {
            assert!(v == 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // Sqrt / Recip / Rsqrt / Pow / Square
    // -----------------------------------------------------------------------

    #[test]
    fn test_sqrt_basic() {
        let a = [1.0f32, 4.0, 9.0, 16.0, 25.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_sqrt_f32(&a, &mut out) };
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sqrt_zero() {
        let a = [0.0f32, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_sqrt_f32(&a, &mut out) };
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_recip_basic() {
        let a = [1.0f32, 2.0, 4.0, 5.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_recip_f32(&a, &mut out) };
        let expected = [1.0f32, 0.5, 0.25, 0.2];
        assert_slices_approx(&out, &expected, 1e-5);
    }

    #[test]
    fn test_recip_remainder() {
        let a = [1.0f32, 2.0, 4.0, 5.0, 10.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_recip_f32(&a, &mut out) };
        let expected = [1.0, 0.5, 0.25, 0.2, 0.1];
        assert_slices_approx(&out, &expected, 1e-5);
    }

    #[test]
    fn test_rsqrt_basic() {
        let a = [1.0f32, 4.0, 9.0, 16.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_rsqrt_f32(&a, &mut out) };
        let expected = [1.0, 0.5, 1.0 / 3.0, 0.25];
        assert_slices_approx(&out, &expected, 1e-3);
    }

    #[test]
    fn test_rsqrt_remainder() {
        let a = [1.0f32, 4.0, 9.0, 16.0, 25.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_rsqrt_f32(&a, &mut out) };
        let expected = [1.0, 0.5, 1.0 / 3.0, 0.25, 0.2];
        assert_slices_approx(&out, &expected, 1e-3);
    }

    #[test]
    fn test_pow_square() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_pow_f32(&a, 2.0, &mut out) };
        let expected = [1.0, 4.0, 9.0, 16.0, 25.0];
        assert_slices_approx(&out, &expected, 1e-5);
    }

    #[test]
    fn test_pow_half() {
        let a = [1.0f32, 4.0, 9.0, 16.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_pow_f32(&a, 0.5, &mut out) };
        let expected = [1.0, 2.0, 3.0, 4.0];
        assert_slices_approx(&out, &expected, 1e-5);
    }

    #[test]
    fn test_pow_zero() {
        let a = [2.0f32, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_pow_f32(&a, 0.0, &mut out) };
        assert_eq!(out, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_square_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_square_f32(&a, &mut out) };
        assert_eq!(out, [1.0, 4.0, 9.0, 16.0, 25.0]);
    }

    #[test]
    fn test_square_negative() {
        let a = [-1.0f32, -2.0, -3.0, -4.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_square_f32(&a, &mut out) };
        assert_eq!(out, [1.0, 4.0, 9.0, 16.0]);
    }

    // -----------------------------------------------------------------------
    // Comparisons
    // -----------------------------------------------------------------------

    #[test]
    fn test_cmpgt_basic() {
        let a = [5.0f32, 1.0, 3.0, 4.0, 2.0];
        let b = [3.0f32, 2.0, 3.0, 1.0, 5.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_cmpgt_f32(&a, &b, &mut out) };
        assert_eq!(out, [1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_cmpgt_equal_values() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_cmpgt_f32(&a, &b, &mut out) };
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cmplt_basic() {
        let a = [1.0f32, 5.0, 3.0, 0.0, 2.0];
        let b = [3.0f32, 2.0, 3.0, 1.0, 5.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_cmplt_f32(&a, &b, &mut out) };
        assert_eq!(out, [1.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_cmplt_all_greater() {
        let a = [5.0f32, 6.0, 7.0, 8.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_cmplt_f32(&a, &b, &mut out) };
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cmpeq_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 0.0, 3.0, 0.0, 5.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_cmpeq_f32(&a, &b, &mut out) };
        assert_eq!(out, [1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_cmpeq_all_equal() {
        let a = [7.0f32, 7.0, 7.0, 7.0];
        let b = [7.0f32, 7.0, 7.0, 7.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_cmpeq_f32(&a, &b, &mut out) };
        assert_eq!(out, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_cmpne_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 0.0, 3.0, 0.0, 5.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_cmpne_f32(&a, &b, &mut out) };
        assert_eq!(out, [0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_cmpne_all_different() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_cmpne_f32(&a, &b, &mut out) };
        assert_eq!(out, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_cmpge_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0f32, 2.0, 1.0, 5.0, 5.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_cmpge_f32(&a, &b, &mut out) };
        assert_eq!(out, [0.0, 1.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_cmpge_all_equal() {
        let a = [3.0f32, 3.0, 3.0, 3.0];
        let b = [3.0f32, 3.0, 3.0, 3.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_cmpge_f32(&a, &b, &mut out) };
        assert_eq!(out, [1.0, 1.0, 1.0, 1.0]);
    }

    // -----------------------------------------------------------------------
    // Conditional select
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_basic() {
        let mask = [1.0f32, 0.0, 1.0, 0.0, 1.0];
        let a = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_select_f32(&mask, &a, &b, &mut out) };
        assert_eq!(out, [10.0, 2.0, 30.0, 4.0, 50.0]);
    }

    #[test]
    fn test_select_all_true() {
        let mask = [1.0f32, 1.0, 1.0, 1.0];
        let a = [10.0f32, 20.0, 30.0, 40.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_select_f32(&mask, &a, &b, &mut out) };
        assert_eq!(out, [10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_select_all_false() {
        let mask = [0.0f32, 0.0, 0.0, 0.0];
        let a = [10.0f32, 20.0, 30.0, 40.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_select_f32(&mask, &a, &b, &mut out) };
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_select_with_cmpgt_mask() {
        let x = [5.0f32, 1.0, 3.0, 7.0];
        let threshold = [3.0f32, 3.0, 3.0, 3.0];
        let mut mask = [0.0f32; 4];
        unsafe { neon_cmpgt_f32(&x, &threshold, &mut mask) };

        let a = [100.0f32, 100.0, 100.0, 100.0];
        let b = [0.0f32, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_select_f32(&mask, &a, &b, &mut out) };
        assert_eq!(out, [100.0, 0.0, 0.0, 100.0]);
    }

    // -----------------------------------------------------------------------
    // Reductions
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_sum_basic() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let result = unsafe { neon_reduce_sum_f32(&data) };
        assert!(approx_eq(result, 15.0, 1e-6));
    }

    #[test]
    fn test_reduce_sum_single() {
        let data = [42.0f32];
        let result = unsafe { neon_reduce_sum_f32(&data) };
        assert!(approx_eq(result, 42.0, 1e-6));
    }

    #[test]
    fn test_reduce_sum_empty() {
        let data: &[f32] = &[];
        let result = unsafe { neon_reduce_sum_f32(data) };
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_reduce_sum_large() {
        let data: Vec<f32> = (1..=1000).map(|i| i as f32).collect();
        let result = unsafe { neon_reduce_sum_f32(&data) };
        let expected = 500500.0f32;
        assert!(approx_eq(result, expected, 1.0));
    }

    #[test]
    fn test_reduce_max_basic() {
        let data = [1.0f32, 5.0, 3.0, 2.0, 4.0];
        let result = unsafe { neon_reduce_max_f32(&data) };
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_reduce_max_negative() {
        let data = [-3.0f32, -1.0, -4.0, -2.0];
        let result = unsafe { neon_reduce_max_f32(&data) };
        assert_eq!(result, -1.0);
    }

    #[test]
    fn test_reduce_max_empty() {
        let data: &[f32] = &[];
        let result = unsafe { neon_reduce_max_f32(data) };
        assert_eq!(result, f32::NEG_INFINITY);
    }

    #[test]
    fn test_reduce_max_single() {
        let data = [99.0f32];
        let result = unsafe { neon_reduce_max_f32(&data) };
        assert_eq!(result, 99.0);
    }

    #[test]
    fn test_reduce_min_basic() {
        let data = [3.0f32, 1.0, 4.0, 2.0, 5.0];
        let result = unsafe { neon_reduce_min_f32(&data) };
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_reduce_min_negative() {
        let data = [-3.0f32, -1.0, -4.0, -2.0];
        let result = unsafe { neon_reduce_min_f32(&data) };
        assert_eq!(result, -4.0);
    }

    #[test]
    fn test_reduce_min_empty() {
        let data: &[f32] = &[];
        let result = unsafe { neon_reduce_min_f32(data) };
        assert_eq!(result, f32::INFINITY);
    }

    #[test]
    fn test_reduce_min_single() {
        let data = [-42.0f32];
        let result = unsafe { neon_reduce_min_f32(&data) };
        assert_eq!(result, -42.0);
    }

    // -----------------------------------------------------------------------
    // Large-scale correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_add() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let mut out = vec![0.0f32; n];
        unsafe { neon_vec_add_f32(&a, &b, &mut out) };
        for i in 0..n {
            assert_eq!(out[i], (i * 3) as f32, "add mismatch at {i}");
        }
    }

    #[test]
    fn test_large_mul() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![2.0; n];
        let mut out = vec![0.0f32; n];
        unsafe { neon_vec_mul_f32(&a, &b, &mut out) };
        for i in 0..n {
            assert_eq!(out[i], (i * 2) as f32, "mul mismatch at {i}");
        }
    }

    #[test]
    fn test_large_clamp() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32 - 512.0).collect();
        let mut out = vec![0.0f32; n];
        unsafe { neon_clamp_f32(&a, -100.0, 100.0, &mut out) };
        for i in 0..n {
            let v = i as f32 - 512.0;
            let expected = v.max(-100.0).min(100.0);
            assert_eq!(out[i], expected, "clamp mismatch at {i}");
        }
    }

    #[test]
    fn test_large_abs() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) }).collect();
        let mut out = vec![0.0f32; n];
        unsafe { neon_abs_f32(&a, &mut out) };
        for i in 0..n {
            assert_eq!(out[i], i as f32, "abs mismatch at {i}");
        }
    }

    #[test]
    fn test_large_select() {
        let n = 1024;
        let mask: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let a: Vec<f32> = vec![100.0; n];
        let b: Vec<f32> = vec![0.0; n];
        let mut out = vec![0.0f32; n];
        unsafe { neon_select_f32(&mask, &a, &b, &mut out) };
        for i in 0..n {
            let expected = if i % 2 == 0 { 100.0 } else { 0.0 };
            assert_eq!(out[i], expected, "select mismatch at {i}");
        }
    }

    #[test]
    fn test_large_cmpgt() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![512.0; n];
        let mut out = vec![0.0f32; n];
        unsafe { neon_cmpgt_f32(&a, &b, &mut out) };
        for i in 0..n {
            let expected = if (i as f32) > 512.0 { 1.0 } else { 0.0 };
            assert_eq!(out[i], expected, "cmpgt mismatch at {i}");
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases: single element
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_element_ops() {
        let a = [3.0f32];
        let b = [2.0f32];
        let mut out = [0.0f32; 1];

        unsafe { neon_vec_add_f32(&a, &b, &mut out) };
        assert_eq!(out[0], 5.0);

        unsafe { neon_vec_sub_f32(&a, &b, &mut out) };
        assert_eq!(out[0], 1.0);

        unsafe { neon_vec_mul_f32(&a, &b, &mut out) };
        assert_eq!(out[0], 6.0);

        unsafe { neon_vec_div_f32(&a, &b, &mut out) };
        assert_eq!(out[0], 1.5);

        unsafe { neon_abs_f32(&[-7.0], &mut out) };
        assert_eq!(out[0], 7.0);

        unsafe { neon_sqrt_f32(&[9.0], &mut out) };
        assert_eq!(out[0], 3.0);
    }

    // -----------------------------------------------------------------------
    // Chained operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_chained_fma_clamp() {
        let a = [10.0f32, 20.0, 30.0, 40.0];
        let b = [2.0f32, 3.0, 4.0, 5.0];
        let c = [-10.0f32, -20.0, -30.0, -40.0];
        let mut fma_out = [0.0f32; 4];
        unsafe { neon_fma_f32(&a, &b, &c, &mut fma_out) };
        // [10, 40, 90, 160]

        let mut clamped = [0.0f32; 4];
        unsafe { neon_clamp_f32(&fma_out, 0.0, 100.0, &mut clamped) };
        assert_eq!(clamped, [10.0, 40.0, 90.0, 100.0]);
    }

    #[test]
    fn test_chained_abs_sqrt() {
        let a = [-4.0f32, -9.0, -16.0, -25.0];
        let mut abs_out = [0.0f32; 4];
        unsafe { neon_abs_f32(&a, &mut abs_out) };

        let mut sqrt_out = [0.0f32; 4];
        unsafe { neon_sqrt_f32(&abs_out, &mut sqrt_out) };
        assert_eq!(sqrt_out, [2.0, 3.0, 4.0, 5.0]);
    }
}
