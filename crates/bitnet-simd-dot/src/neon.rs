//! NEON SIMD implementations (aarch64).

#[allow(clippy::wildcard_imports)]
use std::arch::aarch64::*;

// ── NEON f32 dot product ────────────────────────────────────────────

#[target_feature(enable = "neon")]
pub unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let n = a.len();
        let mut acc = vdupq_n_f32(0.0);
        let chunks = n / 4;
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, va, vb);
        }
        let mut sum = vaddvq_f32(acc);
        for i in (chunks * 4)..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

// ── NEON i8 dot product ─────────────────────────────────────────────

#[target_feature(enable = "neon")]
pub unsafe fn dot_i8_neon(a: &[i8], b: &[i8]) -> i32 {
    unsafe {
        let n = a.len();
        let mut acc = vdupq_n_s32(0);
        let chunks = n / 16;
        for i in 0..chunks {
            let va = vld1q_s8(a.as_ptr().add(i * 16));
            let vb = vld1q_s8(b.as_ptr().add(i * 16));
            let a_lo = vmovl_s8(vget_low_s8(va));
            let b_lo = vmovl_s8(vget_low_s8(vb));
            let a_hi = vmovl_s8(vget_high_s8(va));
            let b_hi = vmovl_s8(vget_high_s8(vb));
            acc = vmlal_s16(acc, vget_low_s16(a_lo), vget_low_s16(b_lo));
            acc = vmlal_s16(acc, vget_high_s16(a_lo), vget_high_s16(b_lo));
            acc = vmlal_s16(acc, vget_low_s16(a_hi), vget_low_s16(b_hi));
            acc = vmlal_s16(acc, vget_high_s16(a_hi), vget_high_s16(b_hi));
        }
        let mut sum = vaddvq_s32(acc);
        for i in (chunks * 16)..n {
            sum += i32::from(a[i]) * i32::from(b[i]);
        }
        sum
    }
}

// ── NEON fused multiply-accumulate dot ──────────────────────────────

#[target_feature(enable = "neon")]
pub unsafe fn fma_dot_f32_neon(a: &[f32], b: &[f32], c: &[f32], d: &[f32]) -> f32 {
    unsafe {
        let n_ab = a.len();
        let n_cd = c.len();
        let mut acc = vdupq_n_f32(0.0);

        let chunks_ab = n_ab / 4;
        for i in 0..chunks_ab {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, va, vb);
        }
        let chunks_cd = n_cd / 4;
        for i in 0..chunks_cd {
            let vc = vld1q_f32(c.as_ptr().add(i * 4));
            let vd = vld1q_f32(d.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, vc, vd);
        }

        let mut sum = vaddvq_f32(acc);
        for i in (chunks_ab * 4)..n_ab {
            sum = a[i].mul_add(b[i], sum);
        }
        for i in (chunks_cd * 4)..n_cd {
            sum = c[i].mul_add(d[i], sum);
        }
        sum
    }
}
