//! ARM NEON vectorized elementwise tensor operations for Apple Silicon.
//!
//! Provides `f32` elementwise add, multiply, scale, and fused multiply-add
//! using NEON intrinsics, with scalar fallback for remainder elements.

use std::arch::aarch64::*;

/// Elementwise addition: `out[i] = a[i] + b[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[target_feature(enable = "neon")]
pub unsafe fn neon_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            let vr = vaddq_f32(va, vb);
            vst1q_f32(o_ptr.add(offset), vr);
        }
    }

    for i in (chunks * 4)..n {
        out[i] = a[i] + b[i];
    }
}

/// Elementwise multiplication: `out[i] = a[i] * b[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[target_feature(enable = "neon")]
pub unsafe fn neon_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            let vr = vmulq_f32(va, vb);
            vst1q_f32(o_ptr.add(offset), vr);
        }
    }

    for i in (chunks * 4)..n {
        out[i] = a[i] * b[i];
    }
}

/// Scalar multiplication: `out[i] = a[i] * scale`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[target_feature(enable = "neon")]
pub unsafe fn neon_scale_f32(a: &[f32], scale: f32, out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let vs = vdupq_n_f32(scale);

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(offset));
            let vr = vmulq_f32(va, vs);
            vst1q_f32(o_ptr.add(offset), vr);
        }
    }

    for i in (chunks * 4)..n {
        out[i] = a[i] * scale;
    }
}

/// Fused multiply-add: `out[i] = a[i] * b[i] + c[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[target_feature(enable = "neon")]
pub unsafe fn neon_fma_f32(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            let vc = vld1q_f32(c_ptr.add(offset));
            // vfmaq_f32(c, a, b) computes a*b + c
            let vr = vfmaq_f32(vc, va, vb);
            vst1q_f32(o_ptr.add(offset), vr);
        }
    }

    for i in (chunks * 4)..n {
        out[i] = a[i].mul_add(b[i], c[i]);
    }
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    #[test]
    fn test_add_exact() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        unsafe { neon_add_f32(&a, &b, &mut out) };
        assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_add_remainder() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let mut out = [0.0f32; 7];
        unsafe { neon_add_f32(&a, &b, &mut out) };
        assert_eq!(out, [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0]);
    }

    #[test]
    fn test_mul_basic() {
        let a = [2.0f32, 3.0, 4.0, 5.0, 6.0];
        let b = [0.5f32, 1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_mul_f32(&a, &b, &mut out) };
        assert_eq!(out, [1.0, 3.0, 8.0, 15.0, 24.0]);
    }

    #[test]
    fn test_scale_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_scale_f32(&a, 3.0, &mut out) };
        assert_eq!(out, [3.0, 6.0, 9.0, 12.0, 15.0]);
    }

    #[test]
    fn test_fma_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0f32, 3.0, 4.0, 5.0, 6.0];
        let c = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut out = [0.0f32; 5];
        unsafe { neon_fma_f32(&a, &b, &c, &mut out) };
        // a*b+c = [12, 26, 42, 60, 80]
        assert_eq!(out, [12.0, 26.0, 42.0, 60.0, 80.0]);
    }

    #[test]
    fn test_empty_slices() {
        let empty: &[f32] = &[];
        let mut out = vec![];
        unsafe { neon_add_f32(empty, empty, &mut out) };
        assert!(out.is_empty());

        unsafe { neon_mul_f32(empty, empty, &mut out) };
        assert!(out.is_empty());

        unsafe { neon_scale_f32(empty, 2.0, &mut out) };
        assert!(out.is_empty());

        unsafe { neon_fma_f32(empty, empty, empty, &mut out) };
        assert!(out.is_empty());
    }

    #[test]
    fn test_large() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let mut out = vec![0.0f32; n];

        unsafe { neon_add_f32(&a, &b, &mut out) };
        for i in 0..n {
            assert_eq!(out[i], (i + i * 2) as f32, "mismatch at index {i}");
        }

        unsafe { neon_mul_f32(&a, &b, &mut out) };
        for i in 0..n {
            assert_eq!(out[i], (i * i * 2) as f32, "mismatch at index {i}");
        }

        unsafe { neon_scale_f32(&a, 0.5, &mut out) };
        for i in 0..n {
            assert_eq!(out[i], i as f32 * 0.5, "mismatch at index {i}");
        }

        let c: Vec<f32> = vec![1.0; n];
        unsafe { neon_fma_f32(&a, &b, &c, &mut out) };
        for i in 0..n {
            let expected = (i as f32).mul_add((i * 2) as f32, 1.0);
            assert_eq!(out[i], expected, "mismatch at index {i}");
        }
    }
}
