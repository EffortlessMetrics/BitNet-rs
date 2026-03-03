//! Dot-product kernels with AVX-512 fast paths and scalar fallbacks.
//!
//! These are the low-level building blocks used by the GEMV routines.

// ── f32 dot product ────────────────────────────────────────────────────

/// Compute the dot product of two `f32` slices using AVX-512 when available.
///
/// Falls back to scalar arithmetic on platforms without AVX-512.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[must_use]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_f32: length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: feature check passed; slices are valid for their length.
            return unsafe { dot_f32_avx512(a, b) };
        }
    }
    dot_f32_scalar(a, b)
}

/// Scalar fallback for `f32` dot product.
#[must_use]
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// AVX-512 accelerated `f32` dot product.
///
/// # Safety
///
/// Caller must ensure the CPU supports AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps,
    };

    unsafe {
        let n = a.len();
        let chunks = n / 16;
        let remainder = n % 16;

        let mut acc = _mm512_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 16;
            let va = _mm512_loadu_ps(a_ptr.add(offset));
            let vb = _mm512_loadu_ps(b_ptr.add(offset));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);

        let tail_start = chunks * 16;
        for j in 0..remainder {
            sum += a[tail_start + j] * b[tail_start + j];
        }

        sum
    }
}

// ── i32 (from i8) dot product ──────────────────────────────────────────

/// Compute the dot product of two `i8` slices, returning `i32`.
///
/// Uses AVX-512 when available; otherwise scalar.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[must_use]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "dot_i8: length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: feature check passed.
            return unsafe { dot_i8_avx512(a, b) };
        }
    }
    dot_i8_scalar(a, b)
}

/// Scalar fallback for `i8` dot product.
#[must_use]
pub fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| i32::from(x) * i32::from(y)).sum()
}

/// AVX-512 accelerated `i8` dot product.
///
/// Widens `i8` to `i32` in 16-element chunks and uses FMA-style accumulation.
///
/// # Safety
///
/// Caller must ensure AVX-512F is supported.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_i8_avx512(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::{
        _mm512_add_epi32, _mm512_mullo_epi32, _mm512_reduce_add_epi32, _mm512_setzero_si512,
    };

    unsafe {
        let n = a.len();
        let chunks = n / 16;
        let remainder = n % 16;

        let mut acc = _mm512_setzero_si512();

        for i in 0..chunks {
            let offset = i * 16;
            let va = widen_i8x16_to_i32x16(a.as_ptr().add(offset));
            let vb = widen_i8x16_to_i32x16(b.as_ptr().add(offset));
            let prod = _mm512_mullo_epi32(va, vb);
            acc = _mm512_add_epi32(acc, prod);
        }

        let mut sum = _mm512_reduce_add_epi32(acc);

        let tail_start = chunks * 16;
        for j in 0..remainder {
            sum += i32::from(a[tail_start + j]) * i32::from(b[tail_start + j]);
        }

        sum
    }
}

/// Load 16 × i8 from `ptr` and sign-extend each to i32, returning a `__m512i`.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available and `ptr` is valid for 16 reads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn widen_i8x16_to_i32x16(ptr: *const i8) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::_mm512_loadu_si512;

    unsafe {
        let mut buf = [0i32; 16];
        for (k, slot) in buf.iter_mut().enumerate() {
            *slot = i32::from(*ptr.add(k));
        }
        _mm512_loadu_si512(buf.as_ptr().cast())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── f32 dot ────────────────────────────────────────────────────────

    #[test]
    fn dot_f32_empty() {
        assert!((dot_f32(&[], &[]) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn dot_f32_single() {
        assert!((dot_f32(&[3.0], &[4.0]) - 12.0).abs() < f32::EPSILON);
    }

    #[test]
    fn dot_f32_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_f32(&a, &b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn dot_f32_large_aligned() {
        let n = 64;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; n];
        let expected: f32 = (0..n).map(|i| i as f32).sum();
        assert!((dot_f32(&a, &b) - expected).abs() < 1e-3);
    }

    #[test]
    fn dot_f32_large_unaligned() {
        let n = 67; // not a multiple of 16
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; n];
        let expected: f32 = (0..n).map(|i| i as f32).sum();
        assert!((dot_f32(&a, &b) - expected).abs() < 1e-2);
    }

    #[test]
    fn dot_f32_negative() {
        let a = vec![-1.0, -2.0, -3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_f32(&a, &b) - (-32.0)).abs() < 1e-5);
    }

    #[test]
    fn dot_f32_scalar_matches_dispatch() {
        let a: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..100).map(|i| (100 - i) as f32 * 0.1).collect();
        let scalar = dot_f32_scalar(&a, &b);
        let dispatched = dot_f32(&a, &b);
        assert!((scalar - dispatched).abs() < 1e-2);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn dot_f32_length_mismatch() {
        let _ = dot_f32(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn dot_f32_zeros() {
        let a = vec![0.0_f32; 32];
        let b = vec![1.0_f32; 32];
        assert!((dot_f32(&a, &b)).abs() < f32::EPSILON);
    }

    #[test]
    fn dot_f32_ones() {
        let n = 48;
        let a = vec![1.0_f32; n];
        let b = vec![1.0_f32; n];
        assert!((dot_f32(&a, &b) - n as f32).abs() < f32::EPSILON);
    }

    // ── i8 dot ─────────────────────────────────────────────────────────

    #[test]
    fn dot_i8_empty() {
        assert_eq!(dot_i8(&[], &[]), 0);
    }

    #[test]
    fn dot_i8_single() {
        assert_eq!(dot_i8(&[3], &[4]), 12);
    }

    #[test]
    fn dot_i8_basic() {
        let a: Vec<i8> = vec![1, 2, 3];
        let b: Vec<i8> = vec![4, 5, 6];
        assert_eq!(dot_i8(&a, &b), 32);
    }

    #[test]
    fn dot_i8_negative() {
        let a: Vec<i8> = vec![-1, -2, -3];
        let b: Vec<i8> = vec![4, 5, 6];
        assert_eq!(dot_i8(&a, &b), -32);
    }

    #[test]
    fn dot_i8_large_aligned() {
        let n = 64;
        let a: Vec<i8> = (0..n).map(|i| (i % 5) as i8).collect();
        let b: Vec<i8> = vec![1; n];
        let expected: i32 = (0..n).map(|i| (i % 5) as i32).sum();
        assert_eq!(dot_i8(&a, &b), expected);
    }

    #[test]
    fn dot_i8_large_unaligned() {
        let n = 67;
        let a: Vec<i8> = (0..n).map(|i| (i % 3) as i8).collect();
        let b: Vec<i8> = vec![2; n];
        let expected: i32 = (0..n).map(|i| ((i % 3) * 2) as i32).sum();
        assert_eq!(dot_i8(&a, &b), expected);
    }

    #[test]
    fn dot_i8_scalar_matches_dispatch() {
        let a: Vec<i8> = (0..100).map(|i| ((i % 127) - 63) as i8).collect();
        let b: Vec<i8> = (0..100).map(|i| ((i % 127) - 63) as i8).collect();
        assert_eq!(dot_i8_scalar(&a, &b), dot_i8(&a, &b));
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn dot_i8_length_mismatch() {
        let _ = dot_i8(&[1, 2], &[1]);
    }

    #[test]
    fn dot_i8_max_values() {
        let a: Vec<i8> = vec![127; 4];
        let b: Vec<i8> = vec![127; 4];
        assert_eq!(dot_i8(&a, &b), 127 * 127 * 4);
    }

    #[test]
    fn dot_i8_min_values() {
        let a: Vec<i8> = vec![-128; 4];
        let b: Vec<i8> = vec![1; 4];
        assert_eq!(dot_i8(&a, &b), -128 * 4);
    }
}
