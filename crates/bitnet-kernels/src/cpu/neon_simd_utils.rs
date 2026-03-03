//! NEON SIMD utility functions for common operations on Apple Silicon / ARM64.
//!
//! Provides zero-overhead wrappers around ARM NEON intrinsics for horizontal
//! reductions, broadcasts, partial loads/stores, fused multiply-add, clamping,
//! conditional select, and interleave/deinterleave operations.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Horizontal sum of all 4 lanes of a `float32x4_t`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_horizontal_sum_f32(v: float32x4_t) -> f32 {
    unsafe { vaddvq_f32(v) }
}

/// Horizontal maximum of all 4 lanes of a `float32x4_t`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_horizontal_max_f32(v: float32x4_t) -> f32 {
    unsafe { vmaxvq_f32(v) }
}

/// Horizontal minimum of all 4 lanes of a `float32x4_t`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_horizontal_min_f32(v: float32x4_t) -> f32 {
    unsafe { vminvq_f32(v) }
}

/// Broadcast a scalar `f32` to all 4 lanes of a `float32x4_t`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_broadcast_f32(val: f32) -> float32x4_t {
    unsafe { vdupq_n_f32(val) }
}

/// Load 1–3 `f32` elements from `ptr`, zero-padding the remaining lanes.
///
/// If `count` is 0 all lanes are zero. If `count >= 4` a full `vld1q_f32` is
/// performed. Values of 1, 2, or 3 load exactly that many elements.
///
/// # Safety
///
/// `ptr` must be valid for reads of `min(count, 4)` contiguous `f32` values.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_load_partial_f32(ptr: *const f32, count: usize) -> float32x4_t {
    unsafe {
        match count {
            0 => vdupq_n_f32(0.0),
            1 => {
                let mut arr = [0.0f32; 4];
                arr[0] = *ptr;
                vld1q_f32(arr.as_ptr())
            }
            2 => {
                let mut arr = [0.0f32; 4];
                arr[0] = *ptr;
                arr[1] = *ptr.add(1);
                vld1q_f32(arr.as_ptr())
            }
            3 => {
                let mut arr = [0.0f32; 4];
                arr[0] = *ptr;
                arr[1] = *ptr.add(1);
                arr[2] = *ptr.add(2);
                vld1q_f32(arr.as_ptr())
            }
            _ => vld1q_f32(ptr),
        }
    }
}

/// Store 1–3 `f32` elements from `v` to `ptr`.
///
/// If `count` is 0 nothing is stored. If `count >= 4` a full `vst1q_f32` is
/// performed.
///
/// # Safety
///
/// `ptr` must be valid for writes of `min(count, 4)` contiguous `f32` values.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_store_partial_f32(ptr: *mut f32, v: float32x4_t, count: usize) {
    unsafe {
        match count {
            0 => {}
            1 => {
                let mut arr = [0.0f32; 4];
                vst1q_f32(arr.as_mut_ptr(), v);
                *ptr = arr[0];
            }
            2 => {
                let mut arr = [0.0f32; 4];
                vst1q_f32(arr.as_mut_ptr(), v);
                *ptr = arr[0];
                *ptr.add(1) = arr[1];
            }
            3 => {
                let mut arr = [0.0f32; 4];
                vst1q_f32(arr.as_mut_ptr(), v);
                *ptr = arr[0];
                *ptr.add(1) = arr[1];
                *ptr.add(2) = arr[2];
            }
            _ => vst1q_f32(ptr, v),
        }
    }
}

/// Fused multiply-add: `a + b * c`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_fma_f32(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> float32x4_t {
    unsafe { vfmaq_f32(a, b, c) }
}

/// Per-lane absolute value.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_abs_f32(v: float32x4_t) -> float32x4_t {
    unsafe { vabsq_f32(v) }
}

/// Per-lane clamp: `max(min_val, min(v, max_val))`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_clamp_f32(
    v: float32x4_t,
    min: float32x4_t,
    max: float32x4_t,
) -> float32x4_t {
    unsafe { vmaxq_f32(min, vminq_f32(v, max)) }
}

/// Bitwise select: for each bit position, choose from `a` where `mask` is 1,
/// from `b` where `mask` is 0.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_select_f32(
    mask: uint32x4_t,
    a: float32x4_t,
    b: float32x4_t,
) -> float32x4_t {
    unsafe { vbslq_f32(mask, a, b) }
}

/// Interleave pairs from two vectors.
///
/// Returns `(low, high)` where `low = [a0, b0, a1, b1]` and
/// `high = [a2, b2, a3, b3]`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_interleave_f32(
    a: float32x4_t,
    b: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    unsafe {
        let lo = vzip1q_f32(a, b);
        let hi = vzip2q_f32(a, b);
        (lo, hi)
    }
}

/// Deinterleave pairs from two vectors.
///
/// Returns `(even, odd)` where `even = [a0, a2, b0, b2]` and
/// `odd = [a1, a3, b1, b3]`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_deinterleave_f32(
    a: float32x4_t,
    b: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    unsafe {
        let even = vuzp1q_f32(a, b);
        let odd = vuzp2q_f32(a, b);
        (even, odd)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    // Helper: create a float32x4_t from an array.
    unsafe fn f32x4(a: f32, b: f32, c: f32, d: f32) -> float32x4_t {
        unsafe {
            let arr = [a, b, c, d];
            vld1q_f32(arr.as_ptr())
        }
    }

    // Helper: extract lanes to an array.
    unsafe fn to_arr(v: float32x4_t) -> [f32; 4] {
        unsafe {
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), v);
            out
        }
    }

    // ---- horizontal_sum ----

    #[test]
    fn test_horizontal_sum_basic() {
        unsafe {
            let v = f32x4(1.0, 2.0, 3.0, 4.0);
            assert_eq!(neon_horizontal_sum_f32(v), 10.0);
        }
    }

    #[test]
    fn test_horizontal_sum_zeros() {
        unsafe {
            let v = f32x4(0.0, 0.0, 0.0, 0.0);
            assert_eq!(neon_horizontal_sum_f32(v), 0.0);
        }
    }

    #[test]
    fn test_horizontal_sum_negatives() {
        unsafe {
            let v = f32x4(-1.0, -2.0, -3.0, -4.0);
            assert_eq!(neon_horizontal_sum_f32(v), -10.0);
        }
    }

    #[test]
    fn test_horizontal_sum_mixed() {
        unsafe {
            let v = f32x4(1.0, -1.0, 2.0, -2.0);
            assert_eq!(neon_horizontal_sum_f32(v), 0.0);
        }
    }

    #[test]
    fn test_horizontal_sum_large() {
        unsafe {
            let v = f32x4(1e30, 1e30, 1e30, 1e30);
            assert_eq!(neon_horizontal_sum_f32(v), 4e30);
        }
    }

    #[test]
    fn test_horizontal_sum_subnormal() {
        unsafe {
            let tiny = f32::MIN_POSITIVE * 0.5;
            let v = f32x4(tiny, tiny, 0.0, 0.0);
            let s = neon_horizontal_sum_f32(v);
            assert!(s > 0.0);
        }
    }

    #[test]
    fn test_horizontal_sum_inf() {
        unsafe {
            let v = f32x4(f32::INFINITY, 1.0, 2.0, 3.0);
            assert_eq!(neon_horizontal_sum_f32(v), f32::INFINITY);
        }
    }

    #[test]
    fn test_horizontal_sum_nan() {
        unsafe {
            let v = f32x4(f32::NAN, 1.0, 2.0, 3.0);
            assert!(neon_horizontal_sum_f32(v).is_nan());
        }
    }

    // ---- horizontal_max ----

    #[test]
    fn test_horizontal_max_basic() {
        unsafe {
            let v = f32x4(1.0, 4.0, 2.0, 3.0);
            assert_eq!(neon_horizontal_max_f32(v), 4.0);
        }
    }

    #[test]
    fn test_horizontal_max_all_same() {
        unsafe {
            let v = f32x4(5.0, 5.0, 5.0, 5.0);
            assert_eq!(neon_horizontal_max_f32(v), 5.0);
        }
    }

    #[test]
    fn test_horizontal_max_negatives() {
        unsafe {
            let v = f32x4(-1.0, -4.0, -2.0, -3.0);
            assert_eq!(neon_horizontal_max_f32(v), -1.0);
        }
    }

    #[test]
    fn test_horizontal_max_mixed() {
        unsafe {
            let v = f32x4(-100.0, 0.0, 50.0, -50.0);
            assert_eq!(neon_horizontal_max_f32(v), 50.0);
        }
    }

    #[test]
    fn test_horizontal_max_inf() {
        unsafe {
            let v = f32x4(1.0, f32::INFINITY, 2.0, 3.0);
            assert_eq!(neon_horizontal_max_f32(v), f32::INFINITY);
        }
    }

    #[test]
    fn test_horizontal_max_neg_inf() {
        unsafe {
            let v = f32x4(f32::NEG_INFINITY, -1.0, -2.0, -3.0);
            assert_eq!(neon_horizontal_max_f32(v), -1.0);
        }
    }

    #[test]
    fn test_horizontal_max_zeros() {
        unsafe {
            let v = f32x4(0.0, 0.0, 0.0, 0.0);
            assert_eq!(neon_horizontal_max_f32(v), 0.0);
        }
    }

    // ---- horizontal_min ----

    #[test]
    fn test_horizontal_min_basic() {
        unsafe {
            let v = f32x4(1.0, 4.0, 2.0, 3.0);
            assert_eq!(neon_horizontal_min_f32(v), 1.0);
        }
    }

    #[test]
    fn test_horizontal_min_all_same() {
        unsafe {
            let v = f32x4(7.0, 7.0, 7.0, 7.0);
            assert_eq!(neon_horizontal_min_f32(v), 7.0);
        }
    }

    #[test]
    fn test_horizontal_min_negatives() {
        unsafe {
            let v = f32x4(-1.0, -4.0, -2.0, -3.0);
            assert_eq!(neon_horizontal_min_f32(v), -4.0);
        }
    }

    #[test]
    fn test_horizontal_min_mixed() {
        unsafe {
            let v = f32x4(-100.0, 0.0, 50.0, -50.0);
            assert_eq!(neon_horizontal_min_f32(v), -100.0);
        }
    }

    #[test]
    fn test_horizontal_min_inf() {
        unsafe {
            let v = f32x4(1.0, f32::NEG_INFINITY, 2.0, 3.0);
            assert_eq!(neon_horizontal_min_f32(v), f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_horizontal_min_pos_inf() {
        unsafe {
            let v = f32x4(f32::INFINITY, 1.0, 2.0, 3.0);
            assert_eq!(neon_horizontal_min_f32(v), 1.0);
        }
    }

    #[test]
    fn test_horizontal_min_zeros() {
        unsafe {
            let v = f32x4(0.0, 0.0, 0.0, 0.0);
            assert_eq!(neon_horizontal_min_f32(v), 0.0);
        }
    }

    // ---- broadcast ----

    #[test]
    fn test_broadcast_positive() {
        unsafe {
            let v = neon_broadcast_f32(3.14);
            let a = to_arr(v);
            assert_eq!(a, [3.14, 3.14, 3.14, 3.14]);
        }
    }

    #[test]
    fn test_broadcast_zero() {
        unsafe {
            let v = neon_broadcast_f32(0.0);
            let a = to_arr(v);
            assert_eq!(a, [0.0; 4]);
        }
    }

    #[test]
    fn test_broadcast_negative() {
        unsafe {
            let v = neon_broadcast_f32(-42.0);
            let a = to_arr(v);
            assert_eq!(a, [-42.0; 4]);
        }
    }

    #[test]
    fn test_broadcast_inf() {
        unsafe {
            let v = neon_broadcast_f32(f32::INFINITY);
            let a = to_arr(v);
            assert_eq!(a, [f32::INFINITY; 4]);
        }
    }

    #[test]
    fn test_broadcast_nan() {
        unsafe {
            let v = neon_broadcast_f32(f32::NAN);
            let a = to_arr(v);
            for lane in a {
                assert!(lane.is_nan());
            }
        }
    }

    // ---- partial load ----

    #[test]
    fn test_load_partial_0() {
        unsafe {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let v = neon_load_partial_f32(data.as_ptr(), 0);
            assert_eq!(to_arr(v), [0.0; 4]);
        }
    }

    #[test]
    fn test_load_partial_1() {
        unsafe {
            let data = [5.0f32, 6.0, 7.0, 8.0];
            let v = neon_load_partial_f32(data.as_ptr(), 1);
            assert_eq!(to_arr(v), [5.0, 0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn test_load_partial_2() {
        unsafe {
            let data = [5.0f32, 6.0, 7.0, 8.0];
            let v = neon_load_partial_f32(data.as_ptr(), 2);
            assert_eq!(to_arr(v), [5.0, 6.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn test_load_partial_3() {
        unsafe {
            let data = [5.0f32, 6.0, 7.0, 8.0];
            let v = neon_load_partial_f32(data.as_ptr(), 3);
            assert_eq!(to_arr(v), [5.0, 6.0, 7.0, 0.0]);
        }
    }

    #[test]
    fn test_load_partial_4_full() {
        unsafe {
            let data = [5.0f32, 6.0, 7.0, 8.0];
            let v = neon_load_partial_f32(data.as_ptr(), 4);
            assert_eq!(to_arr(v), [5.0, 6.0, 7.0, 8.0]);
        }
    }

    #[test]
    fn test_load_partial_negative_values() {
        unsafe {
            let data = [-1.0f32, -2.0, -3.0];
            let v = neon_load_partial_f32(data.as_ptr(), 3);
            assert_eq!(to_arr(v), [-1.0, -2.0, -3.0, 0.0]);
        }
    }

    // ---- partial store ----

    #[test]
    fn test_store_partial_0() {
        unsafe {
            let mut data = [99.0f32; 4];
            let v = f32x4(1.0, 2.0, 3.0, 4.0);
            neon_store_partial_f32(data.as_mut_ptr(), v, 0);
            assert_eq!(data, [99.0; 4]);
        }
    }

    #[test]
    fn test_store_partial_1() {
        unsafe {
            let mut data = [99.0f32; 4];
            let v = f32x4(1.0, 2.0, 3.0, 4.0);
            neon_store_partial_f32(data.as_mut_ptr(), v, 1);
            assert_eq!(data, [1.0, 99.0, 99.0, 99.0]);
        }
    }

    #[test]
    fn test_store_partial_2() {
        unsafe {
            let mut data = [99.0f32; 4];
            let v = f32x4(1.0, 2.0, 3.0, 4.0);
            neon_store_partial_f32(data.as_mut_ptr(), v, 2);
            assert_eq!(data, [1.0, 2.0, 99.0, 99.0]);
        }
    }

    #[test]
    fn test_store_partial_3() {
        unsafe {
            let mut data = [99.0f32; 4];
            let v = f32x4(1.0, 2.0, 3.0, 4.0);
            neon_store_partial_f32(data.as_mut_ptr(), v, 3);
            assert_eq!(data, [1.0, 2.0, 3.0, 99.0]);
        }
    }

    #[test]
    fn test_store_partial_4_full() {
        unsafe {
            let mut data = [99.0f32; 4];
            let v = f32x4(1.0, 2.0, 3.0, 4.0);
            neon_store_partial_f32(data.as_mut_ptr(), v, 4);
            assert_eq!(data, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_load_store_partial_roundtrip() {
        unsafe {
            for count in 0..=4 {
                let src = [10.0f32, 20.0, 30.0, 40.0];
                let v = neon_load_partial_f32(src.as_ptr(), count);
                let mut dst = [0.0f32; 4];
                neon_store_partial_f32(dst.as_mut_ptr(), v, count);
                for i in 0..count.min(4) {
                    assert_eq!(dst[i], src[i], "mismatch at index {i} for count {count}");
                }
            }
        }
    }

    // ---- fma ----

    #[test]
    fn test_fma_basic() {
        unsafe {
            let a = f32x4(1.0, 2.0, 3.0, 4.0);
            let b = f32x4(2.0, 3.0, 4.0, 5.0);
            let c = f32x4(10.0, 10.0, 10.0, 10.0);
            let r = to_arr(neon_fma_f32(a, b, c));
            // a + b*c = [1+20, 2+30, 3+40, 4+50]
            assert_eq!(r, [21.0, 32.0, 43.0, 54.0]);
        }
    }

    #[test]
    fn test_fma_zeros() {
        unsafe {
            let z = f32x4(0.0, 0.0, 0.0, 0.0);
            let a = f32x4(5.0, 5.0, 5.0, 5.0);
            let r = to_arr(neon_fma_f32(a, z, z));
            assert_eq!(r, [5.0, 5.0, 5.0, 5.0]);
        }
    }

    #[test]
    fn test_fma_negative() {
        unsafe {
            let a = f32x4(0.0, 0.0, 0.0, 0.0);
            let b = f32x4(-1.0, -2.0, -3.0, -4.0);
            let c = f32x4(1.0, 1.0, 1.0, 1.0);
            let r = to_arr(neon_fma_f32(a, b, c));
            assert_eq!(r, [-1.0, -2.0, -3.0, -4.0]);
        }
    }

    #[test]
    fn test_fma_accuracy_vs_separate() {
        unsafe {
            let a = f32x4(1.0000001, 2.0000002, 3.0, 4.0);
            let b = f32x4(0.3, 0.7, 1.1, 1.3);
            let c = f32x4(0.9, 0.8, 0.7, 0.6);
            let fma_result = to_arr(neon_fma_f32(a, b, c));
            // Separate: a + b * c computed lane-by-lane
            let expected: [f32; 4] = [
                1.0000001_f32 + 0.3 * 0.9,
                2.0000002 + 0.7 * 0.8,
                3.0 + 1.1 * 0.7,
                4.0 + 1.3 * 0.6,
            ];
            for i in 0..4 {
                let diff = (fma_result[i] - expected[i]).abs();
                assert!(diff < 1e-6, "lane {i}: fma={} sep={} diff={diff}", fma_result[i], expected[i]);
            }
        }
    }

    #[test]
    fn test_fma_identity() {
        unsafe {
            // a + b * 0 == a
            let a = f32x4(7.0, 8.0, 9.0, 10.0);
            let b = f32x4(100.0, 200.0, 300.0, 400.0);
            let z = f32x4(0.0, 0.0, 0.0, 0.0);
            let r = to_arr(neon_fma_f32(a, b, z));
            assert_eq!(r, [7.0, 8.0, 9.0, 10.0]);
        }
    }

    // ---- abs ----

    #[test]
    fn test_abs_positive() {
        unsafe {
            let v = f32x4(1.0, 2.0, 3.0, 4.0);
            assert_eq!(to_arr(neon_abs_f32(v)), [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_abs_negative() {
        unsafe {
            let v = f32x4(-1.0, -2.0, -3.0, -4.0);
            assert_eq!(to_arr(neon_abs_f32(v)), [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_abs_mixed() {
        unsafe {
            let v = f32x4(-5.0, 0.0, 3.0, -7.0);
            assert_eq!(to_arr(neon_abs_f32(v)), [5.0, 0.0, 3.0, 7.0]);
        }
    }

    #[test]
    fn test_abs_zero() {
        unsafe {
            let v = f32x4(0.0, -0.0, 0.0, -0.0);
            let a = to_arr(neon_abs_f32(v));
            for lane in a {
                assert_eq!(lane, 0.0);
            }
        }
    }

    #[test]
    fn test_abs_inf() {
        unsafe {
            let v = f32x4(f32::NEG_INFINITY, f32::INFINITY, -1.0, 1.0);
            let a = to_arr(neon_abs_f32(v));
            assert_eq!(a[0], f32::INFINITY);
            assert_eq!(a[1], f32::INFINITY);
        }
    }

    #[test]
    fn test_abs_nan() {
        unsafe {
            let v = f32x4(f32::NAN, -f32::NAN, 1.0, -1.0);
            let a = to_arr(neon_abs_f32(v));
            assert!(a[0].is_nan());
            assert!(a[1].is_nan());
        }
    }

    // ---- clamp ----

    #[test]
    fn test_clamp_within_range() {
        unsafe {
            let v = f32x4(0.5, 1.5, 2.5, 3.5);
            let lo = neon_broadcast_f32(0.0);
            let hi = neon_broadcast_f32(4.0);
            assert_eq!(to_arr(neon_clamp_f32(v, lo, hi)), [0.5, 1.5, 2.5, 3.5]);
        }
    }

    #[test]
    fn test_clamp_below_min() {
        unsafe {
            let v = f32x4(-5.0, -1.0, -0.1, -100.0);
            let lo = neon_broadcast_f32(0.0);
            let hi = neon_broadcast_f32(10.0);
            assert_eq!(to_arr(neon_clamp_f32(v, lo, hi)), [0.0, 0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn test_clamp_above_max() {
        unsafe {
            let v = f32x4(15.0, 20.0, 100.0, 50.0);
            let lo = neon_broadcast_f32(0.0);
            let hi = neon_broadcast_f32(10.0);
            assert_eq!(to_arr(neon_clamp_f32(v, lo, hi)), [10.0, 10.0, 10.0, 10.0]);
        }
    }

    #[test]
    fn test_clamp_mixed() {
        unsafe {
            let v = f32x4(-5.0, 5.0, 15.0, 10.0);
            let lo = neon_broadcast_f32(0.0);
            let hi = neon_broadcast_f32(10.0);
            assert_eq!(to_arr(neon_clamp_f32(v, lo, hi)), [0.0, 5.0, 10.0, 10.0]);
        }
    }

    #[test]
    fn test_clamp_at_boundaries() {
        unsafe {
            let v = f32x4(0.0, 10.0, 0.0, 10.0);
            let lo = neon_broadcast_f32(0.0);
            let hi = neon_broadcast_f32(10.0);
            assert_eq!(to_arr(neon_clamp_f32(v, lo, hi)), [0.0, 10.0, 0.0, 10.0]);
        }
    }

    #[test]
    fn test_clamp_negative_range() {
        unsafe {
            let v = f32x4(-3.0, -1.0, 0.0, 1.0);
            let lo = neon_broadcast_f32(-2.0);
            let hi = neon_broadcast_f32(-0.5);
            assert_eq!(to_arr(neon_clamp_f32(v, lo, hi)), [-2.0, -1.0, -0.5, -0.5]);
        }
    }

    #[test]
    fn test_clamp_per_lane_bounds() {
        unsafe {
            let v = f32x4(1.0, 5.0, 9.0, 13.0);
            let lo = f32x4(2.0, 2.0, 2.0, 2.0);
            let hi = f32x4(3.0, 6.0, 8.0, 12.0);
            assert_eq!(to_arr(neon_clamp_f32(v, lo, hi)), [2.0, 5.0, 8.0, 12.0]);
        }
    }

    // ---- select ----

    #[test]
    fn test_select_all_a() {
        unsafe {
            let mask = vdupq_n_u32(0xFFFFFFFF);
            let a = f32x4(1.0, 2.0, 3.0, 4.0);
            let b = f32x4(5.0, 6.0, 7.0, 8.0);
            assert_eq!(to_arr(neon_select_f32(mask, a, b)), [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_select_all_b() {
        unsafe {
            let mask = vdupq_n_u32(0x00000000);
            let a = f32x4(1.0, 2.0, 3.0, 4.0);
            let b = f32x4(5.0, 6.0, 7.0, 8.0);
            assert_eq!(to_arr(neon_select_f32(mask, a, b)), [5.0, 6.0, 7.0, 8.0]);
        }
    }

    #[test]
    fn test_select_alternating() {
        unsafe {
            let mask_arr: [u32; 4] = [0xFFFFFFFF, 0, 0xFFFFFFFF, 0];
            let mask = vld1q_u32(mask_arr.as_ptr());
            let a = f32x4(1.0, 2.0, 3.0, 4.0);
            let b = f32x4(10.0, 20.0, 30.0, 40.0);
            assert_eq!(to_arr(neon_select_f32(mask, a, b)), [1.0, 20.0, 3.0, 40.0]);
        }
    }

    #[test]
    fn test_select_from_comparison() {
        unsafe {
            let x = f32x4(1.0, 5.0, 3.0, 7.0);
            let threshold = neon_broadcast_f32(4.0);
            // mask = x > threshold
            let mask = vcgtq_f32(x, threshold);
            let a = f32x4(100.0, 200.0, 300.0, 400.0);
            let b = f32x4(-1.0, -2.0, -3.0, -4.0);
            // lanes 1 and 3 are > 4.0
            assert_eq!(to_arr(neon_select_f32(mask, a, b)), [-1.0, 200.0, -3.0, 400.0]);
        }
    }

    #[test]
    fn test_select_last_only() {
        unsafe {
            let mask_arr: [u32; 4] = [0, 0, 0, 0xFFFFFFFF];
            let mask = vld1q_u32(mask_arr.as_ptr());
            let a = f32x4(1.0, 2.0, 3.0, 4.0);
            let b = f32x4(10.0, 20.0, 30.0, 40.0);
            assert_eq!(to_arr(neon_select_f32(mask, a, b)), [10.0, 20.0, 30.0, 4.0]);
        }
    }

    // ---- interleave ----

    #[test]
    fn test_interleave_basic() {
        unsafe {
            let a = f32x4(1.0, 2.0, 3.0, 4.0);
            let b = f32x4(5.0, 6.0, 7.0, 8.0);
            let (lo, hi) = neon_interleave_f32(a, b);
            assert_eq!(to_arr(lo), [1.0, 5.0, 2.0, 6.0]);
            assert_eq!(to_arr(hi), [3.0, 7.0, 4.0, 8.0]);
        }
    }

    #[test]
    fn test_interleave_zeros_and_ones() {
        unsafe {
            let a = f32x4(0.0, 0.0, 0.0, 0.0);
            let b = f32x4(1.0, 1.0, 1.0, 1.0);
            let (lo, hi) = neon_interleave_f32(a, b);
            assert_eq!(to_arr(lo), [0.0, 1.0, 0.0, 1.0]);
            assert_eq!(to_arr(hi), [0.0, 1.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn test_interleave_same_vector() {
        unsafe {
            let a = f32x4(1.0, 2.0, 3.0, 4.0);
            let (lo, hi) = neon_interleave_f32(a, a);
            assert_eq!(to_arr(lo), [1.0, 1.0, 2.0, 2.0]);
            assert_eq!(to_arr(hi), [3.0, 3.0, 4.0, 4.0]);
        }
    }

    // ---- deinterleave ----

    #[test]
    fn test_deinterleave_basic() {
        unsafe {
            let a = f32x4(1.0, 2.0, 3.0, 4.0);
            let b = f32x4(5.0, 6.0, 7.0, 8.0);
            let (even, odd) = neon_deinterleave_f32(a, b);
            assert_eq!(to_arr(even), [1.0, 3.0, 5.0, 7.0]);
            assert_eq!(to_arr(odd), [2.0, 4.0, 6.0, 8.0]);
        }
    }

    #[test]
    fn test_deinterleave_sequential() {
        unsafe {
            let a = f32x4(10.0, 20.0, 30.0, 40.0);
            let b = f32x4(50.0, 60.0, 70.0, 80.0);
            let (even, odd) = neon_deinterleave_f32(a, b);
            assert_eq!(to_arr(even), [10.0, 30.0, 50.0, 70.0]);
            assert_eq!(to_arr(odd), [20.0, 40.0, 60.0, 80.0]);
        }
    }

    #[test]
    fn test_deinterleave_same_vector() {
        unsafe {
            let a = f32x4(1.0, 2.0, 3.0, 4.0);
            let (even, odd) = neon_deinterleave_f32(a, a);
            assert_eq!(to_arr(even), [1.0, 3.0, 1.0, 3.0]);
            assert_eq!(to_arr(odd), [2.0, 4.0, 2.0, 4.0]);
        }
    }

    // ---- interleave/deinterleave roundtrip ----

    #[test]
    fn test_interleave_deinterleave_roundtrip() {
        unsafe {
            let a = f32x4(1.0, 2.0, 3.0, 4.0);
            let b = f32x4(5.0, 6.0, 7.0, 8.0);
            let (lo, hi) = neon_interleave_f32(a, b);
            let (ra, rb) = neon_deinterleave_f32(lo, hi);
            assert_eq!(to_arr(ra), to_arr(a));
            assert_eq!(to_arr(rb), to_arr(b));
        }
    }

    #[test]
    fn test_deinterleave_interleave_roundtrip() {
        unsafe {
            let a = f32x4(1.0, 2.0, 3.0, 4.0);
            let b = f32x4(5.0, 6.0, 7.0, 8.0);
            let (even, odd) = neon_deinterleave_f32(a, b);
            let (ra, rb) = neon_interleave_f32(even, odd);
            assert_eq!(to_arr(ra), to_arr(a));
            assert_eq!(to_arr(rb), to_arr(b));
        }
    }

    // ---- combined / edge-case tests ----

    #[test]
    fn test_broadcast_then_sum() {
        unsafe {
            let v = neon_broadcast_f32(2.5);
            assert_eq!(neon_horizontal_sum_f32(v), 10.0);
        }
    }

    #[test]
    fn test_abs_then_sum() {
        unsafe {
            let v = f32x4(-1.0, -2.0, 3.0, -4.0);
            let a = neon_abs_f32(v);
            assert_eq!(neon_horizontal_sum_f32(a), 10.0);
        }
    }

    #[test]
    fn test_clamp_then_max() {
        unsafe {
            let v = f32x4(-5.0, 3.0, 15.0, 7.0);
            let lo = neon_broadcast_f32(0.0);
            let hi = neon_broadcast_f32(10.0);
            let clamped = neon_clamp_f32(v, lo, hi);
            assert_eq!(neon_horizontal_max_f32(clamped), 10.0);
        }
    }

    #[test]
    fn test_clamp_then_min() {
        unsafe {
            let v = f32x4(-5.0, 3.0, 15.0, 7.0);
            let lo = neon_broadcast_f32(0.0);
            let hi = neon_broadcast_f32(10.0);
            let clamped = neon_clamp_f32(v, lo, hi);
            assert_eq!(neon_horizontal_min_f32(clamped), 0.0);
        }
    }

    #[test]
    fn test_fma_chained() {
        unsafe {
            // (0 + 1*2) = 2, then (2 + 3*4) = 14
            let z = f32x4(0.0, 0.0, 0.0, 0.0);
            let a = f32x4(1.0, 1.0, 1.0, 1.0);
            let b = f32x4(2.0, 2.0, 2.0, 2.0);
            let step1 = neon_fma_f32(z, a, b);
            let c = f32x4(3.0, 3.0, 3.0, 3.0);
            let d = f32x4(4.0, 4.0, 4.0, 4.0);
            let step2 = neon_fma_f32(step1, c, d);
            assert_eq!(to_arr(step2), [14.0, 14.0, 14.0, 14.0]);
        }
    }

    #[test]
    fn test_select_with_clamp() {
        unsafe {
            let v = f32x4(-10.0, 5.0, 20.0, 0.0);
            let lo = neon_broadcast_f32(0.0);
            let hi = neon_broadcast_f32(10.0);
            let clamped = neon_clamp_f32(v, lo, hi);
            // Select: if original < 0, use 0; else use clamped
            let mask = vcgeq_f32(v, lo);
            let result = neon_select_f32(mask, clamped, lo);
            assert_eq!(to_arr(result), [0.0, 5.0, 10.0, 0.0]);
        }
    }

    #[test]
    fn test_partial_load_sum() {
        unsafe {
            let data = [1.0f32, 2.0, 3.0];
            let v = neon_load_partial_f32(data.as_ptr(), 3);
            assert_eq!(neon_horizontal_sum_f32(v), 6.0);
        }
    }

    #[test]
    fn test_horizontal_max_after_abs() {
        unsafe {
            let v = f32x4(-10.0, 3.0, -7.0, 1.0);
            let a = neon_abs_f32(v);
            assert_eq!(neon_horizontal_max_f32(a), 10.0);
        }
    }

    #[test]
    fn test_horizontal_min_after_abs() {
        unsafe {
            let v = f32x4(-10.0, 3.0, -7.0, 1.0);
            let a = neon_abs_f32(v);
            assert_eq!(neon_horizontal_min_f32(a), 1.0);
        }
    }

    #[test]
    fn test_broadcast_clamp_identity() {
        unsafe {
            let v = neon_broadcast_f32(5.0);
            let lo = neon_broadcast_f32(0.0);
            let hi = neon_broadcast_f32(10.0);
            assert_eq!(to_arr(neon_clamp_f32(v, lo, hi)), [5.0; 4]);
        }
    }

    #[test]
    fn test_fma_with_broadcast() {
        unsafe {
            let a = f32x4(1.0, 2.0, 3.0, 4.0);
            let scale = neon_broadcast_f32(2.0);
            let bias = neon_broadcast_f32(0.5);
            // bias + a * scale = [0.5+2, 0.5+4, 0.5+6, 0.5+8]
            let r = to_arr(neon_fma_f32(bias, a, scale));
            assert_eq!(r, [2.5, 4.5, 6.5, 8.5]);
        }
    }

    #[test]
    fn test_abs_subnormal() {
        unsafe {
            let tiny = f32::MIN_POSITIVE * 0.5;
            let v = f32x4(-tiny, tiny, -tiny, tiny);
            let a = to_arr(neon_abs_f32(v));
            for lane in a {
                assert_eq!(lane, tiny);
            }
        }
    }

    #[test]
    fn test_interleave_with_negatives() {
        unsafe {
            let a = f32x4(-1.0, -2.0, -3.0, -4.0);
            let b = f32x4(1.0, 2.0, 3.0, 4.0);
            let (lo, hi) = neon_interleave_f32(a, b);
            assert_eq!(to_arr(lo), [-1.0, 1.0, -2.0, 2.0]);
            assert_eq!(to_arr(hi), [-3.0, 3.0, -4.0, 4.0]);
        }
    }

    #[test]
    fn test_store_partial_preserves_beyond() {
        unsafe {
            let mut data = [100.0f32, 200.0, 300.0, 400.0];
            let v = f32x4(1.0, 2.0, 3.0, 4.0);
            neon_store_partial_f32(data.as_mut_ptr(), v, 2);
            // Only first 2 elements changed
            assert_eq!(data[2], 300.0);
            assert_eq!(data[3], 400.0);
        }
    }

    #[test]
    fn test_horizontal_sum_neg_inf() {
        unsafe {
            let v = f32x4(f32::NEG_INFINITY, 1.0, 2.0, 3.0);
            assert_eq!(neon_horizontal_sum_f32(v), f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_horizontal_max_nan_propagation() {
        unsafe {
            let v = f32x4(1.0, f32::NAN, 3.0, 4.0);
            // NEON max with NaN: IEEE 754 says NaN compares false,
            // so the result is implementation-defined. We just ensure no crash.
            let _ = neon_horizontal_max_f32(v);
        }
    }

    #[test]
    fn test_horizontal_min_nan_propagation() {
        unsafe {
            let v = f32x4(1.0, f32::NAN, 3.0, 4.0);
            let _ = neon_horizontal_min_f32(v);
        }
    }

    #[test]
    fn test_select_first_two_lanes() {
        unsafe {
            let mask_arr: [u32; 4] = [0xFFFFFFFF, 0xFFFFFFFF, 0, 0];
            let mask = vld1q_u32(mask_arr.as_ptr());
            let a = f32x4(1.0, 2.0, 3.0, 4.0);
            let b = f32x4(10.0, 20.0, 30.0, 40.0);
            assert_eq!(to_arr(neon_select_f32(mask, a, b)), [1.0, 2.0, 30.0, 40.0]);
        }
    }

    #[test]
    fn test_fma_large_values() {
        unsafe {
            let a = f32x4(1e20, 1e20, 1e20, 1e20);
            let b = f32x4(1e10, 1e10, 1e10, 1e10);
            let c = f32x4(1e10, 1e10, 1e10, 1e10);
            let r = to_arr(neon_fma_f32(a, b, c));
            for lane in r {
                assert!(lane.is_finite());
                assert!(lane > 1e20);
            }
        }
    }

    #[test]
    fn test_clamp_equal_bounds() {
        unsafe {
            let v = f32x4(-5.0, 0.0, 5.0, 10.0);
            let bound = neon_broadcast_f32(3.0);
            let r = to_arr(neon_clamp_f32(v, bound, bound));
            assert_eq!(r, [3.0, 3.0, 3.0, 3.0]);
        }
    }
}
