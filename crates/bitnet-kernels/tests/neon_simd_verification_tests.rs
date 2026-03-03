//! NEON SIMD intrinsic verification tests for Apple Silicon.
//!
//! These tests verify actual NEON behavior on arm64 hardware by exercising
//! `std::arch::aarch64` intrinsics directly with known inputs and checking
//! the outputs. They cover vector arithmetic, comparisons, load/store,
//! reductions, and integer operations relevant to 2-bit quantization.
#![cfg(all(feature = "cpu", target_arch = "aarch64"))]
#![allow(
    clippy::undocumented_unsafe_blocks,
    clippy::float_cmp,
    clippy::needless_range_loop,
    unused_unsafe
)]

#[cfg(test)]
mod tests {
    use std::arch::aarch64::*;

    const EPSILON: f32 = 1e-6;

    fn assert_approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < EPSILON, "expected {a} ≈ {b} (diff = {})", (a - b).abs());
    }

    // Helper: extract f32x4 lanes into an array.
    unsafe fn extract_f32x4(v: float32x4_t) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), v);
        out
    }

    // ── Vector arithmetic ────────────────────────────────────────────

    #[test]
    fn test_vaddq_f32() {
        unsafe {
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let b = vld1q_f32([10.0f32, 20.0, 30.0, 40.0].as_ptr());
            let c = vaddq_f32(a, b);
            let r = extract_f32x4(c);
            assert_approx_eq(r[0], 11.0);
            assert_approx_eq(r[1], 22.0);
            assert_approx_eq(r[2], 33.0);
            assert_approx_eq(r[3], 44.0);
        }
    }

    #[test]
    fn test_vmulq_f32() {
        unsafe {
            let a = vld1q_f32([2.0f32, 3.0, 4.0, 5.0].as_ptr());
            let b = vld1q_f32([0.5f32, 1.0, 2.0, 3.0].as_ptr());
            let c = vmulq_f32(a, b);
            let r = extract_f32x4(c);
            assert_approx_eq(r[0], 1.0);
            assert_approx_eq(r[1], 3.0);
            assert_approx_eq(r[2], 8.0);
            assert_approx_eq(r[3], 15.0);
        }
    }

    #[test]
    fn test_vfmaq_f32() {
        // vfmaq_f32(a, b, c) = a + b * c
        unsafe {
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let b = vld1q_f32([2.0f32, 3.0, 4.0, 5.0].as_ptr());
            let c = vld1q_f32([10.0f32, 10.0, 10.0, 10.0].as_ptr());
            let r = extract_f32x4(vfmaq_f32(a, b, c));
            assert_approx_eq(r[0], 21.0); // 1 + 2*10
            assert_approx_eq(r[1], 32.0); // 2 + 3*10
            assert_approx_eq(r[2], 43.0); // 3 + 4*10
            assert_approx_eq(r[3], 54.0); // 4 + 5*10
        }
    }

    #[test]
    fn test_vnegq_f32() {
        unsafe {
            let a = vld1q_f32([1.0f32, -2.0, 0.0, 3.5].as_ptr());
            let r = extract_f32x4(vnegq_f32(a));
            assert_approx_eq(r[0], -1.0);
            assert_approx_eq(r[1], 2.0);
            assert_approx_eq(r[2], 0.0);
            assert_approx_eq(r[3], -3.5);
        }
    }

    #[test]
    fn test_vabsq_f32() {
        unsafe {
            let a = vld1q_f32([-1.0f32, 2.0, -3.0, 0.0].as_ptr());
            let r = extract_f32x4(vabsq_f32(a));
            assert_approx_eq(r[0], 1.0);
            assert_approx_eq(r[1], 2.0);
            assert_approx_eq(r[2], 3.0);
            assert_approx_eq(r[3], 0.0);
        }
    }

    // ── Vector comparison ────────────────────────────────────────────

    #[test]
    fn test_vmaxq_f32() {
        unsafe {
            let a = vld1q_f32([1.0f32, 5.0, 3.0, 8.0].as_ptr());
            let b = vld1q_f32([4.0f32, 2.0, 7.0, 6.0].as_ptr());
            let r = extract_f32x4(vmaxq_f32(a, b));
            assert_approx_eq(r[0], 4.0);
            assert_approx_eq(r[1], 5.0);
            assert_approx_eq(r[2], 7.0);
            assert_approx_eq(r[3], 8.0);
        }
    }

    #[test]
    fn test_vminq_f32() {
        unsafe {
            let a = vld1q_f32([1.0f32, 5.0, 3.0, 8.0].as_ptr());
            let b = vld1q_f32([4.0f32, 2.0, 7.0, 6.0].as_ptr());
            let r = extract_f32x4(vminq_f32(a, b));
            assert_approx_eq(r[0], 1.0);
            assert_approx_eq(r[1], 2.0);
            assert_approx_eq(r[2], 3.0);
            assert_approx_eq(r[3], 6.0);
        }
    }

    #[test]
    fn test_vceqq_f32() {
        unsafe {
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let b = vld1q_f32([1.0f32, 9.0, 3.0, 0.0].as_ptr());
            let mask = vceqq_f32(a, b);
            // Equal lanes → all-ones (0xFFFFFFFF), unequal → 0.
            let mut out = [0u32; 4];
            vst1q_u32(out.as_mut_ptr(), mask);
            assert_eq!(out[0], 0xFFFF_FFFF); // 1.0 == 1.0
            assert_eq!(out[1], 0x0000_0000); // 2.0 != 9.0
            assert_eq!(out[2], 0xFFFF_FFFF); // 3.0 == 3.0
            assert_eq!(out[3], 0x0000_0000); // 4.0 != 0.0
        }
    }

    // ── Vector load / store ──────────────────────────────────────────

    #[test]
    fn test_vld1q_vst1q_f32_roundtrip() {
        unsafe {
            let input = [1.5f32, 2.5, 3.5, 4.5];
            let v = vld1q_f32(input.as_ptr());
            let r = extract_f32x4(v);
            assert_eq!(r, input);
        }
    }

    #[test]
    fn test_aligned_load_f32() {
        unsafe {
            // 16-byte aligned buffer (4 × f32 = 16 bytes).
            #[repr(align(16))]
            struct Aligned([f32; 4]);

            let data = Aligned([10.0, 20.0, 30.0, 40.0]);
            let v = vld1q_f32(data.0.as_ptr());
            let r = extract_f32x4(v);
            assert_eq!(r, [10.0, 20.0, 30.0, 40.0]);
        }
    }

    #[test]
    fn test_vdupq_n_f32() {
        unsafe {
            let v = vdupq_n_f32(42.0);
            let r = extract_f32x4(v);
            assert_eq!(r, [42.0, 42.0, 42.0, 42.0]);
        }
    }

    // ── Reduction operations ─────────────────────────────────────────

    #[test]
    fn test_horizontal_sum_f32() {
        unsafe {
            let v = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let sum = vaddvq_f32(v);
            assert_approx_eq(sum, 10.0);
        }
    }

    #[test]
    fn test_horizontal_max_f32() {
        unsafe {
            let v = vld1q_f32([1.0f32, 7.0, 3.0, 5.0].as_ptr());
            let max = vmaxvq_f32(v);
            assert_approx_eq(max, 7.0);
        }
    }

    // ── Integer operations (quantization-relevant) ───────────────────

    #[test]
    fn test_vaddq_s8() {
        unsafe {
            let a = vld1q_s8([1i8, -1, 0, 127, -128, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].as_ptr());
            let b =
                vld1q_s8([1i8, 1, 0, 0, 1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12].as_ptr());
            let c = vaddq_s8(a, b);
            let mut out = [0i8; 16];
            vst1q_s8(out.as_mut_ptr(), c);
            assert_eq!(out[0], 2); //  1 + 1
            assert_eq!(out[1], 0); // -1 + 1
            assert_eq!(out[2], 0); //  0 + 0
            assert_eq!(out[3], 127); // 127 + 0
            assert_eq!(out[4], -127); // -128 + 1
            assert_eq!(out[5], 0); //  2 + (-2)
            // Remaining lanes sum to zero.
            for i in 6..16 {
                assert_eq!(out[i], 0, "lane {i}");
            }
        }
    }

    #[test]
    fn test_vmull_s8_widening_multiply() {
        // vmull_s8 multiplies int8x8 → int16x8 (lower 8 lanes).
        unsafe {
            let a = vld1_s8([1i8, -1, 2, -2, 3, -3, 4, -4].as_ptr());
            let b = vld1_s8([10i8, 10, 10, 10, 10, 10, 10, 10].as_ptr());
            let c = vmull_s8(a, b);
            let mut out = [0i16; 8];
            vst1q_s16(out.as_mut_ptr(), c);
            assert_eq!(out[0], 10);
            assert_eq!(out[1], -10);
            assert_eq!(out[2], 20);
            assert_eq!(out[3], -20);
            assert_eq!(out[4], 30);
            assert_eq!(out[5], -30);
            assert_eq!(out[6], 40);
            assert_eq!(out[7], -40);
        }
    }
}
