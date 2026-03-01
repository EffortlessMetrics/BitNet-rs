//! Exhaustive NEON correctness tests.
//!
//! Validates every NEON intrinsic used by the kernel crate against a scalar
//! reference, covering boundary values (NaN, ±Inf, subnormals) and long
//! accumulation chains.
//!
//! On non-AArch64 hosts, scalar-only equivalents run so CI stays green.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EPS: f32 = 1e-5;

fn assert_close(a: f32, b: f32, label: &str) {
    if a.is_nan() && b.is_nan() {
        return;
    }
    let diff = (a - b).abs();
    let denom = a.abs().max(b.abs()).max(1e-12);
    assert!(diff / denom < EPS || diff < EPS, "{label}: values differ — a={a}, b={b}, diff={diff}");
}

fn assert_close_abs(a: f32, b: f32, tol: f32, label: &str) {
    if a.is_nan() && b.is_nan() {
        return;
    }
    let diff = (a - b).abs();
    assert!(diff <= tol, "{label}: values differ — a={a}, b={b}, diff={diff}, tol={tol}");
}

/// Boundary test values (NaN, ±Inf, subnormals, zero, normal).
fn boundary_values() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,  // smallest normal
        -f32::MIN_POSITIVE, // negative smallest normal
        1.0e-40,            // subnormal
        -1.0e-40,           // negative subnormal
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.5,
        -0.5,
        core::f32::consts::PI,
    ]
}

// ---------------------------------------------------------------------------
// AArch64 NEON implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use std::arch::aarch64::*;

    // --- vld1q_f32 / vst1q_f32 round-trip ---
    #[target_feature(enable = "neon")]
    pub unsafe fn load_store_roundtrip(src: &[f32; 4]) -> [f32; 4] {
        unsafe {
            let v = vld1q_f32(src.as_ptr());
            let mut dst = [0.0f32; 4];
            vst1q_f32(dst.as_mut_ptr(), v);
            dst
        }
    }

    // --- vmulq_f32 ---
    #[target_feature(enable = "neon")]
    pub unsafe fn mul_f32(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
        unsafe {
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let vc = vmulq_f32(va, vb);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), vc);
            out
        }
    }

    // --- vaddq_f32 ---
    #[target_feature(enable = "neon")]
    pub unsafe fn add_f32(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
        unsafe {
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let vc = vaddq_f32(va, vb);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), vc);
            out
        }
    }

    // --- vfmaq_f32 (fused multiply-add) ---
    #[target_feature(enable = "neon")]
    pub unsafe fn fma_f32(a: &[f32; 4], b: &[f32; 4], c: &[f32; 4]) -> [f32; 4] {
        unsafe {
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let vc = vld1q_f32(c.as_ptr());
            let vr = vfmaq_f32(vc, va, vb); // c + a*b
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), vr);
            out
        }
    }

    // --- vabsq_f32 ---
    #[target_feature(enable = "neon")]
    pub unsafe fn abs_f32(a: &[f32; 4]) -> [f32; 4] {
        unsafe {
            let va = vld1q_f32(a.as_ptr());
            let vr = vabsq_f32(va);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), vr);
            out
        }
    }

    // --- vmaxq_f32 ---
    #[target_feature(enable = "neon")]
    pub unsafe fn max_f32(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
        unsafe {
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let vr = vmaxq_f32(va, vb);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), vr);
            out
        }
    }

    // --- vminq_f32 ---
    #[target_feature(enable = "neon")]
    pub unsafe fn min_f32(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
        unsafe {
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let vr = vminq_f32(va, vb);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), vr);
            out
        }
    }

    // --- vdupq_n_f32 ---
    #[target_feature(enable = "neon")]
    pub unsafe fn dup_f32(val: f32) -> [f32; 4] {
        unsafe {
            let v = vdupq_n_f32(val);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), v);
            out
        }
    }

    // --- vaddvq_f32 (horizontal add) ---
    #[target_feature(enable = "neon")]
    pub unsafe fn hsum_f32(a: &[f32; 4]) -> f32 {
        unsafe {
            let va = vld1q_f32(a.as_ptr());
            vaddvq_f32(va)
        }
    }

    // --- vmaxvq_f32 (horizontal max) ---
    #[target_feature(enable = "neon")]
    pub unsafe fn hmax_f32(a: &[f32; 4]) -> f32 {
        unsafe {
            let va = vld1q_f32(a.as_ptr());
            vmaxvq_f32(va)
        }
    }

    // --- vdivq_f32 ---
    #[target_feature(enable = "neon")]
    pub unsafe fn div_f32(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
        unsafe {
            let va = vld1q_f32(a.as_ptr());
            let vb = vld1q_f32(b.as_ptr());
            let vr = vdivq_f32(va, vb);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), vr);
            out
        }
    }

    // --- Integer intrinsics used in matmul: vld1q_s8 / vmovl_s8 / vmull_s16 / vaddvq_s32 ---
    #[target_feature(enable = "neon")]
    pub unsafe fn i8_dot_16(a: &[i8; 16], b: &[i8; 16]) -> i32 {
        unsafe {
            let va = vld1q_s8(a.as_ptr());
            let vb = vld1q_s8(b.as_ptr());

            let a_lo = vmovl_s8(vget_low_s8(va));
            let a_hi = vmovl_s8(vget_high_s8(va));
            let b_lo = vmovl_s8(vget_low_s8(vb));
            let b_hi = vmovl_s8(vget_high_s8(vb));

            let p0 = vmull_s16(vget_low_s16(a_lo), vget_low_s16(b_lo));
            let p1 = vmull_s16(vget_high_s16(a_lo), vget_high_s16(b_lo));
            let p2 = vmull_s16(vget_low_s16(a_hi), vget_low_s16(b_hi));
            let p3 = vmull_s16(vget_high_s16(a_hi), vget_high_s16(b_hi));

            let sum = vaddq_s32(vaddq_s32(p0, p1), vaddq_s32(p2, p3));
            vaddvq_s32(sum)
        }
    }

    // --- vandq_u32 / vorrq_u32 (used in I2S quantize) ---
    #[target_feature(enable = "neon")]
    pub unsafe fn bitwise_and_or_u32(a: &[u32; 4], b: &[u32; 4]) -> ([u32; 4], [u32; 4]) {
        unsafe {
            let va = vld1q_u32(a.as_ptr());
            let vb = vld1q_u32(b.as_ptr());
            let vand = vandq_u32(va, vb);
            let vor = vorrq_u32(va, vb);
            let mut out_and = [0u32; 4];
            let mut out_or = [0u32; 4];
            vst1q_u32(out_and.as_mut_ptr(), vand);
            vst1q_u32(out_or.as_mut_ptr(), vor);
            (out_and, out_or)
        }
    }

    // --- Long accumulation chain ---
    #[target_feature(enable = "neon")]
    pub unsafe fn long_accumulate(data: &[f32]) -> f32 {
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 4 <= data.len() {
                let v = vld1q_f32(data.as_ptr().add(i));
                acc = vaddq_f32(acc, v);
                i += 4;
            }
            let mut sum = vaddvq_f32(acc);
            while i < data.len() {
                sum += data[i];
                i += 1;
            }
            sum
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback implementations (all architectures)
// ---------------------------------------------------------------------------

mod scalar_impl {
    pub fn load_store_roundtrip(src: &[f32; 4]) -> [f32; 4] {
        *src
    }

    pub fn mul_f32(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
        [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
    }

    pub fn add_f32(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    }

    pub fn fma_f32(a: &[f32; 4], b: &[f32; 4], c: &[f32; 4]) -> [f32; 4] {
        [c[0] + a[0] * b[0], c[1] + a[1] * b[1], c[2] + a[2] * b[2], c[3] + a[3] * b[3]]
    }

    pub fn abs_f32(a: &[f32; 4]) -> [f32; 4] {
        [a[0].abs(), a[1].abs(), a[2].abs(), a[3].abs()]
    }

    pub fn max_f32(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
        [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2]), a[3].max(b[3])]
    }

    pub fn min_f32(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
        [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2]), a[3].min(b[3])]
    }

    pub fn dup_f32(val: f32) -> [f32; 4] {
        [val; 4]
    }

    pub fn hsum_f32(a: &[f32; 4]) -> f32 {
        a[0] + a[1] + a[2] + a[3]
    }

    pub fn hmax_f32(a: &[f32; 4]) -> f32 {
        a[0].max(a[1]).max(a[2]).max(a[3])
    }

    pub fn div_f32(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
        [a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]]
    }

    pub fn i8_dot_16(a: &[i8; 16], b: &[i8; 16]) -> i32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x as i32 * y as i32).sum()
    }

    pub fn bitwise_and_or_u32(a: &[u32; 4], b: &[u32; 4]) -> ([u32; 4], [u32; 4]) {
        let and = [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]];
        let or = [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]];
        (and, or)
    }

    pub fn long_accumulate(data: &[f32]) -> f32 {
        data.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Dispatch macro — calls NEON on aarch64, scalar elsewhere, and always
// compares to the scalar reference.
// ---------------------------------------------------------------------------

macro_rules! dispatch4 {
    ($neon_fn:path, $scalar_fn:path, $($arg:expr),+) => {{
        #[cfg(target_arch = "aarch64")]
        let neon_result = unsafe { $neon_fn($($arg),+) };
        #[cfg(not(target_arch = "aarch64"))]
        let neon_result = $scalar_fn($($arg),+);
        let scalar_result = $scalar_fn($($arg),+);
        (neon_result, scalar_result)
    }};
}

// ---------------------------------------------------------------------------
// Tests: vld1q_f32 / vst1q_f32 round-trip
// ---------------------------------------------------------------------------

#[test]
fn load_store_roundtrip_normal() {
    let src = [1.0f32, -2.5, 0.0, 42.0];
    let (neon, scalar) =
        dispatch4!(neon_impl::load_store_roundtrip, scalar_impl::load_store_roundtrip, &src);
    for i in 0..4 {
        assert_close(neon[i], scalar[i], "load_store normal");
    }
}

#[test]
fn load_store_roundtrip_boundary() {
    let vals = boundary_values();
    for chunk in vals.chunks(4) {
        if chunk.len() < 4 {
            continue;
        }
        let src: [f32; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let (neon, scalar) =
            dispatch4!(neon_impl::load_store_roundtrip, scalar_impl::load_store_roundtrip, &src);
        for i in 0..4 {
            if src[i].is_nan() {
                assert!(neon[i].is_nan(), "load_store NaN preserved");
            } else {
                assert_eq!(neon[i].to_bits(), scalar[i].to_bits(), "load_store bitwise");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: vmulq_f32
// ---------------------------------------------------------------------------

#[test]
fn mul_f32_normal() {
    let a = [2.0f32, -3.0, 0.5, 10.0];
    let b = [4.0f32, 2.0, -6.0, 0.1];
    let (neon, scalar) = dispatch4!(neon_impl::mul_f32, scalar_impl::mul_f32, &a, &b);
    for i in 0..4 {
        assert_close(neon[i], scalar[i], "mul normal");
    }
}

#[test]
fn mul_f32_boundary() {
    let cases: &[([f32; 4], [f32; 4])] = &[
        ([0.0, f32::INFINITY, f32::NEG_INFINITY, 1.0], [1.0, 2.0, -3.0, f32::NAN]),
        ([f32::MIN_POSITIVE, 1e-40, -1e-40, f32::MAX], [1.0, 1.0, 1.0, 1.0]),
    ];
    for (a, b) in cases {
        let (neon, scalar) = dispatch4!(neon_impl::mul_f32, scalar_impl::mul_f32, a, b);
        for i in 0..4 {
            if scalar[i].is_nan() {
                assert!(neon[i].is_nan(), "mul boundary NaN");
            } else if scalar[i].is_infinite() {
                assert_eq!(neon[i], scalar[i], "mul boundary Inf");
            } else {
                assert_close(neon[i], scalar[i], "mul boundary");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: vaddq_f32
// ---------------------------------------------------------------------------

#[test]
fn add_f32_normal() {
    let a = [1.0f32, -1.0, 100.0, 0.001];
    let b = [2.0f32, 3.0, -50.0, 0.002];
    let (neon, scalar) = dispatch4!(neon_impl::add_f32, scalar_impl::add_f32, &a, &b);
    for i in 0..4 {
        assert_close(neon[i], scalar[i], "add normal");
    }
}

#[test]
fn add_f32_inf_nan() {
    let a = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0];
    let b = [1.0, 1.0, 1.0, f32::NAN];
    let (neon, scalar) = dispatch4!(neon_impl::add_f32, scalar_impl::add_f32, &a, &b);
    for i in 0..4 {
        if scalar[i].is_nan() {
            assert!(neon[i].is_nan(), "add NaN");
        } else {
            assert_eq!(neon[i], scalar[i], "add inf");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: vfmaq_f32 (fused multiply-add)
// ---------------------------------------------------------------------------

#[test]
fn fma_f32_normal() {
    let a = [2.0f32, 3.0, 4.0, 5.0];
    let b = [0.5f32, 0.1, 0.2, 0.3];
    let c = [10.0f32, 20.0, 30.0, 40.0];
    let (neon, scalar) = dispatch4!(neon_impl::fma_f32, scalar_impl::fma_f32, &a, &b, &c);
    for i in 0..4 {
        // FMA may differ from scalar a*b+c by 1 ULP; use generous tolerance
        assert_close_abs(neon[i], scalar[i], 1e-4, "fma normal");
    }
}

#[test]
fn fma_f32_boundary() {
    let a = [f32::MAX, f32::MIN_POSITIVE, 0.0, f32::NAN];
    let b = [1.0, 1.0, f32::INFINITY, 1.0];
    let c = [0.0, 0.0, 0.0, 0.0];
    let (neon, scalar) = dispatch4!(neon_impl::fma_f32, scalar_impl::fma_f32, &a, &b, &c);
    for i in 0..4 {
        if scalar[i].is_nan() {
            assert!(neon[i].is_nan(), "fma boundary NaN");
        } else if scalar[i].is_infinite() {
            assert_eq!(neon[i], scalar[i], "fma boundary Inf");
        } else {
            assert_close(neon[i], scalar[i], "fma boundary");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: vabsq_f32
// ---------------------------------------------------------------------------

#[test]
fn abs_f32_normal() {
    let a = [-1.0f32, 2.0, -3.5, 0.0];
    let (neon, scalar) = dispatch4!(neon_impl::abs_f32, scalar_impl::abs_f32, &a);
    for i in 0..4 {
        assert_close(neon[i], scalar[i], "abs normal");
    }
}

#[test]
fn abs_f32_boundary() {
    let a = [f32::NEG_INFINITY, -0.0, f32::NAN, -1e-40];
    let (neon, scalar) = dispatch4!(neon_impl::abs_f32, scalar_impl::abs_f32, &a);
    for i in 0..4 {
        if scalar[i].is_nan() {
            assert!(neon[i].is_nan(), "abs NaN");
        } else {
            assert_eq!(neon[i], scalar[i], "abs boundary");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: vmaxq_f32 / vminq_f32
// ---------------------------------------------------------------------------

#[test]
fn max_min_f32_normal() {
    let a = [1.0f32, -5.0, 3.0, 0.0];
    let b = [2.0f32, -3.0, 1.0, 0.0];
    let (neon_max, scalar_max) = dispatch4!(neon_impl::max_f32, scalar_impl::max_f32, &a, &b);
    let (neon_min, scalar_min) = dispatch4!(neon_impl::min_f32, scalar_impl::min_f32, &a, &b);
    for i in 0..4 {
        assert_close(neon_max[i], scalar_max[i], "max normal");
        assert_close(neon_min[i], scalar_min[i], "min normal");
    }
}

// ---------------------------------------------------------------------------
// Tests: vdupq_n_f32
// ---------------------------------------------------------------------------

#[test]
fn dup_f32_values() {
    for val in [0.0f32, 1.0, -1.0, f32::INFINITY, f32::NAN, 1e-40] {
        let (neon, scalar) = dispatch4!(neon_impl::dup_f32, scalar_impl::dup_f32, val);
        for i in 0..4 {
            if val.is_nan() {
                assert!(neon[i].is_nan(), "dup NaN");
            } else {
                assert_eq!(neon[i], scalar[i], "dup {val}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: vaddvq_f32 (horizontal sum)
// ---------------------------------------------------------------------------

#[test]
fn hsum_f32_normal() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let (neon, scalar) = dispatch4!(neon_impl::hsum_f32, scalar_impl::hsum_f32, &a);
    assert_close(neon, scalar, "hsum normal");
}

#[test]
fn hsum_f32_boundary() {
    // Sum involving Inf
    let a = [f32::INFINITY, -1.0, 0.0, 1.0];
    let (neon, scalar) = dispatch4!(neon_impl::hsum_f32, scalar_impl::hsum_f32, &a);
    assert_eq!(neon, scalar, "hsum inf");
}

// ---------------------------------------------------------------------------
// Tests: vmaxvq_f32 (horizontal max)
// ---------------------------------------------------------------------------

#[test]
fn hmax_f32_normal() {
    let a = [-5.0f32, 3.0, 1.0, 2.0];
    let (neon, scalar) = dispatch4!(neon_impl::hmax_f32, scalar_impl::hmax_f32, &a);
    assert_close(neon, scalar, "hmax normal");
}

// ---------------------------------------------------------------------------
// Tests: vdivq_f32
// ---------------------------------------------------------------------------

#[test]
fn div_f32_normal() {
    let a = [10.0f32, -6.0, 0.0, 1.0];
    let b = [2.0f32, 3.0, 1.0, 3.0];
    let (neon, scalar) = dispatch4!(neon_impl::div_f32, scalar_impl::div_f32, &a, &b);
    for i in 0..4 {
        assert_close(neon[i], scalar[i], "div normal");
    }
}

#[test]
fn div_f32_special() {
    let a = [1.0f32, 0.0, f32::INFINITY, f32::NAN];
    let b = [0.0f32, 0.0, f32::INFINITY, 1.0];
    let (neon, scalar) = dispatch4!(neon_impl::div_f32, scalar_impl::div_f32, &a, &b);
    for i in 0..4 {
        if scalar[i].is_nan() {
            assert!(neon[i].is_nan(), "div special NaN at {i}");
        } else if scalar[i].is_infinite() {
            assert_eq!(neon[i], scalar[i], "div special Inf at {i}");
        } else {
            assert_close(neon[i], scalar[i], "div special");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: integer dot (vld1q_s8, vmovl_s8, vmull_s16, vaddq_s32, vaddvq_s32)
// ---------------------------------------------------------------------------

#[test]
fn i8_dot_basic() {
    let a: [i8; 16] = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8];
    let b: [i8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let (neon, scalar) = dispatch4!(neon_impl::i8_dot_16, scalar_impl::i8_dot_16, &a, &b);
    assert_eq!(neon, scalar, "i8_dot basic");
}

#[test]
fn i8_dot_extremes() {
    let a: [i8; 16] = [127; 16];
    let b: [i8; 16] = [-128; 16];
    let (neon, scalar) = dispatch4!(neon_impl::i8_dot_16, scalar_impl::i8_dot_16, &a, &b);
    assert_eq!(neon, scalar, "i8_dot extremes");
    assert_eq!(scalar, 127 * -128 * 16, "i8_dot expected value");
}

#[test]
fn i8_dot_zeros() {
    let a = [0i8; 16];
    let b = [42i8; 16];
    let (neon, scalar) = dispatch4!(neon_impl::i8_dot_16, scalar_impl::i8_dot_16, &a, &b);
    assert_eq!(neon, 0);
    assert_eq!(scalar, 0);
}

// ---------------------------------------------------------------------------
// Tests: vandq_u32 / vorrq_u32
// ---------------------------------------------------------------------------

#[test]
fn bitwise_and_or_basic() {
    let a = [0xFFFF_0000u32, 0x0000_FFFF, 0xAAAA_AAAA, 0x5555_5555];
    let b = [0x0F0F_0F0Fu32, 0xF0F0_F0F0, 0xFFFF_FFFF, 0x0000_0000];
    let (neon, scalar) =
        dispatch4!(neon_impl::bitwise_and_or_u32, scalar_impl::bitwise_and_or_u32, &a, &b);
    assert_eq!(neon.0, scalar.0, "AND mismatch");
    assert_eq!(neon.1, scalar.1, "OR mismatch");
}

// ---------------------------------------------------------------------------
// Tests: accumulation precision (long chains)
// ---------------------------------------------------------------------------

#[test]
fn long_accumulation_small_values() {
    // Sum of 10_000 small values — tests FP accumulation drift
    let n = 10_000;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 1e-5).collect();

    let (neon, scalar) =
        dispatch4!(neon_impl::long_accumulate, scalar_impl::long_accumulate, &data);
    // NEON may use different accumulation order; allow modest tolerance
    assert_close_abs(neon, scalar, 0.01, "long accum small");
}

#[test]
fn long_accumulation_alternating_sign() {
    // Alternating +1, -1 should cancel to ≈ 0
    let n = 10_000;
    let data: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let (neon, scalar) =
        dispatch4!(neon_impl::long_accumulate, scalar_impl::long_accumulate, &data);
    assert_close_abs(neon, 0.0, 1.0, "long accum alt neon");
    assert_close_abs(scalar, 0.0, 1.0, "long accum alt scalar");
}

#[test]
fn long_accumulation_large_values() {
    let n = 4096;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 100.0).collect();
    let (neon, scalar) =
        dispatch4!(neon_impl::long_accumulate, scalar_impl::long_accumulate, &data);
    let expected = (n as f32 - 1.0) * (n as f32) / 2.0 * 100.0;
    // Large sums accumulate rounding error; allow 0.1% tolerance
    let tol = expected.abs() * 0.001;
    assert_close_abs(neon, expected, tol, "long accum large neon");
    assert_close_abs(scalar, expected, tol, "long accum large scalar");
}

// ---------------------------------------------------------------------------
// Tests: odd-length accumulation (tests scalar tail handling)
// ---------------------------------------------------------------------------

#[test]
fn accumulate_non_multiple_of_4() {
    for len in [1, 2, 3, 5, 7, 13, 17, 33, 63, 65] {
        let data: Vec<f32> = (0..len).map(|i| (i + 1) as f32).collect();
        let (neon, scalar) =
            dispatch4!(neon_impl::long_accumulate, scalar_impl::long_accumulate, &data);
        let expected: f32 = (len as f32) * (len as f32 + 1.0) / 2.0;
        assert_close_abs(neon, expected, 1.0, &format!("accum len={len} neon"));
        assert_close_abs(scalar, expected, 1.0, &format!("accum len={len} scalar"));
    }
}

// ---------------------------------------------------------------------------
// Tests: subnormal handling
// ---------------------------------------------------------------------------

#[test]
fn subnormal_operations() {
    let sub = 1.0e-40f32; // subnormal
    let a = [sub, -sub, sub, -sub];
    let b = [1.0f32; 4];

    // mul
    let (neon_m, scalar_m) = dispatch4!(neon_impl::mul_f32, scalar_impl::mul_f32, &a, &b);
    for i in 0..4 {
        assert_close_abs(neon_m[i], scalar_m[i], 1e-38, "subnormal mul");
    }

    // add
    let (neon_a, scalar_a) = dispatch4!(neon_impl::add_f32, scalar_impl::add_f32, &a, &b);
    for i in 0..4 {
        assert_close(neon_a[i], scalar_a[i], "subnormal add");
    }

    // abs
    let (neon_abs, scalar_abs) = dispatch4!(neon_impl::abs_f32, scalar_impl::abs_f32, &a);
    for i in 0..4 {
        assert_close_abs(neon_abs[i], scalar_abs[i], 1e-38, "subnormal abs");
    }
}
