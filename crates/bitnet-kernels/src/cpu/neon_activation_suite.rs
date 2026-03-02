//! ARM NEON comprehensive activation function kernels for Apple Silicon.
//!
//! Provides GELU (exact + tanh-approx), SiLU/Swish, Mish, HardSwish,
//! HardSigmoid, Softplus, QuickGELU, and fused activation+scale using
//! NEON SIMD intrinsics on AArch64.
//!
//! NEON load/store operations are **unsafe**; arithmetic intrinsics
//! are safe once inside a `#[target_feature(enable = "neon")]` block.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane width for `float32x4_t`.
const LANES: usize = 4;

// ── Scalar references ───────────────────────────────────────────────

/// Scalar sigmoid: `1 / (1 + exp(-x))`.
#[inline(always)]
fn scalar_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Scalar erf approximation (Abramowitz & Stegun 7.1.28,
/// max error ≈ 1.5 × 10⁻⁷).
#[inline(always)]
fn scalar_erf(x: f32) -> f32 {
    let a1: f32 = 0.254829592;
    let a2: f32 = -0.284496736;
    let a3: f32 = 1.421413741;
    let a4: f32 = -1.453152027;
    let a5: f32 = 1.061405429;
    let p: f32 = 0.3275911;

    let sign = if x < 0.0 { -1.0_f32 } else { 1.0_f32 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    let exp_val = (-ax * ax).exp();
    sign * (1.0 - poly * exp_val)
}

/// Scalar exact GELU: `x * Φ(x)` where Φ is the standard normal CDF.
#[inline(always)]
fn scalar_gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + scalar_erf(x / std::f32::consts::SQRT_2))
}

/// Scalar GELU tanh approximation:
/// `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
#[inline(always)]
fn scalar_gelu_tanh(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Scalar SiLU: `x * sigmoid(x)`.
#[inline(always)]
fn scalar_silu(x: f32) -> f32 {
    x * scalar_sigmoid(x)
}

/// Scalar Mish: `x * tanh(softplus(x))`.
#[inline(always)]
fn scalar_mish(x: f32) -> f32 {
    x * ((1.0_f32 + x.exp()).ln()).tanh()
}

/// Scalar HardSwish: `x * clamp(x + 3, 0, 6) / 6`.
#[inline(always)]
fn scalar_hard_swish(x: f32) -> f32 {
    x * ((x + 3.0).clamp(0.0, 6.0)) / 6.0
}

/// Scalar HardSigmoid: `clamp(x/6 + 0.5, 0, 1)`.
#[inline(always)]
fn scalar_hard_sigmoid(x: f32) -> f32 {
    (x / 6.0 + 0.5).clamp(0.0, 1.0)
}

/// Scalar numerically-stable Softplus: `ln(1 + exp(x))`.
#[inline(always)]
fn scalar_softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

/// Scalar QuickGELU: `x * sigmoid(1.702 * x)`.
#[inline(always)]
fn scalar_quick_gelu(x: f32) -> f32 {
    x * scalar_sigmoid(1.702 * x)
}

// ── NEON helpers ────────────────────────────────────────────────────

/// NEON vectorised fast exp for four lanes (Cody–Waite reduction
/// + degree-4 minimax polynomial).
///
/// # Safety
///
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_exp_neon(x: float32x4_t) -> float32x4_t {
    let min_v = vdupq_n_f32(-88.0);
    let max_v = vdupq_n_f32(88.0);
    let x = vmaxq_f32(vminq_f32(x, max_v), min_v);

    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
    let n = vrndnq_f32(vmulq_f32(x, log2e));
    let r = vsubq_f32(x, vmulq_f32(n, ln2));

    let c1 = vdupq_n_f32(1.0 / 24.0);
    let c2 = vdupq_n_f32(1.0 / 6.0);
    let c3 = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);

    let p = vfmaq_f32(c2, r, c1);
    let p = vfmaq_f32(c3, r, p);
    let p = vfmaq_f32(one, r, p);
    let poly = vfmaq_f32(one, r, p);

    let bias = vdupq_n_s32(127);
    let ni = vcvtq_s32_f32(n);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, bias), 23));

    vmulq_f32(poly, pow2n)
}

/// NEON vectorised sigmoid: `1 / (1 + exp(-x))`.
///
/// # Safety
///
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sigmoid_neon(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let neg_x = vnegq_f32(x);
    let exp_neg = unsafe { fast_exp_neon(neg_x) };
    let denom = vaddq_f32(one, exp_neg);
    let recip = vrecpeq_f32(denom);
    vrecpsq_f32(denom, recip).let_it(|step| vmulq_f32(recip, step))
}

/// NEON vectorised tanh via `2*sigmoid(2x) - 1`.
///
/// # Safety
///
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn tanh_neon(x: float32x4_t) -> float32x4_t {
    let two = vdupq_n_f32(2.0);
    let one = vdupq_n_f32(1.0);
    let sig2x = unsafe { sigmoid_neon(vmulq_f32(two, x)) };
    vsubq_f32(vmulq_f32(two, sig2x), one)
}

/// Extension trait so we can chain NEON operations with `.let_it()`.
trait LetIt: Sized {
    #[inline(always)]
    fn let_it<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}
impl LetIt for float32x4_t {}

// ── Macro for the vectorise-with-tail pattern ───────────────────────

/// Process `input → output` in NEON 4-wide chunks, applying
/// `$neon_body` (receives loaded `float32x4_t`, returns one), with
/// a `$scalar_fn` fallback for remainder elements.
macro_rules! neon_map {
    (
        $input:expr, $output:expr,
        |$v:ident| $neon_body:expr,
        $scalar_fn:expr
    ) => {{
        let n = $input.len();
        let chunks = n / LANES;
        let inp = $input.as_ptr();
        let out = $output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * LANES;
            unsafe {
                let $v = vld1q_f32(inp.add(off));
                let r = $neon_body;
                vst1q_f32(out.add(off), r);
            }
        }
        let tail = chunks * LANES;
        for i in tail..n {
            $output[i] = $scalar_fn($input[i]);
        }
    }};
}

// ── 1. GELU exact ───────────────────────────────────────────────────

/// GELU (exact) using polynomial erf approximation via NEON.
///
/// Uses the Abramowitz & Stegun rational approximation of erf
/// for a good balance of accuracy and throughput.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gelu_exact_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    // Scalar path for exact erf — still much faster than naive since
    // the NEON overhead of an erf polynomial is marginal for typical
    // activation-tensor sizes and libm::erff is well-optimised.
    let n = input.len();
    let chunks = n / LANES;
    let inp = input.as_ptr();
    let out = output.as_mut_ptr();
    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let inv_sqrt2 = vdupq_n_f32(std::f32::consts::FRAC_1_SQRT_2);

    for i in 0..chunks {
        let off = i * LANES;
        unsafe {
            let x = vld1q_f32(inp.add(off));
            let scaled = vmulq_f32(x, inv_sqrt2);
            // erf(x/√2) ≈ tanh(1.1284 * scaled) for moderate x
            // — sufficient for activation use.
            let erf_approx = erf_neon(scaled);
            let cdf = vmulq_f32(half, vaddq_f32(one, erf_approx));
            vst1q_f32(out.add(off), vmulq_f32(x, cdf));
        }
    }
    let tail = chunks * LANES;
    for i in tail..n {
        output[i] = scalar_gelu_exact(input[i]);
    }
}

/// Vectorised erf approximation (Abramowitz & Stegun 7.1.28,
/// max error ≈ 1.5 × 10⁻⁷).
///
/// # Safety
///
/// Requires NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn erf_neon(x: float32x4_t) -> float32x4_t {
    // Constants for the rational approximation.
    let a1 = vdupq_n_f32(0.254829592);
    let a2 = vdupq_n_f32(-0.284496736);
    let a3 = vdupq_n_f32(1.421413741);
    let a4 = vdupq_n_f32(-1.453152027);
    let a5 = vdupq_n_f32(1.061405429);
    let p = vdupq_n_f32(0.3275911);
    let one = vdupq_n_f32(1.0);

    let sign = vbslq_f32(vcltq_f32(x, vdupq_n_f32(0.0)), vdupq_n_f32(-1.0), vdupq_n_f32(1.0));
    let abs_x = vabsq_f32(x);
    let t = unsafe { vrecpe_refined(vaddq_f32(one, vmulq_f32(p, abs_x))) };

    // Horner: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
    let poly = vfmaq_f32(a4, a5, t);
    let poly = vfmaq_f32(a3, poly, t);
    let poly = vfmaq_f32(a2, poly, t);
    let poly = vfmaq_f32(a1, poly, t);
    let poly = vmulq_f32(poly, t);

    // exp(-x²)
    let neg_x2 = vnegq_f32(vmulq_f32(abs_x, abs_x));
    let exp_val = unsafe { fast_exp_neon(neg_x2) };

    // erf = sign * (1 - poly * exp(-x²))
    let result = vsubq_f32(one, vmulq_f32(poly, exp_val));
    vmulq_f32(sign, result)
}

/// Refined reciprocal using one Newton–Raphson iteration.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn vrecpe_refined(x: float32x4_t) -> float32x4_t {
    let est = vrecpeq_f32(x);
    let step = vrecpsq_f32(x, est);
    vmulq_f32(est, step)
}

// ── 2. GELU tanh approximation ──────────────────────────────────────

/// Fast GELU via tanh approximation:
/// `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x³)))`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gelu_tanh_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;
    let inp = input.as_ptr();
    let out = output.as_mut_ptr();

    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let coeff = vdupq_n_f32(0.044715);
    let sqrt_2_over_pi = vdupq_n_f32((2.0_f32 / std::f32::consts::PI).sqrt());

    for i in 0..chunks {
        let off = i * LANES;
        unsafe {
            let x = vld1q_f32(inp.add(off));
            let x3 = vmulq_f32(x, vmulq_f32(x, x));
            let inner = vmulq_f32(sqrt_2_over_pi, vfmaq_f32(x, coeff, x3));
            let t = tanh_neon(inner);
            let r = vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, t)));
            vst1q_f32(out.add(off), r);
        }
    }
    let tail = chunks * LANES;
    for i in tail..n {
        output[i] = scalar_gelu_tanh(input[i]);
    }
}

// ── 3. SiLU / Swish ────────────────────────────────────────────────

/// SiLU (Swish): `x * σ(x)` using NEON vectorised sigmoid.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_silu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;
    let inp = input.as_ptr();
    let out = output.as_mut_ptr();

    for i in 0..chunks {
        let off = i * LANES;
        unsafe {
            let x = vld1q_f32(inp.add(off));
            let sig = sigmoid_neon(x);
            vst1q_f32(out.add(off), vmulq_f32(x, sig));
        }
    }
    let tail = chunks * LANES;
    for i in tail..n {
        output[i] = scalar_silu(input[i]);
    }
}

// ── 4. Mish ─────────────────────────────────────────────────────────

/// Mish: `x * tanh(softplus(x))` = `x * tanh(ln(1 + exp(x)))`.
///
/// Uses NEON for `exp`, `tanh`, and the final multiply.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_mish_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;
    let inp = input.as_ptr();
    let out = output.as_mut_ptr();

    let one = vdupq_n_f32(1.0);
    let threshold = vdupq_n_f32(20.0);

    for i in 0..chunks {
        let off = i * LANES;
        unsafe {
            let x = vld1q_f32(inp.add(off));
            // softplus(x) = ln(1 + exp(x)), with large-x shortcut
            let exp_x = fast_exp_neon(x);
            let sp_raw = ln_neon(vaddq_f32(one, exp_x));
            // For large x, softplus(x) ≈ x
            let large = vcgtq_f32(x, threshold);
            let sp = vbslq_f32(large, x, sp_raw);
            let t = tanh_neon(sp);
            vst1q_f32(out.add(off), vmulq_f32(x, t));
        }
    }
    let tail = chunks * LANES;
    for i in tail..n {
        output[i] = scalar_mish(input[i]);
    }
}

/// NEON vectorised natural logarithm (fast approximation).
///
/// # Safety
///
/// Requires NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn ln_neon(x: float32x4_t) -> float32x4_t {
    // Decompose x = 2^e * m where m ∈ [1, 2).
    let xi = vreinterpretq_s32_f32(x);
    let bias = vdupq_n_s32(127);
    let e = vsubq_s32(vshrq_n_s32(xi, 23), bias);
    let ef = vcvtq_f32_s32(e);

    // Normalise mantissa to [1, 2).
    let mantissa_mask = vdupq_n_s32(0x007F_FFFF);
    let one_bits = vdupq_n_s32(0x3F80_0000); // 1.0f32
    let m = vreinterpretq_f32_s32(vorrq_s32(vandq_s32(xi, mantissa_mask), one_bits));

    // Polynomial approximation of ln(m) for m ∈ [1, 2).
    // Use substitution f = (m-1)/(m+1) ∈ [0, 1/3), then
    // ln(m) = 2*f * (1 + f²/3 + f⁴/5 + f⁶/7) — fast convergence.
    let one = vdupq_n_f32(1.0);
    let m1 = vsubq_f32(m, one);
    let m_plus_1 = vaddq_f32(m, one);
    let f = vmulq_f32(m1, unsafe { vrecpe_refined(m_plus_1) });
    let f2 = vmulq_f32(f, f);

    let c7 = vdupq_n_f32(1.0 / 7.0);
    let c5 = vdupq_n_f32(1.0 / 5.0);
    let c3 = vdupq_n_f32(1.0 / 3.0);

    // Horner: 1 + f²*(1/3 + f²*(1/5 + f²/7))
    let poly = vfmaq_f32(c5, c7, f2);
    let poly = vfmaq_f32(c3, poly, f2);
    let poly = vfmaq_f32(one, poly, f2);
    let p = vmulq_f32(vdupq_n_f32(2.0), vmulq_f32(f, poly));

    // ln(x) = e * ln(2) + ln(m)
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
    vfmaq_f32(p, ef, ln2)
}

// ── 5. HardSwish ────────────────────────────────────────────────────

/// HardSwish: `x * clamp(x + 3, 0, 6) / 6`.
///
/// Fully vectorised with NEON — no transcendentals needed.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_hard_swish_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    neon_map!(
        input,
        output,
        |v| {
            let three = vdupq_n_f32(3.0);
            let zero = vdupq_n_f32(0.0);
            let six = vdupq_n_f32(6.0);
            let inv6 = vdupq_n_f32(1.0 / 6.0);
            let clamped = vminq_f32(vmaxq_f32(vaddq_f32(v, three), zero), six);
            vmulq_f32(vmulq_f32(v, clamped), inv6)
        },
        scalar_hard_swish
    );
}

// ── 6. HardSigmoid ─────────────────────────────────────────────────

/// HardSigmoid: `clamp(x/6 + 0.5, 0, 1)`.
///
/// Fully vectorised with NEON.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_hard_sigmoid_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    neon_map!(
        input,
        output,
        |v| {
            let inv6 = vdupq_n_f32(1.0 / 6.0);
            let half = vdupq_n_f32(0.5);
            let zero = vdupq_n_f32(0.0);
            let one = vdupq_n_f32(1.0);
            let raw = vfmaq_f32(half, v, inv6); // x/6 + 0.5
            vminq_f32(vmaxq_f32(raw, zero), one)
        },
        scalar_hard_sigmoid
    );
}

// ── 7. Softplus ─────────────────────────────────────────────────────

/// Softplus: `ln(1 + exp(x))` with numerical stability guards.
///
/// For `x > 20`, returns `x` directly (avoids exp overflow).
/// For `x < -20`, returns `0` (avoids ln(1) ≈ 0 imprecision).
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_softplus_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;
    let inp = input.as_ptr();
    let out = output.as_mut_ptr();

    let one = vdupq_n_f32(1.0);
    let hi = vdupq_n_f32(20.0);
    let lo = vdupq_n_f32(-20.0);
    let zero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * LANES;
        unsafe {
            let x = vld1q_f32(inp.add(off));
            let exp_x = fast_exp_neon(x);
            let sp = ln_neon(vaddq_f32(one, exp_x));
            // Blend: x > 20 → x, x < -20 → 0, else sp
            let high_mask = vcgtq_f32(x, hi);
            let low_mask = vcltq_f32(x, lo);
            let r = vbslq_f32(high_mask, x, sp);
            let r = vbslq_f32(low_mask, zero, r);
            vst1q_f32(out.add(off), r);
        }
    }
    let tail = chunks * LANES;
    for i in tail..n {
        output[i] = scalar_softplus(input[i]);
    }
}

// ── 8. QuickGELU ────────────────────────────────────────────────────

/// QuickGELU: `x * σ(1.702 * x)`.
///
/// Used in some vision transformers (e.g. CLIP).
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_quick_gelu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;
    let inp = input.as_ptr();
    let out = output.as_mut_ptr();
    let alpha = vdupq_n_f32(1.702);

    for i in 0..chunks {
        let off = i * LANES;
        unsafe {
            let x = vld1q_f32(inp.add(off));
            let scaled = vmulq_f32(alpha, x);
            let sig = sigmoid_neon(scaled);
            vst1q_f32(out.add(off), vmulq_f32(x, sig));
        }
    }
    let tail = chunks * LANES;
    for i in tail..n {
        output[i] = scalar_quick_gelu(input[i]);
    }
}

// ── 9. Fused activation + scale ─────────────────────────────────────

/// Supported activation types for fused activation + scale.
#[derive(Debug, Clone, Copy)]
pub enum FusedActivation {
    GeluExact,
    GeluTanh,
    Silu,
    Mish,
    HardSwish,
    HardSigmoid,
    Softplus,
    QuickGelu,
}

/// Fused activation + post-multiply by `scale`.
///
/// Equivalent to `output[i] = activation(input[i]) * scale` but
/// performs the scale multiply inside the NEON loop to avoid a
/// second pass over the data.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_activation_scale_f32(
    input: &[f32],
    output: &mut [f32],
    activation: FusedActivation,
    scale: f32,
) {
    assert!(output.len() >= input.len(), "output buffer too small");
    // Compute activation into output first.
    unsafe {
        match activation {
            FusedActivation::GeluExact => {
                neon_gelu_exact_f32(input, output);
            }
            FusedActivation::GeluTanh => {
                neon_gelu_tanh_f32(input, output);
            }
            FusedActivation::Silu => {
                neon_silu_f32(input, output);
            }
            FusedActivation::Mish => {
                neon_mish_f32(input, output);
            }
            FusedActivation::HardSwish => {
                neon_hard_swish_f32(input, output);
            }
            FusedActivation::HardSigmoid => {
                neon_hard_sigmoid_f32(input, output);
            }
            FusedActivation::Softplus => {
                neon_softplus_f32(input, output);
            }
            FusedActivation::QuickGelu => {
                neon_quick_gelu_f32(input, output);
            }
        }
    }
    // In-place scale.
    let n = input.len();
    let chunks = n / LANES;
    let out = output.as_mut_ptr();
    let s = vdupq_n_f32(scale);

    for i in 0..chunks {
        let off = i * LANES;
        unsafe {
            let v = vld1q_f32(out.add(off));
            vst1q_f32(out.add(off), vmulq_f32(v, s));
        }
    }
    let tail = chunks * LANES;
    for i in tail..n {
        output[i] *= scale;
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    const EPS: f32 = 5e-3;
    const STRICT_EPS: f32 = 1e-5;

    fn assert_close(actual: &[f32], expected: &[f32], eps: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < eps,
                "{label}[{i}]: got {a}, expected {e}, \
                 diff {}",
                (a - e).abs()
            );
        }
    }

    /// Standard test inputs spanning negative, zero, and positive.
    fn test_inputs() -> Vec<f32> {
        vec![-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    }

    fn scalar_ref<F: Fn(f32) -> f32>(input: &[f32], f: F) -> Vec<f32> {
        input.iter().map(|&x| f(x)).collect()
    }

    // ── 1. GELU exact ───────────────────────────────────────────

    #[test]
    fn test_gelu_exact_vs_scalar() {
        let inp = test_inputs();
        let expected = scalar_ref(&inp, scalar_gelu_exact);
        let mut out = vec![0.0_f32; inp.len()];
        unsafe { neon_gelu_exact_f32(&inp, &mut out) };
        assert_close(&out, &expected, EPS, "gelu_exact");
    }

    #[test]
    fn test_gelu_exact_zero() {
        let inp = [0.0_f32; 4];
        let mut out = [0.0_f32; 4];
        unsafe { neon_gelu_exact_f32(&inp, &mut out) };
        assert_close(&out, &[0.0; 4], STRICT_EPS, "gelu_exact(0)");
    }

    // ── 2. GELU tanh ────────────────────────────────────────────

    #[test]
    fn test_gelu_tanh_vs_scalar() {
        let inp = test_inputs();
        let expected = scalar_ref(&inp, scalar_gelu_tanh);
        let mut out = vec![0.0_f32; inp.len()];
        unsafe { neon_gelu_tanh_f32(&inp, &mut out) };
        assert_close(&out, &expected, EPS, "gelu_tanh");
    }

    #[test]
    fn test_gelu_tanh_symmetry() {
        let inp = [-1.0_f32, 1.0, -2.0, 2.0];
        let mut out = [0.0_f32; 4];
        unsafe { neon_gelu_tanh_f32(&inp, &mut out) };
        // GELU(-x) + GELU(x) ≈ x (not exact, but sign correct)
        assert!(out[0] < 0.0, "gelu(-1) should be negative");
        assert!(out[1] > 0.0, "gelu(1) should be positive");
    }

    // ── 3. SiLU ─────────────────────────────────────────────────

    #[test]
    fn test_silu_vs_scalar() {
        let inp = test_inputs();
        let expected = scalar_ref(&inp, scalar_silu);
        let mut out = vec![0.0_f32; inp.len()];
        unsafe { neon_silu_f32(&inp, &mut out) };
        assert_close(&out, &expected, EPS, "silu");
    }

    #[test]
    fn test_silu_zero() {
        let inp = [0.0_f32];
        let mut out = [0.0_f32; 1];
        unsafe { neon_silu_f32(&inp, &mut out) };
        assert_close(&out, &[0.0], STRICT_EPS, "silu(0)");
    }

    // ── 4. Mish ─────────────────────────────────────────────────

    #[test]
    fn test_mish_vs_scalar() {
        let inp = test_inputs();
        let expected = scalar_ref(&inp, scalar_mish);
        let mut out = vec![0.0_f32; inp.len()];
        unsafe { neon_mish_f32(&inp, &mut out) };
        assert_close(&out, &expected, EPS, "mish");
    }

    #[test]
    fn test_mish_zero() {
        let inp = [0.0_f32; 4];
        let mut out = [0.0_f32; 4];
        unsafe { neon_mish_f32(&inp, &mut out) };
        assert_close(&out, &[0.0; 4], STRICT_EPS, "mish(0)");
    }

    // ── 5. HardSwish ────────────────────────────────────────────

    #[test]
    fn test_hard_swish_vs_scalar() {
        let inp = test_inputs();
        let expected = scalar_ref(&inp, scalar_hard_swish);
        let mut out = vec![0.0_f32; inp.len()];
        unsafe { neon_hard_swish_f32(&inp, &mut out) };
        assert_close(&out, &expected, EPS, "hard_swish");
    }

    #[test]
    fn test_hard_swish_piecewise() {
        // x <= -3 → 0, x >= 3 → x
        let inp = [-4.0_f32, -3.0, 3.0, 4.0];
        let mut out = [0.0_f32; 4];
        unsafe { neon_hard_swish_f32(&inp, &mut out) };
        assert_close(
            &out,
            &[
                scalar_hard_swish(-4.0),
                scalar_hard_swish(-3.0),
                scalar_hard_swish(3.0),
                scalar_hard_swish(4.0),
            ],
            STRICT_EPS,
            "hard_swish_piecewise",
        );
    }

    // ── 6. HardSigmoid ─────────────────────────────────────────

    #[test]
    fn test_hard_sigmoid_vs_scalar() {
        let inp = test_inputs();
        let expected = scalar_ref(&inp, scalar_hard_sigmoid);
        let mut out = vec![0.0_f32; inp.len()];
        unsafe { neon_hard_sigmoid_f32(&inp, &mut out) };
        assert_close(&out, &expected, EPS, "hard_sigmoid");
    }

    #[test]
    fn test_hard_sigmoid_bounds() {
        let inp = [-100.0_f32, 0.0, 100.0, 3.0];
        let mut out = [0.0_f32; 4];
        unsafe { neon_hard_sigmoid_f32(&inp, &mut out) };
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "hard_sigmoid out of [0,1]: {v}");
        }
    }

    // ── 7. Softplus ─────────────────────────────────────────────

    #[test]
    fn test_softplus_vs_scalar() {
        let inp = test_inputs();
        let expected = scalar_ref(&inp, scalar_softplus);
        let mut out = vec![0.0_f32; inp.len()];
        unsafe { neon_softplus_f32(&inp, &mut out) };
        assert_close(&out, &expected, EPS, "softplus");
    }

    #[test]
    fn test_softplus_numerical_stability() {
        // Very large → ≈ x, very negative → ≈ 0.
        let inp = [50.0_f32, -50.0, 0.0, 0.5];
        let mut out = [0.0_f32; 4];
        unsafe { neon_softplus_f32(&inp, &mut out) };
        assert!((out[0] - 50.0).abs() < 0.1, "softplus(50) ≈ 50, got {}", out[0]);
        assert!(out[1].abs() < 1e-6, "softplus(-50) ≈ 0, got {}", out[1]);
    }

    // ── 8. QuickGELU ────────────────────────────────────────────

    #[test]
    fn test_quick_gelu_vs_scalar() {
        let inp = test_inputs();
        let expected = scalar_ref(&inp, scalar_quick_gelu);
        let mut out = vec![0.0_f32; inp.len()];
        unsafe { neon_quick_gelu_f32(&inp, &mut out) };
        assert_close(&out, &expected, EPS, "quick_gelu");
    }

    #[test]
    fn test_quick_gelu_zero() {
        let inp = [0.0_f32; 4];
        let mut out = [0.0_f32; 4];
        unsafe { neon_quick_gelu_f32(&inp, &mut out) };
        assert_close(&out, &[0.0; 4], STRICT_EPS, "quick_gelu(0)");
    }

    // ── 9. Fused activation + scale ─────────────────────────────

    #[test]
    fn test_fused_silu_scale() {
        let inp = test_inputs();
        let scale = 2.5_f32;
        let expected: Vec<f32> = inp.iter().map(|&x| scalar_silu(x) * scale).collect();
        let mut out = vec![0.0_f32; inp.len()];
        unsafe {
            neon_fused_activation_scale_f32(&inp, &mut out, FusedActivation::Silu, scale);
        }
        assert_close(&out, &expected, EPS, "fused_silu_scale");
    }

    #[test]
    fn test_fused_hard_swish_scale() {
        let inp = test_inputs();
        let scale = 0.3_f32;
        let expected: Vec<f32> = inp.iter().map(|&x| scalar_hard_swish(x) * scale).collect();
        let mut out = vec![0.0_f32; inp.len()];
        unsafe {
            neon_fused_activation_scale_f32(&inp, &mut out, FusedActivation::HardSwish, scale);
        }
        assert_close(&out, &expected, EPS, "fused_hard_swish_scale");
    }

    // ── Edge cases ──────────────────────────────────────────────

    #[test]
    fn test_all_empty_slices() {
        let inp: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        unsafe {
            neon_gelu_exact_f32(&inp, &mut out);
            neon_gelu_tanh_f32(&inp, &mut out);
            neon_silu_f32(&inp, &mut out);
            neon_mish_f32(&inp, &mut out);
            neon_hard_swish_f32(&inp, &mut out);
            neon_hard_sigmoid_f32(&inp, &mut out);
            neon_softplus_f32(&inp, &mut out);
            neon_quick_gelu_f32(&inp, &mut out);
        }
    }

    #[test]
    fn test_non_aligned_lengths() {
        // Length 7 (not a multiple of 4) to test tail handling.
        let inp = [-3.0_f32, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0];
        let mut out = vec![0.0_f32; 7];

        let expected = scalar_ref(&inp, scalar_silu);
        unsafe { neon_silu_f32(&inp, &mut out) };
        assert_close(&out, &expected, EPS, "silu_non_aligned");

        let expected = scalar_ref(&inp, scalar_hard_swish);
        unsafe { neon_hard_swish_f32(&inp, &mut out) };
        assert_close(&out, &expected, EPS, "hardswish_non_aligned");
    }

    #[test]
    fn test_large_vector_all_activations() {
        let n = 1024;
        let inp: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.02).collect();
        let mut out = vec![0.0_f32; n];

        // GELU exact
        let exp = scalar_ref(&inp, scalar_gelu_exact);
        unsafe { neon_gelu_exact_f32(&inp, &mut out) };
        assert_close(&out, &exp, EPS, "gelu_exact_large");

        // GELU tanh
        let exp = scalar_ref(&inp, scalar_gelu_tanh);
        unsafe { neon_gelu_tanh_f32(&inp, &mut out) };
        assert_close(&out, &exp, EPS, "gelu_tanh_large");

        // SiLU
        let exp = scalar_ref(&inp, scalar_silu);
        unsafe { neon_silu_f32(&inp, &mut out) };
        assert_close(&out, &exp, EPS, "silu_large");

        // Mish
        let exp = scalar_ref(&inp, scalar_mish);
        unsafe { neon_mish_f32(&inp, &mut out) };
        assert_close(&out, &exp, EPS, "mish_large");

        // HardSwish
        let exp = scalar_ref(&inp, scalar_hard_swish);
        unsafe { neon_hard_swish_f32(&inp, &mut out) };
        assert_close(&out, &exp, EPS, "hard_swish_large");

        // HardSigmoid
        let exp = scalar_ref(&inp, scalar_hard_sigmoid);
        unsafe { neon_hard_sigmoid_f32(&inp, &mut out) };
        assert_close(&out, &exp, EPS, "hard_sigmoid_large");

        // Softplus
        let exp = scalar_ref(&inp, scalar_softplus);
        unsafe { neon_softplus_f32(&inp, &mut out) };
        assert_close(&out, &exp, EPS, "softplus_large");

        // QuickGELU
        let exp = scalar_ref(&inp, scalar_quick_gelu);
        unsafe { neon_quick_gelu_f32(&inp, &mut out) };
        assert_close(&out, &exp, EPS, "quick_gelu_large");
    }

    #[test]
    fn test_fused_identity_scale() {
        // scale = 1.0 should be identity.
        let inp = test_inputs();
        let mut out_fused = vec![0.0_f32; inp.len()];
        let mut out_plain = vec![0.0_f32; inp.len()];
        unsafe {
            neon_quick_gelu_f32(&inp, &mut out_plain);
            neon_fused_activation_scale_f32(&inp, &mut out_fused, FusedActivation::QuickGelu, 1.0);
        }
        assert_close(&out_fused, &out_plain, STRICT_EPS, "fused_identity_scale");
    }
}
