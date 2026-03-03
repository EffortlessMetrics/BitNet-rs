//! SIMD-optimized softmax operations for CPU inference.
//!
//! Provides numerically stable softmax variants with runtime dispatch to the
//! best available SIMD instruction set (AVX-512 > AVX2 > SSE4.1 > NEON > scalar).
//!
//! # Variants
//!
//! - [`softmax`] \u2014 standard softmax with max-subtraction trick
//! - [`log_softmax`] \u2014 numerically stable log-softmax
//! - [`online_softmax`] \u2014 single-pass online softmax (Milakov & Gimelshein 2018)
//! - [`softmax_with_temperature`] \u2014 temperature-scaled softmax

#![allow(clippy::wildcard_imports)]

// \u2500\u2500 SIMD level detection \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

/// Available SIMD instruction sets, ordered by preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Sse41,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl std::fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scalar => write!(f, "scalar"),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => write!(f, "neon"),
            #[cfg(target_arch = "x86_64")]
            Self::Sse41 => write!(f, "sse4.1"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => write!(f, "avx2"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => write!(f, "avx512"),
        }
    }
}

/// Detect the best SIMD level available at runtime.
#[must_use]
pub fn detect_simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdLevel::Avx2;
        }
        if is_x86_feature_detected!("sse4.1") {
            return SimdLevel::Sse41;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return SimdLevel::Neon;
    }
    SimdLevel::Scalar
}

// \u2500\u2500 Public API (runtime dispatch) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

/// Compute softmax in place using the best available SIMD level.
///
/// Uses the max-subtraction trick for numerical stability.
/// Empty slices are left unchanged.
pub fn softmax(logits: &mut [f32]) {
    dispatch_softmax(logits, detect_simd_level());
}

/// Compute log-softmax in place using the best available SIMD level.
///
/// Numerically stable: `log_softmax(x_i) = x_i - max - log(sum(exp(x_j - max)))`.
/// Empty slices are left unchanged.
pub fn log_softmax(logits: &mut [f32]) {
    dispatch_log_softmax(logits, detect_simd_level());
}

/// Compute softmax in place using the online algorithm (single-pass).
///
/// Based on Milakov & Gimelshein (2018). Computes max and sum of exponentials
/// in a single pass, then normalizes.
/// Empty slices are left unchanged.
pub fn online_softmax(logits: &mut [f32]) {
    dispatch_online_softmax(logits, detect_simd_level());
}

/// Compute temperature-scaled softmax in place.
///
/// Divides logits by `temperature` before applying softmax.
/// A temperature of 0.0 produces a one-hot output at the argmax position.
/// Negative or NaN temperatures leave the slice unchanged.
pub fn softmax_with_temperature(logits: &mut [f32], temperature: f32) {
    if logits.is_empty() || temperature.is_nan() || temperature < 0.0 {
        return;
    }
    if temperature == 0.0 {
        argmax_one_hot(logits);
        return;
    }
    let inv_temp = 1.0 / temperature;
    for v in logits.iter_mut() {
        *v *= inv_temp;
    }
    dispatch_softmax(logits, detect_simd_level());
}

fn dispatch_softmax(logits: &mut [f32], level: SimdLevel) {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => unsafe { avx512::softmax_avx512(logits) },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe { avx2::softmax_avx2(logits) },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse41 => unsafe { sse41::softmax_sse41(logits) },
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe { neon::softmax_neon(logits) },
        SimdLevel::Scalar => scalar::softmax_scalar(logits),
    }
}

fn dispatch_log_softmax(logits: &mut [f32], level: SimdLevel) {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => unsafe { avx512::log_softmax_avx512(logits) },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe { avx2::log_softmax_avx2(logits) },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse41 => unsafe { sse41::log_softmax_sse41(logits) },
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe { neon::log_softmax_neon(logits) },
        SimdLevel::Scalar => scalar::log_softmax_scalar(logits),
    }
}

fn dispatch_online_softmax(logits: &mut [f32], level: SimdLevel) {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => unsafe { avx512::online_softmax_avx512(logits) },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe { avx2::online_softmax_avx2(logits) },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse41 => unsafe { sse41::online_softmax_sse41(logits) },
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe { neon::online_softmax_neon(logits) },
        SimdLevel::Scalar => scalar::online_softmax_scalar(logits),
    }
}

fn argmax_one_hot(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let mut best = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    for v in logits.iter_mut() {
        *v = 0.0;
    }
    logits[best] = 1.0;
}

pub(crate) mod scalar {
    pub fn softmax_scalar(logits: &mut [f32]) {
        if logits.is_empty() {
            return;
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for v in logits.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 && sum.is_finite() {
            let inv = 1.0 / sum;
            for v in logits.iter_mut() {
                *v *= inv;
            }
        }
    }

    pub fn log_softmax_scalar(logits: &mut [f32]) {
        if logits.is_empty() {
            return;
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
        let log_sum_exp = max + sum_exp.ln();
        for v in logits.iter_mut() {
            *v -= log_sum_exp;
        }
    }

    pub fn online_softmax_scalar(logits: &mut [f32]) {
        if logits.is_empty() {
            return;
        }
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0_f32;
        for &v in logits.iter() {
            if v > running_max {
                running_sum *= (running_max - v).exp();
                running_max = v;
            }
            running_sum += (v - running_max).exp();
        }
        if running_sum > 0.0 && running_sum.is_finite() {
            let inv = 1.0 / running_sum;
            for v in logits.iter_mut() {
                *v = (*v - running_max).exp() * inv;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod sse41 {
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn softmax_sse41(logits: &mut [f32]) {
        crate::scalar::softmax_scalar(logits);
    }

    #[target_feature(enable = "sse4.1")]
    pub unsafe fn log_softmax_sse41(logits: &mut [f32]) {
        crate::scalar::log_softmax_scalar(logits);
    }

    #[target_feature(enable = "sse4.1")]
    pub unsafe fn online_softmax_sse41(logits: &mut [f32]) {
        crate::scalar::online_softmax_scalar(logits);
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn softmax_avx2(logits: &mut [f32]) {
        if logits.is_empty() {
            return;
        }
        unsafe {
            let len = logits.len();
            let ptr = logits.as_mut_ptr();
            let chunks = len / 8;
            let remainder = len % 8;

            let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
            for i in 0..chunks {
                let v = _mm256_loadu_ps(ptr.add(i * 8));
                vmax = _mm256_max_ps(vmax, v);
            }
            let mut max_val = horizontal_max_avx2(vmax);
            for i in 0..remainder {
                let v = *ptr.add(chunks * 8 + i);
                if v > max_val {
                    max_val = v;
                }
            }

            let vmax_bc = _mm256_set1_ps(max_val);
            let mut vsum = _mm256_setzero_ps();
            for i in 0..chunks {
                let v = _mm256_loadu_ps(ptr.add(i * 8));
                let shifted = _mm256_sub_ps(v, vmax_bc);
                let e = exp_ps_avx2(shifted);
                _mm256_storeu_ps(ptr.add(i * 8), e);
                vsum = _mm256_add_ps(vsum, e);
            }
            let mut sum_val = horizontal_sum_avx2(vsum);
            for i in 0..remainder {
                let idx = chunks * 8 + i;
                let e = (*ptr.add(idx) - max_val).exp();
                *ptr.add(idx) = e;
                sum_val += e;
            }

            if sum_val > 0.0 && sum_val.is_finite() {
                let inv = _mm256_set1_ps(1.0 / sum_val);
                for i in 0..chunks {
                    let v = _mm256_loadu_ps(ptr.add(i * 8));
                    _mm256_storeu_ps(ptr.add(i * 8), _mm256_mul_ps(v, inv));
                }
                let inv_s = 1.0 / sum_val;
                for i in 0..remainder {
                    let idx = chunks * 8 + i;
                    *ptr.add(idx) *= inv_s;
                }
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn log_softmax_avx2(logits: &mut [f32]) {
        if logits.is_empty() {
            return;
        }
        unsafe {
            let len = logits.len();
            let ptr = logits.as_mut_ptr();
            let chunks = len / 8;
            let remainder = len % 8;

            let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
            for i in 0..chunks {
                let v = _mm256_loadu_ps(ptr.add(i * 8));
                vmax = _mm256_max_ps(vmax, v);
            }
            let mut max_val = horizontal_max_avx2(vmax);
            for i in 0..remainder {
                let v = *ptr.add(chunks * 8 + i);
                if v > max_val {
                    max_val = v;
                }
            }

            let vmax_bc = _mm256_set1_ps(max_val);
            let mut vsum = _mm256_setzero_ps();
            for i in 0..chunks {
                let v = _mm256_loadu_ps(ptr.add(i * 8));
                let shifted = _mm256_sub_ps(v, vmax_bc);
                vsum = _mm256_add_ps(vsum, exp_ps_avx2(shifted));
            }
            let mut sum_val = horizontal_sum_avx2(vsum);
            for i in 0..remainder {
                sum_val += (*ptr.add(chunks * 8 + i) - max_val).exp();
            }

            let log_sum_exp = max_val + sum_val.ln();
            let vlse = _mm256_set1_ps(log_sum_exp);
            for i in 0..chunks {
                let v = _mm256_loadu_ps(ptr.add(i * 8));
                _mm256_storeu_ps(ptr.add(i * 8), _mm256_sub_ps(v, vlse));
            }
            for i in 0..remainder {
                let idx = chunks * 8 + i;
                *ptr.add(idx) -= log_sum_exp;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn online_softmax_avx2(logits: &mut [f32]) {
        crate::scalar::online_softmax_scalar(logits);
    }

    #[allow(clippy::approx_constant)]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn exp_ps_avx2(x: __m256) -> __m256 {
        let one = _mm256_set1_ps(1.0);
        let half = _mm256_set1_ps(0.5);
        let log2ef = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let c1 = _mm256_set1_ps(0.693_359_4);
        let c2 = _mm256_set1_ps(-2.12_e-4);

        let p0 = _mm256_set1_ps(1.988_e-4);
        let p1 = _mm256_set1_ps(1.398_e-3);
        let p2 = _mm256_set1_ps(8.334_e-3);
        let p3 = _mm256_set1_ps(4.167_e-2);
        let p4 = _mm256_set1_ps(1.666_67_e-1);
        let p5 = _mm256_set1_ps(5.0_e-1);

        let x = _mm256_max_ps(_mm256_set1_ps(-87.0), _mm256_min_ps(_mm256_set1_ps(88.0), x));

        let fx = _mm256_round_ps(
            _mm256_add_ps(_mm256_mul_ps(x, log2ef), half),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );

        let x = _mm256_sub_ps(x, _mm256_mul_ps(fx, c1));
        let x = _mm256_sub_ps(x, _mm256_mul_ps(fx, c2));

        let y = _mm256_fmadd_ps(p0, x, p1);
        let y = _mm256_fmadd_ps(y, x, p2);
        let y = _mm256_fmadd_ps(y, x, p3);
        let y = _mm256_fmadd_ps(y, x, p4);
        let y = _mm256_fmadd_ps(y, x, p5);
        let y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, one));

        let n = _mm256_cvtps_epi32(fx);
        let n = _mm256_add_epi32(n, _mm256_set1_epi32(127));
        let n = _mm256_slli_epi32(n, 23);
        let pow2n = _mm256_castsi256_ps(n);

        _mm256_mul_ps(y, pow2n)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_max_avx2(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let mx = _mm_max_ps(lo, hi);
        let shuf = _mm_shuffle_ps(mx, mx, 0b_01_00_11_10);
        let mx = _mm_max_ps(mx, shuf);
        let shuf2 = _mm_shuffle_ps(mx, mx, 0b_00_01_00_01);
        let mx2 = _mm_max_ps(mx, shuf2);
        _mm_cvtss_f32(mx2)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(lo, hi);
        let shuf = _mm_shuffle_ps(s, s, 0b_01_00_11_10);
        let s = _mm_add_ps(s, shuf);
        let shuf2 = _mm_shuffle_ps(s, s, 0b_00_01_00_01);
        let s2 = _mm_add_ps(s, shuf2);
        _mm_cvtss_f32(s2)
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f")]
    pub unsafe fn softmax_avx512(logits: &mut [f32]) {
        if logits.is_empty() {
            return;
        }
        unsafe {
            let len = logits.len();
            let ptr = logits.as_mut_ptr();
            let chunks = len / 16;
            let remainder = len % 16;

            let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
            for i in 0..chunks {
                let v = _mm512_loadu_ps(ptr.add(i * 16));
                vmax = _mm512_max_ps(vmax, v);
            }
            let mut max_val = _mm512_reduce_max_ps(vmax);
            for i in 0..remainder {
                let v = *ptr.add(chunks * 16 + i);
                if v > max_val {
                    max_val = v;
                }
            }

            let vmax_bc = _mm512_set1_ps(max_val);
            let mut vsum = _mm512_setzero_ps();
            for i in 0..chunks {
                let v = _mm512_loadu_ps(ptr.add(i * 16));
                let shifted = _mm512_sub_ps(v, vmax_bc);
                let e = exp_ps_512_scalar(shifted);
                _mm512_storeu_ps(ptr.add(i * 16), e);
                vsum = _mm512_add_ps(vsum, e);
            }
            let mut sum_val = _mm512_reduce_add_ps(vsum);
            for i in 0..remainder {
                let idx = chunks * 16 + i;
                let e = (*ptr.add(idx) - max_val).exp();
                *ptr.add(idx) = e;
                sum_val += e;
            }

            if sum_val > 0.0 && sum_val.is_finite() {
                let inv = _mm512_set1_ps(1.0 / sum_val);
                for i in 0..chunks {
                    let v = _mm512_loadu_ps(ptr.add(i * 16));
                    _mm512_storeu_ps(ptr.add(i * 16), _mm512_mul_ps(v, inv));
                }
                let inv_s = 1.0 / sum_val;
                for i in 0..remainder {
                    let idx = chunks * 16 + i;
                    *ptr.add(idx) *= inv_s;
                }
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn log_softmax_avx512(logits: &mut [f32]) {
        if logits.is_empty() {
            return;
        }
        unsafe {
            let len = logits.len();
            let ptr = logits.as_mut_ptr();
            let chunks = len / 16;
            let remainder = len % 16;

            let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
            for i in 0..chunks {
                let v = _mm512_loadu_ps(ptr.add(i * 16));
                vmax = _mm512_max_ps(vmax, v);
            }
            let mut max_val = _mm512_reduce_max_ps(vmax);
            for i in 0..remainder {
                let v = *ptr.add(chunks * 16 + i);
                if v > max_val {
                    max_val = v;
                }
            }

            let vmax_bc = _mm512_set1_ps(max_val);
            let mut vsum = _mm512_setzero_ps();
            for i in 0..chunks {
                let v = _mm512_loadu_ps(ptr.add(i * 16));
                let shifted = _mm512_sub_ps(v, vmax_bc);
                vsum = _mm512_add_ps(vsum, exp_ps_512_scalar(shifted));
            }
            let mut sum_val = _mm512_reduce_add_ps(vsum);
            for i in 0..remainder {
                sum_val += (*ptr.add(chunks * 16 + i) - max_val).exp();
            }

            let log_sum_exp = max_val + sum_val.ln();
            let vlse = _mm512_set1_ps(log_sum_exp);
            for i in 0..chunks {
                let v = _mm512_loadu_ps(ptr.add(i * 16));
                _mm512_storeu_ps(ptr.add(i * 16), _mm512_sub_ps(v, vlse));
            }
            for i in 0..remainder {
                let idx = chunks * 16 + i;
                *ptr.add(idx) -= log_sum_exp;
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn online_softmax_avx512(logits: &mut [f32]) {
        crate::scalar::online_softmax_scalar(logits);
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn exp_ps_512_scalar(v: __m512) -> __m512 {
        unsafe {
            let mut buf = [0.0_f32; 16];
            _mm512_storeu_ps(buf.as_mut_ptr(), v);
            for x in &mut buf {
                *x = x.exp();
            }
            _mm512_loadu_ps(buf.as_ptr())
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use std::arch::aarch64::*;

    pub unsafe fn softmax_neon(logits: &mut [f32]) {
        if logits.is_empty() {
            return;
        }
        unsafe {
            let len = logits.len();
            let ptr = logits.as_mut_ptr();
            let chunks = len / 4;
            let remainder = len % 4;

            let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
            for i in 0..chunks {
                let v = vld1q_f32(ptr.add(i * 4));
                vmax = vmaxq_f32(vmax, v);
            }
            let mut max_val = vmaxvq_f32(vmax);
            for i in 0..remainder {
                let v = *ptr.add(chunks * 4 + i);
                if v > max_val {
                    max_val = v;
                }
            }

            let vmax_bc = vdupq_n_f32(max_val);
            let mut vsum = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(ptr.add(i * 4));
                let shifted = vsubq_f32(v, vmax_bc);
                let e = exp_ps_neon(shifted);
                vst1q_f32(ptr.add(i * 4), e);
                vsum = vaddq_f32(vsum, e);
            }
            let mut sum_val = vaddvq_f32(vsum);
            for i in 0..remainder {
                let idx = chunks * 4 + i;
                let e = (*ptr.add(idx) - max_val).exp();
                *ptr.add(idx) = e;
                sum_val += e;
            }

            if sum_val > 0.0 && sum_val.is_finite() {
                let inv = vdupq_n_f32(1.0 / sum_val);
                for i in 0..chunks {
                    let v = vld1q_f32(ptr.add(i * 4));
                    vst1q_f32(ptr.add(i * 4), vmulq_f32(v, inv));
                }
                let inv_s = 1.0 / sum_val;
                for i in 0..remainder {
                    let idx = chunks * 4 + i;
                    *ptr.add(idx) *= inv_s;
                }
            }
        }
    }

    pub unsafe fn log_softmax_neon(logits: &mut [f32]) {
        crate::scalar::log_softmax_scalar(logits);
    }

    pub unsafe fn online_softmax_neon(logits: &mut [f32]) {
        crate::scalar::online_softmax_scalar(logits);
    }

    unsafe fn exp_ps_neon(v: float32x4_t) -> float32x4_t {
        unsafe {
            let mut buf = [0.0_f32; 4];
            vst1q_f32(buf.as_mut_ptr(), v);
            for x in &mut buf {
                *x = x.exp();
            }
            vld1q_f32(buf.as_ptr())
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn reference_softmax(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return vec![];
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    fn reference_log_softmax(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return vec![];
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
        let lse = max + sum_exp.ln();
        logits.iter().map(|&x| x - lse).collect()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "index {i}: {x} vs {y} (diff {})", (x - y).abs());
        }
    }

    fn sums_to_one(v: &[f32]) -> bool {
        let s: f32 = v.iter().sum();
        (s - 1.0).abs() < 1e-4
    }

    fn all_non_negative(v: &[f32]) -> bool {
        v.iter().all(|&x| x >= 0.0)
    }

    // ── Scalar softmax tests ────────────────────────────────────────

    #[test]
    fn scalar_softmax_basic() {
        let mut v = vec![1.0, 2.0, 3.0];
        scalar::softmax_scalar(&mut v);
        let expected = reference_softmax(&[1.0, 2.0, 3.0]);
        assert_close(&v, &expected, EPS);
    }

    #[test]
    fn scalar_softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        scalar::softmax_scalar(&mut v);
        assert!(sums_to_one(&v));
    }

    #[test]
    fn scalar_softmax_non_negative() {
        let mut v = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
        scalar::softmax_scalar(&mut v);
        assert!(all_non_negative(&v));
    }

    #[test]
    fn scalar_softmax_empty() {
        let mut v: Vec<f32> = vec![];
        scalar::softmax_scalar(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn scalar_softmax_single_element() {
        let mut v = vec![42.0];
        scalar::softmax_scalar(&mut v);
        assert!((v[0] - 1.0).abs() < EPS);
    }

    #[test]
    fn scalar_softmax_uniform() {
        let mut v = vec![1.0; 100];
        scalar::softmax_scalar(&mut v);
        let expected = 1.0 / 100.0;
        for &x in &v {
            assert!((x - expected).abs() < EPS);
        }
    }

    #[test]
    fn scalar_softmax_large_values() {
        let mut v = vec![1000.0, 1001.0, 1002.0];
        scalar::softmax_scalar(&mut v);
        assert!(sums_to_one(&v));
        assert!(all_non_negative(&v));
    }

    #[test]
    fn scalar_softmax_very_negative() {
        let mut v = vec![-1000.0, -999.0, -998.0];
        scalar::softmax_scalar(&mut v);
        assert!(sums_to_one(&v));
        assert!(all_non_negative(&v));
    }

    #[test]
    fn scalar_softmax_mixed_extreme() {
        let mut v = vec![-100.0, 0.0, 100.0];
        scalar::softmax_scalar(&mut v);
        assert!(sums_to_one(&v));
        assert!(v[2] > 0.99, "largest logit should dominate");
    }

    #[test]
    fn scalar_softmax_preserves_ordering() {
        let mut v = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        scalar::softmax_scalar(&mut v);
        assert!(v[3] > v[4]);
        assert!(v[4] > v[1]);
        assert!(v[1] > v[2]);
        assert!(v[2] > v[0]);
    }

    #[test]
    fn scalar_softmax_nan_input() {
        let mut v = vec![1.0, f32::NAN, 3.0];
        scalar::softmax_scalar(&mut v);
        // NaN propagation — just ensure no panic
    }

    #[test]
    fn scalar_softmax_inf_input() {
        let mut v = vec![1.0, f32::INFINITY, 3.0];
        scalar::softmax_scalar(&mut v);
        assert!((v[1] - 1.0).abs() < EPS || v[1].is_nan());
    }

    #[test]
    fn scalar_softmax_neg_inf_input() {
        let mut v = vec![f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY];
        scalar::softmax_scalar(&mut v);
        assert!((v[1] - 1.0).abs() < EPS);
    }

    #[test]
    fn scalar_softmax_all_same() {
        let mut v = vec![5.0; 8];
        scalar::softmax_scalar(&mut v);
        for &x in &v {
            assert!((x - 0.125).abs() < EPS);
        }
    }

    #[test]
    fn scalar_softmax_two_elements() {
        let mut v = vec![0.0, 0.0];
        scalar::softmax_scalar(&mut v);
        assert!((v[0] - 0.5).abs() < EPS);
        assert!((v[1] - 0.5).abs() < EPS);
    }

    // ── Scalar log-softmax tests ────────────────────────────────────

    #[test]
    fn scalar_log_softmax_basic() {
        let mut v = vec![1.0, 2.0, 3.0];
        scalar::log_softmax_scalar(&mut v);
        let expected = reference_log_softmax(&[1.0, 2.0, 3.0]);
        assert_close(&v, &expected, EPS);
    }

    #[test]
    fn scalar_log_softmax_all_negative() {
        let mut v = vec![1.0, 2.0, 3.0];
        scalar::log_softmax_scalar(&mut v);
        assert!(v.iter().all(|&x| x <= 0.0));
    }

    #[test]
    fn scalar_log_softmax_exp_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        scalar::log_softmax_scalar(&mut v);
        let exp_sum: f32 = v.iter().map(|&x| x.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn scalar_log_softmax_empty() {
        let mut v: Vec<f32> = vec![];
        scalar::log_softmax_scalar(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn scalar_log_softmax_single() {
        let mut v = vec![42.0];
        scalar::log_softmax_scalar(&mut v);
        assert!(v[0].abs() < EPS);
    }

    #[test]
    fn scalar_log_softmax_large_values() {
        let mut v = vec![500.0, 501.0, 502.0];
        scalar::log_softmax_scalar(&mut v);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    // ── Scalar online softmax tests ─────────────────────────────────

    #[test]
    fn scalar_online_softmax_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let expected = reference_softmax(&input);
        let mut v = input;
        scalar::online_softmax_scalar(&mut v);
        assert_close(&v, &expected, EPS);
    }

    #[test]
    fn scalar_online_softmax_matches_standard() {
        let input = vec![0.5, -0.5, 1.5, -1.5, 2.5];
        let mut standard = input.clone();
        let mut online = input;
        scalar::softmax_scalar(&mut standard);
        scalar::online_softmax_scalar(&mut online);
        assert_close(&standard, &online, EPS);
    }

    #[test]
    fn scalar_online_softmax_empty() {
        let mut v: Vec<f32> = vec![];
        scalar::online_softmax_scalar(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn scalar_online_softmax_single() {
        let mut v = vec![99.0];
        scalar::online_softmax_scalar(&mut v);
        assert!((v[0] - 1.0).abs() < EPS);
    }

    #[test]
    fn scalar_online_softmax_large_range() {
        let mut v = vec![-500.0, 0.0, 500.0];
        scalar::online_softmax_scalar(&mut v);
        assert!(sums_to_one(&v));
        assert!(all_non_negative(&v));
    }

    // ── Dispatched softmax tests ────────────────────────────────────

    #[test]
    fn dispatched_softmax_basic() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax(&mut v);
        let expected = reference_softmax(&[1.0, 2.0, 3.0]);
        assert_close(&v, &expected, EPS);
    }

    #[test]
    fn dispatched_softmax_sums_to_one() {
        let mut v = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        softmax(&mut v);
        assert!(sums_to_one(&v));
    }

    #[test]
    fn dispatched_softmax_large_input() {
        let mut v: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
        softmax(&mut v);
        assert!(sums_to_one(&v));
        assert!(all_non_negative(&v));
    }

    #[test]
    fn dispatched_softmax_non_aligned_length() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        softmax(&mut v);
        let expected = reference_softmax(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_close(&v, &expected, 1e-4);
    }

    #[test]
    fn dispatched_softmax_17_elements() {
        let mut v: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let expected = reference_softmax(&v);
        softmax(&mut v);
        assert_close(&v, &expected, 1e-4);
    }

    #[test]
    fn dispatched_softmax_empty() {
        let mut v: Vec<f32> = vec![];
        softmax(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn dispatched_softmax_single() {
        let mut v = vec![5.0];
        softmax(&mut v);
        assert!((v[0] - 1.0).abs() < EPS);
    }

    // ── Dispatched log-softmax tests ────────────────────────────────

    #[test]
    fn dispatched_log_softmax_basic() {
        let mut v = vec![1.0, 2.0, 3.0];
        log_softmax(&mut v);
        let expected = reference_log_softmax(&[1.0, 2.0, 3.0]);
        assert_close(&v, &expected, EPS);
    }

    #[test]
    fn dispatched_log_softmax_all_negative_output() {
        let mut v = vec![10.0, 20.0, 30.0, 40.0];
        log_softmax(&mut v);
        assert!(v.iter().all(|&x| x <= 0.0));
    }

    #[test]
    fn dispatched_log_softmax_exp_sums_to_one() {
        let mut v: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        log_softmax(&mut v);
        let exp_sum: f32 = v.iter().map(|&x| x.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-3);
    }

    #[test]
    fn dispatched_log_softmax_stable_large_values() {
        let mut v = vec![1000.0, 1001.0, 1002.0];
        log_softmax(&mut v);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    // ── Dispatched online softmax tests ─────────────────────────────

    #[test]
    fn dispatched_online_softmax_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected = reference_softmax(&input);
        let mut v = input;
        online_softmax(&mut v);
        assert_close(&v, &expected, EPS);
    }

    #[test]
    fn dispatched_online_softmax_matches_standard() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let mut standard = input.clone();
        let mut online_v = input;
        softmax(&mut standard);
        online_softmax(&mut online_v);
        assert_close(&standard, &online_v, 1e-4);
    }

    // ── Temperature-scaled softmax tests ────────────────────────────

    #[test]
    fn temperature_softmax_t1_equals_standard() {
        let input = vec![1.0, 2.0, 3.0];
        let mut t1 = input.clone();
        let mut std = input;
        softmax_with_temperature(&mut t1, 1.0);
        softmax(&mut std);
        assert_close(&t1, &std, EPS);
    }

    #[test]
    fn temperature_softmax_high_temp_uniform() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        softmax_with_temperature(&mut v, 100.0);
        let range = v.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - v.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(range < 0.05, "high temperature should flatten distribution");
    }

    #[test]
    fn temperature_softmax_low_temp_peaky() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        softmax_with_temperature(&mut v, 0.01);
        assert!(v[4] > 0.99, "low temperature should make max dominate");
    }

    #[test]
    fn temperature_softmax_zero_one_hot() {
        let mut v = vec![1.0, 5.0, 3.0];
        softmax_with_temperature(&mut v, 0.0);
        assert!((v[0]).abs() < EPS);
        assert!((v[1] - 1.0).abs() < EPS);
        assert!((v[2]).abs() < EPS);
    }

    #[test]
    fn temperature_softmax_negative_temp_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut v = original.clone();
        softmax_with_temperature(&mut v, -1.0);
        assert_eq!(v, original);
    }

    #[test]
    fn temperature_softmax_nan_temp_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut v = original.clone();
        softmax_with_temperature(&mut v, f32::NAN);
        assert_eq!(v, original);
    }

    #[test]
    fn temperature_softmax_empty() {
        let mut v: Vec<f32> = vec![];
        softmax_with_temperature(&mut v, 1.0);
        assert!(v.is_empty());
    }

    #[test]
    fn temperature_softmax_sums_to_one() {
        let mut v = vec![0.5, 1.5, 2.5, 3.5];
        softmax_with_temperature(&mut v, 0.7);
        assert!(sums_to_one(&v));
    }

    // ── SIMD level detection tests ──────────────────────────────────

    #[test]
    fn detect_simd_level_returns_valid() {
        let level = detect_simd_level();
        let _ = format!("{level}");
    }

    #[test]
    fn simd_level_display() {
        assert_eq!(SimdLevel::Scalar.to_string(), "scalar");
    }

    // ── Edge case & numerical stability tests ───────────────────────

    #[test]
    fn softmax_all_zeros() {
        let mut v = vec![0.0; 10];
        softmax(&mut v);
        for &x in &v {
            assert!((x - 0.1).abs() < EPS);
        }
    }

    #[test]
    fn softmax_identical_large_values() {
        let mut v = vec![1e30; 5];
        softmax(&mut v);
        for &x in &v {
            assert!((x - 0.2).abs() < EPS);
        }
    }

    #[test]
    fn softmax_one_hot_like_input() {
        let mut v = vec![0.0, 0.0, 1e6, 0.0, 0.0];
        softmax(&mut v);
        assert!(v[2] > 0.999);
    }

    #[test]
    fn log_softmax_consistency_with_softmax() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut sm = input.clone();
        let mut lsm = input;
        softmax(&mut sm);
        log_softmax(&mut lsm);
        for (i, (&s, &ls)) in sm.iter().zip(lsm.iter()).enumerate() {
            assert!(
                (s.ln() - ls).abs() < 1e-4,
                "index {i}: ln(softmax) {:.6} vs log_softmax {:.6}",
                s.ln(),
                ls
            );
        }
    }

    #[test]
    fn softmax_length_3() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax(&mut v);
        assert!(sums_to_one(&v));
    }

    #[test]
    fn softmax_length_5() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        softmax(&mut v);
        assert!(sums_to_one(&v));
    }

    #[test]
    fn softmax_length_8() {
        let mut v: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        softmax(&mut v);
        assert!(sums_to_one(&v));
    }

    #[test]
    fn softmax_length_9() {
        let mut v: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        softmax(&mut v);
        assert!(sums_to_one(&v));
    }

    #[test]
    fn softmax_length_16() {
        let mut v: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        softmax(&mut v);
        assert!(sums_to_one(&v));
    }

    #[test]
    fn softmax_length_33() {
        let mut v: Vec<f32> = (1..=33).map(|i| i as f32).collect();
        let expected = reference_softmax(&v);
        softmax(&mut v);
        assert_close(&v, &expected, 1e-4);
    }

    #[test]
    fn softmax_length_128() {
        let mut v: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
        softmax(&mut v);
        assert!(sums_to_one(&v));
        assert!(all_non_negative(&v));
    }

    #[test]
    fn softmax_length_1000() {
        let mut v: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.001).collect();
        softmax(&mut v);
        assert!(sums_to_one(&v));
    }

    #[test]
    fn softmax_all_neg_inf() {
        let mut v = vec![f32::NEG_INFINITY; 4];
        softmax(&mut v);
        // Degenerate case — just ensure no panic
    }

    #[test]
    fn online_softmax_length_17() {
        let input: Vec<f32> = (0..17).map(|i| i as f32 * 0.5).collect();
        let expected = reference_softmax(&input);
        let mut v = input;
        online_softmax(&mut v);
        assert_close(&v, &expected, EPS);
    }

    #[test]
    fn online_softmax_large_vector() {
        let input: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.01).collect();
        let expected = reference_softmax(&input);
        let mut v = input;
        online_softmax(&mut v);
        assert_close(&v, &expected, 1e-4);
    }

    #[test]
    fn argmax_one_hot_basic() {
        let mut v = vec![1.0, 5.0, 3.0, 2.0];
        argmax_one_hot(&mut v);
        assert_eq!(v, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn argmax_one_hot_empty() {
        let mut v: Vec<f32> = vec![];
        argmax_one_hot(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn argmax_one_hot_single() {
        let mut v = vec![42.0];
        argmax_one_hot(&mut v);
        assert_eq!(v, vec![1.0]);
    }

    // ── proptest property-based tests ───────────────────────────────

    mod prop_tests {
        use proptest::prelude::*;

        fn finite_f32() -> impl Strategy<Value = f32> {
            proptest::num::f32::NORMAL
                .prop_filter("finite", |x: &f32| x.is_finite())
                .prop_map(|x: f32| x.clamp(-500.0, 500.0))
        }

        fn logits_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
            proptest::collection::vec(finite_f32(), min_len..=max_len)
        }

        proptest! {
            #[test]
            fn softmax_sums_to_one(v in logits_vec(1, 256)) {
                let mut buf = v;
                crate::softmax(&mut buf);
                let sum: f32 = buf.iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-3, "sum = {sum}");
            }

            #[test]
            fn softmax_non_negative(v in logits_vec(1, 256)) {
                let mut buf = v;
                crate::softmax(&mut buf);
                for (i, &x) in buf.iter().enumerate() {
                    prop_assert!(x >= 0.0, "index {i}: {x} < 0");
                }
            }

            #[test]
            fn softmax_preserves_ordering(v in logits_vec(2, 64)) {
                let mut buf = v.clone();
                crate::softmax(&mut buf);
                for i in 0..v.len() {
                    for j in 0..v.len() {
                        if v[i] > v[j] {
                            prop_assert!(buf[i] >= buf[j],
                                "ordering violated at [{i}] vs [{j}]"
                            );
                        }
                    }
                }
            }

            #[test]
            fn log_softmax_all_non_positive(v in logits_vec(1, 256)) {
                let mut buf = v;
                crate::log_softmax(&mut buf);
                for (i, &x) in buf.iter().enumerate() {
                    prop_assert!(x <= 1e-5, "index {i}: log_softmax = {x} > 0");
                }
            }

            #[test]
            fn log_softmax_exp_sums_to_one(v in logits_vec(1, 128)) {
                let mut buf = v;
                crate::log_softmax(&mut buf);
                let sum: f32 = buf.iter().map(|&x| x.exp()).sum();
                prop_assert!((sum - 1.0).abs() < 1e-2, "exp sum = {sum}");
            }

            #[test]
            fn online_softmax_matches_standard(v in logits_vec(1, 256)) {
                let mut standard = v.clone();
                let mut online = v;
                crate::softmax(&mut standard);
                crate::online_softmax(&mut online);
                for (i, (&s, &o)) in standard.iter().zip(online.iter()).enumerate() {
                    prop_assert!(
                        (s - o).abs() < 1e-4,
                        "index {i}: standard={s} online={o}"
                    );
                }
            }

            #[test]
            fn temperature_softmax_sums_to_one(
                v in logits_vec(1, 128),
                temp in 0.01_f32..10.0
            ) {
                let mut buf = v;
                crate::softmax_with_temperature(&mut buf, temp);
                let sum: f32 = buf.iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-3, "sum = {sum}");
            }

            #[test]
            fn softmax_max_element_has_max_probability(v in logits_vec(2, 128)) {
                let max_idx = v.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap().0;
                let mut buf = v;
                crate::softmax(&mut buf);
                let max_prob = buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                prop_assert!(
                    (buf[max_idx] - max_prob).abs() < 1e-5,
                    "max logit at {max_idx} should have max probability"
                );
            }

            #[test]
            fn softmax_shift_invariance(
                v in logits_vec(1, 128),
                shift in -100.0_f32..100.0
            ) {
                let mut a = v.clone();
                let mut b: Vec<f32> = v.iter().map(|&x| x + shift).collect();
                crate::softmax(&mut a);
                crate::softmax(&mut b);
                for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
                    prop_assert!(
                        (x - y).abs() < 1e-4,
                        "shift invariance violated at {i}: {x} vs {y}"
                    );
                }
            }
        }
    }
}
