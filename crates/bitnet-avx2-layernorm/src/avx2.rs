//! AVX2-accelerated normalization kernels.
//!
//! All functions require the caller to verify AVX2+FMA availability via
//! `is_x86_feature_detected!` before calling.

#![allow(clippy::cast_precision_loss, unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehdup_ps, _mm_movehl_ps, _mm256_add_ps,
    _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_fnmadd_ps,
    _mm256_loadu_ps, _mm256_mul_ps, _mm256_rsqrt_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps, _mm256_sub_ps,
};

/// AVX2 horizontal sum of 8 packed f32 values.
#[target_feature(enable = "avx2,fma")]
#[cfg(target_arch = "x86_64")]
unsafe fn hsum_avx(vec: __m256) -> f32 {
    // SAFETY: caller guarantees AVX2 is available.
    let hi128 = _mm256_extractf128_ps(vec, 1);
    let lo128 = _mm256_castps256_ps128(vec);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

/// Compute mean using AVX2.
#[target_feature(enable = "avx2,fma")]
#[cfg(target_arch = "x86_64")]
pub unsafe fn mean_avx2(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }
    let mut accum = _mm256_setzero_ps();
    let chunks = len / 8;
    let ptr = data.as_ptr();

    for ci in 0..chunks {
        let loaded = _mm256_loadu_ps(ptr.add(ci * 8));
        accum = _mm256_add_ps(accum, loaded);
    }

    let mut total = hsum_avx(accum);
    for ri in (chunks * 8)..len {
        total += *ptr.add(ri);
    }
    total / len as f32
}

/// Compute variance using AVX2 given a precomputed mean.
#[target_feature(enable = "avx2,fma")]
#[cfg(target_arch = "x86_64")]
pub unsafe fn variance_avx2(data: &[f32], mean_val: f32) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }
    let mean_vec = _mm256_set1_ps(mean_val);
    let mut accum = _mm256_setzero_ps();
    let chunks = len / 8;
    let ptr = data.as_ptr();

    for ci in 0..chunks {
        let loaded = _mm256_loadu_ps(ptr.add(ci * 8));
        let diff = _mm256_sub_ps(loaded, mean_vec);
        accum = _mm256_fmadd_ps(diff, diff, accum);
    }

    let mut total = hsum_avx(accum);
    for ri in (chunks * 8)..len {
        let diff = *ptr.add(ri) - mean_val;
        total = diff.mul_add(diff, total);
    }
    total / len as f32
}

/// Compute mean of squares using AVX2 (for `RMSNorm`).
#[target_feature(enable = "avx2,fma")]
#[cfg(target_arch = "x86_64")]
pub unsafe fn mean_of_squares_avx2(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }
    let mut accum = _mm256_setzero_ps();
    let chunks = len / 8;
    let ptr = data.as_ptr();

    for ci in 0..chunks {
        let loaded = _mm256_loadu_ps(ptr.add(ci * 8));
        accum = _mm256_fmadd_ps(loaded, loaded, accum);
    }

    let mut total = hsum_avx(accum);
    for ri in (chunks * 8)..len {
        let val = *ptr.add(ri);
        total = val.mul_add(val, total);
    }
    total / len as f32
}

/// Fused AVX2 `LayerNorm`: `gamma * (x - mean) / sqrt(var + eps) + beta`
///
/// Single-pass normalize+scale+bias for cache efficiency.
#[target_feature(enable = "avx2,fma")]
#[cfg(target_arch = "x86_64")]
pub unsafe fn layer_norm_avx2(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
    output: &mut [f32],
) {
    let mu = mean_avx2(input);
    let var = variance_avx2(input, mu);
    let inv_std = 1.0 / (var + epsilon).sqrt();

    let len = input.len();
    let mean_vec = _mm256_set1_ps(mu);
    let inv_std_vec = _mm256_set1_ps(inv_std);
    let chunks = len / 8;

    let in_ptr = input.as_ptr();
    let gamma_ptr = gamma.as_ptr();
    let beta_ptr = beta.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for ci in 0..chunks {
        let offset = ci * 8;
        let xv = _mm256_loadu_ps(in_ptr.add(offset));
        let gv = _mm256_loadu_ps(gamma_ptr.add(offset));
        let bv = _mm256_loadu_ps(beta_ptr.add(offset));
        let centered = _mm256_sub_ps(xv, mean_vec);
        let normed = _mm256_mul_ps(centered, inv_std_vec);
        let scaled = _mm256_fmadd_ps(gv, normed, bv);
        _mm256_storeu_ps(out_ptr.add(offset), scaled);
    }

    for ri in (chunks * 8)..len {
        let normed = (*in_ptr.add(ri) - mu) * inv_std;
        *out_ptr.add(ri) = (*gamma_ptr.add(ri)).mul_add(normed, *beta_ptr.add(ri));
    }
}

/// Fused AVX2 `RMSNorm`: `gamma * x / sqrt(mean(x^2) + eps)`
#[target_feature(enable = "avx2,fma")]
#[cfg(target_arch = "x86_64")]
pub unsafe fn rms_norm_avx2(input: &[f32], gamma: &[f32], epsilon: f32, output: &mut [f32]) {
    let ms = mean_of_squares_avx2(input);
    let inv_rms = 1.0 / (ms + epsilon).sqrt();

    let len = input.len();
    let inv_rms_vec = _mm256_set1_ps(inv_rms);
    let chunks = len / 8;

    let in_ptr = input.as_ptr();
    let gamma_ptr = gamma.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for ci in 0..chunks {
        let offset = ci * 8;
        let xv = _mm256_loadu_ps(in_ptr.add(offset));
        let gv = _mm256_loadu_ps(gamma_ptr.add(offset));
        let normed = _mm256_mul_ps(xv, inv_rms_vec);
        let scaled = _mm256_mul_ps(gv, normed);
        _mm256_storeu_ps(out_ptr.add(offset), scaled);
    }

    for ri in (chunks * 8)..len {
        *out_ptr.add(ri) = *gamma_ptr.add(ri) * *in_ptr.add(ri) * inv_rms;
    }
}

/// Fused AVX2 `BatchNorm` per-element:
/// `gamma * (x - running_mean) / sqrt(running_var + eps) + beta`
#[target_feature(enable = "avx2,fma")]
#[cfg(target_arch = "x86_64")]
pub unsafe fn batch_norm_avx2(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    epsilon: f32,
    output: &mut [f32],
) {
    let len = input.len();
    let eps_vec = _mm256_set1_ps(epsilon);
    let chunks = len / 8;

    let in_ptr = input.as_ptr();
    let gamma_ptr = gamma.as_ptr();
    let beta_ptr = beta.as_ptr();
    let mean_ptr = running_mean.as_ptr();
    let var_ptr = running_var.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for ci in 0..chunks {
        let offset = ci * 8;
        let xv = _mm256_loadu_ps(in_ptr.add(offset));
        let gv = _mm256_loadu_ps(gamma_ptr.add(offset));
        let bv = _mm256_loadu_ps(beta_ptr.add(offset));
        let rmv = _mm256_loadu_ps(mean_ptr.add(offset));
        let rvv = _mm256_loadu_ps(var_ptr.add(offset));

        let var_eps = _mm256_add_ps(rvv, eps_vec);
        // Newton-Raphson refined rsqrt for full precision
        let approx = _mm256_rsqrt_ps(var_eps);
        let half = _mm256_set1_ps(0.5);
        let three = _mm256_set1_ps(3.0);
        // inv_std = approx * 0.5 * (3.0 - var_eps * approx * approx)
        let approx_sq = _mm256_mul_ps(approx, approx);
        let sub = _mm256_fnmadd_ps(var_eps, approx_sq, three);
        let inv_std = _mm256_mul_ps(_mm256_mul_ps(approx, half), sub);
        let centered = _mm256_sub_ps(xv, rmv);
        let normed = _mm256_mul_ps(centered, inv_std);
        let scaled = _mm256_fmadd_ps(gv, normed, bv);
        _mm256_storeu_ps(out_ptr.add(offset), scaled);
    }

    for ri in (chunks * 8)..len {
        let inv_std = 1.0 / (*var_ptr.add(ri) + epsilon).sqrt();
        *out_ptr.add(ri) = (*gamma_ptr.add(ri) * (*in_ptr.add(ri) - *mean_ptr.add(ri)))
            .mul_add(inv_std, *beta_ptr.add(ri));
    }
}
