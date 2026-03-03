//! AVX-512F accelerated softmax kernels.
//!
//! Every function in this module requires the `avx512f` target feature at
//! compile-time (enforced via `#[target_feature(enable = "avx512f")]`).
//! Call sites must guard with a runtime CPUID check before invoking.

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

/// AVX-512 softmax in-place.
///
/// Processes 16 floats per iteration with a scalar tail loop.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available on the current CPU.
#[target_feature(enable = "avx512f")]
pub unsafe fn softmax_inplace_avx512(xs: &mut [f32]) {
    unsafe {
        let len = xs.len();
        let ptr = xs.as_mut_ptr();

        // --- Pass 1: find max ---
        let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
        let chunks = len / 16;
        for i in 0..chunks {
            let v = _mm512_loadu_ps(ptr.add(i * 16));
            vmax = _mm512_max_ps(vmax, v);
        }
        let mut scalar_max = _mm512_reduce_max_ps(vmax);
        for i in (chunks * 16)..len {
            scalar_max = scalar_max.max(*ptr.add(i));
        }
        let vmax_broadcast = _mm512_set1_ps(scalar_max);

        // --- Pass 2: exp(x - max) and accumulate sum ---
        let mut vsum = _mm512_setzero_ps();
        for i in 0..chunks {
            let v = _mm512_loadu_ps(ptr.add(i * 16));
            let shifted = _mm512_sub_ps(v, vmax_broadcast);
            let exps = exp_approx_avx512(shifted);
            _mm512_storeu_ps(ptr.add(i * 16), exps);
            vsum = _mm512_add_ps(vsum, exps);
        }
        let mut scalar_sum = _mm512_reduce_add_ps(vsum);
        for i in (chunks * 16)..len {
            let e = (*ptr.add(i) - scalar_max).exp();
            *ptr.add(i) = e;
            scalar_sum += e;
        }

        // --- Pass 3: normalise ---
        let inv_sum = _mm512_set1_ps(1.0 / scalar_sum);
        for i in 0..chunks {
            let v = _mm512_loadu_ps(ptr.add(i * 16));
            let normed = _mm512_mul_ps(v, inv_sum);
            _mm512_storeu_ps(ptr.add(i * 16), normed);
        }
        let inv_sum_scalar = 1.0 / scalar_sum;
        for i in (chunks * 16)..len {
            *ptr.add(i) *= inv_sum_scalar;
        }
    }
}

/// Fast exp approximation operating on 16 packed f32s.
///
/// Uses the Schraudolph-style approach:
/// `exp(x) ≈ 2^(x * LOG2_E)` decomposed into integer and fractional parts.
///
/// # Safety
///
/// Requires AVX-512F.
#[target_feature(enable = "avx512f")]
unsafe fn exp_approx_avx512(x: __m512) -> __m512 {
    // Constants
    let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E);
    let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);
    let one = _mm512_set1_ps(1.0);
    let half = _mm512_set1_ps(0.5);

    // Clamp to avoid overflow/underflow
    let x_clamped = _mm512_max_ps(_mm512_set1_ps(-88.0), _mm512_min_ps(_mm512_set1_ps(88.0), x));

    // z = x * log2(e)
    let z = _mm512_mul_ps(x_clamped, log2e);

    // Split into integer and fractional parts: n = round(z), f = z - n
    let n = _mm512_roundscale_ps(z, 0); // round to nearest
    let f = _mm512_sub_ps(z, n);

    // Fractional part: 2^f ≈ 1 + f*ln2 + (f*ln2)^2/2 (degree-2 minimax)
    let f_ln2 = _mm512_mul_ps(f, ln2);
    let poly = _mm512_fmadd_ps(_mm512_mul_ps(f_ln2, f_ln2), half, _mm512_fmadd_ps(f_ln2, one, one));

    // Integer part: 2^n via _mm512_scalef_ps(1.0, n) = 1.0 * 2^n
    let pow2n = _mm512_scalef_ps(one, n);

    _mm512_mul_ps(pow2n, poly)
}
