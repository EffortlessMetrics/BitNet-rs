//! `x86_64` SIMD implementations (SSE 4.1, AVX2, AVX-512).

#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

// ── Horizontal helpers ──────────────────────────────────────────────

#[target_feature(enable = "sse4.1")]
unsafe fn hsum_sse(v: __m128) -> f32 {
    let shuf = _mm_movehdup_ps(v);
    let sums = _mm_add_ps(v, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    unsafe {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        hsum_sse(_mm_add_ps(hi, lo))
    }
}

// ── SSE 4.1 ─────────────────────────────────────────────────────────

#[target_feature(enable = "sse4.1")]
pub unsafe fn dot_f32_sse41(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let n = a.len();
        let mut acc = _mm_setzero_ps();
        let chunks = n / 4;
        for i in 0..chunks {
            let va = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
        }
        let mut sum = hsum_sse(acc);
        for i in (chunks * 4)..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn dot_i8_sse41(a: &[i8], b: &[i8]) -> i32 {
    unsafe {
        let n = a.len();
        let mut acc = _mm_setzero_si128();
        let chunks = n / 16;
        for i in 0..chunks {
            let va = _mm_loadu_si128(a.as_ptr().add(i * 16).cast());
            let vb = _mm_loadu_si128(b.as_ptr().add(i * 16).cast());
            let a_lo = _mm_cvtepi8_epi16(va);
            let b_lo = _mm_cvtepi8_epi16(vb);
            let a_hi = _mm_cvtepi8_epi16(_mm_srli_si128::<8>(va));
            let b_hi = _mm_cvtepi8_epi16(_mm_srli_si128::<8>(vb));
            let prod_lo = _mm_madd_epi16(a_lo, b_lo);
            let prod_hi = _mm_madd_epi16(a_hi, b_hi);
            acc = _mm_add_epi32(acc, _mm_add_epi32(prod_lo, prod_hi));
        }
        let shuf = _mm_shuffle_epi32::<0b_00_11_10_01>(acc);
        let sum2 = _mm_add_epi32(acc, shuf);
        let shuf2 = _mm_shuffle_epi32::<0b_00_00_00_10>(sum2);
        let sum4 = _mm_add_epi32(sum2, shuf2);
        let mut sum = _mm_cvtsi128_si32(sum4);
        for i in (chunks * 16)..n {
            sum += i32::from(a[i]) * i32::from(b[i]);
        }
        sum
    }
}

// ── AVX2 ────────────────────────────────────────────────────────────

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let n = a.len();
        let mut acc = _mm256_setzero_ps();
        let chunks = n / 8;
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        let mut sum = hsum_avx2(acc);
        for i in (chunks * 8)..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    unsafe {
        let n = a.len();
        let mut acc = _mm256_setzero_si256();
        let chunks = n / 32;
        for i in 0..chunks {
            let va = _mm256_loadu_si256(a.as_ptr().add(i * 32).cast());
            let vb = _mm256_loadu_si256(b.as_ptr().add(i * 32).cast());
            let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
            let b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
            let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(va));
            let b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(vb));
            let prod_lo = _mm256_madd_epi16(a_lo, b_lo);
            let prod_hi = _mm256_madd_epi16(a_hi, b_hi);
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(prod_lo, prod_hi));
        }
        let hi = _mm256_extracti128_si256::<1>(acc);
        let lo = _mm256_castsi256_si128(acc);
        let sum128 = _mm_add_epi32(hi, lo);
        let shuf = _mm_shuffle_epi32::<0b_00_11_10_01>(sum128);
        let sum2 = _mm_add_epi32(sum128, shuf);
        let shuf2 = _mm_shuffle_epi32::<0b_00_00_00_10>(sum2);
        let sum4 = _mm_add_epi32(sum2, shuf2);
        let mut sum = _mm_cvtsi128_si32(sum4);
        for i in (chunks * 32)..n {
            sum += i32::from(a[i]) * i32::from(b[i]);
        }
        sum
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fma_dot_f32_avx2(a: &[f32], b: &[f32], c: &[f32], d: &[f32]) -> f32 {
    unsafe {
        let n_ab = a.len();
        let n_cd = c.len();

        let mut acc = _mm256_setzero_ps();
        let chunks_ab = n_ab / 8;
        for i in 0..chunks_ab {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        let chunks_cd = n_cd / 8;
        for i in 0..chunks_cd {
            let vc = _mm256_loadu_ps(c.as_ptr().add(i * 8));
            let vd = _mm256_loadu_ps(d.as_ptr().add(i * 8));
            acc = _mm256_fmadd_ps(vc, vd, acc);
        }
        let mut sum = hsum_avx2(acc);

        for i in (chunks_ab * 8)..n_ab {
            sum = a[i].mul_add(b[i], sum);
        }
        for i in (chunks_cd * 8)..n_cd {
            sum = c[i].mul_add(d[i], sum);
        }
        sum
    }
}

// ── AVX-512 ─────────────────────────────────────────────────────────

#[target_feature(enable = "avx512f")]
pub unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let n = a.len();
        let mut acc = _mm512_setzero_ps();
        let chunks = n / 16;
        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn dot_i8_avx512(a: &[i8], b: &[i8]) -> i32 {
    unsafe {
        let n = a.len();
        let mut acc = _mm512_setzero_si512();
        let chunks = n / 64;
        for i in 0..chunks {
            let va = _mm512_loadu_si512(a.as_ptr().add(i * 64).cast());
            let vb = _mm512_loadu_si512(b.as_ptr().add(i * 64).cast());
            let a_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(va));
            let b_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(vb));
            let a_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(va));
            let b_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(vb));
            let prod_lo = _mm512_madd_epi16(a_lo, b_lo);
            let prod_hi = _mm512_madd_epi16(a_hi, b_hi);
            acc = _mm512_add_epi32(acc, _mm512_add_epi32(prod_lo, prod_hi));
        }
        let mut sum = _mm512_reduce_add_epi32(acc);
        for i in (chunks * 64)..n {
            sum += i32::from(a[i]) * i32::from(b[i]);
        }
        sum
    }
}
