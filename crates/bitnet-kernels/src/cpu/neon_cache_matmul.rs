//! Cache-friendly matrix multiplication kernel optimized for ARM NEON on Apple Silicon.
//!
//! Uses tiling/blocking to keep working sets within L1 cache (128 KB on Apple Silicon
//! M-series). Computes C = A × Bᵀ where A is (M×K) row-major and B is (N×K) row-major
//! (i.e. Bᵀ is K×N, so each row of B is dotted against each row of A).

use std::arch::aarch64::*;

/// Default tile size chosen for Apple Silicon's 128 KB L1 data cache.
/// A 32×32 tile of f32 occupies 4 KB, so three tiles (A, B, C) fit comfortably.
const DEFAULT_TILE_SIZE: usize = 32;

/// Cache-friendly matrix multiplication: C = A × Bᵀ.
///
/// * `a` — row-major M×K matrix
/// * `b` — row-major N×K matrix (each row is a column of Bᵀ)
/// * `m` — number of rows of A (and C)
/// * `n` — number of rows of B (columns of Bᵀ, and columns of C)
/// * `k` — shared inner dimension
///
/// Returns a newly allocated M×N row-major result.
///
/// # Panics
///
/// Panics if `a.len() < m * k` or `b.len() < n * k`.
pub fn cache_friendly_matmul_neon(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert!(a.len() >= m * k, "a too short: {} < {}", a.len(), m * k);
    assert!(b.len() >= n * k, "b too short: {} < {}", b.len(), n * k);

    let mut c = vec![0.0f32; m * n];
    tiled_matmul_neon(a, b, &mut c, m, n, k, DEFAULT_TILE_SIZE);
    c
}

/// Tiled (blocked) matrix multiplication: C += A × Bᵀ with explicit tile size.
///
/// Accumulates into `c`; caller must zero-initialise `c` if a fresh result is needed.
///
/// # Panics
///
/// Panics if slice lengths are too small or `tile_size` is zero.
pub fn tiled_matmul_neon(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    tile_size: usize,
) {
    assert!(tile_size > 0, "tile_size must be > 0");
    assert!(a.len() >= m * k, "a too short: {} < {}", a.len(), m * k);
    assert!(b.len() >= n * k, "b too short: {} < {}", b.len(), n * k);
    assert!(c.len() >= m * n, "c too short: {} < {}", c.len(), m * n);

    // Iterate over tiles along all three dimensions.
    let mut i_tile = 0;
    while i_tile < m {
        let i_end = (i_tile + tile_size).min(m);

        let mut j_tile = 0;
        while j_tile < n {
            let j_end = (j_tile + tile_size).min(n);

            let mut kk_tile = 0;
            while kk_tile < k {
                let kk_end = (kk_tile + tile_size).min(k);

                // Process one micro-block.
                for i in i_tile..i_end {
                    let a_row = &a[i * k..];
                    for j in j_tile..j_end {
                        let b_row = &b[j * k..];

                        // SAFETY: NEON is always available on aarch64.
                        let dot = unsafe { neon_dot_partial(a_row, b_row, kk_tile, kk_end) };
                        c[i * n + j] += dot;
                    }
                }

                kk_tile += tile_size;
            }
            j_tile += tile_size;
        }
        i_tile += tile_size;
    }
}

/// Compute the partial dot product of `a[start..end]` and `b[start..end]` using NEON.
///
/// # Safety
///
/// Requires the `neon` target feature (always present on aarch64).
#[target_feature(enable = "neon")]
unsafe fn neon_dot_partial(a: &[f32], b: &[f32], start: usize, end: usize) -> f32 {
    let len = end - start;
    let a_ptr = unsafe { a.as_ptr().add(start) };
    let b_ptr = unsafe { b.as_ptr().add(start) };

    let mut offset = 0usize;
    // Accumulate in a 4-wide NEON register.
    let mut acc = vdupq_n_f32(0.0);

    // Process 16 floats (4 × float32x4) per iteration for better ILP.
    let chunk16 = len / 16;
    for _ in 0..chunk16 {
        let a0 = unsafe { vld1q_f32(a_ptr.add(offset)) };
        let b0 = unsafe { vld1q_f32(b_ptr.add(offset)) };
        acc = vfmaq_f32(acc, a0, b0);

        let a1 = unsafe { vld1q_f32(a_ptr.add(offset + 4)) };
        let b1 = unsafe { vld1q_f32(b_ptr.add(offset + 4)) };
        acc = vfmaq_f32(acc, a1, b1);

        let a2 = unsafe { vld1q_f32(a_ptr.add(offset + 8)) };
        let b2 = unsafe { vld1q_f32(b_ptr.add(offset + 8)) };
        acc = vfmaq_f32(acc, a2, b2);

        let a3 = unsafe { vld1q_f32(a_ptr.add(offset + 12)) };
        let b3 = unsafe { vld1q_f32(b_ptr.add(offset + 12)) };
        acc = vfmaq_f32(acc, a3, b3);

        offset += 16;
    }

    // Process remaining groups of 4.
    while offset + 4 <= len {
        let va = unsafe { vld1q_f32(a_ptr.add(offset)) };
        let vb = unsafe { vld1q_f32(b_ptr.add(offset)) };
        acc = vfmaq_f32(acc, va, vb);
        offset += 4;
    }

    // Horizontal reduction of the 4-lane accumulator.
    let mut sum = vaddvq_f32(acc);

    // Scalar tail for remaining 0–3 elements.
    while offset < len {
        sum += unsafe { *a_ptr.add(offset) * *b_ptr.add(offset) };
        offset += 1;
    }

    sum
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Naïve reference: C = A × Bᵀ (no SIMD, no tiling).
    fn naive_matmul_abt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[j * k + p];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    fn assert_matrices_close(expected: &[f32], actual: &[f32], tol: f32) {
        assert_eq!(expected.len(), actual.len(), "length mismatch");
        for (idx, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (e - a).abs();
            assert!(diff <= tol, "mismatch at index {idx}: expected {e}, got {a} (diff {diff})");
        }
    }

    #[test]
    fn test_small_square_matrix() {
        // 2×3 times 2×3 (A*Bᵀ → 2×2)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expected = naive_matmul_abt(&a, &b, 2, 2, 3);
        let result = cache_friendly_matmul_neon(&a, &b, 2, 2, 3);
        assert_matrices_close(&expected, &result, 1e-5);
    }

    #[test]
    fn test_identity_matmul() {
        // A=I(4×4), B=arbitrary 4×4 → C should equal B
        let a =
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let expected = naive_matmul_abt(&a, &b, 4, 4, 4);
        let result = cache_friendly_matmul_neon(&a, &b, 4, 4, 4);
        assert_matrices_close(&expected, &result, 1e-5);
    }

    #[test]
    fn test_non_square_dimensions() {
        // A: 5×7, B: 3×7 → C: 5×3
        let m = 5;
        let n = 3;
        let k = 7;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.2 - 1.0).collect();
        let expected = naive_matmul_abt(&a, &b, m, n, k);
        let result = cache_friendly_matmul_neon(&a, &b, m, n, k);
        assert_matrices_close(&expected, &result, 1e-4);
    }

    #[test]
    fn test_tile_aligned_large() {
        // Exactly tile-aligned: 32×32
        let m = 32;
        let n = 32;
        let k = 32;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.3 - 2.0).collect();
        let b: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32) * 0.2 + 0.5).collect();
        let expected = naive_matmul_abt(&a, &b, m, n, k);
        let result = cache_friendly_matmul_neon(&a, &b, m, n, k);
        assert_matrices_close(&expected, &result, 1e-3);
    }

    #[test]
    fn test_non_tile_aligned() {
        // Not aligned to any power-of-two: 37×41 inner dim 19
        let m = 37;
        let n = 41;
        let k = 19;
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32).sin())).collect();
        let b: Vec<f32> = (0..n * k).map(|i| ((i as f32).cos())).collect();
        let expected = naive_matmul_abt(&a, &b, m, n, k);
        let result = cache_friendly_matmul_neon(&a, &b, m, n, k);
        assert_matrices_close(&expected, &result, 1e-3);
    }

    #[test]
    fn test_single_element() {
        let a = vec![3.0];
        let b = vec![5.0];
        let result = cache_friendly_matmul_neon(&a, &b, 1, 1, 1);
        assert_matrices_close(&[15.0], &result, 1e-6);
    }

    #[test]
    fn test_tiled_with_custom_tile_size() {
        let m = 10;
        let n = 12;
        let k = 8;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.5).collect();
        let expected = naive_matmul_abt(&a, &b, m, n, k);

        // Test with small tile size 4 (causes many tile iterations).
        let mut c = vec![0.0f32; m * n];
        tiled_matmul_neon(&a, &b, &mut c, m, n, k, 4);
        assert_matrices_close(&expected, &c, 1e-2);

        // Test with tile size larger than dimensions.
        let mut c2 = vec![0.0f32; m * n];
        tiled_matmul_neon(&a, &b, &mut c2, m, n, k, 64);
        assert_matrices_close(&expected, &c2, 1e-2);
    }

    #[test]
    fn test_zeros() {
        let m = 8;
        let n = 8;
        let k = 8;
        let a = vec![0.0f32; m * k];
        let b = vec![1.0f32; n * k];
        let result = cache_friendly_matmul_neon(&a, &b, m, n, k);
        assert_matrices_close(&vec![0.0f32; m * n], &result, 1e-6);
    }

    #[test]
    fn test_larger_than_tile() {
        // Larger than default tile (32) to exercise multi-tile path.
        let m = 50;
        let n = 48;
        let k = 65;
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 100) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..n * k).map(|i| ((i * 11 + 5) % 100) as f32 * 0.01).collect();
        let expected = naive_matmul_abt(&a, &b, m, n, k);
        let result = cache_friendly_matmul_neon(&a, &b, m, n, k);
        assert_matrices_close(&expected, &result, 1e-2);
    }

    #[test]
    fn test_accumulation_into_existing() {
        // Verify tiled_matmul_neon *accumulates* rather than overwrites.
        let m = 4;
        let n = 4;
        let k = 4;
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; n * k];
        let mut c = vec![100.0f32; m * n];
        tiled_matmul_neon(&a, &b, &mut c, m, n, k, 4);
        // Each dot product of two all-ones length-4 vectors = 4.0, plus initial 100.
        for &v in &c {
            assert!((v - 104.0).abs() < 1e-5, "expected 104.0, got {v}");
        }
    }

    #[test]
    #[should_panic(expected = "a too short")]
    fn test_panics_on_short_a() {
        let a = vec![1.0; 3]; // too short for 2×3
        let b = vec![1.0; 6];
        cache_friendly_matmul_neon(&a, &b, 2, 2, 3);
    }

    #[test]
    #[should_panic(expected = "b too short")]
    fn test_panics_on_short_b() {
        let a = vec![1.0; 6];
        let b = vec![1.0; 3]; // too short for 2×3
        cache_friendly_matmul_neon(&a, &b, 2, 2, 3);
    }
}
