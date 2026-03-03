//! ARM NEON int8 matrix multiplication v2 kernels for Apple Silicon.
//!
//! Provides NEON-accelerated int8 matrix multiplication using ARM dot-product
//! instructions (`sdot` / `vdotq_s32`) where available, with scalar fallbacks
//! for portability. All public functions perform runtime feature detection and
//! dispatch to the fastest available implementation.
//!
//! Operations:
//! - [`matmul_i8_f32`] — int8 matmul with f32 output
//! - [`matmul_i8_i32_accum`] — int8 matmul with i32 accumulation
//! - [`gemv_i8_f32`] — int8 matrix-vector multiply (single-column fast path)
//! - [`batched_matmul_i8`] — batch of int8 matmuls
//! - [`symmetric_quantized_matmul`] — symmetric dequant fused with matmul
//! - [`tiled_matmul_i8`] — cache-friendly tiled matmul

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── scalar helpers ─────────────────────────────────────────────────────

/// Scalar i8 dot product of two slices.
#[inline(always)]
fn scalar_dot_i8(a: &[i8], b: &[i8], len: usize) -> i32 {
    let mut acc: i32 = 0;
    for i in 0..len {
        acc += a[i] as i32 * b[i] as i32;
    }
    acc
}

// ── 1. matmul_i8_f32 ──────────────────────────────────────────────────

/// NEON int8 matmul: `C[m×n] = A[m×k] · B[k×n]` with f32 output.
///
/// A is row-major `[m, k]` i8, B is row-major `[k, n]` i8, C is row-major
/// `[m, n]` f32. The scale factor is applied to each output element.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_matmul_i8_f32(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= m * n);

    let chunks = k / 16;
    let tail = k % 16;

    for row in 0..m {
        let a_off = row * k;
        for col in 0..n {
            let mut acc = vdupq_n_s32(0);

            for ch in 0..chunks {
                let base = ch * 16;
                let va = vld1q_s8(a.as_ptr().add(a_off + base));
                // Gather B column
                let mut b_arr = [0i8; 16];
                for i in 0..16 {
                    b_arr[i] = b[(base + i) * n + col];
                }
                let vb = vld1q_s8(b_arr.as_ptr());

                // Widening multiply and accumulate
                let lo_a = vget_low_s8(va);
                let hi_a = vget_high_s8(va);
                let lo_b = vget_low_s8(vb);
                let hi_b = vget_high_s8(vb);

                let prod_lo = vmull_s8(lo_a, lo_b);
                let prod_hi = vmull_s8(hi_a, hi_b);

                acc = vaddq_s32(acc, vpaddlq_s16(prod_lo));
                acc = vaddq_s32(acc, vpaddlq_s16(prod_hi));
            }

            let mut sum = vaddvq_s32(acc);

            // Scalar tail
            let tail_start = chunks * 16;
            for t in 0..tail {
                let idx = tail_start + t;
                sum += a[a_off + idx] as i32 * b[idx * n + col] as i32;
            }

            c[row * n + col] = sum as f32 * scale;
        }
    }
}

fn scalar_matmul_i8_f32(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    for row in 0..m {
        for col in 0..n {
            let mut sum: i32 = 0;
            for i in 0..k {
                sum += a[row * k + i] as i32 * b[i * n + col] as i32;
            }
            c[row * n + col] = sum as f32 * scale;
        }
    }
}

/// Int8 matrix multiply with f32 output.
///
/// Computes `C[m,n] = scale * A[m,k] · B[k,n]` where A and B are i8,
/// C is f32. Uses NEON intrinsics on aarch64, scalar fallback otherwise.
pub fn matmul_i8_f32(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    assert!(a.len() >= m * k, "A too small");
    assert!(b.len() >= k * n, "B too small");
    assert!(c.len() >= m * n, "C too small");

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_matmul_i8_f32(a, b, c, m, k, n, scale) };
        }
    }
    scalar_matmul_i8_f32(a, b, c, m, k, n, scale);
}

// ── 2. matmul_i8_i32_accum ─────────────────────────────────────────────

/// NEON int8 matmul with i32 accumulation (no float conversion).
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_matmul_i8_i32(
    a: &[i8],
    b: &[i8],
    c: &mut [i32],
    m: usize,
    k: usize,
    n: usize,
) {
    let chunks = k / 16;
    let tail = k % 16;

    for row in 0..m {
        let a_off = row * k;
        for col in 0..n {
            let mut acc = vdupq_n_s32(0);

            for ch in 0..chunks {
                let base = ch * 16;
                let va = vld1q_s8(a.as_ptr().add(a_off + base));
                let mut b_arr = [0i8; 16];
                for i in 0..16 {
                    b_arr[i] = b[(base + i) * n + col];
                }
                let vb = vld1q_s8(b_arr.as_ptr());

                let lo_a = vget_low_s8(va);
                let hi_a = vget_high_s8(va);
                let lo_b = vget_low_s8(vb);
                let hi_b = vget_high_s8(vb);

                let prod_lo = vmull_s8(lo_a, lo_b);
                let prod_hi = vmull_s8(hi_a, hi_b);

                acc = vaddq_s32(acc, vpaddlq_s16(prod_lo));
                acc = vaddq_s32(acc, vpaddlq_s16(prod_hi));
            }

            let mut sum = vaddvq_s32(acc);
            let tail_start = chunks * 16;
            for t in 0..tail {
                let idx = tail_start + t;
                sum += a[a_off + idx] as i32 * b[idx * n + col] as i32;
            }

            c[row * n + col] = sum;
        }
    }
}

fn scalar_matmul_i8_i32(
    a: &[i8],
    b: &[i8],
    c: &mut [i32],
    m: usize,
    k: usize,
    n: usize,
) {
    for row in 0..m {
        for col in 0..n {
            let mut sum: i32 = 0;
            for i in 0..k {
                sum += a[row * k + i] as i32 * b[i * n + col] as i32;
            }
            c[row * n + col] = sum;
        }
    }
}

/// Int8 matmul with i32 accumulation for higher precision.
///
/// Computes `C[m,n] = A[m,k] · B[k,n]` where A and B are i8, C is i32.
pub fn matmul_i8_i32_accum(
    a: &[i8],
    b: &[i8],
    c: &mut [i32],
    m: usize,
    k: usize,
    n: usize,
) {
    assert!(a.len() >= m * k, "A too small");
    assert!(b.len() >= k * n, "B too small");
    assert!(c.len() >= m * n, "C too small");

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_matmul_i8_i32(a, b, c, m, k, n) };
        }
    }
    scalar_matmul_i8_i32(a, b, c, m, k, n);
}

// ── 3. gemv_i8_f32 ────────────────────────────────────────────────────

/// NEON int8 GEMV: `y[m] = scale * A[m,k] · x[k]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_gemv_i8_f32(
    a: &[i8],
    x: &[i8],
    y: &mut [f32],
    m: usize,
    k: usize,
    scale: f32,
) {
    let chunks = k / 16;
    let tail = k % 16;

    for row in 0..m {
        let a_off = row * k;
        let mut acc = vdupq_n_s32(0);

        for ch in 0..chunks {
            let base = ch * 16;
            let va = vld1q_s8(a.as_ptr().add(a_off + base));
            let vx = vld1q_s8(x.as_ptr().add(base));

            let lo_a = vget_low_s8(va);
            let hi_a = vget_high_s8(va);
            let lo_x = vget_low_s8(vx);
            let hi_x = vget_high_s8(vx);

            let prod_lo = vmull_s8(lo_a, lo_x);
            let prod_hi = vmull_s8(hi_a, hi_x);

            acc = vaddq_s32(acc, vpaddlq_s16(prod_lo));
            acc = vaddq_s32(acc, vpaddlq_s16(prod_hi));
        }

        let mut sum = vaddvq_s32(acc);
        let tail_start = chunks * 16;
        for t in 0..tail {
            sum += a[a_off + tail_start + t] as i32 * x[tail_start + t] as i32;
        }

        y[row] = sum as f32 * scale;
    }
}

fn scalar_gemv_i8_f32(
    a: &[i8],
    x: &[i8],
    y: &mut [f32],
    m: usize,
    k: usize,
    scale: f32,
) {
    for row in 0..m {
        let mut sum: i32 = 0;
        for i in 0..k {
            sum += a[row * k + i] as i32 * x[i] as i32;
        }
        y[row] = sum as f32 * scale;
    }
}

/// Int8 matrix-vector multiply (GEMV) with f32 output.
///
/// Computes `y[m] = scale * A[m,k] · x[k]`.
/// Optimised single-column case: contiguous vector loads for `x`.
pub fn gemv_i8_f32(
    a: &[i8],
    x: &[i8],
    y: &mut [f32],
    m: usize,
    k: usize,
    scale: f32,
) {
    assert!(a.len() >= m * k, "A too small");
    assert!(x.len() >= k, "x too small");
    assert!(y.len() >= m, "y too small");

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_gemv_i8_f32(a, x, y, m, k, scale) };
        }
    }
    scalar_gemv_i8_f32(a, x, y, m, k, scale);
}

// ── 4. batched_matmul_i8 ──────────────────────────────────────────────

/// Batched int8 matmul: for each batch element, `C_b = scale * A_b · B_b`.
///
/// `a` is `[batch, m, k]`, `b` is `[batch, k, n]`, `c` is `[batch, m, n]`
/// (all contiguous).
pub fn batched_matmul_i8(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;
    assert!(a.len() >= batch * a_stride, "A too small");
    assert!(b.len() >= batch * b_stride, "B too small");
    assert!(c.len() >= batch * c_stride, "C too small");

    for bi in 0..batch {
        let a_slice = &a[bi * a_stride..(bi + 1) * a_stride];
        let b_slice = &b[bi * b_stride..(bi + 1) * b_stride];
        let c_slice = &mut c[bi * c_stride..(bi + 1) * c_stride];
        matmul_i8_f32(a_slice, b_slice, c_slice, m, k, n, scale);
    }
}

// ── 5. symmetric_quantized_matmul ─────────────────────────────────────

/// Symmetric dequantization + matmul fused operation.
///
/// Given int8 matrices A and B with per-tensor symmetric scales `scale_a`
/// and `scale_b`, computes:
///   `C[m,n] = (scale_a * scale_b) * (A_i8[m,k] · B_i8[k,n])`
///
/// This avoids separate dequantization and multiply steps.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_symmetric_quantized_matmul(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale_a: f32,
    scale_b: f32,
) {
    let combined_scale = scale_a * scale_b;
    neon_matmul_i8_f32(a, b, c, m, k, n, combined_scale);
}

fn scalar_symmetric_quantized_matmul(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale_a: f32,
    scale_b: f32,
) {
    let combined_scale = scale_a * scale_b;
    scalar_matmul_i8_f32(a, b, c, m, k, n, combined_scale);
}

/// Symmetric quantized matmul: fuses dequant scales into the matmul.
///
/// `C = (scale_a * scale_b) * A_i8 · B_i8`
pub fn symmetric_quantized_matmul(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale_a: f32,
    scale_b: f32,
) {
    assert!(a.len() >= m * k, "A too small");
    assert!(b.len() >= k * n, "B too small");
    assert!(c.len() >= m * n, "C too small");

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                neon_symmetric_quantized_matmul(a, b, c, m, k, n, scale_a, scale_b)
            };
        }
    }
    scalar_symmetric_quantized_matmul(a, b, c, m, k, n, scale_a, scale_b);
}

// ── 6. tiled_matmul_i8 ────────────────────────────────────────────────

/// Default tile sizes chosen for typical Apple Silicon L1 cache (128 KiB).
pub const DEFAULT_TILE_M: usize = 32;
pub const DEFAULT_TILE_N: usize = 32;
pub const DEFAULT_TILE_K: usize = 64;

/// NEON tiled int8 matmul for cache locality.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_tiled_matmul_i8(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
) {
    // Zero the output
    for v in c[..m * n].iter_mut() {
        *v = 0.0;
    }

    let chunks_per_tile_k = |tk: usize| tk / 16;

    for i0 in (0..m).step_by(tile_m) {
        let i_end = (i0 + tile_m).min(m);
        for j0 in (0..n).step_by(tile_n) {
            let j_end = (j0 + tile_n).min(n);
            for k0 in (0..k).step_by(tile_k) {
                let k_end = (k0 + tile_k).min(k);
                let tk = k_end - k0;
                let chunks = chunks_per_tile_k(tk);
                let tail = tk - chunks * 16;

                for row in i0..i_end {
                    let a_off = row * k + k0;
                    for col in j0..j_end {
                        let mut acc = vdupq_n_s32(0);

                        for ch in 0..chunks {
                            let base = ch * 16;
                            let va = vld1q_s8(a.as_ptr().add(a_off + base));
                            let mut b_arr = [0i8; 16];
                            for bi in 0..16 {
                                b_arr[bi] = b[(k0 + base + bi) * n + col];
                            }
                            let vb = vld1q_s8(b_arr.as_ptr());

                            let lo_a = vget_low_s8(va);
                            let hi_a = vget_high_s8(va);
                            let lo_b = vget_low_s8(vb);
                            let hi_b = vget_high_s8(vb);

                            let prod_lo = vmull_s8(lo_a, lo_b);
                            let prod_hi = vmull_s8(hi_a, hi_b);

                            acc = vaddq_s32(acc, vpaddlq_s16(prod_lo));
                            acc = vaddq_s32(acc, vpaddlq_s16(prod_hi));
                        }

                        let mut sum = vaddvq_s32(acc);
                        let tail_start = chunks * 16;
                        for t in 0..tail {
                            let idx = k0 + tail_start + t;
                            sum += a[row * k + idx] as i32 * b[idx * n + col] as i32;
                        }

                        c[row * n + col] += sum as f32 * scale;
                    }
                }
            }
        }
    }
}

fn scalar_tiled_matmul_i8(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
) {
    for v in c[..m * n].iter_mut() {
        *v = 0.0;
    }

    for i0 in (0..m).step_by(tile_m) {
        let i_end = (i0 + tile_m).min(m);
        for j0 in (0..n).step_by(tile_n) {
            let j_end = (j0 + tile_n).min(n);
            for k0 in (0..k).step_by(tile_k) {
                let k_end = (k0 + tile_k).min(k);
                for row in i0..i_end {
                    for col in j0..j_end {
                        let mut sum: i32 = 0;
                        for i in k0..k_end {
                            sum +=
                                a[row * k + i] as i32 * b[i * n + col] as i32;
                        }
                        c[row * n + col] += sum as f32 * scale;
                    }
                }
            }
        }
    }
}

/// Cache-friendly tiled int8 matmul with configurable tile sizes.
///
/// `C[m,n] = scale * A[m,k] · B[k,n]` using loop tiling for L1 locality.
/// Tiles of size `(tile_m, tile_n, tile_k)` iterate over the K dimension
/// in the innermost loop to maximise register reuse.
pub fn tiled_matmul_i8(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
) {
    assert!(a.len() >= m * k, "A too small");
    assert!(b.len() >= k * n, "B too small");
    assert!(c.len() >= m * n, "C too small");
    assert!(tile_m > 0 && tile_n > 0 && tile_k > 0, "tile sizes must be > 0");

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                neon_tiled_matmul_i8(a, b, c, m, k, n, scale, tile_m, tile_n, tile_k)
            };
        }
    }
    scalar_tiled_matmul_i8(a, b, c, m, k, n, scale, tile_m, tile_n, tile_k);
}

// ── reference scalar matmul for tests ─────────────────────────────────

/// Reference scalar i8 matmul returning f32 results (used in tests).
fn reference_matmul_i8_f32(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum: i32 = 0;
            for i in 0..k {
                sum += a[row * k + i] as i32 * b[i * n + col] as i32;
            }
            out[row * n + col] = sum as f32 * scale;
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn assert_slices_close(got: &[f32], want: &[f32], eps: f32) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                approx_eq(g, w, eps),
                "mismatch at index {i}: got {g}, want {w} (eps={eps})"
            );
        }
    }

    fn assert_slices_eq_i32(got: &[i32], want: &[i32]) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g, w, "mismatch at index {i}");
        }
    }

    /// Deterministic pseudo-random i8 values from a seed.
    fn pseudo_random_i8(len: usize, seed: u64) -> Vec<i8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) as i8).wrapping_add((state >> 16) as i8)
            })
            .collect()
    }

    // ── matmul_i8_f32 tests ───────────────────────────────────────────

    #[test]
    fn test_matmul_i8_f32_1x1x1() {
        let a = [3i8];
        let b = [7i8];
        let mut c = [0.0f32];
        matmul_i8_f32(&a, &b, &mut c, 1, 1, 1, 1.0);
        assert_slices_close(&c, &[21.0], 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_identity_2x2() {
        // A = [[1,0],[0,1]], B = [[5,6],[7,8]]
        let a = [1, 0, 0, 1i8];
        let b = [5, 6, 7, 8i8];
        let mut c = [0.0f32; 4];
        matmul_i8_f32(&a, &b, &mut c, 2, 2, 2, 1.0);
        assert_slices_close(&c, &[5.0, 6.0, 7.0, 8.0], 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_2x3x2() {
        // A(2x3) = [[1,2,3],[4,5,6]]
        // B(3x2) = [[7,8],[9,10],[11,12]]
        // C = [[58,64],[139,154]]
        let a = [1, 2, 3, 4, 5, 6i8];
        let b = [7, 8, 9, 10, 11, 12i8];
        let mut c = [0.0f32; 4];
        matmul_i8_f32(&a, &b, &mut c, 2, 3, 2, 1.0);
        assert_slices_close(&c, &[58.0, 64.0, 139.0, 154.0], 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_with_scale() {
        let a = [1, 2, 3, 4, 5, 6i8];
        let b = [7, 8, 9, 10, 11, 12i8];
        let mut c = [0.0f32; 4];
        matmul_i8_f32(&a, &b, &mut c, 2, 3, 2, 0.5);
        assert_slices_close(&c, &[29.0, 32.0, 69.5, 77.0], 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_negative_values() {
        let a = [-1, 2, -3, 4i8];
        let b = [5, -6, -7, 8i8];
        let mut c = [0.0f32; 4];
        matmul_i8_f32(&a, &b, &mut c, 2, 2, 2, 1.0);
        // row0: (-1*5 + 2*(-7), -1*(-6) + 2*8) = (-19, 22)
        // row1: (-3*5 + 4*(-7), -3*(-6) + 4*8) = (-43, 50)
        assert_slices_close(&c, &[-19.0, 22.0, -43.0, 50.0], 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_zero_matrix() {
        let a = [0i8; 9];
        let b = [1, 2, 3, 4, 5, 6, 7, 8, 9i8];
        let mut c = [999.0f32; 9];
        matmul_i8_f32(&a, &b, &mut c, 3, 3, 3, 1.0);
        assert_slices_close(&c, &[0.0; 9], 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_scale_zero() {
        let a = [1, 2, 3, 4i8];
        let b = [5, 6, 7, 8i8];
        let mut c = [999.0f32; 4];
        matmul_i8_f32(&a, &b, &mut c, 2, 2, 2, 0.0);
        assert_slices_close(&c, &[0.0; 4], 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_negative_scale() {
        let a = [1i8];
        let b = [2i8];
        let mut c = [0.0f32];
        matmul_i8_f32(&a, &b, &mut c, 1, 1, 1, -3.0);
        assert_slices_close(&c, &[-6.0], 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_wide_4x16x4() {
        let m = 4;
        let k = 16;
        let n = 4;
        let a = pseudo_random_i8(m * k, 42);
        let b = pseudo_random_i8(k * n, 99);
        let mut c = vec![0.0f32; m * n];
        matmul_i8_f32(&a, &b, &mut c, m, k, n, 1.0);
        let expected = reference_matmul_i8_f32(&a, &b, m, k, n, 1.0);
        assert_slices_close(&c, &expected, 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_non_aligned_k() {
        // k=17 exercises the scalar tail path
        let m = 2;
        let k = 17;
        let n = 3;
        let a = pseudo_random_i8(m * k, 7);
        let b = pseudo_random_i8(k * n, 13);
        let mut c = vec![0.0f32; m * n];
        matmul_i8_f32(&a, &b, &mut c, m, k, n, 1.0);
        let expected = reference_matmul_i8_f32(&a, &b, m, k, n, 1.0);
        assert_slices_close(&c, &expected, 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_k_less_than_16() {
        let m = 2;
        let k = 7;
        let n = 2;
        let a = pseudo_random_i8(m * k, 1);
        let b = pseudo_random_i8(k * n, 2);
        let mut c = vec![0.0f32; m * n];
        matmul_i8_f32(&a, &b, &mut c, m, k, n, 1.0);
        let expected = reference_matmul_i8_f32(&a, &b, m, k, n, 1.0);
        assert_slices_close(&c, &expected, 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_large_32x64x32() {
        let m = 32;
        let k = 64;
        let n = 32;
        let a = pseudo_random_i8(m * k, 111);
        let b = pseudo_random_i8(k * n, 222);
        let mut c = vec![0.0f32; m * n];
        matmul_i8_f32(&a, &b, &mut c, m, k, n, 0.01);
        let expected = reference_matmul_i8_f32(&a, &b, m, k, n, 0.01);
        assert_slices_close(&c, &expected, 1e-4);
    }

    #[test]
    fn test_matmul_i8_f32_single_row() {
        let k = 20;
        let n = 5;
        let a = pseudo_random_i8(k, 50);
        let b = pseudo_random_i8(k * n, 51);
        let mut c = vec![0.0f32; n];
        matmul_i8_f32(&a, &b, &mut c, 1, k, n, 1.0);
        let expected = reference_matmul_i8_f32(&a, &b, 1, k, n, 1.0);
        assert_slices_close(&c, &expected, 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_single_col() {
        let m = 5;
        let k = 20;
        let a = pseudo_random_i8(m * k, 60);
        let b = pseudo_random_i8(k, 61);
        let mut c = vec![0.0f32; m];
        matmul_i8_f32(&a, &b, &mut c, m, k, 1, 1.0);
        let expected = reference_matmul_i8_f32(&a, &b, m, k, 1, 1.0);
        assert_slices_close(&c, &expected, 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_extreme_values() {
        // i8 extremes: -128 and 127
        let a = [127i8, -128];
        let b = [127i8, -128];
        let mut c = [0.0f32; 4];
        matmul_i8_f32(&a, &b, &mut c, 2, 1, 2, 1.0);
        // row0: 127*127=16129, 127*(-128)=-16256
        // row1: -128*127=-16256, -128*(-128)=16384
        assert_slices_close(&c, &[16129.0, -16256.0, -16256.0, 16384.0], 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_k_exact_16() {
        let m = 2;
        let k = 16;
        let n = 2;
        let a = pseudo_random_i8(m * k, 200);
        let b = pseudo_random_i8(k * n, 201);
        let mut c = vec![0.0f32; m * n];
        matmul_i8_f32(&a, &b, &mut c, m, k, n, 1.0);
        let expected = reference_matmul_i8_f32(&a, &b, m, k, n, 1.0);
        assert_slices_close(&c, &expected, 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_k_exact_32() {
        let m = 3;
        let k = 32;
        let n = 3;
        let a = pseudo_random_i8(m * k, 300);
        let b = pseudo_random_i8(k * n, 301);
        let mut c = vec![0.0f32; m * n];
        matmul_i8_f32(&a, &b, &mut c, m, k, n, 1.0);
        let expected = reference_matmul_i8_f32(&a, &b, m, k, n, 1.0);
        assert_slices_close(&c, &expected, 1e-6);
    }

    #[test]
    fn test_matmul_i8_f32_k_exact_48() {
        let m = 2;
        let k = 48;
        let n = 2;
        let a = pseudo_random_i8(m * k, 400);
        let b = pseudo_random_i8(k * n, 401);
        let mut c = vec![0.0f32; m * n];
        matmul_i8_f32(&a, &b, &mut c, m, k, n, 1.0);
        let expected = reference_matmul_i8_f32(&a, &b, m, k, n, 1.0);
        assert_slices_close(&c, &expected, 1e-6);
    }

    // ── matmul_i8_i32_accum tests ─────────────────────────────────────

    #[test]
    fn test_matmul_i32_1x1x1() {
        let a = [3i8];
        let b = [7i8];
        let mut c = [0i32];
        matmul_i8_i32_accum(&a, &b, &mut c, 1, 1, 1);
        assert_slices_eq_i32(&c, &[21]);
    }

    #[test]
    fn test_matmul_i32_2x3x2() {
        let a = [1, 2, 3, 4, 5, 6i8];
        let b = [7, 8, 9, 10, 11, 12i8];
        let mut c = [0i32; 4];
        matmul_i8_i32_accum(&a, &b, &mut c, 2, 3, 2);
        assert_slices_eq_i32(&c, &[58, 64, 139, 154]);
    }

    #[test]
    fn test_matmul_i32_negative() {
        let a = [-1, 2, -3, 4i8];
        let b = [5, -6, -7, 8i8];
        let mut c = [0i32; 4];
        matmul_i8_i32_accum(&a, &b, &mut c, 2, 2, 2);
        assert_slices_eq_i32(&c, &[-19, 22, -43, 50]);
    }

    #[test]
    fn test_matmul_i32_zero() {
        let a = [0i8; 6];
        let b = [1, 2, 3, 4, 5, 6i8];
        let mut c = [99i32; 4];
        matmul_i8_i32_accum(&a, &b, &mut c, 2, 3, 2);
        assert_slices_eq_i32(&c, &[0, 0, 0, 0]);
    }

    #[test]
    fn test_matmul_i32_extreme_values() {
        let a = [127i8, -128];
        let b = [127i8, -128];
        let mut c = [0i32; 4];
        matmul_i8_i32_accum(&a, &b, &mut c, 2, 1, 2);
        assert_slices_eq_i32(&c, &[16129, -16256, -16256, 16384]);
    }

    #[test]
    fn test_matmul_i32_identity() {
        let a = [1, 0, 0, 1i8];
        let b = [10, 20, 30, 40i8];
        let mut c = [0i32; 4];
        matmul_i8_i32_accum(&a, &b, &mut c, 2, 2, 2);
        assert_slices_eq_i32(&c, &[10, 20, 30, 40]);
    }

    #[test]
    fn test_matmul_i32_large_k() {
        let m = 2;
        let k = 33;
        let n = 2;
        let a = pseudo_random_i8(m * k, 500);
        let b = pseudo_random_i8(k * n, 501);
        let mut c_i32 = vec![0i32; m * n];
        matmul_i8_i32_accum(&a, &b, &mut c_i32, m, k, n);
        // Cross-check with f32 path
        let mut c_f32 = vec![0.0f32; m * n];
        matmul_i8_f32(&a, &b, &mut c_f32, m, k, n, 1.0);
        for i in 0..m * n {
            assert_eq!(c_i32[i], c_f32[i] as i32, "mismatch at {i}");
        }
    }

    #[test]
    fn test_matmul_i32_k_16_aligned() {
        let m = 3;
        let k = 16;
        let n = 2;
        let a = pseudo_random_i8(m * k, 600);
        let b = pseudo_random_i8(k * n, 601);
        let mut c = vec![0i32; m * n];
        matmul_i8_i32_accum(&a, &b, &mut c, m, k, n);
        let ref_f32 = reference_matmul_i8_f32(&a, &b, m, k, n, 1.0);
        for i in 0..m * n {
            assert_eq!(c[i], ref_f32[i] as i32, "mismatch at {i}");
        }
    }

    // ── gemv_i8_f32 tests ──────────────────────────────────────────────

    #[test]
    fn test_gemv_1x1() {
        let a = [5i8];
        let x = [3i8];
        let mut y = [0.0f32];
        gemv_i8_f32(&a, &x, &mut y, 1, 1, 1.0);
        assert_slices_close(&y, &[15.0], 1e-6);
    }

    #[test]
    fn test_gemv_2x3() {
        // A(2x3) = [[1,2,3],[4,5,6]], x = [1,1,1]
        let a = [1, 2, 3, 4, 5, 6i8];
        let x = [1, 1, 1i8];
        let mut y = [0.0f32; 2];
        gemv_i8_f32(&a, &x, &mut y, 2, 3, 1.0);
        assert_slices_close(&y, &[6.0, 15.0], 1e-6);
    }

    #[test]
    fn test_gemv_with_scale() {
        let a = [1, 2, 3, 4, 5, 6i8];
        let x = [1, 1, 1i8];
        let mut y = [0.0f32; 2];
        gemv_i8_f32(&a, &x, &mut y, 2, 3, 2.0);
        assert_slices_close(&y, &[12.0, 30.0], 1e-6);
    }

    #[test]
    fn test_gemv_negative() {
        let a = [-1, 2, 3, -4i8];
        let x = [5, -6i8];
        let mut y = [0.0f32; 2];
        gemv_i8_f32(&a, &x, &mut y, 2, 2, 1.0);
        // row0: -1*5 + 2*(-6) = -17
        // row1: 3*5 + (-4)*(-6) = 39
        assert_slices_close(&y, &[-17.0, 39.0], 1e-6);
    }

    #[test]
    fn test_gemv_zero_vector() {
        let a = [1, 2, 3, 4i8];
        let x = [0, 0i8];
        let mut y = [99.0f32; 2];
        gemv_i8_f32(&a, &x, &mut y, 2, 2, 1.0);
        assert_slices_close(&y, &[0.0, 0.0], 1e-6);
    }

    #[test]
    fn test_gemv_large_k() {
        let m = 4;
        let k = 33;
        let a = pseudo_random_i8(m * k, 700);
        let x = pseudo_random_i8(k, 701);
        let mut y = vec![0.0f32; m];
        gemv_i8_f32(&a, &x, &mut y, m, k, 1.0);
        // Cross-check with matmul (n=1)
        let mut c = vec![0.0f32; m];
        matmul_i8_f32(&a, &x, &mut c, m, k, 1, 1.0);
        assert_slices_close(&y, &c, 1e-6);
    }

    #[test]
    fn test_gemv_k_exact_16() {
        let m = 3;
        let k = 16;
        let a = pseudo_random_i8(m * k, 800);
        let x = pseudo_random_i8(k, 801);
        let mut y = vec![0.0f32; m];
        gemv_i8_f32(&a, &x, &mut y, m, k, 1.0);
        let expected = reference_matmul_i8_f32(&a, &x, m, k, 1, 1.0);
        assert_slices_close(&y, &expected, 1e-6);
    }

    #[test]
    fn test_gemv_k_exact_32() {
        let m = 2;
        let k = 32;
        let a = pseudo_random_i8(m * k, 810);
        let x = pseudo_random_i8(k, 811);
        let mut y = vec![0.0f32; m];
        gemv_i8_f32(&a, &x, &mut y, m, k, 1.0);
        let expected = reference_matmul_i8_f32(&a, &x, m, k, 1, 1.0);
        assert_slices_close(&y, &expected, 1e-6);
    }

    #[test]
    fn test_gemv_k_less_than_16() {
        let m = 3;
        let k = 5;
        let a = pseudo_random_i8(m * k, 820);
        let x = pseudo_random_i8(k, 821);
        let mut y = vec![0.0f32; m];
        gemv_i8_f32(&a, &x, &mut y, m, k, 1.0);
        let expected = reference_matmul_i8_f32(&a, &x, m, k, 1, 1.0);
        assert_slices_close(&y, &expected, 1e-6);
    }

    #[test]
    fn test_gemv_scale_zero() {
        let a = [1, 2, 3, 4i8];
        let x = [5, 6i8];
        let mut y = [99.0f32; 2];
        gemv_i8_f32(&a, &x, &mut y, 2, 2, 0.0);
        assert_slices_close(&y, &[0.0, 0.0], 1e-6);
    }

    #[test]
    fn test_gemv_extreme_values() {
        let a = [127i8, -128];
        let x = [127i8, -128];
        let mut y = [0.0f32; 1];
        gemv_i8_f32(&a, &x, &mut y, 1, 2, 1.0);
        // 127*127 + (-128)*(-128) = 16129 + 16384 = 32513
        assert_slices_close(&y, &[32513.0], 1e-6);
    }

    // ── batched_matmul_i8 tests ────────────────────────────────────────

    #[test]
    fn test_batched_single_batch() {
        let a = [1, 2, 3, 4i8];
        let b = [5, 6, 7, 8i8];
        let mut c = [0.0f32; 4];
        batched_matmul_i8(&a, &b, &mut c, 1, 2, 2, 2, 1.0);
        let mut ref_c = [0.0f32; 4];
        matmul_i8_f32(&a, &b, &mut ref_c, 2, 2, 2, 1.0);
        assert_slices_close(&c, &ref_c, 1e-6);
    }

    #[test]
    fn test_batched_two_batches() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8i8]; // 2 batches of 2x2
        let b = [1, 0, 0, 1, 2, 0, 0, 2i8]; // 2 batches of 2x2
        let mut c = [0.0f32; 8];
        batched_matmul_i8(&a, &b, &mut c, 2, 2, 2, 2, 1.0);
        // batch0: A=[[1,2],[3,4]] * B=[[1,0],[0,1]] = [[1,2],[3,4]]
        // batch1: A=[[5,6],[7,8]] * B=[[2,0],[0,2]] = [[10,12],[14,16]]
        assert_slices_close(&c, &[1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0], 1e-6);
    }

    #[test]
    fn test_batched_with_scale() {
        let a = [1, 2, 3, 4i8];
        let b = [5, 6, 7, 8i8];
        let mut c = [0.0f32; 4];
        batched_matmul_i8(&a, &b, &mut c, 1, 2, 2, 2, 0.5);
        let mut ref_c = [0.0f32; 4];
        matmul_i8_f32(&a, &b, &mut ref_c, 2, 2, 2, 0.5);
        assert_slices_close(&c, &ref_c, 1e-6);
    }

    #[test]
    fn test_batched_three_batches() {
        let m = 2;
        let k = 4;
        let n = 2;
        let batch = 3;
        let a = pseudo_random_i8(batch * m * k, 900);
        let b = pseudo_random_i8(batch * k * n, 901);
        let mut c = vec![0.0f32; batch * m * n];
        batched_matmul_i8(&a, &b, &mut c, batch, m, k, n, 1.0);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let expected = reference_matmul_i8_f32(a_s, b_s, m, k, n, 1.0);
            let c_s = &c[bi * m * n..(bi + 1) * m * n];
            assert_slices_close(c_s, &expected, 1e-6);
        }
    }

    #[test]
    fn test_batched_zero_values() {
        let a = [0i8; 8];
        let b = [1, 2, 3, 4, 5, 6, 7, 8i8];
        let mut c = [99.0f32; 8];
        batched_matmul_i8(&a, &b, &mut c, 2, 2, 2, 2, 1.0);
        assert_slices_close(&c, &[0.0; 8], 1e-6);
    }

    // ── symmetric_quantized_matmul tests ──────────────────────────────

    #[test]
    fn test_symmetric_quant_1x1() {
        let a = [10i8];
        let b = [20i8];
        let mut c = [0.0f32];
        symmetric_quantized_matmul(&a, &b, &mut c, 1, 1, 1, 0.1, 0.2);
        // 10*20 * 0.1 * 0.2 = 200 * 0.02 = 4.0
        assert_slices_close(&c, &[4.0], 1e-5);
    }

    #[test]
    fn test_symmetric_quant_2x2() {
        let a = [1, 2, 3, 4i8];
        let b = [5, 6, 7, 8i8];
        let mut c = [0.0f32; 4];
        symmetric_quantized_matmul(&a, &b, &mut c, 2, 2, 2, 0.5, 0.5);
        // scale = 0.25
        let mut ref_c = [0.0f32; 4];
        matmul_i8_f32(&a, &b, &mut ref_c, 2, 2, 2, 0.25);
        assert_slices_close(&c, &ref_c, 1e-6);
    }

    #[test]
    fn test_symmetric_quant_unit_scales() {
        let a = [1, 2, 3, 4i8];
        let b = [5, 6, 7, 8i8];
        let mut c1 = [0.0f32; 4];
        let mut c2 = [0.0f32; 4];
        symmetric_quantized_matmul(&a, &b, &mut c1, 2, 2, 2, 1.0, 1.0);
        matmul_i8_f32(&a, &b, &mut c2, 2, 2, 2, 1.0);
        assert_slices_close(&c1, &c2, 1e-6);
    }

    #[test]
    fn test_symmetric_quant_scale_commutativity() {
        let a = pseudo_random_i8(6, 1000);
        let b = pseudo_random_i8(6, 1001);
        let mut c1 = [0.0f32; 4];
        let mut c2 = [0.0f32; 4];
        symmetric_quantized_matmul(&a, &b, &mut c1, 2, 3, 2, 0.3, 0.7);
        symmetric_quantized_matmul(&a, &b, &mut c2, 2, 3, 2, 0.7, 0.3);
        assert_slices_close(&c1, &c2, 1e-5);
    }

    #[test]
    fn test_symmetric_quant_zero_scale() {
        let a = [1, 2, 3, 4i8];
        let b = [5, 6, 7, 8i8];
        let mut c = [99.0f32; 4];
        symmetric_quantized_matmul(&a, &b, &mut c, 2, 2, 2, 0.0, 5.0);
        assert_slices_close(&c, &[0.0; 4], 1e-6);
    }

    #[test]
    fn test_symmetric_quant_negative_scale() {
        let a = [1i8];
        let b = [1i8];
        let mut c = [0.0f32];
        symmetric_quantized_matmul(&a, &b, &mut c, 1, 1, 1, -2.0, 3.0);
        assert_slices_close(&c, &[-6.0], 1e-6);
    }

    #[test]
    fn test_symmetric_quant_large() {
        let m = 4;
        let k = 20;
        let n = 4;
        let a = pseudo_random_i8(m * k, 1100);
        let b = pseudo_random_i8(k * n, 1101);
        let mut c = vec![0.0f32; m * n];
        symmetric_quantized_matmul(&a, &b, &mut c, m, k, n, 0.01, 0.02);
        let expected = reference_matmul_i8_f32(&a, &b, m, k, n, 0.01 * 0.02);
        assert_slices_close(&c, &expected, 1e-4);
    }

    // ── tiled_matmul_i8 tests ─────────────────────────────────────────

    #[test]
    fn test_tiled_1x1x1() {
        let a = [3i8];
        let b = [7i8];
        let mut c = [0.0f32];
        tiled_matmul_i8(&a, &b, &mut c, 1, 1, 1, 1.0, 1, 1, 1);
        assert_slices_close(&c, &[21.0], 1e-6);
    }

    #[test]
    fn test_tiled_2x3x2_small_tiles() {
        let a = [1, 2, 3, 4, 5, 6i8];
        let b = [7, 8, 9, 10, 11, 12i8];
        let mut c = [0.0f32; 4];
        tiled_matmul_i8(&a, &b, &mut c, 2, 3, 2, 1.0, 1, 1, 1);
        assert_slices_close(&c, &[58.0, 64.0, 139.0, 154.0], 1e-6);
    }

    #[test]
    fn test_tiled_matches_non_tiled() {
        let m = 8;
        let k = 20;
        let n = 6;
        let a = pseudo_random_i8(m * k, 1200);
        let b = pseudo_random_i8(k * n, 1201);
        let mut c_tiled = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        tiled_matmul_i8(&a, &b, &mut c_tiled, m, k, n, 1.0, 4, 3, 8);
        matmul_i8_f32(&a, &b, &mut c_ref, m, k, n, 1.0);
        assert_slices_close(&c_tiled, &c_ref, 1e-6);
    }

    #[test]
    fn test_tiled_default_tiles() {
        let m = 8;
        let k = 16;
        let n = 8;
        let a = pseudo_random_i8(m * k, 1300);
        let b = pseudo_random_i8(k * n, 1301);
        let mut c = vec![0.0f32; m * n];
        tiled_matmul_i8(
            &a, &b, &mut c, m, k, n, 1.0,
            DEFAULT_TILE_M, DEFAULT_TILE_N, DEFAULT_TILE_K,
        );
        let expected = reference_matmul_i8_f32(&a, &b, m, k, n, 1.0);
        assert_slices_close(&c, &expected, 1e-6);
    }

    #[test]
    fn test_tiled_with_scale() {
        let m = 4;
        let k = 8;
        let n = 4;
        let a = pseudo_random_i8(m * k, 1400);
        let b = pseudo_random_i8(k * n, 1401);
        let mut c = vec![0.0f32; m * n];
        tiled_matmul_i8(&a, &b, &mut c, m, k, n, 0.1, 2, 2, 4);
        let expected = reference_matmul_i8_f32(&a, &b, m, k, n, 0.1);
        assert_slices_close(&c, &expected, 1e-2);
    }

    #[test]
    fn test_tiled_tile_larger_than_matrix() {
        let m = 2;
        let k = 3;
        let n = 2;
        let a = [1, 2, 3, 4, 5, 6i8];
        let b = [7, 8, 9, 10, 11, 12i8];
        let mut c = [0.0f32; 4];
        tiled_matmul_i8(&a, &b, &mut c, m, k, n, 1.0, 64, 64, 64);
        assert_slices_close(&c, &[58.0, 64.0, 139.0, 154.0], 1e-6);
    }

    #[test]
    fn test_tiled_various_tile_sizes() {
        let m = 10;
        let k = 25;
        let n = 10;
        let a = pseudo_random_i8(m * k, 1500);
        let b = pseudo_random_i8(k * n, 1501);
        let expected = reference_matmul_i8_f32(&a, &b, m, k, n, 1.0);

        for &(tm, tn, tk) in &[(1, 1, 1), (2, 5, 3), (5, 5, 5), (10, 10, 25), (3, 7, 11)] {
            let mut c = vec![0.0f32; m * n];
            tiled_matmul_i8(&a, &b, &mut c, m, k, n, 1.0, tm, tn, tk);
            assert_slices_close(&c, &expected, 1e-5);
        }
    }

    #[test]
    fn test_tiled_zero_matrix() {
        let a = [0i8; 16];
        let b = pseudo_random_i8(16, 1600);
        let mut c = [99.0f32; 16];
        tiled_matmul_i8(&a, &b, &mut c, 4, 4, 4, 1.0, 2, 2, 2);
        assert_slices_close(&c, &[0.0; 16], 1e-6);
    }

    #[test]
    fn test_tiled_negative_scale() {
        let a = [1, 2, 3, 4i8];
        let b = [1, 0, 0, 1i8];
        let mut c = [0.0f32; 4];
        tiled_matmul_i8(&a, &b, &mut c, 2, 2, 2, -1.0, 2, 2, 2);
        assert_slices_close(&c, &[-1.0, -2.0, -3.0, -4.0], 1e-6);
    }

    // ── cross-function consistency tests ──────────────────────────────

    #[test]
    fn test_gemv_matches_matmul_n1() {
        let m = 5;
        let k = 19;
        let a = pseudo_random_i8(m * k, 2000);
        let x = pseudo_random_i8(k, 2001);
        let mut y = vec![0.0f32; m];
        let mut c = vec![0.0f32; m];
        gemv_i8_f32(&a, &x, &mut y, m, k, 1.0);
        matmul_i8_f32(&a, &x, &mut c, m, k, 1, 1.0);
        assert_slices_close(&y, &c, 1e-6);
    }

    #[test]
    fn test_tiled_matches_matmul_exact() {
        let m = 6;
        let k = 18;
        let n = 4;
        let a = pseudo_random_i8(m * k, 2100);
        let b = pseudo_random_i8(k * n, 2101);
        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        matmul_i8_f32(&a, &b, &mut c1, m, k, n, 1.5);
        tiled_matmul_i8(&a, &b, &mut c2, m, k, n, 1.5, 3, 2, 6);
        assert_slices_close(&c1, &c2, 1e-4);
    }

    #[test]
    fn test_batched_matches_individual() {
        let m = 2;
        let k = 5;
        let n = 2;
        let batch = 4;
        let a = pseudo_random_i8(batch * m * k, 2200);
        let b = pseudo_random_i8(batch * k * n, 2201);
        let mut c_batched = vec![0.0f32; batch * m * n];
        batched_matmul_i8(&a, &b, &mut c_batched, batch, m, k, n, 1.0);
        for bi in 0..batch {
            let a_s = &a[bi * m * k..(bi + 1) * m * k];
            let b_s = &b[bi * k * n..(bi + 1) * k * n];
            let mut c_single = vec![0.0f32; m * n];
            matmul_i8_f32(a_s, b_s, &mut c_single, m, k, n, 1.0);
            let c_s = &c_batched[bi * m * n..(bi + 1) * m * n];
            assert_slices_close(c_s, &c_single, 1e-6);
        }
    }

    #[test]
    fn test_i32_matches_f32_cast() {
        let m = 3;
        let k = 10;
        let n = 3;
        let a = pseudo_random_i8(m * k, 2300);
        let b = pseudo_random_i8(k * n, 2301);
        let mut c_i32 = vec![0i32; m * n];
        let mut c_f32 = vec![0.0f32; m * n];
        matmul_i8_i32_accum(&a, &b, &mut c_i32, m, k, n);
        matmul_i8_f32(&a, &b, &mut c_f32, m, k, n, 1.0);
        for i in 0..m * n {
            assert_eq!(c_i32[i], c_f32[i] as i32, "mismatch at {i}");
        }
    }

    // ── edge-case and panic tests ─────────────────────────────────────

    #[test]
    #[should_panic(expected = "A too small")]
    fn test_matmul_f32_panics_a_small() {
        let a = [1i8; 3];
        let b = [1i8; 4];
        let mut c = [0.0f32; 4];
        matmul_i8_f32(&a, &b, &mut c, 2, 2, 2, 1.0);
    }

    #[test]
    #[should_panic(expected = "B too small")]
    fn test_matmul_f32_panics_b_small() {
        let a = [1i8; 4];
        let b = [1i8; 3];
        let mut c = [0.0f32; 4];
        matmul_i8_f32(&a, &b, &mut c, 2, 2, 2, 1.0);
    }

    #[test]
    #[should_panic(expected = "C too small")]
    fn test_matmul_f32_panics_c_small() {
        let a = [1i8; 4];
        let b = [1i8; 4];
        let mut c = [0.0f32; 3];
        matmul_i8_f32(&a, &b, &mut c, 2, 2, 2, 1.0);
    }

    #[test]
    #[should_panic(expected = "A too small")]
    fn test_gemv_panics_a_small() {
        let a = [1i8; 3];
        let x = [1i8; 2];
        let mut y = [0.0f32; 2];
        gemv_i8_f32(&a, &x, &mut y, 2, 2, 1.0);
    }

    #[test]
    #[should_panic(expected = "x too small")]
    fn test_gemv_panics_x_small() {
        let a = [1i8; 4];
        let x = [1i8; 1];
        let mut y = [0.0f32; 2];
        gemv_i8_f32(&a, &x, &mut y, 2, 2, 1.0);
    }

    #[test]
    #[should_panic(expected = "y too small")]
    fn test_gemv_panics_y_small() {
        let a = [1i8; 4];
        let x = [1i8; 2];
        let mut y = [0.0f32; 1];
        gemv_i8_f32(&a, &x, &mut y, 2, 2, 1.0);
    }

    #[test]
    #[should_panic(expected = "tile sizes must be > 0")]
    fn test_tiled_panics_zero_tile() {
        let a = [1i8; 4];
        let b = [1i8; 4];
        let mut c = [0.0f32; 4];
        tiled_matmul_i8(&a, &b, &mut c, 2, 2, 2, 1.0, 0, 2, 2);
    }

    #[test]
    #[should_panic(expected = "A too small")]
    fn test_i32_panics_a_small() {
        let a = [1i8; 2];
        let b = [1i8; 4];
        let mut c = [0i32; 4];
        matmul_i8_i32_accum(&a, &b, &mut c, 2, 2, 2);
    }

    #[test]
    #[should_panic(expected = "A too small")]
    fn test_batched_panics_a_small() {
        let a = [1i8; 4];
        let b = [1i8; 8];
        let mut c = [0.0f32; 8];
        batched_matmul_i8(&a, &b, &mut c, 2, 2, 2, 2, 1.0);
    }

    #[test]
    #[should_panic(expected = "A too small")]
    fn test_symmetric_panics_a_small() {
        let a = [1i8; 2];
        let b = [1i8; 4];
        let mut c = [0.0f32; 4];
        symmetric_quantized_matmul(&a, &b, &mut c, 2, 2, 2, 1.0, 1.0);
    }

    // ── scalar_dot_i8 unit test ───────────────────────────────────────

    #[test]
    fn test_scalar_dot_i8_basic() {
        let a = [1, 2, 3i8];
        let b = [4, 5, 6i8];
        assert_eq!(scalar_dot_i8(&a, &b, 3), 32); // 4+10+18
    }

    #[test]
    fn test_scalar_dot_i8_negative() {
        let a = [-1, 2i8];
        let b = [3, -4i8];
        assert_eq!(scalar_dot_i8(&a, &b, 2), -11); // -3 + -8
    }

    #[test]
    fn test_scalar_dot_i8_zero_len() {
        let a = [1, 2i8];
        let b = [3, 4i8];
        assert_eq!(scalar_dot_i8(&a, &b, 0), 0);
    }

    // ── DEFAULT_TILE constants ─────────────────────────────────────────

    #[test]
    fn test_default_tile_constants() {
        assert!(DEFAULT_TILE_M > 0);
        assert!(DEFAULT_TILE_N > 0);
        assert!(DEFAULT_TILE_K > 0);
        assert_eq!(DEFAULT_TILE_M, 32);
        assert_eq!(DEFAULT_TILE_N, 32);
        assert_eq!(DEFAULT_TILE_K, 64);
    }
}
