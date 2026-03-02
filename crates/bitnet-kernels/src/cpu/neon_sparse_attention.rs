//! ARM NEON optimized sparse attention kernels for Apple Silicon.
//!
//! Provides SIMD-accelerated sparse attention operations using `float32x4`
//! NEON intrinsics for 4-wide parallel computation. Includes masked
//! attention scores, sliding window attention, block-sparse matrix
//! multiplication, and local attention with configurable window size.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── Helpers ─────────────────────────────────────────────────────────────

/// Horizontal sum of a `float32x4_t` (scalar fallback-free).
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    let pair = vpaddq_f32(v, v);
    vgetq_lane_f32(vpaddq_f32(pair, pair), 0)
}

/// NEON dot product of two f32 slices (same length, ≥ 1).
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / LANES;
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * LANES;
        let va = vld1q_f32(a.as_ptr().add(off));
        let vb = vld1q_f32(b.as_ptr().add(off));
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut sum = hsum_f32x4(acc);
    for i in (chunks * LANES)..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar dot product (non-NEON fallback for completeness).
fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// In-place softmax over `scores` (finite-length, non-empty).
fn softmax_inplace(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    if sum > 0.0 {
        for s in scores.iter_mut() {
            *s /= sum;
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────

/// Compute attention scores with a sparsity mask.
///
/// For each query position `q` and key position `k`, the raw score is
/// `dot(query[q], key[k]) / sqrt(head_dim)` when `mask[q * seq_len + k]`
/// is `true`, otherwise `-inf`. A row-wise softmax is applied afterwards.
///
/// Returns a flat `seq_len × seq_len` score matrix (row-major).
///
/// # Panics
/// Panics if slice lengths are inconsistent with `head_dim` / `seq_len`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sparse_attention_scores(
    query: &[f32],
    key: &[f32],
    mask: &[bool],
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert_eq!(query.len(), seq_len * head_dim);
    assert_eq!(key.len(), seq_len * head_dim);
    assert_eq!(mask.len(), seq_len * seq_len);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len * seq_len];

    for q in 0..seq_len {
        let q_row = &query[q * head_dim..(q + 1) * head_dim];
        for k in 0..seq_len {
            let idx = q * seq_len + k;
            if mask[idx] {
                let k_row = &key[k * head_dim..(k + 1) * head_dim];
                scores[idx] = neon_dot(q_row, k_row) * scale;
            } else {
                scores[idx] = f32::NEG_INFINITY;
            }
        }
        softmax_inplace(&mut scores[q * seq_len..(q + 1) * seq_len]);
    }
    scores
}

/// Sliding window attention.
///
/// Each query position attends only to key positions within
/// `[max(0, q - window_size + 1) ..= q]`. The output is
/// `seq_len × head_dim` (row-major).
///
/// # Panics
/// Panics if slice lengths are inconsistent with `head_dim` / `seq_len`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sliding_window_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    window_size: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert_eq!(query.len(), seq_len * head_dim);
    assert_eq!(key.len(), seq_len * head_dim);
    assert_eq!(value.len(), seq_len * head_dim);
    assert!(window_size > 0);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for q in 0..seq_len {
        let q_row = &query[q * head_dim..(q + 1) * head_dim];
        let start = q.saturating_sub(window_size - 1);
        let win_len = q - start + 1;

        // Compute scores for the window.
        let mut scores = vec![0.0f32; win_len];
        for (wi, k) in (start..=q).enumerate() {
            let k_row = &key[k * head_dim..(k + 1) * head_dim];
            scores[wi] = neon_dot(q_row, k_row) * scale;
        }
        softmax_inplace(&mut scores);

        // Weighted sum of values.
        let out_row = &mut output[q * head_dim..(q + 1) * head_dim];
        for (wi, k) in (start..=q).enumerate() {
            let v_row = &value[k * head_dim..(k + 1) * head_dim];
            let w = scores[wi];
            let chunks = head_dim / LANES;
            for c in 0..chunks {
                let off = c * LANES;
                let vo = vld1q_f32(out_row.as_ptr().add(off));
                let vv = vld1q_f32(v_row.as_ptr().add(off));
                let vw = vdupq_n_f32(w);
                let res = vfmaq_f32(vo, vv, vw);
                vst1q_f32(out_row.as_mut_ptr().add(off), res);
            }
            for d in (chunks * LANES)..head_dim {
                out_row[d] += w * v_row[d];
            }
        }
    }
    output
}

/// Block-sparse matrix multiplication: `C = A × B` with block-level mask.
///
/// Matrices are row-major: `A` is `m × k`, `B` is `k × n`, `C` is `m × n`.
/// `block_mask` has `(m / block_size) × (n / block_size)` entries; a `true`
/// entry means the corresponding `block_size × block_size` output tile is
/// computed, otherwise it stays zero.
///
/// # Panics
/// Panics on dimension mismatches or if `m`/`n` are not multiples of
/// `block_size`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_block_sparse_matmul(
    a: &[f32],
    b: &[f32],
    block_mask: &[bool],
    block_size: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(m % block_size, 0);
    assert_eq!(n % block_size, 0);
    let brows = m / block_size;
    let bcols = n / block_size;
    assert_eq!(block_mask.len(), brows * bcols);

    let mut c = vec![0.0f32; m * n];

    for br in 0..brows {
        for bc in 0..bcols {
            if !block_mask[br * bcols + bc] {
                continue;
            }
            // Compute the (br, bc) output tile.
            for bi in 0..block_size {
                let row = br * block_size + bi;
                let a_row = &a[row * k..(row + 1) * k];
                for bj in 0..block_size {
                    let col = bc * block_size + bj;
                    // Dot product a_row · b_col.
                    let chunks = k / LANES;
                    let mut acc = vdupq_n_f32(0.0);
                    for ch in 0..chunks {
                        let off = ch * LANES;
                        let va = vld1q_f32(a_row.as_ptr().add(off));
                        // b is row-major k×n, column `col` is
                        // strided — gather scalars.
                        let vb = {
                            let b0 = *b.get_unchecked((off) * n + col);
                            let b1 = *b.get_unchecked((off + 1) * n + col);
                            let b2 = *b.get_unchecked((off + 2) * n + col);
                            let b3 = *b.get_unchecked((off + 3) * n + col);
                            let arr = [b0, b1, b2, b3];
                            vld1q_f32(arr.as_ptr())
                        };
                        acc = vfmaq_f32(acc, va, vb);
                    }
                    let mut dot = hsum_f32x4(acc);
                    for t in (chunks * LANES)..k {
                        dot += a_row[t] * b[t * n + col];
                    }
                    c[row * n + col] = dot;
                }
            }
        }
    }
    c
}

/// Local attention with configurable window.
///
/// Each query position `q` attends to key positions in
/// `[max(0, q - local_window) ..= min(seq_len-1, q + local_window)]`.
/// Returns `seq_len × head_dim` (row-major).
///
/// # Panics
/// Panics if slice lengths are inconsistent with `head_dim` / `seq_len`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_local_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    local_window: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert_eq!(query.len(), seq_len * head_dim);
    assert_eq!(key.len(), seq_len * head_dim);
    assert_eq!(value.len(), seq_len * head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for q in 0..seq_len {
        let q_row = &query[q * head_dim..(q + 1) * head_dim];
        let start = q.saturating_sub(local_window);
        let end = (q + local_window).min(seq_len - 1);
        let win_len = end - start + 1;

        let mut scores = vec![0.0f32; win_len];
        for (wi, k_pos) in (start..=end).enumerate() {
            let k_row = &key[k_pos * head_dim..(k_pos + 1) * head_dim];
            scores[wi] = neon_dot(q_row, k_row) * scale;
        }
        softmax_inplace(&mut scores);

        let out_row = &mut output[q * head_dim..(q + 1) * head_dim];
        for (wi, k_pos) in (start..=end).enumerate() {
            let v_row = &value[k_pos * head_dim..(k_pos + 1) * head_dim];
            let w = scores[wi];
            let chunks = head_dim / LANES;
            for c in 0..chunks {
                let off = c * LANES;
                let vo = vld1q_f32(out_row.as_ptr().add(off));
                let vv = vld1q_f32(v_row.as_ptr().add(off));
                let vw = vdupq_n_f32(w);
                let res = vfmaq_f32(vo, vv, vw);
                vst1q_f32(out_row.as_mut_ptr().add(off), res);
            }
            for d in (chunks * LANES)..head_dim {
                out_row[d] += w * v_row[d];
            }
        }
    }
    output
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    /// Full mask should produce the same result as dense attention.
    #[test]
    fn test_sparse_attention_full_mask() {
        let head_dim = 4;
        let seq_len = 3;
        let query = vec![1.0f32; seq_len * head_dim];
        let key = vec![1.0f32; seq_len * head_dim];
        let mask = vec![true; seq_len * seq_len];

        let scores =
            unsafe { neon_sparse_attention_scores(&query, &key, &mask, head_dim, seq_len) };

        assert_eq!(scores.len(), seq_len * seq_len);
        // All keys are identical so softmax should be uniform.
        for q in 0..seq_len {
            let row = &scores[q * seq_len..(q + 1) * seq_len];
            let expected = 1.0 / seq_len as f32;
            for &s in row {
                assert!((s - expected).abs() < 1e-5, "expected ~{expected}, got {s}");
            }
        }
    }

    /// Sliding window must only attend to positions within the window.
    #[test]
    fn test_sliding_window_basic() {
        let head_dim = 4;
        let seq_len = 4;
        let window_size = 2;

        // Distinct key/value per position so we can verify boundaries.
        let mut query = vec![0.0f32; seq_len * head_dim];
        let mut key = vec![0.0f32; seq_len * head_dim];
        let mut value = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            let val = (i + 1) as f32;
            for d in 0..head_dim {
                query[i * head_dim + d] = 1.0;
                key[i * head_dim + d] = val;
                value[i * head_dim + d] = val;
            }
        }

        let out = unsafe {
            neon_sliding_window_attention(&query, &key, &value, window_size, head_dim, seq_len)
        };
        assert_eq!(out.len(), seq_len * head_dim);

        // Position 0 can only see itself (window=[0]).
        // Value at pos 0 is 1.0 everywhere, so output row
        // should be ~1.0.
        for d in 0..head_dim {
            assert!((out[d] - 1.0).abs() < 1e-4, "pos0 dim{d}: {}", out[d]);
        }
    }

    /// Block-sparse with all-true mask should equal dense matmul.
    #[test]
    fn test_block_sparse_identity() {
        let m = 4;
        let n = 4;
        let k = 4;
        let block_size = 2;

        // A = I (identity), B = some matrix → C should equal B.
        let mut a = vec![0.0f32; m * k];
        for i in 0..m.min(k) {
            a[i * k + i] = 1.0;
        }
        let b: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32).collect();
        let block_mask = vec![true; (m / block_size) * (n / block_size)];

        let c = unsafe { neon_block_sparse_matmul(&a, &b, &block_mask, block_size, m, n, k) };

        assert_eq!(c.len(), m * n);
        for i in 0..m * n {
            assert!((c[i] - b[i]).abs() < 1e-4, "index {i}: expected {}, got {}", b[i], c[i]);
        }
    }

    /// Local attention on a small sequence with window=1 (each position
    /// attends to itself and its immediate neighbours).
    #[test]
    fn test_local_attention_small() {
        let head_dim = 4;
        let seq_len = 3;
        let local_window = 1;

        let query = vec![1.0f32; seq_len * head_dim];
        let key = vec![1.0f32; seq_len * head_dim];
        let value = vec![1.0f32; seq_len * head_dim];

        let out =
            unsafe { neon_local_attention(&query, &key, &value, local_window, head_dim, seq_len) };
        assert_eq!(out.len(), seq_len * head_dim);

        // All values identical → output should be ~1.0 everywhere.
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-5, "index {i}: expected ~1.0, got {v}");
        }
    }
}
