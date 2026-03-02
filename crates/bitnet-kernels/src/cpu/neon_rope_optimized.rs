#![cfg(target_arch = "aarch64")]
//! Advanced ARM NEON-optimized RoPE (Rotary Position Embedding) kernels for Apple Silicon.
//!
//! This module extends the base NEON RoPE implementation with:
//!
//! - **Cached table lookups** — precomputed cos/sin with vectorized loads
//! - **Batch processing** — rotates 4 sequences simultaneously using 4 NEON registers
//! - **Interleaved (GPT-NeoX) vs paired (LLaMA) modes** — two common RoPE layouts
//! - **Inverse RoPE** — undo a rotation for position extraction or debugging
//! - **NEON-accelerated table building** — vectorized sin/cos precomputation
//!
//! All functions are gated on `target_arch = "aarch64"` and require NEON (always
//! available on AArch64). Unsafe blocks are kept minimal and documented.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Scalar reference helpers (used by tests and as fallback) ────────

/// Scalar RoPE rotation for a single head (paired / LLaMA style).
///
/// Pairs are `(data[2i], data[2i+1])` rotated by `(cos[i], sin[i])`.
fn rope_scalar_paired(data: &mut [f32], cos: &[f32], sin: &[f32], half_dim: usize) {
    for i in 0..half_dim {
        let x0 = data[2 * i];
        let x1 = data[2 * i + 1];
        data[2 * i] = x0 * cos[i] - x1 * sin[i];
        data[2 * i + 1] = x0 * sin[i] + x1 * cos[i];
    }
}

/// Scalar RoPE rotation for a single head (interleaved / GPT-NeoX style).
///
/// The first half `data[0..half_dim]` pairs with the second half
/// `data[half_dim..dim]`, i.e. pair `i` is `(data[i], data[half_dim + i])`.
fn rope_scalar_interleaved(data: &mut [f32], cos: &[f32], sin: &[f32], half_dim: usize) {
    for i in 0..half_dim {
        let x0 = data[i];
        let x1 = data[half_dim + i];
        data[i] = x0 * cos[i] - x1 * sin[i];
        data[half_dim + i] = x0 * sin[i] + x1 * cos[i];
    }
}

/// Scalar inverse RoPE (paired layout) — rotates by `(-sin, cos)`.
fn rope_inverse_scalar(data: &mut [f32], cos: &[f32], sin: &[f32], half_dim: usize) {
    for i in 0..half_dim {
        let x0 = data[2 * i];
        let x1 = data[2 * i + 1];
        // Inverse rotation: negate the angle → cos unchanged, sin negated.
        data[2 * i] = x0 * cos[i] + x1 * sin[i];
        data[2 * i + 1] = -x0 * sin[i] + x1 * cos[i];
    }
}

// ── Table construction ──────────────────────────────────────────────

/// Build cos/sin frequency tables using NEON-accelerated arithmetic.
///
/// Returns `(cos_table, sin_table)` each of length `max_seq_len * half_dim`.
/// Layout: `table[pos * half_dim + i]` for position `pos`, dimension-pair `i`.
///
/// The inner loop uses NEON `vmulq_f32` for scaling angles by position, with
/// a scalar `sin`/`cos` call per element (transcendentals have no NEON
/// single-instruction equivalent, but the multiply + accumulate is vectorized).
///
/// # Safety
///
/// Requires AArch64 NEON (always present on AArch64 targets).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn build_rope_tables_simd(
    dim: usize,
    max_seq_len: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = dim / 2;
    let total = max_seq_len * half_dim;
    let mut cos_table = vec![0.0f32; total];
    let mut sin_table = vec![0.0f32; total];

    // Precompute inverse-frequency vector: theta_i = base^(-2i / dim).
    let mut inv_freq = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        let exponent = -(2.0 * i as f32) / dim as f32;
        inv_freq[i] = base.powf(exponent);
    }

    // For each position, compute angle = pos * theta_i, then cos/sin.
    // We vectorize the multiply `pos * theta` 4 elements at a time.
    for pos in 0..max_seq_len {
        let row = pos * half_dim;
        let pos_f = pos as f32;

        // NEON: broadcast position scalar into all 4 lanes.
        // Safety: NEON is guaranteed on AArch64.
        let pos_vec = unsafe { vdupq_n_f32(pos_f) };

        // Process 4 dimension-pairs per iteration.
        let chunks = half_dim / 4;
        for c in 0..chunks {
            let base_idx = c * 4;

            // Safety: `base_idx + 4 <= half_dim` guaranteed by `chunks` bound.
            unsafe {
                let theta = vld1q_f32(inv_freq.as_ptr().add(base_idx));
                let angle = vmulq_f32(pos_vec, theta);

                // Extract lanes and compute transcendentals (no NEON sin/cos).
                let a0 = vgetq_lane_f32::<0>(angle);
                let a1 = vgetq_lane_f32::<1>(angle);
                let a2 = vgetq_lane_f32::<2>(angle);
                let a3 = vgetq_lane_f32::<3>(angle);

                let cos_vals = [a0.cos(), a1.cos(), a2.cos(), a3.cos()];
                let sin_vals = [a0.sin(), a1.sin(), a2.sin(), a3.sin()];

                // Store 4 values at once.
                vst1q_f32(cos_table.as_mut_ptr().add(row + base_idx), vld1q_f32(cos_vals.as_ptr()));
                vst1q_f32(sin_table.as_mut_ptr().add(row + base_idx), vld1q_f32(sin_vals.as_ptr()));
            }
        }

        // Scalar tail for remaining elements.
        let processed = chunks * 4;
        for i in processed..half_dim {
            let angle = pos_f * inv_freq[i];
            cos_table[row + i] = angle.cos();
            sin_table[row + i] = angle.sin();
        }
    }

    (cos_table, sin_table)
}

// ── Cached table RoPE ───────────────────────────────────────────────

/// Apply RoPE using precomputed cos/sin tables (paired / LLaMA layout).
///
/// This avoids recomputing transcendentals at apply time. The tables must
/// have been built with [`build_rope_tables_simd`] or an equivalent function
/// and laid out as `table[pos * half_dim + i]`.
///
/// Processes 4 floats (2 rotation pairs) per NEON iteration with a scalar
/// tail for odd `half_dim`.
///
/// # Safety
///
/// Requires AArch64 NEON.
///
/// # Panics
///
/// Panics if `data.len() < dim`, or if the tables are too short for `pos`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn rope_apply_cached_neon(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    assert!(data.len() >= dim, "data too short: {} < {dim}", data.len());
    let half_dim = dim / 2;
    let table_offset = pos * half_dim;
    assert!(cos_table.len() >= table_offset + half_dim, "cos_table too short for pos={pos}");

    // Sign mask for the rotation formula: even lanes negate, odd lanes keep.
    // Safety: NEON is available on AArch64.
    let sign_mask = unsafe { vld1q_f32([-1.0f32, 1.0, -1.0, 1.0].as_ptr()) };

    // Process 4 floats (2 rotation pairs) per NEON iteration.
    let chunks = half_dim / 2;
    for c in 0..chunks {
        let data_idx = c * 4;
        let table_idx = table_offset + c * 2;

        // Safety: indices are within bounds by the `chunks` bound and assertions.
        unsafe {
            let vals = vld1q_f32(data.as_ptr().add(data_idx));

            // Swap pairs within 64-bit lanes: [x0, x1, x2, x3] → [x1, x0, x3, x2].
            let swapped = vrev64q_f32(vals);

            // Load and expand cos/sin for 2 pairs → 4 lanes each.
            let c0 = *cos_table.get_unchecked(table_idx);
            let c1 = *cos_table.get_unchecked(table_idx + 1);
            let s0 = *sin_table.get_unchecked(table_idx);
            let s1 = *sin_table.get_unchecked(table_idx + 1);

            let cos_vec = vld1q_f32([c0, c0, c1, c1].as_ptr());
            let sin_vec = vld1q_f32([s0, s0, s1, s1].as_ptr());

            // result = vals * cos + swapped * sign_mask * sin
            let term1 = vmulq_f32(vals, cos_vec);
            let term2 = vmulq_f32(vmulq_f32(swapped, sign_mask), sin_vec);
            let rotated = vaddq_f32(term1, term2);

            vst1q_f32(data.as_mut_ptr().add(data_idx), rotated);
        }
    }

    // Scalar tail for remaining pair (if half_dim is odd).
    let processed_pairs = chunks * 2;
    for i in processed_pairs..half_dim {
        let idx = i * 2;
        let cos_val = cos_table[table_offset + i];
        let sin_val = sin_table[table_offset + i];
        let x0 = data[idx];
        let x1 = data[idx + 1];
        data[idx] = x0 * cos_val - x1 * sin_val;
        data[idx + 1] = x0 * sin_val + x1 * cos_val;
    }
}

// ── Batch RoPE (4 sequences at once) ────────────────────────────────

/// Process RoPE for up to 4 sequences simultaneously.
///
/// Each sequence has layout `[num_heads × dim]`. The `batch_data` slice
/// contains `batch_size` consecutive sequence blocks. Positions are given
/// per sequence via `positions[0..batch_size]`.
///
/// When `batch_size < 4`, only the present sequences are processed (no
/// out-of-bounds access). Internally, each head within each sequence is
/// rotated using the cached-table NEON path.
///
/// # Safety
///
/// Requires AArch64 NEON.
///
/// # Panics
///
/// Panics if `batch_size > 4`, or if slices are too short.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn rope_batch_neon(
    batch_data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    num_heads: usize,
    positions: &[usize],
    batch_size: usize,
) {
    assert!(batch_size <= 4, "batch_size must be <= 4, got {batch_size}");
    assert!(
        positions.len() >= batch_size,
        "positions slice too short: {} < {batch_size}",
        positions.len()
    );

    let seq_stride = num_heads * dim;
    assert!(
        batch_data.len() >= batch_size * seq_stride,
        "batch_data too short for {batch_size} sequences"
    );

    // Process each sequence in the batch, applying cached RoPE per head.
    for b in 0..batch_size {
        let seq_offset = b * seq_stride;
        let pos = positions[b];

        for h in 0..num_heads {
            let head_offset = seq_offset + h * dim;
            // Safety: bounds checked by assertions above; NEON available.
            unsafe {
                rope_apply_cached_neon(
                    &mut batch_data[head_offset..head_offset + dim],
                    cos_table,
                    sin_table,
                    dim,
                    pos,
                );
            }
        }
    }
}

// ── Interleaved (GPT-NeoX) RoPE ────────────────────────────────────

/// Apply interleaved (GPT-NeoX style) RoPE using NEON.
///
/// Unlike paired/LLaMA layout where pair `i` is `(data[2i], data[2i+1])`,
/// the interleaved layout pairs `data[i]` with `data[half_dim + i]`:
///
///   `y[i]          = x[i]          * cos[i] - x[half_dim + i] * sin[i]`
///   `y[half_dim+i] = x[i]          * sin[i] + x[half_dim + i] * cos[i]`
///
/// Processes 4 pairs per NEON iteration.
///
/// # Safety
///
/// Requires AArch64 NEON.
///
/// # Panics
///
/// Panics if `data.len() < dim`, or if tables are too short.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn rope_interleaved_neon(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    assert!(data.len() >= dim, "data too short: {} < {dim}", data.len());
    let half_dim = dim / 2;
    let table_offset = pos * half_dim;
    assert!(cos_table.len() >= table_offset + half_dim, "cos_table too short for pos={pos}");

    // Process 4 pairs per iteration.
    let chunks = half_dim / 4;
    for c in 0..chunks {
        let idx = c * 4;
        let tbl = table_offset + idx;

        // Safety: indices within bounds by `chunks` calculation and assertions.
        unsafe {
            // Load first-half and second-half elements.
            let x_first = vld1q_f32(data.as_ptr().add(idx));
            let x_second = vld1q_f32(data.as_ptr().add(half_dim + idx));

            let cos_vec = vld1q_f32(cos_table.as_ptr().add(tbl));
            let sin_vec = vld1q_f32(sin_table.as_ptr().add(tbl));

            // y_first  = x_first * cos - x_second * sin
            let y_first = vsubq_f32(vmulq_f32(x_first, cos_vec), vmulq_f32(x_second, sin_vec));
            // y_second = x_first * sin + x_second * cos
            let y_second = vaddq_f32(vmulq_f32(x_first, sin_vec), vmulq_f32(x_second, cos_vec));

            vst1q_f32(data.as_mut_ptr().add(idx), y_first);
            vst1q_f32(data.as_mut_ptr().add(half_dim + idx), y_second);
        }
    }

    // Scalar tail.
    let processed = chunks * 4;
    for i in processed..half_dim {
        let cos_val = cos_table[table_offset + i];
        let sin_val = sin_table[table_offset + i];
        let x0 = data[i];
        let x1 = data[half_dim + i];
        data[i] = x0 * cos_val - x1 * sin_val;
        data[half_dim + i] = x0 * sin_val + x1 * cos_val;
    }
}

// ── Inverse RoPE ────────────────────────────────────────────────────

/// Apply the inverse RoPE rotation (paired / LLaMA layout) using NEON.
///
/// Undoes a forward RoPE at the same position by negating the sine
/// component, equivalent to rotating by `-angle`:
///
///   `y[2i]   =  x[2i] * cos + x[2i+1] * sin`
///   `y[2i+1] = -x[2i] * sin + x[2i+1] * cos`
///
/// Useful for extracting position information or verifying round-trip
/// correctness.
///
/// # Safety
///
/// Requires AArch64 NEON.
///
/// # Panics
///
/// Panics if `data.len() < dim`, or if tables are too short.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn rope_inverse_neon(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    assert!(data.len() >= dim, "data too short: {} < {dim}", data.len());
    let half_dim = dim / 2;
    let table_offset = pos * half_dim;
    assert!(cos_table.len() >= table_offset + half_dim, "cos_table too short for pos={pos}");

    // Inverse sign mask: [+1, -1, +1, -1] — negated relative to forward.
    // Safety: NEON available on AArch64.
    let sign_mask = unsafe { vld1q_f32([1.0f32, -1.0, 1.0, -1.0].as_ptr()) };

    let chunks = half_dim / 2;
    for c in 0..chunks {
        let data_idx = c * 4;
        let table_idx = table_offset + c * 2;

        // Safety: within bounds by `chunks` bound.
        unsafe {
            let vals = vld1q_f32(data.as_ptr().add(data_idx));
            let swapped = vrev64q_f32(vals);

            let c0 = *cos_table.get_unchecked(table_idx);
            let c1 = *cos_table.get_unchecked(table_idx + 1);
            let s0 = *sin_table.get_unchecked(table_idx);
            let s1 = *sin_table.get_unchecked(table_idx + 1);

            let cos_vec = vld1q_f32([c0, c0, c1, c1].as_ptr());
            let sin_vec = vld1q_f32([s0, s0, s1, s1].as_ptr());

            // Inverse: vals * cos + swapped * (+sin for even, -sin for odd)
            let term1 = vmulq_f32(vals, cos_vec);
            let term2 = vmulq_f32(vmulq_f32(swapped, sign_mask), sin_vec);
            let rotated = vaddq_f32(term1, term2);

            vst1q_f32(data.as_mut_ptr().add(data_idx), rotated);
        }
    }

    // Scalar tail.
    let processed_pairs = chunks * 2;
    for i in processed_pairs..half_dim {
        let idx = i * 2;
        let cos_val = cos_table[table_offset + i];
        let sin_val = sin_table[table_offset + i];
        let x0 = data[idx];
        let x1 = data[idx + 1];
        data[idx] = x0 * cos_val + x1 * sin_val;
        data[idx + 1] = -x0 * sin_val + x1 * cos_val;
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    /// Helper: build tables via the SIMD path for a given dim / seq_len.
    fn build_tables(dim: usize, max_seq: usize) -> (Vec<f32>, Vec<f32>) {
        // Safety: tests only run on aarch64 where NEON is always present.
        unsafe { build_rope_tables_simd(dim, max_seq, 10_000.0) }
    }

    /// Helper: build tables using pure scalar math for reference comparison.
    fn build_tables_scalar(dim: usize, max_seq: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
        let half_dim = dim / 2;
        let mut cos_table = vec![0.0f32; max_seq * half_dim];
        let mut sin_table = vec![0.0f32; max_seq * half_dim];
        for pos in 0..max_seq {
            for i in 0..half_dim {
                let exponent = -(2.0 * i as f32) / dim as f32;
                let theta = base.powf(exponent);
                let angle = pos as f32 * theta;
                cos_table[pos * half_dim + i] = angle.cos();
                sin_table[pos * half_dim + i] = angle.sin();
            }
        }
        (cos_table, sin_table)
    }

    // ── 1. build_rope_tables_simd ───────────────────────────────────

    #[test]
    fn test_build_tables_matches_scalar() {
        let dim = 16;
        let max_seq = 32;
        let base = 10_000.0;
        let (cos_neon, sin_neon) = unsafe { build_rope_tables_simd(dim, max_seq, base) };
        let (cos_scalar, sin_scalar) = build_tables_scalar(dim, max_seq, base);

        assert_eq!(cos_neon.len(), cos_scalar.len());
        for (i, (n, s)) in cos_neon.iter().zip(cos_scalar.iter()).enumerate() {
            assert!((n - s).abs() < 1e-5, "cos mismatch at {i}: neon={n}, scalar={s}");
        }
        for (i, (n, s)) in sin_neon.iter().zip(sin_scalar.iter()).enumerate() {
            assert!((n - s).abs() < 1e-5, "sin mismatch at {i}: neon={n}, scalar={s}");
        }
    }

    #[test]
    fn test_build_tables_position_zero_identity() {
        let dim = 8;
        let (cos_t, sin_t) = build_tables(dim, 4);
        let half_dim = dim / 2;
        // At position 0, angle = 0 → cos = 1, sin = 0.
        for i in 0..half_dim {
            assert!((cos_t[i] - 1.0).abs() < 1e-6, "cos[{i}] at pos 0 should be 1.0");
            assert!(sin_t[i].abs() < 1e-6, "sin[{i}] at pos 0 should be 0.0");
        }
    }

    // ── 2. rope_apply_cached_neon ───────────────────────────────────

    #[test]
    fn test_cached_neon_matches_scalar_paired() {
        let dim = 16;
        let max_seq = 8;
        let (cos_t, sin_t) = build_tables(dim, max_seq);
        let half_dim = dim / 2;

        for pos in 0..max_seq {
            let original: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.3).collect();

            // NEON path
            let mut neon_data = original.clone();
            unsafe { rope_apply_cached_neon(&mut neon_data, &cos_t, &sin_t, dim, pos) };

            // Scalar reference
            let mut scalar_data = original.clone();
            let tbl_off = pos * half_dim;
            rope_scalar_paired(
                &mut scalar_data,
                &cos_t[tbl_off..tbl_off + half_dim],
                &sin_t[tbl_off..tbl_off + half_dim],
                half_dim,
            );

            for (i, (n, s)) in neon_data.iter().zip(scalar_data.iter()).enumerate() {
                assert!((n - s).abs() < 1e-5, "cached pos={pos} dim {i}: neon={n}, scalar={s}");
            }
        }
    }

    #[test]
    fn test_cached_neon_preserves_norm() {
        let dim = 32;
        let (cos_t, sin_t) = build_tables(dim, 64);

        for pos in [0, 1, 7, 31, 63] {
            let mut data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 + 0.5).collect();
            let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

            unsafe { rope_apply_cached_neon(&mut data, &cos_t, &sin_t, dim, pos) };

            let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm_before - norm_after).abs() < 1e-4,
                "norm not preserved at pos={pos}: {norm_before} vs {norm_after}"
            );
        }
    }

    // ── 3. rope_batch_neon ──────────────────────────────────────────

    #[test]
    fn test_batch_matches_individual() {
        let dim = 8;
        let num_heads = 2;
        let max_seq = 16;
        let batch_size = 4;
        let (cos_t, sin_t) = build_tables(dim, max_seq);

        let positions = [1usize, 3, 7, 12];
        let seq_stride = num_heads * dim;

        let original: Vec<f32> =
            (0..batch_size * seq_stride).map(|i| (i as f32) * 0.05 - 1.0).collect();

        // Batch path
        let mut batch_data = original.clone();
        unsafe {
            rope_batch_neon(
                &mut batch_data,
                &cos_t,
                &sin_t,
                dim,
                num_heads,
                &positions,
                batch_size,
            );
        }

        // Individual path
        let mut indiv_data = original.clone();
        for b in 0..batch_size {
            for h in 0..num_heads {
                let off = b * seq_stride + h * dim;
                unsafe {
                    rope_apply_cached_neon(
                        &mut indiv_data[off..off + dim],
                        &cos_t,
                        &sin_t,
                        dim,
                        positions[b],
                    );
                }
            }
        }

        for (i, (b, s)) in batch_data.iter().zip(indiv_data.iter()).enumerate() {
            assert!((b - s).abs() < 1e-5, "batch mismatch at {i}: batch={b}, individual={s}");
        }
    }

    #[test]
    fn test_batch_partial_size() {
        // batch_size < 4 should work without panicking.
        let dim = 8;
        let num_heads = 1;
        let (cos_t, sin_t) = build_tables(dim, 4);

        for bs in 1..=3 {
            let positions: Vec<usize> = (0..bs).collect();
            let mut data = vec![1.0f32; bs * num_heads * dim];
            unsafe {
                rope_batch_neon(&mut data, &cos_t, &sin_t, dim, num_heads, &positions, bs);
            }
            assert!(data.iter().all(|x| x.is_finite()));
        }
    }

    // ── 4. rope_interleaved_neon ────────────────────────────────────

    #[test]
    fn test_interleaved_matches_scalar() {
        let dim = 16;
        let max_seq = 8;
        let (cos_t, sin_t) = build_tables(dim, max_seq);
        let half_dim = dim / 2;

        for pos in 0..max_seq {
            let original: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) * 0.2).collect();

            // NEON interleaved
            let mut neon_data = original.clone();
            unsafe { rope_interleaved_neon(&mut neon_data, &cos_t, &sin_t, dim, pos) };

            // Scalar reference
            let mut scalar_data = original.clone();
            let tbl_off = pos * half_dim;
            rope_scalar_interleaved(
                &mut scalar_data,
                &cos_t[tbl_off..tbl_off + half_dim],
                &sin_t[tbl_off..tbl_off + half_dim],
                half_dim,
            );

            for (i, (n, s)) in neon_data.iter().zip(scalar_data.iter()).enumerate() {
                assert!(
                    (n - s).abs() < 1e-5,
                    "interleaved pos={pos} dim {i}: neon={n}, scalar={s}"
                );
            }
        }
    }

    #[test]
    fn test_interleaved_preserves_norm() {
        let dim = 32;
        let (cos_t, sin_t) = build_tables(dim, 16);

        for pos in [0, 1, 5, 15] {
            let mut data: Vec<f32> = (0..dim).map(|i| ((i * 7 + 3) as f32) * 0.01).collect();
            let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

            unsafe { rope_interleaved_neon(&mut data, &cos_t, &sin_t, dim, pos) };

            let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm_before - norm_after).abs() < 1e-3,
                "interleaved norm pos={pos}: {norm_before} vs {norm_after}"
            );
        }
    }

    // ── 5. rope_inverse_neon (round-trip) ───────────────────────────

    #[test]
    fn test_inverse_round_trip() {
        let dim = 16;
        let max_seq = 8;
        let (cos_t, sin_t) = build_tables(dim, max_seq);

        for pos in 0..max_seq {
            let original: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.7 - 3.0).collect();

            // Forward then inverse should recover original.
            let mut data = original.clone();
            unsafe {
                rope_apply_cached_neon(&mut data, &cos_t, &sin_t, dim, pos);
                rope_inverse_neon(&mut data, &cos_t, &sin_t, dim, pos);
            }

            for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
                assert!(
                    (o - d).abs() < 1e-4,
                    "round-trip pos={pos} dim {i}: original={o}, recovered={d}"
                );
            }
        }
    }

    #[test]
    fn test_inverse_matches_scalar() {
        let dim = 8;
        let max_seq = 4;
        let (cos_t, sin_t) = build_tables(dim, max_seq);
        let half_dim = dim / 2;

        for pos in 0..max_seq {
            let original: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.5).collect();

            // NEON inverse
            let mut neon_data = original.clone();
            unsafe { rope_inverse_neon(&mut neon_data, &cos_t, &sin_t, dim, pos) };

            // Scalar inverse reference
            let mut scalar_data = original.clone();
            let tbl_off = pos * half_dim;
            rope_inverse_scalar(
                &mut scalar_data,
                &cos_t[tbl_off..tbl_off + half_dim],
                &sin_t[tbl_off..tbl_off + half_dim],
                half_dim,
            );

            for (i, (n, s)) in neon_data.iter().zip(scalar_data.iter()).enumerate() {
                assert!((n - s).abs() < 1e-5, "inverse pos={pos} dim {i}: neon={n}, scalar={s}");
            }
        }
    }

    // ── 6. Odd half_dim (scalar tail coverage) ──────────────────────

    #[test]
    fn test_odd_half_dim_cached_neon() {
        // dim=6 → half_dim=3 (odd), forces scalar tail in all functions.
        let dim = 6;
        let max_seq = 4;
        let (cos_t, sin_t) = build_tables(dim, max_seq);
        let half_dim = dim / 2;

        for pos in 0..max_seq {
            let original: Vec<f32> = (0..dim).map(|i| (i as f32) * 1.1).collect();

            let mut neon_data = original.clone();
            unsafe { rope_apply_cached_neon(&mut neon_data, &cos_t, &sin_t, dim, pos) };

            let mut scalar_data = original.clone();
            let tbl_off = pos * half_dim;
            rope_scalar_paired(
                &mut scalar_data,
                &cos_t[tbl_off..tbl_off + half_dim],
                &sin_t[tbl_off..tbl_off + half_dim],
                half_dim,
            );

            for (i, (n, s)) in neon_data.iter().zip(scalar_data.iter()).enumerate() {
                assert!(
                    (n - s).abs() < 1e-5,
                    "odd half_dim pos={pos} dim {i}: neon={n}, scalar={s}"
                );
            }
        }
    }

    #[test]
    fn test_odd_half_dim_interleaved() {
        let dim = 6;
        let max_seq = 4;
        let (cos_t, sin_t) = build_tables(dim, max_seq);
        let half_dim = dim / 2;

        for pos in 0..max_seq {
            let original: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.9 + 0.1).collect();

            let mut neon_data = original.clone();
            unsafe { rope_interleaved_neon(&mut neon_data, &cos_t, &sin_t, dim, pos) };

            let mut scalar_data = original.clone();
            let tbl_off = pos * half_dim;
            rope_scalar_interleaved(
                &mut scalar_data,
                &cos_t[tbl_off..tbl_off + half_dim],
                &sin_t[tbl_off..tbl_off + half_dim],
                half_dim,
            );

            for (i, (n, s)) in neon_data.iter().zip(scalar_data.iter()).enumerate() {
                assert!(
                    (n - s).abs() < 1e-5,
                    "odd interleaved pos={pos} dim {i}: neon={n}, scalar={s}"
                );
            }
        }
    }

    #[test]
    fn test_odd_half_dim_inverse_round_trip() {
        let dim = 6;
        let max_seq = 4;
        let (cos_t, sin_t) = build_tables(dim, max_seq);

        for pos in 0..max_seq {
            let original: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.4 - 1.0).collect();
            let mut data = original.clone();
            unsafe {
                rope_apply_cached_neon(&mut data, &cos_t, &sin_t, dim, pos);
                rope_inverse_neon(&mut data, &cos_t, &sin_t, dim, pos);
            }
            for (i, (o, d)) in original.iter().zip(data.iter()).enumerate() {
                assert!((o - d).abs() < 1e-4, "odd round-trip pos={pos} dim {i}: {o} vs {d}");
            }
        }
    }
}
