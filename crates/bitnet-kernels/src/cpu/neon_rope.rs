//! ARM NEON-optimized RoPE (Rotary Position Embedding) kernels for Apple Silicon.
//!
//! Provides vectorized rotary position encoding using NEON SIMD intrinsics
//! on AArch64. Processes 4 × f32 (2 rotation pairs) at a time with scalar
//! fallback for remainder elements.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Build separate cos and sin frequency tables for NEON RoPE.
///
/// Layout: `table[pos * half_dim + i]` holds the value for position `pos`
/// and dimension-pair index `i`, where `half_dim = dim / 2`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn build_cos_sin_tables_neon(
    dim: usize,
    max_seq: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = dim / 2;
    let mut cos_table = Vec::with_capacity(max_seq * half_dim);
    let mut sin_table = Vec::with_capacity(max_seq * half_dim);

    for pos in 0..max_seq {
        for i in 0..half_dim {
            let exponent = -(2.0 * i as f32) / dim as f32;
            let theta = base.powf(exponent);
            let angle = pos as f32 * theta;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
    }

    (cos_table, sin_table)
}

/// Apply RoPE rotation to a single head vector **in-place** using NEON.
///
/// `data` must have length ≥ `dim`, and `cos_table`/`sin_table` must
/// cover the given `pos` (i.e. have at least `(pos + 1) * dim / 2` entries).
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if tables are too short for the given `pos` and `dim`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_rope_neon(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    let half_dim = dim / 2;
    let table_offset = pos * half_dim;

    // Sign mask: [-1, +1, -1, +1] for the rotation formula.
    let sign_mask = unsafe { vld1q_f32([-1.0f32, 1.0, -1.0, 1.0].as_ptr()) };

    // Process 4 floats (2 rotation pairs) per NEON iteration.
    let chunks = half_dim / 2;
    for c in 0..chunks {
        let data_idx = c * 4;
        let table_idx = table_offset + c * 2;

        unsafe {
            let vals = vld1q_f32(data.as_ptr().add(data_idx));

            // Swap pairs within 64-bit lanes: [x0, x1, x2, x3] → [x1, x0, x3, x2]
            let swapped = vrev64q_f32(vals);

            // Expand cos/sin to match paired layout: [c0, c0, c1, c1]
            let c0 = *cos_table.get_unchecked(table_idx);
            let c1 = *cos_table.get_unchecked(table_idx + 1);
            let s0 = *sin_table.get_unchecked(table_idx);
            let s1 = *sin_table.get_unchecked(table_idx + 1);

            let cos_expanded = vld1q_f32([c0, c0, c1, c1].as_ptr());
            let sin_expanded = vld1q_f32([s0, s0, s1, s1].as_ptr());

            // result = vals * cos + swapped * sign * sin
            //   even lanes: x0*cos - x1*sin
            //   odd  lanes: x1*cos + x0*sin  (= x0*sin + x1*cos)
            let term1 = vmulq_f32(vals, cos_expanded);
            let term2 = vmulq_f32(vmulq_f32(swapped, sign_mask), sin_expanded);
            let rotated = vaddq_f32(term1, term2);

            vst1q_f32(data.as_mut_ptr().add(data_idx), rotated);
        }
    }

    // Scalar tail for the remaining pair (if half_dim is odd).
    let processed_pairs = chunks * 2;
    for i in processed_pairs..half_dim {
        let idx = i * 2;
        let cos_val = unsafe { *cos_table.get_unchecked(table_offset + i) };
        let sin_val = unsafe { *sin_table.get_unchecked(table_offset + i) };

        let x0 = data[idx];
        let x1 = data[idx + 1];

        data[idx] = x0 * cos_val - x1 * sin_val;
        data[idx + 1] = x0 * sin_val + x1 * cos_val;
    }
}

/// Apply RoPE rotation to multiple heads at the same position using NEON.
///
/// `data` layout: `[num_heads × dim]` — each contiguous `dim` block gets
/// the rotation for the given `pos`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_rope_batch_neon(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    num_heads: usize,
    pos: usize,
) {
    for h in 0..num_heads {
        let offset = h * dim;
        unsafe {
            apply_rope_neon(&mut data[offset..offset + dim], cos_table, sin_table, dim, pos);
        }
    }
}
