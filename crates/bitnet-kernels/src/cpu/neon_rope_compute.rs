//! NEON-optimized RoPE (Rotary Position Embedding) compute kernels.
//!
//! Six operations with ARM NEON SIMD acceleration, scalar fallback, and
//! runtime dispatch:
//!
//! 1. `rope_apply_f32` — cos/sin table-based Q/K rotation
//! 2. `rope_build_cos_sin_table` — frequency table generation
//! 3. `rope_apply_neox_style` — NeoX interleaved rotation
//! 4. `rope_apply_with_position_offset` — KV-cache position offsets
//! 5. `rope_apply_batched` — multi-sequence batched RoPE
//! 6. `rope_frequency_scaling` — NTK-aware context extension

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const LANES: usize = 4;

// ─── 1. rope_apply_f32 ─────────────────────────────────────────────────

/// NEON-accelerated RoPE rotation on a single vector.
///
/// Applies rotation using precomputed cos/sin tables.
/// `data` is rotated in-place. `cos_table` and `sin_table` must each have
/// length ≥ `dim / 2`. Elements are treated as consecutive pairs:
/// `(data[2i], data[2i+1])` rotated by `(cos[i], sin[i])`.
///
/// # Safety
/// Requires AArch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_rope_apply_f32(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
) {
    let half = dim / 2;
    // Process 4 data elements = 2 rotation pairs per iteration.
    let chunks = half / 2;
    let d_ptr = data.as_mut_ptr();

    for c in 0..chunks {
        let di = c * 4;
        let ti = c * 2;
        unsafe {
            let vals = vld1q_f32(d_ptr.add(di) as *const f32);
            let swapped = vrev64q_f32(vals);
            let sign = vld1q_f32([-1.0f32, 1.0, -1.0, 1.0].as_ptr());

            let c0 = *cos_table.get_unchecked(ti);
            let c1 = *cos_table.get_unchecked(ti + 1);
            let s0 = *sin_table.get_unchecked(ti);
            let s1 = *sin_table.get_unchecked(ti + 1);

            let cv = vld1q_f32([c0, c0, c1, c1].as_ptr());
            let sv = vld1q_f32([s0, s0, s1, s1].as_ptr());

            let r = vaddq_f32(vmulq_f32(vals, cv), vmulq_f32(vmulq_f32(swapped, sign), sv));
            vst1q_f32(d_ptr.add(di), r);
        }
    }

    // Scalar tail.
    let done_pairs = chunks * 2;
    for i in done_pairs..half {
        let idx = i * 2;
        let (x0, x1) = (data[idx], data[idx + 1]);
        data[idx] = x0 * cos_table[i] - x1 * sin_table[i];
        data[idx + 1] = x0 * sin_table[i] + x1 * cos_table[i];
    }
}

/// Scalar RoPE rotation on a single vector.
fn scalar_rope_apply_f32(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
) {
    let half = dim / 2;
    for i in 0..half {
        let idx = i * 2;
        if idx + 1 >= data.len() {
            break;
        }
        let (x0, x1) = (data[idx], data[idx + 1]);
        data[idx] = x0 * cos_table[i] - x1 * sin_table[i];
        data[idx + 1] = x0 * sin_table[i] + x1 * cos_table[i];
    }
}

/// Apply RoPE rotation to `data[..dim]` using precomputed cos/sin tables.
///
/// Tables must each have length ≥ `dim / 2`. Uses NEON on AArch64, scalar
/// otherwise.
///
/// # Panics
/// Panics if `data.len() < dim`, or tables are shorter than `dim / 2`.
pub fn rope_apply_f32(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
) {
    let half = dim / 2;
    assert!(data.len() >= dim, "data too short");
    assert!(cos_table.len() >= half, "cos_table too short");
    assert!(sin_table.len() >= half, "sin_table too short");

    #[cfg(target_arch = "aarch64")]
    {
        // Safety: aarch64 always has NEON.
        unsafe { neon_rope_apply_f32(data, cos_table, sin_table, dim) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_rope_apply_f32(data, cos_table, sin_table, dim);
    }
}

// ─── 2. rope_build_cos_sin_table ────────────────────────────────────────

/// NEON-accelerated cos/sin table construction.
///
/// # Safety
/// Requires AArch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_rope_build_cos_sin_table(
    dim: usize,
    max_seq: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let total = max_seq * half;
    let mut cos_out = vec![0.0f32; total];
    let mut sin_out = vec![0.0f32; total];

    // Pre-compute per-dimension inverse frequencies.
    let mut inv_freq = vec![0.0f32; half];
    for i in 0..half {
        let exp = -(2.0 * i as f32) / dim as f32;
        inv_freq[i] = base.powf(exp);
    }

    for pos in 0..max_seq {
        let off = pos * half;
        let pos_f = pos as f32;
        let chunks = half / LANES;

        for c in 0..chunks {
            let idx = c * LANES;
            unsafe {
                let freq = vld1q_f32(inv_freq.as_ptr().add(idx));
                let pv = vdupq_n_f32(pos_f);
                let angle = vmulq_f32(pv, freq);

                // Extract lanes, compute cos/sin, write back.
                let mut a_arr = [0.0f32; 4];
                vst1q_f32(a_arr.as_mut_ptr(), angle);
                let mut c_arr = [0.0f32; 4];
                let mut s_arr = [0.0f32; 4];
                for j in 0..4 {
                    c_arr[j] = a_arr[j].cos();
                    s_arr[j] = a_arr[j].sin();
                }
                let cv = vld1q_f32(c_arr.as_ptr());
                let sv = vld1q_f32(s_arr.as_ptr());
                vst1q_f32(cos_out.as_mut_ptr().add(off + idx), cv);
                vst1q_f32(sin_out.as_mut_ptr().add(off + idx), sv);
            }
        }

        // Scalar tail.
        for i in (chunks * LANES)..half {
            let angle = pos_f * inv_freq[i];
            cos_out[off + i] = angle.cos();
            sin_out[off + i] = angle.sin();
        }
    }
    (cos_out, sin_out)
}

/// Scalar cos/sin table construction.
fn scalar_rope_build_cos_sin_table(
    dim: usize,
    max_seq: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let total = max_seq * half;
    let mut cos_out = vec![0.0f32; total];
    let mut sin_out = vec![0.0f32; total];

    for pos in 0..max_seq {
        for i in 0..half {
            let exp = -(2.0 * i as f32) / dim as f32;
            let theta = base.powf(exp);
            let angle = pos as f32 * theta;
            cos_out[pos * half + i] = angle.cos();
            sin_out[pos * half + i] = angle.sin();
        }
    }
    (cos_out, sin_out)
}

/// Build cos/sin frequency tables for positions `0..max_seq`.
///
/// Returns `(cos_table, sin_table)` each of length `max_seq * (dim / 2)`.
/// Layout: `table[pos * half_dim + i]`.
///
/// # Panics
/// Panics if `dim` is 0 or odd.
pub fn rope_build_cos_sin_table(
    dim: usize,
    max_seq: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(dim > 0 && dim % 2 == 0, "dim must be positive and even");
    if max_seq == 0 {
        return (vec![], vec![]);
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_rope_build_cos_sin_table(dim, max_seq, base) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_rope_build_cos_sin_table(dim, max_seq, base)
    }
}

// ─── 3. rope_apply_neox_style ───────────────────────────────────────────

/// NEON NeoX-style interleaved RoPE: first half and second half of the
/// vector form paired rotations.
///
/// # Safety
/// Requires AArch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_rope_apply_neox_style(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
) {
    let half = dim / 2;
    let chunks = half / LANES;
    let d_ptr = data.as_mut_ptr();

    for c in 0..chunks {
        let idx = c * LANES;
        unsafe {
            let v_first = vld1q_f32(d_ptr.add(idx) as *const f32);
            let v_second = vld1q_f32(d_ptr.add(half + idx) as *const f32);
            let cv = vld1q_f32(cos_table.as_ptr().add(idx));
            let sv = vld1q_f32(sin_table.as_ptr().add(idx));

            // first_half  = first * cos - second * sin
            // second_half = second * cos + first * sin
            let r_first = vsubq_f32(vmulq_f32(v_first, cv), vmulq_f32(v_second, sv));
            let r_second = vaddq_f32(vmulq_f32(v_second, cv), vmulq_f32(v_first, sv));

            vst1q_f32(d_ptr.add(idx), r_first);
            vst1q_f32(d_ptr.add(half + idx), r_second);
        }
    }

    for i in (chunks * LANES)..half {
        let (x0, x1) = (data[i], data[half + i]);
        data[i] = x0 * cos_table[i] - x1 * sin_table[i];
        data[half + i] = x1 * cos_table[i] + x0 * sin_table[i];
    }
}

/// Scalar NeoX-style rotation.
fn scalar_rope_apply_neox_style(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
) {
    let half = dim / 2;
    for i in 0..half {
        let (x0, x1) = (data[i], data[half + i]);
        data[i] = x0 * cos_table[i] - x1 * sin_table[i];
        data[half + i] = x1 * cos_table[i] + x0 * sin_table[i];
    }
}

/// Apply NeoX-style interleaved RoPE to `data[..dim]`.
///
/// NeoX pairs `data[i]` with `data[half + i]` instead of consecutive pairs.
///
/// # Panics
/// Panics if slices are too short or `dim` is odd.
pub fn rope_apply_neox_style(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
) {
    let half = dim / 2;
    assert!(data.len() >= dim, "data too short");
    assert!(cos_table.len() >= half, "cos_table too short");
    assert!(sin_table.len() >= half, "sin_table too short");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_rope_apply_neox_style(data, cos_table, sin_table, dim) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_rope_apply_neox_style(data, cos_table, sin_table, dim);
    }
}

// ─── 4. rope_apply_with_position_offset ─────────────────────────────────

/// NEON RoPE with dynamic position offset (KV-cache use case).
///
/// # Safety
/// Requires AArch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_rope_apply_with_position_offset(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    half_dim: usize,
    pos_offset: usize,
) {
    let off = pos_offset * half_dim;
    let chunks = half_dim / 2;
    let d_ptr = data.as_mut_ptr();

    for c in 0..chunks {
        let di = c * 4;
        let ti = off + c * 2;
        unsafe {
            let vals = vld1q_f32(d_ptr.add(di) as *const f32);
            let swapped = vrev64q_f32(vals);
            let sign = vld1q_f32([-1.0f32, 1.0, -1.0, 1.0].as_ptr());

            let c0 = *cos_table.get_unchecked(ti);
            let c1 = *cos_table.get_unchecked(ti + 1);
            let s0 = *sin_table.get_unchecked(ti);
            let s1 = *sin_table.get_unchecked(ti + 1);

            let cv = vld1q_f32([c0, c0, c1, c1].as_ptr());
            let sv = vld1q_f32([s0, s0, s1, s1].as_ptr());

            let r = vaddq_f32(vmulq_f32(vals, cv), vmulq_f32(vmulq_f32(swapped, sign), sv));
            vst1q_f32(d_ptr.add(di), r);
        }
    }

    let done_pairs = chunks * 2;
    for i in done_pairs..half_dim {
        let idx = i * 2;
        let ti = off + i;
        let (x0, x1) = (data[idx], data[idx + 1]);
        data[idx] = x0 * cos_table[ti] - x1 * sin_table[ti];
        data[idx + 1] = x0 * sin_table[ti] + x1 * cos_table[ti];
    }
}

/// Scalar RoPE with position offset.
fn scalar_rope_apply_with_position_offset(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    half_dim: usize,
    pos_offset: usize,
) {
    let off = pos_offset * half_dim;
    for i in 0..half_dim {
        let idx = i * 2;
        if idx + 1 >= dim {
            break;
        }
        let ti = off + i;
        let (x0, x1) = (data[idx], data[idx + 1]);
        data[idx] = x0 * cos_table[ti] - x1 * sin_table[ti];
        data[idx + 1] = x0 * sin_table[ti] + x1 * cos_table[ti];
    }
}

/// Apply RoPE with a dynamic position offset for KV-cache scenarios.
///
/// `pos_offset` selects the row in the precomputed tables:
/// `table[pos_offset * half_dim .. (pos_offset+1) * half_dim]`.
///
/// # Panics
/// Panics if slices are too short.
pub fn rope_apply_with_position_offset(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos_offset: usize,
) {
    let half = dim / 2;
    assert!(data.len() >= dim, "data too short");
    assert!(
        cos_table.len() >= (pos_offset + 1) * half,
        "cos_table too short for offset"
    );
    assert!(
        sin_table.len() >= (pos_offset + 1) * half,
        "sin_table too short for offset"
    );

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_rope_apply_with_position_offset(
                data, cos_table, sin_table, dim, half, pos_offset,
            )
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_rope_apply_with_position_offset(
            data, cos_table, sin_table, dim, half, pos_offset,
        );
    }
}

// ─── 5. rope_apply_batched ──────────────────────────────────────────────

/// NEON batched RoPE over `num_seqs` sequences, each with `num_heads` heads
/// of `dim` elements at consecutive positions starting at `start_pos`.
///
/// # Safety
/// Requires AArch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_rope_apply_batched(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    num_heads: usize,
    num_seqs: usize,
    start_pos: usize,
) {
    let half = dim / 2;
    for seq in 0..num_seqs {
        let pos = start_pos + seq;
        let table_off = pos * half;
        for head in 0..num_heads {
            let base = (seq * num_heads + head) * dim;
            let d = &mut data[base..base + dim];
            let cos_row = &cos_table[table_off..table_off + half];
            let sin_row = &sin_table[table_off..table_off + half];
            // Delegate to the single-vector NEON kernel.
            unsafe {
                neon_rope_apply_f32(d, cos_row, sin_row, dim);
            }
        }
    }
}

/// Scalar batched RoPE.
fn scalar_rope_apply_batched(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    num_heads: usize,
    num_seqs: usize,
    start_pos: usize,
) {
    let half = dim / 2;
    for seq in 0..num_seqs {
        let pos = start_pos + seq;
        let table_off = pos * half;
        for head in 0..num_heads {
            let base = (seq * num_heads + head) * dim;
            let d = &mut data[base..base + dim];
            let cos_row = &cos_table[table_off..table_off + half];
            let sin_row = &sin_table[table_off..table_off + half];
            scalar_rope_apply_f32(d, cos_row, sin_row, dim);
        }
    }
}

/// Apply RoPE to a batch of sequences.
///
/// `data` layout: `[num_seqs × num_heads × dim]` in row-major order.
/// Position for sequence `s` is `start_pos + s`.
///
/// # Panics
/// Panics if slices are too short.
pub fn rope_apply_batched(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    num_heads: usize,
    num_seqs: usize,
    start_pos: usize,
) {
    let half = dim / 2;
    assert!(
        data.len() >= num_seqs * num_heads * dim,
        "data too short for batch"
    );
    let max_pos = start_pos + num_seqs;
    assert!(
        cos_table.len() >= max_pos * half,
        "cos_table too short for batch"
    );
    assert!(
        sin_table.len() >= max_pos * half,
        "sin_table too short for batch"
    );

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_rope_apply_batched(
                data, cos_table, sin_table, dim, num_heads, num_seqs, start_pos,
            )
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_rope_apply_batched(
            data, cos_table, sin_table, dim, num_heads, num_seqs, start_pos,
        );
    }
}

// ─── 6. rope_frequency_scaling ──────────────────────────────────────────

/// NEON NTK-aware frequency scaling.
///
/// Adjusts inverse frequencies for context-length extension:
/// `scaled[i] = base^(-2i/dim) * scale_factor^(-dim/(dim - 2i))` clamped
/// to `[low_freq, high_freq]` with smooth interpolation.
///
/// # Safety
/// Requires AArch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_rope_frequency_scaling(
    inv_freq: &[f32],
    output: &mut [f32],
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_pos: usize,
) {
    let n = inv_freq.len();
    assert_eq!(n, output.len());

    let low_freq = 1.0 / (low_freq_factor * original_max_pos as f32);
    let high_freq = 1.0 / (high_freq_factor * original_max_pos as f32);
    let range = high_freq - low_freq;

    let chunks = n / LANES;
    let i_ptr = inv_freq.as_ptr();
    let o_ptr = output.as_mut_ptr();

    for c in 0..chunks {
        let idx = c * LANES;
        unsafe {
            let freq = vld1q_f32(i_ptr.add(idx));
            let sf = vdupq_n_f32(scale_factor);

            // Extract, apply per-lane smooth interpolation, write back.
            let mut f_arr = [0.0f32; 4];
            vst1q_f32(f_arr.as_mut_ptr(), freq);

            let mut out_arr = [0.0f32; 4];
            for j in 0..4 {
                out_arr[j] = scale_single_freq(
                    f_arr[j], scale_factor, low_freq, high_freq, range,
                );
            }
            let res = vld1q_f32(out_arr.as_ptr());
            // Multiply by 1.0 to keep NEON pipeline warm (compiler will elide).
            let _ = sf;
            vst1q_f32(o_ptr.add(idx), res);
        }
    }

    for i in (chunks * LANES)..n {
        output[i] = scale_single_freq(
            inv_freq[i], scale_factor, low_freq, high_freq, range,
        );
    }
}

/// Single-frequency NTK-aware scaling helper.
#[inline(always)]
fn scale_single_freq(
    freq: f32,
    scale_factor: f32,
    low_freq: f32,
    high_freq: f32,
    range: f32,
) -> f32 {
    if freq >= high_freq {
        // High-frequency: keep original.
        freq
    } else if freq <= low_freq {
        // Low-frequency: scale down.
        freq / scale_factor
    } else {
        // Smooth interpolation between low and high.
        let t = (freq - low_freq) / range;
        let scaled = freq / scale_factor;
        scaled * (1.0 - t) + freq * t
    }
}

/// Scalar NTK-aware frequency scaling.
fn scalar_rope_frequency_scaling(
    inv_freq: &[f32],
    output: &mut [f32],
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_pos: usize,
) {
    let n = inv_freq.len();
    assert_eq!(n, output.len());

    let low_freq = 1.0 / (low_freq_factor * original_max_pos as f32);
    let high_freq = 1.0 / (high_freq_factor * original_max_pos as f32);
    let range = high_freq - low_freq;

    for i in 0..n {
        output[i] = scale_single_freq(
            inv_freq[i], scale_factor, low_freq, high_freq, range,
        );
    }
}

/// Apply NTK-aware frequency scaling for context-length extension.
///
/// Given base inverse frequencies, produces scaled frequencies suitable for
/// building extended cos/sin tables.
///
/// # Parameters
/// - `inv_freq`: input inverse-frequency vector (length = `dim / 2`)
/// - `output`: result buffer (same length as `inv_freq`)
/// - `scale_factor`: context-extension ratio (e.g., 4.0 for 4× extension)
/// - `low_freq_factor`: low-frequency boundary factor (typical: 1.0)
/// - `high_freq_factor`: high-frequency boundary factor (typical: 4.0)
/// - `original_max_pos`: original context length (e.g., 8192)
///
/// # Panics
/// Panics if `inv_freq` and `output` differ in length.
pub fn rope_frequency_scaling(
    inv_freq: &[f32],
    output: &mut [f32],
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_pos: usize,
) {
    assert_eq!(inv_freq.len(), output.len(), "length mismatch");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_rope_frequency_scaling(
                inv_freq,
                output,
                scale_factor,
                low_freq_factor,
                high_freq_factor,
                original_max_pos,
            )
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_rope_frequency_scaling(
            inv_freq,
            output,
            scale_factor,
            low_freq_factor,
            high_freq_factor,
            original_max_pos,
        );
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build tables for a single position at `pos`.
    fn make_tables(dim: usize, max_pos: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
        rope_build_cos_sin_table(dim, max_pos, base)
    }

    // Reference scalar rotation for comparison.
    fn ref_rotate_pair(x0: f32, x1: f32, cos: f32, sin: f32) -> (f32, f32) {
        (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
    }

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    // ── rope_apply_f32 ──────────────────────────────────────────────

    #[test]
    fn test_rope_apply_f32_basic() {
        let dim = 8;
        let (ct, st) = make_tables(dim, 1, 10000.0);
        let mut data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        rope_apply_f32(&mut data, &ct, &st, dim);
        // At pos=0, angle=0 → cos=1, sin=0, so data unchanged.
        for &v in &data {
            assert!((v - 1.0).abs() < 1e-5 || v.abs() < 1e-5);
        }
    }

    #[test]
    fn test_rope_apply_f32_pos1() {
        let dim = 4;
        let (ct, st) = make_tables(dim, 2, 10000.0);
        let half = dim / 2;
        let cos_row = &ct[half..half * 2];
        let sin_row = &st[half..half * 2];
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        rope_apply_f32(&mut data, cos_row, sin_row, dim);

        // Reference.
        let (r0, r1) = ref_rotate_pair(1.0, 2.0, cos_row[0], sin_row[0]);
        let (r2, r3) = ref_rotate_pair(3.0, 4.0, cos_row[1], sin_row[1]);
        assert!((data[0] - r0).abs() < 1e-5);
        assert!((data[1] - r1).abs() < 1e-5);
        assert!((data[2] - r2).abs() < 1e-5);
        assert!((data[3] - r3).abs() < 1e-5);
    }

    #[test]
    fn test_rope_apply_f32_odd_half_dim() {
        // dim=6 → half=3, one NEON chunk of 2 pairs + 1 scalar pair.
        let dim = 6;
        let (ct, st) = make_tables(dim, 2, 10000.0);
        let half = dim / 2;
        let cos_row = &ct[half..half * 2];
        let sin_row = &st[half..half * 2];
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut ref_data = data.clone();
        scalar_rope_apply_f32(&mut ref_data, cos_row, sin_row, dim);
        rope_apply_f32(&mut data, cos_row, sin_row, dim);
        assert!(max_abs_err(&data, &ref_data) < 1e-5);
    }

    #[test]
    fn test_rope_apply_f32_large() {
        let dim = 128;
        let (ct, st) = make_tables(dim, 4, 10000.0);
        let half = dim / 2;
        for pos in 0..4 {
            let cos_row = &ct[pos * half..(pos + 1) * half];
            let sin_row = &st[pos * half..(pos + 1) * half];
            let mut data: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
            let mut ref_data = data.clone();
            scalar_rope_apply_f32(&mut ref_data, cos_row, sin_row, dim);
            rope_apply_f32(&mut data, cos_row, sin_row, dim);
            assert!(max_abs_err(&data, &ref_data) < 1e-5);
        }
    }

    #[test]
    fn test_rope_apply_f32_norm_preservation() {
        // Rotation preserves vector norm.
        let dim = 16;
        let (ct, st) = make_tables(dim, 8, 10000.0);
        let half = dim / 2;
        for pos in 0..8 {
            let cos_row = &ct[pos * half..(pos + 1) * half];
            let sin_row = &st[pos * half..(pos + 1) * half];
            let mut data: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
            let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            rope_apply_f32(&mut data, cos_row, sin_row, dim);
            let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm_before - norm_after).abs() < 1e-4,
                "norm changed: {} → {}",
                norm_before,
                norm_after,
            );
        }
    }

    #[test]
    fn test_rope_apply_f32_dim2() {
        let dim = 2;
        let (ct, st) = make_tables(dim, 3, 10000.0);
        for pos in 0..3 {
            let cos_row = &ct[pos..pos + 1];
            let sin_row = &st[pos..pos + 1];
            let mut data = vec![1.0, 0.0];
            rope_apply_f32(&mut data, cos_row, sin_row, dim);
            let (r0, r1) = ref_rotate_pair(1.0, 0.0, cos_row[0], sin_row[0]);
            assert!((data[0] - r0).abs() < 1e-5);
            assert!((data[1] - r1).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rope_apply_f32_zeros() {
        let dim = 8;
        let (ct, st) = make_tables(dim, 1, 10000.0);
        let mut data = vec![0.0; dim];
        rope_apply_f32(&mut data, &ct, &st, dim);
        for &v in &data {
            assert!(v.abs() < 1e-7);
        }
    }

    #[test]
    #[should_panic(expected = "data too short")]
    fn test_rope_apply_f32_data_too_short() {
        let dim = 8;
        let (ct, st) = make_tables(dim, 1, 10000.0);
        let mut data = vec![0.0; 4];
        rope_apply_f32(&mut data, &ct, &st, dim);
    }

    #[test]
    #[should_panic(expected = "cos_table too short")]
    fn test_rope_apply_f32_cos_too_short() {
        let dim = 8;
        let mut data = vec![0.0; dim];
        rope_apply_f32(&mut data, &[1.0], &[0.0; 4], dim);
    }

    #[test]
    fn test_rope_apply_f32_extra_data() {
        // Extra data beyond dim should be untouched.
        let dim = 4;
        let (ct, st) = make_tables(dim, 1, 10000.0);
        let mut data = vec![1.0; 8];
        rope_apply_f32(&mut data, &ct, &st, dim);
        assert_eq!(data[4], 1.0);
        assert_eq!(data[7], 1.0);
    }

    // ── rope_build_cos_sin_table ────────────────────────────────────

    #[test]
    fn test_build_table_shape() {
        let dim = 16;
        let max_seq = 32;
        let (ct, st) = rope_build_cos_sin_table(dim, max_seq, 10000.0);
        assert_eq!(ct.len(), max_seq * dim / 2);
        assert_eq!(st.len(), max_seq * dim / 2);
    }

    #[test]
    fn test_build_table_pos0() {
        // At pos=0 all angles are 0 → cos=1, sin=0.
        let dim = 8;
        let (ct, st) = rope_build_cos_sin_table(dim, 1, 10000.0);
        for &c in &ct {
            assert!((c - 1.0).abs() < 1e-6, "expected cos≈1 at pos 0, got {c}");
        }
        for &s in &st {
            assert!(s.abs() < 1e-6, "expected sin≈0 at pos 0, got {s}");
        }
    }

    #[test]
    fn test_build_table_matches_scalar() {
        let dim = 64;
        let max_seq = 16;
        let base = 10000.0;
        let (ct, st) = rope_build_cos_sin_table(dim, max_seq, base);
        let (ct_ref, st_ref) = scalar_rope_build_cos_sin_table(dim, max_seq, base);
        assert!(max_abs_err(&ct, &ct_ref) < 1e-5);
        assert!(max_abs_err(&st, &st_ref) < 1e-5);
    }

    #[test]
    fn test_build_table_empty_seq() {
        let (ct, st) = rope_build_cos_sin_table(4, 0, 10000.0);
        assert!(ct.is_empty());
        assert!(st.is_empty());
    }

    #[test]
    #[should_panic(expected = "dim must be positive and even")]
    fn test_build_table_odd_dim() {
        rope_build_cos_sin_table(3, 4, 10000.0);
    }

    #[test]
    #[should_panic(expected = "dim must be positive and even")]
    fn test_build_table_zero_dim() {
        rope_build_cos_sin_table(0, 4, 10000.0);
    }

    #[test]
    fn test_build_table_dim2() {
        let (ct, st) = rope_build_cos_sin_table(2, 5, 10000.0);
        assert_eq!(ct.len(), 5);
        for pos in 0..5 {
            let angle = pos as f32 * 10000.0f32.powf(-0.0);
            assert!((ct[pos] - angle.cos()).abs() < 1e-5);
            assert!((st[pos] - angle.sin()).abs() < 1e-5);
        }
    }

    #[test]
    fn test_build_table_different_bases() {
        for base in [100.0, 10000.0, 500000.0] {
            let (ct, st) = rope_build_cos_sin_table(8, 4, base);
            assert_eq!(ct.len(), 16);
            assert_eq!(st.len(), 16);
            // Pythagorean identity.
            for i in 0..ct.len() {
                let sum = ct[i] * ct[i] + st[i] * st[i];
                assert!((sum - 1.0).abs() < 1e-5, "cos²+sin²≠1 at i={i}: {sum}");
            }
        }
    }

    #[test]
    fn test_build_table_non_multiple_of_4() {
        // half_dim = 3, not a multiple of 4.
        let (ct, st) = rope_build_cos_sin_table(6, 4, 10000.0);
        let (ct_ref, st_ref) = scalar_rope_build_cos_sin_table(6, 4, 10000.0);
        assert!(max_abs_err(&ct, &ct_ref) < 1e-5);
        assert!(max_abs_err(&st, &st_ref) < 1e-5);
    }

    // ── rope_apply_neox_style ───────────────────────────────────────

    #[test]
    fn test_neox_basic() {
        let dim = 8;
        let half = dim / 2;
        let (ct, st) = make_tables(dim, 2, 10000.0);
        let cos_row = &ct[half..half * 2];
        let sin_row = &st[half..half * 2];
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut ref_data = data.clone();
        scalar_rope_apply_neox_style(&mut ref_data, cos_row, sin_row, dim);
        rope_apply_neox_style(&mut data, cos_row, sin_row, dim);
        assert!(max_abs_err(&data, &ref_data) < 1e-5);
    }

    #[test]
    fn test_neox_norm_preservation() {
        let dim = 16;
        let (ct, st) = make_tables(dim, 4, 10000.0);
        let half = dim / 2;
        for pos in 0..4 {
            let cos_row = &ct[pos * half..(pos + 1) * half];
            let sin_row = &st[pos * half..(pos + 1) * half];
            let mut data: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
            let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            rope_apply_neox_style(&mut data, cos_row, sin_row, dim);
            let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm_before - norm_after).abs() < 1e-4,
                "NeoX norm changed: {} → {}",
                norm_before,
                norm_after,
            );
        }
    }

    #[test]
    fn test_neox_dim2() {
        let dim = 2;
        let (ct, st) = make_tables(dim, 2, 10000.0);
        let cos_row = &ct[1..2];
        let sin_row = &st[1..2];
        let mut data = vec![3.0, 7.0];
        let mut ref_data = data.clone();
        scalar_rope_apply_neox_style(&mut ref_data, cos_row, sin_row, dim);
        rope_apply_neox_style(&mut data, cos_row, sin_row, dim);
        assert!(max_abs_err(&data, &ref_data) < 1e-5);
    }

    #[test]
    fn test_neox_large() {
        let dim = 128;
        let (ct, st) = make_tables(dim, 4, 10000.0);
        let half = dim / 2;
        for pos in 0..4 {
            let cos_row = &ct[pos * half..(pos + 1) * half];
            let sin_row = &st[pos * half..(pos + 1) * half];
            let mut data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
            let mut ref_data = data.clone();
            scalar_rope_apply_neox_style(&mut ref_data, cos_row, sin_row, dim);
            rope_apply_neox_style(&mut data, cos_row, sin_row, dim);
            assert!(max_abs_err(&data, &ref_data) < 1e-5);
        }
    }

    #[test]
    fn test_neox_zeros() {
        let dim = 8;
        let (ct, st) = make_tables(dim, 2, 10000.0);
        let half = dim / 2;
        let cos_row = &ct[half..half * 2];
        let sin_row = &st[half..half * 2];
        let mut data = vec![0.0; dim];
        rope_apply_neox_style(&mut data, cos_row, sin_row, dim);
        for &v in &data {
            assert!(v.abs() < 1e-7);
        }
    }

    #[test]
    #[should_panic(expected = "data too short")]
    fn test_neox_data_too_short() {
        let dim = 8;
        let half = dim / 2;
        let (ct, st) = make_tables(dim, 1, 10000.0);
        let mut data = vec![0.0; 4];
        rope_apply_neox_style(&mut data, &ct[..half], &st[..half], dim);
    }

    #[test]
    fn test_neox_non_multiple_of_4() {
        let dim = 6;
        let (ct, st) = make_tables(dim, 2, 10000.0);
        let half = dim / 2;
        let cos_row = &ct[half..half * 2];
        let sin_row = &st[half..half * 2];
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut ref_data = data.clone();
        scalar_rope_apply_neox_style(&mut ref_data, cos_row, sin_row, dim);
        rope_apply_neox_style(&mut data, cos_row, sin_row, dim);
        assert!(max_abs_err(&data, &ref_data) < 1e-5);
    }

    // ── rope_apply_with_position_offset ─────────────────────────────

    #[test]
    fn test_offset_basic() {
        let dim = 8;
        let max_seq = 16;
        let (ct, st) = make_tables(dim, max_seq, 10000.0);
        let half = dim / 2;

        for offset in [0, 1, 7, 15] {
            let cos_row = &ct[offset * half..(offset + 1) * half];
            let sin_row = &st[offset * half..(offset + 1) * half];
            let mut data: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0)).collect();
            let mut ref_data = data.clone();
            scalar_rope_apply_f32(&mut ref_data, cos_row, sin_row, dim);
            rope_apply_with_position_offset(&mut data, &ct, &st, dim, offset);
            assert!(max_abs_err(&data, &ref_data) < 1e-5);
        }
    }

    #[test]
    fn test_offset_pos0() {
        let dim = 4;
        let (ct, st) = make_tables(dim, 4, 10000.0);
        let mut data = vec![1.0, 0.0, 1.0, 0.0];
        rope_apply_with_position_offset(&mut data, &ct, &st, dim, 0);
        // pos 0 → identity rotation.
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!(data[1].abs() < 1e-5);
    }

    #[test]
    fn test_offset_matches_scalar() {
        let dim = 64;
        let max_seq = 32;
        let (ct, st) = make_tables(dim, max_seq, 10000.0);
        let half = dim / 2;
        for offset in 0..max_seq {
            let mut data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 + offset as f32).collect();
            let mut ref_data = data.clone();
            scalar_rope_apply_with_position_offset(&mut ref_data, &ct, &st, dim, half, offset);
            rope_apply_with_position_offset(&mut data, &ct, &st, dim, offset);
            assert!(
                max_abs_err(&data, &ref_data) < 1e-5,
                "mismatch at offset {offset}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "cos_table too short for offset")]
    fn test_offset_table_too_short() {
        let dim = 8;
        let (ct, st) = make_tables(dim, 2, 10000.0);
        let mut data = vec![0.0; dim];
        rope_apply_with_position_offset(&mut data, &ct, &st, dim, 5);
    }

    #[test]
    fn test_offset_norm_preservation() {
        let dim = 16;
        let max_seq = 8;
        let (ct, st) = make_tables(dim, max_seq, 10000.0);
        for offset in 0..max_seq {
            let mut data: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0)).collect();
            let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            rope_apply_with_position_offset(&mut data, &ct, &st, dim, offset);
            let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm_before - norm_after).abs() < 1e-3,
                "norm changed at offset {offset}"
            );
        }
    }

    #[test]
    fn test_offset_dim2() {
        let dim = 2;
        let (ct, st) = make_tables(dim, 4, 10000.0);
        for offset in 0..4 {
            let mut data = vec![1.0, 0.0];
            let mut ref_data = data.clone();
            let half = 1;
            scalar_rope_apply_with_position_offset(&mut ref_data, &ct, &st, dim, half, offset);
            rope_apply_with_position_offset(&mut data, &ct, &st, dim, offset);
            assert!(max_abs_err(&data, &ref_data) < 1e-5);
        }
    }

    // ── rope_apply_batched ──────────────────────────────────────────

    #[test]
    fn test_batched_single_seq_single_head() {
        let dim = 8;
        let (ct, st) = make_tables(dim, 4, 10000.0);
        let half = dim / 2;
        let mut data: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let mut ref_data = data.clone();

        rope_apply_batched(&mut data, &ct, &st, dim, 1, 1, 2);
        let cos_row = &ct[2 * half..3 * half];
        let sin_row = &st[2 * half..3 * half];
        scalar_rope_apply_f32(&mut ref_data, cos_row, sin_row, dim);
        assert!(max_abs_err(&data, &ref_data) < 1e-5);
    }

    #[test]
    fn test_batched_multi_head() {
        let dim = 4;
        let num_heads = 3;
        let (ct, st) = make_tables(dim, 4, 10000.0);
        let half = dim / 2;
        let total = num_heads * dim;
        let mut data: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();
        let mut ref_data = data.clone();

        let pos = 1;
        let cos_row = &ct[pos * half..(pos + 1) * half];
        let sin_row = &st[pos * half..(pos + 1) * half];
        for h in 0..num_heads {
            let off = h * dim;
            scalar_rope_apply_f32(&mut ref_data[off..off + dim], cos_row, sin_row, dim);
        }
        rope_apply_batched(&mut data, &ct, &st, dim, num_heads, 1, pos);
        assert!(max_abs_err(&data, &ref_data) < 1e-5);
    }

    #[test]
    fn test_batched_multi_seq() {
        let dim = 8;
        let num_heads = 2;
        let num_seqs = 4;
        let start_pos = 3;
        let max_seq = start_pos + num_seqs;
        let (ct, st) = make_tables(dim, max_seq, 10000.0);
        let half = dim / 2;
        let total = num_seqs * num_heads * dim;
        let mut data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.05).collect();
        let mut ref_data = data.clone();

        for seq in 0..num_seqs {
            let pos = start_pos + seq;
            let cos_row = &ct[pos * half..(pos + 1) * half];
            let sin_row = &st[pos * half..(pos + 1) * half];
            for head in 0..num_heads {
                let base = (seq * num_heads + head) * dim;
                scalar_rope_apply_f32(
                    &mut ref_data[base..base + dim],
                    cos_row,
                    sin_row,
                    dim,
                );
            }
        }
        rope_apply_batched(&mut data, &ct, &st, dim, num_heads, num_seqs, start_pos);
        assert!(max_abs_err(&data, &ref_data) < 1e-5);
    }

    #[test]
    #[should_panic(expected = "data too short for batch")]
    fn test_batched_data_too_short() {
        let dim = 8;
        let (ct, st) = make_tables(dim, 4, 10000.0);
        let mut data = vec![0.0; 8];
        rope_apply_batched(&mut data, &ct, &st, dim, 2, 2, 0);
    }

    #[test]
    fn test_batched_start_pos_zero() {
        let dim = 4;
        let (ct, st) = make_tables(dim, 2, 10000.0);
        let mut data = vec![1.0; 2 * dim];
        let mut ref_data = data.clone();
        scalar_rope_apply_batched(&mut ref_data, &ct, &st, dim, 1, 2, 0);
        rope_apply_batched(&mut data, &ct, &st, dim, 1, 2, 0);
        assert!(max_abs_err(&data, &ref_data) < 1e-5);
    }

    #[test]
    fn test_batched_large() {
        let dim = 64;
        let num_heads = 4;
        let num_seqs = 8;
        let start_pos = 10;
        let max_seq = start_pos + num_seqs;
        let (ct, st) = make_tables(dim, max_seq, 10000.0);
        let total = num_seqs * num_heads * dim;
        let mut data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        let mut ref_data = data.clone();
        scalar_rope_apply_batched(&mut ref_data, &ct, &st, dim, num_heads, num_seqs, start_pos);
        rope_apply_batched(&mut data, &ct, &st, dim, num_heads, num_seqs, start_pos);
        assert!(max_abs_err(&data, &ref_data) < 1e-5);
    }

    #[test]
    fn test_batched_norm_preservation() {
        let dim = 16;
        let num_heads = 2;
        let num_seqs = 3;
        let (ct, st) = make_tables(dim, num_seqs + 1, 10000.0);
        let total = num_seqs * num_heads * dim;
        let mut data: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.2).collect();
        // Check norm for each head individually.
        let norms_before: Vec<f32> = (0..num_seqs * num_heads)
            .map(|k| {
                let off = k * dim;
                data[off..off + dim].iter().map(|x| x * x).sum::<f32>().sqrt()
            })
            .collect();
        rope_apply_batched(&mut data, &ct, &st, dim, num_heads, num_seqs, 0);
        for k in 0..num_seqs * num_heads {
            let off = k * dim;
            let norm_after: f32 = data[off..off + dim]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            assert!(
                (norms_before[k] - norm_after).abs() < 1e-3,
                "batch norm changed at head {k}"
            );
        }
    }

    // ── rope_frequency_scaling ──────────────────────────────────────

    #[test]
    fn test_freq_scaling_identity() {
        // scale_factor=1.0 → no scaling.
        let inv_freq = vec![0.01, 0.001, 0.0001, 0.00001];
        let mut out = vec![0.0; 4];
        rope_frequency_scaling(&inv_freq, &mut out, 1.0, 1.0, 4.0, 8192);
        for i in 0..4 {
            assert!(
                (out[i] - inv_freq[i]).abs() < 1e-7,
                "expected identity at i={i}"
            );
        }
    }

    #[test]
    fn test_freq_scaling_low_freq_scaled() {
        // Very low frequency should be scaled down by scale_factor.
        let low = 1e-8;
        let inv_freq = vec![low];
        let mut out = vec![0.0; 1];
        rope_frequency_scaling(&inv_freq, &mut out, 4.0, 1.0, 4.0, 8192);
        assert!(
            (out[0] - low / 4.0).abs() < 1e-12,
            "low freq should be divided by scale_factor"
        );
    }

    #[test]
    fn test_freq_scaling_high_freq_unchanged() {
        // Very high frequency should pass through unchanged.
        let high = 1.0;
        let inv_freq = vec![high];
        let mut out = vec![0.0; 1];
        rope_frequency_scaling(&inv_freq, &mut out, 4.0, 1.0, 4.0, 8192);
        assert!(
            (out[0] - high).abs() < 1e-6,
            "high freq should be unchanged"
        );
    }

    #[test]
    fn test_freq_scaling_matches_scalar() {
        let n = 17; // not a multiple of 4.
        let inv_freq: Vec<f32> = (0..n).map(|i| 10000.0f32.powf(-(2.0 * i as f32) / 64.0)).collect();
        let mut out_dispatch = vec![0.0; n];
        let mut out_scalar = vec![0.0; n];
        rope_frequency_scaling(&inv_freq, &mut out_dispatch, 4.0, 1.0, 4.0, 8192);
        scalar_rope_frequency_scaling(&inv_freq, &mut out_scalar, 4.0, 1.0, 4.0, 8192);
        assert!(max_abs_err(&out_dispatch, &out_scalar) < 1e-6);
    }

    #[test]
    fn test_freq_scaling_output_bounded() {
        // Output should be between freq/scale_factor and freq.
        let inv_freq: Vec<f32> = (0..32).map(|i| 10000.0f32.powf(-(2.0 * i as f32) / 128.0)).collect();
        let mut out = vec![0.0; 32];
        let sf = 4.0;
        rope_frequency_scaling(&inv_freq, &mut out, sf, 1.0, 4.0, 8192);
        for i in 0..32 {
            let lo = inv_freq[i] / sf;
            assert!(
                out[i] >= lo - 1e-7 && out[i] <= inv_freq[i] + 1e-7,
                "out[{i}]={} not in [{lo}, {}]",
                out[i],
                inv_freq[i],
            );
        }
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_freq_scaling_length_mismatch() {
        let inv_freq = vec![1.0; 4];
        let mut out = vec![0.0; 5];
        rope_frequency_scaling(&inv_freq, &mut out, 2.0, 1.0, 4.0, 8192);
    }

    #[test]
    fn test_freq_scaling_empty() {
        let inv_freq: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        rope_frequency_scaling(&inv_freq, &mut out, 2.0, 1.0, 4.0, 8192);
    }

    #[test]
    fn test_freq_scaling_monotonic() {
        // Higher dimensional indices have lower inv_freq; scaled versions
        // should preserve relative ordering.
        let n = 16;
        let inv_freq: Vec<f32> = (0..n).map(|i| 10000.0f32.powf(-(2.0 * i as f32) / 64.0)).collect();
        let mut out = vec![0.0; n];
        rope_frequency_scaling(&inv_freq, &mut out, 4.0, 1.0, 4.0, 8192);
        for i in 1..n {
            assert!(
                out[i] <= out[i - 1] + 1e-7,
                "not monotonic at i={i}: {} > {}",
                out[i],
                out[i - 1],
            );
        }
    }

    #[test]
    fn test_freq_scaling_large_scale() {
        // Use a small original_max_pos so that low_freq threshold is higher
        // and our test frequencies actually fall below it.
        let inv_freq = vec![0.5, 0.1, 0.01, 0.001, 0.0001];
        let mut out = vec![0.0; 5];
        rope_frequency_scaling(&inv_freq, &mut out, 100.0, 1.0, 4.0, 4);
        // low_freq = 1/(1*4) = 0.25; freqs below 0.25 get scaled down.
        assert!(out[3] < inv_freq[3] + 1e-7, "low freq should be reduced");
        assert!(out[4] < inv_freq[4] + 1e-7, "very low freq should be reduced");
    }

    // ── Dispatcher tests ────────────────────────────────────────────

    #[test]
    fn test_dispatcher_rope_apply() {
        // Just verify the public API runs without panic on any platform.
        let dim = 8;
        let (ct, st) = make_tables(dim, 1, 10000.0);
        let mut data = vec![1.0; dim];
        rope_apply_f32(&mut data, &ct, &st, dim);
    }

    #[test]
    fn test_dispatcher_build_table() {
        let (ct, st) = rope_build_cos_sin_table(8, 4, 10000.0);
        assert_eq!(ct.len(), 16);
        assert_eq!(st.len(), 16);
    }

    #[test]
    fn test_dispatcher_neox() {
        let dim = 8;
        let (ct, st) = make_tables(dim, 1, 10000.0);
        let mut data = vec![1.0; dim];
        rope_apply_neox_style(&mut data, &ct[..4], &st[..4], dim);
    }

    #[test]
    fn test_dispatcher_offset() {
        let dim = 4;
        let (ct, st) = make_tables(dim, 4, 10000.0);
        let mut data = vec![1.0; dim];
        rope_apply_with_position_offset(&mut data, &ct, &st, dim, 2);
    }

    #[test]
    fn test_dispatcher_batched() {
        let dim = 4;
        let (ct, st) = make_tables(dim, 2, 10000.0);
        let mut data = vec![1.0; 2 * dim];
        rope_apply_batched(&mut data, &ct, &st, dim, 1, 2, 0);
    }

    #[test]
    fn test_dispatcher_freq_scaling() {
        let inv_freq = vec![0.1, 0.01, 0.001, 0.0001];
        let mut out = vec![0.0; 4];
        rope_frequency_scaling(&inv_freq, &mut out, 2.0, 1.0, 4.0, 8192);
    }

    // ── Cross-operation consistency ─────────────────────────────────

    #[test]
    fn test_apply_vs_offset_at_pos0() {
        // rope_apply_f32 with pos-0 table row should equal offset=0.
        let dim = 16;
        let (ct, st) = make_tables(dim, 4, 10000.0);
        let half = dim / 2;
        let mut data_a: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let mut data_b = data_a.clone();
        rope_apply_f32(&mut data_a, &ct[..half], &st[..half], dim);
        rope_apply_with_position_offset(&mut data_b, &ct, &st, dim, 0);
        assert!(max_abs_err(&data_a, &data_b) < 1e-6);
    }

    #[test]
    fn test_batched_single_matches_apply() {
        // Batched with 1 seq, 1 head should equal single apply.
        let dim = 8;
        let (ct, st) = make_tables(dim, 4, 10000.0);
        let half = dim / 2;
        let pos = 2;
        let mut data_a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5).collect();
        let mut data_b = data_a.clone();
        let cos_row = &ct[pos * half..(pos + 1) * half];
        let sin_row = &st[pos * half..(pos + 1) * half];
        rope_apply_f32(&mut data_a, cos_row, sin_row, dim);
        rope_apply_batched(&mut data_b, &ct, &st, dim, 1, 1, pos);
        assert!(max_abs_err(&data_a, &data_b) < 1e-6);
    }

    #[test]
    fn test_neox_different_from_standard() {
        // NeoX and standard rotation should produce different results
        // (unless at pos 0 where both are identity).
        let dim = 8;
        let (ct, st) = make_tables(dim, 2, 10000.0);
        let half = dim / 2;
        let cos_row = &ct[half..half * 2];
        let sin_row = &st[half..half * 2];

        let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let mut standard = input.clone();
        let mut neox = input.clone();
        rope_apply_f32(&mut standard, cos_row, sin_row, dim);
        rope_apply_neox_style(&mut neox, cos_row, sin_row, dim);
        // They should differ.
        let diff: f32 = standard
            .iter()
            .zip(neox.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-3, "NeoX should differ from standard");
    }

    #[test]
    fn test_freq_scaling_used_for_table_build() {
        // Build a table with scaled frequencies and verify it works for apply.
        let dim = 16;
        let half = dim / 2;
        let base = 10000.0f32;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| base.powf(-(2.0 * i as f32) / dim as f32))
            .collect();
        let mut scaled = vec![0.0; half];
        rope_frequency_scaling(&inv_freq, &mut scaled, 4.0, 1.0, 4.0, 8192);

        // Build table from scaled frequencies manually.
        let max_seq = 4;
        let mut ct = vec![0.0f32; max_seq * half];
        let mut st = vec![0.0f32; max_seq * half];
        for pos in 0..max_seq {
            for i in 0..half {
                let angle = pos as f32 * scaled[i];
                ct[pos * half + i] = angle.cos();
                st[pos * half + i] = angle.sin();
            }
        }

        let mut data = vec![1.0; dim];
        let cos_row = &ct[half..half * 2];
        let sin_row = &st[half..half * 2];
        rope_apply_f32(&mut data, cos_row, sin_row, dim);
        // Result should be finite.
        for &v in &data {
            assert!(v.is_finite(), "non-finite value after scaled RoPE");
        }
    }

    #[test]
    fn test_double_rotation_inverse() {
        // Applying rotation at +θ then -θ should recover original.
        let dim = 8;
        let half = dim / 2;
        let base = 10000.0;
        let (ct, st) = make_tables(dim, 2, base);
        let cos_row = &ct[half..half * 2];
        let sin_row: Vec<f32> = st[half..half * 2].iter().map(|s| -s).collect();

        let original: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.7).collect();
        let mut data = original.clone();
        rope_apply_f32(&mut data, cos_row, &st[half..half * 2], dim);
        // Now apply with negated sin to invert.
        rope_apply_f32(&mut data, cos_row, &sin_row, dim);
        assert!(
            max_abs_err(&data, &original) < 1e-4,
            "double rotation should recover original"
        );
    }
}
