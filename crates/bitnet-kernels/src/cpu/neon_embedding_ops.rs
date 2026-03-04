#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! ARM NEON-optimized embedding operations for Apple Silicon.
//!
//! Provides vectorized embedding lookup, sinusoidal position encoding,
//! element-wise embedding addition, and per-token layer normalization
//! using NEON SIMD intrinsics on AArch64.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Batch embedding lookup with NEON-accelerated memory copy.
///
/// For each index in `indices`, copies `embed_dim` floats from the
/// corresponding row in `table`. Returns a flat vector of
/// `indices.len() * embed_dim` elements.
///
/// # Panics
///
/// Panics if any `index * embed_dim + embed_dim` exceeds `table.len()`.
#[cfg(target_arch = "aarch64")]
pub fn neon_embedding_lookup(table: &[f32], indices: &[u32], embed_dim: usize) -> Vec<f32> {
    let num_tokens = indices.len();
    let mut output = vec![0.0f32; num_tokens * embed_dim];

    for (tok, &idx) in indices.iter().enumerate() {
        let src_start = idx as usize * embed_dim;
        assert!(
            src_start + embed_dim <= table.len(),
            "embedding index {idx} out of bounds (table has {} rows)",
            table.len() / embed_dim
        );
        let dst_start = tok * embed_dim;

        // SAFETY: NEON is always available on AArch64.
        unsafe {
            neon_copy_f32(
                &table[src_start..src_start + embed_dim],
                &mut output[dst_start..dst_start + embed_dim],
            );
        }
    }

    output
}

/// NEON-accelerated f32 slice copy (4 elements at a time).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_copy_f32(src: &[f32], dst: &mut [f32]) {
    let n = src.len();
    let chunks = n / 4;
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(src_ptr.add(offset));
            vst1q_f32(dst_ptr.add(offset), v);
        }
    }

    for i in (chunks * 4)..n {
        dst[i] = src[i];
    }
}

/// Sinusoidal position encoding with NEON-accelerated computation.
///
/// Generates a `seq_len × embed_dim` matrix where:
/// - `PE[pos][2i]   = sin(pos / base^(2i / embed_dim))`
/// - `PE[pos][2i+1] = cos(pos / base^(2i / embed_dim))`
#[cfg(target_arch = "aarch64")]
pub fn neon_position_encoding(seq_len: usize, embed_dim: usize, base: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * embed_dim];
    let half_dim = embed_dim / 2;

    // Precompute inverse frequencies: 1 / base^(2i / embed_dim).
    let mut inv_freq = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        inv_freq[i] = 1.0 / base.powf(2.0 * i as f32 / embed_dim as f32);
    }

    for pos in 0..seq_len {
        let row = &mut output[pos * embed_dim..(pos + 1) * embed_dim];
        let pos_f = pos as f32;

        // SAFETY: NEON is always available on AArch64.
        unsafe {
            neon_sincos_row(pos_f, &inv_freq, row, half_dim);
        }
    }

    output
}

/// Compute sin/cos pairs for one position using NEON angle multiplication.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sincos_row(pos: f32, inv_freq: &[f32], output: &mut [f32], half_dim: usize) {
    let chunks = half_dim / 4;
    let freq_ptr = inv_freq.as_ptr();

    unsafe {
        let pos_vec = vdupq_n_f32(pos);

        for i in 0..chunks {
            let offset = i * 4;
            let freq = vld1q_f32(freq_ptr.add(offset));
            let angle = vmulq_f32(pos_vec, freq);

            // Extract angles for scalar sin/cos (no native NEON sin/cos).
            let a: [f32; 4] = std::mem::transmute(angle);
            for j in 0..4 {
                let dim_idx = offset + j;
                output[dim_idx * 2] = a[j].sin();
                output[dim_idx * 2 + 1] = a[j].cos();
            }
        }
    }

    // Scalar tail.
    for i in (chunks * 4)..half_dim {
        let angle = pos * inv_freq[i];
        output[i * 2] = angle.sin();
        output[i * 2 + 1] = angle.cos();
    }
}

/// Element-wise embedding addition with NEON.
///
/// Returns `a[i] + b[i]` for all elements.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[cfg(target_arch = "aarch64")]
pub fn neon_embedding_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "embedding length mismatch");
    let n = a.len();
    let mut output = vec![0.0f32; n];

    // SAFETY: NEON is always available on AArch64.
    unsafe {
        neon_add_inplace(a, b, &mut output);
    }

    output
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_add_inplace(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            vst1q_f32(o_ptr.add(offset), vaddq_f32(va, vb));
        }
    }

    for i in (chunks * 4)..n {
        out[i] = a[i] + b[i];
    }
}

/// Per-token layer normalization with NEON.
///
/// Normalizes each contiguous `embed_dim`-sized token to zero mean and
/// unit variance: `output[i] = (x[i] - mean) / sqrt(variance + eps)`.
///
/// # Panics
///
/// Panics if `input.len()` is not a multiple of `embed_dim` or `embed_dim` is 0.
#[cfg(target_arch = "aarch64")]
pub fn neon_embedding_norm(input: &[f32], embed_dim: usize, eps: f32) -> Vec<f32> {
    assert!(embed_dim > 0, "embed_dim must be > 0");
    assert_eq!(
        input.len() % embed_dim,
        0,
        "input length {} is not a multiple of embed_dim {embed_dim}",
        input.len()
    );

    let num_tokens = input.len() / embed_dim;
    let mut output = vec![0.0f32; input.len()];

    for t in 0..num_tokens {
        let start = t * embed_dim;
        let token = &input[start..start + embed_dim];
        let out_token = &mut output[start..start + embed_dim];

        // SAFETY: NEON is always available on AArch64.
        unsafe {
            neon_layernorm_token(token, out_token, eps);
        }
    }

    output
}

/// Normalize a single token vector using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_layernorm_token(token: &[f32], output: &mut [f32], eps: f32) {
    let n = token.len();
    let chunks = n / 4;
    let ptr = token.as_ptr();

    // Pass 1: compute mean.
    let mut sum_vec = unsafe { vdupq_n_f32(0.0) };
    unsafe {
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            sum_vec = vaddq_f32(sum_vec, v);
        }
    }
    let mut sum = unsafe { vaddvq_f32(sum_vec) };
    for i in (chunks * 4)..n {
        sum += token[i];
    }
    let mean = sum / n as f32;

    // Pass 2: compute variance.
    let mut var_vec = unsafe { vdupq_n_f32(0.0) };
    let mean_vec = unsafe { vdupq_n_f32(mean) };
    unsafe {
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            let diff = vsubq_f32(v, mean_vec);
            var_vec = vfmaq_f32(var_vec, diff, diff);
        }
    }
    let mut var_sum = unsafe { vaddvq_f32(var_vec) };
    for i in (chunks * 4)..n {
        let d = token[i] - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / (var_sum / n as f32 + eps).sqrt();

    // Pass 3: normalize.
    let inv_std_vec = unsafe { vdupq_n_f32(inv_std) };
    let out_ptr = output.as_mut_ptr();
    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(ptr.add(offset));
            let centered = vsubq_f32(v, mean_vec);
            let normed = vmulq_f32(centered, inv_std_vec);
            vst1q_f32(out_ptr.add(offset), normed);
        }
    }
    for i in (chunks * 4)..n {
        output[i] = (token[i] - mean) * inv_std;
    }
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_lookup_basic() {
        // 4 vocab entries, embed_dim = 3
        let table: Vec<f32> = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
            10.0, 11.0, 12.0, // row 3
        ];
        let indices = vec![2, 0, 3];
        let result = neon_embedding_lookup(&table, &indices, 3);

        assert_eq!(result.len(), 9);
        assert_eq!(&result[0..3], &[7.0, 8.0, 9.0]); // row 2
        assert_eq!(&result[3..6], &[1.0, 2.0, 3.0]); // row 0
        assert_eq!(&result[6..9], &[10.0, 11.0, 12.0]); // row 3
    }

    #[test]
    fn test_position_encoding_orthogonality() {
        let seq_len = 8;
        let embed_dim = 16;
        let pe = neon_position_encoding(seq_len, embed_dim, 10000.0);

        // Each position vector should be approximately orthogonal to others.
        // Verify via dot products: |dot(pe[i], pe[j])| should be small
        // relative to ||pe[i]|| * ||pe[j]|| for i ≠ j.
        for i in 0..seq_len {
            let row_i = &pe[i * embed_dim..(i + 1) * embed_dim];
            let norm_i: f32 = row_i.iter().map(|x| x * x).sum::<f32>().sqrt();

            for j in (i + 1)..seq_len {
                let row_j = &pe[j * embed_dim..(j + 1) * embed_dim];
                let norm_j: f32 = row_j.iter().map(|x| x * x).sum::<f32>().sqrt();

                let dot: f32 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
                let cosine_sim = dot / (norm_i * norm_j + 1e-10);

                assert!(
                    cosine_sim.abs() < 0.9,
                    "positions {i} and {j} too similar: cosine_sim = {cosine_sim}"
                );
            }
        }
    }

    #[test]
    fn test_embedding_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let result = neon_embedding_add(&a, &b);

        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0]);
    }

    #[test]
    fn test_embedding_norm() {
        let embed_dim = 8;
        // Two tokens with known values.
        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // token 0
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, // token 1
        ];
        let result = neon_embedding_norm(&input, embed_dim, 1e-5);

        // Verify each token is normalized: mean ≈ 0, variance ≈ 1.
        for t in 0..2 {
            let token = &result[t * embed_dim..(t + 1) * embed_dim];
            let mean: f32 = token.iter().sum::<f32>() / embed_dim as f32;
            let var: f32 =
                token.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / embed_dim as f32;

            assert!(mean.abs() < 1e-5, "token {t}: mean = {mean}, expected ≈ 0");
            assert!((var - 1.0).abs() < 1e-4, "token {t}: variance = {var}, expected ≈ 1");
        }
    }
}
