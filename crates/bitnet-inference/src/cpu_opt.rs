//! CPU-specific optimised kernels for inference.
//!
//! Provides `parallel_matmul` and `parallel_attention` using Rayon-based
//! parallelism.  These are thin utilities; the primary compute path lives in
//! `bitnet-kernels`.

use bitnet_common::{BitNetError, Result};
use rayon::prelude::*;

/// Parallel matrix-multiplication (row-partitioned, Rayon).
///
/// `C = A × B` where `A` is `m×k`, `B` is `k×n`, `C` is `m×n`.
pub fn parallel_matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    num_threads: usize,
) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(BitNetError::Config("matrix dimension mismatch".to_string()));
    }

    let chunk_size = (m + num_threads.max(1) - 1) / num_threads.max(1);

    c.par_chunks_mut(chunk_size * n).enumerate().for_each(|(chunk_idx, c_chunk)| {
        let start_row = chunk_idx * chunk_size;
        let end_row = (start_row + chunk_size).min(m);

        for i in 0..(end_row - start_row) {
            let global_i = start_row + i;
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[global_i * k + l] * b[l * n + j];
                }
                c_chunk[i * n + j] = sum;
            }
        }
    });

    Ok(())
}

/// Parallel scaled dot-product attention with numerically-stable softmax.
///
/// Implements:
/// ```text
/// scores[i, j] = (Q[i] · K[j]) / √head_dim
/// attn[i]      = softmax(scores[i, :])
/// output[i]    = Σ_j attn[i, j] · V[j]
/// ```
///
/// The softmax is numerically-stable (max-subtraction before `exp`).
///
/// `query`, `key`, `value`, and `output` are laid out as
/// `[num_heads, seq_len, head_dim]`.
pub fn parallel_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
) -> Result<()> {
    let scale = 1.0 / (head_dim as f32).sqrt();

    output.par_chunks_mut(seq_len * head_dim).enumerate().try_for_each(
        |(head_idx, head_output)| -> Result<()> {
            if head_idx >= num_heads {
                return Ok(());
            }

            let q_offset = head_idx * seq_len * head_dim;
            let k_offset = head_idx * seq_len * head_dim;
            let v_offset = head_idx * seq_len * head_dim;

            let mut scores = vec![0.0f32; seq_len];

            for i in 0..seq_len {
                // Scaled dot-product Q[i] · K[j] for every j.
                for j in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot +=
                            query[q_offset + i * head_dim + d] * key[k_offset + j * head_dim + d];
                    }
                    scores[j] = dot * scale;
                }

                // Numerically-stable softmax.
                let max_score = scores[..seq_len].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for j in 0..seq_len {
                    scores[j] = (scores[j] - max_score).exp();
                    sum_exp += scores[j];
                }

                // Weighted sum of values; zero the output slot first.
                let out_base = i * head_dim;
                for d in 0..head_dim {
                    head_output[out_base + d] = 0.0;
                }
                if sum_exp > 0.0 {
                    for j in 0..seq_len {
                        let w = scores[j] / sum_exp;
                        for d in 0..head_dim {
                            head_output[out_base + d] += w * value[v_offset + j * head_dim + d];
                        }
                    }
                }
            }

            Ok(())
        },
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity matrix: A × I = A.
    #[test]
    fn test_parallel_matmul_identity() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2×2
        let b = vec![1.0f32, 0.0, 0.0, 1.0]; // identity
        let mut c = vec![0.0f32; 4];
        parallel_matmul(&a, &b, &mut c, 2, 2, 2, 2).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// When V is all-ones, output must be all-ones regardless of attention weights.
    /// This verifies that softmax weights sum to 1.0.
    #[test]
    fn test_attention_softmax_sums_to_one() {
        let seq_len = 4;
        let head_dim = 2;
        let num_heads = 1;
        let q = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let k = q.clone();
        let v = vec![1.0f32; seq_len * head_dim];
        let mut out = vec![0.0f32; seq_len * head_dim];

        parallel_attention(&q, &k, &v, &mut out, seq_len, head_dim, num_heads).unwrap();

        for (i, &val) in out.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "out[{i}] = {val}, expected ~1.0 (softmax must sum to 1)"
            );
        }
    }

    /// Single token: softmax weight is trivially 1.0; output equals value.
    #[test]
    fn test_attention_single_token_passes_through_value() {
        let head_dim = 4;
        let v = vec![2.0f32; head_dim];
        let mut out = vec![0.0f32; head_dim];

        parallel_attention(&v, &v, &v, &mut out, 1, head_dim, 1).unwrap();

        for (i, &val) in out.iter().enumerate() {
            assert!((val - 2.0).abs() < 1e-5, "out[{i}] = {val}, expected 2.0");
        }
    }

    /// Dimension mismatch must return an error, not panic.
    #[test]
    fn test_matmul_dimension_mismatch_returns_error() {
        let a = vec![1.0f32; 4]; // 2×2
        let b = vec![1.0f32; 9]; // 3×3 (mismatched)
        let mut c = vec![0.0f32; 4];
        assert!(parallel_matmul(&a, &b, &mut c, 2, 2, 2, 2).is_err());
    }
}
