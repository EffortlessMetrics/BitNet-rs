//! CPU-specific optimized kernels for inference.
//!
//! Provides:
//! - **Activations/normalizations** from `bitnet-cpu-primitives`.
//! - **Linear algebra**: `parallel_matmul`, `parallel_attention`.

use bitnet_common::{BitNetError, Result};
use rayon::prelude::*;

pub use bitnet_cpu_primitives::{
    apply_activation, apply_norm, gelu, gelu_in_place, layernorm, layernorm_no_bias, relu2,
    relu2_in_place, rmsnorm, silu, silu_in_place,
};

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

    let chunk_size = m.div_ceil(num_threads.max(1));

    c.par_chunks_mut(chunk_size * n)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
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
                for j in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot +=
                            query[q_offset + i * head_dim + d] * key[k_offset + j * head_dim + d];
                    }
                    scores[j] = dot * scale;
                }

                let max_score = scores[..seq_len]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for score in &mut scores[..seq_len] {
                    *score = (*score - max_score).exp();
                    sum_exp += *score;
                }

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

    #[test]
    fn test_parallel_matmul_identity() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut c = vec![0.0f32; 4];
        parallel_matmul(&a, &b, &mut c, 2, 2, 2, 2).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_attention_single_token_passes_through_value() {
        let head_dim = 4;
        let v = vec![2.0f32; head_dim];
        let mut out = vec![0.0f32; head_dim];

        parallel_attention(&v, &v, &v, &mut out, 1, head_dim, 1).unwrap();

        for &val in &out {
            assert!((val - 2.0).abs() < 1e-5);
        }
    }
}
