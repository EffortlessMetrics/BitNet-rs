//! CPU-specific optimised kernels for inference.
//!
//! Provides:
//! - **Activations**: `silu`, `gelu`, `relu2` (+ in-place variants) and
//!   `apply_activation` dispatch.
//! - **Normalisations**: `rmsnorm`, `layernorm`, `layernorm_no_bias` and
//!   `apply_norm` dispatch.
//! - **Linear algebra**: `parallel_matmul`, `parallel_attention`.
//!
//! These are thin utilities; the primary compute path lives in `bitnet-kernels`.

use bitnet_common::{ActivationType, BitNetError, NormType, Result};
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

    let chunk_size = m.div_ceil(num_threads.max(1));

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

/// Scalar RMS normalisation.
///
/// For each row of length `dim`:
/// ```text
/// rms  = sqrt(mean(x²) + eps)
/// out[i] = (x[i] / rms) * weight[i]
/// ```
///
/// `input` and `output` are `[rows, dim]`; `weight` is `[dim]`.
pub fn rmsnorm(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    if input.len() != rows * dim || output.len() != rows * dim {
        return Err(BitNetError::Config("rmsnorm: input/output size mismatch".to_string()));
    }
    if weight.len() != dim {
        return Err(BitNetError::Config("rmsnorm: weight size mismatch".to_string()));
    }

    for row in 0..rows {
        let base = row * dim;
        let slice = &input[base..base + dim];

        let mean_sq: f32 = slice.iter().map(|&v| v * v).sum::<f32>() / dim as f32;
        let rms = (mean_sq + eps).sqrt();

        for d in 0..dim {
            output[base + d] = (slice[d] / rms) * weight[d];
        }
    }

    Ok(())
}

/// SiLU (Sigmoid Linear Unit) activation: `x * σ(x)`.
///
/// Applied element-wise in-place.
pub fn silu_in_place(data: &mut [f32]) {
    for v in data.iter_mut() {
        let sigma = 1.0 / (1.0 + (-*v).exp());
        *v *= sigma;
    }
}

/// Element-wise SiLU returning a new vector.
pub fn silu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
}

// ---------------------------------------------------------------------------
// GELU activation
// ---------------------------------------------------------------------------

/// GELU activation using the fast tanh approximation.
///
/// ```text
/// gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
/// ```
///
/// Matches PyTorch `nn.GELU(approximate='tanh')` within ~0.01.
pub fn gelu(input: &[f32]) -> Vec<f32> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    input
        .iter()
        .map(|&x| {
            let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect()
}

/// GELU activation applied element-wise in-place.
pub fn gelu_in_place(data: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    for v in data.iter_mut() {
        let x = *v;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        *v = 0.5 * x * (1.0 + inner.tanh());
    }
}

// ---------------------------------------------------------------------------
// Squared ReLU (ReLU²) activation
// ---------------------------------------------------------------------------

/// Squared ReLU: `relu2(x) = max(0, x)²`.
///
/// Used by BitNet-specific architectures.
pub fn relu2(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            let r = x.max(0.0);
            r * r
        })
        .collect()
}

/// Squared ReLU applied element-wise in-place.
pub fn relu2_in_place(data: &mut [f32]) {
    for v in data.iter_mut() {
        let r = v.max(0.0);
        *v = r * r;
    }
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

/// Standard LayerNorm with learnable weight and bias.
///
/// For each row of length `dim`:
/// ```text
/// mean = sum(x) / dim
/// var  = sum((x - mean)²) / dim
/// out[i] = ((x[i] - mean) / sqrt(var + eps)) * weight[i] + bias[i]
/// ```
pub fn layernorm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    if input.len() != rows * dim || output.len() != rows * dim {
        return Err(BitNetError::Config("layernorm: input/output size mismatch".to_string()));
    }
    if weight.len() != dim || bias.len() != dim {
        return Err(BitNetError::Config("layernorm: weight/bias size mismatch".to_string()));
    }

    for row in 0..rows {
        let base = row * dim;
        let slice = &input[base..base + dim];

        let mean: f32 = slice.iter().sum::<f32>() / dim as f32;
        let var: f32 = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for d in 0..dim {
            output[base + d] = (slice[d] - mean) * inv_std * weight[d] + bias[d];
        }
    }

    Ok(())
}

/// LayerNorm without bias (weight-only variant).
pub fn layernorm_no_bias(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    if input.len() != rows * dim || output.len() != rows * dim {
        return Err(BitNetError::Config("layernorm: input/output size mismatch".to_string()));
    }
    if weight.len() != dim {
        return Err(BitNetError::Config("layernorm: weight size mismatch".to_string()));
    }

    for row in 0..rows {
        let base = row * dim;
        let slice = &input[base..base + dim];

        let mean: f32 = slice.iter().sum::<f32>() / dim as f32;
        let var: f32 = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for d in 0..dim {
            output[base + d] = (slice[d] - mean) * inv_std * weight[d];
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Apply the activation function indicated by `activation` to `data` in-place.
///
/// This is the primary dispatch entry-point: callers pass the
/// `ActivationType` from the model config and the data buffer gets mutated
/// through the correct activation path.
pub fn apply_activation(activation: ActivationType, data: &mut [f32]) {
    match activation {
        ActivationType::Silu => silu_in_place(data),
        ActivationType::Gelu => gelu_in_place(data),
        ActivationType::Relu2 => relu2_in_place(data),
    }
}

/// Apply the normalisation indicated by `norm` to the input tensor.
///
/// Both `LayerNorm` and `RmsNorm` are supported. For `LayerNorm` the
/// `bias` slice is required (pass a zero-filled slice if not applicable).
pub fn apply_norm(
    norm: NormType,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    match norm {
        NormType::RmsNorm => rmsnorm(input, weight, output, rows, dim, eps),
        NormType::LayerNorm => layernorm(input, weight, bias, output, rows, dim, eps),
    }
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
                for score in scores[..seq_len].iter_mut() {
                    *score = (*score - max_score).exp();
                    sum_exp += *score;
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

    // -----------------------------------------------------------------------
    // Matmul
    // -----------------------------------------------------------------------

    /// Identity matrix: A × I = A.
    #[test]
    fn test_parallel_matmul_identity() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2×2
        let b = vec![1.0f32, 0.0, 0.0, 1.0]; // identity
        let mut c = vec![0.0f32; 4];
        parallel_matmul(&a, &b, &mut c, 2, 2, 2, 2).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// Dimension mismatch must return an error, not panic.
    #[test]
    fn test_matmul_dimension_mismatch_returns_error() {
        let a = vec![1.0f32; 4]; // 2×2
        let b = vec![1.0f32; 9]; // 3×3 (mismatched)
        let mut c = vec![0.0f32; 4];
        assert!(parallel_matmul(&a, &b, &mut c, 2, 2, 2, 2).is_err());
    }

    // -----------------------------------------------------------------------
    // Attention
    // -----------------------------------------------------------------------

    /// When V is all-ones, output must be all-ones regardless of attention weights.
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

    // -----------------------------------------------------------------------
    // RMSNorm
    // -----------------------------------------------------------------------

    /// RMSNorm with uniform weight should rescale by 1/rms.
    #[test]
    fn test_rmsnorm_unit_weight() {
        let dim = 4;
        let input = vec![2.0f32, 0.0, 0.0, 0.0];
        let weight = vec![1.0f32; dim];
        let mut output = vec![0.0f32; dim];

        rmsnorm(&input, &weight, &mut output, 1, dim, 1e-5).unwrap();

        // rms = sqrt(mean([4,0,0,0]) + eps) = sqrt(1 + eps) ≈ 1.0
        assert!((output[0] - 2.0).abs() < 0.01);
        assert!(output[1].abs() < 1e-5);
    }

    /// RMSNorm dimension mismatch returns error.
    #[test]
    fn test_rmsnorm_dimension_mismatch() {
        let weight = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 3]; // wrong size
        assert!(rmsnorm(&[1.0; 4], &weight, &mut output, 1, 4, 1e-5).is_err());
    }

    // -----------------------------------------------------------------------
    // SiLU
    // -----------------------------------------------------------------------

    /// SiLU(0) = 0 and SiLU is monotonically increasing for x > 0.
    #[test]
    fn test_silu_basic() {
        let out = silu(&[0.0, 1.0, -1.0]);
        assert!(out[0].abs() < 1e-6, "silu(0) should be ~0");
        assert!(out[1] > 0.0, "silu(1) should be positive");
        assert!(out[2] < 0.0, "silu(-1) should be negative");
    }

    /// In-place SiLU matches allocating version.
    #[test]
    fn test_silu_in_place_matches() {
        let input = vec![0.5f32, -0.5, 2.0, -2.0];
        let expected = silu(&input);
        let mut data = input;
        silu_in_place(&mut data);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // -----------------------------------------------------------------------
    // GELU
    // -----------------------------------------------------------------------

    #[test]
    fn test_gelu_zero() {
        let out = gelu(&[0.0]);
        assert!(out[0].abs() < 1e-6, "gelu(0) = 0");
    }

    #[test]
    fn test_gelu_positive() {
        let out = gelu(&[1.0, 2.0, 3.0]);
        // GELU(x) ≈ x for large positive x
        for &v in &out {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_gelu_in_place_matches() {
        let input = vec![0.5f32, -0.5, 2.0, -2.0];
        let expected = gelu(&input);
        let mut data = input;
        gelu_in_place(&mut data);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gelu_reference_values() {
        // PyTorch reference: GELU(1.0) ≈ 0.8412, GELU(-1.0) ≈ -0.1588
        let out = gelu(&[1.0, -1.0]);
        assert!((out[0] - 0.8412).abs() < 0.01);
        assert!((out[1] - (-0.1588)).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Squared ReLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_relu2_basic() {
        let out = relu2(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        assert_eq!(out[0], 0.0); // negative → 0
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0); // zero → 0
        assert_eq!(out[3], 1.0); // 1² = 1
        assert_eq!(out[4], 4.0); // 2² = 4
        assert_eq!(out[5], 9.0); // 3² = 9
    }

    #[test]
    fn test_relu2_in_place_matches() {
        let input = vec![-1.0f32, 0.0, 0.5, 2.0];
        let expected = relu2(&input);
        let mut data = input;
        relu2_in_place(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_relu2_all_negative() {
        let out = relu2(&[-5.0, -3.0, -0.001]);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // LayerNorm
    // -----------------------------------------------------------------------

    #[test]
    fn test_layernorm_uniform_input() {
        // Uniform input → all outputs equal to bias (since (x - mean) = 0).
        let dim = 4;
        let input = vec![5.0f32; dim];
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim];
        let mut output = vec![0.0f32; dim];

        layernorm(&input, &weight, &bias, &mut output, 1, dim, 1e-5).unwrap();

        for &v in &output {
            assert!(v.abs() < 1e-3, "uniform input ⇒ output ≈ 0 (+ bias=0)");
        }
    }

    #[test]
    fn test_layernorm_with_bias() {
        let dim = 4;
        let input = vec![5.0f32; dim];
        let weight = vec![1.0f32; dim];
        let bias = vec![3.0f32; dim];
        let mut output = vec![0.0f32; dim];

        layernorm(&input, &weight, &bias, &mut output, 1, dim, 1e-5).unwrap();

        for &v in &output {
            assert!((v - 3.0).abs() < 1e-3, "bias should shift output");
        }
    }

    #[test]
    fn test_layernorm_no_bias_matches_zero_bias() {
        let dim = 4;
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim];
        let mut out_full = vec![0.0f32; dim];
        let mut out_nb = vec![0.0f32; dim];

        layernorm(&input, &weight, &bias, &mut out_full, 1, dim, 1e-5).unwrap();
        layernorm_no_bias(&input, &weight, &mut out_nb, 1, dim, 1e-5).unwrap();

        for (a, b) in out_full.iter().zip(out_nb.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_layernorm_dimension_mismatch() {
        let weight = vec![1.0f32; 4];
        let bias = vec![0.0f32; 4];
        let mut output = vec![0.0f32; 3]; // wrong
        assert!(layernorm(&[1.0; 4], &weight, &bias, &mut output, 1, 4, 1e-5).is_err());
    }

    // -----------------------------------------------------------------------
    // Dispatch: apply_activation
    // -----------------------------------------------------------------------

    #[test]
    fn test_dispatch_silu() {
        let mut data = vec![0.0f32, 1.0, -1.0];
        let expected = silu(&data);
        apply_activation(ActivationType::Silu, &mut data);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dispatch_gelu() {
        let mut data = vec![0.0f32, 1.0, -1.0];
        let expected = gelu(&data);
        apply_activation(ActivationType::Gelu, &mut data);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dispatch_relu2() {
        let mut data = vec![-1.0f32, 0.0, 1.0, 2.0];
        let expected = relu2(&data);
        apply_activation(ActivationType::Relu2, &mut data);
        assert_eq!(data, expected);
    }

    // -----------------------------------------------------------------------
    // Dispatch: apply_norm
    // -----------------------------------------------------------------------

    #[test]
    fn test_dispatch_rmsnorm() {
        let dim = 4;
        let input = vec![2.0f32, 0.0, 0.0, 0.0];
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim]; // ignored by RmsNorm
        let mut out_dispatch = vec![0.0f32; dim];
        let mut out_direct = vec![0.0f32; dim];

        apply_norm(NormType::RmsNorm, &input, &weight, &bias, &mut out_dispatch, 1, dim, 1e-5)
            .unwrap();
        rmsnorm(&input, &weight, &mut out_direct, 1, dim, 1e-5).unwrap();

        for (a, b) in out_dispatch.iter().zip(out_direct.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dispatch_layernorm() {
        let dim = 4;
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32; dim];
        let bias = vec![0.5f32; dim];
        let mut out_dispatch = vec![0.0f32; dim];
        let mut out_direct = vec![0.0f32; dim];

        apply_norm(NormType::LayerNorm, &input, &weight, &bias, &mut out_dispatch, 1, dim, 1e-5)
            .unwrap();
        layernorm(&input, &weight, &bias, &mut out_direct, 1, dim, 1e-5).unwrap();

        for (a, b) in out_dispatch.iter().zip(out_direct.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    /// Verify every ActivationType variant is handled (exhaustive match).
    #[test]
    fn test_dispatch_activation_all_variants() {
        let variants = [ActivationType::Silu, ActivationType::Gelu, ActivationType::Relu2];
        for variant in &variants {
            let mut data = vec![1.0f32, -1.0];
            apply_activation(*variant, &mut data);
            // Must not panic
        }
    }

    /// Verify every NormType variant is handled (exhaustive match).
    #[test]
    fn test_dispatch_norm_all_variants() {
        let dim = 4;
        let input = vec![1.0f32; dim];
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim];
        let mut output = vec![0.0f32; dim];

        let variants = [NormType::LayerNorm, NormType::RmsNorm];
        for variant in &variants {
            apply_norm(*variant, &input, &weight, &bias, &mut output, 1, dim, 1e-5).unwrap();
        }
    }
}
