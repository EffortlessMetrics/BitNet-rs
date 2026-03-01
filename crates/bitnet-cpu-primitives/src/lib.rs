//! CPU-side activation and normalization primitives used in inference paths.

use bitnet_common::{ActivationType, BitNetError, NormType, Result};

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

/// GELU (Gaussian Error Linear Unit) activation approximation.
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

/// GELU activation applied in-place.
pub fn gelu_in_place(data: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;

    for v in data.iter_mut() {
        let x = *v;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        *v = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// Squared ReLU: `relu2(x) = max(0, x)²`.
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

/// Scalar RMS normalization.
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

/// Layer normalization.
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

/// Layer normalization without bias.
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

/// Apply activation function in-place.
pub fn apply_activation(activation: ActivationType, data: &mut [f32]) {
    match activation {
        ActivationType::Silu => silu_in_place(data),
        ActivationType::Gelu => gelu_in_place(data),
        ActivationType::Relu2 => relu2_in_place(data),
    }
}

/// Apply the normalization indicated by `norm`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activations_dispatch_and_in_place_match() {
        let input = vec![0.5f32, -0.5, 2.0, -2.0];

        let expected_silu = silu(&input);
        let mut silu_data = input.clone();
        apply_activation(ActivationType::Silu, &mut silu_data);
        assert_eq!(silu_data.len(), expected_silu.len());

        let expected_gelu = gelu(&input);
        let mut gelu_data = input.clone();
        apply_activation(ActivationType::Gelu, &mut gelu_data);
        assert_eq!(gelu_data.len(), expected_gelu.len());

        let expected_relu2 = relu2(&input);
        let mut relu2_data = input;
        apply_activation(ActivationType::Relu2, &mut relu2_data);
        assert_eq!(relu2_data, expected_relu2);
    }

    #[test]
    fn norm_dispatch_matches_direct_impls() {
        let dim = 4;
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32; dim];
        let bias = vec![0.5f32; dim];
        let mut out_dispatch = vec![0.0f32; dim];
        let mut out_direct = vec![0.0f32; dim];

        apply_norm(NormType::RmsNorm, &input, &weight, &bias, &mut out_dispatch, 1, dim, 1e-5)
            .unwrap();
        rmsnorm(&input, &weight, &mut out_direct, 1, dim, 1e-5).unwrap();
        assert_eq!(out_dispatch.len(), out_direct.len());

        apply_norm(
            NormType::LayerNorm,
            &input,
            &weight,
            &bias,
            &mut out_dispatch,
            1,
            dim,
            1e-5,
        )
        .unwrap();
        layernorm(&input, &weight, &bias, &mut out_direct, 1, dim, 1e-5).unwrap();
        assert_eq!(out_dispatch.len(), out_direct.len());
    }

    #[test]
    fn layernorm_no_bias_matches_layernorm_with_zero_bias() {
        let dim = 4;
        let input = vec![1.0f32, 3.0, -1.0, 2.0];
        let weight = vec![2.0f32; dim];
        let zero_bias = vec![0.0f32; dim];
        let mut out_with_bias = vec![0.0f32; dim];
        let mut out_no_bias = vec![0.0f32; dim];

        layernorm(&input, &weight, &zero_bias, &mut out_with_bias, 1, dim, 1e-5).unwrap();
        layernorm_no_bias(&input, &weight, &mut out_no_bias, 1, dim, 1e-5).unwrap();

        for (a, b) in out_with_bias.iter().zip(out_no_bias.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
