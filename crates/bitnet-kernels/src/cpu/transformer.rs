//! CPU transformer block kernel combining attention, FFN, and layer norm.

use std::fmt;

/// Configuration for a transformer block
#[derive(Debug, Clone)]
pub struct TransformerBlockConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,
    pub use_pre_norm: bool,
    pub residual_scale: f32,
}

impl Default for TransformerBlockConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_heads: 32,
            intermediate_size: 5632,
            layer_norm_eps: 1e-5,
            use_pre_norm: true,
            residual_scale: 1.0,
        }
    }
}

/// Errors from transformer block operations
#[derive(Debug)]
pub enum TransformerBlockError {
    DimensionMismatch { expected: usize, got: usize },
    InvalidConfig(String),
    AttentionError(String),
    FfnError(String),
}

impl fmt::Display for TransformerBlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::AttentionError(msg) => write!(f, "attention error: {msg}"),
            Self::FfnError(msg) => write!(f, "FFN error: {msg}"),
        }
    }
}

impl std::error::Error for TransformerBlockError {}

/// Apply RMS layer normalization
pub fn rms_layer_norm(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), TransformerBlockError> {
    if input.len() != weight.len() || input.len() != output.len() {
        return Err(TransformerBlockError::DimensionMismatch {
            expected: input.len(),
            got: weight.len(),
        });
    }
    let n = input.len();
    let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let rms = (mean_sq + eps).sqrt();
    for i in 0..n {
        output[i] = (input[i] / rms) * weight[i];
    }
    Ok(())
}

/// Apply standard layer normalization
pub fn layer_norm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), TransformerBlockError> {
    if input.len() != weight.len() || input.len() != bias.len() || input.len() != output.len() {
        return Err(TransformerBlockError::DimensionMismatch {
            expected: input.len(),
            got: weight.len(),
        });
    }
    let n = input.len();
    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let std_inv = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        output[i] = ((input[i] - mean) * std_inv) * weight[i] + bias[i];
    }
    Ok(())
}

/// Add residual connection with optional scaling
pub fn add_residual(
    input: &[f32],
    residual: &[f32],
    scale: f32,
    output: &mut [f32],
) -> Result<(), TransformerBlockError> {
    if input.len() != residual.len() || input.len() != output.len() {
        return Err(TransformerBlockError::DimensionMismatch {
            expected: input.len(),
            got: residual.len(),
        });
    }
    for i in 0..input.len() {
        output[i] = input[i] + residual[i] * scale;
    }
    Ok(())
}

/// Execute a complete pre-norm transformer block.
///
/// Flow: hidden → LN → attention → residual → LN → FFN → residual
#[allow(clippy::too_many_arguments)]
pub fn transformer_block_pre_norm(
    input: &[f32],
    attn_ln_weight: &[f32],
    ffn_ln_weight: &[f32],
    attn_qkv_weight: &[f32],
    attn_out_weight: &[f32],
    ffn_gate_weight: &[f32],
    ffn_up_weight: &[f32],
    ffn_down_weight: &[f32],
    config: &TransformerBlockConfig,
    output: &mut [f32],
) -> Result<(), TransformerBlockError> {
    let hidden = config.hidden_size;
    if input.len() != hidden {
        return Err(TransformerBlockError::DimensionMismatch {
            expected: hidden,
            got: input.len(),
        });
    }

    // Step 1: Pre-attention layer norm
    let mut normed = vec![0.0f32; hidden];
    rms_layer_norm(input, attn_ln_weight, config.layer_norm_eps, &mut normed)?;

    // Step 2: Self-attention (simplified single-token path)
    let qkv_size = hidden * 3;
    if attn_qkv_weight.len() != qkv_size * hidden {
        return Err(TransformerBlockError::AttentionError(format!(
            "QKV weight size mismatch: expected {}, got {}",
            qkv_size * hidden,
            attn_qkv_weight.len()
        )));
    }
    let mut qkv = vec![0.0f32; qkv_size];
    for i in 0..qkv_size {
        let mut sum = 0.0f32;
        for j in 0..hidden {
            sum += normed[j] * attn_qkv_weight[i * hidden + j];
        }
        qkv[i] = sum;
    }

    // Split Q, K, V and compute scaled dot-product attention
    let (_q, rest) = qkv.split_at(hidden);
    let (_k, v) = rest.split_at(hidden);

    // Single-token attention: softmax of a single score is always 1.0
    let mut attn_out = vec![0.0f32; hidden];
    attn_out[..hidden].copy_from_slice(&v[..hidden]);

    // Output projection
    if attn_out_weight.len() != hidden * hidden {
        return Err(TransformerBlockError::AttentionError(format!(
            "output weight size mismatch: expected {}, got {}",
            hidden * hidden,
            attn_out_weight.len()
        )));
    }
    let mut projected = vec![0.0f32; hidden];
    for i in 0..hidden {
        let mut sum = 0.0f32;
        for j in 0..hidden {
            sum += attn_out[j] * attn_out_weight[i * hidden + j];
        }
        projected[i] = sum;
    }

    // Step 3: Residual connection
    let mut post_attn = vec![0.0f32; hidden];
    add_residual(&projected, input, config.residual_scale, &mut post_attn)?;

    // Step 4: Pre-FFN layer norm
    let mut ffn_normed = vec![0.0f32; hidden];
    rms_layer_norm(&post_attn, ffn_ln_weight, config.layer_norm_eps, &mut ffn_normed)?;

    // Step 5: FFN (SwiGLU variant)
    let inter = config.intermediate_size;
    if ffn_gate_weight.len() != inter * hidden
        || ffn_up_weight.len() != inter * hidden
        || ffn_down_weight.len() != hidden * inter
    {
        return Err(TransformerBlockError::FfnError("FFN weight size mismatch".to_string()));
    }

    let mut gate = vec![0.0f32; inter];
    let mut up = vec![0.0f32; inter];
    for i in 0..inter {
        let mut g = 0.0f32;
        let mut u = 0.0f32;
        for j in 0..hidden {
            g += ffn_normed[j] * ffn_gate_weight[i * hidden + j];
            u += ffn_normed[j] * ffn_up_weight[i * hidden + j];
        }
        // SiLU activation on gate
        gate[i] = g / (1.0 + (-g).exp());
        up[i] = u;
    }

    // Element-wise multiply gate * up, then down projection
    let ffn_hidden: Vec<f32> = gate.iter().zip(up.iter()).map(|(g, u)| g * u).collect();

    let mut ffn_out = vec![0.0f32; hidden];
    for i in 0..hidden {
        let mut sum = 0.0f32;
        for j in 0..inter {
            sum += ffn_hidden[j] * ffn_down_weight[i * inter + j];
        }
        ffn_out[i] = sum;
    }

    // Step 6: Final residual
    add_residual(&ffn_out, &post_attn, config.residual_scale, output)?;

    Ok(())
}

/// Validate transformer block configuration
pub fn validate_config(config: &TransformerBlockConfig) -> Result<(), TransformerBlockError> {
    if config.hidden_size == 0 {
        return Err(TransformerBlockError::InvalidConfig("hidden_size must be > 0".to_string()));
    }
    if config.num_heads == 0 {
        return Err(TransformerBlockError::InvalidConfig("num_heads must be > 0".to_string()));
    }
    if !config.hidden_size.is_multiple_of(config.num_heads) {
        return Err(TransformerBlockError::InvalidConfig(format!(
            "hidden_size ({}) must be divisible by num_heads ({})",
            config.hidden_size, config.num_heads
        )));
    }
    if config.intermediate_size == 0 {
        return Err(TransformerBlockError::InvalidConfig(
            "intermediate_size must be > 0".to_string(),
        ));
    }
    if config.layer_norm_eps <= 0.0 {
        return Err(TransformerBlockError::InvalidConfig("layer_norm_eps must be > 0".to_string()));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_layer_norm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        rms_layer_norm(&input, &weight, 1e-5, &mut output).unwrap();
        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (7.5_f32 + 1e-5).sqrt();
        for i in 0..4 {
            assert!((output[i] - input[i] / rms).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rms_layer_norm_dimension_mismatch() {
        let input = vec![1.0, 2.0];
        let weight = vec![1.0; 3];
        let mut output = vec![0.0; 2];
        assert!(rms_layer_norm(&input, &weight, 1e-5, &mut output).is_err());
    }

    #[test]
    fn test_layer_norm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];
        let mut output = vec![0.0; 4];
        layer_norm(&input, &weight, &bias, 1e-5, &mut output).unwrap();
        let mean: f32 = input.iter().sum::<f32>() / 4.0;
        let var: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        let std_inv = 1.0 / (var + 1e-5_f32).sqrt();
        for i in 0..4 {
            let expected = (input[i] - mean) * std_inv;
            assert!((output[i] - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn test_add_residual_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let residual = vec![0.5, 1.0, 1.5];
        let mut output = vec![0.0; 3];
        add_residual(&input, &residual, 1.0, &mut output).unwrap();
        assert_eq!(output, vec![1.5, 3.0, 4.5]);
    }

    #[test]
    fn test_add_residual_with_scale() {
        let input = vec![1.0, 2.0];
        let residual = vec![2.0, 4.0];
        let mut output = vec![0.0; 2];
        add_residual(&input, &residual, 0.5, &mut output).unwrap();
        assert_eq!(output, vec![2.0, 4.0]);
    }

    #[test]
    fn test_validate_config_valid() {
        let config = TransformerBlockConfig::default();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_zero_hidden() {
        let config = TransformerBlockConfig { hidden_size: 0, ..Default::default() };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_zero_heads() {
        let config = TransformerBlockConfig { num_heads: 0, ..Default::default() };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_indivisible() {
        let config =
            TransformerBlockConfig { hidden_size: 100, num_heads: 3, ..Default::default() };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_zero_intermediate() {
        let config = TransformerBlockConfig { intermediate_size: 0, ..Default::default() };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_bad_eps() {
        let config = TransformerBlockConfig { layer_norm_eps: 0.0, ..Default::default() };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_default_config() {
        let config = TransformerBlockConfig::default();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.intermediate_size, 5632);
        assert!(config.use_pre_norm);
    }

    #[test]
    fn test_transformer_block_small() {
        let hidden = 4;
        let inter = 8;
        let heads = 2;
        let config = TransformerBlockConfig {
            hidden_size: hidden,
            num_heads: heads,
            intermediate_size: inter,
            layer_norm_eps: 1e-5,
            use_pre_norm: true,
            residual_scale: 1.0,
        };
        validate_config(&config).unwrap();

        let input = vec![0.1, 0.2, 0.3, 0.4];
        let ln_w = vec![1.0; hidden];
        let qkv_w = vec![0.01; hidden * 3 * hidden];
        let out_w = vec![0.01; hidden * hidden];
        let gate_w = vec![0.01; inter * hidden];
        let up_w = vec![0.01; inter * hidden];
        let down_w = vec![0.01; hidden * inter];
        let mut output = vec![0.0; hidden];

        let result = transformer_block_pre_norm(
            &input,
            &ln_w,
            &ln_w,
            &qkv_w,
            &out_w,
            &gate_w,
            &up_w,
            &down_w,
            &config,
            &mut output,
        );
        assert!(result.is_ok());
        // Output should be close to input (small weights = small changes)
        for i in 0..hidden {
            assert!(
                (output[i] - input[i]).abs() < 1.0,
                "output[{i}] = {} diverged from input[{i}] = {}",
                output[i],
                input[i]
            );
        }
    }

    #[test]
    fn test_transformer_block_wrong_input_size() {
        let config = TransformerBlockConfig {
            hidden_size: 4,
            num_heads: 2,
            intermediate_size: 8,
            ..Default::default()
        };
        let input = vec![0.1, 0.2]; // wrong size
        let w = vec![0.01; 100];
        let mut output = vec![0.0; 4];
        let result =
            transformer_block_pre_norm(&input, &w, &w, &w, &w, &w, &w, &w, &config, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_display() {
        let e = TransformerBlockError::DimensionMismatch { expected: 4, got: 3 };
        assert_eq!(e.to_string(), "dimension mismatch: expected 4, got 3");

        let e = TransformerBlockError::InvalidConfig("bad".to_string());
        assert_eq!(e.to_string(), "invalid config: bad");
    }

    #[test]
    fn test_rms_layer_norm_uniform_input() {
        let input = vec![2.0; 8];
        let weight = vec![1.0; 8];
        let mut output = vec![0.0; 8];
        rms_layer_norm(&input, &weight, 1e-5, &mut output).unwrap();
        // All same → RMS ≈ 2.0, output ≈ 1.0
        for val in &output {
            assert!((val - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_layer_norm_with_bias() {
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0; 4];
        let bias = vec![5.0; 4];
        let mut output = vec![0.0; 4];
        layer_norm(&input, &weight, &bias, 1e-5, &mut output).unwrap();
        for val in &output {
            assert!((val - 5.0).abs() < 1e-4);
        }
    }
}
