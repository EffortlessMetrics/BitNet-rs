//! Normalization layer registry.
//!
//! Centralized dispatch for normalization types used by different models.

/// Supported normalization types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormType {
    LayerNorm,
    RmsNorm,
    SubNorm, // BitNet sub-layer norm
    GroupNorm { groups: usize },
    BatchNorm,
}

impl NormType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::LayerNorm => "layer_norm",
            Self::RmsNorm => "rms_norm",
            Self::SubNorm => "sub_norm",
            Self::GroupNorm { .. } => "group_norm",
            Self::BatchNorm => "batch_norm",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "layer_norm" | "layernorm" | "ln" => Some(Self::LayerNorm),
            "rms_norm" | "rmsnorm" => Some(Self::RmsNorm),
            "sub_norm" | "subnorm" | "bitnorm" => Some(Self::SubNorm),
            "batch_norm" | "batchnorm" | "bn" => Some(Self::BatchNorm),
            _ => None,
        }
    }

    pub fn for_family(family: &str) -> Self {
        match family.to_lowercase().as_str() {
            "bitnet" => Self::SubNorm,
            "phi" | "phi2" | "phi3" | "phi4" => Self::RmsNorm,
            "llama" | "llama2" | "llama3" | "mistral" | "mixtral" => Self::RmsNorm,
            "qwen" | "qwen2" => Self::RmsNorm,
            "gemma" | "gemma2" => Self::RmsNorm,
            "gpt2" | "gptneo" => Self::LayerNorm,
            "falcon" => Self::LayerNorm,
            _ => Self::LayerNorm,
        }
    }
}

/// Normalization configuration.
#[derive(Debug, Clone)]
pub struct NormConfig {
    pub norm_type: NormType,
    pub hidden_size: usize,
    pub eps: f64,
    pub affine: bool,
}

impl Default for NormConfig {
    fn default() -> Self {
        Self { norm_type: NormType::LayerNorm, hidden_size: 4096, eps: 1e-5, affine: true }
    }
}

/// Apply LayerNorm to a vector (in f64 for precision).
pub fn layer_norm(data: &mut [f32], weight: &[f32], bias: Option<&[f32]>, eps: f64) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var: f64 = data
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let inv_std = 1.0 / (var + eps).sqrt();

    for i in 0..n {
        let normed = ((data[i] as f64 - mean) * inv_std) as f32;
        data[i] = normed * weight.get(i).copied().unwrap_or(1.0)
            + bias.and_then(|b| b.get(i)).copied().unwrap_or(0.0);
    }
}

/// Apply RMSNorm to a vector (in f64 for precision).
pub fn rms_norm(data: &mut [f32], weight: &[f32], eps: f64) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let rms: f64 =
        (data.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    for i in 0..n {
        data[i] = ((data[i] as f64 * inv_rms) as f32) * weight.get(i).copied().unwrap_or(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_names() {
        assert_eq!(NormType::LayerNorm.name(), "layer_norm");
        assert_eq!(NormType::RmsNorm.name(), "rms_norm");
        assert_eq!(NormType::SubNorm.name(), "sub_norm");
    }

    #[test]
    fn test_from_name() {
        assert_eq!(NormType::from_name("rmsnorm"), Some(NormType::RmsNorm));
        assert_eq!(NormType::from_name("layernorm"), Some(NormType::LayerNorm));
        assert_eq!(NormType::from_name("bitnorm"), Some(NormType::SubNorm));
        assert_eq!(NormType::from_name("unknown"), None);
    }

    #[test]
    fn test_for_family() {
        assert_eq!(NormType::for_family("bitnet"), NormType::SubNorm);
        assert_eq!(NormType::for_family("phi4"), NormType::RmsNorm);
        assert_eq!(NormType::for_family("llama3"), NormType::RmsNorm);
        assert_eq!(NormType::for_family("gpt2"), NormType::LayerNorm);
    }

    #[test]
    fn test_layer_norm() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        layer_norm(&mut data, &weight, None, 1e-5);
        // Mean = 2.5, after norm: close to [-1.34, -0.45, 0.45, 1.34]
        assert!(data[0] < 0.0);
        assert!(data[3] > 0.0);
    }

    #[test]
    fn test_layer_norm_with_bias() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let bias = vec![10.0; 4];
        layer_norm(&mut data, &weight, Some(&bias), 1e-5);
        // All shifted by +10
        assert!(data.iter().all(|&v| v > 5.0));
    }

    #[test]
    fn test_rms_norm() {
        let mut data = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![1.0; 4];
        rms_norm(&mut data, &weight, 1e-5);
        // RMS of [1,1,1,1] = 1, so output ≈ [1,1,1,1]
        for &v in &data {
            assert!((v - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_rms_norm_varied() {
        let mut data = vec![2.0, 0.0, 2.0, 0.0];
        let weight = vec![1.0; 4];
        rms_norm(&mut data, &weight, 1e-5);
        // RMS = sqrt((4+0+4+0)/4) = sqrt(2) ≈ 1.414
        let expected = 2.0 / (2.0f32).sqrt();
        assert!((data[0] - expected).abs() < 0.01);
    }

    #[test]
    fn test_rms_norm_empty() {
        let mut data: Vec<f32> = vec![];
        rms_norm(&mut data, &[], 1e-5);
        assert!(data.is_empty());
    }

    #[test]
    fn test_layer_norm_empty() {
        let mut data: Vec<f32> = vec![];
        layer_norm(&mut data, &[], None, 1e-5);
        assert!(data.is_empty());
    }

    #[test]
    fn test_config_default() {
        let c = NormConfig::default();
        assert_eq!(c.norm_type, NormType::LayerNorm);
        assert!((c.eps - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_group_norm_name() {
        let n = NormType::GroupNorm { groups: 32 };
        assert_eq!(n.name(), "group_norm");
    }

    #[test]
    fn test_norm_type_eq() {
        assert_eq!(NormType::RmsNorm, NormType::RmsNorm);
        assert_ne!(NormType::RmsNorm, NormType::LayerNorm);
    }

    #[test]
    fn test_rms_norm_scaling() {
        let mut data = vec![3.0, 4.0];
        let weight = vec![2.0, 2.0];
        rms_norm(&mut data, &weight, 1e-5);
        // RMS = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        // Output scaled by 2x weight
        assert!(data[0] > 1.0);
        assert!(data[1] > 1.0);
    }
}
