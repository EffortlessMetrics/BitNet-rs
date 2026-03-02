//! Activation function registry.
//!
//! Centralized dispatch for activation functions used by different model families.

/// Supported activation function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationType {
    ReLU,
    ReLU2, // Squared ReLU (BitNet)
    SiLU,  // Swish (Phi/LLaMA/Mistral)
    GeLU,
    GeLUTanh, // GELU with tanh approximation
    Tanh,
    Sigmoid,
    Mish,
}

impl ActivationType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ReLU => "relu",
            Self::ReLU2 => "relu2",
            Self::SiLU => "silu",
            Self::GeLU => "gelu",
            Self::GeLUTanh => "gelu_tanh",
            Self::Tanh => "tanh",
            Self::Sigmoid => "sigmoid",
            Self::Mish => "mish",
        }
    }

    /// Parse from string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "relu" => Some(Self::ReLU),
            "relu2" | "relu_squared" | "squared_relu" => Some(Self::ReLU2),
            "silu" | "swish" => Some(Self::SiLU),
            "gelu" => Some(Self::GeLU),
            "gelu_tanh" | "gelu_new" | "gelu_fast" => Some(Self::GeLUTanh),
            "tanh" => Some(Self::Tanh),
            "sigmoid" => Some(Self::Sigmoid),
            "mish" => Some(Self::Mish),
            _ => None,
        }
    }

    /// Default activation for a model family.
    pub fn for_family(family: &str) -> Self {
        match family.to_lowercase().as_str() {
            "bitnet" => Self::ReLU2,
            "phi" | "phi2" | "phi3" | "phi4" => Self::SiLU,
            "llama" | "llama2" | "llama3" | "mistral" | "mixtral" => Self::SiLU,
            "qwen" | "qwen2" => Self::SiLU,
            "gemma" | "gemma2" => Self::GeLUTanh,
            "gpt2" | "gptneo" => Self::GeLU,
            "falcon" => Self::GeLU,
            _ => Self::SiLU, // safe default
        }
    }
}

/// Apply activation function to a single value.
pub fn activate(x: f32, act_type: ActivationType) -> f32 {
    match act_type {
        ActivationType::ReLU => x.max(0.0),
        ActivationType::ReLU2 => {
            let r = x.max(0.0);
            r * r
        }
        ActivationType::SiLU => x * sigmoid(x),
        ActivationType::GeLU => 0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2)),
        ActivationType::GeLUTanh => {
            let c = (2.0 / std::f32::consts::PI).sqrt();
            0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
        }
        ActivationType::Tanh => x.tanh(),
        ActivationType::Sigmoid => sigmoid(x),
        ActivationType::Mish => x * (softplus(x).tanh()),
    }
}

/// Apply activation function to a slice in-place.
pub fn activate_inplace(data: &mut [f32], act_type: ActivationType) {
    for v in data.iter_mut() {
        *v = activate(*v, act_type);
    }
}

/// Apply activation function to a slice, returning new vector.
pub fn activate_vec(data: &[f32], act_type: ActivationType) -> Vec<f32> {
    data.iter().map(|&x| activate(x, act_type)).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    (1.0 + x.exp()).ln()
}

fn erf(x: f32) -> f32 {
    // Abramowitz and Stegun approximation
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 { result } else { -result }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        assert_eq!(activate(3.0, ActivationType::ReLU), 3.0);
        assert_eq!(activate(-1.0, ActivationType::ReLU), 0.0);
    }

    #[test]
    fn test_relu2() {
        assert!((activate(2.0, ActivationType::ReLU2) - 4.0).abs() < 0.01);
        assert_eq!(activate(-1.0, ActivationType::ReLU2), 0.0);
    }

    #[test]
    fn test_silu() {
        // silu(0) = 0
        assert!(activate(0.0, ActivationType::SiLU).abs() < 0.01);
        // silu(x) > 0 for x > 0
        assert!(activate(2.0, ActivationType::SiLU) > 0.0);
    }

    #[test]
    fn test_gelu() {
        assert!(activate(0.0, ActivationType::GeLU).abs() < 0.01);
        assert!(activate(2.0, ActivationType::GeLU) > 1.5);
    }

    #[test]
    fn test_gelu_tanh() {
        assert!(activate(0.0, ActivationType::GeLUTanh).abs() < 0.01);
        assert!(activate(2.0, ActivationType::GeLUTanh) > 1.5);
    }

    #[test]
    fn test_sigmoid() {
        assert!((activate(0.0, ActivationType::Sigmoid) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_tanh_act() {
        assert!(activate(0.0, ActivationType::Tanh).abs() < 0.01);
    }

    #[test]
    fn test_mish() {
        assert!(activate(0.0, ActivationType::Mish).abs() < 0.01);
        assert!(activate(2.0, ActivationType::Mish) > 1.5);
    }

    #[test]
    fn test_from_name() {
        assert_eq!(ActivationType::from_name("silu"), Some(ActivationType::SiLU));
        assert_eq!(ActivationType::from_name("swish"), Some(ActivationType::SiLU));
        assert_eq!(ActivationType::from_name("relu2"), Some(ActivationType::ReLU2));
        assert_eq!(ActivationType::from_name("unknown"), None);
    }

    #[test]
    fn test_for_family() {
        assert_eq!(ActivationType::for_family("bitnet"), ActivationType::ReLU2);
        assert_eq!(ActivationType::for_family("phi4"), ActivationType::SiLU);
        assert_eq!(ActivationType::for_family("llama3"), ActivationType::SiLU);
        assert_eq!(ActivationType::for_family("gemma"), ActivationType::GeLUTanh);
    }

    #[test]
    fn test_activate_inplace() {
        let mut data = vec![-1.0, 0.0, 1.0, 2.0];
        activate_inplace(&mut data, ActivationType::ReLU);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[2], 1.0);
    }

    #[test]
    fn test_activate_vec() {
        let data = vec![0.0, 1.0, -1.0];
        let out = activate_vec(&data, ActivationType::ReLU);
        assert_eq!(out, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_name_roundtrip() {
        for act in [ActivationType::ReLU, ActivationType::SiLU, ActivationType::GeLU] {
            let name = act.name();
            assert_eq!(ActivationType::from_name(name), Some(act));
        }
    }

    #[test]
    fn test_silu_negative() {
        // SiLU can be negative for negative x (but small)
        let v = activate(-3.0, ActivationType::SiLU);
        assert!(v < 0.0);
        assert!(v > -0.2);
    }
}
