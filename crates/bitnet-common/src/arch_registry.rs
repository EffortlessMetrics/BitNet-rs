//! Centralized registry of known model architectures and their default
//! configurations.

use crate::config::{ActivationType, NormType};

/// Default configuration for a model architecture.
#[derive(Debug, Clone)]
pub struct ArchDefaults {
    pub norm_type: NormType,
    pub activation_type: ActivationType,
    /// If `Some`, override `max_position_embeddings` when it equals 2048.
    pub default_context_length: Option<usize>,
}

/// Registry of known model architectures and their defaults.
pub struct ArchitectureRegistry;

impl ArchitectureRegistry {
    /// Look up defaults for a given architecture string.
    /// Returns `None` for unknown architectures.
    pub fn lookup(architecture: &str) -> Option<ArchDefaults> {
        match architecture.to_lowercase().as_str() {
            "phi" | "phi-4" | "phi-3" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(16384),
            }),
            "llama" | "mistral" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: None,
            }),
            "qwen" | "qwen2" | "qwen2.5" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: None,
            }),
            "gemma" | "gemma2" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Gelu,
                default_context_length: None,
            }),
            "bitnet" | "bitnet-b1.58" => Some(ArchDefaults {
                norm_type: NormType::LayerNorm,
                activation_type: ActivationType::Silu,
                default_context_length: None,
            }),
            "gpt" | "bert" => Some(ArchDefaults {
                norm_type: NormType::LayerNorm,
                activation_type: ActivationType::Gelu,
                default_context_length: None,
            }),
            _ => None,
        }
    }

    /// Get all known architecture strings.
    pub fn known_architectures() -> &'static [&'static str] {
        &[
            "phi",
            "phi-4",
            "phi-3",
            "llama",
            "mistral",
            "qwen",
            "qwen2",
            "qwen2.5",
            "gemma",
            "gemma2",
            "bitnet",
            "bitnet-b1.58",
            "gpt",
            "bert",
        ]
    }

    /// Check if an architecture string is recognized.
    pub fn is_known(architecture: &str) -> bool {
        Self::lookup(architecture).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_known_return_some() {
        for arch in ArchitectureRegistry::known_architectures() {
            assert!(
                ArchitectureRegistry::lookup(arch).is_some(),
                "known architecture '{}' should return Some",
                arch,
            );
        }
    }

    #[test]
    fn test_unknown_returns_none() {
        assert!(ArchitectureRegistry::lookup("unknown_model").is_none());
        assert!(ArchitectureRegistry::lookup("").is_none());
        assert!(ArchitectureRegistry::lookup("mamba").is_none());
    }

    #[test]
    fn test_case_insensitivity() {
        assert!(ArchitectureRegistry::lookup("PHI").is_some());
        assert!(ArchitectureRegistry::lookup("Llama").is_some());
        assert!(ArchitectureRegistry::lookup("BitNet-B1.58").is_some());
        assert!(ArchitectureRegistry::lookup("GEMMA2").is_some());
        assert!(ArchitectureRegistry::lookup("Qwen2.5").is_some());
    }

    #[test]
    fn test_known_architectures_consistent_with_lookup() {
        for arch in ArchitectureRegistry::known_architectures() {
            assert!(
                ArchitectureRegistry::is_known(arch),
                "is_known('{}') should be true",
                arch,
            );
        }
    }

    #[test]
    fn test_phi_defaults() {
        let defaults = ArchitectureRegistry::lookup("phi").unwrap();
        assert_eq!(defaults.norm_type, NormType::RmsNorm);
        assert_eq!(defaults.activation_type, ActivationType::Silu);
        assert_eq!(defaults.default_context_length, Some(16384));
    }

    #[test]
    fn test_bitnet_defaults() {
        let defaults = ArchitectureRegistry::lookup("bitnet").unwrap();
        assert_eq!(defaults.norm_type, NormType::LayerNorm);
        assert_eq!(defaults.activation_type, ActivationType::Silu);
        assert_eq!(defaults.default_context_length, None);
    }

    #[test]
    fn test_gemma_defaults() {
        let defaults = ArchitectureRegistry::lookup("gemma").unwrap();
        assert_eq!(defaults.norm_type, NormType::RmsNorm);
        assert_eq!(defaults.activation_type, ActivationType::Gelu);
    }

    #[test]
    fn test_gpt_defaults() {
        let defaults = ArchitectureRegistry::lookup("gpt").unwrap();
        assert_eq!(defaults.norm_type, NormType::LayerNorm);
        assert_eq!(defaults.activation_type, ActivationType::Gelu);
    }

    #[test]
    fn test_is_known() {
        assert!(ArchitectureRegistry::is_known("llama"));
        assert!(ArchitectureRegistry::is_known("BERT"));
        assert!(!ArchitectureRegistry::is_known("unknown"));
    }
}
