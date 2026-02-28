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
            "deepseek" | "deepseek2" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: None,
            }),
            "starcoder" | "starcoder2" => Some(ArchDefaults {
                norm_type: NormType::LayerNorm,
                activation_type: ActivationType::Gelu,
                default_context_length: None,
            }),
            "falcon" => Some(ArchDefaults {
                norm_type: NormType::LayerNorm,
                activation_type: ActivationType::Gelu,
                default_context_length: None,
            }),
            "codellama" | "code-llama" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: None,
            }),
            "command" | "command-r" | "command-r-plus" | "cohere" => Some(ArchDefaults {
                norm_type: NormType::LayerNorm,
                activation_type: ActivationType::Silu,
                default_context_length: None,
            }),
            "internlm" | "internlm2" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: None,
            }),
            "yi" | "yi-1.5" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: None,
            }),
            "baichuan" | "baichuan2" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: None,
            }),
            "chatglm" | "chatglm2" | "chatglm3" | "glm-4" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: None,
            }),
            "mpt" => Some(ArchDefaults {
                norm_type: NormType::LayerNorm,
                activation_type: ActivationType::Gelu,
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
            "deepseek",
            "deepseek2",
            "starcoder",
            "starcoder2",
            "falcon",
            "codellama",
            "code-llama",
            "command",
            "command-r",
            "command-r-plus",
            "cohere",
            "internlm",
            "internlm2",
            "yi",
            "yi-1.5",
            "baichuan",
            "baichuan2",
            "chatglm",
            "chatglm2",
            "chatglm3",
            "glm-4",
            "mpt",
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
            assert!(ArchitectureRegistry::is_known(arch), "is_known('{}') should be true", arch,);
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
        assert!(ArchitectureRegistry::is_known("starcoder"));
        assert!(ArchitectureRegistry::is_known("falcon"));
        assert!(ArchitectureRegistry::is_known("deepseek"));
        assert!(ArchitectureRegistry::is_known("codellama"));
        assert!(ArchitectureRegistry::is_known("command-r"));
        assert!(ArchitectureRegistry::is_known("internlm"));
        assert!(ArchitectureRegistry::is_known("yi"));
        assert!(ArchitectureRegistry::is_known("baichuan"));
        assert!(ArchitectureRegistry::is_known("chatglm"));
        assert!(ArchitectureRegistry::is_known("mpt"));
        assert!(!ArchitectureRegistry::is_known("unknown"));
    }

    #[test]
    fn test_codellama_defaults() {
        let defaults = ArchitectureRegistry::lookup("codellama").unwrap();
        assert_eq!(defaults.norm_type, NormType::RmsNorm);
        assert_eq!(defaults.activation_type, ActivationType::Silu);
    }

    #[test]
    fn test_cohere_defaults() {
        let defaults = ArchitectureRegistry::lookup("command-r").unwrap();
        assert_eq!(defaults.norm_type, NormType::LayerNorm);
        assert_eq!(defaults.activation_type, ActivationType::Silu);
    }

    #[test]
    fn test_internlm_defaults() {
        let defaults = ArchitectureRegistry::lookup("internlm2").unwrap();
        assert_eq!(defaults.norm_type, NormType::RmsNorm);
        assert_eq!(defaults.activation_type, ActivationType::Silu);
    }

    #[test]
    fn test_yi_defaults() {
        let defaults = ArchitectureRegistry::lookup("yi").unwrap();
        assert_eq!(defaults.norm_type, NormType::RmsNorm);
        assert_eq!(defaults.activation_type, ActivationType::Silu);
    }

    #[test]
    fn test_chatglm_defaults() {
        let defaults = ArchitectureRegistry::lookup("chatglm3").unwrap();
        assert_eq!(defaults.norm_type, NormType::RmsNorm);
        assert_eq!(defaults.activation_type, ActivationType::Silu);
    }

    #[test]
    fn test_mpt_defaults() {
        let defaults = ArchitectureRegistry::lookup("mpt").unwrap();
        assert_eq!(defaults.norm_type, NormType::LayerNorm);
        assert_eq!(defaults.activation_type, ActivationType::Gelu);
    }

    #[test]
    fn test_very_long_string_returns_none() {
        let long = "a".repeat(10_000);
        assert!(ArchitectureRegistry::lookup(&long).is_none());
        assert!(!ArchitectureRegistry::is_known(&long));
    }

    #[test]
    fn test_unicode_strings_return_none() {
        let unicode_strs = ["φι", "模型", "llama-😊", "مدل", "φ-4", "ллама"];
        for s in &unicode_strs {
            assert!(
                ArchitectureRegistry::lookup(s).is_none(),
                "Unicode '{}' should not match any architecture",
                s
            );
        }
    }

    #[test]
    fn test_whitespace_variants_return_none() {
        let ws_strs = [" phi", "phi ", "\tphi", "phi\n", " phi ", "ph i"];
        for s in &ws_strs {
            assert!(
                ArchitectureRegistry::lookup(s).is_none(),
                "Whitespace variant {:?} should not match",
                s
            );
        }
    }

    #[test]
    fn test_special_character_variants_return_none() {
        let special = [
            "phi!", "phi@", "phi#", "phi$", "phi%", "phi&", "phi+", "phi=", "phi;", "phi:", "phi/",
            "phi\\",
        ];
        for s in &special {
            assert!(
                ArchitectureRegistry::lookup(s).is_none(),
                "Special char variant {:?} should not match",
                s
            );
        }
    }

    #[test]
    fn test_all_known_architectures_have_valid_defaults() {
        for arch in ArchitectureRegistry::known_architectures() {
            let defaults = ArchitectureRegistry::lookup(arch)
                .unwrap_or_else(|| panic!("'{}' is known but lookup returned None", arch));
            assert!(
                matches!(defaults.norm_type, NormType::LayerNorm | NormType::RmsNorm),
                "Invalid norm_type for '{}'",
                arch
            );
            assert!(
                matches!(
                    defaults.activation_type,
                    ActivationType::Silu | ActivationType::Relu2 | ActivationType::Gelu
                ),
                "Invalid activation_type for '{}'",
                arch
            );
        }
    }

    #[test]
    fn test_known_architectures_have_unique_lowercase_forms() {
        let mut seen = std::collections::HashSet::new();
        for arch in ArchitectureRegistry::known_architectures() {
            let lower = arch.to_lowercase();
            assert!(
                seen.insert(lower.clone()),
                "Duplicate lowercase form '{}' from arch '{}'",
                lower,
                arch
            );
        }
    }

    #[test]
    fn test_numeric_and_empty_variations() {
        assert!(ArchitectureRegistry::lookup("123").is_none());
        assert!(ArchitectureRegistry::lookup("0").is_none());
        assert!(ArchitectureRegistry::lookup("3.5").is_none());
        assert!(ArchitectureRegistry::lookup("phi4").is_none()); // no dash
    }
}
