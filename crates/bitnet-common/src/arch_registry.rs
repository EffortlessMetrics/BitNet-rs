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
            "phi" | "phi-4" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(16384),
            }),
            "phi-3" | "phi3" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "llama" | "mistral" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: None,
            }),
            "llama-3.1" | "llama3.1" | "llama31" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(131072),
            }),
            "mistral-nemo" | "nemo" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(128000),
            }),
            "llama2" | "llama-2" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "qwen" | "qwen2" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: None,
            }),
            "qwen2.5" | "qwen-2.5" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(32768),
            }),
            "codegemma" | "code-gemma" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Gelu,
                default_context_length: Some(8192),
            }),
            "gemma" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Gelu,
                default_context_length: None,
            }),
            "gemma2" | "gemma-2" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Gelu,
                default_context_length: Some(8192),
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
            "deepseek-v3" | "deepseekv3" | "deepseek3" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(65536),
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
                default_context_length: Some(128000),
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
            "rwkv" | "rwkv5" | "rwkv6" => Some(ArchDefaults {
                norm_type: NormType::LayerNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "olmo" => Some(ArchDefaults {
                norm_type: NormType::LayerNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(2048),
            }),
            "olmo2" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "zephyr" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "vicuna" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "orca" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "solar" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "alpaca" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(2048),
            }),
            "nous-hermes" | "hermes" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "wizardlm" | "wizard" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "openchat" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(8192),
            }),
            "granite" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(8192),
            }),
            "nemotron" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "saiga" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(2048),
            }),
            "gpt" | "bert" => Some(ArchDefaults {
                norm_type: NormType::LayerNorm,
                activation_type: ActivationType::Gelu,
                default_context_length: None,
            }),
            "tinyllama" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(2048),
            }),
            "dolphin" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "chatgpt" | "gpt4" | "gpt-4" => Some(ArchDefaults {
                norm_type: NormType::LayerNorm,
                activation_type: ActivationType::Gelu,
                default_context_length: Some(8192),
            }),
            "mixtral" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(32768),
            }),
            "stablelm" | "stable-lm" | "stablecode" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "bloom" | "bloomz" => Some(ArchDefaults {
                norm_type: NormType::LayerNorm,
                activation_type: ActivationType::Gelu,
                default_context_length: Some(2048),
            }),
            "jamba" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(256000),
            }),
            "persimmon" | "adept" => Some(ArchDefaults {
                norm_type: NormType::LayerNorm,
                activation_type: ActivationType::Gelu,
                default_context_length: Some(16384),
            }),
            "xverse" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(8192),
            }),
            "arctic" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "dbrx" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(32768),
            }),
            "exaone" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
            }),
            "minicpm" => Some(ArchDefaults {
                norm_type: NormType::RmsNorm,
                activation_type: ActivationType::Silu,
                default_context_length: Some(4096),
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
            "phi3",
            "llama",
            "llama2",
            "llama-2",
            "llama-3.1",
            "llama3.1",
            "llama31",
            "mistral",
            "mistral-nemo",
            "nemo",
            "qwen",
            "qwen2",
            "qwen2.5",
            "qwen-2.5",
            "gemma",
            "gemma2",
            "gemma-2",
            "codegemma",
            "code-gemma",
            "deepseek",
            "deepseek2",
            "deepseek-v3",
            "deepseekv3",
            "deepseek3",
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
            "rwkv",
            "rwkv5",
            "rwkv6",
            "olmo",
            "olmo2",
            "bitnet",
            "bitnet-b1.58",
            "gpt",
            "bert",
            "tinyllama",
            "dolphin",
            "chatgpt",
            "gpt4",
            "gpt-4",
            "zephyr",
            "vicuna",
            "orca",
            "solar",
            "alpaca",
            "nous-hermes",
            "hermes",
            "wizardlm",
            "wizard",
            "openchat",
            "granite",
            "nemotron",
            "saiga",
            "mixtral",
            "stablelm",
            "stable-lm",
            "stablecode",
            "bloom",
            "bloomz",
            "jamba",
            "persimmon",
            "adept",
            "xverse",
            "arctic",
            "dbrx",
            "exaone",
            "minicpm",
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

    #[test]
    fn test_minimum_architecture_count() {
        // Regression guard: ensure nobody accidentally removes architectures
        let count = ArchitectureRegistry::known_architectures().len();
        assert!(
            count >= 40,
            "Architecture count dropped to {}! Expected at least 40. \
             Did you accidentally remove entries from the ARCHS array?",
            count
        );
    }

    #[test]
    fn test_core_families_always_present() {
        // These are the core families that must never be removed
        let required = [
            "bitnet", "llama", "phi", "qwen", "gemma", "mistral",
            "deepseek", "starcoder", "falcon", "gpt", "mpt",
        ];
        for family in &required {
            assert!(
                ArchitectureRegistry::is_known(family),
                "Core family '{}' is missing from architecture registry!",
                family
            );
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // Arbitrary strings never panic on lookup
    proptest! {
        #[test]
        fn lookup_never_panics(s in "\\PC{0,200}") {
            let _ = ArchitectureRegistry::lookup(&s);
            let _ = ArchitectureRegistry::is_known(&s);
        }
    }

    // All known architectures always return Some on lookup
    proptest! {
        #[test]
        fn known_archs_always_found(
            idx in 0usize..ArchitectureRegistry::known_architectures().len()
        ) {
            let arch = ArchitectureRegistry::known_architectures()[idx];
            prop_assert!(
                ArchitectureRegistry::lookup(arch).is_some(),
                "known arch '{}' returned None on lookup",
                arch
            );
            prop_assert!(
                ArchitectureRegistry::is_known(arch),
                "known arch '{}' returned false on is_known",
                arch
            );
        }
    }

    // Case-insensitive: uppercase version of a known arch should also match
    proptest! {
        #[test]
        fn case_insensitive_lookup(
            idx in 0usize..ArchitectureRegistry::known_architectures().len()
        ) {
            let arch = ArchitectureRegistry::known_architectures()[idx];
            let upper = arch.to_uppercase();
            prop_assert!(
                ArchitectureRegistry::lookup(&upper).is_some(),
                "uppercase '{}' of arch '{}' not found",
                upper,
                arch
            );
        }
    }

    // Random ASCII strings that don't match any known prefix should return None
    proptest! {
        #[test]
        fn random_ascii_mostly_none(s in "[!-/:-@\\[-`{-~]{1,50}") {
            // Strings of only punctuation should never match an architecture
            prop_assert!(
                ArchitectureRegistry::lookup(&s).is_none(),
                "punctuation-only '{}' should not match",
                s
            );
        }
    }

    // lookup and is_known must agree
    proptest! {
        #[test]
        fn lookup_and_is_known_agree(s in "\\PC{0,100}") {
            let found = ArchitectureRegistry::lookup(&s).is_some();
            let known = ArchitectureRegistry::is_known(&s);
            prop_assert_eq!(
                found, known,
                "lookup().is_some() and is_known() disagree for '{}'",
                s
            );
        }
    }
}
