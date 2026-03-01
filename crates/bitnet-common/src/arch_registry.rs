//! Centralized registry of known model architectures and their default
//! configurations.
//!
//! This module maps human-readable architecture names (e.g. `"llama"`,
//! `"phi-4"`, `"gemma2"`) to sensible defaults for normalization type,
//! activation function, and context length.  Lookups are case-insensitive.

use crate::config::{ActivationType, NormType};

/// Default configuration parameters for a model architecture.
#[derive(Debug, Clone)]
pub struct ArchDefaults {
    /// Normalization layer variant (RmsNorm or LayerNorm).
    pub norm_type: NormType,
    /// Activation function variant (Silu or Gelu).
    pub activation_type: ActivationType,
    /// Default context length, if one is widely agreed upon.
    pub default_context_length: Option<usize>,
}

/// Static registry that maps architecture name strings to [`ArchDefaults`].
pub struct ArchitectureRegistry;

impl ArchitectureRegistry {
    /// Look up the default configuration for `architecture`.
    ///
    /// The match is **case-insensitive**.  Returns `None` for unrecognised
    /// architecture strings.
    pub fn lookup(architecture: &str) -> Option<ArchDefaults> {
        use ActivationType::{Gelu, Silu};
        use NormType::{LayerNorm, RmsNorm};

        let (norm, act, ctx) = match architecture.to_lowercase().as_str() {
            "phi" | "phi-4" => (RmsNorm, Silu, Some(16384)),
            "phi-3" | "phi3" => (RmsNorm, Silu, Some(4096)),
            "phi-2" | "phi2" => (LayerNorm, Gelu, Some(2048)),

            "llama" | "mistral" => (RmsNorm, Silu, None),
            "llama-3.2" | "llama3.2" | "llama32" => (RmsNorm, Silu, Some(131072)),
            "llama-3.1" | "llama3.1" | "llama31" => (RmsNorm, Silu, Some(131072)),
            "mistral-nemo" | "nemo" => (RmsNorm, Silu, Some(128000)),
            "llama2" | "llama-2" => (RmsNorm, Silu, Some(4096)),

            "qwen" | "qwen2" => (RmsNorm, Silu, None),
            "qwen2.5" | "qwen-2.5" => (RmsNorm, Silu, Some(32768)),

            "codegemma" | "code-gemma" => (RmsNorm, Gelu, Some(8192)),
            "gemma" => (RmsNorm, Gelu, None),
            "gemma2" | "gemma-2" => (RmsNorm, Gelu, Some(8192)),

            "bitnet" | "bitnet-b1.58" => (LayerNorm, Silu, None),

            "deepseek" | "deepseek2" => (RmsNorm, Silu, None),
            "deepseek-v3" | "deepseekv3" | "deepseek3" => (RmsNorm, Silu, Some(65536)),

            "starcoder" | "starcoder2" => (LayerNorm, Gelu, None),
            "falcon" => (LayerNorm, Gelu, None),
            "falcon-2" | "falcon2" => (RmsNorm, Silu, Some(8192)),

            "codellama" | "code-llama" => (RmsNorm, Silu, None),

            "command" | "command-r" | "command-r-plus" | "cohere" => {
                (LayerNorm, Silu, Some(128000))
            }

            "aya" => (RmsNorm, Silu, Some(8192)),
            "smollm" | "smol-lm" => (RmsNorm, Silu, Some(2048)),

            "internlm" | "internlm2" => (RmsNorm, Silu, None),
            "yi" | "yi-1.5" => (RmsNorm, Silu, None),
            "baichuan" | "baichuan2" => (RmsNorm, Silu, None),
            "chatglm" | "chatglm2" | "chatglm3" | "glm-4" => (RmsNorm, Silu, None),

            "mpt" => (LayerNorm, Gelu, None),

            "rwkv" | "rwkv5" | "rwkv6" => (LayerNorm, Silu, Some(4096)),
            "olmo" => (LayerNorm, Silu, Some(2048)),
            "olmo2" | "olmo-2" => (RmsNorm, Silu, Some(4096)),

            "zephyr" => (RmsNorm, Silu, Some(4096)),
            "vicuna" => (RmsNorm, Silu, Some(4096)),
            "orca" => (RmsNorm, Silu, Some(4096)),
            "solar" => (RmsNorm, Silu, Some(4096)),
            "alpaca" => (RmsNorm, Silu, Some(2048)),
            "nous-hermes" | "hermes" => (RmsNorm, Silu, Some(4096)),
            "wizardlm" | "wizard" => (RmsNorm, Silu, Some(4096)),
            "openchat" => (RmsNorm, Silu, Some(8192)),
            "granite" => (RmsNorm, Silu, Some(8192)),
            "nemotron" => (RmsNorm, Silu, Some(4096)),
            "saiga" => (RmsNorm, Silu, Some(2048)),

            "gpt" | "bert" => (LayerNorm, Gelu, None),
            "tinyllama" => (RmsNorm, Silu, Some(2048)),
            "dolphin" => (RmsNorm, Silu, Some(4096)),
            "chatgpt" | "gpt4" | "gpt-4" => (LayerNorm, Gelu, Some(8192)),

            "mixtral" => (RmsNorm, Silu, Some(32768)),
            "stablelm" | "stable-lm" | "stablecode" => (RmsNorm, Silu, Some(4096)),
            "bloom" | "bloomz" => (LayerNorm, Gelu, Some(2048)),
            "jamba" => (RmsNorm, Silu, Some(256000)),
            "persimmon" | "adept" => (LayerNorm, Gelu, Some(16384)),
            "xverse" => (RmsNorm, Silu, Some(8192)),
            "arctic" => (RmsNorm, Silu, Some(4096)),
            "dbrx" => (RmsNorm, Silu, Some(32768)),
            "exaone" => (RmsNorm, Silu, Some(4096)),
            "minicpm" => (RmsNorm, Silu, Some(4096)),

            _ => return None,
        };

        Some(ArchDefaults { norm_type: norm, activation_type: act, default_context_length: ctx })
    }

    /// All recognised architecture name strings (lower-case canonical forms).
    pub fn known_architectures() -> &'static [&'static str] {
        &[
            "phi",
            "phi-4",
            "phi-3",
            "phi3",
            "phi-2",
            "phi2",
            "llama",
            "llama2",
            "llama-2",
            "llama-3.2",
            "llama3.2",
            "llama32",
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
            "falcon-2",
            "falcon2",
            "codellama",
            "code-llama",
            "command",
            "command-r",
            "command-r-plus",
            "cohere",
            "aya",
            "smollm",
            "smol-lm",
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
            "olmo-2",
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

    /// Returns `true` when `architecture` is recognised by the registry.
    pub fn is_known(architecture: &str) -> bool {
        Self::lookup(architecture).is_some()
    }
}

// -- Tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ActivationType, NormType};
    use std::collections::HashSet;

    #[test]
    fn test_all_known_return_some() {
        for arch in ArchitectureRegistry::known_architectures() {
            assert!(
                ArchitectureRegistry::lookup(arch).is_some(),
                "known architecture '{}' returned None",
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
        for name in &["PHI", "Llama", "BitNet-B1.58", "GEMMA2", "Qwen2.5"] {
            assert!(
                ArchitectureRegistry::lookup(name).is_some(),
                "case-insensitive lookup failed for '{}'",
                name,
            );
        }
    }

    #[test]
    fn test_known_architectures_consistent_with_lookup() {
        for arch in ArchitectureRegistry::known_architectures() {
            assert!(
                ArchitectureRegistry::lookup(arch).is_some(),
                "'{}' is in known_architectures but lookup returns None",
                arch,
            );
        }
    }

    #[test]
    fn test_phi_defaults() {
        let d = ArchitectureRegistry::lookup("phi").unwrap();
        assert_eq!(d.norm_type, NormType::RmsNorm);
        assert_eq!(d.activation_type, ActivationType::Silu);
        assert_eq!(d.default_context_length, Some(16384));
    }

    #[test]
    fn test_phi2_defaults() {
        let d = ArchitectureRegistry::lookup("phi-2").unwrap();
        assert_eq!(d.norm_type, NormType::LayerNorm);
        assert_eq!(d.activation_type, ActivationType::Gelu);
        assert_eq!(d.default_context_length, Some(2048));
    }

    #[test]
    fn test_aya_defaults() {
        let d = ArchitectureRegistry::lookup("aya").unwrap();
        assert_eq!(d.norm_type, NormType::RmsNorm);
        assert_eq!(d.activation_type, ActivationType::Silu);
        assert_eq!(d.default_context_length, Some(8192));
    }

    #[test]
    fn test_smollm_defaults() {
        let d = ArchitectureRegistry::lookup("smollm").unwrap();
        assert_eq!(d.norm_type, NormType::RmsNorm);
        assert_eq!(d.activation_type, ActivationType::Silu);
        assert_eq!(d.default_context_length, Some(2048));
    }

    #[test]
    fn test_bitnet_defaults() {
        let d = ArchitectureRegistry::lookup("bitnet").unwrap();
        assert_eq!(d.norm_type, NormType::LayerNorm);
        assert_eq!(d.activation_type, ActivationType::Silu);
        assert_eq!(d.default_context_length, None);

        let d2 = ArchitectureRegistry::lookup("bitnet-b1.58").unwrap();
        assert_eq!(d2.norm_type, NormType::LayerNorm);
        assert_eq!(d2.activation_type, ActivationType::Silu);
        assert_eq!(d2.default_context_length, None);
    }

    #[test]
    fn test_gemma_defaults() {
        let d = ArchitectureRegistry::lookup("gemma").unwrap();
        assert_eq!(d.norm_type, NormType::RmsNorm);
        assert_eq!(d.activation_type, ActivationType::Gelu);
        assert_eq!(d.default_context_length, None);
    }

    #[test]
    fn test_gpt_defaults() {
        let d = ArchitectureRegistry::lookup("gpt").unwrap();
        assert_eq!(d.norm_type, NormType::LayerNorm);
        assert_eq!(d.activation_type, ActivationType::Gelu);
        assert_eq!(d.default_context_length, None);
    }

    #[test]
    fn test_is_known() {
        assert!(ArchitectureRegistry::is_known("llama"));
        assert!(ArchitectureRegistry::is_known("PHI"));
        assert!(ArchitectureRegistry::is_known("gemma2"));
        assert!(!ArchitectureRegistry::is_known("unknown"));
        assert!(!ArchitectureRegistry::is_known(""));
        assert!(!ArchitectureRegistry::is_known("mamba"));
    }

    #[test]
    fn test_codellama_defaults() {
        let d = ArchitectureRegistry::lookup("codellama").unwrap();
        assert_eq!(d.norm_type, NormType::RmsNorm);
        assert_eq!(d.activation_type, ActivationType::Silu);
        assert_eq!(d.default_context_length, None);

        let d2 = ArchitectureRegistry::lookup("code-llama").unwrap();
        assert_eq!(d2.norm_type, d.norm_type);
    }

    #[test]
    fn test_cohere_defaults() {
        let d = ArchitectureRegistry::lookup("cohere").unwrap();
        assert_eq!(d.norm_type, NormType::LayerNorm);
        assert_eq!(d.activation_type, ActivationType::Silu);
        assert_eq!(d.default_context_length, Some(128000));
    }

    #[test]
    fn test_internlm_defaults() {
        let d = ArchitectureRegistry::lookup("internlm").unwrap();
        assert_eq!(d.norm_type, NormType::RmsNorm);
        assert_eq!(d.activation_type, ActivationType::Silu);
        assert_eq!(d.default_context_length, None);
    }

    #[test]
    fn test_yi_defaults() {
        let d = ArchitectureRegistry::lookup("yi").unwrap();
        assert_eq!(d.norm_type, NormType::RmsNorm);
        assert_eq!(d.activation_type, ActivationType::Silu);
        assert_eq!(d.default_context_length, None);
    }

    #[test]
    fn test_chatglm_defaults() {
        let d = ArchitectureRegistry::lookup("chatglm").unwrap();
        assert_eq!(d.norm_type, NormType::RmsNorm);
        assert_eq!(d.activation_type, ActivationType::Silu);
        assert_eq!(d.default_context_length, None);

        assert!(ArchitectureRegistry::lookup("glm-4").is_some());
    }

    #[test]
    fn test_mpt_defaults() {
        let d = ArchitectureRegistry::lookup("mpt").unwrap();
        assert_eq!(d.norm_type, NormType::LayerNorm);
        assert_eq!(d.activation_type, ActivationType::Gelu);
        assert_eq!(d.default_context_length, None);
    }

    #[test]
    fn test_very_long_string_returns_none() {
        let long = "a".repeat(10_000);
        assert!(ArchitectureRegistry::lookup(&long).is_none());
    }

    #[test]
    fn test_unicode_strings_return_none() {
        assert!(ArchitectureRegistry::lookup("\u{65E5}\u{672C}\u{8A9E}").is_none());
        assert!(ArchitectureRegistry::lookup("\u{043B}\u{043B}\u{0430}\u{043C}\u{0430}").is_none());
        assert!(ArchitectureRegistry::lookup("\u{03C6}-4").is_none());
        assert!(ArchitectureRegistry::lookup("\u{1F999}").is_none());
    }

    #[test]
    fn test_whitespace_variants_return_none() {
        assert!(ArchitectureRegistry::lookup(" phi").is_none());
        assert!(ArchitectureRegistry::lookup("phi ").is_none());
        assert!(ArchitectureRegistry::lookup("lla ma").is_none());
        assert!(ArchitectureRegistry::lookup("\tgemma").is_none());
        assert!(ArchitectureRegistry::lookup("\n").is_none());
    }

    #[test]
    fn test_special_character_variants_return_none() {
        assert!(ArchitectureRegistry::lookup("llama!").is_none());
        assert!(ArchitectureRegistry::lookup("phi@4").is_none());
        assert!(ArchitectureRegistry::lookup("#gemma").is_none());
        assert!(ArchitectureRegistry::lookup("qwen$2").is_none());
    }

    #[test]
    fn test_all_known_architectures_have_valid_defaults() {
        for arch in ArchitectureRegistry::known_architectures() {
            let d = ArchitectureRegistry::lookup(arch).unwrap();
            if let Some(ctx) = d.default_context_length {
                assert!(ctx > 0, "context length for '{}' must be > 0", arch);
            }
        }
    }

    #[test]
    fn test_known_architectures_have_unique_lowercase_forms() {
        let mut seen = HashSet::new();
        for arch in ArchitectureRegistry::known_architectures() {
            let lc = arch.to_lowercase();
            assert!(
                seen.insert(lc.clone()),
                "duplicate lowercase form '{}' in known_architectures",
                lc,
            );
        }
    }

    #[test]
    fn test_numeric_and_empty_variations() {
        assert!(ArchitectureRegistry::lookup("0").is_none());
        assert!(ArchitectureRegistry::lookup("123").is_none());
        assert!(ArchitectureRegistry::lookup("1.5").is_none());
        assert!(ArchitectureRegistry::lookup("").is_none());
    }

    #[test]
    fn test_minimum_architecture_count() {
        assert!(
            ArchitectureRegistry::known_architectures().len() >= 40,
            "expected at least 40 known architectures, got {}",
            ArchitectureRegistry::known_architectures().len(),
        );
    }

    #[test]
    fn test_core_families_always_present() {
        let core = [
            "llama", "mistral", "phi", "gemma", "qwen", "gpt", "bert", "falcon", "bitnet",
            "deepseek",
        ];
        for family in &core {
            assert!(
                ArchitectureRegistry::is_known(family),
                "core family '{}' missing from registry",
                family,
            );
        }
    }
}

// -- Property-style tests ------------------------------------------------

#[cfg(test)]
mod property_tests {
    use super::*;

    #[test]
    fn lookup_never_panics() {
        let inputs = [
            "",
            " ",
            "\0",
            "null",
            &"a".repeat(100_000),
            "phi\x00phi",
            "LLAMA",
            "\u{1F999}\u{1F999}\u{1F999}",
        ];
        for input in &inputs {
            let _ = ArchitectureRegistry::lookup(input);
        }
    }

    #[test]
    fn known_archs_always_found() {
        for arch in ArchitectureRegistry::known_architectures() {
            assert!(
                ArchitectureRegistry::lookup(arch).is_some(),
                "known arch '{}' not found via lookup",
                arch,
            );
        }
    }

    #[test]
    fn case_insensitive_lookup() {
        for arch in ArchitectureRegistry::known_architectures() {
            let upper = arch.to_uppercase();
            assert!(
                ArchitectureRegistry::lookup(&upper).is_some(),
                "upper-case '{}' should match '{}'",
                upper,
                arch,
            );
        }
    }

    #[test]
    fn random_ascii_mostly_none() {
        let puncts = ["!!!", "???", "@@@", "###", "$$$", "%%%"];
        for p in &puncts {
            assert!(ArchitectureRegistry::lookup(p).is_none());
        }
    }

    #[test]
    fn lookup_and_is_known_agree() {
        let mut samples: Vec<String> =
            ArchitectureRegistry::known_architectures().iter().map(|s| s.to_string()).collect();
        samples.extend(["unknown", "", "mamba", "\u{1F999}"].iter().map(|s| s.to_string()));

        for s in &samples {
            assert_eq!(
                ArchitectureRegistry::lookup(s).is_some(),
                ArchitectureRegistry::is_known(s),
                "lookup and is_known disagree for '{}'",
                s,
            );
        }
    }
}
