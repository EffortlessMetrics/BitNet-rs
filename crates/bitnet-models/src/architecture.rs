//! Typed architecture registry for multi-SLM support.
//!
//! Provides a [`ModelArchitecture`] enum that maps model family names (from
//! GGUF `general.architecture`, HuggingFace `model_type`, or model repo
//! names) to rich per-architecture defaults such as RoPE base frequency,
//! vocabulary size, and typical hidden dimension.
//!
//! This complements the lower-level string-based
//! [`bitnet_common::ArchitectureRegistry`] by adding a typed enum and richer
//! configuration metadata.

use bitnet_common::config::{ActivationType, NormType};

// ---------------------------------------------------------------------------
// ModelArchitecture enum
// ---------------------------------------------------------------------------

/// Known model architecture families.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    /// BitNet 1-bit / 1.58-bit (microsoft/BitNet-b1.58)
    BitNet,
    /// Phi family (Phi-1 / Phi-2 / Phi-3 / Phi-4)
    Phi,
    /// Qwen family (Qwen / Qwen2 / Qwen2.5)
    Qwen,
    /// Gemma family (Gemma / Gemma-2)
    Gemma,
    /// Mistral / Mixtral
    Mistral,
    /// LLaMA family (LLaMA / LLaMA-2 / LLaMA-3)
    Llama,
    /// SmolLM
    SmolLM,
    /// Falcon family (Falcon / Falcon-2)
    Falcon,
    /// MPT (MosaicML)
    Mpt,
    /// BLOOM / BLOOMZ
    Bloom,
    /// StableLM / StableCode
    StableLM,
    /// TinyLlama
    TinyLlama,
    /// DeepSeek family
    DeepSeek,
    /// CodeLlama
    CodeLlama,
    /// StarCoder family
    StarCoder,
    /// Cohere Command-R family
    Cohere,
    /// InternLM family
    InternLM,
    /// Yi family
    Yi,
    /// ChatGLM / GLM-4
    ChatGLM,
    /// Unrecognised architecture (carries the raw string).
    Unknown(String),
}

/// Dense Qwen families supported by the SLM CPU adapter path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenseQwenFamily {
    Qwen2,
    Qwen3,
}

/// Dense Qwen architecture classification for strict preflight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseQwenArchitecture {
    Supported(DenseQwenFamily),
    UnsupportedHybrid { architecture: String, reason: &'static str },
    NotQwen,
}

/// Classify a GGUF/HF architecture string for the dense Qwen adapter.
///
/// The dense CPU lane accepts only plain Qwen2/Qwen3 transformer families.
/// Qwen3.5-style hybrid architectures are intentionally rejected before tensor
/// loading because they need linear-attention / state-space operators outside
/// the current dense adapter contract.
pub fn classify_dense_qwen_architecture(architecture: &str) -> DenseQwenArchitecture {
    let normalized = architecture.trim().to_ascii_lowercase().replace(['-', '.', ' '], "_");

    if matches!(
        normalized.as_str(),
        "qwen35" | "qwen3_5" | "qwen3_5_text" | "qwen_3_5" | "qwen_3_5_text"
    ) {
        return DenseQwenArchitecture::UnsupportedHybrid {
            architecture: architecture.to_string(),
            reason: "Qwen3.5 hybrid linear-attention/vision models are outside the dense CPU lane",
        };
    }

    match normalized.as_str() {
        "qwen2" | "qwen2_5" | "qwen_2" | "qwen_2_5" => {
            DenseQwenArchitecture::Supported(DenseQwenFamily::Qwen2)
        }
        "qwen3" | "qwen_3" => DenseQwenArchitecture::Supported(DenseQwenFamily::Qwen3),
        value if value.starts_with("qwen3_5") || value.starts_with("qwen35") => {
            DenseQwenArchitecture::UnsupportedHybrid {
                architecture: architecture.to_string(),
                reason: "Qwen3.5 hybrid linear-attention/vision models are outside the dense CPU lane",
            }
        }
        value if value.starts_with("qwen2") || value.starts_with("qwen_2") => {
            DenseQwenArchitecture::Supported(DenseQwenFamily::Qwen2)
        }
        value if value.starts_with("qwen3") || value.starts_with("qwen_3") => {
            DenseQwenArchitecture::Supported(DenseQwenFamily::Qwen3)
        }
        value if value.contains("qwen") => DenseQwenArchitecture::UnsupportedHybrid {
            architecture: architecture.to_string(),
            reason: "unrecognized Qwen architecture requires explicit dense adapter support",
        },
        _ => DenseQwenArchitecture::NotQwen,
    }
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BitNet => write!(f, "bitnet"),
            Self::Phi => write!(f, "phi"),
            Self::Qwen => write!(f, "qwen"),
            Self::Gemma => write!(f, "gemma"),
            Self::Mistral => write!(f, "mistral"),
            Self::Llama => write!(f, "llama"),
            Self::SmolLM => write!(f, "smollm"),
            Self::Falcon => write!(f, "falcon"),
            Self::Mpt => write!(f, "mpt"),
            Self::Bloom => write!(f, "bloom"),
            Self::StableLM => write!(f, "stablelm"),
            Self::TinyLlama => write!(f, "tinyllama"),
            Self::DeepSeek => write!(f, "deepseek"),
            Self::CodeLlama => write!(f, "codellama"),
            Self::StarCoder => write!(f, "starcoder"),
            Self::Cohere => write!(f, "cohere"),
            Self::InternLM => write!(f, "internlm"),
            Self::Yi => write!(f, "yi"),
            Self::ChatGLM => write!(f, "chatglm"),
            Self::Unknown(s) => write!(f, "{s}"),
        }
    }
}

// ---------------------------------------------------------------------------
// ArchitectureConfig
// ---------------------------------------------------------------------------

/// Rich default configuration for a model architecture family.
///
/// Values represent the *typical* or *canonical* configuration for the
/// architecture family.  Actual models may override any of these via their
/// own metadata.
#[derive(Debug, Clone)]
pub struct ArchitectureConfig {
    pub architecture: ModelArchitecture,
    pub activation: ActivationType,
    pub normalization: NormType,
    pub rope_base: f32,
    pub max_context: usize,
    pub vocab_size: usize,
    pub typical_hidden_size: usize,
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// Detect the [`ModelArchitecture`] from a model name, repo slug, or GGUF
/// architecture string.
///
/// Detection is **case-insensitive** and matches on sub-strings where
/// appropriate (e.g. `"microsoft/phi-4"` → [`ModelArchitecture::Phi`]).
pub fn detect_architecture(model_name: &str) -> ModelArchitecture {
    let lower = model_name.to_lowercase();

    // Order matters: check more specific patterns before generic ones.
    if lower.contains("bitnet") {
        return ModelArchitecture::BitNet;
    }
    if lower.contains("tinyllama") {
        return ModelArchitecture::TinyLlama;
    }
    if lower.contains("codellama") || lower.contains("code-llama") || lower.contains("code_llama") {
        return ModelArchitecture::CodeLlama;
    }
    if lower.contains("starcoder") {
        return ModelArchitecture::StarCoder;
    }
    if lower.contains("smollm") || lower.contains("smol-lm") || lower.contains("smol_lm") {
        return ModelArchitecture::SmolLM;
    }
    if lower.contains("stablelm") || lower.contains("stable-lm") || lower.contains("stablecode") {
        return ModelArchitecture::StableLM;
    }
    if lower.contains("deepseek") {
        return ModelArchitecture::DeepSeek;
    }
    if lower.contains("chatglm") || lower.contains("glm-4") || lower.contains("glm4") {
        return ModelArchitecture::ChatGLM;
    }
    if lower.contains("internlm") {
        return ModelArchitecture::InternLM;
    }
    if lower.contains("command-r") || lower.contains("command_r") || lower.contains("cohere") {
        return ModelArchitecture::Cohere;
    }
    // Phi must come before general checks for substrings like "phi3"
    if lower.contains("phi") {
        return ModelArchitecture::Phi;
    }
    if lower.contains("qwen") {
        return ModelArchitecture::Qwen;
    }
    if lower.contains("gemma") {
        return ModelArchitecture::Gemma;
    }
    if lower.contains("mixtral") || lower.contains("mistral") {
        return ModelArchitecture::Mistral;
    }
    if lower.contains("falcon") {
        return ModelArchitecture::Falcon;
    }
    if lower.contains("bloom") {
        return ModelArchitecture::Bloom;
    }
    if lower.contains("mpt") {
        return ModelArchitecture::Mpt;
    }
    if lower.starts_with("yi") || lower.contains("/yi") || lower.contains("yi-") {
        return ModelArchitecture::Yi;
    }
    // LLaMA is the most common fallback for GGUF `general.architecture`
    if lower.contains("llama") {
        return ModelArchitecture::Llama;
    }

    ModelArchitecture::Unknown(model_name.to_string())
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

/// Return the canonical default configuration for `arch`.
///
/// For [`ModelArchitecture::Unknown`] a conservative LLaMA-like default is
/// returned.
pub fn get_defaults(arch: &ModelArchitecture) -> ArchitectureConfig {
    match arch {
        ModelArchitecture::BitNet => ArchitectureConfig {
            architecture: ModelArchitecture::BitNet,
            activation: ActivationType::Relu2,
            normalization: NormType::LayerNorm,
            rope_base: 10_000.0,
            max_context: 4096,
            vocab_size: 32_000,
            typical_hidden_size: 2560,
        },
        ModelArchitecture::Phi => ArchitectureConfig {
            architecture: ModelArchitecture::Phi,
            activation: ActivationType::Silu,
            normalization: NormType::RmsNorm,
            rope_base: 10_000.0,
            max_context: 16_384,
            vocab_size: 100_352,
            typical_hidden_size: 5120,
        },
        ModelArchitecture::Qwen => ArchitectureConfig {
            architecture: ModelArchitecture::Qwen,
            activation: ActivationType::Silu,
            normalization: NormType::RmsNorm,
            rope_base: 1_000_000.0,
            max_context: 131_072,
            vocab_size: 152_064,
            typical_hidden_size: 3584,
        },
        ModelArchitecture::Gemma => ArchitectureConfig {
            architecture: ModelArchitecture::Gemma,
            activation: ActivationType::Gelu,
            normalization: NormType::RmsNorm,
            rope_base: 10_000.0,
            max_context: 8192,
            vocab_size: 256_000,
            typical_hidden_size: 3072,
        },
        ModelArchitecture::Mistral => ArchitectureConfig {
            architecture: ModelArchitecture::Mistral,
            activation: ActivationType::Silu,
            normalization: NormType::RmsNorm,
            rope_base: 10_000.0,
            max_context: 32_768,
            vocab_size: 32_000,
            typical_hidden_size: 4096,
        },
        ModelArchitecture::Llama => ArchitectureConfig {
            architecture: ModelArchitecture::Llama,
            activation: ActivationType::Silu,
            normalization: NormType::RmsNorm,
            rope_base: 500_000.0,
            max_context: 8192,
            vocab_size: 128_256,
            typical_hidden_size: 4096,
        },
        ModelArchitecture::SmolLM => ArchitectureConfig {
            architecture: ModelArchitecture::SmolLM,
            activation: ActivationType::Silu,
            normalization: NormType::RmsNorm,
            rope_base: 10_000.0,
            max_context: 2048,
            vocab_size: 49_152,
            typical_hidden_size: 576,
        },
        ModelArchitecture::Falcon => ArchitectureConfig {
            architecture: ModelArchitecture::Falcon,
            activation: ActivationType::Gelu,
            normalization: NormType::LayerNorm,
            rope_base: 10_000.0,
            max_context: 2048,
            vocab_size: 65_024,
            typical_hidden_size: 4544,
        },
        ModelArchitecture::Mpt => ArchitectureConfig {
            architecture: ModelArchitecture::Mpt,
            activation: ActivationType::Gelu,
            normalization: NormType::LayerNorm,
            rope_base: 10_000.0,
            max_context: 2048,
            vocab_size: 50_432,
            typical_hidden_size: 4096,
        },
        ModelArchitecture::Bloom => ArchitectureConfig {
            architecture: ModelArchitecture::Bloom,
            activation: ActivationType::Gelu,
            normalization: NormType::LayerNorm,
            rope_base: 10_000.0,
            max_context: 2048,
            vocab_size: 250_680,
            typical_hidden_size: 2048,
        },
        ModelArchitecture::StableLM => ArchitectureConfig {
            architecture: ModelArchitecture::StableLM,
            activation: ActivationType::Silu,
            normalization: NormType::RmsNorm,
            rope_base: 10_000.0,
            max_context: 4096,
            vocab_size: 100_352,
            typical_hidden_size: 2048,
        },
        ModelArchitecture::TinyLlama => ArchitectureConfig {
            architecture: ModelArchitecture::TinyLlama,
            activation: ActivationType::Silu,
            normalization: NormType::RmsNorm,
            rope_base: 10_000.0,
            max_context: 2048,
            vocab_size: 32_000,
            typical_hidden_size: 2048,
        },
        ModelArchitecture::DeepSeek => ArchitectureConfig {
            architecture: ModelArchitecture::DeepSeek,
            activation: ActivationType::Silu,
            normalization: NormType::RmsNorm,
            rope_base: 10_000.0,
            max_context: 65_536,
            vocab_size: 102_400,
            typical_hidden_size: 4096,
        },
        ModelArchitecture::CodeLlama => ArchitectureConfig {
            architecture: ModelArchitecture::CodeLlama,
            activation: ActivationType::Silu,
            normalization: NormType::RmsNorm,
            rope_base: 1_000_000.0,
            max_context: 16_384,
            vocab_size: 32_016,
            typical_hidden_size: 4096,
        },
        ModelArchitecture::StarCoder => ArchitectureConfig {
            architecture: ModelArchitecture::StarCoder,
            activation: ActivationType::Gelu,
            normalization: NormType::LayerNorm,
            rope_base: 10_000.0,
            max_context: 8192,
            vocab_size: 49_152,
            typical_hidden_size: 6144,
        },
        ModelArchitecture::Cohere => ArchitectureConfig {
            architecture: ModelArchitecture::Cohere,
            activation: ActivationType::Silu,
            normalization: NormType::LayerNorm,
            rope_base: 10_000.0,
            max_context: 128_000,
            vocab_size: 256_000,
            typical_hidden_size: 8192,
        },
        ModelArchitecture::InternLM => ArchitectureConfig {
            architecture: ModelArchitecture::InternLM,
            activation: ActivationType::Silu,
            normalization: NormType::RmsNorm,
            rope_base: 10_000.0,
            max_context: 32_768,
            vocab_size: 103_168,
            typical_hidden_size: 4096,
        },
        ModelArchitecture::Yi => ArchitectureConfig {
            architecture: ModelArchitecture::Yi,
            activation: ActivationType::Silu,
            normalization: NormType::RmsNorm,
            rope_base: 5_000_000.0,
            max_context: 4096,
            vocab_size: 64_000,
            typical_hidden_size: 4096,
        },
        ModelArchitecture::ChatGLM => ArchitectureConfig {
            architecture: ModelArchitecture::ChatGLM,
            activation: ActivationType::Silu,
            normalization: NormType::RmsNorm,
            rope_base: 10_000.0,
            max_context: 131_072,
            vocab_size: 151_552,
            typical_hidden_size: 4096,
        },
        ModelArchitecture::Unknown(_) => ArchitectureConfig {
            architecture: arch.clone(),
            activation: ActivationType::Silu,
            normalization: NormType::RmsNorm,
            rope_base: 10_000.0,
            max_context: 2048,
            vocab_size: 32_000,
            typical_hidden_size: 4096,
        },
    }
}

/// All known (non-Unknown) architecture variants.
pub fn supported_architectures() -> Vec<ModelArchitecture> {
    vec![
        ModelArchitecture::BitNet,
        ModelArchitecture::Phi,
        ModelArchitecture::Qwen,
        ModelArchitecture::Gemma,
        ModelArchitecture::Mistral,
        ModelArchitecture::Llama,
        ModelArchitecture::SmolLM,
        ModelArchitecture::Falcon,
        ModelArchitecture::Mpt,
        ModelArchitecture::Bloom,
        ModelArchitecture::StableLM,
        ModelArchitecture::TinyLlama,
        ModelArchitecture::DeepSeek,
        ModelArchitecture::CodeLlama,
        ModelArchitecture::StarCoder,
        ModelArchitecture::Cohere,
        ModelArchitecture::InternLM,
        ModelArchitecture::Yi,
        ModelArchitecture::ChatGLM,
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Detection from model names / repo slugs --

    #[test]
    fn detect_from_model_repo_names() {
        assert_eq!(detect_architecture("microsoft/phi-4"), ModelArchitecture::Phi);
        assert_eq!(detect_architecture("Qwen/Qwen2.5-7B"), ModelArchitecture::Qwen);
        assert_eq!(detect_architecture("google/gemma-2-9b"), ModelArchitecture::Gemma);
        assert_eq!(detect_architecture("mistralai/Mistral-7B-v0.1"), ModelArchitecture::Mistral);
        assert_eq!(detect_architecture("meta-llama/Llama-3-8B"), ModelArchitecture::Llama);
        assert_eq!(detect_architecture("microsoft/BitNet-b1.58-2B-4T"), ModelArchitecture::BitNet,);
    }

    #[test]
    fn detect_from_gguf_architecture_strings() {
        assert_eq!(detect_architecture("phi3"), ModelArchitecture::Phi);
        assert_eq!(detect_architecture("phi"), ModelArchitecture::Phi);
        assert_eq!(detect_architecture("llama"), ModelArchitecture::Llama);
        assert_eq!(detect_architecture("mistral"), ModelArchitecture::Mistral);
        assert_eq!(detect_architecture("qwen2"), ModelArchitecture::Qwen);
        assert_eq!(detect_architecture("qwen3"), ModelArchitecture::Qwen);
        assert_eq!(detect_architecture("gemma"), ModelArchitecture::Gemma);
        assert_eq!(detect_architecture("bitnet"), ModelArchitecture::BitNet);
        assert_eq!(detect_architecture("falcon"), ModelArchitecture::Falcon);
        assert_eq!(detect_architecture("mpt"), ModelArchitecture::Mpt);
        assert_eq!(detect_architecture("bloom"), ModelArchitecture::Bloom);
    }

    #[test]
    fn detect_case_insensitive() {
        assert_eq!(detect_architecture("PHI"), ModelArchitecture::Phi);
        assert_eq!(detect_architecture("LLAMA"), ModelArchitecture::Llama);
        assert_eq!(detect_architecture("BitNet"), ModelArchitecture::BitNet);
        assert_eq!(detect_architecture("Gemma2"), ModelArchitecture::Gemma);
    }

    #[test]
    fn detect_specific_before_generic() {
        // TinyLlama contains "llama" but should match TinyLlama
        assert_eq!(detect_architecture("tinyllama"), ModelArchitecture::TinyLlama);
        // CodeLlama contains "llama" but should match CodeLlama
        assert_eq!(detect_architecture("codellama"), ModelArchitecture::CodeLlama);
        // Mixtral contains "mistral" substring
        assert_eq!(detect_architecture("mixtral"), ModelArchitecture::Mistral);
    }

    #[test]
    fn detect_stable_lm_variants() {
        assert_eq!(detect_architecture("stablelm"), ModelArchitecture::StableLM);
        assert_eq!(detect_architecture("stable-lm"), ModelArchitecture::StableLM);
        assert_eq!(detect_architecture("stablecode"), ModelArchitecture::StableLM);
    }

    #[test]
    fn detect_unknown() {
        let arch = detect_architecture("some_unknown_model");
        assert!(matches!(arch, ModelArchitecture::Unknown(_)));
        if let ModelArchitecture::Unknown(name) = arch {
            assert_eq!(name, "some_unknown_model");
        }
    }

    #[test]
    fn detect_empty_string() {
        assert!(matches!(detect_architecture(""), ModelArchitecture::Unknown(_)));
    }

    #[test]
    fn dense_qwen_classifies_qwen2_and_qwen3() {
        assert_eq!(
            classify_dense_qwen_architecture("qwen2"),
            DenseQwenArchitecture::Supported(DenseQwenFamily::Qwen2)
        );
        assert_eq!(
            classify_dense_qwen_architecture("qwen2.5"),
            DenseQwenArchitecture::Supported(DenseQwenFamily::Qwen2)
        );
        assert_eq!(
            classify_dense_qwen_architecture("qwen3"),
            DenseQwenArchitecture::Supported(DenseQwenFamily::Qwen3)
        );
    }

    #[test]
    fn dense_qwen_rejects_qwen35_hybrid() {
        for arch in ["qwen35", "qwen3_5", "qwen3_5_text", "qwen-3.5"] {
            let result = classify_dense_qwen_architecture(arch);
            assert!(
                matches!(result, DenseQwenArchitecture::UnsupportedHybrid { .. }),
                "{arch} should be rejected by the dense Qwen adapter"
            );
        }
    }

    #[test]
    fn dense_qwen_ignores_non_qwen_architecture() {
        assert_eq!(classify_dense_qwen_architecture("llama"), DenseQwenArchitecture::NotQwen);
    }

    // -- Default config correctness --

    #[test]
    fn defaults_bitnet() {
        let cfg = get_defaults(&ModelArchitecture::BitNet);
        assert_eq!(cfg.activation, ActivationType::Relu2);
        assert_eq!(cfg.normalization, NormType::LayerNorm);
        assert_eq!(cfg.rope_base, 10_000.0);
        assert_eq!(cfg.max_context, 4096);
        assert_eq!(cfg.vocab_size, 32_000);
    }

    #[test]
    fn defaults_phi() {
        let cfg = get_defaults(&ModelArchitecture::Phi);
        assert_eq!(cfg.activation, ActivationType::Silu);
        assert_eq!(cfg.normalization, NormType::RmsNorm);
        assert_eq!(cfg.rope_base, 10_000.0);
        assert_eq!(cfg.max_context, 16_384);
        assert_eq!(cfg.vocab_size, 100_352);
        assert_eq!(cfg.typical_hidden_size, 5120);
    }

    #[test]
    fn defaults_qwen() {
        let cfg = get_defaults(&ModelArchitecture::Qwen);
        assert_eq!(cfg.activation, ActivationType::Silu);
        assert_eq!(cfg.normalization, NormType::RmsNorm);
        assert_eq!(cfg.rope_base, 1_000_000.0);
        assert_eq!(cfg.max_context, 131_072);
        assert_eq!(cfg.vocab_size, 152_064);
    }

    #[test]
    fn defaults_gemma() {
        let cfg = get_defaults(&ModelArchitecture::Gemma);
        assert_eq!(cfg.activation, ActivationType::Gelu);
        assert_eq!(cfg.normalization, NormType::RmsNorm);
        assert_eq!(cfg.rope_base, 10_000.0);
        assert_eq!(cfg.max_context, 8192);
        assert_eq!(cfg.vocab_size, 256_000);
    }

    #[test]
    fn defaults_mistral() {
        let cfg = get_defaults(&ModelArchitecture::Mistral);
        assert_eq!(cfg.activation, ActivationType::Silu);
        assert_eq!(cfg.normalization, NormType::RmsNorm);
        assert_eq!(cfg.max_context, 32_768);
        assert_eq!(cfg.vocab_size, 32_000);
    }

    #[test]
    fn defaults_llama() {
        let cfg = get_defaults(&ModelArchitecture::Llama);
        assert_eq!(cfg.activation, ActivationType::Silu);
        assert_eq!(cfg.normalization, NormType::RmsNorm);
        assert_eq!(cfg.rope_base, 500_000.0);
        assert_eq!(cfg.max_context, 8192);
        assert_eq!(cfg.vocab_size, 128_256);
    }

    #[test]
    fn defaults_unknown_returns_conservative() {
        let cfg = get_defaults(&ModelArchitecture::Unknown("mystery".into()));
        assert_eq!(cfg.activation, ActivationType::Silu);
        assert_eq!(cfg.normalization, NormType::RmsNorm);
        assert_eq!(cfg.max_context, 2048);
    }

    // -- Supported architectures list --

    #[test]
    fn supported_list_complete() {
        let list = supported_architectures();
        // At least 19 known architectures
        assert!(list.len() >= 19, "expected ≥19, got {}", list.len());
        // No Unknown in the list
        assert!(
            !list.iter().any(|a| matches!(a, ModelArchitecture::Unknown(_))),
            "Unknown should not be in supported_architectures()",
        );
    }

    #[test]
    fn every_supported_has_defaults() {
        for arch in supported_architectures() {
            let cfg = get_defaults(&arch);
            assert_eq!(cfg.architecture, arch);
            assert!(cfg.max_context > 0);
            assert!(cfg.vocab_size > 0);
            assert!(cfg.typical_hidden_size > 0);
        }
    }

    #[test]
    fn display_roundtrip() {
        for arch in supported_architectures() {
            let name = arch.to_string();
            assert!(!name.is_empty(), "display should not be empty for {arch:?}");
        }
    }

    // -- Detection of additional families --

    #[test]
    fn detect_deepseek() {
        assert_eq!(detect_architecture("deepseek-v3"), ModelArchitecture::DeepSeek);
        assert_eq!(detect_architecture("deepseek2"), ModelArchitecture::DeepSeek);
    }

    #[test]
    fn detect_chatglm() {
        assert_eq!(detect_architecture("chatglm3"), ModelArchitecture::ChatGLM);
        assert_eq!(detect_architecture("glm-4"), ModelArchitecture::ChatGLM);
    }

    #[test]
    fn detect_cohere() {
        assert_eq!(detect_architecture("command-r-plus"), ModelArchitecture::Cohere);
        assert_eq!(detect_architecture("cohere"), ModelArchitecture::Cohere);
    }

    #[test]
    fn detect_smollm() {
        assert_eq!(detect_architecture("smollm"), ModelArchitecture::SmolLM);
        assert_eq!(detect_architecture("smol-lm"), ModelArchitecture::SmolLM);
    }

    #[test]
    fn detect_internlm() {
        assert_eq!(detect_architecture("internlm2"), ModelArchitecture::InternLM);
    }

    #[test]
    fn detect_starcoder() {
        assert_eq!(detect_architecture("starcoder2"), ModelArchitecture::StarCoder);
    }
}
