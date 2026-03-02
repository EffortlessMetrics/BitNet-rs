//! Tokenizer compatibility matrix for SLM model families.

/// Tokenizer type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerType {
    SentencePiece,
    Tiktoken,
    HuggingFaceBpe,
    ByteLevelBpe,
}

impl std::fmt::Display for TokenizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SentencePiece => write!(f, "SentencePiece"),
            Self::Tiktoken => write!(f, "Tiktoken"),
            Self::HuggingFaceBpe => write!(f, "HuggingFace BPE"),
            Self::ByteLevelBpe => write!(f, "Byte-Level BPE"),
        }
    }
}

/// Special token configuration.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub pad_token: Option<String>,
    pub unk_token: Option<String>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
}

impl SpecialTokens {
    pub fn new() -> Self {
        Self {
            bos_token: None,
            eos_token: None,
            pad_token: None,
            unk_token: None,
            bos_id: None,
            eos_id: None,
        }
    }
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self::new()
    }
}

/// Compatibility entry for a model family.
#[derive(Debug, Clone)]
pub struct CompatEntry {
    pub model_family: String,
    pub tokenizer_type: TokenizerType,
    pub vocab_size: usize,
    pub special_tokens: SpecialTokens,
    pub add_bos: bool,
    pub add_eos: bool,
    pub chat_template: Option<String>,
}

/// Get the compatibility matrix.
pub fn compatibility_matrix() -> Vec<CompatEntry> {
    vec![
        CompatEntry {
            model_family: "phi-4".to_string(),
            tokenizer_type: TokenizerType::Tiktoken,
            vocab_size: 100352,
            special_tokens: SpecialTokens {
                bos_token: Some("<|endoftext|>".to_string()),
                eos_token: Some("<|endoftext|>".to_string()),
                pad_token: Some("<|endoftext|>".to_string()),
                unk_token: None,
                bos_id: Some(100257),
                eos_id: Some(100257),
            },
            add_bos: true,
            add_eos: false,
            chat_template: Some("chatml".to_string()),
        },
        CompatEntry {
            model_family: "llama-3".to_string(),
            tokenizer_type: TokenizerType::HuggingFaceBpe,
            vocab_size: 128256,
            special_tokens: SpecialTokens {
                bos_token: Some("<|begin_of_text|>".to_string()),
                eos_token: Some("<|eot_id|>".to_string()),
                pad_token: None,
                unk_token: None,
                bos_id: Some(128000),
                eos_id: Some(128009),
            },
            add_bos: true,
            add_eos: false,
            chat_template: Some("llama3".to_string()),
        },
        CompatEntry {
            model_family: "qwen2.5".to_string(),
            tokenizer_type: TokenizerType::HuggingFaceBpe,
            vocab_size: 152064,
            special_tokens: SpecialTokens {
                bos_token: None,
                eos_token: Some("<|endoftext|>".to_string()),
                pad_token: Some("<|endoftext|>".to_string()),
                unk_token: None,
                bos_id: None,
                eos_id: Some(151643),
            },
            add_bos: false,
            add_eos: false,
            chat_template: Some("chatml".to_string()),
        },
        CompatEntry {
            model_family: "gemma-2".to_string(),
            tokenizer_type: TokenizerType::SentencePiece,
            vocab_size: 256000,
            special_tokens: SpecialTokens {
                bos_token: Some("<bos>".to_string()),
                eos_token: Some("<eos>".to_string()),
                pad_token: Some("<pad>".to_string()),
                unk_token: Some("<unk>".to_string()),
                bos_id: Some(2),
                eos_id: Some(1),
            },
            add_bos: true,
            add_eos: false,
            chat_template: Some("gemma".to_string()),
        },
        CompatEntry {
            model_family: "mistral-v0.3".to_string(),
            tokenizer_type: TokenizerType::SentencePiece,
            vocab_size: 32768,
            special_tokens: SpecialTokens {
                bos_token: Some("<s>".to_string()),
                eos_token: Some("</s>".to_string()),
                pad_token: None,
                unk_token: Some("<unk>".to_string()),
                bos_id: Some(1),
                eos_id: Some(2),
            },
            add_bos: true,
            add_eos: false,
            chat_template: Some("mistral".to_string()),
        },
        CompatEntry {
            model_family: "smollm2".to_string(),
            tokenizer_type: TokenizerType::HuggingFaceBpe,
            vocab_size: 49152,
            special_tokens: SpecialTokens {
                bos_token: Some("<|endoftext|>".to_string()),
                eos_token: Some("<|endoftext|>".to_string()),
                pad_token: None,
                unk_token: None,
                bos_id: Some(0),
                eos_id: Some(0),
            },
            add_bos: true,
            add_eos: false,
            chat_template: None,
        },
        CompatEntry {
            model_family: "bitnet".to_string(),
            tokenizer_type: TokenizerType::SentencePiece,
            vocab_size: 32000,
            special_tokens: SpecialTokens {
                bos_token: Some("<s>".to_string()),
                eos_token: Some("</s>".to_string()),
                pad_token: None,
                unk_token: Some("<unk>".to_string()),
                bos_id: Some(1),
                eos_id: Some(2),
            },
            add_bos: true,
            add_eos: false,
            chat_template: None,
        },
    ]
}

/// Look up compatibility entry for a model family.
pub fn lookup(model_family: &str) -> Option<CompatEntry> {
    let family = model_family.to_lowercase();
    compatibility_matrix().into_iter().find(|e| family.contains(&e.model_family.to_lowercase()))
}

/// Get all known model families.
pub fn known_families() -> Vec<String> {
    compatibility_matrix().into_iter().map(|e| e.model_family).collect()
}

/// Format the compatibility matrix as a table.
pub fn format_matrix() -> String {
    let mut out = String::from("Tokenizer Compatibility Matrix\n");
    out.push_str(&format!(
        "{:<15} {:<18} {:>8} {:<10} {:<10}\n",
        "Family", "Tokenizer", "Vocab", "BOS", "Template"
    ));
    out.push_str(&"-".repeat(70));
    out.push('\n');
    for entry in compatibility_matrix() {
        out.push_str(&format!(
            "{:<15} {:<18} {:>8} {:<10} {:<10}\n",
            entry.model_family,
            format!("{}", entry.tokenizer_type),
            entry.vocab_size,
            if entry.add_bos { "yes" } else { "no" },
            entry.chat_template.as_deref().unwrap_or("-"),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_not_empty() {
        let m = compatibility_matrix();
        assert!(m.len() >= 7);
    }

    #[test]
    fn test_lookup_phi4() {
        let e = lookup("phi-4").unwrap();
        assert_eq!(e.vocab_size, 100352);
        assert_eq!(e.tokenizer_type, TokenizerType::Tiktoken);
    }

    #[test]
    fn test_lookup_llama3() {
        let e = lookup("llama-3").unwrap();
        assert_eq!(e.vocab_size, 128256);
    }

    #[test]
    fn test_lookup_qwen() {
        let e = lookup("qwen2.5").unwrap();
        assert_eq!(e.vocab_size, 152064);
    }

    #[test]
    fn test_lookup_gemma() {
        let e = lookup("gemma-2").unwrap();
        assert_eq!(e.tokenizer_type, TokenizerType::SentencePiece);
    }

    #[test]
    fn test_lookup_mistral() {
        let e = lookup("mistral-v0.3").unwrap();
        assert_eq!(e.vocab_size, 32768);
    }

    #[test]
    fn test_lookup_missing() {
        assert!(lookup("nonexistent_model").is_none());
    }

    #[test]
    fn test_lookup_case_insensitive() {
        assert!(lookup("PHI-4").is_some());
        assert!(lookup("Llama-3").is_some());
    }

    #[test]
    fn test_known_families() {
        let families = known_families();
        assert!(families.len() >= 7);
        assert!(families.contains(&"phi-4".to_string()));
    }

    #[test]
    fn test_special_tokens_default() {
        let st = SpecialTokens::default();
        assert!(st.bos_token.is_none());
        assert!(st.eos_token.is_none());
    }

    #[test]
    fn test_phi4_special_tokens() {
        let e = lookup("phi-4").unwrap();
        assert_eq!(e.special_tokens.bos_id, Some(100257));
    }

    #[test]
    fn test_llama3_eos() {
        let e = lookup("llama-3").unwrap();
        assert_eq!(e.special_tokens.eos_token, Some("<|eot_id|>".to_string()));
    }

    #[test]
    fn test_tokenizer_type_display() {
        assert_eq!(format!("{}", TokenizerType::Tiktoken), "Tiktoken");
        assert_eq!(format!("{}", TokenizerType::SentencePiece), "SentencePiece");
    }

    #[test]
    fn test_format_matrix() {
        let out = format_matrix();
        assert!(out.contains("Compatibility Matrix"));
        assert!(out.contains("phi-4"));
        assert!(out.contains("llama-3"));
    }

    #[test]
    fn test_all_have_vocab() {
        for e in compatibility_matrix() {
            assert!(e.vocab_size > 0, "{} has 0 vocab", e.model_family);
        }
    }

    #[test]
    fn test_chat_templates_present() {
        let m = compatibility_matrix();
        let with_template = m.iter().filter(|e| e.chat_template.is_some()).count();
        assert!(with_template >= 5);
    }

    #[test]
    fn test_bitnet_entry() {
        let e = lookup("bitnet").unwrap();
        assert_eq!(e.vocab_size, 32000);
        assert_eq!(e.tokenizer_type, TokenizerType::SentencePiece);
    }
}
