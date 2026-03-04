//! Vocabulary encoding detection and utilities.
//!
//! Detect BPE/SentencePiece/TikToken encoding formats, analyze
//! vocabulary characteristics, and provide encoding metadata.

/// Tokenizer encoding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingType {
    /// Byte-Pair Encoding (HuggingFace style).
    Bpe,
    /// SentencePiece (Unigram or BPE variant).
    SentencePiece,
    /// TikToken (OpenAI/Microsoft style).
    TikToken,
    /// WordPiece (BERT style).
    WordPiece,
    /// Character-level.
    CharLevel,
    /// Unknown encoding.
    Unknown,
}

impl EncodingType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bpe => "bpe",
            Self::SentencePiece => "sentencepiece",
            Self::TikToken => "tiktoken",
            Self::WordPiece => "wordpiece",
            Self::CharLevel => "char_level",
            Self::Unknown => "unknown",
        }
    }

    pub fn supports_byte_fallback(&self) -> bool {
        matches!(self, Self::Bpe | Self::TikToken | Self::SentencePiece)
    }
}

/// Vocabulary statistics for encoding analysis.
#[derive(Debug, Clone)]
pub struct VocabProfile {
    pub encoding: EncodingType,
    pub vocab_size: usize,
    pub has_byte_tokens: bool,
    pub has_special_tokens: bool,
    pub avg_token_length: f64,
    pub max_token_length: usize,
    pub num_merges: usize,
}

impl VocabProfile {
    pub fn compression_ratio_estimate(&self) -> f64 {
        if self.avg_token_length == 0.0 {
            return 1.0;
        }
        self.avg_token_length
    }
}

/// Detect encoding type from vocabulary tokens.
pub fn detect_encoding(tokens: &[&str]) -> EncodingType {
    if tokens.is_empty() {
        return EncodingType::Unknown;
    }

    let has_sentencepiece_prefix = tokens.iter().any(|t| t.starts_with('\u{2581}')); // ▁
    let has_wordpiece_prefix = tokens.iter().any(|t| t.starts_with("##"));
    let has_byte_tokens = tokens.iter().any(|t| t.starts_with("<0x") && t.ends_with('>'));
    let has_gpt_byte = tokens.iter().any(|t| t.starts_with("bytes:"));
    let max_len = tokens.iter().map(|t| t.len()).max().unwrap_or(0);

    if has_gpt_byte {
        return EncodingType::TikToken;
    }
    if has_sentencepiece_prefix {
        return EncodingType::SentencePiece;
    }
    if has_wordpiece_prefix {
        return EncodingType::WordPiece;
    }
    if has_byte_tokens {
        return EncodingType::Bpe;
    }
    if max_len <= 1 {
        return EncodingType::CharLevel;
    }

    EncodingType::Bpe // default assumption
}

/// Analyze vocabulary to produce a profile.
pub fn analyze_vocab(tokens: &[&str], num_merges: usize) -> VocabProfile {
    let encoding = detect_encoding(tokens);
    let vocab_size = tokens.len();
    let has_byte_tokens = tokens.iter().any(|t| t.starts_with("<0x") || t.starts_with("bytes:"));
    let has_special_tokens = tokens.iter().any(|t| {
        (t.starts_with('<') && t.ends_with('>')) || (t.starts_with("<|") && t.ends_with("|>"))
    });

    let total_len: usize = tokens.iter().map(|t| t.len()).sum();
    let avg_token_length = if vocab_size == 0 { 0.0 } else { total_len as f64 / vocab_size as f64 };
    let max_token_length = tokens.iter().map(|t| t.len()).max().unwrap_or(0);

    VocabProfile {
        encoding,
        vocab_size,
        has_byte_tokens,
        has_special_tokens,
        avg_token_length,
        max_token_length,
        num_merges,
    }
}

/// Common vocabulary sizes for known models.
pub fn expected_vocab_size(model: &str) -> Option<usize> {
    let lower = model.to_lowercase();
    if lower.contains("phi-4") || lower.contains("phi4") {
        Some(100352)
    } else if lower.contains("llama-3") || lower.contains("llama3") {
        Some(128256)
    } else if lower.contains("bitnet") {
        Some(32000)
    } else if lower.contains("gpt2") {
        Some(50257)
    } else if lower.contains("qwen") {
        Some(151936)
    } else {
        None
    }
}

/// Check if a token looks like a special/control token.
pub fn is_special_token(token: &str) -> bool {
    (token.starts_with('<') && token.ends_with('>'))
        || (token.starts_with("<|") && token.ends_with("|>"))
        || (token.starts_with("[") && token.ends_with("]") && token.to_uppercase() == token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_bpe() {
        let tokens = vec!["hello", "world", "<0xFF>", "test"];
        assert_eq!(detect_encoding(&tokens), EncodingType::Bpe);
    }

    #[test]
    fn test_detect_sentencepiece() {
        let tokens = vec!["\u{2581}hello", "\u{2581}world", "test"];
        assert_eq!(detect_encoding(&tokens), EncodingType::SentencePiece);
    }

    #[test]
    fn test_detect_wordpiece() {
        let tokens = vec!["hello", "##ing", "##ed"];
        assert_eq!(detect_encoding(&tokens), EncodingType::WordPiece);
    }

    #[test]
    fn test_detect_tiktoken() {
        let tokens = vec!["hello", "bytes:0xFF"];
        assert_eq!(detect_encoding(&tokens), EncodingType::TikToken);
    }

    #[test]
    fn test_detect_char_level() {
        let tokens = vec!["a", "b", "c", "d"];
        assert_eq!(detect_encoding(&tokens), EncodingType::CharLevel);
    }

    #[test]
    fn test_detect_empty() {
        assert_eq!(detect_encoding(&[]), EncodingType::Unknown);
    }

    #[test]
    fn test_analyze_vocab() {
        let tokens = vec!["hello", "world", "<pad>", "<eos>"];
        let profile = analyze_vocab(&tokens, 1000);
        assert_eq!(profile.vocab_size, 4);
        assert!(profile.has_special_tokens);
        assert_eq!(profile.num_merges, 1000);
    }

    #[test]
    fn test_expected_vocab_sizes() {
        assert_eq!(expected_vocab_size("microsoft/phi-4"), Some(100352));
        assert_eq!(expected_vocab_size("meta-llama3"), Some(128256));
        assert_eq!(expected_vocab_size("bitnet-b1.58"), Some(32000));
        assert_eq!(expected_vocab_size("unknown"), None);
    }

    #[test]
    fn test_is_special_token() {
        assert!(is_special_token("<pad>"));
        assert!(is_special_token("<|endoftext|>"));
        assert!(is_special_token("[CLS]"));
        assert!(!is_special_token("hello"));
    }

    #[test]
    fn test_byte_fallback() {
        assert!(EncodingType::Bpe.supports_byte_fallback());
        assert!(EncodingType::TikToken.supports_byte_fallback());
        assert!(!EncodingType::WordPiece.supports_byte_fallback());
    }

    #[test]
    fn test_compression_ratio() {
        let tokens = vec!["hello", "world"];
        let profile = analyze_vocab(&tokens, 0);
        assert!(profile.compression_ratio_estimate() > 1.0);
    }

    #[test]
    fn test_encoding_as_str() {
        assert_eq!(EncodingType::Bpe.as_str(), "bpe");
        assert_eq!(EncodingType::TikToken.as_str(), "tiktoken");
    }
}
