//! Vocabulary analysis utilities for tokenizer diagnostics.

use std::collections::HashMap;

/// Statistics about a token vocabulary.
#[derive(Debug, Clone)]
pub struct VocabStats {
    pub total_tokens: usize,
    pub special_tokens: usize,
    pub byte_tokens: usize,
    pub single_char_tokens: usize,
    pub multi_word_tokens: usize,
    pub max_token_len: usize,
    pub avg_token_len: f64,
}

impl VocabStats {
    /// Analyze a vocabulary (list of token strings).
    pub fn analyze(tokens: &[String]) -> Self {
        let total = tokens.len();
        if total == 0 {
            return Self {
                total_tokens: 0,
                special_tokens: 0,
                byte_tokens: 0,
                single_char_tokens: 0,
                multi_word_tokens: 0,
                max_token_len: 0,
                avg_token_len: 0.0,
            };
        }

        let mut special = 0;
        let mut byte_tok = 0;
        let mut single_char = 0;
        let mut multi_word = 0;
        let mut max_len = 0;
        let mut total_len: usize = 0;

        for t in tokens {
            let len = t.len();
            total_len += len;
            if len > max_len {
                max_len = len;
            }

            if is_special_token(t) {
                special += 1;
            } else if is_byte_token(t) {
                byte_tok += 1;
            } else if t.chars().count() == 1 {
                single_char += 1;
            }
            if t.contains(' ') && t.trim().contains(' ') {
                multi_word += 1;
            }
        }

        Self {
            total_tokens: total,
            special_tokens: special,
            byte_tokens: byte_tok,
            single_char_tokens: single_char,
            multi_word_tokens: multi_word,
            max_token_len: max_len,
            avg_token_len: total_len as f64 / total as f64,
        }
    }

    /// Coverage: non-special / total.
    pub fn content_ratio(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        (self.total_tokens - self.special_tokens) as f64 / self.total_tokens as f64
    }
}

/// Check if a token looks like a special token (excludes byte-level tokens).
pub fn is_special_token(token: &str) -> bool {
    if is_byte_token(token) {
        return false;
    }
    (token.starts_with('<') && token.ends_with('>'))
        || (token.starts_with('[') && token.ends_with(']'))
        || token.starts_with("<|")
        || token.ends_with("|>")
}

/// Check if a token is a byte-level token (e.g., `<0x41>`).
pub fn is_byte_token(token: &str) -> bool {
    token.starts_with("<0x") && token.ends_with('>') && token.len() <= 6
}

/// Estimate tokenizer type from vocabulary characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabTokenizerType {
    /// Byte-pair encoding (e.g., GPT-2, TikToken).
    Bpe,
    /// SentencePiece / Unigram.
    SentencePiece,
    /// WordPiece (BERT-style).
    WordPiece,
    /// Unknown type.
    Unknown,
}

impl std::fmt::Display for VocabTokenizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bpe => write!(f, "BPE"),
            Self::SentencePiece => write!(f, "SentencePiece"),
            Self::WordPiece => write!(f, "WordPiece"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Guess tokenizer type from tokens.
pub fn detect_tokenizer_type(tokens: &[String]) -> VocabTokenizerType {
    let mut has_byte_fallback = false;
    let mut has_wordpiece_prefix = false;
    let mut has_sp_prefix = false;

    for t in tokens.iter().take(1000) {
        if is_byte_token(t) {
            has_byte_fallback = true;
        }
        if t.starts_with("##") {
            has_wordpiece_prefix = true;
        }
        if t.starts_with('\u{2581}') {
            has_sp_prefix = true;
        }
    }

    if has_wordpiece_prefix {
        VocabTokenizerType::WordPiece
    } else if has_sp_prefix {
        VocabTokenizerType::SentencePiece
    } else if has_byte_fallback {
        VocabTokenizerType::Bpe
    } else {
        VocabTokenizerType::Unknown
    }
}

/// Find character coverage (which Unicode code points have dedicated tokens).
pub fn character_coverage(tokens: &[String]) -> f64 {
    let mut covered: std::collections::HashSet<char> = std::collections::HashSet::new();
    for t in tokens {
        if t.chars().count() == 1 {
            if let Some(c) = t.chars().next() {
                covered.insert(c);
            }
        }
    }
    // Basic ASCII range coverage
    let ascii_range = 128;
    let ascii_covered = covered.iter().filter(|c| c.is_ascii()).count();
    ascii_covered as f64 / ascii_range as f64
}

/// Count token length distribution (bucket by length).
pub fn length_distribution(tokens: &[String]) -> HashMap<usize, usize> {
    let mut dist = HashMap::new();
    for t in tokens {
        *dist.entry(t.len()).or_insert(0) += 1;
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vocab() -> Vec<String> {
        vec![
            "<s>".into(),
            "</s>".into(),
            "<unk>".into(),
            "<0x41>".into(),
            "<0x42>".into(),
            "hello".into(),
            "world".into(),
            " the".into(),
            "a".into(),
            "b".into(),
            "c".into(),
            "good morning".into(),
        ]
    }

    #[test]
    fn test_vocab_stats() {
        let tokens = sample_vocab();
        let stats = VocabStats::analyze(&tokens);
        assert_eq!(stats.total_tokens, 12);
        assert_eq!(stats.special_tokens, 3);
        assert_eq!(stats.byte_tokens, 2);
    }

    #[test]
    fn test_vocab_stats_empty() {
        let stats = VocabStats::analyze(&[]);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.avg_token_len, 0.0);
    }

    #[test]
    fn test_is_special_token() {
        assert!(is_special_token("<s>"));
        assert!(is_special_token("</s>"));
        assert!(is_special_token("[CLS]"));
        assert!(is_special_token("<|endoftext|>"));
        assert!(!is_special_token("hello"));
        assert!(!is_special_token("<0x41>"));
    }

    #[test]
    fn test_is_byte_token() {
        assert!(is_byte_token("<0x41>"));
        assert!(is_byte_token("<0xFF>"));
        assert!(!is_byte_token("<s>"));
        assert!(!is_byte_token("hello"));
    }

    #[test]
    fn test_detect_bpe() {
        let tokens = vec!["<0x41>".into(), "<0x42>".into(), "hello".into()];
        assert_eq!(detect_tokenizer_type(&tokens), VocabTokenizerType::Bpe);
    }

    #[test]
    fn test_detect_wordpiece() {
        let tokens = vec!["hello".into(), "##ing".into(), "##ed".into()];
        assert_eq!(detect_tokenizer_type(&tokens), VocabTokenizerType::WordPiece);
    }

    #[test]
    fn test_detect_sentencepiece() {
        let tokens = vec!["\u{2581}hello".into(), "\u{2581}world".into()];
        assert_eq!(detect_tokenizer_type(&tokens), VocabTokenizerType::SentencePiece);
    }

    #[test]
    fn test_detect_unknown() {
        let tokens = vec!["hello".into(), "world".into()];
        assert_eq!(detect_tokenizer_type(&tokens), VocabTokenizerType::Unknown);
    }

    #[test]
    fn test_content_ratio() {
        let tokens = sample_vocab();
        let stats = VocabStats::analyze(&tokens);
        let ratio = stats.content_ratio();
        assert!(ratio > 0.5); // 9/12 = 0.75
    }

    #[test]
    fn test_content_ratio_empty() {
        let stats = VocabStats::analyze(&[]);
        assert_eq!(stats.content_ratio(), 0.0);
    }

    #[test]
    fn test_character_coverage() {
        let tokens: Vec<String> = ('a'..='z').map(|c| c.to_string()).collect();
        let coverage = character_coverage(&tokens);
        assert!(coverage > 0.1); // 26/128 ~ 0.2
    }

    #[test]
    fn test_character_coverage_empty() {
        let coverage = character_coverage(&[]);
        assert_eq!(coverage, 0.0);
    }

    #[test]
    fn test_length_distribution() {
        let tokens = vec!["a".into(), "bb".into(), "cc".into(), "ddd".into()];
        let dist = length_distribution(&tokens);
        assert_eq!(dist[&1], 1);
        assert_eq!(dist[&2], 2);
        assert_eq!(dist[&3], 1);
    }

    #[test]
    fn test_tokenizer_type_display() {
        assert_eq!(format!("{}", VocabTokenizerType::Bpe), "BPE");
        assert_eq!(format!("{}", VocabTokenizerType::WordPiece), "WordPiece");
    }

    #[test]
    fn test_single_char_counting() {
        let tokens = vec!["a".into(), "b".into(), "cd".into()];
        let stats = VocabStats::analyze(&tokens);
        assert_eq!(stats.single_char_tokens, 2);
    }

    #[test]
    fn test_max_token_len() {
        let tokens = vec!["ab".into(), "abcdef".into(), "x".into()];
        let stats = VocabStats::analyze(&tokens);
        assert_eq!(stats.max_token_len, 6);
    }

    #[test]
    fn test_avg_token_len() {
        let tokens = vec!["ab".into(), "cd".into()];
        let stats = VocabStats::analyze(&tokens);
        assert!((stats.avg_token_len - 2.0).abs() < 0.01);
    }
}
