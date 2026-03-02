//! Vocabulary analysis utilities for tokenizer diagnostics.

use std::collections::HashMap;

/// Statistics about a token vocabulary.
#[derive(Debug, Clone)]
pub struct VocabStats {
    pub total_tokens: usize,
    pub special_tokens: usize,
    pub byte_tokens: usize,
    pub single_char_tokens: usize,
    pub max_token_len: usize,
    pub avg_token_len: f64,
}

impl VocabStats {
    pub fn analyze(tokens: &[String]) -> Self {
        let total = tokens.len();
        if total == 0 {
            return Self {
                total_tokens: 0,
                special_tokens: 0,
                byte_tokens: 0,
                single_char_tokens: 0,
                max_token_len: 0,
                avg_token_len: 0.0,
            };
        }
        let mut special = 0;
        let mut byte_tok = 0;
        let mut single_char = 0;
        let mut max_len = 0;
        let mut total_len: usize = 0;
        for t in tokens {
            let len = t.len();
            total_len += len;
            if len > max_len {
                max_len = len;
            }
            if is_byte_token(t) {
                byte_tok += 1;
            } else if is_special_token(t) {
                special += 1;
            } else if t.chars().count() == 1 {
                single_char += 1;
            }
        }
        Self {
            total_tokens: total,
            special_tokens: special,
            byte_tokens: byte_tok,
            single_char_tokens: single_char,
            max_token_len: max_len,
            avg_token_len: total_len as f64 / total as f64,
        }
    }

    pub fn content_ratio(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        (self.total_tokens - self.special_tokens) as f64 / self.total_tokens as f64
    }
}

pub fn is_special_token(token: &str) -> bool {
    (token.starts_with('<') && token.ends_with('>'))
        || (token.starts_with('[') && token.ends_with(']'))
        || token.starts_with("<|")
        || token.ends_with("|>")
}

pub fn is_byte_token(token: &str) -> bool {
    token.starts_with("<0x") && token.ends_with('>') && token.len() <= 6
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerType {
    Bpe,
    SentencePiece,
    WordPiece,
    Unknown,
}

impl std::fmt::Display for TokenizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bpe => write!(f, "BPE"),
            Self::SentencePiece => write!(f, "SentencePiece"),
            Self::WordPiece => write!(f, "WordPiece"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

pub fn detect_tokenizer_type(tokens: &[String]) -> TokenizerType {
    let mut has_byte = false;
    let mut has_wp = false;
    let mut has_sp = false;
    for t in tokens.iter().take(1000) {
        if is_byte_token(t) {
            has_byte = true;
        }
        if t.starts_with("##") {
            has_wp = true;
        }
        if t.starts_with('\u{2581}') {
            has_sp = true;
        }
    }
    if has_wp {
        TokenizerType::WordPiece
    } else if has_sp {
        TokenizerType::SentencePiece
    } else if has_byte {
        TokenizerType::Bpe
    } else {
        TokenizerType::Unknown
    }
}

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
            "a".into(),
        ]
    }

    #[test]
    fn test_vocab_stats() {
        let stats = VocabStats::analyze(&sample_vocab());
        assert_eq!(stats.total_tokens, 8);
        assert_eq!(stats.special_tokens, 3);
        assert_eq!(stats.byte_tokens, 2);
    }

    #[test]
    fn test_empty() {
        let stats = VocabStats::analyze(&[]);
        assert_eq!(stats.total_tokens, 0);
    }

    #[test]
    fn test_special_token() {
        assert!(is_special_token("<s>"));
        assert!(is_special_token("[CLS]"));
        assert!(is_special_token("<|endoftext|>"));
        assert!(!is_special_token("hello"));
    }

    #[test]
    fn test_byte_token() {
        assert!(is_byte_token("<0x41>"));
        assert!(!is_byte_token("<s>"));
    }

    #[test]
    fn test_detect_bpe() {
        let t = vec!["<0x41>".into(), "hi".into()];
        assert_eq!(detect_tokenizer_type(&t), TokenizerType::Bpe);
    }

    #[test]
    fn test_detect_wordpiece() {
        let t = vec!["hello".into(), "##ing".into()];
        assert_eq!(detect_tokenizer_type(&t), TokenizerType::WordPiece);
    }

    #[test]
    fn test_detect_sp() {
        let t = vec!["\u{2581}hello".into()];
        assert_eq!(detect_tokenizer_type(&t), TokenizerType::SentencePiece);
    }

    #[test]
    fn test_detect_unknown() {
        let t = vec!["hello".into()];
        assert_eq!(detect_tokenizer_type(&t), TokenizerType::Unknown);
    }

    #[test]
    fn test_content_ratio() {
        let stats = VocabStats::analyze(&sample_vocab());
        assert!(stats.content_ratio() > 0.5);
    }

    #[test]
    fn test_length_dist() {
        let t = vec!["a".into(), "bb".into(), "cc".into()];
        let d = length_distribution(&t);
        assert_eq!(d[&1], 1);
        assert_eq!(d[&2], 2);
    }

    #[test]
    fn test_type_display() {
        assert_eq!(format!("{}", TokenizerType::Bpe), "BPE");
    }

    #[test]
    fn test_single_char() {
        let t = vec!["a".into(), "b".into(), "cd".into()];
        let stats = VocabStats::analyze(&t);
        assert_eq!(stats.single_char_tokens, 2);
    }

    #[test]
    fn test_max_len() {
        let t = vec!["ab".into(), "abcdef".into()];
        let stats = VocabStats::analyze(&t);
        assert_eq!(stats.max_token_len, 6);
    }

    #[test]
    fn test_content_ratio_empty() {
        let stats = VocabStats::analyze(&[]);
        assert_eq!(stats.content_ratio(), 0.0);
    }
}
