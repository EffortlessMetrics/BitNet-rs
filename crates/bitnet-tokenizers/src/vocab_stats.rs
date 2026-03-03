//! Vocabulary statistics and analysis.
//!
//! Analyze tokenizer vocabulary: token length distribution,
//! special token detection, script coverage, merge statistics.

use std::collections::HashMap;

/// Token category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenCategory {
    /// Regular word/subword token.
    Word,
    /// Punctuation or symbol.
    Punctuation,
    /// Special token (BOS, EOS, PAD, etc.).
    Special,
    /// Whitespace or space-prefixed.
    Whitespace,
    /// Byte fallback token (e.g., <0xFF>).
    ByteFallback,
    /// Unknown category.
    Unknown,
}

impl TokenCategory {
    pub fn name(&self) -> &'static str {
        match self {
            TokenCategory::Word => "word",
            TokenCategory::Punctuation => "punctuation",
            TokenCategory::Special => "special",
            TokenCategory::Whitespace => "whitespace",
            TokenCategory::ByteFallback => "byte_fallback",
            TokenCategory::Unknown => "unknown",
        }
    }
}

/// Classify a token string into a category.
pub fn classify_token(token: &str) -> TokenCategory {
    if token.is_empty() {
        return TokenCategory::Unknown;
    }
    // Special tokens
    if (token.starts_with('<') && token.ends_with('>'))
        || (token.starts_with("[") && token.ends_with("]"))
        || token.starts_with("<|")
    {
        if token.starts_with("<0x") || token.starts_with("<0X") {
            return TokenCategory::ByteFallback;
        }
        return TokenCategory::Special;
    }
    // Whitespace
    if token.chars().all(|c| c.is_whitespace()) || token.starts_with('\u{2581}') {
        return TokenCategory::Whitespace;
    }
    // Punctuation
    if token.chars().all(|c| c.is_ascii_punctuation()) {
        return TokenCategory::Punctuation;
    }
    TokenCategory::Word
}

/// Statistics about a vocabulary.
#[derive(Debug)]
pub struct VocabStats {
    pub vocab_size: usize,
    pub category_counts: HashMap<TokenCategory, usize>,
    pub avg_token_length: f64,
    pub max_token_length: usize,
    pub min_token_length: usize,
    pub length_histogram: HashMap<usize, usize>,
}

impl VocabStats {
    /// Compute stats from a list of token strings.
    pub fn compute(tokens: &[&str]) -> Self {
        let n = tokens.len();
        if n == 0 {
            return Self {
                vocab_size: 0,
                category_counts: HashMap::new(),
                avg_token_length: 0.0,
                max_token_length: 0,
                min_token_length: 0,
                length_histogram: HashMap::new(),
            };
        }

        let mut category_counts: HashMap<TokenCategory, usize> = HashMap::new();
        let mut total_len = 0usize;
        let mut max_len = 0usize;
        let mut min_len = usize::MAX;
        let mut length_histogram: HashMap<usize, usize> = HashMap::new();

        for &token in tokens {
            let cat = classify_token(token);
            *category_counts.entry(cat).or_insert(0) += 1;
            let len = token.len();
            total_len += len;
            max_len = max_len.max(len);
            min_len = min_len.min(len);
            *length_histogram.entry(len).or_insert(0) += 1;
        }

        Self {
            vocab_size: n,
            category_counts,
            avg_token_length: total_len as f64 / n as f64,
            max_token_length: max_len,
            min_token_length: min_len,
            length_histogram,
        }
    }

    pub fn special_token_count(&self) -> usize {
        self.category_counts.get(&TokenCategory::Special).copied().unwrap_or(0)
    }

    pub fn word_token_count(&self) -> usize {
        self.category_counts.get(&TokenCategory::Word).copied().unwrap_or(0)
    }

    pub fn byte_fallback_count(&self) -> usize {
        self.category_counts.get(&TokenCategory::ByteFallback).copied().unwrap_or(0)
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        format!(
            "vocab={}, avg_len={:.1}, special={}, words={}, byte_fb={}",
            self.vocab_size,
            self.avg_token_length,
            self.special_token_count(),
            self.word_token_count(),
            self.byte_fallback_count(),
        )
    }
}

/// Detect common special tokens in a vocabulary.
pub fn find_special_tokens(tokens: &[&str]) -> Vec<(usize, String)> {
    let special_patterns = [
        "<s>",
        "</s>",
        "<unk>",
        "<pad>",
        "<mask>",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "[UNK]",
        "[MASK]",
    ];
    let mut found = Vec::new();
    for (i, &token) in tokens.iter().enumerate() {
        if special_patterns.contains(&token) {
            found.push((i, token.to_string()));
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_word() {
        assert_eq!(classify_token("hello"), TokenCategory::Word);
    }

    #[test]
    fn test_classify_special() {
        assert_eq!(classify_token("<s>"), TokenCategory::Special);
        assert_eq!(classify_token("[CLS]"), TokenCategory::Special);
        assert_eq!(classify_token("<|endoftext|>"), TokenCategory::Special);
    }

    #[test]
    fn test_classify_byte_fallback() {
        assert_eq!(classify_token("<0xFF>"), TokenCategory::ByteFallback);
    }

    #[test]
    fn test_classify_punctuation() {
        assert_eq!(classify_token("..."), TokenCategory::Punctuation);
        assert_eq!(classify_token(","), TokenCategory::Punctuation);
    }

    #[test]
    fn test_classify_whitespace() {
        assert_eq!(classify_token("\u{2581}hello"), TokenCategory::Whitespace);
    }

    #[test]
    fn test_vocab_stats() {
        let tokens = vec!["hello", "world", "<s>", "</s>", ",", "the"];
        let stats = VocabStats::compute(&tokens);
        assert_eq!(stats.vocab_size, 6);
        assert_eq!(stats.special_token_count(), 2);
        assert!(stats.word_token_count() >= 3);
    }

    #[test]
    fn test_empty_vocab() {
        let stats = VocabStats::compute(&[]);
        assert_eq!(stats.vocab_size, 0);
        assert_eq!(stats.avg_token_length, 0.0);
    }

    #[test]
    fn test_length_histogram() {
        let tokens = vec!["a", "bb", "ccc", "dd"];
        let stats = VocabStats::compute(&tokens);
        assert_eq!(stats.length_histogram[&1], 1);
        assert_eq!(stats.length_histogram[&2], 2);
        assert_eq!(stats.length_histogram[&3], 1);
    }

    #[test]
    fn test_find_special_tokens() {
        let tokens = vec!["hello", "<s>", "world", "</s>", "<pad>"];
        let found = find_special_tokens(&tokens);
        assert_eq!(found.len(), 3);
        assert_eq!(found[0].1, "<s>");
    }

    #[test]
    fn test_summary() {
        let tokens = vec!["hello", "<s>"];
        let stats = VocabStats::compute(&tokens);
        let s = stats.summary();
        assert!(s.contains("vocab=2"));
    }

    #[test]
    fn test_category_name() {
        assert_eq!(TokenCategory::Word.name(), "word");
        assert_eq!(TokenCategory::ByteFallback.name(), "byte_fallback");
    }

    #[test]
    fn test_min_max_length() {
        let tokens = vec!["a", "hello", "hi"];
        let stats = VocabStats::compute(&tokens);
        assert_eq!(stats.min_token_length, 1);
        assert_eq!(stats.max_token_length, 5);
    }
}
