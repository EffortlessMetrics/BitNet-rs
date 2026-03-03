//! Vocabulary analyzer.
//!
//! Analyze tokenizer vocabularies: coverage, overlap, special tokens.

use std::collections::{HashMap, HashSet};

/// Vocabulary statistics.
#[derive(Debug, Clone)]
pub struct VocabStats {
    pub total_tokens: usize,
    pub special_count: usize,
    pub byte_tokens: usize,
    pub single_char: usize,
    pub multi_char: usize,
    pub max_token_len: usize,
    pub avg_token_len: f64,
}

/// Analyze a vocabulary.
pub fn analyze_vocab(tokens: &[String], special_ids: &HashSet<u32>) -> VocabStats {
    if tokens.is_empty() {
        return VocabStats {
            total_tokens: 0,
            special_count: 0,
            byte_tokens: 0,
            single_char: 0,
            multi_char: 0,
            max_token_len: 0,
            avg_token_len: 0.0,
        };
    }

    let mut byte_tokens = 0;
    let mut single_char = 0;
    let mut multi_char = 0;
    let mut max_len = 0;
    let mut total_len = 0usize;

    for (i, tok) in tokens.iter().enumerate() {
        if special_ids.contains(&(i as u32)) {
            continue;
        }
        let len = tok.len();
        total_len += len;
        if len > max_len {
            max_len = len;
        }
        if tok.starts_with("<0x") && tok.ends_with('>') {
            byte_tokens += 1;
        } else if tok.chars().count() == 1 {
            single_char += 1;
        } else {
            multi_char += 1;
        }
    }

    let non_special = tokens.len() - special_ids.len().min(tokens.len());
    VocabStats {
        total_tokens: tokens.len(),
        special_count: special_ids.len().min(tokens.len()),
        byte_tokens,
        single_char,
        multi_char,
        max_token_len: max_len,
        avg_token_len: if non_special > 0 { total_len as f64 / non_special as f64 } else { 0.0 },
    }
}

/// Compare two vocabularies for overlap.
#[derive(Debug, Clone)]
pub struct VocabOverlap {
    pub common: usize,
    pub only_left: usize,
    pub only_right: usize,
    pub jaccard: f64,
}

pub fn compare_vocabs(left: &[String], right: &[String]) -> VocabOverlap {
    let left_set: HashSet<_> = left.iter().collect();
    let right_set: HashSet<_> = right.iter().collect();
    let common = left_set.intersection(&right_set).count();
    let union = left_set.union(&right_set).count();
    VocabOverlap {
        common,
        only_left: left_set.len() - common,
        only_right: right_set.len() - common,
        jaccard: if union > 0 { common as f64 / union as f64 } else { 0.0 },
    }
}

/// Character coverage analysis.
#[derive(Debug, Clone)]
pub struct CharCoverage {
    pub ascii_printable: usize,
    pub unicode_basic: usize,
    pub unicode_extended: usize,
    pub total_unique_chars: usize,
}

pub fn analyze_char_coverage(tokens: &[String]) -> CharCoverage {
    let mut chars = HashSet::new();
    for tok in tokens {
        for c in tok.chars() {
            chars.insert(c);
        }
    }
    let ascii = chars.iter().filter(|c| c.is_ascii_graphic() || **c == ' ').count();
    let basic = chars.iter().filter(|c| (**c as u32) < 0x10000 && !c.is_ascii()).count();
    let extended = chars.iter().filter(|c| (**c as u32) >= 0x10000).count();
    CharCoverage {
        ascii_printable: ascii,
        unicode_basic: basic,
        unicode_extended: extended,
        total_unique_chars: chars.len(),
    }
}

/// Token length distribution.
pub fn length_distribution(tokens: &[String]) -> HashMap<usize, usize> {
    let mut dist = HashMap::new();
    for tok in tokens {
        *dist.entry(tok.len()).or_insert(0) += 1;
    }
    dist
}

/// Find special tokens by common patterns.
pub fn detect_special_tokens(tokens: &[String]) -> Vec<(u32, String)> {
    let patterns = &[
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
        "<mask>",
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "[UNK]",
        "[MASK]",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|begin_of_text|>",
        "<|end_of_text|>",
    ];
    let mut found = Vec::new();
    for (i, tok) in tokens.iter().enumerate() {
        if patterns.contains(&tok.as_str()) {
            found.push((i as u32, tok.clone()));
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_empty() {
        let stats = analyze_vocab(&[], &HashSet::new());
        assert_eq!(stats.total_tokens, 0);
    }

    #[test]
    fn test_analyze_basic() {
        let tokens = vec!["hello".into(), "world".into(), "<s>".into()];
        let special = HashSet::from([2]);
        let stats = analyze_vocab(&tokens, &special);
        assert_eq!(stats.total_tokens, 3);
        assert_eq!(stats.special_count, 1);
    }

    #[test]
    fn test_byte_tokens() {
        let tokens = vec!["<0x00>".into(), "<0xFF>".into(), "hi".into()];
        let stats = analyze_vocab(&tokens, &HashSet::new());
        assert_eq!(stats.byte_tokens, 2);
    }

    #[test]
    fn test_single_char() {
        let tokens = vec!["a".into(), "b".into(), "hello".into()];
        let stats = analyze_vocab(&tokens, &HashSet::new());
        assert_eq!(stats.single_char, 2);
        assert_eq!(stats.multi_char, 1);
    }

    #[test]
    fn test_compare_vocabs_identical() {
        let a = vec!["a".into(), "b".into()];
        let b = vec!["a".into(), "b".into()];
        let overlap = compare_vocabs(&a, &b);
        assert_eq!(overlap.common, 2);
        assert!((overlap.jaccard - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compare_vocabs_disjoint() {
        let a = vec!["a".into(), "b".into()];
        let b = vec!["c".into(), "d".into()];
        let overlap = compare_vocabs(&a, &b);
        assert_eq!(overlap.common, 0);
        assert_eq!(overlap.jaccard, 0.0);
    }

    #[test]
    fn test_compare_vocabs_partial() {
        let a = vec!["a".into(), "b".into(), "c".into()];
        let b = vec!["b".into(), "c".into(), "d".into()];
        let overlap = compare_vocabs(&a, &b);
        assert_eq!(overlap.common, 2);
        assert_eq!(overlap.only_left, 1);
        assert_eq!(overlap.only_right, 1);
    }

    #[test]
    fn test_char_coverage() {
        let tokens = vec!["hello".into(), "world".into()];
        let cov = analyze_char_coverage(&tokens);
        assert!(cov.ascii_printable > 0);
        assert_eq!(cov.unicode_extended, 0);
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
    fn test_detect_special() {
        let tokens = vec!["hello".into(), "<s>".into(), "</s>".into(), "world".into()];
        let special = detect_special_tokens(&tokens);
        assert_eq!(special.len(), 2);
        assert_eq!(special[0].1, "<s>");
    }

    #[test]
    fn test_detect_no_special() {
        let tokens = vec!["hello".into(), "world".into()];
        assert!(detect_special_tokens(&tokens).is_empty());
    }

    #[test]
    fn test_max_token_len() {
        let tokens = vec!["a".into(), "hello".into(), "ab".into()];
        let stats = analyze_vocab(&tokens, &HashSet::new());
        assert_eq!(stats.max_token_len, 5);
    }
}
