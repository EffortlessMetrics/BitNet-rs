//! Token ID validation utilities.
//!
//! Ensures token sequences are valid before inference.

/// Validation errors for token sequences.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenError {
    EmptySequence,
    ExceedsVocab { token_id: u32, vocab_size: u32 },
    ExceedsMaxLength { length: usize, max: usize },
    MissingBos,
    DuplicateBos { count: usize },
    InvalidSpecialToken { token_id: u32 },
}

impl std::fmt::Display for TokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySequence => write!(f, "empty token sequence"),
            Self::ExceedsVocab { token_id, vocab_size } => {
                write!(f, "token {token_id} exceeds vocab size {vocab_size}")
            }
            Self::ExceedsMaxLength { length, max } => {
                write!(f, "sequence length {length} exceeds max {max}")
            }
            Self::MissingBos => write!(f, "missing BOS token"),
            Self::DuplicateBos { count } => write!(f, "duplicate BOS tokens: {count}"),
            Self::InvalidSpecialToken { token_id } => {
                write!(f, "invalid special token: {token_id}")
            }
        }
    }
}

/// Configuration for token validation.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub vocab_size: u32,
    pub max_length: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub require_bos: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            max_length: 4096,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            pad_token_id: None,
            require_bos: false,
        }
    }
}

/// Validate a token sequence.
pub fn validate_tokens(tokens: &[u32], config: &ValidationConfig) -> Result<(), Vec<TokenError>> {
    let mut errors = Vec::new();

    if tokens.is_empty() {
        errors.push(TokenError::EmptySequence);
        return Err(errors);
    }

    if tokens.len() > config.max_length {
        errors.push(TokenError::ExceedsMaxLength { length: tokens.len(), max: config.max_length });
    }

    for &token_id in tokens {
        if token_id >= config.vocab_size {
            errors.push(TokenError::ExceedsVocab { token_id, vocab_size: config.vocab_size });
        }
    }

    if config.require_bos
        && let Some(bos) = config.bos_token_id
    {
        let bos_count = tokens.iter().filter(|&&t| t == bos).count();
        if bos_count == 0 {
            errors.push(TokenError::MissingBos);
        } else if bos_count > 1 {
            errors.push(TokenError::DuplicateBos { count: bos_count });
        }
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// Quick check: are all tokens within vocab range?
pub fn all_in_vocab(tokens: &[u32], vocab_size: u32) -> bool {
    tokens.iter().all(|&t| t < vocab_size)
}

/// Count special tokens in a sequence.
pub fn count_special_tokens(tokens: &[u32], special_ids: &[u32]) -> usize {
    tokens.iter().filter(|t| special_ids.contains(t)).count()
}

/// Remove padding tokens from the end of a sequence.
pub fn strip_padding(tokens: &[u32], pad_id: u32) -> &[u32] {
    let end = tokens.iter().rposition(|&t| t != pad_id).map(|i| i + 1).unwrap_or(0);
    &tokens[..end]
}

/// Find the position of the first EOS token.
pub fn find_eos(tokens: &[u32], eos_id: u32) -> Option<usize> {
    tokens.iter().position(|&t| t == eos_id)
}

/// Truncate a sequence to max_length, preserving BOS if present.
pub fn truncate(tokens: &[u32], max_length: usize, bos_id: Option<u32>) -> Vec<u32> {
    if tokens.len() <= max_length {
        return tokens.to_vec();
    }
    if max_length == 0 {
        return vec![];
    }

    // If first token is BOS, keep it
    if let Some(bos) = bos_id
        && !tokens.is_empty()
        && tokens[0] == bos
        && max_length >= 2
    {
        let mut result = vec![bos];
        let skip = tokens.len() - (max_length - 1);
        result.extend_from_slice(&tokens[skip..]);
        return result;
    }

    tokens[tokens.len() - max_length..].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> ValidationConfig {
        ValidationConfig::default()
    }

    #[test]
    fn test_valid_tokens() {
        let r = validate_tokens(&[1, 100, 500], &default_config());
        assert!(r.is_ok());
    }

    #[test]
    fn test_empty() {
        let r = validate_tokens(&[], &default_config());
        assert!(r.is_err());
        assert!(r.unwrap_err().contains(&TokenError::EmptySequence));
    }

    #[test]
    fn test_exceeds_vocab() {
        let r = validate_tokens(&[1, 99999], &default_config());
        assert!(r.is_err());
    }

    #[test]
    fn test_exceeds_length() {
        let mut config = default_config();
        config.max_length = 2;
        let r = validate_tokens(&[1, 2, 3], &config);
        assert!(r.is_err());
    }

    #[test]
    fn test_missing_bos() {
        let mut config = default_config();
        config.require_bos = true;
        config.bos_token_id = Some(1);
        let r = validate_tokens(&[100, 200], &config);
        assert!(r.is_err());
    }

    #[test]
    fn test_duplicate_bos() {
        let mut config = default_config();
        config.require_bos = true;
        config.bos_token_id = Some(1);
        let r = validate_tokens(&[1, 100, 1], &config);
        assert!(r.is_err());
    }

    #[test]
    fn test_all_in_vocab() {
        assert!(all_in_vocab(&[0, 1, 99], 100));
        assert!(!all_in_vocab(&[0, 100], 100));
    }

    #[test]
    fn test_count_special() {
        assert_eq!(count_special_tokens(&[1, 5, 2, 5], &[1, 2]), 2);
    }

    #[test]
    fn test_strip_padding() {
        let tokens = &[1, 5, 10, 0, 0, 0];
        let stripped = strip_padding(tokens, 0);
        assert_eq!(stripped, &[1, 5, 10]);
    }

    #[test]
    fn test_strip_all_padding() {
        let tokens = &[0, 0, 0];
        let stripped = strip_padding(tokens, 0);
        assert!(stripped.is_empty());
    }

    #[test]
    fn test_find_eos() {
        assert_eq!(find_eos(&[1, 5, 2, 10], 2), Some(2));
        assert_eq!(find_eos(&[1, 5, 10], 2), None);
    }

    #[test]
    fn test_truncate_short() {
        let t = truncate(&[1, 2, 3], 5, None);
        assert_eq!(t, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_long() {
        let t = truncate(&[1, 2, 3, 4, 5], 3, None);
        assert_eq!(t, vec![3, 4, 5]); // keep last 3
    }

    #[test]
    fn test_truncate_with_bos() {
        let t = truncate(&[1, 2, 3, 4, 5], 3, Some(1));
        assert_eq!(t, vec![1, 4, 5]); // keep BOS + last 2
    }

    #[test]
    fn test_display() {
        let e = TokenError::ExceedsVocab { token_id: 50000, vocab_size: 32000 };
        assert!(e.to_string().contains("50000"));
    }
}

