//! Tokenizer preprocessor.
//!
//! Text normalization and preprocessing before tokenization.

/// Preprocessing rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreprocessRule {
    Lowercase,
    StripWhitespace,
    NormalizeUnicode,
    RemoveAccents,
    AddPrefixSpace,
    CollapseWhitespace,
}

/// Preprocessor configuration.
#[derive(Debug, Clone, Default)]
pub struct PreprocessConfig {
    pub rules: Vec<PreprocessRule>,
    pub max_length: Option<usize>,
    pub add_bos: bool,
    pub add_eos: bool,
}

impl PreprocessConfig {
    pub fn gpt_style() -> Self {
        Self {
            rules: vec![PreprocessRule::AddPrefixSpace, PreprocessRule::CollapseWhitespace],
            max_length: None,
            add_bos: false,
            add_eos: false,
        }
    }

    pub fn llama_style() -> Self {
        Self {
            rules: vec![PreprocessRule::CollapseWhitespace],
            max_length: None,
            add_bos: true,
            add_eos: false,
        }
    }

    pub fn phi_style() -> Self {
        Self {
            rules: vec![PreprocessRule::CollapseWhitespace],
            max_length: None,
            add_bos: false,
            add_eos: false,
        }
    }
}

/// Apply a single preprocessing rule.
pub fn apply_rule(text: &str, rule: PreprocessRule) -> String {
    match rule {
        PreprocessRule::Lowercase => text.to_lowercase(),
        PreprocessRule::StripWhitespace => text.trim().to_string(),
        PreprocessRule::NormalizeUnicode => normalize_basic(text),
        PreprocessRule::RemoveAccents => remove_accents(text),
        PreprocessRule::AddPrefixSpace => {
            if text.starts_with(' ') {
                text.to_string()
            } else {
                format!(" {text}")
            }
        }
        PreprocessRule::CollapseWhitespace => collapse_whitespace(text),
    }
}

/// Apply all preprocessing rules.
pub fn preprocess(text: &str, config: &PreprocessConfig) -> String {
    let mut result = text.to_string();
    for rule in &config.rules {
        result = apply_rule(&result, *rule);
    }
    if let Some(max_len) = config.max_length
        && result.len() > max_len
    {
        result.truncate(max_len);
    }
    result
}

fn collapse_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_space {
                result.push(' ');
                prev_space = true;
            }
        } else {
            result.push(c);
            prev_space = false;
        }
    }
    result
}

fn normalize_basic(text: &str) -> String {
    // Basic NFC-like normalization: just handle common cases
    text.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' => '\'',
            '\u{201C}' | '\u{201D}' => '"',
            '\u{2014}' => '-',
            '\u{2026}' => '.',
            _ => c,
        })
        .collect()
}

fn remove_accents(text: &str) -> String {
    // Simplified accent removal for common Latin characters
    text.chars()
        .map(|c| match c {
            '\u{00E0}'..='\u{00E5}' => 'a',
            '\u{00E8}'..='\u{00EB}' => 'e',
            '\u{00EC}'..='\u{00EF}' => 'i',
            '\u{00F2}'..='\u{00F6}' => 'o',
            '\u{00F9}'..='\u{00FC}' => 'u',
            '\u{00C0}'..='\u{00C5}' => 'A',
            '\u{00C8}'..='\u{00CB}' => 'E',
            '\u{00CC}'..='\u{00CF}' => 'I',
            '\u{00D2}'..='\u{00D6}' => 'O',
            '\u{00D9}'..='\u{00DC}' => 'U',
            _ => c,
        })
        .collect()
}

/// Split text into chunks respecting token boundaries.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() || chunk_size == 0 {
        return vec![];
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end >= words.len() {
            break;
        }
        start = end.saturating_sub(overlap);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowercase() {
        assert_eq!(apply_rule("Hello World", PreprocessRule::Lowercase), "hello world");
    }

    #[test]
    fn test_strip() {
        assert_eq!(apply_rule("  hello  ", PreprocessRule::StripWhitespace), "hello");
    }

    #[test]
    fn test_prefix_space() {
        assert_eq!(apply_rule("hello", PreprocessRule::AddPrefixSpace), " hello");
        assert_eq!(apply_rule(" hello", PreprocessRule::AddPrefixSpace), " hello");
    }

    #[test]
    fn test_collapse_whitespace() {
        assert_eq!(apply_rule("hello   world", PreprocessRule::CollapseWhitespace), "hello world");
    }

    #[test]
    fn test_normalize_unicode() {
        assert_eq!(
            apply_rule("\u{201C}hello\u{201D}", PreprocessRule::NormalizeUnicode),
            "\"hello\""
        );
    }

    #[test]
    fn test_remove_accents() {
        assert_eq!(apply_rule("\u{00E9}t\u{00E9}", PreprocessRule::RemoveAccents), "ete");
    }

    #[test]
    fn test_preprocess_chain() {
        let config = PreprocessConfig {
            rules: vec![PreprocessRule::CollapseWhitespace, PreprocessRule::Lowercase],
            max_length: None,
            add_bos: false,
            add_eos: false,
        };
        assert_eq!(preprocess("Hello   World", &config), "hello world");
    }

    #[test]
    fn test_max_length() {
        let config =
            PreprocessConfig { rules: vec![], max_length: Some(5), add_bos: false, add_eos: false };
        assert_eq!(preprocess("hello world", &config), "hello");
    }

    #[test]
    fn test_gpt_style() {
        let config = PreprocessConfig::gpt_style();
        let result = preprocess("hello  world", &config);
        assert!(result.starts_with(' '));
        assert!(!result.contains("  "));
    }

    #[test]
    fn test_llama_style() {
        let config = PreprocessConfig::llama_style();
        assert!(config.add_bos);
    }

    #[test]
    fn test_chunk_text() {
        let chunks = chunk_text("a b c d e f", 3, 1);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0], "a b c");
    }

    #[test]
    fn test_chunk_empty() {
        assert!(chunk_text("", 3, 1).is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = PreprocessConfig::default();
        assert!(config.rules.is_empty());
        assert!(!config.add_bos);
    }

    #[test]
    fn test_phi_style() {
        let config = PreprocessConfig::phi_style();
        assert!(!config.add_bos);
    }
}
