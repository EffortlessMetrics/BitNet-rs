//! SRP tokenizer text preprocessing primitives.
//!
//! This crate isolates pre-tokenization and lightweight text normalization so
//! tokenizer pipelines can share deterministic behavior.

/// Strategy used to split raw text before token encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizeStrategy {
    /// Split on Unicode whitespace boundaries.
    Whitespace,
    /// Byte-level preparation for BPE (no whitespace collapse).
    ByteLevel,
    /// Split on punctuation boundaries.
    Punctuation,
}

/// Pre-tokenizer: splits raw text into candidate tokens before encoding.
#[derive(Debug, Clone)]
pub struct PreTokenizer {
    strategy: PreTokenizeStrategy,
}

impl PreTokenizer {
    /// Create a new pre-tokenizer with the given strategy.
    #[must_use]
    pub const fn new(strategy: PreTokenizeStrategy) -> Self {
        Self { strategy }
    }

    /// Split `text` into pre-tokenized segments.
    #[must_use]
    pub fn pre_tokenize(&self, text: &str) -> Vec<String> {
        match self.strategy {
            PreTokenizeStrategy::Whitespace => text.split_whitespace().map(String::from).collect(),
            PreTokenizeStrategy::ByteLevel => {
                if text.is_empty() {
                    return Vec::new();
                }
                let replaced = text.replace(' ', "\u{0120}");
                replaced.chars().map(|c| c.to_string()).collect()
            }
            PreTokenizeStrategy::Punctuation => split_on_punctuation(text),
        }
    }
}

/// Flags controlling which normalization passes to apply.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NormalizationFlags {
    /// Apply Unicode NFKC-like folding (ASCII approximation).
    pub nfkc: bool,
    /// Lower-case the text.
    pub lowercase: bool,
    /// Strip combining diacritical marks (accent stripping).
    pub strip_accents: bool,
}

/// Text normalizer applied before pre-tokenization.
#[derive(Debug, Clone)]
pub struct TokenNormalizer {
    flags: NormalizationFlags,
}

impl TokenNormalizer {
    /// Create a normalizer with the given flags.
    #[must_use]
    pub const fn new(flags: NormalizationFlags) -> Self {
        Self { flags }
    }

    /// Create a normalizer that lower-cases text.
    #[must_use]
    pub fn lowercase() -> Self {
        Self { flags: NormalizationFlags { lowercase: true, ..Default::default() } }
    }

    /// Normalize `text` according to the configured flags.
    #[must_use]
    pub fn normalize(&self, text: &str) -> String {
        let mut out = text.to_string();
        if self.flags.nfkc {
            out = ascii_nfkc_fold(&out);
        }
        if self.flags.strip_accents {
            out = strip_accents_ascii(&out);
        }
        if self.flags.lowercase {
            out = out.to_lowercase();
        }
        out
    }
}

/// Split text on punctuation boundaries, keeping punctuation tokens.
#[must_use]
pub fn split_on_punctuation(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_punctuation() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            tokens.push(ch.to_string());
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Minimal ASCII NFKC-like folding: curly quotes → straight, em-dash → -, etc.
#[must_use]
pub fn ascii_nfkc_fold(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{201C}' | '\u{201D}' => '"',
            '\u{2018}' | '\u{2019}' => '\'',
            '\u{2014}' => '-',
            '\u{2026}' => '.',
            '\u{00A0}' => ' ',
            other => other,
        })
        .collect()
}

/// Strip common accented Latin characters to their base letter.
#[must_use]
pub fn strip_accents_ascii(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'á' | 'à' | 'â' | 'ä' | 'ã' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'ô' | 'ö' | 'õ' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ñ' => 'n',
            'ç' => 'c',
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punctuation_split_keeps_marks() {
        assert_eq!(split_on_punctuation("hello, world!"), vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn normalizer_pipeline_applies_all_flags() {
        let normalizer = TokenNormalizer::new(NormalizationFlags {
            nfkc: true,
            strip_accents: true,
            lowercase: true,
        });
        assert_eq!(normalizer.normalize("“Café” — déjà vu"), "\"cafe\" - deja vu");
    }
}
