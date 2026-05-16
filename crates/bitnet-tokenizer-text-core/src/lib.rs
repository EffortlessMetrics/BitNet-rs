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

    #[test]
    fn pre_tokenize_whitespace_splits_on_runs() {
        let tk = PreTokenizer::new(PreTokenizeStrategy::Whitespace);
        assert_eq!(tk.pre_tokenize("hello   world"), vec!["hello", "world"]);
        // Tabs and newlines count as whitespace.
        assert_eq!(tk.pre_tokenize("a\tb\nc"), vec!["a", "b", "c"]);
    }

    #[test]
    fn pre_tokenize_whitespace_empty_input() {
        let tk = PreTokenizer::new(PreTokenizeStrategy::Whitespace);
        assert!(tk.pre_tokenize("").is_empty());
        assert!(tk.pre_tokenize("   \t\n").is_empty());
    }

    #[test]
    fn pre_tokenize_byte_level_replaces_spaces_with_marker() {
        let tk = PreTokenizer::new(PreTokenizeStrategy::ByteLevel);
        let out = tk.pre_tokenize("a b");
        // Three chars: 'a', U+0120, 'b'.
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], "a");
        assert_eq!(out[1], "\u{0120}");
        assert_eq!(out[2], "b");
    }

    #[test]
    fn pre_tokenize_byte_level_empty_input_returns_empty() {
        let tk = PreTokenizer::new(PreTokenizeStrategy::ByteLevel);
        assert!(tk.pre_tokenize("").is_empty());
    }

    #[test]
    fn pre_tokenize_byte_level_single_char() {
        let tk = PreTokenizer::new(PreTokenizeStrategy::ByteLevel);
        assert_eq!(tk.pre_tokenize("x"), vec!["x".to_string()]);
    }

    #[test]
    fn pre_tokenize_punctuation_delegates_to_split_on_punctuation() {
        let tk = PreTokenizer::new(PreTokenizeStrategy::Punctuation);
        assert_eq!(tk.pre_tokenize("hi, bye."), vec!["hi", ",", "bye", "."]);
    }

    #[test]
    fn split_on_punctuation_empty_input() {
        assert!(split_on_punctuation("").is_empty());
    }

    #[test]
    fn split_on_punctuation_only_whitespace() {
        // Whitespace alone produces no tokens and no punctuation tokens.
        assert!(split_on_punctuation("   \t  ").is_empty());
    }

    #[test]
    fn split_on_punctuation_only_punctuation() {
        assert_eq!(split_on_punctuation("!?."), vec!["!", "?", "."]);
    }

    #[test]
    fn split_on_punctuation_leading_and_trailing_punctuation() {
        assert_eq!(split_on_punctuation("!hi"), vec!["!", "hi"]);
        assert_eq!(split_on_punctuation("hi!"), vec!["hi", "!"]);
    }

    #[test]
    fn split_on_punctuation_internal_whitespace_does_not_emit_token() {
        // A space ends the current token but does not become its own token.
        let tokens = split_on_punctuation("ab cd");
        assert_eq!(tokens, vec!["ab", "cd"]);
    }

    #[test]
    fn normalizer_lowercase_constructor_sets_only_lowercase() {
        let n = TokenNormalizer::lowercase();
        assert_eq!(n.normalize("ABC Café"), "abc café");
    }

    #[test]
    fn normalizer_default_flags_is_identity() {
        let n = TokenNormalizer::new(NormalizationFlags::default());
        let s = "Hello — World";
        assert_eq!(n.normalize(s), s);
    }

    #[test]
    fn normalizer_only_nfkc_does_not_change_case_or_accents() {
        let n = TokenNormalizer::new(NormalizationFlags { nfkc: true, ..Default::default() });
        // Curly quote folded; case and accent preserved.
        assert_eq!(n.normalize("“Café”"), "\"Café\"");
    }

    #[test]
    fn normalizer_only_strip_accents_keeps_case() {
        let n =
            TokenNormalizer::new(NormalizationFlags { strip_accents: true, ..Default::default() });
        assert_eq!(n.normalize("Café Déjà"), "Cafe Deja");
    }

    #[test]
    fn ascii_nfkc_fold_curly_quotes() {
        assert_eq!(ascii_nfkc_fold("\u{201C}hi\u{201D}"), "\"hi\"");
        assert_eq!(ascii_nfkc_fold("\u{2018}hi\u{2019}"), "'hi'");
    }

    #[test]
    fn ascii_nfkc_fold_em_dash_ellipsis_nbsp() {
        assert_eq!(ascii_nfkc_fold("a\u{2014}b"), "a-b");
        assert_eq!(ascii_nfkc_fold("e\u{2026}"), "e.");
        // Non-breaking space → regular space.
        assert_eq!(ascii_nfkc_fold("a\u{00A0}b"), "a b");
    }

    #[test]
    fn ascii_nfkc_fold_leaves_unaffected_chars_alone() {
        let s = "abc XYZ 123 !@# αβ漢";
        assert_eq!(ascii_nfkc_fold(s), s);
    }

    #[test]
    fn strip_accents_ascii_covers_each_vowel_family() {
        assert_eq!(strip_accents_ascii("áàâäã"), "aaaaa");
        assert_eq!(strip_accents_ascii("éèêë"), "eeee");
        assert_eq!(strip_accents_ascii("íìîï"), "iiii");
        assert_eq!(strip_accents_ascii("óòôöõ"), "ooooo");
        assert_eq!(strip_accents_ascii("úùûü"), "uuuu");
        assert_eq!(strip_accents_ascii("ñ"), "n");
        assert_eq!(strip_accents_ascii("ç"), "c");
    }

    #[test]
    fn strip_accents_ascii_preserves_unrelated_unicode() {
        // Upper-case accented letters are not covered by this table.
        assert_eq!(strip_accents_ascii("ÁÉ漢"), "ÁÉ漢");
    }

    #[test]
    fn pre_tokenize_strategy_is_copy_and_eq() {
        let s = PreTokenizeStrategy::ByteLevel;
        let copy = s;
        assert_eq!(s, copy);
        // Debug should render without panicking.
        let _ = format!("{s:?}");
    }

    #[test]
    fn normalization_flags_default_is_all_false() {
        let flags = NormalizationFlags::default();
        assert!(!flags.nfkc);
        assert!(!flags.lowercase);
        assert!(!flags.strip_accents);
    }
}
