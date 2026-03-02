//! Shared vocabulary management primitives.
//!
//! Provides efficient token↔ID lookup, special token handling, and vocabulary
//! operations such as merging multiple vocabularies for multi-model scenarios.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use bitnet_common::{BitNetError, Result};

/// Configuration for special tokens in a vocabulary.
#[derive(Debug, Clone, Default, serde::Serialize, Deserialize)]
pub struct VocabConfig {
    pub unk_token: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub pad_token: Option<String>,
    pub additional_special_tokens: Vec<String>,
}

/// Resolved special-token IDs for fast runtime checks.
#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    pub unk_id: Option<u32>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub pad_id: Option<u32>,
    /// All special token IDs (including BOS/EOS/UNK/PAD and additional).
    all_ids: HashSet<u32>,
}

impl SpecialTokens {
    /// Returns `true` if `id` is any registered special token.
    pub fn contains(&self, id: u32) -> bool {
        self.all_ids.contains(&id)
    }
}

/// Bidirectional vocabulary with O(1) lookups in both directions.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    config: VocabConfig,
    special: SpecialTokens,
}

impl Vocabulary {
    /// Build a vocabulary from an explicit token→ID map and config.
    pub fn new(token_to_id: HashMap<String, u32>, config: VocabConfig) -> Self {
        let id_to_token: HashMap<u32, String> =
            token_to_id.iter().map(|(t, &id)| (id, t.clone())).collect();
        let special = Self::resolve_special(&token_to_id, &config);
        Self { token_to_id, id_to_token, config, special }
    }

    /// Load vocabulary from the "model" → "vocab" section of a
    /// HuggingFace `tokenizer.json` file.
    pub fn from_json(data: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct ModelSection {
            vocab: HashMap<String, u32>,
        }
        #[derive(Deserialize)]
        struct Root {
            model: ModelSection,
            #[serde(default)]
            added_tokens: Vec<AddedToken>,
        }
        #[derive(Deserialize)]
        struct AddedToken {
            content: String,
            id: u32,
            special: bool,
        }

        let root: Root = serde_json::from_str(data)
            .map_err(|e| BitNetError::Config(format!("failed to parse vocabulary JSON: {e}")))?;

        let mut token_to_id = root.model.vocab;
        let mut additional: Vec<String> = Vec::new();
        for at in &root.added_tokens {
            token_to_id.entry(at.content.clone()).or_insert(at.id);
            if at.special {
                additional.push(at.content.clone());
            }
        }

        let config =
            VocabConfig { additional_special_tokens: additional, ..VocabConfig::default() };

        Ok(Self::new(token_to_id, config))
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    pub fn is_special_token(&self, id: u32) -> bool {
        self.special.contains(id)
    }

    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> {
        self.token_to_id.iter().map(|(t, &id)| (t.as_str(), id))
    }

    pub fn config(&self) -> &VocabConfig {
        &self.config
    }

    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    pub fn merge_vocabularies(vocabs: &[Vocabulary]) -> Vocabulary {
        let mut merged: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut all_additional: Vec<String> = Vec::new();

        for vocab in vocabs {
            let mut pairs: Vec<(&str, u32)> = vocab.iter().collect();
            pairs.sort_by_key(|&(_, id)| id);
            for (tok, _) in pairs {
                if seen.insert(tok.to_string()) {
                    merged.push(tok.to_string());
                }
            }
            for extra in &vocab.config.additional_special_tokens {
                if !all_additional.contains(extra) {
                    all_additional.push(extra.clone());
                }
            }
        }

        let token_to_id: HashMap<String, u32> =
            merged.into_iter().enumerate().map(|(i, t)| (t, i as u32)).collect();

        let config =
            VocabConfig { additional_special_tokens: all_additional, ..VocabConfig::default() };

        Vocabulary::new(token_to_id, config)
    }

    fn resolve_special(map: &HashMap<String, u32>, config: &VocabConfig) -> SpecialTokens {
        let resolve = |opt: &Option<String>| -> Option<u32> {
            opt.as_ref().and_then(|t| map.get(t.as_str()).copied())
        };

        let unk_id = resolve(&config.unk_token);
        let bos_id = resolve(&config.bos_token);
        let eos_id = resolve(&config.eos_token);
        let pad_id = resolve(&config.pad_token);

        let mut all_ids: HashSet<u32> = HashSet::new();
        for id in [unk_id, bos_id, eos_id, pad_id].into_iter().flatten() {
            all_ids.insert(id);
        }
        for tok in &config.additional_special_tokens {
            if let Some(&id) = map.get(tok.as_str()) {
                all_ids.insert(id);
            }
        }

        SpecialTokens { unk_id, bos_id, eos_id, pad_id, all_ids }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_map() -> HashMap<String, u32> {
        HashMap::from([
            ("hello".into(), 0),
            ("world".into(), 1),
            ("<unk>".into(), 2),
            ("<s>".into(), 3),
            ("</s>".into(), 4),
            ("<pad>".into(), 5),
        ])
    }

    fn sample_config() -> VocabConfig {
        VocabConfig {
            unk_token: Some("<unk>".into()),
            bos_token: Some("<s>".into()),
            eos_token: Some("</s>".into()),
            pad_token: Some("<pad>".into()),
            additional_special_tokens: vec![],
        }
    }

    fn sample_vocab() -> Vocabulary {
        Vocabulary::new(sample_map(), sample_config())
    }

    #[test]
    fn test_basic_construction_and_lookup() {
        let v = sample_vocab();
        assert_eq!(v.token_to_id("hello"), Some(0));
        assert_eq!(v.token_to_id("world"), Some(1));
        assert_eq!(v.id_to_token(0), Some("hello"));
        assert_eq!(v.id_to_token(1), Some("world"));
    }

    #[test]
    fn test_vocab_size() {
        let v = sample_vocab();
        assert_eq!(v.vocab_size(), 6);
    }

    #[test]
    fn test_special_token_identification() {
        let v = sample_vocab();
        assert!(v.is_special_token(2));
        assert!(v.is_special_token(3));
        assert!(v.is_special_token(4));
        assert!(v.is_special_token(5));
        assert!(!v.is_special_token(0));
        assert!(!v.is_special_token(1));
    }

    #[test]
    fn test_special_tokens_struct_accessors() {
        let v = sample_vocab();
        let st = v.special_tokens();
        assert_eq!(st.unk_id, Some(2));
        assert_eq!(st.bos_id, Some(3));
        assert_eq!(st.eos_id, Some(4));
        assert_eq!(st.pad_id, Some(5));
    }
}
