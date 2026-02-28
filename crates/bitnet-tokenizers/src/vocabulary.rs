//! Vocabulary management for tokenizer crate.
//!
//! Provides efficient token↔ID lookup, special token handling, and vocabulary
//! operations such as merging multiple vocabularies for multi-model scenarios.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use bitnet_common::{BitNetError, Result};

// ── Core types ───────────────────────────────────────────────────────

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
    // ── Constructors ─────────────────────────────────────────────────

    /// Build a vocabulary from an explicit token→ID map and config.
    pub fn new(token_to_id: HashMap<String, u32>, config: VocabConfig) -> Self {
        let id_to_token: HashMap<u32, String> =
            token_to_id.iter().map(|(t, &id)| (id, t.clone())).collect();
        let special = Self::resolve_special(&token_to_id, &config);
        Self { token_to_id, id_to_token, config, special }
    }

    /// Load vocabulary from the `"model" → "vocab"` section of a
    /// HuggingFace `tokenizer.json` file.
    ///
    /// Expects the top-level JSON to contain `{ "model": { "vocab": { … } } }`.
    pub fn from_json(data: &str) -> Result<Self> {
        // Deserialize just the parts we need.
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

        // Collect additional special tokens from `added_tokens`.
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

    // ── Lookups ──────────────────────────────────────────────────────

    /// Map a token string to its numeric ID (O(1)).
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Map a numeric ID back to the token string (O(1)).
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Returns `true` when the token string exists in the vocabulary.
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Returns `true` when `id` is a registered special token.
    pub fn is_special_token(&self, id: u32) -> bool {
        self.special.contains(id)
    }

    /// Number of entries in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Iterate over `(token, id)` pairs in arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> {
        self.token_to_id.iter().map(|(t, &id)| (t.as_str(), id))
    }

    /// Read-only access to the vocabulary config.
    pub fn config(&self) -> &VocabConfig {
        &self.config
    }

    /// Read-only access to resolved special tokens.
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    // ── Merge ────────────────────────────────────────────────────────

    /// Merge multiple vocabularies into one.
    ///
    /// Tokens are added in order; if the same string appears in more than
    /// one vocabulary the **first** occurrence wins.  IDs are reassigned
    /// sequentially starting from 0 so the merged vocabulary is dense.
    pub fn merge_vocabularies(vocabs: &[Vocabulary]) -> Vocabulary {
        let mut merged: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut all_additional: Vec<String> = Vec::new();

        for vocab in vocabs {
            // Iterate in ID order for determinism.
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

    // ── Internal helpers ─────────────────────────────────────────────

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

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ─────────────────────────────────────────────────────────

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

    // Basic construction & lookup ─────────────────────────────────────

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

    // Special token identification ────────────────────────────────────

    #[test]
    fn test_special_token_identification() {
        let v = sample_vocab();
        assert!(v.is_special_token(2), "UNK should be special");
        assert!(v.is_special_token(3), "BOS should be special");
        assert!(v.is_special_token(4), "EOS should be special");
        assert!(v.is_special_token(5), "PAD should be special");
        assert!(!v.is_special_token(0), "hello is not special");
        assert!(!v.is_special_token(1), "world is not special");
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

    #[test]
    fn test_additional_special_tokens() {
        let mut config = sample_config();
        config.additional_special_tokens = vec!["hello".into()];
        let v = Vocabulary::new(sample_map(), config);
        assert!(v.is_special_token(0), "hello promoted to special");
    }

    // Unknown token handling ──────────────────────────────────────────

    #[test]
    fn test_unknown_token_lookup() {
        let v = sample_vocab();
        assert_eq!(v.token_to_id("nonexistent"), None);
        assert_eq!(v.id_to_token(999), None);
    }

    // Round-trip ──────────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_token_id_token() {
        let v = sample_vocab();
        for (tok, id) in v.iter() {
            let recovered_id = v.token_to_id(tok).unwrap();
            let recovered_tok = v.id_to_token(recovered_id).unwrap();
            assert_eq!(recovered_id, id);
            assert_eq!(recovered_tok, tok);
        }
    }

    // Contains ────────────────────────────────────────────────────────

    #[test]
    fn test_contains() {
        let v = sample_vocab();
        assert!(v.contains("hello"));
        assert!(!v.contains("missing"));
    }

    // Vocabulary merging ──────────────────────────────────────────────

    #[test]
    fn test_merge_vocabularies_disjoint() {
        let v1 = Vocabulary::new(
            HashMap::from([("a".into(), 0), ("b".into(), 1)]),
            VocabConfig::default(),
        );
        let v2 = Vocabulary::new(
            HashMap::from([("c".into(), 0), ("d".into(), 1)]),
            VocabConfig::default(),
        );
        let merged = Vocabulary::merge_vocabularies(&[v1, v2]);
        assert_eq!(merged.vocab_size(), 4);
        assert!(merged.contains("a"));
        assert!(merged.contains("d"));
    }

    #[test]
    fn test_merge_vocabularies_overlap() {
        let v1 = Vocabulary::new(
            HashMap::from([("a".into(), 0), ("b".into(), 1)]),
            VocabConfig::default(),
        );
        let v2 = Vocabulary::new(
            HashMap::from([("b".into(), 0), ("c".into(), 1)]),
            VocabConfig::default(),
        );
        let merged = Vocabulary::merge_vocabularies(&[v1, v2]);
        // "b" appears in both but should be deduplicated.
        assert_eq!(merged.vocab_size(), 3);
    }

    #[test]
    fn test_merge_preserves_additional_special() {
        let v1 = Vocabulary::new(
            HashMap::from([("x".into(), 0)]),
            VocabConfig { additional_special_tokens: vec!["x".into()], ..VocabConfig::default() },
        );
        let v2 = Vocabulary::new(
            HashMap::from([("y".into(), 0)]),
            VocabConfig { additional_special_tokens: vec!["y".into()], ..VocabConfig::default() },
        );
        let merged = Vocabulary::merge_vocabularies(&[v1, v2]);
        assert!(merged.is_special_token(merged.token_to_id("x").unwrap()));
        assert!(merged.is_special_token(merged.token_to_id("y").unwrap()));
    }

    // Empty vocabulary ────────────────────────────────────────────────

    #[test]
    fn test_empty_vocabulary() {
        let v = Vocabulary::new(HashMap::new(), VocabConfig::default());
        assert_eq!(v.vocab_size(), 0);
        assert_eq!(v.token_to_id("x"), None);
        assert_eq!(v.id_to_token(0), None);
        assert!(!v.is_special_token(0));
        assert_eq!(v.iter().count(), 0);
    }

    // Large vocabulary ────────────────────────────────────────────────

    #[test]
    fn test_large_vocabulary() {
        let map: HashMap<String, u32> = (0..10_000u32).map(|i| (format!("tok_{i}"), i)).collect();
        let v = Vocabulary::new(map, VocabConfig::default());
        assert_eq!(v.vocab_size(), 10_000);
        assert_eq!(v.token_to_id("tok_0"), Some(0));
        assert_eq!(v.token_to_id("tok_9999"), Some(9999));
        assert_eq!(v.id_to_token(5000), Some("tok_5000"));
    }

    // JSON loading ────────────────────────────────────────────────────

    #[test]
    fn test_from_json_basic() {
        let json = r#"{
            "model": {
                "vocab": {
                    "hello": 0,
                    "world": 1
                }
            }
        }"#;
        let v = Vocabulary::from_json(json).unwrap();
        assert_eq!(v.vocab_size(), 2);
        assert_eq!(v.token_to_id("hello"), Some(0));
    }

    #[test]
    fn test_from_json_with_added_tokens() {
        let json = r#"{
            "model": {
                "vocab": {
                    "hi": 0
                }
            },
            "added_tokens": [
                { "content": "<s>", "id": 1, "special": true },
                { "content": "</s>", "id": 2, "special": true },
                { "content": "extra", "id": 3, "special": false }
            ]
        }"#;
        let v = Vocabulary::from_json(json).unwrap();
        assert_eq!(v.vocab_size(), 4);
        assert!(v.is_special_token(1));
        assert!(v.is_special_token(2));
        assert!(!v.is_special_token(3));
    }

    #[test]
    fn test_from_json_invalid() {
        let result = Vocabulary::from_json("not json");
        assert!(result.is_err());
    }

    // Iterator correctness ────────────────────────────────────────────

    #[test]
    fn test_iter_covers_all_entries() {
        let v = sample_vocab();
        let collected: HashMap<String, u32> = v.iter().map(|(t, id)| (t.to_string(), id)).collect();
        assert_eq!(collected.len(), v.vocab_size());
        assert_eq!(collected.get("hello"), Some(&0));
    }

    #[test]
    fn test_iter_empty() {
        let v = Vocabulary::new(HashMap::new(), VocabConfig::default());
        assert_eq!(v.iter().count(), 0);
    }
}
