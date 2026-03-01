//! Edge-case tests for `bitnet-tokenizers` vocabulary and discovery modules.
//!
//! Tests Vocabulary construction, lookups, special tokens, merge, JSON parsing,
//! and ModelCompatibilityMatrix defaults.

use bitnet_tokenizers::vocabulary::{VocabConfig, Vocabulary};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn simple_vocab() -> Vocabulary {
    let map: HashMap<String, u32> =
        [("hello", 0), ("world", 1), ("<unk>", 2), ("<bos>", 3), ("<eos>", 4)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
    let config = VocabConfig {
        unk_token: Some("<unk>".to_string()),
        bos_token: Some("<bos>".to_string()),
        eos_token: Some("<eos>".to_string()),
        pad_token: None,
        additional_special_tokens: vec![],
    };
    Vocabulary::new(map, config)
}

// ---------------------------------------------------------------------------
// VocabConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn vocab_config_default_is_empty() {
    let cfg = VocabConfig::default();
    assert!(cfg.unk_token.is_none());
    assert!(cfg.bos_token.is_none());
    assert!(cfg.eos_token.is_none());
    assert!(cfg.pad_token.is_none());
    assert!(cfg.additional_special_tokens.is_empty());
}

// ---------------------------------------------------------------------------
// Vocabulary construction
// ---------------------------------------------------------------------------

#[test]
fn vocabulary_size() {
    let vocab = simple_vocab();
    assert_eq!(vocab.vocab_size(), 5);
}

#[test]
fn empty_vocabulary() {
    let vocab = Vocabulary::new(HashMap::new(), VocabConfig::default());
    assert_eq!(vocab.vocab_size(), 0);
}

#[test]
fn vocabulary_from_single_token() {
    let map: HashMap<String, u32> = [("a".to_string(), 0)].into_iter().collect();
    let vocab = Vocabulary::new(map, VocabConfig::default());
    assert_eq!(vocab.vocab_size(), 1);
    assert_eq!(vocab.token_to_id("a"), Some(0));
}

// ---------------------------------------------------------------------------
// Token lookups
// ---------------------------------------------------------------------------

#[test]
fn token_to_id_found() {
    let vocab = simple_vocab();
    assert_eq!(vocab.token_to_id("hello"), Some(0));
    assert_eq!(vocab.token_to_id("world"), Some(1));
}

#[test]
fn token_to_id_not_found() {
    let vocab = simple_vocab();
    assert_eq!(vocab.token_to_id("missing"), None);
}

#[test]
fn id_to_token_found() {
    let vocab = simple_vocab();
    assert_eq!(vocab.id_to_token(0), Some("hello"));
    assert_eq!(vocab.id_to_token(1), Some("world"));
}

#[test]
fn id_to_token_not_found() {
    let vocab = simple_vocab();
    assert_eq!(vocab.id_to_token(999), None);
}

#[test]
fn contains_positive() {
    let vocab = simple_vocab();
    assert!(vocab.contains("hello"));
    assert!(vocab.contains("<unk>"));
}

#[test]
fn contains_negative() {
    let vocab = simple_vocab();
    assert!(!vocab.contains("missing"));
    assert!(!vocab.contains(""));
}

// ---------------------------------------------------------------------------
// Special tokens
// ---------------------------------------------------------------------------

#[test]
fn special_tokens_resolved() {
    let vocab = simple_vocab();
    let special = vocab.special_tokens();
    assert_eq!(special.unk_id, Some(2));
    assert_eq!(special.bos_id, Some(3));
    assert_eq!(special.eos_id, Some(4));
    assert_eq!(special.pad_id, None);
}

#[test]
fn is_special_token_true() {
    let vocab = simple_vocab();
    assert!(vocab.is_special_token(2)); // <unk>
    assert!(vocab.is_special_token(3)); // <bos>
    assert!(vocab.is_special_token(4)); // <eos>
}

#[test]
fn is_special_token_false() {
    let vocab = simple_vocab();
    assert!(!vocab.is_special_token(0)); // hello
    assert!(!vocab.is_special_token(1)); // world
    assert!(!vocab.is_special_token(999)); // out of range
}

#[test]
fn special_tokens_with_pad() {
    let map: HashMap<String, u32> =
        [("<pad>", 0), ("a", 1)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    let config = VocabConfig { pad_token: Some("<pad>".to_string()), ..VocabConfig::default() };
    let vocab = Vocabulary::new(map, config);
    assert_eq!(vocab.special_tokens().pad_id, Some(0));
    assert!(vocab.is_special_token(0));
}

#[test]
fn special_token_not_in_vocab_resolves_to_none() {
    let map: HashMap<String, u32> = [("a".to_string(), 0)].into_iter().collect();
    let config = VocabConfig {
        bos_token: Some("<bos>".to_string()), // not in map
        ..VocabConfig::default()
    };
    let vocab = Vocabulary::new(map, config);
    assert_eq!(vocab.special_tokens().bos_id, None);
}

#[test]
fn additional_special_tokens_tracked() {
    let map: HashMap<String, u32> = [("<sep>", 0), ("<cls>", 1), ("tok", 2)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    let config = VocabConfig {
        additional_special_tokens: vec!["<sep>".to_string(), "<cls>".to_string()],
        ..VocabConfig::default()
    };
    let vocab = Vocabulary::new(map, config);
    assert!(vocab.is_special_token(0)); // <sep>
    assert!(vocab.is_special_token(1)); // <cls>
    assert!(!vocab.is_special_token(2)); // tok
}

// ---------------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------------

#[test]
fn iter_covers_all_tokens() {
    let vocab = simple_vocab();
    let pairs: Vec<_> = vocab.iter().collect();
    assert_eq!(pairs.len(), 5);
}

// ---------------------------------------------------------------------------
// Config accessor
// ---------------------------------------------------------------------------

#[test]
fn config_accessor() {
    let vocab = simple_vocab();
    assert_eq!(vocab.config().unk_token.as_deref(), Some("<unk>"));
    assert_eq!(vocab.config().bos_token.as_deref(), Some("<bos>"));
}

// ---------------------------------------------------------------------------
// Merge vocabularies
// ---------------------------------------------------------------------------

#[test]
fn merge_two_disjoint_vocabularies() {
    let v1 = {
        let map: HashMap<String, u32> =
            [("a", 0), ("b", 1)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        Vocabulary::new(map, VocabConfig::default())
    };
    let v2 = {
        let map: HashMap<String, u32> =
            [("c", 0), ("d", 1)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        Vocabulary::new(map, VocabConfig::default())
    };
    let merged = Vocabulary::merge_vocabularies(&[v1, v2]);
    assert_eq!(merged.vocab_size(), 4);
    assert!(merged.contains("a"));
    assert!(merged.contains("d"));
}

#[test]
fn merge_overlapping_vocabularies_first_wins() {
    let v1 = {
        let map: HashMap<String, u32> =
            [("x", 0), ("y", 1)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        Vocabulary::new(map, VocabConfig::default())
    };
    let v2 = {
        let map: HashMap<String, u32> =
            [("y", 0), ("z", 1)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        Vocabulary::new(map, VocabConfig::default())
    };
    let merged = Vocabulary::merge_vocabularies(&[v1, v2]);
    // x, y (from v1), z — no duplicate y
    assert_eq!(merged.vocab_size(), 3);
}

#[test]
fn merge_empty_vocabularies() {
    let v1 = Vocabulary::new(HashMap::new(), VocabConfig::default());
    let v2 = Vocabulary::new(HashMap::new(), VocabConfig::default());
    let merged = Vocabulary::merge_vocabularies(&[v1, v2]);
    assert_eq!(merged.vocab_size(), 0);
}

#[test]
fn merge_preserves_additional_special_tokens() {
    let v1 = Vocabulary::new(
        HashMap::new(),
        VocabConfig {
            additional_special_tokens: vec!["<sep>".to_string()],
            ..VocabConfig::default()
        },
    );
    let v2 = Vocabulary::new(
        HashMap::new(),
        VocabConfig {
            additional_special_tokens: vec!["<cls>".to_string()],
            ..VocabConfig::default()
        },
    );
    let merged = Vocabulary::merge_vocabularies(&[v1, v2]);
    assert_eq!(merged.config().additional_special_tokens.len(), 2);
}

#[test]
fn merge_deduplicates_additional_special_tokens() {
    let v1 = Vocabulary::new(
        HashMap::new(),
        VocabConfig {
            additional_special_tokens: vec!["<sep>".to_string()],
            ..VocabConfig::default()
        },
    );
    let v2 = Vocabulary::new(
        HashMap::new(),
        VocabConfig {
            additional_special_tokens: vec!["<sep>".to_string()],
            ..VocabConfig::default()
        },
    );
    let merged = Vocabulary::merge_vocabularies(&[v1, v2]);
    assert_eq!(merged.config().additional_special_tokens.len(), 1);
}

#[test]
fn merged_ids_are_dense() {
    let v1 = {
        let map: HashMap<String, u32> =
            [("a", 10), ("b", 20)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        Vocabulary::new(map, VocabConfig::default())
    };
    let merged = Vocabulary::merge_vocabularies(&[v1]);
    // After merge, IDs should be reassigned 0, 1 (dense)
    let ids: Vec<u32> = merged.iter().map(|(_, id)| id).collect();
    assert!(ids.contains(&0));
    assert!(ids.contains(&1));
    assert!(!ids.contains(&10));
}

// ---------------------------------------------------------------------------
// from_json
// ---------------------------------------------------------------------------

#[test]
fn from_json_basic() {
    let json = r#"{
        "model": {
            "vocab": {
                "hello": 0,
                "world": 1
            }
        }
    }"#;
    let vocab = Vocabulary::from_json(json).unwrap();
    assert_eq!(vocab.vocab_size(), 2);
    assert_eq!(vocab.token_to_id("hello"), Some(0));
}

#[test]
fn from_json_with_added_tokens() {
    let json = r#"{
        "model": {
            "vocab": {
                "a": 0,
                "b": 1
            }
        },
        "added_tokens": [
            {"content": "<bos>", "id": 2, "special": true},
            {"content": "<eos>", "id": 3, "special": true},
            {"content": "extra", "id": 4, "special": false}
        ]
    }"#;
    let vocab = Vocabulary::from_json(json).unwrap();
    assert_eq!(vocab.vocab_size(), 5);
    assert_eq!(vocab.token_to_id("<bos>"), Some(2));
    // additional_special_tokens should include bos and eos but not "extra"
    assert!(vocab.config().additional_special_tokens.contains(&"<bos>".to_string()));
    assert!(!vocab.config().additional_special_tokens.contains(&"extra".to_string()));
}

#[test]
fn from_json_invalid_json_returns_error() {
    let result = Vocabulary::from_json("not valid json");
    assert!(result.is_err());
}

#[test]
fn from_json_missing_model_key_returns_error() {
    let json = r#"{"vocab": {"a": 0}}"#;
    let result = Vocabulary::from_json(json);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// VocabConfig serde roundtrip
// ---------------------------------------------------------------------------

#[test]
fn vocab_config_serde_roundtrip() {
    let cfg = VocabConfig {
        unk_token: Some("<unk>".to_string()),
        bos_token: Some("<s>".to_string()),
        eos_token: Some("</s>".to_string()),
        pad_token: Some("<pad>".to_string()),
        additional_special_tokens: vec!["<sep>".to_string()],
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: VocabConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg2.unk_token, cfg.unk_token);
    assert_eq!(cfg2.bos_token, cfg.bos_token);
    assert_eq!(cfg2.eos_token, cfg.eos_token);
    assert_eq!(cfg2.pad_token, cfg.pad_token);
    assert_eq!(cfg2.additional_special_tokens, cfg.additional_special_tokens);
}

// ---------------------------------------------------------------------------
// ModelCompatibilityMatrix
// ---------------------------------------------------------------------------

#[test]
fn compatibility_matrix_default_has_llama3() {
    use bitnet_tokenizers::discovery::ModelCompatibilityMatrix;
    let matrix = ModelCompatibilityMatrix::default();
    assert!(matrix.llama3_128k.repo.contains("llama"));
    assert_eq!(matrix.llama3_128k.expected_vocab, Some(128256));
}

#[test]
fn compatibility_matrix_default_has_phi4() {
    use bitnet_tokenizers::discovery::ModelCompatibilityMatrix;
    let matrix = ModelCompatibilityMatrix::default();
    assert!(matrix.phi4_100k.repo.contains("phi-4"));
    assert_eq!(matrix.phi4_100k.expected_vocab, Some(100352));
}

#[test]
fn compatibility_matrix_default_has_qwen2() {
    use bitnet_tokenizers::discovery::ModelCompatibilityMatrix;
    let matrix = ModelCompatibilityMatrix::default();
    assert!(matrix.qwen2_150k.repo.contains("Qwen"));
    assert_eq!(matrix.qwen2_150k.expected_vocab, Some(151936));
}

#[test]
fn compatibility_matrix_cache_keys_unique() {
    use bitnet_tokenizers::discovery::ModelCompatibilityMatrix;
    let matrix = ModelCompatibilityMatrix::default();
    let keys = vec![
        &matrix.llama3_128k.cache_key,
        &matrix.llama2_32k.cache_key,
        &matrix.gpt2_50k.cache_key,
        &matrix.phi4_100k.cache_key,
        &matrix.qwen2_150k.cache_key,
        &matrix.gemma_256k.cache_key,
        &matrix.mistral_32k.cache_key,
        &matrix.deepseek_100k.cache_key,
        &matrix.starcoder_49k.cache_key,
        &matrix.falcon_65k.cache_key,
    ];
    let mut seen = std::collections::HashSet::new();
    for key in &keys {
        assert!(seen.insert(key.as_str()), "duplicate cache_key: {}", key);
    }
}

// ---------------------------------------------------------------------------
// Clone semantics
// ---------------------------------------------------------------------------

#[test]
fn vocabulary_clone_is_independent() {
    let vocab = simple_vocab();
    let cloned = vocab.clone();
    assert_eq!(cloned.vocab_size(), vocab.vocab_size());
    assert_eq!(cloned.token_to_id("hello"), vocab.token_to_id("hello"));
}

// ---------------------------------------------------------------------------
// Large vocabulary
// ---------------------------------------------------------------------------

#[test]
fn large_vocabulary_lookups() {
    let n = 10_000;
    let map: HashMap<String, u32> = (0..n).map(|i| (format!("tok_{i}"), i)).collect();
    let vocab = Vocabulary::new(map, VocabConfig::default());
    assert_eq!(vocab.vocab_size(), n as usize);
    assert_eq!(vocab.token_to_id("tok_0"), Some(0));
    assert_eq!(vocab.token_to_id("tok_9999"), Some(9999));
    assert_eq!(vocab.id_to_token(5000), Some("tok_5000"));
}
