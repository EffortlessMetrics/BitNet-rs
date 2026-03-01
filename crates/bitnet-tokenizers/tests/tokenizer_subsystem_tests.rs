//! Tests for tokenizer subsystem components that don't require model files.
//!
//! Covers MockTokenizer roundtrip, ModelTypeDetector, ModelCompatibilityMatrix
//! structure, Vocabulary operations, and token estimation utilities.

use bitnet_tokenizers::discovery::ModelCompatibilityMatrix;
use bitnet_tokenizers::error_handling::ModelTypeDetector;
use bitnet_tokenizers::MockTokenizer;
use bitnet_tokenizers::utils::{estimate_tokens, validate_roundtrip};
use bitnet_tokenizers::vocabulary::{VocabConfig, Vocabulary};
use bitnet_tokenizers::Tokenizer;
use std::collections::HashMap;

// --- MockTokenizer tests ---

#[test]
fn mock_tokenizer_encode_decode_ascii_roundtrip() {
    let tok = MockTokenizer::new();
    let text = "Hello, world!";
    let encoded = tok.encode(text, false, false).unwrap();
    let decoded = tok.decode(&encoded).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn mock_tokenizer_encode_decode_utf8_roundtrip() {
    let tok = MockTokenizer::new();
    let text = "Héllo wörld";
    let encoded = tok.encode(text, false, false).unwrap();
    let decoded = tok.decode(&encoded).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn mock_tokenizer_empty_string() {
    let tok = MockTokenizer::new();
    let encoded = tok.encode("", false, false).unwrap();
    assert!(encoded.is_empty());
    let decoded = tok.decode(&[]).unwrap();
    assert!(decoded.is_empty());
}

#[test]
fn mock_tokenizer_single_char() {
    let tok = MockTokenizer::new();
    let encoded = tok.encode("a", false, false).unwrap();
    assert_eq!(encoded, vec![97]); // 'a' = 97
}

#[test]
fn mock_tokenizer_vocab_size() {
    let tok = MockTokenizer::new();
    assert_eq!(tok.vocab_size(), 50257);
}

#[test]
fn mock_tokenizer_token_to_piece_ascii() {
    let tok = MockTokenizer::new();
    assert_eq!(tok.token_to_piece(65), Some("A".to_string()));
}

#[test]
fn mock_tokenizer_token_to_piece_special() {
    let tok = MockTokenizer::new();
    let piece = tok.token_to_piece(1000);
    assert_eq!(piece, Some("<token_1000>".to_string()));
}

#[test]
fn mock_tokenizer_with_special_tokens() {
    let tok = MockTokenizer::with_special_tokens(&[("<bos>", 1), ("<eos>", 2), ("<pad>", 0)]);
    assert_eq!(tok.token_to_id("<bos>"), Some(1));
    assert_eq!(tok.token_to_id("<eos>"), Some(2));
    assert_eq!(tok.token_to_id("<pad>"), Some(0));
    assert_eq!(tok.token_to_id("<unknown>"), None);
}

#[test]
fn mock_tokenizer_default_same_as_new() {
    let new_tok = MockTokenizer::new();
    let default_tok = MockTokenizer::default();
    assert_eq!(new_tok.vocab_size(), default_tok.vocab_size());
}

#[test]
fn mock_tokenizer_ignores_bos_flag() {
    let tok = MockTokenizer::new();
    let with_bos = tok.encode("test", true, false).unwrap();
    let without_bos = tok.encode("test", false, false).unwrap();
    assert_eq!(with_bos, without_bos);
}

// --- ModelTypeDetector tests ---

#[test]
fn detect_llama2_from_vocab_size() {
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(32000), "llama2");
}

#[test]
fn detect_llama3_from_vocab_size() {
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(128256), "llama3");
}

#[test]
fn detect_codellama_from_vocab_size() {
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(32016), "codellama");
}

#[test]
fn detect_gpt2_from_vocab_size() {
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(50257), "gpt2");
}

#[test]
fn detect_unknown_from_unrecognized_vocab_size() {
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(100352), "unknown");
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(1), "unknown");
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(999999), "unknown");
}

#[test]
fn requires_gpu_for_large_vocab() {
    assert!(ModelTypeDetector::requires_gpu_acceleration(65537));
    assert!(ModelTypeDetector::requires_gpu_acceleration(128256));
    assert!(ModelTypeDetector::requires_gpu_acceleration(100000));
}

#[test]
fn no_gpu_for_small_vocab() {
    assert!(!ModelTypeDetector::requires_gpu_acceleration(32000));
    assert!(!ModelTypeDetector::requires_gpu_acceleration(50257));
    assert!(!ModelTypeDetector::requires_gpu_acceleration(65536));
}

#[test]
fn validate_vocab_size_zero() {
    assert!(ModelTypeDetector::validate_vocab_size(0).is_err());
}

#[test]
fn validate_vocab_size_too_large() {
    assert!(ModelTypeDetector::validate_vocab_size(2_000_001).is_err());
}

#[test]
fn validate_vocab_size_at_max() {
    assert!(ModelTypeDetector::validate_vocab_size(2_000_000).is_ok());
}

#[test]
fn validate_vocab_size_reasonable() {
    assert!(ModelTypeDetector::validate_vocab_size(32000).is_ok());
    assert!(ModelTypeDetector::validate_vocab_size(128256).is_ok());
    assert!(ModelTypeDetector::validate_vocab_size(1).is_ok());
}

#[test]
fn expected_vocab_size_known_models() {
    assert_eq!(ModelTypeDetector::expected_vocab_size("llama2"), Some(32000));
    assert_eq!(ModelTypeDetector::expected_vocab_size("llama3"), Some(128256));
    assert_eq!(ModelTypeDetector::expected_vocab_size("codellama"), Some(32016));
    assert_eq!(ModelTypeDetector::expected_vocab_size("gpt2"), Some(50257));
}

#[test]
fn expected_vocab_size_unknown() {
    assert_eq!(ModelTypeDetector::expected_vocab_size("phi4"), None);
    assert_eq!(ModelTypeDetector::expected_vocab_size("gemma"), None);
}

// --- ModelCompatibilityMatrix tests ---

#[test]
fn compatibility_matrix_default_has_all_entries() {
    let matrix = ModelCompatibilityMatrix::default();

    // Key entries should have non-empty repo strings
    assert!(!matrix.llama3_128k.repo.is_empty());
    assert!(!matrix.llama2_32k.repo.is_empty());
    assert!(!matrix.gpt2_50k.repo.is_empty());
    assert!(!matrix.phi4_100k.repo.is_empty());
    assert!(!matrix.qwen2_150k.repo.is_empty());
    assert!(!matrix.gemma_256k.repo.is_empty());
    assert!(!matrix.mistral_32k.repo.is_empty());
    assert!(!matrix.deepseek_100k.repo.is_empty());
}

#[test]
fn compatibility_matrix_expected_vocabs() {
    let matrix = ModelCompatibilityMatrix::default();

    assert_eq!(matrix.llama3_128k.expected_vocab, Some(128256));
    assert_eq!(matrix.llama2_32k.expected_vocab, Some(32000));
    assert_eq!(matrix.gpt2_50k.expected_vocab, Some(50257));
}

#[test]
fn compatibility_matrix_all_have_cache_keys() {
    let matrix = ModelCompatibilityMatrix::default();

    assert!(!matrix.llama3_128k.cache_key.is_empty());
    assert!(!matrix.phi4_100k.cache_key.is_empty());
    assert!(!matrix.gemma_256k.cache_key.is_empty());
    assert!(!matrix.qwen2_150k.cache_key.is_empty());
}

#[test]
fn compatibility_matrix_all_have_files() {
    let matrix = ModelCompatibilityMatrix::default();

    assert!(!matrix.llama3_128k.files.is_empty());
    assert!(!matrix.phi4_100k.files.is_empty());
    assert!(!matrix.mistral_32k.files.is_empty());
}

#[test]
fn compatibility_matrix_new_slm_entries_present() {
    let matrix = ModelCompatibilityMatrix::default();

    // Verify newer SLM entries added during multi-SLM expansion
    assert!(!matrix.falcon_65k.repo.is_empty());
    assert!(!matrix.codellama_32k.repo.is_empty());
    assert!(!matrix.command_256k.repo.is_empty());
    assert!(!matrix.internlm_103k.repo.is_empty());
    assert!(!matrix.yi_64k.repo.is_empty());
    assert!(!matrix.baichuan_64k.repo.is_empty());
    assert!(!matrix.chatglm_65k.repo.is_empty());
    assert!(!matrix.mpt_50k.repo.is_empty());
    assert!(!matrix.rwkv_65k.repo.is_empty());
    assert!(!matrix.olmo_50k.repo.is_empty());
}

#[test]
fn compatibility_matrix_wave12_entries_present() {
    let matrix = ModelCompatibilityMatrix::default();

    // Wave 12+ entries
    assert!(!matrix.tinyllama_32k.repo.is_empty());
    assert!(!matrix.dolphin_32k.repo.is_empty());
    assert!(!matrix.mixtral_32k.repo.is_empty());
    assert!(!matrix.stablelm_32k.repo.is_empty());
    assert!(!matrix.bloom_250k.repo.is_empty());
    assert!(!matrix.jamba_256k.repo.is_empty());
    assert!(!matrix.dbrx_32k.repo.is_empty());
    assert!(!matrix.exaone_32k.repo.is_empty());
    assert!(!matrix.minicpm_122k.repo.is_empty());
}

#[test]
fn compatibility_matrix_latest_entries_present() {
    let matrix = ModelCompatibilityMatrix::default();

    // Latest additions
    assert!(!matrix.codegemma_256k.repo.is_empty());
    assert!(!matrix.llama31_128k.repo.is_empty());
    assert!(!matrix.deepseekv3_100k.repo.is_empty());
    assert!(!matrix.aya_256k.repo.is_empty());
    assert!(!matrix.smollm_49k.repo.is_empty());
    assert!(!matrix.phi2_51k.repo.is_empty());
    assert!(!matrix.falcon2_32k.repo.is_empty());
    assert!(!matrix.olmo2_100k.repo.is_empty());
    assert!(!matrix.llama32_128k.repo.is_empty());
}

// --- Vocabulary tests ---

#[test]
fn vocabulary_from_hashmap() {
    let mut token_to_id = HashMap::new();
    token_to_id.insert("hello".to_string(), 0);
    token_to_id.insert("world".to_string(), 1);
    token_to_id.insert("<unk>".to_string(), 2);

    let config = VocabConfig {
        unk_token: Some("<unk>".to_string()),
        bos_token: None,
        eos_token: None,
        pad_token: None,
        additional_special_tokens: vec![],
    };

    let vocab = Vocabulary::new(token_to_id, config);
    assert_eq!(vocab.vocab_size(), 3);
    assert_eq!(vocab.token_to_id("hello"), Some(0));
    assert_eq!(vocab.token_to_id("world"), Some(1));
    assert_eq!(vocab.id_to_token(0), Some("hello"));
    assert_eq!(vocab.id_to_token(1), Some("world"));
}

#[test]
fn vocabulary_contains() {
    let mut token_to_id = HashMap::new();
    token_to_id.insert("test".to_string(), 0);

    let vocab = Vocabulary::new(token_to_id, VocabConfig::default());
    assert!(vocab.contains("test"));
    assert!(!vocab.contains("missing"));
}

#[test]
fn vocabulary_is_special_token() {
    let mut token_to_id = HashMap::new();
    token_to_id.insert("<bos>".to_string(), 0);
    token_to_id.insert("hello".to_string(), 1);

    let config = VocabConfig {
        bos_token: Some("<bos>".to_string()),
        ..VocabConfig::default()
    };

    let vocab = Vocabulary::new(token_to_id, config);
    assert!(vocab.is_special_token(0));
    assert!(!vocab.is_special_token(1));
}

#[test]
fn vocabulary_all_special_tokens() {
    let mut token_to_id = HashMap::new();
    token_to_id.insert("<s>".to_string(), 0);
    token_to_id.insert("</s>".to_string(), 1);
    token_to_id.insert("<unk>".to_string(), 2);
    token_to_id.insert("<pad>".to_string(), 3);
    token_to_id.insert("hello".to_string(), 4);

    let config = VocabConfig {
        bos_token: Some("<s>".to_string()),
        eos_token: Some("</s>".to_string()),
        unk_token: Some("<unk>".to_string()),
        pad_token: Some("<pad>".to_string()),
        additional_special_tokens: vec![],
    };

    let vocab = Vocabulary::new(token_to_id, config);
    let st = vocab.special_tokens();
    assert_eq!(st.bos_id, Some(0));
    assert_eq!(st.eos_id, Some(1));
    assert_eq!(st.unk_id, Some(2));
    assert_eq!(st.pad_id, Some(3));
    assert!(vocab.is_special_token(0));
    assert!(vocab.is_special_token(1));
    assert!(vocab.is_special_token(2));
    assert!(vocab.is_special_token(3));
    assert!(!vocab.is_special_token(4));
}

#[test]
fn vocabulary_iter() {
    let mut token_to_id = HashMap::new();
    token_to_id.insert("a".to_string(), 0);
    token_to_id.insert("b".to_string(), 1);

    let vocab = Vocabulary::new(token_to_id, VocabConfig::default());
    let items: Vec<_> = vocab.iter().collect();
    assert_eq!(items.len(), 2);
}

#[test]
fn vocabulary_merge_disjoint() {
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
fn vocabulary_merge_overlap_deduplicates() {
    let v1 = Vocabulary::new(
        HashMap::from([("a".into(), 0), ("b".into(), 1)]),
        VocabConfig::default(),
    );
    let v2 = Vocabulary::new(
        HashMap::from([("b".into(), 0), ("c".into(), 1)]),
        VocabConfig::default(),
    );
    let merged = Vocabulary::merge_vocabularies(&[v1, v2]);
    assert_eq!(merged.vocab_size(), 3);
}

#[test]
fn vocabulary_from_json() {
    let json = r#"{
        "model": { "vocab": { "hello": 0, "world": 1 } }
    }"#;
    let v = Vocabulary::from_json(json).unwrap();
    assert_eq!(v.vocab_size(), 2);
    assert_eq!(v.token_to_id("hello"), Some(0));
}

#[test]
fn vocabulary_from_json_with_special_added_tokens() {
    let json = r#"{
        "model": { "vocab": { "hi": 0 } },
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
fn vocabulary_from_json_invalid() {
    assert!(Vocabulary::from_json("not json").is_err());
}

// --- Token estimation tests ---

#[test]
fn estimate_tokens_empty() {
    assert_eq!(estimate_tokens(""), 0);
}

#[test]
fn estimate_tokens_short_text() {
    let estimate = estimate_tokens("Hello, world!");
    assert!(estimate > 0);
    assert!(estimate < 20);
}

#[test]
fn estimate_tokens_long_text() {
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let estimate = estimate_tokens(&text);
    assert!(estimate > 100);
    assert!(estimate < 5000);
}

// --- Validate roundtrip utility ---

#[test]
fn validate_roundtrip_with_mock_tokenizer() {
    let tok = MockTokenizer::new();
    assert!(validate_roundtrip(&tok, "Hello, world!"));
    assert!(validate_roundtrip(&tok, "test"));
    assert!(validate_roundtrip(&tok, ""));
}

#[test]
fn validate_roundtrip_unicode() {
    let tok = MockTokenizer::new();
    assert!(validate_roundtrip(&tok, "日本語"));
    assert!(validate_roundtrip(&tok, "émojis: 🎉🚀"));
}
