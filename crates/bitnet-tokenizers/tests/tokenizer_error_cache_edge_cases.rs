//! Edge-case tests for tokenizer error handling, cache management, model detection,
//! and strategy/config types.

use bitnet_common::{BitNetError, ModelError};
use bitnet_tokenizers::Tokenizer;
use bitnet_tokenizers::discovery::ModelCompatibilityMatrix;
use bitnet_tokenizers::{
    BasicTokenizer, CacheManager, ModelTypeDetector, TokenizerBuilder, TokenizerConfig,
    TokenizerDownloadInfo, TokenizerErrorHandler, TokenizerStrategy,
};
use std::path::{Path, PathBuf};

// ── TokenizerErrorHandler ────────────────────────────────────────────

#[test]
fn error_handler_file_io_error_preserves_path() {
    let path = PathBuf::from("missing/tokenizer.json");
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
    let err = TokenizerErrorHandler::file_io_error(&path, io_err);
    match err {
        BitNetError::Model(ModelError::FileIOError { path: p, .. }) => {
            assert_eq!(p, path);
        }
        other => panic!("expected FileIOError, got {:?}", other),
    }
}

#[test]
fn error_handler_loading_failed_preserves_reason() {
    let err = TokenizerErrorHandler::loading_failed_error("bad vocab".into());
    match err {
        BitNetError::Model(ModelError::LoadingFailed { reason }) => {
            assert_eq!(reason, "bad vocab");
        }
        other => panic!("expected LoadingFailed, got {:?}", other),
    }
}

#[test]
fn error_handler_config_error_preserves_message() {
    let err = TokenizerErrorHandler::config_error("invalid setting".into());
    match err {
        BitNetError::Config(msg) => assert_eq!(msg, "invalid setting"),
        other => panic!("expected Config, got {:?}", other),
    }
}

#[test]
fn validate_file_exists_nonexistent_path_returns_err() {
    let result = TokenizerErrorHandler::validate_file_exists(
        Path::new("surely/does/not/exist.json"),
        "test",
    );
    assert!(result.is_err());
}

#[test]
fn validate_file_exists_with_real_file_returns_ok() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let result = TokenizerErrorHandler::validate_file_exists(tmp.path(), "test");
    assert!(result.is_ok());
}

#[test]
fn validate_file_exists_directory_returns_err() {
    let dir = tempfile::tempdir().unwrap();
    let result = TokenizerErrorHandler::validate_file_exists(dir.path(), "test");
    assert!(result.is_err(), "a directory is not a regular file");
}

// ── CacheManager ─────────────────────────────────────────────────────

#[test]
fn cache_directory_returns_path_containing_tokenizers() {
    let path = CacheManager::cache_directory().unwrap();
    assert!(
        path.to_string_lossy().contains("tokenizers"),
        "cache path should contain 'tokenizers': {}",
        path.display()
    );
}

#[test]
fn ensure_cache_directory_creates_missing_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let nested = tmp.path().join("a").join("b").join("c");
    assert!(!nested.exists());
    CacheManager::ensure_cache_directory(&nested).unwrap();
    assert!(nested.exists());
}

#[test]
fn ensure_cache_directory_existing_dir_is_ok() {
    let tmp = tempfile::tempdir().unwrap();
    // Call twice — second call must not fail.
    CacheManager::ensure_cache_directory(tmp.path()).unwrap();
    CacheManager::ensure_cache_directory(tmp.path()).unwrap();
}

#[test]
fn model_cache_dir_with_vocab_includes_segments() {
    let p = CacheManager::model_cache_dir("llama2", Some(32000)).unwrap();
    let s = p.to_string_lossy();
    assert!(s.contains("llama2"), "path should include model type");
    assert!(s.contains("vocab_32000"), "path should include vocab size");
}

#[test]
fn model_cache_dir_without_vocab() {
    let p = CacheManager::model_cache_dir("gpt2", None).unwrap();
    let s = p.to_string_lossy();
    assert!(s.contains("gpt2"));
    assert!(!s.contains("vocab_"), "no vocab segment when None");
}

// ── ModelTypeDetector ────────────────────────────────────────────────

#[test]
fn detect_from_vocab_size_known_models() {
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(32000), "llama2");
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(128256), "llama3");
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(32016), "codellama");
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(50257), "gpt2");
}

#[test]
fn detect_from_vocab_size_unknown() {
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(100352), "unknown");
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(1), "unknown");
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(999999), "unknown");
}

#[test]
fn requires_gpu_acceleration_threshold() {
    assert!(!ModelTypeDetector::requires_gpu_acceleration(32000));
    assert!(!ModelTypeDetector::requires_gpu_acceleration(65536));
    assert!(ModelTypeDetector::requires_gpu_acceleration(65537));
    assert!(ModelTypeDetector::requires_gpu_acceleration(128256));
}

#[test]
fn validate_vocab_size_boundaries() {
    assert!(ModelTypeDetector::validate_vocab_size(0).is_err());
    assert!(ModelTypeDetector::validate_vocab_size(1).is_ok());
    assert!(ModelTypeDetector::validate_vocab_size(2_000_000).is_ok());
    assert!(ModelTypeDetector::validate_vocab_size(2_000_001).is_err());
}

#[test]
fn expected_vocab_size_roundtrips() {
    for model in &["llama2", "llama3", "codellama", "gpt2"] {
        let size = ModelTypeDetector::expected_vocab_size(model).unwrap();
        assert_eq!(&ModelTypeDetector::detect_from_vocab_size(size), *model);
    }
    assert_eq!(ModelTypeDetector::expected_vocab_size("phi"), None);
}

// ── TokenizerStrategy ────────────────────────────────────────────────

#[test]
fn strategy_exact_properties() {
    let s = TokenizerStrategy::Exact(PathBuf::from("tok.json"));
    assert!(!s.requires_network());
    assert!(!s.uses_cache());
    assert_eq!(s.description(), "user-specified tokenizer");
}

#[test]
fn strategy_discovered_properties() {
    let s = TokenizerStrategy::Discovered(PathBuf::from("tok.json"));
    assert!(!s.requires_network());
    assert!(s.uses_cache());
    assert_eq!(s.description(), "auto-discovered tokenizer");
}

#[test]
fn strategy_needs_download_properties() {
    let info = TokenizerDownloadInfo {
        repo: "meta-llama/Meta-Llama-3-8B".into(),
        files: vec!["tokenizer.json".into()],
        cache_key: "llama3".into(),
        expected_vocab: Some(128256),
    };
    let s = TokenizerStrategy::NeedsDownload(info);
    assert!(s.requires_network());
    assert!(s.uses_cache());
    assert_eq!(s.description(), "smart download required");
}

#[test]
fn strategy_embedded_gguf_properties() {
    let tok: std::sync::Arc<dyn Tokenizer> = std::sync::Arc::new(BasicTokenizer::new());
    let s = TokenizerStrategy::EmbeddedGguf(tok);
    assert!(!s.requires_network());
    assert!(!s.uses_cache());
    assert_eq!(s.description(), "GGUF-embedded tokenizer");
}

#[test]
fn strategy_mock_properties() {
    let s = TokenizerStrategy::Mock;
    assert!(!s.requires_network());
    assert!(!s.uses_cache());
    assert_eq!(s.description(), "mock tokenizer (testing only)");
}

// ── TokenizerDownloadInfo ────────────────────────────────────────────

#[test]
fn download_info_construction_and_clone() {
    let info = TokenizerDownloadInfo {
        repo: "org/repo".into(),
        files: vec!["a.json".into(), "b.model".into()],
        cache_key: "test-key".into(),
        expected_vocab: Some(50257),
    };
    let cloned = info.clone();
    assert_eq!(cloned.repo, "org/repo");
    assert_eq!(cloned.files.len(), 2);
    assert_eq!(cloned.cache_key, "test-key");
    assert_eq!(cloned.expected_vocab, Some(50257));
}

#[test]
fn download_info_none_vocab() {
    let info = TokenizerDownloadInfo {
        repo: "x/y".into(),
        files: vec![],
        cache_key: "k".into(),
        expected_vocab: None,
    };
    assert!(info.expected_vocab.is_none());
    assert!(info.files.is_empty());
}

// ── BasicTokenizer ───────────────────────────────────────────────────

#[test]
fn basic_tokenizer_defaults() {
    let tok = BasicTokenizer::new();
    assert_eq!(tok.vocab_size(), 50257);
    assert_eq!(tok.eos_token_id(), Some(50256));
    assert_eq!(tok.bos_token_id(), None);
    assert_eq!(tok.pad_token_id(), None);
}

#[test]
fn basic_tokenizer_encode_empty() {
    let tok = BasicTokenizer::new();
    assert!(tok.encode("", false, false).unwrap().is_empty());
    assert!(tok.encode("", true, true).unwrap().is_empty());
}

#[test]
fn basic_tokenizer_encode_ascii() {
    let tok = BasicTokenizer::new();
    let tokens = tok.encode("Hi", false, false).unwrap();
    assert_eq!(tokens, vec![b'H' as u32, b'i' as u32]);
}

#[test]
fn basic_tokenizer_encode_with_bos_eos() {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), None);
    let tokens = tok.encode("A", true, true).unwrap();
    assert_eq!(tokens.first(), Some(&1)); // BOS
    assert_eq!(tokens.last(), Some(&2)); // EOS
    assert_eq!(tokens[1], b'A' as u32);
}

#[test]
fn basic_tokenizer_decode_roundtrip() {
    let tok = BasicTokenizer::new();
    let text = "hello world";
    let tokens = tok.encode(text, false, false).unwrap();
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn basic_tokenizer_decode_skips_special() {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), Some(3));
    let decoded = tok.decode(&[1, b'X' as u32, 2, 3]).unwrap();
    assert_eq!(decoded, "X");
}

#[test]
fn basic_tokenizer_decode_empty() {
    let tok = BasicTokenizer::new();
    assert_eq!(tok.decode(&[]).unwrap(), "");
}

#[test]
fn basic_tokenizer_token_to_piece() {
    let tok = BasicTokenizer::new();
    assert_eq!(tok.token_to_piece(65), Some("A".into()));
    assert_eq!(tok.token_to_piece(500), Some("<token_500>".into()));
}

#[test]
fn basic_tokenizer_is_special_token() {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), Some(3));
    assert!(tok.is_special_token(1));
    assert!(tok.is_special_token(2));
    assert!(tok.is_special_token(3));
    assert!(!tok.is_special_token(65));
}

#[test]
fn basic_tokenizer_family_name_is_unknown() {
    let tok = BasicTokenizer::new();
    assert_eq!(tok.get_family_name(), "unknown");
}

#[test]
fn basic_tokenizer_real_vocab_equals_vocab() {
    let tok = BasicTokenizer::with_config(1000, None, None, None);
    assert_eq!(tok.real_vocab_size(), tok.vocab_size());
}

// ── TokenizerConfig ──────────────────────────────────────────────────

#[test]
fn tokenizer_config_default_values() {
    let cfg = TokenizerConfig::default();
    assert_eq!(cfg.vocab_size, 0);
    assert!(cfg.model_type.is_empty());
    assert!(!cfg.add_bos);
    assert!(!cfg.add_eos);
    assert!(!cfg.add_space_prefix);
    assert!(!cfg.byte_fallback);
    assert!(cfg.bos_token_id.is_none());
    assert!(cfg.eos_token_id.is_none());
    assert!(cfg.pad_token_id.is_none());
    assert!(cfg.unk_token_id.is_none());
    assert!(cfg.vocabulary.is_none());
    assert!(cfg.bpe_merges.is_none());
    assert!(cfg.pre_tokenizer.is_none());
}

#[test]
fn tokenizer_config_new_equals_default() {
    let a = TokenizerConfig::new();
    let b = TokenizerConfig::default();
    assert_eq!(a.vocab_size, b.vocab_size);
    assert_eq!(a.model_type, b.model_type);
}

#[test]
fn tokenizer_config_field_access() {
    let cfg = TokenizerConfig {
        model_type: "llama3".into(),
        vocab_size: 128256,
        add_bos: true,
        add_eos: true,
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        ..Default::default()
    };
    assert_eq!(cfg.model_type, "llama3");
    assert_eq!(cfg.vocab_size, 128256);
    assert!(cfg.add_bos);
    assert_eq!(cfg.bos_token_id, Some(1));
}

// ── ModelCompatibilityMatrix ─────────────────────────────────────────

#[test]
fn compatibility_matrix_llama3_entry() {
    let m = ModelCompatibilityMatrix::default();
    assert_eq!(m.llama3_128k.repo, "meta-llama/Meta-Llama-3-8B");
    assert!(m.llama3_128k.files.contains(&"tokenizer.json".to_string()));
    assert_eq!(m.llama3_128k.expected_vocab, Some(128256));
}

#[test]
fn compatibility_matrix_gpt2_entry() {
    let m = ModelCompatibilityMatrix::default();
    assert_eq!(m.gpt2_50k.repo, "openai-community/gpt2");
    assert_eq!(m.gpt2_50k.expected_vocab, Some(50257));
}

#[test]
fn compatibility_matrix_phi4_entry() {
    let m = ModelCompatibilityMatrix::default();
    assert_eq!(m.phi4_100k.repo, "microsoft/phi-4");
    assert_eq!(m.phi4_100k.expected_vocab, Some(100352));
}

#[test]
fn compatibility_matrix_clone() {
    let m = ModelCompatibilityMatrix::default();
    let c = m.clone();
    assert_eq!(c.llama3_128k.repo, m.llama3_128k.repo);
    assert_eq!(c.gpt2_50k.expected_vocab, m.gpt2_50k.expected_vocab);
}

// ── TokenizerStrategy::Exact / Discovered with temp paths ────────────

#[test]
fn strategy_exact_with_tempfile() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let s = TokenizerStrategy::Exact(tmp.path().to_path_buf());
    assert!(!s.requires_network());
    assert_eq!(s.description(), "user-specified tokenizer");
}

#[test]
fn strategy_discovered_with_tempfile() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let s = TokenizerStrategy::Discovered(tmp.path().to_path_buf());
    assert!(s.uses_cache());
    assert_eq!(s.description(), "auto-discovered tokenizer");
}

#[test]
fn strategy_exact_nonexistent_path() {
    let s = TokenizerStrategy::Exact(PathBuf::from("nonexistent/path/tok.json"));
    // Strategy itself doesn't validate the file; it's just metadata.
    assert!(!s.requires_network());
    assert_eq!(s.description(), "user-specified tokenizer");
}

// ── TokenizerBuilder ─────────────────────────────────────────────────

#[test]
fn builder_from_pretrained_gpt2() {
    let tok = TokenizerBuilder::from_pretrained("gpt2").unwrap();
    assert_eq!(tok.vocab_size(), 50257);
    assert_eq!(tok.eos_token_id(), Some(50256));
}

#[test]
fn builder_from_pretrained_unknown_returns_default() {
    let tok = TokenizerBuilder::from_pretrained("completely-unknown-model").unwrap();
    assert_eq!(tok.vocab_size(), 50257); // BasicTokenizer default
}

#[test]
fn builder_from_file_nonexistent_returns_err() {
    let result = TokenizerBuilder::from_file("nonexistent/tokenizer.json");
    assert!(result.is_err());
}

// ── to_anyhow_error / create_actionable_error ────────────────────────

#[test]
fn to_anyhow_error_contains_context() {
    let err = BitNetError::Config("test".into());
    let anyhow_err = TokenizerErrorHandler::to_anyhow_error(err, "context msg");
    let s = format!("{}", anyhow_err);
    assert!(s.contains("context msg") || s.contains("test"));
}

#[test]
fn create_actionable_error_returns_err_with_suggestions() {
    let err = BitNetError::Model(ModelError::LoadingFailed { reason: "network failure".into() });
    let result = TokenizerErrorHandler::create_actionable_error(err, "download");
    assert!(result.is_err());
    let chain = format!("{:#}", result.unwrap_err());
    assert!(chain.contains("download"));
    assert!(chain.contains("Suggestion"));
}
