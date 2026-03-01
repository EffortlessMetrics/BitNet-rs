//! Edge-case tests for tokenizer types: MockTokenizer, BasicTokenizer,
//! TokenizerConfig, ModelTypeDetector, FallbackStrategy, FallbackError,
//! TokenizerResolution, LlamaVariant, estimate_tokens, and DownloadProgress.

use bitnet_tokenizers::error_handling::ModelTypeDetector;
use bitnet_tokenizers::fallback::{FallbackError, FallbackStrategy, TokenizerResolution};
use bitnet_tokenizers::strategy::LlamaVariant;
use bitnet_tokenizers::utils::estimate_tokens;
use bitnet_tokenizers::{
    BasicTokenizer, DownloadProgress, MockTokenizer, Tokenizer, TokenizerConfig,
};
use std::path::PathBuf;

// ===========================================================================
// MockTokenizer
// ===========================================================================

#[test]
fn mock_tokenizer_default_vocab_size() {
    let tok = MockTokenizer::new();
    assert_eq!(tok.vocab_size(), 50257);
}

#[test]
fn mock_tokenizer_encode_ascii() {
    let tok = MockTokenizer::new();
    let tokens = tok.encode("Hi", false, false).unwrap();
    assert_eq!(tokens, vec![b'H' as u32, b'i' as u32]);
}

#[test]
fn mock_tokenizer_decode_roundtrip() {
    let tok = MockTokenizer::new();
    let text = "Hello, world!";
    let encoded = tok.encode(text, false, false).unwrap();
    let decoded = tok.decode(&encoded).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn mock_tokenizer_empty_string() {
    let tok = MockTokenizer::new();
    let tokens = tok.encode("", false, false).unwrap();
    assert!(tokens.is_empty());
    let decoded = tok.decode(&[]).unwrap();
    assert!(decoded.is_empty());
}

#[test]
fn mock_tokenizer_with_special_tokens() {
    let tok = MockTokenizer::with_special_tokens(&[("<bos>", 1), ("<eos>", 2)]);
    assert_eq!(tok.token_to_id("<bos>"), Some(1));
    assert_eq!(tok.token_to_id("<eos>"), Some(2));
    assert_eq!(tok.token_to_id("<unknown>"), None);
}

#[test]
fn mock_tokenizer_token_to_piece_ascii() {
    let tok = MockTokenizer::new();
    assert_eq!(tok.token_to_piece(65), Some("A".to_string()));
    assert_eq!(tok.token_to_piece(0), Some("\0".to_string()));
}

#[test]
fn mock_tokenizer_token_to_piece_special() {
    let tok = MockTokenizer::new();
    let piece = tok.token_to_piece(50000).unwrap();
    assert!(piece.starts_with("<token_"));
}

// ===========================================================================
// BasicTokenizer
// ===========================================================================

#[test]
fn basic_tokenizer_new() {
    let tok = BasicTokenizer::new();
    assert!(tok.vocab_size() > 0);
}

#[test]
fn basic_tokenizer_encode_decode() {
    let tok = BasicTokenizer::new();
    let text = "test";
    let encoded = tok.encode(text, false, false).unwrap();
    assert!(!encoded.is_empty());
    let decoded = tok.decode(&encoded).unwrap();
    assert!(!decoded.is_empty());
}

#[test]
fn basic_tokenizer_empty_string() {
    let tok = BasicTokenizer::new();
    let encoded = tok.encode("", false, false).unwrap();
    assert!(encoded.is_empty());
}

// ===========================================================================
// TokenizerConfig
// ===========================================================================

#[test]
fn tokenizer_config_default() {
    let cfg = TokenizerConfig::new();
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("TokenizerConfig"));
}

// ===========================================================================
// ModelTypeDetector
// ===========================================================================

#[test]
fn model_type_detect_llama2() {
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(32000), "llama2");
}

#[test]
fn model_type_detect_llama3() {
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(128256), "llama3");
}

#[test]
fn model_type_detect_gpt2() {
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(50257), "gpt2");
}

#[test]
fn model_type_detect_codellama() {
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(32016), "codellama");
}

#[test]
fn model_type_detect_unknown() {
    assert_eq!(ModelTypeDetector::detect_from_vocab_size(99999), "unknown");
}

#[test]
fn model_type_requires_gpu_large() {
    assert!(ModelTypeDetector::requires_gpu_acceleration(100000));
}

#[test]
fn model_type_no_gpu_small() {
    assert!(!ModelTypeDetector::requires_gpu_acceleration(32000));
}

#[test]
fn model_type_gpu_boundary() {
    assert!(!ModelTypeDetector::requires_gpu_acceleration(65536));
    assert!(ModelTypeDetector::requires_gpu_acceleration(65537));
}

#[test]
fn model_type_validate_zero() {
    assert!(ModelTypeDetector::validate_vocab_size(0).is_err());
}

#[test]
fn model_type_validate_too_large() {
    assert!(ModelTypeDetector::validate_vocab_size(3_000_000).is_err());
}

#[test]
fn model_type_validate_normal() {
    assert!(ModelTypeDetector::validate_vocab_size(32000).is_ok());
}

#[test]
fn model_type_expected_llama2() {
    assert_eq!(ModelTypeDetector::expected_vocab_size("llama2"), Some(32000));
}

#[test]
fn model_type_expected_unknown() {
    assert_eq!(ModelTypeDetector::expected_vocab_size("unknown_model"), None);
}

// ===========================================================================
// FallbackStrategy
// ===========================================================================

#[test]
fn fallback_strategy_descriptions() {
    let strategies = [
        FallbackStrategy::GgufMetadata,
        FallbackStrategy::ColocatedFiles,
        FallbackStrategy::StandardCache,
        FallbackStrategy::SmartDownload,
        FallbackStrategy::MockFallback,
    ];
    for s in &strategies {
        assert!(!s.description().is_empty());
    }
}

#[test]
fn fallback_strategy_requires_network() {
    assert!(!FallbackStrategy::GgufMetadata.requires_network());
    assert!(!FallbackStrategy::ColocatedFiles.requires_network());
    assert!(!FallbackStrategy::StandardCache.requires_network());
    assert!(FallbackStrategy::SmartDownload.requires_network());
    assert!(!FallbackStrategy::MockFallback.requires_network());
}

#[test]
fn fallback_strategy_strict_mode() {
    assert!(FallbackStrategy::GgufMetadata.allowed_in_strict_mode());
    assert!(FallbackStrategy::ColocatedFiles.allowed_in_strict_mode());
    assert!(FallbackStrategy::StandardCache.allowed_in_strict_mode());
    assert!(FallbackStrategy::SmartDownload.allowed_in_strict_mode());
    assert!(!FallbackStrategy::MockFallback.allowed_in_strict_mode());
}

#[test]
fn fallback_strategy_suggestions_nonempty() {
    let strategies = [
        FallbackStrategy::GgufMetadata,
        FallbackStrategy::ColocatedFiles,
        FallbackStrategy::StandardCache,
        FallbackStrategy::SmartDownload,
        FallbackStrategy::MockFallback,
    ];
    for s in &strategies {
        assert!(!s.failure_suggestions().is_empty());
    }
}

// ===========================================================================
// FallbackError
// ===========================================================================

#[test]
fn fallback_error_strategy_failed_display() {
    let err = FallbackError::StrategyFailed {
        strategy: FallbackStrategy::GgufMetadata,
        reason: "no tokens found".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("failed"));
    assert!(msg.contains("no tokens found"));
}

#[test]
fn fallback_error_all_failed_display() {
    let err = FallbackError::AllStrategiesFailed { summary: "tried 3 strategies".into() };
    let msg = format!("{err}");
    assert!(msg.contains("exhausted"));
}

#[test]
fn fallback_error_strict_mode_display() {
    let err = FallbackError::StrictModeViolation { reason: "mock not allowed".into() };
    let msg = format!("{err}");
    assert!(msg.contains("Strict mode"));
}

#[test]
fn fallback_error_offline_display() {
    let err = FallbackError::OfflineModeViolation { strategy: FallbackStrategy::SmartDownload };
    let msg = format!("{err}");
    assert!(msg.contains("Offline"));
}

#[test]
fn fallback_error_suggestions() {
    let err = FallbackError::AllStrategiesFailed { summary: "all failed".into() };
    let suggestions = err.suggestions();
    assert!(!suggestions.is_empty());
}

// ===========================================================================
// TokenizerResolution
// ===========================================================================

#[test]
fn tokenizer_resolution_file_description() {
    let res = TokenizerResolution::File(PathBuf::from("tokenizer.json"));
    assert_eq!(res.description(), "file-based tokenizer");
}

#[test]
fn tokenizer_resolution_mock_description() {
    let res = TokenizerResolution::Mock(MockTokenizer::new());
    assert_eq!(res.description(), "mock tokenizer");
}

#[test]
fn tokenizer_resolution_mock_into_tokenizer() {
    let res = TokenizerResolution::Mock(MockTokenizer::new());
    let tok = res.into_tokenizer().unwrap();
    assert_eq!(tok.vocab_size(), 50257);
}

// ===========================================================================
// LlamaVariant
// ===========================================================================

#[test]
fn llama_variant_expected_vocab_sizes() {
    assert_eq!(LlamaVariant::Llama2.expected_vocab_size(), 32000);
    assert_eq!(LlamaVariant::Llama3.expected_vocab_size(), 128256);
    assert_eq!(LlamaVariant::CodeLlama.expected_vocab_size(), 32016);
}

#[test]
fn llama_variant_gpu_acceleration() {
    assert!(!LlamaVariant::Llama2.requires_gpu_acceleration());
    assert!(LlamaVariant::Llama3.requires_gpu_acceleration());
    assert!(!LlamaVariant::CodeLlama.requires_gpu_acceleration());
}

// ===========================================================================
// estimate_tokens
// ===========================================================================

#[test]
fn estimate_tokens_empty() {
    assert_eq!(estimate_tokens(""), 0);
}

#[test]
fn estimate_tokens_single_word() {
    let est = estimate_tokens("hello");
    assert!(est >= 1);
}

#[test]
fn estimate_tokens_sentence() {
    let est = estimate_tokens("The quick brown fox jumps over the lazy dog.");
    assert!(est > 5);
}

// ===========================================================================
// DownloadProgress
// ===========================================================================

#[test]
fn download_progress_percentage_with_total() {
    let p = DownloadProgress {
        downloaded_bytes: 500,
        total_bytes: Some(1000),
        current_file: "model.safetensors".into(),
        completed_files: 0,
        total_files: 1,
    };
    assert!((p.percentage().unwrap() - 0.5).abs() < 0.01);
}

#[test]
fn download_progress_percentage_no_total() {
    let p = DownloadProgress {
        downloaded_bytes: 500,
        total_bytes: None,
        current_file: "model.safetensors".into(),
        completed_files: 0,
        total_files: 1,
    };
    assert!(p.percentage().is_none());
}

#[test]
fn download_progress_zero_total() {
    let p = DownloadProgress {
        downloaded_bytes: 0,
        total_bytes: Some(0),
        current_file: "empty.bin".into(),
        completed_files: 0,
        total_files: 1,
    };
    let _pct = p.percentage();
}
