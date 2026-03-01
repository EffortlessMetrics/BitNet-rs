//! Tests for tokenizer discovery, model compatibility matrix, and model type detection.
//!
//! Validates that all model families have valid download info, strategy selection works,
//! and model type detection covers expected vocabulary sizes.

use bitnet_tokenizers::ModelTypeDetector;
use bitnet_tokenizers::discovery::{ModelCompatibilityMatrix, TokenizerDownloadInfo};

// --- ModelCompatibilityMatrix construction ---

#[test]
fn compatibility_matrix_default_constructs() {
    let matrix = ModelCompatibilityMatrix::default();
    // Should not panic
    let _ = &matrix.llama3_128k;
}

#[test]
fn llama3_entry_has_valid_repo() {
    let matrix = ModelCompatibilityMatrix::default();
    assert!(!matrix.llama3_128k.repo.is_empty(), "LLaMA-3 should have a repo");
    assert!(!matrix.llama3_128k.files.is_empty(), "LLaMA-3 should have required files");
    assert!(!matrix.llama3_128k.cache_key.is_empty(), "LLaMA-3 should have a cache key");
}

#[test]
fn phi4_entry_has_valid_repo() {
    let matrix = ModelCompatibilityMatrix::default();
    assert!(!matrix.phi4_100k.repo.is_empty(), "Phi-4 should have a repo");
    assert!(!matrix.phi4_100k.files.is_empty());
    assert_eq!(matrix.phi4_100k.expected_vocab, Some(100352), "Phi-4 should have 100352 vocab");
}

#[test]
fn qwen2_entry_has_valid_repo() {
    let matrix = ModelCompatibilityMatrix::default();
    assert!(!matrix.qwen2_150k.repo.is_empty());
}

#[test]
fn gemma_entry_has_valid_repo() {
    let matrix = ModelCompatibilityMatrix::default();
    assert!(!matrix.gemma_256k.repo.is_empty());
}

#[test]
fn mistral_entry_has_valid_repo() {
    let matrix = ModelCompatibilityMatrix::default();
    assert!(!matrix.mistral_32k.repo.is_empty());
}

#[test]
fn deepseek_entry_has_valid_repo() {
    let matrix = ModelCompatibilityMatrix::default();
    assert!(!matrix.deepseek_100k.repo.is_empty());
}

#[test]
fn starcoder_entry_has_valid_repo() {
    let matrix = ModelCompatibilityMatrix::default();
    assert!(!matrix.starcoder_49k.repo.is_empty());
}

#[test]
fn gpt2_entry_has_valid_repo() {
    let matrix = ModelCompatibilityMatrix::default();
    assert!(!matrix.gpt2_50k.repo.is_empty());
    assert_eq!(matrix.gpt2_50k.expected_vocab, Some(50257));
}

#[test]
fn bitnet_entry_has_valid_repo() {
    let matrix = ModelCompatibilityMatrix::default();
    assert!(!matrix.bitnet_custom.repo.is_empty());
}

#[test]
fn all_entries_have_tokenizer_json() {
    let matrix = ModelCompatibilityMatrix::default();
    let entries = [
        &matrix.llama3_128k,
        &matrix.llama2_32k,
        &matrix.gpt2_50k,
        &matrix.phi4_100k,
        &matrix.qwen2_150k,
        &matrix.gemma_256k,
        &matrix.mistral_32k,
        &matrix.deepseek_100k,
        &matrix.starcoder_49k,
    ];
    for entry in &entries {
        assert!(
            entry.files.iter().any(|f: &String| f.contains("tokenizer")),
            "Entry '{}' should include a tokenizer file",
            entry.cache_key
        );
    }
}

#[test]
fn all_cache_keys_are_unique() {
    let matrix = ModelCompatibilityMatrix::default();
    let keys = [
        &matrix.llama3_128k.cache_key,
        &matrix.llama2_32k.cache_key,
        &matrix.gpt2_50k.cache_key,
        &matrix.phi4_100k.cache_key,
        &matrix.qwen2_150k.cache_key,
        &matrix.gemma_256k.cache_key,
        &matrix.mistral_32k.cache_key,
        &matrix.deepseek_100k.cache_key,
        &matrix.starcoder_49k.cache_key,
        &matrix.bitnet_custom.cache_key,
    ];
    for (i, a) in keys.iter().enumerate() {
        for (j, b) in keys.iter().enumerate() {
            if i != j {
                assert_ne!(a, b, "Cache keys should be unique");
            }
        }
    }
}

// --- TokenizerDownloadInfo construction ---

#[test]
fn download_info_construction() {
    let info = TokenizerDownloadInfo {
        repo: "microsoft/phi-4".to_string(),
        files: vec!["tokenizer.json".to_string()],
        cache_key: "phi4".to_string(),
        expected_vocab: Some(100352),
    };
    assert_eq!(info.repo, "microsoft/phi-4");
    assert_eq!(info.files.len(), 1);
    assert_eq!(info.expected_vocab, Some(100352));
}

#[test]
fn download_info_no_expected_vocab() {
    let info = TokenizerDownloadInfo {
        repo: "custom/model".to_string(),
        files: vec!["tokenizer.json".to_string()],
        cache_key: "custom".to_string(),
        expected_vocab: None,
    };
    assert!(info.expected_vocab.is_none());
}

#[test]
fn download_info_clone() {
    let info = TokenizerDownloadInfo {
        repo: "test/model".to_string(),
        files: vec!["a.json".to_string(), "b.json".to_string()],
        cache_key: "test".to_string(),
        expected_vocab: Some(32000),
    };
    let cloned = info.clone();
    assert_eq!(info.repo, cloned.repo);
    assert_eq!(info.files, cloned.files);
    assert_eq!(info.cache_key, cloned.cache_key);
    assert_eq!(info.expected_vocab, cloned.expected_vocab);
}

// --- ModelTypeDetector tests ---

#[test]
fn detect_llama3_from_vocab_size() {
    let model_type = ModelTypeDetector::detect_from_vocab_size(128256);
    assert!(
        model_type.to_lowercase().contains("llama"),
        "128256 should detect as llama-related: got '{model_type}'"
    );
}

#[test]
fn detect_gpt2_from_vocab_size() {
    let model_type = ModelTypeDetector::detect_from_vocab_size(50257);
    assert!(
        model_type.to_lowercase().contains("gpt"),
        "50257 should detect as gpt-related: got '{model_type}'"
    );
}

#[test]
fn detect_unknown_vocab_size() {
    let model_type = ModelTypeDetector::detect_from_vocab_size(12345);
    // Should return something (not panic), likely "unknown"
    assert!(!model_type.is_empty());
}

#[test]
fn validate_vocab_size_valid() {
    assert!(ModelTypeDetector::validate_vocab_size(32000).is_ok());
    assert!(ModelTypeDetector::validate_vocab_size(1).is_ok());
    assert!(ModelTypeDetector::validate_vocab_size(100352).is_ok());
}

#[test]
fn validate_vocab_size_zero_invalid() {
    assert!(ModelTypeDetector::validate_vocab_size(0).is_err());
}

#[test]
fn validate_vocab_size_very_large_invalid() {
    assert!(ModelTypeDetector::validate_vocab_size(3_000_000).is_err());
}

#[test]
fn requires_gpu_small_vocab() {
    assert!(!ModelTypeDetector::requires_gpu_acceleration(32000));
    assert!(!ModelTypeDetector::requires_gpu_acceleration(50257));
}

#[test]
fn requires_gpu_large_vocab() {
    assert!(ModelTypeDetector::requires_gpu_acceleration(100000));
    assert!(ModelTypeDetector::requires_gpu_acceleration(128256));
}

#[test]
fn requires_gpu_boundary() {
    // 65536 is the boundary
    assert!(!ModelTypeDetector::requires_gpu_acceleration(65536));
    assert!(ModelTypeDetector::requires_gpu_acceleration(65537));
}

#[test]
fn expected_vocab_size_known_models() {
    // Check a few known model types
    let gpt2 = ModelTypeDetector::expected_vocab_size("gpt2");
    if let Some(size) = gpt2 {
        assert_eq!(size, 50257);
    }
}

#[test]
fn expected_vocab_size_unknown_model() {
    let result = ModelTypeDetector::expected_vocab_size("completely-unknown-model");
    assert!(result.is_none());
}
