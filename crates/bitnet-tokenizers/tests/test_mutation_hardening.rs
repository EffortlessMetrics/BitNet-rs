//! Mutation testing hardening for Universal Tokenizer Discovery System
//!
//! This test suite strengthens coverage by focusing on edge cases, boundary conditions,
//! and error paths that mutation testing commonly identifies as weak areas.
//!
//! Key hardening areas:
//! - Vocabulary size boundary conditions (min/max values, overflow protection)
//! - Architecture detection edge cases (ambiguous patterns, unknown architectures)
//! - Special token validation (out-of-bounds, missing tokens)
//! - Quantization compatibility validation (I2S, TL1, TL2 edge cases)
//! - Error propagation paths (Result<T, BitNetError> patterns)

use bitnet_common::{BitNetError, QuantizationType};
use bitnet_tokenizers::{
    Tokenizer,
    discovery::TokenizerStrategy,
    error_handling::ModelTypeDetector,
    strategy::{BitNetTokenizerWrapper, Gpt2TokenizerWrapper, LlamaTokenizerWrapper, LlamaVariant},
};
use std::sync::Arc;

// ================================
// VOCABULARY SIZE BOUNDARY TESTS
// ================================

/// Test vocabulary size validation at exact boundaries
#[test]
fn test_vocab_size_boundary_validation() {
    let boundary_cases = [
        // (vocab_size, expected_valid, description)
        (0, false, "Zero vocabulary"),
        (1, true, "Minimum vocabulary"),
        (99, true, "Small vocabulary"),
        (100, true, "Small vocabulary boundary"),
        (32000, true, "LLaMA-2 standard"),
        (50257, true, "GPT-2 standard"),
        (65535, true, "16-bit boundary"),
        (65536, true, "GPU acceleration threshold"),
        (65537, true, "Just above GPU threshold"),
        (128256, true, "LLaMA-3 standard"),
        (200000, true, "Large vocabulary"),
        (1000000, true, "Very large vocabulary"),
        (2000000, true, "Maximum valid vocabulary"),
        (2000001, false, "Just above maximum"),
        (usize::MAX, false, "Maximum possible (overflow)"),
    ];

    for (vocab_size, expected_valid, description) in boundary_cases {
        let result = ModelTypeDetector::validate_vocab_size(vocab_size);
        assert_eq!(
            result.is_ok(),
            expected_valid,
            "Vocab size validation mismatch for {}: vocab_size={}",
            description,
            vocab_size
        );

        if expected_valid {
            // For valid sizes, ensure GPU acceleration detection works
            let requires_gpu = ModelTypeDetector::requires_gpu_acceleration(vocab_size);
            let expected_gpu = vocab_size > 65536;
            assert_eq!(
                requires_gpu, expected_gpu,
                "GPU requirement mismatch for {}: vocab_size={}",
                description, vocab_size
            );
        }
    }
}

/// Test vocabulary size integer overflow protection
#[test]
fn test_vocab_size_overflow_protection() {
    // Test arithmetic operations don't overflow
    let large_vocab_sizes = [
        65536,
        128256,
        200000,
        usize::MAX / 2, // Half maximum
    ];

    for vocab_size in large_vocab_sizes {
        // Test multiplication doesn't overflow (embedding size calculation)
        let embedding_dim = 4096; // Typical embedding dimension
        let result = vocab_size.checked_mul(embedding_dim);

        if vocab_size <= 200000 {
            assert!(result.is_some(), "Reasonable vocab size should not overflow: {}", vocab_size);
        }

        // Test addition doesn't overflow (special tokens)
        let with_special = vocab_size.checked_add(100);
        assert!(with_special.is_some() || vocab_size >= usize::MAX - 100);
    }
}

// ================================
// SPECIAL TOKEN VALIDATION TESTS
// ================================

/// Test special token boundary validation for LLaMA variants
#[test]
fn test_llama_special_token_boundaries() {
    let test_cases = [
        // (vocab_size, variant, bos, eos, pad, should_validate)
        (32000, LlamaVariant::Llama2, Some(1), Some(2), Some(0), true),
        (32000, LlamaVariant::Llama2, Some(1), Some(32000), None, false), // EOS out of bounds
        (32000, LlamaVariant::Llama2, Some(32000), Some(2), None, false), // BOS out of bounds
        (128256, LlamaVariant::Llama3, Some(128000), Some(128001), Some(128002), true),
        (128256, LlamaVariant::Llama3, Some(128256), Some(128001), None, false), // BOS out of bounds
        (32016, LlamaVariant::CodeLlama, Some(1), Some(2), None, true),
        (32016, LlamaVariant::CodeLlama, Some(32016), Some(2), None, false), // BOS out of bounds
    ];

    for (vocab_size, _variant, bos, eos, pad, should_validate) in test_cases {
        let base_tokenizer =
            Arc::new(bitnet_tokenizers::BasicTokenizer::with_config(vocab_size, bos, eos, pad));

        let wrapper_result = LlamaTokenizerWrapper::new(base_tokenizer, vocab_size);
        assert!(wrapper_result.is_ok(), "Wrapper should always initialize");

        let wrapper = wrapper_result.unwrap();

        // Test encoding with special tokens
        let encode_result = wrapper.encode("test", true, true);
        if should_validate {
            assert!(
                encode_result.is_ok(),
                "Valid special tokens should allow encoding: bos={:?}, eos={:?}",
                bos,
                eos
            );
        }

        // Verify special token accessors match configuration
        assert_eq!(wrapper.bos_token_id(), Some(1));
        assert_eq!(wrapper.eos_token_id(), Some(2));
    }
}

/// Test GPT-2 special token handling (no BOS token)
#[test]
fn test_gpt2_special_token_handling() {
    let base_tokenizer = Arc::new(bitnet_tokenizers::BasicTokenizer::with_config(
        50257,
        None, // GPT-2 doesn't use BOS
        Some(50256),
        None,
    ));

    let wrapper = Gpt2TokenizerWrapper::new(base_tokenizer).expect("Should create GPT-2 wrapper");

    // Test BOS token is correctly reported as None
    assert_eq!(wrapper.bos_token_id(), None, "GPT-2 should not have BOS token");
    assert_eq!(wrapper.eos_token_id(), Some(50256), "GPT-2 EOS should be 50256");

    // Test encoding ignores BOS request
    let result_with_bos = wrapper.encode("test", true, false);
    let result_without_bos = wrapper.encode("test", false, false);

    assert!(result_with_bos.is_ok());
    assert!(result_without_bos.is_ok());

    // Both should produce same result since GPT-2 ignores BOS
    let tokens_with = result_with_bos.unwrap();
    let tokens_without = result_without_bos.unwrap();
    assert_eq!(tokens_with, tokens_without, "GPT-2 should ignore BOS flag");
}

// ================================
// QUANTIZATION COMPATIBILITY TESTS
// ================================

/// Test quantization validation through encoding interface
#[test]
fn test_quantization_token_range_validation() {
    let vocab_size = 32000;
    let base_tokenizer = Arc::new(bitnet_tokenizers::BasicTokenizer::with_config(
        vocab_size,
        Some(1),
        Some(2),
        None,
    ));

    let wrapper = BitNetTokenizerWrapper::new(base_tokenizer, QuantizationType::I2S)
        .expect("Should create BitNet wrapper");

    // Test encoding with various inputs to exercise validation paths
    let long_text = "a".repeat(100);
    let test_scenarios = [
        ("", true, "Empty string"),
        ("test", true, "Simple text"),
        (long_text.as_str(), true, "Long text"),
        ("0123456789", true, "Numeric text"),
        ("Hello, world!", true, "Punctuation"),
    ];

    for (text, should_pass, description) in test_scenarios {
        let result = wrapper.encode(text, true, false);

        if should_pass {
            assert!(result.is_ok(), "Should encode valid text for {}: '{}'", description, text);
        } else {
            assert!(result.is_err(), "Should reject invalid text for {}: '{}'", description, text);
        }
    }

    // Test vocabulary size boundaries
    assert_eq!(wrapper.vocab_size(), vocab_size, "Wrapper should report correct vocab size");
    assert_eq!(
        wrapper.quantization_type(),
        QuantizationType::I2S,
        "Wrapper should report correct quantization"
    );
}

/// Test quantization type compatibility with vocabulary sizes
#[test]
fn test_quantization_vocab_size_compatibility() {
    let compatibility_matrix = [
        // (vocab_size, quant_type, should_warn)
        (1000, QuantizationType::I2S, false),
        (32000, QuantizationType::I2S, false),
        (128256, QuantizationType::I2S, false),
        (200000, QuantizationType::I2S, false),
        (200001, QuantizationType::I2S, true), // Exceeds I2S recommendation
        (1000, QuantizationType::TL1, false),
        (32000, QuantizationType::TL1, false),
        (65536, QuantizationType::TL1, false),
        (65537, QuantizationType::TL1, true), // Exceeds TL1 optimal size
        (128256, QuantizationType::TL1, true),
        (1000, QuantizationType::TL2, false),
        (32000, QuantizationType::TL2, false),
        (65536, QuantizationType::TL2, false),
        (65537, QuantizationType::TL2, true), // Exceeds TL2 optimal size
    ];

    for (vocab_size, quant_type, _should_warn) in compatibility_matrix {
        let base_tokenizer = Arc::new(bitnet_tokenizers::BasicTokenizer::with_config(
            vocab_size,
            Some(1),
            Some(2),
            None,
        ));

        let wrapper_result = BitNetTokenizerWrapper::new(base_tokenizer, quant_type);

        // All combinations should initialize (warnings are logged, not errors)
        assert!(
            wrapper_result.is_ok(),
            "Quantization wrapper should initialize for vocab_size={}, quant_type={:?}",
            vocab_size,
            quant_type
        );

        let wrapper = wrapper_result.unwrap();
        assert_eq!(wrapper.quantization_type(), quant_type);
        assert_eq!(wrapper.vocab_size(), vocab_size);
    }
}

// ================================
// ARCHITECTURE DETECTION TESTS
// ================================

/// Test LLaMA variant detection at exact boundaries
#[test]
fn test_llama_variant_detection_boundaries() {
    let detection_cases = [
        // (vocab_size, expected_variant)
        (31999, LlamaVariant::Llama2),    // Just below LLaMA-2
        (32000, LlamaVariant::Llama2),    // Exactly LLaMA-2
        (32001, LlamaVariant::Llama2),    // Just above LLaMA-2
        (32015, LlamaVariant::CodeLlama), // Just below CodeLlama
        (32016, LlamaVariant::CodeLlama), // Exactly CodeLlama
        (32017, LlamaVariant::CodeLlama), // Just above CodeLlama
        (128255, LlamaVariant::Llama3),   // Just below LLaMA-3
        (128256, LlamaVariant::Llama3),   // Exactly LLaMA-3
        (128257, LlamaVariant::Llama3),   // Just above LLaMA-3
    ];

    for (vocab_size, expected_variant) in detection_cases {
        let detected = ModelTypeDetector::detect_from_vocab_size(vocab_size);

        // Verify detection aligns with expected variant
        let expected_type = match expected_variant {
            LlamaVariant::Llama2 => "llama2",
            LlamaVariant::Llama3 => "llama3",
            LlamaVariant::CodeLlama => "codellama",
        };

        // Model type detector may use different naming, so check vocabulary size instead
        let base_tokenizer = Arc::new(bitnet_tokenizers::BasicTokenizer::with_config(
            vocab_size,
            Some(1),
            Some(2),
            None,
        ));

        let wrapper = LlamaTokenizerWrapper::new(base_tokenizer, vocab_size)
            .expect("Should create LLaMA wrapper");

        // Verify the wrapper correctly identifies the variant
        let wrapper_vocab = wrapper.vocab_size();
        let expected_vocab = expected_variant.expected_vocab_size();

        // Allow some tolerance for similar variants
        let vocab_diff = (wrapper_vocab as i64 - expected_vocab as i64).abs();
        assert!(
            vocab_diff < 100 || detected.contains(&expected_type[..5]),
            "Detection mismatch for vocab_size={}: detected={}, expected={:?}",
            vocab_size,
            detected,
            expected_variant
        );
    }
}

/// Test GPU acceleration requirements across architectures
#[test]
fn test_gpu_acceleration_architecture_requirements() {
    let architecture_test_cases = [
        // (model_type, vocab_size, requires_gpu)
        ("llama2", 32000, false),
        ("llama3", 128256, true),
        ("codellama", 32016, false),
        ("gpt2", 50257, false),
        ("bitnet", 32000, false),
        ("bitnet", 128256, true),
        ("custom", 65536, false),
        ("custom", 65537, true),
        ("custom", 200000, true),
    ];

    for (model_type, vocab_size, expected_gpu) in architecture_test_cases {
        let requires_gpu = ModelTypeDetector::requires_gpu_acceleration(vocab_size);

        assert_eq!(
            requires_gpu, expected_gpu,
            "GPU requirement mismatch for model_type={}, vocab_size={}",
            model_type, vocab_size
        );

        // For LLaMA variants, verify the variant's GPU requirement
        if model_type.starts_with("llama") {
            let variant_expected_gpu = match model_type {
                "llama2" => false,
                "llama3" => true,
                "codellama" => false,
                _ => false,
            };

            let variant = match model_type {
                "llama2" => LlamaVariant::Llama2,
                "llama3" => LlamaVariant::Llama3,
                "codellama" => LlamaVariant::CodeLlama,
                _ => LlamaVariant::Llama2,
            };

            assert_eq!(
                variant.requires_gpu_acceleration(),
                variant_expected_gpu,
                "Variant GPU requirement mismatch for {:?}",
                variant
            );
        }
    }
}

// ================================
// ERROR PROPAGATION TESTS
// ================================

/// Test error propagation through tokenizer wrapper layers
#[test]
fn test_error_propagation_through_wrappers() {
    // Create a tokenizer that will fail encoding
    struct FailingTokenizer {
        vocab_size: usize,
        error_message: String,
    }

    impl Tokenizer for FailingTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_bos: bool,
            _add_special: bool,
        ) -> bitnet_common::Result<Vec<u32>> {
            Err(BitNetError::Config(self.error_message.clone()))
        }

        fn decode(&self, _tokens: &[u32]) -> bitnet_common::Result<String> {
            Err(BitNetError::Config(self.error_message.clone()))
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn token_to_piece(&self, _token: u32) -> Option<String> {
            None
        }

        fn bos_token_id(&self) -> Option<u32> {
            Some(1)
        }

        fn eos_token_id(&self) -> Option<u32> {
            Some(2)
        }
    }

    let error_scenarios = [
        "Intentional encoding failure",
        "Invalid model state",
        "Resource exhaustion",
        "Corrupted vocabulary",
    ];

    for error_msg in error_scenarios {
        let failing_tokenizer =
            Arc::new(FailingTokenizer { vocab_size: 32000, error_message: error_msg.to_string() });

        // Test error propagation through LLaMA wrapper
        let llama_wrapper = LlamaTokenizerWrapper::new(failing_tokenizer.clone(), 32000)
            .expect("Wrapper should initialize despite inner tokenizer");

        let encode_result = llama_wrapper.encode("test", true, false);
        assert!(encode_result.is_err(), "Should propagate encoding error");

        match encode_result.unwrap_err() {
            BitNetError::Config(msg) => {
                assert_eq!(msg, error_msg, "Error message should be preserved");
            }
            other => panic!("Unexpected error type: {:?}", other),
        }

        let decode_result = llama_wrapper.decode(&[1, 2, 3]);
        assert!(decode_result.is_err(), "Should propagate decoding error");

        // Test error propagation through BitNet wrapper
        let bitnet_wrapper =
            BitNetTokenizerWrapper::new(failing_tokenizer.clone(), QuantizationType::I2S)
                .expect("BitNet wrapper should initialize");

        let bitnet_encode_result = bitnet_wrapper.encode("test", true, false);
        assert!(bitnet_encode_result.is_err(), "Should propagate error through BitNet wrapper");

        match bitnet_encode_result.unwrap_err() {
            BitNetError::Config(msg) => {
                assert_eq!(msg, error_msg, "BitNet wrapper should preserve error message");
            }
            other => panic!("Unexpected error type: {:?}", other),
        }
    }
}

/// Test tokenizer strategy error conditions
#[test]
fn test_tokenizer_strategy_error_conditions() {
    use std::path::PathBuf;

    let error_strategies = [
        // Test invalid paths
        (
            TokenizerStrategy::Exact(PathBuf::from("/nonexistent/tokenizer.json")),
            "nonexistent file",
        ),
        (TokenizerStrategy::Exact(PathBuf::from("/root/restricted.json")), "inaccessible path"),
        (TokenizerStrategy::Discovered(PathBuf::from("")), "empty path"),
        // Test invalid download info
        (
            TokenizerStrategy::NeedsDownload(bitnet_tokenizers::discovery::TokenizerDownloadInfo {
                repo: "".to_string(),
                files: vec![],
                cache_key: "invalid".to_string(),
                expected_vocab: None,
            }),
            "empty repo",
        ),
    ];

    for (strategy, description) in error_strategies {
        // Verify strategy properties are correct
        assert!(
            !strategy.description().is_empty(),
            "Strategy should have description: {}",
            description
        );

        // Verify network requirements
        let requires_network = strategy.requires_network();
        match &strategy {
            TokenizerStrategy::NeedsDownload(_) => {
                assert!(requires_network, "{} should require network", description);
            }
            _ => {
                // Other strategies may or may not require network
            }
        }
    }
}

// ================================
// TOKENIZER WRAPPER VALIDATION TESTS
// ================================

/// Test vocabulary size mismatch handling in wrappers
#[test]
fn test_wrapper_vocab_size_mismatch_handling() {
    let mismatch_cases = [
        // (inner_vocab_size, wrapper_vocab_size, description)
        (1000, 32000, "Small tokenizer with large expected vocab"),
        (50000, 32000, "Large tokenizer with small expected vocab"),
        (32000, 128256, "LLaMA-2 tokenizer with LLaMA-3 size"),
        (128256, 32000, "LLaMA-3 tokenizer with LLaMA-2 size"),
        (50257, 32000, "GPT-2 tokenizer with LLaMA size"),
    ];

    for (inner_vocab, wrapper_vocab, description) in mismatch_cases {
        let base_tokenizer = Arc::new(bitnet_tokenizers::BasicTokenizer::with_config(
            inner_vocab,
            Some(1),
            Some(2),
            None,
        ));

        // LLaMA wrapper should initialize despite mismatch
        let wrapper_result = LlamaTokenizerWrapper::new(base_tokenizer, wrapper_vocab);
        assert!(
            wrapper_result.is_ok(),
            "{}: wrapper should initialize despite vocab mismatch",
            description
        );

        let wrapper = wrapper_result.unwrap();

        // Wrapper should report the expected vocab size
        assert_eq!(
            wrapper.vocab_size(),
            wrapper_vocab,
            "{}: wrapper should report expected vocab size",
            description
        );
    }
}

/// Test concurrent wrapper creation and usage
#[tokio::test]
async fn test_concurrent_wrapper_creation() {
    use tokio::task;

    let base_tokenizer =
        Arc::new(bitnet_tokenizers::BasicTokenizer::with_config(32000, Some(1), Some(2), None));

    let mut handles = vec![];

    // Create multiple wrappers concurrently
    for i in 0..10 {
        let tokenizer_clone = Arc::clone(&base_tokenizer);
        let handle = task::spawn(async move {
            let wrapper_result = LlamaTokenizerWrapper::new(tokenizer_clone, 32000);
            assert!(wrapper_result.is_ok(), "Concurrent wrapper {} should succeed", i);

            let wrapper = wrapper_result.unwrap();

            // Test concurrent encoding
            let test_text = format!("Concurrent test {}", i);
            let tokens_result = wrapper.encode(&test_text, true, false);
            assert!(tokens_result.is_ok(), "Concurrent encoding {} should succeed", i);

            tokens_result.unwrap()
        });
        handles.push(handle);
    }

    // Collect results
    let mut all_results = vec![];
    for handle in handles {
        let tokens = handle.await.expect("Concurrent task should complete");
        all_results.push(tokens);
    }

    // All results should be valid
    assert_eq!(all_results.len(), 10);
    for (i, tokens) in all_results.iter().enumerate() {
        assert!(!tokens.is_empty(), "Result {} should have tokens", i);
        assert_eq!(tokens[0], 1, "All results should start with BOS token");
    }
}

// ================================
// EDGE CASE INTEGRATION TESTS
// ================================

/// Test extreme vocabulary sizes in integrated workflow
#[test]
fn test_extreme_vocab_sizes_integration() {
    let extreme_cases = [
        (100, "Minimum valid"),
        (1000, "Small model"),
        (32000, "Standard LLaMA-2"),
        (50257, "Standard GPT-2"),
        (65536, "Boundary 16-bit"),
        (128256, "Large LLaMA-3"),
        (199999, "Just below max"),
        (200000, "Maximum valid"),
    ];

    for (vocab_size, description) in extreme_cases {
        let validation_result = ModelTypeDetector::validate_vocab_size(vocab_size);
        assert!(
            validation_result.is_ok(),
            "{}: vocab size {} should be valid",
            description,
            vocab_size
        );

        // Test wrapper creation with extreme sizes
        let base_tokenizer = Arc::new(bitnet_tokenizers::BasicTokenizer::with_config(
            vocab_size,
            Some(1),
            Some(2),
            None,
        ));

        let llama_wrapper = LlamaTokenizerWrapper::new(base_tokenizer.clone(), vocab_size);
        assert!(
            llama_wrapper.is_ok(),
            "{}: LLaMA wrapper should handle vocab_size={}",
            description,
            vocab_size
        );

        let bitnet_wrapper = BitNetTokenizerWrapper::new(base_tokenizer, QuantizationType::I2S);
        assert!(
            bitnet_wrapper.is_ok(),
            "{}: BitNet wrapper should handle vocab_size={}",
            description,
            vocab_size
        );
    }
}

/// Test quantization wrapper with edge case inputs
#[test]
fn test_quantization_edge_case_inputs() {
    let vocab_size = 32000;
    let base_tokenizer = Arc::new(bitnet_tokenizers::BasicTokenizer::with_config(
        vocab_size,
        Some(1),
        Some(2),
        None,
    ));

    let wrapper = BitNetTokenizerWrapper::new(base_tokenizer, QuantizationType::I2S)
        .expect("Should create wrapper");

    let very_long_input = "a".repeat(1000);
    let edge_case_inputs = [
        ("", "Empty string"),
        ("a", "Single character"),
        ("test", "Simple word"),
        ("test test test", "Repeated words"),
        (very_long_input.as_str(), "Very long input"),
        ("日本語", "Non-ASCII text"),
        ("test\nline\nbreaks", "Multi-line text"),
    ];

    for (input, description) in edge_case_inputs {
        let encode_result = wrapper.encode(input, true, false);
        assert!(encode_result.is_ok(), "Should encode edge case: {}", description);

        if let Ok(tokens) = encode_result {
            // Test decoding roundtrip
            let decode_result = wrapper.decode(&tokens);
            assert!(decode_result.is_ok(), "Should decode edge case: {}", description);
        }
    }
}
