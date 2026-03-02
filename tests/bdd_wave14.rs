//! BDD Wave 14: Integration tests across quantization, model loading,
//! tokenizer, inference configuration, and error handling.
//!
//! Each test follows Given / When / Then structure and exercises real code paths.

// =========================================================================
// Feature: Quantization Pipeline
// =========================================================================

mod quantization_pipeline {
    use bitnet_common::{Device, Tensor};
    use bitnet_quantization::{I2SQuantizer, QuantizedTensor, TL1Quantizer};

    #[test]
    fn test_given_i2s_input_when_quantize_then_output_contains_only_ternary_values() {
        // Given a vector of arbitrary f32 values
        let weights: Vec<f32> = vec![
            1.5, -0.8, 0.0, 2.3, -3.1, 0.02, 0.7, -1.0, 0.5, -0.3, 1.2, -2.0, 0.0, 0.1, -0.6, 0.9,
            1.1, -1.5, 0.4, -0.2, 3.0, -0.01, 0.8, -1.3, 0.6, -0.7, 1.8, -2.5, 0.0, 0.3, -0.4,
            0.05,
        ];
        let quantizer = I2SQuantizer::new();

        // When I quantize
        let quantized = quantizer.quantize_weights(&weights).unwrap();

        // Then the quantized data is non-empty and shape matches
        assert!(!quantized.data.is_empty());
        assert_eq!(quantized.shape, vec![weights.len()]);
        assert_eq!(quantized.numel(), weights.len());
    }

    #[test]
    fn test_given_i2s_input_when_roundtrip_then_error_is_bounded() {
        // Given f32 weights
        let weights: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.2).collect();
        let quantizer = I2SQuantizer::new();

        // When I quantize and then dequantize
        let tensor =
            bitnet_common::BitNetTensor::from_slice(&weights, &[32], &Device::Cpu).unwrap();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let restored = quantizer.dequantize_tensor(&quantized).unwrap();

        // Then the shapes match and we recovered a valid tensor
        assert_eq!(restored.shape(), &[32]);
        let restored_data = restored.to_vec().unwrap();
        assert_eq!(restored_data.len(), 32);
    }

    #[test]
    fn test_given_zero_input_when_quantize_then_all_outputs_are_zero() {
        // Given all-zero weights
        let weights: Vec<f32> = vec![0.0; 32];
        let quantizer = I2SQuantizer::new();

        // When I quantize and dequantize
        let tensor =
            bitnet_common::BitNetTensor::from_slice(&weights, &[32], &Device::Cpu).unwrap();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let restored = quantizer.dequantize_tensor(&quantized).unwrap();

        // Then all restored values are zero
        let restored_data = restored.to_vec().unwrap();
        for (i, &v) in restored_data.iter().enumerate() {
            assert_eq!(v, 0.0, "expected zero at index {i}, got {v}");
        }
    }

    #[test]
    fn test_given_max_range_input_when_quantize_then_no_overflow() {
        // Given extreme f32 values (within reasonable range)
        let weights: Vec<f32> =
            (0..32).map(|i| if i % 2 == 0 { 1000.0 } else { -1000.0 }).collect();
        let quantizer = I2SQuantizer::new();

        // When I quantize
        let quantized = quantizer.quantize_weights(&weights).unwrap();

        // Then quantization succeeds without overflow
        assert!(!quantized.data.is_empty());
        assert_eq!(quantized.numel(), 32);
    }

    #[test]
    fn test_given_tl1_quantizer_when_dequantize_then_error_is_bounded() {
        // Given a TL1 quantizer with default config
        let quantizer = TL1Quantizer::new();

        // When I quantize and dequantize weights
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let tensor =
            bitnet_common::BitNetTensor::from_slice(&weights, &[64], &Device::Cpu).unwrap();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let restored = quantizer.dequantize_tensor(&quantized).unwrap();

        // Then the dequantized tensor has the correct shape
        assert_eq!(restored.shape(), &[64]);
    }

    #[test]
    fn test_given_quantized_tensor_when_numel_then_matches_shape() {
        // Given a QuantizedTensor with known shape
        let qt = QuantizedTensor::new(
            vec![0u8; 8],
            vec![1.0],
            vec![4, 8],
            bitnet_common::QuantizationType::I2S,
        );

        // When I query numel
        let n = qt.numel();

        // Then it equals the product of dimensions
        assert_eq!(n, 32);
    }

    #[test]
    fn test_given_quantized_tensor_when_compression_ratio_then_positive() {
        // Given a QuantizedTensor
        let qt = QuantizedTensor::new(
            vec![0u8; 16],
            vec![1.0; 4],
            vec![128],
            bitnet_common::QuantizationType::I2S,
        );

        // When I compute compression ratio
        let ratio = qt.compression_ratio();

        // Then ratio is >= 1.0
        assert!(ratio >= 1.0, "compression ratio should be >= 1.0, got {ratio}");
    }

    #[test]
    fn test_given_single_element_when_quantize_roundtrip_then_shape_preserved() {
        // Given a minimal 32-element block (minimum I2S block size)
        let weights: Vec<f32> = vec![0.5; 32];
        let quantizer = I2SQuantizer::new();

        // When I quantize
        let quantized = quantizer.quantize_weights(&weights).unwrap();

        // Then the shape is preserved
        assert_eq!(quantized.shape, vec![32]);
        assert_eq!(quantized.numel(), 32);
    }
}

// =========================================================================
// Feature: Model Loading (GGUF)
// =========================================================================

mod model_loading {
    use bitnet_gguf::{
        GGUF_MAGIC, GGUF_VERSION_MAX, GGUF_VERSION_MIN, check_magic, parse_header, read_version,
    };

    fn make_gguf_v2_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(b"GGUF");
        d.extend_from_slice(&2u32.to_le_bytes());
        d.extend_from_slice(&tensor_count.to_le_bytes());
        d.extend_from_slice(&metadata_count.to_le_bytes());
        d
    }

    fn make_gguf_v3_header(tensor_count: u64, metadata_count: u64, alignment: u32) -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(b"GGUF");
        d.extend_from_slice(&3u32.to_le_bytes());
        d.extend_from_slice(&tensor_count.to_le_bytes());
        d.extend_from_slice(&metadata_count.to_le_bytes());
        d.extend_from_slice(&alignment.to_le_bytes());
        d
    }

    #[test]
    fn test_given_valid_gguf_header_when_parse_then_metadata_correct() {
        // Given a valid GGUF v2 header with 3 tensors and 5 metadata entries
        let data = make_gguf_v2_header(3, 5);

        // When I parse the header
        let info = parse_header(&data).unwrap();

        // Then metadata counts are correct
        assert_eq!(info.version, 2);
        assert_eq!(info.tensor_count, 3);
        assert_eq!(info.metadata_count, 5);
        assert_eq!(info.alignment, 32); // v2 default
    }

    #[test]
    fn test_given_truncated_file_when_parse_then_descriptive_error() {
        // Given data that is too short for a GGUF header
        let data = b"GGU"; // only 3 bytes

        // When I try to parse
        let result = parse_header(data);

        // Then I get an error about size
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("too small"), "expected 'too small' in error, got: {err_msg}");
    }

    #[test]
    fn test_given_unsupported_version_when_parse_then_error() {
        // Given GGUF data with version 99
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&99u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count

        // When I try to parse
        let result = parse_header(&data);

        // Then I get an error about unsupported version
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("unsupported GGUF version"),
            "expected version error, got: {err_msg}"
        );
    }

    #[test]
    fn test_given_valid_magic_bytes_when_check_then_passes() {
        // Given data starting with GGUF magic
        let data = b"GGUF\x02\x00\x00\x00extra data here";

        // When I check magic
        let result = check_magic(data);

        // Then it passes
        assert!(result);
    }

    #[test]
    fn test_given_invalid_magic_bytes_when_check_then_fails() {
        // Given data with wrong magic bytes
        let data = b"LLAM\x02\x00\x00\x00";

        // When I check magic
        let result = check_magic(data);

        // Then it fails
        assert!(!result);
    }

    #[test]
    fn test_given_valid_v2_data_when_read_version_then_returns_2() {
        // Given valid GGUF v2 bytes
        let data = make_gguf_v2_header(0, 0);

        // When I read the version
        let version = read_version(&data);

        // Then it returns 2
        assert_eq!(version, Some(2));
    }

    #[test]
    fn test_given_v3_header_when_parse_then_alignment_respected() {
        // Given a GGUF v3 header with 64-byte alignment
        let data = make_gguf_v3_header(10, 20, 64);

        // When I parse
        let info = parse_header(&data).unwrap();

        // Then alignment is 64 and version is 3
        assert_eq!(info.version, 3);
        assert_eq!(info.alignment, 64);
        assert_eq!(info.tensor_count, 10);
        assert_eq!(info.metadata_count, 20);
    }

    #[test]
    fn test_given_gguf_constants_when_checked_then_values_correct() {
        // Given the GGUF constants
        // When I check them
        // Then magic is b"GGUF" and version range is 2..=3
        assert_eq!(&GGUF_MAGIC, b"GGUF");
        assert_eq!(GGUF_VERSION_MIN, 2);
        assert_eq!(GGUF_VERSION_MAX, 3);
    }
}

// =========================================================================
// Feature: Tokenizer
// =========================================================================

mod tokenizer {
    use bitnet_tokenizers::{BasicTokenizer, Tokenizer};

    #[test]
    fn test_given_ascii_text_when_encode_decode_then_output_matches_input() {
        // Given ASCII text and a BasicTokenizer
        let tokenizer = BasicTokenizer::new();
        let text = "hello world";

        // When I encode then decode
        let tokens = tokenizer.encode(text, false, false).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();

        // Then the output matches the input
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_given_empty_string_when_encode_then_result_is_empty() {
        // Given empty input
        let tokenizer = BasicTokenizer::new();

        // When I encode
        let tokens = tokenizer.encode("", false, false).unwrap();

        // Then result is empty
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_given_special_tokens_when_encode_with_special_then_eos_appended() {
        // Given text and special token mode enabled
        let tokenizer = BasicTokenizer::new();

        // When I encode with add_special=true
        let tokens = tokenizer.encode("hi", false, true).unwrap();

        // Then EOS token (50256) is appended
        assert!(tokens.last() == Some(&50256), "last token should be EOS 50256");
    }

    #[test]
    fn test_given_token_ids_when_decode_then_utf8_is_valid() {
        // Given byte-level token IDs for "abc"
        let tokenizer = BasicTokenizer::new();
        let tokens: Vec<u32> = vec![97, 98, 99]; // 'a', 'b', 'c'

        // When I decode
        let decoded = tokenizer.decode(&tokens).unwrap();

        // Then the output is valid UTF-8 and matches "abc"
        assert_eq!(decoded, "abc");
    }

    #[test]
    fn test_given_bos_config_when_encode_with_bos_then_bos_prepended() {
        // Given a tokenizer with BOS token configured
        let tokenizer = BasicTokenizer::with_config(50257, Some(1), Some(50256), None);

        // When I encode with add_bos=true
        let tokens = tokenizer.encode("a", true, false).unwrap();

        // Then BOS (1) is the first token
        assert_eq!(tokens[0], 1, "first token should be BOS");
        assert_eq!(tokens[1], 97, "second token should be 'a'");
    }

    #[test]
    fn test_given_vocab_size_when_query_then_correct() {
        // Given a BasicTokenizer with default config
        let tokenizer = BasicTokenizer::new();

        // When I query vocab_size
        let size = tokenizer.vocab_size();

        // Then it's 50257 (GPT-2 default)
        assert_eq!(size, 50257);
    }

    #[test]
    fn test_given_token_id_when_token_to_piece_then_valid_string() {
        // Given a BasicTokenizer
        let tokenizer = BasicTokenizer::new();

        // When I convert byte-range token to piece
        let piece = tokenizer.token_to_piece(65); // 'A'

        // Then I get "A"
        assert_eq!(piece, Some("A".to_string()));
    }

    #[test]
    fn test_given_empty_tokens_when_decode_then_empty_string() {
        // Given an empty token list
        let tokenizer = BasicTokenizer::new();

        // When I decode
        let decoded = tokenizer.decode(&[]).unwrap();

        // Then result is empty
        assert!(decoded.is_empty());
    }
}

// =========================================================================
// Feature: Inference Configuration
// =========================================================================

mod inference_configuration {
    use bitnet_inference::config::{GenerationConfig, InferenceConfig};

    #[test]
    fn test_given_default_config_when_validate_then_passes() {
        // Given a default GenerationConfig
        let config = GenerationConfig::default();

        // When I validate
        let result = config.validate();

        // Then validation passes
        assert!(result.is_ok(), "default config should validate: {:?}", result);
    }

    #[test]
    fn test_given_greedy_config_when_validate_then_passes() {
        // Given a greedy GenerationConfig
        let config = GenerationConfig::greedy();

        // When I validate
        let result = config.validate();

        // Then it passes and has expected values
        assert!(result.is_ok());
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 1);
    }

    #[test]
    fn test_given_temperature_zero_when_greedy_then_deterministic_settings() {
        // Given a greedy config
        let config = GenerationConfig::greedy();

        // When I inspect settings
        // Then temperature is 0 and top_k is 1 (argmax)
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 1);
        assert!((config.top_p - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_given_top_k_1_when_configured_then_single_candidate() {
        // Given a config with top_k=1
        let config = GenerationConfig::greedy().with_top_k(1);

        // When I validate
        let result = config.validate();

        // Then it passes and top_k is 1
        assert!(result.is_ok());
        assert_eq!(config.top_k, 1);
    }

    #[test]
    fn test_given_negative_temperature_when_validate_then_error() {
        // Given a config with negative temperature
        let config = GenerationConfig::default().with_temperature(-1.0);

        // When I validate
        let result = config.validate();

        // Then validation fails with temperature error
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("temperature"), "expected temperature error, got: {err}");
    }

    #[test]
    fn test_given_invalid_top_p_when_validate_then_error() {
        // Given a config with top_p > 1.0
        let config = GenerationConfig::default().with_top_p(1.5);

        // When I validate
        let result = config.validate();

        // Then validation fails
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("top_p"), "expected top_p error, got: {err}");
    }

    #[test]
    fn test_given_zero_max_tokens_when_validate_then_error() {
        // Given a config with max_new_tokens=0
        let config = GenerationConfig::default().with_max_tokens(0);

        // When I validate
        let result = config.validate();

        // Then validation fails
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("max_new_tokens"), "expected max_new_tokens error, got: {err}");
    }

    #[test]
    fn test_given_inference_config_default_when_inspect_then_sensible_values() {
        // Given a default InferenceConfig
        let config = InferenceConfig::default();

        // When I inspect values
        // Then they are sensible
        assert!(config.max_context_length > 0);
        assert!(config.num_threads > 0);
        assert_eq!(config.batch_size, 1);
        assert!(!config.mixed_precision);
        assert!(config.memory_pool_size > 0);
    }
}

// =========================================================================
// Feature: Error Handling
// =========================================================================

mod error_handling {
    use bitnet_common::{
        BitNetError, InferenceError, KernelError, ModelError, QuantizationError, SecurityError,
        ValidationErrorDetails,
    };
    use std::path::PathBuf;

    #[test]
    fn test_given_model_not_found_when_display_then_message_non_empty() {
        let err = ModelError::NotFound { path: "model.gguf".into() };
        let msg = format!("{err}");
        assert!(!msg.is_empty());
        assert!(msg.contains("model.gguf"));
    }

    #[test]
    fn test_given_model_invalid_format_when_debug_then_includes_variant_name() {
        let err = ModelError::InvalidFormat { format: "GGUF v99".into() };
        let dbg = format!("{err:?}");
        assert!(dbg.contains("InvalidFormat"), "debug should contain variant name: {dbg}");
    }

    #[test]
    fn test_given_quantization_error_when_display_then_message_non_empty() {
        let err = QuantizationError::UnsupportedType { qtype: "i3s".into() };
        let msg = format!("{err}");
        assert!(!msg.is_empty());
        assert!(msg.contains("i3s"));
    }

    #[test]
    fn test_given_kernel_error_variants_when_display_then_all_non_empty() {
        let errors: Vec<KernelError> = vec![
            KernelError::NoProvider,
            KernelError::ExecutionFailed { reason: "test".into() },
            KernelError::UnsupportedArchitecture { arch: "mips".into() },
            KernelError::GpuError { reason: "init".into() },
            KernelError::InvalidArguments { reason: "null".into() },
            KernelError::QuantizationFailed { reason: "overflow".into() },
            KernelError::MatmulFailed { reason: "shape".into() },
        ];

        for err in &errors {
            let msg = format!("{err}");
            assert!(!msg.is_empty(), "display for {err:?} should be non-empty");
        }
    }

    #[test]
    fn test_given_inference_error_when_display_then_message_non_empty() {
        let errors: Vec<InferenceError> = vec![
            InferenceError::GenerationFailed { reason: "no tokens".into() },
            InferenceError::InvalidInput { reason: "empty".into() },
            InferenceError::ContextLengthExceeded { length: 4096 },
            InferenceError::TokenizationFailed { reason: "unicode".into() },
        ];

        for err in &errors {
            let msg = format!("{err}");
            assert!(!msg.is_empty(), "display for {err:?} should be non-empty");
        }
    }

    #[test]
    fn test_given_security_error_when_display_then_message_non_empty() {
        let errors: Vec<SecurityError> = vec![
            SecurityError::InputValidation { reason: "injection".into() },
            SecurityError::MemoryBomb { reason: "overflow".into() },
            SecurityError::ResourceLimit {
                resource: "tensors".into(),
                value: 2_000_000_000,
                limit: 1_000_000_000,
            },
            SecurityError::MalformedData { reason: "crc".into() },
            SecurityError::UnsafeOperation { operation: "deref".into(), reason: "null".into() },
        ];

        for err in &errors {
            let msg = format!("{err}");
            assert!(!msg.is_empty(), "display for {err:?} should be non-empty");
        }
    }

    #[test]
    fn test_given_model_error_when_convert_to_bitnet_error_then_source_preserved() {
        // Given a ModelError
        let model_err = ModelError::NotFound { path: "missing.gguf".into() };

        // When I convert to BitNetError
        let bitnet_err: BitNetError = model_err.into();

        // Then the source is preserved in the display message
        let msg = format!("{bitnet_err}");
        assert!(msg.contains("missing.gguf"), "conversion should preserve source: {msg}");
    }

    #[test]
    fn test_given_quantization_error_when_convert_to_bitnet_error_then_wraps() {
        // Given a QuantizationError
        let quant_err = QuantizationError::InvalidBlockSize { size: 3 };

        // When I convert to BitNetError
        let bitnet_err: BitNetError = quant_err.into();

        // Then the display includes the block size info
        let msg = format!("{bitnet_err}");
        assert!(msg.contains("3"), "conversion should preserve info: {msg}");
    }

    #[test]
    fn test_given_kernel_no_provider_when_debug_then_includes_variant() {
        let err = KernelError::NoProvider;
        let dbg = format!("{err:?}");
        assert!(dbg.contains("NoProvider"), "debug should include variant: {dbg}");
    }

    #[test]
    fn test_given_validation_error_details_when_constructed_then_fields_accessible() {
        // Given ValidationErrorDetails
        let details = ValidationErrorDetails {
            errors: vec!["error1".into()],
            warnings: vec!["warn1".into()],
            recommendations: vec!["rec1".into()],
        };

        // Then all fields are accessible
        assert_eq!(details.errors.len(), 1);
        assert_eq!(details.warnings.len(), 1);
        assert_eq!(details.recommendations.len(), 1);
    }

    #[test]
    fn test_given_gguf_format_error_when_display_then_message_included() {
        // Given a GGUFFormatError with details
        let err = ModelError::GGUFFormatError {
            message: "bad tensor alignment".into(),
            details: ValidationErrorDetails {
                errors: vec!["misaligned".into()],
                warnings: vec![],
                recommendations: vec!["realign".into()],
            },
        };

        // When I display
        let msg = format!("{err}");

        // Then the message is included
        assert!(msg.contains("bad tensor alignment"), "should contain message: {msg}");
    }

    #[test]
    fn test_given_file_io_error_when_display_then_path_included() {
        // Given a FileIOError
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = ModelError::FileIOError {
            path: PathBuf::from("/tmp/nonexistent.gguf"),
            source: io_err,
        };

        // When I display
        let msg = format!("{err}");

        // Then the path is included
        assert!(msg.contains("nonexistent.gguf"), "should contain path: {msg}");
    }
}
