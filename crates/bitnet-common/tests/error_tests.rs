//! Comprehensive error handling tests for bitnet-common

use bitnet_common::*;
use proptest::prelude::*;
use std::io;

#[test]
fn test_bitnet_error_variants() {
    // Test all BitNetError variants
    let config_error = BitNetError::Config("test config error".to_string());
    assert!(matches!(config_error, BitNetError::Config(_)));
    assert_eq!(format!("{}", config_error), "Configuration error: test config error");

    let validation_error = BitNetError::Validation("test validation error".to_string());
    assert!(matches!(validation_error, BitNetError::Validation(_)));
    assert_eq!(format!("{}", validation_error), "Validation error: test validation error");

    // Test nested error types
    let model_error = BitNetError::Model(ModelError::NotFound { path: "/test/path".to_string() });
    assert!(matches!(model_error, BitNetError::Model(_)));
    assert!(format!("{}", model_error).contains("/test/path"));

    let quantization_error =
        BitNetError::Quantization(QuantizationError::UnsupportedType { qtype: "TEST".to_string() });
    assert!(matches!(quantization_error, BitNetError::Quantization(_)));
    assert!(format!("{}", quantization_error).contains("TEST"));

    let kernel_error = BitNetError::Kernel(KernelError::NoProvider);
    assert!(matches!(kernel_error, BitNetError::Kernel(_)));
    assert_eq!(format!("{}", kernel_error), "Kernel error: No available kernel provider");

    let inference_error = BitNetError::Inference(InferenceError::GenerationFailed {
        reason: "test reason".to_string(),
    });
    assert!(matches!(inference_error, BitNetError::Inference(_)));
    assert!(format!("{}", inference_error).contains("test reason"));
}

#[test]
fn test_model_error_variants() {
    let not_found = ModelError::NotFound { path: "/nonexistent/model.gguf".to_string() };
    assert_eq!(format!("{}", not_found), "Model not found: /nonexistent/model.gguf");

    let invalid_format = ModelError::InvalidFormat { format: "UNKNOWN".to_string() };
    assert_eq!(format!("{}", invalid_format), "Invalid model format: UNKNOWN");

    let loading_failed = ModelError::LoadingFailed { reason: "corrupted file".to_string() };
    assert_eq!(format!("{}", loading_failed), "Model loading failed: corrupted file");

    let unsupported_version = ModelError::UnsupportedVersion { version: "v2.0".to_string() };
    assert_eq!(format!("{}", unsupported_version), "Unsupported model version: v2.0");
}

#[test]
fn test_quantization_error_variants() {
    let unsupported_type = QuantizationError::UnsupportedType { qtype: "INT8".to_string() };
    assert_eq!(format!("{}", unsupported_type), "Unsupported quantization type: INT8");

    let quantization_failed =
        QuantizationError::QuantizationFailed { reason: "precision loss".to_string() };
    assert_eq!(format!("{}", quantization_failed), "Quantization failed: precision loss");

    let invalid_block_size = QuantizationError::InvalidBlockSize { size: 0 };
    assert_eq!(format!("{}", invalid_block_size), "Invalid block size: 0");
}

#[test]
fn test_kernel_error_variants() {
    let no_provider = KernelError::NoProvider;
    assert_eq!(format!("{}", no_provider), "No available kernel provider");

    let execution_failed =
        KernelError::ExecutionFailed { reason: "CUDA out of memory".to_string() };
    assert_eq!(format!("{}", execution_failed), "Kernel execution failed: CUDA out of memory");

    let unsupported_arch = KernelError::UnsupportedArchitecture { arch: "ARM64".to_string() };
    assert_eq!(format!("{}", unsupported_arch), "Unsupported architecture: ARM64");

    let gpu_error = KernelError::GpuError { reason: "device not found".to_string() };
    assert_eq!(format!("{}", gpu_error), "GPU error: device not found");
}

#[test]
fn test_inference_error_variants() {
    let generation_failed =
        InferenceError::GenerationFailed { reason: "model crashed".to_string() };
    assert_eq!(format!("{}", generation_failed), "Generation failed: model crashed");

    let invalid_input = InferenceError::InvalidInput { reason: "empty prompt".to_string() };
    assert_eq!(format!("{}", invalid_input), "Invalid input: empty prompt");

    let context_exceeded = InferenceError::ContextLengthExceeded { length: 4096 };
    assert_eq!(format!("{}", context_exceeded), "Context length exceeded: 4096");

    let tokenization_failed =
        InferenceError::TokenizationFailed { reason: "unknown token".to_string() };
    assert_eq!(format!("{}", tokenization_failed), "Tokenization failed: unknown token");
}

#[test]
fn test_error_conversions() {
    // Test From implementations for nested errors
    let model_error = ModelError::NotFound { path: "test".to_string() };
    let bitnet_error: BitNetError = model_error.into();
    assert!(matches!(bitnet_error, BitNetError::Model(_)));

    let quantization_error = QuantizationError::UnsupportedType { qtype: "test".to_string() };
    let bitnet_error: BitNetError = quantization_error.into();
    assert!(matches!(bitnet_error, BitNetError::Quantization(_)));

    let kernel_error = KernelError::NoProvider;
    let bitnet_error: BitNetError = kernel_error.into();
    assert!(matches!(bitnet_error, BitNetError::Kernel(_)));

    let inference_error = InferenceError::InvalidInput { reason: "test".to_string() };
    let bitnet_error: BitNetError = inference_error.into();
    assert!(matches!(bitnet_error, BitNetError::Inference(_)));

    // Test IO error conversion
    let io_error = io::Error::new(io::ErrorKind::NotFound, "file not found");
    let bitnet_error: BitNetError = io_error.into();
    assert!(matches!(bitnet_error, BitNetError::Io(_)));
}

#[test]
fn test_result_type_alias() {
    // Test that our Result type alias works correctly
    fn test_function() -> Result<i32> {
        Ok(42)
    }

    fn test_error_function() -> Result<i32> {
        Err(BitNetError::Config("test error".to_string()))
    }

    assert_eq!(test_function().unwrap(), 42);
    assert!(test_error_function().is_err());
}

#[test]
fn test_error_debug_formatting() {
    let error = BitNetError::Config("debug test".to_string());
    let debug_str = format!("{:?}", error);
    assert!(debug_str.contains("Config"));
    assert!(debug_str.contains("debug test"));

    let model_error = ModelError::NotFound { path: "debug/path".to_string() };
    let debug_str = format!("{:?}", model_error);
    assert!(debug_str.contains("NotFound"));
    assert!(debug_str.contains("debug/path"));
}

#[test]
fn test_error_chain() {
    // Test error chaining with nested errors
    let model_error = ModelError::LoadingFailed { reason: "inner error".to_string() };
    let bitnet_error = BitNetError::Model(model_error);

    let error_string = format!("{}", bitnet_error);
    assert!(error_string.contains("Model error"));
    assert!(error_string.contains("inner error"));
}

// Property-based tests for error handling
proptest! {
    #[test]
    fn test_config_error_with_arbitrary_strings(message in "\\PC*") {
        let error = BitNetError::Config(message.clone());
        let formatted = format!("{}", error);
        assert!(formatted.contains(&message));
    }

    #[test]
    fn test_model_path_error_with_arbitrary_paths(path in "\\PC*") {
        let error = ModelError::NotFound { path: path.clone() };
        let formatted = format!("{}", error);
        assert!(formatted.contains(&path));
    }

    #[test]
    fn test_quantization_type_error_with_arbitrary_types(qtype in "\\PC*") {
        let error = QuantizationError::UnsupportedType { qtype: qtype.clone() };
        let formatted = format!("{}", error);
        assert!(formatted.contains(&qtype));
    }

    #[test]
    fn test_block_size_error_with_arbitrary_sizes(size in any::<usize>()) {
        let error = QuantizationError::InvalidBlockSize { size };
        let formatted = format!("{}", error);
        assert!(formatted.contains(&size.to_string()));
    }

    #[test]
    fn test_context_length_error_with_arbitrary_lengths(length in any::<usize>()) {
        let error = InferenceError::ContextLengthExceeded { length };
        let formatted = format!("{}", error);
        assert!(formatted.contains(&length.to_string()));
    }
}

#[test]
fn test_error_source_chain() {
    // Test that errors properly implement the Error trait's source method
    use std::error::Error;

    let model_error = ModelError::LoadingFailed { reason: "test".to_string() };
    let bitnet_error = BitNetError::Model(model_error);

    // The source should be the inner model error
    assert!(bitnet_error.source().is_some());
}

#[test]
fn test_error_send_sync() {
    // Ensure errors are Send + Sync for use in async contexts
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<BitNetError>();
    assert_send_sync::<ModelError>();
    assert_send_sync::<QuantizationError>();
    assert_send_sync::<KernelError>();
    assert_send_sync::<InferenceError>();
}
