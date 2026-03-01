//! Edge-case tests for bitnet-common error types and security limits.
//!
//! Tests cover:
//! - BitNetError variant construction and Display formatting
//! - ModelError, QuantizationError, KernelError, InferenceError, SecurityError
//! - From conversions between error types
//! - ValidationErrorDetails construction and field access
//! - SecurityLimits defaults and custom values
//! - Device enum: construction, is_cpu, is_cuda, Debug

use bitnet_common::{
    BitNetError, Device, InferenceError, KernelError, ModelError, QuantizationError,
    SecurityError, SecurityLimits, ValidationErrorDetails,
};

// ---------------------------------------------------------------------------
// BitNetError — variant construction
// ---------------------------------------------------------------------------

#[test]
fn bitnet_error_model_variant() {
    let err = BitNetError::Model(ModelError::NotFound { path: "test.gguf".to_string() });
    let msg = format!("{}", err);
    assert!(msg.contains("test.gguf") || msg.contains("not found") || msg.contains("Model"));
}

#[test]
fn bitnet_error_quantization_variant() {
    let err =
        BitNetError::Quantization(QuantizationError::UnsupportedType { qtype: "Q99".into() });
    let msg = format!("{}", err);
    assert!(msg.contains("Q99") || msg.contains("unsupported") || msg.contains("Quantization"));
}

#[test]
fn bitnet_error_kernel_variant() {
    let err = BitNetError::Kernel(KernelError::NoProvider);
    let msg = format!("{}", err);
    assert!(
        msg.contains("provider") || msg.contains("kernel") || msg.contains("Kernel"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn bitnet_error_inference_variant() {
    let err = BitNetError::Inference(InferenceError::InvalidInput {
        reason: "bad prompt".to_string(),
    });
    let msg = format!("{}", err);
    assert!(
        msg.contains("bad prompt") || msg.contains("Invalid") || msg.contains("input"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn bitnet_error_security_variant() {
    let err = BitNetError::Security(SecurityError::MemoryBomb {
        reason: "allocation too large".to_string(),
    });
    let msg = format!("{}", err);
    assert!(
        msg.contains("allocation")
            || msg.contains("memory")
            || msg.contains("Security")
            || msg.contains("attack"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn bitnet_error_io_variant() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
    let err = BitNetError::Io(io_err);
    let msg = format!("{}", err);
    assert!(
        msg.contains("file missing") || msg.contains("Io") || msg.contains("I/O"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn bitnet_error_config_variant() {
    let err = BitNetError::Config("bad config value".to_string());
    let msg = format!("{}", err);
    assert!(
        msg.contains("bad config value") || msg.contains("Config"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn bitnet_error_validation_variant() {
    let err = BitNetError::Validation("invalid tensor shape".to_string());
    let msg = format!("{}", err);
    assert!(
        msg.contains("invalid tensor shape") || msg.contains("Validation"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn bitnet_error_strict_mode_variant() {
    let err = BitNetError::StrictMode("mock path not allowed".to_string());
    let msg = format!("{}", err);
    assert!(
        msg.contains("mock path") || msg.contains("Strict"),
        "unexpected: {}",
        msg
    );
}

// ---------------------------------------------------------------------------
// ModelError variants
// ---------------------------------------------------------------------------

#[test]
fn model_error_loading_failed() {
    let err = ModelError::LoadingFailed { reason: "corrupt header".to_string() };
    let msg = format!("{}", err);
    assert!(
        msg.contains("corrupt header") || msg.contains("loading"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn model_error_invalid_format() {
    let err = ModelError::InvalidFormat { format: "not GGUF".to_string() };
    let msg = format!("{}", err);
    assert!(
        msg.contains("not GGUF") || msg.contains("format"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn model_error_unsupported_version() {
    let err = ModelError::UnsupportedVersion { version: "99".to_string() };
    let msg = format!("{}", err);
    assert!(msg.contains("99") || msg.contains("version"), "unexpected: {}", msg);
}

// ---------------------------------------------------------------------------
// KernelError variants
// ---------------------------------------------------------------------------

#[test]
fn kernel_error_execution_failed() {
    let err = KernelError::ExecutionFailed { reason: "CUDA OOM".to_string() };
    let msg = format!("{}", err);
    assert!(
        msg.contains("CUDA OOM") || msg.contains("execution"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn kernel_error_matmul_failed() {
    let err = KernelError::MatmulFailed { reason: "dimension mismatch".to_string() };
    let msg = format!("{}", err);
    assert!(
        msg.contains("dimension") || msg.contains("matmul"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn kernel_error_gpu_error() {
    let err = KernelError::GpuError { reason: "out of memory".to_string() };
    let msg = format!("{}", err);
    assert!(
        msg.contains("out of memory") || msg.contains("GPU"),
        "unexpected: {}",
        msg
    );
}

// ---------------------------------------------------------------------------
// InferenceError variants
// ---------------------------------------------------------------------------

#[test]
fn inference_error_generation_failed() {
    let err = InferenceError::GenerationFailed { reason: "max tokens exceeded".to_string() };
    let msg = format!("{}", err);
    assert!(
        msg.contains("max tokens") || msg.contains("generation"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn inference_error_context_length_exceeded() {
    let err = InferenceError::ContextLengthExceeded { length: 32768 };
    let msg = format!("{}", err);
    assert!(
        msg.contains("32768") || msg.contains("context"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn inference_error_tokenization_failed() {
    let err = InferenceError::TokenizationFailed { reason: "unknown token".to_string() };
    let msg = format!("{}", err);
    assert!(
        msg.contains("unknown token") || msg.contains("tokenization"),
        "unexpected: {}",
        msg
    );
}

// ---------------------------------------------------------------------------
// SecurityError variants
// ---------------------------------------------------------------------------

#[test]
fn security_error_input_validation() {
    let err = SecurityError::InputValidation { reason: "injection attempt".to_string() };
    let msg = format!("{}", err);
    assert!(
        msg.contains("injection") || msg.contains("validation"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn security_error_resource_limit() {
    let err = SecurityError::ResourceLimit {
        resource: "tensor_elements".to_string(),
        value: 2_000_000_000,
        limit: 1_000_000_000,
    };
    let msg = format!("{}", err);
    assert!(
        msg.contains("tensor_elements") || msg.contains("resource") || msg.contains("limit"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn security_error_memory_bomb() {
    let err = SecurityError::MemoryBomb { reason: "allocation too large".to_string() };
    let msg = format!("{}", err);
    assert!(
        msg.contains("allocation") || msg.contains("memory") || msg.contains("attack"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn security_error_malformed_data() {
    let err = SecurityError::MalformedData { reason: "invalid UTF-8 in header".to_string() };
    let msg = format!("{}", err);
    assert!(
        msg.contains("invalid UTF-8") || msg.contains("malformed"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn security_error_unsafe_operation() {
    let err = SecurityError::UnsafeOperation {
        operation: "eval".to_string(),
        reason: "code injection".to_string(),
    };
    let msg = format!("{}", err);
    assert!(
        msg.contains("eval") || msg.contains("unsafe") || msg.contains("injection"),
        "unexpected: {}",
        msg
    );
}

// ---------------------------------------------------------------------------
// From conversions
// ---------------------------------------------------------------------------

#[test]
fn from_io_error_to_bitnet_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
    let err: BitNetError = io_err.into();
    match err {
        BitNetError::Io(_) => {} // expected
        other => panic!("expected Io variant, got: {:?}", other),
    }
}

#[test]
fn from_model_error_to_bitnet_error() {
    let model_err = ModelError::NotFound { path: "model.gguf".to_string() };
    let err: BitNetError = model_err.into();
    match err {
        BitNetError::Model(_) => {} // expected
        other => panic!("expected Model variant, got: {:?}", other),
    }
}

#[test]
fn from_kernel_error_to_bitnet_error() {
    let kernel_err = KernelError::NoProvider;
    let err: BitNetError = kernel_err.into();
    match err {
        BitNetError::Kernel(_) => {} // expected
        other => panic!("expected Kernel variant, got: {:?}", other),
    }
}

#[test]
fn from_quantization_error_to_bitnet_error() {
    let quant_err = QuantizationError::InvalidBlockSize { size: 0 };
    let err: BitNetError = quant_err.into();
    match err {
        BitNetError::Quantization(_) => {} // expected
        other => panic!("expected Quantization variant, got: {:?}", other),
    }
}

#[test]
fn from_inference_error_to_bitnet_error() {
    let inf_err = InferenceError::GenerationFailed { reason: "test".to_string() };
    let err: BitNetError = inf_err.into();
    match err {
        BitNetError::Inference(_) => {} // expected
        other => panic!("expected Inference variant, got: {:?}", other),
    }
}

#[test]
fn from_security_error_to_bitnet_error() {
    let sec_err = SecurityError::MemoryBomb { reason: "test".to_string() };
    let err: BitNetError = sec_err.into();
    match err {
        BitNetError::Security(_) => {} // expected
        other => panic!("expected Security variant, got: {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// ValidationErrorDetails
// ---------------------------------------------------------------------------

#[test]
fn validation_error_details_empty() {
    let details = ValidationErrorDetails {
        errors: vec![],
        warnings: vec![],
        recommendations: vec![],
    };
    assert!(details.errors.is_empty());
    assert!(details.warnings.is_empty());
    assert!(details.recommendations.is_empty());
}

#[test]
fn validation_error_details_with_content() {
    let details = ValidationErrorDetails {
        errors: vec!["Missing weight tensor".to_string()],
        warnings: vec!["Large vocab size".to_string()],
        recommendations: vec!["Use quantized model".to_string()],
    };
    assert_eq!(details.errors.len(), 1);
    assert_eq!(details.warnings.len(), 1);
    assert_eq!(details.recommendations.len(), 1);
    assert!(details.errors[0].contains("Missing weight"));
}

#[test]
fn validation_error_details_debug() {
    let details = ValidationErrorDetails {
        errors: vec!["err1".to_string()],
        warnings: vec![],
        recommendations: vec![],
    };
    let dbg = format!("{:?}", details);
    assert!(dbg.contains("err1"));
}

#[test]
fn validation_error_details_clone() {
    let details = ValidationErrorDetails {
        errors: vec!["test".to_string()],
        warnings: vec![],
        recommendations: vec![],
    };
    let cloned = details.clone();
    assert_eq!(cloned.errors, details.errors);
}

// ---------------------------------------------------------------------------
// SecurityLimits
// ---------------------------------------------------------------------------

#[test]
fn security_limits_default() {
    let limits = SecurityLimits::default();
    assert_eq!(limits.max_tensor_elements, 1_000_000_000);
    assert_eq!(limits.max_memory_allocation, 4 * 1024 * 1024 * 1024); // 4 GB
    assert_eq!(limits.max_metadata_size, 100 * 1024 * 1024); // 100 MB
    assert_eq!(limits.max_string_length, 1024 * 1024); // 1 MB
    assert_eq!(limits.max_array_length, 1_000_000);
}

#[test]
fn security_limits_custom() {
    let limits = SecurityLimits {
        max_tensor_elements: 500_000,
        max_memory_allocation: 1024,
        max_metadata_size: 256,
        max_string_length: 128,
        max_array_length: 64,
    };
    assert_eq!(limits.max_tensor_elements, 500_000);
    assert_eq!(limits.max_memory_allocation, 1024);
    assert_eq!(limits.max_metadata_size, 256);
    assert_eq!(limits.max_string_length, 128);
    assert_eq!(limits.max_array_length, 64);
}

// ---------------------------------------------------------------------------
// Device enum
// ---------------------------------------------------------------------------

#[test]
fn device_cpu() {
    let dev = Device::Cpu;
    assert!(dev.is_cpu());
    assert!(!dev.is_cuda());
}

#[test]
fn device_cuda() {
    let dev = Device::new_cuda(0).unwrap();
    assert!(!dev.is_cpu());
    assert!(dev.is_cuda());
}

#[test]
fn device_cuda_direct() {
    let dev = Device::Cuda(1);
    assert!(dev.is_cuda());
    assert!(!dev.is_cpu());
}

#[test]
fn device_debug() {
    let dev = Device::Cpu;
    let dbg = format!("{:?}", dev);
    assert!(dbg.contains("Cpu"));
}

#[test]
fn device_hip() {
    let dev = Device::Hip(0);
    assert!(!dev.is_cpu());
    assert!(!dev.is_cuda());
    assert!(dev.is_hip());
}

#[test]
fn device_npu() {
    let dev = Device::Npu;
    assert!(!dev.is_cpu());
    assert!(!dev.is_cuda());
    assert!(dev.is_npu());
}

#[test]
fn device_metal() {
    let dev = Device::Metal;
    assert!(!dev.is_cpu());
    assert!(!dev.is_cuda());
}

#[test]
fn device_opencl() {
    let dev = Device::new_opencl(0).unwrap();
    assert!(!dev.is_cpu());
    assert!(dev.is_opencl());
}

#[test]
fn device_default_is_cpu() {
    let dev = Device::default();
    assert!(dev.is_cpu());
}

#[test]
fn device_equality() {
    assert_eq!(Device::Cpu, Device::Cpu);
    assert_ne!(Device::Cpu, Device::Cuda(0));
    assert_eq!(Device::Cuda(0), Device::Cuda(0));
    assert_ne!(Device::Cuda(0), Device::Cuda(1));
}

#[test]
fn device_clone_copy() {
    let dev = Device::Cuda(0);
    let copied = dev;
    assert_eq!(dev, copied);
}

// ---------------------------------------------------------------------------
// Error Debug impls
// ---------------------------------------------------------------------------

#[test]
fn bitnet_error_debug_impl() {
    let err = BitNetError::Config("test config".to_string());
    let dbg = format!("{:?}", err);
    assert!(dbg.contains("Config") || dbg.contains("test config"));
}

#[test]
fn model_error_debug_impl() {
    let err = ModelError::NotFound { path: "model.bin".to_string() };
    let dbg = format!("{:?}", err);
    assert!(dbg.contains("NotFound") || dbg.contains("model.bin"));
}

#[test]
fn security_error_debug_impl() {
    let err = SecurityError::UnsafeOperation {
        operation: "eval".to_string(),
        reason: "injection".to_string(),
    };
    let dbg = format!("{:?}", err);
    assert!(dbg.contains("UnsafeOperation") || dbg.contains("eval"));
}

// ---------------------------------------------------------------------------
// QuantizationError variants
// ---------------------------------------------------------------------------

#[test]
fn quantization_error_invalid_block_size() {
    let err = QuantizationError::InvalidBlockSize { size: 0 };
    let msg = format!("{}", err);
    assert!(msg.contains("0") || msg.contains("block"), "unexpected: {}", msg);
}

#[test]
fn quantization_error_resource_limit() {
    let err = QuantizationError::ResourceLimit { reason: "out of memory".to_string() };
    let msg = format!("{}", err);
    assert!(
        msg.contains("out of memory") || msg.contains("resource"),
        "unexpected: {}",
        msg
    );
}
