//! Integration tests for bitnet-common error types.
//!
//! Validates that all error variants implement Display with meaningful
//! messages, error chains are preserved through conversions, and error
//! codes remain unique and contextual.

use bitnet_common::{
    BitNetError, InferenceError, KernelError, ModelError, QuantizationError, SecurityError,
    ValidationErrorDetails,
};

// ── Display implementations ──────────────────────────────────────────

#[test]
fn integration_bitnet_error_display_model() {
    let err = BitNetError::Model(ModelError::NotFound { path: "/tmp/missing.gguf".to_string() });
    let msg = format!("{err}");
    assert!(!msg.is_empty());
    assert!(msg.contains("missing.gguf"), "Display should include path: {msg}");
}

#[test]
fn integration_bitnet_error_display_quantization() {
    let err = BitNetError::Quantization(QuantizationError::UnsupportedType {
        qtype: "Q4_K_M".to_string(),
    });
    let msg = format!("{err}");
    assert!(msg.contains("Q4_K_M"), "Display should include qtype: {msg}");
}

#[test]
fn integration_bitnet_error_display_kernel() {
    let err = BitNetError::Kernel(KernelError::NoProvider);
    let msg = format!("{err}");
    assert!(
        msg.contains("provider") || msg.contains("kernel"),
        "Display should mention provider: {msg}"
    );
}

#[test]
fn integration_bitnet_error_display_inference() {
    let err = BitNetError::Inference(InferenceError::ContextLengthExceeded { length: 8192 });
    let msg = format!("{err}");
    assert!(msg.contains("8192"), "Display should include length: {msg}");
}

#[test]
fn integration_bitnet_error_display_config() {
    let err = BitNetError::Config("invalid batch size".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("invalid batch size"), "Display should include config message: {msg}");
}

#[test]
fn integration_bitnet_error_display_validation() {
    let err = BitNetError::Validation("LayerNorm gamma out of range".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("LayerNorm"), "Display should include validation detail: {msg}");
}

#[test]
fn integration_bitnet_error_display_strict_mode() {
    let err = BitNetError::StrictMode("mock inference rejected".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("mock"), "Display should include strict mode reason: {msg}");
}

// ── Error chains (From conversions) ──────────────────────────────────

#[test]
fn integration_model_error_converts_to_bitnet_error() {
    let model_err = ModelError::InvalidFormat { format: "safetensors-v3".to_string() };
    let bitnet_err: BitNetError = model_err.into();
    let msg = format!("{bitnet_err}");
    assert!(msg.contains("safetensors-v3"));
}

#[test]
fn integration_quantization_error_converts_to_bitnet_error() {
    let q_err = QuantizationError::InvalidBlockSize { size: 13 };
    let bitnet_err: BitNetError = q_err.into();
    let msg = format!("{bitnet_err}");
    assert!(msg.contains("13"));
}

#[test]
fn integration_kernel_error_converts_to_bitnet_error() {
    let k_err = KernelError::ExecutionFailed { reason: "out of memory".to_string() };
    let bitnet_err: BitNetError = k_err.into();
    let msg = format!("{bitnet_err}");
    assert!(msg.contains("out of memory"));
}

#[test]
fn integration_inference_error_converts_to_bitnet_error() {
    let i_err = InferenceError::TokenizationFailed { reason: "unknown token".to_string() };
    let bitnet_err: BitNetError = i_err.into();
    let msg = format!("{bitnet_err}");
    assert!(msg.contains("unknown token"));
}

#[test]
fn integration_security_error_converts_to_bitnet_error() {
    let s_err = SecurityError::MemoryBomb { reason: "tensor too large".to_string() };
    let bitnet_err: BitNetError = s_err.into();
    let msg = format!("{bitnet_err}");
    assert!(msg.contains("tensor too large"));
}

#[test]
fn integration_io_error_converts_to_bitnet_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
    let bitnet_err: BitNetError = io_err.into();
    let msg = format!("{bitnet_err}");
    assert!(msg.contains("gone") || msg.contains("IO"), "IO error chain: {msg}");
}

// ── Error messages are meaningful ────────────────────────────────────

#[test]
fn integration_all_model_error_variants_have_content() {
    let variants: Vec<ModelError> = vec![
        ModelError::NotFound { path: "/a".to_string() },
        ModelError::InvalidFormat { format: "xyz".to_string() },
        ModelError::LoadingFailed { reason: "corrupt".to_string() },
        ModelError::UnsupportedVersion { version: "99".to_string() },
        ModelError::GGUFFormatError {
            message: "bad header".to_string(),
            details: ValidationErrorDetails {
                errors: vec!["e1".into()],
                warnings: vec![],
                recommendations: vec![],
            },
        },
    ];
    for v in &variants {
        let msg = format!("{v}");
        assert!(msg.len() > 3, "ModelError Display too short: '{msg}'");
    }
}

#[test]
fn integration_all_kernel_error_variants_have_content() {
    let variants: Vec<KernelError> = vec![
        KernelError::NoProvider,
        KernelError::ExecutionFailed { reason: "fail".to_string() },
        KernelError::UnsupportedArchitecture { arch: "riscv".to_string() },
        KernelError::GpuError { reason: "oom".to_string() },
        KernelError::UnsupportedHardware {
            required: "avx512".to_string(),
            available: "sse2".to_string(),
        },
        KernelError::InvalidArguments { reason: "null ptr".to_string() },
        KernelError::QuantizationFailed { reason: "bad block".to_string() },
        KernelError::MatmulFailed { reason: "dim".to_string() },
    ];
    for v in &variants {
        let msg = format!("{v}");
        assert!(msg.len() > 3, "KernelError Display too short: '{msg}'");
    }
}

#[test]
fn integration_all_security_error_variants_have_content() {
    let variants: Vec<SecurityError> = vec![
        SecurityError::InputValidation { reason: "xss".to_string() },
        SecurityError::MemoryBomb { reason: "2TB alloc".to_string() },
        SecurityError::ResourceLimit {
            resource: "tensors".to_string(),
            value: 2_000_000_000,
            limit: 1_000_000_000,
        },
        SecurityError::MalformedData { reason: "truncated".to_string() },
        SecurityError::UnsafeOperation {
            operation: "raw_ptr".to_string(),
            reason: "unaligned".to_string(),
        },
    ];
    for v in &variants {
        let msg = format!("{v}");
        assert!(msg.len() > 3, "SecurityError short: '{msg}'");
    }
}

// ── Error codes / variant discrimination ─────────────────────────────

#[test]
fn integration_error_variants_are_distinguishable() {
    let errors: Vec<BitNetError> = vec![
        BitNetError::Config("cfg".into()),
        BitNetError::Validation("val".into()),
        BitNetError::StrictMode("strict".into()),
        BitNetError::Kernel(KernelError::NoProvider),
        BitNetError::Inference(InferenceError::InvalidInput { reason: "bad".into() }),
    ];
    let messages: Vec<String> = errors.iter().map(|e| format!("{e}")).collect();
    // All messages should be unique
    let unique: std::collections::HashSet<&String> = messages.iter().collect();
    assert_eq!(unique.len(), messages.len(), "error messages should be unique: {messages:?}");
}

#[test]
fn integration_error_debug_includes_variant_name() {
    let err = BitNetError::Kernel(KernelError::NoProvider);
    let debug = format!("{err:?}");
    assert!(debug.contains("Kernel"), "Debug should include variant: {debug}");
    assert!(debug.contains("NoProvider"), "Debug should include inner: {debug}");
}

#[test]
fn integration_security_resource_limit_includes_all_fields() {
    let err = SecurityError::ResourceLimit {
        resource: "memory".to_string(),
        value: 5_000_000_000,
        limit: 4_000_000_000,
    };
    let msg = format!("{err}");
    assert!(msg.contains("memory"), "missing resource: {msg}");
    assert!(msg.contains("5000000000"), "missing value: {msg}");
    assert!(msg.contains("4000000000"), "missing limit: {msg}");
}
