//! Property-based tests for error type conversions.
//!
//! Key invariants tested:
//! - `From<ModelError>` → `BitNetError::Model`: Display output contains original message
//! - `From<QuantizationError>` → `BitNetError::Quantization`: message preserved
//! - `From<KernelError>` → `BitNetError::Kernel`: message preserved
//! - `From<InferenceError>` → `BitNetError::Inference`: message preserved
//! - `From<SecurityError>` → `BitNetError::Security`: message preserved
//! - All `BitNetError` variants produce non-empty Display strings
//! - `ValidationErrorDetails`: default has empty vecs; pushed items are preserved

use bitnet_common::error::{
    BitNetError, InferenceError, KernelError, ModelError, QuantizationError, SecurityError,
    ValidationErrorDetails,
};
use proptest::prelude::*;

// ── Arbitrary error message strategy ─────────────────────────────────────────

fn arb_nonempty_string() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_ ]{1,80}".prop_map(|s| s.trim().to_string()).prop_filter(
        "must be non-empty after trim",
        |s| !s.is_empty(),
    )
}

// ── From<ModelError> preserves message ───────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Converting `ModelError::NotFound` to `BitNetError` preserves the path
    /// in the Display output.
    #[test]
    fn prop_model_error_not_found_preserves_path(path in arb_nonempty_string()) {
        let err = ModelError::NotFound { path: path.clone() };
        let bitnet_err: BitNetError = err.into();
        let display = format!("{}", bitnet_err);
        prop_assert!(
            display.contains(&path),
            "Display '{}' must contain path '{}'", display, path
        );
    }

    /// Converting `ModelError::InvalidFormat` preserves the format string.
    #[test]
    fn prop_model_error_invalid_format_preserves(fmt in arb_nonempty_string()) {
        let err = ModelError::InvalidFormat { format: fmt.clone() };
        let bitnet_err: BitNetError = err.into();
        let display = format!("{}", bitnet_err);
        prop_assert!(
            display.contains(&fmt),
            "Display '{}' must contain format '{}'", display, fmt
        );
    }

    /// Converting `ModelError::LoadingFailed` preserves the reason.
    #[test]
    fn prop_model_error_loading_failed_preserves(reason in arb_nonempty_string()) {
        let err = ModelError::LoadingFailed { reason: reason.clone() };
        let bitnet_err: BitNetError = err.into();
        let display = format!("{}", bitnet_err);
        prop_assert!(
            display.contains(&reason),
            "Display '{}' must contain reason '{}'", display, reason
        );
    }
}

// ── From<QuantizationError> preserves message ────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// QuantizationError::UnsupportedType → BitNetError preserves qtype string.
    #[test]
    fn prop_quant_error_unsupported_preserves(qtype in arb_nonempty_string()) {
        let err = QuantizationError::UnsupportedType { qtype: qtype.clone() };
        let bitnet_err: BitNetError = err.into();
        let display = format!("{}", bitnet_err);
        prop_assert!(display.contains(&qtype));
    }

    /// QuantizationError::QuantizationFailed → BitNetError preserves reason.
    #[test]
    fn prop_quant_error_failed_preserves(reason in arb_nonempty_string()) {
        let err = QuantizationError::QuantizationFailed { reason: reason.clone() };
        let bitnet_err: BitNetError = err.into();
        let display = format!("{}", bitnet_err);
        prop_assert!(display.contains(&reason));
    }

    /// QuantizationError::InvalidBlockSize → BitNetError preserves size.
    #[test]
    fn prop_quant_error_invalid_block_size_preserves(size in 1usize..1_000_000) {
        let err = QuantizationError::InvalidBlockSize { size };
        let bitnet_err: BitNetError = err.into();
        let display = format!("{}", bitnet_err);
        prop_assert!(
            display.contains(&size.to_string()),
            "Display '{}' must contain size '{}'", display, size
        );
    }
}

// ── From<KernelError> preserves message ──────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// KernelError::ExecutionFailed → BitNetError preserves reason.
    #[test]
    fn prop_kernel_error_exec_preserves(reason in arb_nonempty_string()) {
        let err = KernelError::ExecutionFailed { reason: reason.clone() };
        let bitnet_err: BitNetError = err.into();
        let display = format!("{}", bitnet_err);
        prop_assert!(display.contains(&reason));
    }

    /// KernelError::UnsupportedArchitecture → BitNetError preserves arch.
    #[test]
    fn prop_kernel_error_arch_preserves(arch in arb_nonempty_string()) {
        let err = KernelError::UnsupportedArchitecture { arch: arch.clone() };
        let bitnet_err: BitNetError = err.into();
        let display = format!("{}", bitnet_err);
        prop_assert!(display.contains(&arch));
    }

    /// KernelError::NoProvider → BitNetError display is non-empty.
    #[test]
    fn prop_kernel_error_no_provider_nonempty(_seed in 0u8..1) {
        let err = KernelError::NoProvider;
        let bitnet_err: BitNetError = err.into();
        let display = format!("{}", bitnet_err);
        prop_assert!(!display.is_empty());
    }
}

// ── From<InferenceError> preserves message ───────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// InferenceError::GenerationFailed → BitNetError preserves reason.
    #[test]
    fn prop_inference_error_gen_preserves(reason in arb_nonempty_string()) {
        let err = InferenceError::GenerationFailed { reason: reason.clone() };
        let bitnet_err: BitNetError = err.into();
        let display = format!("{}", bitnet_err);
        prop_assert!(display.contains(&reason));
    }

    /// InferenceError::ContextLengthExceeded → BitNetError preserves length.
    #[test]
    fn prop_inference_error_context_preserves(length in 1usize..1_000_000) {
        let err = InferenceError::ContextLengthExceeded { length };
        let bitnet_err: BitNetError = err.into();
        let display = format!("{}", bitnet_err);
        prop_assert!(display.contains(&length.to_string()));
    }
}

// ── From<SecurityError> preserves message ────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// SecurityError::InputValidation → BitNetError preserves reason.
    #[test]
    fn prop_security_error_input_preserves(reason in arb_nonempty_string()) {
        let err = SecurityError::InputValidation { reason: reason.clone() };
        let bitnet_err: BitNetError = err.into();
        let display = format!("{}", bitnet_err);
        prop_assert!(display.contains(&reason));
    }

    /// SecurityError::ResourceLimit → BitNetError preserves resource name and values.
    #[test]
    fn prop_security_error_resource_preserves(
        resource in arb_nonempty_string(),
        value in 1u64..1_000_000,
        limit in 1u64..1_000_000,
    ) {
        let err = SecurityError::ResourceLimit {
            resource: resource.clone(),
            value,
            limit,
        };
        let bitnet_err: BitNetError = err.into();
        let display = format!("{}", bitnet_err);
        prop_assert!(display.contains(&resource));
        prop_assert!(display.contains(&value.to_string()));
        prop_assert!(display.contains(&limit.to_string()));
    }
}

// ── ValidationErrorDetails ───────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Pushed errors, warnings, and recommendations are preserved.
    #[test]
    fn prop_validation_details_preserves_items(
        errors in prop::collection::vec(arb_nonempty_string(), 0..5),
        warnings in prop::collection::vec(arb_nonempty_string(), 0..5),
        recommendations in prop::collection::vec(arb_nonempty_string(), 0..5),
    ) {
        let details = ValidationErrorDetails {
            errors: errors.clone(),
            warnings: warnings.clone(),
            recommendations: recommendations.clone(),
        };
        prop_assert_eq!(&details.errors, &errors);
        prop_assert_eq!(&details.warnings, &warnings);
        prop_assert_eq!(&details.recommendations, &recommendations);
    }
}

// ── All BitNetError variants have non-empty Display ──────────────────────────

proptest! {
    /// Config, Validation, and StrictMode string variants produce non-empty Display.
    #[test]
    fn prop_string_variant_display_nonempty(msg in arb_nonempty_string()) {
        let config_err = BitNetError::Config(msg.clone());
        let config_display = format!("{}", config_err);
        prop_assert!(!config_display.is_empty());

        let val_err = BitNetError::Validation(msg.clone());
        let val_display = format!("{}", val_err);
        prop_assert!(!val_display.is_empty());

        let strict_err = BitNetError::StrictMode(msg);
        let strict_display = format!("{}", strict_err);
        prop_assert!(!strict_display.is_empty());
    }
}
