//! Wave 26 snapshot tests for `bitnet-common` — shape errors, tensor
//! validation errors, model errors, config enums, memory pool stats,
//! strict mode computation type, and RoPE scaling.
//!
//! Pins Debug/Display output so unintentional changes are caught at review.

// =========================================================================
// Section 1 — ShapeError Display
// =========================================================================

use bitnet_common::tensor_validation::ShapeError;

#[test]
fn w26_shape_error_matmul_mismatch_display() {
    let e = ShapeError::MatmulMismatch { a_inner: 512, b_inner: 768 };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_shape_error_matmul_rank_display() {
    let e = ShapeError::MatmulRank { a_ndim: 0, b_ndim: 2 };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_shape_error_matmul_batch_mismatch_display() {
    let e = ShapeError::MatmulBatchMismatch { a_batch: vec![2, 4], b_batch: vec![3, 4] };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_shape_error_broadcast_incompatible_display() {
    let e = ShapeError::BroadcastIncompatible { dim: 1, a: 3, b: 5 };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_shape_error_attention_shape_display() {
    let e = ShapeError::AttentionShape {
        q: vec![1, 8, 64, 128],
        k: vec![1, 8, 32, 128],
        v: vec![1, 8, 32, 64],
        reason: "V head_dim != K head_dim".into(),
    };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_shape_error_reshape_element_count_display() {
    let e = ShapeError::ReshapeElementCount { from_count: 1024, to: vec![32, 33], to_count: 1056 };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_shape_error_transpose_axis_out_of_range_display() {
    let e = ShapeError::TransposeAxisOutOfRange { axis: 5, ndim: 3 };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_shape_error_transpose_not_permutation_display() {
    let e = ShapeError::TransposeNotPermutation { axes: vec![0, 0, 2], ndim: 3 };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_shape_error_empty_shape_display() {
    let e = ShapeError::EmptyShape;
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_shape_error_all_variants_debug() {
    let errors: Vec<ShapeError> = vec![
        ShapeError::MatmulMismatch { a_inner: 64, b_inner: 128 },
        ShapeError::MatmulRank { a_ndim: 0, b_ndim: 1 },
        ShapeError::BroadcastIncompatible { dim: 0, a: 1, b: 7 },
        ShapeError::EmptyShape,
    ];
    insta::assert_debug_snapshot!(errors);
}

// =========================================================================
// Section 2 — TensorValidationError Display
// =========================================================================

use bitnet_common::tensor_validation::TensorValidationError;

#[test]
fn w26_tensor_val_zero_dimension_display() {
    let e = TensorValidationError::ZeroDimension { axis: 2, shape: vec![8, 4, 0, 16] };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_tensor_val_total_elements_exceeded_display() {
    let e =
        TensorValidationError::TotalElementsExceeded { total: 2_000_000_000, max: 1_000_000_000 };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_tensor_val_dimensions_exceeded_display() {
    let e = TensorValidationError::DimensionsExceeded { ndim: 12, max: 8 };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_tensor_val_nan_detected_display() {
    let e = TensorValidationError::NanDetected { index: 42 };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_tensor_val_inf_detected_display() {
    let e = TensorValidationError::InfDetected { index: 99, value: f32::INFINITY };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_tensor_val_value_out_of_range_display() {
    let e = TensorValidationError::ValueOutOfRange { index: 7, value: 100.5, min: -1.0, max: 1.0 };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_tensor_val_stride_inconsistent_display() {
    let e = TensorValidationError::StrideInconsistent { axis: 1, expected: 128, actual: 64 };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_tensor_val_alignment_violation_display() {
    let e = TensorValidationError::AlignmentViolation { required: 64, actual: 17 };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_tensor_val_all_variants_debug() {
    let errors: Vec<TensorValidationError> = vec![
        TensorValidationError::ZeroDimension { axis: 0, shape: vec![0] },
        TensorValidationError::NanDetected { index: 0 },
        TensorValidationError::AlignmentViolation { required: 32, actual: 1 },
    ];
    insta::assert_debug_snapshot!(errors);
}

// =========================================================================
// Section 3 — ModelError Display
// =========================================================================

use bitnet_common::{ModelError, ValidationErrorDetails};

#[test]
fn w26_model_error_not_found_display() {
    let e = ModelError::NotFound { path: "/models/missing.gguf".into() };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_model_error_invalid_format_display() {
    let e = ModelError::InvalidFormat { format: "safetensors-v99".into() };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_model_error_loading_failed_display() {
    let e = ModelError::LoadingFailed { reason: "header checksum mismatch".into() };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_model_error_unsupported_version_display() {
    let e = ModelError::UnsupportedVersion { version: "4.0".into() };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_model_error_gguf_format_display() {
    let e = ModelError::GGUFFormatError {
        message: "invalid tensor alignment".into(),
        details: ValidationErrorDetails {
            errors: vec!["alignment must be power of 2".into()],
            warnings: vec!["unusual block size 37".into()],
            recommendations: vec!["re-export with standard alignment".into()],
        },
    };
    insta::assert_snapshot!(e.to_string());
}

#[test]
fn w26_validation_error_details_debug() {
    let details = ValidationErrorDetails {
        errors: vec!["bad magic".into(), "truncated header".into()],
        warnings: vec![],
        recommendations: vec!["regenerate GGUF".into()],
    };
    insta::assert_debug_snapshot!(details);
}

// =========================================================================
// Section 4 — Config enums
// =========================================================================

use bitnet_common::config::{ActivationType, ModelFormat, NormType, RopeScaling};

#[test]
fn w26_activation_type_all_debug() {
    let types = vec![ActivationType::Silu, ActivationType::Relu2, ActivationType::Gelu];
    insta::assert_debug_snapshot!(types);
}

#[test]
fn w26_norm_type_all_debug() {
    let types = vec![NormType::LayerNorm, NormType::RmsNorm];
    insta::assert_debug_snapshot!(types);
}

#[test]
fn w26_model_format_all_debug() {
    let formats = vec![ModelFormat::Gguf, ModelFormat::SafeTensors, ModelFormat::HuggingFace];
    insta::assert_debug_snapshot!(formats);
}

#[test]
fn w26_rope_scaling_debug() {
    let rs = RopeScaling { scaling_type: "linear".into(), factor: 4.0 };
    insta::assert_debug_snapshot!(rs);
}

#[test]
fn w26_rope_scaling_dynamic() {
    let rs = RopeScaling { scaling_type: "dynamic".into(), factor: 2.0 };
    insta::assert_debug_snapshot!(rs);
}

// =========================================================================
// Section 5 — Memory pool stats
// =========================================================================

use bitnet_common::memory_pool::PoolStats;

#[test]
fn w26_pool_stats_default() {
    let stats = PoolStats::default();
    insta::assert_debug_snapshot!(stats);
}

#[test]
fn w26_pool_stats_active() {
    let stats =
        PoolStats { hits: 950, misses: 50, pooled_bytes: 8_388_608, active_bytes: 2_097_152 };
    insta::assert_debug_snapshot!(stats);
}

#[test]
fn w26_pool_stats_total_allocations() {
    let stats = PoolStats { hits: 100, misses: 25, pooled_bytes: 0, active_bytes: 0 };
    insta::assert_snapshot!(format!("total_allocations={}", stats.total_allocations()));
}

// =========================================================================
// Section 6 — Strict mode ComputationType
// =========================================================================

use bitnet_common::strict_mode::ComputationType;

#[test]
fn w26_computation_type_real_debug() {
    insta::assert_debug_snapshot!(ComputationType::Real);
}

#[test]
fn w26_computation_type_mock_debug() {
    insta::assert_debug_snapshot!(ComputationType::Mock);
}

#[test]
fn w26_computation_type_serde_roundtrip() {
    let real = serde_json::to_string(&ComputationType::Real).unwrap();
    let mock = serde_json::to_string(&ComputationType::Mock).unwrap();
    insta::assert_snapshot!(format!("real={real} mock={mock}"));
}
