//! Snapshot wave 9 — capability matrix & CPU fusion Display/Debug formatting.
//!
//! Pins the human-readable output of key kernel types to catch
//! accidental regressions in user-facing diagnostics.

use bitnet_kernels::capability_matrix::{
    CapabilityEntry, CompatibilityReport, DeviceCapabilityMatrix, DeviceClass, OperationCategory,
    PrecisionSupport, SupportLevel,
};
use bitnet_kernels::cpu::fusion::{FusedOp, FusionConfig, FusionError};

// ── DeviceClass Display ────────────────────────────────────────────

#[test]
fn device_class_display_all_variants() {
    let output: Vec<String> = DeviceClass::ALL.iter().map(|d| format!("{d}")).collect();
    insta::assert_snapshot!(output.join("\n"));
}

#[test]
fn device_class_debug_all_variants() {
    let output: Vec<String> = DeviceClass::ALL.iter().map(|d| format!("{d:?}")).collect();
    insta::assert_snapshot!(output.join("\n"));
}

// ── OperationCategory Display ──────────────────────────────────────

#[test]
fn operation_category_display_all_variants() {
    let output: Vec<String> = OperationCategory::ALL.iter().map(|o| format!("{o}")).collect();
    insta::assert_snapshot!(output.join("\n"));
}

// ── PrecisionSupport Display ───────────────────────────────────────

#[test]
fn precision_support_display_all_variants() {
    let output: Vec<String> = PrecisionSupport::ALL.iter().map(|p| format!("{p}")).collect();
    insta::assert_snapshot!(output.join("\n"));
}

// ── SupportLevel Display ───────────────────────────────────────────

#[test]
fn support_level_display_full() {
    insta::assert_snapshot!(format!("{}", SupportLevel::Full(0.85)));
}

#[test]
fn support_level_display_partial() {
    insta::assert_snapshot!(format!("{}", SupportLevel::Partial("no FP16 dp4a".into())));
}

#[test]
fn support_level_display_emulated() {
    insta::assert_snapshot!(format!("{}", SupportLevel::Emulated));
}

#[test]
fn support_level_display_unsupported() {
    insta::assert_snapshot!(format!("{}", SupportLevel::Unsupported));
}

// ── CompatibilityReport ────────────────────────────────────────────

#[test]
fn compatibility_report_ready_device() {
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    let profile = matrix.profile_for_class(DeviceClass::CpuSimd).unwrap();
    let required = vec![
        (OperationCategory::MatrixOps, PrecisionSupport::FP32),
        (OperationCategory::NormOps, PrecisionSupport::FP32),
    ];
    let report = CompatibilityReport::generate(profile, &required);
    insta::assert_snapshot!(format!("{report}"));
}

#[test]
fn compatibility_report_incomplete_device() {
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    let profile = matrix.profile_for_class(DeviceClass::CpuScalar).unwrap();
    let required = vec![
        (OperationCategory::MatrixOps, PrecisionSupport::FP16),
        (OperationCategory::AttentionOps, PrecisionSupport::BF16),
    ];
    let report = CompatibilityReport::generate(profile, &required);
    insta::assert_snapshot!(format!("{report}"));
}

#[test]
fn compatibility_report_summary_ready() {
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    let profile = matrix.profile_for_class(DeviceClass::NvidiaCuda).unwrap();
    let required = vec![(OperationCategory::MatrixOps, PrecisionSupport::FP32)];
    let report = CompatibilityReport::generate(profile, &required);
    insta::assert_snapshot!(report.summary());
}

// ── FusedOp Display ────────────────────────────────────────────────

#[test]
fn fused_op_display_all_variants() {
    let ops = [
        FusedOp::RmsNormLinear,
        FusedOp::GeluLinear,
        FusedOp::SoftmaxMask,
        FusedOp::AddNormalize,
        FusedOp::ScaleAndAdd,
    ];
    let output: Vec<String> = ops.iter().map(|o| format!("{o}")).collect();
    insta::assert_snapshot!(output.join("\n"));
}

// ── FusionError Display ────────────────────────────────────────────

#[test]
fn fusion_error_dimension_mismatch_display() {
    let err = FusionError::DimensionMismatch { expected: 512, got: 256 };
    insta::assert_snapshot!(format!("{err}"));
}

#[test]
fn fusion_error_invalid_config_display() {
    let err = FusionError::InvalidConfig("min_fusion_size must be > 0".into());
    insta::assert_snapshot!(format!("{err}"));
}

#[test]
fn fusion_error_empty_input_display() {
    insta::assert_snapshot!(format!("{}", FusionError::EmptyInput));
}

// ── FusionConfig Debug ─────────────────────────────────────────────

#[test]
fn fusion_config_default_debug() {
    insta::assert_debug_snapshot!(FusionConfig::default());
}

#[test]
fn fusion_config_disabled_debug() {
    insta::assert_debug_snapshot!(FusionConfig::disabled());
}

// ── Builtin profiles ───────────────────────────────────────────────

#[test]
fn builtin_profile_count() {
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    insta::assert_snapshot!(format!("profile_count={}", matrix.profiles().len()));
}

#[test]
fn builtin_profile_names() {
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    let names: Vec<&str> = matrix.profiles().iter().map(|p| p.name.as_str()).collect();
    insta::assert_snapshot!(names.join("\n"));
}

#[test]
fn capability_entry_debug_snapshot() {
    let entry = CapabilityEntry::new(
        OperationCategory::QuantizedOps,
        PrecisionSupport::Binary,
        SupportLevel::Full(0.95),
    );
    insta::assert_debug_snapshot!(entry);
}
