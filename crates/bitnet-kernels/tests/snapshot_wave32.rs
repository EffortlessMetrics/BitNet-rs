//! Snapshot wave 32 — kernel capability matrix, device profiles, SIMD
//! capability descriptions, and support level formatting.

use bitnet_kernels::capability_matrix::{
    CapabilityEntry, CompatibilityReport, DeviceCapabilityMatrix, DeviceClass, OperationCategory,
    PrecisionSupport, SupportLevel, cpu_scalar, intel_arc_a770,
};

// ── DeviceClass display ─────────────────────────────────────────────

#[test]
fn device_class_display_all_variants() {
    let output: Vec<String> = DeviceClass::ALL.iter().map(|d| d.to_string()).collect();
    insta::assert_snapshot!(output.join("\n"));
}

#[test]
fn device_class_debug_all() {
    insta::assert_debug_snapshot!(DeviceClass::ALL);
}

// ── OperationCategory display ───────────────────────────────────────

#[test]
fn operation_category_display_all() {
    let output: Vec<String> = OperationCategory::ALL.iter().map(|o| o.to_string()).collect();
    insta::assert_snapshot!(output.join("\n"));
}

// ── PrecisionSupport display ────────────────────────────────────────

#[test]
fn precision_support_display_all() {
    let precisions = [
        PrecisionSupport::FP32,
        PrecisionSupport::FP16,
        PrecisionSupport::BF16,
        PrecisionSupport::INT8,
        PrecisionSupport::INT4,
        PrecisionSupport::Binary,
    ];
    let output: Vec<String> = precisions.iter().map(|p| p.to_string()).collect();
    insta::assert_snapshot!(output.join("\n"));
}

// ── SupportLevel display ────────────────────────────────────────────

#[test]
fn support_level_full() {
    insta::assert_snapshot!(SupportLevel::Full(0.85).to_string());
}

#[test]
fn support_level_partial() {
    insta::assert_snapshot!(
        SupportLevel::Partial("limited memory bandwidth".to_string()).to_string()
    );
}

#[test]
fn support_level_emulated() {
    insta::assert_snapshot!(SupportLevel::Emulated.to_string());
}

#[test]
fn support_level_unsupported() {
    insta::assert_snapshot!(SupportLevel::Unsupported.to_string());
}

// ── CapabilityEntry debug ───────────────────────────────────────────

#[test]
fn capability_entry_debug() {
    let entry = CapabilityEntry::new(
        OperationCategory::MatrixOps,
        PrecisionSupport::FP16,
        SupportLevel::Full(0.92),
    );
    insta::assert_debug_snapshot!(entry);
}

#[test]
fn capability_entry_quantized_binary() {
    let entry = CapabilityEntry::new(
        OperationCategory::QuantizedOps,
        PrecisionSupport::Binary,
        SupportLevel::Full(0.95),
    );
    insta::assert_debug_snapshot!(entry);
}

// ── DeviceCapabilityMatrix ──────────────────────────────────────────

#[test]
fn capability_matrix_profile_count() {
    let matrix = DeviceCapabilityMatrix::new();
    let profile_names: Vec<&str> = matrix.profiles().iter().map(|p| p.name.as_str()).collect();
    insta::assert_debug_snapshot!(profile_names);
}

// ── CompatibilityReport ─────────────────────────────────────────────

#[test]
fn compatibility_report_cpu_scalar() {
    let profile = cpu_scalar();
    let required = vec![
        (OperationCategory::MatrixOps, PrecisionSupport::FP32),
        (OperationCategory::NormOps, PrecisionSupport::FP32),
    ];
    let report = CompatibilityReport::generate(&profile, &required);
    insta::assert_snapshot!(report.to_string());
}

#[test]
fn compatibility_report_intel_arc_summary() {
    let profile = intel_arc_a770();
    let required = vec![
        (OperationCategory::MatrixOps, PrecisionSupport::FP32),
        (OperationCategory::AttentionOps, PrecisionSupport::FP16),
    ];
    let report = CompatibilityReport::generate(&profile, &required);
    insta::assert_snapshot!(report.summary());
}
