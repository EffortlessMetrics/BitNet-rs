//! Snapshot wave 27 — bitnet-kernels Display/Debug output stability.

use std::time::Duration;

use bitnet_kernels::capability_matrix::{
    CapabilityEntry, CompatibilityReport, DeviceCapabilityMatrix, DeviceClass, DeviceProfile,
    OperationCategory, PrecisionSupport, SupportLevel,
};
use bitnet_kernels::opencl_registry::{DeviceConstraints, KernelOp, KernelRegistry, KernelVariant};
use bitnet_kernels::perf_tracker::{KernelTiming, PerfTracker, format_perf_report};
use bitnet_kernels::simd_diagnostics::SimdCapabilities;

// ---------------------------------------------------------------------------
// DeviceClass Display
// ---------------------------------------------------------------------------

#[test]
fn snapshot_device_class_display_all() {
    let displays: Vec<String> = DeviceClass::ALL.iter().map(|d| d.to_string()).collect();
    insta::assert_debug_snapshot!("device_class_display_all", displays);
}

// ---------------------------------------------------------------------------
// OperationCategory Display
// ---------------------------------------------------------------------------

#[test]
fn snapshot_operation_category_display_all() {
    let displays: Vec<String> = OperationCategory::ALL.iter().map(|o| o.to_string()).collect();
    insta::assert_debug_snapshot!("operation_category_display_all", displays);
}

// ---------------------------------------------------------------------------
// PrecisionSupport Display
// ---------------------------------------------------------------------------

#[test]
fn snapshot_precision_support_display_all() {
    let displays: Vec<String> = PrecisionSupport::ALL.iter().map(|p| p.to_string()).collect();
    insta::assert_debug_snapshot!("precision_support_display_all", displays);
}

// ---------------------------------------------------------------------------
// SupportLevel Display
// ---------------------------------------------------------------------------

#[test]
fn snapshot_support_level_display_variants() {
    let levels = vec![
        SupportLevel::Full(0.95),
        SupportLevel::Full(1.0),
        SupportLevel::Partial("no FP16 native".into()),
        SupportLevel::Emulated,
        SupportLevel::Unsupported,
    ];
    let displays: Vec<String> = levels.iter().map(|l| l.to_string()).collect();
    insta::assert_debug_snapshot!("support_level_display_variants", displays);
}

// ---------------------------------------------------------------------------
// CompatibilityReport
// ---------------------------------------------------------------------------

#[test]
fn snapshot_compatibility_report_ready() {
    let profile = DeviceProfile {
        device_class: DeviceClass::NvidiaCuda,
        name: "NVIDIA RTX 4090".into(),
        compute_units: 128,
        memory_gb: 24,
        capabilities: vec![CapabilityEntry::new(
            OperationCategory::MatrixOps,
            PrecisionSupport::FP16,
            SupportLevel::Full(0.98),
        )],
    };
    let required = vec![(OperationCategory::MatrixOps, PrecisionSupport::FP16)];
    let report = CompatibilityReport::generate(&profile, &required);
    insta::assert_snapshot!("compat_report_ready", report.to_string());
}

#[test]
fn snapshot_compatibility_report_incomplete() {
    let profile = DeviceProfile {
        device_class: DeviceClass::CpuScalar,
        name: "CPU Scalar".into(),
        compute_units: 1,
        memory_gb: 16,
        capabilities: vec![CapabilityEntry::new(
            OperationCategory::MatrixOps,
            PrecisionSupport::FP32,
            SupportLevel::Full(0.5),
        )],
    };
    let required = vec![
        (OperationCategory::MatrixOps, PrecisionSupport::FP32),
        (OperationCategory::QuantizedOps, PrecisionSupport::Binary),
        (OperationCategory::AttentionOps, PrecisionSupport::FP16),
    ];
    let report = CompatibilityReport::generate(&profile, &required);
    insta::assert_snapshot!("compat_report_incomplete", report.to_string());
}

#[test]
fn snapshot_compatibility_report_debug() {
    let profile = DeviceProfile {
        device_class: DeviceClass::IntelArc,
        name: "Intel Arc A770".into(),
        compute_units: 32,
        memory_gb: 16,
        capabilities: vec![
            CapabilityEntry::new(
                OperationCategory::MatrixOps,
                PrecisionSupport::FP16,
                SupportLevel::Full(0.92),
            ),
            CapabilityEntry::new(
                OperationCategory::NormOps,
                PrecisionSupport::FP32,
                SupportLevel::Full(0.85),
            ),
        ],
    };
    let required = vec![
        (OperationCategory::MatrixOps, PrecisionSupport::FP16),
        (OperationCategory::NormOps, PrecisionSupport::FP32),
    ];
    let report = CompatibilityReport::generate(&profile, &required);
    insta::assert_debug_snapshot!("compat_report_debug", report);
}

// ---------------------------------------------------------------------------
// DeviceCapabilityMatrix builtin profiles
// ---------------------------------------------------------------------------

#[test]
fn snapshot_builtin_profiles_names() {
    let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
    let names: Vec<&str> = matrix.profiles().iter().map(|p| p.name.as_str()).collect();
    insta::assert_debug_snapshot!("builtin_profile_names", names);
}

// ---------------------------------------------------------------------------
// KernelOp Display
// ---------------------------------------------------------------------------

#[test]
fn snapshot_kernel_op_display_all() {
    let displays: Vec<String> = KernelOp::ALL.iter().map(|o| o.to_string()).collect();
    insta::assert_debug_snapshot!("kernel_op_display_all", displays);
}

// ---------------------------------------------------------------------------
// KernelVariant Display
// ---------------------------------------------------------------------------

#[test]
fn snapshot_kernel_variant_display_all() {
    let variants = [
        KernelVariant::OpenClScalar,
        KernelVariant::OpenClTiled,
        KernelVariant::OpenClVectorized,
        KernelVariant::CpuSimd,
        KernelVariant::CpuScalar,
    ];
    let displays: Vec<String> = variants.iter().map(|v| v.to_string()).collect();
    insta::assert_debug_snapshot!("kernel_variant_display_all", displays);
}

// ---------------------------------------------------------------------------
// KernelRegistry summary
// ---------------------------------------------------------------------------

#[test]
fn snapshot_kernel_registry_a770_summary() {
    let registry = KernelRegistry::with_default_a770_kernels();
    insta::assert_snapshot!("kernel_registry_a770_summary", registry.summary());
}

// ---------------------------------------------------------------------------
// DeviceConstraints
// ---------------------------------------------------------------------------

#[test]
fn snapshot_device_constraints_a770_debug() {
    let constraints = DeviceConstraints::a770_defaults();
    insta::assert_debug_snapshot!("device_constraints_a770", constraints);
}

// ---------------------------------------------------------------------------
// PerfTracker report
// ---------------------------------------------------------------------------

#[test]
fn snapshot_perf_report_empty() {
    let tracker = PerfTracker::new();
    let report = format_perf_report(&tracker);
    insta::assert_snapshot!("perf_report_empty", report);
}

#[test]
fn snapshot_perf_report_with_timings() {
    let mut tracker = PerfTracker::new();
    tracker.record(KernelTiming::new("matmul_fp16", Duration::from_micros(1200), 65536));
    tracker.record(KernelTiming::new("matmul_fp16", Duration::from_micros(1100), 65536));
    tracker.record(KernelTiming::new("softmax", Duration::from_micros(300), 32000));
    tracker
        .record(KernelTiming::new("layer_norm", Duration::from_micros(150), 4096).with_flops(8192));
    let report = format_perf_report(&tracker);
    insta::assert_snapshot!("perf_report_with_timings", report);
}

// ---------------------------------------------------------------------------
// SimdCapabilities
// ---------------------------------------------------------------------------

#[test]
fn snapshot_simd_capabilities_debug() {
    let caps = SimdCapabilities::detect();
    // Only snapshot the arch field for stability across machines
    insta::assert_snapshot!("simd_caps_arch", caps.arch);
}

#[test]
fn snapshot_simd_capabilities_summary() {
    let caps = SimdCapabilities::detect();
    let summary = caps.summary();
    // Summary is machine-dependent but should be non-empty
    assert!(!summary.is_empty());
}
