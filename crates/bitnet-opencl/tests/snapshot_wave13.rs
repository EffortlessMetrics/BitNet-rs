//! Snapshot wave 13 — stabilize bitnet-opencl public API surface.
//!
//! Pins Debug and Display representations of dispatcher, SPIR-V, and
//! validation types so that changes to public formatting are caught by CI.

use bitnet_opencl::{
    BackendInfo, BackendStatus, ComparisonResult, CompileOptions, CompilerBackend,
    DispatchDecision, DispatchStrategy, DistributionStats, DivergencePoint, KernelSource,
    Operation, OptimizationLevel, SpirVModule, ValidationFinding, ValidationReport,
    ValidationSeverity,
};

// ── Operation ────────────────────────────────────────────────────────────────

#[test]
fn operation_all_variants_debug() {
    let ops = [
        Operation::MatMul,
        Operation::Quantize,
        Operation::Dequantize,
        Operation::Softmax,
        Operation::LayerNorm,
        Operation::Attention,
        Operation::RoPE,
        Operation::Sampling,
    ];
    insta::assert_debug_snapshot!("operation_variants", ops);
}

// ── BackendStatus ────────────────────────────────────────────────────────────

#[test]
fn backend_status_variants_debug() {
    let statuses = vec![
        BackendStatus::Available,
        BackendStatus::Unavailable("driver not found".into()),
        BackendStatus::Degraded("thermal throttling".into()),
    ];
    insta::assert_debug_snapshot!("backend_status_variants", statuses);
}

// ── DispatchStrategy ─────────────────────────────────────────────────────────

#[test]
fn dispatch_strategy_variants_debug() {
    let strategies = vec![
        DispatchStrategy::Priority,
        DispatchStrategy::RoundRobin,
        DispatchStrategy::LoadBased,
        DispatchStrategy::SpecificBackend("cuda".into()),
    ];
    insta::assert_debug_snapshot!("dispatch_strategy_variants", strategies);
}

// ── DispatchDecision ─────────────────────────────────────────────────────────

#[test]
fn dispatch_decision_debug() {
    let decision = DispatchDecision {
        chosen_backend: "opencl-arc".into(),
        reason: "highest priority score".into(),
        alternatives_available: vec!["cpu".into(), "vulkan".into()],
        operation: Operation::MatMul,
    };
    insta::assert_debug_snapshot!("dispatch_decision_typical", decision);
}

// ── BackendInfo ──────────────────────────────────────────────────────────────

#[test]
fn backend_info_debug() {
    let info = BackendInfo {
        name: "intel-arc-a770".into(),
        status: BackendStatus::Available,
        capabilities: vec![Operation::MatMul, Operation::Softmax, Operation::Attention],
        priority_score: 100,
    };
    insta::assert_debug_snapshot!("backend_info_available", info);
}

// ── CompileOptions ───────────────────────────────────────────────────────────

#[test]
fn compile_options_default_debug() {
    let opts = CompileOptions::default();
    insta::assert_debug_snapshot!("compile_options_default", opts);
}

#[test]
fn compile_options_custom_debug() {
    let opts = CompileOptions {
        target_device: Some("Intel Arc A770".into()),
        optimization_level: OptimizationLevel::Basic,
        defines: vec![("BLOCK_SIZE".into(), "256".into()), ("USE_FP16".into(), "1".into())],
    };
    insta::assert_debug_snapshot!("compile_options_custom", opts);
}

// ── SpirVModule ──────────────────────────────────────────────────────────────

#[test]
fn spirv_module_debug() {
    let module = SpirVModule {
        bytecode: vec![0x03, 0x02, 0x23, 0x07], // SPIR-V magic
        source_hash: "a1b2c3d4e5f6".into(),
        compiler: Some(CompilerBackend::Clang),
    };
    insta::assert_debug_snapshot!("spirv_module_with_compiler", module);
}

// ── KernelSource ─────────────────────────────────────────────────────────────

#[test]
fn kernel_source_variants_debug() {
    let sources = vec![
        KernelSource::ClSource("__kernel void add() {}".into()),
        KernelSource::SpirV(vec![0x03, 0x02, 0x23, 0x07]),
    ];
    insta::assert_debug_snapshot!("kernel_source_variants", sources);
}

// ── ValidationReport ─────────────────────────────────────────────────────────

#[test]
fn validation_report_empty_debug() {
    let report = ValidationReport::new();
    insta::assert_debug_snapshot!("validation_report_empty", report);
}

#[test]
fn validation_report_with_findings_debug() {
    let mut report = ValidationReport::new();
    report.add(ValidationSeverity::Info, "Model size: 2.1 GB", None);
    report.add(
        ValidationSeverity::Warning,
        "FP16 not supported on device",
        Some("Enable mixed precision fallback".into()),
    );
    report.add(
        ValidationSeverity::Error,
        "Insufficient VRAM: need 4 GB, have 2 GB",
        Some("Use a smaller model or enable CPU offloading".into()),
    );
    insta::assert_debug_snapshot!("validation_report_mixed_findings", report);
}

#[test]
fn validation_finding_display() {
    let finding = ValidationFinding {
        severity: ValidationSeverity::Warning,
        message: "LayerNorm weights appear quantized".into(),
        suggestion: Some("Re-export with F16 LayerNorm".into()),
    };
    insta::assert_snapshot!("validation_finding_display", finding.to_string());
}

// ── DistributionStats Display ────────────────────────────────────────────────

#[test]
fn distribution_stats_normal_display() {
    let stats = DistributionStats {
        mean: 0.001_234,
        std_dev: 0.998_765,
        min: -3.125,
        max: 3.125,
        nan_count: 0,
        inf_count: 0,
        element_count: 2048,
    };
    insta::assert_snapshot!("distribution_stats_normal", stats.to_string());
}

#[test]
fn distribution_stats_with_anomalies_display() {
    let stats = DistributionStats {
        mean: f64::NAN,
        std_dev: f64::INFINITY,
        min: f32::NEG_INFINITY,
        max: f32::INFINITY,
        nan_count: 5,
        inf_count: 3,
        element_count: 100,
    };
    insta::assert_snapshot!("distribution_stats_anomalies", stats.to_string());
}

// ── ComparisonResult Display ─────────────────────────────────────────────────

#[test]
fn comparison_result_match_display() {
    let result = ComparisonResult {
        matching: true,
        max_diff: 0.000_001,
        mean_diff: 0.000_000_1,
        outlier_count: 0,
        element_count: 4096,
    };
    insta::assert_snapshot!("comparison_result_match", result.to_string());
}

#[test]
fn comparison_result_mismatch_display() {
    let result = ComparisonResult {
        matching: false,
        max_diff: 0.523,
        mean_diff: 0.042,
        outlier_count: 17,
        element_count: 1024,
    };
    insta::assert_snapshot!("comparison_result_mismatch", result.to_string());
}

// ── DivergencePoint ──────────────────────────────────────────────────────────

#[test]
fn divergence_point_debug() {
    let point = DivergencePoint {
        step: 42,
        metric: 0.987_654_321,
        description: "L2 distance exceeded threshold after attention layer 12".into(),
    };
    insta::assert_debug_snapshot!("divergence_point_typical", point);
}
