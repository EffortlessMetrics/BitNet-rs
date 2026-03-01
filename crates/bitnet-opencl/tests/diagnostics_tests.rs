//! Integration tests for GPU runtime diagnostics.

use bitnet_opencl::{
    DiagnosticReport, GpuDiagnostics, collect_system_info, format_json, format_report,
};

// ── System info tests ────────────────────────────────────────────────────────

#[test]
fn system_info_collection_works() {
    let info = collect_system_info();
    assert!(!info.os_name.is_empty());
    assert!(info.cpu_cores >= 1);
    assert!(!info.bitnet_version.is_empty());
    assert!(!info.rust_version.is_empty());
}

#[test]
fn system_info_os_is_valid() {
    let info = collect_system_info();
    let known = ["windows", "linux", "macos"];
    assert!(known.contains(&info.os_name.as_str()), "unexpected OS: {}", info.os_name);
}

#[test]
fn system_info_cpu_name_not_empty() {
    let info = collect_system_info();
    assert!(!info.cpu_name.is_empty(), "cpu_name should not be empty");
}

// ── Report formatting tests ──────────────────────────────────────────────────

#[test]
fn report_formatting_includes_all_sections() {
    let diag = GpuDiagnostics::new();
    let report = diag.run_full();
    let text = format_report(&report);

    assert!(text.contains("System"), "missing System section");
    assert!(text.contains("GPU Driver"), "missing GPU Driver section");
    assert!(text.contains("CUDA"), "missing CUDA section");
    assert!(text.contains("OpenCL"), "missing OpenCL section");
    assert!(text.contains("Vulkan"), "missing Vulkan section");
    assert!(text.contains("GPU Memory"), "missing GPU Memory section");
    assert!(text.contains("Feature Flags"), "missing Feature Flags section");
    assert!(text.contains("Smoke Test"), "missing Smoke Test section");
    assert!(text.contains("Issues"), "missing Issues section");
}

#[test]
fn report_formatting_shows_system_info() {
    let diag = GpuDiagnostics::new();
    let report = diag.run_full();
    let text = format_report(&report);

    assert!(text.contains("OS:"));
    assert!(text.contains("CPU:"));
    assert!(text.contains("Rust:"));
    assert!(text.contains("BitNet:"));
}

#[test]
fn report_formatting_shows_feature_flags() {
    let diag = GpuDiagnostics::new();
    let report = diag.run_full();
    let text = format_report(&report);

    assert!(text.contains("cpu:"));
    assert!(text.contains("gpu:"));
    assert!(text.contains("cuda:"));
    assert!(text.contains("opencl:"));
    assert!(text.contains("vulkan:"));
}

// ── JSON formatting tests ────────────────────────────────────────────────────

#[test]
fn json_report_is_valid_json() {
    let diag = GpuDiagnostics::new();
    let report = diag.run_full();
    let json = format_json(&report);

    let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should be valid");
    assert!(parsed.is_object());
}

#[test]
fn json_report_contains_expected_keys() {
    let diag = GpuDiagnostics::new();
    let report = diag.run_full();
    let json = format_json(&report);
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let obj = parsed.as_object().unwrap();

    for key in &[
        "system",
        "driver",
        "opencl",
        "vulkan",
        "cuda",
        "memory",
        "features",
        "smoke_test",
        "issues",
        "recommendations",
    ] {
        assert!(obj.contains_key(*key), "missing key: {key}");
    }
}

#[test]
fn json_system_has_required_fields() {
    let diag = GpuDiagnostics::new();
    let report = diag.run_full();
    let json = format_json(&report);
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let system = parsed.get("system").unwrap().as_object().unwrap();

    for field in &[
        "os_name",
        "os_version",
        "kernel_version",
        "cpu_name",
        "cpu_cores",
        "total_memory_mb",
        "rust_version",
        "bitnet_version",
    ] {
        assert!(system.contains_key(*field), "missing system field: {field}");
    }
}

// ── Feature status tests ─────────────────────────────────────────────────────

#[test]
fn feature_status_reflects_compiled_features() {
    let diag = GpuDiagnostics::new();
    let f = diag.check_features();
    assert_eq!(f.cpu, cfg!(feature = "cpu"));
    assert_eq!(f.gpu, cfg!(feature = "gpu"));
    assert_eq!(f.cuda, cfg!(feature = "cuda"));
    assert_eq!(f.opencl, cfg!(feature = "opencl"));
    assert_eq!(f.vulkan, cfg!(feature = "vulkan"));
}

// ── Driver detection tests ───────────────────────────────────────────────────

#[test]
fn unknown_driver_returns_not_found() {
    let diag = GpuDiagnostics::new();
    let status = diag.check_drivers();
    // Without real GPU hardware the driver is "not found" (not an error).
    if !status.found {
        assert_eq!(status.description, "not found");
    }
}

// ── No GPU helpful message ───────────────────────────────────────────────────

#[test]
fn report_with_no_gpu_shows_helpful_issues() {
    let diag = GpuDiagnostics::new();
    let report = diag.run_full();

    // When gpu/cuda features are not compiled, we expect an issue.
    if !cfg!(any(feature = "gpu", feature = "cuda")) {
        assert!(
            report.issues.iter().any(|i| i.summary.contains("GPU feature not compiled")),
            "should mention GPU feature not compiled"
        );
    }
}

#[test]
fn report_without_gpu_has_recommendations() {
    let diag = GpuDiagnostics::new();
    let report = diag.run_full();

    if !cfg!(any(feature = "gpu", feature = "cuda")) {
        assert!(
            !report.recommendations.is_empty(),
            "should have recommendations without GPU features"
        );
    }
}

// ── Smoke test ───────────────────────────────────────────────────────────────

#[test]
fn smoke_test_with_mock_backend_passes() {
    let diag = GpuDiagnostics::new();
    let result = diag.run_smoke_test();
    assert!(result.passed, "smoke test should pass: {}", result.message);
    assert!(result.elapsed_ms < 5000, "smoke test too slow");
}

#[test]
fn smoke_test_message_is_descriptive() {
    let diag = GpuDiagnostics::new();
    let result = diag.run_smoke_test();
    assert!(
        result.message.contains("dot product") || result.message.contains("Smoke test"),
        "message should describe the test: {}",
        result.message
    );
}

// ── Serialization round-trip ─────────────────────────────────────────────────

#[test]
fn diagnostic_report_roundtrips_json() {
    let diag = GpuDiagnostics::new();
    let report = diag.run_full();
    let json = format_json(&report);
    let parsed: DiagnosticReport = serde_json::from_str(&json).expect("should deserialize back");
    assert_eq!(parsed.system.os_name, report.system.os_name);
    assert_eq!(parsed.features.cpu, report.features.cpu);
}

// ── Memory status test ───────────────────────────────────────────────────────

#[test]
fn memory_status_without_gpu_is_unavailable() {
    let diag = GpuDiagnostics::new();
    let mem = diag.check_memory();
    // Without real GPU memory probing, should report unavailable.
    assert!(!mem.available);
    assert_eq!(mem.total_mb, 0);
    assert_eq!(mem.free_mb, 0);
}
