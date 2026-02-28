//! Integration tests for GPU receipt validation.
//!
//! Covers all 8 validation gates, JSON serialization, hybrid merging,
//! and strict-mode environment interaction.

use bitnet_opencl::{GpuReceipt, GpuReceiptValidator, HybridReceipt};
use serial_test::serial;
use std::collections::HashMap;

/// Build a receipt that passes all gates.
fn valid_receipt() -> GpuReceipt {
    let mut timings = HashMap::new();
    timings.insert("i2s_gemv".to_string(), 2.3);
    timings.insert("rope_apply".to_string(), 0.8);
    GpuReceipt {
        backend: "cuda".to_string(),
        device_name: "NVIDIA A100".to_string(),
        device_memory_mb: 40960,
        kernel_timings: timings,
        total_gpu_time_ms: 55.0,
        memory_peak_mb: 8192,
        compute_path: "real".to_string(),
        driver_version: "535.129.03".to_string(),
        opencl_version: None,
    }
}

// ── Gate 1: compute_path ────────────────────────────────────────────

#[test]
fn gate1_real_compute_path_accepted() {
    let v = GpuReceiptValidator::new(false);
    assert!(v.validate(&valid_receipt()).is_ok());
}

#[test]
fn gate1_mock_compute_path_rejected() {
    let mut r = valid_receipt();
    r.compute_path = "mock".to_string();
    let errs = GpuReceiptValidator::new(false).validate(&r).unwrap_err();
    assert!(errs.iter().any(|e| e.gate == 1));
}

#[test]
fn gate1_arbitrary_compute_path_rejected() {
    let mut r = valid_receipt();
    r.compute_path = "simulated".to_string();
    let errs = GpuReceiptValidator::new(false).validate(&r).unwrap_err();
    assert!(errs.iter().any(|e| e.gate == 1));
}

// ── Gate 2: backend ─────────────────────────────────────────────────

#[test]
fn gate2_cuda_accepted() {
    let mut r = valid_receipt();
    r.backend = "cuda".to_string();
    assert!(GpuReceiptValidator::new(false).validate(&r).is_ok());
}

#[test]
fn gate2_opencl_accepted() {
    let mut r = valid_receipt();
    r.backend = "opencl".to_string();
    assert!(GpuReceiptValidator::new(false).validate(&r).is_ok());
}

#[test]
fn gate2_vulkan_accepted() {
    let mut r = valid_receipt();
    r.backend = "vulkan".to_string();
    assert!(GpuReceiptValidator::new(false).validate(&r).is_ok());
}

#[test]
fn gate2_cpu_accepted() {
    let mut r = valid_receipt();
    r.backend = "cpu".to_string();
    assert!(GpuReceiptValidator::new(false).validate(&r).is_ok());
}

#[test]
fn gate2_unknown_backend_rejected() {
    let mut r = valid_receipt();
    r.backend = "tpu".to_string();
    let errs = GpuReceiptValidator::new(false).validate(&r).unwrap_err();
    assert!(errs.iter().any(|e| e.gate == 2));
}

// ── Gate 3: device_name ─────────────────────────────────────────────

#[test]
fn gate3_empty_device_name_rejected() {
    let mut r = valid_receipt();
    r.device_name = String::new();
    let errs = GpuReceiptValidator::new(false).validate(&r).unwrap_err();
    assert!(errs.iter().any(|e| e.gate == 3));
}

#[test]
fn gate3_nonempty_device_name_accepted() {
    let r = valid_receipt();
    assert!(GpuReceiptValidator::new(false).validate(&r).is_ok());
}

// ── Gate 4: kernel_timings ──────────────────────────────────────────

#[test]
fn gate4_no_kernel_timings_rejected() {
    let mut r = valid_receipt();
    r.kernel_timings.clear();
    let errs = GpuReceiptValidator::new(false).validate(&r).unwrap_err();
    assert!(errs.iter().any(|e| e.gate == 4));
}

#[test]
fn gate4_single_kernel_timing_accepted() {
    let mut r = valid_receipt();
    r.kernel_timings.clear();
    r.kernel_timings.insert("k".to_string(), 1.0);
    assert!(GpuReceiptValidator::new(false).validate(&r).is_ok());
}

// ── Gate 5: total_gpu_time ──────────────────────────────────────────

#[test]
fn gate5_zero_gpu_time_rejected() {
    let mut r = valid_receipt();
    r.total_gpu_time_ms = 0.0;
    let errs = GpuReceiptValidator::new(false).validate(&r).unwrap_err();
    assert!(errs.iter().any(|e| e.gate == 5));
}

#[test]
fn gate5_negative_gpu_time_rejected() {
    let mut r = valid_receipt();
    r.total_gpu_time_ms = -0.001;
    let errs = GpuReceiptValidator::new(false).validate(&r).unwrap_err();
    assert!(errs.iter().any(|e| e.gate == 5));
}

// ── Gate 6: memory peak <= device memory ────────────────────────────

#[test]
fn gate6_memory_over_budget_rejected() {
    let mut r = valid_receipt();
    r.memory_peak_mb = 65536;
    r.device_memory_mb = 40960;
    let errs = GpuReceiptValidator::new(false).validate(&r).unwrap_err();
    assert!(errs.iter().any(|e| e.gate == 6));
}

#[test]
fn gate6_memory_at_limit_accepted() {
    let mut r = valid_receipt();
    r.memory_peak_mb = 40960;
    r.device_memory_mb = 40960;
    assert!(GpuReceiptValidator::new(false).validate(&r).is_ok());
}

// ── Gate 7: driver version ──────────────────────────────────────────

#[test]
fn gate7_invalid_driver_version_rejected() {
    let mut r = valid_receipt();
    r.driver_version = "abc.def".to_string();
    let errs = GpuReceiptValidator::new(false).validate(&r).unwrap_err();
    assert!(errs.iter().any(|e| e.gate == 7));
}

#[test]
fn gate7_empty_driver_version_rejected() {
    let mut r = valid_receipt();
    r.driver_version = String::new();
    let errs = GpuReceiptValidator::new(false).validate(&r).unwrap_err();
    assert!(errs.iter().any(|e| e.gate == 7));
}

#[test]
fn gate7_semver_accepted() {
    let mut r = valid_receipt();
    r.driver_version = "1.2.3".to_string();
    assert!(GpuReceiptValidator::new(false).validate(&r).is_ok());
}

// ── Gate 8: strict mode ─────────────────────────────────────────────

#[test]
#[serial(bitnet_env)]
fn gate8_strict_mode_rejects_mock_device_name() {
    temp_env::with_var("BITNET_STRICT_MODE", Some("1"), || {
        let mut r = valid_receipt();
        r.device_name = "Mock GPU".to_string();
        let errs = GpuReceiptValidator::new(true).validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 8));
    });
}

#[test]
#[serial(bitnet_env)]
fn gate8_strict_mode_rejects_fake_device_name() {
    temp_env::with_var("BITNET_STRICT_MODE", Some("1"), || {
        let mut r = valid_receipt();
        r.device_name = "Fake Device".to_string();
        let errs = GpuReceiptValidator::new(true).validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 8));
    });
}

#[test]
#[serial(bitnet_env)]
fn gate8_strict_mode_accepts_real_device() {
    temp_env::with_var("BITNET_STRICT_MODE", Some("1"), || {
        let r = valid_receipt();
        assert!(GpuReceiptValidator::new(true).validate(&r).is_ok());
    });
}

#[test]
fn gate8_non_strict_allows_mock_device() {
    let mut r = valid_receipt();
    r.device_name = "Mock GPU".to_string();
    assert!(GpuReceiptValidator::new(false).validate(&r).is_ok());
}

// ── JSON serialization ──────────────────────────────────────────────

#[test]
fn json_round_trip_gpu_receipt() {
    let r = valid_receipt();
    let json = serde_json::to_string_pretty(&r).unwrap();
    let de: GpuReceipt = serde_json::from_str(&json).unwrap();
    assert_eq!(r.backend, de.backend);
    assert_eq!(r.device_name, de.device_name);
    assert_eq!(r.compute_path, de.compute_path);
    assert_eq!(r.driver_version, de.driver_version);
    assert_eq!(r.kernel_timings.len(), de.kernel_timings.len());
}

#[test]
fn json_omits_none_opencl_version() {
    let mut r = valid_receipt();
    r.opencl_version = None;
    let json = serde_json::to_string(&r).unwrap();
    assert!(!json.contains("opencl_version"));
}

#[test]
fn json_includes_some_opencl_version() {
    let mut r = valid_receipt();
    r.opencl_version = Some("OpenCL 3.0".to_string());
    let json = serde_json::to_string(&r).unwrap();
    assert!(json.contains("opencl_version"));
    assert!(json.contains("OpenCL 3.0"));
}

// ── Hybrid receipt merging ──────────────────────────────────────────

#[test]
fn merge_preserves_gpu_fields() {
    let r = valid_receipt();
    let hybrid = r.merge_with_cpu_receipt("real", "cpu", 100.0);
    assert_eq!(hybrid.gpu.backend, "cuda");
    assert_eq!(hybrid.gpu.device_name, "NVIDIA A100");
}

#[test]
fn merge_preserves_cpu_fields() {
    let r = valid_receipt();
    let hybrid = r.merge_with_cpu_receipt("real", "cpu", 100.0);
    assert_eq!(hybrid.cpu_compute_path, "real");
    assert_eq!(hybrid.cpu_backend, "cpu");
}

#[test]
fn merge_computes_combined_time() {
    let r = valid_receipt();
    let gpu_time = r.total_gpu_time_ms;
    let cpu_time = 123.456;
    let hybrid = r.merge_with_cpu_receipt("real", "cpu", cpu_time);
    let expected = gpu_time + cpu_time;
    assert!((hybrid.combined_time_ms - expected).abs() < f64::EPSILON);
}

#[test]
fn hybrid_receipt_json_round_trip() {
    let r = valid_receipt();
    let hybrid = r.merge_with_cpu_receipt("real", "cpu", 50.0);
    let json = serde_json::to_string_pretty(&hybrid).unwrap();
    let de: HybridReceipt = serde_json::from_str(&json).unwrap();
    assert_eq!(de.gpu.device_name, "NVIDIA A100");
    assert_eq!(de.cpu_backend, "cpu");
}

// ── Multiple gate failures ──────────────────────────────────────────

#[test]
fn multiple_failures_collected() {
    let mut r = valid_receipt();
    r.compute_path = "mock".to_string();
    r.device_name = String::new();
    r.kernel_timings.clear();
    r.total_gpu_time_ms = 0.0;
    let errs = GpuReceiptValidator::new(false).validate(&r).unwrap_err();
    let gates: Vec<u8> = errs.iter().map(|e| e.gate).collect();
    assert!(gates.contains(&1));
    assert!(gates.contains(&3));
    assert!(gates.contains(&4));
    assert!(gates.contains(&5));
}
