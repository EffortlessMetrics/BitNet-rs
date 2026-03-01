//! GPU receipt validation with 8 validation gates.
//!
//! Extends the existing receipt schema (`ci/inference.json`) with GPU-specific
//! fields: backend, device info, kernel timings, and memory statistics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Allowed GPU backend identifiers.
const ALLOWED_BACKENDS: &[&str] = &["cuda", "opencl", "vulkan", "cpu"];

/// GPU inference receipt extending the existing receipt schema.
///
/// Captures backend, device, kernel timing, and memory statistics for
/// GPU inference runs. JSON-compatible with `ci/inference.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuReceipt {
    /// GPU backend: "cuda", "opencl", "vulkan", or "cpu"
    pub backend: String,

    /// Human-readable device name (e.g. "NVIDIA A100", "Intel Arc A770")
    pub device_name: String,

    /// Total device memory in MiB
    pub device_memory_mb: u64,

    /// Per-kernel average execution time in milliseconds
    pub kernel_timings: HashMap<String, f64>,

    /// Total GPU execution time in milliseconds
    pub total_gpu_time_ms: f64,

    /// Peak GPU memory usage in MiB
    pub memory_peak_mb: u64,

    /// Compute path: must be "real" — never "mock"
    pub compute_path: String,

    /// GPU driver version (semver-like, e.g. "535.129.03")
    pub driver_version: String,

    /// OpenCL version string (present only for OpenCL backend)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opencl_version: Option<String>,
}

/// Merged CPU+GPU receipt for hybrid inference runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridReceipt {
    /// GPU portion of the receipt
    pub gpu: GpuReceipt,

    /// CPU compute path (from existing `ci/inference.json`)
    pub cpu_compute_path: String,

    /// CPU backend identifier (from existing receipt)
    pub cpu_backend: String,

    /// Combined total time (CPU + GPU) in milliseconds
    pub combined_time_ms: f64,
}

/// Validation error from one of the 8 gates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateError {
    pub gate: u8,
    pub message: String,
}

impl std::fmt::Display for GateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gate {}: {}", self.gate, self.message)
    }
}

impl std::error::Error for GateError {}

/// 8-gate validator for [`GpuReceipt`].
pub struct GpuReceiptValidator {
    strict_mode: bool,
}

impl GpuReceiptValidator {
    /// Create a validator. When `strict_mode` is true, gate 8 rejects any
    /// mock data (mirroring `BITNET_STRICT_MODE=1` behaviour).
    pub fn new(strict_mode: bool) -> Self {
        Self { strict_mode }
    }

    /// Run all 8 gates, collecting every failure.
    pub fn validate(&self, receipt: &GpuReceipt) -> Result<(), Vec<GateError>> {
        let mut errors = Vec::new();

        // Gate 1: compute_path must be "real"
        if receipt.compute_path != "real" {
            errors.push(GateError {
                gate: 1,
                message: format!("compute_path must be \"real\", got \"{}\"", receipt.compute_path),
            });
        }

        // Gate 2: backend must be in allowed list
        if !ALLOWED_BACKENDS.contains(&receipt.backend.as_str()) {
            errors.push(GateError {
                gate: 2,
                message: format!(
                    "backend \"{}\" not in allowed list {:?}",
                    receipt.backend, ALLOWED_BACKENDS
                ),
            });
        }

        // Gate 3: device_name must be non-empty
        if receipt.device_name.is_empty() {
            errors
                .push(GateError { gate: 3, message: "device_name must not be empty".to_string() });
        }

        // Gate 4: kernel_timings must have at least 1 entry
        if receipt.kernel_timings.is_empty() {
            errors.push(GateError {
                gate: 4,
                message: "kernel_timings must have at least 1 entry".to_string(),
            });
        }

        // Gate 5: total_gpu_time > 0
        if receipt.total_gpu_time_ms <= 0.0 {
            errors.push(GateError {
                gate: 5,
                message: format!(
                    "total_gpu_time_ms must be > 0, got {}",
                    receipt.total_gpu_time_ms
                ),
            });
        }

        // Gate 6: memory_peak <= device_memory
        if receipt.memory_peak_mb > receipt.device_memory_mb {
            errors.push(GateError {
                gate: 6,
                message: format!(
                    "memory_peak_mb ({}) exceeds device_memory_mb ({})",
                    receipt.memory_peak_mb, receipt.device_memory_mb
                ),
            });
        }

        // Gate 7: driver_version is valid semver-like (digits and dots)
        if !is_valid_driver_version(&receipt.driver_version) {
            errors.push(GateError {
                gate: 7,
                message: format!(
                    "driver_version \"{}\" is not valid semver-like",
                    receipt.driver_version
                ),
            });
        }

        // Gate 8: strict mode — reject anything that looks mock
        if self.strict_mode {
            if receipt.compute_path == "mock" {
                errors.push(GateError {
                    gate: 8,
                    message: "strict mode: mock compute_path rejected".to_string(),
                });
            }
            if receipt.device_name.to_lowercase().contains("mock")
                || receipt.device_name.to_lowercase().contains("fake")
            {
                errors.push(GateError {
                    gate: 8,
                    message: format!(
                        "strict mode: suspicious device_name \"{}\"",
                        receipt.device_name
                    ),
                });
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

/// Check that `version` looks like a semver-ish string: non-empty, starts
/// with a digit, contains only digits and dots, and has at least one dot.
fn is_valid_driver_version(version: &str) -> bool {
    if version.is_empty() {
        return false;
    }
    let first = version.as_bytes()[0];
    if !first.is_ascii_digit() {
        return false;
    }
    if !version.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
        return false;
    }
    version.contains('.')
}

impl GpuReceipt {
    /// Merge this GPU receipt with CPU receipt fields to produce a
    /// [`HybridReceipt`].
    pub fn merge_with_cpu_receipt(
        self,
        cpu_compute_path: &str,
        cpu_backend: &str,
        cpu_time_ms: f64,
    ) -> HybridReceipt {
        let combined_time_ms = self.total_gpu_time_ms + cpu_time_ms;
        HybridReceipt {
            gpu: self,
            cpu_compute_path: cpu_compute_path.to_string(),
            cpu_backend: cpu_backend.to_string(),
            combined_time_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a valid receipt that passes all 8 gates.
    fn valid_receipt() -> GpuReceipt {
        let mut timings = HashMap::new();
        timings.insert("gemm_kernel".to_string(), 1.5);
        GpuReceipt {
            backend: "opencl".to_string(),
            device_name: "Intel Arc A770".to_string(),
            device_memory_mb: 16384,
            kernel_timings: timings,
            total_gpu_time_ms: 42.0,
            memory_peak_mb: 4096,
            compute_path: "real".to_string(),
            driver_version: "23.17.26241".to_string(),
            opencl_version: Some("OpenCL 3.0".to_string()),
        }
    }

    fn validator() -> GpuReceiptValidator {
        GpuReceiptValidator::new(false)
    }

    fn strict_validator() -> GpuReceiptValidator {
        GpuReceiptValidator::new(true)
    }

    #[test]
    fn valid_receipt_passes_all_gates() {
        assert!(validator().validate(&valid_receipt()).is_ok());
    }

    #[test]
    fn valid_receipt_passes_strict_mode() {
        assert!(strict_validator().validate(&valid_receipt()).is_ok());
    }

    #[test]
    fn gate1_mock_compute_path_rejected() {
        let mut r = valid_receipt();
        r.compute_path = "mock".to_string();
        let errs = validator().validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 1));
    }

    #[test]
    fn gate2_invalid_backend_rejected() {
        let mut r = valid_receipt();
        r.backend = "metal".to_string();
        let errs = validator().validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 2));
    }

    #[test]
    fn gate2_all_allowed_backends() {
        for backend in &["cuda", "opencl", "vulkan", "cpu"] {
            let mut r = valid_receipt();
            r.backend = backend.to_string();
            assert!(validator().validate(&r).is_ok(), "backend '{}' should be allowed", backend);
        }
    }

    #[test]
    fn gate3_empty_device_name_rejected() {
        let mut r = valid_receipt();
        r.device_name = String::new();
        let errs = validator().validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 3));
    }

    #[test]
    fn gate4_no_kernel_timings_rejected() {
        let mut r = valid_receipt();
        r.kernel_timings.clear();
        let errs = validator().validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 4));
    }

    #[test]
    fn gate5_zero_gpu_time_rejected() {
        let mut r = valid_receipt();
        r.total_gpu_time_ms = 0.0;
        let errs = validator().validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 5));
    }

    #[test]
    fn gate5_negative_gpu_time_rejected() {
        let mut r = valid_receipt();
        r.total_gpu_time_ms = -1.0;
        let errs = validator().validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 5));
    }

    #[test]
    fn gate6_memory_over_budget_rejected() {
        let mut r = valid_receipt();
        r.memory_peak_mb = 32768;
        r.device_memory_mb = 16384;
        let errs = validator().validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 6));
    }

    #[test]
    fn gate6_memory_at_limit_passes() {
        let mut r = valid_receipt();
        r.memory_peak_mb = 16384;
        r.device_memory_mb = 16384;
        assert!(validator().validate(&r).is_ok());
    }

    #[test]
    fn gate7_invalid_driver_version_rejected() {
        let mut r = valid_receipt();
        r.driver_version = "not-a-version".to_string();
        let errs = validator().validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 7));
    }

    #[test]
    fn gate7_empty_driver_version_rejected() {
        let mut r = valid_receipt();
        r.driver_version = String::new();
        let errs = validator().validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 7));
    }

    #[test]
    fn gate7_valid_driver_versions() {
        for v in &["535.129.03", "23.17.26241", "1.0", "31.0.101.5768"] {
            let mut r = valid_receipt();
            r.driver_version = v.to_string();
            assert!(validator().validate(&r).is_ok(), "driver version '{}' should be valid", v);
        }
    }

    #[test]
    fn gate7_no_dot_rejected() {
        let mut r = valid_receipt();
        r.driver_version = "12345".to_string();
        let errs = validator().validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 7));
    }

    #[test]
    fn gate8_strict_mode_rejects_mock_device() {
        let mut r = valid_receipt();
        r.device_name = "Mock GPU Device".to_string();
        let errs = strict_validator().validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 8));
    }

    #[test]
    fn gate8_strict_mode_rejects_fake_device() {
        let mut r = valid_receipt();
        r.device_name = "FAKE_GPU".to_string();
        let errs = strict_validator().validate(&r).unwrap_err();
        assert!(errs.iter().any(|e| e.gate == 8));
    }

    #[test]
    fn gate8_non_strict_allows_mock_device() {
        let mut r = valid_receipt();
        r.device_name = "Mock GPU Device".to_string();
        assert!(validator().validate(&r).is_ok());
    }

    #[test]
    fn multiple_failures_reported() {
        let mut r = valid_receipt();
        r.compute_path = "mock".to_string();
        r.device_name = String::new();
        r.kernel_timings.clear();
        let errs = validator().validate(&r).unwrap_err();
        assert!(errs.len() >= 3);
    }

    #[test]
    fn json_round_trip() {
        let r = valid_receipt();
        let json = serde_json::to_string_pretty(&r).unwrap();
        let deserialized: GpuReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(r.backend, deserialized.backend);
        assert_eq!(r.device_name, deserialized.device_name);
        assert_eq!(r.compute_path, deserialized.compute_path);
        assert_eq!(r.kernel_timings.len(), deserialized.kernel_timings.len());
    }

    #[test]
    fn json_opencl_version_absent_when_none() {
        let mut r = valid_receipt();
        r.opencl_version = None;
        let json = serde_json::to_string(&r).unwrap();
        assert!(!json.contains("opencl_version"));
    }

    #[test]
    fn json_opencl_version_present_when_some() {
        let r = valid_receipt();
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("opencl_version"));
    }

    #[test]
    fn merge_with_cpu_receipt_preserves_both() {
        let r = valid_receipt();
        let gpu_time = r.total_gpu_time_ms;
        let hybrid = r.merge_with_cpu_receipt("real", "cpu", 100.0);
        assert_eq!(hybrid.gpu.backend, "opencl");
        assert_eq!(hybrid.cpu_compute_path, "real");
        assert_eq!(hybrid.cpu_backend, "cpu");
        assert!((hybrid.combined_time_ms - (gpu_time + 100.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn hybrid_receipt_json_round_trip() {
        let r = valid_receipt();
        let hybrid = r.merge_with_cpu_receipt("real", "cpu", 50.0);
        let json = serde_json::to_string_pretty(&hybrid).unwrap();
        let deserialized: HybridReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.gpu.device_name, "Intel Arc A770");
        assert_eq!(deserialized.cpu_backend, "cpu");
    }
}
