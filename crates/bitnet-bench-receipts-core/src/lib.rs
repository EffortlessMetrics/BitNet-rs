//! Core benchmark receipt contracts.

use serde::{Deserialize, Serialize};

/// Errors from receipt serialization and deserialization.
#[derive(Debug, thiserror::Error)]
pub enum ReceiptError {
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// A single benchmark measurement for a compute-kernel dispatch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchReceipt {
    pub kernel_name: String,
    pub workgroup_size: [u32; 3],
    pub dispatch_size: [u32; 3],
    pub elapsed_us: u64,
    pub throughput_gflops: f64,
    pub timestamp: u64,
    pub device_name: String,
    pub backend: String,
}

impl BenchReceipt {
    /// Create a new benchmark receipt.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel_name: impl Into<String>,
        workgroup_size: [u32; 3],
        dispatch_size: [u32; 3],
        elapsed_us: u64,
        throughput_gflops: f64,
        timestamp: u64,
        device_name: impl Into<String>,
        backend: impl Into<String>,
    ) -> Self {
        Self {
            kernel_name: kernel_name.into(),
            workgroup_size,
            dispatch_size,
            elapsed_us,
            throughput_gflops,
            timestamp,
            device_name: device_name.into(),
            backend: backend.into(),
        }
    }

    /// Serialize to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("BenchReceipt is always serializable")
    }

    /// Deserialize from a JSON string.
    pub fn from_json(s: &str) -> Result<Self, ReceiptError> {
        Ok(serde_json::from_str(s)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_receipt(name: &str, elapsed_us: u64) -> BenchReceipt {
        BenchReceipt::new(
            name,
            [256, 1, 1],
            [1024, 1, 1],
            elapsed_us,
            42.0,
            1_700_000_000,
            "Test GPU",
            "vulkan",
        )
    }

    #[test]
    fn test_new_sets_all_fields() {
        let r = sample_receipt("matmul", 500);
        assert_eq!(r.kernel_name, "matmul");
        assert_eq!(r.workgroup_size, [256, 1, 1]);
        assert_eq!(r.dispatch_size, [1024, 1, 1]);
        assert_eq!(r.elapsed_us, 500);
        assert_eq!(r.device_name, "Test GPU");
        assert_eq!(r.backend, "vulkan");
    }

    #[test]
    fn test_to_json_produces_valid_json() {
        let r = sample_receipt("softmax", 100);
        let json = r.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["kernel_name"], "softmax");
    }

    #[test]
    fn test_from_json_roundtrip() {
        let r = sample_receipt("rms_norm", 250);
        let json = r.to_json();
        let r2 = BenchReceipt::from_json(&json).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_from_json_invalid_returns_error() {
        let result = BenchReceipt::from_json("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_json_missing_field() {
        let result = BenchReceipt::from_json(r#"{"kernel_name":"x"}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization_preserves_workgroup_array() {
        let r = sample_receipt("conv", 300);
        let json = r.to_json();
        assert!(json.contains("[256,1,1]"));
    }

    #[test]
    fn test_throughput_precision() {
        let r = BenchReceipt::new("k", [1, 1, 1], [1, 1, 1], 1, 3.141_592_653_589_793, 0, "", "");
        let r2 = BenchReceipt::from_json(&r.to_json()).unwrap();
        assert!((r2.throughput_gflops - std::f64::consts::PI).abs() < 1e-10);
    }
}
