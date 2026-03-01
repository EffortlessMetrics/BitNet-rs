//! Benchmark receipts for tracking kernel performance over time.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::Path;

/// Errors from receipt I/O and serialization.
#[derive(Debug, thiserror::Error)]
pub enum ReceiptError {
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// A single benchmark measurement for a wgpu compute kernel dispatch.
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

/// Append-only JSON-lines store for benchmark receipts.
pub struct ReceiptStore;

impl ReceiptStore {
    /// Load all receipts from a JSON-lines file.
    pub fn load(path: &Path) -> Result<Vec<BenchReceipt>, ReceiptError> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut receipts = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            receipts.push(BenchReceipt::from_json(trimmed)?);
        }
        Ok(receipts)
    }

    /// Append a single receipt to a JSON-lines file, creating it if absent.
    pub fn append(path: &Path, receipt: &BenchReceipt) -> Result<(), ReceiptError> {
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(file, "{}", receipt.to_json())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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
    fn test_store_append_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("receipts.jsonl");

        let r1 = sample_receipt("k1", 100);
        let r2 = sample_receipt("k2", 200);
        ReceiptStore::append(&path, &r1).unwrap();
        ReceiptStore::append(&path, &r2).unwrap();

        let loaded = ReceiptStore::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0], r1);
        assert_eq!(loaded[1], r2);
    }

    #[test]
    fn test_store_load_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.jsonl");
        std::fs::File::create(&path).unwrap();

        let loaded = ReceiptStore::load(&path).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_store_load_nonexistent_file() {
        let result = ReceiptStore::load(Path::new("/nonexistent/path.jsonl"));
        assert!(result.is_err());
    }

    #[test]
    fn test_store_skips_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blanks.jsonl");
        let r = sample_receipt("k1", 100);
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{}", r.to_json()).unwrap();
        writeln!(f).unwrap();
        writeln!(f, "{}", r.to_json()).unwrap();
        drop(f);

        let loaded = ReceiptStore::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_throughput_precision() {
        let r = BenchReceipt::new("k", [1, 1, 1], [1, 1, 1], 1, 3.141_592_653_589_793, 0, "", "");
        let r2 = BenchReceipt::from_json(&r.to_json()).unwrap();
        assert!((r2.throughput_gflops - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_store_append_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.jsonl");
        assert!(!path.exists());

        ReceiptStore::append(&path, &sample_receipt("k", 1)).unwrap();
        assert!(path.exists());
    }
}
