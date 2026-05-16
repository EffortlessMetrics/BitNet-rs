//! Receipt data model and JSON-lines persistence.

use crate::ReceiptError;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::Path;

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
