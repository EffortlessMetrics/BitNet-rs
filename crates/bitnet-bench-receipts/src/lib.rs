//! Benchmark receipts for tracking kernel performance over time.
//!
//! This crate provides filesystem persistence over the core receipt contracts
//! from `bitnet-bench-receipts-core`.

pub use bitnet_bench_receipts_core::{BenchReceipt, ReceiptError};
use std::io::{BufRead, Write};
use std::path::Path;

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
    fn test_store_append_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.jsonl");
        assert!(!path.exists());

        ReceiptStore::append(&path, &sample_receipt("k", 1)).unwrap();
        assert!(path.exists());
    }
}
