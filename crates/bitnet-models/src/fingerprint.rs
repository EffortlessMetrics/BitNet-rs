//! GGUF file fingerprinting for correction policy matching
//!
//! This module provides SHA256 fingerprinting of GGUF files to identify
//! specific model versions for policy-driven corrections.

use bitnet_common::{BitNetError, Result};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

/// Compute SHA256 fingerprint of a GGUF file
///
/// Returns a string in the format "sha256-<hex_digest>"
pub fn compute_gguf_fingerprint(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    format!("sha256-{:x}", result)
}

/// Compute SHA256 fingerprint from a file path
pub fn compute_file_fingerprint(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).map_err(|e| {
        BitNetError::Validation(format!("Failed to open file for fingerprinting: {}", e))
    })?;

    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024]; // 1MB buffer

    loop {
        let n = file.read(&mut buffer).map_err(|e| {
            BitNetError::Validation(format!("Failed to read file for fingerprinting: {}", e))
        })?;

        if n == 0 {
            break;
        }

        hasher.update(&buffer[..n]);
    }

    let result = hasher.finalize();
    Ok(format!("sha256-{:x}", result))
}

/// Extract fingerprint from metadata (if already computed)
pub fn format_fingerprint(hash_bytes: &[u8]) -> String {
    format!("sha256-{}", hex::encode(hash_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_compute_fingerprint_empty() {
        let data = b"";
        let fp = compute_gguf_fingerprint(data);
        // SHA256 of empty string
        assert_eq!(fp, "sha256-e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn test_compute_fingerprint_known() {
        let data = b"hello world";
        let fp = compute_gguf_fingerprint(data);
        // SHA256 of "hello world"
        assert_eq!(fp, "sha256-b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    }

    #[test]
    fn test_compute_fingerprint_stability() {
        let data = b"test data for fingerprinting";
        let fp1 = compute_gguf_fingerprint(data);
        let fp2 = compute_gguf_fingerprint(data);
        assert_eq!(fp1, fp2, "Fingerprint should be stable");
    }

    #[test]
    fn test_compute_fingerprint_different_data() {
        let data1 = b"data1";
        let data2 = b"data2";
        let fp1 = compute_gguf_fingerprint(data1);
        let fp2 = compute_gguf_fingerprint(data2);
        assert_ne!(fp1, fp2, "Different data should have different fingerprints");
    }

    #[test]
    fn test_compute_file_fingerprint() {
        let mut temp = tempfile::NamedTempFile::new().unwrap();
        temp.write_all(b"test file content").unwrap();
        temp.flush().unwrap();

        let fp = compute_file_fingerprint(temp.path()).unwrap();
        assert!(fp.starts_with("sha256-"));
        assert_eq!(fp.len(), 71); // "sha256-" + 64 hex chars
    }

    #[test]
    fn test_compute_file_fingerprint_large() {
        let mut temp = tempfile::NamedTempFile::new().unwrap();
        // Write 10MB of data
        let chunk = vec![0xAB; 1024 * 1024];
        for _ in 0..10 {
            temp.write_all(&chunk).unwrap();
        }
        temp.flush().unwrap();

        let fp = compute_file_fingerprint(temp.path()).unwrap();
        assert!(fp.starts_with("sha256-"));
        assert_eq!(fp.len(), 71);
    }

    #[test]
    fn test_format_fingerprint() {
        let hash_bytes = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef];
        let fp = format_fingerprint(&hash_bytes);
        assert_eq!(fp, "sha256-0123456789abcdef");
    }
}
