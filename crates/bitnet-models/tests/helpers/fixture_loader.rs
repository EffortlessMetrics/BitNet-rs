//! Disk-Based Fixture Loading for BitNet-rs QK256 Tests
//!
//! Provides utilities to load GGUF fixtures from `ci/fixtures/qk256/` directory.
//! This complements the in-memory fixture generators in `qk256_fixtures.rs`.
//!
//! ## Usage Patterns
//!
//! ### Pattern 1: Load from disk (CI/CD, deterministic)
//!
//! ```rust
//! use helpers::fixture_loader;
//!
//! let fixture_path = fixture_loader::fixture_path("qk256_4x256.gguf");
//! let result = load_gguf_full(&fixture_path, Device::Cpu, config);
//! ```
//!
//! ### Pattern 2: Generate in-memory (unit tests, fast)
//!
//! ```rust
//! use helpers::qk256_fixtures;
//!
//! let fixture_bytes = qk256_fixtures::generate_qk256_4x256(42);
//! let mut file = NamedTempFile::new().unwrap();
//! file.write_all(&fixture_bytes).unwrap();
//! ```

use std::path::{Path, PathBuf};

/// Get the absolute path to a fixture in `ci/fixtures/qk256/`
///
/// # Arguments
///
/// * `filename` - Fixture filename (e.g., "qk256_4x256.gguf")
///
/// # Returns
///
/// Absolute path to the fixture file
///
/// # Panics
///
/// Panics if the workspace root cannot be determined or fixture doesn't exist
pub fn fixture_path(filename: &str) -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    // CARGO_MANIFEST_DIR points to crates/bitnet-models
    // Navigate from crates/bitnet-models → workspace root
    let workspace_root = manifest_dir
        .parent() // crates
        .and_then(|p| p.parent()) // workspace root
        .expect("Failed to determine workspace root");

    let fixture_path = workspace_root.join("ci/fixtures/qk256").join(filename);

    if !fixture_path.exists() {
        panic!(
            "Fixture not found: {}\nExpected path: {:?}\nWorkspace root: {:?}\nManifest dir: {:?}",
            filename, fixture_path, workspace_root, manifest_dir
        );
    }

    fixture_path
}

/// Load fixture bytes from disk
///
/// # Arguments
///
/// * `filename` - Fixture filename (e.g., "qk256_4x256.gguf")
///
/// # Returns
///
/// Fixture file contents as bytes
///
/// # Panics
///
/// Panics if the file cannot be read
pub fn load_fixture_bytes(filename: &str) -> Vec<u8> {
    let path = fixture_path(filename);
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!("Failed to read fixture {}: {}", filename, e);
    })
}

/// Verify fixture SHA256 checksum
///
/// # Arguments
///
/// * `filename` - Fixture filename
/// * `expected_sha256` - Expected SHA256 hex string (64 chars)
///
/// # Returns
///
/// `true` if checksum matches, `false` otherwise
#[cfg(feature = "fixtures")]
pub fn verify_checksum(filename: &str, expected_sha256: &str) -> bool {
    use sha2::{Digest, Sha256};

    let bytes = load_fixture_bytes(filename);
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let result = hasher.finalize();
    let actual_hex = format!("{:x}", result);

    actual_hex == expected_sha256
}

/// Known fixture checksums (from SHA256SUMS)
pub mod checksums {
    #[allow(dead_code)]
    pub const QK256_4X256: &str =
        "a41cc62c893bcf1d4c03c30ed3da12da03c339847c4d564e9e5794b5d4c6932a";
    #[allow(dead_code)]
    pub const BITNET32_2X64: &str =
        "c1568a0a08e38ef2865ce0816bfd2c617e5589c113114cd731e4c5014b7fbb20";
    #[allow(dead_code)]
    pub const QK256_3X300: &str =
        "6e5a4f21607c0064affbcb86133627478eb34d812b59807a7123ff386c63bd3e";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixture_path_exists() {
        let path = fixture_path("qk256_4x256.gguf");
        assert!(path.exists(), "Fixture path should exist: {:?}", path);
        assert!(path.is_file(), "Fixture path should be a file");
    }

    #[test]
    fn test_load_fixture_bytes() {
        let bytes = load_fixture_bytes("qk256_4x256.gguf");
        assert!(bytes.len() > 8, "Fixture should have GGUF header");
        assert_eq!(&bytes[0..4], b"GGUF", "Should have GGUF magic");
    }

    #[test]
    #[cfg(feature = "fixtures")]
    fn test_verify_checksum_qk256_4x256() {
        assert!(
            verify_checksum("qk256_4x256.gguf", checksums::QK256_4X256),
            "QK256 4x256 checksum should match"
        );
    }

    #[test]
    #[cfg(feature = "fixtures")]
    fn test_verify_checksum_bitnet32_2x64() {
        assert!(
            verify_checksum("bitnet32_2x64.gguf", checksums::BITNET32_2X64),
            "BitNet32 2x64 checksum should match"
        );
    }

    #[test]
    #[cfg(feature = "fixtures")]
    fn test_verify_checksum_qk256_3x300() {
        assert!(
            verify_checksum("qk256_3x300.gguf", checksums::QK256_3X300),
            "QK256 3x300 checksum should match"
        );
    }
}
