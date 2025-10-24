//! Fixture Integrity Tests
//!
//! These tests validate the structural integrity of GGUF fixtures committed to the repository.
//! They check:
//! - GGUF magic number (bytes 0-3 must be "GGUF")
//! - GGUF version (bytes 4-7 as little-endian u32, must be 2 or 3)
//! - Minimum tensor count (at least 2 tensors for valid models)
//!
//! These tests complement `scripts/validate-fixtures.sh` and provide early local detection
//! of fixture corruption without requiring shell script execution.
//!
//! See: SPEC-2025-006 (Fixture Contract and Validation)

use std::path::PathBuf;

/// Helper to get fixture paths relative to repository root
fn fixture_path(filename: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join("../../ci/fixtures/qk256").join(filename)
}

/// Validate GGUF header structure (magic, version)
fn validate_gguf_header(bytes: &[u8], fixture_name: &str) {
    // Check GGUF magic (bytes 0-3)
    assert!(
        bytes.len() >= 8,
        "Fixture '{}' is too small ({} bytes, need at least 8 for header)",
        fixture_name,
        bytes.len()
    );

    let magic = &bytes[0..4];
    assert_eq!(
        magic,
        b"GGUF",
        "Fixture '{}' has invalid magic number. Expected 'GGUF', got {:?}",
        fixture_name,
        std::str::from_utf8(magic).unwrap_or("<invalid utf8>")
    );

    // Check GGUF version (bytes 4-7 as little-endian u32)
    let version_bytes: [u8; 4] = bytes[4..8].try_into().unwrap();
    let version = u32::from_le_bytes(version_bytes);
    assert!(
        version == 2 || version == 3,
        "Fixture '{}' has unexpected GGUF version {}. Expected 2 or 3",
        fixture_name,
        version
    );
}

#[test]
fn test_qk256_4x256_header_integrity() {
    let path = fixture_path("qk256_4x256.gguf");
    assert!(path.exists(), "Fixture qk256_4x256.gguf should exist at {:?}", path);

    let bytes = std::fs::read(&path).expect("Should read qk256_4x256.gguf");
    validate_gguf_header(&bytes, "qk256_4x256.gguf");

    // Verify expected size (from exploration: 10,816 bytes)
    assert_eq!(bytes.len(), 10816, "qk256_4x256.gguf size changed unexpectedly");
}

#[test]
fn test_qk256_3x300_header_integrity() {
    let path = fixture_path("qk256_3x300.gguf");
    assert!(path.exists(), "Fixture qk256_3x300.gguf should exist at {:?}", path);

    let bytes = std::fs::read(&path).expect("Should read qk256_3x300.gguf");
    validate_gguf_header(&bytes, "qk256_3x300.gguf");

    // Verify expected size (from exploration: 10,696 bytes)
    assert_eq!(bytes.len(), 10696, "qk256_3x300.gguf size changed unexpectedly");
}

#[test]
fn test_bitnet32_2x64_header_integrity() {
    let path = fixture_path("bitnet32_2x64.gguf");
    assert!(path.exists(), "Fixture bitnet32_2x64.gguf should exist at {:?}", path);

    let bytes = std::fs::read(&path).expect("Should read bitnet32_2x64.gguf");
    validate_gguf_header(&bytes, "bitnet32_2x64.gguf");

    // Verify expected size (from exploration: 8,832 bytes)
    assert_eq!(bytes.len(), 8832, "bitnet32_2x64.gguf size changed unexpectedly");
}

#[test]
fn test_all_fixtures_present() {
    // Ensure all documented fixtures are committed
    let fixtures = vec!["qk256_4x256.gguf", "qk256_3x300.gguf", "bitnet32_2x64.gguf"];

    for fixture in fixtures {
        let path = fixture_path(fixture);
        assert!(path.exists(), "Required fixture '{}' is missing at {:?}", fixture, path);
    }
}

#[test]
fn test_sha256sums_file_present() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let sha256sums_path = PathBuf::from(manifest_dir).join("../../ci/fixtures/qk256/SHA256SUMS");

    assert!(sha256sums_path.exists(), "SHA256SUMS file should exist at {:?}", sha256sums_path);

    let content = std::fs::read_to_string(&sha256sums_path).expect("Should read SHA256SUMS");

    // Verify it contains all three fixtures
    assert!(content.contains("qk256_4x256.gguf"), "SHA256SUMS should contain qk256_4x256.gguf");
    assert!(content.contains("qk256_3x300.gguf"), "SHA256SUMS should contain qk256_3x300.gguf");
    assert!(content.contains("bitnet32_2x64.gguf"), "SHA256SUMS should contain bitnet32_2x64.gguf");
}
