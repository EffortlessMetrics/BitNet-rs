//! Disk-Based Fixture Loading Tests for QK256
//!
//! Validates that fixtures can be loaded from `ci/fixtures/qk256/` and
//! that checksums match expected values.

#![cfg(feature = "fixtures")]

mod helpers;

use helpers::fixture_loader;

#[test]
fn test_load_qk256_4x256_from_disk() {
    let path = fixture_loader::fixture_path("qk256_4x256.gguf");
    assert!(path.exists(), "QK256 4x256 fixture should exist");

    let bytes = fixture_loader::load_fixture_bytes("qk256_4x256.gguf");
    assert_eq!(&bytes[0..4], b"GGUF", "Should have GGUF magic");
    assert_eq!(bytes.len(), 10816, "Should match expected fixture size");
}

#[test]
fn test_load_bitnet32_2x64_from_disk() {
    let path = fixture_loader::fixture_path("bitnet32_2x64.gguf");
    assert!(path.exists(), "BitNet32 2x64 fixture should exist");

    let bytes = fixture_loader::load_fixture_bytes("bitnet32_2x64.gguf");
    assert_eq!(&bytes[0..4], b"GGUF", "Should have GGUF magic");
    assert_eq!(bytes.len(), 8832, "Should match expected fixture size");
}

#[test]
fn test_load_qk256_3x300_from_disk() {
    let path = fixture_loader::fixture_path("qk256_3x300.gguf");
    assert!(path.exists(), "QK256 3x300 fixture should exist");

    let bytes = fixture_loader::load_fixture_bytes("qk256_3x300.gguf");
    assert_eq!(&bytes[0..4], b"GGUF", "Should have GGUF magic");
    assert_eq!(bytes.len(), 10696, "Should match expected fixture size");
}

#[test]
fn test_verify_qk256_4x256_checksum() {
    assert!(
        fixture_loader::verify_checksum("qk256_4x256.gguf", fixture_loader::checksums::QK256_4X256),
        "QK256 4x256 checksum should match"
    );
}

#[test]
fn test_verify_bitnet32_2x64_checksum() {
    assert!(
        fixture_loader::verify_checksum(
            "bitnet32_2x64.gguf",
            fixture_loader::checksums::BITNET32_2X64
        ),
        "BitNet32 2x64 checksum should match"
    );
}

#[test]
fn test_verify_qk256_3x300_checksum() {
    assert!(
        fixture_loader::verify_checksum("qk256_3x300.gguf", fixture_loader::checksums::QK256_3X300),
        "QK256 3x300 checksum should match"
    );
}

#[test]
fn test_all_fixtures_load_successfully() {
    // Ensure all fixtures can be loaded without panics
    let fixtures = ["qk256_4x256.gguf", "bitnet32_2x64.gguf", "qk256_3x300.gguf"];

    for fixture in &fixtures {
        let bytes = fixture_loader::load_fixture_bytes(fixture);
        assert!(bytes.len() > 8, "Fixture {} should have valid GGUF header", fixture);
        assert_eq!(&bytes[0..4], b"GGUF", "Fixture {} should have GGUF magic", fixture);
    }
}
