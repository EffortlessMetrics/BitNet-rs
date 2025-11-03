//! QK256 Fixture Validation Tests
//!
//! This test suite validates the fixture generators in the helpers module.

mod helpers;

use helpers::{generate_bitnet32_2x64, generate_qk256_3x300, generate_qk256_4x256};

#[test]
fn test_qk256_4x256_generation() {
    let fixture = generate_qk256_4x256(42);

    // Verify GGUF magic
    assert_eq!(&fixture[0..4], b"GGUF", "Missing GGUF magic");

    // Verify version is 3
    let version = u32::from_le_bytes([fixture[4], fixture[5], fixture[6], fixture[7]]);
    assert_eq!(version, 3, "Expected GGUF v3");

    // Fixture should contain tensor data (4 rows × 64 bytes = 256 bytes)
    assert!(fixture.len() >= 256, "Fixture should contain at least 256 bytes of tensor data");
}

#[test]
fn test_bitnet32_2x64_generation() {
    let fixture = generate_bitnet32_2x64(43);

    assert_eq!(&fixture[0..4], b"GGUF");

    let version = u32::from_le_bytes([fixture[4], fixture[5], fixture[6], fixture[7]]);
    assert_eq!(version, 3);

    // Fixture should contain tensor data (2 rows × 20 bytes = 40 bytes)
    assert!(fixture.len() >= 40, "Fixture should contain at least 40 bytes of tensor data");
}

#[test]
fn test_qk256_3x300_generation() {
    let fixture = generate_qk256_3x300(44);

    assert_eq!(&fixture[0..4], b"GGUF");

    let version = u32::from_le_bytes([fixture[4], fixture[5], fixture[6], fixture[7]]);
    assert_eq!(version, 3);

    // Fixture should contain tensor data (3 rows × 128 bytes = 384 bytes)
    assert!(fixture.len() >= 384, "Fixture should contain at least 384 bytes of tensor data");
}

#[test]
fn test_deterministic_fixtures() {
    // Same seed should produce identical fixtures
    let fixture1 = generate_qk256_4x256(42);
    let fixture2 = generate_qk256_4x256(42);
    assert_eq!(fixture1, fixture2, "Fixtures with same seed should be identical");

    // Different seeds should produce different fixtures
    let fixture3 = generate_qk256_4x256(99);
    assert_ne!(fixture1, fixture3, "Fixtures with different seeds should differ");
}

#[test]
fn test_all_fixtures_have_valid_structure() {
    let fixtures = [generate_qk256_4x256(42), generate_bitnet32_2x64(43), generate_qk256_3x300(44)];

    for (i, fixture) in fixtures.iter().enumerate() {
        // All should have GGUF magic
        assert_eq!(&fixture[0..4], b"GGUF", "Fixture {} missing GGUF magic", i);

        // All should be version 3
        let version = u32::from_le_bytes([fixture[4], fixture[5], fixture[6], fixture[7]]);
        assert_eq!(version, 3, "Fixture {} not version 3", i);

        // All should have reasonable size (header + metadata + tensor data)
        assert!(fixture.len() > 100, "Fixture {} too small", i);
    }
}
