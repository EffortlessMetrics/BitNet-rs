//! Wave 14 property tests: GGUF header parsing invariants.
//!
//! Key invariants tested (10 properties):
//! - Valid v2 magic + version produces valid parse with correct counts
//! - Valid v3 header with power-of-two alignment is preserved
//! - Invalid magic always errors
//! - Too-short data always errors
//! - Version outside [2,3] always errors
//! - tensor_count roundtrips correctly
//! - metadata_count roundtrips correctly
//! - check_magic is true iff first 4 bytes are "GGUF"
//! - Arbitrary bytes never cause panic in parse_header
//! - v3 non-power-of-two alignment defaults to 32

use bitnet_gguf::{GGUF_MAGIC, check_magic, parse_header, read_version};
use proptest::prelude::*;

/// Build a valid GGUF v2 header from parts.
fn build_v2_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut d = Vec::with_capacity(24);
    d.extend_from_slice(&GGUF_MAGIC);
    d.extend_from_slice(&2u32.to_le_bytes());
    d.extend_from_slice(&tensor_count.to_le_bytes());
    d.extend_from_slice(&metadata_count.to_le_bytes());
    d
}

/// Build a valid GGUF v3 header with alignment.
fn build_v3_header(tensor_count: u64, metadata_count: u64, alignment: u32) -> Vec<u8> {
    let mut d = Vec::with_capacity(28);
    d.extend_from_slice(&GGUF_MAGIC);
    d.extend_from_slice(&3u32.to_le_bytes());
    d.extend_from_slice(&tensor_count.to_le_bytes());
    d.extend_from_slice(&metadata_count.to_le_bytes());
    d.extend_from_slice(&alignment.to_le_bytes());
    d
}

// ===================================================================
// GGUF header parsing properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Valid v2 header always parses successfully with correct fields.
    #[test]
    fn prop_valid_v2_header_roundtrips(
        tensor_count in 0u64..10_000_000,
        metadata_count in 0u64..10_000_000,
    ) {
        let data = build_v2_header(tensor_count, metadata_count);
        let info = parse_header(&data).expect("valid v2 header must parse");
        prop_assert_eq!(info.version, 2);
        prop_assert_eq!(info.tensor_count, tensor_count);
        prop_assert_eq!(info.metadata_count, metadata_count);
        prop_assert_eq!(info.alignment, 32); // v2 default
    }

    /// Valid v3 header with power-of-two alignment preserves alignment.
    #[test]
    fn prop_valid_v3_header_preserves_alignment(
        tensor_count in 0u64..10_000_000,
        metadata_count in 0u64..10_000_000,
        align_exp in 0u32..=10, // 1, 2, 4, ..., 1024
    ) {
        let alignment = 1u32 << align_exp;
        let data = build_v3_header(tensor_count, metadata_count, alignment);
        let info = parse_header(&data).expect("valid v3 header must parse");
        prop_assert_eq!(info.version, 3);
        prop_assert_eq!(info.tensor_count, tensor_count);
        prop_assert_eq!(info.metadata_count, metadata_count);
        prop_assert_eq!(info.alignment, alignment);
    }

    /// v3 with non-power-of-two alignment defaults to 32.
    #[test]
    fn prop_v3_non_pow2_alignment_defaults(
        tensor_count in 0u64..1_000_000,
        metadata_count in 0u64..1_000_000,
        alignment in 3u32..1000,
    ) {
        // Skip power-of-two values
        if alignment.is_power_of_two() {
            return Ok(());
        }
        let data = build_v3_header(tensor_count, metadata_count, alignment);
        let info = parse_header(&data).expect("v3 with non-pow2 alignment must still parse");
        prop_assert_eq!(info.alignment, 32, "non-pow2 alignment should default to 32");
    }

    /// Invalid magic (first 4 bytes != "GGUF") always produces an error.
    #[test]
    fn prop_invalid_magic_always_errors(
        b0 in 0u8..=255u8,
        b1 in 0u8..=255u8,
        b2 in 0u8..=255u8,
        b3 in 0u8..=255u8,
        rest in prop::collection::vec(0u8..=255, 20..=40),
    ) {
        let mut data = vec![b0, b1, b2, b3];
        data.extend_from_slice(&rest);
        if &data[0..4] != b"GGUF" {
            prop_assert!(parse_header(&data).is_err(), "non-GGUF magic must error");
        }
    }

    /// Data shorter than 24 bytes always errors (even with valid magic).
    #[test]
    fn prop_too_short_always_errors(
        data in prop::collection::vec(0u8..=255, 0..24),
    ) {
        prop_assert!(parse_header(&data).is_err(), "< 24 bytes must error");
    }

    /// Unsupported version always errors.
    #[test]
    fn prop_unsupported_version_errors(
        version in prop::sample::select(vec![0u32, 1, 4, 5, 100, u32::MAX]),
        tensor_count in 0u64..1_000,
        metadata_count in 0u64..1_000,
    ) {
        let mut d = Vec::new();
        d.extend_from_slice(&GGUF_MAGIC);
        d.extend_from_slice(&version.to_le_bytes());
        d.extend_from_slice(&tensor_count.to_le_bytes());
        d.extend_from_slice(&metadata_count.to_le_bytes());
        prop_assert!(parse_header(&d).is_err(), "version {version} should be unsupported");
    }

    /// check_magic returns true iff first 4 bytes are "GGUF".
    #[test]
    fn prop_check_magic_iff_gguf(
        data in prop::collection::vec(0u8..=255, 4..=64),
    ) {
        let expected = data.starts_with(b"GGUF");
        prop_assert_eq!(check_magic(&data), expected);
    }

    /// Arbitrary bytes never cause panic in parse_header.
    #[test]
    fn prop_parse_header_no_panic(
        data in prop::collection::vec(0u8..=255, 0..=128),
    ) {
        let _ = parse_header(&data);
    }

    /// read_version returns Some iff magic is valid and len >= 8.
    #[test]
    fn prop_read_version_consistency(
        data in prop::collection::vec(0u8..=255, 0..=32),
    ) {
        let result = read_version(&data);
        if data.len() < 8 || !check_magic(&data) {
            prop_assert!(result.is_none(), "read_version should be None for invalid input");
        } else {
            prop_assert!(result.is_some(), "read_version should be Some for valid magic + len >= 8");
        }
    }

    /// tensor_count field is always >= 0 (it's u64, so this tests parsing fidelity).
    #[test]
    fn prop_tensor_count_preserves_value(
        tc in 0u64..u64::MAX / 2,
    ) {
        let data = build_v2_header(tc, 0);
        let info = parse_header(&data).unwrap();
        prop_assert_eq!(info.tensor_count, tc);
    }
}
