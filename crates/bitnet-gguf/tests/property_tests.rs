//! Property tests for `bitnet-gguf` header parser.
//!
//! Uses proptest to verify parser invariants across the full range of
//! syntactically-valid and invalid input shapes.

use bitnet_gguf::{GGUF_MAGIC, GgufValueType, check_magic, parse_header};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helper: build a minimal valid GGUF header byte slice
// ---------------------------------------------------------------------------

/// Build a well-formed GGUF header: magic + 4-byte version + 8-byte tensor_count +
/// 8-byte metadata_kv_count = 24 bytes.  Version is clamped to [2, 3].
fn make_valid_header(version: u32) -> Vec<u8> {
    let version = version.clamp(2, 3);
    let mut data = Vec::with_capacity(24);
    data.extend_from_slice(&GGUF_MAGIC); // 4 bytes: magic
    data.extend_from_slice(&version.to_le_bytes()); // 4 bytes: version
    data.extend_from_slice(&0u64.to_le_bytes()); // 8 bytes: tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // 8 bytes: metadata_kv_count
    data
}

// ---------------------------------------------------------------------------
// Properties: check_magic
// ---------------------------------------------------------------------------

proptest! {
    /// `check_magic` returns true if and only if the first 4 bytes are exactly
    /// the GGUF magic, regardless of what follows.
    #[test]
    fn prop_check_magic_iff_starts_with_magic(
        rest in prop::collection::vec(any::<u8>(), 0..64)
    ) {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend(rest.iter());
        prop_assert!(check_magic(&data), "data starting with GGUF magic must pass check_magic");
    }

    /// Data shorter than 4 bytes never passes `check_magic`.
    #[test]
    fn prop_check_magic_rejects_short_data(
        data in prop::collection::vec(any::<u8>(), 0..4)
    ) {
        prop_assert!(!check_magic(&data), "data shorter than 4 bytes must fail check_magic");
    }

    /// If the first byte differs from 'G', `check_magic` must return false.
    #[test]
    fn prop_check_magic_rejects_wrong_first_byte(
        first_byte in any::<u8>().prop_filter("not G", |&b| b != b'G'),
        rest in prop::collection::vec(any::<u8>(), 3..32)
    ) {
        let mut data = vec![first_byte];
        data.extend(rest);
        prop_assert!(!check_magic(&data));
    }
}

// ---------------------------------------------------------------------------
// Properties: parse_header
// ---------------------------------------------------------------------------

proptest! {
    /// A valid GGUF header (magic + version in [2,3] + zeros) must always parse
    /// successfully.
    #[test]
    fn prop_parse_header_succeeds_on_valid_data(version in 2u32..=3u32) {
        let data = make_valid_header(version);
        prop_assert!(parse_header(&data).is_ok(), "valid GGUF header must parse; v={version}");
    }

    /// Data shorter than 24 bytes must always fail `parse_header`.
    #[test]
    fn prop_parse_header_rejects_truncated_data(
        data in prop::collection::vec(any::<u8>(), 0..24)
    ) {
        prop_assert!(parse_header(&data).is_err(), "truncated data must fail parse_header");
    }

    /// `parse_header` on random bytes that don't start with the GGUF magic must
    /// fail.
    #[test]
    fn prop_parse_header_rejects_wrong_magic(
        bad_first in any::<u8>().prop_filter("not G", |&b| b != b'G'),
        tail in prop::collection::vec(any::<u8>(), 23..64)
    ) {
        let mut data = vec![bad_first];
        data.extend(tail);
        prop_assert!(parse_header(&data).is_err(), "wrong magic must fail parse_header");
    }
}

// ---------------------------------------------------------------------------
// Properties: GgufValueType round-trip
// ---------------------------------------------------------------------------

proptest! {
    /// `GgufValueType::from_u32` returns `None` for values outside [0, 12].
    #[test]
    fn prop_value_type_from_u32_rejects_out_of_range(
        v in 13u32..u32::MAX
    ) {
        prop_assert_eq!(
            GgufValueType::from_u32(v),
            None,
            "value type discriminant {} is out of range and must return None",
            v
        );
    }

    /// `GgufValueType::from_u32` succeeds for all discriminants in [0, 12].
    #[test]
    fn prop_value_type_from_u32_succeeds_in_range(v in 0u32..=12u32) {
        prop_assert!(
            GgufValueType::from_u32(v).is_some(),
            "value type discriminant {} must be valid",
            v
        );
    }
}
