//! Property tests for `bitnet-gguf` header parser.
//!
//! Uses proptest to verify parser invariants across the full range of
//! syntactically-valid and invalid input shapes.

use bitnet_gguf::{GGUF_MAGIC, GgufValueType, check_magic, parse_header, read_version};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helper: build a minimal valid GGUF header byte slice
// ---------------------------------------------------------------------------

/// Build a well-formed GGUF header: magic + 4-byte version + 8-byte `tensor_count` +
/// 8-byte `metadata_kv_count` = 24 bytes.  Version is clamped to [2, 3].
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

// ---------------------------------------------------------------------------
// Properties: read_version
// ---------------------------------------------------------------------------

proptest! {
    /// `read_version` agrees with `parse_header` on the version field for any
    /// well-formed GGUF v2/v3 header.
    #[test]
    fn prop_read_version_consistent_with_parse_header(version in 2u32..=3u32) {
        let data = make_valid_header(version);
        let rv = read_version(&data);
        prop_assert_eq!(rv, Some(version), "read_version must agree with make_valid_header");
        let info = parse_header(&data).unwrap();
        prop_assert_eq!(rv, Some(info.version), "read_version and parse_header must agree on version");
    }

    /// `read_version` returns `None` for data shorter than 8 bytes.
    #[test]
    fn prop_read_version_rejects_short_data(data in prop::collection::vec(any::<u8>(), 0..8)) {
        // We only assert None when the magic is present but data is too short;
        // if magic is absent the result is already None.
        if data.len() < 8 {
            prop_assert_eq!(read_version(&data), None);
        }
    }

    /// `read_version` returns `None` when the magic is wrong, regardless of length.
    #[test]
    fn prop_read_version_returns_none_on_bad_magic(
        bad_first in any::<u8>().prop_filter("not G", |&b| b != b'G'),
        tail in prop::collection::vec(any::<u8>(), 7..32),
    ) {
        let mut data = vec![bad_first];
        data.extend(tail);
        prop_assert_eq!(read_version(&data), None);
    }
}

// ---------------------------------------------------------------------------
// Properties: unsupported version rejection
// ---------------------------------------------------------------------------

proptest! {
    /// Versions below the minimum (< 2) must be rejected by `parse_header`.
    #[test]
    fn prop_parse_header_rejects_version_below_minimum(version in 0u32..2u32) {
        let mut data = make_valid_header(2);
        data[4..8].copy_from_slice(&version.to_le_bytes());
        prop_assert!(
            parse_header(&data).is_err(),
            "version {version} is below minimum and must be rejected"
        );
    }

    /// Versions above the maximum (> 3) must be rejected by `parse_header`.
    #[test]
    fn prop_parse_header_rejects_version_above_maximum(version in 4u32..u32::MAX) {
        let mut data = make_valid_header(2);
        data[4..8].copy_from_slice(&version.to_le_bytes());
        prop_assert!(
            parse_header(&data).is_err(),
            "version {version} is above maximum and must be rejected"
        );
    }
}

// ---------------------------------------------------------------------------
// Properties: v3 alignment field
// ---------------------------------------------------------------------------

proptest! {
    /// For GGUF v3, when the alignment field is a power of two it must be
    /// preserved in the parsed result.
    #[test]
    fn prop_v3_alignment_power_of_two_is_preserved(
        // Generate powers of two in the range [1, 65536].
        exp in 0u32..=16u32,
    ) {
        let alignment: u32 = 1u32 << exp;
        let mut data = make_valid_header(3);
        data.extend_from_slice(&alignment.to_le_bytes());
        let info = parse_header(&data).expect("valid v3 header must parse");
        prop_assert_eq!(
            info.alignment, alignment,
            "power-of-two alignment {} must be preserved", alignment
        );
    }

    /// For GGUF v3, a non-power-of-two alignment field must fall back to 32.
    #[test]
    fn prop_v3_non_power_of_two_alignment_falls_back_to_32(
        // Values that are guaranteed not to be powers of two (>= 3 and odd, or
        // specific non-power values in a safe range).
        alignment in (3u32..=u32::MAX).prop_filter("not power of two", |&a| !a.is_power_of_two()),
    ) {
        let mut data = make_valid_header(3);
        data.extend_from_slice(&alignment.to_le_bytes());
        let info = parse_header(&data).expect("valid v3 header (with bad alignment) must still parse");
        prop_assert_eq!(
            info.alignment, 32,
            "non-power-of-two alignment {} must fall back to 32", alignment
        );
    }

    /// For GGUF v2, the alignment is always 32 regardless of what follows
    /// the 24-byte header.
    #[test]
    fn prop_v2_alignment_is_always_32(extra in prop::collection::vec(any::<u8>(), 0..16)) {
        let mut data = make_valid_header(2);
        data.extend(extra);
        let info = parse_header(&data).expect("valid v2 header must parse");
        prop_assert_eq!(info.alignment, 32, "v2 alignment must always be 32");
    }
}
