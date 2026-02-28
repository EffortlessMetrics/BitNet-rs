//! Extended property-based and integration tests for `bitnet-gguf`.
//!
//! Covers gaps not addressed by the existing suites:
//!   - Little-endian encoding correctness for header fields
//!   - Exact 24-byte boundary (valid) vs 23-byte (too small)
//!   - Empty and single-byte inputs never panic
//!   - `u64::MAX` tensor/metadata counts preserved
//!   - `TensorInfo` with empty and very long names
//!   - `GgufValue::Array` with String elements
//!   - v3 alignment = 0 (not power-of-two → falls back to 32)
//!   - v3 alignment = 1 (power-of-two, 2^0 → preserved)
//!   - Systematically corrupt each of the 4 magic bytes → `parse_header` error
//!   - Byte-level truncation sweep: sizes 0..24 all fail `parse_header`
//!   - `GgufValue::String` empty string preserved
//!   - `GgufValue` clone depth: Array of cloned elements is independent

use bitnet_gguf::{GgufValue, GgufValueType, TensorInfo, parse_header};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn valid_v2_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut d = Vec::with_capacity(24);
    d.extend_from_slice(b"GGUF");
    d.extend_from_slice(&2u32.to_le_bytes());
    d.extend_from_slice(&tensor_count.to_le_bytes());
    d.extend_from_slice(&metadata_count.to_le_bytes());
    d
}

fn valid_v3_header(tensor_count: u64, metadata_count: u64, alignment: u32) -> Vec<u8> {
    let mut d = valid_v2_header(tensor_count, metadata_count);
    d[4..8].copy_from_slice(&3u32.to_le_bytes());
    d.extend_from_slice(&alignment.to_le_bytes());
    d
}

// ---------------------------------------------------------------------------
// 1. Little-endian encoding: header fields decoded from known byte patterns
// ---------------------------------------------------------------------------

proptest! {
    /// The 8-byte tensor_count field at offset 8 is always decoded as little-endian.
    /// We write a known u64 value byte-by-byte in LE order and verify parse_header
    /// returns the exact same integer.
    #[test]
    fn prop_tensor_count_little_endian_encoding(
        tensor_count in 0u64..u64::MAX,
        metadata_count in 0u64..u64::MAX,
    ) {
        let d = valid_v2_header(tensor_count, metadata_count);
        // Verify the raw bytes at offset 8..16 match little-endian layout.
        let encoded: [u8; 8] = d[8..16].try_into().unwrap();
        let decoded = u64::from_le_bytes(encoded);
        prop_assert_eq!(decoded, tensor_count, "tensor_count must be LE-encoded at offset 8");
        // And parse_header must agree.
        let info = parse_header(&d).expect("valid v2 header must parse");
        prop_assert_eq!(info.tensor_count, tensor_count);
    }

    /// The 8-byte metadata_count field at offset 16 is always decoded as little-endian.
    #[test]
    fn prop_metadata_count_little_endian_encoding(
        tensor_count in 0u64..u64::MAX,
        metadata_count in 0u64..u64::MAX,
    ) {
        let d = valid_v2_header(tensor_count, metadata_count);
        let encoded: [u8; 8] = d[16..24].try_into().unwrap();
        let decoded = u64::from_le_bytes(encoded);
        prop_assert_eq!(decoded, metadata_count, "metadata_count must be LE-encoded at offset 16");
        let info = parse_header(&d).expect("valid v2 header must parse");
        prop_assert_eq!(info.metadata_count, metadata_count);
    }
}

// ---------------------------------------------------------------------------
// 2. Exact boundary: 24 bytes = minimum valid, 23 bytes = always invalid
// ---------------------------------------------------------------------------

#[test]
fn parse_header_exactly_24_bytes_valid_v2_succeeds() {
    let d = valid_v2_header(0, 0);
    assert_eq!(d.len(), 24, "helper must produce exactly 24 bytes");
    assert!(parse_header(&d).is_ok(), "exactly 24 valid bytes must parse");
}

#[test]
fn parse_header_exactly_23_bytes_always_fails() {
    // A valid header up to byte 23 (one byte short).
    let mut d = valid_v2_header(0, 0);
    d.truncate(23);
    assert!(parse_header(&d).is_err(), "23 bytes must always fail `parse_header`");
}

proptest! {
    /// Any input strictly shorter than 24 bytes must fail `parse_header` regardless
    /// of whether the magic bytes are present.
    #[test]
    fn prop_truncation_sweep_below_24_bytes_always_fails(
        len in 0usize..24,
    ) {
        // Build the longest possible valid prefix (may be <= 24 bytes).
        let full = valid_v2_header(0, 0); // exactly 24 bytes
        let truncated = &full[..len];
        prop_assert!(
            parse_header(truncated).is_err(),
            "truncated input of length {len} must fail"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Empty and single-byte inputs never panic
// ---------------------------------------------------------------------------

#[test]
fn parse_header_empty_input_returns_error() {
    let result = parse_header(&[]);
    assert!(result.is_err(), "empty input must return an error, not panic");
}

#[test]
fn parse_header_single_byte_returns_error() {
    for b in 0u8..=255 {
        assert!(parse_header(&[b]).is_err(), "single byte {b:#04x} must return error");
    }
}

// ---------------------------------------------------------------------------
// 4. u64::MAX counts are preserved without truncation
// ---------------------------------------------------------------------------

#[test]
fn parse_header_v2_max_tensor_count_preserved() {
    let d = valid_v2_header(u64::MAX, 0);
    let info = parse_header(&d).expect("u64::MAX tensor_count must parse");
    assert_eq!(info.tensor_count, u64::MAX, "tensor_count u64::MAX must be preserved");
}

#[test]
fn parse_header_v2_max_metadata_count_preserved() {
    let d = valid_v2_header(0, u64::MAX);
    let info = parse_header(&d).expect("u64::MAX metadata_count must parse");
    assert_eq!(info.metadata_count, u64::MAX, "metadata_count u64::MAX must be preserved");
}

// ---------------------------------------------------------------------------
// 5. v3 alignment edge cases: 0 falls back to 32, 1 is preserved
// ---------------------------------------------------------------------------

#[test]
fn parse_header_v3_alignment_zero_falls_back_to_32() {
    // 0 is NOT a power of two (Rust's u32::is_power_of_two returns false for 0).
    let d = valid_v3_header(0, 0, 0);
    let info = parse_header(&d).expect("v3 header with alignment=0 must parse");
    assert_eq!(info.alignment, 32, "alignment=0 (not power-of-two) must fall back to 32");
}

#[test]
fn parse_header_v3_alignment_one_is_preserved() {
    // 1 == 2^0, so it is a power of two and must be preserved.
    let d = valid_v3_header(0, 0, 1);
    let info = parse_header(&d).expect("v3 header with alignment=1 must parse");
    assert_eq!(info.alignment, 1, "alignment=1 (2^0, power-of-two) must be preserved");
}

proptest! {
    /// For v3 headers, alignment equal to any power of two in [1, 2^30] is preserved.
    #[test]
    fn prop_v3_alignment_any_power_of_two_preserved(exp in 0u32..=30u32) {
        let alignment = 1u32 << exp;
        let d = valid_v3_header(0, 0, alignment);
        let info = parse_header(&d).expect("v3 header with power-of-two alignment must parse");
        prop_assert_eq!(info.alignment, alignment, "alignment 2^{}={} must be preserved", exp, alignment);
    }
}

// ---------------------------------------------------------------------------
// 6. Corrupting each magic byte individually causes parse_header to fail
// ---------------------------------------------------------------------------

proptest! {
    /// Flipping any one of the 4 magic bytes to a wrong value causes parse_header
    /// to return an error (as long as the replacement doesn't accidentally reconstruct
    /// the original byte, leaving magic intact).
    #[test]
    fn prop_single_magic_byte_corruption_causes_error(
        (pos, replacement) in (0usize..4usize).prop_flat_map(|p| {
            let orig = b"GGUF"[p];
            (Just(p), any::<u8>().prop_filter("not original byte", move |&b| b != orig))
        }),
    ) {
        let mut d = valid_v2_header(0, 0);
        d[pos] = replacement;
        // The magic is now corrupted at position `pos`.
        prop_assert!(
            parse_header(&d).is_err(),
            "corrupting magic byte {} to {:#04x} must cause parse_header to fail",
            pos, replacement,
        );
    }
}

// ---------------------------------------------------------------------------
// 7. TensorInfo: empty name and long name are preserved without modification
// ---------------------------------------------------------------------------

#[test]
fn tensor_info_empty_name_is_preserved() {
    let info = TensorInfo { name: String::new(), n_dims: 0, dims: vec![], dtype: 0, offset: 0 };
    assert!(info.name.is_empty(), "empty name must be preserved in TensorInfo");
}

proptest! {
    /// A TensorInfo with an arbitrarily long name does not truncate it.
    #[test]
    fn prop_tensor_info_long_name_preserved(
        name in prop::collection::vec(prop::char::range('a', 'z'), 256..=1024),
    ) {
        let s: String = name.into_iter().collect();
        let expected_len = s.len();
        let info = TensorInfo {
            name: s.clone(),
            n_dims: 0,
            dims: vec![],
            dtype: 0,
            offset: 0,
        };
        prop_assert_eq!(
            info.name.len(), expected_len,
            "long TensorInfo name must not be truncated ({} chars)", expected_len
        );
        prop_assert_eq!(&info.name, &s);
    }
}

// ---------------------------------------------------------------------------
// 8. `GgufValue::Array` with String elements
// ---------------------------------------------------------------------------

proptest! {
    /// `GgufValue::Array` with String elements preserves all element values and count.
    #[test]
    fn prop_gguf_value_array_string_elements_preserved(
        strings in prop::collection::vec("[a-z]{0,32}", 0..=16_usize),
    ) {
        let elems: Vec<GgufValue> = strings.iter().map(|s| GgufValue::String(s.clone())).collect();
        let expected_count = elems.len();
        let arr = GgufValue::Array(GgufValueType::String, elems);
        let GgufValue::Array(elem_type, stored) = arr else {
            panic!("unexpected GgufValue variant");
        };
        assert_eq!(elem_type, GgufValueType::String, "element type must be String");
        prop_assert_eq!(stored.len(), expected_count, "array length must be preserved");
        for (i, (stored_val, orig)) in stored.iter().zip(strings.iter()).enumerate() {
            let GgufValue::String(s) = stored_val else {
                prop_assert!(false, "element {i} changed variant");
                return Ok(());
            };
            prop_assert_eq!(s, orig, "string element must be preserved verbatim ({})", i);
        }
    }
}

// ---------------------------------------------------------------------------
// 9. GgufValue::String: empty string is stored without modification
// ---------------------------------------------------------------------------

#[test]
fn gguf_value_string_empty_is_preserved() {
    let GgufValue::String(stored) = GgufValue::String(String::new()) else {
        panic!("unexpected variant");
    };
    assert!(stored.is_empty(), "empty string must be preserved in GgufValue::String");
}

// ---------------------------------------------------------------------------
// 10. GgufValue::Array clone independence: mutating clone does not affect original
// ---------------------------------------------------------------------------

proptest! {
    /// After cloning a GgufValue::Array, modifying the clone's inner Vec does not
    /// affect the original (deep clone invariant).
    #[test]
    fn prop_gguf_value_array_clone_is_deep(
        values in prop::collection::vec(any::<u8>(), 1..=32),
    ) {
        let elems: Vec<GgufValue> = values.iter().map(|&v| GgufValue::Uint8(v)).collect();
        let original_len = elems.len();
        let original = GgufValue::Array(GgufValueType::Uint8, elems);
        let mut cloned = original.clone();

        // Truncate the clone's inner vec.
        if let GgufValue::Array(_, ref mut inner) = cloned {
            inner.clear();
        }

        // The original must still have all its elements.
        let GgufValue::Array(_, original_inner) = original else {
            panic!("unexpected variant");
        };
        prop_assert_eq!(
            original_inner.len(), original_len,
            "mutating clone must not affect original array length"
        );
    }
}

// ---------------------------------------------------------------------------
// 11. parse_header never panics on large arbitrary inputs
// ---------------------------------------------------------------------------

proptest! {
    /// parse_header must never panic on large arbitrary byte slices (up to 4 KiB).
    #[test]
    fn prop_parse_header_no_panic_large_arbitrary_input(
        data in prop::collection::vec(any::<u8>(), 0..=4096),
    ) {
        // Must not panic under any circumstances.
        let _ = parse_header(&data);
    }
}

// ---------------------------------------------------------------------------
// 12. Version field little-endian: explicit byte verification
// ---------------------------------------------------------------------------

proptest! {
    /// The 4-byte version field at bytes 4..8 is always decoded as little-endian.
    /// We verify that writing version=2 in LE produces the expected raw bytes, and
    /// that parse_header reads the correct value back from those bytes.
    #[test]
    fn prop_version_field_little_endian_roundtrip(version in 2u32..=3u32) {
        let d = valid_v2_header(0, 0);
        // Manually re-read the version bytes and decode as LE.
        let raw: [u8; 4] = d[4..8].try_into().unwrap();
        let decoded_version = u32::from_le_bytes(raw);
        prop_assert_eq!(decoded_version, 2u32, "helper writes version=2; LE decode must agree");
        // Now test an actual v2 or v3 header.
        let mut hdr = valid_v2_header(0, 0);
        hdr[4..8].copy_from_slice(&version.to_le_bytes());
        let info = parse_header(&hdr).expect("valid header must parse");
        let raw2: [u8; 4] = hdr[4..8].try_into().unwrap();
        prop_assert_eq!(u32::from_le_bytes(raw2), version, "LE bytes must decode to version");
        prop_assert_eq!(info.version, version, "parse_header must agree with LE decode");
    }
}
