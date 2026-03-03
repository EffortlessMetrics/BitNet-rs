//! Property-based tests — wave 37: GGUF header encoding/decoding, metadata
//! serialization round-trips, KV parsing, and value type invariants.

use bitnet_gguf::kv::{self, GgufHeader};
use bitnet_gguf::{
    GGUF_MAGIC, GGUF_VERSION_MAX, GGUF_VERSION_MIN, GgufMetadataKv, GgufValue, GgufValueType,
    check_magic, parse_header, read_version,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a valid GGUF v2 header from components.
fn build_v2_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(24);
    buf.extend_from_slice(&GGUF_MAGIC);
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&tensor_count.to_le_bytes());
    buf.extend_from_slice(&metadata_count.to_le_bytes());
    buf
}

/// Build a valid GGUF v3 header with specified alignment.
fn build_v3_header(tensor_count: u64, metadata_count: u64, alignment: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(28);
    buf.extend_from_slice(&GGUF_MAGIC);
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&tensor_count.to_le_bytes());
    buf.extend_from_slice(&metadata_count.to_le_bytes());
    buf.extend_from_slice(&alignment.to_le_bytes());
    buf
}

// ---------------------------------------------------------------------------
// Magic & version
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// check_magic returns true iff the first 4 bytes are "GGUF".
    #[test]
    fn check_magic_iff_gguf(data in proptest::collection::vec(0u8..=255, 4..40)) {
        let expected = data.starts_with(b"GGUF");
        prop_assert_eq!(check_magic(&data), expected);
    }

    /// read_version returns Some for valid GGUF data >= 8 bytes.
    #[test]
    fn read_version_valid_data(version in 0u32..=255) {
        let mut buf = Vec::with_capacity(8);
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&version.to_le_bytes());
        let result = read_version(&buf);
        prop_assert_eq!(result, Some(version));
    }

    /// read_version returns None for data shorter than 8 bytes.
    #[test]
    fn read_version_short_data(data in proptest::collection::vec(0u8..=255, 0..8)) {
        // Unless data happens to have valid magic + enough bytes, should be None.
        if data.len() < 8 || !data.starts_with(b"GGUF") {
            prop_assert_eq!(read_version(&data), None);
        }
    }

    /// read_version returns None for bad magic.
    #[test]
    fn read_version_bad_magic(
        b0 in 0u8..=255, b1 in 0u8..=255, b2 in 0u8..=255, b3 in 0u8..=255,
        version in 0u32..=10,
    ) {
        prop_assume!(&[b0, b1, b2, b3] != b"GGUF");
        let mut buf = vec![b0, b1, b2, b3];
        buf.extend_from_slice(&version.to_le_bytes());
        prop_assert_eq!(read_version(&buf), None);
    }
}

// ---------------------------------------------------------------------------
// parse_header: v2
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Valid v2 headers always parse and preserve tensor/metadata counts.
    #[test]
    fn v2_header_roundtrip(
        tensor_count in 0u64..=1_000_000,
        metadata_count in 0u64..=1_000_000,
    ) {
        let buf = build_v2_header(tensor_count, metadata_count);
        let info = parse_header(&buf).unwrap();
        prop_assert_eq!(info.version, 2);
        prop_assert_eq!(info.tensor_count, tensor_count);
        prop_assert_eq!(info.metadata_count, metadata_count);
        prop_assert_eq!(info.alignment, 32);
    }

    /// v2 headers always have alignment == 32 (no alignment field in v2).
    #[test]
    fn v2_alignment_always_32(
        tensor_count in 0u64..=u64::MAX,
        metadata_count in 0u64..=u64::MAX,
    ) {
        let buf = build_v2_header(tensor_count, metadata_count);
        let info = parse_header(&buf).unwrap();
        prop_assert_eq!(info.alignment, 32);
    }
}

// ---------------------------------------------------------------------------
// parse_header: v3
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// v3 headers with power-of-two alignment preserve it.
    #[test]
    fn v3_power_of_two_alignment(
        tensor_count in 0u64..=1_000_000,
        metadata_count in 0u64..=1_000_000,
        exp in 0u32..=20,
    ) {
        let alignment = 1u32 << exp;
        let buf = build_v3_header(tensor_count, metadata_count, alignment);
        let info = parse_header(&buf).unwrap();
        prop_assert_eq!(info.version, 3);
        prop_assert_eq!(info.alignment, alignment);
    }

    /// v3 headers with non-power-of-two alignment fall back to 32.
    #[test]
    fn v3_non_power_of_two_falls_back(
        tensor_count in 0u64..=1_000_000,
        metadata_count in 0u64..=1_000_000,
        alignment in 3u32..=1_000_000,
    ) {
        prop_assume!(!alignment.is_power_of_two());
        let buf = build_v3_header(tensor_count, metadata_count, alignment);
        let info = parse_header(&buf).unwrap();
        prop_assert_eq!(info.alignment, 32);
    }
}

// ---------------------------------------------------------------------------
// parse_header: rejection
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Buffers shorter than 24 bytes are rejected.
    #[test]
    fn short_buffer_rejected(data in proptest::collection::vec(0u8..=255, 0..24)) {
        prop_assert!(parse_header(&data).is_err());
    }

    /// Version outside [2, 3] is rejected (assuming valid magic + 24 bytes).
    #[test]
    fn unsupported_version_rejected(version in prop_oneof![
        (0u32..GGUF_VERSION_MIN),
        ((GGUF_VERSION_MAX + 1)..=255u32),
    ]) {
        let mut buf = build_v2_header(0, 0);
        buf[4..8].copy_from_slice(&version.to_le_bytes());
        prop_assert!(parse_header(&buf).is_err());
    }

    /// Corrupted magic is always rejected.
    #[test]
    fn corrupted_magic_rejected(pos in 0usize..4, byte in 0u8..=255) {
        let expected_byte = b"GGUF"[pos];
        prop_assume!(byte != expected_byte);
        let mut buf = build_v2_header(0, 0);
        buf[pos] = byte;
        prop_assert!(parse_header(&buf).is_err());
    }

    /// parse_header never panics on arbitrary input.
    #[test]
    fn parse_header_no_panic(data in proptest::collection::vec(0u8..=255, 0..128)) {
        let _ = parse_header(&data);
    }
}

// ---------------------------------------------------------------------------
// GgufValueType discriminant round-trip
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// All 13 value-type discriminants [0..12] have a corresponding variant.
    #[test]
    fn value_type_discriminant_coverage(d in 0u32..=12) {
        let vt = GgufValueType::from_u32(d);
        prop_assert!(vt.is_some(), "missing variant for discriminant {}", d);
    }

    /// Discriminants > 12 return None.
    #[test]
    fn value_type_out_of_range(d in 13u32..=1000) {
        prop_assert!(GgufValueType::from_u32(d).is_none());
    }
}

// ---------------------------------------------------------------------------
// GgufValue construction and preservation
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// GgufValue::Uint8 preserves its value.
    #[test]
    fn value_uint8_preserves(v in 0u8..=255) {
        match GgufValue::Uint8(v) {
            GgufValue::Uint8(x) => prop_assert_eq!(x, v),
            _ => prop_assert!(false),
        }
    }

    /// GgufValue::Int32 preserves its value.
    #[test]
    fn value_int32_preserves(v in i32::MIN..=i32::MAX) {
        match GgufValue::Int32(v) {
            GgufValue::Int32(x) => prop_assert_eq!(x, v),
            _ => prop_assert!(false),
        }
    }

    /// GgufValue::Float32 preserves finite values.
    #[test]
    fn value_float32_preserves(v in -1e30f32..1e30) {
        match GgufValue::Float32(v) {
            GgufValue::Float32(x) => prop_assert!((x - v).abs() < 1e-10),
            _ => prop_assert!(false),
        }
    }

    /// GgufValue::Uint64 preserves its value.
    #[test]
    fn value_uint64_preserves(v in 0u64..=u64::MAX) {
        match GgufValue::Uint64(v) {
            GgufValue::Uint64(x) => prop_assert_eq!(x, v),
            _ => prop_assert!(false),
        }
    }

    /// GgufValue::Float64 preserves finite values.
    #[test]
    fn value_float64_preserves(v in -1e100f64..1e100) {
        match GgufValue::Float64(v) {
            GgufValue::Float64(x) => prop_assert!((x - v).abs() < 1e-10),
            _ => prop_assert!(false),
        }
    }

    /// GgufValue::Bool round-trips.
    #[test]
    fn value_bool_preserves(b in proptest::bool::ANY) {
        match GgufValue::Bool(b) {
            GgufValue::Bool(x) => prop_assert_eq!(x, b),
            _ => prop_assert!(false),
        }
    }

    /// GgufValue::String preserves arbitrary UTF-8.
    #[test]
    fn value_string_preserves(s in "[a-zA-Z0-9_ ]{0,100}") {
        match GgufValue::String(s.clone()) {
            GgufValue::String(x) => prop_assert_eq!(x, s),
            _ => prop_assert!(false),
        }
    }

    /// GgufValue::Array preserves element count.
    #[test]
    fn value_array_count(vals in proptest::collection::vec(0u32..100, 0..20)) {
        let arr: Vec<GgufValue> = vals.iter().map(|&v| GgufValue::Uint32(v)).collect();
        let n = arr.len();
        match GgufValue::Array(GgufValueType::Uint32, arr) {
            GgufValue::Array(_, items) => prop_assert_eq!(items.len(), n),
            _ => prop_assert!(false),
        }
    }
}

// ---------------------------------------------------------------------------
// GgufMetadataKv
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// GgufMetadataKv preserves key and value.
    #[test]
    fn metadata_kv_preserves(key in "[a-z.]{1,30}", val in -1000i32..1000) {
        let kv = GgufMetadataKv {
            key: key.clone(),
            value: GgufValue::Int32(val),
        };
        prop_assert_eq!(&kv.key, &key);
        match &kv.value {
            GgufValue::Int32(v) => prop_assert_eq!(*v, val),
            _ => prop_assert!(false, "wrong variant"),
        }
    }
}

// ---------------------------------------------------------------------------
// TensorInfo
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// TensorInfo n_dims matches dims.len().
    #[test]
    fn tensor_info_ndims_matches(dims in proptest::collection::vec(1u64..1000, 1..5)) {
        let ti = bitnet_gguf::TensorInfo {
            name: "test".to_string(),
            n_dims: dims.len() as u32,
            dims: dims.clone(),
            dtype: 0,
            offset: 0,
        };
        prop_assert_eq!(ti.n_dims as usize, ti.dims.len());
    }

    /// TensorInfo clone is independent.
    #[test]
    fn tensor_info_clone_independent(
        name in "[a-z]{1,20}",
        dtype in 0u32..=20,
        offset in 0u64..=u64::MAX,
    ) {
        let ti = bitnet_gguf::TensorInfo {
            name: name.clone(),
            n_dims: 2,
            dims: vec![10, 20],
            dtype,
            offset,
        };
        let cloned = ti.clone();
        prop_assert_eq!(cloned.name, name);
        prop_assert_eq!(cloned.dtype, dtype);
        prop_assert_eq!(cloned.offset, offset);
    }
}

// ---------------------------------------------------------------------------
// kv::parse_header (the kv module's version)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// kv::parse_header on a valid 24-byte buffer succeeds.
    #[test]
    fn kv_parse_header_valid(
        version in 1u32..=3,
        n_tensors in 0u64..=1_000_000,
        n_kv in 0u64..=1_000_000,
    ) {
        let mut buf = [0u8; 24];
        buf[0..4].copy_from_slice(b"GGUF");
        buf[4..8].copy_from_slice(&version.to_le_bytes());
        buf[8..16].copy_from_slice(&n_tensors.to_le_bytes());
        buf[16..24].copy_from_slice(&n_kv.to_le_bytes());
        let hdr = kv::parse_header(&buf).unwrap();
        prop_assert_eq!(hdr.version, version);
        prop_assert_eq!(hdr.n_tensors, n_tensors);
        prop_assert_eq!(hdr.n_kv, n_kv);
    }

    /// kv::parse_header rejects short buffers.
    #[test]
    fn kv_parse_header_short(data in proptest::collection::vec(0u8..=255, 0..24)) {
        let result = kv::parse_header(&data);
        prop_assert!(result.is_err());
    }

    /// kv::parse_header rejects bad magic.
    #[test]
    fn kv_parse_header_bad_magic(
        b0 in 0u8..=255, b1 in 0u8..=255, b2 in 0u8..=255, b3 in 0u8..=255,
        rest in proptest::collection::vec(0u8..=255, 20..30),
    ) {
        prop_assume!(&[b0, b1, b2, b3] != b"GGUF");
        let mut data = vec![b0, b1, b2, b3];
        data.extend_from_slice(&rest);
        let result = kv::parse_header(&data);
        prop_assert!(result.is_err());
    }

    /// kv::parse_header rejects unsupported versions (0, >= 4).
    #[test]
    fn kv_parse_header_bad_version(version in prop_oneof![Just(0u32), (4u32..=255)]) {
        let mut buf = [0u8; 24];
        buf[0..4].copy_from_slice(b"GGUF");
        buf[4..8].copy_from_slice(&version.to_le_bytes());
        let result = kv::parse_header(&buf);
        prop_assert!(result.is_err());
    }
}

// ---------------------------------------------------------------------------
// GgufHeader serde (kv module)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// GgufHeader JSON round-trip.
    #[test]
    fn gguf_header_json_roundtrip(
        version in 1u32..=3,
        n_tensors in 0u64..=1_000_000,
        n_kv in 0u64..=1_000_000,
    ) {
        let hdr = GgufHeader { version, n_tensors, n_kv };
        let json = serde_json::to_string(&hdr).unwrap();
        let back: GgufHeader = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(back, hdr);
    }
}

// ---------------------------------------------------------------------------
// Little-endian encoding consistency
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Tensor count is encoded as little-endian u64 at bytes [8..16].
    #[test]
    fn tensor_count_le_encoding(count in 0u64..=u64::MAX) {
        let buf = build_v2_header(count, 0);
        let encoded = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        prop_assert_eq!(encoded, count);
    }

    /// Metadata count is encoded as little-endian u64 at bytes [16..24].
    #[test]
    fn metadata_count_le_encoding(count in 0u64..=u64::MAX) {
        let buf = build_v2_header(0, count);
        let encoded = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        prop_assert_eq!(encoded, count);
    }

    /// Version is encoded as little-endian u32 at bytes [4..8].
    #[test]
    fn version_le_encoding(version in GGUF_VERSION_MIN..=GGUF_VERSION_MAX) {
        let mut buf = build_v2_header(0, 0);
        buf[4..8].copy_from_slice(&version.to_le_bytes());
        let info = parse_header(&buf).unwrap();
        prop_assert_eq!(info.version, version);
    }
}
