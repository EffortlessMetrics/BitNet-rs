//! Extended tests for `bitnet-gguf`.
//!
//! Covers gaps not addressed by the existing suites:
//!   - `GgufValue::Uint8` (0, 255, mid-range) stored verbatim
//!   - `GgufValue::Uint32` stored verbatim
//!   - `GgufValue::Float32` stored with bit-identical precision
//!   - `GgufMetadataKv` with Float32 / Int32 values
//!   - `GgufMetadataKv` key with dot-separated segments preserved
//!   - v3 header with exactly 24 bytes (no alignment field) → defaults to 32
//!   - `read_version` returns `Some(3)` for a v3 header
//!   - `parse_header` v3 with alignment = 32 (preserved exactly)
//!   - `parse_header` v3 with alignment = 65536 (large power-of-two, preserved)
//!   - `TensorInfo` offset field and dtype field preserved
//!   - `TensorInfo` with multiple dims stored in order
//!   - `GgufValue::Array` with Bool elements
//!   - `GgufFileInfo` Debug output contains "version"
//!   - `GGUF_MAGIC`, `GGUF_VERSION_MIN`, `GGUF_VERSION_MAX` constant assertions
//!   - `GgufValueType` specific discriminants: Bool=7, String=8, Array=9
//!   - Property tests: `GgufValue::Uint8` and `GgufValue::Uint32` round-trips
//!   - Property test: arbitrary ASCII tensor names stored verbatim

use bitnet_gguf::{
    GGUF_MAGIC, GGUF_VERSION_MAX, GGUF_VERSION_MIN, GgufFileInfo, GgufMetadataKv, GgufValue,
    GgufValueType, TensorInfo, check_magic, parse_header, read_version,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn v2_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut d = Vec::with_capacity(24);
    d.extend_from_slice(b"GGUF");
    d.extend_from_slice(&2u32.to_le_bytes());
    d.extend_from_slice(&tensor_count.to_le_bytes());
    d.extend_from_slice(&metadata_count.to_le_bytes());
    d
}

fn v3_header(tensor_count: u64, metadata_count: u64, alignment: u32) -> Vec<u8> {
    let mut d = Vec::with_capacity(28);
    d.extend_from_slice(b"GGUF");
    d.extend_from_slice(&3u32.to_le_bytes());
    d.extend_from_slice(&tensor_count.to_le_bytes());
    d.extend_from_slice(&metadata_count.to_le_bytes());
    d.extend_from_slice(&alignment.to_le_bytes());
    d
}

// ---------------------------------------------------------------------------
// 1. GgufValue::Uint8 unit tests
// ---------------------------------------------------------------------------

#[test]
fn gguf_value_uint8_zero_stored() {
    let GgufValue::Uint8(v) = GgufValue::Uint8(0) else { panic!("wrong variant") };
    assert_eq!(v, 0u8);
}

#[test]
fn gguf_value_uint8_max_stored() {
    let GgufValue::Uint8(v) = GgufValue::Uint8(255) else { panic!("wrong variant") };
    assert_eq!(v, 255u8);
}

#[test]
fn gguf_value_uint8_mid_value_stored() {
    let GgufValue::Uint8(v) = GgufValue::Uint8(128) else { panic!("wrong variant") };
    assert_eq!(v, 128u8);
}

// ---------------------------------------------------------------------------
// 2. GgufValue::Uint32 unit test
// ---------------------------------------------------------------------------

#[test]
fn gguf_value_uint32_stored() {
    let GgufValue::Uint32(v) = GgufValue::Uint32(42) else { panic!("wrong variant") };
    assert_eq!(v, 42u32);
}

#[test]
fn gguf_value_uint32_max_stored() {
    let GgufValue::Uint32(v) = GgufValue::Uint32(u32::MAX) else { panic!("wrong variant") };
    assert_eq!(v, u32::MAX);
}

// ---------------------------------------------------------------------------
// 3. GgufValue::Float32 unit test
// ---------------------------------------------------------------------------

#[test]
fn gguf_value_float32_pi_stored() {
    let pi = std::f32::consts::PI;
    let GgufValue::Float32(v) = GgufValue::Float32(pi) else { panic!("wrong variant") };
    // Exact bit-identity (no conversions involved).
    assert_eq!(v.to_bits(), pi.to_bits());
}

#[test]
fn gguf_value_float32_zero_stored() {
    let GgufValue::Float32(v) = GgufValue::Float32(0.0f32) else { panic!("wrong variant") };
    assert_eq!(v, 0.0f32);
}

// ---------------------------------------------------------------------------
// 4. GgufValue::Bool unit tests
// ---------------------------------------------------------------------------

#[test]
fn gguf_value_bool_true_stored() {
    let GgufValue::Bool(v) = GgufValue::Bool(true) else { panic!("wrong variant") };
    assert!(v);
}

#[test]
fn gguf_value_bool_false_stored() {
    let GgufValue::Bool(v) = GgufValue::Bool(false) else { panic!("wrong variant") };
    assert!(!v);
}

// ---------------------------------------------------------------------------
// 5. GgufMetadataKv with various value types
// ---------------------------------------------------------------------------

#[test]
fn metadata_kv_float32_value_stored() {
    let kv =
        GgufMetadataKv { key: "general.temperature".to_string(), value: GgufValue::Float32(0.7) };
    let GgufValue::Float32(v) = &kv.value else { panic!("expected Float32") };
    assert!((v - 0.7f32).abs() < 1e-6, "Float32 metadata value must be preserved");
}

#[test]
fn metadata_kv_int32_value_stored() {
    let kv =
        GgufMetadataKv { key: "general.context_length".to_string(), value: GgufValue::Int32(-1) };
    let GgufValue::Int32(v) = &kv.value else { panic!("expected Int32") };
    assert_eq!(*v, -1i32);
}

/// Dot-separated key (GGUF convention: "general.architecture") is preserved.
#[test]
fn metadata_kv_dot_separated_key_preserved() {
    let key = "general.architecture";
    let kv = GgufMetadataKv { key: key.to_string(), value: GgufValue::String("bitnet".into()) };
    assert_eq!(&kv.key, key, "dot-separated key must be preserved verbatim");
}

/// Nested dot key with multiple segments is preserved.
#[test]
fn metadata_kv_nested_dot_key_preserved() {
    let key = "llama.attention.head_count";
    let kv = GgufMetadataKv { key: key.to_string(), value: GgufValue::Uint32(32) };
    assert_eq!(&kv.key, key);
}

// ---------------------------------------------------------------------------
// 6. v3 header with exactly 24 bytes (no alignment field) → default 32
// ---------------------------------------------------------------------------

#[test]
fn parse_header_v3_no_alignment_field_defaults_to_32() {
    // Build a header that looks like v3 but has only 24 bytes (no alignment u32 at [24..28]).
    let mut d = v2_header(0, 0);
    d[4..8].copy_from_slice(&3u32.to_le_bytes()); // set version=3
    assert_eq!(d.len(), 24, "must be exactly 24 bytes");
    let info = parse_header(&d).expect("v3 with 24 bytes must parse (no alignment byte)");
    assert_eq!(info.version, 3);
    assert_eq!(info.alignment, 32, "missing alignment field must fall back to 32");
}

// ---------------------------------------------------------------------------
// 7. read_version for v3
// ---------------------------------------------------------------------------

#[test]
fn read_version_v3() {
    let d = v3_header(5, 10, 32);
    let version = read_version(&d).expect("read_version must succeed for a v3 header");
    assert_eq!(version, 3u32);
}

#[test]
fn read_version_v2_agrees_with_parse_header() {
    let d = v2_header(3, 5);
    let version = read_version(&d).expect("read_version must succeed");
    let info = parse_header(&d).expect("parse_header must succeed");
    assert_eq!(version, info.version, "read_version and parse_header must agree on version");
}

// ---------------------------------------------------------------------------
// 8. parse_header v3 with alignment = 32 and alignment = 65536
// ---------------------------------------------------------------------------

#[test]
fn parse_header_v3_alignment_32_preserved() {
    let d = v3_header(0, 0, 32);
    let info = parse_header(&d).expect("v3 with alignment=32 must parse");
    assert_eq!(info.alignment, 32, "alignment=32 (power-of-two) must be preserved");
}

#[test]
fn parse_header_v3_alignment_65536_preserved() {
    let d = v3_header(0, 0, 65536);
    let info = parse_header(&d).expect("v3 with alignment=65536 must parse");
    assert_eq!(info.alignment, 65536, "alignment=65536 (2^16) must be preserved");
}

#[test]
fn parse_header_v3_alignment_u32_max_falls_back_to_32() {
    // u32::MAX is not a power of two.
    let d = v3_header(0, 0, u32::MAX);
    let info = parse_header(&d).expect("v3 with alignment=u32::MAX must still parse");
    assert_eq!(info.alignment, 32, "u32::MAX alignment must fall back to 32");
}

// ---------------------------------------------------------------------------
// 9. TensorInfo: offset, dtype, dims ordering
// ---------------------------------------------------------------------------

#[test]
fn tensor_info_offset_max_preserved() {
    let info = TensorInfo {
        name: "weight".to_string(),
        n_dims: 1,
        dims: vec![1024],
        dtype: 0,
        offset: u64::MAX,
    };
    assert_eq!(info.offset, u64::MAX, "offset u64::MAX must be preserved in TensorInfo");
}

#[test]
fn tensor_info_dtype_preserved() {
    let info =
        TensorInfo { name: "bias".to_string(), n_dims: 1, dims: vec![512], dtype: 7, offset: 0 };
    assert_eq!(info.dtype, 7u32, "dtype must be preserved verbatim in TensorInfo");
}

#[test]
fn tensor_info_dims_order_preserved() {
    let dims = vec![32u64, 128, 256, 4096];
    let info = TensorInfo {
        name: "hidden".to_string(),
        n_dims: dims.len() as u32,
        dims: dims.clone(),
        dtype: 0,
        offset: 0,
    };
    assert_eq!(info.dims, dims, "dims must be stored in the original order");
}

// ---------------------------------------------------------------------------
// 10. GgufValue::Array with Bool elements
// ---------------------------------------------------------------------------

#[test]
fn gguf_value_array_bool_elements_preserved() {
    let elems = vec![GgufValue::Bool(true), GgufValue::Bool(false), GgufValue::Bool(true)];
    let arr = GgufValue::Array(GgufValueType::Bool, elems);
    let GgufValue::Array(elem_type, stored) = arr else { panic!("expected Array") };
    assert_eq!(elem_type, GgufValueType::Bool);
    assert_eq!(stored.len(), 3);
    let bools: Vec<bool> = stored
        .iter()
        .map(|v| match v {
            GgufValue::Bool(b) => *b,
            _ => panic!("expected Bool element"),
        })
        .collect();
    assert_eq!(bools, vec![true, false, true]);
}

// ---------------------------------------------------------------------------
// 11. GgufFileInfo Debug output smoke test
// ---------------------------------------------------------------------------

#[test]
fn gguf_file_info_debug_contains_version() {
    let info = GgufFileInfo { version: 2, tensor_count: 5, metadata_count: 3, alignment: 32 };
    let debug = format!("{info:?}");
    assert!(
        debug.contains("version"),
        "GgufFileInfo Debug output must contain 'version'; got: {debug}"
    );
}

// ---------------------------------------------------------------------------
// 12. Constants assertions
// ---------------------------------------------------------------------------

#[test]
fn gguf_magic_constant_is_gguf() {
    assert_eq!(&GGUF_MAGIC, b"GGUF", "GGUF_MAGIC must be exactly b\"GGUF\"");
}

#[test]
fn gguf_version_min_is_2() {
    assert_eq!(GGUF_VERSION_MIN, 2u32, "GGUF_VERSION_MIN must be 2");
}

#[test]
fn gguf_version_max_is_3() {
    assert_eq!(GGUF_VERSION_MAX, 3u32, "GGUF_VERSION_MAX must be 3");
}

// ---------------------------------------------------------------------------
// 13. GgufValueType specific discriminant values
// ---------------------------------------------------------------------------

#[test]
fn gguf_value_type_bool_discriminant_is_7() {
    assert_eq!(GgufValueType::Bool as u32, 7u32);
    assert_eq!(GgufValueType::from_u32(7), Some(GgufValueType::Bool));
}

#[test]
fn gguf_value_type_string_discriminant_is_8() {
    assert_eq!(GgufValueType::String as u32, 8u32);
    assert_eq!(GgufValueType::from_u32(8), Some(GgufValueType::String));
}

#[test]
fn gguf_value_type_array_discriminant_is_9() {
    assert_eq!(GgufValueType::Array as u32, 9u32);
    assert_eq!(GgufValueType::from_u32(9), Some(GgufValueType::Array));
}

#[test]
fn gguf_value_type_float32_discriminant_is_6() {
    assert_eq!(GgufValueType::Float32 as u32, 6u32);
    assert_eq!(GgufValueType::from_u32(6), Some(GgufValueType::Float32));
}

#[test]
fn gguf_value_type_uint32_discriminant_is_4() {
    assert_eq!(GgufValueType::Uint32 as u32, 4u32);
    assert_eq!(GgufValueType::from_u32(4), Some(GgufValueType::Uint32));
}

// ---------------------------------------------------------------------------
// 14. check_magic edge cases
// ---------------------------------------------------------------------------

#[test]
fn check_magic_exact_four_bytes_gguf() {
    assert!(check_magic(b"GGUF"), "exactly b\"GGUF\" must pass check_magic");
}

#[test]
fn check_magic_with_trailing_zeros() {
    let mut d = vec![0u8; 32];
    d[..4].copy_from_slice(b"GGUF");
    assert!(check_magic(&d), "GGUF followed by zeros must pass check_magic");
}

#[test]
fn check_magic_ggml_fails() {
    assert!(!check_magic(b"GGML"), "GGML must not pass check_magic");
}

// ---------------------------------------------------------------------------
// 15. Property tests
// ---------------------------------------------------------------------------

proptest! {
    /// GgufValue::Uint8 stores any u8 value verbatim (proptest).
    #[test]
    fn prop_gguf_value_uint8_round_trip(v in any::<u8>()) {
        let GgufValue::Uint8(stored) = GgufValue::Uint8(v) else {
            panic!("unexpected variant");
        };
        prop_assert_eq!(stored, v);
    }

    /// GgufValue::Uint32 stores any u32 value verbatim (proptest).
    #[test]
    fn prop_gguf_value_uint32_round_trip(v in any::<u32>()) {
        let GgufValue::Uint32(stored) = GgufValue::Uint32(v) else {
            panic!("unexpected variant");
        };
        prop_assert_eq!(stored, v);
    }

    /// Arbitrary printable ASCII tensor names are stored verbatim in TensorInfo.
    #[test]
    fn prop_tensor_info_ascii_name_stored_verbatim(
        name in "[a-zA-Z][a-zA-Z0-9_./]{0,63}",
    ) {
        let info = TensorInfo {
            name: name.clone(),
            n_dims: 0,
            dims: vec![],
            dtype: 0,
            offset: 0,
        };
        prop_assert_eq!(&info.name, &name, "tensor name must be stored verbatim");
    }

    /// GgufValue::Array with Uint32 elements preserves every element value.
    #[test]
    fn prop_gguf_value_array_uint32_elements_preserved(
        values in prop::collection::vec(any::<u32>(), 0..=32),
    ) {
        let elems: Vec<GgufValue> = values.iter().map(|&v| GgufValue::Uint32(v)).collect();
        let expected = values.clone();
        let arr = GgufValue::Array(GgufValueType::Uint32, elems);
        let GgufValue::Array(_, stored) = arr else { panic!("expected Array") };
        prop_assert_eq!(stored.len(), expected.len());
        for (i, (sv, &ev)) in stored.iter().zip(expected.iter()).enumerate() {
            let GgufValue::Uint32(x) = sv else {
                prop_assert!(false, "element {i} is not Uint32");
                return Ok(());
            };
            prop_assert_eq!(*x, ev);
        }
    }

    /// valid v3 headers with various tensor/metadata counts always parse.
    #[test]
    fn prop_v3_header_with_any_counts_parses(
        tensor_count in 0u64..1_000_000u64,
        metadata_count in 0u64..1_000_000u64,
        exp in 0u32..=16u32,
    ) {
        let alignment = 1u32 << exp;
        let d = v3_header(tensor_count, metadata_count, alignment);
        let info = parse_header(&d).expect("v3 header with valid counts must parse");
        prop_assert_eq!(info.version, 3);
        prop_assert_eq!(info.tensor_count, tensor_count);
        prop_assert_eq!(info.metadata_count, metadata_count);
        prop_assert_eq!(info.alignment, alignment);
    }
}
