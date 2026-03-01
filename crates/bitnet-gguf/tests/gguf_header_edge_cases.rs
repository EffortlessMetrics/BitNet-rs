//! Edge-case tests for the GGUF header parser and type system.
//!
//! Tests cover:
//! - GGUF magic validation (check_magic)
//! - Version reading (read_version)
//! - Header parsing (parse_header) — valid, invalid, boundary
//! - GgufValueType discriminant conversion
//! - GgufValue variants construction and Debug
//! - GgufFileInfo fields
//! - GgufMetadataKv construction
//! - TensorInfo construction
//! - Constants (GGUF_MAGIC, GGUF_VERSION_MIN, GGUF_VERSION_MAX)

use bitnet_gguf::{
    GGUF_MAGIC, GGUF_VERSION_MAX, GGUF_VERSION_MIN, GgufFileInfo, GgufMetadataKv, GgufValue,
    GgufValueType, TensorInfo, check_magic, parse_header, read_version,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#[test]
fn magic_constant_is_gguf() {
    assert_eq!(&GGUF_MAGIC, b"GGUF");
}

#[test]
fn version_range() {
    assert_eq!(GGUF_VERSION_MIN, 2);
    assert_eq!(GGUF_VERSION_MAX, 3);
    assert!(GGUF_VERSION_MIN <= GGUF_VERSION_MAX);
}

// ---------------------------------------------------------------------------
// check_magic
// ---------------------------------------------------------------------------

#[test]
fn check_magic_valid() {
    let data =
        b"GGUF\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    assert!(check_magic(data));
}

#[test]
fn check_magic_invalid_prefix() {
    let data = b"XXXX\x02\x00\x00\x00";
    assert!(!check_magic(data));
}

#[test]
fn check_magic_empty() {
    assert!(!check_magic(&[]));
}

#[test]
fn check_magic_too_short() {
    assert!(!check_magic(b"GGU"));
}

#[test]
fn check_magic_exact_four_bytes() {
    assert!(check_magic(b"GGUF"));
}

#[test]
fn check_magic_case_sensitive() {
    assert!(!check_magic(b"gguf"));
    assert!(!check_magic(b"Gguf"));
}

// ---------------------------------------------------------------------------
// read_version
// ---------------------------------------------------------------------------

#[test]
fn read_version_v2() {
    let mut data = vec![0u8; 8];
    data[..4].copy_from_slice(b"GGUF");
    data[4..8].copy_from_slice(&2u32.to_le_bytes());
    assert_eq!(read_version(&data), Some(2));
}

#[test]
fn read_version_v3() {
    let mut data = vec![0u8; 8];
    data[..4].copy_from_slice(b"GGUF");
    data[4..8].copy_from_slice(&3u32.to_le_bytes());
    assert_eq!(read_version(&data), Some(3));
}

#[test]
fn read_version_too_short() {
    assert_eq!(read_version(&[0u8; 4]), None);
    assert_eq!(read_version(&[]), None);
}

#[test]
fn read_version_bad_magic() {
    let mut data = vec![0u8; 8];
    data[..4].copy_from_slice(b"XXXX");
    data[4..8].copy_from_slice(&2u32.to_le_bytes());
    assert_eq!(read_version(&data), None);
}

#[test]
fn read_version_v1_still_returns_value() {
    // read_version just reads the u32 — it doesn't validate the range
    let mut data = vec![0u8; 8];
    data[..4].copy_from_slice(b"GGUF");
    data[4..8].copy_from_slice(&1u32.to_le_bytes());
    assert_eq!(read_version(&data), Some(1));
}

// ---------------------------------------------------------------------------
// parse_header — valid
// ---------------------------------------------------------------------------

fn make_v2_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut data = vec![0u8; 24];
    data[..4].copy_from_slice(b"GGUF");
    data[4..8].copy_from_slice(&2u32.to_le_bytes());
    data[8..16].copy_from_slice(&tensor_count.to_le_bytes());
    data[16..24].copy_from_slice(&metadata_count.to_le_bytes());
    data
}

fn make_v3_header(tensor_count: u64, metadata_count: u64, alignment: u32) -> Vec<u8> {
    let mut data = vec![0u8; 28];
    data[..4].copy_from_slice(b"GGUF");
    data[4..8].copy_from_slice(&3u32.to_le_bytes());
    data[8..16].copy_from_slice(&tensor_count.to_le_bytes());
    data[16..24].copy_from_slice(&metadata_count.to_le_bytes());
    data[24..28].copy_from_slice(&alignment.to_le_bytes());
    data
}

#[test]
fn parse_header_v2_valid() {
    let data = make_v2_header(10, 5);
    let info = parse_header(&data).unwrap();
    assert_eq!(info.version, 2);
    assert_eq!(info.tensor_count, 10);
    assert_eq!(info.metadata_count, 5);
    assert_eq!(info.alignment, 32); // default for v2
}

#[test]
fn parse_header_v3_valid() {
    let data = make_v3_header(100, 50, 64);
    let info = parse_header(&data).unwrap();
    assert_eq!(info.version, 3);
    assert_eq!(info.tensor_count, 100);
    assert_eq!(info.metadata_count, 50);
    assert_eq!(info.alignment, 64);
}

#[test]
fn parse_header_v3_non_power_of_two_alignment_defaults() {
    // Non-power-of-two alignment should fall back to 32
    let data = make_v3_header(1, 1, 48); // 48 is not power of 2
    let info = parse_header(&data).unwrap();
    assert_eq!(info.alignment, 32);
}

#[test]
fn parse_header_v3_alignment_1() {
    let data = make_v3_header(1, 1, 1); // 1 is power of 2
    let info = parse_header(&data).unwrap();
    assert_eq!(info.alignment, 1);
}

#[test]
fn parse_header_zero_counts() {
    let data = make_v2_header(0, 0);
    let info = parse_header(&data).unwrap();
    assert_eq!(info.tensor_count, 0);
    assert_eq!(info.metadata_count, 0);
}

#[test]
fn parse_header_large_counts() {
    let data = make_v2_header(u64::MAX, u64::MAX);
    let info = parse_header(&data).unwrap();
    assert_eq!(info.tensor_count, u64::MAX);
    assert_eq!(info.metadata_count, u64::MAX);
}

// ---------------------------------------------------------------------------
// parse_header — invalid
// ---------------------------------------------------------------------------

#[test]
fn parse_header_too_small() {
    let result = parse_header(&[0u8; 10]);
    assert!(result.is_err());
}

#[test]
fn parse_header_bad_magic() {
    let mut data = make_v2_header(1, 1);
    data[0] = b'X';
    assert!(parse_header(&data).is_err());
}

#[test]
fn parse_header_unsupported_version_v1() {
    let mut data = vec![0u8; 24];
    data[..4].copy_from_slice(b"GGUF");
    data[4..8].copy_from_slice(&1u32.to_le_bytes()); // v1 unsupported
    let result = parse_header(&data);
    assert!(result.is_err());
}

#[test]
fn parse_header_unsupported_version_v99() {
    let mut data = vec![0u8; 24];
    data[..4].copy_from_slice(b"GGUF");
    data[4..8].copy_from_slice(&99u32.to_le_bytes());
    assert!(parse_header(&data).is_err());
}

#[test]
fn parse_header_empty() {
    assert!(parse_header(&[]).is_err());
}

// ---------------------------------------------------------------------------
// GgufValueType
// ---------------------------------------------------------------------------

#[test]
fn value_type_from_u32_all_valid() {
    let expected = [
        (0, GgufValueType::Uint8),
        (1, GgufValueType::Int8),
        (2, GgufValueType::Uint16),
        (3, GgufValueType::Int16),
        (4, GgufValueType::Uint32),
        (5, GgufValueType::Int32),
        (6, GgufValueType::Float32),
        (7, GgufValueType::Bool),
        (8, GgufValueType::String),
        (9, GgufValueType::Array),
        (10, GgufValueType::Uint64),
        (11, GgufValueType::Int64),
        (12, GgufValueType::Float64),
    ];
    for (raw, expected_type) in expected {
        assert_eq!(GgufValueType::from_u32(raw), Some(expected_type));
    }
}

#[test]
fn value_type_from_u32_invalid() {
    assert_eq!(GgufValueType::from_u32(13), None);
    assert_eq!(GgufValueType::from_u32(100), None);
    assert_eq!(GgufValueType::from_u32(u32::MAX), None);
}

#[test]
fn value_type_equality() {
    assert_eq!(GgufValueType::Float32, GgufValueType::Float32);
    assert_ne!(GgufValueType::Float32, GgufValueType::Float64);
}

#[test]
fn value_type_debug() {
    let vt = GgufValueType::String;
    let dbg = format!("{:?}", vt);
    assert!(dbg.contains("String"));
}

#[test]
fn value_type_clone_copy() {
    let original = GgufValueType::Array;
    let copied = original;
    assert_eq!(original, copied);
}

// ---------------------------------------------------------------------------
// GgufValue
// ---------------------------------------------------------------------------

#[test]
fn gguf_value_uint8() {
    let v = GgufValue::Uint8(42);
    let dbg = format!("{:?}", v);
    assert!(dbg.contains("42"));
}

#[test]
fn gguf_value_string() {
    let v = GgufValue::String("hello".to_string());
    let dbg = format!("{:?}", v);
    assert!(dbg.contains("hello"));
}

#[test]
fn gguf_value_bool_true() {
    let v = GgufValue::Bool(true);
    let dbg = format!("{:?}", v);
    assert!(dbg.contains("true"));
}

#[test]
fn gguf_value_float32() {
    let v = GgufValue::Float32(3.14);
    let dbg = format!("{:?}", v);
    assert!(dbg.contains("3.14"));
}

#[test]
fn gguf_value_array_empty() {
    let v = GgufValue::Array(GgufValueType::Uint32, vec![]);
    let dbg = format!("{:?}", v);
    assert!(dbg.contains("Array"));
}

#[test]
fn gguf_value_array_with_elements() {
    let elements = vec![GgufValue::Uint32(1), GgufValue::Uint32(2), GgufValue::Uint32(3)];
    let v = GgufValue::Array(GgufValueType::Uint32, elements);
    let dbg = format!("{:?}", v);
    assert!(dbg.contains("Array"));
}

#[test]
fn gguf_value_clone() {
    let original = GgufValue::Int64(-12345);
    let cloned = original.clone();
    let dbg = format!("{:?}", cloned);
    assert!(dbg.contains("-12345"));
}

// ---------------------------------------------------------------------------
// GgufMetadataKv
// ---------------------------------------------------------------------------

#[test]
fn metadata_kv_construction() {
    let kv = GgufMetadataKv {
        key: "general.architecture".to_string(),
        value: GgufValue::String("llama".to_string()),
    };
    assert_eq!(kv.key, "general.architecture");
}

#[test]
fn metadata_kv_debug() {
    let kv = GgufMetadataKv {
        key: "general.name".to_string(),
        value: GgufValue::String("test model".to_string()),
    };
    let dbg = format!("{:?}", kv);
    assert!(dbg.contains("general.name"));
}

#[test]
fn metadata_kv_clone() {
    let kv =
        GgufMetadataKv { key: "llama.context_length".to_string(), value: GgufValue::Uint32(4096) };
    let cloned = kv.clone();
    assert_eq!(cloned.key, "llama.context_length");
}

// ---------------------------------------------------------------------------
// TensorInfo
// ---------------------------------------------------------------------------

#[test]
fn tensor_info_construction() {
    let info = TensorInfo {
        name: "token_embd.weight".to_string(),
        n_dims: 2,
        dims: vec![100352, 5120],
        dtype: 1, // F16 or similar
        offset: 0,
    };
    assert_eq!(info.name, "token_embd.weight");
    assert_eq!(info.n_dims, 2);
    assert_eq!(info.dims.len(), 2);
}

#[test]
fn tensor_info_debug() {
    let info = TensorInfo {
        name: "blk.0.attn_q.weight".to_string(),
        n_dims: 2,
        dims: vec![5120, 5120],
        dtype: 0,
        offset: 1024,
    };
    let dbg = format!("{:?}", info);
    assert!(dbg.contains("blk.0.attn_q.weight"));
}

#[test]
fn tensor_info_clone() {
    let info = TensorInfo {
        name: "output.weight".to_string(),
        n_dims: 2,
        dims: vec![100352, 5120],
        dtype: 2,
        offset: 999999,
    };
    let cloned = info.clone();
    assert_eq!(cloned.name, info.name);
    assert_eq!(cloned.dims, info.dims);
    assert_eq!(cloned.offset, info.offset);
}

#[test]
fn tensor_info_1d() {
    let info = TensorInfo {
        name: "blk.0.attn_norm.weight".to_string(),
        n_dims: 1,
        dims: vec![5120],
        dtype: 0,
        offset: 0,
    };
    assert_eq!(info.n_dims, 1);
    assert_eq!(info.dims.len(), 1);
}

// ---------------------------------------------------------------------------
// GgufFileInfo
// ---------------------------------------------------------------------------

#[test]
fn file_info_debug() {
    let info = GgufFileInfo { version: 3, tensor_count: 291, metadata_count: 24, alignment: 32 };
    let dbg = format!("{:?}", info);
    assert!(dbg.contains("291"));
    assert!(dbg.contains("version"));
}

#[test]
fn file_info_clone() {
    let info = GgufFileInfo { version: 2, tensor_count: 100, metadata_count: 10, alignment: 32 };
    let cloned = info.clone();
    assert_eq!(cloned.version, info.version);
    assert_eq!(cloned.tensor_count, info.tensor_count);
}
