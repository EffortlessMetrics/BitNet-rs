//! Edge-case tests for bitnet-gguf: magic validation, version reading,
//! header parsing, GgufValueType enum, GgufValue variants, and file info.

use bitnet_gguf::{
    GGUF_MAGIC, GGUF_VERSION_MAX, GGUF_VERSION_MIN, GgufFileInfo, GgufMetadataKv, GgufValue,
    GgufValueType, TensorInfo, check_magic, parse_header, read_version,
};

// ===========================================================================
// Constants
// ===========================================================================

#[test]
fn gguf_magic_bytes() {
    assert_eq!(&GGUF_MAGIC, b"GGUF");
}

#[test]
fn gguf_version_range() {
    assert!(GGUF_VERSION_MIN <= GGUF_VERSION_MAX);
    assert_eq!(GGUF_VERSION_MIN, 2);
    assert_eq!(GGUF_VERSION_MAX, 3);
}

// ===========================================================================
// check_magic
// ===========================================================================

#[test]
fn check_magic_valid() {
    assert!(check_magic(b"GGUF"));
}

#[test]
fn check_magic_valid_with_extra() {
    assert!(check_magic(b"GGUF\x03\x00\x00\x00"));
}

#[test]
fn check_magic_invalid() {
    assert!(!check_magic(b"LLAM"));
}

#[test]
fn check_magic_too_short() {
    assert!(!check_magic(b"GGU"));
}

#[test]
fn check_magic_empty() {
    assert!(!check_magic(b""));
}

// ===========================================================================
// read_version
// ===========================================================================

#[test]
fn read_version_v2() {
    let buf = *b"GGUF\x02\x00\x00\x00";
    assert_eq!(read_version(&buf), Some(2));
}

#[test]
fn read_version_v3() {
    let buf = *b"GGUF\x03\x00\x00\x00";
    assert_eq!(read_version(&buf), Some(3));
}

#[test]
fn read_version_too_short() {
    assert_eq!(read_version(b"GGUF\x02"), None);
}

#[test]
fn read_version_bad_magic() {
    assert_eq!(read_version(b"LLAM\x02\x00\x00\x00"), None);
}

// ===========================================================================
// parse_header
// ===========================================================================

fn gguf_v3_header(tensors: u64, metadata: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&tensors.to_le_bytes());
    buf.extend_from_slice(&metadata.to_le_bytes());
    buf.extend_from_slice(&32u32.to_le_bytes()); // alignment
    buf
}

fn gguf_v2_header(tensors: u64, metadata: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&tensors.to_le_bytes());
    buf.extend_from_slice(&metadata.to_le_bytes());
    buf
}

#[test]
fn parse_header_v3() {
    let buf = gguf_v3_header(10, 5);
    let info = parse_header(&buf).unwrap();
    assert_eq!(info.version, 3);
    assert_eq!(info.tensor_count, 10);
    assert_eq!(info.metadata_count, 5);
    assert_eq!(info.alignment, 32);
}

#[test]
fn parse_header_v2() {
    let buf = gguf_v2_header(8, 3);
    let info = parse_header(&buf).unwrap();
    assert_eq!(info.version, 2);
    assert_eq!(info.tensor_count, 8);
    assert_eq!(info.metadata_count, 3);
    assert_eq!(info.alignment, 32); // default
}

#[test]
fn parse_header_too_short() {
    assert!(parse_header(b"GGUF\x03\x00").is_err());
}

#[test]
fn parse_header_bad_magic() {
    let mut buf = gguf_v3_header(0, 0);
    buf[0] = b'X';
    assert!(parse_header(&buf).is_err());
}

#[test]
fn parse_header_unsupported_version() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&99u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    assert!(parse_header(&buf).is_err());
}

#[test]
fn parse_header_zero_counts() {
    let buf = gguf_v3_header(0, 0);
    let info = parse_header(&buf).unwrap();
    assert_eq!(info.tensor_count, 0);
    assert_eq!(info.metadata_count, 0);
}

// ===========================================================================
// GgufValueType
// ===========================================================================

#[test]
fn value_type_from_u32_all_valid() {
    for i in 0..=12 {
        assert!(GgufValueType::from_u32(i).is_some(), "expected Some for {i}");
    }
}

#[test]
fn value_type_from_u32_invalid() {
    assert!(GgufValueType::from_u32(13).is_none());
    assert!(GgufValueType::from_u32(255).is_none());
}

#[test]
fn value_type_specific() {
    assert_eq!(GgufValueType::from_u32(0), Some(GgufValueType::Uint8));
    assert_eq!(GgufValueType::from_u32(6), Some(GgufValueType::Float32));
    assert_eq!(GgufValueType::from_u32(8), Some(GgufValueType::String));
    assert_eq!(GgufValueType::from_u32(12), Some(GgufValueType::Float64));
}

// ===========================================================================
// GgufValue
// ===========================================================================

#[test]
fn gguf_value_uint32_debug() {
    let v = GgufValue::Uint32(42);
    assert!(format!("{v:?}").contains("42"));
}

#[test]
fn gguf_value_string_debug() {
    let v = GgufValue::String("hello".into());
    assert!(format!("{v:?}").contains("hello"));
}

#[test]
fn gguf_value_array_debug() {
    let v =
        GgufValue::Array(GgufValueType::Uint32, vec![GgufValue::Uint32(1), GgufValue::Uint32(2)]);
    let dbg = format!("{v:?}");
    assert!(dbg.contains("Array"));
}

#[test]
fn gguf_value_bool() {
    let v = GgufValue::Bool(true);
    let cloned = v.clone();
    assert!(format!("{cloned:?}").contains("true"));
}

#[test]
fn gguf_value_float64() {
    let v = GgufValue::Float64(3.14);
    let dbg = format!("{v:?}");
    assert!(dbg.contains("3.14"));
}

// ===========================================================================
// GgufMetadataKv
// ===========================================================================

#[test]
fn metadata_kv_construction() {
    let kv = GgufMetadataKv {
        key: "general.architecture".into(),
        value: GgufValue::String("llama".into()),
    };
    assert_eq!(kv.key, "general.architecture");
}

// ===========================================================================
// TensorInfo
// ===========================================================================

#[test]
fn tensor_info_construction() {
    let ti = TensorInfo {
        name: "blk.0.attn_q.weight".into(),
        n_dims: 2,
        dims: vec![4096, 4096],
        dtype: 1, // F16
        offset: 0,
    };
    assert_eq!(ti.n_dims, 2);
    assert_eq!(ti.dims.len(), 2);
}

#[test]
fn tensor_info_clone() {
    let ti = TensorInfo {
        name: "embedding".into(),
        n_dims: 2,
        dims: vec![32000, 4096],
        dtype: 0,
        offset: 1024,
    };
    let cloned = ti.clone();
    assert_eq!(cloned.name, ti.name);
    assert_eq!(cloned.offset, ti.offset);
}

// ===========================================================================
// GgufFileInfo
// ===========================================================================

#[test]
fn file_info_debug() {
    let info = GgufFileInfo { version: 3, tensor_count: 100, metadata_count: 20, alignment: 32 };
    let dbg = format!("{info:?}");
    assert!(dbg.contains("100"));
}
