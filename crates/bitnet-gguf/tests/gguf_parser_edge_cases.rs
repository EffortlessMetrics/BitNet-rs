//! Edge case and boundary tests for GGUF parsing primitives.
//!
//! Tests focus on malformed input handling, boundary values, and type system coverage.

use bitnet_gguf::{
    GGUF_MAGIC, GGUF_VERSION_MAX, GGUF_VERSION_MIN, GgufFileInfo, GgufMetadataKv, GgufValue,
    GgufValueType, TensorInfo, check_magic, parse_header, read_version,
};

// --- Magic constant tests ---

#[test]
fn magic_constant_is_gguf_bytes() {
    assert_eq!(&GGUF_MAGIC, b"GGUF");
}

#[test]
fn check_magic_valid() {
    assert!(check_magic(b"GGUF"));
    assert!(check_magic(b"GGUF\x03\x00\x00\x00"));
    assert!(check_magic(b"GGUFextra bytes after"));
}

#[test]
fn check_magic_too_short() {
    assert!(!check_magic(b""));
    assert!(!check_magic(b"G"));
    assert!(!check_magic(b"GG"));
    assert!(!check_magic(b"GGU"));
}

#[test]
fn check_magic_wrong_bytes() {
    assert!(!check_magic(b"GGML"));
    assert!(!check_magic(b"gguf"));
    assert!(!check_magic(b"\x00\x00\x00\x00"));
    assert!(!check_magic(b"FUGG"));
}

// --- Version reading tests ---

#[test]
fn version_constants() {
    assert_eq!(GGUF_VERSION_MIN, 2);
    assert_eq!(GGUF_VERSION_MAX, 3);
}

#[test]
fn read_version_v2() {
    let mut data = Vec::from(b"GGUF" as &[u8]);
    data.extend_from_slice(&2u32.to_le_bytes());
    assert_eq!(read_version(&data), Some(2));
}

#[test]
fn read_version_v3() {
    let mut data = Vec::from(b"GGUF" as &[u8]);
    data.extend_from_slice(&3u32.to_le_bytes());
    assert_eq!(read_version(&data), Some(3));
}

#[test]
fn read_version_too_short() {
    assert_eq!(read_version(b""), None);
    assert_eq!(read_version(b"GGUF"), None);
    assert_eq!(read_version(b"GGUF\x03"), None);
    assert_eq!(read_version(b"GGUF\x03\x00"), None);
    assert_eq!(read_version(b"GGUF\x03\x00\x00"), None);
}

#[test]
fn read_version_bad_magic() {
    let mut data = Vec::from(b"GGML" as &[u8]);
    data.extend_from_slice(&3u32.to_le_bytes());
    assert_eq!(read_version(&data), None);
}

#[test]
fn read_version_future_version() {
    let mut data = Vec::from(b"GGUF" as &[u8]);
    data.extend_from_slice(&99u32.to_le_bytes());
    // read_version just reads the bytes, doesn't validate range
    assert_eq!(read_version(&data), Some(99));
}

// --- parse_header tests ---

fn make_v3_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&tensor_count.to_le_bytes());
    data.extend_from_slice(&metadata_count.to_le_bytes());
    data
}

#[test]
fn parse_header_v3_zero_counts() {
    let data = make_v3_header(0, 0);
    let info = parse_header(&data).unwrap();
    assert_eq!(info.version, 3);
    assert_eq!(info.tensor_count, 0);
    assert_eq!(info.metadata_count, 0);
}

#[test]
fn parse_header_v3_large_counts() {
    let data = make_v3_header(1000, 500);
    let info = parse_header(&data).unwrap();
    assert_eq!(info.tensor_count, 1000);
    assert_eq!(info.metadata_count, 500);
}

#[test]
fn parse_header_v2() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&10u64.to_le_bytes());
    data.extend_from_slice(&5u64.to_le_bytes());
    let info = parse_header(&data).unwrap();
    assert_eq!(info.version, 2);
    assert_eq!(info.tensor_count, 10);
    assert_eq!(info.metadata_count, 5);
}

#[test]
fn parse_header_too_short() {
    assert!(parse_header(b"").is_err());
    assert!(parse_header(b"GGUF").is_err());
    assert!(parse_header(b"GGUF\x03\x00\x00\x00").is_err());
}

#[test]
fn parse_header_bad_magic() {
    let mut data = Vec::from(b"GGML" as &[u8]);
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    assert!(parse_header(&data).is_err());
}

#[test]
fn parse_header_version_1_rejected() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    assert!(parse_header(&data).is_err());
}

#[test]
fn parse_header_version_4_rejected() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    assert!(parse_header(&data).is_err());
}

#[test]
fn parse_header_alignment_defaults_to_32() {
    let data = make_v3_header(0, 0);
    let info = parse_header(&data).unwrap();
    assert_eq!(info.alignment, 32);
}

// --- GgufValueType tests ---

#[test]
fn value_type_from_u32_all_valid() {
    let valid_pairs = [
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
    for (disc, expected) in &valid_pairs {
        assert_eq!(
            GgufValueType::from_u32(*disc),
            Some(*expected),
            "from_u32({disc}) should return {expected:?}"
        );
    }
}

#[test]
fn value_type_from_u32_invalid() {
    assert_eq!(GgufValueType::from_u32(13), None);
    assert_eq!(GgufValueType::from_u32(100), None);
    assert_eq!(GgufValueType::from_u32(u32::MAX), None);
}

#[test]
fn value_type_debug_is_non_empty() {
    let types =
        [GgufValueType::Uint8, GgufValueType::Float32, GgufValueType::String, GgufValueType::Array];
    for t in &types {
        assert!(!format!("{t:?}").is_empty());
    }
}

#[test]
fn value_type_copy_semantics() {
    let a = GgufValueType::Float32;
    let b = a;
    assert_eq!(a, b);
}

// --- GgufValue construction tests ---

#[test]
fn gguf_value_numeric_types() {
    let _ = GgufValue::Uint8(0);
    let _ = GgufValue::Uint8(255);
    let _ = GgufValue::Int8(-128);
    let _ = GgufValue::Int8(127);
    let _ = GgufValue::Uint16(0);
    let _ = GgufValue::Uint16(65535);
    let _ = GgufValue::Int16(-32768);
    let _ = GgufValue::Uint32(0);
    let _ = GgufValue::Uint32(u32::MAX);
    let _ = GgufValue::Int32(i32::MIN);
    let _ = GgufValue::Float32(std::f32::NAN);
    let _ = GgufValue::Float32(std::f32::INFINITY);
    let _ = GgufValue::Uint64(0);
    let _ = GgufValue::Uint64(u64::MAX);
    let _ = GgufValue::Int64(i64::MIN);
    let _ = GgufValue::Float64(std::f64::NAN);
    let _ = GgufValue::Float64(std::f64::INFINITY);
    let _ = GgufValue::Bool(true);
    let _ = GgufValue::Bool(false);
}

#[test]
fn gguf_value_string() {
    let v = GgufValue::String("hello".to_string());
    match &v {
        GgufValue::String(s) => assert_eq!(s, "hello"),
        _ => panic!("Expected String variant"),
    }
}

#[test]
fn gguf_value_empty_string() {
    let v = GgufValue::String(String::new());
    match &v {
        GgufValue::String(s) => assert!(s.is_empty()),
        _ => panic!("Expected String variant"),
    }
}

#[test]
fn gguf_value_unicode_string() {
    let v = GgufValue::String("🎉 unicode ñ".to_string());
    match &v {
        GgufValue::String(s) => assert!(s.contains("🎉")),
        _ => panic!("Expected String variant"),
    }
}

#[test]
fn gguf_value_array_empty() {
    let v = GgufValue::Array(GgufValueType::Uint32, vec![]);
    match &v {
        GgufValue::Array(_, items) => assert!(items.is_empty()),
        _ => panic!("Expected Array variant"),
    }
}

#[test]
fn gguf_value_array_nested() {
    let inner = GgufValue::Array(GgufValueType::Uint8, vec![GgufValue::Uint8(1)]);
    let outer = GgufValue::Array(GgufValueType::Array, vec![inner]);
    match &outer {
        GgufValue::Array(_, items) => assert_eq!(items.len(), 1),
        _ => panic!("Expected Array variant"),
    }
}

#[test]
fn gguf_value_clone() {
    let v = GgufValue::Float32(3.14);
    let v2 = v.clone();
    match (&v, &v2) {
        (GgufValue::Float32(a), GgufValue::Float32(b)) => assert_eq!(a, b),
        _ => panic!("Clone should preserve variant"),
    }
}

// --- GgufMetadataKv tests ---

#[test]
fn metadata_kv_construction() {
    let kv = GgufMetadataKv {
        key: "model.name".to_string(),
        value: GgufValue::String("test".to_string()),
    };
    assert_eq!(kv.key, "model.name");
}

#[test]
fn metadata_kv_empty_key() {
    let kv = GgufMetadataKv { key: String::new(), value: GgufValue::Bool(true) };
    assert!(kv.key.is_empty());
}

#[test]
fn metadata_kv_debug_non_empty() {
    let kv = GgufMetadataKv { key: "test".to_string(), value: GgufValue::Uint32(42) };
    assert!(!format!("{kv:?}").is_empty());
}

// --- TensorInfo tests ---

#[test]
fn tensor_info_construction() {
    let info = TensorInfo {
        name: "weight.0".to_string(),
        n_dims: 2,
        dims: vec![1024, 512],
        dtype: 0,
        offset: 0,
    };
    assert_eq!(info.name, "weight.0");
    assert_eq!(info.n_dims, 2);
    assert_eq!(info.dims.len(), 2);
}

#[test]
fn tensor_info_zero_dims() {
    let info =
        TensorInfo { name: "scalar".to_string(), n_dims: 0, dims: vec![], dtype: 0, offset: 0 };
    assert_eq!(info.n_dims, 0);
    assert!(info.dims.is_empty());
}

#[test]
fn tensor_info_large_dims() {
    let info = TensorInfo {
        name: "embedding".to_string(),
        n_dims: 2,
        dims: vec![100352, 5120],
        dtype: 1,
        offset: 1024,
    };
    assert_eq!(info.dims[0], 100352);
    assert_eq!(info.dims[1], 5120);
}

#[test]
fn tensor_info_max_offset() {
    let info = TensorInfo {
        name: "last_tensor".to_string(),
        n_dims: 1,
        dims: vec![1],
        dtype: 0,
        offset: u64::MAX,
    };
    assert_eq!(info.offset, u64::MAX);
}

#[test]
fn tensor_info_clone() {
    let info = TensorInfo {
        name: "test".to_string(),
        n_dims: 3,
        dims: vec![2, 3, 4],
        dtype: 6,
        offset: 100,
    };
    let cloned = info.clone();
    assert_eq!(info.name, cloned.name);
    assert_eq!(info.dims, cloned.dims);
}

// --- GgufFileInfo tests ---

#[test]
fn file_info_debug_contains_version() {
    let info = GgufFileInfo { version: 3, tensor_count: 10, metadata_count: 5, alignment: 32 };
    let debug = format!("{info:?}");
    assert!(debug.contains("3"), "Debug should contain version");
}

#[test]
fn file_info_clone() {
    let info = GgufFileInfo { version: 3, tensor_count: 10, metadata_count: 5, alignment: 32 };
    let cloned = info.clone();
    assert_eq!(info.version, cloned.version);
    assert_eq!(info.tensor_count, cloned.tensor_count);
}
