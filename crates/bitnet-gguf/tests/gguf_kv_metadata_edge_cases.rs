//! Edge-case tests for bitnet-gguf: KV metadata, value types, and file open.
//!
//! Complements existing test files by covering gaps in:
//! - `read_kv_pairs` with array values, all scalar types, invalid types, truncation
//! - `open()` and `read_header_blocking` with empty/corrupt files
//! - `kv::parse_header` vs `lib::parse_header` version-acceptance differences
//! - GgufError IO variant, serde round-trips for complex values

use std::io::Write;
use std::path::Path;

use bitnet_gguf::kv::{GgufError, GgufHeader, GgufKv, GgufValue as KvGgufValue};
use bitnet_gguf::{
    GgufFileInfo, GgufMetadataKv, GgufValue, GgufValueType, TensorInfo, check_magic, open,
    parse_header, read_version,
};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build a lib-level GGUF v2 header (24 bytes).
fn lib_v2_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(24);
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&tensor_count.to_le_bytes());
    buf.extend_from_slice(&metadata_count.to_le_bytes());
    buf
}

/// Build a lib-level GGUF v3 header (28 bytes) with alignment.
fn lib_v3_header(tensor_count: u64, metadata_count: u64, alignment: u32) -> Vec<u8> {
    let mut buf = lib_v2_header(tensor_count, metadata_count);
    buf[4..8].copy_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&alignment.to_le_bytes());
    buf
}

/// Build a kv-module GGUF header (24 bytes).
fn kv_header(version: u32, n_tensors: u64, n_kv: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(24);
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&version.to_le_bytes());
    buf.extend_from_slice(&n_tensors.to_le_bytes());
    buf.extend_from_slice(&n_kv.to_le_bytes());
    buf
}

/// Write a synthetic GGUF file with typed KV pairs.
fn write_gguf_kv(f: &mut impl Write, version: u32, n_tensors: u64, kvs: &[(&str, u32, &[u8])]) {
    f.write_all(b"GGUF").unwrap();
    f.write_all(&version.to_le_bytes()).unwrap();
    f.write_all(&n_tensors.to_le_bytes()).unwrap();
    f.write_all(&(kvs.len() as u64).to_le_bytes()).unwrap();
    for &(key, ty, value_bytes) in kvs {
        f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        f.write_all(key.as_bytes()).unwrap();
        f.write_all(&ty.to_le_bytes()).unwrap();
        f.write_all(value_bytes).unwrap();
    }
}

/// Encode a string value as [u64 len][bytes] for GGUF type 8.
fn encode_string_value(s: &str) -> Vec<u8> {
    let mut buf = (s.len() as u64).to_le_bytes().to_vec();
    buf.extend_from_slice(s.as_bytes());
    buf
}

/// Encode a numeric array as [u32 elem_type][u64 len][elem_bytes...] for GGUF type 9.
fn encode_u32_array(values: &[u32]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&4u32.to_le_bytes()); // elem type = U32
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Encode a string array as [u32 elem_type=8][u64 len][string entries...].
fn encode_string_array(values: &[&str]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&8u32.to_le_bytes()); // elem type = String
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for s in values {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
    buf
}

// ═══════════════════════════════════════════════════════════════════════════
// read_kv_pairs — all scalar types
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn read_kv_pairs_u8() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("u8.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_gguf_kv(&mut f, 3, 0, &[("val", 0, &[0xAB])]);
    drop(f);
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].value, KvGgufValue::U8(0xAB));
}

#[test]
fn read_kv_pairs_u16() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("u16.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_gguf_kv(&mut f, 3, 0, &[("val", 2, &1000u16.to_le_bytes())]);
    drop(f);
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].value, KvGgufValue::U16(1000));
}

#[test]
fn read_kv_pairs_i16() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i16.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_gguf_kv(&mut f, 3, 0, &[("val", 3, &(-500i16).to_le_bytes())]);
    drop(f);
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].value, KvGgufValue::I16(-500));
}

#[test]
fn read_kv_pairs_i32() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i32.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_gguf_kv(&mut f, 3, 0, &[("val", 5, &(-42i32).to_le_bytes())]);
    drop(f);
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].value, KvGgufValue::I32(-42));
}

#[test]
fn read_kv_pairs_i64() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_gguf_kv(&mut f, 3, 0, &[("val", 11, &i64::MIN.to_le_bytes())]);
    drop(f);
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].value, KvGgufValue::I64(i64::MIN));
}

#[test]
fn read_kv_pairs_f64() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_gguf_kv(&mut f, 3, 0, &[("val", 12, &std::f64::consts::E.to_le_bytes())]);
    drop(f);
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    if let KvGgufValue::F64(x) = kvs[0].value {
        assert!((x - std::f64::consts::E).abs() < 1e-15);
    } else {
        panic!("expected F64");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// read_kv_pairs — array values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn read_kv_pairs_u32_array() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("arr_u32.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    let arr = encode_u32_array(&[10, 20, 30]);
    write_gguf_kv(&mut f, 3, 0, &[("arr", 9, &arr)]);
    drop(f);
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    if let KvGgufValue::Array(items) = &kvs[0].value {
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], KvGgufValue::U32(10));
        assert_eq!(items[1], KvGgufValue::U32(20));
        assert_eq!(items[2], KvGgufValue::U32(30));
    } else {
        panic!("expected Array");
    }
}

#[test]
fn read_kv_pairs_string_array() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("arr_str.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    let arr = encode_string_array(&["alpha", "beta", "gamma"]);
    write_gguf_kv(&mut f, 3, 0, &[("tags", 9, &arr)]);
    drop(f);
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    if let KvGgufValue::Array(items) = &kvs[0].value {
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], KvGgufValue::String("alpha".to_string()));
        assert_eq!(items[2], KvGgufValue::String("gamma".to_string()));
    } else {
        panic!("expected Array");
    }
}

#[test]
fn read_kv_pairs_empty_array() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("arr_empty.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    let arr = encode_u32_array(&[]);
    write_gguf_kv(&mut f, 3, 0, &[("empty", 9, &arr)]);
    drop(f);
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    if let KvGgufValue::Array(items) = &kvs[0].value {
        assert!(items.is_empty());
    } else {
        panic!("expected Array");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// read_kv_pairs — error & edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn read_kv_pairs_invalid_kv_type() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_type.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    // Use type 99 which is invalid
    write_gguf_kv(&mut f, 3, 0, &[("bad", 99, &[0u8; 4])]);
    drop(f);
    let result = bitnet_gguf::kv::read_kv_pairs(&path, None);
    assert!(result.is_err());
}

#[test]
fn read_kv_pairs_limit_zero() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("limit0.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_gguf_kv(&mut f, 3, 0, &[("a", 4, &1u32.to_le_bytes()), ("b", 4, &2u32.to_le_bytes())]);
    drop(f);
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, Some(0)).unwrap();
    assert!(kvs.is_empty());
}

#[test]
fn read_kv_pairs_limit_exceeds_n_kv() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("overlimit.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_gguf_kv(&mut f, 3, 0, &[("only", 4, &7u32.to_le_bytes())]);
    drop(f);
    // Limit 100 but only 1 KV in file — should return 1
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, Some(100)).unwrap();
    assert_eq!(kvs.len(), 1);
}

#[test]
fn read_kv_pairs_truncated_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trunc_val.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    // Write header saying 1 KV, but provide incomplete value data
    f.write_all(b"GGUF").unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    f.write_all(&1u64.to_le_bytes()).unwrap(); // n_kv=1
    // Write key
    let key = "trunc";
    f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    f.write_all(key.as_bytes()).unwrap();
    // Write type = U32 (4)
    f.write_all(&4u32.to_le_bytes()).unwrap();
    // Write only 2 bytes instead of 4
    f.write_all(&[0x01, 0x02]).unwrap();
    drop(f);
    let result = bitnet_gguf::kv::read_kv_pairs(&path, None);
    assert!(result.is_err());
}

#[test]
fn read_kv_pairs_unicode_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unicode.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_gguf_kv(&mut f, 3, 0, &[("模型.名前", 7, &[1u8])]);
    drop(f);
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].key, "模型.名前");
    assert_eq!(kvs[0].value, KvGgufValue::Bool(true));
}

#[test]
fn read_kv_pairs_unicode_string_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unicode_val.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    let val = encode_string_value("こんにちは🌍");
    write_gguf_kv(&mut f, 3, 0, &[("greeting", 8, &val)]);
    drop(f);
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].value, KvGgufValue::String("こんにちは🌍".to_string()));
}

#[test]
fn read_kv_pairs_multiple_types_in_one_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    let str_val = encode_string_value("model");
    let arr_val = encode_u32_array(&[1, 2]);
    write_gguf_kv(
        &mut f,
        3,
        0,
        &[
            ("name", 8, &str_val),
            ("count", 4, &42u32.to_le_bytes()),
            ("flag", 7, &[1u8]),
            ("scores", 9, &arr_val),
            ("ratio", 6, &1.5f32.to_le_bytes()),
        ],
    );
    drop(f);
    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs.len(), 5);
    assert_eq!(kvs[0].key, "name");
    assert_eq!(kvs[0].value, KvGgufValue::String("model".to_string()));
    assert_eq!(kvs[1].value, KvGgufValue::U32(42));
    assert_eq!(kvs[2].value, KvGgufValue::Bool(true));
    if let KvGgufValue::Array(items) = &kvs[3].value {
        assert_eq!(items.len(), 2);
    } else {
        panic!("expected Array");
    }
    if let KvGgufValue::F32(x) = kvs[4].value {
        assert!((x - 1.5).abs() < 1e-7);
    } else {
        panic!("expected F32");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// kv::parse_header vs lib::parse_header — version acceptance
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn kv_parse_header_accepts_v1() {
    // kv::parse_header accepts versions 1–3
    let buf = kv_header(1, 5, 2);
    let h = bitnet_gguf::kv::parse_header(&buf).unwrap();
    assert_eq!(h.version, 1);
    assert_eq!(h.n_tensors, 5);
    assert_eq!(h.n_kv, 2);
}

#[test]
fn lib_parse_header_rejects_v1() {
    // lib::parse_header only accepts 2–3
    let mut buf = lib_v2_header(0, 0);
    buf[4..8].copy_from_slice(&1u32.to_le_bytes());
    assert!(parse_header(&buf).is_err());
}

#[test]
fn both_parse_headers_accept_v2() {
    let buf = kv_header(2, 10, 3);
    assert!(bitnet_gguf::kv::parse_header(&buf).is_ok());
    assert!(parse_header(&buf).is_ok());
}

#[test]
fn both_parse_headers_accept_v3() {
    let buf = kv_header(3, 10, 3);
    assert!(bitnet_gguf::kv::parse_header(&buf).is_ok());
    // lib::parse_header needs 24 bytes minimum, same buffer works
    assert!(parse_header(&buf).is_ok());
}

#[test]
fn both_parse_headers_reject_v4() {
    let buf = kv_header(4, 0, 0);
    assert!(bitnet_gguf::kv::parse_header(&buf).is_err());
    assert!(parse_header(&buf).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// open() edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn open_empty_file_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.gguf");
    std::fs::File::create(&path).unwrap();
    let result = open(&path);
    assert!(result.is_err());
}

#[test]
fn open_truncated_header_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trunc.gguf");
    // Write only 16 bytes (need 24 minimum)
    std::fs::write(&path, b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
    let result = open(&path);
    assert!(result.is_err());
}

#[test]
fn open_nonexistent_returns_error() {
    let result = open(Path::new("C:\\nonexistent\\surely\\missing.gguf"));
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("cannot open"));
}

#[test]
fn open_valid_v2_temp_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v2.gguf");
    std::fs::write(&path, lib_v2_header(7, 3)).unwrap();
    let info = open(&path).unwrap();
    assert_eq!(info.version, 2);
    assert_eq!(info.tensor_count, 7);
    assert_eq!(info.metadata_count, 3);
    assert_eq!(info.alignment, 32);
}

#[test]
fn open_valid_v3_temp_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v3.gguf");
    std::fs::write(&path, lib_v3_header(50, 10, 64)).unwrap();
    let info = open(&path).unwrap();
    assert_eq!(info.version, 3);
    assert_eq!(info.tensor_count, 50);
    assert_eq!(info.metadata_count, 10);
    assert_eq!(info.alignment, 64);
}

// ═══════════════════════════════════════════════════════════════════════════
// read_header_blocking edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn read_header_blocking_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.gguf");
    std::fs::File::create(&path).unwrap();
    let result = bitnet_gguf::kv::read_header_blocking(&path);
    assert!(result.is_err());
}

#[test]
fn read_header_blocking_bad_magic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_magic.gguf");
    let mut buf = kv_header(3, 0, 0);
    buf[0..4].copy_from_slice(b"NOPE");
    std::fs::write(&path, &buf).unwrap();
    let result = bitnet_gguf::kv::read_header_blocking(&path);
    assert!(result.is_err());
    match result.unwrap_err() {
        GgufError::BadMagic(m) => assert_eq!(&m, b"NOPE"),
        e => panic!("expected BadMagic, got {e:?}"),
    }
}

#[test]
fn read_header_blocking_v1_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v1.gguf");
    std::fs::write(&path, kv_header(1, 0, 0)).unwrap();
    let h = bitnet_gguf::kv::read_header_blocking(&path).unwrap();
    assert_eq!(h.version, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// GgufError variants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gguf_error_io_variant() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file gone");
    let gguf_err: GgufError = io_err.into();
    let msg = gguf_err.to_string();
    assert!(msg.contains("file gone"));
}

#[test]
fn gguf_error_is_debug() {
    let errors: Vec<GgufError> = vec![
        GgufError::BadMagic([0, 0, 0, 0]),
        GgufError::UnsupportedVersion(99),
        GgufError::ShortHeader(10),
        GgufError::Malformed,
        GgufError::InvalidKvType(255),
        GgufError::StringTooLarge(u64::MAX),
    ];
    for e in &errors {
        // All variants must format without panicking
        let _ = format!("{e:?}");
        let _ = format!("{e}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GgufValueType — Hash trait
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn value_type_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    for i in 0..=12 {
        set.insert(GgufValueType::from_u32(i).unwrap());
    }
    assert_eq!(set.len(), 13, "all 13 value types should be distinct in a HashSet");
}

#[test]
fn value_type_from_u32_boundary() {
    // Last valid is 12, first invalid is 13
    assert!(GgufValueType::from_u32(12).is_some());
    assert!(GgufValueType::from_u32(13).is_none());
    // u32 boundaries
    assert!(GgufValueType::from_u32(0).is_some());
    assert!(GgufValueType::from_u32(u32::MAX).is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// lib::GgufValue — all variants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lib_gguf_value_all_variants() {
    let values: Vec<GgufValue> = vec![
        GgufValue::Uint8(u8::MAX),
        GgufValue::Int8(i8::MIN),
        GgufValue::Uint16(u16::MAX),
        GgufValue::Int16(i16::MIN),
        GgufValue::Uint32(u32::MAX),
        GgufValue::Int32(i32::MIN),
        GgufValue::Float32(f32::NEG_INFINITY),
        GgufValue::Bool(false),
        GgufValue::String(String::new()),
        GgufValue::Array(GgufValueType::Uint8, vec![]),
        GgufValue::Uint64(u64::MAX),
        GgufValue::Int64(i64::MIN),
        GgufValue::Float64(f64::NAN),
    ];
    // All should debug-format without panic
    for v in &values {
        let _ = format!("{v:?}");
    }
    assert_eq!(values.len(), 13);
}

#[test]
fn lib_gguf_value_deep_clone() {
    let original = GgufValue::Array(
        GgufValueType::String,
        vec![GgufValue::String("first".to_string()), GgufValue::String("second".to_string())],
    );
    let cloned = original.clone();
    if let (GgufValue::Array(_, orig), GgufValue::Array(_, cl)) = (&original, &cloned) {
        assert_eq!(orig.len(), cl.len());
    } else {
        panic!("clone should preserve Array variant");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// kv::GgufValue — serde round-trips for all variants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn kv_value_serde_numeric_roundtrip() {
    let cases: Vec<KvGgufValue> = vec![
        KvGgufValue::U8(255),
        KvGgufValue::I8(-1),
        KvGgufValue::U16(1000),
        KvGgufValue::I16(-500),
        KvGgufValue::U32(u32::MAX),
        KvGgufValue::I32(i32::MIN),
        KvGgufValue::F32(1.5),
        KvGgufValue::U64(u64::MAX),
        KvGgufValue::I64(i64::MIN),
        KvGgufValue::F64(2.718),
        KvGgufValue::Bool(true),
        KvGgufValue::Bool(false),
    ];
    for v in &cases {
        let json = serde_json::to_string(v).unwrap();
        let back: KvGgufValue = serde_json::from_str(&json).unwrap();
        assert_eq!(v, &back, "serde round-trip failed for {v:?}");
    }
}

#[test]
fn kv_value_serde_string_roundtrip() {
    let v = KvGgufValue::String("hello world".to_string());
    let json = serde_json::to_string(&v).unwrap();
    let back: KvGgufValue = serde_json::from_str(&json).unwrap();
    assert_eq!(v, back);
}

#[test]
fn kv_value_serde_array_roundtrip() {
    let v = KvGgufValue::Array(vec![KvGgufValue::U32(1), KvGgufValue::U32(2), KvGgufValue::U32(3)]);
    let json = serde_json::to_string(&v).unwrap();
    let back: KvGgufValue = serde_json::from_str(&json).unwrap();
    assert_eq!(v, back);
}

#[test]
fn kv_value_serde_nested_array_roundtrip() {
    let inner = KvGgufValue::Array(vec![KvGgufValue::Bool(true), KvGgufValue::Bool(false)]);
    let v = KvGgufValue::Array(vec![inner]);
    let json = serde_json::to_string(&v).unwrap();
    let back: KvGgufValue = serde_json::from_str(&json).unwrap();
    assert_eq!(v, back);
}

#[test]
fn kv_value_serde_empty_string() {
    let v = KvGgufValue::String(String::new());
    let json = serde_json::to_string(&v).unwrap();
    let back: KvGgufValue = serde_json::from_str(&json).unwrap();
    assert_eq!(v, back);
}

// ═══════════════════════════════════════════════════════════════════════════
// GgufKv serde
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gguf_kv_serde_roundtrip() {
    let kv = GgufKv {
        key: "general.architecture".to_string(),
        value: KvGgufValue::String("llama".to_string()),
    };
    let json = serde_json::to_string(&kv).unwrap();
    let back: GgufKv = serde_json::from_str(&json).unwrap();
    assert_eq!(back.key, "general.architecture");
    assert_eq!(back.value, KvGgufValue::String("llama".to_string()));
}

// ═══════════════════════════════════════════════════════════════════════════
// GgufHeader serde
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gguf_header_serde_roundtrip_all_versions() {
    for version in 1..=3 {
        let h = GgufHeader { version, n_tensors: 100, n_kv: 50 };
        let json = serde_json::to_string(&h).unwrap();
        let back: GgufHeader = serde_json::from_str(&json).unwrap();
        assert_eq!(h, back);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// lib::parse_header — v3 alignment edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parse_header_v3_alignment_zero_falls_back() {
    let data = lib_v3_header(1, 1, 0);
    let info = parse_header(&data).unwrap();
    // 0 is not a power of two, should fall back to 32
    assert_eq!(info.alignment, 32);
}

#[test]
fn parse_header_v3_alignment_max_power_of_two() {
    let data = lib_v3_header(1, 1, 1 << 30); // 1 GiB alignment
    let info = parse_header(&data).unwrap();
    assert_eq!(info.alignment, 1 << 30);
}

#[test]
fn parse_header_v3_alignment_not_power_of_two_falls_back() {
    for bad_align in [3, 5, 6, 7, 12, 48, 100] {
        let data = lib_v3_header(0, 0, bad_align);
        let info = parse_header(&data).unwrap();
        assert_eq!(info.alignment, 32, "alignment {bad_align} should fall back to 32");
    }
}

#[test]
fn parse_header_v3_alignment_powers_of_two_accepted() {
    for shift in 0..=20 {
        let align = 1u32 << shift;
        let data = lib_v3_header(0, 0, align);
        let info = parse_header(&data).unwrap();
        assert_eq!(info.alignment, align, "alignment {align} should be accepted");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// lib::check_magic — single byte off
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn check_magic_each_byte_corrupted() {
    for i in 0..4 {
        let mut data = *b"GGUF";
        data[i] ^= 0xFF;
        assert!(!check_magic(&data), "corrupting byte {i} should fail");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// lib::read_version — exact boundary
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn read_version_exactly_8_bytes() {
    let buf = *b"GGUF\x02\x00\x00\x00";
    assert_eq!(read_version(&buf), Some(2));
}

#[test]
fn read_version_7_bytes_returns_none() {
    assert_eq!(read_version(b"GGUF\x02\x00\x00"), None);
}

// ═══════════════════════════════════════════════════════════════════════════
// TensorInfo edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn tensor_info_4d() {
    let ti = TensorInfo {
        name: "blk.0.attn_output.weight".to_string(),
        n_dims: 4,
        dims: vec![2, 3, 4, 5],
        dtype: 0,
        offset: 0,
    };
    assert_eq!(ti.n_dims, 4);
    assert_eq!(ti.dims, vec![2, 3, 4, 5]);
}

#[test]
fn tensor_info_empty_name() {
    let ti = TensorInfo { name: String::new(), n_dims: 1, dims: vec![1], dtype: 0, offset: 0 };
    assert!(ti.name.is_empty());
}

#[test]
fn tensor_info_all_dtype_values() {
    // dtype is just a u32, any value should be storable
    for dtype in [0, 1, 2, 6, 10, 20, u32::MAX] {
        let ti = TensorInfo { name: "t".to_string(), n_dims: 0, dims: vec![], dtype, offset: 0 };
        assert_eq!(ti.dtype, dtype);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GgufMetadataKv edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn metadata_kv_with_array_value() {
    let kv = GgufMetadataKv {
        key: "tokenizer.tokens".to_string(),
        value: GgufValue::Array(
            GgufValueType::String,
            vec![GgufValue::String("hello".to_string()), GgufValue::String("world".to_string())],
        ),
    };
    assert_eq!(kv.key, "tokenizer.tokens");
    if let GgufValue::Array(ty, items) = &kv.value {
        assert_eq!(*ty, GgufValueType::String);
        assert_eq!(items.len(), 2);
    } else {
        panic!("expected Array");
    }
}

#[test]
fn metadata_kv_numeric_value_types() {
    let cases = vec![
        ("u8", GgufValue::Uint8(0)),
        ("i8", GgufValue::Int8(-1)),
        ("u16", GgufValue::Uint16(256)),
        ("i16", GgufValue::Int16(-256)),
        ("u32", GgufValue::Uint32(65536)),
        ("i32", GgufValue::Int32(-65536)),
        ("f32", GgufValue::Float32(0.0)),
        ("u64", GgufValue::Uint64(1)),
        ("i64", GgufValue::Int64(-1)),
        ("f64", GgufValue::Float64(0.0)),
        ("bool", GgufValue::Bool(true)),
    ];
    for (key, value) in cases {
        let kv = GgufMetadataKv { key: key.to_string(), value };
        assert_eq!(kv.key, key);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GgufFileInfo edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn file_info_zero_alignment_not_produced_by_parser() {
    // Parser should never produce alignment=0 — it falls back to 32
    let data = lib_v3_header(0, 0, 0);
    let info = parse_header(&data).unwrap();
    assert_ne!(info.alignment, 0);
}

#[test]
fn file_info_direct_construction() {
    let info =
        GgufFileInfo { version: 3, tensor_count: u64::MAX, metadata_count: u64::MAX, alignment: 1 };
    assert_eq!(info.tensor_count, u64::MAX);
    assert_eq!(info.metadata_count, u64::MAX);
    assert_eq!(info.alignment, 1);
}
