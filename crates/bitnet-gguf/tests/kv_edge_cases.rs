//! Edge-case tests for bitnet-gguf KV header parsing.
//!
//! Tests cover `parse_header`, `GgufError` variants, `GgufHeader` fields,
//! `GgufValue` enum, `GgufKv` struct, and `read_kv_pairs` with synthetic GGUF files.

use bitnet_gguf::kv::{GGUF_HEADER_LEN, GgufError, GgufHeader, GgufValue, parse_header};
use std::io::Write;

// ── Helper: build a minimal GGUF header buffer ──────────────────────────────

fn build_header(version: u32, n_tensors: u64, n_kv: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(GGUF_HEADER_LEN);
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&version.to_le_bytes());
    buf.extend_from_slice(&n_tensors.to_le_bytes());
    buf.extend_from_slice(&n_kv.to_le_bytes());
    buf
}

// ── parse_header: valid inputs ───────────────────────────────────────────────

#[test]
fn parse_header_v1() {
    let buf = build_header(1, 10, 5);
    let h = parse_header(&buf).unwrap();
    assert_eq!(h.version, 1);
    assert_eq!(h.n_tensors, 10);
    assert_eq!(h.n_kv, 5);
}

#[test]
fn parse_header_v2() {
    let buf = build_header(2, 100, 50);
    let h = parse_header(&buf).unwrap();
    assert_eq!(h.version, 2);
    assert_eq!(h.n_tensors, 100);
    assert_eq!(h.n_kv, 50);
}

#[test]
fn parse_header_v3() {
    let buf = build_header(3, 0, 0);
    let h = parse_header(&buf).unwrap();
    assert_eq!(h.version, 3);
    assert_eq!(h.n_tensors, 0);
    assert_eq!(h.n_kv, 0);
}

#[test]
fn parse_header_max_counts() {
    let buf = build_header(3, u64::MAX, u64::MAX);
    let h = parse_header(&buf).unwrap();
    assert_eq!(h.n_tensors, u64::MAX);
    assert_eq!(h.n_kv, u64::MAX);
}

#[test]
fn parse_header_zero_counts() {
    let buf = build_header(1, 0, 0);
    let h = parse_header(&buf).unwrap();
    assert_eq!(h.n_tensors, 0);
    assert_eq!(h.n_kv, 0);
}

#[test]
fn parse_header_extra_bytes_ignored() {
    let mut buf = build_header(2, 5, 3);
    buf.extend_from_slice(&[0xFF; 100]); // extra bytes after header
    let h = parse_header(&buf).unwrap();
    assert_eq!(h.version, 2);
    assert_eq!(h.n_tensors, 5);
    assert_eq!(h.n_kv, 3);
}

// ── parse_header: error cases ────────────────────────────────────────────────

#[test]
fn parse_header_too_short_empty() {
    let result = parse_header(&[]);
    assert!(result.is_err());
    match result.unwrap_err() {
        GgufError::ShortHeader(n) => assert_eq!(n, 0),
        e => panic!("unexpected error: {e:?}"),
    }
}

#[test]
fn parse_header_too_short_partial() {
    let result = parse_header(&[0x47, 0x47, 0x55, 0x46]); // just "GGUF"
    assert!(result.is_err());
    match result.unwrap_err() {
        GgufError::ShortHeader(4) => {}
        e => panic!("unexpected error: {e:?}"),
    }
}

#[test]
fn parse_header_too_short_23_bytes() {
    let buf = &build_header(2, 5, 3)[..23]; // one byte too short
    let result = parse_header(buf);
    assert!(result.is_err());
    match result.unwrap_err() {
        GgufError::ShortHeader(23) => {}
        e => panic!("unexpected error: {e:?}"),
    }
}

#[test]
fn parse_header_bad_magic_null() {
    let mut buf = build_header(2, 1, 1);
    buf[0..4].copy_from_slice(&[0, 0, 0, 0]);
    let result = parse_header(&buf);
    assert!(result.is_err());
    match result.unwrap_err() {
        GgufError::BadMagic(m) => assert_eq!(m, [0, 0, 0, 0]),
        e => panic!("unexpected error: {e:?}"),
    }
}

#[test]
fn parse_header_bad_magic_wrong_bytes() {
    let mut buf = build_header(2, 1, 1);
    buf[0..4].copy_from_slice(b"GGML");
    let result = parse_header(&buf);
    assert!(result.is_err());
    match result.unwrap_err() {
        GgufError::BadMagic(m) => assert_eq!(&m, b"GGML"),
        e => panic!("unexpected error: {e:?}"),
    }
}

#[test]
fn parse_header_unsupported_version_0() {
    let buf = build_header(0, 1, 1);
    let result = parse_header(&buf);
    assert!(result.is_err());
    match result.unwrap_err() {
        GgufError::UnsupportedVersion(0) => {}
        e => panic!("unexpected error: {e:?}"),
    }
}

#[test]
fn parse_header_unsupported_version_4() {
    let buf = build_header(4, 1, 1);
    let result = parse_header(&buf);
    assert!(result.is_err());
    match result.unwrap_err() {
        GgufError::UnsupportedVersion(4) => {}
        e => panic!("unexpected error: {e:?}"),
    }
}

#[test]
fn parse_header_unsupported_version_max() {
    let buf = build_header(u32::MAX, 1, 1);
    let result = parse_header(&buf);
    assert!(result.is_err());
    match result.unwrap_err() {
        GgufError::UnsupportedVersion(v) => assert_eq!(v, u32::MAX),
        e => panic!("unexpected error: {e:?}"),
    }
}

// ── GgufHeader: struct traits ────────────────────────────────────────────────

#[test]
fn gguf_header_clone_copy() {
    let h = GgufHeader { version: 3, n_tensors: 10, n_kv: 5 };
    let h2 = h; // copy
    let h3 = h.clone();
    assert_eq!(h, h2);
    assert_eq!(h, h3);
}

#[test]
fn gguf_header_debug_display() {
    let h = GgufHeader { version: 2, n_tensors: 42, n_kv: 7 };
    let dbg = format!("{h:?}");
    assert!(dbg.contains("GgufHeader"));
    assert!(dbg.contains("42"));
    assert!(dbg.contains("7"));
}

#[test]
fn gguf_header_equality() {
    let h1 = GgufHeader { version: 3, n_tensors: 10, n_kv: 5 };
    let h2 = GgufHeader { version: 3, n_tensors: 10, n_kv: 5 };
    let h3 = GgufHeader { version: 2, n_tensors: 10, n_kv: 5 };
    assert_eq!(h1, h2);
    assert_ne!(h1, h3);
}

// ── GgufValue: variant construction and Debug ────────────────────────────────

#[test]
fn gguf_value_u8() {
    let v = GgufValue::U8(255);
    let dbg = format!("{v:?}");
    assert!(dbg.contains("U8"));
    assert!(dbg.contains("255"));
}

#[test]
fn gguf_value_i8() {
    let v = GgufValue::I8(-128);
    let dbg = format!("{v:?}");
    assert!(dbg.contains("I8"));
}

#[test]
fn gguf_value_u16() {
    let v = GgufValue::U16(65535);
    if let GgufValue::U16(x) = v {
        assert_eq!(x, 65535);
    }
}

#[test]
fn gguf_value_i16() {
    let v = GgufValue::I16(-32768);
    if let GgufValue::I16(x) = v {
        assert_eq!(x, -32768);
    }
}

#[test]
fn gguf_value_u32() {
    let v = GgufValue::U32(u32::MAX);
    if let GgufValue::U32(x) = v {
        assert_eq!(x, u32::MAX);
    }
}

#[test]
fn gguf_value_i32() {
    let v = GgufValue::I32(i32::MIN);
    if let GgufValue::I32(x) = v {
        assert_eq!(x, i32::MIN);
    }
}

#[test]
fn gguf_value_f32() {
    let v = GgufValue::F32(std::f32::consts::PI);
    if let GgufValue::F32(x) = v {
        assert!((x - std::f32::consts::PI).abs() < 1e-7);
    }
}

#[test]
fn gguf_value_bool_true() {
    let v = GgufValue::Bool(true);
    assert_eq!(v, GgufValue::Bool(true));
    assert_ne!(v, GgufValue::Bool(false));
}

#[test]
fn gguf_value_bool_false() {
    let v = GgufValue::Bool(false);
    assert_eq!(v, GgufValue::Bool(false));
}

#[test]
fn gguf_value_string() {
    let v = GgufValue::String("hello".to_string());
    if let GgufValue::String(s) = &v {
        assert_eq!(s, "hello");
    }
}

#[test]
fn gguf_value_string_empty() {
    let v = GgufValue::String(String::new());
    if let GgufValue::String(s) = &v {
        assert!(s.is_empty());
    }
}

#[test]
fn gguf_value_u64() {
    let v = GgufValue::U64(u64::MAX);
    if let GgufValue::U64(x) = v {
        assert_eq!(x, u64::MAX);
    }
}

#[test]
fn gguf_value_i64() {
    let v = GgufValue::I64(i64::MIN);
    if let GgufValue::I64(x) = v {
        assert_eq!(x, i64::MIN);
    }
}

#[test]
fn gguf_value_f64() {
    let v = GgufValue::F64(std::f64::consts::E);
    if let GgufValue::F64(x) = v {
        assert!((x - std::f64::consts::E).abs() < 1e-15);
    }
}

#[test]
fn gguf_value_array_empty() {
    let v = GgufValue::Array(vec![]);
    if let GgufValue::Array(arr) = &v {
        assert!(arr.is_empty());
    }
}

#[test]
fn gguf_value_array_mixed() {
    let v = GgufValue::Array(vec![
        GgufValue::U8(1),
        GgufValue::String("two".to_string()),
        GgufValue::F32(3.0),
    ]);
    if let GgufValue::Array(arr) = &v {
        assert_eq!(arr.len(), 3);
    }
}

#[test]
fn gguf_value_nested_array() {
    let v = GgufValue::Array(vec![GgufValue::Array(vec![GgufValue::U8(42)])]);
    if let GgufValue::Array(outer) = &v {
        if let GgufValue::Array(inner) = &outer[0] {
            assert_eq!(inner[0], GgufValue::U8(42));
        }
    }
}

#[test]
fn gguf_value_clone_eq() {
    let v1 = GgufValue::String("test".to_string());
    let v2 = v1.clone();
    assert_eq!(v1, v2);
}

// ── GgufKv: struct tests ─────────────────────────────────────────────────────

#[test]
fn gguf_kv_construction() {
    let kv = bitnet_gguf::kv::GgufKv {
        key: "general.name".to_string(),
        value: GgufValue::String("TestModel".to_string()),
    };
    assert_eq!(kv.key, "general.name");
    if let GgufValue::String(s) = &kv.value {
        assert_eq!(s, "TestModel");
    }
}

#[test]
fn gguf_kv_debug() {
    let kv = bitnet_gguf::kv::GgufKv { key: "test.key".to_string(), value: GgufValue::U32(42) };
    let dbg = format!("{kv:?}");
    assert!(dbg.contains("GgufKv"));
    assert!(dbg.contains("test.key"));
}

#[test]
fn gguf_kv_clone() {
    let kv = bitnet_gguf::kv::GgufKv { key: "a.b".to_string(), value: GgufValue::Bool(true) };
    let kv2 = kv.clone();
    assert_eq!(kv.key, kv2.key);
}

// ── GgufError: Display ──────────────────────────────────────────────────────

#[test]
fn gguf_error_bad_magic_display() {
    let e = GgufError::BadMagic([0xDE, 0xAD, 0xBE, 0xEF]);
    let msg = e.to_string();
    assert!(msg.contains("bad magic"));
}

#[test]
fn gguf_error_unsupported_version_display() {
    let e = GgufError::UnsupportedVersion(99);
    let msg = e.to_string();
    assert!(msg.contains("unsupported GGUF version"));
    assert!(msg.contains("99"));
}

#[test]
fn gguf_error_short_header_display() {
    let e = GgufError::ShortHeader(12);
    let msg = e.to_string();
    assert!(msg.contains("short header"));
    assert!(msg.contains("12"));
}

#[test]
fn gguf_error_malformed_display() {
    let e = GgufError::Malformed;
    let msg = e.to_string();
    assert!(msg.contains("malformed"));
}

#[test]
fn gguf_error_invalid_kv_type_display() {
    let e = GgufError::InvalidKvType(255);
    let msg = e.to_string();
    assert!(msg.contains("invalid KV type"));
    assert!(msg.contains("255"));
}

#[test]
fn gguf_error_string_too_large_display() {
    let e = GgufError::StringTooLarge(999999);
    let msg = e.to_string();
    assert!(msg.contains("string too large"));
    assert!(msg.contains("999999"));
}

// ── read_kv_pairs: synthetic GGUF files ─────────────────────────────────────

fn write_synthetic_gguf_with_kv(
    f: &mut impl Write,
    version: u32,
    n_tensors: u64,
    kvs: &[(&str, u32, &[u8])],
) {
    // Header
    f.write_all(b"GGUF").unwrap();
    f.write_all(&version.to_le_bytes()).unwrap();
    f.write_all(&n_tensors.to_le_bytes()).unwrap();
    f.write_all(&(kvs.len() as u64).to_le_bytes()).unwrap();

    // KV pairs
    for &(key, ty, value_bytes) in kvs {
        // key: u64 len + utf8 bytes
        f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        f.write_all(key.as_bytes()).unwrap();
        // type
        f.write_all(&ty.to_le_bytes()).unwrap();
        // value
        f.write_all(value_bytes).unwrap();
    }
}

#[test]
fn read_kv_pairs_u32_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_synthetic_gguf_with_kv(
        &mut f,
        3,
        0,
        &[("test.count", 4, &42u32.to_le_bytes())], // type 4 = U32
    );
    drop(f);

    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs.len(), 1);
    assert_eq!(kvs[0].key, "test.count");
    assert_eq!(kvs[0].value, GgufValue::U32(42));
}

#[test]
fn read_kv_pairs_string_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    let mut f = std::fs::File::create(&path).unwrap();

    // String type = 8, value = u64 len + bytes
    let val = "hello";
    let mut val_bytes = (val.len() as u64).to_le_bytes().to_vec();
    val_bytes.extend_from_slice(val.as_bytes());

    write_synthetic_gguf_with_kv(&mut f, 3, 0, &[("general.name", 8, &val_bytes)]);
    drop(f);

    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs.len(), 1);
    assert_eq!(kvs[0].key, "general.name");
    assert_eq!(kvs[0].value, GgufValue::String("hello".to_string()));
}

#[test]
fn read_kv_pairs_bool_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_synthetic_gguf_with_kv(&mut f, 3, 0, &[("flag.on", 7, &[1u8])]);
    drop(f);

    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].value, GgufValue::Bool(true));
}

#[test]
fn read_kv_pairs_bool_false() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_synthetic_gguf_with_kv(&mut f, 3, 0, &[("flag.off", 7, &[0u8])]);
    drop(f);

    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].value, GgufValue::Bool(false));
}

#[test]
fn read_kv_pairs_multiple() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_synthetic_gguf_with_kv(
        &mut f,
        3,
        5,
        &[("count", 4, &10u32.to_le_bytes()), ("flag", 7, &[1u8])],
    );
    drop(f);

    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs.len(), 2);
    assert_eq!(kvs[0].key, "count");
    assert_eq!(kvs[1].key, "flag");
}

#[test]
fn read_kv_pairs_with_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_synthetic_gguf_with_kv(
        &mut f,
        3,
        0,
        &[
            ("a", 4, &1u32.to_le_bytes()),
            ("b", 4, &2u32.to_le_bytes()),
            ("c", 4, &3u32.to_le_bytes()),
        ],
    );
    drop(f);

    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, Some(2)).unwrap();
    assert_eq!(kvs.len(), 2);
    assert_eq!(kvs[0].key, "a");
    assert_eq!(kvs[1].key, "b");
}

#[test]
fn read_kv_pairs_i8_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_synthetic_gguf_with_kv(&mut f, 3, 0, &[("val", 1, &[0x80u8])]); // type 1 = I8
    drop(f);

    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].value, GgufValue::I8(-128));
}

#[test]
fn read_kv_pairs_f32_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_synthetic_gguf_with_kv(&mut f, 3, 0, &[("pi", 6, &std::f32::consts::PI.to_le_bytes())]);
    drop(f);

    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    if let GgufValue::F32(x) = kvs[0].value {
        assert!((x - std::f32::consts::PI).abs() < 1e-7);
    } else {
        panic!("expected F32");
    }
}

#[test]
fn read_kv_pairs_u64_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    write_synthetic_gguf_with_kv(
        &mut f,
        3,
        0,
        &[("big", 10, &u64::MAX.to_le_bytes())], // type 10 = U64
    );
    drop(f);

    let kvs = bitnet_gguf::kv::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].value, GgufValue::U64(u64::MAX));
}

#[test]
fn read_kv_pairs_nonexistent_file() {
    let result = bitnet_gguf::kv::read_kv_pairs("/nonexistent/file.gguf", None);
    assert!(result.is_err());
}

#[test]
fn read_kv_pairs_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.gguf");
    std::fs::File::create(&path).unwrap();

    let result = bitnet_gguf::kv::read_kv_pairs(&path, None);
    assert!(result.is_err());
}

#[test]
fn read_kv_pairs_bad_magic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(b"GGML").unwrap(); // wrong magic
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    drop(f);

    let result = bitnet_gguf::kv::read_kv_pairs(&path, None);
    assert!(result.is_err());
}

// ── GGUF_HEADER_LEN constant ────────────────────────────────────────────────

#[test]
fn gguf_header_len_is_24() {
    assert_eq!(GGUF_HEADER_LEN, 24);
}

// ── Serialization round-trip ─────────────────────────────────────────────────

#[test]
fn gguf_header_serde_roundtrip() {
    let h = GgufHeader { version: 3, n_tensors: 42, n_kv: 7 };
    let json = serde_json::to_string(&h).unwrap();
    let h2: GgufHeader = serde_json::from_str(&json).unwrap();
    assert_eq!(h, h2);
}

#[test]
fn gguf_value_serde_roundtrip() {
    let v = GgufValue::String("model_name".to_string());
    let json = serde_json::to_string(&v).unwrap();
    let v2: GgufValue = serde_json::from_str(&json).unwrap();
    assert_eq!(v, v2);
}

#[test]
fn gguf_value_array_serde_roundtrip() {
    let v = GgufValue::Array(vec![GgufValue::U32(1), GgufValue::U32(2), GgufValue::U32(3)]);
    let json = serde_json::to_string(&v).unwrap();
    let v2: GgufValue = serde_json::from_str(&json).unwrap();
    assert_eq!(v, v2);
}

// ── read_header_blocking ─────────────────────────────────────────────────────

#[test]
fn read_header_blocking_valid() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&build_header(3, 10, 5)).unwrap();
    drop(f);

    let h = bitnet_gguf::kv::read_header_blocking(&path).unwrap();
    assert_eq!(h.version, 3);
    assert_eq!(h.n_tensors, 10);
    assert_eq!(h.n_kv, 5);
}

#[test]
fn read_header_blocking_truncated() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trunc.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&build_header(3, 10, 5)[..20]).unwrap(); // 20 < 24
    drop(f);

    let result = bitnet_gguf::kv::read_header_blocking(&path);
    assert!(result.is_err());
}

#[test]
fn read_header_blocking_nonexistent() {
    let result = bitnet_gguf::kv::read_header_blocking("/no/such/file.gguf");
    assert!(result.is_err());
}
