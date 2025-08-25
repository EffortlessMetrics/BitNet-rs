#[test]
fn parses_min_header() {
    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(b"GGUF");
    buf[4..8].copy_from_slice(2u32.to_le_bytes().as_slice()); // version=2 (example)
    buf[8..16].copy_from_slice(0u64.to_le_bytes().as_slice()); // n_tensors
    buf[16..24].copy_from_slice(0u64.to_le_bytes().as_slice()); // n_kv

    let h = bitnet_inference::gguf::parse_header(&buf).expect("header");
    assert_eq!(h.version, 2);
    assert_eq!(h.n_tensors, 0);
    assert_eq!(h.n_kv, 0);
}

#[test]
fn rejects_bad_magic() {
    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(b"NOPE");
    buf[4..8].copy_from_slice(2u32.to_le_bytes().as_slice());

    let err = bitnet_inference::gguf::parse_header(&buf).unwrap_err();
    assert!(matches!(err, bitnet_inference::gguf::GgufError::BadMagic(_)));
}

#[test]
fn rejects_short_buffer() {
    let buf = [0u8; 20]; // < 24
    let err = bitnet_inference::gguf::parse_header(&buf).unwrap_err();
    assert!(matches!(err, bitnet_inference::gguf::GgufError::ShortHeader(20)));
}

#[test]
fn rejects_unsupported_version() {
    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(b"GGUF");
    buf[4..8].copy_from_slice(999u32.to_le_bytes().as_slice()); // unsupported version

    let err = bitnet_inference::gguf::parse_header(&buf).unwrap_err();
    assert!(matches!(err, bitnet_inference::gguf::GgufError::UnsupportedVersion(999)));
}

#[test]
fn accepts_large_counts() {
    // Parser is format-only, doesn't reject based on counts
    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(b"GGUF");
    buf[4..8].copy_from_slice(2u32.to_le_bytes().as_slice());
    buf[8..16].copy_from_slice(20_000_000u64.to_le_bytes().as_slice()); // many tensors

    let header = bitnet_inference::gguf::parse_header(&buf).unwrap();
    assert_eq!(header.n_tensors, 20_000_000);
    assert_eq!(header.n_kv, 0);
}

#[test]
fn test_kv_types() {
    use bitnet_inference::gguf::GgufValue;
    
    // Test value creation and equality
    let u8_val = GgufValue::U8(42);
    assert_eq!(u8_val, GgufValue::U8(42));
    
    let str_val = GgufValue::String("test".to_string());
    assert_eq!(str_val, GgufValue::String("test".to_string()));
    
    let bool_val = GgufValue::Bool(true);
    assert_eq!(bool_val, GgufValue::Bool(true));
    
    let f32_val = GgufValue::F32(3.14);
    if let GgufValue::F32(v) = f32_val {
        assert!((v - 3.14).abs() < 0.001);
    }
}

#[test]
fn test_kv_reader_with_mock_file() {
    use std::io::Write;
    
    // Create a simple GGUF file with KV pairs
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_kv.gguf");
    let mut file = std::fs::File::create(&path).unwrap();
    
    // Write GGUF header
    file.write_all(b"GGUF").unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap();  // version
    file.write_all(&0u64.to_le_bytes()).unwrap();  // n_tensors
    file.write_all(&2u64.to_le_bytes()).unwrap();  // n_kv = 2
    
    // Write first KV pair: "test.u32" = 42u32
    file.write_all(&8u64.to_le_bytes()).unwrap();  // key length
    file.write_all(b"test.u32").unwrap();          // key
    file.write_all(&4u32.to_le_bytes()).unwrap();  // type = UINT32
    file.write_all(&42u32.to_le_bytes()).unwrap(); // value
    
    // Write second KV pair: "test.str" = "hello"
    file.write_all(&8u64.to_le_bytes()).unwrap();  // key length
    file.write_all(b"test.str").unwrap();          // key
    file.write_all(&8u32.to_le_bytes()).unwrap();  // type = STRING
    file.write_all(&5u64.to_le_bytes()).unwrap();  // string length
    file.write_all(b"hello").unwrap();             // string value
    
    file.flush().unwrap();
    drop(file);
    
    // Test reading KV pairs
    let kvs = bitnet_inference::gguf::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs.len(), 2);
    
    assert_eq!(kvs[0].key, "test.u32");
    assert_eq!(kvs[0].value, bitnet_inference::gguf::GgufValue::U32(42));
    
    assert_eq!(kvs[1].key, "test.str");
    assert_eq!(kvs[1].value, bitnet_inference::gguf::GgufValue::String("hello".to_string()));
    
    // Test with limit
    let limited_kvs = bitnet_inference::gguf::read_kv_pairs(&path, Some(1)).unwrap();
    assert_eq!(limited_kvs.len(), 1);
    assert_eq!(limited_kvs[0].key, "test.u32");
}

#[test]
fn test_blocking_reader() {
    // Create a temporary GGUF file
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");

    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(b"GGUF");
    buf[4..8].copy_from_slice(1u32.to_le_bytes().as_slice());
    std::fs::write(&path, &buf).unwrap();

    let header = bitnet_inference::gguf::read_header_blocking(&path).unwrap();
    assert_eq!(header.version, 1);
    assert_eq!(header.n_tensors, 0);
    assert_eq!(header.n_kv, 0);
}
