#[test]
fn parses_min_header() {
    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(b"GGUF");
    buf[4..8].copy_from_slice(2u32.to_le_bytes().as_slice());      // version=2 (example)
    buf[8..16].copy_from_slice(0u64.to_le_bytes().as_slice());     // n_tensors
    buf[16..24].copy_from_slice(0u64.to_le_bytes().as_slice());    // n_kv

    let h = bitnet_inference::gguf::parse_header(&buf).expect("header");
    assert_eq!(h.version, 2);
    assert_eq!(h.n_tensors, 0);
    assert_eq!(h.n_kv, 0);
}

#[test]
fn rejects_bad_magic() {
    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(b"BAAD");
    
    let result = bitnet_inference::gguf::parse_header(&buf);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("bad magic"));
}

#[test]
fn rejects_short_buffer() {
    let buf = [0u8; 23]; // one byte short
    
    let result = bitnet_inference::gguf::parse_header(&buf);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("header too short"));
}