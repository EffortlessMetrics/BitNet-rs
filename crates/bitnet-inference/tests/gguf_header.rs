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
fn rejects_unreasonable_counts() {
    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(b"GGUF");
    buf[4..8].copy_from_slice(2u32.to_le_bytes().as_slice());
    buf[8..16].copy_from_slice(20_000_000u64.to_le_bytes().as_slice()); // too many tensors

    let err = bitnet_inference::gguf::parse_header(&buf).unwrap_err();
    assert!(matches!(err, bitnet_inference::gguf::GgufError::Malformed));
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
