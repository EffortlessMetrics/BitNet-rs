use proptest::prelude::*;

proptest! {
    #[test]
    fn rejects_short_or_garbage(buf in prop::collection::vec(any::<u8>(), 0..24)) {
        assert!(bitnet_inference::gguf::parse_header(&buf).is_err());
    }

    #[test]
    fn rejects_random_data(buf in prop::collection::vec(any::<u8>(), 24..100)) {
        // Random 24+ byte buffers are almost certainly invalid
        let result = bitnet_inference::gguf::parse_header(&buf);
        // Most should fail, but if magic + version accidentally match, that's ok
        if let Ok(header) = result {
            assert!((1..=3).contains(&header.version));
        }
    }

    #[test]
    fn handles_all_buffer_sizes(buf in prop::collection::vec(any::<u8>(), 0..100)) {
        // Should never panic, regardless of input
        let _ = bitnet_inference::gguf::parse_header(&buf);
    }
}

#[test]
fn accepts_minimal_valid() {
    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(b"GGUF");
    buf[4..8].copy_from_slice(2u32.to_le_bytes().as_slice());
    // n_tensors / n_kv already zero
    let hdr = bitnet_inference::gguf::parse_header(&buf).unwrap();
    assert_eq!(hdr.version, 2);
}

#[test]
fn accepts_all_valid_versions() {
    for version in 1u32..=3 {
        let mut buf = [0u8; 24];
        buf[0..4].copy_from_slice(b"GGUF");
        buf[4..8].copy_from_slice(version.to_le_bytes().as_slice());

        let hdr = bitnet_inference::gguf::parse_header(&buf).unwrap();
        assert_eq!(hdr.version, version);
    }
}
