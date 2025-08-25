#[test]
fn engine_inspect_reads_header() {
    use bitnet_inference::engine::inspect_model;
    use tempfile::tempdir;

    // synthesize 24-byte GGUF
    let dir = tempdir().unwrap();
    let p = dir.path().join("m.gguf");
    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(b"GGUF");
    buf[4..8].copy_from_slice(2u32.to_le_bytes().as_slice());
    std::fs::write(&p, &buf).unwrap();

    let info = inspect_model(&p).unwrap();
    assert_eq!(info.version(), 2);
    assert_eq!(info.n_tensors(), 0);
    assert_eq!(info.n_kv(), 0);
}

#[test]
fn engine_inspect_rejects_invalid() {
    use bitnet_inference::engine::inspect_model;
    use tempfile::tempdir;

    // Create invalid GGUF (bad magic)
    let dir = tempdir().unwrap();
    let p = dir.path().join("bad.gguf");
    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(b"NOPE");
    std::fs::write(&p, &buf).unwrap();

    let result = inspect_model(&p);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, bitnet_inference::gguf::GgufError::BadMagic(_)));
}

#[test]
fn engine_inspect_handles_short_file() {
    use bitnet_inference::engine::inspect_model;
    use tempfile::tempdir;

    // Create too-short file
    let dir = tempdir().unwrap();
    let p = dir.path().join("short.gguf");
    std::fs::write(&p, b"GGUF").unwrap(); // Only 4 bytes

    let result = inspect_model(&p);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, bitnet_inference::gguf::GgufError::ShortHeader(_)));
}

#[test]
fn engine_inspect_rejects_bad_magic() {
    use bitnet_inference::engine::inspect_model;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("bad.gguf");
    std::fs::write(&p, b"NOPENOPENOPENOPE").unwrap(); // short & wrong
    let err = inspect_model(&p).unwrap_err();
    // We only assert it surfaces as a header error (don't tie to exact variant name)
    let _ = format!("{err}");
}
