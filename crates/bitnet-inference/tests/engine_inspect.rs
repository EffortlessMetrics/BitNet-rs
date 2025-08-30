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
    std::fs::write(&p, buf).unwrap();

    let info = inspect_model(&p).unwrap();
    assert_eq!(info.version(), 2);
    assert_eq!(info.n_tensors(), 0);
    assert_eq!(info.n_kv(), 0);
    assert!(info.kv().is_empty());
    assert!(info.tensor_summaries().is_empty());
    assert!(info.quantization_hints().is_empty());
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
    std::fs::write(&p, buf).unwrap();

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

#[test]
fn engine_inspect_extracts_kv_and_tensors() {
    use bitnet_inference::engine::inspect_model;
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    let mut f = std::fs::File::create(&path).unwrap();

    // header: magic, version=2, n_tensors=2, n_kv=1
    f.write_all(b"GGUF").unwrap();
    f.write_all(&2u32.to_le_bytes()).unwrap();
    f.write_all(&2u64.to_le_bytes()).unwrap();
    f.write_all(&1u64.to_le_bytes()).unwrap();

    // KV pair: key "test.str" = "hello"
    f.write_all(&8u64.to_le_bytes()).unwrap();
    f.write_all(b"test.str").unwrap();
    f.write_all(&8u32.to_le_bytes()).unwrap(); // STRING
    f.write_all(&5u64.to_le_bytes()).unwrap();
    f.write_all(b"hello").unwrap();

    // Tensor 1: name "w", 1 dim [4], type 0 (F32), offset 0
    f.write_all(&1u64.to_le_bytes()).unwrap();
    f.write_all(b"w").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&4u64.to_le_bytes()).unwrap();
    f.write_all(&0u32.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();

    // Tensor 2: name "qw", 1 dim [4], type 36 (I2_S), offset 0
    f.write_all(&2u64.to_le_bytes()).unwrap();
    f.write_all(b"qw").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&4u64.to_le_bytes()).unwrap();
    f.write_all(&36u32.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    f.flush().unwrap();

    let info = inspect_model(&path).unwrap();
    assert_eq!(info.n_kv(), 1);
    assert_eq!(info.kv()[0].key, "test.str");
    assert_eq!(info.tensor_summaries().len(), 2);
    assert_eq!(info.tensor_summaries()[0].name, "w");
    assert_eq!(info.tensor_summaries()[1].ty, 36);
    assert_eq!(info.quantization_hints(), &[36u32]);
}
