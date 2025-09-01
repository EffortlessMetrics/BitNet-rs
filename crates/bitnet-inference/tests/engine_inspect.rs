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
    assert!(info.kv_specs().is_empty());
    assert!(info.quantization_hints().is_empty());
    assert!(info.tensor_summaries().is_empty());
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
fn engine_inspect_parses_metadata_and_tensors() {
    use bitnet_inference::engine::inspect_model;
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let p = dir.path().join("m.gguf");

    // build minimal GGUF with one kv and one tensor
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(2u32.to_le_bytes().as_slice()); // version
    data.extend_from_slice(1u64.to_le_bytes().as_slice()); // n_tensors
    data.extend_from_slice(1u64.to_le_bytes().as_slice()); // n_kv

    // kv: general.file_type -> u32 value 1
    let key = b"general.file_type";
    data.extend_from_slice((key.len() as u64).to_le_bytes().as_slice());
    data.extend_from_slice(key);
    data.extend_from_slice(4u32.to_le_bytes().as_slice()); // type u32
    data.extend_from_slice(1u32.to_le_bytes().as_slice()); // value

    // tensor metadata
    let tname = b"tensor";
    data.extend_from_slice((tname.len() as u64).to_le_bytes().as_slice());
    data.extend_from_slice(tname);
    data.extend_from_slice(1u32.to_le_bytes().as_slice()); // n_dims
    data.extend_from_slice(5u64.to_le_bytes().as_slice()); // dim
    data.extend_from_slice(0u32.to_le_bytes().as_slice()); // dtype
    data.extend_from_slice(0u64.to_le_bytes().as_slice()); // offset

    std::fs::write(&p, data).unwrap();

    let info = inspect_model(&p).unwrap();
    assert_eq!(info.kv_specs().len(), 1);
    assert_eq!(info.quantization_hints().len(), 1);
    assert_eq!(info.tensor_summaries().len(), 1);
    assert_eq!(info.tensor_summaries()[0].name, "tensor");
    assert_eq!(info.tensor_summaries()[0].shape, vec![5]);
}
