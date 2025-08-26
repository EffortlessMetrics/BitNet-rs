use bitnet_compat::gguf_fixer::GgufCompatibilityFixer;
use bitnet_models::formats::gguf::GgufReader;
use std::fs;
use tempfile::TempDir;

#[test]
fn edits_metadata_and_is_idempotent() {
    let temp_dir = TempDir::new().unwrap();
    let src = temp_dir.path().join("mini.gguf");
    let dst = temp_dir.path().join("fixed.gguf");

    // Write minimal GGUF header (magic + version + counts)
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    fs::write(&src, &data).unwrap();

    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();

    let out = fs::read(&dst).unwrap();
    let reader = GgufReader::new(&out).unwrap();
    assert!(reader.get_bool_metadata("bitnet.compat.fixed").unwrap_or(false));
    assert_eq!(reader.get_u32_metadata("tokenizer.ggml.bos_token_id"), Some(1));

    assert!(GgufCompatibilityFixer::verify_idempotent(&dst).unwrap());
}
