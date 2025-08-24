#[test]
fn mini_gguf_metadata_correct() {
    use bitnet_models::formats::gguf::GgufReader;
    use bitnet_models::loader::MmapFile;
    use std::path::Path;

    // Test v3 fixture
    let v3_path = Path::new("target").join("mini_v3.gguf");
    if v3_path.exists() {
        let mmap = MmapFile::open(&v3_path).expect("Failed to mmap");
        let reader = GgufReader::new(mmap.as_slice()).expect("Should parse v3");

        // Check expected metadata keys
        assert_eq!(
            reader.get_string_metadata("general.architecture"),
            Some("test".to_string()),
            "general.architecture should be 'test'"
        );

        assert_eq!(
            reader.get_string_metadata("general.name"),
            Some("mini_test_model".to_string()),
            "general.name should be 'mini_test_model'"
        );
    }
}

#[test]
fn verify_spec_compliance() {
    // Verify the exact byte structure of generated files
    use std::fs;
    use std::path::Path;

    let v3_path = Path::new("target").join("mini_v3.gguf");
    if v3_path.exists() {
        let bytes = fs::read(&v3_path).expect("Failed to read v3");

        // Check magic
        assert_eq!(&bytes[0..4], b"GGUF", "v3 magic should be GGUF");

        // Check version
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(version, 3, "v3 version should be 3");

        // Check tensor count (u64 for v3)
        let tensor_count = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        assert_eq!(tensor_count, 0, "v3 should have 0 tensors");

        // Check KV count (u64 for v3)
        let kv_count = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        assert_eq!(kv_count, 4, "v3 should have 4 KV pairs");

        // Check alignment field
        let alignment = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        assert_eq!(alignment, 32, "v3 alignment should be 32");

        // Check data offset
        let data_offset = u64::from_le_bytes([
            bytes[28], bytes[29], bytes[30], bytes[31], bytes[32], bytes[33], bytes[34], bytes[35],
        ]);
        assert_eq!(
            data_offset as usize,
            bytes.len(),
            "data_offset should equal file size for 0-tensor file"
        );
    }
}

#[test]
fn test_v2_requested_tag() {
    // Test that requesting v2 creates v3 with the tag
    use bitnet_models::formats::gguf::GgufReader;
    use bitnet_models::loader::MmapFile;
    use std::path::Path;

    let path = Path::new("target").join("mini_v2_tagged.gguf");

    // Generate with --version 2 (should create v3 with tag)
    std::process::Command::new("cargo")
        .args(["run", "-p", "xtask", "--", "gen-mini-gguf"])
        .arg("--output")
        .arg(&path)
        .arg("--version")
        .arg("2")
        .output()
        .expect("Failed to generate");

    // Verify it's actually v3
    let bytes = std::fs::read(&path).expect("Failed to read");
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(version, 3, "Should always emit v3");

    // Verify the tag is present
    let mmap = MmapFile::open(&path).expect("Failed to mmap");
    let reader = GgufReader::new(mmap.as_slice()).expect("Should parse");

    assert_eq!(
        reader.get_string_metadata("compat.v2_requested"),
        Some("true".to_string()),
        "Should have v2_requested=true when --version 2 is passed"
    );
}
