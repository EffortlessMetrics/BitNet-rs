#![cfg(feature = "integration-tests")]

#[cfg(test)]
mod tests {
    use bitnet_compat::gguf_fixer::GgufCompatibilityFixer;
    use bitnet_models::GgufReader;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn fixes_metadata_and_is_idempotent() {
        let temp_dir = TempDir::new().unwrap();
        let src = temp_dir.path().join("mini.gguf");
        let dst = temp_dir.path().join("fixed.gguf");

        // minimal GGUF header (magic + version + empty counts)
        let mut gguf_data = Vec::new();
        gguf_data.extend_from_slice(b"GGUF");
        gguf_data.extend_from_slice(&3u32.to_le_bytes());
        gguf_data.extend_from_slice(&0u64.to_le_bytes());
        gguf_data.extend_from_slice(&0u64.to_le_bytes());
        fs::write(&src, &gguf_data).unwrap();

        GgufCompatibilityFixer::fix_and_export(&src, &dst).unwrap();

        let data = fs::read(&dst).unwrap();
        let reader = GgufReader::new(&data).unwrap();
        assert!(reader.get_u32_metadata("tokenizer.ggml.bos_token_id").is_some());
        assert!(reader.get_u32_metadata("tokenizer.ggml.eos_token_id").is_some());
        assert!(reader.get_u32_metadata("tokenizer.ggml.vocab_size").is_some());

        assert!(GgufCompatibilityFixer::verify_idempotent(&dst).unwrap());
        assert!(!dst.with_extension("compat.json").exists());
    }
}
