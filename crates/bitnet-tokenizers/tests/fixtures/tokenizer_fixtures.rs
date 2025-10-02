//! Tokenizer Test Data Fixtures for Universal Tokenizer Discovery
//!
//! This module provides HuggingFace JSON and SentencePiece tokenizer test data.

#![cfg(test)]

use serde_json::json;
use std::fs::File;
use std::io::{Result as IoResult, Write};
use std::path::Path;

/// Generate a minimal HuggingFace tokenizer JSON
pub fn generate_hf_tokenizer_json(path: &Path, vocab_size: usize) -> IoResult<()> {
    let vocab: Vec<String> = (0..vocab_size)
        .map(|i| {
            if i == 0 {
                "<s>".to_string()
            } else if i == 1 {
                "</s>".to_string()
            } else if i == 2 {
                "<unk>".to_string()
            } else {
                format!("token_{}", i)
            }
        })
        .collect();

    let tokenizer_json = json!({
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab.iter().enumerate()
                .map(|(i, token)| (token.clone(), i))
                .collect::<std::collections::HashMap<_, _>>(),
            "merges": []
        },
        "added_tokens": [
            {"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel"},
        "post_processor": null,
        "decoder": {"type": "ByteLevel"}
    });

    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, &tokenizer_json)?;
    Ok(())
}

/// Generate a mock SentencePiece model file
/// Note: This is a simplified binary format for testing
pub fn generate_spm_model(path: &Path, vocab_size: u32) -> IoResult<()> {
    let mut file = File::create(path)?;

    // Simplified SentencePiece binary header (not fully compatible, just for testing)
    // Real SPM models use protobuf, but we create a mock for basic testing

    // Mock header
    file.write_all(b"SPM_MOCK")?;
    file.write_all(&vocab_size.to_le_bytes())?;

    // Mock vocabulary entries
    for i in 0..vocab_size.min(100) {
        let token = if i == 0 {
            "<unk>"
        } else if i == 1 {
            "<s>"
        } else if i == 2 {
            "</s>"
        } else {
            "▁token"
        };

        file.write_all(&(token.len() as u32).to_le_bytes())?;
        file.write_all(token.as_bytes())?;
        file.write_all(&(-100.0f32 - i as f32).to_le_bytes())?; // score
    }

    Ok(())
}

/// Generate all tokenizer fixtures
pub fn generate_all_tokenizer_fixtures() -> IoResult<()> {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizers");
    std::fs::create_dir_all(&fixtures_dir)?;

    // HuggingFace tokenizer fixtures
    generate_hf_tokenizer_json(&fixtures_dir.join("hf_tokenizer.json"), 32000)?;
    generate_hf_tokenizer_json(&fixtures_dir.join("hf_128k.json"), 128256)?;
    generate_hf_tokenizer_json(&fixtures_dir.join("hf_50k.json"), 50257)?;

    // SentencePiece model fixtures
    generate_spm_model(&fixtures_dir.join("spm_model.model"), 32000)?;
    generate_spm_model(&fixtures_dir.join("spm_128k.model"), 128256)?;

    // BPE vocabulary file
    let bpe_vocab_path = fixtures_dir.join("bpe_vocab.txt");
    let mut bpe_file = File::create(bpe_vocab_path)?;
    for i in 0..32000 {
        if i == 0 {
            writeln!(bpe_file, "<s>")?;
        } else if i == 1 {
            writeln!(bpe_file, "</s>")?;
        } else if i == 2 {
            writeln!(bpe_file, "<unk>")?;
        } else {
            writeln!(bpe_file, "token_{}", i)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_hf_tokenizer() {
        let temp_dir = TempDir::new().unwrap();
        let test_path = temp_dir.path().join("tokenizer.json");

        generate_hf_tokenizer_json(&test_path, 1000).unwrap();
        assert!(test_path.exists());

        // Verify it's valid JSON
        let contents = std::fs::read_to_string(&test_path).unwrap();
        let _json: serde_json::Value = serde_json::from_str(&contents).unwrap();
    }

    #[test]
    fn test_generate_spm_model() {
        let temp_dir = TempDir::new().unwrap();
        let test_path = temp_dir.path().join("model.model");

        generate_spm_model(&test_path, 32000).unwrap();
        assert!(test_path.exists());

        // Verify header
        let contents = std::fs::read(&test_path).unwrap();
        assert_eq!(&contents[0..8], b"SPM_MOCK");
    }
}
