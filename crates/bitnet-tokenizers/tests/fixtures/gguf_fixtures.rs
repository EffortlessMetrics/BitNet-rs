//! GGUF Test Fixture Generator for Universal Tokenizer Discovery
//!
//! This module generates realistic GGUF model fixtures for testing tokenizer discovery,
//! architecture detection, and vocabulary size resolution.

#![cfg(test)]

use std::fs::File;
use std::io::{BufWriter, Result as IoResult, Write};
use std::path::Path;

/// GGUF fixture configuration
#[derive(Debug, Clone)]
pub struct GgufFixtureConfig {
    pub model_type: String,
    pub vocab_size: u32,
    pub has_embedded_tokenizer: bool,
    pub tokenizer_type: Option<String>, // "hf" or "spm"
    pub corrupted: bool,
}

/// Generate a minimal valid GGUF file for testing
pub fn generate_gguf_fixture(path: &Path, config: &GgufFixtureConfig) -> IoResult<()> {
    let mut writer = BufWriter::new(File::create(path)?);

    // Write GGUF header
    if config.corrupted {
        writer.write_all(b"FAKE")?; // Invalid magic for error testing
    } else {
        writer.write_all(b"GGUF")?; // GGUF magic
    }
    writer.write_all(&3u32.to_le_bytes())?; // Version 3

    // Metadata count
    let mut metadata_count = 2u64; // model.type + vocab_size
    if config.has_embedded_tokenizer {
        metadata_count += 3; // tokenizer.ggml.* keys
    }

    // Tensor info count
    writer.write_all(&0u64.to_le_bytes())?; // No tensor info for minimal fixtures

    // KV metadata count
    writer.write_all(&metadata_count.to_le_bytes())?;

    // Write model.type metadata
    write_string_kv(&mut writer, "general.architecture", &config.model_type)?;

    // Write vocab_size metadata
    write_uint32_kv(&mut writer, &format!("{}.vocab_size", config.model_type), config.vocab_size)?;

    // Write embedded tokenizer metadata if present
    if config.has_embedded_tokenizer {
        if let Some(ref tokenizer_type) = config.tokenizer_type {
            write_string_kv(&mut writer, "tokenizer.ggml.model", tokenizer_type)?;

            // Write minimal tokenizer data based on type
            if tokenizer_type == "hf" {
                write_string_kv(
                    &mut writer,
                    "tokenizer.ggml.tokens",
                    &vec!["<s>".to_string(), "</s>".to_string(), "<unk>".to_string()]
                        .iter()
                        .take(config.vocab_size.min(100) as usize)
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join("\n"),
                )?;
            } else if tokenizer_type == "spm" {
                write_uint32_kv(&mut writer, "tokenizer.ggml.bos_token_id", 1)?;
                write_uint32_kv(&mut writer, "tokenizer.ggml.eos_token_id", 2)?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

/// Write a string key-value pair
fn write_string_kv(writer: &mut BufWriter<File>, key: &str, value: &str) -> IoResult<()> {
    // Key
    writer.write_all(&(key.len() as u64).to_le_bytes())?;
    writer.write_all(key.as_bytes())?;

    // Type (8 = string)
    writer.write_all(&8u32.to_le_bytes())?;

    // Value
    writer.write_all(&(value.len() as u64).to_le_bytes())?;
    writer.write_all(value.as_bytes())?;

    Ok(())
}

/// Write a uint32 key-value pair
fn write_uint32_kv(writer: &mut BufWriter<File>, key: &str, value: u32) -> IoResult<()> {
    // Key
    writer.write_all(&(key.len() as u64).to_le_bytes())?;
    writer.write_all(key.as_bytes())?;

    // Type (5 = uint32)
    writer.write_all(&5u32.to_le_bytes())?;

    // Value
    writer.write_all(&value.to_le_bytes())?;

    Ok(())
}

/// Generate all required GGUF fixtures for tests
pub fn generate_all_gguf_fixtures() -> IoResult<()> {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gguf");
    std::fs::create_dir_all(&fixtures_dir)?;

    // AC1: Embedded tokenizer fixtures
    generate_gguf_fixture(
        &fixtures_dir.join("llama3-with-hf-tokenizer.gguf"),
        &GgufFixtureConfig {
            model_type: "llama".to_string(),
            vocab_size: 128256,
            has_embedded_tokenizer: true,
            tokenizer_type: Some("hf".to_string()),
            corrupted: false,
        },
    )?;

    generate_gguf_fixture(
        &fixtures_dir.join("llama2-with-sentencepiece.gguf"),
        &GgufFixtureConfig {
            model_type: "llama".to_string(),
            vocab_size: 32000,
            has_embedded_tokenizer: true,
            tokenizer_type: Some("spm".to_string()),
            corrupted: false,
        },
    )?;

    // AC2: Architecture detection fixtures
    let architectures = [
        ("bitnet", "bitnet-b1.58-2B.gguf", 32000),
        ("llama", "llama2-7b.gguf", 32000),
        ("llama", "llama3-8b.gguf", 128256),
        ("gpt2", "gpt2-medium.gguf", 50257),
        ("gptneox", "gpt-neo-1.3b.gguf", 50257),
        ("bert", "bert-base-uncased.gguf", 30522),
        ("t5", "t5-base.gguf", 32128),
    ];

    for (arch, filename, vocab) in &architectures {
        generate_gguf_fixture(
            &fixtures_dir.join(filename),
            &GgufFixtureConfig {
                model_type: arch.to_string(),
                vocab_size: *vocab,
                has_embedded_tokenizer: false,
                tokenizer_type: None,
                corrupted: false,
            },
        )?;
    }

    // AC3: Vocabulary size fixtures
    let vocab_sizes = [1000u32, 32000, 50257, 128256];
    for vocab in &vocab_sizes {
        generate_gguf_fixture(
            &fixtures_dir.join(format!("vocab-{}.gguf", vocab)),
            &GgufFixtureConfig {
                model_type: "llama".to_string(),
                vocab_size: *vocab,
                has_embedded_tokenizer: false,
                tokenizer_type: None,
                corrupted: false,
            },
        )?;
    }

    // Edge cases: corrupted and missing metadata
    generate_gguf_fixture(
        &fixtures_dir.join("corrupted-embedded-tokenizer.gguf"),
        &GgufFixtureConfig {
            model_type: "llama".to_string(),
            vocab_size: 32000,
            has_embedded_tokenizer: false,
            tokenizer_type: None,
            corrupted: true,
        },
    )?;

    // Standard vocabulary size test files
    generate_gguf_fixture(
        &fixtures_dir.join("llama3-128k.gguf"),
        &GgufFixtureConfig {
            model_type: "llama".to_string(),
            vocab_size: 128256,
            has_embedded_tokenizer: true,
            tokenizer_type: Some("hf".to_string()),
            corrupted: false,
        },
    )?;

    generate_gguf_fixture(
        &fixtures_dir.join("llama2-32k.gguf"),
        &GgufFixtureConfig {
            model_type: "llama".to_string(),
            vocab_size: 32000,
            has_embedded_tokenizer: true,
            tokenizer_type: Some("spm".to_string()),
            corrupted: false,
        },
    )?;

    generate_gguf_fixture(
        &fixtures_dir.join("gpt2-50k.gguf"),
        &GgufFixtureConfig {
            model_type: "gpt2".to_string(),
            vocab_size: 50257,
            has_embedded_tokenizer: true,
            tokenizer_type: Some("hf".to_string()),
            corrupted: false,
        },
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_minimal_gguf() {
        let temp_dir = TempDir::new().unwrap();
        let test_path = temp_dir.path().join("test.gguf");

        let config = GgufFixtureConfig {
            model_type: "llama".to_string(),
            vocab_size: 32000,
            has_embedded_tokenizer: true,
            tokenizer_type: Some("hf".to_string()),
            corrupted: false,
        };

        generate_gguf_fixture(&test_path, &config).unwrap();
        assert!(test_path.exists());

        // Verify it's a valid GGUF file
        let contents = std::fs::read(&test_path).unwrap();
        assert_eq!(&contents[0..4], b"GGUF");
    }
}
