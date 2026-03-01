use anyhow::Result;
use bitnet_cli::tokenizer_discovery::resolve_tokenizer;
use std::fs;
use tempfile::TempDir;

fn create_mock_tokenizer() -> String {
    r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {},
    "merges": []
  }
}"#
    .to_string()
}

fn create_mock_gguf() -> Vec<u8> {
    b"GGUF\x03\x00\x00\x00".to_vec()
}

#[test]
fn explicit_path_takes_precedence() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let model_path = temp_dir.path().join("model.gguf");
    let sibling_tokenizer = temp_dir.path().join("tokenizer.json");
    let explicit_tokenizer = temp_dir.path().join("explicit_tokenizer.json");

    fs::write(&model_path, create_mock_gguf())?;
    fs::write(&sibling_tokenizer, create_mock_tokenizer())?;
    fs::write(&explicit_tokenizer, create_mock_tokenizer())?;

    let result = resolve_tokenizer(&model_path, Some(explicit_tokenizer.clone()))?;
    assert_eq!(result.canonicalize()?, explicit_tokenizer.canonicalize()?);
    Ok(())
}

#[test]
fn sibling_precedes_parent() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let model_dir = temp_dir.path().join("models");
    fs::create_dir(&model_dir)?;

    let model_path = model_dir.join("model.gguf");
    let sibling_tokenizer = model_dir.join("tokenizer.json");
    let parent_tokenizer = temp_dir.path().join("tokenizer.json");

    fs::write(&model_path, create_mock_gguf())?;
    fs::write(&sibling_tokenizer, create_mock_tokenizer())?;
    fs::write(&parent_tokenizer, create_mock_tokenizer())?;

    let result = resolve_tokenizer(&model_path, None)?;
    assert_eq!(result.canonicalize()?, sibling_tokenizer.canonicalize()?);
    Ok(())
}

#[test]
fn clear_error_when_not_found() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let model_path = temp_dir.path().join("model.gguf");
    fs::write(&model_path, create_mock_gguf())?;

    let error = resolve_tokenizer(&model_path, None).expect_err("should fail when no tokenizer");
    let msg = error.to_string();
    assert!(msg.contains("Tokenizer not found"));
    assert!(msg.contains("--tokenizer /path/to/tokenizer.json"));
    Ok(())
}
