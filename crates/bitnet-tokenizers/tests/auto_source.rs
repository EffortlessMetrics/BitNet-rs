use anyhow::Result;
use bitnet_tokenizers::auto::{TokenizerSource, resolve_tokenizer};
use tempfile::TempDir;

fn write_model_stub(dir: &TempDir) -> std::path::PathBuf {
    let model_path = dir.path().join("model.gguf");
    std::fs::write(&model_path, b"GGUF\x03\x00\x00\x00").expect("write model stub");
    model_path
}

fn write_tokenizer(path: &std::path::Path) {
    let json = include_str!("fixtures/minimal_tokenizer.json");
    std::fs::write(path, json).expect("write tokenizer");
}

#[test]
fn explicit_tokenizer_source_is_recorded() -> Result<()> {
    let dir = TempDir::new()?;
    let model_path = write_model_stub(&dir);
    let explicit_path = dir.path().join("custom-tokenizer.json");
    write_tokenizer(&explicit_path);

    let resolved = resolve_tokenizer(&model_path, Some(&explicit_path), true)?;

    assert_eq!(resolved.source, TokenizerSource::Explicit);
    assert_eq!(resolved.path.as_deref(), Some(explicit_path.as_path()));
    assert_eq!(resolved.source.as_str(), "explicit");
    assert!(resolved.tokenizer.vocab_size() > 0);
    Ok(())
}

#[test]
fn sibling_tokenizer_source_is_recorded() -> Result<()> {
    let dir = TempDir::new()?;
    let model_path = write_model_stub(&dir);
    let sibling_path = dir.path().join("tokenizer.json");
    write_tokenizer(&sibling_path);

    let resolved = resolve_tokenizer(&model_path, None, true)?;

    assert_eq!(resolved.source, TokenizerSource::Sibling);
    assert_eq!(resolved.path.as_deref(), Some(sibling_path.as_path()));
    assert_eq!(resolved.source.as_str(), "sibling");
    assert!(resolved.tokenizer.vocab_size() > 0);
    Ok(())
}

#[test]
fn missing_tokenizer_fails_without_mock_or_basic_fallback() -> Result<()> {
    let dir = TempDir::new()?;
    let model_path = write_model_stub(&dir);

    let error = match resolve_tokenizer(&model_path, None, true) {
        Ok(_) => panic!("missing tokenizer must fail"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(message.contains("No tokenizer found"), "unexpected error: {message}");
    assert!(message.contains("--tokenizer <path>"), "error must be actionable: {message}");
    Ok(())
}
