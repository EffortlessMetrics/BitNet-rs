use anyhow::{Context, Result};
use std::path::PathBuf;

pub fn validate_explicit_path(path: PathBuf) -> Result<PathBuf> {
    if !path.exists() {
        anyhow::bail!(
            "Explicit tokenizer path does not exist: {}\n\
             \n\
             Please provide a valid tokenizer.json file path.",
            path.display()
        );
    }

    if !path.is_file() {
        anyhow::bail!("Explicit tokenizer path is not a file: {}", path.display());
    }

    path.canonicalize().context("Failed to canonicalize explicit tokenizer path")
}
