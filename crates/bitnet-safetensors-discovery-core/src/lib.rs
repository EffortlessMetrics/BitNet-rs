//! Pure discovery helpers for resolving SafeTensors inputs.

use std::path::{Path, PathBuf};
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Debug, Error)]
pub enum SafetensorsDiscoveryError {
    #[error("input path does not exist: {0}")]
    InputDoesNotExist(PathBuf),
    #[error("input file is not .safetensors: {0}")]
    InputIsNotSafetensors(PathBuf),
    #[error("no .safetensors files found in {0}")]
    NoSafetensorsFiles(PathBuf),
    #[error(transparent)]
    Walkdir(#[from] walkdir::Error),
}

/// Collect all `.safetensors` files from a single file input or a directory.
///
/// For directory input, only depth-1 files are considered and output paths are sorted.
pub fn collect_safetensors_files(input: &Path) -> Result<Vec<PathBuf>, SafetensorsDiscoveryError> {
    if !input.exists() {
        return Err(SafetensorsDiscoveryError::InputDoesNotExist(input.to_path_buf()));
    }

    let mut out = vec![];
    if input.is_file() {
        if is_safetensors(input) {
            out.push(input.to_path_buf());
        } else {
            return Err(SafetensorsDiscoveryError::InputIsNotSafetensors(input.to_path_buf()));
        }
    } else {
        for entry in WalkDir::new(input).min_depth(1).max_depth(1) {
            let entry = entry?;
            if entry.file_type().is_file() {
                let path = entry.path();
                if is_safetensors(path) {
                    out.push(path.to_path_buf());
                }
            }
        }
    }

    if out.is_empty() {
        return Err(SafetensorsDiscoveryError::NoSafetensorsFiles(input.to_path_buf()));
    }

    out.sort();
    Ok(out)
}

/// Resolve the first `.safetensors` file for an input file or directory.
pub fn resolve_first_safetensors_file(input: &Path) -> Result<PathBuf, SafetensorsDiscoveryError> {
    let mut files = collect_safetensors_files(input)?;
    Ok(files.remove(0))
}

fn is_safetensors(path: &Path) -> bool {
    path.extension().and_then(|s| s.to_str()) == Some("safetensors")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn collects_from_single_file() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("model.safetensors");
        fs::write(&file, b"x").unwrap();

        let files = collect_safetensors_files(&file).unwrap();
        assert_eq!(files, vec![file]);
    }

    #[test]
    fn rejects_non_safetensors_single_file() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("model.bin");
        fs::write(&file, b"x").unwrap();

        let err = collect_safetensors_files(&file).unwrap_err();
        assert!(matches!(err, SafetensorsDiscoveryError::InputIsNotSafetensors(_)));
    }

    #[test]
    fn collects_depth_one_files_sorted() {
        let dir = tempdir().unwrap();
        let b = dir.path().join("b.safetensors");
        let a = dir.path().join("a.safetensors");
        let subdir = dir.path().join("nested");
        fs::create_dir_all(&subdir).unwrap();
        let nested = subdir.join("nested.safetensors");
        fs::write(&a, b"a").unwrap();
        fs::write(&b, b"b").unwrap();
        fs::write(&nested, b"n").unwrap();

        let files = collect_safetensors_files(dir.path()).unwrap();
        assert_eq!(files, vec![a, b]);
    }

    #[test]
    fn resolve_first_returns_sorted_first() {
        let dir = tempdir().unwrap();
        let b = dir.path().join("b.safetensors");
        let a = dir.path().join("a.safetensors");
        fs::write(&a, b"a").unwrap();
        fs::write(&b, b"b").unwrap();

        let first = resolve_first_safetensors_file(dir.path()).unwrap();
        assert_eq!(first, a);
    }
}
