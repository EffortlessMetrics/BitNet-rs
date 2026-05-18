//! Checkpoint metadata extraction and file hashing.

use crate::checkpoint::error::CheckpointError;
use crate::checkpoint::format::CheckpointFormat;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Metadata associated with a single model checkpoint file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Detected or overridden checkpoint format.
    pub format: CheckpointFormat,
    /// Model name (derived from the file stem by default).
    pub model_name: String,
    /// Optional version string.
    pub version: Option<String>,
    /// Timestamp when the metadata entry was created.
    pub created_at: SystemTime,
    /// File size in bytes.
    pub file_size: u64,
    /// SHA-256 hex digest of the file contents.
    pub hash: String,
    /// Canonical path to the checkpoint file.
    pub path: PathBuf,
    /// Last modification time reported by the filesystem.
    pub modified_at: Option<SystemTime>,
}

/// Compute the SHA-256 hex digest of a file using a streaming 1 MiB buffer.
pub fn compute_sha256(path: &Path) -> Result<String, CheckpointError> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Build [`CheckpointMetadata`] by inspecting a file on disk.
pub fn extract_metadata(path: &Path) -> Result<CheckpointMetadata, CheckpointError> {
    let meta = std::fs::metadata(path)?;
    let hash = compute_sha256(path)?;
    let format = CheckpointFormat::detect(path);
    let modified_at = meta.modified().ok();
    let canonical_path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());

    Ok(CheckpointMetadata {
        format,
        model_name: model_name_from_path(path),
        version: None,
        created_at: SystemTime::now(),
        file_size: meta.len(),
        hash,
        path: canonical_path,
        modified_at,
    })
}

/// Derive a model name from a file path (uses the file stem).
fn model_name_from_path(path: &Path) -> String {
    path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string()
}
