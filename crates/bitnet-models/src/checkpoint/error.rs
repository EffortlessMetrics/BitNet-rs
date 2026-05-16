//! Error types for checkpoint management.

/// Errors specific to checkpoint management.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("checkpoint not found: {0}")]
    NotFound(String),
    #[error("duplicate checkpoint: {0}")]
    Duplicate(String),
    #[error("hash mismatch for {path}: expected {expected}, got {actual}")]
    HashMismatch { path: String, expected: String, actual: String },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
