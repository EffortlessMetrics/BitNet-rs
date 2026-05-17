//! Error type for benchmark receipt parsing, persistence, and validation.

/// Errors from receipt I/O and serialization.
#[derive(Debug, thiserror::Error)]
pub enum ReceiptError {
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("validation error: {0}")]
    Validation(String),
}
