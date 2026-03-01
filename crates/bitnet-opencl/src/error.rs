//! Error types for model format detection and loading.

use std::fmt;

/// Errors arising from model format detection and loading.
#[derive(Debug)]
pub enum ModelFormatError {
    /// The specified file does not exist.
    FileNotFound(String),
    /// An I/O error occurred while reading the file.
    IoError(String),
    /// The file format could not be determined.
    UnknownFormat { path: String, extension: String, suggestion: String },
    /// The file header is corrupt or truncated.
    CorruptHeader { path: String, position: usize, detail: String },
    /// No loader is registered for the given format.
    NoLoaderRegistered(String),
    /// The loader encountered an error while parsing.
    LoaderError(String),
}

impl fmt::Display for ModelFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileNotFound(path) => {
                write!(f, "model file not found: {path}")
            }
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
            Self::UnknownFormat { path, extension, suggestion } => {
                write!(
                    f,
                    "unknown model format for '{path}' \
                     (extension: .{extension}). {suggestion}"
                )
            }
            Self::CorruptHeader { path, position, detail } => {
                write!(
                    f,
                    "corrupt header in '{path}' at byte {position}: \
                     {detail}"
                )
            }
            Self::NoLoaderRegistered(fmt_name) => {
                write!(f, "no loader registered for format: {fmt_name}")
            }
            Self::LoaderError(msg) => {
                write!(f, "model loader error: {msg}")
            }
        }
    }
}

impl std::error::Error for ModelFormatError {}
