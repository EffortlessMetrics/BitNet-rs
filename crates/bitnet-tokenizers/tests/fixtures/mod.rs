//! Test Fixtures for Universal Tokenizer Discovery (Issue #336)
//!
//! This module provides realistic test data and mock infrastructure for comprehensive
//! tokenizer discovery testing including GGUF models, HuggingFace tokenizers, and
//! SentencePiece models.

#![cfg(test)]

pub mod gguf_fixtures;
pub mod tokenizer_fixtures;

#[cfg(feature = "cpu")]
pub mod mock;

use std::path::PathBuf;

/// Get the path to a test fixture file
pub fn fixture_path(relative_path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(relative_path)
}

/// Check if a fixture exists
pub fn fixture_exists(relative_path: &str) -> bool {
    fixture_path(relative_path).exists()
}

/// Load fixture contents as bytes
#[allow(dead_code)]
pub fn load_fixture_bytes(relative_path: &str) -> std::io::Result<Vec<u8>> {
    std::fs::read(fixture_path(relative_path))
}

/// Load fixture contents as string
#[allow(dead_code)]
pub fn load_fixture_string(relative_path: &str) -> std::io::Result<String> {
    std::fs::read_to_string(fixture_path(relative_path))
}
