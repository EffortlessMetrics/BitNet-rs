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

use bitnet_test_fixtures_core::{
    fixture_path_from_manifest, load_fixture_bytes as load_bytes,
    load_fixture_string as load_string,
};
use std::path::Path;

/// Get the path to a test fixture file
pub fn fixture_path(relative_path: &str) -> std::path::PathBuf {
    fixture_path_from_manifest(
        Path::new(env!("CARGO_MANIFEST_DIR")),
        Path::new("tests/fixtures").join(relative_path),
    )
}

/// Check if a fixture exists
#[allow(dead_code)]
pub fn fixture_exists(relative_path: &str) -> bool {
    fixture_path(relative_path).exists()
}

/// Load fixture contents as bytes
#[allow(dead_code)]
pub fn load_fixture_bytes(relative_path: &str) -> std::io::Result<Vec<u8>> {
    load_bytes(fixture_path(relative_path))
}

/// Load fixture contents as string
#[allow(dead_code)]
pub fn load_fixture_string(relative_path: &str) -> std::io::Result<String> {
    load_string(fixture_path(relative_path))
}
