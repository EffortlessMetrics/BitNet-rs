//! Test fixtures for cross-validation
//!
//! This module provides small test models and datasets for comparing
//! Rust and C++ implementations.

use crate::{CrossvalError, Result};
use std::path::{Path, PathBuf};

/// Test fixture containing model and test data
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TestFixture {
    pub name: String,
    pub model_path: PathBuf,
    pub test_prompts: Vec<String>,
    pub expected_tokens: Option<Vec<Vec<u32>>>,
}

impl TestFixture {
    /// Load a test fixture by name
    pub fn load(name: &str) -> Result<Self> {
        let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fixture_path = fixtures_dir.join(format!("{}.json", name));

        if !fixture_path.exists() {
            return Err(CrossvalError::ModelLoadError(format!(
                "Fixture '{}' not found at {:?}",
                name, fixture_path
            )));
        }

        let content = std::fs::read_to_string(&fixture_path)?;
        let fixture: TestFixture = serde_json::from_str(&content)?;

        // Validate that model file exists
        let model_path = if fixture.model_path.is_absolute() {
            fixture.model_path.clone()
        } else {
            fixtures_dir.join(&fixture.model_path)
        };

        if !model_path.exists() {
            return Err(CrossvalError::ModelLoadError(format!(
                "Model file not found: {:?}",
                model_path
            )));
        }

        Ok(TestFixture {
            model_path,
            ..fixture
        })
    }

    /// Get all available fixture names
    pub fn list_available() -> Result<Vec<String>> {
        let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");

        if !fixtures_dir.exists() {
            return Ok(vec![]);
        }

        let mut fixtures = Vec::new();
        for entry in std::fs::read_dir(fixtures_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    fixtures.push(name.to_string());
                }
            }
        }

        fixtures.sort();
        Ok(fixtures)
    }
}

/// Standard test prompts for cross-validation
pub const STANDARD_PROMPTS: &[&str] = &[
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "To be or not to be, that is the question.",
    "1 + 1 = 2",
];

/// Create a minimal test fixture for development
pub fn create_minimal_fixture() -> TestFixture {
    TestFixture {
        name: "minimal".to_string(),
        model_path: PathBuf::from("minimal_model.gguf"),
        test_prompts: STANDARD_PROMPTS.iter().map(|s| s.to_string()).collect(),
        expected_tokens: None,
    }
}
