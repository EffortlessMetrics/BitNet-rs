//! Integration test fixtures and utilities
//!
//! Provides fixtures for comprehensive integration testing of GGUF weight loading
//! across different devices, model types, and performance scenarios.

pub mod devices;
pub mod models;
pub mod performance;

use anyhow::Result;
use std::path::{Path, PathBuf};

/// Get integration fixtures directory path
pub fn get_integration_fixtures_dir() -> PathBuf {
    super::get_fixtures_dir().join("integration")
}

/// Validate all integration fixtures are available
pub fn validate_integration_fixtures() -> Result<()> {
    let integration_dir = get_integration_fixtures_dir();

    if !integration_dir.exists() {
        return Err(anyhow::anyhow!(
            "Integration fixtures directory not found: {}",
            integration_dir.display()
        ));
    }

    // Check for required subdirectories
    let required_dirs = ["devices", "models", "performance"];
    for dir_name in &required_dirs {
        let dir_path = integration_dir.join(dir_name);
        if !dir_path.exists() {
            return Err(anyhow::anyhow!(
                "Required integration fixture directory not found: {}",
                dir_path.display()
            ));
        }
    }

    Ok(())
}
