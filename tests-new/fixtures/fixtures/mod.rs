//! BitNet.rs Test Fixtures Infrastructure
//!
//! Comprehensive test fixtures for GGUF weight loading validation (Issue #159).
//! Provides realistic test data for neural network components with proper
//! feature-gated compilation and device-aware testing.

pub mod cross_validation_data;
pub mod fixture_loader;
pub mod gguf_generator;
pub mod integration;
pub mod network_mocks;
pub mod quantization_test_vectors;
pub mod tokenizer_fixtures;

// Quantization test fixtures
pub mod quantization {
    pub mod bitnet_quantization_fixtures;
}

// Cross-validation fixtures
pub mod crossval {
    pub mod cpp_reference_mocks;
}

// Re-export types needed by test files
pub use fixture_loader::{FixtureConfig, TestTier};

use anyhow::{Context, Result};
use candle_core::Tensor as CandleTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// GGUF model test fixture specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufModelFixture {
    pub name: String,
    pub path: PathBuf,
    pub description: String,
    pub model_type: String,
    pub quantization_types: Vec<String>,
    pub expected_tensors: HashMap<String, TensorMetadata>,
    pub validation_rules: ValidationRules,
}

/// Tensor metadata for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub quantization_type: Option<String>,
    pub device_placement: String,
    pub non_zero_required: bool,
    pub alignment_bytes: usize,
}

/// Validation rules for fixture testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRules {
    pub require_all_tensors: bool,
    pub check_tensor_shapes: bool,
    pub validate_quantization_accuracy: bool,
    pub min_accuracy_threshold: f32,
    pub memory_efficiency_check: bool,
    pub performance_baseline_ms: Option<u64>,
}

/// Get fixtures directory path
pub fn get_fixtures_dir() -> PathBuf {
    if let Ok(custom_dir) = std::env::var("BITNET_FIXTURES_DIR") {
        PathBuf::from(custom_dir)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
    }
}

/// Get valid GGUF test files
pub fn get_valid_gguf_files() -> Vec<PathBuf> {
    let fixtures_dir = get_fixtures_dir();
    let gguf_dir = fixtures_dir.join("gguf/valid");

    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&gguf_dir) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "gguf" {
                    files.push(entry.path());
                }
            }
        }
    }

    files
}

/// Get invalid GGUF test files
pub fn get_invalid_gguf_files() -> Vec<PathBuf> {
    let fixtures_dir = get_fixtures_dir();
    let gguf_dir = fixtures_dir.join("gguf/invalid");

    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&gguf_dir) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "gguf" {
                    files.push(entry.path());
                }
            }
        }
    }

    files
}

/// Validate fixtures are available
pub fn validate_fixtures_available() -> Result<()> {
    let fixtures_dir = get_fixtures_dir();

    if !fixtures_dir.exists() {
        return Err(anyhow::anyhow!("Fixtures directory not found: {}", fixtures_dir.display()));
    }

    // Check for valid GGUF files
    let valid_files = get_valid_gguf_files();
    if valid_files.is_empty() {
        return Err(anyhow::anyhow!("No valid GGUF fixtures found"));
    }

    // Check for invalid GGUF files
    let invalid_files = get_invalid_gguf_files();
    if invalid_files.is_empty() {
        return Err(anyhow::anyhow!("No invalid GGUF fixtures found"));
    }

    Ok(())
}

/// Configuration for test environment (scaffolding)
#[derive(Debug, Clone)]
pub struct TestEnvironmentConfig {
    pub tier: TestTier,
    pub gpu_enabled: bool,
    pub strict_mode: bool,
}

impl TestEnvironmentConfig {
    pub fn gpu_features_enabled(&self) -> bool {
        self.gpu_enabled && !self.strict_mode
    }

    pub fn from_env() -> Self {
        let tier = if std::env::var("CI").is_ok() { TestTier::Fast } else { TestTier::Standard };

        Self {
            tier,
            gpu_enabled: false,
            strict_mode: std::env::var("BITNET_STRICT_NO_FAKE_GPU").is_ok(),
        }
    }
}

/// Real model integration fixtures (scaffolding)
pub struct RealModelIntegrationFixtures {
    pub config: TestEnvironmentConfig,
}

impl RealModelIntegrationFixtures {
    pub fn new() -> Self {
        let tier = if std::env::var("CI").is_ok() { TestTier::Fast } else { TestTier::Standard };

        Self {
            config: TestEnvironmentConfig {
                tier,
                gpu_enabled: false,
                strict_mode: std::env::var("BITNET_STRICT_NO_FAKE_GPU").is_ok(),
            },
        }
    }

    pub async fn initialize(&self) -> Result<()> {
        // Initialize real model integration fixtures
        // This is a scaffolding method for future real model testing
        Ok(())
    }

    pub fn get_fixtures_for_ac(&self, ac_id: u32) -> Result<ACTestFixtures> {
        match ac_id {
            1 => Ok(ACTestFixtures::AC1 {
                model_fixtures: "Scaffolding model fixtures".to_string(),
                device_fixtures: "Scaffolding device fixtures".to_string(),
            }),
            2 => Ok(ACTestFixtures::AC2 {
                quantization_fixtures: "Scaffolding quantization fixtures".to_string(),
                device_fixtures: "Scaffolding device fixtures".to_string(),
            }),
            3 => Ok(ACTestFixtures::AC3 {
                device_fixtures: "Scaffolding device fixtures".to_string(),
                performance_fixtures: "Scaffolding performance fixtures".to_string(),
            }),
            4 | 5 => Ok(ACTestFixtures::AC4_5 {
                cross_validation_fixtures: "Scaffolding cross-validation fixtures".to_string(),
                performance_fixtures: "Scaffolding performance fixtures".to_string(),
            }),
            6..=10 => Ok(ACTestFixtures::AC6_10 {
                integration_fixtures: "Scaffolding integration fixtures".to_string(),
                validation_fixtures: "Scaffolding validation fixtures".to_string(),
                environment_fixtures: "Scaffolding environment fixtures".to_string(),
            }),
            _ => Err(anyhow::anyhow!("Unknown AC ID: {}", ac_id)),
        }
    }
}

/// Acceptance criteria test fixtures (scaffolding)
pub enum ACTestFixtures {
    AC1 {
        model_fixtures: String,
        device_fixtures: String,
    },
    AC2 {
        quantization_fixtures: String,
        device_fixtures: String,
    },
    AC3 {
        device_fixtures: String,
        performance_fixtures: String,
    },
    AC4_5 {
        cross_validation_fixtures: String,
        performance_fixtures: String,
    },
    AC6_10 {
        integration_fixtures: String,
        validation_fixtures: String,
        environment_fixtures: String,
    },
}
