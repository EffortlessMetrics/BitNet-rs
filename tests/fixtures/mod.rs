//! Test fixtures for real BitNet model integration tests
//!
//! This module provides comprehensive test fixtures for BitNet.rs neural network components,
//! supporting Issue #218 "MVP Requirement: Real BitNet Model Integration and Validation".
//!
//! ## Architecture Overview
//!
//! The fixtures are organized into three tiers for efficient CI testing:
//!
//! ### Tier 1: Fast Validation (Mock Models)
//! - Lightweight mock implementations for rapid feedback
//! - Suitable for unit tests and quick validation
//! - No external dependencies or large files
//!
//! ### Tier 2: Standard Validation (Cached Real Models)
//! - Real model artifacts cached for consistent testing
//! - Production-grade validation with known test vectors
//! - Intelligent caching and fallback strategies
//!
//! ### Tier 3: Full Validation (Cross-Validation)
//! - Complete cross-validation against C++ implementation
//! - Numerical accuracy testing with configurable tolerances
//! - Performance benchmarking and regression detection

// Allow warnings for test scaffolding code
#![allow(dead_code, unused_variables, unused_imports, clippy::all)]

// Existing fixture modules
pub mod cross_validation;
pub mod device_aware;
pub mod error_handling;
pub mod model_artifacts;
pub mod performance;
pub mod quantization;

// New comprehensive fixture modules for Issue #249 tokenizer testing
pub mod cross_validation_data;
pub mod fixture_loader;
pub mod network_mocks;
pub mod quantization_test_vectors;
pub mod simple_validation;
pub mod tokenizer_fixtures;

// Re-export common fixture types
pub use cross_validation::CrossValidationFixtures;
pub use device_aware::DeviceAwareFixtures;
pub use error_handling::ErrorHandlingFixtures;
pub use model_artifacts::ModelFixtures;
pub use performance::PerformanceFixtures;
pub use quantization::QuantizationFixtures;

// Re-export new tokenizer testing fixture types
pub use cross_validation_data::{
    CrossValidationFixtures as TokenizerCrossValidationFixtures, CrossValidationTestCase,
};
pub use fixture_loader::{
    FixtureConfig, FixtureLoadResult, FixtureLoader, TestTier as FixtureTier,
};
pub use network_mocks::{HuggingFaceApiResponse, NetworkMockFixtures, NetworkTestScenario};
pub use quantization_test_vectors::{
    DeviceValidationData, MixedPrecisionTestData, QuantizationTestVector,
};
pub use tokenizer_fixtures::{
    MockGgufModel, TokenizerFixtures, TokenizerTestFixture, TokenizerType,
};

use bitnet_common::{Device, Result};
use std::env;
use std::path::PathBuf;
use std::time::Duration;

/// Main fixture manager for real model integration tests
pub struct RealModelIntegrationFixtures {
    pub model_fixtures: ModelFixtures,
    pub device_fixtures: DeviceAwareFixtures,
    pub quantization_fixtures: QuantizationFixtures,
    pub cross_validation_fixtures: CrossValidationFixtures,
    pub performance_fixtures: PerformanceFixtures,
    #[allow(dead_code)]
    pub error_handling_fixtures: ErrorHandlingFixtures,
    pub config: TestEnvironmentConfig,
}

/// Test environment configuration
#[derive(Debug, Clone)]
pub struct TestEnvironmentConfig {
    pub model_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub use_real_models: bool,
    pub device_preference: Device,
    #[allow(dead_code)]
    pub strict_mode: bool,
    #[allow(dead_code)]
    pub timeout: Duration,
    pub tier: TestTier,
}

/// Test execution tier for CI optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TestTier {
    /// Fast mock-based testing
    Fast,
    /// Standard testing with cached models
    Standard,
    /// Full cross-validation testing
    Full,
}

impl TestEnvironmentConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let model_path = env::var("BITNET_GGUF").ok().map(PathBuf::from);
        let tokenizer_path = env::var("BITNET_TOKENIZER").ok().map(PathBuf::from);

        let device_preference = match env::var("BITNET_DEVICE").as_deref() {
            Ok("cpu") => Device::Cpu,
            Ok("cuda") | Ok("gpu") => Device::Cuda(0),
            _ => Device::Cpu, // Default to CPU
        };

        let tier = match env::var("BITNET_TEST_TIER").as_deref() {
            Ok("fast") => TestTier::Fast,
            Ok("standard") => TestTier::Standard,
            Ok("full") => TestTier::Full,
            _ => {
                // Auto-detect based on CI environment and model availability
                if env::var("CI").is_ok() {
                    if model_path.is_some() { TestTier::Standard } else { TestTier::Fast }
                } else {
                    TestTier::Standard
                }
            }
        };

        let use_real_models = tier != TestTier::Fast && model_path.is_some();

        Self {
            model_path,
            tokenizer_path,
            use_real_models,
            device_preference,
            strict_mode: env::var("BITNET_STRICT_TOKENIZERS").map(|v| v == "1").unwrap_or(false)
                || env::var("BITNET_STRICT_NO_FAKE_GPU").map(|v| v == "1").unwrap_or(false),
            timeout: Duration::from_secs(
                env::var("BITNET_TEST_TIMEOUT").ok().and_then(|s| s.parse().ok()).unwrap_or(300), // 5 minutes default
            ),
            tier,
        }
    }

    /// Check if real model testing is enabled and available
    pub fn real_models_available(&self) -> bool {
        self.use_real_models && self.model_path.as_ref().map(|p| p.exists()).unwrap_or(false)
    }

    /// Get model path or skip test with appropriate message
    #[allow(dead_code)]
    pub fn get_model_path_or_skip(&self) -> PathBuf {
        match &self.model_path {
            Some(path) if path.exists() => path.clone(),
            Some(path) => {
                panic!(
                    "Model file not found at {}, set BITNET_GGUF or use BITNET_TEST_TIER=fast",
                    path.display()
                );
            }
            None => {
                match self.tier {
                    TestTier::Fast => {
                        // Return mock model path
                        PathBuf::from("tests/fixtures/mock_model.gguf")
                    }
                    _ => {
                        if env::var("CI").is_ok() {
                            panic!("Real model tests require BITNET_GGUF in CI environment");
                        }
                        // For local development, skip gracefully
                        eprintln!(
                            "Skipping real model test - set BITNET_GGUF environment variable"
                        );
                        std::process::exit(0);
                    }
                }
            }
        }
    }

    /// Check if GPU features should be tested
    pub fn gpu_features_enabled(&self) -> bool {
        cfg!(feature = "gpu") && !self.strict_mode_no_fake_gpu()
    }

    /// Check if strict mode prevents fake GPU backends
    fn strict_mode_no_fake_gpu(&self) -> bool {
        env::var("BITNET_STRICT_NO_FAKE_GPU").map(|v| v == "1").unwrap_or(false)
    }
}

impl RealModelIntegrationFixtures {
    /// Create new fixture manager with environment-based configuration
    pub fn new() -> Self {
        let config = TestEnvironmentConfig::from_env();

        Self {
            model_fixtures: ModelFixtures::new(&config),
            device_fixtures: DeviceAwareFixtures::new(&config),
            quantization_fixtures: QuantizationFixtures::new(&config),
            cross_validation_fixtures: CrossValidationFixtures::new(&config),
            performance_fixtures: PerformanceFixtures::new(&config),
            error_handling_fixtures: ErrorHandlingFixtures::new(),
            config,
        }
    }

    /// Initialize all fixtures and prepare test environment
    pub async fn initialize(&mut self) -> Result<()> {
        // Initialize in dependency order
        self.model_fixtures.initialize().await?;
        self.device_fixtures.initialize().await?;
        self.quantization_fixtures.initialize(&self.model_fixtures).await?;
        self.performance_fixtures.initialize().await?;
        self.error_handling_fixtures.initialize().await?;

        // Cross-validation for all tiers (with different levels of detail)
        self.cross_validation_fixtures.initialize(&self.model_fixtures).await?;

        Ok(())
    }

    /// Cleanup all fixtures and test resources
    #[allow(dead_code)]
    pub async fn cleanup(&mut self) -> Result<()> {
        // Cleanup in reverse order
        self.cross_validation_fixtures.cleanup().await?;
        self.error_handling_fixtures.cleanup().await?;
        self.performance_fixtures.cleanup().await?;
        self.quantization_fixtures.cleanup().await?;
        self.device_fixtures.cleanup().await?;
        self.model_fixtures.cleanup().await?;

        Ok(())
    }

    /// Get fixtures appropriate for the given acceptance criteria
    #[allow(dead_code)]
    pub fn get_fixtures_for_ac(&self, ac_id: u8) -> Result<ACTestFixtures<'_>> {
        match ac_id {
            1 => Ok(ACTestFixtures::AC1 {
                model_fixtures: &self.model_fixtures,
                device_fixtures: &self.device_fixtures,
            }),
            2 => Ok(ACTestFixtures::AC2 {
                quantization_fixtures: &self.quantization_fixtures,
                device_fixtures: &self.device_fixtures,
            }),
            3 => Ok(ACTestFixtures::AC3 {
                device_fixtures: &self.device_fixtures,
                performance_fixtures: &self.performance_fixtures,
            }),
            4..=5 => Ok(ACTestFixtures::AC4_5 {
                cross_validation_fixtures: &self.cross_validation_fixtures,
                performance_fixtures: &self.performance_fixtures,
            }),
            6..=10 => Ok(ACTestFixtures::AC6_10 {
                model_fixtures: &self.model_fixtures,
                error_handling_fixtures: &self.error_handling_fixtures,
                performance_fixtures: &self.performance_fixtures,
            }),
            _ => Err(bitnet_common::BitNetError::Validation(format!(
                "Invalid acceptance criteria ID: {}",
                ac_id
            ))),
        }
    }
}

/// Fixtures grouped by acceptance criteria for organized testing
#[allow(dead_code)]
pub enum ACTestFixtures<'a> {
    AC1 {
        model_fixtures: &'a ModelFixtures,
        device_fixtures: &'a DeviceAwareFixtures,
    },
    AC2 {
        quantization_fixtures: &'a QuantizationFixtures,
        device_fixtures: &'a DeviceAwareFixtures,
    },
    AC3 {
        device_fixtures: &'a DeviceAwareFixtures,
        performance_fixtures: &'a PerformanceFixtures,
    },
    AC4_5 {
        cross_validation_fixtures: &'a CrossValidationFixtures,
        performance_fixtures: &'a PerformanceFixtures,
    },
    AC6_10 {
        model_fixtures: &'a ModelFixtures,
        error_handling_fixtures: &'a ErrorHandlingFixtures,
        performance_fixtures: &'a PerformanceFixtures,
    },
}

/// Helper macros for fixture-based testing
#[macro_export]
macro_rules! skip_if_tier_insufficient {
    ($config:expr, $required_tier:expr) => {
        if $config.tier < $required_tier {
            eprintln!(
                "Skipping test - requires tier {:?}, current tier {:?}",
                $required_tier, $config.tier
            );
            return;
        }
    };
}

#[macro_export]
macro_rules! skip_if_no_gpu {
    ($config:expr) => {
        if !$config.gpu_features_enabled() {
            eprintln!("Skipping GPU test - GPU features not enabled or strict mode active");
            return;
        }
    };
}

/// Test helper utilities
pub mod test_utils {
    use super::*;

    /// Create a temporary test environment with cleanup
    pub struct TempTestEnv {
        pub fixtures: RealModelIntegrationFixtures,
        _temp_dir: tempfile::TempDir,
    }

    impl TempTestEnv {
        pub async fn new() -> Result<Self> {
            let temp_dir = tempfile::tempdir().map_err(|e| bitnet_common::BitNetError::Io(e))?;

            let mut fixtures = RealModelIntegrationFixtures::new();
            fixtures.initialize().await?;

            Ok(Self { fixtures, _temp_dir: temp_dir })
        }
    }

    impl Drop for TempTestEnv {
        fn drop(&mut self) {
            // Async cleanup in drop - use tokio::task::block_in_place for test contexts
            let _ = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(self.fixtures.cleanup())
            });
        }
    }

    /// Skip test with appropriate message based on configuration
    pub fn skip_test_with_reason(config: &TestEnvironmentConfig, reason: &str) -> ! {
        eprintln!(
            "SKIPPED: {} (tier: {:?}, real_models: {})",
            reason,
            config.tier,
            config.real_models_available()
        );
        std::process::exit(0);
    }
}
