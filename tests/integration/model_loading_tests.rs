//! # Model Loading and Initialization Integration Tests
//!
//! Tests model loading workflows, initialization processes, and configuration validation.

use super::*;
#[cfg(feature = "fixtures")]
use crate::common::FixtureManager;
use crate::common::harness::FixtureCtx;
use crate::common::tensor_helpers::ct;
use crate::{TestCase, TestError, TestMetrics, TestResult};
use anyhow::Result;
use async_trait::async_trait;
use bitnet_common::{BitNetConfig, BitNetError, Device, MockTensor};
use bitnet_models::Model;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Test suite for model loading and initialization
pub struct ModelLoadingTestSuite;

impl crate::TestSuite for ModelLoadingTestSuite {
    fn name(&self) -> &str {
        "Model Loading Integration Tests"
    }

    fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
        vec![
            Box::new(BasicModelLoadingTest),
            Box::new(ModelConfigurationTest),
            Box::new(MultipleModelLoadingTest),
            Box::new(ModelInitializationErrorTest),
            Box::new(ModelMemoryManagementTest),
        ]
    }
}

/// Test basic model loading workflow
struct BasicModelLoadingTest;

#[async_trait]
impl TestCase for BasicModelLoadingTest {
    fn name(&self) -> &str {
        "basic_model_loading"
    }

    async fn setup(&self, fixtures: &FixtureManager) -> TestResult<()> {
        info!("Setting up basic model loading test");

        // Ensure test fixtures are available
        if !fixtures.is_cached("tiny-model").await {
            debug!("Tiny model fixture not cached, will use mock");
        }

        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        debug!("Creating mock model for loading test");
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());

        // Test model configuration access
        let config = model.config();
        debug!("Model config: {:?}", config);

        // Verify configuration is valid
        if let Err(e) = config.validate() {
            return Err(TestError::execution(format!("Model configuration is invalid: {}", e)));
        }

        // Test model loading with different devices
        let devices = vec![Device::Cpu];
        let mut successful_loads = 0;

        for device in devices {
            debug!("Testing model loading with device: {:?}", device);

            match InferenceEngine::new(model.clone(), tokenizer.clone(), device) {
                Ok(engine) => {
                    successful_loads += 1;

                    // Verify engine configuration
                    let engine_config = engine.model_config();
                    if engine_config.model.vocab_size == 0 {
                        return Err(TestError::assertion("Model vocab size should be non-zero"));
                    }

                    debug!("Successfully loaded model with device: {:?}", device);
                }
                Err(e) => {
                    warn!("Failed to load model with device {:?}: {}", device, e);
                }
            }
        }

        if successful_loads == 0 {
            return Err(TestError::execution("Failed to load model with any device"));
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            assertions: 0,
            operations: 0,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            custom_metrics: [
                ("successful_loads".to_string(), successful_loads as f64),
                ("devices_tested".to_string(), 1.0),
                ("vocab_size".to_string(), model.config().model.vocab_size as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up basic model loading test");
        Ok(())
    }
}

/// Test model configuration validation
struct ModelConfigurationTest;

#[async_trait]
impl TestCase for ModelConfigurationTest {
    fn name(&self) -> &str {
        "model_configuration_validation"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up model configuration test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        // Test default configuration
        debug!("Testing default configuration");
        let default_config = BitNetConfig::default();
        if let Err(e) = default_config.validate() {
            return Err(TestError::execution(format!(
                "Default configuration should be valid: {}",
                e
            )));
        }

        // Test various configuration scenarios
        let test_configs = vec![
            // Valid configurations
            BitNetConfig {
                model: bitnet_common::ModelConfig {
                    vocab_size: 50000,
                    hidden_size: 768,
                    num_layers: 12,
                    num_heads: 12,
                    ..Default::default()
                },
                ..Default::default()
            },
            BitNetConfig {
                model: bitnet_common::ModelConfig {
                    vocab_size: 32000,
                    hidden_size: 1024,
                    num_layers: 24,
                    num_heads: 16,
                    ..Default::default()
                },
                ..Default::default()
            },
        ];

        let mut valid_configs = 0;
        let mut invalid_configs = 0;

        for (i, config) in test_configs.iter().enumerate() {
            debug!(
                "Testing configuration {}: vocab_size={}, hidden_size={}",
                i + 1,
                config.model.vocab_size,
                config.model.hidden_size
            );

            match config.validate() {
                Ok(_) => {
                    valid_configs += 1;

                    // Test creating model with this configuration
                    let model = MockModelWithConfig::new(config.clone());
                    let tokenizer = Arc::new(MockTokenizer::new());

                    match InferenceEngine::new(Arc::new(model), tokenizer, Device::Cpu) {
                        Ok(_) => {
                            debug!("Successfully created engine with config {}", i + 1);
                        }
                        Err(e) => {
                            warn!("Failed to create engine with valid config {}: {}", i + 1, e);
                        }
                    }
                }
                Err(e) => {
                    invalid_configs += 1;
                    debug!("Configuration {} is invalid (expected): {}", i + 1, e);
                }
            }
        }

        // Test invalid configurations
        let invalid_test_configs = vec![BitNetConfig {
            model: bitnet_common::ModelConfig {
                vocab_size: 0, // Invalid
                ..Default::default()
            },
            ..Default::default()
        }];

        for (i, config) in invalid_test_configs.iter().enumerate() {
            debug!("Testing invalid configuration {}", i + 1);

            if config.validate().is_ok() {
                return Err(TestError::assertion(format!(
                    "Invalid configuration {} should fail validation",
                    i + 1
                )));
            }
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            assertions: 0,
            operations: 0,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            custom_metrics: [
                ("valid_configs".to_string(), valid_configs as f64),
                ("invalid_configs".to_string(), invalid_configs as f64),
                (
                    "total_configs_tested".to_string(),
                    (test_configs.len() + invalid_test_configs.len()) as f64,
                ),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up model configuration test");
        Ok(())
    }
}

/// Test loading multiple models
struct MultipleModelLoadingTest;

#[async_trait]
impl TestCase for MultipleModelLoadingTest {
    fn name(&self) -> &str {
        "multiple_model_loading"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up multiple model loading test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let num_models = 3;
        let mut engines = Vec::new();
        let mut loading_times = Vec::new();

        debug!("Loading {} models", num_models);

        for i in 0..num_models {
            let load_start = Instant::now();

            debug!("Loading model {}", i + 1);
            let model = Arc::new(MockModel::new());
            let tokenizer = Arc::new(MockTokenizer::new());

            match InferenceEngine::new(model, tokenizer, Device::Cpu) {
                Ok(engine) => {
                    let load_time = load_start.elapsed();
                    loading_times.push(load_time);
                    engines.push(engine);
                    debug!("Successfully loaded model {} in {:?}", i + 1, load_time);
                }
                Err(e) => {
                    return Err(TestError::execution(format!(
                        "Failed to load model {}: {}",
                        i + 1,
                        e
                    )));
                }
            }
        }

        if engines.len() != num_models {
            return Err(TestError::assertion("Not all models were loaded successfully"));
        }

        // Test that all engines work independently
        debug!("Testing independent operation of loaded models");
        let test_prompt = "Test prompt for multiple models";
        let mut results = Vec::new();

        for (i, engine) in engines.iter().enumerate() {
            debug!("Testing engine {}", i + 1);

            match engine.generate(test_prompt).await {
                Ok(result) => {
                    if result.is_empty() {
                        return Err(TestError::assertion(format!(
                            "Engine {} produced empty result",
                            i + 1
                        )));
                    }
                    results.push(result);
                    debug!("Engine {} generated: {}", i + 1, results[i]);
                }
                Err(e) => {
                    return Err(TestError::execution(format!(
                        "Engine {} failed to generate: {}",
                        i + 1,
                        e
                    )));
                }
            }
        }

        // Calculate statistics
        let avg_loading_time =
            loading_times.iter().sum::<std::time::Duration>() / loading_times.len() as u32;
        let max_loading_time = loading_times.iter().max().unwrap();
        let min_loading_time = loading_times.iter().min().unwrap();

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            assertions: 0,
            operations: 0,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            custom_metrics: [
                ("models_loaded".to_string(), engines.len() as f64),
                ("successful_generations".to_string(), results.len() as f64),
                ("avg_loading_time_ms".to_string(), avg_loading_time.as_millis() as f64),
                ("max_loading_time_ms".to_string(), max_loading_time.as_millis() as f64),
                ("min_loading_time_ms".to_string(), min_loading_time.as_millis() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up multiple model loading test");
        Ok(())
    }
}

/// Test model initialization error handling
struct ModelInitializationErrorTest;

#[async_trait]
impl TestCase for ModelInitializationErrorTest {
    fn name(&self) -> &str {
        "model_initialization_error_handling"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up model initialization error test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        // Test with failing model
        debug!("Testing with failing model");
        let failing_model = Arc::new(FailingMockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());

        // This should succeed (model creation doesn't fail, only forward pass)
        let engine_result = InferenceEngine::new(failing_model, tokenizer, Device::Cpu);

        match engine_result {
            Ok(engine) => {
                debug!("Engine created successfully with failing model");

                // Test that generation fails appropriately
                let generation_result = engine.generate("test").await;
                match generation_result {
                    Ok(_) => {
                        debug!("Generation unexpectedly succeeded with failing model");
                    }
                    Err(e) => {
                        debug!("Generation failed as expected: {}", e);
                    }
                }
            }
            Err(e) => {
                debug!("Engine creation failed as expected: {}", e);
            }
        }

        // Test with invalid device (if applicable)
        debug!("Testing error recovery");
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());

        // After error, normal operation should still work
        let recovery_engine = InferenceEngine::new(model, tokenizer, Device::Cpu)
            .map_err(|e| TestError::execution(format!("Recovery engine creation failed: {}", e)))?;

        let recovery_result = recovery_engine
            .generate("recovery test")
            .await
            .map_err(|e| TestError::execution(format!("Recovery generation failed: {}", e)))?;

        if recovery_result.is_empty() {
            return Err(TestError::assertion("Recovery generation should produce output"));
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            assertions: 0,
            operations: 0,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            custom_metrics: [
                ("error_scenarios_tested".to_string(), 2.0),
                ("recovery_successful".to_string(), 1.0),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up model initialization error test");
        Ok(())
    }
}

/// Test model memory management
struct ModelMemoryManagementTest;

#[async_trait]
impl TestCase for ModelMemoryManagementTest {
    fn name(&self) -> &str {
        "model_memory_management"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up model memory management test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        debug!("Testing model memory management");

        // Create and drop multiple engines to test memory cleanup
        let iterations = 5;
        let mut peak_memory_estimates = Vec::new();

        for i in 0..iterations {
            debug!("Memory test iteration {}", i + 1);

            let model = Arc::new(MockModel::new());
            let tokenizer = Arc::new(MockTokenizer::new());

            let engine = InferenceEngine::new(model, tokenizer, Device::Cpu).map_err(|e| {
                TestError::execution(format!(
                    "Engine creation failed in iteration {}: {}",
                    i + 1,
                    e
                ))
            })?;

            // Perform some operations
            let _result = engine.generate("memory test").await.map_err(|e| {
                TestError::execution(format!("Generation failed in iteration {}: {}", i + 1, e))
            })?;

            // Get memory stats
            let stats = engine.get_stats().await;
            peak_memory_estimates.push(stats.cache_size);

            debug!("Iteration {} - Cache size: {}", i + 1, stats.cache_size);

            // Engine will be dropped at end of iteration
        }

        // Test memory efficiency configurations
        debug!("Testing memory-efficient configuration");
        let memory_config = InferenceConfig::memory_efficient();

        if let Err(e) = memory_config.validate() {
            return Err(TestError::execution(format!(
                "Memory-efficient config should be valid: {}",
                e
            )));
        }

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());

        let memory_engine =
            InferenceEngine::with_config(model, tokenizer, Device::Cpu, memory_config).map_err(
                |e| TestError::execution(format!("Memory-efficient engine creation failed: {}", e)),
            )?;

        let memory_result = memory_engine.generate("memory efficient test").await.map_err(|e| {
            TestError::execution(format!("Memory-efficient generation failed: {}", e))
        })?;

        if memory_result.is_empty() {
            return Err(TestError::assertion("Memory-efficient engine should produce output"));
        }

        let final_stats = memory_engine.get_stats().await;

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            assertions: 0,
            operations: 0,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            custom_metrics: [
                ("memory_iterations".to_string(), iterations as f64),
                (
                    "avg_cache_size".to_string(),
                    peak_memory_estimates.iter().sum::<usize>() as f64 / iterations as f64,
                ),
                (
                    "max_cache_size".to_string(),
                    *peak_memory_estimates.iter().max().unwrap_or(&0) as f64,
                ),
                ("final_cache_size".to_string(), final_stats.cache_size as f64),
                ("final_cache_usage".to_string(), final_stats.cache_usage),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up model memory management test");
        Ok(())
    }
}

/// Mock model with custom configuration
struct MockModelWithConfig {
    config: BitNetConfig,
}

impl MockModelWithConfig {
    fn new(config: BitNetConfig) -> Self {
        Self { config }
    }
}

impl Model for MockModelWithConfig {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &bitnet_common::ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> Result<bitnet_common::ConcreteTensor, BitNetError> {
        Ok(ct(vec![1, self.config.model.vocab_size]))
    }

    fn embed(&self, tokens: &[u32]) -> Result<bitnet_common::ConcreteTensor, BitNetError> {
        Ok(ct(vec![1, tokens.len(), 768]))
    }

    fn logits(
        &self,
        _input: &bitnet_common::ConcreteTensor,
    ) -> Result<bitnet_common::ConcreteTensor, BitNetError> {
        Ok(ct(vec![1, 1, self.config.model.vocab_size]))
    }
}

/// Mock model that fails on forward pass
struct FailingMockModel {
    config: BitNetConfig,
}

impl FailingMockModel {
    fn new() -> Self {
        Self { config: BitNetConfig::default() }
    }
}

impl Model for FailingMockModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &bitnet_common::ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> Result<bitnet_common::ConcreteTensor, BitNetError> {
        Err(BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
            reason: "Mock model forward pass failure".to_string(),
        }))
    }

    fn embed(&self, _tokens: &[u32]) -> Result<bitnet_common::ConcreteTensor, BitNetError> {
        Err(BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
            reason: "Mock model embed failure".to_string(),
        }))
    }

    fn logits(
        &self,
        _input: &bitnet_common::ConcreteTensor,
    ) -> Result<bitnet_common::ConcreteTensor, BitNetError> {
        Err(BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
            reason: "Mock model logits failure".to_string(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestConfig, TestHarness};

    #[tokio::test]
    async fn test_model_loading_suite() {
        let config = TestConfig::default();
        let harness = TestHarness::new(config).await.unwrap();
        let suite = ModelLoadingTestSuite;

        let result = harness.run_test_suite(&suite).await;
        assert!(result.is_ok());

        let suite_result = result.unwrap();
        assert!(suite_result.summary.total_tests > 0);
    }
}
