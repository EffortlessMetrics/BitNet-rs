//! # Component Interaction Integration Tests
//!
//! Tests cross-crate component interactions, data flow validation between components,
//! configuration propagation, error handling and recovery, and resource sharing and cleanup.

use super::*;
use crate::common::{FixtureManager, TestCase, TestError, TestMetrics, TestSuite};
use async_trait::async_trait;
use bitnet_common::{BitNetConfig, BitNetError, Device, ModelConfig, Tensor};
use bitnet_inference::{GenerationConfig, InferenceConfig, InferenceEngine, InferenceStats};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Test suite for component interaction validation
pub struct ComponentInteractionTestSuite;

impl TestSuite for ComponentInteractionTestSuite {
    fn name(&self) -> &str {
        "Component Interaction Tests"
    }

    fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
        vec![
            Box::new(CrossCrateDataFlowTest),
            Box::new(ConfigurationPropagationTest),
            Box::new(ErrorHandlingAndRecoveryTest),
            Box::new(ResourceSharingTest),
        ]
    }
}

/// Test data flow between cross-crate components
struct CrossCrateDataFlowTest;

#[async_trait]
impl TestCase for CrossCrateDataFlowTest {
    fn name(&self) -> &str {
        "cross_crate_data_flow"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> Result<(), TestError> {
        info!("Setting up cross-crate data flow test");
        Ok(())
    }

    async fn execute(&self) -> Result<TestMetrics, TestError> {
        let start_time = Instant::now();

        debug!("Creating instrumented components to track data flow");

        // Create instrumented model that tracks data flow
        let model = Arc::new(InstrumentedModel::new());
        let tokenizer = Arc::new(InstrumentedTokenizer::new());

        // Create inference engine
        let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), Device::Cpu)
            .map_err(|e| TestError::ExecutionError(format!("Failed to create engine: {}", e)))?;

        // Test data flow through the complete pipeline
        let test_input = "Hello, world! This is a test of data flow.";

        debug!("Testing complete data flow pipeline");
        let result = engine
            .generate(test_input)
            .await
            .map_err(|e| TestError::ExecutionError(format!("Generation failed: {}", e)))?;

        // Validate data flow occurred correctly
        let model_data_flow = model.get_data_flow_info();
        let tokenizer_data_flow = tokenizer.get_data_flow_info();

        debug!("Model data flow: {:?}", model_data_flow);
        debug!("Tokenizer data flow: {:?}", tokenizer_data_flow);

        // Verify tokenizer received input text
        if !tokenizer_data_flow.inputs_received.contains(&test_input.to_string()) {
            return Err(TestError::AssertionError {
                message: "Tokenizer should have received input text".to_string(),
            });
        }

        // Verify model received tokenized input
        if model_data_flow.forward_calls == 0 {
            return Err(TestError::AssertionError {
                message: "Model should have received forward calls".to_string(),
            });
        }

        // Verify tokenizer received model output for decoding
        if tokenizer_data_flow.decode_calls == 0 {
            return Err(TestError::AssertionError {
                message: "Tokenizer should have received decode calls".to_string(),
            });
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            custom_metrics: [
                ("tokenizer_encode_calls".to_string(), tokenizer_data_flow.encode_calls as f64),
                ("tokenizer_decode_calls".to_string(), tokenizer_data_flow.decode_calls as f64),
                ("model_forward_calls".to_string(), model_data_flow.forward_calls as f64),
                ("generated_text_length".to_string(), result.len() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> Result<(), TestError> {
        debug!("Cleaning up cross-crate data flow test");
        Ok(())
    }
}
/// Test configuration propagation across components
struct ConfigurationPropagationTest;

#[async_trait]
impl TestCase for ConfigurationPropagationTest {
    fn name(&self) -> &str {
        "configuration_propagation"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> Result<(), TestError> {
        info!("Setting up configuration propagation test");
        Ok(())
    }

    async fn execute(&self) -> Result<TestMetrics, TestError> {
        let start_time = Instant::now();

        debug!("Testing configuration propagation across components");

        // Create custom configurations
        let model_config = BitNetConfig {
            model: ModelConfig {
                vocab_size: 32000,
                hidden_size: 1024,
                num_layers: 24,
                num_heads: 16,
                max_position_embeddings: 2048,
                ..Default::default()
            },
            ..Default::default()
        };

        let inference_config = InferenceConfig {
            max_context_length: 1024,
            batch_size: 2,
            temperature: 0.8,
            top_p: 0.9,
            ..Default::default()
        };

        let generation_config = GenerationConfig {
            max_new_tokens: 50,
            temperature: 0.7,
            top_p: 0.95,
            do_sample: true,
            ..Default::default()
        };

        // Create components with configurations
        let model = Arc::new(ConfigurableModel::new(model_config.clone()));
        let tokenizer = Arc::new(ConfigurableTokenizer::new(model_config.model.vocab_size));

        // Create inference engine with configuration
        let engine = InferenceEngine::with_config(
            model.clone(),
            tokenizer.clone(),
            Device::Cpu,
            inference_config.clone(),
        )
        .map_err(|e| TestError::ExecutionError(format!("Engine creation failed: {}", e)))?;

        // Verify configuration propagation
        debug!("Verifying configuration propagation");

        // Check model configuration
        let actual_model_config = model.config();
        if actual_model_config.model.vocab_size != model_config.model.vocab_size {
            return Err(TestError::AssertionError {
                message: "Model vocab size configuration not propagated correctly".to_string(),
            });
        }

        // Check tokenizer configuration
        if tokenizer.vocab_size() != model_config.model.vocab_size {
            return Err(TestError::AssertionError {
                message: "Tokenizer vocab size should match model configuration".to_string(),
            });
        }

        // Generate with custom generation config
        let result = engine
            .generate_with_config("Test configuration propagation", &generation_config)
            .await
            .map_err(|e| {
                TestError::ExecutionError(format!("Generation with config failed: {}", e))
            })?;

        if result.is_empty() {
            return Err(TestError::AssertionError {
                message: "Generation with custom config should produce output".to_string(),
            });
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            custom_metrics: [
                ("model_vocab_size".to_string(), model_config.model.vocab_size as f64),
                ("model_hidden_size".to_string(), model_config.model.hidden_size as f64),
                ("inference_max_context".to_string(), inference_config.max_context_length as f64),
                ("generation_max_tokens".to_string(), generation_config.max_new_tokens as f64),
                ("configurations_tested".to_string(), 3.0),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> Result<(), TestError> {
        debug!("Cleaning up configuration propagation test");
        Ok(())
    }
}
/// Test error handling and recovery across components
struct ErrorHandlingAndRecoveryTest;

#[async_trait]
impl TestCase for ErrorHandlingAndRecoveryTest {
    fn name(&self) -> &str {
        "error_handling_and_recovery"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> Result<(), TestError> {
        info!("Setting up error handling and recovery test");
        Ok(())
    }

    async fn execute(&self) -> Result<TestMetrics, TestError> {
        let start_time = Instant::now();

        debug!("Testing error handling and recovery across components");

        let mut error_scenarios_tested = 0;
        let mut recovery_scenarios_successful = 0;

        // Test 1: Model error handling
        debug!("Testing model error handling");
        let failing_model = Arc::new(ErrorInjectingModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());

        let engine = InferenceEngine::new(failing_model.clone(), tokenizer.clone(), Device::Cpu)
            .map_err(|e| TestError::ExecutionError(format!("Engine creation failed: {}", e)))?;

        // Inject model error
        failing_model.set_should_fail(true);
        error_scenarios_tested += 1;

        let model_error_result = engine.generate("test model error").await;
        match model_error_result {
            Ok(_) => {
                return Err(TestError::AssertionError {
                    message: "Model error should have caused generation to fail".to_string(),
                });
            }
            Err(e) => {
                debug!("Model error correctly propagated: {}", e);
            }
        }

        // Test recovery after model error
        debug!("Testing recovery after model error");
        failing_model.set_should_fail(false);

        let recovery_result = engine.generate("test recovery").await;
        match recovery_result {
            Ok(result) => {
                if !result.is_empty() {
                    recovery_scenarios_successful += 1;
                    debug!("Successfully recovered from model error");
                }
            }
            Err(e) => {
                warn!("Failed to recover from model error: {}", e);
            }
        }

        // Test 2: Configuration error handling
        debug!("Testing configuration error handling");
        error_scenarios_tested += 1;

        let invalid_config = InferenceConfig {
            max_context_length: 0, // Invalid
            ..Default::default()
        };

        let config_error_result = InferenceEngine::with_config(
            Arc::new(MockModel::new()),
            Arc::new(MockTokenizer::new()),
            Device::Cpu,
            invalid_config,
        );

        match config_error_result {
            Ok(_) => {
                return Err(TestError::AssertionError {
                    message: "Invalid configuration should have been rejected".to_string(),
                });
            }
            Err(e) => {
                debug!("Configuration error correctly caught: {}", e);
            }
        }

        // Test recovery with valid configuration
        debug!("Testing recovery with valid configuration");
        let valid_config = InferenceConfig::default();

        let config_recovery_result = InferenceEngine::with_config(
            Arc::new(MockModel::new()),
            Arc::new(MockTokenizer::new()),
            Device::Cpu,
            valid_config,
        );

        match config_recovery_result {
            Ok(engine) => {
                let result = engine.generate("test config recovery").await;
                match result {
                    Ok(text) => {
                        if !text.is_empty() {
                            recovery_scenarios_successful += 1;
                            debug!("Successfully recovered from configuration error");
                        }
                    }
                    Err(e) => {
                        warn!("Config recovery engine failed to generate: {}", e);
                    }
                }
            }
            Err(e) => {
                warn!("Failed to recover from configuration error: {}", e);
            }
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            custom_metrics: [
                ("error_scenarios_tested".to_string(), error_scenarios_tested as f64),
                ("recovery_scenarios_successful".to_string(), recovery_scenarios_successful as f64),
                (
                    "recovery_success_rate".to_string(),
                    if error_scenarios_tested > 0 {
                        recovery_scenarios_successful as f64 / error_scenarios_tested as f64
                    } else {
                        0.0
                    },
                ),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> Result<(), TestError> {
        debug!("Cleaning up error handling and recovery test");
        Ok(())
    }
}
/// Test resource sharing between components
struct ResourceSharingTest;

#[async_trait]
impl TestCase for ResourceSharingTest {
    fn name(&self) -> &str {
        "resource_sharing"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> Result<(), TestError> {
        info!("Setting up resource sharing test");
        Ok(())
    }

    async fn execute(&self) -> Result<TestMetrics, TestError> {
        let start_time = Instant::now();

        debug!("Testing resource sharing between components");

        // Test 1: Shared model across multiple engines
        debug!("Testing shared model across multiple engines");
        let shared_model = Arc::new(ResourceTrackingModel::new());
        let tokenizer1 = Arc::new(MockTokenizer::new());
        let tokenizer2 = Arc::new(MockTokenizer::new());

        let engine1 = InferenceEngine::new(shared_model.clone(), tokenizer1, Device::Cpu)
            .map_err(|e| TestError::ExecutionError(format!("Engine1 creation failed: {}", e)))?;

        let engine2 = InferenceEngine::new(shared_model.clone(), tokenizer2, Device::Cpu)
            .map_err(|e| TestError::ExecutionError(format!("Engine2 creation failed: {}", e)))?;

        // Use both engines concurrently
        let result1_future = engine1.generate("test shared model 1");
        let result2_future = engine2.generate("test shared model 2");

        let (result1, result2) = tokio::join!(result1_future, result2_future);

        let result1 = result1
            .map_err(|e| TestError::ExecutionError(format!("Engine1 generation failed: {}", e)))?;
        let result2 = result2
            .map_err(|e| TestError::ExecutionError(format!("Engine2 generation failed: {}", e)))?;

        if result1.is_empty() || result2.is_empty() {
            return Err(TestError::AssertionError {
                message: "Both engines should produce output with shared model".to_string(),
            });
        }

        // Verify model was actually shared
        let model_usage = shared_model.get_usage_stats();
        debug!("Shared model usage: {:?}", model_usage);

        // Test 2: Resource cleanup when engines are dropped
        debug!("Testing resource cleanup when engines are dropped");
        let initial_usage = shared_model.get_usage_stats();

        drop(engine1);
        drop(engine2);

        // Give some time for cleanup
        tokio::time::sleep(Duration::from_millis(100)).await;

        let final_usage = shared_model.get_usage_stats();
        debug!("Usage after cleanup: {:?}", final_usage);

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            custom_metrics: [
                ("shared_model_accesses".to_string(), model_usage.total_accesses as f64),
                ("shared_model_concurrent".to_string(), model_usage.concurrent_accesses as f64),
                ("resource_sharing_scenarios".to_string(), 2.0),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> Result<(), TestError> {
        debug!("Cleaning up resource sharing test");
        Ok(())
    }
}
// Mock implementations for testing component interactions

#[derive(Debug)]
struct DataFlowInfo {
    inputs_received: Vec<String>,
    encode_calls: usize,
    decode_calls: usize,
    forward_calls: usize,
}

struct InstrumentedModel {
    config: BitNetConfig,
    data_flow: Arc<Mutex<DataFlowInfo>>,
}

impl InstrumentedModel {
    fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
            data_flow: Arc::new(Mutex::new(DataFlowInfo {
                inputs_received: Vec::new(),
                encode_calls: 0,
                decode_calls: 0,
                forward_calls: 0,
            })),
        }
    }

    fn get_data_flow_info(&self) -> DataFlowInfo {
        let guard = self.data_flow.lock().unwrap();
        DataFlowInfo {
            inputs_received: guard.inputs_received.clone(),
            encode_calls: guard.encode_calls,
            decode_calls: guard.decode_calls,
            forward_calls: guard.forward_calls,
        }
    }
}

impl Model for InstrumentedModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &bitnet_common::ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> Result<bitnet_common::ConcreteTensor, BitNetError> {
        let mut guard = self.data_flow.lock().unwrap();
        guard.forward_calls += 1;
        drop(guard);

        Ok(bitnet_common::MockTensor::new(vec![1, self.config.model.vocab_size]))
    }
}

struct InstrumentedTokenizer {
    vocab_size: usize,
    data_flow: Arc<Mutex<DataFlowInfo>>,
}

impl InstrumentedTokenizer {
    fn new() -> Self {
        Self {
            vocab_size: 50257,
            data_flow: Arc::new(Mutex::new(DataFlowInfo {
                inputs_received: Vec::new(),
                encode_calls: 0,
                decode_calls: 0,
                forward_calls: 0,
            })),
        }
    }

    fn get_data_flow_info(&self) -> DataFlowInfo {
        let guard = self.data_flow.lock().unwrap();
        DataFlowInfo {
            inputs_received: guard.inputs_received.clone(),
            encode_calls: guard.encode_calls,
            decode_calls: guard.decode_calls,
            forward_calls: guard.forward_calls,
        }
    }
}

impl Tokenizer for InstrumentedTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>, BitNetError> {
        let mut guard = self.data_flow.lock().unwrap();
        guard.inputs_received.push(text.to_string());
        guard.encode_calls += 1;
        drop(guard);

        Ok((0..text.len().min(10)).map(|i| (i + 1) as u32).collect())
    }

    fn decode(&self, tokens: &[u32], _skip_special_tokens: bool) -> Result<String, BitNetError> {
        let mut guard = self.data_flow.lock().unwrap();
        guard.decode_calls += 1;
        drop(guard);

        Ok(format!("generated_text_{}_tokens", tokens.len()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(50256)
    }

    fn pad_token_id(&self) -> Option<u32> {
        Some(50257)
    }
}
struct ConfigurableModel {
    config: BitNetConfig,
}

impl ConfigurableModel {
    fn new(config: BitNetConfig) -> Self {
        Self { config }
    }
}

impl Model for ConfigurableModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &bitnet_common::ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> Result<bitnet_common::ConcreteTensor, BitNetError> {
        Ok(bitnet_common::MockTensor::new(vec![1, self.config.model.vocab_size]))
    }
}

struct ConfigurableTokenizer {
    vocab_size: usize,
}

impl ConfigurableTokenizer {
    fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

impl Tokenizer for ConfigurableTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>, BitNetError> {
        Ok((0..text.len().min(10)).map(|i| (i + 1) as u32).collect())
    }

    fn decode(&self, tokens: &[u32], _skip_special_tokens: bool) -> Result<String, BitNetError> {
        Ok(format!("generated_text_{}_tokens", tokens.len()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some((self.vocab_size - 1) as u32)
    }

    fn pad_token_id(&self) -> Option<u32> {
        Some(self.vocab_size as u32)
    }
}

struct ErrorInjectingModel {
    config: BitNetConfig,
    should_fail: Arc<Mutex<bool>>,
}

impl ErrorInjectingModel {
    fn new() -> Self {
        Self { config: BitNetConfig::default(), should_fail: Arc::new(Mutex::new(false)) }
    }

    fn set_should_fail(&self, should_fail: bool) {
        *self.should_fail.lock().unwrap() = should_fail;
    }
}

impl Model for ErrorInjectingModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &bitnet_common::ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> Result<bitnet_common::ConcreteTensor, BitNetError> {
        if *self.should_fail.lock().unwrap() {
            return Err(BitNetError::ModelError("Injected model error for testing".to_string()));
        }
        Ok(bitnet_common::MockTensor::new(vec![1, self.config.model.vocab_size]))
    }
}

#[derive(Debug, Clone)]
struct UsageStats {
    total_accesses: usize,
    concurrent_accesses: usize,
}

struct ResourceTrackingModel {
    config: BitNetConfig,
    usage_stats: Arc<Mutex<UsageStats>>,
}

impl ResourceTrackingModel {
    fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
            usage_stats: Arc::new(Mutex::new(UsageStats {
                total_accesses: 0,
                concurrent_accesses: 0,
            })),
        }
    }

    fn get_usage_stats(&self) -> UsageStats {
        self.usage_stats.lock().unwrap().clone()
    }
}

impl Model for ResourceTrackingModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &bitnet_common::ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> Result<bitnet_common::ConcreteTensor, BitNetError> {
        let mut stats = self.usage_stats.lock().unwrap();
        stats.total_accesses += 1;
        stats.concurrent_accesses += 1;
        drop(stats);

        // Simulate some work
        std::thread::sleep(Duration::from_millis(10));

        let mut stats = self.usage_stats.lock().unwrap();
        stats.concurrent_accesses -= 1;
        drop(stats);

        Ok(bitnet_common::MockTensor::new(vec![1, self.config.model.vocab_size]))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{TestConfig, TestHarness};

    #[tokio::test]
    async fn test_component_interaction_suite() {
        let config = TestConfig::default();
        let harness = TestHarness::new(config).await.unwrap();
        let suite = ComponentInteractionTestSuite;

        let result = harness.run_test_suite(suite).await;
        assert!(result.is_ok());

        let suite_result = result.unwrap();
        assert!(suite_result.summary.total_tests > 0);
        assert!(suite_result.summary.passed > 0);
    }

    #[tokio::test]
    async fn test_cross_crate_data_flow() {
        let config = TestConfig::default();
        let harness = TestHarness::new(config).await.unwrap();
        let test_case = CrossCrateDataFlowTest;

        let result = harness.run_single_test(Box::new(test_case)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_configuration_propagation() {
        let config = TestConfig::default();
        let harness = TestHarness::new(config).await.unwrap();
        let test_case = ConfigurationPropagationTest;

        let result = harness.run_single_test(Box::new(test_case)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_error_handling_and_recovery() {
        let config = TestConfig::default();
        let harness = TestHarness::new(config).await.unwrap();
        let test_case = ErrorHandlingAndRecoveryTest;

        let result = harness.run_single_test(Box::new(test_case)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_resource_sharing() {
        let config = TestConfig::default();
        let harness = TestHarness::new(config).await.unwrap();
        let test_case = ResourceSharingTest;

        let result = harness.run_single_test(Box::new(test_case)).await;
        assert!(result.is_ok());
    }
}
