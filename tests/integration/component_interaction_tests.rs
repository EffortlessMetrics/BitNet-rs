//! # Component Interaction Integration Tests
//!
//! Tests cross-crate component interactions, data flow validation between components,
//! configuration propagation, error handling and recovery, and resource sharing and cleanup.

use super::*;
use crate::common::{FixtureManager, TestCase, TestError, TestMetrics, TestResult, TestSuite};
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
            Box::new(ComponentLifecycleTest),
            Box::new(ConcurrentComponentAccessTest),
            Box::new(ComponentStateConsistencyTest),
            Box::new(InterfaceBoundaryTest),
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
        let result = engine.generate(test_input).await
            .map_err(|e| TestError::ExecutionError(format!("Generation failed: {}", e)))?;

        // Validate data flow occurred correctly
        let model_data_flow = model.get_data_flow_info();
        let tokenizer_data_flow = tokenizer.get_data_flow_info();

        debug!("Model data flow: {:?}", model_data_flow);
        debug!("Tokenizer data flow: {:?}", tokenizer_data_flow);

        // Verify tokenizer received input text
        if !tokenizer_data_flow.inputs_received.contains(&test_input.to_string()) {
            return Err(TestError::AssertionError { 
                message: "Tokenizer should have received input text".to_string() 
            });
        }

        // Verify model received tokenized input
        if model_data_flow.forward_calls == 0 {
            return Err(TestError::AssertionError { 
                message: "Model should have received forward calls".to_string() 
            });
        }

        // Verify tokenizer received model output for decoding
        if tokenizer_data_flow.decode_calls == 0 {
            return Err(TestError::AssertionError { 
                message: "Tokenizer should have received decode calls".to_string() 
            });
        }

        // Test data transformation consistency
        debug!("Testing data transformation consistency");
        let tokens = tokenizer.encode(test_input, true)
            .map_err(|e| TestError::ExecutionError(format!("Tokenization failed: {}", e)))?;
        
        let decoded = tokenizer.decode(&tokens, true)
            .map_err(|e| TestError::ExecutionError(format!("Detokenization failed: {}", e)))?;

        // Verify round-trip consistency (allowing for some tokenization artifacts)
        if decoded.trim().is_empty() {
            return Err(TestError::AssertionError { 
                message: "Round-trip tokenization should not produce empty result".to_string() 
            });
        }

        // Test tensor data flow
        debug!("Testing tensor data flow");
        let mock_input_tensor = bitnet_common::MockTensor::new(vec![1, tokens.len()]);
        let mut cache = HashMap::new();
        
        let output_tensor = model.forward(&mock_input_tensor, &mut cache)
            .map_err(|e| TestError::ExecutionError(format!("Model forward failed: {}", e)))?;

        // Verify tensor dimensions are consistent
        let output_shape = output_tensor.shape();
        if output_shape.len() != 2 {
            return Err(TestError::AssertionError { 
                message: "Output tensor should have 2 dimensions".to_string() 
            });
        }

        if output_shape[1] != model.config().model.vocab_size {
            return Err(TestError::AssertionError { 
                message: "Output tensor vocab dimension should match model config".to_string() 
            });
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            custom_metrics: [
                ("tokenizer_encode_calls".to_string(), tokenizer_data_flow.encode_call