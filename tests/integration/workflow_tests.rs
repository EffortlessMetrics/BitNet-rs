//! # End-to-End Workflow Integration Tests
//!
//! Tests complete inference workflows from prompt to generated text,
//! validating the integration between all components.

use super::*;
#[cfg(feature = "fixtures")]
use crate::common::FixtureManager;
use crate::common::harness::FixtureCtx;
use crate::{TestCase, TestError, TestMetrics, TestResult};
use async_trait::async_trait;
use std::time::Instant;
use tracing::{debug, info};

/// Test suite for end-to-end workflow integration
pub struct WorkflowIntegrationTestSuite;

impl crate::TestSuite for WorkflowIntegrationTestSuite {
    fn name(&self) -> &str {
        "Workflow Integration Tests"
    }

    fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
        vec![
            Box::new(BasicInferenceWorkflowTest),
            Box::new(MultiPromptWorkflowTest),
            Box::new(ConfigurationVariationTest),
            Box::new(ErrorHandlingWorkflowTest),
            Box::new(ResourceCleanupWorkflowTest),
        ]
    }
}

/// Test basic end-to-end inference workflow
struct BasicInferenceWorkflowTest;

#[async_trait]
impl TestCase for BasicInferenceWorkflowTest {
    fn name(&self) -> &str {
        "basic_inference_workflow"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up basic inference workflow test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        debug!("Creating mock components");
        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        debug!("Creating inference engine");
        let engine =
            InferenceEngine::new(model.clone(), tokenizer.clone(), device).map_err(|e| {
                TestError::execution(format!("Failed to create inference engine: {}", e))
            })?;

        debug!("Testing basic text generation");
        let prompt = "Hello, world!";
        let result = engine
            .generate(prompt)
            .await
            .map_err(|e| TestError::execution(format!("Generation failed: {}", e)))?;

        // Validate result
        if result.is_empty() {
            return Err(TestError::assertion("Generated text should not be empty"));
        }

        debug!("Generated text: {}", result);

        // Verify component interactions
        let model_calls = model.forward_call_count();
        let encode_calls = tokenizer.encode_call_count();
        let decode_calls = tokenizer.decode_call_count();

        if model_calls == 0 {
            return Err(TestError::assertion("Model forward should have been called"));
        }

        if encode_calls == 0 {
            return Err(TestError::assertion("Tokenizer encode should have been called"));
        }

        if decode_calls == 0 {
            return Err(TestError::assertion("Tokenizer decode should have been called"));
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("model_forward_calls".to_string(), model_calls as f64),
                ("tokenizer_encode_calls".to_string(), encode_calls as f64),
                ("tokenizer_decode_calls".to_string(), decode_calls as f64),
                ("generated_text_length".to_string(), result.len() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up basic inference workflow test");
        Ok(())
    }
}

/// Test workflow with multiple prompts
struct MultiPromptWorkflowTest;

#[async_trait]
impl TestCase for MultiPromptWorkflowTest {
    fn name(&self) -> &str {
        "multi_prompt_workflow"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up multi-prompt workflow test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine =
            InferenceEngine::new(model.clone(), tokenizer.clone(), device).map_err(|e| {
                TestError::execution(format!("Failed to create inference engine: {}", e))
            })?;

        let test_prompts = IntegrationTestData::test_prompts();
        let mut results = Vec::new();
        let mut total_generated_length = 0;

        debug!("Testing {} prompts", test_prompts.len());

        for (i, prompt) in test_prompts.iter().enumerate() {
            debug!("Processing prompt {}: {}", i + 1, prompt);

            let result = engine.generate(prompt).await.map_err(|e| {
                TestError::execution(format!("Generation failed for prompt {}: {}", i + 1, e))
            })?;

            if result.is_empty() {
                return Err(TestError::assertion(format!(
                    "Generated text should not be empty for prompt {}",
                    i + 1
                )));
            }

            total_generated_length += result.len();
            results.push(result);
        }

        // Verify all prompts were processed
        if results.len() != test_prompts.len() {
            return Err(TestError::assertion("Not all prompts were processed"));
        }

        // Verify component call counts
        let model_calls = model.forward_call_count();
        let encode_calls = tokenizer.encode_call_count();
        let decode_calls = tokenizer.decode_call_count();

        // Should have at least one call per prompt
        if model_calls < test_prompts.len() {
            return Err(TestError::assertion("Insufficient model forward calls"));
        }

        if encode_calls < test_prompts.len() {
            return Err(TestError::assertion("Insufficient tokenizer encode calls"));
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("prompts_processed".to_string(), test_prompts.len() as f64),
                ("total_generated_length".to_string(), total_generated_length as f64),
                (
                    "average_generated_length".to_string(),
                    total_generated_length as f64 / test_prompts.len() as f64,
                ),
                ("model_forward_calls".to_string(), model_calls as f64),
                ("tokenizer_encode_calls".to_string(), encode_calls as f64),
                ("tokenizer_decode_calls".to_string(), decode_calls as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up multi-prompt workflow test");
        Ok(())
    }
}

/// Test workflow with different configurations
struct ConfigurationVariationTest;

#[async_trait]
impl TestCase for ConfigurationVariationTest {
    fn name(&self) -> &str {
        "configuration_variation_workflow"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up configuration variation workflow test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let test_configs = IntegrationTestData::test_generation_configs();
        let prompt = "Test prompt for configuration variation";
        let mut results = Vec::new();

        debug!("Testing {} different configurations", test_configs.len());

        for (i, config) in test_configs.iter().enumerate() {
            debug!("Testing configuration {}: {:?}", i + 1, config);

            let engine =
                InferenceEngine::new(model.clone(), tokenizer.clone(), device).map_err(|e| {
                    TestError::execution(format!("Failed to create inference engine: {}", e))
                })?;

            let result = engine.generate_with_config(prompt, config).await.map_err(|e| {
                TestError::execution(format!("Generation failed for config {}: {}", i + 1, e))
            })?;

            if result.is_empty() {
                return Err(TestError::assertion(format!(
                    "Generated text should not be empty for config {}",
                    i + 1
                )));
            }

            results.push(result);
        }

        // Verify all configurations were tested
        if results.len() != test_configs.len() {
            return Err(TestError::assertion("Not all configurations were tested"));
        }

        // Verify results are different (configurations should produce different outputs)
        let unique_results: std::collections::HashSet<_> = results.iter().collect();
        if unique_results.len() < 2 {
            debug!("Warning: All configurations produced identical results");
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("configurations_tested".to_string(), test_configs.len() as f64),
                ("unique_results".to_string(), unique_results.len() as f64),
                ("model_forward_calls".to_string(), model.forward_call_count() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up configuration variation workflow test");
        Ok(())
    }
}

/// Test error handling in workflows
struct ErrorHandlingWorkflowTest;

#[async_trait]
impl TestCase for ErrorHandlingWorkflowTest {
    fn name(&self) -> &str {
        "error_handling_workflow"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up error handling workflow test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        let engine =
            InferenceEngine::new(model.clone(), tokenizer.clone(), device).map_err(|e| {
                TestError::execution(format!("Failed to create inference engine: {}", e))
            })?;

        // Test with empty prompt
        debug!("Testing empty prompt handling");
        let empty_result = engine.generate("").await;

        // Should handle empty prompt gracefully (either succeed or fail cleanly)
        match empty_result {
            Ok(text) => {
                debug!("Empty prompt handled successfully: '{}'", text);
            }
            Err(e) => {
                debug!("Empty prompt failed cleanly: {}", e);
            }
        }

        // Test with very long prompt
        debug!("Testing long prompt handling");
        let long_prompt = "a".repeat(10000);
        let long_result = engine.generate(&long_prompt).await;

        match long_result {
            Ok(text) => {
                debug!("Long prompt handled successfully, generated {} chars", text.len());
            }
            Err(e) => {
                debug!("Long prompt failed cleanly: {}", e);
            }
        }

        // Test with invalid configuration
        debug!("Testing invalid configuration handling");
        let invalid_config = GenerationConfig {
            max_new_tokens: 0, // Invalid
            ..Default::default()
        };

        // Validate configuration first
        if let Err(validation_error) = invalid_config.validate() {
            debug!("Configuration validation correctly failed: {}", validation_error);
        } else {
            return Err(TestError::assertion("Invalid configuration should fail validation"));
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("error_scenarios_tested".to_string(), 3.0),
                ("model_forward_calls".to_string(), model.forward_call_count() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up error handling workflow test");
        Ok(())
    }
}

/// Test resource cleanup in workflows
struct ResourceCleanupWorkflowTest;

#[async_trait]
impl TestCase for ResourceCleanupWorkflowTest {
    fn name(&self) -> &str {
        "resource_cleanup_workflow"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up resource cleanup workflow test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start_time = Instant::now();

        let model = Arc::new(MockModel::new());
        let tokenizer = Arc::new(MockTokenizer::new());
        let device = Device::Cpu;

        debug!("Creating inference engine");
        let engine =
            InferenceEngine::new(model.clone(), tokenizer.clone(), device).map_err(|e| {
                TestError::execution(format!("Failed to create inference engine: {}", e))
            })?;

        // Get initial stats
        let initial_stats = engine.get_stats().await;
        debug!(
            "Initial cache stats: size={}, usage={:.2}%",
            initial_stats.cache_size, initial_stats.cache_usage
        );

        // Perform some operations to populate cache
        let prompts = vec!["Test 1", "Test 2", "Test 3"];
        for prompt in &prompts {
            let _result = engine
                .generate(prompt)
                .await
                .map_err(|e| TestError::execution(format!("Generation failed: {}", e)))?;
        }

        // Get stats after operations
        let after_ops_stats = engine.get_stats().await;
        debug!(
            "After operations cache stats: size={}, usage={:.2}%",
            after_ops_stats.cache_size, after_ops_stats.cache_usage
        );

        // Clear cache
        debug!("Clearing cache");
        engine.clear_cache().await;

        // Get stats after cleanup
        let after_cleanup_stats = engine.get_stats().await;
        debug!(
            "After cleanup cache stats: size={}, usage={:.2}%",
            after_cleanup_stats.cache_size, after_cleanup_stats.cache_usage
        );

        // Verify cache was cleared (usage should be lower)
        if after_cleanup_stats.cache_usage > after_ops_stats.cache_usage {
            return Err(TestError::assertion("Cache usage should decrease after cleanup"));
        }

        // Test that engine still works after cleanup
        debug!("Testing functionality after cleanup");
        let post_cleanup_result = engine
            .generate("Post cleanup test")
            .await
            .map_err(|e| TestError::execution(format!("Generation failed after cleanup: {}", e)))?;

        if post_cleanup_result.is_empty() {
            return Err(TestError::assertion("Engine should work after cleanup"));
        }

        let duration = start_time.elapsed();

        Ok(TestMetrics {
            wall_time: duration,
            memory_peak: None,
            memory_average: None,
            cpu_time: Some(duration),
            assertions: 0,
            operations: 0,
            custom_metrics: [
                ("initial_cache_size".to_string(), initial_stats.cache_size as f64),
                ("initial_cache_usage".to_string(), initial_stats.cache_usage),
                ("after_ops_cache_size".to_string(), after_ops_stats.cache_size as f64),
                ("after_ops_cache_usage".to_string(), after_ops_stats.cache_usage),
                ("after_cleanup_cache_size".to_string(), after_cleanup_stats.cache_size as f64),
                ("after_cleanup_cache_usage".to_string(), after_cleanup_stats.cache_usage),
                ("prompts_processed".to_string(), prompts.len() as f64),
            ]
            .into_iter()
            .collect(),
        })
    }

    async fn cleanup(&self) -> TestResult<()> {
        debug!("Cleaning up resource cleanup workflow test");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestConfig, TestHarness};

    #[tokio::test]
    async fn test_workflow_integration_suite() {
        let config = TestConfig::default();
        let harness = TestHarness::new(config).await.unwrap();
        let suite = WorkflowIntegrationTestSuite;

        let result = harness.run_test_suite(&suite).await;
        assert!(result.is_ok());

        let suite_result = result.unwrap();
        assert!(suite_result.summary.total_tests > 0);
        assert!(suite_result.summary.passed > 0);
    }
}
