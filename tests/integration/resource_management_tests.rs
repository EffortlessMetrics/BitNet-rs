//! Resource management integration tests
//!
//! These tests validate that resources such as memory are properly
//! allocated and released during model lifecycle events. They also
//! exercise failure paths like exceeding configured memory limits to
//! ensure graceful degradation under constrained environments.

use super::{MockModel, MockTokenizer};
use crate::{TestError, TestResult};
use bitnet_common::Device;
use bitnet_inference::{InferenceConfig, InferenceEngine};
use std::sync::Arc;

/// Run resource management scenarios and return an error if any scenario fails.
pub async fn run_resource_management_tests() -> TestResult<()> {
    // === Scenario 1: cache usage changes with lifecycle events ===
    let model = Arc::new(MockModel::new());
    let tokenizer = Arc::new(MockTokenizer::new());
    let mut engine = InferenceEngine::new(model, tokenizer, Device::Cpu)
        .map_err(|e| TestError::execution(format!("Failed to create engine: {}", e)))?;

    let initial_stats = engine.get_stats().await;
    // Run a forward pass to exercise cache allocation
    let _ = engine
        .eval_ids(&[1, 2, 3])
        .await
        .map_err(|e| TestError::execution(format!("Forward pass failed: {}", e)))?;
    let after_stats = engine.get_stats().await;
    if after_stats.cache_usage < initial_stats.cache_usage {
        return Err(TestError::assertion("Cache usage should not decrease after forward pass"));
    }

    engine.clear_cache().await;
    let cleared_stats = engine.get_stats().await;
    if cleared_stats.cache_usage > initial_stats.cache_usage {
        return Err(TestError::assertion(
            "Cache usage should return to baseline after clearing cache",
        ));
    }

    // === Scenario 2: enforcing memory limits ===
    let model = Arc::new(MockModel::new());
    let tokenizer = Arc::new(MockTokenizer::new());
    let tiny_config = InferenceConfig { memory_pool_size: 1024, ..InferenceConfig::default() };
    match InferenceEngine::with_config(model, tokenizer, Device::Cpu, tiny_config) {
        Ok(engine) => {
            let big_prompt = "x".repeat(10_000);
            if engine.generate(&big_prompt).await.is_ok() {
                return Err(TestError::assertion(
                    "Generation succeeded despite tiny memory pool".to_string(),
                ));
            }
        }
        Err(_) => {
            // Engine creation failed due to memory limits – expected outcome
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resource_management_scenarios() {
        if let Err(e) = run_resource_management_tests().await {
            panic!("resource management tests failed: {}", e);
        }
    }
}
