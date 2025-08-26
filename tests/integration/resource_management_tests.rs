//! Resource management integration tests
//!
//! Validates that model and inference engine resources are released after use.

use crate::utils::get_memory_usage;
use crate::{TestError, TestResult};
use bitnet_common::Device;
use bitnet_inference::InferenceEngine;
use std::sync::Arc;

use super::{MockModel, MockTokenizer};

/// Allocate model and engine resources and report memory usage before and after cleanup.
pub async fn run_resource_management_tests() -> TestResult<ResourceUsage> {
    // Record memory before any allocations
    let memory_before = get_memory_usage();

    // Allocate model and tokenizer resources
    let model = Arc::new(MockModel::new());
    let tokenizer = Arc::new(MockTokenizer::new());

    // Create inference engine which should allocate additional resources
    let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), Device::Cpu)
        .map_err(|e| TestError::execution(format!("Engine creation failed: {e}")))?;

    let memory_after_engine = get_memory_usage();

    if memory_after_engine <= memory_before {
        return Err(TestError::assertion(
            "memory usage did not increase after engine creation".to_string(),
        ));
    }

    // Drop all handles to release resources
    drop(engine);
    drop(model);
    drop(tokenizer);

    // Allow some time for the allocator to release memory back to the OS
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let memory_after_cleanup = get_memory_usage();

    Ok(ResourceUsage { memory_before, memory_after_engine, memory_after_cleanup })
}

/// Memory usage measurements at various stages of the test
pub struct ResourceUsage {
    pub memory_before: u64,
    pub memory_after_engine: u64,
    pub memory_after_cleanup: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resource_cleanup() {
        let usage = run_resource_management_tests().await.expect("resource management test failed");

        // Engine allocation should increase memory usage
        assert!(
            usage.memory_after_engine > usage.memory_before,
            "engine allocation did not increase memory usage"
        );

        // After dropping the engine and model, memory should decrease or remain similar
        assert!(
            usage.memory_after_cleanup <= usage.memory_after_engine,
            "cleanup should not increase memory usage"
        );
    }
}
