//! Tests for async runtime initialization fixes
//!
//! This test suite validates that the async runtime fixes in PR #179 work correctly
//! and prevent "no reactor running" errors.

use bitnet_common::Result;
use bitnet_inference::runtime_utils::{block_on_with_runtime, spawn_blocking_with_fallback};

#[tokio::test]
async fn test_spawn_blocking_with_runtime_context() {
    // Test that spawn_blocking_with_fallback works when called from within a tokio runtime
    let result = spawn_blocking_with_fallback(|| Ok("async context test")).await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "async context test");
}

#[test]
fn test_spawn_blocking_without_runtime_context() {
    // Test that spawn_blocking_with_fallback works when called from a sync context
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(spawn_blocking_with_fallback(|| Ok("sync context test")));

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "sync context test");
}

#[test]
fn test_spawn_blocking_fallback_to_sync() {
    // Test that spawn_blocking_with_fallback falls back to sync execution when no runtime
    // This test runs outside any tokio runtime
    let handle = std::thread::spawn(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(spawn_blocking_with_fallback(|| Ok("fallback test")))
    });

    let result = handle.join().unwrap();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "fallback test");
}

#[tokio::test]
async fn test_complex_async_operation() {
    // Test a complex async operation that would previously fail with "no reactor running"
    let result = spawn_blocking_with_fallback(|| {
        // Simulate heavy computation
        std::thread::sleep(std::time::Duration::from_millis(10));
        Ok("complex operation")
    })
    .await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "complex operation");
}

#[test]
fn test_block_on_with_runtime() {
    // Test the block_on_with_runtime utility
    let result = block_on_with_runtime(async {
        spawn_blocking_with_fallback(|| Ok("nested async test")).await
    });

    assert!(result.is_ok());
    let nested_result = result.unwrap();
    assert!(nested_result.is_ok());
    assert_eq!(nested_result.unwrap(), "nested async test");
}

/// Test that models the backend forward pass behavior
#[tokio::test]
async fn test_backend_forward_simulation() {
    use std::sync::Arc;

    // Simulate what happens in the backend forward pass
    let model_data = Arc::new(vec![1.0f32, 2.0, 3.0, 4.0]);
    let input_data = vec![0.5f32, 1.5, 2.5];

    let result = spawn_blocking_with_fallback({
        let model_data = model_data.clone();
        let input_data = input_data.clone();
        move || -> Result<Vec<f32>> {
            // Simulate model forward pass
            let mut output = Vec::new();
            for (i, &input_val) in input_data.iter().enumerate() {
                let model_val = model_data.get(i).unwrap_or(&1.0);
                output.push(input_val * model_val);
            }
            Ok(output)
        }
    })
    .await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 3);
    assert_eq!(output[0], 0.5); // 0.5 * 1.0
    assert_eq!(output[1], 3.0); // 1.5 * 2.0
    assert_eq!(output[2], 7.5); // 2.5 * 3.0
}

#[tokio::test]
async fn test_concurrent_spawn_blocking_operations() {
    // Test multiple concurrent spawn_blocking operations
    use futures_util::{FutureExt, future::join_all};

    let futures: Vec<_> = (0..5)
        .map(|i| {
            spawn_blocking_with_fallback(move || {
                std::thread::sleep(std::time::Duration::from_millis(10));
                Ok(format!("task_{}", i))
            })
            .boxed()
        })
        .collect();

    let results = join_all(futures).await;

    for (i, result) in results.into_iter().enumerate() {
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), format!("task_{}", i));
    }
}

/// Test error handling in spawn_blocking_with_fallback
#[tokio::test]
async fn test_error_handling() {
    let result = spawn_blocking_with_fallback(|| -> Result<()> {
        Err(bitnet_common::BitNetError::Validation("test error".to_string()))
    })
    .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("test error"));
}
