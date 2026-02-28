//! Async runtime utilities for inference engine
//!
//! This module provides utilities for handling async runtime contexts,
//! particularly for operations that need to spawn blocking tasks with proper
//! runtime management.

use bitnet_common::{BitNetError, InferenceError, Result};
use std::future::Future;

/// Execute a future with a runtime context, creating one if needed
pub fn block_on_with_runtime<F, T>(future: F) -> Result<T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    // Try to use existing runtime if available
    match tokio::runtime::Handle::try_current() {
        Ok(_handle) => {
            // We're already in a runtime context, so we need to spawn a blocking task
            // to avoid blocking the runtime
            let result = std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().map_err(|e| {
                    BitNetError::Inference(InferenceError::GenerationFailed {
                        reason: format!("Failed to create runtime: {e}"),
                    })
                })?;
                Ok::<T, BitNetError>(rt.block_on(future))
            })
            .join()
            .map_err(|_| {
                BitNetError::Inference(InferenceError::GenerationFailed {
                    reason: "Thread panicked during block_on".to_string(),
                })
            })??;
            Ok(result)
        }
        Err(_) => {
            // No runtime context, create a new one
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                BitNetError::Inference(InferenceError::GenerationFailed {
                    reason: format!("Failed to create runtime: {}", e),
                })
            })?;
            Ok(rt.block_on(future))
        }
    }
}

/// Spawn a blocking task with fallback for different runtime contexts
pub async fn spawn_blocking_with_fallback<F, R>(f: F) -> Result<R>
where
    F: FnOnce() -> Result<R> + Send + 'static,
    R: Send + 'static,
{
    // Try to use tokio's spawn_blocking if we're in a tokio runtime
    match tokio::runtime::Handle::try_current() {
        Ok(_handle) => {
            // We're in a tokio runtime context
            tokio::task::spawn_blocking(f).await.map_err(|e| {
                BitNetError::Inference(InferenceError::GenerationFailed {
                    reason: format!("spawn_blocking failed: {}", e),
                })
            })?
        }
        Err(_) => {
            // No tokio runtime available, fall back to sync execution
            // This can happen in pure sync contexts or other async runtimes
            f()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_spawn_blocking_with_fallback() {
        let result = spawn_blocking_with_fallback(|| Ok("test")).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test");
    }

    #[test]
    fn test_block_on_with_runtime() {
        let result = block_on_with_runtime(async { "test" });
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test");
    }
}
