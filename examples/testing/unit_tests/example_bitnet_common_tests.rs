//! Example unit tests for bitnet-common crate
//!
//! This file demonstrates comprehensive unit testing patterns for the bitnet-rs project,
//! including property-based testing, error handling validation, and performance testing.

use bitnet_common::{BitNetError, ModelConfig, QuantizationConfig};
use proptest::prelude::*;
use std::time::Instant;

#[cfg(test)]
mod bitnet_common_examples {
    use super::*;

    /// Example: Basic functionality test with setup and teardown
    #[tokio::test]
    async fn test_model_config_creation() {
        // Arrange
        let config = ModelConfig::builder()
            .model_type("bitnet_b1_58".to_string())
            .vocab_size(32000)
            .hidden_size(4096)
            .num_layers(32)
            .build();

        // Act & Assert
        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.model_type(), "bitnet_b1_58");
        assert_eq!(config.vocab_size(), 32000);
        assert_eq!(config.hidden_size(), 4096);
        assert_eq!(config.num_layers(), 32);
    }

    /// Example: Error handling validation
    #[tokio::test]
    async fn test_invalid_model_config_returns_error() {
        // Test invalid vocab size
        let result = ModelConfig::builder()
            .model_type("bitnet_b1_58".to_string())
            .vocab_size(0) // Invalid: must be > 0
            .hidden_size(4096)
            .num_layers(32)
            .build();

        assert!(result.is_err());
        match result.unwrap_err() {
            BitNetError::InvalidConfiguration { field, reason } => {
                assert_eq!(field, "vocab_size");
                assert!(reason.contains("must be greater than 0"));
            }
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }

    /// Example: Property-based testing for invariants
    proptest! {
        #[test]
        fn test_quantization_config_invariants(
            bits in 1u8..=8u8,
            group_size in 32u32..=1024u32,
            symmetric in any::<bool>()
        ) {
            let config = QuantizationConfig::new(bits, group_size, symmetric);

            // Invariant: bits should always be within valid range
            prop_assert!(config.bits() >= 1 && config.bits() <= 8);

            // Invariant: group_size should be power of 2 or adjusted to nearest power of 2
            prop_assert!(config.group_size().is_power_of_two());

            // Invariant: configuration should be serializable and deserializable
            let serialized = serde_json::to_string(&config).unwrap();
            let deserialized: QuantizationConfig = serde_json::from_str(&serialized).unwrap();
            prop_assert_eq!(config, deserialized);
        }
    }

    /// Example: Performance testing with benchmarking
    #[tokio::test]
    async fn test_model_config_serialization_performance() {
        let config = ModelConfig::builder()
            .model_type("bitnet_b1_58".to_string())
            .vocab_size(32000)
            .hidden_size(4096)
            .num_layers(32)
            .build()
            .unwrap();

        // Benchmark serialization
        let start = Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            let _serialized = serde_json::to_string(&config).unwrap();
        }

        let duration = start.elapsed();
        let avg_duration = duration / iterations;

        // Assert performance requirement: < 1ms per serialization
        assert!(
            avg_duration.as_millis() < 1,
            "Serialization took {}ms, expected < 1ms",
            avg_duration.as_millis()
        );

        println!("Average serialization time: {:?}", avg_duration);
    }

    /// Example: Edge case testing
    #[tokio::test]
    async fn test_model_config_edge_cases() {
        // Test minimum valid values
        let min_config = ModelConfig::builder()
            .model_type("minimal".to_string())
            .vocab_size(1)
            .hidden_size(1)
            .num_layers(1)
            .build();
        assert!(min_config.is_ok());

        // Test maximum reasonable values
        let max_config = ModelConfig::builder()
            .model_type("maximal".to_string())
            .vocab_size(1_000_000)
            .hidden_size(16384)
            .num_layers(128)
            .build();
        assert!(max_config.is_ok());

        // Test empty model type
        let empty_type_config = ModelConfig::builder()
            .model_type("".to_string())
            .vocab_size(32000)
            .hidden_size(4096)
            .num_layers(32)
            .build();
        assert!(empty_type_config.is_err());
    }

    /// Example: Async operation testing
    #[tokio::test]
    async fn test_async_config_validation() {
        let config = ModelConfig::builder()
            .model_type("bitnet_b1_58".to_string())
            .vocab_size(32000)
            .hidden_size(4096)
            .num_layers(32)
            .build()
            .unwrap();

        // Simulate async validation
        let validation_result = tokio::spawn(async move {
            // Simulate some async work
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            config.validate_async().await
        })
        .await;

        assert!(validation_result.is_ok());
        assert!(validation_result.unwrap().is_ok());
    }

    /// Example: Memory usage testing
    #[tokio::test]
    async fn test_model_config_memory_usage() {
        use std::mem;

        let config = ModelConfig::builder()
            .model_type("bitnet_b1_58".to_string())
            .vocab_size(32000)
            .hidden_size(4096)
            .num_layers(32)
            .build()
            .unwrap();

        // Check memory footprint
        let size = mem::size_of_val(&config);

        // Assert reasonable memory usage (< 1KB for config)
        assert!(size < 1024, "Config uses {}B, expected < 1KB", size);

        println!("ModelConfig memory usage: {} bytes", size);
    }

    /// Example: Concurrent access testing
    #[tokio::test]
    async fn test_concurrent_config_access() {
        use std::sync::Arc;
        use tokio::task::JoinSet;

        let config = Arc::new(
            ModelConfig::builder()
                .model_type("bitnet_b1_58".to_string())
                .vocab_size(32000)
                .hidden_size(4096)
                .num_layers(32)
                .build()
                .unwrap(),
        );

        let mut join_set = JoinSet::new();

        // Spawn multiple tasks accessing the config concurrently
        for i in 0..10 {
            let config_clone = Arc::clone(&config);
            join_set.spawn(async move {
                // Simulate concurrent read access
                let model_type = config_clone.model_type();
                let vocab_size = config_clone.vocab_size();

                // Verify data consistency
                assert_eq!(model_type, "bitnet_b1_58");
                assert_eq!(vocab_size, 32000);

                i // Return task id for verification
            });
        }

        // Wait for all tasks to complete
        let mut completed_tasks = Vec::new();
        while let Some(result) = join_set.join_next().await {
            completed_tasks.push(result.unwrap());
        }

        // Verify all tasks completed successfully
        assert_eq!(completed_tasks.len(), 10);
        completed_tasks.sort();
        assert_eq!(completed_tasks, (0..10).collect::<Vec<_>>());
    }
}

/// Example: Custom test utilities and helpers
pub mod test_utils {
    use super::*;

    /// Helper function to create a default test configuration
    pub fn create_test_config() -> ModelConfig {
        ModelConfig::builder()
            .model_type("test_model".to_string())
            .vocab_size(1000)
            .hidden_size(512)
            .num_layers(8)
            .build()
            .unwrap()
    }

    /// Helper function to create a quantization config for testing
    pub fn create_test_quantization_config() -> QuantizationConfig {
        QuantizationConfig::new(4, 128, true)
    }

    /// Assertion helper for floating point comparisons
    pub fn assert_float_eq(a: f32, b: f32, epsilon: f32) {
        assert!(
            (a - b).abs() < epsilon,
            "Float values not equal: {} vs {} (epsilon: {})",
            a,
            b,
            epsilon
        );
    }

    /// Helper to measure execution time
    pub async fn measure_async<F, T>(f: F) -> (T, std::time::Duration)
    where
        F: std::future::Future<Output = T>,
    {
        let start = Instant::now();
        let result = f.await;
        let duration = start.elapsed();
        (result, duration)
    }
}

#[cfg(test)]
mod integration_helpers {
    /// Example: Test data builders for complex scenarios
    pub struct ModelConfigBuilder {
        model_type: Option<String>,
        vocab_size: Option<u32>,
        hidden_size: Option<u32>,
        num_layers: Option<u32>,
    }

    impl ModelConfigBuilder {
        pub fn new() -> Self {
            Self {
                model_type: None,
                vocab_size: None,
                hidden_size: None,
                num_layers: None,
            }
        }

        pub fn with_defaults(mut self) -> Self {
            self.model_type = Some("default".to_string());
            self.vocab_size = Some(32000);
            self.hidden_size = Some(4096);
            self.num_layers = Some(32);
            self
        }

        pub fn small_model(mut self) -> Self {
            self.model_type = Some("small".to_string());
            self.vocab_size = Some(1000);
            self.hidden_size = Some(256);
            self.num_layers = Some(4);
            self
        }

        pub fn large_model(mut self) -> Self {
            self.model_type = Some("large".to_string());
            self.vocab_size = Some(100000);
            self.hidden_size = Some(8192);
            self.num_layers = Some(64);
            self
        }
    }
}
