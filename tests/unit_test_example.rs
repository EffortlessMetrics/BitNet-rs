#![cfg(feature = "integration-tests")]
//! Example unit test demonstrating common patterns
//!
//! This example shows how to write effective unit tests using the BitNet.rs
//! testing framework, including setup, teardown, error testing, and assertions.

use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;

/// Example data structure for testing
#[derive(Debug, Clone, PartialEq)]
pub struct TextProcessor {
    config: ProcessorConfig,
    cache: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProcessorConfig {
    max_length: usize,
    uppercase: bool,
    prefix: String,
}

#[derive(Debug, PartialEq)]
pub enum ProcessorError {
    InputTooLong(usize),
    EmptyInput,
    InvalidConfig(String),
}

impl TextProcessor {
    pub fn new(config: ProcessorConfig) -> Result<Self, ProcessorError> {
        if config.max_length == 0 {
            return Err(ProcessorError::InvalidConfig("max_length cannot be zero".to_string()));
        }

        Ok(Self { config, cache: HashMap::new() })
    }

    pub async fn process(&mut self, input: &str) -> Result<String, ProcessorError> {
        if input.is_empty() {
            return Err(ProcessorError::EmptyInput);
        }

        if input.len() > self.config.max_length {
            return Err(ProcessorError::InputTooLong(input.len()));
        }

        // Check cache first
        if let Some(cached) = self.cache.get(input) {
            return Ok(cached.clone());
        }

        // Simulate async processing
        tokio::time::sleep(Duration::from_millis(10)).await;

        let mut result = format!("{}{}", self.config.prefix, input);

        if self.config.uppercase {
            result = result.to_uppercase();
        }

        // Cache the result
        self.cache.insert(input.to_string(), result.clone());

        Ok(result)
    }

    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create default config
    fn default_config() -> ProcessorConfig {
        ProcessorConfig { max_length: 100, uppercase: false, prefix: "Processed: ".to_string() }
    }

    // Helper function to create processor with default config
    fn create_processor() -> TextProcessor {
        TextProcessor::new(default_config()).unwrap()
    }

    #[tokio::test]
    async fn test_processor_creation_success() {
        // Test successful creation with valid config
        let config =
            ProcessorConfig { max_length: 50, uppercase: true, prefix: "Test: ".to_string() };

        let processor = TextProcessor::new(config.clone());

        assert!(processor.is_ok());
        let processor = processor.unwrap();
        assert_eq!(processor.config, config);
        assert_eq!(processor.cache_size(), 0);
    }

    #[tokio::test]
    async fn test_processor_creation_invalid_config() {
        // Test creation with invalid config
        let invalid_config = ProcessorConfig {
            max_length: 0, // Invalid: zero length
            uppercase: false,
            prefix: "Test: ".to_string(),
        };

        let result = TextProcessor::new(invalid_config);

        assert!(result.is_err());
        match result.unwrap_err() {
            ProcessorError::InvalidConfig(msg) => {
                assert_eq!(msg, "max_length cannot be zero");
            }
            other => panic!("Expected InvalidConfig error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_process_basic_functionality() {
        // Test basic text processing
        let mut processor = create_processor();
        let input = "hello world";

        let result = processor.process(input).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output, "Processed: hello world");

        // Verify caching
        assert_eq!(processor.cache_size(), 1);
    }

    #[tokio::test]
    async fn test_process_with_uppercase() {
        // Test processing with uppercase option
        let config =
            ProcessorConfig { max_length: 100, uppercase: true, prefix: "TEST: ".to_string() };
        let mut processor = TextProcessor::new(config).unwrap();

        let result = processor.process("hello world").await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "TEST: HELLO WORLD");
    }

    #[tokio::test]
    async fn test_process_empty_input() {
        // Test error handling for empty input
        let mut processor = create_processor();

        let result = processor.process("").await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ProcessorError::EmptyInput => {
                // Expected error
            }
            other => panic!("Expected EmptyInput error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_process_input_too_long() {
        // Test error handling for input that's too long
        let config = ProcessorConfig { max_length: 10, uppercase: false, prefix: "".to_string() };
        let mut processor = TextProcessor::new(config).unwrap();
        let long_input = "this input is definitely too long";

        let result = processor.process(long_input).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            ProcessorError::InputTooLong(length) => {
                assert_eq!(length, long_input.len());
            }
            other => panic!("Expected InputTooLong error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_caching_functionality() {
        // Test that caching works correctly
        let mut processor = create_processor();
        let input = "test input";

        // First call should process and cache
        let result1 = processor.process(input).await.unwrap();
        assert_eq!(processor.cache_size(), 1);

        // Second call should return cached result
        let result2 = processor.process(input).await.unwrap();
        assert_eq!(result1, result2);
        assert_eq!(processor.cache_size(), 1); // Still only one cached item

        // Different input should create new cache entry
        let result3 = processor.process("different input").await.unwrap();
        assert_ne!(result1, result3);
        assert_eq!(processor.cache_size(), 2);
    }

    #[tokio::test]
    async fn test_cache_clearing() {
        // Test cache clearing functionality
        let mut processor = create_processor();

        // Add some items to cache
        processor.process("input1").await.unwrap();
        processor.process("input2").await.unwrap();
        assert_eq!(processor.cache_size(), 2);

        // Clear cache
        processor.clear_cache();
        assert_eq!(processor.cache_size(), 0);

        // Processing same input should work again
        let result = processor.process("input1").await;
        assert!(result.is_ok());
        assert_eq!(processor.cache_size(), 1);
    }

    #[tokio::test]
    async fn test_multiple_configurations() {
        // Test with multiple different configurations
        let test_cases = vec![
            (
                ProcessorConfig { max_length: 20, uppercase: false, prefix: "A: ".to_string() },
                "hello",
                "A: hello",
            ),
            (
                ProcessorConfig { max_length: 20, uppercase: true, prefix: "B: ".to_string() },
                "world",
                "B: WORLD",
            ),
            (
                ProcessorConfig { max_length: 50, uppercase: false, prefix: "".to_string() },
                "no prefix",
                "no prefix",
            ),
        ];

        for (config, input, expected) in test_cases {
            let mut processor = TextProcessor::new(config).unwrap();
            let result = processor.process(input).await.unwrap();
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
    }

    #[tokio::test]
    async fn test_concurrent_processing() {
        // Test that processor works correctly with concurrent access
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let processor = Arc::new(Mutex::new(create_processor()));
        let mut handles = vec![];

        // Spawn multiple concurrent tasks
        for i in 0..10 {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                let input = format!("input_{}", i);
                let mut proc = processor_clone.lock().await;
                proc.process(&input).await
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let mut results = vec![];
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
            results.push(result.unwrap());
        }

        // Verify all results are unique and correct
        assert_eq!(results.len(), 10);
        for (i, result) in results.iter().enumerate() {
            let expected = format!("Processed: input_{}", i);
            assert_eq!(*result, expected);
        }
    }

    #[tokio::test]
    async fn test_performance_requirements() {
        // Test that processing meets performance requirements
        let mut processor = create_processor();
        let input = "performance test input";

        // Warm up
        for _ in 0..3 {
            processor.process(input).await.unwrap();
        }
        processor.clear_cache(); // Clear cache for accurate timing

        // Measure processing time
        let start = std::time::Instant::now();
        let result = processor.process(input).await;
        let duration = start.elapsed();

        assert!(result.is_ok());
        assert!(
            duration < Duration::from_millis(50),
            "Processing should complete within 50ms, took {:?}",
            duration
        );
    }

    #[tokio::test]
    async fn test_with_temporary_files() {
        // Example of testing with temporary files
        let temp_dir = TempDir::new().unwrap();
        let config_file = temp_dir.path().join("config.json");

        // Create test configuration file
        let config_json = r#"
        {
            "max_length": 200,
            "uppercase": true,
            "prefix": "FILE: "
        }
        "#;
        tokio::fs::write(&config_file, config_json.as_bytes()).await.unwrap();

        // Verify file was created
        assert!(config_file.exists());
        let content = tokio::fs::read(&config_file).await.unwrap();
        assert_eq!(content, config_json.as_bytes());

        // Use the configuration (in a real scenario, you'd parse the JSON)
        let config =
            ProcessorConfig { max_length: 200, uppercase: true, prefix: "FILE: ".to_string() };
        let mut processor = TextProcessor::new(config).unwrap();

        let result = processor.process("test").await.unwrap();
        assert_eq!(result, "FILE: TEST");

        // Cleanup happens automatically when temp_dir goes out of scope
    }

    #[tokio::test]
    async fn test_error_recovery() {
        // Test error recovery scenarios
        let mut processor = create_processor();

        // First, cause an error
        let error_result = processor.process("").await;
        assert!(error_result.is_err());

        // Processor should still work for valid input after error
        let success_result = processor.process("valid input").await;
        assert!(success_result.is_ok());
        assert_eq!(success_result.unwrap(), "Processed: valid input");

        // Cache should still work normally
        assert_eq!(processor.cache_size(), 1);
    }

    #[tokio::test]
    async fn test_boundary_conditions() {
        // Test boundary conditions
        let config = ProcessorConfig { max_length: 5, uppercase: false, prefix: "".to_string() };
        let mut processor = TextProcessor::new(config).unwrap();

        // Test exactly at the boundary
        let boundary_input = "12345"; // Exactly 5 characters
        let result = processor.process(boundary_input).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "12345");

        // Test just over the boundary
        let over_boundary = "123456"; // 6 characters
        let result = processor.process(over_boundary).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ProcessorError::InputTooLong(6)));

        // Test just under the boundary
        let under_boundary = "1234"; // 4 characters
        let result = processor.process(under_boundary).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "1234");
    }
}

// Example of how to run this test:
// cargo test --test unit_test_example
//
// To run a specific test:
// cargo test --test unit_test_example test_processor_creation_success
//
// To run with output:
// cargo test --test unit_test_example -- --nocapture
