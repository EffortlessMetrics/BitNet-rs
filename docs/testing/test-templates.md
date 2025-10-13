# Test Templates and Examples

This document provides copy-paste templates for common testing scenarios in BitNet.rs.

## Unit Test Templates

### Basic Unit Test Template

```rust
use bitnet_tests::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_function_name_success() {
        // Arrange
        let input = setup_test_input();

        // Act
        let result = function_under_test(input).await;

        // Assert
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.expected_field, expected_value);
    }

    #[tokio::test]
    async fn test_function_name_error_condition() {
        // Arrange
        let invalid_input = setup_invalid_input();

        // Act
        let result = function_under_test(invalid_input).await;

        // Assert
        assert!(result.is_err());
        match result.unwrap_err() {
            ExpectedError::SpecificVariant(msg) => {
                assert_eq!(msg, "Expected error message");
            }
            other => panic!("Expected SpecificVariant, got {:?}", other),
        }
    }

    // Helper functions
    fn setup_test_input() -> TestInput {
        TestInput {
            field1: "test_value".to_string(),
            field2: 42,
        }
    }

    fn setup_invalid_input() -> TestInput {
        TestInput {
            field1: "".to_string(), // Invalid empty string
            field2: -1,              // Invalid negative value
        }
    }
}
```

### Property-Based Test Template

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_property_holds_for_all_inputs(
        input in any::<String>().prop_filter("non-empty", |s| !s.is_empty())
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let result = function_under_test(&input).await;

            // Property: function should never panic and always return valid result
            prop_assert!(result.is_ok());

            let output = result.unwrap();
            // Property: output length should be reasonable
            prop_assert!(output.len() <= input.len() * 2);

            // Property: output should contain some part of input
            prop_assert!(output.contains(&input) || input.contains(&output));
        });
    }
}
```

### Parameterized Test Template

```rust
#[tokio::test]
async fn test_multiple_scenarios() {
    let test_cases = vec![
        TestCase {
            name: "scenario_1",
            input: "input1",
            expected: "expected1",
        },
        TestCase {
            name: "scenario_2",
            input: "input2",
            expected: "expected2",
        },
        TestCase {
            name: "edge_case",
            input: "",
            expected: "default",
        },
    ];

    for case in test_cases {
        let result = function_under_test(case.input).await;
        assert!(result.is_ok(), "Failed for case: {}", case.name);
        assert_eq!(result.unwrap(), case.expected, "Wrong output for case: {}", case.name);
    }
}

struct TestCase {
    name: &'static str,
    input: &'static str,
    expected: &'static str,
}
```

## Integration Test Templates

### Workflow Integration Test Template

```rust
use bitnet_tests::common::{TestUtilities, TestConfig};
use std::time::Duration;

#[tokio::test]
async fn test_complete_workflow() {
    // Setup
    let temp_dir = TestUtilities::create_temp_dir("workflow_test").await.unwrap();
    let config = TestConfig::default();

    // Step 1: Initialize system
    let system = initialize_system(&config).await
        .expect("System initialization should succeed");

    // Step 2: Load test data
    let test_data_path = temp_dir.join("test_data.json");
    TestUtilities::write_test_file(&test_data_path, TEST_DATA_JSON).await.unwrap();

    let data = system.load_data(&test_data_path).await
        .expect("Data loading should succeed");

    // Step 3: Process data
    let result = system.process_data(data).await
        .expect("Data processing should succeed");

    // Step 4: Validate results
    assert!(!result.is_empty(), "Result should not be empty");
    assert!(result.len() > 0, "Result should contain processed items");

    // Step 5: Verify side effects
    let output_file = temp_dir.join("output.json");
    assert!(output_file.exists(), "Output file should be created");

    let output_content = TestUtilities::read_test_file(&output_file).await.unwrap();
    assert!(!output_content.is_empty(), "Output file should not be empty");

    // Cleanup happens automatically when temp_dir is dropped
}

const TEST_DATA_JSON: &[u8] = br#"
{
    "items": [
        {"id": 1, "name": "test_item_1"},
        {"id": 2, "name": "test_item_2"}
    ]
}
"#;
```

### Component Interaction Test Template

```rust
use bitnet_tests::common::TestUtilities;

#[tokio::test]
async fn test_component_interaction() {
    // Setup components
    let component_a = ComponentA::new().await.unwrap();
    let component_b = ComponentB::new().await.unwrap();

    // Test data flow: A -> B
    let input_data = create_test_data();

    // Component A processes input
    let intermediate_result = component_a.process(input_data).await
        .expect("Component A should process successfully");

    // Validate intermediate result
    assert!(intermediate_result.is_valid(), "Intermediate result should be valid");

    // Component B processes A's output
    let final_result = component_b.process(intermediate_result).await
        .expect("Component B should process successfully");

    // Validate final result
    assert!(final_result.is_complete(), "Final result should be complete");
    assert_eq!(final_result.status(), ProcessingStatus::Success);

    // Test error propagation: A fails -> B handles gracefully
    let invalid_input = create_invalid_data();
    let error_result = component_a.process(invalid_input).await;

    assert!(error_result.is_err(), "Component A should fail with invalid input");

    // Component B should handle the error gracefully
    let error_handling_result = component_b.handle_error(error_result.unwrap_err()).await;
    assert!(error_handling_result.is_ok(), "Component B should handle errors gracefully");
}
```

## Performance Test Templates

### Basic Performance Test Template

```rust
use std::time::{Duration, Instant};
use bitnet_tests::common::TestUtilities;

#[tokio::test]
async fn test_performance_requirements() {
    let test_data = create_performance_test_data();

    // Warm up
    for _ in 0..3 {
        let _ = function_under_test(&test_data).await;
    }

    // Measure performance
    let start = Instant::now();
    let result = function_under_test(&test_data).await;
    let duration = start.elapsed();

    // Verify correctness
    assert!(result.is_ok(), "Function should succeed");

    // Verify performance requirements
    assert!(duration < Duration::from_millis(100),
           "Function should complete within 100ms, took {:?}", duration);

    // Measure memory usage
    let memory_before = TestUtilities::get_memory_usage();
    let _result = function_under_test(&test_data).await;
    let memory_after = TestUtilities::get_memory_usage();
    let memory_used = memory_after.saturating_sub(memory_before);

    assert!(memory_used < 10 * 1024 * 1024,
           "Function should use less than 10MB, used {} bytes", memory_used);
}

fn create_performance_test_data() -> TestData {
    // Create data that represents realistic workload
    TestData {
        items: (0..1000).map(|i| TestItem {
            id: i,
            data: format!("test_data_{}", i),
        }).collect(),
    }
}
```

### Benchmark Comparison Template

```rust
use std::time::{Duration, Instant};

#[tokio::test]
async fn test_performance_comparison() {
    let test_data = create_test_data();
    let iterations = 100;

    // Benchmark old implementation
    let mut old_times = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let _result = old_implementation(&test_data).await;
        old_times.push(start.elapsed());
    }

    // Benchmark new implementation
    let mut new_times = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let _result = new_implementation(&test_data).await;
        new_times.push(start.elapsed());
    }

    // Calculate statistics
    let old_avg = old_times.iter().sum::<Duration>() / old_times.len() as u32;
    let new_avg = new_times.iter().sum::<Duration>() / new_times.len() as u32;

    let improvement_ratio = old_avg.as_secs_f64() / new_avg.as_secs_f64();

    println!("Old implementation average: {:?}", old_avg);
    println!("New implementation average: {:?}", new_avg);
    println!("Improvement ratio: {:.2}x", improvement_ratio);

    // Assert performance improvement
    assert!(improvement_ratio >= 1.1,
           "New implementation should be at least 10% faster, got {:.2}x", improvement_ratio);
}
```

## Error Handling Test Templates

### Comprehensive Error Test Template

```rust
#[tokio::test]
async fn test_error_scenarios() {
    let error_scenarios = vec![
        ErrorScenario {
            name: "invalid_input",
            setup: || create_invalid_input(),
            expected_error: MyError::InvalidInput,
        },
        ErrorScenario {
            name: "network_failure",
            setup: || create_network_failure_scenario(),
            expected_error: MyError::NetworkError,
        },
        ErrorScenario {
            name: "timeout",
            setup: || create_timeout_scenario(),
            expected_error: MyError::Timeout,
        },
    ];

    for scenario in error_scenarios {
        let input = (scenario.setup)();
        let result = function_under_test(input).await;

        assert!(result.is_err(), "Scenario '{}' should fail", scenario.name);

        let error = result.unwrap_err();
        assert!(matches!(error, scenario.expected_error),
               "Scenario '{}' should produce {:?}, got {:?}",
               scenario.name, scenario.expected_error, error);
    }
}

struct ErrorScenario<F> {
    name: &'static str,
    setup: F,
    expected_error: MyError,
}
```

### Recovery Test Template

```rust
#[tokio::test]
async fn test_error_recovery() {
    let mut system = System::new().await.unwrap();

    // Simulate failure
    system.inject_failure(FailureType::NetworkError).await;

    // Attempt operation (should fail)
    let result = system.perform_operation().await;
    assert!(result.is_err(), "Operation should fail with injected failure");

    // Clear failure and retry
    system.clear_failures().await;

    // Operation should now succeed
    let result = system.perform_operation().await;
    assert!(result.is_ok(), "Operation should succeed after recovery");

    // Verify system state is consistent
    assert!(system.is_healthy().await, "System should be healthy after recovery");
}
```

## Mock and Test Double Templates

### Mock Implementation Template

```rust
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct MockService {
    calls: Arc<Mutex<Vec<String>>>,
    responses: Arc<Mutex<Vec<Result<String, ServiceError>>>>,
}

impl MockService {
    fn new() -> Self {
        Self::default()
    }

    fn expect_call(&self, method: &str) -> &Self {
        self.calls.lock().unwrap().push(method.to_string());
        self
    }

    fn return_response(&self, response: Result<String, ServiceError>) -> &Self {
        self.responses.lock().unwrap().push(response);
        self
    }

    fn verify_calls(&self, expected: &[&str]) {
        let calls = self.calls.lock().unwrap();
        assert_eq!(calls.len(), expected.len(), "Wrong number of calls");
        for (actual, expected) in calls.iter().zip(expected.iter()) {
            assert_eq!(actual, expected, "Unexpected method call");
        }
    }
}

#[async_trait::async_trait]
impl ServiceTrait for MockService {
    async fn method_a(&self, input: &str) -> Result<String, ServiceError> {
        self.calls.lock().unwrap().push(format!("method_a({})", input));

        let mut responses = self.responses.lock().unwrap();
        if let Some(response) = responses.pop() {
            response
        } else {
            Ok(format!("mock_response_for_{}", input))
        }
    }
}

#[tokio::test]
async fn test_with_mock() {
    let mock_service = MockService::new();
    mock_service.return_response(Ok("expected_response".to_string()));

    let system = System::new(Box::new(mock_service.clone()));

    let result = system.process("test_input").await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "expected_response");

    mock_service.verify_calls(&["method_a(test_input)"]);
}
```

## Resource Management Test Templates

### File Resource Test Template

```rust
use tempfile::TempDir;
use bitnet_tests::common::TestUtilities;

#[tokio::test]
async fn test_file_operations() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test_file.txt");

    // Test file creation
    let content = b"test file content";
    TestUtilities::write_test_file(&file_path, content).await.unwrap();

    assert!(file_path.exists(), "File should be created");

    // Test file reading
    let read_content = TestUtilities::read_test_file(&file_path).await.unwrap();
    assert_eq!(read_content, content, "File content should match");

    // Test file verification
    let is_valid = TestUtilities::verify_file(&file_path, Some(content.len() as u64)).await.unwrap();
    assert!(is_valid, "File should be valid");

    // Cleanup happens automatically when temp_dir is dropped
}
```

### Memory Resource Test Template

```rust
#[tokio::test]
async fn test_memory_management() {
    let initial_memory = TestUtilities::get_memory_usage();

    // Allocate resources
    let large_data = create_large_test_data();
    let peak_memory = TestUtilities::get_peak_memory_usage();

    // Process data
    let result = process_large_data(large_data).await;
    assert!(result.is_ok(), "Processing should succeed");

    // Drop large data
    drop(result);

    // Allow garbage collection
    tokio::task::yield_now().await;

    // Check memory usage returned to reasonable level
    let final_memory = TestUtilities::get_memory_usage();
    let memory_increase = final_memory.saturating_sub(initial_memory);

    assert!(memory_increase < 100 * 1024 * 1024,
           "Memory usage should not increase by more than 100MB, increased by {} bytes",
           memory_increase);

    assert!(peak_memory > initial_memory, "Peak memory should be higher than initial");
}

fn create_large_test_data() -> Vec<u8> {
    vec![0u8; 50 * 1024 * 1024] // 50MB of test data
}
```

## Usage Instructions

1. **Copy the appropriate template** for your test scenario
2. **Replace placeholder names** with your actual function/type names
3. **Customize the test data** and assertions for your specific case
4. **Add the test to the appropriate location** in the `tests/` directory
5. **Run the test** to verify it works: `cargo test --no-default-features --features cpu your_test_name`

## Template Customization Tips

- **Test Names**: Use descriptive names that explain what you're testing
- **Test Data**: Create realistic test data that represents actual usage
- **Assertions**: Be specific about what you're checking
- **Error Messages**: Include helpful context in assertion messages
- **Cleanup**: Use RAII patterns or testing utilities for automatic cleanup
- **Documentation**: Add comments explaining complex test logic

These templates provide a solid starting point for most testing scenarios in BitNet.rs. Customize them based on your specific needs and refer to the [Test Authoring Guide](test-authoring-guide.md) for more advanced patterns.
