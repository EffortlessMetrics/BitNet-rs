# Quick Start Guide for Test Authors

This guide gets you writing tests in the BitNet-rs testing framework in under 10 minutes.

## Prerequisites

- Rust toolchain installed
- BitNet-rs repository cloned
- Basic familiarity with Rust and async programming

## Your First Test in 5 Minutes

### 1. Create a Simple Unit Test

Create a new file `tests/unit/my_first_test.rs`:

```rust
use bitnet_tests::prelude::*;

#[tokio::test]
async fn test_basic_functionality() {
    // Arrange - Set up test data
    let input = "Hello, BitNet!";

    // Act - Perform the operation
    let result = process_input(input).await;

    // Assert - Verify the result
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Processed: Hello, BitNet!");
}

async fn process_input(input: &str) -> Result<String, String> {
    Ok(format!("Processed: {}", input))
}
```

### 2. Run Your Test

```bash
cargo test --no-default-features --features cpu test_basic_functionality
```

That's it! You've written and run your first test.

## Common Test Patterns

### Testing with Test Data

```rust
use bitnet_tests::common::TestUtilities;

#[tokio::test]
async fn test_with_temporary_data() {
    // Create temporary directory
    let temp_dir = TestUtilities::create_temp_dir("my_test").await.unwrap();

    // Create test file
    let test_file = temp_dir.join("test_data.txt");
    TestUtilities::write_test_file(&test_file, b"test content").await.unwrap();

    // Your test logic here
    let content = TestUtilities::read_test_file(&test_file).await.unwrap();
    assert_eq!(content, b"test content");

    // Cleanup happens automatically when temp_dir goes out of scope
}
```

### Testing Error Conditions

```rust
#[tokio::test]
async fn test_error_handling() {
    let result = risky_operation().await;

    assert!(result.is_err());
    match result.unwrap_err() {
        MyError::InvalidInput(msg) => {
            assert_eq!(msg, "Expected error message");
        }
        _ => panic!("Expected InvalidInput error"),
    }
}

async fn risky_operation() -> Result<String, MyError> {
    Err(MyError::InvalidInput("Expected error message".to_string()))
}

#[derive(Debug, PartialEq)]
enum MyError {
    InvalidInput(String),
}
```

### Testing with Timeouts

```rust
use std::time::Duration;
use bitnet_tests::common::TestUtilities;

#[tokio::test]
async fn test_with_timeout() {
    let result = TestUtilities::run_with_timeout(
        || async {
            // Your async operation here
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok("completed")
        },
        Duration::from_secs(1), // 1 second timeout
    ).await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "completed");
}
```

## Test Organization

### File Structure
```
tests/
├── unit/                  # Unit tests
│   ├── bitnet_common/     # Tests for bitnet-common
│   ├── bitnet_models/     # Tests for bitnet-models
│   └── my_module/         # Your module tests
├── integration/           # Integration tests
│   └── my_workflow_test.rs
└── common/               # Shared utilities (already provided)
```

### Naming Conventions
- Test files: `test_<feature>.rs`
- Test functions: `test_<action>_<condition>_<expected_result>`
- Use descriptive names that explain what you're testing

## Running Tests

```bash
# Run all tests
cargo test --no-default-features --features cpu

# Run specific test
cargo test --no-default-features --features cpu test_basic_functionality

# Run tests in a specific file
cargo test --no-default-features --features cpu --test my_first_test

# Run with output
cargo test --no-default-features --features cpu -- --nocapture

# Run tests in parallel (default)
cargo test --no-default-features --features cpu -- --test-threads=4
```

## Next Steps

Once you're comfortable with basic tests:

1. Read the [Test Authoring Guide](test-authoring-guide.md) for advanced patterns
2. Check out [Framework Overview](framework-overview.md) to understand the architecture
3. Look at [Cross-Validation Guide](cross-validation-guide.md) for comparing implementations
4. Browse existing tests in the `tests/` directory for examples

## Common Gotchas

### 1. Async Tests
Always use `#[tokio::test]` for async tests, not `#[test]`:

```rust
// ✅ Correct
#[tokio::test]
async fn test_async_function() {
    let result = my_async_function().await;
    assert!(result.is_ok());
}

// ❌ Wrong
#[test]
fn test_async_function() {
    // This won't work with async code
}
```

### 2. Test Independence
Each test should be independent and not rely on other tests:

```rust
// ✅ Good - each test creates its own data
#[tokio::test]
async fn test_feature_a() {
    let data = create_test_data();
    // test logic
}

#[tokio::test]
async fn test_feature_b() {
    let data = create_test_data(); // Independent data
    // test logic
}

// ❌ Bad - tests depend on shared state
static mut SHARED_DATA: Option<TestData> = None;
```

### 3. Resource Cleanup
Use RAII patterns or the testing utilities for automatic cleanup:

```rust
// ✅ Good - automatic cleanup
#[tokio::test]
async fn test_with_resources() {
    let temp_dir = TestUtilities::create_temp_dir("test").await.unwrap();
    // Use temp_dir...
    // Cleanup happens automatically
}

// ❌ Risky - manual cleanup might be skipped on panic
#[tokio::test]
async fn test_with_manual_cleanup() {
    let temp_dir = create_temp_dir();
    // test logic...
    cleanup_temp_dir(temp_dir); // Might not run if test panics
}
```

## Getting Help

- Check the [Troubleshooting Guide](troubleshooting-guide.md) for common issues
- Look at existing tests for examples
- Ask questions in GitHub issues or discussions

Happy testing! 🧪
