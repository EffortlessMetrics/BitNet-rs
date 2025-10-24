# [Testing] Consolidate Backend Testing Infrastructure

## Problem Description

The `tests` module in `crates/bitnet-inference/src/backends.rs` contains extensive mock objects like `MockModel` that could be reused across multiple test files but are currently embedded within individual modules, leading to potential code duplication and maintenance overhead.

## Environment

- **Component**: `crates/bitnet-inference/src/backends.rs`
- **Test Infrastructure**: `MockModel` and related testing utilities
- **Impact**: Test maintainability and code reuse

## Current Implementation

The `MockModel` struct is defined within the test module of `backends.rs`:

```rust
#[cfg(test)]
mod tests {
    struct MockModel {
        config: BitNetConfig,
        // ... extensive implementation
    }

    impl Model for MockModel {
        // ... many method implementations
    }
}
```

## Issues Identified

1. **Code duplication potential**: Mock objects may need to be recreated in other test modules
2. **Maintenance overhead**: Changes to model interface require updates in multiple places
3. **Limited reusability**: Test utilities are not accessible from other test files
4. **Inconsistent testing**: Different test modules may create slightly different mock implementations

## Proposed Solution

### 1. Create Centralized Test Utilities Module

```rust
// crates/bitnet-inference/tests/common/mod.rs
pub mod mock_objects;
pub mod test_data;
pub mod assertions;

// crates/bitnet-inference/tests/common/mock_objects.rs
use bitnet_common::{BitNetConfig, Model, Result, BitNetTensor};

pub struct MockModel {
    pub config: BitNetConfig,
    pub forward_calls: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl MockModel {
    pub fn new() -> Self {
        Self::with_config(BitNetConfig::default())
    }

    pub fn with_config(config: BitNetConfig) -> Self {
        Self {
            config,
            forward_calls: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    pub fn get_forward_calls(&self) -> Vec<String> {
        self.forward_calls.lock().unwrap().clone()
    }
}

impl Model for MockModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        // Track method calls for testing
        self.forward_calls.lock().unwrap().push(format!("forward({:?})", input.dims()));

        // Return predictable test output
        let output_shape = vec![input.dims()[0], self.config.vocab_size];
        BitNetTensor::zeros(&output_shape, input.dtype(), input.device())
    }

    // Implement other required methods with minimal, testable behavior
}

pub struct MockBackend {
    pub device: Device,
    pub operations: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl MockBackend {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            operations: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    pub fn record_operation(&self, op: &str) {
        self.operations.lock().unwrap().push(op.to_string());
    }

    pub fn get_operations(&self) -> Vec<String> {
        self.operations.lock().unwrap().clone()
    }
}
```

### 2. Test Data Generation Utilities

```rust
// crates/bitnet-inference/tests/common/test_data.rs
use bitnet_common::{BitNetTensor, Device, DType};

pub struct TestDataGenerator;

impl TestDataGenerator {
    pub fn random_tensor(shape: &[usize], device: &Device) -> BitNetTensor {
        let numel = shape.iter().product();
        let data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1).collect();
        BitNetTensor::from_vec(data, shape, device).unwrap()
    }

    pub fn sequential_tensor(shape: &[usize], device: &Device) -> BitNetTensor {
        let numel = shape.iter().product();
        let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
        BitNetTensor::from_vec(data, shape, device).unwrap()
    }

    pub fn zeros_tensor(shape: &[usize], device: &Device) -> BitNetTensor {
        BitNetTensor::zeros(shape, DType::F32, device).unwrap()
    }

    pub fn ones_tensor(shape: &[usize], device: &Device) -> BitNetTensor {
        BitNetTensor::ones(shape, DType::F32, device).unwrap()
    }
}
```

### 3. Common Assertions and Test Helpers

```rust
// crates/bitnet-inference/tests/common/assertions.rs
use bitnet_common::{BitNetTensor, Result};

pub fn assert_tensors_equal(actual: &BitNetTensor, expected: &BitNetTensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "Tensor dimensions don't match");
    assert_eq!(actual.dtype(), expected.dtype(), "Tensor data types don't match");

    let actual_data = actual.to_vec1::<f32>()?;
    let expected_data = expected.to_vec1::<f32>()?;

    for (i, (&a, &e)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "Tensor values differ at index {}: {} vs {}",
            i, a, e
        );
    }

    Ok(())
}

pub fn assert_tensors_approximately_equal(
    actual: &BitNetTensor,
    expected: &BitNetTensor,
    tolerance: f32,
) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "Tensor dimensions don't match");

    let actual_data = actual.to_vec1::<f32>()?;
    let expected_data = expected.to_vec1::<f32>()?;

    for (i, (&a, &e)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (a - e).abs() < tolerance,
            "Tensor values differ at index {} beyond tolerance {}: {} vs {}",
            i, tolerance, a, e
        );
    }

    Ok(())
}

pub fn assert_performance_within_bounds(
    actual_duration: std::time::Duration,
    expected_duration: std::time::Duration,
    tolerance_factor: f64,
) {
    let ratio = actual_duration.as_secs_f64() / expected_duration.as_secs_f64();
    assert!(
        ratio <= tolerance_factor,
        "Performance test failed: actual duration {:?} exceeds expected {:?} by factor {}",
        actual_duration, expected_duration, ratio
    );
}
```

### 4. Updated Backend Tests

```rust
// crates/bitnet-inference/src/backends.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::{MockModel, MockBackend, TestDataGenerator};

    #[test]
    fn test_cpu_backend_forward() {
        let model = MockModel::new();
        let backend = CpuBackend::new();

        let input = TestDataGenerator::random_tensor(&[1, 512], &Device::Cpu);
        let result = backend.forward(&model, &input);

        assert!(result.is_ok());
        assert_eq!(model.get_forward_calls().len(), 1);
    }

    #[test]
    fn test_gpu_backend_initialization() {
        let backend = GpuBackend::new(0);
        assert!(backend.is_ok());
    }
}
```

## Implementation Breakdown

### Phase 1: Create Test Infrastructure
- [ ] Create `tests/common/` module structure
- [ ] Implement centralized mock objects
- [ ] Add test data generation utilities
- [ ] Create common assertion helpers

### Phase 2: Migrate Existing Tests
- [ ] Update backend tests to use common utilities
- [ ] Remove duplicated mock implementations
- [ ] Standardize test patterns across modules
- [ ] Add integration test utilities

### Phase 3: Enhanced Testing Capabilities
- [ ] Add performance testing helpers
- [ ] Implement parameterized test utilities
- [ ] Create device-specific test helpers
- [ ] Add test configuration management

### Phase 4: Documentation and Examples
- [ ] Document testing utilities usage
- [ ] Create testing best practices guide
- [ ] Add example test implementations
- [ ] Create testing architecture documentation

## Testing Strategy

### Validate Test Infrastructure
```rust
#[cfg(test)]
mod infrastructure_tests {
    use super::*;

    #[test]
    fn test_mock_model_consistency() {
        let model = MockModel::new();
        let input = TestDataGenerator::random_tensor(&[2, 10], &Device::Cpu);

        let output1 = model.forward(&input).unwrap();
        let output2 = model.forward(&input).unwrap();

        // Mock should produce consistent outputs
        assert_tensors_equal(&output1, &output2).unwrap();
    }

    #[test]
    fn test_assertion_helpers() {
        let tensor1 = TestDataGenerator::ones_tensor(&[3, 3], &Device::Cpu);
        let tensor2 = TestDataGenerator::ones_tensor(&[3, 3], &Device::Cpu);

        // Should not panic
        assert_tensors_equal(&tensor1, &tensor2).unwrap();
    }
}
```

## Acceptance Criteria

- [ ] Centralized test utilities accessible across all test modules
- [ ] Eliminated code duplication in test mock objects
- [ ] Consistent testing patterns across the codebase
- [ ] Comprehensive test helpers for common operations
- [ ] Performance testing infrastructure
- [ ] Clear documentation for testing utilities
- [ ] Minimal overhead for test execution
- [ ] Easy to extend for new testing needs

## Related Issues/PRs

- **Related to**: Test infrastructure improvements
- **Depends on**: Common testing framework
- **Blocks**: Comprehensive test coverage expansion
- **References**: Code quality and maintainability improvements

## Additional Context

Consolidating testing infrastructure will improve code quality, reduce maintenance overhead, and make it easier to write comprehensive tests across the BitNet.rs codebase. The centralized approach ensures consistency and reusability while maintaining the simplicity needed for effective testing.
