# [Test Infrastructure] Refactor extensive mock objects in backends test module

## Problem Description

The test module in `crates/bitnet-inference/src/backends.rs` contains extensive `MockModel` implementation that could be optimized for maintainability and reusability across backend testing scenarios.

## Root Cause Analysis

Similar to other mock objects in the codebase, the `MockModel` in backends.rs is comprehensive but could benefit from:
1. Simplification for specific test scenarios
2. Extraction to shared test utilities
3. Better configurability for different backend tests

## Proposed Solution

### Extract to Shared Test Utilities

```rust
// crates/bitnet-inference/tests/utils/mock_models.rs

pub struct ConfigurableMockModel {
    config: BitNetConfig,
    forward_behavior: ForwardBehavior,
    device_compatibility: DeviceCompatibility,
}

impl ConfigurableMockModel {
    pub fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
            forward_behavior: ForwardBehavior::Default,
            device_compatibility: DeviceCompatibility::Permissive,
        }
    }

    pub fn with_device_requirements(mut self, device: Device) -> Self {
        self.device_compatibility = DeviceCompatibility::RequireSpecific(device);
        self
    }

    pub fn with_forward_behavior(mut self, behavior: ForwardBehavior) -> Self {
        self.forward_behavior = behavior;
        self
    }
}

impl Model for ConfigurableMockModel {
    fn config(&self) -> &BitNetConfig { &self.config }

    fn forward(&self, input: &ConcreteTensor, cache: &mut dyn Any) -> Result<ConcreteTensor> {
        match &self.forward_behavior {
            ForwardBehavior::Default => Ok(ConcreteTensor::mock(vec![1, 50257])),
            ForwardBehavior::CustomShape(shape) => Ok(ConcreteTensor::mock(shape.clone())),
            ForwardBehavior::Error(msg) => Err(anyhow::anyhow!(msg.clone())),
            ForwardBehavior::DeviceSpecific => {
                // Simulate device-specific behavior
                match input.device() {
                    Device::Cpu => Ok(ConcreteTensor::mock(vec![1, 50257])),
                    Device::Cuda(_) => Ok(ConcreteTensor::mock(vec![1, 50257])),
                    _ => Err(anyhow::anyhow!("Unsupported device for mock")),
                }
            }
        }
    }

    fn logits(&self, hidden: &ConcreteTensor) -> Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, 10, 50257]))
    }
}

#[derive(Debug, Clone)]
pub enum ForwardBehavior {
    Default,
    CustomShape(Vec<usize>),
    Error(String),
    DeviceSpecific,
}

#[derive(Debug, Clone)]
pub enum DeviceCompatibility {
    Permissive,
    RequireSpecific(Device),
    RejectAll,
}
```

### Simplified Backend Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::mock_models::*;

    #[test]
    fn test_cpu_backend_basic_inference() {
        let mock_model = ConfigurableMockModel::new()
            .with_device_requirements(Device::Cpu);

        let backend = CpuBackend::new(Arc::new(mock_model)).unwrap();

        // Test basic backend functionality
        assert_eq!(backend.device_type(), DeviceType::Cpu);
    }

    #[test]
    fn test_gpu_backend_device_validation() {
        let mock_model = ConfigurableMockModel::new()
            .with_device_requirements(Device::Cuda(0));

        let result = GpuBackend::new(Arc::new(mock_model));

        // Should handle device requirements appropriately
        assert!(result.is_ok());
    }
}
```

## Acceptance Criteria

- [ ] Mock objects extracted to shared test utilities
- [ ] Configurable mock behavior for different test scenarios
- [ ] Simplified inline mocks in backend tests
- [ ] Consistent mock architecture across test modules
- [ ] No duplication of mock functionality

## Priority: Low

Code quality improvement that enhances test maintainability without affecting production functionality.