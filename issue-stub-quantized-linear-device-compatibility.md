# [IMPLEMENTATION] Implement strict device compatibility checking in QuantizedLinear

## Problem Description
The `QuantizedLinear::is_device_compatible` function in `crates/bitnet-inference/src/layers/quantized_linear.rs` always returns `true` instead of enforcing strict device matching between tensors and layers.

## Environment
- **File**: `crates/bitnet-inference/src/layers/quantized_linear.rs`
- **Function**: `QuantizedLinear::is_device_compatible`
- **Current State**: Always returns true regardless of device compatibility

## Root Cause Analysis
```rust
fn is_device_compatible(&self, _tensor: &BitNetTensor) -> bool {
    // For now, allow any device combination
    // In a full implementation, this would enforce strict device matching
    true
}
```

**Issues:**
1. No actual device compatibility checking
2. Risk of cross-device operations that will fail at runtime
3. Poor error reporting for device mismatches
4. Potential performance issues with unexpected device transfers

## Proposed Solution
```rust
impl QuantizedLinear {
    fn is_device_compatible(&self, tensor: &BitNetTensor) -> bool {
        match (self.device(), tensor.device()) {
            (Device::Cpu, Device::Cpu) => true,
            (Device::Cuda(id1), Device::Cuda(id2)) => id1 == id2,
            (Device::Metal, Device::Metal) => true,
            _ => false, // Cross-device operations not allowed
        }
    }

    fn ensure_device_compatibility(&self, tensor: &BitNetTensor) -> Result<()> {
        if !self.is_device_compatible(tensor) {
            return Err(BitNetError::DeviceMismatch {
                layer_device: self.device().clone(),
                tensor_device: tensor.device().clone(),
                layer_name: self.name().to_string(),
            });
        }
        Ok(())
    }

    fn device(&self) -> &Device {
        // Return the device this layer is placed on
        &self.device
    }

    fn name(&self) -> &str {
        // Return layer name for error reporting
        &self.layer_name
    }
}

// Enhanced error type
#[derive(Debug, thiserror::Error)]
pub enum BitNetError {
    #[error("Device mismatch: layer '{layer_name}' on {layer_device:?}, tensor on {tensor_device:?}")]
    DeviceMismatch {
        layer_device: Device,
        tensor_device: Device,
        layer_name: String,
    },
    // ... other errors
}
```

## Implementation Plan
### Phase 1: Device Tracking (1 day)
- [ ] Add device field to QuantizedLinear struct
- [ ] Implement device() method for layers
- [ ] Update layer constructors to track device placement

### Phase 2: Compatibility Checking (1 day)
- [ ] Implement strict device compatibility logic
- [ ] Add comprehensive error types for device mismatches
- [ ] Create ensure_device_compatibility helper method

## Acceptance Criteria
- [ ] Strict device compatibility enforcement
- [ ] Clear error messages for device mismatches
- [ ] Support for all device types (CPU, CUDA, Metal)
- [ ] Prevention of runtime device transfer errors

**Labels**: `implementation`, `device-management`, `error-handling`, `P2-medium`
**Effort**: 2 days
