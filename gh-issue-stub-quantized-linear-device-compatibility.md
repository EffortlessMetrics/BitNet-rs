# [Device Management] Implement strict device compatibility validation

## Problem Description

The `QuantizedLinear::is_device_compatible` function always returns `true`, bypassing important device validation that could prevent runtime errors and performance issues.

## Root Cause Analysis

```rust
fn is_device_compatible(&self, _tensor: &BitNetTensor) -> bool {
    // For now, allow any device combination
    // In a full implementation, this would enforce strict device matching
    true
}
```

## Proposed Solution

```rust
impl QuantizedLinear {
    fn is_device_compatible(&self, tensor: &BitNetTensor) -> bool {
        // Strict device matching
        if tensor.device() != &self.device {
            warn!("Device mismatch: layer on {:?}, tensor on {:?}",
                  self.device, tensor.device());
            return false;
        }

        // Check device-specific capabilities
        match (&self.device, tensor.device()) {
            (Device::Cuda(a), Device::Cuda(b)) => a == b,
            (Device::Cpu, Device::Cpu) => true,
            (Device::Metal, Device::Metal) => true,
            _ => false,
        }
    }

    fn validate_tensor_compatibility(&self, tensor: &BitNetTensor) -> Result<()> {
        if !self.is_device_compatible(tensor) {
            return Err(BitNetError::DeviceMismatch {
                layer_device: self.device.clone(),
                tensor_device: tensor.device().clone(),
            });
        }

        // Additional validation for quantization compatibility
        if !self.supports_quantization_type(tensor.quantization_type()) {
            return Err(BitNetError::QuantizationMismatch {
                layer_qtype: self.quantization_type.clone(),
                tensor_qtype: tensor.quantization_type().clone(),
            });
        }

        Ok(())
    }
}
```

## Acceptance Criteria

- [ ] Strict device type matching (CPU/GPU/Metal)
- [ ] CUDA device ID validation
- [ ] Quantization type compatibility checks
- [ ] Clear error messages for mismatches

## Priority: High