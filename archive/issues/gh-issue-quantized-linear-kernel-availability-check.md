# [Layers] Implement Device Compatibility Checking in QuantizedLinear

## Problem Description

The `QuantizedLinear::is_device_compatible` function in `crates/bitnet-inference/src/layers/quantized_linear.rs` always returns `true` instead of checking device compatibility, which can lead to runtime errors when tensors and kernels are on incompatible devices.

## Environment

- **Component**: `crates/bitnet-inference/src/layers/quantized_linear.rs`
- **Function**: `QuantizedLinear::is_device_compatible`
- **Impact**: Device compatibility validation for quantized operations

## Current Implementation

```rust
fn is_device_compatible(&self, _tensor: &BitNetTensor) -> bool {
    // For now, allow any device combination
    // In a full implementation, this would enforce strict device matching
    true
}
```

## Proposed Solution

Implement proper device compatibility checking:

```rust
fn is_device_compatible(&self, tensor: &BitNetTensor) -> bool {
    match (&self.device, tensor.device()) {
        (Device::Cpu, Device::Cpu) => true,
        (Device::Cuda(gpu_id), Device::Cuda(tensor_gpu_id)) => gpu_id == tensor_gpu_id,
        (Device::Metal(_), Device::Metal(_)) => true, // Simplified Metal check
        _ => false, // Cross-device operations not supported
    }
}
```

## Implementation Tasks

- [ ] Compare tensor device with layer device
- [ ] Handle CPU, CUDA, and Metal device types
- [ ] Add proper error messages for incompatibility
- [ ] Implement device migration suggestions

## Acceptance Criteria

- [ ] Correctly identifies compatible device combinations
- [ ] Prevents runtime errors from device mismatches
- [ ] Provides clear error messages for incompatible operations
- [ ] Supports all target device types (CPU, CUDA, Metal)

## Related Issues

- **Related to**: Device management improvements
- **Blocks**: Robust cross-device operation handling
