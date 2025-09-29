# Dead code: `quantize_cuda` method in `I2SQuantizer` is never used

The `quantize_cuda` method in `crates/bitnet-quantization/src/i2s.rs` is never used. This was identified by `cargo clippy`.

**File:** `crates/bitnet-quantization/src/i2s.rs`
**Line:** 240

## Description

The `quantize_cuda` method is intended to provide a CUDA-accelerated implementation of the I2S quantization algorithm. However, it is currently not called from anywhere in the codebase. This means that the code is either dead and can be removed, or it is intended to be used but is not yet integrated.

## Proposed Fix

There are two possible solutions to this issue:

1.  **Remove the dead code:** If the `quantize_cuda` method is not intended to be used, it should be removed to reduce the size of the codebase and improve maintainability.

2.  **Integrate the `quantize_cuda` method:** If the `quantize_cuda` method is intended to be used, it should be integrated into the quantization process. This would involve adding a new `QuantizationDevice` enum that allows the user to select between the CPU and CUDA implementations of the quantization algorithm. The `I2SQuantizer::quantize` method would then dispatch to the appropriate implementation based on the selected device.

### Example Implementation for Solution 2

```rust
// In crates/bitnet-common/src/types.rs

pub enum QuantizationDevice {
    Cpu,
    Cuda,
}

// In crates/bitnet-quantization/src/i2s.rs

impl I2SQuantizer {
    pub fn quantize(&self, tensor: &BitNetTensor, device: QuantizationDevice) -> Result<QuantizedTensor> {
        match device {
            QuantizationDevice::Cpu => self.quantize_cpu(tensor),
            QuantizationDevice::Cuda => self.quantize_cuda(tensor),
        }
    }

    fn quantize_cpu(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        // ... existing CPU implementation ...
    }

    fn quantize_cuda(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        // ... existing CUDA implementation ...
    }
}
```

This change would require updating the call sites of the `I2SQuantizer::quantize` method to pass the selected device. A new configuration option would also need to be added to allow the user to select the quantization device.