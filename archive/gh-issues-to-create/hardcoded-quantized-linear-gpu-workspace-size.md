# Hardcoded values: `QuantizedLinear::calculate_gpu_workspace_size` in `quantized_linear.rs` has hardcoded values

The `QuantizedLinear::calculate_gpu_workspace_size` function in `crates/bitnet-inference/src/layers/quantized_linear.rs` has hardcoded values for `available_memory` and `max_batch_size`. These values may not be appropriate for all GPUs. This is a form of stubbing.

**File:** `crates/bitnet-inference/src/layers/quantized_linear.rs`

**Function:** `QuantizedLinear::calculate_gpu_workspace_size`

**Code:**
```rust
    fn calculate_gpu_workspace_size(&self) -> Result<usize> {
        // GPU kernels need temporary storage for different quantization types
        let base_weight_size = self.in_features * self.out_features;

        let (dequant_multiplier, intermediate_multiplier) = match self.qtype {
            QuantizationType::I2S => (2, 4), // FP16 + FP32
            QuantizationType::TL1 => (2, 4), // FP16 + FP32
            QuantizationType::TL2 => (4, 4), // FP32 + FP32 (larger tables)
        };

        // Conservative batch size estimate based on available GPU memory
        let max_batch_size = match self.device {
            Device::Cuda(_) => {
                // Estimate based on 6GB GPU memory target
                let available_memory: usize = 6 * 1024 * 1024 * 1024; // 6GB
                let model_memory = base_weight_size * dequant_multiplier;
                let remaining = available_memory.saturating_sub(model_memory);
                (remaining / (self.out_features * intermediate_multiplier)).min(128)
            }
            _ => 64, // Conservative default
        };

        let dequant_size = base_weight_size * dequant_multiplier;
        let intermediate_size = max_batch_size * self.out_features * intermediate_multiplier;
        let total_size = dequant_size + intermediate_size;

        // Clamp to maximum workspace size to prevent OOM
        let workspace_size = total_size.min(MAX_WORKSPACE_SIZE);

        log::debug!(
            "GPU workspace size: {} MB (batch_size: {}, qtype: {:?})",
            workspace_size / (1024 * 1024),
            max_batch_size,
            self.qtype
        );

        Ok(workspace_size)
    }
```

## Proposed Fix

The `QuantizedLinear::calculate_gpu_workspace_size` function should be implemented to dynamically query the available GPU memory and use it to estimate the maximum batch size. This would involve using a library like `cuda` to get the available GPU memory.

### Example Implementation

```rust
    fn calculate_gpu_workspace_size(&self) -> Result<usize> {
        // GPU kernels need temporary storage for different quantization types
        let base_weight_size = self.in_features * self.out_features;

        let (dequant_multiplier, intermediate_multiplier) = match self.qtype {
            QuantizationType::I2S => (2, 4), // FP16 + FP32
            QuantizationType::TL1 => (2, 4), // FP16 + FP32
            QuantizationType::TL2 => (4, 4), // FP32 + FP32 (larger tables)
        };

        // Estimate based on available GPU memory
        let available_memory: usize = match self.device {
            Device::Cuda(id) => cuda::device_available_memory(id)?,
            _ => 6 * 1024 * 1024 * 1024, // Fallback for non-CUDA devices
        };

        let model_memory = base_weight_size * dequant_multiplier;
        let remaining = available_memory.saturating_sub(model_memory);
        let max_batch_size = (remaining / (self.out_features * intermediate_multiplier)).min(128);

        let dequant_size = base_weight_size * dequant_multiplier;
        let intermediate_size = max_batch_size * self.out_features * intermediate_multiplier;
        let total_size = dequant_size + intermediate_size;

        // Clamp to maximum workspace size to prevent OOM
        let workspace_size = total_size.min(MAX_WORKSPACE_SIZE);

        log::debug!(
            "GPU workspace size: {} MB (batch_size: {}, qtype: {:?})",
            workspace_size / (1024 * 1024),
            max_batch_size,
            self.qtype
        );

        Ok(workspace_size)
    }
```
