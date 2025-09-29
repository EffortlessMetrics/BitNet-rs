# Stub code: `QuantizedLinear::can_use_native_quantized_matmul` in `quantized_linear.rs` has a fallback for `_`

The `QuantizedLinear::can_use_native_quantized_matmul` function in `crates/bitnet-inference/src/layers/quantized_linear.rs` has a `_ => true` fallback, which might hide cases where native quantized matmul is not actually available. This is a form of stubbing.

**File:** `crates/bitnet-inference/src/layers/quantized_linear.rs`

**Function:** `QuantizedLinear::can_use_native_quantized_matmul`

**Code:**
```rust
    fn can_use_native_quantized_matmul(&self) -> bool {
        match (&self.device, &self.qtype) {
            (Device::Cuda(_), QuantizationType::I2S) => true, // GPU I2S kernel available
            (Device::Cpu, QuantizationType::I2S) => true, // CPU I2S kernel always available via fallback
            (Device::Cpu, QuantizationType::TL1) => true, // CPU TL1 kernel available
            (Device::Cpu, QuantizationType::TL2) => true, // CPU TL2 kernel available
            _ => true,                                    // Default to native quantized operations
        }
    }
```

## Proposed Fix

The `QuantizedLinear::can_use_native_quantized_matmul` function should be implemented to accurately check if native quantized matmul is available for the given device and quantization type. This would involve querying the `KernelManager` for available kernels.

### Example Implementation

```rust
    fn can_use_native_quantized_matmul(&self) -> bool {
        self.kernel_manager.has_kernel_for(&self.device, &self.qtype)
    }
```
