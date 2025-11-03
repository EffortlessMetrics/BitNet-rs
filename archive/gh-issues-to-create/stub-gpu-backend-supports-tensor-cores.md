# Stub code: `GpuBackend::supports_tensor_cores` in `gpu.rs` is a placeholder

The `GpuBackend::supports_tensor_cores` function in `crates/bitnet-inference/src/gpu.rs` just returns the value from `self.performance_config.enable_tensor_cores`. It doesn't actually check the GPU's capabilities for tensor core availability. This is a form of stubbing and should be replaced with a real implementation.

**File:** `crates/bitnet-inference/src/gpu.rs`

**Function:** `GpuBackend::supports_tensor_cores`

**Code:**
```rust
    pub fn supports_tensor_cores(&self) -> bool {
        // Would check actual GPU capabilities (Volta+)
        self.performance_config.enable_tensor_cores
    }
```

## Proposed Fix

The `GpuBackend::supports_tensor_cores` function should be implemented to check the GPU's capabilities for tensor core availability. This would involve using a library like `cuda` to query the GPU for its tensor core capabilities.

### Example Implementation

```rust
    pub fn supports_tensor_cores(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            cuda::device_supports_tensor_cores(self.device_id).unwrap_or(false)
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
```
