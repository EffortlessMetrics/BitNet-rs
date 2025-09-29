# Stub code: `GpuPerformanceMetrics::gpu_utilization` in `gpu.rs` is a placeholder

The `GpuPerformanceMetrics::gpu_utilization` field in `crates/bitnet-inference/src/gpu.rs` is hardcoded to `0.85`. It doesn't actually measure GPU utilization. This is a form of stubbing and should be replaced with a real implementation.

**File:** `crates/bitnet-inference/src/gpu.rs`

**Field:** `GpuPerformanceMetrics::gpu_utilization`

**Code:**
```rust
/// GPU-specific performance metrics
#[derive(Debug, Clone, Default)]
pub struct GpuPerformanceMetrics {
    // ...
    pub gpu_utilization: f64,
    // ...
}

// In `generate_tokens_gpu`:
            metrics.gpu_utilization = 0.85; // Placeholder
```

## Proposed Fix

The `GpuPerformanceMetrics::gpu_utilization` field should be implemented to measure GPU utilization. This would involve using a library like `cuda` to query the GPU for its utilization.

### Example Implementation

```rust
// In `generate_tokens_gpu`:
            metrics.gpu_utilization = cuda::device_utilization(self.backend.device_id).unwrap_or(0.0);
```
