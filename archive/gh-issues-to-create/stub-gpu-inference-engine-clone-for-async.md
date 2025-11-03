# Stub code: Simplified `GpuInferenceEngine::clone_for_async` in `gpu.rs`

The `GpuInferenceEngine::clone_for_async` function in `crates/bitnet-inference/src/gpu.rs` is a simplified clone that creates new instances of `KVCache` and `SamplingStrategy`. It doesn't properly clone the existing state. This is a form of stubbing and should be replaced with a real implementation.

**File:** `crates/bitnet-inference/src/gpu.rs`

**Function:** `GpuInferenceEngine::clone_for_async`

**Code:**
```rust
    fn clone_for_async(&self) -> Self {
        // This is a simplified clone - in practice would need proper cloning
        Self {
            model: self.model.clone(),
            backend: GpuBackend::with_config(self.backend.performance_config().clone()).unwrap(),
            cache: Arc::new(Mutex::new(
                KVCache::new(
                    &bitnet_common::BitNetConfig::default(),
                    self.config.max_sequence_length
                ).unwrap()
            )),
            sampling: Arc::new(Mutex::new(
                SamplingStrategy::new(crate::SamplingConfig::default()).unwrap()
            )),
            metrics: Arc::new(Mutex::new(GpuPerformanceMetrics::default())),
            config: self.config.clone(),
        }
    }
```

## Proposed Fix

The `GpuInferenceEngine::clone_for_async` function should be implemented to properly clone the existing state of the `GpuInferenceEngine`. This would involve cloning the `KVCache` and `SamplingStrategy` instances, rather than creating new ones.

### Example Implementation

```rust
    fn clone_for_async(&self) -> Self {
        Self {
            model: self.model.clone(),
            backend: GpuBackend::with_config(self.backend.performance_config().clone()).unwrap(),
            cache: Arc::new(Mutex::new(self.cache.lock().unwrap().clone())),
            sampling: Arc::new(Mutex::new(self.sampling.lock().unwrap().clone())),
            metrics: Arc::new(Mutex::new(self.metrics.lock().unwrap().clone())),
            config: self.config.clone(),
        }
    }
```
