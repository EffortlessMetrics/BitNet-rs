# [Async Infrastructure] Implement proper stateful cloning for CPU inference engine

## Problem Description

The `CpuInferenceEngine::clone_for_async` function creates new instances instead of properly cloning existing state, which breaks async inference scenarios where state continuity is required.

## Root Cause Analysis

```rust
fn clone_for_async(&self) -> Self {
    // This is a simplified clone - in practice would need proper cloning
    Self {
        model: self.model.clone(),
        backend: CpuBackend::with_config(self.backend.performance_config().clone()).unwrap(),
        cache: Arc::new(Mutex::new(
            KVCache::new(&bitnet_common::BitNetConfig::default(), self.config.max_sequence_length).unwrap()
        )),
        sampling: Arc::new(Mutex::new(
            SamplingStrategy::new(crate::SamplingConfig::default()).unwrap()
        )),
        metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
        config: self.config.clone(),
    }
}
```

## Proposed Solution

```rust
impl CpuInferenceEngine {
    fn clone_for_async(&self) -> Result<Self> {
        // Deep clone with proper state preservation
        let cloned_cache = {
            let cache_guard = self.cache.lock().unwrap();
            cache_guard.clone()
        };

        let cloned_sampling = {
            let sampling_guard = self.sampling.lock().unwrap();
            sampling_guard.clone()
        };

        let cloned_metrics = {
            let metrics_guard = self.metrics.lock().unwrap();
            metrics_guard.clone()
        };

        Ok(Self {
            model: self.model.clone(),
            backend: self.backend.clone_for_async()?,
            cache: Arc::new(Mutex::new(cloned_cache)),
            sampling: Arc::new(Mutex::new(cloned_sampling)),
            metrics: Arc::new(Mutex::new(cloned_metrics)),
            config: self.config.clone(),
        })
    }

    fn clone_with_shared_cache(&self) -> Self {
        // For scenarios where cache should be shared between instances
        Self {
            model: self.model.clone(),
            backend: self.backend.clone_lightweight(),
            cache: self.cache.clone(), // Shared cache
            sampling: Arc::new(Mutex::new(
                self.sampling.lock().unwrap().clone()
            )),
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            config: self.config.clone(),
        }
    }
}
```

## Acceptance Criteria

- [ ] Proper deep cloning of KV cache state
- [ ] Sampling strategy state preservation
- [ ] Backend configuration continuity
- [ ] Option for shared vs independent cache
- [ ] Thread-safe async operation support

## Priority: High
