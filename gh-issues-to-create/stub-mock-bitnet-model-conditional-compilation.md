# Stub code: `MockBitNetModel` in `production_loader.rs` is conditionally compiled

The `MockBitNetModel` struct and its implementation in `crates/bitnet-models/src/production_loader.rs` are conditionally compiled with `#[cfg(not(feature = "inference"))]`. If the `inference` feature is enabled, this mock model is not available. This is a form of stubbing.

**File:** `crates/bitnet-models/src/production_loader.rs`

**Struct:** `MockBitNetModel`

**Code:**
```rust
/// Mock model implementation for testing when inference features are disabled
#[cfg(not(feature = "inference"))]
pub struct MockBitNetModel {
    config: bitnet_common::BitNetConfig,
}

#[cfg(not(feature = "inference"))]
impl Default for MockBitNetModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "inference"))]
impl MockBitNetModel {
    pub fn new() -> Self {
        Self { config: bitnet_common::BitNetConfig::default() }
    }

    pub fn get_memory_requirements(&self, device: &str) -> MemoryRequirements {
        MemoryRequirements {
            total_mb: 100,
            gpu_memory_mb: if device == "gpu" { Some(80) } else { None },
            cpu_memory_mb: 20,
            kv_cache_mb: 0,
            activation_mb: 0,
            headroom_mb: 0,
        }
    }

    pub fn get_optimal_device_config(&self) -> DeviceConfig {
        DeviceConfig {
            strategy: Some(DeviceStrategy::CpuOnly),
            cpu_threads: Some(1),
            gpu_memory_fraction: None,
            recommended_batch_size: 1,
        }
    }
}

#[cfg(not(feature = "inference"))]
impl Model for MockBitNetModel {
    fn config(&self) -> &bitnet_common::BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &bitnet_common::ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 10, 1000]))
    }

    fn embed(&self, tokens: &[u32]) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        Ok(bitnet_common::ConcreteTensor::mock(vec![1, tokens.len(), 768]))
    }

    fn logits(
        &self,
        _hidden: &bitnet_common::ConcreteTensor,
    ) -> bitnet_common::Result<bitnet_common::ConcreteTensor> {
        Ok(bitnet_common::ConcreteTensor::mock(vec![1, 1, 1000]))
    }
}
```

## Proposed Fix

The `MockBitNetModel` struct and its implementation should not be conditionally compiled. Instead, the mock model should be available regardless of the `inference` feature. This would involve:

1.  **Removing conditional compilation:** Remove the `#[cfg(not(feature = "inference"))]` attributes.
2.  **Moving to a dedicated mock module:** Move the `MockBitNetModel` to a dedicated mock module (e.g., `crates/bitnet-models/src/mock.rs`).
3.  **Using a proper mock framework:** Use a proper mock framework to create mock models for testing.

### Example Implementation

```rust
// In crates/bitnet-models/src/mock.rs

pub struct MockBitNetModel {
    config: bitnet_common::BitNetConfig,
}

// ... (rest of the implementation) ...
```
