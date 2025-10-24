# Stub code: Mock constructor in `ProductionInferenceEngine` when inference feature is disabled

The `ProductionInferenceEngine::new` function in `crates/bitnet-inference/src/production_engine.rs` has a `#[cfg(not(feature = "inference"))]` block that creates a mock engine and returns an error if the `inference` feature is not enabled. This is a form of stubbing and should be replaced with a more robust solution.

**File:** `crates/bitnet-inference/src/production_engine.rs`

**Function:** `ProductionInferenceEngine::new`

**Code:**
```rust
    #[cfg(not(feature = "inference"))]
    pub fn new(
        _model: Arc<dyn Model>,
        _tokenizer: Arc<dyn Tokenizer>,
        _device: Device,
    ) -> Result<Self> {
        info!("Creating mock production inference engine");
        Err(bitnet_common::BitNetError::Inference(InferenceError::GenerationFailed {
            reason: "Inference feature not enabled - compile with --features inference".to_string(),
        }))
    }
```

## Proposed Fix

Instead of returning an error, the `ProductionInferenceEngine::new` function should return a `ProductionInferenceEngine` that is configured to use a CPU backend and a default model. This will allow the engine to be used even if the `inference` feature is not enabled.

### Example Implementation

```rust
    #[cfg(not(feature = "inference"))]
    pub fn new(
        model: Arc<dyn Model>,
        tokenizer: Arc<dyn Tokenizer>,
        device: Device,
    ) -> Result<Self> {
        info!("Creating mock production inference engine");
        let engine =
            InferenceEngine::new(model.clone(), tokenizer.clone(), device).map_err(|e| {
                BitNetError::Inference(InferenceError::GenerationFailed {
                    reason: format!("Engine creation failed: {}", e),
                })
            })?;
        let device_manager = DeviceManager::new(device);
        let config = ProductionInferenceConfig::default();

        let mut metrics_collector = PerformanceMetricsCollector::new();
        metrics_collector.set_device_type(&device_manager.primary_device);

        Ok(Self {
            engine,
            model,
            tokenizer,
            metrics_collector: Arc::new(RwLock::new(metrics_collector)),
            device_manager,
            config,
        })
    }
```
