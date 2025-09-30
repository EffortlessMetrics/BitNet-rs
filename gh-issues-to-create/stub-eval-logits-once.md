# Stub code: `eval_logits_once` in `parity.rs` falls back to a mock model

The `eval_logits_once` function in `crates/bitnet-inference/src/parity.rs` attempts to load a model, but if it fails, it falls back to a mock model. This is a form of stubbing and can hide real issues with model loading.

**File:** `crates/bitnet-inference/src/parity.rs`

**Function:** `eval_logits_once`

**Code:**
```rust
pub fn eval_logits_once(model_path: &str, tokens: &[i32]) -> Result<Vec<f32>> {
    // Try to load model tensors; fall back to a mock model if unavailable
    let (config, model) = match load_gguf(Path::new(model_path), Device::Cpu) {
        Ok((cfg, tensors)) => {
            let model = BitNetModel::from_gguf(cfg.clone(), tensors, Device::Cpu)?;
            (cfg, model)
        }
        Err(_) => {
            let cfg = BitNetConfig::default();
            let model = BitNetModel::new(cfg.clone(), Device::Cpu);
            (cfg, model)
        }
    };
    // ...
}
```

## Proposed Fix

Instead of falling back to a mock model, the `eval_logits_once` function should return an error if the model fails to load. This will make it easier to identify and debug issues with model loading.

### Example Implementation

```rust
pub fn eval_logits_once(model_path: &str, tokens: &[i32]) -> Result<Vec<f32>> {
    let (config, model) = load_gguf(Path::new(model_path), Device::Cpu)
        .map_err(|e| anyhow::anyhow!("Failed to load model from {}: {}", model_path, e))?;

    let model = BitNetModel::from_gguf(config.clone(), model, Device::Cpu)?;

    // ... rest of the function ...
}
```
