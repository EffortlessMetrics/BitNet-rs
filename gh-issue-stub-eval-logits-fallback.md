# [Model Loading] Remove silent fallback to mock model in logits evaluation

## Problem Description

The `eval_logits_once` function silently falls back to a mock model when real model loading fails, masking important errors and producing unreliable results for parity testing.

## Root Cause Analysis

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

## Proposed Solution

```rust
pub fn eval_logits_once(model_path: &str, tokens: &[i32]) -> Result<Vec<f32>> {
    // Fail fast with clear error messages
    let (config, model) = load_gguf(Path::new(model_path), Device::Cpu)
        .map_err(|e| anyhow::anyhow!("Failed to load model from {}: {}", model_path, e))?;

    let model = BitNetModel::from_gguf(config.clone(), model, Device::Cpu)
        .map_err(|e| anyhow::anyhow!("Failed to create BitNet model: {}", e))?;

    // Proceed with actual model evaluation
    let input_tensor = create_input_tensor(tokens, &config)?;
    let logits = model.forward(&input_tensor)?;

    Ok(logits.to_vec1()?)
}

// Separate function for testing with mock models
#[cfg(test)]
pub fn eval_logits_with_mock(tokens: &[i32]) -> Result<Vec<f32>> {
    let cfg = BitNetConfig::default();
    let model = BitNetModel::new(cfg.clone(), Device::Cpu);

    let input_tensor = create_input_tensor(tokens, &cfg)?;
    let logits = model.forward(&input_tensor)?;

    Ok(logits.to_vec1()?)
}
```

## Acceptance Criteria

- [ ] No silent fallback to mock models
- [ ] Clear error messages for model loading failures
- [ ] Separate mock model functions for testing
- [ ] Reliable parity testing with real models

## Priority: High