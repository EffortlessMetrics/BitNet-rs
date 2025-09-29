# [ENHANCEMENT] Remove mock model fallback in eval_logits_once for better error handling

## Problem Description
The `eval_logits_once` function in `crates/bitnet-inference/src/parity.rs` falls back to a mock model when real model loading fails, potentially hiding legitimate model loading issues.

## Environment
- **File**: `crates/bitnet-inference/src/parity.rs`
- **Function**: `eval_logits_once`
- **Current State**: Silent fallback to mock model on load failure

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

**Issues:**
1. Silent fallback to mock model masks real loading failures
2. Users may unknowingly get mock results instead of real inference
3. Debugging model loading issues becomes difficult
4. Inconsistent behavior depending on file availability

## Proposed Solution
```rust
pub fn eval_logits_once(model_path: &str, tokens: &[i32]) -> Result<Vec<f32>> {
    // Load model with explicit error handling
    let (config, tensors) = load_gguf(Path::new(model_path), Device::Cpu)
        .with_context(|| format!("Failed to load GGUF model from '{}'", model_path))?;

    let model = BitNetModel::from_gguf(config.clone(), tensors, Device::Cpu)
        .with_context(|| "Failed to create BitNet model from GGUF data")?;

    // Validate tokens are within vocabulary
    let vocab_size = config.model.vocab_size;
    for &token in tokens {
        if token < 0 || token as usize >= vocab_size {
            return Err(anyhow::anyhow!(
                "Token {} is out of vocabulary range [0, {})",
                token, vocab_size
            ));
        }
    }

    // Convert tokens to appropriate format
    let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();

    // Perform inference
    let logits = model.forward_tokens(&token_ids)
        .with_context(|| "Forward pass failed during logits evaluation")?;

    // Extract final logits (for last token)
    let final_logits = logits.get_last_token_logits()
        .with_context(|| "Failed to extract final token logits")?;

    Ok(final_logits)
}

// Add test-specific function for mock model usage
#[cfg(test)]
pub fn eval_logits_once_mock(tokens: &[i32]) -> Result<Vec<f32>> {
    let config = BitNetConfig::default();
    let model = BitNetModel::new(config.clone(), Device::Cpu);

    let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let logits = model.forward_tokens(&token_ids)?;
    let final_logits = logits.get_last_token_logits()?;

    Ok(final_logits)
}

// Enhanced version with fallback options
pub fn eval_logits_once_with_fallback(
    model_path: &str,
    tokens: &[i32],
    fallback_options: &FallbackOptions,
) -> Result<(Vec<f32>, ModelSource)> {
    // Try primary model path
    match eval_logits_once(model_path, tokens) {
        Ok(logits) => Ok((logits, ModelSource::Primary(model_path.to_string()))),
        Err(primary_error) => {
            if fallback_options.allow_fallback {
                // Try fallback paths if configured
                for fallback_path in &fallback_options.fallback_paths {
                    if let Ok(logits) = eval_logits_once(fallback_path, tokens) {
                        tracing::warn!(
                            "Primary model '{}' failed, using fallback '{}'",
                            model_path, fallback_path
                        );
                        return Ok((logits, ModelSource::Fallback(fallback_path.clone())));
                    }
                }
            }

            // Return original error if no fallbacks work
            Err(primary_error)
        }
    }
}

#[derive(Debug, Clone)]
pub struct FallbackOptions {
    pub allow_fallback: bool,
    pub fallback_paths: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ModelSource {
    Primary(String),
    Fallback(String),
}
```

## Implementation Plan
### Phase 1: Error Handling Enhancement (1 day)
- [ ] Remove silent mock model fallback
- [ ] Add comprehensive error messages with context
- [ ] Implement proper token validation

### Phase 2: Structured Fallback System (1 day)
- [ ] Create optional fallback mechanism with explicit configuration
- [ ] Add test-specific mock function for unit tests
- [ ] Implement model source tracking

### Phase 3: Documentation & Testing (1 day)
- [ ] Update function documentation to clarify behavior
- [ ] Add tests for error scenarios
- [ ] Create migration guide for existing code

## Acceptance Criteria
- [ ] No silent fallback to mock models
- [ ] Clear error messages for model loading failures
- [ ] Optional explicit fallback mechanism
- [ ] Separate test utilities for mock models
- [ ] Improved debugging experience

**Labels**: `enhancement`, `error-handling`, `reliability`, `P2-medium`
**Effort**: 3 days