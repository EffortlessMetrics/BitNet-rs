# [Inference] Remove Mock Model Fallback in eval_logits_once Function

## Problem Description

The `eval_logits_once` function in `crates/bitnet-inference/src/parity.rs` currently falls back to a mock model when GGUF model loading fails. This silent fallback masks real issues with model loading and can lead to incorrect inference results being returned without any indication that the actual model failed to load.

## Environment

- **Component**: `crates/bitnet-inference/src/parity.rs`
- **Function**: `eval_logits_once`
- **Feature Context**: Cross-validation and parity testing functionality
- **Impact**: Model loading reliability and debugging capabilities

## Current Implementation Analysis

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

**Issues Identified:**
1. **Silent failure**: Model loading errors are completely ignored and swallowed
2. **Mock model fallback**: Returns results from a default/empty model instead of real model
3. **Misleading results**: Users receive output that appears valid but comes from wrong model
4. **Debugging difficulties**: Makes it impossible to diagnose model loading issues
5. **Loss of error context**: Original error information is discarded
6. **Inconsistent behavior**: Function may succeed even when intended model is unusable

## Impact Assessment

**Severity**: Medium-High
**Affected Users**: Developers running cross-validation tests, users debugging model loading issues
**Functional Impact**:
- Incorrect inference results masquerading as correct output
- Inability to diagnose model loading problems
- False positive test results in cross-validation
- Waste of compute resources on meaningless mock inference

## Root Cause Analysis

The current implementation appears to prioritize "graceful degradation" over correctness, but in the context of a `eval_logits_once` function used for testing and validation, returning incorrect results is worse than failing with a clear error. This pattern indicates:

1. **Misplaced error handling**: Fallbacks are appropriate for user-facing applications, not validation functions
2. **Testing anti-pattern**: Test functions should fail fast and provide clear diagnostics
3. **Hidden dependencies**: The mock model behavior may not match real model behavior

## Proposed Solution

### 1. Proper Error Propagation

Replace the silent fallback with explicit error handling that provides clear diagnostic information:

```rust
pub fn eval_logits_once(model_path: &str, tokens: &[i32]) -> Result<Vec<f32>> {
    // Validate input parameters
    if model_path.is_empty() {
        bail!("Model path cannot be empty");
    }

    if tokens.is_empty() {
        bail!("Token sequence cannot be empty");
    }

    // Attempt to load model with detailed error context
    let model_path = Path::new(model_path);
    let (config, tensors) = load_gguf(model_path, Device::Cpu)
        .with_context(|| format!("Failed to load GGUF model from {}", model_path.display()))?;

    // Validate model configuration
    validate_model_config(&config)
        .with_context(|| "Model configuration validation failed")?;

    // Construct BitNet model from loaded tensors
    let model = BitNetModel::from_gguf(config.clone(), tensors, Device::Cpu)
        .with_context(|| "Failed to construct BitNet model from GGUF tensors")?;

    // Validate tokens against model vocabulary
    validate_tokens_for_model(&model, tokens)
        .with_context(|| "Token validation failed for model")?;

    // Perform actual inference
    eval_logits_with_model(&model, &config, tokens)
        .with_context(|| "Inference execution failed")
}

fn validate_model_config(config: &BitNetConfig) -> Result<()> {
    if config.vocab_size == 0 {
        bail!("Model configuration has invalid vocabulary size: {}", config.vocab_size);
    }

    if config.hidden_size == 0 {
        bail!("Model configuration has invalid hidden size: {}", config.hidden_size);
    }

    if config.num_layers == 0 {
        bail!("Model configuration has invalid number of layers: {}", config.num_layers);
    }

    if config.num_attention_heads == 0 {
        bail!("Model configuration has invalid number of attention heads: {}", config.num_attention_heads);
    }

    // Validate that hidden_size is divisible by num_attention_heads
    if config.hidden_size % config.num_attention_heads != 0 {
        bail!(
            "Hidden size {} is not divisible by number of attention heads {}",
            config.hidden_size,
            config.num_attention_heads
        );
    }

    Ok(())
}

fn validate_tokens_for_model(model: &BitNetModel, tokens: &[i32]) -> Result<()> {
    let vocab_size = model.vocab_size() as i32;

    for (idx, &token) in tokens.iter().enumerate() {
        if token < 0 {
            bail!("Invalid negative token {} at position {}", token, idx);
        }

        if token >= vocab_size {
            bail!(
                "Token {} at position {} exceeds vocabulary size {}",
                token, idx, vocab_size
            );
        }
    }

    // Check for reasonable sequence length
    if tokens.len() > model.max_sequence_length() {
        bail!(
            "Token sequence length {} exceeds model maximum {}",
            tokens.len(),
            model.max_sequence_length()
        );
    }

    Ok(())
}

fn eval_logits_with_model(
    model: &BitNetModel,
    config: &BitNetConfig,
    tokens: &[i32],
) -> Result<Vec<f32>> {
    // Convert tokens to appropriate tensor format
    let token_tensor = create_token_tensor(tokens)?;

    // Perform forward pass
    let logits = model.forward(&token_tensor)
        .with_context(|| "Model forward pass failed")?;

    // Extract and validate logits
    let logits_vec = extract_logits_vector(logits, config.vocab_size)
        .with_context(|| "Failed to extract logits from model output")?;

    // Validate output sanity
    validate_logits_output(&logits_vec)
        .with_context(|| "Logits output validation failed")?;

    Ok(logits_vec)
}

fn create_token_tensor(tokens: &[i32]) -> Result<BitNetTensor> {
    let device = Device::Cpu;
    let token_data: Vec<u32> = tokens.iter()
        .map(|&t| t as u32)
        .collect();

    let tensor = BitNetTensor::from_vec(
        token_data,
        &[1, tokens.len()], // batch_size=1, seq_len=tokens.len()
        &device,
    )?;

    Ok(tensor)
}

fn extract_logits_vector(logits_tensor: BitNetTensor, vocab_size: usize) -> Result<Vec<f32>> {
    // Validate tensor dimensions
    let dims = logits_tensor.dims();
    if dims.len() != 2 {
        bail!("Expected 2D logits tensor, got {}D tensor with shape {:?}", dims.len(), dims);
    }

    if dims[1] != vocab_size {
        bail!(
            "Logits tensor vocabulary dimension {} doesn't match expected {}",
            dims[1], vocab_size
        );
    }

    // Extract the last token's logits (for next token prediction)
    let last_token_logits = logits_tensor.slice(0, dims[0] - 1)?;
    let logits_vec = last_token_logits.to_vec1::<f32>()?;

    Ok(logits_vec)
}

fn validate_logits_output(logits: &[f32]) -> Result<()> {
    if logits.is_empty() {
        bail!("Logits output is empty");
    }

    // Check for NaN or infinite values
    for (idx, &logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            bail!("Non-finite logit value {} at position {}", logit, idx);
        }
    }

    // Sanity check: logits should have reasonable dynamic range
    let min_logit = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if (max_logit - min_logit) < 1e-6 {
        warn!("Logits have very small dynamic range: {} to {}", min_logit, max_logit);
    }

    if (max_logit - min_logit) > 100.0 {
        warn!("Logits have very large dynamic range: {} to {}", min_logit, max_logit);
    }

    Ok(())
}
```

### 2. Enhanced Error Context and Diagnostics

```rust
pub fn eval_logits_once_with_diagnostics(
    model_path: &str,
    tokens: &[i32],
) -> Result<(Vec<f32>, InferenceDiagnostics)> {
    let start_time = std::time::Instant::now();

    // Collect diagnostic information during execution
    let mut diagnostics = InferenceDiagnostics::new();

    // File system validation
    let model_path = Path::new(model_path);
    diagnostics.model_path = model_path.to_string_lossy().to_string();
    diagnostics.model_exists = model_path.exists();
    diagnostics.model_size = std::fs::metadata(model_path)
        .map(|m| m.len())
        .unwrap_or(0);

    if !diagnostics.model_exists {
        bail!("Model file does not exist: {}", model_path.display());
    }

    // Model loading diagnostics
    let load_start = std::time::Instant::now();
    let (config, tensors) = load_gguf(model_path, Device::Cpu)?;
    diagnostics.load_time = load_start.elapsed();

    // Model construction diagnostics
    let construct_start = std::time::Instant::now();
    let model = BitNetModel::from_gguf(config.clone(), tensors, Device::Cpu)?;
    diagnostics.construct_time = construct_start.elapsed();

    // Inference diagnostics
    let inference_start = std::time::Instant::now();
    let logits = eval_logits_with_model(&model, &config, tokens)?;
    diagnostics.inference_time = inference_start.elapsed();

    diagnostics.total_time = start_time.elapsed();
    diagnostics.model_config = config;
    diagnostics.input_tokens = tokens.to_vec();
    diagnostics.output_logits_shape = logits.len();

    Ok((logits, diagnostics))
}

#[derive(Debug, Clone)]
pub struct InferenceDiagnostics {
    pub model_path: String,
    pub model_exists: bool,
    pub model_size: u64,
    pub load_time: std::time::Duration,
    pub construct_time: std::time::Duration,
    pub inference_time: std::time::Duration,
    pub total_time: std::time::Duration,
    pub model_config: BitNetConfig,
    pub input_tokens: Vec<i32>,
    pub output_logits_shape: usize,
}

impl InferenceDiagnostics {
    fn new() -> Self {
        Self {
            model_path: String::new(),
            model_exists: false,
            model_size: 0,
            load_time: std::time::Duration::ZERO,
            construct_time: std::time::Duration::ZERO,
            inference_time: std::time::Duration::ZERO,
            total_time: std::time::Duration::ZERO,
            model_config: BitNetConfig::default(),
            input_tokens: Vec::new(),
            output_logits_shape: 0,
        }
    }

    pub fn print_summary(&self) {
        println!("=== Inference Diagnostics ===");
        println!("Model: {}", self.model_path);
        println!("Model Size: {:.2} MB", self.model_size as f64 / 1024.0 / 1024.0);
        println!("Load Time: {:.2}ms", self.load_time.as_secs_f64() * 1000.0);
        println!("Construct Time: {:.2}ms", self.construct_time.as_secs_f64() * 1000.0);
        println!("Inference Time: {:.2}ms", self.inference_time.as_secs_f64() * 1000.0);
        println!("Total Time: {:.2}ms", self.total_time.as_secs_f64() * 1000.0);
        println!("Input Tokens: {} tokens", self.input_tokens.len());
        println!("Output Shape: {} logits", self.output_logits_shape);
        println!("Model Config: {} layers, {} heads, {} hidden",
                 self.model_config.num_layers,
                 self.model_config.num_attention_heads,
                 self.model_config.hidden_size);
    }
}
```

### 3. Backward Compatibility and Migration

```rust
// For backward compatibility, provide a deprecated wrapper
#[deprecated(
    since = "0.3.0",
    note = "Use eval_logits_once or eval_logits_once_with_diagnostics instead"
)]
pub fn eval_logits_once_fallback(model_path: &str, tokens: &[i32]) -> Result<Vec<f32>> {
    warn!("Using deprecated eval_logits_once_fallback function");

    match eval_logits_once(model_path, tokens) {
        Ok(logits) => Ok(logits),
        Err(e) => {
            warn!("Model loading failed: {}", e);
            warn!("Falling back to mock model (deprecated behavior)");

            // Create mock logits with warning
            let mock_logits = create_mock_logits(32000); // Typical vocab size
            Ok(mock_logits)
        }
    }
}

fn create_mock_logits(vocab_size: usize) -> Vec<f32> {
    let mut logits = vec![0.0; vocab_size];

    // Create a simple uniform distribution with slight perturbation
    let base_logit = -((vocab_size as f32).ln());
    for (i, logit) in logits.iter_mut().enumerate() {
        *logit = base_logit + (i as f32 * 0.001).sin() * 0.1;
    }

    logits
}
```

## Implementation Breakdown

### Phase 1: Error Handling Improvement
- [ ] Remove mock model fallback logic
- [ ] Add comprehensive error context and propagation
- [ ] Implement input validation functions
- [ ] Add unit tests for error conditions

### Phase 2: Enhanced Diagnostics
- [ ] Implement diagnostic data collection
- [ ] Add performance timing measurements
- [ ] Create diagnostic reporting functionality
- [ ] Add integration tests with diagnostics

### Phase 3: Validation Framework
- [ ] Implement model configuration validation
- [ ] Add token sequence validation
- [ ] Create logits output validation
- [ ] Add comprehensive test coverage

### Phase 4: Backward Compatibility
- [ ] Create deprecated fallback function
- [ ] Add migration documentation
- [ ] Update existing callers
- [ ] Add integration tests for migration path

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_logits_once_with_valid_model() {
        let model_path = "tests/data/test_model.gguf";
        let tokens = vec![1, 2, 3, 4, 5];

        let result = eval_logits_once(model_path, &tokens);
        assert!(result.is_ok());

        let logits = result.unwrap();
        assert!(!logits.is_empty());
        assert!(logits.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_eval_logits_once_with_nonexistent_model() {
        let model_path = "nonexistent/model.gguf";
        let tokens = vec![1, 2, 3];

        let result = eval_logits_once(model_path, &tokens);
        assert!(result.is_err());

        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("nonexistent/model.gguf"));
    }

    #[test]
    fn test_eval_logits_once_with_empty_tokens() {
        let model_path = "tests/data/test_model.gguf";
        let tokens = vec![];

        let result = eval_logits_once(model_path, &tokens);
        assert!(result.is_err());

        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("Token sequence cannot be empty"));
    }

    #[test]
    fn test_eval_logits_once_with_invalid_tokens() {
        let model_path = "tests/data/test_model.gguf";
        let tokens = vec![-1, 2, 3]; // negative token

        let result = eval_logits_once(model_path, &tokens);
        assert!(result.is_err());

        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("Invalid negative token"));
    }

    #[test]
    fn test_diagnostics_collection() {
        let model_path = "tests/data/test_model.gguf";
        let tokens = vec![1, 2, 3];

        let result = eval_logits_once_with_diagnostics(model_path, &tokens);
        assert!(result.is_ok());

        let (logits, diagnostics) = result.unwrap();
        assert!(!logits.is_empty());
        assert!(diagnostics.total_time > std::time::Duration::ZERO);
        assert!(diagnostics.model_exists);
        assert_eq!(diagnostics.input_tokens, tokens);
    }
}
```

### Integration Tests
```rust
#[cfg(test)]
mod integration_tests {
    #[test]
    fn test_cross_validation_error_propagation() {
        // Test that cross-validation properly handles model loading errors
        let invalid_model_path = "invalid.gguf";
        let tokens = vec![1, 2, 3];

        let result = eval_logits_once(invalid_model_path, &tokens);
        assert!(result.is_err());

        // Ensure error contains useful information
        let error = result.unwrap_err();
        assert!(error.chain().any(|e| e.to_string().contains("invalid.gguf")));
    }

    #[test]
    fn test_real_model_inference() {
        // Test with an actual model file if available
        if let Ok(model_path) = std::env::var("BITNET_TEST_MODEL") {
            let tokens = vec![1, 2, 3, 4, 5];
            let result = eval_logits_once(&model_path, &tokens);

            match result {
                Ok(logits) => {
                    assert!(!logits.is_empty());
                    assert!(logits.iter().all(|&x| x.is_finite()));
                }
                Err(e) => {
                    panic!("Failed to evaluate logits with real model: {}", e);
                }
            }
        }
    }
}
```

## Risk Assessment

**Low Risk Changes:**
- Adding input validation functions
- Implementing diagnostic data collection

**Medium Risk Changes:**
- Removing mock model fallback
- Changing error handling behavior

**High Risk Changes:**
- Modifying function signature or behavior for existing callers

**Mitigation Strategies:**
- Comprehensive testing with various error conditions
- Backward compatibility layer with deprecation warnings
- Clear migration documentation
- Gradual rollout with feature flags

## Acceptance Criteria

- [ ] Function fails fast with clear error messages for invalid inputs
- [ ] No silent fallbacks to mock models
- [ ] Comprehensive input validation (file existence, token validity, model format)
- [ ] Detailed error context including file paths and failure reasons
- [ ] Diagnostic information collection for performance analysis
- [ ] Backward compatibility maintained through deprecated wrapper
- [ ] 100% test coverage for error conditions
- [ ] Integration tests pass with real model files

## Related Issues/PRs

- **Related to**: Model loading reliability improvements
- **Depends on**: GGUF loader error handling enhancements
- **Blocks**: Cross-validation framework reliability
- **References**: Testing infrastructure improvements

## Additional Context

This change is crucial for improving the reliability and debuggability of the BitNet.rs inference pipeline. By removing silent fallbacks and providing clear error messages, developers will be able to quickly identify and resolve model loading issues, leading to more robust applications and better user experiences.
