# [SIMULATION] Layer Normalization Falls Back to RMSNorm When Bias Missing - Incorrect Behavior

## Problem Description

The `layer_norm_with_optional_bias` function in `crates/bitnet-models/src/transformer.rs` automatically falls back to RMSNorm when the bias tensor is missing, masking potential model loading errors and creating inconsistent normalization behavior. This silent fallback can lead to model accuracy degradation and debugging difficulties, as RMSNorm and LayerNorm have different mathematical properties.

## Environment

- **File**: `crates/bitnet-models/src/transformer.rs`
- **Function**: `layer_norm_with_optional_bias` (lines 11-32)
- **Component**: Transformer model loading and layer construction
- **Build Configuration**: All feature configurations
- **Model Formats**: GGUF, SafeTensors with embedded layer normalization

## Root Cause Analysis

### Technical Issues

1. **Silent Fallback Behavior**:
   ```rust
   match vb.get((normalized_shape,), "bias") {
       Ok(bias) => {
           // Standard LayerNorm with bias
           Ok(LayerNorm::new(weight, bias, eps))
       }
       Err(_) => {
           // Silent fallback to RMSNorm - PROBLEMATIC
           tracing::debug!("Bias tensor missing; using RMSNorm");
           Ok(LayerNorm::rms_norm(weight, eps))
       }
   }
   ```

2. **Mathematical Inconsistency**:
   - **LayerNorm**: `output = γ * (x - μ) / σ + β` (mean and variance normalization)
   - **RMSNorm**: `output = γ * x / √(E[x²])` (only variance normalization)
   - Different normalization statistics affect model behavior and accuracy

3. **Error Masking**:
   - Legitimate model loading errors are hidden by fallback
   - Missing bias tensors may indicate corrupted or incompatible models
   - Debugging becomes difficult when normalization type changes silently

4. **Inconsistent Model Behavior**:
   - Same model architecture may use different normalization depending on tensor availability
   - Cross-validation with reference implementations becomes unreliable
   - Model accuracy may degrade without clear indication

### Impact Assessment

- **Accuracy**: Potential model accuracy degradation due to incorrect normalization
- **Debugging**: Difficult to identify model loading issues and normalization mismatches
- **Compatibility**: Inconsistent behavior with reference implementations
- **Reliability**: Silent errors reduce system trustworthiness

## Reproduction Steps

1. Create a model with missing bias tensors in layer normalization:
   ```bash
   # Load model with incomplete layer norm specification
   cargo run -p bitnet-cli -- load-model incomplete_layernorm_model.gguf
   ```

2. Observe the function behavior:
   ```rust
   let layer_norm = layer_norm_with_optional_bias(768, 1e-5, vb)?;
   // Function succeeds but uses different normalization type
   ```

3. **Expected**: Clear error indicating missing bias tensor
4. **Actual**: Silent fallback to RMSNorm with debug message only

## Proposed Solution

### Primary Approach: Explicit Normalization Type Specification

Replace the silent fallback with explicit normalization type configuration:

```rust
#[derive(Debug, Clone, Copy)]
pub enum NormalizationType {
    LayerNorm,      // Standard LayerNorm with bias
    LayerNormNoBias, // LayerNorm without bias (zero bias)
    RMSNorm,        // RMS normalization
}

fn create_normalization_layer(
    normalized_shape: usize,
    eps: f64,
    norm_type: NormalizationType,
    vb: VarBuilder,
) -> candle_core::Result<LayerNorm> {
    let weight = vb.get((normalized_shape,), "weight")?;

    match norm_type {
        NormalizationType::LayerNorm => {
            let bias = vb.get((normalized_shape,), "bias")
                .map_err(|e| candle_core::Error::Msg(
                    format!("LayerNorm requires bias tensor but it's missing: {}", e)
                ))?;
            tracing::debug!("Creating LayerNorm with bias [{}]", normalized_shape);
            Ok(LayerNorm::new(weight, bias, eps))
        }
        NormalizationType::LayerNormNoBias => {
            // Explicitly create zero bias for LayerNorm without bias
            let bias = Tensor::zeros(normalized_shape, DType::F32, vb.device())?;
            tracing::debug!("Creating LayerNorm with zero bias [{}]", normalized_shape);
            Ok(LayerNorm::new(weight, bias, eps))
        }
        NormalizationType::RMSNorm => {
            tracing::debug!("Creating RMSNorm [{}]", normalized_shape);
            Ok(LayerNorm::rms_norm(weight, eps))
        }
    }
}

// Updated function with explicit error handling
fn layer_norm_with_explicit_type(
    normalized_shape: usize,
    eps: f64,
    vb: VarBuilder,
) -> candle_core::Result<LayerNorm> {
    let weight = vb.get((normalized_shape,), "weight")?;

    // Check if bias exists and handle explicitly
    match vb.get((normalized_shape,), "bias") {
        Ok(bias) => {
            tracing::debug!("Found bias tensor, creating LayerNorm [{}]", normalized_shape);
            Ok(LayerNorm::new(weight, bias, eps))
        }
        Err(e) => {
            // Return clear error instead of silent fallback
            Err(candle_core::Error::Msg(format!(
                "LayerNorm bias tensor missing for layer with shape [{}]. \
                 Original error: {}. \
                 If RMSNorm is intended, please use the appropriate constructor.",
                normalized_shape, e
            )))
        }
    }
}

// Model configuration-based approach
#[derive(Debug, Clone)]
pub struct ModelNormConfig {
    pub norm_type: NormalizationType,
    pub eps: f64,
    pub require_bias: bool,
}

impl ModelNormConfig {
    pub fn from_model_metadata(metadata: &ModelMetadata) -> Result<Self> {
        // Determine normalization type from model configuration
        let norm_type = match metadata.architecture.as_str() {
            "llama" | "mistral" => NormalizationType::RMSNorm,
            "gpt2" | "bert" => NormalizationType::LayerNorm,
            arch => {
                return Err(Error::Msg(format!(
                    "Unknown architecture normalization for: {}", arch
                )));
            }
        };

        Ok(ModelNormConfig {
            norm_type,
            eps: metadata.layer_norm_eps.unwrap_or(1e-5),
            require_bias: norm_type == NormalizationType::LayerNorm,
        })
    }
}

fn layer_norm_from_config(
    normalized_shape: usize,
    config: &ModelNormConfig,
    vb: VarBuilder,
) -> candle_core::Result<LayerNorm> {
    create_normalization_layer(
        normalized_shape,
        config.eps,
        config.norm_type,
        vb,
    )
}
```

### Alternative Approaches

1. **Configuration-Driven**: Use model configuration to specify normalization type
2. **Validation Mode**: Add strict validation mode that prevents any fallbacks
3. **Warning System**: Maintain fallback but add prominent warnings and metrics

## Implementation Plan

### Phase 1: Error Handling Improvement (Priority: Critical)
- [ ] Replace silent fallback with explicit error reporting
- [ ] Add clear error messages indicating missing bias tensors
- [ ] Update function documentation to clarify expected behavior
- [ ] Add unit tests for error conditions

### Phase 2: Explicit Configuration System (Priority: High)
- [ ] Implement `NormalizationType` enum and configuration
- [ ] Add model metadata parsing for normalization type detection
- [ ] Create constructor functions for each normalization type
- [ ] Add validation for normalization type consistency

### Phase 3: Model Loading Integration (Priority: High)
- [ ] Update model loading to specify normalization type explicitly
- [ ] Add GGUF metadata parsing for normalization configuration
- [ ] Implement SafeTensors normalization type detection
- [ ] Add backward compatibility for existing models

### Phase 4: Validation & Testing (Priority: Medium)
- [ ] Cross-validation with reference implementations
- [ ] Add comprehensive test suite for all normalization types
- [ ] Performance validation for different normalization approaches
- [ ] Documentation updates and migration guide

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_layer_norm_explicit_error_on_missing_bias() {
    let device = Device::Cpu;
    let vb = VarBuilder::from_tensors(
        HashMap::from([
            ("weight".to_string(), Tensor::ones(768, DType::F32, &device).unwrap())
        ]),
        DType::F32,
        &device,
    );

    let result = layer_norm_with_explicit_type(768, 1e-5, vb);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("bias tensor missing"));
}

#[test]
fn test_normalization_type_from_model_config() {
    let llama_config = ModelMetadata {
        architecture: "llama".to_string(),
        layer_norm_eps: Some(1e-6),
        ..Default::default()
    };

    let norm_config = ModelNormConfig::from_model_metadata(&llama_config).unwrap();
    assert_eq!(norm_config.norm_type, NormalizationType::RMSNorm);
    assert_eq!(norm_config.eps, 1e-6);
}

#[test]
fn test_layer_norm_mathematical_correctness() {
    let input = create_test_tensor(32, 768);
    let weight = Tensor::ones(768, DType::F32, &Device::Cpu).unwrap();
    let bias = Tensor::zeros(768, DType::F32, &Device::Cpu).unwrap();

    let layer_norm = LayerNorm::new(weight.clone(), bias, 1e-5);
    let rms_norm = LayerNorm::rms_norm(weight, 1e-5);

    let ln_output = layer_norm.forward(&input).unwrap();
    let rms_output = rms_norm.forward(&input).unwrap();

    // LayerNorm and RMSNorm should produce different outputs
    assert!(!tensors_equal(&ln_output, &rms_output, 1e-6));
}
```

### Integration Tests
```bash
# Test with various model architectures
cargo test --no-default-features --features cpu test_model_loading_normalization

# Cross-validation with reference implementations
cargo run -p xtask -- crossval --component layer_norm

# Model compatibility testing
cargo test test_normalization_model_compatibility
```

### Regression Tests
- Ensure existing models still load correctly
- Validate that explicit configuration produces identical results
- Test error handling doesn't break existing functionality

## Acceptance Criteria

### Functional Requirements
- [ ] No silent fallbacks to different normalization types
- [ ] Clear error messages for missing required tensors
- [ ] Explicit configuration for normalization type selection
- [ ] Backward compatibility with existing model loading

### Quality Requirements
- [ ] 100% test coverage for error conditions
- [ ] Mathematical correctness validation for each normalization type
- [ ] Clear documentation of normalization behavior
- [ ] Migration guide for existing code

### Performance Requirements
- [ ] No performance regression in model loading
- [ ] Efficient tensor creation for explicit zero bias
- [ ] Minimal overhead for configuration parsing

## Related Issues

- Model loading and validation framework improvements
- Cross-validation accuracy issues with reference implementations
- Documentation updates for normalization layer behavior
- Error handling consistency across model loading

## Dependencies

- Candle tensor operations for LayerNorm construction
- Model metadata parsing utilities
- Error handling and logging infrastructure
- Backward compatibility testing framework

## Migration Impact

- **Breaking Change**: Functions may now return errors instead of silent fallbacks
- **Configuration**: New explicit configuration required for normalization type
- **Testing**: Existing tests may need updates for error handling
- **Documentation**: Clear migration path for affected code

---

**Labels**: `critical`, `simulation`, `model-loading`, `layer-normalization`, `error-handling`, `mathematical-correctness`
**Assignee**: Core team member with model loading and tensor operations experience
**Milestone**: Robust Model Loading (v0.3.0)
**Estimated Effort**: 1-2 weeks for implementation and comprehensive testing