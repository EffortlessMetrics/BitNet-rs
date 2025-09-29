# [Model Loading] Replace GGUF minimal loader mock tensor creation with proper error handling

## Problem Description

The `load_gguf_minimal` and `create_mock_tensor_layout` functions in `crates/bitnet-models/src/gguf_simple.rs` create mock tensors instead of returning appropriate errors when encountering incomplete or invalid GGUF files. This stub implementation masks real parsing failures and can lead to unexpected behavior in production.

## Environment

- **File**: `crates/bitnet-models/src/gguf_simple.rs`
- **Functions**: `load_gguf_minimal`, `create_mock_tensor_layout`
- **Component**: GGUF model loading and parsing
- **Features**: Model loading, GGUF format support
- **MSRV**: Rust 1.90.0

## Reproduction Steps

1. Load an incomplete or corrupted GGUF file using the minimal loader
2. Observe that mock tensors are created for missing transformer layers
3. Note that `create_mock_tensor_layout` generates fake tensor structures
4. Verify that errors are not properly propagated to calling code

**Expected**: Clear error messages for incomplete or invalid GGUF files
**Actual**: Mock tensors created silently, masking real parsing issues

## Root Cause Analysis

The current implementation prioritizes fallback functionality over proper error reporting:

**Technical Issues:**
1. **Silent Failure Masking**: Mock tensors hide real GGUF parsing failures
2. **Production Risk**: Invalid models may appear to load successfully
3. **Debugging Difficulty**: Hard to identify root cause of model loading issues
4. **Test Confusion**: Mock helpers not clearly marked as test-only utilities

**Current Problematic Implementation:**
```rust
fn load_gguf_minimal(
    path: &Path,
    device: Device,
) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
    let two = match crate::gguf_min::load_two(path) {
        Ok(two_tensors) => two_tensors,
        Err(_) => {
            // Instead of propagating error, creates mock tensors
            return create_mock_tensor_layout(device);
        }
    };

    // ... potentially creates mock tensors for missing layers
    for layer in 0..num_layers {
        let prefix = format!("blk.{}", layer);
        if !tensor_map.contains_key(&format!("{}.attn_q.weight", prefix)) {
            // Creates mock tensor instead of erroring
            let mock_tensor = create_mock_attention_weight(device);
            tensor_map.insert(format!("{}.attn_q.weight", prefix), mock_tensor);
        }
    }
}

fn create_mock_tensor_layout(device: Device) -> Result<(BitNetConfig, HashMap<String, CandleTensor>)> {
    // Creates fake model structure for test compatibility
    // Should be marked as test-only or removed
}
```

**Impact on System:**
- Incomplete models appear to load successfully
- Runtime failures occur later during inference
- Hard to distinguish between valid fallback and parsing errors
- Test utilities pollute production code paths

## Impact Assessment

**Severity**: Medium - Can cause silent failures and debugging difficulties
**Type**: Error handling and code organization improvement

**Affected Components**:
- GGUF model loading pipeline
- Model validation and compatibility checking
- Error reporting and debugging experience
- Test infrastructure organization

**Problems Caused**:
- **Silent Failures**: Incomplete models appear to load successfully
- **Runtime Errors**: Failures manifest later during inference rather than at load time
- **Debugging Difficulty**: Hard to identify whether failure is due to incomplete model or parsing bug
- **Test Pollution**: Test utilities mixed with production code

## Proposed Solution

### Primary Solution: Proper Error Handling with Validation

Replace mock tensor creation with comprehensive error reporting and validation:

```rust
// Enhanced error types for better diagnostics
#[derive(Debug, thiserror::Error)]
pub enum GgufLoadingError {
    #[error("Failed to parse GGUF header: {reason}")]
    HeaderParsingFailed { reason: String },

    #[error("Incomplete GGUF file: missing required tensors: {missing_tensors:?}")]
    IncompleteTensorSet { missing_tensors: Vec<String> },

    #[error("Unsupported GGUF version: {version} (supported: {supported:?})")]
    UnsupportedVersion { version: String, supported: Vec<String> },

    #[error("Invalid tensor metadata: {tensor_name} - {reason}")]
    InvalidTensorMetadata { tensor_name: String, reason: String },

    #[error("GGUF file corruption detected: {details}")]
    FileCorruption { details: String },
}

// Improved load_gguf_minimal with proper error handling
fn load_gguf_minimal(
    path: &Path,
    device: Device,
) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>), GgufLoadingError> {
    info!("Loading GGUF model from: {}", path.display());

    // Attempt to load with enhanced parser first
    let two = crate::gguf_min::load_two(path)
        .map_err(|e| GgufLoadingError::HeaderParsingFailed {
            reason: format!("Enhanced parser failed: {}", e)
        })?;

    // Extract model configuration with validation
    let config = extract_model_config(&two)
        .ok_or_else(|| GgufLoadingError::HeaderParsingFailed {
            reason: "Could not extract valid model configuration".to_string()
        })?;

    let num_layers = config.model.num_layers;
    info!("Model configuration: {} layers, {} hidden size", num_layers, config.model.hidden_size);

    // Build tensor map and validate completeness
    let mut tensor_map = HashMap::new();
    let mut missing_tensors = Vec::new();

    // Validate embedding layers
    if let Some(embedding) = two.get("token_embd.weight") {
        tensor_map.insert("token_embd.weight".to_string(), embedding.to_device(&device)?);
    } else {
        missing_tensors.push("token_embd.weight".to_string());
    }

    // Validate output layers
    if let Some(output) = two.get("output.weight") {
        tensor_map.insert("output.weight".to_string(), output.to_device(&device)?);
    } else {
        missing_tensors.push("output.weight".to_string());
    }

    // Validate transformer layers
    for layer in 0..num_layers {
        let layer_tensors = validate_transformer_layer(&two, layer, &device)?;
        for (name, tensor) in layer_tensors {
            if let Some(tensor) = tensor {
                tensor_map.insert(name, tensor);
            } else {
                missing_tensors.push(name);
            }
        }
    }

    // Return error if any required tensors are missing
    if !missing_tensors.is_empty() {
        return Err(GgufLoadingError::IncompleteTensorSet { missing_tensors });
    }

    // Validate tensor compatibility
    validate_tensor_compatibility(&tensor_map, &config)?;

    info!("Successfully loaded GGUF model with {} tensors", tensor_map.len());
    Ok((config, tensor_map))
}

// Comprehensive layer validation
fn validate_transformer_layer(
    tensors: &HashMap<String, CandleTensor>,
    layer: usize,
    device: &Device
) -> Result<HashMap<String, Option<CandleTensor>>, GgufLoadingError> {
    let prefix = format!("blk.{}", layer);
    let mut layer_tensors = HashMap::new();

    // Required tensors for each transformer layer
    let required_tensors = vec![
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        "attn_norm.weight", "ffn_norm.weight"
    ];

    for tensor_suffix in required_tensors {
        let tensor_name = format!("{}.{}", prefix, tensor_suffix);

        if let Some(tensor) = tensors.get(&tensor_name) {
            // Validate tensor properties
            validate_tensor_properties(tensor, &tensor_name)?;
            layer_tensors.insert(tensor_name.clone(), Some(tensor.to_device(device)?));
        } else {
            layer_tensors.insert(tensor_name, None);
        }
    }

    Ok(layer_tensors)
}

// Tensor property validation
fn validate_tensor_properties(tensor: &CandleTensor, name: &str) -> Result<(), GgufLoadingError> {
    let shape = tensor.shape();

    // Basic shape validation
    if shape.dims().is_empty() {
        return Err(GgufLoadingError::InvalidTensorMetadata {
            tensor_name: name.to_string(),
            reason: "Tensor has empty shape".to_string(),
        });
    }

    // Check for reasonable dimensions (not too large or too small)
    for (i, &dim) in shape.dims().iter().enumerate() {
        if dim == 0 {
            return Err(GgufLoadingError::InvalidTensorMetadata {
                tensor_name: name.to_string(),
                reason: format!("Dimension {} is zero", i),
            });
        }
        if dim > 1_000_000 {
            warn!("Large tensor dimension detected: {} has dimension {} = {}", name, i, dim);
        }
    }

    // Validate data type
    match tensor.dtype() {
        DType::F32 | DType::F16 | DType::BF16 | DType::I8 | DType::U8 => {
            // Supported types
        }
        other => {
            return Err(GgufLoadingError::InvalidTensorMetadata {
                tensor_name: name.to_string(),
                reason: format!("Unsupported data type: {:?}", other),
            });
        }
    }

    Ok(())
}

// Overall tensor compatibility validation
fn validate_tensor_compatibility(
    tensors: &HashMap<String, CandleTensor>,
    config: &BitNetConfig
) -> Result<(), GgufLoadingError> {
    // Validate embedding dimensions
    if let Some(embedding) = tensors.get("token_embd.weight") {
        let shape = embedding.shape();
        if shape.dims().len() != 2 {
            return Err(GgufLoadingError::InvalidTensorMetadata {
                tensor_name: "token_embd.weight".to_string(),
                reason: format!("Expected 2D tensor, got {}D", shape.dims().len()),
            });
        }

        let [vocab_size, hidden_size] = shape.dims() else {
            return Err(GgufLoadingError::InvalidTensorMetadata {
                tensor_name: "token_embd.weight".to_string(),
                reason: "Could not extract dimensions".to_string(),
            });
        };

        if *hidden_size != config.model.hidden_size {
            return Err(GgufLoadingError::InvalidTensorMetadata {
                tensor_name: "token_embd.weight".to_string(),
                reason: format!(
                    "Hidden size mismatch: tensor={}, config={}",
                    hidden_size, config.model.hidden_size
                ),
            });
        }

        if *vocab_size != config.model.vocab_size {
            warn!(
                "Vocabulary size mismatch: tensor={}, config={}",
                vocab_size, config.model.vocab_size
            );
        }
    }

    // Validate layer consistency
    for layer in 0..config.model.num_layers {
        validate_layer_consistency(tensors, layer, config)?;
    }

    Ok(())
}
```

### Alternative Solution: Configurable Fallback Mode

For cases where fallback behavior is desired, make it explicit and configurable:

```rust
#[derive(Debug, Clone)]
pub struct GgufLoadingConfig {
    pub strict_validation: bool,
    pub allow_missing_layers: bool,
    pub create_fallback_tensors: bool,
    pub max_missing_tensors: usize,
}

impl Default for GgufLoadingConfig {
    fn default() -> Self {
        Self {
            strict_validation: true,
            allow_missing_layers: false,
            create_fallback_tensors: false,
            max_missing_tensors: 0,
        }
    }
}

fn load_gguf_with_config(
    path: &Path,
    device: Device,
    config: GgufLoadingConfig,
) -> Result<(BitNetConfig, HashMap<String, CandleTensor>), GgufLoadingError> {
    // Attempt standard loading first
    match load_gguf_minimal(path, device) {
        Ok(result) => Ok(result),
        Err(GgufLoadingError::IncompleteTensorSet { missing_tensors })
            if config.allow_missing_layers && missing_tensors.len() <= config.max_missing_tensors => {

            warn!("Loading with missing tensors (fallback mode): {:?}", missing_tensors);

            if config.create_fallback_tensors {
                create_fallback_model(path, device, &missing_tensors)
            } else {
                Err(GgufLoadingError::IncompleteTensorSet { missing_tensors })
            }
        }
        Err(other) => Err(other),
    }
}
```

### Test Utilities Organization

Move test-specific functionality to dedicated test modules:

```rust
#[cfg(test)]
pub mod test_utils {
    use super::*;

    /// Create a mock tensor layout for testing purposes only
    ///
    /// This function creates fake tensors that can be used in unit tests
    /// to verify model loading logic without requiring real GGUF files.
    pub fn create_test_tensor_layout(device: Device) -> Result<(BitNetConfig, HashMap<String, CandleTensor>)> {
        let config = BitNetConfig {
            model: ModelConfig {
                num_layers: 2,
                hidden_size: 256,
                vocab_size: 1000,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut tensor_map = HashMap::new();

        // Create minimal test tensors
        tensor_map.insert(
            "token_embd.weight".to_string(),
            CandleTensor::zeros(&[config.model.vocab_size, config.model.hidden_size], DType::F32, &device)?,
        );

        tensor_map.insert(
            "output.weight".to_string(),
            CandleTensor::zeros(&[config.model.hidden_size, config.model.vocab_size], DType::F32, &device)?,
        );

        // Create minimal transformer layers
        for layer in 0..config.model.num_layers {
            let prefix = format!("blk.{}", layer);

            tensor_map.insert(
                format!("{}.attn_q.weight", prefix),
                CandleTensor::zeros(&[config.model.hidden_size, config.model.hidden_size], DType::F32, &device)?,
            );

            tensor_map.insert(
                format!("{}.attn_output.weight", prefix),
                CandleTensor::zeros(&[config.model.hidden_size, config.model.hidden_size], DType::F32, &device)?,
            );
        }

        Ok((config, tensor_map))
    }

    /// Load a minimal test model for unit testing
    pub fn load_test_model(device: Device) -> Result<BitNetModel> {
        let (config, tensors) = create_test_tensor_layout(device)?;
        BitNetModel::from_tensors(config, tensors)
    }
}
```

## Implementation Plan

### Phase 1: Error Handling Infrastructure (1 day)
- [ ] Define comprehensive `GgufLoadingError` enum with detailed error types
- [ ] Implement error conversion and context propagation
- [ ] Add logging infrastructure for debugging GGUF loading issues
- [ ] Create error reporting utilities for client applications

### Phase 2: Validation Implementation (2 days)
- [ ] Replace mock tensor creation with proper validation
- [ ] Implement tensor property validation (shape, dtype, dimensions)
- [ ] Add layer consistency checking
- [ ] Validate tensor compatibility with model configuration

### Phase 3: Test Utilities Refactoring (1 day)
- [ ] Move `create_mock_tensor_layout` to `#[cfg(test)]` module
- [ ] Rename test utilities to clearly indicate test-only usage
- [ ] Create proper test fixture infrastructure
- [ ] Add documentation for test utilities

### Phase 4: Configuration and Fallback (1 day)
- [ ] Implement configurable loading modes for different use cases
- [ ] Add optional fallback behavior for development/testing
- [ ] Ensure production mode defaults to strict validation
- [ ] Add configuration documentation and examples

## Testing Strategy

### Unit Tests for Error Handling
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incomplete_gguf_file_error() {
        let incomplete_path = create_incomplete_test_gguf();
        let result = load_gguf_minimal(&incomplete_path, Device::Cpu);

        assert!(result.is_err());
        match result.unwrap_err() {
            GgufLoadingError::IncompleteTensorSet { missing_tensors } => {
                assert!(!missing_tensors.is_empty());
                assert!(missing_tensors.iter().any(|t| t.contains("attn_q.weight")));
            }
            other => panic!("Expected IncompleteTensorSet, got {:?}", other),
        }
    }

    #[test]
    fn test_corrupted_gguf_header() {
        let corrupted_path = create_corrupted_test_gguf();
        let result = load_gguf_minimal(&corrupted_path, Device::Cpu);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GgufLoadingError::HeaderParsingFailed { .. }));
    }

    #[test]
    fn test_tensor_validation() {
        let invalid_tensor = CandleTensor::zeros(&[], DType::F32, &Device::Cpu).unwrap();
        let result = validate_tensor_properties(&invalid_tensor, "test_tensor");

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GgufLoadingError::InvalidTensorMetadata { .. }));
    }
}
```

### Integration Tests
```rust
#[test]
fn test_valid_gguf_loading() {
    let valid_model_path = "tests/fixtures/valid_model.gguf";
    let result = load_gguf_minimal(Path::new(valid_model_path), Device::Cpu);

    assert!(result.is_ok());
    let (config, tensors) = result.unwrap();

    assert!(tensors.contains_key("token_embd.weight"));
    assert!(tensors.contains_key("output.weight"));
    assert_eq!(tensors.len(), expected_tensor_count(&config));
}

#[test]
fn test_fallback_mode_configuration() {
    let config = GgufLoadingConfig {
        strict_validation: false,
        allow_missing_layers: true,
        create_fallback_tensors: true,
        max_missing_tensors: 5,
    };

    let incomplete_path = create_incomplete_test_gguf();
    let result = load_gguf_with_config(&incomplete_path, Device::Cpu, config);

    // Should succeed with fallback tensors
    assert!(result.is_ok());
}
```

### Error Reporting Tests
```rust
#[test]
fn test_error_message_quality() {
    let missing_layers_error = GgufLoadingError::IncompleteTensorSet {
        missing_tensors: vec!["blk.0.attn_q.weight".to_string(), "blk.1.ffn_gate.weight".to_string()],
    };

    let error_message = missing_layers_error.to_string();
    assert!(error_message.contains("missing required tensors"));
    assert!(error_message.contains("blk.0.attn_q.weight"));
    assert!(error_message.contains("blk.1.ffn_gate.weight"));
}
```

## Error Handling Examples

### Client Code Error Handling
```rust
// Example of how client code should handle loading errors
match load_gguf_minimal(&model_path, device) {
    Ok((config, tensors)) => {
        info!("Successfully loaded model with {} tensors", tensors.len());
        let model = BitNetModel::from_tensors(config, tensors)?;
        Ok(model)
    }
    Err(GgufLoadingError::IncompleteTensorSet { missing_tensors }) => {
        error!("Model file is incomplete. Missing tensors: {:?}", missing_tensors);
        error!("Please download a complete model file or use a different model.");
        Err(ModelLoadingError::IncompleteModel { missing_tensors })
    }
    Err(GgufLoadingError::HeaderParsingFailed { reason }) => {
        error!("Failed to parse GGUF file header: {}", reason);
        error!("The file may be corrupted or not a valid GGUF file.");
        Err(ModelLoadingError::InvalidFormat { reason })
    }
    Err(other) => {
        error!("Unexpected error loading model: {}", other);
        Err(ModelLoadingError::Unknown { source: other.into() })
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] `load_gguf_minimal` returns proper errors for incomplete GGUF files
- [ ] `create_mock_tensor_layout` is moved to test utilities and clearly marked
- [ ] Comprehensive validation of tensor properties and compatibility
- [ ] Clear, actionable error messages for different failure modes

### Quality Requirements
- [ ] No mock tensors created in production code paths
- [ ] Test utilities separated from production code
- [ ] Error messages provide sufficient context for debugging
- [ ] Backward compatibility maintained for valid GGUF files

### Documentation Requirements
- [ ] Error types documented with examples
- [ ] Test utilities clearly marked and documented
- [ ] Configuration options explained with use cases
- [ ] Troubleshooting guide for common loading errors

## Related Issues/PRs

- Model loading validation improvements (#TBD)
- GGUF format compatibility testing (#TBD)
- Error handling standardization (#TBD)
- Test infrastructure organization (#TBD)

## Labels

`model-loading`, `error-handling`, `gguf`, `validation`, `medium-priority`, `stub-removal`

## Definition of Done

- [ ] Mock tensor creation removed from production code paths
- [ ] Proper error handling implemented for all GGUF loading scenarios
- [ ] Test utilities moved to appropriate test modules
- [ ] Comprehensive validation of loaded tensors and model configuration
- [ ] Clear error messages with actionable information
- [ ] All tests pass with new error handling
- [ ] Documentation updated to reflect new error handling approach