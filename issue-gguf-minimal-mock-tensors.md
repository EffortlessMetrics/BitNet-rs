# [Models] Eliminate mock tensor fallbacks in GGUF minimal loader for production reliability

## Problem Description

The `load_gguf_minimal` and `create_mock_tensor_layout` functions in `crates/bitnet-models/src/gguf_simple.rs` currently create mock tensors as fallbacks when GGUF parsing fails. This approach masks real parsing errors and can lead to silent failures in production environments where models appear to load successfully but contain synthetic data instead of actual trained weights.

## Environment

- **File:** `crates/bitnet-models/src/gguf_simple.rs` (lines 154-177, 670-719)
- **Functions:** `load_gguf_minimal`, `create_mock_tensor_layout`
- **Current Behavior:** Creates mock tensors when GGUF parsing fails
- **Affected Components:** Model loading pipeline, production inference

## Current Implementation Analysis

### load_gguf_minimal Function (lines 154-177)
```rust
fn load_gguf_minimal(path: &Path, device: Device) -> Result<(BitNetConfig, HashMap<String, CandleTensor>)> {
    let two = match crate::gguf_min::load_two(path) {
        Ok(two_tensors) => two_tensors,
        Err(_) => {
            // Problematic: falls back to mock tensors instead of failing
            if let Ok(content) = std::fs::read(path) && content == b"mock_gguf_content" {
                tracing::warn!("Detected mock test file, creating default tensor layout for compatibility");
                return create_mock_tensor_layout(device);
            }
            return Err(BitNetError::Validation("Failed to parse GGUF file...".to_string()));
        }
    };
    // ... continues with real parsing
}
```

### create_mock_tensor_layout Function (lines 670-719)
```rust
fn create_mock_tensor_layout(device: Device) -> Result<(BitNetConfig, HashMap<String, CandleTensor>)> {
    // Creates synthetic tensors with mathematical patterns
    let tok_emb_data: Vec<f32> = (0..(vocab_size * hidden_size))
        .map(|i| {
            let pattern = (i as f32 * 0.001).sin() * 0.5;
            if pattern.abs() < 1e-6 { 0.001 } else { pattern }
        })
        .collect();
    // ... continues creating mock data for all tensor types
}
```

## Root Cause Analysis

1. **Development Convenience**: Mock tensors were introduced to enable testing without complete GGUF files
2. **Legacy Compatibility**: Fallback mechanism intended to support incomplete test files
3. **Error Masking**: Mock fallback hides actual parsing failures that should be debugged
4. **Production Risk**: No clear separation between test and production code paths

**Issues with Current Approach:**
1. **Silent Failures**: Production code may unknowingly use mock data
2. **Debugging Complexity**: Real GGUF parsing errors are masked by mock fallbacks
3. **Test Contamination**: Test-specific code mixed with production logic
4. **Performance Misleading**: Mock tensors may perform differently than real model weights
5. **Security Risk**: Mock tensors could be triggered by malformed files in production

## Impact Assessment

**Severity:** High
**Component:** Model Loading Pipeline
**Affected Areas:**
- Production model reliability
- Error reporting and debugging
- Test isolation and clarity
- Security and data integrity

**Production Risks:**
- Models may appear to load successfully but contain synthetic weights
- Inference results will be meaningless with mock tensors
- Debugging failures becomes significantly more complex
- Security implications of processing malformed files

## Proposed Solution

### 1. Strict Production Error Handling

Replace mock fallbacks with proper error handling:

```rust
fn load_gguf_minimal(
    path: &Path,
    device: Device,
) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
    // Try the existing minimal GGUF parser with strict error handling
    let two = crate::gguf_min::load_two(path).map_err(|e| {
        BitNetError::Validation(format!(
            "GGUF minimal parser failed for '{}': {}. This file may be corrupted, incomplete, or incompatible.",
            path.display(),
            e
        ))
    })?;

    // Start from default config and update basic dimensions from the file
    let mut config = bitnet_common::BitNetConfig::default();
    config.model.vocab_size = two.vocab as usize;
    config.model.hidden_size = two.dim as usize;

    // Validate basic configuration parameters
    validate_config_parameters(&config)?;

    // Continue with real tensor parsing...
    let tensor_map = parse_real_tensors(&two, &config, device)?;

    // Validate that all required tensors are present
    validate_required_tensors(&tensor_map, &config)?;

    Ok((config, tensor_map))
}

fn validate_config_parameters(config: &bitnet_common::BitNetConfig) -> Result<()> {
    if config.model.vocab_size == 0 {
        return Err(BitNetError::Validation(
            "Invalid GGUF: vocabulary size cannot be zero".to_string()
        ));
    }

    if config.model.hidden_size == 0 {
        return Err(BitNetError::Validation(
            "Invalid GGUF: hidden size cannot be zero".to_string()
        ));
    }

    if config.model.num_layers == 0 {
        return Err(BitNetError::Validation(
            "Invalid GGUF: number of layers cannot be zero".to_string()
        ));
    }

    Ok(())
}

fn validate_required_tensors(
    tensor_map: &HashMap<String, CandleTensor>,
    config: &bitnet_common::BitNetConfig,
) -> Result<()> {
    // Check for essential tensors
    let required_tensors = [
        "token_embd.weight",
        "output.weight",
    ];

    for tensor_name in &required_tensors {
        if !tensor_map.contains_key(*tensor_name) {
            return Err(BitNetError::Validation(format!(
                "Missing required tensor '{}' in GGUF file",
                tensor_name
            )));
        }
    }

    // Check for layer-specific tensors
    for layer in 0..config.model.num_layers {
        let layer_tensors = [
            format!("blk.{}.attn_q.weight", layer),
            format!("blk.{}.attn_k.weight", layer),
            format!("blk.{}.attn_v.weight", layer),
            format!("blk.{}.attn_output.weight", layer),
            format!("blk.{}.ffn_gate.weight", layer),
            format!("blk.{}.ffn_up.weight", layer),
            format!("blk.{}.ffn_down.weight", layer),
        ];

        for tensor_name in &layer_tensors {
            if !tensor_map.contains_key(tensor_name) {
                return Err(BitNetError::Validation(format!(
                    "Missing layer {} tensor '{}' in GGUF file. File may be incomplete.",
                    layer, tensor_name
                )));
            }
        }
    }

    Ok(())
}
```

### 2. Dedicated Test Infrastructure

Move mock functionality to test-only modules:

```rust
#[cfg(test)]
pub mod test_utils {
    use super::*;

    /// Create mock tensor layout specifically for testing
    ///
    /// This function should ONLY be used in test environments and provides
    /// synthetic tensors for testing model loading pipelines without real GGUF files.
    pub fn create_mock_tensor_layout_for_testing(
        device: Device,
    ) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
        // Clear indication this is for testing only
        tracing::warn!("Creating mock tensors for TESTING ONLY - not for production use");

        let config = bitnet_common::BitNetConfig::default();
        let num_layers = config.model.num_layers;
        let intermediate_size = config.model.intermediate_size;
        let hidden_size = config.model.hidden_size;
        let vocab_size = config.model.vocab_size;

        let cdevice = convert_device(device)?;
        let mut tensor_map = HashMap::new();

        // Create identifiable mock tensors with clear patterns
        create_mock_embeddings(&mut tensor_map, vocab_size, hidden_size, &cdevice)?;
        create_mock_transformer_layers(&mut tensor_map, num_layers, hidden_size, intermediate_size, &cdevice)?;
        create_mock_output_projection(&mut tensor_map, hidden_size, vocab_size, &cdevice)?;

        Ok((config, tensor_map))
    }

    fn create_mock_embeddings(
        tensor_map: &mut HashMap<String, CandleTensor>,
        vocab_size: usize,
        hidden_size: usize,
        device: &CDevice,
    ) -> Result<()> {
        // Create predictable patterns for testing
        let tok_emb_data: Vec<f32> = (0..(vocab_size * hidden_size))
            .map(|i| {
                // Use a clear test pattern that's easy to verify
                let row = i / hidden_size;
                let col = i % hidden_size;
                ((row as f32 * 0.01) + (col as f32 * 0.001)).sin() * 0.5
            })
            .collect();

        tensor_map.insert(
            "token_embd.weight".to_string(),
            CandleTensor::from_vec(tok_emb_data, (vocab_size, hidden_size), device)?,
        );

        Ok(())
    }

    fn create_mock_transformer_layers(
        tensor_map: &mut HashMap<String, CandleTensor>,
        num_layers: usize,
        hidden_size: usize,
        intermediate_size: usize,
        device: &CDevice,
    ) -> Result<()> {
        for layer in 0..num_layers {
            let layer_prefix = format!("blk.{}", layer);

            // Attention weights with layer-specific patterns
            let layer_factor = (layer as f32 + 1.0) * 0.1;

            create_layer_tensor(
                tensor_map,
                &format!("{}.attn_q.weight", layer_prefix),
                hidden_size,
                hidden_size,
                layer_factor,
                device,
            )?;

            create_layer_tensor(
                tensor_map,
                &format!("{}.attn_k.weight", layer_prefix),
                hidden_size,
                hidden_size,
                layer_factor,
                device,
            )?;

            create_layer_tensor(
                tensor_map,
                &format!("{}.attn_v.weight", layer_prefix),
                hidden_size,
                hidden_size,
                layer_factor,
                device,
            )?;

            create_layer_tensor(
                tensor_map,
                &format!("{}.attn_output.weight", layer_prefix),
                hidden_size,
                hidden_size,
                layer_factor,
                device,
            )?;

            // FFN weights
            create_layer_tensor(
                tensor_map,
                &format!("{}.ffn_gate.weight", layer_prefix),
                intermediate_size,
                hidden_size,
                layer_factor,
                device,
            )?;

            create_layer_tensor(
                tensor_map,
                &format!("{}.ffn_up.weight", layer_prefix),
                intermediate_size,
                hidden_size,
                layer_factor,
                device,
            )?;

            create_layer_tensor(
                tensor_map,
                &format!("{}.ffn_down.weight", layer_prefix),
                hidden_size,
                intermediate_size,
                layer_factor,
                device,
            )?;

            // Normalization weights (all ones for simplicity)
            create_norm_tensor(
                tensor_map,
                &format!("{}.attn_norm.weight", layer_prefix),
                hidden_size,
                device,
            )?;

            create_norm_tensor(
                tensor_map,
                &format!("{}.ffn_norm.weight", layer_prefix),
                hidden_size,
                device,
            )?;
        }

        Ok(())
    }

    fn create_layer_tensor(
        tensor_map: &mut HashMap<String, CandleTensor>,
        name: &str,
        rows: usize,
        cols: usize,
        layer_factor: f32,
        device: &CDevice,
    ) -> Result<()> {
        let data: Vec<f32> = (0..(rows * cols))
            .map(|i| {
                let pattern = (i as f32 * 0.001 * layer_factor).sin() * 0.3;
                if pattern.abs() < 1e-6 { 0.001 * layer_factor } else { pattern }
            })
            .collect();

        tensor_map.insert(
            name.to_string(),
            CandleTensor::from_vec(data, (rows, cols), device)?,
        );

        Ok(())
    }

    fn create_norm_tensor(
        tensor_map: &mut HashMap<String, CandleTensor>,
        name: &str,
        size: usize,
        device: &CDevice,
    ) -> Result<()> {
        let data = vec![1.0f32; size];  // All ones for normalization
        tensor_map.insert(
            name.to_string(),
            CandleTensor::from_vec(data, (size,), device)?,
        );

        Ok(())
    }

    /// Create a mock GGUF file for testing
    pub fn create_mock_gguf_file(path: &Path) -> std::io::Result<()> {
        std::fs::write(path, b"mock_gguf_content")?;
        Ok(())
    }
}

// Helper function for device conversion
fn convert_device(device: Device) -> Result<CDevice> {
    match device {
        Device::Cpu => Ok(CDevice::Cpu),
        Device::Cuda(id) => {
            CDevice::new_cuda(id).map_err(|e| {
                BitNetError::Validation(format!("Failed to create CUDA device {}: {}", id, e))
            })
        }
        Device::Metal => {
            Err(BitNetError::Validation(
                "Metal device not supported in current implementation".to_string()
            ))
        }
    }
}
```

### 3. Enhanced Error Reporting

Improve error messages with actionable guidance:

```rust
#[derive(Debug, thiserror::Error)]
pub enum GgufLoadError {
    #[error("GGUF file not found: {path}")]
    FileNotFound { path: String },

    #[error("GGUF parsing failed: {reason}. File may be corrupted or incompatible.")]
    ParseError { reason: String },

    #[error("GGUF validation failed: {reason}")]
    ValidationError { reason: String },

    #[error("Missing required tensors: {missing_tensors:?}. File appears incomplete.")]
    MissingTensors { missing_tensors: Vec<String> },

    #[error("Device error: {message}")]
    DeviceError { message: String },

    #[error("Incompatible GGUF version: expected {expected}, found {found}")]
    VersionMismatch { expected: String, found: String },
}

impl From<GgufLoadError> for BitNetError {
    fn from(err: GgufLoadError) -> Self {
        BitNetError::Validation(err.to_string())
    }
}
```

### 4. Configuration-Based Fallback Control

Add explicit configuration for fallback behavior:

```rust
#[derive(Debug, Clone)]
pub struct GgufLoadConfig {
    /// Allow mock tensor fallback (default: false for production)
    pub allow_mock_fallback: bool,
    /// Strict validation of tensor completeness
    pub strict_validation: bool,
    /// Maximum file size to process (safety limit)
    pub max_file_size_gb: usize,
}

impl Default for GgufLoadConfig {
    fn default() -> Self {
        Self {
            allow_mock_fallback: false,  // NEVER allow in production by default
            strict_validation: true,
            max_file_size_gb: 50,  // 50GB limit
        }
    }
}

impl GgufLoadConfig {
    pub fn for_testing() -> Self {
        Self {
            allow_mock_fallback: true,
            strict_validation: false,
            max_file_size_gb: 1,
        }
    }
}
```

## Implementation Plan

### Phase 1: Error Handling Enhancement
- [ ] Remove mock tensor fallback from `load_gguf_minimal`
- [ ] Add comprehensive error types for GGUF loading failures
- [ ] Implement strict validation for configuration parameters
- [ ] Add required tensor validation with detailed error messages

### Phase 2: Test Infrastructure Separation
- [ ] Move `create_mock_tensor_layout` to `#[cfg(test)]` module
- [ ] Rename to `create_mock_tensor_layout_for_testing`
- [ ] Add clear documentation about test-only usage
- [ ] Create test utilities for mock GGUF file creation

### Phase 3: Configuration and Control
- [ ] Add `GgufLoadConfig` for explicit fallback control
- [ ] Implement production-safe defaults
- [ ] Add file size and safety validation
- [ ] Create environment variable overrides for testing

### Phase 4: Enhanced Error Reporting
- [ ] Implement detailed error types with actionable messages
- [ ] Add logging for debugging GGUF parsing issues
- [ ] Create error recovery suggestions
- [ ] Add validation reports for incomplete files

### Phase 5: Testing and Validation
- [ ] Comprehensive test coverage for error conditions
- [ ] Integration tests with real GGUF files
- [ ] Performance testing for validation overhead
- [ ] Security testing with malformed files

## Testing Strategy

### Unit Tests
```bash
# Test strict error handling
cargo test --package bitnet-models gguf_error_handling

# Test validation functions
cargo test --package bitnet-models tensor_validation

# Test mock utilities (test-only)
cargo test --package bitnet-models test_utils::mock_tensors
```

### Integration Tests
```bash
# Test with various GGUF file conditions
cargo test --package bitnet-models --test gguf_loading_scenarios

# Test production vs test configuration
cargo test --package bitnet-models --test config_scenarios
```

### Error Condition Tests
```bash
# Test with corrupted files
cargo test --package bitnet-models --test malformed_gguf

# Test with incomplete files
cargo test --package bitnet-models --test incomplete_gguf
```

## Migration Strategy

### Phase 1: Backward Compatibility
1. Keep existing mock fallback with deprecation warnings
2. Add configuration flag to control behavior
3. Update documentation with migration timeline

### Phase 2: Progressive Enforcement
1. Default to strict mode in new installations
2. Provide clear migration path for existing users
3. Add validation tools for GGUF file health

### Phase 3: Complete Transition
1. Remove mock fallback entirely
2. Ensure all tests use dedicated test utilities
3. Final validation of production safety

## Success Criteria

1. **Production Safety**: No mock tensors can be created in production environments
2. **Clear Error Messages**: All GGUF loading failures provide actionable feedback
3. **Test Isolation**: Mock functionality clearly separated and marked as test-only
4. **Performance**: Validation overhead < 5% of loading time
5. **Reliability**: 100% detection of corrupted or incomplete GGUF files

## Related Issues

- GGUF parsing performance optimization
- Model loading pipeline robustness
- Test infrastructure improvements
- Production deployment safety

---

**Labels:** `models`, `production-safety`, `high-priority`, `bug`
**Assignee:** Model Loading Team
**Epic:** Production Reliability