# [Model Loading] Replace mock tensor creation in GGUF minimal loader with proper error handling

## Problem Description

The `load_gguf_minimal` function in `crates/bitnet-models/src/gguf_simple.rs` contains problematic mock tensor creation logic that compromises production reliability. When GGUF parsing fails, instead of returning appropriate errors, the function detects "mock test files" by checking for hardcoded content (`b"mock_gguf_content"`) and creates mock tensor layouts, which can mask real parsing issues and lead to unexpected behavior in production.

## Environment

- **File**: `crates/bitnet-models/src/gguf_simple.rs`
- **Functions**: `load_gguf_minimal`, `create_mock_tensor_layout`
- **Context**: GGUF model loading with enhanced/minimal parser fallback chain
- **Architecture**: Device-aware model loading with tensor placement and quantization support

## Root Cause Analysis

### Current Implementation Issues

```rust
fn load_gguf_minimal(
    path: &Path,
    device: Device,
) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
    let two = match crate::gguf_min::load_two(path) {
        Ok(two_tensors) => two_tensors,
        Err(_) => {
            // PROBLEMATIC: Hardcoded mock detection
            if let Ok(content) = std::fs::read(path)
                && content == b"mock_gguf_content"
            {
                tracing::warn!(
                    "Detected mock test file, creating default tensor layout for compatibility"
                );
                // Creates mock tensors instead of failing appropriately
                return create_mock_tensor_layout(device);
            }
            // Only fails here if not a "mock" file
            return Err(BitNetError::Validation(
                "Failed to parse GGUF file with both enhanced and minimal parsers".to_string(),
            ));
        }
    };
    // ... rest of implementation
}

fn create_mock_tensor_layout(
    device: Device,
) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
    // Creates entirely artificial tensor layout
    // Not clearly marked as test-only functionality
}
```

### Problems Identified

1. **Production/Test Boundary Violation**: Production code contains test-specific logic
2. **Hardcoded Magic Values**: `b"mock_gguf_content"` creates implicit coupling with test infrastructure
3. **Silent Failure Masking**: Real GGUF parsing errors may be masked by mock creation
4. **Misleading Function Purpose**: `load_gguf_minimal` suggests minimal parsing, not mock creation
5. **Debugging Complications**: Mock tensor creation can hide real model loading issues
6. **Test Infrastructure Leakage**: Test utilities are embedded in production model loading code

### Architecture Impact

The current implementation creates a problematic fallback chain:
1. Enhanced GGUF parser attempts loading
2. On failure, falls back to minimal parser
3. On minimal parser failure, checks for hardcoded mock content
4. Creates mock tensors instead of proper error reporting

This design violates separation of concerns and makes debugging difficult.

## Impact Assessment

- **Severity**: High - Production reliability and debuggability issues
- **Production Risk**: High - May mask real model loading failures
- **Debugging Difficulty**: High - Mock creation can hide real issues
- **Test Infrastructure**: Medium - Test-specific code leaks into production
- **Code Maintainability**: Medium - Hardcoded values create implicit dependencies

## Proposed Solution

### Primary Approach: Clean Separation and Proper Error Handling

#### Phase 1: Extract Test Infrastructure

```rust
// crates/bitnet-models/src/testing/mod.rs (new module)

#[cfg(test)]
pub mod mock_loaders {
    use super::*;

    /// Create mock tensor layout for testing purposes
    /// This function should ONLY be used in test contexts
    pub fn create_mock_tensor_layout_for_testing(
        device: Device,
    ) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
        let config = bitnet_common::BitNetConfig::default();
        let num_layers = config.model.num_layers;
        let intermediate_size = config.model.intermediate_size;
        let hidden_size = config.model.hidden_size;
        let vocab_size = config.model.vocab_size;

        let cdevice = match device {
            Device::Cpu => CDevice::Cpu,
            Device::Cuda(id) => match CDevice::new_cuda(id) {
                Ok(cuda_device) => {
                    tracing::info!("Using CUDA device {} for mock tensor placement", id);
                    cuda_device
                },
                Err(_) => {
                    tracing::warn!("CUDA device {} unavailable for mock, falling back to CPU", id);
                    CDevice::Cpu
                }
            },
            Device::Metal => CDevice::Metal(0),
        };

        let mut tensor_map = HashMap::new();

        // Create well-documented mock tensors
        for layer in 0..num_layers {
            let prefix = format!("blk.{}", layer);

            // Attention weights
            tensor_map.insert(
                format!("{}.attn_q.weight", prefix),
                create_mock_weight_tensor([hidden_size, hidden_size], &cdevice, "attention_q")?,
            );
            // ... other tensors with clear naming and documentation
        }

        tracing::warn!("Created mock tensor layout for testing - DO NOT USE IN PRODUCTION");
        Ok((config, tensor_map))
    }

    fn create_mock_weight_tensor(
        shape: [usize; 2],
        device: &CDevice,
        layer_type: &str
    ) -> Result<CandleTensor> {
        // Create deterministic test data based on layer type
        let data: Vec<f32> = (0..shape[0] * shape[1])
            .map(|i| {
                // Different patterns for different layer types for testing
                match layer_type {
                    "attention_q" => (i as f32 * 0.01) % 1.0,
                    "attention_k" => (i as f32 * 0.02) % 1.0,
                    "attention_v" => (i as f32 * 0.03) % 1.0,
                    "ffn" => (i as f32 * 0.05) % 1.0,
                    _ => (i as f32 * 0.01) % 1.0,
                }
            })
            .collect();

        CTensor::from_slice(&data, shape, device)
            .map_err(|e| BitNetError::TensorOperation(e.to_string()))
            .map(CandleTensor::from)
    }
}

/// Mock GGUF file creator for testing
#[cfg(test)]
pub fn create_mock_gguf_file(path: &Path) -> Result<()> {
    // Create a minimal but valid GGUF file structure for testing
    // This should create actual GGUF format, not hardcoded content
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;

    // Write minimal GGUF header and metadata
    file.write_all(b"GGUF")?; // Magic number
    file.write_all(&[3u8, 0, 0, 0])?; // Version
    file.write_all(&[0u8, 0, 0, 0, 0, 0, 0, 0])?; // Tensor count
    file.write_all(&[1u8, 0, 0, 0, 0, 0, 0, 0])?; // Metadata KV count

    // Add minimal metadata
    // ... (implement minimal valid GGUF structure)

    file.flush()?;
    Ok(())
}
```

#### Phase 2: Clean Production Implementation

```rust
// Cleaned up load_gguf_minimal function

/// Minimal GGUF loading for legacy compatibility and fallback scenarios
///
/// This function provides a simplified GGUF loading implementation for:
/// - Legacy compatibility during development
/// - Fallback when enhanced parsing encounters non-critical issues
/// - Simplified model loading for testing infrastructure
///
/// # Error Handling
/// This function will return appropriate errors for:
/// - File not found or inaccessible
/// - Invalid GGUF format
/// - Incompatible GGUF version
/// - Missing required tensors
/// - Device compatibility issues
fn load_gguf_minimal(
    path: &Path,
    device: Device,
) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
    // Validate file existence and accessibility
    if !path.exists() {
        return Err(BitNetError::Validation(format!(
            "GGUF file not found: {}", path.display()
        )));
    }

    if !path.is_file() {
        return Err(BitNetError::Validation(format!(
            "Path is not a file: {}", path.display()
        )));
    }

    // Attempt minimal GGUF parsing
    let two = match crate::gguf_min::load_two(path) {
        Ok(two_tensors) => {
            debug!("Successfully loaded GGUF with minimal parser: {}", path.display());
            two_tensors
        },
        Err(e) => {
            // Provide detailed error information
            return Err(BitNetError::Validation(format!(
                "Failed to parse GGUF file '{}' with minimal parser: {}. \
                 Ensure the file is a valid GGUF format and contains required tensor data.",
                path.display(), e
            )));
        }
    };

    // Validate basic GGUF content
    if two.vocab == 0 {
        return Err(BitNetError::Validation(
            "Invalid GGUF file: vocabulary size is zero".to_string()
        ));
    }

    if two.dim == 0 {
        return Err(BitNetError::Validation(
            "Invalid GGUF file: hidden dimension is zero".to_string()
        ));
    }

    // Create configuration from parsed GGUF data
    let mut config = bitnet_common::BitNetConfig::default();
    config.model.vocab_size = two.vocab as usize;
    config.model.hidden_size = two.dim as usize;

    // Validate configuration makes sense
    if config.model.vocab_size > 1_000_000 {
        return Err(BitNetError::Validation(format!(
            "Vocabulary size too large: {} (max: 1,000,000)", config.model.vocab_size
        )));
    }

    if config.model.hidden_size > 32768 {
        return Err(BitNetError::Validation(format!(
            "Hidden size too large: {} (max: 32,768)", config.model.hidden_size
        )));
    }

    let num_layers = config.model.num_layers;
    let intermediate_size = config.model.intermediate_size;

    // Convert device for tensor operations
    let cdevice = convert_device_with_fallback(device)?;

    let mut tensor_map = HashMap::new();

    // Load required tensors with validation
    load_embedding_tensors(&two, &mut tensor_map, &cdevice, &config)?;
    load_transformer_layers(&two, &mut tensor_map, &cdevice, num_layers, &config)?;
    load_output_tensors(&two, &mut tensor_map, &cdevice, &config)?;

    // Validate that all required tensors are present
    validate_required_tensors(&tensor_map, &config)?;

    info!(
        "Successfully loaded GGUF model with minimal parser: {} layers, vocab_size={}, hidden_size={}",
        num_layers, config.model.vocab_size, config.model.hidden_size
    );

    Ok((config, tensor_map))
}

/// Load and validate embedding tensors
fn load_embedding_tensors(
    gguf_data: &TwoTensors,
    tensor_map: &mut HashMap<String, CandleTensor>,
    device: &CDevice,
    config: &BitNetConfig,
) -> Result<()> {
    // Load token embeddings
    if let Some(embedding_data) = gguf_data.token_embeddings.as_ref() {
        let embedding_tensor = CTensor::from_slice(
            embedding_data,
            [config.model.vocab_size, config.model.hidden_size],
            device,
        ).map_err(|e| BitNetError::TensorOperation(format!("Failed to create embedding tensor: {}", e)))?;

        tensor_map.insert("token_embd.weight".to_string(), CandleTensor::from(embedding_tensor));
    } else {
        return Err(BitNetError::Validation(
            "Missing required token embeddings in GGUF file".to_string()
        ));
    }

    Ok(())
}

/// Validate that all required tensors are present
fn validate_required_tensors(
    tensor_map: &HashMap<String, CandleTensor>,
    config: &BitNetConfig,
) -> Result<()> {
    // Check for required embedding tensors
    if !tensor_map.contains_key("token_embd.weight") {
        return Err(BitNetError::Validation(
            "Missing required tensor: token_embd.weight".to_string()
        ));
    }

    // Check for required layer tensors
    for layer in 0..config.model.num_layers {
        let required_tensors = [
            format!("blk.{}.attn_q.weight", layer),
            format!("blk.{}.attn_k.weight", layer),
            format!("blk.{}.attn_v.weight", layer),
            format!("blk.{}.attn_output.weight", layer),
            format!("blk.{}.ffn_gate.weight", layer),
            format!("blk.{}.ffn_up.weight", layer),
            format!("blk.{}.ffn_down.weight", layer),
        ];

        for tensor_name in &required_tensors {
            if !tensor_map.contains_key(tensor_name) {
                return Err(BitNetError::Validation(format!(
                    "Missing required tensor: {}", tensor_name
                )));
            }
        }
    }

    // Check for output tensors
    if !tensor_map.contains_key("output.weight") && !tensor_map.contains_key("lm_head.weight") {
        return Err(BitNetError::Validation(
            "Missing required output tensor (output.weight or lm_head.weight)".to_string()
        ));
    }

    debug!("All required tensors validated successfully");
    Ok(())
}
```

#### Phase 3: Test Infrastructure Integration

```rust
// Updated test infrastructure

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::mock_loaders::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mock_loader_creates_valid_tensors() {
        let (config, tensor_map) = create_mock_tensor_layout_for_testing(Device::Cpu).unwrap();

        assert_eq!(config.model.vocab_size, 50257);
        assert!(tensor_map.contains_key("token_embd.weight"));
        assert!(tensor_map.contains_key("blk.0.attn_q.weight"));
    }

    #[test]
    fn test_load_gguf_minimal_with_valid_file() {
        let temp_file = NamedTempFile::new().unwrap();
        create_mock_gguf_file(temp_file.path()).unwrap();

        let result = load_gguf_minimal(temp_file.path(), Device::Cpu);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_gguf_minimal_with_nonexistent_file() {
        let result = load_gguf_minimal(Path::new("/nonexistent/file.gguf"), Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("GGUF file not found"));
    }

    #[test]
    fn test_load_gguf_minimal_with_invalid_file() {
        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), b"invalid gguf content").unwrap();

        let result = load_gguf_minimal(temp_file.path(), Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to parse GGUF file"));
    }
}
```

## Implementation Plan

### Phase 1: Test Infrastructure Extraction (Week 1)
- [ ] Create dedicated `testing` module for mock functionality
- [ ] Extract `create_mock_tensor_layout` to test-specific module
- [ ] Implement proper mock GGUF file creation for tests
- [ ] Update all tests to use extracted test utilities

### Phase 2: Production Code Cleanup (Week 2)
- [ ] Remove hardcoded mock detection from `load_gguf_minimal`
- [ ] Implement comprehensive error handling and validation
- [ ] Add proper GGUF content validation
- [ ] Enhance error messages with actionable information

### Phase 3: Enhanced Validation (Week 3)
- [ ] Add tensor presence validation
- [ ] Implement configuration sanity checking
- [ ] Add device compatibility validation
- [ ] Create comprehensive test coverage

### Phase 4: Integration and Documentation (Week 4)
- [ ] Update all affected tests
- [ ] Add documentation for test infrastructure
- [ ] Create examples of proper GGUF loading
- [ ] Validate across different model types and devices

## Testing Strategy

### Unit Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_loader_error_handling() {
        // Test file not found
        let result = load_gguf_minimal(Path::new("/nonexistent.gguf"), Device::Cpu);
        assert!(result.is_err());

        // Test invalid file format
        let temp = NamedTempFile::new().unwrap();
        std::fs::write(temp.path(), b"not a gguf file").unwrap();
        let result = load_gguf_minimal(temp.path(), Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_validation() {
        let temp = NamedTempFile::new().unwrap();
        create_incomplete_gguf_file(temp.path()).unwrap(); // Missing required tensors
        let result = load_gguf_minimal(temp.path(), Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Missing required tensor"));
    }

    #[test]
    fn test_configuration_validation() {
        let temp = NamedTempFile::new().unwrap();
        create_invalid_config_gguf_file(temp.path()).unwrap(); // Invalid dimensions
        let result = load_gguf_minimal(temp.path(), Device::Cpu);
        assert!(result.is_err());
    }
}
```

### Integration Testing

```bash
# Test with different model types
cargo test --package bitnet-models gguf_simple::load_gguf_minimal --no-default-features --features cpu
cargo test --package bitnet-models gguf_simple::load_gguf_minimal --no-default-features --features gpu

# Test mock infrastructure
cargo test --package bitnet-models testing::mock_loaders --no-default-features --features cpu
```

## Related Issues/PRs

- Related to model loading reliability improvements
- Connects to test infrastructure standardization
- May inform GGUF compatibility and validation framework
- Links to production deployment reliability

## Acceptance Criteria

- [ ] Production model loading code contains no test-specific logic
- [ ] Mock functionality is clearly isolated in test modules
- [ ] Comprehensive error handling for all failure scenarios
- [ ] Clear, actionable error messages for debugging
- [ ] All required tensors are validated before successful return
- [ ] Configuration validation prevents obviously invalid models
- [ ] Test infrastructure provides realistic mock models
- [ ] All existing tests continue to pass with updated infrastructure
- [ ] Documentation clearly explains when to use different loading methods

## Priority: High

This addresses production reliability issues where test infrastructure leaks into production code, potentially masking real model loading failures. The hardcoded mock detection represents a significant debugging and reliability risk that should be resolved promptly.