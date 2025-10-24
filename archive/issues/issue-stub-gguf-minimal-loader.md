# [IMPLEMENTATION] Replace mock tensor fallback in GGUF minimal loader with robust error handling

## Problem Description

The `load_gguf_minimal` function in `crates/bitnet-models/src/gguf_simple.rs` contains fallback logic that creates mock tensors when GGUF parsing fails, instead of providing proper error handling and validation. This approach masks real parsing issues and can lead to silent failures in production environments.

## Environment

- **File**: `crates/bitnet-models/src/gguf_simple.rs`
- **Functions**: `load_gguf_minimal`, `create_mock_tensor_layout`
- **Context**: GGUF model loading infrastructure
- **Rust Version**: 1.90.0+
- **Feature Flags**: `cpu`, `gpu`

## Root Cause Analysis

The current implementation in `load_gguf_minimal` includes problematic fallback logic:

```rust
fn load_gguf_minimal(
    path: &Path,
    device: Device,
) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
    let two = match crate::gguf_min::load_two(path) {
        Ok(two_tensors) => two_tensors,
        Err(_) => {
            // If minimal GGUF parsing also fails, check if this is a mock file from tests
            if let Ok(content) = std::fs::read(path)
                && content == b"mock_gguf_content"
            {
                tracing::warn!(
                    "Detected mock test file, creating default tensor layout for compatibility"
                );
                // Create mock TwoTensors for test compatibility
                return create_mock_tensor_layout(device);
            }
            // Real parsing failure - re-throw original error
            return Err(BitNetError::Validation(
                "Failed to parse GGUF file with both enhanced and minimal parsers".to_string(),
            ));
        }
    };
    // ...
}
```

### Technical Issues Identified

1. **Test Code in Production**: Mock tensor creation logic exists in production code path
2. **Silent Failure Masking**: Real GGUF parsing errors are hidden by mock fallbacks
3. **Poor Error Reporting**: Generic error messages provide insufficient debugging information
4. **Test Dependency**: Production code depends on test-specific file content detection
5. **Inconsistent Behavior**: Different behavior for similar parsing failures

### Impact Assessment

**Severity**: Medium-High
**Category**: Production Reliability / Code Quality

**Current Impact**:
- Production deployments may silently fail with mock data instead of real models
- Debugging GGUF parsing issues is difficult due to masked errors
- Test code pollution in production paths reduces maintainability
- Risk of deploying non-functional models in production

**Future Risks**:
- Silent failures in production environments
- Difficulty diagnosing model loading issues
- Inconsistent behavior across different GGUF file types
- Maintenance burden from mixed test/production logic

## Proposed Solution

### Primary Approach: Robust Error Handling with Clear Separation

Replace the mock tensor fallback with comprehensive error handling and proper test infrastructure separation.

**Implementation Plan:**

```rust
/// Enhanced minimal GGUF loader with robust error handling
fn load_gguf_minimal(
    path: &Path,
    device: Device,
) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
    // Validate file exists and is readable
    validate_gguf_file(path)?;

    // Parse GGUF with detailed error context
    let two = crate::gguf_min::load_two(path)
        .map_err(|e| create_detailed_parsing_error(path, e))?;

    // Validate parsed data integrity
    validate_parsed_gguf_data(&two)?;

    // Create configuration from parsed data
    let config = create_config_from_gguf(&two)?;

    // Load tensors with proper validation
    let tensor_map = load_tensors_from_gguf(&two, device, &config)?;

    // Final validation of loaded model
    validate_loaded_model(&config, &tensor_map)?;

    Ok((config, tensor_map))
}

/// Comprehensive GGUF file validation
fn validate_gguf_file(path: &Path) -> Result<()> {
    // Check file exists
    if !path.exists() {
        return Err(BitNetError::Validation(
            format!("GGUF file does not exist: {}", path.display())
        ));
    }

    // Check file permissions
    let metadata = std::fs::metadata(path)
        .map_err(|e| BitNetError::Validation(
            format!("Cannot read GGUF file metadata '{}': {}", path.display(), e)
        ))?;

    if !metadata.is_file() {
        return Err(BitNetError::Validation(
            format!("Path is not a file: {}", path.display())
        ));
    }

    // Check minimum file size (GGUF header minimum)
    const MIN_GGUF_SIZE: u64 = 64; // Minimum viable GGUF file size
    if metadata.len() < MIN_GGUF_SIZE {
        return Err(BitNetError::Validation(
            format!(
                "GGUF file too small ({} bytes, minimum {} bytes): {}",
                metadata.len(), MIN_GGUF_SIZE, path.display()
            )
        ));
    }

    // Validate file extension
    if let Some(extension) = path.extension() {
        if extension != "gguf" {
            tracing::warn!("File does not have .gguf extension: {}", path.display());
        }
    }

    Ok(())
}

/// Create detailed parsing error with context
fn create_detailed_parsing_error(path: &Path, original_error: impl std::fmt::Display) -> BitNetError {
    let file_info = gather_file_diagnostics(path);

    BitNetError::Validation(format!(
        "Failed to parse GGUF file '{}': {}\nFile diagnostics: {}",
        path.display(),
        original_error,
        file_info
    ))
}

/// Gather diagnostic information about the file
fn gather_file_diagnostics(path: &Path) -> String {
    let mut diagnostics = Vec::new();

    // File size
    if let Ok(metadata) = std::fs::metadata(path) {
        diagnostics.push(format!("size={}bytes", metadata.len()));
    }

    // Magic number check
    if let Ok(mut file) = std::fs::File::open(path) {
        use std::io::Read;
        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_ok() {
            diagnostics.push(format!("magic={:?}", String::from_utf8_lossy(&magic)));
        }
    }

    // First few bytes for debugging
    if let Ok(content) = std::fs::read(path) {
        let preview = content.iter()
            .take(16)
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(" ");
        diagnostics.push(format!("header_preview={}", preview));
    }

    diagnostics.join(", ")
}

/// Validate parsed GGUF data integrity
fn validate_parsed_gguf_data(two: &crate::gguf_min::TwoTensors) -> Result<()> {
    // Validate vocabulary size
    if two.vocab == 0 {
        return Err(BitNetError::Validation(
            "Invalid GGUF data: vocabulary size is zero".to_string()
        ));
    }

    if two.vocab > 1_000_000 { // Reasonable upper bound
        return Err(BitNetError::Validation(
            format!("Invalid GGUF data: vocabulary size too large ({})", two.vocab)
        ));
    }

    // Validate dimensions
    if two.dim == 0 {
        return Err(BitNetError::Validation(
            "Invalid GGUF data: hidden dimension is zero".to_string()
        ));
    }

    if two.dim > 32768 { // Reasonable upper bound for hidden dimension
        return Err(BitNetError::Validation(
            format!("Invalid GGUF data: hidden dimension too large ({})", two.dim)
        ));
    }

    // Validate tensor count
    if two.tensors.is_empty() {
        return Err(BitNetError::Validation(
            "Invalid GGUF data: no tensors found".to_string()
        ));
    }

    tracing::debug!(
        "GGUF validation passed: vocab={}, dim={}, tensor_count={}",
        two.vocab, two.dim, two.tensors.len()
    );

    Ok(())
}

/// Create BitNet configuration from validated GGUF data
fn create_config_from_gguf(two: &crate::gguf_min::TwoTensors) -> Result<bitnet_common::BitNetConfig> {
    let mut config = bitnet_common::BitNetConfig::default();

    // Update configuration with parsed values
    config.model.vocab_size = two.vocab as usize;
    config.model.hidden_size = two.dim as usize;

    // Validate configuration consistency
    validate_config_consistency(&config)?;

    Ok(config)
}

/// Load tensors from GGUF data with device-aware placement
fn load_tensors_from_gguf(
    two: &crate::gguf_min::TwoTensors,
    device: Device,
    config: &bitnet_common::BitNetConfig,
) -> Result<HashMap<String, CandleTensor>> {
    let cdevice = convert_device_to_candle(device)?;
    let mut tensor_map = HashMap::new();

    // Process each tensor with validation
    for (name, tensor_data) in &two.tensors {
        let tensor = process_gguf_tensor(name, tensor_data, &cdevice, config)
            .map_err(|e| BitNetError::Validation(
                format!("Failed to process tensor '{}': {}", name, e)
            ))?;

        tensor_map.insert(name.clone(), tensor);
    }

    // Validate minimum required tensors are present
    validate_required_tensors(&tensor_map, config)?;

    tracing::info!("Successfully loaded {} tensors from GGUF", tensor_map.len());
    Ok(tensor_map)
}

/// Validate that required tensors are present
fn validate_required_tensors(
    tensor_map: &HashMap<String, CandleTensor>,
    config: &bitnet_common::BitNetConfig,
) -> Result<()> {
    let required_tensors = vec![
        "token_embd.weight",
        "output.weight",
    ];

    for required in &required_tensors {
        if !tensor_map.contains_key(*required) {
            return Err(BitNetError::Validation(
                format!("Missing required tensor: {}", required)
            ));
        }
    }

    // Check for at least one attention layer
    let has_attention = tensor_map.keys()
        .any(|name| name.contains("attn_q.weight"));

    if !has_attention {
        return Err(BitNetError::Validation(
            "No attention layers found in GGUF file".to_string()
        ));
    }

    Ok(())
}

/// Final validation of the loaded model
fn validate_loaded_model(
    config: &bitnet_common::BitNetConfig,
    tensor_map: &HashMap<String, CandleTensor>,
) -> Result<()> {
    // Validate tensor shapes are consistent with configuration
    if let Some(token_embd) = tensor_map.get("token_embd.weight") {
        let shape = token_embd.shape();
        if shape.dims().len() != 2 {
            return Err(BitNetError::Validation(
                format!("Invalid token embedding shape: expected 2D, got {:?}", shape)
            ));
        }

        if shape.dims()[0] != config.model.vocab_size {
            return Err(BitNetError::Validation(
                format!(
                    "Token embedding vocab size mismatch: config={}, tensor={}",
                    config.model.vocab_size, shape.dims()[0]
                )
            ));
        }

        if shape.dims()[1] != config.model.hidden_size {
            return Err(BitNetError::Validation(
                format!(
                    "Token embedding hidden size mismatch: config={}, tensor={}",
                    config.model.hidden_size, shape.dims()[1]
                )
            ));
        }
    }

    tracing::info!("Model validation completed successfully");
    Ok(())
}

/// Convert Device to Candle device with proper error handling
fn convert_device_to_candle(device: Device) -> Result<CDevice> {
    match device {
        Device::Cpu => Ok(CDevice::Cpu),
        Device::Cuda(id) => {
            CDevice::new_cuda(id).map_err(|e| BitNetError::Validation(
                format!("Failed to initialize CUDA device {}: {}", id, e)
            ))
        },
        Device::Metal => {
            Err(BitNetError::Validation(
                "Metal device not supported in minimal GGUF loader".to_string()
            ))
        }
    }
}

/// Validate configuration consistency
fn validate_config_consistency(config: &bitnet_common::BitNetConfig) -> Result<()> {
    if config.model.vocab_size == 0 {
        return Err(BitNetError::Validation(
            "Configuration error: vocab_size cannot be zero".to_string()
        ));
    }

    if config.model.hidden_size == 0 {
        return Err(BitNetError::Validation(
            "Configuration error: hidden_size cannot be zero".to_string()
        ));
    }

    if config.model.num_layers == 0 {
        return Err(BitNetError::Validation(
            "Configuration error: num_layers cannot be zero".to_string()
        ));
    }

    Ok(())
}
```

### Enhanced Test Infrastructure

Move mock tensor creation to test-specific modules:

```rust
#[cfg(test)]
pub mod test_utils {
    use super::*;

    /// Create mock tensor layout for testing purposes only
    pub fn create_test_tensor_layout(
        device: Device,
    ) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
        let config = bitnet_common::BitNetConfig::default();
        let cdevice = convert_device_to_candle(device)?;

        let mut tensor_map = HashMap::new();

        // Create minimal test tensors
        let vocab_size = config.model.vocab_size;
        let hidden_size = config.model.hidden_size;

        // Token embeddings
        let token_embd = CandleTensor::zeros((vocab_size, hidden_size), DType::F32, &cdevice)?;
        tensor_map.insert("token_embd.weight".to_string(), token_embd);

        // Output weights
        let output = CandleTensor::zeros((vocab_size, hidden_size), DType::F32, &cdevice)?;
        tensor_map.insert("output.weight".to_string(), output);

        // Basic attention layer for layer 0
        let attn_q = CandleTensor::zeros((hidden_size, hidden_size), DType::F32, &cdevice)?;
        tensor_map.insert("blk.0.attn_q.weight".to_string(), attn_q);

        Ok((config, tensor_map))
    }

    /// Create a mock GGUF file for testing
    pub fn create_mock_gguf_file(path: &Path) -> Result<()> {
        std::fs::write(path, b"mock_gguf_content")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test_utils::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_gguf_file_validation() {
        // Test non-existent file
        let result = validate_gguf_file(&Path::new("nonexistent.gguf"));
        assert!(result.is_err());

        // Test with mock file
        let temp_file = NamedTempFile::new().unwrap();
        create_mock_gguf_file(temp_file.path()).unwrap();

        let result = validate_gguf_file(temp_file.path());
        assert!(result.is_err()); // Should fail due to insufficient size
    }

    #[test]
    fn test_mock_tensor_layout_creation() {
        let result = create_test_tensor_layout(Device::Cpu);
        assert!(result.is_ok());

        let (config, tensors) = result.unwrap();
        assert!(tensors.contains_key("token_embd.weight"));
        assert!(tensors.contains_key("output.weight"));
        assert_eq!(config.model.vocab_size, 50257);
    }
}
```

## Implementation Roadmap

### Phase 1: Error Handling Infrastructure (2-3 days)
- [ ] Implement comprehensive file validation functions
- [ ] Create detailed error reporting with diagnostic information
- [ ] Add GGUF data integrity validation
- [ ] Create robust device conversion with error handling

### Phase 2: Core Loading Logic (2-3 days)
- [ ] Refactor `load_gguf_minimal` with enhanced error handling
- [ ] Implement tensor validation and consistency checks
- [ ] Add required tensor presence validation
- [ ] Create comprehensive model validation

### Phase 3: Test Infrastructure Separation (1-2 days)
- [ ] Move mock tensor creation to test-only modules
- [ ] Update existing tests to use new test utilities
- [ ] Ensure production code has no test dependencies
- [ ] Add comprehensive test coverage for error scenarios

### Phase 4: Integration and Validation (1-2 days)
- [ ] Integrate enhanced minimal loader with existing infrastructure
- [ ] Test with various GGUF file types (valid, invalid, corrupted)
- [ ] Validate error messages are helpful for debugging
- [ ] Performance testing to ensure no regression

## Testing Strategy

### Test Coverage Requirements
- [ ] Unit tests for all validation functions
- [ ] Integration tests with real GGUF files
- [ ] Error scenario testing (corrupted files, invalid data)
- [ ] Device compatibility testing (CPU/GPU)
- [ ] Performance regression testing

### Error Scenario Testing
```rust
#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_corrupted_gguf_handling() {
        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), b"corrupted_data").unwrap();

        let result = load_gguf_minimal(temp_file.path(), Device::Cpu);
        assert!(result.is_err());

        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("Failed to parse GGUF file"));
        assert!(error_msg.contains("File diagnostics"));
    }

    #[test]
    fn test_empty_file_handling() {
        let temp_file = NamedTempFile::new().unwrap();
        // File exists but is empty

        let result = load_gguf_minimal(temp_file.path(), Device::Cpu);
        assert!(result.is_err());

        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("too small"));
    }

    #[test]
    fn test_nonexistent_file_handling() {
        let result = load_gguf_minimal(&Path::new("nonexistent.gguf"), Device::Cpu);
        assert!(result.is_err());

        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("does not exist"));
    }
}
```

## Acceptance Criteria

- [ ] **Mock Code Removal**: All mock tensor creation removed from production code paths
- [ ] **Robust Error Handling**: Comprehensive error handling with detailed diagnostic information
- [ ] **Test Separation**: Clean separation between test utilities and production code
- [ ] **Validation Framework**: Complete validation of GGUF files, data, and loaded models
- [ ] **Error Messages**: User-friendly, actionable error messages for all failure scenarios
- [ ] **Backward Compatibility**: Existing functionality preserved for valid GGUF files
- [ ] **Performance**: No performance regression in GGUF loading

## Related Issues

- Enhanced GGUF parser improvements
- Test infrastructure standardization across crates
- Error handling consistency project
- Production deployment validation framework

---

**Labels**: `bug`, `production-ready`, `error-handling`, `P2-medium`, `models`
**Priority**: Medium-High - Important for production reliability
**Effort**: 6-8 days
