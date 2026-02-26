# [MODELS] Replace minimal GGUF stub implementation with robust error handling and proper test infrastructure

## Problem Description

The current `load_gguf_minimal` and `create_mock_tensor_layout` functions in `crates/bitnet-models/src/gguf_simple.rs` create mock tensors instead of returning proper validation errors when encountering incomplete GGUF files. This stubbing pattern:

1. **Masks real validation failures**: Creates mock tensors for missing transformer layers instead of failing gracefully
2. **Misleads developers**: Mock tensors appear as valid data, making debugging incomplete GGUF files difficult
3. **Introduces test infrastructure leakage**: `create_mock_tensor_layout` is used in production code paths rather than being test-only
4. **Violates BitNet-rs architecture principles**: Goes against the robust error handling and validation-first approach

## Current Implementation Analysis

### Root Cause Investigation

The problem stems from the fallback mechanism in `load_gguf_minimal()` (lines 154-271):

```rust
// If minimal GGUF parsing also fails, check if this is a mock file from tests
if let Ok(content) = std::fs::read(path)
    && content == b"mock_gguf_content"
{
    // Creates mock tensors instead of failing
    return create_mock_tensor_layout(device);
}

// Creates mock tensors for missing transformer layers (lines 220-262)
for layer in 0..num_layers {
    let prefix = format!("blk.{}", layer);
    tensor_map.insert(
        format!("{}.attn_q.weight", prefix),
        CandleTensor::zeros(&[hidden_size, hidden_size], dtype, &cdevice)?,
    );
    // ... creates more mock tensors
}
```

### Affected Components

**File:** `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/gguf_simple.rs`

**Functions:**
- `load_gguf_minimal()` (lines 154-271) - Creates mock tensors for missing transformer layers
- `create_mock_tensor_layout()` (lines 670-804) - Test helper used in production code

**Impact Assessment:**
- **Severity**: High - Silent failures and incorrect validation
- **Affected Users**: All developers using incomplete GGUF files or test infrastructure
- **Performance**: Low - Mock tensors consume memory unnecessarily
- **Business Impact**: Moderate - Impedes debugging and proper error handling

## Reproduction Steps

1. **Setup incomplete GGUF file:**
   ```bash
   echo "mock_gguf_content" > incomplete_model.gguf
   ```

2. **Load using BitNet-rs:**
   ```rust
   use bitnet_models::load_gguf;
   use bitnet_common::Device;

   // This should fail but creates mock tensors instead
   let result = load_gguf(Path::new("incomplete_model.gguf"), Device::Cpu);
   println!("Result: {:?}", result); // Shows success with mock data
   ```

3. **Expected vs Actual Results:**
   - **Expected**: `BitNetError::Validation("Failed to parse GGUF file")`
   - **Actual**: `Ok((config, tensor_map))` with mock tensors

## Root Cause Analysis

### Technical Investigation

1. **Fallback Logic Issue**: The `load_gguf_minimal` function uses `create_mock_tensor_layout` as a fallback when GGUF parsing fails
2. **Test Infrastructure Leakage**: Mock creation logic is embedded in production code rather than being test-only
3. **Validation Bypass**: Missing tensors are filled with zeros/ones instead of triggering validation errors
4. **Error Context Loss**: Original parsing errors are masked by mock tensor creation

### Architecture Violation

This pattern violates BitNet-rs core principles:
- **Validation-First**: Should fail fast on invalid inputs
- **Robust Error Handling**: Should provide descriptive error messages
- **Test Isolation**: Test helpers should not be in production paths

## Proposed Solution

### Primary Implementation: Robust Error Handling

Replace stub implementation with proper validation and error reporting:

```rust
/// Enhanced minimal GGUF loading with proper validation
fn load_gguf_minimal(
    path: &Path,
    device: Device
) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
    // AC4: Enhanced error handling with context
    let two = crate::gguf_min::load_two(path).map_err(|e| {
        BitNetError::Model(ModelError::GGUFFormatError {
            message: format!("Failed to parse GGUF file '{}'", path.display()),
            details: ValidationErrorDetails {
                errors: vec![format!("Minimal parser error: {}", e)],
                warnings: vec![],
                recommendations: vec![
                    "Verify GGUF file integrity".to_string(),
                    "Check file format compatibility".to_string(),
                    "Ensure complete model download".to_string(),
                ],
            },
        })
    })?;

    // Extract configuration and validate completeness
    let mut config = bitnet_common::BitNetConfig::default();
    config.model.vocab_size = two.vocab;
    config.model.hidden_size = two.dim;

    // AC3: Strict validation - fail if essential tensors missing
    validate_minimal_tensor_completeness(&two, &config)?;

    // Create tensor map with only available tensors
    let cdevice = device_to_candle_device(device)?;
    let mut tensor_map = HashMap::new();

    // Load only the tensors we successfully parsed
    tensor_map.insert(
        "token_embd.weight".to_string(),
        CandleTensor::from_vec(two.tok_embeddings, (two.vocab, two.dim), &cdevice)?,
    );
    tensor_map.insert(
        "output.weight".to_string(),
        CandleTensor::from_vec(two.lm_head, (two.dim, two.vocab), &cdevice)?,
    );

    // AC4: Fail fast for incomplete transformer layers
    validate_transformer_layer_completeness(&tensor_map, &config)?;

    Ok((config, tensor_map))
}

/// Validate that minimal parser extracted essential tensors
fn validate_minimal_tensor_completeness(
    two: &TwoTensors,
    config: &BitNetConfig,
) -> Result<()> {
    if two.vocab == 0 || two.dim == 0 {
        return Err(BitNetError::Validation(
            "Invalid model dimensions: vocab_size and hidden_size must be > 0".to_string()
        ));
    }

    if two.tok_embeddings.len() != two.vocab * two.dim {
        return Err(BitNetError::Validation(format!(
            "Token embeddings size mismatch: expected {}, got {}",
            two.vocab * two.dim,
            two.tok_embeddings.len()
        )));
    }

    if two.lm_head.len() != two.dim * two.vocab {
        return Err(BitNetError::Validation(format!(
            "LM head size mismatch: expected {}, got {}",
            two.dim * two.vocab,
            two.lm_head.len()
        )));
    }

    Ok(())
}

/// Validate transformer layer completeness - fail if missing critical layers
fn validate_transformer_layer_completeness(
    tensor_map: &HashMap<String, CandleTensor>,
    config: &BitNetConfig,
) -> Result<()> {
    let mut missing_layers = Vec::new();

    // Check for any transformer layers - if none exist, this is incomplete
    let has_any_layer_tensors = (0..config.model.num_layers).any(|layer| {
        let prefix = format!("blk.{}", layer);
        tensor_map.keys().any(|k| k.starts_with(&prefix))
    });

    if !has_any_layer_tensors {
        return Err(BitNetError::Model(ModelError::GGUFFormatError {
            message: "Incomplete GGUF file: missing all transformer layers".to_string(),
            details: ValidationErrorDetails {
                errors: vec![
                    "No transformer layer weights found".to_string(),
                    "Minimal parser can only extract embeddings and output projection".to_string(),
                ],
                warnings: vec![],
                recommendations: vec![
                    "Use a complete GGUF file with full transformer weights".to_string(),
                    "Verify model file integrity and completeness".to_string(),
                    "Check if enhanced GGUF parser can handle this file".to_string(),
                ],
            },
        }));
    }

    Ok(())
}
```

### Test Infrastructure Isolation

Move mock creation to test-only module:

```rust
#[cfg(test)]
pub mod test_helpers {
    use super::*;

    /// Create mock tensor layout EXCLUSIVELY for testing
    ///
    /// # Safety
    /// This function is only available in test builds and should NEVER
    /// be called from production code. Mock tensors do not represent
    /// real model weights and will produce invalid inference results.
    pub fn create_mock_tensor_layout_for_testing(
        device: Device,
    ) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
        let config = bitnet_common::BitNetConfig::default();
        // ... existing mock creation logic ...

        tracing::warn!(
            "TESTING ONLY: Created mock tensor layout with {} tensors. \
             This is not suitable for production inference!",
            tensor_map.len()
        );

        Ok((config, tensor_map))
    }

    /// Create mock GGUF file content for testing
    pub fn create_mock_gguf_file(path: &Path) -> Result<()> {
        std::fs::write(path, b"test_mock_gguf_content")
            .map_err(|e| BitNetError::Io(e))?;
        Ok(())
    }
}
```

### Alternative Approaches

1. **Enhanced Fallback with Warning**: Keep mock creation but add prominent warnings
2. **Partial Loading Mode**: Load available tensors and clearly mark missing ones
3. **Compatibility Layer**: Separate compatibility mode for legacy test files

**Recommended**: Primary solution with strict validation and test isolation

## Implementation Plan

### Phase 1: Core Validation Implementation
**Duration**: 2-3 days
**Priority**: High

1. **Replace stub logic in `load_gguf_minimal`**:
   - Remove mock tensor creation from production path
   - Add comprehensive validation with descriptive errors
   - Implement `validate_minimal_tensor_completeness`
   - Add `validate_transformer_layer_completeness`

2. **Update error handling**:
   - Use `ModelError::GGUFFormatError` with `ValidationErrorDetails`
   - Provide actionable error messages and recommendations
   - Include context about minimal vs enhanced parser capabilities

3. **Device handling improvements**:
   - Extract `device_to_candle_device` helper function
   - Add proper error handling for device failures
   - Maintain GPU fallback behavior

### Phase 2: Test Infrastructure Cleanup
**Duration**: 1-2 days
**Priority**: Medium

1. **Isolate test helpers**:
   - Move `create_mock_tensor_layout` to `#[cfg(test)]` module
   - Rename to `create_mock_tensor_layout_for_testing`
   - Add safety warnings and documentation

2. **Update test infrastructure**:
   - Modify test files to use test-only helpers
   - Replace hardcoded `b"mock_gguf_content"` with test helper
   - Add proper test file creation utilities

3. **Clean up test detection logic**:
   - Remove production code that detects test files
   - Ensure tests use proper test infrastructure

### Phase 3: Enhanced Error Reporting
**Duration**: 1 day
**Priority**: Low

1. **Improve error context**:
   - Add file size and format detection to error messages
   - Include hints about required GGUF structure
   - Provide guidance for fixing incomplete files

2. **Add validation metrics**:
   - Report which tensors were successfully loaded
   - Show completeness percentage for debugging
   - Add tensor count and size information

### Phase 4: Integration and Testing
**Duration**: 1-2 days
**Priority**: High

1. **Comprehensive testing**:
   - Test with various incomplete GGUF files
   - Verify error messages are helpful and actionable
   - Ensure no regression in valid file loading

2. **Cross-validation**:
   - Test fallback behavior from enhanced to minimal parser
   - Verify device-specific error handling
   - Check memory usage and performance impact

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod validation_tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_load_gguf_minimal_rejects_incomplete_files() {
        let temp_dir = TempDir::new().unwrap();
        let incomplete_path = temp_dir.path().join("incomplete.gguf");

        // Create file with invalid content
        std::fs::write(&incomplete_path, b"invalid_gguf").unwrap();

        let result = load_gguf_minimal(&incomplete_path, Device::Cpu);

        // Should fail with descriptive error
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Failed to parse GGUF file"));
        assert!(error_msg.contains("Verify GGUF file integrity"));
    }

    #[test]
    fn test_mock_tensor_layout_only_in_tests() {
        // This should compile only in test builds
        let result = test_helpers::create_mock_tensor_layout_for_testing(Device::Cpu);
        assert!(result.is_ok());

        let (config, tensor_map) = result.unwrap();
        assert!(!tensor_map.is_empty());
        assert!(config.model.vocab_size > 0);
    }

    #[test]
    fn test_validation_error_details() {
        let temp_dir = TempDir::new().unwrap();
        let empty_path = temp_dir.path().join("empty.gguf");
        std::fs::write(&empty_path, b"").unwrap();

        let result = load_gguf_minimal(&empty_path, Device::Cpu);

        match result.unwrap_err() {
            BitNetError::Model(ModelError::GGUFFormatError { details, .. }) => {
                assert!(!details.errors.is_empty());
                assert!(!details.recommendations.is_empty());
            }
            _ => panic!("Expected GGUFFormatError with details"),
        }
    }
}
```

### Integration Tests
```rust
#[test]
fn test_enhanced_to_minimal_fallback() {
    // Test that fallback from enhanced parser works correctly
    // and provides proper error messages when minimal parser also fails
}

#[test]
fn test_device_specific_error_handling() {
    // Test CUDA device unavailable scenarios
    // Verify graceful fallback to CPU with proper error reporting
}
```

### Property-Based Tests
```rust
proptest! {
    #[test]
    fn test_validation_never_silently_succeeds(
        file_size in 0usize..1024,
        content in prop::collection::vec(any::<u8>(), 0..1024)
    ) {
        // Property: Invalid GGUF files should never silently succeed
        // They should either load correctly or fail with descriptive errors
    }
}
```

## Acceptance Criteria

### ✅ Primary Requirements

1. **AC1: Eliminate mock tensor creation in production**
   - `load_gguf_minimal` never creates mock tensors for missing layers
   - Function returns validation errors for incomplete GGUF files
   - No silent failures that mask real parsing issues

2. **AC2: Comprehensive error reporting**
   - Use `ModelError::GGUFFormatError` with `ValidationErrorDetails`
   - Include actionable recommendations in error messages
   - Provide context about minimal vs enhanced parser capabilities

3. **AC3: Test infrastructure isolation**
   - Move `create_mock_tensor_layout` to `#[cfg(test)]` module
   - Rename to `create_mock_tensor_layout_for_testing`
   - Add clear documentation about test-only usage

4. **AC4: Robust validation**
   - Validate tensor dimensions and completeness
   - Check for required embeddings and output projections
   - Fail fast on missing transformer layers with clear error messages

### ✅ Quality Requirements

5. **AC5: Backward compatibility**
   - Valid GGUF files continue to load correctly
   - Fallback mechanism from enhanced parser still works
   - No breaking changes to public API

6. **AC6: Performance optimization**
   - No unnecessary memory allocation for mock tensors
   - Efficient validation without full tensor loading
   - Maintain zero-copy operations where possible

7. **AC7: Enhanced debugging**
   - Error messages include file path and format details
   - Recommendations guide users to fix incomplete files
   - Validation reports which tensors were found vs missing

### ✅ Testing Requirements

8. **AC8: Comprehensive test coverage**
   - Unit tests for all validation scenarios
   - Integration tests for parser fallback behavior
   - Property-based tests for robustness validation

9. **AC9: Cross-validation testing**
   - Test with various incomplete GGUF files
   - Verify error handling across different devices (CPU/GPU)
   - Ensure proper fallback behavior and error propagation

## Related Issues and Dependencies

### Cross-References
- **Related to**: Issue #251 - Production-Ready Inference Server (error handling improvements)
- **Depends on**: GGUF format validation infrastructure in `bitnet-models`
- **Blocks**: Complete BitNet-rs validation framework implementation
- **References**: `docs/explanation/issue-260-mock-elimination-spec.md`

### Affected Components
- `crates/bitnet-models/src/gguf_simple.rs` - Primary implementation
- `crates/bitnet-models/src/gguf_min.rs` - Minimal parser interface
- `crates/bitnet-common/src/error.rs` - Error type definitions
- `crates/bitnet-models/tests/gguf_*_tests.rs` - Test infrastructure

### Integration Points
- GGUF enhanced parser fallback mechanism
- Device-aware tensor placement with error handling
- BitNet quantization error propagation
- Test infrastructure and mock file handling

## Labels and Classification

**Labels:**
- `area: models` - GGUF loading and model infrastructure
- `area: testing` - Test infrastructure improvements
- `priority: high` - Blocks proper error handling and debugging
- `type: enhancement` - Improves existing functionality
- `type: refactor` - Code structure and architecture improvements

**Priority:** High - Masks validation failures and impedes debugging

**Effort Estimate:** 5-7 days (Medium)

**Assignee Suggestion:** Developer familiar with GGUF format and BitNet error handling

---

**Implementation Notes:**
- Follow BitNet-rs validation-first architecture principles
- Maintain compatibility with existing enhanced GGUF parser
- Ensure proper error propagation through the inference stack
- Use `tracing` for debugging information in development builds
- Consider security implications of file validation and resource limits
