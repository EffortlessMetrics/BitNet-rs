# [TEST] GGUF Two-Tensor Loading Test Implementation for Production Validation

## Problem Description

The `loads_two_tensors` test in the GGUF minimal loader is currently ignored and depends on external environment variables, preventing continuous integration validation of critical tensor loading functionality. This test gap leaves the two-tensor loading path unvalidated, potentially allowing regressions in model loading logic to go undetected.

## Environment

- **Component**: `bitnet-models` crate
- **File**: `crates/bitnet-models/src/gguf_min.rs`
- **Test**: `loads_two_tensors`
- **Current State**: Ignored with `#[ignore]` attribute
- **Dependencies**: External `BITNET_GGUF` environment variable

## Current Implementation Analysis

### Incomplete Test Implementation
```rust
#[test]
#[ignore] // set BITNET_GGUF to a real path to run
fn loads_two_tensors() {
    let p = std::env::var_os("BITNET_GGUF").expect("set BITNET_GGUF");
    let two = load_two(p).unwrap();
    assert!(two.vocab > 0 && two.dim > 0);
    assert_eq!(two.tok_embeddings.len(), two.vocab * two.dim);
    assert_eq!(two.lm_head.len(), two.dim * two.vocab);
}
```

### Issues with Current Approach
1. **External Dependency**: Requires specific GGUF file to be available
2. **Ignored in CI**: Test doesn't run in continuous integration
3. **No Validation**: Missing comprehensive validation of loaded tensor data
4. **Brittle Setup**: Manual environment configuration required
5. **Limited Coverage**: Only tests basic size assertions

## Root Cause Analysis

1. **Test Infrastructure Gap**: Missing synthetic GGUF file generation for testing
2. **Environment Dependencies**: Reliance on external files rather than controlled test data
3. **Incomplete Validation**: Limited assertions on tensor content and structure
4. **CI Integration**: Test disabled preventing regression detection

## Impact Assessment

**Severity**: Medium-High - Critical model loading path unvalidated

**Testing Gap Impact**:
- Two-tensor loading regressions undetected
- GGUF format parsing issues missed
- Vocabulary and dimension extraction failures
- Memory layout and tensor alignment issues

**Development Impact**:
- Reduced confidence in model loading reliability
- Potential production failures not caught early
- Manual testing burden on developers

## Proposed Solution

### Comprehensive Two-Tensor Loading Test Suite

```rust
use std::io::{Write, Cursor};
use tempfile::NamedTempFile;
use byteorder::{LittleEndian, WriteBytesExt};

/// GGUF format constants
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
const GGUF_VERSION: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// Test data structure for two-tensor GGUF validation
#[derive(Debug, Clone)]
struct TwoTensorTestCase {
    vocab_size: usize,
    hidden_dim: usize,
    embedding_data: Vec<f32>,
    lm_head_data: Vec<f32>,
    metadata: Vec<(String, MetadataValue)>,
}

#[derive(Debug, Clone)]
enum MetadataValue {
    String(String),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
}

impl TwoTensorTestCase {
    fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        let embedding_elements = vocab_size * hidden_dim;
        let lm_head_elements = hidden_dim * vocab_size;

        // Generate deterministic test data
        let embedding_data: Vec<f32> = (0..embedding_elements)
            .map(|i| (i as f32 * 0.001) % 2.0 - 1.0) // Values in [-1, 1]
            .collect();

        let lm_head_data: Vec<f32> = (0..lm_head_elements)
            .map(|i| ((i + embedding_elements) as f32 * 0.001) % 2.0 - 1.0)
            .collect();

        let metadata = vec![
            ("general.architecture".to_string(), MetadataValue::String("llama".to_string())),
            ("llama.vocab_size".to_string(), MetadataValue::UInt32(vocab_size as u32)),
            ("llama.embedding_length".to_string(), MetadataValue::UInt32(hidden_dim as u32)),
            ("llama.block_count".to_string(), MetadataValue::UInt32(12)),
            ("llama.attention.head_count".to_string(), MetadataValue::UInt32(8)),
        ];

        Self {
            vocab_size,
            hidden_dim,
            embedding_data,
            lm_head_data,
            metadata,
        }
    }

    /// Generate a complete GGUF file for testing
    fn generate_gguf_file(&self) -> Result<Vec<u8>, std::io::Error> {
        let mut buffer = Vec::new();

        // Write GGUF header
        self.write_gguf_header(&mut buffer)?;

        // Write metadata
        self.write_metadata(&mut buffer)?;

        // Write tensor info
        self.write_tensor_info(&mut buffer)?;

        // Align to tensor data boundary
        self.align_buffer(&mut buffer, GGUF_DEFAULT_ALIGNMENT)?;

        // Write tensor data
        self.write_tensor_data(&mut buffer)?;

        Ok(buffer)
    }

    fn write_gguf_header(&self, buffer: &mut Vec<u8>) -> Result<(), std::io::Error> {
        buffer.write_u32::<LittleEndian>(GGUF_MAGIC)?;
        buffer.write_u32::<LittleEndian>(GGUF_VERSION)?;
        buffer.write_u64::<LittleEndian>(2)?; // tensor_count
        buffer.write_u64::<LittleEndian>(self.metadata.len() as u64)?; // metadata_kv_count
        Ok(())
    }

    fn write_metadata(&self, buffer: &mut Vec<u8>) -> Result<(), std::io::Error> {
        for (key, value) in &self.metadata {
            // Write key
            buffer.write_u64::<LittleEndian>(key.len() as u64)?;
            buffer.write_all(key.as_bytes())?;

            // Write value type and data
            match value {
                MetadataValue::String(s) => {
                    buffer.write_u32::<LittleEndian>(8)?; // GGUF_TYPE_STRING
                    buffer.write_u64::<LittleEndian>(s.len() as u64)?;
                    buffer.write_all(s.as_bytes())?;
                }
                MetadataValue::UInt32(v) => {
                    buffer.write_u32::<LittleEndian>(4)?; // GGUF_TYPE_UINT32
                    buffer.write_u32::<LittleEndian>(*v)?;
                }
                MetadataValue::UInt64(v) => {
                    buffer.write_u32::<LittleEndian>(5)?; // GGUF_TYPE_UINT64
                    buffer.write_u64::<LittleEndian>(*v)?;
                }
                MetadataValue::Float32(v) => {
                    buffer.write_u32::<LittleEndian>(2)?; // GGUF_TYPE_FLOAT32
                    buffer.write_f32::<LittleEndian>(*v)?;
                }
            }
        }
        Ok(())
    }

    fn write_tensor_info(&self, buffer: &mut Vec<u8>) -> Result<(), std::io::Error> {
        let mut tensor_offset = 0u64;

        // Write tok_embeddings tensor info
        self.write_single_tensor_info(
            buffer,
            "tok_embeddings.weight",
            &[self.vocab_size as u64, self.hidden_dim as u64],
            0, // GGUF_TYPE_F32
            tensor_offset,
        )?;

        tensor_offset += (self.embedding_data.len() * 4) as u64; // 4 bytes per f32

        // Write lm_head tensor info
        self.write_single_tensor_info(
            buffer,
            "lm_head.weight",
            &[self.hidden_dim as u64, self.vocab_size as u64],
            0, // GGUF_TYPE_F32
            tensor_offset,
        )?;

        Ok(())
    }

    fn write_single_tensor_info(
        &self,
        buffer: &mut Vec<u8>,
        name: &str,
        shape: &[u64],
        tensor_type: u32,
        offset: u64,
    ) -> Result<(), std::io::Error> {
        // Write tensor name
        buffer.write_u64::<LittleEndian>(name.len() as u64)?;
        buffer.write_all(name.as_bytes())?;

        // Write number of dimensions
        buffer.write_u32::<LittleEndian>(shape.len() as u32)?;

        // Write shape
        for &dim in shape {
            buffer.write_u64::<LittleEndian>(dim)?;
        }

        // Write tensor type
        buffer.write_u32::<LittleEndian>(tensor_type)?;

        // Write offset
        buffer.write_u64::<LittleEndian>(offset)?;

        Ok(())
    }

    fn align_buffer(&self, buffer: &mut Vec<u8>, alignment: u64) -> Result<(), std::io::Error> {
        let current_len = buffer.len() as u64;
        let padding = (alignment - (current_len % alignment)) % alignment;
        buffer.extend(vec![0u8; padding as usize]);
        Ok(())
    }

    fn write_tensor_data(&self, buffer: &mut Vec<u8>) -> Result<(), std::io::Error> {
        // Write tok_embeddings data
        for &value in &self.embedding_data {
            buffer.write_f32::<LittleEndian>(value)?;
        }

        // Write lm_head data
        for &value in &self.lm_head_data {
            buffer.write_f32::<LittleEndian>(value)?;
        }

        Ok(())
    }
}

/// Comprehensive test suite for two-tensor GGUF loading
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_two_tensors_basic() {
        let test_case = TwoTensorTestCase::new(1000, 512);
        let gguf_data = test_case.generate_gguf_file().unwrap();

        // Create temporary file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&gguf_data).unwrap();

        // Test the actual loading function
        let two = load_two(temp_file.path()).unwrap();

        // Validate basic properties
        assert_eq!(two.vocab, 1000);
        assert_eq!(two.dim, 512);
        assert_eq!(two.tok_embeddings.len(), two.vocab * two.dim);
        assert_eq!(two.lm_head.len(), two.dim * two.vocab);
    }

    #[test]
    fn loads_two_tensors_data_validation() {
        let test_case = TwoTensorTestCase::new(100, 64);
        let gguf_data = test_case.generate_gguf_file().unwrap();

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&gguf_data).unwrap();

        let two = load_two(temp_file.path()).unwrap();

        // Validate tensor data integrity
        let expected_embedding_len = test_case.embedding_data.len();
        let expected_lm_head_len = test_case.lm_head_data.len();

        assert_eq!(two.tok_embeddings.len(), expected_embedding_len);
        assert_eq!(two.lm_head.len(), expected_lm_head_len);

        // Validate first few elements to ensure data was loaded correctly
        for i in 0..std::cmp::min(10, expected_embedding_len) {
            let expected = test_case.embedding_data[i];
            let actual = two.tok_embeddings[i];
            assert!((expected - actual).abs() < 1e-6,
                   "Embedding mismatch at {}: expected {}, got {}", i, expected, actual);
        }

        for i in 0..std::cmp::min(10, expected_lm_head_len) {
            let expected = test_case.lm_head_data[i];
            let actual = two.lm_head[i];
            assert!((expected - actual).abs() < 1e-6,
                   "LM head mismatch at {}: expected {}, got {}", i, expected, actual);
        }
    }

    #[test]
    fn loads_two_tensors_various_sizes() {
        let test_cases = vec![
            (100, 64),    // Small model
            (32000, 4096), // LLaMA-like dimensions
            (50257, 768),  // GPT-2 dimensions
            (8192, 8192),  // Square dimensions
        ];

        for (vocab_size, hidden_dim) in test_cases {
            let test_case = TwoTensorTestCase::new(vocab_size, hidden_dim);
            let gguf_data = test_case.generate_gguf_file().unwrap();

            let mut temp_file = NamedTempFile::new().unwrap();
            temp_file.write_all(&gguf_data).unwrap();

            let two = load_two(temp_file.path()).unwrap();

            assert_eq!(two.vocab, vocab_size,
                      "Vocab size mismatch for {}x{} model", vocab_size, hidden_dim);
            assert_eq!(two.dim, hidden_dim,
                      "Hidden dim mismatch for {}x{} model", vocab_size, hidden_dim);
            assert_eq!(two.tok_embeddings.len(), vocab_size * hidden_dim);
            assert_eq!(two.lm_head.len(), hidden_dim * vocab_size);
        }
    }

    #[test]
    fn loads_two_tensors_memory_layout() {
        let test_case = TwoTensorTestCase::new(256, 128);
        let gguf_data = test_case.generate_gguf_file().unwrap();

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&gguf_data).unwrap();

        let two = load_two(temp_file.path()).unwrap();

        // Validate memory layout assumptions
        // Check that tensors are contiguous and properly aligned
        let embedding_ptr = two.tok_embeddings.as_ptr();
        let lm_head_ptr = two.lm_head.as_ptr();

        // Ensure pointers are properly aligned (at least 4-byte aligned for f32)
        assert_eq!(embedding_ptr as usize % 4, 0, "Embedding tensor not aligned");
        assert_eq!(lm_head_ptr as usize % 4, 0, "LM head tensor not aligned");

        // Check that data is laid out as expected
        // (This depends on the specific memory layout of your tensors)
        assert_ne!(embedding_ptr, lm_head_ptr, "Tensors should not share memory");
    }

    #[test]
    fn loads_two_tensors_error_handling() {
        // Test various error conditions

        // Invalid GGUF magic
        let mut invalid_data = vec![0u8; 1024];
        invalid_data[0..4].copy_from_slice(b"BADD"); // Wrong magic
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&invalid_data).unwrap();
        assert!(load_two(temp_file.path()).is_err());

        // Empty file
        let empty_file = NamedTempFile::new().unwrap();
        assert!(load_two(empty_file.path()).is_err());

        // Non-existent file
        assert!(load_two("non_existent_file.gguf").is_err());
    }

    #[test]
    fn loads_two_tensors_metadata_validation() {
        let test_case = TwoTensorTestCase::new(500, 256);
        let gguf_data = test_case.generate_gguf_file().unwrap();

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&gguf_data).unwrap();

        let two = load_two(temp_file.path()).unwrap();

        // Validate that metadata was correctly parsed and used
        assert_eq!(two.vocab, 500);
        assert_eq!(two.dim, 256);

        // Additional validation could include checking that metadata
        // values match what we expect from the test case
    }

    #[test]
    fn loads_two_tensors_performance() {
        // Performance test to ensure loading doesn't regress
        let test_case = TwoTensorTestCase::new(10000, 1024); // Reasonably large
        let gguf_data = test_case.generate_gguf_file().unwrap();

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&gguf_data).unwrap();

        let start = std::time::Instant::now();
        let two = load_two(temp_file.path()).unwrap();
        let duration = start.elapsed();

        // Should load reasonably quickly (adjust threshold as needed)
        assert!(duration.as_secs() < 5, "Loading took too long: {:?}", duration);

        // Validate it actually loaded the data
        assert_eq!(two.vocab, 10000);
        assert_eq!(two.dim, 1024);
    }

    /// Integration test that validates the entire pipeline
    #[test]
    fn two_tensor_loading_integration() {
        let test_cases = vec![
            ("small", 32, 16),
            ("medium", 1000, 512),
            ("large", 8192, 2048),
        ];

        for (name, vocab_size, hidden_dim) in test_cases {
            println!("Testing {} model ({}x{})", name, vocab_size, hidden_dim);

            let test_case = TwoTensorTestCase::new(vocab_size, hidden_dim);
            let gguf_data = test_case.generate_gguf_file().unwrap();

            let mut temp_file = NamedTempFile::new().unwrap();
            temp_file.write_all(&gguf_data).unwrap();

            // Test loading
            let two = load_two(temp_file.path()).unwrap();

            // Comprehensive validation
            assert_eq!(two.vocab, vocab_size);
            assert_eq!(two.dim, hidden_dim);
            assert_eq!(two.tok_embeddings.len(), vocab_size * hidden_dim);
            assert_eq!(two.lm_head.len(), hidden_dim * vocab_size);

            // Validate non-zero data
            assert!(two.tok_embeddings.iter().any(|&x| x != 0.0),
                   "Embedding tensor should contain non-zero values");
            assert!(two.lm_head.iter().any(|&x| x != 0.0),
                   "LM head tensor should contain non-zero values");

            // Validate reasonable value ranges
            let embedding_max = two.tok_embeddings.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let embedding_min = two.tok_embeddings.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            assert!(embedding_max <= 1.0 && embedding_min >= -1.0,
                   "Embedding values outside expected range [-1, 1]");

            println!("✓ {} model validation passed", name);
        }
    }
}

/// Utility functions for test file generation
mod test_utils {
    use super::*;

    /// Create a minimal valid GGUF file for testing
    pub fn create_minimal_gguf(vocab_size: usize, hidden_dim: usize) -> Vec<u8> {
        TwoTensorTestCase::new(vocab_size, hidden_dim)
            .generate_gguf_file()
            .expect("Failed to generate test GGUF file")
    }

    /// Create a temporary GGUF file and return its path
    pub fn create_temp_gguf_file(vocab_size: usize, hidden_dim: usize) -> NamedTempFile {
        let gguf_data = create_minimal_gguf(vocab_size, hidden_dim);
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(&gguf_data).expect("Failed to write GGUF data");
        temp_file
    }
}
```

## Implementation Plan

### Phase 1: Test Infrastructure (Week 1)
- [ ] Implement GGUF file generation utilities
- [ ] Create comprehensive test case framework
- [ ] Add metadata and tensor info writing
- [ ] Establish error handling test cases

### Phase 2: Test Suite Implementation (Week 2)
- [ ] Implement basic two-tensor loading tests
- [ ] Add data validation and integrity checks
- [ ] Create performance and memory layout tests
- [ ] Add error condition testing

### Phase 3: Integration & CI (Week 3)
- [ ] Remove `#[ignore]` attributes and enable in CI
- [ ] Add test variations for different model sizes
- [ ] Integrate with existing test harness
- [ ] Add regression detection

### Phase 4: Documentation & Maintenance (Week 4)
- [ ] Document test architecture and utilities
- [ ] Add troubleshooting guide for test failures
- [ ] Create test data generation tools
- [ ] Establish test maintenance procedures

## Success Criteria

- [ ] **CI Integration**: Tests run automatically in continuous integration
- [ ] **Comprehensive Coverage**: All two-tensor loading paths validated
- [ ] **Data Validation**: Tensor content and structure verification
- [ ] **Error Handling**: Comprehensive error condition testing
- [ ] **Performance Monitoring**: Loading performance regression detection
- [ ] **Self-Contained**: No external file dependencies

## Related Issues

- #XXX: GGUF format validation comprehensive framework
- #XXX: Model loading error handling standardization
- #XXX: CI/CD test coverage improvement
- #XXX: Memory layout optimization for tensor loading

## Implementation Notes

This comprehensive test implementation eliminates the external dependency on environment variables while providing thorough validation of the two-tensor loading functionality. The synthetic GGUF generation enables reliable CI integration and comprehensive error condition testing.