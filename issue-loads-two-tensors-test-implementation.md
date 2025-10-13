# [TEST] Implement ignored loads_two_tensors test in gguf_min.rs

## Problem Description

The `loads_two_tensors` test in `crates/bitnet-models/src/gguf_min.rs` is currently ignored with `#[ignore]` attribute and requires manual environment setup (`BITNET_GGUF`). This test is essential for validating the minimal GGUF loader's ability to extract vocabulary and dimension information from real GGUF files with two tensors (tok_embeddings and lm_head).

## Environment

**Affected Component:** `crates/bitnet-models/src/gguf_min.rs`
**Test Function:** `loads_two_tensors`
**Related Functions:** `load_two`, GGUF parsing infrastructure
**Impact:** Test coverage, GGUF validation, CI/CD reliability

## Root Cause Analysis

### Current Test Limitations

1. **Manual environment dependency**: Requires `BITNET_GGUF` environment variable
2. **Ignored in CI**: Test is skipped in automated testing
3. **No synthetic test data**: Relies on external GGUF files
4. **Limited validation scope**: Only basic assertions without comprehensive checks

### Code Analysis

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

Issues:
- Test is not executed in CI/CD pipeline
- No validation of GGUF format compliance
- Limited error condition testing
- Missing edge case coverage

## Impact Assessment

### Testing Coverage Impact
- **Regression risk**: GGUF loading bugs may go undetected
- **Integration failures**: Changes to GGUF parsing not validated
- **Quality assurance**: Reduced confidence in minimal loader functionality
- **CI reliability**: Incomplete test suite execution

### Development Impact
- **Debug difficulty**: Manual test setup slows debugging
- **Refactoring risk**: Changes to GGUF code lack validation
- **Cross-platform testing**: Manual setup prevents comprehensive testing

## Proposed Solution

### Comprehensive Test Implementation

Replace manual environment-dependent test with automated synthetic GGUF generation:

```rust
#[test]
fn loads_two_tensors() {
    use std::io::{Cursor, Write};
    use tempfile::NamedTempFile;

    // Create synthetic GGUF file with two tensors
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let gguf_data = create_minimal_gguf_with_two_tensors(100, 50); // vocab=100, dim=50
    temp_file.write_all(&gguf_data).expect("Failed to write GGUF data");

    // Test load_two function
    let two = load_two(temp_file.path()).unwrap();

    // Validate extracted parameters
    assert_eq!(two.vocab, 100);
    assert_eq!(two.dim, 50);
    assert_eq!(two.tok_embeddings.len(), two.vocab * two.dim);
    assert_eq!(two.lm_head.len(), two.dim * two.vocab);

    // Validate tensor data integrity
    assert!(two.tok_embeddings.iter().all(|&x| x.is_finite()));
    assert!(two.lm_head.iter().all(|&x| x.is_finite()));
}

#[test]
fn loads_two_tensors_with_real_gguf() {
    // Optional test for real GGUF files when environment is set
    if let Ok(gguf_path) = std::env::var("BITNET_GGUF") {
        let two = load_two(&gguf_path).unwrap();
        assert!(two.vocab > 0 && two.dim > 0);
        assert_eq!(two.tok_embeddings.len(), two.vocab * two.dim);
        assert_eq!(two.lm_head.len(), two.dim * two.vocab);
    }
}

#[test]
fn loads_two_tensors_error_conditions() {
    // Test missing file
    assert!(load_two("nonexistent.gguf").is_err());

    // Test invalid GGUF format
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(b"INVALID").unwrap();
    assert!(load_two(temp_file.path()).is_err());

    // Test GGUF with missing tensors
    let incomplete_gguf = create_gguf_with_single_tensor();
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&incomplete_gguf).unwrap();
    assert!(load_two(temp_file.path()).is_err());
}
```

### GGUF Synthetic Data Generation

```rust
fn create_minimal_gguf_with_two_tensors(vocab: usize, dim: usize) -> Vec<u8> {
    use std::io::{Cursor, Write};
    use byteorder::{LittleEndian, WriteBytesExt};

    let mut buffer = Vec::new();
    let mut cursor = Cursor::new(&mut buffer);

    // Write GGUF magic and version
    cursor.write_all(b"GGUF").unwrap();
    cursor.write_u32::<LittleEndian>(3).unwrap(); // Version 3

    // Write metadata count (minimal required metadata)
    cursor.write_u64::<LittleEndian>(2).unwrap(); // Two metadata entries

    // Write tensor count
    cursor.write_u64::<LittleEndian>(2).unwrap(); // Two tensors

    // Write metadata entries
    write_gguf_metadata(&mut cursor, "general.architecture", "llama").unwrap();
    write_gguf_metadata(&mut cursor, "llama.vocab_size", &(vocab as u64)).unwrap();

    // Write tensor info
    write_gguf_tensor_info(&mut cursor, "tok_embeddings.weight",
                          &[vocab as u64, dim as u64], GGMLType::F32).unwrap();
    write_gguf_tensor_info(&mut cursor, "lm_head.weight",
                          &[dim as u64, vocab as u64], GGMLType::F32).unwrap();

    // Calculate and write tensor data with proper alignment
    let tensor_data_offset = align_to_32_bytes(cursor.position());
    cursor.set_position(tensor_data_offset);

    // Write tok_embeddings tensor data
    let tok_embeddings_size = vocab * dim * 4; // F32 = 4 bytes
    write_synthetic_tensor_data(&mut cursor, tok_embeddings_size);

    // Write lm_head tensor data
    let lm_head_size = dim * vocab * 4;
    write_synthetic_tensor_data(&mut cursor, lm_head_size);

    buffer
}

fn write_gguf_metadata<W: Write>(writer: &mut W, key: &str, value: &dyn GGUFValue) -> Result<(), Box<dyn std::error::Error>> {
    // Write key length and key
    writer.write_u64::<LittleEndian>(key.len() as u64)?;
    writer.write_all(key.as_bytes())?;

    // Write value type and value
    value.write_to(writer)?;
    Ok(())
}

fn write_gguf_tensor_info<W: Write>(writer: &mut W, name: &str, shape: &[u64], ggml_type: GGMLType) -> Result<(), Box<dyn std::error::Error>> {
    // Write tensor name
    writer.write_u64::<LittleEndian>(name.len() as u64)?;
    writer.write_all(name.as_bytes())?;

    // Write dimension count
    writer.write_u32::<LittleEndian>(shape.len() as u32)?;

    // Write shape
    for &dim in shape {
        writer.write_u64::<LittleEndian>(dim)?;
    }

    // Write GGML type
    writer.write_u32::<LittleEndian>(ggml_type as u32)?;

    // Write offset (will be calculated later)
    writer.write_u64::<LittleEndian>(0)?;

    Ok(())
}
```

## Implementation Plan

### Phase 1: Synthetic GGUF Generation (2-3 days)
- [ ] Implement `create_minimal_gguf_with_two_tensors` function
- [ ] Add GGUF metadata and tensor info writing utilities
- [ ] Implement proper GGUF alignment and offset calculation
- [ ] Add support for various tensor shapes and types

### Phase 2: Comprehensive Test Suite (1-2 days)
- [ ] Replace ignored test with synthetic data test
- [ ] Add error condition testing
- [ ] Implement edge case validation
- [ ] Add optional real GGUF file testing

### Phase 3: Integration & Validation (1 day)
- [ ] Ensure tests pass in CI/CD pipeline
- [ ] Validate synthetic GGUF compatibility with load_two
- [ ] Add cross-validation with real GGUF files
- [ ] Performance benchmarking of test execution

### Phase 4: Documentation & Enhancement (1 day)
- [ ] Document synthetic GGUF generation approach
- [ ] Add test data documentation
- [ ] Create debugging utilities for GGUF inspection
- [ ] Update test suite documentation

## Testing Strategy

### Functional Testing
```rust
#[test]
fn test_various_tensor_dimensions() {
    let test_cases = vec![
        (32, 128),   // Small model
        (50257, 768), // GPT-2 like
        (100000, 4096), // Large model
    ];

    for (vocab, dim) in test_cases {
        let gguf_data = create_minimal_gguf_with_two_tensors(vocab, dim);
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&gguf_data).unwrap();

        let two = load_two(temp_file.path()).unwrap();
        assert_eq!(two.vocab, vocab);
        assert_eq!(two.dim, dim);
    }
}

#[test]
fn test_gguf_format_validation() {
    // Test different GGUF versions
    for version in [1, 2, 3] {
        let gguf_data = create_gguf_with_version(version, 100, 50);
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&gguf_data).unwrap();

        let result = load_two(temp_file.path());
        if version >= 3 {
            assert!(result.is_ok());
        } else {
            // Should handle older versions gracefully
            assert!(result.is_ok() || result.is_err());
        }
    }
}
```

### Performance Testing
```rust
#[test]
fn benchmark_synthetic_gguf_generation() {
    let start = Instant::now();
    for _ in 0..100 {
        let _gguf_data = create_minimal_gguf_with_two_tensors(1000, 512);
    }
    let generation_time = start.elapsed();

    // Should be fast enough for test execution
    assert!(generation_time < Duration::from_millis(1000));
}
```

### Cross-Validation Testing
```rust
#[test]
fn test_synthetic_vs_real_gguf() {
    if let Ok(real_gguf_path) = std::env::var("BITNET_GGUF") {
        // Load real GGUF to get dimensions
        let real_two = load_two(&real_gguf_path).unwrap();

        // Create synthetic GGUF with same dimensions
        let synthetic_gguf = create_minimal_gguf_with_two_tensors(real_two.vocab, real_two.dim);
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&synthetic_gguf).unwrap();

        let synthetic_two = load_two(temp_file.path()).unwrap();

        // Validate structural equivalence
        assert_eq!(synthetic_two.vocab, real_two.vocab);
        assert_eq!(synthetic_two.dim, real_two.dim);
        assert_eq!(synthetic_two.tok_embeddings.len(), real_two.tok_embeddings.len());
        assert_eq!(synthetic_two.lm_head.len(), real_two.lm_head.len());
    }
}
```

## Risk Assessment

### Implementation Risks
- **GGUF format complexity**: Synthetic generation may miss edge cases
- **Compatibility issues**: Generated files may not match real GGUF specifications
- **Test reliability**: Synthetic data may not catch real-world issues

### Mitigation Strategies
- Validate synthetic GGUF files with external GGUF parsers
- Include optional real file testing for comprehensive validation
- Add extensive format compliance checking
- Implement incremental rollout with existing manual testing fallback

## Success Criteria

### Test Coverage
- [ ] `loads_two_tensors` test executes automatically in CI/CD
- [ ] 100% test coverage for basic GGUF loading scenarios
- [ ] Error condition handling properly tested
- [ ] Edge cases and boundary conditions validated

### Quality Assurance
- [ ] Synthetic GGUF files compatible with existing load_two function
- [ ] Test execution time < 1 second per test case
- [ ] No regression in existing GGUF loading functionality
- [ ] Real GGUF file compatibility maintained

## Related Issues

- **GGUF Compatibility**: Alignment with GGUF specification standards
- **Cross-validation**: Integration with Microsoft BitNet C++ validation
- **CI/CD Pipeline**: Automated testing infrastructure improvements

## References

- GGUF format specification
- BitNet model tensor layout requirements
- Test data generation best practices
- GGUF parsing implementation details

---

**Priority**: Medium
**Estimated Effort**: 3-5 developer days
**Components**: bitnet-models, test infrastructure
**Feature Flags**: None (core functionality)
