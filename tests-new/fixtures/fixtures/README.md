# BitNet.rs Test Fixtures

Comprehensive test fixtures for GGUF weight loading validation (Issue #159).

## Overview

This directory contains realistic test data for BitNet.rs neural network components:

- **GGUF Model Fixtures**: Valid and invalid GGUF files with realistic tensor structures
- **Quantization Test Data**: I2_S, TL1, TL2 test vectors with known accuracy properties
- **Cross-Validation Data**: Reference outputs compatible with `cargo run -p xtask -- crossval`
- **Integration Fixtures**: Multi-crate test scenarios spanning bitnet-models + bitnet-quantization
- **Error Handling Data**: Corrupted files and edge cases for robustness testing

## Structure

```
fixtures/
├── gguf/
│   ├── valid/                  # Valid GGUF files for positive testing
│   │   ├── minimal_bitnet_i2s.gguf    # Full-size I2_S model (39 tensors)
│   │   └── small_bitnet_test.gguf     # Small model for fast testing (21 tensors)
│   ├── invalid/                # Corrupted/invalid files for error testing
│   │   ├── invalid_magic.gguf         # Invalid GGUF magic number
│   │   ├── invalid_version.gguf       # Unsupported version
│   │   ├── truncated_header.gguf      # Truncated header
│   │   ├── invalid_tensor_count.gguf  # Invalid tensor count
│   │   ├── misaligned_tensors.gguf    # Misaligned tensor data
│   │   ├── incomplete.gguf            # Incomplete file
│   │   ├── invalid_metadata.gguf      # Corrupted metadata
│   │   ├── zero_byte.gguf            # Empty file
│   │   ├── random_data.gguf          # Random binary data
│   │   └── missing_tensors.gguf      # Claims tensors but incomplete
│   └── quantized/              # Different quantization formats
├── tensors/
│   ├── reference/              # FP32 reference data
│   ├── quantized/              # Quantized test tensors
│   │   ├── quantization_test_vectors.json  # All test vectors (36 vectors)
│   │   ├── i2s_test_vectors.json          # I2_S specific vectors (12 vectors)
│   │   ├── tl1_test_vectors.json          # TL1 specific vectors (12 vectors)
│   │   ├── tl2_test_vectors.json          # TL2 specific vectors (12 vectors)
│   │   └── binary/                        # Binary test data for efficient loading
│   └── crossval/               # Cross-validation test vectors
│       ├── crossval_references.json       # All references (15 references)
│       ├── crossval_i2s.json             # I2_S cross-validation data
│       ├── crossval_tl1.json             # TL1 cross-validation data
│       ├── crossval_tl2.json             # TL2 cross-validation data
│       ├── xtask_crossval_config.json    # xtask-compatible configuration
│       └── binary/                       # Binary reference data
└── integration/
    ├── models/                 # Multi-crate integration fixtures
    ├── devices/                # CPU/GPU specific test data
    └── performance/            # Memory efficiency test scenarios
```

## Usage

### Basic Fixture Loading

```rust
use anyhow::Result;
use bitnet_common::Device;

#[cfg(feature = "cpu")]
#[test]
fn test_with_valid_gguf_fixture() -> Result<()> {
    // Get available test fixtures
    let valid_files = bitnet_test_fixtures::get_valid_gguf_files();
    let small_model = valid_files.iter()
        .find(|p| p.file_name().unwrap().to_str().unwrap().contains("small"))
        .unwrap();

    // Load model using fixture
    let (config, tensors) = bitnet_models::gguf_simple::load_gguf(small_model, Device::Cpu)?;

    // Validate basic expectations
    assert_eq!(config.model.vocab_size, 1000);
    assert_eq!(config.model.hidden_size, 256);
    assert_eq!(config.model.num_layers, 2);

    // Check expected tensors are present
    assert!(tensors.contains_key("token_embd.weight"));
    assert!(tensors.contains_key("output.weight"));

    Ok(())
}
```

### Error Handling Testing

```rust
#[cfg(feature = "cpu")]
#[test]
fn test_invalid_gguf_error_handling() -> Result<()> {
    let invalid_files = bitnet_test_fixtures::get_invalid_gguf_files();

    for invalid_file in invalid_files {
        let result = bitnet_models::gguf_simple::load_gguf(&invalid_file, Device::Cpu);

        // Should fail gracefully with descriptive error
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("GGUF") || error_msg.contains("parsing") || error_msg.contains("invalid"));
    }

    Ok(())
}
```

### Cross-Validation Testing

```rust
#[cfg(feature = "crossval")]
#[test]
fn test_crossval_reference_data() -> Result<()> {
    use std::env;

    // Set environment variables for cross-validation
    env::set_var("BITNET_CROSSVAL_CONFIG", "tests/fixtures/tensors/crossval/xtask_crossval_config.json");
    env::set_var("BITNET_DETERMINISTIC", "1");
    env::set_var("BITNET_SEED", "42");

    // Run cross-validation (normally done via xtask)
    // cargo run -p xtask -- crossval

    Ok(())
}
```

## Fixture Generation

To regenerate test fixtures:

```bash
# Generate GGUF test files
cd tests/fixtures/gguf/valid
python3 minimal_bitnet_i2s.py

# Generate corrupted GGUF files
cd tests/fixtures/gguf/invalid
python3 generate_corrupted_gguf.py

# Generate quantization test vectors
cd tests/fixtures/tensors/quantized
python3 generate_quantization_test_data.py

# Generate cross-validation references
cd tests/fixtures/tensors/crossval
python3 generate_crossval_data.py
```

## Test Data Statistics

### GGUF Model Fixtures

- **minimal_bitnet_i2s.gguf**: 39 tensors, 15 metadata entries, ~2B parameters
- **small_bitnet_test.gguf**: 21 tensors, 15 metadata entries, ~256M parameters

### Quantization Test Vectors

- **Total Test Vectors**: 36 (12 per quantization type)
- **I2_S Accuracy**: Mean 43.78%, Min 1.99%, Max 100%
- **TL1 Accuracy**: Mean 26.00%, Min 0%, Max 100%
- **TL2 Accuracy**: Mean 90.34%, Min 0%, Max 100%
- **Vectors ≥99% Accuracy**: 18/36 (50%)

### Cross-Validation References

- **Total References**: 15 test cases
- **I2_S Tests**: 4 cases (small matrix, attention Q, FFN up/down)
- **TL1 Tests**: 3 cases (small test, medium matrix, FFN gate)
- **TL2 Tests**: 3 cases (attention KV, embedding, output projection)
- **E2E Inference**: 1 case (mixed quantization)
- **Performance**: 4 benchmark cases

## Integration with Test Scaffolding

These fixtures integrate with the existing test scaffolding in:

- `crates/bitnet-models/tests/gguf_weight_loading_tests.rs`
- `crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs`
- `crates/bitnet-models/tests/gguf_weight_loading_feature_matrix_tests.rs`
- `crates/bitnet-models/tests/gguf_weight_loading_integration_tests.rs`

## Environment Variables

- `BITNET_DETERMINISTIC=1`: Use deterministic test data (seed=42)
- `BITNET_FIXTURES_DIR`: Override fixtures directory location
- `BITNET_CROSSVAL_CONFIG`: Path to cross-validation configuration
- `BITNET_SEED=42`: Set specific seed for reproducible tests

## Feature Flags

Fixtures support BitNet.rs feature-gated compilation:

```bash
# CPU-only testing
cargo test --no-default-features --features cpu

# GPU testing with fallback
cargo test --no-default-features --features gpu

# Cross-validation testing
cargo test --no-default-features --features crossval

# FFI bridge testing
cargo test --no-default-features --features ffi
```

## Performance Characteristics

### Loading Times (CPU baseline)

- Small model (21 tensors): ~50-100ms
- Full model (39 tensors): ~200-500ms
- Invalid files: <10ms (fast failure)

### Memory Usage

- Small model: ~16-32MB peak, ~16MB steady
- Full model: ~128-256MB peak, ~128MB steady
- Memory efficiency: 1.1-1.5x overhead factor

### Accuracy Thresholds

- I2_S quantization: ≥95% accuracy required for small models, ≥99% for full models
- TL1 quantization: ≥90% accuracy required
- TL2 quantization: ≥99% accuracy required
- Cross-validation: 1e-6 absolute, 1e-4 relative tolerance

## Validation

To validate all fixtures are present and correct:

```rust
#[test]
fn test_fixtures_validation() -> anyhow::Result<()> {
    bitnet_test_fixtures::validate_fixtures_available()?;
    Ok(())
}
```

## Troubleshooting

### Missing Fixtures

If fixtures are missing, regenerate using the Python scripts in each subdirectory.

### Permission Issues

Ensure the fixtures directory is readable:
```bash
chmod -R 644 tests/fixtures/
chmod 755 tests/fixtures/ tests/fixtures/*/
```

### Path Issues

Set explicit fixtures directory:
```bash
export BITNET_FIXTURES_DIR=/absolute/path/to/fixtures
```

### Memory Issues

Use smaller test fixtures for resource-constrained environments:
- Prefer `small_bitnet_test.gguf` over `minimal_bitnet_i2s.gguf`
- Set `RAYON_NUM_THREADS=1` to reduce parallelism
- Use `BITNET_STRICT_NO_FAKE_GPU=1` to avoid GPU allocation attempts
