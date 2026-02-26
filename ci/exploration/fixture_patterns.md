# BitNet-rs Test Fixture Patterns and Architecture

## Executive Summary

This document catalogs the current test fixture patterns, GGUF loading mechanisms, and helper utilities in the BitNet-rs codebase. It provides guidance for creating the three new test fixtures (`qk256_4x256.gguf`, `bitnet32_2x64.gguf`, `qk256_3x300.gguf`) with best practices derived from existing implementations.

## Table of Contents

1. [Current Fixture Locations](#current-fixture-locations)
2. [GGUF Loading and Helper Functions](#gguf-loading-and-helper-functions)
3. [Test Fixture Patterns](#test-fixture-patterns)
4. [GGUF Generation Utilities](#gguf-generation-utilities)
5. [Best Practices](#best-practices)
6. [Recommended Approach for New Fixtures](#recommended-approach-for-new-fixtures)

---

## Current Fixture Locations

### Primary Test Directories

```
crates/bitnet-models/tests/          # Integration tests for model loading
├── qk256_integration.rs             # QK256 kernel integration tests
├── qk256_loader_tests.rs            # QK256 loader and dispatch tests
├── i2s_flavor_detection.rs          # I2S flavor detection tests
├── qk256_detection.rs               # QK256 detection logic tests
├── qk256_detection_storage_tests.rs # QK256 storage convention tests
├── comprehensive_tests.rs           # End-to-end model loading tests
└── [50+ other integration test files]

tests-new/fixtures/                  # Comprehensive test fixture framework
├── fixtures/
│   ├── gguf/                        # GGUF fixture directory
│   │   ├── valid/                   # Valid GGUF files
│   │   └── invalid/                 # Invalid/malformed GGUF files
│   ├── gguf_generator.rs            # GGUF generation utilities
│   ├── fixture_loader.rs            # Centralized fixture loading
│   ├── device_aware_fixtures.rs     # Device-aware test fixtures
│   └── [15+ fixture support files]
└── integration/
    ├── models/mod.rs                # Model fixture definitions
    └── [integration test files]

crates/bitnet-models/src/
├── formats/
│   └── gguf/
│       └── tests.rs                 # GGUF parsing tests with helpers
└── loader/
    └── tests.rs                     # Model loader tests
```

### Fixture Organization Strategy

**Three-Layer Fixture Architecture:**

1. **Static/Inline Fixtures**: Small test data defined directly in test files
   - Examples: `create_qk256_tensor()` in `qk256_integration.rs`
   - Best for: Unit tests, small synthetic data
   - Location: Test file itself

2. **Generated Fixtures**: Dynamically created at runtime or test setup
   - Examples: `GgufFixtureGenerator` in `tests-new/fixtures/fixtures/gguf_generator.rs`
   - Best for: Flexible, reproducible test data
   - Location: Temporary directory or workspace cache

3. **File-Based Fixtures**: Pre-generated GGUF files for integration tests
   - Examples: Files in `tests-new/fixtures/fixtures/gguf/valid/`
   - Best for: End-to-end testing, cross-validation
   - Location: `tests-new/fixtures/fixtures/gguf/` or equivalent

---

## GGUF Loading and Helper Functions

### Helper Functions in Test Files

#### 1. **qk256_integration.rs** - QK256 Tensor Creation

**Location**: `/crates/bitnet-models/tests/qk256_integration.rs:34-50`

```rust
/// Helper to create QK256 packed tensor from code pattern
fn create_qk256_tensor(rows: usize, cols: usize, code: u8) -> anyhow::Result<CandleTensor> {
    assert!(code < 4, "QK256 code must be 0..=3");

    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;

    // Pack code into byte pattern
    let packed_byte = code | (code << 2) | (code << 4) | (code << 6);
    let qs_bytes = vec![packed_byte; rows * row_stride_bytes];

    // Create U8 tensor [rows, row_stride_bytes]
    let tensor = CandleTensor::from_vec(qs_bytes, &[rows, row_stride_bytes], &CDevice::Cpu)?
        .to_dtype(DType::U8)?;

    Ok(tensor)
}

/// Helper to decode QK256 code to float
#[inline]
fn code_to_float(code: u8) -> f32 {
    match code {
        0 => -2.0,
        1 => -1.0,
        2 => 1.0,
        3 => 2.0,
        _ => panic!("Invalid QK256 code: {}", code),
    }
}
```

**Key Characteristics**:
- Direct Candle tensor creation from raw bytes
- QK256-specific: uses `QK256_BLOCK` (256), `QK256_PACKED_BYTES` (64)
- Deterministic code-to-float mapping (0→-2.0, 1→-1.0, 2→+1.0, 3→+2.0)
- Returns `anyhow::Result<CandleTensor>` for seamless integration

**Usage Pattern**:
```rust
#[test]
fn test_qk256_single_block() {
    let qk256_tensor = create_qk256_tensor(2, 256, 2).expect("tensor creation failed");
    // Extract bytes and test with gemv_qk256()
}
```

#### 2. **GGUF tests.rs** - GGUF Bytes Construction

**Location**: `/crates/bitnet-models/src/formats/gguf/tests.rs:16-93`

```rust
/// Helper to build valid GGUF bytes for testing
fn build_gguf_bytes(metadata: Vec<(&str, GgufValue)>) -> Vec<u8> {
    let mut data = Vec::<u8>::new();
    const GGUF_VERSION: u32 = 2;
    const ALIGN: usize = 32;

    // Header (v2 shape)
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    let n_tensors = 0u64;
    let n_kv = metadata.len() as u64;
    data.extend_from_slice(&n_tensors.to_le_bytes());
    data.extend_from_slice(&n_kv.to_le_bytes());

    // KV section
    for (key, value) in metadata {
        let kb = key.as_bytes();
        data.extend_from_slice(&(kb.len() as u64).to_le_bytes());
        data.extend_from_slice(kb);
        write_gguf_value(&mut data, value);
    }

    // Align to 32 bytes
    let pad = (ALIGN - (data.len() % ALIGN)) % ALIGN;
    data.resize(data.len() + pad, 0);

    data
}

/// Helper to write a GGUF value to a byte vector
fn write_gguf_value(data: &mut Vec<u8>, value: GgufValue) {
    match value {
        GgufValue::U8(v) => {
            data.extend_from_slice(&0u32.to_le_bytes()); // Type 0
            data.push(v);
        }
        GgufValue::String(ref s) => {
            data.extend_from_slice(&8u32.to_le_bytes()); // Type 8
            data.extend_from_slice(&(s.len() as u64).to_le_bytes());
            data.extend_from_slice(s.as_bytes());
        }
        // ... other type handlers
    }
}
```

**Key Characteristics**:
- Direct binary writing with `byteorder::WriteBytesExt`
- GGUF v2 header format (magic, version, tensor count, KV count)
- Metadata KV pairs with type-specific serialization
- 32-byte alignment for data sections
- Returns raw bytes for testing parsers

#### 3. **gguf_generator.rs** - Full GGUF File Generation

**Location**: `/tests-new/fixtures/fixtures/gguf_generator.rs:88-196`

```rust
pub struct GgufFixtureGenerator {
    output_dir: PathBuf,
    seed: u64,
}

impl GgufFixtureGenerator {
    pub fn new(output_dir: PathBuf, seed: u64) -> Self {
        Self { output_dir, seed }
    }

    /// Generate a complete GGUF fixture
    pub fn generate_fixture(&self, config: &GgufFixtureConfig) -> Result<GgufFixture> {
        let file_path = self.output_dir.join(format!("{}.gguf", config.name));
        
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut writer = BufWriter::new(File::create(&file_path)?);
        let mut tensors = Vec::new();
        let mut current_offset = 0u64;

        // Write header
        current_offset += self.write_gguf_header(&mut writer, config)?;

        // Write metadata
        current_offset += self.write_metadata(&mut writer, config)?;

        // Generate and write tensors
        let tensor_configs =
            self.get_tensor_configs_for_model(&config.model_type, &config.quantization_type);

        for tensor_config in tensor_configs {
            let tensor_info =
                self.write_tensor(&mut writer, &tensor_config, config, current_offset)?;
            current_offset += tensor_info.size;
            tensors.push(tensor_info);
        }

        writer.flush()?;
        drop(writer);

        let file_size = std::fs::metadata(&file_path)?.len();
        let checksum = self.calculate_checksum(&file_path)?;

        Ok(GgufFixture { path: file_path, config: config.clone(), tensors, file_size, checksum })
    }
}
```

**Key Characteristics**:
- Full GGUF file writing with proper structure
- Configuration-driven tensor generation
- Supports multiple quantization types (I2S, TL1, TL2, IQ2S, FP32)
- File integrity tracking (checksum, file size)
- Deterministic data generation (seeded RNG)

### Current Test Helper Utilities

#### EnvGuard Pattern (Workspace Tests)

**Location**: `/tests/support/env_guard.rs`

```rust
mod env_guard {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/support/env_guard.rs"));
}
use env_guard::EnvGuard;
```

Used in GGUF tests to safely manage test environment variables without affecting other tests.

#### NamedTempFile Pattern (Standard in Tests)

```rust
use tempfile::NamedTempFile;

#[test]
fn test_model_loading() {
    let mut temp_file = NamedTempFile::with_suffix(".gguf").unwrap();
    temp_file.write_all(b"GGUF").unwrap();
    temp_file.flush().unwrap();

    // Test with temp file
    let result = loader.load(temp_file.path());
    assert!(result.is_ok());
}
```

---

## Test Fixture Patterns

### Pattern 1: Inline Synthetic Fixtures (Unit Tests)

**Used for**: Small, focused unit tests
**Example**: `qk256_integration.rs` tests

```rust
#[test]
fn test_qk256_single_block_predictable_output() {
    // Create in-memory test data
    let rows = 2;
    let cols = 256;
    let code = 2u8; // → +1.0

    let qk256_tensor = create_qk256_tensor(rows, cols, code)
        .expect("tensor creation failed");

    // Create input
    let input_data: Vec<f32> = (0..cols)
        .map(|i| (i + 1) as f32 * 0.01)
        .collect();
    let expected_sum: f32 = input_data.iter().sum();

    // Extract and test
    let bytes_2d = qk256_tensor.to_vec2::<u8>()
        .expect("to_vec2 failed");
    let flat_bytes: Vec<u8> = bytes_2d.iter()
        .flatten()
        .copied()
        .collect();

    let mut output = vec![0.0f32; rows];
    gemv_qk256(&flat_bytes, &input_data, &mut output, rows, cols, QK256_PACKED_BYTES)
        .expect("gemv_qk256 failed");

    // Verify
    for (i, &val) in output.iter().enumerate() {
        assert!(
            (val - expected_sum).abs() < 1e-3,
            "Row {}: expected {}, got {}",
            i,
            expected_sum,
            val
        );
    }
}
```

**Advantages**:
- Fast (no I/O)
- Reproducible (deterministic)
- Self-contained
- Easy to debug

**Disadvantages**:
- Limited scale
- Can't test file I/O
- Code duplication across tests

### Pattern 2: Fixture Builder Functions (Helper Methods)

**Used for**: Reusable test setup
**Example**: GGUF helper functions

```rust
fn build_gguf_bytes(metadata: Vec<(&str, GgufValue)>) -> Vec<u8> {
    // Common GGUF construction logic
}

#[test]
fn test_gguf_header_parsing() {
    let data_v2 = build_gguf_bytes(vec![
        ("model.type", GgufValue::String("test".to_string())),
    ]);

    // Parse and validate
    let result = GgufParser::parse(&data_v2);
    assert!(result.is_ok());
}
```

**Advantages**:
- DRY (Don't Repeat Yourself)
- Consistent test setup
- Easy to maintain

**Disadvantages**:
- Adds complexity to test files
- May not scale to complex fixtures

### Pattern 3: File-Based Fixtures (Integration Tests)

**Used for**: End-to-end testing, cross-validation
**Example**: `tests-new/fixtures/fixtures/gguf/`

```rust
#[test]
fn test_load_valid_gguf() {
    let fixture_path = "tests-new/fixtures/fixtures/gguf/valid/model.gguf";
    let result = loader.load(fixture_path);
    assert!(result.is_ok());
}
```

**Advantages**:
- Realistic test scenarios
- Can test file I/O edge cases
- Reusable across multiple tests

**Disadvantages**:
- Repository size growth
- Binary diffs in version control
- Difficult to version/track changes

### Pattern 4: Generated Fixtures (Dynamic Generation)

**Used for**: Large test sets, parametric testing
**Example**: `GgufFixtureGenerator`

```rust
#[test]
fn test_model_generation() -> anyhow::Result<()> {
    let temp_dir = tempfile::TempDir::new()?;
    let config = GgufFixtureConfig {
        name: "test_model".to_string(),
        model_type: ModelType::Minimal,
        quantization_type: QuantizationType::I2S,
        vocab_size: 1000,
        hidden_size: 128,
        num_layers: 2,
        tensor_alignment: 32,
        generate_invalid: false,
        seed: 42,
    };

    let fixture = GgufFixtureGenerator::new(temp_dir.path().to_path_buf(), 42)
        .generate_fixture(&config)?;

    assert!(fixture.path.exists());
    assert!(fixture.file_size > 0);
    assert!(!fixture.tensors.is_empty());

    Ok(())
}
```

**Advantages**:
- No repository bloat
- Deterministic (seeded)
- Parametric flexibility
- Easy to scale

**Disadvantages**:
- Generation overhead in tests
- Requires fixture code maintenance

---

## GGUF Generation Utilities

### GgufFixtureGenerator API

**Location**: `/tests-new/fixtures/fixtures/gguf_generator.rs`

#### Configuration Structures

```rust
pub struct GgufFixtureConfig {
    pub name: String,
    pub model_type: ModelType,                    // BitNet158_1B, BitNetB1_58_2B, Minimal
    pub quantization_type: QuantizationType,      // I2S, TL1, TL2, IQ2S, FP32
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub tensor_alignment: u64,                    // 32 (default) or 64
    pub generate_invalid: bool,                   // For error testing
    pub seed: u64,                                // For deterministic data
}

pub enum ModelType {
    BitNet158_1B,
    BitNet158_3B,
    BitNetB1_58_2B,
    Minimal,  // ~10 tensors, fast generation
}

pub enum QuantizationType {
    I2S,   // 2-bit signed (BitNet32F16)
    TL1,   // Table lookup, 4-bit indices
    TL2,   // Table lookup, 8-bit indices
    IQ2S,  // GGML-compatible 2-bit
    FP32,  // Unquantized baseline
}
```

#### Generation Methods

```rust
impl GgufFixtureGenerator {
    // Constructor
    pub fn new(output_dir: PathBuf, seed: u64) -> Self

    // Generate complete fixture set
    pub fn generate_bitnet_fixture_set(&self) -> Result<Vec<GgufFixture>>

    // Generate single fixture
    pub fn generate_fixture(&self, config: &GgufFixtureConfig) -> Result<GgufFixture>

    // Private implementation methods
    fn write_gguf_header(&self, writer: &mut BufWriter<File>, config: &GgufFixtureConfig) -> Result<u64>
    fn write_metadata(&self, writer: &mut BufWriter<File>, config: &GgufFixtureConfig) -> Result<u64>
    fn write_tensor(&self, writer: &mut BufWriter<File>, config: &TensorConfig, fixture_config: &GgufFixtureConfig, offset: u64) -> Result<TensorInfo>
    fn get_tensor_configs_for_model(&self, model_type: &ModelType, quant_type: &QuantizationType) -> Vec<TensorConfig>
}
```

#### Output Structure

```rust
pub struct GgufFixture {
    pub path: PathBuf,                // File system location
    pub config: GgufFixtureConfig,    // Generation configuration
    pub tensors: Vec<TensorInfo>,     // Tensor metadata
    pub file_size: u64,               // File size in bytes
    pub checksum: String,             // SHA256 or similar
}

pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub data_type: DataType,
    pub offset: u64,
    pub size: u64,
    pub quantized: bool,
}
```

### Quantization Type Specifics

#### I2S (BitNet32F16)

**Format**: 2-bit signed quantization with inline F16 scales
- Block size: 32 elements
- Data per block: 8 bytes (2 bits × 32 elements, packed)
- Scale per block: 2 bytes (F16)
- Total per block: 10 bytes

**Code Mapping**:
- `0` → -2.0
- `1` → -1.0
- `2` → +1.0
- `3` → +2.0

#### QK256 (GGML I2_S)

**Format**: 2-bit signed quantization without inline scales
- Block size: 256 elements
- Data per block: 64 bytes (2 bits × 256 elements, packed)
- Scale per block: None (scale passed separately)
- Total per block: 64 bytes

**Tensor Naming Convention**:
- Original: `layers.0.attention.q_proj.weight`
- Stored as: `layers.0.attention.q_proj.weight.qk256_qs`

---

## Best Practices

### 1. **Test Data Determinism**

Always use seeded RNGs for reproducible test data:

```rust
fn generate_f32_value(&self, index: usize, seed: u64) -> f32 {
    let mut state = seed.wrapping_add(index as u64);
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;

    // Convert to -1.0 to 1.0 range
    ((state as f32) / (u64::MAX as f32)) * 2.0 - 1.0
}
```

### 2. **Fixture Organization**

Create a clear directory structure:

```
tests/fixtures/
├── gguf/
│   ├── valid/
│   │   ├── bitnet32_2x64.gguf        # All file sizes < 1MB
│   │   ├── qk256_3x300.gguf
│   │   └── qk256_4x256.gguf
│   └── invalid/                       # For error testing
├── configs/                           # JSON/YAML metadata
└── README.md                          # Fixture documentation
```

### 3. **Metadata Documentation**

For each fixture, document:

```toml
# fixtures/gguf/README.md

## qk256_4x256.gguf
- **Purpose**: QK256 single-block tensor with 4 rows × 256 cols
- **Quantization**: QK256 (GGML I2_S), 2-bit signed
- **Tensor Count**: 1 (just weight matrix)
- **File Size**: ~512 bytes
- **Generated**: Deterministic (seed=42)
- **Test Coverage**: Single-block QK256 kernels, dimension validation
```

### 4. **Error Handling**

Test both success and error paths:

```rust
#[test]
fn test_valid_gguf() {
    // Happy path
}

#[test]
fn test_invalid_header() {
    // Error handling
}

#[test]
fn test_truncated_file() {
    // Boundary conditions
}
```

### 5. **Fixture Cleanup**

Use temporary directories for generated fixtures:

```rust
#[test]
fn test_with_generated_fixture() -> anyhow::Result<()> {
    let temp_dir = tempfile::TempDir::new()?;
    
    let config = GgufFixtureConfig { /* ... */ };
    let fixture = GgufFixtureGenerator::new(
        temp_dir.path().to_path_buf(),
        42
    ).generate_fixture(&config)?;
    
    // Test with fixture
    // temp_dir automatically cleaned up when dropped
    Ok(())
}
```

### 6. **Parametric Testing**

Use test matrices for comprehensive coverage:

```rust
#[test]
fn test_qk256_various_dimensions() {
    let test_cases = [
        (1, 256, "single_row_single_block"),
        (4, 256, "multi_row_single_block"),
        (4, 300, "multi_row_with_tail"),
        (256, 512, "large_matrix"),
    ];

    for (rows, cols, desc) in &test_cases {
        let fixture = create_qk256_tensor(*rows, *cols, 2)
            .expect(&format!("Failed for {}", desc));
        // Test fixture
    }
}
```

---

## Recommended Approach for New Fixtures

### Strategy: Hybrid Approach

**Combine three methods for optimal coverage**:

1. **Generated Fixtures** (Primary)
   - Use `GgufFixtureGenerator` for on-demand creation
   - Deterministic (seeded) for reproducibility
   - No repository size impact
   - Perfect for CI/CD

2. **Inline Helpers** (Secondary - for unit tests)
   - Create small tensor helpers in test files
   - For < 1KB of data
   - Close to test code (easier to understand)

3. **Pre-generated Files** (Optional - for large integration tests)
   - Only if file-specific features need testing (e.g., mmap edge cases)
   - Store in `tests/fixtures/gguf/valid/` subdirectory
   - Add README with generation instructions

### Implementation Plan for the 3 New Fixtures

#### Fixture 1: qk256_4x256.gguf

**Specification**:
- Quantization: QK256 (GGML I2_S)
- Tensor: 4 rows × 256 cols (single 256-element block)
- Purpose: Single-block QK256 kernel validation
- Code Pattern: Uniform codes (0-3) for mathematical validation

**Generation Code**:

```rust
#[test]
fn generate_qk256_4x256_fixture() -> anyhow::Result<()> {
    let temp_dir = tempfile::TempDir::new()?;
    
    let config = GgufFixtureConfig {
        name: "qk256_4x256".to_string(),
        model_type: ModelType::Minimal,
        quantization_type: QuantizationType::I2S,  // Will be stored as QK256
        vocab_size: 256,
        hidden_size: 256,
        num_layers: 1,
        tensor_alignment: 32,
        generate_invalid: false,
        seed: 42,
    };
    
    let fixture = GgufFixtureGenerator::new(
        temp_dir.path().to_path_buf(),
        config.seed
    ).generate_fixture(&config)?;
    
    assert_eq!(fixture.tensors[0].shape, vec![4, 256]);
    assert!(fixture.path.exists());
    
    Ok(())
}
```

**Test Coverage**:
- Single block dimension edge case
- QK256 kernel dispatch verification
- Output dimension validation

#### Fixture 2: bitnet32_2x64.gguf

**Specification**:
- Quantization: BitNet32F16 (I2S with inline F16 scales)
- Tensor: 2 rows × 64 cols (2 blocks of 32)
- Purpose: BitNet32F16 flavor detection and kernel testing
- Format: Blocks of [8B packed data + 2B F16 scale] = 10 bytes per block

**Generation Code**:

```rust
#[test]
fn generate_bitnet32_2x64_fixture() -> anyhow::Result<()> {
    let temp_dir = tempfile::TempDir::new()?;
    
    let config = GgufFixtureConfig {
        name: "bitnet32_2x64".to_string(),
        model_type: ModelType::Minimal,
        quantization_type: QuantizationType::I2S,  // BitNet32F16 variant
        vocab_size: 64,
        hidden_size: 64,
        num_layers: 1,
        tensor_alignment: 32,
        generate_invalid: false,
        seed: 43,
    };
    
    let fixture = GgufFixtureGenerator::new(
        temp_dir.path().to_path_buf(),
        config.seed
    ).generate_fixture(&config)?;
    
    assert_eq!(fixture.tensors[0].shape, vec![2, 64]);
    // Expected size: 2 rows × 2 blocks × 10 bytes = 40 bytes
    assert_eq!(fixture.tensors[0].size, 40);
    
    Ok(())
}
```

**Test Coverage**:
- BitNet32F16 flavor detection (size-based)
- Multi-block dimension handling
- Inline scale format validation

#### Fixture 3: qk256_3x300.gguf

**Specification**:
- Quantization: QK256 (GGML I2_S)
- Tensor: 3 rows × 300 cols (1.17 blocks, with tail)
- Purpose: Multi-block QK256 with tail handling
- Blocks: 2 blocks of 256 + 44-element tail
- Expected storage: 3 rows × 2 blocks × 64 bytes = 384 bytes

**Generation Code**:

```rust
#[test]
fn generate_qk256_3x300_fixture() -> anyhow::Result<()> {
    let temp_dir = tempfile::TempDir::new()?;
    
    let config = GgufFixtureConfig {
        name: "qk256_3x300".to_string(),
        model_type: ModelType::Minimal,
        quantization_type: QuantizationType::I2S,  // Will be stored as QK256
        vocab_size: 300,
        hidden_size: 300,
        num_layers: 1,
        tensor_alignment: 32,
        generate_invalid: false,
        seed: 44,
    };
    
    let fixture = GgufFixtureGenerator::new(
        temp_dir.path().to_path_buf(),
        config.seed
    ).generate_fixture(&config)?;
    
    assert_eq!(fixture.tensors[0].shape, vec![3, 300]);
    // Expected size: 3 rows × 2 blocks × 64 bytes = 384 bytes (tail packed)
    assert_eq!(fixture.tensors[0].size, 384);
    
    Ok(())
}
```

**Test Coverage**:
- Multi-block QK256 with tail element handling
- QK256 dimension calculation (ceil division)
- Correct block packing for non-aligned column counts

### Integration with Existing Test Framework

```rust
// tests-new/fixtures/fixtures/gguf_generator.rs - Add to fixture set

impl GgufFixtureGenerator {
    /// Generate fixtures for QK256 edge cases
    pub fn generate_qk256_edge_case_fixtures(&self) -> Result<Vec<GgufFixture>> {
        vec![
            self.generate_fixture(&GgufFixtureConfig {
                name: "qk256_4x256".to_string(),
                model_type: ModelType::Minimal,
                quantization_type: QuantizationType::I2S,
                vocab_size: 256,
                hidden_size: 256,
                num_layers: 1,
                tensor_alignment: 32,
                generate_invalid: false,
                seed: 42,
            })?,
            self.generate_fixture(&GgufFixtureConfig {
                name: "qk256_3x300".to_string(),
                model_type: ModelType::Minimal,
                quantization_type: QuantizationType::I2S,
                vocab_size: 300,
                hidden_size: 300,
                num_layers: 1,
                tensor_alignment: 32,
                generate_invalid: false,
                seed: 44,
            })?,
        ]
    }

    /// Generate fixtures for BitNet32F16 edge cases
    pub fn generate_bitnet32_edge_case_fixtures(&self) -> Result<Vec<GgufFixture>> {
        vec![
            self.generate_fixture(&GgufFixtureConfig {
                name: "bitnet32_2x64".to_string(),
                model_type: ModelType::Minimal,
                quantization_type: QuantizationType::I2S,
                vocab_size: 64,
                hidden_size: 64,
                num_layers: 1,
                tensor_alignment: 32,
                generate_invalid: false,
                seed: 43,
            })?,
        ]
    }
}
```

### Test File Integration

```rust
// crates/bitnet-models/tests/qk256_edge_case_fixtures.rs

use bitnet_models::quant::i2s_qk256::{
    I2SQk256NoScale, QK256_BLOCK, QK256_PACKED_BYTES, gemv_qk256,
};
use tempfile::TempDir;

#[test]
fn test_qk256_4x256_fixture() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let config = /* ... */;
    let fixture = generate_fixture(&config)?;
    
    // Load and test
    let tensor_bytes = std::fs::read(&fixture.path)?;
    
    // Validate dimensions
    assert_eq!(fixture.tensors[0].shape, vec![4, 256]);
    
    // Test with gemv_qk256
    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 4];
    // ... gemv_qk256 test
    
    Ok(())
}

#[test]
fn test_bitnet32_2x64_fixture() -> anyhow::Result<()> {
    // Similar structure for BitNet32F16
    Ok(())
}

#[test]
fn test_qk256_3x300_fixture() -> anyhow::Result<()> {
    // Test multi-block with tail
    Ok(())
}
```

---

## Summary Table: Fixture Patterns

| Pattern | Size | Speed | Reproducibility | Maintenance | Use Case |
|---------|------|-------|-----------------|-------------|----------|
| **Inline Synthetic** | < 1KB | Very Fast | High | Low | Unit tests |
| **Helper Functions** | < 10KB | Fast | High | Medium | Test setup |
| **Generated (gguf_generator)** | 1MB | Medium | High | Low | Integration tests |
| **File-Based** | > 1MB | Fast | Medium | High | Full end-to-end |

**Recommended for 3 new fixtures**: **Generated approach** using `GgufFixtureGenerator`
- No repository bloat
- Deterministic and reproducible
- Easy to parametrize and scale
- Perfect for CI/CD pipelines

---

## References

- GGUF Format Spec: `/crates/bitnet-models/src/formats/gguf/`
- QK256 Tests: `/crates/bitnet-models/tests/qk256_integration.rs`
- Fixture Generator: `/tests-new/fixtures/fixtures/gguf_generator.rs`
- Test Patterns: `/crates/bitnet-models/tests/`
