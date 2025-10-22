# Test Fixture Exploration - Summary Report

## Overview

This exploration cataloged BitNet.rs test fixture patterns, GGUF loading mechanisms, and provided comprehensive guidance for creating three new test fixtures: `qk256_4x256.gguf`, `bitnet32_2x64.gguf`, and `qk256_3x300.gguf`.

## Key Findings

### 1. Existing Fixture Architecture

**Three-Layer System**:
- **Inline Fixtures** (< 1KB): Test data defined directly in test files
  - Example: `create_qk256_tensor()` in `qk256_integration.rs`
  - Use for: Unit tests, small synthetic data
  
- **Generated Fixtures** (1MB): Dynamic creation via `GgufFixtureGenerator`
  - Example: Complete GGUF file generation with configurable parameters
  - Use for: Integration tests, parametric testing
  
- **File-Based Fixtures** (> 1MB): Pre-generated GGUF files
  - Example: Files in `tests-new/fixtures/fixtures/gguf/valid/`
  - Use for: End-to-end testing, cross-validation

### 2. Primary Test Directories

- **crates/bitnet-models/tests/** - 50+ integration test files
  - `qk256_integration.rs` - QK256 kernel tests with helpers
  - `i2s_flavor_detection.rs` - Flavor detection validation
  - `qk256_loader_tests.rs` - Loader integration tests
  
- **tests-new/fixtures/fixtures/** - Comprehensive fixture framework
  - `gguf_generator.rs` - Full GGUF generation utility (700+ lines)
  - `fixture_loader.rs` - Centralized loading with feature gates
  - `device_aware_fixtures.rs` - Device-aware test data
  
- **crates/bitnet-models/src/formats/gguf/tests.rs** - GGUF parsing tests
  - Helper functions for building GGUF bytes
  - Binary format validation

### 3. GGUF Helper Functions

#### QK256 Tensor Creation (qk256_integration.rs:34-50)
```rust
fn create_qk256_tensor(rows: usize, cols: usize, code: u8) -> anyhow::Result<CandleTensor>
```
- Direct Candle tensor creation
- QK256-specific (256-element blocks, 64 bytes per block)
- Deterministic code mapping: 0→-2.0, 1→-1.0, 2→+1.0, 3→+2.0
- Returns `anyhow::Result<CandleTensor>`

#### GGUF Bytes Construction (gguf/tests.rs:16-93)
```rust
fn build_gguf_bytes(metadata: Vec<(&str, GgufValue)>) -> Vec<u8>
```
- Direct binary writing with `byteorder::WriteBytesExt`
- GGUF v2 header format
- Metadata KV pairs with type-specific serialization
- 32-byte alignment

#### Full GGUF File Generation (gguf_generator.rs:160-196)
```rust
pub fn generate_fixture(&self, config: &GgufFixtureConfig) -> Result<GgufFixture>
```
- Complete GGUF file writing
- Supports I2S, TL1, TL2, IQ2S, FP32 quantization types
- Deterministic data generation (seeded RNG)
- File integrity tracking (checksum, file size)

### 4. Configuration Structures

**GgufFixtureConfig**:
```rust
pub struct GgufFixtureConfig {
    pub name: String,
    pub model_type: ModelType,           // Minimal, BitNet158_1B, etc.
    pub quantization_type: QuantizationType,  // I2S, TL1, TL2, IQ2S, FP32
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub tensor_alignment: u64,           // 32 or 64
    pub generate_invalid: bool,          // For error testing
    pub seed: u64,                       // For determinism
}
```

### 5. Test Patterns Identified

| Pattern | Size | Speed | Reproducibility | Maintenance | Best For |
|---------|------|-------|-----------------|-------------|----------|
| Inline Synthetic | < 1KB | Very Fast | High | Low | Unit tests |
| Helper Functions | < 10KB | Fast | High | Medium | Test setup |
| Generated (GgufFixtureGenerator) | 1MB | Medium | High | Low | Integration tests |
| File-Based | > 1MB | Fast | Medium | High | Full e2e |

## Recommended Approach for New Fixtures

### Strategy: Generated Fixtures (Primary Recommendation)

**Rationale**:
1. No repository size impact (no binary files)
2. Deterministic and reproducible (seeded)
3. Easy to parametrize and scale
4. Perfect for CI/CD pipelines
5. Already established in codebase (`GgufFixtureGenerator`)

### The Three Fixtures

#### 1. qk256_4x256.gguf
- **Spec**: 4 rows × 256 cols (single-block QK256)
- **Purpose**: Single-block QK256 kernel validation
- **Seed**: 42
- **Test Coverage**: Dimension edge case, kernel dispatch, output validation

#### 2. bitnet32_2x64.gguf
- **Spec**: 2 rows × 64 cols (2 blocks of 32)
- **Purpose**: BitNet32F16 flavor detection and kernel testing
- **Seed**: 43
- **Expected Size**: 40 bytes (2 rows × 2 blocks × 10 bytes/block)
- **Test Coverage**: Flavor detection, multi-block handling, inline scale format

#### 3. qk256_3x300.gguf
- **Spec**: 3 rows × 300 cols (2 blocks + tail)
- **Purpose**: Multi-block QK256 with tail handling
- **Seed**: 44
- **Expected Size**: 384 bytes (3 rows × 2 blocks × 64 bytes/block)
- **Test Coverage**: Tail element handling, ceil division, non-aligned columns

## Implementation Steps

### Step 1: Extend GgufFixtureGenerator

Add methods to `tests-new/fixtures/fixtures/gguf_generator.rs`:

```rust
impl GgufFixtureGenerator {
    pub fn generate_qk256_edge_case_fixtures(&self) -> Result<Vec<GgufFixture>> { ... }
    pub fn generate_bitnet32_edge_case_fixtures(&self) -> Result<Vec<GgufFixture>> { ... }
}
```

### Step 2: Create Test File

New file: `crates/bitnet-models/tests/qk256_edge_case_fixtures.rs`

```rust
#[test]
fn test_qk256_4x256_fixture() -> anyhow::Result<()> { ... }

#[test]
fn test_bitnet32_2x64_fixture() -> anyhow::Result<()> { ... }

#[test]
fn test_qk256_3x300_fixture() -> anyhow::Result<()> { ... }
```

### Step 3: Documentation

Add fixture documentation to `tests/fixtures/gguf/README.md`:

```markdown
## qk256_4x256.gguf
- Quantization: QK256 (GGML I2_S), 2-bit signed
- Dimensions: 4 rows × 256 cols (single 256-element block)
- File Size: ~512 bytes
- Generated: Deterministic (seed=42)
- Tests: Single-block QK256 kernels, dimension validation

## bitnet32_2x64.gguf
- Quantization: BitNet32F16 (I2S with inline F16 scales)
- Dimensions: 2 rows × 64 cols (2 blocks of 32)
- File Size: 40 bytes (2 rows × 2 blocks × 10 bytes/block)
- Generated: Deterministic (seed=43)
- Tests: BitNet32F16 flavor detection, multi-block handling

## qk256_3x300.gguf
- Quantization: QK256 (GGML I2_S), 2-bit signed
- Dimensions: 3 rows × 300 cols (2 blocks + 44-element tail)
- File Size: 384 bytes
- Generated: Deterministic (seed=44)
- Tests: Tail handling, non-aligned column counts
```

## Best Practices Applied

1. **Determinism**: All fixtures use seeded RNGs (seeds: 42, 43, 44)
2. **Organization**: Clear separation of concern (generator utility, test file, documentation)
3. **Reusability**: Generated fixtures can be used across multiple tests
4. **Scalability**: Adding new fixtures requires minimal code changes
5. **Error Handling**: Pattern for testing both success and error paths
6. **Parametric Testing**: Use test matrices for comprehensive coverage

## Related Codebase Locations

- **GGUF Format Spec**: `/crates/bitnet-models/src/formats/gguf/`
- **QK256 Kernel**: `/crates/bitnet-models/src/quant/i2s_qk256.rs`
- **QK256 Tests**: `/crates/bitnet-models/tests/qk256_integration.rs`
- **Fixture Generator**: `/tests-new/fixtures/fixtures/gguf_generator.rs`
- **Test Patterns**: `/crates/bitnet-models/tests/` (50+ integration test files)
- **Helper Utilities**: `/crates/bitnet-tokenizers/src/test_utils.rs`

## Next Steps

1. Review the detailed documentation in `fixture_patterns.md`
2. Extend `GgufFixtureGenerator` with edge-case fixture methods
3. Create integration tests using the generated fixtures
4. Validate quantization flavor detection (I2S vs QK256)
5. Test kernel dispatch and mathematical correctness
6. Document fixture specifications and test coverage

## Files Generated

- `/ci/exploration/fixture_patterns.md` (27KB, 946 lines)
  - Comprehensive guide to fixture patterns and architecture
  - Detailed API documentation for `GgufFixtureGenerator`
  - Implementation examples for each fixture
  - Best practices and recommendations

---

**Generated**: 2025-10-22
**Codebase**: BitNet.rs Neural Network Inference Library
**Scope**: Test fixture patterns, GGUF loading, fixture creation guidance
