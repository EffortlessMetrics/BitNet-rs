# BitNet.rs Fixture Contract & Validation - Comprehensive Exploration Report

## Executive Summary

The BitNet.rs project implements a comprehensive fixture contract and validation system for QK256 (GGML I2_S) quantization format testing. The system combines:

1. **Deterministic fixture generation** via in-memory generators
2. **Disk-based persistent fixtures** in `ci/fixtures/qk256/`
3. **SHA256 checksum verification** for integrity assurance
4. **GGUF structure validation** with strict alignment enforcement
5. **Schema-aware loading** with format detection and error handling

---

## Part 1: Fixture Directory Catalog

### Location
```
ci/fixtures/qk256/
├── qk256_4x256.gguf      # 10,816 bytes
├── bitnet32_2x64.gguf    #  8,832 bytes
├── qk256_3x300.gguf      # 10,696 bytes
├── SHA256SUMS            # Checksum file
├── README.md             # Comprehensive documentation
└── QUICK_REFERENCE.md    # Command reference
```

### Fixture Specifications

| File | Size | Tensor Shape | Format | Purpose |
|------|------|--------------|--------|---------|
| `qk256_4x256.gguf` | 10,816 B | [4, 256] | QK256 (GGML I2_S) | Single-block edge case (cols = block_size) |
| `bitnet32_2x64.gguf` | 8,832 B | [2, 64] | BitNet32-F16 | Two-block format (inline F16 scales) |
| `qk256_3x300.gguf` | 10,696 B | [3, 300] | QK256 (GGML I2_S) | Multi-block with tail (256 + 44 tail) |

---

## Part 2: SHA256 Verification

### Checksum File: `SHA256SUMS`
```
c1568a0a08e38ef2865ce0816bfd2c617e5589c113114cd731e4c5014b7fbb20  bitnet32_2x64.gguf
6e5a4f21607c0064affbcb86133627478eb34d812b59807a7123ff386c63bd3e  qk256_3x300.gguf
a41cc62c893bcf1d4c03c30ed3da12da03c339847c4d564e9e5794b5d4c6932a  qk256_4x256.gguf
```

### Verification Implementation

**Location**: `crates/bitnet-models/tests/helpers/fixture_loader.rs:94-105`

```rust
#[cfg(feature = "fixtures")]
pub fn verify_checksum(filename: &str, expected_sha256: &str) -> bool {
    use sha2::{Digest, Sha256};

    let bytes = load_fixture_bytes(filename);
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let result = hasher.finalize();
    let actual_hex = format!("{:x}", result);

    actual_hex == expected_sha256
}
```

### Checksums Module
**Location**: `crates/bitnet-models/tests/helpers/fixture_loader.rs:108-118`

```rust
pub mod checksums {
    pub const QK256_4X256: &str =
        "a41cc62c893bcf1d4c03c30ed3da12da03c339847c4d564e9e5794b5d4c6932a";
    pub const BITNET32_2X64: &str =
        "c1568a0a08e38ef2865ce0816bfd2c617e5589c113114cd731e4c5014b7fbb20";
    pub const QK256_3X300: &str =
        "6e5a4f21607c0064affbcb86133627478eb34d812b59807a7123ff386c63bd3e";
}
```

### Verification Tests
**Location**: `crates/bitnet-models/tests/qk256_fixture_loader_tests.rs:42-67`

Tests verify that each fixture matches its expected SHA256 checksum:
- `test_verify_qk256_4x256_checksum()`
- `test_verify_bitnet32_2x64_checksum()`
- `test_verify_qk256_3x300_checksum()`

---

## Part 3: GGUF Header & Structure Validation

### GGUF v3 Structure

All fixtures follow the GGUF v3 specification:

```
┌─────────────────────────────────────────────────────────┐
│ GGUF File Structure                                     │
├─────────────────────────────────────────────────────────┤
│ Header                                                  │
│  - Magic (4B):      "GGUF"                             │
│  - Version (4B):    3 (little-endian u32)              │
│  - Tensor Count (8B): 2 (little-endian u64)            │
│  - KV Count (8B):   8 (little-endian u64)              │
├─────────────────────────────────────────────────────────┤
│ Key-Value Pairs (8 KVs)                                │
│  - general.name                 (string)               │
│  - general.architecture         (string)               │
│  - tokenizer.ggml.tokens       (string array, 1000)    │
│  - bitnet-b1.58.embedding_length (u32, 512)            │
│  - bitnet-b1.58.block_count    (u32, 1)                │
│  - bitnet-b1.58.attention.head_count (u32, 8)          │
│  - bitnet-b1.58.attention.head_count_kv (u32, 8)       │
│  - bitnet-b1.58.feed_forward_length (u32, 2048)        │
├─────────────────────────────────────────────────────────┤
│ Tensor Metadata (2 tensors)                            │
│  Tensor 1: tok_embeddings.weight                       │
│   - Name length (8B) + name bytes                      │
│   - Dimensions: 2 (n_dims = 2)                         │
│   - Shape: [rows, cols] (each 8B u64)                 │
│   - Type: 36 (GGUF_TYPE_I2S for QK256)                 │
│   - Offset: relative to data section (8B u64)          │
│  Tensor 2: output.weight                               │
│   - Name, dims, shape (same structure)                 │
│   - Type: 1 (GGUF_TYPE_F16)                           │
│   - Offset: relative to data section                   │
├─────────────────────────────────────────────────────────┤
│ Alignment Padding                                       │
│  - Aligned to 32-byte boundary before data section      │
├─────────────────────────────────────────────────────────┤
│ Data Section (32-byte aligned)                         │
│  - tok_embeddings data (I2_S quantized)                │
│  - 32-byte alignment padding between tensors           │
│  - output.weight data (F16 format)                     │
└─────────────────────────────────────────────────────────┘
```

### Generator Implementation

**Location**: `crates/bitnet-models/tests/helpers/qk256_fixtures.rs`

#### Constants
```rust
const QK256_BLOCK: usize = 256;              // QK256 block size
const QK256_PACKED_BYTES: usize = 64;        // 2 bits × 256 / 8 = 64 bytes per block
const BITNET32_BLOCK: usize = 32;            // BitNet32 block size
const BITNET32_BYTES_PER_BLOCK: usize = 10;  // 8 data + 2 F16 scale bytes
const GGUF_ALIGNMENT: usize = 32;            // 32-byte alignment for data section
const GGUF_VERSION: u32 = 3;
const GGUF_TYPE_I2S: u32 = 36;               // I2_S quantization type
const GGUF_VALUE_TYPE_STRING: u32 = 8;       // String KV type
```

#### Fixture Generators

1. **`generate_qk256_4x256(seed: u64) -> Vec<u8>`**
   - Shape: [4, 256]
   - Single block per row (256 cols = 1 full block)
   - Row stride: 1 × 64 = 64 bytes
   - Deterministic via seed-based code generation

2. **`generate_bitnet32_2x64(seed: u64) -> Vec<u8>`**
   - Shape: [2, 64]
   - Two blocks per row (64 cols = 2 full blocks)
   - Bytes per row: 2 × 10 = 20 bytes
   - Includes F16 scale values (0x3C00 = 1.0f)

3. **`generate_qk256_3x300(seed: u64) -> Vec<u8>`**
   - Shape: [3, 300]
   - Multi-block layout: 2 blocks per row (256 + 44 tail)
   - Row stride: 2 × 64 = 128 bytes
   - Tests tail block handling

#### Key Implementation Details

**Offset Calculation** (lines 227-230):
```rust
let data_start = buf.len() as u64;
let tok_offset_absolute = buf.len() as u64;
let tok_offset_relative = tok_offset_absolute - data_start; // Should be 0
buf[tok_offset_pos..tok_offset_pos + 8].copy_from_slice(&tok_offset_relative.to_le_bytes());
```

**32-byte Alignment Enforcement** (lines 218-246):
```rust
// Alignment to 32-byte boundary before data section
let current_len = buf.len();
let padding = (GGUF_ALIGNMENT - (current_len % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;
buf.resize(current_len + padding, 0);

// ... later, after tensor 1 data ...

// CRITICAL: Add 32-byte alignment padding between tensors
let current_pos = buf.len();
let padding_needed = (GGUF_ALIGNMENT - (current_pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;
if padding_needed > 0 {
    buf.resize(current_pos + padding_needed, 0);
}
```

**Determinism via Seed** (line 69):
```rust
let code = ((seed % 4) as u8).clamp(0, 3);
let packed_byte = code | (code << 2) | (code << 4) | (code << 6);
let tensor_data = vec![packed_byte; tensor_bytes];
```

---

## Part 4: GGUF Parser Validation

### Minimal Parser: `gguf_min.rs`

**Purpose**: Lightweight GGUF reader for critical tensor extraction with robust alignment checking

#### Header Parsing (lines 156-215)

**Magic & Version Check**:
```rust
let mut magic = [0u8; 4];
r.read_exact(&mut magic)?;
if &magic != b"GGUF" {
    bail!("not a GGUF file (missing magic)");
}

let version = read_u32(r)?;
if version != 2 && version != 3 {
    bail!("unsupported GGUF version {version} (only v2/v3)");
}
```

**Alignment Validation** (lines 199-204):
```rust
ensure!(
    offset % alignment == 0,
    "tensor '{}' offset {} not aligned to {alignment}",
    name,
    offset
);
```

**Data Section Alignment** (lines 209-212):
```rust
let here = r.stream_position()?;
let data_offset = align_up(here, alignment);
ensure!(data_offset.is_multiple_of(alignment), "data section not aligned to {alignment}");
```

#### Tensor Materialization: `tensor_as_f32()` (lines 278-349)

**Size Calculation & Overflow Check**:
```rust
let nelems: usize = info
    .dims
    .iter()
    .try_fold(1u64, |acc, &d| acc.checked_mul(d))  // Overflow check
    .ok_or_else(|| anyhow::anyhow!("tensor size overflow"))?
    .try_into()
    .map_err(|_| anyhow::anyhow!("tensor too large"))?;
```

**Alignment Revalidation**:
```rust
ensure!(
    info.offset.is_multiple_of(alignment),
    "tensor '{}' offset {} not aligned to {alignment}",
    info.name,
    info.offset
);
```

**Out-of-Bounds Checking**:
```rust
// F32 example
let need = nelems * 4;
if offset + need > mmap.len() {
    bail!("f32 tensor out of bounds");
}

// I2_S example
let need = num_blocks * layout.bytes_per_block;
ensure!(offset + need <= mmap.len(), "{}", i2s_oob!(info, offset, need, mmap.len()));
```

**I2_S Block/Shape Validation** (lines 339-348):
```rust
ensure!(
    num_blocks * layout.block_size >= nelems
        && num_blocks * layout.block_size - nelems < layout.block_size,
    "I2_S blocks/shape mismatch for tensor '{}': nelems={}, blocks={}, block_size={}, shape={:?}",
    info.name,
    nelems,
    num_blocks,
    layout.block_size,
    info.dims
);
```

---

## Part 5: Fixture-Based Integration Tests

### Test Suite: `qk256_dual_flavor_tests.rs`

**Approach**: In-memory fixture generation with deterministic seeds

#### Test 1: QK256 Detection by Size (lines 112-165)

```rust
#[test]
#[cfg_attr(not(feature = "fixtures"), ignore = "Requires real or generated GGUF fixtures")]
fn test_qk256_detection_by_size() {
    // Shape: [4, 256] → 1024 elements
    // QK256 expects: ceil(256/256) = 1 block × 4 rows = 256 bytes
    
    let fixture_bytes = helpers::qk256_fixtures::generate_qk256_4x256(42);
    let mut file = NamedTempFile::new()?;
    file.write_all(&fixture_bytes)?;
    
    let result = load_gguf_full(file.path(), Device::Cpu, config)?;
    
    // Verify QK256 detection
    assert_eq!(result.i2s_qk256.len(), 1);
    assert!(result.i2s_qk256.contains_key("tok_embeddings.weight"));
    
    // Verify structure
    let qk256 = result.i2s_qk256.get("tok_embeddings.weight").unwrap();
    assert_eq!(qk256.rows, 4);
    assert_eq!(qk256.cols, 256);
    assert_eq!(qk256.row_stride_bytes, 64);
}
```

**Validations**:
- GGUF magic and version
- Tensor presence in QK256 map (not in regular tensors)
- Shape consistency (rows, cols, stride)

#### Test 2: BitNet32 Dequantization Path (lines 167-211)

```rust
#[test]
fn test_bitnet32_still_uses_fp_path() {
    // Shape: [2, 64]
    // BitNet-32 expects: ceil(64/32) = 2 blocks × 10 bytes × 2 rows = 40 bytes
    
    let fixture_bytes = helpers::qk256_fixtures::generate_bitnet32_2x64(43);
    let result = load_gguf_full(file.path(), Device::Cpu, config)?;
    
    // Verify NOT in QK256 map
    assert_eq!(result.i2s_qk256.len(), 0);
    
    // Verify in regular tensors (dequantized to F32)
    assert!(result.tensors.contains_key("token_embd.weight"));
}
```

**Validations**:
- BitNet32-F16 takes FP dequantization path (not QK256)
- Tensor name normalization (tok_embeddings.weight → token_embd.weight)
- Shape normalization to config metadata

#### Test 3: Multi-Block with Tail (lines 213-244)

```rust
#[test]
fn test_qk256_with_non_multiple_cols() {
    // Shape: [3, 300] (ceil(300/256) = 2 blocks per row)
    let fixture_bytes = helpers::qk256_fixtures::generate_qk256_3x300(44);
    let result = load_gguf_full(file.path(), Device::Cpu, config)?;
    
    assert_eq!(result.i2s_qk256.len(), 1);
    let qk256 = result.i2s_qk256.get("tok_embeddings.weight").unwrap();
    assert_eq!(qk256.rows, 3);
    assert_eq!(qk256.cols, 300);
    assert_eq!(qk256.row_stride_bytes, 128);  // 2 blocks × 64
}
```

**Validations**:
- Multi-block handling with tail elements
- Correct stride calculation for partial blocks

### Test Suite: `qk256_fixture_loader_tests.rs`

**Approach**: Disk-based fixtures with SHA256 verification

#### Tests

1. **Path Existence Tests**: Verify fixtures exist on disk
2. **Size Validation Tests**: Check expected file sizes
3. **Checksum Verification**: SHA256 against known values
4. **GGUF Header Tests**: Magic number and version
5. **Batch Loading**: Load all fixtures successfully

**Key Test**: `test_verify_qk256_4x256_checksum()` (lines 43-47)
```rust
#[test]
#[cfg(feature = "fixtures")]
fn test_verify_qk256_4x256_checksum() {
    assert!(
        fixture_loader::verify_checksum("qk256_4x256.gguf", fixture_loader::checksums::QK256_4X256),
        "QK256 4x256 checksum should match"
    );
}
```

---

## Part 6: Loader Infrastructure

### Fixture Loading Module: `fixture_loader.rs`

**Location**: `crates/bitnet-models/tests/helpers/fixture_loader.rs`

#### Core Functions

1. **`fixture_path(filename: &str) -> PathBuf`** (lines 42-62)
   - Resolves workspace root from CARGO_MANIFEST_DIR
   - Navigates: `crates/bitnet-models` → `workspace root` → `ci/fixtures/qk256/`
   - Panics if fixture doesn't exist with helpful path debugging

2. **`load_fixture_bytes(filename: &str) -> Vec<u8>`** (lines 77-82)
   - Loads fixture file from disk
   - Panics on read failure with context

3. **`verify_checksum(filename: &str, expected_sha256: &str) -> bool`** (lines 94-105)
   - Computes SHA256 of fixture bytes
   - Returns boolean (enabled with `feature = "fixtures"`)

#### Checksum Module (lines 108-118)
```rust
pub mod checksums {
    pub const QK256_4X256: &str = "a41cc62c893bcf1d4c03c30ed3da12da03c339847c4d564e9e5794b5d4c6932a";
    pub const BITNET32_2X64: &str = "c1568a0a08e38ef2865ce0816bfd2c617e5589c113114cd731e4c5014b7fbb20";
    pub const QK256_3X300: &str = "6e5a4f21607c0064affbcb86133627478eb34d812b59807a7123ff386c63bd3e";
}
```

#### Built-in Tests (lines 120-164)
- `test_fixture_path_exists()`: Verify path resolution
- `test_load_fixture_bytes()`: Load and verify GGUF magic
- `test_verify_checksum_*()`: Verify each fixture's checksum

---

## Part 7: Fixture Validation Patterns

### Pattern 1: In-Memory Fixture Generation (Fast Unit Tests)

```rust
#[test]
fn test_with_generated_fixture() {
    use helpers::qk256_fixtures;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    let fixture_bytes = qk256_fixtures::generate_qk256_4x256(42);
    let mut file = NamedTempFile::new()?;
    file.write_all(&fixture_bytes)?;
    file.flush()?;
    
    let result = load_gguf_full(
        file.path(),
        Device::Cpu,
        GGUFLoaderConfig::default(),
    )?;
    
    // Validate result
}
```

**Advantages**:
- No disk I/O (fast)
- Deterministic via seed
- Lightweight (in-process generation)

### Pattern 2: Disk-Based Fixtures (CI/CD, Deterministic)

```rust
#[test]
#[cfg(feature = "fixtures")]
fn test_with_disk_fixture() {
    use helpers::fixture_loader;
    
    let path = fixture_loader::fixture_path("qk256_4x256.gguf");
    assert!(path.exists());
    
    // Verify checksum before loading
    let valid = fixture_loader::verify_checksum(
        "qk256_4x256.gguf",
        fixture_loader::checksums::QK256_4X256,
    );
    assert!(valid);
    
    let result = load_gguf_full(&path, Device::Cpu, config)?;
}
```

**Advantages**:
- Version-controlled fixtures
- Reproducible CI/CD
- Detectable corruption via checksums

---

## Part 8: Fixture Regeneration Workflow

### Step-by-Step Process

**From README.md (lines 69-81)**:

```bash
# Step 1: Generate fixtures to /tmp
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features fixtures test_dump_fixture_for_debug -- --nocapture

# Step 2: Copy to ci/fixtures/qk256/
cp /tmp/test_qk256_4x256.gguf ci/fixtures/qk256/qk256_4x256.gguf
cp /tmp/test_bitnet32_2x64.gguf ci/fixtures/qk256/bitnet32_2x64.gguf
cp /tmp/test_qk256_3x300.gguf ci/fixtures/qk256/qk256_3x300.gguf

# Step 3: Update checksums
cd ci/fixtures/qk256 && sha256sum *.gguf > SHA256SUMS

# Step 4: Verify loader tests pass
cargo test -p bitnet-models --test qk256_fixture_loader_tests --no-default-features --features fixtures
```

### Validation After Regeneration

```bash
# Verify checksums
sha256sum -c ci/fixtures/qk256/SHA256SUMS

# Run fixture loader tests
cargo test -p bitnet-models --test qk256_fixture_loader_tests --no-default-features --features fixtures

# Run dual-flavor detection tests
cargo test -p bitnet-models --test qk256_dual_flavor_tests --no-default-features --features fixtures

# Inspect with CLI
cargo run -p bitnet-cli --features cpu,full-cli -- compat-check ci/fixtures/qk256/qk256_4x256.gguf
```

---

## Part 9: GGUF Header Validation Gates

### Gate 1: Magic Number
- **What**: First 4 bytes must be "GGUF"
- **Where**: `gguf_min.rs:159-161`
- **Failure**: "not a GGUF file (missing magic)"

### Gate 2: Version Support
- **What**: Version must be 2 or 3
- **Where**: `gguf_min.rs:164-167`
- **Failure**: "unsupported GGUF version X (only v2/v3)"

### Gate 3: Tensor Offset Alignment
- **What**: Each tensor offset % alignment == 0
- **Where**: `gguf_min.rs:199-204`
- **Failure**: "tensor 'X' offset Y not aligned to Z"
- **Alignment**: Default 32 bytes (from general.alignment KV)

### Gate 4: Data Section Alignment
- **What**: Data section starts at aligned boundary
- **Where**: `gguf_min.rs:209-212`
- **Failure**: "data section not aligned to X"

### Gate 5: Tensor Size Bounds
- **What**: offset + size ≤ mmap.len()
- **Where**: `gguf_min.rs:305-320` (F32/F16), `gguf_min.rs:336` (I2_S)
- **Failure**: "f32/f16/I2_S tensor out of bounds"

### Gate 6: Element Count Overflow
- **What**: dims.product() doesn't overflow u64
- **Where**: `gguf_min.rs:286-292`
- **Failure**: "tensor size overflow" or "tensor too large"

### Gate 7: I2_S Block Consistency
- **What**: nelems and block count are consistent
- **Where**: `gguf_min.rs:339-348`
- **Failure**: "I2_S blocks/shape mismatch for tensor X"

---

## Part 10: Format Detection: QK256 vs BitNet32-F16

### Detection Logic (from dual-flavor tests)

| Fixture | Tensor Size | Shape | Calculation | Detected Format |
|---------|------------|-------|------------|-----------------|
| qk256_4x256 | 256 bytes | [4, 256] | 4 × ceil(256/256) × 64 = 256 ✓ | QK256 |
| bitnet32_2x64 | 40 bytes | [2, 64] | 2 × ceil(64/32) × 10 = 40 ✓ | BitNet32-F16 |
| qk256_3x300 | 384 bytes | [3, 300] | 3 × ceil(300/256) × 64 = 384 ✓ | QK256 |

### Decision Logic

1. **Calculate expected size for QK256**:
   ```
   blocks_per_row = ceil(cols / 256)
   expected_qk256_size = rows × blocks_per_row × 64
   ```

2. **If actual_size == expected_qk256_size**: Route to QK256 kernel

3. **Else**: Try BitNet32-F16
   ```
   blocks_per_row = ceil(cols / 32)
   expected_bitnet32_size = rows × blocks_per_row × 10
   ```

4. **If size matches**: Route to BitNet32-F16 dequantization

5. **Else**: Error or fallback

---

## Part 11: Test Infrastructure Integration

### Helper Module: `crates/bitnet-models/tests/helpers/mod.rs`

**Re-exports** (for convenience):

```rust
pub mod alignment_validator;
pub mod fixture_loader;
pub mod qk256_fixtures;
pub mod qk256_tolerance;

pub use qk256_fixtures::{
    generate_bitnet32_2x64,
    generate_qk256_3x300,
    generate_qk256_4x256,
};

pub use fixture_loader::{fixture_path, load_fixture_bytes};

pub use alignment_validator::{
    AlignmentConfig, ValidationResult, validate_all_tensors,
    validate_candle_tensor, validate_gguf_tensor_metadata,
};

pub use qk256_tolerance::{approx_eq, approx_eq_with_len};
```

### Feature Gate

All fixture tests require `feature = "fixtures"`:

```rust
#[test]
#[cfg_attr(not(feature = "fixtures"), ignore = "Requires real or generated GGUF fixtures")]
fn test_qk256_detection_by_size() { ... }
```

**Enable with**:
```bash
cargo test --features fixtures
```

---

## Part 12: Fixture Validation Checklist

### Pre-Deployment

- [ ] Fixture generation is deterministic (same seed → same bytes)
- [ ] GGUF magic = "GGUF"
- [ ] GGUF version = 3
- [ ] All tensor offsets 32-byte aligned
- [ ] Data section 32-byte aligned
- [ ] No out-of-bounds reads (offset + size ≤ file length)
- [ ] Correct block/shape consistency for I2_S tensors
- [ ] SHA256 checksums computed and stored
- [ ] All loader tests pass
- [ ] CI integration tests pass

### Post-Regeneration

1. Generate new fixtures to /tmp
2. Compute fresh SHA256 checksums
3. Copy to ci/fixtures/qk256/
4. Run `qk256_fixture_loader_tests` → all pass
5. Run `qk256_dual_flavor_tests` → all pass
6. Run CLI inspection:
   ```bash
   cargo run -p bitnet-cli --features cpu,full-cli -- \
     compat-check ci/fixtures/qk256/qk256_4x256.gguf
   ```
7. Commit SHA256SUMS, fixtures to version control

---

## Part 13: Key Insights

### Design Principles

1. **Determinism First**: Fixtures are regenerable via seed-based generators, ensuring reproducible tests
2. **32-Byte Alignment**: Strict GGUF v3 compliance for cross-platform compatibility
3. **Dual-Path Detection**: Supports both QK256 (256-elem blocks) and BitNet32-F16 (32-elem blocks) via size analysis
4. **Checksum Assurance**: SHA256 verification detects any fixture corruption or unintended mutations
5. **Feature-Gated**: Disk fixtures optional; in-memory generation always available

### Test Coverage

- **Structure Validation**: GGUF header, tensor info, alignment
- **Format Detection**: QK256 vs BitNet32-F16 routing
- **Integration**: End-to-end GGUF loading with shape/metadata validation
- **Checksum Integrity**: SHA256 verification against known values
- **Determinism**: Seeded generation for reproducible CI

### Alignment Strategy

```
┌────────────────────────────────────────────────┐
│ GGUF Header + Metadata (variable length)       │
└────────────────────────────────────────────────┘
               ↓ (align to 32 bytes)
┌────────────────────────────────────────────────┐
│ Data Section Start (32-byte aligned)           │
├────────────────────────────────────────────────┤
│ Tensor 1 data                                  │
│ (e.g., tok_embeddings.weight, I2_S quantized) │
└────────────────────────────────────────────────┘
               ↓ (align to 32 bytes between tensors)
┌────────────────────────────────────────────────┐
│ Tensor 2 data                                  │
│ (e.g., output.weight, F16)                    │
└────────────────────────────────────────────────┘
```

---

## Part 14: Related Documentation

### README Files
- `ci/fixtures/qk256/README.md` - Comprehensive fixture documentation
- `ci/fixtures/qk256/QUICK_REFERENCE.md` - Quick command reference

### Source Code Files
- `crates/bitnet-models/tests/helpers/qk256_fixtures.rs` - Generator implementation
- `crates/bitnet-models/tests/helpers/fixture_loader.rs` - Disk-based loader
- `crates/bitnet-models/src/gguf_min.rs` - GGUF parser with validation
- `crates/bitnet-models/src/gguf_simple.rs` - Enhanced GGUF loader (fallback path)

### Test Files
- `crates/bitnet-models/tests/qk256_dual_flavor_tests.rs` - Integration tests
- `crates/bitnet-models/tests/qk256_fixture_loader_tests.rs` - Disk-based loading tests
- `crates/bitnet-models/tests/qk256_fixture_validation.rs` - Fixture generation validation
- `crates/bitnet-models/tests/qk256_integration.rs` - Full integration tests

---

## Summary

BitNet.rs implements a **comprehensive fixture contract** that combines:

1. **Persistent GGUF fixtures** (3 variants covering QK256 and BitNet32-F16)
2. **SHA256 checksum verification** for integrity assurance
3. **Strict GGUF structure validation** with 7 gates (magic, version, alignment, bounds, overflow, block consistency)
4. **Deterministic fixture generation** via seeded generators
5. **Format auto-detection** for proper quantization routing
6. **Feature-gated testing** for disk-based or in-memory fixtures
7. **Production-ready validation** in `gguf_min.rs` with comprehensive error messages

The system ensures reproducible CI/CD, detectable fixture corruption, and robust handling of edge cases (multi-block tensors, tail elements, tied embeddings).
