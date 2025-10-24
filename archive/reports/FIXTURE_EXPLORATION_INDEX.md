# BitNet.rs Fixture Contract & Validation - Exploration Index

## Overview

This exploration thoroughly documents the BitNet.rs fixture contract and validation system, covering fixture generation, storage, verification, and integration testing.

## Comprehensive Report

The main findings are documented in a 768-line comprehensive report:

**Location**: `/home/steven/code/Rust/BitNet-rs/FIXTURE_CONTRACT_AND_VALIDATION_REPORT.md` (26 KB)

## Quick Summary

### 1. Fixture Inventory (3 files)
- `qk256_4x256.gguf` (10,816 B) - Single-block QK256 format
- `bitnet32_2x64.gguf` (8,832 B) - BitNet32-F16 format  
- `qk256_3x300.gguf` (10,696 B) - Multi-block QK256 with tail

### 2. SHA256 Verification
- **Location**: `crates/bitnet-models/tests/helpers/fixture_loader.rs`
- **Checksums file**: `ci/fixtures/qk256/SHA256SUMS`
- **Implementation**: `verify_checksum(filename, expected_sha256) → bool`
- **3 hardcoded checksums** in `checksums` module for runtime verification

### 3. GGUF Validation (7 gates)
- **Parser**: `crates/bitnet-models/src/gguf_min.rs`
- **Gates**: Magic, Version, Offset alignment, Data alignment, Bounds, Overflow, Block consistency

### 4. Fixture Generators (Deterministic)
- **Location**: `crates/bitnet-models/tests/helpers/qk256_fixtures.rs`
- **Functions**: `generate_qk256_4x256()`, `generate_bitnet32_2x64()`, `generate_qk256_3x300()`
- **All seeded for reproducibility**

### 5. Test Suites (15+ tests)
- **qk256_dual_flavor_tests.rs**: Format detection & routing (4 tests)
- **qk256_fixture_loader_tests.rs**: Disk loading & verification (7 tests)
- **qk256_fixture_validation.rs**: Generator validation (5 tests)

### 6. Loader Infrastructure
- **Fixture path resolution**: Workspace-aware navigation
- **Byte loading**: Disk-based fixture reading
- **Checksum verification**: Feature-gated SHA256 checking

### 7. Format Detection System
- **QK256**: Size = rows × ceil(cols/256) × 64
- **BitNet32-F16**: Size = rows × ceil(cols/32) × 10
- **All three fixtures verified** to match expected sizes

### 8. Regeneration Workflow
Four-step process documented in `ci/fixtures/qk256/README.md`:
1. Generate to /tmp
2. Copy to ci/fixtures/qk256/
3. Update SHA256SUMS
4. Verify tests pass

### 9. Key Design Principles
- **Determinism**: Seeded generators for reproducible tests
- **Alignment**: Strict 32-byte GGUF v3 compliance
- **Detection**: Automatic format routing via size analysis
- **Integrity**: SHA256 checksums for corruption detection
- **Flexibility**: Feature-gated disk or in-memory fixtures

### 10. Edge Cases Handled
- Single-block tensors (cols = block_size)
- Multi-block with tail (cols > block_size)
- Different block sizes (256 vs 32)
- F16 scale factors
- Tied embeddings

## Key Files (11 primary sources)

### Fixture Assets
- `/home/steven/code/Rust/BitNet-rs/ci/fixtures/qk256/README.md`
- `/home/steven/code/Rust/BitNet-rs/ci/fixtures/qk256/QUICK_REFERENCE.md`
- `/home/steven/code/Rust/BitNet-rs/ci/fixtures/qk256/SHA256SUMS`

### Generator & Loader
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/helpers/qk256_fixtures.rs`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/helpers/fixture_loader.rs`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/helpers/mod.rs`

### GGUF Parsers
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/gguf_min.rs` (validation)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/gguf_simple.rs` (enhanced loader)

### Test Suites
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/qk256_dual_flavor_tests.rs`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/qk256_fixture_loader_tests.rs`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/qk256_fixture_validation.rs`

## Structure Visualization

```
GGUF File Structure (32-byte aligned)
┌─────────────────────────────────┐
│ Header (4B magic + version)      │
├─────────────────────────────────┤
│ Metadata (8 KV pairs)           │
├─────────────────────────────────┤
│ Tensor Info (2 tensors)         │
├─────────────────────────────────┤
│ [32-byte alignment padding]     │
├─────────────────────────────────┤
│ Tensor Data Section             │
│  ├─ tok_embeddings (I2_S)      │
│  ├─ [32-byte alignment padding] │
│  └─ output.weight (F16)         │
└─────────────────────────────────┘

Format Detection Logic
QK256: rows × ceil(cols/256) × 64
BitNet32-F16: rows × ceil(cols/32) × 10

Verification Gates (7 total)
1. Magic number = "GGUF"
2. Version ∈ {2, 3}
3. Tensor offset % alignment == 0
4. Data section start % alignment == 0
5. offset + size ≤ file size
6. dims.product() fits in u64
7. Block count consistent with element count
```

## Testing Approach

### In-Memory Fixtures (Fast)
```rust
let fixture_bytes = qk256_fixtures::generate_qk256_4x256(42);
let mut file = NamedTempFile::new()?;
file.write_all(&fixture_bytes)?;
let result = load_gguf_full(file.path(), Device::Cpu, config)?;
```

### Disk-Based Fixtures (CI/CD)
```rust
let path = fixture_loader::fixture_path("qk256_4x256.gguf");
assert!(fixture_loader::verify_checksum("qk256_4x256.gguf", 
    fixture_loader::checksums::QK256_4X256));
let result = load_gguf_full(&path, Device::Cpu, config)?;
```

## Report Contents (14 sections)

1. Executive Summary
2. Fixture Directory Catalog
3. SHA256 Verification System
4. GGUF Header & Structure Validation
5. GGUF Parser Validation (7 gates detailed)
6. Fixture-Based Integration Tests
7. Loader Infrastructure
8. Fixture Validation Patterns
9. Fixture Regeneration Workflow
10. GGUF Header Validation Gates
11. Format Detection: QK256 vs BitNet32-F16
12. Test Infrastructure Integration
13. Fixture Validation Checklist
14. Key Insights & Design Principles

## Checksums (For Reference)

```
a41cc62c893bcf1d4c03c30ed3da12da03c339847c4d564e9e5794b5d4c6932a  qk256_4x256.gguf
c1568a0a08e38ef2865ce0816bfd2c617e5589c113114cd731e4c5014b7fbb20  bitnet32_2x64.gguf
6e5a4f21607c0064affbcb86133627478eb34d812b59807a7123ff386c63bd3e  qk256_3x300.gguf
```

## Regeneration Quick Reference

```bash
# Generate new fixtures
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features fixtures test_dump_fixture_for_debug -- --nocapture

# Copy to fixtures directory
cp /tmp/test_qk256_*.gguf ci/fixtures/qk256/

# Update checksums
cd ci/fixtures/qk256 && sha256sum *.gguf > SHA256SUMS

# Verify
cargo test -p bitnet-models --test qk256_fixture_loader_tests --features fixtures
```

## Key Findings

1. **Comprehensive System**: 3 fixture types, 7 validation gates, 15+ tests, 2 loading patterns
2. **Deterministic**: All fixtures regenerable via seed-based generators
3. **Verified**: SHA256 checksums for integrity, 32-byte alignment for compliance
4. **Flexible**: Both in-memory (fast) and disk-based (CI/CD) patterns supported
5. **Production-Ready**: Robust error handling with 7 validation gates
6. **Edge-Case Handling**: Single/multi-block, different block sizes, F16 scales, tied embeddings

## Exploration Methodology

- **Thoroughness Level**: Medium (Complete)
- **Source Files Examined**: 12 primary + 3 documentation
- **Lines of Code Analyzed**: 1000+ (generators, loaders, tests, parsers)
- **Test Coverage**: 15+ specific test functions detailed
- **Documentation**: 2 README files, 1 Quick Reference, inline code comments
