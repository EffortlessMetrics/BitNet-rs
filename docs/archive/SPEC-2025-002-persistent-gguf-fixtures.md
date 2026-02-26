# SPEC-2025-002: Persistent GGUF Fixtures and Integration Test Wiring

**Status**: Draft
**Created**: 2025-10-23
**Priority**: P1
**Category**: Test Infrastructure
**Related Issues**: None
**Related PRs**: None

---

## Executive Summary

Create persistent disk-based GGUF fixtures and wire integration tests to use them, providing deterministic, version-controlled test data for QK256 dual-flavor quantization validation. This eliminates runtime fixture generation overhead and ensures reproducible CI/CD builds.

**Current State**: Tests generate fixtures in-memory at runtime using `helpers::qk256_fixtures` module (~200ms overhead per test suite).

**Target State**: Tests load fixtures from `ci/fixtures/qk256/` with SHA256 verification, reducing fixture overhead to <10ms.

**Impact**:
- **CI Speed**: ~20× faster fixture loading (200ms → <10ms per suite)
- **Reproducibility**: Version-controlled fixtures with cryptographic integrity
- **Debugging**: Persistent fixtures enable manual inspection and tooling validation

---

## Requirements Analysis

### Functional Requirements

1. **FR1: Persistent Fixture Storage**
   - Store 3 GGUF fixtures in `ci/fixtures/qk256/`:
     - `qk256_4x256.gguf`: Single-block QK256 tensor [4, 256] (10,816 bytes)
     - `qk256_3x300.gguf`: Multi-block QK256 with tail [3, 300] (10,696 bytes)
     - `bitnet32_2x64.gguf`: BitNet32-F16 tensor [2, 64] (8,832 bytes)
   - Include `SHA256SUMS` file for integrity verification
   - Include `README.md` with fixture metadata and regeneration instructions

2. **FR2: Test Migration**
   - Migrate 12 fixture-based tests in `crates/bitnet-models/tests/qk256_dual_flavor_tests.rs` from in-memory to disk-based loading
   - Preserve existing test logic (only change fixture loading mechanism)
   - Add SHA256 verification as optional pre-flight check

3. **FR3: Feature Gate Control**
   - Maintain `fixtures` feature gate for optional fixture-based testing
   - Gracefully skip tests when fixture files missing (with clear error messages)
   - Support both in-memory (fallback) and disk-based (preferred) modes

### Non-Functional Requirements

1. **NFR1: Performance**
   - Fixture loading must complete in <10ms per test (vs. 200ms in-memory generation)
   - SHA256 verification overhead <5ms per fixture

2. **NFR2: Reproducibility**
   - Fixtures must be deterministic (same seeds → same binary output)
   - Version control must track fixture changes via SHA256SUMS
   - CI builds must fail if fixture integrity compromised

3. **NFR3: Maintainability**
   - Clear regeneration workflow documented in `ci/fixtures/qk256/README.md`
   - Fixtures regenerated via single command: `cargo test test_dump_fixture_for_debug`
   - Integration with existing GGUF validation tooling (`bitnet-cli compat-check`)

---

## Architecture Approach

### Crate-Specific Implementation Strategy

**Primary Crate**: `bitnet-models` (GGUF parsing and quantization detection)

**Affected Files**:
```
ci/fixtures/qk256/
├── qk256_4x256.gguf           # ✅ Already exists (PR #475)
├── qk256_3x300.gguf           # ✅ Already exists (PR #475)
├── bitnet32_2x64.gguf         # ✅ Already exists (PR #475)
├── SHA256SUMS                 # ✅ Already exists (PR #475)
└── README.md                  # ✅ Already exists (PR #475)

crates/bitnet-models/tests/
├── qk256_dual_flavor_tests.rs # Needs migration from in-memory → disk-based
└── helpers/
    └── qk256_fixtures.rs      # Keep for fallback/regeneration
```

**Implementation Pattern**:
```rust
// Current approach (in-memory generation)
#[test]
#[cfg_attr(not(feature = "fixtures"), ignore)]
fn test_qk256_detection() {
    let fixture_bytes = helpers::qk256_fixtures::generate_qk256_4x256(42);
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&fixture_bytes).unwrap();
    // ... test logic
}

// Target approach (disk-based with SHA256 verification)
#[test]
#[cfg(feature = "fixtures")]
fn test_qk256_detection() {
    let fixture_path = helpers::load_fixture_path("qk256_4x256.gguf");
    helpers::verify_fixture_integrity(&fixture_path); // Optional SHA256 check
    // ... test logic
}
```

### Workspace Integration

**Feature Flag Strategy**:
```toml
# crates/bitnet-models/Cargo.toml
[features]
fixtures = []  # Enables disk-based fixture loading

[dev-dependencies]
sha2 = "0.10"  # For SHA256 verification (test-only)
```

**CI Integration**:
```yaml
# .github/workflows/ci.yml
- name: Run fixture-based tests
  run: |
    cargo test -p bitnet-models --test qk256_dual_flavor_tests \
      --no-default-features --features cpu,fixtures

    # Verify fixture integrity
    cd ci/fixtures/qk256 && sha256sum -c SHA256SUMS
```

---

## Quantization Strategy

**Not Applicable**: This spec focuses on test infrastructure, not quantization algorithm changes.

**Fixture Format Validation**:
- Fixtures use existing QK256 and BitNet32-F16 formats (no changes)
- QK256: 256-element blocks, 2-bit packing (64 bytes/block)
- BitNet32-F16: 32-element blocks, F16 scales (10 bytes/block)

---

## GPU/CPU Implementation

**Not Applicable**: Test infrastructure changes only. Fixtures are format-agnostic and used for both CPU and GPU quantization validation.

---

## GGUF Integration

### Format Compatibility

**GGUF Version**: 3 (32-byte tensor alignment)

**Metadata Requirements**:
- Minimal KV pairs: `vocab_size`, `hidden_size`, `block_count`, `gguf.version`
- 2 tensors per fixture: `tok_embeddings.weight`, `output.weight`
- Deterministic seed-based data generation (documented in README)

### Tensor Alignment Validation

**Validation Command**:
```bash
# Verify GGUF v3 compliance and tensor alignment
cargo run -p bitnet-cli --features cpu,full-cli -- \
  compat-check ci/fixtures/qk256/qk256_4x256.gguf

# Expected output:
# Status:    ✓ Valid GGUF
# Version:   3 (supported)
# Tensors:   2
# Alignment: 32-byte (compliant)
```

**Integration Test Example**:
```rust
#[test]
#[cfg(feature = "fixtures")]
fn test_fixture_alignment_compliance() {
    let fixture = helpers::load_fixture("qk256_4x256.gguf");

    // GGUF v3 requires 32-byte tensor alignment
    let header = GgufHeader::parse(&fixture).unwrap();
    assert_eq!(header.version, 3);
    assert_eq!(header.alignment, 32);

    // Verify tensor data starts at 32-byte boundary
    for tensor in header.tensors {
        assert_eq!(tensor.offset % 32, 0, "Tensor {} not aligned", tensor.name);
    }
}
```

---

## Performance Specifications

### Throughput Targets

| Metric | In-Memory (Current) | Disk-Based (Target) | Improvement |
|--------|---------------------|---------------------|-------------|
| Fixture generation | 200ms per suite | N/A (pre-generated) | — |
| Fixture loading | N/A | <10ms per fixture | 20× faster |
| SHA256 verification | N/A | <5ms per fixture | Minimal overhead |
| Total test overhead | 200ms | <15ms | 93% reduction |

### Memory Usage

| Metric | In-Memory | Disk-Based | Delta |
|--------|-----------|------------|-------|
| Peak memory | 2MB (temp buffers) | <100KB (mmap) | 95% reduction |
| Fixture lifetime | Per-test generation | Persistent on disk | — |

**Validation Command**:
```bash
# Benchmark fixture loading performance
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu,fixtures \
  -- --nocapture test_qk256_detection 2>&1 | grep "elapsed"

# Target: <10ms elapsed time for fixture loading
```

---

## Cross-Validation Plan

### Fixture Integrity Validation

**SHA256 Checksum Verification**:
```bash
# CI pre-flight check
cd ci/fixtures/qk256
sha256sum -c SHA256SUMS

# Expected checksums (from PR #475):
# c1568a0a08e38ef2865ce0816bfd2c617e5589c113114cd731e4c5014b7fbb20  bitnet32_2x64.gguf
# 6e5a4f21607c0064affbcb86133627478eb34d812b59807a7123ff386c63bd3e  qk256_3x300.gguf
# a41cc62c893bcf1d4c03c30ed3da12da03c339847c4d564e9e5794b5d4c6932a  qk256_4x256.gguf
```

### Test Migration Validation

**Parity Verification**:
```bash
# 1. Run existing in-memory tests (baseline)
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu \
  -- --nocapture 2>&1 | tee /tmp/in-memory.log

# 2. Run migrated disk-based tests
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu,fixtures \
  -- --nocapture 2>&1 | tee /tmp/disk-based.log

# 3. Compare test outcomes (all must pass)
diff <(grep "test result:" /tmp/in-memory.log) \
     <(grep "test result:" /tmp/disk-based.log)

# Expected: 12 tests passing in both modes
```

---

## Feature Flag Analysis

### Build Configurations

**Default Features**: Empty (following BitNet-rs convention)

**Test Execution Matrix**:
```bash
# Without fixtures feature (in-memory fallback)
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu

# With fixtures feature (disk-based, preferred)
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu,fixtures

# CI configuration (fixtures enabled)
cargo nextest run --profile ci -p bitnet-models \
  --no-default-features --features cpu,fixtures
```

**Feature Interaction**:
- `fixtures` feature is **additive** (no breaking changes)
- Tests gracefully skip when fixture files missing
- In-memory generation remains available as fallback

---

## Testing Strategy

### Unit Tests

**Fixture Integrity Tests** (`crates/bitnet-models/tests/helpers/mod.rs`):
```rust
#[test]
#[cfg(feature = "fixtures")]
fn test_fixture_sha256_verification() {
    let fixtures = ["qk256_4x256.gguf", "qk256_3x300.gguf", "bitnet32_2x64.gguf"];

    for fixture_name in fixtures {
        let path = helpers::load_fixture_path(fixture_name);
        assert!(helpers::verify_fixture_integrity(&path).is_ok(),
                "SHA256 mismatch for {}", fixture_name);
    }
}

#[test]
#[cfg(feature = "fixtures")]
fn test_fixture_gguf_compliance() {
    let fixture = helpers::load_fixture("qk256_4x256.gguf");
    let header = GgufHeader::parse(&fixture).unwrap();

    // GGUF v3 compliance
    assert_eq!(header.version, 3);
    assert_eq!(header.tensor_count, 2);
    assert_eq!(header.alignment, 32);
}
```

### Integration Tests

**Existing Tests to Migrate** (`crates/bitnet-models/tests/qk256_dual_flavor_tests.rs`):
- `test_qk256_4x256_detection` → Load from `qk256_4x256.gguf`
- `test_qk256_3x300_multiblock` → Load from `qk256_3x300.gguf`
- `test_bitnet32_2x64_detection` → Load from `bitnet32_2x64.gguf`
- 9 additional format detection tests

**Total Migration**: 12 tests

### Strict Mode Validation

**Not Applicable**: Strict mode applies to model validation, not fixture loading.

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Fixture corruption** | Low | High | SHA256 verification in CI; version control tracking |
| **Missing fixture files** | Low | Medium | Graceful test skipping; clear error messages |
| **CI storage overhead** | Low | Low | Fixtures total <32KB; negligible Git bloat |
| **Regeneration drift** | Low | Medium | Deterministic seeds; checksum verification post-regen |

### Validation Commands

**Risk Validation**:
```bash
# 1. Verify fixture integrity (corruption detection)
cd ci/fixtures/qk256 && sha256sum -c SHA256SUMS

# 2. Test graceful skipping (missing files)
mv ci/fixtures/qk256/qk256_4x256.gguf /tmp/
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu,fixtures -- test_qk256_detection 2>&1 \
  | grep "skipped.*fixture not found"
mv /tmp/qk256_4x256.gguf ci/fixtures/qk256/

# 3. Verify regeneration determinism
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features fixtures test_dump_fixture_for_debug
cd ci/fixtures/qk256 && sha256sum -c SHA256SUMS
```

---

## Success Criteria

### Measurable Acceptance Criteria

**AC1: Persistent Fixtures**
- ✅ 3 GGUF fixtures exist in `ci/fixtures/qk256/`
- ✅ SHA256SUMS file present with correct checksums
- ✅ README.md documents fixture metadata and regeneration

**Validation**:
```bash
ls -lh ci/fixtures/qk256/
# Expected files: qk256_4x256.gguf, qk256_3x300.gguf, bitnet32_2x64.gguf, SHA256SUMS, README.md

sha256sum -c ci/fixtures/qk256/SHA256SUMS
# Expected: OK for all 3 fixtures
```

**AC2: Test Migration**
- ✅ 12 tests migrated from in-memory → disk-based loading
- ✅ All tests pass with `--features cpu,fixtures`
- ✅ Fixture loading overhead <10ms per test

**Validation**:
```bash
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu,fixtures \
  -- --nocapture 2>&1 | tee /tmp/fixture-tests.log

# Expected: test result: ok. 12 passed; 0 failed
# Expected: fixture loading <10ms per test
```

**AC3: Feature Gate Control**
- ✅ Tests skip gracefully when `fixtures` feature disabled
- ✅ Clear error messages when fixture files missing
- ✅ In-memory fallback remains functional

**Validation**:
```bash
# Without fixtures feature (tests should skip or use fallback)
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu 2>&1 | grep "test result:"

# With missing fixture file
mv ci/fixtures/qk256/qk256_4x256.gguf /tmp/
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu,fixtures 2>&1 \
  | grep -E "(skipped|fixture not found)"
mv /tmp/qk256_4x256.gguf ci/fixtures/qk256/
```

**AC4: CI Integration**
- ✅ CI runs fixture-based tests with integrity checks
- ✅ Fixture corruption fails CI builds
- ✅ Performance improvement measurable (200ms → <15ms)

**Validation**:
```yaml
# .github/workflows/ci.yml integration test
- name: Verify fixture-based tests
  run: |
    cd ci/fixtures/qk256 && sha256sum -c SHA256SUMS
    cargo nextest run --profile ci -p bitnet-models \
      --no-default-features --features cpu,fixtures
```

---

## Performance Thresholds

| Metric | Threshold | Validation Command |
|--------|-----------|-------------------|
| Fixture loading | <10ms per fixture | `time cargo test test_qk256_detection --features fixtures` |
| SHA256 verification | <5ms per fixture | `time sha256sum ci/fixtures/qk256/*.gguf` |
| Total test overhead | <15ms per suite | Compare `in-memory` vs `disk-based` test runs |
| Fixture file size | <12KB per fixture | `ls -lh ci/fixtures/qk256/*.gguf` |

---

## Implementation Notes

### Fixture Regeneration Workflow

**Current Implementation** (PR #475 established this):
```bash
# 1. Generate fixtures to /tmp
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features fixtures test_dump_fixture_for_debug \
  -- --nocapture

# 2. Copy to ci/fixtures/qk256/
cp /tmp/test_qk256_4x256.gguf ci/fixtures/qk256/qk256_4x256.gguf
cp /tmp/test_bitnet32_2x64.gguf ci/fixtures/qk256/bitnet32_2x64.gguf
cp /tmp/test_qk256_3x300.gguf ci/fixtures/qk256/qk256_3x300.gguf

# 3. Update checksums
cd ci/fixtures/qk256 && sha256sum *.gguf > SHA256SUMS

# 4. Verify integrity
sha256sum -c SHA256SUMS
```

### Test Helper API

**Proposed Helper Functions** (`crates/bitnet-models/tests/helpers/mod.rs`):
```rust
#[cfg(feature = "fixtures")]
pub fn load_fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("ci/fixtures/qk256")
        .join(name)
}

#[cfg(feature = "fixtures")]
pub fn verify_fixture_integrity(path: &Path) -> Result<(), String> {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    hasher.update(&bytes);
    let hash = format!("{:x}", hasher.finalize());

    // Compare against SHA256SUMS file
    let sums_path = path.parent().unwrap().join("SHA256SUMS");
    let expected = std::fs::read_to_string(sums_path)
        .map_err(|e| e.to_string())?;

    if !expected.contains(&hash) {
        return Err(format!("SHA256 mismatch for {}", path.display()));
    }
    Ok(())
}
```

---

## BitNet-rs Alignment

### TDD Practices

✅ **Alignment**: Fixtures already generated via TDD tests (`test_dump_fixture_for_debug`)

### Feature-Gated Architecture

✅ **Alignment**: Proper use of `#[cfg(feature = "fixtures")]` guards

### Workspace Structure

✅ **Alignment**: Fixtures stored in `ci/fixtures/` (not in crate directories)

### Cross-Platform Support

✅ **Alignment**: Fixtures are binary-portable (GGUF format is platform-agnostic)

---

## Neural Network References

### Existing Quantization Implementations

**Fixture Generator** (already implemented):
```bash
# Generator location
crates/bitnet-models/tests/helpers/qk256_fixtures.rs

# Generation logic (deterministic seed-based)
pub fn generate_qk256_4x256(seed: u64) -> Vec<u8> {
    // Seed 42 → code 2 (→ +1.0)
    // Generates GGUF v3 with [4, 256] QK256 tensor
}
```

**QK256 Kernel References**:
```bash
# QK256 dequantization kernels (consume fixtures)
find crates/bitnet-models/src/quant/ -name "*qk256*.rs"
# crates/bitnet-models/src/quant/i2s_qk256.rs
# crates/bitnet-models/src/quant/i2s_qk256_avx2.rs
```

### GGUF Compatibility Validation

**Format Specification Reference**:
- GGUF v3: 32-byte tensor alignment
- Metadata schema: Minimal KV pairs (8 keys)
- Tensor data: QK256 (64 bytes/block) or BitNet32-F16 (10 bytes/block)

**Validation Tools**:
```bash
# GGUF format inspection
cargo run -p bitnet-cli --features cpu,full-cli -- \
  compat-check ci/fixtures/qk256/qk256_4x256.gguf --show-kv

# Expected metadata keys:
# - vocab_size: 1000
# - hidden_size: 512
# - block_count: 1
# - gguf.version: 3
```

---

## Related Documentation

- **Fixture README**: `ci/fixtures/qk256/README.md` (already exists from PR #475)
- **QK256 Architecture**: `docs/explanation/i2s-dual-flavor.md`
- **Test Suite Guide**: `docs/development/test-suite.md`
- **GGUF Compatibility**: `docs/reference/quantization-support.md`

---

## Migration Checklist

**Phase 1: Fixture Verification** (5 minutes)
- [ ] Verify fixtures exist: `ls -lh ci/fixtures/qk256/`
- [ ] Run SHA256 verification: `cd ci/fixtures/qk256 && sha256sum -c SHA256SUMS`
- [ ] Validate GGUF format: `cargo run -p bitnet-cli --features cpu,full-cli -- compat-check ci/fixtures/qk256/*.gguf`

**Phase 2: Test Migration** (30 minutes)
- [ ] Add helper functions: `load_fixture_path()`, `verify_fixture_integrity()`
- [ ] Migrate 12 tests from in-memory → disk-based loading
- [ ] Add `#[cfg(feature = "fixtures")]` guards
- [ ] Test with fixtures feature: `cargo test --features cpu,fixtures`
- [ ] Test without fixtures feature: `cargo test --features cpu`

**Phase 3: CI Integration** (10 minutes)
- [ ] Update CI workflow: Add SHA256 verification step
- [ ] Run CI tests: `cargo nextest run --profile ci --features cpu,fixtures`
- [ ] Verify performance improvement: Compare logs

**Phase 4: Documentation** (10 minutes)
- [ ] Update test-suite.md: Add fixture loading section
- [ ] Update CLAUDE.md: Add fixture validation commands
- [ ] Review ci/fixtures/qk256/README.md: Ensure regeneration workflow documented

---

## Status

**Current Phase**: Draft Specification
**Next Steps**: Review and approval
**Estimated Implementation Time**: 1 hour
**Risk Level**: Low (test infrastructure only, no production code changes)

---

**Last Updated**: 2025-10-23
**Spec Author**: BitNet-rs Spec Analyzer Agent
**Review Status**: Pending
