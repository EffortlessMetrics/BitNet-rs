# QK256 GGUF Test Fixtures

Minimal GGUF v3 test fixtures for BitNet-rs QK256 (GGML I2_S) and BitNet32-F16 quantization format validation.

## Fixture Inventory

| File | Size | Description | Tensor Shape | Quantization Format |
|------|------|-------------|--------------|---------------------|
| `qk256_4x256.gguf` | 10,816 bytes | Single-block QK256 tensor | [4, 256] | QK256 (256-elem blocks, 64 bytes/block) |
| `bitnet32_2x64.gguf` | 8,832 bytes | Two-block BitNet32-F16 | [2, 64] | BitNet32-F16 (32-elem blocks, 10 bytes/block) |
| `qk256_3x300.gguf` | 10,696 bytes | Multi-block QK256 with tail | [3, 300] | QK256 (2 blocks: 256 + 44 tail) |

## Purpose

These fixtures support integration tests for:

1. **Format Detection**: QK256 vs BitNet32-F16 detection via tensor size analysis
2. **Block Handling**: Single-block (256 cols), multi-block with tail (300 cols)
3. **Parser Validation**: GGUF v3 compliance (32-byte tensor alignment, minimal metadata)
4. **Deterministic Testing**: Fixed seed values ensure reproducible test results

## Generation Details

- **Generator**: `crates/bitnet-models/tests/helpers/qk256_fixtures.rs`
- **GGUF Version**: 3
- **Tensor Count**: 2 per fixture (`tok_embeddings.weight` + `output.weight`)
- **Metadata**: Minimal required keys (vocab_size=1000, hidden_size=512, block_count=1)
- **Alignment**: 32-byte GGUF v3 compliance for tensor data

### Deterministic Seeds

- `qk256_4x256.gguf`: seed=42 → code=2 (→ +1.0)
- `bitnet32_2x64.gguf`: seed=43 → code=3 (→ +2.0)
- `qk256_3x300.gguf`: seed=44 → code=0 (→ -2.0)

## Verification

### SHA256 Checksums

```bash
sha256sum -c SHA256SUMS
```

Expected checksums:

```
c1568a0a08e38ef2865ce0816bfd2c617e5589c113114cd731e4c5014b7fbb20  bitnet32_2x64.gguf
6e5a4f21607c0064affbcb86133627478eb34d812b59807a7123ff386c63bd3e  qk256_3x300.gguf
a41cc62c893bcf1d4c03c30ed3da12da03c339847c4d564e9e5794b5d4c6932a  qk256_4x256.gguf
```

### GGUF Parser Validation

```bash
# Validate fixture format
cargo run -p bitnet-cli --features cpu,full-cli -- compat-check ci/fixtures/qk256/qk256_4x256.gguf

# Expected output:
# Status:    ✓ Valid GGUF
# Version:   3 (supported)
# Tensors:   2
# KV pairs:  8
```

## Regeneration

To regenerate fixtures with updated generator logic:

```bash
# Generate fixtures to /tmp
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features fixtures test_dump_fixture_for_debug -- --nocapture

# Copy to ci/fixtures/qk256/
cp /tmp/test_qk256_4x256.gguf ci/fixtures/qk256/qk256_4x256.gguf
cp /tmp/test_bitnet32_2x64.gguf ci/fixtures/qk256/bitnet32_2x64.gguf
cp /tmp/test_qk256_3x300.gguf ci/fixtures/qk256/qk256_3x300.gguf

# Update checksums
cd ci/fixtures/qk256 && sha256sum *.gguf > SHA256SUMS
```

## Usage in Tests

### Current Approach (In-Memory Generation)

Tests currently use in-memory fixture generation via `helpers::qk256_fixtures`:

```rust
use helpers::qk256_fixtures;

#[test]
#[cfg_attr(not(feature = "fixtures"), ignore = "Requires real or generated GGUF fixtures")]
fn test_qk256_detection() {
    let fixture_bytes = helpers::qk256_fixtures::generate_qk256_4x256(42);
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&fixture_bytes).unwrap();
    // ... test logic
}
```

### Disk-Based Fixtures (Optional)

For CI/CD environments preferring disk-based fixtures:

```rust
use std::path::Path;

#[test]
#[cfg(feature = "fixtures")]
fn test_qk256_from_disk() {
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("ci/fixtures/qk256/qk256_4x256.gguf");

    assert!(fixture_path.exists(), "Fixture file not found");
    // ... test logic
}
```

## QK256 Format Details

### QK256 (GGML I2_S)

- **Block size**: 256 elements
- **Packed bytes**: 64 bytes/block (2 bits/element)
- **Code mapping**: 0 → -2.0, 1 → -1.0, 2 → +1.0, 3 → +2.0
- **Row stride**: `ceil(cols/256) × 64` bytes
- **Size formula**: `rows × ceil(cols/256) × 64` bytes

### BitNet32-F16

- **Block size**: 32 elements
- **Packed bytes**: 10 bytes/block (8 bytes data + 2 bytes F16 scale)
- **Code mapping**: Same as QK256, scaled by F16 scale factor
- **Row stride**: `ceil(cols/32) × 10` bytes
- **Size formula**: `rows × ceil(cols/32) × 10` bytes

## Related Documentation

- **Generator Implementation**: `crates/bitnet-models/tests/helpers/qk256_fixtures.rs`
- **Integration Tests**: `crates/bitnet-models/tests/qk256_dual_flavor_tests.rs`
- **QK256 Kernels**: `crates/bitnet-models/src/quant/i2s_qk256.rs`
- **Dual Flavor Spec**: `docs/explanation/i2s-dual-flavor.md`

## Maintenance Notes

- **Version Control**: Fixtures are committed to repository for deterministic CI
- **Size**: All fixtures < 12KB (minimal overhead for version control)
- **Updates**: Regenerate when GGUF format or metadata requirements change
- **Validation**: Run `sha256sum -c SHA256SUMS` before/after updates to detect drift
