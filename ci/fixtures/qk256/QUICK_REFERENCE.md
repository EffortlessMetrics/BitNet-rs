# QK256 Fixture Quick Reference

Quick command reference for working with BitNet.rs QK256 test fixtures.

## Fixture Files

```
ci/fixtures/qk256/
├── qk256_4x256.gguf    # 10,816 bytes [4×256] single-block
├── bitnet32_2x64.gguf  #  8,832 bytes [2×64]  two-block BitNet32-F16
└── qk256_3x300.gguf    # 10,696 bytes [3×300] multi-block with tail
```

## Verify Checksums

```bash
cd ci/fixtures/qk256
sha256sum -c SHA256SUMS
```

## Inspect Fixtures

```bash
# Validate GGUF format
cargo run -p bitnet-cli --features cpu,full-cli -- \
  compat-check ci/fixtures/qk256/qk256_4x256.gguf

# Inspect all fixtures
for f in ci/fixtures/qk256/*.gguf; do \
  cargo run -p bitnet-cli --features cpu,full-cli -- compat-check "$f"; \
done
```

## Run Tests

```bash
# In-memory fixture generation tests (default - fast)
cargo test -p bitnet-models --test qk256_dual_flavor_tests --no-default-features --features fixtures

# Disk-based fixture loading tests
cargo test -p bitnet-models --test qk256_fixture_loader_tests --no-default-features --features fixtures

# Integration kernel tests
cargo test -p bitnet-models --test qk256_integration --no-default-features --features cpu

# Run all QK256 tests
cargo test -p bitnet-models --no-default-features --features fixtures,cpu qk256
```

## Regenerate Fixtures

```bash
# Step 1: Generate to /tmp
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features fixtures test_dump_fixture_for_debug -- --nocapture

# Step 2: Copy to fixture directory
cp /tmp/test_qk256_4x256.gguf ci/fixtures/qk256/qk256_4x256.gguf
cp /tmp/test_bitnet32_2x64.gguf ci/fixtures/qk256/bitnet32_2x64.gguf
cp /tmp/test_qk256_3x300.gguf ci/fixtures/qk256/qk256_3x300.gguf

# Step 3: Update checksums
cd ci/fixtures/qk256 && sha256sum *.gguf > SHA256SUMS

# Step 4: Verify with loader tests
cargo test -p bitnet-models --test qk256_fixture_loader_tests --no-default-features --features fixtures
```

## Use in Tests

### Option 1: In-Memory Generation (Fast)

```rust
use helpers::qk256_fixtures;

let fixture_bytes = qk256_fixtures::generate_qk256_4x256(42);
let mut file = NamedTempFile::new().unwrap();
file.write_all(&fixture_bytes).unwrap();
```

### Option 2: Disk-Based Loading (Deterministic)

```rust
use helpers::fixture_loader;

let fixture_path = fixture_loader::fixture_path("qk256_4x256.gguf");
assert!(fixture_loader::verify_checksum("qk256_4x256.gguf",
    fixture_loader::checksums::QK256_4X256));
```

## Fixture Checksums

```
c1568a0a08e38ef2865ce0816bfd2c617e5589c113114cd731e4c5014b7fbb20  bitnet32_2x64.gguf
6e5a4f21607c0064affbcb86133627478eb34d812b59807a7123ff386c63bd3e  qk256_3x300.gguf
a41cc62c893bcf1d4c03c30ed3da12da03c339847c4d564e9e5794b5d4c6932a  qk256_4x256.gguf
```

## Related Files

- **Generator**: `crates/bitnet-models/tests/helpers/qk256_fixtures.rs`
- **Loader**: `crates/bitnet-models/tests/helpers/fixture_loader.rs`
- **Documentation**: `ci/fixtures/qk256/README.md`
- **Full Report**: `QK256_FIXTURE_INTEGRATION_REPORT.md`
