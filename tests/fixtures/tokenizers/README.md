# Tokenizer Test Fixtures

This directory contains tokenizer fixtures for TokenizerAuthority E2E integration tests.

## Purpose

These fixtures enable comprehensive end-to-end testing of the parity-both command's
TokenizerAuthority cross-lane validation system without requiring live model downloads
during test execution.

## Fixtures

### valid_tokenizer_a.json
- **Purpose**: Reference LLaMA-3 tokenizer (128000 vocab)
- **Source**: Copy of microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json
- **Usage**: Happy path tests (TC1), hash determinism tests (TC5)

### valid_tokenizer_b.json
- **Purpose**: Byte-identical clone of valid_tokenizer_a.json
- **Source**: Exact copy of valid_tokenizer_a.json
- **Usage**: Hash determinism tests (TC5.2) - verifies same file → same hash

### different_vocab_size.json
- **Purpose**: Modified tokenizer with 64000 vocab (vs 128000)
- **Source**: Modified version of valid_tokenizer_a.json with truncated vocab
- **Usage**: Hash divergence tests (TC5.3) - verifies different vocab → different hash

### corrupted.json
- **Purpose**: Malformed JSON (truncated content)
- **Source**: First 500 bytes of valid_tokenizer_a.json
- **Usage**: Error handling tests (TC4.2) - verifies graceful failure

## Regeneration

To regenerate all fixtures from the reference model:

```bash
# Ensure model is downloaded
cargo run -p xtask -- download-model

# Generate fixtures (requires jq for JSON manipulation)
./scripts/generate_tokenizer_fixtures.sh
```

## Requirements

- **Model**: `microsoft-bitnet-b1.58-2B-4T-gguf` downloaded via xtask
- **jq**: Required for JSON manipulation (different_vocab_size.json generation)
- **Disk**: ~10MB total for all fixtures

## Test Coverage

These fixtures support 12 E2E integration tests in `xtask/tests/tokenizer_authority_e2e_tests.rs`:

- **TC1: Happy Path** (4 tests) - valid_tokenizer_a.json
- **TC2: Mismatch Detection** (3 tests) - validation logic tests (in-process)
- **TC3: Schema v2 Compatibility** (2 tests) - valid_tokenizer_a.json
- **TC4: Edge Cases** (3 tests) - corrupted.json, missing files
- **TC5: Hash Determinism** (3 tests) - all fixtures

## Specification Reference

- **Test Spec**: `docs/specs/tokenizer-authority-validation-tests.md`
- **Feature Spec**: `docs/specs/parity-both-preflight-tokenizer-integration.md`
- **Implementation**: `crossval/src/receipt.rs`, `xtask/src/crossval/parity_both.rs`

## Notes

- Fixtures are **read-only** during tests - tests copy to temp directories
- All tests use `#[serial(bitnet_env)]` to prevent race conditions
- Fixture generation is **idempotent** - safe to run multiple times
- Corrupted.json is **intentionally malformed** - do not fix

## Maintenance

When updating the reference model (microsoft-bitnet):
1. Re-run fixture generation script
2. Verify all E2E tests still pass
3. Update this README if fixture structure changes

Last generated: 2025-10-27 (placeholder - script needs implementation)
