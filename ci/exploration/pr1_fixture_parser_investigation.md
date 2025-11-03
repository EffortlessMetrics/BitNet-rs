# PR1 Fixture Parser Investigation

## Problem
GGUF fixtures generated for QK256 dual-flavor tests fail to load with error:
"Failed to parse GGUF file with both enhanced and minimal parsers"

## Root Cause Analysis

### Findings
1. **Fixture structure is valid**: `compat-check` CLI successfully validates generated fixtures
2. **GgufReader can parse**: The enhanced GGUF reader (used by compat-check) successfully reads fixture structure
3. **Both parsers fail in tests**: `load_gguf_full()` reports both enhanced AND minimal parsers failed

### Changes Made
1. **Fixed tensor naming**: Changed from test-specific names (`qk256_4x256_weight`) to canonical GGUF names (`tok_embeddings.weight`, `output.weight`)
   - Minimal parser requires specific tensor names for embeddings and output layers
   
2. **Fixed tensor name padding**: Applied correct 8-byte alignment padding after tensor names in GGUF format

3. **Added required metadata**:
   - `bitnet-b1.58.block_count`: Prevents layer discovery from tensor names (would fail with no layer tensors)
   - `bitnet-b1.58.attention.head_count`: Required by config extraction
   - `bitnet-b1.58.attention.head_count_kv`: Required by config extraction  
   - `bitnet-b1.58.feed_forward_length`: Required by config extraction

4. **Updated test assertions**: Changed to expect canonical tensor names

### Current Status
- Fixture generation produces valid GGUF v3 files (validated by compat-check)
- Fixture size: 9056 bytes with complete metadata (8 KV pairs, 2 tensors)
- Tests still fail with "both parsers failed" error

### Hypothesis
The issue is likely in the interaction between:
1. `load_gguf_enhanced()` - May fail due to missing layer tensors or other validation
2. `load_gguf_minimal()` → `load_two()` - May fail due to tensor shape expectations

The minimal parser expects:
- Token embeddings with shape [vocab, dim] or [dim, vocab]
- LM head with compatible shape
- Current fixtures use [4, 256] for both tensors

### Next Steps
1. Add detailed error logging to `load_gguf_enhanced()` and `load_gguf_minimal()` to see which parser fails first and why
2. Verify tensor shape expectations in `pick_tensors()` and `load_two()` 
3. Consider if minimal fixtures need mock layer tensors or if config validation is too strict
4. Test with different tensor shapes that match real model expectations (e.g., [1000, 512] for vocab=1000, hidden=512)

### Files Modified
- `crates/bitnet-models/tests/helpers/qk256_fixtures.rs`: Fixture generator with canonical names and complete metadata
- `crates/bitnet-models/tests/qk256_dual_flavor_tests.rs`: Updated test assertions for canonical tensor names

### Verification
```bash
# Generate and validate fixture
cargo test -p bitnet-models --test qk256_dual_flavor_tests --features fixtures,cpu test_dump_fixture_for_debug
cargo run -p bitnet-cli --features cpu,full-cli -- compat-check /tmp/test_generated_fixture.gguf
# Output: ✓ Valid GGUF (Version 3, 2 tensors, 8 KV pairs)

# Tests still fail
cargo test -p bitnet-models --test qk256_dual_flavor_tests --features fixtures,cpu
# Output: 3 tests fail with "both parsers failed"
```
