# BitNet.rs Validation Infrastructure

## Overview

BitNet.rs implements a comprehensive validation framework to ensure bit-for-bit correctness with the C++ reference implementation. This document describes the validation infrastructure and how to verify correctness.

## Key Components

### 1. Strict Execution Modes

#### `--strict-mapping`
- **Purpose**: Fail if any tensor names cannot be mapped
- **Behavior**: Returns error on unmapped tensors instead of warning
- **Validation**: Ensures all model weights are correctly recognized
- **Output**: JSON shows `"unmapped": 0` when successful

#### `--strict-tokenizer`
- **Purpose**: Require real tokenizer, no mock fallback
- **Behavior**: Fails if tokenizer cannot be loaded from GGUF or external file
- **Validation**: Ensures real tokenization, not placeholders
- **Output**: JSON shows actual tokenizer type (sentencepiece, etc.)

### 2. Real Tokenizer Implementation

#### SentencePiece Integration
```rust
// Real encode/decode, no placeholders
encode("text") -> Vec<PieceWithId> -> Vec<u32>
decode(&[u32]) -> String
```

#### GGUF Tokenizer Extraction
- Reads `tokenizer.ggml.model` from GGUF metadata
- Handles both binary and U8 array storage formats
- Falls back to external file if not embedded

### 3. Tensor Name Mapping

#### Dry-Run Validation
```rust
// Test mapping without loading tensors
let unmapped = dry_run_remap_names(tensor_names);
assert!(unmapped.is_empty());
```

#### Supported Mappings
- Standard Transformers: `attn_q`, `ffn_gate`, etc.
- BitNet-specific: `attn_sub_norm`, `ffn_sub_norm`
- LLaMA-style: `attention.wq`, `feed_forward.w1`

### 4. JSON Output Metrics

```json
{
  "prompt": "Input text",
  "ids": [token_ids],
  "text": "Generated output",
  "first_token_ms": 123,
  "tok_per_sec": 45.6,
  "total_ms": 1000,
  "counts": {
    "n_kv": 256,      // GGUF metadata keys
    "n_tensors": 163, // Total tensor count
    "unmapped": 0     // Must be 0 in strict mode
  },
  "tokenizer": {
    "type": "sentencepiece",
    "bos": 1,
    "eos": 2
  }
}
```

### 5. A/B Token ID Comparison

The `scripts/ab-smoke.sh` script compares token IDs between Rust and C++:

```bash
# With embedded tokenizer
./scripts/ab-smoke.sh models/model.gguf

# With external tokenizer (required for MS BitNet)
./scripts/ab-smoke.sh models/model.gguf tokenizer.model
```

**Validation Criteria:**
- Token IDs must match exactly
- Deterministic execution (SEED=42)
- Same temperature (0.0) for both engines
- PASS: ≥2/3 prompts have matching IDs

## Validation Workflow

### Quick Validation
```bash
# Run validation suite
./scripts/validate-strict.sh
```

### Full Cross-Validation
```bash
# Set environment
export BITNET_GGUF=/path/to/model.gguf
export RAYON_NUM_THREADS=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42

# Build
cargo build --release --no-default-features --features cpu

# Run with strict modes
./target/release/bitnet run \
  --model $BITNET_GGUF \
  --prompt "Test prompt" \
  --max-new-tokens 16 \
  --temperature 0 \
  --strict-mapping \
  --strict-tokenizer \
  --json-out /tmp/output.json

# Verify
jq '.counts.unmapped' /tmp/output.json  # Must be 0
```

### Microsoft BitNet Validation
```bash
# MS BitNet requires external tokenizer
export BITNET_MS_MODEL=models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
export MS_TOKENIZER=models/tokenizer.model

# Test mapping
cargo test ms_bitnet_names_map_clean

# A/B comparison
./scripts/ab-smoke.sh $BITNET_MS_MODEL $MS_TOKENIZER
```

## Performance Baselines

### CI Gates
- **Speed**: Must be ≥95% of baseline tok/s
- **Memory**: Must be ≤103% of baseline RSS
- **Correctness**: Token IDs must match C++

### Baseline Metrics (ci/baseline.json)
```json
{
  "tok_per_sec": 50.0,
  "max_rss_mb": 512,
  "first_token_ms": 100
}
```

## Test Coverage

### Unit Tests
- `sp_roundtrip`: SentencePiece encode/decode
- `ms_bitnet_names_map_clean`: Tensor mapping validation
- `gguf_reader_tests`: GGUF parsing and metadata extraction

### Integration Tests
- Cross-validation against C++ (crossval/)
- Streaming inference validation
- Multi-batch correctness

### Nightly Tests
- Full Microsoft BitNet validation
- Perplexity scoring (when implemented)
- Memory leak detection

## Troubleshooting

### Common Issues

1. **"Unmapped tensors found"**
   - Check model architecture matches supported types
   - Verify GGUF version compatibility
   - Use `--allow-mock` for unsupported models

2. **"Failed to load tokenizer"**
   - Ensure GGUF contains tokenizer.ggml.model
   - Provide external tokenizer with `--tokenizer`
   - Check tokenizer format (SentencePiece .model)

3. **"Token IDs don't match"**
   - Verify BOS/EOS token handling
   - Check tokenizer version compatibility
   - Ensure deterministic mode is enabled

### Debug Commands

```bash
# Check GGUF metadata
./target/release/bitnet compat-check model.gguf

# Extract tokenizer
./target/release/bitnet extract-tokenizer model.gguf output.model

# Dump token IDs
./target/release/bitnet run --model model.gguf \
  --prompt "test" --dump-ids --max-new-tokens 1
```

## Future Enhancements

### Planned Features
- [ ] Perplexity scoring (`score` subcommand)
- [ ] Mean NLL validation (±1e-2 tolerance)
- [ ] Batch inference validation
- [ ] CUDA kernel correctness tests

### Validation Targets
- TinyLlama: Full compatibility ✓
- Microsoft BitNet: Tensor mapping ✓
- Llama 3: In progress
- Custom architectures: Extensible framework

## Summary

The BitNet.rs validation infrastructure ensures:
1. **Zero unmapped tensors** in strict mode
2. **Real tokenization** with SentencePiece
3. **Exact token ID matching** with C++
4. **Performance within 5%** of baseline
5. **Comprehensive test coverage** at all levels

This infrastructure provides confidence that BitNet.rs is a correct, performant drop-in replacement for bitnet.cpp.