# Cross-Validation Parity Testing Guide

This guide provides step-by-step instructions for running the BitNet.cpp parity harness to validate that Rust and C++ implementations produce identical outputs.

## Prerequisites

1. **BitNet C++ Build**: You need a built copy of Microsoft's BitNet.cpp (llama.cpp fork)
2. **GGUF Model**: A valid BitNet GGUF model for testing
3. **FFI Feature**: The `bitnet-sys/ffi` feature must be enabled

## Environment Setup

### 1. Set BitNet C++ Directory

Point to your built BitNet.cpp installation:

```bash
export BITNET_CPP_DIR=/path/to/bitnet.cpp/build
```

**Important:** This should point to the `build` directory (not the source root) where the compiled libraries are located.

### 2. Verify Build Structure

Your `$BITNET_CPP_DIR` should contain:

```
build/
├── 3rdparty/llama.cpp/
│   ├── src/libllama.so (or .dylib/.a)
│   └── include/llama.h
├── lib/
└── ...
```

### 3. Fetch and Configure Model

Use the xtask model fetcher:

```bash
# Fetch model from lockfile
cargo run -p xtask -- fetch-models --lock crossval-models.lock.json | tee /tmp/fetch.json

# Extract model path
export CROSSVAL_GGUF=$(jq -r '.local // .[0].local' /tmp/fetch.json)
```

Or manually set a model path:

```bash
export CROSSVAL_GGUF=/path/to/your/model.gguf
```

### 4. Configure for Determinism

For reproducible results:

```bash
export RAYON_NUM_THREADS=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
```

## Running the Parity Test

### Basic Test Run

```bash
cargo test -p crossval \
  --features crossval,integration-tests \
  -- parity_bitnetcpp --nocapture
```

### With Custom Prompt

```bash
export CROSSVAL_PROMPT="What is the capital of France?"

cargo test -p crossval \
  --features crossval,integration-tests \
  -- parity_bitnetcpp --nocapture
```

### Expected Output

A successful run produces:

```
=== Parity Harness ===
Model: "/path/to/model.gguf"
Prompt: Q: 2+2? A:
Commit: 88a5ffe9
Template: Instruct
Formatted prompt: Q: 2+2? A:
Tokenized 12 tokens (add_bos=true, eos_id=2)
Rust logits shape: [32000]
Rust decoded 8 tokens: [...]
✓ Tokenization exact match
C++ parity check completed:
  Cosine similarity: 0.999876
  Cosine OK (≥0.99): true
  Exact match rate: 1.0000
  No divergence detected
✓ Parity receipt written to: "docs/baselines/2025-10-16/parity-bitnetcpp.json"
```

### Success Criteria

The test passes when:

- ✅ `cpp_available: true`
- ✅ `cosine_similarity >= 0.99` (logits match)
- ✅ `exact_match_rate == 1.0` (tokens match)
- ✅ `first_divergence_step == null` (no divergence)
- ✅ `status: "ok"`

## Inspecting Receipts

Parity receipts are written to `docs/baselines/YYYY-MM-DD/parity-bitnetcpp.json`:

```bash
# View latest receipt
jq . docs/baselines/$(date +%Y-%m-%d)/parity-bitnetcpp.json
```

**Receipt structure:**

```json
{
  "timestamp": "2025-10-16T12:34:56Z",
  "commit": "88a5ffe9",
  "model_path": "/path/to/model.gguf",
  "model_sha256": "abc123...",
  "template": "Instruct",
  "prompt": "Q: 2+2? A:",
  "rust": {
    "token_count": 12,
    "add_bos": true,
    "add_special": false,
    "eos_id": 2,
    "vocab_size": 32000,
    "logits_dim": 32000,
    "decoded_tokens": [4, 29, 29, 15, 18, 13, 2],
    "n_steps": 8
  },
  "parity": {
    "cpp_available": true,
    "cosine_similarity": 0.999876,
    "cosine_ok": true,
    "exact_match_rate": 1.0,
    "first_divergence_step": null,
    "status": "ok"
  },
  "validation": {
    "rust_engine": "production",
    "deterministic": true,
    "threads": 1,
    "seed": 0
  }
}
```

## Troubleshooting

### "BITNET_CPP_DIR not set"

```bash
# Verify environment variable
echo $BITNET_CPP_DIR

# Check directory exists
ls -la $BITNET_CPP_DIR
```

### "cannot open source file llama.h"

This is a VS Code IntelliSense warning - it doesn't affect builds. The build.rs script correctly configures include paths during compilation.

### "Failed to load model"

```bash
# Verify model exists and is readable
ls -lh $CROSSVAL_GGUF
file $CROSSVAL_GGUF

# Check GGUF header
xxd -l 16 $CROSSVAL_GGUF
# Should start with: 4747 5546 (GGUF magic)
```

### "Tokenization mismatch"

This indicates BOS/template handling differences between Rust and C++. Check:

1. Both sides receive the **same formatted prompt**
2. `add_bos` parameter is consistent
3. Template detection is working (check logs)

### Link errors (libllama.so not found)

The build.rs adds RPATH automatically, but if you get runtime link errors:

**Linux:**
```bash
export LD_LIBRARY_PATH=$BITNET_CPP_DIR/3rdparty/llama.cpp/src:$LD_LIBRARY_PATH
```

**macOS:**
```bash
export DYLD_LIBRARY_PATH=$BITNET_CPP_DIR/3rdparty/llama.cpp/src:$DYLD_LIBRARY_PATH
```

### Low cosine similarity (<0.99)

Potential causes:
1. Different quantization formats between Rust and C++
2. Numerical precision differences
3. Model corruption
4. Non-deterministic execution (check thread count and seed)

### First divergence detected

The test shows the first token where outputs differ:

```
First divergence at step: 3
```

This indicates the Rust and C++ decoders disagree. Check:
1. Logits match (cosine should still be high)
2. Sampling strategy is identical (greedy in both)
3. EOS token ID is correct

## CI/CD Integration (Future)

The parity test will be integrated into CI with:

1. **Label-triggered workflow**: Run on PRs with `crossval` label
2. **Nightly baseline updates**: Automatic receipt generation
3. **Model provisioning**: Automatic GGUF download from lockfile
4. **Receipt archival**: Upload to `docs/baselines/`

Example workflow snippet:

```yaml
- name: Set up cross-validation
  run: |
    export BITNET_CPP_DIR=${{ secrets.BITNET_CPP_DIR }}
    export CROSSVAL_GGUF=$(cargo run -p xtask -- fetch-models --lock crossval-models.lock.json | jq -r '.local')

- name: Run parity tests
  run: |
    cargo test -p crossval --features crossval,integration-tests -- parity_bitnetcpp
```

## Advanced Usage

### Custom Thresholds

Edit `crossval/prompts.yaml` to adjust acceptance criteria:

```yaml
thresholds:
  cosine_similarity: 0.99
  exact_match_rate: 1.0
```

### Multi-Model Testing

Test against multiple models in sequence:

```bash
for model in models/*.gguf; do
  export CROSSVAL_GGUF="$model"
  cargo test -p crossval --features crossval,integration-tests -- parity_bitnetcpp --nocapture
done
```

### Debugging Failed Parity

Enable verbose logging:

```bash
RUST_LOG=debug cargo test -p crossval --features crossval,integration-tests -- parity_bitnetcpp --nocapture
```

## Summary

- ✅ **C++ shim compiles** and links against llama.cpp
- ✅ **Function signatures synced** (llama_tokenize parameter mapping fixed)
- ✅ **Template/BOS/EOT contract** is consistent between Rust and C++
- ✅ **Atomic receipt writing** with SHA256 model fingerprinting
- ✅ **Deterministic execution** with configurable seed and threads

The parity harness is production-ready for #468!
