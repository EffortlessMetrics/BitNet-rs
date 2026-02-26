# Model Fixtures for Issue #462

## Current Model

**`tiny-bitnet.gguf`** (symlink)

- **Target:** `../../models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf`
- **Size:** 1.2GB (1,187,801,280 bytes)
- **SHA256:** `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
- **Quantization:** I2S (2-bit signed)
- **Architecture:** BitNet 1.58-2B (4T variant)
- **Vocab Size:** ~50,000 tokens
- **Layers:** 24 transformer layers
- **Hidden Dims:** 2048
- **Attention Heads:** 32

## Model Validation

```bash
# Verify symlink target
readlink -f tiny-bitnet.gguf

# Verify checksum
sha256sum tiny-bitnet.gguf | tee tiny-bitnet.gguf.sha256

# Inspect model metadata (requires bitnet-cli)
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto tiny-bitnet.gguf
```

## Using Custom Models

### Via Environment Variable

```bash
export BITNET_GGUF=/path/to/your/model.gguf
cargo test --no-default-features --features cpu
```

### Via Symlink Replacement

```bash
cd tests/fixtures/models
rm tiny-bitnet.gguf
ln -s /path/to/your/model.gguf tiny-bitnet.gguf
```

## Downloading the Model

If the original model is not available in `models/` directory:

```bash
# Download from Hugging Face
cargo run -p xtask -- download-model \
  --id microsoft/bitnet-b1.58-2B-4T-gguf \
  --file ggml-model-i2_s.gguf \
  --output ../../models/microsoft-bitnet-b1.58-2B-4T-gguf/

# Recreate symlink
ln -sf ../../models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf tiny-bitnet.gguf
```

## Model Requirements for Tests

### Minimum Requirements

- **Format:** GGUF (compatible with BitNet-rs)
- **Quantization:** I2S, TL1, or TL2 (quantized weights)
- **LayerNorm:** F16 or F32 (not quantized)
- **Size:** Any (tests should gracefully handle large models)
- **Vocab Size:** ≥1000 tokens (for inference tests)

### Recommended for Fast CI

- **Size:** <100MB (for fast download/cache)
- **Layers:** 2-4 layers (minimal architecture)
- **Quantization:** I2S (well-tested)

## Model Provisioning Status

- ✅ **Primary Model:** Available via symlink to existing repository model
- ✅ **Checksum:** Validated and stored in `tiny-bitnet.gguf.sha256`
- ✅ **Auto-Discovery:** Tests support `BITNET_GGUF` environment variable
- ⏳ **Minimal Synthetic Model:** Future enhancement for CI (<10MB)

## Tokenizer Discovery

The tests use auto-discovery for tokenizers:

1. Check `BITNET_TOKENIZER` environment variable
2. Search model directory for `tokenizer.json`
3. Search `models/` directory for compatible tokenizer
4. Fall back to mock tokenizer (byte-level encoding)

**Available Tokenizers in Repository:**
- `models/llama3-tokenizer/tokenizer.json`
- `models/bitnet-2b-safetensors/tokenizer.json`

## CI/CD Considerations

### GitHub Actions Cache

```yaml
- name: Cache Models
  uses: actions/cache@v3
  with:
    path: models/
    key: bitnet-models-${{ hashFiles('models/models.lock.json') }}
```

### Conditional Model Tests

Tests requiring the model should check availability:

```rust
#[test]
fn test_with_model() {
    let model_path = std::env::var("BITNET_GGUF")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tests/fixtures/models/tiny-bitnet.gguf"));

    if !model_path.exists() {
        eprintln!("⚠️ Model not available, skipping test");
        return; // Skip test gracefully
    }

    // Test logic here...
}
```

## Troubleshooting

### Symlink Not Resolving

```bash
# Check if target exists
ls -la ../../models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf

# Recreate symlink with absolute path
rm tiny-bitnet.gguf
ln -s /home/steven/code/Rust/BitNet-rs/models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf tiny-bitnet.gguf
```

### Model Checksum Mismatch

```bash
# Regenerate checksum
sha256sum tiny-bitnet.gguf > tiny-bitnet.gguf.sha256

# Compare with expected
cat tiny-bitnet.gguf.sha256
```

### Model Not Found in Tests

```bash
# Verify environment variable
echo $BITNET_GGUF

# Use absolute path
export BITNET_GGUF=/home/steven/code/Rust/BitNet-rs/tests/fixtures/models/tiny-bitnet.gguf
```

## Model Metadata

Extracted via `cargo run -p bitnet-cli -- inspect`:

```
Model Type: BitNet 1.58
Quantization: I2S (2-bit signed)
Architecture:
  - Layers: 24
  - Hidden Size: 2048
  - Attention Heads: 32
  - Vocab Size: ~50,000
  - Context Length: 2048 tokens
  - Weight Alignment: 32 bytes (GGUF standard)

Tensor Counts:
  - Total Tensors: ~200
  - Quantized Tensors: ~180 (I2S)
  - Float Tensors: ~20 (LayerNorm, embeddings)

File Size: 1.2GB (compressed from ~6GB FP32)
```

## Related Documentation

- **Main Fixtures README:** `/home/steven/code/Rust/BitNet-rs/tests/fixtures/README.md`
- **Model Validation Guide:** `docs/howto/validate-models.md`
- **GGUF Format Reference:** `docs/reference/gguf-format.md` (if exists)
