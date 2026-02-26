# How to Use QK256 (GGML I2_S) Models

**Overview:** BitNet-rs now supports GGML I2_S format (QK256) models in pure Rust without requiring FFI or C++ dependencies. This guide walks through using QK256 models for inference.

## What is QK256?

QK256 refers to GGML's I2_S quantization format with:
- **Block size:** 256 elements (QK_K = 256 per GGML conventions)
- **Data format:** 64 bytes per block (no per-block scales)
- **Scales:** Stored in separate tensor (1 f32 scale per block)
- **2-bit mapping:** [-2, -1, +1, +2] for signed symmetric quantization
- **Accuracy:** ≥99.8% vs FP32 baseline

BitNet-rs includes pure-Rust kernel support for QK256, enabling fast inference on GGML-compatible models without external dependencies.

## Quick Start

### 1. Build BitNet-rs with CPU Feature

The `cpu` feature includes QK256 support:

```bash
# Clone repository
git clone https://github.com/EffortlessMetrics/BitNet-rs
cd BitNet-rs

# Build with CPU support (includes QK256)
cargo build --release --no-default-features --features cpu
```

### 2. Download a QK256 Model

BitNet GGUF models use GGML I2_S format:

```bash
# Download Microsoft BitNet model (uses QK256 format)
cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf

# Model is saved to:
# ./models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
```

### 3. Verify QK256 Model Loading

Verify that the model is recognized and loaded correctly:

```bash
# Verify model (automatic tokenizer discovery from GGUF)
cargo run -p xtask -- verify --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf

# Or with explicit tokenizer
cargo run -p xtask -- verify \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json
```

**Expected output:** Model loads successfully with QK256 format detected automatically.

### 4. Run Inference

Generate text using QK256 model:

```bash
# Simple inference with auto-detected template
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --prompt "What is machine learning?" \
  --max-tokens 32

# Or use explicit prompt template
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --prompt-template instruct \
  --prompt "Explain quantum computing" \
  --max-tokens 64 \
  --temperature 0.7
```

### 5. Interactive Chat

BitNet-rs provides interactive chat mode with auto-detected templates:

```bash
# Interactive chat with QK256 model
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- chat \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json

# Type your prompts and press Enter. Available chat commands:
# /help - Show available commands
# /clear - Clear conversation history
# /metrics - Display performance metrics
# /exit or /quit - Exit chat mode
```

## Verification: Confirming QK256 Kernel Usage

To verify that QK256 kernels are being used (not FFI fallback):

### 1. Enable Deterministic Inference

Combine strict mode with deterministic inference to ensure real QK256 kernels:

```bash
# Deterministic inference with strict mode (guarantees real kernels)
export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --prompt "Test" \
  --max-tokens 16 \
  --seed 42
```

If this succeeds, pure-Rust QK256 kernel is being used.

### 2. Check Receipt for Kernel IDs

Run a benchmark and inspect the receipt:

```bash
# Generate benchmark receipt
cargo run -p xtask -- benchmark --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --tokens 128

# Verify receipt shows Rust kernels
cargo run -p xtask -- verify-receipt ci/inference.json

# Expected output includes:
# "backend": "cpu"
# "compute": "rust"
# "kernels": ["i2s_qk256", ...] (NOT "cpp_ffi")
```

## Benchmarking QK256 Models

Measure inference performance:

```bash
# CPU benchmark (QK256 kernels)
cargo run -p xtask -- benchmark \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokens 128 \
  --runs 5

# GPU benchmark (if GPU feature available)
cargo build --release --no-default-features --features gpu
cargo run -p xtask -- benchmark \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokens 128 \
  --device cuda
```

## Troubleshooting

### Issue: Model fails to load with "Unsupported format" error

**Solution:** Ensure you're using `cpu` feature which includes QK256 support:

```bash
# Verify features are enabled
cargo build --release --no-default-features --features cpu

# Try model loading again
cargo run -p xtask -- verify --model <model.gguf>
```

### Issue: Very slow inference (suggests FFI fallback)

**Cause:** Strict mode disabled, allowing FFI fallback for missing kernels.

**Solution:** Enable strict mode to force real QK256 kernels:

```bash
export BITNET_STRICT_MODE=1
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model <model.gguf> \
  --prompt "Test" \
  --max-tokens 16

# If this fails, kernel not available for your architecture
# If this succeeds, QK256 kernel is confirmed working
```

### Issue: "Feature 'ggml-i2s-kernel' not enabled"

**Cause:** Pure-Rust QK256 kernel not available (Phase 2 not complete).

**Current Status:** Pure-Rust kernels ARE available - verify cargo features:

```bash
# Check feature gates
cargo tree --features cpu | grep bitnet

# Rebuild with explicit features
cargo clean
cargo build --release --no-default-features --features cpu
```

### Issue: Tokenizer not found

**Cause:** Tokenizer metadata not found in GGUF file.

**Solution:** Provide explicit tokenizer path:

```bash
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model <model.gguf> \
  --tokenizer <tokenizer.json> \
  --prompt "Test" \
  --max-tokens 16
```

Or extract tokenizer from GGUF if available:

```bash
# Check GGUF metadata for tokenizer
cargo run -p bitnet-cli --features cpu,full-cli -- compat-check <model.gguf> --show-kv
```

## Performance Characteristics

### QK256 Kernel Performance

> **MVP Status**: The QK256 MVP uses scalar-only kernels (~0.1 tok/s for 2B models).
> For quick validation, use `--max-new-tokens 4-16`.
> SIMD acceleration is planned for v0.2.0 (targeting ≥3× uplift via AVX2 nibble-LUT + FMA tiling).

Run the benchmark to measure and record actual throughput on your hardware:

```bash
cargo run -p xtask -- benchmark --model <path/to/model.gguf> --tokens 128
cargo run -p xtask -- verify-receipt
```

**Target envelopes (v0.2.0 goals, not current MVP performance):**

| Backend | Target tok/s | Model |
|---------|-------------|-------|
| CPU AVX2 | 15–25 | 2B, batch=1 |
| CPU AVX-512 | 25–35 | 2B, batch=1 |
| CPU NEON (ARM64) | 10–20 | 2B, batch=1 |
| GPU RTX 4090 | 100–150 | 2B, batch=1 |
| GPU A100 | 200–400 | 2B, batch=1 |

*Actual performance depends on model size, batch size, sequence length, and hardware.
Always measure with `cargo run -p xtask -- benchmark` to get a verifiable receipt.*

## Advanced Usage

### Cross-Validation Against C++ Reference

Validate QK256 implementation against Microsoft BitNet C++ reference:

```bash
# Set C++ reference path
export BITNET_CPP_DIR=/path/to/BitNet.cpp
export BITNET_GGUF=/path/to/model.gguf

# Run cross-validation
cargo test -p crossval --no-default-features --features "cpu,ffi,crossval" -- --nocapture

# Output shows Rust vs C++ parity metrics
```

### Strict Quantization Validation

Enforce real quantized kernels (no FP32 fallback):

```bash
# Strict mode prevents any FP32 dequantization fallback
export BITNET_STRICT_MODE=1
export BITNET_STRICT_REQUIRE_QUANTIZATION=1

cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model <model.gguf> \
  --prompt "Test" \
  --max-tokens 16

# Fails if FP32 fallback would occur
# Succeeds if real QK256 kernels execute
```

### Development: Testing New QK256 Kernel Optimizations

Test custom QK256 kernel implementations:

```bash
# Run QK256-specific tests
cargo test -p bitnet-models --no-default-features --features cpu test_qk256

# Benchmark custom kernels
cargo bench -p bitnet-kernels --no-default-features --features cpu --bench qk256_bench

# Profile with flamegraph
cargo install flamegraph
cargo flamegraph --bin bitnet-cli --no-default-features --features cpu,full-cli -- \
  run --model <model.gguf> --prompt "Test" --max-tokens 32
```

## Environment Variables Reference

### BITNET_DISABLE_MINIMAL_LOADER

**Purpose:** Enforce fail-fast behavior when enhanced GGUF loader cannot load model.

**Use cases:**
- CI/CD pipelines: Prevent silent fallback to minimal loader with incorrect defaults (32 layers, 0 kv_heads)
- Parity validation: Ensure enhanced loader stays active for accurate cross-validation
- Production: Fail early on model incompatibilities instead of using degraded defaults

**Example:**

```bash
# Fail-fast if enhanced loader cannot load model
export BITNET_DISABLE_MINIMAL_LOADER=1

cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --prompt "Test" \
  --max-tokens 16

# Without this flag: falls back to minimal loader (may use 32/0 defaults)
# With this flag: fails immediately with descriptive error message
```

**When to use:**
- ✅ CI/CD parity tests (scripts/parity_smoke.sh)
- ✅ Production inference (ensures correct model dimensions)
- ✅ Debugging model loading issues (surfaces real errors)
- ❌ Local development with experimental models (may want fallback)

**Related:** See `scripts/parity_smoke.sh` for production usage pattern.

### Model Loading: Enhanced vs Minimal Loader

BitNet-rs has two GGUF loading paths:

| Loader | Capabilities | When Used |
|--------|-------------|-----------|
| **Enhanced** | Full tensor parsing, QK256 support, accurate config extraction | Default (preferred) |
| **Minimal** | Basic embedding/projection only, mock layer weights, default config values (32 layers, 0 kv_heads) | Fallback on enhanced failure |

**Problem:** Silent fallback to minimal loader can cause inference errors if model dimensions differ from defaults.

**Solution:** Use `BITNET_DISABLE_MINIMAL_LOADER=1` to fail-fast instead of silently degrading.

### Hidden×Hidden K/V Exporter Quirk

Some GGUF exporters emit K/V projections as **[hidden, hidden]** square matrices instead of the correct **[kv_dim, hidden]** shape for GQA (Grouped Query Attention) models.

**Auto-fix:** BitNet-rs weight mapper automatically detects and slices these to the correct shape:

- **Input:** K weight as [hidden_size, hidden_size] (e.g., 2560×2560)
- **Output:** K weight as [kv_dim, hidden_size] (e.g., 640×2560 for n_kv_heads=5, head_dim=128)
- **Method:** Selects first head from each GQA group

**You'll see this log when the fix is applied:**

```
WARN  layer0: K projection has shape [2560, 2560] but expected [640, 2560] (GQA: n_kv_heads=5)
INFO  Slicing K projection to [640, 2560] by selecting first head of each group
```

**No action required** - this is handled automatically. The warning is informational only.

**Technical details:** See regression test `test_kv_slicing_for_gqa` in `crates/bitnet-models/src/weight_mapper.rs`.

## Related Documentation

- **Reference:** [Quantization Support - QK256](../reference/quantization-support.md) - Technical specifications
- **Explanation:** [Dual I2_S Flavor Architecture](../explanation/i2s-dual-flavor.md) - Deep dive on BitNet vs GGML formats
- **Getting Started:** [BitNet-rs Quickstart](../quickstart.md) - 5-minute setup guide
- **How-To:** [Model Validation Guide](./validate-models.md) - Comprehensive validation workflow
- **CLI Reference:** [Command-Line Reference](../reference/cli-reference.md) - Detailed CLI documentation

## What's Next?

- Explore [GPU acceleration](../development/gpu-development.md) for faster inference
- Read about [strict mode validation](./strict-mode-validation-workflows.md) for production deployments
- Check [performance benchmarking guide](../performance-benchmarking.md) for detailed performance analysis
- Review [model compatibility](./validate-models.md) for other quantization formats
