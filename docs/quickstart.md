# BitNet.rs 5-Minute Quickstart

**Get BitNet neural network inference running in under 5 minutes.**

This guide gets you from zero to running BitNet 1-bit quantized neural network inference immediately. For comprehensive development setup, see [development/](development/).

## Prerequisites (1 minute)

```bash
# Check Rust version (1.90.0+ required)
rustc --version

# Clone repository
git clone https://github.com/microsoft/BitNet-rs
cd BitNet-rs
```

## Step 1: Build BitNet.rs (1 minute)

```bash
# CPU inference (fastest setup)
cargo build --release --no-default-features --features cpu

# OR GPU inference (if CUDA available)
cargo build --release --no-default-features --features gpu
```

## Step 2: Download BitNet Model (1 minute)

```bash
# Download Microsoft's 1.58-bit quantized model
cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf --file ggml-model-i2_s.gguf
```

## Step 3: Automatic Tokenizer Discovery (30 seconds)

BitNet.rs automatically discovers and loads tokenizers from GGUF files:

```bash
# Verify GGUF model with automatic tokenizer discovery
cargo run -p xtask -- verify --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf

# Or specify tokenizer explicitly if needed
cargo run -p xtask -- verify --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json
```

**What Just Happened?**
- BitNet.rs extracted tokenizer metadata from GGUF file
- Detected model architecture (BitNet, LLaMA, GPT-2, etc.)
- Resolved vocabulary size (32K, 128K, or custom)
- Applied model-specific tokenizer configuration

## Step 4: Run Neural Network Inference (30 seconds)

```bash
# Generate text with automatic tokenizer discovery
cargo run -p xtask -- infer --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --prompt "BitNet is a neural network architecture that" --deterministic

# Stream inference (real-time generation) with automatic tokenizer
cargo run -p xtask -- infer --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --prompt "Explain 1-bit quantization:" --stream

# Or specify tokenizer explicitly if needed
cargo run -p xtask -- infer --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json --prompt "Test" --deterministic
```

## Step 5: Benchmark Performance (1 minute)

```bash
# Benchmark inference throughput with automatic tokenizer
RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo build --no-default-features --features cpu --release -p xtask
cargo run -p xtask -- benchmark --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --tokens 128
```

**Expected Performance:**
- I2_S quantization: >99% accuracy retention with real transformer computation
- Inference speed: 20-100 tokens/second (CPU), 100-500 tokens/second (GPU)
- Memory usage: ~2GB for 2B parameter model
- Production-ready: Real neural network inference with quantized linear layers, multi-head attention, and KV-cache optimization

## What Just Happened?

You've successfully:
1. **Built BitNet.rs** with device-aware quantization and complete transformer implementation
2. **Downloaded a real BitNet model** (Microsoft's 1.58-bit GGUF) with production GGUF weight loading
3. **Automatic tokenizer discovery** extracted tokenizer from GGUF metadata, detected model architecture, and applied optimal configuration
4. **Verified model compatibility** with comprehensive tensor validation, I2S quantization accuracy (≥99%), and GGUF metadata extraction
5. **Ran production-grade neural network inference** with real transformer weights, multi-head attention, feed-forward layers, and autoregressive generation
6. **Benchmarked performance** achieving 20+ tokens/second with native CPU optimization and device-aware quantization

## Next Steps

- **Tokenizer Discovery**: Learn about automatic tokenizer discovery in [reference/tokenizer-discovery-api.md](reference/tokenizer-discovery-api.md)
- **API Integration**: See [reference/real-model-api-contracts.md](reference/real-model-api-contracts.md) for Rust API usage
- **Model Formats**: Learn about GGUF, I2_S, TL1, TL2 quantization in [explanation/](explanation/)
- **GPU Setup**: Enable CUDA acceleration in [development/gpu-setup-guide.md](development/gpu-setup-guide.md)
- **Troubleshooting**: Common issues in [troubleshooting.md](troubleshooting/troubleshooting.md)

## Quick Commands Reference

```bash
# CPU build and test
cargo build --no-default-features --features cpu
cargo test --workspace --no-default-features --features cpu

# GPU build and test
cargo build --no-default-features --features gpu
cargo test --workspace --no-default-features --features gpu

# Download and verify model (automatic tokenizer discovery)
cargo run -p xtask -- download-model
cargo run -p xtask -- verify --model PATH

# Neural network inference with automatic tokenizer
cargo run -p xtask -- infer --model PATH --prompt "TEXT" --deterministic
cargo run -p xtask -- benchmark --model PATH --tokens 128

# Explicit tokenizer specification (optional)
cargo run -p xtask -- verify --model PATH --tokenizer PATH
cargo run -p xtask -- infer --model PATH --tokenizer PATH --prompt "TEXT"
```

**Total time: ~5 minutes to working BitNet neural network inference**
