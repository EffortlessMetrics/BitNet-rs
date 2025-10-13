# Getting Started with Tokenizer Auto-Discovery

This tutorial will guide you through using BitNet.rs's intelligent tokenizer discovery system with real GGUF neural network models. You'll learn how to automatically find, download, and integrate tokenizers for production-ready neural network inference with actual trained weights.

## What You'll Learn

- How tokenizer auto-discovery works with neural network models
- Step-by-step setup of tokenizer integration for LLaMA-3, LLaMA-2, and GPT-2
- Using the `--auto-download` flag with xtask inference
- Working with different tokenizer strategies and fallback mechanisms
- Environment variable configuration for production deployments

## Prerequisites

- BitNet.rs workspace properly installed
- Basic familiarity with GGUF model format
- Understanding of neural network tokenization concepts

## Step 1: Basic Tokenizer Discovery

Let's start with a simple example using a LLaMA-2 model:

```bash
# Download a LLaMA-2 compatible model
cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf --file ggml-model-i2_s.gguf

# Run inference with automatic tokenizer discovery
cargo run -p xtask -- infer \
    --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    --prompt "The capital of France is" \
    --auto-download
```

**What happens behind the scenes:**

1. BitNet.rs analyzes the GGUF model metadata
2. Extracts vocabulary size (32,000 for LLaMA-2) and model type ("llama")
3. Checks for co-located tokenizer files in the model directory
4. If not found, automatically downloads the compatible tokenizer from HuggingFace Hub
5. Caches the tokenizer for future use

## Step 2: Understanding Tokenizer Strategies

BitNet.rs uses different strategies based on what it finds:

### Strategy 1: Co-located Tokenizer (Preferred)

When a tokenizer file exists alongside your model:

```bash
# Create a model directory with both model and tokenizer
models/my-model/
├── model.gguf
└── tokenizer.json  # Auto-discovered!
```

```bash
cargo run -p xtask -- infer \
    --model models/my-model/model.gguf \
    --prompt "Hello world"
    # No --auto-download needed - tokenizer found automatically
```

### Strategy 2: Smart Download

For models without co-located tokenizers:

```bash
# This will trigger automatic download
cargo run -p xtask -- infer \
    --model models/standalone-model.gguf \
    --prompt "Hello world" \
    --auto-download
```

**Download process:**
1. Model analysis identifies: LLaMA-3 with 128K vocabulary
2. Downloads from: `meta-llama/Meta-Llama-3-8B` repository
3. Caches as: `~/.cache/bitnet/tokenizers/llama3-128k/tokenizer.json`
4. Future runs use cached version instantly

### Strategy 3: GGUF-Embedded Tokenizer

Some models include tokenizer data in the GGUF metadata:

```bash
# Works automatically with embedded tokenizers
cargo run -p xtask -- infer \
    --model models/model-with-embedded-tokenizer.gguf \
    --prompt "Hello world"
    # No download required - tokenizer extracted from model
```

## Step 3: Large Vocabulary Models (LLaMA-3)

Large vocabulary models require special consideration for performance:

```bash
# LLaMA-3 with 128K vocabulary - GPU recommended
cargo run -p xtask -- infer \
    --model models/llama3-model.gguf \
    --prompt "The future of AI is" \
    --auto-download \
    --features gpu
```

**GPU Acceleration Benefits:**
- 128K+ vocabulary: 10x faster embedding lookup
- Reduced memory pressure on CPU
- Better performance for large batch sizes

**CPU Fallback:**
```bash
# Still works on CPU, but slower for large vocabularies
BITNET_DETERMINISTIC=1 cargo run -p xtask -- infer \
    --model models/llama3-model.gguf \
    --prompt "The future of AI is" \
    --auto-download \
    --no-default-features --features cpu
```

## Step 4: Production Configuration

For production deployments, configure behavior with environment variables:

### Strict Mode (Recommended for Production)

```bash
# Prevent fallback to mock tokenizers
export BITNET_STRICT_TOKENIZERS=1

# Enable deterministic behavior
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42

cargo run -p xtask -- infer \
    --model models/production-model.gguf \
    --prompt "Production inference" \
    --auto-download
```

### Offline Mode

```bash
# Use only cached tokenizers, no downloads
export BITNET_OFFLINE=1

cargo run -p xtask -- infer \
    --model models/cached-model.gguf \
    --prompt "Offline inference"
    # Will fail if tokenizer not cached
```

### Performance Tuning

```bash
# Optimize for throughput
export BITNET_DETERMINISTIC=0  # Allow non-deterministic optimizations
export RAYON_NUM_THREADS=8     # Control CPU parallelism

cargo run -p xtask -- infer \
    --model models/high-throughput-model.gguf \
    --prompt "Fast inference" \
    --auto-download \
    --features gpu
```

## Step 5: Model-Specific Examples

### LLaMA-2 (32K Vocabulary)

Perfect for CPU inference with TL1/TL2 quantization:

```bash
cargo test --no-default-features --features cpu
cargo run -p xtask -- infer \
    --model models/llama2-model.gguf \
    --prompt "Explain quantum computing" \
    --auto-download \
    --no-default-features --features cpu
```

**Expected Output:**
```
[INFO] Discovering tokenizer strategy for llama model (vocab_size: 32000)
[INFO] Can download compatible tokenizer from: meta-llama/Llama-2-7b-hf
[INFO] Downloading tokenizer for repo: meta-llama/Llama-2-7b-hf
[INFO] Successfully downloaded tokenizer: ~/.cache/bitnet/tokenizers/llama2-32k/tokenizer.json
```

### GPT-2 (50K Vocabulary)

Standard BPE tokenization:

```bash
cargo run -p xtask -- infer \
    --model models/gpt2-model.gguf \
    --prompt "The weather today is" \
    --auto-download \
    --no-default-features --features cpu
```

**Expected behavior:**
- Downloads from: `openai-community/gpt2`
- Cache key: `gpt2-50k`
- Works efficiently on CPU

### BitNet Custom Models

Specialized tokenizers for BitNet architectures:

```bash
cargo run -p xtask -- infer \
    --model models/bitnet-custom.gguf \
    --prompt "Neural network quantization" \
    --auto-download \
    --features gpu
```

**Features:**
- Downloads both `tokenizer.json` and `tokenizer.model`
- Optimized for 1-bit neural network inference
- GPU acceleration for large vocabularies

## Step 6: Troubleshooting Common Issues

### Issue 1: No Compatible Tokenizer Found

**Error:**
```
Error: No compatible tokenizer found for transformer model with vocab_size 99999 (strict mode)
```

**Solution:**
```bash
# Check model metadata
cargo run -p bitnet-cli -- compat-check model.gguf

# Try with explicit tokenizer
cargo run -p xtask -- infer \
    --model model.gguf \
    --tokenizer path/to/tokenizer.json \
    --prompt "Test"
```

### Issue 2: Download Failures

**Error:**
```
Error: HTTP error 404: https://huggingface.co/unknown/repo/resolve/main/tokenizer.json
```

**Solution:**
```bash
# Verify model type detection
cargo run -p xtask -- verify --model model.gguf

# Use offline mode with cached tokenizer
export BITNET_OFFLINE=1
cargo run -p xtask -- infer --model model.gguf --prompt "Test"
```

### Issue 3: Vocabulary Size Mismatch

**Warning:**
```
Warning: Vocabulary size mismatch: expected 32000, got 50257
```

**Solution:**
```bash
# Clear cache and re-download
rm -rf ~/.cache/bitnet/tokenizers/
cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Test" \
    --auto-download
```

## Step 7: Advanced Usage Patterns

### Programmatic API Usage

```rust
use bitnet_tokenizers::{TokenizerDiscovery, SmartTokenizerDownload};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Discover strategy
    let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;
    println!("Model type: {}", discovery.model_type());
    println!("Vocab size: {}", discovery.vocab_size());

    let strategy = discovery.discover_tokenizer_strategy()?;

    // Handle different strategies
    match strategy {
        TokenizerStrategy::NeedsDownload(info) => {
            let downloader = SmartTokenizerDownload::new()?;
            let path = downloader.download_tokenizer(&info).await?;
            println!("Downloaded: {}", path.display());
        },
        _ => println!("Strategy: {}", strategy.description()),
    }

    Ok(())
}
```

### Cache Management

```bash
# List cached tokenizers
find ~/.cache/bitnet/tokenizers/ -name "*.json"

# Clear all caches (note: clean-cache takes no arguments)
cargo run -p xtask -- clean-cache
```

### Performance Monitoring

```bash
# Enable detailed logging
RUST_LOG=bitnet_tokenizers=debug cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Performance test" \
    --auto-download

# Monitor cache hit rates
RUST_LOG=bitnet_tokenizers::download=info cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Cache test"
```

## Next Steps

Now that you understand tokenizer auto-discovery:

1. **Read the Architecture Guide**: Learn how the discovery system works internally
2. **Explore How-To Guides**: Advanced configuration and troubleshooting
3. **Check Performance Benchmarks**: Optimize for your specific use case
4. **Integration Examples**: Use with your applications

## Summary

You've learned how to:
- ✅ Use automatic tokenizer discovery with `--auto-download`
- ✅ Understand different tokenizer strategies and when they apply
- ✅ Configure production environments with environment variables
- ✅ Handle model-specific requirements (LLaMA-3, GPT-2, etc.)
- ✅ Troubleshoot common issues
- ✅ Use advanced programmatic APIs

The tokenizer discovery system makes BitNet.rs inference seamless across different neural network architectures while maintaining production-grade reliability and performance.
