# Getting Started with BitNet Rust

This guide will help you get up and running with BitNet Rust, a production-ready implementation of 1-bit neural network inference with real quantized computation. BitNet.rs eliminates mock fallbacks and implements native I2S, TL1, and TL2 quantization kernels, enabling authentic neural network inference with realistic performance baselines.

## Installation

### Prerequisites

- Rust 1.90.0 or later
- CUDA 11.8+ (optional, for GPU acceleration)
- Python 3.8+ (optional, for Python bindings)

### Install from crates.io

```bash
cargo install bitnet-cli
```

### Build from source

```bash
git clone https://github.com/microsoft/BitNet-rs.git
cd BitNet-rs
cargo build --release --no-default-features --features cpu
```

### Feature flags

BitNet Rust supports several feature flags for customization:

- `cpu`: Enable CPU inference with SIMD optimizations
- `gpu`: Enable CUDA GPU acceleration with device-aware quantization
- `ffi`: Enable C++ FFI bridge for cross-validation
- `crossval`: Enable cross-validation against Microsoft BitNet C++

**Important**: Default features are **empty** - always specify features explicitly.

```bash
# Build with CPU support
cargo build --no-default-features --features cpu

# Build with GPU support
cargo build --no-default-features --features gpu

# Build with both CPU and GPU
cargo build --no-default-features --features "cpu,gpu"
```

## Quick Start

### Using the CLI with Real GGUF Models

1. **Download a real BitNet model with trained weights**:
```bash
# Download official Microsoft BitNet model with I2_S quantization
cargo run -p xtask -- download-model \
    --id microsoft/bitnet-b1.58-2B-4T-gguf \
    --file ggml-model-i2_s.gguf
```

2. **Validate GGUF model and inspect real weights**:
```bash
# Verify GGUF compatibility and tensor completeness
cargo run -p bitnet-cli -- compat-check \
    models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf

# Inspect real model weights and quantization format
cargo run -p bitnet-cli -- inspect \
    --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
```

3. **Run inference with real neural network weights**:
```bash
# Production inference with strict mode (prevents mock fallbacks)
BITNET_STRICT_MODE=1 cargo run -p xtask -- infer \
    --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
    --prompt "Explain quantum computing in simple terms" \
    --deterministic

# Stream generation with device-aware quantization
BITNET_STRICT_MODE=1 cargo run -p xtask -- infer \
    --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
    --prompt "Write a story about AI and humans working together" \
    --stream

# Performance measurement with realistic expectations
# CPU: 10-20 tokens/sec, GPU: 50-100 tokens/sec
BITNET_DETERMINISTIC=1 BITNET_SEED=42 cargo run -p xtask -- infer \
    --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
    --prompt "Benchmark inference performance" \
    --metrics
```

### Using the Rust API

Add BitNet to your `Cargo.toml`:

```toml
[dependencies]
bitnet = "0.1.0"
```

Real GGUF model loading with trained weights:

```rust
use bitnet::prelude::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Load real GGUF model with actual trained neural network weights
    let model = BitNetModel::from_file(
        "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
    ).await?;

    // Verify real weights were loaded (not mock tensors)
    println!("Loaded {} tensors with {} parameters",
             model.tensor_count(), model.parameter_count());

    // Create inference engine with device-aware backend selection
    // Strict mode prevents mock fallbacks for production use
    std::env::set_var("BITNET_STRICT_MODE", "1");

    let engine = InferenceEngine::builder()
        .model(model)
        .backend(Backend::Auto)  // Automatically selects GPU if available
        .quantization(QuantizationType::I2S)  // Use I2_S quantization
        .strict_mode(true)  // Prevent mock inference fallbacks
        .build()?;

    // Configure generation with performance metrics
    let config = GenerationConfig {
        max_new_tokens: 100,
        temperature: 0.7,
        enable_metrics: true,  // Track performance
        ..Default::default()
    };

    // Generate text with real neural network inference
    let response = engine.generate_with_config(
        "Explain the benefits of 1-bit neural networks",
        &config
    ).await?;

    println!("Generated: {}", response.text);

    // Access performance metrics from real quantized inference
    // Realistic performance: CPU 10-20 tok/s, GPU 50-100 tok/s
    if let Some(metrics) = response.metrics {
        println!("Inference time: {:.2}ms", metrics.timing.total);
        println!("Throughput: {:.1} tokens/sec", metrics.throughput.e2e);
        println!("Quantization: {}", metrics.quantization_type);
        println!("Device: {}", metrics.device_info);
    }

    Ok(())
}
```

### Streaming Generation

```rust
use bitnet::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    // Load GGUF model
    let model = BitNetModel::from_file("model.gguf").await?;

    // Create inference engine
    let engine = InferenceEngine::builder()
        .model(model)
        .backend(Backend::Auto)
        .build()?;

    // Configure streaming generation
    let config = GenerationConfig {
        max_new_tokens: 100,
        temperature: 0.7,
        ..Default::default()
    };

    let mut stream = engine.generate_stream_with_config("Tell me a story", &config);

    while let Some(result) = stream.next().await {
        match result {
            Ok(response) => print!("{}", response.text),
            Err(e) => eprintln!("Error: {}", e),
        }
    }

    Ok(())
}
```

## Model Formats

BitNet Rust supports multiple model formats:

### GGUF Format
```bash
# Load GGUF model
bitnet-cli inference --model path/to/model.gguf --prompt "Hello"
```

### SafeTensors Format
```bash
# Load SafeTensors model
bitnet-cli inference --model path/to/model.safetensors --prompt "Hello"
```

### HuggingFace Directory
```bash
# Load a local HuggingFace model directory
bitnet-cli inference --model path/to/hf-model --prompt "Hello"
```

### HuggingFace Hub
```bash
# Load from HuggingFace Hub
bitnet-cli inference --model microsoft/bitnet-b1_58-large --prompt "Hello"
```

## Configuration

### Configuration File

Create a `bitnet.toml` configuration file:

```toml
[model]
default_model = "microsoft/bitnet-b1_58-large"
cache_dir = "~/.cache/bitnet"

[inference]
device = "auto"  # "cpu", "cuda", or "auto"
max_batch_size = 8
kv_cache_size = 2048

[generation]
max_new_tokens = 512
temperature = 0.7
top_p = 0.9
top_k = 50
repetition_penalty = 1.0
```

### Environment Variables

BitNet Rust respects these environment variables:

- `BITNET_MODEL_CACHE`: Model cache directory
- `BITNET_DEVICE`: Default device ("cpu", "cuda", "auto")
- `BITNET_LOG_LEVEL`: Log level ("trace", "debug", "info", "warn", "error")
- `BITNET_STRICT_MODE`: Prevent mock inference fallbacks ("1" enables strict mode)
- `BITNET_DETERMINISTIC`: Enable deterministic inference for reproducible results
- `BITNET_SEED`: Set seed for reproducible inference (works with BITNET_DETERMINISTIC=1)
- `CUDA_VISIBLE_DEVICES`: GPU device selection

## Performance Optimization

### Performance Expectations (Issue #260 Completed)

BitNet.rs provides realistic performance baselines based on real quantized computation without mock fallbacks:

- **CPU Performance**: 10-20 tokens/sec with I2S quantization (real computation, not mock)
- **GPU Performance**: 50-100 tokens/sec with mixed precision acceleration
- **Quantization Accuracy**: I2S ≥99.8%, TL1/TL2 ≥99.6% correlation with FP32
- **Strict Mode**: Use `BITNET_STRICT_MODE=1` to prevent any mock fallbacks

### CPU Optimization

1. **Enable CPU features with strict mode**:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features --features cpu
BITNET_STRICT_MODE=1 bitnet-cli inference --model model.gguf --prompt "Hello"
```

2. **Tune thread count for optimal performance**:
```bash
export RAYON_NUM_THREADS=8
BITNET_STRICT_MODE=1 bitnet-cli inference --model model.gguf --prompt "Hello"
```

### GPU Optimization

1. **Enable mixed precision with strict mode**:
```rust
// Enable strict mode to prevent mock GPU fallbacks
std::env::set_var("BITNET_STRICT_MODE", "1");

let config = InferenceConfig {
    use_mixed_precision: true,
    precision_mode: PrecisionMode::Auto,  // FP16/BF16 based on device capability
    device_aware: true,  // Enable device-aware quantization selection
    ..Default::default()
};
```

2. **Optimize batch size and validate GPU usage**:
```rust
let config = InferenceConfig {
    max_batch_size: 16,  // Adjust based on GPU memory
    enable_gpu_validation: true,  // Validate GPU kernel execution
    ..Default::default()
};
```

## Troubleshooting

### Common Issues

1. **CUDA not found**:
   - Install CUDA 11.8 or later
   - Set `CUDA_HOME` environment variable
   - Build with `--no-default-features --features cli` for CPU-only

2. **Model loading fails**:
   - Check model format compatibility
   - Verify model file integrity
   - Ensure sufficient disk space and memory

3. **Poor performance**:
   - **CRITICAL**: Verify strict mode is enabled: `BITNET_STRICT_MODE=1` to prevent mock fallbacks
   - Check for mock inference warnings in logs (should be eliminated in Issue #260)
   - Enable native CPU features with `RUSTFLAGS="-C target-cpu=native"`
   - Use GPU acceleration if available with proper feature flags
   - Adjust batch size and thread count
   - Expect realistic performance with real quantization: CPU 10-20 tok/s, GPU 50-100 tok/s

### Debug Mode

Enable debug logging:

```bash
RUST_LOG=debug bitnet-cli inference --model model.gguf --prompt "Hello"
```

### Memory Issues

Monitor memory usage:

```bash
bitnet-cli benchmark --model model.gguf --monitor-memory
```

## Next Steps

- Read the [API Reference](reference/api-reference.md) for detailed API documentation
- Check out [Examples](examples/) for more usage patterns
- See [Migration Guide](migration-guide.md) for migrating from Python/C++
- Review [Performance Tuning](performance-tuning.md) for optimization tips

## Getting Help

- [GitHub Issues](https://github.com/microsoft/BitNet-rs/issues)
- [Documentation](https://docs.rs/bitnet)
- [Microsoft BitNet Repository](https://github.com/microsoft/BitNet)