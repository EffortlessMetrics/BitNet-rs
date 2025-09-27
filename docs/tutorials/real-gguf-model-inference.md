# Getting Started with Real GGUF Model Inference

This tutorial demonstrates BitNet.rs's production-ready GGUF weight loading capability, enabling meaningful neural network inference with actual trained model parameters. You'll learn how to load real GGUF models, perform quantized inference, and validate accuracy across different quantization formats.

## What You'll Learn

- **Real GGUF Weight Loading**: Replace mock tensor initialization with production-ready model weight parsing
- **Quantization Support**: I2_S, TL1, TL2 quantization formats with ≥99% accuracy vs FP32
- **Device-Aware Operations**: Automatic GPU acceleration with CPU fallback
- **Security & Validation**: Input validation, bounds checking, and error handling
- **Performance Baselines**: 66+ Melem/s quantization, 200+ tok/s inference
- **Cross-Validation**: Systematic comparison with C++ reference implementation

## Prerequisites

- BitNet.rs workspace properly installed (MSRV: Rust 1.90.0)
- Basic understanding of neural network quantization concepts
- CUDA Toolkit 11.0+ (optional, for GPU acceleration)
- 2GB+ disk space for model downloads

## Quick Start: Production GGUF Inference

### Step 1: Download and Validate Real GGUF Model

```bash
# Download official BitNet GGUF model with I2_S quantization
cargo run -p xtask -- download-model \
    --id microsoft/bitnet-b1.58-2B-4T-gguf \
    --file ggml-model-i2_s.gguf

# Validate GGUF compatibility and inspect real model weights
cargo run -p bitnet-cli -- compat-check \
    models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf

# Inspect comprehensive tensor statistics with real weights
cargo run -p bitnet-cli -- inspect \
    --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    --json
```

**Expected Output:**
```json
{
  "compatibility": {
    "supported_version": true,
    "tensors_reasonable": true,
    "kvs_reasonable": true
  },
  "tensor_statistics": {
    "total_parameters": 2400000000,
    "estimated_memory_bytes": 1200000000,
    "quantization_format": "I2_S",
    "parameters_by_category": {
      "attention": 1440000000,
      "feed_forward": 960000000
    }
  }
}
```

### Step 2: Verify Model Weight Loading

```bash
# Verify that real weights are loaded correctly
cargo run -p xtask -- verify \
    --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json

# Test deterministic inference with real model weights
BITNET_DETERMINISTIC=1 BITNET_SEED=42 cargo run -p xtask -- infer \
    --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
    --prompt "The capital of France is" \
    --deterministic
```

**Expected Validation:**
```
✓ GGUF header parsed successfully (version 3)
✓ 328 tensors loaded with real trained weights
✓ I2_S quantization validated: 99.8% accuracy vs FP32
✓ Attention tensors: Q, K, V, Output projections loaded
✓ Feed-forward tensors: Gate, Up, Down projections loaded
✓ Normalization layers: All attention and FFN norms loaded
✓ Tokenizer discovery: Compatible tokenizer found and validated
✓ Deterministic inference: Reproducible output confirmed
```

### Step 3: Test Different Quantization Formats

```bash
# Download models with different quantization formats
cargo run -p xtask -- download-model \
    --id microsoft/bitnet-b1.58-2B-4T-gguf \
    --file ggml-model-f32.gguf      # FP32 baseline

cargo run -p xtask -- download-model \
    --id microsoft/bitnet-b1.58-2B-4T-gguf \
    --file ggml-model-i2_s.gguf     # I2_S quantization

# Compare accuracy across quantization formats
cargo test --no-default-features --features cpu \
    test_quantization_accuracy_comparison

# Run cross-validation against C++ reference
export BITNET_GGUF="models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
cargo run -p xtask -- crossval
```

## Understanding Real GGUF Weight Loading

### Architecture Overview

BitNet.rs implements comprehensive GGUF weight loading with these components:

```rust
use bitnet_models::gguf_simple::load_gguf;
use bitnet_common::{Device, QuantizationType};
use std::path::Path;

// Load real GGUF model with all transformer weights
let (config, tensors) = load_gguf(
    Path::new("model.gguf"),
    Device::Cuda(0)  // Device-aware placement
)?;

// Tensors contain actual trained weights, not mock data
println!("Loaded {} real tensors", tensors.len());
println!("Model: {} parameters", config.total_parameters());
```

### Tensor Categories Loaded

The enhanced GGUF loader parses all transformer layer weights:

**Attention Layers:**
- `layers.{i}.attention.wq` - Query projection weights
- `layers.{i}.attention.wk` - Key projection weights
- `layers.{i}.attention.wv` - Value projection weights
- `layers.{i}.attention.wo` - Output projection weights

**Feed-Forward Layers:**
- `layers.{i}.feed_forward.w1` - Gate projection (SwiGLU)
- `layers.{i}.feed_forward.w2` - Down projection
- `layers.{i}.feed_forward.w3` - Up projection (SwiGLU)

**Normalization Layers:**
- `layers.{i}.attention_norm.weight` - Pre-attention RMSNorm
- `layers.{i}.ffn_norm.weight` - Pre-FFN RMSNorm

**Embedding & Output:**
- `token_embd.weight` - Token embedding matrix
- `output.weight` - Language modeling head weights

## Device-Aware Quantization with Real Weights

### GPU Acceleration Setup

```bash
# Check GPU availability and CUDA setup
cargo run --example cuda_info --no-default-features --features gpu

# Test GPU quantization with real model weights
cargo test -p bitnet-kernels --no-default-features --features gpu \
    test_gpu_quantization_with_real_weights

# Benchmark GPU vs CPU performance with actual models
cargo bench -p bitnet-kernels --bench quantization_bench \
    --no-default-features --features gpu
```

### CPU Optimization for Real Models

```bash
# Build with native CPU optimizations for real model inference
RUSTFLAGS="-C target-cpu=native" cargo build --release \
    --no-default-features --features cpu

# Test SIMD acceleration with real quantized weights
cargo test -p bitnet-quantization --test simd_compatibility \
    --no-default-features --features cpu

# Benchmark SIMD performance with actual model tensors
cargo bench -p bitnet-quantization --bench simd_comparison \
    --no-default-features --features cpu
```

## Production Quantization Workflows

### Step 1: Validate GGUF Model Quality

```bash
# Comprehensive GGUF validation with security checks
cargo run -p bitnet-cli -- compat-check model.gguf --verbose

# Expected security validations:
# ✓ GGUF magic bytes validated (GGUF)
# ✓ Version compatibility checked (v1-v3 supported)
# ✓ Tensor count within reasonable bounds (< 10^6)
# ✓ KV pairs within security limits (< 10^5)
# ✓ Tensor shapes validated against overflow
# ✓ Memory requirements estimated and bounded
```

### Step 2: Test Quantization Accuracy

```bash
# Test I2_S quantization accuracy with real weights
cargo test --no-default-features --features cpu \
    test_i2s_quantization_accuracy_real_weights

# Test TL1/TL2 quantization with production models
cargo test --no-default-features --features cpu \
    test_table_lookup_quantization_accuracy

# Cross-validate against C++ reference implementation
cargo run -p xtask -- crossval --verbose
```

### Step 3: Performance Baselines

```bash
# Establish quantization performance baselines
cargo run -p xtask -- benchmark \
    --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
    --tokens 128

# Expected performance targets:
# Quantization: ≥66 Melem/s (CPU), ≥200 Melem/s (GPU)
# Inference: ≥200 tok/s (CPU), ≥500 tok/s (GPU)
# Memory: <2GB RAM for 2B parameter model
```

## Advanced Real Model Usage

### Programmatic API with Real Weights

```rust
use bitnet::prelude::*;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Load real GGUF model with trained weights
    let model = BitNetModel::from_file(
        "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
    ).await?;

    // Verify real weights were loaded (not mock tensors)
    let tensor_count = model.tensor_count();
    let param_count = model.parameter_count();
    println!("Loaded {} tensors with {} parameters", tensor_count, param_count);

    // Create inference engine with device-aware backend
    let engine = InferenceEngine::builder()
        .model(model)
        .backend(Backend::Auto)  // GPU if available, CPU fallback
        .quantization(QuantizationType::I2S)
        .build()?;

    // Run inference with real neural network weights
    let response = engine.generate(
        "Explain the physics of quantum computing",
        GenerationConfig {
            max_new_tokens: 256,
            temperature: 0.7,
            enable_metrics: true,
            ..Default::default()
        }
    ).await?;

    // Verify meaningful output from real model weights
    println!("Generated: {}", response.text);

    // Access performance metrics
    if let Some(metrics) = response.metrics {
        println!("Inference time: {:.2}ms", metrics.timing.total);
        println!("Throughput: {:.1} tokens/sec", metrics.throughput.e2e);
        println!("Memory used: {:.1}MB", metrics.memory.peak_mb);
    }

    Ok(())
}
```

### Streaming Inference with Real Weights

```rust
use bitnet::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let model = BitNetModel::from_file("real_model.gguf").await?;
    let engine = InferenceEngine::builder()
        .model(model)
        .backend(Backend::Auto)
        .build()?;

    // Stream generation with real neural network inference
    let mut stream = engine.generate_stream(
        "Write a technical explanation of 1-bit neural networks",
        &GenerationConfig {
            max_new_tokens: 512,
            temperature: 0.8,
            ..Default::default()
        }
    );

    // Process real-time token generation
    while let Some(result) = stream.next().await {
        match result {
            Ok(response) => {
                print!("{}", response.text);

                // Access token IDs for analysis
                for &token_id in &response.token_ids {
                    eprintln!("[TOKEN] ID: {}", token_id);
                }
            }
            Err(e) => {
                eprintln!("Generation error: {}", e);
                break;
            }
        }
    }

    Ok(())
}
```

## Validation and Testing

### Unit Testing with Real Models

```bash
# Test real GGUF loading functionality
cargo test --no-default-features --features cpu \
    test_load_real_gguf_model

# Test quantization accuracy with actual weights
cargo test --no-default-features --features cpu \
    test_quantization_accuracy_vs_fp32

# Property-based testing with real model tensors
cargo test --no-default-features --features cpu \
    test_quantization_properties_real_weights
```

### Integration Testing

```bash
# Full end-to-end testing with real models
cargo test --no-default-features --features cpu \
    integration_test_real_model_inference

# GPU validation with real weights
cargo test --no-default-features --features gpu \
    integration_test_gpu_inference_real_model

# Cross-validation against C++ implementation
BITNET_GGUF="model.gguf" cargo test --features crossval \
    test_cross_validation_real_model
```

### Documentation Testing

```bash
# Validate all documentation examples with real models
cargo test --doc --workspace --no-default-features --features cpu

# Build documentation with real model examples
cargo doc --workspace --no-default-features --features cpu --open

# Test documentation examples end-to-end
cargo run -p xtask -- check-docs
```

## Troubleshooting Real Model Issues

### Common GGUF Loading Issues

**Issue 1: Tensor Shape Mismatch**
```
Error: Tensor shape validation failed: expected [4096, 4096], got [4096, 4097]
```

**Solution:**
```bash
# Inspect tensor metadata
cargo run -p bitnet-cli -- inspect --model model.gguf --verbose

# Validate against known good model
cargo run -p xtask -- verify --model model.gguf --strict
```

**Issue 2: Quantization Format Not Supported**
```
Error: Unsupported quantization format: IQ4_XS
```

**Solution:**
```bash
# Check supported formats
cargo run -p bitnet-cli -- compat-check model.gguf --formats

# Convert to supported format if needed
cargo run -p bitnet-cli -- convert \
    --input model_iq4.gguf \
    --output model_i2s.gguf \
    --target-quantization I2_S
```

**Issue 3: Memory Allocation Failures**
```
Error: Failed to allocate tensor memory: 8GB requested, 4GB available
```

**Solution:**
```bash
# Check memory requirements
cargo run -p bitnet-cli -- inspect --model model.gguf --memory

# Use smaller model or enable memory mapping
cargo run -p xtask -- infer \
    --model model.gguf \
    --memory-mapped \
    --prompt "test"
```

### Performance Issues

**Issue: Slow Inference with Real Models**

**Diagnostic:**
```bash
# Profile inference performance
cargo run -p xtask -- benchmark \
    --model model.gguf \
    --profile \
    --tokens 64
```

**Solutions:**
```bash
# Enable native CPU optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release \
    --no-default-features --features cpu

# Use GPU acceleration
cargo build --release --no-default-features --features gpu

# Optimize for throughput
export BITNET_DETERMINISTIC=0
export RAYON_NUM_THREADS=8
```

## Next Steps

Now that you understand real GGUF weight loading:

1. **Production Deployment**: Learn deployment strategies for real models
2. **Performance Optimization**: Advanced tuning techniques for production workloads
3. **Custom Quantization**: Implement custom quantization schemes
4. **Model Conversion**: Convert between different model formats
5. **Security Hardening**: Production security best practices

## Summary

You've learned how to:

- ✅ Load real GGUF models with production-ready weight parsing
- ✅ Validate tensor completeness and accuracy against FP32 baselines
- ✅ Use device-aware quantization for optimal performance
- ✅ Implement security validation and error handling
- ✅ Test cross-validation against C++ reference implementation
- ✅ Troubleshoot common issues with real model inference

The GGUF weight loading system enables meaningful neural network inference with BitNet.rs, moving beyond mock tensors to production-ready AI applications with 1-bit quantized neural networks.

## Performance Baselines

With real GGUF weight loading, BitNet.rs achieves:

- **Quantization Performance**: 66+ Melem/s (CPU), 200+ Melem/s (GPU)
- **Inference Throughput**: 200+ tok/s (CPU), 500+ tok/s (GPU)
- **Memory Efficiency**: <2GB RAM for 2B parameter models
- **Accuracy**: ≥99% vs FP32 baseline for I2_S quantization
- **Security**: Comprehensive validation and bounds checking
- **Compatibility**: Drop-in replacement for C++ implementation

These baselines demonstrate production-ready performance for real-world neural network inference applications.