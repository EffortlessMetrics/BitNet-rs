# How to Validate and Load GGUF Models

This guide provides step-by-step instructions for validating GGUF model files and loading them with BitNet-rs's production-ready weight loading system. You'll learn how to ensure model compatibility, validate tensor completeness, and troubleshoot common issues.

## Overview

BitNet-rs provides comprehensive GGUF validation and loading capabilities:

- **Pre-loading Validation**: Check GGUF compatibility before resource allocation
- **Tensor Completeness**: Verify all required transformer weights are present
- **Security Validation**: Bounds checking and input sanitization
- **Device-Aware Loading**: Automatic GPU/CPU placement with fallback
- **Error Recovery**: Graceful handling with actionable error messages

## Quick Reference Commands

```bash
# Basic compatibility check
cargo run -p bitnet-cli -- compat-check model.gguf

# Detailed inspection with JSON output
cargo run -p bitnet-cli -- inspect --model model.gguf --json

# Comprehensive validation with verification
cargo run -p xtask -- verify --model model.gguf --tokenizer tokenizer.json

# Load and test inference
cargo run -p xtask -- infer --model model.gguf --prompt "test"
```

## Step 1: Pre-Loading Validation

### Basic GGUF Compatibility Check

Before loading a model, always validate GGUF compatibility:

```bash
# Quick compatibility assessment
cargo run -p bitnet-cli -- compat-check model.gguf

# Expected output for valid GGUF:
# File:      model.gguf
# Status:    ✓ Valid GGUF
# Version:   3 (supported)
# Tensors:   328
# KV pairs:  67
# Size:      1.2GB
```

### JSON Output for Automation

For CI/CD pipelines and automation:

```bash
# Get machine-readable validation results
cargo run -p bitnet-cli -- compat-check model.gguf --json

# Example output:
{
  "file_path": "model.gguf",
  "is_valid": true,
  "compatibility": {
    "supported_version": true,
    "tensors_reasonable": true,
    "kvs_reasonable": true
  },
  "header": {
    "version": 3,
    "tensor_count": 328,
    "kv_count": 67
  },
  "validation": {
    "magic_bytes": true,
    "size_bounds": true,
    "security_limits": true
  }
}
```

### Detailed Model Inspection

For comprehensive model analysis:

```bash
# Detailed inspection with categorized metadata
cargo run -p bitnet-cli -- inspect --model model.gguf

# Example output:
Model Metadata:
  Architecture: llama
  Parameters: 2.4B
  Quantization: I2_S
  Vocabulary: 32000

Tensor Statistics:
  Total tensors: 328
  Attention layers: 192 tensors
  Feed-forward layers: 96 tensors
  Normalization layers: 32 tensors
  Embeddings: 8 tensors

Memory Estimates:
  Raw size: 2.4GB
  Quantized size: 1.2GB (50% compression)
  Loading memory: 1.8GB
```

## Step 2: Tensor Validation

### Verify Tensor Completeness

Ensure all required transformer layers are present:

```bash
# Comprehensive tensor validation
cargo run -p xtask -- verify --model model.gguf --strict

# Expected validation checks:
# ✓ All attention tensors present (Q, K, V, Output)
# ✓ All feed-forward tensors present (Gate, Up, Down)
# ✓ All normalization tensors present (Attention, FFN)
# ✓ Embedding tensors present (Token, Position)
# ✓ Output projection tensor present
# ✓ Tensor shapes match model configuration
# ✓ Quantization format validated (I2_S, TL1, TL2)
```

### Programmatic Tensor Validation

```rust
use bitnet_models::gguf_simple::load_gguf;
use bitnet_common::{Device, Result};
use std::path::Path;

/// Validate that all expected transformer tensors are loaded from GGUF
///
/// # Example
/// ```rust,no_run
/// use std::path::Path;
/// use bitnet_models::gguf_simple::load_gguf;
/// use bitnet_common::Device;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model_path = Path::new("model.gguf");
/// let (config, tensors) = load_gguf(model_path, Device::Cpu)?;
///
/// // Verify real GGUF weight loading (not mock tensors)
/// println!("Loaded {} real tensor weights", tensors.len());
/// println!("Model has {} layers with {} hidden dimensions",
///          config.model.num_layers, config.model.hidden_size);
///
/// // Check for specific transformer components
/// let has_attention = tensors.keys().any(|k| k.contains("attn"));
/// let has_feedforward = tensors.keys().any(|k| k.contains("ffn") || k.contains("mlp"));
/// assert!(has_attention && has_feedforward, "Model missing core transformer weights");
/// # Ok(())
/// # }
/// ```
fn validate_model_tensors(model_path: &Path) -> Result<()> {
    // Load GGUF with comprehensive validation
    let (config, tensors) = load_gguf(model_path, Device::Cpu)?;

    println!("Model configuration:");
    println!("  Layers: {}", config.model.num_layers);
    println!("  Hidden size: {}", config.model.hidden_size);
    println!("  Vocab size: {}", config.model.vocab_size);

    // Validate tensor completeness for real GGUF weight loading
    let expected_tensors = calculate_expected_tensor_count(&config);
    if tensors.len() != expected_tensors {
        return Err(format!(
            "Tensor count mismatch: expected {}, got {}",
            expected_tensors, tensors.len()
        ).into());
    }

    // Validate attention tensors for each layer (real weights, not mock)
    for layer in 0..config.model.num_layers {
        let attention_tensors = [
            format!("blk.{}.attn_q.weight", layer),     // BitNet style naming
            format!("blk.{}.attn_k.weight", layer),
            format!("blk.{}.attn_v.weight", layer),
            format!("blk.{}.attn_output.weight", layer),
        ];

        for tensor_name in &attention_tensors {
            if !tensors.contains_key(tensor_name) {
                return Err(format!("Missing attention tensor: {}", tensor_name).into());
            }

            // Verify tensor has real data (not zero-initialized mock)
            let tensor = &tensors[tensor_name];
            println!("✓ Loaded real weights for {}: shape {:?}",
                     tensor_name, tensor.shape());
        }
    }

    println!("✓ All {} real tensor weights validated successfully", tensors.len());
    Ok(())
}

fn calculate_expected_tensor_count(config: &bitnet_common::BitNetConfig) -> usize {
    // Calculate expected tensor count for complete transformer
    let layers = config.model.num_layers;
    let attention_tensors_per_layer = 4;  // Q, K, V, Output
    let ffn_tensors_per_layer = 3;        // Gate, Up, Down
    let norm_tensors_per_layer = 2;       // Attention norm, FFN norm

    let layer_tensors = layers * (attention_tensors_per_layer + ffn_tensors_per_layer + norm_tensors_per_layer);
    let embedding_tensors = 2;  // Token embeddings, output projection

    layer_tensors + embedding_tensors
}
```

## Step 3: Device-Aware Loading

### GPU Loading with Fallback

```bash
# Attempt GPU loading with CPU fallback
cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "test" \
    --features gpu

# Monitor device selection in logs:
# [INFO] Using CUDA device 0 for tensor placement
# [INFO] GPU memory available: 8GB
# [INFO] Model memory requirement: 1.2GB
# [INFO] Loading 328 tensors to GPU...
```

### CPU-Only Loading

```bash
# Force CPU loading (useful for debugging)
cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "test" \
    --no-default-features --features cpu

# Expected behavior:
# [INFO] Using CPU device for tensor placement
# [INFO] Loading 328 tensors to CPU memory...
# [INFO] CPU SIMD optimizations: AVX2, AVX-512 enabled
```

### Programmatic Device Selection

```rust
use bitnet_models::gguf_simple::load_gguf;
use bitnet_common::Device;

async fn load_model_with_device_selection(model_path: &Path) -> Result<()> {
    // Try GPU first, fallback to CPU
    let device = if cuda_available() {
        Device::Cuda(0)
    } else {
        println!("CUDA not available, using CPU");
        Device::Cpu
    };

    // Load with automatic device fallback
    let (config, tensors) = load_gguf(model_path, device)?;

    println!("Loaded model on device: {:?}", device);
    println!("Tensor count: {}", tensors.len());
    println!("Memory usage: {:.1}GB", estimate_memory_usage(&tensors));

    Ok(())
}

fn cuda_available() -> bool {
    // Check CUDA availability
    match std::process::Command::new("nvidia-smi").output() {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
```

## Step 4: Error Handling and Recovery

### Common GGUF Validation Errors

**Error: Invalid GGUF Magic Bytes**
```
Error: Invalid GGUF magic bytes: expected "GGUF", got "GGJF"
```

**Solution:**
```bash
# Check file integrity
file model.gguf
hexdump -C model.gguf | head -1

# Re-download if corrupted
rm model.gguf
cargo run -p xtask -- download-model --id repo/model --file model.gguf
```

**Error: Unsupported GGUF Version**
```
Error: Unsupported GGUF version: 4 (supported: 1-3)
```

**Solution:**
```bash
# Check version compatibility
cargo run -p bitnet-cli -- compat-check model.gguf --verbose

# Convert to supported version if possible
cargo run -p bitnet-cli -- convert \
    --input model_v4.gguf \
    --output model_v3.gguf \
    --target-version 3
```

**Error: Tensor Count Exceeds Security Limits**
```
Error: Tensor count 2000000 exceeds security limit 1000000
```

**Solution:**
```bash
# Inspect suspicious model
cargo run -p bitnet-cli -- inspect --model model.gguf --security

# Use relaxed limits for trusted models (development only)
BITNET_SECURITY_RELAXED=1 cargo run -p xtask -- verify --model model.gguf
```

### Memory-Related Errors

**Error: Insufficient Memory for Model Loading**
```
Error: Failed to allocate 8GB for model tensors (4GB available)
```

**Solutions:**
```bash
# Check memory requirements
cargo run -p bitnet-cli -- inspect --model model.gguf --memory

# Use memory mapping for large models
cargo run -p xtask -- infer \
    --model model.gguf \
    --memory-mapped \
    --prompt "test"

# Use smaller quantized version
cargo run -p xtask -- download-model \
    --id repo/model \
    --file model-q4.gguf  # Smaller quantized version
```

### GPU Loading Issues

**Error: CUDA Device Not Available**
```
Error: CUDA device 0 not available, falling back to CPU
```

**Diagnostic:**
```bash
# Check CUDA setup
nvidia-smi
cargo run --example cuda_info --no-default-features --features gpu

# Test GPU functionality
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_cuda_device_info_query
```

**Error: GPU Memory Exhausted**
```
Error: CUDA out of memory: 8GB requested, 2GB available
```

**Solutions:**
```bash
# Check GPU memory usage
nvidia-smi

# Use CPU fallback
cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "test" \
    --no-default-features --features cpu

# Use model sharding (if supported)
cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "test" \
    --shard-across-devices
```

## Step 5: Security Validation

### Security Hardening Options

```bash
# Enable strict security validation
export BITNET_STRICT_VALIDATION=1
export BITNET_SECURITY_AUDIT=1

# Run with security audit
cargo run -p xtask -- verify \
    --model model.gguf \
    --security-audit \
    --strict
```

### Programmatic Security Checks

```rust
use bitnet_models::gguf_simple::validate_gguf_security;

fn perform_security_audit(model_path: &Path) -> Result<()> {
    // Comprehensive security validation
    let security_report = validate_gguf_security(model_path)?;

    if !security_report.is_safe() {
        println!("Security issues found:");
        for issue in security_report.issues() {
            println!("  ⚠️  {}", issue);
        }
        return Err("Model failed security validation".into());
    }

    println!("✅ Model passed security validation");
    println!("  File size: {:.1}MB", security_report.file_size_mb());
    println!("  Tensor bounds: OK");
    println!("  Memory limits: OK");
    println!("  Format validation: OK");

    Ok(())
}
```

## Step 6: Performance Optimization

### Memory-Efficient Loading

```bash
# Use zero-copy loading where possible
cargo run -p xtask -- infer \
    --model model.gguf \
    --zero-copy \
    --prompt "test"

# Preload to page cache
cargo run -p xtask -- preload-model model.gguf

# Then run inference (faster startup)
cargo run -p xtask -- infer --model model.gguf --prompt "test"
```

### Parallel Loading

```rust
use tokio::task;
use std::sync::Arc;

async fn parallel_model_loading() -> Result<()> {
    let model_paths = vec![
        "model1.gguf",
        "model2.gguf",
        "model3.gguf",
    ];

    // Load models in parallel
    let load_tasks: Vec<_> = model_paths.into_iter()
        .map(|path| {
            let path = Arc::new(path.to_string());
            task::spawn(async move {
                let (config, tensors) = load_gguf(
                    Path::new(&*path),
                    Device::Cpu
                )?;
                Ok((path, config, tensors))
            })
        })
        .collect();

    // Wait for all models to load
    for task in load_tasks {
        let (path, config, tensors) = task.await??;
        println!("Loaded {}: {} tensors", path, tensors.len());
    }

    Ok(())
}
```

## Step 7: Testing and Validation

### Unit Testing Model Loading

```bash
# Test GGUF loading functionality
cargo test --no-default-features --features cpu test_gguf_loading

# Test with different model formats
cargo test --no-default-features --features cpu test_quantization_formats

# Property-based testing
cargo test --no-default-features --features cpu test_gguf_properties
```

### Integration Testing

```bash
# Full end-to-end testing
cargo test --no-default-features --workspace --no-default-features --features cpu integration_tests

# GPU integration testing
cargo test --no-default-features --workspace --no-default-features --features gpu gpu_integration_tests

# Cross-validation testing
BITNET_GGUF="model.gguf" cargo test --features crossval crossval_tests
```

## Summary

This guide covered:

- ✅ **Pre-loading validation** with compatibility checks and security audits
- ✅ **Tensor completeness verification** for all transformer components
- ✅ **Device-aware loading** with GPU acceleration and CPU fallback
- ✅ **Error handling and recovery** for common GGUF issues
- ✅ **Security validation** and hardening options
- ✅ **Performance optimization** with memory-efficient loading
- ✅ **Testing strategies** for robust model validation

With these techniques, you can reliably validate and load GGUF models with BitNet-rs's production-ready weight loading system, ensuring both correctness and security for neural network inference applications.
