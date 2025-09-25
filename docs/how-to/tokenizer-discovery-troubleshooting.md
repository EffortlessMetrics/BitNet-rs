# How to Troubleshoot Tokenizer Discovery Issues

This guide provides step-by-step solutions for common tokenizer discovery and download problems in BitNet.rs neural network inference.

## Quick Diagnostic Commands

Before diving into specific issues, use these commands to gather diagnostic information:

```bash
# Check model compatibility and metadata
cargo run -p bitnet-cli -- compat-check model.gguf

# Verify xtask functionality
cargo run -p xtask -- verify --model model.gguf --allow-mock

# Test inference with verbose logging
RUST_LOG=bitnet_tokenizers=debug cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Test" \
    --auto-download
```

## Common Issues and Solutions

### 1. "No compatible tokenizer found" Error

**Symptoms:**
```
Error: No compatible tokenizer found for llama model with vocab_size 128256 (strict mode)
```

**Cause:** Model architecture or vocabulary size doesn't match known patterns, and strict mode prevents fallback.

**Solutions:**

#### Option A: Disable Strict Mode (Development Only)
```bash
# Allow mock tokenizer fallback
unset BITNET_STRICT_TOKENIZERS
cargo run -p xtask -- infer --model model.gguf --prompt "Test" --auto-download
```

#### Option B: Provide Explicit Tokenizer
```bash
# Download compatible tokenizer manually
curl -L "https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/tokenizer.json" \
    -o tokenizer.json

# Use explicit tokenizer path
cargo run -p xtask -- infer \
    --model model.gguf \
    --tokenizer tokenizer.json \
    --prompt "Test"
```

#### Option C: Fix Model Metadata
```bash
# Check what metadata the model has
cargo run -p bitnet-cli -- inspect-gguf model.gguf

# Look for these metadata keys:
# - general.architecture
# - tokenizer.ggml.vocab_size
# - llama.vocab_size
```

### 2. Download Failures

**Symptoms:**
```
Error: HTTP error 404: https://huggingface.co/nonexistent/repo/resolve/main/tokenizer.json
Error: HTTP request failed for https://huggingface.co/...: timeout
```

**Solutions:**

#### Network Issues
```bash
# Test network connectivity
curl -I "https://huggingface.co"

# Use offline mode with cached tokenizers
export BITNET_OFFLINE=1
cargo run -p xtask -- infer --model model.gguf --prompt "Test"
```

#### Repository Issues
```bash
# Verify the repository exists
curl -I "https://huggingface.co/meta-llama/Llama-2-7b-hf"

# Check available files
curl -s "https://huggingface.co/api/models/meta-llama/Llama-2-7b-hf" | grep -o '"[^"]*tokenizer[^"]*"'

# Use alternative repository
cargo run -p xtask -- infer \
    --model model.gguf \
    --tokenizer "https://huggingface.co/alternative-repo/resolve/main/tokenizer.json" \
    --prompt "Test"
```

#### Timeout Issues
```bash
# Increase timeout and retry
export BITNET_DOWNLOAD_TIMEOUT=600  # 10 minutes
cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Test" \
    --auto-download
```

### 3. Cache Corruption Issues

**Symptoms:**
```
Error: Invalid JSON in downloaded tokenizer: expected `,` or `}` at line 1 column 42
Error: Downloaded tokenizer file is empty
```

**Solutions:**

#### Clear and Re-download
```bash
# Clear specific tokenizer cache
rm -rf ~/.cache/bitnet/tokenizers/llama2-32k/

# Clear all tokenizer caches
rm -rf ~/.cache/bitnet/tokenizers/

# Re-download
cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Test" \
    --auto-download
```

#### Verify Cache Integrity
```bash
# Check cached files
find ~/.cache/bitnet/tokenizers/ -name "*.json" -exec sh -c '
    echo "Checking: $1"
    jq empty "$1" 2>/dev/null && echo "✓ Valid JSON" || echo "✗ Invalid JSON"
' _ {} \;

# Manually validate specific tokenizer
jq empty ~/.cache/bitnet/tokenizers/llama2-32k/tokenizer.json
```

#### Resume Interrupted Downloads
```bash
# BitNet.rs automatically resumes partial downloads
# If resume fails, clear the partial file:
rm ~/.cache/bitnet/tokenizers/*/tokenizer.json.partial

# Then retry download
cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Test" \
    --auto-download
```

### 4. Vocabulary Size Mismatches

**Symptoms:**
```
Warning: Vocabulary size mismatch: expected 32000, got 50257
Model: 128256, Tokenizer: 50257
```

**Causes and Solutions:**

#### Incorrect Model Type Detection
```bash
# Check detected model type
RUST_LOG=bitnet_tokenizers::discovery=debug cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Test" \
    --auto-download 2>&1 | grep "model type"

# If incorrectly detected as "gpt2" instead of "llama":
# This is a model metadata issue - check GGUF general.architecture
```

#### Wrong Repository Mapping
```bash
# Check what repository is being used
RUST_LOG=bitnet_tokenizers::discovery=debug cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Test" \
    --auto-download 2>&1 | grep "download compatible tokenizer"

# Use explicit tokenizer for correct vocabulary size
wget "https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/tokenizer.json"
cargo run -p xtask -- infer \
    --model model.gguf \
    --tokenizer tokenizer.json \
    --prompt "Test"
```

### 5. Permission and File System Issues

**Symptoms:**
```
Error: Permission denied (os error 13)
Error: No space left on device (os error 28)
```

**Solutions:**

#### Permission Issues
```bash
# Check cache directory permissions
ls -la ~/.cache/bitnet/

# Fix permissions
mkdir -p ~/.cache/bitnet/tokenizers
chmod 755 ~/.cache/bitnet/tokenizers

# Use custom cache directory
export BITNET_CACHE_DIR="/tmp/bitnet-cache"
mkdir -p "$BITNET_CACHE_DIR"
cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Test" \
    --auto-download
```

#### Disk Space Issues
```bash
# Check available space
df -h ~/.cache/bitnet/

# Clear old caches
du -sh ~/.cache/bitnet/tokenizers/*
rm -rf ~/.cache/bitnet/tokenizers/old-unused-*

# Use temporary cache location
export BITNET_CACHE_DIR="/tmp/bitnet-temp-cache"
cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Test" \
    --auto-download
```

### 6. GPU vs CPU Performance Issues

**Symptoms:**
```
Warning: Large vocabulary (128256) running on CPU - consider GPU acceleration
GPU operations: 0, CPU operations: 1000, GPU efficiency: 0.0%
```

**Solutions:**

#### Enable GPU Features
```bash
# Build with GPU support
cargo build --no-default-features --features gpu

# Run inference with GPU
cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Test" \
    --auto-download \
    --no-default-features --features gpu
```

#### Verify CUDA Setup
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Test CUDA functionality
cargo test -p bitnet-kernels --no-default-features --features gpu gpu_smoke_test
```

#### Fallback to CPU Gracefully
```bash
# Force CPU-only inference for large vocabularies
BITNET_FORCE_CPU=1 cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Test" \
    --auto-download \
    --no-default-features --features cpu
```

## Advanced Troubleshooting

### Environment Variable Configuration

Create a troubleshooting environment:

```bash
# Create debug configuration
cat > debug-tokenizer.env << 'EOF'
RUST_LOG=bitnet_tokenizers=debug,bitnet_models=info
BITNET_DETERMINISTIC=1
BITNET_SEED=42
BITNET_STRICT_TOKENIZERS=0
BITNET_OFFLINE=0
EOF

# Source and test
source debug-tokenizer.env
cargo run -p xtask -- infer --model model.gguf --prompt "Debug test" --auto-download
```

### Cross-Validation Testing

```bash
# Test tokenizer compatibility with cross-validation
export BITNET_GGUF="path/to/model.gguf"
cargo run -p xtask -- crossval --tokenizer-only

# Compare tokenizer outputs
cargo test -p bitnet-tokenizers --no-default-features --features cpu test_tokenizer_contract
```

### Performance Profiling

```bash
# Profile tokenizer discovery performance
cargo build --release --no-default-features --features cpu
time cargo run --release -p xtask -- infer \
    --model model.gguf \
    --prompt "Performance test" \
    --auto-download

# Monitor memory usage
/usr/bin/time -v cargo run -p xtask -- infer \
    --model model.gguf \
    --prompt "Memory test" \
    --auto-download
```

### Manual Tokenizer Testing

```bash
# Test tokenizer functionality independently
cargo test -p bitnet-tokenizers --no-default-features --features cpu

# Test specific tokenizer types
cargo test -p bitnet-tokenizers --no-default-features --features "cpu,spm" test_sentencepiece_tokenizer_contract

# Test with real fixtures
SPM_MODEL=tests/fixtures/spm/tiny.model \
cargo test -p bitnet-tokenizers --no-default-features --features "cpu,spm,integration-tests"
```

## Prevention Strategies

### 1. Pre-deployment Validation

```bash
# Validate all models in your deployment
for model in models/*.gguf; do
    echo "Validating: $model"
    cargo run -p xtask -- verify --model "$model" --allow-mock || echo "❌ Failed: $model"
done

# Pre-download all required tokenizers
cargo run -p xtask -- pre-download-tokenizers --model-dir models/
```

### 2. Production Monitoring

```bash
# Health check script
cat > tokenizer-health-check.sh << 'EOF'
#!/bin/bash
set -e

export BITNET_STRICT_TOKENIZERS=1
export BITNET_DETERMINISTIC=1

for model in models/*.gguf; do
    if ! cargo run -p xtask -- infer \
        --model "$model" \
        --prompt "Health check" \
        --max-tokens 1 >/dev/null 2>&1; then
        echo "❌ Health check failed: $model"
        exit 1
    fi
done

echo "✅ All models healthy"
EOF

chmod +x tokenizer-health-check.sh
./tokenizer-health-check.sh
```

### 3. Cache Management

```bash
# Automated cache cleanup
cat > cleanup-tokenizer-cache.sh << 'EOF'
#!/bin/bash

CACHE_DIR="$HOME/.cache/bitnet/tokenizers"
MAX_AGE_DAYS=30

# Remove caches older than MAX_AGE_DAYS
find "$CACHE_DIR" -type f -mtime +$MAX_AGE_DAYS -delete

# Keep only the most recent 10 cached tokenizers
ls -t "$CACHE_DIR" | tail -n +11 | xargs -I {} rm -rf "$CACHE_DIR/{}"

echo "Cache cleanup complete"
EOF

chmod +x cleanup-tokenizer-cache.sh

# Run weekly
(crontab -l 2>/dev/null; echo "0 2 * * 0 /path/to/cleanup-tokenizer-cache.sh") | crontab -
```

## Getting Help

If you're still experiencing issues:

1. **Check GitHub Issues**: [BitNet-rs Issues](https://github.com/EffortlessMetrics/BitNet-rs/issues)
2. **Enable Debug Logging**: `RUST_LOG=debug` for comprehensive logs
3. **Create Minimal Reproduction**:
   ```bash
   # Share this information in bug reports
   cargo --version
   rustc --version
   echo "Model: $(file model.gguf)"
   echo "Environment: $(env | grep BITNET_)"
   ```

4. **Community Support**: Join the discussion in GitHub Discussions

Remember: Most tokenizer issues are configuration or environment related. This guide covers 95% of common problems you'll encounter in production deployments.