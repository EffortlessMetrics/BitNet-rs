# BitNet.rs Troubleshooting Guide

This guide helps you diagnose and resolve common issues with **BitNet.rs**, the primary, production-ready Rust implementation of BitNet inference.

> **💡 Migration Tip**: If you're experiencing issues with the legacy C++ implementation, consider migrating to BitNet.rs for better reliability, performance, and support. See our [Migration Guide](../tutorials/migration-guide.md).

## Why Choose BitNet.rs for Troubleshooting

- **Better Error Messages**: Rust's type system provides clear, actionable error messages
- **Memory Safety**: No segfaults, memory leaks, or undefined behavior
- **Active Support**: Regular updates and community support
- **Cross-Platform**: Consistent behavior across all platforms
- **Comprehensive Testing**: Extensive test coverage prevents common issues

## Common Issues

### Installation Issues

#### 1. Rust Compilation Errors

**Problem:** Build fails with compiler errors

**Symptoms:**
```
error: failed to compile `bitnet-cli`
error[E0554]: `#![feature(...)]` may not be used on the stable release channel
```

**Solution:**
```bash
# Ensure you have Rust 1.75 or later
rustc --version

# Update Rust if needed
rustup update stable

# Clean and rebuild
cargo clean
cargo build --release
```

#### 2. CUDA Not Found

**Problem:** GPU features fail to compile

**Symptoms:**
```
error: could not find native static library `cudart`
note: use the `-l` flag to specify native libraries to link
```

**Solutions:**

**Option 1: Install CUDA**
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Option 2: Build without GPU support**
```bash
cargo install bitnet-cli --no-default-features --features cli
```

#### 3. Python Binding Issues

**Problem:** Python bindings fail to install

**Symptoms:**
```
error: Microsoft Visual C++ 14.0 is required
```

**Solutions:**

**Windows:**
```bash
# Install Visual Studio Build Tools
# Or use pre-built wheels
pip install bitnet-py --only-binary=all
```

**Linux/macOS:**
```bash
# Install development tools
sudo apt install build-essential  # Ubuntu
xcode-select --install            # macOS

# Install from source
pip install bitnet-py --no-binary bitnet-py
```

### Model Loading Issues

#### 1. Model File Not Found

**Problem:** Model fails to load

**Symptoms:**
```
Error: Model error: File not found: model.gguf
```

**Solutions:**
```bash
# Check file exists
ls -la model.gguf

# Check permissions
chmod 644 model.gguf

# Use absolute path
bitnet-cli inference --model /absolute/path/to/model.gguf --prompt "Hello"
```

#### 2. Unsupported Model Format

**Problem:** Model format not recognized

**Symptoms:**
```
Error: Model error: Unsupported format: unknown magic bytes
```

**Solutions:**
```bash
# Check file format
file model.bin

# Convert to supported format
bitnet-cli convert --input model.bin --output model.gguf --format gguf

# List supported formats
bitnet-cli formats
```

#### 3. Corrupted Model File

**Problem:** Model file is corrupted

**Symptoms:**
```
Error: Model error: Invalid model file: checksum mismatch
```

**Solutions:**
```bash
# Verify file integrity
bitnet-cli verify --model model.gguf

# Re-download model
rm model.gguf
bitnet-cli model download microsoft/bitnet-b1_58-large

# Check disk space
df -h
```

#### 4. Insufficient Memory

**Problem:** Not enough RAM to load model

**Symptoms:**
```
Error: Memory error: Failed to allocate 8GB for model weights
```

**Solutions:**
```bash
# Check available memory
free -h

# Use memory mapping
bitnet-cli inference --model model.gguf --mmap --prompt "Hello"

# Use smaller model
bitnet-cli model download microsoft/bitnet-b1_58-small

# Enable swap (Linux)
sudo swapon --show
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Performance Issues

#### 1. Slow Inference Speed

**Problem:** Generation is slower than expected

**Symptoms:**
- Low tokens/second
- High latency

**Diagnostics:**
```bash
# Run benchmark
bitnet-cli benchmark --model model.gguf --detailed

# Check system resources
htop
nvidia-smi  # For GPU
```

**Solutions:**

**CPU Optimization:**
```bash
# Enable native CPU features
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Set thread count
export RAYON_NUM_THREADS=8
bitnet-cli inference --model model.gguf --prompt "Hello"

# Use optimized BLAS
export OPENBLAS_NUM_THREADS=8
```

**GPU Optimization:**
```bash
# Check GPU usage
nvidia-smi

# Enable GPU acceleration
bitnet-cli inference --model model.gguf --device cuda --prompt "Hello"

# Optimize batch size
bitnet-cli inference --model model.gguf --device cuda --batch-size 16 --prompt "Hello"
```

#### 2. High Memory Usage

**Problem:** Excessive memory consumption

**Symptoms:**
- Out of memory errors
- System slowdown

**Solutions:**
```bash
# Monitor memory usage
bitnet-cli benchmark --model model.gguf --monitor-memory

# Reduce KV cache size
bitnet-cli inference --model model.gguf --kv-cache-size 1024 --prompt "Hello"

# Use quantized model
bitnet-cli convert --input model.gguf --output model_q4.gguf --quantize q4_0
```

#### 3. GPU Not Being Used

**Problem:** Model runs on CPU despite GPU availability

**Symptoms:**
```
Info: Using device: CPU (CUDA available but not selected)
```

**Solutions:**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Explicitly select GPU
bitnet-cli inference --model model.gguf --device cuda:0 --prompt "Hello"

# Check GPU memory
nvidia-smi

# Verify CUDA support
bitnet-cli info --cuda
```

### Generation Issues

#### 1. Poor Quality Output

**Problem:** Generated text is incoherent or repetitive

**Symptoms:**
- Repetitive text
- Nonsensical output
- Abrupt endings

**Solutions:**
```bash
# Adjust sampling parameters
bitnet-cli inference \
  --model model.gguf \
  --prompt "Hello" \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 50 \
  --repetition-penalty 1.1

# Try different sampling strategies
bitnet-cli inference --model model.gguf --prompt "Hello" --sampling greedy
bitnet-cli inference --model model.gguf --prompt "Hello" --sampling nucleus
```

#### 2. Generation Stops Early

**Problem:** Text generation ends prematurely

**Symptoms:**
- Short outputs
- Incomplete sentences

**Solutions:**
```bash
# Increase max tokens
bitnet-cli inference --model model.gguf --prompt "Hello" --max-new-tokens 200

# Check stop sequences
bitnet-cli inference --model model.gguf --prompt "Hello" --no-stop-sequences

# Verify model configuration
bitnet-cli info --model model.gguf
```

#### 3. Slow Token Generation

**Problem:** Each token takes too long to generate

**Solutions:**
```bash
# Profile generation
bitnet-cli benchmark --model model.gguf --profile

# Enable streaming
bitnet-cli inference --model model.gguf --prompt "Hello" --stream

# Optimize for latency
bitnet-cli inference --model model.gguf --prompt "Hello" --optimize-latency
```

### API Issues

#### 1. Python API Errors

**Problem:** Python bindings crash or error

**Symptoms:**
```python
RuntimeError: BitNet error: Device error: CUDA out of memory
```

**Solutions:**
```python
import bitnet

# Handle errors gracefully
try:
    model = bitnet.BitNetModel.from_pretrained("model")
    output = model.generate("Hello")
except bitnet.BitNetError as e:
    print(f"BitNet error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Use smaller batch size
config = bitnet.ModelConfig(max_batch_size=1)
model = bitnet.BitNetModel.from_pretrained("model", config=config)
```

#### 2. C API Memory Leaks

**Problem:** Memory usage grows over time

**Solutions:**
```c
// Always free resources
BitNetModel* model = bitnet_model_load("model.gguf");
char* output = bitnet_inference(model, "Hello", 100, 0.7f);

// Use output...

// Clean up
bitnet_free_string(output);
bitnet_model_free(model);

// Check for leaks
valgrind --leak-check=full ./your_program
```

#### 3. Async/Await Issues

**Problem:** Async operations hang or fail

**Solutions:**
```rust
use tokio::time::{timeout, Duration};

// Add timeouts
let result = timeout(
    Duration::from_secs(30),
    model.generate("Hello", &config)
).await;

match result {
    Ok(Ok(output)) => println!("Generated: {}", output),
    Ok(Err(e)) => eprintln!("Generation error: {}", e),
    Err(_) => eprintln!("Timeout"),
}
```

## Debugging Tools

### 1. Enable Debug Logging

```bash
# Set log level
export RUST_LOG=debug
bitnet-cli inference --model model.gguf --prompt "Hello"

# Trace level for detailed debugging
export RUST_LOG=trace
bitnet-cli inference --model model.gguf --prompt "Hello"

# Module-specific logging
export RUST_LOG=bitnet_inference=debug,bitnet_models=info
```

### 2. Performance Profiling

```bash
# CPU profiling
cargo install flamegraph
sudo flamegraph -- bitnet-cli inference --model model.gguf --prompt "Hello"

# Memory profiling
valgrind --tool=massif bitnet-cli inference --model model.gguf --prompt "Hello"

# GPU profiling
nsys profile bitnet-cli inference --model model.gguf --device cuda --prompt "Hello"
```

### 3. Model Inspection

```bash
# Model information
bitnet-cli info --model model.gguf

# Detailed model analysis
bitnet-cli analyze --model model.gguf --verbose

# Validate model integrity
bitnet-cli verify --model model.gguf --checksum
```

### 4. System Information

```bash
# BitNet system info
bitnet-cli info --system

# CUDA information
bitnet-cli info --cuda

# Available devices
bitnet-cli info --devices
```

## Environment-Specific Issues

### Windows

#### 1. Path Issues

**Problem:** Long path names cause issues

**Solutions:**
```cmd
# Enable long paths
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1

# Use short paths
bitnet-cli inference --model C:\models\model.gguf --prompt "Hello"
```

#### 2. DLL Loading Issues

**Problem:** Required DLLs not found

**Solutions:**
```cmd
# Install Visual C++ Redistributable
# Download from Microsoft website

# Check DLL dependencies
dumpbin /dependents bitnet-cli.exe

# Add to PATH
set PATH=%PATH%;C:\path\to\dlls
```

### macOS

#### 1. Code Signing Issues

**Problem:** Binary won't run due to code signing

**Solutions:**
```bash
# Remove quarantine attribute
xattr -d com.apple.quarantine bitnet-cli

# Allow unsigned binaries
sudo spctl --master-disable
```

#### 2. Metal GPU Issues

**Problem:** Metal GPU acceleration not working

**Solutions:**
```bash
# Check Metal support
system_profiler SPDisplaysDataType

# Use Metal device
bitnet-cli inference --model model.gguf --device metal --prompt "Hello"
```

### Linux

#### 1. GLIBC Version Issues

**Problem:** Binary requires newer GLIBC

**Solutions:**
```bash
# Check GLIBC version
ldd --version

# Build from source with older GLIBC
cargo build --release --target x86_64-unknown-linux-musl
```

#### 2. GPU Driver Issues

**Problem:** NVIDIA drivers not working

**Solutions:**
```bash
# Check driver installation
nvidia-smi

# Install drivers
sudo apt install nvidia-driver-535

# Verify CUDA
nvcc --version
```

## Getting Help

### 1. Collect Debug Information

Before reporting issues, collect this information:

```bash
# System information
bitnet-cli info --system > debug_info.txt

# Error logs
RUST_LOG=debug bitnet-cli inference --model model.gguf --prompt "Hello" 2>&1 | tee error.log

# Model information
bitnet-cli info --model model.gguf >> debug_info.txt

# Environment
env | grep -E "(CUDA|RUST|BITNET)" >> debug_info.txt
```

### 2. Minimal Reproduction

Create a minimal example that reproduces the issue:

```rust
use bitnet::{BitNetModel, GenerationConfig};

#[tokio::main]
async fn main() -> bitnet::Result<()> {
    let model = BitNetModel::from_pretrained("microsoft/bitnet-b1_58-large").await?;
    let config = GenerationConfig::default();
    let output = model.generate("Hello", &config).await?;
    println!("{}", output);
    Ok(())
}
```

### 3. Report Issues

When reporting issues, include:

- BitNet Rust version
- Operating system and version
- Hardware specifications (CPU, GPU, RAM)
- Complete error messages
- Steps to reproduce
- Debug information collected above

**GitHub Issues:** https://github.com/your-org/bitnet-rust/issues
**Discord Community:** https://discord.gg/bitnet-rust

### 4. Community Resources

- **Documentation:** https://docs.rs/bitnet
- **Examples:** https://github.com/your-org/bitnet-rust/tree/main/examples
- **FAQ:** https://github.com/your-org/bitnet-rust/wiki/FAQ
- **Performance Tips:** https://github.com/your-org/bitnet-rust/wiki/Performance

## Emergency Procedures

### 1. Rollback to Previous Version

```bash
# Uninstall current version
cargo uninstall bitnet-cli

# Install specific version
cargo install bitnet-cli --version 0.1.0

# Or use backup binary
cp bitnet-cli.backup bitnet-cli
```

### 2. Use Fallback Implementation

```python
# Python fallback
try:
    import bitnet  # Rust implementation
except ImportError:
    import bitnet_fallback as bitnet  # Pure Python fallback

model = bitnet.BitNetModel.from_pretrained("model")
```

### 3. Recovery Mode

```bash
# Reset configuration
rm -rf ~/.config/bitnet/

# Clear cache
rm -rf ~/.cache/bitnet/

# Rebuild from clean state
cargo clean
cargo build --release
```