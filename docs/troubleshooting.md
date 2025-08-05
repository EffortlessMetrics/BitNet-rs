# Troubleshooting Guide

Common issues and solutions when using BitNet.rs.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Model Loading Issues](#model-loading-issues)
- [Inference Issues](#inference-issues)
- [Performance Issues](#performance-issues)
- [Memory Issues](#memory-issues)
- [GPU Issues](#gpu-issues)
- [Server Issues](#server-issues)
- [Debugging Tips](#debugging-tips)

## Installation Issues

### Issue: Compilation Fails with Missing Dependencies

**Symptoms:**
```
error: failed to run custom build command for `bitnet-kernels`
  --- stderr
  thread 'main' panicked at 'Unable to find libclang'
```

**Solutions:**

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install build-essential pkg-config libssl-dev cmake libclang-dev
```

**On macOS:**
```bash
# Install Xcode command line tools
xcode-select --install

# Or using Homebrew
brew install cmake llvm
```

**On Windows:**
```powershell
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Or using chocolatey
choco install visualstudio2022buildtools
choco install cmake
```

### Issue: CUDA Compilation Fails

**Symptoms:**
```
error: nvcc not found in PATH
```

**Solutions:**

1. **Install CUDA Toolkit:**
   ```bash
   # Ubuntu
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

2. **Set Environment Variables:**
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

3. **Disable CUDA if not needed:**
   ```bash
   cargo build --no-default-features --features="cpu"
   ```

### Issue: Rust Version Too Old

**Symptoms:**
```
error: package `bitnet-rs` requires Rust version 1.75.0 or newer
```

**Solution:**
```bash
# Update Rust
rustup update

# Or install specific version
rustup install 1.75.0
rustup default 1.75.0
```

## Model Loading Issues

### Issue: Model File Not Found

**Symptoms:**
```
Error: Model error: No such file or directory (os error 2)
```

**Solutions:**

1. **Check file path:**
   ```rust
   use std::path::Path;
   
   let model_path = "models/bitnet-1.58b.gguf";
   if !Path::new(model_path).exists() {
       eprintln!("Model file not found: {}", model_path);
       return Err("Model file not found".into());
   }
   ```

2. **Use absolute path:**
   ```rust
   let model_path = std::env::current_dir()?
       .join("models")
       .join("bitnet-1.58b.gguf");
   let model = BitNetModel::from_file(model_path).await?;
   ```

3. **Download model:**
   ```bash
   # Using CLI
   bitnet download bitnet-1.58b-i2s --output ./models/
   
   # Or manually
   mkdir -p models
   wget https://huggingface.co/bitnet/bitnet-1.58b/resolve/main/model.gguf -O models/bitnet-1.58b.gguf
   ```

### Issue: Unsupported Model Format

**Symptoms:**
```
Error: Model error: Unsupported model format
```

**Solutions:**

1. **Check supported formats:**
   ```rust
   // Supported formats: .gguf, .safetensors, .bin
   let supported_extensions = vec![".gguf", ".safetensors", ".bin"];
   ```

2. **Convert model format:**
   ```bash
   # Convert to GGUF
   bitnet convert input.safetensors output.gguf --format gguf
   
   # Convert with quantization
   bitnet convert input.bin output.gguf --format gguf --quantization i2s
   ```

3. **Use format-specific loading:**
   ```rust
   // Explicitly specify format
   let model = BitNetModel::from_gguf("model.gguf").await?;
   let model = BitNetModel::from_safetensors("model.safetensors").await?;
   ```

### Issue: Model Corruption

**Symptoms:**
```
Error: Model error: Invalid magic number in model file
```

**Solutions:**

1. **Verify file integrity:**
   ```bash
   # Check file size
   ls -lh models/bitnet-1.58b.gguf
   
   # Verify checksum if available
   sha256sum models/bitnet-1.58b.gguf
   ```

2. **Re-download model:**
   ```bash
   rm models/bitnet-1.58b.gguf
   bitnet download bitnet-1.58b-i2s --output ./models/
   ```

3. **Check disk space:**
   ```bash
   df -h .
   ```

## Inference Issues

### Issue: Generation Produces Gibberish

**Symptoms:**
- Output contains random characters
- Repeated tokens
- Nonsensical text

**Solutions:**

1. **Check tokenizer compatibility:**
   ```rust
   // Ensure tokenizer matches model
   let tokenizer = TokenizerBuilder::from_pretrained("gpt2")?;
   
   // Test tokenization
   let tokens = tokenizer.encode("Hello, world!", true)?;
   let decoded = tokenizer.decode(&tokens, true)?;
   assert_eq!(decoded.trim(), "Hello, world!");
   ```

2. **Adjust generation parameters:**
   ```rust
   let config = GenerationConfig {
       temperature: 0.7,  // Lower for more deterministic output
       top_p: 0.9,        // Nucleus sampling
       top_k: 50,         // Limit vocabulary
       repetition_penalty: 1.1,  // Reduce repetition
       ..Default::default()
   };
   ```

3. **Validate model quantization:**
   ```rust
   // Check if model is properly quantized
   let model_info = model.info();
   println!("Quantization: {:?}", model_info.quantization);
   ```

### Issue: Generation is Too Slow

**Symptoms:**
- High latency (>1 second per token)
- Low throughput

**Solutions:**

1. **Use GPU acceleration:**
   ```rust
   let device = Device::Cuda(0);  // Use GPU 0
   let engine = InferenceEngine::new(model, tokenizer, device)?;
   ```

2. **Optimize CPU usage:**
   ```rust
   // Set number of threads
   std::env::set_var("RAYON_NUM_THREADS", "8");
   
   // Use optimized kernels
   let device = Device::Cpu;  // Will auto-select best CPU kernels
   ```

3. **Reduce model size:**
   ```bash
   # Use more aggressive quantization
   bitnet convert model.gguf model-q4.gguf --quantization tl2
   ```

### Issue: Out of Memory During Inference

**Symptoms:**
```
Error: CUDA out of memory
Error: Cannot allocate memory
```

**Solutions:**

1. **Reduce batch size:**
   ```rust
   let config = InferenceConfig {
       batch_size: 1,  // Reduce from default
       ..Default::default()
   };
   ```

2. **Use CPU instead of GPU:**
   ```rust
   let device = Device::Cpu;  // Fallback to CPU
   ```

3. **Limit sequence length:**
   ```rust
   let config = GenerationConfig {
       max_new_tokens: 50,  // Reduce from default
       ..Default::default()
   };
   ```

## Performance Issues

### Issue: Poor CPU Performance

**Symptoms:**
- Slow inference on CPU
- High CPU usage but low throughput

**Solutions:**

1. **Check CPU features:**
   ```rust
   // Verify SIMD support
   #[cfg(target_arch = "x86_64")]
   {
       if is_x86_feature_detected!("avx2") {
           println!("AVX2 supported");
       } else {
           println!("AVX2 not supported - performance will be limited");
       }
   }
   ```

2. **Enable CPU optimizations:**
   ```bash
   # Build with native CPU features
   RUSTFLAGS="-C target-cpu=native" cargo build --release
   ```

3. **Tune thread count:**
   ```rust
   // Set optimal thread count
   let num_threads = std::thread::available_parallelism()?.get();
   std::env::set_var("RAYON_NUM_THREADS", num_threads.to_string());
   ```

### Issue: GPU Underutilization

**Symptoms:**
- Low GPU usage (<50%)
- GPU memory not fully utilized

**Solutions:**

1. **Increase batch size:**
   ```rust
   let config = InferenceConfig {
       batch_size: 8,  // Increase for better GPU utilization
       ..Default::default()
   };
   ```

2. **Use mixed precision:**
   ```rust
   let config = ModelConfig {
       dtype: DType::F16,  // Use half precision
       ..Default::default()
   };
   ```

3. **Check GPU memory:**
   ```bash
   nvidia-smi  # Monitor GPU usage
   ```

## Memory Issues

### Issue: Memory Leak

**Symptoms:**
- Memory usage increases over time
- Eventually runs out of memory

**Solutions:**

1. **Check for resource leaks:**
   ```rust
   // Ensure proper cleanup
   {
       let engine = InferenceEngine::new(model, tokenizer, device)?;
       // engine is automatically dropped here
   }
   ```

2. **Monitor memory usage:**
   ```rust
   use std::alloc::{GlobalAlloc, Layout, System};
   
   // Custom allocator for monitoring
   struct MonitoringAllocator;
   
   unsafe impl GlobalAlloc for MonitoringAllocator {
       unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
           let ptr = System.alloc(layout);
           println!("Allocated {} bytes", layout.size());
           ptr
       }
       
       unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
           println!("Deallocated {} bytes", layout.size());
           System.dealloc(ptr, layout);
       }
   }
   ```

3. **Use memory profiling:**
   ```bash
   # Install valgrind (Linux)
   sudo apt install valgrind
   
   # Run with memory checking
   valgrind --tool=memcheck --leak-check=full ./target/release/bitnet-cli
   ```

### Issue: High Memory Usage

**Symptoms:**
- Uses more memory than expected
- System becomes unresponsive

**Solutions:**

1. **Reduce model precision:**
   ```rust
   let config = ModelConfig {
       dtype: DType::F16,  // Use 16-bit instead of 32-bit
       ..Default::default()
   };
   ```

2. **Limit cache size:**
   ```rust
   let config = InferenceConfig {
       max_cache_size: 1024 * 1024 * 512,  // 512MB limit
       ..Default::default()
   };
   ```

3. **Use memory mapping:**
   ```rust
   let loader = ModelLoader::new()
       .memory_map(true)  // Use memory mapping for large models
       .build();
   ```

## GPU Issues

### Issue: CUDA Driver Version Mismatch

**Symptoms:**
```
Error: CUDA driver version is insufficient for CUDA runtime version
```

**Solutions:**

1. **Update NVIDIA drivers:**
   ```bash
   # Ubuntu
   sudo apt update
   sudo apt install nvidia-driver-525
   
   # Or use the official installer
   wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
   sudo sh cuda_12.0.0_525.60.13_linux.run
   ```

2. **Check compatibility:**
   ```bash
   nvidia-smi  # Check driver version
   nvcc --version  # Check CUDA version
   ```

### Issue: GPU Not Detected

**Symptoms:**
```
Error: No CUDA devices found
```

**Solutions:**

1. **Verify GPU is available:**
   ```bash
   nvidia-smi
   lspci | grep -i nvidia
   ```

2. **Check CUDA installation:**
   ```bash
   nvcc --version
   ls /usr/local/cuda/lib64/
   ```

3. **Fallback to CPU:**
   ```rust
   let device = if Device::cuda_is_available() {
       Device::Cuda(0)
   } else {
       println!("CUDA not available, using CPU");
       Device::Cpu
   };
   ```

## Server Issues

### Issue: Server Won't Start

**Symptoms:**
```
Error: Address already in use (os error 98)
```

**Solutions:**

1. **Check port availability:**
   ```bash
   # Check what's using port 3000
   lsof -i :3000
   netstat -tulpn | grep :3000
   ```

2. **Use different port:**
   ```rust
   let server = BitNetServer::builder()
       .bind("0.0.0.0:3001")  // Use different port
       .build()
       .await?;
   ```

3. **Kill existing process:**
   ```bash
   # Find and kill process using port
   sudo kill -9 $(lsof -t -i:3000)
   ```

### Issue: High Server Latency

**Symptoms:**
- Slow API responses
- Timeouts

**Solutions:**

1. **Increase worker threads:**
   ```rust
   let server = BitNetServer::builder()
       .workers(8)  // Increase from default
       .build()
       .await?;
   ```

2. **Enable connection pooling:**
   ```rust
   let server = BitNetServer::builder()
       .connection_pool_size(100)
       .build()
       .await?;
   ```

3. **Add caching:**
   ```rust
   let server = BitNetServer::builder()
       .enable_caching(true)
       .cache_size(1000)
       .build()
       .await?;
   ```

## Debugging Tips

### Enable Debug Logging

```bash
# Set log level
export RUST_LOG=debug

# Or more specific
export RUST_LOG=bitnet=debug,bitnet_inference=trace

# Run with logging
cargo run --bin bitnet-cli -- generate "Hello" --model model.gguf
```

### Use Debug Builds

```bash
# Build in debug mode for better error messages
cargo build

# Run with debug assertions
cargo run --bin bitnet-cli
```

### Profiling

```bash
# Install profiling tools
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --bin bitnet-cli -- generate "Hello" --model model.gguf

# Use perf (Linux)
perf record --call-graph=dwarf ./target/release/bitnet-cli generate "Hello" --model model.gguf
perf report
```

### Memory Debugging

```bash
# Use AddressSanitizer
RUSTFLAGS="-Z sanitizer=address" cargo run --target x86_64-unknown-linux-gnu

# Use Miri for undefined behavior detection
cargo +nightly miri run --bin bitnet-cli
```

### Testing

```rust
// Add comprehensive tests
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_loading() {
        let model = BitNetModel::from_file("test_model.gguf").await;
        assert!(model.is_ok(), "Failed to load test model: {:?}", model.err());
    }

    #[tokio::test]
    async fn test_inference() {
        let model = create_test_model().await.unwrap();
        let tokenizer = create_test_tokenizer().unwrap();
        let mut engine = InferenceEngine::new(model, tokenizer, Device::Cpu).unwrap();
        
        let response = engine.generate("Hello").await;
        assert!(response.is_ok(), "Inference failed: {:?}", response.err());
        assert!(!response.unwrap().is_empty(), "Empty response");
    }
}
```

## Getting Help

If these solutions don't resolve your issue:

1. **Check GitHub Issues**: [https://github.com/bitnet-rs/bitnet-rs/issues](https://github.com/bitnet-rs/bitnet-rs/issues)
2. **Create a Bug Report**: Include:
   - Operating system and version
   - Rust version (`rustc --version`)
   - BitNet.rs version
   - Full error message
   - Minimal reproduction case
3. **Join Discord**: [https://discord.gg/bitnet-rs](https://discord.gg/bitnet-rs)
4. **Stack Overflow**: Tag questions with `bitnet-rs`

## Reporting Bugs

When reporting bugs, please include:

```rust
// System information
println!("OS: {}", std::env::consts::OS);
println!("Arch: {}", std::env::consts::ARCH);
println!("Rust version: {}", env!("RUSTC_VERSION"));
println!("BitNet.rs version: {}", env!("CARGO_PKG_VERSION"));

// Error context
eprintln!("Error occurred at: {}:{}", file!(), line!());
eprintln!("Error: {:?}", error);
```