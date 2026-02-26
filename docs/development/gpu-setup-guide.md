# GPU Setup and Usage Guide

This guide covers setting up and using GPU acceleration with BitNet-rs, including CUDA configuration, memory optimization, and performance tuning.

## Prerequisites

### CUDA Requirements
- **CUDA Toolkit 11.0+** (12.x recommended for best performance)
- **NVIDIA Driver** compatible with your CUDA version
- **cuDNN** (optional, for additional optimizations)
- **GPU with Compute Capability 6.0+** (Pascal architecture or newer)

### System Requirements
- **Linux**: CUDA Toolkit from NVIDIA
- **Windows**: CUDA Toolkit + Visual Studio Build Tools
- **macOS**: Not supported (CUDA discontinued for macOS)

## Installation

### 1. Install CUDA Toolkit

#### Ubuntu/Debian
```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install cuda-toolkit-12-3
```

#### Windows
1. Download CUDA Toolkit from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
2. Run the installer with default settings
3. Add CUDA to your PATH:
   ```powershell
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
   ```

### 2. Verify CUDA Installation
```bash
# Check CUDA version
nvcc --version

# Check GPU devices
nvidia-smi

# Test CUDA runtime
cargo run --example cuda_info --no-default-features --features gpu
```

Expected output:
```
CUDA available: true
CUDA device count: 1
Device 0: NVIDIA GeForce RTX 4080
  Compute capability: (8, 9)
  Total memory: 16 GB
  Multiprocessor count: 76
  Supports FP16: true
  Supports BF16: true
```

## Building with GPU Support

### Basic CUDA Build
```bash
# Build with CUDA support
cargo build --no-default-features --release --no-default-features --features gpu

# Run tests to verify GPU functionality
cargo test --no-default-features --workspace --no-default-features --features gpu
```

### Advanced GPU Features
```bash
# Build with all GPU optimizations
cargo build --no-default-features --release --no-default-features --features "cuda,mixed-precision"

# Build with validation framework for debugging
cargo build --no-default-features --release --no-default-features --features "cuda,gpu-validation"
```

## Usage Examples

### Basic GPU Inference

```rust
use bitnet::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Load model with GPU backend
    let model = BitNetModel::from_file("model.gguf").await?;

    let engine = InferenceEngine::builder()
        .model(model)
        .backend(Backend::Cuda { device_id: 0 })  // Use GPU 0
        .build()?;

    let response = engine.generate(
        "Explain GPU acceleration",
        GenerationConfig::default()
    ).await?;

    println!("GPU Generated: {}", response.text);
    Ok(())
}
```

### GPU Kernel Selection

```rust
use bitnet_kernels::{KernelManager, select_gpu_kernel};

// Automatic kernel selection (prefers GPU if available)
let manager = KernelManager::new();
let kernel = manager.select_best()?;
println!("Selected kernel: {}", kernel.name());

// Force GPU kernel selection
let gpu_kernel = select_gpu_kernel(0)?;  // Use device 0
println!("GPU kernel: {}", gpu_kernel.name());

// List all available kernels
let available = manager.list_available_providers();
println!("Available kernels: {:?}", available);
```

### Memory Management

```rust
use bitnet_kernels::gpu::memory_optimization::{OptimizedMemoryPool, MemoryPoolConfig};
use std::time::Duration;

// Configure GPU memory pool
let config = MemoryPoolConfig {
    max_pool_size: 8 * 1024 * 1024 * 1024, // 8GB
    max_cached_buffers: 1000,
    enable_memory_tracking: true,
    cleanup_interval: Duration::from_secs(30),
};

let mut pool = OptimizedMemoryPool::new(0, config);

// Allocate GPU memory
let buffer = pool.allocate(1024 * 1024)?; // 1MB buffer

// Check memory stats
let stats = pool.stats();
println!("Current usage: {} MB", stats.current_usage / (1024 * 1024));
println!("Peak usage: {} MB", stats.peak_usage / (1024 * 1024));
println!("Cache hit rate: {:.1}%",
    stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64 * 100.0);
```

## Performance Optimization

### Device Selection

For multi-GPU systems, select the optimal device:

```rust
use bitnet_kernels::gpu::cuda::list_cuda_devices;

// List all CUDA devices
let devices = list_cuda_devices()?;
for device in devices {
    println!("Device {}: {}", device.device_id, device.name);
    println!("  Memory: {} GB", device.total_memory / (1024 * 1024 * 1024));
    println!("  Compute: {}.{}", device.compute_capability.0, device.compute_capability.1);
    println!("  SMs: {}", device.multiprocessor_count);
}

// Select device with most memory
let best_device = devices.iter()
    .max_by_key(|d| d.total_memory)
    .map(|d| d.device_id)
    .unwrap_or(0);
```

### Mixed Precision

Enable mixed precision for better performance on modern GPUs:

```rust
use bitnet_kernels::gpu::mixed_precision::{MixedPrecisionKernel, PrecisionMode};

let mut mixed_kernel = MixedPrecisionKernel::new(0)?;

// Set precision mode
mixed_kernel.set_precision_mode(PrecisionMode::FP16);  // For Tensor Cores

// Check precision support
if mixed_kernel.supports_fp16() {
    println!("FP16 supported - using Tensor Cores");
} else if mixed_kernel.supports_bf16() {
    println!("BF16 supported - using modern Tensor Cores");
}
```

### Memory Layout Optimization

```rust
use bitnet_kernels::gpu::memory_optimization::{MemoryLayoutOptimizer, AccessPattern};

// Analyze access patterns
let access_indices = vec![0, 1, 2, 3, 4]; // Sequential access
let pattern = MemoryLayoutOptimizer::analyze_access_pattern(&access_indices);

// Optimize data layout
let mut data = vec![0.0f32; 1024];
MemoryLayoutOptimizer::optimize_layout(&mut data, pattern);

// Calculate optimal alignment
let alignment = MemoryLayoutOptimizer::calculate_alignment(data.len() * 4);
println!("Optimal alignment: {} bytes", alignment);
```

## Performance Validation

### GPU Kernel Validation

Run comprehensive GPU validation tests:

```bash
# Run numerical accuracy validation
cargo test --no-default-features --features gpu gpu_numerical_accuracy

# Run performance benchmarks
cargo test --no-default-features --features gpu gpu_performance_benchmark

# Run memory leak detection
cargo test --no-default-features --features gpu gpu_memory_leaks
```

### Custom Validation

```rust
use bitnet_kernels::gpu::validation::{GpuValidator, ValidationConfig};

// Configure validation parameters
let config = ValidationConfig {
    tolerance: 1e-6,
    benchmark_iterations: 100,
    test_sizes: vec![
        (256, 256, 256),    // Medium
        (1024, 1024, 1024), // Large
        (2048, 1024, 512),  // Rectangular
    ],
    check_memory_leaks: true,
    test_mixed_precision: true,
};

// Run validation
let validator = GpuValidator::with_config(config);
let results = validator.validate()?;

// Print results
bitnet_kernels::gpu::validation::print_validation_results(&results);

// Check memory health (useful for production monitoring)
let memory_result = validator.check_memory_health()?;
if memory_result.leaks_detected {
    eprintln!("⚠️ Memory leaks detected!");
}
```

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found
```
error: CUDA toolkit not found
```
**Solution:**
- Install CUDA toolkit
- Add CUDA to PATH: `export PATH=/usr/local/cuda/bin:$PATH`
- Set library path: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`

#### 2. GPU Memory Errors
```
error: out of memory
```
**Solution:**
- Reduce batch size or model size
- Enable memory optimization: `--features gpu,memory-pool`
- Check GPU memory usage: `nvidia-smi`

#### 3. Kernel Compilation Errors
```
error: ptx compilation failed
```
**Solution:**
- Update CUDA toolkit to latest version
- Check compute capability compatibility
- Set `CUDA_ARCH` environment variable: `export CUDA_ARCH=sm_80`

#### 4. Performance Issues
**Symptoms:**
- GPU slower than CPU
- Low GPU utilization

**Solutions:**
- Use larger batch sizes for better GPU utilization
- Enable mixed precision: `PrecisionMode::FP16`
- Check for memory bandwidth bottlenecks
- Profile with `nsight-compute`

### Debug Tools

#### GPU Profiling
```bash
# Profile GPU kernels
nsight-compute --target-processes all cargo run --example gpu_benchmark

# Memory profiling
nsight-systems --trace=cuda cargo test --no-default-features --features cpu gpu_memory_test
```

#### Validation Scripts
```bash
# Quick GPU health check
./scripts/gpu-health-check.sh

# Comprehensive GPU validation
./scripts/gpu-validation-suite.sh

# GPU vs CPU accuracy comparison
./scripts/gpu-cpu-parity-check.sh
```

## Production Deployment

### Environment Variables
```bash
# GPU device selection
export CUDA_VISIBLE_DEVICES=0,1  # Use only GPUs 0 and 1

# Memory management
export BITNET_GPU_MEMORY_FRACTION=0.8  # Use 80% of GPU memory
export BITNET_GPU_ALLOW_GROWTH=1       # Allow memory pool to grow

# Performance tuning
export BITNET_GPU_FORCE_FP16=1          # Force FP16 precision
export BITNET_GPU_KERNEL_TIMEOUT=10     # 10 second kernel timeout
```

### Docker Deployment
```dockerfile
FROM nvidia/cuda:12.3-devel-ubuntu22.04

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy source and build with GPU support
COPY . /app
WORKDIR /app
RUN cargo build --no-default-features --release --no-default-features --features gpu

# Run with GPU access
CMD ["./target/release/bitnet-server", "--gpu", "0"]
```

Run with:
```bash
docker run --gpus all -p 8080:8080 bitnet-rs:gpu
```

### Monitoring and Alerting

```rust
// Production GPU monitoring
use bitnet_kernels::gpu::validation::GpuValidator;

let validator = GpuValidator::new();
let memory_result = validator.check_memory_health()?;

// Alert if memory efficiency is low
if memory_result.efficiency_score < 0.7 {
    log::warn!("GPU memory efficiency low: {:.1}%",
              memory_result.efficiency_score * 100.0);
}

// Alert if leaks detected
if memory_result.leaks_detected {
    log::error!("GPU memory leaks detected!");
}
```

## Best Practices

1. **Always validate GPU setup** before production deployment
2. **Monitor GPU memory usage** to prevent OOM errors
3. **Use mixed precision** on supported hardware for better performance
4. **Profile regularly** to identify bottlenecks
5. **Test GPU failover** to CPU backend for reliability
6. **Keep CUDA toolkit updated** for latest optimizations
7. **Use memory pools** for frequent allocations
8. **Implement proper error handling** for GPU-specific errors

## Performance Tuning Checklist

- [ ] CUDA toolkit 12.x installed
- [ ] GPU compute capability ≥ 6.0
- [ ] Mixed precision enabled for Tensor Core GPUs
- [ ] Memory pool configured appropriately
- [ ] GPU memory usage < 90%
- [ ] Numerical accuracy validation passed
- [ ] Performance benchmarks meet requirements
- [ ] Memory leak detection passed
- [ ] Proper error handling implemented
- [ ] Monitoring and alerting configured

## Further Reading

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)
- [GPU Kernel Architecture](../gpu-kernel-architecture.md)
- [Performance Guide](../performance-guide.md)
