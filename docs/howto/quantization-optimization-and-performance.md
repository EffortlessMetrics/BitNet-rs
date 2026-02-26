# How to Optimize Quantization and Performance

This comprehensive guide shows you how to optimize BitNet-rs quantization performance for production workloads. You'll learn to configure device-aware quantization, enable strict mode to prevent mock fallbacks, tune performance parameters, and achieve realistic performance baselines with actual quantized computation.

## Overview

BitNet-rs provides advanced quantization optimization features:

- **Device-Aware Quantization**: Automatic GPU acceleration with CPU fallback
- **Multiple Quantization Formats**: I2_S, TL1, TL2 with accuracy ≥99% vs FP32
- **SIMD Acceleration**: AVX2/AVX-512 (x86_64), NEON (ARM64) optimizations
- **Memory Optimization**: Zero-copy operations and efficient memory layout
- **Performance Monitoring**: Detailed metrics and benchmarking tools

## Quick Reference Commands

```bash
# Benchmark quantization performance
cargo bench --no-default-features -p bitnet-quantization --bench quantization_bench --no-default-features --features cpu

# Test GPU quantization
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_gpu_quantization_performance

# Profile inference performance
cargo run -p xtask -- benchmark --model model.gguf --profile --tokens 128

# CPU optimization build
RUSTFLAGS="-C target-cpu=native" cargo build --no-default-features --release --no-default-features --features cpu
```

## Step 1: Understanding Quantization Formats

### I2_S Quantization (Recommended for Production)

I2_S (2-bit signed) quantization offers the best balance of accuracy and performance:

```bash
# Test I2_S quantization accuracy
cargo test --no-default-features --features cpu test_i2s_quantization_accuracy

# Benchmark I2_S performance
cargo bench --no-default-features -p bitnet-quantization --bench i2s_bench --no-default-features --features cpu

# Expected performance targets:
# CPU: 10-20 tokens/sec with I2S quantization
# GPU: 50-100 tokens/sec with mixed precision
# Accuracy: I2S ≥99.8%, TL1/TL2 ≥99.6% vs FP32
```

### TL1/TL2 Table Lookup Quantization

Table lookup quantization for specialized use cases:

```bash
# Test TL1 quantization
cargo test --no-default-features --features cpu test_tl1_quantization

# Test TL2 quantization
cargo test --no-default-features --features cpu test_tl2_quantization

# Benchmark table lookup performance
cargo bench --no-default-features -p bitnet-quantization --bench table_lookup_bench --no-default-features --features cpu
```

### Programmatic Quantization Format Selection

```rust
use bitnet_kernels::device_aware::{DeviceAwareQuantizer, DeviceAwareQuantizerFactory};
use bitnet_common::{Device, QuantizationType};

async fn optimize_quantization_format() -> Result<()> {
    // Auto-detect best device
    let quantizer = DeviceAwareQuantizerFactory::auto_detect()?;
    println!("Using device: {:?}", quantizer.device());

    let test_data = vec![1.0, -1.0, 0.5, -0.5, 0.1, -0.1, 0.0, 0.25];
    let mut output = vec![0u8; 2];
    let mut scales = vec![0.0f32; 1];

    // Test different quantization formats
    let formats = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];

    for format in &formats {
        let start = std::time::Instant::now();

        for _ in 0..1000 {
            quantizer.quantize(&test_data, &mut output, &mut scales, *format)?;
        }

        let elapsed = start.elapsed();
        let throughput = (1000.0 * test_data.len() as f64) / elapsed.as_secs_f64() / 1_000_000.0;

        println!("{:?}: {:.1} Melem/s", format, throughput);
    }

    Ok(())
}
```

## Step 2: Device-Aware Optimization

### GPU Acceleration Setup

```bash
# Check GPU availability and capabilities
cargo run --example cuda_info --no-default-features --features gpu

# Test GPU quantization performance
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu \
    test_gpu_quantization_comprehensive

# Benchmark GPU vs CPU performance
cargo bench --no-default-features -p bitnet-kernels --bench device_comparison --no-default-features --features gpu
```

**Expected GPU Performance:**
- Inference: 50-100 tokens/sec with mixed precision acceleration
- Memory bandwidth: High utilization of GPU memory bus
- Concurrent operations: Multiple quantization streams
- CUDA optimization: FP16/BF16 mixed precision support

### CPU SIMD Optimization

```bash
# Build with native CPU optimizations
RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo build --no-default-features --features cpu --release \
    --no-default-features --features cpu

# Test SIMD acceleration
cargo test --no-default-features --features cpu -p bitnet-quantization --test simd_compatibility \
    --no-default-features --features cpu

# Benchmark SIMD vs scalar performance
cargo bench --no-default-features --features cpu -p bitnet-quantization --bench simd_comparison \
    --no-default-features --features cpu
```

**SIMD Optimizations Available:**
- **x86_64**: AVX2, AVX-512 with runtime detection
- **ARM64**: NEON with optimized memory access patterns
- **Fallback**: Portable scalar implementation

### Programmatic Device Selection

```rust
use bitnet_kernels::device_aware::{DeviceAwareQuantizer, DeviceAwareQuantizerFactory};
use bitnet_common::Device;

fn select_optimal_device() -> Result<DeviceAwareQuantizer> {
    // List available devices
    let devices = DeviceAwareQuantizerFactory::list_available_devices();
    println!("Available devices: {:?}", devices);

    // Benchmark each device
    let mut best_device = Device::Cpu;
    let mut best_throughput = 0.0;

    for device in &devices {
        if let Ok(quantizer) = DeviceAwareQuantizer::new(*device) {
            let throughput = benchmark_quantizer(&quantizer)?;
            println!("{:?}: {:.1} Melem/s", device, throughput);

            if throughput > best_throughput {
                best_throughput = throughput;
                best_device = *device;
            }
        }
    }

    println!("Selected device: {:?} ({:.1} Melem/s)", best_device, best_throughput);
    DeviceAwareQuantizer::new(best_device)
}

fn benchmark_quantizer(quantizer: &DeviceAwareQuantizer) -> Result<f64> {
    let test_data = vec![1.0; 1024];
    let mut output = vec![0u8; 256];
    let mut scales = vec![0.0f32; 4];

    let start = std::time::Instant::now();
    for _ in 0..100 {
        quantizer.quantize(&test_data, &mut output, &mut scales, QuantizationType::I2S)?;
    }
    let elapsed = start.elapsed();

    Ok((100.0 * test_data.len() as f64) / elapsed.as_secs_f64() / 1_000_000.0)
}
```

## Step 3: Memory Optimization

### Zero-Copy Operations

```bash
# Test zero-copy quantization
cargo test --no-default-features --features cpu test_zero_copy_quantization

# Benchmark memory-efficient operations
cargo bench --no-default-features -p bitnet-quantization --bench memory_efficiency --no-default-features --features cpu
```

### Memory Layout Optimization

```rust
use bitnet_quantization::{I2SQuantizer, QuantizerTrait};
use std::alloc::{alloc, dealloc, Layout};

fn optimize_memory_layout() -> Result<()> {
    let quantizer = I2SQuantizer::new()?;

    // Use aligned memory for better SIMD performance
    let element_count = 1024;
    let layout = Layout::from_size_align(element_count * 4, 32)?; // 32-byte alignment

    unsafe {
        let ptr = alloc(layout) as *mut f32;
        let aligned_data = std::slice::from_raw_parts_mut(ptr, element_count);

        // Initialize with test pattern
        for (i, val) in aligned_data.iter_mut().enumerate() {
            *val = (i as f32 - 512.0) / 512.0; // Range [-1, 1]
        }

        // Quantize with optimal memory layout
        let mut output = vec![0u8; element_count / 4];
        let mut scales = vec![0.0f32; element_count / 256];

        let start = std::time::Instant::now();
        quantizer.quantize(aligned_data, &mut output, &mut scales)?;
        let elapsed = start.elapsed();

        println!("Quantization time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!("Throughput: {:.1} Melem/s",
                 element_count as f64 / elapsed.as_secs_f64() / 1_000_000.0);

        dealloc(ptr as *mut u8, layout);
    }

    Ok(())
}
```

### Memory Pool Management

```rust
use std::collections::VecDeque;
use std::sync::Mutex;

struct QuantizationMemoryPool {
    buffers: Mutex<VecDeque<Vec<u8>>>,
    scales: Mutex<VecDeque<Vec<f32>>>,
}

impl QuantizationMemoryPool {
    fn new() -> Self {
        Self {
            buffers: Mutex::new(VecDeque::new()),
            scales: Mutex::new(VecDeque::new()),
        }
    }

    fn get_buffer(&self, size: usize) -> Vec<u8> {
        let mut buffers = self.buffers.lock().unwrap();

        if let Some(mut buffer) = buffers.pop_front() {
            if buffer.len() >= size {
                buffer.truncate(size);
                buffer.fill(0);
                return buffer;
            }
        }

        vec![0u8; size]
    }

    fn return_buffer(&self, buffer: Vec<u8>) {
        let mut buffers = self.buffers.lock().unwrap();
        buffers.push_back(buffer);
    }

    fn get_scales(&self, size: usize) -> Vec<f32> {
        let mut scales = self.scales.lock().unwrap();

        if let Some(mut scale_buffer) = scales.pop_front() {
            if scale_buffer.len() >= size {
                scale_buffer.truncate(size);
                scale_buffer.fill(0.0);
                return scale_buffer;
            }
        }

        vec![0.0f32; size]
    }

    fn return_scales(&self, scales: Vec<f32>) {
        let mut scale_buffers = self.scales.lock().unwrap();
        scale_buffers.push_back(scales);
    }
}
```

## Step 4: Performance Monitoring

### Built-in Benchmarking

```bash
# Comprehensive performance benchmarking with strict mode
BITNET_STRICT_MODE=1 cargo run -p xtask -- benchmark \
    --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
    --tokens 128 \
    --warmup 3 \
    --iterations 10

# Expected output (realistic performance):
# Benchmark Results:
#   Device: GPU (CUDA)
#   Quantization: I2S 2-bit signed
#   Inference: 75.3 tok/s (target: 50-100 GPU)
#   Memory: 1.8GB peak usage
#   GPU utilization: 85%
#   Accuracy: 99.8% correlation vs FP32
```

### Custom Performance Metrics

```rust
use std::time::Instant;
use sysinfo::{System, SystemExt};

#[derive(Debug)]
struct PerformanceMetrics {
    quantization_throughput: f64,  // Melem/s
    inference_throughput: f64,     // tok/s
    memory_usage_mb: f64,
    gpu_utilization: f64,
    cpu_utilization: f64,
}

fn measure_performance() -> Result<PerformanceMetrics> {
    let mut system = System::new_all();

    // Measure quantization performance
    let quantizer = DeviceAwareQuantizer::new(Device::Cuda(0))?;
    let test_data = vec![1.0; 1_000_000];
    let mut output = vec![0u8; 250_000];
    let mut scales = vec![0.0f32; 1000];

    let start = Instant::now();
    quantizer.quantize(&test_data, &mut output, &mut scales, QuantizationType::I2S)?;
    let quantization_time = start.elapsed();

    let quantization_throughput = test_data.len() as f64 / quantization_time.as_secs_f64() / 1_000_000.0;

    // Measure inference performance (simplified)
    let start = Instant::now();
    let tokens_generated = simulate_inference(100)?; // Generate 100 tokens
    let inference_time = start.elapsed();

    let inference_throughput = tokens_generated as f64 / inference_time.as_secs_f64();

    // Measure memory usage
    system.refresh_memory();
    let memory_usage_mb = system.used_memory() as f64 / 1_000_000.0;

    // Measure CPU utilization
    system.refresh_cpu();
    let cpu_utilization = system.global_cpu_info().cpu_usage() as f64;

    // GPU utilization (simplified - would use NVML in practice)
    let gpu_utilization = if quantizer.is_gpu_active() { 80.0 } else { 0.0 };

    Ok(PerformanceMetrics {
        quantization_throughput,
        inference_throughput,
        memory_usage_mb,
        gpu_utilization,
        cpu_utilization,
    })
}

fn simulate_inference(token_count: usize) -> Result<usize> {
    // Simplified inference simulation
    std::thread::sleep(std::time::Duration::from_millis(token_count as u64 * 5));
    Ok(token_count)
}
```

## Step 5: Production Optimization

### Environment Configuration

```bash
# Optimize for production throughput with strict mode
export BITNET_STRICT_MODE=1            # Prevent mock inference fallbacks
export BITNET_DETERMINISTIC=0          # Allow non-deterministic optimizations
export RAYON_NUM_THREADS=8             # Match CPU cores
export CUDA_VISIBLE_DEVICES=0          # Use specific GPU
export BITNET_MEMORY_POOL_SIZE=1GB     # Pre-allocate memory pool

# Optimize for deterministic results with strict mode
export BITNET_STRICT_MODE=1            # Essential for production deployment
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1             # Single-threaded for reproducibility
```

### Build Optimizations

```bash
# Maximum performance build
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat" \
cargo build --no-default-features --release --no-default-features --features gpu

# Profile-guided optimization (PGO)
RUSTFLAGS="-C profile-generate=/tmp/pgo-data" \
cargo build --no-default-features --release --no-default-features --features gpu

# Run representative workload for PGO
cargo run -p xtask -- benchmark --model model.gguf --tokens 1000

# Build with PGO data
RUSTFLAGS="-C profile-use=/tmp/pgo-data -C target-cpu=native" \
cargo build --no-default-features --release --no-default-features --features gpu
```

### Runtime Tuning

```rust
use std::env;

fn configure_runtime_optimization() -> Result<()> {
    // Set thread pool size based on workload
    let cpu_cores = num_cpus::get();
    let optimal_threads = if is_compute_intensive() {
        cpu_cores
    } else {
        cpu_cores / 2 // Leave headroom for other processes
    };

    env::set_var("RAYON_NUM_THREADS", optimal_threads.to_string());

    // Configure memory allocation
    if is_memory_intensive() {
        env::set_var("BITNET_MEMORY_POOL_SIZE", "2GB");
        env::set_var("BITNET_ZERO_COPY", "1");
    }

    // Configure GPU settings
    if gpu_available() {
        env::set_var("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
        env::set_var("CUDA_LAUNCH_BLOCKING", "0"); // Async kernel launches
    }

    Ok(())
}

fn is_compute_intensive() -> bool {
    // Determine if workload is compute-bound
    true // Quantization is typically compute-intensive
}

fn is_memory_intensive() -> bool {
    // Determine if workload is memory-bound
    false // Quantization is typically compute-bound
}

fn gpu_available() -> bool {
    env::var("CUDA_VISIBLE_DEVICES").is_ok()
}
```

## Step 6: Testing and Validation

### Performance Regression Testing

```bash
# Run performance regression tests
cargo test --no-default-features --features cpu performance_regression_tests

# Benchmark against baseline
cargo bench --no-default-features --features cpu -- --save-baseline main

# Compare against baseline after changes
cargo bench --no-default-features --features cpu -- --baseline main
```

### Accuracy Validation

```bash
# Test quantization accuracy against FP32 baseline
cargo test --no-default-features --features cpu test_quantization_accuracy_vs_fp32

# Cross-validate with C++ reference implementation
BITNET_GGUF="model.gguf" cargo test --features crossval test_quantization_cross_validation

# Property-based accuracy testing
cargo test --no-default-features --features cpu test_quantization_properties
```

### Performance Testing

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_quantization_performance(c: &mut Criterion) {
    let quantizer = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
    let test_data = vec![1.0; 1024];
    let mut output = vec![0u8; 256];
    let mut scales = vec![0.0f32; 4];

    c.bench_function("i2s_quantization", |b| {
        b.iter(|| {
            quantizer.quantize(&test_data, &mut output, &mut scales, QuantizationType::I2S)
        })
    });

    // Performance assertion
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        quantizer.quantize(&test_data, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    }
    let elapsed = start.elapsed();
    let throughput = (1000.0 * test_data.len() as f64) / elapsed.as_secs_f64() / 1_000_000.0;

    // Assert realistic performance targets (CPU: 10-20 tok/s, GPU: 50-100 tok/s)
    let expected_min = if cfg!(feature = "gpu") { 50.0 } else { 10.0 };
    assert!(throughput >= expected_min, "Inference throughput {:.1} tok/s below target {:.1} tok/s", throughput, expected_min);
}

criterion_group!(benches, benchmark_quantization_performance);
criterion_main!(benches);
```

## Step 7: Troubleshooting Performance Issues

### Common Performance Problems

**Issue: Low Quantization Throughput**

**Diagnostic:**
```bash
# Profile quantization performance
cargo run -p xtask -- benchmark --model model.gguf --profile-quantization

# Check SIMD availability
cargo test --no-default-features --features cpu test_simd_availability
```

**Solutions:**
```bash
# Enable native CPU optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --no-default-features --release --no-default-features --features cpu

# Use GPU acceleration
cargo build --no-default-features --release --no-default-features --features gpu

# Check memory alignment
cargo test --no-default-features --features cpu test_memory_alignment
```

**Issue: GPU Utilization Low**

**Diagnostic:**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check GPU memory bandwidth
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_gpu_memory_bandwidth
```

**Solutions:**
```bash
# Increase batch size
export BITNET_BATCH_SIZE=64

# Use concurrent streams
export BITNET_GPU_STREAMS=4

# Optimize memory access patterns
export BITNET_GPU_MEMORY_COALESCING=1
```

## Summary

This guide covered:

- ✅ **Quantization format optimization** with I2_S, TL1, TL2 performance comparison
- ✅ **Device-aware optimization** with GPU acceleration and CPU SIMD
- ✅ **Memory optimization** using zero-copy operations and memory pools
- ✅ **Performance monitoring** with built-in benchmarks and custom metrics
- ✅ **Production optimization** with environment configuration and build tuning
- ✅ **Testing and validation** for performance regression and accuracy
- ✅ **Troubleshooting** common performance issues and solutions

With these optimization techniques, you can achieve production-ready performance with BitNet-rs's real quantized computation:

**Realistic Performance Targets:**
- **CPU Performance**: 10-20 tokens/sec with I2S quantization
- **GPU Performance**: 50-100 tokens/sec with mixed precision acceleration
- **Quantization Accuracy**: I2S ≥99.8%, TL1/TL2 ≥99.6% correlation with FP32
- **Cross-Validation**: <5% variance from C++ reference implementation

**Key Improvements:**
- ✅ Mock inference elimination with BITNET_STRICT_MODE=1
- ✅ Device-aware quantization selection (CPU SIMD, GPU CUDA)
- ✅ Real quantized matrix multiplication (I2S, TL1, TL2)
- ✅ Performance monitoring with realistic baselines
- ✅ Cross-validation against Microsoft C++ reference
