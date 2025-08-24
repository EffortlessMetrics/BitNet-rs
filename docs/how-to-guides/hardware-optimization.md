# How to Optimize for Hardware

This guide explains how to optimize BitNet.rs for your specific hardware.

## CPU Optimization

### 1. Enable CPU Features

```bash
# Build with native CPU optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Or specify features explicitly
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
```

### 2. Thread Configuration

```rust
use bitnet_rs::prelude::*;

// Set optimal thread count
let num_threads = std::thread::available_parallelism()?.get();
std::env::set_var("RAYON_NUM_THREADS", num_threads.to_string());

// Or configure per-engine
let config = InferenceConfig::builder()
    .num_threads(num_threads)
    .build();
```

### 3. CPU Affinity

```rust
// Pin threads to specific CPU cores (Linux)
use core_affinity;

fn set_cpu_affinity() {
    let core_ids = core_affinity::get_core_ids().unwrap();
    for (i, core_id) in core_ids.iter().enumerate() {
        std::thread::spawn(move || {
            core_affinity::set_for_current(*core_id);
            // Worker thread code here
        });
    }
}
```

### 4. NUMA Optimization

```bash
# Check NUMA topology
numactl --hardware

# Run with NUMA binding
numactl --cpunodebind=0 --membind=0 ./bitnet-server
```

## GPU Optimization

### 1. GPU Selection

```rust
use bitnet_rs::prelude::*;

// Auto-select best GPU
let device = Device::best_cuda_device()?;

// Or select specific GPU
let device = Device::Cuda(0);  // Use GPU 0

// Check GPU capabilities
if device.supports_mixed_precision() {
    println!("GPU supports mixed precision");
}
```

### 2. Memory Management

```rust
// Configure GPU memory
let config = GpuConfig::builder()
    .memory_pool_size(4 * 1024 * 1024 * 1024)  // 4GB pool
    .enable_memory_mapping(true)
    .build();

let engine = InferenceEngine::with_gpu_config(model, tokenizer, device, config)?;
```

### 3. Batch Processing

```rust
// Optimize batch size for GPU
let optimal_batch_size = device.optimal_batch_size(model.parameter_count())?;

let config = InferenceConfig::builder()
    .batch_size(optimal_batch_size)
    .build();
```

### 4. Mixed Precision

```rust
// Enable mixed precision for faster inference
let config = ModelConfig::builder()
    .dtype(DType::F16)  // Use half precision
    .enable_tensor_cores(true)  // Use Tensor Cores if available
    .build();
```
