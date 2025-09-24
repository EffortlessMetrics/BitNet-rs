# GPU/CUDA Development Guide

This document covers GPU/CUDA development practices, testing strategies, and troubleshooting for BitNet.rs.

## GPU Backend Detection and Hardware Querying

BitNet.rs provides comprehensive GPU detection utilities supporting multiple backends (CUDA, Metal, ROCm, WebGPU) alongside production-ready CUDA device querying using the cudarc API to enable intelligent GPU acceleration and automatic fallback mechanisms.

### GPU Detection API

The new GPU detection utilities provide backend-agnostic GPU availability checking:

```rust
use bitnet_kernels::gpu_utils::{gpu_available, get_gpu_info, preflight_check};

// Quick availability check
if gpu_available() {
    println!("GPU acceleration available");
}

// Detailed backend information
let gpu_info = get_gpu_info();
println!("{}", gpu_info.summary());

// Available backends:
println!("CUDA: {}", gpu_info.cuda);
println!("Metal: {}", gpu_info.metal); 
println!("ROCm: {}", gpu_info.rocm);
println!("WebGPU: {}", gpu_info.wgpu);

// Version information (when available)
if let Some(version) = gpu_info.cuda_version {
    println!("CUDA Version: {}", version);
}

// Preflight check with helpful error messages
match preflight_check() {
    Ok(()) => println!("GPU ready for acceleration"),
    Err(msg) => eprintln!("GPU setup issue: {}", msg),
}
```

### GPU Detection Commands

```bash
# Test GPU detection functionality
cargo test -p bitnet-kernels --no-default-features test_gpu_info_summary

# Run xtask commands with GPU detection
cargo run -p xtask -- download-model  # Uses GPU detection for optimizations

# Mock GPU scenarios for testing (see Testing section)
BITNET_GPU_FAKE="cuda,rocm" cargo test -p bitnet-kernels test_gpu_info_mocked_scenarios
```

### Backend-Specific Detection

1. **CUDA Detection**:
   - Uses `nvidia-smi` to query available GPUs
   - Extracts CUDA version from `nvcc --version`
   - Provides compute capability and memory information

2. **Metal Detection**:
   - Automatic detection on macOS systems
   - Uses system information to identify Apple Silicon

3. **ROCm Detection**:
   - Uses `rocm-smi` to query AMD GPUs
   - Extracts ROCm version information
   - Supports multiple AMD GPU configurations

4. **WebGPU Detection**:
   - Available when any other backend is present
   - Provides fallback compatibility for unsupported hardware

### Mock Testing Support

The GPU detection system includes comprehensive mock testing capabilities:

```bash
# Test scenarios without actual GPU hardware
export BITNET_GPU_FAKE="cuda"        # Mock CUDA-only
export BITNET_GPU_FAKE="metal"       # Mock Metal-only  
export BITNET_GPU_FAKE="cuda,rocm"   # Mock multiple backends
export BITNET_GPU_FAKE=""            # Mock no GPU available

# Run tests with mocked GPU environments
cargo test -p bitnet-kernels test_gpu_info_mocked_scenarios
```

### Performance Environment Variables

The performance tracking system supports configuration through environment variables:

```bash
# Performance configuration for GPU workloads
export BITNET_BATCH_SIZE=8              # Optimal batch size for GPU processing
export BITNET_MEMORY_LIMIT=2GB          # Memory limit for GPU operations
export BITNET_NUM_THREADS=4             # Thread count for CPU fallback operations

# Deterministic performance testing
export BITNET_DETERMINISTIC=1           # Enable deterministic mode
export BITNET_SEED=42                   # Set seed for reproducible results
export RAYON_NUM_THREADS=1              # Single-threaded CPU operations

# GPU-specific performance tuning
cargo test -p bitnet-inference --features integration-tests test_engine_performance_tracking_integration

# Test performance with different configurations
BITNET_BATCH_SIZE=4 cargo test -p bitnet-kernels --features gpu test_gpu_memory_management
BITNET_MEMORY_LIMIT=512MB cargo test -p bitnet-kernels --features gpu test_cuda_validation_comprehensive
```

## CUDA Device Querying and Hardware Detection

BitNet.rs implements production-ready CUDA device querying using the cudarc API to enable intelligent GPU acceleration and automatic fallback mechanisms.

### Device Information Available

The `CudaDeviceInfo` structure provides comprehensive hardware details:

```rust
pub struct CudaDeviceInfo {
    pub device_id: usize,                    // CUDA device index
    pub name: String,                        // Device name (e.g., "GeForce RTX 4090")
    pub compute_capability: (i32, i32),      // Major.minor compute capability
    pub total_memory: usize,                 // Total device memory in bytes
    pub multiprocessor_count: i32,           // Number of streaming multiprocessors
    pub max_threads_per_block: i32,          // Maximum threads per block
    pub max_shared_memory_per_block: usize,  // Maximum shared memory per block
    pub supports_fp16: bool,                 // Half-precision floating point support
    pub supports_bf16: bool,                 // Brain floating point support
}
```

### Enhanced GPU Memory Optimization and Debug Tracing (New in PR #201)

BitNet.rs now provides advanced GPU memory optimization with comprehensive debug stack trace capture for improved developer productivity and production debugging:

#### Memory Pool with Stack Trace Debugging

The `OptimizedMemoryPool` now captures stack traces for every allocation in debug builds:

```rust
use bitnet_kernels::gpu::memory_optimization::{OptimizedMemoryPool, MemoryPoolConfig};

// Create memory pool with debug tracing enabled
let config = MemoryPoolConfig::default();
let mut pool = OptimizedMemoryPool::new(0, config);

// Get device ID for multi-GPU scenarios
println!("Memory pool device: {}", pool.device_id());

// Allocate memory (stack trace captured in debug builds)
let buffer = pool.allocate(1024 * 1024)?; // 1MB allocation

// Check for memory leaks with stack trace information
let leaks = pool.check_leaks();
for leak in leaks {
    eprintln!("Memory leak detected: {}", leak);
    // In debug builds, includes full stack trace:
    // Device 0: potential leak: 1048576 bytes at 0x7f8b4c000000
    // Stack trace:
    //    0: rust_begin_unwind
    //    1: core::panicking::panic_fmt
    //    2: bitnet_kernels::gpu::memory_optimization::OptimizedMemoryPool::allocate
    //    3: my_application::main
}

// Deallocate memory properly
pool.deallocate(buffer);
```

#### Device ID Tracking for Mixed Precision Kernels

Mixed precision kernels now expose device ID and capability methods for multi-GPU debugging:

```rust
use bitnet_kernels::gpu::{MixedPrecisionKernel, PrecisionMode};

// Create kernel and get device tracking information
let kernel = MixedPrecisionKernel::new(0)?;

// Device identification for multi-GPU scenarios
println!("Kernel device ID: {}", kernel.device_id());
println!("Device supports FP16: {}", kernel.supports_fp16());
println!("Device supports BF16: {}", kernel.supports_bf16());

// Use device capabilities for optimization decisions
if kernel.supports_bf16() {
    kernel.set_precision_mode(PrecisionMode::BF16);
    println!("Using BF16 precision on device {}", kernel.device_id());
} else if kernel.supports_fp16() {
    kernel.set_precision_mode(PrecisionMode::FP16);
    println!("Using FP16 precision on device {}", kernel.device_id());
} else {
    kernel.set_precision_mode(PrecisionMode::FP32);
    println!("Using FP32 precision on device {}", kernel.device_id());
}
```

#### Memory Leak Detection with Stack Traces

The enhanced memory pool provides production-ready leak detection:

```rust
use std::time::Duration;

// Configure memory pool with custom leak detection threshold
let config = MemoryPoolConfig {
    max_pool_size: 2 * 1024 * 1024 * 1024, // 2GB pool
    cleanup_interval: Duration::from_secs(30),
    ..Default::default()
};

let mut pool = OptimizedMemoryPool::new(0, config);

// Simulate long-running allocations
let long_lived_buffer = pool.allocate(64 * 1024)?;

// After 1 hour, check for leaks (configurable threshold)
std::thread::sleep(Duration::from_secs(3601));
let leaks = pool.check_leaks();

for leak_report in leaks {
    // In debug builds, includes complete stack trace
    eprintln!("LEAK DETECTED: {}", leak_report);

    // Example output:
    // Device 0: potential leak: 65536 bytes at 0x7f8b4c010000
    // Stack trace:
    //    0: std::backtrace::Backtrace::force_capture
    //    1: bitnet_kernels::gpu::memory_optimization::OptimizedMemoryPool::track_allocation
    //    2: bitnet_kernels::gpu::memory_optimization::OptimizedMemoryPool::allocate
    //    3: my_application::process_batch
    //    4: my_application::main
}

// Clean up
pool.deallocate(long_lived_buffer);
```

#### Memory Statistics and Performance Monitoring

Enhanced memory statistics provide comprehensive tracking:

```rust
// Get detailed memory usage statistics
let stats = pool.stats();
println!("Memory Statistics:");
println!("  Total allocated: {:.2} MB", stats.total_allocated as f64 / (1024.0 * 1024.0));
println!("  Current usage: {:.2} MB", stats.current_usage as f64 / (1024.0 * 1024.0));
println!("  Peak usage: {:.2} MB", stats.peak_usage as f64 / (1024.0 * 1024.0));
println!("  Allocation count: {}", stats.allocation_count);
println!("  Cache hit ratio: {:.1}%",
    (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64) * 100.0);

// Memory access pattern analysis for optimization
use bitnet_kernels::gpu::memory_optimization::{MemoryLayoutOptimizer, AccessPattern};

let access_indices = vec![0, 2, 4, 6, 8]; // Strided access pattern
let pattern = MemoryLayoutOptimizer::analyze_access_pattern(&access_indices);

match pattern {
    AccessPattern::Sequential => println!("Optimal sequential access detected"),
    AccessPattern::Strided { stride } => {
        println!("Strided access pattern detected with stride: {}", stride);
        // Can optimize memory layout for this pattern
    }
    AccessPattern::Random => println!("Random access pattern - consider data reorganization"),
}
```

### Device Memory Tracking

BitNet.rs provides comprehensive memory tracking capabilities for both CPU and GPU devices:

```rust
use bitnet_kernels::device_aware::{DeviceAwareQuantizer, DeviceStats};
use bitnet_common::Device;

// Create device-aware quantizer with memory tracking
let quantizer = DeviceAwareQuantizer::new(Device::Cuda(0))?;

// Perform operations with automatic memory tracking
let result = quantizer.quantize(&input, QuantizationType::I2S)?;

// Get comprehensive device statistics including memory usage
if let Some(stats) = quantizer.get_stats() {
    println!("Memory Usage: {:.1} MB / {:.1} MB ({:.1}%)", 
        stats.memory_used_bytes as f64 / (1024.0 * 1024.0),
        stats.memory_total_bytes as f64 / (1024.0 * 1024.0),
        (stats.memory_used_bytes as f64 / stats.memory_total_bytes as f64) * 100.0
    );
    
    println!("Operations: {} GPU, {} CPU, {} fallbacks",
        stats.gpu_operations, stats.cpu_operations, stats.fallback_count);
    
    // Check memory efficiency
    if let Some(efficiency) = stats.memory_efficiency() {
        println!("Memory Efficiency: {:.2}%", efficiency * 100.0);
    }
}
```

#### Memory Tracking Features

- **Real-time Host Memory**: Uses `memory-stats` crate for accurate process-specific memory usage
- **System Memory Monitoring**: Uses `sysinfo` crate for total system memory tracking
- **GPU Memory Integration**: CUDA cuMemGetInfo_v2 for GPU memory statistics (when available)
- **Thread-Safe Tracking**: Arc<Mutex<DeviceStatsInternal>> for safe concurrent access
- **Memory Efficiency Metrics**: Calculated ratios and usage percentages
- **Automatic Updates**: Memory stats updated during quantization and matrix operations

### Device Querying Commands

```bash
# Test CUDA device detection and querying
cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_device_info_query

# List all available CUDA devices with detailed information
cargo run --example gpu_validation --no-default-features --features gpu

# Test CUDA availability in your application
cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_availability

# Validate device capabilities for BitNet quantization
cargo test -p bitnet-kernels --no-default-features --features gpu test_device_capability_validation

# Test comprehensive memory tracking on GPU devices
cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_tracking_comprehensive

# Test device-aware memory statistics collection
cargo test -p bitnet-kernels --no-default-features --features gpu test_device_memory_tracking

# Test GPU memory management and leak detection
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_memory_management
```

#### Enhanced Debugging Commands (New in PR #201)

```bash
# Test memory pool creation with device ID tracking
cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_pool_creation

# Test stack trace capture in debug builds (requires debug build)
cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_allocation -- --nocapture

# Test memory leak detection with comprehensive stack traces
cargo test -p bitnet-kernels --no-default-features --features gpu test_check_leaks -- --nocapture

# Test device ID tracking for mixed precision kernels
cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_device_tracking

# Test memory access pattern analysis and optimization
cargo test -p bitnet-kernels --no-default-features --features gpu test_access_pattern_analysis

# Run comprehensive memory optimization tests with debug output
RUST_LOG=debug cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_optimization -- --nocapture

# Test multi-GPU device ID tracking (requires multiple GPUs)
cargo test -p bitnet-kernels --no-default-features --features gpu test_multi_device_memory_pools --ignored

# Test enhanced memory statistics with stack trace integration
cargo test -p bitnet-kernels --no-default-features --features gpu test_enhanced_memory_stats --ignored
```

### Hardware-Aware Optimization

The CUDA implementation automatically optimizes based on detected hardware:

1. **Compute Capability Detection**:
   - **CC 6.0+**: Basic CUDA operations with FP32
   - **CC 6.1+**: FP16 tensor core operations enabled
   - **CC 8.0+**: BF16 tensor core operations enabled
   - **CC 9.0+**: FP8 operations (future enhancement)

2. **Memory-Based Optimization**:
   - Large memory devices (>16GB): Larger batch processing
   - Limited memory devices (<8GB): Conservative memory allocation
   - Automatic shared memory configuration based on device limits

3. **Multiprocessor Scaling**:
   - Grid dimensions automatically scaled to multiprocessor count
   - Work distribution optimized for available execution units

### Device Selection and Fallback

```bash
# Test multi-GPU device selection
cargo test -p bitnet-kernels --no-default-features --features gpu test_multi_gpu_selection

# Test automatic CPU fallback when GPU operations fail
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_quantization_fallback --ignored

# Test concurrent GPU operations across devices
cargo test -p bitnet-kernels --no-default-features --features gpu test_concurrent_gpu_operations --ignored
```

### API Usage Examples

```rust
use bitnet_kernels::gpu::cuda::{list_cuda_devices, CudaKernel};

// Query all available CUDA devices
let devices = list_cuda_devices()?;
for device in devices {
    println!("Device {}: {} (CC {}.{})", 
        device.device_id, device.name, 
        device.compute_capability.0, device.compute_capability.1);
    println!("  Memory: {:.1} GB", device.total_memory as f64 / 1e9);
    println!("  FP16: {}, BF16: {}", device.supports_fp16, device.supports_bf16);
}

// Create kernel with specific device
let kernel = CudaKernel::new_with_device(0)?;
let info = kernel.device_info();
println!("Using device: {} with {} SMs", info.name, info.multiprocessor_count);

// Automatic optimization based on device capabilities (Enhanced in PR #199)
// calculate_optimal_launch_params now used internally in matmul operations
// Instead of hardcoded 16x16 blocks, uses device-specific optimization
```

## Multi-GPU Development and Device ID Tracking (New in PR #201)

BitNet.rs now provides comprehensive device ID tracking and multi-GPU support for advanced deployment scenarios:

### Device ID Tracking Examples

#### Basic Device ID Querying

```rust
use bitnet_kernels::gpu::{MixedPrecisionKernel, OptimizedMemoryPool, MemoryPoolConfig};

// Create kernels on specific devices and track their IDs
let kernel_0 = MixedPrecisionKernel::new(0)?;
let kernel_1 = MixedPrecisionKernel::new(1)?;

println!("Kernel 0 device: {}", kernel_0.device_id());
println!("Kernel 1 device: {}", kernel_1.device_id());

// Create memory pools with device tracking
let config = MemoryPoolConfig::default();
let mut pool_0 = OptimizedMemoryPool::new(0, config.clone());
let mut pool_1 = OptimizedMemoryPool::new(1, config);

println!("Memory pool 0 device: {}", pool_0.device_id());
println!("Memory pool 1 device: {}", pool_1.device_id());
```

#### Multi-GPU Capability Detection

```rust
use bitnet_kernels::gpu::{MixedPrecisionKernel, PrecisionMode};

// Query capabilities across multiple devices
for device_id in 0..2 {
    match MixedPrecisionKernel::new(device_id) {
        Ok(kernel) => {
            println!("Device {}: {}", device_id, kernel.device_info().name);
            println!("  Device ID: {}", kernel.device_id());
            println!("  Supports FP16: {}", kernel.supports_fp16());
            println!("  Supports BF16: {}", kernel.supports_bf16());
            println!("  Optimal precision: {:?}", kernel.optimal_precision());
        }
        Err(e) => {
            println!("Device {} unavailable: {}", device_id, e);
        }
    }
}
```

#### Load Balancing Across Multiple GPUs

```rust
use std::collections::HashMap;

struct MultiGpuManager {
    kernels: HashMap<usize, MixedPrecisionKernel>,
    memory_pools: HashMap<usize, OptimizedMemoryPool>,
}

impl MultiGpuManager {
    fn new(device_ids: &[usize]) -> Result<Self> {
        let mut kernels = HashMap::new();
        let mut memory_pools = HashMap::new();

        for &device_id in device_ids {
            let kernel = MixedPrecisionKernel::new(device_id)?;
            let config = MemoryPoolConfig::default();
            let pool = OptimizedMemoryPool::new(device_id, config);

            // Verify device ID consistency
            assert_eq!(kernel.device_id(), device_id);
            assert_eq!(pool.device_id(), device_id);

            kernels.insert(device_id, kernel);
            memory_pools.insert(device_id, pool);
        }

        Ok(Self { kernels, memory_pools })
    }

    fn get_optimal_device(&self) -> Option<usize> {
        // Find device with best capabilities
        self.kernels.iter()
            .filter(|(_, kernel)| kernel.supports_bf16())
            .map(|(&device_id, _)| device_id)
            .next()
            .or_else(|| {
                // Fallback to FP16 devices
                self.kernels.iter()
                    .filter(|(_, kernel)| kernel.supports_fp16())
                    .map(|(&device_id, _)| device_id)
                    .next()
            })
    }

    fn process_batch_on_device(&mut self, device_id: usize, data: &[f32]) -> Result<Vec<f32>> {
        let kernel = self.kernels.get_mut(&device_id)
            .ok_or_else(|| format!("Device {} not available", device_id))?;

        let pool = self.memory_pools.get_mut(&device_id)
            .ok_or_else(|| format!("Memory pool for device {} not available", device_id))?;

        // Verify we're using the correct device
        assert_eq!(kernel.device_id(), device_id);
        assert_eq!(pool.device_id(), device_id);

        // Allocate memory for processing
        let buffer = pool.allocate(data.len() * std::mem::size_of::<f32>())?;

        // Process data with optimal precision for this device
        let mut result = vec![0.0f32; data.len()];
        kernel.matmul_auto(data, data, &mut result, 1, data.len(), 1)?;

        // Clean up
        pool.deallocate(buffer);

        Ok(result)
    }
}

// Usage example
let device_ids = vec![0, 1, 2];
let mut manager = MultiGpuManager::new(&device_ids)?;

if let Some(optimal_device) = manager.get_optimal_device() {
    println!("Using optimal device: {}", optimal_device);
    let data = vec![1.0f32; 1024];
    let result = manager.process_batch_on_device(optimal_device, &data)?;
    println!("Processed {} elements on device {}", result.len(), optimal_device);
}
```

### Multi-GPU Memory Debugging

```rust
use std::time::Duration;

// Track memory usage across multiple devices
fn debug_multi_gpu_memory(device_ids: &[usize]) -> Result<()> {
    let mut pools = Vec::new();

    // Create memory pools for each device
    for &device_id in device_ids {
        let config = MemoryPoolConfig::default();
        let pool = OptimizedMemoryPool::new(device_id, config);
        pools.push(pool);
    }

    // Allocate memory on each device
    let mut buffers = Vec::new();
    for (i, pool) in pools.iter_mut().enumerate() {
        let device_id = device_ids[i];
        println!("Allocating on device {}", device_id);

        let buffer = pool.allocate(1024 * 1024)?; // 1MB
        buffers.push(buffer);

        // Verify device tracking
        assert_eq!(pool.device_id(), device_id);

        let stats = pool.stats();
        println!("Device {} stats: {} MB allocated",
            device_id,
            stats.current_usage as f64 / (1024.0 * 1024.0)
        );
    }

    // Simulate long-running operations
    std::thread::sleep(Duration::from_secs(2));

    // Check for leaks across all devices
    for (i, pool) in pools.iter().enumerate() {
        let device_id = device_ids[i];
        let leaks = pool.check_leaks();

        if !leaks.is_empty() {
            eprintln!("Device {} has {} potential leaks:", device_id, leaks.len());
            for leak in leaks {
                eprintln!("  {}", leak);
            }
        } else {
            println!("Device {} has no memory leaks", device_id);
        }
    }

    // Clean up all buffers
    for (i, pool) in pools.iter_mut().enumerate() {
        let buffer = buffers.remove(0);
        pool.deallocate(buffer);
        println!("Cleaned up device {}", device_ids[i]);
    }

    Ok(())
}
```

### Device-Specific Performance Optimization

```rust
use bitnet_kernels::gpu::PrecisionMode;

struct DeviceOptimizer {
    device_configurations: HashMap<usize, DeviceConfig>,
}

#[derive(Debug, Clone)]
struct DeviceConfig {
    device_id: usize,
    optimal_precision: PrecisionMode,
    max_batch_size: usize,
    preferred_memory_limit: usize,
}

impl DeviceOptimizer {
    fn new() -> Self {
        Self {
            device_configurations: HashMap::new(),
        }
    }

    fn analyze_device(&mut self, device_id: usize) -> Result<DeviceConfig> {
        let kernel = MixedPrecisionKernel::new(device_id)?;

        // Verify device ID consistency
        assert_eq!(kernel.device_id(), device_id);

        // Determine optimal configuration based on capabilities
        let optimal_precision = if kernel.supports_bf16() {
            PrecisionMode::BF16
        } else if kernel.supports_fp16() {
            PrecisionMode::FP16
        } else {
            PrecisionMode::FP32
        };

        let device_info = kernel.device_info();
        let max_batch_size = (device_info.total_memory / (1024 * 1024 * 100)).min(128); // Conservative estimate
        let preferred_memory_limit = device_info.total_memory * 80 / 100; // 80% of total memory

        let config = DeviceConfig {
            device_id,
            optimal_precision,
            max_batch_size,
            preferred_memory_limit,
        };

        self.device_configurations.insert(device_id, config.clone());

        println!("Device {} configuration:", device_id);
        println!("  Name: {}", device_info.name);
        println!("  Device ID: {}", device_id);
        println!("  Optimal precision: {:?}", optimal_precision);
        println!("  Max batch size: {}", max_batch_size);
        println!("  Memory limit: {:.1} GB", preferred_memory_limit as f64 / 1e9);

        Ok(config)
    }

    fn get_device_config(&self, device_id: usize) -> Option<&DeviceConfig> {
        self.device_configurations.get(&device_id)
    }
}

// Usage
let mut optimizer = DeviceOptimizer::new();
for device_id in 0..4 {
    match optimizer.analyze_device(device_id) {
        Ok(_) => println!("Successfully analyzed device {}", device_id),
        Err(e) => println!("Device {} analysis failed: {}", device_id, e),
    }
}
```

### Multi-GPU Debugging Best Practices

1. **Always Verify Device IDs**: Use `device_id()` methods to ensure operations are on the expected device
2. **Track Memory Per Device**: Use separate memory pools for each device with device ID tracking
3. **Monitor Cross-Device Operations**: Be aware of memory transfers between devices
4. **Use Consistent Naming**: Include device ID in log messages and error reporting
5. **Test Device Fallback**: Ensure your application handles device unavailability gracefully

### Mixed Precision GPU Acceleration (New in PR #202)

BitNet.rs now provides comprehensive mixed precision support with native CUDA kernels for enhanced GPU performance:

#### MixedPrecisionKernel API

The `MixedPrecisionKernel` provides device-aware mixed precision operations:

```rust
use bitnet_kernels::gpu::{MixedPrecisionKernel, PrecisionMode};

// Create mixed precision kernel with automatic device detection
let mut kernel = MixedPrecisionKernel::new(0)?;
println!("Device: {}", kernel.device_info().name);
println!("Supports FP16: {}", kernel.supports_fp16());
println!("Supports BF16: {}", kernel.supports_bf16());
println!("Optimal precision: {:?}", kernel.optimal_precision());

// Perform matrix multiplication with automatic precision selection
let a = vec![1.0f32; 64 * 64];
let b = vec![2.0f32; 64 * 64];
let mut c = vec![0.0f32; 64 * 64];

kernel.matmul_auto(&a, &b, &mut c, 64, 64, 64)?;

// Get performance metrics
let metrics = kernel.metrics();
println!("Total operations: {}", metrics.total_operations);
println!("FP16 time: {:.2}ms", metrics.fp16_execution_time.as_secs_f64() * 1000.0);
println!("Memory allocated: {} MB", metrics.memory_allocated / (1024 * 1024));
```

#### Precision Mode Selection

```rust
// Explicit precision mode selection
kernel.set_precision_mode(PrecisionMode::FP16);
kernel.matmul_fp16(&a, &b, &mut c, 64, 64, 64)?;

kernel.set_precision_mode(PrecisionMode::BF16);
kernel.matmul_bf16(&a, &b, &mut c, 64, 64, 64)?;

// Automatic precision (recommended)
kernel.set_precision_mode(PrecisionMode::Auto);
kernel.matmul_auto(&a, &b, &mut c, 64, 64, 64)?;
```

#### Device Capability Detection

```rust
use bitnet_kernels::gpu::detect_best_precision;

// Automatic precision detection based on hardware
let optimal = detect_best_precision(kernel.device_info());
match optimal {
    PrecisionMode::BF16 => println!("Using BF16 (Ampere+ architecture)"),
    PrecisionMode::FP16 => println!("Using FP16 (Pascal+ architecture)"),
    PrecisionMode::FP32 => println!("Using FP32 (older architecture)"),
    PrecisionMode::Auto => unreachable!(),
}
```

#### Performance Monitoring

```rust
// Reset metrics for clean measurement
kernel.reset_metrics();

// Perform operations
for _ in 0..10 {
    kernel.matmul_auto(&a, &b, &mut c, 64, 64, 64)?;
}

// Analyze performance
let metrics = kernel.metrics();
println!("Average FP16 time: {:.2}ms", 
    metrics.fp16_execution_time.as_secs_f64() * 1000.0 / metrics.total_operations as f64);
println!("Memory efficiency: {:.1}%", 
    (kernel.current_memory_usage() as f64 / kernel.peak_memory_usage() as f64) * 100.0);
```

#### Memory Tracking

```rust
// Monitor GPU memory usage
println!("Current GPU memory: {} MB", kernel.current_memory_usage() / (1024 * 1024));
println!("Peak GPU memory: {} MB", kernel.peak_memory_usage() / (1024 * 1024));

// Memory transfer statistics
let metrics = kernel.metrics();
println!("Host to device transfers: {}", metrics.memory_transfers_h2d);
println!("Device to host transfers: {}", metrics.memory_transfers_d2h);
println!("Total bytes transferred: {} MB", 
    (metrics.bytes_transferred_h2d + metrics.bytes_transferred_d2h) / (1024 * 1024));
```

#### Supported Hardware Requirements

- **FP16 Support**: NVIDIA Pascal architecture or newer (Compute Capability 6.1+)
  - GTX 1080/1070, GTX 1060, Tesla P100, etc.
  - Uses native CUDA `__half` arithmetic
  - Automatic Tensor Core acceleration on CC 7.0+

- **BF16 Support**: NVIDIA Ampere architecture or newer (Compute Capability 8.0+)
  - RTX 3060/3070/3080/3090, A100, etc.
  - Uses native CUDA `__nv_bfloat16` arithmetic
  - Better numerical stability than FP16

- **Tensor Core Acceleration**: Available on CC 7.0+ with WMMA API
  - Volta (V100), Turing (RTX 20-series), Ampere (RTX 30-series, A100)
  - Automatic 16x16x16 matrix multiplication acceleration

#### Mixed Precision Commands

```bash
# Test mixed precision kernel creation and capabilities
cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_kernel_creation

# Test precision mode validation and device compatibility
cargo test -p bitnet-kernels --no-default-features --features gpu test_precision_mode_validation

# Test FP16 matrix multiplication accuracy
cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_fp16_accuracy

# Test BF16 matrix multiplication accuracy
cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_bf16_accuracy

# Benchmark mixed precision performance
cargo bench -p bitnet-kernels --bench mixed_precision_bench --no-default-features --features gpu

# Test precision conversion utilities
cargo test -p bitnet-kernels --no-default-features --features gpu test_precision_conversion_utilities

# Test memory tracking and performance metrics
cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_metrics_tracking
```

### Advanced GPU Infrastructure Access (New in PR #199)

BitNet.rs now provides access to low-level CUDA infrastructure for advanced GPU programming:

```rust
use bitnet_kernels::gpu::cuda::CudaKernel;
use cudarc::driver::{CudaModule, CudaFunction, LaunchConfig};

// Create CUDA kernel with full infrastructure access
let kernel = CudaKernel::new_with_device(0)?;

// Access CUDA context for advanced memory operations
let context = kernel.context();
println!("CUDA context access enabled for device {}", 
    kernel.device_info().device_id);

// Access CUDA module for loading custom kernels
let module = kernel.module();

// Load additional custom kernels from PTX
let custom_ptx = r#"
.version 8.0
.target sm_75
.address_size 64

.visible .entry my_custom_kernel(
    .param .u64 param_0
) {
    // Custom kernel implementation
    ret;
}
"#;

// Compile and load custom kernel (example usage)
// let custom_module = context.load_ptx(custom_ptx.into(), "my_custom_kernel", &[])?;
// let custom_function = custom_module.get_func("my_custom_kernel")?;

// Device-aware launch parameter optimization
let (block_size, grid_x, grid_y) = (16, 64, 64); // Would use internal calculation
let cfg = LaunchConfig {
    grid_dim: (grid_x as u32, grid_y as u32, 1),
    block_dim: (block_size as u32, block_size as u32, 1),
    shared_mem_bytes: 0,
};
```

**Use Cases for Advanced Infrastructure Access:**

1. **Custom Kernel Loading**: Load specialized PTX kernels for domain-specific operations
2. **Advanced Memory Management**: Implement custom memory pools with CUDA context access
3. **Multi-Stream Operations**: Coordinate multiple CUDA streams for overlapped execution
4. **Profiling Integration**: Hook into CUDA profiling APIs for performance analysis
5. **Inter-GPU Communication**: Implement peer-to-peer transfers between GPUs

### GPU Infrastructure Development Sequence

PR #199 is the first in a planned GPU infrastructure enhancement sequence:

#### **Phase 1: Foundation (PR #199) ✅**
- Expose CUDA context and module through public accessors
- Remove dead_code allowances from GPU infrastructure fields  
- Integrate optimal launch parameters in matrix multiplication operations
- Enable foundation for custom kernel loading and advanced GPU operations

#### **Phase 2: Advanced Management (PR #202) 🔄**
- Enhanced GPU memory management with custom allocation strategies
- Advanced custom kernel loading with PTX compilation pipeline
- Multi-stream coordination and overlapped execution support
- Performance profiling integration with CUDA events

#### **Phase 3: Multi-GPU Orchestration (PR #206) 📋**
- Multi-GPU support with device topology awareness
- Peer-to-peer memory transfers and communication
- Load balancing across multiple GPU devices
- Advanced GPU cluster management for distributed inference

**Testing Strategy for Infrastructure Sequence:**

```bash
# Phase 1 (PR #199): Foundation testing
cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_kernel_creation
cargo test -p bitnet-kernels --no-default-features --features gpu test_optimal_launch_params

# Phase 2 (PR #202): Advanced management testing  
cargo test -p bitnet-kernels --no-default-features --features gpu test_custom_kernel_loading
cargo test -p bitnet-kernels --no-default-features --features gpu test_advanced_memory_pools

# Phase 3 (PR #206): Multi-GPU testing
cargo test -p bitnet-kernels --no-default-features --features gpu test_multi_gpu_coordination
cargo test -p bitnet-kernels --no-default-features --features gpu test_peer_to_peer_transfers
```

### Integration with Quantization

The CUDA device querying integrates with BitNet's quantization system:

- **Device-Aware Quantization**: Selects optimal quantization kernels based on compute capability
- **Automatic GPU Acceleration**: Falls back to CPU when GPU is unavailable or insufficient
- **Memory-Constrained Operation**: Adjusts quantization batch sizes based on available memory
- **Performance Monitoring**: Tracks GPU utilization and performance across operations
- **Host Memory Tracking**: Real-time monitoring of system memory usage with detailed statistics

## GPU Testing Strategy

GPU testing requires special consideration due to hardware dependencies and resource management:

### Test Classification by Hardware Requirements

```bash
# Always available (no GPU required)
cargo test --workspace --no-default-features --features cpu

# GPU smoke tests (basic availability, run on CI with GPU)
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_smoke

# GPU integration tests (comprehensive, manual execution)
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_quantization --ignored

# GPU performance tests (benchmarking, development only)
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_performance --ignored
```

### GPU Test Categories

- **Availability Tests**: Check CUDA installation and device access
- **Functionality Tests**: Verify GPU operations produce correct results
- **Accuracy Tests**: Compare GPU vs CPU results for numerical consistency
- **Performance Tests**: Benchmark GPU acceleration vs CPU baseline
- **Resource Tests**: Validate memory management and concurrent operations
- **Fallback Tests**: Ensure graceful degradation when GPU unavailable

### Hardware-Specific Test Configuration

```bash
# Test matrix for different hardware scenarios
SCENARIOS=(
  "no-gpu:cpu-only"
  "gpu-low-mem:4gb-gpu" 
  "gpu-mid-mem:8gb-16gb-gpu"
  "gpu-high-mem:16gb-plus-gpu"
  "multi-gpu:multiple-devices"
)

# Compute capability matrix
CC_TARGETS=(
  "6.0:maxwell-pascal"
  "7.0:volta" 
  "8.0:ampere"
  "9.0:hopper"
)
```

### GPU Test Best Practices

- Use `#[ignore]` for hardware-dependent tests
- Implement comprehensive error handling and fallback testing
- Test both successful GPU operations and failure scenarios
- Validate memory cleanup and resource management
- Include cross-device testing for multi-GPU scenarios
- Test performance regression detection

### CI/CD GPU Testing

- **Tier 1**: CPU-only tests (always run)
- **Tier 2**: GPU availability and smoke tests (run on GPU CI)
- **Tier 3**: Integration tests with `--ignored` (manual/scheduled)
- **Tier 4**: Performance and multi-GPU tests (development/release)

For comprehensive test execution strategies and test suite configuration, see the [Test Suite Guide](test-suite.md).

## GPU/CUDA Development Best Practices

### PR Scope Management for GPU Features

Based on lessons learned from PR #102, follow these guidelines for GPU/CUDA development:

1. **Break Large Features into Focused PRs**:
   - **Device Querying PR**: Focus only on CUDA device property querying
   - **Quantization Enhancement PR**: Focus only on device-aware quantization
   - **Memory Management PR**: Focus only on GPU memory optimization
   - **Integration PR**: Combine smaller, well-tested components

2. **GPU Feature Development Workflow**:
   ```bash
   # Step 1: Implement core CUDA functionality (small PR)
   cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_smoke
   
   # Step 2: Add device querying (focused PR) 
   cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_device_info_query
   
   # Step 3: Enhance quantization with device awareness (focused PR)
   cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_vs_cpu_quantization_accuracy
   
   # Step 4: Integration testing (final PR)
   cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_quantization --ignored
   ```

3. **CUDA Implementation Guidelines**:
   - Use cudarc API consistently for all CUDA operations
   - Implement comprehensive error handling with meaningful messages
   - Add automatic CPU fallback for GPU operation failures
   - Include device capability detection for optimization
   - Test on multiple CUDA compute capabilities when possible

### Hardware-Dependent Testing Strategy

GPU/CUDA tests require special handling due to hardware dependencies:

1. **Test Categories**:
   - **Smoke Tests**: Basic functionality, run on CI
   - **Integration Tests**: Marked with `#[ignore]`, run manually with `--ignored`
   - **Performance Tests**: Benchmark comparisons, run locally
   - **Cross-Device Tests**: Multiple GPU testing, manual verification

2. **CI/CD Considerations**:
   - Default tests should pass without GPU hardware
   - Use feature gates to conditionally compile GPU code
   - Provide clear error messages when GPU is unavailable
   - Include CPU fallback path testing in all scenarios

## Memory Tracking and Performance Monitoring

### Host Memory Statistics

BitNet.rs now includes comprehensive host memory tracking using the `sysinfo` crate, providing real-time monitoring of system memory usage alongside GPU operations.

#### DeviceStats with Memory Tracking

The `DeviceStats` structure now includes actual memory usage statistics:

```rust
use bitnet_kernels::device_aware::DeviceAwareQuantizer;

let quantizer = DeviceAwareQuantizer::new(Device::Cpu)?;

// Perform some operations
let input = vec![1.0f32; 1024];
let mut output = vec![0u8; 256];
let mut scales = vec![0.0f32; 8];
quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S)?;

// Get comprehensive statistics including memory usage
if let Some(stats) = quantizer.get_stats() {
    println!("Device stats: {}", stats.summary());
    println!("Memory used: {:.2} MB", stats.memory_used_bytes as f64 / (1024.0 * 1024.0));
    println!("Memory total: {:.2} MB", stats.memory_total_bytes as f64 / (1024.0 * 1024.0));
    println!("Memory usage: {:.1}%", 
        (stats.memory_used_bytes as f64 / stats.memory_total_bytes as f64) * 100.0);
}
```

#### Memory Tracking Features

- **Real-time Monitoring**: Memory statistics are updated on each request using `sysinfo::System`
- **Byte-accurate Reporting**: Both used and total memory reported in bytes for precise tracking
- **Human-readable Display**: The `summary()` method includes memory usage with percentage
- **Performance Integration**: Memory tracking integrated with existing performance statistics

#### Platform-Specific CPU Kernel Selection

The device-aware quantizer now automatically selects the best CPU kernel based on platform architecture:

```rust
// Automatic platform detection and optimization
let quantizer = DeviceAwareQuantizer::new(Device::Cpu)?;
println!("Active kernel: {}", quantizer.active_provider());

// Expected outputs:
// - x86_64 with AVX2: "AVX2Kernel"  
// - aarch64 with NEON: "NeonKernel"
// - Fallback systems: "FallbackKernel"
```

#### Memory Tracking Commands

```bash
# Test comprehensive memory tracking implementation with device-aware stats
cargo test -p bitnet-kernels --no-default-features --features cpu test_memory_tracking

# Test device-aware performance tracking with integrated memory statistics
cargo test -p bitnet-kernels --no-default-features --features cpu test_performance_tracking

# Test platform-specific kernel selection with memory monitoring
cargo test -p bitnet-kernels --no-default-features --features cpu test_platform_kernel_selection

# Test CPU provider creation across architectures
cargo test -p bitnet-kernels --no-default-features --features cpu test_cpu_provider_creation

# Architecture-specific feature detection tests
cargo test -p bitnet-kernels --no-default-features --features cpu test_x86_64_feature_detection  # x86_64 only
cargo test -p bitnet-kernels --no-default-features --features cpu test_aarch64_feature_detection  # aarch64 only
```

#### Memory and Performance Analysis

The enhanced statistics provide comprehensive monitoring capabilities:

```rust
#[derive(Debug, Clone)]
pub struct DeviceStats {
    pub device_type: String,
    pub target_device: Device,
    pub total_operations: u64,
    pub quantization_operations: u64,
    pub matmul_operations: u64,
    pub total_time_ms: f64,
    pub quantization_time_ms: f64,
    pub matmul_time_ms: f64,
    pub gpu_operations: u64,
    pub cpu_operations: u64,
    pub fallback_count: u64,
    pub gpu_efficiency: f64,         // Ratio of GPU operations to total operations
    pub last_gpu_error: Option<String>,
    pub last_cpu_error: Option<String>,
    pub memory_used_bytes: u64,      // Host memory currently used in bytes
    pub memory_total_bytes: u64,     // Total host memory available in bytes
}
```

Key statistics methods:
- `summary()`: Human-readable summary with memory usage percentage
- `is_gpu_effective()`: Checks if GPU is being used effectively (>80% efficiency)
- `avg_quantization_time_ms()`: Average time per quantization operation
- `avg_matmul_time_ms()`: Average time per matrix multiplication operation

## Advanced GPU/CUDA Troubleshooting

### Memory Debugging and Stack Trace Analysis (New in PR #201)

BitNet.rs now provides comprehensive memory debugging capabilities with stack trace capture for production debugging:

1. **Memory Leak Investigation with Stack Traces**:
   ```bash
   # Build in debug mode to enable stack trace capture
   cargo build --debug -p bitnet-kernels --no-default-features --features gpu

   # Run tests with detailed memory leak detection
   cargo test -p bitnet-kernels --no-default-features --features gpu test_check_leaks -- --nocapture

   # Enable comprehensive logging for memory operations
   RUST_LOG=debug cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_optimization -- --nocapture

   # Test long-running memory patterns (look for stack traces in output)
   cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_allocation -- --nocapture
   ```

2. **Device ID Tracking for Multi-GPU Debugging**:
   ```bash
   # Test device ID consistency across operations
   cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_device_tracking

   # Validate memory pool device assignment
   cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_pool_creation

   # Test multi-device scenarios (requires multiple GPUs)
   CUDA_VISIBLE_DEVICES=0,1 cargo test -p bitnet-kernels --no-default-features --features gpu test_multi_device_memory_pools --ignored
   ```

3. **Memory Access Pattern Analysis**:
   ```bash
   # Analyze memory access patterns for optimization opportunities
   cargo test -p bitnet-kernels --no-default-features --features gpu test_access_pattern_analysis

   # Test pattern detection algorithms
   cargo test -p bitnet-kernels --no-default-features --features gpu test_analyze_access_pattern

   # Run memory layout optimization tests
   cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_layout_optimization
   ```

4. **Interpreting Stack Trace Output**:

   When memory leaks are detected in debug builds, you'll see output like:
   ```
   Device 0: potential leak: 1048576 bytes at 0x7f8b4c000000
   Stack trace:
      0: std::backtrace::Backtrace::force_capture
         at /rustc/.../library/std/src/backtrace.rs:101
      1: bitnet_kernels::gpu::memory_optimization::OptimizedMemoryPool::track_allocation
         at crates/bitnet-kernels/src/gpu/memory_optimization.rs:166
      2: bitnet_kernels::gpu::memory_optimization::OptimizedMemoryPool::allocate
         at crates/bitnet-kernels/src/gpu/memory_optimization.rs:123
      3: my_application::process_batch
         at src/main.rs:45
      4: my_application::main
         at src/main.rs:20
   ```

   This tells you:
   - **Device ID**: Which GPU device has the leak
   - **Memory size**: How much memory was leaked
   - **Memory address**: The pointer address of the leaked memory
   - **Call stack**: Exact code path that led to the allocation

5. **Production Memory Debugging**:
   ```bash
   # Enable memory leak detection in production builds (minimal overhead)
   cargo build --release -p bitnet-kernels --no-default-features --features gpu

   # Set leak detection threshold (default: 1 hour)
   export BITNET_LEAK_THRESHOLD_SECS=3600

   # Configure memory pool settings for production
   export BITNET_POOL_SIZE_GB=4
   export BITNET_CLEANUP_INTERVAL_SECS=60
   ```

### Mixed Precision Issues (New in PR #202)

1. **Mixed Precision Kernel Creation Fails**:
   ```bash
   # Test mixed precision kernel initialization
   cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_kernel_creation
   
   # Check device capabilities
   cargo run --example gpu_validation --no-default-features --features gpu | grep -E "FP16|BF16"
   
   # Test with specific device
   CUDA_VISIBLE_DEVICES=0 cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_kernel_creation
   ```

2. **Precision Mode Not Supported**:
   ```bash
   # Check compute capability requirements
   # FP16 requires CC 6.1+, BF16 requires CC 8.0+
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader
   
   # Test precision detection logic
   cargo test -p bitnet-kernels --no-default-features --features gpu test_precision_detection_optimization
   
   # Verify PTX kernel compilation
   cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_ptx_compilation
   ```

3. **Matrix Multiplication Accuracy Issues**:
   ```bash
   # Test FP16 vs FP32 accuracy
   cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_fp16_accuracy --ignored
   
   # Test BF16 vs FP32 accuracy
   cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_bf16_accuracy --ignored
   
   # Check numerical precision simulation
   cargo test -p bitnet-kernels --no-default-features --features gpu test_precision_conversion_utilities
   ```

4. **Performance Degradation**:
   ```bash
   # Benchmark precision modes
   cargo bench -p bitnet-kernels --bench mixed_precision_bench --no-default-features --features gpu
   
   # Check Tensor Core utilization (CC 7.0+)
   cargo test -p bitnet-kernels --no-default-features --features gpu test_tensor_core_acceleration --ignored
   
   # Monitor GPU memory efficiency
   cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_metrics_tracking
   ```

5. **Memory Issues with Mixed Precision**:
   ```bash
   # Test GPU memory tracking
   cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_memory_management --ignored
   
   # Check for memory leaks
   cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_memory_cleanup --ignored
   
   # Monitor memory transfer efficiency
   nvidia-smi dmon -s puc -d 1  # Monitor during mixed precision operations
   ```

6. **PTX Compilation Errors**:
   ```bash
   # Check CUDA toolkit compatibility
   nvcc --version
   
   # Test PTX compilation manually
   cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_kernel_loading
   
   # Enable CUDA compilation debugging
   RUST_LOG=debug cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_kernel_creation
   ```

**Common Error Messages and Solutions**:

- **"FP16 not supported on this device"**: Upgrade to Pascal (GTX 10-series) or newer GPU
- **"BF16 not supported on this device"**: Upgrade to Ampere (RTX 30-series) or newer GPU
- **"Failed to compile mixed precision PTX"**: Check CUDA toolkit installation and version compatibility
- **"Tensor Core operations require CC 7.0+"**: Use standard FP16 kernels on older architectures
- **"Mixed precision kernel not available"**: Verify PTX compilation succeeded and device capabilities

### GPU Backend Detection Issues

1. **GPU Detection Fails**:
   ```bash
   # Test GPU detection manually
   cargo test -p bitnet-kernels --no-default-features test_gpu_info_summary
   
   # Check system tools availability
   which nvidia-smi rocm-smi
   
   # Test with mock environment
   BITNET_GPU_FAKE="cuda" cargo run -p xtask -- download-model --dry-run
   ```

2. **Incorrect Backend Detection**:
   ```bash
   # Verify system detection (using existing GPU validation example)
   cargo run --example gpu_validation --no-default-features --features gpu
   
   # Override detection for testing
   export BITNET_GPU_FAKE="cuda,metal"
   cargo test -p bitnet-kernels test_gpu_info_mocked_scenarios
   ```

3. **Version Detection Issues**:
   ```bash
   # Check CUDA toolkit installation
   nvcc --version
   which nvcc
   
   # Check ROCm installation  
   rocm-smi --version
   which rocm-smi
   
   # Test GPU detection functionality
   cargo test -p bitnet-kernels --no-default-features test_gpu_info_summary
   ```

4. **Missing System Commands**:
   ```bash
   # Install missing NVIDIA tools
   sudo apt-get install nvidia-utils-* nvidia-cuda-toolkit
   
   # Install missing AMD tools
   sudo apt-get install rocm-smi-lib rocm-dev
   
   # Verify installation
   nvidia-smi --query-gpu=gpu_name --format=csv,noheader
   rocm-smi --showid
   ```

### GPU Detection and Initialization Issues

1. **CUDA Driver/Runtime Mismatch**:
   ```bash
   # Check NVIDIA driver version
   nvidia-smi
   
   # Check CUDA runtime version
   nvcc --version
   
   # Verify cudarc compatibility
   cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_kernel_creation
   ```

2. **GPU Memory Issues**:
   ```bash
   # Monitor GPU memory usage
   nvidia-smi -l 1  # Update every second
   
   # Test memory allocation patterns
   cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_memory_management --ignored
   
   # Check for memory leaks
   cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_cleanup --ignored
   
   # Test host memory tracking and comprehensive device statistics
   cargo test -p bitnet-kernels --no-default-features --features cpu test_memory_tracking
   
   # Test performance tracking with memory integration
   cargo test -p bitnet-kernels --no-default-features --features cpu test_performance_tracking
   ```

3. **Compute Capability Issues**:
   ```bash
   # Query device compute capability
   cargo run --example gpu_validation --no-default-features --features gpu | grep "compute capability"
   
   # Test operations on different compute capabilities
   cargo test -p bitnet-kernels --no-default-features --features gpu test_device_capability_validation
   ```

### Performance Monitoring and Analysis

1. **Comprehensive Performance Tracking**:
   ```bash
   # Test comprehensive GPU performance monitoring
   cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_validation_comprehensive
   
   # Validate performance metrics collection
   cargo test -p bitnet-inference --features integration-tests test_engine_performance_tracking_integration
   
   # Test memory usage tracking with device-aware execution
   cargo test -p bitnet-kernels --no-default-features --features cpu test_memory_tracking
   cargo test -p bitnet-kernels --no-default-features --features cpu test_performance_tracking
   ```

2. **GPU Performance Analysis**:
   ```bash
   # Run comprehensive performance comparison
   cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_vs_cpu_quantization_accuracy --ignored
   
   # Profile GPU kernel execution
   cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_numerical_accuracy --ignored
   
   # GPU memory leak detection and performance benchmarking
   cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_memory_management
   ```

3. **Platform-Specific Performance Testing**:
   ```bash
   # Test platform-specific CPU kernel selection with performance monitoring
   cargo test -p bitnet-kernels --no-default-features --features cpu test_cpu_provider_creation
   
   # Test architecture-specific feature detection
   cargo test -p bitnet-kernels --no-default-features --features cpu test_x86_64_feature_detection  # x86_64 only
   cargo test -p bitnet-kernels --no-default-features --features cpu test_aarch64_feature_detection  # aarch64 only
   ```

### Performance Debugging

1. **Memory Transfer Optimization**:
   ```bash
   # Test memory access patterns
   cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_access_patterns --ignored
   
   # Validate optimized memory layouts
   cargo test -p bitnet-kernels --no-default-features --features gpu test_memory_optimization --ignored
   ```

### Fallback and Error Handling

1. **GPU Unavailable Scenarios**:
   ```bash
   # Test CPU fallback when GPU unavailable
   CUDA_VISIBLE_DEVICES="" cargo test --workspace --no-default-features --features gpu
   
   # Test partial GPU failure handling
   cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_quantization_fallback --ignored
   ```

2. **Multi-GPU Configuration**:
   ```bash
   # Test device selection
   cargo test -p bitnet-kernels --no-default-features --features gpu test_multi_gpu_selection --ignored
   
   # Test concurrent operations
   cargo test -p bitnet-kernels --no-default-features --features gpu test_concurrent_gpu_operations --ignored
   ```

### Common Error Messages and Solutions

1. **"CUDA driver version is insufficient"**:
   - Update NVIDIA drivers to support installed CUDA toolkit
   - Check compatibility matrix at nvidia.com/drivers

2. **"out of memory" during GPU operations**:
   - Reduce batch sizes or model parameters
   - Enable GPU memory management optimizations
   - Check for memory leaks in previous operations

3. **"device kernel execution timed out"**:
   - Reduce operation complexity or batch size
   - Check for infinite loops in CUDA kernels
   - Monitor GPU temperature and power limits

4. **"no CUDA-capable device is detected"**:
   - Verify GPU is CUDA-compatible (not AMD/Intel)
   - Check GPU is not being used by other processes
   - Ensure proper driver installation

### Debug Logging and Monitoring

1. **Enable GPU Debug Logging**:
   ```bash
   # Enable CUDA-specific logging
   RUST_LOG=bitnet_kernels::gpu=debug cargo test -p bitnet-kernels --no-default-features --features gpu
   
   # Enable cudarc internal logging
   CUDA_LOG_LEVEL=debug cargo test -p bitnet-kernels --no-default-features --features gpu
   ```

2. **Performance Monitoring**:
   ```bash
   # Profile GPU operations
   nsys profile cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_performance --ignored
   
   # Monitor GPU utilization
   nvidia-smi dmon -s puc -d 1
   ```

## GPU Development Recipes

```bash
# GPU backend detection and availability
cargo test -p bitnet-kernels --no-default-features test_gpu_info_summary

# Mock GPU testing scenarios
BITNET_GPU_FAKE="cuda,rocm" cargo test -p bitnet-kernels test_gpu_info_mocked_scenarios

# GPU smoke test (basic availability)
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_smoke

# Enhanced GPU Memory Debugging and Stack Trace Recipes (New in PR #201)

# Memory leak detection with comprehensive stack traces (debug builds)
cargo test -p bitnet-kernels --features gpu test_check_leaks -- --nocapture

# Device ID tracking and memory pool management
cargo test -p bitnet-kernels --features gpu test_memory_pool_creation
cargo test -p bitnet-kernels --features gpu test_mixed_precision_device_tracking

# Memory access pattern analysis and optimization
cargo test -p bitnet-kernels --features gpu test_access_pattern_analysis
cargo test -p bitnet-kernels --features gpu test_analyze_access_pattern

# Multi-GPU device tracking (requires multiple GPUs)
CUDA_VISIBLE_DEVICES=0,1 cargo test -p bitnet-kernels --features gpu test_multi_device_memory_pools --ignored

# Production memory debugging with minimal overhead
cargo build --release -p bitnet-kernels --no-default-features --features gpu
BITNET_LEAK_THRESHOLD_SECS=3600 cargo test -p bitnet-kernels --features gpu test_memory_optimization

# Comprehensive memory debugging with full logging
RUST_LOG=debug cargo test -p bitnet-kernels --features gpu test_memory_optimization -- --nocapture

# CUDA device information and capabilities
cargo run --example gpu_validation --no-default-features --features gpu

# GPU quantization accuracy validation
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_vs_cpu_quantization_accuracy --ignored

# GPU memory management and cleanup testing
cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_memory_management --ignored

# GPU fallback mechanism testing
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_quantization_fallback --ignored

# Multi-GPU and concurrent operations
cargo test -p bitnet-kernels --no-default-features --features gpu test_concurrent_gpu_operations --ignored

# GPU numerical accuracy verification
cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_numerical_accuracy --ignored

# GPU vs CPU parity testing across quantization schemes
cargo test --workspace --no-default-features --features cuda gpu_parity

# Mixed precision GPU operations (New in PR #202)
# Test mixed precision kernel creation and device detection
cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_kernel_creation

# Test automatic precision mode selection
cargo test -p bitnet-kernels --no-default-features --features gpu test_precision_detection_optimization

# Test FP16 matrix multiplication accuracy
cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_fp16_accuracy --ignored

# Test BF16 matrix multiplication accuracy 
cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_bf16_accuracy --ignored

# Mixed precision performance benchmarks
cargo bench -p bitnet-kernels --bench mixed_precision_bench --no-default-features --features gpu

# Test Tensor Core acceleration (CC 7.0+)
cargo test -p bitnet-kernels --no-default-features --features gpu test_tensor_core_acceleration --ignored

# Test precision conversion utilities
cargo test -p bitnet-kernels --no-default-features --features gpu test_precision_conversion_utilities

# Test mixed precision memory management
cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_memory_management --ignored

# Test comprehensive mixed precision metrics
cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_metrics_tracking
```