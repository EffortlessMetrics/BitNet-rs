# CUDA Configuration and Memory Optimization Guide

This guide provides detailed instructions for configuring CUDA and optimizing memory usage with BitNet.rs GPU kernels.

## Table of Contents

1. [CUDA Configuration](#cuda-configuration)
2. [Memory Pool Configuration](#memory-pool-configuration)
3. [Device-Specific Optimization](#device-specific-optimization)
4. [Memory Access Pattern Optimization](#memory-access-pattern-optimization)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)

## CUDA Configuration

### Environment Setup

#### Required Environment Variables
```bash
# Core CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Optional: Force specific GPU devices
export CUDA_VISIBLE_DEVICES=0,1  # Only use GPUs 0 and 1

# Optional: Set compute capability target
export CUDA_ARCH=sm_80  # For RTX 30xx series
```

#### CUDA Runtime Configuration
```bash
# Memory management
export CUDA_LAUNCH_BLOCKING=0        # Async kernel launches (default)
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Consistent device ordering

# Memory pool settings
export CUDA_MEMORY_POOL_SIZE=2G      # Pre-allocate 2GB memory pool
export CUDA_MEMORY_POOL_TRIM=1       # Enable automatic memory trimming
```

### Device Selection and Validation

```rust
use bitnet_kernels::gpu::cuda::{CudaKernel, list_cuda_devices, cuda_device_count};

// Check CUDA availability
if !bitnet_kernels::gpu::cuda::is_cuda_available() {
    eprintln!("CUDA not available on this system");
    return Err("CUDA required".into());
}

// List all available devices
let devices = list_cuda_devices()?;
println!("Found {} CUDA devices:", devices.len());

for device in &devices {
    println!("Device {}: {}", device.device_id, device.name);
    println!("  Compute capability: {}.{}", 
             device.compute_capability.0, device.compute_capability.1);
    println!("  Total memory: {:.2} GB", 
             device.total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("  Multiprocessors: {}", device.multiprocessor_count);
    println!("  Max threads per block: {}", device.max_threads_per_block);
    println!("  Max shared memory: {} KB", 
             device.max_shared_memory_per_block / 1024);
    println!("  FP16 support: {}", device.supports_fp16);
    println!("  BF16 support: {}", device.supports_bf16);
    println!();
}

// Select optimal device (highest memory)
let optimal_device = devices.iter()
    .max_by_key(|d| d.total_memory)
    .map(|d| d.device_id)
    .unwrap_or(0);

println!("Selected optimal device: {}", optimal_device);
```

### Kernel Configuration

```rust
use bitnet_kernels::gpu::cuda::CudaKernel;

// Create kernel with specific device
let kernel = CudaKernel::new_with_device(optimal_device)?;

// Verify kernel is available
if !kernel.is_available() {
    return Err("CUDA kernel not available".into());
}

println!("CUDA kernel initialized successfully");
println!("Kernel provider: {}", kernel.name());

// Get device information
let device_info = kernel.device_info();
println!("Device info: {:?}", device_info);
```

## Memory Pool Configuration

### Basic Memory Pool Setup

```rust
use bitnet_kernels::gpu::memory_optimization::{
    OptimizedMemoryPool, MemoryPoolConfig
};
use std::time::Duration;

// Configure memory pool for production use
let config = MemoryPoolConfig {
    // Maximum pool size (should be < 90% of GPU memory)
    max_pool_size: 6 * 1024 * 1024 * 1024, // 6GB for 8GB GPU
    
    // Maximum number of cached buffers per size
    max_cached_buffers: 1000,
    
    // Enable detailed memory tracking
    enable_memory_tracking: true,
    
    // Cleanup frequency (balance between performance and memory usage)
    cleanup_interval: Duration::from_secs(30),
};

let mut memory_pool = OptimizedMemoryPool::new(device_id, config);
```

### Dynamic Memory Management

```rust
// Allocate memory from pool
let buffer_size = 10 * 1024 * 1024; // 10MB
let buffer = memory_pool.allocate(buffer_size)?;

// Check memory statistics
let stats = memory_pool.stats();
println!("Memory Statistics:");
println!("  Current usage: {:.2} MB", 
         stats.current_usage as f64 / (1024.0 * 1024.0));
println!("  Peak usage: {:.2} MB", 
         stats.peak_usage as f64 / (1024.0 * 1024.0));
println!("  Total allocated: {:.2} MB", 
         stats.total_allocated as f64 / (1024.0 * 1024.0));
println!("  Cache hit rate: {:.1}%", 
         stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64 * 100.0);

// Return buffer to pool for reuse
memory_pool.deallocate(buffer);

// Periodic maintenance
if memory_pool.stats().current_usage > config.max_pool_size / 2 {
    // Force cleanup if using too much memory
    memory_pool.cleanup_expired_buffers();
}
```

### Memory Leak Detection

```rust
// Check for memory leaks
let leaks = memory_pool.check_leaks();
if !leaks.is_empty() {
    eprintln!("Memory leaks detected:");
    for leak in leaks {
        eprintln!("  {}", leak);
    }
    
    // Log leak information for debugging
    log::error!("Memory leaks detected: {:?}", leaks);
}

// Reset statistics for fresh monitoring
memory_pool.reset_stats();
```

## Device-Specific Optimization

### Compute Capability-Based Configuration

```rust
fn configure_for_compute_capability(
    device_info: &CudaDeviceInfo
) -> Result<LaunchConfig> {
    let (major, minor) = device_info.compute_capability;
    
    match major {
        // Pascal (GTX 10xx, Tesla P100)
        6 => {
            LaunchConfig {
                grid_dim: (32, 32, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            }
        },
        
        // Volta/Turing (RTX 20xx, Tesla V100)
        7 => {
            LaunchConfig {
                grid_dim: (64, 64, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 8192, // Use more shared memory
            }
        },
        
        // Ampere (RTX 30xx, A100)
        8 => {
            LaunchConfig {
                grid_dim: (128, 128, 1),
                block_dim: (32, 32, 1),
                shared_mem_bytes: 16384, // Even more shared memory
            }
        },
        
        // Ada Lovelace/Hopper (RTX 40xx, H100)
        9 => {
            LaunchConfig {
                grid_dim: (256, 256, 1),
                block_dim: (32, 32, 1),
                shared_mem_bytes: 32768, // Maximum shared memory
            }
        },
        
        _ => {
            // Default configuration for unknown architectures
            LaunchConfig {
                grid_dim: (32, 32, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            }
        }
    }
}
```

### Mixed Precision Configuration

```rust
use bitnet_kernels::gpu::mixed_precision::{MixedPrecisionKernel, PrecisionMode};

let mut mixed_kernel = MixedPrecisionKernel::new(device_id)?;

// Configure precision based on hardware capabilities
let precision_mode = match device_info.compute_capability {
    (major, _) if major >= 8 => {
        // Modern GPUs support BF16
        if mixed_kernel.supports_bf16() {
            PrecisionMode::BF16
        } else {
            PrecisionMode::FP16
        }
    },
    (major, _) if major >= 6 => {
        // Tensor Core GPUs support FP16
        if mixed_kernel.supports_fp16() {
            PrecisionMode::FP16
        } else {
            PrecisionMode::FP32
        }
    },
    _ => {
        // Older GPUs use FP32
        PrecisionMode::FP32
    }
};

mixed_kernel.set_precision_mode(precision_mode);
println!("Using precision mode: {:?}", precision_mode);
```

## Memory Access Pattern Optimization

### Access Pattern Analysis

```rust
use bitnet_kernels::gpu::memory_optimization::{
    MemoryLayoutOptimizer, AccessPattern
};

// Analyze memory access patterns for optimization
fn optimize_memory_layout<T>(
    data: &mut [T], 
    access_indices: &[usize]
) -> AccessPattern {
    let pattern = MemoryLayoutOptimizer::analyze_access_pattern(access_indices);
    
    match pattern {
        AccessPattern::Sequential => {
            // Already optimal for GPU
            println!("Sequential access detected - no optimization needed");
        },
        
        AccessPattern::Strided { stride } => {
            println!("Strided access detected with stride {}", stride);
            // Could reorganize data to improve coalescing
            MemoryLayoutOptimizer::optimize_layout(data, pattern);
        },
        
        AccessPattern::Random => {
            println!("Random access detected - consider data reorganization");
            // May benefit from tiling or blocking strategies
            MemoryLayoutOptimizer::optimize_layout(data, pattern);
        },
    }
    
    pattern
}

// Example usage
let mut matrix_data = vec![0.0f32; 1024 * 1024];
let access_pattern = vec![0, 1, 2, 3, 4]; // Sequential
let pattern = optimize_memory_layout(&mut matrix_data, &access_pattern);
```

### Memory Alignment Optimization

```rust
// Calculate optimal alignment for different data sizes
fn optimize_memory_alignment(data_size: usize) -> usize {
    let alignment = MemoryLayoutOptimizer::calculate_alignment(data_size);
    
    println!("Data size: {} bytes", data_size);
    println!("Optimal alignment: {} bytes", alignment);
    
    // Ensure alignment is suitable for GPU memory coalescing
    let coalesced_alignment = if alignment < 128 {
        128 // Minimum for good coalescing
    } else {
        alignment
    };
    
    println!("GPU-optimized alignment: {} bytes", coalesced_alignment);
    coalesced_alignment
}

// Example for different data sizes
let alignments = vec![
    (1024, optimize_memory_alignment(1024)),
    (64 * 1024, optimize_memory_alignment(64 * 1024)),
    (1024 * 1024, optimize_memory_alignment(1024 * 1024)),
];

for (size, alignment) in alignments {
    println!("Size: {} KB -> Alignment: {} bytes", size / 1024, alignment);
}
```

## Performance Tuning

### Kernel Launch Optimization

```rust
impl CudaKernel {
    /// Calculate optimal launch parameters for matrix multiplication
    fn calculate_optimal_launch_params(
        &self, 
        m: usize, 
        n: usize, 
        k: usize
    ) -> LaunchConfig {
        let device_info = self.device_info();
        
        // Calculate block size based on shared memory and thread limits
        let max_threads = device_info.max_threads_per_block as usize;
        let max_shared_mem = device_info.max_shared_memory_per_block;
        
        // Estimate shared memory usage per thread
        let shared_mem_per_element = 2 * std::mem::size_of::<i8>();
        
        // Find largest square block that fits constraints
        let mut block_size = 16;
        while block_size <= 32 {
            let threads_needed = block_size * block_size;
            let shared_mem_needed = 2 * block_size * block_size * shared_mem_per_element;
            
            if threads_needed > max_threads || shared_mem_needed > max_shared_mem {
                block_size /= 2;
                break;
            }
            block_size *= 2;
        }
        
        // Ensure minimum block size for efficiency
        block_size = block_size.clamp(8, 32);
        
        // Calculate grid dimensions
        let grid_x = (m + block_size - 1) / block_size;
        let grid_y = (n + block_size - 1) / block_size;
        
        // Optimize for SM occupancy
        let sm_count = device_info.multiprocessor_count as usize;
        let total_blocks = grid_x * grid_y;
        
        // Adjust if we have too few blocks per SM
        let blocks_per_sm = total_blocks / sm_count;
        if blocks_per_sm < 4 && block_size > 8 {
            // Use smaller blocks for better occupancy
            block_size /= 2;
        }
        
        println!("Optimal kernel configuration:");
        println!("  Block size: {}x{}", block_size, block_size);
        println!("  Grid size: {}x{}", grid_x, grid_y);
        println!("  Total blocks: {}", total_blocks);
        println!("  Blocks per SM: {:.1}", total_blocks as f64 / sm_count as f64);
        
        LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 2 * block_size * block_size * shared_mem_per_element,
        }
    }
}
```

### Batch Processing Optimization

```rust
use bitnet_kernels::gpu::cuda::{BatchMatmulParams, CudaKernel};

impl CudaKernel {
    /// Optimized batch matrix multiplication
    pub fn batch_matmul_optimized(
        &self,
        batches: &mut [BatchMatmulParams<'_>]
    ) -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }
        
        // Sort batches by size for better memory locality
        batches.sort_by_key(|(a, _b, _c, m, n, k)| {
            ((*m / 256) * 256, (*n / 256) * 256, (*k / 256) * 256)
        });
        
        // Process batches in groups to maximize GPU utilization
        const BATCH_SIZE: usize = 8;
        
        for batch_group in batches.chunks_mut(BATCH_SIZE) {
            // Process group concurrently
            for (a, b, c, m, n, k) in batch_group.iter_mut() {
                self.launch_matmul(a, b, c, *m, *n, *k)?;
            }
            
            // Synchronize after each group to prevent memory buildup
            self.synchronize_all()?;
        }
        
        Ok(())
    }
}
```

## Troubleshooting

### Memory Issues

#### Out of Memory Errors
```rust
// Check available GPU memory before allocation
let (free_memory, total_memory) = kernel.memory_stats();
let memory_usage = (total_memory - free_memory) as f64 / total_memory as f64;

if memory_usage > 0.9 {
    log::warn!("GPU memory usage high: {:.1}%", memory_usage * 100.0);
    
    // Force garbage collection
    memory_pool.cleanup_expired_buffers();
    
    // Reduce allocation size if still high
    let updated_stats = kernel.memory_stats();
    let updated_usage = (updated_stats.1 - updated_stats.0) as f64 / updated_stats.1 as f64;
    
    if updated_usage > 0.85 {
        return Err("GPU memory usage too high".into());
    }
}
```

#### Memory Fragmentation
```rust
// Monitor memory fragmentation
let stats = memory_pool.stats();
let fragmentation_ratio = stats.allocation_count as f64 / stats.deallocation_count as f64;

if fragmentation_ratio > 1.5 {
    log::warn!("Potential memory fragmentation detected");
    
    // Reset memory pool to defragment
    memory_pool = OptimizedMemoryPool::new(device_id, config);
}
```

### Performance Issues

#### Low GPU Utilization
```rust
// Monitor kernel performance
let perf_stats = kernel.performance_stats();

if perf_stats.total_kernel_launches > 100 {
    let avg_kernel_time = perf_stats.total_execution_time_ms / perf_stats.total_kernel_launches as f64;
    
    if avg_kernel_time < 0.1 {
        log::warn!("Kernel execution time very short: {:.3}ms - consider larger batch sizes", avg_kernel_time);
    }
    
    let transfer_ratio = (perf_stats.bytes_transferred_h2d + perf_stats.bytes_transferred_d2h) as f64 
                        / (1024.0 * 1024.0) / perf_stats.total_execution_time_ms;
    
    if transfer_ratio > 1000.0 {
        log::warn!("High memory transfer rate: {:.1} GB/s - may be memory bound", transfer_ratio);
    }
}
```

### Debugging Tools

#### CUDA Error Handling
```rust
use bitnet_common::KernelError;

fn handle_cuda_error(error: &KernelError) {
    match error {
        KernelError::GpuError { reason } => {
            eprintln!("CUDA Error: {}", reason);
            
            // Try to get more specific error information
            if reason.contains("out of memory") {
                eprintln!("Suggestion: Reduce batch size or enable memory optimization");
                eprintln!("Current GPU memory usage:");
                // Print memory stats
            } else if reason.contains("invalid configuration") {
                eprintln!("Suggestion: Check kernel launch parameters");
            } else if reason.contains("context") {
                eprintln!("Suggestion: Check CUDA installation and device permissions");
            }
        },
        _ => {
            eprintln!("Non-CUDA error: {:?}", error);
        }
    }
}
```

#### Memory Debugging
```rust
// Enable detailed memory tracking for debugging
#[cfg(debug_assertions)]
fn debug_memory_usage(pool: &OptimizedMemoryPool) {
    let stats = pool.stats();
    
    println!("=== Memory Debug Info ===");
    println!("Current usage: {} bytes", stats.current_usage);
    println!("Peak usage: {} bytes", stats.peak_usage);
    println!("Allocations: {} total, {} current", 
             stats.allocation_count, 
             stats.allocation_count - stats.deallocation_count);
    println!("Cache efficiency: {:.1}% hit rate",
             stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64 * 100.0);
    
    // Check for leaks
    let leaks = pool.check_leaks();
    if !leaks.is_empty() {
        println!("Potential leaks:");
        for leak in leaks {
            println!("  {}", leak);
        }
    }
    println!("========================");
}
```

## Configuration Examples

### Development Configuration
```rust
// Optimized for development (more debugging, less aggressive optimization)
let dev_config = MemoryPoolConfig {
    max_pool_size: 2 * 1024 * 1024 * 1024, // 2GB - conservative
    max_cached_buffers: 100,                // Smaller cache
    enable_memory_tracking: true,           // Full tracking
    cleanup_interval: Duration::from_secs(10), // Frequent cleanup
};
```

### Production Configuration
```rust
// Optimized for production (maximum performance)
let prod_config = MemoryPoolConfig {
    max_pool_size: 7 * 1024 * 1024 * 1024, // 7GB - aggressive
    max_cached_buffers: 2000,               // Large cache
    enable_memory_tracking: false,          // Minimal tracking
    cleanup_interval: Duration::from_secs(60), // Less frequent cleanup
};
```

### Testing Configuration
```rust
// Optimized for testing (deterministic, leak detection)
let test_config = MemoryPoolConfig {
    max_pool_size: 1 * 1024 * 1024 * 1024, // 1GB - small
    max_cached_buffers: 10,                 // Minimal cache
    enable_memory_tracking: true,           // Full tracking
    cleanup_interval: Duration::from_secs(1), // Very frequent
};
```

This comprehensive guide provides the foundation for optimal CUDA configuration and memory management in BitNet.rs GPU kernels.