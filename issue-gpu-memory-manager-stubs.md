# [GPU] Implement real GPU memory management replacing simplified allocation stubs

## Problem Description

The `GpuMemoryManager::allocate_from_pool` and `allocate_direct` functions in `crates/bitnet-inference/src/gpu.rs` are simplified stub implementations that only increment an `allocated_memory` counter instead of performing actual GPU memory operations. This prevents real GPU acceleration and can lead to memory management issues in production.

## Environment

- **File**: `crates/bitnet-inference/src/gpu.rs`
- **Functions**: `GpuMemoryManager::allocate_from_pool`, `GpuMemoryManager::allocate_direct`
- **Component**: GPU inference backend and memory management
- **Features**: `gpu`, CUDA acceleration
- **MSRV**: Rust 1.90.0

## Reproduction Steps

1. Examine the `GpuMemoryManager` implementation in gpu.rs
2. Note that allocation functions only update a counter
3. Observe no actual GPU memory allocation or device operations
4. Verify that GPU tensors cannot be properly allocated

**Expected**: Real GPU memory allocation with device-specific operations
**Actual**: Simple counter incrementation without actual memory management

## Root Cause Analysis

The current stub implementation masks the complexity of real GPU memory management:

**Current Simplified Implementation:**
```rust
impl GpuMemoryManager {
    fn allocate_from_pool(&mut self, size: usize) -> Result<usize> {
        // Simplified pool allocation - just incrementing counter
        self.allocated_memory += size;
        Ok(self.allocated_memory - size)
    }

    fn allocate_direct(&mut self, size: usize) -> Result<usize> {
        // No real GPU allocation - just counter increment
        self.allocated_memory += size;
        Ok(self.allocated_memory - size)
    }
}
```

**Technical Issues:**
1. **No Real Allocation**: Functions return fake memory addresses (counter values)
2. **Missing Device Context**: No CUDA device management or context setting
3. **No Memory Pools**: Pool allocation doesn't use actual memory pools
4. **No Error Handling**: Missing CUDA error checking and resource management
5. **Memory Leaks**: No deallocation or cleanup mechanisms
6. **Missing Synchronization**: No handling of concurrent memory operations

## Impact Assessment

**Severity**: High - Blocks GPU acceleration functionality entirely
**Type**: Core functionality implementation

**Affected Components**:
- GPU inference backend
- CUDA kernel execution
- Mixed precision operations
- Large model loading and inference
- Memory-intensive operations

**Current Limitations**:
- GPU backend cannot actually use GPU memory
- CUDA kernels cannot execute with real tensors
- Memory usage reporting is completely inaccurate
- No protection against GPU memory exhaustion
- Cannot leverage GPU-specific optimizations

## Proposed Solution

### Primary Solution: Real CUDA Memory Management

Implement comprehensive GPU memory management with CUDA integration:

```rust
use candle_core::{Device as CandleDevice, Tensor as CandleTensor, DType};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Real GPU memory manager with CUDA integration
pub struct GpuMemoryManager {
    device_id: usize,
    total_memory: usize,
    allocated_memory: usize,
    memory_pools: Vec<MemoryPool>,
    allocated_blocks: HashMap<usize, AllocatedBlock>,
    allocation_alignment: usize,
    max_pool_size: usize,
}

#[derive(Debug)]
struct MemoryPool {
    pool_id: usize,
    block_size: usize,
    total_blocks: usize,
    free_blocks: Vec<usize>,
    allocated_blocks: Vec<usize>,
    cuda_ptr: *mut u8,
}

#[derive(Debug, Clone)]
struct AllocatedBlock {
    cuda_ptr: *mut u8,
    size: usize,
    pool_id: Option<usize>,
    allocated_at: std::time::Instant,
}

impl GpuMemoryManager {
    /// Create a new GPU memory manager for specified device
    pub fn new(device_id: usize, total_pool_size: usize) -> Result<Self> {
        // Set CUDA device context
        #[cfg(feature = "gpu")]
        {
            unsafe {
                let result = cuInit(0);
                if result != 0 {
                    return Err(anyhow::anyhow!("Failed to initialize CUDA: error {}", result));
                }

                let result = cuDeviceGet(&mut device_id as *mut i32, device_id as i32);
                if result != 0 {
                    return Err(anyhow::anyhow!("Failed to get CUDA device {}: error {}", device_id, result));
                }

                let mut context: *mut std::ffi::c_void = std::ptr::null_mut();
                let result = cuCtxCreate(&mut context, 0, device_id as i32);
                if result != 0 {
                    return Err(anyhow::anyhow!("Failed to create CUDA context: error {}", result));
                }
            }
        }

        // Query device memory information
        let total_memory = Self::get_device_memory_info(device_id)?;
        info!("GPU {} total memory: {:.2} GB", device_id, total_memory as f64 / (1024.0 * 1024.0 * 1024.0));

        let mut manager = Self {
            device_id,
            total_memory,
            allocated_memory: 0,
            memory_pools: Vec::new(),
            allocated_blocks: HashMap::new(),
            allocation_alignment: 256, // 256-byte alignment for optimal performance
            max_pool_size: total_pool_size.min(total_memory / 4), // Use at most 25% of total memory
        };

        // Initialize memory pools for common allocation sizes
        manager.initialize_memory_pools()?;

        Ok(manager)
    }

    /// Get device memory information
    fn get_device_memory_info(device_id: usize) -> Result<usize> {
        #[cfg(feature = "gpu")]
        {
            let device = CandleDevice::new_cuda(device_id)?;
            // Use candle's device memory query
            match device {
                CandleDevice::Cuda(cuda_device) => {
                    // Query total device memory
                    Ok(8 * 1024 * 1024 * 1024) // Default to 8GB, should be queried from device
                }
                _ => Err(anyhow::anyhow!("Invalid CUDA device")),
            }
        }
        #[cfg(not(feature = "gpu"))]
        Err(anyhow::anyhow!("GPU feature not enabled"))
    }

    /// Initialize memory pools for different allocation sizes
    fn initialize_memory_pools(&mut self) -> Result<()> {
        // Create pools for common tensor sizes
        let pool_configs = vec![
            (1024,     1024 * 512),   // 512MB pool for small tensors (1KB blocks)
            (64 * 1024, 1024 * 256),  // 256MB pool for medium tensors (64KB blocks)
            (1024 * 1024, 1024 * 128), // 128MB pool for large tensors (1MB blocks)
        ];

        for (i, (block_size, pool_size)) in pool_configs.into_iter().enumerate() {
            if self.allocated_memory + pool_size <= self.max_pool_size {
                let pool = self.create_memory_pool(i, block_size, pool_size)?;
                self.memory_pools.push(pool);
                self.allocated_memory += pool_size;
                info!("Created memory pool {}: {} blocks of {} bytes", i, pool_size / block_size, block_size);
            }
        }

        Ok(())
    }

    /// Create a memory pool with CUDA allocation
    fn create_memory_pool(&self, pool_id: usize, block_size: usize, total_size: usize) -> Result<MemoryPool> {
        let total_blocks = total_size / block_size;

        #[cfg(feature = "gpu")]
        {
            // Allocate GPU memory for the pool
            let mut cuda_ptr: *mut u8 = std::ptr::null_mut();
            unsafe {
                let result = cuMemAlloc(&mut cuda_ptr as *mut *mut u8 as *mut *mut std::ffi::c_void, total_size);
                if result != 0 {
                    return Err(anyhow::anyhow!("Failed to allocate CUDA memory pool: error {}", result));
                }
            }

            Ok(MemoryPool {
                pool_id,
                block_size,
                total_blocks,
                free_blocks: (0..total_blocks).collect(),
                allocated_blocks: Vec::new(),
                cuda_ptr,
            })
        }
        #[cfg(not(feature = "gpu"))]
        Err(anyhow::anyhow!("GPU feature not enabled"))
    }

    /// Allocate memory from the most suitable pool
    pub fn allocate_from_pool(&mut self, size: usize) -> Result<*mut u8> {
        let aligned_size = self.align_size(size);

        // Find the best fitting pool
        let pool_index = self.memory_pools
            .iter()
            .position(|pool| pool.block_size >= aligned_size && !pool.free_blocks.is_empty())
            .ok_or_else(|| anyhow::anyhow!("No suitable memory pool found for size {}", size))?;

        let pool = &mut self.memory_pools[pool_index];

        // Allocate a block from the pool
        let block_index = pool.free_blocks.pop()
            .ok_or_else(|| anyhow::anyhow!("Memory pool {} exhausted", pool_index))?;

        pool.allocated_blocks.push(block_index);

        // Calculate the actual pointer
        let block_ptr = unsafe {
            pool.cuda_ptr.add(block_index * pool.block_size)
        };

        // Record the allocation
        let allocated_block = AllocatedBlock {
            cuda_ptr: block_ptr,
            size: aligned_size,
            pool_id: Some(pool_index),
            allocated_at: std::time::Instant::now(),
        };

        self.allocated_blocks.insert(block_ptr as usize, allocated_block);
        self.allocated_memory += pool.block_size;

        debug!("Allocated {} bytes from pool {} (block {})", aligned_size, pool_index, block_index);
        Ok(block_ptr)
    }

    /// Allocate memory directly from CUDA (for large allocations)
    pub fn allocate_direct(&mut self, size: usize) -> Result<*mut u8> {
        let aligned_size = self.align_size(size);

        #[cfg(feature = "gpu")]
        {
            // Direct CUDA allocation for large tensors
            let mut cuda_ptr: *mut u8 = std::ptr::null_mut();
            unsafe {
                let result = cuMemAlloc(&mut cuda_ptr as *mut *mut u8 as *mut *mut std::ffi::c_void, aligned_size);
                if result != 0 {
                    return Err(anyhow::anyhow!("Failed to allocate CUDA memory directly: error {}", result));
                }
            }

            // Record the allocation
            let allocated_block = AllocatedBlock {
                cuda_ptr,
                size: aligned_size,
                pool_id: None,
                allocated_at: std::time::Instant::now(),
            };

            self.allocated_blocks.insert(cuda_ptr as usize, allocated_block);
            self.allocated_memory += aligned_size;

            debug!("Direct allocated {} bytes at {:p}", aligned_size, cuda_ptr);
            Ok(cuda_ptr)
        }
        #[cfg(not(feature = "gpu"))]
        Err(anyhow::anyhow!("GPU feature not enabled"))
    }

    /// Deallocate memory block
    pub fn deallocate(&mut self, ptr: *mut u8) -> Result<()> {
        let ptr_addr = ptr as usize;

        let allocated_block = self.allocated_blocks.remove(&ptr_addr)
            .ok_or_else(|| anyhow::anyhow!("Attempted to deallocate unknown pointer {:p}", ptr))?;

        match allocated_block.pool_id {
            Some(pool_id) => {
                // Return block to pool
                let pool = &mut self.memory_pools[pool_id];
                let block_index = unsafe {
                    ptr.offset_from(pool.cuda_ptr) as usize / pool.block_size
                };

                pool.allocated_blocks.retain(|&b| b != block_index);
                pool.free_blocks.push(block_index);
                self.allocated_memory -= pool.block_size;

                debug!("Returned block {} to pool {}", block_index, pool_id);
            }
            None => {
                // Direct deallocation
                #[cfg(feature = "gpu")]
                {
                    unsafe {
                        let result = cuMemFree(ptr as *mut std::ffi::c_void);
                        if result != 0 {
                            warn!("Failed to free CUDA memory at {:p}: error {}", ptr, result);
                        }
                    }
                }
                self.allocated_memory -= allocated_block.size;

                debug!("Direct deallocated {} bytes at {:p}", allocated_block.size, ptr);
            }
        }

        Ok(())
    }

    /// Align size to optimal GPU memory alignment
    fn align_size(&self, size: usize) -> usize {
        (size + self.allocation_alignment - 1) & !(self.allocation_alignment - 1)
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> GpuMemoryStats {
        let pool_usage: Vec<PoolStats> = self.memory_pools.iter().map(|pool| {
            PoolStats {
                pool_id: pool.pool_id,
                block_size: pool.block_size,
                total_blocks: pool.total_blocks,
                allocated_blocks: pool.allocated_blocks.len(),
                free_blocks: pool.free_blocks.len(),
            }
        }).collect();

        GpuMemoryStats {
            device_id: self.device_id,
            total_memory: self.total_memory,
            allocated_memory: self.allocated_memory,
            free_memory: self.total_memory - self.allocated_memory,
            pool_usage,
            active_allocations: self.allocated_blocks.len(),
        }
    }

    /// Cleanup all allocated memory
    pub fn cleanup(&mut self) -> Result<()> {
        info!("Cleaning up GPU memory manager for device {}", self.device_id);

        // Deallocate all active blocks
        let ptr_addrs: Vec<usize> = self.allocated_blocks.keys().copied().collect();
        for ptr_addr in ptr_addrs {
            self.deallocate(ptr_addr as *mut u8)?;
        }

        // Free memory pools
        #[cfg(feature = "gpu")]
        {
            for pool in &self.memory_pools {
                unsafe {
                    let result = cuMemFree(pool.cuda_ptr as *mut std::ffi::c_void);
                    if result != 0 {
                        warn!("Failed to free memory pool {}: error {}", pool.pool_id, result);
                    }
                }
            }
        }

        self.memory_pools.clear();
        self.allocated_memory = 0;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GpuMemoryStats {
    pub device_id: usize,
    pub total_memory: usize,
    pub allocated_memory: usize,
    pub free_memory: usize,
    pub pool_usage: Vec<PoolStats>,
    pub active_allocations: usize,
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub pool_id: usize,
    pub block_size: usize,
    pub total_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
}

impl Drop for GpuMemoryManager {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            error!("Error during GPU memory manager cleanup: {}", e);
        }
    }
}

// CUDA FFI declarations (in a real implementation, use cuda-sys or similar)
#[cfg(feature = "gpu")]
extern "C" {
    fn cuInit(flags: u32) -> i32;
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
    fn cuCtxCreate(pctx: *mut *mut std::ffi::c_void, flags: u32, dev: i32) -> i32;
    fn cuMemAlloc(dptr: *mut *mut std::ffi::c_void, bytesize: usize) -> i32;
    fn cuMemFree(dptr: *mut std::ffi::c_void) -> i32;
}
```

### Integration with Tensor Operations

```rust
// Integration with BitNet tensor operations
impl GpuMemoryManager {
    /// Allocate GPU tensor with proper memory management
    pub fn allocate_tensor(&mut self, shape: &[usize], dtype: DType) -> Result<CandleTensor> {
        let total_elements: usize = shape.iter().product();
        let element_size = dtype.size_in_bytes();
        let total_size = total_elements * element_size;

        // Choose allocation strategy based on size
        let cuda_ptr = if total_size > 1024 * 1024 {
            // Large tensors use direct allocation
            self.allocate_direct(total_size)?
        } else {
            // Small tensors use pool allocation
            self.allocate_from_pool(total_size)?
        };

        // Create CUDA device
        let cuda_device = CandleDevice::new_cuda(self.device_id)?;

        // Create tensor from raw CUDA memory
        let tensor = unsafe {
            CandleTensor::from_raw_buffer(
                cuda_ptr as *const u8,
                dtype,
                shape,
                &cuda_device,
            )?
        };

        debug!("Allocated GPU tensor: shape {:?}, dtype {:?}, size {} bytes", shape, dtype, total_size);
        Ok(tensor)
    }

    /// Transfer CPU tensor to GPU
    pub fn transfer_to_gpu(&mut self, cpu_tensor: &CandleTensor) -> Result<CandleTensor> {
        let cuda_device = CandleDevice::new_cuda(self.device_id)?;
        let gpu_tensor = cpu_tensor.to_device(&cuda_device)?;

        // Track the allocation
        let shape = gpu_tensor.shape();
        let total_size = shape.elem_count() * gpu_tensor.dtype().size_in_bytes();
        self.allocated_memory += total_size;

        debug!("Transferred tensor to GPU: shape {:?}, size {} bytes", shape.dims(), total_size);
        Ok(gpu_tensor)
    }
}
```

## Implementation Plan

### Phase 1: CUDA Integration Foundation (2 days)
- [ ] Set up CUDA FFI bindings and device context management
- [ ] Implement device memory querying and initialization
- [ ] Add CUDA error handling and resource cleanup
- [ ] Create basic memory allocation and deallocation functions

### Phase 2: Memory Pool System (2 days)
- [ ] Design and implement memory pool architecture
- [ ] Add pool creation, allocation, and deallocation logic
- [ ] Implement size-based allocation strategy (pool vs direct)
- [ ] Add memory alignment and optimization

### Phase 3: Tensor Integration (1 day)
- [ ] Integrate memory manager with Candle tensor operations
- [ ] Implement GPU tensor allocation and transfer functions
- [ ] Add tensor memory tracking and lifecycle management
- [ ] Test tensor operations with real GPU memory

### Phase 4: Advanced Features (1 day)
- [ ] Add memory usage statistics and monitoring
- [ ] Implement memory leak detection and prevention
- [ ] Add concurrent allocation support
- [ ] Create performance profiling hooks

## Testing Strategy

### Unit Tests
```rust
#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_memory_manager_creation() {
        if !cuda_available() {
            return;
        }

        let manager = GpuMemoryManager::new(0, 512 * 1024 * 1024).unwrap();
        assert_eq!(manager.device_id, 0);
        assert!(manager.total_memory > 0);
        assert!(!manager.memory_pools.is_empty());
    }

    #[test]
    fn test_pool_allocation() {
        if !cuda_available() {
            return;
        }

        let mut manager = GpuMemoryManager::new(0, 512 * 1024 * 1024).unwrap();

        // Allocate small tensor that should use pool
        let ptr = manager.allocate_from_pool(1024).unwrap();
        assert!(!ptr.is_null());

        // Verify allocation tracking
        assert!(manager.allocated_blocks.contains_key(&(ptr as usize)));

        // Clean up
        manager.deallocate(ptr).unwrap();
        assert!(!manager.allocated_blocks.contains_key(&(ptr as usize)));
    }

    #[test]
    fn test_direct_allocation() {
        if !cuda_available() {
            return;
        }

        let mut manager = GpuMemoryManager::new(0, 512 * 1024 * 1024).unwrap();

        // Allocate large tensor that should use direct allocation
        let ptr = manager.allocate_direct(10 * 1024 * 1024).unwrap();
        assert!(!ptr.is_null());

        // Verify it's tracked as direct allocation
        let block = manager.allocated_blocks.get(&(ptr as usize)).unwrap();
        assert!(block.pool_id.is_none());

        manager.deallocate(ptr).unwrap();
    }

    #[test]
    fn test_tensor_allocation() {
        if !cuda_available() {
            return;
        }

        let mut manager = GpuMemoryManager::new(0, 512 * 1024 * 1024).unwrap();

        let tensor = manager.allocate_tensor(&[128, 256], DType::F32).unwrap();
        assert_eq!(tensor.shape().dims(), &[128, 256]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert!(matches!(tensor.device(), CandleDevice::Cuda(_)));
    }

    fn cuda_available() -> bool {
        CandleDevice::new_cuda(0).is_ok()
    }
}
```

### Integration Tests
```rust
#[tokio::test]
async fn test_gpu_backend_with_real_memory() {
    if !GpuBackend::is_available() {
        return;
    }

    let mut memory_manager = GpuMemoryManager::new(0, 1024 * 1024 * 1024).unwrap();

    // Create tensors for a small model
    let input_tensor = memory_manager.allocate_tensor(&[1, 512], DType::F32).unwrap();
    let weight_tensor = memory_manager.allocate_tensor(&[512, 256], DType::F32).unwrap();

    // Perform basic tensor operations
    let output = input_tensor.matmul(&weight_tensor).unwrap();
    assert_eq!(output.shape().dims(), &[1, 256]);

    // Check memory usage
    let stats = memory_manager.memory_stats();
    assert!(stats.allocated_memory > 0);
    assert!(stats.active_allocations > 0);
}
```

### Performance Tests
```rust
#[bench]
fn bench_gpu_allocation_performance(b: &mut Bencher) {
    if !cuda_available() {
        return;
    }

    let mut manager = GpuMemoryManager::new(0, 1024 * 1024 * 1024).unwrap();

    b.iter(|| {
        let ptr = manager.allocate_from_pool(4096).unwrap();
        manager.deallocate(ptr).unwrap();
    });
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Real CUDA memory allocation and deallocation
- [ ] Memory pool management for efficient small allocations
- [ ] Direct allocation for large tensors
- [ ] Proper tensor integration with GPU memory
- [ ] Device context management and error handling

### Performance Requirements
- [ ] Pool allocation faster than direct allocation for small tensors
- [ ] Memory fragmentation minimized through pool strategy
- [ ] Proper memory alignment for optimal GPU access
- [ ] Memory usage tracking accurate within 1%

### Quality Requirements
- [ ] No memory leaks in long-running operations
- [ ] Graceful handling of out-of-memory conditions
- [ ] Proper cleanup on manager destruction
- [ ] Thread-safe memory operations

## Related Issues/PRs

- GPU acceleration infrastructure (#TBD)
- CUDA kernel integration (#TBD)
- Memory management optimization (#TBD)
- Mixed precision GPU operations (#TBD)

## Labels

`gpu`, `memory-management`, `cuda`, `high-priority`, `infrastructure`, `stub-removal`

## Definition of Done

- [ ] Real CUDA memory allocation replaces stub implementations
- [ ] Memory pool system functional for different allocation sizes
- [ ] GPU tensor operations work with real device memory
- [ ] Memory usage statistics and monitoring available
- [ ] All GPU tests pass with real memory management
- [ ] No memory leaks detected in stress testing
- [ ] Performance benchmarks meet expectations for GPU operations
