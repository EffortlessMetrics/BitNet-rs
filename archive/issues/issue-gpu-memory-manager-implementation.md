# [GPU] Implement production-ready GPU memory manager with real CUDA memory queries

## Problem Description

The `GpuMemoryManager::new` function in `crates/bitnet-inference/src/gpu.rs` currently uses a hardcoded placeholder value (8GB) for GPU memory instead of querying the actual GPU memory. This prevents optimal memory allocation, can cause out-of-memory errors on smaller GPUs, and wastes resources on larger GPUs.

## Environment
- **File**: `crates/bitnet-inference/src/gpu.rs`
- **Function**: `GpuMemoryManager::new`
- **Hardware**: NVIDIA GPUs (RTX series, Tesla, A100, H100)
- **CUDA Toolkit**: 11.8+ or 12.x
- **Feature Flags**: `--no-default-features --features gpu`
- **Dependencies**: CUDA runtime, cuBLAS, cuDNN

## Reproduction Steps

1. Build BitNet-rs with GPU support:
   ```bash
   cargo build --no-default-features --features gpu
   ```

2. Run inference on GPU with memory monitoring:
   ```bash
   nvidia-smi -l 1 &  # Monitor GPU memory usage
   cargo run -p xtask -- infer --model model.gguf --backend gpu
   ```

3. Observe memory allocation behavior

**Expected Results**:
- Memory manager should detect actual GPU memory (e.g., 24GB on RTX 4090)
- Memory allocation should be optimized for available memory
- Should work correctly across different GPU models

**Actual Results**:
- Always assumes 8GB regardless of actual GPU memory
- May cause OOM on smaller GPUs or waste memory on larger ones
- No device-specific memory optimization

## Root Cause Analysis

### Current Implementation Limitations

```rust
impl GpuMemoryManager {
    /// Create a new GPU memory manager
    pub fn new(device_id: usize, enable_pooling: bool) -> Result<Self> {
        // Query GPU memory (placeholder implementation)
        let total_memory = 8 * 1024 * 1024 * 1024; // 8GB placeholder

        Ok(Self {
            device_id,
            total_memory,
            allocated_memory: 0,
            memory_pools: Vec::new(),
            enable_memory_pooling: enable_pooling,
        })
    }
```

### Issues Identified

1. **No Real GPU Query**: Hardcoded 8GB regardless of actual hardware
2. **Missing CUDA Integration**: No use of CUDA memory management APIs
3. **No Error Handling**: Cannot detect GPU memory query failures
4. **No Device Validation**: Doesn't verify device_id exists
5. **Missing Memory Fragmentation Handling**: No consideration for available vs total memory

### CUDA Memory Management Requirements

```c
// Required CUDA operations for proper implementation:
size_t free_memory, total_memory;
cudaMemGetInfo(&free_memory, &total_memory);  // Get actual memory info
cudaSetDevice(device_id);                     // Set active device
cudaDeviceGetProperties(&props, device_id);   // Get device capabilities
```

## Impact Assessment

- **Severity**: High (production readiness blocker)
- **Performance Impact**:
  - Cannot utilize full GPU memory capacity on high-end hardware
  - May cause unexpected OOM failures on lower-end GPUs
  - Inefficient memory allocation strategies

- **Compatibility Impact**:
  - Prevents deployment on diverse GPU hardware configurations
  - Cannot adapt to different memory sizes (4GB to 80GB+ range)
  - Breaks multi-GPU scenarios with heterogeneous memory

- **User Experience**:
  - Unpredictable memory behavior
  - Suboptimal performance on all hardware
  - Potential inference failures

## Proposed Solution

Implement comprehensive GPU memory management with real CUDA queries, device validation, memory optimization, and multi-GPU support.

### Technical Implementation

#### 1. CUDA Memory Query Integration

```rust
use cudarc::driver::{CudaDevice, DriverError};
use cudarc::runtime::{CudaRuntime, CudaRuntimeError};

impl GpuMemoryManager {
    /// Create a new GPU memory manager with real memory detection
    pub fn new(device_id: usize, enable_pooling: bool) -> Result<Self> {
        // Validate device exists
        let device_count = Self::get_device_count()?;
        if device_id >= device_count {
            return Err(BitNetError::InvalidDevice(format!(
                "Device {} not available. Found {} devices", device_id, device_count
            )));
        }

        // Query actual GPU memory
        let memory_info = Self::query_gpu_memory(device_id)?;
        let device_properties = Self::query_device_properties(device_id)?;

        // Calculate usable memory (reserve some for system/driver)
        let reserved_memory = Self::calculate_reserved_memory(&memory_info, &device_properties);
        let usable_memory = memory_info.total.saturating_sub(reserved_memory);

        log::info!(
            "GPU {} initialized: {:.2} GB total, {:.2} GB usable, {:.2} GB free",
            device_id,
            memory_info.total as f64 / 1e9,
            usable_memory as f64 / 1e9,
            memory_info.free as f64 / 1e9
        );

        Ok(Self {
            device_id,
            total_memory: memory_info.total,
            usable_memory,
            allocated_memory: 0,
            memory_pools: Self::initialize_memory_pools(&device_properties, enable_pooling)?,
            enable_memory_pooling: enable_pooling,
            device_properties,
            cuda_context: Self::create_cuda_context(device_id)?,
        })
    }

    fn query_gpu_memory(device_id: usize) -> Result<GpuMemoryInfo> {
        // Set device context
        unsafe {
            cudarc::driver::sys::cuDeviceGet(&mut device_id as *mut _ as _, device_id as i32)?;
            cudarc::driver::sys::cuCtxSetCurrent(cuda_context)?;
        }

        let mut free_bytes = 0;
        let mut total_bytes = 0;

        // Query memory information
        unsafe {
            let result = cudarc::driver::sys::cuMemGetInfo_v2(&mut free_bytes, &mut total_bytes);
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(BitNetError::CudaError(format!(
                    "Failed to query GPU memory: {:?}", result
                )));
            }
        }

        Ok(GpuMemoryInfo {
            total: total_bytes,
            free: free_bytes,
            used: total_bytes - free_bytes,
        })
    }

    fn query_device_properties(device_id: usize) -> Result<GpuDeviceProperties> {
        let mut props = std::mem::MaybeUninit::uninit();

        unsafe {
            let result = cudarc::driver::sys::cuDeviceGetProperties(
                props.as_mut_ptr(),
                device_id as i32
            );

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(BitNetError::CudaError(format!(
                    "Failed to query device properties: {:?}", result
                )));
            }

            let props = props.assume_init();
            Ok(GpuDeviceProperties::from_cuda_props(props))
        }
    }

    fn calculate_reserved_memory(
        memory_info: &GpuMemoryInfo,
        properties: &GpuDeviceProperties
    ) -> usize {
        // Reserve memory for:
        // 1. CUDA driver overhead (~200-500MB)
        // 2. GPU context and streams (~100MB)
        // 3. Emergency buffer for stability (~10% of total)

        let driver_overhead = 500 * 1024 * 1024; // 500MB
        let context_overhead = 100 * 1024 * 1024; // 100MB
        let emergency_buffer = memory_info.total / 10; // 10%

        let total_reserved = driver_overhead + context_overhead + emergency_buffer;

        // Cap at reasonable maximum (2GB on any system)
        total_reserved.min(2 * 1024 * 1024 * 1024)
    }
}

#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub total: usize,
    pub free: usize,
    pub used: usize,
}

#[derive(Debug, Clone)]
pub struct GpuDeviceProperties {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub max_threads_per_block: i32,
    pub max_threads_per_multiprocessor: i32,
    pub multiprocessor_count: i32,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
    pub total_constant_memory: usize,
    pub shared_memory_per_block: usize,
    pub unified_addressing: bool,
}
```

#### 2. Memory Pool Management

```rust
impl GpuMemoryManager {
    fn initialize_memory_pools(
        properties: &GpuDeviceProperties,
        enable_pooling: bool
    ) -> Result<Vec<MemoryPool>> {
        if !enable_pooling {
            return Ok(Vec::new());
        }

        let mut pools = Vec::new();

        // Small allocations pool (1KB - 1MB)
        pools.push(MemoryPool::new(
            "small",
            1024,           // min_block_size
            1024 * 1024,    // max_block_size
            64,             // initial_blocks
        )?);

        // Medium allocations pool (1MB - 100MB)
        pools.push(MemoryPool::new(
            "medium",
            1024 * 1024,       // min_block_size
            100 * 1024 * 1024, // max_block_size
            16,                // initial_blocks
        )?);

        // Large allocations pool (100MB+)
        pools.push(MemoryPool::new(
            "large",
            100 * 1024 * 1024, // min_block_size
            usize::MAX,        // max_block_size
            4,                 // initial_blocks
        )?);

        log::info!("Initialized {} memory pools for GPU {}", pools.len(), properties.name);
        Ok(pools)
    }

    pub fn allocate(&mut self, size: usize, alignment: usize) -> Result<GpuMemoryPtr> {
        // Check if allocation would exceed usable memory
        if self.allocated_memory + size > self.usable_memory {
            return Err(BitNetError::OutOfMemory(format!(
                "Cannot allocate {} bytes: would exceed usable memory ({} + {} > {})",
                size, self.allocated_memory, size, self.usable_memory
            )));
        }

        let ptr = if self.enable_memory_pooling {
            self.allocate_from_pool(size, alignment)?
        } else {
            self.allocate_direct(size, alignment)?
        };

        self.allocated_memory += size;
        log::debug!("Allocated {} bytes on GPU {}, total: {} bytes",
                   size, self.device_id, self.allocated_memory);

        Ok(ptr)
    }

    fn allocate_from_pool(&mut self, size: usize, alignment: usize) -> Result<GpuMemoryPtr> {
        // Find appropriate pool
        for pool in &mut self.memory_pools {
            if pool.can_allocate(size) {
                if let Ok(ptr) = pool.allocate(size, alignment) {
                    return Ok(ptr);
                }
            }
        }

        // Fallback to direct allocation if pools can't satisfy
        log::debug!("Pool allocation failed for {} bytes, using direct allocation", size);
        self.allocate_direct(size, alignment)
    }

    fn allocate_direct(&mut self, size: usize, alignment: usize) -> Result<GpuMemoryPtr> {
        let mut ptr = std::ptr::null_mut();

        // Ensure alignment
        let aligned_size = (size + alignment - 1) & !(alignment - 1);

        unsafe {
            let result = cudarc::driver::sys::cuMemAlloc_v2(&mut ptr as *mut _ as _, aligned_size);
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(BitNetError::CudaError(format!(
                    "Failed to allocate {} bytes: {:?}", aligned_size, result
                )));
            }
        }

        Ok(GpuMemoryPtr {
            ptr: ptr as *mut u8,
            size: aligned_size,
            device_id: self.device_id,
        })
    }

    pub fn deallocate(&mut self, ptr: GpuMemoryPtr) -> Result<()> {
        if self.enable_memory_pooling {
            // Try to return to pool first
            for pool in &mut self.memory_pools {
                if pool.owns_pointer(&ptr) {
                    pool.deallocate(ptr)?;
                    self.allocated_memory = self.allocated_memory.saturating_sub(ptr.size);
                    return Ok(());
                }
            }
        }

        // Direct deallocation
        unsafe {
            let result = cudarc::driver::sys::cuMemFree_v2(ptr.ptr as _);
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                log::warn!("Failed to deallocate GPU memory: {:?}", result);
            }
        }

        self.allocated_memory = self.allocated_memory.saturating_sub(ptr.size);
        log::debug!("Deallocated {} bytes on GPU {}, remaining: {} bytes",
                   ptr.size, self.device_id, self.allocated_memory);

        Ok(())
    }
}
```

#### 3. Multi-GPU Support

```rust
pub struct MultiGpuMemoryManager {
    managers: Vec<GpuMemoryManager>,
    device_affinities: HashMap<String, usize>, // Tensor name -> preferred device
    load_balancer: LoadBalancer,
}

impl MultiGpuMemoryManager {
    pub fn new(device_ids: &[usize], enable_pooling: bool) -> Result<Self> {
        let mut managers = Vec::new();

        for &device_id in device_ids {
            let manager = GpuMemoryManager::new(device_id, enable_pooling)?;
            managers.push(manager);
        }

        Ok(Self {
            managers,
            device_affinities: HashMap::new(),
            load_balancer: LoadBalancer::new(device_ids)?,
        })
    }

    pub fn allocate_on_best_device(&mut self, size: usize, tensor_name: &str) -> Result<(usize, GpuMemoryPtr)> {
        // Check for device affinity
        if let Some(&preferred_device) = self.device_affinities.get(tensor_name) {
            if let Ok(ptr) = self.managers[preferred_device].allocate(size, 256) {
                return Ok((preferred_device, ptr));
            }
        }

        // Use load balancer to find best device
        let device_id = self.load_balancer.select_device_for_allocation(size)?;
        let ptr = self.managers[device_id].allocate(size, 256)?;

        // Remember this affinity for future allocations
        self.device_affinities.insert(tensor_name.to_string(), device_id);

        Ok((device_id, ptr))
    }

    pub fn get_memory_stats(&self) -> Vec<GpuMemoryStats> {
        self.managers.iter().enumerate().map(|(device_id, manager)| {
            GpuMemoryStats {
                device_id,
                total_memory: manager.total_memory,
                allocated_memory: manager.allocated_memory,
                utilization: manager.allocated_memory as f64 / manager.usable_memory as f64,
                pool_stats: manager.get_pool_stats(),
            }
        }).collect()
    }
}
```

#### 4. Memory Monitoring and Diagnostics

```rust
impl GpuMemoryManager {
    pub fn get_detailed_stats(&self) -> GpuMemoryStats {
        let current_memory = Self::query_gpu_memory(self.device_id)
            .unwrap_or_else(|_| GpuMemoryInfo {
                total: self.total_memory,
                free: self.total_memory.saturating_sub(self.allocated_memory),
                used: self.allocated_memory,
            });

        GpuMemoryStats {
            device_id: self.device_id,
            total_memory: self.total_memory,
            allocated_memory: self.allocated_memory,
            actual_free_memory: current_memory.free,
            utilization: self.allocated_memory as f64 / self.usable_memory as f64,
            pool_stats: self.get_pool_stats(),
            fragmentation_ratio: self.calculate_fragmentation(),
        }
    }

    fn calculate_fragmentation(&self) -> f64 {
        if !self.enable_memory_pooling {
            return 0.0; // No fragmentation tracking for direct allocation
        }

        let total_pool_memory: usize = self.memory_pools.iter()
            .map(|pool| pool.total_allocated_memory())
            .sum();

        let total_pool_used: usize = self.memory_pools.iter()
            .map(|pool| pool.used_memory())
            .sum();

        if total_pool_memory == 0 {
            return 0.0;
        }

        1.0 - (total_pool_used as f64 / total_pool_memory as f64)
    }

    pub fn log_memory_usage(&self) {
        let stats = self.get_detailed_stats();

        log::info!(
            "GPU {} Memory Usage: {:.2} GB / {:.2} GB ({:.1}% utilization)",
            self.device_id,
            stats.allocated_memory as f64 / 1e9,
            stats.total_memory as f64 / 1e9,
            stats.utilization * 100.0
        );

        if stats.fragmentation_ratio > 0.3 {
            log::warn!(
                "GPU {} has high memory fragmentation: {:.1}%",
                self.device_id,
                stats.fragmentation_ratio * 100.0
            );
        }
    }
}
```

## Implementation Plan

### Phase 1: Core CUDA Integration (Week 1-2)
- [ ] Integrate cudarc for CUDA memory operations
- [ ] Implement real GPU memory querying
- [ ] Add device validation and property detection
- [ ] Create basic memory allocation/deallocation

### Phase 2: Memory Pool Implementation (Week 3)
- [ ] Design and implement memory pool architecture
- [ ] Add pool-based allocation strategies
- [ ] Implement fragmentation monitoring
- [ ] Add pool statistics and diagnostics

### Phase 3: Multi-GPU Support (Week 4)
- [ ] Implement multi-GPU memory manager
- [ ] Add device affinity and load balancing
- [ ] Create cross-device memory transfer utilities
- [ ] Add comprehensive monitoring across devices

### Phase 4: Performance & Optimization (Week 5)
- [ ] Optimize allocation patterns for BitNet workloads
- [ ] Add memory prefetching for model loading
- [ ] Implement dynamic memory scaling
- [ ] Performance testing and benchmarking

## Testing Strategy

### Unit Tests
```rust
#[cfg(feature = "gpu")]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_memory_detection() {
        let manager = GpuMemoryManager::new(0, false).unwrap();
        assert!(manager.total_memory > 0);
        assert!(manager.usable_memory <= manager.total_memory);
    }

    #[test]
    fn test_memory_allocation_limits() {
        let mut manager = GpuMemoryManager::new(0, false).unwrap();

        // Should succeed for reasonable allocation
        let ptr = manager.allocate(1024 * 1024, 256).unwrap(); // 1MB

        // Should fail for excessive allocation
        let result = manager.allocate(manager.usable_memory + 1, 256);
        assert!(result.is_err());

        manager.deallocate(ptr).unwrap();
    }

    #[test]
    fn test_pool_allocation() {
        let mut manager = GpuMemoryManager::new(0, true).unwrap();

        let ptr1 = manager.allocate(1024, 256).unwrap();
        let ptr2 = manager.allocate(2048, 256).unwrap();

        manager.deallocate(ptr1).unwrap();
        manager.deallocate(ptr2).unwrap();

        // Verify memory was returned to pool
        let stats = manager.get_detailed_stats();
        assert!(stats.fragmentation_ratio < 0.1);
    }
}
```

### Integration Tests
```rust
#[cfg(feature = "gpu")]
mod integration_tests {
    #[test]
    fn test_realistic_model_loading() {
        let mut manager = GpuMemoryManager::new(0, true).unwrap();

        // Simulate loading a 7B parameter model
        let model_size = 7_000_000_000 * 2; // 2 bytes per parameter (FP16)
        let model_ptr = manager.allocate(model_size, 256);

        match model_ptr {
            Ok(ptr) => {
                let stats = manager.get_detailed_stats();
                println!("Successfully loaded 7B model, utilization: {:.1}%",
                         stats.utilization * 100.0);
                manager.deallocate(ptr).unwrap();
            },
            Err(e) => {
                println!("Cannot load 7B model on this GPU: {}", e);
                // This is expected on smaller GPUs
            }
        }
    }
}
```

## Performance Targets

- **Memory Query Time**: <10ms for device initialization
- **Allocation Time**: <1ms for typical allocations (<100MB)
- **Pool Efficiency**: >95% memory utilization with pooling enabled
- **Fragmentation**: <20% fragmentation under normal workloads
- **Multi-GPU Overhead**: <5% performance penalty vs single GPU

## Acceptance Criteria

- [ ] Real GPU memory detection on all supported NVIDIA GPUs
- [ ] Memory allocation respects actual hardware limits
- [ ] Proper error handling for OOM conditions
- [ ] Memory pools reduce allocation overhead by >50%
- [ ] Multi-GPU support enables memory scaling
- [ ] Comprehensive logging and monitoring
- [ ] All tests pass on test hardware
- [ ] Performance meets target benchmarks
- [ ] Integration with BitNet inference pipeline
- [ ] Cross-validation with CUDA memory profilers

## Dependencies

- cudarc or similar CUDA wrapper crate
- CUDA toolkit (11.8+ or 12.x)
- NVIDIA driver (compatible with CUDA version)
- Test hardware: Multiple NVIDIA GPU models

## Related Issues

- GPU backend implementation
- CUDA kernel optimization
- Multi-GPU inference support
- Memory leak detection and prevention
- Performance monitoring and profiling

## Labels
- `gpu`
- `cuda`
- `memory-management`
- `performance`
- `priority-high`
- `infrastructure`
