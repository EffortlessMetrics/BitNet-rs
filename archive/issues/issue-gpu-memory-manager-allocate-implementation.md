# [GPU] GPU Memory Manager Allocation Implementation

## Problem Description

The `GpuMemoryManager::allocate_from_pool` and `allocate_direct` functions are simplified stubs that only update counters without performing actual GPU memory allocation, preventing real GPU memory management.

## Environment

- **File**: `crates/bitnet-inference/src/gpu.rs`
- **Functions**: `GpuMemoryManager::allocate_from_pool`, `GpuMemoryManager::allocate_direct`
- **Component**: GPU Memory Management System

## Root Cause Analysis

### **Current Implementation:**
```rust
fn allocate_from_pool(&mut self, size: usize) -> Result<usize> {
    // Simplified pool allocation
    self.allocated_memory += size;
    Ok(self.allocated_memory - size)
}

fn allocate_direct(&mut self, size: usize) -> Result<usize> {
    self.allocated_memory += size;
    Ok(self.allocated_memory - size)
}
```

### **Issues:**
1. **No Actual Allocation**: Only counter updates, no GPU memory operations
2. **Missing Pool Management**: No real memory pool implementation
3. **Invalid Return Values**: Returns memory offsets instead of GPU pointers

## Proposed Solution

Implement real CUDA memory allocation:

```rust
fn allocate_from_pool(&mut self, size: usize) -> Result<*mut u8> {
    // Check if pool has suitable block
    if let Some(ptr) = self.memory_pool.allocate(size) {
        self.allocated_memory += size;
        Ok(ptr)
    } else {
        // Expand pool or allocate directly
        self.expand_pool(size)?;
        self.memory_pool.allocate(size)
            .ok_or_else(|| anyhow::anyhow!("Pool allocation failed after expansion"))
    }
}

fn allocate_direct(&mut self, size: usize) -> Result<*mut u8> {
    unsafe {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = cuda_sys::cudaMalloc(&mut ptr, size);
        if result == cuda_sys::cudaError_t::cudaSuccess {
            self.allocated_memory += size;
            Ok(ptr as *mut u8)
        } else {
            Err(anyhow::anyhow!("CUDA allocation failed: {:?}", result))
        }
    }
}
```

## Implementation Plan

### **Week 1: Core Allocation**
- Integrate CUDA memory allocation APIs
- Implement memory pool management
- Add error handling and validation

### **Week 2: Pool Optimization**
- Add memory pool expansion logic
- Implement memory defragmentation
- Add allocation tracking and metrics

## Success Metrics

- [ ] Actual GPU memory allocation using CUDA APIs
- [ ] Efficient memory pool management with reuse
- [ ] Proper error handling for out-of-memory conditions
- [ ] Memory leak prevention with proper deallocation

## Labels

- `gpu-memory`
- `cuda-integration`
- `memory-management`
