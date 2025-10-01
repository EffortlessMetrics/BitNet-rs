# [HARDCODED] Replace Hardcoded GPU Workspace Size with Dynamic Memory Calculation

## Problem Description

The `QuantizedLinear::calculate_gpu_workspace_size` function in `crates/bitnet-inference/src/layers/quantized_linear.rs` contains hardcoded GPU memory assumptions, specifically a fixed 6GB memory target for workspace size calculations. This approach fails to adapt to different GPU configurations and can lead to suboptimal memory utilization or out-of-memory errors on GPUs with different memory capacities.

## Environment

- **File**: `crates/bitnet-inference/src/layers/quantized_linear.rs`
- **Function**: `QuantizedLinear::calculate_gpu_workspace_size`
- **Crate**: `bitnet-inference`
- **Feature Flags**: Requires `gpu` feature flag
- **Affected Devices**: All CUDA-capable GPUs

## Current Implementation Issues

```rust
fn calculate_gpu_workspace_size(&self) -> Result<usize> {
    let max_batch_size = match self.device {
        Device::Cuda(_) => {
            // Estimate based on 6GB GPU memory target - HARDCODED!
            let available_memory: usize = 6 * 1024 * 1024 * 1024; // 6GB
            let model_memory = base_weight_size * dequant_multiplier;
            let remaining = available_memory.saturating_sub(model_memory);
            (remaining / (self.out_features * intermediate_multiplier)).min(128)
        }
        _ => 64, // Conservative default
    };
}
```

## Root Cause Analysis

### Hardcoded Memory Assumptions
1. **Fixed 6GB Target**: Assumes all GPUs have 6GB+ memory
2. **No Runtime Querying**: Doesn't check actual available GPU memory
3. **Static Configuration**: Cannot adapt to different GPU models or memory pressure
4. **Poor Error Handling**: May silently fail or perform poorly on low-memory GPUs

### Impact on Different GPU Classes
- **High-End GPUs (24GB+)**: Severely underutilized, missing performance opportunities
- **Mid-Range GPUs (8-12GB)**: Reasonable but not optimal allocation
- **Entry-Level GPUs (4-6GB)**: Risk of OOM errors and allocation failures
- **Mobile/Embedded GPUs (2-4GB)**: Complete failure or poor performance

## Impact Assessment

- **Severity**: Medium-High - Affects GPU utilization and compatibility
- **Affected Users**: All users with non-6GB GPU configurations
- **Performance Impact**: 20-50% performance loss due to suboptimal memory usage
- **Compatibility Impact**: Failures on low-memory and high-memory GPUs

## Proposed Solution

Implement dynamic GPU memory detection and adaptive workspace calculation:

### 1. GPU Memory Query Infrastructure
```rust
use cuda_runtime_sys::*;

#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub total_memory: usize,
    pub available_memory: usize,
    pub reserved_memory: usize,
    pub device_id: i32,
}

impl GpuMemoryManager {
    pub fn query_device_memory(device_id: i32) -> Result<GpuMemoryInfo> {
        unsafe {
            let mut free_mem: usize = 0;
            let mut total_mem: usize = 0;

            let result = cudaMemGetInfo(&mut free_mem, &mut total_mem);
            if result != cudaError_t::cudaSuccess {
                return Err(BitNetError::GpuMemoryQuery {
                    device_id,
                    error: cuda_error_to_string(result),
                });
            }

            // Reserve memory for system and other processes
            let reserved = (total_mem as f64 * 0.1) as usize; // 10% reserve
            let available = free_mem.saturating_sub(reserved);

            Ok(GpuMemoryInfo {
                total_memory: total_mem,
                available_memory: available,
                reserved_memory: reserved,
                device_id,
            })
        }
    }
}
```

### 2. Adaptive Workspace Calculation
```rust
impl QuantizedLinear {
    fn calculate_gpu_workspace_size(&self) -> Result<usize> {
        let base_weight_size = self.in_features * self.out_features;

        let (dequant_multiplier, intermediate_multiplier) = match self.qtype {
            QuantizationType::I2S => (2, 4), // FP16 + FP32
            QuantizationType::TL1 => (2, 4), // FP16 + FP32
            QuantizationType::TL2 => (4, 4), // FP32 + FP32 (larger tables)
        };

        let max_batch_size = match &self.device {
            Device::Cuda(device_id) => {
                let memory_info = GpuMemoryManager::query_device_memory(*device_id)?;
                self.calculate_optimal_batch_size(&memory_info, base_weight_size,
                                                 dequant_multiplier, intermediate_multiplier)?
            }
            _ => 64, // Conservative default for non-CUDA devices
        };

        let dequant_size = base_weight_size * dequant_multiplier;
        let intermediate_size = max_batch_size * self.out_features * intermediate_multiplier;
        let total_size = dequant_size + intermediate_size;

        // Apply safety margins and maximum limits
        let workspace_size = self.apply_workspace_constraints(total_size, &memory_info)?;

        log::debug!(
            "GPU workspace size: {} MB (batch_size: {}, available: {} MB, qtype: {:?})",
            workspace_size / (1024 * 1024),
            max_batch_size,
            memory_info.available_memory / (1024 * 1024),
            self.qtype
        );

        Ok(workspace_size)
    }

    fn calculate_optimal_batch_size(
        &self,
        memory_info: &GpuMemoryInfo,
        base_weight_size: usize,
        dequant_multiplier: usize,
        intermediate_multiplier: usize,
    ) -> Result<usize> {
        // Calculate model memory footprint
        let model_memory = base_weight_size * dequant_multiplier;

        // Reserve memory for other GPU operations and fragmentation
        let usable_memory = (memory_info.available_memory as f64 * 0.8) as usize;
        let workspace_memory = usable_memory.saturating_sub(model_memory);

        if workspace_memory < MIN_WORKSPACE_SIZE {
            return Err(BitNetError::InsufficientGpuMemory {
                required: model_memory + MIN_WORKSPACE_SIZE,
                available: memory_info.available_memory,
                device_id: memory_info.device_id,
            });
        }

        // Calculate maximum batch size based on available memory
        let per_sample_memory = self.out_features * intermediate_multiplier;
        let max_batch_from_memory = workspace_memory / per_sample_memory;

        // Apply reasonable limits based on GPU class
        let max_batch_size = self.apply_gpu_class_limits(
            max_batch_from_memory,
            memory_info.total_memory,
        );

        Ok(max_batch_size.max(1).min(MAX_BATCH_SIZE))
    }

    fn apply_gpu_class_limits(&self, calculated_batch: usize, total_memory: usize) -> usize {
        match total_memory {
            // High-end GPUs (24GB+): Allow large batches
            mem if mem >= 24 * 1024 * 1024 * 1024 => calculated_batch.min(512),
            // Mid-range GPUs (8-24GB): Moderate batches
            mem if mem >= 8 * 1024 * 1024 * 1024 => calculated_batch.min(256),
            // Entry-level GPUs (4-8GB): Conservative batches
            mem if mem >= 4 * 1024 * 1024 * 1024 => calculated_batch.min(128),
            // Low-memory GPUs (<4GB): Very conservative
            _ => calculated_batch.min(64),
        }
    }
}
```

### 3. Memory Pressure Handling
```rust
impl QuantizedLinear {
    fn apply_workspace_constraints(
        &self,
        calculated_size: usize,
        memory_info: &GpuMemoryInfo,
    ) -> Result<usize> {
        // Apply absolute maximum workspace size
        let size_with_max = calculated_size.min(MAX_WORKSPACE_SIZE);

        // Ensure workspace doesn't exceed available memory
        let memory_constrained = size_with_max.min(
            (memory_info.available_memory as f64 * 0.6) as usize
        );

        // Apply minimum workspace size for functionality
        if memory_constrained < MIN_WORKSPACE_SIZE {
            return Err(BitNetError::WorkspaceAllocationFailed {
                requested: calculated_size,
                available: memory_info.available_memory,
                minimum_required: MIN_WORKSPACE_SIZE,
            });
        }

        Ok(memory_constrained)
    }

    pub fn handle_memory_pressure(&mut self) -> Result<()> {
        // Reduce batch size under memory pressure
        let current_memory = GpuMemoryManager::query_device_memory(self.device_id())?;

        if current_memory.available_memory < self.workspace_size {
            log::warn!("GPU memory pressure detected, reducing workspace size");

            let new_workspace = self.calculate_gpu_workspace_size()?;
            if new_workspace < self.workspace_size {
                self.workspace_size = new_workspace;
                self.reallocate_workspace()?;
            }
        }

        Ok(())
    }
}
```

## Implementation Plan

### Phase 1: GPU Memory Query Infrastructure
- [ ] Add CUDA memory querying functions
- [ ] Implement `GpuMemoryInfo` structure
- [ ] Add error handling for memory query failures
- [ ] Create memory information caching system

### Phase 2: Dynamic Workspace Calculation
- [ ] Replace hardcoded 6GB with dynamic calculation
- [ ] Implement memory-based batch size calculation
- [ ] Add GPU class-specific optimizations
- [ ] Create safety margin and constraint systems

### Phase 3: Memory Pressure Handling
- [ ] Add runtime memory monitoring
- [ ] Implement adaptive workspace resizing
- [ ] Create memory pressure detection system
- [ ] Add graceful degradation mechanisms

### Phase 4: Multi-GPU Support
- [ ] Extend to support multiple GPU devices
- [ ] Add per-device memory management
- [ ] Implement cross-GPU memory balancing
- [ ] Create unified memory allocation interface

### Phase 5: Optimization and Testing
- [ ] Add performance benchmarks across GPU classes
- [ ] Create memory utilization optimization algorithms
- [ ] Add comprehensive error handling and recovery
- [ ] Implement memory debugging and profiling tools

## Testing Strategy

### Memory Configuration Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "gpu")]
    fn test_dynamic_workspace_calculation() {
        let layer = create_test_quantized_linear();
        let workspace_size = layer.calculate_gpu_workspace_size().unwrap();

        // Verify workspace size is reasonable for detected GPU
        let memory_info = GpuMemoryManager::query_device_memory(0).unwrap();
        assert!(workspace_size <= memory_info.available_memory);
        assert!(workspace_size >= MIN_WORKSPACE_SIZE);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_memory_pressure_handling() {
        let mut layer = create_test_quantized_linear();

        // Simulate memory pressure
        simulate_memory_pressure();

        layer.handle_memory_pressure().unwrap();

        // Verify workspace was reduced appropriately
        assert!(layer.workspace_size <= calculate_expected_reduced_size());
    }

    #[test]
    fn test_gpu_class_optimizations() {
        // Test different GPU memory configurations
        for &total_memory in &[4_GB, 8_GB, 16_GB, 24_GB, 48_GB] {
            let mock_info = create_mock_memory_info(total_memory);
            let batch_size = calculate_batch_size_for_memory(&mock_info);

            // Verify batch size is appropriate for GPU class
            verify_batch_size_for_gpu_class(batch_size, total_memory);
        }
    }
}
```

## BitNet.rs Integration Notes

### CUDA Feature Flag Integration
```rust
#[cfg(feature = "gpu")]
mod gpu_memory {
    // GPU memory management implementation
}

#[cfg(not(feature = "gpu"))]
mod gpu_memory {
    // Stub implementation for non-GPU builds
}
```

### Device Management Integration
- Integrate with existing `DeviceManager` for device selection
- Maintain compatibility with device enumeration and selection
- Support fallback to CPU when GPU memory is insufficient

### Performance Monitoring
- Add memory utilization metrics to performance monitoring
- Track workspace allocation efficiency
- Monitor memory pressure and allocation failures

## Dependencies

```toml
[dependencies]
cuda-runtime-sys = { version = "0.3", optional = true }

[features]
gpu = ["cuda-runtime-sys"]
```

## Acceptance Criteria

- [ ] Dynamic GPU memory detection replacing hardcoded 6GB assumption
- [ ] Adaptive workspace calculation based on actual available memory
- [ ] Support for GPU memory ranging from 2GB to 80GB+
- [ ] Memory pressure detection and graceful handling
- [ ] GPU class-specific optimization strategies
- [ ] Comprehensive error handling for memory allocation failures
- [ ] Performance benchmarks showing optimal memory utilization
- [ ] Multi-GPU support with per-device memory management
- [ ] Integration with existing device management and error systems
- [ ] Full test coverage including edge cases and OOM scenarios

## Related Issues

- GPU memory manager implementation
- Multi-GPU support and device selection
- Performance monitoring and optimization
- Error handling standardization

## Priority

**Medium-High** - Directly affects GPU utilization efficiency and compatibility across different hardware configurations. Critical for production deployment on diverse GPU infrastructure.