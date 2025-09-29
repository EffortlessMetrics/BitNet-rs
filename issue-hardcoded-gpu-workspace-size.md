# [GPU] Hardcoded GPU memory values in QuantizedLinear workspace size calculation

## Problem Description

The `QuantizedLinear::calculate_gpu_workspace_size` function uses hardcoded GPU memory assumptions (6GB target) and arbitrary batch size limits, leading to suboptimal performance on different GPU configurations and potential out-of-memory errors or underutilization of available resources.

## Environment

- **File**: `crates/bitnet-inference/src/layers/quantized_linear.rs`
- **Function**: `QuantizedLinear::calculate_gpu_workspace_size`
- **Component**: GPU quantized matrix multiplication kernels
- **Affected Features**: GPU inference with I2S, TL1, TL2 quantization
- **MSRV**: Rust 1.90.0
- **Build Config**: `--features gpu` (CUDA path)

## Root Cause Analysis

The current implementation makes several problematic assumptions:

```rust
fn calculate_gpu_workspace_size(&self) -> Result<usize> {
    // Conservative batch size estimate based on available GPU memory
    let max_batch_size = match self.device {
        Device::Cuda(_) => {
            // Estimate based on 6GB GPU memory target - HARDCODED
            let available_memory: usize = 6 * 1024 * 1024 * 1024; // 6GB HARDCODED
            let model_memory = base_weight_size * dequant_multiplier;
            let remaining = available_memory.saturating_sub(model_memory);
            (remaining / (self.out_features * intermediate_multiplier)).min(128) // ARBITRARY LIMIT
        }
        _ => 64, // Conservative default - HARDCODED
    };
    // ... rest of calculation
}
```

**Critical Issues:**
1. **Hardcoded Memory Assumption**: 6GB assumption doesn't match actual GPU capabilities
2. **No Dynamic Detection**: Fails to query actual available GPU memory
3. **Arbitrary Batch Limits**: 128 token limit regardless of GPU capability
4. **Suboptimal Resource Usage**: Underutilizes high-memory GPUs, overestimates low-memory GPUs
5. **No Device Specificity**: Same calculation regardless of GPU model/generation

## Impact Assessment

**Severity**: High - Affects GPU performance and reliability
**Scope**: All GPU-accelerated quantized linear operations

**Performance Impact**:
- **High-end GPUs (24GB+)**: Severe underutilization, reduced throughput
- **Mid-range GPUs (8-16GB)**: Modest underutilization
- **Low-end GPUs (4GB)**: Potential OOM errors, crashes
- **Cloud/Datacenter GPUs**: Inefficient resource allocation, increased costs

**Affected Operations**:
- Quantized matrix multiplication for all quantization types
- Batch processing performance
- Memory allocation efficiency
- GPU kernel launch optimization

## Proposed Solution

### Primary Approach: Dynamic GPU Memory Detection

Implement runtime GPU memory detection with intelligent workspace sizing:

```rust
fn calculate_gpu_workspace_size(&self) -> Result<usize> {
    let base_weight_size = self.in_features * self.out_features;

    let (dequant_multiplier, intermediate_multiplier) = match self.qtype {
        QuantizationType::I2S => (2, 4), // FP16 + FP32
        QuantizationType::TL1 => (2, 4), // FP16 + FP32
        QuantizationType::TL2 => (4, 4), // FP32 + FP32 (larger tables)
    };

    let max_batch_size = match self.device {
        Device::Cuda(device_id) => {
            // Query actual GPU memory and calculate optimal batch size
            let gpu_info = self.query_gpu_memory_info(device_id)?;
            self.calculate_optimal_batch_size(
                &gpu_info,
                base_weight_size,
                dequant_multiplier,
                intermediate_multiplier,
            )?
        }
        _ => 64, // Fallback for non-CUDA devices
    };

    let dequant_size = base_weight_size * dequant_multiplier;
    let intermediate_size = max_batch_size * self.out_features * intermediate_multiplier;
    let total_size = dequant_size + intermediate_size;

    // Clamp to maximum workspace size with device-aware limits
    let max_workspace = self.get_device_max_workspace_size()?;
    let workspace_size = total_size.min(max_workspace);

    log::debug!(
        "GPU workspace: {}MB (batch: {}, available: {}MB, qtype: {:?})",
        workspace_size / (1024 * 1024),
        max_batch_size,
        gpu_info.available_mb,
        self.qtype
    );

    Ok(workspace_size)
}
```

### GPU Memory Information System

```rust
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub total_mb: usize,
    pub available_mb: usize,
    pub used_mb: usize,
    pub reserved_mb: usize,
    pub device_name: String,
    pub compute_capability: (u32, u32),
    pub memory_bandwidth_gb_s: f32,
}

impl QuantizedLinear {
    fn query_gpu_memory_info(&self, device_id: usize) -> Result<GpuMemoryInfo> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::{CudaDevice, MemoryInfo};

            let device = CudaDevice::new(device_id)?;
            let memory_info = device.memory_info()?;

            // Get device properties
            let device_name = device.name()?;
            let (major, minor) = device.compute_capability()?;

            // Calculate available memory with safety margin
            let total_mb = memory_info.total / (1024 * 1024);
            let free_mb = memory_info.free / (1024 * 1024);

            // Reserve 10% for CUDA overhead and other operations
            let safety_margin = total_mb / 10;
            let available_mb = free_mb.saturating_sub(safety_margin);

            Ok(GpuMemoryInfo {
                total_mb,
                available_mb,
                used_mb: total_mb - free_mb,
                reserved_mb: safety_margin,
                device_name,
                compute_capability: (major, minor),
                memory_bandwidth_gb_s: self.estimate_memory_bandwidth(major, minor)?,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(anyhow::anyhow!("CUDA support not compiled"))
        }
    }

    fn calculate_optimal_batch_size(
        &self,
        gpu_info: &GpuMemoryInfo,
        base_weight_size: usize,
        dequant_multiplier: usize,
        intermediate_multiplier: usize,
    ) -> Result<usize> {
        // Calculate memory requirements for model components
        let model_memory_mb = (base_weight_size * dequant_multiplier) / (1024 * 1024);

        // Account for already loaded model memory
        let available_for_workspace = gpu_info.available_mb.saturating_sub(model_memory_mb);

        // Calculate optimal batch size based on available memory
        let bytes_per_token = self.out_features * intermediate_multiplier;
        let max_tokens_by_memory = (available_for_workspace * 1024 * 1024) / bytes_per_token;

        // Apply device-specific constraints
        let max_batch_by_capability = self.get_max_batch_for_compute_capability(
            gpu_info.compute_capability
        );

        // Apply quantization-specific constraints
        let max_batch_by_quantization = self.get_max_batch_for_quantization_type();

        // Take the minimum of all constraints
        let optimal_batch = [
            max_tokens_by_memory,
            max_batch_by_capability,
            max_batch_by_quantization,
            1024, // Absolute maximum for stability
        ].iter().min().copied().unwrap_or(64);

        // Ensure minimum viable batch size
        Ok(optimal_batch.max(1))
    }

    fn get_max_batch_for_compute_capability(&self, (major, minor): (u32, u32)) -> usize {
        match major {
            8 | 9 => 1024, // Modern GPUs (A100, H100, RTX 40xx)
            7 => 512,      // Volta/Turing (V100, RTX 20xx/30xx)
            6 => 256,      // Pascal (GTX 10xx)
            _ => 128,      // Older architectures
        }
    }

    fn get_max_batch_for_quantization_type(&self) -> usize {
        match self.qtype {
            QuantizationType::I2S => 512,  // Efficient 2-bit quantization
            QuantizationType::TL1 => 256,  // Table lookup overhead
            QuantizationType::TL2 => 128,  // Larger table overhead
        }
    }

    fn get_device_max_workspace_size(&self) -> Result<usize> {
        match self.device {
            Device::Cuda(device_id) => {
                let gpu_info = self.query_gpu_memory_info(device_id)?;
                // Use up to 80% of available memory for workspace
                Ok((gpu_info.available_mb * 1024 * 1024 * 80) / 100)
            }
            _ => Ok(MAX_WORKSPACE_SIZE), // Fallback constant
        }
    }

    fn estimate_memory_bandwidth(&self, major: u32, minor: u32) -> Result<f32> {
        // Rough estimates based on compute capability
        let bandwidth_gb_s = match major {
            9 => 3000.0, // H100
            8 => match minor {
                0 => 1555.0, // A100 40GB
                6 => 2039.0, // A100 80GB
                _ => 1500.0,
            },
            7 => match minor {
                5 => 900.0,  // RTX 30xx
                0 => 900.0,  // V100
                _ => 800.0,
            },
            6 => 484.0, // GTX 10xx
            _ => 300.0, // Older GPUs
        };

        Ok(bandwidth_gb_s)
    }
}
```

### Adaptive Workspace Management

```rust
impl QuantizedLinear {
    /// Dynamic workspace size adjustment based on runtime conditions
    pub fn adjust_workspace_size(&mut self, actual_usage: WorkspaceUsageStats) -> Result<()> {
        // Monitor actual memory usage and adjust future allocations
        if actual_usage.peak_usage_ratio < 0.5 {
            // Underutilizing workspace, can increase batch size
            self.target_batch_size = (self.target_batch_size * 1.2) as usize;
        } else if actual_usage.peak_usage_ratio > 0.9 {
            // Close to OOM, reduce batch size
            self.target_batch_size = (self.target_batch_size * 0.8) as usize;
        }

        // Recalculate workspace size with new target
        self.workspace_size = self.calculate_gpu_workspace_size()?;
        log::info!(
            "Adjusted workspace: batch_size={}, workspace={}MB",
            self.target_batch_size,
            self.workspace_size / (1024 * 1024)
        );

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct WorkspaceUsageStats {
    pub peak_usage_ratio: f32,
    pub average_usage_ratio: f32,
    pub allocation_failures: u32,
    pub performance_score: f32,
}
```

## Implementation Plan

### Phase 1: GPU Memory Detection (1.5 days)
- [ ] Implement CUDA memory information querying
- [ ] Add device capability detection
- [ ] Create GPU memory info data structures
- [ ] Add error handling for unsupported devices

### Phase 2: Dynamic Batch Size Calculation (1.5 days)
- [ ] Implement optimal batch size calculation
- [ ] Add compute capability constraints
- [ ] Integrate quantization-specific limits
- [ ] Add memory safety margins and validation

### Phase 3: Adaptive Workspace Management (1 day)
- [ ] Implement runtime workspace adjustment
- [ ] Add memory usage monitoring
- [ ] Create feedback loop for optimization
- [ ] Add performance metrics collection

### Phase 4: Testing and Validation (1 day)
- [ ] Unit tests for memory detection
- [ ] Integration tests with various GPU models
- [ ] Performance benchmarking across different configurations
- [ ] Memory usage validation and OOM prevention testing

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_memory_detection() {
        let device = Device::new_cuda(0).unwrap();
        let linear = QuantizedLinear::new(1024, 1024, QuantizationType::I2S, device).unwrap();

        let gpu_info = linear.query_gpu_memory_info(0).unwrap();

        assert!(gpu_info.total_mb > 0);
        assert!(gpu_info.available_mb <= gpu_info.total_mb);
        assert!(!gpu_info.device_name.is_empty());
        assert!(gpu_info.compute_capability.0 >= 6); // Minimum supported
    }

    #[test]
    fn test_batch_size_calculation() {
        let gpu_info = GpuMemoryInfo {
            total_mb: 24000,  // RTX 4090
            available_mb: 20000,
            used_mb: 4000,
            reserved_mb: 2400,
            device_name: "RTX 4090".to_string(),
            compute_capability: (8, 9),
            memory_bandwidth_gb_s: 1008.0,
        };

        let linear = create_test_quantized_linear();
        let batch_size = linear.calculate_optimal_batch_size(
            &gpu_info, 1024 * 1024, 2, 4
        ).unwrap();

        // Should utilize high-memory GPU effectively
        assert!(batch_size >= 256);
        assert!(batch_size <= 1024);
    }

    #[test]
    fn test_low_memory_gpu_handling() {
        let gpu_info = GpuMemoryInfo {
            total_mb: 4000,   // GTX 1650
            available_mb: 2500,
            used_mb: 1500,
            reserved_mb: 400,
            device_name: "GTX 1650".to_string(),
            compute_capability: (7, 5),
            memory_bandwidth_gb_s: 192.0,
        };

        let linear = create_test_quantized_linear();
        let batch_size = linear.calculate_optimal_batch_size(
            &gpu_info, 1024 * 1024, 2, 4
        ).unwrap();

        // Should be conservative for low-memory GPU
        assert!(batch_size >= 1);
        assert!(batch_size <= 128);
    }
}
```

### Integration Tests
```rust
#[test]
#[cfg(feature = "cuda")]
fn test_dynamic_workspace_across_gpus() {
    let gpu_configs = vec![
        (0, "Primary GPU"),
        // Add more GPUs if available
    ];

    for (device_id, description) in gpu_configs {
        if !cuda::device_exists(device_id) {
            continue;
        }

        let device = Device::new_cuda(device_id).unwrap();
        let linear = QuantizedLinear::new(2048, 2048, QuantizationType::I2S, device).unwrap();

        let workspace_size = linear.calculate_gpu_workspace_size().unwrap();

        // Verify workspace size is reasonable for the device
        assert!(workspace_size > 1024 * 1024); // At least 1MB
        assert!(workspace_size < 20 * 1024 * 1024 * 1024); // Less than 20GB

        println!("{}: workspace = {}MB", description, workspace_size / (1024 * 1024));
    }
}
```

### Performance Tests
```rust
#[test]
#[cfg(feature = "cuda")]
fn test_workspace_performance_scaling() {
    let device = Device::new_cuda(0).unwrap();
    let linear = QuantizedLinear::new(4096, 4096, QuantizationType::I2S, device).unwrap();

    // Test performance across different batch sizes
    let batch_sizes = vec![1, 8, 16, 32, 64, 128, 256];
    let mut performance_results = Vec::new();

    for batch_size in batch_sizes {
        let input = create_test_input(batch_size, 4096);

        let start = std::time::Instant::now();
        let _output = linear.forward(&input).unwrap();
        let duration = start.elapsed();

        let throughput = batch_size as f64 / duration.as_secs_f64();
        performance_results.push((batch_size, throughput));
    }

    // Verify performance scaling makes sense
    let max_throughput = performance_results.iter()
        .map(|(_, throughput)| *throughput)
        .fold(0.0f64, f64::max);

    let optimal_batch = performance_results.iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(batch, _)| *batch)
        .unwrap();

    println!("Optimal batch size: {}, Max throughput: {:.2} samples/sec",
             optimal_batch, max_throughput);

    // The calculated optimal batch should be reasonably close to measured optimum
    let calculated_batch = linear.target_batch_size;
    assert!(
        (calculated_batch as f64 - optimal_batch as f64).abs() < optimal_batch as f64 * 0.5,
        "Calculated batch size {} should be within 50% of optimal {}",
        calculated_batch, optimal_batch
    );
}
```

## Performance Expectations

### Memory Utilization Improvements

| GPU Model | Memory | Current Batch | Optimal Batch | Improvement |
|-----------|--------|---------------|---------------|-------------|
| RTX 4090  | 24GB   | 128          | 512-1024      | 4-8x        |
| RTX 3080  | 10GB   | 128          | 256-512       | 2-4x        |
| RTX 3060  | 12GB   | 128          | 256-512       | 2-4x        |
| GTX 1660  | 6GB    | 128          | 64-128        | 0.5-1x      |
| Tesla V100| 32GB   | 128          | 1024+         | 8x+         |

### Throughput Improvements

- **High-end GPUs**: 3-8x throughput improvement
- **Mid-range GPUs**: 2-4x throughput improvement
- **Low-end GPUs**: Stable performance, OOM prevention
- **Cloud instances**: Better cost efficiency through optimal utilization

## Acceptance Criteria

### Functional Requirements
- [ ] Dynamic GPU memory detection works on all supported CUDA devices
- [ ] Batch size calculation adapts to available memory
- [ ] Workspace size scales appropriately with GPU capabilities
- [ ] No hardcoded memory assumptions remain

### Performance Requirements
- [ ] High-memory GPUs achieve >75% memory utilization
- [ ] Throughput improves by at least 2x on GPUs with >8GB memory
- [ ] Low-memory GPUs maintain stability without OOM errors
- [ ] Memory detection overhead <10ms

### Reliability Requirements
- [ ] Graceful fallback when GPU memory detection fails
- [ ] Automatic adjustment when memory conditions change
- [ ] Comprehensive error handling for edge cases
- [ ] Memory safety margins prevent OOM conditions

### Quality Requirements
- [ ] Comprehensive test coverage across different GPU models
- [ ] Performance benchmarks validate improvements
- [ ] No regressions in existing functionality
- [ ] Clear logging of memory allocation decisions

## Alternative Approaches

### Approach 1: Conservative Fixed Scaling
- Use percentage-based scaling from detected total memory
- Simpler implementation but less optimal
- Good fallback option

### Approach 2: Profile-Based Optimization
- Run benchmark workloads to determine optimal settings
- Store profiles for different GPU models
- Higher accuracy but more complex setup

### Approach 3: Gradual Adaptive Scaling
- Start with conservative estimates
- Gradually increase batch size based on success
- Self-tuning approach with runtime learning

## Related Issues

- GPU memory management optimization (#TBD)
- CUDA kernel optimization (#TBD)
- Multi-GPU support improvements (#TBD)
- Quantization performance optimization (#TBD)

## Labels

`gpu-optimization`, `memory-management`, `performance`, `cuda`, `high-priority`

## Definition of Done

- [ ] Hardcoded values replaced with dynamic detection
- [ ] GPU memory detection implemented and tested
- [ ] Batch size calculation optimized for different GPU configurations
- [ ] Performance improvements validated across multiple GPU models
- [ ] Comprehensive testing with various hardware configurations
- [ ] Documentation updated with GPU optimization guidance