# [Hardcoded Values] QuantizedLinear::calculate_gpu_workspace_size uses fixed 6GB memory assumption

## Problem Description

The `QuantizedLinear::calculate_gpu_workspace_size` function in `crates/bitnet-inference/src/layers/quantized_linear.rs` uses a hardcoded 6GB GPU memory assumption for workspace size calculations. This causes suboptimal memory utilization on GPUs with different memory capacities and can lead to OOM errors or underutilization of available resources.

## Environment

- **File**: `crates/bitnet-inference/src/layers/quantized_linear.rs`
- **Function**: `QuantizedLinear::calculate_gpu_workspace_size`
- **Feature Flags**: `gpu`, `cuda`
- **Crate**: `bitnet-inference`
- **Dependencies**: CUDA runtime, GPU memory management

## Current Implementation Analysis

The function assumes a fixed 6GB GPU memory target:

```rust
fn calculate_gpu_workspace_size(&self) -> Result<usize> {
    // GPU kernels need temporary storage for different quantization types
    let base_weight_size = self.in_features * self.out_features;

    let (dequant_multiplier, intermediate_multiplier) = match self.qtype {
        QuantizationType::I2S => (2, 4), // FP16 + FP32
        QuantizationType::TL1 => (2, 4), // FP16 + FP32
        QuantizationType::TL2 => (4, 4), // FP32 + FP32 (larger tables)
    };

    // Conservative batch size estimate based on available GPU memory
    let max_batch_size = match self.device {
        Device::Cuda(_) => {
            // Estimate based on 6GB GPU memory target
            let available_memory: usize = 6 * 1024 * 1024 * 1024; // 6GB HARDCODED
            let model_memory = base_weight_size * dequant_multiplier;
            let remaining = available_memory.saturating_sub(model_memory);
            (remaining / (self.out_features * intermediate_multiplier)).min(128)
        }
        _ => 64, // Conservative default
    };

    let dequant_size = base_weight_size * dequant_multiplier;
    let intermediate_size = max_batch_size * self.out_features * intermediate_multiplier;
    let total_size = dequant_size + intermediate_size;

    // Clamp to maximum workspace size to prevent OOM
    let workspace_size = total_size.min(MAX_WORKSPACE_SIZE);

    log::debug!(
        "GPU workspace size: {} MB (batch_size: {}, qtype: {:?})",
        workspace_size / (1024 * 1024),
        max_batch_size,
        self.qtype
    );

    Ok(workspace_size)
}
```

## Root Cause Analysis

1. **Fixed Memory Assumption**: Hardcoded 6GB doesn't reflect actual GPU memory availability
2. **No Runtime Detection**: Doesn't query actual GPU memory using CUDA APIs
3. **Suboptimal Batch Sizing**: Batch size calculations based on incorrect memory assumptions
4. **Poor Resource Utilization**: Underutilizes high-memory GPUs, may fail on low-memory GPUs
5. **No Multi-GPU Awareness**: Doesn't consider different GPUs may have different memory capacities

## Impact Assessment

**Severity**: High - Performance & Reliability Critical
**Affected Components**:
- GPU memory utilization efficiency
- Batch size optimization for inference
- OOM error prevention
- Multi-GPU deployment scenarios
- Performance scaling with GPU memory

**Production Impact**:
- **RTX 4090 (24GB)**: Massive underutilization, only 25% memory usage
- **RTX 3060 (8GB)**: Slightly conservative but acceptable
- **RTX 3050 (4GB)**: Potential OOM errors with 6GB assumption
- **A100 (40GB/80GB)**: Severe underutilization of expensive hardware
- **Consumer GPUs**: Inconsistent behavior across different memory sizes

## Proposed Solution

### Primary Implementation: Dynamic GPU Memory Detection

Replace hardcoded values with runtime GPU memory detection:

```rust
use bitnet_kernels::gpu::cuda::{get_device_memory_info, CudaDevice};

fn calculate_gpu_workspace_size(&self) -> Result<usize> {
    let base_weight_size = self.in_features * self.out_features;

    let (dequant_multiplier, intermediate_multiplier) = match self.qtype {
        QuantizationType::I2S => (2, 4), // FP16 + FP32
        QuantizationType::TL1 => (2, 4), // FP16 + FP32
        QuantizationType::TL2 => (4, 4), // FP32 + FP32 (larger tables)
    };

    let max_batch_size = match self.device {
        Device::Cuda(device_id) => {
            self.calculate_optimal_batch_size_cuda(
                device_id,
                base_weight_size,
                dequant_multiplier,
                intermediate_multiplier,
            )?
        }
        _ => 64, // Conservative default for non-CUDA devices
    };

    let dequant_size = base_weight_size * dequant_multiplier;
    let intermediate_size = max_batch_size * self.out_features * intermediate_multiplier;
    let total_size = dequant_size + intermediate_size;

    let workspace_size = total_size.min(MAX_WORKSPACE_SIZE);

    log::debug!(
        "GPU workspace size: {} MB (batch_size: {}, qtype: {:?}, device: {:?})",
        workspace_size / (1024 * 1024),
        max_batch_size,
        self.qtype,
        self.device
    );

    Ok(workspace_size)
}

fn calculate_optimal_batch_size_cuda(
    &self,
    device_id: usize,
    base_weight_size: usize,
    dequant_multiplier: usize,
    intermediate_multiplier: usize,
) -> Result<usize> {
    // Query actual GPU memory
    let memory_info = get_device_memory_info(device_id)?;
    let total_memory = memory_info.total;
    let free_memory = memory_info.free;

    log::debug!(
        "GPU {} memory: {:.1} GB total, {:.1} GB free",
        device_id,
        total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        free_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Conservative memory usage: use 80% of free memory
    let usable_memory = (free_memory as f64 * 0.8) as usize;

    // Reserve memory for model weights and overhead
    let model_memory = base_weight_size * dequant_multiplier;
    let overhead_memory = total_memory / 20; // 5% overhead

    let available_for_workspace = usable_memory
        .saturating_sub(model_memory)
        .saturating_sub(overhead_memory);

    // Calculate maximum batch size based on available memory
    let max_batch_size = if available_for_workspace > 0 {
        (available_for_workspace / (self.out_features * intermediate_multiplier))
            .min(256) // Cap at reasonable maximum
            .max(1)   // Ensure at least batch size 1
    } else {
        1 // Fallback to minimal batch size
    };

    log::debug!(
        "Calculated max batch size: {} (available workspace: {} MB)",
        max_batch_size,
        available_for_workspace / (1024 * 1024)
    );

    Ok(max_batch_size)
}
```

## Implementation Plan

### Phase 1: GPU Memory Detection Infrastructure
- [ ] Implement CUDA memory query functions in `bitnet-kernels`
- [ ] Add GPU memory information structures and error handling
- [ ] Create memory detection utilities for different GPU types
- [ ] Add logging and debugging support for memory information

### Phase 2: Dynamic Workspace Calculation
- [ ] Replace hardcoded values with dynamic memory detection
- [ ] Implement optimal batch size calculation algorithms
- [ ] Add safety margins and conservative memory usage
- [ ] Handle edge cases (low memory, fragmented memory)

### Phase 3: Configuration & Fallbacks
- [ ] Add configuration options for memory limits
- [ ] Implement environment variable overrides
- [ ] Create graceful fallbacks for detection failures
- [ ] Add validation for memory limit configurations

### Phase 4: Testing & Optimization
- [ ] Test on various GPU models and memory sizes
- [ ] Validate memory usage accuracy and efficiency
- [ ] Performance benchmark workspace size impact
- [ ] Add comprehensive error handling and recovery

## Testing Strategy

### Memory Detection Testing
```rust
#[test]
#[cfg(feature = "cuda")]
fn test_gpu_memory_detection() {
    let device_id = 0;
    let memory_info = get_device_memory_info(device_id).unwrap();

    assert!(memory_info.total > 0);
    assert!(memory_info.free <= memory_info.total);
    println!("GPU memory: {} GB total", memory_info.total / (1024 * 1024 * 1024));
}

#[test]
fn test_workspace_calculation_different_gpus() {
    let test_cases = vec![
        (4 * 1024 * 1024 * 1024, "RTX 3050 4GB"),  // 4GB
        (8 * 1024 * 1024 * 1024, "RTX 3060 8GB"),  // 8GB
        (24 * 1024 * 1024 * 1024, "RTX 4090 24GB"), // 24GB
    ];

    for (memory_size, gpu_name) in test_cases {
        let workspace_size = calculate_workspace_for_memory(memory_size);
        println!("{}: {} MB workspace", gpu_name, workspace_size / (1024 * 1024));
        assert!(workspace_size > 0);
    }
}
```

### Performance Testing
```rust
#[test]
fn test_memory_utilization_efficiency() {
    let quantized_linear = create_test_quantized_linear();
    let workspace_size = quantized_linear.calculate_gpu_workspace_size().unwrap();

    // Measure actual memory usage during inference
    let actual_usage = benchmark_memory_usage(&quantized_linear);

    // Efficiency should be >70% (not too conservative)
    let efficiency = actual_usage as f64 / workspace_size as f64;
    assert!(efficiency > 0.7, "Memory efficiency too low: {:.2}%", efficiency * 100.0);
}
```

## Related Issues/PRs

- GPU memory management and allocation optimization
- CUDA kernel workspace requirements
- Multi-GPU inference support
- Performance benchmarking across different GPU configurations

## Acceptance Criteria

- [ ] Dynamic GPU memory detection replaces hardcoded 6GB assumption
- [ ] Optimal batch size calculation based on actual available memory
- [ ] Support for GPUs ranging from 4GB to 80GB+ memory
- [ ] Graceful fallbacks when memory detection fails
- [ ] Configurable memory limits via environment variables or config
- [ ] Memory utilization efficiency >70% on typical GPU configurations
- [ ] No OOM errors on supported GPU memory sizes
- [ ] Comprehensive logging of memory usage and batch size decisions
- [ ] Performance impact <5% overhead for memory detection

## Notes

This fix is crucial for optimal GPU utilization across the wide range of GPU memory configurations in production deployments. The current 6GB assumption severely limits performance on high-end GPUs while potentially causing failures on lower-end hardware.

Priority should be given to robust error handling and conservative memory usage to prevent OOM errors, while maximizing utilization of available GPU resources for better inference performance.
