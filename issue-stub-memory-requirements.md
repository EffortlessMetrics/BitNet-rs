# [Stub Implementation] ProductionLoader::get_memory_requirements uses hardcoded values instead of model analysis

## Problem Description

The `get_memory_requirements` function in `crates/bitnet-models/src/production_loader.rs` contains a simplified stub implementation that uses hardcoded memory values instead of analyzing the actual model file to calculate precise memory requirements. This impacts memory planning, device selection, and resource allocation in production environments.

## Environment

- **File**: `crates/bitnet-models/src/production_loader.rs`
- **Function**: `get_memory_requirements`
- **Crate**: `bitnet-models`
- **Related Components**: Model loading, device allocation, memory management

## Current Implementation Analysis

The stub implementation uses fixed hardcoded values:

```rust
pub fn get_memory_requirements(&self, device: &str) -> MemoryRequirements {
    // This is a simplified implementation
    // In reality, this would analyze the model file and calculate precise memory needs

    let base_memory = 1000; // Base memory in MB

    match device {
        "cpu" => MemoryRequirements {
            total_mb: base_memory,
            gpu_memory_mb: None,
            cpu_memory_mb: base_memory - 200,
            kv_cache_mb: 100,
            activation_mb: 50,
            headroom_mb: 50,
        },
        "gpu" => MemoryRequirements {
            total_mb: base_memory,
            gpu_memory_mb: Some(800),
            cpu_memory_mb: 200,
            kv_cache_mb: 100,
            activation_mb: 50,
            headroom_mb: 50,
        },
        _ => MemoryRequirements {
            total_mb: base_memory,
            gpu_memory_mb: None,
            cpu_memory_mb: base_memory,
            kv_cache_mb: 0,
            activation_mb: 0,
            headroom_mb: 0,
        },
    }
}
```

## Root Cause Analysis

1. **No Model Analysis**: Function doesn't examine actual model parameters, layers, or quantization
2. **Fixed Memory Assumptions**: 1GB base memory doesn't scale with model size or architecture
3. **Inaccurate Device Planning**: Hardcoded values don't reflect real GPU/CPU memory usage patterns
4. **Missing Quantization Awareness**: Doesn't account for I2S, TL1, TL2 quantization memory differences
5. **No Configuration Integration**: Ignores batch size, sequence length, and other runtime parameters

## Impact Assessment

**Severity**: High - Production Critical
**Affected Components**:
- Memory allocation and planning
- Device selection logic
- OOM prevention mechanisms
- Production deployment sizing
- Resource utilization optimization

**Production Impact**:
- Incorrect memory planning leading to OOM errors
- Suboptimal device selection (GPU vs CPU)
- Over/under-provisioning of resources
- Poor user experience with memory-related failures
- Inability to predict scaling requirements

## Proposed Solution

### Primary Implementation: Model-Aware Memory Calculation

Replace hardcoded values with dynamic analysis based on model structure:

```rust
pub fn get_memory_requirements(&self, device: &str) -> MemoryRequirements {
    let model_config = self.base_loader.get_model_config();
    let model_metadata = self.base_loader.get_model_metadata();

    // Calculate base model memory requirements
    let model_memory = self.calculate_model_memory(&model_config, &model_metadata);

    // Calculate KV cache requirements based on architecture
    let kv_cache_memory = self.calculate_kv_cache_memory(&model_config);

    // Calculate activation memory for inference
    let activation_memory = self.calculate_activation_memory(&model_config);

    // Add quantization-specific overhead
    let quantization_overhead = self.calculate_quantization_overhead(&model_config);

    // Device-specific allocation strategy
    match device {
        "cpu" => self.create_cpu_memory_requirements(
            model_memory,
            kv_cache_memory,
            activation_memory,
            quantization_overhead,
        ),
        "gpu" => self.create_gpu_memory_requirements(
            model_memory,
            kv_cache_memory,
            activation_memory,
            quantization_overhead,
        ),
        _ => MemoryRequirements::empty(),
    }
}

fn calculate_model_memory(&self, config: &ModelConfig, metadata: &ModelMetadata) -> usize {
    let mut total_params = 0;

    // Count parameters from model architecture
    for layer_config in &config.layers {
        total_params += self.calculate_layer_parameters(layer_config);
    }

    // Account for quantization type
    let bytes_per_param = match config.quantization_type {
        QuantizationType::I2S => 0.25,   // 2 bits per weight
        QuantizationType::TL1 => 0.125,  // 1 bit per weight + lookup table
        QuantizationType::TL2 => 0.125,  // 1 bit per weight + larger lookup table
        QuantizationType::FP16 => 2.0,   // 16 bits per weight
        QuantizationType::FP32 => 4.0,   // 32 bits per weight
    };

    // Add quantization metadata overhead (scales, zero points, lookup tables)
    let quantization_metadata = self.calculate_quantization_metadata_size(config);

    ((total_params as f32 * bytes_per_param) as usize + quantization_metadata) / (1024 * 1024) // Convert to MB
}

fn calculate_kv_cache_memory(&self, config: &ModelConfig) -> usize {
    // KV cache size depends on: num_layers * num_heads * head_dim * max_seq_len * batch_size
    let num_layers = config.num_layers;
    let num_heads = config.num_attention_heads;
    let head_dim = config.hidden_size / num_heads;
    let max_seq_len = config.max_position_embeddings.unwrap_or(2048);
    let max_batch_size = 8; // Configurable default

    // Key and Value tensors for each layer
    let kv_size_per_layer = 2 * num_heads * head_dim * max_seq_len * max_batch_size;
    let total_kv_size = kv_size_per_layer * num_layers;

    // Account for data type (typically FP16 for KV cache)
    let bytes_per_element = 2; // FP16

    (total_kv_size * bytes_per_element) / (1024 * 1024) // Convert to MB
}

fn calculate_activation_memory(&self, config: &ModelConfig) -> usize {
    // Activation memory for intermediate computations
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size.unwrap_or(hidden_size * 4);
    let max_batch_size = 8; // Configurable default
    let max_seq_len = config.max_position_embeddings.unwrap_or(2048);

    // Largest activation tensors: attention matrices and feed-forward intermediate
    let attention_activation = max_batch_size * config.num_attention_heads * max_seq_len * max_seq_len;
    let ff_activation = max_batch_size * max_seq_len * intermediate_size;

    let max_activation = attention_activation.max(ff_activation);
    let bytes_per_element = 4; // FP32 for intermediate computations

    (max_activation * bytes_per_element) / (1024 * 1024) // Convert to MB
}

fn create_gpu_memory_requirements(
    &self,
    model_memory: usize,
    kv_cache_memory: usize,
    activation_memory: usize,
    quantization_overhead: usize,
) -> MemoryRequirements {
    let gpu_total = model_memory + kv_cache_memory + activation_memory + quantization_overhead;
    let safety_margin = (gpu_total as f32 * 0.1) as usize; // 10% safety margin

    MemoryRequirements {
        total_mb: gpu_total + safety_margin,
        gpu_memory_mb: Some(gpu_total + safety_margin),
        cpu_memory_mb: 100, // Minimal CPU overhead for GPU inference
        kv_cache_mb: kv_cache_memory,
        activation_mb: activation_memory,
        headroom_mb: safety_margin,
    }
}
```

### Alternative Approach: Tiered Memory Estimation

For cases where full model analysis isn't feasible:

```rust
pub fn get_memory_requirements(&self, device: &str) -> MemoryRequirements {
    // Get basic model information
    let file_size_mb = self.base_loader.get_model_file_size_mb();
    let param_count = self.estimate_parameter_count();

    // Use lookup table based on known model architectures
    let memory_profile = self.get_memory_profile_for_model_size(param_count);

    // Apply device-specific adjustments
    match device {
        "gpu" => memory_profile.adjust_for_gpu(),
        "cpu" => memory_profile.adjust_for_cpu(),
        _ => MemoryRequirements::empty(),
    }
}
```

## Implementation Plan

### Phase 1: Model Analysis Infrastructure
- [ ] Implement model metadata extraction from GGUF/SafeTensors files
- [ ] Create parameter counting utilities for different layer types
- [ ] Add quantization-aware memory calculation functions
- [ ] Implement configuration parsing for memory-relevant parameters

### Phase 2: Memory Calculation Engine
- [ ] Develop accurate model memory calculation algorithms
- [ ] Implement KV cache size estimation based on architecture
- [ ] Create activation memory calculation for different batch sizes
- [ ] Add quantization metadata size calculations

### Phase 3: Device-Specific Optimization
- [ ] Implement GPU memory layout optimization
- [ ] Add CPU memory management strategies
- [ ] Create memory fragmentation handling
- [ ] Implement safety margins and error handling

### Phase 4: Integration & Validation
- [ ] Integrate with existing model loading pipeline
- [ ] Add configuration options for memory planning
- [ ] Implement memory requirement caching
- [ ] Add comprehensive testing across model sizes

## Testing Strategy

### Accuracy Testing
```rust
#[test]
fn test_memory_requirements_accuracy() {
    let loader = ProductionLoader::new("test-model.gguf")?;
    let requirements = loader.get_memory_requirements("gpu");

    // Load model and measure actual memory usage
    let model = loader.load_model()?;
    let actual_usage = measure_gpu_memory_usage(&model);

    // Memory estimate should be within 20% of actual usage
    assert!(
        requirements.gpu_memory_mb.unwrap() as f32 * 0.8 < actual_usage as f32 &&
        actual_usage as f32 < requirements.gpu_memory_mb.unwrap() as f32 * 1.2
    );
}

#[test]
fn test_quantization_memory_differences() {
    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let requirements = calculate_memory_for_quantization(qtype);
        // Verify quantization type affects memory calculations appropriately
        assert_quantization_memory_scaling(qtype, requirements);
    }
}
```

### Performance Testing
```rust
#[test]
fn test_memory_calculation_performance() {
    let start = Instant::now();
    let _requirements = loader.get_memory_requirements("gpu");
    let duration = start.elapsed();

    // Memory calculation should be fast (<100ms)
    assert!(duration < Duration::from_millis(100));
}
```

## Related Issues/PRs

- Model loading and device selection optimization
- GPU memory management and allocation strategies
- Production deployment resource planning
- Quantization-aware memory optimization

## Acceptance Criteria

- [ ] Memory requirements calculated based on actual model analysis
- [ ] Accurate prediction within 20% of actual memory usage
- [ ] Support for all quantization types (I2S, TL1, TL2, FP16, FP32)
- [ ] Device-specific memory layout optimization
- [ ] Configurable safety margins and batch size parameters
- [ ] Fast calculation performance (<100ms for typical models)
- [ ] Comprehensive error handling for invalid models
- [ ] Documentation for memory planning in production
- [ ] Backward compatibility with existing memory planning code

## Notes

Accurate memory requirement calculation is critical for production deployment, especially for GPU inference where memory is limited and expensive. The implementation should be conservative with safety margins while providing actionable insights for resource planning.

Consider implementing a memory profiling mode that can validate predictions against actual usage during development and testing phases.
