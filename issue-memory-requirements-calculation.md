# [FEATURE] Implement accurate memory requirements calculation for production model loader

## Problem Description

The `get_memory_requirements` function in `crates/bitnet-models/src/production_loader.rs` uses hardcoded values and simplified calculations instead of analyzing the actual model to determine precise memory requirements. This limitation prevents accurate resource planning, optimal device selection, and proper memory allocation for production deployments.

## Environment

**Affected Component:** `crates/bitnet-models/src/production_loader.rs`
**Function:** `get_memory_requirements`
**Impact:** Memory planning, resource allocation, production deployment reliability
**Related Features:** Model loading, device selection, memory management

## Root Cause Analysis

### Current Implementation Limitations

1. **Hardcoded base memory**: Uses fixed 1000MB base regardless of model size
2. **No model analysis**: Doesn't examine actual model tensors or architecture
3. **Oversimplified device handling**: Basic CPU/GPU differentiation only
4. **Missing quantization awareness**: Doesn't account for quantization type impact on memory

### Code Analysis

```rust
pub fn get_memory_requirements(&self, device: &str) -> MemoryRequirements {
    // This is a simplified implementation
    let base_memory = 1000; // Base memory in MB

    match device {
        "cpu" => MemoryRequirements {
            total_mb: base_memory,
            gpu_memory_mb: None,
            cpu_memory_mb: base_memory - 200,
            kv_cache_mb: 100,       // Hardcoded
            activation_mb: 50,      // Hardcoded
            headroom_mb: 50,        // Hardcoded
        },
        // ... more hardcoded values
    }
}
```

Issues:
- No correlation between model size and memory requirements
- Fixed KV cache size regardless of model architecture
- No consideration of batch size, sequence length, or quantization
- Lacks precision needed for production resource planning

## Impact Assessment

### Production Impact
- **Resource waste**: Over-allocation due to conservative estimates
- **Deployment failures**: Under-allocation causing OOM errors
- **Cost inefficiency**: Suboptimal instance sizing in cloud environments
- **Performance degradation**: Incorrect memory assumptions affecting optimization

### Development Impact
- **Testing inconsistency**: Unreliable memory requirements for test environments
- **Debugging difficulty**: Cannot accurately predict memory-related issues
- **Optimization barriers**: Cannot make informed memory vs. performance trade-offs

## Proposed Solution

### Comprehensive Memory Analysis System

Implement accurate memory calculation based on model introspection and runtime requirements:

```rust
#[derive(Debug, Clone)]
pub struct DetailedMemoryRequirements {
    pub model_weights: MemorySegment,
    pub kv_cache: MemorySegment,
    pub activations: MemorySegment,
    pub quantization_overhead: MemorySegment,
    pub system_overhead: MemorySegment,
    pub total: MemorySegment,
    pub peak_usage: MemorySegment,
    pub recommendations: MemoryRecommendations,
}

#[derive(Debug, Clone)]
pub struct MemorySegment {
    pub cpu_mb: u64,
    pub gpu_mb: Option<u64>,
    pub shared_mb: u64,
    pub peak_multiplier: f32,
}

#[derive(Debug, Clone)]
pub struct MemoryRecommendations {
    pub min_system_ram: u64,
    pub recommended_system_ram: u64,
    pub min_gpu_memory: Option<u64>,
    pub recommended_gpu_memory: Option<u64>,
    pub swap_requirements: u64,
    pub optimization_suggestions: Vec<String>,
}

impl ProductionModelLoader {
    pub fn get_detailed_memory_requirements(
        &self,
        config: &MemoryAnalysisConfig,
    ) -> Result<DetailedMemoryRequirements> {
        let model_info = self.analyze_model_structure()?;
        let runtime_config = &config.runtime;

        let model_weights = self.calculate_model_weights_memory(&model_info, config)?;
        let kv_cache = self.calculate_kv_cache_memory(&model_info, runtime_config)?;
        let activations = self.calculate_activation_memory(&model_info, runtime_config)?;
        let quantization_overhead = self.calculate_quantization_overhead(&model_info, config)?;
        let system_overhead = self.calculate_system_overhead(config)?;

        let total = self.aggregate_memory_segments(&[
            &model_weights,
            &kv_cache,
            &activations,
            &quantization_overhead,
            &system_overhead,
        ]);

        let peak_usage = self.calculate_peak_memory_usage(&total, &model_info, runtime_config)?;
        let recommendations = self.generate_memory_recommendations(&peak_usage, config)?;

        Ok(DetailedMemoryRequirements {
            model_weights,
            kv_cache,
            activations,
            quantization_overhead,
            system_overhead,
            total,
            peak_usage,
            recommendations,
        })
    }

    fn analyze_model_structure(&self) -> Result<ModelStructureInfo> {
        let metadata = self.base_loader.get_model_metadata()?;
        let tensor_info = self.base_loader.analyze_tensors()?;

        Ok(ModelStructureInfo {
            architecture: metadata.architecture.clone(),
            num_layers: metadata.num_layers,
            hidden_size: metadata.hidden_size,
            num_attention_heads: metadata.num_attention_heads,
            vocab_size: metadata.vocab_size,
            tensors: tensor_info,
            quantization_type: self.detect_quantization_type(&tensor_info)?,
            parameter_count: self.calculate_parameter_count(&tensor_info),
        })
    }

    fn calculate_model_weights_memory(
        &self,
        model_info: &ModelStructureInfo,
        config: &MemoryAnalysisConfig,
    ) -> Result<MemorySegment> {
        let mut total_bytes = 0u64;

        for tensor in &model_info.tensors {
            let tensor_bytes = match &tensor.quantization {
                QuantizationType::I2S => {
                    // 2 bits per parameter + metadata
                    (tensor.element_count * 2 + 7) / 8 + tensor.metadata_overhead()
                }
                QuantizationType::TL1 | QuantizationType::TL2 => {
                    // Lookup table quantization overhead
                    tensor.element_count / 4 + tensor.lookup_table_size()
                }
                QuantizationType::F32 => tensor.element_count * 4,
                QuantizationType::F16 => tensor.element_count * 2,
            };
            total_bytes += tensor_bytes;
        }

        let device_allocation = self.calculate_device_allocation(total_bytes, config)?;

        Ok(MemorySegment {
            cpu_mb: device_allocation.cpu_bytes / 1024 / 1024,
            gpu_mb: device_allocation.gpu_bytes.map(|b| b / 1024 / 1024),
            shared_mb: device_allocation.shared_bytes / 1024 / 1024,
            peak_multiplier: 1.0, // Model weights don't vary during inference
        })
    }

    fn calculate_kv_cache_memory(
        &self,
        model_info: &ModelStructureInfo,
        runtime_config: &RuntimeConfig,
    ) -> Result<MemorySegment> {
        let num_layers = model_info.num_layers;
        let num_heads = model_info.num_attention_heads;
        let head_dim = model_info.hidden_size / num_heads;
        let max_batch_size = runtime_config.max_batch_size;
        let max_sequence_length = runtime_config.max_sequence_length;

        // KV cache: 2 (K+V) * num_layers * batch_size * num_heads * seq_len * head_dim * precision
        let precision_bytes = match runtime_config.kv_cache_precision {
            Precision::F32 => 4,
            Precision::F16 => 2,
            Precision::BF16 => 2,
        };

        let kv_cache_bytes = 2 * num_layers * max_batch_size * num_heads
            * max_sequence_length * head_dim * precision_bytes;

        let device_allocation = self.calculate_device_allocation(kv_cache_bytes as u64, &MemoryAnalysisConfig {
            device: runtime_config.device.clone(),
            ..Default::default()
        })?;

        Ok(MemorySegment {
            cpu_mb: device_allocation.cpu_bytes / 1024 / 1024,
            gpu_mb: device_allocation.gpu_bytes.map(|b| b / 1024 / 1024),
            shared_mb: device_allocation.shared_bytes / 1024 / 1024,
            peak_multiplier: 1.0, // KV cache grows to max size during inference
        })
    }

    fn calculate_activation_memory(
        &self,
        model_info: &ModelStructureInfo,
        runtime_config: &RuntimeConfig,
    ) -> Result<MemorySegment> {
        let hidden_size = model_info.hidden_size;
        let intermediate_size = model_info.intermediate_size.unwrap_or(hidden_size * 4);
        let max_batch_size = runtime_config.max_batch_size;
        let max_sequence_length = runtime_config.max_sequence_length;

        // Activation memory for forward pass through transformer layers
        let activation_elements = max_batch_size * max_sequence_length *
            (hidden_size + intermediate_size + hidden_size); // Input + FFN + Output

        let precision_bytes = match runtime_config.activation_precision {
            Precision::F32 => 4,
            Precision::F16 => 2,
            Precision::BF16 => 2,
        };

        let activation_bytes = activation_elements * precision_bytes;

        let device_allocation = self.calculate_device_allocation(activation_bytes as u64, &MemoryAnalysisConfig {
            device: runtime_config.device.clone(),
            ..Default::default()
        })?;

        Ok(MemorySegment {
            cpu_mb: device_allocation.cpu_bytes / 1024 / 1024,
            gpu_mb: device_allocation.gpu_bytes.map(|b| b / 1024 / 1024),
            shared_mb: device_allocation.shared_bytes / 1024 / 1024,
            peak_multiplier: 1.5, // Activations can spike during computation
        })
    }

    fn calculate_quantization_overhead(
        &self,
        model_info: &ModelStructureInfo,
        config: &MemoryAnalysisConfig,
    ) -> Result<MemorySegment> {
        let overhead_bytes = match &model_info.quantization_type {
            QuantizationType::I2S => {
                // Scale and zero-point metadata per tensor
                model_info.tensors.len() * 8 // 4 bytes scale + 4 bytes zero_point
            }
            QuantizationType::TL1 | QuantizationType::TL2 => {
                // Lookup table storage
                model_info.tensors.iter()
                    .map(|t| t.lookup_table_size())
                    .sum::<usize>()
            }
            _ => 0,
        };

        let device_allocation = self.calculate_device_allocation(overhead_bytes as u64, config)?;

        Ok(MemorySegment {
            cpu_mb: device_allocation.cpu_bytes / 1024 / 1024,
            gpu_mb: device_allocation.gpu_bytes.map(|b| b / 1024 / 1024),
            shared_mb: device_allocation.shared_bytes / 1024 / 1024,
            peak_multiplier: 1.0,
        })
    }

    fn generate_memory_recommendations(
        &self,
        peak_usage: &MemorySegment,
        config: &MemoryAnalysisConfig,
    ) -> Result<MemoryRecommendations> {
        let mut suggestions = Vec::new();

        // System RAM recommendations
        let min_system_ram = peak_usage.cpu_mb + peak_usage.shared_mb + 2048; // 2GB OS overhead
        let recommended_system_ram = min_system_ram * 150 / 100; // 50% safety margin

        // GPU memory recommendations
        let (min_gpu_memory, recommended_gpu_memory) = if let Some(gpu_mb) = peak_usage.gpu_mb {
            let min_gpu = gpu_mb + 512; // GPU driver overhead
            let recommended_gpu = min_gpu * 130 / 100; // 30% safety margin
            (Some(min_gpu), Some(recommended_gpu))
        } else {
            (None, None)
        };

        // Optimization suggestions
        if peak_usage.cpu_mb > 8192 {
            suggestions.push("Consider using GPU acceleration to reduce CPU memory usage".to_string());
        }

        if let Some(gpu_mb) = peak_usage.gpu_mb {
            if gpu_mb > 16384 {
                suggestions.push("Consider model quantization to reduce GPU memory usage".to_string());
            }
        }

        if config.runtime.max_batch_size > 1 {
            suggestions.push("Reduce batch size if experiencing memory pressure".to_string());
        }

        Ok(MemoryRecommendations {
            min_system_ram,
            recommended_system_ram,
            min_gpu_memory,
            recommended_gpu_memory,
            swap_requirements: min_system_ram / 4, // 25% of RAM as swap
            optimization_suggestions: suggestions,
        })
    }
}

#[derive(Debug, Clone)]
pub struct MemoryAnalysisConfig {
    pub device: Device,
    pub runtime: RuntimeConfig,
    pub safety_margin: f32,
    pub include_system_overhead: bool,
    pub detailed_breakdown: bool,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub kv_cache_precision: Precision,
    pub activation_precision: Precision,
    pub device: Device,
}
```

## Implementation Plan

### Phase 1: Model Analysis Infrastructure (2-3 days)
- [ ] Implement model structure analysis and tensor introspection
- [ ] Add quantization type detection from model metadata
- [ ] Create parameter counting and memory calculation utilities
- [ ] Add device allocation planning algorithms

### Phase 2: Memory Calculation Engine (2-3 days)
- [ ] Implement accurate model weights memory calculation
- [ ] Add KV cache memory estimation based on architecture
- [ ] Create activation memory calculation for different precisions
- [ ] Implement quantization overhead calculations

### Phase 3: Recommendation System (1-2 days)
- [ ] Add memory usage optimization suggestions
- [ ] Implement device-specific recommendations
- [ ] Create safety margin calculations and warnings
- [ ] Add configuration validation and error handling

### Phase 4: Integration & Testing (1-2 days)
- [ ] Integrate with existing production loader
- [ ] Add comprehensive unit tests for memory calculations
- [ ] Validate accuracy with real model deployments
- [ ] Performance benchmarking and optimization

## Testing Strategy

### Memory Calculation Testing
```rust
#[test]
fn test_memory_calculation_accuracy() {
    let loader = ProductionModelLoader::new("test_model.gguf").unwrap();
    let config = MemoryAnalysisConfig::default();

    let requirements = loader.get_detailed_memory_requirements(&config).unwrap();

    // Test model weights calculation
    let expected_model_mb = 1500; // Known test model size
    assert!((requirements.model_weights.cpu_mb as i64 - expected_model_mb).abs() < 100);

    // Test KV cache scaling
    let config_large_batch = MemoryAnalysisConfig {
        runtime: RuntimeConfig {
            max_batch_size: 8,
            max_sequence_length: 2048,
            ..Default::default()
        },
        ..config
    };

    let large_requirements = loader.get_detailed_memory_requirements(&config_large_batch).unwrap();
    assert!(large_requirements.kv_cache.cpu_mb > requirements.kv_cache.cpu_mb);
}

#[test]
fn test_quantization_memory_impact() {
    let test_cases = vec![
        ("model_f32.gguf", QuantizationType::F32),
        ("model_i2s.gguf", QuantizationType::I2S),
        ("model_tl1.gguf", QuantizationType::TL1),
    ];

    for (model_path, expected_quant_type) in test_cases {
        let loader = ProductionModelLoader::new(model_path).unwrap();
        let requirements = loader.get_detailed_memory_requirements(&MemoryAnalysisConfig::default()).unwrap();

        match expected_quant_type {
            QuantizationType::I2S => {
                // I2S should use significantly less memory than F32
                assert!(requirements.model_weights.cpu_mb < 1000); // Significantly compressed
            }
            QuantizationType::F32 => {
                // F32 should use full precision memory
                assert!(requirements.model_weights.cpu_mb > 2000);
            }
            _ => {} // Other quantization types
        }
    }
}
```

### Recommendation Testing
```rust
#[test]
fn test_memory_recommendations() {
    let loader = ProductionModelLoader::new("large_model.gguf").unwrap();
    let config = MemoryAnalysisConfig::default();

    let requirements = loader.get_detailed_memory_requirements(&config).unwrap();

    // Should recommend sufficient system RAM
    assert!(requirements.recommendations.recommended_system_ram > requirements.recommendations.min_system_ram);

    // Should provide optimization suggestions for large models
    assert!(!requirements.recommendations.optimization_suggestions.is_empty());

    // GPU recommendations should be present for GPU-capable models
    if config.device.is_gpu() {
        assert!(requirements.recommendations.min_gpu_memory.is_some());
    }
}
```

### Cross-Validation Testing
```rust
#[test]
fn test_actual_vs_predicted_memory() {
    let loader = ProductionModelLoader::new("validation_model.gguf").unwrap();
    let predicted = loader.get_detailed_memory_requirements(&MemoryAnalysisConfig::default()).unwrap();

    // Load model and measure actual memory usage
    let model = loader.load().unwrap();
    let actual_usage = measure_actual_memory_usage(&model);

    // Predictions should be within 20% of actual usage
    let prediction_accuracy = predicted.total.cpu_mb as f32 / actual_usage.cpu_mb as f32;
    assert!(prediction_accuracy > 0.8 && prediction_accuracy < 1.2);
}
```

## Risk Assessment

### Implementation Risks
- **Accuracy concerns**: Model analysis may not capture all memory usage patterns
- **Performance impact**: Detailed analysis may slow model loading
- **Compatibility issues**: Different model formats may require specialized handling

### Mitigation Strategies
- Validate calculations against actual memory usage in production
- Implement caching for repeated memory analysis operations
- Add fallback to simplified calculations for unsupported models
- Include safety margins in all recommendations

## Success Criteria

### Accuracy Improvements
- [ ] Memory predictions within 15% of actual usage for supported models
- [ ] Correct handling of different quantization types and their memory impact
- [ ] Accurate KV cache and activation memory scaling with batch size and sequence length
- [ ] Useful optimization recommendations for memory-constrained environments

### Production Readiness
- [ ] Integration with existing production loader without breaking changes
- [ ] Performance overhead < 100ms for memory analysis
- [ ] Comprehensive error handling for unsupported model formats
- [ ] Clear documentation and usage examples

## Related Issues

- **Device Selection**: Integration with optimal device configuration selection
- **Resource Planning**: Cloud deployment and instance sizing optimization
- **Memory Management**: Dynamic memory allocation and optimization systems

## References

- Model architecture memory requirements documentation
- Quantization impact on memory usage patterns
- Production deployment memory planning best practices
- GPU memory management and optimization techniques

---

**Priority**: High
**Estimated Effort**: 5-7 developer days
**Components**: bitnet-models, memory management
**Feature Flags**: `cpu`, `gpu`