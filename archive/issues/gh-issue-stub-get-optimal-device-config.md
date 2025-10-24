# [Model Loading] Implement intelligent device configuration optimization

## Problem Description

The `get_optimal_device_config` function in `crates/bitnet-models/src/production_loader.rs` currently returns a hardcoded `DeviceConfig` with fixed CPU-only settings. This represents a significant missed opportunity for performance optimization, as the function should intelligently analyze model requirements and available hardware to determine the optimal device configuration.

## Environment

- **File**: `crates/bitnet-models/src/production_loader.rs`
- **Function**: `get_optimal_device_config`
- **Architecture**: Production model loading with device-aware optimization
- **Current Implementation**: Hardcoded CPU-only configuration

## Root Cause Analysis

### Current Implementation
```rust
pub fn get_optimal_device_config(&self) -> DeviceConfig {
    DeviceConfig {
        strategy: Some(DeviceStrategy::CpuOnly),
        cpu_threads: Some(4),
        gpu_memory_fraction: None,
        recommended_batch_size: 1,
    }
}
```

### Issues Identified
1. **No Hardware Detection**: Function ignores available GPU/accelerator hardware
2. **Fixed Thread Count**: CPU threads hardcoded to 4 regardless of system capabilities
3. **No Model Analysis**: Doesn't consider model size or computational requirements
4. **Missing Optimization**: No performance-based device selection
5. **Static Configuration**: Cannot adapt to different deployment scenarios

## Impact Assessment

- **Severity**: High - Significant performance impact
- **Performance Impact**: High - May severely underutilize available hardware
- **User Experience**: High - Poor performance on GPU-enabled systems
- **Production Readiness**: High - Critical for production deployment optimization

## Proposed Solution

### Comprehensive Device Configuration Optimization

```rust
impl ProductionModelLoader {
    /// Analyze system and model to determine optimal device configuration
    pub fn get_optimal_device_config(&self) -> DeviceConfig {
        let model_requirements = self.analyze_model_requirements();
        let hardware_capabilities = self.detect_hardware_capabilities();
        let performance_preferences = self.get_performance_preferences();

        self.optimize_device_configuration(
            &model_requirements,
            &hardware_capabilities,
            &performance_preferences,
        )
    }

    fn analyze_model_requirements(&self) -> ModelRequirements {
        let config = self.base_loader.get_model_config();
        let memory_requirements = self.get_memory_requirements("auto");

        ModelRequirements {
            total_parameters: self.estimate_parameter_count(&config),
            memory_footprint_mb: memory_requirements.total_mb,
            compute_intensity: self.estimate_compute_intensity(&config),
            quantization_type: config.quantization_type.clone(),
            preferred_precision: self.get_preferred_precision(&config),
            batch_sensitivity: self.estimate_batch_sensitivity(&config),
        }
    }

    fn detect_hardware_capabilities(&self) -> HardwareCapabilities {
        let cpu_info = self.detect_cpu_capabilities();
        let gpu_info = self.detect_gpu_capabilities();
        let memory_info = self.detect_memory_capabilities();

        HardwareCapabilities {
            cpu: cpu_info,
            gpu: gpu_info,
            memory: memory_info,
            accelerators: self.detect_accelerators(),
        }
    }

    fn optimize_device_configuration(
        &self,
        model_reqs: &ModelRequirements,
        hw_caps: &HardwareCapabilities,
        prefs: &PerformancePreferences,
    ) -> DeviceConfig {
        let strategy = self.select_optimal_strategy(model_reqs, hw_caps, prefs);
        let batch_size = self.calculate_optimal_batch_size(model_reqs, hw_caps, &strategy);

        match strategy {
            DeviceStrategy::GpuOnly => {
                DeviceConfig {
                    strategy: Some(strategy),
                    cpu_threads: None,
                    gpu_memory_fraction: Some(self.calculate_optimal_gpu_memory_fraction(model_reqs, hw_caps)),
                    recommended_batch_size: batch_size,
                }
            },
            DeviceStrategy::CpuOnly => {
                DeviceConfig {
                    strategy: Some(strategy),
                    cpu_threads: Some(self.calculate_optimal_cpu_threads(hw_caps)),
                    gpu_memory_fraction: None,
                    recommended_batch_size: batch_size,
                }
            },
            DeviceStrategy::Hybrid => {
                DeviceConfig {
                    strategy: Some(strategy),
                    cpu_threads: Some(self.calculate_optimal_cpu_threads_for_hybrid(hw_caps)),
                    gpu_memory_fraction: Some(0.7), // Reserve some GPU memory for hybrid operations
                    recommended_batch_size: batch_size,
                }
            },
        }
    }

    fn select_optimal_strategy(
        &self,
        model_reqs: &ModelRequirements,
        hw_caps: &HardwareCapabilities,
        prefs: &PerformancePreferences,
    ) -> DeviceStrategy {
        // Score different strategies based on expected performance
        let gpu_score = self.score_gpu_strategy(model_reqs, hw_caps, prefs);
        let cpu_score = self.score_cpu_strategy(model_reqs, hw_caps, prefs);
        let hybrid_score = self.score_hybrid_strategy(model_reqs, hw_caps, prefs);

        let best_strategy = [(DeviceStrategy::GpuOnly, gpu_score),
                           (DeviceStrategy::CpuOnly, cpu_score),
                           (DeviceStrategy::Hybrid, hybrid_score)]
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(strategy, _)| strategy.clone())
            .unwrap_or(DeviceStrategy::CpuOnly);

        info!(
            "Selected device strategy: {:?} (GPU: {:.2}, CPU: {:.2}, Hybrid: {:.2})",
            best_strategy, gpu_score, cpu_score, hybrid_score
        );

        best_strategy
    }

    fn score_gpu_strategy(
        &self,
        model_reqs: &ModelRequirements,
        hw_caps: &HardwareCapabilities,
        prefs: &PerformancePreferences,
    ) -> f64 {
        if hw_caps.gpu.devices.is_empty() {
            return 0.0; // No GPU available
        }

        let best_gpu = &hw_caps.gpu.devices[0]; // Assume first GPU is best

        // Check memory requirements
        if model_reqs.memory_footprint_mb > best_gpu.memory_mb as usize {
            return 0.1; // GPU memory insufficient
        }

        // Score based on compute capability and model requirements
        let compute_score = match model_reqs.quantization_type {
            QuantizationType::I2S => {
                if best_gpu.supports_int8_ops { 0.9 } else { 0.6 }
            },
            QuantizationType::TL1 | QuantizationType::TL2 => {
                if best_gpu.tensor_cores { 0.95 } else { 0.7 }
            },
            _ => 0.8,
        };

        let memory_score = 1.0 - (model_reqs.memory_footprint_mb as f64 / best_gpu.memory_mb as f64);
        let throughput_score = best_gpu.compute_capability * 0.1;

        let base_score = (compute_score + memory_score + throughput_score) / 3.0;

        // Apply preferences
        match prefs.priority {
            OptimizationPriority::Throughput => base_score * 1.2,
            OptimizationPriority::Latency => base_score * 1.1,
            OptimizationPriority::Memory => base_score * 0.9,
            OptimizationPriority::Balanced => base_score,
        }
    }

    fn calculate_optimal_cpu_threads(&self, hw_caps: &HardwareCapabilities) -> usize {
        let logical_cores = hw_caps.cpu.logical_cores;
        let physical_cores = hw_caps.cpu.physical_cores;

        // Use 75% of logical cores, but not less than 1 or more than 16
        let optimal_threads = ((logical_cores as f64 * 0.75) as usize)
            .max(1)
            .min(16)
            .min(physical_cores); // Don't exceed physical cores for compute-bound tasks

        info!("Calculated optimal CPU threads: {} (from {} logical, {} physical cores)",
              optimal_threads, logical_cores, physical_cores);

        optimal_threads
    }

    fn calculate_optimal_batch_size(
        &self,
        model_reqs: &ModelRequirements,
        hw_caps: &HardwareCapabilities,
        strategy: &DeviceStrategy,
    ) -> usize {
        match strategy {
            DeviceStrategy::GpuOnly => {
                // GPU can handle larger batches
                if model_reqs.memory_footprint_mb < 2000 {
                    8 // Small model, larger batch
                } else if model_reqs.memory_footprint_mb < 8000 {
                    4 // Medium model, moderate batch
                } else {
                    2 // Large model, small batch
                }
            },
            DeviceStrategy::CpuOnly => {
                // CPU typically better with smaller batches
                if hw_caps.cpu.physical_cores >= 8 {
                    2 // Multi-core CPU
                } else {
                    1 // Few cores, single batch
                }
            },
            DeviceStrategy::Hybrid => {
                2 // Balanced approach for hybrid
            },
        }
    }
}

#[derive(Debug, Clone)]
struct ModelRequirements {
    total_parameters: usize,
    memory_footprint_mb: usize,
    compute_intensity: ComputeIntensity,
    quantization_type: QuantizationType,
    preferred_precision: Precision,
    batch_sensitivity: BatchSensitivity,
}

#[derive(Debug, Clone)]
struct HardwareCapabilities {
    cpu: CpuCapabilities,
    gpu: GpuCapabilities,
    memory: MemoryCapabilities,
    accelerators: Vec<AcceleratorInfo>,
}

#[derive(Debug, Clone)]
struct CpuCapabilities {
    logical_cores: usize,
    physical_cores: usize,
    architecture: CpuArchitecture,
    features: Vec<CpuFeature>,
    cache_sizes: CacheSizes,
    base_frequency_mhz: u32,
    max_frequency_mhz: u32,
}

#[derive(Debug, Clone)]
struct GpuCapabilities {
    devices: Vec<GpuDeviceInfo>,
    cuda_version: Option<String>,
    driver_version: Option<String>,
}

#[derive(Debug, Clone)]
struct GpuDeviceInfo {
    name: String,
    memory_mb: u64,
    compute_capability: f64,
    tensor_cores: bool,
    supports_mixed_precision: bool,
    supports_int8_ops: bool,
    max_threads_per_block: u32,
    max_shared_memory_per_block: u64,
}
```

## Implementation Plan

### Phase 1: Hardware Detection Infrastructure (Week 1)
- [ ] Implement CPU capability detection (cores, features, cache)
- [ ] Add GPU detection and capability analysis
- [ ] Create memory and system resource detection
- [ ] Add accelerator discovery (if applicable)

### Phase 2: Model Analysis Framework (Week 2)
- [ ] Implement model requirement analysis
- [ ] Add memory footprint estimation
- [ ] Create compute intensity classification
- [ ] Add quantization-aware optimization

### Phase 3: Optimization Algorithm (Week 3)
- [ ] Implement strategy scoring system
- [ ] Add performance-based device selection
- [ ] Create optimal configuration calculation
- [ ] Add preference-based optimization

### Phase 4: Integration and Validation (Week 4)
- [ ] Integrate with existing production loader
- [ ] Add comprehensive testing across device types
- [ ] Create performance validation suite
- [ ] Add monitoring and logging

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_only_optimization() {
        let loader = create_test_loader_cpu_only();
        let config = loader.get_optimal_device_config();

        assert_eq!(config.strategy, Some(DeviceStrategy::CpuOnly));
        assert!(config.cpu_threads.unwrap() > 0);
        assert!(config.gpu_memory_fraction.is_none());
    }

    #[test]
    fn test_gpu_optimization() {
        let loader = create_test_loader_with_gpu();
        let config = loader.get_optimal_device_config();

        // Should prefer GPU for large models with sufficient GPU memory
        assert_eq!(config.strategy, Some(DeviceStrategy::GpuOnly));
        assert!(config.gpu_memory_fraction.is_some());
    }

    #[test]
    fn test_memory_constrained_optimization() {
        let loader = create_test_loader_memory_constrained();
        let config = loader.get_optimal_device_config();

        // Should optimize for memory efficiency
        assert!(config.recommended_batch_size <= 2);
    }
}
```

## Acceptance Criteria

- [ ] Intelligent hardware detection across CPU/GPU/accelerators
- [ ] Model-aware device strategy selection
- [ ] Optimal thread and batch size calculation
- [ ] Performance-based optimization scoring
- [ ] Comprehensive logging of optimization decisions
- [ ] Graceful fallback for unsupported hardware
- [ ] Performance improvements measurable in benchmarks

## Priority: High

This is critical for production performance optimization and user experience, enabling the system to automatically utilize available hardware effectively.
