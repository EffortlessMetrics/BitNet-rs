# [IMPLEMENTATION] Implement intelligent device configuration optimization in ProductionModelLoader

## Problem Description

The `get_optimal_device_config` method in `crates/bitnet-models/src/production_loader.rs` returns hardcoded device configurations instead of analyzing system capabilities and model requirements to determine optimal settings. This prevents the system from adapting to different hardware configurations and workloads.

## Environment

- **File**: `crates/bitnet-models/src/production_loader.rs`
- **Method**: `ProductionModelLoader::get_optimal_device_config`
- **Current Implementation**: Hardcoded return values
- **Rust Version**: 1.90.0+
- **Feature Flags**: `cpu`, `gpu`

## Root Cause Analysis

The current implementation provides static configurations regardless of system state:

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

Additionally, there's a similar hardcoded implementation in the mock model:

```rust
pub fn get_optimal_device_config(&self) -> DeviceConfig {
    DeviceConfig {
        strategy: Some(DeviceStrategy::CpuOnly),
        cpu_threads: Some(1),
        gpu_memory_fraction: None,
        recommended_batch_size: 1,
    }
}
```

### Technical Issues Identified

1. **No Hardware Analysis**: Doesn't query available CPU cores, memory, or GPU capabilities
2. **No Model Analysis**: Doesn't consider model size, quantization type, or computational requirements
3. **Static Configuration**: Same configuration returned regardless of system state
4. **Suboptimal Performance**: May not utilize available hardware effectively
5. **No Adaptability**: Cannot adjust to different deployment scenarios

### Impact Assessment

**Severity**: Medium
**Category**: Performance Optimization / Production Readiness

**Current Impact**:
- Suboptimal resource utilization across different hardware configurations
- Poor performance on systems with different capabilities
- No automatic scaling based on available resources
- Manual configuration required for optimal performance

**Future Risks**:
- Performance bottlenecks in production deployments
- Inefficient resource usage leading to higher operational costs
- Poor user experience on various hardware configurations
- Difficulty in automated deployment scenarios

## Proposed Solution

### Primary Approach: Intelligent Device Configuration System

Implement a comprehensive device configuration optimization system that analyzes both system capabilities and model requirements.

**Implementation Plan:**

```rust
use std::sync::Arc;
use std::collections::HashMap;

/// Enhanced device configuration with system analysis
#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    /// Number of physical CPU cores
    pub physical_cores: usize,
    /// Number of logical CPU cores (with hyperthreading)
    pub logical_cores: usize,
    /// Total system memory in MB
    pub total_memory_mb: u64,
    /// Available system memory in MB
    pub available_memory_mb: u64,
    /// CPU features (AVX, AVX2, AVX-512, etc.)
    pub cpu_features: Vec<String>,
    /// GPU information
    pub gpu_info: Vec<GpuInfo>,
    /// System load information
    pub system_load: SystemLoad,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU device ID
    pub device_id: usize,
    /// GPU name/model
    pub name: String,
    /// Total VRAM in MB
    pub vram_total_mb: u64,
    /// Available VRAM in MB
    pub vram_available_mb: u64,
    /// Compute capability (for CUDA)
    pub compute_capability: Option<(i32, i32)>,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: Option<f64>,
    /// Is currently available for use
    pub available: bool,
}

#[derive(Debug, Clone)]
pub struct SystemLoad {
    /// Current CPU utilization (0.0-1.0)
    pub cpu_utilization: f64,
    /// Current memory utilization (0.0-1.0)
    pub memory_utilization: f64,
    /// GPU utilization per device
    pub gpu_utilization: HashMap<usize, f64>,
}

#[derive(Debug, Clone)]
pub struct ModelRequirements {
    /// Estimated model memory footprint in MB
    pub model_memory_mb: u64,
    /// Estimated working memory needed in MB
    pub working_memory_mb: u64,
    /// Computational intensity (operations per token)
    pub computational_intensity: f64,
    /// Quantization type
    pub quantization_type: Option<QuantizationType>,
    /// Preferred batch sizes for optimal performance
    pub optimal_batch_sizes: Vec<usize>,
    /// Memory bandwidth requirements in GB/s
    pub memory_bandwidth_requirement: Option<f64>,
}

impl ProductionModelLoader {
    /// Get optimal device configuration based on system analysis
    pub fn get_optimal_device_config(&self) -> DeviceConfig {
        // Gather system capabilities
        let system_capabilities = self.analyze_system_capabilities()
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to analyze system capabilities: {}, using defaults", e);
                self.get_default_capabilities()
            });

        // Analyze model requirements
        let model_requirements = self.analyze_model_requirements()
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to analyze model requirements: {}, using defaults", e);
                self.get_default_model_requirements()
            });

        // Generate optimal configuration
        self.optimize_device_configuration(&system_capabilities, &model_requirements)
    }

    /// Analyze current system capabilities
    fn analyze_system_capabilities(&self) -> Result<SystemCapabilities> {
        let mut system = sysinfo::System::new_all();
        system.refresh_all();

        // CPU analysis
        let physical_cores = system.physical_core_count().unwrap_or(1);
        let logical_cores = system.cpus().len();

        // Memory analysis
        let total_memory_mb = system.total_memory() / 1024 / 1024;
        let available_memory_mb = system.available_memory() / 1024 / 1024;

        // CPU features detection
        let cpu_features = self.detect_cpu_features();

        // GPU analysis
        let gpu_info = self.analyze_gpu_capabilities()?;

        // System load
        let system_load = self.get_current_system_load(&system)?;

        Ok(SystemCapabilities {
            physical_cores,
            logical_cores,
            total_memory_mb,
            available_memory_mb,
            cpu_features,
            gpu_info,
            system_load,
        })
    }

    /// Analyze model-specific requirements
    fn analyze_model_requirements(&self) -> Result<ModelRequirements> {
        // Get model configuration if available
        let model_config = self.base_loader.get_model_config()
            .unwrap_or_else(|| bitnet_common::BitNetConfig::default());

        // Estimate memory requirements
        let model_memory_mb = self.estimate_model_memory(&model_config)?;
        let working_memory_mb = self.estimate_working_memory(&model_config)?;

        // Analyze computational requirements
        let computational_intensity = self.calculate_computational_intensity(&model_config);

        // Determine quantization type
        let quantization_type = self.detect_quantization_type(&model_config);

        // Calculate optimal batch sizes
        let optimal_batch_sizes = self.calculate_optimal_batch_sizes(
            &model_config,
            model_memory_mb,
            working_memory_mb,
        );

        // Estimate bandwidth requirements
        let memory_bandwidth_requirement = self.estimate_bandwidth_requirement(&model_config);

        Ok(ModelRequirements {
            model_memory_mb,
            working_memory_mb,
            computational_intensity,
            quantization_type,
            optimal_batch_sizes,
            memory_bandwidth_requirement,
        })
    }

    /// Optimize device configuration based on capabilities and requirements
    fn optimize_device_configuration(
        &self,
        capabilities: &SystemCapabilities,
        requirements: &ModelRequirements,
    ) -> DeviceConfig {
        // Determine optimal device strategy
        let strategy = self.select_optimal_device_strategy(capabilities, requirements);

        // Optimize CPU configuration
        let cpu_threads = self.optimize_cpu_threads(capabilities, requirements, &strategy);

        // Optimize GPU configuration
        let gpu_memory_fraction = self.optimize_gpu_memory(capabilities, requirements, &strategy);

        // Determine optimal batch size
        let recommended_batch_size = self.optimize_batch_size(capabilities, requirements, &strategy);

        DeviceConfig {
            strategy: Some(strategy),
            cpu_threads,
            gpu_memory_fraction,
            recommended_batch_size,
        }
    }

    /// Select optimal device strategy
    fn select_optimal_device_strategy(
        &self,
        capabilities: &SystemCapabilities,
        requirements: &ModelRequirements,
    ) -> DeviceStrategy {
        // Check if GPU is available and suitable
        for gpu in &capabilities.gpu_info {
            if gpu.available && gpu.vram_available_mb >= requirements.model_memory_mb + requirements.working_memory_mb {
                // GPU has sufficient memory
                if self.is_gpu_faster_than_cpu(capabilities, requirements, gpu) {
                    tracing::info!(
                        "Selected GPU strategy: {} with {}MB VRAM available",
                        gpu.name, gpu.vram_available_mb
                    );
                    return DeviceStrategy::GpuOnly;
                }
            }
        }

        // Check if hybrid approach is beneficial
        if !capabilities.gpu_info.is_empty()
           && capabilities.available_memory_mb >= requirements.model_memory_mb + requirements.working_memory_mb {
            // Consider hybrid approach for large models
            if requirements.model_memory_mb > 4096 { // 4GB threshold
                tracing::info!("Selected hybrid strategy for large model");
                return DeviceStrategy::Hybrid;
            }
        }

        // Fallback to CPU
        tracing::info!("Selected CPU-only strategy");
        DeviceStrategy::CpuOnly
    }

    /// Optimize CPU thread count
    fn optimize_cpu_threads(
        &self,
        capabilities: &SystemCapabilities,
        requirements: &ModelRequirements,
        strategy: &DeviceStrategy,
    ) -> Option<usize> {
        match strategy {
            DeviceStrategy::GpuOnly => {
                // Minimal CPU threads for GPU-only strategy
                Some(2.min(capabilities.physical_cores))
            }
            DeviceStrategy::CpuOnly | DeviceStrategy::Hybrid => {
                // Optimize for computational intensity and memory bandwidth
                let base_threads = capabilities.physical_cores;

                // Adjust based on system load
                let load_factor = 1.0 - capabilities.system_load.cpu_utilization.min(0.8);
                let adjusted_threads = ((base_threads as f64) * load_factor) as usize;

                // Consider memory bandwidth limitations
                let bandwidth_limited_threads = if let Some(bandwidth_req) = requirements.memory_bandwidth_requirement {
                    // Estimate optimal threads based on bandwidth
                    let estimated_bandwidth = self.estimate_system_bandwidth(capabilities);
                    let max_parallel = (estimated_bandwidth / bandwidth_req) as usize;
                    max_parallel.min(base_threads)
                } else {
                    base_threads
                };

                let optimal_threads = adjusted_threads.min(bandwidth_limited_threads).max(1);

                tracing::debug!(
                    "Optimized CPU threads: base={}, load_adjusted={}, bandwidth_limited={}, final={}",
                    base_threads, adjusted_threads, bandwidth_limited_threads, optimal_threads
                );

                Some(optimal_threads)
            }
            DeviceStrategy::Auto => {
                // Auto strategy - conservative approach
                Some(capabilities.physical_cores / 2)
            }
        }
    }

    /// Optimize GPU memory allocation
    fn optimize_gpu_memory(
        &self,
        capabilities: &SystemCapabilities,
        requirements: &ModelRequirements,
        strategy: &DeviceStrategy,
    ) -> Option<f32> {
        match strategy {
            DeviceStrategy::GpuOnly | DeviceStrategy::Hybrid => {
                if let Some(gpu) = capabilities.gpu_info.first() {
                    let required_memory = requirements.model_memory_mb + requirements.working_memory_mb;
                    let safety_margin = 0.9; // Leave 10% headroom

                    let optimal_fraction = ((required_memory as f64) / (gpu.vram_total_mb as f64) * (1.0 / safety_margin))
                        .min(0.95) // Never use more than 95%
                        .max(0.1); // Always use at least 10%

                    tracing::debug!(
                        "Optimized GPU memory fraction: {:.2} ({}MB required, {}MB total)",
                        optimal_fraction, required_memory, gpu.vram_total_mb
                    );

                    Some(optimal_fraction as f32)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Optimize batch size based on memory and performance characteristics
    fn optimize_batch_size(
        &self,
        capabilities: &SystemCapabilities,
        requirements: &ModelRequirements,
        strategy: &DeviceStrategy,
    ) -> usize {
        // Start with model-recommended batch sizes
        let candidate_batches = if requirements.optimal_batch_sizes.is_empty() {
            vec![1, 2, 4, 8, 16]
        } else {
            requirements.optimal_batch_sizes.clone()
        };

        // Filter based on memory constraints
        let memory_constrained_batches: Vec<usize> = candidate_batches
            .into_iter()
            .filter(|&batch_size| {
                let estimated_memory = requirements.working_memory_mb * (batch_size as u64);
                match strategy {
                    DeviceStrategy::GpuOnly => {
                        capabilities.gpu_info.first()
                            .map(|gpu| estimated_memory < gpu.vram_available_mb * 8 / 10) // 80% threshold
                            .unwrap_or(false)
                    }
                    _ => estimated_memory < capabilities.available_memory_mb * 8 / 10
                }
            })
            .collect();

        // Select largest viable batch size
        let optimal_batch = memory_constrained_batches
            .into_iter()
            .max()
            .unwrap_or(1);

        tracing::debug!("Optimized batch size: {}", optimal_batch);
        optimal_batch
    }

    // Helper methods for capability detection

    fn detect_cpu_features(&self) -> Vec<String> {
        let mut features = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                features.push("AVX".to_string());
            }
            if is_x86_feature_detected!("avx2") {
                features.push("AVX2".to_string());
            }
            if is_x86_feature_detected!("avx512f") {
                features.push("AVX512F".to_string());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                features.push("NEON".to_string());
            }
        }

        features
    }

    fn analyze_gpu_capabilities(&self) -> Result<Vec<GpuInfo>> {
        let mut gpu_info = Vec::new();

        #[cfg(feature = "gpu")]
        {
            // Use CUDA runtime to detect GPUs
            if let Ok(device_count) = cudarc::driver::result::device::get_count() {
                for device_id in 0..device_count {
                    if let Ok(device) = cudarc::driver::CudaDevice::new(device_id) {
                        let total_memory = device.total_memory().unwrap_or(0) / (1024 * 1024); // Convert to MB
                        let available_memory = total_memory; // Simplified - would need actual free memory query

                        gpu_info.push(GpuInfo {
                            device_id,
                            name: format!("CUDA Device {}", device_id),
                            vram_total_mb: total_memory,
                            vram_available_mb: available_memory,
                            compute_capability: None, // Would need to query actual capability
                            memory_bandwidth_gbps: None,
                            available: true,
                        });
                    }
                }
            }
        }

        Ok(gpu_info)
    }

    fn is_gpu_faster_than_cpu(
        &self,
        capabilities: &SystemCapabilities,
        requirements: &ModelRequirements,
        gpu: &GpuInfo,
    ) -> bool {
        // Heuristic-based decision
        // GPU is generally faster for:
        // 1. Large models (>1GB)
        // 2. High computational intensity
        // 3. Parallel workloads

        let model_size_factor = requirements.model_memory_mb >= 1024; // 1GB threshold
        let compute_factor = requirements.computational_intensity > 1000.0; // High compute threshold
        let parallelism_factor = capabilities.physical_cores <= 8; // CPU core count threshold

        model_size_factor || compute_factor || parallelism_factor
    }

    // ... additional helper methods for memory estimation, etc.
}
```

### Alternative Approaches

**Option 2: Profile-Based Configuration**
```rust
impl ProductionModelLoader {
    pub fn get_optimal_device_config(&self) -> DeviceConfig {
        // Use pre-computed performance profiles
        let system_profile = self.get_system_performance_profile();
        let model_profile = self.get_model_performance_profile();

        self.lookup_optimal_config(&system_profile, &model_profile)
    }
}
```

**Option 3: Adaptive Learning System**
```rust
impl ProductionModelLoader {
    pub fn get_optimal_device_config(&self) -> DeviceConfig {
        // Use historical performance data to optimize
        let historical_data = self.load_performance_history();
        let current_system = self.analyze_current_system();

        self.predict_optimal_config(&historical_data, &current_system)
    }
}
```

## Implementation Roadmap

### Phase 1: System Analysis Infrastructure (3-4 days)
- [ ] Implement system capabilities detection (CPU, memory, GPU)
- [ ] Create model requirements analysis framework
- [ ] Add system load monitoring capabilities
- [ ] Implement basic device strategy selection

### Phase 2: Configuration Optimization (3-4 days)
- [ ] Implement CPU thread optimization algorithms
- [ ] Add GPU memory fraction optimization
- [ ] Create batch size optimization logic
- [ ] Add performance heuristics and decision trees

### Phase 3: Integration and Testing (2-3 days)
- [ ] Integrate optimization system with existing loader
- [ ] Add comprehensive testing across different hardware
- [ ] Implement fallback mechanisms for analysis failures
- [ ] Add performance benchmarking and validation

### Phase 4: Fine-tuning and Documentation (1-2 days)
- [ ] Optimize algorithms based on test results
- [ ] Add configuration override mechanisms
- [ ] Create comprehensive documentation
- [ ] Add usage examples and best practices

## Testing Strategy

### Test Coverage Requirements
- [ ] Unit tests for all optimization algorithms
- [ ] Integration tests on different hardware configurations
- [ ] Performance regression testing
- [ ] Mock testing for systems without GPU
- [ ] Edge case testing (low memory, high load)

### Hardware Compatibility Testing
```rust
#[cfg(test)]
mod device_optimization_tests {
    use super::*;

    #[test]
    fn test_cpu_only_optimization() {
        let loader = ProductionModelLoader::new();
        // Mock system with no GPU
        let config = loader.get_optimal_device_config();
        assert_eq!(config.strategy, Some(DeviceStrategy::CpuOnly));
        assert!(config.cpu_threads.unwrap() > 0);
    }

    #[test]
    fn test_gpu_optimization() {
        // Test with mock GPU system
        let loader = ProductionModelLoader::new();
        // Would need GPU simulation
        let config = loader.get_optimal_device_config();
        // Verify GPU-specific optimizations
    }
}
```

## Acceptance Criteria

- [ ] **Dynamic Configuration**: Device configuration adapts to system capabilities
- [ ] **Performance Optimization**: Configurations optimize for available hardware
- [ ] **Memory Awareness**: Configurations respect memory constraints
- [ ] **Fallback Handling**: Graceful degradation when optimal hardware unavailable
- [ ] **Documentation**: Clear documentation of optimization algorithms
- [ ] **Backward Compatibility**: Existing functionality preserved
- [ ] **Performance**: Configuration generation completes in <100ms

## Related Issues

- System capabilities detection infrastructure
- Performance profiling and benchmarking framework
- GPU memory management optimization
- Adaptive batch size selection system

---

**Labels**: `enhancement`, `performance`, `infrastructure`, `P2-medium`, `models`
**Priority**: Medium - Important for performance optimization
**Effort**: 9-12 days