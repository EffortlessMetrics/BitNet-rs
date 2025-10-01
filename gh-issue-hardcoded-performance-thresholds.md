# [Configuration] Replace hardcoded performance thresholds with adaptive system

## Problem Description

The `PerformanceThresholds` struct in `crates/bitnet-inference/src/validation.rs` uses hardcoded default values that don't adapt to different hardware configurations, model sizes, or deployment scenarios, leading to inappropriate performance expectations and validation failures.

## Root Cause Analysis

### Current Implementation
```rust
impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            min_tokens_per_second: 10.0,
            max_latency_ms: 5000.0,
            max_memory_usage_mb: 8192.0,
            min_speedup_factor: 1.5, // Expect at least 1.5x speedup over Python
        }
    }
}
```

### Issues Identified
1. **Hardware Agnostic**: Same thresholds for CPU vs GPU vs accelerators
2. **Model Size Ignorant**: Doesn't scale with model complexity
3. **Environment Blind**: No consideration of deployment context
4. **Static Values**: Cannot adapt to runtime conditions

## Proposed Solution

### Adaptive Performance Thresholds

```rust
impl PerformanceThresholds {
    pub fn adaptive(
        device_info: &DeviceInfo,
        model_config: &BitNetConfig,
        deployment_context: &DeploymentContext,
    ) -> Self {
        let base_thresholds = Self::compute_base_thresholds(device_info, model_config);
        let context_adjustments = Self::compute_context_adjustments(deployment_context);

        Self::apply_adjustments(base_thresholds, context_adjustments)
    }

    fn compute_base_thresholds(device_info: &DeviceInfo, model_config: &BitNetConfig) -> Self {
        let device_multiplier = Self::get_device_performance_multiplier(device_info);
        let model_complexity = Self::estimate_model_complexity(model_config);

        Self {
            min_tokens_per_second: Self::compute_min_throughput(device_info, model_complexity),
            max_latency_ms: Self::compute_max_latency(device_info, model_complexity),
            max_memory_usage_mb: Self::compute_memory_limit(device_info, model_config),
            min_speedup_factor: Self::compute_speedup_expectation(device_info),
        }
    }

    fn compute_min_throughput(device_info: &DeviceInfo, complexity: ModelComplexity) -> f64 {
        let base_throughput = match device_info.device_type {
            DeviceType::Cpu(ref cpu_info) => {
                // Scale based on CPU cores and frequency
                let core_factor = cpu_info.physical_cores as f64 * 0.8;
                let freq_factor = cpu_info.base_frequency_ghz;
                2.0 * core_factor * freq_factor
            },
            DeviceType::Gpu(ref gpu_info) => {
                // Scale based on GPU compute capability and memory
                let compute_factor = gpu_info.compute_capability * 10.0;
                let memory_factor = (gpu_info.memory_gb / 8.0).min(4.0);
                compute_factor * memory_factor
            },
            DeviceType::Accelerator(ref acc_info) => {
                // Custom scaling for specialized accelerators
                acc_info.theoretical_tflops * 0.6
            }
        };

        // Adjust for model complexity
        match complexity {
            ModelComplexity::Small => base_throughput * 2.0,
            ModelComplexity::Medium => base_throughput,
            ModelComplexity::Large => base_throughput * 0.4,
            ModelComplexity::ExtraLarge => base_throughput * 0.15,
        }
    }

    fn compute_max_latency(device_info: &DeviceInfo, complexity: ModelComplexity) -> f64 {
        let base_latency = match device_info.device_type {
            DeviceType::Cpu(_) => 2000.0, // 2 seconds base for CPU
            DeviceType::Gpu(_) => 500.0,  // 500ms base for GPU
            DeviceType::Accelerator(_) => 200.0, // 200ms for specialized hardware
        };

        // Scale with model complexity
        let complexity_multiplier = match complexity {
            ModelComplexity::Small => 0.3,
            ModelComplexity::Medium => 1.0,
            ModelComplexity::Large => 3.0,
            ModelComplexity::ExtraLarge => 8.0,
        };

        base_latency * complexity_multiplier
    }

    fn compute_memory_limit(device_info: &DeviceInfo, model_config: &BitNetConfig) -> f64 {
        let available_memory = device_info.get_available_memory_gb() * 1024.0; // Convert to MB
        let model_memory_estimate = Self::estimate_model_memory_mb(model_config);

        // Use 80% of available memory, but at least 2x model size
        let conservative_limit = available_memory * 0.8;
        let minimum_required = model_memory_estimate * 2.0;

        conservative_limit.max(minimum_required)
    }

    pub fn from_environment() -> Self {
        let mut thresholds = Self::default();

        // Override with environment variables if present
        if let Ok(min_tps) = std::env::var("BITNET_MIN_TOKENS_PER_SECOND") {
            if let Ok(value) = min_tps.parse::<f64>() {
                thresholds.min_tokens_per_second = value;
            }
        }

        if let Ok(max_lat) = std::env::var("BITNET_MAX_LATENCY_MS") {
            if let Ok(value) = max_lat.parse::<f64>() {
                thresholds.max_latency_ms = value;
            }
        }

        if let Ok(max_mem) = std::env::var("BITNET_MAX_MEMORY_MB") {
            if let Ok(value) = max_mem.parse::<f64>() {
                thresholds.max_memory_usage_mb = value;
            }
        }

        if let Ok(min_speedup) = std::env::var("BITNET_MIN_SPEEDUP_FACTOR") {
            if let Ok(value) = min_speedup.parse::<f64>() {
                thresholds.min_speedup_factor = value;
            }
        }

        thresholds
    }
}

#[derive(Debug, Clone)]
pub enum ModelComplexity {
    Small,      // < 1B parameters
    Medium,     // 1B - 7B parameters
    Large,      // 7B - 30B parameters
    ExtraLarge, // > 30B parameters
}

#[derive(Debug, Clone)]
pub struct DeploymentContext {
    pub environment_type: EnvironmentType,
    pub performance_priority: PerformancePriority,
    pub resource_constraints: ResourceConstraints,
    pub quality_requirements: QualityRequirements,
}

#[derive(Debug, Clone)]
pub enum EnvironmentType {
    Development,
    Testing,
    Staging,
    Production,
    Edge,
}

#[derive(Debug, Clone)]
pub enum PerformancePriority {
    Latency,    // Optimize for response time
    Throughput, // Optimize for tokens per second
    Memory,     // Optimize for memory usage
    Balanced,   // Balance all factors
}
```

## Implementation Plan

### Phase 1: Hardware Detection (Week 1)
- [ ] Implement comprehensive device capability detection
- [ ] Add model complexity estimation algorithms
- [ ] Create adaptive threshold calculation

### Phase 2: Configuration System (Week 2)
- [ ] Add environment variable support
- [ ] Implement configuration file loading
- [ ] Create deployment context awareness

### Phase 3: Runtime Adaptation (Week 3)
- [ ] Add dynamic threshold adjustment
- [ ] Implement performance feedback loops
- [ ] Create monitoring and alerting

### Phase 4: Integration (Week 4)
- [ ] Update validation pipeline
- [ ] Add comprehensive testing
- [ ] Create documentation and examples

## Acceptance Criteria

- [ ] Hardware-aware threshold calculation
- [ ] Model complexity-based scaling
- [ ] Environment variable configuration support
- [ ] Runtime performance adaptation
- [ ] Comprehensive logging of threshold decisions

## Priority: Medium

Improves system adaptability and reduces false positive validation failures across different deployment scenarios.