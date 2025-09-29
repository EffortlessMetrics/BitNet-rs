# [Models] Implement intelligent device configuration optimization for production model loading

## Problem Description

The `get_optimal_device_config` function in `crates/bitnet-models/src/production_loader.rs` currently returns hardcoded device configurations instead of performing intelligent analysis of model requirements and system capabilities. This limits the system's ability to automatically select optimal device configurations for different deployment scenarios and hardware configurations.

## Environment

- **File:** `crates/bitnet-models/src/production_loader.rs` (lines 403-410, 449-456)
- **Function:** `ProductionModelLoader::get_optimal_device_config` and `MockBitNetModel::get_optimal_device_config`
- **Current Implementation:** Hardcoded `DeviceStrategy::CpuOnly` with fixed parameters
- **Related Components:** Device management, memory estimation, performance optimization

## Current Implementation Analysis

### ProductionModelLoader Implementation (lines 403-410)
```rust
pub fn get_optimal_device_config(&self) -> DeviceConfig {
    DeviceConfig {
        strategy: Some(DeviceStrategy::CpuOnly),
        cpu_threads: Some(4),                    // Hardcoded value
        gpu_memory_fraction: None,               // No GPU consideration
        recommended_batch_size: 1,               // Fixed batch size
    }
}
```

### MockBitNetModel Implementation (lines 449-456)
```rust
pub fn get_optimal_device_config(&self) -> DeviceConfig {
    DeviceConfig {
        strategy: Some(DeviceStrategy::CpuOnly),
        cpu_threads: Some(1),                    // Different hardcoded value
        gpu_memory_fraction: None,
        recommended_batch_size: 1,
    }
}
```

**Issues with Current Implementation:**
1. **No System Analysis**: Doesn't query actual system capabilities
2. **Ignores Model Requirements**: Doesn't consider model size, memory needs, or computational complexity
3. **No GPU Support**: Always defaults to CPU-only strategy
4. **Fixed Parameters**: Hardcoded thread counts and batch sizes
5. **No Performance Optimization**: Misses opportunities for optimal resource utilization

## Root Cause Analysis

1. **Development Priority**: Basic functionality implemented first, optimization deferred
2. **Complexity Avoidance**: System querying and optimization logic requires significant implementation effort
3. **Cross-Platform Challenges**: Device detection varies across operating systems
4. **Testing Constraints**: Automated optimization difficult to test across different hardware configurations

## Impact Assessment

**Severity:** Medium-High
**Component:** Model Loading and Device Management
**Affected Areas:**
- Suboptimal performance on high-end hardware
- Poor resource utilization in production deployments
- Manual configuration burden on users
- Inconsistent performance across different systems

**Performance Impact:**
- GPU-capable systems forced to use CPU inference
- Suboptimal thread utilization on multi-core systems
- Poor batch size selection affecting throughput
- Memory inefficiency from conservative defaults

## Proposed Solution

### 1. System Capability Detection

Implement comprehensive system analysis:

```rust
use std::sync::OnceLock;
use sysinfo::{System, SystemExt, ProcessorExt};

/// System capability analyzer for optimal device selection
#[derive(Debug, Clone)]
pub struct SystemAnalyzer {
    cpu_info: CpuInfo,
    memory_info: MemoryInfo,
    gpu_info: Option<GpuInfo>,
}

#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub core_count: usize,
    pub thread_count: usize,
    pub base_frequency_mhz: Option<u64>,
    pub cache_size_kb: Option<u64>,
    pub supports_avx2: bool,
    pub supports_avx512: bool,
    pub supports_neon: bool,
}

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_gb: f64,
    pub available_gb: f64,
    pub swap_gb: f64,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub device_count: usize,
    pub devices: Vec<GpuDevice>,
}

#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: usize,
    pub name: String,
    pub memory_gb: f64,
    pub compute_capability: Option<(u32, u32)>,
    pub cuda_available: bool,
    pub utilization_percent: Option<f32>,
}

impl SystemAnalyzer {
    pub fn analyze() -> Result<Self> {
        static ANALYZER: OnceLock<SystemAnalyzer> = OnceLock::new();

        Ok(ANALYZER.get_or_init(|| {
            Self::perform_analysis().unwrap_or_else(|e| {
                tracing::warn!("System analysis failed, using defaults: {}", e);
                Self::default()
            })
        }).clone())
    }

    fn perform_analysis() -> Result<Self> {
        let cpu_info = Self::analyze_cpu()?;
        let memory_info = Self::analyze_memory()?;
        let gpu_info = Self::analyze_gpu();

        Ok(Self {
            cpu_info,
            memory_info,
            gpu_info,
        })
    }

    fn analyze_cpu() -> Result<CpuInfo> {
        let mut system = System::new();
        system.refresh_cpu();

        let processors = system.processors();
        let core_count = num_cpus::get_physical();
        let thread_count = num_cpus::get();

        // CPU feature detection
        let supports_avx2 = {
            #[cfg(target_arch = "x86_64")]
            {
                std::arch::is_x86_feature_detected!("avx2")
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        };

        let supports_avx512 = {
            #[cfg(target_arch = "x86_64")]
            {
                std::arch::is_x86_feature_detected!("avx512f")
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        };

        let supports_neon = {
            #[cfg(target_arch = "aarch64")]
            {
                std::arch::is_aarch64_feature_detected!("neon")
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                false
            }
        };

        let base_frequency_mhz = processors
            .first()
            .map(|p| (p.frequency() as f64 / 1000.0) as u64);

        Ok(CpuInfo {
            core_count,
            thread_count,
            base_frequency_mhz,
            cache_size_kb: None, // Could be detected with platform-specific code
            supports_avx2,
            supports_avx512,
            supports_neon,
        })
    }

    fn analyze_memory() -> Result<MemoryInfo> {
        let mut system = System::new();
        system.refresh_memory();

        let total_gb = system.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let available_gb = system.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let swap_gb = system.total_swap() as f64 / (1024.0 * 1024.0 * 1024.0);

        Ok(MemoryInfo {
            total_gb,
            available_gb,
            swap_gb,
        })
    }

    fn analyze_gpu() -> Option<GpuInfo> {
        #[cfg(feature = "gpu")]
        {
            Self::detect_cuda_devices().or_else(|| Self::detect_other_gpu_devices())
        }
        #[cfg(not(feature = "gpu"))]
        {
            None
        }
    }

    #[cfg(feature = "gpu")]
    fn detect_cuda_devices() -> Option<GpuInfo> {
        use candle_core::Device as CDevice;

        let mut devices = Vec::new();

        // Try to detect CUDA devices
        for i in 0..8 {  // Check up to 8 GPUs
            if let Ok(_device) = CDevice::new_cuda(i) {
                // In a real implementation, we'd query device properties
                devices.push(GpuDevice {
                    id: i,
                    name: format!("CUDA Device {}", i),
                    memory_gb: 8.0,  // Placeholder - would query actual memory
                    compute_capability: Some((7, 5)),  // Placeholder
                    cuda_available: true,
                    utilization_percent: None,
                });
            } else {
                break;
            }
        }

        if devices.is_empty() {
            None
        } else {
            Some(GpuInfo {
                device_count: devices.len(),
                devices,
            })
        }
    }

    #[cfg(feature = "gpu")]
    fn detect_other_gpu_devices() -> Option<GpuInfo> {
        // Could detect Metal, OpenCL, or other GPU APIs
        None
    }
}

impl Default for SystemAnalyzer {
    fn default() -> Self {
        Self {
            cpu_info: CpuInfo {
                core_count: 4,
                thread_count: 8,
                base_frequency_mhz: Some(2400),
                cache_size_kb: None,
                supports_avx2: false,
                supports_avx512: false,
                supports_neon: false,
            },
            memory_info: MemoryInfo {
                total_gb: 8.0,
                available_gb: 4.0,
                swap_gb: 0.0,
            },
            gpu_info: None,
        }
    }
}
```

### 2. Model Requirement Analysis

Add intelligent model analysis:

```rust
#[derive(Debug, Clone)]
pub struct ModelRequirements {
    pub model_size_gb: f64,
    pub min_memory_gb: f64,
    pub optimal_memory_gb: f64,
    pub compute_intensity: ComputeIntensity,
    pub supports_quantization: QuantizationSupport,
    pub parallelism_potential: ParallelismPotential,
}

#[derive(Debug, Clone, Copy)]
pub enum ComputeIntensity {
    Low,     // Simple models, limited compute
    Medium,  // Standard transformer models
    High,    // Large models with complex operations
}

#[derive(Debug, Clone)]
pub struct QuantizationSupport {
    pub i2s_supported: bool,
    pub tl1_supported: bool,
    pub tl2_supported: bool,
    pub fp16_supported: bool,
}

#[derive(Debug, Clone)]
pub struct ParallelismPotential {
    pub layer_parallel: bool,
    pub tensor_parallel: bool,
    pub data_parallel: bool,
    pub optimal_thread_count: usize,
}

impl ProductionModelLoader {
    fn analyze_model_requirements(&self) -> Result<ModelRequirements> {
        let config = self.base_loader.get_model_config();

        // Calculate model size based on parameters
        let param_count = self.estimate_parameter_count(&config);
        let model_size_gb = self.calculate_model_size_gb(param_count);

        // Memory requirements with overhead
        let min_memory_gb = model_size_gb * 1.2;  // 20% overhead
        let optimal_memory_gb = model_size_gb * 2.0;  // 100% overhead for activations

        // Analyze compute intensity
        let compute_intensity = if param_count > 7_000_000_000 {
            ComputeIntensity::High
        } else if param_count > 1_000_000_000 {
            ComputeIntensity::Medium
        } else {
            ComputeIntensity::Low
        };

        // Quantization support analysis
        let supports_quantization = QuantizationSupport {
            i2s_supported: true,  // BitNet always supports I2S
            tl1_supported: true,
            tl2_supported: true,
            fp16_supported: true,
        };

        // Parallelism analysis
        let parallelism_potential = ParallelismPotential {
            layer_parallel: config.model.num_layers > 4,
            tensor_parallel: config.model.hidden_size > 1024,
            data_parallel: true,
            optimal_thread_count: (config.model.num_layers / 2).max(1).min(16),
        };

        Ok(ModelRequirements {
            model_size_gb,
            min_memory_gb,
            optimal_memory_gb,
            compute_intensity,
            supports_quantization,
            parallelism_potential,
        })
    }

    fn estimate_parameter_count(&self, config: &bitnet_common::BitNetConfig) -> u64 {
        let hidden_size = config.model.hidden_size as u64;
        let num_layers = config.model.num_layers as u64;
        let vocab_size = config.model.vocab_size as u64;
        let intermediate_size = config.model.intermediate_size as u64;

        // Embeddings
        let embedding_params = vocab_size * hidden_size;

        // Transformer layers
        let attention_params_per_layer = hidden_size * hidden_size * 4; // Q, K, V, O
        let ffn_params_per_layer = hidden_size * intermediate_size * 3; // Gate, Up, Down
        let layer_params = (attention_params_per_layer + ffn_params_per_layer) * num_layers;

        // Output projection
        let output_params = hidden_size * vocab_size;

        embedding_params + layer_params + output_params
    }

    fn calculate_model_size_gb(&self, param_count: u64) -> f64 {
        // For 1-bit quantization, approximate size calculation
        let bits_per_param = 1.5; // Account for 1-bit weights + metadata overhead
        let bytes_per_param = bits_per_param / 8.0;
        let total_bytes = param_count as f64 * bytes_per_param;
        total_bytes / (1024.0 * 1024.0 * 1024.0)
    }
}
```

### 3. Intelligent Device Configuration Optimization

Implement the core optimization logic:

```rust
#[derive(Debug, Clone)]
pub struct DeviceOptimizer {
    system: SystemAnalyzer,
    preferences: OptimizationPreferences,
}

#[derive(Debug, Clone)]
pub struct OptimizationPreferences {
    pub prefer_gpu: bool,
    pub memory_safety_margin: f64,  // 0.1 = 10% safety margin
    pub performance_priority: PerformancePriority,
    pub power_efficiency: PowerEfficiency,
}

#[derive(Debug, Clone, Copy)]
pub enum PerformancePriority {
    Latency,    // Optimize for low latency
    Throughput, // Optimize for high throughput
    Balanced,   // Balance latency and throughput
}

#[derive(Debug, Clone, Copy)]
pub enum PowerEfficiency {
    HighPerformance,  // Maximum performance, ignore power
    Balanced,         // Balance performance and power
    PowerSaver,       // Minimize power consumption
}

impl DeviceOptimizer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            system: SystemAnalyzer::analyze()?,
            preferences: OptimizationPreferences::default(),
        })
    }

    pub fn with_preferences(mut self, preferences: OptimizationPreferences) -> Self {
        self.preferences = preferences;
        self
    }

    pub fn optimize_for_model(&self, requirements: &ModelRequirements) -> DeviceConfig {
        // Strategy selection
        let strategy = self.select_optimal_strategy(requirements);

        // Thread optimization
        let cpu_threads = self.optimize_cpu_threads(requirements, &strategy);

        // GPU memory optimization
        let gpu_memory_fraction = self.optimize_gpu_memory(requirements, &strategy);

        // Batch size optimization
        let recommended_batch_size = self.optimize_batch_size(requirements, &strategy);

        DeviceConfig {
            strategy: Some(strategy),
            cpu_threads,
            gpu_memory_fraction,
            recommended_batch_size,
        }
    }

    fn select_optimal_strategy(&self, requirements: &ModelRequirements) -> DeviceStrategy {
        // Check if GPU is available and beneficial
        if let Some(gpu_info) = &self.system.gpu_info {
            if self.preferences.prefer_gpu && !gpu_info.devices.is_empty() {
                let best_gpu = &gpu_info.devices[0]; // Use first/best GPU

                // Check if GPU has sufficient memory
                if best_gpu.memory_gb >= requirements.min_memory_gb * (1.0 + self.preferences.memory_safety_margin) {
                    // GPU can handle the entire model
                    return DeviceStrategy::GpuOnly;
                } else if self.system.memory_info.available_gb >= requirements.min_memory_gb {
                    // Hybrid approach: some layers on GPU, some on CPU
                    let gpu_layers = self.calculate_gpu_layers(requirements, best_gpu);
                    let cpu_layers = requirements.parallelism_potential.optimal_thread_count.saturating_sub(gpu_layers);

                    if gpu_layers > 0 {
                        return DeviceStrategy::Hybrid { cpu_layers, gpu_layers };
                    }
                }
            }
        }

        // Fallback to CPU-only
        DeviceStrategy::CpuOnly
    }

    fn optimize_cpu_threads(&self, requirements: &ModelRequirements, strategy: &DeviceStrategy) -> Option<usize> {
        match strategy {
            DeviceStrategy::CpuOnly => {
                let optimal_threads = match self.preferences.performance_priority {
                    PerformancePriority::Latency => {
                        // For latency, use fewer threads to reduce synchronization overhead
                        (self.system.cpu_info.core_count / 2).max(1).min(8)
                    }
                    PerformancePriority::Throughput => {
                        // For throughput, use more threads
                        self.system.cpu_info.thread_count.min(requirements.parallelism_potential.optimal_thread_count)
                    }
                    PerformancePriority::Balanced => {
                        // Balanced approach
                        requirements.parallelism_potential.optimal_thread_count
                            .min(self.system.cpu_info.core_count)
                    }
                };

                Some(optimal_threads)
            }
            DeviceStrategy::Hybrid { cpu_layers, .. } => {
                // For hybrid, optimize threads for CPU portion
                Some((*cpu_layers).min(self.system.cpu_info.core_count))
            }
            DeviceStrategy::GpuOnly => {
                // Still may need some CPU threads for coordination
                Some(2)
            }
        }
    }

    fn optimize_gpu_memory(&self, requirements: &ModelRequirements, strategy: &DeviceStrategy) -> Option<f32> {
        match strategy {
            DeviceStrategy::GpuOnly | DeviceStrategy::Hybrid { .. } => {
                if let Some(gpu_info) = &self.system.gpu_info {
                    let gpu = &gpu_info.devices[0];
                    let required_fraction = requirements.min_memory_gb / gpu.memory_gb;
                    let safe_fraction = required_fraction * (1.0 + self.preferences.memory_safety_margin);

                    Some((safe_fraction as f32).min(0.95)) // Never use more than 95%
                } else {
                    None
                }
            }
            DeviceStrategy::CpuOnly => None,
        }
    }

    fn optimize_batch_size(&self, requirements: &ModelRequirements, strategy: &DeviceStrategy) -> usize {
        let base_batch_size = match requirements.compute_intensity {
            ComputeIntensity::Low => 4,
            ComputeIntensity::Medium => 2,
            ComputeIntensity::High => 1,
        };

        let memory_factor = match strategy {
            DeviceStrategy::GpuOnly => {
                if let Some(gpu_info) = &self.system.gpu_info {
                    let gpu = &gpu_info.devices[0];
                    if gpu.memory_gb > requirements.optimal_memory_gb * 2.0 {
                        2.0 // Can afford larger batches
                    } else {
                        1.0
                    }
                } else {
                    1.0
                }
            }
            DeviceStrategy::CpuOnly => {
                if self.system.memory_info.available_gb > requirements.optimal_memory_gb * 2.0 {
                    1.5
                } else {
                    1.0
                }
            }
            DeviceStrategy::Hybrid { .. } => 1.0, // Conservative for hybrid
        };

        let performance_factor = match self.preferences.performance_priority {
            PerformancePriority::Latency => 0.5, // Smaller batches for lower latency
            PerformancePriority::Throughput => 2.0, // Larger batches for throughput
            PerformancePriority::Balanced => 1.0,
        };

        ((base_batch_size as f64 * memory_factor * performance_factor) as usize).max(1)
    }

    fn calculate_gpu_layers(&self, requirements: &ModelRequirements, gpu: &GpuDevice) -> usize {
        let total_layers = requirements.parallelism_potential.optimal_thread_count;
        let gpu_memory_ratio = (gpu.memory_gb / requirements.min_memory_gb).min(1.0);

        (total_layers as f64 * gpu_memory_ratio) as usize
    }
}

impl Default for OptimizationPreferences {
    fn default() -> Self {
        Self {
            prefer_gpu: true,
            memory_safety_margin: 0.1,
            performance_priority: PerformancePriority::Balanced,
            power_efficiency: PowerEfficiency::Balanced,
        }
    }
}
```

### 4. Enhanced get_optimal_device_config Implementation

Update the main function:

```rust
impl ProductionModelLoader {
    /// Get optimal device configuration for the model
    pub fn get_optimal_device_config(&self) -> DeviceConfig {
        self.get_optimal_device_config_with_preferences(OptimizationPreferences::default())
    }

    /// Get optimal device configuration with custom preferences
    pub fn get_optimal_device_config_with_preferences(
        &self,
        preferences: OptimizationPreferences,
    ) -> DeviceConfig {
        // Attempt intelligent optimization
        match self.perform_intelligent_optimization(preferences) {
            Ok(config) => {
                tracing::info!(
                    "Intelligent device optimization successful: strategy={:?}, threads={:?}, batch_size={}",
                    config.strategy,
                    config.cpu_threads,
                    config.recommended_batch_size
                );
                config
            }
            Err(e) => {
                tracing::warn!(
                    "Intelligent optimization failed ({}), using conservative defaults",
                    e
                );
                self.get_conservative_fallback_config()
            }
        }
    }

    fn perform_intelligent_optimization(
        &self,
        preferences: OptimizationPreferences,
    ) -> Result<DeviceConfig> {
        // Analyze model requirements
        let requirements = self.analyze_model_requirements()?;

        // Create optimizer
        let optimizer = DeviceOptimizer::new()?.with_preferences(preferences);

        // Optimize configuration
        let config = optimizer.optimize_for_model(&requirements);

        // Validate the configuration
        self.validate_device_config(&config)?;

        Ok(config)
    }

    fn get_conservative_fallback_config(&self) -> DeviceConfig {
        DeviceConfig {
            strategy: Some(DeviceStrategy::CpuOnly),
            cpu_threads: Some(num_cpus::get().min(4)),  // Conservative thread count
            gpu_memory_fraction: None,
            recommended_batch_size: 1,
        }
    }

    fn validate_device_config(&self, config: &DeviceConfig) -> Result<()> {
        // Validate thread count
        if let Some(threads) = config.cpu_threads {
            if threads == 0 {
                return Err(BitNetError::Validation(
                    "Invalid device config: CPU thread count cannot be zero".to_string()
                ));
            }
            if threads > num_cpus::get() * 2 {
                tracing::warn!(
                    "High CPU thread count ({}), may cause contention",
                    threads
                );
            }
        }

        // Validate batch size
        if config.recommended_batch_size == 0 {
            return Err(BitNetError::Validation(
                "Invalid device config: batch size cannot be zero".to_string()
            ));
        }

        // Validate GPU memory fraction
        if let Some(fraction) = config.gpu_memory_fraction {
            if fraction <= 0.0 || fraction > 1.0 {
                return Err(BitNetError::Validation(
                    format!("Invalid GPU memory fraction: {}", fraction)
                ));
            }
        }

        Ok(())
    }
}
```

## Implementation Plan

### Phase 1: System Analysis Infrastructure
- [ ] Implement `SystemAnalyzer` with CPU, memory, and GPU detection
- [ ] Add cross-platform capability detection (AVX2, AVX-512, NEON)
- [ ] Create `GpuInfo` detection for CUDA devices
- [ ] Add comprehensive error handling and fallbacks

### Phase 2: Model Requirement Analysis
- [ ] Implement `ModelRequirements` analysis from BitNet config
- [ ] Add parameter count estimation and memory calculation
- [ ] Create compute intensity classification
- [ ] Add parallelism potential analysis

### Phase 3: Device Optimization Engine
- [ ] Implement `DeviceOptimizer` with strategy selection
- [ ] Add CPU thread optimization logic
- [ ] Implement GPU memory fraction optimization
- [ ] Create batch size optimization based on strategy and memory

### Phase 4: Integration and Configuration
- [ ] Update `get_optimal_device_config` with intelligent optimization
- [ ] Add `OptimizationPreferences` for customizable behavior
- [ ] Implement validation and conservative fallbacks
- [ ] Add comprehensive logging and debugging information

### Phase 5: Testing and Validation
- [ ] Unit tests for each optimization component
- [ ] Integration tests across different hardware configurations
- [ ] Performance benchmarking and validation
- [ ] Documentation and usage examples

## Testing Strategy

### Unit Tests
```bash
# Test system analysis
cargo test --package bitnet-models system_analyzer

# Test model requirement analysis
cargo test --package bitnet-models model_requirements

# Test device optimization
cargo test --package bitnet-models device_optimizer

# Test config validation
cargo test --package bitnet-models device_config_validation
```

### Integration Tests
```bash
# Test across different hardware configurations
cargo test --package bitnet-models --test hardware_optimization

# Test fallback behavior
cargo test --package bitnet-models --test optimization_fallbacks
```

### Performance Tests
```bash
# Benchmark optimization overhead
cargo bench --package bitnet-models device_optimization_performance
```

## Dependencies Required

Add to `Cargo.toml`:
```toml
[dependencies]
sysinfo = "0.30"
num_cpus = "1.16"

[target.'cfg(feature = "gpu")'.dependencies]
# GPU detection libraries as needed
```

## Success Criteria

1. **Intelligent Detection**: Accurate system capability detection across platforms
2. **Optimal Performance**: Measurable performance improvements over hardcoded configs
3. **Robust Fallbacks**: Graceful degradation when optimization fails
4. **Resource Efficiency**: Better memory and compute resource utilization
5. **User Experience**: Automatic optimal configuration without manual tuning

## Related Issues

- System requirement validation implementation
- GPU backend optimization
- Performance monitoring and metrics
- Production deployment configuration

---

**Labels:** `models`, `optimization`, `performance`, `enhancement`
**Assignee:** Device Management Team
**Epic:** Intelligent System Optimization