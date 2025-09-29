# [Models] Implement Intelligent Device Configuration Optimization in Production Loader

## Problem Description

The `get_optimal_device_config` method in `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/production_loader.rs:403` returns hardcoded device configurations instead of intelligently analyzing model requirements and system capabilities to determine optimal device placement strategies. This limits performance optimization and prevents automatic adaptation to different hardware environments.

## Environment

- **File**: `crates/bitnet-models/src/production_loader.rs`
- **Method**: `ProductionModelLoader::get_optimal_device_config` (line 403)
- **MSRV**: Rust 1.90.0
- **Feature flags**: Both `cpu` and `gpu` features need intelligent selection
- **Dependencies**: `bitnet-common`, potential system info crates

## Current Implementation Analysis

### Existing Code
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

### Available Infrastructure
```rust
pub struct DeviceConfig {
    pub strategy: Option<DeviceStrategy>,
    pub cpu_threads: Option<usize>,
    pub gpu_memory_fraction: Option<f32>,
    pub recommended_batch_size: usize,
}

pub enum DeviceStrategy {
    CpuOnly,
    GpuOnly,
    Hybrid { cpu_layers: usize, gpu_layers: usize },
}
```

### Gap Analysis

**Missing Capabilities:**
1. **System Analysis**: No hardware capability detection
2. **Model Analysis**: No consideration of model size, quantization method, or computational requirements
3. **Performance Modeling**: No estimation of relative performance across devices
4. **Memory Planning**: No intelligent memory allocation across CPU/GPU
5. **Adaptive Configuration**: No runtime adaptation based on available resources

## Root Cause Analysis

1. **Placeholder Implementation**: Method was stubbed during initial development
2. **Missing System Integration**: No connection to hardware detection systems
3. **Lack of Performance Models**: No data-driven optimization decisions
4. **Incomplete Model Analysis**: Not utilizing model metadata for optimization

## Impact Assessment

### Severity: Medium-High
### Affected Components: Model loading performance, production deployment efficiency

**Performance Impact:**
- Suboptimal device utilization leading to 2-10x performance loss
- Inefficient memory allocation causing OOM failures or underutilization
- Manual tuning required for each deployment environment
- No automatic scaling across different hardware configurations

**Operational Impact:**
- Increased deployment complexity requiring manual configuration
- Inconsistent performance across different hardware environments
- Higher resource costs due to suboptimal utilization
- Reduced user experience with default configurations

## Proposed Solution

### Primary Approach: Intelligent Device Configuration System

Implement a comprehensive device optimization system that analyzes model requirements, system capabilities, and performance characteristics to automatically select optimal configurations.

#### Implementation Plan

**1. Hardware Detection and Analysis**

```rust
use sysinfo::{System, SystemExt, CpuExt};
use std::sync::OnceLock;

/// System hardware information cache
static SYSTEM_INFO: OnceLock<SystemInfo> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub cpu_info: CpuInfo,
    pub memory_info: MemoryInfo,
    pub gpu_info: Option<GpuInfo>,
    pub performance_tier: PerformanceTier,
}

#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub frequency_mhz: u64,
    pub simd_capabilities: SimdCapabilities,
    pub performance_score: f64,
}

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_gb: f64,
    pub available_gb: f64,
    pub bandwidth_gbps: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub available: bool,
    pub memory_gb: f64,
    pub compute_capability: Option<(u32, u32)>,
    pub performance_score: f64,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
}

#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub avx2: bool,
    pub avx512: bool,
    pub neon: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PerformanceTier {
    Low,      // Basic CPU, limited memory
    Medium,   // Modern CPU or basic GPU
    High,     // High-end CPU or mid-range GPU
    Extreme,  // High-end GPU or server hardware
}

impl ProductionModelLoader {
    /// Get cached system information
    fn get_system_info(&self) -> &SystemInfo {
        SYSTEM_INFO.get_or_init(|| {
            SystemInfo::detect()
        })
    }
}

impl SystemInfo {
    pub fn detect() -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        let cpu_info = CpuInfo::detect(&system);
        let memory_info = MemoryInfo::detect(&system);
        let gpu_info = GpuInfo::detect();

        let performance_tier = Self::calculate_performance_tier(&cpu_info, &memory_info, &gpu_info);

        Self {
            cpu_info,
            memory_info,
            gpu_info,
            performance_tier,
        }
    }

    fn calculate_performance_tier(
        cpu: &CpuInfo,
        memory: &MemoryInfo,
        gpu: &Option<GpuInfo>,
    ) -> PerformanceTier {
        // GPU-first classification
        if let Some(gpu) = gpu {
            if gpu.available && gpu.performance_score > 8.0 {
                return PerformanceTier::Extreme;
            } else if gpu.available && gpu.performance_score > 4.0 {
                return PerformanceTier::High;
            }
        }

        // CPU/Memory classification
        if cpu.performance_score > 15.0 && memory.total_gb > 32.0 {
            PerformanceTier::High
        } else if cpu.performance_score > 8.0 && memory.total_gb > 16.0 {
            PerformanceTier::Medium
        } else {
            PerformanceTier::Low
        }
    }
}

impl CpuInfo {
    fn detect(system: &System) -> Self {
        let logical_cores = system.cpus().len();
        let physical_cores = logical_cores / 2; // Approximation
        let frequency_mhz = system.cpus().first()
            .map(|cpu| cpu.frequency())
            .unwrap_or(2000);

        let simd_capabilities = SimdCapabilities::detect();
        let performance_score = Self::calculate_performance_score(
            logical_cores,
            frequency_mhz,
            &simd_capabilities,
        );

        Self {
            logical_cores,
            physical_cores,
            frequency_mhz,
            simd_capabilities,
            performance_score,
        }
    }

    fn calculate_performance_score(cores: usize, freq_mhz: u64, simd: &SimdCapabilities) -> f64 {
        let base_score = cores as f64 * (freq_mhz as f64 / 1000.0);
        let simd_multiplier = if simd.avx512 {
            2.0
        } else if simd.avx2 {
            1.5
        } else if simd.neon {
            1.3
        } else {
            1.0
        };

        base_score * simd_multiplier
    }
}

impl SimdCapabilities {
    fn detect() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            avx2: is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            avx512: is_x86_feature_detected!("avx512f"),
            #[cfg(not(target_arch = "x86_64"))]
            avx2: false,
            #[cfg(not(target_arch = "x86_64"))]
            avx512: false,

            #[cfg(target_arch = "aarch64")]
            neon: true, // Standard on ARM64
            #[cfg(not(target_arch = "aarch64"))]
            neon: false,
        }
    }
}

impl MemoryInfo {
    fn detect(system: &System) -> Self {
        let total_gb = system.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let available_gb = system.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

        Self {
            total_gb,
            available_gb,
            bandwidth_gbps: None, // Could be detected with more sophisticated tools
        }
    }
}

#[cfg(feature = "gpu")]
impl GpuInfo {
    fn detect() -> Option<Self> {
        // Use cudarc or similar for GPU detection
        match Self::detect_cuda() {
            Ok(info) => Some(info),
            Err(_) => None,
        }
    }

    fn detect_cuda() -> Result<Self, Box<dyn std::error::Error>> {
        let device_count = cudarc::driver::result::device::get_device_count()?;

        if device_count == 0 {
            return Err("No CUDA devices found".into());
        }

        let device = cudarc::driver::CudaDevice::new(0)?;
        let memory_info = device.memory_info()?;
        let (major, minor) = device.compute_capability()?;

        let memory_gb = memory_info.total as f64 / (1024.0 * 1024.0 * 1024.0);
        let performance_score = Self::calculate_gpu_performance_score(major, minor, memory_gb);

        Ok(Self {
            available: true,
            memory_gb,
            compute_capability: Some((major, minor)),
            performance_score,
            supports_fp16: major >= 6, // CC 6.0+ for efficient FP16
            supports_bf16: major >= 8, // CC 8.0+ for BF16
        })
    }

    fn calculate_gpu_performance_score(major: u32, minor: u32, memory_gb: f64) -> f64 {
        let cc_score = match major {
            9.. => 10.0,        // Ada Lovelace, Hopper
            8 => 8.0,           // Ampere
            7 => 6.0,           // Turing, Volta
            6 => 4.0,           // Pascal
            _ => 2.0,           // Older architectures
        };

        let memory_score = (memory_gb / 4.0).min(4.0); // Normalize to 16GB
        (cc_score + memory_score) / 2.0
    }
}

#[cfg(not(feature = "gpu"))]
impl GpuInfo {
    fn detect() -> Option<Self> {
        None
    }
}
```

**2. Model Analysis and Requirements**

```rust
#[derive(Debug, Clone)]
pub struct ModelRequirements {
    pub memory_requirements: ModelMemoryRequirements,
    pub compute_requirements: ComputeRequirements,
    pub quantization_method: QuantizationMethod,
    pub optimization_hints: OptimizationHints,
}

#[derive(Debug, Clone)]
pub struct ModelMemoryRequirements {
    pub base_model_mb: u64,
    pub kv_cache_mb_per_token: f64,
    pub activation_memory_mb: u64,
    pub quantization_overhead_mb: u64,
    pub total_min_mb: u64,
    pub total_recommended_mb: u64,
}

#[derive(Debug, Clone)]
pub struct ComputeRequirements {
    pub operations_per_token: u64,
    pub matrix_dimensions: Vec<(usize, usize)>,
    pub requires_fp16: bool,
    pub requires_int8: bool,
    pub supports_batching: bool,
    pub parallel_layers: usize,
}

#[derive(Debug, Clone)]
pub struct OptimizationHints {
    pub prefers_gpu: bool,
    pub memory_bound: bool,
    pub compute_bound: bool,
    pub cache_friendly: bool,
}

impl ProductionModelLoader {
    fn analyze_model_requirements(&self) -> Result<ModelRequirements> {
        let config = self.base_loader.get_model_config();

        let memory_requirements = self.calculate_memory_requirements(&config)?;
        let compute_requirements = self.analyze_compute_requirements(&config)?;
        let quantization_method = self.detect_quantization_method()?;
        let optimization_hints = self.generate_optimization_hints(&config, &memory_requirements)?;

        Ok(ModelRequirements {
            memory_requirements,
            compute_requirements,
            quantization_method,
            optimization_hints,
        })
    }

    fn calculate_memory_requirements(&self, config: &ModelConfig) -> Result<ModelMemoryRequirements> {
        // Parse model configuration to estimate memory usage
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;
        let max_context = config.max_position_embeddings;

        // Estimate model parameters
        let embedding_params = vocab_size * hidden_size;
        let layer_params = num_layers * (
            hidden_size * hidden_size * 4 + // Attention weights
            hidden_size * config.intermediate_size * 2 + // FFN weights
            hidden_size * 4 // Layer norms and biases
        );
        let total_params = embedding_params + layer_params;

        // Calculate memory based on quantization
        let bytes_per_param = match self.detect_quantization_method()? {
            QuantizationMethod::I2S => 0.25,
            QuantizationMethod::TL1 | QuantizationMethod::TL2 => 0.125,
            QuantizationMethod::IQ2S => 0.25,
            QuantizationMethod::None => 4.0,
        };

        let base_model_mb = (total_params as f64 * bytes_per_param / (1024.0 * 1024.0)) as u64;

        // KV cache estimation (per token)
        let kv_cache_mb_per_token = (hidden_size * num_layers * 2 * 2) as f64 / (1024.0 * 1024.0); // FP16

        // Activation memory (batch size 1)
        let activation_memory_mb = (max_context * hidden_size * 4) as u64 / (1024 * 1024); // FP32 activations

        // Quantization overhead
        let quantization_overhead_mb = base_model_mb / 10; // 10% overhead estimate

        let total_min_mb = base_model_mb + activation_memory_mb;
        let total_recommended_mb = total_min_mb + quantization_overhead_mb +
                                 (kv_cache_mb_per_token * max_context as f64) as u64;

        Ok(ModelMemoryRequirements {
            base_model_mb,
            kv_cache_mb_per_token,
            activation_memory_mb,
            quantization_overhead_mb,
            total_min_mb,
            total_recommended_mb,
        })
    }

    fn analyze_compute_requirements(&self, config: &ModelConfig) -> Result<ComputeRequirements> {
        let seq_len = config.max_position_embeddings;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;
        let vocab_size = config.vocab_size;

        // Estimate operations per token (forward pass)
        let attention_ops = num_layers * seq_len * hidden_size * hidden_size * 4; // Q, K, V, O projections
        let ffn_ops = num_layers * seq_len * hidden_size * config.intermediate_size * 2; // Up and down
        let embedding_ops = seq_len * hidden_size * vocab_size; // Final projection
        let operations_per_token = (attention_ops + ffn_ops + embedding_ops) / seq_len;

        // Determine key matrix dimensions for optimization
        let matrix_dimensions = vec![
            (hidden_size, hidden_size), // Attention projections
            (hidden_size, config.intermediate_size), // FFN up
            (config.intermediate_size, hidden_size), // FFN down
            (hidden_size, vocab_size), // Output projection
        ];

        Ok(ComputeRequirements {
            operations_per_token: operations_per_token as u64,
            matrix_dimensions,
            requires_fp16: true, // BitNet typically benefits from FP16
            requires_int8: false, // Use quantized weights
            supports_batching: true,
            parallel_layers: num_layers,
        })
    }

    fn generate_optimization_hints(&self, config: &ModelConfig, memory: &ModelMemoryRequirements) -> Result<OptimizationHints> {
        let large_model = memory.base_model_mb > 1000; // > 1GB
        let memory_intensive = memory.kv_cache_mb_per_token > 0.1; // Large cache requirements
        let compute_intensive = config.num_layers > 24; // Deep model

        Ok(OptimizationHints {
            prefers_gpu: large_model || compute_intensive,
            memory_bound: memory_intensive,
            compute_bound: compute_intensive && !memory_intensive,
            cache_friendly: config.max_position_embeddings <= 2048,
        })
    }
}
```

**3. Intelligent Device Configuration Selection**

```rust
impl ProductionModelLoader {
    pub fn get_optimal_device_config(&self) -> DeviceConfig {
        match self.calculate_optimal_config() {
            Ok(config) => {
                info!("Optimal device configuration selected: {:?}", config);
                config
            }
            Err(e) => {
                warn!("Failed to calculate optimal config, using fallback: {}", e);
                self.get_fallback_config()
            }
        }
    }

    fn calculate_optimal_config(&self) -> Result<DeviceConfig> {
        let system_info = self.get_system_info();
        let model_reqs = self.analyze_model_requirements()?;

        // Memory feasibility check
        let memory_check = self.check_memory_feasibility(system_info, &model_reqs)?;

        // Performance estimation
        let cpu_performance = self.estimate_cpu_performance(system_info, &model_reqs);
        let gpu_performance = self.estimate_gpu_performance(system_info, &model_reqs);

        // Strategy selection
        let strategy = self.select_optimal_strategy(
            system_info,
            &model_reqs,
            &memory_check,
            cpu_performance,
            gpu_performance,
        )?;

        // Configuration optimization
        let config = self.optimize_configuration(system_info, &model_reqs, strategy)?;

        Ok(config)
    }

    fn check_memory_feasibility(
        &self,
        system: &SystemInfo,
        model: &ModelRequirements,
    ) -> Result<MemoryFeasibility> {
        let available_cpu_mb = (system.memory_info.available_gb * 0.8 * 1024.0) as u64; // 80% safety margin
        let available_gpu_mb = system.gpu_info.as_ref()
            .map(|gpu| (gpu.memory_gb * 0.9 * 1024.0) as u64) // 90% safety margin
            .unwrap_or(0);

        Ok(MemoryFeasibility {
            cpu_feasible: available_cpu_mb >= model.memory_requirements.total_min_mb,
            gpu_feasible: available_gpu_mb >= model.memory_requirements.total_min_mb,
            hybrid_feasible: available_cpu_mb + available_gpu_mb >= model.memory_requirements.total_recommended_mb,
            recommended_split: if available_gpu_mb > 0 {
                Some(available_gpu_mb as f32 / (available_cpu_mb + available_gpu_mb) as f32)
            } else {
                None
            },
        })
    }

    fn estimate_cpu_performance(&self, system: &SystemInfo, model: &ModelRequirements) -> f64 {
        let base_performance = system.cpu_info.performance_score;

        // Quantization performance bonus
        let quant_multiplier = match model.quantization_method {
            QuantizationMethod::I2S => 1.5,   // Good SIMD optimization
            QuantizationMethod::TL1 => 2.0,   // Excellent lookup performance
            QuantizationMethod::TL2 => 2.0,   // Excellent lookup performance
            QuantizationMethod::IQ2S => 1.2,  // Limited optimization
            QuantizationMethod::None => 1.0,  // No quantization benefit
        };

        // SIMD capability bonus
        let simd_multiplier = if system.cpu_info.simd_capabilities.avx512 {
            1.8
        } else if system.cpu_info.simd_capabilities.avx2 {
            1.4
        } else {
            1.0
        };

        base_performance * quant_multiplier * simd_multiplier
    }

    fn estimate_gpu_performance(&self, system: &SystemInfo, model: &ModelRequirements) -> Option<f64> {
        let gpu_info = system.gpu_info.as_ref()?;

        if !gpu_info.available {
            return None;
        }

        let base_performance = gpu_info.performance_score;

        // Memory bandwidth advantage
        let memory_multiplier = if model.optimization_hints.memory_bound {
            2.0 // GPU has much higher memory bandwidth
        } else {
            1.0
        };

        // Compute advantage
        let compute_multiplier = if model.optimization_hints.compute_bound {
            5.0 // GPU excels at parallel compute
        } else {
            1.5 // Still faster for matrix operations
        };

        // Precision support
        let precision_multiplier = if model.compute_requirements.requires_fp16 && gpu_info.supports_fp16 {
            1.5 // Efficient FP16 support
        } else {
            1.0
        };

        Some(base_performance * memory_multiplier * compute_multiplier * precision_multiplier)
    }

    fn select_optimal_strategy(
        &self,
        system: &SystemInfo,
        model: &ModelRequirements,
        memory: &MemoryFeasibility,
        cpu_perf: f64,
        gpu_perf: Option<f64>,
    ) -> Result<DeviceStrategy> {
        // GPU-first strategy if available and feasible
        if let Some(gpu_perf) = gpu_perf {
            if memory.gpu_feasible && gpu_perf > cpu_perf * 1.5 {
                return Ok(DeviceStrategy::GpuOnly);
            }

            // Hybrid strategy for large models
            if memory.hybrid_feasible && model.memory_requirements.base_model_mb > 2000 {
                let gpu_layers = self.calculate_optimal_gpu_layers(system, model, memory)?;
                let cpu_layers = model.compute_requirements.parallel_layers - gpu_layers;

                if gpu_layers > 0 && cpu_layers > 0 {
                    return Ok(DeviceStrategy::Hybrid { cpu_layers, gpu_layers });
                }
            }
        }

        // CPU fallback
        if memory.cpu_feasible {
            Ok(DeviceStrategy::CpuOnly)
        } else {
            Err(anyhow::anyhow!("Model too large for available memory"))
        }
    }

    fn calculate_optimal_gpu_layers(&self, system: &SystemInfo, model: &ModelRequirements, memory: &MemoryFeasibility) -> Result<usize> {
        let gpu_info = system.gpu_info.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No GPU available"))?;

        let available_gpu_mb = (gpu_info.memory_gb * 0.9 * 1024.0) as u64;
        let layer_size_mb = model.memory_requirements.base_model_mb / model.compute_requirements.parallel_layers as u64;

        // Reserve memory for activations and overhead
        let reserved_mb = model.memory_requirements.activation_memory_mb + 200; // 200MB overhead
        let usable_gpu_mb = available_gpu_mb.saturating_sub(reserved_mb);

        let max_gpu_layers = (usable_gpu_mb / layer_size_mb).min(model.compute_requirements.parallel_layers as u64);

        Ok(max_gpu_layers as usize)
    }

    fn optimize_configuration(
        &self,
        system: &SystemInfo,
        model: &ModelRequirements,
        strategy: DeviceStrategy,
    ) -> Result<DeviceConfig> {
        let cpu_threads = match strategy {
            DeviceStrategy::CpuOnly => Some(system.cpu_info.logical_cores),
            DeviceStrategy::GpuOnly => Some(2), // Minimal CPU threads for coordination
            DeviceStrategy::Hybrid { .. } => Some(system.cpu_info.logical_cores / 2),
        };

        let gpu_memory_fraction = match strategy {
            DeviceStrategy::GpuOnly => Some(0.9),
            DeviceStrategy::Hybrid { .. } => Some(0.8), // Leave room for CPU coordination
            DeviceStrategy::CpuOnly => None,
        };

        let recommended_batch_size = self.calculate_optimal_batch_size(system, model, &strategy);

        Ok(DeviceConfig {
            strategy: Some(strategy),
            cpu_threads,
            gpu_memory_fraction,
            recommended_batch_size,
        })
    }

    fn calculate_optimal_batch_size(&self, system: &SystemInfo, model: &ModelRequirements, strategy: &DeviceStrategy) -> usize {
        let base_batch_size = match system.performance_tier {
            PerformanceTier::Extreme => 8,
            PerformanceTier::High => 4,
            PerformanceTier::Medium => 2,
            PerformanceTier::Low => 1,
        };

        // Adjust for memory constraints
        let memory_limited_batch = {
            let available_mb = match strategy {
                DeviceStrategy::GpuOnly => {
                    system.gpu_info.as_ref()
                        .map(|gpu| (gpu.memory_gb * 0.5 * 1024.0) as usize) // 50% for batching
                        .unwrap_or(512)
                }
                _ => (system.memory_info.available_gb * 0.3 * 1024.0) as usize, // 30% for batching
            };

            let batch_memory_mb = model.memory_requirements.activation_memory_mb as usize * 2; // Conservative estimate
            (available_mb / batch_memory_mb).max(1)
        };

        base_batch_size.min(memory_limited_batch)
    }

    fn get_fallback_config(&self) -> DeviceConfig {
        DeviceConfig {
            strategy: Some(DeviceStrategy::CpuOnly),
            cpu_threads: Some(4),
            gpu_memory_fraction: None,
            recommended_batch_size: 1,
        }
    }
}

#[derive(Debug, Clone)]
struct MemoryFeasibility {
    cpu_feasible: bool,
    gpu_feasible: bool,
    hybrid_feasible: bool,
    recommended_split: Option<f32>,
}

#[derive(Debug, Clone)]
enum QuantizationMethod {
    I2S,
    TL1,
    TL2,
    IQ2S,
    None,
}
```

**4. Performance Monitoring and Adaptive Optimization**

```rust
impl ProductionModelLoader {
    pub fn get_adaptive_device_config(&self, previous_performance: Option<&PerformanceMetrics>) -> DeviceConfig {
        let mut config = self.get_optimal_device_config();

        // Adapt based on previous performance if available
        if let Some(metrics) = previous_performance {
            config = self.adapt_config_from_metrics(config, metrics);
        }

        config
    }

    fn adapt_config_from_metrics(&self, mut config: DeviceConfig, metrics: &PerformanceMetrics) -> DeviceConfig {
        // If GPU utilization is low, consider reducing GPU usage
        if let Some(gpu_util) = metrics.gpu_utilization {
            if gpu_util < 0.3 && matches!(config.strategy, Some(DeviceStrategy::GpuOnly)) {
                info!("Low GPU utilization detected, suggesting hybrid strategy");
                // Switch to hybrid if possible
                if let Ok(model_reqs) = self.analyze_model_requirements() {
                    let total_layers = model_reqs.compute_requirements.parallel_layers;
                    config.strategy = Some(DeviceStrategy::Hybrid {
                        cpu_layers: total_layers / 2,
                        gpu_layers: total_layers / 2,
                    });
                }
            }
        }

        // Adjust batch size based on memory pressure
        if metrics.memory_pressure > 0.9 {
            config.recommended_batch_size = (config.recommended_batch_size / 2).max(1);
        } else if metrics.memory_pressure < 0.5 && config.recommended_batch_size < 8 {
            config.recommended_batch_size *= 2;
        }

        config
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub throughput_tokens_per_sec: f64,
    pub memory_pressure: f64, // 0.0 to 1.0
    pub gpu_utilization: Option<f64>, // 0.0 to 1.0
    pub cpu_utilization: f64, // 0.0 to 1.0
}
```

### Alternative Solutions Considered

1. **Static Configuration Tables**: Predefined configurations for common hardware
2. **User-Guided Selection**: Interactive configuration wizard
3. **Machine Learning Based**: Use ML to predict optimal configurations

## Implementation Breakdown

### Phase 1: Hardware Detection Infrastructure (Week 1)
- [ ] Add system information detection (CPU, memory, GPU)
- [ ] Implement performance tier classification
- [ ] Create hardware capability abstractions
- [ ] Add SIMD and GPU capability detection

### Phase 2: Model Analysis Framework (Week 1)
- [ ] Implement model memory requirement calculation
- [ ] Add compute requirement analysis
- [ ] Create quantization method detection
- [ ] Generate optimization hints based on model characteristics

### Phase 3: Intelligence Core (Week 2)
- [ ] Implement device strategy selection algorithm
- [ ] Add memory feasibility checking
- [ ] Create performance estimation models
- [ ] Implement optimal configuration calculation

### Phase 4: Configuration Optimization (Week 2)
- [ ] Add batch size optimization
- [ ] Implement GPU layer distribution for hybrid mode
- [ ] Create thread count optimization
- [ ] Add memory fraction calculation

### Phase 5: Adaptive Enhancement (Week 3)
- [ ] Implement performance-based adaptation
- [ ] Add monitoring integration
- [ ] Create configuration recommendation system
- [ ] Add fallback mechanisms

### Phase 6: Testing and Validation (Week 3)
- [ ] Create comprehensive test suite across hardware configurations
- [ ] Add performance benchmarking
- [ ] Validate against manual configurations
- [ ] Test edge cases and error handling

## Testing Strategy

### Hardware Simulation Testing
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_end_gpu_selection() {
        let system = SystemInfo {
            cpu_info: CpuInfo { performance_score: 10.0, ..Default::default() },
            gpu_info: Some(GpuInfo {
                performance_score: 15.0,
                memory_gb: 24.0,
                available: true,
                ..Default::default()
            }),
            performance_tier: PerformanceTier::Extreme,
            ..Default::default()
        };

        let loader = ProductionModelLoader::new();
        // Mock system info injection for testing
        let config = loader.calculate_optimal_config_with_system(&system).unwrap();

        assert!(matches!(config.strategy, Some(DeviceStrategy::GpuOnly)));
        assert!(config.recommended_batch_size > 1);
    }

    #[test]
    fn test_memory_constrained_selection() {
        let system = SystemInfo {
            memory_info: MemoryInfo {
                total_gb: 4.0,
                available_gb: 2.0,
                ..Default::default()
            },
            gpu_info: None,
            performance_tier: PerformanceTier::Low,
            ..Default::default()
        };

        let loader = ProductionModelLoader::new();
        let config = loader.calculate_optimal_config_with_system(&system).unwrap();

        assert!(matches!(config.strategy, Some(DeviceStrategy::CpuOnly)));
        assert_eq!(config.recommended_batch_size, 1);
    }

    #[test]
    fn test_hybrid_strategy_large_model() {
        // Test that large models on moderate hardware use hybrid strategy
        let system = SystemInfo {
            gpu_info: Some(GpuInfo {
                memory_gb: 8.0,  // Moderate GPU memory
                performance_score: 6.0,
                available: true,
                ..Default::default()
            }),
            memory_info: MemoryInfo { total_gb: 32.0, available_gb: 24.0, ..Default::default() },
            performance_tier: PerformanceTier::High,
            ..Default::default()
        };

        // Would test with a large model that doesn't fit entirely on GPU
    }
}
```

### Performance Benchmarking
```bash
# Test optimal configurations against manual configurations
cargo test --release --no-default-features --features cpu,gpu optimal_config_performance

# Benchmark configuration calculation time
cargo bench device_config_optimization

# Test with different model sizes
BITNET_MODEL_SIZE=1B cargo test device_config_small_model
BITNET_MODEL_SIZE=7B cargo test device_config_large_model
```

## Acceptance Criteria

- [ ] Intelligent hardware detection for CPU (cores, SIMD) and GPU (memory, compute capability)
- [ ] Model requirement analysis including memory and compute requirements
- [ ] Automatic device strategy selection (CPU-only, GPU-only, hybrid)
- [ ] Optimized configuration parameters (threads, batch size, memory allocation)
- [ ] Performance estimation models for different hardware/model combinations
- [ ] Fallback mechanisms for unsupported hardware or analysis failures
- [ ] Configuration optimization completes in < 100ms
- [ ] Adaptive configuration based on runtime performance metrics
- [ ] Comprehensive test coverage for various hardware scenarios
- [ ] Documentation includes hardware requirement guides and optimization explanations

## Dependencies

### New Dependencies
```toml
[dependencies]
sysinfo = "0.30"  # System information
num_cpus = "1.16"  # CPU detection

[dependencies.cudarc]
version = "0.10"
optional = true
features = ["driver"]
```

## Related Issues

- System requirements validation integration
- Performance monitoring system
- Hardware capability detection standardization
- Memory management optimization

## BitNet-Specific Considerations

- **Quantization Awareness**: Different quantization methods have different hardware optimization profiles
- **Memory Patterns**: BitNet's 1-bit quantization creates unique memory access patterns
- **SIMD Optimization**: Table lookup quantization methods excel on CPUs with good SIMD support
- **GPU Efficiency**: I2S quantization may have different GPU efficiency characteristics
- **Context Length Impact**: Large context lengths significantly impact memory requirements and optimal batch sizes

This intelligent device configuration system will dramatically improve out-of-the-box performance and reduce the manual configuration burden for BitNet.rs deployments across diverse hardware environments.