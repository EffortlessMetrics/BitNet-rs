# [Configuration] Make performance thresholds configurable for diverse hardware environments

## Problem Description

The `PerformanceThresholds` struct in `crates/bitnet-inference/src/validation.rs` uses hardcoded default values that are inappropriate for the wide range of hardware configurations where BitNet-rs may be deployed. These fixed thresholds cause validation failures on lower-end hardware and don't fully utilize capabilities of high-end systems.

## Environment
- **File**: `crates/bitnet-inference/src/validation.rs`
- **Struct**: `PerformanceThresholds`
- **Hardware Range**:
  - CPU: ARM Cortex-A55 to Intel Xeon/AMD EPYC
  - GPU: GT 1030 (2GB) to H100 (80GB)
  - Memory: 4GB to 1TB+ systems
- **Deployment Targets**: Edge devices, workstations, data center servers

## Reproduction Steps

1. Run validation on different hardware configurations:
   ```bash
   # On low-end hardware (4GB RAM, integrated GPU)
   cargo test --features crossval validation::performance_tests

   # On high-end hardware (64GB RAM, RTX 4090)
   cargo test --features crossval validation::performance_tests
   ```

2. Observe threshold validation results:
   ```bash
   cargo run -p xtask -- benchmark --model small-model.gguf --validate-thresholds
   ```

**Expected Results**:
- Thresholds should adapt to hardware capabilities
- Low-end hardware should have achievable performance targets
- High-end hardware should have ambitious but realistic targets

**Actual Results**:
- Fixed thresholds cause failures on resource-constrained devices
- High-end hardware operates far below potential due to conservative thresholds
- No adaptation to available resources

## Root Cause Analysis

### Current Hardcoded Implementation

```rust
pub struct PerformanceThresholds {
    pub min_tokens_per_second: f64,
    pub max_latency_ms: f64,
    pub max_memory_usage_mb: f64,
    pub min_speedup_factor: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            min_tokens_per_second: 10.0,      // Too high for edge devices
            max_latency_ms: 5000.0,           // Too conservative for servers
            max_memory_usage_mb: 8192.0,      // Doesn't scale with available RAM
            min_speedup_factor: 1.5,          // May not be achievable on all hardware
        }
    }
}
```

### Problems with Fixed Thresholds

1. **Edge Device Failures**: 10 tokens/second may be impossible on ARM Cortex-A55
2. **Server Underutilization**: 5-second latency limit is too conservative for data centers
3. **Memory Mismatch**: 8GB limit ignores available system memory (4GB or 128GB)
4. **GPU Variance**: Speedup expectations don't account for GPU capabilities

### Hardware Performance Variability

| Hardware Class | Expected Tokens/Sec | Typical Latency | Memory Capacity |
|----------------|-------------------|----------------|-----------------|
| Edge (Pi 4)    | 1-5 tokens/sec    | 10-30 seconds  | 4-8 GB         |
| Mobile (ARM)   | 3-15 tokens/sec   | 5-15 seconds   | 8-16 GB        |
| Laptop (x86)   | 10-50 tokens/sec  | 2-8 seconds    | 16-32 GB       |
| Workstation    | 30-200 tokens/sec | 0.5-3 seconds  | 32-128 GB      |
| Server/Cloud   | 100-1000+ tokens/sec | 0.1-1 second | 64GB-1TB+      |

## Impact Assessment

- **Severity**: Medium-High (deployment flexibility)
- **Deployment Impact**:
  - Cannot deploy on edge devices due to unrealistic thresholds
  - Inefficient resource utilization on high-end hardware
  - Manual threshold adjustment required for each deployment

- **Development Impact**:
  - CI/CD failures on different hardware configurations
  - Difficulty benchmarking across hardware tiers
  - Reduced confidence in performance validation

- **User Experience**:
  - False performance failures on capable hardware
  - Missed optimization opportunities
  - Inconsistent performance expectations

## Proposed Solution

Implement adaptive performance thresholds that automatically configure based on hardware detection, with overrides for custom scenarios and deployment-specific requirements.

### Technical Implementation

#### 1. Hardware Detection and Classification

```rust
use sysinfo::{System, SystemExt, ProcessorExt, MemoryExt};

#[derive(Debug, Clone, PartialEq)]
pub enum HardwareClass {
    Edge,        // Raspberry Pi, embedded systems
    Mobile,      // Laptops, tablets
    Workstation, // Desktop, gaming rigs
    Server,      // Data center, cloud instances
}

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub class: HardwareClass,
    pub cpu_cores: usize,
    pub cpu_frequency_mhz: u64,
    pub cpu_architecture: CpuArchitecture,
    pub total_memory_gb: f64,
    pub gpu_info: Option<GpuInfo>,
    pub estimated_performance: PerformanceEstimate,
}

#[derive(Debug, Clone)]
pub enum CpuArchitecture {
    X86_64,
    AArch64,
    Other(String),
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub device_name: String,
    pub memory_gb: f64,
    pub compute_capability: Option<(u32, u32)>,
    pub estimated_flops: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceEstimate {
    pub estimated_tokens_per_second: f64,
    pub estimated_latency_ms: f64,
    pub safe_memory_usage_gb: f64,
    pub gpu_speedup_factor: f64,
}

pub struct HardwareDetector;

impl HardwareDetector {
    pub fn detect_hardware_profile() -> Result<HardwareProfile> {
        let mut system = System::new_all();
        system.refresh_all();

        let cpu_cores = system.physical_core_count().unwrap_or(1);
        let cpu_frequency = system.global_processor_info().frequency();
        let total_memory_bytes = system.total_memory();
        let total_memory_gb = total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

        let cpu_architecture = Self::detect_cpu_architecture();
        let gpu_info = Self::detect_gpu_info()?;

        let hardware_class = Self::classify_hardware(
            cpu_cores,
            cpu_frequency,
            total_memory_gb,
            &gpu_info,
            &cpu_architecture,
        );

        let estimated_performance = Self::estimate_performance(
            &hardware_class,
            cpu_cores,
            cpu_frequency,
            total_memory_gb,
            &gpu_info,
        );

        Ok(HardwareProfile {
            class: hardware_class,
            cpu_cores,
            cpu_frequency_mhz: cpu_frequency,
            cpu_architecture,
            total_memory_gb,
            gpu_info,
            estimated_performance,
        })
    }

    fn classify_hardware(
        cpu_cores: usize,
        cpu_frequency: u64,
        memory_gb: f64,
        gpu_info: &Option<GpuInfo>,
        architecture: &CpuArchitecture,
    ) -> HardwareClass {
        // Edge devices: limited cores, memory, often ARM
        if memory_gb < 6.0 || cpu_cores <= 4 && cpu_frequency < 2000 {
            return HardwareClass::Edge;
        }

        // Server/Cloud: high core count, large memory, often multiple GPUs
        if cpu_cores >= 16 && memory_gb >= 32.0 {
            if let Some(gpu) = gpu_info {
                if gpu.memory_gb >= 16.0 {
                    return HardwareClass::Server;
                }
            }
            if memory_gb >= 64.0 {
                return HardwareClass::Server;
            }
        }

        // Workstation: good CPU, good GPU, reasonable memory
        if cpu_cores >= 8 && memory_gb >= 16.0 {
            if let Some(gpu) = gpu_info {
                if gpu.memory_gb >= 8.0 {
                    return HardwareClass::Workstation;
                }
            }
        }

        // Default to mobile for everything else
        HardwareClass::Mobile
    }

    fn estimate_performance(
        hardware_class: &HardwareClass,
        cpu_cores: usize,
        cpu_frequency: u64,
        memory_gb: f64,
        gpu_info: &Option<GpuInfo>,
    ) -> PerformanceEstimate {
        let base_performance = match hardware_class {
            HardwareClass::Edge => PerformanceEstimate {
                estimated_tokens_per_second: 2.0,
                estimated_latency_ms: 15000.0,
                safe_memory_usage_gb: (memory_gb * 0.7).min(4.0),
                gpu_speedup_factor: 1.1,
            },
            HardwareClass::Mobile => PerformanceEstimate {
                estimated_tokens_per_second: 8.0,
                estimated_latency_ms: 8000.0,
                safe_memory_usage_gb: memory_gb * 0.75,
                gpu_speedup_factor: 1.3,
            },
            HardwareClass::Workstation => PerformanceEstimate {
                estimated_tokens_per_second: 25.0,
                estimated_latency_ms: 3000.0,
                safe_memory_usage_gb: memory_gb * 0.8,
                gpu_speedup_factor: 2.5,
            },
            HardwareClass::Server => PerformanceEstimate {
                estimated_tokens_per_second: 100.0,
                estimated_latency_ms: 1000.0,
                safe_memory_usage_gb: memory_gb * 0.85,
                gpu_speedup_factor: 5.0,
            },
        };

        // Adjust based on specific hardware characteristics
        Self::adjust_for_cpu_performance(base_performance, cpu_cores, cpu_frequency, gpu_info)
    }

    fn adjust_for_cpu_performance(
        mut estimate: PerformanceEstimate,
        cpu_cores: usize,
        cpu_frequency: u64,
        gpu_info: &Option<GpuInfo>,
    ) -> PerformanceEstimate {
        // CPU scaling factor based on cores and frequency
        let cpu_factor = (cpu_cores as f64).sqrt() * (cpu_frequency as f64 / 2500.0);
        estimate.estimated_tokens_per_second *= cpu_factor.max(0.3).min(3.0);

        // GPU adjustment
        if let Some(gpu) = gpu_info {
            let gpu_factor = (gpu.memory_gb / 8.0).clamp(0.5, 10.0);
            estimate.gpu_speedup_factor *= gpu_factor;

            // High-end GPUs can significantly improve performance
            if gpu.memory_gb >= 16.0 {
                estimate.estimated_tokens_per_second *= 1.5;
                estimate.estimated_latency_ms *= 0.7;
            }
        }

        estimate
    }

    fn detect_gpu_info() -> Result<Option<GpuInfo>> {
        #[cfg(feature = "gpu")]
        {
            use cudarc::driver::CudaDevice;

            match CudaDevice::new(0) {
                Ok(device) => {
                    let memory_bytes = device.total_memory()?;
                    let memory_gb = memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

                    // Get device name and compute capability
                    let device_name = device.name()?;
                    let (major, minor) = device.compute_capability()?;

                    // Estimate FLOPS based on known GPU characteristics
                    let estimated_flops = Self::estimate_gpu_flops(&device_name, memory_gb);

                    Ok(Some(GpuInfo {
                        device_name,
                        memory_gb,
                        compute_capability: Some((major, minor)),
                        estimated_flops,
                    }))
                }
                Err(_) => Ok(None),
            }
        }
        #[cfg(not(feature = "gpu"))]
        Ok(None)
    }

    fn estimate_gpu_flops(device_name: &str, memory_gb: f64) -> f64 {
        // Rough FLOPS estimates for common GPUs (TFLOPS)
        if device_name.contains("H100") {
            60.0
        } else if device_name.contains("A100") {
            30.0
        } else if device_name.contains("RTX 4090") {
            35.0
        } else if device_name.contains("RTX 3090") {
            25.0
        } else if device_name.contains("RTX") && memory_gb >= 8.0 {
            15.0 + memory_gb * 1.5
        } else if device_name.contains("GTX") {
            5.0 + memory_gb * 1.0
        } else {
            // Generic estimate based on memory
            memory_gb * 2.0
        }
    }
}
```

#### 2. Adaptive Threshold Configuration

```rust
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct AdaptivePerformanceThresholds {
    pub hardware_profile: HardwareProfile,
    pub thresholds: PerformanceThresholds,
    pub override_config: Option<PerformanceOverrides>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PerformanceOverrides {
    pub min_tokens_per_second: Option<f64>,
    pub max_latency_ms: Option<f64>,
    pub max_memory_usage_mb: Option<f64>,
    pub min_speedup_factor: Option<f64>,
    pub custom_scaling_factors: Option<ScalingFactors>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ScalingFactors {
    pub performance_target_multiplier: f64,  // 1.0 = realistic, 1.5 = ambitious
    pub memory_safety_factor: f64,           // 0.8 = conservative, 0.95 = aggressive
    pub latency_tolerance_factor: f64,       // 1.0 = strict, 2.0 = relaxed
}

impl AdaptivePerformanceThresholds {
    pub fn detect_and_configure() -> Result<Self> {
        let hardware_profile = HardwareDetector::detect_hardware_profile()?;
        let thresholds = Self::calculate_thresholds(&hardware_profile, None);

        Ok(Self {
            hardware_profile,
            thresholds,
            override_config: None,
        })
    }

    pub fn with_overrides(overrides: PerformanceOverrides) -> Result<Self> {
        let hardware_profile = HardwareDetector::detect_hardware_profile()?;
        let thresholds = Self::calculate_thresholds(&hardware_profile, Some(&overrides));

        Ok(Self {
            hardware_profile,
            thresholds,
            override_config: Some(overrides),
        })
    }

    pub fn from_config_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let config_content = std::fs::read_to_string(path)?;
        let loaded_config: AdaptivePerformanceThresholds = serde_yaml::from_str(&config_content)?;

        // Re-detect hardware in case it has changed
        let current_hardware = HardwareDetector::detect_hardware_profile()?;

        if current_hardware.class != loaded_config.hardware_profile.class {
            log::warn!(
                "Hardware class changed from {:?} to {:?}, recalculating thresholds",
                loaded_config.hardware_profile.class,
                current_hardware.class
            );

            return Self::with_overrides(
                loaded_config.override_config.unwrap_or_default()
            );
        }

        Ok(loaded_config)
    }

    fn calculate_thresholds(
        hardware_profile: &HardwareProfile,
        overrides: Option<&PerformanceOverrides>,
    ) -> PerformanceThresholds {
        let estimate = &hardware_profile.estimated_performance;

        // Apply scaling factors from overrides
        let scaling = overrides
            .and_then(|o| o.custom_scaling_factors.as_ref())
            .cloned()
            .unwrap_or_default();

        let base_thresholds = PerformanceThresholds {
            min_tokens_per_second: (estimate.estimated_tokens_per_second * 0.7 * scaling.performance_target_multiplier).max(0.5),
            max_latency_ms: estimate.estimated_latency_ms * scaling.latency_tolerance_factor,
            max_memory_usage_mb: (estimate.safe_memory_usage_gb * 1024.0 * scaling.memory_safety_factor) as f64,
            min_speedup_factor: (estimate.gpu_speedup_factor * 0.8).max(1.1),
        };

        // Apply direct overrides
        PerformanceThresholds {
            min_tokens_per_second: overrides
                .and_then(|o| o.min_tokens_per_second)
                .unwrap_or(base_thresholds.min_tokens_per_second),
            max_latency_ms: overrides
                .and_then(|o| o.max_latency_ms)
                .unwrap_or(base_thresholds.max_latency_ms),
            max_memory_usage_mb: overrides
                .and_then(|o| o.max_memory_usage_mb)
                .unwrap_or(base_thresholds.max_memory_usage_mb),
            min_speedup_factor: overrides
                .and_then(|o| o.min_speedup_factor)
                .unwrap_or(base_thresholds.min_speedup_factor),
        }
    }

    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let yaml_content = serde_yaml::to_string(self)?;
        std::fs::write(path, yaml_content)?;
        Ok(())
    }

    pub fn log_configuration(&self) {
        log::info!("Hardware Profile:");
        log::info!("  Class: {:?}", self.hardware_profile.class);
        log::info!("  CPU: {} cores @ {} MHz",
                  self.hardware_profile.cpu_cores,
                  self.hardware_profile.cpu_frequency_mhz);
        log::info!("  Memory: {:.1} GB", self.hardware_profile.total_memory_gb);

        if let Some(gpu) = &self.hardware_profile.gpu_info {
            log::info!("  GPU: {} ({:.1} GB)", gpu.device_name, gpu.memory_gb);
        }

        log::info!("Performance Thresholds:");
        log::info!("  Min tokens/sec: {:.1}", self.thresholds.min_tokens_per_second);
        log::info!("  Max latency: {:.0} ms", self.thresholds.max_latency_ms);
        log::info!("  Max memory: {:.0} MB", self.thresholds.max_memory_usage_mb);
        log::info!("  Min speedup: {:.1}x", self.thresholds.min_speedup_factor);
    }
}

impl Default for ScalingFactors {
    fn default() -> Self {
        Self {
            performance_target_multiplier: 1.0,
            memory_safety_factor: 0.8,
            latency_tolerance_factor: 1.0,
        }
    }
}

impl Default for PerformanceOverrides {
    fn default() -> Self {
        Self {
            min_tokens_per_second: None,
            max_latency_ms: None,
            max_memory_usage_mb: None,
            min_speedup_factor: None,
            custom_scaling_factors: None,
        }
    }
}
```

#### 3. Environment-Specific Configuration

```rust
// Configuration file examples for different deployment scenarios

// Edge deployment (Raspberry Pi, embedded)
// config/edge-performance.yml
hardware_profile:
  class: Edge
  cpu_cores: 4
  cpu_frequency_mhz: 1800
  total_memory_gb: 4.0
  gpu_info: null

thresholds:
  min_tokens_per_second: 1.0
  max_latency_ms: 30000.0
  max_memory_usage_mb: 2048.0
  min_speedup_factor: 1.1

override_config:
  custom_scaling_factors:
    performance_target_multiplier: 0.8  # Conservative for stability
    memory_safety_factor: 0.7           # Extra conservative on limited RAM
    latency_tolerance_factor: 2.0       # Relaxed latency expectations

// Workstation deployment (Gaming/Development machine)
// config/workstation-performance.yml
hardware_profile:
  class: Workstation
  cpu_cores: 16
  cpu_frequency_mhz: 3600
  total_memory_gb: 32.0
  gpu_info:
    device_name: "RTX 4090"
    memory_gb: 24.0
    compute_capability: [8, 9]

thresholds:
  min_tokens_per_second: 25.0
  max_latency_ms: 2000.0
  max_memory_usage_mb: 20480.0
  min_speedup_factor: 3.0

override_config:
  custom_scaling_factors:
    performance_target_multiplier: 1.2  # Ambitious targets
    memory_safety_factor: 0.85          # Can use more memory
    latency_tolerance_factor: 0.8       # Strict latency requirements

// Server deployment (Data center/Cloud)
// config/server-performance.yml
hardware_profile:
  class: Server
  cpu_cores: 64
  cpu_frequency_mhz: 2800
  total_memory_gb: 128.0
  gpu_info:
    device_name: "A100"
    memory_gb: 40.0
    compute_capability: [8, 0]

thresholds:
  min_tokens_per_second: 100.0
  max_latency_ms: 500.0
  max_memory_usage_mb: 102400.0
  min_speedup_factor: 8.0

override_config:
  custom_scaling_factors:
    performance_target_multiplier: 1.5  # High performance expectations
    memory_safety_factor: 0.9           # Can use most memory
    latency_tolerance_factor: 0.5       # Very strict latency requirements
```

#### 4. Integration with Validation Framework

```rust
impl ValidationFramework {
    pub fn new_with_adaptive_thresholds() -> Result<Self> {
        let adaptive_config = AdaptivePerformanceThresholds::detect_and_configure()?;
        adaptive_config.log_configuration();

        Ok(Self {
            performance_thresholds: adaptive_config.thresholds,
            hardware_profile: Some(adaptive_config.hardware_profile),
            // ... other fields
        })
    }

    pub fn new_with_config_file<P: AsRef<std::path::Path>>(config_path: P) -> Result<Self> {
        let adaptive_config = AdaptivePerformanceThresholds::from_config_file(config_path)?;
        adaptive_config.log_configuration();

        Ok(Self {
            performance_thresholds: adaptive_config.thresholds,
            hardware_profile: Some(adaptive_config.hardware_profile),
            // ... other fields
        })
    }

    pub fn validate_performance(&self, metrics: &PerformanceMetrics) -> ValidationResult {
        let mut results = Vec::new();

        // Tokens per second validation with context
        if metrics.tokens_per_second < self.performance_thresholds.min_tokens_per_second {
            let severity = if metrics.tokens_per_second < self.performance_thresholds.min_tokens_per_second * 0.5 {
                ValidationSeverity::Critical
            } else {
                ValidationSeverity::Warning
            };

            results.push(ValidationIssue {
                severity,
                category: ValidationCategory::Performance,
                message: format!(
                    "Tokens per second {:.1} below threshold {:.1} for {:?} hardware",
                    metrics.tokens_per_second,
                    self.performance_thresholds.min_tokens_per_second,
                    self.hardware_profile.as_ref().map(|h| &h.class).unwrap_or(&HardwareClass::Mobile)
                ),
                suggestion: self.suggest_performance_improvement(metrics),
            });
        }

        // Latency validation with hardware-specific expectations
        if metrics.latency_ms > self.performance_thresholds.max_latency_ms {
            results.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                category: ValidationCategory::Performance,
                message: format!(
                    "Latency {:.0}ms exceeds threshold {:.0}ms",
                    metrics.latency_ms,
                    self.performance_thresholds.max_latency_ms
                ),
                suggestion: self.suggest_latency_improvement(metrics),
            });
        }

        // Memory usage with adaptive limits
        if metrics.memory_usage_mb > self.performance_thresholds.max_memory_usage_mb {
            results.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                category: ValidationCategory::Memory,
                message: format!(
                    "Memory usage {:.0}MB exceeds safe limit {:.0}MB",
                    metrics.memory_usage_mb,
                    self.performance_thresholds.max_memory_usage_mb
                ),
                suggestion: "Consider reducing model size or batch size".to_string(),
            });
        }

        ValidationResult {
            passed: results.is_empty(),
            issues: results,
        }
    }

    fn suggest_performance_improvement(&self, metrics: &PerformanceMetrics) -> String {
        if let Some(hardware) = &self.hardware_profile {
            match hardware.class {
                HardwareClass::Edge => {
                    "Try reducing model size, using quantization, or lowering batch size".to_string()
                }
                HardwareClass::Mobile => {
                    "Consider enabling CPU optimizations or using a smaller model variant".to_string()
                }
                HardwareClass::Workstation => {
                    "Check GPU utilization and consider using mixed precision".to_string()
                }
                HardwareClass::Server => {
                    "Enable parallel processing and check for resource contention".to_string()
                }
            }
        } else {
            "Check hardware utilization and consider optimization options".to_string()
        }
    }
}
```

## Implementation Plan

### Phase 1: Hardware Detection (Week 1-2)
- [ ] Implement system information detection
- [ ] Add hardware classification logic
- [ ] Create performance estimation algorithms
- [ ] Add GPU detection and capability assessment

### Phase 2: Adaptive Thresholds (Week 3)
- [ ] Implement threshold calculation based on hardware
- [ ] Add configuration override system
- [ ] Create scaling factor mechanisms
- [ ] Add serialization for configuration persistence

### Phase 3: Integration & Testing (Week 4)
- [ ] Integrate with validation framework
- [ ] Add comprehensive test suite for different hardware classes
- [ ] Create example configurations for common scenarios
- [ ] Add performance regression testing

### Phase 4: Documentation & Tools (Week 5)
- [ ] Create configuration generator tool
- [ ] Add comprehensive documentation
- [ ] Create deployment guides for different environments
- [ ] Add monitoring and alerting for threshold violations

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_classification() {
        // Edge device
        let edge_class = HardwareDetector::classify_hardware(
            4, 1800, 4.0, &None, &CpuArchitecture::AArch64
        );
        assert_eq!(edge_class, HardwareClass::Edge);

        // Workstation
        let workstation_class = HardwareDetector::classify_hardware(
            16, 3600, 32.0, &Some(GpuInfo {
                device_name: "RTX 4090".to_string(),
                memory_gb: 24.0,
                compute_capability: Some((8, 9)),
                estimated_flops: 35.0,
            }), &CpuArchitecture::X86_64
        );
        assert_eq!(workstation_class, HardwareClass::Workstation);
    }

    #[test]
    fn test_threshold_adaptation() {
        let edge_profile = HardwareProfile {
            class: HardwareClass::Edge,
            cpu_cores: 4,
            cpu_frequency_mhz: 1800,
            cpu_architecture: CpuArchitecture::AArch64,
            total_memory_gb: 4.0,
            gpu_info: None,
            estimated_performance: PerformanceEstimate {
                estimated_tokens_per_second: 2.0,
                estimated_latency_ms: 15000.0,
                safe_memory_usage_gb: 2.8,
                gpu_speedup_factor: 1.1,
            },
        };

        let thresholds = AdaptivePerformanceThresholds::calculate_thresholds(&edge_profile, None);

        assert!(thresholds.min_tokens_per_second < 5.0); // Should be achievable for edge
        assert!(thresholds.max_latency_ms > 10000.0);    // Should allow reasonable latency
        assert!(thresholds.max_memory_usage_mb < 4096.0); // Should not exceed available memory
    }

    #[test]
    fn test_override_application() {
        let profile = create_test_workstation_profile();
        let overrides = PerformanceOverrides {
            min_tokens_per_second: Some(50.0),
            custom_scaling_factors: Some(ScalingFactors {
                performance_target_multiplier: 2.0,
                memory_safety_factor: 0.95,
                latency_tolerance_factor: 0.5,
            }),
            ..Default::default()
        };

        let thresholds = AdaptivePerformanceThresholds::calculate_thresholds(&profile, Some(&overrides));

        assert_eq!(thresholds.min_tokens_per_second, 50.0); // Direct override
        assert!(thresholds.max_memory_usage_mb > profile.total_memory_gb * 1024.0 * 0.9); // High memory factor
    }
}
```

### Integration Tests
```rust
#[cfg(test)]
mod integration_tests {
    #[test]
    fn test_end_to_end_configuration() {
        // Test automatic detection and configuration
        let adaptive_config = AdaptivePerformanceThresholds::detect_and_configure().unwrap();

        // Verify thresholds are reasonable
        assert!(adaptive_config.thresholds.min_tokens_per_second > 0.0);
        assert!(adaptive_config.thresholds.max_latency_ms > 0.0);
        assert!(adaptive_config.thresholds.max_memory_usage_mb > 0.0);

        // Test serialization round-trip
        let temp_file = "/tmp/test-performance-config.yml";
        adaptive_config.save_to_file(temp_file).unwrap();
        let loaded_config = AdaptivePerformanceThresholds::from_config_file(temp_file).unwrap();

        assert_eq!(adaptive_config.hardware_profile.class, loaded_config.hardware_profile.class);
    }

    #[test]
    fn test_validation_with_adaptive_thresholds() {
        let framework = ValidationFramework::new_with_adaptive_thresholds().unwrap();

        // Test metrics that should pass for the detected hardware
        let reasonable_metrics = PerformanceMetrics {
            tokens_per_second: framework.performance_thresholds.min_tokens_per_second * 1.2,
            latency_ms: framework.performance_thresholds.max_latency_ms * 0.8,
            memory_usage_mb: framework.performance_thresholds.max_memory_usage_mb * 0.7,
            gpu_speedup: framework.performance_thresholds.min_speedup_factor * 1.1,
        };

        let result = framework.validate_performance(&reasonable_metrics);
        assert!(result.passed, "Reasonable metrics should pass validation");
    }
}
```

## Performance Impact

- **Hardware Detection**: ~10-50ms one-time cost at startup
- **Configuration Loading**: ~1-5ms for YAML parsing
- **Runtime Overhead**: Zero (thresholds calculated once)
- **Memory Usage**: ~1-2KB for configuration storage

## Acceptance Criteria

- [ ] Hardware detection works across all supported platforms
- [ ] Thresholds automatically adapt to detected hardware capabilities
- [ ] Configuration can be overridden for custom scenarios
- [ ] Edge devices get achievable performance targets
- [ ] High-end hardware gets ambitious but realistic targets
- [ ] Configuration can be saved and loaded from files
- [ ] Integration with existing validation framework
- [ ] Comprehensive test coverage for different hardware classes
- [ ] Clear documentation and usage examples
- [ ] Performance regression testing validates adaptive behavior

## Dependencies

- `sysinfo` for hardware detection
- `serde` and `serde_yaml` for configuration serialization
- `cudarc` (optional) for GPU information detection
- Existing validation framework

## Related Issues

- Performance validation improvements
- Cross-platform deployment support
- Hardware-specific optimizations
- Deployment automation and configuration management

## Labels
- `configuration`
- `performance`
- `validation`
- `hardware-detection`
- `deployment`
- `priority-medium`
- `enhancement`
