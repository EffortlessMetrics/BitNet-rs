# [HARDCODED] PerformanceThresholds uses fixed values inappropriate for different hardware configurations

## Problem Description

The `PerformanceThresholds` struct in `validation.rs` uses hardcoded default values that may be inappropriate for different hardware configurations, model sizes, and deployment scenarios, leading to false positives or negatives in performance validation.

## Environment

**File**: `crates/bitnet-inference/src/validation.rs`
**Component**: Performance Validation Framework
**Issue Type**: Hardcoded Values / Inflexible Configuration

## Root Cause Analysis

**Current Implementation:**
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
            min_tokens_per_second: 10.0,
            max_latency_ms: 5000.0,
            max_memory_usage_mb: 8192.0,
            min_speedup_factor: 1.5, // Expect at least 1.5x speedup over Python
        }
    }
}
```

**Analysis:**
1. **Hardware Agnostic**: Same thresholds for high-end GPUs and low-end CPUs
2. **Model Size Ignorant**: Same limits for small and large models
3. **Context Insensitive**: No consideration for deployment scenarios
4. **Static Configuration**: No runtime adjustment based on actual performance

## Impact Assessment

**Severity**: Medium
**Affected Areas**:
- Performance validation accuracy
- CI/CD pipeline reliability
- Production deployment confidence
- Cross-platform compatibility

**Validation Impact**:
- False failures on low-end hardware
- False passes on high-end systems with poor optimization
- Inconsistent validation results across environments
- Reduced trust in performance metrics

**Business Impact**:
- Delayed deployments due to incorrect validation failures
- Reduced confidence in performance regression detection
- Poor user experience on diverse hardware configurations

## Proposed Solution

### Adaptive Performance Threshold System

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub min_tokens_per_second: f64,
    pub max_latency_ms: f64,
    pub max_memory_usage_mb: f64,
    pub min_speedup_factor: f64,
    pub max_gpu_memory_mb: Option<f64>,
    pub min_accuracy_score: f64,
    pub max_compile_time_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveThresholds {
    /// Hardware-specific thresholds
    pub hardware_profiles: HashMap<HardwareProfile, PerformanceThresholds>,

    /// Model-size specific adjustments
    pub model_size_factors: HashMap<ModelSizeCategory, ThresholdFactors>,

    /// Environment-specific overrides
    pub environment_overrides: HashMap<Environment, ThresholdAdjustments>,

    /// Base thresholds used for calculation
    pub base_thresholds: PerformanceThresholds,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum HardwareProfile {
    HighEndGPU,      // RTX 4090, A100, etc.
    MidRangeGPU,     // RTX 3070, etc.
    IntegratedGPU,   // Intel/AMD integrated
    HighEndCPU,      // 16+ cores, recent architecture
    MidRangeCPU,     // 8-16 cores
    LowEndCPU,       // 4-8 cores
    Mobile,          // ARM, limited power
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ModelSizeCategory {
    Tiny,     // < 100M parameters
    Small,    // 100M - 1B parameters
    Medium,   // 1B - 7B parameters
    Large,    // 7B - 30B parameters
    ExtraLarge, // > 30B parameters
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum Environment {
    Development,
    CI,
    Staging,
    Production,
    Benchmarking,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdFactors {
    pub tokens_per_second_factor: f64,
    pub latency_factor: f64,
    pub memory_factor: f64,
    pub speedup_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdAdjustments {
    pub tokens_per_second_adjustment: Option<f64>,
    pub latency_adjustment: Option<f64>,
    pub memory_adjustment: Option<f64>,
    pub speedup_adjustment: Option<f64>,
    pub relative_adjustments: bool,
}

impl AdaptiveThresholds {
    pub fn new() -> Self {
        let mut hardware_profiles = HashMap::new();

        // High-end GPU thresholds
        hardware_profiles.insert(HardwareProfile::HighEndGPU, PerformanceThresholds {
            min_tokens_per_second: 100.0,
            max_latency_ms: 500.0,
            max_memory_usage_mb: 16384.0,
            min_speedup_factor: 10.0,
            max_gpu_memory_mb: Some(24576.0),
            min_accuracy_score: 0.95,
            max_compile_time_ms: Some(30000.0),
        });

        // Mid-range GPU thresholds
        hardware_profiles.insert(HardwareProfile::MidRangeGPU, PerformanceThresholds {
            min_tokens_per_second: 50.0,
            max_latency_ms: 1000.0,
            max_memory_usage_mb: 8192.0,
            min_speedup_factor: 5.0,
            max_gpu_memory_mb: Some(12288.0),
            min_accuracy_score: 0.95,
            max_compile_time_ms: Some(60000.0),
        });

        // High-end CPU thresholds
        hardware_profiles.insert(HardwareProfile::HighEndCPU, PerformanceThresholds {
            min_tokens_per_second: 20.0,
            max_latency_ms: 2000.0,
            max_memory_usage_mb: 32768.0,
            min_speedup_factor: 3.0,
            max_gpu_memory_mb: None,
            min_accuracy_score: 0.95,
            max_compile_time_ms: Some(120000.0),
        });

        // Low-end CPU thresholds
        hardware_profiles.insert(HardwareProfile::LowEndCPU, PerformanceThresholds {
            min_tokens_per_second: 2.0,
            max_latency_ms: 10000.0,
            max_memory_usage_mb: 4096.0,
            min_speedup_factor: 1.2,
            max_gpu_memory_mb: None,
            min_accuracy_score: 0.90,
            max_compile_time_ms: Some(300000.0),
        });

        let mut model_size_factors = HashMap::new();
        model_size_factors.insert(ModelSizeCategory::Tiny, ThresholdFactors {
            tokens_per_second_factor: 5.0,   // Expect higher throughput for tiny models
            latency_factor: 0.1,              // Much lower latency
            memory_factor: 0.1,               // Much less memory
            speedup_factor: 1.0,              // No adjustment
        });

        model_size_factors.insert(ModelSizeCategory::Large, ThresholdFactors {
            tokens_per_second_factor: 0.1,   // Lower throughput for large models
            latency_factor: 10.0,             // Higher latency acceptable
            memory_factor: 10.0,              // Much more memory allowed
            speedup_factor: 0.5,              // Lower speedup expectations
        });

        let mut environment_overrides = HashMap::new();
        environment_overrides.insert(Environment::CI, ThresholdAdjustments {
            tokens_per_second_adjustment: Some(0.5), // More lenient for CI
            latency_adjustment: Some(2.0),
            memory_adjustment: Some(1.5),
            speedup_adjustment: Some(0.8),
            relative_adjustments: true,
        });

        environment_overrides.insert(Environment::Production, ThresholdAdjustments {
            tokens_per_second_adjustment: Some(1.2), // Stricter for production
            latency_adjustment: Some(0.8),
            memory_adjustment: Some(0.9),
            speedup_adjustment: Some(1.1),
            relative_adjustments: true,
        });

        Self {
            hardware_profiles,
            model_size_factors,
            environment_overrides,
            base_thresholds: PerformanceThresholds::conservative_defaults(),
        }
    }

    pub fn get_thresholds(
        &self,
        hardware: &HardwareProfile,
        model_size: &ModelSizeCategory,
        environment: &Environment,
    ) -> PerformanceThresholds {
        // Start with hardware-specific base thresholds
        let mut thresholds = self.hardware_profiles
            .get(hardware)
            .cloned()
            .unwrap_or_else(|| self.base_thresholds.clone());

        // Apply model size factors
        if let Some(factors) = self.model_size_factors.get(model_size) {
            thresholds.min_tokens_per_second *= factors.tokens_per_second_factor;
            thresholds.max_latency_ms *= factors.latency_factor;
            thresholds.max_memory_usage_mb *= factors.memory_factor;
            thresholds.min_speedup_factor *= factors.speedup_factor;
        }

        // Apply environment adjustments
        if let Some(adjustments) = self.environment_overrides.get(environment) {
            if adjustments.relative_adjustments {
                if let Some(adj) = adjustments.tokens_per_second_adjustment {
                    thresholds.min_tokens_per_second *= adj;
                }
                if let Some(adj) = adjustments.latency_adjustment {
                    thresholds.max_latency_ms *= adj;
                }
                if let Some(adj) = adjustments.memory_adjustment {
                    thresholds.max_memory_usage_mb *= adj;
                }
                if let Some(adj) = adjustments.speedup_adjustment {
                    thresholds.min_speedup_factor *= adj;
                }
            } else {
                // Absolute adjustments
                if let Some(adj) = adjustments.tokens_per_second_adjustment {
                    thresholds.min_tokens_per_second = adj;
                }
                // ... similar for other fields
            }
        }

        thresholds
    }

    pub fn detect_hardware_profile() -> Result<HardwareProfile> {
        // Auto-detect hardware configuration
        let gpu_info = detect_gpu_capabilities()?;
        let cpu_info = detect_cpu_capabilities()?;

        if let Some(gpu) = gpu_info {
            match gpu.memory_gb {
                memory if memory >= 20 => Ok(HardwareProfile::HighEndGPU),
                memory if memory >= 8 => Ok(HardwareProfile::MidRangeGPU),
                _ => Ok(HardwareProfile::IntegratedGPU),
            }
        } else {
            match cpu_info.cores {
                cores if cores >= 16 => Ok(HardwareProfile::HighEndCPU),
                cores if cores >= 8 => Ok(HardwareProfile::MidRangeCPU),
                _ => Ok(HardwareProfile::LowEndCPU),
            }
        }
    }

    pub fn detect_model_size(model_path: &str) -> Result<ModelSizeCategory> {
        // Detect model size from file size or metadata
        let model_size = std::fs::metadata(model_path)?.len();
        let size_gb = model_size as f64 / (1024.0 * 1024.0 * 1024.0);

        Ok(match size_gb {
            size if size < 0.5 => ModelSizeCategory::Tiny,
            size if size < 2.0 => ModelSizeCategory::Small,
            size if size < 15.0 => ModelSizeCategory::Medium,
            size if size < 60.0 => ModelSizeCategory::Large,
            _ => ModelSizeCategory::ExtraLarge,
        })
    }

    pub fn from_config_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }

    pub fn from_environment() -> Self {
        let mut thresholds = Self::new();

        // Override from environment variables
        if let Ok(val) = std::env::var("BITNET_MIN_TOKENS_PER_SEC") {
            if let Ok(parsed) = val.parse::<f64>() {
                thresholds.base_thresholds.min_tokens_per_second = parsed;
            }
        }

        if let Ok(val) = std::env::var("BITNET_MAX_LATENCY_MS") {
            if let Ok(parsed) = val.parse::<f64>() {
                thresholds.base_thresholds.max_latency_ms = parsed;
            }
        }

        if let Ok(val) = std::env::var("BITNET_MAX_MEMORY_MB") {
            if let Ok(parsed) = val.parse::<f64>() {
                thresholds.base_thresholds.max_memory_usage_mb = parsed;
            }
        }

        thresholds
    }
}

impl PerformanceThresholds {
    fn conservative_defaults() -> Self {
        Self {
            min_tokens_per_second: 1.0,
            max_latency_ms: 30000.0,
            max_memory_usage_mb: 2048.0,
            min_speedup_factor: 1.0,
            max_gpu_memory_mb: None,
            min_accuracy_score: 0.85,
            max_compile_time_ms: None,
        }
    }
}
```

## Implementation Plan

### Task 1: Adaptive Threshold System
- [ ] Implement hardware profile detection
- [ ] Create model size categorization system
- [ ] Add environment-aware threshold adjustments
- [ ] Support configuration file loading

### Task 2: Configuration Integration
- [ ] Add TOML/JSON configuration file support
- [ ] Implement environment variable overrides
- [ ] Create configuration validation system
- [ ] Add runtime threshold adjustment APIs

### Task 3: Hardware Detection
- [ ] Implement GPU capability detection
- [ ] Add CPU performance profiling
- [ ] Create memory availability detection
- [ ] Support mobile/embedded platform detection

### Task 4: Testing and Validation
- [ ] Add tests for different hardware profiles
- [ ] Validate threshold calculations
- [ ] Test configuration loading and overrides
- [ ] Benchmark adaptive threshold selection

## Testing Strategy

### Configuration Tests
```rust
#[test]
fn test_adaptive_thresholds() {
    let adaptive = AdaptiveThresholds::new();

    let high_end_thresholds = adaptive.get_thresholds(
        &HardwareProfile::HighEndGPU,
        &ModelSizeCategory::Small,
        &Environment::Production
    );

    let low_end_thresholds = adaptive.get_thresholds(
        &HardwareProfile::LowEndCPU,
        &ModelSizeCategory::Large,
        &Environment::CI
    );

    // High-end should have stricter requirements
    assert!(high_end_thresholds.min_tokens_per_second > low_end_thresholds.min_tokens_per_second);
    assert!(high_end_thresholds.max_latency_ms < low_end_thresholds.max_latency_ms);
}

#[test]
fn test_environment_overrides() {
    let adaptive = AdaptiveThresholds::new();

    let prod_thresholds = adaptive.get_thresholds(
        &HardwareProfile::MidRangeCPU,
        &ModelSizeCategory::Medium,
        &Environment::Production
    );

    let ci_thresholds = adaptive.get_thresholds(
        &HardwareProfile::MidRangeCPU,
        &ModelSizeCategory::Medium,
        &Environment::CI
    );

    // CI should be more lenient
    assert!(ci_thresholds.min_tokens_per_second < prod_thresholds.min_tokens_per_second);
}
```

## Related Issues/PRs

- Part of comprehensive validation framework improvements
- Related to cross-platform compatibility
- Connected to CI/CD pipeline reliability

## Acceptance Criteria

- [ ] Performance thresholds adapt to hardware capabilities
- [ ] Configuration can be loaded from files and environment variables
- [ ] Model size affects threshold calculations appropriately
- [ ] Environment-specific adjustments work correctly
- [ ] Hardware detection provides reasonable profile classification
- [ ] All existing validation tests pass with adaptive thresholds

## Risk Assessment

**Low Risk**: Configuration improvements should not break existing functionality.

**Mitigation Strategies**:
- Provide conservative fallback defaults for unknown configurations
- Implement comprehensive validation for configuration files
- Add gradual rollout mechanism for new threshold systems
- Maintain backwards compatibility with existing hardcoded values
