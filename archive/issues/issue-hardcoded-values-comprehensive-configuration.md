# [Configuration] Replace Hardcoded Values with Comprehensive Configuration System

## Problem Description

Several critical components in BitNet-rs contain hardcoded values that should be configurable to support different models, hardware configurations, and deployment scenarios. This reduces system flexibility, makes it difficult to support new model architectures, and prevents optimization for different hardware environments.

## Environment

- **Affected Files**:
  - `crates/bitnet-inference/src/loader.rs` - GPT-2 model type check
  - `crates/bitnet-inference/src/validation.rs` - Performance thresholds
  - Various model loading and validation components
- **Impact**: Model compatibility, performance validation, deployment flexibility

## Current Implementation Analysis

### 1. Hardcoded Model Type Check
**Location**: `crates/bitnet-inference/src/loader.rs`
```rust
// Model-specific overrides
if config.model_type.contains("gpt2") {
    policy.add_bos = false;  // GPT-2 doesn't use BOS
}
```

### 2. Hardcoded Performance Thresholds
**Location**: `crates/bitnet-inference/src/validation.rs`
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

**Issues Identified:**
1. **Limited extensibility**: Only handles specific model types like GPT-2
2. **Hardware assumptions**: Fixed thresholds don't account for device capabilities
3. **Deployment inflexibility**: Values don't adapt to different environments
4. **Maintenance overhead**: Each new model type requires code changes

## Impact Assessment

**Severity**: Medium-High
**Affected Users**: All users, especially those with non-standard hardware or custom models
**Functional Impact**:
- Limited model compatibility beyond hardcoded types
- Inappropriate performance expectations on different hardware
- Reduced deployment flexibility across environments
- Maintenance burden for supporting new architectures

## Root Cause Analysis

The current approach uses hardcoded values instead of a flexible configuration system that can adapt to:
1. **Different model architectures**: GPT-2, LLaMA, BitNet variants
2. **Various hardware configurations**: CPU-only, GPU, mixed deployments
3. **Different deployment scenarios**: Development, production, edge devices
4. **Custom model implementations**: User-defined architectures

## Proposed Solution

### 1. Comprehensive Configuration Architecture

Implement a layered configuration system with multiple sources and intelligent defaults:

```rust
// crates/bitnet-common/src/config/mod.rs
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitNetConfiguration {
    pub model: ModelConfiguration,
    pub performance: PerformanceConfiguration,
    pub deployment: DeploymentConfiguration,
    pub validation: ValidationConfiguration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfiguration {
    pub architecture_hints: ArchitectureHints,
    pub tokenization: TokenizationConfig,
    pub scoring_policy: Option<ScoringPolicyConfig>,
    pub quantization: QuantizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureHints {
    pub model_family: ModelFamily,
    pub attention_type: AttentionType,
    pub normalization_type: NormalizationType,
    pub layer_types: Vec<LayerType>,
    pub custom_attributes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFamily {
    Gpt2,
    Llama,
    BitNet,
    Mistral,
    Phi,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizationConfig {
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub special_tokens: HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfiguration {
    pub thresholds: PerformanceThresholds,
    pub device_specific: HashMap<DeviceType, DevicePerformanceConfig>,
    pub adaptive: AdaptivePerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub min_tokens_per_second: Option<f64>,
    pub max_latency_ms: Option<f64>,
    pub max_memory_usage_mb: Option<f64>,
    pub min_speedup_factor: Option<f64>,
    pub max_cpu_utilization: Option<f64>,
    pub max_gpu_memory_usage: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevicePerformanceConfig {
    pub optimal_batch_size: Option<usize>,
    pub thread_count: Option<usize>,
    pub memory_limit_mb: Option<f64>,
    pub kernel_preferences: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePerformanceConfig {
    pub enable_auto_tuning: bool,
    pub learning_rate: f64,
    pub adaptation_interval_seconds: u64,
    pub performance_history_size: usize,
}
```

### 2. Configuration Loading System

```rust
// crates/bitnet-common/src/config/loader.rs
use anyhow::{Context, Result};
use std::path::Path;

pub struct ConfigurationLoader {
    sources: Vec<Box<dyn ConfigurationSource>>,
    cache: Option<BitNetConfiguration>,
}

impl ConfigurationLoader {
    pub fn new() -> Self {
        Self {
            sources: vec![
                Box::new(EnvironmentVariableSource::new()),
                Box::new(ConfigFileSource::new()),
                Box::new(ModelEmbeddedSource::new()),
                Box::new(HardwareDetectionSource::new()),
                Box::new(DefaultsSource::new()),
            ],
            cache: None,
        }
    }

    pub fn load_configuration(&mut self, model_path: Option<&Path>) -> Result<BitNetConfiguration> {
        if let Some(cached) = &self.cache {
            return Ok(cached.clone());
        }

        let mut config = BitNetConfiguration::default();

        // Apply configuration from each source in priority order
        for source in &self.sources {
            let partial_config = source.load_configuration(model_path)?;
            config = self.merge_configurations(config, partial_config)?;
        }

        // Validate the final configuration
        self.validate_configuration(&config)?;

        self.cache = Some(config.clone());
        Ok(config)
    }

    fn merge_configurations(
        &self,
        base: BitNetConfiguration,
        overlay: PartialBitNetConfiguration,
    ) -> Result<BitNetConfiguration> {
        let mut merged = base;

        // Merge model configuration
        if let Some(model_config) = overlay.model {
            merged.model = self.merge_model_config(merged.model, model_config)?;
        }

        // Merge performance configuration
        if let Some(perf_config) = overlay.performance {
            merged.performance = self.merge_performance_config(merged.performance, perf_config)?;
        }

        Ok(merged)
    }

    fn validate_configuration(&self, config: &BitNetConfiguration) -> Result<()> {
        // Validate performance thresholds are reasonable
        if let Some(tokens_per_sec) = config.performance.thresholds.min_tokens_per_second {
            if tokens_per_sec <= 0.0 {
                return Err(anyhow::anyhow!("min_tokens_per_second must be positive"));
            }
        }

        // Validate memory limits
        if let Some(memory_mb) = config.performance.thresholds.max_memory_usage_mb {
            if memory_mb < 512.0 {
                return Err(anyhow::anyhow!("max_memory_usage_mb must be at least 512MB"));
            }
        }

        // Validate tokenization configuration
        if let Some(bos) = config.model.tokenization.add_bos_token {
            if let Some(eos) = config.model.tokenization.add_eos_token {
                // Warn about potentially problematic combinations
                if !bos && !eos {
                    warn!("Neither BOS nor EOS tokens enabled - this may affect generation quality");
                }
            }
        }

        Ok(())
    }
}

trait ConfigurationSource {
    fn load_configuration(&self, model_path: Option<&Path>) -> Result<PartialBitNetConfiguration>;
    fn priority(&self) -> u8; // Higher number = higher priority
}
```

### 3. Model-Specific Configuration Resolution

```rust
// crates/bitnet-inference/src/loader.rs - Updated implementation
impl ModelLoader {
    fn determine_scoring_policy(&self, model: &Arc<dyn Model>) -> Result<ScoringPolicy> {
        let config = model.config();
        let mut policy = ScoringPolicy::default();

        // Use explicit scoring policy if provided
        if let Some(scoring_config) = &config.scoring_policy {
            return Ok(ScoringPolicy::from_config(scoring_config));
        }

        // Use architecture-aware configuration
        match config.architecture_hints.model_family {
            ModelFamily::Gpt2 => {
                policy.add_bos = config.tokenization.add_bos_token.unwrap_or(false);
                policy.add_eos = config.tokenization.add_eos_token.unwrap_or(true);
                policy.temperature = 1.0;
                policy.top_p = 0.9;
            }
            ModelFamily::Llama => {
                policy.add_bos = config.tokenization.add_bos_token.unwrap_or(true);
                policy.add_eos = config.tokenization.add_eos_token.unwrap_or(false);
                policy.temperature = 0.7;
                policy.top_p = 0.95;
            }
            ModelFamily::BitNet => {
                policy.add_bos = config.tokenization.add_bos_token.unwrap_or(true);
                policy.add_eos = config.tokenization.add_eos_token.unwrap_or(true);
                policy.temperature = 0.8;
                policy.top_p = 0.9;
            }
            ModelFamily::Custom(ref name) => {
                // Load from configuration database or use safe defaults
                policy = self.load_custom_scoring_policy(name)
                    .unwrap_or_else(|e| {
                        warn!("Failed to load scoring policy for {}: {}, using defaults", name, e);
                        ScoringPolicy::conservative_defaults()
                    });
            }
            _ => {
                // Use safe defaults for unknown architectures
                policy = ScoringPolicy::conservative_defaults();
            }
        }

        Ok(policy)
    }

    fn load_custom_scoring_policy(&self, model_name: &str) -> Result<ScoringPolicy> {
        // Try to load from model-specific configuration file
        let config_path = format!("models/{}/scoring_policy.toml", model_name);
        if Path::new(&config_path).exists() {
            let config_str = std::fs::read_to_string(&config_path)?;
            let policy_config: ScoringPolicyConfig = toml::from_str(&config_str)?;
            return Ok(ScoringPolicy::from_config(&policy_config));
        }

        // Try to load from environment variable
        if let Ok(policy_json) = std::env::var(&format!("BITNET_SCORING_POLICY_{}", model_name.to_uppercase())) {
            let policy_config: ScoringPolicyConfig = serde_json::from_str(&policy_json)?;
            return Ok(ScoringPolicy::from_config(&policy_config));
        }

        Err(anyhow::anyhow!("No custom scoring policy found for model: {}", model_name))
    }
}
```

### 4. Hardware-Adaptive Performance Thresholds

```rust
// crates/bitnet-inference/src/validation.rs - Updated implementation
impl PerformanceThresholds {
    pub fn from_configuration(
        config: &PerformanceConfiguration,
        hardware_info: &HardwareInfo,
        model_info: &ModelInfo,
    ) -> Self {
        let device_config = config.device_specific
            .get(&hardware_info.primary_device_type)
            .cloned()
            .unwrap_or_default();

        Self {
            min_tokens_per_second: config.thresholds.min_tokens_per_second
                .or(device_config.calculate_min_tokens_per_second(hardware_info, model_info))
                .unwrap_or_else(|| Self::default_tokens_per_second(hardware_info)),

            max_latency_ms: config.thresholds.max_latency_ms
                .unwrap_or_else(|| Self::calculate_max_latency(model_info, hardware_info)),

            max_memory_usage_mb: config.thresholds.max_memory_usage_mb
                .or(device_config.memory_limit_mb)
                .unwrap_or_else(|| Self::calculate_memory_limit(hardware_info)),

            min_speedup_factor: config.thresholds.min_speedup_factor
                .unwrap_or_else(|| Self::calculate_expected_speedup(hardware_info)),
        }
    }

    fn default_tokens_per_second(hardware_info: &HardwareInfo) -> f64 {
        match hardware_info.primary_device_type {
            DeviceType::Cpu => {
                // Scale with CPU cores and frequency
                let base_rate = 2.0; // tokens/sec per core
                let cores = hardware_info.cpu_cores as f64;
                let frequency_factor = hardware_info.cpu_frequency_ghz / 2.0; // 2GHz baseline
                base_rate * cores * frequency_factor
            }
            DeviceType::Gpu => {
                // Scale with GPU memory and compute capability
                let base_rate = match hardware_info.gpu_info.as_ref() {
                    Some(gpu) if gpu.has_tensor_cores => 80.0,
                    Some(_) => 40.0,
                    None => 20.0,
                };
                base_rate * hardware_info.gpu_info.as_ref()
                    .map(|g| (g.memory_gb / 8.0).max(1.0))
                    .unwrap_or(1.0)
            }
            DeviceType::Auto => 25.0, // Conservative estimate
        }
    }

    fn calculate_max_latency(model_info: &ModelInfo, hardware_info: &HardwareInfo) -> f64 {
        // Base latency scales with model size
        let base_latency = match model_info.parameter_count {
            p if p < 1_000_000_000 => 1000.0,      // < 1B params: 1s
            p if p < 10_000_000_000 => 3000.0,     // < 10B params: 3s
            _ => 5000.0,                            // >= 10B params: 5s
        };

        // Adjust for hardware capabilities
        let hardware_factor = match hardware_info.primary_device_type {
            DeviceType::Gpu => 0.5,  // GPUs are faster
            DeviceType::Cpu => 1.5,  // CPUs are slower
            DeviceType::Auto => 1.0,
        };

        base_latency * hardware_factor
    }

    fn calculate_memory_limit(hardware_info: &HardwareInfo) -> f64 {
        match hardware_info.primary_device_type {
            DeviceType::Cpu => {
                // Use 75% of available system memory
                hardware_info.total_memory_gb * 1024.0 * 0.75
            }
            DeviceType::Gpu => {
                // Use 90% of GPU memory
                hardware_info.gpu_info.as_ref()
                    .map(|g| g.memory_gb * 1024.0 * 0.9)
                    .unwrap_or(8192.0) // 8GB fallback
            }
            DeviceType::Auto => {
                // Conservative estimate
                8192.0
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub primary_device_type: DeviceType,
    pub cpu_cores: usize,
    pub cpu_frequency_ghz: f64,
    pub total_memory_gb: f64,
    pub gpu_info: Option<GpuInfo>,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub memory_gb: f64,
    pub compute_capability: (u32, u32),
    pub has_tensor_cores: bool,
    pub multiprocessor_count: u32,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub parameter_count: u64,
    pub model_size_gb: f64,
    pub architecture: ModelFamily,
    pub quantization_type: Option<QuantizationType>,
}
```

### 5. Configuration File Support

```toml
# bitnet_config.toml
[model]
family = "bitnet"

[model.tokenization]
add_bos_token = true
add_eos_token = true
pad_token_id = 0

[model.architecture_hints]
model_family = "BitNet"
attention_type = "MultiHead"
normalization_type = "RMSNorm"

[performance.thresholds]
min_tokens_per_second = 15.0
max_latency_ms = 4000
max_memory_usage_mb = 12288
min_speedup_factor = 1.5

[performance.device_specific.cpu]
optimal_batch_size = 1
thread_count = 8
memory_limit_mb = 8192

[performance.device_specific.gpu]
optimal_batch_size = 4
memory_limit_mb = 16384

[performance.adaptive]
enable_auto_tuning = true
learning_rate = 0.1
adaptation_interval_seconds = 60
```

### 6. Environment Variable Support

```rust
// Environment variables with fallback hierarchy
pub struct EnvironmentVariableSource;

impl ConfigurationSource for EnvironmentVariableSource {
    fn load_configuration(&self, _model_path: Option<&Path>) -> Result<PartialBitNetConfiguration> {
        let mut config = PartialBitNetConfiguration::default();

        // Performance thresholds
        if let Ok(val) = std::env::var("BITNET_MIN_TOKENS_PER_SECOND") {
            config.performance.get_or_insert_default().thresholds.min_tokens_per_second =
                Some(val.parse().context("Invalid BITNET_MIN_TOKENS_PER_SECOND")?);
        }

        // Model configuration
        if let Ok(val) = std::env::var("BITNET_ADD_BOS_TOKEN") {
            config.model.get_or_insert_default().tokenization.add_bos_token =
                Some(val.parse().context("Invalid BITNET_ADD_BOS_TOKEN")?);
        }

        // Device-specific overrides
        if let Ok(device_type) = std::env::var("BITNET_DEVICE_TYPE") {
            let device = device_type.parse().context("Invalid BITNET_DEVICE_TYPE")?;

            if let Ok(memory_limit) = std::env::var("BITNET_MEMORY_LIMIT_MB") {
                config.performance.get_or_insert_default()
                    .device_specific.entry(device)
                    .or_default()
                    .memory_limit_mb = Some(memory_limit.parse()?);
            }
        }

        Ok(config)
    }

    fn priority(&self) -> u8 { 90 } // High priority
}
```

## Implementation Breakdown

### Phase 1: Core Configuration Infrastructure
- [ ] Design configuration schema and data structures
- [ ] Implement configuration loading and merging system
- [ ] Add validation framework
- [ ] Create basic configuration sources

### Phase 2: Model-Specific Configuration
- [ ] Extend ModelConfig with architecture hints
- [ ] Update scoring policy determination logic
- [ ] Implement custom model configuration support
- [ ] Add model-specific defaults

### Phase 3: Performance Configuration
- [ ] Implement hardware-adaptive thresholds
- [ ] Add device-specific configuration support
- [ ] Implement adaptive performance tuning
- [ ] Add performance monitoring integration

### Phase 4: Configuration Sources
- [ ] Implement environment variable source
- [ ] Add configuration file support (TOML/JSON)
- [ ] Create model-embedded configuration loader
- [ ] Add hardware detection source

### Phase 5: Integration and Migration
- [ ] Update all hardcoded value usage sites
- [ ] Add configuration documentation
- [ ] Create migration utilities
- [ ] Implement backward compatibility

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_configuration_loading_priority() {
        // Set up test environment
        std::env::set_var("BITNET_MIN_TOKENS_PER_SECOND", "25.0");

        let mut loader = ConfigurationLoader::new();
        let config = loader.load_configuration(None).unwrap();

        // Environment variable should override defaults
        assert_eq!(config.performance.thresholds.min_tokens_per_second, Some(25.0));
    }

    #[test]
    fn test_model_family_scoring_policy() {
        let mut model_config = ModelConfiguration::default();
        model_config.architecture_hints.model_family = ModelFamily::Gpt2;

        let loader = ModelLoader::new();
        let policy = loader.determine_scoring_policy_from_config(&model_config).unwrap();

        // GPT-2 should not use BOS token by default
        assert_eq!(policy.add_bos, false);
    }

    #[test]
    fn test_hardware_adaptive_thresholds() {
        let hardware_info = HardwareInfo {
            primary_device_type: DeviceType::Gpu,
            gpu_info: Some(GpuInfo {
                memory_gb: 24.0,
                has_tensor_cores: true,
                ..Default::default()
            }),
            ..Default::default()
        };

        let model_info = ModelInfo {
            parameter_count: 7_000_000_000,
            ..Default::default()
        };

        let config = PerformanceConfiguration::default();
        let thresholds = PerformanceThresholds::from_configuration(&config, &hardware_info, &model_info);

        // GPU with tensor cores should have higher performance expectations
        assert!(thresholds.min_tokens_per_second > 50.0);
    }
}
```

### Integration Tests
```rust
#[test]
fn test_end_to_end_configuration() {
    // Test complete configuration loading and application
    let config_content = r#"
        [model.tokenization]
        add_bos_token = false

        [performance.thresholds]
        min_tokens_per_second = 20.0
    "#;

    let temp_file = create_temp_config_file(config_content);
    std::env::set_var("BITNET_CONFIG_FILE", temp_file.path());

    let mut loader = ConfigurationLoader::new();
    let config = loader.load_configuration(None).unwrap();

    assert_eq!(config.model.tokenization.add_bos_token, Some(false));
    assert_eq!(config.performance.thresholds.min_tokens_per_second, Some(20.0));
}
```

## Risk Assessment

**Low Risk Changes:**
- Adding new configuration structures
- Implementing configuration loading infrastructure

**Medium Risk Changes:**
- Replacing hardcoded values with configuration lookups
- Changing default behavior based on hardware detection

**High Risk Changes:**
- Modifying core model loading logic
- Changing performance validation behavior

**Mitigation Strategies:**
- Comprehensive backward compatibility testing
- Gradual rollout with feature flags
- Extensive validation of configuration values
- Clear migration documentation

## Acceptance Criteria

- [ ] No hardcoded model type checks remain in the codebase
- [ ] Performance thresholds automatically adapt to hardware capabilities
- [ ] Configuration can be specified via multiple sources with clear priority
- [ ] All existing models (GPT-2, LLaMA, BitNet) work without configuration changes
- [ ] New model architectures can be added without code changes
- [ ] Configuration validation provides helpful error messages
- [ ] Performance regression < 5% compared to hardcoded implementation
- [ ] Comprehensive documentation with examples for all configuration options

## Related Issues/PRs

- **Related to**: Model compatibility improvements
- **Depends on**: Hardware detection infrastructure
- **Blocks**: Support for new model architectures
- **References**: Performance optimization framework

## Additional Context

This comprehensive configuration system is essential for making BitNet-rs truly flexible and maintainable. By replacing hardcoded values with intelligent, hardware-aware configuration, the system becomes more adaptable to different deployment scenarios and easier to extend with new model architectures. The multi-source configuration approach ensures that users can customize behavior at the appropriate level while maintaining sensible defaults.
