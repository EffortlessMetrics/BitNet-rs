# [Configuration] Implement comprehensive configurable system to replace all hardcoded values

## Problem Description

Multiple components throughout the codebase contain hardcoded values that should be part of a unified configuration system, including model type checks, performance thresholds, memory limits, and device settings.

## Affected Components

### 1. Model Type Handling (loader.rs)
```rust
if config.model_type.contains("gpt2") {
    policy.add_bos = false;  // GPT-2 doesn't use BOS
}
```

### 2. Performance Thresholds (validation.rs)
```rust
impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            min_tokens_per_second: 10.0,
            max_latency_ms: 5000.0,
            max_memory_usage_mb: 8192.0,
            min_speedup_factor: 1.5,
        }
    }
}
```

## Proposed Solution

### Unified Configuration Architecture

```rust
// Core configuration system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitNetConfiguration {
    pub model: ModelConfiguration,
    pub performance: PerformanceConfiguration,
    pub device: DeviceConfiguration,
    pub inference: InferenceConfiguration,
    pub validation: ValidationConfiguration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfiguration {
    pub type_mappings: HashMap<String, ModelTypeConfig>,
    pub default_policy: ScoringPolicy,
    pub quantization_settings: QuantizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTypeConfig {
    pub add_bos_token: bool,
    pub add_eos_token: bool,
    pub special_tokens: Vec<SpecialToken>,
    pub scoring_policy: ScoringPolicy,
}

impl BitNetConfiguration {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_environment() -> Result<Self> {
        let mut config = Self::default();

        // Model configuration
        if let Ok(model_types) = std::env::var("BITNET_MODEL_TYPES_CONFIG") {
            config.model.type_mappings = serde_json::from_str(&model_types)?;
        }

        // Performance configuration
        if let Ok(min_tps) = std::env::var("BITNET_MIN_TOKENS_PER_SECOND") {
            config.performance.thresholds.min_tokens_per_second = min_tps.parse()?;
        }

        // Device configuration
        if let Ok(device_strategy) = std::env::var("BITNET_DEVICE_STRATEGY") {
            config.device.default_strategy = device_strategy.parse()?;
        }

        Ok(config)
    }

    pub fn merged_with_overrides(self, overrides: ConfigurationOverrides) -> Self {
        // Apply overrides with priority
        let mut config = self;

        if let Some(perf_overrides) = overrides.performance {
            config.performance = config.performance.merge(perf_overrides);
        }

        if let Some(device_overrides) = overrides.device {
            config.device = config.device.merge(device_overrides);
        }

        config
    }
}

// Configuration loading with priority
pub struct ConfigurationLoader {
    search_paths: Vec<PathBuf>,
    environment_prefix: String,
}

impl ConfigurationLoader {
    pub fn new() -> Self {
        Self {
            search_paths: vec![
                PathBuf::from("./bitnet.toml"),
                PathBuf::from("~/.config/bitnet/config.toml"),
                PathBuf::from("/etc/bitnet/config.toml"),
            ],
            environment_prefix: "BITNET_".to_string(),
        }
    }

    pub fn load_configuration(&self) -> Result<BitNetConfiguration> {
        // Priority order:
        // 1. Environment variables
        // 2. Local config file
        // 3. User config file
        // 4. System config file
        // 5. Defaults

        let mut config = BitNetConfiguration::default();

        // Load from files (reverse priority order)
        for path in self.search_paths.iter().rev() {
            if path.exists() {
                let file_config = BitNetConfiguration::from_file(path)?;
                config = config.merge(file_config);
                info!("Loaded configuration from {}", path.display());
            }
        }

        // Apply environment overrides
        let env_config = BitNetConfiguration::from_environment()?;
        config = config.merge(env_config);

        Ok(config)
    }
}
```

### Configuration File Format

```toml
# bitnet.toml

[model]
default_add_bos = true
default_add_eos = false

[model.types.gpt2]
add_bos_token = false
add_eos_token = false
special_tokens = ["<|endoftext|>"]

[model.types.llama]
add_bos_token = true
add_eos_token = false
special_tokens = ["<s>", "</s>"]

[performance.thresholds]
min_tokens_per_second = 10.0
max_latency_ms = 5000.0
max_memory_usage_mb = 8192.0
min_speedup_factor = 1.5

[performance.cpu]
min_tokens_per_second = 8.0
max_memory_usage_mb = 8192.0

[performance.gpu]
min_tokens_per_second = 40.0
max_memory_usage_mb = 24576.0

[device]
default_strategy = "auto"
cpu_threads = "auto"
gpu_memory_fraction = 0.9

[inference]
default_batch_size = 1
max_sequence_length = 2048
enable_kv_cache = true

[validation]
strict_mode = false
cross_validation_enabled = true
```

## Implementation Plan

### Phase 1: Core Configuration System (Week 1)
- [ ] Design configuration schema and data structures
- [ ] Implement configuration loading from files and environment
- [ ] Add configuration validation and error handling
- [ ] Create configuration merging and override system

### Phase 2: Component Integration (Week 2)
- [ ] Replace hardcoded model type checks with configuration
- [ ] Convert performance thresholds to configurable system
- [ ] Update device configuration to use unified system
- [ ] Integrate inference parameters with configuration

### Phase 3: Advanced Features (Week 3)
- [ ] Add runtime configuration updates
- [ ] Implement configuration hot-reloading
- [ ] Add configuration validation and schema checking
- [ ] Create configuration migration utilities

### Phase 4: Documentation and Testing (Week 4)
- [ ] Create comprehensive configuration documentation
- [ ] Add configuration examples for common scenarios
- [ ] Implement configuration testing framework
- [ ] Add validation for configuration compatibility

## Acceptance Criteria

- [ ] No hardcoded values remain in component implementations
- [ ] Configuration loaded from multiple sources with clear priority
- [ ] Model type handling completely configurable
- [ ] Performance thresholds adapt to hardware and deployment context
- [ ] Configuration changes don't require code recompilation
- [ ] Comprehensive validation and error reporting
- [ ] Backward compatibility maintained

## Priority: Medium

Significantly improves system flexibility and maintainability while reducing technical debt from scattered hardcoded values.