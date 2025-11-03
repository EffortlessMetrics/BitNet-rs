# [Configuration] Replace hardcoded values with configurable system

## Problem Description

Several critical components contain hardcoded values that should be configurable to support different models, hardware configurations, and deployment scenarios. This reduces system flexibility and adaptability.

## Environment
- **Affected Files**:
  - `crates/bitnet-inference/src/loader.rs` - GPT-2 model type check
  - `crates/bitnet-inference/src/validation.rs` - Performance thresholds
- **Impact**: Model compatibility, performance validation, deployment flexibility

## Issues Identified

### 1. Hardcoded Model Type Check (`loader.rs`)
**Location**: `crates/bitnet-inference/src/loader.rs`
```rust
// Model-specific overrides
if config.model_type.contains("gpt2") {
    policy.add_bos = false;  // GPT-2 doesn't use BOS
}
```

**Problem**: Only handles GPT-2 specifically, not extensible to other model types.

### 2. Hardcoded Performance Thresholds (`validation.rs`)
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

**Problem**: Fixed thresholds don't account for different hardware capabilities or model sizes.

## Root Cause Analysis

1. **Limited Model Support**: Hardcoded model type checks prevent supporting new architectures
2. **Hardware Assumptions**: Performance thresholds assume specific hardware capabilities
3. **Deployment Inflexibility**: Fixed values don't adapt to different environments
4. **Maintenance Overhead**: Each new model type requires code changes

## Impact Assessment
- **Severity**: Medium-High
- **Impact**:
  - Limited model compatibility
  - Inappropriate performance expectations on different hardware
  - Reduced deployment flexibility
  - Maintenance burden for new model support
- **Affected Components**: Model loading, validation system, scoring policies

## Proposed Solution

Implement a comprehensive configuration system that replaces hardcoded values with configurable, model-specific, and environment-aware settings.

### Implementation Plan

#### 1. Model Configuration System

**A. Extend ModelConfig Structure** (`bitnet-models/src/config.rs`):
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Existing fields...

    // Tokenization behavior
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub pad_token_id: Option<u32>,

    // Model-specific scoring policies
    pub scoring_policy: Option<ScoringPolicy>,

    // Architecture-specific settings
    pub architecture_hints: ArchitectureHints,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureHints {
    pub model_family: ModelFamily,
    pub attention_type: AttentionType,
    pub normalization_type: NormalizationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFamily {
    Gpt2,
    Llama,
    BitNet,
    Custom(String),
}
```

**B. Update Loader Logic** (`bitnet-inference/src/loader.rs`):
```rust
fn determine_scoring_policy(&self, model: &Arc<dyn Model>, tokenizer: &Arc<dyn Tokenizer>) -> ScoringPolicy {
    let config = model.config();
    let mut policy = ScoringPolicy::default();

    // Use model-specific configuration if available
    if let Some(model_policy) = &config.scoring_policy {
        return model_policy.clone();
    }

    // Use architecture-aware defaults
    match config.architecture_hints.model_family {
        ModelFamily::Gpt2 => {
            policy.add_bos = false;  // GPT-2 doesn't use BOS
        },
        ModelFamily::Llama => {
            policy.add_bos = true;   // Llama uses BOS
        },
        ModelFamily::BitNet => {
            policy.add_bos = config.add_bos_token.unwrap_or(true);
        },
        ModelFamily::Custom(ref name) => {
            // Load from configuration database or use safe defaults
            policy.add_bos = config.add_bos_token.unwrap_or(true);
            warn!("Using default scoring policy for custom model family: {}", name);
        }
    }

    policy
}
```

#### 2. Configurable Performance Thresholds

**A. Environment-Aware Thresholds** (`bitnet-inference/src/validation.rs`):
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub min_tokens_per_second: f64,
    pub max_latency_ms: f64,
    pub max_memory_usage_mb: f64,
    pub min_speedup_factor: f64,
}

impl PerformanceThresholds {
    pub fn from_config(config: &ValidationConfig) -> Self {
        Self {
            min_tokens_per_second: config.min_tokens_per_second.unwrap_or_else(|| {
                Self::default_tokens_per_second_for_device(&config.device_type)
            }),
            max_latency_ms: config.max_latency_ms.unwrap_or_else(|| {
                Self::default_latency_for_model_size(config.model_size_gb)
            }),
            max_memory_usage_mb: config.max_memory_usage_mb.unwrap_or_else(|| {
                Self::default_memory_for_device(&config.device_type)
            }),
            min_speedup_factor: config.min_speedup_factor.unwrap_or(1.2),
        }
    }

    fn default_tokens_per_second_for_device(device: &DeviceType) -> f64 {
        match device {
            DeviceType::Cpu => 10.0,
            DeviceType::Gpu => 50.0,
            DeviceType::Auto => 25.0,
        }
    }

    fn default_latency_for_model_size(model_size_gb: f64) -> f64 {
        // Scale latency expectations with model size
        (model_size_gb * 1000.0).max(2000.0).min(10000.0)
    }

    fn default_memory_for_device(device: &DeviceType) -> f64 {
        match device {
            DeviceType::Cpu => 8192.0,   // 8GB for CPU
            DeviceType::Gpu => 16384.0,  // 16GB for GPU
            DeviceType::Auto => 12288.0, // 12GB average
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub device_type: DeviceType,
    pub model_size_gb: f64,
    pub min_tokens_per_second: Option<f64>,
    pub max_latency_ms: Option<f64>,
    pub max_memory_usage_mb: Option<f64>,
    pub min_speedup_factor: Option<f64>,
}
```

#### 3. Configuration Loading System

**A. Configuration Sources** (Priority order):
1. Explicit API parameters
2. Model-embedded configuration
3. Environment variables
4. Configuration files
5. Hardware-detected defaults

**B. Environment Variable Support**:
```rust
// Performance thresholds
BITNET_MIN_TOKENS_PER_SECOND=20.0
BITNET_MAX_LATENCY_MS=3000
BITNET_MAX_MEMORY_MB=16384
BITNET_MIN_SPEEDUP_FACTOR=1.8

// Model behavior
BITNET_ADD_BOS_TOKEN=true
BITNET_ADD_EOS_TOKEN=false
```

**C. Configuration File Support** (`bitnet_config.toml`):
```toml
[model]
add_bos_token = true
add_eos_token = false

[performance.thresholds]
min_tokens_per_second = 15.0
max_latency_ms = 4000
max_memory_usage_mb = 12288
min_speedup_factor = 1.5

[performance.cpu]
min_tokens_per_second = 8.0
max_memory_usage_mb = 8192

[performance.gpu]
min_tokens_per_second = 40.0
max_memory_usage_mb = 24576
```

## Testing Strategy
- **Configuration Loading Tests**: Verify all configuration sources work correctly
- **Model Compatibility Tests**: Test scoring policies with different model types
- **Performance Threshold Tests**: Verify thresholds adapt to hardware configuration
- **Backward Compatibility Tests**: Ensure existing models continue to work
- **Environment Variable Tests**: Test environment-based configuration
- **Configuration File Tests**: Test TOML/JSON configuration loading

## Implementation Tasks
- [ ] Design configuration schema and data structures
- [ ] Implement ModelConfig extensions with architecture hints
- [ ] Update scoring policy determination logic
- [ ] Implement configurable performance thresholds
- [ ] Add environment variable support
- [ ] Add configuration file loading
- [ ] Create configuration validation
- [ ] Update existing hardcoded usage sites
- [ ] Add configuration documentation and examples
- [ ] Implement configuration migration utilities

## Acceptance Criteria
- [ ] No hardcoded model type checks remain
- [ ] Performance thresholds adapt to hardware and model characteristics
- [ ] Configuration can be specified via multiple sources (API, env vars, files)
- [ ] Model loading works correctly for GPT-2, Llama, and BitNet architectures
- [ ] Performance validation uses appropriate thresholds for deployment environment
- [ ] Configuration system is well-documented with examples
- [ ] Backward compatibility maintained for existing usage
- [ ] Configuration validation provides helpful error messages

## Migration Guide
- Document configuration migration for existing deployments
- Provide configuration examples for common scenarios
- Create tooling to generate configuration from existing hardcoded values

## Benefits After Implementation
- **Multi-Model Support**: Easy addition of new model architectures
- **Hardware Adaptability**: Performance expectations match hardware capabilities
- **Deployment Flexibility**: Configuration adapts to different environments
- **Maintenance Reduction**: New models don't require code changes
- **Better Testing**: Configurable thresholds enable comprehensive testing

## Labels
- `configuration`
- `enhancement`
- `flexibility`
- `priority-medium`
- `breaking-change`

## Related Issues
- Model compatibility improvements
- Performance validation enhancements
- Configuration system architecture
