# [VALIDATION] Make Performance Thresholds Configurable in `validation.rs` for Hardware-Adaptive Testing

## Problem Description

The `PerformanceThresholds` struct in `crates/bitnet-inference/src/validation.rs` contains hardcoded default values that do not adapt to different hardware configurations, model sizes, or deployment contexts. This creates inflexible validation criteria that may be too restrictive for low-end hardware or too lenient for high-performance systems.

**Current Issues:**
- Fixed thresholds (`min_tokens_per_second: 10.0`, `max_latency_ms: 5000.0`, etc.) don't account for hardware diversity
- No configuration mechanism for different deployment scenarios (development, staging, production)
- Limited adaptability for different model sizes and quantization methods
- Potential false positives/negatives in CI/CD pipelines across different hardware configurations

## Environment

**Affected Components:**
- `crates/bitnet-inference/src/validation.rs` (Primary)
- `crates/bitnet-common/src/config.rs` (Configuration integration)
- `docs/environment-variables.md` (Documentation updates)

**Hardware Context:**
- CPU-only inference on various architectures (AVX2, AVX-512, ARM64 NEON)
- GPU acceleration across different vendors (NVIDIA CUDA, AMD ROCm, Apple Metal)
- Memory-constrained environments vs high-memory servers
- Development laptops vs production inference servers

## Reproduction Steps

1. **Examine hardcoded values:**
   ```bash
   cd /home/steven/code/Rust/BitNet-rs
   grep -n "min_tokens_per_second\|max_latency_ms\|max_memory_usage_mb\|min_speedup_factor" crates/bitnet-inference/src/validation.rs
   ```

2. **Observe inflexibility in different scenarios:**
   ```bash
   # Low-end hardware may fail these thresholds unnecessarily
   BITNET_STRICT_MODE=1 cargo test -p bitnet-inference test_validation_comprehensive

   # High-end hardware may pass despite suboptimal performance
   cargo run -p xtask -- crossval --model model.gguf
   ```

3. **Identify configuration gap:**
   ```bash
   # No environment variables exist for performance threshold configuration
   grep -r "BITNET.*THRESHOLD\|BITNET.*PERFORMANCE" crates/ docs/
   ```

## Root Cause Analysis

**Technical Investigation:**

1. **Hardcoded Default Implementation:**
   ```rust
   impl Default for PerformanceThresholds {
       fn default() -> Self {
           Self {
               min_tokens_per_second: 10.0,        // Fixed for all hardware
               max_latency_ms: 5000.0,             // No model size consideration
               max_memory_usage_mb: 8192.0,        // Static memory limit
               min_speedup_factor: 1.5,            // Fixed baseline comparison
           }
       }
   }
   ```

2. **Lack of Configuration Integration:**
   - No connection to existing `BitNetConfig` system in `bitnet-common`
   - Missing environment variable support following established `BITNET_*` patterns
   - No file-based configuration option for different deployment profiles

3. **Architectural Gap:**
   - `ValidationConfig` contains `PerformanceThresholds` but no dynamic loading mechanism
   - No hardware introspection or adaptive threshold calculation
   - Missing integration with device capability detection

## Impact Assessment

**Severity:** Medium-High
**Affected Users:** Developers, CI/CD systems, production deployments

**Business Impact:**
- **Development Friction:** Developers on different hardware configurations experience inconsistent validation results
- **CI/CD Reliability:** Build pipelines may fail or pass incorrectly based on runner hardware specifications
- **Production Deployment:** Inflexible thresholds prevent proper performance validation in diverse deployment environments

**Technical Impact:**
- **False Negatives:** High-end hardware may pass validation despite suboptimal performance tuning
- **False Positives:** Low-end or constrained environments fail validation despite acceptable performance for their context
- **Maintenance Overhead:** Manual threshold adjustment requires code changes rather than configuration updates

## Proposed Solution

### Primary Implementation: Environment-Based Configuration

**Core Architecture:**
1. **Extend Configuration System:** Integrate performance thresholds into existing `BitNetConfig` ecosystem
2. **Environment Variables:** Add `BITNET_PERF_*` environment variables following established patterns
3. **Configuration Files:** Support threshold specification in TOML/JSON configuration files
4. **Hardware Adaptation:** Optional automatic threshold adjustment based on detected hardware capabilities

**Implementation Details:**

```rust
// 1. Enhanced PerformanceThresholds with configuration support
impl PerformanceThresholds {
    /// Create from configuration with fallback to defaults
    pub fn from_config(config: &ValidationConfig) -> Self {
        Self {
            min_tokens_per_second: config.performance_thresholds
                .as_ref()
                .and_then(|t| t.min_tokens_per_second)
                .unwrap_or_else(Self::default_min_tokens_per_second),
            max_latency_ms: config.performance_thresholds
                .as_ref()
                .and_then(|t| t.max_latency_ms)
                .unwrap_or_else(Self::default_max_latency_ms),
            max_memory_usage_mb: config.performance_thresholds
                .as_ref()
                .and_then(|t| t.max_memory_usage_mb)
                .unwrap_or_else(Self::default_max_memory_usage_mb),
            min_speedup_factor: config.performance_thresholds
                .as_ref()
                .and_then(|t| t.min_speedup_factor)
                .unwrap_or_else(Self::default_min_speedup_factor),
        }
    }

    /// Hardware-adaptive defaults with fallback to constants
    fn default_min_tokens_per_second() -> f64 {
        std::env::var("BITNET_PERF_MIN_TOKENS_PER_SECOND")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                // Hardware-adaptive logic
                let detected_cores = num_cpus::get();
                if detected_cores >= 16 {
                    20.0  // High-end hardware
                } else if detected_cores >= 8 {
                    15.0  // Mid-range hardware
                } else {
                    10.0  // Default for lower-end hardware
                }
            })
    }

    fn default_max_latency_ms() -> f64 {
        std::env::var("BITNET_PERF_MAX_LATENCY_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5000.0)
    }

    fn default_max_memory_usage_mb() -> f64 {
        std::env::var("BITNET_PERF_MAX_MEMORY_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                // Adaptive based on available system memory
                let available_memory = get_available_memory_mb().unwrap_or(16384);
                (available_memory as f64 * 0.5).min(8192.0) // Use up to 50% of available memory, capped at 8GB
            })
    }

    fn default_min_speedup_factor() -> f64 {
        std::env::var("BITNET_PERF_MIN_SPEEDUP_FACTOR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.5)
    }
}

// 2. Configuration file support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurablePerformanceThresholds {
    pub min_tokens_per_second: Option<f64>,
    pub max_latency_ms: Option<f64>,
    pub max_memory_usage_mb: Option<f64>,
    pub min_speedup_factor: Option<f64>,
}

// 3. Validation profile support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationProfile {
    Development,    // Relaxed thresholds for dev machines
    CI,            // Moderate thresholds for CI runners
    Staging,       // Production-like thresholds
    Production,    // Strict thresholds for deployment validation
    Custom(ConfigurablePerformanceThresholds),
}

impl ValidationProfile {
    pub fn to_thresholds(&self) -> PerformanceThresholds {
        match self {
            ValidationProfile::Development => PerformanceThresholds {
                min_tokens_per_second: 5.0,
                max_latency_ms: 10000.0,
                max_memory_usage_mb: 16384.0,
                min_speedup_factor: 1.2,
            },
            ValidationProfile::CI => PerformanceThresholds {
                min_tokens_per_second: 8.0,
                max_latency_ms: 7500.0,
                max_memory_usage_mb: 12288.0,
                min_speedup_factor: 1.3,
            },
            ValidationProfile::Staging => PerformanceThresholds {
                min_tokens_per_second: 12.0,
                max_latency_ms: 5000.0,
                max_memory_usage_mb: 8192.0,
                min_speedup_factor: 1.5,
            },
            ValidationProfile::Production => PerformanceThresholds {
                min_tokens_per_second: 15.0,
                max_latency_ms: 3000.0,
                max_memory_usage_mb: 6144.0,
                min_speedup_factor: 2.0,
            },
            ValidationProfile::Custom(thresholds) => PerformanceThresholds::from_config_thresholds(thresholds),
        }
    }
}
```

### Alternative Approaches

**Option A: Simple Environment-Only Configuration**
- **Pros:** Minimal code changes, follows existing patterns
- **Cons:** Limited flexibility, no file-based profiles

**Option B: Hardware Auto-Detection**
- **Pros:** Zero-configuration adaptive behavior
- **Cons:** Complexity in detection logic, potential inconsistencies

**Option C: Model-Aware Thresholds**
- **Pros:** Optimal thresholds per model size/type
- **Cons:** Requires extensive benchmarking data, complex implementation

## Implementation Plan

### Phase 1: Foundation (Priority: High)
**Estimated Time:** 3-4 days
**Dependencies:** None

1. **Environment Variable Support**
   - [ ] Add `BITNET_PERF_*` environment variables to `bitnet-common/src/config.rs`
   - [ ] Implement parsing with validation and error handling
   - [ ] Update `apply_env_overrides()` method with performance threshold support
   - [ ] Add unit tests for environment variable parsing

2. **Configuration Integration**
   - [ ] Extend `ValidationConfig` with optional `ConfigurablePerformanceThresholds`
   - [ ] Implement `from_config()` method in `PerformanceThresholds`
   - [ ] Add configuration file schema support (TOML/JSON)
   - [ ] Update validation logic to use configurable thresholds

### Phase 2: Profiles and Adaptation (Priority: Medium)
**Estimated Time:** 2-3 days
**Dependencies:** Phase 1 complete

1. **Validation Profiles**
   - [ ] Implement `ValidationProfile` enum with predefined threshold sets
   - [ ] Add profile selection via `BITNET_VALIDATION_PROFILE` environment variable
   - [ ] Create profile configuration files for common scenarios
   - [ ] Add profile validation and error reporting

2. **Hardware Adaptation**
   - [ ] Implement basic hardware detection (CPU cores, available memory)
   - [ ] Add adaptive threshold calculation algorithms
   - [ ] Create fallback mechanisms for detection failures
   - [ ] Add logging for threshold adaptation decisions

### Phase 3: Integration and Documentation (Priority: Medium)
**Estimated Time:** 2 days
**Dependencies:** Phase 1-2 complete

1. **CLI and Tools Integration**
   - [ ] Update `bitnet-cli` commands to support threshold configuration
   - [ ] Add `--validation-profile` command-line option
   - [ ] Integrate with `xtask` validation commands
   - [ ] Update configuration validation in tools

2. **Documentation and Examples**
   - [ ] Update `docs/environment-variables.md` with new `BITNET_PERF_*` variables
   - [ ] Create configuration examples for different deployment scenarios
   - [ ] Add troubleshooting guide for threshold tuning
   - [ ] Update validation framework documentation

## Testing Strategy

### Unit Tests (Required)
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_performance_thresholds_from_env() {
        env::set_var("BITNET_PERF_MIN_TOKENS_PER_SECOND", "25.0");
        env::set_var("BITNET_PERF_MAX_LATENCY_MS", "2000.0");

        let config = ValidationConfig::from_env().unwrap();
        let thresholds = PerformanceThresholds::from_config(&config);

        assert_eq!(thresholds.min_tokens_per_second, 25.0);
        assert_eq!(thresholds.max_latency_ms, 2000.0);

        env::remove_var("BITNET_PERF_MIN_TOKENS_PER_SECOND");
        env::remove_var("BITNET_PERF_MAX_LATENCY_MS");
    }

    #[test]
    fn test_validation_profile_selection() {
        let dev_profile = ValidationProfile::Development;
        let thresholds = dev_profile.to_thresholds();

        assert_eq!(thresholds.min_tokens_per_second, 5.0);
        assert_eq!(thresholds.max_latency_ms, 10000.0);
    }

    #[test]
    fn test_hardware_adaptive_defaults() {
        // Test with mocked hardware detection
        let thresholds = PerformanceThresholds::default();
        assert!(thresholds.min_tokens_per_second > 0.0);
        assert!(thresholds.max_memory_usage_mb > 0.0);
    }

    #[test]
    fn test_config_file_loading() {
        let config_content = r#"
        [validation.performance_thresholds]
        min_tokens_per_second = 20.0
        max_latency_ms = 3000.0
        max_memory_usage_mb = 4096.0
        min_speedup_factor = 2.0
        "#;

        // Test TOML configuration loading
        let config: BitNetConfig = toml::from_str(config_content).unwrap();
        let validation_config = ValidationConfig {
            performance_thresholds: Some(config.validation.performance_thresholds),
            ..Default::default()
        };
        let thresholds = PerformanceThresholds::from_config(&validation_config);

        assert_eq!(thresholds.min_tokens_per_second, 20.0);
        assert_eq!(thresholds.max_latency_ms, 3000.0);
    }
}
```

### Integration Tests
```bash
# Test environment variable integration
BITNET_PERF_MIN_TOKENS_PER_SECOND=30.0 BITNET_PERF_MAX_LATENCY_MS=1000.0 \
cargo test -p bitnet-inference test_configurable_validation_comprehensive

# Test profile-based validation
BITNET_VALIDATION_PROFILE=production \
cargo test -p bitnet-inference test_validation_with_profiles

# Test hardware adaptation
cargo test -p bitnet-inference test_hardware_adaptive_thresholds

# Test configuration file integration
cargo test -p bitnet-inference test_validation_config_file_loading
```

### Performance Impact Tests
```rust
#[bench]
fn bench_threshold_calculation_overhead(b: &mut Bencher) {
    b.iter(|| {
        let config = ValidationConfig::default();
        let thresholds = PerformanceThresholds::from_config(&config);
        black_box(thresholds);
    });
}
```

## Risk Assessment and Mitigation

### Implementation Risks

**Risk 1: Backward Compatibility**
- **Impact:** Existing validation tests may fail with new adaptive defaults
- **Mitigation:**
  - Preserve exact current defaults when no configuration is provided
  - Add feature flag for gradual rollout: `BITNET_ADAPTIVE_THRESHOLDS=1`
  - Comprehensive regression testing across hardware configurations

**Risk 2: Hardware Detection Reliability**
- **Impact:** Inconsistent threshold calculation across systems
- **Mitigation:**
  - Conservative fallback to current hardcoded values
  - Extensive logging for debugging detection issues
  - Manual override capability via environment variables

**Risk 3: Configuration Complexity**
- **Impact:** Users overwhelmed by configuration options
- **Mitigation:**
  - Sensible defaults that work without configuration
  - Clear documentation with recommended profiles
  - Validation and helpful error messages for invalid configurations

### Performance Considerations

**Configuration Loading Overhead:**
- Lazy loading of environment variables (cached after first read)
- Minimal computational overhead for hardware detection
- Configuration validation only during initialization

**Memory Impact:**
- Additional configuration fields: ~64 bytes per `ValidationConfig` instance
- Hardware detection metadata: ~256 bytes cached globally
- Total overhead: <1KB per validation session

## Acceptance Criteria

### Functional Requirements
- [ ] **Environment Variables:** All `BITNET_PERF_*` environment variables are supported and properly parsed
- [ ] **Configuration Files:** Performance thresholds can be specified in TOML and JSON configuration files
- [ ] **Validation Profiles:** Predefined profiles (Development, CI, Staging, Production) work correctly
- [ ] **Hardware Adaptation:** Automatic threshold adjustment based on detected hardware capabilities
- [ ] **Backward Compatibility:** Existing code works unchanged with same default behavior

### Quality Requirements
- [ ] **Test Coverage:** >95% line coverage for all new configuration code
- [ ] **Error Handling:** Graceful degradation with informative error messages for invalid configurations
- [ ] **Performance:** Configuration loading adds <10ms overhead to validation setup
- [ ] **Documentation:** Comprehensive documentation with examples for all configuration options

### Integration Requirements
- [ ] **CLI Integration:** Command-line tools support threshold configuration options
- [ ] **CI/CD Compatibility:** Works correctly in GitHub Actions and other CI environments
- [ ] **Cross-Platform:** Functions identically across Linux, macOS, and Windows
- [ ] **Hardware Diversity:** Validated on CPU-only, CUDA, and Apple Metal configurations

## Related Issues and PRs

### Dependencies
- **Issue #251:** Production-Ready Inference Server (configuration system integration)
- **Issue #260:** Mock Elimination (strict mode compatibility)

### Related Components
- **`bitnet-common/src/config.rs`:** Core configuration system
- **`docs/environment-variables.md`:** Environment variable documentation
- **`crates/bitnet-cli/src/config.rs`:** CLI configuration handling

### Future Enhancements
- **Model-Aware Thresholds:** Automatic threshold calculation based on model metadata
- **Performance History:** Adaptive thresholds based on historical performance data
- **Cloud Integration:** Threshold profiles for AWS, GCP, Azure instance types

## Labels and Classification

**Labels:** `enhancement`, `validation`, `configuration`, `priority:medium`, `component:inference`

**Milestone:** v0.3.0 (Configurable Infrastructure)

**Priority:** Medium (Developer experience improvement, CI/CD reliability)

**Effort:** Medium (6-9 person-days across 3 phases)

**Review Requirements:**
- Technical review by infrastructure team (configuration system changes)
- Performance review for overhead assessment
- Documentation review for user-facing changes

---

**Implementation Notes:**
- Follow existing BitNet-rs configuration patterns established in `bitnet-common`
- Ensure compatibility with strict mode testing (`BITNET_STRICT_MODE=1`)
- Maintain zero-configuration usability with sensible adaptive defaults
- Integrate with existing hardware detection in `bitnet-kernels` crate
