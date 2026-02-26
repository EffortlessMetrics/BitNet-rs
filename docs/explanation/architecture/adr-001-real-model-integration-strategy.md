# ADR-001: Real BitNet Model Integration Strategy

## Status

**PROPOSED** - Architectural blueprint for Issue #218 implementation

## Context

BitNet-rs currently uses mock models and placeholder data for testing and examples, which prevents validation of real-world neural network inference accuracy and performance. For MVP completion, we need to integrate actual BitNet model artifacts while maintaining development velocity and CI reliability.

### Current Architecture Limitations

1. **Mock-Only Testing**: All examples and tests use synthetic data, preventing validation of actual model quality
2. **No Cross-Validation**: Cannot verify accuracy against C++ reference implementation
3. **Performance Unknown**: Real-world performance characteristics unmeasured
4. **GGUF Compatibility**: Limited validation of actual model format compatibility
5. **Tokenizer Integration**: No integration with real model vocabularies

### Technical Constraints

- **CI Memory Limits**: Large models (2-4GB) challenge CI environments
- **Network Dependencies**: Model downloads introduce external dependencies
- **Cross-Platform Consistency**: Different platforms may have numerical variations
- **Device Availability**: GPU support varies across development and CI environments
- **Development Velocity**: Must not slow down development with mandatory large downloads

## Decision

We will implement a **Hybrid Model Architecture** with intelligent real/mock selection based on availability and environment configuration, supported by a comprehensive feature flag strategy.

### Core Decision Points

#### 1. Feature Flag Strategy

**Decision**: Implement empty default features with explicit feature selection

```rust
# Cargo.toml default features
[features]
default = []  # Empty default - forces explicit selection
cpu = ["bitnet-quantization/cpu", "bitnet-kernels/cpu"]
gpu = ["bitnet-quantization/gpu", "bitnet-kernels/gpu", "bitnet-kernels/cuda"]
inference = ["bitnet-inference/real-models", "bitnet-tokenizers/real"]
crossval = ["bitnet-crossval/cpp-integration"]
```

**Rationale**:
- **Explicit Intent**: Developers must explicitly choose capabilities
- **Minimal Dependencies**: Default build has minimal external dependencies
- **CI Flexibility**: Different CI lanes can use different feature combinations
- **Development Efficiency**: Fast builds for unit tests, comprehensive builds for integration

#### 2. Model Discovery and Selection Strategy

**Decision**: Implement automatic model discovery with fallback hierarchy

```rust
pub enum ModelSelectionStrategy {
    /// Prefer real models, fallback to mock if unavailable
    PreferReal { fallback_to_mock: bool },
    /// Force real models, fail if unavailable
    ForceReal,
    /// Use mock models only (development/testing)
    MockOnly,
    /// Environment-controlled selection
    EnvironmentControlled,
}
```

**Implementation Priority**:
1. **Environment Variables**: `BITNET_GGUF`, `BITNET_TOKENIZER`
2. **Standard Locations**: `./models/`, `~/.cache/bitnet/`
3. **Download Integration**: `cargo xtask download-model`
4. **Mock Fallback**: When real models unavailable (with warnings)

**Rationale**:
- **Developer Flexibility**: Choose appropriate model for task
- **CI Reliability**: Graceful degradation when models unavailable
- **Production Readiness**: Force real models in production environments
- **Development Velocity**: Mock models for rapid iteration

#### 3. CI Integration Strategy

**Decision**: Implement three-tier CI testing strategy

```yaml
# Tier 1: Fast Lane (< 5 minutes)
fast_tests:
  features: ["cpu"]
  models: "mock_only"
  parallel: high

# Tier 2: Standard Lane (< 15 minutes)
standard_tests:
  features: ["cpu", "inference"]
  models: "cached_real_with_mock_fallback"
  parallel: medium

# Tier 3: Full Validation (< 45 minutes)
full_validation:
  features: ["cpu", "gpu", "crossval"]
  models: "real_required"
  parallel: low
```

**Model Caching Strategy**:
- **Cache Key**: `bitnet-models-${{ hashFiles('models.lock') }}`
- **Cache Scope**: Workflow-level caching across jobs
- **Compression**: Model compression for faster cache operations
- **Fallback**: Download retry with exponential backoff

**Rationale**:
- **CI Speed**: Fast feedback for most development scenarios
- **Resource Management**: Prevent CI resource exhaustion
- **Quality Assurance**: Comprehensive validation when needed
- **Cost Efficiency**: Minimize CI resource usage

#### 4. Device-Aware Execution Strategy

**Decision**: Implement automatic device detection with manual override

```rust
pub struct DeviceStrategy {
    pub preference: DevicePreference,
    pub fallback_enabled: bool,
    pub gpu_memory_limit: Option<usize>,
    pub cpu_thread_limit: Option<usize>,
}

pub enum DevicePreference {
    Auto,           // Automatic selection based on capability
    ForceGPU,       // GPU required, fail if unavailable
    ForceCPU,       // CPU only, skip GPU detection
    Hybrid,         // GPU for compute, CPU for control
}
```

**Environment Integration**:
```bash
export BITNET_DEVICE="auto"              # Device preference
export BITNET_GPU_MEMORY_LIMIT="4096"    # GPU memory limit (MB)
export BITNET_CPU_THREADS="16"           # CPU thread count
export BITNET_STRICT_NO_FAKE_GPU="1"     # Disable mock GPU for testing
```

**Rationale**:
- **Hardware Flexibility**: Adapt to available hardware
- **Performance Optimization**: Use best available device
- **Testing Reliability**: Consistent behavior across environments
- **Resource Management**: Prevent resource exhaustion

#### 5. Cross-Validation Integration Strategy

**Decision**: Implement configurable cross-validation with tolerance management

```rust
pub struct CrossValidationConfig {
    pub enabled: bool,
    pub cpp_binary_path: Option<PathBuf>,
    pub tolerance_config: ToleranceConfig,
    pub performance_comparison: bool,
}

pub struct ToleranceConfig {
    pub inference_tolerance: f32,        // Default: 1e-4
    pub quantization_tolerance: f32,     // Default: 1e-5
    pub perplexity_tolerance: f64,       // Default: 1e-3
    pub performance_tolerance: f32,      // Default: 0.05 (5%)
}
```

**Environment Integration**:
```bash
export BITNET_CPP_DIR="/path/to/cpp/implementation"
export BITNET_VALIDATION_TOLERANCE="1e-4"
export CROSSVAL_GGUF="/path/to/validation/model.gguf"
```

**Rationale**:
- **Accuracy Validation**: Ensure correctness against reference
- **Regression Prevention**: Catch accuracy degradation early
- **Platform Adaptation**: Account for platform-specific variations
- **Development Efficiency**: Optional validation for faster iteration

## Consequences

### Positive Outcomes

#### 1. Production Readiness
- **Real Model Validation**: Confidence in production deployment
- **Performance Characteristics**: Measured real-world performance
- **Accuracy Assurance**: Cross-validation against reference implementation
- **Format Compatibility**: Validated GGUF format support

#### 2. Development Velocity
- **Fast Development**: Mock models for rapid iteration
- **Incremental Validation**: Optional real model testing
- **CI Efficiency**: Tiered testing strategy
- **Resource Management**: Intelligent resource usage

#### 3. Quality Assurance
- **Comprehensive Testing**: Multiple validation levels
- **Platform Consistency**: Cross-platform validation
- **Regression Detection**: Automated accuracy monitoring
- **Error Recovery**: Graceful fallback strategies

### Potential Challenges

#### 1. Implementation Complexity
- **Multiple Code Paths**: Real vs mock model handling
- **Configuration Management**: Complex feature flag interactions
- **Error Handling**: Comprehensive fallback strategies
- **Documentation**: Complex setup and configuration guidance

**Mitigation Strategy**:
- **Unified Interfaces**: Common API for real and mock implementations
- **Configuration Validation**: Early validation of incompatible combinations
- **Comprehensive Testing**: Test all code paths and configurations
- **Clear Documentation**: Step-by-step setup guides

#### 2. CI Reliability
- **Network Dependencies**: Model download failures
- **Resource Constraints**: Memory and storage limitations
- **Timing Variability**: Performance test consistency
- **Platform Differences**: Cross-platform numerical variations

**Mitigation Strategy**:
- **Robust Caching**: Multiple cache levels and fallback strategies
- **Resource Monitoring**: Proactive resource management
- **Statistical Validation**: Use correlation metrics instead of exact equality
- **Platform Baselines**: Platform-specific reference values

#### 3. Developer Experience
- **Setup Complexity**: Multiple configuration options
- **Debugging Difficulty**: Complex error scenarios
- **Performance Overhead**: Large model handling
- **Documentation Burden**: Comprehensive guides needed

**Mitigation Strategy**:
- **Smart Defaults**: Sensible default configurations
- **Diagnostic Tools**: Comprehensive error reporting and recovery guidance
- **Performance Optimization**: Efficient model loading and caching
- **Interactive Guides**: Step-by-step configuration tools

### Risk Assessment

#### High-Risk Areas
1. **CI Timeout**: Large model downloads exceeding CI limits
2. **Memory Exhaustion**: Large models in memory-constrained environments
3. **Network Failures**: Dependency on external model repositories
4. **Platform Variations**: Numerical inconsistencies across platforms

#### Mitigation Strategies
1. **Intelligent Caching**: Multi-level cache with compression
2. **Memory Management**: Memory-mapped loading and efficient allocation
3. **Fallback Repositories**: Multiple model sources with retry logic
4. **Tolerance Configuration**: Configurable numerical tolerances

## Implementation Plan

### Phase 1: Foundation (Week 1)
1. **Feature Flag Infrastructure**: Implement empty default features
2. **Model Discovery**: Environment-based model location
3. **Basic Integration**: Real model loading with mock fallback
4. **CI Foundation**: Basic three-tier testing structure

### Phase 2: Core Implementation (Week 2)
1. **Real Model Loading**: Production GGUF loader with validation
2. **Device Detection**: Automatic GPU/CPU selection
3. **Performance Metrics**: Comprehensive timing and throughput measurement
4. **Error Handling**: Diagnostic errors with recovery guidance

### Phase 3: Validation (Week 3)
1. **Cross-Validation**: C++ parity testing framework
2. **Tokenizer Integration**: GGUF metadata extraction
3. **Quality Assurance**: Perplexity and accuracy validation
4. **Performance Benchmarking**: Real model performance testing

### Phase 4: Production (Week 4)
1. **CI Integration**: Full three-tier testing deployment
2. **Documentation**: Comprehensive setup and usage guides
3. **Performance Optimization**: Memory and throughput optimization
4. **Quality Validation**: End-to-end production readiness testing

## Monitoring and Success Metrics

### Functional Metrics
- [ ] **Real Model Loading**: 99%+ success rate for valid GGUF files
- [ ] **Cross-Validation**: Pass rate ≥95% within configured tolerance
- [ ] **CI Reliability**: <5% failure rate due to model-related issues
- [ ] **Performance**: Meet throughput targets for real models

### Quality Metrics
- [ ] **Accuracy Preservation**: Perplexity within ±0.1% of reference
- [ ] **Deterministic Behavior**: 99%+ reproducibility for deterministic inputs
- [ ] **Error Recovery**: Comprehensive error messages with actionable guidance
- [ ] **Documentation Quality**: Developer setup success rate ≥90%

### Operational Metrics
- [ ] **CI Performance**: Three-tier strategy maintains target execution times
- [ ] **Resource Usage**: Memory and storage within acceptable limits
- [ ] **Developer Velocity**: Development workflow impact <10%
- [ ] **Maintenance Overhead**: Automated model management and validation

## Conclusion

This architectural decision establishes a comprehensive strategy for real BitNet model integration that balances production readiness with development velocity. The hybrid approach enables validation of real-world performance while maintaining the flexibility needed for efficient development and CI operations.

The three-tier CI strategy and intelligent fallback mechanisms ensure reliability while the comprehensive feature flag system provides precise control over capabilities and dependencies. This foundation enables BitNet-rs to transition from a development framework to a production-ready neural network inference system with confidence in accuracy and performance characteristics.
