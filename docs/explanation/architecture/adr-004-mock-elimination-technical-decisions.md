# ADR-004: Mock Elimination Technical Decisions for BitNet.rs Quantized Inference

## Status
**PROPOSED** - Pending approval for Issue #260 implementation

## Context

BitNet.rs currently suffers from a critical architectural flaw where mock inference paths dominate execution, reporting false performance metrics (~200 tok/s) instead of real quantized neural network computation. This ADR documents the key technical decisions required to eliminate mock fallbacks and implement authentic quantized inference using I2S, TL1, and TL2 quantization algorithms.

### Current Architecture Problems

1. **Mock Dominance**: `ConcreteTensor::mock()` calls throughout inference pipeline
2. **Compilation Barriers**: 21+ compilation errors prevent real quantization execution
3. **False Metrics**: Performance reporting based on dummy computation rather than real matrix multiplication
4. **Validation Impossibility**: Cannot cross-validate against Microsoft C++ reference due to mock paths

### Scope of Technical Decisions

This ADR covers architectural decisions for:
- Compilation error resolution strategies
- Quantization kernel integration patterns
- Device-aware execution model
- Performance measurement framework
- Strict mode implementation
- Cross-validation architecture

## Decision

### Decision 1: Strict Mode Environment Variable Architecture

**Decision**: Implement `BITNET_STRICT_MODE=1` environment variable that forces real quantization or fails fast, preventing any mock fallback usage.

**Rationale**:
- **Production Safety**: Ensures production deployments cannot accidentally use mock paths
- **CI Integration**: Enables automated detection of mock usage in test pipelines
- **Development Flexibility**: Allows gradual migration from mock to real quantization
- **Error Clarity**: Provides clear failure modes with descriptive error messages

**Implementation Pattern**:
```rust
pub struct StrictModeConfig {
    pub enabled: bool,                   // BITNET_STRICT_MODE=1
    pub fail_on_mock: bool,             // Fail fast on mock fallbacks
    pub require_quantization: bool,     // Require real quantization kernels
}

impl StrictModeConfig {
    pub fn validate_inference_path(&self, path: &InferencePath) -> Result<()> {
        if self.enabled && path.uses_mock_computation() {
            return Err(anyhow!(
                "Strict mode: Mock computation detected in inference path: {}",
                path.description()
            ));
        }
        Ok(())
    }
}
```

**Alternatives Considered**:
- **Compile-time feature flags**: Rejected due to runtime configuration needs
- **Configuration file approach**: Rejected for simplicity and CI integration
- **Automatic detection**: Rejected due to complexity and false positive potential

### Decision 2: Quantization Kernel Integration Strategy

**Decision**: Implement device-aware quantization kernel selection with fallback hierarchy: Native Quantized > SIMD Optimized > Generic Implementation.

**Rationale**:
- **Performance Optimization**: Native quantized kernels avoid dequantization overhead
- **Device Awareness**: Automatic selection of optimal kernel per device (CPU SIMD, GPU CUDA)
- **Graceful Degradation**: Maintains functionality across different hardware capabilities
- **Cross-Platform Support**: Works on x86_64 (AVX2), aarch64 (NEON), and CUDA GPUs

**Implementation Pattern**:
```rust
pub trait KernelProvider: Send + Sync {
    fn supports_quantization(&self, qtype: QuantizationType) -> bool;

    async fn quantized_matmul(
        &self,
        input: &BitNetTensor,
        weights: &QuantizedTensor,
        qtype: QuantizationType
    ) -> Result<BitNetTensor>;
}

impl QuantizedLinear {
    async fn select_kernel_provider(&self) -> Result<&dyn KernelProvider> {
        // Priority order: Native Quantized > SIMD > Generic
        if self.supports_native_quantized() {
            Ok(self.native_provider.as_ref())
        } else if self.supports_simd() {
            Ok(self.simd_provider.as_ref())
        } else {
            Ok(self.generic_provider.as_ref())
        }
    }
}
```

**Alternatives Considered**:
- **Single kernel approach**: Rejected due to performance limitations
- **Runtime compilation**: Rejected due to complexity and build system requirements
- **External library dependencies**: Rejected to maintain BitNet.rs self-contained architecture

### Decision 3: Memory Layout Optimization Strategy

**Decision**: Implement quantization-specific memory layout optimization with SIMD alignment and zero-copy patterns.

**Rationale**:
- **Cache Efficiency**: Quantization-specific alignment (I2S: 82-byte blocks, TL1: 16-byte NEON, TL2: 32-byte AVX)
- **Zero-Copy Operations**: Memory-mapped models with efficient lifetime management
- **Memory Pool Management**: Pre-allocated workspaces to avoid runtime allocation overhead
- **Device Optimization**: GPU memory layout differs from CPU for optimal bandwidth utilization

**Implementation Pattern**:
```rust
impl QuantizedLinear {
    fn optimize_memory_layout(&mut self) -> Result<()> {
        match self.qtype {
            QuantizationType::I2S => {
                // 82-byte blocks with SIMD alignment
                let aligned_size = self.calculate_i2s_alignment();
                self.allocate_aligned_memory_pool(aligned_size, 64)?;
            }
            QuantizationType::TL1 => {
                // ARM NEON 16-byte alignment
                #[cfg(target_arch = "aarch64")]
                self.allocate_aligned_memory_pool(size, 16)?;
            }
            QuantizationType::TL2 => {
                // x86 AVX 32/64-byte alignment
                #[cfg(target_arch = "x86_64")]
                let alignment = if cfg!(target_feature = "avx512f") { 64 } else { 32 };
                self.allocate_aligned_memory_pool(size, alignment)?;
            }
        }
        Ok(())
    }
}
```

**Alternatives Considered**:
- **Generic alignment**: Rejected due to suboptimal performance for specific quantization types
- **Dynamic allocation**: Rejected due to performance overhead and fragmentation
- **External memory allocators**: Rejected to maintain dependency minimization

### Decision 4: Performance Measurement Architecture

**Decision**: Implement realistic performance baselines with comprehensive metrics collection that distinguishes real computation from mock fallbacks.

**Rationale**:
- **Realistic Targets**: CPU 10-20 tok/s, GPU 50-100 tok/s based on quantization overhead analysis
- **Mock Detection**: Automatic identification of suspiciously high performance indicating mock usage
- **Comprehensive Breakdown**: Separate timing for quantization, kernel execution, and memory transfers
- **Cross-Validation Integration**: Performance comparison framework against C++ reference

**Implementation Pattern**:
```rust
pub struct EnhancedPerformanceMetrics {
    pub base_metrics: PerformanceMetrics,
    pub quantization_breakdown: QuantizationBreakdown,
    pub kernel_utilization: KernelUtilization,
    pub is_mock_detected: bool,              // Flag for mock computation detection
}

impl PerformanceTracker {
    pub fn validate_realistic_performance(&self, metrics: &EnhancedPerformanceMetrics) -> Result<()> {
        // Detect suspiciously high performance indicating mock usage
        if metrics.base_metrics.tokens_per_second > 150.0 {
            return Err(anyhow!(
                "Unrealistic performance detected: {:.2} tok/s suggests mock computation",
                metrics.base_metrics.tokens_per_second
            ));
        }

        // Validate performance is within expected quantization overhead
        self.validate_quantization_overhead(metrics)?;
        Ok(())
    }
}
```

**Alternatives Considered**:
- **Simple throughput measurement**: Rejected due to inability to detect mock usage
- **External benchmarking tools**: Rejected for integration complexity
- **Statistical performance modeling**: Rejected due to hardware variability complexity

### Decision 5: Cross-Validation Framework Architecture

**Decision**: Implement automated cross-validation against Microsoft C++ reference with tolerance-based accuracy validation and performance comparison.

**Rationale**:
- **Accuracy Validation**: Ensures Rust implementation maintains >99% correlation with reference
- **Performance Baseline**: Establishes realistic expectations based on C++ performance
- **Continuous Validation**: Integrates with CI pipeline for regression detection
- **Tolerance Management**: Handles acceptable numerical differences between implementations

**Implementation Pattern**:
```rust
pub trait CrossValidationProvider {
    async fn validate_against_reference(
        &self,
        rust_output: &BitNetTensor,
        cpp_reference: &CppReferenceOutput,
        tolerance: f32
    ) -> Result<CrossValidationReport>;
}

pub struct CrossValidationReport {
    pub correlation: f32,           // Pearson correlation coefficient
    pub mse: f32,                  // Mean squared error
    pub max_absolute_error: f32,   // Maximum absolute difference
    pub passed: bool,              // Within tolerance
    pub performance_ratio: f32,    // Rust vs C++ performance
}

impl CrossValidationFramework {
    pub async fn establish_baseline(&self, workload: &Workload) -> Result<PerformanceBaseline> {
        let cpp_result = self.run_cpp_reference(workload).await?;
        let rust_result = self.run_rust_implementation(workload).await?;

        Ok(PerformanceBaseline {
            cpp_tokens_per_sec: cpp_result.throughput,
            rust_target_min: cpp_result.throughput * 0.8,  // Allow 20% performance difference
            rust_target_max: cpp_result.throughput * 1.2,
            accuracy_correlation_min: 0.99,
        })
    }
}
```

**Alternatives Considered**:
- **Manual validation approach**: Rejected due to scalability and CI integration needs
- **Approximate validation**: Rejected due to accuracy requirements for production use
- **Single-point validation**: Rejected for comprehensive coverage needs

### Decision 6: Compilation Error Resolution Strategy

**Decision**: Implement incremental compilation fixing with feature-flag isolation and type system harmonization.

**Rationale**:
- **Systematic Approach**: Address compilation errors in dependency order to avoid cascade failures
- **Feature Isolation**: Separate CPU and GPU compilation paths to isolate CUDA-specific issues
- **Type Harmonization**: Unify `BitNetTensor` and `ConcreteTensor` usage patterns across crates
- **Async Safety**: Resolve lifetime and Send/Sync issues in async quantization contexts

**Implementation Strategy**:
```rust
// Type unification pattern
pub type InferenceTensor = BitNetTensor;  // Unified type across crates

// Feature-isolated compilation
#[cfg(feature = "cpu")]
impl CPUKernelProvider {
    pub async fn quantized_matmul(&self, ...) -> Result<InferenceTensor> {
        // CPU-specific implementation
    }
}

#[cfg(feature = "gpu")]
impl GPUKernelProvider {
    pub async fn quantized_matmul(&self, ...) -> Result<InferenceTensor> {
        // GPU-specific implementation
    }
}

// Async safety pattern
impl QuantizedLinear {
    pub async fn forward(&self, input: &InferenceTensor) -> Result<InferenceTensor> {
        let kernel_provider = self.select_kernel_provider()?;
        kernel_provider.quantized_matmul(input, &self.weights, self.qtype).await
    }
}
```

**Alternatives Considered**:
- **Big-bang compilation fix**: Rejected due to high risk of introducing new issues
- **External dependencies**: Rejected to maintain BitNet.rs architectural principles
- **Type erasure patterns**: Rejected due to performance implications

### Decision 7: CI Pipeline Integration Strategy

**Decision**: Implement CI pipeline enhancements that automatically detect mock usage and validate realistic performance baselines.

**Rationale**:
- **Mock Detection**: Prevents accidental merge of mock-dependent code
- **Performance Regression**: Detects performance degradation in real quantization paths
- **Cross-Platform Validation**: Ensures consistent behavior across target platforms
- **Strict Mode Enforcement**: Validates that production builds cannot use mock fallbacks

**Implementation Pattern**:
```yaml
# CI Pipeline Integration
- name: Validate Mock Elimination
  run: |
    export BITNET_STRICT_MODE=1
    cargo test --workspace --no-default-features --features cpu

- name: Performance Baseline Validation
  run: |
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42
    cargo run -p xtask -- performance-baseline --validate

- name: Cross-Validation Check
  run: |
    cargo run -p xtask -- crossval --tolerance 0.05
```

**Alternatives Considered**:
- **Manual testing approach**: Rejected due to human error potential
- **Post-merge validation**: Rejected due to late feedback cycle
- **External CI tools**: Rejected for simplicity and control

## Consequences

### Positive Consequences

1. **Production Readiness**: BitNet.rs will provide authentic quantized neural network inference
2. **Performance Accuracy**: Realistic performance reporting enables informed deployment decisions
3. **Research Validation**: Accurate cross-validation against C++ reference enables research publication
4. **Quality Assurance**: Strict mode prevents accidental mock usage in production
5. **Platform Optimization**: Device-aware quantization maximizes performance per platform

### Negative Consequences

1. **Performance Reduction**: Real quantization will show lower throughput than mock (10-20 tok/s vs 200 tok/s)
2. **Complexity Increase**: Device-aware kernel selection adds implementation complexity
3. **Build Dependencies**: GPU features require CUDA toolkit installation
4. **Memory Requirements**: Optimized memory layouts may increase memory usage
5. **Development Overhead**: Comprehensive testing and validation requirements

### Mitigation Strategies

1. **Performance Communication**: Clear documentation of realistic vs mock performance expectations
2. **Gradual Migration**: Phased implementation allows incremental complexity management
3. **Optional Dependencies**: GPU features remain optional with CPU fallback
4. **Memory Tuning**: Configurable memory pool sizes for different deployment scenarios
5. **Development Tools**: Enhanced tooling and documentation for complex validation scenarios

## Implementation Notes

### Phase 1: Foundation (Weeks 1-2)
- **Focus**: Compilation error resolution and strict mode implementation
- **Risk**: Cascading compilation issues may extend timeline
- **Mitigation**: Feature-flag isolation and incremental fixing

### Phase 2: Quantization Integration (Weeks 3-4)
- **Focus**: I2S, TL1, TL2 kernel integration with device awareness
- **Risk**: Kernel performance may not meet targets
- **Mitigation**: Algorithm optimization and hardware-specific tuning

### Phase 3: Inference Transformation (Weeks 5-6)
- **Focus**: QLinear replacement and performance framework
- **Risk**: Memory layout optimization complexity
- **Mitigation**: Conservative memory management with performance profiling

### Phase 4: Validation and CI (Weeks 7-8)
- **Focus**: Cross-validation and CI integration
- **Risk**: C++ reference integration complexity
- **Mitigation**: Tolerance-based validation with documented acceptable differences

### Success Criteria

1. **Compilation**: 100% success for `cargo build --no-default-features --features cpu|gpu`
2. **Accuracy**: >99% correlation with FP32 reference for all quantization types
3. **Performance**: CPU 10-20 tok/s, GPU 50-100 tok/s sustained throughput
4. **Validation**: <5% performance difference from C++ reference
5. **CI Integration**: Automated mock detection with <5% false positive rate

### Monitoring and Validation

1. **Performance Tracking**: Continuous monitoring of throughput and accuracy metrics
2. **Memory Profiling**: Regular validation of memory usage patterns and leak detection
3. **Cross-Platform Testing**: Validation across x86_64, aarch64, and CUDA environments
4. **Regression Detection**: Automated detection of performance or accuracy degradation
5. **Documentation Validation**: Ensure performance claims match measured characteristics

## Related ADRs

- **ADR-001**: BitNet.rs Architecture Principles
- **ADR-002**: Quantization Algorithm Selection
- **ADR-003**: Device-Aware Execution Model
- **ADR-005**: Performance Optimization Strategies (planned)

## References

- **Issue #260**: Mock Inference Performance Reporting
- **Issue #261**: 10 Atomic Acceptance Criteria
- **Microsoft BitNet C++ Reference**: Cross-validation baseline
- **GGUF Specification**: Model format compatibility requirements
- **BitNet.rs Documentation**: Architecture and development guidelines

---

**ADR Status**: PROPOSED
**Decision Date**: 2024-09-27
**Review Date**: After Phase 1 completion
**Next Update**: Upon implementation completion