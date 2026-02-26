# Issue #261: Mock Inference Performance Reporting Elimination - Technical Specification

## Executive Summary

This specification outlines the comprehensive technical approach for eliminating mock inference paths that report false 200.0 tokens/sec performance metrics in BitNet-rs. The implementation will transform the neural network inference pipeline from mock fallbacks to real quantized computation using I2S, TL1, and TL2 quantization kernels, enabling accurate performance evaluation and production deployment.

**Classification**: Breaking (Architecture Change)
**Risk Level**: High (Core inference pipeline transformation)
**Estimated Timeline**: 9-15 days for complete implementation
**Migration Path**: Phased elimination with strict mode enforcement

## Table of Contents

1. [Context and Problem Statement](#context-and-problem-statement)
2. [User Stories](#user-stories)
3. [Acceptance Criteria](#acceptance-criteria)
4. [Technical Architecture](#technical-architecture)
5. [API Contracts](#api-contracts)
6. [Quantization Integration Strategy](#quantization-integration-strategy)
7. [Performance Framework](#performance-framework)
8. [Feature Flag Compatibility](#feature-flag-compatibility)
9. [GGUF Format Compatibility](#gguf-format-compatibility)
10. [Cross-Validation Requirements](#cross-validation-requirements)
11. [Implementation Roadmap](#implementation-roadmap)
12. [Risk Assessment](#risk-assessment)
13. [Testing Strategy](#testing-strategy)

---

## Context and Problem Statement

### Current State Analysis

BitNet-rs currently exhibits critical architectural flaws in performance reporting:

**Problem 1: False Performance Metrics**
- Mock inference paths report ~200.0 tok/s through `ConcreteTensor::mock()` fallbacks
- Real quantized computation blocked by compilation errors
- Production users cannot validate actual BitNet quantization performance

**Problem 2: Pipeline Architecture Gaps**
```
Model Loading → [✅ WORKING] GGUF integration complete
     ↓
Quantization → [❌ BROKEN] I2S/TL1/TL2 kernels not integrated
     ↓
Kernels → [⚠️ BYPASSED] SIMD/CUDA compute uses fallbacks
     ↓
Inference → [🚫 MOCK] Autoregressive generation with dummy computation
     ↓
Output → [❌ FALSE] Performance metrics from mock rather than real computation
```

**Problem 3: Neural Network Validation Gap**
- Cannot compare against Microsoft C++ reference implementation
- No cross-validation of quantization accuracy (I2S ≥99.8% vs FP32)
- Missing device-aware optimization validation (GPU/CPU)

### Affected Workspace Crates

| Crate | Impact | Required Changes |
|-------|--------|------------------|
| `bitnet-quantization` | High | Activate I2S/TL1/TL2 kernel integration |
| `bitnet-inference` | Critical | Remove mock paths, integrate QLinear layers |
| `bitnet-kernels` | High | Activate SIMD/CUDA quantized matrix multiplication |
| `bitnet-models` | Medium | QLinear layer replacement in transformer architecture |
| `bitnet-common` | Medium | Enhance strict mode enforcement |
| `crossval` | Medium | Establish real computation baselines |

---

## User Stories

### Primary User Story - Neural Network Performance Engineer

**As a** neural network performance engineer evaluating BitNet-rs for production deployment,
**I want** accurate performance reporting from real quantized inference computation,
**So that** I can make evidence-based decisions about model deployment, establish realistic performance baselines, and validate the 1-bit quantization accuracy claims against FP32 references.

**Business Value**: Enables production adoption decisions based on authentic performance data rather than mock fallback metrics.

**Acceptance Criteria Coverage**: AC1-AC10

### Secondary User Stories

#### Research Validation Engineer
**As a** research validation engineer,
**I want** cross-validation against Microsoft C++ reference within 1e-5 numerical tolerance,
**So that** I can verify quantization implementation correctness and publish peer-reviewed results.

**Acceptance Criteria Coverage**: AC9

#### Quality Assurance Team
**As a** QA engineer validating BitNet-rs releases,
**I want** BITNET_STRICT_MODE=1 environment variable that prevents all mock fallbacks,
**So that** I can ensure production builds use only real quantized computation.

**Acceptance Criteria Coverage**: AC2, AC6

#### DevOps Performance Team
**As a** DevOps engineer establishing performance SLOs,
**I want** realistic CPU (10-20 tok/s) and GPU (50-100 tok/s) performance baselines,
**So that** I can configure autoscaling policies and capacity planning for inference workloads.

**Acceptance Criteria Coverage**: AC7, AC8

---

## Acceptance Criteria

### AC1: Compilation Error Resolution ✅
**AC_ID**: AC1
**Priority**: Critical
**Status**: COMPLETE (as of Issue #260 implementation)

**Requirement**: Fix all compilation errors blocking real quantized inference execution with proper error context using anyhow::Result patterns.

**Definition of Done**:
```bash
cargo build --workspace --no-default-features --features cpu
# Must complete successfully without compilation errors
```

**Test Strategy**:
- Integration test validates compilation success across feature flag combinations
- Test tag: `// AC:AC1`
- Validation: `cargo test -p bitnet-quantization test_ac1_cpu_compilation_success`

**Evidence**: Successful CI builds across cpu/gpu/none feature combinations

---

### AC2: Strict Mode Implementation
**AC_ID**: AC2
**Priority**: Critical
**Status**: IN PROGRESS

**Requirement**: Implement strict mode environment variable (BITNET_STRICT_MODE=1) that prevents mock fallbacks and fails fast with descriptive errors when quantization kernels unavailable.

**Definition of Done**:
```bash
BITNET_STRICT_MODE=1 cargo test -p bitnet-inference --no-default-features --features cpu
# Must fail with descriptive error when mock paths are attempted
```

**API Contract**:
```rust
pub struct StrictModeConfig {
    pub enabled: bool,                   // BITNET_STRICT_MODE=1
    pub fail_on_mock: bool,             // Fail fast on mock fallbacks
    pub require_quantization: bool,     // Require real quantization kernels
    pub validate_performance: bool,     // Reject suspicious performance metrics
}

impl StrictModeConfig {
    /// Validate inference path for mock usage
    pub fn validate_inference_path(&self, path: &MockInferencePath) -> Result<()>;

    /// Validate kernel availability before execution
    pub fn validate_kernel_availability(&self, scenario: &MissingKernelScenario) -> Result<()>;

    /// Validate performance metrics for suspicious values
    pub fn validate_performance_metrics(&self, metrics: &PerformanceMetrics) -> Result<()>;
}
```

**Test Strategy**:
- Unit tests verify strict mode behavior with `// AC:AC2` tags
- Validation: `cargo test -p bitnet-common test_ac2_strict_mode_environment_variable`
- CI integration: Enforce strict mode in production pipelines

**Evidence**: Test failures when mock paths attempted under BITNET_STRICT_MODE=1

---

### AC3: I2S Quantization Kernel Integration
**AC_ID**: AC3
**Priority**: High
**Status**: PENDING

**Requirement**: Integrate I2S (2-bit signed) quantization kernels for device-aware execution (CPU SIMD/GPU CUDA) without dequantization fallback.

**Definition of Done**:
```bash
cargo test -p bitnet-quantization --no-default-features --features cpu test_i2s_simd_scalar_parity
# I2S kernels must execute with >99.8% correlation vs FP32 reference
```

**API Contract**:
```rust
pub trait I2SKernelProvider: Send + Sync {
    /// Execute native quantized matrix multiplication
    async fn quantized_matmul_i2s(
        &self,
        input: &BitNetTensor,
        weights: &QuantizedTensor,
    ) -> Result<BitNetTensor>;

    /// Validate quantization accuracy against FP32 reference
    fn validate_accuracy(&self, reference: &BitNetTensor, quantized: &BitNetTensor) -> Result<f64>;
}

impl QuantizedLinear {
    /// Forward pass with native I2S quantization (no dequantization)
    async fn forward_i2s(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        let provider = self.kernel_manager.select_best_i2s()?;
        self.quantized_matmul_i2s(&input, provider).await
    }
}
```

**Quantization Accuracy Requirements**:
- **Target**: ≥99.8% correlation with FP32 reference
- **Numerical Tolerance**: MSE < 1e-6 for matrix operations
- **Block Size**: 82 elements (SIMD-aligned)
- **Memory Efficiency**: 4:1 compression ratio (2-bit vs 8-bit)

**Test Strategy**:
- Cross-validation tests with C++ reference using `// AC:AC3` tags
- Device-aware testing: CPU SIMD (AVX2/AVX-512) and GPU CUDA
- Validation: `cargo test -p bitnet-quantization test_i2s_native_quantized_matmul`

**Evidence**: Direct kernel invocation in profiling traces, no dequantization fallback

---

### AC4: TL1/TL2 Quantization Kernel Integration
**AC_ID**: AC4
**Priority**: High
**Status**: PENDING

**Requirement**: Integrate device-aware TL1 (table lookup 1, ARM NEON optimized) and TL2 (table lookup 2, x86 AVX optimized) quantization kernels with architecture-specific optimizations.

**Definition of Done**:
```bash
cargo test -p bitnet-kernels --no-default-features --features cpu test_mixed_precision_matmul_accuracy
# TL1/TL2 kernels must execute with >99.6% correlation vs FP32 reference
```

**API Contract**:
```rust
pub trait TableLookupKernelProvider: Send + Sync {
    /// Get optimal table size for current device architecture
    fn get_optimal_table_size(&self) -> usize;

    /// Execute table lookup quantized matrix multiplication
    async fn quantized_matmul_tl(
        &self,
        input: &BitNetTensor,
        weights: &QuantizedTensor,
        table_type: TableLookupType,
    ) -> Result<BitNetTensor>;
}

#[derive(Debug, Clone, Copy)]
pub enum TableLookupType {
    TL1 { table_size: usize },  // ARM NEON optimized (16-256 entries)
    TL2 { table_size: usize },  // x86 AVX optimized (256-4096 entries)
}

impl QuantizedLinear {
    /// Select optimal table lookup strategy based on device architecture
    fn select_table_lookup_strategy(&self) -> Result<TableLookupType> {
        match self.device.arch() {
            Architecture::Aarch64 => Ok(TableLookupType::TL1 { table_size: 128 }),
            Architecture::X86_64 => Ok(TableLookupType::TL2 { table_size: 1024 }),
            _ => Err(UnsupportedArchitecture),
        }
    }
}
```

**Quantization Accuracy Requirements**:
- **TL1 Target**: ≥99.6% correlation with FP32 reference (ARM NEON)
- **TL2 Target**: ≥99.6% correlation with FP32 reference (x86 AVX)
- **Cache Efficiency**: Lookup table fits in L1/L2 cache
- **Vectorization**: NEON 16-byte, AVX2 32-byte, AVX-512 64-byte alignment

**Test Strategy**:
- Architecture-specific tests with `// AC:AC4` tags
- Device-aware kernel selection validation
- Validation: `cargo test -p bitnet-kernels test_device_aware_quantization_selection`

**Evidence**: Architecture-specific kernel selection in device profiling

---

### AC5: QLinear Layer Replacement
**AC_ID**: AC5
**Priority**: High
**Status**: PENDING

**Requirement**: Replace standard linear layers with QuantizedLinear in transformer architecture, eliminating all mock tensor operations in the inference pipeline.

**Definition of Done**:
```bash
cargo test -p bitnet-models --test gguf_min -- test_tensor_alignment
# QLinear layers must load from GGUF and execute without mock operations
```

**API Contract**:
```rust
pub struct QuantizedLinearLayer {
    pub weights: QuantizedTensor,
    pub quantization_type: QuantizationType,
    pub device: Device,
    pub kernel_provider: Arc<dyn KernelProvider>,
}

impl QuantizedLinearLayer {
    /// Load quantized layer from GGUF tensor metadata
    pub fn from_gguf_tensor(
        name: &str,
        tensor: &GGUFTensor,
        device: Device,
    ) -> Result<Self>;

    /// Forward pass with native quantized computation
    pub async fn forward(&self, input: &BitNetTensor) -> Result<BitNetTensor>;

    /// Validate layer configuration and weight alignment
    pub fn validate_layer_config(&self) -> Result<()>;
}

pub trait BitNetModelLayer {
    /// Replace standard linear layers with quantized equivalents
    fn replace_with_quantized_layers(&mut self, quantization_type: QuantizationType) -> Result<()>;
}
```

**Integration Requirements**:
- Load quantized weights from GGUF tensor metadata
- Detect quantization type automatically (I2S/TL1/TL2)
- Validate tensor alignment for SIMD/CUDA operations
- Support mixed quantization types within single model

**Test Strategy**:
- Integration tests verify QLinear replacement with `// AC:AC5` tags
- GGUF compatibility validation
- Validation: `cargo test -p bitnet-models test_quantized_linear_layer_integration`

**Evidence**: No mock tensor operations in inference pipeline profiling

---

### AC6: CI Mock Evidence Rejection
**AC_ID**: AC6
**Priority**: High
**Status**: PENDING

**Requirement**: Configure CI pipeline to reject performance claims from mock inference, enforcing strict mode validation in automated testing.

**Definition of Done**:
```yaml
# .github/workflows/ci.yml
- name: Strict Mode Validation
  run: |
    BITNET_STRICT_MODE=1 cargo test --workspace --no-default-features --features cpu
    # Must fail if mock evidence detected in performance metrics
```

**CI Configuration**:
```yaml
env:
  BITNET_STRICT_MODE: "1"
  BITNET_CI_ENHANCED_STRICT: "1"
  BITNET_STRICT_FAIL_ON_MOCK: "1"

jobs:
  strict-mode-enforcement:
    steps:
      - name: Run Strict Mode Tests
        run: cargo test --workspace --no-default-features --features cpu

      - name: Validate Performance Metrics
        run: cargo run -p xtask -- validate-performance-metrics

      - name: Reject Mock Evidence
        run: |
          if grep -r "mock.*200.*tok" target/performance-reports/; then
            echo "ERROR: Mock performance evidence detected"
            exit 1
          fi
```

**Test Strategy**:
- CI pipeline integration tests with `// AC:AC6` tags
- Automated mock detection in performance reports
- Validation: CI workflow configuration changes

**Evidence**: Failed CI builds when mock evidence detected in performance reports

---

### AC7: CPU Performance Baselines
**AC_ID**: AC7
**Priority**: Medium
**Status**: PENDING

**Requirement**: Establish realistic CPU performance baselines for I2S quantization (10-20 tokens/sec on modern x86_64 CPU).

**Definition of Done**:
```bash
cargo bench --workspace --no-default-features --features cpu -- cpu_i2s_baseline
# Must establish consistent baseline measurements across test runs
```

**API Contract**:
```rust
pub struct CPUPerformanceBaseline {
    pub architecture: CpuArchitecture,
    pub quantization_type: QuantizationType,
    pub target_tokens_per_sec: RangeInclusive<f64>,
    pub latency_percentiles: LatencyPercentiles,
}

#[derive(Debug, Clone)]
pub struct LatencyPercentiles {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

impl CPUPerformanceBaseline {
    pub fn i2s_x86_64_avx2() -> Self {
        Self {
            architecture: CpuArchitecture::X86_64_AVX2,
            quantization_type: QuantizationType::I2S,
            target_tokens_per_sec: 15.0..=20.0,
            latency_percentiles: LatencyPercentiles {
                p50_ms: 50.0,
                p95_ms: 80.0,
                p99_ms: 100.0,
            },
        }
    }
}
```

**Performance Targets**:
- **I2S (AVX2)**: 15-20 tokens/sec
- **I2S (AVX-512)**: 20-25 tokens/sec
- **TL1 (ARM NEON)**: 12-18 tokens/sec
- **TL2 (x86 AVX)**: 10-15 tokens/sec

**Test Strategy**:
- Performance benchmarks with `// AC:AC7` tags
- Statistical validation across multiple test runs
- Validation: `cargo run -p xtask -- benchmark --cpu-baseline`

**Evidence**: Consistent performance measurements within target ranges

---

### AC8: GPU Performance Baselines
**AC_ID**: AC8
**Priority**: Medium
**Status**: PENDING

**Requirement**: Establish realistic GPU performance baselines with mixed precision acceleration (50-100 tokens/sec with FP16/BF16).

**Definition of Done**:
```bash
cargo test --workspace --no-default-features --features gpu test_gpu_smoke
# Must validate GPU utilization >80% during inference
```

**API Contract**:
```rust
pub struct GPUPerformanceBaseline {
    pub device_name: String,
    pub compute_capability: (u32, u32),
    pub mixed_precision: MixedPrecisionConfig,
    pub target_tokens_per_sec: RangeInclusive<f64>,
    pub gpu_utilization_target: f64,
}

#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    pub activation_dtype: DType,  // FP16 or BF16
    pub weight_dtype: DType,      // INT2 or INT4
    pub accumulator_dtype: DType, // FP32
}

impl GPUPerformanceBaseline {
    pub fn cuda_mixed_precision_i2s() -> Self {
        Self {
            device_name: "NVIDIA GPU".to_string(),
            compute_capability: (8, 0),
            mixed_precision: MixedPrecisionConfig {
                activation_dtype: DType::F16,
                weight_dtype: DType::I2S,
                accumulator_dtype: DType::F32,
            },
            target_tokens_per_sec: 60.0..=100.0,
            gpu_utilization_target: 0.80,
        }
    }
}
```

**Performance Targets**:
- **I2S (CUDA FP16)**: 60-100 tokens/sec
- **I2S (CUDA BF16)**: 50-90 tokens/sec
- **GPU Utilization**: >80% during inference
- **Memory Bandwidth**: 85-95% efficiency

**Test Strategy**:
- GPU performance benchmarks with `// AC:AC8` tags
- Device capability detection and validation
- Validation: `cargo test --features gpu test_gpu_performance_baseline`

**Evidence**: GPU utilization >80% in performance profiling

---

### AC9: Cross-Validation Accuracy
**AC_ID**: AC9
**Priority**: Critical
**Status**: PENDING

**Requirement**: Maintain <5% performance variance from Microsoft C++ reference implementation with >99.5% correlation for quantization accuracy.

**Definition of Done**:
```bash
export BITNET_GGUF="path/to/model.gguf"
cargo run -p xtask -- crossval --release --tolerance 0.05
# Must achieve >99.5% correlation with C++ reference
```

**API Contract**:
```rust
pub struct CrossValidationMetrics {
    pub correlation_coefficient: f64,
    pub mse_tolerance: f64,
    pub performance_variance: f64,
    pub numerical_tolerance: f64,
}

impl CrossValidationMetrics {
    pub fn validate_against_cpp_reference(&self) -> Result<()> {
        if self.correlation_coefficient < 0.995 {
            return Err(anyhow!(
                "Correlation too low: {:.4} < 0.995",
                self.correlation_coefficient
            ));
        }

        if self.mse_tolerance > 1e-5 {
            return Err(anyhow!(
                "MSE tolerance exceeded: {:.2e} > 1e-5",
                self.mse_tolerance
            ));
        }

        Ok(())
    }
}
```

**Validation Requirements**:
- **Correlation**: >99.5% with Microsoft C++ implementation
- **MSE Tolerance**: <1e-5 for FP32 comparisons
- **Performance Variance**: <5% from C++ baseline
- **Numerical Tolerance**: <1e-6 for individual operations

**Test Strategy**:
- Cross-validation tests with `// AC:AC9` tags
- Automated C++ reference comparison
- Validation: `cargo run -p xtask -- crossval`

**Evidence**: Correlation >99.5% in cross-validation reports

---

### AC10: Documentation Updates
**AC_ID**: AC10
**Priority**: Low
**Status**: PENDING

**Requirement**: Update performance documentation to reflect real capabilities, removing all mock-based performance claims.

**Definition of Done**:
```bash
# Documentation must reflect realistic performance baselines
grep -r "200.*tok" docs/ && exit 1 || echo "No mock claims found"
```

**Documentation Requirements**:
- Remove all references to 200 tok/s mock performance
- Update performance benchmarking documentation
- Add realistic CPU/GPU baseline documentation
- Document strict mode usage and enforcement

**Test Strategy**:
- Documentation review and accuracy verification
- Validation: `cargo run -p xtask -- verify-documentation`

**Evidence**: Removal of mock-based performance claims from docs/

---

## Technical Architecture

### Inference Pipeline Transformation

**Current Architecture (Mock-Dominated)**:
```
┌─────────────────────────────────────────────────────────────────┐
│ Model Loading (GGUF)                                            │
│ ✅ Zero-copy memory mapping                                     │
│ ✅ Tensor metadata extraction                                   │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ Quantization Layer                                              │
│ ❌ I2S/TL1/TL2 kernels not integrated                          │
│ ⚠️  Fallback to dequantization                                 │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ Kernel Execution                                                │
│ ⚠️  SIMD/CUDA kernels bypassed                                 │
│ 🚫 Mock ConcreteTensor operations                              │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ Inference Engine                                                │
│ 🚫 Autoregressive generation with mock computation             │
│ ❌ Reports false 200.0 tok/s performance                       │
└─────────────────────────────────────────────────────────────────┘
```

**Target Architecture (Real Quantization)**:
```
┌─────────────────────────────────────────────────────────────────┐
│ Model Loading (GGUF)                                            │
│ ✅ Zero-copy memory mapping                                     │
│ ✅ Tensor metadata with quantization type detection            │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ QLinear Layer Construction                                      │
│ ✅ Replace linear layers with QuantizedLinear                  │
│ ✅ Detect I2S/TL1/TL2 from GGUF metadata                       │
│ ✅ Device-aware kernel selection                               │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ Native Quantized Kernels                                        │
│ ✅ I2S: SIMD (AVX2/AVX-512/NEON) + CUDA mixed precision        │
│ ✅ TL1: ARM NEON optimized table lookup                        │
│ ✅ TL2: x86 AVX optimized table lookup                         │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ Production Inference Engine                                     │
│ ✅ Autoregressive generation with real quantized computation   │
│ ✅ Accurate performance metrics (10-20 tok/s CPU, 50-100 GPU)  │
│ ✅ Strict mode enforcement (BITNET_STRICT_MODE=1)              │
└─────────────────────────────────────────────────────────────────┘
```

### Device-Aware Execution Model

**Kernel Selection Priority**:
```rust
pub struct KernelSelectionStrategy {
    priority_order: Vec<KernelType>,
}

impl KernelSelectionStrategy {
    pub fn for_device(device: &Device, qtype: QuantizationType) -> Self {
        match (device, qtype) {
            // GPU: Highest priority for CUDA mixed precision
            (Device::Cuda(_), QuantizationType::I2S) => Self {
                priority_order: vec![
                    KernelType::CudaMixedPrecision,
                    KernelType::CudaInt2,
                    KernelType::FallbackCpu,
                ],
            },

            // CPU x86_64: AVX-512 > AVX2 > scalar
            (Device::Cpu, QuantizationType::I2S) if is_x86_64_avx512() => Self {
                priority_order: vec![
                    KernelType::SimdAvx512,
                    KernelType::SimdAvx2,
                    KernelType::Scalar,
                ],
            },

            // CPU aarch64: NEON > scalar
            (Device::Cpu, QuantizationType::TL1) if is_aarch64() => Self {
                priority_order: vec![
                    KernelType::SimdNeon,
                    KernelType::Scalar,
                ],
            },

            _ => Self::default(),
        }
    }
}
```

### Strict Mode Enforcement Architecture

**Validation Points**:
```rust
pub struct StrictModeValidator {
    config: StrictModeConfig,
}

impl StrictModeValidator {
    /// Validate at model loading time
    pub fn validate_model_load(&self, model: &Model) -> Result<()> {
        if self.config.enabled && model.has_mock_layers() {
            return Err(StrictModeError::MockLayersDetected);
        }
        Ok(())
    }

    /// Validate at inference time
    pub fn validate_inference_start(&self, engine: &InferenceEngine) -> Result<()> {
        if self.config.enabled && !engine.uses_real_quantization() {
            return Err(StrictModeError::MockInferenceAttempted);
        }
        Ok(())
    }

    /// Validate performance metrics
    pub fn validate_performance_report(&self, metrics: &PerformanceMetrics) -> Result<()> {
        if self.config.validate_performance {
            if metrics.computation_type == ComputationType::Mock {
                return Err(StrictModeError::MockPerformanceReported);
            }

            if metrics.tokens_per_second > 150.0 {
                return Err(StrictModeError::SuspiciousPerformance {
                    reported: metrics.tokens_per_second,
                    expected_max: 150.0,
                });
            }
        }
        Ok(())
    }
}
```

---

## API Contracts

### Quantization Kernel Interface

```rust
/// Core trait for quantization kernel providers
pub trait QuantizationKernelProvider: Send + Sync {
    /// Get supported quantization types for this kernel provider
    fn supported_quantization_types(&self) -> &[QuantizationType];

    /// Get device this kernel provider targets
    fn target_device(&self) -> Device;

    /// Execute quantized matrix multiplication
    async fn quantized_matmul(
        &self,
        input: &BitNetTensor,
        weights: &QuantizedTensor,
        qtype: QuantizationType,
    ) -> Result<BitNetTensor>;

    /// Validate quantization accuracy against reference
    fn validate_accuracy(
        &self,
        reference: &BitNetTensor,
        quantized: &BitNetTensor,
    ) -> Result<QuantizationAccuracyMetrics>;
}

/// Accuracy metrics for quantization validation
#[derive(Debug, Clone)]
pub struct QuantizationAccuracyMetrics {
    pub correlation_coefficient: f64,
    pub mse: f64,
    pub max_absolute_error: f64,
    pub mean_absolute_error: f64,
}

impl QuantizationAccuracyMetrics {
    pub fn meets_i2s_requirements(&self) -> bool {
        self.correlation_coefficient >= 0.998 && self.mse < 1e-6
    }

    pub fn meets_tl_requirements(&self) -> bool {
        self.correlation_coefficient >= 0.996 && self.mse < 1e-5
    }
}
```

### Performance Measurement Interface

```rust
/// Performance metrics collector for real inference
pub trait PerformanceMetricsCollector: Send + Sync {
    /// Record prefill phase metrics
    fn record_prefill(&mut self, tokens: usize, duration: Duration);

    /// Record decode phase metrics
    fn record_decode(&mut self, tokens: usize, duration: Duration);

    /// Record GPU utilization (if available)
    fn record_gpu_utilization(&mut self, utilization: f64);

    /// Finalize and validate metrics
    fn finalize(&mut self, total_duration: Duration) -> Result<PerformanceReport>;
}

/// Comprehensive performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub computation_type: ComputationType,
    pub quantization_type: QuantizationType,
    pub device: DeviceInfo,
    pub timing: TimingBreakdown,
    pub throughput: ThroughputMetrics,
    pub accuracy: Option<QuantizationAccuracyMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingBreakdown {
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub tokenization_encode_ms: u64,
    pub tokenization_decode_ms: u64,
    pub total_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    pub prefill_tokens_per_sec: f64,
    pub decode_tokens_per_sec: f64,
    pub end_to_end_tokens_per_sec: f64,
    pub total_tokens: usize,
}
```

### Strict Mode Validation Interface

```rust
/// Strict mode enforcement interface
pub trait StrictModeEnforcement {
    /// Check if strict mode is enabled
    fn is_strict_mode_enabled(&self) -> bool;

    /// Validate inference path before execution
    fn validate_inference_path(&self, path: &InferencePath) -> Result<()>;

    /// Validate kernel availability for required quantization type
    fn validate_kernel_availability(
        &self,
        device: &Device,
        qtype: QuantizationType,
    ) -> Result<()>;

    /// Validate performance metrics before reporting
    fn validate_performance_metrics(&self, metrics: &PerformanceMetrics) -> Result<()>;
}

/// Inference path descriptor for validation
#[derive(Debug, Clone)]
pub struct InferencePath {
    pub description: String,
    pub uses_mock_computation: bool,
    pub quantization_type: Option<QuantizationType>,
    pub device: Device,
    pub fallback_reason: Option<String>,
}
```

---

## Quantization Integration Strategy

### I2S (2-bit Signed) Quantization

**Precision Requirements**:
- Target: ≥99.8% correlation with FP32 reference
- Block size: 82 elements (SIMD-aligned)
- Memory efficiency: 4:1 compression ratio

**Implementation Strategy**:
```rust
pub struct I2SQuantizer {
    block_size: usize,  // 82 elements
    device: Device,
    kernel_provider: Arc<dyn QuantizationKernelProvider>,
}

impl I2SQuantizer {
    pub async fn quantize_weights(&self, weights: &BitNetTensor) -> Result<QuantizedTensor> {
        // Validate block alignment
        self.validate_block_alignment(weights)?;

        // Compute per-block scales
        let scales = self.compute_block_scales(weights)?;

        // Quantize to 2-bit signed representation
        let quantized = self.quantize_to_i2s(weights, &scales)?;

        Ok(QuantizedTensor {
            data: quantized,
            scales,
            quantization_type: QuantizationType::I2S,
            block_size: self.block_size,
        })
    }

    pub async fn dequantize_for_validation(&self, tensor: &QuantizedTensor) -> Result<BitNetTensor> {
        // Only for cross-validation, not used in production inference
        self.dequantize_i2s(&tensor.data, &tensor.scales)
    }
}
```

**SIMD Optimization**:
```rust
// AVX2 optimization for x86_64
#[cfg(target_arch = "x86_64")]
pub fn i2s_matmul_avx2(
    input: &[f32],
    weights_i2s: &[u8],
    scales: &[f32],
    output: &mut [f32],
) -> Result<()> {
    unsafe {
        // AVX2 SIMD intrinsics for 2-bit signed quantization
        // Process 32 elements per iteration (256-bit registers)
    }
}

// CUDA kernel for GPU acceleration
#[cfg(feature = "gpu")]
pub fn i2s_matmul_cuda(
    input: &CudaTensor,
    weights_i2s: &CudaTensor,
    scales: &CudaTensor,
) -> Result<CudaTensor> {
    // CUDA kernel with FP16 mixed precision
    // Coalesced memory access for optimal bandwidth
}
```

### TL1 (Table Lookup 1) Quantization - ARM NEON

**Precision Requirements**:
- Target: ≥99.6% correlation with FP32 reference
- Lookup table size: 16-256 entries (cache-friendly)
- Vectorization: ARM NEON 16-byte alignment

**Implementation Strategy**:
```rust
pub struct TL1Quantizer {
    table_size: usize,  // 128 entries for optimal L1 cache usage
    device: Device,
}

impl TL1Quantizer {
    pub fn create_lookup_table(&self, weights: &BitNetTensor) -> Result<Vec<f32>> {
        // Create lookup table optimized for ARM NEON
        let mut table = vec![0.0; self.table_size];

        // Populate table with representative weight values
        self.populate_table_from_weights(weights, &mut table)?;

        Ok(table)
    }

    pub async fn quantize_with_table(&self, weights: &BitNetTensor) -> Result<QuantizedTensor> {
        let lookup_table = self.create_lookup_table(weights)?;
        let indices = self.map_weights_to_indices(weights, &lookup_table)?;

        Ok(QuantizedTensor {
            data: indices,
            lookup_table: Some(lookup_table),
            quantization_type: QuantizationType::TL1,
            block_size: 1,
        })
    }
}

// ARM NEON optimization
#[cfg(target_arch = "aarch64")]
pub fn tl1_matmul_neon(
    input: &[f32],
    weight_indices: &[u8],
    lookup_table: &[f32],
    output: &mut [f32],
) -> Result<()> {
    unsafe {
        // ARM NEON SIMD intrinsics for table lookup
        // Process 16 bytes per iteration
    }
}
```

### TL2 (Table Lookup 2) Quantization - x86 AVX

**Precision Requirements**:
- Target: ≥99.6% correlation with FP32 reference
- Lookup table size: 256-4096 entries (larger tables for better accuracy)
- Vectorization: AVX2 32-byte, AVX-512 64-byte alignment

**Implementation Strategy**:
```rust
pub struct TL2Quantizer {
    table_size: usize,  // 1024 entries for optimal L2 cache usage
    device: Device,
}

impl TL2Quantizer {
    pub fn create_hierarchical_table(&self, weights: &BitNetTensor) -> Result<Vec<f32>> {
        // Create larger lookup table with hierarchical structure
        let mut table = vec![0.0; self.table_size];

        // Two-level table for better accuracy
        self.populate_hierarchical_table(weights, &mut table)?;

        Ok(table)
    }
}

// x86 AVX2 optimization
#[cfg(target_arch = "x86_64")]
pub fn tl2_matmul_avx2(
    input: &[f32],
    weight_indices: &[u16],  // 16-bit indices for larger tables
    lookup_table: &[f32],
    output: &mut [f32],
) -> Result<()> {
    unsafe {
        // AVX2 SIMD intrinsics with gather operations
        // Process 32 bytes per iteration
    }
}
```

---

## Performance Framework

### CPU Performance Baselines

**Architecture-Specific Targets**:

| Architecture | Quantization | Target (tok/s) | SIMD Features |
|--------------|--------------|----------------|---------------|
| x86_64 (AVX2) | I2S | 15-20 | 256-bit vectors |
| x86_64 (AVX-512) | I2S | 20-25 | 512-bit vectors |
| x86_64 (AVX2) | TL2 | 10-15 | Gather operations |
| aarch64 (NEON) | I2S | 10-15 | 128-bit vectors |
| aarch64 (NEON) | TL1 | 12-18 | Table lookup |

**Performance Measurement Strategy**:
```rust
pub struct CPUPerformanceBenchmark {
    pub architecture: CpuArchitecture,
    pub quantization_type: QuantizationType,
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
}

impl CPUPerformanceBenchmark {
    pub async fn measure_baseline(&self, model: &Model) -> Result<PerformanceBaseline> {
        // Warmup phase
        for _ in 0..self.warmup_iterations {
            self.run_inference_iteration(model).await?;
        }

        // Measurement phase
        let mut samples = Vec::with_capacity(self.measurement_iterations);
        for _ in 0..self.measurement_iterations {
            let start = Instant::now();
            let tokens = self.run_inference_iteration(model).await?;
            let duration = start.elapsed();

            samples.push(PerformanceSample {
                tokens,
                duration,
                tokens_per_sec: tokens as f64 / duration.as_secs_f64(),
            });
        }

        Ok(PerformanceBaseline::from_samples(samples))
    }
}
```

### GPU Performance Baselines

**Mixed Precision Configuration**:

| Precision | Activation | Weight | Accumulator | Target (tok/s) |
|-----------|------------|--------|-------------|----------------|
| FP16 | FP16 | INT2 (I2S) | FP32 | 60-100 |
| BF16 | BF16 | INT2 (I2S) | FP32 | 50-90 |
| FP16 | FP16 | INT4 | FP32 | 80-120 |

**GPU Utilization Validation**:
```rust
pub struct GPUPerformanceBenchmark {
    pub device: Device,
    pub mixed_precision: MixedPrecisionConfig,
    pub target_utilization: f64,  // 0.80 = 80%
}

impl GPUPerformanceBenchmark {
    pub async fn measure_with_profiling(&self, model: &Model) -> Result<GPUPerformanceReport> {
        // Start GPU profiling
        let profiler = self.start_gpu_profiler()?;

        // Run inference with profiling
        let start = Instant::now();
        let tokens = self.run_inference_with_profiling(model).await?;
        let duration = start.elapsed();

        // Collect GPU metrics
        let metrics = profiler.stop()?;

        Ok(GPUPerformanceReport {
            tokens_per_sec: tokens as f64 / duration.as_secs_f64(),
            gpu_utilization: metrics.average_utilization,
            memory_bandwidth: metrics.memory_bandwidth_efficiency,
            compute_efficiency: metrics.compute_efficiency,
        })
    }
}
```

---

## Feature Flag Compatibility

### Build Configuration Matrix

**Default Configuration** (Empty features by design):
```bash
cargo build --no-default-features
# Compiles library code only, no inference runtime
```

**CPU-Only Configuration**:
```bash
cargo build --no-default-features --features cpu
# Enables: SIMD kernels (AVX2/AVX-512/NEON), CPU device support
```

**GPU-Only Configuration**:
```bash
cargo build --no-default-features --features gpu
# Enables: CUDA kernels, mixed precision, GPU memory management
```

**Full Development Configuration**:
```bash
cargo build --no-default-features --features cpu,gpu,crossval
# Enables: All kernels, cross-validation framework
```

### Feature Flag Dependencies

```toml
[features]
default = []  # Intentionally empty

# Core feature flags
cpu = ["dep:candle-core", "bitnet-kernels/cpu"]
gpu = ["dep:candle-core", "dep:candle-nn", "dep:cudarc", "bitnet-kernels/gpu"]

# Cross-validation features
crossval = ["cpu", "dep:bitnet-ffi"]
ffi = ["dep:cbindgen"]

# Strict mode enforcement
strict = []  # Environment variable based, no code dependencies
```

### Fallback Testing Strategy

```rust
pub struct FeatureFlagValidator {
    enabled_features: HashSet<String>,
}

impl FeatureFlagValidator {
    pub fn validate_cpu_fallback(&self) -> Result<()> {
        #[cfg(not(feature = "cpu"))]
        {
            return Err(anyhow!("CPU feature required for fallback execution"));
        }

        #[cfg(feature = "cpu")]
        {
            // Validate CPU kernels are available
            self.validate_cpu_kernels_available()
        }
    }

    pub fn validate_gpu_graceful_degradation(&self) -> Result<()> {
        #[cfg(feature = "gpu")]
        {
            match self.check_cuda_available() {
                Ok(true) => self.validate_gpu_kernels(),
                Ok(false) => {
                    // GPU feature enabled but CUDA unavailable
                    // Should gracefully fall back to CPU
                    self.validate_cpu_fallback()
                }
                Err(e) => Err(e),
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            // GPU not available, ensure CPU fallback works
            self.validate_cpu_fallback()
        }
    }
}
```

---

## GGUF Format Compatibility

### Quantization Type Detection

```rust
pub struct GGUFQuantizationDetector {
    supported_types: HashSet<GGMLQuantizationType>,
}

impl GGUFQuantizationDetector {
    pub fn detect_quantization_type(&self, tensor_meta: &TensorMetadata) -> Result<QuantizationType> {
        match tensor_meta.ggml_type {
            GGMLQuantizationType::I2_S => Ok(QuantizationType::I2S),
            GGMLQuantizationType::I4_S => Ok(QuantizationType::I4S),
            GGMLQuantizationType::TL1 => Ok(QuantizationType::TL1),
            GGMLQuantizationType::TL2 => Ok(QuantizationType::TL2),
            unsupported => Err(anyhow!(
                "Unsupported GGML quantization type: {:?}",
                unsupported
            )),
        }
    }

    pub fn validate_tensor_alignment(&self, tensor: &GGUFTensor) -> Result<()> {
        let required_alignment = match tensor.quantization_type {
            QuantizationType::I2S => 82,  // I2S block size
            QuantizationType::TL1 | QuantizationType::TL2 => 1,
            _ => return Err(anyhow!("Unknown quantization type alignment")),
        };

        if tensor.shape.iter().product::<usize>() % required_alignment != 0 {
            return Err(anyhow!(
                "Tensor shape {:?} not aligned to {} for {:?}",
                tensor.shape,
                required_alignment,
                tensor.quantization_type
            ));
        }

        Ok(())
    }
}
```

### QLinear Layer GGUF Integration

```rust
pub struct GGUFQLinearLoader {
    detector: GGUFQuantizationDetector,
    device: Device,
}

impl GGUFQLinearLoader {
    pub fn load_qlinear_from_gguf(
        &self,
        layer_name: &str,
        tensors: &HashMap<String, GGUFTensor>,
    ) -> Result<QuantizedLinearLayer> {
        // Find weight tensor for this layer
        let weight_key = format!("{}.weight", layer_name);
        let weight_tensor = tensors.get(&weight_key)
            .ok_or_else(|| anyhow!("Weight tensor not found: {}", weight_key))?;

        // Detect quantization type
        let qtype = self.detector.detect_quantization_type(&weight_tensor.metadata)?;

        // Validate tensor alignment
        self.detector.validate_tensor_alignment(weight_tensor)?;

        // Create quantized layer
        Ok(QuantizedLinearLayer {
            weights: self.load_quantized_weights(weight_tensor, qtype)?,
            quantization_type: qtype,
            device: self.device.clone(),
            kernel_provider: self.create_kernel_provider(qtype)?,
        })
    }

    fn create_kernel_provider(&self, qtype: QuantizationType) -> Result<Arc<dyn QuantizationKernelProvider>> {
        match (self.device.clone(), qtype) {
            (Device::Cuda(_), QuantizationType::I2S) => {
                Ok(Arc::new(CudaI2SKernelProvider::new()?))
            }
            (Device::Cpu, QuantizationType::I2S) => {
                Ok(Arc::new(CpuI2SKernelProvider::new()?))
            }
            (Device::Cpu, QuantizationType::TL1) => {
                Ok(Arc::new(CpuTL1KernelProvider::new()?))
            }
            _ => Err(anyhow!("Unsupported device/quantization combination")),
        }
    }
}
```

---

## Cross-Validation Requirements

### C++ Reference Parity

**Validation Approach**:
```rust
pub struct CrossValidationFramework {
    cpp_reference: Option<CppReferenceImplementation>,
    tolerance: CrossValidationTolerance,
}

#[derive(Debug, Clone)]
pub struct CrossValidationTolerance {
    pub correlation_min: f64,      // 0.995 = 99.5%
    pub mse_max: f64,              // 1e-5
    pub performance_variance_max: f64,  // 0.05 = 5%
    pub numerical_tolerance: f64,  // 1e-6
}

impl CrossValidationFramework {
    pub async fn validate_against_cpp(&self, model_path: &Path) -> Result<CrossValidationReport> {
        // Load model in both Rust and C++ implementations
        let rust_model = self.load_rust_model(model_path).await?;
        let cpp_model = self.cpp_reference
            .as_ref()
            .ok_or_else(|| anyhow!("C++ reference not available"))?
            .load_model(model_path)?;

        // Run identical inference
        let prompt = "The quick brown fox";
        let rust_output = self.run_rust_inference(&rust_model, prompt).await?;
        let cpp_output = self.run_cpp_inference(&cpp_model, prompt)?;

        // Compare outputs
        let correlation = self.compute_correlation(&rust_output, &cpp_output)?;
        let mse = self.compute_mse(&rust_output, &cpp_output)?;

        // Validate against tolerance
        self.validate_correlation(correlation)?;
        self.validate_mse(mse)?;

        Ok(CrossValidationReport {
            correlation,
            mse,
            rust_performance: rust_output.performance,
            cpp_performance: cpp_output.performance,
            passed: true,
        })
    }

    fn validate_correlation(&self, correlation: f64) -> Result<()> {
        if correlation < self.tolerance.correlation_min {
            return Err(anyhow!(
                "Correlation {:.4} below minimum {:.4}",
                correlation,
                self.tolerance.correlation_min
            ));
        }
        Ok(())
    }

    fn validate_mse(&self, mse: f64) -> Result<()> {
        if mse > self.tolerance.mse_max {
            return Err(anyhow!(
                "MSE {:.2e} exceeds maximum {:.2e}",
                mse,
                self.tolerance.mse_max
            ));
        }
        Ok(())
    }
}
```

### Numerical Accuracy Validation

```rust
pub struct NumericalAccuracyValidator {
    tolerance: f64,  // 1e-6
}

impl NumericalAccuracyValidator {
    pub fn validate_operation(
        &self,
        operation: &str,
        rust_result: &[f32],
        cpp_result: &[f32],
    ) -> Result<()> {
        if rust_result.len() != cpp_result.len() {
            return Err(anyhow!(
                "Result length mismatch in {}: Rust={}, C++={}",
                operation,
                rust_result.len(),
                cpp_result.len()
            ));
        }

        for (i, (r, c)) in rust_result.iter().zip(cpp_result.iter()).enumerate() {
            let diff = (r - c).abs();
            if diff > self.tolerance {
                return Err(anyhow!(
                    "Numerical mismatch in {} at index {}: Rust={}, C++={}, diff={}",
                    operation,
                    i,
                    r,
                    c,
                    diff
                ));
            }
        }

        Ok(())
    }
}
```

---

## Implementation Roadmap

### Phase 1: Mock Elimination (Days 1-2)
**Goal**: Remove all hardcoded mock responses

**Tasks**:
- ✅ AC1: Compilation error resolution (COMPLETE)
- 🚧 AC2: Implement BITNET_STRICT_MODE=1 enforcement
- 🚧 Remove mock inference paths in engine.rs, streaming.rs
- 🚧 Eliminate ConcreteTensor::mock() calls in backends.rs

**Validation**:
```bash
BITNET_STRICT_MODE=1 cargo test --workspace --no-default-features --features cpu
```

### Phase 2: Kernel Integration (Days 3-7)
**Goal**: Activate real quantized computation

**Tasks**:
- 🚧 AC3: I2S kernel integration without dequantization fallback
- 🚧 AC4: TL1/TL2 device-aware kernel selection
- 🚧 Implement SIMD optimizations (AVX2/AVX-512/NEON)
- 🚧 Implement CUDA mixed precision kernels

**Validation**:
```bash
cargo test -p bitnet-quantization test_i2s_simd_scalar_parity
cargo test -p bitnet-kernels test_device_aware_quantization_selection
```

### Phase 3: QLinear Integration (Days 8-10)
**Goal**: Replace mock layers with real quantized layers

**Tasks**:
- 🚧 AC5: QLinear layer replacement in transformer architecture
- 🚧 GGUF tensor loading with quantization type detection
- 🚧 Validate tensor alignment for SIMD/CUDA operations

**Validation**:
```bash
cargo test -p bitnet-models test_quantized_linear_layer_integration
cargo run -p bitnet-cli -- compat-check model.gguf
```

### Phase 4: Performance Baselines (Days 11-13)
**Goal**: Establish realistic performance baselines

**Tasks**:
- 🚧 AC7: CPU performance baselines (10-20 tok/s)
- 🚧 AC8: GPU performance baselines (50-100 tok/s)
- 🚧 AC9: Cross-validation against C++ reference
- 🚧 AC6: CI pipeline mock evidence rejection

**Validation**:
```bash
cargo bench --workspace --no-default-features --features cpu
cargo test --workspace --no-default-features --features gpu test_gpu_smoke
cargo run -p xtask -- crossval --release
```

### Phase 5: Documentation and Finalization (Days 14-15)
**Goal**: Production readiness

**Tasks**:
- 🚧 AC10: Update performance documentation
- 🚧 Remove all mock-based performance claims
- 🚧 Comprehensive testing across feature flag combinations
- 🚧 Production stress testing and validation

**Validation**:
```bash
cargo run -p xtask -- verify
cargo run -p xtask -- verify-documentation
```

---

## Risk Assessment

### High Risk: Quantization Accuracy

**Risk**: Numerical precision loss in kernel implementations could result in >1% accuracy degradation vs FP32 reference.

**Impact**: Production deployment blocked, quantization claims invalidated

**Mitigation Strategy**:
- Comprehensive cross-validation with Microsoft C++ reference
- Unit tests for each kernel with accuracy validation
- Statistical validation across multiple models
- Gradual kernel activation with accuracy monitoring

**Validation**:
```bash
cargo test --features cpu test_quantization_accuracy_targets
export BITNET_GGUF="path/to/model.gguf"
cargo run -p xtask -- crossval --tolerance 0.01
```

### Medium Risk: Performance Regression

**Risk**: Real computation slower than expected baselines, blocking production adoption.

**Impact**: Performance targets not met, competitive disadvantage

**Mitigation Strategy**:
- Performance benchmarking and optimization iteration
- SIMD/CUDA profiling and bottleneck identification
- Conservative performance targets with optimization roadmap
- Device-aware optimization (AVX-512, CUDA mixed precision)

**Validation**:
```bash
cargo bench --features cpu -- --save-baseline baseline
cargo test --features cpu test_performance_baseline_establishment
```

### Medium Risk: Device Compatibility

**Risk**: GPU kernel failures on different CUDA versions or architectures.

**Impact**: GPU acceleration unavailable for some users

**Mitigation Strategy**:
- Device capability detection at runtime
- Graceful fallback to CPU when GPU unavailable
- CUDA version compatibility matrix testing
- Clear error messages for unsupported devices

**Validation**:
```bash
cargo test --features gpu test_device_aware_fallback_mechanisms
cargo test --features cpu,gpu test_gpu_cpu_fallback_chain
```

### Low Risk: GGUF Format Changes

**Risk**: Breaking changes in GGUF quantization format specifications.

**Impact**: Model loading failures for new GGUF versions

**Mitigation Strategy**:
- GGUF version detection and backward compatibility
- Legacy format support with deprecation warnings
- Comprehensive format validation in bitnet-cli
- Documentation of supported GGUF versions

**Validation**:
```bash
cargo run -p bitnet-cli -- compat-check --model path/to/model.gguf
cargo test -p bitnet-models test_gguf_version_detection
```

---

## Testing Strategy

### Unit Testing Approach

**Test-Driven Development with AC Tagging**:
```rust
#[test]
fn test_strict_mode_prevents_mock_fallback() { // AC:AC2
    std::env::set_var("BITNET_STRICT_MODE", "1");

    let enforcer = StrictModeEnforcer::new();
    let mock_path = MockInferencePath {
        description: "Test mock path".to_string(),
        uses_mock_computation: true,
        fallback_reason: "Kernel unavailable".to_string(),
    };

    let result = enforcer.validate_inference_path(&mock_path);
    assert!(result.is_err(), "Strict mode should prevent mock fallback");
}

#[test]
fn test_i2s_kernel_integration() { // AC:AC3
    let quantizer = I2SQuantizer::new();
    let input = create_test_tensor(vec![1.0, 2.0, 3.0, 4.0]);

    let quantized = quantizer.quantize_weights(&input).unwrap();
    assert_eq!(quantized.quantization_type, QuantizationType::I2S);

    // Validate no dequantization fallback in forward pass
    let provider = CpuI2SKernelProvider::new().unwrap();
    let result = provider.quantized_matmul_i2s(&input, &quantized).await.unwrap();
    assert!(result.is_real_quantized(), "Must use real quantized computation");
}

#[test]
fn test_qlinear_layer_replacement() { // AC:AC5
    let gguf_tensor = load_test_gguf_tensor("transformer.q_proj.weight");
    let qlinear = QuantizedLinearLayer::from_gguf_tensor("q_proj", &gguf_tensor, Device::Cpu).unwrap();

    assert_eq!(qlinear.quantization_type, QuantizationType::I2S);
    assert!(!qlinear.uses_mock_computation(), "QLinear must use real computation");
}
```

### Integration Testing Matrix

| Feature Combo | Test Focus | Command |
|---------------|------------|---------|
| `cpu` | SIMD kernel validation | `cargo test --no-default-features --features cpu` |
| `gpu` | CUDA kernel validation | `cargo test --no-default-features --features gpu` |
| `cpu,gpu` | Graceful fallback | `cargo test --no-default-features --features cpu,gpu` |
| `crossval` | C++ reference parity | `cargo test --no-default-features --features crossval` |
| None | Compilation only | `cargo build --no-default-features` |

### Performance Regression Testing

```bash
# Establish baseline
cargo bench --features cpu -- --save-baseline cpu-baseline

# Run after changes
cargo bench --features cpu -- --baseline cpu-baseline

# Validate no regression
cargo run -p xtask -- validate-performance-regression --tolerance 0.05
```

### Cross-Validation Testing

```bash
# Full cross-validation
export BITNET_GGUF="microsoft/bitnet-b1.58-2B-4T-gguf/model.gguf"
BITNET_DETERMINISTIC=1 BITNET_SEED=42 cargo run -p xtask -- crossval --release

# Validate correlation >99.5%
cargo test -p crossval test_cpp_reference_correlation
```

---

## Success Criteria

### Measurable Acceptance Criteria Summary

| AC | Status | Validation Command | Evidence |
|----|--------|-------------------|----------|
| AC1 | ✅ COMPLETE | `cargo build --workspace --no-default-features --features cpu` | Successful CI builds |
| AC2 | 🚧 PENDING | `BITNET_STRICT_MODE=1 cargo test -p bitnet-inference` | Test failures on mock paths |
| AC3 | 🚧 PENDING | `cargo test -p bitnet-quantization test_i2s_simd_scalar_parity` | >99.8% accuracy vs FP32 |
| AC4 | 🚧 PENDING | `cargo test -p bitnet-kernels test_device_aware_quantization` | Architecture-specific kernels |
| AC5 | 🚧 PENDING | `cargo test -p bitnet-models test_quantized_linear_layer_integration` | No mock tensor operations |
| AC6 | 🚧 PENDING | CI configuration changes | Failed builds on mock evidence |
| AC7 | 🚧 PENDING | `cargo bench --features cpu` | 10-20 tok/s CPU baseline |
| AC8 | 🚧 PENDING | `cargo test --features gpu test_gpu_smoke` | 50-100 tok/s GPU baseline |
| AC9 | 🚧 PENDING | `cargo run -p xtask -- crossval` | >99.5% correlation with C++ |
| AC10 | 🚧 PENDING | Documentation review | No mock performance claims |

### Production Readiness Checklist

- ✅ Compilation success across feature flag combinations
- 🚧 Strict mode enforcement prevents all mock fallbacks
- 🚧 I2S/TL1/TL2 quantization kernels integrated
- 🚧 QLinear layers replace mock linear layers
- 🚧 Realistic CPU performance baselines established
- 🚧 Realistic GPU performance baselines established
- 🚧 Cross-validation against C++ reference within tolerance
- 🚧 CI pipeline rejects mock evidence
- 🚧 Documentation updated with real capabilities
- 🚧 Performance monitoring and observability active

---

## Conclusion

This specification provides a comprehensive roadmap for eliminating mock inference paths in BitNet-rs and implementing real quantized computation. The approach prioritizes:

1. **Numerical Accuracy**: ≥99.8% correlation for I2S, ≥99.6% for TL1/TL2 vs FP32 reference
2. **Performance Transparency**: Realistic baselines (10-20 tok/s CPU, 50-100 tok/s GPU)
3. **Cross-Validation**: <5% variance from Microsoft C++ reference implementation
4. **Production Readiness**: Strict mode enforcement, device-aware optimization, comprehensive testing

**Key Success Factors**:
- Systematic mock elimination with strict mode enforcement
- Native quantized kernel integration without dequantization fallback
- Device-aware execution with graceful CPU/GPU fallback
- Cross-validation framework for numerical accuracy validation
- Comprehensive testing across feature flag combinations

**Timeline**: 9-15 days for complete implementation across all 10 acceptance criteria

**Next Steps**: NEXT → schema-validator for domain schema validation
