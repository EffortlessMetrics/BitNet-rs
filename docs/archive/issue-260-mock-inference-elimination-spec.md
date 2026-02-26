# Issue #260: Mock Inference Elimination - Technical Specification

## Executive Summary

This specification outlines the technical approach for eliminating mock inference paths in BitNet-rs and implementing real quantized computation. The current system reports misleading performance metrics (200.0 tokens/sec) through mock inference fallbacks, blocking accurate evaluation of 1-bit neural network capabilities.

**Classification**: Breaking (Architecture Change)
**Migration Path**: [docs/explanation/migration/mock-elimination-migration.md] (to be created)
**Risk Level**: High (Core inference pipeline changes)

## Requirements Analysis

### Functional Requirements

1. **Real Quantized Computation**: Replace all mock inference paths with actual I2S, TL1, TL2 quantized matrix multiplication
2. **Performance Accuracy**: Report realistic throughput based on device-aware quantization kernels
3. **Strict Mode Enforcement**: Implement BITNET_STRICT_MODE=1 to prevent mock fallbacks
4. **Cross-Validation**: Maintain compatibility with Microsoft C++ reference implementation
5. **Compilation Fixes**: Resolve all blocking compilation errors preventing real inference execution

### Quantization Constraints

- **I2S (2-bit signed)**: Target 99.8% correlation with FP32 reference, production accuracy
- **TL1 (table lookup 1)**: Target 99.6% correlation, optimized for ARM NEON
- **TL2 (table lookup 2)**: Target 99.6% correlation, optimized for x86 AVX2/AVX-512
- **Memory Efficiency**: Support models up to 8GB with device-aware quantization selection
- **Numerical Stability**: Maintain deterministic inference with proper seeding (BITNET_DETERMINISTIC=1)

### Performance Requirements

- **CPU Baseline**: 10-20 tokens/sec for I2S quantization on modern x86_64
- **GPU Baseline**: 50-100 tokens/sec with mixed precision FP16/BF16 acceleration
- **Memory Bandwidth**: Optimize for cache efficiency with SIMD-aligned layouts
- **Cross-Validation Tolerance**: Within 5% performance variance from C++ reference

## Architecture Approach

### Crate-Specific Implementation Strategy

#### 1. bitnet-quantization: Kernel Integration (Critical Path)

**Current State**: Quantization algorithms exist but kernels not integrated into forward pass
**Required Changes**:
- Activate I2S, TL1, TL2 kernel integration in quantized matrix multiplication
- Replace dequantization fallbacks with native quantized operations
- Implement device-aware quantization selection logic

**Implementation Details**:
```rust
// Enhanced quantized_linear.rs integration
impl QuantizedLinear {
    async fn forward_i2s(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        // BEFORE: self.fallback_i2s_matmul(&input_2d).await?
        // AFTER: Direct kernel invocation without dequantization
        let provider = self.kernel_manager.select_best()?;
        self.quantized_matmul_i2s(&input_2d, provider).await
    }
}
```

**Validation Commands**:
```bash
cargo test --no-default-features -p bitnet-quantization --no-default-features --features cpu test_i2s_simd_scalar_parity
cargo test --no-default-features -p bitnet-quantization --no-default-features --features gpu test_device_aware_quantization
```

#### 2. bitnet-inference: Mock Removal (Primary Focus)

**Current State**: Mock inference paths in engine.rs line 1804 and backends.rs warmup
**Required Changes**:
- Remove `"mock generated text"` return paths
- Eliminate dummy tensor creation in backend warmup
- Implement strict mode validation to prevent mock fallbacks

**Mock Elimination Strategy**:
- **engine.rs**: Replace mock tokenizer with real tokenizer integration
- **backends.rs**: Remove `ConcreteTensor::mock()` warmup calls
- **streaming.rs**: Eliminate hardcoded "mock" responses

**Validation Commands**:
```bash
# Strict mode testing
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-inference --no-default-features --features cpu
# Performance regression prevention
cargo test --no-default-features -p bitnet-inference test_real_vs_mock_comparison --no-default-features --features cpu
```

#### 3. bitnet-kernels: I2S/TL1/TL2 Activation (Core Infrastructure)

**Current State**: Kernel providers exist but fallback paths still active
**Required Changes**:
- Ensure I2S kernels work without dequantization fallback
- Optimize TL1 for ARM NEON vectorization
- Optimize TL2 for x86 AVX2/AVX-512 with larger lookup tables

**Device-Aware Optimization**:
```rust
// Kernel selection priority order
1. GPU: CUDA I2S kernel (highest throughput)
2. CPU x86: AVX-512 > AVX2 > fallback
3. CPU ARM: NEON > fallback
4. Lookup tables: TL1 (ARM), TL2 (x86)
```

**Validation Commands**:
```bash
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_gpu_info_summary
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_mixed_precision_matmul_accuracy
```

#### 4. bitnet-models: QLinear Integration (GGUF Enhancement)

**Current State**: GGUF tensor loading complete, but no QLinear layer replacement
**Required Changes**:
- Replace standard linear layers with QuantizedLinear in transformer architecture
- Enhance tensor validation for quantized weight formats
- Support mixed quantization types within single model

**GGUF Integration Strategy**:
```rust
// Enhanced model loading with quantized layers
impl BitNetModel {
    pub fn load_quantized_layers(&mut self, tensors: &HashMap<String, CandleTensor>) -> Result<()> {
        // Detect quantization type from tensor metadata
        // Replace linear layers with QuantizedLinear
        // Validate tensor alignment and weight mapping
    }
}
```

**Validation Commands**:
```bash
cargo test --no-default-features --features cpu -p bitnet-models --test gguf_min -- test_tensor_alignment
cargo run -p bitnet-cli -- compat-check --model path/to/model.gguf
```

#### 5. crossval: Baseline Establishment (Validation Framework)

**Current State**: C++ reference comparison available but performance baselines undefined
**Required Changes**:
- Establish realistic CPU/GPU performance baselines
- Implement regression detection for performance claims
- Validate numerical accuracy against Microsoft C++ implementation

**Validation Commands**:
```bash
export BITNET_GGUF="path/to/model.gguf"
cargo run -p xtask -- crossval --release
cargo run -p xtask -- fetch-cpp && cargo run -p xtask -- crossval
```

### Feature Flag Architecture

**Build Configurations**:
```bash
# CPU-only build with strict validation
cargo build --no-default-features --features cpu
cargo test --no-default-features --workspace --no-default-features --features cpu

# GPU-accelerated build with mixed precision
cargo build --no-default-features --features gpu
cargo test --no-default-features --workspace --no-default-features --features gpu

# Cross-validation build
cargo build --no-default-features --features cpu,crossval
```

**Feature Dependencies**:
- `cpu`: SIMD-optimized CPU inference (AVX2/AVX-512/NEON)
- `gpu`: CUDA acceleration with mixed precision (FP16/BF16)
- `crossval`: C++ reference validation framework
- `strict`: Prevent mock fallbacks (environment variable)

## Quantization Strategy

### I2S (2-bit Signed) Implementation

**Precision Analysis**:
- Target: 99.8% correlation with FP32 reference
- Block size: 82 elements (SIMD-aligned)
- Memory efficiency: 4:1 compression ratio
- Device support: CPU SIMD, GPU CUDA

**Optimization Approach**:
```rust
// I2S kernel optimization
- CPU: AVX2/AVX-512 vectorization for 82-element blocks
- GPU: CUDA mixed precision (FP16 scales, INT2 weights)
- Cache: Precomputed lookup tables for common scale values
- Memory: Zero-copy dequantization when possible
```

### TL1 (Table Lookup 1) Implementation

**ARM NEON Optimization**:
- Target: 99.6% correlation with FP32 reference
- Lookup table size: 16-256 entries (cache-friendly)
- Vectorization: ARM NEON 16-byte alignment
- Memory pattern: Sequential lookup for cache efficiency

### TL2 (Table Lookup 2) Implementation

**x86 AVX Optimization**:
- Target: 99.6% correlation with FP32 reference
- Lookup table size: 256-4096 entries (larger tables)
- Vectorization: AVX2 32-byte, AVX-512 64-byte alignment
- Memory pattern: Blocked access for larger lookup tables

## GPU/CPU Implementation

### Device-Aware Execution Strategy

**Automatic Backend Selection**:
1. **GPU Detection**: Check CUDA availability and device capabilities
2. **CPU Fallback**: Graceful degradation to optimized CPU kernels
3. **Mixed Precision**: FP16/BF16 on GPU, maintain FP32 accuracy on CPU
4. **Memory Management**: Device-aware allocation and transfer optimization

**GPU Acceleration (CUDA)**:
```rust
// GPU implementation highlights
- Mixed precision: FP16 activations, INT2/INT4 weights
- Kernel fusion: Combine quantization and matrix multiplication
- Memory optimization: Minimize CPU-GPU transfers
- Batch processing: Optimize for throughput over latency
```

**CPU Optimization (SIMD)**:
```rust
// CPU implementation highlights
- SIMD vectorization: AVX2/AVX-512 (x86), NEON (ARM)
- Cache optimization: Block-wise processing, aligned memory
- Threading: Rayon parallelization for large matrices
- Numerical stability: Proper rounding and overflow handling
```

## GGUF Integration

### Format Compatibility Assessment

**Current Capabilities**:
- ✅ GGUF header parsing and metadata extraction
- ✅ Tensor loading and memory mapping
- ✅ Weight validation and alignment checking
- ❌ QLinear layer replacement (missing)
- ❌ Quantized tensor format detection (partial)

**Enhanced Integration Requirements**:
```rust
// GGUF quantized layer detection
impl GGUFLoader {
    fn detect_quantization_type(tensor_meta: &TensorMetadata) -> Result<QuantizationType> {
        match tensor_meta.dtype {
            GGMLType::I2_S => Ok(QuantizationType::I2S),
            GGMLType::TL1 => Ok(QuantizationType::TL1),
            GGMLType::TL2 => Ok(QuantizationType::TL2),
            _ => Err(UnsupportedQuantization(tensor_meta.dtype))
        }
    }
}
```

**Validation Strategy**:
- Tensor alignment validation for quantized formats
- Metadata consistency checking across quantization types
- Weight mapping verification for QLinear layer replacement
- Cross-validation against reference implementations

## Performance Specifications

### Throughput Targets

**CPU Performance (Intel x86_64)**:
- I2S quantization: 15-20 tokens/sec (AVX2), 20-25 tokens/sec (AVX-512)
- TL1 quantization: 12-18 tokens/sec (optimized lookup)
- TL2 quantization: 10-15 tokens/sec (larger lookup overhead)
- Memory bandwidth: 80-90% efficiency with SIMD alignment

**GPU Performance (NVIDIA CUDA)**:
- I2S quantization: 60-100 tokens/sec (mixed precision)
- TL1/TL2 quantization: 50-80 tokens/sec (lookup overhead)
- Memory bandwidth: 85-95% efficiency with coalesced access
- Batch processing: 2-4x throughput improvement with batch_size > 1

**ARM Performance (Apple Silicon)**:
- I2S quantization: 10-15 tokens/sec (NEON optimization)
- TL1 quantization: 12-18 tokens/sec (optimized for ARM)
- TL2 quantization: 8-12 tokens/sec (fallback)
- Memory bandwidth: 75-85% efficiency with NEON alignment

### Memory Usage Targets

**Model Size Support**:
- Small models (1-2B parameters): 2-4GB memory usage
- Medium models (3-7B parameters): 4-8GB memory usage
- Large models (13B+ parameters): 8GB+ with streaming
- Quantization overhead: <5% additional memory for lookup tables

## Cross-Validation Plan

### C++ Reference Compatibility

**Validation Approach**:
```bash
# Numerical accuracy validation
export BITNET_GGUF="microsoft/bitnet-b1.58-2B-4T-gguf/model.gguf"
cargo run -p xtask -- crossval --release

# Performance comparison
BITNET_DETERMINISTIC=1 BITNET_SEED=42 cargo run -p xtask -- crossval
```

**Accuracy Requirements**:
- Correlation with C++ implementation: >99.5%
- MSE tolerance: <1e-6 for FP32 comparisons
- Performance variance: <5% from C++ baseline
- Deterministic reproduction: Identical outputs with same seed

**FFI Bridge Validation**:
```bash
# FFI accuracy testing (when available)
cargo test --no-default-features -p bitnet-kernels --no-default-features --features ffi
cargo test --no-default-features --workspace --no-default-features --features cpu,ffi test_ffi_parity
```

## Feature Flag Analysis

### Build Configuration Matrix

**Default Configuration** (Empty features by design):
```bash
cargo build --no-default-features  # Compilation only, no inference
```

**CPU Configuration**:
```bash
cargo build --no-default-features --features cpu
# Enables: SIMD kernels, CPU fallback, optimized quantization
```

**GPU Configuration**:
```bash
cargo build --no-default-features --features gpu
# Enables: CUDA kernels, mixed precision, GPU memory management
```

**Development Configuration**:
```bash
cargo build --no-default-features --features cpu,gpu,crossval,strict
# Full feature matrix for development and testing
```

### Dependency Management

**Feature-Gated Dependencies**:
- `candle-core`: Always available (tensor operations)
- `candle-nn`: GPU feature only (neural network layers)
- `cudarc`: GPU feature only (CUDA runtime)
- Cross-validation: C++ FFI bridge when available

## Testing Strategy

### Unit Testing Approach

**Test-Driven Development**:
```rust
// AC:ID tagging for acceptance criteria traceability
#[test]
fn test_strict_mode_prevents_mock_fallback() { // AC:2
    std::env::set_var("BITNET_STRICT_MODE", "1");
    // Verify mock fallback causes failure
}

#[test]
fn test_i2s_kernel_integration() { // AC:3
    // Verify I2S kernels work without dequantization
}
```

**Testing Matrix**:
- Unit tests: Individual component validation
- Integration tests: Cross-crate interaction
- Smoke tests: Basic functionality on CPU/GPU
- Performance tests: Regression detection
- Cross-validation: C++ reference comparison

### Validation Commands

**Development Validation**:
```bash
# Basic compilation and unit tests
cargo test --no-default-features --workspace --no-default-features --features cpu

# Strict mode validation
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-inference --no-default-features --features cpu

# Performance regression testing
cargo test --no-default-features -p bitnet-inference test_performance_targets --no-default-features --features cpu
```

**Production Validation**:
```bash
# Cross-validation against C++ reference
export BITNET_GGUF="path/to/production/model.gguf"
cargo run -p xtask -- crossval --release

# GPU smoke testing
cargo test --no-default-features --workspace --no-default-features --features gpu test_gpu_smoke
```

## Risk Assessment

### Technical Risks

**High Risk - Quantization Accuracy**:
- Risk: Numerical precision loss in kernel implementations
- Mitigation: Comprehensive cross-validation with C++ reference
- Validation: `cargo test --no-default-features --features cpu test_quantization_accuracy_targets`
- Fallback: Gradual kernel activation with accuracy monitoring

**Medium Risk - Performance Regression**:
- Risk: Real computation slower than expected baselines
- Mitigation: Performance benchmarking and optimization iteration
- Validation: `cargo test --no-default-features --features cpu test_performance_baseline_establishment`
- Fallback: Conservative performance targets with optimization roadmap

**Medium Risk - Device Compatibility**:
- Risk: GPU kernel failures on different CUDA versions
- Mitigation: Device capability detection and graceful fallback
- Validation: `cargo test --no-default-features --features cpu test_device_aware_fallback_mechanisms`
- Fallback: CPU-only operation with warning messages

**Low Risk - GGUF Format Changes**:
- Risk: Breaking changes in GGUF quantization format
- Mitigation: Version detection and backward compatibility
- Validation: `cargo run -p bitnet-cli -- compat-check`
- Fallback: Legacy format support with deprecation warnings

### Compilation Risk Mitigation

**Current State**: No blocking compilation errors detected (Sept 2025)
**Risk**: Potential issues in CI environments or feature combinations
**Mitigation Strategy**:
```bash
# Comprehensive build matrix validation
cargo build --no-default-features --features cpu --workspace --no-default-features                    # Minimal
cargo build --no-default-features --workspace --no-default-features --features cpu     # CPU
cargo build --no-default-features --workspace --no-default-features --features gpu     # GPU
cargo build --no-default-features --workspace --no-default-features --features cpu,gpu # Full
```

## Success Criteria

### Measurable Acceptance Criteria

**AC1: Compilation Fixes** ✅
- Status: All workspace crates compile successfully
- Validation: `cargo build --workspace --no-default-features --features cpu`
- Evidence: Successful CI builds without compilation errors

**AC2: Strict Mode Implementation**
- Target: BITNET_STRICT_MODE=1 prevents all mock fallbacks
- Validation: `BITNET_STRICT_MODE=1 cargo test --no-default-features --features cpu test_strict_mode_enforcement`
- Evidence: Test failures when mock paths attempted

**AC3: I2S Kernel Integration**
- Target: I2S quantization without dequantization fallback
- Validation: `cargo test --no-default-features --features cpu test_i2s_native_quantized_matmul`
- Evidence: Direct kernel invocation in profiling traces

**AC4: TL1/TL2 Kernel Integration**
- Target: Device-aware TL1 (ARM) and TL2 (x86) optimization
- Validation: `cargo test --no-default-features --features cpu test_device_aware_quantization_selection`
- Evidence: Architecture-specific kernel selection

**AC5: QLinear Replacement**
- Target: Replace mock linear layers with real quantized operations
- Validation: `cargo test --no-default-features --features cpu test_quantized_linear_layer_integration`
- Evidence: No mock tensor operations in inference pipeline

**AC6: CI Mock Evidence Rejection**
- Target: CI pipeline rejects performance claims from mock inference
- Validation: CI configuration changes and test integration
- Evidence: Failed CI builds when mock evidence detected

**AC7: CPU Performance Baselines**
- Target: 10-20 tokens/sec I2S quantization on modern CPU
- Validation: `cargo run -p xtask -- benchmark --cpu-baseline`
- Evidence: Consistent performance measurements across test runs

**AC8: GPU Performance Baselines**
- Target: 50-100 tokens/sec with mixed precision acceleration
- Validation: `cargo run -p xtask -- benchmark --gpu-baseline`
- Evidence: GPU utilization >80% during inference

**AC9: Cross-Validation Accuracy**
- Target: <5% performance variance from C++ reference
- Validation: `cargo run -p xtask -- crossval --tolerance 0.05`
- Evidence: Correlation >99.5% with Microsoft implementation

**AC10: Documentation Updates**
- Target: Performance documentation reflects real capabilities
- Validation: Documentation review and accuracy verification
- Evidence: Removal of mock-based performance claims

### Performance Validation Thresholds

**Quantization Accuracy Targets**:
- I2S: 99.8% correlation with FP32 reference (production requirement)
- TL1: 99.6% correlation with FP32 reference (optimized fallback)
- TL2: 99.6% correlation with FP32 reference (optimized fallback)
- Cross-validation: 99.5% correlation with C++ implementation

**Throughput Validation Ranges**:
- CPU I2S: 10-25 tokens/sec (device dependent)
- GPU I2S: 50-120 tokens/sec (mixed precision, device dependent)
- Memory efficiency: >80% bandwidth utilization
- Latency: <100ms first token, <50ms subsequent tokens

## Implementation Phases

### Phase 1: Mock Elimination (Immediate)
- Remove hardcoded mock responses in engine.rs and streaming.rs
- Implement BITNET_STRICT_MODE environment variable
- Add compilation error resolution for any remaining issues
- Timeline: 1-2 days

### Phase 2: Kernel Integration (Core)
- Activate I2S, TL1, TL2 kernels in quantized_linear.rs
- Remove dequantization fallbacks where native kernels available
- Implement device-aware quantization selection
- Timeline: 3-5 days

### Phase 3: QLinear Integration (GGUF)
- Replace standard linear layers with QuantizedLinear in models
- Enhance GGUF loading for quantized layer detection
- Validate tensor alignment and weight mapping
- Timeline: 2-3 days

### Phase 4: Performance Baselines (Validation)
- Establish realistic CPU/GPU performance baselines
- Implement cross-validation against C++ reference
- Configure CI pipeline to reject mock evidence
- Timeline: 2-3 days

### Phase 5: Documentation and Validation (Finalization)
- Update performance documentation with real capabilities
- Comprehensive testing across feature flag combinations
- Production readiness validation and stress testing
- Timeline: 1-2 days

**Total Estimated Timeline**: 9-15 days for complete implementation

## Conclusion

This specification provides a comprehensive roadmap for eliminating mock inference paths in BitNet-rs while implementing real quantized computation. The approach prioritizes numerical accuracy, performance transparency, and cross-validation against reference implementations.

Key success factors:
1. **Systematic Mock Elimination**: Remove all fallback paths that bypass real quantization
2. **Kernel Integration**: Activate optimized I2S, TL1, TL2 kernels for device-aware execution
3. **Performance Accuracy**: Establish realistic baselines based on actual quantized computation
4. **Cross-Validation**: Maintain compatibility with Microsoft C++ reference implementation
5. **Production Readiness**: Ensure reliability, determinism, and observability for deployment

The implementation will transform BitNet-rs from a prototype with mock inference to a production-grade 1-bit neural network inference engine with accurate performance reporting and validated quantization accuracy.
