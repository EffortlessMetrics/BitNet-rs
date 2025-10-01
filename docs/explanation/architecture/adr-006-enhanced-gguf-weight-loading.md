# ADR-006: Enhanced GGUF Weight Loading with Device-Aware Quantization

## Status

Proposed

## Context

Issue #159 requires implementing real GGUF model weight loading for production neural network inference. The existing specifications provide a solid foundation, but gaps exist in:

1. **Device-Aware Tensor Placement**: No comprehensive strategy for GPU/CPU optimization
2. **Progressive Loading**: Limited framework for models >4GB with memory efficiency
3. **Quantization Integration**: Insufficient detail on I2_S, TL1, TL2 accuracy preservation
4. **Cross-Validation**: Basic framework needs enhancement for C++ reference compatibility
5. **Feature Flag Discipline**: Alignment with BitNet.rs `--no-default-features --features cpu|gpu`

## Decision

We will implement an enhanced GGUF weight loading architecture with the following components:

### 1. Device-Aware Tensor Placement System

**Implementation**: `DevicePlacementStrategy` trait with specialized implementations:
- `GpuOptimizedPlacement`: Prioritizes GPU memory with intelligent fallback
- `CpuOptimizedPlacement`: NUMA-aware memory allocation with SIMD optimization
- `HybridPlacement`: Dynamic placement based on tensor characteristics

**Rationale**:
- Enables automatic GPU/CPU selection (AC6)
- Provides graceful fallback mechanisms for production deployment
- Optimizes memory utilization up to 80% VRAM usage

### 2. Progressive Loading Architecture

**Implementation**: `ProgressiveGgufLoader` with streaming capabilities:
- Phase-based loading: Critical tensors → Layer batches → Remaining tensors
- Memory-mapped metadata with streaming data access
- Garbage collection between batches to maintain <150% memory footprint

**Rationale**:
- Enables loading models >4GB in constrained memory environments (AC7)
- Maintains memory efficiency requirements
- Provides loading progress monitoring for user feedback

### 3. Enhanced Quantization Integration

**Implementation**: `QuantizationAwareLoader` trait with format-specific loaders:
- `I2SQuantizationLoader`: 82-byte block GGML compatibility with accuracy validation
- `TLQuantizationLoader`: SIMD-optimized table lookup for TL1/TL2
- Real-time accuracy validation with ≥99% preservation requirement

**Rationale**:
- Ensures quantization accuracy preservation (AC2)
- Provides format-specific optimizations for performance
- Integrates validation into loading pipeline for early error detection

### 4. Cross-Validation Framework Enhancement

**Implementation**: `GgufCrossValidator` with C++ reference bridge:
- Tensor-by-tensor validation with <1e-5 numerical tolerance
- Full inference pipeline validation with deterministic configuration
- Comprehensive error reporting with actionable diagnostics

**Rationale**:
- Ensures compatibility with C++ reference implementation (AC5)
- Provides confidence in quantization accuracy
- Enables regression testing for future changes

### 5. Feature Flag Integration

**Implementation**: Feature-aware factory pattern:
```rust
// CPU-only compilation
cargo build --no-default-features --features cpu

// GPU-accelerated compilation
cargo build --no-default-features --features gpu

// Minimal compilation (testing only)
cargo build --no-default-features --features cpu
```

**Rationale**:
- Maintains BitNet.rs feature flag discipline
- Enables selective compilation for deployment environments
- Reduces binary size and dependencies for specific use cases

## Architecture Components

### Core Interfaces

```rust
pub trait DevicePlacementStrategy {
    fn place_tensor(&self, tensor_info: &TensorInfo, available_memory: &DeviceMemoryInfo, model_config: &ModelConfig) -> Result<TensorPlacement>;
    fn validate_memory_requirements(&self, total_size_bytes: u64, tensor_count: usize) -> Result<MemoryValidationResult>;
    fn handle_placement_failure(&self, error: &DevicePlacementError, fallback_options: &[Device]) -> Result<Device>;
}

pub trait QuantizationAwareLoader {
    fn load_quantized_tensor(&self, tensor_info: &TensorInfo, target_device: &Device, validation_config: &QuantizationValidationConfig) -> Result<QuantizedTensorResult>;
    fn validate_quantization_accuracy(&self, quantized_tensor: &QuantizedTensor, reference_tensor: &ReferenceTensor, tolerance: f32) -> Result<QuantizationAccuracyReport>;
    fn supported_quantization_types(&self) -> &[QuantizationType];
}
```

### Memory Efficiency Requirements

- **Loading Overhead**: <150% of model size during loading (AC7)
- **Zero-Copy Operations**: Memory-mapped access for tensors ≥1MB
- **Progressive Thresholds**: Enable for models >4GB
- **Garbage Collection**: Automatic cleanup between loading phases

### Performance Targets

- **Quantization Speed**: >66 Melem/s for I2_S operations
- **Inference Throughput**: >200 tok/s for loaded models
- **Memory Bandwidth**: 80% theoretical utilization on target hardware
- **Loading Time**: <30 seconds for 7B parameter models

### Validation Requirements

- **Accuracy Preservation**: ≥99% for all quantization formats (AC2)
- **Numerical Tolerance**: <1e-5 absolute error vs C++ reference (AC5)
- **Deterministic Validation**: `BITNET_DETERMINISTIC=1 BITNET_SEED=42`
- **Cross-Platform Consistency**: Identical results across Linux/macOS/Windows

## Consequences

### Positive

1. **Production Readiness**: Enables real neural network inference with trained parameters
2. **Memory Efficiency**: Supports large models in constrained environments
3. **Device Optimization**: Automatic GPU/CPU placement with performance optimization
4. **Accuracy Assurance**: Comprehensive validation against reference implementation
5. **Feature Flexibility**: Selective compilation based on deployment requirements

### Negative

1. **Implementation Complexity**: Sophisticated placement and loading logic increases codebase complexity
2. **Testing Requirements**: Extensive cross-validation and device-specific testing needed
3. **Performance Overhead**: Validation and placement logic adds computational cost
4. **Memory Fragmentation**: Progressive loading may increase memory fragmentation

### Mitigation Strategies

1. **Modular Design**: Clear separation of concerns with well-defined interfaces
2. **Comprehensive Testing**: TDD approach with `// AC:ID` tags for traceability
3. **Performance Benchmarking**: Continuous monitoring of performance regressions
4. **Memory Profiling**: Real-time monitoring of memory usage patterns

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)
- Implement `DevicePlacementStrategy` trait and basic implementations
- Create `ProgressiveGgufLoader` with metadata parsing
- Set up feature flag integration patterns

### Phase 2: Quantization Integration (Week 3-4)
- Implement `I2SQuantizationLoader` with accuracy validation
- Create `TLQuantizationLoader` with SIMD optimization
- Integrate quantization loaders with progressive loading

### Phase 3: Cross-Validation (Week 5-6)
- Implement `GgufCrossValidator` with C++ bridge
- Create comprehensive test suite with reference validation
- Validate accuracy requirements across all quantization formats

### Phase 4: Optimization & Testing (Week 7-8)
- Performance optimization and memory profiling
- Comprehensive integration testing
- Documentation and usage examples

## Validation Criteria

The implementation will be considered successful when:

1. **AC1**: All transformer layer weights load correctly from GGUF files
2. **AC2**: Quantization accuracy ≥99% vs FP32 reference for I2_S, TL1, TL2
3. **AC3**: Comprehensive tensor validation with actionable error messages
4. **AC5**: Cross-validation with C++ reference <1e-5 tolerance
5. **AC6**: Device-aware placement with automatic GPU/CPU selection
6. **AC7**: Memory efficiency <150% during loading, progressive loading >4GB
7. **AC9**: Backward compatibility with existing mock tensor functionality
8. **AC10**: Complete documentation with usage examples

## References

- [Issue #159 Specification](/docs/explanation/issue-159-spec.md)
- [GGUF Weight Loading Architecture](/docs/explanation/gguf-weight-loading.md)
- [Enhanced Architecture Specification](/docs/explanation/gguf-weight-loading-enhanced-architecture.md)
- [Quantization-Aware API Contracts](/docs/explanation/quantization-aware-weight-loading-contracts.md)
- [BitNet.rs Feature Flag Discipline](/CLAUDE.md)

## Decision Date

2025-09-26

## Decision Makers

- spec-creator (BitNet.rs Generative Flow)
- Architecture review pending