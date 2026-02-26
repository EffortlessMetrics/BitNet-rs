# GGUF Model Weight Loading Architecture Specification

## Executive Summary

This document provides the comprehensive architectural blueprint for implementing real GGUF model weight loading in BitNet-rs (Issue #159). The current implementation only loads token embeddings and output projections while mock-initializing all transformer layer weights with zeros and ones. This specification defines the complete architecture to parse, validate, and load all quantized model weights from GGUF files, enabling real neural network inference.

## Business Value

Implementing real GGUF weight loading unlocks the complete BitNet-rs inference pipeline:

- **Model Loading**: Parse all transformer layer weights from GGUF files with quantization support
- **Quantization**: Validate real I2_S, TL1, TL2 quantized weights against C++ reference implementation
- **Inference**: Enable meaningful computations with trained parameters instead of zero weights
- **Output**: Generate valid text and perform useful neural network tasks
- **Performance**: Achieve production-ready inference with memory optimization and GPU acceleration

## Scope

### Affected Workspace Crates

**Primary Impact:**
- `bitnet-models` - Core GGUF weight loading and parsing logic
- `bitnet-quantization` - I2_S dequantization integration and validation
- `bitnet-inference` - Integration with loaded weights for real inference

**Secondary Impact:**
- `bitnet-kernels` - Device-aware tensor operations with real data
- `bitnet-common` - Enhanced error handling for weight parsing failures
- `crossval` - Cross-validation framework for accuracy verification

### Pipeline Stages

- **Model Loading** (Primary): Replace mock initialization with real weight parsing
- **Quantization** (Secondary): Validate quantized weights accuracy (≥99%)
- **Inference** (Secondary): Integration with meaningful model parameters
- **Kernels** (Secondary): GPU/CPU operations on real quantized data
- **Output** (Secondary): Valid text generation with trained weights

## User Stories

### US1: Real Weight Loading
**As a** BitNet-rs developer
**I want to** load all transformer layer weights from GGUF files
**So that** the neural network can perform meaningful inference with trained parameters instead of producing meaningless outputs from zero-initialized weights

**Value:** Enables functional neural network inference
**Acceptance Criteria:** AC1, AC2, AC10

### US2: Quantization Accuracy
**As a** machine learning engineer
**I want** quantized weights to maintain ≥99% accuracy compared to FP32 reference
**So that** model performance is preserved during quantization

**Value:** Ensures model quality and numerical stability
**Acceptance Criteria:** AC2, AC5

### US3: Robust Error Handling
**As a** production system integrator
**I want** descriptive error messages and graceful fallbacks for unsupported tensors
**So that** the system fails safely with actionable diagnostics

**Value:** Enables reliable production deployment
**Acceptance Criteria:** AC4, AC9

### US4: Device-Aware Performance
**As a** performance engineer
**I want** memory-efficient loading with GPU/CPU optimization
**So that** large models can be loaded efficiently on different hardware configurations

**Value:** Enables scalable inference across hardware configurations
**Acceptance Criteria:** AC6, AC7

## Technical Requirements

### TR1: GGUF Parsing Architecture (AC1)
Replace mock tensor initialization in `crates/bitnet-models/src/gguf_simple.rs` with comprehensive weight parsing:

```rust
// Current (mock initialization)
tensor_map.insert(
    format!("{}.attn_q.weight", prefix),
    CandleTensor::zeros(&[hidden_size, hidden_size], dtype, &cdevice)?,
);

// Required (real weight parsing)
let tensor_info = find_tensor(&parsed.tensors, &format!("{}.attn_q.weight", prefix))?;
let weight_data = parse_tensor_data(&mmap, &tensor_info, &quantizer)?;
tensor_map.insert(format!("{}.attn_q.weight", prefix), weight_data);
```

**Implementation Points:**
- Extend `gguf_min.rs` tensor selection logic to cover all layer types
- Add tensor name mapping for different model architectures (LLaMA, BitNet variants)
- Implement progressive tensor loading with memory management
- Support tensor shape validation and alignment verification

### TR2: Quantization Integration (AC2)
Integrate with existing `bitnet-quantization` infrastructure for I2_S, TL1, TL2 support:

```rust
// Quantization workflow integration
match tensor_info.qtype {
    GgufTensorType::F32 => load_f32_tensor(&mmap, &tensor_info),
    GgufTensorType::F16 => load_f16_tensor(&mmap, &tensor_info),
    GgufTensorType::I2_S => {
        let quantizer = I2SQuantizer::with_block_size(32);
        let quantized = parse_i2s_tensor(&mmap, &tensor_info)?;
        quantizer.dequantize(&quantized, device)
    },
    // TL1, TL2 support via existing quantizers
}
```

**Integration Requirements:**
- Leverage existing `I2SQuantizer`, `TL1Quantizer`, `TL2Quantizer` implementations
- Maintain ≥99% accuracy against FP32 reference through cross-validation
- Support block-wise quantization parameters and scale factors
- Implement quantization parameter validation and consistency checks

### TR3: Memory Efficiency (AC7)
Implement zero-copy operations and memory-mapped file access:

```rust
// Memory-efficient tensor loading
pub struct MmapTensorLoader<'a> {
    mmap: &'a Mmap,
    data_offset: u64,
    alignment: u64,
}

impl<'a> MmapTensorLoader<'a> {
    pub fn load_tensor_zero_copy(&self, info: &TensorInfo) -> Result<Cow<'a, [u8]>> {
        // Zero-copy when possible, copy only when required for alignment
        let offset = (self.data_offset + info.offset) as usize;
        if is_properly_aligned(offset, info.dtype) {
            Ok(Cow::Borrowed(&self.mmap[offset..offset + info.size_bytes]))
        } else {
            // Copy to aligned buffer when necessary
            Ok(Cow::Owned(copy_aligned(&self.mmap[offset..offset + info.size_bytes])))
        }
    }
}
```

### TR4: Device-Aware Operations (AC6)
Support CPU and GPU feature flags with proper tensor placement:

```rust
// Device-aware tensor placement
pub fn create_tensor_on_device(
    data: Vec<f32>,
    shape: &[usize],
    device: Device
) -> Result<CandleTensor> {
    let cdevice = match device {
        Device::Cpu => CDevice::Cpu,
        Device::Cuda(id) => CDevice::new_cuda(id)?,
        Device::Metal => return Err(BitNetError::UnsupportedDevice),
    };

    CandleTensor::from_vec(data, shape, &cdevice)
        .map_err(|e| BitNetError::TensorOperation(e.to_string()))
}
```

## Public API Contracts

### Core Loading Interface
```rust
/// Enhanced GGUF loader with full weight parsing
pub struct GgufWeightLoader {
    quantization_enabled: bool,
    device_placement: DevicePlacement,
    memory_optimization: MemoryOptimization,
}

impl GgufWeightLoader {
    /// Load complete model with all transformer weights
    pub fn load_complete_model(
        &self,
        path: &Path,
        device: Device,
    ) -> Result<(BitNetConfig, HashMap<String, CandleTensor>)> {
        // AC1: Parse all transformer layer weights
        // AC2: Support quantization formats with ≥99% accuracy
        // AC6: Device-aware tensor placement
        // AC7: Memory-efficient loading
    }

    /// Validate loaded weights against expected schema
    pub fn validate_weights(
        &self,
        weights: &HashMap<String, CandleTensor>,
        config: &BitNetConfig,
    ) -> Result<ValidationReport> {
        // AC3: Tensor metadata validation
        // AC4: Descriptive error messages
    }
}
```

### Error Handling Contracts
```rust
/// Enhanced error types for weight loading
#[derive(Debug, thiserror::Error)]
pub enum WeightLoadingError {
    #[error("Tensor '{name}' not found in GGUF file")]
    TensorNotFound { name: String },

    #[error("Tensor '{name}' has invalid shape: expected {expected:?}, got {actual:?}")]
    InvalidTensorShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>
    },

    #[error("Unsupported quantization type {qtype} for tensor '{name}'")]
    UnsupportedQuantization { name: String, qtype: String },

    #[error("Quantization accuracy below threshold: {accuracy}% < {required}%")]
    QuantizationAccuracyError { accuracy: f32, required: f32 },
}
```

### Tensor Schema Validation
```rust
/// Tensor naming and shape validation
pub struct TensorSchema {
    pub attention_layers: AttentionLayerSchema,
    pub feedforward_layers: FeedforwardLayerSchema,
    pub normalization_layers: NormalizationLayerSchema,
}

pub struct AttentionLayerSchema {
    pub query_weight: TensorSpec,      // [hidden_size, hidden_size]
    pub key_weight: TensorSpec,        // [hidden_size, hidden_size]
    pub value_weight: TensorSpec,      // [hidden_size, hidden_size]
    pub output_weight: TensorSpec,     // [hidden_size, hidden_size]
}
```

## Implementation Schema

### Phase 1: Core Weight Parsing
**File:** `crates/bitnet-models/src/gguf_weight_loader.rs`
```rust
// New comprehensive weight loading module
pub mod gguf_weight_loader {
    use super::*;
    use crate::gguf_min::{Parsed, TensorInfo};

    /// Parse all transformer layer weights from GGUF
    pub fn parse_all_weights(
        parsed: &Parsed,
        mmap: &[u8],
        config: &BitNetConfig,
        device: Device,
    ) -> Result<HashMap<String, CandleTensor>> {
        let mut weights = HashMap::new();

        // Parse attention layers (AC1)
        for layer_idx in 0..config.model.num_layers {
            parse_attention_layer(&mut weights, parsed, mmap, layer_idx, device)?;
            parse_feedforward_layer(&mut weights, parsed, mmap, layer_idx, device)?;
            parse_normalization_layer(&mut weights, parsed, mmap, layer_idx, device)?;
        }

        // Validate all loaded weights (AC3)
        validate_weight_consistency(&weights, config)?;

        Ok(weights)
    }
}
```

### Phase 2: Quantization Integration
**File:** `crates/bitnet-models/src/quantized_weight_parser.rs`
```rust
// Integration with bitnet-quantization
pub struct QuantizedWeightParser {
    i2s_quantizer: I2SQuantizer,
    tl1_quantizer: TL1Quantizer,
    tl2_quantizer: TL2Quantizer,
}

impl QuantizedWeightParser {
    /// Parse and dequantize weight tensor (AC2)
    pub fn parse_quantized_tensor(
        &self,
        tensor_info: &TensorInfo,
        mmap: &[u8],
        device: &Device,
    ) -> Result<CandleTensor> {
        match tensor_info.qtype {
            GgufTensorType::I2_S => {
                let quantized = self.extract_i2s_data(tensor_info, mmap)?;
                let dequantized = self.i2s_quantizer.dequantize(&quantized, device)?;
                Ok(candle_tensor_from_bitnet(dequantized))
            },
            // Additional quantization types...
        }
    }
}
```

### Phase 3: Enhanced gguf_simple.rs
**File:** `crates/bitnet-models/src/gguf_simple.rs`
```rust
// Replace mock initialization with real weight loading
pub fn load_gguf(
    path: &Path,
    device: Device,
) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
    let parsed = crate::gguf_min::parse_gguf_file(path)?;
    let mmap = create_memory_map(path)?;

    // Load embeddings and output (existing functionality)
    let mut tensor_map = HashMap::new();
    load_embeddings_and_output(&mut tensor_map, &parsed, &mmap, device)?;

    // NEW: Load all transformer layer weights (AC1)
    let weight_loader = GgufWeightLoader::new();
    let transformer_weights = weight_loader.parse_all_weights(&parsed, &mmap, device)?;
    tensor_map.extend(transformer_weights);

    // NEW: Cross-validation against C++ reference (AC5)
    if std::env::var("BITNET_CROSSVAL_WEIGHTS").is_ok() {
        crossvalidate_loaded_weights(&tensor_map)?;
    }

    Ok((config, tensor_map))
}
```

## Performance Specifications

### Memory Efficiency Requirements

**P1: Zero-Copy Operations (AC7)**
- Memory-mapped file access for tensor data ≥1MB
- Zero-copy tensor creation when alignment permits
- Copy-on-demand for misaligned data with 4KB boundaries
- Target: <10% memory overhead for zero-copy operations

**P2: Memory Footprint (AC7)**
- Maximum 150% of model size in memory during loading
- Progressive tensor loading for models >4GB
- Garbage collection of temporary buffers after tensor creation
- Target: Load 7B parameter model in <12GB RAM

### Performance Baselines

**P3: Loading Performance**
- 7B parameter model loading: <30 seconds on NVMe SSD
- GPU tensor placement: <5 seconds additional overhead
- Quantized weight dequantization: <20% loading time overhead
- Target: 2GB/s sustained read throughput from storage

**P4: Cross-Validation Performance (AC5)**
- Numerical accuracy validation: <1 second per layer
- C++ reference comparison: <5% total loading time overhead
- Deterministic inference validation: <10 seconds additional
- Target: Cross-validation adds <30 seconds to total load time

### Device-Aware Optimization

**P5: GPU Memory Management (AC6)**
- Automatic GPU memory optimization based on available VRAM
- CPU fallback for layers that don't fit in GPU memory
- Mixed precision (FP16/BF16) support for memory efficiency
- Target: Support models up to 80% of available VRAM

**P6: CPU SIMD Optimization**
- AVX2/NEON optimization for quantization operations
- Multi-threaded tensor parsing with Rayon thread pool
- NUMA-aware memory allocation on multi-socket systems
- Target: 80% of theoretical memory bandwidth utilization

## Validation Requirements

### Cross-Validation Framework (AC5)

**V1: C++ Reference Compatibility**
```rust
// Cross-validation integration
pub fn crossvalidate_loaded_weights(
    weights: &HashMap<String, CandleTensor>,
) -> Result<CrossValidationReport> {
    let cpp_reference = load_cpp_reference_weights()?;
    let mut report = CrossValidationReport::new();

    for (name, weight) in weights {
        let reference = cpp_reference.get(name)
            .ok_or_else(|| CrossValidationError::MissingReference(name.clone()))?;

        let accuracy = calculate_numerical_accuracy(weight, reference)?;
        if accuracy < 0.99 {  // AC5: ≥99% accuracy requirement
            return Err(CrossValidationError::AccuracyBelowThreshold {
                tensor: name.clone(),
                accuracy,
                threshold: 0.99,
            });
        }

        report.add_result(name.clone(), accuracy);
    }

    Ok(report)
}
```

**V2: Deterministic Validation (AC5)**
- Deterministic inference with `BITNET_DETERMINISTIC=1 BITNET_SEED=42`
- Reproducible outputs across CPU/GPU devices
- Numerical tolerance: <1e-5 for cross-validation comparisons
- Validation against golden reference outputs

### Accuracy Metrics (AC2)

**V3: Quantization Accuracy Validation**
- I2_S quantization: ≥99% cosine similarity vs FP32
- TL1/TL2 quantization: ≥99% numerical accuracy
- Block-wise scale factor validation
- Range and distribution preservation analysis

**V4: End-to-End Validation**
- Complete inference pipeline validation
- Text generation quality metrics (BLEU, perplexity)
- Performance regression testing
- Memory usage profiling and optimization validation

## Test Architecture

### Test Data Requirements

**T1: GGUF Test Fixtures**
```bash
# Test model structure
tests/fixtures/
├── bitnet-test-model.gguf          # Complete BitNet model for integration testing
├── llama-test-model.gguf           # LLaMA model for compatibility testing
├── malformed-tensors.gguf          # Error handling testing
└── quantized-weights.gguf          # Quantization-specific testing
```

**T2: Cross-Validation Test Data**
- C++ reference implementation outputs for comparison
- Golden reference tensors for numerical accuracy validation
- Deterministic inference test cases with expected outputs
- Performance benchmark datasets

### Test Implementation Strategy

**T3: TDD Scaffolding with AC Tags**
```rust
// Test structure aligned with acceptance criteria
#[cfg(test)]
mod weight_loading_tests {
    use super::*;

    #[test]
    // AC1: Parse and load all transformer layer weights
    fn test_complete_weight_parsing_ac1() {
        let model_path = "tests/fixtures/bitnet-test-model.gguf";
        let (config, weights) = load_gguf(Path::new(model_path), Device::Cpu).unwrap();

        // Verify all expected weights are loaded
        for layer in 0..config.model.num_layers {
            assert!(weights.contains_key(&format!("blk.{}.attn_q.weight", layer)));
            assert!(weights.contains_key(&format!("blk.{}.attn_k.weight", layer)));
            assert!(weights.contains_key(&format!("blk.{}.attn_v.weight", layer)));
            assert!(weights.contains_key(&format!("blk.{}.attn_output.weight", layer)));
        }
    }

    #[test]
    // AC2: Support quantization formats with ≥99% accuracy
    fn test_quantization_accuracy_ac2() {
        let quantizer = I2SQuantizer::new();
        let test_tensor = create_test_tensor();

        let quantized = quantizer.quantize_tensor(&test_tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

        let accuracy = calculate_cosine_similarity(&test_tensor, &dequantized);
        assert!(accuracy >= 0.99, "Accuracy {} below required 99%", accuracy);
    }

    #[test]
    // AC3: Implement robust tensor metadata validation
    fn test_tensor_validation_ac3() {
        // Test shape validation, alignment checks, parameter validation
    }

    // Additional tests for AC4-AC10...
}
```

**T4: Property-Based Testing**
```rust
// Property-based tests for quantization validation
#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        // Quantization round-trip property testing
        fn quantization_roundtrip_preserves_distribution(
            data in prop::collection::vec(-10.0f32..10.0f32, 32..1024)
        ) {
            let quantizer = I2SQuantizer::new();
            let tensor = create_tensor_from_data(data.clone());

            let quantized = quantizer.quantize_tensor(&tensor)?;
            let dequantized = quantizer.dequantize_tensor(&quantized)?;

            // Property: distribution should be approximately preserved
            let original_std = calculate_std(&data);
            let dequant_data = extract_tensor_data(&dequantized);
            let dequant_std = calculate_std(&dequant_data);

            prop_assert!((original_std - dequant_std).abs() / original_std < 0.1);
        }
    }
}
```

### Integration Test Strategy

**T5: End-to-End Pipeline Testing**
```rust
#[test]
// AC9: Maintain backward compatibility with mock loading
fn test_backward_compatibility_ac9() {
    // Ensure mock loading still works for development
    let mock_weights = load_gguf_with_mocks(test_path, Device::Cpu).unwrap();
    let real_weights = load_gguf(test_path, Device::Cpu).unwrap();

    // Verify API compatibility
    assert_eq!(mock_weights.0.model.vocab_size, real_weights.0.model.vocab_size);
    assert_eq!(mock_weights.1.len(), real_weights.1.len()); // Same number of tensors
}

#[test]
// AC5: Cross-validation against C++ reference
fn test_cpp_cross_validation_ac5() {
    std::env::set_var("BITNET_DETERMINISTIC", "1");
    std::env::set_var("BITNET_SEED", "42");

    let (config, weights) = load_gguf(test_model_path, Device::Cpu).unwrap();
    let validation_result = crossvalidate_loaded_weights(&weights).unwrap();

    // All weights must meet accuracy threshold
    assert!(validation_result.all_above_threshold(0.99));
}
```

## Risk Mitigation

### Performance Risks

**R1: Memory Usage**
- **Risk**: Excessive memory usage during loading large models
- **Mitigation**: Progressive loading, memory monitoring, garbage collection
- **Validation**: Memory usage tests with 7B+ parameter models

**R2: Loading Performance**
- **Risk**: Slow model loading affecting user experience
- **Mitigation**: Memory-mapped files, zero-copy operations, parallel parsing
- **Validation**: Performance benchmarks with loading time targets

### Compatibility Risks

**R3: GGUF Format Changes**
- **Risk**: New GGUF versions breaking compatibility
- **Mitigation**: Version detection, graceful degradation, comprehensive testing
- **Validation**: Testing across GGUF v2/v3, compatibility matrix

**R4: Quantization Accuracy**
- **Risk**: Quantized weights causing inference quality degradation
- **Mitigation**: Rigorous cross-validation, accuracy thresholds, fallback mechanisms
- **Validation**: Continuous accuracy monitoring, reference implementation comparison

### Integration Risks

**R5: Device Compatibility**
- **Risk**: GPU/CPU feature flag interactions causing failures
- **Mitigation**: Device-aware fallbacks, comprehensive device testing
- **Validation**: Multi-device test matrix, graceful degradation testing

**R6: Backward Compatibility**
- **Risk**: Breaking existing mock loading functionality
- **Mitigation**: Feature flags, parallel implementation paths, API preservation
- **Validation**: Regression testing, API compatibility validation

## Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
- [ ] Implement `GgufWeightLoader` with basic tensor parsing
- [ ] Extend `gguf_min.rs` for comprehensive tensor selection
- [ ] Add tensor schema validation and error handling
- [ ] Basic unit tests with AC tags

### Phase 2: Quantization Integration (Weeks 3-4)
- [ ] Integrate I2_S, TL1, TL2 quantization support
- [ ] Implement quantization accuracy validation
- [ ] Add cross-validation framework integration
- [ ] Quantization-specific test coverage

### Phase 3: Performance Optimization (Weeks 5-6)
- [ ] Memory-mapped file optimization
- [ ] Zero-copy tensor operations
- [ ] Device-aware memory management
- [ ] Performance benchmarking and profiling

### Phase 4: Integration & Validation (Weeks 7-8)
- [ ] Integration with `gguf_simple.rs`
- [ ] End-to-end pipeline testing
- [ ] Cross-validation with C++ reference
- [ ] Documentation and API stabilization

## Success Metrics

### Functional Metrics
- **Weight Loading**: 100% of transformer layer weights loaded from GGUF files
- **Quantization Accuracy**: ≥99% accuracy maintained for all quantization formats
- **Cross-Validation**: 100% pass rate against C++ reference implementation
- **Error Handling**: Descriptive errors for all failure modes with recovery suggestions

### Performance Metrics
- **Memory Efficiency**: <150% of model size memory usage during loading
- **Loading Speed**: <30 seconds for 7B parameter model on NVMe SSD
- **GPU Utilization**: Support for models up to 80% of available VRAM
- **CPU Optimization**: 80% of theoretical memory bandwidth utilization

### Quality Metrics
- **Test Coverage**: >95% line coverage for weight loading code paths
- **Regression Testing**: 100% backward compatibility with existing APIs
- **Documentation**: Complete API documentation with usage examples
- **Cross-Platform**: Support for Linux, macOS, Windows with consistent behavior

## Conclusion

This architectural blueprint provides a comprehensive foundation for implementing real GGUF model weight loading in BitNet-rs. The specification addresses all 10 acceptance criteria from Issue #159 while maintaining compatibility with the existing neural network inference pipeline.

Key architectural decisions:
- **Incremental Enhancement**: Build upon existing `gguf_min.rs` infrastructure
- **Device-Aware Design**: Support CPU/GPU with automatic optimization
- **Quantization Integration**: Leverage existing quantization implementations
- **Performance Focus**: Memory efficiency and zero-copy operations
- **Quality Assurance**: Comprehensive testing and cross-validation

The implementation will enable BitNet-rs to perform meaningful neural network inference with trained model parameters, unlocking the full potential of the inference pipeline for production use cases.

## Referenced Documents

**Core Documentation:**
- `COMPATIBILITY.md` - API stability guarantees
- `MIGRATION.md` - Migration guide from llama.cpp
- `docs/development/validation-framework.md` - Testing and cross-validation framework
- `docs/development/gpu-development.md` - CUDA development guidelines

**API Reference Cross-Links:**
- [`docs/reference/api-reference.md`](../reference/api-reference.md) - Comprehensive API documentation for all crates
- [`docs/reference/quantization-support.md`](../reference/quantization-support.md) - Quantization format specifications (I2_S, TL1, TL2)
- [`docs/reference/implementation-schemas.md`](../reference/implementation-schemas.md) - Implementation contracts and schemas
- [`docs/reference/real-model-api-contracts.md`](../reference/real-model-api-contracts.md) - Real model API contracts and interfaces
- [`docs/reference/api-compatibility.md`](../reference/api-compatibility.md) - API compatibility and stability guarantees

**Related Specifications:**
- [`docs/explanation/gguf-weight-loading-api-contracts.md`](./gguf-weight-loading-api-contracts.md) - Detailed API contract specifications
- [`docs/explanation/gguf-weight-loading-performance-validation.md`](./gguf-weight-loading-performance-validation.md) - Performance requirements and validation
- [`docs/explanation/gguf-weight-loading-integration-testing.md`](./gguf-weight-loading-integration-testing.md) - Integration testing framework
