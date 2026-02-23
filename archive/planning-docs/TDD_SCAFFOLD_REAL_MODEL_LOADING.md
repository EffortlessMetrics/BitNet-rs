# TDD Scaffold Implementation Guide: Real Model Loading Tests

**File**: `crates/bitnet-models/tests/real_model_loading.rs`
**Priority**: MEDIUM - Requires BITNET_GGUF environment variable
**Estimated Complexity**: Low-Medium

## Overview

This file contains 7 tests for loading and validating real GGUF models. All tests are marked `#[ignore]` and require `BITNET_GGUF` environment variable pointing to a real model file.

## Scaffolds to Implement

### 1. Real Model Basic Loading (Line 63)
**Test**: `test_real_model_basic_loading`
**Goal**: Load real GGUF and validate basic metadata

**Implementation needs**:
- Load GGUF from `$BITNET_GGUF`
- Validate architecture metadata
- Check tensor count and names
- Verify no loading errors

### 2. Real Model Tensor Inspection (Line 106)
**Test**: `test_real_model_tensor_inspection`
**Goal**: Inspect individual tensors in real model
**Status**: Requires direct GGUF tensor access enhancement

### 3. Real Model GPU Loading (Line 128)
**Test**: `test_real_model_gpu_loading`
**Goal**: Load model to GPU if available

**Implementation needs**:
- Detect GPU availability
- Load model with GPU placement
- Validate tensors on GPU
- Graceful fallback if GPU unavailable

### 4. Real Model Inference Smoke Test (Line 151)
**Test**: `test_real_model_inference_smoke`
**Goal**: Run basic inference with real model

**Implementation needs**:
- Load model and tokenizer
- Run simple inference ("Hello world")
- Validate output tokens generated
- Check no crashes or errors

### 5. Real Model Cross-Validation (Line 185)
**Test**: `test_real_model_cross_validation`
**Goal**: Compare Rust vs C++ reference outputs

**Implementation needs**:
- Requires `BITNET_CPP_DIR` setup
- Run inference in both implementations
- Compare outputs with cosine similarity
- Validate accuracy ≥99%

### 6. Real Model Large Batch Loading (Line 206)
**Test**: `test_real_model_large_batch_loading`
**Goal**: Test loading large models (≥7B params)

**Implementation needs**:
- Load large model file
- Monitor memory usage
- Validate efficient memory-mapped loading
- Check no OOM errors

### 7. Real Model Quantization Validation (Line 247)
**Test**: `test_real_model_quantization_validation`
**Goal**: Validate quantization format detection

**Implementation needs**:
- Detect quantization types in model
- Validate I2S/TL1/TL2 tensors
- Check block sizes and scales
- Verify format compatibility

## Testing Commands

```bash
# Set environment variable to real model
export BITNET_GGUF=/path/to/model.gguf

# Run all real model tests
cargo test -p bitnet-models --features cpu --test real_model_loading -- --ignored

# Run specific test
cargo test -p bitnet-models --features cpu test_real_model_basic_loading -- --ignored --exact
```

## Success Criteria

- All tests pass with real GGUF model
- Loading is efficient (memory-mapped)
- GPU loading works or gracefully falls back
- Basic inference produces coherent output
- No crashes or panics with real models
