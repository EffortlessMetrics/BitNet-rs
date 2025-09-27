# Quantization Support

This document describes the quantization formats and device-aware acceleration supported by BitNet.rs.

## Supported Quantization Formats

BitNet-rs supports multiple quantization formats with advanced device-aware acceleration:

### I2_S - Native Rust Implementation (Production Ready)
- Native Rust implementation with intelligent GPU/CPU selection and automatic fallback
- Device-aware quantization with CUDA kernel acceleration and CPU SIMD optimization
- **Accuracy**: ≥99.8% correlation with FP32 reference (production requirement)
- **Performance**: CPU 10-20 tok/s, GPU 50-100 tok/s with mixed precision
- 2-bit signed quantization with optimized bit-packing (4 values per byte)
- **Strict Mode**: Use `BITNET_STRICT_MODE=1` to prevent mock fallbacks

### TL1 - Table Lookup Quantization (ARM Optimized)
- Table lookup quantization optimized for ARM NEON architecture
- **Accuracy**: ≥99.6% correlation with FP32 reference
- **Performance**: Device-aware selection with ARM NEON vectorization
- Memory-efficient lookup tables (16-256 entries, cache-friendly)
- Parallel processing with configurable block sizes

### TL2 - Advanced Table Lookup (x86 Optimized)
- Advanced table lookup quantization optimized for x86 AVX2/AVX-512
- **Accuracy**: ≥99.6% correlation with FP32 reference
- **Performance**: Optimized for x86 architectures with larger lookup tables
- Enhanced vectorized operations (256-4096 entry tables) for large tensor processing
- CPU feature detection with SIMD optimization fallbacks (AVX2 32-byte, AVX-512 64-byte)

### IQ2_S - GGML-Compatible
- GGML-compatible quantization with 82-byte block layout and 4-level [-2,-1,1,2] mapping

### Standard Formats (Planned)
- Q4_0, Q5_0, Q8_0, etc. (planned for future releases)

## Device-Aware Operations

All quantizers support device-aware operations with:

- **Automatic GPU acceleration**: CUDA kernels with performance monitoring (50-100 tok/s)
- **Transparent CPU fallback**: Graceful degradation with maintained accuracy (10-20 tok/s)
- **Memory optimization**: GPU memory leak detection and efficient allocation
- **Feature gating**: Proper `#[cfg(feature = "gpu")]` guards for CPU-only builds
- **Strict Mode Enforcement**: `BITNET_STRICT_MODE=1` prevents mock fallbacks
- **FFI Bridge Support**: C++ kernel integration for I2S, TL1, and TL2 quantization (requires `--features ffi`)
- **Cross-Validation**: <5% performance variance from C++ reference implementation

## FFI Quantization Bridge

The FFI bridge enables gradual migration from C++ to Rust while maintaining functionality:

- **Quantization Types**: Full support for I2S, TL1, and TL2 via C++ kernels
- **Performance Comparison**: Built-in tools to compare FFI vs Rust quantization
- **Migration Path**: Systematic approach to replace C++ kernels with native Rust
- **Safety**: Safe Rust wrappers with proper error handling and memory management
- **Testing**: Comprehensive test suite ensuring FFI/Rust quantization parity

## Mixed Precision GPU Acceleration

BitNet.rs provides native CUDA mixed precision support for enhanced GPU performance:

### Supported Precision Modes
- **FP32**: Full precision (reference implementation)
- **FP16**: Half-precision floating point with Tensor Core acceleration (compute capability 6.1+)
- **BF16**: Brain floating point format for modern architectures (compute capability 8.0+)
- **Auto**: Automatic precision selection based on device capabilities

### Device-Aware Precision Selection
- **Automatic Detection**: Hardware capability detection determines optimal precision
- **Device ID Tracking**: GPU kernels expose device ID for multi-GPU debugging scenarios (PR #201)
- **Capability Querying**: Direct access to FP16/BF16 support via `supports_fp16()` and `supports_bf16()` methods (PR #201)
- **Graceful Fallback**: Automatic CPU fallback when GPU operations fail
- **Performance Monitoring**: Comprehensive metrics for each precision mode
- **Memory Tracking**: GPU memory allocation and deallocation monitoring
- **Tensor Core Optimization**: Leverages WMMA API for maximum performance (CC 7.0+)

### Mixed Precision Features
- **Native CUDA Kernels**: Custom PTX kernels optimized for each precision mode
- **Matrix Multiplication**: Optimized matmul operations with device-specific launch parameters
- **Precision Conversion**: Efficient FP32↔FP16↔BF16 conversion utilities
- **Memory Optimization**: Vectorized memory operations and bandwidth optimization
- **Error Handling**: Comprehensive error propagation with detailed diagnostics

## Testing Commands

### Device-Aware Quantization Testing
```bash
# Test device-aware quantization with strict mode (prevents mock fallbacks)
BITNET_STRICT_MODE=1 cargo test -p bitnet-quantization --no-default-features --features gpu test_dequantize_cpu_and_gpu_paths

# GPU kernel validation with numerical accuracy testing
BITNET_STRICT_MODE=1 cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_vs_cpu_quantization_accuracy

# Enhanced GPU validation with performance metrics and error handling
BITNET_STRICT_MODE=1 cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_validation_comprehensive

# Validate quantization accuracy targets (I2S >99.8%, TL1/TL2 >99.6%)
cargo test -p bitnet-quantization --no-default-features --features cpu test_quantization_accuracy_targets
```

### Mixed Precision Testing
```bash
# Test mixed precision with strict mode (no mock GPU fallbacks)
BITNET_STRICT_MODE=1 cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_kernel_creation

# Test FP16/BF16 matrix multiplication accuracy against FP32 reference
BITNET_STRICT_MODE=1 cargo test -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_matmul_accuracy

# Test precision mode validation and automatic fallback
BITNET_STRICT_MODE=1 cargo test -p bitnet-kernels --no-default-features --features gpu test_precision_mode_validation

# Benchmark mixed precision performance with strict mode (realistic baselines)
BITNET_STRICT_MODE=1 cargo bench -p bitnet-kernels --bench mixed_precision_bench --no-default-features --features gpu

# Test device-aware precision selection and optimization
BITNET_STRICT_MODE=1 cargo test -p bitnet-kernels --no-default-features --features gpu test_precision_detection_optimization
```

### FFI Quantization Testing
```bash
# FFI quantization bridge validation with strict mode
BITNET_STRICT_MODE=1 cargo test -p bitnet-kernels --features ffi test_ffi_quantize_matches_rust

# FFI kernel creation and availability testing
BITNET_STRICT_MODE=1 cargo test -p bitnet-kernels --features ffi test_ffi_kernel_creation

# FFI performance comparison against C++ reference (cross-validation)
BITNET_STRICT_MODE=1 cargo test -p bitnet-kernels --features ffi --release test_performance_comparison_structure

# Cross-validation with C++ reference implementation
BITNET_GGUF="path/to/model.gguf" BITNET_STRICT_MODE=1 cargo run -p xtask -- crossval
```

### SIMD Testing
```bash
# SIMD kernel validation and performance testing
cargo test -p bitnet-quantization --test simd_compatibility --no-default-features --features cpu
cargo bench -p bitnet-quantization --bench simd_comparison --no-default-features --features cpu

# SIMD vs scalar parity testing
cargo test -p bitnet-quantization test_i2s_simd_scalar_parity
cargo test -p bitnet-quantization test_simd_performance_baseline
```

For more information, see:
- [GPU Development Guide](../development/gpu-development.md) - GPU-specific quantization details
- [Build Commands](../development/build-commands.md) - Build commands for different quantization features
- [FFI Threading Architecture](../ffi-threading-architecture.md) - FFI bridge details