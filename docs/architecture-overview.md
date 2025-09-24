# Architecture Overview

This document provides a high-level overview of BitNet.rs architecture, design patterns, and key components.

## Workspace Structure

BitNet.rs is organized as a Rust workspace with specialized crates:

### Core Library
- **`bitnet`** (root): Main library with unified public API
- **`bitnet-common`**: Shared types, traits, and utilities
- **`bitnet-models`**: Model loading and format handling (GGUF, SafeTensors)
- **`bitnet-quantization`**: 1-bit quantization algorithms
- **`bitnet-kernels`**: High-performance SIMD/CUDA kernels with mixed precision support (FP16/BF16), FFI bridge for gradual C++ migration, plus comprehensive GPU detection utilities supporting CUDA, Metal, ROCm, and WebGPU backends
- **`bitnet-inference`**: Inference engine with streaming support
- **`bitnet-tokenizers`**: Universal tokenizer with GGUF integration and mock fallback system
- **`bitnet-server`**: HTTP server for BitNet inference with comprehensive health monitoring and real-time system metrics collection (CPU, memory, disk, network I/O)

### Compatibility Layer
- **`bitnet-compat`**: GGUF compatibility fixes and diagnostics
- **`bitnet-ffi`**: C API for llama.cpp drop-in replacement
- **`bitnet-py`**: Python 3.12+ bindings compatible with llama-cpp-python (PyO3 ABI3-py312)
- **`bitnet-wasm`**: WebAssembly bindings with enhanced browser/Node.js compatibility and optimized SIMD intrinsics

### Cross-Validation
- **`crossval`**: Framework for testing against C++ implementation
- Tests use `BITNET_GGUF` or `CROSSVAL_GGUF` environment variable for model path

## Key Design Patterns

1. **Feature-Gated Architecture**: Default features are **empty** - always specify features explicitly
2. **Zero-Copy Operations**: Memory-mapped models, careful lifetime management
3. **SIMD Abstraction**: Unified interface over platform-specific instructions
4. **Cross-Validation**: Systematic comparison with C++ for correctness
5. **Enhanced Validation Framework**: Comprehensive GPU/CPU validation with performance metrics and error tolerance
6. **FFI Bridge Architecture**: Safe C++ kernel integration for gradual migration with comprehensive testing and error handling
7. **Multi-Backend GPU Detection**: System-aware GPU detection with automatic fallback, supporting CUDA, Metal, ROCm, and WebGPU with mock testing capabilities
8. **GPU Infrastructure Access**: Low-level CUDA context and module access for advanced GPU programming (PR #199), enabling custom kernel loading and device-specific optimization
9. **Mixed Precision Computing**: Native CUDA kernels for FP16/BF16 operations with device-aware precision selection and automatic fallback (PR #202)

## Enhanced Quality Assurance Framework

BitNet.rs includes a comprehensive quality assurance system designed for production reliability:

### System Metrics and Monitoring (Enhanced in PR #208)
- **Real-Time System Monitoring**: Comprehensive system metrics collection using `sysinfo` crate
- **Performance Correlation**: Application performance metrics correlated with system resource usage
- **Prometheus Integration**: System metrics exposed via Prometheus endpoints for alerting and dashboards
- **Resource Tracking**: CPU usage, memory utilization, disk usage, and network I/O monitoring
- **Health Monitoring**: Service uptime tracking and performance regression detection

### Kernel Validation System
- **GPU/CPU Parity Testing**: Systematic validation between GPU and CPU implementations
- **Performance Benchmarking**: Built-in performance measurement with speedup calculations
- **Numerical Accuracy Testing**: Configurable tolerance testing for quantization operations
- **Memory Leak Detection**: Automatic GPU memory monitoring and leak prevention
- **Error Handling Validation**: Comprehensive error path testing with recovery verification

### Model Compatibility Validation System
- **Weight Mapper Integration**: GGUF tensor validation using weight mapper for compatibility checks
- **Unmapped Tensor Detection**: Detailed reporting of unmapped tensors with debugging metrics
- **Fixture-Based Testing**: Comprehensive test coverage for both success and corruption scenarios
- **Enhanced Error Reporting**: ValidationResult metrics include `unmapped_count` and `unmapped_tensors`
- **GGUF Parsing Integration**: Direct model file analysis for compatibility validation

### Universal Tokenizer Architecture (Enhanced in PR #171)
- **Auto-Detection**: Automatic backend selection based on GGUF model metadata
- **Enhanced GGUF Integration**: Direct extraction of tokenizer configuration from model files with optimized byte mapping
- **O(1) Byte Lookup Performance**: `byte_to_id[256]` array replaces HashMap for faster tokenization
- **Improved UTF-8 Handling**: Proper byte buffer management in decode operations for robust text processing
- **BOS Token Support**: Enhanced BasicTokenizer with vocab boundary checks and special token handling
- **SPM Compilation Fix**: Resolved critical compilation error in SentencePiece tokenizer integration
- **Fallback Strategy**: Graceful degradation to mock tokenizer for unsupported formats
- **Runtime Construction**: Build tokenizers from vocabulary and merge rules without external dependencies
- **Cross-Format Support**: BPE, SentencePiece, and custom tokenizer formats

### FFI Bridge System (New in PR #137)
- **Gradual Migration Support**: Safe C++ kernel integration enabling gradual transition to pure Rust
- **Quantization Bridge**: Complete FFI quantization support for I2S, TL1, and TL2 types
- **Performance Comparison Framework**: Built-in tools for comparing FFI vs Rust implementations
- **Error Handling Integration**: Enhanced C++ error propagation with `get_last_error()` bridge
- **Feature-Gated Safety**: Proper conditional compilation and graceful fallback when FFI unavailable
- **Migration Decision Support**: Automated recommendations based on performance and accuracy metrics

### Code Quality Enforcement
- **Comprehensive Clippy Integration**: Zero-tolerance policy for clippy warnings
- **Type Safety Improvements**: Enhanced type annotations and error handling
- **Documentation Standards**: Comprehensive inline documentation with examples
- **Test Coverage**: Extensive test suites with property-based testing
- **Performance Regression Testing**: Automated performance monitoring and validation

## Compatibility Guarantees

We maintain strict compatibility with llama.cpp while providing enhanced validation:
- C API functions have exact signature matches
- Python API is drop-in compatible
- We handle models that llama.cpp fails on (e.g., GPT-2 without pre-tokenizer)
- Enhanced GGUF parsing with tensor alignment validation for better error detection
- Robust handling of malformed GGUF files with detailed error messages
- See COMPATIBILITY.md for detailed guarantees

## Development Workflow

1. **Making Changes**: Always run tests for affected crates
2. **Before Committing**: Run `cargo fmt` and `cargo clippy`
3. **Cross-Validation**: Run `cargo xtask crossval` for inference changes
4. **Compatibility**: Check COMPATIBILITY.md before changing public APIs

For detailed information on specific components, see:
- [Quantization Support](quantization-support.md)
- [GPU Development Guide](gpu-development.md)
- [Tokenizer Architecture](tokenizer-architecture.md)
- [FFI Bridge Documentation](ffi-bridge.md)
- [Test Suite Guide](test-suite.md)