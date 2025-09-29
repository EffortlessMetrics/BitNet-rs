# Architecture Overview

This document provides a high-level overview of BitNet.rs architecture, design patterns, and key components.

## Workspace Structure

BitNet.rs is organized as a Rust workspace with specialized crates:

### Core Library
- **`bitnet`** (root): Main library with unified public API and production-ready GGUF weight loading
- **`bitnet-common`**: Shared types, traits, utilities, and enhanced error types for GGUF operations
- **`bitnet-models`**: **Enhanced model loading with real GGUF weight parsing** - replaces mock tensor initialization with comprehensive transformer layer weight loading (AC1), supporting all quantization formats with device-aware placement
- **`bitnet-quantization`**: Real quantized computation with I2S (≥99.8%), TL1/TL2 (≥99.6%) accuracy validation vs FP32 baselines - **STRICT MODE ENFORCED** to prevent mock fallbacks
- **`bitnet-kernels`**: **Device-aware quantization kernels** with SIMD/CUDA acceleration, mixed precision support (FP16/BF16), automatic CPU/GPU selection, FFI bridge for C++ cross-validation, plus comprehensive GPU detection utilities supporting CUDA, Metal, ROCm, and WebGPU backends
- **`bitnet-inference`**: **Mock-free inference engine** with autoregressive generation, multi-head attention, real quantized linear layers (QLinear), KV-cache optimization, streaming support, and authentic GGUF model integration - **STRICT MODE prevents all mock fallbacks**
- **`bitnet-tokenizers`**: Universal tokenizer with GGUF integration, automatic discovery, and graceful fallback system

### Application Layer
- **`bitnet-server`**: **Production HTTP/REST inference server** providing scalable inference endpoints with batch processing, model hot-swapping capabilities, comprehensive health monitoring (liveness/readiness/startup probes), real-time system metrics collection (CPU, memory, disk, network I/O), Prometheus metrics integration, OpenTelemetry observability, streaming inference support, and deployment-ready configurations for Docker and Kubernetes environments
- **`bitnet-cli`**: Command-line interface for local inference, model verification, and compatibility checking

### Compatibility Layer
- **`bitnet-compat`**: GGUF compatibility fixes and diagnostics
- **`bitnet-ffi`**: C API for llama.cpp drop-in replacement
- **`bitnet-py`**: Python 3.12+ bindings compatible with llama-cpp-python (PyO3 ABI3-py312)
- **`bitnet-wasm`**: WebAssembly bindings with enhanced browser/Node.js compatibility and optimized SIMD intrinsics

### Cross-Validation
- **`crossval`**: Framework for testing against C++ implementation
- Tests use `BITNET_GGUF` or `CROSSVAL_GGUF` environment variable for model path

## Production-Ready GGUF Weight Loading Architecture

BitNet.rs implements a comprehensive GGUF weight loading system that replaces mock tensor initialization with real neural network model parsing. This system represents a major architectural advancement enabling meaningful neural network inference.

### Core GGUF Loading Pipeline

#### 1. Enhanced GGUF Parser (`bitnet-models::gguf_simple`)

```rust
pub fn load_gguf(
    path: &Path,
    device: Device,
) -> Result<(BitNetConfig, HashMap<String, CandleTensor>)>
```

**Pipeline Stages:**

1. **Memory-Mapped File Access** - Zero-copy GGUF file access via `MmapFile`
2. **Enhanced Parser Attempt** - Try comprehensive GGUF reader with full validation
3. **Fallback Parser** - Graceful degradation to minimal parser for backward compatibility
4. **Device-Aware Tensor Placement** - Automatic GPU/CPU placement with fallback
5. **Comprehensive Validation** - Security checks, tensor completeness, shape validation

#### 2. Transformer Weight Categories Loaded

**Attention Layers (All Transformer Blocks):**
- `layers.{i}.attention.wq` - Query projection weights
- `layers.{i}.attention.wk` - Key projection weights
- `layers.{i}.attention.wv` - Value projection weights
- `layers.{i}.attention.wo` - Output projection weights

**Feed-Forward Layers (SwiGLU Architecture):**
- `layers.{i}.feed_forward.w1` - Gate projection
- `layers.{i}.feed_forward.w2` - Down projection
- `layers.{i}.feed_forward.w3` - Up projection

**Normalization Layers:**
- `layers.{i}.attention_norm.weight` - Pre-attention RMSNorm
- `layers.{i}.ffn_norm.weight` - Pre-FFN RMSNorm

**Embedding & Output:**
- `token_embd.weight` - Token embedding matrix
- `output.weight` - Language modeling head

#### 3. Quantization Format Support

**I2_S (2-bit Signed) - Production Recommended:**
- Values: [-2, -1, 1, 2] with optimal accuracy
- Performance: 66+ Melem/s (CPU), 200+ Melem/s (GPU)
- Accuracy: ≥99% vs FP32 baseline

**TL1/TL2 (Table Lookup Quantization):**
- TL1: Linear mapping optimized for ARM (NEON)
- TL2: Non-linear mapping optimized for x86 (AVX2/AVX-512)
- Device-aware selection for optimal performance

**Legacy Format Support:**
- F32, F16: Full/half precision for accuracy comparison
- IQ2_S: GGML-compatible 82-byte blocks via FFI bridge

#### 4. Device-Aware Architecture

**GPU Acceleration:**
```rust
let cdevice = match device {
    Device::Cuda(id) => match CDevice::new_cuda(id) {
        Ok(cuda_device) => {
            tracing::info!("Using CUDA device {} for tensor placement", id);
            cuda_device
        }
        Err(e) => {
            tracing::warn!("CUDA device {} unavailable, falling back to CPU: {}", id, e);
            CDevice::Cpu
        }
    },
    // ... other device types
};
```

**CPU Fallback Strategy:**
- Automatic detection of GPU availability
- Graceful degradation with performance logging
- Optimal SIMD kernel selection (AVX2/AVX-512/NEON)

#### 5. Security and Validation Framework

**Pre-Loading Security Checks:**
- GGUF magic byte validation ('GGUF')
- Version compatibility (1-3 supported)
- Tensor count bounds checking (< 10^6 security limit)
- KV pair count validation (< 10^5 security limit)
- File size sanity checks

**Tensor Completeness Validation:**
```rust
fn validate_tensor_completeness(
    tensor_infos: &HashMap<String, TensorInfo>,
    config: &BitNetConfig,
) -> Result<()>
```

- Verifies all required transformer layers present
- Validates tensor shapes against model configuration
- Checks quantization format compatibility
- Ensures memory alignment requirements

#### 6. Error Handling and Recovery

**Enhanced Error Types:**
- `GgufParseError`: Detailed GGUF parsing errors with context
- `QuantizationError`: Quantization-specific errors with recovery suggestions
- `ValidationError`: Model validation failures with diagnostic information
- `SecurityError`: Security limit violations with actionable guidance

**Recovery Strategies:**
- Automatic fallback from enhanced to minimal parser
- Mock tensor generation for test compatibility
- CPU fallback for GPU memory failures
- Alternative quantization format suggestions

### Performance Characteristics

**Loading Performance:**
- Zero-copy operations where possible
- Memory-mapped file access for large models
- Parallel tensor loading for multi-core systems
- Device-aware placement optimization

**Memory Efficiency:**
- 2GB parameter models load in <1.5GB RAM
- GPU memory pooling for tensor operations
- Efficient cache management for repeated loads
- Memory-mapped model sharing across instances

**Accuracy Guarantees:**
- I2_S quantization: ≥99% accuracy vs FP32
- Cross-validation against C++ reference implementation
- Systematic regression testing for accuracy preservation
- Property-based testing for numerical stability

## Production Server Architecture

### `bitnet-server` Crate Overview

The `bitnet-server` crate provides a production-ready HTTP/REST inference server built on the BitNet.rs inference engine. It serves as the application layer for deploying BitNet models in production environments.

**Key Components:**
- **Inference Engine Integration**: Direct integration with `bitnet-inference` for autoregressive generation
- **Model Management**: Hot-swappable model loading with graceful failover and validation
- **Health Monitoring**: Three-tier health check system (liveness, readiness, startup)
- **System Metrics**: Real-time collection of CPU, memory, disk, and network I/O metrics via `sysinfo`
- **Observability**: Prometheus metrics and OpenTelemetry integration for distributed tracing
- **Streaming Support**: Server-sent events (SSE) for real-time token streaming
- **Batch Processing**: Request batching for improved throughput

**Architecture Position:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              bitnet-server (HTTP/REST)                │   │
│  │  • Axum web framework                                 │   │
│  │  • Health endpoints (/health, /ready, /live)         │   │
│  │  • Inference endpoints (/v1/completions)             │   │
│  │  • Metrics endpoints (/metrics)                       │   │
│  │  • Streaming support (SSE)                            │   │
│  └──────────────────┬───────────────────────────────────┘   │
└────────────────────┼────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Inference Engine                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            bitnet-inference                           │   │
│  │  • Autoregressive generation                          │   │
│  │  • Multi-head attention                               │   │
│  │  • KV-cache optimization                              │   │
│  └──────────────────┬───────────────────────────────────┘   │
└────────────────────┼────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Core Components                             │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │bitnet-models│  │bitnet-quant │  │bitnet-tokenizers │    │
│  └─────────────┘  └─────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Integration Points:**
1. **Model Loading**: Uses `bitnet-models` for GGUF parsing and tensor loading
2. **Tokenization**: Integrates `bitnet-tokenizers` for universal tokenizer support
3. **Inference**: Wraps `bitnet-inference` engine with HTTP/REST interface
4. **Monitoring**: Exposes internal metrics to Prometheus and OpenTelemetry

**Deployment Support:**
- **Docker**: Multi-stage builds for CPU and GPU variants (see `infra/docker/`)
- **Kubernetes**: Helm charts with autoscaling and health probes (see `infra/helm/bitnet/`)
- **Configuration**: Environment variables and TOML configuration files
- **Security**: Non-root execution, read-only filesystems, minimal dependencies

For detailed deployment guides, see:
- [Docker Deployment Guide](how-to/production-server-docker-deployment.md)
- [Kubernetes Deployment Guide](how-to/production-server-kubernetes-deployment.md)
- [Health Endpoints Documentation](health-endpoints.md)

## Key Design Patterns

1. **Feature-Gated Architecture**: Default features are **empty** - always specify features explicitly
2. **Production GGUF Loading**: Comprehensive tensor parsing replacing mock initialization with real model weights
3. **Zero-Copy Operations**: Memory-mapped models, careful lifetime management with enhanced tensor loading
4. **Device-Aware Quantization**: Automatic GPU acceleration with CPU fallback for all quantization formats
5. **SIMD Abstraction**: Unified interface over platform-specific instructions with enhanced performance
6. **Cross-Validation**: Systematic comparison with C++ for correctness using real model weights
7. **Enhanced Validation Framework**: Comprehensive GPU/CPU validation with performance metrics and error tolerance
8. **Security-First Design**: Input validation, bounds checking, and resource limits for production deployment
9. **FFI Bridge Architecture**: Safe C++ kernel integration for gradual migration with comprehensive testing and error handling
10. **Multi-Backend GPU Detection**: System-aware GPU detection with automatic fallback, supporting CUDA, Metal, ROCm, and WebGPU with mock testing capabilities
11. **GPU Infrastructure Access**: Low-level CUDA context and module access for advanced GPU programming (PR #199), enabling custom kernel loading and device-specific optimization
12. **Mixed Precision Computing**: Native CUDA kernels for FP16/BF16 operations with device-aware precision selection and automatic fallback (PR #202)
13. **Production-Ready Server Architecture**: Scalable HTTP/REST inference server with comprehensive health monitoring, system metrics, and deployment automation (PR #422)

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
- **Fallback Strategy**: Graceful degradation with compatibility validation for unsupported formats
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
- [Quantization Support](reference/quantization-support.md)
- [GPU Development Guide](development/gpu-development.md)
- [Tokenizer Architecture](tokenizer-architecture.md)
- [FFI Bridge Documentation](ffi-threading-architecture.md)
- [Test Suite Guide](development/test-suite.md)