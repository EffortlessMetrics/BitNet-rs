# Architecture Overview

This document provides a high-level overview of BitNet-rs architecture, design patterns, and key components.

For validation authority across scalar, AVX2, NEON, Apple Silicon, x86
accelerators, CUDA, Intel GPU/NPU lanes, dense SLMs, and BitNet model families,
see [Reference Topology](architecture/reference-topology.md).

## Workspace Structure

BitNet-rs is organized as a Rust workspace with **110 crates** (as of this writing):

```
bitnet-tokenizers → bitnet-models (GGUF loader) → bitnet-quantization → bitnet-kernels → bitnet-inference → bitnet-server / bitnet-cli
                                                                              ↑
                          bitnet-logits → bitnet-sampling → bitnet-generation ─┘
                          bitnet-gguf, bitnet-device-probe, bitnet-engine-core
                          bitnet-prompt-templates, bitnet-receipts, bitnet-honest-compute
                          bitnet-runtime-feature-flags, bitnet-simd, bitnet-rope
```

### Core Library
- **`bitnet`** (root): Main library with unified public API and GGUF weight loading
- **`bitnet-common`**: Shared types, traits, utilities, and enhanced error types for GGUF operations
- **`bitnet-models`**: **Enhanced model loading with real GGUF weight parsing** - replaces mock tensor initialization with comprehensive transformer layer weight loading (AC1), supporting all quantization formats with device-aware placement
- **`bitnet-quantization`**: Real quantized computation with I2S, TL1/TL2 accuracy validation vs FP32 baselines (target thresholds defined in test fixtures) - **STRICT MODE ENFORCED** to prevent mock fallbacks
- **`bitnet-kernels`**: **Device-aware quantization kernels** with SIMD/CUDA acceleration, mixed precision support (FP16/BF16), automatic CPU/GPU selection, FFI bridge for C++ cross-validation, plus comprehensive GPU detection utilities supporting CUDA, Metal (#992), Vulkan (#993), ROCm (#995), Intel oneAPI (#986), and OpenGL/OpenCL probing
- **`bitnet-inference`**: **Real neural network inference engine** ([Issue #254](explanation/issue-254-real-inference-spec.md)) with autoregressive generation, multi-head attention, quantized linear layers (I2S/TL1/TL2 GEMV), RoPE positional embeddings, GQA support, KV-cache optimization, deterministic generation, and receipt-backed performance validation - **compute_path="real"** enforced
- **`bitnet-tokenizers`**: Universal tokenizer with GGUF integration, automatic discovery, and graceful fallback system

**See also**: [Issue #254 Real Inference Specification](explanation/issue-254-real-inference-spec.md) for comprehensive real inference architecture

### SRP Microcrates (inference pipeline)

These small, single-responsibility crates reduce coupling in `bitnet-inference` and are re-exported from their original locations for zero breaking changes.

- **`bitnet-logits`**: Pure logit-transform functions (`apply_temperature`, `apply_top_k`, `apply_top_p`, `softmax_in_place`, `apply_repetition_penalty`, `argmax`) — no external dependencies, suitable for `no_std`
- **`bitnet-logits-filters`**: Extended logit filtering (min-p, tail-free sampling, typical sampling)
- **`bitnet-sampling`**: Token-sampling strategies (greedy, temperature, top-k, top-p, repetition penalty) built on `bitnet-logits`; seeded with `ChaCha8Rng` for reproducibility
- **`bitnet-generation`**: Decode-loop contracts: `StopCriteria`, `check_stop` priority logic, `StopReason`, `GenerationConfig`, streaming `StreamEvent` / `GenerationStats`
- **`bitnet-engine-core`**: Session/orchestration contracts: `SessionConfig`, `SessionMetrics`, `BackendInfo`
- **`bitnet-device-probe`**: OS/GPU probing and capability snapshot (`gpu_compiled()`, `gpu_available_runtime()`, `detect_simd_level()`, `DeviceCapabilities`) — respects `BITNET_GPU_FAKE` and `BITNET_STRICT_MODE`
- **`bitnet-gguf`**: GGUF parser surface with property and fuzz tests
- **`bitnet-prompt-templates`** / **`bitnet-prompt-templates-core`**: Chat template types (`PromptTemplate`, `TemplateType`, `ChatTurn`) and formatters
- **`bitnet-receipts`** / **`bitnet-receipts-core`**: Honest-compute receipt schema v1.0.0 and serialization
- **`bitnet-honest-compute`**: `compute_path=real` enforcement and kernel-ID hygiene policy (no-mock gate)
- **`bitnet-runtime-feature-flags`** / **`bitnet-runtime-feature-flags-core`**: Runtime snapshot of compiled features (`cpu`, `gpu`, `cuda` flags reported independently)
- **`bitnet-simd`**: Portable SIMD abstraction layer
- **`bitnet-rope`**: RoPE positional-embedding table generation
- **`bitnet-math`**: Numerically-stable math primitives (log-sum-exp, softmax helpers)
- **`bitnet-validation`**: Production tensor shape/dtype validation
- **`bitnet-warn-once`**: `warn_once!` macro for rate-limited hot-path logging
- **`bitnet-cpu-detect`**: Runtime CPU feature detection (AVX2, AVX-512, NEON)
- **`bitnet-cpu-activations`**: CPU activation functions (GELU, SiLU, etc.)
- **`bitnet-qk256-dispatch`**: QK256 format dispatch with scalar/AVX2 runtime selection
- **`bitnet-transformer`**: Transformer block contracts and layer composition
- **`bitnet-runtime-bootstrap`**: Runtime initialization and startup sequencing
- **`bitnet-runtime-context`** / **`bitnet-runtime-context-core`**: Runtime execution context
- **`bitnet-runtime-profile`** / **`bitnet-runtime-profile-core`**: Runtime profiling framework

### GPU Backend Crates

- **`bitnet-gpu-hal`**: Unified GPU hardware abstraction layer with backend selector, async runtime, checkpoint manager, and deployment manager
- **`bitnet-opencl`**: Intel Arc OpenCL backend with built-in kernel registry and work-size optimization (experimental; feature `opencl`)
- **`bitnet-metal`**: Metal GPU backend for macOS/iOS Apple Silicon (feature `metal`)
- **`bitnet-vulkan`** / **`bitnet-vulkan-shaders`**: Vulkan compute backend (feature `vulkan`)
- **`bitnet-spirv`**: SPIR-V shader compilation support
- **`bitnet-nvidia`**: NVIDIA-specific GPU utilities
- **`bitnet-rocm`**: AMD ROCm detection and device probe (feature `rocm`)
- **`bitnet-intel-gpu-id`**: Intel GPU identification utilities
- **`bitnet-webgpu`** / **`bitnet-wgpu`** / **`bitnet-wgpu-runner`** / **`bitnet-wgpu-bench`** / **`bitnet-wgpu-shaders`** / **`bitnet-wgpu-shaders-i2s`**: WebGPU/wgpu compute backends and I2_S shader kernels

### Testing / CI Microcrates

- **`bitnet-bdd-grid`** / **`bitnet-bdd-grid-core`**: BDD compatibility grid with compile-coverage enforcement (`xtask grid-check`)
- **`bitnet-feature-matrix`** / **`bitnet-feature-contract`**: Feature-lattice contracts and enforcement tests
- **`bitnet-testing-policy`** / **`bitnet-testing-policy-core`** / **`bitnet-testing-policy-runtime`** / **`bitnet-testing-policy-kit`**: Test policy framework and runtime enforcement
- **`bitnet-testing-scenarios`** / **`bitnet-testing-scenarios-core`**: Test scenario definitions
- **`bitnet-testing-profile`** / **`bitnet-runtime-profile-contract`**: Testing profile primitives
- **`bitnet-test-fixtures-core`**: Shared test fixtures (GGUF fixtures, mock tensors)
- **`bitnet-test-env`** / **`bitnet-test-support`**: Environment isolation and test helpers
- **`bitnet-bench-receipts`** / **`bitnet-bench-regression-core`**: Benchmark receipt validation and regression detection

### Application Layer
- **`bitnet-server`**: **Production HTTP/REST inference server** providing scalable inference endpoints with batch processing, model hot-swapping capabilities, comprehensive health monitoring (liveness/readiness/startup probes), real-time system metrics collection (CPU, memory, disk, network I/O), Prometheus metrics integration, OpenTelemetry observability, streaming inference support, and deployment-ready configurations for Docker and Kubernetes environments
- **`bitnet-cli`**: Command-line interface for local inference, model verification, and compatibility checking

### Compatibility Layer
- **`bitnet-compat`**: GGUF compatibility fixes and diagnostics
- **`bitnet-ffi`**: C API for llama.cpp-compatible API (validation pending)
- **`bitnet-py`**: Python 3.12+ bindings compatible with llama-cpp-python (PyO3 ABI3-py312)
- **`bitnet-wasm`**: WebAssembly bindings with enhanced browser/Node.js compatibility and optimized SIMD intrinsics

### Cross-Validation
- **`crossval`**: Framework for testing against C++ implementation
- Tests use `BITNET_GGUF` or `CROSSVAL_GGUF` environment variable for model path

## Kernel Module Architecture

The `bitnet-kernels` crate contains **58 source files** organized into backend-specific subdirectories. See [Kernel Module Reference](reference/kernel-modules.md) for the complete module table.

### CPU Kernels (`cpu/` — 66 modules)

Generic compute kernels with platform-specific SIMD accelerations:

- **Core ops**: `matmul`, `softmax`, `layernorm`, `attention`, `ffn`, `embedding`, `rope`, `gating`, `pooling`, `convolution`, `transpose`, `reduction`, `residual`, `loss`
- **x86 SIMD** (`x86.rs`): AVX2/AVX-512 paths for QK256 dequantization and GEMV; property tests in `x86_qk256_property_tests.rs`
- **ARM NEON** (24 modules): `neon_activations`, `neon_quantized_gemm`, `neon_quantized_matmul`, `neon_rope`, `neon_softmax`, `neon_layernorm`, `neon_kv_cache`, `neon_scatter_gather`, `neon_sliding_window_attention`, `neon_batch_norm`, `neon_elementwise`, `neon_reductions`, `neon_transpose`, `neon_inference_bridge`, among others
- **Scatter/gather** (`scatter_gather.rs`, `cpu/scatter_gather.rs`): Memory-layout-aware data movement
- **Fusion**: `layer_fusion.rs`, `fusion.rs` for fused attention and FFN passes
- **Parallelism**: `pipeline_parallel.rs`, `tensor_parallel.rs` for multi-device distribution

### CUDA Kernels (`cuda/` — 39 modules)

GPU compute kernels targeting NVIDIA CUDA, gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`:

- **Core ops**: `matmul`, `softmax`, `layernorm`, `rmsnorm`, `attention`, `fused_attention`, `multi_head_attention`, `ffn`, `embedding`, `rope`, `gating`, `linear`, `dequant`, `quantize`
- **Quantized**: `quantized_gemm`, `quantized_matmul`, `qk256_gemv` for 2-bit GEMV
- **Memory management**: `memory_pool` (device memory pooling), `kv_cache`, `kv_cache_gpu`
- **Execution management**: `stream_mgmt` (CUDA stream manager), `warp_ops` (warp-level primitives), `cooperative_groups`
- **Optimization**: `graph_exec` (CUDA graph execution), `shader_cache`, `profiling`, `sparse`
- **Other ops**: `batch_norm`, `conv1d`, `elementwise`, `pooling`, `residual`, `loss`, `transpose`, `fusion`

### OpenCL Modules (42 modules)

Experimental Intel Arc / OpenCL backend, each as a top-level module in `bitnet-kernels/src/`:

- **Compute**: `opencl_attention`, `opencl_flash_attention`, `opencl_gqa`, `opencl_ffn`, `opencl_layer_norm`, `opencl_reductions`, `opencl_softmax_variants`, `opencl_elementwise`, `opencl_embedding`, `opencl_token_embed`
- **Quantized**: `opencl_quantized`, `opencl_quantized_matmul`, `opencl_matmul_variants`, `opencl_mixed_precision`
- **Infrastructure**: `opencl_context`, `opencl_cmd_queue`, `opencl_buffer`, `opencl_memory`, `opencl_device_caps`, `opencl_work_size`, `opencl_kernel_sources`, `opencl_program_cache`, `opencl_registry`
- **Pipeline**: `opencl_pipeline`, `opencl_continuous_batch`, `opencl_graph_compiler`, `opencl_layer_compose`, `opencl_transformer`, `opencl_engine_bridge`, `opencl_model_converter`
- **Caching**: `opencl_cache`, `opencl_prefix_cache`, `opencl_kv_cache`, `opencl_rope_cache`
- **Utilities**: `opencl_autotuner`, `opencl_profiling`, `opencl_telemetry`, `opencl_async_executor`, `opencl_numerical_stability`, `opencl_weight_manager`, `opencl_token_gen`

### Other Backend Modules

| Module | Feature gate | Purpose |
|--------|-------------|---------|
| `metal_compute` | `feature = "metal"` | Apple Metal compute kernels |
| `rocm/` (4 modules) | `feature = "rocm"` | AMD ROCm attention, QK256 GEMV, RMSNorm |
| `npu/` | `feature = "npu-backend"` | NPU bridge (C++ interop) |
| `gpu/` (mixed-backend) | `feature = "gpu"` | Shared GPU utilities, OpenCL dispatch, validation, benchmarks |

### Shared Kernel Infrastructure

| Module | Purpose |
|--------|---------|
| `kernels.rs` | `KernelManager` — runtime provider selection (CUDA > CPU fallback) |
| `capability_matrix.rs` | Kernel capability reporting per backend |
| `device_aware.rs` | Device-aware kernel dispatch |
| `device_features.rs` | `gpu_compiled()`, `gpu_available_runtime()` helpers |
| `convolution.rs` | Generic convolution ops |
| `reduction.rs` / `shaped_reduction.rs` | Reduction primitives |
| `scatter_gather.rs` | Top-level scatter/gather |
| `tl_lut.rs` | Table-lookup (TL1/TL2) LUT generation |
| `simd_diagnostics.rs` | SIMD feature detection and diagnostics |
| `perf_tracker.rs` | Kernel performance tracking |
| `gpu_utils.rs` | GPU utility functions |
| `stubs.rs` | Stub implementations for disabled backends |
| `ffi.rs` / `ffi/` | C++ FFI bridge (feature `ffi`) |

## Test Infrastructure

BitNet-rs has comprehensive test infrastructure spanning multiple strategies:

| Strategy | Scope | Details |
|----------|-------|---------|
| **Property-based (proptest)** | 63 crates | Randomized invariant testing across quantization, tokenization, KV-cache, tensor shapes |
| **Snapshot (insta)** | 49 crates, ~1 233 `.snap` files | Struct/output stability for serialization, CLI output, receipt schemas |
| **Fuzz (cargo-fuzz)** | 100+ registered targets | Nightly `fuzz-ci.yml` selected matrix: RoPE table gen, tokenizer encode, GGUF parsing, runtime validation, cache structures, and more |
| **BDD grid** | `bitnet-bdd-grid` | Compile-coverage matrix (`xtask grid-check`) |
| **Feature-lattice** | `bitnet-feature-matrix` | Orthogonal feature-gate contracts |
| **Fixture-based** | `bitnet-models`, `bitnet-test-fixtures-core` | GGUF dual-flavor detection, alignment (12/12 passing) |
| **Environment isolation** | `EnvGuard` + `#[serial(bitnet_env)]` | Parallel-safe env mutation via `temp_env` |
| **CPU golden-path E2E** | `tests/` | 7 deterministic tests always in PR CI (no model download) |
| **Criterion benchmarks** | `benches/srp_ops.rs` | logits pipeline, top-k, repetition penalty, argmax, RoPE, KV cache |

## GGUF Weight Loading Architecture

BitNet-rs implements a comprehensive GGUF weight loading system that replaces mock tensor initialization with real neural network model parsing. This system represents a major architectural advancement enabling meaningful neural network inference.

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

**I2_S (2-bit Signed) - Two Flavors with Automatic Detection:**
- **BitNet Native (32-element blocks):** Values [-2, -1, 1, 2] with 10 bytes/block format
- **GGML QK256 (256-element blocks):** Values [-2, -1, 1, 2] with 64 bytes/block format, separate scales
- **Automatic Detection:** Loader inspects tensor sizes to identify format
- **Transparent Dispatch:** Forwards automatically use appropriate kernel
- Performance: 66+ Melem/s (CPU), 200+ Melem/s (GPU)
- Accuracy: Target accuracy thresholds defined in test fixtures
- **Pure-Rust Support:** Both formats run without FFI dependency

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
- I2_S quantization: Target accuracy thresholds defined in test fixtures
- Cross-validation against C++ reference implementation
- Systematic regression testing for accuracy preservation
- Property-based testing for numerical stability

## Production Server Architecture

### `bitnet-server` Crate Overview

The `bitnet-server` crate provides an HTTP/REST inference server built on the BitNet-rs inference engine. It serves as the application layer for deploying BitNet models in production environments.

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
10. **Multi-Backend GPU Detection**: System-aware GPU detection with automatic fallback, supporting CUDA, Metal (#992), Vulkan (#993), ROCm (#995), Intel oneAPI (#986), and OpenGL/OpenCL probing (#984/#985)
11. **GPU Infrastructure Access**: Low-level CUDA context and module access for advanced GPU programming (PR #199), enabling custom kernel loading and device-specific optimization
12. **Mixed Precision Computing**: Native CUDA kernels for FP16/BF16 operations with device-aware precision selection and automatic fallback (PR #202)
13. **Server Architecture**: Scalable HTTP/REST inference server with comprehensive health monitoring, system metrics, and deployment automation (PR #422)

## Enhanced Quality Assurance Framework

BitNet-rs includes a comprehensive quality assurance system designed for production reliability:

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
- Python API is llama-cpp-python compatible
- We handle models that llama.cpp fails on (e.g., GPT-2 without pre-tokenizer)
- Enhanced GGUF parsing with tensor alignment validation for better error detection
- Robust handling of malformed GGUF files with detailed error messages
- See COMPATIBILITY.md for detailed guarantees

## Current Limitations

- **Pre-Alpha Status**: Correctness, performance, and validation are ongoing. Do not use in production.
- **QK256 Performance**: Scalar kernels only (~0.1 tok/s for 2B models). AVX2 nibble-LUT + FMA tiling planned for ≥3× uplift. Limit to `--max-tokens 4-16` for validation.
- **GPU Backends**: Metal, Vulkan, oneAPI, ROCm, OpenCL, and WebGPU backends are scaffolded but not validated end-to-end. CUDA is the furthest along.
- **OpenCL Modules**: 42 OpenCL modules in `bitnet-kernels` are experimental (Intel Arc focus); API surface may change.
- **Model Quality**: microsoft-bitnet-b1.58-2B-4T-gguf produces non-sensical output in some configurations (known model quality issue, not an inference bug).
- **Test Scaffolding**: ~466 `#[ignore]` tests across the workspace, all with justification strings. Categories: real-model, CUDA, slow mock-inference, crossval, and TDD scaffolds.

## Development Workflow

1. **Making Changes**: Always run tests for affected crates
2. **Before Committing**: Run `cargo fmt` and `cargo clippy`
3. **Cross-Validation**: Run `cargo xtask crossval` for inference changes
4. **Compatibility**: Check COMPATIBILITY.md before changing public APIs

For detailed information on specific components, see:
- [Kernel Module Reference](reference/kernel-modules.md)
- [Quantization Support](reference/quantization-support.md)
- [GPU Development Guide](development/gpu-development.md)
- [Tokenizer Architecture](tokenizer-architecture.md)
- [FFI Bridge Documentation](ffi-threading-architecture.md)
- [Test Suite Guide](development/test-suite.md)
