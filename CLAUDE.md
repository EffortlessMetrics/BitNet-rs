# CLAUDE.md

This file provides guidance to Claude (claude.ai) when working with the BitNet.rs codebase.

## Build and Development Commands

### Core Development Commands
```bash
# Build with CPU support (no default features)
cargo build --release --no-default-features --features cpu

# Build with GPU support and device-aware quantization
cargo build --release --no-default-features --features gpu

# Build with IQ2_S quantization support (requires GGML FFI)
cargo build --release --no-default-features --features "cpu,iq2s-ffi"

# Build with FFI quantization bridge support (requires C++ library)
cargo build --release --no-default-features --features "cpu,ffi"

# Run tests (fast, Rust-only)
cargo test --workspace --no-default-features --features cpu

# Run GPU tests with device-aware quantization (requires CUDA)
cargo test --workspace --no-default-features --features gpu

# Run GGUF validation tests
cargo test -p bitnet-inference --test gguf_header
cargo test -p bitnet-inference --test gguf_fuzz
cargo test -p bitnet-inference --test engine_inspect

# Run convolution kernel tests
cargo test -p bitnet-kernels --no-default-features --features cpu convolution

# Run PyTorch reference convolution tests (requires Python and PyTorch)
cargo test -p bitnet-kernels conv2d_reference_cases -- --ignored

# Run verification script
./scripts/verify-tests.sh

# Run benchmarks
cargo bench --workspace --no-default-features --features cpu

# Cross-validation testing (requires C++ dependencies)
cargo test --workspace --features "cpu,ffi,crossval"

# FFI quantization bridge tests (compares FFI vs Rust implementations)
cargo test -p bitnet-kernels --features ffi test_ffi_quantize_matches_rust

# FFI mock model tests (validates C-API testing infrastructure)
cargo test -p bitnet-ffi test_mock_model_embed_and_logits
```

### Code Quality Commands
```bash
# Format all code
cargo fmt --all

# Run clippy with pedantic lints
cargo clippy --all-targets --all-features -- -D warnings

# Security audit
cargo audit

# Run tests with concurrency caps (prevents resource storms)
scripts/preflight.sh && cargo t2                     # 2-thread CPU tests
scripts/preflight.sh && cargo crossval-capped        # Cross-validation with caps
scripts/e2e-gate.sh cargo test --features crossval   # Gate heavy E2E tests

# Generate code coverage
cargo llvm-cov --workspace --features cpu --html
```

### Development Tools (xtask)
```bash
# Download BitNet model from Hugging Face
cargo run -p xtask -- download-model

# Vendor GGML quantization files for IQ2_S support
cargo run -p xtask -- vendor-ggml --commit <llama.cpp-commit>

# Fetch and build Microsoft BitNet C++ for cross-validation
cargo run -p xtask -- fetch-cpp

# Run cross-validation tests
cargo run -p xtask -- crossval
# Full workflow (download + fetch + test)
cargo run -p xtask -- full-crossval

# Check feature flag consistency
cargo run -p xtask -- check-features
```

## High-Level Architecture

### Workspace Structure
BitNet.rs is organized as a Rust workspace with specialized crates:

#### Core Library
- **`bitnet`** (root): Main library with unified public API
- **`bitnet-common`**: Shared types, traits, and utilities
- **`bitnet-models`**: Model loading and format handling (GGUF, SafeTensors)
- **`bitnet-quantization`**: 1-bit quantization algorithms
- **`bitnet-kernels`**: High-performance SIMD/CUDA kernels with 2D convolution support, comprehensive memory statistics tracking with real-time host/GPU memory monitoring, platform-specific kernel selection (x86_64 AVX2/aarch64 NEON), FFI bridge for gradual C++ migration, plus comprehensive GPU detection utilities supporting CUDA, Metal, ROCm, and WebGPU backends
- **`bitnet-inference`**: Inference engine with streaming support
- **`bitnet-tokenizers`**: Universal tokenizer with GGUF integration and mock fallback system
- **`bitnet-server`**: HTTP server for BitNet inference with health monitoring

#### Compatibility Layer
- **`bitnet-compat`**: GGUF compatibility fixes and diagnostics
- **`bitnet-ffi`**: C API for llama.cpp drop-in replacement
- **`bitnet-py`**: Python bindings compatible with llama-cpp-python

#### Cross-Validation
- **`crossval`**: Framework for testing against C++ implementation
- Tests use `BITNET_GGUF` or `CROSSVAL_GGUF` environment variable for model path

### Key Design Patterns

1. **Feature-Gated Architecture**: Default features are **empty** - always specify features explicitly
2. **Zero-Copy Operations**: Memory-mapped models, careful lifetime management
3. **SIMD Abstraction**: Unified interface over platform-specific instructions
4. **Cross-Validation**: Systematic comparison with C++ for correctness
5. **Enhanced Validation Framework**: Comprehensive GPU/CPU validation with performance metrics and error tolerance
6. **FFI Bridge Architecture**: Safe C++ kernel integration for gradual migration with comprehensive testing and error handling
7. **Multi-Backend GPU Detection**: System-aware GPU detection with automatic fallback, supporting CUDA, Metal, ROCm, and WebGPU with mock testing capabilities

### Enhanced Quality Assurance Framework

BitNet.rs includes a comprehensive quality assurance system designed for production reliability:

#### Kernel Validation System
- **GPU/CPU Parity Testing**: Systematic validation between GPU and CPU implementations
- **Performance Benchmarking**: Built-in performance measurement with speedup calculations
- **Numerical Accuracy Testing**: Configurable tolerance testing for quantization operations
- **Memory Leak Detection**: Automatic GPU memory monitoring and leak prevention
- **Host Memory Tracking**: Real-time host memory usage monitoring with memory-stats and sysinfo integration
- **Device Memory Statistics**: Comprehensive memory tracking for both GPU and CPU with efficiency metrics
- **Thread-Safe Memory Monitoring**: Arc<Mutex<DeviceStatsInternal>> for safe concurrent memory access
- **Platform-Specific Selection**: Automatic AVX2/NEON kernel selection based on CPU architecture
- **Error Handling Validation**: Comprehensive error path testing with recovery verification

#### Model Compatibility Validation System
- **Weight Mapper Integration**: GGUF tensor validation using weight mapper for compatibility checks
- **Unmapped Tensor Detection**: Detailed reporting of unmapped tensors with debugging metrics
- **Fixture-Based Testing**: Comprehensive test coverage for both success and corruption scenarios
- **Enhanced Error Reporting**: ValidationResult metrics include `unmapped_count` and `unmapped_tensors`
- **GGUF Parsing Integration**: Direct model file analysis for compatibility validation

#### Universal Tokenizer Architecture
- **Auto-Detection**: Automatic backend selection based on GGUF model metadata
- **GGUF Integration**: Direct extraction of tokenizer configuration from model files
- **Fallback Strategy**: Graceful degradation to mock tokenizer for unsupported formats
- **Runtime Construction**: Build tokenizers from vocabulary and merge rules without external dependencies
- **Cross-Format Support**: BPE, SentencePiece, and custom tokenizer formats

#### FFI Bridge System (Enhanced in PR #186)
- **Gradual Migration Support**: Safe C++ kernel integration enabling gradual transition to pure Rust
- **Quantization Bridge**: Complete FFI quantization support for I2S, TL1, and TL2 types
- **Performance Comparison Framework**: Built-in tools for comparing FFI vs Rust implementations
- **Error Handling Integration**: Enhanced C++ error propagation with `get_last_error()` bridge
- **Feature-Gated Safety**: Proper conditional compilation and graceful fallback when FFI unavailable
- **Migration Decision Support**: Automated recommendations based on performance and accuracy metrics
- **Mock Testing Infrastructure**: Comprehensive C-API testing with mock embed/logits implementations

#### Code Quality Enforcement
- **Comprehensive Clippy Integration**: Zero-tolerance policy for clippy warnings
- **Type Safety Improvements**: Enhanced type annotations and error handling
- **Documentation Standards**: Comprehensive inline documentation with examples
- **Test Coverage**: Extensive test suites with property-based testing
- **Performance Regression Testing**: Automated performance monitoring and validation

## Important Considerations

### MSRV
Minimum Supported Rust Version: **1.89.0** (uses Rust 2024 edition)

### Feature Flags
Default features are **empty** to prevent unwanted dependencies:
- `cpu`: CPU inference with SIMD optimizations, includes native I2_S support
- `gpu`: NVIDIA GPU support with advanced device-aware quantization and automatic fallback
- `cuda`: Backward-compatible alias for `gpu` feature
- `iq2s-ffi`: IQ2_S quantization via GGML FFI (requires vendored GGML files)
- `ffi`: C++ FFI bridge with quantization support for gradual migration (includes FfiKernel with I2S/TL1/TL2 quantization)
- `crossval`: Cross-validation against C++ (increases build time)

### Quantization Support
BitNet-rs supports multiple quantization formats with advanced device-aware acceleration:
- **I2_S**: Native Rust implementation with intelligent GPU/CPU selection and automatic fallback
  - Device-aware dequantization with CUDA kernel acceleration
  - Automatic CPU fallback for unsupported hardware or initialization failures
  - 2-bit signed quantization with optimized bit-packing (4 values per byte)
- **TL1**: Table lookup quantization with GPU acceleration and CPU fallback
  - Device-aware table lookup with GPU memory optimization
  - Parallel processing with configurable block sizes
- **TL2**: Advanced table lookup quantization with optimized GPU kernels and device-aware execution
  - Enhanced vectorized operations for large tensor processing
  - CPU feature detection with SIMD optimization fallbacks
- **IQ2_S**: GGML-compatible quantization with 82-byte block layout and 4-level [-2,-1,1,2] mapping
- **Standard formats**: Q4_0, Q5_0, Q8_0, etc. (planned)

All quantizers support device-aware operations with:
- **Automatic GPU acceleration**: CUDA kernels with performance monitoring
- **Transparent CPU fallback**: Graceful degradation with maintained accuracy
- **Memory optimization**: GPU memory leak detection and efficient allocation
- **Feature gating**: Proper `#[cfg(feature = "gpu")]` guards for CPU-only builds
- **FFI Bridge Support**: C++ kernel integration for I2S, TL1, and TL2 quantization (requires `--features ffi`)

### FFI Quantization Bridge (New)

The FFI bridge enables gradual migration from C++ to Rust while maintaining functionality:

- **Quantization Types**: Full support for I2S, TL1, and TL2 via C++ kernels
- **Performance Comparison**: Built-in tools to compare FFI vs Rust quantization
- **Migration Path**: Systematic approach to replace C++ kernels with native Rust
- **Safety**: Safe Rust wrappers with proper error handling and memory management
- **Testing**: Comprehensive test suite ensuring FFI/Rust quantization parity

### Universal Tokenizer Architecture

BitNet.rs includes a comprehensive tokenizer system with GGUF integration:

- **UniversalTokenizer**: Auto-detecting tokenizer that handles multiple formats
  - **GGUF Integration**: Extracts tokenizer configuration directly from GGUF model files
  - **Automatic Backend Selection**: Chooses appropriate tokenizer backend based on model type
  - **Mock Tokenizer Fallback**: Provides testing-compatible tokenizer for unsupported formats
  - **Configuration-Driven**: Supports pre-tokenization, special tokens, and BPE merges

#### Supported Tokenizer Formats
- **GPT-2/BPE**: Modern BPE tokenization with merge rules (via HuggingFace tokenizers)
- **SentencePiece**: Subword tokenization via SentencePiece library (feature-gated)
- **LLaMA/LLaMA3**: LLaMA-specific tokenization variants
- **TikToken**: OpenAI's tiktoken format
- **Mock Backend**: Minimal tokenizer for testing and compatibility

#### BPE Backend Features (New in this release)
- **Runtime Construction**: Build tokenizers from vocabulary and merge rules without JSON files
- **GGUF Metadata Integration**: Automatically extract BPE data from model files
- **Byte-Level Processing**: GPT-2 compatible pre-tokenization and decoding
- **Fallback Support**: Graceful degradation to mock tokenizer when data incomplete

#### GGUF Tokenizer Metadata Extraction
The universal tokenizer automatically parses GGUF metadata:
- Vocabulary extraction from `tokenizer.ggml.tokens`
- Special token IDs (BOS, EOS, PAD, UNK) from metadata
- BPE merge rules from `tokenizer.ggml.merges`
- Configuration flags (add_bos, add_eos, add_space_prefix, byte_fallback)
- Score arrays for token prioritization

### Compatibility Guarantees
We maintain strict compatibility with llama.cpp:
- C API functions have exact signature matches
- Python API is drop-in compatible
- We handle models that llama.cpp fails on (e.g., GPT-2 without pre-tokenizer)
- See COMPATIBILITY.md for detailed guarantees

## Troubleshooting

### Common Build Issues

1. **FFI Linker Errors**: Either disable FFI (`--no-default-features --features cpu`) or build C++ (`cargo xtask fetch-cpp`)

2. **Compiler Compatibility**: The FFI components support both GCC and Clang. Set `CC` and `CXX` environment variables to specify compiler:
   - GCC: `export CC=gcc CXX=g++`
   - Clang: `export CC=clang CXX=clang++`
   - CI tests both compilers automatically via matrix builds

3. **CUDA Compilation**: Ensure CUDA toolkit is installed and `nvcc` is in PATH

4. **CUDA Device Query Failures**: See [GPU Development Guide](docs/gpu-development.md#advanced-gpucuda-troubleshooting) for comprehensive troubleshooting

5. **Cross-Validation Path**: Set `BITNET_GGUF` environment variable to model path

6. **Git Metadata in Builds**: The `bitnet-server` crate uses `vergen-gix` v1.x to capture Git metadata. Ensure `.git` is available during builds or set `VERGEN_GIT_SHA` and `VERGEN_GIT_BRANCH` environment variables

7. **FFI Quantization Issues**: 
   - Ensure C++ library is built: `cargo xtask fetch-cpp`
   - Test FFI availability: `cargo test -p bitnet-kernels --features ffi test_ffi_kernel_creation`
   - Compare FFI vs Rust: `cargo test -p bitnet-kernels --features ffi test_ffi_quantize_matches_rust`
   - Check C++ errors: Look for detailed error messages from `get_last_error()` bridge

## Development Workflow

1. **Making Changes**: Always run tests for affected crates
2. **Before Committing**: Run `cargo fmt` and `cargo clippy`
3. **Cross-Validation**: Run `cargo xtask crossval` for inference changes
4. **Compatibility**: Check COMPATIBILITY.md before changing public APIs

## Key Files

- `COMPATIBILITY.md`: API stability guarantees and truth tables
- `MIGRATION.md`: Step-by-step migration guide from llama.cpp
- `.github/workflows/compatibility.yml`: CI compatibility tests
- `crossval/`: Cross-validation test suite

## Specialized Documentation

For detailed information on specific topics, see:

- **[GPU Development Guide](docs/gpu-development.md)**: CUDA device querying, GPU testing strategies, and troubleshooting
- **[Test Suite Guide](docs/test-suite.md)**: Comprehensive testing framework, configuration, and specialized testing strategies  
- **[GGUF Inspection Guide](docs/gguf-inspection.md)**: Metadata inspection, categorization, and JSON serialization
- **[Streaming API Guide](docs/streaming-api.md)**: Real-time token streaming with Server-Sent Events
- **[Concurrency Caps Guide](docs/concurrency-caps.md)**: Resource management and concurrency control
- **[Performance Tracking Guide](docs/performance-tracking.md)**: Comprehensive performance monitoring, metrics collection, and optimization analysis
## Key Dependencies

### Memory Tracking Dependencies

BitNet.rs uses specialized crates for cross-platform memory monitoring:

- **`memory-stats` (1.1)**: Process-specific memory statistics
  - Provides accurate current process physical memory usage
  - Cross-platform support (Windows, macOS, Linux)
  - Minimal overhead suitable for real-time tracking
  - Used for `DeviceStats::memory_used_bytes` in device-aware quantization

- **`sysinfo` (0.30)**: System information and monitoring
  - Comprehensive system memory information
  - Optimized memory refresh to minimize system calls  
  - Cross-platform API consistency
  - Used for `DeviceStats::memory_total_bytes` and GPU detection

### GPU/CUDA Dependencies

- **`cudarc` (0.12)**: Rust CUDA bindings for GPU acceleration
  - Safe CUDA memory management and kernel execution
  - Device querying and capability detection
  - Used for GPU memory tracking via `cuMemGetInfo_v2`

- **`half` (2.3)**: Half-precision floating point support
  - FP16 and BF16 operations for modern GPU architectures
  - Required for tensor core operations on CC 6.1+ devices

### Cross-Platform Compatibility

- **FFI Dependencies**: Optional C++ library integration
  - **`cc`**: C++ compilation for FFI bridge
  - **`bindgen`**: Automatic binding generation
  - **`pkg-config`**: Library discovery on Unix systems

## Environment Variables

### Runtime Variables
- `BITNET_GGUF` / `CROSSVAL_GGUF`: Path to test model
- `BITNET_CPP_DIR`: Path to C++ implementation
- `HF_TOKEN`: Hugging Face token for private repos
- `BITNET_DETERMINISTIC`: Enable deterministic mode for testing
- `BITNET_SEED`: Set seed for reproducible runs
- `RAYON_NUM_THREADS`: Control CPU parallelism
- `BITNET_GPU_FAKE`: Mock GPU backend detection for testing (e.g., "cuda", "metal", "cuda,rocm")

### Performance Configuration Variables
- `BITNET_BATCH_SIZE`: Configure inference batch size (e.g., "4", "8")
- `BITNET_MEMORY_LIMIT`: Set memory usage limits (e.g., "1GB", "512MB")
- `BITNET_NUM_THREADS`: Control inference thread count for parallel processing

### Build-time Variables (for Git metadata)
- `VERGEN_GIT_SHA`: Override Git SHA (useful in CI/Docker without .git)
- `VERGEN_GIT_BRANCH`: Override Git branch
- `VERGEN_GIT_DESCRIBE`: Override Git describe output
- `VERGEN_IDEMPOTENT`: Set to "1" for reproducible builds

### GPU/CUDA Development
For GPU development best practices, PR management, and hardware-dependent testing strategies, see the [GPU Development Guide](docs/gpu-development.md).

#### GPU Detection Commands
```bash
# Test GPU backend detection 
cargo test -p bitnet-kernels --no-default-features test_gpu_info_summary

# Mock GPU scenarios for testing
BITNET_GPU_FAKE="cuda" cargo test -p bitnet-kernels test_gpu_info_mocked_scenarios
BITNET_GPU_FAKE="metal" cargo run -p xtask -- download-model --dry-run
BITNET_GPU_FAKE="cuda,rocm" cargo test -p bitnet-kernels --features gpu

# GPU validation example (includes preflight-style checks)
cargo run --example gpu_validation --no-default-features --features gpu

# Test memory tracking and platform-specific kernel selection
cargo test -p bitnet-kernels --no-default-features --features cpu test_memory_tracking
cargo test -p bitnet-kernels --no-default-features --features cpu test_platform_kernel_selection

# Test device-aware quantizer with comprehensive memory statistics
cargo test -p bitnet-kernels --no-default-features --features cpu test_performance_tracking
cargo test -p bitnet-kernels --no-default-features --features cpu test_memory_tracking_comprehensive

# Test device-aware memory tracking for both CPU and GPU
cargo test -p bitnet-kernels --no-default-features --features gpu test_device_memory_tracking
```
## Repository Contracts (for Claude)

### Safe Operations
- **Default features are empty** → always pass `--no-default-features --features cpu|gpu`
- **Never mutate large binaries or GGUF in place** → use `bitnet-compat export-fixed` for new files
- **Prefer `xtask` over ad-hoc scripts** for downloads/crossval/build steps
- **Print commands before long operations** → use `--dry-run` where available
- **No destructive cleanup** without confirmation → avoid `rm -rf target/` or `~/.cache/bitnet_cpp/`

### Determinism & Environment
- `BITNET_DETERMINISTIC=1` + `BITNET_SEED=42` → force stable runs
- `RAYON_NUM_THREADS=1` → single-threaded CPU determinism
- `RUSTFLAGS="-C target-cpu=native"` → local perf builds (not CI)
- macOS FFI: set `DYLD_LIBRARY_PATH=target/release`
- Linux FFI: set `LD_LIBRARY_PATH=target/release`

## Fast Recipes

```bash
# Quick compile & test (CPU, MSRV-accurate)
rustup run 1.89.0 cargo test --workspace --no-default-features --features cpu

# Run comprehensive memory tracking demo
cargo run --example device_stats_demo --no-default-features --features cpu

# Quick compile & test with concurrency caps
scripts/preflight.sh && cargo t2

# Validate GGUF file
cargo run -p bitnet-cli -- compat-check model.gguf
cargo run -p bitnet-cli -- compat-check model.gguf --json  # JSON output

# Inspect model metadata (human-readable with categorization)
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- model.gguf

# Export model metadata as JSON
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- --json model.gguf

# Teacher-forcing evaluation with perplexity calculation
cargo run -p bitnet-cli -- score --model model.gguf --file test.txt
cargo run -p bitnet-cli -- score --model model.gguf --file validation.txt --device gpu --batch-size 8 --json-out results.json

# Model evaluation with external tokenizer and token limits
cargo run -p bitnet-cli -- score --model model.gguf --file large-dataset.txt --tokenizer tokenizer.json --max-tokens 1000

# Test GPU backend detection and mock scenarios
cargo test -p bitnet-kernels --no-default-features test_gpu_info_summary
BITNET_GPU_FAKE="cuda,rocm" cargo test -p bitnet-kernels test_gpu_info_mocked_scenarios

# Test universal tokenizer with BPE backend (new feature)
cargo test -p bitnet-tokenizers --no-default-features

# Test BPE tokenizer round-trip functionality (includes new BPE tests)
cargo test -p bitnet-tokenizers --test universal_roundtrip --no-default-features --features integration-tests

# Enhanced GPU validation with performance metrics and error handling
cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_validation_comprehensive

# Test platform-specific CPU kernel selection (x86_64 AVX2 / aarch64 NEON)
cargo test -p bitnet-kernels --no-default-features --features cpu test_cpu_provider_creation

# Test architecture-specific feature detection
cargo test -p bitnet-kernels --no-default-features --features cpu test_x86_64_feature_detection  # x86_64 only
cargo test -p bitnet-kernels --no-default-features --features cpu test_aarch64_feature_detection  # aarch64 only

# Test comprehensive memory tracking with actual system memory usage
cargo test -p bitnet-kernels --no-default-features --features cpu test_memory_tracking

# Test device-aware performance tracking with comprehensive memory statistics
cargo test -p bitnet-kernels --no-default-features --features cpu test_performance_tracking

# Test memory efficiency metrics and host memory monitoring
cargo test -p bitnet-kernels --no-default-features --features cpu test_memory_efficiency_tracking

# GPU kernel validation with numerical accuracy testing
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_vs_cpu_quantization_accuracy

# GPU memory leak detection and performance benchmarking
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_memory_management

# Device-aware quantization validation (I2S, TL1, TL2)
cargo test -p bitnet-quantization --no-default-features --features gpu test_dequantize_cpu_and_gpu_paths

# Comprehensive GPU integration tests
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_integration

# Test universal tokenizer with automatic GGUF integration
cargo test -p bitnet-tokenizers --no-default-features test_universal_tokenizer_gguf_integration

# Model compatibility validation with weight mapper
cargo test -p crossval --no-default-features test_validate_model_compatibility
cargo test -p crossval --no-default-features test_validate_model_compatibility_reports_unmapped

# FFI quantization bridge validation (compares FFI vs Rust implementations)
cargo test -p bitnet-kernels --features ffi test_ffi_quantize_matches_rust

# FFI kernel creation and availability testing
cargo test -p bitnet-kernels --features ffi test_ffi_kernel_creation

# FFI mock model testing (validates C-API infrastructure)
cargo test -p bitnet-ffi test_mock_model_embed_and_logits

# Performance tracking infrastructure tests (validates comprehensive metrics collection)
cargo test -p bitnet-inference --features integration-tests --test performance_tracking_tests

# Run specific performance tracking test categories
cargo test --test performance_tracking_tests performance_metrics_tests
cargo test --test performance_tracking_tests performance_tracker_tests  
cargo test --test performance_tracking_tests environment_variable_tests

# Test InferenceEngine performance integration
cargo test -p bitnet-inference --features integration-tests test_engine_performance_tracking_integration

# FFI performance comparison (if C++ library available)
cargo test -p bitnet-kernels --features ffi --release test_performance_comparison_structure

# Full cross-validation (deterministic)
export BITNET_GGUF="$PWD/models/bitnet/ggml-model-i2_s.gguf"
export BITNET_DETERMINISTIC=1 BITNET_SEED=42
cargo run -p xtask -- full-crossval

# Check model compatibility (read-only)
cargo run -p bitnet-cli -- compat-check "$BITNET_GGUF"

# Export fixed GGUF safely (non-destructive)
cargo run -p bitnet-cli -- compat-fix "$BITNET_GGUF" fixed.gguf
cat fixed.gguf.compat.json   # audit stamp

# FFI smoke test (build + run) - supports both GCC and Clang
export CC=gcc CXX=g++  # or CC=clang CXX=clang++
cargo build -p bitnet-ffi --release --no-default-features --features cpu
export LD_LIBRARY_PATH=target/release  # or DYLD_LIBRARY_PATH on macOS
./scripts/ffi_smoke.sh

# Quick lint check
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

## Test Suite

### Running Tests

```bash
# Run all tests with CPU features
cargo test --workspace --no-default-features --features cpu

# Run all tests with GPU features
cargo test --workspace --no-default-features --features gpu

# Test device-aware quantization for all quantizers (I2S, TL1, TL2)
cargo test -p bitnet-quantization --no-default-features --features gpu test_dequantize_cpu_and_gpu_paths

# Run specific test suites
cargo test --package bitnet-tests --no-default-features --features fixtures
cargo test --package bitnet-tests --features reporting,trend

# Run integration tests
cargo test --package bitnet-tests --features integration-tests,fixtures

# Run examples
cargo run --example reporting_example --features reporting
cargo run --example ci_reporting_example --features reporting,trend
cargo run --example debugging_example --features fixtures

# Run GGUF format validation tests
cargo test -p bitnet-inference --test gguf_header                 # Pure parser test
cargo test -p bitnet-inference --no-default-features --features rt-tokio --test smoke -- --nocapture  # Async smoke test

# Run verification script for all tests
./scripts/verify-tests.sh
```

### Test Configuration

The test suite uses a feature-gated configuration system:
- `fixtures`: Enables fixture management and test data generation
- `reporting`: Enables test reporting (JSON, HTML, Markdown, JUnit)
- `trend`: Enables trend analysis and performance tracking
- `integration-tests`: Enables full integration test suite

### Test Features

- **Parallel Test Execution**: Configurable parallelism with resource limits
- **Fixture Management**: Automatic test data generation and caching
- **CI Integration**: JUnit output, exit codes, and CI-specific optimizations
- **Error Reporting**: Detailed error messages with recovery suggestions
- **Performance Tracking**: Benchmark results and regression detection

## Validation Framework

### Evaluation Commands

```bash
# Evaluate perplexity on a corpus (token-weighted NLL)
target/release/bitnet eval \
  --model models/bitnet/model.gguf \
  --tokenizer models/bitnet/tokenizer.json \
  --text-file crossval/data/ppl_smoke.txt

# Teacher-forcing with explicit token IDs + logit dump
target/release/bitnet eval \
  --model models/bitnet/model.gguf \
  --tokenizer models/bitnet/tokenizer.json \
  --teacher-force-ids 1,2,3,4,5,6 \
  --dump-logit-steps 6 --logits-topk 10 \
  --json-out /tmp/tf_eval.json

# Deterministic greedy generation with logit tapping
target/release/bitnet run \
  --model models/bitnet/model.gguf \
  --tokenizer models/bitnet/tokenizer.json \
  --prompt "Define entropy." \
  --max-new-tokens 32 --greedy \
  --deterministic --threads 1 \
  --dump-logit-steps 8 --logits-topk 10 \
  --json-out /tmp/run.json
```

### Validation Tests

```bash
# Tokenizer parity check
BITNET_BIN=target/release/bitnet \
MODEL_PATH=models/bitnet/model.gguf \
TOKENIZER=models/bitnet/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
scripts/test-tokenizer-parity.py --smoke

# Logit parity with tau-b correlation
PROP_EXAMPLES=10 TAU_STEPS=24 LOGIT_TOPK=10 TAU_MIN=0.60 \
MODEL_PATH=models/bitnet/model.gguf \
TOKENIZER=models/bitnet/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
scripts/logit-parity.sh

# NLL parity (token-weighted)
DELTA_NLL_MAX=1e-2 \
MODEL_PATH=models/bitnet/model.gguf \
TOKENIZER=models/bitnet/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
PPL_FILE=crossval/data/ppl_smoke.txt \
scripts/nll-parity.sh
```

### Key Validation Features

1. **Token-Weighted NLL**: Proper corpus perplexity using `Σ(token_nlls) / Σ(predicted_tokens)`
2. **Teacher-Forcing**: Exact decode path with causal masking and position encoding
3. **Deterministic Top-K**: Stable sorting with tie-breaking by token ID, NaN demotion
4. **Logit Dumping**: Capture top-k logits at each generation step for analysis
5. **Tau-b Correlation**: Score-aware rank correlation for quantization robustness
