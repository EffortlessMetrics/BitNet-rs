# CLAUDE.md

This file provides guidance to Claude (claude.ai) when working with the BitNet.rs codebase.

## Build and Development Commands

### Core Development Commands
```bash
# Build with CPU support (no default features)
cargo build --release --no-default-features --features cpu

# Build with GPU support and device-aware quantization
cargo build --release --no-default-features --features gpu

# Backward compatibility with cuda alias
cargo build --release --no-default-features --features cuda

# Build with IQ2_S quantization support (requires GGML FFI)
cargo build --release --no-default-features --features "cpu,iq2s-ffi"

# Build with native I2_S support (no external dependencies)
cargo build --release --no-default-features --features cpu

# Run tests (fast, Rust-only)
cargo test --workspace --no-default-features --features cpu

# Run GPU tests with device-aware quantization (requires CUDA)
cargo test --workspace --no-default-features --features gpu

# Run comprehensive GPU quantization tests
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_quantization

# Run GPU quantization integration tests (comprehensive validation)
cargo test -p bitnet-kernels --no-default-features --features gpu --test gpu_quantization --ignored

# Test real CUDA device property querying (compute capability, memory, multiprocessors)
cargo test -p bitnet-kernels --no-default-features --features gpu test_cuda_device_info_query

# Test GPU vs CPU quantization accuracy
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_vs_cpu_quantization_accuracy --ignored

# Test automatic GPU fallback mechanism
cargo test -p bitnet-kernels --no-default-features --features gpu test_gpu_quantization_fallback --ignored

# Test concurrent GPU operations
cargo test -p bitnet-kernels --no-default-features --features gpu test_concurrent_gpu_operations --ignored

# Legacy GPU parity tests (backward compatibility)
cargo test --workspace --no-default-features --features cuda gpu_parity

# Run GGUF validation tests
cargo test -p bitnet-inference --test gguf_header
cargo test -p bitnet-inference --test gguf_fuzz
cargo test -p bitnet-inference --test engine_inspect

# Run async smoke test with synthetic GGUF
printf "GGUF\x02\x00\x00\x00" > /tmp/t.gguf && \
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf && \
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf && \
BITNET_GGUF=/tmp/t.gguf cargo test -p bitnet-inference --features rt-tokio --test smoke

# Run verification script
./scripts/verify-tests.sh

# Run benchmarks
cargo bench --workspace --no-default-features --features cpu

# Cross-validation testing (requires C++ dependencies)
cargo test --workspace --features "cpu,ffi,crossval"
```

### Code Quality Commands
```bash
# Format all code
cargo fmt --all

# Run clippy with pedantic lints
cargo clippy --all-targets --all-features -- -D warnings

# Security audit
cargo audit

# Generate code coverage
cargo llvm-cov --workspace --features cpu --html

# Run tests with concurrency caps (prevents resource storms)
scripts/preflight.sh && cargo t2                     # 2-thread CPU tests
scripts/preflight.sh && cargo crossval-capped        # Cross-validation with caps
scripts/e2e-gate.sh cargo test --features crossval   # Gate heavy E2E tests
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
- **`bitnet-kernels`**: High-performance SIMD/CUDA kernels
- **`bitnet-inference`**: Inference engine with streaming support
- **`bitnet-tokenizers`**: Universal tokenizer support
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

## Important Considerations

### MSRV
Minimum Supported Rust Version: **1.89.0** (uses Rust 2024 edition)

### Feature Flags
Default features are **empty** to prevent unwanted dependencies:
- `cpu`: CPU inference with SIMD optimizations, includes native I2_S support
- `gpu`: NVIDIA GPU support with advanced device-aware quantization and automatic fallback
- `cuda`: Backward-compatible alias for `gpu` feature
- `iq2s-ffi`: IQ2_S quantization via GGML FFI (requires vendored GGML files)
- `ffi`: C++ FFI bridge (required for cross-validation) with enhanced safety documentation
- `crossval`: Cross-validation against C++ (increases build time)

### Quantization Support
BitNet-rs supports multiple quantization formats with device-aware acceleration:
- **I2_S**: Native Rust implementation with intelligent GPU/CPU selection and automatic fallback
- **TL1**: Table lookup quantization with GPU acceleration and CPU fallback
- **TL2**: Advanced table lookup quantization with optimized GPU kernels and device-aware execution
- **IQ2_S**: Dual implementation:
  - Native Rust: Optimized implementation with SIMD support
  - GGML FFI: Via `iq2s-ffi` feature for llama.cpp compatibility
  - Backend parity testing ensures both implementations produce identical results
- **Standard formats**: Q4_0, Q5_0, Q8_0, etc. (planned)

All quantizers support device-aware operations:
- **Production-Ready GPU Detection**: Real CUDA device property querying using cudarc API
- **Comprehensive Device Information**: Compute capability, memory size, multiprocessor count, thread limits
- **Hardware-Aware Optimization**: Feature detection for FP16/BF16 support based on compute capability
- **Automatic GPU acceleration** when available with transparent CPU fallback
- **Consistent results across devices** via comprehensive parity testing

To test IQ2_S implementations:
```bash
# Run IQ2_S backend validation
./scripts/test-iq2s-backend.sh

# Run unit tests
cargo test --package bitnet-models --no-default-features --features "cpu,iq2s-ffi"
```

### Testing Strategy
- **Unit tests**: Each crate has comprehensive tests
- **Integration tests**: Cross-crate tests in `tests/`
- **Property-based testing**: Fuzz testing for GGUF parser robustness
- **Cross-validation**: Automated testing against C++ implementation
- **CI gates**: Compatibility tests block on every PR

### Compatibility Guarantees
We maintain strict compatibility with llama.cpp:
- C API functions have exact signature matches
- Python API is drop-in compatible
- We handle models that llama.cpp fails on (e.g., GPT-2 without pre-tokenizer)
- See COMPATIBILITY.md for detailed guarantees

## Troubleshooting

### Common Build Issues

1. **FFI Linker Errors**: Either disable FFI (`--no-default-features --features cpu`) or build C++ (`cargo xtask fetch-cpp`)

2. **CUDA Compilation**: Ensure CUDA toolkit is installed and `nvcc` is in PATH

3. **CUDA Device Query Failures**: If CUDA device information queries fail:
   - Verify CUDA runtime is properly installed: `nvidia-smi` and `nvcc --version`
   - Check driver compatibility: Ensure CUDA driver supports the installed toolkit
   - Test device access: `cargo test -p bitnet-kernels --features gpu test_cuda_device_info_query`
   - For permission issues on Linux, add user to `video` group: `sudo usermod -a -G video $USER`
   - Verify cudarc compatibility: Requires CUDA 11.0+ and compute capability 6.0+

4. **Cross-Validation Path**: Set `BITNET_GGUF` environment variable to model path

5. **Git Metadata in Builds**: The `bitnet-server` crate uses `vergen-gix` v1.x to capture Git metadata. Ensure `.git` is available during builds or set `VERGEN_GIT_SHA` and `VERGEN_GIT_BRANCH` environment variables

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

## Environment Variables

### Runtime Variables
- `BITNET_GGUF` / `CROSSVAL_GGUF`: Path to test model
- `BITNET_CPP_DIR`: Path to C++ implementation
- `HF_TOKEN`: Hugging Face token for private repos
- `BITNET_DETERMINISTIC`: Enable deterministic mode for testing
- `BITNET_SEED`: Set seed for reproducible runs
- `RAYON_NUM_THREADS`: Control CPU parallelism

### Build-time Variables (for Git metadata)
- `VERGEN_GIT_SHA`: Override Git SHA (useful in CI/Docker without .git)
- `VERGEN_GIT_BRANCH`: Override Git branch
- `VERGEN_GIT_DESCRIBE`: Override Git describe output
- `VERGEN_IDEMPOTENT`: Set to "1" for reproducible builds

## Concurrency Caps Implementation

BitNet-rs implements a comprehensive concurrency capping strategy to prevent resource exhaustion during parallel test execution:

### Core Components

1. **Preflight Script** (`scripts/preflight.sh`):
   - Monitors system resource usage (PIDs, file descriptors, load)
   - Sets conservative concurrency defaults
   - Auto-degrades to single-threaded mode under high system load
   - Configures BLAS thread limits for Python validation scripts

2. **E2E Gate** (`scripts/e2e-gate.sh`):
   - Limits concurrent heavy test suites (cross-validation, integration tests)
   - Uses file locking to queue test runs when system is busy
   - Integrates with preflight caps for resource management

3. **Cargo Aliases** (`.cargo/config.toml`):
   - `t2`: Run tests with 2-thread cap
   - `t1`: Run tests with 1-thread (deterministic mode)
   - `crossval-capped`: Cross-validation with thread caps
   - `gpu-capped`: GPU tests with concurrency limits

4. **Test Infrastructure** (`tests/common/concurrency_caps.rs`):
   - Rayon thread pool initialization with caps
   - Async task concurrency limits
   - Deterministic execution helpers

5. **Container Limits** (`docker-compose.test.yml`):
   - Hard resource limits (CPU, memory, PIDs, file descriptors)
   - Isolated test environments for different workloads
   - Volume caching for faster subsequent builds

### Environment Variables

```bash
# Thread control
RUST_TEST_THREADS=2      # Rust test parallelism
RAYON_NUM_THREADS=2      # Rayon thread pool size
CROSSVAL_WORKERS=2       # Cross-validation test workers

# BLAS thread limits (Python scripts)
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1

# BitNet deterministic execution
BITNET_DETERMINISTIC=1
BITNET_SEED=42
```

### Usage Examples

```bash
# Standard capped test execution
scripts/preflight.sh && cargo t2

# Gate heavy E2E tests (max 2 concurrent)
scripts/e2e-gate.sh ./scripts/crossval.sh

# Container-isolated testing
docker-compose -f docker-compose.test.yml up rust-cpu-tests

# CI/deterministic mode
RUST_TEST_THREADS=1 RAYON_NUM_THREADS=1 cargo t1
```

## Repository Contracts (for Claude)

### Safe Operations
- **Default features are empty** → always pass `--no-default-features --features cpu|cuda`
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

## GGUF Metadata Inspection (Enhanced)

BitNet.rs provides comprehensive GGUF metadata inspection capabilities with advanced categorization and JSON serialization for detailed analysis without loading tensors into memory:

### Core Features

1. **Lightweight Header Parsing**: Only reads GGUF header for fast analysis
2. **Comprehensive Metadata Extraction**: KV pairs, quantization hints, tensor summaries with categorization
3. **Smart Categorization**: Automatically categorizes metadata into model parameters, architecture, tokenizer, training, quantization, and other categories
4. **Enhanced Tensor Analysis**: Detailed tensor categorization (embeddings, weights, biases, normalization, attention, feed-forward, output heads)
5. **Memory Estimation**: Automatic memory footprint calculation based on tensor dtypes
6. **JSON Serialization**: Full JSON export capability for programmatic processing
7. **Statistical Analysis**: Parameter counts, dtype distribution, and memory usage statistics
8. **Memory Efficient**: No tensor data loading required
9. **Error Resilient**: Handles malformed GGUF files gracefully
10. **Performance Optimized**: Suitable for CI/CD pipelines and automation

### API Usage

```rust
use bitnet_inference::engine::inspect_model;

// Lightweight inspection with enhanced features
let mut model_info = inspect_model("model.gguf")?;

// Access raw metadata (backward compatible)
let kv_specs = model_info.kv_specs();           // All key-value metadata
let quant_hints = model_info.quantization_hints(); // Quantization-related metadata  
let tensors = model_info.tensor_summaries();    // Enhanced tensor summaries with categorization

// Access enhanced categorized metadata
let categorized = model_info.get_categorized_metadata();
println!("Model parameters: {:?}", categorized.model_params);
println!("Architecture info: {:?}", categorized.architecture);
println!("Tokenizer config: {:?}", categorized.tokenizer);
println!("Training metadata: {:?}", categorized.training);
println!("Quantization details: {:?}", categorized.quantization);

// Access tensor statistics
let stats = model_info.get_tensor_statistics();
println!("Total parameters: {}", stats.total_parameters);
println!("Memory estimate: {} bytes", stats.estimated_memory_bytes);
println!("Parameters by category: {:?}", stats.parameters_by_category);
println!("Largest tensor: {:?}", stats.largest_tensor);

// JSON serialization
let json_pretty = model_info.to_json()?;        // Pretty-printed JSON
let json_compact = model_info.to_json_compact()?; // Compact JSON
```

### Commands

```bash
# Enhanced CLI inspection (planned integration)
cargo run -p bitnet-cli -- inspect --model model.gguf --json

# Enhanced example with categorized human-readable output
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- model.gguf

# Enhanced example with JSON output for programmatic processing
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- --json model.gguf

# Using environment variable
BITNET_GGUF=model.gguf cargo run --example inspect_gguf_metadata --no-default-features --features cpu

# JSON output with environment variable
BITNET_GGUF=model.gguf cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- --json

# Quick header validation (fast path)
cargo test -p bitnet-inference --test engine_inspect

# Test enhanced categorization and JSON features
cargo test -p bitnet-inference --test engine_inspect -- comprehensive_metadata_categorization
cargo test -p bitnet-inference --test engine_inspect -- json_serialization
cargo test -p bitnet-inference --test engine_inspect -- categorization_functions
```

### Use Cases

- **CI/CD**: Fast model validation in deployment pipelines with comprehensive metadata extraction
- **Model Analysis**: Understand quantization schemes, architecture, and parameter distribution without full loading
- **Model Cataloging**: Automated metadata extraction for model management systems with JSON export
- **Performance Planning**: Memory usage estimation and parameter analysis for deployment sizing
- **Debugging**: Inspect GGUF structure for compatibility issues with detailed categorization
- **Integration**: Programmatic access to model metadata through JSON API for downstream tools
- **Research**: Analyze model architecture patterns and quantization strategies across model families

## Streaming Generation with Token IDs

BitNet.rs provides comprehensive streaming generation capabilities with real-time token ID access, supporting both library API and Server-Sent Events (SSE) for web applications.

### Core Features

1. **StreamResponse Structure**: Enhanced streaming response with both text and token IDs (added in v0.1.0)
2. **Real-time Generation**: Token-by-token streaming with configurable buffering
3. **Server-Sent Events**: HTTP/1.1 SSE endpoint with JSON token metadata
4. **Error Resilience**: Comprehensive error handling with recovery suggestions
5. **Performance Optimization**: Configurable buffer sizes and flush intervals

### Library API Usage

```rust
use bitnet_inference::{InferenceEngine, GenerationConfig, StreamingConfig};
use futures::StreamExt;

// Create streaming configuration
let streaming_config = StreamingConfig {
    buffer_size: 10,
    flush_interval_ms: 50,
    max_retries: 3,
    token_timeout_ms: 5000,
    cancellable: true,
};

// Generate with streaming
let mut stream = engine.generate_stream_with_config("Explain quantum computing", &config);

while let Some(result) = stream.next().await {
    match result {
        Ok(stream_response) => {
            // Access generated text
            print!("{}", stream_response.text);
            
            // Access token IDs (new in v0.1.0)
            for &token_id in &stream_response.token_ids {
                eprintln!("[DEBUG] Token ID: {}", token_id);
            }
        }
        Err(e) => eprintln!("Stream error: {}", e),
    }
}
```

### Server-Sent Events (SSE) API

Start the BitNet server with streaming support:

```bash
# Start server with model
cargo run -p bitnet-server -- --port 8080 --model model.gguf

# Test streaming endpoint
curl -X POST http://localhost:8080/v1/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing", 
    "max_tokens": 100, 
    "temperature": 0.7,
    "detailed_errors": true
  }' \
  --no-buffer
```

### SSE Response Format

The streaming endpoint returns Server-Sent Events with the following structure:

```javascript
// Token event (generated text + token ID)
event: token
data: {
  "token": "quantum",
  "token_id": 24975,
  "cumulative_time_ms": 150,
  "position": 1
}

// Completion event (final statistics)
event: complete
data: {
  "total_tokens": 95,
  "total_time_ms": 3200,
  "tokens_per_second": 29.69,
  "completed_normally": true,
  "completion_reason": "Generation completed successfully"
}

// Error event (if issues occur)
event: error
data: {
  "error_type": "generation",
  "message": "Token generation failed",
  "recovery_hints": ["Try reducing max_tokens", "Check model compatibility"],
  "tokens_before_error": 15
}
```

### Testing Streaming Functionality

```bash
# Test streaming generation
cargo run --example streaming_generation --no-default-features --features cpu

# Test server streaming
cargo test -p bitnet-server --no-default-features --features cpu streaming

# Test token ID accuracy
cargo test -p bitnet-inference --no-default-features --features cpu test_token_id_streaming

# Test concurrent streaming
cargo test -p bitnet-inference --no-default-features --features cpu test_concurrent_streams

# Validate SSE format
cargo test -p bitnet-server --no-default-features --features cpu sse_token_ids_match_model_outputs
```

### Troubleshooting Streaming Issues

```bash
# Debug streaming with detailed logging
RUST_LOG=bitnet_inference::streaming=debug cargo run --example streaming_generation

# Test with different buffer configurations
cargo run --example streaming_generation -- --buffer-size 5 --flush-interval 25ms

# Validate token ID consistency
cargo test -p bitnet-inference test_streaming_token_id_consistency

# Check server streaming health
curl -X GET http://localhost:8080/health
```

### PyO3 Security Note

Starting with v0.1.0, BitNet.rs uses PyO3 v0.25.1 to resolve CVE-2024-9979 security vulnerability. This affects Python bindings and server components:

```bash
# Verify PyO3 version
cargo tree -p bitnet-py | grep pyo3

# Expected: pyo3 v0.25.1 or later
```

## Fast Recipes

```bash
# Quick compile & test (CPU, MSRV-accurate)
rustup run 1.89.0 cargo test --workspace --no-default-features --features cpu

# Quick compile & test with concurrency caps
scripts/preflight.sh && cargo t2

# Validate GGUF file
cargo run -p bitnet-cli -- compat-check model.gguf
cargo run -p bitnet-cli -- compat-check model.gguf --json  # JSON output

# Inspect model metadata (human-readable with categorization)
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- model.gguf

# Export model metadata as JSON
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- --json model.gguf

# Full cross-validation (deterministic)
export BITNET_GGUF="$PWD/models/bitnet/ggml-model-i2_s.gguf"
export BITNET_DETERMINISTIC=1 BITNET_SEED=42
cargo run -p xtask -- full-crossval

# Check model compatibility (read-only)
cargo run -p bitnet-cli -- compat-check "$BITNET_GGUF"

# Inspect comprehensive GGUF metadata (enhanced)
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- "$BITNET_GGUF"         # Human-readable with categorization
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- --json "$BITNET_GGUF"  # JSON output
BITNET_GGUF="$BITNET_GGUF" cargo run --example inspect_gguf_metadata --no-default-features --features cpu # Using environment variable

# Export fixed GGUF safely (non-destructive)
cargo run -p bitnet-cli -- compat-fix "$BITNET_GGUF" fixed.gguf
cat fixed.gguf.compat.json   # audit stamp

# FFI smoke test (build + run)
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