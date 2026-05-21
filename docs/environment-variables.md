# Environment Variables Reference

This document describes all environment variables used throughout BitNet-rs for configuration, testing, and development.

## Runtime Variables

### Model and Testing Configuration
- `BITNET_GGUF` / `CROSSVAL_GGUF`: Path to test model
- `BITNET_CPP_DIR`: Path to C++ implementation
- `HF_TOKEN`: Hugging Face token for private repos
- `BITNET_DETERMINISTIC`: Enable deterministic mode for testing
- `BITNET_SEED`: Set seed for reproducible runs
- `BITNET_STRICT_MODE`: Prevent mock inference fallbacks and validate LayerNorm gamma statistics ("1" enables strict mode for production)
  - Prevents all mock inference paths
  - Validates LayerNorm gamma weights have mean ≈ 1.0
  - Fails immediately on suspicious LayerNorm statistics (mean outside [0.5, 2.0])
  - In non-strict mode (default), issues warnings but continues

### Model Validation and Correction Policy
- `BITNET_CORRECTION_POLICY`: Path to YAML policy file defining model-specific corrections
  - **Value**: Absolute or relative path to policy YAML file (e.g., `/path/to/policy.yml`)
  - **Purpose**: Enable runtime corrections for known-bad models with fingerprinted, auditable fixes
  - **Format**: YAML file specifying model fingerprints and correction parameters
  - **Usage**:
    ```bash
    # Enable policy-driven corrections
    export BITNET_CORRECTION_POLICY=/path/to/correction-policy.yml
    export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
    cargo run -p bitnet-cli -- run --model model.gguf
    ```
  - **Important**: Both `BITNET_CORRECTION_POLICY` and `BITNET_ALLOW_RUNTIME_CORRECTIONS` must be set

- `BITNET_ALLOW_RUNTIME_CORRECTIONS`: Enable runtime corrections (must be used with BITNET_CORRECTION_POLICY)
  - **Value**: "1" to enable (disabled by default)
  - **Purpose**: Safety gate preventing accidental application of corrections
  - **Warning**: CI blocks correction flags - runtime corrections are for known-bad models only
  - **Proper fix**: Always prefer regenerating GGUF with LayerNorm weights in FP16/FP32 (not quantized)
  - **Usage**:
    ```bash
    # Inspect model statistics first
    cargo run -p bitnet-cli -- inspect --ln-stats model.gguf

    # Apply corrections if needed (temporary workaround)
    export BITNET_CORRECTION_POLICY=./model-corrections.yml
    export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
    cargo run -p bitnet-cli -- run --model model.gguf
    ```

- `BITNET_RESCALE_GAMMA_ON_LOAD`: **Experimental** - Rescale LayerNorm gamma by √hidden_size during loading
  - **Value**: "1" to enable (disabled by default)
  - **Purpose**: Test hypothesis that bitnet.cpp rescales pre-scaled gamma weights on load
  - **Algorithm**: For LayerNorm tensors, applies `gamma' = gamma * sqrt(hidden_size)`
  - **Use case**: If gamma RMS ≈ 0.018 = 1/√2560, this rescales to RMS ≈ 1.0
  - **Safety**: Disabled in strict mode (`BITNET_STRICT_MODE=1`)
  - **Status**: Experimental feature for investigating activation magnitude discrepancies
  - **Usage**:
    ```bash
    # Enable experimental gamma rescaling
    export BITNET_RESCALE_GAMMA_ON_LOAD=1
    cargo run -p bitnet-cli --features cpu,full-cli -- run \
      --model model.gguf \
      --tokenizer tokenizer.json \
      --prompt "Test" \
      --max-tokens 16

    # Check rescaling logs (look for "EXPERIMENTAL: Rescaled" messages)
    RUST_LOG=info BITNET_RESCALE_GAMMA_ON_LOAD=1 \
    cargo run -p bitnet-cli --features cpu,full-cli -- run \
      --model model.gguf --tokenizer tokenizer.json --prompt "Test"
    ```
  - **Important**: This is an experimental diagnostic tool, not a production fix. Always prefer regenerating GGUF with correct LayerNorm weights.

### Performance and Parallelism
- `RAYON_NUM_THREADS`: Control CPU parallelism (Rayon thread pool)
- `BITNET_CPU_THREADS`: CPU thread count for inference (overrides CLI config)
- `BITNET_NUM_THREADS`: Alternative thread count setting (used in some crates)

### Device Selection
- `BITNET_DEVICE`: Device for inference — `cpu`, `cuda`, `metal`, `vulkan` (default: `cpu`)
- `BITNET_LOG_LEVEL`: Log level — `trace`, `debug`, `info`, `warn`, `error`

### Lunar Lake OpenVINO Ask Runtime

- `BITNET_LUNAR_LAKE_OPENVINO_MODEL_DIR`: OpenVINO IR directory used by `bitnet lunar-lake ask` when an auto-selected OpenVINO GPU/NPU route needs a local model and `--model` is omitted. This is intended for ignored local model exports such as `models/openvino/qwen2.5-0.5b-instruct-int4-sym`; it does not commit model binaries or change route promotion.
- `BITNET_LUNAR_LAKE_OPENVINO_PYTHON`: Python executable used by the Lunar Lake OpenVINO GenAI helper. Set this when the checkout-local `.venv/Scripts/python.exe` is absent or does not include `openvino_genai`.

### Model Configuration (Environment Overrides)
These override CLI arguments and config file values:
- `BITNET_MODEL_PATH`: Path to model file (GGUF or SafeTensors)
- `BITNET_MODEL_FORMAT`: Model format — `gguf`, `safetensors`
- `BITNET_ARCHITECTURE`: Model architecture hint — `bitnet`, `llama`, `phi`, etc.
- `BITNET_HIDDEN_SIZE`: Hidden dimension size (e.g., `2560`)
- `BITNET_NUM_LAYERS`: Number of transformer layers
- `BITNET_NUM_HEADS`: Number of attention heads
- `BITNET_VOCAB_SIZE`: Vocabulary size
- `BITNET_BLOCK_SIZE`: Transformer block size
- `BITNET_REQUIRE_LAYER_NORM_BIAS`: Require LayerNorm bias tensors (`1` to enable)

### Generation Parameters
- `BITNET_MAX_TOKENS`: Maximum number of tokens to generate
- `BITNET_MAX_NEW_TOKENS`: Maximum new tokens (alias for `MAX_TOKENS` in some contexts)
- `BITNET_MAX_LENGTH`: Maximum total sequence length
- `BITNET_TEMPERATURE`: Sampling temperature (e.g., `0.7`)
- `BITNET_TOP_K`: Top-k sampling parameter (e.g., `40`)
- `BITNET_TOP_P`: Top-p (nucleus) sampling parameter (e.g., `0.9`)
- `BITNET_BATCH_SIZE`: Batch size for inference

### Tokenizer
- `BITNET_TOKENIZER`: Path to tokenizer file (tokenizer.json)
- `BITNET_OFFLINE`: Disable network access for tokenizer downloads (`1` to enable)
- `BITNET_CACHE_DIR`: Cache directory for downloaded tokenizers

### Quantization and Debug
- `BITNET_QUANTIZATION_TYPE`: Force quantization type — `i2s`, `tl1`, `tl2`, `qk256`
- `BITNET_IQ2S_IMPL`: I2S implementation selector
- `BITNET_QUANT_SANITY`: Enable quantization sanity checks (`1` to enable)
- `BITNET_DISABLE_MINIMAL_LOADER`: Disable minimal GGUF loader (`1` to disable)
- `BITNET_PARITY`: Enable parity checking mode

### Tracing and Debug Output
- `BITNET_TRACE_DIR`: Directory for tensor activation trace output
- `BITNET_TRACE_QUANT`: Enable quantization tracing (`1` to enable)
- `BITNET_TRACE_RMS`: Enable RMS norm tracing
- `BITNET_TRACE_TIMING`: Enable timing trace output
- `BITNET_DEBUG_LOGITS`: Enable logits debugging output
- `BITNET_DEBUG_ATTN_SCALE`: Debug attention scaling
- `BITNET_DEBUG_GQA`: Debug grouped-query attention
- `BITNET_DEBUG_MLP`: Debug MLP/FFN layer
- `BITNET_DEBUG_RMSNORM`: Debug RMS normalization
- `BITNET_DEBUG_ROPE`: Debug rotary position embeddings
- `BITNET_DEBUG_TIMEOUT_SECS`: Debug timeout in seconds

### GPU Configuration
- `BITNET_USE_GPU`: Enable GPU acceleration (`1` to enable)
- `BITNET_GPU_MEMORY_LIMIT`: GPU memory limit in bytes
- `BITNET_GPU_CACHE`: GPU kernel cache directory
- `BITNET_GPU_DEBUG`: Enable GPU debug output (`1` to enable)
- `BITNET_ENABLE_NPU`: Enable NPU backend (`1` to enable)
- `BITNET_NPU_BACKEND`: NPU backend selection
- `BITNET_ENABLE_ROCM`: Enable ROCm backend (`1` to enable)

### GPU Feature Detection (Issue #439)
- `BITNET_GPU_FAKE`: Override GPU detection for deterministic testing and device-aware fallback validation
  - **Values**:
    - `none`: Disable GPU detection (test CPU fallback paths)
    - `cuda` or `gpu`: Enable fake GPU detection (test GPU code paths without hardware)
    - `metal`, `rocm`: Simulate specific GPU backends
    - Multiple backends: `cuda,rocm` (comma-separated)
  - **Usage with Preflight**:
    ```bash
    # Test CPU fallback behavior
    BITNET_GPU_FAKE=none cargo run -p xtask -- preflight
    # Expected: "✗ GPU: Not available at runtime"

    # Test GPU path without hardware
    BITNET_GPU_FAKE=cuda cargo run -p xtask -- preflight
    # Expected: "✓ GPU: Available"
    ```
  - **Device-Aware Testing**:
    ```bash
    # Test quantization device selection with fake GPU
    BITNET_GPU_FAKE=cuda cargo test --no-default-features --features gpu -p bitnet-quantization

    # Test CPU fallback in GPU-compiled binary
    BITNET_GPU_FAKE=none cargo test --no-default-features --features gpu -p bitnet-inference
    ```

## Strict Testing Mode Variables

These variables prevent "Potemkin passes" (false positives) in performance and integration tests by eliminating mock inference paths and ensuring honest quantized computation:

### Primary Strict Mode (Issue #453 - Three-Tier Validation)
- `BITNET_STRICT_MODE=1`: **Primary strict mode** - Prevents ALL mock inference fallbacks and FP32 quantization fallbacks, essential for production deployment and accurate performance measurement
  - **Tier 1 (Development):** Debug assertions catch fallbacks in debug builds (panics immediately)
  - **Tier 2 (Production):** Strict mode returns `Err(BitNetError::StrictMode(...))` in release builds
  - **Tier 3 (Verification):** Receipt validation ensures honest computation claims
  - Enables `fail_on_mock`, `require_quantization`, `enforce_quantized_inference`, and `validate_performance` checks
  - Fails fast when mock computation or FP32 fallback is detected
  - Validates performance metrics to reject suspicious values (>150 tok/s flagged as potentially mock)
  - Required for production deployments to ensure real quantized inference
  - **Usage:**
    ```bash
    # Production inference with strict mode
    BITNET_STRICT_MODE=1 \
    cargo run --release -p bitnet-cli --no-default-features --features cpu -- \
      infer --model model.gguf --prompt "Test" --max-tokens 16

    # If FP32 fallback would occur:
    # Error: Strict mode: FP32 fallback rejected - qtype=I2S, device=Cpu, layer_dims=[2048, 2048], reason=kernel_unavailable
    ```

### Detailed Strict Mode Controls (Issue #453 - Granular Configuration)

- `BITNET_STRICT_FAIL_ON_MOCK=1`: Fail immediately when mock computation is detected in inference pipeline
  - Activated automatically when `BITNET_STRICT_MODE=1`
  - Can be enabled independently for targeted testing
  - Validates all tensor operations and kernel calls for mock usage
  - **Usage:**
    ```bash
    # Fail on mock detection only
    BITNET_STRICT_FAIL_ON_MOCK=1 \
    cargo test -p bitnet-inference test_inference_real_computation
    ```

- `BITNET_STRICT_REQUIRE_QUANTIZATION=1`: Require real quantization kernels (I2S/TL1/TL2) to be available and used **(Issue #453 - AC3)**
  - Activated automatically when `BITNET_STRICT_MODE=1`
  - Prevents fallback to FP32 dequantization staging when quantization expected
  - Validates device-aware quantization kernel selection
  - Rejects FP32 fallback in `QuantizedLinear::forward` with detailed error
  - Error includes: quantization type, device, layer dimensions, fallback reason
  - **Usage:**
    ```bash
    # Require quantization kernels only
    BITNET_STRICT_REQUIRE_QUANTIZATION=1 \
    cargo test -p bitnet-quantization test_quantization_kernel_integration

    # If kernel unavailable:
    # Error: Strict mode: FP32 fallback rejected - qtype=I2S, device=Cpu,
    #        layer_dims=[2048, 2048], reason=kernel_unavailable
    ```

- `BITNET_STRICT_VALIDATE_PERFORMANCE=1`: Validate performance metrics for realistic values **(Issue #453 - AC6)**
  - Activated automatically when `BITNET_STRICT_MODE=1`
  - Rejects performance metrics from mock computation paths
  - Flags unrealistic throughput (>150 tok/s) as suspicious
  - Validates `tokens_per_second` against baseline thresholds
  - **Usage:**
    ```bash
    # Validate performance metrics only
    BITNET_STRICT_VALIDATE_PERFORMANCE=1 \
    cargo run -p xtask -- benchmark --model model.gguf --tokens 128

    # Then verify receipt
    cargo run -p xtask -- verify-receipt --validate-performance ci/inference.json
    ```

- `BITNET_CI_ENHANCED_STRICT=1`: Enhanced strict mode for CI environments **(Issue #453 - AC6)**
  - Activates when both `CI` environment variable and this flag are set
  - Enables `ci_enhanced_mode`, `log_all_validations`, and `fail_fast_on_any_mock`
  - Provides comprehensive logging for CI pipeline debugging
  - Ensures production-grade validation in automated testing
  - **Usage:**
    ```yaml
    # .github/workflows/strict-mode-ci.yml
    - name: Run strict mode tests
      env:
        CI: "1"
        BITNET_CI_ENHANCED_STRICT: "1"
        BITNET_STRICT_MODE: "1"
      run: cargo test --workspace --no-default-features --features cpu
    ```

### Legacy Strict Mode Variables
- `BITNET_STRICT_TOKENIZERS=1`: Forbid mock tokenizer fallbacks in perf/integration tests (includes SPM tokenizer fallbacks)
- `BITNET_STRICT_NO_FAKE_GPU=1`: Forbid fake GPU backends in perf/integration tests

## Build-time Variables

For Git metadata capture (used by `bitnet-server` crate with `vergen-gix`):

- `VERGEN_GIT_SHA`: Override Git SHA (useful in CI/Docker without .git)
- `VERGEN_GIT_BRANCH`: Override Git branch
- `VERGEN_GIT_DESCRIBE`: Override Git describe output
- `VERGEN_IDEMPOTENT`: Set to "1" for reproducible builds

## FFI Configuration

### Compiler Selection
```bash
# GCC (default)
export CC=gcc CXX=g++

# Clang
export CC=clang CXX=clang++
```

### Library Path Configuration
```bash
# Linux FFI
export LD_LIBRARY_PATH=target/release

# macOS FFI
export DYLD_LIBRARY_PATH=target/release
```

## Server Configuration

Environment variables for `bitnet-server` (axum HTTP server). All are optional with sensible defaults.

### Server Settings
- `BITNET_SERVER_HOST`: Bind address (default: `0.0.0.0`)
- `BITNET_SERVER_PORT`: Port number (default: `3000`)
- `BITNET_SERVER_WORKERS`: Worker thread count (default: auto-detected)
- `BITNET_REQUEST_TIMEOUT`: Request timeout in seconds (default: `30`)
- `BITNET_DEFAULT_MODEL_PATH`: Default model file path
- `BITNET_DEFAULT_TOKENIZER_PATH`: Default tokenizer file path
- `BITNET_DEFAULT_DEVICE`: Default inference device — `cpu`, `cuda`

### Model Manager
- `BITNET_MAX_CONCURRENT_LOADS`: Maximum concurrent model loads (default: `2`)
- `BITNET_MODEL_CACHE_SIZE`: Model cache capacity (default: `4`)
- `BITNET_MEMORY_LIMIT_GB`: Memory limit in GB for model loading
- `BITNET_MODEL_VALIDATION`: Enable model validation on load (`true`/`false`)

### Execution Router
- `BITNET_DEVICE_STRATEGY`: Device selection strategy — `auto`, `cpu`, `gpu`, `hybrid`
- `BITNET_FALLBACK_ENABLED`: Enable CPU fallback when GPU fails (`true`/`false`)
- `BITNET_BENCHMARK_ON_STARTUP`: Run benchmark on startup for routing decisions (`true`/`false`)

### Batch Engine
- `BITNET_MAX_BATCH_SIZE`: Maximum batch size (default: `32`)
- `BITNET_BATCH_TIMEOUT_MS`: Batch collection timeout in milliseconds
- `BITNET_MAX_CONCURRENT_BATCHES`: Maximum concurrent batch executions
- `BITNET_ADAPTIVE_BATCHING`: Enable adaptive batch sizing (`true`/`false`)
- `BITNET_QUANTIZATION_AWARE`: Enable quantization-aware batching (`true`/`false`)

### Concurrency and Rate Limiting
- `BITNET_MAX_CONCURRENT_REQUESTS`: Maximum concurrent requests
- `BITNET_MAX_REQUESTS_PER_SECOND`: Rate limit (requests/second)
- `BITNET_MAX_REQUESTS_PER_MINUTE`: Rate limit (requests/minute)
- `BITNET_BACKPRESSURE_THRESHOLD`: Backpressure activation threshold
- `BITNET_CIRCUIT_BREAKER_ENABLED`: Enable circuit breaker (`true`/`false`)
- `BITNET_PER_IP_RATE_LIMIT`: Per-IP rate limit

### Security
- `BITNET_JWT_SECRET`: JWT signing secret for authentication
- `BITNET_REQUIRE_AUTHENTICATION`: Require authentication (`true`/`false`)
- `BITNET_MAX_PROMPT_LENGTH`: Maximum prompt length (characters)
- `BITNET_MAX_TOKENS_PER_REQUEST`: Maximum tokens per request
- `BITNET_ALLOWED_ORIGINS`: Allowed CORS origins (comma-separated)
- `BITNET_BLOCKED_IPS`: Blocked IP addresses (comma-separated)
- `BITNET_INPUT_SANITIZATION`: Enable input sanitization (`true`/`false`)
- `BITNET_CONTENT_FILTERING`: Enable content filtering (`true`/`false`)
- `BITNET_ALLOWED_MODEL_DIRECTORIES`: Allowed model directories (comma-separated). When unset or empty, server model-path validation allows relative model names but rejects absolute paths; configure explicit directories before accepting absolute model paths from requests.

### Observability
- `BITNET_PROMETHEUS_ENABLED`: Enable Prometheus metrics (`true`/`false`)
- `BITNET_OPENTELEMETRY_ENABLED`: Enable OpenTelemetry tracing (`true`/`false`)
- `BITNET_OTLP_ENDPOINT`: OTLP collector endpoint URL

## Testing Variables

- `BITNET_SKIP_SLOW_TESTS`: Skip slow tests — set to `1` (used in CI Core)
- `BITNET_RUN_SLOW_TESTS`: Explicitly opt-in to slow tests
- `BITNET_FAST_TESTS`: Run only fast tests
- `BITNET_RUN_E2E`: Enable end-to-end tests
- `BITNET_FORCE_GPU_TESTS`: Force GPU tests even without hardware
- `BITNET_GENERATE_FIXTURES`: Generate test fixtures (`1` to enable)
- `BITNET_QUIET_BACKEND`: Suppress backend output in tests
- `BITNET_TEST_ENV`: Test environment identifier
- `BITNET_TEST_SCENARIO`: Test scenario selector
- `BITNET_MOCK_DETECTION_THRESHOLD`: Threshold for mock computation detection
- `BITNET_VALIDATION_LEVEL`: Validation strictness level
- `BITNET_VALIDATION_TOLERANCE`: Numerical validation tolerance

### Cross-Validation
- `BITNET_CROSSVAL_ENABLED`: Enable cross-validation against C++ reference
- `BITNET_CROSSVAL_WEIGHTS`: Path to cross-validation weights
- `BITNET_CPP_PATH`: Path to bitnet.cpp binary

## GPU Development Variables

For GPU development, testing, and mock scenarios:

```bash
# Test GPU backend detection
cargo test --no-default-features --features cpu -p bitnet-kernels --no-default-features test_gpu_info_summary

# Mock GPU scenarios for testing
BITNET_GPU_FAKE="cuda" cargo test --no-default-features --features cpu -p bitnet-kernels test_gpu_info_mocked_scenarios
BITNET_GPU_FAKE="metal" cargo run -p xtask -- download-model --dry-run
BITNET_GPU_FAKE="cuda,rocm" cargo test --no-default-features -p bitnet-kernels --features gpu
```

## Determinism Configuration

For reproducible builds and testing:

```bash
# Force stable runs with strict mode (no mock fallbacks)
export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42

# Single-threaded CPU determinism for testing
export RAYON_NUM_THREADS=1

# Production deterministic inference with real quantization
BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo run -p xtask -- infer --model model.gguf --prompt "Test"

# Local performance builds (not CI)
export RUSTFLAGS="-C target-cpu=native"
```

## Strict Testing Examples

### Basic Strict Mode Usage (Issue #261)
```bash
# Primary strict mode - prevents ALL mock inference fallbacks
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-inference --features cpu
BITNET_STRICT_MODE=1 cargo run -p xtask -- infer --model model.gguf --prompt "Test"

# Production inference with strict mode (SIMD-optimised CPU, GPU-accelerated alpha)
BITNET_STRICT_MODE=1 cargo run -p xtask -- infer \
  --model models/bitnet-model.gguf \
  --prompt "Explain quantum computing" \
  --deterministic
```

### Granular Strict Mode Controls (Issue #261)
```bash
# Fail immediately on mock detection
BITNET_STRICT_FAIL_ON_MOCK=1 \
cargo test -p bitnet-inference --no-default-features --features cpu test_inference_real_computation

# Require real quantization kernels (I2S/TL1/TL2)
BITNET_STRICT_REQUIRE_QUANTIZATION=1 \
cargo test -p bitnet-quantization --no-default-features --features cpu test_quantization_kernel_integration

# Validate performance metrics for realistic values
BITNET_STRICT_VALIDATE_PERFORMANCE=1 \
cargo run -p xtask -- benchmark --model model.gguf --tokens 128

# CI enhanced strict mode (comprehensive validation)
CI=1 BITNET_CI_ENHANCED_STRICT=1 BITNET_STRICT_MODE=1 \
cargo test --workspace --no-default-features --features cpu
```

### Performance Testing with Strict Mode
```bash
# CPU baseline with real quantization (no mocks)
BITNET_STRICT_MODE=1 \
cargo bench --no-default-features --features cpu -p bitnet-quantization --bench simd_comparison

# GPU performance with strict hardware validation
BITNET_STRICT_NO_FAKE_GPU=1 \
BITNET_STRICT_MODE=1 \
cargo bench -p bitnet-kernels --bench mixed_precision_bench --features gpu

# Realistic CPU performance baselines (Issue #261 - AC7)
# Expected: SIMD-optimised throughput (hardware-dependent)
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
cargo run -p xtask -- benchmark --features cpu --quantization i2s

# Realistic GPU performance baselines (Issue #261 - AC8)
# Expected: GPU-accelerated (alpha), GPU utilization >80%
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
cargo run -p xtask -- benchmark --features gpu --quantization i2s
```

### Strict Integration Testing
```bash
# Strict tokenizer tests (no mock fallbacks)
BITNET_STRICT_TOKENIZERS=1 \
BITNET_STRICT_MODE=1 \
cargo test --features cpu -p bitnet-tokenizers -- --quiet

# Strict GPU kernel tests (real hardware only)
BITNET_STRICT_NO_FAKE_GPU=1 \
BITNET_STRICT_MODE=1 \
cargo test --no-default-features -p bitnet-kernels --features gpu -- --quiet

# Combined strict testing for production validation
BITNET_STRICT_MODE=1 \
BITNET_STRICT_TOKENIZERS=1 \
BITNET_STRICT_NO_FAKE_GPU=1 \
scripts/verify-tests.sh

# Cross-validation with strict mode (Issue #261 - AC9)
# Validates quantization accuracy: I2S ≥99.8%, TL1/TL2 ≥99.6% vs FP32
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
cargo run -p xtask -- crossval
```

## System Metrics Variables

For server monitoring and system metrics collection:

```bash
# Test system metrics collection in server
cargo test --no-default-features -p bitnet-server --features prometheus test_system_metrics_collection

# Run server with system metrics enabled
cargo run -p bitnet-server --features prometheus --bin server &
curl http://localhost:8080/metrics | grep "system_"

# Test memory tracking integration with system metrics
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_memory_tracking_comprehensive

# Validate system metrics in monitoring stack
cd monitoring && docker-compose up -d
curl http://localhost:9090/api/v1/query?query=system_cpu_usage_percent
```

For more information on specific topics, see:
- [GPU Development Guide](development/gpu-development.md) - GPU-specific environment variables and testing
- [Test Suite Guide](development/test-suite.md) - Testing configuration and variables
- [Performance Benchmarking Guide](performance-benchmarking.md) - Performance testing variables
