# BitNet-rs xtask Crate - Comprehensive Exploration

## Overview

The `xtask` crate is BitNet-rs's developer automation tool built with Clap CLI framework. It provides:

1. **Model Management**: Download from HuggingFace, tokenizer management, model validation
2. **Cross-Validation**: Against Microsoft BitNet C++ reference implementation
3. **Infrastructure**: C++ FFI setup, fixture generation, benchmarking, receipt verification
4. **CI/CD**: Metrics comparison, breaking change detection, gate validation

**Location**: `/home/steven/code/Rust/BitNet-rs/xtask/src/`
**Main File**: `main.rs` (~4700 lines)
**Modules**: `ffi.rs`, `gates.rs`, `tokenizers.rs`, `lib.rs`

---

## Project Structure

### xtask/Cargo.toml Features

```toml
[features]
default = ["gpu"]
inference = [
    "dep:bitnet-inference",      # For logits comparison
    "dep:bitnet",                # Full inference engine
    "dep:tokio",                 # Async runtime
    "dep:futures",               # Future utilities
    "dep:bitnet-crossval",       # C++ parity testing
    "dep:bitnet-sys"             # FFI bindings
]
gpu = ["bitnet-kernels/gpu"]     # GPU support
```

**Key Dependencies**:
- `bitnet-models` (cpu): GGUF/SafeTensors loading
- `bitnet-kernels` (cpu): SIMD/CUDA kernels
- `bitnet-tokenizers`: Universal tokenizer loader
- `bitnet-crossval` (optional): C++ validation framework
- `bitnet-sys` (optional): FFI bindings
- `xtask-build-helper`: Shared build utilities
- Standard: `clap`, `serde_json`, `reqwest`, `sha2`, `tokio`, `indicatif`

---

## Command Structure

### Main Commands (from enum Cmd)

#### 1. **download-model** (Production-grade)
```
cargo run -p xtask -- download-model \
  --id microsoft/bitnet-b1.58-2B-4T-gguf \
  --file ggml-model-i2_s.gguf \
  --out models \
  --sha256 <hash> \
  --force
```

**Features**:
- Resumable downloads with Content-Range validation
- 429 rate-limiting with Retry-After support
- ETag/Last-Modified caching for 304 optimization
- Concurrent download protection via file locking
- SHA256 verification with automatic retry
- Disk space validation before download
- Ctrl-C graceful handling with resume support
- Progress bar with throughput metrics
- Automatic tokenizer.json download attempt
- JSON event output for CI/CD pipelines

**Implementation**: Lines 208-251 (args), 1100-1700+ (logic)

---

#### 2. **tokenizer** (LLaMA-3 specific)
```
cargo run -p xtask -- tokenizer \
  --out models \
  --force
```

**Features**:
- Official source: meta-llama/Meta-Llama-3-8B (requires HF_TOKEN)
- Mirror source: baseten/Meta-Llama-3-tokenizer (no auth)
- Vocab size verification (~128k for LLaMA-3)
- Idempotent (skips if exists unless --force)
- Retry with exponential backoff

---

#### 3. **fetch-cpp** (C++ Reference Setup)
```
cargo run -p xtask -- fetch-cpp \
  --tag main \
  --backend cpu|cuda \
  --cmake-flags "-DCMAKE_CUDA_ARCHITECTURES=80;86" \
  --force \
  --clean
```

**Implementation**: Lines 2387-2500
**What it does**:
1. Calls `ci/fetch_bitnet_cpp.sh` with parsed arguments
2. Configures static builds (`BUILD_SHARED_LIBS=OFF`)
3. Adds backend-specific CMake flags:
   - **CPU**: `-DGGML_NATIVE=ON`
   - **CUDA**: `-DGGML_CUDA=ON -DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=80;86`
4. Verifies build artifacts in `~/.cache/bitnet_cpp/build/`
5. Searches for `.so`/`.dylib`/`.a` libraries and executables

**Cache Location**: `~/.cache/bitnet_cpp` (overridable via `BITNET_CPP_DIR`)

---

#### 4. **crossval** (Deterministic Cross-Validation)
```
cargo run -p xtask -- crossval \
  --model models/model.gguf \
  --cpp-dir ~/.cache/bitnet_cpp \
  --release \
  -- --nocapture
```

**Implementation**: Lines 2604-2856
**Workflow**:
1. **Validates Rust model loading**: Checks GGUF version, n_kv, n_tensors, data_offset
2. **C++ header preflight**: Tests if C++ can parse GGUF header
3. **Runs deterministic tests**:
   - Sets `BITNET_CPP_DIR`, `CROSSVAL_GGUF` environment vars
   - Applies platform-specific library paths (via `apply_cpp_env()`)
   - Sets deterministic env: `RAYON_NUM_THREADS=1`, `BITNET_DETERMINISTIC=1`, `BITNET_SEED=42`
   - Runs: `cargo test -p bitnet-crossval --features crossval --release -- --nocapture --test-threads=1`

**Key Features**:
- Auto-discovers GGUF in `models/` if not specified
- `CROSSVAL_ALLOW_CPP_FAIL=1` for graceful degradation on C++ errors
- Platform-specific library path handling (Linux/macOS/Windows)
- Report generation at `target/crossval_report.json`

**Environment Setup** (applies_cpp_env functions):
```rust
// Linux
LD_LIBRARY_PATH=~/.cache/bitnet_cpp/build/bin:~/..../llama.cpp/src:~/..../ggml/src

// macOS
DYLD_LIBRARY_PATH=<same paths>
```

---

#### 5. **crossval-per-token** (Logits Divergence Detection)
```
cargo run -p xtask --features inference -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --cos-tol 0.999 \
  --format text|json
```

**Implementation**: Lines 2857-2976
**Requires**: `#[cfg(feature = "inference")]`
**Dependencies**: `bitnet-crossval::logits_compare`, `bitnet-inference::parity`

**What it does**:
1. Tokenizes prompt with unified tokenizer loader
2. Gets Rust logits for all positions via `eval_logits_all_positions()`
3. Gets C++ logits via FFI wrapper:
   - `bitnet_sys::wrapper::init_backend()`
   - `bitnet_sys::wrapper::Session::load_deterministic()`
   - `session.tokenize()` → `context.eval()` → `context.get_all_logits()`
4. Compares per-position using `compare_per_position_logits()`
5. Reports first divergence where cosine_sim < cos_tol threshold

**Output**:
- **Text**: Per-token cosine similarity, L2 distance, first divergence location
- **JSON**: Structured output with divergence_token, per-token metrics, status

---

#### 6. **full-crossval** (One-Command Workflow)
```
cargo run -p xtask -- full-crossval \
  --tag main \
  --backend cpu \
  --force
```

**Implementation**: Lines 2978-3050+
**Orchestrates**:
1. Download model (via `download_model_cmd()`)
2. Fetch C++ (via `fetch_cpp_cmd()`)
3. Run crossval tests (via `crossval_cmd()`)

---

#### 7. **benchmark** (Performance Measurement)
```
cargo run -p xtask -- benchmark \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --tokens 128 \
  --prompt "The capital of France is" \
  --gpu \
  --warmup-tokens 10 \
  --json ci/inference.json
```

**Features**:
- Generates warmup tokens and discards them
- Measures tokens-per-second throughput
- Writes production receipts with measured TPS and real kernel IDs
- Supports GPU acceleration via `--gpu` flag

**Output**: JSON receipt at `ci/inference.json` used by `verify-receipt`

---

#### 8. **verify-receipt** (Receipt Validation)
```
cargo run -p xtask -- verify-receipt \
  --path ci/inference.json \
  --require-gpu-kernels
```

**Implementation**: Lines 4552-4676
**Validates**:
1. **Schema**: Must be "1.0.0" or "1.0"
2. **Compute path**: Must be "real" (rejects mock inference)
3. **Kernels array**:
   - Non-empty
   - All strings
   - No empty kernel IDs
   - IDs ≤ 128 characters
   - ≤ 10,000 total entries
   - Warns on duplicates
4. **GPU Kernel Enforcement**:
   - **Auto**: `backend == "cuda"` requires GPU kernels
   - **Manual**: `--require-gpu-kernels` flag
   - Detects GPU kernels via `is_gpu_kernel_id()` pattern matching
   - Examples: `gemm_*`, `i2s_gpu_*`
5. **CPU Backend Validation**: `validate_cpu_backend_kernels()`
6. **Quantization Claims**: `verify_quantization_claims()`

---

#### 9. **gen-fixtures** (Test Data Generation)
```
cargo run -p xtask -- gen-fixtures \
  --size small|medium|tiny \
  --output crossval/fixtures/
```

**Purpose**: Creates realistic GGUF-like metadata and weight files for unit testing

---

#### 10. **preflight** (GPU Detection)
```
cargo run -p xtask -- preflight
```

**Exit codes**:
- 0: GPU backend available
- 1: No GPU backend (can continue with CPU)

---

#### 11. **gate** (CI Validation)
```
cargo run -p xtask -- gate mapper --model models/model.gguf
```

**Emits JSON for CI/CD detection**
**Example gate**: `mapper` - validates tensor name mapping (dry-run)

---

### Other Commands

- **setup-crossval**: Configure cross-validation environment
- **clean-cache**: Interactive cache cleanup
- **check-features**: Validate feature flag consistency
- **compare-metrics**: Regression detection
- **detect-breaking**: API surface change detection
- **vendor-ggml**: Download IQ2_S support from llama.cpp
- **fetch-models**: Download models from lockfile with SHA256 verification

---

## Supporting Infrastructure

### trace_diff.py (Divergence Detection)
**Location**: `scripts/trace_diff.py`
**Purpose**: Compare trace files from Rust vs C++ to find first divergence

**Usage**:
```bash
python3 scripts/trace_diff.py /tmp/rust_traces /tmp/cpp_traces
```

**Trace Format**: JSONL with per-record keys: `(seq, layer, stage)`

**Comparison**:
- Shape match
- Dtype match
- Blake3 hash comparison (detects value divergence)
- Reports: rms, num_elements for diverged positions

---

### run_crossval_sweep.sh (Comprehensive Scenario Testing)
**Location**: `scripts/run_crossval_sweep.sh`
**Purpose**: Multi-scenario deterministic testing with tracing

**Usage**:
```bash
./scripts/run_crossval_sweep.sh model.gguf tokenizer.json /tmp/crossval
```

**Runs**:
- 3 deterministic scenarios (1, 2, 4 tokens)
- Captures 90+ trace files per scenario in `rs-traces/`
- C++ parity comparison (graceful fallback to Rust-only)
- Generates scenario reports and summary.md

**Output Structure**:
```
/tmp/crossval/
├── scenario1/
│   ├── rs-traces/      (90+ files)
│   ├── rs-output.txt
│   ├── cpp-output.txt  (if C++ available)
│   ├── logits-comparison.json
│   └── report.txt
├── scenario2/, scenario3/
└── summary.md          (divergence report with recommendations)
```

---

### ci/fetch_bitnet_cpp.sh (C++ Build Script)
**Location**: `ci/fetch_bitnet_cpp.sh`
**Called by**: `fetch_cpp_cmd()` in xtask

**Responsibilities**:
1. Clone Microsoft BitNet from GitHub
2. Configure CMake with backend-specific flags
3. Build libllama.so and related libraries
4. Install to `~/.cache/bitnet_cpp/build/`

**Cache Layout**:
```
~/.cache/bitnet_cpp/
├── bitnet/                    (git clone)
├── build/                     (CMake build)
│   ├── bin/                   (executables)
│   ├── lib/                   (libraries)
│   └── 3rdparty/llama.cpp/    (submodule)
│       └── src/               (libllama.so location)
└── .git/                      (git state)
```

---

## Cross-Validation Architecture

### Data Flow

```
[GGUF Model] → Rust bitnet-inference → Rust Logits
                ↓
         compare_per_position_logits()
                ↓
     [Cosine Similarity Report]
         
[GGUF Model] → C++ llama.cpp FFI → C++ Logits
                ↓
         via bitnet-sys::wrapper
```

### Key Integration Points

1. **FFI Module** (`xtask/src/ffi.rs`):
   - Re-exports from `xtask-build-helper`
   - Handles C++ symbol resolution

2. **Crossval Crate** (`bitnet-crossval`):
   - `logits_compare::compare_per_position_logits()`
   - Per-token metrics calculation
   - Divergence detection

3. **Inference Crate** (`bitnet-inference`):
   - `parity::eval_logits_all_positions()`
   - Deterministic logits extraction
   - All-positions evaluation mode

4. **Sys Crate** (`bitnet-sys`):
   - `wrapper::Session::load_deterministic()`
   - `wrapper::init_backend()`, `free_backend()`
   - FFI tokenization and evaluation

---

## Deterministic Execution Model

### Environment Variables Applied

```bash
# From apply_deterministic_env()
RAYON_NUM_THREADS=1              # Single-threaded
BITNET_DETERMINISTIC=1           # Deterministic mode
BITNET_SEED=42                   # Fixed seed

# From deterministic config
--test-threads=1                 # Serial test execution
```

### Purpose
Ensures reproducible results when comparing Rust vs C++:
- No thread scheduling variance
- Fixed random seeds
- Serial execution order

---

## Feature Gate Dependencies

### Building with Crossval

```bash
# Minimal (fetch-cpp only)
cargo build -p xtask

# With cross-validation testing
cargo build -p xtask --features inference

# Full (all commands)
cargo build -p xtask --features inference,gpu
```

### Feature-Gated Commands

| Command | Feature | Notes |
|---------|---------|-------|
| crossval | none | Basic crossval support |
| crossval-per-token | `inference` | Requires logits comparison |
| benchmark | none | Basic benchmarking |
| preflight | none | GPU detection |
| gpu-preflight | none | Enhanced GPU checking |
| gate mapper | none | CI validation |

---

## Error Handling & Exit Codes

### classify_exit() function (lines 762-788)

```rust
EXIT_SUCCESS = 0
EXIT_USAGE = 1
EXIT_INTERRUPTED = 3          // Ctrl-C
EXIT_HASH_MISMATCH = 6        // SHA256 mismatch
EXIT_NO_SPACE = 7             // Insufficient disk
EXIT_VERIFICATION_FAILED = 8  // Receipt verification failure
EXIT_NETWORK = 4              // HTTP/network errors
EXIT_RATE_LIMIT = 5           // 429 Too Many Requests
EXIT_AUTH = 2                 // 401/403 Auth errors
```

### Error Detection Patterns

- **Network**: `reqwest::Error`, 4xx/5xx status codes
- **Disk**: "not enough disk", "insufficient disk space"
- **Hash**: "sha" && "mismatch"
- **Interruption**: "interrupted"
- **Verification**: "verification failed"

---

## Gates Module (Validation)

### mapper_gate() (lines 20-44)
**Purpose**: Dry-run tensor name mapping validation

**Logic**:
1. Reads GGUF header (tensor names only, no data)
2. Calls `weight_mapper::dry_run_remap_names()`
3. Reports unmapped tensor count
4. Emits JSON with:
   - `ok`: true if all tensors mapped
   - `unmapped_count`: number of unmapped tensors
   - `counts.n_kv`: metadata key count
   - `counts.n_tensors`: tensor count

**CI Integration**: Gate fails (exit 1) if unmapped tensors found

---

## Configuration & Caching

### Default Locations

```
Model downloads:        models/
C++ reference:          ~/.cache/bitnet_cpp/
Tokenizers:             models/tokenizer.json
Receipts:               ci/inference.json
Crossval reports:       target/crossval_report.json
Model cache:            ~/.cache/bitnet/models/<sha256>/
```

### Environment Variable Overrides

```bash
BITNET_CPP_DIR=<path>              # C++ reference location
BITNET_GGUF=<path>                 # Model override
HF_TOKEN=<token>                   # HuggingFace auth
CROSSVAL_ALLOW_CPP_FAIL=1          # Soft-fail on C++ errors
CROSSVAL_TIMEOUT_SECS=180          # Scenario timeout
CROSSVAL_GGUF=<path>               # Model for crossval tests
RUST_BACKTRACE=1                   # Enhanced backtraces
HTTP_PROXY / HTTPS_PROXY           # Proxy configuration
```

---

## Production Quality Features

### Download Resilience
- **Resumable**: Tracks partial downloads with temporary files
- **Atomic**: Renames only after successful verification
- **Fsync**: Parent directory synced for journaling
- **Metadata**: ETag/Last-Modified cached for 304 optimization
- **Retry**: Exponential backoff with configurable max attempts
- **Disk check**: Space verification before and during download

### Concurrent Safety
- **File locking**: Prevents concurrent downloads of same file
- **Lock cleanup**: RAII guard ensures cleanup on failure
- **Atomic rename**: No partial/corrupted files left behind

### Cross-Platform Support
- Linux: `LD_LIBRARY_PATH` for C++ libraries
- macOS: `DYLD_LIBRARY_PATH` for dylib support
- Windows: WSL/Git Bash warning (native support planned)

---

## Integration Points with CLAUDE.md

### Commands Referenced in CLAUDE.md

```bash
# Model management
cargo run -p xtask -- download-model
cargo run -p xtask -- tokenizer

# C++ setup
cargo xtask fetch-cpp
cargo xtask fetch-cpp --backend cuda

# Cross-validation
cargo run -p xtask -- crossval
cargo run -p xtask -- full-crossval
cargo run -p xtask --features inference -- crossval-per-token

# Benchmarking & validation
cargo run -p xtask -- benchmark --model <model.gguf> --tokens 128
cargo run -p xtask -- verify-receipt

# Quality gates
cargo xtask verify-receipt --require-gpu-kernels
cargo run -p xtask -- gate mapper --model models/model.gguf

# Preflight
cargo run -p xtask -- preflight
```

---

## Testing & Development Workflow

### Typical Developer Workflow

1. **Setup**:
   ```bash
   cargo run -p xtask -- download-model          # Download model
   cargo run -p xtask -- fetch-cpp                # Build C++ reference
   ```

2. **Development**:
   ```bash
   cargo test --workspace --no-default-features --features cpu
   ```

3. **Cross-validation**:
   ```bash
   cargo run -p xtask -- crossval                 # Full suite
   cargo run -p xtask --features inference -- \
     crossval-per-token --prompt "Test" --max-tokens 4
   ```

4. **Benchmarking**:
   ```bash
   cargo run -p xtask -- benchmark \
     --model models/model.gguf \
     --tokens 128 \
     --json ci/inference.json
   cargo run -p xtask -- verify-receipt
   ```

### CI/CD Workflow

1. **Setup** (in container or VM):
   ```bash
   cargo run -p xtask -- full-crossval --force
   ```

2. **Validation**:
   ```bash
   cargo run -p xtask -- benchmark --json ci/inference.json
   cargo run -p xtask -- verify-receipt --require-gpu-kernels
   ```

3. **Metrics**:
   ```bash
   cargo run -p xtask -- compare-metrics \
     --baseline baseline.json \
     --current ci/metrics.json
   ```

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Location** | `/home/steven/code/Rust/BitNet-rs/xtask/` |
| **Main File** | `src/main.rs` (~4700 lines) |
| **Modules** | ffi, gates, tokenizers, lib |
| **Build Tool** | Clap CLI parser |
| **Primary Purpose** | Developer automation + C++ cross-validation |
| **Key Commands** | fetch-cpp, crossval, benchmark, verify-receipt |
| **C++ Integration** | Via `~/.cache/bitnet_cpp`, FFI bindings |
| **Test Modes** | Deterministic, per-token, suite-based |
| **Report Format** | JSON (receipts, metrics, gates) |
| **Production Ready** | Yes (download resilience, error classification) |
| **Feature Gates** | `inference`, `gpu` |

