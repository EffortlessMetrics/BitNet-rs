# BitNet.rs xtask Crate - File Inventory & Code Locations

## Key Files

### Core Command Implementation

#### 1. `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (4700+ lines)
**Primary xtask implementation**

**Key Sections**:
- Lines 1-39: Imports and module declarations
- Lines 40-100: CrossValReport struct definition
- Lines 185-189: CLI argument parser struct (Clap)
- Lines 192-739: Cmd enum with all subcommands
- Lines 751-760: main() entry point
- Lines 762-788: classify_exit() error code mapping
- Lines 801-970: real_main() - command dispatcher
- Lines 2387-2500: fetch_cpp_cmd() - C++ reference setup
- Lines 2504-2534: apply_cpp_env() - Platform-specific library paths
- Lines 2604-2856: crossval_cmd() - Deterministic cross-validation
- Lines 2857-2976: crossval_per_token_cmd() - Per-token logits divergence detection
- Lines 2978-3050+: full_crossval_cmd() - One-command workflow
- Lines 4552-4676: verify_receipt_cmd() - Receipt validation with 8 gates
- Lines 1100-1700+: download_model logic (resumable, atomic, verified)

**Command Dispatch** (real_main, lines 801-970):
```rust
fn real_main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::DownloadModel { ... } => download_model_cmd(...),
        Cmd::Tokenizer { ... } => download_tokenizer_cmd(...),
        Cmd::FetchCpp { ... } => fetch_cpp_cmd(...),
        Cmd::Crossval { ... } => crossval_cmd(...),
        Cmd::FullCrossval { ... } => full_crossval_cmd(...),
        Cmd::CrossvalPerToken { ... } => crossval_per_token_cmd(...),
        Cmd::Benchmark { ... } => benchmark_cmd(...),
        Cmd::VerifyReceipt { ... } => verify_receipt_cmd(...),
        ...
    }
}
```

---

### Supporting Modules

#### 2. `/home/steven/code/Rust/BitNet-rs/xtask/src/ffi.rs`
```rust
//! FFI Build Hygiene (Issue #469 AC6)
//! Re-exports from xtask-build-helper

pub use xtask_build_helper::*;
```

**Purpose**: FFI build system hygiene consolidation, delegating to separate crate to avoid cyclic deps

---

#### 3. `/home/steven/code/Rust/BitNet-rs/xtask/src/gates.rs`
```rust
pub fn mapper_gate(model: PathBuf) -> Result<i32> {
    // Dry-run tensor name mapping validation
    // Reads GGUF header, calls weight_mapper::dry_run_remap_names()
    // Emits JSON result
}
```

**Lines 1-45**: Mapper gate implementation
- Validates all tensor names can be mapped
- Reports unmapped count + samples
- CI integration point for tensor name hygiene

---

#### 4. `/home/steven/code/Rust/BitNet-rs/xtask/src/tokenizers.rs`
**Tokenizer download and management**

---

#### 5. `/home/steven/code/Rust/BitNet-rs/xtask/src/lib.rs`
```rust
pub mod ffi;  // Re-exported for shared library access
```

---

### Configuration & Build

#### 6. `/home/steven/code/Rust/BitNet-rs/xtask/Cargo.toml`
```toml
[package]
name = "xtask"
version = "0.1.0"
edition = "2024"

[dependencies]
# Core
bitnet-models = { path = "../crates/bitnet-models", default-features = false, features = ["cpu"] }
bitnet-kernels = { path = "../crates/bitnet-kernels", default-features = false, features = ["cpu"] }
bitnet-tokenizers = { path = "../crates/bitnet-tokenizers", default-features = false, features = ["spm"] }

# Optional (feature-gated)
bitnet-inference = { path = "../crates/bitnet-inference", default-features = false, optional = true }
bitnet = { path = "..", default-features = false, features = ["cpu"], optional = true }
bitnet-crossval = { path = "../crossval", default-features = false, features = ["crossval"], optional = true }
bitnet-sys = { path = "../crates/bitnet-sys", default-features = false, features = ["ffi"], optional = true }

# Standard libraries
clap = { version = "4.5.49", features = ["derive"] }
serde = { version = "1.0.228", features = ["derive"] }
serde_json = "1.0.145"
reqwest = { version = "0.12.24", features = ["blocking", "rustls-tls"] }
sha2 = "0.10.9"
indicatif = "0.18.0"  # Progress bars

[features]
default = ["gpu"]
inference = [
    "dep:bitnet-inference",
    "dep:bitnet",
    "dep:tokio",
    "dep:futures",
    "dep:bitnet-crossval",
    "dep:bitnet-sys"
]
gpu = ["bitnet-kernels/gpu"]
```

---

## Supporting Scripts

### 7. `/home/steven/code/Rust/BitNet-rs/ci/fetch_bitnet_cpp.sh`
**Called by**: fetch_cpp_cmd() (main.rs:2452)
**Purpose**: Clone and build Microsoft BitNet C++ reference

```bash
#!/usr/bin/env bash
# Fetches and builds the Microsoft BitNet C++ implementation

CACHE_DIR="${BITNET_CPP_DIR:-$HOME/.cache/bitnet_cpp}"
REPO_URL="https://github.com/microsoft/BitNet.git"
DEFAULT_REV="main"

# Responsibilities:
# 1. Clone repo if not exists
# 2. Checkout specified branch/tag
# 3. Configure CMake with backend flags
# 4. Build with -DBUILD_SHARED_LIBS=OFF -DLLAMA_STATIC=ON
# 5. Create ~/...cache/bitnet_cpp/build/
```

**Integration**:
- Called from xtask with `run()` function
- Passes backend, cmake-flags, force, clean, tag, repo
- Output: Libraries in ~/.cache/bitnet_cpp/build/

---

### 8. `/home/steven/code/Rust/BitNet-rs/scripts/run_crossval_sweep.sh` (100+ lines)
**Purpose**: Multi-scenario deterministic cross-validation with tracing

**Usage**:
```bash
./scripts/run_crossval_sweep.sh model.gguf tokenizer.json /tmp/crossval
```

**Scenarios**:
- 1 token
- 2 tokens
- 4 tokens

**Output**:
```
/tmp/crossval/
├── scenario1/
│   ├── rs-traces/      (90+ JSONL files)
│   ├── rs-output.txt
│   ├── cpp-output.txt
│   ├── logits-comparison.json
│   └── report.txt
├── scenario2/, scenario3/
└── summary.md          (final divergence report)
```

---

### 9. `/home/steven/code/Rust/BitNet-rs/scripts/trace_diff.py`
**Purpose**: Compare trace files from Rust vs C++ implementations

**Usage**:
```bash
python3 scripts/trace_diff.py /tmp/rust_traces /tmp/cpp_traces
```

**Implementation** (143 lines):
- Lines 15-53: load_traces() - Load JSONL trace files into dict keyed by (seq, layer, stage)
- Lines 55-116: compare_traces() - Compare two trace sets
- Lines 118-142: main() - CLI entry point

**Comparison Logic**:
1. Load all .trace and .jsonl files
2. For each tracepoint (seq, layer, stage):
   - Check both exist in Rust and C++
   - Compare shape (must match)
   - Compare dtype (must match)
   - Compare blake3 hash (detects value divergence)
   - Report rms and num_elements for diverged positions

**Output**:
- `✓ All tracepoints match` (exit 0)
- `✗ First divergence at seq=X, layer=Y, stage=Z` (exit 1)

---

## Key Functions by Location

### Download & Verification (main.rs)

| Function | Lines | Purpose |
|----------|-------|---------|
| download_model_cmd() | 1100-1750+ | Resumable HTTP download with SHA256 |
| download_tokenizer_cmd() | 1750+ | LLaMA-3 tokenizer fetching |
| sha256_file() | ? | Compute SHA256 of file |
| exp_backoff_ms() | ? | Exponential backoff for retries |

### Cross-Validation (main.rs)

| Function | Lines | Purpose |
|----------|-------|---------|
| fetch_cpp_cmd() | 2387-2500 | Setup C++ reference via ci/fetch_bitnet_cpp.sh |
| apply_cpp_env() | 2504-2540 | Platform-specific library path setup |
| crossval_cmd() | 2604-2856 | Run deterministic cross-validation |
| crossval_per_token_cmd() | 2857-2976 | Per-position logits divergence detection |
| full_crossval_cmd() | 2978-3050+ | Full workflow: download → fetch-cpp → test |
| validate_rust_model_loading() | ? | Check Rust can load GGUF |
| cpp_header_preflight() | ? | Check C++ can parse GGUF header |
| apply_deterministic_env() | ? | Set RAYON_NUM_THREADS=1, BITNET_DETERMINISTIC=1, BITNET_SEED=42 |

### Benchmarking & Validation (main.rs)

| Function | Lines | Purpose |
|----------|-------|---------|
| benchmark_cmd() | 865-890+ | Run inference benchmark, write receipt |
| verify_receipt_cmd() | 4552-4676 | Validate receipt JSON (schema, compute_path, kernels) |
| is_gpu_kernel_id() | ? | Pattern match for GPU kernel names |
| validate_cpu_backend_kernels() | ? | Ensure CPU backend uses quantized kernels |
| verify_quantization_claims() | ? | AC6: Verify claims match kernels |
| preflight_cmd() | ? | GPU detection + capabilities check |
| gpu_preflight_cmd() | ? | Enhanced GPU checking |

### Gates Module (gates.rs)

| Function | Lines | Purpose |
|----------|-------|---------|
| mapper_gate() | 20-44 | Dry-run tensor name mapping (CI gate) |

---

## Data Structures & Types

### CrossValReport (main.rs:62-101)
```rust
struct CrossValReport {
    model: String,
    rust_ok: bool,
    cpp_header_ok: bool,
    cpp_full_ok: bool,
    xfail: bool,
    notes: String,
    timestamp: String,
    platform: String,
    gguf_version_detected: Option<u32>,
    n_kv: Option<u64>,
    n_tensors: Option<u64>,
    data_offset: Option<u64>,
    file_size: Option<u64>,
}
```

### CLI Arguments (main.rs:192-739)
```rust
#[derive(Subcommand)]
enum Cmd {
    DownloadModel { id, file, out, sha256, force, rev, no_progress, verbose, base_url, json, retries, timeout },
    Tokenizer { out, force, force_mirror, verbose },
    FetchCpp { tag, force, clean, backend, cmake_flags, repo },
    Crossval { model, cpp_dir, release, dry_run, extra },
    FullCrossval { force, tag, backend, cmake_flags, repo },
    CrossvalPerToken { model, tokenizer, prompt, max_tokens, cos_tol, format },
    SetupCrossval,
    Benchmark { model, tokenizer, tokens, prompt, gpu, allow_mock, no_output, json, warmup_tokens },
    VerifyReceipt { path, require_gpu_kernels },
    GenFixtures { size, output },
    GenMiniGguf { output, version },
    CleanCache,
    CheckFeatures,
    Gate { which: GateWhich },
    CompareMetrics { baseline, current, ppl_max, latency_p95_max, tok_s_min },
    DetectBreaking { baseline, current, format },
    VendorGgml { commit, force, output },
    FetchModels { lock },
    Preflight,
    GpuPreflight { require, format },
}
```

---

## Error Handling

### Exit Codes (main.rs:762-788)
```rust
EXIT_SUCCESS = 0
EXIT_USAGE = 1
EXIT_INTERRUPTED = 3
EXIT_HASH_MISMATCH = 6
EXIT_NO_SPACE = 7
EXIT_VERIFICATION_FAILED = 8
EXIT_NETWORK = 4
EXIT_RATE_LIMIT = 5
EXIT_AUTH = 2
```

### Error Classification
- Reqwest HTTP errors → EXIT_NETWORK, EXIT_RATE_LIMIT, EXIT_AUTH
- "not enough disk" / "insufficient disk space" → EXIT_NO_SPACE
- "sha" && "mismatch" → EXIT_HASH_MISMATCH
- "interrupted" → EXIT_INTERRUPTED
- "verification failed" → EXIT_VERIFICATION_FAILED

---

## Environment Variable Integration

### Set by xtask

| Variable | Set By | Used For |
|----------|--------|----------|
| BITNET_CPP_DIR | crossval_cmd() | Point to C++ reference |
| CROSSVAL_GGUF | crossval_cmd() | Model path for cross-validation tests |
| LD_LIBRARY_PATH | apply_cpp_env() | C++ library loading (Linux) |
| DYLD_LIBRARY_PATH | apply_cpp_env() | C++ library loading (macOS) |
| RAYON_NUM_THREADS | apply_deterministic_env() | Serial execution (=1) |
| BITNET_DETERMINISTIC | apply_deterministic_env() | Deterministic mode (=1) |
| BITNET_SEED | apply_deterministic_env() | Fixed seed (=42) |
| RUST_BACKTRACE | crossval_cmd() | Enhanced backtraces (=1) |

### Read by xtask

| Variable | Used In | Purpose |
|----------|---------|---------|
| BITNET_CPP_DIR | crossval_cmd() | Override C++ location |
| CROSSVAL_ALLOW_CPP_FAIL | crossval_cmd() | Soft-fail on C++ errors |
| BITNET_GGUF | download_model_cmd() | Model override |
| HF_TOKEN | download_model_cmd() | HuggingFace authentication |
| HTTP[S]_PROXY | download_model_cmd() | Proxy configuration |

---

## Integration Points with Other Crates

### bitnet-models
- Used for: GGUF reading, tensor name mapping
- Functions: `GgufReader::new()`, `weight_mapper::dry_run_remap_names()`

### bitnet-inference
- Used for: Logits evaluation (per-token mode)
- Feature: `inference` (optional)
- Functions: `parity::eval_logits_all_positions()`

### bitnet-crossval
- Used for: C++ parity testing, logits comparison
- Feature: `inference` (optional)
- Functions: `logits_compare::compare_per_position_logits()`

### bitnet-sys
- Used for: FFI wrapper, C++ backend interaction
- Feature: `inference` (optional)
- Functions: `wrapper::Session::load_deterministic()`, `wrapper::init_backend()`, `wrapper::free_backend()`

### bitnet-tokenizers
- Used for: Universal tokenizer loading
- Functions: `loader::load_tokenizer()`

### xtask-build-helper
- Used for: FFI build hygiene utilities
- Location: Re-exported via `xtask/src/ffi.rs`

---

## Summary of Absolute File Paths

```
/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs              (4700+ lines)
/home/steven/code/Rust/BitNet-rs/xtask/src/ffi.rs               (12 lines)
/home/steven/code/Rust/BitNet-rs/xtask/src/gates.rs             (45 lines)
/home/steven/code/Rust/BitNet-rs/xtask/src/tokenizers.rs        (TBD)
/home/steven/code/Rust/BitNet-rs/xtask/src/lib.rs               (7 lines)
/home/steven/code/Rust/BitNet-rs/xtask/Cargo.toml               (52 lines)

/home/steven/code/Rust/BitNet-rs/ci/fetch_bitnet_cpp.sh         (200+ lines)
/home/steven/code/Rust/BitNet-rs/scripts/run_crossval_sweep.sh  (500+ lines)
/home/steven/code/Rust/BitNet-rs/scripts/trace_diff.py          (143 lines)

Documentation:
/home/steven/code/Rust/BitNet-rs/docs/howto/cpp-setup.md
/home/steven/code/Rust/BitNet-rs/docs/development/build-commands.md
/home/steven/code/Rust/BitNet-rs/CLAUDE.md                       (cross-validation section)
```

---

## Commands Quick Reference

```bash
# Download model
cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf

# Setup C++ reference
cargo run -p xtask -- fetch-cpp --backend cpu

# Run cross-validation suite
cargo run -p xtask -- crossval

# Per-token logits divergence
cargo run -p xtask --features inference -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --cos-tol 0.999

# One-command full workflow
cargo run -p xtask -- full-crossval --backend cpu

# Benchmark and verify
cargo run -p xtask -- benchmark --model models/model.gguf --tokens 128 --json ci/inference.json
cargo run -p xtask -- verify-receipt --path ci/inference.json

# Trace comparison
python3 scripts/trace_diff.py /tmp/rust_traces /tmp/cpp_traces

# Comprehensive scenario sweep
./scripts/run_crossval_sweep.sh models/model.gguf models/tokenizer.json /tmp/crossval
```

