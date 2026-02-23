# XTask Architecture Exploration Report

## Executive Summary

The `xtask` crate is a comprehensive task automation and development infrastructure system for the BitNet.rs project. It serves as the bridge between user-facing CLI commands and the underlying libraries, with sophisticated feature gating to control FFI (Foreign Function Interface) dependencies. The architecture is designed to minimize FFI coupling while providing optional cross-validation capabilities.

**Key Finding**: FFI-dependent code (bitnet-crossval, bitnet-sys) is **lazily imported** and feature-gated, not eagerly loaded at xtask's top level. This prevents compilation failures when FFI is not available.

---

## 1. Feature Flag Architecture

### 1.1 xtask/Cargo.toml Feature Configuration

```toml
[features]
default = ["gpu"]
inference = ["dep:bitnet-inference", "dep:bitnet", "dep:tokio", "dep:futures", "dep:bitnet-crossval", "dep:bitnet-sys"]
crossval = ["dep:bitnet-crossval"]
ffi = ["dep:bitnet-sys"]
crossval-all = ["inference", "crossval", "ffi"]
gpu = ["bitnet-kernels/gpu"]
```

**Feature Dependency Graph**:
```
xtask features:
├── default → gpu (optional GPU support)
├── inference → bitnet-inference, bitnet, tokio, futures, bitnet-crossval, bitnet-sys
│   └── Enables: crossval-per-token command, run_inference_internal function
├── crossval → bitnet-crossval
│   └── Enables: crossval command and testing
├── ffi → bitnet-sys
│   └── Enables: FFI calls in crossval-per-token
└── crossval-all → inference + crossval + ffi
    └── Enables: ALL cross-validation features (recommended for CI)
```

### 1.2 Feature Usage in Code

**Commands Requiring Features**:
- `crossval-per-token` → requires `#[cfg(feature = "inference")]` (line 404)
- `crossval` → dynamically works with/without FFI
- `benchmark`, `infer` → optional `inference` feature
- `verify-receipt` → always available (no feature gate)
- `setup-cpp-auto`, `trace-diff` → always available (no feature gate)

---

## 2. Top-Level Module Organization (xtask/src/main.rs)

### 2.1 Module Declarations

```rust
mod cpp_setup_auto;  // C++ reference setup (always available)
pub mod ffi;         // Re-exports from xtask-build-helper
mod gates;           // CI gate implementations
mod tokenizers;      // Tokenizer utilities
mod trace_diff;      // Trace comparison wrapper
```

**Note**: `ffi` module is unconditional and re-exports from `xtask-build-helper`, providing build infrastructure helpers. This is NOT the bitnet-sys FFI.

### 2.2 Key Imports (Lines 1-40)

```rust
use anyhow::{Context, Result, anyhow, bail};
use bitnet_common::Device;
use bitnet_kernels::gpu_utils::get_gpu_info;
use clap::{Parser, Subcommand};
use once_cell::sync::Lazy;
use regex::Regex;
use std::process::Command;
use walkdir::WalkDir;

// Module declarations
mod cpp_setup_auto;
pub mod ffi;
mod gates;
mod tokenizers;
mod trace_diff;
```

**Critical Design Pattern**: No top-level imports of:
- `bitnet_sys` (conditionally used inside functions)
- `bitnet_crossval` (conditionally used inside functions)
- `bitnet_inference` (conditionally used inside functions when `inference` feature enabled)

This prevents compilation failures when these crates are not available.

---

## 3. FFI Integration Points

### 3.1 crossval_per_token_cmd (Lines 2901-3020)

**Location**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (lines 2901-3020)

**Feature Gate**: `#[cfg(feature = "inference")]` (line 2900)

**FFI Integration**:
```rust
#[cfg(feature = "inference")]
fn crossval_per_token_cmd(
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    _max_tokens: usize,
    cos_tol: f32,
    format: &str,
) -> Result<()> {
    use bitnet_crossval::logits_compare::compare_per_position_logits;
    use bitnet_inference::parity::eval_logits_all_positions;

    // ... tokenization setup ...

    // Line 2944: Check C++ FFI availability
    if !bitnet_sys::is_available() {
        anyhow::bail!(
            "C++ FFI not available. Compile with --features crossval or set BITNET_CPP_DIR"
        );
    }

    // Lines 2951-2954: Initialize C++ backend via FFI
    bitnet_sys::wrapper::init_backend();
    let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
    let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;

    // Lines 2957-2963: Use C++ FFI for inference
    let cpp_tokens = cpp_session.tokenize(prompt)?;
    cpp_session.context.eval(&cpp_tokens, 0)?;
    let cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())?;

    // Line 2974: Compare logits
    let divergence = compare_per_position_logits(&rust_logits, &cpp_logits);
    
    // Output results (JSON or text)
}
```

**FFI Imports Used**:
1. `bitnet_sys::is_available()` - Runtime FFI availability check
2. `bitnet_sys::wrapper::init_backend()` - Initialize C++ backend
3. `bitnet_sys::wrapper::free_backend()` - Cleanup FFI resources
4. `bitnet_sys::wrapper::Session::load_deterministic()` - Load model via C++ FFI
5. `bitnet_crossval::logits_compare::compare_per_position_logits()` - Compare inference results

**RAII Guard Pattern** (Line 2952):
```rust
let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
```
Ensures FFI backend cleanup even if errors occur.

### 3.2 crossval_cmd (Lines 2648-2830)

**Location**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs`

**Key Points**:
- Does NOT directly use bitnet_sys or bitnet_crossval
- Spawns subprocess: `cargo test -p bitnet-crossval --features crossval` (line 2745)
- Delegates FFI usage to bitnet-crossval tests
- Sets up environment variables for subprocess:
  - `BITNET_CPP_DIR` - C++ reference location
  - `CROSSVAL_GGUF` - Model path
  - `RAYON_NUM_THREADS=1` - Deterministic execution

**Command Dispatch** (Line 2744-2745):
```rust
let mut cmd = Command::new("cargo");
cmd.arg("test").args(["-p", "bitnet-crossval", "--features", "crossval"]);
```

### 3.3 C++ Reference Setup (cpp_setup_auto.rs)

**Location**: `/home/steven/code/Rust/BitNet-rs/xtask/src/cpp_setup_auto.rs`

**Purpose**: One-command bootstrap for C++ reference (no FFI calls)

**Workflow**:
1. Resolve `BITNET_CPP_DIR` (default: `~/.cache/bitnet_cpp`)
2. If not present, run `cargo run -p xtask -- fetch-cpp` (spawned as subprocess)
3. Verify build directory exists
4. Emit platform-specific dynamic loader exports

**Shell Exports Emitted**:
- Linux: `LD_LIBRARY_PATH={build}/lib:$LD_LIBRARY_PATH`
- macOS: `DYLD_LIBRARY_PATH={build}/lib:$DYLD_LIBRARY_PATH`
- Windows: `PATH={build};$PATH`

**Key Design**: No FFI calls in this module - just environment setup for subprocess execution.

### 3.4 Trace Diff (trace_diff.rs)

**Location**: `/home/steven/code/Rust/BitNet-rs/xtask/src/trace_diff.rs`

**Purpose**: Compare Rust vs C++ traces using Python

**Workflow**:
1. Validate trace directories exist
2. Verify `scripts/trace_diff.py` exists
3. Execute Python script via subprocess: `python3 scripts/trace_diff.py /rust/traces /cpp/traces`
4. Propagate exit code

**Key Design**: Delegates to Python script, not direct FFI.

### 3.5 Benchmark & Inference Commands (Lines 3355-5520)

**Location**: Multiple functions in main.rs

**Feature Gate**: `#[cfg(feature = "inference")]` with fallback

**FFI Usage Pattern**:
```rust
#[cfg(feature = "inference")]
{
    use bitnet::prelude::*;
    use bitnet_inference::*;
    // Real inference code
}

#[cfg(not(feature = "inference"))]
{
    // Mock/fallback code
}
```

**Key Functions**:
1. `run_inference_internal()` (lines 5302-5475) - Optional inference with GPU support
2. `select_device()` (lines 1066-1081) - Device selection with loud fallback
3. `benchmark_cmd()` (lines 3355-3620) - Performance measurement and receipt generation

---

## 4. FFI Dependency Delivery

### 4.1 bitnet-sys Cargo.toml

**Dependency Declaration** (xtask/Cargo.toml line 40):
```toml
bitnet-sys = { path = "../crates/bitnet-sys", default-features = false, features = ["ffi"], optional = true }
```

**How It's Enabled**:
- Direct: Use `--features ffi` when building xtask
- Automatic: Use `--features inference` (pulls in `dep:bitnet-sys`)
- Automatic: Use `--features crossval-all` (pulls in all FFI features)

### 4.2 bitnet-crossval Cargo.toml

**Dependency Declaration** (xtask/Cargo.toml line 39):
```toml
bitnet-crossval = { path = "../crossval", default-features = false, features = ["crossval"], optional = true }
```

**How It's Enabled**:
- Direct: Use `--features crossval` when building xtask
- Automatic: Use `--features inference` (pulls in `dep:bitnet-crossval`)

### 4.3 Lazy Static Usage

**Located**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (line 10)

```rust
use once_cell::sync::Lazy;
```

**Current Usage**: Minimal - only used for static patterns in regex and settings (not shown in this excerpt).

**NOT used for**:
- FFI initialization (uses explicit guards instead)
- Bitnet-sys loading (imported locally when needed)
- Bitnet-inference (imported locally when needed)

---

## 5. Command Dispatching Architecture

### 5.1 Cmd Enum (Lines 187-774)

All commands are declared as enum variants in the `Cmd` enum, with conditional compilation for feature-gated commands:

```rust
#[cfg(feature = "inference")]
#[command(name = "crossval-per-token")]
CrossvalPerToken {
    #[arg(long)]
    model: PathBuf,
    // ... more fields ...
},
```

### 5.2 real_main() Dispatch (Lines 836-1001)

Command dispatching logic:
```rust
fn real_main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::DownloadModel { /* args */ } => download_model_cmd(/* args */),
        Cmd::Tokenizer { /* args */ } => /* tokenizer logic */,
        Cmd::SetupCppAuto { emit } => {
            let emit_format = cpp_setup_auto::Emit::from(&emit);
            cpp_setup_auto::run(emit_format)?;
            Ok(())
        }
        Cmd::TraceDiff { rs_dir, cpp_dir } => {
            trace_diff::run(&rs_dir, &cpp_dir)?;
            Ok(())
        }
        Cmd::Crossval { /* args */ } => crossval_cmd(/* args */),
        
        #[cfg(feature = "inference")]
        Cmd::CrossvalPerToken { /* args */ } => {
            crossval_per_token_cmd(&model, &tokenizer, &prompt, max_tokens, cos_tol, &format)?;
            Ok(())
        }
        
        // ... more commands ...
    }
}
```

---

## 6. Receipt Verification System

### 6.1 verify_receipt_cmd (Lines 4596-4720)

**Purpose**: Validate inference receipts against strict quality gates

**Gates Implemented**:
1. Schema version: Must be "1.0.0" or "1.0"
2. Compute path: Must be "real" (not "mock")
3. Kernels array: Non-empty, all strings, no empty IDs
4. Kernel ID hygiene: Length ≤ 128, count ≤ 10,000
5. GPU kernel validation: Auto-enforce for `backend="cuda"`
6. Quantization claims verification (AC6)

**Exit Codes**:
- 0: Receipt valid
- Non-zero: Receipt invalid (with detailed error message)

**Key Code Section** (Lines 4669-4694):
```rust
// GPU kernel validation (auto-enforce for CUDA backend or if explicitly requested)
let require_gpu = require_gpu_kernels || must_require_gpu;
if require_gpu {
    let has_gpu_kernel = kernel_ids.iter().any(|id| is_gpu_kernel_id(id));
    if !has_gpu_kernel {
        bail!(
            "GPU kernel verification required ({}) but no GPU kernels found.\n\
             Expected (examples): {}\n\
             Actual kernels: {:?}\n\n\
             This likely indicates silent CPU fallback. Verify:\n\
             1) GPU build: cargo build --features gpu\n\
             2) CUDA runtime: nvidia-smi\n\
             3) Device selection: Device::Cuda(0) in inference",
            reason,
            GPU_KERNEL_EXAMPLES.join(", "),
            kernels
        );
    }
}
```

---

## 7. Execution Flow: How FFI Emerges

### 7.1 Scenario: User Runs crossval-per-token

```bash
cargo run -p xtask --features inference -- \
  crossval-per-token \
    --model models/model.gguf \
    --tokenizer models/tokenizer.json \
    --prompt "Test" \
    --max-tokens 4
```

**Execution Flow**:
1. main() → Cli::parse() → real_main()
2. Match Cmd::CrossvalPerToken → crossval_per_token_cmd()
3. Inside crossval_per_token_cmd():
   - Line 2909-2910: Local imports of bitnet_crossval and bitnet_inference
   - Line 2944: Call bitnet_sys::is_available() → check FFI availability
   - If available: Lines 2951-2954 → Initialize FFI, load C++ session
   - Line 2974: Compare logits with bitnet_crossval
   - Output results

**FFI Lazy Import Guarantee**: bitnet-sys is only linked/loaded when:
- xtask built with `--features inference` or `--features ffi`
- crossval_per_token_cmd() actually executes
- bitnet_sys::is_available() returns true (FFI compiled successfully)

### 7.2 Scenario: User Runs crossval Command

```bash
cargo run -p xtask -- crossval --model models/model.gguf
```

**Execution Flow**:
1. main() → Cli::parse() → real_main()
2. Match Cmd::Crossval → crossval_cmd()
3. Inside crossval_cmd():
   - Line 2745: Spawn subprocess: `cargo test -p bitnet-crossval --features crossval`
   - Line 2758: Set BITNET_CPP_DIR, CROSSVAL_GGUF env vars
   - Line 2761: Add test runner args (--nocapture --test-threads=1)
   - Line 2775: Run subprocess and capture output
   - Lines 2783-2828: Parse output and save report

**FFI Not Loaded in xtask**: FFI usage is delegated to bitnet-crossval subprocess tests.

### 7.3 Scenario: User Runs setup-cpp-auto

```bash
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
```

**Execution Flow**:
1. main() → Cli::parse() → real_main()
2. Match Cmd::SetupCppAuto { emit } → cpp_setup_auto::run()
3. Inside cpp_setup_auto::run():
   - Line 71-74: Resolve BITNET_CPP_DIR
   - Lines 77-88: If not present, spawn subprocess: `cargo run -p xtask -- fetch-cpp`
   - Lines 92-98: Verify build directory exists
   - Line 101: Emit platform-specific shell exports to stdout
4. Parent shell evals stdout: `eval "$(...)"`
5. LD_LIBRARY_PATH/DYLD_LIBRARY_PATH/PATH set in parent shell

**FFI Not Triggered**: Pure setup, no FFI imports or calls.

---

## 8. Gatekeeping & Safety Mechanisms

### 8.1 Feature Gate Guards

**Pattern 1: Conditional Function**
```rust
#[cfg(feature = "inference")]
fn crossval_per_token_cmd(...) { ... }
```

**Pattern 2: Conditional Command Variant**
```rust
#[cfg(feature = "inference")]
#[command(name = "crossval-per-token")]
CrossvalPerToken { ... }
```

**Pattern 3: Local Import Inside Feature Block**
```rust
#[cfg(feature = "inference")]
{
    use bitnet::prelude::*;
    use bitnet_inference::*;
    // Code using inference features
}
```

### 8.2 Runtime Availability Checks

**Example** (Line 2944):
```rust
if !bitnet_sys::is_available() {
    anyhow::bail!(
        "C++ FFI not available. Compile with --features crossval or set BITNET_CPP_DIR"
    );
}
```

**Function Signature** (bitnet-sys):
```rust
pub fn is_available() -> bool {
    // Checks BITNET_CPP_DIR env var or default cache location
    // Returns true only if C++ binary found and loadable
}
```

### 8.3 Fallback Device Selection (Lines 1066-1081)

```rust
fn select_device(gpu: bool) -> (Device, &'static str) {
    if gpu {
        #[cfg(feature = "inference")]
        {
            let cuda_device = Device::Cuda(0);
            eprintln!("🚀 Using GPU (CUDA)");
            return (cuda_device, "gpu");
        }
        #[cfg(not(feature = "inference"))]
        {
            eprintln!("⚠️  GPU requested but inference feature not enabled; falling back to CPU");
        }
    }
    (Device::Cpu, "cpu")
}
```

---

## 9. Comparison with crossval/Cargo.toml

### 9.1 How bitnet-crossval Uses FFI

**Location**: `/home/steven/code/Rust/BitNet-rs/crossval/Cargo.toml`

```toml
[features]
default = []
crossval = ["dep:bindgen", "dep:cc", "dep:bitnet-sys", "bitnet-sys/ffi", "bitnet-inference/ffi", "ffi"]
ffi = ["dep:cc", "bitnet-inference/ffi"]
```

**Key Difference from xtask**: 
- bitnet-crossval has its own build scripts (build.rs)
- Directly generates FFI bindings via bindgen
- xtask delegates to bitnet-crossval for FFI usage

### 9.2 Integration Point

When xtask runs `cargo test -p bitnet-crossval --features crossval`:
1. bitnet-crossval is built with FFI enabled
2. Its build script (build.rs) generates FFI bindings
3. Tests execute with full C++ interop
4. xtask just orchestrates the subprocess

---

## 10. Python Integration (trace_diff.rs)

### 10.1 Python Script Invocation

**Code** (Lines 87-92):
```rust
let status = Command::new("python3")
    .arg(script)
    .arg(rs_dir)
    .arg(cpp_dir)
    .status()
    .context("failed to spawn python3 for trace_diff.py")?;
```

**Workflow**:
1. Verify trace directories exist (Rust side)
2. Verify trace_diff.py script exists (Rust side)
3. Spawn Python subprocess (no Rust-Python FFI)
4. Python compares Blake3 hashes of trace files
5. Propagate exit code back to caller

**No FFI Coupling**: Pure subprocess execution.

---

## 11. Current Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│ User runs: cargo run -p xtask -- [COMMAND]                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ xtask/src/main.rs                                       │ │
│ │ - No top-level imports of bitnet-sys/bitnet-crossval   │ │
│ │ - Local imports inside feature-gated functions         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  │                                                           │
│  ├── [Always Available]                                      │
│  │   ├─ download-model → HTTP client                        │
│  │   ├─ tokenizer → Tokenizer download                      │
│  │   ├─ fetch-cpp → Git clone + CMake build                │
│  │   ├─ setup-cpp-auto → Shell export generator             │
│  │   ├─ trace-diff → Python subprocess                      │
│  │   ├─ verify-receipt → JSON schema validation             │
│  │   └─ verify-receipt → Quality gates (no FFI)             │
│  │                                                           │
│  ├── [--features inference]                                 │
│  │   ├─ crossval-per-token → bitnet-sys + bitnet-crossval   │
│  │   ├─ benchmark → bitnet-inference + receipts             │
│  │   ├─ infer → bitnet-inference                            │
│  │   └─ (on error) → Mock fallback if --allow-mock          │
│  │                                                           │
│  ├── [--features crossval]                                  │
│  │   └─ crossval → Spawns bitnet-crossval tests             │
│  │        └─ bitnet-crossval tests use FFI internally       │
│  │                                                           │
│  └── [--features ffi]                                       │
│      └─ (No direct xtask commands; used internally)         │
│                                                              │
│  Subprocess-based (No FFI in xtask):                         │
│  └─ crossval → cargo test -p bitnet-crossval --features ..  │
│     └─ bitnet-crossval (subprocess) loads FFI                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 12. Known Issues & Limitations

### 12.1 Issue #469: Tokenizer Parity

**Impact**: Affects crossval-per-token when using C++ tokenizer vs Rust tokenizer

**Workaround**: Set environment variables for tokenizer path validation

### 12.2 Feature Gate Complexity

**Challenge**: Users must understand feature implications:
- `cargo build --no-default-features --features cpu` → No FFI
- `cargo build --no-default-features --features cpu,inference` → Has FFI (needs bitnet-sys)
- `cargo build --no-default-features --features cpu,crossval-all` → Full FFI

### 12.3 Subprocess Execution Overhead

**Challenge**: `crossval` command spawns `cargo test` subprocess, creating overhead:
- Recompilation of bitnet-crossval (if not cached)
- Environment variable passing complexity
- Process spawning overhead

---

## 13. Recommendations for Implementation

### 13.1 For Refactoring FFI Usage

1. **Maintain Lazy Import Pattern**: Keep bitnet-sys imports local to functions
2. **Expand Feature Coverage**: Document feature requirements clearly
3. **Add Feature Predicates**: Use consistent predicates:
   ```rust
   #[cfg(any(feature = "inference", feature = "crossval"))]
   ```

### 13.2 For Adding New FFI Commands

1. Declare command in Cmd enum with feature gate
2. Implement function with feature-gated imports
3. Add runtime availability check (bitnet_sys::is_available())
4. Use RAII guards for cleanup (scopeguard)
5. Provide clear error messages on FFI failure

### 13.3 For Testing

1. Test with `--no-default-features` (no FFI)
2. Test with `--features cpu` (minimal)
3. Test with `--features cpu,inference` (FFI available)
4. Test with `--features cpu,crossval-all` (full FFI)

### 13.4 For Documentation

1. Update CLAUDE.md with feature requirements per command
2. Add feature matrix table in docs/explanation/FEATURES.md
3. Document FFI availability checks and fallback behavior

---

## Appendix A: File Locations

| File | Lines | Purpose |
|------|-------|---------|
| xtask/Cargo.toml | 1-56 | Feature configuration, dependencies |
| xtask/src/main.rs | 1-5520 | Main command dispatch, FFI integration points |
| xtask/src/lib.rs | 1-7 | Library exports (minimal) |
| xtask/src/cpp_setup_auto.rs | 1-172 | C++ reference bootstrap |
| xtask/src/trace_diff.rs | 1-117 | Python trace comparison wrapper |
| xtask/src/gates.rs | 1-45 | CI gate implementations |
| xtask/src/ffi.rs | 1-13 | Re-exports from xtask-build-helper |
| xtask/src/tokenizers.rs | (not shown) | Tokenizer utilities |
| crossval/Cargo.toml | 1-90 | bitnet-crossval FFI configuration |

---

## Appendix B: Key Function Signatures

```rust
// FFI runtime check
pub fn is_available() -> bool { /* bitnet_sys */ }

// FFI initialization/cleanup
pub fn init_backend() { /* bitnet_sys::wrapper */ }
pub fn free_backend() { /* bitnet_sys::wrapper */ }

// FFI session management
pub struct Session {
    context: Context,
    tokenizer: /* C++ tokenizer */,
}
impl Session {
    pub fn load_deterministic(path: &str) -> Result<Self>;
    pub fn tokenize(&mut self, text: &str) -> Result<Vec<Token>>;
}
impl Context {
    pub fn eval(&mut self, tokens: &[Token], pos: usize) -> Result<()>;
    pub fn get_all_logits(&self, len: usize) -> Result<Vec<Vec<f32>>>;
}

// Cross-validation comparison
pub fn compare_per_position_logits(
    rust: &[Vec<f32>],
    cpp: &[Vec<f32>]
) -> DivergenceReport { /* bitnet_crossval */ }

// Inference with optional FFI
#[cfg(feature = "inference")]
pub async fn prefill(&mut self, ids: &[u32]) -> Result<()>;
pub async fn decode_next(&mut self) -> Result<u32>;
```

---

## Appendix C: Environment Variables Used

| Variable | Command | Purpose |
|----------|---------|---------|
| BITNET_CPP_DIR | setup-cpp-auto, crossval-per-token, crossval | C++ reference location |
| CROSSVAL_GGUF | crossval | Model path for tests |
| BITNET_DETERMINISTIC | all inference commands | Enable deterministic inference |
| BITNET_SEED | all inference commands | Reproducible randomness |
| RAYON_NUM_THREADS | crossval | Single-threaded determinism |
| HF_TOKEN | download-model, tokenizer | Hugging Face authentication |
| RUST_BACKTRACE | all commands | Error diagnostics |
| CROSSVAL_ALLOW_CPP_FAIL | crossval | Allow C++ failures for xfail models |

---

**Document Date**: 2025-10-24
**Architecture Version**: 0.1.0-qna-mvp
**Last Reviewed**: crossval per-token implementation (lines 2901-3020)
