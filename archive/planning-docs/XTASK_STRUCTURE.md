# Xtask Structure & Crossval Commands: Comprehensive Analysis

## Current Repository Structure

**Working Directory**: `/home/steven/code/Rust/BitNet-rs`
**Current Branch**: `feat/comprehensive-integration-qk256-envguard-receipts-strict-avx2`

---

## 1. XTASK Command Structure

### Location
- **Main File**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (5624 lines)
- **Cargo.toml**: `/home/steven/code/Rust/BitNet-rs/xtask/Cargo.toml`

### Clap Command Enum Definition

**Location**: Lines 185-706 in `xtask/src/main.rs`

```rust
#[derive(Parser)]
#[command(name = "xtask", about = "Developer tasks for BitNet.rs")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    DownloadModel { ... },
    Tokenizer { ... },
    FetchCpp { ... },
    Crossval { ... },
    FullCrossval { ... },
    GenFixtures { ... },
    GenMiniGguf { ... },
    SetupCrossval,
    CleanCache,
    CheckFeatures,
    Gate { #[command(subcommand)] which: GateWhich },
    Benchmark { ... },
    CompareMetrics { ... },
    DetectBreaking { ... },
    VendorGgml { ... },
    Preflight,
    GpuPreflight { ... },
    GpuSmoke { ... },
    Demo { ... },
    Verify { ... },
    Infer { ... },
    BenchCompare { ... },
    VerifyReceipt { ... },
    FetchModels { ... },
}
```

### Command Dispatch Pattern

**Location**: Lines 758-909 in `main.rs`

```rust
fn real_main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::DownloadModel { ... } => download_model_cmd(...),
        Cmd::Tokenizer { ... } => { /* handle */ },
        Cmd::FetchCpp { ... } => fetch_cpp_cmd(...),
        Cmd::Crossval { model, cpp_dir, release, dry_run, extra } => {
            let model_path = match model {
                Some(p) => p,
                None => resolve_default_model()?,
            };
            crossval_cmd(&model_path, cpp_dir.as_deref(), release, &extra, dry_run)
        }
        // ... other commands
    }
}
```

---

## 2. EXISTING CROSSVAL COMMANDS

### 2.1 Crossval Command

**Definition** (Lines 306-326):
```rust
/// Run deterministic cross-validation tests against C++ implementation
///
/// Auto-discovers GGUF models in the models/ directory if not specified.
/// Requires the C++ implementation to be built first (use fetch-cpp).
Crossval {
    /// Path to GGUF model (auto-discovers if not specified)
    #[arg(long)]
    model: Option<PathBuf>,
    /// Path to C++ checkout (default: $HOME/.cache/bitnet_cpp)
    #[arg(long)]
    cpp_dir: Option<PathBuf>,
    /// Release build
    #[arg(long, default_value_t = true)]
    release: bool,
    /// Print env and cargo test command, then exit
    #[arg(long, help = "Print env and cargo test command, then exit")]
    dry_run: bool,
    /// Extra args to pass to cargo test after `--`
    #[arg(last = true)]
    extra: Vec<String>,
},
```

**Implementation** (Lines 2556-2738):
- Validates Rust can load the model
- Runs C++ header preflight check
- Sets up deterministic environment (single-threaded, seed=42)
- Executes `cargo test -p bitnet-crossval --features crossval`
- Handles C++ failures gracefully with `CROSSVAL_ALLOW_CPP_FAIL` flag
- Saves report to `target/crossval_report.json`

**Key Functions**:
- `validate_rust_model_loading()` - Lines 2742-2785: Validates GGUF loading
- `cpp_header_preflight()` - Checks C++ can parse GGUF
- `apply_cpp_env()` - Sets platform-specific C++ library paths
- `apply_deterministic_env()` - Ensures reproducible runs

### 2.2 FullCrossval Command

**Definition** (Lines 328-352):
```rust
/// Run full cross-validation workflow (download + fetch + test)
FullCrossval {
    /// Force redownload/rebuild
    #[arg(long, default_value_t = false)]
    force: bool,
    /// Branch/tag to fetch (default: main)
    #[arg(long, default_value = DEFAULT_CPP_TAG)]
    tag: String,
    /// Backend: "cpu" (default) | "cuda"
    #[arg(long, default_value = "cpu")]
    backend: String,
    /// Additional CMake flags
    #[arg(long, default_value = "")]
    cmake_flags: String,
    /// Git repository URL (default: official Microsoft BitNet)
    #[arg(long, default_value = "https://github.com/microsoft/BitNet.git")]
    repo: String,
},
```

**Pattern**: Downloads model → fetches C++ → runs crossval tests

---

## 3. MODEL LOADING PATTERN (Inference Reference)

### 3.1 run_inference_internal() Function

**Location**: Lines 5087-5308

**Signature**:
```rust
fn run_inference_internal(
    model_path: &Path,
    tokenizer_path: Option<&Path>,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    seed: u64,
    gpu: bool,
    allow_mock: bool,
    add_bos: bool,
    add_special: bool,
) -> Result<InferenceOutcome>
```

**Key Steps**:

1. **Load Tokenizer** (Lines 5173-5207):
```rust
let tokenizer: Arc<dyn bitnet_tokenizers::Tokenizer> = match tokenizer_path {
    Some(p) => {
        let tok = bitnet_tokenizers::loader::load_tokenizer(p)?;
        Arc::new(TokenizerWrapper(tok))
    }
    None if allow_mock => Arc::new(bitnet_tokenizers::MockTokenizer::new()),
    None => bail!("tokenizer required"),
};
```

2. **Create Device** (Lines 5210-5211):
```rust
let (device, _actual_device) = select_device(gpu);
```

3. **Load Model** (Lines 5213-5214):
```rust
let loader = ModelLoader::new(device);
let model = loader.load(model_path)?;
let model_arc: Arc<dyn bitnet_models::Model> = model.into();
```

4. **Create Engine** (Lines 5220-5221):
```rust
let mut engine = InferenceEngine::new(model_arc, tokenizer.clone(), device)?;
```

5. **Encode & Run Inference** (Lines 5224-5252):
```rust
let ids = tokenizer.encode(prompt, add_bos, add_special)?;
let outcome = tokio::runtime::Runtime::new()
    .ok()
    .map(|rt| rt.block_on(...))
    .unwrap_or_else(|_| futures::executor::block_on(...))?;
```

### 3.2 InferenceOutcome Structure

**Location**: Lines 5076-5082

```rust
struct InferenceOutcome {
    generated: String,
    tokens_generated: usize,
    prefill_ms: u64,
    decode_ms: u64,
    kernels: Vec<String>,
}
```

### 3.3 Benchmark Command Pattern

**Location**: Lines 3140-3450+

**Example Pattern for Reference**:
```rust
fn benchmark_cmd(
    model: &Path,
    tokenizer: Option<&Path>,
    tokens: usize,
    prompt: &str,
    gpu: bool,
    allow_mock: bool,
    no_output: bool,
    json_path: Option<&Path>,
    warmup_tokens: usize,
) -> Result<()>
{
    // Validates inputs
    let (_device, device_str) = select_device(gpu);
    
    // Runs warmup (calls run_inference_internal)
    // Runs benchmark (calls run_inference_internal)
    // Captures timing and kernels
    // Generates JSON report
    // Writes to file if requested
}
```

---

## 4. CROSSVAL INFRASTRUCTURE

### 4.1 Crossval Library Structure

**Location**: `/home/steven/code/Rust/BitNet-rs/crossval/src/`

**Key Modules**:
- `lib.rs` - Core types, `CrossvalConfig`, error types
- `logits_compare.rs` - **Per-position logits comparison** (CRITICAL)
- `comparison.rs` - Numerical comparison utilities
- `validation.rs` - Validation checks
- `fixtures.rs` - Test fixtures
- `utils.rs` - Helper utilities
- `cpp_bindings.rs` - FFI to C++ (conditional on `feature = "crossval"`)

### 4.2 logits_compare.rs Module (Critical for Sprint 1.2)

**Location**: `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs` (252 lines)

**Core Public API**:

```rust
/// Result of per-position logits comparison
pub struct LogitsDivergence {
    /// First token position where logits diverged (None if all match)
    pub first_divergence_token: Option<usize>,
    /// Cosine similarity for each token position
    pub per_token_cosine_sim: Vec<f32>,
    /// L2 distance for each token position
    pub per_token_l2_dist: Vec<f32>,
    /// Maximum absolute difference across all positions and logits
    pub max_absolute_diff: f32,
}

pub const COSINE_SIMILARITY_THRESHOLD: f32 = 1e-4;

/// Compare logits at each token position between Rust and C++
pub fn compare_per_position_logits(
    rs_logits: &[Vec<f32>],
    cpp_logits: &[Vec<f32>],
) -> LogitsDivergence
```

**Comparison Metrics**:
1. **Cosine Similarity**: Measures angle between logit vectors (0-1 scale)
2. **L2 Distance**: Euclidean distance between vectors
3. **Max Absolute Difference**: Largest single logit difference
4. **First Divergence Token**: Position where cosine similarity drops below threshold

### 4.3 Existing Per-Position Tests

**Location**: `/home/steven/code/Rust/BitNet-rs/crossval/tests/per_position_logits.rs` (295 lines)

**Feature Gate**: `#[cfg(feature = "crossval")]` and `#[cfg(feature = "integration-tests")]`

**Test Categories**:

1. **test_single_token_logits_parity()** (Lines 30-91)
   - Evaluates prompt at prefill phase
   - Compares Rust vs C++ logits for last token
   - Checks cosine similarity > 0.9999

2. **test_multi_token_generation_divergence()** (Lines 93-181)
   - Generates 5 tokens step-by-step
   - Collects logits at each position
   - Reports first divergence point
   - Uses greedy sampling (argmax)

3. **test_prefill_decode_logits_comparison()** (Lines 183-249)
   - Tests full prompt prefill
   - Tests single-token decode
   - Compares both phases for parity
   - Asserts cosine similarity > 0.9999

4. **test_logits_compare_module()** (Lines 261-294)
   - Unit test (no FFI required)
   - Tests identical logits → no divergence
   - Validates cosine similarity = 1.0, L2 distance = 0.0

**Key Helper**:
```rust
/// Get argmax token from logits
fn argmax(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}
```

---

## 5. PATTERN FOR ADDING NEW SUBCOMMANDS

### Step 1: Add to Cmd Enum

**Example from benchmark_cmd** (Lines 399-431):

```rust
#[derive(Subcommand)]
enum Cmd {
    // ... existing commands ...
    
    /// Run decode performance benchmarks
    ///
    /// Measures tokens/sec by running deterministic inference
    Benchmark {
        /// Path to GGUF model file
        #[arg(long)]
        model: PathBuf,
        /// Path to tokenizer file
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Number of tokens to generate
        #[arg(long, default_value_t = 128)]
        tokens: usize,
        /// ... more args ...
    },
}
```

### Step 2: Add to Command Dispatch

**Location**: Lines 758-909, in `real_main()`

```rust
Cmd::Benchmark {
    model,
    tokenizer,
    tokens,
    // ... more fields ...
} => benchmark_cmd(
    &model,
    tokenizer.as_deref(),
    tokens,
    // ... pass fields ...
),
```

### Step 3: Implement Command Function

```rust
fn your_command_cmd(
    model: &Path,
    tokenizer: Option<&Path>,
    // ... more args ...
) -> Result<()> {
    // Implementation
    Ok(())
}
```

---

## 6. DEPENDENCIES & IMPORTS

### xtask/Cargo.toml Key Dependencies

```toml
[dependencies]
bitnet-models = { path = "../crates/bitnet-models", features = ["cpu"] }
bitnet-kernels = { path = "../crates/bitnet-kernels", features = ["cpu"] }
bitnet-tokenizers = { path = "../crates/bitnet-tokenizers", features = ["spm"] }
bitnet-inference = { optional = true }  # For inference features
bitnet = { features = ["cpu"], optional = true }

[features]
inference = ["dep:bitnet-inference", "dep:bitnet", "dep:tokio", "dep:futures"]
gpu = ["bitnet-kernels/gpu"]
```

### Key xtask/src/main.rs Imports for Model Loading

```rust
use bitnet::prelude::*;
use bitnet_inference::eval_logits_once;
use bitnet_models::loader::ModelLoader;
use bitnet_tokenizers::loader::load_tokenizer;
use std::sync::Arc;
```

### Key Crossval Imports

```rust
use bitnet_inference::eval_logits_once;
use bitnet_sys::wrapper::{self, Session as CppSession};
use bitnet_crossval::logits_compare::compare_per_position_logits;
```

---

## 7. KEY UTILITIES & HELPERS

### resolve_default_model()

**Location**: Lines 1734-1760

```rust
fn resolve_default_model() -> Result<PathBuf> {
    let root = PathBuf::from("models");
    if !root.exists() {
        return Err(anyhow!("No models directory found"));
    }
    
    // Prefer default model
    let preferred = root.join(format!(
        "{}/{}",
        DEFAULT_MODEL_ID.replace('/', "-"),
        DEFAULT_MODEL_FILE
    ));
    if preferred.exists() {
        return Ok(preferred);
    }
    
    // Fallback: first *.gguf in models/
    for entry in WalkDir::new(&root) {
        if entry.extension() == "gguf" {
            return Ok(entry.path().to_path_buf());
        }
    }
    
    Err(anyhow!("No GGUF model found"))
}
```

### select_device()

**Location**: Lines 974-989

```rust
fn select_device(gpu: bool) -> (Device, &'static str) {
    if gpu {
        #[cfg(feature = "inference")]
        {
            eprintln!("🚀 Using GPU (CUDA)");
            return (Device::Cuda(0), "gpu");
        }
    }
    (Device::Cpu, "cpu")
}
```

### apply_deterministic_env()

**Location**: Lines 2662-2663 (called in crossval_cmd)

Sets:
- `RAYON_NUM_THREADS=1` (single-threaded)
- `BITNET_DETERMINISTIC=1`
- `BITNET_SEED=42`

---

## 8. ENVIRONMENT VARIABLES

### Crossval-Related

| Variable | Purpose | Example |
|----------|---------|---------|
| `BITNET_CPP_DIR` | Path to C++ BitNet checkout | `~/.cache/bitnet_cpp` |
| `CROSSVAL_GGUF` | GGUF model path for tests | `/path/to/model.gguf` |
| `CROSSVAL_ALLOW_CPP_FAIL` | Allow C++ failures (soft-fail) | `1` or `true` |
| `BITNET_DETERMINISTIC` | Enable deterministic inference | `1` |
| `BITNET_SEED` | Random seed for inference | `42` |
| `RAYON_NUM_THREADS` | Thread count (1 = serial) | `1` |

### Feature Flags

| Feature | Purpose |
|---------|---------|
| `crossval` | Enable C++ FFI and crossval tests |
| `integration-tests` | Enable integration test suite |
| `inference` | Enable model inference (async runtime) |
| `gpu` | Enable CUDA/GPU support |

---

## 9. SPRINT 1.2: IMPLEMENTATION ROADMAP

### What Needs to Be Built

**New Command**: `crossval-per-token` (or similar name)

**Location for Implementation**:
1. **Command Definition**: Add to `Cmd` enum in `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (around line 400)
2. **Command Handler**: Add new function `crossval_per_token_cmd()` in same file (around line 4700)
3. **Test Integration**: Can reuse existing `/home/steven/code/Rust/BitNet-rs/crossval/tests/per_position_logits.rs` 

### Command Signature Template

```rust
/// Compare per-token logits between Rust and C++ implementations
///
/// Generates detailed per-position metrics for multi-token generation,
/// identifying the first divergence point. Output: JSON with cosine similarity,
/// L2 distance, and maximum absolute difference for each token position.
CrossvalPerToken {
    /// Path to GGUF model
    #[arg(long)]
    model: Option<PathBuf>,
    
    /// Number of tokens to compare (default: 5)
    #[arg(long, default_value_t = 5)]
    num_tokens: usize,
    
    /// Test prompt
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,
    
    /// Path to C++ checkout (default: $HOME/.cache/bitnet_cpp)
    #[arg(long)]
    cpp_dir: Option<PathBuf>,
    
    /// Output JSON file for comparison data
    #[arg(long)]
    json: Option<PathBuf>,
    
    /// Print detailed per-position metrics
    #[arg(long, default_value_t = false)]
    verbose: bool,
},
```

### Function Template

```rust
fn crossval_per_token_cmd(
    model: Option<PathBuf>,
    num_tokens: usize,
    prompt: &str,
    cpp_dir: Option<&Path>,
    json_path: Option<&Path>,
    verbose: bool,
) -> Result<()> {
    // 1. Resolve model path
    let model_path = model.or_else(|| resolve_default_model().ok())
        .ok_or_else(|| anyhow!("Model required"))?;
    
    // 2. Validate C++ backend available
    let cpp = cpp_dir.map(|p| p.to_path_buf())
        .or_else(|| std::env::var_os("BITNET_CPP_DIR").map(PathBuf::from))
        .unwrap_or_else(|| dirs::home_dir().unwrap().join(".cache/bitnet_cpp"));
    
    if !cpp.exists() {
        bail!("C++ implementation not found. Run: cargo xtask fetch-cpp");
    }
    
    // 3. Initialize C++ backend
    // wrapper::init_backend()
    // let _guard = scopeguard::guard((), |_| wrapper::free_backend())
    
    // 4. Load tokenizer from model or mock
    // bitnet_tokenizers::loader::load_tokenizer()
    
    // 5. Generate multi-token logits
    // Use eval_logits_once() for Rust, CppSession::eval_and_get_logits() for C++
    
    // 6. Compare with logits_compare::compare_per_position_logits()
    // let divergence = compare_per_position_logits(&rust_logits, &cpp_logits)
    
    // 7. Format and output results
    // JSON serialization: serde_json::to_string_pretty(&divergence)?
    
    // 8. Write to file if requested
    // fs::write(json_path, json)?
    
    Ok(())
}
```

---

## 10. CRITICAL FILES SUMMARY

| File | Lines | Purpose |
|------|-------|---------|
| `xtask/src/main.rs` | 5624 | Command enum, dispatch, implementations |
| `crossval/src/logits_compare.rs` | 252 | Per-position comparison logic |
| `crossval/tests/per_position_logits.rs` | 295 | Integration tests (reference) |
| `xtask/src/ffi.rs` | ? | C++ FFI bridge |
| `crossval/src/cpp_bindings.rs` | ? | C++ bindings wrapper |

---

## 11. EXISTING PATTERNS TO FOLLOW

### Pattern 1: Model Loading (from benchmark_cmd)
1. Select device with `select_device(gpu)`
2. Load tokenizer with `bitnet_tokenizers::loader::load_tokenizer()`
3. Load model with `ModelLoader::new(device).load(model_path)`
4. Wrap in Arc: `Arc::<dyn bitnet_models::Model>::from(model)`
5. Create engine: `InferenceEngine::new(model_arc, tokenizer, device)`

### Pattern 2: C++ Integration (from per_position_logits tests)
1. Check C++ available: `if !bitnet_sys::is_available() { return Ok(()); }`
2. Initialize: `wrapper::init_backend()`
3. Cleanup guard: `let _guard = scopeguard::guard((), |_| wrapper::free_backend())`
4. Create session: `CppSession::load_deterministic(&model_path)?`
5. Tokenize: `cpp_session.tokenize(prompt)?`
6. Evaluate: `cpp_session.eval_and_get_logits(&tokens, 0)?`

### Pattern 3: Comparison (from logits_compare)
1. Collect logits as `Vec<Vec<f32>>` (positions × vocab)
2. Call `compare_per_position_logits(&rust_logits, &cpp_logits)`
3. Inspect `LogitsDivergence` fields
4. Report `first_divergence_token`, `per_token_cosine_sim`, `per_token_l2_dist`, `max_absolute_diff`

### Pattern 4: Command Structure
1. Add to `Cmd` enum with doc comments and arg definitions
2. Add case to `real_main()` match statement
3. Implement `fn command_name_cmd(...)` with full implementation
4. Use `anyhow::Result<()>` return type
5. Handle errors with `context()` for better error messages

---

## 12. TESTING STRATEGY

### Unit Tests
- Add to `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs`
- Already comprehensive tests for comparison metrics

### Integration Tests
- Location: `/home/steven/code/Rust/BitNet-rs/crossval/tests/per_position_logits.rs`
- Tests exist but can be enhanced with the new command
- Requires:
  - `feature = "crossval"` (C++ backend)
  - `BITNET_CPP_DIR` environment variable
  - `CROSSVAL_GGUF` pointing to test model

### Manual Testing
```bash
# Download model
cargo xtask download-model

# Fetch C++
cargo xtask fetch-cpp

# Test the new command
cargo xtask crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --num-tokens 5 \
  --prompt "The capital of France is" \
  --verbose \
  --json /tmp/divergence.json
```

---

## 13. KEY FUNCTIONS TO REUSE

| Function | Location | Purpose |
|----------|----------|---------|
| `resolve_default_model()` | L1734 | Auto-find GGUF in models/ |
| `select_device()` | L974 | CPU/GPU selection |
| `apply_deterministic_env()` | L2662 | Reproducible runs |
| `compare_per_position_logits()` | logits_compare.rs | Core comparison logic |
| `run_inference_internal()` | L5087 | Rust inference (not needed for C++) |
| `eval_logits_once()` | bitnet-inference | Rust logits extraction |

---

## 14. FEATURE GATE REQUIREMENTS

For the new command to work:

**Compile-Time**:
```toml
# Must add to xtask/Cargo.toml
[dependencies]
bitnet-crossval = { path = "../crossval" }
bitnet-sys = { path = "../crates/bitnet-sys", optional = true }

[features]
crossval = ["dep:bitnet-sys", "bitnet-inference/crossval"]
```

**Runtime**:
- `BITNET_CPP_DIR` must be set or `~/.cache/bitnet_cpp` must exist
- C++ binary must be built with `cargo xtask fetch-cpp`
- `CROSSVAL_GGUF` or `--model` argument required

---

## 15. OUTPUT FORMATS

### JSON Output Structure

```json
{
  "first_divergence_token": null,
  "per_token_cosine_sim": [0.999997, 0.999995, 0.999990],
  "per_token_l2_dist": [1.5e-5, 2.1e-5, 3.8e-5],
  "max_absolute_diff": 0.000423
}
```

### Human-Readable Output

```
Per-Token Logits Comparison
Model: models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
Prompt: "The capital of France is"
Tokens to compare: 5

Position 0: cosine_sim=0.999997, L2_dist=1.50e-05
Position 1: cosine_sim=0.999995, L2_dist=2.10e-05
Position 2: cosine_sim=0.999990, L2_dist=3.80e-05
Position 3: cosine_sim=0.999987, L2_dist=4.20e-05
Position 4: cosine_sim=0.999985, L2_dist=4.50e-05

First divergence: None
Max absolute difference: 0.000423
```

---

## 16. GLOSSARY

| Term | Definition |
|------|-----------|
| **Divergence** | Point where logits differ significantly (cosine sim < threshold) |
| **Cosine Similarity** | Angle between logit vectors (1.0 = identical, 0.0 = orthogonal) |
| **L2 Distance** | Euclidean distance between logit vectors |
| **Prefill** | Processing entire input sequence at once |
| **Decode** | Generating one token at a time, KV-cache reuse |
| **FFI** | Foreign Function Interface (Rust ↔ C++) |
| **GGUF** | Tensor serialization format (supports quantization) |
| **Parity** | Numerical equivalence between implementations |

---

## 17. NEXT STEPS FOR SPRINT 1.2

1. **Week 1**: Implement basic command structure
   - Add `CrossvalPerToken` to `Cmd` enum
   - Add dispatch case in `real_main()`
   - Scaffold `crossval_per_token_cmd()` function

2. **Week 2**: Integrate per-token comparison
   - Load model and tokenizer
   - Collect multi-token logits from both Rust and C++
   - Apply `compare_per_position_logits()`
   - Format output (JSON + human-readable)

3. **Week 3**: Testing and refinement
   - Run integration tests with existing models
   - Validate against known-good divergence points
   - Benchmark performance
   - Document results

4. **Week 4**: Polish and documentation
   - Add comprehensive doc comments
   - Update CLAUDE.md with usage examples
   - Create analysis script for batch divergence detection
   - Prepare PR and release notes

---

## 18. REFERENCES & LINKS

**In-Codebase**:
- Cross-validation framework: `/home/steven/code/Rust/BitNet-rs/crossval/`
- Xtask entry point: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs`
- Test infrastructure: `/home/steven/code/Rust/BitNet-rs/crossval/tests/per_position_logits.rs`

**Related Commands**:
- `cargo xtask crossval` - Existing cross-validation
- `cargo xtask benchmark` - Performance measurement
- `cargo xtask fetch-cpp` - Build C++ reference
- `cargo test --features crossval` - Run crossval tests

**Feature Flags**:
- `crossval` - Enable C++ comparison tests
- `integration-tests` - Enable full integration test suite
- `inference` - Enable async inference engine

