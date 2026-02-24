# Comprehensive Crossval Crate Exploration Report

## Executive Summary

The `bitnet-crossval` crate is a comprehensive cross-validation framework (~6500 LOC) designed to validate the Rust BitNet implementation against the C++ reference implementation. It provides:

- **Token-level parity validation** with fail-fast pre-gates
- **FFI boundary management** with safe C++ bindings
- **Per-position logits comparison** for divergence detection
- **Performance measurement** and receipt generation
- **Test fixtures and validation suites** for comprehensive testing

The crate is feature-gated for zero overhead when disabled, with clear separation between Rust-only and FFI-dependent components.

---

## 1. Module Hierarchy & Dependency Graph

### 1.1 Core Modules (Always Available)

```
crossval/
├── src/
│   ├── lib.rs                    # Root module (100 lines)
│   ├── token_parity.rs           # Token validation pre-gate (475 lines)
│   ├── logits_compare.rs         # Per-position divergence detection (200+ lines)
│   ├── validation.rs             # Comprehensive validation suite (200+ lines)
│   ├── score.rs                  # Perplexity & NLL evaluation (200+ lines)
│   ├── utils.rs                  # Comparison & performance utilities (150+ lines)
│   ├── fixtures.rs               # Test fixture management (94 lines)
│   ├── cpp_bindings.rs           # Safe FFI wrappers (300+ lines)
│   ├── comparison.rs             # High-level validation orchestration (150+ lines)
│   └── bitnet_cpp_wrapper.c      # C wrapper for C++ functions (64 lines)
```

### 1.2 Module Visibility & Exports

**From `lib.rs` (public API):**
```rust
pub mod cpp_bindings;          // FFI bindings (feature-gated: ffi, crossval)
pub mod comparison;            // Comparison logic (feature-gated: crossval)
pub mod fixtures;              // Test fixtures (always available)
pub mod logits_compare;        // Logits analysis (always available)
pub mod score;                 // Perplexity evaluation (always available)
pub mod token_parity;          // Token validation (always available)
pub mod utils;                 // Utilities (always available)
pub mod validation;            // Validation suite (always available)

pub enum CrossvalError { ... }  // Error types
pub struct CrossvalConfig { ... } // Configuration
pub fn assert_first_logits_match() // Feature-gated entry point
```

### 1.3 Dependency Graph

```
lib.rs (root)
├── token_parity.rs
│   └── (no internal deps; uses console crate)
├── logits_compare.rs
│   └── (no internal deps; pure math)
├── utils.rs
│   ├── perf module (timing)
│   ├── logging module (output)
│   └── CrossvalConfig
├── validation.rs
│   ├── bitnet_models (GgufReader)
│   └── ValidationSuite
├── score.rs
│   ├── Model trait (abstract)
│   ├── Tokenizer trait (abstract)
│   └── ScoreOutput
├── fixtures.rs
│   └── TestFixture (JSON-based)
├── comparison.rs [feature: crossval]
│   ├── cpp_bindings::CppModel
│   ├── fixtures::TestFixture
│   ├── utils (compare_tokens, logging)
│   └── CrossValidator
└── cpp_bindings.rs [feature: ffi/crossval]
    └── C FFI extern "C" { ... }
```

**External Dependencies:**
- `bitnet_inference` - Rust inference engine (real logits)
- `bitnet_models` - GGUF loading, model info
- `bitnet_sys` - C++ FFI wrapper (when available)
- `bitnet_tokenizers` - Token encoding/decoding
- `bitnet_common` - Device types, enums
- `serde/serde_json` - Serialization
- `anyhow` - Error handling
- `console` - Colored terminal output (token_parity only)
- `chrono` - Timestamps
- `sha2`, `blake3` - Hashing
- `tokio` - Async test infrastructure
- `criterion` - Benchmarking

---

## 2. Public API Entry Points

### 2.1 Top-Level API (lib.rs)

```rust
// Error type
pub enum CrossvalError {
    CppNotAvailable,
    ModelLoadError(String),
    InferenceError(String),
    ComparisonError(String),
    IoError(#[from] std::io::Error),
    SerializationError(#[from] serde_json::Error),
}
pub type Result<T> = std::result::Result<T, CrossvalError>;

// Configuration
pub struct CrossvalConfig {
    pub tolerance: f64,        // Float comparison tolerance
    pub max_tokens: usize,     // Max tokens to compare
    pub benchmark: bool,       // Enable performance measurement
}

// Main entry point (feature-gated: crossval)
pub fn assert_first_logits_match(model_path: &str, prompt: &str) {
    // 1. Init C++ backend (wrapper::init_backend)
    // 2. Load C++ model (CppSession::load_deterministic)
    // 3. Tokenize prompt (cpp_session.tokenize)
    // 4. Eval first token (cpp_session.eval_and_get_logits)
    // 5. Eval Rust (eval_logits_once)
    // 6. Compare with 1e-4 tolerance
}
```

### 2.2 Token Parity Pre-Gate (token_parity.rs)

**Primary Function:**
```rust
pub fn validate_token_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[i32],    // FFI returns i32
    prompt: &str,
) -> anyhow::Result<()>
    // Returns Ok(()) if tokens match (silent)
    // Returns Err + prints diagnostic if mismatch
    // Prints colored error with suggestions
```

**Supporting Functions:**
```rust
pub fn find_first_diff(rust_tokens: &[u32], cpp_tokens: &[u32]) -> usize
    // Finds first position where tokens differ

pub fn format_token_mismatch_error(error: &TokenParityError) -> String
    // Formats diagnostic error with:
    // - Token sequences (limited to 64 tokens)
    // - First diff position
    // - 4 actionable suggestions
    // - Copy-paste example command
```

**Error Type:**
```rust
pub struct TokenParityError {
    pub rust_tokens: Vec<u32>,
    pub cpp_tokens: Vec<u32>,
    pub first_diff_index: usize,
    pub prompt: String,
}
```

### 2.3 Logits Comparison (logits_compare.rs)

```rust
pub struct LogitsDivergence {
    pub first_divergence_token: Option<usize>,
    pub per_token_cosine_sim: Vec<f32>,
    pub per_token_l2_dist: Vec<f32>,
    pub max_absolute_diff: f32,
}
pub const COSINE_SIMILARITY_THRESHOLD: f32 = 1e-4;

pub fn compare_per_position_logits(
    rs_logits: &[Vec<f32>],      // positions × vocab
    cpp_logits: &[Vec<f32>],
) -> LogitsDivergence
    // Identifies first divergence position
    // Computes cosine similarity & L2 distance per token
    // Tracks max absolute difference
```

### 2.4 Validation Suite (validation.rs)

```rust
pub struct ValidationSuite {
    pub model_path: String,
    pub tokenizer_path: Option<String>,
    pub deterministic: bool,
}

impl ValidationSuite {
    pub fn new(model_path: impl Into<String>) -> Self
    pub fn with_tokenizer(mut self, path: impl Into<String>) -> Self
    
    // Individual validation gates
    pub fn validate_model_compatibility(&self) -> Result<ValidationResult>
    pub fn validate_token_parity(&self, prompts: &[String]) -> Result<TokenParityResult>
    pub fn validate_nll_parity(&self, dataset: &str) -> Result<NllParityResult>
    pub fn validate_performance(&self, baseline_path: Option<&Path>) -> Result<PerformanceResult>
    
    // Composite validation
    pub fn run_all(&self) -> Result<Vec<ValidationResult>>
}

// Result types
pub struct ValidationResult {
    pub gate: String,
    pub passed: bool,
    pub metrics: HashMap<String, serde_json::Value>,
    pub message: String,
}
pub struct TokenParityResult {
    pub total_prompts: usize,
    pub exact_matches: usize,
    pub match_rate: f64,
    pub divergences: Vec<TokenDivergence>,
}
pub struct NllParityResult {
    pub rust_nll: f64,
    pub cpp_nll: f64,
    pub delta: f64,
    pub rust_ppl: f64,
    pub cpp_ppl: f64,
}
pub struct PerformanceResult {
    pub tokens_per_second: f64,
    pub rss_mb: f64,
    pub baseline_tok_s: Option<f64>,
    pub baseline_rss_mb: Option<f64>,
    pub throughput_ratio: Option<f64>,
    pub memory_ratio: Option<f64>,
}
```

### 2.5 Comparison Orchestration (comparison.rs)

```rust
pub struct ComparisonResult {
    pub test_name: String,
    pub prompt: String,
    pub rust_tokens: Vec<u32>,
    pub cpp_tokens: Vec<u32>,
    pub tokens_match: bool,
    pub rust_performance: Option<f64>,
    pub cpp_performance: Option<f64>,
    pub error: Option<String>,
}

pub struct CrossValidator {
    config: CrossvalConfig,
}

impl CrossValidator {
    pub fn new(config: CrossvalConfig) -> Self
    pub fn validate_fixture(&self, fixture: &TestFixture) -> Result<Vec<ComparisonResult>>
}

pub fn validate_all_fixtures(config: CrossvalConfig) -> Result<Vec<ComparisonResult>>
```

### 2.6 FFI Bindings (cpp_bindings.rs)

**Public Info Structs:**
```rust
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub parameter_count: u64,
    pub quantization: String,
}

pub struct InferenceStats {
    pub tokens_generated: usize,
    pub inference_time_ms: u64,
    pub tokens_per_second: f64,
    pub memory_used_mb: f64,
}
```

**C++ Model Wrapper (feature: ffi, have_cpp):**
```rust
pub struct CppModel { handle: *mut c_void }

impl CppModel {
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self>
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<Vec<u32>>
    pub fn model_info(&self) -> Result<ModelInfo>
    pub fn is_ready(&self) -> bool
}

impl Drop for CppModel { ... }
unsafe impl Send for CppModel { ... }

// Availability detection
pub fn is_available() -> bool
pub fn version_info() -> Result<String>
```

**Fallback (no FFI):**
```rust
pub struct CppModel;  // Stub
// All methods return Err(CppNotAvailable)
pub fn is_available() -> bool { false }
```

**C FFI Boundary:**
```c
// From bitnet_cpp_wrapper.c (extern "C")
void* bitnet_cpp_create_model(const char* model_path);
int bitnet_cpp_generate(void* model, const char* prompt, 
                        int max_tokens, unsigned int* tokens_out, 
                        int* tokens_count);
void bitnet_cpp_destroy_model(void* model);
```

### 2.7 Test Fixtures (fixtures.rs)

```rust
pub struct TestFixture {
    pub name: String,
    pub model_path: PathBuf,
    pub test_prompts: Vec<String>,
    pub expected_tokens: Option<Vec<Vec<u32>>>,
}

impl TestFixture {
    pub fn load(name: &str) -> Result<Self>
    pub fn list_available() -> Result<Vec<String>>
}

pub fn create_minimal_fixture() -> TestFixture
pub const STANDARD_PROMPTS: &[&str] = &[...]
```

### 2.8 Utilities (utils.rs)

```rust
// Token comparison
pub fn compare_tokens(
    rust_tokens: &[u32],
    cpp_tokens: &[u32],
    config: &CrossvalConfig,
) -> Result<bool>

// Float comparison
pub fn compare_floats(
    rust_values: &[f32],
    cpp_values: &[f32],
    config: &CrossvalConfig,
) -> Result<bool>

// Performance measurement
pub mod perf {
    pub struct PerfMeasurement {
        pub duration: Duration,
        pub tokens_per_second: f64,
    }
    pub fn measure<F>(f: F) -> (PerfMeasurement, F::Output)
        where F: FnOnce() -> usize
}

// Logging
pub mod logging {
    pub fn log_comparison(test_name: &str, rust: usize, cpp: usize, success: bool)
    pub fn log_performance(test_name: &str, rust_tps: f64, cpp_tps: f64)
}
```

---

## 3. FFI Integration Architecture

### 3.1 FFI Boundary Locations

**Primary FFI Boundary: `cpp_bindings.rs`**

Feature gates:
- `#[cfg(all(feature = "ffi", have_cpp))]` - Real FFI implementation
- `#[cfg(any(not(feature = "ffi"), not(have_cpp)))]` - Stub implementation

**C Wrapper: `src/bitnet_cpp_wrapper.c`**
- Compiled via `build.rs` with `cc::Build`
- Provides extern "C" interface:
  - `bitnet_cpp_create_model(const char* path) -> void*`
  - `bitnet_cpp_generate(void*, prompt, max_tokens, tokens_out, count) -> int`
  - `bitnet_cpp_destroy_model(void*)`

**Build Integration: `build.rs`**

```rust
#[cfg(feature = "ffi")]
fn compile_ffi() {
    // 1. Compile C wrapper
    cc::Build::new()
        .file("src/bitnet_cpp_wrapper.c")
        .compile("bitnet_cpp_wrapper");
    
    // 2. Search for C++ libraries in priority order:
    //    - BITNET_CROSSVAL_LIBDIR (explicit from setup-cpp-auto)
    //    - BITNET_CPP_DIR + build/3rdparty/llama.cpp/src
    //    - BITNET_CPP_DIR + build/3rdparty/llama.cpp/ggml/src
    //    - BITNET_CPP_DIR + build/bin (alternative)
    //    - BITNET_CPP_DIR + build/lib (fallback)
    
    // 3. Link all found libraries (libbitnet, libllama, libggml)
    // 4. Link C++ standard library (stdc++ or c++ depending on OS)
}
```

### 3.2 FFI Call Chain

```
User CLI or Test
  ↓
xtask::crossval_per_token_cmd() [xtask/src/main.rs]
  ↓
token_parity::validate_token_parity() [crossval/src/token_parity.rs]
  ↓
bitnet_sys::wrapper (C++ tokenizer) ← FFI boundary
  ↓
bitnet_cpp_wrapper.c (extern "C")
  ↓
llama.cpp or mock implementation
```

**Example: Per-token crossval flow**

```
1. xtask calls crossval_per_token_cmd(model, tokenizer, prompt)
2. Create Rust inference engine
   - Load model: ModelLoader::load(model_path)
   - Create engine: InferenceEngine::new(model)
3. Tokenize with Rust
   - Use bitnet_tokenizers::load_tokenizer_from_gguf_reader()
   - Generate rust_tokens: Vec<u32>
4. Tokenize with C++ (via FFI)
   - Call bitnet_sys::wrapper::Session::tokenize(prompt)
   - Returns cpp_tokens: Vec<i32>
5. Pre-gate: Token parity validation
   - Call validate_token_parity(&rust_tokens, &cpp_tokens, prompt)
   - If mismatch: print diagnostic and exit(2)
   - If match: continue
6. Logits comparison
   - Rust: Call engine.eval_ids(&rust_tokens)
   - C++: Call cpp_session.eval_and_get_logits(&tokens, pos)
   - Compare with compare_per_position_logits()
   - Report divergence position
```

### 3.3 Configuration & Environment Variables

**Build-time:**
```bash
BITNET_CPP_DIR=/path/to/bitnet.cpp        # C++ source root
BITNET_CROSSVAL_LIBDIR=/path/to/libs      # Explicit library dir (highest priority)
```

**Runtime:**
```bash
CROSSVAL_GGUF=/path/to/model.gguf         # Test model path
BITNET_DETERMINISTIC=1                     # Reproducible inference
BITNET_SEED=42                             # Deterministic seed
RAYON_NUM_THREADS=1                        # Single-threaded (for determinism)
LD_LIBRARY_PATH=/path/to/llama.cpp/src    # C++ library path (Linux)
DYLD_LIBRARY_PATH=/path/to/llama.cpp/src  # C++ library path (macOS)
```

---

## 4. Test Structure & Coverage

### 4.1 Test Files (13 test modules, ~165 KB)

```
crossval/tests/
├── smoke.rs                    (2.9 KB)  - Environment preflight checks
├── parity_bitnetcpp.rs        (34.7 KB) - Real parity tests (async)
├── parity_receipts.rs         (18.7 KB) - Receipt generation & validation
├── parity.rs                  (12.8 KB) - Deterministic logits comparison
├── per_position_logits.rs     (10.5 KB) - Per-token divergence detection
├── qk256_crossval.rs          (15.8 KB) - QK256 vs FP32 reference
├── ffi_integration.rs         (8.1 KB)  - FFI lifecycle & error handling
├── framework_validation.rs    (17.8 KB) - Comprehensive validation suite
├── performance_validation.rs  (19.5 KB) - Throughput & memory benchmarks
├── iq2s_validation.rs         (10.4 KB) - IQ2_S quantization parity
├── token_equivalence.rs       (5.0 KB)  - Token ID A/B testing
├── cpp_probe.rs               (0.4 KB)  - C++ availability detection
└── ms_bitnet_mapping.rs       (1.0 KB)  - Model name mapping
```

### 4.2 Test Organization

**Feature Requirements:**
```toml
[tests]
# Base tests (no features)
- smoke.rs
- cpp_probe.rs
- ms_bitnet_mapping.rs

# FFI-dependent tests (requires "ffi" feature)
[[test]]
name = "ffi_integration"
required-features = ["ffi"]

[[test]]
name = "token_equivalence"
required-features = ["ffi"]

# Full crossval tests (requires "crossval" + "integration-tests")
[[test]]
name = "parity_bitnetcpp"
required-features = ["crossval", "integration-tests"]

# IQ2_S specific (requires "iq2s-ffi")
[[test]]
name = "iq2s_validation"
required-features = ["iq2s-ffi"]

# All others: unguarded (run with default features)
```

### 4.3 Test Coverage Analysis

**Token Parity Tests (token_parity.rs):**
- AC1: Detect token mismatch before logits eval ✓
- AC2: Display both sequences on mismatch ✓
- AC3: Identify first diff position ✓
- AC4: Exit code 2 on mismatch (TODO: subprocess test)
- AC5-8: Error message quality ✓
- AC9: Silent success when tokens match ✓
- AC10: Performance <100ms for <1000 tokens ✓
- Scenarios: Duplicate BOS, tokens match, length mismatch, edge cases ✓

**Parity Tests (parity.rs, parity_bitnetcpp.rs):**
- Single-token logits parity (~1e-4 tolerance)
- Multi-token generation with greedy sampling
- Template auto-detection (raw, instruct, llama3-chat)
- BOS/EOS handling variations
- Deterministic seed propagation
- Receipt generation & validation

**Per-Position Logits (per_position_logits.rs):**
- Cosine similarity computation
- L2 distance calculation
- First divergence detection
- Per-token comparison at multiple positions

**QK256 Validation (qk256_crossval.rs):**
- QK256 kernel vs FP32 reference
- Packed format correctness
- Cosine similarity threshold validation
- Production model scenario tests

**FFI Integration (ffi_integration.rs):**
- C++ availability detection
- Model loading error handling
- Lifecycle (load → info → cleanup)
- Generation with validation
- Null pointer safety

**Performance (performance_validation.rs):**
- Throughput benchmarks (tokens/sec)
- Memory profiling (RSS)
- Baseline comparison
- Performance ratio computation

**Validation Suite (framework_validation.rs):**
- Model compatibility checks
- Tensor mapping validation
- NLL/Perplexity parity
- Comprehensive test runner

### 4.4 Test Execution Patterns

```bash
# Run all enabled tests
cargo test --workspace --features cpu

# Run with nextest (recommended)
cargo nextest run --workspace --features cpu

# Run specific test module
cargo test -p bitnet-crossval --test parity

# Run with C++ FFI
BITNET_CPP_DIR=/path cargo test --features crossval

# Run with full integration tests
cargo test --features crossval,integration-tests

# Run per-token crossval
cargo run -p xtask -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4
```

---

## 5. Configuration & Environment Patterns

### 5.1 Feature Flags

```toml
[features]
# Core features
default = []              # No default features
crossval = [
    "dep:bindgen",        # C++ binding generation
    "dep:cc",             # C compilation
    "dep:bitnet-sys",     # C++ FFI wrapper
    "bitnet-sys/ffi",     # Enable FFI in bitnet-sys
    "bitnet-inference/ffi",
    "ffi"                 # Enable local FFI
]
ffi = ["dep:cc", "bitnet-inference/ffi"]
iq2s-ffi = [
    "bitnet-models/iq2s-ffi",
    "bitnet-ggml-ffi/iq2s-ffi"
]
cpp-probe = []            # C++ environment detection
integration-tests = []    # Enable expensive tests
cpu = ["bitnet-inference/cpu", "bitnet-models/cpu"]
gpu = ["bitnet-inference/gpu", "bitnet-models/gpu"]
```

### 5.2 Cargo.toml Dependencies

```toml
[dependencies]
# Core (always available)
serde = { version = "1.0.228", features = ["derive"] }
serde_json = "1.0.145"
anyhow = "1.0.100"
thiserror = "2.0.17"
chrono = { version = "0.4.42", features = ["serde"] }
toml = "0.9.8"

# Feature-gated (FFI/crossval)
bindgen = { version = "0.72.1", optional = true }
cc = { version = "1.2.41", optional = true }
bitnet-sys = { path = "../crates/bitnet-sys", optional = true }

# Internal crates (always available for compilation)
bitnet-inference = { path = "../crates/bitnet-inference" }
bitnet-models = { path = "../crates/bitnet-models" }
bitnet-common = { path = "../crates/bitnet-common" }
bitnet-tokenizers = { path = "../crates/bitnet-tokenizers", features = ["spm"] }
bitnet-ggml-ffi = { path = "../crates/bitnet-ggml-ffi" }

# Utilities
scopeguard = "1.2.0"                      # RAII guards for cleanup
dirs = "6.0.0"                            # Home directory detection
humantime = "2.3.0"                       # Human-readable durations
tokio = { version = "1.48.0", features = ["full"] }  # Async test support
sha2 = "0.10.9"                           # Model SHA256
blake3 = "1.8.2"                          # Prompt hashing
console = "0.16.1"                        # Colored terminal output (token_parity)

[dev-dependencies]
criterion = { version = "0.7.0", features = ["html_reports"] }
tempfile = "3.23.0"
rand = "0.9.2"
half = "2.7.1"
```

### 5.3 Build-time Configuration

**build.rs flow:**
1. Export rustc version & target triple to env vars
2. Check `cfg(feature = "ffi")`
3. If enabled:
   - Compile C wrapper (`src/bitnet_cpp_wrapper.c`)
   - Search for C++ libraries in priority order
   - Link available libraries (libbitnet, libllama, libggml)
   - Link C++ std library (stdc++/c++)
   - Set `have_cpp` cfg flag if libraries found

**Library Search Paths (prioritized):**
1. `$BITNET_CROSSVAL_LIBDIR` (explicit, highest priority)
2. `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src`
3. `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src`
4. `$BITNET_CPP_DIR/build/bin`
5. `$BITNET_CPP_DIR/build/lib`
6. `$BITNET_CPP_DIR/lib`
7. `$BITNET_CPP_DIR/build`

---

## 6. Data Flow Architecture

### 6.1 CLI Entry Points (xtask integration)

**Command: `crossval-per-token`**
```
xtask main.rs::Cmd::CrossvalPerToken
  └─→ crossval_per_token_cmd(model, tokenizer, prompt, max_tokens, cos_tol, format)
        ├─→ Load Rust model & engine
        ├─→ Rust tokenization
        ├─→ C++ tokenization (via bitnet_sys)
        ├─→ validate_token_parity() [pre-gate]
        │   └─→ Exit(2) on mismatch
        ├─→ Rust eval_ids()
        ├─→ C++ eval_and_get_logits()
        ├─→ compare_per_position_logits()
        └─→ Write receipt to ci/inference.json
```

**Command: `crossval` (sweep)**
```
xtask main.rs::Cmd::Crossval
  └─→ crossval_cmd(model, cpp_dir, release, extra_args, dry_run)
        ├─→ Run 1-token, 2-token, 4-token scenarios
        ├─→ Capture traces & receipts
        ├─→ Compare Rust vs C++
        └─→ Generate report → target/crossval_report.json
```

**Command: `setup-crossval`**
```
xtask main.rs::Cmd::SetupCrossval
  └─→ setup_crossval()
        ├─→ Generate test fixtures
        ├─→ Build with crossval features
        └─→ Prepare C++ environment
```

### 6.2 Test Execution Flow

**Parity Test (parity_bitnetcpp.rs)**
```
Test harness
  └─→ test_model_path()  [env detection]
        ├─→ Check bitnet_sys::is_available()
        ├─→ Check CROSSVAL_GGUF env
        └─→ Return Some(path) or skip
  
  └─→ wrapper::init_backend()  [C++ setup]
  
  └─→ CppSession::load_deterministic()
        ├─→ Load model from path
        └─→ Set seed=0, deterministic=true
  
  └─→ For each prompt:
        ├─→ cpp_session.tokenize(prompt)
        ├─→ rust_side_tokenize_and_meta(model, prompt)
        ├─→ rust_eval_last_logits(model, tokens)
        ├─→ cpp_session.eval_and_get_logits(tokens, pos)
        ├─→ compare_per_position_logits()
        ├─→ Assert cosine_sim > threshold
        └─→ Write receipt
```

### 6.3 Data Structures in Flight

**Token Sequence:**
```
prompt: "What is 2+2?"
  ↓
[Rust Tokenizer]
  → Vec<u32>: [128000, 1229, 374, 220, 17]
  
[C++ Tokenizer via FFI]
  → Vec<i32>: [128000, 1229, 374, 220, 17]  (returned as i32)
  
[Token Parity Validation]
  → Convert i32 → u32
  → Compare sequences
  → If diff: TokenParityError { rust, cpp, first_diff_index, prompt }
```

**Logits Pipeline:**
```
tokens: [128000, 1229, 374, 220, 17]
  ↓
[Rust Inference Engine]
  → Vec<Vec<f32>>: [
      [0.1, 0.2, 0.3, ...],  // logits after token 0
      [0.15, 0.25, 0.35, ...], // logits after token 1
      ...
    ]
  
[C++ Inference via FFI]
  → Vec<Vec<f32>>: [same layout]
  
[Per-Position Comparison]
  → LogitsDivergence {
      first_divergence_token: None,
      per_token_cosine_sim: [0.9999, 0.9998, ...],
      per_token_l2_dist: [0.0001, 0.00015, ...],
      max_absolute_diff: 0.0005
    }
```

**Receipt JSON:**
```json
{
  "timestamp": "2025-10-25T...",
  "model_path": "model.gguf",
  "prompt": "What is 2+2?",
  "compute_path": "real",
  "backend": "cpu|cuda",
  "parity": {
    "cpp_available": true,
    "cosine_similarity": 0.9923,
    "exact_match_rate": 1.0,
    "status": "ok"
  },
  "metrics": {
    "tokens_per_second": 12.5,
    "memory_mb": 2048
  },
  "kernel_ids": ["scalar_i2s_qk256", "gemv_fp32"],
  "validation_gates": {
    "token_parity": "pass",
    "logits_parity": "pass",
    "performance": "pass"
  }
}
```

---

## 7. Backend-Specific Code Patterns

### 7.1 Feature-Gated Code Pattern

**General Pattern:**
```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_specific_function() { ... }

#[cfg(all(feature = "ffi", have_cpp))]
pub fn cpp_function() { ... }

#[cfg(any(not(feature = "ffi"), not(have_cpp)))]
pub fn fallback_function() { ... }
```

**In cpp_bindings.rs:**
```rust
#[cfg(all(feature = "ffi", have_cpp))]
mod imp {
    unsafe extern "C" {
        fn bitnet_cpp_create_model(model_path: *const c_char) -> *mut c_void;
        // ...
    }
    pub struct CppModel { handle: *mut c_void }
    impl CppModel {
        pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> { ... }
        // Real implementation
    }
}

#[cfg(any(not(feature = "ffi"), not(have_cpp)))]
pub struct CppModel;  // Stub
impl CppModel {
    pub fn load<P: AsRef<Path>>(_model_path: P) -> Result<Self> {
        Err(CrossvalError::CppNotAvailable)
    }
}

// Public API always available
pub fn is_available() -> bool {
    #[cfg(all(feature = "ffi", have_cpp))]
    {
        // Check availability
    }
    #[cfg(any(not(feature = "ffi"), not(have_cpp)))]
    {
        false
    }
}
```

### 7.2 Device Feature Detection

**At Build Time:**
```bash
# build.rs sets cfg flag
println!("cargo:rustc-cfg=have_cpp");
```

**At Compile Time:**
```rust
#[cfg(have_cpp)]
// Only compiled if C++ libraries found
```

**At Runtime:**
```rust
pub fn is_available() -> bool {
    // May check library symbols, environment, etc.
}
```

### 7.3 Determinism & Reproducibility

**Deterministic Inference:**
```rust
// In parity tests
env::set_var("BITNET_DETERMINISTIC", "1");
env::set_var("BITNET_SEED", "42");
env::set_var("RAYON_NUM_THREADS", "1");

// In C++ session
CppSession::load_deterministic(model_path)
    // Sets seed=0, temp=0.0 (greedy)
```

**Tolerance Thresholds:**
- Token comparison: Exact match required
- Logits comparison: 1e-4 absolute difference (float tolerance)
- Cosine similarity: >0.9999 (1e-4 threshold for divergence)
- NLL/Perplexity: Configurable tolerance in CrossvalConfig

---

## 8. Configuration Patterns

### 8.1 Environment Variable Usage

```rust
// In parity tests
std::env::var("CROSSVAL_GGUF")       // Model path
std::env::var("BITNET_CPP_DIR")      // C++ source root
std::env::var("BITNET_DETERMINISTIC")
std::env::var("BITNET_SEED")
std::env::var("RAYON_NUM_THREADS")

// In build.rs
std::env::var("BITNET_CPP_DIR")      // C++ root
std::env::var("BITNET_CROSSVAL_LIBDIR") // Explicit lib dir
std::env::var("HOME")                // Default cache

// In token_parity tests
std::env::var("RUST_LOG")            // Logging level
```

### 8.2 Configuration Struct

```rust
pub struct CrossvalConfig {
    pub tolerance: f64,      // 1e-6 default (logits)
    pub max_tokens: usize,   // 1000 default
    pub benchmark: bool,     // false default
}

impl Default for CrossvalConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            max_tokens: 1000,
            benchmark: false,
        }
    }
}
```

### 8.3 Validation Policy Pattern

```rust
// Custom policies (from validation.rs)
pub enum ValidationGate {
    ModelCompatibility,
    TokenParity,
    NllParity,
    Performance,
}

impl ValidationSuite {
    pub fn validate_token_parity(&self, prompts: &[String]) -> Result<TokenParityResult> {
        // Configuration parameters:
        // - prompts: Vec of test strings
        // - (could add thresholds, timeout, etc.)
    }
}
```

---

## 9. Test Coverage Gaps & TODOs

### 9.1 Known Test Gaps

```rust
// In token_parity.rs tests:
#[test]
#[ignore = "TODO: Capture stderr to validate error output format"]
fn test_error_displays_both_sequences() { ... }

#[test]
#[ignore = "TODO: Use std::process::Command to spawn subprocess and check exit code"]
fn test_exit_code_on_mismatch() { ... }

#[test]
#[ignore = "TODO: Decide how to handle negative i32 tokens from C++"]
fn test_negative_cpp_tokens() { ... }
```

### 9.2 Missing Coverage Areas

1. **Subprocess Exit Code Testing** - Need integration test that spawns process
2. **Stderr Capture** - No test for full error message format
3. **Negative Token Handling** - FFI i32 conversion edge case
4. **C++ Library Not Found** - Fallback behavior untested
5. **Multi-GPU Scenarios** - GPU parity not fully covered
6. **Large Model Stress** - Memory scaling not tested

### 9.3 Blocked Tests (Issue Dependencies)

Tests marked `#[ignore]` are often blocked by:
- **Issue #254** - Shape mismatch in layer norm (affects real inference tests)
- **Issue #260** - Mock elimination (transition to real inference)
- **Issue #469** - Tokenizer parity & FFI build hygiene

---

## 10. Key Design Decisions

### 10.1 Token Parity Pre-Gate

**Why separate from logits comparison:**
- Fail-fast before expensive inference
- Catches template/BOS mismatches early
- Provides diagnostic output with suggestions
- Prevents silent divergence from affecting downstream tests

**Error handling:**
- Returns `Err` instead of exiting (testable)
- Caller (xtask) should exit with code 2
- Prints colored diagnostic to stderr

### 10.2 Per-Position Logits Comparison

**Why position-by-position:**
- Identifies exact divergence point
- Enables debugging of intermediate computation
- Supports multi-token generation validation
- Computes cosine similarity AND L2 distance

**Thresholds:**
- Cosine similarity: 1e-4 (1.0 - threshold)
- Max absolute diff: 1e-4
- Per-token metrics for post-hoc analysis

### 10.3 Receipt Generation

**Why JSON-based:**
- Serializable, diff-able across runs
- Can be parsed for CI/CD integration
- Includes schema version for compatibility
- Captures both metrics and metadata

**Receipt Sections:**
- Compute path (real vs mock)
- Backend ID (cpu vs cuda)
- Parity metrics (similarity, match rate)
- Kernel IDs for performance attribution
- Validation gate results

---

## 11. Known Limitations & Workarounds

### 11.1 C++ Wrapper Status

Current state: **Mock implementation** in `src/bitnet_cpp_wrapper.c`

Actual behavior:
```c
void* bitnet_cpp_create_model(const char* model_path) {
    // Just checks file exists, returns malloc'd struct
    // Does NOT actually load model via llama.cpp
}

int bitnet_cpp_generate(...) {
    // Returns hardcoded dummy tokens [100, 101, ...]
    // Does NOT perform actual generation
}
```

**When enabled:** Tests that depend on C++ inference will fail or return dummy data

**Workaround:** Set `BITNET_CPP_DIR` and build with real C++ libraries (requires `setup-cpp-auto`)

### 11.2 IQ2_S FFI Limitations

- Requires `--features iq2s-ffi` 
- Only available when GGML libraries compiled with IQ2_S support
- Validation tests skipped if not available

### 11.3 GPU Parity

- GPU backend requires `--features gpu`
- CUDA kernel dispatch not yet fully integrated
- Some tests marked `#[ignore]` pending GPU feature unification (Issue #439 resolved)

---

## 12. Integration with xtask

### 12.1 Registered Commands

In `xtask/src/main.rs`:

```rust
enum Cmd {
    // ...
    CrossvalPerToken { model, tokenizer, prompt, max_tokens, cos_tol, format },
    Crossval { model, cpp_dir, release, extra, dry_run },
    FullCrossval { force, tag, backend, cmake_flags, repo },
    SetupCrossval,
    // ...
}

fn main() {
    // ...
    match cmd {
        Cmd::CrossvalPerToken { ... } => crossval_per_token_cmd(...)?,
        Cmd::Crossval { ... } => crossval_cmd(...)?,
        Cmd::FullCrossval { ... } => full_crossval_cmd(...)?,
        Cmd::SetupCrossval => setup_crossval()?,
        // ...
    }
}
```

### 12.2 Entry Point Examples

```bash
# Per-token validation
cargo run -p xtask -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --cos-tol 0.999

# Full sweep
cargo run -p xtask -- crossval \
  --model models/model.gguf \
  --cpp-dir ~/.cache/bitnet_cpp \
  --release

# Setup infrastructure
cargo run -p xtask -- setup-crossval
```

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Total LOC** | ~6,500 (src + tests) |
| **Modules** | 8 core + 13 test modules |
| **Public API Functions** | 30+ (token_parity, logits_compare, validation, etc.) |
| **FFI Boundary** | cpp_bindings.rs ↔ bitnet_cpp_wrapper.c |
| **Feature Flags** | 6 (crossval, ffi, iq2s-ffi, cpp-probe, integration-tests, cpu, gpu) |
| **Test Categories** | Token parity, logits, QK256, FFI, performance, validation |
| **Configuration Patterns** | Env vars, CrossvalConfig struct, YAML fixtures |
| **Backend Support** | CPU (primary), GPU (via features), CPU-specific (SIMD) |
| **Error Handling** | CrossvalError enum + anyhow::Result |
| **Key Dependencies** | bitnet_inference, bitnet_models, bitnet_sys, serde_json |

