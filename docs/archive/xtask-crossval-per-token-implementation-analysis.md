# xtask crossval-per-token Command Implementation Report

## Executive Summary

The `crossval-per-token` command is a per-token logits comparison tool between Rust and C++ implementations. **CRITICAL ISSUE FOUND**: The implementation evaluates Rust logits at lines 2933, then C++ logits at lines 2954-2975, but the token parity validation happens in BETWEEN (lines 2959-2968), AFTER logits have already been computed. This violates the specification which states parity must be validated before expensive logits comparisons.

## 1. CLI Command Definition

### Location
**File**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs`

### Lines 389-430: Command Documentation & Argument Definition

```rust
/// Compare Rust vs C++ logits position-by-position (find first diverging token)
///
/// Runs deterministic inference with both Rust and C++ implementations,
/// comparing output logits at each token position. Reports the first position
/// where cosine similarity falls below the threshold, helping identify
/// divergence points in cross-validation workflows.
///
/// Example:
///   cargo run -p xtask -- crossval-per-token \
///     --model models/model.gguf \
///     --tokenizer models/tokenizer.json \
///     --prompt "Hello world" \
///     --max-tokens 4 \
///     --cos-tol 0.999 \
///     --format text
#[cfg(feature = "inference")]
#[command(name = "crossval-per-token")]
CrossvalPerToken {
    /// Path to GGUF model file
    #[arg(long)]
    model: PathBuf,

    /// Path to tokenizer file
    #[arg(long)]
    tokenizer: PathBuf,

    /// Input prompt to process
    #[arg(long)]
    prompt: String,

    /// Maximum tokens to generate (excluding prompt)
    #[arg(long, default_value_t = 4)]
    max_tokens: usize,

    /// Cosine similarity tolerance (0.0-1.0, where 1.0 = identical)
    #[arg(long, default_value_t = 0.999)]
    cos_tol: f32,

    /// Output format: "text" or "json"
    #[arg(long, default_value = "text")]
    format: String,
},
```

### CLI Signature

```bash
cargo run -p xtask -- crossval-per-token \
  --model <PATH>           # Required: GGUF model file
  --tokenizer <PATH>       # Required: tokenizer.json
  --prompt <STRING>        # Required: input prompt
  --max-tokens <N>         # Optional: defaults to 4 (UNUSED - reserved)
  --cos-tol <F>           # Optional: defaults to 0.999
  --format <STR>          # Optional: "text" or "json", defaults to "text"
```

## 2. Argument Flow to Handler

### Line 897-899: Command Dispatch

```rust
#[cfg(feature = "inference")]
Cmd::CrossvalPerToken { model, tokenizer, prompt, max_tokens, cos_tol, format } => {
    crossval_per_token_cmd(&model, &tokenizer, &prompt, max_tokens, cos_tol, &format)?;
    Ok(())
}
```

Arguments pass directly from clap-parsed structs to the handler function without transformation.

## 3. Handler Function Signature

### Line 2901-2908: Function Definition

```rust
#[cfg(feature = "inference")]
fn crossval_per_token_cmd(
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    _max_tokens: usize, // Reserved for future generation mode
    cos_tol: f32,
    format: &str,
) -> Result<()> {
```

**Note**: `_max_tokens` is intentionally unused (prefixed with `_`), indicating this is scaffolding for future generation mode. Currently, only prompt tokens are evaluated.

## 4. COMPLETE EXECUTION FLOW WITH EXACT LINE NUMBERS

### Phase 0: Imports (Lines 2909-2910)
```rust
use bitnet_crossval::logits_compare::compare_per_position_logits;
use bitnet_inference::parity::eval_logits_all_positions;
```

### Phase 1: Initialize & Load Tokenizer (Lines 2912-2927)
```
Line 2912-2916:  Print header with model, prompt, tolerance
Line 2918-2919:  Print "📝 Tokenizing prompt..."
Line 2920:       Load tokenizer: bitnet_tokenizers::loader::load_tokenizer(tokenizer_path)
Line 2921:       Encode prompt: tokenizer.encode(prompt, false, false)
Line 2922:       Convert tokens to i32: token_ids = vec![...]
Line 2925-2926:  Print token count (prompt only)
```

### Phase 2: RUST LOGITS EVALUATION (Lines 2929-2938) ⚠️ **BUG LOCATION 1**
```
Line 2930:       Print "🦀 Evaluating Rust logits for all positions..."
Line 2931-2932:  Convert model_path to &str
Line 2933:       ⚠️ CALL: rust_logits = eval_logits_all_positions(model_path_str, &token_ids)
                    Function: crates/bitnet-inference/src/parity.rs:157
                    - Loads GGUF model
                    - Initializes KV cache
                    - Runs forward pass for all positions
                    - Returns Vec<Vec<f32>> [positions][vocab_logits]
Line 2934-2938:  Print rust logits dimensions
```

### Phase 3: C++ BACKEND INITIALIZATION (Lines 2940-2957)
```
Line 2941:       Print "🔧 Evaluating C++ logits for all positions..."
Line 2943-2948:  Check if C++ FFI is available: bitnet_sys::is_available()
                 - If not: bail with error message about --features crossval
Line 2950-2952:  Initialize C++ backend with RAII guard:
                 - bitnet_sys::wrapper::init_backend()
                 - Create scopeguard to free on drop
Line 2954:       Create session: Session::load_deterministic(model_path_str)
                 - Loads C++ model wrapper
                 - Creates context with deterministic settings (n_ctx=2048, n_batch=512, n_threads=1)
Line 2957:       Tokenize with C++ tokenizer: cpp_session.tokenize(prompt)
                 - Returns Vec<i32> from C++ implementation
```

### Phase 4: TOKEN PARITY VALIDATION (Lines 2959-2969) ⚠️ **BUG LOCATION 2 - WRONG ORDERING**
```
Line 2960:       Print "🔒 Validating token parity..."
Line 2961:       Convert rust_tokens to u32: rust_tokens_u32 = token_ids as u32
Line 2962-2963:  ⚠️ CALL: validate_token_parity(&rust_tokens_u32, &cpp_tokens, prompt)
                    Location: crossval/src/token_parity.rs:79
                    Validates that Rust and C++ token sequences match exactly
                    Returns Ok(()) on match, Err on mismatch
                    Prints diagnostic error to stderr on mismatch
Line 2962-2967:  Error handling:
                 - If Err: eprintln!("Error: {}", e)
                 - std::process::exit(2) [EXIT CODE 2 for token mismatch]
Line 2968:       Print "✓ Token sequences match" on success
```

### Phase 5: C++ LOGITS EVALUATION (Lines 2971-2982) ⚠️ **BUG LOCATION 3**
```
Line 2971:       Comment: "Evaluate all positions"
Line 2972:       CALL: cpp_session.context.eval(&cpp_tokens, 0)
                    Location: crates/bitnet-sys/src/wrapper.rs:209
                    Runs C++ forward pass with all tokens
                    n_past=0 (process from beginning)
Line 2973-2975:  CALL: cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())
                    Location: crates/bitnet-sys/src/wrapper.rs:285
                    Iterates 0..n_tokens calling get_logits_ith(i)
                    Returns Vec<Vec<f32>> [positions][vocab_logits]
Line 2977-2981:  Print C++ logits dimensions
```

### Phase 6: LOGITS COMPARISON (Lines 2984-2986)
```
Line 2985:       Print "📊 Comparing logits per position..."
Line 2986:       CALL: compare_per_position_logits(&rust_logits, &cpp_logits)
                    Location: crossval/src/logits_compare.rs:49
                    - Iterates through positions 0..min(len(rust), len(cpp))
                    - Calculates cosine_similarity(rs_vec[pos], cpp_vec[pos])
                    - Calculates l2_distance(rs_vec[pos], cpp_vec[pos])
                    - Tracks first position where cosine < (1 - COSINE_SIMILARITY_THRESHOLD)
                    - Returns LogitsDivergence struct
```

### Phase 7: OUTPUT FORMATTING & REPORTING (Lines 2988-3052)

#### JSON Mode (Lines 2989-3001)
```
Line 2989:       Match on format string
Line 2990-3000:  JSON output with structure:
                 {
                   "first_divergence_token": Option<usize>,
                   "per_token_cosine_sim": Vec<f32>,
                   "per_token_l2_dist": Vec<f32>,
                   "max_absolute_diff": f32,
                   "threshold": f32,
                   "status": "ok" | "diverged"
                 }
Line 3000:       Print JSON via serde_json::to_string_pretty()
```

#### Text Mode (Lines 3002-3050)
```
Line 3004-3016:  Per-position output loop:
                 For each token position:
                 - Print: "✓ t=N cosine=X.XXXXXX l2=X.XXe+N"
                 - If first divergence: print "↑ First divergence at token N"

Line 3019-3020:  Print max absolute difference

Line 3022-3045:  If divergence found:
                 - Print "❌ First divergence at token N"
                 - Print helper commands for trace capture:
                   • Rust trace command (BITNET_TRACE_DIR=/tmp/rs)
                   • C++ trace command (BITNET_TRACE_DIR_CPP=/tmp/cpp)
                   • trace-diff command
                 - std::process::exit(1) [EXIT CODE 1 for divergence]

Line 3046-3048:  If no divergence:
                 - Print "✅ All positions match within tolerance"
                 - Return Ok(())
```

## 5. CRITICAL BUG: EXECUTION ORDER VIOLATION

### Current Buggy Order

```
1. ✓ Parse arguments
2. ✓ Load tokenizer
3. ❌ EVAL RUST LOGITS (lines 2933)       <-- TOO EARLY!
4. ✓ Initialize C++ backend (lines 2954)
5. ✓ Tokenize with C++ (lines 2957)
6. ⚠️  VALIDATE TOKEN PARITY (lines 2963) <-- AFTER Rust logits!
7. ❌ EVAL C++ LOGITS (lines 2972)         <-- WASTED if parity fails!
8. ✓ Compare logits (line 2986)
```

### Specification-Compliant Order (What SHOULD happen)

**From `docs/explanation/token-parity-pregate.md`**:
- Token parity must be validated **before** expensive logits comparisons
- This prevents wasted computation if tokenization differs between Rust and C++

```
1. ✓ Parse arguments
2. ✓ Load tokenizer
3. ✓ Initialize C++ backend
4. ✓ Tokenize both Rust & C++ tokenizers
5. ✓ VALIDATE TOKEN PARITY FIRST
6. ✓ IF PARITY OK: Eval Rust logits
7. ✓ IF PARITY OK: Eval C++ logits
8. ✓ Compare logits
```

### Impact

- **Wasted computation**: If Rust and C++ tokenizers produce different sequences, we've already computed expensive Rust logits before discovering the mismatch
- **For 2B models**: Rust logits evaluation takes ~20-30 seconds; this is completely wasted
- **Silent performance regression**: Users don't notice the inefficiency, especially with smaller models

## 6. Integration Points & Configuration

### A. Tokenizer Integration

**Location**: Line 2920
**Module**: `bitnet_tokenizers::loader`
**Function**: `load_tokenizer(path)`
**Parameters**:
- `tokenizer_path`: Path to tokenizer.json
**Returns**: `Tokenizer` (supports encode/decode)

**Tokenization Parameters** (Line 2921):
```rust
tokenizer.encode(prompt, false, false)
// Parameters: (text, add_special_tokens, add_padding)
```

**Missing flags**:
- No `--prompt-template` support
- No `--add-bos` or `--no-bos` control
- No special token injection control

### B. Model Loading Integration

**Rust Side**:
- **Location**: `crates/bitnet-inference/src/parity.rs:157`
- **Function**: `eval_logits_all_positions(model_path, tokens)`
- **Formats supported**: BitNet I2_S (32-elem blocks), GGML I2_S/QK256 (256-elem blocks), standard formats
- **Device**: Always CPU (hardcoded in parity.rs)
- **No feature gates for device selection**

**C++ Side**:
- **Location**: `crates/bitnet-sys/src/wrapper.rs:344-348`
- **Function**: `Session::load_deterministic(model_path)`
- **Settings**: Hardcoded deterministic config:
  - n_ctx=2048
  - n_batch=512
  - n_threads=1
- **No customization options**

### C. Logits Comparison Integration

**Location**: `crossval/src/logits_compare.rs:49`
**Function**: `compare_per_position_logits(rs_logits, cpp_logits)`
**Metrics Computed**:
1. Cosine similarity per position
2. L2 distance per position
3. Max absolute difference (element-wise)
4. First divergence detection

**Threshold Constants**:
```rust
const COSINE_SIMILARITY_THRESHOLD: f32 = 1e-4;
// Divergence detected when: (1.0 - cosine_sim) > 1e-4
```

**Note**: CLI `--cos-tol` parameter (line 2997) is only used for OUTPUT formatting in text mode (line 3010), NOT for divergence detection logic. The hardcoded 1e-4 threshold in logits_compare.rs is what actually triggers "divergence" detection.

### D. C++ FFI Availability Check

**Location**: Line 2944
**Function**: `bitnet_sys::is_available()`
**Behavior**:
- Returns `false` if C++ FFI not compiled (missing `--features crossval`)
- Returns `false` if `BITNET_CPP_DIR` not set and C++ not found
- Returns `true` if FFI properly initialized
**Error Exit**: Code 1 (anyhow::bail!)

### E. Backend Initialization & Cleanup

**Location**: Lines 2950-2952
```rust
bitnet_sys::wrapper::init_backend();
let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
```
**Pattern**: RAII guard ensures C++ cleanup even on error

## 7. Error Handling & Exit Codes

| Exit Code | Trigger | Location | Message |
|-----------|---------|----------|---------|
| 0 | Success (text mode, no divergence) | Line 3048 | "✅ All positions match..." |
| 1 | Divergence found (text mode) | Line 3045 | "❌ First divergence at token..." |
| 2 | Token parity mismatch | Line 2966 | "Error: {parity error}" |
| 1 | FFI not available | Line 2945 | "C++ FFI not available..." |
| 1 | Model load error | Various | anyhow::bail! propagates |
| 1 | Tokenization error | Line 2921 | encode() fails |

**Note**: JSON mode does NOT exit on divergence (line 3000 just prints JSON). This may be unintended.

## 8. Missing Flags & Configuration Options

### Current Missing Features

1. **`--prompt-template`**
   - Needed to control template formatting before tokenization
   - C++ reference might use different template than Rust default
   - Would avoid BOS duplication issues
   - Documented in CLAUDE.md but not wired to crossval-per-token

2. **`--cpp-backend`**
   - Would allow selection of CPU vs GPU for C++ inference
   - Currently hardcoded to C++ FFI defaults
   - Needed for fair GPU comparison

3. **`--no-bos` / `--add-bos`**
   - Control BOS token injection
   - Common source of token sequence mismatches
   - Not exposed to CLI

4. **`--cos-similarity-threshold`**
   - Override hardcoded 1e-4 threshold in logits_compare.rs
   - Currently only `--cos-tol` exists but doesn't affect divergence detection
   - Confusing dual-threshold behavior

5. **`--max-positions`**
   - Limit evaluation to first N positions for faster testing
   - Useful for debugging early divergences

6. **`--seed`**
   - Not exposed (though parity.rs uses deterministic CPU)
   - C++ session hardcoded to deterministic (threads=1)

7. **`--device`** (cpu|gpu)
   - Force device for Rust evaluation
   - Currently always CPU in parity.rs
   - C++ always uses FFI default

8. **`--trace-dir`**
   - Would enable automatic trace capture on divergence
   - Currently user must manually copy-paste commands

## 9. Diagnostic & Logging Integration Points

### Console Output Phases

| Phase | Lines | Output | Color |
|-------|-------|--------|-------|
| Header | 2912-2916 | Model path, prompt, tolerance | Plain |
| Tokenization | 2919, 2926 | "📝 Tokenizing...", token count | Emoji |
| Rust eval | 2930, 2934-2938 | "🦀 Evaluating...", rust dims | Emoji |
| C++ init | 2941 | "🔧 Evaluating..." | Emoji |
| Parity check | 2960, 2968 | "🔒 Validating...", "✓ Token match" | Emoji + color |
| Parity error | 2965 | Error message (red, detailed) | `token_parity::format_token_mismatch_error` |
| Per-position output | 3004-3016 | "✓/✗ t=N cosine=X l2=X" | Symbol varies |
| Final results | 3020-3048 | "Max diff:", "❌ Divergence" or "✅ Match" | Red/green emoji |
| Helper commands | 3026-3043 | Trace capture instructions | Plain code block |

### Stderr vs Stdout

- **Stderr**: Token parity diagnostic errors (eprintln! on line 2965)
- **Stderr**: Rust logits DEBUG lines (if parity.rs debug flags enabled)
- **Stdout**: All other output (println!)

### No Tracing Integration

- No `BITNET_TRACE_DIR` support
- No per-operation timing
- No intermediate state dumps
- User must manually implement trace capture from printed instructions

## 10. Detailed Component Analysis

### A. eval_logits_all_positions() - Rust Side

**Location**: `crates/bitnet-inference/src/parity.rs:157-223`

**Execution Steps**:
1. Load GGUF model via `load_gguf_full(model_path, Device::Cpu, ...)`
2. Convert i2s_qk256 raw tensors to Candle format
3. Remap GGUF tensor keys to model weight keys
4. Initialize model: `BitNetModel::from_gguf(...)`
5. Convert i32 tokens to u32
6. Get embeddings: `model.embed(&tokens_u32)`
7. Create KV cache: `KVCache::new(&config, 1, Device::Cpu)`
8. Run forward pass: `model.forward(&embedded, cache)`
9. Extract logits: `model.logits(&output)`
10. Extract per-position: `extract_all_position_logits(logits_tensor, seq_len)`

**Output**: `Vec<Vec<f32>>` where dims are [seq_len][vocab_size]

**Error Handling**: anyhow::bail! on any step failure (fail-closed, no FFI routing)

**Performance**: ~20-30s for 2B model on CPU (reason for parity pre-gate optimization)

### B. validate_token_parity() - Pre-Gate

**Location**: `crossval/src/token_parity.rs:79-110`

**Algorithm**:
1. Convert C++ i32 tokens to u32
2. Compare slices element-by-element
3. If mismatch: find_first_diff() returns index
4. Format error message via format_token_mismatch_error()
5. Print to stderr
6. Return Err(anyhow!())

**Output on Success**: `Ok(())`, silent (no stderr)

**Output on Failure**: Formatted error with:
- Token sequence mismatch header
- Both token vectors (limited to 64 tokens for readability)
- First diff index
- Suggested fixes (--prompt-template raw, --no-bos, etc.)
- Example command with actual prompt

### C. compare_per_position_logits() - Comparison

**Location**: `crossval/src/logits_compare.rs:49-102`

**Algorithm Per Position**:
```
for pos in 0..n_positions:
  cosine_sim = dot_product(rs[pos], cpp[pos]) / (norm(rs[pos]) * norm(cpp[pos]))
  l2_dist = sqrt(sum((rs[pos] - cpp[pos])^2))
  max_abs_diff = max(max_abs_diff, max(abs(rs[pos] - cpp[pos])))
  
  if (1.0 - cosine_sim) > COSINE_SIMILARITY_THRESHOLD (1e-4):
    if first_divergence == None:
      first_divergence = Some(pos)
```

**Output**: `LogitsDivergence` struct containing:
- `first_divergence_token: Option<usize>`
- `per_token_cosine_sim: Vec<f32>`
- `per_token_l2_dist: Vec<f32>`
- `max_absolute_diff: f32`

**Note**: The `cos_tol` CLI parameter is only used for display (line 3010), not for divergence detection!

### D. Session::load_deterministic() - C++ FFI

**Location**: `crates/bitnet-sys/src/wrapper.rs:343-348`

**Hardcoded Settings**:
```rust
Model::load(model_path)?;
Context::new(&model, 2048, 512, 1)?;
// Parameters: (model, n_ctx, n_batch, n_threads)
```

**Determinism Constraints**:
- Single thread (n_threads=1)
- No random seeding exposed
- Greedy sampling if used

**No Configuration Options**: Cannot override context size, batch size, or thread count

### E. Context::eval() & get_all_logits()

**Location**: `crates/bitnet-sys/src/wrapper.rs:209, 285-293`

**eval() Behavior** (Line 2972):
- Calls C++ forward pass on given tokens
- `n_past` parameter allows incremental evaluation (set to 0 for full prefill)
- Updates internal KV cache
- No explicit "logits_all" mode needed

**get_all_logits() Behavior** (Line 2975):
```rust
pub fn get_all_logits(&self, n_tokens: usize) -> Result<Vec<Vec<f32>>> {
    let mut all_logits = Vec::with_capacity(n_tokens);
    for i in 0..n_tokens {
        all_logits.push(self.get_logits_ith(i as i32)?);
    }
    Ok(all_logits)
}
```
- Iterates through 0..n_tokens
- Calls get_logits_ith(i) for each position
- Returns Vec<Vec<f32>>

## 11. Feature Dependencies

### Required Features

```toml
# Must have at least one of:
--features cpu                    # Enables bitnet-inference, CPU kernels
--features gpu                    # Enables CUDA kernels

# For C++ FFI:
--features crossval-all           # Unified: enables inference + crossval + ffi
OR
--features inference,crossval,ffi # Individual

# Without these, CFG blocks prevent compilation:
#[cfg(feature = "inference")]
fn crossval_per_token_cmd(...)    # Entire command gated!
```

### Conditional Compilation

- Command only available when `feature = "inference"` enabled
- C++ FFI check at runtime: `bitnet_sys::is_available()`
- If FFI unavailable: error message suggesting `--features crossval`

## 12. Test Infrastructure

### Test File
`/home/steven/code/Rust/BitNet-rs/xtask/tests/crossval_token_parity.rs`

### Current Test Coverage

1. **Token parity tests** (token_parity.rs):
   - AC1-AC10: 9 tests for parity validation
   - ~4 tests marked #[ignore] (TODO: stderr capture)
   - Test scenarios: duplicate BOS, length mismatch, empty sequences, edge cases

2. **Logits comparison tests** (logits_compare.rs):
   - Cosine similarity tests (identical, orthogonal vectors)
   - L2 distance tests
   - Per-position comparison tests
   - ~3 tests in module

3. **Integration tests**:
   - TODO: End-to-end with real models (blocked by issue #254, #260)

### Ignored Tests Pattern

```rust
#[test]
#[ignore] // TODO: Capture stderr to validate error output format
fn test_error_displays_both_sequences()
```

Reasons for ignoring:
- Requires stderr capture test harness
- Subprocess exit code validation needed for AC4
- Known model quality issues block real inference tests

## 13. Known Issues & Limitations

### Issue #254: Shape Mismatch in Layer-Norm
**Impact**: Blocks real inference tests
**Workaround**: Tests use mock or synthetic data

### Issue #260: Mock Elimination
**Impact**: Test infrastructure still contains mock paths
**Workaround**: Temporary until refactoring complete

### Issue #469: Tokenizer Parity & FFI Build Hygiene
**Impact**: Blocks cross-validation test completion
**Status**: Active development

### Execution Order Bug (THIS REPORT'S FOCUS)
**Impact**: Wasted Rust logits computation before parity check
**Lines**: 2933 (Rust eval) before 2963 (parity check)
**Severity**: Medium (performance, not correctness)
**Fix Effort**: Low (reorder function calls + update comments)

### CLI Parameter Inconsistency
**Issue**: `--cos-tol` parameter only affects display, not divergence detection
**Root Cause**: Hardcoded `COSINE_SIMILARITY_THRESHOLD = 1e-4` in logits_compare.rs
**Impact**: User expects `--cos-tol 0.95` to use 0.95 threshold; actually uses 1e-4

### Missing Template Support
**Impact**: Cannot test different tokenization paths (raw, instruct, llama3-chat)
**Workaround**: Pre-tokenize with different tools and compare manually

## 14. Documentation Cross-References

### Specification Documents
- `docs/explanation/token-parity-pregate.md` - Token parity design spec
- `docs/explanation/cpp-eval-with-tokens-*.md` - C++ evaluation details
- `docs/howto/cpp-setup.md` - C++ reference setup

### Related Code
- `CLAUDE.md` - Project status and crossval commands
- `crossval/src/token_parity.rs` - Token parity implementation
- `crossval/src/logits_compare.rs` - Logits comparison implementation
- `crates/bitnet-inference/src/parity.rs` - Rust logits evaluation

## 15. Recommendations for Enhancement

### Critical Fixes (Correctness/Performance)

1. **Fix Execution Order (HIGH PRIORITY)**
   - Move Rust logits eval (line 2933) to after parity check (line 2968)
   - Savings: 20-30s per divergent token sequence
   - Update comments to reflect specification compliance

2. **Fix CLI Threshold Confusion**
   - Remove `--cos-tol` or make it control actual divergence threshold
   - Alternatively: document that `--cos-tol` is display-only
   - Consider adding `--divergence-threshold` parameter

### Feature Additions

3. **Add `--prompt-template` Support**
   - Wire to bitnet_tokenizers::loader configuration
   - Allow raw, instruct, llama3-chat templates
   - Document interaction with Rust vs C++ template selection

4. **Add `--cpp-backend` Parameter**
   - Control GPU vs CPU for C++ inference
   - Currently hardcoded to C++ default

5. **Add `--no-bos` / `--add-bos` Flags**
   - Control BOS token injection
   - Common source of tokenization mismatches

6. **Add Per-Position Limit**
   - `--max-positions N` to evaluate only first N positions
   - Useful for debugging early divergences

7. **Add Automatic Trace Capture**
   - `--auto-trace` flag to capture traces on divergence
   - Eliminates manual copy-paste of trace commands

### Testing & Validation

8. **Implement Stderr Capture Tests**
   - Complete AC2, AC4 test scenarios
   - Validate error message format and exit codes

9. **Add Real Model Tests**
   - Unblock by resolving #254, #260, #469
   - Test with bitnet-b1.58-2B model suite

10. **Add Performance Benchmarks**
    - Measure eval time per model size
    - Track regression in parity overhead

## Appendix A: Complete Call Stack

```
xtask/src/main.rs:897
  └─> crossval_per_token_cmd()  [line 2901]
        ├─> bitnet_tokenizers::loader::load_tokenizer()  [line 2920]
        │
        ├─> bitnet_inference::parity::eval_logits_all_positions()  [line 2933]
        │     ├─> bitnet_models::load_gguf_full()
        │     ├─> BitNetModel::from_gguf()
        │     ├─> model.embed(&tokens)
        │     ├─> model.forward(&embedded, cache)
        │     └─> extract_all_position_logits()  [parity.rs:369]
        │
        ├─> bitnet_sys::is_available()  [line 2944]
        │
        ├─> bitnet_sys::wrapper::init_backend()  [line 2951]
        │
        ├─> bitnet_sys::wrapper::Session::load_deterministic()  [line 2954]
        │     ├─> Model::load()
        │     └─> Context::new()
        │
        ├─> Session::tokenize()  [line 2957]
        │
        ├─> bitnet_crossval::token_parity::validate_token_parity()  [line 2963]
        │     └─> find_first_diff()  [token_parity.rs:116]
        │         format_token_mismatch_error()  [token_parity.rs:148]
        │
        ├─> Context::eval()  [line 2972]
        │
        ├─> Context::get_all_logits()  [line 2975]
        │     └─> get_logits_ith(i)  [wrapper.rs:254] (per position)
        │
        ├─> bitnet_crossval::logits_compare::compare_per_position_logits()  [line 2986]
        │     ├─> cosine_similarity()  [logits_compare.rs:107]
        │     └─> l2_distance()  [logits_compare.rs:126]
        │
        └─> Output formatting [lines 2989-3050]
              ├─> JSON: serde_json::to_string_pretty()
              └─> Text: println! loops + exit code
```

## Appendix B: Data Structures

### Input Arguments (clap-parsed)
```rust
model: PathBuf              // ./models/model.gguf
tokenizer: PathBuf         // ./models/tokenizer.json
prompt: &str              // "What is 2+2?"
max_tokens: usize         // 4 (UNUSED)
cos_tol: f32             // 0.999
format: &str             // "text" or "json"
```

### Tokenization Output
```rust
tokens: Vec<u32>          // [128000, 1229, 374, 220, 17]
token_ids: Vec<i32>       // Same, but i32
cpp_tokens: Vec<i32>      // [128000, 1229, 374, 220, 17] from C++
```

### Rust Logits Output
```rust
rust_logits: Vec<Vec<f32>> = [
  [0.1, 0.2, ..., -0.5],    // Position 0, vocab_size logits
  [0.3, 0.1, ..., 0.2],     // Position 1, vocab_size logits
  [0.0, 0.4, ..., 0.1],     // Position 2, vocab_size logits
  ...
]
```

### Divergence Result
```rust
LogitsDivergence {
  first_divergence_token: Option<usize>,  // Some(2) or None
  per_token_cosine_sim: Vec<f32>,        // [1.0, 0.9999, 0.8765, ...]
  per_token_l2_dist: Vec<f32>,           // [0.0, 1.2e-4, 0.45, ...]
  max_absolute_diff: f32,                // 0.67
}
```

### JSON Output
```json
{
  "first_divergence_token": 2,
  "per_token_cosine_sim": [1.0, 0.9999, 0.8765],
  "per_token_l2_dist": [0.0, 1.2e-4, 0.45],
  "max_absolute_diff": 0.67,
  "threshold": 0.999,
  "status": "diverged"
}
```

---

**Report Generated**: Analysis of BitNet-rs `xtask crossval-per-token` implementation  
**Status**: Complete with bug identification  
**Critical Issue**: Execution order violation (lines 2933 vs 2963)  
**Recommendation**: Reorder to evaluate token parity before expensive logits computations

