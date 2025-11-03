# xtask crossval-per-token Command: Critical Analysis Summary

## Quick Facts

- **Command Location**: `xtask/src/main.rs` lines 389-430 (CLI def), 897-899 (dispatch), 2901-3053 (handler)
- **Feature Gate**: `#[cfg(feature = "inference")]` - requires `--features crossval-all` or `--features inference,crossval,ffi`
- **Main Function**: `crossval_per_token_cmd()` at line 2901

## Critical Finding: Execution Order Bug

### The Problem
**Lines 2933 vs 2963: Logits evaluated BEFORE parity validation**

```
CURRENT (BUGGY):
├─ Line 2933: eval_logits_all_positions() ← Rust logits (20-30 seconds)
├─ Line 2954-2957: Init C++, tokenize
├─ Line 2963: validate_token_parity() ← CHECKS TOKENS TOO LATE!
├─ Line 2972-2975: eval C++ logits ← WASTED if parity fails
└─ Line 2986: compare_per_position_logits()

CORRECT (SPEC-COMPLIANT):
├─ Init C++, tokenize (lines 2954-2957)
├─ validate_token_parity() ← MUST BE FIRST
├─ IF PARITY OK: eval_logits_all_positions() (line 2933)
├─ IF PARITY OK: eval C++ logits (line 2972)
└─ compare_per_position_logits() (line 2986)
```

### Impact
- **Wasted Computation**: 20-30 seconds of Rust inference for every divergent tokenization mismatch
- **Specification Violation**: `docs/explanation/token-parity-pregate.md` explicitly requires parity check before expensive comparisons
- **Silent Regression**: Users don't notice the inefficiency; performance degrades without visible warning

### Severity
- **Type**: Performance bug (not correctness)
- **Fix Effort**: Low (reorder 3 blocks of code, ~20 lines)
- **Test Coverage**: Blocked by #254, #260, #469 on real models

---

## Command Signature

```bash
cargo run -p xtask -- crossval-per-token \
  --model <PATH>           # Required: GGUF model file
  --tokenizer <PATH>       # Required: tokenizer.json (must match model)
  --prompt <STRING>        # Required: input text to tokenize
  --max-tokens <N>         # Optional: defaults to 4 (UNUSED - reserved for future)
  --cos-tol <F>           # Optional: defaults to 0.999 (display-only! not used for detection)
  --format <STR>          # Optional: "text" (default) or "json"
```

---

## End-to-End Execution Flow (Correct Order Should Be)

### Phase 1: Parse Arguments
- Input: CLI flags
- Output: PathBuf(model), PathBuf(tokenizer), String(prompt), usize(max_tokens), f32(cos_tol), &str(format)

### Phase 2: Tokenization (Rust + C++)
- **Rust**: `bitnet_tokenizers::loader::load_tokenizer()` → encode() → Vec<i32>
- **C++**: `Session::load_deterministic()` → tokenize() → Vec<i32>

### Phase 3: Token Parity Check (MUST BE FIRST)
- **Function**: `bitnet_crossval::token_parity::validate_token_parity()`
- **Input**: Rust tokens (u32), C++ tokens (i32), prompt (string)
- **Output**: Ok(()) or Err with diagnostic
- **Exit Code**: 2 on token mismatch

### Phase 4: Rust Logits Evaluation
- **Function**: `bitnet_inference::parity::eval_logits_all_positions()`
- **Process**:
  1. Load GGUF model: `bitnet_models::load_gguf_full()`
  2. Create KV cache: `KVCache::new()`
  3. Forward pass: `model.forward(embedded, cache)`
  4. Extract per-position logits: `extract_all_position_logits()`
- **Output**: `Vec<Vec<f32>>` where dims=[seq_len][vocab_size]

### Phase 5: C++ Logits Evaluation
- **Function**: `bitnet_sys::wrapper::Session::eval(tokens) + get_all_logits()`
- **Hardcoded Settings**: n_ctx=2048, n_batch=512, n_threads=1
- **Output**: `Vec<Vec<f32>>` same shape as Rust

### Phase 6: Logits Comparison
- **Function**: `bitnet_crossval::logits_compare::compare_per_position_logits()`
- **Metrics**: Per-position cosine similarity, L2 distance, max absolute difference
- **Divergence Threshold**: Hardcoded 1e-4 (NOT controlled by `--cos-tol`!)
- **Output**: `LogitsDivergence` struct with first_divergence_token, per_token metrics

### Phase 7: Output & Exit
**Text Mode** (default):
- Print per-position metrics: "✓/✗ t=N cosine=X l2=X"
- Exit 0 on full match, 1 on divergence
- Print trace capture instructions if diverged

**JSON Mode**:
- Print JSON object, no exit code on divergence (may be unintended)

---

## Key Integration Points

| Component | Location | Function | Parameters | Output |
|-----------|----------|----------|------------|--------|
| Tokenizer | Line 2920 | `load_tokenizer(path)` | tokenizer.json path | Tokenizer object |
| Rust Eval | Line 2933 | `eval_logits_all_positions(model, tokens)` | Model path, token IDs | Vec<Vec<f32>> |
| C++ FFI | Lines 2954-2957 | `Session::load_deterministic()` + `tokenize()` | Model path, prompt | Vec<i32> tokens |
| Parity Check | Line 2963 | `validate_token_parity(rust, cpp, prompt)` | Both token sequences | Ok(()) or Err |
| C++ Eval | Lines 2972-2975 | `Context::eval() + get_all_logits()` | Token IDs, count | Vec<Vec<f32>> |
| Comparison | Line 2986 | `compare_per_position_logits()` | Both logit matrices | LogitsDivergence |

---

## Error Handling & Exit Codes

| Exit Code | Trigger | Line | Condition |
|-----------|---------|------|-----------|
| 0 | Success | 3048 | Text mode, all positions match |
| 1 | Divergence | 3045 | Text mode, first_divergence_token != None |
| 2 | Token mismatch | 2966 | Parity validation fails |
| 1 | FFI unavailable | 2945 | `!bitnet_sys::is_available()` |
| 1 | Other errors | Various | anyhow::bail! (tokenization, model load, etc.) |

**Note**: JSON mode does NOT exit on divergence (just prints JSON at line 3000). This inconsistency may be unintended.

---

## Missing Flags (Current Gaps)

1. **`--prompt-template`** - Control tokenization template (raw/instruct/llama3-chat)
2. **`--cpp-backend`** - Select C++ device (cpu/gpu)
3. **`--no-bos` / `--add-bos`** - Control BOS injection
4. **`--cos-similarity-threshold`** - Override hardcoded 1e-4 divergence threshold
5. **`--max-positions`** - Limit evaluation to first N positions
6. **`--seed`** - Not exposed (though deterministic by default)
7. **`--device`** - Force Rust device (always CPU currently)
8. **`--trace-dir`** - Automatic trace capture on divergence

---

## Known Issues

### Issue #254: Shape Mismatch in Layer-Norm
- Blocks real inference tests
- Affects multiple architectures
- Status: In analysis phase

### Issue #260: Mock Elimination
- Test infrastructure contains mock paths
- Blocks end-to-end tests
- Status: Awaiting refactoring

### Issue #469: Tokenizer Parity & FFI Build
- Blocks cross-validation completion
- Status: Active development

### THIS REPORT'S BUG: Execution Order Violation
- **Lines**: 2933 (Rust eval) before 2963 (parity check)
- **Type**: Performance bug, not correctness
- **Fix**: Reorder function calls
- **Status**: Needs fixing

### CLI Threshold Confusion
- `--cos-tol` parameter only affects display (line 3010)
- Actual divergence detection uses hardcoded 1e-4 threshold
- Users expect `--cos-tol 0.95` to affect divergence detection; it doesn't

---

## Feature Dependencies

```
Feature Gate: #[cfg(feature = "inference")]
└─ Requires: --features crossval-all 
   OR: --features inference,crossval,ffi
   
C++ FFI Availability: bitnet_sys::is_available()
└─ Checks: 
   • Compiled with --features crossval
   • BITNET_CPP_DIR environment variable
```

---

## Detailed Findings

### A. CLI Definition (Lines 389-430)
- 6 arguments, 3 required (model, tokenizer, prompt)
- 3 optional (max_tokens=4, cos_tol=0.999, format="text")
- max_tokens prefixed with underscore at line 2905 (unused, reserved)
- Feature-gated entire command with `#[cfg(feature = "inference")]`

### B. Argument Dispatch (Lines 897-899)
- Direct pass-through from clap struct to handler
- No transformation or validation

### C. Handler Phases (Lines 2901-3053)
1. Imports (2909-2910): Import logits_compare, eval_logits_all_positions
2. Tokenization (2912-2927): Load tokenizer, encode prompt
3. **Rust logits eval (2929-2938)** ← WRONG LOCATION
4. C++ init (2940-2957): Check FFI, load session, tokenize
5. **Token parity check (2959-2969)** ← SHOULD BE #4
6. **C++ logits eval (2971-2982)** ← CONDITIONAL
7. Comparison (2984-2986): Call compare_per_position_logits()
8. Output (2988-3052): JSON or text formatting + exit codes

---

## Data Flow Structures

### Input Arguments
```rust
PathBuf(model)           // Model GGUF file path
PathBuf(tokenizer)       // Tokenizer JSON file path
&str(prompt)            // Input prompt text
usize(max_tokens)       // Reserved for future (currently unused)
f32(cos_tol)           // Display threshold (0.0-1.0, default 0.999)
&str(format)           // Output format "text" or "json"
```

### Tokenization Output
```rust
Vec<u32>(tokens)         // Rust tokenizer output
Vec<i32>(token_ids)      // Converted to i32 for comparison
Vec<i32>(cpp_tokens)     // C++ tokenizer output
```

### Logits Tensors
```rust
Vec<Vec<f32>>           // [positions][vocab_logits]
// Dimensions: [seq_len][vocab_size]
// Example: [[0.1, 0.2, ..., -0.5], [0.3, 0.1, ..., 0.2], ...]
```

### Comparison Output
```rust
LogitsDivergence {
  first_divergence_token: Option<usize>,  // Some(2) or None
  per_token_cosine_sim: Vec<f32>,        // [1.0, 0.9999, 0.8765]
  per_token_l2_dist: Vec<f32>,           // [0.0, 1.2e-4, 0.45]
  max_absolute_diff: f32,                // 0.67
}
```

---

## Recommendations (Priority Order)

### CRITICAL (Fix Now)
1. **Reorder logits evaluation to AFTER parity check**
   - Move lines 2933-2938 to after line 2968
   - Update comments to reflect spec compliance
   - Saves 20-30s per divergent tokenization case

### HIGH (Add Before v0.2)
2. **Fix CLI threshold confusion**
   - Document that `--cos-tol` is display-only OR
   - Make it control actual divergence threshold OR
   - Add separate `--divergence-threshold` parameter

3. **Add `--prompt-template` support**
   - Wire to tokenizer configuration
   - Allow raw, instruct, llama3-chat templates

4. **Add `--no-bos` / `--add-bos` flags**
   - Common source of tokenization mismatches

### MEDIUM (Future Enhancement)
5. Add `--max-positions` (limit eval to first N positions)
6. Add `--auto-trace` (automatic trace capture on divergence)
7. Add `--cpp-backend` (GPU vs CPU for C++ inference)
8. Complete test coverage (blocked by #254, #260, #469)

---

## Testing Status

- **Unit Tests**: Token parity (9 tests), logits comparison (3 tests)
- **Ignored Tests**: 4 tests marked #[ignore] (require stderr capture)
- **Integration Tests**: Blocked by #254, #260, #469 (real model tests)
- **Pass Rate**: 12/12 on parity unit tests (when not blocked)

---

## Files Modified/Affected

- **Main**: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (lines 2901-3053)
- **Parity Check**: `/home/steven/code/Rust/BitNet-rs/crossval/src/token_parity.rs` (lines 79-110)
- **Comparison**: `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs` (lines 49-102)
- **Rust Eval**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs` (lines 157-223)
- **C++ FFI**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/src/wrapper.rs` (lines 285-293)

---

**Full Report**: `/home/steven/code/Rust/BitNet-rs/docs/explanation/xtask-crossval-per-token-implementation-analysis.md`

**Report Size**: 28KB, comprehensive coverage with line numbers, call stacks, data structures, test status

**Key Deliverable**: Identification of execution order bug + complete specification of all configuration flows

