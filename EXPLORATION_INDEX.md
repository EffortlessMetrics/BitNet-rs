# Backend Routing Exploration: Complete File Index

This index documents all files analyzed during the backend routing exploration for the `crossval-per-token` command.

## Documentation Generated

Three comprehensive analysis documents have been created in the repository root:

1. **BACKEND_ROUTING_ANALYSIS.md** (15 KB)
   - Complete architectural analysis
   - 8-step execution flow with status markers
   - Call site identification
   - Error handling patterns
   - Preflight validation details
   - Implementation roadmap (3 phases)
   - Summary table with component status

2. **QUICK_IMPLEMENTATION_GUIDE.md** (8.1 KB)
   - TL;DR implementation checklist
   - Exact code patterns to use
   - Minimal change strategy
   - Testing procedures
   - Variable scope considerations
   - Verification checklist

3. **DISPATCHER_ARCHITECTURE_SUMMARY.md** (This file)
   - Visual ASCII architecture diagram
   - Backend infrastructure overview
   - Two routing points clearly identified
   - Component status matrix
   - Implementation checklist (3 phases)
   - Key insights and strategy

## Source Files Analyzed

### Command Handler
- `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs`
  - Lines 437-483: CLI argument definition for CrossvalPerToken
  - Lines 2974-3189: crossval_per_token_cmd() handler function
  - Lines 2992-3004: Backend selection and preflight validation
  - Lines 3050-3080: **[ROUTING POINT #1]** C++ tokenization (hardcoded)
  - Lines 3104-3118: **[ROUTING POINT #2]** C++ logits evaluation (hardcoded)

### Backend Infrastructure
- `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/backend.rs`
  - Lines 10-106: CppBackend enum definition and methods
  - Lines 50-61: Auto-detection from model path
  - Lines 72-106: Metadata methods (name, required_libs, setup_command)
  - Lines 108-138: Unit tests

- `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/preflight.rs`
  - Lines 34-79: preflight_backend_libs() function
  - Lines 85-121: Backend status diagnostics

- `/home/steven/code/Rust/BitNet-rs/xtask/src/crossval/mod.rs`
  - Module exports (backend, preflight)

### Cross-Validation Framework
- `/home/steven/code/Rust/BitNet-rs/crossval/src/lib.rs`
  - Lines 1-70: Module documentation and exports
  - Lines 73-93: assert_first_logits_match() helper
  - Lines 101-116: Backend detection tests

- `/home/steven/code/Rust/BitNet-rs/crossval/src/token_parity.rs`
  - Token sequence validation (used in crossval-per-token)

- `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs`
  - Per-position logits comparison (used in crossval-per-token)

### Inference Parity Module
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs`
  - Lines 14-111: eval_logits_all_positions() function
  - Pure Rust logits evaluation for all token positions
  - Supports BitNet I2_S and GGML I2_S (QK256) formats

- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/lib.rs`
  - Module exports (parity module)

## Key Findings Summary

### Infrastructure Status: 90% Complete

**Working (✅)**:
- CLI argument parsing with CppBackend enum
- Backend auto-detection from model path
- Preflight library validation
- Rust tokenization (universal tokenizer)
- Token parity validation
- Rust logits evaluation (pure Rust, all formats)
- Logits comparison and diagnostics

**Not Implemented (❌)**:
- C++ tokenization dispatch (hardcoded to llama.cpp)
- C++ logits evaluation dispatch (hardcoded to llama.cpp)
- BitNet.cpp FFI wrappers

### Call Sites Requiring Backend Dispatch

**Call Site #1: C++ Tokenization**
- File: `xtask/src/main.rs`
- Line: ~3074
- Current: `bitnet_sys::wrapper::Session::load_deterministic(...)`
- Need: Match on `backend` with arms for BitNet and Llama

**Call Site #2: C++ Logits Evaluation**
- File: `xtask/src/main.rs`
- Line: ~3108
- Current: `cpp_session.context.eval(...)`
- Need: Match on `backend` with arms for BitNet and Llama

### No Changes Needed

These areas are already backend-agnostic and require no modification:
- Prompt formatting and template application
- Rust tokenization
- Token parity validation
- Rust logits evaluation
- Logits comparison
- Output formatting (text/JSON)

---

## Implementation Phases

### Phase 1: Insert Backend Match Statements (IMMEDIATE)
**Effort**: 1-2 hours
**Risk**: Minimal (preserves existing behavior)
**Outcome**: Infrastructure ready for Phase 2

Detailed steps in QUICK_IMPLEMENTATION_GUIDE.md

### Phase 2: Implement BitNet.cpp FFI Wrappers (FUTURE)
**Effort**: TBD (depends on bitnet.cpp FFI availability)
**Risk**: Moderate (new FFI integration)
**Outcome**: Full dual-backend support

Expected wrappers:
- `tokenize_bitnet(model_path: &str, prompt: &str) -> Result<Vec<u32>>`
- `evaluate_logits_bitnet(model_path: &str, tokens: &[u32]) -> Result<Vec<Vec<f32>>>`

### Phase 3: Optimization & Polish (OPTIONAL)
**Effort**: 1-2 hours
**Risk**: Low (optional enhancements)
**Outcome**: Better diagnostics and performance

Planned enhancements:
- Timing metrics per backend
- Verbose FFI call tracing
- Session initialization optimization

---

## Testing Strategy

### Regression Testing (Phase 1)
```bash
# Test with llama model (existing functionality)
cargo run -p xtask -- crossval-per-token \
  --model models/llama-3-8b-instruct.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" --max-tokens 4 --verbose
```
Expected: Works exactly as before

### Placeholder Testing (Phase 1)
```bash
# Test with bitnet model (should fail gracefully)
cargo run -p xtask -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" --max-tokens 4
```
Expected: Panics with "BitNet.cpp tokenization not yet implemented"

### Auto-Detection Testing (Phase 1)
```bash
# Test auto-detection (no --cpp-backend flag)
cargo run -p xtask -- crossval-per-token \
  --model models/llama-3-8b-instruct.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" --verbose
```
Expected: Prints "Selected backend: llama.cpp (auto-detected)"

### Full Integration (Phase 2+)
Once BitNet.cpp FFI wrappers are implemented:
```bash
# Test with BitNet model using explicit backend
cargo run -p xtask -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" --cpp-backend bitnet
```
Expected: Full per-token parity validation

---

## Related Code Patterns

### Existing Backend Selection (Already Working)
```rust
let backend = cpp_backend.unwrap_or_else(|| CppBackend::from_model_path(model_path));
crate::crossval::preflight_backend_libs(backend, verbose)?;
```

### Current llama.cpp Usage Pattern (To Be Refactored)
```rust
bitnet_sys::wrapper::init_backend();
let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());
let mut cpp_session = bitnet_sys::wrapper::Session::load_deterministic(model_path_str)?;
let cpp_tokens = cpp_session.tokenize(&formatted_prompt)?;
cpp_session.context.eval(&cpp_tokens, 0)?;
let cpp_logits = cpp_session.context.get_all_logits(cpp_tokens.len())?;
```

### Error Handling Pattern (Already Established)
```rust
use anyhow::{Result, bail};

// Parse with context
let model_path_str = model_path.to_str()
    .ok_or_else(|| anyhow::anyhow!("Invalid model path"))?;

// Propagate with ?
let result = some_operation()?;

// Bail with diagnostic message
bail!("Helpful error message");
```

---

## File Size Reference

| File | Size | Status |
|------|------|--------|
| BACKEND_ROUTING_ANALYSIS.md | 15 KB | Reference documentation |
| QUICK_IMPLEMENTATION_GUIDE.md | 8.1 KB | Implementation guide |
| DISPATCHER_ARCHITECTURE_SUMMARY.md | This file | Overview |
| xtask/src/main.rs | 31 KB (full file) | **Requires modification** |
| xtask/src/crossval/backend.rs | 3.5 KB | Ready to use |
| xtask/src/crossval/preflight.rs | 3.6 KB | Ready to use |
| crossval/src/lib.rs | 4 KB | Ready to use |

---

## Dependency Map

```
crossval-per-token command
├─ CppBackend enum (backend.rs)
│  ├─ Auto-detection logic
│  ├─ Metadata methods
│  └─ Unit tests
├─ Preflight validation (preflight.rs)
│  ├─ Library detection
│  └─ Setup instructions
├─ Rust tokenization (bitnet_tokenizers)
│  └─ Universal tokenizer
├─ C++ FFI wrapper (bitnet_sys::wrapper)
│  ├─ Session management
│  ├─ Tokenization
│  └─ Logits evaluation
├─ Token parity (crossval::token_parity)
│  └─ Sequence validation
├─ Rust logits (bitnet_inference::parity)
│  └─ Pure Rust evaluation (all formats)
├─ Logits comparison (crossval::logits_compare)
│  └─ Per-position comparison
└─ Output formatting (serde_json)
   └─ JSON/text output
```

---

## Next Steps

1. **Review QUICK_IMPLEMENTATION_GUIDE.md** - Start here for implementation
2. **Review BACKEND_ROUTING_ANALYSIS.md** - Detailed reference
3. **Execute Phase 1** - Insert match statements (1-2 hours)
4. **Test with llama models** - Verify no regression
5. **Document Phase 2 requirements** - BitNet.cpp FFI wrappers

---

## Questions This Exploration Answers

1. ✅ Where are C++ calls currently made?
   - Line 3074 (tokenization), Line 3108 (logits)

2. ✅ Which wrapper functions are being called?
   - `Session::load_deterministic()`, `session.tokenize()`, `session.context.eval()`, `context.get_all_logits()`

3. ✅ What's the current error handling pattern?
   - anyhow::Result with context, scopeguard for cleanup

4. ✅ Where should the backend match go?
   - Right before the C++ calls (2 locations)

5. ✅ What function signatures must we match?
   - Tokenization: `&str, &str -> Result<Vec<u32>>`
   - Logits: `&[u32] -> Result<Vec<Vec<f32>>>`

6. ✅ What crossval exports are available?
   - token_parity::validate_token_parity(), logits_compare::compare_per_position_logits()

7. ✅ Is backend infrastructure already in place?
   - Yes, 90% complete. Only C++ dispatch needed.

8. ✅ What needs no changes?
   - Rust paths, comparison logic, output formatting

