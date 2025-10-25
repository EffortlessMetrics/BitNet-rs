# C++ FFI Exploration - Complete Index

## Overview

Comprehensive analysis of the C++ FFI interface in bitnet-crossval for implementing direct token ID evaluation with per-position logits support.

**Status**: Complete - All research findings documented and ready for implementation.

---

## Key Findings (TL;DR)

### 1. Direct Token ID Interface Already Exists
- `bitnet-sys` crate has full token ID evaluation support
- `Session::eval_and_get_logits(&[i32])` accepts pre-tokenized sequences
- Custom BitNet C shim provides `bitnet_eval_tokens()` for bypass
- **No implementation required** - API is production-ready

### 2. Per-Position Logits Need Minimal Work
- `Context::get_all_logits()` exists (line 285 of wrapper.rs)
- Just needs public wrapper in Session API
- 5-line addition to bitnet-sys/src/wrapper.rs
- **Effort: 1 hour**

### 3. Crossval Integration Opportunity
- crossval currently uses mock C wrapper
- per_position_logits.rs tests already import bitnet_sys
- Replacing mock with real Session would unify the stack
- **Effort: 2-3 hours**

---

## Documentation Structure

### 1. FFI_INTERFACE_ANALYSIS.md (471 lines, 16 KB)
**Complete architectural analysis**

Contents:
- Current API analysis with full context
- String-based vs token-based flow comparison
- Session/Context/Model struct breakdown
- Custom BitNet C shim API documentation
- Per-position logits comparison methodology
- Architecture diagram (ASCII)
- Detailed effort assessment (3 phases)
- Key files reference guide

**When to read**: Need complete understanding of FFI architecture

### 2. FFI_QUICK_REFERENCE.md (79 lines, 2.9 KB)
**One-page cheat sheet**

Contents:
- Current state summary
- Direct Token ID API (exists!)
- Per-Position Logits Wrapper (what's missing)
- Logits comparison format
- Integration roadmap table
- Key files (quick lookup)
- Gotchas (4 common issues)
- Next steps checklist

**When to read**: Need quick reference during implementation

### 3. FFI_FUNCTION_SIGNATURES.md (296 lines, 8 KB)
**Exact Rust function signatures and usage**

Contents:
- Session API (public)
- Context API (lower-level)
- Model API (loading)
- Custom BitNet C Shim API (with line numbers)
- Utility functions
- 3 usage examples with code
- Return type breakdown
- Function call patterns
- Integration code snippets

**When to read**: Implementing token ID wrapper functions

---

## Implementation Roadmap

### Phase 1: Per-Position Logits (LOW EFFORT - 1 hour)

**Goal**: Surface `get_all_logits()` through Session API

**File**: `crates/bitnet-sys/src/wrapper.rs`

**Change**: Add 5 lines to Session impl
```rust
pub fn eval_and_get_all_logits(&mut self, tokens: &[i32], n_past: i32) -> Result<Vec<Vec<f32>>> {
    self.context.eval(tokens, n_past)?;
    self.context.get_all_logits(tokens.len())
}
```

**Test**: Update `crossval/tests/per_position_logits.rs` to use new method

**Verification**: All cosine similarities should remain > 0.9999 for multi-position

### Phase 2: Update Tests (LOW EFFORT - 1-2 hours)

**Goal**: Make full use of per-position API in existing tests

**Files**: 
- `crossval/tests/per_position_logits.rs` - main test file
- `crossval/src/logits_compare.rs` - comparison infrastructure (ready)

**Changes**:
1. Line 52-53: Replace single-position eval with new method
2. Collect all positions in Vec<Vec<f32>>
3. Pass to compare_per_position_logits()
4. Verify multi-position divergence detection

### Phase 3: Crossval Integration (MEDIUM EFFORT - 2-3 hours, OPTIONAL)

**Goal**: Replace mock C wrapper with real bitnet-sys::Session

**Files**:
- `crossval/src/cpp_bindings.rs` - mock wrapper (224 lines)
- `crossval/src/bitnet_cpp_wrapper.c` - C stub (remove)
- `crossval/Cargo.toml` - dependencies

**Changes**:
1. Replace CppModel.handle with bitnet_sys::wrapper::Session
2. Update generate() to use Session API
3. Add token evaluation methods
4. Remove bitnet_cpp_wrapper.c

**Impact**: Unified FFI stack, real C++ backend for comparison

---

## Specific Function Signatures

### Session API (Ready to use)

```rust
pub fn eval_and_get_logits(&mut self, tokens: &[i32], n_past: i32) -> Result<Vec<f32>>
pub fn tokenize(&self, text: &str) -> Result<Vec<i32>>
pub fn decode(&self, tokens: &[i32]) -> Result<String>
pub fn generate_greedy(&mut self, prompt: &str, max_tokens: usize) -> Result<Vec<i32>>
```

### Context API (Existing, needs wrapping)

```rust
pub fn eval(&mut self, tokens: &[i32], n_past: i32) -> Result<()>
pub fn get_logits(&self) -> Result<Vec<f32>>
pub fn get_all_logits(&self, n_tokens: usize) -> Result<Vec<Vec<f32>>>  // ← KEY!
pub fn get_logits_ith(&self, i: i32) -> Result<Vec<f32>>
```

### Custom BitNet C Shim (Optional for later)

```rust
pub fn bitnet_eval_tokens(ctx: &BitnetContext, ids: &[i32], vocab_size: usize) -> Result<Vec<f32>>
pub fn bitnet_prefill(ctx: &BitnetContext, ids: &[i32]) -> Result<()>
pub fn cpp_decode_greedy(..., out: &mut [i32]) -> Result<usize>
pub fn bitnet_tokenize_text(model: &BitnetModel, text: &str, ...) -> Result<Vec<i32>>
```

---

## Return Types

### Token IDs
- Input: `&[i32]` (i32 for llama.cpp compatibility)
- Output: `Vec<i32>`
- Conversion: `tokens.iter().map(|t| *t as u32).collect()`

### Logits
- **Single position**: `Vec<f32>` (length = vocab_size)
- **Multiple positions**: `Vec<Vec<f32>>` (outer = positions, inner = vocab)
- **Comparison input**: `&[Vec<f32>]` for both Rust and C++ implementations

### Logits Comparison Output

```rust
pub struct LogitsDivergence {
    pub first_divergence_token: Option<usize>,    // Which position diverged first
    pub per_token_cosine_sim: Vec<f32>,          // Similarity per position
    pub per_token_l2_dist: Vec<f32>,             // Distance per position
    pub max_absolute_diff: f32,                   // Largest element-wise diff
}
```

---

## File Locations Reference

### FFI Definition
- **Main wrapper**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/src/wrapper.rs` (660 lines)
  - Session: lines 330-394
  - Context: lines 120-327
  - Model: lines 58-117
  - Custom BitNet: lines 400-659

- **Exports**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/src/lib.rs`

### Crossval Current
- **Mock wrapper**: `crossval/src/cpp_bindings.rs` (224 lines)
- **C stub**: `crossval/src/bitnet_cpp_wrapper.c` (64 lines)
- **Comparison logic**: `crossval/src/logits_compare.rs`

### Tests Using FFI
- **Main test**: `crossval/tests/per_position_logits.rs` (295 lines)
  - Already imports bitnet_sys::wrapper
  - Uses Session::eval_and_get_logits()
  - Could be extended to per-position

- **FFI integration tests**: `crossval/tests/ffi_integration.rs`
- **Token equivalence**: `crossval/tests/token_equivalence.rs`

### Inference
- **Rust logits**: `crates/bitnet-inference/src/parity.rs`
  - Function: `eval_logits_once(&str, &[i32]) -> Vec<f32>`

---

## Common Gotchas

1. **Token ID type mismatch**: bitnet-sys uses `i32`, crossval returns `u32`
   - Fix: `tokens.iter().map(|t| *t as i32).collect()`

2. **Logits return format**: Most APIs return single position only
   - Solution: Use Context::get_all_logits() directly or add wrapper

3. **Decoupled crossval**: Uses mock C wrapper, doesn't leverage bitnet-sys
   - Opportunity: Replace with real Session

4. **KV cache management**: BitNet shim may require bitnet_prefill() before eval
   - Check: Does bitnet_eval_tokens() call prefill internally?

---

## Quick Implementation Checklist

- [ ] Review FFI_INTERFACE_ANALYSIS.md sections 1-2 (current API)
- [ ] Read FFI_FUNCTION_SIGNATURES.md usage examples
- [ ] Add Session::eval_and_get_all_logits() wrapper (5 lines)
- [ ] Update per_position_logits.rs test (collect all positions)
- [ ] Verify multi-position comparison works
- [ ] (Optional) Replace crossval mock with bitnet_sys::Session

---

## Questions & Answers

**Q: Do we need to write new C++ code to accept token IDs?**
A: No! Both Session and custom BitNet shim already accept `&[i32]` directly.

**Q: What's the difference between Session and BitNet C Shim?**
A: Session wraps llama.cpp API; BitNet shim is a custom implementation for BitNet-specific optimizations.

**Q: How do we get per-position logits?**
A: Context::get_all_logits() exists but isn't exposed. Add 5-line Session wrapper.

**Q: What effort to implement token ID interface?**
A: 1-3 hours (mostly low-effort wrapping of existing code; per-position is 1h)

**Q: Is crossval currently using real C++ backend?**
A: No, it uses a mock C wrapper. The per_position_logits.rs test imports bitnet_sys but could use it better.

---

## Related Documentation

- `docs/FFI_TRACE_INFRASTRUCTURE.md` - FFI tracing framework
- `docs/INFERENCE_ENGINE_LAYER_ANALYSIS.md` - Inference architecture
- `CLAUDE.md` - Project guidelines and status

---

## Navigation

| Need | Document | Link |
|------|----------|------|
| Complete understanding | FFI_INTERFACE_ANALYSIS.md | Full 471-line breakdown |
| Quick reference | FFI_QUICK_REFERENCE.md | One-page cheat |
| Code signatures | FFI_FUNCTION_SIGNATURES.md | With usage examples |
| Implementation guide | This file (roadmap section) | Above |

---

**Last Updated**: 2025-10-25
**Coverage**: bitnet-sys FFI, crossval integration, per-position logits API
**Status**: Ready for Phase 1 implementation
