# C++ FFI Quick Reference for Token ID Interface

## One-Page Summary

### Current State
- **bitnet-sys**: Full token ID support (accepts `&[i32]`)
- **crossval**: Mock C wrapper only (string-based)
- **Both layers**: Return last-position logits only

### Direct Token ID API (Already Exists!)

**Location**: `crates/bitnet-sys/src/wrapper.rs`

```rust
// Session API (llama.cpp)
pub fn eval_and_get_logits(&mut self, tokens: &[i32], n_past: i32) -> Result<Vec<f32>>
pub fn get_all_logits(&self, n_tokens: usize) -> Result<Vec<Vec<f32>>>

// Custom BitNet C Shim
pub fn bitnet_eval_tokens(ctx: &BitnetContext, ids: &[i32], vocab_size: usize) -> Result<Vec<f32>>
pub fn bitnet_prefill(ctx: &BitnetContext, ids: &[i32]) -> Result<()>
pub fn cpp_decode_greedy(..., out: &mut [i32]) -> Result<usize>
pub fn bitnet_tokenize_text(model: &BitnetModel, text: &str, ...) -> Result<Vec<i32>>
pub fn cpp_vocab_size(ctx: &BitnetContext) -> Result<usize>
```

### Missing: Per-Position Logits Wrapper

**Current limitation**: Wrapper returns only Vec<f32> (last position)

**Quick fix**:
```rust
// Add to Session impl:
pub fn eval_and_get_all_logits(&mut self, tokens: &[i32], n_past: i32) -> Result<Vec<Vec<f32>>> {
    self.context.eval(tokens, n_past)?;
    self.context.get_all_logits(tokens.len())  // Line 285 in wrapper.rs
}
```

### Logits Comparison Format

**Input**: `&[Vec<f32>]` where outer = positions, inner = vocab_size

**Output**: `LogitsDivergence`
```rust
pub struct LogitsDivergence {
    pub first_divergence_token: Option<usize>,
    pub per_token_cosine_sim: Vec<f32>,
    pub per_token_l2_dist: Vec<f32>,
    pub max_absolute_diff: f32,
}
```

### Integration Roadmap

| Phase | Task | Effort | Impact |
|-------|------|--------|--------|
| 1 | Connect crossval to bitnet-sys Session | MEDIUM (2h) | Enable real C++ backend |
| 2 | Add Session::eval_and_get_all_logits() | LOW (1h) | Multi-position support |
| 3 | Integrate with custom BitNet shim | MEDIUM-HIGH (3-5h) | Full optimization potential |

### Key Files
- FFI bindings: `crates/bitnet-sys/src/wrapper.rs` (660 lines)
- Comparison logic: `crossval/src/logits_compare.rs`
- Integration tests: `crossval/tests/per_position_logits.rs`
- Mock wrapper: `crossval/src/cpp_bindings.rs` (224 lines - to be replaced)

### Gotchas
1. **Token ID type**: bitnet-sys uses `i32`, crossval returns `u32` (needs conversion)
2. **Logits return**: Currently last-position only; need to surface `get_all_logits()`
3. **Decoupling**: crossval has its own mock C wrapper, doesn't use bitnet-sys (yet)
4. **KV cache**: BitNet shim may require explicit `bitnet_prefill()` before eval

### Next Steps
1. Review `crates/bitnet-sys/src/wrapper.rs` lines 330-394 (Session struct)
2. Check `crossval/tests/per_position_logits.rs` line 12 import (uses bitnet-sys::wrapper already!)
3. Add 10-line wrapper for `eval_and_get_all_logits()`
4. Test with `crossval-per-token` scenario

