# Stop-Sequence Implementation Analysis - Executive Summary

## Location of Key Code
- **Primary Generation Loop**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/engine.rs:1080-1232`
- **Stop-Sequence Checker**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/engine.rs:1318-1352`
- **Streaming Implementation**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/streaming.rs:456-496`
- **Configuration**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/config.rs:38-119`

## The "One Token Late" Bug - Root Cause

### The Problem
Stop-sequence matching is performed AFTER sampling a token but BEFORE including it in the text being checked:

```
Generation Step:
1. Sample token (e.g., "</s>")
2. Call should_stop(token, generated_tokens)  ← Token NOT in generated_tokens yet
3. Check if decoded_text ends with stop_sequence
4. Problem: decoded_text doesn't include the just-sampled token!
5. Result: Stop sequence not detected until next iteration
```

### Why Token ID Stops Work But String Stops Don't

**Token ID stops** (line 1323-1325 in engine.rs):
```rust
if !config.stop_token_ids.is_empty() && config.stop_token_ids.contains(&token) {
    return true;  // ✅ Checks the candidate token directly
}
```

**String stops** (line 1337-1349 in engine.rs):
```rust
let current_text = self.tokenizer.decode(tail_tokens).unwrap_or_default();
for stop_seq in &config.stop_sequences {
    if current_text.ends_with(stop_seq) {
        return true;  // ❌ Never checks what candidate would add
    }
}
```

---

## Affected Code Paths

| Component | File | Line(s) | Issue |
|-----------|------|---------|-------|
| **Engine** | `engine.rs` | 1080-1232 (loop), 1319-1352 (should_stop) | Main generation loop with buggy stop check |
| **Streaming** | `streaming.rs` | 456-496 | Same pattern: sample → check with old state |
| **Autoregressive** | `generation/autoregressive.rs` | 305-309 | Uses should_stop() with same logic |

---

## The Fix: Include Candidate Token in Stop Check

### Simple Solution
Add the candidate token to the decoded text before checking:

```rust
fn should_stop(&self, token: u32, generated_tokens: &[u32], config: &GenerationConfig) -> bool {
    // ... token ID and EOS checks ...
    
    // String-based stop sequences WITH CANDIDATE (fixes "one token late" bug)
    if !config.stop_sequences.is_empty() {
        // Build test sequence: tail tokens + candidate
        let window_size = config.stop_string_window.min(generated_tokens.len() + 1);
        let tail_start = generated_tokens.len().saturating_sub(window_size - 1);
        let mut test_tokens = generated_tokens[tail_start..].to_vec();
        test_tokens.push(token);  // ← THE FIX: Include candidate
        
        // Decode and check
        if let Ok(text) = self.tokenizer.decode(&test_tokens) {
            for stop_seq in &config.stop_sequences {
                if text.ends_with(stop_seq) {
                    return true;
                }
            }
        }
    }
    
    false
}
```

### Key Changes Required
1. **engine.rs** (line ~1337-1349): Update `should_stop()` to include candidate token
2. **streaming.rs** (line ~347): Apply same fix in streaming `should_stop()` call
3. **autoregressive.rs** (line ~897): Update `should_stop()` method

---

## Impact Assessment

### Performance Impact
- **Current**: O(window_size) tokenizer.decode() per step
- **Fixed**: O(window_size + 1) tokenizer.decode() per step
- **Overhead**: +1 token decode, ~<5% cost increase
- **Evaluation**: Negligible for typical use (token ID stops checked first)

### Correctness Impact
- **Before**: String stop sequences trigger ~1 token too late
- **After**: Exact token detection (matches token ID stop behavior)
- **Risk**: Low - only fixes correctness, doesn't change API

---

## Test Cases Needed

```rust
#[test]
fn test_stop_sequence_matches_on_candidate() {
    // "</s>" should be detected when it's the sampled token
    let config = GenerationConfig {
        stop_sequences: vec!["</s>".to_string()],
        ..Default::default()
    };
    let generated_tokens = vec![1, 2, 3];
    let eos_token = 4;  // Decodes to "</s>"
    
    // Should return true because candidate "</s>" completes the sequence
    assert!(should_stop(eos_token, &generated_tokens, &config));
}

#[test]
fn test_stop_sequence_boundary_crossing() {
    // Test "end" where token 1="e" and token 2="nd"
    let config = GenerationConfig {
        stop_sequences: vec!["end".to_string()],
        ..Default::default()
    };
    let generated_tokens = vec![1];  // Just the "e"
    let candidate = 2;  // The "nd"
    
    // Should detect "e" + "nd" = "end"
    assert!(should_stop(candidate, &generated_tokens, &config));
}
```

---

## Rolling Tail Buffer Management

### Current Implementation (engine.rs:1338-1341)
```rust
let window_size = config.stop_string_window.min(generated_tokens.len());
let tail_start = generated_tokens.len().saturating_sub(window_size);
let tail_tokens = &generated_tokens[tail_start..];
```

**Problem**: 
- `window_size` capped at `generated_tokens.len()` 
- Doesn't account for adding candidate token
- Can decode more than `stop_string_window` tokens when candidate is added

### Fixed Implementation
```rust
// Size must accommodate both tail + candidate
let window_size = config.stop_string_window.min(generated_tokens.len() + 1);
let tail_start = generated_tokens.len().saturating_sub(window_size - 1);
let tail_tokens = &generated_tokens[tail_start..];

// Then append candidate
let mut test_tokens = tail_tokens.to_vec();
test_tokens.push(candidate_token);
```

---

## Architecture Overview

### Generation Loop (engine.rs:1098-1222)
```
FOR each step in max_new_tokens:
  1. Forward pass → get logits
  2. Sample next_token from logits
  3. should_stop(next_token, generated_tokens)  ← BUG HERE
     └─ generated_tokens still doesn't include next_token
     └─ String check misses newly-sampled token
  4. Push next_token to generated_tokens
  5. Push next_token to current_tokens
```

### String Stop Matching (engine.rs:1337-1349)
```
should_stop(token, generated_tokens, config):
  1. Check token IDs ✅ (works correctly)
  2. Check EOS ✅ (works correctly)  
  3. Check string sequences ❌ (misses token)
     └─ Decode tail of generated_tokens
     └─ Check if ends_with(stop_seq)
     └─ Never sees what token would add!
```

---

## Configuration Fields Involved

From `config.rs`:

```rust
pub struct GenerationConfig {
    pub stop_sequences: Vec<String>,      // Sequences to check (currently buggy)
    pub stop_token_ids: Vec<u32>,         // Token IDs to check (works fine)
    pub stop_string_window: usize,        // Window size (default: 64) - needs adjustment
    pub eos_token_id: Option<u32>,        // EOS token (works fine)
}
```

**Note**: `stop_string_window` documentation correctly describes "tail-based optimization" but implementation doesn't account for candidate token being added.

---

## Recommended Implementation Order

1. **Phase 1**: Add `matches_with_candidate()` helper function
   - Separate concern: checking with candidate included
   - Easier to test and review
   - File: `engine.rs`

2. **Phase 2**: Update `should_stop()` to use new helper
   - Minimal changes to existing logic
   - All generation paths can use same method
   - Files: `engine.rs`, `streaming.rs`, `autoregressive.rs`

3. **Phase 3**: Add comprehensive tests
   - Unit tests for `matches_with_candidate()`
   - Integration tests with real models
   - Regression tests for token ID stops

4. **Phase 4**: Performance validation
   - Benchmark decode overhead
   - Verify <5% throughput impact
   - Profile typical use cases

---

## Files Modified (Complete List)

| File | Lines | Change |
|------|-------|--------|
| `crates/bitnet-inference/src/engine.rs` | 1319-1352 | Update `should_stop()` + add helper |
| `crates/bitnet-inference/src/streaming.rs` | 456-496 | Use updated stop-check logic |
| `crates/bitnet-inference/src/generation/autoregressive.rs` | 305-309, 897-917 | Update generation loop + `should_stop()` |

---

## Summary

**Problem**: Stop-sequence matching doesn't include the just-sampled token, causing detection to be delayed by one token.

**Root Cause**: `should_stop()` receives the candidate token but passes it only for token ID checks, not for string matching.

**Solution**: Include candidate token when decoding text for string-based stop sequence matching.

**Risk**: Low - improves correctness without API changes.

**Performance**: <5% overhead from one additional token decode per generation step.

**Effort**: Medium - affects 3 files, requires 2-3 methods to update, comprehensive tests needed.

---

## Quick Reference: Where to Look

1. **Main bug location**: `engine.rs:1337-1349` (string stop check without candidate)
2. **Generation loop**: `engine.rs:1098-1222` (where should_stop is called)
3. **Streaming parallel**: `streaming.rs:347` (same bug in streaming)
4. **Config structure**: `config.rs:38-119` (stop_sequences field)
5. **Alternative implementation**: `generation/autoregressive.rs:897-917` (third affected path)
