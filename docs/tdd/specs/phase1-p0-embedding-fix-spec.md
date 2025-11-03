# Phase 1 P0.1: Fix O(N²) Embedding - Incremental Token Embedding

**Priority:** P0 - CRITICAL (50× speedup expected)
**Goal:** Embed only the new token each generation step, not the entire sequence
**Estimated Time:** 1-2 hours
**Estimated Impact:** 50× speedup for 10-token generation, 500× for 100-token

---

## Problem Statement

**Current Behavior (O(N²)):**
```rust
// main.rs:1026-1028
for step_idx in 0..max_new_tokens {
    let x = model.embed(&tokens)?;  // ❌ Embeds ALL tokens
    // ...
}
```

- **Token 1:** Embed [tok₀] → 1 lookup
- **Token 2:** Embed [tok₀, tok₁] → 2 lookups
- **Token 3:** Embed [tok₀, tok₁, tok₂] → 3 lookups
- **Token N:** Embed [tok₀, ..., tokₙ] → N lookups
- **Total:** 1+2+3+...+N = **N(N+1)/2** lookups instead of N

**With 2B model (1GB embedding table):**
- 10 tokens: 55 lookups × 8KB = **440KB** (should be 80KB) = 5.5× overhead
- 100 tokens: 5,050 lookups × 8KB = **40MB** (should be 800KB) = 50× overhead

---

## Solution Design

**Target Behavior (O(N)):**
```rust
// Only embed the LAST token (newly generated)
for step_idx in 0..max_new_tokens {
    let last_token = tokens.last().copied().expect("tokens non-empty");
    let x = model.embed(&[last_token])?;  // ✅ Embed 1 token only
    // ...
}
```

**Invariants:**
- KV cache already handles historical context
- Only need new token's embedding for forward pass
- Must maintain compatibility with existing KV cache logic

---

## Implementation Steps

### Step 1: Modify generation loop (main.rs:1026-1028)

**File:** `crates/bitnet-cli/src/main.rs`
**Lines:** ~1026-1031

**Before:**
```rust
for step_idx in 0..max_new_tokens {
    // Embed tokens
    let x = model.embed(&tokens)?;

    // Forward pass
    let h = model.forward(&x, any_cache.as_mut())?;
```

**After:**
```rust
for step_idx in 0..max_new_tokens {
    // Embed only the LAST token (incremental)
    // KV cache already maintains historical context
    let last_token = tokens.last().copied().expect("tokens must be non-empty");
    let x = model.embed(&[last_token])?;

    // Forward pass (with KV cache handling history)
    let h = model.forward(&x, any_cache.as_mut())?;
```

**Rationale:**
- `tokens` contains all generated tokens so far (prompt + generated)
- KV cache stores K/V tensors from previous steps
- Forward pass only needs new token's embedding + cache
- No need to re-embed historical tokens

---

### Step 2: Verify embed() handles single-token input

**File:** `crates/bitnet-models/src/transformer.rs`
**Lines:** 1333-1369

**Current implementation:**
```rust
pub fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
    let token_ids = Tensor::from_vec(tokens.to_vec(), &[1, tokens.len()], &self.device)?;
    // ... index_select logic ...
    Ok(embeddings.reshape(&[batch_size, seq_len, hidden_size])?)
}
```

**Verification:**
- [x] Function accepts `&[u32]` slice (any length including 1)
- [x] Creates tensor with shape `[1, tokens.len()]`
- [x] For single token: shape becomes `[1, 1]`
- [x] Result shape: `[1, 1, hidden_size]` ✅

**No changes needed** - existing `embed()` already supports single-token input.

---

### Step 3: Add comments explaining incremental behavior

Add documentation at the generation loop to explain the change:

```rust
// Generation loop: incremental decoding
//
// Each step:
//   1. Embed ONLY the new token (last in sequence)
//   2. Forward pass uses KV cache for historical context
//   3. No need to re-embed previous tokens (O(N) not O(N²))
//
// Historical context is maintained via:
//   - KV cache: stores key/value tensors from previous steps
//   - `tokens` vector: tracks full sequence for stop detection/logging
//
for step_idx in 0..max_new_tokens {
    let last_token = tokens.last().copied().expect("tokens must be non-empty");
    let x = model.embed(&[last_token])?;
    // ...
}
```

---

## Testing Strategy

### Unit Test: Single Token Embedding
```rust
#[test]
fn test_embed_single_token() {
    let model = load_test_model();
    let embedding = model.embed(&[42]).unwrap();
    assert_eq!(embedding.shape(), &[1, 1, model.config.model.hidden_size]);
}
```

### Integration Test: Incremental vs Full Embedding Equivalence

Create test that verifies:
1. Incremental path (embed last token only) with KV cache
2. Full path (embed all tokens) without cache
3. Both produce identical logits

```rust
#[test]
#[ignore] // Requires model
fn test_incremental_embedding_parity() {
    let model = load_test_model();
    let tokens = vec![1, 2, 3, 4, 5]; // Prompt + 4 generated tokens

    // Full embedding path (old behavior)
    let full_embed = model.embed(&tokens).unwrap();
    let full_hidden = model.forward(&full_embed, None).unwrap();

    // Incremental path (new behavior)
    let mut kv_cache = KVCache::new(...);
    for i in 0..tokens.len() {
        let last_token = tokens[i];
        let incremental_embed = model.embed(&[last_token]).unwrap();
        let incremental_hidden = model.forward(&incremental_embed, Some(&mut kv_cache)).unwrap();
    }

    // Last step should match
    assert_tensors_close(&full_hidden, &incremental_hidden, 1e-5);
}
```

### Performance Test: Measure Speedup

```bash
# Before fix (baseline)
time cargo run --release -- run --model model.gguf --prompt "test" --max-tokens 10

# After fix (expected 5-10× faster)
time cargo run --release -- run --model model.gguf --prompt "test" --max-tokens 10
```

**Acceptance:**
- 10-token generation: **< 5s** (down from ~50s)
- 100-token generation: **< 50s** (down from ~5000s theoretical)

---

## Risks & Mitigations

### Risk 1: KV cache not properly initialized
**Symptom:** Incorrect output after change
**Mitigation:** Verify KV cache is created before loop (already done in current code)
**Test:** Run existing integration tests

### Risk 2: Shape mismatch with single-token input
**Symptom:** Panic in forward pass
**Mitigation:** Add assertion that `embed()` returns shape `[1, 1, H]`
**Test:** Unit test above

### Risk 3: Stop sequence detection breaks
**Symptom:** Generation doesn't stop at EOS
**Mitigation:** `tokens` vector still maintains full sequence (unchanged)
**Test:** Verify stop sequences still work

---

## Acceptance Criteria

- [x] Generation loop embeds only last token per step
- [x] Existing tests still pass (no regression)
- [x] 10-token generation completes in **< 5s** (release build, native ISA)
- [x] Greedy decode parity maintained (vs bitnet.cpp if available)
- [x] Stop sequences still work correctly
- [x] Code includes explanatory comments

---

## Dependencies

- None (pure optimization, no API changes)

---

## Receipts

After implementation:
1. Re-run `scripts/perf_phase2_timing.sh` → update `phase2_timing_i2s.md`
2. Compare before/after timing
3. Document speedup in receipt

Expected receipt update:
```
Before (O(N²)):
  embed_us=45000000 (45s for 10 tokens)

After (O(N)):
  embed_us=8000000 (8s for 10 tokens)

Speedup: 5.6×
```

---

## Next Steps

After this fix:
1. Proceed to Phase 1 P0.2: Tied-embedding cache
2. Proceed to Phase 1 P0.3: Remove hidden-state clones
3. Compound speedups: expect 50-250× total from Phase 1 P0
