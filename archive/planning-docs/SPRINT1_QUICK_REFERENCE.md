# Sprint 1: Per-Position Logits Extraction - Quick Reference

## Problem Statement
Current `eval_logits_once()` returns logits for only the **last token** position.  
Need: Function that returns logits for **all token positions** to enable per-position cross-validation.

---

## Solution Overview

Add 2 new functions to `crates/bitnet-inference/src/parity.rs`:

| Function | Purpose | Returns | Location |
|----------|---------|---------|----------|
| `eval_logits_all_positions()` | Public API for all-positions logits | `Result<Vec<Vec<f32>>>` | Add after line 128 |
| `extract_all_position_logits()` | Helper to extract per-position logits | `Result<Vec<Vec<f32>>>` | Add after line 261 |

---

## Key Findings

### Current Flow (Single Token)
```
tokens → embed() → forward() → logits() → extract_last_token_logits() → Vec<f32>
                    ↑
                  [1, H]
                  
                              ↓
                            [1, V]
                            
                                      ↓
                                    vocab_size floats
```

### New Flow (All Positions)
```
tokens → embed() → forward_full() → logits() → extract_all_position_logits() → Vec<Vec<f32>>
                    ↑                                                              ↑
                  [1, T, H]                                                   seq_len × vocab_size
                  
                              ↓
                            [1, T, V]
```

---

## What's Already in Place ✅

1. **`model.forward_full()`** exists at `transformer.rs:1416`
   - Handles full sequences `[B, T, H] → [B, T, H]`

2. **`model.logits()`** handles rank-3 tensors at `transformer.rs:1635-1687`
   - Can process `[B, T, H] → [B, T, V]`
   - Currently only rank-2 path is used

3. **`compare_per_position_logits()`** ready in `crossval/logits_compare.rs:49-102`
   - Takes `Vec<Vec<f32>>` from both Rust & C++
   - Returns per-position divergence metrics

---

## Implementation Checklist

- [ ] Copy `eval_logits_once()` code block (lines 30-111)
- [ ] Rename copy to `eval_logits_all_positions()`
- [ ] Change `model.embed()` to use full sequence
- [ ] Change `model.forward()` → `model.forward_full()`
- [ ] Remove KV cache initialization
- [ ] Change extraction call to `extract_all_position_logits()`
- [ ] Copy `extract_last_token_logits()` code (lines 223-261)
- [ ] Rename to `extract_all_position_logits()`
- [ ] Change loop to iterate all positions (not just last)
- [ ] Update error messages for 3D tensor expectations
- [ ] Add pub visibility to `eval_logits_all_positions()`
- [ ] Add comprehensive doc comments
- [ ] Add unit tests
- [ ] Test against C++ reference

---

## File Locations (Absolute Paths)

| What | File | Key Lines |
|------|------|-----------|
| **Primary target** | `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs` | 30-128, 223-261 |
| **Model methods** | `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/transformer.rs` | 1416, 1507, 1548 |
| **Comparison** | `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs` | 49-102 |
| **Analysis doc** | `/home/steven/code/Rust/BitNet-rs/LOGITS_EXTRACTION_ANALYSIS.md` | (full reference) |

---

## Code Template: `eval_logits_all_positions()`

```rust
pub fn eval_logits_all_positions(model_path: &str, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
    // 1. Load model (copy lines 31-89 from eval_logits_once)
    let (config, model) = match load_gguf_full(...) {
        Ok(result) => { /* QK256 conversion code */ (result.config, model) }
        Err(e) => anyhow::bail!("Failed to load GGUF model: {}", e),
    };

    // 2. Convert tokens
    let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();

    // 3. Get embeddings for ALL tokens (not single token)
    let embedded = model.embed(&tokens_u32)?;

    // 4. Forward pass (NO KV CACHE)
    let output = model.forward_full(&embedded)?;

    // 5. Get logits (uses 3D path internally)
    let logits = model.logits(&output)?;

    // 6. Extract all positions
    extract_all_position_logits(logits)
}
```

---

## Code Template: `extract_all_position_logits()`

```rust
fn extract_all_position_logits(logits: bitnet_common::ConcreteTensor) -> Result<Vec<Vec<f32>>> {
    match logits {
        ConcreteTensor::BitNet(tensor) => {
            let candle_tensor = tensor.as_candle();
            let dims = candle_tensor.dims();
            
            if dims.len() != 3 {
                anyhow::bail!("Expected 3D [B, T, V], got {:?}", dims);
            }
            
            let (batch, seq_len, vocab) = (dims[0], dims[1], dims[2]);
            if batch != 1 {
                anyhow::bail!("Expected batch=1, got {}", batch);
            }
            
            let mut all_logits = Vec::with_capacity(seq_len);
            
            // Extract each position
            for pos in 0..seq_len {
                let pos_logits = candle_tensor
                    .narrow(1, pos, 1)?      // Get position
                    .squeeze(1)?              // Remove T dim
                    .i(0)?;                   // Get batch[0]
                
                let pos_logits = if pos_logits.dtype() != DType::F32 {
                    pos_logits.to_dtype(DType::F32)?
                } else {
                    pos_logits.clone()
                };
                
                all_logits.push(pos_logits.to_vec1::<f32>()?);
            }
            
            Ok(all_logits)
        }
        ConcreteTensor::Mock(mock) => {
            let seq_len = mock.shape()[1];
            let vocab = mock.shape()[2];
            Ok(vec![vec![0.0; vocab]; seq_len])
        }
    }
}
```

---

## Integration with Crossval Tests

```rust
#[cfg(feature = "crossval")]
#[test]
fn test_per_position_parity_with_cpp() {
    use bitnet_inference::eval_logits_all_positions;
    use crossval::logits_compare::compare_per_position_logits;
    
    let model_path = "models/test.gguf";
    let tokens = vec![1, 2, 3, 4];
    
    // Get Rust logits for all positions
    let rs_logits = eval_logits_all_positions(model_path, &tokens)?;
    
    // Get C++ logits for all positions (from FFI)
    let cpp_logits = cpp_reference.eval_all_positions(&tokens)?;
    
    // Compare per-position
    let divergence = compare_per_position_logits(&rs_logits, &cpp_logits);
    
    // Assertions
    assert!(divergence.first_divergence_token.is_none());
    for sim in divergence.per_token_cosine_sim {
        assert!(sim > 0.9999);
    }
}
```

---

## Differences from `eval_logits_once()`

| Aspect | `eval_logits_once()` | `eval_logits_all_positions()` |
|--------|---------------------|-----|
| Input tokens | `[i32]` | `[i32]` (same) |
| Embedding | Single token sequence | Full token sequence |
| Forward call | `model.forward()` | `model.forward_full()` |
| KV cache | Yes (for incremental) | No (batch processing) |
| Logits input | 2D `[B, V]` | 3D `[B, T, V]` |
| Logits path | Rank 2 handler | Rank 3 handler |
| Output | `Vec<f32>` | `Vec<Vec<f32>>` |
| Size | vocab_size | seq_len × vocab_size |

---

## Testing Strategy

### Unit Tests
- [ ] Test structure: correct number of positions and vocab items
- [ ] Test types: all values are finite F32
- [ ] Test with different sequence lengths
- [ ] Test with different models (2B, 7B, etc.)

### Integration Tests
- [ ] Test with C++ reference (requires `--features crossval`)
- [ ] Verify per-position cosine similarity > 0.9999
- [ ] Check max absolute difference < 1e-5
- [ ] Verify no early divergence

### Performance Tests
- [ ] Measure latency for 2B model, 128 tokens
- [ ] Verify memory usage stays reasonable
- [ ] Check if faster than calling `eval_logits_once()` 128 times

---

## No Breaking Changes

- ✅ `eval_logits_once()` unchanged - no migration needed
- ✅ Existing callers unaffected
- ✅ New functions are additions only
- ✅ Backward compatible with all quantization formats

---

## Estimated Effort

- **Implementation**: 150-200 lines of code
- **Testing**: 100-150 lines of test code
- **Documentation**: Already in `LOGITS_EXTRACTION_ANALYSIS.md`
- **Total**: ~250-350 lines

---

## Success Criteria

1. ✅ Function compiles without warnings
2. ✅ All logits values are finite F32
3. ✅ Output has correct shape: `[seq_len][vocab_size]`
4. ✅ Matches C++ reference within tolerance
5. ✅ Works with all supported quantization formats
6. ✅ Comprehensive test coverage
7. ✅ Documented with examples

---

## Next Steps

1. Read full analysis: `/home/steven/code/Rust/BitNet-rs/LOGITS_EXTRACTION_ANALYSIS.md`
2. Implement functions following template
3. Run tests: `cargo test -p bitnet-inference --no-default-features --features cpu`
4. Test with C++: `cargo test -p bitnet-inference --features crossval`
5. Benchmark: Compare vs calling `eval_logits_once()` multiple times

