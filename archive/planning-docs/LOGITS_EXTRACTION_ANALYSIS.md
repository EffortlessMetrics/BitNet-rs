# BitNet-rs Logits Extraction Analysis - Comprehensive Report

**Status**: Complete exploration of logits extraction system for Sprint 1  
**Date**: 2025-10-24  
**Focus**: All-positions logits extraction for per-position cross-validation  

---

## Executive Summary

The BitNet-rs codebase uses a **single-path logits extraction mechanism** that currently returns only the last token's logits. To enable per-position logits comparison (critical for Sprint 1), we need to:

1. **Modify `parity.rs`** to extract logits at all positions from the forward pass output
2. **Leverage existing tensor operations** (no new computation needed - already computed)
3. **Add two new public functions** to the parity module
4. **Reuse existing comparison infrastructure** in `crossval/src/logits_compare.rs`

---

## Part 1: Current Logits Extraction Flow (Single Position)

### Call Chain: Step-by-Step

```
eval_logits_once(model_path, tokens)
    ↓
    [Lines 31-89: Load GGUF + Model Creation]
    ↓
    model.embed(&tokens_u32)          [Line 99: Get embeddings]
        Returns: Tensor [batch=1, hidden_size]
    ↓
    model.forward(&embedded, cache)   [Line 102: Forward pass through all layers]
        Input:  [1, hidden_size]
        Output: [1, hidden_size]       ← This is JUST the LAST token!
        Processes: embedded → layers[0] → ... → layers[n-1] → ln_f
    ↓
    model.logits(&output)             [Line 105: Project hidden → vocab]
        Input:  [1, hidden_size]
        Output: [1, vocab_size]
    ↓
    extract_last_token_logits(logits) [Line 108: Extract from tensor]
        Input:  ConcreteTensor (logits for batch=1)
        Output: Vec<f32> with vocab_size elements
```

**Key Insight**: The forward pass itself is **already lossy** - it only returns the last token's hidden state. The sequence information is lost during the forward pass, not during logits extraction.

---

## Part 2: Data Structures Involved

### 1. Input Tensor (Embeddings)
**Location**: `crates/bitnet-models/src/transformer.rs` (embed method)
- **Shape**: `[batch_size, hidden_size]` where batch_size=1
- **Type**: `Tensor` (Candle abstraction over `ConcreteTensor`)
- **Note**: Embeddings for ALL tokens are concatenated but forward pass processes only final hidden state

### 2. Forward Pass Output
**Location**: `crates/bitnet-models/src/transformer.rs:1507-1546` (`forward` method)
```rust
pub fn forward(&self, hidden: Tensor, mut kv_cache: Option<&mut KVCache>) -> Result<Tensor> {
    let mut x = hidden;  // Take ownership
    
    for (i, layer) in self.layers.iter().enumerate() {
        let layer_cache = kv_cache.as_mut().and_then(|c| c.layer_mut(i));
        x = layer.forward(&x, layer_cache, &self.raw_tensors)?;  // x is [B, H]
    }
    
    let normalized = self.norm.forward(&x)?;  // Final LayerNorm
    Ok(normalized)  // Returns [B, H] - LAST token only
}
```

**Issue Identified**: 
- Input `hidden` is shape `[batch=1, hidden_size]` (single token after embedding)
- Forward pass processes through all layers
- Output is also `[batch=1, hidden_size]` - represents final token position

### 3. Logits Tensor
**Location**: `crates/bitnet-models/src/transformer.rs:1548-1691` (`logits` method)

The `logits` method has TWO code paths based on input rank:

#### Path A: Rank 2 (Last Token Only) - Currently Used
```rust
match hidden.rank() {
    2 => {
        // [B, H] - last token only
        let (b, _h) = (hidden.dims()[0], hidden.dims()[1]);
        
        let logits = if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(hidden)?  // [B, V]
        } else {
            // Tied embeddings path...
            hidden.matmul(embed_weight)?  // [B, V]
        };
        
        Ok(logits)  // Returns [1, vocab_size]
    }
```
**Line Range**: 1552-1633 (82 lines)

#### Path B: Rank 3 (All Positions) - NOT CURRENTLY USED
```rust
3 => {
    // [B, T, H] - all timesteps
    let (b, t, h) = (hidden.dims()[0], hidden.dims()[1], hidden.dims()[2]);
    
    if let Some(ref lm_head) = self.lm_head {
        let hidden_2d = hidden.reshape(&[b * t, h])?;
        let logits_2d = lm_head.forward(&hidden_2d)?;
        Ok(logits_2d.reshape(&[b, t, vocab_size])?)  // [B, T, V]
    } else {
        // Tied embeddings path...
        hidden_2d.matmul(embed_weight)?;
        Ok(...reshape to [b, t, vocab_size])
    }
}
```
**Line Range**: 1635-1687 (53 lines)

**Critical Finding**: The model already knows how to handle rank-3 tensors! The infrastructure exists but isn't used in `eval_logits_once`.

### 4. Extract Last Token Logits Function
**Location**: `crates/bitnet-inference/src/parity.rs:223-261`

```rust
fn extract_last_token_logits(logits: bitnet_common::ConcreteTensor) -> Result<Vec<f32>> {
    use bitnet_common::ConcreteTensor;
    
    match logits {
        ConcreteTensor::BitNet(tensor) => {
            let candle_tensor = tensor.as_candle();
            let dims = candle_tensor.dims();
            if dims.len() != 3 {
                anyhow::bail!("Expected 3D logits tensor, got {:?}", dims);
            }
            
            let seq_len = dims[1];
            
            // Extract last token: narrow to last position in sequence dimension
            let last_token_logits = candle_tensor
                .narrow(1, seq_len - 1, 1)?  // Get last position in seq dim
                .squeeze(1)?                   // Remove seq dimension
                .i(0)?;                        // Get first (and only) batch
            
            let last_token_logits = if last_token_logits.dtype() != DType::F32 {
                last_token_logits.to_dtype(DType::F32)?
            } else {
                last_token_logits.clone()
            };
            
            Ok(last_token_logits.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(mock) => {
            let vocab_size = mock.shape()[2];
            Ok(vec![0.0f32; vocab_size])
        }
    }
}
```

**Critical Issue**: This function EXPECTS a 3D tensor with shape `[B, T, V]` but the current `eval_logits_once` flow returns 2D `[B, V]`, causing the dimension check to fail!

**Evidence of Expected 3D Design**:
- Line 232: `if dims.len() != 3` suggests the function was designed for all positions
- Line 237: `let seq_len = dims[1];` assumes sequence dimension exists
- Line 240-241: The narrowing operation assumes sequence positions are present

---

## Part 3: The Central Problem

The `extract_last_token_logits` function is **misnamed and misaligned** with the current flow:

### Current (Broken) Flow:
```
eval_logits_once()
  ↓
model.logits(output: [1, H])      ← Input is 2D
  ↓
extract_last_token_logits()       ← Function EXPECTS 3D input!
  ↓
ERROR: "Expected 3D logits tensor"
```

### Why It Works Now:
Looking at the model definition, there's actually **two separate paths**:
1. **Single-token path** (used by eval_logits_once):
   - Input: embeddings for single token `[1, H]`
   - Forward through layers (processes incrementally via KV cache)
   - Output to logits: `[1, V]` (via 2D rank path in logits())
   
2. **Full-sequence path** (NOT used by eval_logits_once):
   - Input: embeddings for all tokens at once `[1, T, H]`
   - Forward through layers (processes all at once)
   - Output to logits: `[1, T, V]` (via 3D rank path in logits())

The single-token path must use a different code path that doesn't call `extract_last_token_logits`.

---

## Part 4: Exact Line Numbers of Key Functions

| Function | File | Lines | Purpose |
|----------|------|-------|---------|
| `eval_logits_once()` | `parity.rs` | 30-111 | Main entry point, single token |
| `eval_logits_once_for_parity()` | `parity.rs` | 125-128 | Wrapper for parity tests |
| `eval_logits_incremental()` | `parity.rs` | 212-220 | Stub for multi-step evaluation |
| `extract_last_token_logits()` | `parity.rs` | 223-261 | Extract logits from tensor |
| `model.embed()` | `transformer.rs` | (not shown) | Token embedding |
| `model.forward()` | `transformer.rs` | 1507-1546 | Forward pass (single token) |
| `model.logits()` | `transformer.rs` | 1548-1691 | Project hidden → logits (handles 2D & 3D) |
| `compare_per_position_logits()` | `crossval/logits_compare.rs` | 49-102 | Already exists for comparison! |

---

## Part 5: Data Flow Diagram (Current vs Desired)

### Current Single-Token Flow:
```
tokens [i32]
  ↓
embed() → [1, H]
  ↓
forward() → [1, H]
  ↓
logits() [2D path] → [1, V]
  ↓
Vec<f32> (vocab_size elements)
```

### Desired All-Positions Flow (Sprint 1):
```
tokens [i32]
  ↓
embed() → [1, T, H]          ← ALL tokens
  ↓
forward_full() → [1, T, H]   ← All positions
  ↓
logits() [3D path] → [1, T, V]
  ↓
Vec<Vec<f32>>                ← Per-position logits
```

**Available in Model**:
- `forward_full()` at `transformer.rs:1416` - handles full sequences!

---

## Part 6: What Needs to Change for Sprint 1

### Minimal Changes Required:

#### 1. Add New Function: `eval_logits_all_positions()`
**Location**: `crates/bitnet-inference/src/parity.rs` (after line 128)

```rust
/// Perform a forward pass and return logits for ALL token positions
///
/// This function processes the entire token sequence at once and returns
/// logits for each position, enabling per-position cross-validation.
///
/// # Arguments
/// * `model_path` - Path to the GGUF model file
/// * `tokens` - Token IDs
///
/// # Returns
/// * Vec of logits vectors, one per token position (seq_len × vocab_size)
pub fn eval_logits_all_positions(model_path: &str, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
    // (Implementation follows existing pattern - see Part 7)
}
```

#### 2. Modify or Create: `extract_all_position_logits()`
**Location**: `crates/bitnet-inference/src/parity.rs` (after line 261)

```rust
/// Extract logits for all token positions from the model output
fn extract_all_position_logits(logits: bitnet_common::ConcreteTensor) -> Result<Vec<Vec<f32>>> {
    // (Implementation follows existing pattern - see Part 7)
}
```

#### 3. No Changes Needed:
- ✅ `transformer.rs` already has `forward_full()` for full sequences
- ✅ `transformer.rs::logits()` already handles 3D tensors (rank 3 path)
- ✅ `crossval/logits_compare.rs` has comparison functions ready
- ✅ Model loading and config handling unchanged

---

## Part 7: Detailed Implementation Template

### Function 1: `eval_logits_all_positions()`

```rust
/// Perform a forward pass and return logits for ALL token positions
pub fn eval_logits_all_positions(model_path: &str, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
    // Load model (same as eval_logits_once - lines 31-89)
    let (config, model) = match load_gguf_full(
        Path::new(model_path),
        Device::Cpu,
        bitnet_models::GGUFLoaderConfig::default(),
    ) {
        Ok(result) => {
            // Convert QK256 tensors (lines 45-75)
            // ... same code as eval_logits_once ...
            (result.config, model)
        }
        Err(e) => anyhow::bail!("Failed to load GGUF model: {}", e),
    };

    // Convert i32 tokens to u32
    let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();

    // Get embeddings for ALL tokens at once (not single token!)
    let embedded = model.embed(&tokens_u32)?;
    // NOTE: embedded should have shape [1, seq_len, hidden_size]
    
    // No KV cache needed for full sequence - process all at once
    // Run forward pass with forward_full() instead of forward()
    let output = model.forward_full(&embedded)?;
    // output shape: [1, seq_len, hidden_size]

    // Get logits from the output (logits() method handles 3D tensors)
    let logits = model.logits(&output)?;
    // logits shape: [1, seq_len, vocab_size]

    // Extract logits for ALL positions
    extract_all_position_logits(logits)
}
```

**Key Differences from `eval_logits_once()`**:
1. Use full-sequence embedding path (no truncation)
2. Call `forward_full()` instead of `forward()`
3. Don't use KV cache (not needed for non-incremental path)
4. Call `extract_all_position_logits()` instead of `extract_last_token_logits()`

### Function 2: `extract_all_position_logits()`

```rust
/// Extract logits for all token positions from the model output
fn extract_all_position_logits(logits: bitnet_common::ConcreteTensor) -> Result<Vec<Vec<f32>>> {
    use bitnet_common::ConcreteTensor;

    match logits {
        ConcreteTensor::BitNet(tensor) => {
            let candle_tensor = tensor.as_candle();
            let dims = candle_tensor.dims();
            
            if dims.len() != 3 {
                anyhow::bail!("Expected 3D logits tensor [B, T, V], got {:?}", dims);
            }
            
            let (batch_size, seq_len, vocab_size) = (dims[0], dims[1], dims[2]);
            
            // Ensure we have batch_size=1
            if batch_size != 1 {
                anyhow::bail!("Expected batch_size=1, got {}", batch_size);
            }
            
            let mut all_logits = Vec::with_capacity(seq_len);
            
            // Extract logits for each position
            for pos in 0..seq_len {
                let pos_logits = candle_tensor
                    .narrow(1, pos, 1)?      // Get single position in seq dim
                    .squeeze(1)?              // Remove seq dimension
                    .i(0)?;                   // Get first (and only) batch
                
                // Convert to F32 if needed
                let pos_logits = if pos_logits.dtype() != DType::F32 {
                    pos_logits.to_dtype(DType::F32)?
                } else {
                    pos_logits.clone()
                };
                
                // Convert to Vec<f32>
                let logits_vec = pos_logits.to_vec1::<f32>()?;
                all_logits.push(logits_vec);
            }
            
            Ok(all_logits)
        }
        ConcreteTensor::Mock(mock) => {
            // For mock tensors, return mock logits for each position
            let seq_len = mock.shape()[1];
            let vocab_size = mock.shape()[2];
            let mock_logits = vec![0.0f32; vocab_size];
            Ok(vec![mock_logits; seq_len])
        }
    }
}
```

---

## Part 8: Integration with Crossval

The `crossval/src/logits_compare.rs` module is **already prepared** for this:

```rust
pub fn compare_per_position_logits(
    rs_logits: &[Vec<f32>],    // From eval_logits_all_positions()
    cpp_logits: &[Vec<f32>],   // From C++ reference
) -> LogitsDivergence {
    // Returns per-position metrics:
    // - first_divergence_token
    // - per_token_cosine_sim
    // - per_token_l2_dist
    // - max_absolute_diff
}
```

**Usage in tests**:
```rust
let rs_logits = eval_logits_all_positions(model_path, tokens)?;
let cpp_logits = cpp_reference.eval_all_positions(tokens)?;

let divergence = compare_per_position_logits(&rs_logits, &cpp_logits);
println!("Divergence at token: {:?}", divergence.first_divergence_token);
for (pos, sim) in divergence.per_token_cosine_sim.iter().enumerate() {
    println!("  Position {}: cosine_sim={}", pos, sim);
}
```

---

## Part 9: Critical Dependencies & Prerequisites

### Must Have Before Implementation:
1. ✅ `model.forward_full()` exists at `transformer.rs:1416`
2. ✅ `model.logits()` handles 3D tensors (rank 3 path at line 1635)
3. ✅ `extract_last_token_logits()` can serve as template
4. ✅ Comparison infrastructure in `crossval/logits_compare.rs`

### Model Methods Used:
- `BitNetModel::embed()` - for embeddings
- `BitNetModel::forward_full()` - for all-positions forward pass
- `BitNetModel::logits()` - for logits computation (uses 3D path)

### Type System Understanding:
- `ConcreteTensor::BitNet(tensor)` - Candle tensor wrapper
- `tensor.as_candle()` - get underlying Candle tensor
- `narrow()`, `squeeze()`, `i()` - Candle tensor operations
- `dims()` - get tensor dimensions
- `to_vec1::<f32>()` - convert tensor to Vec

---

## Part 10: Validation Checklist

After implementing `eval_logits_all_positions()`:

- [ ] Function accepts `&str` model_path and `&[i32]` tokens
- [ ] Returns `Result<Vec<Vec<f32>>>`
- [ ] Outer vec has length = seq_len (number of tokens)
- [ ] Each inner vec has length = vocab_size
- [ ] Values are finite F32 floats
- [ ] Matches C++ reference within tolerance (< 1e-5)
- [ ] Handles all quantization formats (I2_S, QK256, TL1/TL2)
- [ ] Works with various model sizes (2B, 7B, 13B)
- [ ] Completes in reasonable time (< 1 min for 2B, 128 tokens)

---

## Part 11: Example Test Case

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_logits_all_positions() {
        let model_path = "models/test-model.gguf";
        let tokens = vec![1, 2, 3];
        
        let logits = eval_logits_all_positions(model_path, &tokens)
            .expect("Failed to eval logits");
        
        // Verify structure
        assert_eq!(logits.len(), 3, "Should have logits for 3 positions");
        assert_eq!(logits[0].len(), 32000, "Each position should have vocab_size logits");
        
        // Verify values are finite
        for (pos, pos_logits) in logits.iter().enumerate() {
            for (i, &logit) in pos_logits.iter().enumerate() {
                assert!(logit.is_finite(), 
                    "Non-finite logit at pos={}, vocab_idx={}: {}", pos, i, logit);
            }
        }
    }
    
    #[test]
    #[cfg(feature = "crossval")]
    fn test_per_position_parity() {
        use crossval::logits_compare::compare_per_position_logits;
        
        let model_path = "models/test-model.gguf";
        let tokens = vec![1, 2, 3];
        
        let rs_logits = eval_logits_all_positions(model_path, &tokens).unwrap();
        let cpp_logits = cpp_reference(model_path, &tokens).unwrap();
        
        let divergence = compare_per_position_logits(&rs_logits, &cpp_logits);
        
        // All positions should match closely
        assert!(divergence.first_divergence_token.is_none());
        for sim in &divergence.per_token_cosine_sim {
            assert!(sim > 0.9999, "Cosine similarity too low: {}", sim);
        }
    }
}
```

---

## Part 12: Performance Considerations

### Time Complexity:
- **Current `eval_logits_once()`**: O(seq_len × hidden_size × vocab_size) 
  - But only returns last position
  
- **New `eval_logits_all_positions()`**: O(seq_len × hidden_size × vocab_size)
  - Processes all at once (parallel opportunity)
  - Returns all positions

### Space Complexity:
- **Current**: O(vocab_size) - single position logits
- **New**: O(seq_len × vocab_size) - all positions

### Optimization Opportunities:
1. Use batch matrix operations for all-positions at once (likely faster than single token repeated)
2. Leverage existing GPU kernels if available
3. Process in smaller batches if memory becomes constraint

---

## Part 13: Summary Table

| Aspect | Current | New Function |
|--------|---------|---|
| **Entry Function** | `eval_logits_once()` | `eval_logits_all_positions()` |
| **Embedding** | Single token | Full sequence |
| **Forward Pass** | `forward()` with KV cache | `forward_full()` without cache |
| **Logits Path** | 2D rank path `[B, V]` | 3D rank path `[B, T, V]` |
| **Extraction** | `extract_last_token_logits()` | `extract_all_position_logits()` |
| **Output** | `Vec<f32>` | `Vec<Vec<f32>>` |
| **Use Case** | Single-step generation | Cross-validation, debugging |
| **Comparison** | N/A | `compare_per_position_logits()` |

---

## Conclusion

**Sprint 1 is achievable with ~150 lines of code** in `parity.rs`:

1. Copy `eval_logits_once()` → `eval_logits_all_positions()`
2. Change embedding/forward calls to full-sequence variants
3. Create `extract_all_position_logits()` by adapting `extract_last_token_logits()`
4. Integrate with existing comparison infrastructure
5. Add comprehensive tests

**Key Files to Modify**:
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs` (primary)

**Key Files Already Ready**:
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/transformer.rs` (no changes)
- `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs` (no changes)

**No Breaking Changes**: Existing `eval_logits_once()` and callers remain unchanged.

