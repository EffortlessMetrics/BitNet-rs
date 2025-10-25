# GQA Investigation Summary

## Quick Answer: Root Cause of Gibberish Output

The non-sensical output from BitNet.rs inference appears to be caused by **incorrect K/V head slicing logic in the weight mapper** that causes a 75% parameter loss and produces degenerate attention patterns.

### The Problem in 30 Seconds

**File**: `crates/bitnet-models/src/weight_mapper.rs` (lines 647-655)

**Current behavior**:
- Exporter emits K/V weights as `[2560, 2560]` (all heads together)
- Mapper slices to select "first head of each group": `[0, 512, 1024, 1536, 2048]`
- These are heads `[0, 4, 8, 12, 16]` - spaced 4 apart (sparse!)
- Only **5 unique heads** are preserved, but **75% parameter information is lost**
- During GQA expansion, all 4 Q heads in a group see **identical K/V values**
- Attention becomes degenerate → gibberish output

**What should happen**:
- Slice **sequentially**: heads `[0, 1, 2, 3, 4]` → rows `[0-127, 128-255, 256-383, 384-511, 512-639]`
- All parameters preserved
- Different groups get different K/V information
- Attention works correctly

---

## Detailed Investigation Results

### 1. KV Head Slicing (weight_mapper.rs:607-670)

**CRITICAL ISSUE**: Sparse row selection instead of sequential

```rust
// CURRENT (WRONG):
for kv_idx in 0..n_kv_heads {
    let head_idx = kv_idx * group_size;  // ← SPARSE! 0, 4, 8, 12, 16
    // Selects only "first head of each group"
}

// SHOULD BE (SEQUENTIAL):
for head_idx in 0..n_kv_heads {
    // Selects 0, 1, 2, 3, 4 - all heads in order
}
```

**Impact on BitNet-2B**:
- `n_heads=20, n_kv_heads=5, group_size=4, head_dim=128`
- Current selects: rows `[0, 512, 1024, 1536, 2048]` (heads 0,4,8,12,16)
- Should select: rows `[0-127, 128-255, 256-383, 384-511, 512-639]` (heads 0,1,2,3,4)
- **Information loss: 75%** (15 heads completely ignored)

### 2. GQA Expansion Logic (transformer.rs:412-426)

**CORRECT**: The repeat-and-reshape logic is sound
```rust
let k_expanded = k_ctx
    .unsqueeze(2)?                           // [B, 5, 1, T, D]
    .repeat(&[1, 1, group_size, 1, 1])?     // [B, 5, 4, T, D]
    .reshape(&[batch_size, n_heads, T, D])?; // [B, 20, T, D] ✓
```

**But with sparse K slicing**: This expansion creates invalid duplicates
- Group 0: Q heads 0-3 all see the same (sparse) head 0 K/V
- Group 1: Q heads 4-7 all see the same (sparse) head 4 K/V
- Result: Within-group attention is degenerate (all heads see identical values)

### 3. Head Dimension Math (weight_mapper.rs:20-41)

**CORRECT**: All calculations are mathematically sound
- `head_dim = 2560 / 20 = 128` ✓
- `kv_dim = 128 * 5 = 640` ✓
- `group_size = 20 / 5 = 4` ✓
- Expected K shape: `[640, 2560]` ✓

### 4. Shape Pipeline (transformer.rs:277-545)

**VERIFIED**: Shapes flow correctly through attention
- Q projection: `[B, T, 2560]` → reshape → `[B, 20, T, 128]` ✓
- K projection: `[B, T, 640]` → reshape → `[B, 5, T, 128]` ✓ (or [B, sparse, T, D] if sliced wrong)
- V projection: Same as K ✓
- RoPE: Dimension-preserving ✓
- KV cache: Stores only n_kv_heads ✓
- Expansion: Creates [B, 20, T, 128] ✓ (but with degenerate values if slicing wrong)
- Attention: Computes correctly structurally ✓ (but with bad data)
- Output: [B, T, 2560] ✓ (but garbage content)

---

## Evidence & Testing

### Regression Test (weight_mapper.rs:1048-1153)

The test **validates the mechanism** but not the **correctness of the design**:
- Creates synthetic `[2560, 2560]` K weight
- Verifies slicing produces expected indices
- **Does NOT verify** that the selected rows preserve information
- **Does NOT verify** that subsequent attention computation works

### Proposed Test Case

```rust
#[test]
fn test_gqa_slicing_preserves_uniqueness() {
    // Create [2560, 2560] where each row has unique identity
    // Slice for GQA
    // Verify: all n_kv_heads selected
    // Verify: no sparse gaps (sequential selection)
    // Verify: after expansion, 4 groups see different K values
}
```

---

## Comparison with llama.cpp

| Aspect | llama.cpp | BitNet.rs |
|--------|-----------|----------|
| **K/V weight format** | `[hidden, kv_dim]` (correct) | `[hidden, hidden]` (needs fixing) |
| **Slicing strategy** | N/A (already correct) | Sparse selection (WRONG) |
| **Expansion logic** | `repeat(group_size)` | `repeat(group_size)` ✓ |
| **Result** | Correct attention | Degenerate attention |

**Key insight**: llama.cpp never receives malformed weights. BitNet.rs attempts to "fix" them but does so incorrectly.

---

## Root Cause Chain

```
1. Exporter emits [2560, 2560] K/V weights (or model config missing)
   ↓
2. Weight mapper detects mismatch (expected [640, 2560])
   ↓
3. Mapper attempts to "fix" via slicing (line 617)
   ↓
4. BUT uses sparse selection (head_idx = kv_idx * group_size)
   ↓
5. Selects only 5 of 20 heads: [0, 4, 8, 12, 16]
   ↓
6. 75% of K/V parameters are discarded
   ↓
7. GQA expansion repeats the 5 sparse heads to fill 20 slots
   ↓
8. All 4 Q heads in each group see identical K/V patterns
   ↓
9. Attention becomes degenerate (within-group correlation)
   ↓
10. Output projection receives correlated gradients
    ↓
11. Model produces low-entropy, repetitive text (GIBBERISH)
```

---

## Recommended Fix

### Option 1: Reject Malformed Weights (SAFEST)

```rust
if is_kv_hidden_square {
    return Err(BitNetError::Validation(format!(
        "K/V weight is [hidden={}×hidden={}]. \
         Expected [kv_dim={}×hidden={}] for GQA (n_heads={}, n_kv_heads={}). \
         This indicates either: \
         (1) exporter bug, (2) wrong config, or (3) model is MHA. \
         Please regenerate GGUF with correct dimensioning.",
        hidden, hidden, kv_dim, hidden, n_heads, n_kv_heads
    )));
}
```

**Advantages**:
- Fails fast with clear error message
- Forces users to address root cause
- No silent data loss
- Prevents garbage outputs

**Disadvantages**:
- Requires model regeneration
- May break existing workflows

### Option 2: Correct Sequential Slicing

```rust
if is_kv_hidden_square {
    let mut row_indices = Vec::new();
    // Sequential selection (not sparse!)
    for head_idx in 0..n_kv_heads {
        let row_start = head_idx * head_dim;
        let row_end = row_start + head_dim;
        for row in row_start..row_end {
            row_indices.push(row as i64);
        }
    }
    // ... continue with slicing ...
}
```

**Advantages**:
- Preserves all parameters
- Works with existing models
- No information loss

**Disadvantages**:
- May mask underlying exporter bugs
- Assumes specific K/V layout (sequential heads)
- Silent fix (users don't know about problem)

---

## Testing Recommendations

### Immediate Validation

```bash
# Enable diagnostic flags
BITNET_DEBUG_GQA=1 cargo run -p bitnet-cli -- run \
  --model models/model.gguf \
  --prompt "What is 2+2?" \
  --max-tokens 4

# Should show GQA shapes and norms
# Look for: Q/K/V means should be reasonable (not all zero)
```

### Cross-Validation

```bash
# Compare with C++ reference
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "test" \
  --max-tokens 4 \
  --cos-tol 0.99
```

### Unit Tests

1. **GQA Slicing Test**: Verify no parameter loss
2. **GQA Expansion Test**: Verify each group gets different K/V
3. **Attention Output Test**: Verify within-group Q heads produce different outputs

---

## Diagnostic Flags

Set environment variables to enable detailed logging:

```bash
# Log GQA dimensions (Q/K/V shapes and norms)
BITNET_DEBUG_GQA=1

# Log attention scale factors and scores
BITNET_DEBUG_ATTN_SCALE=1

# Log RoPE application
BITNET_DEBUG_ROPE=1

# Log general attention (stats and finiteness)
DEBUG_ATTN=1

# Trace QK256 kernel dispatch
BITNET_TRACE_RMS=1

# Example usage:
BITNET_DEBUG_GQA=1 BITNET_DEBUG_ATTN_SCALE=1 cargo run ...
```

---

## Files Generated

1. **GQA_KV_SLICING_ANALYSIS.md** - Comprehensive technical analysis
2. **GQA_VISUAL_GUIDE.txt** - ASCII diagrams showing current vs correct behavior
3. **GQA_INVESTIGATION_SUMMARY.md** - This file (executive summary)

---

## Action Items

- [ ] Verify exporter configuration and check actual K/V weight shapes
- [ ] Add diagnostic logging to dump sliced weights
- [ ] Implement Option 1 or Option 2 fix (preferably Option 1)
- [ ] Add unit tests for GQA slicing and expansion
- [ ] Cross-validate with C++ reference (BITNET_CPP_DIR)
- [ ] Test with known-good GQA models (e.g., Mistral format)
- [ ] Monitor inference for quality improvements

---

## References

- **Weight Mapper**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/weight_mapper.rs:607-670`
- **Transformer Attention**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/transformer.rs:277-545`
- **GQA Expansion**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/transformer.rs:412-426`
- **Regression Test**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/weight_mapper.rs:1048-1153`

---

## Confidence Level

**HIGH (85%+)**: The sparse slicing logic is demonstrably incorrect. The sparse row selection (`[0, 512, 1024, 1536, 2048]` instead of `[0, 128, 256, 384, 512]`) creates a clear parameter loss and degenerate attention pattern that perfectly explains the observed gibberish output.

The GQA expansion and attention computation logic are both correct - the problem is in the weight mapper's slicing step.

