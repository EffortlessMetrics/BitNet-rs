# GQA (Grouped Query Attention) and KV Head Slicing Analysis

## Executive Summary

The BitNet-rs codebase implements Grouped Query Attention (GQA) with sophisticated KV head slicing in the weight mapper and dynamic expansion during inference. Analysis reveals **CRITICAL SHAPE MISMATCH ISSUE** in the weight mapper's GQA handling that could cause attention corruption.

## Issue Overview

**Symptom**: Non-sensical gibberish output from inference
**Root Cause**: Incorrect row selection logic in `weight_mapper.rs` line 617-656 for GQA K/V slicing

**Log Evidence**:
```
WARN layer23: Sliced K/V [hidden,hidden] -> [kv_dim,hidden] (GQA group_size=1)
```

This warning indicates the mapper is attempting to slice K/V projections, suggesting the exporter emitted square `[hidden, hidden]` matrices instead of proper `[kv_dim, hidden]` dimensions.

---

## 1. KV Head Slicing Logic (weight_mapper.rs)

### Current Implementation (Lines 607-670)

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/weight_mapper.rs:617`

```rust
// Special case: allow hidden×hidden only for K/V (exporter emitted full hidden)
// We need to slice heads NOW in the mapper to produce correct [kv_dim, hidden] shape
let is_kv_hidden_square = matches!(shape, [o, i] if (name == "k_proj" || name == "v_proj") && *o == hidden && *i == hidden);

if is_kv_hidden_square {
    let n_heads = dims.n_head;
    let n_kv_heads = dims.n_kv_head;
    let head_dim = hidden / n_heads;
    let group_size = n_heads / n_kv_heads;

    // Slice heads: [hidden, hidden] -> [kv_dim, hidden]
    // Build row indices for selected heads
    let mut row_indices = Vec::with_capacity(n_kv_heads * head_dim);
    for kv_idx in 0..n_kv_heads {
        let head_idx = kv_idx * group_size;  // First head of this group
        let row_start = head_idx * head_dim;
        let row_end = row_start + head_dim;
        for row in row_start..row_end {
            row_indices.push(row as i64);
        }
    }

    // Create index tensor and slice
    let idx_tensor = candle_core::Tensor::new(row_indices.as_slice(), weight.device())?;
    let sliced = weight.index_select(&idx_tensor, 0)?; // Select rows
    tensors.insert(format!("layers.{}.attention.{}.weight", layer_idx, name), sliced);
}
```

### Critical Problem: Incorrect Row Selection

**ISSUE**: The current logic selects the "first head of each group" but this is WRONG for GQA expansion.

**Current Behavior**:
```
For GQA with n_heads=20, n_kv_heads=5, group_size=4, head_dim=128:
- Selects rows: [0-127, 512-639, 1024-1151, 1536-1663, 2048-2175]
- These are: head 0, head 4, head 8, head 12, head 16
```

**Why This Is Wrong**:
1. **The K/V weights are NOT distributed as one head per group**
2. The exporter likely emitted weights structured as `[all_hidden, hidden]`
3. Slicing to just the "first head" of each group loses 75% of parameter capacity
4. During expansion (repeat `group_size` times), this means **all group members see identical attention patterns**

### What Should Happen Instead

For GQA, the K/V projections should:
1. **Output shape** should be `[n_kv_heads * head_dim, hidden]` = `[kv_dim, hidden]`
2. If exporter emitted `[hidden, hidden]`:
   - This is an error in the exporter (or model format issue)
   - Should NOT be silently "fixed" by slicing
   - Should emit a clear error instead

### Evidence from Regression Test

The test in `weight_mapper.rs` (lines 1068-1153) creates a synthetic `[2560, 2560]` K weight and verifies slicing produces the expected indices. **However**, this test validates the MECHANISM, not the CORRECTNESS of the design.

---

## 2. GQA Expansion Logic (transformer.rs)

### Current Implementation (Lines 412-426)

```rust
// GQA core: expand K/V to Hq heads (repeat along head axis)
let t_k = k_ctx.dims()[2];

// Expand K: [B, HKV, Tk, D] -> [B, Hq, Tk, D]
let k_expanded = k_ctx
    .unsqueeze(2)?                               // [B, HKV, 1, Tk, D]
    .repeat(&[1, 1, self.group_size, 1, 1])?    // [B, HKV, group, Tk, D]
    .reshape(&[batch_size, self.n_heads, t_k, self.head_dim])?; // [B, Hq, Tk, D]

// Expand V: [B, HKV, Tk, D] -> [B, Hq, Tk, D]
let v_expanded = v_ctx
    .unsqueeze(2)?                               // [B, HKV, 1, Tk, D]
    .repeat(&[1, 1, self.group_size, 1, 1])?    // [B, HKV, group, Tk, D]
    .reshape(&[batch_size, self.n_heads, t_k, self.head_dim])?; // [B, Hq, Tk, D]
```

### Analysis

**CORRECT**: The expansion logic is sound:
- Takes `[B, n_kv_heads, Tk, head_dim]` K/V
- Repeats along head dimension `group_size` times
- Reshapes to `[B, n_heads, Tk, head_dim]`
- Each KV head is shared by `group_size` Q heads

**Shape Flow**:
```
Input:  [B, 5, T, 128]  (5 KV heads)
↓ unsqueeze(2)
        [B, 5, 1, T, 128]
↓ repeat [1,1,4,1,1]  (group_size=4)
        [B, 5, 4, T, 128]
↓ reshape [B, 20, T, 128]  (5*4=20 Q heads)
Output: [B, 20, T, 128] ✓
```

---

## 3. Head Dimension Calculations

### Verification

**Correct Calculations** (from `weight_mapper.rs:20-41`):

```rust
fn head_dim(&self) -> Result<usize> {
    let d = self.hidden / self.n_head;
    Ok(d)
}

fn kv_head_dim(&self) -> Result<usize> {
    self.head_dim()  // Same as head_dim
}

fn q_dim(&self) -> Result<usize> {
    Ok(self.head_dim()? * self.n_head)  // = hidden
}

fn kv_dim(&self) -> Result<usize> {
    Ok(self.kv_head_dim()? * self.n_kv_head)  // = head_dim * n_kv_heads
}
```

**Verification for BitNet-2B**:
- `hidden_size = 2560`
- `n_heads = 20`
- `n_kv_heads = 5`
- `head_dim = 2560 / 20 = 128` ✓
- `kv_dim = 128 * 5 = 640` ✓
- `group_size = 20 / 5 = 4` ✓

**Expected Weight Shapes**:
- Q: `[2560, 2560]` ✓ (q_dim × hidden)
- K: `[640, 2560]` ✓ (kv_dim × hidden)
- V: `[640, 2560]` ✓ (kv_dim × hidden)
- O: `[2560, 2560]` ✓ (hidden × q_dim)

---

## 4. Shape Transformations Through Pipeline

### Layer 0: Weight Mapper
```
Raw GGUF K:  [2560, 2560]  (exporter issue!)
Mapped K:    [640, 2560]   (via row slicing - QUESTIONABLE)
```

### Layer 1: Forward Pass (Projection)
```
Input X:         [B, T, 2560]
K weight:        [640, 2560]  (or [640, 2560] from mapper)
K projection:    X @ K^T → [B, T, 640]
```

**PROBLEM**: If weight_mapper sliced incorrectly:
```
Expected K @ X: [640, 2560] × [B, T, 2560]^T → [B, T, 640]
Actual output:  ???? (depends on slicing correctness)
```

### Layer 2: Reshape to Heads
```
K output:        [B, T, 640]
Reshape:         [B, T, 5, 128]  (n_kv_heads=5, head_dim=128)
Transpose:       [B, 5, T, 128]  ✓
```

### Layer 3: RoPE Application
```
Apply to:        [B, 5, T, 128]
Result:          [B, 5, T, 128]  ✓ (dimension-preserving)
```

### Layer 4: KV Cache Storage
```
Cache stores:    [B, 5, T, 128]  ✓ (only n_kv_heads)
```

### Layer 5: GQA Expansion
```
Input:           [B, 5, T, 128]
Expand:          [B, 20, T, 128]  ✓
Broadcast:       Each KV head → 4 Q heads
```

---

## 5. Comparison with llama.cpp

### How llama.cpp Handles GQA

In **llama.cpp** (struct `llm_layer`):

```cpp
// K/V projections are pre-dimensioned:
// w_kv is shape [hidden, kv_dim]
// V is shape [hidden, kv_dim]

// During forward:
llama_tensor_k = X @ w_k  // [seq_len, hidden] @ [hidden, kv_dim] → [seq_len, kv_dim]
llama_tensor_v = X @ w_v  // [seq_len, hidden] @ [hidden, kv_dim] → [seq_len, kv_dim]

// Reshape to heads:
llama_tensor_k_heads = reshape([seq_len, n_kv_heads, head_dim])

// GQA expansion (repeat):
for (int q = 0; q < n_heads; q++) {
    int kv_idx = q / group_size;  // Which KV head to use
    llama_tensor_k_expanded[q] = llama_tensor_k_heads[kv_idx];  // repeat
}
```

### Key Difference

**llama.cpp**: K/V weights are correctly dimensioned `[hidden, kv_dim]` from the exporter
**BitNet-rs**: Attempts to "fix" `[hidden, hidden]` weights via row slicing

---

## 6. Root Cause Analysis

### Why Does the Exporter Emit [hidden, hidden]?

Possible causes:

1. **Model Format Issue**: Some GGUF exporters may output full-hidden dimensions for all attention projections
2. **Multi-Query Attention Confusion**: Exporter might emit Q, K, V with same shape
3. **Missing Configuration**: Config lacks `num_key_value_heads` or specifies it as 0

### Code Evidence (transformer.rs:216)
```rust
let n_kv_heads = config.model.num_key_value_heads.max(1).min(n_heads);
```

If `num_key_value_heads == 0` or missing:
- Falls back to `n_heads` (Multi-Head Attention)
- `group_size = n_heads / n_heads = 1` ✓
- In this case, K/V should be `[hidden, hidden]` - **and slicing produces wrong results!**

---

## 7. Attention Computation Impact

### If Slicing is Wrong

```
Scenario: n_heads=20, n_kv_heads=5, group_size=4

Correct K slicing (5 heads, all parameters):
  Row indices: [0..128, 128..256, 256..384, 384..512, 512..640]
  Selected: 5 complete heads

Current implementation:
  Row indices: [0..128, 512..640, 1024..1152, 1536..1664, 2048..2176]
  Selected: 1 head per group (sparse selection)

Result:
  - During expansion, all 4 Q heads in a group see IDENTICAL K/V patterns
  - Loss of 75% of parameter information
  - Attention becomes degenerate: all Q heads in a group produce same output
```

### Symptom: Gibberish Output

When attention weights are degenerate:
1. Many attention patterns collapse to identical values
2. Different query heads produce correlated outputs
3. Output projection can't distinguish between heads
4. Results in low-entropy, repetitive text

**This matches the observed "non-sensical output"!**

---

## 8. Proposed Fixes

### Option 1: Reject Malformed Weights (RECOMMENDED)

```rust
if is_kv_hidden_square {
    tracing::error!(
        "layer{}: K/V weight is [hidden, hidden] which indicates model format issue. \
         Expected [kv_dim={}, hidden={}] where kv_dim=n_kv_heads*head_dim={} from config",
        layer_idx,
        kv_dim,
        hidden,
        n_kv_heads * (hidden / n_heads)
    );
    
    return Err(BitNetError::Validation(format!(
        "Cannot auto-fix GQA K/V shapes: exporter produced [{}×{}], \
         expected [{}×{}] for GQA with n_heads={}, n_kv_heads={}. \
         Regenerate GGUF with correct dimensioning or update exporter config.",
        hidden, hidden, kv_dim, hidden, n_heads, n_kv_heads
    )));
}
```

### Option 2: Correct Slicing (If Intentional)

If `[hidden, hidden]` means "all heads mixed together":

```rust
if is_kv_hidden_square {
    // All n_heads worth of K/V are embedded in [hidden, hidden] matrix
    // Extract only the first n_kv_heads heads worth
    let mut row_indices = Vec::new();
    for head_idx in 0..n_kv_heads {
        let row_start = head_idx * head_dim;
        let row_end = row_start + head_dim;
        for row in row_start..row_end {
            row_indices.push(row as i64);
        }
    }
    // (sequential head selection, not sparse)
}
```

---

## 9. Testing Recommendations

### Test Case 1: Verify Slicing Preserves Information

```rust
#[test]
fn test_gqa_slicing_preserves_head_parameters() {
    // Create [2560, 2560] matrix with unique values per head
    // Slice it for GQA
    // Verify NO parameters are lost (sequential selection)
    // Verify all n_kv_heads × head_dim rows selected
}
```

### Test Case 2: Verify Expansion Correctness

```rust
#[test]
fn test_gqa_expansion_creates_correct_replication() {
    // Input: [B, n_kv_heads, T, D] K/V
    // Expand: [B, n_heads, T, D]
    // Verify: each row i of expanded K == K[i / group_size]
}
```

### Test Case 3: Cross-Model Validation

```bash
# Test with a known-good GQA model (e.g., Mistral-7B format)
# Verify inference output matches llama.cpp reference
cargo run -p xtask -- crossval-per-token \
  --model models/test_gqa.gguf \
  --prompt "Test prompt" \
  --max-tokens 4
```

---

## 10. Summary Table

| Component | Current | Status | Impact |
|-----------|---------|--------|--------|
| **Weight Mapper Slicing** | Selects 1 head/group | ⚠️ WRONG | 75% parameter loss |
| **GQA Expansion** | Repeat along axis | ✓ CORRECT | Proper broadcasting |
| **Head Dimension Math** | `hidden/n_heads` | ✓ CORRECT | Validated |
| **Shape Through Pipeline** | Traced (see above) | ⚠️ DEPENDS | On mapper correctness |

---

## 11. Immediate Actions

1. **Verify Exporter Configuration**:
   - Check if exporter is setting `num_key_value_heads` correctly
   - Check if K/V weights are truly `[hidden, hidden]` or if mapper is misinterpreting

2. **Add Diagnostics**:
   - Log actual vs. expected K/V weight shapes
   - Dump first few rows of sliced weights to verify content

3. **Implement Fix** (Option 1 recommended):
   - Either reject malformed weights with clear error
   - Or implement correct sequential slicing

4. **Validate with Cross-Validation**:
   - Compare sliced weights with C++ reference
   - Verify attention output matches llama.cpp

---

## References

- **BitNet-rs Weight Mapper**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/weight_mapper.rs:607-670`
- **BitNet-rs Transformer**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/transformer.rs:277-545`
- **Test Case**: `weight_mapper.rs:1048-1153`
- **GQA Expansion Diagnostic**: Set `BITNET_DEBUG_GQA=1` to log dimensions

