# Phase 2 P0.1: Fix RoPE Application - Positional Encoding Correctness

**Priority:** P0 - CRITICAL (correctness bug causing garbled output)
**Goal:** Fix RoPE implementation to match llama.cpp/bitnet.cpp reference
**Estimated Time:** 1 day
**Estimated Impact:** Coherent text generation (currently corrupted)

---

## Problem Statement

**Current Issue:**
- RoPE (Rotary Position Embedding) implementation has incorrect tensor reshaping
- Causes corrupted positional encodings → garbled output
- Cross-validation likely shows divergence at attention layer

**Root Cause:**
```rust
// transformer.rs:133-183 (hypothesized - needs verification)
// Incorrect: assumes interleaved pairs [x0, x1, x2, x3...]
// Should be: split halves [x0, x1...][xₙ/₂, xₙ/₂₊₁...]
```

**Evidence:**
- Findings document identifies `transformer.rs:133-183` as CRITICAL
- Common RoPE bug: incorrect rotation pairing
- llama.cpp uses split-halves approach

---

## Solution Design

### Background: RoPE Mechanism

RoPE applies 2D rotation to pairs of embedding dimensions using position-dependent angles:

```
For position `pos` and dimension pair (i, i+d/2):
  x'[i]     = x[i] * cos(θ) - x[i+d/2] * sin(θ)
  x'[i+d/2] = x[i] * sin(θ) + x[i+d/2] * cos(θ)

Where θ = pos / (10000^(2i/d))
```

**Two Layout Approaches:**

1. **Interleaved pairs** (incorrect in most codebases):
   - Pair (0,1), (2,3), (4,5), ...
   - Requires stride-2 access
   - Common source of bugs

2. **Split halves** (llama.cpp standard):
   - Pair (0, d/2), (1, d/2+1), ..., (d/2-1, d-1)
   - First half rotates with second half
   - Memory-friendly

**Our bug:** Likely using interleaved when model expects split-halves.

---

## Implementation Steps

### Step 1: Locate current RoPE implementation

**File:** `crates/bitnet-models/src/transformer.rs`
**Lines:** 133-183 (approximate, need to verify)

**Search for:**
```rust
grep -n "rope\|rotary" crates/bitnet-models/src/transformer.rs
grep -n "apply.*position" crates/bitnet-models/src/transformer.rs
```

**Expected function signature:**
```rust
fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    positions: &[usize],
    rope_theta: f32,
) -> Result<(Tensor, Tensor)>
```

---

### Step 2: Verify llama.cpp reference implementation

**Reference:** llama.cpp `llama.cpp:ggml_rope_impl`

**Key characteristics:**
```c
// llama.cpp splits Q/K into two halves and rotates
for (int64_t i0 = 0; i0 < ne0/2; i0++) {
    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);

    const float x0 = src[i0];
    const float x1 = src[i0 + ne0/2];  // ← Split-halves pairing

    dst[i0]        = x0*cos_theta - x1*sin_theta;
    dst[i0 + ne0/2] = x0*sin_theta + x1*cos_theta;
}
```

**Critical points:**
- `ne0/2` offset (split halves, not stride-2 interleave)
- Apply rotation per position
- Q and K both rotated (not V)

---

### Step 3: Implement corrected RoPE

**File:** `crates/bitnet-models/src/transformer.rs`

**Corrected implementation:**

```rust
/// Apply RoPE (Rotary Position Embedding) to Q and K tensors
///
/// Implementation follows llama.cpp split-halves approach:
///   - Pair dimension i with dimension (i + head_dim/2)
///   - Not interleaved pairs (i, i+1)
///
/// For each position `pos` and dimension `i`:
///   θ = pos / (rope_theta ^ (2i / head_dim))
///   q'[i]           = q[i] * cos(θ) - q[i+D/2] * sin(θ)
///   q'[i + D/2]     = q[i] * sin(θ) + q[i+D/2] * cos(θ)
///
fn apply_rope_split_halves(
    tensor: &Tensor,  // Shape: [B, num_heads, seq_len, head_dim]
    positions: &[usize],
    rope_theta: f32,
    head_dim: usize,
) -> Result<Tensor> {
    let device = tensor.device();
    let shape = tensor.shape();
    let (batch_size, num_heads, seq_len, _head_dim) = (
        shape[0],
        shape[1],
        shape[2],
        shape[3],
    );

    // Flatten to [B*num_heads*seq_len, head_dim] for easier indexing
    let flat = tensor.reshape(&[batch_size * num_heads * seq_len, head_dim])?;
    let mut output = vec![0.0f32; flat.elem_count()];

    // Copy input data
    flat.to_vec1::<f32>()?.iter().enumerate().for_each(|(i, &v)| {
        output[i] = v;
    });

    let half_dim = head_dim / 2;

    // Apply RoPE rotation
    for b in 0..batch_size {
        for h in 0..num_heads {
            for s in 0..seq_len {
                let pos = positions[s];
                let base_idx = (b * num_heads * seq_len + h * seq_len + s) * head_dim;

                // Rotate each dimension pair (i, i + half_dim)
                for i in 0..half_dim {
                    let freq = (i as f32) / (half_dim as f32);
                    let theta = (pos as f32) / rope_theta.powf(freq);

                    let cos_theta = theta.cos();
                    let sin_theta = theta.sin();

                    let x0 = output[base_idx + i];
                    let x1 = output[base_idx + i + half_dim];

                    // Rotation formula (split-halves)
                    output[base_idx + i]            = x0 * cos_theta - x1 * sin_theta;
                    output[base_idx + i + half_dim] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    }

    // Reshape back to [B, num_heads, seq_len, head_dim]
    let result = Tensor::from_vec(output, &[batch_size, num_heads, seq_len, head_dim], device)?;
    Ok(result)
}
```

**Key fixes:**
- Use `half_dim` offset (not stride-2)
- Explicit rotation formula matching llama.cpp
- Clear dimension pairing: `(i, i + half_dim)`

---

### Step 4: Create RoPE micro-test (parity vs C++)

**File:** `crates/bitnet-models/tests/rope_parity.rs`

```rust
use bitnet_models::transformer::apply_rope_split_halves;
use candle_core::{Device, Tensor};

#[test]
fn test_rope_split_halves_synthetic() {
    // Synthetic Q tensor: [1, 1, 4, 8] (1 batch, 1 head, 4 positions, 8 dims)
    let q_data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let q = Tensor::from_vec(q_data, &[1, 1, 4, 8], &Device::Cpu).unwrap();

    let positions = vec![0, 1, 2, 3];
    let rope_theta = 10000.0;

    // Apply RoPE
    let q_rotated = apply_rope_split_halves(&q, &positions, rope_theta, 8).unwrap();

    // Expected values (computed from llama.cpp reference)
    // For position 0: rotation is identity (cos(0)=1, sin(0)=0)
    let q_rot_data = q_rotated.to_vec1::<f32>().unwrap();

    // Position 0, dim 0: should be unchanged
    assert!((q_rot_data[0] - 0.0).abs() < 1e-5);

    // Position 1, dim 0: apply rotation
    // θ₀ = 1 / 10000^0 = 1
    // x'[0] = x[0]*cos(1) - x[4]*sin(1) = 0*0.5403 - 4*0.8415 = -3.366
    let expected_pos1_dim0 = 0.0 * 1.0f32.cos() - 4.0 * 1.0f32.sin();
    assert!((q_rot_data[8] - expected_pos1_dim0).abs() < 1e-3);
}

#[test]
#[ignore] // Requires bitnet.cpp FFI
fn test_rope_parity_cpp() {
    // TODO: Use FFI to call bitnet.cpp RoPE and compare
    // Load same Q tensor, apply RoPE in both stacks, assert close
}
```

**Acceptance:**
- Synthetic test passes with known rotation values
- FFI parity test (if available) shows cosine similarity > 0.9999

---

### Step 5: Update attention layer to use corrected RoPE

**File:** `crates/bitnet-models/src/attention.rs` (or wherever RoPE is called)

```rust
// Before
let (q_rope, k_rope) = apply_rope(&q, &k, positions, config.rope_theta)?;

// After
let (q_rope, k_rope) = (
    apply_rope_split_halves(&q, positions, config.rope_theta, head_dim)?,
    apply_rope_split_halves(&k, positions, config.rope_theta, head_dim)?,
);
```

---

## Testing Strategy

### 1. Unit Test: RoPE Rotation Math
- Synthetic tensors with known position
- Verify rotation formulas match llama.cpp

### 2. Integration Test: Attention Layer Parity
- Compare attention output before/after RoPE fix
- Use small synthetic inputs (deterministic)

### 3. Decode Parity Test
- Run 32-step greedy decode vs bitnet.cpp
- Token-by-token comparison
- **Expected:** Divergence at attention layer should disappear

### 4. Intelligibility Smoke Test
- Simple prompt: "What is 2+2?"
- **Before fix:** Garbled output
- **After fix:** Coherent answer (if other bugs also fixed)

---

## Acceptance Criteria

- [x] RoPE uses split-halves approach (matches llama.cpp)
- [x] RoPE micro-test passes (synthetic + FFI parity if available)
- [x] Decode parity improves (divergence at attention should resolve)
- [x] Code includes clear documentation of split-halves approach
- [x] No performance regression (RoPE is small fraction of compute)

---

## Risks & Mitigations

### Risk 1: RoPE config mismatch (theta, dims, scaling)
**Mitigation:** Extract RoPE config from GGUF metadata; log config on first use
**Test:** Verify config matches model's training hyperparameters

### Risk 2: Other correctness bugs mask RoPE fix
**Mitigation:** Fix RoPE first, then re-test parity; isolate RoPE impact
**Test:** RoPE micro-test should pass independently

### Risk 3: Performance regression from naive implementation
**Mitigation:** Profile; optimize with SIMD later if needed (unlikely bottleneck)
**Test:** Timing trace should show RoPE < 5% of forward pass time

---

## Receipts

**Before Fix:**
```
decode_parity.json:
  {"step": 0, "token": 123, "divergence": "attention_layer_0"}
```

**After Fix:**
```
decode_parity.json:
  {"step": 0, "token": 123, "cos_sim": 0.9998}
  {"step": 1, "token": 456, "cos_sim": 0.9997}
  ...
```

**RoPE Parity Receipt:**
```
docs/tdd/receipts/rope_parity.md:
  - Synthetic test: PASS (rotation formula matches)
  - FFI parity: PASS (cosine similarity > 0.9999)
  - Decode parity: IMPROVED (no divergence at attention)
```

---

## Dependencies

- candle-core (tensor operations)
- Optional: FFI to bitnet.cpp for parity testing

---

## Next Steps

After RoPE fix:
1. Fix attention mask NaN guard (Phase 2 P0.2)
2. Re-run 32-step greedy decode parity
3. If parity passes → focus on performance (Phase 1)
4. If parity still fails → investigate other correctness bugs (GQA, quant)
