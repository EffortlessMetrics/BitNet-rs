# LayerNorm Comprehensive Fix Report - Issue #254

**Date**: 2025-10-24
**Status**: Fixes Applied, Testing In Progress
**Root Cause**: Multiple shape bugs + LayerNorm vs RMSNorm semantics

---

## Executive Summary

Investigation revealed that garbled inference output is caused by a combination of:
1. **Shape handling bugs** in `forward_full()` method (singleton dimension not squeezed)
2. **LayerNorm vs RMSNorm semantics** (should use LayerNorm with mean subtraction, not RMSNorm)
3. **Possible gamma rescaling requirement** (GGUF contains pre-scaled gamma ≈ 1/√H)

All identified bugs have been fixed, but output remains garbled, suggesting additional investigation needed.

---

## Fixes Applied

### Fix 1: Shape Bug - Squeeze Singleton Dimension

**File**: `crates/bitnet-models/src/transformer.rs`
**Line**: 1429

**Problem**: `narrow(1, t, 1)` creates `[B, 1, H]` tensor, but `forward()` expects `[B, H]`

**Fix Applied**:
```rust
// Before:
let step_hidden = hidden.narrow(1, t, 1)?;

// After:
let step_hidden = hidden.narrow(1, t, 1)?.squeeze(1)?;
```

**Impact**: Ensures correct 2D tensor shape `[B, H]` for per-token processing

---

### Fix 2: Shape Assertion for Forward Output

**File**: `crates/bitnet-models/src/transformer.rs`
**Lines**: 1435-1441

**Fix Applied**:
```rust
// Ensure forward preserves expected shape [B, H]
if step_hidden.dims().len() != 2 {
    return Err(BitNetError::Validation(format!(
        "forward() should return [B, H] shape, got {:?}",
        step_hidden.dims()
    )));
}
```

**Impact**: Early detection of shape mismatches in forward pass

---

### Fix 3: Logits Concatenation Logic

**File**: `crates/bitnet-models/src/transformer.rs`
**Lines**: 1447-1455

**Problem**: Concatenation assumed `[B, 1, V]` logits, but after squeeze should be `[B, V]`

**Fix Applied**:
```rust
// Stack logits: handle both [B,V] and [B,1,V] shapes
let logits = if logits_steps[0].dims().len() == 2 {
    // logits are [B, V], stack them to [B, T, V]
    let logits_2d: Vec<_> = logits_steps
        .iter()
        .map(|t| t.unsqueeze(1))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Tensor::cat(&logits_2d, 1)?
} else {
    // logits are [B, 1, V], concatenate along time dimension
    Tensor::cat(&logits_steps, 1)?
};
Ok(logits)
```

**Impact**: Correctly handles both 2D and 3D logit shapes

---

### Fix 4: LayerNorm Semantics (RMSNorm → LayerNorm)

**File**: `crates/bitnet-models/src/transformer.rs`
**Lines**: 77-87

**Problem**: Used `LayerNorm::rms_norm()` (no mean subtraction) instead of `LayerNorm::new_no_bias()` (with mean subtraction)

**Fix Applied**:
```rust
// Before:
Err(_) => {
    tracing::debug!(
        "Bias tensor missing for norm layer; using RMSNorm (no mean subtraction) [{}]",
        normalized_shape
    );
    Ok(LayerNorm::rms_norm(weight, eps))
}

// After:
Err(_) => {
    // IMPORTANT: Use LayerNorm::new_no_bias (remove_mean=true) NOT rms_norm (remove_mean=false)
    // because the gamma weights in GGUF are calibrated for LayerNorm semantics (mean subtraction).
    // bitnet.cpp uses full LayerNorm even when bias is absent.
    tracing::debug!(
        "Bias tensor missing for norm layer; using LayerNorm without bias (mean subtraction enabled) [{}]",
        normalized_shape
    );
    Ok(LayerNorm::new_no_bias(weight, eps))
}
```

**Impact**: Matches bitnet.cpp LayerNorm semantics (mean subtraction even without bias)

---

## Test Results

### LayerNorm Unit Tests: ✅ All Pass (7/7)

```bash
cargo test -p bitnet-models --test layernorm_fix_tests --no-default-features --features cpu
```

**Tests**:
1. ✅ `test_layernorm_tensor_names_never_classified_as_quantized` - Validates LN names detected
2. ✅ `test_layernorm_uses_mean_subtraction_not_rmsnorm` - Proves mean subtraction occurs
3. ✅ `test_layernorm_normalizes_over_last_dimension` - Validates normalization axis
4. ✅ `test_layernorm_normalizes_per_position_independently` - Per-token normalization
5. ✅ `test_layernorm_output_differs_from_rmsnorm` - LayerNorm ≠ RMSNorm
6. ✅ `test_layernorm_with_gamma_scaling` - Gamma scaling works correctly
7. ✅ `test_layernorm_stability_with_various_inputs` - Edge case handling

**Key Finding from Tests**:
```
✓ LayerNorm mean: -0.000001 (≈ 0)
✓ RMSNorm mean: 0.932934 (≠ 0)
✓ Average absolute difference: 0.932935
✓ This confirms LayerNorm (mean subtraction) ≠ RMSNorm (no mean subtraction)
```

---

### Inference Test: ❌ Output Still Garbled

**Command**:
```bash
export RUST_LOG=warn BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=4
target/release/bitnet run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 16 \
  --temperature 0.0 \
  --greedy
```

**Output**:
```
'E-lived,SIGNALConvert Paperback Gab Rug、、 ventgetModelれている ${!"
 coefficient maximize Fiber
```

**Analysis**: Output remains non-sensical despite all fixes

---

## Investigation Findings

### Gamma Weight RMS Analysis

**Observation**: `attn_norm` gamma weights have RMS ≈ 0.018

**From `inspect --ln-stats`**:
```
blk.0.attn_norm.weight    [LN]  rms=0.0180   ❌
blk.0.ffn_norm.weight     [LN]  rms=1.2915   ❌
blk.1.attn_norm.weight    [LN]  rms=0.0169   ❌
blk.1.ffn_norm.weight     [LN]  rms=1.2318   ❌
```

**Mathematical Evidence**:
- Expected if gamma = 1/√2560: 0.01976
- Observed: 0.0180
- Ratio: 0.0180 / 0.01976 = 0.911 (91% match)
- √2560 = 50.60

**Hypothesis**: GGUF contains pre-scaled gamma that needs rescaling on load

---

## Candle LayerNorm Implementation Analysis

**Source**: `~/.cargo/registry/src/.../candle-nn-*/src/layer_norm.rs`

**Key Code**:
```rust
impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden_size = x.dim(D::Minus1)?;  // Last dimension size
        let x = if self.remove_mean {
            let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
            x.broadcast_sub(&mean_x)?  // Subtract mean over last dim
        } else {
            x  // RMSNorm: no mean subtraction
        };
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        // ...
    }
}
```

**Confirmed Behavior**:
- `D::Minus1` = Last dimension only
- `new_no_bias`: `remove_mean = true` (LayerNorm)
- `rms_norm`: `remove_mean = false` (RMSNorm)
- Normalization is per-token over hidden dimension ✅ Correct

---

## Current Status

### What's Working ✅
1. Shape handling - `[B, 1, H]` → `[B, H]` squeeze applied
2. LayerNorm semantics - mean subtraction enabled via `new_no_bias()`
3. Normalization axis - last dimension (hidden_size) confirmed correct
4. Tests - all 7 LayerNorm unit tests pass

### What's Not Working ❌
1. Inference output still garbled
2. Gamma RMS values abnormally small (~0.018 for attn_norm)
3. Unknown if bitnet.cpp rescales gamma on load

---

## Next Steps

### Option A: Test Gamma Rescaling
Test with `BITNET_RESCALE_GAMMA_ON_LOAD=1` to see if output improves:
```bash
export BITNET_RESCALE_GAMMA_ON_LOAD=1
target/release/bitnet run --model <model.gguf> --tokenizer <tokenizer.json> \
  --prompt "What is 2+2?" --max-tokens 16
```

**If output improves**: Gamma rescaling is needed (model-specific quirk or bitnet.cpp behavior)
**If output still garbled**: Additional bugs exist in inference pipeline

### Option B: Compare Against bitnet.cpp
If `BITNET_CPP_DIR` is available, run cross-validation:
```bash
export BITNET_CPP_DIR=/path/to/bitnet.cpp
cargo run -p xtask -- crossval
```

Check if bitnet.cpp rescales gamma on load by inspecting their GGUF loader.

### Option C: Debug Activation Norms
Add debug logging to track activation magnitudes:
```bash
export DEBUG_ATTN=1
target/release/bitnet run --model <model.gguf> --prompt "Test" --max-tokens 4
```

Compare activation norms at each layer to identify where values diverge.

---

## Files Modified

1. `crates/bitnet-models/src/transformer.rs`
   - Line 1429: Add `.squeeze(1)` to narrow operation
   - Lines 1435-1441: Add shape assertion for forward output
   - Lines 1447-1455: Fix logits concatenation logic
   - Lines 77-87: Change `rms_norm()` to `new_no_bias()`

2. `crates/bitnet-models/tests/layernorm_fix_tests.rs` (NEW)
   - 7 comprehensive LayerNorm validation tests

---

## References

- **Investigation Documents**:
  - `INVESTIGATION_FINDINGS_SUMMARY.txt`
  - `RMSNORM_SEMANTIC_MISMATCH_ANALYSIS.md`
  - `RMSNORM_DIAGNOSTIC_RESULTS.md`
  - `LAYERNORM_INVESTIGATION.md`
  - `LAYERNORM_CODE_ANALYSIS.md`

- **Candle Documentation**:
  - LayerNorm source: `~/.cargo/registry/src/.../candle-nn-*/src/layer_norm.rs`
  - Formula: `y = ((x - mean) / sqrt(var + eps)) * gamma + beta`

- **BitNet-rs Documentation**:
  - `docs/environment-variables.md` - Gamma rescaling flags
  - `CLAUDE.md` - Known issues and workarounds

---

## Conclusion

All identified shape bugs and LayerNorm semantic issues have been fixed. Code compiles successfully and unit tests pass. However, inference output remains garbled, suggesting either:

1. Gamma rescaling IS required for this specific GGUF (model export quirk)
2. bitnet.cpp applies additional transformations we haven't replicated
3. There are additional bugs in the inference pipeline not yet identified

**Recommended Action**: Test with gamma rescaling enabled to determine if this is the missing piece, then investigate bitnet.cpp's GGUF loader to confirm their behavior.
