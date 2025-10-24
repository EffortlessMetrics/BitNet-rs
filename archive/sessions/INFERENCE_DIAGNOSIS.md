# BitNet.rs Inference Diagnosis Report

**Date**: 2025-10-12
**Issue**: Non-grammatical output from inference pipeline
**Status**: Root causes identified

## Executive Summary

The inference pipeline produces garbage output due to a cascading numerical instability that starts at Layer 0. The diagnostic run reveals that **attention scores explode to millions** before masking, the **mask produces all -inf values**, and **RMSNorm outputs are abnormally small**, leading to inf propagation through subsequent layers.

## Critical Findings (from diagnostic run)

### 1. **Attention Scores Explosion (Layer 0)**
```
scores pre-mask: mean=172548.890625 std=5911857.500000
scores post-mask: mean=-inf std=NaN
⚠️  scores post-mask: non-finite values: NaN=0 Inf=1320
Layer 0 scores post-mask range: min=-19440120.000000, max=39320316.000000
```

**Expected**: Scores should be O(1) to O(10) after scaling
**Actual**: Scores are in the **millions** before masking, then become -inf after masking
**Impact**: Softmax becomes degenerate (uniform distribution), losing all attention signal

### 2. **Q/K/V Values Are Abnormally Large (Layer 0)**
```
Q: mean=150.180283 std=935.747375
K: mean=78.541489 std=1005.442566
V: mean=235.969208 std=997.908936
```

**Expected**: Q/K/V values should be O(-5 to +5) for normalized models
**Actual**: Mean values of **150-235** with std dev of **935-1005**
**Impact**: Q @ K^T produces enormous values (~150 * 78 * 128 ≈ 1.5M per attention score)

### 3. **RMSNorm Output Is Too Small (Layer 0)**
```
RMSNorm (attn, layer 0) - input mean(x^2): 6.495476e-1, approx_rms: 8.059452e-1
RMSNorm (attn, layer 0) - output L2 norm: 1.807546e-2
```

**Expected**: RMSNorm should normalize to ~1.0 per element
**Actual**: L2 norm of **0.018** for a 2560-dim vector → RMS per element ≈ 0.00036
**Impact**: Tiny RMSNorm outputs are amplified by projection weights → Q/K/V explosion

### 4. **Cascade to Infinity After Layer 0**
```
[norm] post-attn: 2.622171e8
[norm] post-ffn: inf
[norm] layer 0: inf
[norm] input: inf  (Layer 1 input)
```

**Impact**: Once inf propagates past Layer 0 FFN, all subsequent layers produce inf

### 5. **Logits Are All Zeros**
```
tied logits sanity check - mean/std: 0.0000/0.0000
float ref logits - mean/std: 0.0000/0.0000
```

**Expected**: Logits should have large std dev (~10-100) for vocabulary distribution
**Actual**: Both tied and float reference produce **all zeros**
**Impact**: Sampling degenerates to random token selection (explains garbage output)

## Root Cause Chain

```
1. RMSNorm produces tiny outputs (0.018 L2 norm) [WHY?]
   ↓
2. Q/K/V projection weights are I2_S quantized with large scales
   ↓
3. Small RMSNorm × Large weights = Huge Q/K/V values (mean ~150)
   ↓
4. Q @ K^T produces scores in millions
   ↓
5. Causal mask adds -inf, but scores already too large
   ↓
6. Max-subtraction applied, but softmax still degenerates
   ↓
7. Attention output is corrupted
   ↓
8. FFN norm receives inf input
   ↓
9. Everything goes to inf
   ↓
10. Final norm = 0.05, logits = 0.0
```

## Hypotheses for RMSNorm Issue

### Hypothesis A: RMSNorm Weight Vector Is Too Small
The `attention_norm.weight` tensor might have been:
- Incorrectly dequantized (wrong scale factor)
- Stored with wrong quantization type in GGUF
- Corrupted during model conversion

**Test**: Inspect `blk.0.attn_norm.weight` tensor stats

### Hypothesis B: I2_S Scale Clamping Is Too Aggressive
Current clamp: `s.clamp(1e-6, 1e6)` in `i2s_dequant_block`

Some scales we saw:
```
scale_fp16=1.020312e1  → final_s=1.020312e1  (10.2)
scale_fp16=1.688004e-4 → final_s=1.688004e-4 (0.000168)
```

The second scale (0.000168) might be causing tiny dequantized values in the RMSNorm weight.

**Test**: Tighten clamp to `s.clamp(1e-3, 1e3)` and observe if Q/K/V values normalize

### Hypothesis C: RMSNorm Epsilon Configuration Mismatch
Reported epsilon: `0.00001` (1e-5)

If the epsilon is being applied incorrectly (e.g., added after sqrt instead of before), the normalization magnitude would be wrong.

**Test**: Add explicit epsilon logging inside RMSNorm forward pass

## Immediate Fixes to Test

### Fix 1: Inspect RMSNorm Weights
```rust
// Add after line 577 in transformer.rs
if std::env::var("BITNET_DEBUG_RMSNORM").is_ok() {
    let weight = self.attention_norm.weight();
    let weight_stats = weight.mean_all()?.to_scalar::<f32>()?;
    let weight_l2 = weight.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    tracing::info!("RMSNorm weight - mean: {:.6e}, L2: {:.6e}", weight_stats, weight_l2);
}
```

### Fix 2: Tighten I2_S Scale Clamping
```rust
// In crates/bitnet-models/src/quant/i2s.rs:71
- s = s.clamp(1e-6, 1e6);
+ s = s.clamp(1e-3, 1e3);
```

### Fix 3: Add Post-Norm Validation
```rust
// After line 596 in transformer.rs
let x_normalized = self.attention_norm.forward(x)?;
let norm_check = x_normalized.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
if norm_check > 10.0 || norm_check < 0.01 {
    tracing::warn!("⚠️  RMSNorm output L2 norm out of range: {:.6e}", norm_check);
}
```

### Fix 4: Scale Attention Scores Defensively
```rust
// After line 358 in transformer.rs
+ // Additional defensive scaling if scores are too large
+ if let Ok(max_score) = scores.abs()?.max_keepdim(D::Minus1)?
+     && let Ok(max_val) = max_score.max_all()?.to_scalar::<f32>()
+     && max_val > 1000.0
+ {
+     tracing::warn!("⚠️  Attention scores too large (max={:.2e}), applying emergency scaling", max_val);
+     let scores = scores.affine(1.0 / max_val as f64, 0.0)?;
+ }
```

## Next Steps (Recommended Order)

1. **Run Fix 1** → Inspect RMSNorm weight statistics
2. **If RMSNorm weights are tiny** → Investigate GGUF loading/dequantization for `blk.*.attn_norm.weight`
3. **Test Fix 2** → Tighten I2_S clamp and rerun diagnostic
4. **If still broken** → Add detailed logging around LayerNorm::rms_norm forward pass (Candle internals)
5. **Emergency fallback** → Use Fix 4 to prevent score explosion while debugging root cause

## Acceptance Criteria

After fixes, the diagnostic should show:
```
✅ RMSNorm (attn, layer 0) - output L2 norm: ~5.0 to 15.0
✅ Q: mean=O(-1 to +1), std=O(1 to 3)
✅ scores pre-mask: mean=O(-10 to +10), std=O(1 to 20)
✅ scores post-mask: min=O(-50 to 0), max=O(0 to 10)
✅ [norm] post-ffn: O(1 to 10) (NOT inf)
✅ tied logits - mean/std: O(-5 to +5) / O(10 to 100)
```

## Files Requiring Investigation

1. `crates/bitnet-models/src/transformer.rs:577-612` (RMSNorm diagnostics)
2. `crates/bitnet-models/src/quant/i2s.rs:71` (Scale clamping)
3. `crates/bitnet-models/src/formats/gguf/loader.rs:932` (Embedding transpose detection)
4. `crates/bitnet-models/src/weight_mapper.rs` (Tensor mapping logic)

---

**Status**: Ready for targeted fixes. Recommend starting with Fix 1 (inspect RMSNorm weights).
