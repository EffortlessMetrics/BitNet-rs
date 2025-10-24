# Policy System Test Results

## Executive Summary

✅ **Policy infrastructure is fully working**
❌ **Output still incoherent despite corrections**

## Test Configuration

- **Model**: `models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf`
- **Fingerprint**: `sha256-4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
- **Policy File**: `correction-policy.yml`
- **Corrections Applied**: 150 total
  - 90 I2_S Q/K/V projection corrections
  - 60 LayerNorm gamma rescaling corrections

## What's Working ✅

### 1. Fingerprint System
- Fingerprint computed correctly on model load
- Policy matched by fingerprint
- Corrections applied based on fingerprint match

### 2. I2_S Policy Corrections (90 tensors)
All Q/K/V projection tensors received `inv=true` corrections:
```
POLICY: I2_S override for 'blk.N.attn_k.weight': inv=true, k=1
POLICY: I2_S override for 'blk.N.attn_q.weight': inv=true, k=1
POLICY: I2_S override for 'blk.N.attn_v.weight': inv=true, k=1
```

Pattern matching working correctly:
- Policy uses `ends_with` matching
- Microsoft BitNet naming (`attn_q.weight`, `attn_k.weight`, `attn_v.weight`)
- All 30 layers × 3 tensors = 90 corrections applied

### 3. LayerNorm Corrections (60 tensors)
Environment variable `BITNET_FIX_LN_SCALE=1` working:
```
BITNET_FIX_LN_SCALE: rescaling 'blk.N.attn_norm.weight' gamma RMS 0.01801→≈1.0 (factor 55.529)
BITNET_FIX_LN_SCALE: rescaling 'blk.N.ffn_norm.weight' gamma RMS 1.29151→≈1.0 (factor 0.774)
```

Before correction:
- `attn_norm.weight`: RMS ~0.01-0.02 (❌ suspicious)
- `ffn_norm.weight`: RMS ~1.2-1.5 (✅ acceptable)

After correction:
- Both normalized to RMS ≈ 1.0

## What's Not Working ❌

### Inference Output Still Gibberish

**Input**: `"Test"`
**Output**: `"<<<<<<< Oprah phốJK �"`

This suggests one of:
1. **Corrections are insufficient**: Additional tensors may need inv=true
2. **Corrections are incorrect**: inv=true might not be the right fix for this model
3. **Deeper corruption**: Model file may have additional issues beyond Q/K/V and LayerNorm
4. **Inference engine issue**: Problem in attention mechanism, ROPE, or GQA

## Diagnostic Observations

### Missing RMS Logging
Q/K/V projection RMS values are NOT being logged because `is_projection_weight` uses `ends_with` patterns that don't match Microsoft naming:
- Expected patterns: `.q_proj.weight`, `.k_proj.weight`, `.v_proj.weight`
- Actual names: `.attn_q.weight`, `.attn_k.weight`, `.attn_v.weight`

This means we can't verify if Q/K/V projections have reasonable RMS values after correction.

### Transposition
All Q/K/V tensors go through transposition path:
```
DEBUG: Transposing projection tensor 'blk.0.attn_k.weight' from [2560, 640] to [640, 2560]
DEBUG: Transposing projection tensor 'blk.0.attn_q.weight' from [2560, 2560] to [2560, 2560]
DEBUG: Transposing projection tensor 'blk.0.attn_v.weight' from [2560, 640] to [640, 2560]
```

Configuration is being passed through `create_transposed_i2s_tensor_with_cfg`.

## Next Steps

### Option 1: Verify Projection RMS Values
Add logging to the transposed I2_S path to see if Q/K/V projections have reasonable RMS after correction.

**Expected**: RMS ~O(1) after inv=true
**If not**: The correction isn't working as intended

### Option 2: Test with Different Corrections
Try these variations:
1. **No inv, different k**: `inv=false, k=0.01` (scale down by 100x)
2. **Different tensor patterns**: Add other projection names
3. **All I2_S tensors**: Apply inv=true to ALL I2_S tensors, not just Q/K/V

### Option 3: Deep Diagnostic Run
Enable all debug flags and check:
```bash
export BITNET_DEBUG_ATTN_SCALE=1
export DEBUG_ATTN=1
export BITNET_DEBUG_RMSNORM=1
export BITNET_DEBUG_GQA=1
export BITNET_DEBUG_LOGITS=1
export BITNET_DEBUG_MLP=1
export BITNET_DEBUG_ROPE=1
```

Look for:
- Attention scores range (should be O(±10))
- Q/K/V means (should be O(1))
- GQA shapes correct
- Logits sanity

### Option 4: Compare with Reference
Run inference with same prompt on reference implementation (C++/Python) to verify model file is valid.

## Files Generated

1. ✅ `correction-policy.yml` - Policy file with fingerprint
2. ✅ `scripts/generate_policy.sh` - Policy generation script
3. ✅ `scripts/test_policy.sh` - Test script with both corrections

## Recommendations

1. **Short-term**: Use debug flags to diagnose where values go wrong
2. **Medium-term**: Add RMS logging for Microsoft-style projection names
3. **Long-term**: Regenerate GGUF with proper float LayerNorm and correct I2_S scales

## Test Reproducibility

```bash
# Generate policy file
./scripts/generate_policy.sh models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf

# Test with both corrections
export BITNET_CORRECTION_POLICY=./correction-policy.yml
export BITNET_FIX_LN_SCALE=1
./scripts/test_policy.sh
```

Corrections applied: **150**
Output: **Still gibberish** 😞
