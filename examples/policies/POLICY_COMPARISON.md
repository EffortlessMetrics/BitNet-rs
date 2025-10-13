# Policy Comparison Guide

Quick reference for selecting the appropriate validation policy for your model.

## Policy Selection Flowchart

```
┌─────────────────────────────────────────┐
│ What architecture is your model?       │
└─────────────────────────────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
    BitNet b1.58        LLaMA/Mistral/
    Architecture        Generic RMSNorm
        │                    │
        │                    │
        v                    v
┌──────────────────┐   ┌─────────────────────┐
│ What precision/  │   │ llama-generic.yml   │
│ quantization?    │   │                     │
└──────────────────┘   │ - Conservative      │
        │              │ - RMS ~[0.8, 1.2]   │
        │              └─────────────────────┘
        │
┌───────┴────────┐
│                │
F16          I2_S/Quantized
(unquantized)     │
│                │
v                v
┌─────────────────────────────┐   ┌──────────────────────────────┐
│ bitnet-b158-f16-clean.yml   │   │ bitnet-b158-i2s-quantized.yml│
│                             │   │                              │
│ - Clean F16 exports         │   │ - I2_S quantized models      │
│ - FFN norm: [0.05, 2.0]     │   │ - Attn norm: [0.01, 2.0]     │
│ - Attn norm: [0.50, 2.0]    │   │ - FFN norm: [0.50, 2.0]      │
│ - Proj RMS: [0.01, 0.40]    │   │ - Proj RMS: [0.002, 0.20]    │
└─────────────────────────────┘   └──────────────────────────────┘

┌─────────────────────────────────────────┐
│ None of the above?                      │
│                                         │
│ custom-model-example.yml (template)    │
│ - Create your own based on inspection  │
│ - Measure RMS statistics first         │
└─────────────────────────────────────────┘
```

## Threshold Comparison Table

### LayerNorm Gamma RMS Thresholds

| Layer Type | LLaMA-Generic | BitNet-F16 | BitNet-I2_S | Notes |
|------------|---------------|------------|-------------|-------|
| **attn_norm / attention_norm** | [0.80, 1.20] | [0.50, 2.0] | [0.01, 2.0] | I2_S legitimately has very low RMS (~0.01-0.02) |
| **ffn_norm / ffn_layernorm** | [0.80, 1.20] | [0.05, 2.0] | [0.50, 2.0] | BitNet FFN norms architecturally low |
| **post_attention_layernorm** | [0.80, 1.20] | [0.25, 2.0] | N/A | BitNet-specific naming |
| **input_layernorm** | [0.80, 1.20] | [0.35, 2.0] | N/A | BitNet-specific naming |
| **final_norm / output_norm** | [0.80, 1.20] | [0.50, 2.0] | [0.50, 2.0] | Critical for stability |
| **Generic norm fallback** | [0.80, 1.20] | [0.50, 2.0] | [0.25, 2.0] | Catches any norm not matched above |

### Projection Weight RMS Thresholds

| Policy | Min RMS | Max RMS | Notes |
|--------|---------|---------|-------|
| **LLaMA-Generic** | null | null | No projection validation (too architecture-dependent) |
| **BitNet-F16** | 0.01 | 0.40 | F16 projection weights (typical: 0.01-0.25) |
| **BitNet-I2_S** | 0.002 | 0.20 | I2_S dequantization produces smaller RMS |

## Use Case Matrix

| Scenario | Recommended Policy | Gate Mode | Command Example |
|----------|-------------------|-----------|-----------------|
| **BitNet F16 from st2gguf** | `bitnet-b158-f16-clean.yml` | `auto` or `policy` | `--gate auto` or `--policy bitnet-b158-f16-clean.yml --policy-key bitnet-b1.58:f16` |
| **BitNet I2_S quantized** | `bitnet-b158-i2s-quantized.yml` | `auto` or `policy` | `--gate auto` or `--policy bitnet-b158-i2s-quantized.yml --policy-key bitnet-b1.58:i2_s` |
| **LLaMA-3 8B** | `llama-generic.yml` | `auto` | `--gate auto` |
| **Mistral 7B** | `llama-generic.yml` | `auto` | `--gate auto` |
| **Custom architecture** | `custom-model-example.yml` (customize) | `policy` | `--policy my-custom-policy.yml --policy-key my-model:f16` |
| **Debugging validation** | None | `none` | `--gate none` |

## Policy Strictness Levels

From most permissive to most strict:

1. **`none` (no validation)**
   - No checks performed
   - Use only for debugging validation system
   - ⚠️ Disables important safety checks

2. **`bitnet-b158-i2s-quantized.yml` (most permissive)**
   - Accommodates I2_S quantization artifacts
   - attn_norm min: 0.01 (very low, but legitimate for I2_S)
   - Suitable for quantized BitNet models

3. **`bitnet-b158-f16-clean.yml` (moderate)**
   - Wider envelopes than generic LLaMA
   - FFN norm min: 0.05 (architectural design)
   - Suitable for unquantized BitNet

4. **`llama-generic.yml` (strict)**
   - Narrow envelope [0.80, 1.20]
   - Assumes standard RMSNorm initialization (~1.0)
   - Suitable for most transformer models

## Auto-Detection Logic

When using `--gate auto`, policy is selected based on GGUF metadata:

```python
def select_policy(gguf_metadata):
    arch = gguf_metadata["general.architecture"].lower()
    file_type = gguf_metadata["general.file_type"]

    if "bitnet" in arch or "b1.58" in arch:
        if file_type == 1:  # F16
            return "bitnet-b158-f16-clean.yml"
        else:  # Quantized (I2_S, Q4_0, etc.)
            return "bitnet-b158-i2s-quantized.yml"
    else:
        return "llama-generic.yml"
```

### File Type Constants

| file_type | Quantization | Description |
|-----------|--------------|-------------|
| 0 | F32 | 32-bit float |
| 1 | F16 | 16-bit float (unquantized) |
| 2 | Q4_0 | 4-bit quantization (block-wise) |
| 3 | Q4_1 | 4-bit quantization (with offsets) |
| 28 | I2_S | 2-bit signed (BitNet) |

## Common Validation Scenarios

### Scenario 1: Clean F16 Export Passes Validation

```bash
$ cargo run -p bitnet-cli -- inspect --ln-stats --gate auto model-f16.gguf

LayerNorm Statistics Analysis
=============================
Model: bitnet-b1.58-2B-4T (F16)
Policy: bitnet-b158-f16-clean.yml (auto-selected)

✓ Layer  0: attn_norm RMS = 0.95  [OK: 0.50-2.0]
✓ Layer  0: ffn_norm  RMS = 0.08  [OK: 0.05-2.0]  ← Architecturally low
✓ Layer  1: attn_norm RMS = 0.98  [OK: 0.50-2.0]
...
✓ All 24 layers pass validation

✓ Projection RMS range: [0.018, 0.32] [OK: 0.01-0.40]

Status: CLEAN - No issues detected
```

### Scenario 2: Quantized LayerNorm Detected

```bash
$ cargo run -p bitnet-cli -- inspect --ln-stats --gate auto bad-model.gguf

LayerNorm Statistics Analysis
=============================
Model: bitnet-b1.58-2B-4T (I2_S)
Policy: bitnet-b158-i2s-quantized.yml (auto-selected)

✗ Layer  0: attn_norm RMS = 0.018 [SUSPICIOUS - expected ~1.0]
✗ Layer  1: attn_norm RMS = 0.019 [SUSPICIOUS - expected ~1.0]
...
✗ 24/24 layers have suspicious gamma RMS values

Root cause: LayerNorm weights quantized to I2_S (should be FP16/FP32)

Status: SUSPICIOUS
Recommendation: Regenerate GGUF with --skip-layernorm-quantization
Temporary workaround: Use correction policy (see docs/explanation/correction-policy.md)
```

### Scenario 3: Wrong Policy Selected

```bash
$ cargo run -p bitnet-cli -- inspect --ln-stats \
  --policy llama-generic.yml --policy-key generic \
  bitnet-model-f16.gguf

LayerNorm Statistics Analysis
=============================
Model: bitnet-b1.58-2B-4T (F16)
Policy: llama-generic.yml (manually specified)

✗ Layer  0: ffn_norm  RMS = 0.08  [OUT OF RANGE: expected 0.80-1.20]
                                   ^^^ FALSE POSITIVE

Status: FAILED
Note: This model may require bitnet-specific policy (try --gate auto)
```

## Troubleshooting Decision Tree

```
Model validation fails?
│
├─ Check 1: Are you using the right policy?
│  │
│  ├─ BitNet model? → Use bitnet-b158-*.yml
│  └─ LLaMA/generic? → Use llama-generic.yml
│
├─ Check 2: Is your model quantized?
│  │
│  ├─ F16 (unquantized) → Use *-f16-*.yml variant
│  └─ I2_S/quantized → Use *-i2s-*.yml or *-quantized.yml variant
│
├─ Check 3: Inspect actual RMS statistics
│  │
│  └─ cargo run -p bitnet-cli -- inspect --ln-stats --verbose model.gguf
│     │
│     ├─ RMS far from expected (e.g., 0.02 vs 1.0)
│     │  └─ LayerNorm was quantized → Regenerate GGUF
│     │
│     └─ RMS slightly outside envelope (e.g., 0.78 vs min 0.80)
│        └─ Create custom policy with wider envelope
│
└─ Check 4: Is this a custom architecture?
   │
   └─ Use custom-model-example.yml as template
      └─ Define thresholds based on your model's measurements
```

## Performance Considerations

### Validation Overhead

| Gate Mode | Load Time Impact | Inference Impact | Use Case |
|-----------|------------------|------------------|----------|
| `none` | 0ms | 0ms | Debugging only |
| `auto` | +5-10ms | 0ms | Production (recommended) |
| `policy` | +5-10ms | 0ms | Custom policies |
| `policy` + corrections | +10-50ms | +1-2ms per inference | Dev only (CI blocks) |

Validation runs once during model loading - negligible impact on inference throughput.

## Quick Reference Commands

### Inspect Model Statistics

```bash
# Auto-select policy and validate
cargo run -p bitnet-cli -- inspect --ln-stats --gate auto model.gguf

# Use specific policy
cargo run -p bitnet-cli -- inspect --ln-stats \
  --policy examples/policies/bitnet-b158-f16-clean.yml \
  --policy-key bitnet-b1.58:f16 \
  model.gguf

# Skip validation (debugging)
cargo run -p bitnet-cli -- inspect --ln-stats --gate none model.gguf
```

### Compare Policies

```bash
# Test model against multiple policies
for policy in llama-generic bitnet-b158-f16-clean bitnet-b158-i2s-quantized; do
  echo "Testing $policy..."
  cargo run -p bitnet-cli -- inspect --ln-stats \
    --policy examples/policies/${policy}.yml \
    model.gguf
done
```

### Validate in CI

```bash
# Strict validation (fail on any warnings)
export BITNET_STRICT_MODE=1
export BITNET_VALIDATION_GATE=auto

cargo run -p bitnet-cli -- inspect --ln-stats model.gguf

if [ $? -ne 0 ]; then
  echo "ERROR: Model validation failed"
  exit 1
fi
```

## Related Documentation

- **[README.md](./README.md)**: Comprehensive policy system guide
- **[custom-model-example.yml](./custom-model-example.yml)**: Template for creating custom policies
- **[docs/explanation/correction-policy.md](../../docs/explanation/correction-policy.md)**: Correction policy system
- **[CLAUDE.md](../../CLAUDE.md)**: Quick reference for LayerNorm troubleshooting

## Summary

**Choose your policy based on:**

1. **Architecture**: BitNet vs LLaMA/generic
2. **Quantization**: F16 (unquantized) vs I2_S (quantized)
3. **Use case**: Production (strict) vs development (permissive)

**When in doubt:**
- Start with `--gate auto` (automatic policy selection)
- If validation fails, inspect with `--ln-stats --verbose`
- Create custom policy if your model doesn't fit standard profiles

**Remember:**
- Validation catches export issues early (before inference)
- Corrections are temporary - always regenerate clean GGUF for production
- CI blocks correction flags to prevent deployment with workarounds
