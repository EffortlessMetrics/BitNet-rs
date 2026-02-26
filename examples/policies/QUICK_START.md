# Policy System Quick Start

Get started with bitnet-rs validation policies in 60 seconds.

## TL;DR

```bash
# Validate your model (auto-selects policy based on GGUF metadata)
cargo run -p bitnet-cli -- inspect --ln-stats --gate auto your-model.gguf

# If validation fails, check what's wrong
cargo run -p bitnet-cli -- inspect --ln-stats --verbose your-model.gguf
```

## Three Steps to Validation

### Step 1: Inspect Your Model

```bash
cargo run -p bitnet-cli -- inspect --ln-stats your-model.gguf
```

**Output tells you:**
- LayerNorm gamma RMS statistics
- Projection weight RMS ranges
- Whether values are suspicious or OK

### Step 2: Choose Your Policy

| If your model is... | Use this policy |
|---------------------|-----------------|
| BitNet b1.58 F16 (unquantized) | `--gate auto` or `bitnet-b158-f16-clean.yml` |
| BitNet b1.58 I2_S (quantized) | `--gate auto` or `bitnet-b158-i2s-quantized.yml` |
| LLaMA, Mistral, or generic RMSNorm | `--gate auto` or `llama-generic.yml` |
| Custom architecture | Create custom policy (see below) |

### Step 3: Validate

```bash
# Automatic policy selection (recommended)
cargo run -p bitnet-cli -- inspect --ln-stats --gate auto your-model.gguf

# Or use specific policy
cargo run -p bitnet-cli -- inspect --ln-stats \
  --policy examples/policies/bitnet-b158-f16-clean.yml \
  --policy-key bitnet-b1.58:f16 \
  your-model.gguf
```

## Common Scenarios

### ✅ Scenario 1: Your Model Passes Validation

```bash
$ cargo run -p bitnet-cli -- inspect --ln-stats --gate auto model.gguf

✓ All 24 layers pass validation
✓ Projection weights in acceptable range

Status: CLEAN - No issues detected
```

**Action:** You're good to go! Use the model for inference.

### ⚠️ Scenario 2: "Suspicious LayerNorm gamma" Warnings

```bash
$ cargo run -p bitnet-cli -- inspect --ln-stats --gate auto model.gguf

✗ Layer 0: attn_norm RMS = 0.018 [SUSPICIOUS - expected ~1.0]
✗ 24/24 layers have suspicious gamma RMS values

Root cause: LayerNorm weights quantized to I2_S (should be FP16/FP32)
```

**Action:** Regenerate your GGUF with LayerNorm in FP16/FP32:

```bash
# Using st2gguf (recommended for BitNet)
cargo run -p bitnet-st2gguf -- \
  --input model.safetensors \
  --output model-fixed.gguf \
  --strict

# Or using llama.cpp quantization tools
./quantize model-f16.gguf model-i2s.gguf i2_s --skip-layernorm
```

**Temporary workaround** (development only):
```bash
# See correction-policy-sample.yml for examples
export BITNET_CORRECTION_POLICY=./config/correction-policy.yml
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
cargo run -p bitnet-cli -- run --model model.gguf
```

### ⚠️ Scenario 3: "Projection weight RMS out of range"

```bash
$ cargo run -p bitnet-cli -- inspect --ln-stats --gate auto model.gguf

✗ blk.0.attn_q.weight RMS = 150.0 [OUT OF RANGE: expected 0.01-0.40]
```

**Possible causes:**
1. I2_S scales are inverted
2. Wrong quantization type in metadata
3. Export tool configuration issue

**Diagnostic:**
```bash
# Check projection RMS with debug logs
RUST_LOG=info cargo run -p bitnet-cli -- run --model model.gguf 2>&1 | grep "PROJ load"

# If Q/K/V RMS >> FFN RMS (by 10×+), scales may be inverted
# Use I2S_DEQUANT_OVERRIDE correction (see correction-policy-sample.yml)
```

### ❌ Scenario 4: Wrong Policy Auto-Selected

```bash
$ cargo run -p bitnet-cli -- inspect --ln-stats --gate auto model.gguf

✗ Layer 0: ffn_norm RMS = 0.08 [OUT OF RANGE: expected 0.80-1.20]

Policy: llama-generic.yml (auto-selected)
```

**Action:** Your model needs a BitNet-specific policy:

```bash
# Use explicit policy
cargo run -p bitnet-cli -- inspect --ln-stats \
  --policy examples/policies/bitnet-b158-f16-clean.yml \
  --policy-key bitnet-b1.58:f16 \
  model.gguf
```

Or fix GGUF metadata so auto-detection works correctly.

## Creating a Custom Policy

If none of the standard policies fit your model:

### 1. Inspect and Measure

```bash
cargo run -p bitnet-cli -- inspect --ln-stats --verbose your-model.gguf > stats.txt
```

Review `stats.txt` to find:
- Min/max RMS for each LayerNorm type
- Projection weight RMS distribution

### 2. Copy Template

```bash
cp examples/policies/custom-model-example.yml my-model-policy.yml
```

### 3. Define Thresholds

Edit `my-model-policy.yml`:

```yaml
version: 1

rules:
  my-model:f16:
    name: "My Custom Model F16"
    ln:
      # Based on inspection: attn_norm RMS range [0.92, 1.05]
      # Policy envelope: [0.85, 1.15] (add 10% margin)
      - pattern: "attn_norm\\.weight$"
        min: 0.85
        max: 1.15
        description: "Attention LayerNorm (measured: 0.92-1.05)"

      # Add more patterns as needed...

    proj_weight_rms_min: 0.015
    proj_weight_rms_max: 0.35
```

### 4. Test Your Policy

```bash
cargo run -p bitnet-cli -- inspect --ln-stats \
  --policy my-model-policy.yml \
  --policy-key my-model:f16 \
  your-model.gguf
```

## Environment Variables

### Quick Setup for Different Modes

```bash
# Mode 1: No validation (debugging only)
export BITNET_VALIDATION_GATE=none

# Mode 2: Auto-detect policy (recommended)
export BITNET_VALIDATION_GATE=auto

# Mode 3: Explicit policy
export BITNET_VALIDATION_GATE=policy
export BITNET_VALIDATION_POLICY=examples/policies/my-policy.yml
export BITNET_VALIDATION_POLICY_KEY=my-model:f16

# Mode 4: Strict validation (fail on warnings)
export BITNET_STRICT_MODE=1
```

## Integration with Inference

Validation runs automatically during model loading:

```bash
# Run inference with validation
export BITNET_VALIDATION_GATE=auto
cargo run -p bitnet-cli --no-default-features --features cpu -- run \
  --model your-model.gguf \
  --prompt "Test prompt"

# Validation happens once during model load
# No performance impact on inference throughput
```

## CI/CD Integration

Add to your CI pipeline:

```bash
#!/bin/bash
# ci-validate-model.sh

set -euo pipefail

MODEL="${1:?Missing model path}"

# Enable strict validation
export BITNET_STRICT_MODE=1
export BITNET_VALIDATION_GATE=auto

# Validate model
if ! cargo run -p bitnet-cli -- inspect --ln-stats "$MODEL"; then
  echo "ERROR: Model validation failed"
  echo "LayerNorm weights may be corrupted or incorrectly quantized"
  exit 1
fi

echo "✓ Model validation passed"
```

Usage in GitHub Actions:

```yaml
# .github/workflows/validate-models.yml
- name: Validate model export
  run: |
    ./scripts/ci-validate-model.sh artifacts/model.gguf
```

## Troubleshooting Commands

```bash
# 1. Check GGUF metadata (architecture, file_type)
cargo run -p bitnet-cli -- inspect --metadata model.gguf

# 2. View detailed LayerNorm statistics
cargo run -p bitnet-cli -- inspect --ln-stats --verbose model.gguf

# 3. Test different policies
for policy in bitnet-b158-f16-clean bitnet-b158-i2s-quantized llama-generic; do
  echo "=== Testing $policy ==="
  cargo run -p bitnet-cli -- inspect --ln-stats \
    --policy examples/policies/${policy}.yml \
    model.gguf
  echo ""
done

# 4. Check projection weight RMS
RUST_LOG=info cargo run -p bitnet-cli -- run --model model.gguf 2>&1 | grep "PROJ load"

# 5. Compare two models
diff <(cargo run -p bitnet-cli -- inspect --ln-stats model-a.gguf) \
     <(cargo run -p bitnet-cli -- inspect --ln-stats model-b.gguf)
```

## FAQ

**Q: Do I always need to specify a policy?**

A: No. Use `--gate auto` to automatically select the right policy based on your model's GGUF metadata.

**Q: What's the performance impact?**

A: Validation adds 5-10ms to model loading time. No impact on inference throughput.

**Q: Can I skip validation?**

A: Yes, with `--gate none`, but this disables important safety checks. Only use for debugging.

**Q: What if my model legitimately has unusual RMS values?**

A: Create a custom policy with appropriate thresholds based on your model's empirical statistics.

**Q: Are corrections suitable for production?**

A: No. Corrections are temporary workarounds. CI blocks correction flags to prevent production deployment. Always regenerate GGUF with proper weight formats.

## Next Steps

- **Read full guide**: [`README.md`](./README.md)
- **Compare policies**: [`POLICY_COMPARISON.md`](./POLICY_COMPARISON.md)
- **Understand corrections**: [`docs/explanation/correction-policy.md`](../../docs/explanation/correction-policy.md)
- **Clean GGUF export**: [`docs/howto/export-clean-gguf.md`](../../docs/howto/export-clean-gguf.md)

## Support

If validation fails unexpectedly:

1. Run `cargo run -p bitnet-cli -- inspect --ln-stats --verbose model.gguf`
2. Review the RMS statistics and compare with expected values
3. Check if you're using the right policy for your architecture
4. Create a custom policy if needed (use `custom-model-example.yml` as template)
5. Open a GitHub issue if you believe validation is incorrect

## Examples Directory Contents

```
examples/policies/
├── README.md                           ← Full documentation
├── QUICK_START.md                      ← This file
├── POLICY_COMPARISON.md                ← Policy comparison table
├── bitnet-b158-f16-clean.yml          ← BitNet F16 policy
├── bitnet-b158-i2s-quantized.yml      ← BitNet I2_S policy
├── llama-generic.yml                   ← LLaMA/generic policy
└── custom-model-example.yml            ← Template for custom policies
```

## License

These policy files are provided under the same license as bitnet-rs (MIT OR Apache-2.0).
