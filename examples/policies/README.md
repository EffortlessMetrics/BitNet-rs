# BitNet.rs Validation Policy Examples

This directory contains example validation policies for the BitNet.rs LayerNorm and projection weight validation system. These policies define acceptable RMS (Root Mean Square) ranges for model weights to detect quantization corruption, export issues, and architectural anomalies.

## Overview

BitNet.rs includes a policy-driven validation system that checks model weights during loading and inference. Policies serve two purposes:

1. **Validation Rules**: Define acceptable RMS ranges for LayerNorm gamma and projection weights
2. **Correction Policies**: Provide runtime fixes for known-bad models (temporary workarounds)

## Available Policy Files

### Production Policies

| File | Architecture | Quantization | Use Case |
|------|-------------|--------------|----------|
| [`bitnet-b158-f16-clean.yml`](./bitnet-b158-f16-clean.yml) | BitNet b1.58 | F16 (unquantized) | Clean F16 exports from st2gguf |
| [`bitnet-b158-i2s-quantized.yml`](./bitnet-b158-i2s-quantized.yml) | BitNet b1.58 | I2_S (2-bit signed) | I2_S quantized models |
| [`llama-generic.yml`](./llama-generic.yml) | LLaMA-style | Any | Standard RMSNorm transformers |

### Templates and Examples

| File | Description |
|------|-------------|
| [`custom-model-example.yml`](./custom-model-example.yml) | Comprehensive template for creating custom policies |

## Quick Start

### 1. Inspect Your Model

Before selecting or creating a policy, inspect your model's LayerNorm statistics:

```bash
cargo run -p bitnet-cli -- inspect --ln-stats path/to/model.gguf
```

Example output:
```
LayerNorm Statistics Analysis
=============================
Model: bitnet-b1.58-2B-4T (I2_S quantization)
Total layers: 24

Per-layer gamma RMS statistics:
Layer  0: attn_norm RMS = 0.018 [SUSPICIOUS - expected ~1.0]
Layer  1: attn_norm RMS = 0.019 [SUSPICIOUS - expected ~1.0]
...

Summary:
- 24/24 layers have suspicious gamma RMS values
- Root cause: LayerNorm weights quantized to I2_S (should be FP16/FP32)
- Recommendation: Use bitnet-b158-i2s-quantized.yml policy
```

### 2. Select Appropriate Policy

Based on your model architecture and quantization:

**For BitNet b1.58 models:**
- F16 export: Use `bitnet-b158-f16-clean.yml`
- I2_S quantized: Use `bitnet-b158-i2s-quantized.yml`

**For LLaMA/Mistral/standard RMSNorm models:**
- Use `llama-generic.yml`

**For custom architectures:**
- Start with `custom-model-example.yml` and customize

### 3. Validate with Policy

```bash
# Validate using specific policy
cargo run -p bitnet-cli -- inspect --ln-stats \
  --policy examples/policies/bitnet-b158-f16-clean.yml \
  --policy-key bitnet-b1.58:f16 \
  path/to/model.gguf

# Validate with automatic policy selection (based on GGUF metadata)
cargo run -p bitnet-cli -- inspect --ln-stats \
  --gate auto \
  path/to/model.gguf
```

## Validation Gates

BitNet.rs supports three validation gate modes:

### 1. `none` - No Validation (Legacy Behavior)

```bash
# Skip validation entirely
cargo run -p bitnet-cli -- inspect --ln-stats --gate none model.gguf
```

**Use when:**
- Debugging validation system itself
- Working with experimental models where validation is not yet defined
- Testing inference without validation overhead

**Warning:** This disables important safety checks. Only use for development.

### 2. `auto` - Automatic Policy Selection

```bash
# Auto-select policy based on GGUF metadata
cargo run -p bitnet-cli -- inspect --ln-stats --gate auto model.gguf

# Or set environment variable
export BITNET_VALIDATION_GATE=auto
cargo run -p bitnet-cli -- run --model model.gguf
```

**Auto-detection logic:**
- Reads `general.architecture` and `general.file_type` from GGUF metadata
- BitNet models (architecture contains "bitnet" or "b1.58"):
  - `file_type = 1` (F16) → `bitnet-b158-f16-clean.yml`
  - `file_type ≠ 1` (quantized) → `bitnet-b158-i2s-quantized.yml`
- Other architectures → `llama-generic.yml`

**Use when:**
- Working with standard model formats (BitNet F16/I2_S, LLaMA, etc.)
- CI/CD pipelines requiring automatic validation
- Production inference where policy selection should be deterministic

### 3. `policy` - Explicit Policy File

```bash
# Use specific policy file and key
cargo run -p bitnet-cli -- inspect --ln-stats \
  --gate policy \
  --policy examples/policies/custom-model-example.yml \
  --policy-key custom-model:f16 \
  model.gguf

# Or set environment variables
export BITNET_VALIDATION_GATE=policy
export BITNET_VALIDATION_POLICY=examples/policies/custom-model-example.yml
export BITNET_VALIDATION_POLICY_KEY=custom-model:f16
cargo run -p bitnet-cli -- run --model model.gguf
```

**Use when:**
- Working with custom model architectures
- Overriding auto-detection for specific validation requirements
- Testing new policy definitions before committing to auto-detection

## Policy File Structure

### Validation Rules Section

Defines acceptable RMS ranges for LayerNorm and projection weights:

```yaml
version: 1

rules:
  # Policy key (use with --policy-key flag)
  model-name:variant:
    name: "Human-readable name"

    # LayerNorm gamma weight validation
    ln:
      # Patterns are regex tested against tensor names
      - pattern: "attn_norm\\.weight$"
        min: 0.85      # Minimum acceptable RMS
        max: 1.15      # Maximum acceptable RMS
        description: |
          Explanation of why these thresholds are appropriate

      - pattern: "ffn_norm\\.weight$"
        min: 0.40
        max: 1.50

    # Projection weight RMS validation (set to null to skip)
    proj_weight_rms_min: 0.015
    proj_weight_rms_max: 0.35

    notes: |
      Additional context about this policy
```

### Corrections Section (Optional)

Defines runtime fixes for known-bad models (**temporary workarounds only**):

```yaml
version: 1

models:
  # Fingerprint identifies specific model
  - fingerprint: "sha256-abc123def456..."
    notes: |
      Description of issue and proper fix

    corrections:
      # Rescale LayerNorm gamma to target RMS
      - type: LN_GAMMA_RESCALE_RMS
        target_rms: 1.0
        clamp: [0.01, 100.0]

      # Override I2_S dequantization for specific tensors
      - type: I2S_DEQUANT_OVERRIDE
        tensors:
          - "attn_q\\.weight"
          - "attn_k\\.weight"
        inv: false
        k: 1.0
```

**Important:** Corrections require:
- `BITNET_CORRECTION_POLICY=/path/to/policy.yml`
- `BITNET_ALLOW_RUNTIME_CORRECTIONS=1`
- CI blocks these flags to prevent production deployment

See [`docs/explanation/correction-policy.md`](../../docs/explanation/correction-policy.md) for detailed correction policy documentation.

## Creating a Custom Policy

### Step-by-Step Guide

#### 1. Inspect Your Model

```bash
cargo run -p bitnet-cli -- inspect --ln-stats your-model.gguf > stats.txt
```

Review the output to understand:
- LayerNorm gamma RMS distribution across layers
- Projection weight RMS ranges
- Any suspicious values or outliers

#### 2. Determine Thresholds

Based on inspection output, calculate policy envelopes:

**General approach:**
- Observe min/max RMS across all layers of each type
- Add 5-10% margin on each side for safety
- Be stricter for critical layers (e.g., final output norm)
- Be more permissive for architecturally variable layers

**Example calculation:**

| Layer Type | Observed Min | Observed Max | Policy Min (−10%) | Policy Max (+10%) |
|------------|--------------|--------------|-------------------|-------------------|
| attn_norm  | 0.92         | 1.05         | 0.85              | 1.15              |
| ffn_norm   | 0.42         | 0.52         | 0.35              | 0.60              |
| final_norm | 0.98         | 1.02         | 0.95              | 1.05              |

#### 3. Create Policy File

Copy template and customize:

```bash
cp examples/policies/custom-model-example.yml my-model-policy.yml
nano my-model-policy.yml
```

Define rules based on your threshold calculations:

```yaml
version: 1

rules:
  my-model:f16:
    name: "My Custom Model F16"
    ln:
      - pattern: "attn_norm\\.weight$"
        min: 0.85
        max: 1.15
        description: "Attention LayerNorm (observed RMS ~0.92-1.05)"

      - pattern: "ffn_norm\\.weight$"
        min: 0.35
        max: 0.60
        description: "FFN LayerNorm (architectural low gamma ~0.40-0.52)"

      - pattern: "final_norm\\.weight$"
        min: 0.95
        max: 1.05
        description: "Final norm (critical for stability)"

    proj_weight_rms_min: 0.015
    proj_weight_rms_max: 0.35
```

#### 4. Validate Policy

Test your policy against the model:

```bash
cargo run -p bitnet-cli -- inspect --ln-stats \
  --policy my-model-policy.yml \
  --policy-key my-model:f16 \
  your-model.gguf
```

Expected output:
```
✓ All LayerNorm weights pass validation (24/24 layers)
✓ Projection weights pass validation (RMS range: 0.018-0.32)
```

#### 5. Test Inference

Run inference with your policy:

```bash
export BITNET_VALIDATION_GATE=policy
export BITNET_VALIDATION_POLICY=my-model-policy.yml
export BITNET_VALIDATION_POLICY_KEY=my-model:f16

cargo run -p bitnet-cli --no-default-features --features cpu -- run \
  --model your-model.gguf \
  --prompt "Test prompt"
```

#### 6. Document and Commit

Add comprehensive documentation to your policy:

- **Architecture details**: Model type, layer count, special characteristics
- **Threshold justification**: Why these specific min/max values
- **Known issues**: Any quirks or edge cases
- **Related models**: Other models using the same policy

Commit to version control:

```bash
git add my-model-policy.yml
git commit -m "feat(validation): add policy for My Custom Model architecture

- Defines LayerNorm RMS envelopes based on empirical analysis
- attn_norm: [0.85, 1.15] (observed ~0.92-1.05)
- ffn_norm: [0.35, 0.60] (architectural low gamma)
- Projection RMS: [0.015, 0.35]
- Tested on my-custom-model v2.0 (F16 export)"
```

## Common Issues and Solutions

### Issue: "Suspicious LayerNorm gamma RMS" Warnings

**Symptom:**
```
WARNING: blk.0.attn_norm.weight RMS=0.018 [SUSPICIOUS - expected ~1.0]
```

**Possible Causes:**

1. **LayerNorm was quantized** (should always be FP16/FP32)
   - **Solution:** Regenerate GGUF with `--skip-layernorm-quantization`
   - **Temporary workaround:** Use correction policy (see `correction-policy-sample.yml`)

2. **Using wrong policy** (e.g., LLaMA policy for BitNet model)
   - **Solution:** Use correct policy for your architecture
   - BitNet I2_S: Low attn_norm RMS (~0.01-0.02) is EXPECTED

3. **Architectural design** (model legitimately has low gamma)
   - **Solution:** Create custom policy with appropriate thresholds
   - Document why low gamma is intentional

**Diagnostic Commands:**

```bash
# Inspect LayerNorm statistics in detail
cargo run -p bitnet-cli -- inspect --ln-stats --verbose model.gguf

# Check GGUF metadata (architecture, file_type)
cargo run -p bitnet-cli -- inspect --metadata model.gguf

# Test with different policies
cargo run -p bitnet-cli -- inspect --ln-stats \
  --policy examples/policies/bitnet-b158-i2s-quantized.yml \
  --policy-key bitnet-b1.58:i2_s \
  model.gguf
```

### Issue: "Projection weight RMS out of range"

**Symptom:**
```
WARNING: blk.0.attn_q.weight RMS=150.0 [OUT OF RANGE: expected 0.01-0.40]
```

**Possible Causes:**

1. **I2_S scales are inverted**
   - **Solution:** Use `I2S_DEQUANT_OVERRIDE` correction (see `correction-policy-sample.yml`)
   - Check export tool configuration

2. **Wrong quantization type in metadata**
   - **Solution:** Verify `general.file_type` in GGUF matches actual quantization
   - Regenerate with correct metadata

3. **Policy thresholds too narrow**
   - **Solution:** Inspect actual projection RMS distribution
   - Create custom policy with appropriate bounds

**Diagnostic Commands:**

```bash
# View projection weight statistics
RUST_LOG=info cargo run -p bitnet-cli -- run --model model.gguf 2>&1 | grep "PROJ load"

# Example output:
# INFO PROJ load: blk.0.attn_q.weight RMS=150.0 (inv=false)
# INFO PROJ load: blk.0.ffn_gate.weight RMS=0.85 (inv=false)
# ^^^ Q/K/V have RMS >> FFN: scales likely inverted

# Compare with inverted dequantization
export BITNET_I2S_INV_OVERRIDE=1
RUST_LOG=info cargo run -p bitnet-cli -- run --model model.gguf 2>&1 | grep "PROJ load"
# If RMS becomes reasonable (~0.8-1.0), use I2S_DEQUANT_OVERRIDE correction
```

### Issue: Validation Passes but Inference Produces Gibberish

**Symptom:**
- Policy validation succeeds
- Inference outputs are nonsensical or tied logits

**Possible Causes:**

1. **Policy thresholds are too permissive**
   - **Solution:** Tighten min/max bounds based on expected values
   - Compare with known-good model of same architecture

2. **Corruption in non-validated tensors** (embeddings, projections, etc.)
   - **Solution:** Enable projection weight validation
   - Inspect full tensor statistics with `--verbose`

3. **Incorrect model metadata** (vocab size, context length, etc.)
   - **Solution:** Use `cargo run -p bitnet-cli -- inspect --metadata` to verify
   - Check tokenizer compatibility

**Diagnostic Commands:**

```bash
# Enable strict validation mode
export BITNET_STRICT_MODE=1
cargo run -p bitnet-cli -- run --model model.gguf

# Inspect all tensor statistics
cargo run -p bitnet-cli -- inspect --full-stats model.gguf

# Compare with reference model
cargo run -p bitnet-cli -- compare-stats reference-model.gguf your-model.gguf
```

## Best Practices

### 1. Always Measure Before Defining Thresholds

❌ **Bad:** Copy thresholds from another policy without inspection
```yaml
# Don't do this!
ln:
  - pattern: ".*norm\\.weight$"
    min: 0.80    # Copied from LLaMA policy
    max: 1.20
```

✅ **Good:** Inspect your model and define thresholds based on measurements
```yaml
# Measured attn_norm RMS: min=0.92, max=1.05
# Adding 10% margin: [0.85, 1.15]
ln:
  - pattern: "attn_norm\\.weight$"
    min: 0.85
    max: 1.15
    description: "Based on empirical RMS distribution [0.92, 1.05] with 10% margin"
```

### 2. Add Safety Margins to Envelopes

Add 5-10% margin beyond observed min/max to accommodate:
- Fine-tuning variance
- Edge cases not in your test set
- Numerical precision differences across platforms

```yaml
# Observed: [0.90, 1.05]
# Policy:   [0.85, 1.10]  ← 5% margin on each side
```

### 3. Be Stricter for Critical Layers

Final output norm is critical for stability - use narrow envelope:

```yaml
ln:
  # Internal norms: wider envelope
  - pattern: "blk\\.[0-9]+\\.attn_norm\\.weight$"
    min: 0.80
    max: 1.20

  # Final norm: narrow envelope (critical for output quality)
  - pattern: "output_norm\\.weight$"
    min: 0.95
    max: 1.05
```

### 4. Document Architectural Quirks

If your model legitimately has unusual RMS values, document why:

```yaml
ln:
  - pattern: "ffn_norm\\.weight$"
    min: 0.40    # Unusually low!
    max: 1.50
    description: |
      FFN LayerNorm uses small gamma initialization (architectural choice).
      This is NOT corruption - it's by design in this architecture.
      Observed RMS: 0.42-0.52 across all layers.
```

### 5. Version Control Your Policies

Commit policies to git with clear messages:

```bash
git add examples/policies/my-model-v2.yml
git commit -m "feat(validation): update my-model policy for v2.0 architecture

- Tighten attn_norm envelope to [0.85, 1.15] (was [0.70, 1.30])
- Add projection weight validation (RMS [0.015, 0.35])
- Tested on 10 model checkpoints across training run
- No false positives or negatives in validation"
```

### 6. Test Across Model Variants

If you have multiple quantization types, test each variant:

```bash
# Test F16 variant
cargo run -p bitnet-cli -- inspect --ln-stats \
  --policy my-policy.yml --policy-key my-model:f16 \
  my-model-f16.gguf

# Test I2_S variant
cargo run -p bitnet-cli -- inspect --ln-stats \
  --policy my-policy.yml --policy-key my-model:i2_s \
  my-model-i2s.gguf

# Test Q4_0 variant
cargo run -p bitnet-cli -- inspect --ln-stats \
  --policy my-policy.yml --policy-key my-model:q4_0 \
  my-model-q4_0.gguf
```

### 7. Integrate into CI/CD Pipelines

Add policy validation to your CI workflow:

```yaml
# .github/workflows/validate-models.yml
- name: Validate model export
  run: |
    cargo run -p bitnet-cli -- inspect --ln-stats \
      --policy examples/policies/my-model-policy.yml \
      --policy-key my-model:f16 \
      artifacts/my-model-f16.gguf

    # Fail CI if validation fails
    if [ $? -ne 0 ]; then
      echo "ERROR: Model validation failed - LayerNorm corruption detected"
      exit 1
    fi
```

### 8. Never Use Corrections Long-Term

Corrections are **temporary workarounds** - always plan migration to clean GGUF:

❌ **Bad:** Deploy with corrections in production
```bash
# Don't do this in production!
export BITNET_CORRECTION_POLICY=fix-bad-model.yml
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
./inference-server --model bad-model.gguf
```

✅ **Good:** Use corrections in development, regenerate for production
```bash
# Development: Use corrections while waiting for proper fix
export BITNET_CORRECTION_POLICY=fix-bad-model.yml
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
cargo run -p bitnet-cli -- run --model bad-model.gguf

# Production: Deploy regenerated clean GGUF
unset BITNET_CORRECTION_POLICY BITNET_ALLOW_RUNTIME_CORRECTIONS
cargo run -p bitnet-cli -- run --model clean-model.gguf
```

## Environment Variables

### Validation Configuration

| Variable | Values | Description |
|----------|--------|-------------|
| `BITNET_VALIDATION_GATE` | `none`, `auto`, `policy` | Validation gate mode |
| `BITNET_VALIDATION_POLICY` | Path | Policy file path (for `gate=policy`) |
| `BITNET_VALIDATION_POLICY_KEY` | String | Policy key (for `gate=policy`) |
| `BITNET_STRICT_MODE` | `0`, `1` | Enable strict validation (fail on warnings) |

### Correction Configuration

| Variable | Values | Description |
|----------|--------|-------------|
| `BITNET_CORRECTION_POLICY` | Path | Correction policy file path |
| `BITNET_ALLOW_RUNTIME_CORRECTIONS` | `0`, `1` | Enable runtime corrections (**dev only**) |

**Warning:** CI blocks `BITNET_ALLOW_RUNTIME_CORRECTIONS=1` to prevent production deployment.

## Examples

### Example 1: Validate BitNet F16 Model

```bash
cargo run -p bitnet-cli -- inspect --ln-stats \
  --policy examples/policies/bitnet-b158-f16-clean.yml \
  --policy-key bitnet-b1.58:f16 \
  models/bitnet-model-f16.gguf
```

### Example 2: Auto-Detect Policy for LLaMA Model

```bash
# Auto-detection will select llama-generic.yml
cargo run -p bitnet-cli -- inspect --ln-stats \
  --gate auto \
  models/llama-3-8b.gguf
```

### Example 3: Use Custom Policy in CI

```bash
# In CI/CD pipeline
export BITNET_VALIDATION_GATE=policy
export BITNET_VALIDATION_POLICY=company/policies/our-model-policy.yml
export BITNET_VALIDATION_POLICY_KEY=our-model:f16
export BITNET_STRICT_MODE=1  # Fail on warnings

cargo run -p bitnet-cli -- inspect --ln-stats artifacts/our-model.gguf

if [ $? -ne 0 ]; then
  echo "Model validation failed - check LayerNorm statistics"
  exit 1
fi
```

### Example 4: Apply Corrections for Known-Bad Model

```bash
# Development workflow only - NOT for production
export BITNET_CORRECTION_POLICY=examples/policies/custom-model-example.yml
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42

cargo run -p bitnet-cli -- run \
  --model models/bad-model.gguf \
  --prompt "Test prompt"

# Check logs for correction application:
# INFO Applied 1 correction(s) during model load
# INFO Correction: LN_GAMMA_RESCALE_RMS (target_rms=1.0)
```

## Related Documentation

- **[`docs/explanation/correction-policy.md`](../../docs/explanation/correction-policy.md)**: Detailed correction policy system documentation
- **[`docs/howto/export-clean-gguf.md`](../../docs/howto/export-clean-gguf.md)**: Clean GGUF export workflow and validation
- **[`CLAUDE.md`](../../CLAUDE.md)**: Quick reference for LayerNorm troubleshooting
- **[`crates/bitnet-cli/src/ln_rules.rs`](../../crates/bitnet-cli/src/ln_rules.rs)**: Rust implementation of validation rules
- **[`correction-policy-sample.yml`](../../correction-policy-sample.yml)**: Sample correction policy with examples

## Contributing

When contributing new policies:

1. **Test thoroughly** across multiple model checkpoints
2. **Document assumptions** in policy file comments
3. **Include examples** of usage in policy file
4. **Validate in CI** as part of PR checks
5. **Update this README** with new policy in table above

## Support

If you encounter validation issues:

1. **Inspect your model**: `cargo run -p bitnet-cli -- inspect --ln-stats model.gguf`
2. **Check existing policies**: Review policy files in this directory for similar architectures
3. **Create custom policy**: Use `custom-model-example.yml` as template
4. **File an issue**: If validation seems incorrect, open GitHub issue with inspection output

## License

These policy files are provided under the same license as BitNet.rs (MIT OR Apache-2.0).
