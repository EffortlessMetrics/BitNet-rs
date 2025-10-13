# How to Validate BitNet Models

**Audience:** Developers and researchers working with GGUF models who need to ensure model quality and catch quantization errors before deployment.

**Goal:** Learn the complete 3-stage validation workflow to verify GGUF models are production-ready with correct LayerNorm weights and healthy projection weights.

---

## Overview

BitNet.rs provides a comprehensive validation system to catch common model export issues:

- **Quantized LayerNorm weights**: LayerNorm gamma weights quantized to I2_S/Q4 (should be F16/F32)
- **Incorrect projection scales**: Inverted I2_S dequantization or corrupted weight scales
- **Tokenizer mismatches**: Wrong tokenizer causing gibberish outputs
- **Export corruption**: Metadata errors or tensor misalignment

The validation system uses a **3-stage pipeline**:

1. **LayerNorm & Projection RMS Check**: Architecture-aware statistical validation
2. **Model Loading Check**: Verify weights load correctly with healthy RMS values
3. **Linguistic Sanity Check**: Greedy inference produces coherent output

---

## Quick Start

### Validate an Existing GGUF

```bash
# Validate with automatic architecture detection
./scripts/validate_gguf.sh \
  models/bitnet-2b.gguf \
  models/tokenizer.json

# Output:
# ===================================================
# 1/3: LayerNorm and Projection Weight Statistics Check
# ===================================================
# ✅ LN RMS gate passed (bitnet-b1.58:f16)
# ✅ Projection RMS gate passed
#
# ===================================================
# 2/3: Projection Weight RMS Check (via model loading)
# ===================================================
# ✅ Projection weights loaded
#
# ===================================================
# 3/3: Greedy Inference Probe
# ===================================================
# ✅ Output contains recognizable words
#
# ✅✅✅ ALL VALIDATION CHECKS PASSED ✅✅✅
```

### Convert SafeTensors to Clean GGUF

```bash
# Export F16 GGUF with LayerNorm preservation
./scripts/export_clean_gguf.sh \
  models/safetensors-checkpoint \
  models/tokenizer.json \
  models/clean

# Validate the exported model
./scripts/validate_gguf.sh \
  models/clean/clean-f16.gguf \
  models/tokenizer.json
```

---

## The 3-Stage Validation Pipeline

### Stage 1: LayerNorm & Projection RMS Check

**Purpose:** Detect quantized LayerNorm weights and projection weight anomalies using architecture-aware statistical validation.

**What it checks:**
- LayerNorm gamma RMS values are in expected envelope (architecture-specific)
- Projection weight RMS values are reasonable for the model format
- Uses pattern-based thresholds tailored to BitNet b1.58 F16/I2_S or LLaMA-style models

**How to run:**

```bash
# Auto-detect architecture from GGUF metadata
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto \
  models/model.gguf

# With strict mode (fail on warnings)
BITNET_STRICT_MODE=1 \
  cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto \
  models/model.gguf
```

**Expected output (healthy model):**

```
model_sha256: a1b2c3d4e5f6...
ruleset: bitnet-b1.58:f16

blk.0.attn_norm.weight                                     [LN]     rms=0.9523   ✅
blk.0.ffn_norm.weight                                      [LN]     rms=0.0847   ✅
blk.1.attn_norm.weight                                     [LN]     rms=0.9412   ✅
blk.1.ffn_norm.weight                                      [LN]     rms=0.0851   ✅
...
output_norm.weight                                         [LN]     rms=0.9998   ✅

blk.0.attn_q.weight                                        [PROJ]   rms=0.0214   ✅
blk.0.attn_k.weight                                        [PROJ]   rms=0.0218   ✅
blk.0.attn_v.weight                                        [PROJ]   rms=0.0216   ✅
...

✅ LN RMS gate passed (bitnet-b1.58:f16)
✅ Projection RMS gate passed (bitnet-b1.58:f16)
```

**Expected output (quantized LayerNorm - BAD):**

```
model_sha256: f9e8d7c6b5a4...
ruleset: bitnet-b1.58:f16

blk.0.attn_norm.weight                                     [LN]     rms=0.0127   ❌
blk.0.ffn_norm.weight                                      [LN]     rms=0.0093   ❌
blk.1.attn_norm.weight                                     [LN]     rms=0.0131   ❌
...

❌ LN RMS gate failed: 24/24 out of envelope (bitnet-b1.58:f16)

ERROR: Model has suspicious LayerNorm weights (quantized or corrupted).
Recommendation: Regenerate GGUF with LayerNorm weights in float format (F16/F32).
See docs/howto/export-clean-gguf.md for proper export workflow.
```

**Exit codes:**
- `0`: All checks passed
- `8` (`EXIT_LN_SUSPICIOUS`): LayerNorm or projection validation failed in strict mode

**See also:** [Validation Gates Reference](../reference/validation-gates.md) for detailed threshold definitions.

---

### Stage 2: Model Loading Check

**Purpose:** Verify weights load correctly and have healthy RMS values during actual model initialization.

**What it checks:**
- All projection weights (Q/K/V/O, FFN gate/up/down) load successfully
- RMS values are in expected range (typically O(10³) for quantized weights)
- No NaN/Inf values in loaded tensors
- I2_S dequantization produces reasonable scales

**How to run:**

```bash
# Enable RUST_LOG=info to see projection RMS values
RUST_LOG=info \
  cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  run --model models/model.gguf --tokenizer models/tokenizer.json \
  --prompt "Warmup." --max-new-tokens 1 --temperature 0.0
```

**Expected output:**

```
INFO PROJ load: blk.0.attn_q.weight RMS=0.0214 (inv=false)
INFO PROJ load: blk.0.attn_k.weight RMS=0.0218 (inv=false)
INFO PROJ load: blk.0.attn_v.weight RMS=0.0216 (inv=false)
INFO PROJ load: blk.0.attn_o.weight RMS=0.0219 (inv=false)
INFO PROJ load: blk.0.ffn_gate.weight RMS=0.0201 (inv=false)
INFO PROJ load: blk.0.ffn_up.weight RMS=0.0198 (inv=false)
INFO PROJ load: blk.0.ffn_down.weight RMS=0.0203 (inv=false)
...
```

**Warning signs:**

```
# Extremely high RMS (inverted scales?)
INFO PROJ load: blk.0.attn_q.weight RMS=150.3 (inv=false)  ⚠️

# Wildly different RMS values (corruption?)
INFO PROJ load: blk.0.attn_q.weight RMS=0.02 (inv=false)
INFO PROJ load: blk.0.attn_k.weight RMS=100.5 (inv=false)  ⚠️
```

**Exit codes:**
- `0`: Model loaded successfully
- `1`: Model loading failed (missing tensors, format errors, etc.)

---

### Stage 3: Linguistic Sanity Check

**Purpose:** Ensure the model produces coherent output, not gibberish or tied logits.

**What it checks:**
- Greedy deterministic inference produces recognizable words
- Output contains at least one word with 3+ ASCII letters
- No immediate tokenizer decode errors
- Model doesn't repeat same token indefinitely

**How to run:**

```bash
# Deterministic greedy inference
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
RAYON_NUM_THREADS=1 \
  cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  run --model models/model.gguf --tokenizer models/tokenizer.json \
  --prompt "The capital of France is" \
  --max-new-tokens 8 \
  --temperature 0.0
```

**Expected output (healthy):**

```
The capital of France is Paris.
```

**Warning signs (issues):**

```
# Gibberish (quantized LayerNorm or wrong tokenizer)
The capital of France is █▓▒░█▓▒

# Repetition (tied logits or attention collapse)
The capital of France is the the the the the the

# Empty or decode errors (tokenizer mismatch)
The capital of France is
```

**Exit codes:**
- `0`: Linguistic sanity check passed
- `1`: Inference failed to run
- Non-zero: Check logs for specific failure mode

---

## Validation Modes

### Auto-Detection Mode (Recommended)

Automatically selects validation rules based on GGUF metadata:

```bash
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto \
  models/model.gguf
```

**Auto-detection logic:**

| Architecture | File Type | Selected Ruleset | LayerNorm Envelope | Projection Envelope |
|--------------|-----------|------------------|-------------------|---------------------|
| `bitnet` or `b1.58` | `1` (F16) | `bitnet-b1.58:f16` | Pattern-based (0.05-2.0 typical) | [0.01, 0.40] |
| `bitnet` or `b1.58` | Other (quantized) | `bitnet-b1.58:i2_s` | Pattern-based (0.01-2.0 typical) | [0.002, 0.20] |
| Other | Any | `generic` | [0.80, 1.20] | None |

**When to use:**
- ✅ Standard BitNet b1.58 models (F16 or I2_S)
- ✅ LLaMA/Mistral/standard RMSNorm architectures
- ✅ CI/CD pipelines requiring deterministic validation
- ✅ When you trust your GGUF metadata is correct

**Environment variables:**

```bash
# Set auto mode via environment
export BITNET_VALIDATION_GATE=auto
cargo run -p bitnet-cli -- inspect --ln-stats model.gguf
```

---

### Policy Mode (Custom Architectures)

Use custom validation policies for non-standard architectures:

```bash
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats \
  --gate policy \
  --policy examples/policies/custom-model.yml \
  --policy-key my-model:f16 \
  models/model.gguf
```

**When to use:**
- ✅ Custom or experimental architectures
- ✅ Models with unusual LayerNorm patterns
- ✅ Overriding auto-detection for specific requirements
- ✅ Testing new policy definitions

**Example policy file:**

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
        description: "FFN LayerNorm (architectural low gamma)"

    proj_weight_rms_min: 0.015
    proj_weight_rms_max: 0.35
```

**Environment variables:**

```bash
export BITNET_VALIDATION_GATE=policy
export BITNET_VALIDATION_POLICY=examples/policies/custom-model.yml
export BITNET_VALIDATION_POLICY_KEY=my-model:f16
cargo run -p bitnet-cli -- inspect --ln-stats model.gguf
```

**See also:** [Policy Examples README](../../examples/policies/README.md) for creating custom policies.

---

### None Mode (Skip Validation)

Disable validation entirely:

```bash
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate none \
  models/model.gguf
```

**When to use:**
- ⚠️ Debugging validation system itself
- ⚠️ Experimental models where validation rules don't exist yet
- ⚠️ Testing inference without validation overhead

**Warning:** This disables important safety checks. Only use for development.

---

## Complete Workflows

### Workflow 1: Validate Existing GGUF

**Scenario:** You have a GGUF model from Hugging Face or a third-party export tool and need to verify it's production-ready.

**Steps:**

```bash
# 1. Inspect LayerNorm and projection statistics
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto \
  models/model.gguf

# 2. Run full 3-stage validation
./scripts/validate_gguf.sh \
  models/model.gguf \
  models/tokenizer.json

# 3. If validation passes, model is ready for use
cargo run -p bitnet-cli --no-default-features --features cpu -- \
  run --model models/model.gguf --tokenizer models/tokenizer.json \
  --prompt "Your prompt here"
```

**If validation fails:**

See [Troubleshooting](#troubleshooting-validation-failures) section below.

---

### Workflow 2: Convert SafeTensors to Clean GGUF

**Scenario:** You have a SafeTensors checkpoint (from training or fine-tuning) and need to create a validated GGUF.

**Steps:**

```bash
# 1. Export to F16 GGUF with LayerNorm preservation
./scripts/export_clean_gguf.sh \
  models/safetensors-checkpoint \
  models/tokenizer.json \
  models/clean

# Output:
# INFO: Using Rust st2gguf converter
# INFO: Converting SafeTensors to GGUF (F16 output, LayerNorm preserved)...
# ✅ Export complete!
#   Output: models/clean/clean-f16.gguf
#   Fingerprint: sha256-abc123...

# 2. Validate the exported GGUF
./scripts/validate_gguf.sh \
  models/clean/clean-f16.gguf \
  models/tokenizer.json

# 3. If validation passes, you're done!
# If validation fails, check export logs and retry
```

**Advanced: Use Rust st2gguf directly**

```bash
# Build st2gguf converter
cargo build --release -p bitnet-st2gguf

# Convert with strict validation
target/release/st2gguf \
  --input models/checkpoint.safetensors \
  --output models/clean-f16.gguf \
  --config models/config.json \
  --strict

# Validate
./scripts/validate_gguf.sh \
  models/clean-f16.gguf \
  models/tokenizer.json
```

**See also:** [Export Clean GGUF Guide](./export-clean-gguf.md) for detailed export instructions.

---

### Workflow 3: Validate Custom Architecture

**Scenario:** You have a custom or experimental architecture that doesn't match BitNet b1.58 or standard LLaMA patterns.

**Steps:**

```bash
# 1. Inspect LayerNorm statistics to understand patterns
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate none \
  models/custom-model.gguf > ln-stats.txt

# Review the output to identify RMS patterns
cat ln-stats.txt

# 2. Create custom policy based on observed patterns
cp examples/policies/custom-model-example.yml my-model-policy.yml
nano my-model-policy.yml

# Define rules based on your inspection:
# - LayerNorm patterns and RMS envelopes
# - Projection weight RMS ranges
# - Architecture-specific quirks

# 3. Validate with custom policy
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats \
  --gate policy \
  --policy my-model-policy.yml \
  --policy-key my-model:f16 \
  models/custom-model.gguf

# 4. Run linguistic sanity check
BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo run -p bitnet-cli --no-default-features --features cpu -- \
  run --model models/custom-model.gguf --tokenizer models/tokenizer.json \
  --prompt "Test prompt" --max-new-tokens 32 --temperature 0.0

# 5. If output is coherent, commit your policy
git add my-model-policy.yml
git commit -m "feat(validation): add policy for my-model architecture"
```

**See also:** [Policy Examples README](../../examples/policies/README.md) for policy creation guide.

---

### Workflow 4: Policy-Based Runtime Corrections (Development Only)

**Scenario:** You have a known-bad model (quantized LayerNorm) and need to unblock inference development while waiting for proper GGUF regeneration.

**⚠️ WARNING:** This is a **temporary workaround** for development only. CI blocks correction flags to prevent production deployment.

**Steps:**

```bash
# 1. Diagnose the issue
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto \
  models/bad-model.gguf

# Output:
# ❌ LN RMS gate failed: 24/24 out of envelope
# blk.0.attn_norm.weight RMS=0.0127 [SUSPICIOUS - expected ~1.0]

# 2. Create correction policy (see docs/explanation/correction-policy.md)
nano correction-policy.yml

# Example correction policy:
# version: 1
# models:
#   - fingerprint: "sha256-abc123..."
#     corrections:
#       - type: LN_GAMMA_RESCALE_RMS
#         target_rms: 1.0

# 3. Enable runtime corrections (DEVELOPMENT ONLY)
export BITNET_CORRECTION_POLICY=./correction-policy.yml
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42

# 4. Run inference with corrections
cargo run -p bitnet-cli --no-default-features --features cpu -- \
  run --model models/bad-model.gguf --tokenizer models/tokenizer.json \
  --prompt "Test prompt"

# 5. IMPORTANT: Regenerate clean GGUF for production use
./scripts/export_clean_gguf.sh \
  models/original-checkpoint \
  models/tokenizer.json \
  models/clean

# 6. Validate clean GGUF and retire correction policy
unset BITNET_CORRECTION_POLICY BITNET_ALLOW_RUNTIME_CORRECTIONS
./scripts/validate_gguf.sh models/clean/clean-f16.gguf models/tokenizer.json
```

**See also:** [Correction Policy Documentation](../explanation/correction-policy.md) for detailed correction workflow.

---

## Troubleshooting Validation Failures

### Issue: LayerNorm RMS Validation Failed

**Symptom:**

```
❌ LN RMS gate failed: 24/24 out of envelope (bitnet-b1.58:f16)
blk.0.attn_norm.weight RMS=0.0127 [SUSPICIOUS - expected ~1.0]
```

**Root Cause:** LayerNorm gamma weights were quantized during export (should be F16/F32).

**Solutions:**

1. **Best solution:** Regenerate GGUF with LayerNorm weights in float format

   ```bash
   # Using Rust st2gguf (automatic LayerNorm preservation)
   cargo run --release -p bitnet-st2gguf -- \
     --input models/checkpoint.safetensors \
     --output models/clean-f16.gguf \
     --strict

   # Validate
   ./scripts/validate_gguf.sh models/clean-f16.gguf models/tokenizer.json
   ```

2. **Temporary workaround:** Use correction policy (development only)

   See [Workflow 4](#workflow-4-policy-based-runtime-corrections-development-only) above.

3. **Alternative:** Check if you're using the wrong policy

   ```bash
   # BitNet I2_S models legitimately have low attn_norm RMS (~0.01-0.02)
   # Use correct policy for quantized models
   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
     inspect --ln-stats \
     --policy examples/policies/bitnet-b158-i2s-quantized.yml \
     --policy-key bitnet-b1.58:i2_s \
     models/model-i2s.gguf
   ```

---

### Issue: Projection Weight RMS Out of Range

**Symptom:**

```
⚠️ WARNING: suspicious projection weights detected (6/144 tensors)
blk.0.attn_q.weight RMS=150.3 [OUT OF RANGE: expected 0.01-0.40]
```

**Root Cause:** I2_S dequantization scales are inverted or weights are corrupted.

**Solutions:**

1. **Inspect RMS distribution:**

   ```bash
   RUST_LOG=info \
     cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
     run --model model.gguf --tokenizer tokenizer.json \
     --prompt "Test" --max-new-tokens 1 2>&1 | grep "PROJ load"
   ```

   Look for patterns:
   - Q/K/V have very high RMS (~100-150) but FFN is normal (~0.8-1.0) → Inverted scales
   - All projections have similar anomalous RMS → Export corruption
   - Single layer has issues → Layer-specific corruption

2. **Re-export from source checkpoint:**

   ```bash
   ./scripts/export_clean_gguf.sh \
     models/source-checkpoint \
     models/tokenizer.json \
     models/clean
   ```

3. **Check GGUF metadata:**

   ```bash
   # Verify file_type matches actual quantization
   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
     inspect --metadata model.gguf
   ```

---

### Issue: Gibberish Output in Linguistic Sanity Check

**Symptom:**

```
The capital of France is █▓▒░█▓▒░▓▒
⚠️ Output does not contain recognizable words
```

**Root Causes and Solutions:**

1. **Tokenizer mismatch:**

   ```bash
   # Try different tokenizer
   cargo run -p bitnet-cli --no-default-features --features cpu -- \
     run --model model.gguf --tokenizer different-tokenizer.json \
     --prompt "The capital of France is"
   ```

2. **Quantized LayerNorm:**

   See [LayerNorm RMS Validation Failed](#issue-layernorm-rms-validation-failed) above.

3. **RoPE parameter mismatch:**

   Check `config.json` for RoPE settings:
   - `rope_theta` (base frequency)
   - `rope_scaling` (scaling factors)
   - Verify they match model training configuration

4. **Model corruption:**

   ```bash
   # Check SHA256 fingerprint
   sha256sum model.gguf

   # Re-download or re-export if hash doesn't match
   ```

---

### Issue: Policy Key Not Found

**Symptom:**

```
Error: policy key not found: my-model:f16
```

**Solutions:**

1. **List available policy keys:**

   ```bash
   # View policy file structure
   cat examples/policies/custom-model-example.yml

   # Look for keys under "rules:" section
   # Example:
   # rules:
   #   bitnet-b1.58:f16:  ← This is the policy key
   #     name: "BitNet b1.58 F16"
   ```

2. **Use correct key format:**

   ```bash
   # Key format: architecture:variant
   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
     inspect --ln-stats \
     --policy examples/policies/bitnet-b158-f16-clean.yml \
     --policy-key bitnet-b1.58:f16 \
     model.gguf
   ```

3. **Create missing policy:**

   See [Workflow 3: Validate Custom Architecture](#workflow-3-validate-custom-architecture) above.

---

## Command Reference

### Inspect Command

**Purpose:** Examine LayerNorm and projection weight statistics with architecture-aware validation.

**Syntax:**

```bash
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats \
  [--gate none|auto|policy] \
  [--policy PATH] \
  [--policy-key KEY] \
  [--json] \
  MODEL
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `MODEL` | Yes | Path to GGUF model file |
| `--ln-stats` | Yes | Enable LayerNorm statistics analysis |
| `--gate` | No | Validation mode: `none`, `auto`, `policy` (default: `auto`) |
| `--policy` | No | Path to YAML policy file (required for `gate=policy`) |
| `--policy-key` | No | Policy key for rules lookup (default: uses architecture from GGUF) |
| `--json` | No | Output results as JSON |

**Examples:**

```bash
# Auto-detect architecture
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto model.gguf

# Use custom policy
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats \
  --gate policy \
  --policy examples/policies/custom.yml \
  --policy-key my-model:f16 \
  model.gguf

# JSON output for CI
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto --json model.gguf > validation.json
```

**Exit Codes:**

| Code | Name | Description |
|------|------|-------------|
| `0` | `EXIT_SUCCESS` | All validation checks passed |
| `8` | `EXIT_LN_SUSPICIOUS` | LayerNorm or projection validation failed (strict mode only) |

**See also:** [Validation Gates Reference](../reference/validation-gates.md) for technical details.

---

### Validation Script

**Purpose:** Run complete 3-stage validation pipeline (LayerNorm, projection, linguistic sanity).

**Syntax:**

```bash
./scripts/validate_gguf.sh MODEL TOKENIZER
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `MODEL` | Yes | Path to GGUF model file |
| `TOKENIZER` | Yes | Path to tokenizer.json file |

**Examples:**

```bash
# Basic validation
./scripts/validate_gguf.sh \
  models/bitnet-2b.gguf \
  models/tokenizer.json

# Validation in CI (exit code check)
./scripts/validate_gguf.sh model.gguf tokenizer.json
if [ $? -ne 0 ]; then
  echo "Validation failed - model is not production-ready"
  exit 1
fi
```

**Exit Codes:**

| Code | Description |
|------|-------------|
| `0` | All validation checks passed |
| `10` | LayerNorm validation failed |
| `13` | Model loading failed |
| `14` | Inference probe failed |
| `15` | Linguistic sanity check failed |

---

### Export Script

**Purpose:** Convert SafeTensors to clean F16 GGUF with LayerNorm preservation.

**Syntax:**

```bash
./scripts/export_clean_gguf.sh MODEL_DIR TOKENIZER OUT_DIR
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `MODEL_DIR` | Yes | Directory containing SafeTensors or HF checkpoint |
| `TOKENIZER` | Yes | Path to tokenizer.json file |
| `OUT_DIR` | Yes | Output directory for GGUF |

**Environment Variables:**

| Variable | Values | Description |
|----------|--------|-------------|
| `CONVERTER` | Path or `rust`/`st2gguf` | Override converter selection |
| `STRICT` | `1` | Enable strict validation in st2gguf |

**Examples:**

```bash
# Export with automatic converter selection
./scripts/export_clean_gguf.sh \
  models/safetensors-checkpoint \
  models/tokenizer.json \
  models/clean

# Force Rust st2gguf converter with strict validation
CONVERTER=rust STRICT=1 \
  ./scripts/export_clean_gguf.sh \
  models/checkpoint \
  models/tokenizer.json \
  models/clean
```

**Output Files:**

| File | Description |
|------|-------------|
| `clean-f16.gguf` | Main GGUF model (F16 precision) |
| `clean-f16.fingerprint` | SHA256 fingerprint (sha256-...) |
| `clean-f16.meta.json` | Export metadata (source, date, converter, etc.) |

**See also:** [Export Clean GGUF Guide](./export-clean-gguf.md) for detailed export documentation.

---

## Environment Variables

### Validation Configuration

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `BITNET_VALIDATION_GATE` | `none`, `auto`, `policy` | `auto` | Validation gate mode |
| `BITNET_VALIDATION_POLICY` | Path | None | Policy file path (for `gate=policy`) |
| `BITNET_VALIDATION_POLICY_KEY` | String | Architecture | Policy key (for `gate=policy`) |
| `BITNET_STRICT_MODE` | `0`, `1` | `0` | Enable strict validation (fail on warnings) |

### Inference Configuration

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `BITNET_DETERMINISTIC` | `0`, `1` | `0` | Enable deterministic inference |
| `BITNET_SEED` | Integer | Random | Random seed (requires `BITNET_DETERMINISTIC=1`) |
| `RAYON_NUM_THREADS` | Integer | Auto | Thread count for parallel operations |
| `RUST_LOG` | `error`, `warn`, `info`, `debug`, `trace` | `error` | Logging level |

### Correction Configuration (Development Only)

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `BITNET_CORRECTION_POLICY` | Path | None | Correction policy file path |
| `BITNET_ALLOW_RUNTIME_CORRECTIONS` | `0`, `1` | `0` | Enable runtime corrections (**dev only**) |

**⚠️ Warning:** CI blocks `BITNET_ALLOW_RUNTIME_CORRECTIONS=1` to prevent production deployment with workarounds.

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
# .github/workflows/validate-model.yml
name: Validate GGUF Model

on:
  push:
    paths:
      - 'models/**/*.gguf'
  workflow_dispatch:

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Build BitNet.rs CLI
        run: |
          cargo build --release -p bitnet-cli \
            --no-default-features --features cpu,full-cli

      - name: Run Validation
        run: |
          ./scripts/validate_gguf.sh \
            models/model.gguf \
            models/tokenizer.json

      - name: Check for corrections (security gate)
        run: |
          if [ -n "$BITNET_CORRECTION_POLICY" ] || \
             [ -n "$BITNET_ALLOW_RUNTIME_CORRECTIONS" ]; then
            echo "ERROR: Runtime correction flags detected"
            echo "Corrections are dev-only - regenerate clean GGUF"
            exit 1
          fi
```

### CI Best Practices

1. **Always use strict mode:**

   ```bash
   BITNET_STRICT_MODE=1 ./scripts/validate_gguf.sh model.gguf tokenizer.json
   ```

2. **Block correction flags:**

   ```bash
   # Fail CI if correction flags are set
   if [ -n "$BITNET_ALLOW_RUNTIME_CORRECTIONS" ]; then
     echo "ERROR: Corrections not allowed in CI"
     exit 1
   fi
   ```

3. **Validate on every model change:**

   ```yaml
   on:
     push:
       paths:
         - 'models/**/*.gguf'
         - 'models/**/tokenizer.json'
   ```

4. **Archive validation reports:**

   ```yaml
   - name: Upload validation report
     uses: actions/upload-artifact@v3
     with:
       name: validation-report
       path: validation.json
   ```

---

## FAQ

### Q: What's the difference between validation and correction policies?

**A:**

- **Validation policies** define acceptable RMS ranges for weights (example: `examples/policies/bitnet-b158-f16-clean.yml`)
- **Correction policies** provide runtime fixes for known-bad models (example: see `docs/explanation/correction-policy.md`)

Validation policies are used during inspection to check if a model is healthy. Correction policies are temporary workarounds to fix known defects at runtime (dev only).

### Q: When should I use `BITNET_STRICT_MODE=1`?

**A:**

Use strict mode in:
- ✅ CI/CD pipelines
- ✅ Production validation
- ✅ Release qualification
- ✅ When you need zero-tolerance for warnings

Skip strict mode when:
- ⚠️ Debugging validation rules
- ⚠️ Working with experimental models
- ⚠️ You understand the warnings and accept the risk

### Q: Can I skip LayerNorm validation for quantized models?

**A:**

No. LayerNorm weights should **never** be quantized, even in I2_S models. The validator uses architecture-specific rules that account for legitimate RMS variations in quantized models (e.g., BitNet I2_S attn_norm legitimately has RMS ~0.01-0.02).

If validation fails, either:
1. Regenerate GGUF with LayerNorm weights in float format, or
2. Create a custom policy if your architecture legitimately has unusual RMS values

### Q: What if auto-detection selects the wrong ruleset?

**A:**

Use explicit policy mode:

```bash
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats \
  --gate policy \
  --policy examples/policies/correct-policy.yml \
  --policy-key correct-key \
  model.gguf
```

If auto-detection is consistently wrong, file an issue with:
- GGUF metadata (`cargo run -p bitnet-cli -- inspect --metadata model.gguf`)
- Expected architecture
- Inspection output

### Q: How do I know if my custom policy is correct?

**A:**

1. **Test on multiple checkpoints:** Validate 3-5 models from different training stages
2. **Check false positives:** Policy should not reject healthy models
3. **Check false negatives:** Policy should catch known-bad models
4. **Compare with reference:** Use known-good model as baseline
5. **Test inference:** Models passing validation should produce coherent output

See [Workflow 3](#workflow-3-validate-custom-architecture) for detailed custom policy creation.

---

## Related Documentation

- **[Export Clean GGUF Guide](./export-clean-gguf.md)**: How to create clean GGUF models with proper LayerNorm format
- **[Validation Gates Reference](../reference/validation-gates.md)**: Technical details on validation system architecture
- **[Correction Policy Documentation](../explanation/correction-policy.md)**: Runtime correction system for known-bad models
- **[Policy Examples README](../../examples/policies/README.md)**: Example policies and creation guide
- **[Build Commands Reference](../development/build-commands.md)**: CLI build instructions with `full-cli` feature
- **[CLAUDE.md Quick Reference](../../CLAUDE.md)**: Quick command reference and troubleshooting

---

## Summary

| Task | Command | Purpose |
|------|---------|---------|
| **Validate existing GGUF** | `./scripts/validate_gguf.sh model.gguf tokenizer.json` | 3-stage validation pipeline |
| **Inspect LayerNorm stats** | `cargo run -p bitnet-cli -- inspect --ln-stats --gate auto model.gguf` | Architecture-aware RMS validation |
| **Export clean GGUF** | `./scripts/export_clean_gguf.sh checkpoint/ tokenizer.json output/` | Convert SafeTensors to F16 GGUF |
| **Custom policy validation** | `cargo run -p bitnet-cli -- inspect --ln-stats --gate policy --policy policy.yml model.gguf` | Validate with custom rules |
| **Strict mode validation** | `BITNET_STRICT_MODE=1 cargo run -p bitnet-cli -- inspect --ln-stats model.gguf` | Fail on warnings |

**Key Principle:** Always validate models before deployment. Clean models must pass all 3 stages without corrections or workarounds.

---

For questions or issues, see:
- **GitHub Issues**: [BitNet-rs/issues](https://github.com/microsoft/BitNet/issues)
- **Documentation Index**: `docs/` directory
- **Quick Reference**: [CLAUDE.md](../../CLAUDE.md)
