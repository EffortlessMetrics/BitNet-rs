# Correction Policy System

This document describes the policy-driven correction system for handling known-bad GGUF models in BitNet.rs. The system provides fingerprinted, auditable runtime corrections as a **temporary workaround** while you regenerate properly formatted models.

## Overview

### Purpose

The correction policy system enables BitNet.rs to apply targeted fixes to models with known defects (e.g., quantized LayerNorm weights) without modifying the GGUF file on disk. This is a **temporary workaround** - the proper fix is always to regenerate the GGUF with correct weight formats.

### Design Principles

1. **Fingerprinted**: Corrections are tied to specific model fingerprints (hash-based identification)
2. **Auditable**: All applied corrections are logged in inference receipts
3. **Explicit**: Requires both policy file and runtime flag to activate
4. **Temporary**: CI blocks correction flags to prevent production deployment
5. **Safe**: Policy validation occurs before any corrections are applied

## When to Use Runtime Corrections

### Appropriate Use Cases

✅ **Development and testing**: Unblock inference development while waiting for proper GGUF regeneration
✅ **Known-bad models**: Models with documented defects and fingerprinted in policy file
✅ **Short-term workaround**: Temporary solution with clear migration path to fixed GGUF
✅ **Research and experimentation**: Exploring correction effectiveness before regenerating

### Inappropriate Use Cases

❌ **Production deployment**: CI blocks correction flags - never deploy with corrections enabled
❌ **Unknown models**: Applying corrections without fingerprint validation
❌ **Long-term solution**: Corrections are not a substitute for proper model regeneration
❌ **Performance optimization**: Corrections add overhead - regenerate for production use

## Policy File Format

### YAML Structure

```yaml
# correction-policy.yml - Example policy file
version: 1
policies:
  - name: "Microsoft BitNet 2B 4T - Quantized LayerNorm Fix"
    description: "Rescale LayerNorm gamma weights from quantized I2_S to proper FP32 range"

    # Model fingerprint (SHA256 of GGUF file or metadata hash)
    fingerprint:
      type: "gguf_sha256"
      value: "a1b2c3d4e5f6..."  # Full or partial hash

    # Alternative: metadata-based fingerprint
    # fingerprint:
    #   type: "metadata"
    #   model_name: "bitnet-b1.58-2B-4T"
    #   quantization: "I2_S"
    #   vocab_size: 128256

    # Corrections to apply
    corrections:
      - type: "layernorm_scale"
        description: "Rescale LayerNorm gamma RMS to ~1.0"
        parameters:
          target_rms: 1.0
          tolerance: 0.1
          layers: "all"  # or specific layer indices [0, 1, 5]

        # Validation before applying
        preconditions:
          - rms_out_of_range: [0.5, 2.0]  # Apply if RMS outside this envelope

        # Expected outcome
        postconditions:
          - rms_in_range: [0.9, 1.1]  # Verify RMS after correction

  - name: "Custom Model - Attention Bias Correction"
    description: "Fix attention bias scaling for specific quantization scheme"
    fingerprint:
      type: "gguf_sha256"
      value: "f7e8d9c0b1a2..."
    corrections:
      - type: "attention_scale"
        parameters:
          scale_factor: 1.0883883
          apply_to: "all_layers"
```

### Fingerprint Types

1. **GGUF SHA256**: Full or partial hash of GGUF file
   ```yaml
   fingerprint:
     type: "gguf_sha256"
     value: "a1b2c3d4e5f6..."
   ```

2. **Metadata-based**: Match based on GGUF metadata fields
   ```yaml
   fingerprint:
     type: "metadata"
     model_name: "bitnet-b1.58-2B-4T"
     quantization: "I2_S"
     vocab_size: 128256
   ```

3. **Combined**: Multiple fingerprint criteria (all must match)
   ```yaml
   fingerprint:
     type: "combined"
     criteria:
       - type: "gguf_sha256"
         value: "a1b2c3d4e5f6..."
       - type: "metadata"
         model_name: "bitnet-b1.58-2B-4T"
   ```

### Correction Types

#### LayerNorm Scale Correction

Rescales LayerNorm gamma weights with incorrect RMS statistics:

```yaml
corrections:
  - type: "layernorm_scale"
    description: "Rescale quantized LayerNorm gamma to FP32 range"
    parameters:
      target_rms: 1.0        # Target RMS value
      tolerance: 0.1         # Acceptable deviation
      layers: "all"          # Apply to all layers
    preconditions:
      - rms_out_of_range: [0.5, 2.0]
    postconditions:
      - rms_in_range: [0.9, 1.1]
```

#### Attention Scale Correction

Adjusts attention scaling factors:

```yaml
corrections:
  - type: "attention_scale"
    description: "Fix attention scale factor for quantized Q/K tensors"
    parameters:
      scale_factor: 1.0883883  # 1/sqrt(head_dim) for head_dim=128
      apply_to: "all_layers"
```

## Usage Workflow

### 1. Diagnose Model Issues

Use `bitnet inspect` to examine LayerNorm statistics:

```bash
# Inspect LayerNorm gamma statistics
cargo run -p bitnet-cli -- inspect --ln-stats \
  models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf

# Example output:
# LayerNorm Statistics Analysis
# =============================
# Model: bitnet-b1.58-2B-4T (I2_S quantization)
# Total layers: 24
#
# Per-layer gamma RMS statistics:
# Layer  0: RMS = 0.127 [SUSPICIOUS - expected ~1.0]
# Layer  1: RMS = 0.131 [SUSPICIOUS - expected ~1.0]
# ...
# Layer 23: RMS = 0.129 [SUSPICIOUS - expected ~1.0]
#
# Summary:
# - 24/24 layers have suspicious gamma RMS values
# - Root cause: LayerNorm weights quantized to I2_S (should be FP16/FP32)
# - Recommendation: Regenerate GGUF with --skip-layernorm-quantization
# - Temporary workaround: Use correction policy (see docs/explanation/correction-policy.md)
```

### 2. Create Correction Policy

Create a YAML policy file based on inspection results:

```bash
# Generate policy template
cargo run -p bitnet-cli -- inspect --ln-stats \
  --generate-policy config/correction-policy.yml \
  models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf

# Edit policy.yml as needed
nano config/correction-policy.yml
```

### 3. Validate Policy

Test policy application in dry-run mode:

```bash
# Validate policy without applying corrections
export BITNET_CORRECTION_POLICY=./config/correction-policy.yml
cargo run -p bitnet-cli -- validate-policy \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --dry-run

# Example output:
# Policy Validation Report
# ========================
# Policy: correction-policy.yml
# Model: ggml-model-i2_s.gguf
#
# Fingerprint Match: ✓ (GGUF SHA256: a1b2c3d4...)
# Preconditions: ✓ (LayerNorm RMS out of range: 0.127-0.131)
# Corrections: 1 policy matched
#   - layernorm_scale: Would rescale 24 layers
# Postconditions: ✓ (Expected RMS: 0.9-1.1)
#
# Status: READY (use BITNET_ALLOW_RUNTIME_CORRECTIONS=1 to apply)
```

### 4. Apply Corrections

Enable runtime corrections for inference:

```bash
# Set environment variables
export BITNET_CORRECTION_POLICY=./config/correction-policy.yml
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42

# Run inference with corrections
cargo run -p bitnet-cli --no-default-features --features cpu -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/llama3-tokenizer/tokenizer.json \
  --prompt "Answer in one short sentence: Why is the sky blue?" \
  --max-new-tokens 32

# Corrections will be logged in inference receipt
```

### 5. Verify Receipt Enrichment

Check that corrections are documented in inference receipts:

```bash
# Inference receipt example:
{
  "model": "ggml-model-i2_s.gguf",
  "fingerprint": "a1b2c3d4e5f6...",
  "corrections_applied": [
    {
      "policy": "Microsoft BitNet 2B 4T - Quantized LayerNorm Fix",
      "type": "layernorm_scale",
      "parameters": {
        "target_rms": 1.0,
        "tolerance": 0.1,
        "layers_affected": 24
      },
      "validation": {
        "precondition_rms_range": [0.127, 0.131],
        "postcondition_rms_range": [0.98, 1.02],
        "status": "applied"
      }
    }
  ],
  "warning": "Runtime corrections applied - regenerate GGUF for production use",
  "output": "The sky appears blue because..."
}
```

## Receipt Enrichment Format

### Standard Receipt Fields

When corrections are applied, inference receipts include additional metadata:

```json
{
  "receipt_version": "1.0",
  "model": {
    "path": "models/ggml-model-i2_s.gguf",
    "fingerprint": "sha256:a1b2c3d4e5f6...",
    "size_bytes": 2147483648
  },
  "corrections": {
    "policy_file": "./config/correction-policy.yml",
    "policy_version": 1,
    "applied": [
      {
        "policy_name": "Microsoft BitNet 2B 4T - Quantized LayerNorm Fix",
        "correction_type": "layernorm_scale",
        "timestamp": "2025-10-12T19:30:00Z",
        "parameters": {
          "target_rms": 1.0,
          "tolerance": 0.1,
          "layers": "all"
        },
        "validation": {
          "precondition": {
            "check": "rms_out_of_range",
            "expected": [0.5, 2.0],
            "actual": [0.127, 0.131],
            "status": "pass"
          },
          "postcondition": {
            "check": "rms_in_range",
            "expected": [0.9, 1.1],
            "actual": [0.98, 1.02],
            "status": "pass"
          }
        },
        "impact": {
          "layers_modified": 24,
          "weights_affected": "layernorm.gamma (all layers)",
          "overhead_ms": 12.5
        }
      }
    ],
    "warnings": [
      "Runtime corrections are temporary workarounds",
      "Regenerate GGUF with proper LayerNorm weights for production use",
      "CI blocks correction flags - not suitable for deployment"
    ]
  },
  "inference": {
    "prompt": "Answer in one short sentence: Why is the sky blue?",
    "output": "The sky appears blue because...",
    "tokens_generated": 32,
    "throughput_tokens_per_sec": 15.2
  }
}
```

### CI Validation

CI workflows **reject** builds with correction flags enabled:

```bash
# CI check in .github/workflows/guards.yml
- name: Validate no runtime corrections in CI
  run: |
    if [ -n "$BITNET_CORRECTION_POLICY" ] || [ -n "$BITNET_ALLOW_RUNTIME_CORRECTIONS" ]; then
      echo "ERROR: Runtime correction flags detected in CI"
      echo "Corrections are dev-only workarounds - regenerate GGUF for production"
      exit 1
    fi
```

## Migration Path

### From Corrected Model to Proper GGUF

1. **Identify defect**: Use `bitnet inspect --ln-stats` to diagnose
2. **Apply corrections**: Use policy file for temporary inference
3. **Regenerate GGUF**: Create new model with proper weight formats
4. **Validate**: Compare inference outputs with and without corrections
5. **Retire policy**: Remove correction policy once new GGUF deployed

### Example Migration Workflow

```bash
# Step 1: Diagnose and document defect
cargo run -p bitnet-cli -- inspect --ln-stats \
  --output diagnosis.json \
  old-model.gguf

# Step 2: Apply corrections temporarily
export BITNET_CORRECTION_POLICY=./config/correction-policy.yml
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
cargo run -p bitnet-cli -- run --model old-model.gguf \
  --output corrected-results.json

# Step 3: Regenerate GGUF with proper format
# (External tool - e.g., llama.cpp with --skip-layernorm-quantization)
python convert.py --model original-weights/ \
  --output new-model.gguf \
  --skip-layernorm-quantization

# Step 4: Validate new model
unset BITNET_CORRECTION_POLICY BITNET_ALLOW_RUNTIME_CORRECTIONS
cargo run -p bitnet-cli -- run --model new-model.gguf \
  --output regenerated-results.json

# Step 5: Compare outputs
cargo run -p bitnet-cli -- compare-outputs \
  corrected-results.json regenerated-results.json

# Expected: High similarity (>99%) with no corrections needed
# If outputs match, retire correction-policy.yml and use new-model.gguf
```

## Security and Auditability

### Policy File Integrity

Correction policies should be version-controlled and reviewed:

```bash
# Store policy files in git
git add correction-policies/*.yml
git commit -m "docs: add correction policy for microsoft-bitnet-2b-4t"

# Include policy hash in receipts for auditability
cargo run -p bitnet-cli -- run \
  --model model.gguf \
  --correction-policy config/correction-policy.yml \
  --record-policy-hash
```

### Audit Trail

All corrections are logged with:
- Policy file path and version
- Model fingerprint (GGUF hash)
- Correction parameters applied
- Validation results (pre/post conditions)
- Timestamp and BitNet.rs version

## Limitations

### Not Supported

❌ **Arbitrary transformations**: Only well-defined correction types (LayerNorm scale, attention scale)
❌ **Model modification**: Corrections are runtime-only - GGUF is never modified on disk
❌ **Complex policies**: Conditional logic, dynamic parameters based on runtime state
❌ **Performance optimization**: Corrections add overhead - not suitable for production deployment

### Performance Impact

Runtime corrections introduce overhead:
- **Policy validation**: 5-10ms per model load
- **Correction application**: 10-50ms per layer (LayerNorm scale)
- **Receipt enrichment**: 1-2ms per inference

For production use, **always regenerate GGUF** with proper weight formats.

## Related Documentation

- [Environment Variables Reference](../environment-variables.md) - `BITNET_CORRECTION_POLICY` and `BITNET_ALLOW_RUNTIME_CORRECTIONS`
- [CLAUDE.md](../../CLAUDE.md) - Quick reference for LayerNorm troubleshooting
- [INFERENCE_MVP.md](../../INFERENCE_MVP.md) - CPU MVP acceptance checklist including policy workflow
- [Model Validation](../reference/model-validation.md) - GGUF compatibility and validation
- [Troubleshooting Guide](../troubleshooting/layernorm-issues.md) - LayerNorm debugging and correction workflow

## FAQ

### Q: When should I use correction policies vs regenerating GGUF?

**A**: Use correction policies only as a **temporary workaround** while waiting for GGUF regeneration. Always prefer regenerating with proper weight formats for production use.

### Q: Can I deploy with correction policies in production?

**A**: No. CI blocks correction flags specifically to prevent production deployment with workarounds. Regenerate your GGUF instead.

### Q: What happens if fingerprint doesn't match?

**A**: If the model fingerprint doesn't match any policy, corrections are **not applied** and inference proceeds with the raw model (may produce warnings or errors).

### Q: How do I know if corrections were applied?

**A**: Check the inference receipt JSON output - it includes a `corrections` section with detailed information about applied policies.

### Q: Can I chain multiple corrections?

**A**: Yes, a single policy file can define multiple correction types. They are applied in order and validated sequentially.

### Q: What's the performance overhead?

**A**: Typically 10-50ms per inference for LayerNorm corrections. For production workloads, this overhead is unacceptable - regenerate your GGUF.

## Examples

### Minimal Policy for LayerNorm Correction

```yaml
version: 1
policies:
  - name: "BitNet 2B - Fix Quantized LayerNorm"
    fingerprint:
      type: "gguf_sha256"
      value: "a1b2c3d4e5f6"
    corrections:
      - type: "layernorm_scale"
        parameters:
          target_rms: 1.0
          layers: "all"
```

### Multi-Correction Policy

```yaml
version: 1
policies:
  - name: "Custom Model - Multiple Fixes"
    fingerprint:
      type: "metadata"
      model_name: "custom-bitnet"
    corrections:
      - type: "layernorm_scale"
        parameters:
          target_rms: 1.0
          layers: [0, 1, 2, 3]
      - type: "attention_scale"
        parameters:
          scale_factor: 1.0883883
          apply_to: "all_layers"
```

### Development Workflow Script

```bash
#!/bin/bash
# dev-inference-with-corrections.sh
set -euo pipefail

MODEL="${1:-models/ggml-model-i2_s.gguf}"
POLICY="${2:-./config/correction-policy.yml}"

# Validate policy exists
if [ ! -f "$POLICY" ]; then
  echo "ERROR: Policy file not found: $POLICY"
  exit 1
fi

# Enable corrections
export BITNET_CORRECTION_POLICY="$POLICY"
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42

# Run inference with corrections
cargo run -p bitnet-cli --no-default-features --features cpu -- run \
  --model "$MODEL" \
  --tokenizer models/llama3-tokenizer/tokenizer.json \
  --prompt "Test prompt" \
  --max-new-tokens 32 \
  --output inference-receipt.json

echo "Inference complete with corrections applied"
echo "Receipt: inference-receipt.json"
echo ""
echo "REMINDER: Regenerate GGUF for production use!"
```
