# How to Export and Validate Clean GGUF Models

**Audience:** Developers and researchers who need to export BitNet models to GGUF format with correct LayerNorm preservation.

**Goal:** Produce production-ready GGUF models that pass strict validation without policy corrections.

---

## Overview

This guide shows you how to export BitNet models from SafeTensors or Hugging Face checkpoints to GGUF format while ensuring **LayerNorm weights remain in float format** (F16 or F32). Quantizing LayerNorm weights causes inference quality degradation and must be avoided.

The export process produces "clean" GGUF files that:
- ✅ Pass `BITNET_STRICT_MODE=1` validation
- ✅ Require no runtime policy corrections
- ✅ Produce coherent outputs on CPU inference
- ✅ Have reproducible fingerprints for auditing

---

## Required Metadata (Strict Mode)

When using `st2gguf --strict` or validating with `BITNET_STRICT_MODE=1`, the following metadata keys are **required** in the GGUF file:

| Metadata Key | Description | Example Value |
|--------------|-------------|---------------|
| `general.architecture` | Model architecture type | `"bitnet-b1.58"` |
| `bitnet.hidden_size` | Hidden dimension size | `2560` |
| `bitnet.num_layers` | Number of transformer layers | `30` |
| `bitnet.num_heads` | Number of attention heads | `20` |
| `bitnet.vocab_size` | Vocabulary size | `128256` |
| `bitnet.context_length` | Maximum context length | `4096` |
| `general.file_type` | File type indicator (1=F16) | `1` |

These keys are automatically extracted from `config.json` during conversion. If your export fails strict validation with "missing required metadata key", ensure your `config.json` contains the corresponding fields:
- `model_type` → `general.architecture`
- `hidden_size` → `bitnet.hidden_size`
- `num_hidden_layers` → `bitnet.num_layers`
- `num_attention_heads` → `bitnet.num_heads`
- `vocab_size` → `bitnet.vocab_size`
- `max_position_embeddings` → `bitnet.context_length`

---

## Prerequisites

### Software Requirements

- **Rust toolchain** (MSRV 1.90.0+)
- **Python 3.8+** with packages:
  - `safetensors`
  - `torch`
  - `numpy`
- **BitNet-rs repository** cloned and built

### Model Requirements

You need one of:
1. **SafeTensors checkpoint**: One or more `.safetensors` files with `config.json`
2. **Hugging Face checkpoint**: Full model directory with PyTorch weights

Plus:
- **Tokenizer JSON**: `tokenizer.json` that matches the model's training tokenizer

---

## Quick Start

### Export from SafeTensors

```bash
# Export clean F16 GGUF
./scripts/export_clean_gguf.sh \
  models/bitnet-2b-4t \
  models/llama3-tokenizer/tokenizer.json \
  models/clean

# Validate the output
./scripts/validate_gguf.sh \
  models/clean/clean-f16.gguf \
  models/llama3-tokenizer/tokenizer.json
```

### Export from Hugging Face

```bash
# Assuming HF model directory structure
./scripts/export_clean_gguf.sh \
  /path/to/hf/model \
  /path/to/tokenizer.json \
  output/directory
```

---

## Step-by-Step Guide

### Step 1: Prepare Your Model

Ensure your model directory has the required files:

```bash
# For SafeTensors format
$ ls models/bitnet-2b-4t/
config.json
model.safetensors  # or multiple files: model-00001-of-00002.safetensors, etc.

# For HF format
$ ls /path/to/hf/model/
config.json
pytorch_model.bin  # or model.safetensors
tokenizer.json
tokenizer_config.json
```

### Step 2: Export Clean GGUF

The export script automatically:
- Detects model format (SafeTensors or HF)
- Converts weights to F16 format
- Preserves LayerNorm weights in float (not quantized)
- Generates fingerprint for reproducibility
- Creates metadata JSON

```bash
./scripts/export_clean_gguf.sh <model_dir> <tokenizer.json> <output_dir>
```

**Example:**

```bash
./scripts/export_clean_gguf.sh \
  models/bitnet-2b-4t \
  models/llama3-tokenizer/tokenizer.json \
  models/clean
```

**Output:**

```
INFO: Using BitNet-rs SafeTensors converter
INFO: Found 1 SafeTensors file(s), using: models/bitnet-2b-4t/model.safetensors
INFO: Converting SafeTensors to GGUF (F16 output type)...
[conversion progress...]
INFO: Computing fingerprint...
INFO: ✅ Export complete!
  Output: models/clean/clean-f16.gguf
  Fingerprint: sha256-abc123...
  Metadata: models/clean/clean-f16.meta.json
```

### Step 3: Validate the Clean GGUF

The validation script runs three checks:

1. **LayerNorm RMS Check**: Ensures LN weights have RMS ≈ 1.0 (not quantized)
2. **Projection Weight Check**: Verifies Q/K/V/O and FFN weights load correctly
3. **Linguistic Sanity Check**: Runs greedy inference to confirm non-gibberish output

```bash
./scripts/validate_gguf.sh <model.gguf> <tokenizer.json>
```

**Example:**

```bash
./scripts/validate_gguf.sh \
  models/clean/clean-f16.gguf \
  models/llama3-tokenizer/tokenizer.json
```

**Expected Output:**

```
===================================================
1/3: LayerNorm Statistics Check (Strict Mode)
===================================================
INFO: Checking LayerNorm RMS values (must be ~1.0)...
INFO: Running with BITNET_STRICT_MODE=1 (no corrections allowed)
[LN stats output...]
INFO: ✅ Found 24 healthy LayerNorm layers (RMS in [0.5, 2.0])

===================================================
2/3: Projection Weight RMS Check
===================================================
INFO: Loading model and checking projection weight statistics...
[PROJ RMS output...]
INFO: ✅ Projection weights loaded (see RMS values above)

===================================================
3/3: Greedy Inference Probe (Linguistic Sanity)
===================================================
INFO: Running deterministic greedy inference probe...
Generated output:
----------------------------------------
The capital of France is Paris.
----------------------------------------
INFO: ✅ Output contains recognizable words (linguistic sanity check passed)

===================================================
Validation Report
===================================================
INFO: ✅✅✅ ALL VALIDATION CHECKS PASSED ✅✅✅
```

### Step 4: Use the Clean Model

Once validated, you can use the model for inference:

```bash
cargo run -p bitnet-cli --no-default-features --features cpu -- \
  run \
  --model models/clean/clean-f16.gguf \
  --tokenizer models/llama3-tokenizer/tokenizer.json \
  --prompt "The meaning of life is"
```

---

## Optional: I2_S Quantization

If you need a quantized model (smaller size), you can convert to I2_S **after** producing the clean F16 GGUF:

```bash
./scripts/quantize_i2s_clean.sh \
  models/clean/clean-f16.gguf \
  models/clean
```

⚠️ **Important:** The quantizer **must exclude LayerNorm weights** from quantization. The provided script is currently a **stub** that requires a quantizer implementation with pattern-based exclusion support.

See: `docs/explanation/quantization-support.md` for I2_S specification.

---

## Troubleshooting

### Validation Fails: Suspicious LayerNorm RMS

**Symptom:**

```
ERROR: Model has suspicious LayerNorm weights (quantized or corrupted).
```

**Cause:** LayerNorm weights were quantized during export.

**Solution:**
1. Check your converter supports F16 output type
2. Ensure LayerNorm tensors are not being quantized
3. Re-export with `--outtype f16` flag (for llama.cpp converter)

### Gibberish Output During Validation

**Symptom:**

```
WARN: Output does not contain recognizable words
```

**Possible Causes:**
1. **Tokenizer mismatch**: Wrong `tokenizer.json` for this model
2. **RoPE parameter mismatch**: Model trained with different RoPE config
3. **Model corruption**: Export failed or source checkpoint is corrupted

**Solutions:**
1. Verify tokenizer matches model's training tokenizer
2. Check `config.json` for RoPE parameters (base, dim, scale)
3. Try re-exporting from source checkpoint

### Projection RMS Values Look Strange

**Symptom:** Projection weight RMS values are vastly different (e.g., Q=100, K=10000, V=50)

**Expected:** All projection weights (Q/K/V/O, FFN gate/up/down) should have similar RMS, typically O(10³).

**Solutions:**
1. Check source checkpoint quality
2. Verify conversion script preserves weight scales correctly
3. Compare with known-good model's projection RMS

### Export Script Can't Find Converter

**Symptom:**

```
ERROR: No converter found.
```

**Solutions:**
1. Set explicit converter: `CONVERTER=/path/to/convert.py ./scripts/export_clean_gguf.sh ...`
2. Ensure `scripts/convert_safetensors_to_gguf.py` exists (BitNet-rs built-in)
3. Vendor llama.cpp: `git submodule add https://github.com/ggerganov/llama.cpp third_party/llama.cpp`

---

## Advanced Usage

### Custom Converter

If you have a custom GGUF converter:

```bash
CONVERTER=/path/to/my_converter.py \
  ./scripts/export_clean_gguf.sh \
  models/my-model \
  tokenizer.json \
  output
```

Your converter must:
- Accept input model path and output path
- Support `--tokenizer` flag
- Write GGUF v3 format
- Preserve LayerNorm in float format

### Reproducible Builds

For reproducible exports, lock your Python environment:

```bash
# Create requirements.txt
pip freeze > requirements-export.txt

# Later, restore exact versions
pip install -r requirements-export.txt
```

The fingerprint in `*.fingerprint` files ensures you can verify reproducibility:

```bash
# Export on machine A
./scripts/export_clean_gguf.sh ...
cat models/clean/clean-f16.fingerprint
# sha256-abc123...

# Export on machine B with same inputs
./scripts/export_clean_gguf.sh ...
cat models/clean/clean-f16.fingerprint
# sha256-abc123... (should match!)
```

### CI Integration

Use the provided GitHub Actions workflow for automated builds:

```yaml
# .github/workflows/gguf_build_and_validate.yml
# Trigger with workflow_dispatch and provide model paths
```

See: `.github/workflows/gguf_build_and_validate.yml`

---

## Files Created

After export and validation, you'll have:

```
models/clean/
├── clean-f16.gguf           # Main GGUF model (F16 precision)
├── clean-f16.fingerprint    # SHA256 fingerprint (sha256-...)
└── clean-f16.meta.json      # Export metadata (source, date, converter, etc.)
```

---

## References

- **GGUF Format**: [ggml-org/gguf](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- **BitNet-rs Quantization**: `docs/explanation/quantization-support.md`
- **Validation Framework**: `docs/development/validation-framework.md`
- **Correction Policy** (for triage only): `docs/explanation/correction-policy.md`

---

## Summary

| Step | Command | Purpose |
|------|---------|---------|
| 1. Export | `./scripts/export_clean_gguf.sh <model> <tok> <out>` | Convert to F16 GGUF |
| 2. Validate | `./scripts/validate_gguf.sh <gguf> <tok>` | Check LN, PROJ, and inference |
| 3. Use | `cargo run -p bitnet-cli -- run --model <gguf> ...` | Run inference |

**Key Principle:** Clean models must pass strict validation **without** policy corrections. If validation fails, re-export—do not use runtime policies in production.

---

For questions or issues, see:
- **GitHub Issues**: [BitNet-rs/issues](https://github.com/microsoft/BitNet/issues)
- **Documentation**: `docs/` directory
- **Development Guide**: `docs/development/build-commands.md`
