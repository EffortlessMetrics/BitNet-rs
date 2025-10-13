# Clean GGUF Export Scripts

Production-ready scripts for exporting and validating BitNet models in GGUF format with LayerNorm preservation.

---

## Quick Start

```bash
# 1. Export clean F16 GGUF
./scripts/export_clean_gguf.sh \
  models/bitnet-2b-4t \
  models/llama3-tokenizer/tokenizer.json \
  models/clean

# 2. Validate (strict mode)
./scripts/validate_gguf.sh \
  models/clean/clean-f16.gguf \
  models/llama3-tokenizer/tokenizer.json

# 3. (Optional) Quantize to I2_S
./scripts/quantize_i2s_clean.sh \
  models/clean/clean-f16.gguf \
  models/clean
```

**Recommended:** Use Justfile targets instead:

```bash
just model-clean models/bitnet-2b-4t models/llama3-tokenizer/tokenizer.json
```

---

## Scripts Overview

### `export_clean_gguf.sh`

**Purpose:** Convert SafeTensors or HF checkpoint to clean F16 GGUF.

**Ensures:**
- LayerNorm weights remain in float format (F16)
- Generates SHA256 fingerprint
- Creates metadata JSON with provenance

**Usage:**
```bash
./scripts/export_clean_gguf.sh <model_dir> <tokenizer.json> <out_dir>
```

**Outputs:**
- `<out_dir>/clean-f16.gguf` - Main GGUF file
- `<out_dir>/clean-f16.fingerprint` - SHA256 hash (`sha256-...`)
- `<out_dir>/clean-f16.meta.json` - Export metadata

**Requirements:**
- Python 3.8+ with `safetensors`, `torch`, `numpy`
- Rust toolchain (for optional validation)
- Converter script (auto-detected or set via `CONVERTER` env var)

**Environment Variables:**
- `CONVERTER` - Explicit path to converter script (optional)

---

### `validate_gguf.sh`

**Purpose:** Comprehensive validation of GGUF models in strict mode.

**Checks:**
1. **LayerNorm RMS** - Must be ≈1.0 (not quantized)
2. **Projection Weight RMS** - Q/K/V/O and FFN weights load correctly
3. **Linguistic Sanity** - Greedy inference produces words, not gibberish

**Usage:**
```bash
./scripts/validate_gguf.sh <model.gguf> <tokenizer.json>
```

**Exit Codes:**
- `0` - All checks passed ✅
- `10` - LN inspection failed
- `11` - Suspicious LN weights detected
- `12` - No healthy LN layers found
- `13` - Model loading failed
- `14` - Inference probe failed
- `15` - Output is gibberish

**Requirements:**
- Rust toolchain (builds `bitnet-cli` if needed)
- `cargo` in PATH

**Environment Variables (auto-set by script):**
- `BITNET_STRICT_MODE=1` - No policy corrections
- `BITNET_DETERMINISTIC=1` - Reproducible inference
- `BITNET_SEED=42` - Inference seed
- `RAYON_NUM_THREADS=1` - Single-threaded

---

### `quantize_i2s_clean.sh`

**Purpose:** Quantize F16 GGUF to I2_S while excluding LayerNorm.

**Status:** 🚧 Stub implementation (requires quantizer binary)

**Usage:**
```bash
./scripts/quantize_i2s_clean.sh <clean-f16.gguf> <out_dir>
```

**Critical Constraint:**
- ✅ Quantize: `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`
- ❌ Never quantize: `attn_norm`, `ffn_norm`, `token_embd`, `output`

**Requirements (when implemented):**
- Quantizer binary with pattern-based exclusion support
- Set via `QUANTIZER` env var or build `bitnet-quantize` binary

**Outputs (when working):**
- `<out_dir>/clean-i2s.gguf` - Quantized model
- `<out_dir>/clean-i2s.fingerprint` - SHA256 hash
- `<out_dir>/clean-i2s.meta.json` - Quantization metadata

---

## Typical Workflow

### New Model Export

```bash
# Step 1: Export
just model-export \
  models/my-model \
  models/tokenizer.json \
  models/clean

# Step 2: Validate
just model-validate \
  models/clean/clean-f16.gguf \
  models/tokenizer.json

# Step 3: Generate baseline
just model-baseline \
  models/clean/clean-f16.gguf \
  models/tokenizer.json \
  docs/baselines/my-model-f16.md
```

### Quick Check

```bash
# One command: export + validate + baseline
just model-check models/my-model models/tokenizer.json
```

### Inspect Existing Model

```bash
# Check LayerNorm statistics
just model-inspect-ln models/existing.gguf

# Run inference probe
just model-probe \
  models/existing.gguf \
  models/tokenizer.json \
  "Once upon a time"
```

---

## Validation Criteria

### LayerNorm RMS

**Healthy Range:** [0.5, 2.0] (ideally ≈1.0)

**Example Good Output:**
```
blk.0.attn_norm  rms=1.023
blk.0.ffn_norm   rms=0.987
blk.1.attn_norm  rms=1.011
...
```

**Example Bad Output (quantized LN):**
```
blk.0.attn_norm  rms=0.018  ⚠️ SUSPICIOUS
blk.0.ffn_norm   rms=0.023  ⚠️ SUSPICIOUS
```

### Projection Weight RMS

**Expected:** All projection weights (Q/K/V/O, FFN) should have similar RMS, typically O(10³).

**Example Good Output:**
```
PROJ load: blk.0.attn_q    rms=1234.5
PROJ load: blk.0.attn_k    rms=1198.7
PROJ load: blk.0.attn_v    rms=1256.2
PROJ load: blk.0.attn_o    rms=1187.3
```

**Example Concerning Output:**
```
PROJ load: blk.0.attn_q    rms=100.5   ⚠️ Order of magnitude off
PROJ load: blk.0.attn_k    rms=12345.9 ⚠️ Very different from Q
```

### Linguistic Sanity

**Test Prompt:** "The capital of France is"

**Expected:** Output contains recognizable words like "Paris", "France", "capital"

**Good Example:**
```
The capital of France is Paris.
```

**Bad Example (gibberish):**
```
Th���e c��ap��i��t��al�� o��f�� F��ran��ce��
```

---

## Troubleshooting

### Export fails: "No converter found"

**Cause:** Converter script not detected.

**Solutions:**
1. Check `scripts/convert_safetensors_to_gguf.py` exists
2. Set explicit converter: `CONVERTER=/path/to/script ./scripts/export_clean_gguf.sh ...`
3. Vendor llama.cpp: `git submodule add https://github.com/ggerganov/llama.cpp third_party/llama.cpp`

### Validation fails: "Suspicious LayerNorm weights"

**Cause:** LayerNorm weights were quantized during export.

**Solution:** Re-export with F16 output type:
```bash
# For SafeTensors converter, ensure it writes F16
# For llama.cpp converter, use --outtype f16
```

### Validation fails: "Output is gibberish"

**Possible Causes:**
1. Tokenizer mismatch (wrong `tokenizer.json`)
2. RoPE parameter mismatch (check `config.json`)
3. Model corruption

**Diagnosis:**
```bash
# Check tokenizer
file models/tokenizer.json  # Should be JSON

# Manually test inference
cargo run -p bitnet-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "Test" --max-new-tokens 5
```

### Quantization fails: "No quantizer found"

**Cause:** I2_S quantizer binary not available (stub implementation).

**Solution:** Use F16 model directly, or wait for quantizer implementation.

---

## CI Integration

These scripts are integrated into CI via `.github/workflows/gguf_build_and_validate.yml`.

**Manual Trigger:**
1. Go to GitHub Actions → "Build & Validate Clean GGUF"
2. Click "Run workflow"
3. Provide inputs (model_dir, tokenizer_json, out_dir)

**Automatic Trigger:** On changes to export/validation scripts (push to main).

**Artifacts:** GGUF, fingerprint, metadata, and logs are uploaded as workflow artifacts.

---

## Environment Variables Reference

### Export

- `CONVERTER` - Explicit converter path (optional)

### Validation

These are **automatically set** by `validate_gguf.sh`:

- `BITNET_STRICT_MODE=1` - Enforce strict validation
- `BITNET_DETERMINISTIC=1` - Deterministic inference
- `BITNET_SEED=42` - Inference seed
- `RAYON_NUM_THREADS=1` - Single-threaded

These are **prohibited** (will cause validation failure):

- ❌ `BITNET_CORRECTION_POLICY` - No policy corrections
- ❌ `BITNET_ALLOW_RUNTIME_CORRECTIONS` - No runtime fixes
- ❌ `BITNET_FIX_LN_SCALE` - No LN rescaling

---

## Further Reading

- **User Guide:** `docs/howto/export-clean-gguf.md`
- **Baselines:** `docs/baselines/README.md`
- **Implementation:** `docs/CLEAN_GGUF_IMPLEMENTATION.md`
- **Policy System (Option 3):** `docs/explanation/correction-policy.md`
- **Quantization:** `docs/explanation/quantization-support.md`

---

## Summary

| Script | Purpose | Output |
|--------|---------|--------|
| `export_clean_gguf.sh` | Convert to F16 GGUF | `clean-f16.gguf`, fingerprint, metadata |
| `validate_gguf.sh` | Strict validation | Exit code (0=pass) |
| `quantize_i2s_clean.sh` | I2_S quantization | `clean-i2s.gguf` (stub) |

**Best Practice:** Use Justfile targets (`just model-*`) for convenience.

**Key Principle:** Clean models pass validation **without** policy corrections. If validation fails, re-export—do not use runtime policies in production.
