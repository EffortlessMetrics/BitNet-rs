# Model Baselines

This directory contains reproducible baselines for BitNet models exported to GGUF format.

## Purpose

Baselines serve as **known-good references** for:
- **Reproducibility**: Verify new exports match established fingerprints
- **Quality Assurance**: Ensure consistent inference outputs across builds
- **Regression Detection**: Catch unintended changes in model behavior
- **Auditing**: Provide traceable provenance for production models

## Baseline Format

Each baseline file documents:
1. **Model Identification**: Name, version, source
2. **Export Configuration**: Converter, precision, date
3. **Fingerprints**: SHA256 of GGUF and tokenizer
4. **Validation Results**: LN RMS, projection RMS, probe outputs
5. **Probe Outputs**: Deterministic greedy generations for canonical prompts

## Directory Structure

```
docs/baselines/
├── README.md                           # This file
├── bitnet-2b-4t-clean-f16.md          # Example baseline (template)
└── <model>-<variant>-<format>.md      # Per-model baselines
```

## Creating a Baseline

After exporting and validating a clean GGUF:

```bash
# 1. Export clean model
./scripts/export_clean_gguf.sh \
  models/bitnet-2b-4t \
  models/llama3-tokenizer/tokenizer.json \
  models/clean

# 2. Validate
./scripts/validate_gguf.sh \
  models/clean/clean-f16.gguf \
  models/llama3-tokenizer/tokenizer.json

# 3. Record baseline
# Copy fingerprint and validation outputs to a new baseline file:
cat > docs/baselines/bitnet-2b-4t-clean-f16.md <<EOF
# Baseline: BitNet-2B-4T Clean F16
[... see template below ...]
EOF
```

## Baseline Template

Use this template for new baselines:

```markdown
# Baseline: <Model Name> <Variant> <Format>

**Model:** <Full model name>
**Variant:** <e.g., 2B-4T, 1.58B-3B>
**Format:** <e.g., F16, I2_S>
**Export Date:** <YYYY-MM-DD>
**Exporter:** <converter script and version>

---

## Fingerprints

### GGUF Model
- **Path:** `models/clean/<name>.gguf`
- **SHA256:** `sha256-<hash>`
- **Size:** <file size in MB>

### Tokenizer
- **Path:** `models/llama3-tokenizer/tokenizer.json`
- **SHA256:** `sha256-<hash>`

---

## Export Configuration

- **Source:** <path or HF repo>
- **Converter:** `scripts/export_clean_gguf.sh`
- **Precision:** F16
- **LayerNorm:** Float (not quantized)

---

## Validation Results

### LayerNorm Statistics

- **Healthy Layers:** <count>
- **RMS Range:** [<min>, <max>]
- **Suspicious Layers:** None

Example LN RMS values:
```
blk.0.attn_norm  rms=1.023
blk.0.ffn_norm   rms=0.987
blk.1.attn_norm  rms=1.011
...
```

### Projection Weights

Example projection RMS values:
```
PROJ load: blk.0.attn_q    rms=1234.5
PROJ load: blk.0.attn_k    rms=1198.7
PROJ load: blk.0.attn_v    rms=1256.2
...
```

---

## Probe Outputs

### Configuration
- **Deterministic:** `BITNET_DETERMINISTIC=1`
- **Seed:** `BITNET_SEED=42`
- **Threads:** `RAYON_NUM_THREADS=1`
- **Temperature:** 0.0 (greedy)
- **Max Tokens:** 8

### Probe 1: Geography
**Prompt:** "The capital of France is"

**Expected Output:**
```
The capital of France is Paris.
```

### Probe 2: Storytelling
**Prompt:** "Once upon a time"

**Expected Output:**
```
Once upon a time there was a
```

### Probe 3: Programming
**Prompt:** "def factorial(n):"

**Expected Output:**
```
def factorial(n):
    if n == 0:
```

---

## Reproduction Steps

To reproduce this baseline:

```bash
# 1. Export
./scripts/export_clean_gguf.sh \
  <source_model_dir> \
  models/llama3-tokenizer/tokenizer.json \
  models/clean

# 2. Validate
./scripts/validate_gguf.sh \
  models/clean/<name>.gguf \
  models/llama3-tokenizer/tokenizer.json

# 3. Verify fingerprint
cat models/clean/<name>.fingerprint
# Should match: sha256-<hash>

# 4. Run probe
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo run -p bitnet-cli --no-default-features --features cpu -- \
  run --model models/clean/<name>.gguf \
  --tokenizer models/llama3-tokenizer/tokenizer.json \
  --prompt "The capital of France is" \
  --max-new-tokens 8 \
  --temperature 0.0
# Should match probe output above
```

---

## Notes

- This baseline was created on: <date>
- Git commit: `<commit_sha>`
- Rust version: `<rustc --version>`
- Python version: `<python --version>`

---

## Changelog

- **YYYY-MM-DD:** Initial baseline created
```

## Using Baselines

### Verifying a Model

To verify a newly exported model matches a baseline:

```bash
# 1. Check fingerprint
NEW_FP=$(cat models/clean/your-model.fingerprint)
BASELINE_FP="sha256-..."  # from baseline doc

if [[ "$NEW_FP" == "$BASELINE_FP" ]]; then
  echo "✅ Fingerprint matches baseline"
else
  echo "⚠️  Fingerprint differs from baseline"
  echo "   New:      $NEW_FP"
  echo "   Baseline: $BASELINE_FP"
fi

# 2. Run probe and compare outputs
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo run -p bitnet-cli --no-default-features --features cpu -- \
  run --model models/clean/your-model.gguf \
  --tokenizer tokenizer.json \
  --prompt "The capital of France is" \
  --max-new-tokens 8 \
  --temperature 0.0

# Compare output with baseline probe output
```

### Updating a Baseline

When intentionally changing export process:

1. Export and validate new model
2. Document changes in baseline changelog
3. Update fingerprints and probe outputs
4. Commit baseline changes with explanation

## Baseline Maintenance

- **Review Frequency:** On each major export tooling change
- **Retention:** Keep baselines for all production models
- **Deprecation:** Mark outdated baselines with `[DEPRECATED]` prefix

---

## See Also

- **Export Guide:** `docs/howto/export-clean-gguf.md`
- **Validation Framework:** `docs/development/validation-framework.md`
- **Quantization Support:** `docs/explanation/quantization-support.md`
