# CPU MVP Acceptance - Implementation Summary

## Overview

This document summarizes the implementation of the deterministic CPU acceptance script for MVP validation of BitNet-rs inference.

## Deliverables

### 1. Acceptance Script (`scripts/accept_mvp_cpu.sh`)

**Location**: `/home/steven/code/Rust/BitNet-rs/scripts/accept_mvp_cpu.sh`

**Features**:
- ✅ Automated test execution with 6 test cases
- ✅ Model and tokenizer auto-discovery
- ✅ Deterministic inference validation (two runs → identical tokens)
- ✅ NaN/Inf detection in logs
- ✅ Output quality validation (keyword checks)
- ✅ Receipt integrity validation
- ✅ Comprehensive logging and artifact collection
- ✅ Color-coded output for readability
- ✅ Exit code 0 on success, 1 on failure

**Test Sequence**:
```bash
1. Strict Mode Inspection       - BITNET_STRICT_MODE=1, detect bad LN
2. Non-Strict Inspection         - Warn but continue
3. Deterministic Inference (R1)  - Generate 32 tokens, check for NaN/Inf
4. Deterministic Inference (R2)  - Verify identical tokens vs R1
5. Additional Quality Checks     - Counting, translation prompts
6. Receipt Validation            - JSON structure, compute_path, kernels
```

**Usage**:
```bash
# Auto-discover model and tokenizer
./scripts/accept_mvp_cpu.sh

# Specify paths explicitly
./scripts/accept_mvp_cpu.sh model.gguf tokenizer.json

# With correction policy
CORRECTION_POLICY=./policy.yml ./scripts/accept_mvp_cpu.sh

# Custom output directory
OUTPUT_DIR=/tmp/results ./scripts/accept_mvp_cpu.sh
```

### 2. Documentation (`INFERENCE_MVP.md`)

**Location**: `/home/steven/code/Rust/BitNet-rs/INFERENCE_MVP.md`

**Contents**:
- MVP acceptance criteria (5 categories)
- Test command examples
- Correction policy format (YAML)
- CI integration guide
- Troubleshooting procedures
- Related documentation links

**Key Sections**:
1. Model Loading and Validation
2. Numerical Stability (zero NaN/Inf)
3. Deterministic Inference (reproducibility)
4. Output Quality (keyword validation)
5. Receipt Validation (compute path, backend, kernels)

### 3. Acceptance Checklist (`MVP_ACCEPTANCE_CHECKLIST.md`)

**Location**: `/home/steven/code/Rust/BitNet-rs/MVP_ACCEPTANCE_CHECKLIST.md`

**Contents**:
- 6 major sections with checkboxes
- Pre-flight checks
- Detailed test commands
- Success criteria for each test
- Failure modes and resolutions
- Sign-off section for manual review

**Categories**:
1. Model Loading and Validation (3 subsections)
2. Numerical Stability (3 subsections: attention, RMSNorm, logits)
3. Deterministic Inference (3 subsections: single-threaded, greedy, seed)
4. Output Quality (4 subsections: factual, counting, translation, no gibberish)
5. Receipt Validation (5 subsections: JSON, compute_path, backend, kernels, corrections)
6. Performance Metrics (3 subsections: latency, throughput, token counts)

## Test Sequence Details

### Test 1: Strict Inspection
```bash
BITNET_STRICT_MODE=1 cargo run -p bitnet-cli -- inspect --model "$MODEL"
```

**Acceptance**:
- Detects suspicious LayerNorm weights
- Issues warnings or errors for bad weights
- Completes without crashes

### Test 2: Non-Strict Inspection
```bash
cargo run -p bitnet-cli -- inspect --model "$MODEL"
```

**Acceptance**:
- Completes successfully
- Issues warnings (not errors) for suspicious weights
- Model metadata is displayed

### Test 3: Deterministic Inference (Run 1)
```bash
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo run -p bitnet-cli -- run \
  --model "$MODEL" --tokenizer "$TOKENIZER" \
  --prompt "Why is the sky blue?" \
  --max-new-tokens 32 --temperature 0.0 \
  --seed 42 --deterministic \
  --json-out run1.json
```

**Acceptance**:
- Inference completes successfully
- No NaN/Inf in logs
- Output contains expected keywords (rayleigh, scatter, light, atmosphere)
- JSON receipt is valid

### Test 4: Deterministic Inference (Run 2)
```bash
# Same command as Run 1, output to run2.json
```

**Acceptance**:
- Token IDs match Run 1 exactly: `diff <(jq -c '.tokens.ids' run1.json) <(jq -c '.tokens.ids' run2.json)`
- Generated text is identical
- Determinism is verified

### Test 5: Additional Quality Checks

**Test 5a: Counting**
```bash
Prompt: "Count to five:"
Expected: Contains 1, 2, 3, 4, 5 in sequence
```

**Test 5b: Translation**
```bash
Prompt: "Translate 'bonjour' to English:"
Expected: Contains "hello"
```

**Note**: These tests issue warnings only, not failures (to avoid brittleness).

### Test 6: Receipt Validation
```bash
# Validate receipt JSON structure
jq . run1.json

# Check required fields
jq -r '.compute_path' run1.json  # Must be "real"
jq -r '.backend' run1.json        # Must be "cpu"
jq '.kernels | length' run1.json  # Must be > 0
```

**Acceptance**:
- Valid JSON format
- `compute_path: "real"` (not mock)
- `backend: "cpu"`
- `kernels[]` is non-empty
- If corrections present: RMS and factors are within bounds

## Output Artifacts

All test outputs are saved to `$OUTPUT_DIR` (default: `target/mvp-acceptance`) with timestamp:

```
target/mvp-acceptance/
├── mvp_acceptance_YYYYMMDD_HHMMSS.log           # Full test log
├── inspect_strict_YYYYMMDD_HHMMSS.txt           # Strict inspection
├── inspect_normal_YYYYMMDD_HHMMSS.txt           # Non-strict inspection
├── inference_run1_YYYYMMDD_HHMMSS.txt           # Run 1 text
├── inference_run1_YYYYMMDD_HHMMSS.json          # Run 1 receipt
├── inference_run1_YYYYMMDD_HHMMSS.log           # Run 1 logs
├── inference_run2_YYYYMMDD_HHMMSS.txt           # Run 2 text
├── inference_run2_YYYYMMDD_HHMMSS.json          # Run 2 receipt
├── inference_run2_YYYYMMDD_HHMMSS.log           # Run 2 logs
├── inference_count_YYYYMMDD_HHMMSS.txt          # Counting test
├── inference_count_YYYYMMDD_HHMMSS.json         # Counting receipt
├── inference_translate_YYYYMMDD_HHMMSS.txt      # Translation test
└── inference_translate_YYYYMMDD_HHMMSS.json     # Translation receipt
```

## Exit Codes

- **0**: All tests passed (MVP accepted)
- **1**: One or more tests failed (see log for details)

## Environment Variables

### Required for Deterministic Mode
- `BITNET_DETERMINISTIC=1`: Enable deterministic inference
- `BITNET_SEED=42`: Set RNG seed
- `RAYON_NUM_THREADS=1`: Single-threaded execution

### Optional Configuration
- `MODEL_PATH`: Override model path (default: auto-discover)
- `TOKENIZER_PATH`: Override tokenizer path (default: auto-discover)
- `CORRECTION_POLICY`: Path to YAML correction policy
- `BITNET_ALLOW_RUNTIME_CORRECTIONS=1`: Enable corrections (requires policy)
- `OUTPUT_DIR`: Override output directory (default: `target/mvp-acceptance`)

### Debug Flags (from INFERENCE_FIXES.md)
- `BITNET_DEBUG_ATTN_SCALE=1`: Log attention scaling
- `BITNET_DEBUG_RMSNORM=1`: Log RMSNorm statistics
- `BITNET_DEBUG_GQA=1`: Log GQA shapes
- `BITNET_DEBUG_LOGITS=1`: Log tied embeddings
- `BITNET_DEBUG_MLP=1`: Log MLP norms
- `BITNET_DEBUG_ROPE=1`: Log ROPE application

## Correction Policy Format

If a model has known-bad LayerNorm weights, you can provide a correction policy:

```yaml
# correction-policy.yml
model_fingerprint: "sha256-hash-of-gguf-metadata"

corrections:
  - layer: "model.layers.*.ln_attn.weight"
    rule: "rescale_rms"
    target_rms: 1.0
    clamp_min: 0.1
    clamp_max: 10.0

  - layer: "model.layers.*.ln_mlp.weight"
    rule: "rescale_rms"
    target_rms: 1.0
    clamp_min: 0.1
    clamp_max: 10.0

validation:
  rms_envelope: [0.5, 2.0]
  fail_on_out_of_bounds: true
```

**Usage**:
```bash
export BITNET_CORRECTION_POLICY=./correction-policy.yml
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
./scripts/accept_mvp_cpu.sh
```

**Important**: Runtime corrections are a **temporary workaround**. The proper fix is to regenerate the GGUF with LayerNorm weights in FP16/FP32 (not quantized).

## CI Integration

### GitHub Actions Example

```yaml
name: MVP Acceptance

on: [push, pull_request]

jobs:
  acceptance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Download Test Model
        run: cargo run -p xtask -- download-model

      - name: Run MVP Acceptance
        run: ./scripts/accept_mvp_cpu.sh
        env:
          MODEL_PATH: models/test-model.gguf
          OUTPUT_DIR: ${{ github.workspace }}/acceptance-results

      - name: Upload Artifacts
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: mvp-acceptance-results
          path: acceptance-results/
```

## Validation Checklist Summary

| Category | Tests | Pass Criteria |
|----------|-------|---------------|
| Model Loading | 3 | GGUF parsed, LN validated, warnings issued |
| Numerical Stability | 3 | Zero NaN/Inf in attention, RMSNorm, logits |
| Deterministic Inference | 3 | Identical tokens, greedy selection, seed reproducibility |
| Output Quality | 4 | Keywords present, counting works, translation correct, no gibberish |
| Receipt Validation | 5 | Valid JSON, real compute, CPU backend, kernels present, corrections validated |
| Performance Metrics | 3 | Latency reasonable, throughput 10-20 tok/s, token counts consistent |

**Total**: 21 validation points

## Known Limitations

1. **Keyword validation**: Tests check for presence of keywords but don't validate semantic correctness
2. **Quality thresholds**: Acceptable output quality is subjective and model-dependent
3. **Correction policy**: Not all models will have correction policies defined
4. **Auto-discovery**: May fail if models are in non-standard locations

## Troubleshooting

### Model Not Found
```bash
# Download test model
cargo run -p xtask -- download-model

# Or set explicit path
MODEL_PATH=/path/to/model.gguf ./scripts/accept_mvp_cpu.sh
```

### Non-Deterministic Outputs
```bash
# Verify deterministic flags
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
export BITNET_GPU_FAKE=none  # Disable GPU

./scripts/accept_mvp_cpu.sh
```

### NaN/Inf Detected
```bash
# Enable debug logging
RUST_LOG=debug BITNET_DEBUG_RMSNORM=1 BITNET_DEBUG_ATTN_SCALE=1 \
./scripts/accept_mvp_cpu.sh 2>&1 | tee debug.log

# Inspect LayerNorm statistics
cargo run -p bitnet-cli -- inspect --model model.gguf --ln-stats
```

### Poor Output Quality
```bash
# Validate model
cargo run -p bitnet-cli -- compat-check model.gguf --strict

# Run comprehensive diagnostics
./scripts/debug_inference.sh model.gguf tokenizer.json "Test prompt"
```

## Related Documentation

| Document | Purpose |
|----------|---------|
| `INFERENCE_MVP.md` | Full MVP documentation with usage examples |
| `MVP_ACCEPTANCE_CHECKLIST.md` | Detailed checklist with manual validation steps |
| `INFERENCE_FIXES.md` | Surgical fixes for inference quality (attention, RMSNorm, I2S) |
| `INFERENCE_DIAGNOSIS.md` | Diagnostic procedures for inference issues |
| `docs/environment-variables.md` | Complete environment variable reference |
| `scripts/debug_inference.sh` | Comprehensive debug script with all flags |
| `CLAUDE.md` | Essential project guidance and quick reference |

## Future Enhancements

1. **Receipt validation**: Implement `cargo run -p xtask -- verify-receipt` command
2. **Inspect command**: Add `--ln-stats` flag to display LayerNorm RMS statistics
3. **Policy validation**: Validate correction policy YAML against schema
4. **Performance baselines**: Record and compare against historical baselines
5. **Regression detection**: Flag performance degradation >10% from baseline
6. **Semantic validation**: Check output quality with BLEU/ROUGE scores

## Sign-Off

**Implementation Date**: 2025-10-12

**Implementation**:
- ✅ Acceptance script (`scripts/accept_mvp_cpu.sh`)
- ✅ MVP documentation (`INFERENCE_MVP.md`)
- ✅ Acceptance checklist (`MVP_ACCEPTANCE_CHECKLIST.md`)
- ✅ Summary document (`MVP_ACCEPTANCE_SUMMARY.md`)

**Testing**:
- ✅ Syntax validation (bash -n)
- ✅ Script is executable (chmod +x)
- ✅ All environment variables documented
- ✅ Usage examples provided

**Readiness**: Ready for integration and testing

## Quick Reference

### Run Acceptance Test
```bash
./scripts/accept_mvp_cpu.sh
```

### Run with Correction Policy
```bash
CORRECTION_POLICY=./policy.yml \
BITNET_ALLOW_RUNTIME_CORRECTIONS=1 \
./scripts/accept_mvp_cpu.sh
```

### Run with Debug Logging
```bash
RUST_LOG=debug ./scripts/accept_mvp_cpu.sh 2>&1 | tee acceptance.log
```

### View Results
```bash
ls -lh target/mvp-acceptance/
cat target/mvp-acceptance/mvp_acceptance_*.log
```

### Check Determinism
```bash
# Compare two JSON receipts
diff <(jq -c '.tokens.ids' target/mvp-acceptance/inference_run1_*.json) \
     <(jq -c '.tokens.ids' target/mvp-acceptance/inference_run2_*.json)
```

---

**Status**: ✅ Implementation Complete

**Next Steps**:
1. Test script with real model
2. Integrate into CI pipeline
3. Document any edge cases discovered
4. Update troubleshooting guide based on real-world usage
