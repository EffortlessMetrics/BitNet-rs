# Inference MVP - Minimum Viable Product for CPU Inference

This document describes the Minimum Viable Product (MVP) acceptance criteria for BitNet-rs CPU inference and provides usage instructions for the acceptance test suite.

## Overview

The CPU inference MVP ensures that BitNet-rs can:
1. Load and validate GGUF models with proper error handling
2. Perform deterministic inference with reproducible outputs
3. Generate coherent text that meets quality baselines
4. Provide accurate performance receipts for validation

## MVP Acceptance Criteria

### 1. Model Loading and Validation

**Requirement**: Models must be loaded with proper validation of weights and metadata.

**Acceptance**:
- ✅ GGUF file parsing succeeds
- ✅ Tensor metadata is correctly extracted
- ✅ LayerNorm gamma statistics are validated (mean ≈ 1.0, RMS in [0.5, 2.0])
- ✅ Strict mode detects and rejects suspicious weights
- ✅ Non-strict mode issues warnings but continues

**Test Command**:
```bash
# Strict inspection (should detect LN issues)
BITNET_STRICT_MODE=1 cargo run -p bitnet-cli -- inspect --model model.gguf

# Non-strict inspection (warns but continues)
cargo run -p bitnet-cli -- inspect --model model.gguf
```

### 2. Numerical Stability

**Requirement**: Zero NaN/Inf values in inference path.

**Acceptance**:
- ✅ No NaN/Inf in attention scores
- ✅ No NaN/Inf in RMSNorm outputs
- ✅ No NaN/Inf in logits
- ✅ All tensors remain finite throughout forward pass

**Test Command**:
```bash
BITNET_DEBUG_ATTN_SCALE=1 BITNET_DEBUG_RMSNORM=1 \
cargo run -p bitnet-cli -- run --model model.gguf \
  --prompt "Test" --max-new-tokens 10 2>&1 | grep -i "nan\|inf"
# Should return no matches
```

### 3. Deterministic Inference

**Requirement**: Identical inputs produce identical outputs across runs.

**Acceptance**:
- ✅ Two runs with same seed produce identical token sequences
- ✅ Greedy decoding (temperature=0.0) is truly greedy
- ✅ Single-threaded mode eliminates non-determinism
- ✅ Token IDs match exactly between runs

**Test Command**:
```bash
# Run 1
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo run -p bitnet-cli -- run --model model.gguf \
  --prompt "Why is the sky blue?" --max-new-tokens 32 \
  --temperature 0.0 --json-out run1.json

# Run 2
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo run -p bitnet-cli -- run --model model.gguf \
  --prompt "Why is the sky blue?" --max-new-tokens 32 \
  --temperature 0.0 --json-out run2.json

# Compare
diff <(jq -c '.tokens.ids' run1.json) <(jq -c '.tokens.ids' run2.json)
# Should show no differences
```

### 4. Output Quality

**Requirement**: Generated text must be coherent and relevant to prompts.

**Acceptance**:
- ✅ Factual prompts produce relevant content
- ✅ Physics prompt ("Why is the sky blue?") mentions scattering/Rayleigh/light/atmosphere
- ✅ Counting prompt produces number sequences
- ✅ Translation prompt produces correct translations
- ✅ No gibberish or repetitive tokens

**Test Prompts**:
```bash
# Factual question
"Why is the sky blue?"
# Expected keywords: rayleigh, scatter, light, atmosphere, wavelength

# Counting
"Count to five:"
# Expected pattern: 1, 2, 3, 4, 5

# Translation
"Translate 'bonjour' to English:"
# Expected: hello
```

### 5. Receipt Validation

**Requirement**: Performance receipts accurately reflect compute path and backend.

**Acceptance**:
- ✅ `compute_path: "real"` (not "mock")
- ✅ `backend: "cpu"` for CPU inference
- ✅ `kernels[]` array is non-empty
- ✅ If corrections applied: `rms_before`, `rms_after`, `factor` within safe bounds
- ✅ Correction factors in [0.1, 10.0] (clamped for safety)
- ✅ Post-correction RMS in [0.5, 2.0] (target ≈ 1.0)

**Receipt Structure**:
```json
{
  "compute_path": "real",
  "backend": "cpu",
  "kernels": ["i2s_cpu_quantize", "avx2_matmul", "rmsnorm"],
  "corrections": [
    {
      "layer": "model.layers.0.ln_attn",
      "rms_before": 0.3,
      "rms_after": 1.05,
      "factor": 3.5
    }
  ],
  "tokens": {
    "generated": 32,
    "prompt": 10
  },
  "latency": {
    "total_ms": 2543.2
  },
  "throughput": {
    "tokens_per_second": 12.6
  }
}
```

## Acceptance Test Suite

### Quick Start

```bash
# Auto-discover model and tokenizer
./scripts/accept_mvp_cpu.sh

# Specify model and tokenizer
./scripts/accept_mvp_cpu.sh path/to/model.gguf path/to/tokenizer.json

# With correction policy (for known-bad models)
CORRECTION_POLICY=./corrections.yml ./scripts/accept_mvp_cpu.sh
```

### Environment Variables

- `MODEL_PATH`: Path to GGUF model (auto-discovers if not set)
- `TOKENIZER_PATH`: Path to tokenizer (auto-discovers if not set)
- `CORRECTION_POLICY`: Path to correction policy YAML (optional)
- `OUTPUT_DIR`: Output directory for test artifacts (default: `target/mvp-acceptance`)

### Test Sequence

The acceptance script runs the following tests:

1. **Strict Mode Inspection**
   - Validates model with `BITNET_STRICT_MODE=1`
   - Detects suspicious LayerNorm weights
   - Passes if warnings are issued for bad weights

2. **Non-Strict Inspection**
   - Validates model without strict mode
   - Should complete successfully with warnings

3. **Deterministic Inference (Run 1)**
   - Generates 32 tokens with seed=42
   - Checks for NaN/Inf in logs
   - Validates output contains expected keywords

4. **Deterministic Inference (Run 2)**
   - Repeats inference with same seed
   - Compares token IDs with Run 1
   - Fails if outputs differ

5. **Additional Quality Checks**
   - Counting test: "Count to five"
   - Translation test: "Translate 'bonjour' to English"
   - Validates keyword presence (warnings only, not failures)

6. **Receipt Validation**
   - Validates JSON structure
   - Checks compute_path and backend
   - Validates correction parameters if present

### Exit Codes

- `0`: All tests passed
- `1`: One or more tests failed (see log for details)

### Output Artifacts

All test outputs are saved to `$OUTPUT_DIR` with timestamp:

```
target/mvp-acceptance/
├── mvp_acceptance_20251012_143022.log          # Full test log
├── inspect_strict_20251012_143022.txt          # Strict inspection output
├── inspect_normal_20251012_143022.txt          # Non-strict inspection output
├── inference_run1_20251012_143022.txt          # Inference run 1 text output
├── inference_run1_20251012_143022.json         # Inference run 1 JSON receipt
├── inference_run1_20251012_143022.log          # Inference run 1 debug logs
├── inference_run2_20251012_143022.txt          # Inference run 2 text output
├── inference_run2_20251012_143022.json         # Inference run 2 JSON receipt
├── inference_count_20251012_143022.txt         # Counting test output
├── inference_count_20251012_143022.json        # Counting test receipt
├── inference_translate_20251012_143022.txt     # Translation test output
└── inference_translate_20251012_143022.json    # Translation test receipt
```

## Correction Policy (Optional)

For models with known-bad LayerNorm weights, you can provide a correction policy. This is a **temporary workaround** - the proper fix is always to regenerate the GGUF with proper weight formats.

### Policy-Driven Correction Workflow

```bash
# Step 1: Diagnose model issues
cargo run -p bitnet-cli -- inspect --ln-stats model.gguf

# Output will show suspicious LayerNorm gamma RMS values

# Step 2: Generate correction policy from inspection results
cargo run -p bitnet-cli -- inspect --ln-stats \
  --generate-policy correction-policy.yml \
  model.gguf

# Step 3: Validate policy in dry-run mode
export BITNET_CORRECTION_POLICY=./correction-policy.yml
cargo run -p bitnet-cli -- validate-policy \
  --model model.gguf \
  --dry-run

# Step 4: Apply corrections and run inference
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
export BITNET_DETERMINISTIC=1 BITNET_SEED=42

cargo run -p bitnet-cli --no-default-features --features cpu -- run \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "Test prompt" \
  --max-new-tokens 32 \
  --output inference-receipt.json

# Step 5: Verify corrections in receipt
jq '.corrections' inference-receipt.json
```

### Policy File Format (YAML)

```yaml
version: 1
policies:
  - name: "Microsoft BitNet 2B 4T - Quantized LayerNorm Fix"
    description: "Rescale LayerNorm gamma weights from quantized I2_S to proper FP32 range"

    # Model fingerprint (SHA256 of GGUF file or metadata hash)
    fingerprint:
      type: "gguf_sha256"
      value: "a1b2c3d4e5f6..."  # Full or partial hash

    # Corrections to apply
    corrections:
      - type: "layernorm_scale"
        description: "Rescale LayerNorm gamma RMS to ~1.0"
        parameters:
          target_rms: 1.0
          tolerance: 0.1
          layers: "all"  # or specific layer indices

        # Validation before applying
        preconditions:
          - rms_out_of_range: [0.5, 2.0]

        # Expected outcome
        postconditions:
          - rms_in_range: [0.9, 1.1]
```

### Using Correction Policy

```bash
# Run acceptance tests with policy-driven corrections
export BITNET_CORRECTION_POLICY=./correction-policy.yml
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
./scripts/accept_mvp_cpu.sh

# Corrections will be logged in inference receipts
```

**Important**:
- Runtime corrections are a **temporary workaround** for known-bad models
- Both `BITNET_CORRECTION_POLICY` and `BITNET_ALLOW_RUNTIME_CORRECTIONS` must be set
- CI blocks correction flags to prevent production deployment with workarounds
- The proper fix is to regenerate the GGUF file with LayerNorm weights in FP16/FP32 (not quantized)
- See [Correction Policy System](docs/explanation/correction-policy.md) for detailed documentation

## CI Integration

The acceptance script is designed for CI integration:

```yaml
# .github/workflows/acceptance.yml
- name: Run CPU MVP Acceptance
  run: ./scripts/accept_mvp_cpu.sh
  env:
    MODEL_PATH: models/test-model.gguf
    TOKENIZER_PATH: models/tokenizer.json
    OUTPUT_DIR: ${{ github.workspace }}/acceptance-results

- name: Upload Artifacts
  uses: actions/upload-artifact@v3
  if: always()
  with:
    name: mvp-acceptance-results
    path: acceptance-results/
```

## Troubleshooting

### Model Not Found

```bash
# Download test model
cargo run -p xtask -- download-model

# Or specify explicit path
MODEL_PATH=/path/to/model.gguf ./scripts/accept_mvp_cpu.sh
```

### Non-Deterministic Outputs

```bash
# Ensure deterministic mode is enabled
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# Disable any GPU usage
export BITNET_GPU_FAKE=none

# Run tests
./scripts/accept_mvp_cpu.sh
```

### NaN/Inf in Logs

If NaN/Inf values are detected:

1. Check LayerNorm statistics:
   ```bash
   cargo run -p bitnet-cli -- inspect --ln-stats model.gguf
   ```

2. Enable debug logging:
   ```bash
   RUST_LOG=debug BITNET_DEBUG_RMSNORM=1 \
   ./scripts/accept_mvp_cpu.sh 2>&1 | grep -i "nan\|inf"
   ```

3. Review attention and normalization:
   ```bash
   BITNET_DEBUG_ATTN_SCALE=1 BITNET_DEBUG_RMSNORM=1 \
   cargo run -p bitnet-cli -- run --model model.gguf \
     --prompt "Test" --max-new-tokens 5
   ```

### Poor Output Quality

If outputs are gibberish:

1. Verify model integrity:
   ```bash
   cargo run -p bitnet-cli -- compat-check model.gguf --strict
   ```

2. Check for quantized LayerNorm weights:
   ```bash
   BITNET_STRICT_MODE=1 cargo run -p bitnet-cli -- inspect --model model.gguf
   ```

3. Enable comprehensive diagnostics:
   ```bash
   ./scripts/debug_inference.sh model.gguf tokenizer.json "Test prompt"
   ```

## Related Documentation

- [INFERENCE_FIXES.md](INFERENCE_FIXES.md) - Surgical fixes for inference quality
- [INFERENCE_DIAGNOSIS.md](INFERENCE_DIAGNOSIS.md) - Diagnostic procedures
- [docs/environment-variables.md](docs/environment-variables.md) - Environment variable reference
- [docs/development/test-suite.md](docs/development/test-suite.md) - Test suite guide

## References

- Issue #447: Compilation failures and inference quality
- [CLAUDE.md](CLAUDE.md): Essential guidance for working with BitNet-rs
- [docs/quickstart.md](docs/quickstart.md): 5-minute setup guide
