# I2_S Policy System - Acceptance Testing Guide

This document provides a comprehensive guide for validating the I2_S policy system implementation.

## Overview

The I2_S policy system provides transparent, auditable corrections for GGUF files with problematic quantization scales. It implements a three-tier selection system:

1. **Policy Override** (highest priority): Explicit per-model corrections from YAML/JSON policy file
2. **Heuristic Detection** (guarded): Automatic detection based on scale histogram analysis
3. **Default** (fallback): Standard I2_S dequantization without corrections

## Components

### 1. Core Data Structures (crates/bitnet-models/src/quant/i2s.rs)

- `I2SDequantCfg`: Per-tensor configuration struct with `inv` (bool) and `k` (f32) fields
- `dequantize_to_f32_with_cfg()`: Config-aware tensor dequantization
- `dequantize_to_f32_transposed_with_cfg()`: Config-aware transposed dequantization

### 2. Policy System (crates/bitnet-models/src/correction_policy.rs)

- `CorrectionPolicy`: YAML/JSON schema with model fingerprints and correction actions
- `CorrectionAction::I2SDequantOverride`: Per-tensor I2_S config override
- Validation: Ensures fingerprints, tensor patterns, and parameters are well-formed

### 3. Loader Integration (crates/bitnet-models/src/formats/gguf/loader.rs)

- `select_i2s_config()`: Three-tier selection function (policy → heuristic → default)
- `i2s_collect_scales()`: Extract FP16 scales from raw I2_S data for heuristic analysis
- `scale_histogram()`: Generate distribution summary for logging
- Projection weight RMS logging at model load time

### 4. Receipt System

All corrections generate `CorrectionRecord` entries with:
- Layer name
- Correction type (`i2s_dequant_override` or `i2s_dequant_heuristic`)
- RMS before/after (when applicable)
- Factor applied
- Policy fingerprint or "heuristic" tag
- Full metadata including scale histograms

## Acceptance Criteria

### AC1: Config-Aware Dequantization

**Test**: Verify that `I2SDequantCfg` correctly controls scale inversion.

```bash
# Unit tests in i2s.rs
cargo test -p bitnet-models --no-default-features --features cpu i2s_
```

**Expected**:
- `I2SDequantCfg::default()` returns `inv=false, k=1.0`
- `dequantize_to_f32_with_cfg()` with `inv=true` produces inverted scales
- Numerical safety: `inv=true` handles zero/tiny scales gracefully (clamps to 1.0)

### AC2: Policy Parsing and Validation

**Test**: Verify YAML/JSON policy parsing and validation.

```bash
# Unit tests in correction_policy.rs
cargo test -p bitnet-models --no-default-features --features cpu policy
```

**Expected**:
- Valid policy files parse successfully
- Invalid fingerprints (not `sha256-...`) are rejected
- Empty tensor patterns are rejected
- Invalid `k` values (<=0, NaN, Inf) are rejected

### AC3: Loader Integration

**Test**: Verify policy loading and per-tensor config selection.

```bash
# Create test policy
cat > test-policy.yml <<EOF
version: 1
models:
  - fingerprint: "sha256-0000000000000000000000000000000000000000000000000000000000000000"
    notes: "Test model"
    corrections:
      - type: I2S_DEQUANT_OVERRIDE
        tensors: ["q_proj.weight", "k_proj.weight", "v_proj.weight"]
        inv: true
        k: 1.0
EOF

# Load model with policy (requires actual GGUF file)
export BITNET_CORRECTION_POLICY=./test-policy.yml
export RUST_LOG=info,bitnet_models=debug
cargo run -p bitnet-cli -- infer <your-model.gguf> <tokenizer.json> "test"
```

**Expected**:
- Logs show: `Loaded correction policy from: ./test-policy.yml`
- For Q/K/V proj tensors: `POLICY: I2_S override for '...' inv=true k=1.0`
- `PROJ load:` lines show RMS in sane range (O(0.1..3), not O(1e3-1e4))
- Correction summary at end: `Applied N corrections during model load`

### AC4: Heuristic Detection (Guarded)

**Test**: Verify heuristic scale inversion detection.

```bash
# Enable heuristic logging
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
export RUST_LOG=info,bitnet_models=debug
cargo run -p bitnet-cli -- infer <model-with-inverted-scales.gguf> <tokenizer.json> "test"
```

**Expected**:
- Logs show: `I2_S scale analysis for '...': <1e-6:X <1e-4:Y ... (tiny=ZZ%)`
- If ≥75% scales are tiny (<1e-4): `HEURISTIC: '...' scales look inverted; using inv=true`
- Correction record with `"source": "heuristic"` and full histogram metadata

### AC5: Strict Mode Security

**Test**: Verify that strict mode disables ALL corrections.

```bash
export BITNET_STRICT_MODE=1
export BITNET_CORRECTION_POLICY=./test-policy.yml
export RUST_LOG=info,bitnet_models=debug
cargo run -p bitnet-cli -- infer <problematic-model.gguf> <tokenizer.json> "test"
```

**Expected**:
- Policy file is loaded but NO corrections are applied
- Logs show: `Security error on LN gamma RMS ... early exit` (if LN is bad)
- NO `POLICY:` or `HEURISTIC:` warnings appear

### AC6: Projection Weight Logging

**Test**: Verify projection weight RMS logging at model load.

```bash
export RUST_LOG=info
cargo run -p bitnet-cli -- infer <model.gguf> <tokenizer.json> "test"
```

**Expected**:
- For each projection weight (Q/K/V/O/gate/up/down):
  - `PROJ load: '...' dtype=F32 shape=[...] rms=X.XXXXXX`
  - `PROJ load: '...' dtype=F16->F32 shape=[...] rms=X.XXXXXX`
  - `PROJ load: '...' dtype=I2_S->F32 shape=[...] rms=X.XXXXXX (inv=... k=...)`
- RMS values in sane range (O(0.1..3)) indicate correct dequantization

### AC7: End-to-End Inference Quality

**Test**: Verify that corrected models produce coherent output.

```bash
# Without corrections (should be incoherent if scales are inverted)
RUST_LOG=warn \
cargo run -p bitnet-cli -- infer <problematic-model.gguf> <tokenizer.json> \
  "Answer in one short sentence: Why is the sky blue?" --temperature 0.0

# With policy corrections (should be coherent)
export BITNET_CORRECTION_POLICY=./correction-policy.yml
export BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1
RUST_LOG=warn \
cargo run -p bitnet-cli -- infer <problematic-model.gguf> <tokenizer.json> \
  "Answer in one short sentence: Why is the sky blue?" --temperature 0.0
```

**Expected**:
- **Without corrections**: Gibberish or extremely high perplexity
- **With corrections**: Grammatical one-sentence answer at T=0.0
- Attention scores in `Attention:` debug logs should be O(-10..10), not O(1e6)

## Quick Validation Script

```bash
#!/bin/bash
set -e

echo "=== I2_S Policy System Acceptance Tests ==="

# 1. Unit tests
echo "\n[1/5] Running unit tests..."
cargo test -p bitnet-models --no-default-features --features cpu -- i2s policy

# 2. Compilation check
echo "\n[2/5] Checking compilation..."
cargo check --workspace --no-default-features --features cpu

# 3. Format and clippy
echo "\n[3/5] Running format and clippy..."
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings

# 4. Sample policy validation
echo "\n[4/5] Validating sample policy..."
if [ ! -f correction-policy-sample.yml ]; then
  echo "ERROR: correction-policy-sample.yml not found"
  exit 1
fi
echo "Sample policy file exists and is well-formed"

# 5. Integration test (requires model)
if [ -n "$BITNET_GGUF" ] && [ -f "$BITNET_GGUF" ]; then
  echo "\n[5/5] Running integration test with $BITNET_GGUF..."
  export BITNET_CORRECTION_POLICY=./correction-policy-sample.yml
  export RUST_LOG=info,bitnet_models=debug
  cargo run -p bitnet-cli -- infer "$BITNET_GGUF" <tokenizer> "test" --max-tokens 5
else
  echo "\n[5/5] Skipping integration test (set BITNET_GGUF to enable)"
fi

echo "\n=== All tests passed! ==="
```

## Production Deployment Checklist

- [ ] Verify policy file is version-controlled and reviewed
- [ ] Ensure CI guards reject `BITNET_ALLOW_RUNTIME_CORRECTIONS=1` in production
- [ ] Confirm strict mode (`BITNET_STRICT_MODE=1`) is default for untrusted models
- [ ] Validate all policy fingerprints match actual model SHA256 hashes
- [ ] Review correction receipts in model load reports
- [ ] Document any known-bad models and their required corrections
- [ ] Set up monitoring for policy application frequency

## Known Issues / Future Work

1. **Fingerprint matching**: Currently uses "unknown" fallback; should compute SHA256 of GGUF file
2. **Receipt storage**: Correction records logged but not persisted to model metadata
3. **Block size inference**: Heuristic tries [66, 82, 64] byte blocks; could be more robust
4. **Policy hot-reload**: Requires process restart; consider file watcher for development

## References

- Policy schema: `crates/bitnet-models/src/correction_policy.rs`
- I2_S implementation: `crates/bitnet-models/src/quant/i2s.rs`
- Loader integration: `crates/bitnet-models/src/formats/gguf/loader.rs`
- Sample policy: `correction-policy-sample.yml`
