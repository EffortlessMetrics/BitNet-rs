# CPU MVP Acceptance - Quick Reference Card

## One-Line Commands

```bash
# Run full acceptance test suite
./scripts/accept_mvp_cpu.sh

# With explicit paths
./scripts/accept_mvp_cpu.sh model.gguf tokenizer.json

# With correction policy
CORRECTION_POLICY=policy.yml ./scripts/accept_mvp_cpu.sh

# With custom output directory
OUTPUT_DIR=/tmp/results ./scripts/accept_mvp_cpu.sh
```

## Test Sequence (6 Tests)

| # | Test | Pass Criteria |
|---|------|---------------|
| 1 | Strict Inspection | Detects bad LN weights, issues warning/error |
| 2 | Non-Strict Inspection | Warns but continues |
| 3 | Deterministic Run 1 | Zero NaN/Inf, keywords present |
| 4 | Deterministic Run 2 | Identical tokens as Run 1 |
| 5 | Quality Checks | Counting/translation work |
| 6 | Receipt Validation | JSON valid, compute_path="real", backend="cpu" |

## Environment Variables

### Required for Determinism
```bash
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
```

### Optional Configuration
```bash
export MODEL_PATH=/path/to/model.gguf
export TOKENIZER_PATH=/path/to/tokenizer.json
export CORRECTION_POLICY=/path/to/policy.yml
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
export OUTPUT_DIR=/custom/output/dir
```

## Validation Checklist

- [ ] Zero NaN/Inf in logs
- [ ] Identical tokens across deterministic runs
- [ ] Output contains expected keywords
- [ ] Receipt: compute_path="real"
- [ ] Receipt: backend="cpu"
- [ ] Receipt: kernels[] non-empty
- [ ] Performance: 10-20 tok/s (CPU)

## Manual Tests

### Test Determinism
```bash
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo run -p bitnet-cli -- run --model model.gguf \
  --prompt "Why is the sky blue?" --max-new-tokens 32 \
  --temperature 0.0 --json-out run1.json

# Run again with same seed
# ... output to run2.json

# Compare
diff <(jq -c '.tokens.ids' run1.json) <(jq -c '.tokens.ids' run2.json)
# Should show no differences
```

### Check for NaN/Inf
```bash
BITNET_DEBUG_ATTN_SCALE=1 BITNET_DEBUG_RMSNORM=1 \
cargo run -p bitnet-cli -- run --model model.gguf \
  --prompt "Test" --max-new-tokens 5 2>&1 | grep -i "nan\|inf"
# Should return no matches
```

### Validate Receipt
```bash
jq . inference_output.json              # Valid JSON
jq -r '.compute_path' *.json            # Should be "real"
jq -r '.backend' *.json                 # Should be "cpu"
jq '.kernels | length' *.json           # Should be > 0
```

## Output Artifacts

All outputs saved to `$OUTPUT_DIR` (default: `target/mvp-acceptance`):

- `mvp_acceptance_*.log` - Full test log
- `inspect_strict_*.txt` - Strict inspection output
- `inspect_normal_*.txt` - Non-strict inspection
- `inference_run1_*.{txt,json,log}` - Run 1 artifacts
- `inference_run2_*.{txt,json,log}` - Run 2 artifacts
- `inference_count_*.{txt,json}` - Counting test
- `inference_translate_*.{txt,json}` - Translation test

## Exit Codes

- `0` - All tests passed (MVP accepted)
- `1` - One or more tests failed

## Troubleshooting

### Model Not Found
```bash
cargo run -p xtask -- download-model
# Or set: MODEL_PATH=/path/to/model.gguf
```

### Non-Deterministic
```bash
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
export BITNET_GPU_FAKE=none
```

### NaN/Inf Detected
```bash
cargo run -p bitnet-cli -- inspect --model model.gguf --ln-stats
RUST_LOG=debug BITNET_DEBUG_RMSNORM=1 ./scripts/debug_inference.sh
```

### Poor Quality
```bash
cargo run -p bitnet-cli -- compat-check model.gguf --strict
./scripts/debug_inference.sh model.gguf tokenizer.json "Test"
```

## Correction Policy (Optional)

```yaml
# correction-policy.yml
version: 1
policies:
  - name: "LayerNorm Gamma Rescale"
    fingerprint:
      type: "gguf_sha256"
      value: "abc123..."
    corrections:
      - type: "layernorm_scale"
        parameters:
          target_rms: 1.0
          tolerance: 0.1
          layers: "all"
```

**Usage**:
```bash
export BITNET_CORRECTION_POLICY=./correction-policy.yml
export BITNET_ALLOW_RUNTIME_CORRECTIONS=1
./scripts/accept_mvp_cpu.sh
```

## Test Prompts

| Prompt | Expected Keywords |
|--------|------------------|
| "Why is the sky blue?" | rayleigh, scatter, light, atmosphere, wavelength |
| "Count to five:" | 1, 2, 3, 4, 5 (in order) |
| "Translate 'bonjour' to English:" | hello |

## Performance Baselines (CPU)

| Quantization | Tokens/Second | Notes |
|--------------|---------------|-------|
| I2_S | 10-20 | Production baseline |
| TL1 | 12-18 | ARM optimized |
| TL2 | 10-15 | x86 optimized |

> ⚠️ Values >150 tok/s suggest mock inference

## Related Commands

```bash
# Inspect model
cargo run -p bitnet-cli -- inspect --model model.gguf

# Check compatibility
cargo run -p bitnet-cli -- compat-check model.gguf --strict

# Debug inference
./scripts/debug_inference.sh model.gguf tokenizer.json "Prompt"

# Run with all debug flags
BITNET_DEBUG_ATTN_SCALE=1 \
BITNET_DEBUG_RMSNORM=1 \
BITNET_DEBUG_GQA=1 \
BITNET_DEBUG_LOGITS=1 \
cargo run -p bitnet-cli -- run --model model.gguf --prompt "Test"
```

## Documentation

| File | Purpose |
|------|---------|
| `scripts/accept_mvp_cpu.sh` | Automated test script |
| `INFERENCE_MVP.md` | Full MVP documentation |
| `MVP_ACCEPTANCE_CHECKLIST.md` | Manual validation checklist |
| `MVP_ACCEPTANCE_SUMMARY.md` | Implementation summary |
| `MVP_QUICK_REFERENCE.md` | This quick reference |
| `INFERENCE_FIXES.md` | Surgical fixes reference |
| `docs/environment-variables.md` | Complete env var reference |

## CI Integration

```yaml
- name: Run MVP Acceptance
  run: ./scripts/accept_mvp_cpu.sh
  env:
    MODEL_PATH: models/test-model.gguf
    OUTPUT_DIR: acceptance-results
```

---

**TIP**: Run `./scripts/accept_mvp_cpu.sh 2>&1 | tee acceptance.log` to save full output for review.
