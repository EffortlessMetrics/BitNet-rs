# BitNet.rs Validation Quick Start

## One-Button Validation

```bash
# Quick validation (unit tests only)
./scripts/quick-validate.sh

# Full validation with model
MODEL_PATH=path/to/model.gguf \
TOKENIZER=path/to/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
./scripts/quick-validate.sh
```

## What Gets Validated

### 1. **Tokenizer Parity** ✅
- Ensures BitNet and HuggingFace tokenize identically
- Validates BOS/EOS handling, normalization, merges

### 2. **Greedy Argmax Invariant** ✅
- Verifies greedy decoding always picks argmax
- Uses `--assert-greedy` flag for runtime enforcement

### 3. **Logit Parity (τ-b)** ✅
- Score-aware Kendall's tau-b correlation on teacher-forced path
- Handles quantization-induced ties correctly
- Threshold: τ ≥ 0.60 (PR), τ ≥ 0.70 (nightly)

### 4. **NLL Parity** ✅
- Token-weighted negative log-likelihood comparison
- Teacher-forcing with proper BOS/PAD handling
- Threshold: |Δ| ≤ 1e-2 (FP32), ≤ 2e-2 (quantized)

### 5. **Performance Gate** ✅
- Decode throughput (tokens/second)
- First token latency
- Fails on ≥10% regression from baseline

## Environment Variables

### Required
- `MODEL_PATH`: Path to GGUF model file
- `TOKENIZER`: Path to tokenizer.json (or embedded)
- `HF_MODEL_ID`: Compatible HuggingFace model ID

### Optional Tuning
- `PROP_EXAMPLES`: Number of prompts to test (default: 12)
- `TAU_STEPS`: Steps to capture for τ-b (default: 24)
- `TAU_MIN`: Minimum τ-b threshold (default: 0.60)
- `DELTA_NLL_MAX`: Max NLL difference (default: 1e-2)
- `BITNET_BIN`: Path to bitnet binary (auto-detected)

## Troubleshooting

### Missing Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install jq bc

# macOS
brew install jq bc

# Python dependencies
pip install transformers torch scipy
```

### Common Issues

**"Tokenizer parity failed"**
- Check BOS/EOS auto-insertion settings
- Verify vocabulary and merges match
- Ensure normalization is consistent

**"Low τ-b correlation"**
- Verify using same teacher-forced path
- Check if HF side includes BOS token
- For quantized models, expected τ ≥ 0.50

**"NLL difference too high"**
- Verify token-weighted calculation
- Check BOS policy matches
- Use `DELTA_NLL_MAX=2e-2` for quantized models

**"Performance regression detected"**
- Check CPU governor: `sudo cpupower frequency-set -g performance`
- Ensure no background processes
- Update baseline after legitimate improvements

## Artifacts

On validation failure, check:
- `artifacts/parity_failures.jsonl` - Detailed failure records
- `perf_results.json` - Latest performance metrics

Replay specific failures:
```bash
python3 scripts/replay_parity.py --row 1 artifacts/parity_failures.jsonl
```

## CI Integration

Validation runs automatically on:
- **Pull Requests**: Quick validation (τ ≥ 0.60)
- **Nightly**: Strict validation (τ ≥ 0.70, 100 examples)
- **Release**: Full suite with performance gates

## Updating Baselines

After performance improvements:
```bash
# Run benchmark
scripts/bench-decode.sh

# If improvement is legitimate, update baseline
cp perf_results.json perf_baseline.json
git add perf_baseline.json
git commit -m "perf: Update baseline after <optimization>"
```

## Development Workflow

1. **Make changes** to inference/model code
2. **Run quick validation**: `./scripts/quick-validate.sh`
3. **Fix any failures** using artifacts and replay tools
4. **Update baselines** if performance improved
5. **Push** - CI will run full validation

## Need Help?

- Check `docs/VALIDATION.md` for detailed documentation
- Review `artifacts/parity_failures.jsonl` for failure details
- Use `scripts/replay_parity.py` to debug specific failures
- File issues with validation artifacts attached
