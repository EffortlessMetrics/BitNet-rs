# CI Validation Framework

BitNet-rs uses a strict, deterministic CI validation framework with no compromises. All gates must pass - no skips, no mocks, no excuses.

## Quick Start

### PR Mode (Fast, Embedded Tokenizer)
```bash
# Fetch PR test model (TinyLlama with embedded tokenizer)
./scripts/fetch-pr-model.sh

# Run CI acceptance gates
CI_PR=1 ./scripts/ci-acceptance-gate.sh
```

### Nightly Mode (Full, External Tokenizer)
```bash
# Download BitNet model and tokenizer
cargo run -p xtask -- download-model

# Run CI acceptance gates
NIGHTLY=1 ./scripts/ci-acceptance-gate.sh
```

## Validation Gates

The CI runs 8 strict validation gates:

1. **Build & Binary Discovery** - Dynamic binary path discovery via cargo metadata
2. **Unit Tests** - All workspace tests must pass (excludes Python bindings)
3. **Model Selection** - Verifies required models/tokenizers exist
4. **Tensor Mapping** - All tensors must map correctly (no unmapped allowed)
5. **Strict Inference** - Inference with strict validation flags
6. **Tokenization** - Multiple prompts must tokenize correctly
7. **Determinism** - Token IDs must be identical across runs
8. **Performance** - Must meet baseline thresholds (95% perf, 103% memory)

## Exit Codes

Precise exit codes for failure triage:

- `0` - Success
- `1` - General error
- `2` - Missing model
- `3` - Tensor mapping failed
- `4` - Tokenizer failed
- `5` - Inference failed
- `6` - Tokenization failed
- `7` - Determinism failed
- `8` - Unit tests failed
- `9` - Performance regression
- `10` - Memory regression

## Models

### PR Model (TinyLlama)
- **Size**: ~400MB
- **Type**: Q2_K quantization
- **Tokenizer**: Embedded SentencePiece
- **Purpose**: Fast PR validation

### Nightly Model (MS BitNet)
- **Size**: ~1.8GB
- **Type**: I2_S quantization
- **Tokenizer**: External tokenizer.model
- **Purpose**: Full compatibility testing

## Updating Baselines

Performance baselines prevent regressions:

```bash
# Update all baselines
./scripts/update-baseline.sh --model all

# Update specific model
./scripts/update-baseline.sh --model tinyllama

# Force update without confirmation
./scripts/update-baseline.sh --model all --force
```

Baselines are stored in `ci/baseline.json` with:
- Tokens per second (median of 3 runs)
- RSS memory usage
- Model metadata

## Environment Variables

### Required for Determinism
```bash
export RAYON_NUM_THREADS=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export OMP_NUM_THREADS=1
export GGML_NUM_THREADS=1
```

### Model Paths
- `PR_MODEL` - Override PR model path
- `BITNET_GGUF` - BitNet model path for nightly
- `TOKENIZER_PATH` - External tokenizer path
- `TINYLLAMA_URL` - Custom URL for TinyLlama download
- `TINYLLAMA_SHA256` - Expected checksum for verification

## Key Features

### No Compromises
- ✅ All gates must pass
- ✅ No skipped tests
- ✅ No mock implementations
- ✅ Strict JSON validation
- ✅ Deterministic execution

### Robust Implementation
- **Binary Discovery** - No hardcoded paths
- **JSON Assertions** - Machine-verifiable conditions
- **Temp File Cleanup** - Automatic via trap
- **Retry Logic** - Download with 3 attempts
- **Checksum Verification** - SHA256 validation

### Performance Tracking
- **Regression Thresholds** - 95% performance, 103% memory
- **Baseline Comparison** - Against known good measurements
- **Noise Detection** - Warns if too few tokens decoded

## CI Integration

### GitHub Actions
```yaml
- name: Run CI Validation
  env:
    CI_PR: "1"  # or NIGHTLY: "1" for nightly
  run: ./scripts/ci-acceptance-gate.sh
```

### Exit Code Handling
```bash
./scripts/ci-acceptance-gate.sh
case $? in
    0) echo "All gates passed" ;;
    2) echo "Model missing - run fetch script" ;;
    9) echo "Performance regression detected" ;;
    10) echo "Memory regression detected" ;;
    *) echo "Validation failed" ;;
esac
```

## Troubleshooting

### Model Download Issues
```bash
# Verify checksum manually
sha256sum models/tinyllama-q2.gguf

# Use aria2c for faster downloads
sudo apt-get install aria2
./scripts/fetch-pr-model.sh
```

### Performance Variations
```bash
# Ensure deterministic environment
export BITNET_DETERMINISTIC=1
export RAYON_NUM_THREADS=1

# Check CPU frequency scaling
sudo cpupower frequency-set -g performance
```

### Binary Discovery Fails
```bash
# Fallback to manual build
cargo build -p bitnet-cli --release \
    --no-default-features --features "cpu,full-cli"

# Find binary manually
find target -name bitnet -type f -executable
```

## Maintenance

### Adding New Gates
1. Add exit code constant in script
2. Implement gate with JSON output
3. Use `jq -e` for assertions
4. Update this documentation

### Updating Models
1. Download new model
2. Generate SHA256: `sha256sum model.gguf`
3. Update fetch script with new URL and checksum
4. Run baseline update
5. Commit changes

## Philosophy

This CI framework embodies BitNet-rs's commitment to quality:

- **Strict** - No compromises, all gates must pass
- **Deterministic** - Reproducible results every time
- **Transparent** - JSON output for debugging
- **Fast** - PR mode with small model
- **Comprehensive** - Nightly mode with full validation

Remember: If it's not tested, it's broken. If it's flaky, it's broken. If it skips tests, it's broken.
