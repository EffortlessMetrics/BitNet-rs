# BitNet-rs Validation Framework

## Overview

The BitNet-rs validation framework provides comprehensive testing and benchmarking tools to ensure correctness and performance of the 1-bit LLM implementation. It includes tokenizer parity testing, logit correlation analysis, perplexity evaluation, performance benchmarking infrastructure, and enhanced GGUF metadata inspection capabilities.

⚠️ **Performance Benchmarking Status**: As documented in [GOALS_VS_REALITY_ANALYSIS.md](GOALS_VS_REALITY_ANALYSIS.md), the performance benchmarking framework requires development to provide verified performance comparisons against the C++ implementation.

## Components

### 1. Main Validation Script (`scripts/validate_all.sh`)

Runs the complete validation suite including:
- Unit tests
- Tokenizer parity
- Greedy argmax invariant
- Logit parity (τ-b correlation)
- NLL parity (perplexity)
- Optional throughput benchmarking

**Usage:**
```bash
MODEL_PATH=models/bitnet/model.gguf \
TOKENIZER=models/bitnet/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
scripts/validate_all.sh
```

**For quantized models:**
```bash
DELTA_NLL_MAX=2e-2 scripts/validate_all.sh
```

### 2. Greedy Argmax Checker (`scripts/check_greedy_argmax.py`)

Validates that greedy decoding always selects the argmax token from logits.

**Usage:**
```bash
bitnet run --model model.gguf --tokenizer tokenizer.json \
  --prompt "Test" --greedy --dump-logit-steps 10 \
  --json-out output.json

python3 scripts/check_greedy_argmax.py output.json
```

### 3. Decode Throughput Benchmark (`scripts/bench-decode.sh`)

Measures generation throughput and first-token latency.

**Usage:**
```bash
MODEL_PATH=model.gguf TOKENIZER=tokenizer.json \
scripts/bench-decode.sh
```

**With baseline comparison:**
```bash
BENCH_BASELINE=baseline.json scripts/bench-decode.sh
```

### 4. Performance Gate (`scripts/perf-gate.sh`)

Automated performance regression detection with configurable thresholds.

**Usage:**
```bash
MODEL_PATH=model.gguf TOKENIZER=tokenizer.json \
scripts/perf-gate.sh
```

### 5. Replay Tool (`scripts/replay_parity.py`)

Debug parity failures by replaying specific test cases.

**Usage:**
```bash
python3 scripts/replay_parity.py --row 1 artifacts/parity_failures.jsonl
```

## CLI Enhancements

The `bitnet` CLI now supports advanced validation features:

### Run Command
```bash
bitnet run --model model.gguf --tokenizer tokenizer.json \
  --prompt "Hello world" \
  --greedy                    # Force greedy decoding
  --deterministic            # Single-threaded determinism
  --threads 1                # Explicit thread count
  --dump-logit-steps 10      # Capture first 10 steps
  --logits-topk 10          # Top-10 logits per step
  --assert-greedy           # Fail on non-argmax selection
  --json-out results.json
```

### Eval Command
```bash
bitnet eval --model model.gguf --tokenizer tokenizer.json \
  --text-file corpus.txt \
  --deterministic \
  --dump-logit-steps 24 \
  --logits-topk 10 \
  --json-out eval.json
```

### Inspect Command (Enhanced GGUF Metadata with Categorization)
```bash
# Comprehensive GGUF metadata inspection with categorization and statistics
bitnet inspect --model model.gguf                     # Categorized human-readable format
bitnet inspect --model model.gguf --json              # Structured JSON with tensor statistics

# Enhanced example with JSON support
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- model.gguf        # Human-readable
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- --json model.gguf  # JSON output

# Quick header validation in tests (enhanced)
cargo test -p bitnet-inference --test engine_inspect
```

**Enhanced Features:**
- **Memory Efficient**: Only reads GGUF header, not tensor data
- **Categorized Metadata**: Organized by model params, architecture, tokenizer, training, quantization
- **Tensor Statistics**: Parameter counts, memory estimates, data type distribution
- **JSON Serialization**: Both compact and pretty-printed formats for automation
- **Error Resilient**: Handles malformed GGUF files with detailed error messages
- **CI/CD Ready**: Fast inspection suitable for automated pipelines

**Validation Benefits:**
- **Pre-flight Checks**: Validate model format before expensive loading with enhanced categorization
- **Quantization Analysis**: Understand compression schemes with automated classification
- **Tensor Statistics**: Get parameter counts and memory estimates without loading
- **Architecture Inspection**: Examine model structure through categorized metadata
- **Debugging Support**: Inspect GGUF structure with organized output for troubleshooting

## Validation Thresholds

### PR Gates (Default)
- **Tau-b correlation**: ≥ 0.60
- **NLL delta**: ≤ 1e-2 (FP32), ≤ 2e-2 (quantized)
- **Examples**: 12 prompts, 24 steps
- **Performance regression**: ≤ 10%

### Nightly Validation (Strict)
- **Tau-b correlation**: ≥ 0.70
- **NLL delta**: ≤ 1e-2
- **Examples**: 100 prompts, 32 steps
- **Performance regression**: ≤ 10%

## Exit Codes

| Code | Meaning | Action Required |
|------|---------|-----------------|
| 0 | Success | None |
| 3 | Strict mapping failed | Check tensor mapping |
| 4 | Strict tokenizer failed | Fix tokenizer loading |
| 5 | NLL too high | Check perplexity calculation |
| 6 | Tau-b too low | Check logit correlation |
| 7 | Argmax mismatch | Fix greedy decoding |
| 9 | Performance regression | Profile and optimize |

## Environment Variables

### Determinism
- `BITNET_DETERMINISTIC=1`: Force deterministic execution
- `RAYON_NUM_THREADS=1`: Single-threaded CPU
- `BITNET_SEED=42`: Fixed random seed

### Validation Tuning
- `PROP_EXAMPLES`: Number of test examples
- `TAU_STEPS`: Steps for tau-b calculation
- `TAU_MIN`: Minimum tau-b threshold
- `DELTA_NLL_MAX`: Maximum NLL delta
- `LOGIT_TOPK`: Top-k logits to capture

### Performance
- `BENCH_PROMPTS`: Number of benchmark prompts
- `MAX_NEW_TOKENS`: Tokens per benchmark
- `PERF_REGRESSION_THRESHOLD`: Max acceptable regression %

## CI Integration

### GitHub Actions Workflow
```yaml
- name: Run Validation
  env:
    MODEL_PATH: ${{ env.MODEL_PATH }}
    TOKENIZER: ${{ env.TOKENIZER }}
    HF_MODEL_ID: ${{ env.HF_MODEL_ID }}
  run: scripts/validate_all.sh
```

### Nightly Validation
See `.github/workflows/nightly-validation.yml` for automated strict validation with artifact collection.

## Troubleshooting

### Tokenizer Parity Fails
1. Check BOS/EOS token handling
2. Verify vocabulary mapping
3. Compare special token IDs

### Low Tau-b Correlation
1. Check teacher-forcing path construction
2. Verify attention masks and position encoding
3. Look for NaN/inf in logits
4. Check tie-breaking determinism

### High NLL Delta
1. Verify token-weighted aggregation
2. Check PAD masking policy
3. For quantized models, increase threshold to 2e-2

### Performance Regression
1. Profile with `cargo bench`
2. Check recent changes to hot paths
3. Compare with `perf record/report`
4. Update baseline if regression is expected

## Quick Recipes

### Full validation (deterministic)
```bash
export MODEL_PATH="models/bitnet/model.gguf"
export TOKENIZER="models/bitnet/tokenizer.json"
export HF_MODEL_ID="1bitLLM/bitnet_b1_58-3B"
export BITNET_DETERMINISTIC=1 BITNET_SEED=42
scripts/validate_all.sh
```

### Quick smoke test
```bash
cargo test -p bitnet-cli --no-default-features --features cpu
```

### Performance baseline
```bash
MODEL_PATH=model.gguf TOKENIZER=tokenizer.json \
scripts/perf-gate.sh
```

### Debug parity failure
```bash
# Capture failure
PARITY_ARTIFACT=debug.jsonl scripts/validate_all.sh

# Replay specific case
python3 scripts/replay_parity.py --row 1 debug.jsonl
```
