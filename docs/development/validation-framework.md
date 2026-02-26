# Validation Framework

This document describes BitNet-rs's comprehensive validation system for model evaluation, performance testing, and cross-validation.

## Evaluation Commands

### Perplexity Evaluation
```bash
# Evaluate perplexity on a corpus (token-weighted NLL)
target/release/bitnet eval \
  --model models/bitnet/model.gguf \
  --tokenizer models/bitnet/tokenizer.json \
  --text-file crossval/data/ppl_smoke.txt

# Teacher-forcing evaluation with perplexity calculation
cargo run -p bitnet-cli -- score --model model.gguf --file test.txt
cargo run -p bitnet-cli -- score --model model.gguf --file validation.txt --device gpu --batch-size 8 --json-out results.json

# Model evaluation with external tokenizer and token limits
cargo run -p bitnet-cli -- score --model model.gguf --file large-dataset.txt --tokenizer tokenizer.json --max-tokens 1000
cargo run -p bitnet-cli -- score --model model.gguf --file large-dataset.txt --tokenizer tokenizer.model --max-tokens 1000  # SPM tokenizer
```

### Teacher-Forcing and Logit Analysis
```bash
# Teacher-forcing with explicit token IDs + logit dump
target/release/bitnet eval \
  --model models/bitnet/model.gguf \
  --tokenizer models/bitnet/tokenizer.json \
  --teacher-force-ids 1,2,3,4,5,6 \
  --dump-logit-steps 6 --logits-topk 10 \
  --json-out /tmp/tf_eval.json

# Deterministic greedy generation with logit tapping
target/release/bitnet run \
  --model models/bitnet/model.gguf \
  --tokenizer models/bitnet/tokenizer.json \
  --prompt "Define entropy." \
  --max-new-tokens 32 --greedy \
  --deterministic --threads 1 \
  --dump-logit-steps 8 --logits-topk 10 \
  --json-out /tmp/run.json
```

### Batch Inference with Performance Metrics
```bash
# Enhanced batch inference with prefill timing and structured performance metrics
cargo run -p bitnet-cli -- run --input-file prompts.txt --batch-size 4 --metrics --format json
cargo run -p bitnet-cli -- run --model model.gguf --prompt "Test prefill performance" --metrics --deterministic --seed 42

# Prefill performance testing with detailed metrics export
cargo run -p bitnet-cli -- run --model model.gguf --prompt "Analyze prefill performance" --metrics --format json
cargo run -p bitnet-cli -- run --input-file batch_prompts.txt --batch-size 8 --metrics --format json

# Performance comparison with comprehensive metrics (prefill is always enabled)
cargo run -p bitnet-cli -- run --model model.gguf --prompt "Compare performance" --metrics --format json
cargo run -p bitnet-cli -- run --model model.gguf --prompt "Standard inference" --metrics --verbose
```

## Cross-Validation Tests

### Tokenizer Parity
```bash
# Tokenizer parity check
BITNET_BIN=target/release/bitnet \
MODEL_PATH=models/bitnet/model.gguf \
TOKENIZER=models/bitnet/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
scripts/test-tokenizer-parity.py --smoke
```

### Logit Parity Testing
```bash
# Logit parity with tau-b correlation
PROP_EXAMPLES=10 TAU_STEPS=24 LOGIT_TOPK=10 TAU_MIN=0.60 \
MODEL_PATH=models/bitnet/model.gguf \
TOKENIZER=models/bitnet/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
scripts/logit-parity.sh
```

### NLL Parity Testing
```bash
# NLL parity (token-weighted)
DELTA_NLL_MAX=1e-2 \
MODEL_PATH=models/bitnet/model.gguf \
TOKENIZER=models/bitnet/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
PPL_FILE=crossval/data/ppl_smoke.txt \
scripts/nll-parity.sh
```

### Full Cross-Validation Workflow
```bash
# Full cross-validation (deterministic)
export BITNET_GGUF="$PWD/models/bitnet/ggml-model-i2_s.gguf"
export BITNET_DETERMINISTIC=1 BITNET_SEED=42
cargo run -p xtask -- full-crossval

# Model compatibility validation with weight mapper
cargo test --no-default-features --features cpu -p crossval --no-default-features test_validate_model_compatibility
cargo test --no-default-features --features cpu -p crossval --no-default-features test_validate_model_compatibility_reports_unmapped
```

## Performance Benchmarking

### Comprehensive Performance Testing
```bash
# Enhanced Performance Benchmarking and Regression Detection
# Setup performance environment and run comprehensive benchmarks
./scripts/setup-perf-env.sh && ./scripts/run-performance-benchmarks.sh

# Run GPU performance benchmarks
./scripts/run-performance-benchmarks.sh --features gpu --timeout 600

# Generate performance baselines from actual benchmark runs
./scripts/generate-performance-baselines.sh --platforms linux-x86_64 --iterations 10

# Detect performance regressions with detailed analysis
python3 scripts/detect-performance-regression.py benchmark-results/performance-report.json --fail-on-regression

# Quick benchmark comparison (Rust vs C++)
python3 benchmark_comparison.py --model model.gguf --iterations 3 --tokens 32

# Cross-platform performance testing
./scripts/run-performance-benchmarks.sh --target aarch64-unknown-linux-gnu --use-cross
```

## GGUF Validation

### Model Inspection and Validation
```bash
# Validate GGUF file
cargo run -p bitnet-cli -- compat-check model.gguf
cargo run -p bitnet-cli -- compat-check model.gguf --json  # JSON output

# Check model compatibility (read-only)
cargo run -p bitnet-cli -- compat-check "$BITNET_GGUF"

# Export fixed GGUF safely (non-destructive)
cargo run -p bitnet-cli -- compat-fix "$BITNET_GGUF" fixed.gguf
cat fixed.gguf.compat.json   # audit stamp
```

### Metadata Inspection
```bash
# Inspect model metadata (human-readable with categorization)
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- model.gguf

# Export model metadata as JSON
cargo run --example inspect_gguf_metadata --no-default-features --features cpu -- --json model.gguf
```

## Key Validation Features

1. **Token-Weighted NLL**: Proper corpus perplexity using `Σ(token_nlls) / Σ(predicted_tokens)`
2. **Teacher-Forcing**: Exact decode path with causal masking and position encoding
3. **Deterministic Top-K**: Stable sorting with tie-breaking by token ID, NaN demotion
4. **Logit Dumping**: Capture top-k logits at each generation step for analysis
5. **Tau-b Correlation**: Score-aware rank correlation for quantization robustness

## Enhanced Inference Engine Architecture

BitNet-rs features a production-ready inference engine with comprehensive performance monitoring:

### Core Inference Features
- **Explicit Prefill Support**: Dedicated `engine.prefill()` method for cache warming and latency measurement
- **Structured Performance Metrics**: Comprehensive timing breakdown including prefill, decode, and end-to-end metrics
- **Batch Inference Optimization**: Enhanced batch processing with proper prefill integration
- **Mock Infrastructure**: Comprehensive mock model and tokenizer implementations for testing
- **Safe Environment Management**: Proper unsafe block handling for environment variable operations

### Performance Monitoring and Metrics
- **TimingMetrics**: Structured performance measurement with prefill, decode, tokenization, and total timing
- **ThroughputMetrics**: Tokens per second calculation for prefill, decode, and end-to-end performance
- **Latency Tracking**: Detailed latency measurement for each inference stage
- **Memory Monitoring**: Optional memory usage tracking throughout inference operations

For more detailed information, see:
- [VALIDATION.md](VALIDATION.md) - Complete validation guide
- [VALIDATION_QUICK_START.md](VALIDATION_QUICK_START.md) - Quick start guide
- [Performance Benchmarking Guide](performance-benchmarking.md) - Detailed benchmarking
- [Test Suite Guide](test-suite.md) - Test configuration and execution
