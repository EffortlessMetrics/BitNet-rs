# BitNet.rs Validation Framework

This document describes the comprehensive validation framework for BitNet.rs, ensuring correctness and parity with reference implementations.

## 🎯 Overview

The BitNet.rs validation framework uses a three-tier pyramid approach to catch regressions without false positives from legitimate quantization effects:

```
         NLL Parity
        /          \
    Logit Parity (τ-b)
   /                  \
Tokenizer Parity (exact)
```

Each layer builds on the previous one, providing increasingly sophisticated validation.

## 🔧 Components

### 1. Tokenizer Parity (Foundation)

**Purpose**: Ensure exact token ID matching between Rust and HuggingFace tokenizers.

**Command**:
```bash
BITNET_BIN=target/release/bitnet \
MODEL_PATH=models/bitnet/model.gguf \
TOKENIZER=models/bitnet/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
scripts/test-tokenizer-parity.py --smoke
```

**What it validates**:
- Exact token ID sequences
- BOS/EOS token handling
- Special token processing
- Vocabulary size consistency

### 2. Logit Parity (Belief Layer)

**Purpose**: Validate that model outputs preserve relative rankings despite quantization.

**Command**:
```bash
PROP_EXAMPLES=10 TAU_STEPS=24 LOGIT_TOPK=10 TAU_MIN=0.60 \
MODEL_PATH=models/bitnet/model.gguf \
TOKENIZER=models/bitnet/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
scripts/logit-parity.sh
```

**Key Features**:
- **Score-aware Kendall's tau-b**: Handles tied ranks from quantization
- **Deterministic top-k**: Stable sorting with tie-breaking by token ID
- **NaN robustness**: Demotes NaN/Inf to -∞ before ranking
- **Configurable thresholds**:
  - `TAU_MIN=0.60`: Default (allows quantization variance)
  - `TAU_MIN=0.70`: Strict mode for nightly tests

**Mathematical Foundation**:
```python
tau_b = (P - Q) / sqrt((P + Q + T_x) * (P + Q + T_y))
```
Where:
- P = concordant pairs
- Q = discordant pairs
- T_x, T_y = ties in each ranking

### 3. NLL Parity (Probability Layer)

**Purpose**: Ensure corpus-level perplexity matches reference implementations.

**Command**:
```bash
DELTA_NLL_MAX=1e-2 \
MODEL_PATH=models/bitnet/model.gguf \
TOKENIZER=models/bitnet/tokenizer.json \
HF_MODEL_ID=1bitLLM/bitnet_b1_58-3B \
PPL_FILE=crossval/data/ppl_smoke.txt \
scripts/nll-parity.sh
```

**Implementation Details**:
- **Token-weighted mean**: `Σ(token_nlls) / Σ(predicted_tokens)`
- **Teacher-forcing**: Exact decode path with causal masking
- **PAD masking**: Correctly excludes padding from loss
- **T-1 accounting**: Metrics use predicted tokens only

**Tolerance Thresholds**:
- `DELTA_NLL_MAX=1e-2`: FP32 vs FP32
- `DELTA_NLL_MAX=2e-2`: Quantized vs FP32
- `DELTA_NLL_MAX=5e-2`: Heavily quantized (i2s)

## 📊 Evaluation Commands

### Basic Perplexity Evaluation

```bash
# Evaluate on a corpus
target/release/bitnet eval \
  --model models/bitnet/model.gguf \
  --tokenizer models/bitnet/tokenizer.json \
  --text-file crossval/data/ppl_smoke.txt
```

### Teacher-Forcing with Logit Dump

```bash
# Explicit token path with logit capture
target/release/bitnet eval \
  --model models/bitnet/model.gguf \
  --tokenizer models/bitnet/tokenizer.json \
  --teacher-force-ids 1,2,3,4,5,6 \
  --dump-logit-steps 6 \
  --logits-topk 10 \
  --json-out /tmp/tf_eval.json
```

### Deterministic Generation

```bash
# Greedy generation with full determinism
target/release/bitnet run \
  --model models/bitnet/model.gguf \
  --tokenizer models/bitnet/tokenizer.json \
  --prompt "Define entropy." \
  --max-new-tokens 32 \
  --greedy \
  --deterministic \
  --threads 1 \
  --dump-logit-steps 8 \
  --logits-topk 10 \
  --json-out /tmp/run.json
```

## 🧪 Property-Based Testing

The validation framework uses Hypothesis for exhaustive testing:

```python
# crossval/props/test_logit_parity.py
@given(
    prompt=st.text(min_size=1, max_size=200),
    seed=st.integers(min_value=0, max_value=2**32-1),
    max_tokens=st.integers(min_value=1, max_value=32)
)
def test_logit_parity(prompt, seed, max_tokens):
    # Runs both implementations
    # Computes tau-b correlation
    # Asserts tau >= TAU_MIN
```

## 🔒 Determinism Controls

Full control over non-deterministic sources:

```bash
# Environment variables
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLAS_NUM_THREADS=1

# CLI flags
--deterministic  # Sets all above
--threads 1      # Single-threaded execution
--seed 42        # Fixed RNG seed
```

## 📈 CI Integration

### PR Gates (Required)

```yaml
# .github/workflows/validation.yml
validation:
  runs-on: ubuntu-latest
  steps:
    - name: Tokenizer Parity
      run: scripts/test-tokenizer-parity.py --smoke

    - name: Logit Parity
      run: |
        PROP_EXAMPLES=10 TAU_MIN=0.60 scripts/logit-parity.sh

    - name: NLL Parity
      run: |
        DELTA_NLL_MAX=2e-2 scripts/nll-parity.sh
```

### Nightly Strict Tests

```yaml
# .github/workflows/nightly.yml
nightly-strict:
  schedule:
    - cron: '0 2 * * *'
  steps:
    - name: Strict Validation
      run: |
        PROP_EXAMPLES=100 \
        TAU_MIN=0.70 \
        TAU_STEPS=32 \
        DELTA_NLL_MAX=1e-2 \
        scripts/full-validation.sh
```

## 🛠️ Debugging Tools

### Greedy Argmax Checker

```python
# scripts/check_greedy_argmax.py
import json, sys

def check(path):
    j = json.load(open(path))
    for step in j.get("logits_dump", []):
        topk = step["topk"]
        chosen = step.get("chosen_id")
        argmax = max(topk, key=lambda x: (x[1], -x[0]))[0]
        if argmax != chosen:
            raise SystemExit(f"Non-argmax at step {step['step']}")
    print("OK: greedy argmax invariant holds")
```

### Artifact Replay

```python
# scripts/replay_artifact.py
import json

def replay(artifact_path, row_idx):
    with open(artifact_path) as f:
        row = json.loads(f.readlines()[row_idx])

    # Re-run both implementations with row["prompt"], row["seed"]
    # Compare tau-b and NLL
    # Print detailed diff for debugging
```

## 📝 Key Invariants

1. **Greedy Argmax**: In greedy mode, chosen token = argmax(logits)
2. **Teacher-Forcing**: NLL computed through decode path, not special forward
3. **Token Weighting**: Corpus NLL = mean over tokens, not sequences
4. **Deterministic Top-K**: Ties broken by token ID (ascending)
5. **NaN Safety**: NaN/Inf → -∞ before any comparisons

## 🚀 Performance Considerations

- **Batching**: Validation uses single sequences for correctness
- **Threading**: Set to 1 for deterministic validation
- **Caching**: 15-minute cache for tokenizer/model loads
- **Artifacts**: JSONL format for efficient streaming

## 📚 References

- [Kendall's Tau-b](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)
- [Perplexity Calculation](https://huggingface.co/docs/transformers/perplexity)
- [Teacher Forcing](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/)

---

The validation framework ensures BitNet.rs maintains correctness while allowing for legitimate quantization effects. It catches real regressions early without false positives from numerical precision differences.
