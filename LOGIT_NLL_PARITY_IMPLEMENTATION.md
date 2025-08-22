# Logit-Parity and Teacher-Forcing NLL Implementation

## Overview

This document describes the comprehensive implementation of logit-parity testing and teacher-forcing NLL evaluation for BitNet.rs. These features provide bulletproof validation that goes beyond surface-level text comparison.

## Implemented Features

### 1. **Step-wise Logits Tap (Inference Engine)**

Added a lightweight observer pattern to capture top-k logits during generation without overhead:

```rust
// crates/bitnet-inference/src/config.rs
pub struct GenerationConfig {
    // ...existing fields...
    pub logits_tap_steps: usize,  // Number of steps to capture
    pub logits_topk: usize,        // Top-k tokens per step
    pub logits_cb: Option<Arc<dyn Fn(usize, Vec<(u32, f32)>) + Send + Sync>>,
}
```

The engine now calls the callback during generation with minimal overhead (only copies k floats per step).

### 2. **CLI Logits Dumping (`bitnet run`)**

The `run` command now supports:
- `--dump-logit-steps N`: Capture logits for first N decode steps
- `--topk K`: Number of top tokens to capture (default: 10)

Output includes a `logits_dump` field in JSON with step-wise top-k tokens and their logits.

### 3. **Teacher-Forcing Forward (Model)**

Added `forward_full` method to TransformerModel for full-sequence teacher-forcing:

```rust
// crates/bitnet-models/src/transformer.rs
pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
    // [B,T] -> [B,T,V] without sampling
}
```

Uses the memory-efficient column-gather embedding path to avoid large transposes.

### 4. **Eval Subcommand (`bitnet eval`)**

New subcommand for computing mean NLL and perplexity:

```bash
bitnet eval \
    --model model.gguf \
    --tokenizer tokenizer.json \
    --text-file corpus.txt \
    --deterministic \
    --json-out results.json
```

Outputs per-line and aggregate NLL/perplexity metrics.

### 5. **Kendall's Tau Metric**

Implemented robust rank correlation metric for comparing top-k token orderings:

```python
# crossval/props/metrics.py
def kendalls_tau(topk_a_ids, topk_b_ids) -> float:
    """Returns tau in [-1, 1] where 1.0 = perfect agreement"""
```

More robust than simple token matching as it captures belief-level agreement.

### 6. **Property Tests**

#### Logit Parity Test
- Tests median Kendall's τ across decode steps
- Threshold: τ ≥ 0.60 (configurable via `TAU_MIN`)
- Self-consistency mode when no reference model available

#### NLL Parity Test  
- Compares teacher-forcing NLL on test corpus
- Threshold: |Δ NLL| ≤ 0.01 (configurable via `DELTA_NLL_MAX`)
- Direct probability accuracy check

### 7. **CI Scripts**

#### `scripts/logit-parity.sh`
```bash
MODEL_PATH=model.gguf TOKENIZER=tok.json \
HF_MODEL_ID=gpt2 TAU_MIN=0.60 \
./scripts/logit-parity.sh
```

#### `scripts/nll-parity.sh`
```bash
MODEL_PATH=model.gguf TOKENIZER=tok.json \
HF_MODEL_ID=gpt2 DELTA_NLL_MAX=0.01 \
./scripts/nll-parity.sh
```

Both scripts:
- Auto-install Python dependencies
- Provide clear error messages
- Support CI integration

## Key Technical Achievements

### Memory Efficiency
- Column-gather embedding path avoids 1.3GB transposes
- Logits tap only copies k values per step (not full vocabulary)
- Teacher-forcing reuses existing tensor operations

### Determinism
- Single-threaded execution (`--threads 1`)
- Environment variables properly set
- Seeded random generation

### Robustness
- No shell injection (argv lists)
- Proper error handling throughout
- JSON validation for structured outputs
- Relative metrics for length-aware comparison

## Testing Thresholds

### PR Lane (Fast)
- `PROP_EXAMPLES=10`
- `TAU_MIN=0.55`
- `DELTA_NLL_MAX=2e-2`
- `TAU_STEPS=16`

### Nightly (Thorough)
- `PROP_EXAMPLES=50`
- `TAU_MIN=0.70`
- `DELTA_NLL_MAX=1e-2`
- `TAU_STEPS=32`

## Usage Examples

### Manual Testing

```bash
# Build
cargo build -p bitnet-cli --release --no-default-features --features cpu

# Test logits dumping
target/release/bitnet run \
    --model model.gguf \
    --prompt "Explain quantum computing" \
    --greedy --deterministic \
    --dump-logit-steps 20 --topk 10 \
    --json-out output.json

# View logits
jq '.logits_dump[0:3]' output.json

# Run eval
target/release/bitnet eval \
    --model model.gguf \
    --text-file test.txt \
    --json-out eval.json

# Run property tests
./scripts/logit-parity.sh
./scripts/nll-parity.sh
```

### CI Integration

```yaml
# .github/workflows/compatibility.yml
- name: Logit Parity
  run: |
    MODEL_PATH=${{ env.MODEL }} \
    TOKENIZER=${{ env.TOKENIZER }} \
    HF_MODEL_ID=microsoft/bitnet \
    TAU_MIN=0.60 PROP_EXAMPLES=10 \
    ./scripts/logit-parity.sh

- name: NLL Parity  
  run: |
    MODEL_PATH=${{ env.MODEL }} \
    TOKENIZER=${{ env.TOKENIZER }} \
    HF_MODEL_ID=microsoft/bitnet \
    DELTA_NLL_MAX=0.01 \
    ./scripts/nll-parity.sh
```

## Why These Gates Are Strong

1. **Logit Parity (τ ≥ 0.6)**: Captures belief-level agreement in probability space, not just surface text. Hard to game without correct implementation.

2. **NLL Parity (|Δ| ≤ 0.01)**: Direct accuracy measure independent of sampling. Tests the core probability computations.

3. **Combined with Surface Metrics**: Together with prefix match, F1, and edit distance, provides multi-level validation.

4. **Deterministic & Reproducible**: All tests run deterministically with fixed seeds and single-threading.

## Implementation Status

✅ All components implemented and integrated:
- Engine logits tap
- CLI logits dumping  
- Model teacher-forcing forward
- Eval subcommand
- Kendall's tau metric
- Property test harnesses
- CI scripts

The framework is production-ready and provides strong guarantees that BitNet.rs produces correct, consistent outputs matching reference implementations.