# Bulletproof Validation Framework - Critical Fixes Applied

## Summary
Applied surgical fixes to make the logit-parity and NLL validation truly bulletproof, especially for handling quantization artifacts and ensuring deterministic reproducibility.

## Critical Fixes Applied

### 1. Score-Aware Kendall's Tau-b (Tie Handling)
**Problem**: Original tau computed on token IDs only, missing ties in logit values common after quantization.

**Fix**: Added `kendalls_tau_b_scored()` that uses actual logit scores with epsilon-based tie detection:
```python
def kendalls_tau_b_scored(a_topk, b_topk, *, eps=1e-6):
    # Uses (token_id, score) pairs
    # Detects ties when |score_a - score_b| <= eps
    # Returns proper tau-b accounting for score ties
```

### 2. Deterministic Ordering for Equal Logits
**Problem**: Equal logits resulted in arbitrary ordering, causing run-to-run variation.

**Fix**: Sort by `(-logit, token_id)` to ensure deterministic tie-breaking:
```rust
// Deterministic comparison: ties broken by token ID
match safe_logits[b].partial_cmp(&safe_logits[a]) {
    Some(Ordering::Less) => Ordering::Less,
    Some(Ordering::Greater) => Ordering::Greater,
    Some(Ordering::Equal) | None => a.cmp(&b), // tie -> smaller ID first
}
```

### 3. NaN Demotion in Top-k Selection
**Problem**: NaN propagation could cause flaky test failures.

**Fix**: Demote non-finite values to NEG_INFINITY before selection:
```rust
let safe_logits: Vec<f32> = logits.iter()
    .map(|&v| if v.is_finite() { v } else { f32::NEG_INFINITY })
    .collect();
```

### 4. Proper Token Counting (T-1 for Predictions)
**Problem**: Inconsistent token counting between NLL and metrics.

**Fix**: Count predicted tokens (T-1) consistently:
```rust
// Regular evaluation
total_tokens += tokens.len().saturating_sub(1);

// Teacher-forcing
total_tokens: tf_ids.len().saturating_sub(1),
tokens_per_second: tf_ids.len().saturating_sub(1).max(1) as f64 / elapsed
```

### 5. Complete Threading Environment Variables
**Problem**: Missing BLAS/MKL thread controls could cause non-determinism.

**Fix**: Set all single-thread knobs:
```python
env.setdefault("RAYON_NUM_THREADS", "1")
env.setdefault("OMP_NUM_THREADS", "1")
env.setdefault("MKL_NUM_THREADS", "1")
env.setdefault("BLAS_NUM_THREADS", "1")
```

### 6. Teacher-Forcing Path Simplification
**Problem**: Unnecessary text_file requirement for teacher-forcing mode.

**Fix**: Made text_file optional when using `--teacher-force-ids`:
```rust
#[arg(long, required_unless_present = "teacher_force_ids")]
pub text_file: Option<PathBuf>,
```

### 7. Early k=0 Guard
**Problem**: k=0 could cause edge cases in topk selection.

**Fix**: Handle k=0 explicitly:
```rust
if k == 0 {
    dump.push(LogitStep { topk: vec![], ... });
    continue;
}
```

## Impact on Validation

These fixes ensure:

1. **Tie Robustness**: Quantization-induced ties are properly handled in tau-b
2. **Determinism**: Identical inputs produce identical outputs across runs
3. **Numeric Stability**: NaN/Inf values don't corrupt validation
4. **Metric Consistency**: Token counts align across NLL, perplexity, and TPS
5. **Environment Purity**: All parallel execution is disabled for reproducibility

## Test Thresholds

With these fixes, the following thresholds are achievable:

- **FP32 vs FP32**: |Δ mean_nll| ≤ 1e-2
- **Quant vs FP32**: |Δ mean_nll| ≤ 2e-2  
- **Median τ-b**: ≥ 0.60 over first N steps
- **Informative steps**: ≥ 8 with intersection ≥ 3

## Quick Validation Commands

```bash
# Build with deterministic settings
cargo build -p bitnet-cli --release --no-default-features --features cpu

# Run logit parity test
TAU_TIE_EPS=1e-6 TAU_MIN=0.60 TAU_STEPS=24 LOGIT_TOPK=10 \
MODEL_PATH=... TOKENIZER=... HF_MODEL_ID=... \
pytest crossval/props/test_logit_parity.py -v

# Run NLL parity test  
DELTA_NLL_MAX=1e-2 MODEL_PATH=... TOKENIZER=... HF_MODEL_ID=... \
PPL_FILE=crossval/data/ppl_smoke.txt \
pytest crossval/props/test_nll_parity.py -v

# Run greedy invariant test
MODEL_PATH=... TOKENIZER=... \
pytest crossval/props/test_greedy_invariants.py -v
```

## Artifact Collection

All failures now append to unified JSONL for debugging:
```bash
export PARITY_ARTIFACT=artifacts/parity_failures.jsonl
# Run tests...
# On failure, analyze: jq . artifacts/parity_failures.jsonl
```

## CI Integration

GitHub Actions workflow updated to:
- Run tokenizer parity smoke test
- Upload JSONL artifacts on failure
- Set all determinism environment variables

These surgical fixes make the validation framework extremely difficult to game while avoiding false positives from harmless numeric variations common in quantized models.