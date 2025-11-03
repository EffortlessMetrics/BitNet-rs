# Parity Receipt Format

## Overview

Parity receipts provide structured JSON output from per-token logits comparison between Rust and C++ implementations. They enable systematic debugging of divergence points and automated CI validation.

## Schema Version

Current version: **v1.0.0**

## Example Usage

Generate a receipt during cross-validation:

```bash
cargo run -p xtask --features inference -- crossval-per-token \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4 \
  --receipt /tmp/parity-receipt.json
```

## Receipt Structure

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | u32 | Schema version (currently 1) |
| `timestamp` | string | RFC3339 timestamp when receipt was generated |
| `model` | string | Path to GGUF model file |
| `backend` | string | C++ backend used: "bitnet" or "llama" |
| `prompt` | string | Input prompt (after template formatting) |
| `positions` | usize | Number of token positions compared |
| `thresholds` | object | Quality thresholds for validation (see below) |
| `rows` | array | Per-position metrics (see below) |
| `summary` | object | Aggregate summary metrics (see below) |

### Thresholds Object

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mse` | f32 | 1e-4 | Mean squared error threshold (lower is better) |
| `kl` | f32 | 0.1 | Kullback-Leibler divergence threshold (lower is better) |
| `topk` | f32 | 0.8 | Top-K agreement threshold (0.0-1.0, higher is better) |

### Position Metrics (rows)

Each row represents metrics for one token position:

| Field | Type | Description |
|-------|------|-------------|
| `pos` | usize | Token position (0-indexed) |
| `mse` | f32 | Mean squared error between Rust and C++ logits |
| `max_abs` | f32 | Maximum absolute difference across all logits at this position |
| `kl` | f32? | Kullback-Leibler divergence (optional - requires softmax normalization) |
| `topk_agree` | f32? | Top-K agreement (fraction of top-K tokens that match, optional) |
| `top5_rust` | array | Top-5 token IDs from Rust logits (highest to lowest) |
| `top5_cpp` | array | Top-5 token IDs from C++ logits (highest to lowest) |

### Summary Object

| Field | Type | Description |
|-------|------|-------------|
| `all_passed` | bool | True if all positions passed quality thresholds |
| `first_divergence` | usize? | First position where divergence was detected (None if all passed) |
| `mean_mse` | f32 | Mean MSE across all positions |
| `mean_kl` | f32? | Mean KL divergence across all positions (optional) |

## Example Receipt

See [parity-receipt-example.json](./parity-receipt-example.json) for a complete example.

## Interpreting Results

### Success Criteria

A receipt indicates successful parity if:

1. **`summary.all_passed == true`**: No position exceeded quality thresholds
2. **`summary.first_divergence == null`**: No divergence point detected
3. **Low MSE values**: `summary.mean_mse < thresholds.mse`
4. **High Top-K agreement**: `topk_agree >= thresholds.topk` (if computed)

### Debugging Divergence

If divergence is detected:

1. **Check `summary.first_divergence`**: Identifies the first problematic position
2. **Examine `rows[pos]`**: Review metrics for the divergent position
   - **High MSE**: Large overall difference in logit distributions
   - **High KL divergence**: Probability distributions are significantly different
   - **Low Top-K agreement**: Different tokens are being predicted
3. **Compare top5_rust vs top5_cpp**: Identify which tokens differ
4. **Use position info for trace capture**:

```bash
# Capture Rust trace at divergent position
BITNET_TRACE_DIR=/tmp/rs RUST_LOG=warn BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
  cargo run -p bitnet-cli --features cpu,full-cli -- run \
  --model model.gguf --tokenizer tokenizer.json \
  --prompt "Your prompt" --max-tokens <first_divergence + 1>

# Compare with C++ trace
cargo run -p xtask -- trace-diff /tmp/rs /tmp/cpp
```

## CI Integration

Receipts can be used in CI pipelines for automated quality gates:

```python
import json
import sys

with open('parity-receipt.json') as f:
    receipt = json.load(f)

if not receipt['summary']['all_passed']:
    print(f"❌ Parity check failed at position {receipt['summary']['first_divergence']}")
    sys.exit(1)

mean_mse = receipt['summary']['mean_mse']
if mean_mse > 1e-4:
    print(f"❌ Mean MSE too high: {mean_mse}")
    sys.exit(1)

print("✓ Parity check passed")
```

## Schema Evolution

Future versions may add:

- **v2**: Additional metrics (cosine similarity per-position, L2 distance, etc.)
- **v3**: Attention mask parity metrics
- **v4**: Generation mode support (multiple tokens generated, not just prompt)

All versions maintain backward compatibility for fields in v1.
