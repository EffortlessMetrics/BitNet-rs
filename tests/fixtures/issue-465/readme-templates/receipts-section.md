## Receipt Verification

BitNet-rs implements **honest compute** verification through production receipts. Every inference run generates a receipt with kernel IDs proving real computation.

### Receipt Schema v1.0.0

```json
{
  "schema_version": "1.0.0",
  "compute_path": "real",
  "backend": "cpu",
  "model": "microsoft/bitnet-b1.58-2B-4T-gguf",
  "quantization": "i2s",
  "tokens_generated": 128,
  "throughput_tokens_per_sec": 15.3,
  "success": true,
  "kernels": [
    "i2s_cpu_quantized_matmul",
    "tl1_lut_dequant_forward",
    "attention_kv_cache_update",
    "layernorm_forward"
  ],
  "timestamp": "2025-10-15T12:00:00Z"
}
```

### xtask Commands

```bash
# Generate receipt (writes ci/inference.json)
cargo run -p xtask -- benchmark --model <model.gguf> --tokens 128

# Verify receipt passes quality gates
cargo run -p xtask -- verify-receipt ci/inference.json

# Strict mode (fail on warnings)
BITNET_STRICT_MODE=1 cargo run -p xtask -- verify-receipt ci/inference.json
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BITNET_DETERMINISTIC` | Enable deterministic inference | `0` |
| `BITNET_SEED` | Random seed for deterministic mode | `42` |
| `RAYON_NUM_THREADS` | Thread count (use `1` for determinism) | auto |
| `BITNET_STRICT_MODE` | Fail on validation warnings | `0` |
| `BITNET_GGUF` | Override model path | auto-discover `models/` |

### Receipt Requirements

**Honest Compute:**
- `compute_path` must be `"real"` (not `"mocked"`)
- `kernels` array must be non-empty
- Kernel IDs must be valid (non-empty, ≤128 chars, ≤10,000 count)

**CI Enforcement:**
- Model Gates (CPU) workflow requires valid receipts
- Branch protection blocks PRs with mocked receipts
- See [.github/workflows/model-gates.yml](.github/workflows/model-gates.yml)

### Baseline Receipts

Reference receipts are stored in [docs/baselines/](docs/baselines/) with datestamped filenames (`YYYYMMDD-cpu.json`). These baselines establish reproducible performance benchmarks for CPU inference.
