# Add deterministic CPU microbenchmark with receipt generation

**Labels:** `enhancement`, `benchmark`, `ci`, `receipt`

**Priority:** High ⚠️ (Unblocks CI gate)

**Depends on:** PR #452 (receipt verification)

## Summary

Implement a tiny, deterministic CPU benchmark (~128 tokens) that writes `ci/inference.json` receipt with real compute evidence.

## Problem

Currently, the `verify-receipt` gate exists but has no benchmark to generate receipts. CI cannot validate real compute paths without actual inference receipts.

## Acceptance Criteria

- [ ] Implement `cargo run -p xtask -- benchmark --tokens 128 --deterministic`
- [ ] Benchmark runs against small GGUF model (e.g., `tests/models/tiny.gguf`)
- [ ] Sets deterministic environment (`BITNET_DETERMINISTIC=1`, `BITNET_SEED=42`, `RAYON_NUM_THREADS=1`)
- [ ] Writes `ci/inference.json` with:
  - `compute_path: "real"`
  - `kernels: ["avx2_matmul", "i2s_quantize", ...]` (actual CPU kernels used)
  - `backend: "cpu"`
  - Timing and token generation metadata
- [ ] Local gates script uncomments benchmark step
- [ ] CI workflow adds `xtask verify-receipt` step after benchmark

## Implementation Notes

- Keep benchmark fast (<5s on CI hardware)
- Use existing `bitnet-inference` streaming API
- Model should be committed to repo or auto-downloaded
- Receipt schema version must match `RECEIPT_SCHEMA` in `bitnet-inference`

## Example Usage

```bash
# Run benchmark (deterministic)
cargo run -p xtask -- benchmark --tokens 128 --deterministic

# Verify receipt
cargo run -p xtask -- verify-receipt --path ci/inference.json
```

## Expected Receipt Format

```json
{
  "schema_version": "1.0.0",
  "compute_path": "real",
  "backend": "cpu",
  "kernels": [
    "avx2_matmul",
    "i2s_quantize",
    "i2s_dequantize",
    "rope_apply",
    "softmax_inplace"
  ],
  "timing_ms": 2847,
  "tokens_generated": 128,
  "tokens_per_sec": 44.9,
  "environment": {
    "BITNET_VERSION": "0.1.0",
    "OS": "Linux",
    "BITNET_DETERMINISTIC": "1",
    "BITNET_SEED": "42",
    "RAYON_NUM_THREADS": "1"
  }
}
```

## CI Integration

```yaml
- name: Run CPU microbench
  run: cargo run -p xtask -- benchmark --tokens 128 --deterministic

- name: Verify receipt
  run: cargo run -p xtask -- verify-receipt --path ci/inference.json
```

## Files to Modify

- `xtask/src/main.rs` - Add `benchmark` subcommand
- `scripts/local_gates.sh` - Uncomment benchmark step
- `.github/workflows/*.yml` - Add CI steps
- `tests/models/` - Add or document tiny test model

## Related

- Depends on: PR #452 (receipt verification)
- Blocks: CI enforcement of receipt gate
- Related: Issue #3 (performance benchmarking infrastructure)

## Estimated Effort

~2 days
