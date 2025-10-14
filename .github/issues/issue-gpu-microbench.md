# Add GPU microbenchmark with graceful skip on non-CUDA hosts

**Labels:** `enhancement`, `benchmark`, `gpu`, `ci`, `receipt`

**Priority:** Medium

**Depends on:** CPU microbench (previous issue)

## Summary

Implement GPU benchmark that generates receipts with GPU kernel evidence and cleanly skips on hosts without CUDA.

## Problem

GPU CI lanes need receipt verification, but non-GPU hosts should skip gracefully without failing the build.

## Acceptance Criteria

- [ ] Implement `cargo run -p xtask -- benchmark --backend gpu --tokens 128 --deterministic`
- [ ] Detects GPU availability at runtime
- [ ] If no GPU: prints "SKIP: No CUDA device available" and exits 0
- [ ] If GPU present:
  - Writes `ci/inference.json` with `backend: "cuda"`
  - Includes GPU kernel IDs (e.g., `gemm_fp16`, `wmma_m16n16`, `i2s_quantize`)
  - Sets `compute_path: "real"`
- [ ] CI workflow adds GPU-specific verification:
  ```yaml
  - name: Verify GPU receipt (requires GPU kernels)
    if: matrix.backend == 'cuda'
    run: cargo run -p xtask -- verify-receipt --path ci/inference.json --require-gpu-kernels
  ```
- [ ] Update `scripts/local_gates.sh` to support GPU lane

## Implementation Notes

- Use `bitnet_kernels::device_features::gpu_available_runtime()` for detection
- Don't require `BITNET_GPU_FAKE` override (production detection)
- Ensure deterministic seed works on GPU too

## Example Usage

```bash
# Run GPU benchmark (auto-detects GPU)
cargo run -p xtask -- benchmark --backend gpu --tokens 128 --deterministic

# On GPU host: writes ci/inference.json with GPU kernels
# On CPU-only host: prints SKIP message and exits 0

# Verify GPU receipt
cargo run -p xtask -- verify-receipt --path ci/inference.json --require-gpu-kernels
```

## Expected GPU Receipt

```json
{
  "schema_version": "1.0.0",
  "compute_path": "real",
  "backend": "cuda",
  "kernels": [
    "gemm_fp16",
    "wmma_m16n16k16",
    "i2s_quantize",
    "cublas_gemm",
    "tl1_gpu_pack"
  ],
  "timing_ms": 487,
  "tokens_generated": 128,
  "tokens_per_sec": 262.8,
  "environment": {
    "BITNET_VERSION": "0.1.0",
    "OS": "Linux",
    "CUDA_VISIBLE_DEVICES": "0",
    "BITNET_DETERMINISTIC": "1"
  }
}
```

## CI Integration

```yaml
test-gpu:
  runs-on: [self-hosted, cuda]
  steps:
    - name: Check GPU
      run: nvidia-smi

    - name: Run GPU microbench
      run: cargo run -p xtask -- benchmark --backend gpu --tokens 128 --deterministic

    - name: Verify GPU receipt (requires GPU kernels)
      run: cargo run -p xtask -- verify-receipt --path ci/inference.json --require-gpu-kernels
```

## Files to Modify

- `xtask/src/main.rs` - Add `--backend gpu` support to benchmark command
- `scripts/local_gates.sh` - Add GPU benchmark lane (optional)
- `.github/workflows/gpu.yml` - Add GPU verification steps

## Related

- Depends on: CPU microbench (previous issue)
- Related: PR #439 (GPU feature flag unification)

## Estimated Effort

~1 day (reuses CPU microbench infrastructure)
