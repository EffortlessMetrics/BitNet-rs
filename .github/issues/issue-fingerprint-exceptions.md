# Add receipt fingerprinting to prevent false positives on fast GPUs

**Labels:** `enhancement`, `receipt`, `gpu`

**Priority:** Low (Future-proofing)

**Depends on:** GPU microbench (previous issue)

## Summary

Extend receipt schema with hardware fingerprints to allow fast GPUs without triggering false positives in duration-based validation.

## Problem

Very fast GPUs (e.g., A100, H100) might process 128 tokens so quickly that duration checks could flag them as suspicious. Need fingerprinting to allowlist known-good hardware.

## Acceptance Criteria

- [ ] Add receipt fields:
  - `gpu_cc: "8.0"` (CUDA compute capability)
  - `cpu_id: "GenuineIntel-i9-13900K"` (optional)
  - `os: "Linux 6.6.87"`
  - `rustc: "1.90.0"`
  - `bitnet_version: "0.1.0"`
- [ ] `verify-receipt` can optionally accept allowlist file:
  ```bash
  cargo run -p xtask -- verify-receipt --allowlist ci/known_fast_gpus.yml
  ```
- [ ] Allowlist format:
  ```yaml
  fast_gpus:
    - gpu_cc: "8.0"  # A100
      min_tokens_per_sec: 10000
    - gpu_cc: "9.0"  # H100
      min_tokens_per_sec: 20000
  ```
- [ ] Update `RECEIPT_SCHEMA` version to `1.1.0`
- [ ] Backward compatible with `1.0` receipts (fingerprints optional)

## Implementation Notes

- This is future-proofing - not urgent for CPU MVP
- Consider adding this when GPU benchmarks are stable
- Could also fingerprint for reproducibility (e.g., CI environment tracking)

## Example Extended Receipt

```json
{
  "schema_version": "1.1.0",
  "compute_path": "real",
  "backend": "cuda",
  "kernels": ["gemm_fp16", "wmma_m16n16k16"],
  "timing_ms": 142,
  "tokens_generated": 128,
  "tokens_per_sec": 901.4,
  "fingerprint": {
    "gpu_cc": "8.0",
    "gpu_name": "NVIDIA A100-SXM4-40GB",
    "cpu_id": "GenuineIntel-Xeon-Gold-6338",
    "os": "Linux 5.15.0-1047-gcp",
    "rustc": "1.90.0",
    "bitnet_version": "0.1.0"
  }
}
```

## Example Allowlist

```yaml
# ci/known_fast_gpus.yml
fast_gpus:
  - name: "A100"
    gpu_cc: "8.0"
    min_tokens_per_sec: 800
    note: "A100 can achieve >800 tok/s on small models"

  - name: "H100"
    gpu_cc: "9.0"
    min_tokens_per_sec: 1500
    note: "H100 2x faster than A100"
```

## Usage

```bash
# Verify with allowlist (relaxed duration checks for known-fast hardware)
cargo run -p xtask -- verify-receipt \
  --path ci/inference.json \
  --allowlist ci/known_fast_gpus.yml
```

## Files to Modify

- `xtask/src/main.rs` - Add `--allowlist` flag and fingerprint validation
- `crates/bitnet-inference/src/receipt.rs` - Update schema to v1.1
- `ci/known_fast_gpus.yml` - Example allowlist

## Related

- Depends on: GPU microbench issue
- Related: Receipt schema v1.0 (PR #452)

## Estimated Effort

~1 day
