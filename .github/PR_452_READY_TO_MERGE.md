# PR #452: Receipt Verification Gate - Ready to Merge

## Summary

Incorporated review feedback & hardening:

* **GPU kernels:** broadened detection (added `i2s_(quantize|dequantize)`, `cublas_*`, `cutlass_*`, `tl1/tl2_gpu_*`; explicitly exclude `i2s_cpu_*`)
* **Kernels typing:** all `kernels[]` entries must be strings
* **Schema const:** `RECEIPT_SCHEMA` alias; verification accepts `"1.0.0"` or `"1.0"`
* **Portable tests:** `xtask` tests resolve workspace without `.git` via `CARGO_WORKSPACE_DIR` or `[workspace]` in `Cargo.toml`
* **Shell portability:** `printf` instead of `echo -e`; `exit "$FAILED"` in `local_gates.sh`
* **Deps:** `once_cell` + `regex` centralize GPU patterns
* **Tests:** added unit test for `is_gpu_kernel_id()` (positive/negative cases)
* **Docs:** `CONTRIBUTING.md` updated with `verify-receipt` workflow

## Testing

All tests pass locally:
- ✅ Format check (`cargo fmt`)
- ✅ Clippy clean (`cargo clippy --all-targets --all-features`)
- ✅ Unit tests pass (`cargo test --workspace --no-default-features --features cpu`)
- ✅ GPU kernel detection test validates positive/negative cases
- ✅ Local gates script runs successfully

## What This Enables

This PR establishes the **keystone gate** for CPU MVP enforcement:

1. **Schema validation:** Receipts must use schema v1.0 with required fields
2. **Honest compute:** `compute_path` must be `"real"` (not `"mock"`)
3. **Real kernels:** `kernels[]` must be non-empty and contain actual kernel IDs
4. **GPU verification:** Optional `--require-gpu-kernels` flag ensures GPU claims are backed by GPU kernel execution
5. **Silent fallback detection:** Catches cases where GPU code silently falls back to CPU

## Status

**✅ Ready to merge**

All hardening complete. This PR only adds:
- Tooling (`xtask verify-receipt` command)
- Tests (unit test for GPU kernel detection)
- Documentation (CONTRIBUTING.md workflow)

No runtime hot paths affected. Safe to merge with maintainer override if CI is noisy.

## Next Steps (After Merge)

1. Wire gate in CI (see `.github/CI_INTEGRATION.md`)
2. Enable branch protection on receipt verification job
3. Create follow-up issues from roadmap (see `.github/NEXT_ROADMAP_ISSUES.md`)
4. Implement CPU microbench to generate receipts in CI
