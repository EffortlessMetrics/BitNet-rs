# GPU Release Checklist

Pre-release verification checklist for GPU backend changes in BitNet-rs.
Run `scripts/gpu_release_check.sh` to execute all checks automatically.

## Automated Checks

The release check script (`scripts/gpu_release_check.sh`) verifies:

### 1. Feature Flag Compilation Matrix

All supported feature flag combinations must compile cleanly:

| Features | Target |
|---|---|
| `--features cpu` | CPU-only build |
| `--features gpu` | Full GPU (CUDA + Metal) |
| `--features cuda` | CUDA backend only |
| `--features cpu,gpu` | Combined CPU + GPU |
| `--features cpu,cuda` | Combined CPU + CUDA |
| `--features oneapi` | Intel oneAPI backend |

### 2. Kernel Test Suite

- All kernel unit tests pass (or skip cleanly when no GPU hardware is present)
- GPU-gated tests compile and are either executed or correctly skipped
- `BITNET_GPU_FAKE=cuda` tests exercise GPU code paths in CI
- `BITNET_GPU_FAKE=none` confirms graceful fallback to CPU

### 3. Documentation Currency

- `docs/GPU_SETUP.md` references current CUDA toolkit version
- `docs/INTEL_GPU_SETUP.md` references current driver packages
- `docs/gpu-kernel-architecture.md` reflects current module layout
- `CHANGELOG.md` contains GPU-related entries under `[Unreleased]`

### 4. CHANGELOG Audit

- Every GPU-facing PR has a corresponding CHANGELOG entry
- Entries use the correct category (`Added`, `Changed`, `Fixed`)
- Feature flag changes are documented

### 5. Clippy & Formatting

- `cargo clippy` passes with `-D warnings` for each GPU feature set
- `cargo fmt --check` confirms formatting is clean

## Manual Checks (Pre-Release Only)

These require actual GPU hardware and cannot be automated in CI:

- [ ] CUDA inference produces correct output on NVIDIA GPU
- [ ] Intel Arc inference produces correct output via OpenCL
- [ ] Performance regression check against previous release baselines
- [ ] Memory leak check (run extended inference session, monitor RSS)
- [ ] Multi-GPU device selection works correctly
- [ ] Receipt verification passes with `compute_path = "real"`

## Running the Checklist

```bash
# Full automated check (CI or local)
./scripts/gpu_release_check.sh

# Quick check (skip slow compilation tests)
./scripts/gpu_release_check.sh --quick

# Verbose output for debugging
./scripts/gpu_release_check.sh --verbose
```

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | All checks passed |
| 1 | Feature compilation failed |
| 2 | Kernel tests failed |
| 3 | Documentation check failed |
| 4 | CHANGELOG missing GPU entries |
| 5 | Clippy/fmt check failed |

## Integration with CI

The release checklist runs automatically via the `gpu-smoke.yml` workflow
on weekly schedule and can be triggered manually for release candidates.

## Adding New Checks

When adding a new GPU backend or kernel:

1. Add the feature combination to the compilation matrix in the script
2. Add any new documentation files to the docs-currency check
3. Ensure kernel tests use `#[cfg(any(feature = "gpu", feature = "cuda"))]`
   or the appropriate backend gate
4. Update this checklist document
