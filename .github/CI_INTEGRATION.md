# CI Integration: Receipt Verification Gate

## Overview

Wire the `xtask verify-receipt` gate into CI to enforce honest compute evidence.

## CPU Verification Step

Add this step to your main CI workflow (after benchmark writes `ci/inference.json`):

```yaml
- name: Verify CPU receipt (strict)
  # Only run if receipt exists (until microbench is implemented)
  if: hashFiles('ci/inference.json') != ''
  run: |
    cargo run -p xtask -- verify-receipt --path ci/inference.json
```

**Location:** Add after your inference benchmark step, before any artifact uploads.

**Exit codes:**
- `0`: Receipt valid
- `1`: Receipt invalid or missing

## GPU Verification Step (Optional)

For GPU CI lanes, add GPU-specific verification:

```yaml
- name: Verify GPU receipt (requires GPU kernels)
  if: matrix.backend == 'cuda' || matrix.backend == 'gpu'
  run: |
    cargo run -p xtask -- verify-receipt --path ci/inference.json --require-gpu-kernels
```

**Important:** This step will fail if:
- Receipt claims GPU backend but contains no GPU kernels
- Common failure mode: Silent CPU fallback
- Diagnostic: Check `nvidia-smi`, verify `--features gpu` build, confirm `Device::Cuda(0)` in inference code

## Temporary Guard (Until Microbench Lands)

The `if: hashFiles('ci/inference.json') != ''` condition prevents failures when no receipt exists yet.

**Remove this guard** once the CPU microbench is implemented (roadmap issue).

## Example Integration

```yaml
jobs:
  test-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Run tests
        run: cargo test --workspace --no-default-features --features cpu

      # TODO: Add microbench step here
      # - name: Run CPU microbench
      #   run: cargo run -p xtask -- benchmark --tokens 128 --deterministic

      - name: Verify CPU receipt (strict)
        if: hashFiles('ci/inference.json') != ''
        run: cargo run -p xtask -- verify-receipt --path ci/inference.json

  test-gpu:
    runs-on: [self-hosted, cuda]
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Check GPU availability
        run: nvidia-smi

      - name: Run GPU tests
        run: cargo test --workspace --no-default-features --features gpu

      # TODO: Add GPU microbench step here
      # - name: Run GPU microbench
      #   run: cargo run -p xtask -- benchmark --backend gpu --tokens 128 --deterministic

      - name: Verify GPU receipt (requires GPU kernels)
        if: hashFiles('ci/inference.json') != ''
        run: cargo run -p xtask -- verify-receipt --path ci/inference.json --require-gpu-kernels
```

## Branch Protection

After wiring the gate, enable branch protection:

1. Navigate to: **Settings → Branches → Branch protection rules**
2. Select `main` branch
3. Enable: **Require status checks to pass before merging**
4. Add required jobs:
   - `test-cpu / Verify CPU receipt`
   - `test-gpu / Verify GPU receipt` (if you have GPU CI)
5. **Keep this required** once microbench is implemented

## Verification

Test locally before pushing:

```bash
# Generate a test receipt (manual for now)
mkdir -p ci
cat > ci/inference.json << 'EOF'
{
  "schema_version": "1.0.0",
  "compute_path": "real",
  "backend": "cpu",
  "kernels": ["avx2_matmul", "i2s_quantize"]
}
EOF

# Verify it passes
cargo run -p xtask -- verify-receipt --path ci/inference.json

# Should output:
# ✅ Receipt verification passed
#    Schema: 1.0.0
#    Compute path: real
#    Kernels: 2 executed
```

## Troubleshooting

### Receipt not found
```
Error: Failed to read receipt: ci/inference.json
```
**Fix:** Ensure benchmark step runs before verification, or add `if: hashFiles(...)` guard.

### GPU verification fails
```
Error: GPU kernel verification requested, but no GPU kernels found
```
**Fix:** Check:
1. Built with `--features gpu`
2. CUDA runtime available (`nvidia-smi`)
3. Inference uses `Device::Cuda(0)` (not `Device::Cpu`)

### Schema version mismatch
```
Error: Unsupported schema_version '2.0' (expected '1.0.0' or '1.0')
```
**Fix:** Update receipt generator to use schema v1.0, or update `verify_receipt_cmd()` to support new schema.
