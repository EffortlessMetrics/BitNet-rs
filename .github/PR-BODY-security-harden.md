# Security Hardening — CPU MVP Finalization

## Overview
This PR finalizes the CPU MVP by:
- Eliminating flaky timeouts and external dependencies during receipt validation
- Ensuring strict, deterministic, offline-verifiable inference receipts
- Adding GPU test skip guards to prevent spurious failures on CPU-only machines
- Improving developer experience with comprehensive Justfile recipes

## Changes

### Test Infrastructure
- **GPU Skip Guards**: Added `tests/common/gpu.rs` with `gpu_tests_enabled()` helper
  - Requires `BITNET_ENABLE_GPU_TESTS=1` to run GPU tests
  - Prevents failures on CPU-only CI runners
  - Applied to GPU performance baseline tests

- **TL Packing Tests**: Added `crates/bitnet-kernels/tests/tl_packed_correctness.rs`
  - Verifies TL1 4-bit nibble packing/unpacking
  - Verifies TL2 8-bit byte packing with boundary conditions

### Strict Mode Enforcement
- **QuantizedLinear FP32 Guard**: Added runtime error when strict mode requires quantized projections but encounters FP32 layers
  - Aligns with policy: attention projections must be quantized in strict mode
  - Prevents silent FP32 fallback in production

### Receipt Infrastructure
- **GGUF Fallback**: `xtask verify-receipt` now auto-discovers models in `models/` directory via `BITNET_GGUF` environment variable
- **Rust Version**: Receipts now populate `rust_version` field from `rustc --version` for toolchain fingerprinting
- **Schema Updates**: Baseline fixtures include `tokens_per_second` field

### Developer Experience
- **Justfile Improvements**:
  - Added `help` command to list available recipes
  - Updated `test-tl-stress` to run both correctness and stress tests
  - Consolidated common workflows into single commands

### Documentation
- Updated validation summary and clarified receipt gate requirements
- Documented kernel hygiene rules (length ≤ 128, count ≤ 10K, no empty strings)
- Added troubleshooting guide for common validation failures

## Verification

### Local Validation
```bash
# CPU lane (clippy + tests)
cargo clippy --workspace --all-targets --no-default-features --features cpu -D warnings
cargo test   --workspace --no-default-features --features cpu

# TL correctness and stress
cargo test -p bitnet-kernels --test tl_packed_correctness -- --test-threads=1
cargo test -p bitnet-kernels --test fuzz_tl_lut_stress -- --test-threads=1

# Pinned baseline verification (strict + deterministic)
BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 RAYON_NUM_THREADS=1 \
cargo run -p xtask --no-default-features --features inference -- \
  verify-receipt --path docs/baselines/20251015-cpu.json
```

### Using Justfile (Recommended)
```bash
just mvp-pretag      # Run all pre-tag validation checks
just gate-smoke      # Verify bad receipts fail as expected
just test-tl-stress  # Run TL correctness + stress tests
```

## Breaking Changes
None. All changes are additive or improve stability/governance.

## Security Considerations
- Receipts are fully deterministic and offline-verifiable
- Strict mode prevents FP32 fallback in production
- No network dependencies during validation
- Kernel ID hygiene prevents injection attacks

## Related Issues
- Closes #465 (CPU MVP finalization)
- Relates to #261 (mock elimination)

## Next Steps (Post-Merge)
1. Enable branch protection for CPU receipt gate
2. Create smoke test PR to verify gate rejects bad receipts
3. Tag `v0.1.0-mvp` with release notes and binaries
