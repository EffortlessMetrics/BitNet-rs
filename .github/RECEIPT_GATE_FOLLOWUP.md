# Receipt Verification Gate - Follow-up Issues

This document outlines the follow-up work needed to complete the CPU MVP with fully enforceable receipt verification gates.

## Status

✅ **Implemented** (Current PR):
- PR #452: KernelRecorder infrastructure integrated into bitnet-inference
- PR #452: InferenceEngine records kernel execution via optional KernelRecorder
- PR #452: `xtask benchmark` writes production receipts with measured TPS and real kernel IDs
- PR #452: `xtask verify-receipt` validates receipt schema and honest compute gates
- PR #452: CI workflow enforces receipt verification via `.github/workflows/model-gates.yml`
- PR #452: Documentation updated (CLAUDE.md, docs/explanation/receipt-validation.md)
- ✅ Issue #1: Replace write-receipt stub with real benchmark (COMPLETE)

⏳ **Pending** (Follow-up issues):

---

## Issue 1: Replace write-receipt stub with real benchmark ✅ COMPLETE

**Title**: `[Tooling] Enhance benchmark command to write receipts`

**Status**: ✅ **IMPLEMENTED** in PR #452

**Description**:
Replace the `write-receipt` stub with real measurement by enhancing the existing `xtask benchmark` command to write receipts in the verified format.

**Implementation Summary**:
- ✅ KernelRecorder infrastructure added to `bitnet-inference`
- ✅ InferenceEngine records kernel IDs via optional KernelRecorder
- ✅ `xtask benchmark` writes production receipts with measured TPS and real kernel IDs
- ✅ Receipt schema v1.0.0 with all required fields
- ✅ CI workflow updated to use `benchmark` instead of `write-receipt`
- ✅ `write-receipt` stub command removed

**Files Modified**:
- `crates/bitnet-inference/src/kernel_recorder.rs` (NEW)
- `crates/bitnet-inference/src/engine.rs`
- `xtask/src/main.rs`
- `.github/workflows/model-gates.yml`
- `CLAUDE.md`, `docs/explanation/receipt-validation.md`

---

## Issue 2: Enforce quantized hot-path (no FP32 staging)

**Title**: `[Validation] Enforce quantized inference path with strict guards`

**Description**:
Add runtime guards to prevent silent fallback to FP32 staging in quantized layers and attention projections.

**Acceptance Criteria**:
- Debug asserts in `QuantizedLinear::forward` and attention Q/K/V/O projections
- Strict mode (`BITNET_STRICT_MODE=1`) returns `Err` if fallback would occur
- Unit test that simulates fallback path panics in debug mode
- Integration test that runs 16-token decode in strict mode and succeeds
- Receipts cannot claim quantized compute without actual quantized kernels

**Why**:
Receipts prove performance, but not correctness. This ensures the math matches the claims.

**Files to Create/Modify**:
- `crates/bitnet-inference/src/.../quantized_linear.rs`
- `crates/bitnet-inference/src/.../attention.rs`
- `crates/bitnet-inference/tests/strict_quantization_test.rs`

---

## Issue 3: Branch protection - require CPU Receipt Gate

**Title**: `[CI] Make CPU Receipt Gate required on main branch`

**Description**:
Update GitHub branch protection rules to require the "CPU Receipt Gate" job to pass before merging to `main`.

**Acceptance Criteria**:
- "CPU Receipt Gate" job marked as **required** in branch protection
- PRs cannot merge without passing receipt verification
- Path filters ensure gate runs only when relevant code changes (crates/bitnet-*, xtask/, .github/workflows/model-gates.yml)

**Why**:
Enforceable quality bar - no PR merges without honest compute receipts.

**Configuration**:
- GitHub repo settings → Branches → main → Branch protection rules
- Add "CPU Receipt Gate" to required status checks

---

## Issue 4: GPU receipt verification (skip-clean)

**Title**: `[CI] Add GPU receipt gate with skip-clean fallback`

**Description**:
Add GPU receipt verification that runs on CUDA-capable runners and cleanly skips on CPU-only runners.

**Acceptance Criteria**:
- `xtask benchmark --backend cuda` writes GPU receipt to `ci/inference-gpu.json`
- Receipt contains GPU-specific metadata (CUDA version, GPU model, compute capability)
- CI job runs on GPU runners and verifies with `--require-gpu-kernels`
- CI job exits 0 with clear "skipped (no CUDA)" message on CPU-only runners
- Job remains green when skipped (doesn't block PR merge)

**Why**:
Complete the GPU verification lane without breaking non-CUDA hosts.

**Files**:
- `.github/workflows/model-gates.yml`: Add `gpu-receipt-gate` job
- `xtask/src/main.rs`: Add `--backend cuda` support to benchmark

---

## Issue 5: Cross-validation harness (opt-in)

**Title**: `[Validation] Add cross-validation test harness for correctness`

**Description**:
Add opt-in cross-validation tests that compare BitNet.rs output against the C++ reference implementation.

**Acceptance Criteria**:
- Feature gate `crossval` for tests
- Test reads `BITNET_GGUF` environment variable (defaults to `tests/models/tiny.gguf`)
- Runs short forward pass and compares against C++ reference or pre-baked baseline
- Asserts correlation ≥ 0.995 or error within acceptable bounds
- Skips cleanly if feature/env not set (doesn't break default test runs)
- CI runs cross-validation when `BITNET_GGUF` is set

**Why**:
Sanity beyond performance - ensures mathematical correctness and parity with reference.

**Files**:
- `crates/bitnet-inference/tests/crossval_harness.rs`
- `.github/workflows/model-gates.yml`: Add optional crossval step

---

## Issue 6: Fingerprint exceptions for fast GPUs

**Title**: `[Validation] Add allowlist for fast GPU fingerprints`

**Description**:
Prevent false positives on high-end hardware by allowing specific fingerprints to bypass strict throughput checks.

**Acceptance Criteria**:
- Receipt enriched with fingerprint metadata:
  - `gpu_compute_capability` (e.g., "8.9")
  - `cudart_version` (e.g., "12.4")
  - `cpu_brand` (e.g., "AMD Ryzen 9 7950X")
  - `os` (e.g., "linux-x86_64")
  - `rustc_version` (e.g., "1.90.0")
- `verify-receipt` supports `--allowlist .ci/receipts-allow.yml` flag
- Allowlist file keys by fingerprint combination
- Documentation on how to add exceptions (for dev machines with high-end GPUs)

**Why**:
Avoid false positives on fast hardware without loosening global policy.

**Files**:
- `xtask/src/main.rs`: Enhance receipt generation and verification
- `.ci/receipts-allow.yml`: Allowlist examples
- `docs/howto/receipt-verification.md`: Documentation

---

## Issue 7: Extract validation logic to shared crate

**Title**: `[Refactor] Create bitnet-validation crate for shared rules`

**Description**:
Factor out LayerNorm/projection validation logic into a shared crate to prevent drift between CLI and st-tools.

**Acceptance Criteria**:
- New crate `bitnet-validation` (or extend `bitnet-models`) with:
  - LN/projection validation rules
  - Regex patterns for kernel detection
  - Policy YAML loader
  - RMS calculation helpers
- `bitnet-cli` and `bitnet-st-tools` refactored to use shared crate
- Both tools produce identical validation decisions for same inputs
- No code duplication for validation logic

**Why**:
Single source of truth for validation rules prevents drift and ensures consistency.

**Files**:
- `crates/bitnet-validation/` (new crate)
- `crates/bitnet-cli/src/commands/inspect.rs`: Refactor to use bitnet-validation
- `crates/bitnet-st-tools/src/ln_inspect.rs`: Refactor to use bitnet-validation

---

## Issue 8: Docs sweep - receipts over prose

**Title**: `[Docs] Replace performance claims with receipt-driven examples`

**Description**:
Replace all performance claims in documentation with reproducible receipt generation examples.

**Acceptance Criteria**:
- No legacy "200 tok/s CPU" or similar claims in docs
- All performance examples show:
  ```bash
  cargo run -p xtask -- benchmark --model <path> --tokens 128
  cargo run -p xtask -- verify-receipt
  ```
- Feature flag usage standardized to `--no-default-features --features cpu|gpu`
- Examples are executable and produce verifiable receipts
- Typical ranges documented as "receipt-driven envelopes" (CPU 10-25 tok/s, GPU 50-100 tok/s)

**Why**:
Replace claims with proof - teach users to produce and verify honest receipts.

**Files**:
- `README.md`
- `docs/quickstart.md`
- `docs/performance-benchmarking.md`
- `docs/howto/*.md`

---

## Issue 9: Tag and release v0.1.0-mvp-validation

**Title**: `[Release] Tag v0.1.0-mvp-validation with receipt gate`

**Description**:
Create release with receipt verification gate and attach validation tool binaries.

**Acceptance Criteria**:
- Git tag: `v0.1.0-mvp-validation`
- Release notes include:
  - "Mock inference eliminated; receipts required"
  - How to produce receipts: `xtask write-receipt` (or `benchmark` after Issue #1)
  - How to verify: `xtask verify-receipt`
  - Typical performance envelopes with receipt-driven caveat
- Attached binaries:
  - `bitnet` (CLI)
  - `st2gguf`
  - `st-ln-inspect`
  - `st-merge-ln-f16`
- README/CLAUDE.md/CONTRIBUTING point to receipt workflow

**Why**:
Give the team a pinned, repeatable chassis with honest compute verification.

**Deliverables**:
- GitHub release with binaries
- Release notes
- Updated documentation

---

## Priority Order (Recommended)

1. **Issue #1**: Replace write-receipt stub → unblocks production receipts
2. **Issue #3**: Branch protection → makes gate enforceable
3. **Issue #2**: Quantized hot-path enforcement → proves math correctness
4. **Issue #4**: GPU receipts (skip-clean) → completes GPU lane
5. **Issue #5**: Cross-validation → mathematical sanity checks
6. **Issue #7**: Shared validation crate → prevents drift
7. **Issue #8**: Docs sweep → teach receipt-driven workflows
8. **Issue #6**: Fingerprint exceptions → handle fast hardware
9. **Issue #9**: Release v0.1.0-mvp-validation → ship it!

---

## Summary

The current PR establishes the foundation for receipt verification with:
- ✅ Verifier that enforces schema, compute_path, and kernel evidence
- ✅ Stub producer that writes valid receipts
- ✅ CI workflow that runs and verifies receipts
- ✅ Local gates script that mirrors CI behavior

The follow-up issues complete the CPU MVP by:
1. Replacing stub with real measurements
2. Enforcing quantized hot-path (no silent fallbacks)
3. Making the gate enforceable via branch protection
4. Extending to GPU verification
5. Adding cross-validation for correctness
6. Refactoring to prevent validation drift
7. Documenting receipt-driven workflows
8. Shipping a release with honest compute receipts

After these issues are complete, every PR to `main` will prove honest compute with verifiable receipts.
