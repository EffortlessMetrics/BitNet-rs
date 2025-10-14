# Next Roadmap Issues for BitNet.rs Receipt Infrastructure

These issues should be created after PR #452 is merged.

---

## Issue: Enforce quantized hot-path (no FP32 staging)

**Title:** Enforce quantized projections in attention mechanism (no FP32 staging)

**Labels:** `enhancement`, `quantization`, `validation`

**Description:**

### Summary
Add runtime assertions to ensure attention mechanisms use quantized projections without unnecessary FP32 staging, preventing silent performance degradation.

### Problem
Currently, the attention mechanism could theoretically fall back to FP32 staging without detection, defeating the purpose of quantization and causing silent performance regression.

### Acceptance Criteria
- [ ] Add `debug_assert!(self.qkv_proj.is_quantized(), "Attention must use quantized projections")` in attention forward pass
- [ ] Confirm `QuantizedLinear::forward(..)` dispatches to `i2s/tl1/tl2` kernels
- [ ] Verify output is FP32 without full-weight dequant staging
- [ ] Add tests for attention Q/K/V/O via quantized linears
- [ ] Remove remaining placeholder `#[ignore]` on TL1/TL2 tests when tables are wired

### Implementation Notes
- Should be debug assertions (no runtime overhead in release builds)
- Focus on `bitnet-inference/src/attention.rs`
- Update corresponding tests in `tests/attention_quantized.rs`

### Related
- Depends on: PR #452 (receipt verification)
- Blocks: Full quantization validation

---

## Issue: CPU microbench + receipt

**Title:** Add deterministic CPU microbenchmark with receipt generation

**Labels:** `enhancement`, `benchmark`, `ci`, `receipt`

**Priority:** High

**Description:**

### Summary
Implement a tiny, deterministic CPU benchmark (~128 tokens) that writes `ci/inference.json` receipt with real compute evidence.

### Problem
Currently, the `verify-receipt` gate exists but has no benchmark to generate receipts. CI cannot validate real compute paths without actual inference receipts.

### Acceptance Criteria
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

### Implementation Notes
- Keep benchmark fast (<5s on CI hardware)
- Use existing `bitnet-inference` streaming API
- Model should be committed to repo or auto-downloaded
- Receipt schema version must match `RECEIPT_SCHEMA` in `bitnet-inference`

### CI Integration
```yaml
- name: Run CPU microbench
  run: cargo run -p xtask -- benchmark --tokens 128 --deterministic

- name: Verify receipt
  run: cargo run -p xtask -- verify-receipt --path ci/inference.json
```

### Related
- Depends on: PR #452 (receipt verification)
- Related: Issue #3 (performance benchmarking infrastructure)

---

## Issue: GPU microbench (skip-clean if no CUDA)

**Title:** Add GPU microbenchmark with graceful skip on non-CUDA hosts

**Labels:** `enhancement`, `benchmark`, `gpu`, `ci`, `receipt`

**Priority:** Medium

**Description:**

### Summary
Implement GPU benchmark that generates receipts with GPU kernel evidence and cleanly skips on hosts without CUDA.

### Problem
GPU CI lanes need receipt verification, but non-GPU hosts should skip gracefully without failing the build.

### Acceptance Criteria
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

### Implementation Notes
- Use `bitnet_kernels::device_features::gpu_available_runtime()` for detection
- Don't require `BITNET_GPU_FAKE` override (production detection)
- Ensure deterministic seed works on GPU too

### Related
- Depends on: CPU microbench (previous issue)
- Related: PR #439 (GPU feature flag unification)

---

## Issue: Cross-validation harness (opt-in)

**Title:** Implement opt-in cross-validation harness with C++ reference

**Labels:** `enhancement`, `testing`, `cross-validation`

**Priority:** Medium

**Description:**

### Summary
Create feature-gated cross-validation tests that compare BitNet.rs outputs with C++ reference implementation.

### Problem
Currently, cross-validation is manual and not integrated into the test suite. Need automated, opt-in tests that validate inference accuracy.

### Acceptance Criteria
- [ ] Feature-gated behind `#[cfg(feature = "crossval")]`
- [ ] Tests require `BITNET_GGUF` environment variable set to valid model path
- [ ] If `BITNET_GGUF` not set: tests print "SKIP: BITNET_GGUF not set" and pass
- [ ] If model present:
  - Run same prompts through Rust and C++ implementations
  - Assert correlation ≥ 0.995
  - Assert element-wise error bounds (e.g., max abs error < 0.01)
  - Compare token generation (first 10 tokens should match)
- [ ] Add to CI as optional job (only runs when model fixture available)
- [ ] Document in `docs/development/validation-framework.md`

### Implementation Notes
- Use existing `crossval` crate as foundation
- Tests in `tests/crossval/*.rs`
- Don't block CI on this - make it opt-in for contributors with models

### Test Example
```rust
#[test]
#[cfg(feature = "crossval")]
fn test_inference_matches_cpp_reference() {
    let model_path = std::env::var("BITNET_GGUF")
        .expect("BITNET_GGUF must be set for cross-validation");

    let rust_output = run_rust_inference(&model_path, "Hello world");
    let cpp_output = run_cpp_reference(&model_path, "Hello world");

    let correlation = compute_correlation(&rust_output, &cpp_output);
    assert!(correlation >= 0.995, "Correlation too low: {}", correlation);
}
```

### Related
- Depends on: PR #452 (receipt verification)
- Related: `docs/development/cross-validation-setup.md`

---

## Issue: Fingerprint exceptions for fast GPUs

**Title:** Add receipt fingerprinting to prevent false positives on fast GPUs

**Labels:** `enhancement`, `receipt`, `gpu`

**Priority:** Low

**Description:**

### Summary
Extend receipt schema with hardware fingerprints to allow fast GPUs without triggering false positives in duration-based validation.

### Problem
Very fast GPUs (e.g., A100, H100) might process 128 tokens so quickly that duration checks could flag them as suspicious. Need fingerprinting to allowlist known-good hardware.

### Acceptance Criteria
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

### Implementation Notes
- This is future-proofing - not urgent for CPU MVP
- Consider adding this when GPU benchmarks are stable
- Could also fingerprint for reproducibility (e.g., CI environment tracking)

### Related
- Depends on: GPU microbench issue
- Related: Receipt schema v1.0 (PR #452)

---

## Issue: Validation shared crate

**Title:** Refactor validation rules into shared `bitnet-validation` crate

**Labels:** `refactor`, `validation`, `architecture`

**Priority:** Medium

**Description:**

### Summary
Migrate validation rules and helpers duplicated between CLI inspector and st-tools into a shared `bitnet-validation` crate to prevent policy drift.

### Problem
Currently, LayerNorm validation, projection checks, and correction policies are duplicated across:
- `bitnet-cli/src/inspect.rs`
- `bitnet-st-tools/src/validation/*.rs`
- Some logic in `bitnet-models`

This creates maintenance burden and risks policy drift.

### Acceptance Criteria
- [ ] Create new crate: `crates/bitnet-validation/`
- [ ] Migrate shared validation logic:
  - LayerNorm RMS validation with architecture-aware envelopes
  - Projection weight validation
  - Correction policy parsing and application
  - Strict mode handling
- [ ] Consumers:
  - `bitnet-cli` uses `bitnet-validation` for `inspect` command
  - `bitnet-st-tools` uses for GGUF validation
  - `bitnet-models` can optionally use for load-time validation
- [ ] No duplicated validation code
- [ ] Update documentation to reference single validation source

### API Design
```rust
// Proposed API
pub struct ValidationConfig {
    pub mode: ValidationMode,
    pub strict: bool,
    pub policy_path: Option<PathBuf>,
}

pub fn validate_layernorm_weights(
    weights: &[f32],
    config: &ValidationConfig,
) -> Result<ValidationReport>;

pub fn validate_projection_weights(
    weights: &[f32],
    config: &ValidationConfig,
) -> Result<ValidationReport>;
```

### Implementation Notes
- Keep `bitnet-validation` dependency-light (only `anyhow`, `serde`, `regex`)
- Use builder pattern for configs
- Consider making correction policy application explicit (not automatic)

### Related
- Related: PR #448 (validation MVP)
- Related: `docs/howto/validate-models.md`

---

## Summary

**Create these issues in order:**
1. **Enforce quantized hot-path** - Quick win, adds safety
2. **CPU microbench** - Unblocks receipt verification in CI
3. **GPU microbench** - Completes receipt infrastructure
4. **Cross-validation harness** - Improves accuracy confidence
5. **Fingerprint exceptions** - Future-proofing for fast hardware
6. **Validation shared crate** - Code quality improvement

**Estimated effort:**
- Issues 1-2: ~2 days (high priority)
- Issues 3-4: ~3 days (medium priority)
- Issues 5-6: ~2 days (low priority, can defer)

All issues depend on PR #452 being merged first.
