# Issue #459 Research & Implementation Plan
## Replace Performance Claims with Receipt-Driven Examples

**Date**: 2025-11-11
**Status**: Ready for implementation
**Blocking Items**: NONE (PR #475 and PR #452 both resolved)

---

## Executive Summary

I've completed a comprehensive analysis of issue #459 focusing on:
1. **Current performance claims** in documentation (7 files identified)
2. **Receipt verification system** implementation status (PR #452 - 25/25 tests passing)
3. **Feature flag standardization** requirements (unified in PR #475)
4. **Receipt envelope definitions** for CPU/GPU/QK256 validation
5. **Reproducible implementation plan** with phase breakdown

**Key Finding**: The receipt infrastructure is production-ready (schema v1.0.0, all validation gates operational), but documentation still contains bare performance claims without linking to verifiable receipts. This issue addresses credibility by replacing all claims with receipt-backed examples.

---

## Current Performance Claims Analysis

### Performance Claims Located

| File | Lines | Claim | Status |
|------|-------|-------|--------|
| README.md | 220-222 | CPU: ~10-20 tok/s, GPU: ~50-100 tok/s | Unverified |
| README.md | 513-514 | Same claims, repeated | Unverified |
| docs/quickstart.md | 105-109 | Performance table with quantization formats | Unverified |
| docs/quickstart.md | 147-150 | CPU/GPU expectations | Unverified |
| docs/performance-benchmarking.md | 459-490+ | Performance baselines table | **Backed by schema explanation** |
| docs/troubleshooting/slow-inference.md | Multiple | Throughput comparisons across formats | Unverified |
| CLAUDE.md | Quick reference section | Performance expectations | Unverified |

### Receipt Infrastructure Status

**Receipt Schema v1.0.0** (`bitnet-inference/receipts.rs`):
- **Schema Version**: 1.0.0 (stable)
- **Validation Gates**: 8 gates (all 25/25 tests passing)
- **Verification Command**: `cargo run -p xtask -- verify-receipt ci/inference.json`
- **Current Artifacts**: `ci/inference.json`, `docs/baselines/*.json`

**Receipt Validation Gates** (all enabled):
1. Schema version must be "1.0.0" ✅
2. `compute_path` must be "real" (not "mock") ✅
3. `kernels` array must be non-empty ✅
4. Kernel IDs must be valid (non-empty, ≤128 chars, ≤10K count) ✅
5. No "mock_*" kernel IDs present ✅
6. Environment must include BITNET_VERSION, OS ✅
7. Timestamp must be valid ISO 8601 ✅
8. GPU kernels required if backend=="cuda" ✅

**Current Receipt Example** (from `ci/inference.json`):
```json
{
  "schema_version": "1.0.0",
  "compute_path": "real",
  "backend": "cpu",
  "deterministic": true,
  "tokens_per_second": 15.3,
  "kernels": ["i2s_gemv", "rope_apply", "attention_real", ...],
  "timestamp": "2025-10-23T00:39:09Z"
}
```

---

## Receipt-Driven Envelopes (Validated Performance Ranges)

### CPU Envelope (I2S BitNet32-F16)

**Expected Range**: 10-20 tok/s

**Validation Criteria**:
```
- compute_path == "real" (not "mock")
- backend == "cpu"
- kernels contains: "i2s_gemv", "attention_real" (no "mock_*")
- deterministic == true
- No GPU kernels present
```

**Acceptable Variance**: ±15% (8.5-23 tok/s due to hardware variance)

**Receipt Generation Command**:
```bash
BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo run -p xtask -- benchmark --model <model.gguf> --tokens 128

# Verify receipt
cat ci/inference.json | jq 'select(.compute_path == "real" and .backend == "cpu") | .tokens_per_second'
```

### GPU Envelope (CUDA with Mixed Precision)

**Expected Range**: 50-100 tok/s

**Validation Criteria**:
```
- compute_path == "real"
- backend == "cuda"
- kernels contains GPU kernel IDs: "gemm_*", "attention_cuda", etc.
- deterministic == false (GPU scheduling variance allowed)
- Memory constraint: ~500MB for 2B model
```

**Acceptable Variance**: ±25% (37.5-125 tok/s due to GPU scheduling)

**Receipt Generation Command**:
```bash
BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=0 \
cargo run -p xtask -- benchmark --model <model.gguf> --tokens 128 --features gpu

# Verify receipt
cat ci/inference.json | jq 'select(.compute_path == "real" and .backend == "cuda") | .tokens_per_second'
```

### QK256 MVP Envelope (Scalar Kernels Only)

**Expected Range**: ~0.1 tok/s

**Validation Criteria**:
```
- compute_path == "real"
- kernels contains "qk256_scalar_*" (no SIMD optimization)
- STATUS: ⚠️ Validation only - NOT production-ready
- Limitation: SIMD optimizations planned for v0.2.0
```

**Important**: This should NOT be used for production. Use I2S BitNet32-F16 instead.

**Receipt Command** (use `--max-tokens 4-8` only):
```bash
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo run -p xtask -- benchmark --model <model.gguf> --tokens 4

# Expected ~40 seconds for 4 tokens (~0.1 tok/s)
```

---

## Implementation Plan - 6 Phases

### Phase 1: Documentation Audit & Mapping

**Files to Update** (priority order):
1. `/README.md` - Lines 220-222, 513-514 (main entry point)
2. `/docs/quickstart.md` - Lines 105-109, 147-150 (getting started)
3. `/docs/performance-benchmarking.md` - Enhance existing receipt section
4. `/docs/troubleshooting/slow-inference.md` - Add receipt examples
5. `/CLAUDE.md` - Quick reference section
6. `/docs/howto/*.md` - Performance context sections

**For Each File**:
- Remove bare performance claims without verification
- Add receipt generation command with output example
- Link to relevant baseline artifacts in `docs/baselines/`
- Include feature flag standardization

### Phase 2: Receipt-Driven Example Templates

**Template 1: Quick Validation** (4-token run):
```bash
# Generate receipt for validation
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
cargo run -p xtask -- benchmark --model <model> --tokens 4

# View tokens per second from receipt
cat ci/inference.json | jq '.tokens_per_second'
```

**Template 2: Production Benchmark** (128-token run):
```bash
# CPU performance baseline with strict validation
BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo run -p xtask -- benchmark --model <model> --tokens 128

# Verify receipt shows real computation
cat ci/inference.json | jq '{compute_path, backend, tokens_per_second, kernels}'
```

**Template 3: Dated Baseline** (archive performance):
```bash
# Generate and store dated baseline
mkdir -p docs/baselines/$(date +%Y-%m-%d)
BITNET_STRICT_MODE=1 cargo run -p xtask -- benchmark \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokens 128 \
  --output docs/baselines/$(date +%Y-%m-%d)/cpu-baseline-i2s.json

# Verify receipt
cat docs/baselines/$(date +%Y-%m-%d)/cpu-baseline-i2s.json | jq '.tokens_per_second'
```

### Phase 3: Feature Flag Standardization

**Requirement**: All examples use explicit feature flags (PR #475 unified these):

```bash
# CPU inference (always explicit)
cargo build --no-default-features --features cpu
cargo run -p xtask -- benchmark --model <model> --tokens N

# GPU inference (requires CUDA)
cargo build --no-default-features --features gpu
cargo run -p xtask -- benchmark --model <model> --tokens N

# With native CPU optimization
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu
```

**Standardization Checklist**:
- [ ] All performance examples use: `--no-default-features --features cpu|gpu`
- [ ] No examples rely on default (empty) features
- [ ] GPU examples clearly marked as requiring CUDA
- [ ] Deterministic runs: `BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1`
- [ ] Strict validation: `BITNET_STRICT_MODE=1` for honest compute

### Phase 4: Documentation Replacement Strategy

**README.md Changes**:

Replace bare claims with receipt examples and reference to performance-benchmarking.md.

**docs/quickstart.md Changes**:

Add receipt generation step and link to baseline artifacts.

**docs/performance-benchmarking.md Changes** (no removal needed):

Already contains receipt schema and strict mode section. Enhance with comprehensive envelope documentation.

### Phase 5: Receipt Envelope Documentation

Create comprehensive envelope documentation with validation criteria for CPU, GPU, and QK256.

### Phase 6: PR Validation Checklist

Before submitting PR, verify:
- All documentation audit complete
- Receipt examples tested and working
- Feature flags consistent across all examples
- Environment variables standardized
- Documentation links correct
- Examples reproducible

---

## Files Requiring Changes

| File | Lines | Change Type | Scope |
|------|-------|-------------|-------|
| README.md | 220-222, 513-514 | Replace bare claims with receipt examples | High |
| docs/quickstart.md | 105-109, 147-150 | Add receipt generation commands | High |
| docs/performance-benchmarking.md | 459-490+ | Already has receipt info, enhance section | Medium |
| docs/troubleshooting/slow-inference.md | Multiple | Add receipt verification examples | Medium |
| CLAUDE.md | Quick reference | Add receipt generation pattern | Low |
| docs/howto/*.md | Context sections | Add receipt linking where appropriate | Low |

---

## Feature Flag Standardization Requirements

### Current State (Post-PR #475)
- GPU feature: `gpu` (preferred) or `cuda` (backward-compat)
- CPU feature: `cpu`
- Default features: **EMPTY** (must be explicit)

### Documentation Requirements
- All examples use `--no-default-features --features cpu|gpu`
- No bare `cargo build` or `cargo run`
- Environment variables standardized:
  - Deterministic: `BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1`
  - Strict: `BITNET_STRICT_MODE=1`
  - Optimization: `RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin"`

---

## Success Criteria

1. **100% Receipt Backing**: Every performance claim backed by verifiable receipt artifact
2. **Reproducibility**: All documented commands copy-paste executable with predictable output
3. **Feature Consistency**: All examples use standardized feature flag pattern (PR #475 compliance)
4. **CI Integration**: Receipt verification gates prevent regression of undocumented claims
5. **Maintenance**: New performance claims require dated baseline receipts (`docs/baselines/<YYYY-MM-DD>/`)

---

## Dependencies & Timeline

### Blocking Items (RESOLVED)
- PR #475 (GPU/CPU feature unification) - ✅ MERGED
- PR #452 (Receipt verification schema v1.0.0) - ✅ MERGED

### Recommended Dependencies (Should Stabilize First)
- Issue #417 (QK256 SIMD optimization) - For accurate updated envelopes
- Issue #346/#401 (TL1/TL2 production impl) - For TL quantization performance
- Issue #379 (Top-K sampling optimization) - For sampling throughput impact

### Recommendation
**Create implementation now, execute after #417-#401 stabilize** to ensure receipt-driven examples reflect real performance improvements (≥3× for QK256, proper TL1/TL2 kernels).

---

## Code Reference Locations

**Receipt Infrastructure**:
- Schema: `/crates/bitnet-inference/src/receipts.rs` (Receipt struct v1.0.0)
- Verification: `/xtask/src/main.rs` (verify_receipt_cmd function)
- Tests: `/crates/bitnet-inference/tests/*receipt*` (25/25 passing)

**Baseline Artifacts**:
- Location: `/docs/baselines/`
- Format: JSON with performance metrics and parity validation
- Dated: `YYYY-MM-DD/` directory structure

**Feature Flags**:
- Unified predicates: `#[cfg(any(feature = "gpu", feature = "cuda"))]`
- See: `/CLAUDE.md` line 51-60 for unified pattern

---

## Conclusion

The receipt infrastructure is production-ready with all validation gates operational. This issue addresses documentation credibility by replacing bare performance claims with verifiable receipt-backed examples. The implementation can proceed immediately with the proposed 6-phase approach, taking advantage of already-landed infrastructure (PR #452, #475).

**Next Steps for Implementation**:
1. Review and refine this implementation plan
2. Create PR with Phase 1-3 changes (documentation audit + examples + features)
3. Execute Phase 4-6 in follow-up PR after performance optimizations (#417, #346, #401) stabilize
4. Add CI gates to verify all performance examples pass receipt validation

---

*Research completed: 2025-11-11*
*Issue #459 Status: Ready for implementation (Phase 1 can start immediately)*
