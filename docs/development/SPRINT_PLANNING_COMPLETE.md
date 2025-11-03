# QK256 AVX2 Optimization Sprint - Planning Complete

**Date:** 2025-10-22

**Status:** ✅ Sprint Planning Complete - Ready for Implementation

---

## Documents Created

1. **[qk256-avx2-optimization-sprint.md](./qk256-avx2-optimization-sprint.md)** (942 lines, 27KB)
   - Comprehensive sprint plan with detailed technical analysis
   - 4-phase optimization strategy with intrinsics
   - Performance projections and risk mitigation
   - Benchmark methodology and validation strategy

2. **[qk256-avx2-sprint-summary.md](./qk256-avx2-sprint-summary.md)** (3.8KB)
   - Quick reference guide for developers
   - Key commands and performance targets
   - Sprint timeline and success criteria

3. **[qk256-avx2-sprint-issue-template.md](./qk256-avx2-sprint-issue-template.md)** (7.4KB)
   - GitHub issue tracking template
   - Phase-by-phase task breakdown
   - Acceptance criteria and milestone tracking

---

## Performance Analysis

### Current State (Baseline Metrics)

```
Test Size: 1024 elements (4 QK256 blocks)
-------------------------------------------
Scalar:   697 ns (1.47 Gelem/s)
AVX2:     577 ns (1.78 Gelem/s)
Speedup:  1.21× (vs. target ≥3×)
```

### Target Performance

```
Time:       ≤233 ns (3× speedup)
Throughput: ≥4.4 Gelem/s
Real-world: ~0.1 → ~0.3 tok/s for 2B models
```

---

## Optimization Strategy (4 Phases)

| Phase | Optimization | Target Speedup | Key Intrinsic |
|-------|-------------|----------------|---------------|
| 1 | Nibble LUT Unpack | 4× (unpacking) | `_mm256_shuffle_epi8` |
| 2 | FMA Tiling (8×16) | 2.5× (compute) | `_mm256_fmadd_ps` |
| 3 | Prefetch | 1.7× (memory) | `_mm_prefetch` |
| 4 | SIMD LUT | 3× (lookup) | `_mm256_permutevar8x32_ps` |

**Combined:** 2.96× speedup (195 ns) → Achieves ≥3× target ✅

---

## Sprint Timeline (2 Weeks)

### Week 1: Core Optimizations
- Days 1-2: Phase 1 (Nibble LUT)
- Days 3-4: Phase 4 (SIMD LUT)
- Day 5: Integration (target: ≥2× speedup)

### Week 2: Advanced Optimizations
- Days 6-8: Phase 2 (FMA Tiling)
- Days 9-10: Phase 3 (Prefetch)
- Days 11-12: Integration & profiling
- Days 13-14: Validation & docs (target: ≥3× speedup)

---

## Key Implementation Files

| File | Purpose | Lines |
|------|---------|-------|
| `crates/bitnet-kernels/src/cpu/x86.rs` | Dequantize kernel | ~1216 |
| `crates/bitnet-models/src/quant/i2s_qk256_avx2.rs` | GEMV kernel | ~200 |
| `crates/bitnet-models/tests/qk256_avx2_correctness.rs` | Validation | ~597 |
| `crates/bitnet-kernels/benches/kernel_benchmarks.rs` | Benchmarks | ~444 |

---

## Validation Strategy

### Unit Tests (Per Phase)
- Phase 1: Unpacking correctness (100+ random seeds)
- Phase 2: Multi-row GEMV (8-64 rows)
- Phase 3: Prefetch no-regression (small sizes)
- Phase 4: SIMD LUT (all 4 codes)

### Integration Tests
- 12 tests in `qk256_avx2_correctness.rs` (all passing)
- Tolerance: ≤1e-4 max absolute error
- Edge cases: partial blocks, non-aligned columns

### Benchmarks
- Baseline comparison: `--baseline pre-optimization`
- Target: ≥3× speedup for 1024 elements
- Scalability: ≥2.5× for 256-4096 elements

---

## Success Criteria

✅ **Primary Goals**
- Speedup ≥3× for 1024 elements (≤233 ns)
- All 12 correctness tests passing (tolerance 1e-4)
- No performance regression on small sizes

✅ **Secondary Goals**
- Speedup ≥2.5× for 256-4096 elements
- End-to-end inference: ~0.1 → ~0.3 tok/s
- Documentation updated (CLAUDE.md, benchmark guide)

---

## Next Steps

1. **Create GitHub Issue** using `qk256-avx2-sprint-issue-template.md`
2. **Assign Sprint Team** (developers, reviewers)
3. **Set Sprint Dates** (2-week timeline)
4. **Run Baseline Benchmarks** and save with `--save-baseline`
5. **Begin Phase 1** (Nibble LUT Unpack)

---

## References

- Full Plan: `docs/development/qk256-avx2-optimization-sprint.md`
- Quick Reference: `docs/development/qk256-avx2-sprint-summary.md`
- Issue Template: `docs/development/qk256-avx2-sprint-issue-template.md`
- Benchmark Guide: `docs/benchmarks/qk256-dequant-benchmark.md`
- CLAUDE.md: QK256 AVX2 Fast Path section

---

**Sprint Owner:** Development Team

**Status:** Ready for Implementation

**Target Completion:** [Sprint Start + 2 weeks]

**Performance Goal:** ≥3× speedup (1.21× → 3.0×) ✅
