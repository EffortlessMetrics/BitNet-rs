# QK256 AVX2 Optimization Sprint - Quick Reference

**Full Plan:** [qk256-avx2-optimization-sprint.md](./qk256-avx2-optimization-sprint.md)

---

## Quick Stats

| Metric | Current (MVP) | Target | Status |
|--------|---------------|--------|--------|
| **1024 elements** | 577 ns (1.78 Gelem/s) | ≤233 ns (≥4.4 Gelem/s) | 1.21× (need 3×) |
| **Real inference** | ~0.1 tok/s | ~0.3 tok/s | MVP phase |

---

## 4-Phase Optimization Plan

### Phase 1: Nibble LUT Unpack via `pshufb` (+80% speedup)
- **What:** Replace scalar 2-bit extraction with AVX2 shuffle
- **Intrinsic:** `_mm256_shuffle_epi8`
- **Impact:** 40% → 10% time in unpacking (4× faster)

### Phase 2: FMA Tiling (8-16 Rows) (+60% speedup)
- **What:** Unroll dot-product with 8 accumulators × 16 columns
- **Intrinsic:** `_mm256_fmadd_ps` with multiple accumulators
- **Impact:** 30% → 12% time in compute (2.5× faster)

### Phase 3: Load Combine & Prefetch (+30% speedup)
- **What:** Reduce memory latency with aligned loads + software prefetch
- **Intrinsic:** `_mm_prefetch` with `_MM_HINT_T0`
- **Impact:** 10% → 6% time in memory (1.7× faster)

### Phase 4: SIMD LUT via Permute (+40% speedup)
- **What:** Vectorize code→weight mapping with permute
- **Intrinsic:** `_mm256_permutevar8x32_ps`
- **Impact:** 30% → 10% time in LUT (3× faster)

---

## Sprint Timeline (2 Weeks)

### Week 1: Core Optimizations
- **Day 1-2:** Phase 1 (Nibble LUT)
- **Day 3-4:** Phase 4 (SIMD LUT)
- **Day 5:** Integration & baseline (target: ≥2× speedup)

### Week 2: Advanced Optimizations
- **Day 6-8:** Phase 2 (FMA Tiling)
- **Day 9-10:** Phase 3 (Prefetch)
- **Day 11-12:** Integration & profiling
- **Day 13-14:** Validation & docs (target: ≥3× speedup)

---

## Quick Commands

### Benchmark (before/after)
```bash
# Baseline
cargo bench --bench kernel_benchmarks --features cpu,avx2 -- qk256_dequant

# Save baseline
cargo bench --bench kernel_benchmarks --features cpu,avx2 \
  -- qk256_dequant --save-baseline pre-optimization

# Compare after optimization
cargo bench --bench kernel_benchmarks --features cpu,avx2 \
  -- qk256_dequant --baseline pre-optimization
```

### Validate Correctness
```bash
cargo test --release -p bitnet-models --test qk256_avx2_correctness
```

### Profile Bottlenecks
```bash
perf record -g -- cargo bench --bench kernel_benchmarks --features cpu,avx2 \
  -- qk256_dequant/avx2/1024 --quick
perf report
```

---

## Success Criteria

✅ **Primary:** ≤233 ns for 1024 elements (3× speedup)

✅ **Secondary:**
- All 12 correctness tests passing (tolerance 1e-4)
- Speedup ≥2.5× for 256-4096 elements
- End-to-end inference: ~0.1 → ~0.3 tok/s

---

## Key Files

| File | Purpose |
|------|---------|
| `crates/bitnet-kernels/src/cpu/x86.rs` | Dequantize kernel (main optimization target) |
| `crates/bitnet-models/src/quant/i2s_qk256_avx2.rs` | GEMV kernel with AVX2 |
| `crates/bitnet-models/tests/qk256_avx2_correctness.rs` | Validation tests (12 tests) |
| `crates/bitnet-kernels/benches/kernel_benchmarks.rs` | Benchmark suite |

---

## Performance Projection

| Phase | Time (ns) | Speedup | Cumulative |
|-------|-----------|---------|------------|
| Baseline (MVP) | 577 | 1.0× | 1.0× |
| After Phase 1 | 404 | 1.43× | 1.43× |
| After Phase 1+4 | 258 | 2.24× | 2.24× |
| After All Phases | **195** | **2.96×** | **2.96×** ✅ |
| Stretch Goal | 175 | 3.30× | 3.30× |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Numerical instability | Tolerance 1e-4, property tests |
| No AVX2 at runtime | Keep scalar fallback path |
| Perf regression on small sizes | Hybrid dispatch for < 128 elements |
| Compiler auto-vectorization | `#[inline(never)]` on scalar reference |

---

**Created:** 2025-10-22

**Status:** Ready for Implementation

**Next Step:** Phase 1 - Nibble LUT Unpack
