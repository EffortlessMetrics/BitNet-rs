# QK256 AVX2 Optimization Sprint - Issue Tracking Template

**Copy this to GitHub Issues for tracking**

---

## Title

QK256 AVX2 Optimization Sprint: Achieve ≥3× Performance Uplift

---

## Description

Comprehensive optimization sprint to improve QK256 dequantization performance from current 1.2× speedup to ≥3× through AVX2 SIMD optimizations.

### Current State
- Baseline (scalar): 697 ns for 1024 elements (1.47 Gelem/s)
- Current (AVX2 MVP): 577 ns for 1024 elements (1.78 Gelem/s)
- **Speedup: 1.21× (vs. target ≥3×)**

### Target Performance
- Time: ≤233 ns for 1024 elements (3× speedup)
- Throughput: ≥4.4 Gelem/s
- Real-world impact: Inference TPS uplift from ~0.1 tok/s → ~0.3 tok/s for 2B models

### Documentation
- **Full Plan:** `docs/development/qk256-avx2-optimization-sprint.md`
- **Quick Reference:** `docs/development/qk256-avx2-sprint-summary.md`

---

## Implementation Plan

### Phase 1: Nibble LUT Unpack via `pshufb` (Days 1-2)

**Goal:** Replace scalar 2-bit extraction with AVX2 shuffle-based unpacking

**Tasks:**
- [ ] Implement `unpack_qk256_block_avx2_v2` with shuffle instructions
- [ ] Write unit tests with random packed data (100+ iterations)
- [ ] Validate correctness against scalar reference (exact match)
- [ ] Benchmark unpacking throughput (target: 4× speedup)
- [ ] Update `crates/bitnet-kernels/src/cpu/x86.rs`

**Success Criteria:**
- Unpacking time: 231 ns → ≤58 ns
- All unit tests passing

**Key Intrinsics:**
```rust
_mm256_shuffle_epi8    // Nibble extraction
_mm256_and_si256       // Masking
_mm256_srli_epi16      // Shift-right for upper nibbles
```

---

### Phase 2: FMA Tiling (8-16 Rows) (Days 6-8)

**Goal:** Unroll dot-product loops with 8 accumulators × 16 columns per iteration

**Tasks:**
- [ ] Implement `gemv_qk256_tiled_avx2` with multi-accumulator design
- [ ] Unroll inner loop by 16 columns (2 AVX2 vectors)
- [ ] Add support for 8-row tile batching
- [ ] Validate multi-row GEMV correctness (tolerance 1e-4)
- [ ] Benchmark compute throughput (target: 2.5× speedup)
- [ ] Update `crates/bitnet-models/src/quant/i2s_qk256_avx2.rs`

**Success Criteria:**
- FMA time: 87 ns → ≤35 ns
- All correctness tests passing

**Key Intrinsics:**
```rust
_mm256_fmadd_ps        // Fused multiply-add
_mm256_loadu_ps        // Unaligned load for inputs
```

---

### Phase 3: Load Combine & Prefetch (Days 9-10)

**Goal:** Reduce memory latency with aligned loads and software prefetch

**Tasks:**
- [ ] Combine small loads into aligned 32-byte accesses
- [ ] Add `_mm_prefetch` for next block (distance tuning: 32, 64, 128 bytes)
- [ ] Measure memory bandwidth utilization with perf
- [ ] Validate no performance regression on small sizes
- [ ] Update load patterns in AVX2 kernels

**Success Criteria:**
- Memory time: 58 ns → ≤35 ns
- L1 cache hit rate: ≥95%

**Key Intrinsics:**
```rust
_mm_prefetch           // Software prefetch
_MM_HINT_T0            // L1 cache hint
```

---

### Phase 4: SIMD LUT via Permute (Days 3-4)

**Goal:** Vectorize code→weight mapping using permute instructions

**Tasks:**
- [ ] Implement SIMD LUT lookup with `_mm256_permutevar8x32_ps`
- [ ] Replace scalar array indexing with parallel permute
- [ ] Integrate with FMA loop (replace `weights` array)
- [ ] Validate numerical correctness (exact f32 match)
- [ ] Benchmark LUT throughput (target: 3× speedup)

**Success Criteria:**
- LUT time: 173 ns → ≤58 ns
- All unit tests passing

**Key Intrinsics:**
```rust
_mm256_permutevar8x32_ps   // Parallel LUT lookup
_mm256_cvtepu8_epi32       // Code widening (u8 → i32)
```

---

## Validation & Testing

### Unit Tests (Per Phase)

- [ ] Phase 1: `test_unpack_avx2_vs_scalar` (100+ random seeds)
- [ ] Phase 2: `test_gemv_tiled_vs_scalar` (8-64 rows, random data)
- [ ] Phase 3: `test_prefetch_no_regression` (small sizes < 256)
- [ ] Phase 4: `test_simd_lut_vs_scalar` (all 4 LUT codes)

### Integration Tests

- [ ] All 12 tests in `qk256_avx2_correctness.rs` passing
- [ ] Tolerance: max absolute error ≤1e-4
- [ ] Edge cases: partial blocks, non-aligned columns
- [ ] Stress test: 10,000 iterations with random seeds

### Benchmarks

- [ ] Baseline saved: `--save-baseline pre-optimization`
- [ ] Final speedup: ≥3× for 1024 elements
- [ ] Speedup across sizes: ≥2.5× for 256-4096 elements
- [ ] End-to-end inference: ~0.1 → ~0.3 tok/s (measured)

---

## Performance Tracking

### Baseline (Pre-Optimization)

```
qk256_dequant/scalar/1024   time:   [697 ns]   thrpt:  [1.47 Gelem/s]
qk256_dequant/avx2/1024     time:   [577 ns]   thrpt:  [1.78 Gelem/s]
Speedup: 1.21×
```

### Milestone 1: After Phase 1+4 (Week 1)

```
Target: ≥2× speedup
Expected time: ~350 ns
Expected throughput: ~3.0 Gelem/s
```

### Final Target: After All Phases (Week 2)

```
Target: ≥3× speedup
Expected time: ≤233 ns
Expected throughput: ≥4.4 Gelem/s
```

---

## Success Criteria Checklist

### Primary Goals

- [ ] **Speedup ≥3×** for 1024 elements (≤233 ns)
- [ ] All 12 correctness tests passing (tolerance 1e-4)
- [ ] No performance regression on small sizes (< 256 elements)

### Secondary Goals

- [ ] Speedup ≥2.5× for 256, 512, 4096 elements
- [ ] Speedup ≥2× for 16384 elements (memory-bound)
- [ ] End-to-end inference TPS: ~0.1 → ~0.3 tok/s
- [ ] Documentation updated (CLAUDE.md, benchmark guide)

### Stretch Goals

- [ ] Speedup ≥3.3× with micro-optimizations (≤175 ns)
- [ ] Throughput ≥5.5 Gelem/s
- [ ] L1 cache hit rate ≥98%

---

## Risk Register

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| Numerical instability | High | Tolerance 1e-4, property tests | Planned |
| No AVX2 at runtime | Medium | Keep scalar fallback | Implemented |
| Perf regression (small sizes) | Medium | Hybrid dispatch | Planned |
| Compiler auto-vectorization | Low | `#[inline(never)]` on scalar | Planned |

---

## Timeline

**Sprint Duration:** 2 weeks (10 working days)

**Week 1:**
- Days 1-2: Phase 1 (Nibble LUT)
- Days 3-4: Phase 4 (SIMD LUT)
- Day 5: Integration & baseline validation (target: ≥2× speedup)

**Week 2:**
- Days 6-8: Phase 2 (FMA Tiling)
- Days 9-10: Phase 3 (Prefetch)
- Days 11-12: Integration, profiling, fine-tuning
- Days 13-14: Final validation, documentation, PR

**Sprint Start:** [To be filled]

**Sprint End:** [To be filled]

---

## Acceptance Criteria

**Ready for Review:**
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Benchmark results showing ≥3× speedup
- [ ] Perf report showing bottleneck elimination
- [ ] Documentation updated (CLAUDE.md, benchmark guide)

**Ready for Merge:**
- [ ] Code review approved
- [ ] CI passing (including benchmark regression check)
- [ ] No performance regression on any test size
- [ ] All correctness tests passing with tolerance 1e-4

---

## Related Issues

- Depends on: (none - standalone optimization)
- Blocks: [Issue #XYZ - QK256 End-to-End Inference Performance]
- Related to: [Issue #469 - QK256 MVP Validation]

---

## References

- **Full Plan:** `docs/development/qk256-avx2-optimization-sprint.md`
- **Quick Reference:** `docs/development/qk256-avx2-sprint-summary.md`
- **Benchmark Guide:** `docs/benchmarks/qk256-dequant-benchmark.md`
- **Implementation:** `crates/bitnet-kernels/src/cpu/x86.rs`
- **GEMV Kernel:** `crates/bitnet-models/src/quant/i2s_qk256_avx2.rs`

---

**Labels:** `performance`, `optimization`, `avx2`, `qk256`, `sprint`

**Milestone:** v0.2.0 - QK256 Performance Optimization

**Assignees:** [To be assigned]
