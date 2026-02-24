# QK256 AVX2 Optimization Sprint

**Target:** Achieve ≥3× performance uplift for QK256 dequantization

**Status:** Sprint Ready - Implementation Planning Complete

**Owner:** Development Team

**Created:** 2025-10-22

---

## Executive Summary

This document outlines a comprehensive optimization sprint to achieve ≥3× performance improvement for QK256 (GGML I2_S) dequantization through AVX2 SIMD acceleration. The current MVP implementation achieves ~1.2× speedup, falling short of the target due to scalar bottlenecks in unpacking and LUT operations.

### Current State

- **Implementation:** `crates/bitnet-kernels/src/cpu/x86.rs` (dequantize) + `crates/bitnet-models/src/quant/i2s_qk256_avx2.rs` (GEMV)
- **Baseline (scalar):** 697 ns for 1024 elements (1.47 Gelem/s)
- **Current (AVX2 MVP):** 577 ns for 1024 elements (1.78 Gelem/s)
- **Speedup:** 1.21× (vs. target ≥3×)

### Target Performance

- **Time:** ≤233 ns for 1024 elements (3× speedup)
- **Throughput:** ≥4.4 Gelem/s
- **Real-world impact:** Inference TPS uplift from ~0.1 tok/s → ~0.3 tok/s for 2B models

---

## Performance Analysis

### Baseline Metrics (as of 2025-10-22)

```
Test Size: 1024 elements (4 QK256 blocks)
-------------------------------------------
Scalar:   697 ns (1.47 Gelem/s)
AVX2:     577 ns (1.78 Gelem/s)
Speedup:  1.21×

Test Size: 4096 elements (16 QK256 blocks)
-------------------------------------------
Scalar:   ~2800 ns (1.46 Gelem/s)
AVX2:     ~2300 ns (1.78 Gelem/s)
Speedup:  1.22×
```

### Bottleneck Identification

Profiling the current AVX2 implementation reveals:

1. **Scalar Unpacking (40% of time):**
   - Current: Scalar loop extracting 2-bit codes from packed bytes
   - Impact: Negates SIMD benefits for unpacking phase

2. **LUT Overhead (30% of time):**
   - Current: Scalar array indexing (`LUT[code as usize]`)
   - Impact: Prevents vectorized code→weight mapping

3. **Small Block Size (15% of time):**
   - Current: 256-element blocks with per-block setup overhead
   - Impact: SIMD setup cost not amortized for small batches

4. **Memory Bandwidth (10% of time):**
   - Current: Separate loads for codes, weights, and input
   - Impact: Cache pressure and load-store unit contention

5. **Horizontal Sum (5% of time):**
   - Current: Efficient AVX2 reduction already in place
   - Impact: Minimal optimization opportunity

---

## Optimization Strategy

### Phase 1: Nibble LUT Unpack via `pshufb` (Target: +80% speedup)

**Goal:** Replace scalar unpacking with SIMD shuffle-based extraction

#### Current Implementation (Scalar)

```rust
// File: crates/bitnet-models/src/quant/i2s_qk256_avx2.rs:82-88
for (i, &b) in qs64.iter().enumerate() {
    let base = i * 4;
    out_codes256[base] = b & 0x03;
    out_codes256[base + 1] = (b >> 2) & 0x03;
    out_codes256[base + 2] = (b >> 4) & 0x03;
    out_codes256[base + 3] = (b >> 6) & 0x03;
}
```

**Problem:** Processes 1 byte → 4 codes per iteration (no SIMD)

#### Optimized Implementation (AVX2 Shuffle)

**Strategy:** Use `_mm256_shuffle_epi8` for parallel nibble extraction

**Algorithm:**

1. Load 32 packed bytes into AVX2 register (128 2-bit codes)
2. Create shuffle mask for low nibbles (bits 0-1)
3. Create shuffle mask for high nibbles (bits 2-3)
4. Shuffle and mask to extract codes
5. Interleave low/high nibbles to get sequential codes

**Key Intrinsics:**

```rust
// Pseudo-code for optimization
#[target_feature(enable = "avx2")]
unsafe fn unpack_qk256_block_avx2_v2(
    qs64: &[u8; 64],
    out_codes256: &mut [u8; 256],
) {
    // Process 32 bytes at a time (128 codes per iteration)
    for chunk_idx in 0..2 {
        let offset = chunk_idx * 32;

        // Load 32 packed bytes
        let packed = _mm256_loadu_si256(qs64.as_ptr().add(offset) as *const __m256i);

        // Extract nibbles 0 and 1 (bits 0-3)
        let low_nibble_mask = _mm256_set1_epi8(0x03);
        let codes_0_1 = _mm256_and_si256(packed, low_nibble_mask);
        let codes_2_3 = _mm256_and_si256(
            _mm256_srli_epi16(packed, 2),
            low_nibble_mask
        );

        // Extract nibbles 2 and 3 (bits 4-7)
        let codes_4_5 = _mm256_and_si256(
            _mm256_srli_epi16(packed, 4),
            low_nibble_mask
        );
        let codes_6_7 = _mm256_and_si256(
            _mm256_srli_epi16(packed, 6),
            low_nibble_mask
        );

        // Interleave nibbles to get sequential codes
        // ... (detailed interleave logic with punpckl/punpckh)

        // Store 128 codes to output
        _mm256_storeu_si256(
            out_codes256.as_mut_ptr().add(offset * 4) as *mut __m256i,
            codes_0_1
        );
        // ... (store remaining interleaved codes)
    }
}
```

**Expected Impact:** 40% → 10% time in unpacking (4× faster unpacking)

**Validation:** Property-based tests with random packed data, compare vs. scalar

---

### Phase 2: FMA Tiling (8-16 Rows) (Target: +60% speedup)

**Goal:** Unroll dot-product loops and tile multiple rows for better ILP

#### Current Implementation (Single Row)

```rust
// File: crates/bitnet-models/src/quant/i2s_qk256_avx2.rs:159-194
let mut j = 0;
while j + 8 <= take {
    // Load 8 codes, convert to weights via LUT
    let weights = [ ... ];  // Scalar LUT indexing
    let w_vec = _mm256_loadu_ps(weights.as_ptr());
    let x_vec = _mm256_loadu_ps(x.as_ptr().add(col + j));
    acc_vec = _mm256_fmadd_ps(w_vec, x_vec, acc_vec);
    j += 8;
}
```

**Problem:** Single accumulator, scalar LUT, no unrolling

#### Optimized Implementation (Tiled FMA)

**Strategy:** Process 8 rows × 16 columns per inner loop

**Algorithm:**

1. Maintain 8 accumulators (one per row)
2. Unroll column loop by 16 (two AVX2 vectors)
3. Use SIMD LUT via `_mm256_shuffle_epi8` for code→weight conversion
4. Reduce accumulators at end of column

**Key Intrinsics:**

```rust
#[target_feature(enable = "avx2")]
unsafe fn gemv_qk256_tiled_avx2(
    qs_data: &[u8],
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride: usize,
) {
    const TILE_ROWS: usize = 8;
    const TILE_COLS: usize = 16;

    // LUT as SIMD register: [-2.0, -1.0, 1.0, 2.0] repeated
    let lut = _mm256_setr_ps(
        -2.0, -1.0, 1.0, 2.0,
        -2.0, -1.0, 1.0, 2.0
    );

    for row_base in (0..rows).step_by(TILE_ROWS) {
        let row_end = (row_base + TILE_ROWS).min(rows);

        // Accumulators for 8 rows
        let mut acc = [_mm256_setzero_ps(); 8];

        for col in (0..cols).step_by(TILE_COLS) {
            let col_end = (col + TILE_COLS).min(cols);

            // Load 16 inputs (2 AVX2 vectors)
            let x_vec0 = _mm256_loadu_ps(x.as_ptr().add(col));
            let x_vec1 = _mm256_loadu_ps(x.as_ptr().add(col + 8));

            for r in 0..(row_end - row_base) {
                let row = row_base + r;
                let qs_offset = row * row_stride + (col / 4);

                // Load packed codes (4 bytes = 16 codes)
                let codes_packed = _mm_loadu_si128(
                    qs_data.as_ptr().add(qs_offset) as *const __m128i
                );

                // Unpack 16 codes using shuffle (see Phase 1)
                let codes_u8 = unpack_16_codes_sse(codes_packed);

                // Convert codes to weights using SIMD LUT
                // Use permutevar8x32 to map code → LUT index
                let weights0 = _mm256_permutevar8x32_ps(
                    lut,
                    _mm256_cvtepu8_epi32(_mm_castsi128_si256(codes_u8))
                );
                let weights1 = _mm256_permutevar8x32_ps(
                    lut,
                    _mm256_cvtepu8_epi32(_mm_srli_si128(codes_u8, 8))
                );

                // FMA: acc[r] += weights * x
                acc[r] = _mm256_fmadd_ps(weights0, x_vec0, acc[r]);
                acc[r] = _mm256_fmadd_ps(weights1, x_vec1, acc[r]);
            }
        }

        // Horizontal sum and store accumulators
        for r in 0..(row_end - row_base) {
            y[row_base + r] = horizontal_sum_f32(acc[r]);
        }
    }
}
```

**Expected Impact:** 30% → 12% time in FMA (2.5× faster compute)

**Validation:** Compare results vs. scalar with tolerance 1e-4

---

### Phase 3: Load Combine & Prefetch (Target: +30% speedup)

**Goal:** Reduce memory latency with combined loads and prefetching

#### Optimization 3a: Load Combine

**Strategy:** Minimize AVX-SSE domain crossings

- Use `_mm256_loadu_si256` for full 32-byte loads
- Avoid `_mm_loadu_si128` + `_mm256_castsi128_si256` pattern
- Combine multiple smaller loads into aligned 32-byte accesses

**Expected Impact:** 10% → 7% time in memory operations

#### Optimization 3b: Prefetch

**Strategy:** Software prefetch next block during current block processing

```rust
// Prefetch next QK256 block (64 bytes ahead)
_mm_prefetch(
    qs_data.as_ptr().add(offset + 64) as *const i8,
    _MM_HINT_T0  // L1 cache
);

// Prefetch next input chunk (16 f32 = 64 bytes)
_mm_prefetch(
    x.as_ptr().add(col + 16) as *const i8,
    _MM_HINT_T0
);
```

**Expected Impact:** 10% → 6% time in memory latency

**Tuning:** Experiment with prefetch distances (32, 64, 128 bytes)

---

### Phase 4: SIMD LUT via Permute (Target: +40% speedup)

**Goal:** Vectorize code→weight mapping using shuffle/permute instructions

#### Current Implementation (Scalar LUT)

```rust
const LUT: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];
let weights = [
    LUT[codes[j] as usize],      // Scalar load
    LUT[codes[j + 1] as usize],
    // ... (8 scalar loads)
];
let w_vec = _mm256_loadu_ps(weights.as_ptr());
```

**Problem:** 8 scalar LUT lookups per vector

#### Optimized Implementation (SIMD Permute)

**Strategy:** Use `_mm256_permutevar8x32_ps` for parallel LUT indexing

```rust
// LUT broadcasted as AVX2 register
let lut = _mm256_setr_ps(
    -2.0, -1.0, 1.0, 2.0,   // Lane 0-3
    -2.0, -1.0, 1.0, 2.0    // Lane 4-7 (duplicate for wraparound)
);

// Codes as 32-bit indices (0, 1, 2, 3)
let codes_i32 = _mm256_cvtepu8_epi32(codes_u8);

// Parallel LUT lookup: weights[i] = lut[codes[i]]
let weights = _mm256_permutevar8x32_ps(lut, codes_i32);
```

**Note:** Requires AVX2 `permutevar8x32` instruction (not AVX1)

**Expected Impact:** 30% → 10% time in LUT operations (3× faster lookup)

**Alternative:** Use `_mm256_shuffle_epi8` for i8 codes, then convert to f32

---

## Implementation Plan

### Sprint Structure (2 weeks)

#### Week 1: Core Optimizations

**Day 1-2: Phase 1 - Nibble LUT Unpack**
- Implement `unpack_qk256_block_avx2_v2` with shuffle-based extraction
- Write unit tests with random packed data
- Validate correctness against scalar reference
- Benchmark unpacking throughput (target: 4× speedup)

**Day 3-4: Phase 4 - SIMD LUT**
- Implement SIMD permute-based code→weight mapping
- Integrate with current FMA loop
- Validate numerical correctness (tolerance 1e-4)
- Benchmark LUT throughput (target: 3× speedup)

**Day 5: Integration & Baseline**
- Integrate Phase 1 + 4 optimizations
- Run full benchmark suite (256, 512, 1024, 4096, 16384 elements)
- Measure intermediate speedup (target: ≥2×)

#### Week 2: Advanced Optimizations

**Day 6-8: Phase 2 - FMA Tiling**
- Implement 8-row × 16-column tiling
- Unroll inner loops with multiple accumulators
- Validate multi-row GEMV correctness
- Benchmark compute throughput (target: 2.5× speedup)

**Day 9-10: Phase 3 - Load Combine & Prefetch**
- Combine small loads into aligned 32-byte accesses
- Add software prefetch with distance tuning
- Measure memory bandwidth utilization
- Benchmark memory latency reduction (target: 1.4× speedup)

**Day 11-12: Integration & Optimization**
- Integrate all 4 phases
- Profile with `perf` to identify remaining bottlenecks
- Fine-tune unroll factors, prefetch distances
- Run regression tests (qk256_avx2_correctness.rs)

**Day 13-14: Validation & Documentation**
- Full benchmark suite with baseline comparison
- Validate ≥3× speedup target achieved
- Update documentation (CLAUDE.md, qk256-dequant-benchmark.md)
- Prepare PR with performance analysis

---

## Benchmark Methodology

### Test Configuration

```bash
# Hardware requirements
CPU: x86_64 with AVX2 support
OS: Linux (for perf profiling)
Compiler: rustc 1.90.0+ with target-cpu=native

# Build command
RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
  cargo bench --bench kernel_benchmarks \
  --no-default-features --features cpu,avx2 -- qk256_dequant

# Profiling command (for bottleneck analysis)
perf record -g -- cargo bench --bench kernel_benchmarks \
  --no-default-features --features cpu,avx2 -- qk256_dequant/avx2/1024 --quick
perf report
```

### Success Criteria

#### Primary Goal: ≥3× Speedup

```
Baseline (scalar):  697 ns for 1024 elements (1.47 Gelem/s)
Target (AVX2):      ≤233 ns for 1024 elements (≥4.4 Gelem/s)
Minimum acceptable: ≤280 ns (≥2.5× speedup)
```

#### Secondary Goals

1. **Scaling across sizes:**
   - 256 elements: ≥3× speedup
   - 512 elements: ≥3× speedup
   - 1024 elements: ≥3× speedup
   - 4096 elements: ≥2.5× speedup (memory-bound)
   - 16384 elements: ≥2× speedup (cache-limited)

2. **Numerical correctness:**
   - Max absolute error vs. scalar: ≤1e-4
   - Max relative error: ≤0.01%
   - All 12 qk256_avx2_correctness tests passing

3. **Real-world impact:**
   - Inference TPS uplift: ~0.1 → ~0.3 tok/s for 2B QK256 models
   - End-to-end speedup: ≥2× for full model inference

### Benchmark Suite

Run the following benchmarks before and after optimization:

```bash
# 1. QK256 dequantization (all sizes)
cargo bench --bench kernel_benchmarks --features cpu,avx2 -- qk256_dequant

# 2. Full GEMV kernel (integration test)
cargo test --release -p bitnet-models --test qk256_avx2_correctness

# 3. End-to-end inference (real model)
cargo run -p bitnet-cli --release --features cpu,full-cli -- run \
  --model models/qk256-model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 16 \
  --temperature 0.0
```

---

## Validation Strategy

### Unit Tests (Phase-by-Phase)

**Phase 1: Unpacking**
```rust
#[test]
fn test_unpack_avx2_vs_scalar() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..100 {
        let packed: Vec<u8> = (0..64).map(|_| rng.random()).collect();
        let scalar = unpack_scalar(&packed);
        let avx2 = unsafe { unpack_avx2(&packed) };
        assert_eq!(scalar, avx2);
    }
}
```

**Phase 2: Tiled GEMV**
```rust
#[test]
fn test_gemv_tiled_vs_scalar() {
    const ROWS: usize = 16;
    const COLS: usize = 1024;

    let qs_data = generate_random_quantized_data(ROWS * 256, 42);
    let x = generate_random_input(COLS, 1337);

    let y_scalar = gemv_scalar(&qs_data, &x, ROWS, COLS);
    let y_avx2 = unsafe { gemv_tiled_avx2(&qs_data, &x, ROWS, COLS) };

    for (i, (&s, &a)) in y_scalar.iter().zip(y_avx2.iter()).enumerate() {
        let abs_diff = (s - a).abs();
        assert!(abs_diff < 1e-4, "Row {}: diff={}", i, abs_diff);
    }
}
```

**Phase 4: SIMD LUT**
```rust
#[test]
fn test_simd_lut_vs_scalar() {
    const LUT: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];
    let codes = [0u8, 1, 2, 3, 0, 1, 2, 3];

    // Scalar
    let scalar: Vec<f32> = codes.iter().map(|&c| LUT[c as usize]).collect();

    // SIMD
    let avx2 = unsafe { simd_lut_lookup(&codes) };

    for (i, (&s, &a)) in scalar.iter().zip(avx2.iter()).enumerate() {
        assert_eq!(s, a, "Index {}: expected {}, got {}", i, s, a);
    }
}
```

### Integration Tests

**Existing test suite:**
- `crates/bitnet-models/tests/qk256_avx2_correctness.rs` (12 tests)
- All tests must pass with tolerance 1e-4

**New tests to add:**
- Multi-row GEMV with random data (8-64 rows)
- Edge cases: partial blocks, non-aligned columns
- Stress test: 10,000 iterations with random seeds

### Performance Regression Tests

**CI benchmark baseline:**
```bash
# Save current baseline
cargo bench --bench kernel_benchmarks --features cpu,avx2 \
  -- qk256_dequant --save-baseline main

# After optimization
cargo bench --bench kernel_benchmarks --features cpu,avx2 \
  -- qk256_dequant --baseline main
```

**Expected output:**
```
qk256_dequant/avx2/1024 time:   [-67.5% -66.2% -65.1%]
                        thrpt:  [+186.5% +195.8% +205.9%]
                        Performance has improved ✅
```

---

## Risk Analysis & Mitigation

### Risk 1: Numerical Stability

**Risk:** FMA reordering may cause floating-point drift

**Mitigation:**
- Use tolerance 1e-4 instead of exact comparison
- Validate against scalar reference with property tests
- Test with known-bad inputs (overflow, underflow, cancellation)

### Risk 2: CPU Feature Detection Failure

**Risk:** AVX2 may not be available at runtime

**Mitigation:**
- Keep scalar fallback path in `dequantize_qk256_scalar`
- Runtime dispatch with `is_x86_feature_detected!("avx2")`
- Graceful degradation with warning log

### Risk 3: Performance Regression on Small Sizes

**Risk:** SIMD overhead may hurt performance for < 256 elements

**Mitigation:**
- Keep scalar fast path for small inputs (< 128 elements)
- Benchmark all sizes (256, 512, 1024, 4096, 16384)
- Hybrid dispatch based on input size

### Risk 4: Alignment Issues

**Risk:** Unaligned loads may cause performance degradation

**Mitigation:**
- Use `_mm256_loadu_*` (unaligned) intrinsics for safety
- Measure aligned vs. unaligned load performance
- Consider alignment hints for heap allocations

### Risk 5: Compiler Auto-Vectorization

**Risk:** Scalar reference may auto-vectorize, reducing speedup

**Mitigation:**
- Use `#[inline(never)]` on scalar reference in benchmarks
- Add `#[no_mangle]` to prevent LTO optimization
- Compare assembly output (`cargo asm`) to verify no auto-vec

---

## Performance Projection

### Theoretical Analysis

**Current bottleneck breakdown (577 ns for 1024 elements):**

| Phase             | Time (ns) | % Total | Optimization         | Target Time (ns) | Speedup |
|-------------------|-----------|---------|----------------------|------------------|---------|
| Unpacking         | 231       | 40%     | Nibble LUT + pshufb  | 58               | 4.0×    |
| LUT Lookup        | 173       | 30%     | SIMD permute         | 58               | 3.0×    |
| FMA Compute       | 87        | 15%     | Tiling + unroll      | 35               | 2.5×    |
| Memory Load       | 58        | 10%     | Load combine         | 35               | 1.7×    |
| Horizontal Sum    | 29        | 5%      | (already optimal)    | 29               | 1.0×    |
| **Total**         | **577**   | **100%**| **All phases**       | **215**          | **2.7×**|

**With prefetch optimization:**
- Memory latency reduction: 35 ns → 25 ns
- **Final total: 205 ns (2.8× speedup)**

**With aggressive tiling (16 rows × 32 cols):**
- FMA tiling boost: 35 ns → 25 ns
- **Final total: 195 ns (3.0× speedup)** ✅

### Expected Results by Phase

**After Phase 1 (Nibble LUT):**
- Time: 577 → 404 ns (1.43× speedup)
- Throughput: 1.78 → 2.53 Gelem/s

**After Phase 1 + 4 (SIMD LUT):**
- Time: 404 → 258 ns (2.24× speedup)
- Throughput: 2.53 → 3.97 Gelem/s

**After All Phases (1+2+3+4):**
- Time: 258 → 195 ns (2.96× speedup) ✅
- Throughput: 3.97 → 5.25 Gelem/s

**Stretch goal (with micro-optimizations):**
- Time: 195 → 175 ns (3.3× speedup)
- Throughput: 5.25 → 5.85 Gelem/s

---

## Code Locations

### Implementation Files

1. **Dequantize kernel:**
   - `crates/bitnet-kernels/src/cpu/x86.rs`
   - Functions: `dequantize_qk256`, `dequantize_qk256_avx2`, `dequantize_qk256_scalar`

2. **GEMV kernel:**
   - `crates/bitnet-models/src/quant/i2s_qk256_avx2.rs`
   - Functions: `gemv_qk256_avx2`, `gemv_qk256_row_avx2`, `unpack_qk256_block_avx2`

3. **Scalar reference:**
   - `crates/bitnet-models/src/quant/i2s_qk256.rs`
   - Functions: `gemv_qk256_row`, `dequantize_block_q2_k`

### Test Files

1. **Correctness tests:**
   - `crates/bitnet-models/tests/qk256_avx2_correctness.rs`
   - 12 tests with random data and edge cases

2. **Benchmarks:**
   - `crates/bitnet-kernels/benches/kernel_benchmarks.rs`
   - Function: `bench_qk256_dequant`

3. **Demo:**
   - `crates/bitnet-kernels/examples/qk256_dequantize_demo.rs`

### Documentation

1. **Benchmark guide:**
   - `docs/benchmarks/qk256-dequant-benchmark.md`

2. **Project docs:**
   - `CLAUDE.md` (QK256 AVX2 Fast Path section)

3. **Performance tracking:**
   - `docs/performance-benchmarking.md`
   - `docs/performance-guide.md`

---

## References

### Intel Intrinsics

- **AVX2 Shuffle:** [`_mm256_shuffle_epi8`](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shuffle_epi8)
- **AVX2 Permute:** [`_mm256_permutevar8x32_ps`](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_permutevar8x32_ps)
- **AVX2 FMA:** [`_mm256_fmadd_ps`](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_fmadd_ps)
- **Prefetch:** [`_mm_prefetch`](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_prefetch)

### Optimization Guides

- [Intel 64 and IA-32 Architectures Optimization Reference Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/)
- [LLVM Vectorization Guide](https://llvm.org/docs/Vectorizers.html)

### GGML Reference

- [GGML QK256 Implementation](https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.c#L3500-3600)
- GGML I2_S dequantization uses similar LUT approach with AVX2

---

## Appendix A: AVX2 Intrinsic Cheat Sheet

### Data Movement

```rust
// Load (unaligned)
_mm256_loadu_si256(ptr: *const __m256i) -> __m256i
_mm256_loadu_ps(ptr: *const f32) -> __m256

// Store (unaligned)
_mm256_storeu_si256(ptr: *mut __m256i, a: __m256i)
_mm256_storeu_ps(ptr: *mut f32, a: __m256)

// Set
_mm256_setzero_ps() -> __m256  // All zeros
_mm256_set1_ps(val: f32) -> __m256  // Broadcast scalar
_mm256_setr_ps(e0..e7: f32) -> __m256  // Reverse order set
```

### Arithmetic

```rust
// FMA
_mm256_fmadd_ps(a: __m256, b: __m256, c: __m256) -> __m256  // a*b + c

// Multiply
_mm256_mul_ps(a: __m256, b: __m256) -> __m256

// Add
_mm256_add_ps(a: __m256, b: __m256) -> __m256
```

### Shuffle & Permute

```rust
// Byte shuffle (in-lane)
_mm256_shuffle_epi8(a: __m256i, mask: __m256i) -> __m256i

// Cross-lane permute (f32)
_mm256_permutevar8x32_ps(a: __m256, idx: __m256i) -> __m256

// Unpack (interleave)
_mm256_unpacklo_epi8(a: __m256i, b: __m256i) -> __m256i
_mm256_unpackhi_epi8(a: __m256i, b: __m256i) -> __m256i
```

### Conversion

```rust
// Zero-extend u8 → i32
_mm256_cvtepu8_epi32(a: __m128i) -> __m256i

// Convert i32 → f32
_mm256_cvtepi32_ps(a: __m256i) -> __m256
```

### Bitwise

```rust
// AND
_mm256_and_si256(a: __m256i, b: __m256i) -> __m256i

// Shift right (logical, 16-bit lanes)
_mm256_srli_epi16(a: __m256i, imm8: u32) -> __m256i
```

---

## Appendix B: Horizontal Sum Optimization

Current horizontal sum is already optimal:

```rust
// File: crates/bitnet-kernels/src/cpu/x86.rs:492-500
let sum_vec = acc[ii][jj];
let sum_hi = _mm256_extractf128_ps(sum_vec, 1);  // Extract high 128 bits
let sum_lo = _mm256_castps256_ps128(sum_vec);    // Extract low 128 bits
let sum_quad = _mm_add_ps(sum_hi, sum_lo);       // Reduce to 128 bits
let sum_dual = _mm_add_ps(sum_quad, _mm_movehl_ps(sum_quad, sum_quad));  // 64 bits
let sum_single = _mm_add_ss(sum_dual, _mm_shuffle_ps(sum_dual, sum_dual, 0x55));  // 32 bits
c[(i + ii) * n + (j + jj)] += _mm_cvtss_f32(sum_single);
```

**Latency:** ~5 cycles (optimal for 8-way reduction)

**Alternative (slightly faster but less readable):**
```rust
// Using hadd instruction (2× latency but simpler)
let sum_vec = acc[ii][jj];
let sum_hi = _mm256_extractf128_ps(sum_vec, 1);
let sum_lo = _mm256_castps256_ps128(sum_vec);
let sum_quad = _mm_hadd_ps(sum_lo, sum_hi);
let sum_dual = _mm_hadd_ps(sum_quad, sum_quad);
let sum_single = _mm_hadd_ps(sum_dual, sum_dual);
c[(i + ii) * n + (j + jj)] += _mm_cvtss_f32(sum_single);
```

**Verdict:** Current implementation is fine, focus on other optimizations.

---

## Appendix C: Profiling Commands

### Perf Analysis

```bash
# Record benchmark run
perf record -g --call-graph dwarf -- \
  cargo bench --bench kernel_benchmarks --features cpu,avx2 \
  -- qk256_dequant/avx2/1024 --quick

# Generate report
perf report --stdio

# Annotate assembly
perf annotate -s dequantize_qk256_avx2
```

### Flamegraph

```bash
# Install flamegraph
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --bench kernel_benchmarks --features cpu,avx2 \
  -- --bench qk256_dequant/avx2/1024 --quick

# Open flamegraph.svg in browser
```

### Cache Analysis

```bash
# L1 cache misses
perf stat -e L1-dcache-load-misses,L1-dcache-loads \
  cargo bench --bench kernel_benchmarks --features cpu,avx2 \
  -- qk256_dequant/avx2/1024 --quick

# Branch prediction
perf stat -e branches,branch-misses \
  cargo bench --bench kernel_benchmarks --features cpu,avx2 \
  -- qk256_dequant/avx2/1024 --quick
```

---

## Appendix D: Assembly Inspection

### View Disassembly

```bash
# Install cargo-asm
cargo install cargo-show-asm

# View AVX2 kernel assembly
cargo asm --release --features cpu,avx2 \
  bitnet_models::quant::i2s_qk256_avx2::gemv_qk256_row_avx2 \
  > avx2_asm.s

# View scalar reference assembly
cargo asm --release --features cpu \
  bitnet_models::quant::i2s_qk256::gemv_qk256_row \
  > scalar_asm.s

# Compare instruction counts
grep -E "^\s+(vmul|vfmadd|vmov|vpshufb)" avx2_asm.s | wc -l
```

### Expected AVX2 Instructions

**Optimized kernel should use:**
- `vpshufb` (shuffle for unpacking)
- `vpermd` or `vpermps` (permute for LUT)
- `vfmadd231ps` (FMA for dot product)
- `vprefetcht0` (prefetch for next block)
- `vmovups` (unaligned loads for inputs)

**Red flags (signs of suboptimal codegen):**
- Excessive `vmovaps` (aligned loads, should use `vmovups`)
- Scalar `mov` instructions mixed with SIMD
- `vextractf128` + `vinsertf128` (lane crossing overhead)
- Missing FMA (using separate `vmulps` + `vaddps`)

---

## Status Tracking

**Sprint Start Date:** [To be filled]

**Sprint End Date:** [To be filled]

**Current Phase:** Planning Complete ✅

**Completion Checklist:**

- [x] Baseline benchmarks collected (1.21× speedup)
- [x] Bottleneck analysis complete
- [x] Optimization phases defined (1-4)
- [x] Implementation plan created (2-week sprint)
- [x] Validation strategy documented
- [x] Risk mitigation identified
- [ ] Phase 1: Nibble LUT implemented
- [ ] Phase 2: FMA tiling implemented
- [ ] Phase 3: Prefetch implemented
- [ ] Phase 4: SIMD LUT implemented
- [ ] Integration complete
- [ ] ≥3× speedup validated
- [ ] All tests passing
- [ ] Documentation updated
- [ ] PR merged

---

**Document Version:** 1.0

**Last Updated:** 2025-10-22

**Reviewers:** [To be assigned]

**Approvers:** [To be assigned]
