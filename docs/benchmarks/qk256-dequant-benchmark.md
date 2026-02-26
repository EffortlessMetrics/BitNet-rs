# QK256 Dequantization Benchmark

## Overview

This benchmark measures the performance improvement of AVX2-accelerated QK256 dequantization compared to the scalar reference implementation.

**Benchmark file:** `crates/bitnet-kernels/benches/kernel_benchmarks.rs` (function `bench_qk256_dequant`)

**Target speedup:** ≥3× on AVX2 hardware

## Running the Benchmark

### Quick Test (Single Size)

Run a quick benchmark for a specific size (e.g., 256 elements):

```bash
cargo bench --bench kernel_benchmarks --no-default-features --features cpu,avx2 -- qk256_dequant/scalar/256 --quick
```

### Full Benchmark Suite

Run all QK256 dequantization benchmarks (scalar and AVX2 across all sizes):

```bash
cargo bench --bench kernel_benchmarks --no-default-features --features cpu,avx2 -- qk256_dequant
```

### Compare Scalar vs AVX2

To compare scalar and AVX2 performance for all sizes:

```bash
cargo bench --bench kernel_benchmarks --no-default-features --features cpu,avx2 -- qk256_dequant --save-baseline qk256
```

## Test Sizes

The benchmark tests the following sizes (all multiples of QK256 block size = 256):

- **256 elements** (1 block): Minimal overhead measurement
- **512 elements** (2 blocks): Small buffer test
- **1024 elements** (4 blocks): L1 cache friendly
- **4096 elements** (16 blocks): L2 cache friendly
- **16384 elements** (64 blocks): L3 cache / memory bound

## Metrics

The benchmark reports:

- **Time per iteration**: Wall-clock time for dequantization
- **Throughput**: Elements per second (Gelem/s)
- **Speedup**: AVX2 time / scalar time (target ≥3×)

## Expected Performance

Based on the current MVP implementation (see `crates/bitnet-models/src/quant/i2s_qk256_avx2.rs`):

### Current Status (MVP)

- **Scalar baseline**: ~1.7-1.8 Gelem/s (145 ns for 256 elements)
- **AVX2 (MVP)**: ~0.76× speedup (not yet meeting 3× target)

**Why AVX2 is not faster yet:**

1. **Scalar unpacking bottleneck**: 2-bit extraction is not vectorized
2. **LUT overhead**: Scalar array indexing prevents full SIMD utilization
3. **Compiler auto-vectorization**: The scalar reference may be auto-vectorized
4. **Small block size**: 256 elements may not amortize SIMD setup overhead

### Optimization Opportunities

To achieve target speedup (see `i2s_qk256_avx2.rs` for details):

1. **SIMD LUT with VPSHUFB**: Use `_mm256_shuffle_epi8` for parallel code→weight mapping
2. **Proper byte-level unpacking**: Use shuffle-based extraction instead of scalar loops
3. **Batch processing**: Process multiple blocks together
4. **Fused unpack+convert**: Eliminate intermediate `codes` buffer

## Interpreting Results

### Successful AVX2 Optimization

If you see output like:

```
qk256_dequant/scalar/4096   time:   [2.1 µs 2.2 µs 2.3 µs]
                            thrpt:  [1.78 Gelem/s 1.86 Gelem/s 1.95 Gelem/s]

qk256_dequant/avx2/4096     time:   [680 ns 700 ns 720 ns]
                            thrpt:  [5.69 Gelem/s 5.85 Gelem/s 6.02 Gelem/s]
                            speedup: 3.14× ✅
```

This indicates successful AVX2 optimization with **3.14× speedup** (meeting target).

### Current MVP Performance

If you see output like:

```
qk256_dequant/scalar/4096   time:   [2.1 µs 2.2 µs 2.3 µs]
                            thrpt:  [1.78 Gelem/s 1.86 Gelem/s 1.95 Gelem/s]

qk256_dequant/avx2/4096     time:   [2.8 µs 2.9 µs 3.0 µs]
                            thrpt:  [1.37 Gelem/s 1.41 Gelem/s 1.46 Gelem/s]
                            speedup: 0.76× ⚠️ (MVP limitation)
```

This indicates the AVX2 implementation is **not yet faster** than scalar (0.76× speedup). This is expected for the MVP and documented in `i2s_qk256_avx2.rs`.

## Hardware Requirements

- **CPU**: x86_64 with AVX2 support
- **OS**: Linux, macOS, or Windows
- **Rust**: 1.90.0+ (MSRV)

To check AVX2 support:

```bash
# Linux
grep -o 'avx2' /proc/cpuinfo | head -1

# macOS
sysctl -a | grep machdep.cpu.features | grep AVX2

# Windows
wmic cpu get caption,name,description | findstr "AVX2"
```

## Troubleshooting

### AVX2 benchmark not running

If you only see scalar benchmarks, check:

1. **AVX2 feature flag**: Ensure `--features avx2` is enabled
2. **Runtime detection**: AVX2 benchmarks only run if CPU supports AVX2
3. **Architecture**: AVX2 is only available on x86_64

### Performance Regression

If AVX2 performance degrades:

1. **Baseline comparison**: Use `--save-baseline` and `--baseline` to compare
2. **Compiler optimization**: Ensure `--release` or benchmark profile is used
3. **CPU frequency scaling**: Disable CPU throttling for consistent results

## See Also

- **Implementation**: `crates/bitnet-models/src/quant/i2s_qk256_avx2.rs`
- **Scalar reference**: `crates/bitnet-models/src/quant/i2s.rs`
- **CLAUDE.md**: Project documentation for BitNet-rs
