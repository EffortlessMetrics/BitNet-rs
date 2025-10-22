# Benchmark Summary: QK256 Dequantization

## Changes Made

Added comprehensive criterion benchmarks to measure AVX2 QK256 performance vs scalar baseline.

## Files Modified

1. **`crates/bitnet-kernels/benches/kernel_benchmarks.rs`**
   - Added `bench_qk256_dequant()` function (lines 302-440)
   - Benchmarks both scalar and AVX2 implementations
   - Tests sizes: 256, 512, 1024, 4096, 16384 elements
   - Calculates throughput in elements/sec
   - Added to `criterion_group!` macro

## Files Created

1. **`docs/benchmarks/qk256-dequant-benchmark.md`**
   - Complete documentation for running and interpreting benchmarks
   - Hardware requirements and troubleshooting guide
   - Expected performance characteristics for MVP

2. **`docs/benchmarks/BENCHMARK_SUMMARY.md`**
   - This file

## Benchmark Design

### Structure

```rust
fn bench_qk256_dequant(c: &mut Criterion) {
    let mut group = c.benchmark_group("qk256_dequant");

    for size in [256, 512, 1024, 4096, 16384] {
        let quantized = vec![-1i8; size];
        let scales = vec![0.5f32; size / 256];

        group.throughput(Throughput::Elements(size as u64));

        // Benchmark scalar
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &size,
            |b, _| { /* scalar dequantization */ }
        );

        // Benchmark AVX2 (if available)
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        if is_x86_feature_detected!("avx2") {
            group.bench_with_input(
                BenchmarkId::new("avx2", size),
                &size,
                |b, _| { /* AVX2 dequantization */ }
            );
        }
    }

    group.finish();
}
```

### Test Data

- **Packed data**: `vec![0x1Bu8; size / 4]` (0x1B = 0b00011011, mixed 2-bit values)
- **Scales**: `vec![0.5f32; num_blocks]` (one scale per 256-element block)
- **Sizes**: Powers of 2 from 256 to 16384 (all multiples of QK256 block size)

### Metrics

- **Time per iteration**: Wall-clock time for dequantization
- **Throughput**: Elements per second (Gelem/s)
- **Speedup**: Calculated from time ratio (AVX2 / scalar)

## Running the Benchmarks

### Quick Test

```bash
cargo bench --bench kernel_benchmarks --no-default-features --features cpu,avx2 -- qk256_dequant/scalar/256 --quick
```

### Full Suite

```bash
cargo bench --bench kernel_benchmarks --no-default-features --features cpu,avx2 -- qk256_dequant
```

### Save Baseline

```bash
cargo bench --bench kernel_benchmarks --no-default-features --features cpu,avx2 -- qk256_dequant --save-baseline qk256
```

## Current Performance (MVP)

Based on test run on x86_64 with AVX2 support:

| Size   | Scalar (Gelem/s) | AVX2 (Gelem/s) | Speedup |
|--------|------------------|----------------|---------|
| 256    | 1.22             | 1.58           | 1.30×   |
| 512    | 1.27             | 1.31           | 1.03×   |
| 1024   | 1.14             | 1.48           | 1.30×   |
| 4096   | 1.56             | 1.83-1.92      | 1.17×-1.23× |
| 16384  | 1.41             | 1.73           | 1.23×   |

**Conclusion**: AVX2 implementation shows modest improvements (1.0×-1.3× speedup) but has not reached the 3× target yet.

### Why AVX2 is Not Faster Yet

1. **Scalar unpacking bottleneck**: 2-bit extraction is not vectorized
2. **LUT overhead**: Scalar array indexing prevents full SIMD utilization
3. **Compiler auto-vectorization**: The scalar reference may be auto-vectorized
4. **Small block size**: 256 elements may not amortize SIMD setup overhead

See `crates/bitnet-models/src/quant/i2s_qk256_avx2.rs` for detailed optimization notes.

## Expected Performance Gains

Once AVX2 optimizations are implemented (SIMD LUT, proper byte-level unpacking, batch processing):

- **Target speedup**: ≥3× on AVX2 hardware
- **Expected throughput**: ~4.5-6.0 Gelem/s for scalar baseline of ~1.5 Gelem/s

## Integration with CI/CD

These benchmarks can be integrated into CI/CD pipelines:

```bash
# Run benchmarks and save results
cargo bench --bench kernel_benchmarks --no-default-features --features cpu,avx2 -- qk256_dequant --save-baseline main

# Compare against baseline
cargo bench --bench kernel_benchmarks --no-default-features --features cpu,avx2 -- qk256_dequant --baseline main

# Generate regression report
# (requires criterion's HTML output enabled)
```

## See Also

- **Implementation**: `crates/bitnet-models/src/quant/i2s_qk256_avx2.rs`
- **Scalar reference**: `crates/bitnet-models/src/quant/i2s.rs`
- **Documentation**: `docs/benchmarks/qk256-dequant-benchmark.md`
- **CLAUDE.md**: Project documentation
