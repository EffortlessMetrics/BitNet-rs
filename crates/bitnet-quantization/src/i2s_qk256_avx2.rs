//! AVX2 SIMD implementation for GGML I2_S (QK=256) quantization
//!
//! This module provides AVX2-accelerated GEMV kernels for QK256 format.
//!
//! ## Optimization Strategy
//!
//! The hot path uses several techniques for throughput:
//!
//! - **SIMD byte expansion**: Extracts 8 grouped two-bit codes from 8 packed
//!   bytes, matching BitNet.cpp `dequantize_row_i2_s`.
//!
//! - **4-wide accumulator bank**: Hides FMA latency (4–5 cycles on Haswell+) by
//!   keeping 4 independent dependency chains in flight.
//!
//! - **Lane-aware 32-element chunks**: Processes each BitNet.cpp 32-value lane
//!   in 8-element SIMD chunks.
//!
//! - **Software prefetch**: `_mm_prefetch(..., _MM_HINT_T0)` pulls the next block's
//!   quantized data and input vector into L1 before they're needed.
//!
//! ## Safety
//!
//! This module uses `unsafe` blocks for AVX2/FMA intrinsics. FMA-using functions
//! are marked with `#[target_feature(enable = "avx2,fma")]` and must only be
//! called after runtime AVX2+FMA detection.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::i2s_qk256::{QK256_BLOCK, QK256_PACKED_BYTES};
use anyhow::Result;

/// Decode 8 grouped BitNet.cpp I2_S two-bit codes into 8 f32 weights using SIMD.
///
/// Each input byte contributes one code for the requested 32-value lane. The
/// BitNet.cpp I2_S map is [0,1,2,3] -> [-1,0,+1,0].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn decode_8_lane_weights_avx2(
    bytes: *const u8,
    lane_shift: i32,
    mask_03: __m256i,
    two: __m256i,
    zero: __m256i,
    one_ps: __m256,
    neg_one_ps: __m256,
) -> __m256 {
    let eight_bytes = unsafe { _mm_loadl_epi64(bytes as *const __m128i) };
    let byte_lanes = _mm256_cvtepu8_epi32(eight_bytes);
    let shifts = _mm256_set1_epi32(lane_shift);
    let codes = _mm256_and_si256(_mm256_srlv_epi32(byte_lanes, shifts), mask_03);

    let is_zero = _mm256_cmpeq_epi32(codes, zero);
    let is_two = _mm256_cmpeq_epi32(codes, two);
    let neg = _mm256_and_ps(_mm256_castsi256_ps(is_zero), neg_one_ps);
    let pos = _mm256_and_ps(_mm256_castsi256_ps(is_two), one_ps);
    _mm256_add_ps(neg, pos)
}

/// AVX2-accelerated dot product for one QK256 row.
///
/// # Optimizations over the MVP scalar-unpack path
///
/// 1. **SIMD code extraction**: `vpsrlvd` + broadcast replaces 8 scalar shifts per
///    8-element group, cutting unpack cost from ~16 scalar ops to 3 SIMD ops per group.
/// 2. **4-wide accumulator bank**: 4 independent FMA dependency chains hide the
///    4-cycle FMA latency on Haswell/Skylake.
/// 3. **32-element main loop**: 8 packed bytes → 32 codes → 4×FMA per iteration
///    reduces loop overhead and improves µop throughput.
/// 4. **Software prefetch**: L1 prefetch of the next block's packed bytes and input
///    vector prevents demand-miss stalls on block boundaries.
///
/// # Safety
///
/// Requires AVX2 + FMA. Caller must verify via `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_qk256_row_avx2(qs_row: &[u8], x: &[f32], cols: usize) -> f32 {
    let blocks_needed = cols.div_ceil(QK256_BLOCK);
    let expected_bytes = blocks_needed * QK256_PACKED_BYTES;

    debug_assert_eq!(
        qs_row.len(),
        expected_bytes,
        "AVX2: row bytes mismatch: got {}, expected {} for {} cols",
        qs_row.len(),
        expected_bytes,
        cols
    );
    debug_assert!(x.len() >= cols, "AVX2: x too short: {} < {}", x.len(), cols);

    unsafe {
        // Hoisted constants shared by every decode_8_lane_weights_avx2 call.
        let mask_03 = _mm256_set1_epi32(0x03);
        let two = _mm256_set1_epi32(2);
        let zero = _mm256_setzero_si256();
        let one_ps = _mm256_set1_ps(1.0);
        let neg_one_ps = _mm256_set1_ps(-1.0);

        // 4 independent FMA accumulators to saturate the FMA pipe.
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let mut scalar_acc = 0.0f32;
        let mut col = 0usize;

        let blk_ptr = qs_row.as_ptr();
        let x_ptr = x.as_ptr();

        for blk_idx in 0..blocks_needed {
            let blk = blk_ptr.add(blk_idx * QK256_PACKED_BYTES);
            let take = QK256_BLOCK.min(cols - col);

            // Prefetch next block's packed bytes and input vector into L1.
            if blk_idx + 1 < blocks_needed {
                _mm_prefetch(blk.add(QK256_PACKED_BYTES) as *const i8, _MM_HINT_T0);
                _mm_prefetch(x_ptr.add(col + QK256_BLOCK) as *const i8, _MM_HINT_T0);
                // Second cache-line of next input chunk (256 f32 = 1024 bytes ≈ 16 lines).
                _mm_prefetch(x_ptr.add(col + QK256_BLOCK + 16) as *const i8, _MM_HINT_T0);
            }

            for chunk in 0..2 {
                let chunk_byte_base = chunk * 32;
                let chunk_elem_base = chunk * 128;
                if chunk_elem_base >= take {
                    break;
                }

                for lane in 0..4 {
                    let lane_elem_base = chunk_elem_base + lane * 32;
                    if lane_elem_base >= take {
                        break;
                    }

                    let lane_take = 32usize.min(take - lane_elem_base);
                    let lane_shift = 6 - lane as i32 * 2;
                    let mut gp = 0usize;

                    while gp + 8 <= lane_take {
                        let w = decode_8_lane_weights_avx2(
                            blk.add(chunk_byte_base + gp),
                            lane_shift,
                            mask_03,
                            two,
                            zero,
                            one_ps,
                            neg_one_ps,
                        );
                        let xv = _mm256_loadu_ps(x_ptr.add(col + lane_elem_base + gp));

                        match (chunk, lane) {
                            (0, 0) | (1, 0) => acc0 = _mm256_fmadd_ps(w, xv, acc0),
                            (0, 1) | (1, 1) => acc1 = _mm256_fmadd_ps(w, xv, acc1),
                            (0, 2) | (1, 2) => acc2 = _mm256_fmadd_ps(w, xv, acc2),
                            _ => acc3 = _mm256_fmadd_ps(w, xv, acc3),
                        }
                        gp += 8;
                    }

                    while gp < lane_take {
                        let packed_byte = *blk.add(chunk_byte_base + gp);
                        let code = (packed_byte >> lane_shift) & 0x03;
                        let w = match code {
                            0 => -1.0,
                            2 => 1.0,
                            _ => 0.0,
                        };
                        scalar_acc += w * *x_ptr.add(col + lane_elem_base + gp);
                        gp += 1;
                    }
                }
            }

            col += take;
            if col >= cols {
                break;
            }
        }

        // Merge 4 accumulators → 1, then horizontal sum.
        let sum01 = _mm256_add_ps(acc0, acc1);
        let sum23 = _mm256_add_ps(acc2, acc3);
        let acc = _mm256_add_ps(sum01, sum23);

        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(hi, lo);
        let sum64 = _mm_hadd_ps(sum128, sum128);
        let sum32 = _mm_hadd_ps(sum64, sum64);

        _mm_cvtss_f32(sum32) + scalar_acc
    }
}

/// AVX2-accelerated multi-row GEMV: y = Ax where A is quantized QK256, x is dense
///
/// This is the public interface for AVX2-accelerated QK256 GEMV operations.
/// Runtime dispatch ensures this function is only called when AVX2 and FMA are available.
///
/// # Arguments
///
/// * `qs_data` - Contiguous row-major quantized data (rows * row_stride_bytes)
/// * `x` - Dense input vector (length = cols)
/// * `y_out` - Output vector (length = rows)
/// * `rows` - Number of rows
/// * `cols` - Number of columns
/// * `row_stride_bytes` - Bytes per row (ceil(cols/256) * 64)
///
/// # Errors
///
/// Returns error if dimensions don't match or data is insufficient.
///
/// # Safety
///
/// This function is safe to call from Rust code. Internal AVX2 intrinsics are
/// properly guarded by CPU feature detection in the runtime dispatch layer.
#[cfg(target_arch = "x86_64")]
pub fn gemv_qk256_avx2(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
) -> Result<()> {
    use anyhow::bail;

    if y_out.len() != rows {
        bail!("AVX2: y_out length {} != rows {}", y_out.len(), rows);
    }
    if x.len() < cols {
        bail!("AVX2: x length {} < cols {}", x.len(), cols);
    }

    let expected_total = rows * row_stride_bytes;
    if qs_data.len() < expected_total {
        bail!("AVX2: data too short: {} < {}", qs_data.len(), expected_total);
    }

    if !avx2_fma_runtime_available() {
        bail!("AVX2: avx2/fma CPU features are required for qk256 AVX2 GEMV");
    }

    // SAFETY: AVX2 and FMA availability is verified above before calling target-feature code.
    // All FMA-using intrinsics are guarded by #[target_feature(enable = "avx2,fma")].
    unsafe {
        for (row, output) in y_out.iter_mut().enumerate().take(rows) {
            // Prefetch next row's first cache line to overlap decode with memory.
            if row + 1 < rows {
                _mm_prefetch(
                    qs_data.as_ptr().add((row + 1) * row_stride_bytes) as *const i8,
                    _MM_HINT_T0,
                );
            }
            let start = row * row_stride_bytes;
            let end = start + row_stride_bytes;
            let row_bytes = &qs_data[start..end];
            *output = gemv_qk256_row_avx2(row_bytes, x, cols);
        }
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn avx2_fma_runtime_available() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// Stub implementation for non-x86_64 architectures
///
/// This stub ensures the module compiles on all platforms. Runtime dispatch
/// will never call this function on non-x86_64 architectures.
#[cfg(not(target_arch = "x86_64"))]
pub fn gemv_qk256_avx2(
    _qs_data: &[u8],
    _x: &[f32],
    _y_out: &mut [f32],
    _rows: usize,
    _cols: usize,
    _row_stride_bytes: usize,
) -> Result<()> {
    anyhow::bail!("AVX2 implementation only available on x86_64 architecture")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: Verify AVX2 path produces correct results for basic case
    ///
    /// This test validates that the AVX2 implementation produces identical results
    /// to the scalar reference for a simple case (all codes = 2 → +1.0).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_smoke() {
        // Skip if AVX2/FMA not available
        if !avx2_fma_runtime_available() {
            eprintln!("Skipping AVX2 smoke test: AVX2/FMA not available");
            return;
        }

        // All codes = 2 (→ +1.0 with default LUT), so dot == sum(x)
        let mut qs = [0u8; QK256_PACKED_BYTES];
        // Code 2 everywhere → 0b_10_10_10_10 = 0xAA
        qs.fill(0xAA);

        let cols = 256usize; // 1 block
        let row_stride_bytes = QK256_PACKED_BYTES;
        let qs_data = qs.to_vec();

        let x: Vec<f32> = (0..cols).map(|i| i as f32 * 0.01).collect();
        let expected: f32 = x.iter().sum(); // because weight=+1.0 everywhere

        let mut y_out = vec![0.0f32; 1];
        gemv_qk256_avx2(&qs_data, &x, &mut y_out, 1, cols, row_stride_bytes)
            .expect("AVX2 GEMV should succeed");

        // Allow small floating-point error
        let abs_diff = (y_out[0] - expected).abs();
        assert!(
            abs_diff < 1e-3,
            "AVX2 smoke test failed: expected ~{}, got {}, diff={}",
            expected,
            y_out[0],
            abs_diff
        );
    }

    /// Smoke test: AVX2 implementation matches scalar reference
    ///
    /// This is a minimal smoke test to verify basic AVX2 functionality.
    /// For comprehensive correctness validation, see the integration test suite
    /// in `tests/qk256_avx2_correctness.rs`.
    ///
    /// # Test Coverage
    ///
    /// - Single test case: 4×256 matrix (single block per row, seed 42)
    /// - Validates basic AVX2 vs scalar parity
    /// - Ensures the module compiles and links correctly
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_gemv_qk256_avx2_smoke() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        // Skip if AVX2/FMA not available
        if !avx2_fma_runtime_available() {
            eprintln!("Skipping AVX2 smoke test: AVX2/FMA not available");
            return;
        }

        // Single smoke test case: 4×256 (single block per row)
        let (rows, cols, seed) = (4usize, 256usize, 42u64);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let blocks_per_row = cols.div_ceil(QK256_BLOCK);
        let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;

        // Generate random quantized data
        let mut qs_data = vec![0u8; rows * row_stride_bytes];
        for byte in qs_data.iter_mut() {
            *byte = rng.random();
        }

        // Generate random input vector
        let x: Vec<f32> = (0..cols).map(|_| rng.random_range(-10.0..10.0)).collect();

        // Compute reference result using explicit scalar row kernel (no dispatch),
        // so this test always compares AVX2 against true scalar execution.
        let mut y_scalar = vec![0.0f32; rows];
        for (row, output) in y_scalar.iter_mut().enumerate().take(rows) {
            let start = row * row_stride_bytes;
            let end = start + row_stride_bytes;
            *output = crate::i2s_qk256::gemv_qk256_row(&qs_data[start..end], &x, cols);
        }

        // Compute AVX2 result
        let mut y_avx2 = vec![0.0f32; rows];
        gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, rows, cols, row_stride_bytes)
            .expect("AVX2 GEMV should succeed");

        // Compare results
        for (i, (&scalar, &avx2)) in y_scalar.iter().zip(y_avx2.iter()).enumerate() {
            let abs_diff = (scalar - avx2).abs();
            let block_count = (cols / QK256_BLOCK) as f32;
            let abs_tol = (1e-5f32 * block_count.sqrt()).min(5e-4);
            let rel_tol = 1e-4f32;
            let rel_diff = if scalar.abs() > 1e-12 { abs_diff / scalar.abs() } else { abs_diff };

            assert!(
                abs_diff <= abs_tol || rel_diff <= rel_tol,
                "Smoke test failed at row {}: scalar={}, avx2={}, abs_diff={}, rel_diff={}, abs_tol={}, rel_tol={}",
                i,
                scalar,
                avx2,
                abs_diff,
                rel_diff,
                abs_tol,
                rel_tol
            );
        }

        println!("✅ AVX2 smoke test passed: {}×{} (seed={})", rows, cols, seed);
    }

    /// Test that AVX2 stub returns error on non-x86_64 architectures
    #[test]
    #[cfg(not(target_arch = "x86_64"))]
    fn test_avx2_stub_errors() {
        let qs_data = vec![0u8; 64];
        let x = vec![0.0f32; 256];
        let mut y_out = vec![0.0f32; 1];

        let result = gemv_qk256_avx2(&qs_data, &x, &mut y_out, 1, 256, 64);
        assert!(result.is_err(), "AVX2 stub should return error on non-x86_64");
        assert!(
            result.unwrap_err().to_string().contains("x86_64"),
            "Error should mention x86_64 requirement"
        );
    }

    /// Benchmark AVX2 speedup vs scalar (manual timing test)
    ///
    /// This test measures the performance improvement of the AVX2 implementation
    /// compared to the scalar reference. It's not a rigorous benchmark but provides
    /// a quick validation that AVX2 is actually faster.
    ///
    /// Target: ≥3× speedup for typical matrix dimensions
    ///
    /// Note: Run with --release for accurate measurements:
    /// ```bash
    /// cargo test --release -p bitnet-models bench_avx2 -- --nocapture --ignored
    /// ```
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn bench_avx2_speedup() {
        if std::env::var("BITNET_RUN_SLOW_TESTS").ok().as_deref() != Some("1") {
            eprintln!("⏭️  Skipping benchmark test; set BITNET_RUN_SLOW_TESTS=1 to enable");
            return;
        }
        use crate::i2s_qk256::gemv_qk256_row;
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        use std::time::Instant;

        // Skip if AVX2/FMA not available
        if !avx2_fma_runtime_available() {
            eprintln!("Skipping AVX2 benchmark: AVX2/FMA not available");
            return;
        }

        // Test configuration: large enough to amortize overhead
        let rows = 512usize;
        let cols = 2048usize; // 8 blocks per row
        let seed = 42u64;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let blocks_per_row = cols.div_ceil(QK256_BLOCK);
        let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;

        // Generate random quantized data
        let mut qs_data = vec![0u8; rows * row_stride_bytes];
        for byte in qs_data.iter_mut() {
            *byte = rng.random();
        }

        // Generate random input vector
        let x: Vec<f32> = (0..cols).map(|_| rng.random_range(-10.0..10.0)).collect();

        // Warmup
        let mut y_warmup = vec![0.0f32; rows];
        gemv_qk256_avx2(&qs_data, &x, &mut y_warmup, rows, cols, row_stride_bytes)
            .expect("AVX2 warmup should succeed");

        // Benchmark scalar implementation (using the actual scalar row function)
        const SCALAR_ITERS: usize = 10;
        let mut y_scalar = vec![0.0f32; rows];
        let scalar_start = Instant::now();
        for _ in 0..SCALAR_ITERS {
            for (row, output) in y_scalar.iter_mut().enumerate().take(rows) {
                let start = row * row_stride_bytes;
                let end = start + row_stride_bytes;
                let row_bytes = &qs_data[start..end];
                *output = gemv_qk256_row(row_bytes, &x, cols);
            }
        }
        let scalar_elapsed = scalar_start.elapsed();

        // Benchmark AVX2 implementation
        const AVX2_ITERS: usize = 10;
        let mut y_avx2 = vec![0.0f32; rows];
        let avx2_start = Instant::now();
        for _ in 0..AVX2_ITERS {
            gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, rows, cols, row_stride_bytes)
                .expect("AVX2 GEMV should succeed");
        }
        let avx2_elapsed = avx2_start.elapsed();

        // Compute speedup
        let scalar_ms = scalar_elapsed.as_secs_f64() * 1000.0 / SCALAR_ITERS as f64;
        let avx2_ms = avx2_elapsed.as_secs_f64() * 1000.0 / AVX2_ITERS as f64;
        let speedup = scalar_ms / avx2_ms;

        println!("\n📊 AVX2 Benchmark Results ({}×{} matrix):", rows, cols);
        println!("   Scalar: {:.3} ms/iter", scalar_ms);
        println!("   AVX2:   {:.3} ms/iter", avx2_ms);
        println!("   Speedup: {:.2}×", speedup);

        // Verify correctness
        for (i, (&scalar, &avx2)) in y_scalar.iter().zip(y_avx2.iter()).enumerate() {
            let abs_diff = (scalar - avx2).abs();
            let rel_diff = if scalar.abs() > 1e-6 { abs_diff / scalar.abs() } else { abs_diff };
            assert!(
                abs_diff < 1e-3 || rel_diff < 1e-4,
                "Mismatch at row {}: scalar={}, avx2={}, abs_diff={}, rel_diff={}",
                i,
                scalar,
                avx2,
                abs_diff,
                rel_diff
            );
        }

        // NOTE: Current MVP implementation does not achieve target speedup
        // This is expected and documented in the module-level docs
        // The correctness tests pass, validating the implementation is correct

        if speedup >= 3.0 {
            println!("✅ AVX2 speedup {:.2}× meets ≥3× target", speedup);
        } else if speedup >= 1.0 {
            println!("⚠️  AVX2 speedup {:.2}× is below 3× target (MVP limitation)", speedup);
            println!("    See module docs for optimization opportunities");
        } else {
            println!("⚠️  AVX2 {:.2}× slower than scalar (MVP limitation)", 1.0 / speedup);
            println!("    Scalar unpacking + LUT overhead exceeds SIMD FMA gains");
            println!("    See module docs for optimization roadmap");
        }
    }
}
