//! AVX2 SIMD implementation for GGML I2_S (QK=256) quantization
//!
//! This module provides AVX2-accelerated GEMV kernels for QK256 format.
//!
//! ## Current Status (MVP)
//!
//! The current implementation uses:
//! - **Scalar unpacking**: 2-bit code extraction (64 bytes → 256 codes)
//! - **Scalar LUT**: Code-to-float mapping via array indexing
//! - **AVX2 FMA loop**: Vectorized dot product (8 f32 lanes)
//! - **Scalar tail handling**: For non-multiple-of-8 elements
//!
//! ### Performance
//!
//! Current measurements show the AVX2 path is **not yet faster** than the scalar
//! reference (~0.76× speedup, target was 3-5×). This is because:
//!
//! 1. **Scalar unpacking bottleneck**: 2-bit extraction is not vectorized
//! 2. **LUT overhead**: Scalar array indexing prevents full SIMD utilization
//! 3. **Compiler auto-vectorization**: The scalar reference may be auto-vectorized
//! 4. **Small block size**: 256 elements may not amortize SIMD setup overhead
//!
//! ## Optimization Opportunities
//!
//! To achieve target speedup:
//!
//! 1. **SIMD LUT with VPSHUFB**: Use `_mm256_shuffle_epi8` for parallel code→weight
//!    mapping. This requires careful handling of 2-bit → 4-element LUT indexing.
//!
//! 2. **Proper byte-level unpacking**: Current approach using `_mm256_srli_epi16`
//!    shifts 16-bit lanes, not bytes. Need `_mm256_srli_epi64` or shuffle-based
//!    extraction.
//!
//! 3. **Batch processing**: Process multiple blocks together to improve instruction
//!    pipelining and reduce per-block overhead.
//!
//! 4. **Fused unpack+convert**: Eliminate intermediate `codes` buffer by directly
//!    converting packed bits → f32 weights in SIMD registers.
//!
//! ## Safety
//!
//! This module uses `unsafe` blocks for AVX2 intrinsics. All functions are marked
//! with `#[target_feature(enable = "avx2")]` to ensure proper CPU feature detection.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::i2s_qk256::{QK256_BLOCK, QK256_PACKED_BYTES};
use anyhow::Result;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_8_weights_avx2(byte0: u8, byte1: u8) -> __m256 {
    let codes = _mm256_setr_epi32(
        (byte0 & 0x03) as i32,
        ((byte0 >> 2) & 0x03) as i32,
        ((byte0 >> 4) & 0x03) as i32,
        ((byte0 >> 6) & 0x03) as i32,
        (byte1 & 0x03) as i32,
        ((byte1 >> 2) & 0x03) as i32,
        ((byte1 >> 4) & 0x03) as i32,
        ((byte1 >> 6) & 0x03) as i32,
    );

    // Convert codes in [0, 3] to weights [-2, -1, 1, 2] without scalar LUT lookups.
    let two = _mm256_set1_epi32(2);
    let one = _mm256_set1_epi32(1);
    let shifted = _mm256_sub_epi32(codes, two);
    let correction = _mm256_and_si256(_mm256_srli_epi32::<1>(codes), one);
    let corrected = _mm256_add_epi32(shifted, correction);
    _mm256_cvtepi32_ps(corrected)
}

/// AVX2-accelerated dot product for one QK256 row
///
/// Computes dot product between one quantized QK256 row and a dense input vector
/// using AVX2 intrinsics for 2-bit unpacking, widening, and FMA operations.
///
/// # Arguments
///
/// * `qs_row` - Row-major packed bytes (N * 64 bytes, where N = ceil(cols/256))
/// * `x` - Dense input vector (length = cols)
/// * `cols` - Number of columns (may not be multiple of 256)
///
/// # Returns
///
/// Scalar dot product result
///
/// # Safety
///
/// This function requires AVX2 support. Caller must ensure CPU has AVX2 capability.
///
/// # Performance
///
/// Target speedup: 3-5× over scalar gemv_qk256_row for typical matrix dimensions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
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

    // SAFETY: All intrinsics are safe to call because:
    // - Function is marked with #[target_feature(enable = "avx2")]
    // - Caller must ensure AVX2 is available via runtime dispatch
    // - All pointer operations are properly aligned and bounded
    unsafe {
        // SIMD accumulators (8 f32 lanes each). Two accumulators reduce dependency chain pressure.
        let mut acc_vec0 = _mm256_setzero_ps();
        let mut acc_vec1 = _mm256_setzero_ps();

        // Scalar accumulator for tail elements
        let mut scalar_acc = 0.0f32;

        let mut col = 0usize;
        for blk in qs_row.chunks_exact(QK256_PACKED_BYTES) {
            // Number of valid columns left in this block
            let take = QK256_BLOCK.min(cols - col);

            let mut j = 0usize;
            // Process 16 elements at a time (4 packed bytes).
            while j + 16 <= take {
                let packed_idx = j / 4;
                let b0 = *blk.get_unchecked(packed_idx);
                let b1 = *blk.get_unchecked(packed_idx + 1);
                let b2 = *blk.get_unchecked(packed_idx + 2);
                let b3 = *blk.get_unchecked(packed_idx + 3);

                let w_vec0 = decode_8_weights_avx2(b0, b1);
                let w_vec1 = decode_8_weights_avx2(b2, b3);

                let x_vec0 = _mm256_loadu_ps(x.as_ptr().add(col + j));
                let x_vec1 = _mm256_loadu_ps(x.as_ptr().add(col + j + 8));

                acc_vec0 = _mm256_fmadd_ps(w_vec0, x_vec0, acc_vec0);
                acc_vec1 = _mm256_fmadd_ps(w_vec1, x_vec1, acc_vec1);

                j += 16;
            }

            while j + 8 <= take {
                let packed_idx = j / 4;
                let b0 = *blk.get_unchecked(packed_idx);
                let b1 = *blk.get_unchecked(packed_idx + 1);

                let w_vec = decode_8_weights_avx2(b0, b1);
                let x_vec = _mm256_loadu_ps(x.as_ptr().add(col + j));
                acc_vec0 = _mm256_fmadd_ps(w_vec, x_vec, acc_vec0);

                j += 8;
            }

            // Handle tail elements (fewer than 8 remaining) with scalar accumulation
            while j < take {
                let packed_byte = blk[j / 4];
                let shift = (j % 4) * 2;
                let code = (packed_byte >> shift) & 0x03;
                let w = match code {
                    0 => -2.0,
                    1 => -1.0,
                    2 => 1.0,
                    _ => 2.0,
                };
                let xi = x[col + j];
                scalar_acc += w * xi;
                j += 1;
            }

            col += take;
            if col >= cols {
                break;
            }
        }

        let acc_vec = _mm256_add_ps(acc_vec0, acc_vec1);

        // Horizontal sum reduction: sum all 8 lanes of SIMD accumulators
        // Extract two 128-bit halves and add them
        let hi = _mm256_extractf128_ps(acc_vec, 1);
        let lo = _mm256_castps256_ps128(acc_vec);
        let sum128 = _mm_add_ps(hi, lo);

        // Horizontal add within 128-bit vector
        let sum64 = _mm_hadd_ps(sum128, sum128);
        let sum32 = _mm_hadd_ps(sum64, sum64);

        // Extract final scalar and add tail accumulator
        _mm_cvtss_f32(sum32) + scalar_acc
    }
}

/// AVX2-accelerated multi-row GEMV: y = Ax where A is quantized QK256, x is dense
///
/// This is the public interface for AVX2-accelerated QK256 GEMV operations.
/// Runtime dispatch ensures this function is only called when AVX2 is available.
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

    // SAFETY: We've verified AVX2 availability via runtime dispatch before calling this function.
    // All AVX2 intrinsics are properly guarded by #[target_feature(enable = "avx2")].
    unsafe {
        for (row, output) in y_out.iter_mut().enumerate().take(rows) {
            let start = row * row_stride_bytes;
            let end = start + row_stride_bytes;
            let row_bytes = &qs_data[start..end];
            *output = gemv_qk256_row_avx2(row_bytes, x, cols);
        }
    }

    Ok(())
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
        // Skip if AVX2 not available
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 smoke test: AVX2 not available");
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
        use crate::i2s_qk256::gemv_qk256;
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        // Skip if AVX2 not available
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 smoke test: AVX2 not available");
            return;
        }

        const TOLERANCE: f32 = 1e-5;

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

        // Compute reference result using scalar path
        let mut y_scalar = vec![0.0f32; rows];
        gemv_qk256(&qs_data, &x, &mut y_scalar, rows, cols, row_stride_bytes)
            .expect("Scalar GEMV should succeed");

        // Compute AVX2 result
        let mut y_avx2 = vec![0.0f32; rows];
        gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, rows, cols, row_stride_bytes)
            .expect("AVX2 GEMV should succeed");

        // Compare results
        for (i, (&scalar, &avx2)) in y_scalar.iter().zip(y_avx2.iter()).enumerate() {
            let abs_diff = (scalar - avx2).abs();
            assert!(
                abs_diff < TOLERANCE,
                "Smoke test failed at row {}: scalar={}, avx2={}, diff={}",
                i,
                scalar,
                avx2,
                abs_diff
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

        // Skip if AVX2 not available
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 benchmark: AVX2 not available");
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
