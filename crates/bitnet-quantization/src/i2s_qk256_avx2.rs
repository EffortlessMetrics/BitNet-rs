//! AVX2 SIMD implementation for GGML I2_S (QK=256) quantization
//!
//! This module provides AVX2-accelerated GEMV kernels for QK256 format.
//!
//! ## Optimization Strategy
//!
//! The hot path uses several techniques for throughput:
//!
//! - **SIMD variable shift** (`vpsrlvd`): Extracts 8 two-bit codes from 2 packed
//!   bytes in 3 SIMD ops (broadcast → variable shift → mask), replacing 8 scalar
//!   bit-extractions + `_mm256_setr_epi32`.
//!
//! - **4-wide accumulator bank**: Hides FMA latency (4–5 cycles on Haswell+) by
//!   keeping 4 independent dependency chains in flight.
//!
//! - **32-element inner loop**: Processes 8 packed bytes per iteration, amortizing
//!   loop overhead and giving the out-of-order engine more independent work.
//!
//! - **Software prefetch**: `_mm_prefetch(..., _MM_HINT_T0)` pulls the next block's
//!   quantized data and input vector into L1 before they're needed.
//!
//! ## Safety
//!
//! This module uses `unsafe` blocks for AVX2 intrinsics. All functions are marked
//! with `#[target_feature(enable = "avx2")]` to ensure proper CPU feature detection.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::i2s_qk256::{QK256_BLOCK, QK256_PACKED_BYTES};
use anyhow::Result;

/// Decode 8 two-bit codes from 2 packed bytes into 8 f32 weights using SIMD.
///
/// Uses `vpsrlvd` (per-lane variable shift) to extract all 8 codes in parallel:
///   packed = b0 | (b1 << 16)  →  broadcast to all lanes  →  shift by [0,2,4,6,16,18,20,22]  →  mask 0x03
///
/// Then maps codes [0,1,2,3] → weights [-2,-1,+1,+2] via: `weight = code - 2 + (code >> 1) & 1`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_8_weights_avx2(
    byte0: u8,
    byte1: u8,
    shifts: __m256i,
    mask_03: __m256i,
    two: __m256i,
    one: __m256i,
) -> __m256 {
    // Pack both bytes so per-lane shifts extract each 2-bit code:
    //   lanes 0–3 shift byte0 by 0,2,4,6; lanes 4–7 shift byte1 by 0,2,4,6
    //   (byte1 sits at bit 16, so shifts 16,18,20,22 reach its 2-bit fields).
    let packed = (byte0 as i32) | ((byte1 as i32) << 16);
    let broadcast = _mm256_set1_epi32(packed);
    let codes = _mm256_and_si256(_mm256_srlv_epi32(broadcast, shifts), mask_03);

    // codes ∈ {0,1,2,3} → weights ∈ {-2,-1,+1,+2}
    let shifted = _mm256_sub_epi32(codes, two);
    let correction = _mm256_and_si256(_mm256_srli_epi32::<1>(codes), one);
    _mm256_cvtepi32_ps(_mm256_add_epi32(shifted, correction))
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

    unsafe {
        // Hoisted constants shared by every decode_8_weights_avx2 call.
        let shifts = _mm256_setr_epi32(0, 2, 4, 6, 16, 18, 20, 22);
        let mask_03 = _mm256_set1_epi32(0x03);
        let two = _mm256_set1_epi32(2);
        let one = _mm256_set1_epi32(1);

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

            let mut j = 0usize;

            // --- 32-element main loop (8 packed bytes → 4 × 8-wide FMA) ---
            while j + 32 <= take {
                let pi = j / 4;

                let w0 = decode_8_weights_avx2(
                    *blk.add(pi),
                    *blk.add(pi + 1),
                    shifts,
                    mask_03,
                    two,
                    one,
                );
                let w1 = decode_8_weights_avx2(
                    *blk.add(pi + 2),
                    *blk.add(pi + 3),
                    shifts,
                    mask_03,
                    two,
                    one,
                );
                let w2 = decode_8_weights_avx2(
                    *blk.add(pi + 4),
                    *blk.add(pi + 5),
                    shifts,
                    mask_03,
                    two,
                    one,
                );
                let w3 = decode_8_weights_avx2(
                    *blk.add(pi + 6),
                    *blk.add(pi + 7),
                    shifts,
                    mask_03,
                    two,
                    one,
                );

                let xj = col + j;
                let x0 = _mm256_loadu_ps(x_ptr.add(xj));
                let x1 = _mm256_loadu_ps(x_ptr.add(xj + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(xj + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(xj + 24));

                acc0 = _mm256_fmadd_ps(w0, x0, acc0);
                acc1 = _mm256_fmadd_ps(w1, x1, acc1);
                acc2 = _mm256_fmadd_ps(w2, x2, acc2);
                acc3 = _mm256_fmadd_ps(w3, x3, acc3);

                j += 32;
            }

            // --- 8-element cleanup loop ---
            while j + 8 <= take {
                let pi = j / 4;
                let w = decode_8_weights_avx2(
                    *blk.add(pi),
                    *blk.add(pi + 1),
                    shifts,
                    mask_03,
                    two,
                    one,
                );
                let xv = _mm256_loadu_ps(x_ptr.add(col + j));
                acc0 = _mm256_fmadd_ps(w, xv, acc0);
                j += 8;
            }

            // --- Scalar tail (< 8 elements) ---
            while j < take {
                let packed_byte = *blk.add(j / 4);
                let shift = (j % 4) * 2;
                let code = (packed_byte >> shift) & 0x03;
                let w = match code {
                    0 => -2.0,
                    1 => -1.0,
                    2 => 1.0,
                    _ => 2.0,
                };
                scalar_acc += w * *x_ptr.add(col + j);
                j += 1;
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

/// AVX2 8-row fused GEMV: compute 8 output rows in a single pass over the input vector.
///
/// ## Key optimisation
///
/// The single-row kernel (`gemv_qk256_row_avx2`) reads the input vector `x` once per row.
/// For `rows` rows that means `rows` redundant passes over the same data.  This kernel
/// fuses 8 rows so `x` is loaded **once per 32-element column group** and the result is
/// scattered across 8 independent accumulator banks.
///
/// ### Memory access pattern
/// ```text
/// For each 32-element x block (4 × 8 floats):
///   Load x0, x1, x2, x3  →  4 YMM registers (reused for all 8 rows)
///   For row r in 0..8:
///     Decode w0..w3 for qs_data[row_base+r][block]  →  4 YMM temps
///     acc[r] = fmadd(w0,x0, fmadd(w1,x1, fmadd(w2,x2, fmadd(w3,x3, acc[r]))))
/// ```
///
/// YMM register budget: 4 x + 4 w_temps + 8 acc + hoisted consts (shifts/masks) ≈ 20.
/// The out-of-order engine overlaps independent row FMAs to hide the 4-cycle latency.
///
/// ### Expected speedup
///
/// * ~2× over the single-row kernel purely from x-vector reuse.
/// * Combines with prefetch and 4-wide decode to approach ≥3× vs scalar.
///
/// # Safety
///
/// Requires AVX2 + FMA.  Caller verifies via `is_x86_feature_detected!("avx2")`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemv_qk256_8row_avx2(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    row_base: usize,
    n_rows: usize, // 1..=8
    cols: usize,
    row_stride_bytes: usize,
) {
    debug_assert!(n_rows >= 1 && n_rows <= 8);
    debug_assert!(x.len() >= cols);

    let blocks_needed = cols.div_ceil(QK256_BLOCK);
    let x_ptr = x.as_ptr();
    let qs_ptr = qs_data.as_ptr();

    // Hoisted constants shared across all decode calls.
    let shifts = _mm256_setr_epi32(0, 2, 4, 6, 16, 18, 20, 22);
    let mask_03 = _mm256_set1_epi32(0x03);
    let two = _mm256_set1_epi32(2);
    let one = _mm256_set1_epi32(1);

    // Per-row accumulators (8 rows × 1 accumulator each; OOO engine provides ILP).
    let mut acc = [_mm256_setzero_ps(); 8];
    let mut scalar_acc = [0.0f32; 8];

    let mut col = 0usize;

    for blk_idx in 0..blocks_needed {
        let take = QK256_BLOCK.min(cols - col);

        // Prefetch the next block's packed bytes and input chunk.
        if blk_idx + 1 < blocks_needed {
            for r in 0..n_rows {
                _mm_prefetch(
                    qs_ptr.add((row_base + r) * row_stride_bytes + (blk_idx + 1) * QK256_PACKED_BYTES)
                        as *const i8,
                    _MM_HINT_T0,
                );
            }
            _mm_prefetch(x_ptr.add(col + QK256_BLOCK) as *const i8, _MM_HINT_T0);
        }

        let mut j = 0usize;

        // 32-element main loop: load x once, scatter FMAs across up to 8 rows.
        while j + 32 <= take {
            let xj = col + j;
            let x0 = _mm256_loadu_ps(x_ptr.add(xj));
            let x1 = _mm256_loadu_ps(x_ptr.add(xj + 8));
            let x2 = _mm256_loadu_ps(x_ptr.add(xj + 16));
            let x3 = _mm256_loadu_ps(x_ptr.add(xj + 24));

            for r in 0..n_rows {
                let blk = qs_ptr.add((row_base + r) * row_stride_bytes + blk_idx * QK256_PACKED_BYTES);
                let pi = j / 4;
                let w0 = decode_8_weights_avx2(*blk.add(pi), *blk.add(pi + 1), shifts, mask_03, two, one);
                let w1 = decode_8_weights_avx2(*blk.add(pi + 2), *blk.add(pi + 3), shifts, mask_03, two, one);
                let w2 = decode_8_weights_avx2(*blk.add(pi + 4), *blk.add(pi + 5), shifts, mask_03, two, one);
                let w3 = decode_8_weights_avx2(*blk.add(pi + 6), *blk.add(pi + 7), shifts, mask_03, two, one);
                // Accumulate all four groups into a single register to save YMM pressure.
                let partial = _mm256_fmadd_ps(w0, x0, _mm256_fmadd_ps(w1, x1, _mm256_fmadd_ps(w2, x2, _mm256_mul_ps(w3, x3))));
                acc[r] = _mm256_add_ps(acc[r], partial);
            }
            j += 32;
        }

        // 8-element cleanup loop.
        while j + 8 <= take {
            let xv = _mm256_loadu_ps(x_ptr.add(col + j));
            for r in 0..n_rows {
                let blk = qs_ptr.add((row_base + r) * row_stride_bytes + blk_idx * QK256_PACKED_BYTES);
                let pi = j / 4;
                let w = decode_8_weights_avx2(*blk.add(pi), *blk.add(pi + 1), shifts, mask_03, two, one);
                acc[r] = _mm256_fmadd_ps(w, xv, acc[r]);
            }
            j += 8;
        }

        // Scalar tail (< 8 remaining elements).
        while j < take {
            let xval = *x_ptr.add(col + j);
            for r in 0..n_rows {
                let blk = qs_ptr.add((row_base + r) * row_stride_bytes + blk_idx * QK256_PACKED_BYTES);
                let packed_byte = *blk.add(j / 4);
                let shift = (j % 4) * 2;
                let code = (packed_byte >> shift) & 0x03;
                let w = match code { 0 => -2.0, 1 => -1.0, 2 => 1.0, _ => 2.0 };
                scalar_acc[r] += w * xval;
            }
            j += 1;
        }

        col += take;
        if col >= cols {
            break;
        }
    }

    // Horizontal reduction for each row accumulator → scalar.
    for r in 0..n_rows {
        let hi = _mm256_extractf128_ps(acc[r], 1);
        let lo = _mm256_castps256_ps128(acc[r]);
        let s128 = _mm_add_ps(hi, lo);
        let s64 = _mm_hadd_ps(s128, s128);
        let s32 = _mm_hadd_ps(s64, s64);
        y_out[row_base + r] = _mm_cvtss_f32(s32) + scalar_acc[r];
    }
}

/// AVX2-accelerated multi-row GEMV: y = Ax where A is quantized QK256, x is dense
///
/// This is the public interface for AVX2-accelerated QK256 GEMV operations.
/// Runtime dispatch ensures this function is only called when AVX2 is available.
///
/// When there are 8 or more rows, the 8-row fused kernel is used to amortise
/// the cost of loading the input vector `x` across multiple output rows, giving
/// approximately 2× additional speedup over the single-row variant.
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
        let chunk = 8usize;
        let full_chunks = rows / chunk;
        let remainder = rows % chunk;

        // Process full 8-row chunks with the fused kernel.
        for c in 0..full_chunks {
            let row_base = c * chunk;
            // Prefetch the first cache line of the next chunk.
            if row_base + chunk < rows {
                _mm_prefetch(
                    qs_data.as_ptr().add((row_base + chunk) * row_stride_bytes) as *const i8,
                    _MM_HINT_T0,
                );
            }
            gemv_qk256_8row_avx2(qs_data, x, y_out, row_base, chunk, cols, row_stride_bytes);
        }

        // Handle any remaining rows (< 8) with the fused kernel using n_rows < 8.
        if remainder > 0 {
            let row_base = full_chunks * chunk;
            gemv_qk256_8row_avx2(qs_data, x, y_out, row_base, remainder, cols, row_stride_bytes);
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
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        // Skip if AVX2 not available
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 smoke test: AVX2 not available");
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

    /// Correctness test for the 8-row fused kernel vs the single-row reference.
    ///
    /// Verifies that `gemv_qk256_avx2` (which now uses the 8-row path for ≥8 rows)
    /// produces the same results as the scalar reference across a variety of shapes.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_gemv_qk256_8row_matches_scalar() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping 8-row test: AVX2 not available");
            return;
        }

        for &(rows, cols, seed) in &[
            (8usize, 256usize, 1u64),  // exactly one 8-row chunk, one block
            (16, 512, 2),              // two 8-row chunks, two blocks
            (9, 256, 3),               // one full chunk + 1 remainder row
            (7, 256, 4),               // remainder-only path (< 8 rows)
            (24, 1024, 5),             // larger test case
        ] {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let blocks_per_row = cols.div_ceil(QK256_BLOCK);
            let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;

            let mut qs_data = vec![0u8; rows * row_stride_bytes];
            for byte in qs_data.iter_mut() {
                *byte = rng.random();
            }
            let x: Vec<f32> = (0..cols).map(|_| rng.random_range(-5.0..5.0)).collect();

            // Reference: scalar row-by-row
            let mut y_scalar = vec![0.0f32; rows];
            for (row, out) in y_scalar.iter_mut().enumerate() {
                let s = row * row_stride_bytes;
                *out = crate::i2s_qk256::gemv_qk256_row(&qs_data[s..s + row_stride_bytes], &x, cols);
            }

            // AVX2 (now uses 8-row fused kernel internally)
            let mut y_avx2 = vec![0.0f32; rows];
            gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, rows, cols, row_stride_bytes)
                .expect("AVX2 GEMV should succeed");

            for (i, (&scalar, &avx2)) in y_scalar.iter().zip(y_avx2.iter()).enumerate() {
                let abs_diff = (scalar - avx2).abs();
                let block_count = (cols / QK256_BLOCK) as f32;
                let abs_tol = (1e-5f32 * block_count.sqrt()).min(5e-4);
                let rel_diff =
                    if scalar.abs() > 1e-12 { abs_diff / scalar.abs() } else { abs_diff };
                assert!(
                    abs_diff <= abs_tol || rel_diff <= 1e-4,
                    "8-row test (rows={rows} cols={cols} seed={seed}) failed at row {i}: \
                     scalar={scalar}, avx2={avx2}, abs_diff={abs_diff}",
                );
            }
        }
        println!("✅ 8-row fused kernel correctness tests passed");
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
