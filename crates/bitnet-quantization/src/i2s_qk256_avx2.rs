//! AVX2 SIMD implementation for GGML I2_S (QK=256) quantization
//!
//! This module provides AVX2-accelerated GEMV kernels for QK256 format.
//!
//! ## Optimization Strategy
//!
//! The full-block hot path is an AVX2/FMA optimization candidate. It must be
//! treated as a local F32/no-scale QK256 kernel until receipt/counter evidence
//! and exact-profile benchmarks accept any broader performance claim.
//!
//! It uses several mechanical changes that are validated by scalar/AVX2 parity
//! tests, but are not themselves proof of speedup:
//!
//! - **Single-load multi-lane decode**: Each 8-byte read produces codes for all
//!   four BitNet.cpp lanes (shifts 6/4/2/0).
//!
//! - **VPERMPS table lookup**: `_mm256_permutevar8x32_ps` maps the 2-bit
//!   codes `{0,1,2,3}` directly to weights `{-1, 0, +1, 0}`.
//!
//! - **Immediate shifts**: `_mm256_srli_epi32` with const-immediate shifts is
//!   used for the full-block lane decode.
//!
//! - **8-wide accumulator bank**: Four lane accumulators x two pipeline banks
//!   keep eight independent FMA dependency chains in flight.
//!
//! - **Software prefetch**: `_mm_prefetch(..., _MM_HINT_T0)` pulls the next
//!   block's quantized data and input vector into L1 before they're needed.
//!
//! Partial trailing blocks (where `cols` is not a multiple of 256) fall back
//! to a slower per-lane decode that preserves the original tail handling.
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

/// LUT for code → weight: codes `0..=3` map to `[-1.0, 0.0, +1.0, 0.0]`.
///
/// The upper four lanes mirror the lower four so that `_mm256_permutevar8x32_ps`
/// produces correct results even if higher bits of `codes` were ever set
/// (defense-in-depth — the AND with `mask_03` already guarantees `code < 4`).
#[cfg(target_arch = "x86_64")]
const QK256_WEIGHT_LUT: [f32; 8] = [-1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0];

/// Decode 8 codes per lane for all 4 BitNet.cpp lanes from a single 8-byte read.
///
/// Returns four `__m256` weight vectors `(w_lane0, w_lane1, w_lane2, w_lane3)`,
/// each holding eight f32 weights drawn from `{-1.0, 0.0, +1.0}` per the
/// BitNet.cpp I2_S code map `[0,1,2,3] -> [-1,0,+1,0]`.
///
/// # Safety
///
/// Requires AVX2 + FMA. Caller must verify via `is_x86_feature_detected!` or
/// equivalent before calling. Reads 8 bytes from `bytes`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn decode_8_codes_all_lanes_avx2(
    bytes: *const u8,
    lut: __m256,
    mask_03: __m256i,
) -> (__m256, __m256, __m256, __m256) {
    unsafe {
        // Single 8-byte load → 8 i32 lanes (each holding one packed byte).
        let eight_bytes = _mm_loadl_epi64(bytes as *const __m128i);
        let byte_lanes = _mm256_cvtepu8_epi32(eight_bytes);

        // Per-lane codes via const-immediate shifts for the full-block path.
        let codes0 = _mm256_and_si256(_mm256_srli_epi32::<6>(byte_lanes), mask_03);
        let codes1 = _mm256_and_si256(_mm256_srli_epi32::<4>(byte_lanes), mask_03);
        let codes2 = _mm256_and_si256(_mm256_srli_epi32::<2>(byte_lanes), mask_03);
        let codes3 = _mm256_and_si256(byte_lanes, mask_03);

        // VPERMPS table lookup: codes in {0,1,2,3} -> weights in {-1, 0, +1, 0}.
        let w0 = _mm256_permutevar8x32_ps(lut, codes0);
        let w1 = _mm256_permutevar8x32_ps(lut, codes1);
        let w2 = _mm256_permutevar8x32_ps(lut, codes2);
        let w3 = _mm256_permutevar8x32_ps(lut, codes3);

        (w0, w1, w2, w3)
    }
}

/// Decode 8 codes for a single lane via VPERMPS LUT (slow path tail helper).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn decode_8_codes_one_lane_avx2(
    bytes: *const u8,
    lane_shift: i32,
    lut: __m256,
    mask_03: __m256i,
) -> __m256 {
    unsafe {
        let eight_bytes = _mm_loadl_epi64(bytes as *const __m128i);
        let byte_lanes = _mm256_cvtepu8_epi32(eight_bytes);
        let shifts = _mm256_set1_epi32(lane_shift);
        let codes = _mm256_and_si256(_mm256_srlv_epi32(byte_lanes, shifts), mask_03);
        _mm256_permutevar8x32_ps(lut, codes)
    }
}

/// Process one full 256-element block, updating 8 lane×bank accumulators.
///
/// Each block contains two 128-element chunks. Each chunk holds 32 packed
/// bytes that decode to four 32-element lanes (shifts 6/4/2/0). We process
/// each chunk in four group-position iterations of 8 codes per lane,
/// alternating between two accumulator banks to keep the FMA pipeline full.
///
/// # Safety
///
/// `blk` must point to at least 64 valid bytes, and `x_chunk` (`x_ptr +
/// col`) must allow reads of 256 contiguous f32 values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn process_full_block_avx2(
    blk: *const u8,
    x_block_base: *const f32,
    lut: __m256,
    mask_03: __m256i,
    acc_a0: &mut __m256,
    acc_a1: &mut __m256,
    acc_a2: &mut __m256,
    acc_a3: &mut __m256,
    acc_b0: &mut __m256,
    acc_b1: &mut __m256,
    acc_b2: &mut __m256,
    acc_b3: &mut __m256,
) {
    // Chunk 0: bytes [0..32], lanes cover x[0..128]
    // Chunk 1: bytes [32..64], lanes cover x[128..256]
    //
    // Unrolling both chunks × four gp-iterations gives 8 SIMD iterations per
    // block, alternating banks A/B every iteration. Each iteration does
    // 4 lane FMAs, for 32 FMAs per block.
    unsafe {
        for chunk in 0..2 {
            let chunk_byte_base = chunk * 32;
            let chunk_elem_base = chunk * 128;
            let x_chunk = x_block_base.add(chunk_elem_base);

            // gp 0: bank A
            let (w0, w1, w2, w3) =
                decode_8_codes_all_lanes_avx2(blk.add(chunk_byte_base), lut, mask_03);
            let xv0 = _mm256_loadu_ps(x_chunk);
            let xv1 = _mm256_loadu_ps(x_chunk.add(32));
            let xv2 = _mm256_loadu_ps(x_chunk.add(64));
            let xv3 = _mm256_loadu_ps(x_chunk.add(96));
            *acc_a0 = _mm256_fmadd_ps(w0, xv0, *acc_a0);
            *acc_a1 = _mm256_fmadd_ps(w1, xv1, *acc_a1);
            *acc_a2 = _mm256_fmadd_ps(w2, xv2, *acc_a2);
            *acc_a3 = _mm256_fmadd_ps(w3, xv3, *acc_a3);

            // gp 8: bank B
            let (w0, w1, w2, w3) =
                decode_8_codes_all_lanes_avx2(blk.add(chunk_byte_base + 8), lut, mask_03);
            let xv0 = _mm256_loadu_ps(x_chunk.add(8));
            let xv1 = _mm256_loadu_ps(x_chunk.add(32 + 8));
            let xv2 = _mm256_loadu_ps(x_chunk.add(64 + 8));
            let xv3 = _mm256_loadu_ps(x_chunk.add(96 + 8));
            *acc_b0 = _mm256_fmadd_ps(w0, xv0, *acc_b0);
            *acc_b1 = _mm256_fmadd_ps(w1, xv1, *acc_b1);
            *acc_b2 = _mm256_fmadd_ps(w2, xv2, *acc_b2);
            *acc_b3 = _mm256_fmadd_ps(w3, xv3, *acc_b3);

            // gp 16: bank A
            let (w0, w1, w2, w3) =
                decode_8_codes_all_lanes_avx2(blk.add(chunk_byte_base + 16), lut, mask_03);
            let xv0 = _mm256_loadu_ps(x_chunk.add(16));
            let xv1 = _mm256_loadu_ps(x_chunk.add(32 + 16));
            let xv2 = _mm256_loadu_ps(x_chunk.add(64 + 16));
            let xv3 = _mm256_loadu_ps(x_chunk.add(96 + 16));
            *acc_a0 = _mm256_fmadd_ps(w0, xv0, *acc_a0);
            *acc_a1 = _mm256_fmadd_ps(w1, xv1, *acc_a1);
            *acc_a2 = _mm256_fmadd_ps(w2, xv2, *acc_a2);
            *acc_a3 = _mm256_fmadd_ps(w3, xv3, *acc_a3);

            // gp 24: bank B
            let (w0, w1, w2, w3) =
                decode_8_codes_all_lanes_avx2(blk.add(chunk_byte_base + 24), lut, mask_03);
            let xv0 = _mm256_loadu_ps(x_chunk.add(24));
            let xv1 = _mm256_loadu_ps(x_chunk.add(32 + 24));
            let xv2 = _mm256_loadu_ps(x_chunk.add(64 + 24));
            let xv3 = _mm256_loadu_ps(x_chunk.add(96 + 24));
            *acc_b0 = _mm256_fmadd_ps(w0, xv0, *acc_b0);
            *acc_b1 = _mm256_fmadd_ps(w1, xv1, *acc_b1);
            *acc_b2 = _mm256_fmadd_ps(w2, xv2, *acc_b2);
            *acc_b3 = _mm256_fmadd_ps(w3, xv3, *acc_b3);
        }
    }
}

/// Partial-block tail path: handles a final block where `take < 256`.
///
/// Mirrors the original MVP's per-lane decode while using the VPERMPS LUT.
/// Only invoked when `cols` is not a multiple of 256, which is rare in
/// production model dimensions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn process_partial_block_avx2(
    blk: *const u8,
    x_block_base: *const f32,
    take: usize,
    lut: __m256,
    mask_03: __m256i,
    acc0: &mut __m256,
    acc1: &mut __m256,
    acc2: &mut __m256,
    acc3: &mut __m256,
    scalar_acc: &mut f32,
) {
    unsafe {
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
                    let w = decode_8_codes_one_lane_avx2(
                        blk.add(chunk_byte_base + gp),
                        lane_shift,
                        lut,
                        mask_03,
                    );
                    let xv = _mm256_loadu_ps(x_block_base.add(lane_elem_base + gp));

                    match lane {
                        0 => *acc0 = _mm256_fmadd_ps(w, xv, *acc0),
                        1 => *acc1 = _mm256_fmadd_ps(w, xv, *acc1),
                        2 => *acc2 = _mm256_fmadd_ps(w, xv, *acc2),
                        _ => *acc3 = _mm256_fmadd_ps(w, xv, *acc3),
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
                    *scalar_acc += w * *x_block_base.add(lane_elem_base + gp);
                    gp += 1;
                }
            }
        }
    }
}

/// AVX2-accelerated dot product for one QK256 row.
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
        let mask_03 = _mm256_set1_epi32(0x03);
        let lut = _mm256_loadu_ps(QK256_WEIGHT_LUT.as_ptr());

        // 8 independent FMA accumulators: 4 lanes × 2 banks. Two banks shorten
        // the critical-path FMA dependency chain while preserving scalar parity.
        let mut acc_a0 = _mm256_setzero_ps();
        let mut acc_a1 = _mm256_setzero_ps();
        let mut acc_a2 = _mm256_setzero_ps();
        let mut acc_a3 = _mm256_setzero_ps();
        let mut acc_b0 = _mm256_setzero_ps();
        let mut acc_b1 = _mm256_setzero_ps();
        let mut acc_b2 = _mm256_setzero_ps();
        let mut acc_b3 = _mm256_setzero_ps();

        let mut scalar_acc = 0.0f32;
        let mut col = 0usize;

        let blk_ptr = qs_row.as_ptr();
        let x_ptr = x.as_ptr();

        for blk_idx in 0..blocks_needed {
            let blk = blk_ptr.add(blk_idx * QK256_PACKED_BYTES);
            let take = QK256_BLOCK.min(cols - col);
            let x_block_base = x_ptr.add(col);

            // Prefetch next block's packed bytes and input vector into L1.
            if blk_idx + 1 < blocks_needed {
                _mm_prefetch(blk.add(QK256_PACKED_BYTES) as *const i8, _MM_HINT_T0);
                _mm_prefetch(x_ptr.add(col + QK256_BLOCK) as *const i8, _MM_HINT_T0);
                _mm_prefetch(x_ptr.add(col + QK256_BLOCK + 16) as *const i8, _MM_HINT_T0);
            }

            if take == QK256_BLOCK {
                process_full_block_avx2(
                    blk,
                    x_block_base,
                    lut,
                    mask_03,
                    &mut acc_a0,
                    &mut acc_a1,
                    &mut acc_a2,
                    &mut acc_a3,
                    &mut acc_b0,
                    &mut acc_b1,
                    &mut acc_b2,
                    &mut acc_b3,
                );
            } else {
                process_partial_block_avx2(
                    blk,
                    x_block_base,
                    take,
                    lut,
                    mask_03,
                    &mut acc_a0,
                    &mut acc_a1,
                    &mut acc_a2,
                    &mut acc_a3,
                    &mut scalar_acc,
                );
            }

            col += take;
            if col >= cols {
                break;
            }
        }

        // Reduce 8 accumulators → 1, then horizontal sum.
        let s0 = _mm256_add_ps(acc_a0, acc_b0);
        let s1 = _mm256_add_ps(acc_a1, acc_b1);
        let s2 = _mm256_add_ps(acc_a2, acc_b2);
        let s3 = _mm256_add_ps(acc_a3, acc_b3);
        let s01 = _mm256_add_ps(s0, s1);
        let s23 = _mm256_add_ps(s2, s3);
        let acc = _mm256_add_ps(s01, s23);

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

    /// Timing probe for AVX2 vs scalar (manual, non-claiming test).
    ///
    /// This test records a rough scalar/AVX2 timing ratio while also checking
    /// parity. It is not a rigorous benchmark and does not establish a speedup
    /// claim.
    ///
    /// Note: Run with --release for accurate measurements:
    /// ```bash
    /// cargo test --release -p bitnet-quantization bench_avx2_timing_probe --no-default-features --features cpu,avx2 -- --nocapture
    /// ```
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn bench_avx2_timing_probe() {
        if std::env::var("BITNET_RUN_SLOW_TESTS").ok().as_deref() != Some("1") {
            eprintln!("Skipping timing probe; set BITNET_RUN_SLOW_TESTS=1 to enable");
            return;
        }
        use crate::i2s_qk256::gemv_qk256_row;
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        use std::time::Instant;

        // Skip if AVX2/FMA not available
        if !avx2_fma_runtime_available() {
            eprintln!("Skipping AVX2 timing probe: AVX2/FMA not available");
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

        // Time scalar implementation (using the actual scalar row function)
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

        // Time AVX2 implementation
        const AVX2_ITERS: usize = 10;
        let mut y_avx2 = vec![0.0f32; rows];
        let avx2_start = Instant::now();
        for _ in 0..AVX2_ITERS {
            gemv_qk256_avx2(&qs_data, &x, &mut y_avx2, rows, cols, row_stride_bytes)
                .expect("AVX2 GEMV should succeed");
        }
        let avx2_elapsed = avx2_start.elapsed();

        // Compute a local timing ratio. This is diagnostic only.
        let scalar_ms = scalar_elapsed.as_secs_f64() * 1000.0 / SCALAR_ITERS as f64;
        let avx2_ms = avx2_elapsed.as_secs_f64() * 1000.0 / AVX2_ITERS as f64;
        let timing_ratio = scalar_ms / avx2_ms;

        println!("\nAVX2 timing probe ({}x{} matrix):", rows, cols);
        println!("   Scalar: {:.3} ms/iter", scalar_ms);
        println!("   AVX2:   {:.3} ms/iter", avx2_ms);
        println!("   Scalar/AVX2 timing ratio: {:.2}x", timing_ratio);

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

        println!("   Timing probe is diagnostic only; it does not promote a speedup claim.");
        if timing_ratio >= 1.0 {
            println!("   Local probe ratio was >= 1.0; this is diagnostic evidence only.");
        } else {
            println!("   Local probe ratio was < 1.0; this is diagnostic evidence only.");
        }
    }
}
