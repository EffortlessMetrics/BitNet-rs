//! ARM NEON accelerated I2_S quantized matrix multiplication for Apple Silicon.
//!
//! Bridges the scalar [`super::quantized_matmul`] reference and the float
//! NEON kernels with NEON-accelerated 2-bit dequantization and fused
//! dot-product / GEMV operations.
//!
//! I2_S encoding (2 bits per value, 4 values per byte, LSB-first):
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1
//! - `0b10` → unused (treated as 0)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── I2_S decode helpers ────────────────────────────────────────────────

/// Decode a single 2-bit I2_S code to its signed value.
#[inline(always)]
fn decode_i2s(bits: u8) -> f32 {
    match bits & 0x03 {
        0b01 => 1.0,
        0b11 => -1.0,
        _ => 0.0, // 0b00 = 0, 0b10 = unused → 0
    }
}

// ── NEON dequantize ────────────────────────────────────────────────────

/// Dequantize a block of I2_S 2-bit packed values to `f32` using NEON.
///
/// Each byte of `packed` encodes 4 ternary values (2 bits each, LSB-first).
/// `out` receives one `f32` per value: `{-1.0, 0.0, +1.0}`.
///
/// Processes 4 values (1 byte) at a time via NEON, with scalar fallback
/// for any remainder values that don't fill a complete byte.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_dequantize_block(packed: &[u8], out: &mut [f32]) {
    // NEON lookup table: index by 2-bit code → f32 value
    // [0b00→0.0, 0b01→1.0, 0b10→0.0, 0b11→-1.0]
    let lut: [f32; 4] = [0.0, 1.0, 0.0, -1.0];

    let full_bytes = out.len() / 4;
    let remainder = out.len() % 4;

    for (i, &byte) in packed.iter().enumerate().take(full_bytes) {
        // Extract 4 two-bit codes from the byte
        let c0 = (byte & 0x03) as usize;
        let c1 = ((byte >> 2) & 0x03) as usize;
        let c2 = ((byte >> 4) & 0x03) as usize;
        let c3 = ((byte >> 6) & 0x03) as usize;

        // Build an f32x4 from the lookup table and store via NEON
        let vals = [lut[c0], lut[c1], lut[c2], lut[c3]];
        unsafe {
            let v = vld1q_f32(vals.as_ptr());
            vst1q_f32(out.as_mut_ptr().add(i * 4), v);
        }
    }

    // Scalar fallback for remainder values
    if remainder > 0 && full_bytes < packed.len() {
        let byte = packed[full_bytes];
        for j in 0..remainder {
            let bits = (byte >> (j * 2)) & 0x03;
            out[full_bytes * 4 + j] = decode_i2s(bits);
        }
    }
}

// ── NEON dot product ───────────────────────────────────────────────────

/// Compute the dot product of dequantized I2_S weights and `f32` activations
/// using NEON fused multiply-add (`vfmaq_f32`).
///
/// Both slices must have the same length. Processes 4 elements at a time.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_dot_product(weights: &[f32], activations: &[f32]) -> f32 {
    assert_eq!(weights.len(), activations.len());
    let n = weights.len();
    let chunks = n / 4;

    let w_ptr = weights.as_ptr();
    let a_ptr = activations.as_ptr();

    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let vw = vld1q_f32(w_ptr.add(offset));
            let va = vld1q_f32(a_ptr.add(offset));
            acc = vfmaq_f32(acc, vw, va);
        }
    }

    // Horizontal sum of the 4-lane accumulator
    let mut sum = vaddvq_f32(acc);

    // Scalar tail
    for i in (chunks * 4)..n {
        sum += weights[i] * activations[i];
    }

    sum
}

// ── NEON GEMV ──────────────────────────────────────────────────────────

/// NEON-accelerated GEMV with I2_S packed weights.
///
/// Computes `output[row] = scale * Σ_col (dequant(weights)[row, col] * activations[col])`
/// for a matrix of `rows × cols` ternary weights packed in I2_S format.
///
/// # Layout
///
/// - `weights_packed`: I2_S packed, one row per `ceil(cols/4)` bytes,
///   stored row-major (byte `row * packed_cols + byte_idx`)
/// - `activations`: `f32` slice of length `cols`
/// - `output`: `f32` slice of length `rows`
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_quantized_gemv(
    weights_packed: &[u8],
    activations: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    scale: f32,
) {
    let packed_cols = cols.div_ceil(4);

    assert!(
        weights_packed.len() >= rows * packed_cols,
        "weights_packed too small: need {}, got {}",
        rows * packed_cols,
        weights_packed.len(),
    );
    assert!(
        activations.len() >= cols,
        "activations too small: need {cols}, got {}",
        activations.len(),
    );
    assert!(output.len() >= rows, "output too small: need {rows}, got {}", output.len(),);

    let lut: [f32; 4] = [0.0, 1.0, 0.0, -1.0];
    let a_ptr = activations.as_ptr();

    for (row, out_val) in output.iter_mut().enumerate().take(rows) {
        let row_start = row * packed_cols;
        let mut acc = vdupq_n_f32(0.0);
        let full_bytes = cols / 4;

        // Process 4 values (1 packed byte) per iteration with NEON
        for b in 0..full_bytes {
            let byte = weights_packed[row_start + b];
            let c0 = (byte & 0x03) as usize;
            let c1 = ((byte >> 2) & 0x03) as usize;
            let c2 = ((byte >> 4) & 0x03) as usize;
            let c3 = ((byte >> 6) & 0x03) as usize;

            let w_arr = [lut[c0], lut[c1], lut[c2], lut[c3]];
            unsafe {
                let vw = vld1q_f32(w_arr.as_ptr());
                let va = vld1q_f32(a_ptr.add(b * 4));
                acc = vfmaq_f32(acc, vw, va);
            }
        }

        // Horizontal sum
        let mut sum = vaddvq_f32(acc);

        // Scalar tail for remaining columns
        let tail_start = full_bytes * 4;
        if tail_start < cols {
            let byte = weights_packed[row_start + full_bytes];
            for j in 0..(cols - tail_start) {
                let bits = (byte >> (j * 2)) & 0x03;
                sum += decode_i2s(bits) * activations[tail_start + j];
            }
        }

        *out_val = sum * scale;
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── scalar reference helpers ───────────────────────────────────────

    /// Scalar dequantize for oracle comparison.
    fn scalar_dequantize(packed: &[u8], count: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; count];
        for i in 0..count {
            let byte_idx = i / 4;
            let bit_off = (i % 4) * 2;
            let bits = (packed[byte_idx] >> bit_off) & 0x03;
            out[i] = decode_i2s(bits);
        }
        out
    }

    /// Scalar dot product for oracle comparison.
    fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Scalar GEMV for oracle comparison.
    fn scalar_gemv(
        weights_packed: &[u8],
        activations: &[f32],
        rows: usize,
        cols: usize,
        scale: f32,
    ) -> Vec<f32> {
        let packed_cols = cols.div_ceil(4);
        let mut output = vec![0.0f32; rows];
        for row in 0..rows {
            let row_start = row * packed_cols;
            let mut sum = 0.0f32;
            for col in 0..cols {
                let byte_idx = row_start + col / 4;
                let bit_off = (col % 4) * 2;
                let bits = (weights_packed[byte_idx] >> bit_off) & 0x03;
                sum += decode_i2s(bits) * activations[col];
            }
            output[row] = sum * scale;
        }
        output
    }

    /// Pack a row-major i8 ternary matrix into I2_S row-major bytes.
    fn pack_row_major(vals: &[i8], rows: usize, cols: usize) -> Vec<u8> {
        let packed_cols = cols.div_ceil(4);
        let mut packed = vec![0u8; rows * packed_cols];
        for row in 0..rows {
            for col in 0..cols {
                let v = vals[row * cols + col];
                let code: u8 = match v {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                let byte_idx = row * packed_cols + col / 4;
                let bit_off = (col % 4) * 2;
                packed[byte_idx] |= code << bit_off;
            }
        }
        packed
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (tol {tol})");
        }
    }

    // ── dequantize tests ──────────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dequantize_basic() {
        // Pack [+1, -1, 0, +1] → 0b01_00_11_01
        let packed = [0b01_00_11_01u8];
        let mut out = [0.0f32; 4];
        unsafe { neon_i2s_dequantize_block(&packed, &mut out) };
        assert_eq!(out, [1.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dequantize_all_zero() {
        let packed = [0x00u8; 2];
        let mut out = [999.0f32; 8];
        unsafe { neon_i2s_dequantize_block(&packed, &mut out) };
        assert_eq!(out, [0.0; 8]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dequantize_all_plus_one() {
        let packed = [0b01_01_01_01u8; 3];
        let mut out = [0.0f32; 12];
        unsafe { neon_i2s_dequantize_block(&packed, &mut out) };
        assert!(out.iter().all(|&v| v == 1.0));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dequantize_all_minus_one() {
        let packed = [0b11_11_11_11u8; 2];
        let mut out = [0.0f32; 8];
        unsafe { neon_i2s_dequantize_block(&packed, &mut out) };
        assert!(out.iter().all(|&v| v == -1.0));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dequantize_with_remainder() {
        // 5 values → 1 full byte + 1 remainder value
        let packed = [0b01_00_11_01u8, 0b00_00_00_11u8];
        let mut out = [0.0f32; 5];
        unsafe { neon_i2s_dequantize_block(&packed, &mut out) };
        assert_eq!(out, [1.0, -1.0, 0.0, 1.0, -1.0]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dequantize_roundtrip() {
        let original: [i8; 8] = [1, -1, 0, 1, -1, 0, 0, 1];
        let packed = pack_row_major(&original, 1, 8);
        let mut out = [0.0f32; 8];
        unsafe { neon_i2s_dequantize_block(&packed, &mut out) };
        let expected: Vec<f32> = original.iter().map(|&v| v as f32).collect();
        assert_eq!(out.as_slice(), expected.as_slice());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dequantize_vs_scalar() {
        let packed = [0b01_00_11_01u8, 0b11_01_00_11, 0b00_11_01_00];
        let count = 12;
        let expected = scalar_dequantize(&packed, count);
        let mut out = vec![0.0f32; count];
        unsafe { neon_i2s_dequantize_block(&packed, &mut out) };
        assert_close(&out, &expected, 0.0);
    }

    // ── dot product tests ─────────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dot_product_basic() {
        let w = [1.0f32, -1.0, 0.0, 1.0];
        let a = [2.0f32, 3.0, 4.0, 5.0];
        let result = unsafe { neon_i2s_dot_product(&w, &a) };
        // 1*2 + (-1)*3 + 0*4 + 1*5 = 2 - 3 + 0 + 5 = 4
        assert!((result - 4.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dot_product_zero_weights() {
        let w = [0.0f32; 8];
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = unsafe { neon_i2s_dot_product(&w, &a) };
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dot_product_single_element() {
        let w = [1.0f32];
        let a = [42.0f32];
        let result = unsafe { neon_i2s_dot_product(&w, &a) };
        assert!((result - 42.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dot_product_non_aligned() {
        // 7 elements — not a multiple of 4
        let w = [1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let a = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = unsafe { neon_i2s_dot_product(&w, &a) };
        // alternating +1/-1 with 7 elements → 1-1+1-1+1-1+1 = 1
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dot_product_vs_scalar() {
        let w: Vec<f32> = (0..33).map(|i| [1.0, -1.0, 0.0][i % 3]).collect();
        let a: Vec<f32> = (0..33).map(|i| (i as f32) * 0.1).collect();
        let expected = scalar_dot(&w, &a);
        let result = unsafe { neon_i2s_dot_product(&w, &a) };
        assert!((result - expected).abs() < 1e-4);
    }

    // ── GEMV tests ────────────────────────────────────────────────────

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemv_identity_like() {
        // 2×2 identity-like: row0=[+1,0], row1=[0,+1]
        let w: [i8; 4] = [1, 0, 0, 1];
        let packed = pack_row_major(&w, 2, 2);
        let act = [3.0f32, 7.0];
        let mut out = [0.0f32; 2];
        unsafe { neon_quantized_gemv(&packed, &act, &mut out, 2, 2, 1.0) };
        assert_close(&out, &[3.0, 7.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemv_all_ones() {
        let w = vec![1i8; 4 * 8]; // 4 rows × 8 cols
        let packed = pack_row_major(&w, 4, 8);
        let act: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let mut out = [0.0f32; 4];
        unsafe { neon_quantized_gemv(&packed, &act, &mut out, 4, 8, 1.0) };
        // Each row sums 1+2+...+8 = 36
        assert_close(&out, &[36.0, 36.0, 36.0, 36.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemv_with_scale() {
        let w = vec![1i8; 3 * 4];
        let packed = pack_row_major(&w, 3, 4);
        let act = [1.0f32, 1.0, 1.0, 1.0];
        let mut out = [0.0f32; 3];
        unsafe { neon_quantized_gemv(&packed, &act, &mut out, 3, 4, 2.5) };
        // Each row: sum=4, scaled=10.0
        assert_close(&out, &[10.0, 10.0, 10.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemv_zero_weights() {
        let w = vec![0i8; 4 * 4];
        let packed = pack_row_major(&w, 4, 4);
        let act = [42.0f32; 4];
        let mut out = [999.0f32; 4];
        unsafe { neon_quantized_gemv(&packed, &act, &mut out, 4, 4, 1.0) };
        assert_close(&out, &[0.0; 4], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemv_single_element() {
        let w = [1i8];
        let packed = pack_row_major(&w, 1, 1);
        let act = [5.0f32];
        let mut out = [0.0f32; 1];
        unsafe { neon_quantized_gemv(&packed, &act, &mut out, 1, 1, 3.0) };
        assert_close(&out, &[15.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemv_non_aligned_cols() {
        // 3 rows × 7 cols (not multiple of 4)
        let w: Vec<i8> = (0..21).map(|i| [1, -1, 0][i % 3]).collect();
        let packed = pack_row_major(&w, 3, 7);
        let act: Vec<f32> = (0..7).map(|i| (i as f32) + 1.0).collect();
        let mut out = [0.0f32; 3];
        let expected = scalar_gemv(&packed, &act, 3, 7, 1.0);
        unsafe { neon_quantized_gemv(&packed, &act, &mut out, 3, 7, 1.0) };
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemv_vs_scalar_medium() {
        let rows = 16;
        let cols = 33;
        let w: Vec<i8> = (0..rows * cols).map(|i| [1, -1, 0, 1, -1][i % 5]).collect();
        let packed = pack_row_major(&w, rows, cols);
        let act: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let expected = scalar_gemv(&packed, &act, rows, cols, 0.75);
        let mut out = vec![0.0f32; rows];
        unsafe { neon_quantized_gemv(&packed, &act, &mut out, rows, cols, 0.75) };
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_gemv_negative_scale() {
        let w = vec![1i8; 2 * 4];
        let packed = pack_row_major(&w, 2, 4);
        let act = [1.0f32; 4];
        let mut out = [0.0f32; 2];
        unsafe { neon_quantized_gemv(&packed, &act, &mut out, 2, 4, -1.0) };
        assert_close(&out, &[-4.0, -4.0], 1e-5);
    }

    // ── property tests ────────────────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Generate a vector of ternary values with length from a range.
        fn ternary_vec(
            len: impl Into<proptest::collection::SizeRange>,
        ) -> impl Strategy<Value = Vec<i8>> {
            proptest::collection::vec(prop_oneof![-1i8..=1], len)
        }

        proptest! {
            #[test]
            fn prop_dequantize_matches_scalar(
                vals in ternary_vec(1..=64),
            ) {
                let rows = 1;
                let cols = vals.len();
                let packed = pack_row_major(&vals, rows, cols);
                let expected = scalar_dequantize(&packed, cols);
                let mut out = vec![0.0f32; cols];
                unsafe { neon_i2s_dequantize_block(&packed, &mut out) };
                assert_close(&out, &expected, 0.0);
            }

            #[test]
            fn prop_dot_product_matches_scalar(
                len in 1usize..=64,
                seed in 0u64..1000,
            ) {
                let w: Vec<f32> = (0..len).map(|i| {
                    [1.0, -1.0, 0.0][(i + seed as usize) % 3]
                }).collect();
                let a: Vec<f32> = (0..len).map(|i| {
                    ((i as f64 + seed as f64) * 0.1).sin() as f32
                }).collect();
                let expected = scalar_dot(&w, &a);
                let result = unsafe { neon_i2s_dot_product(&w, &a) };
                prop_assert!((result - expected).abs() < 1e-3,
                    "neon={result}, scalar={expected}, diff={}",
                    (result - expected).abs());
            }

            #[test]
            fn prop_gemv_matches_scalar(
                rows in 1usize..=16,
                cols in 1usize..=32,
                seed in 0u64..1000,
            ) {
                let w: Vec<i8> = (0..rows * cols).map(|i| {
                    [-1, 0, 1][(i + seed as usize) % 3]
                }).collect();
                let packed = pack_row_major(&w, rows, cols);
                let act: Vec<f32> = (0..cols).map(|i| {
                    ((i as f64 + seed as f64) * 0.07).cos() as f32
                }).collect();
                let scale = 1.0 + (seed % 10) as f32 * 0.1;
                let expected = scalar_gemv(&packed, &act, rows, cols, scale);
                let mut out = vec![0.0f32; rows];
                unsafe { neon_quantized_gemv(&packed, &act, &mut out, rows, cols, scale) };
                for (i, (&e, &o)) in expected.iter().zip(out.iter()).enumerate() {
                    prop_assert!((e - o).abs() < 1e-3,
                        "row {i}: expected={e}, got={o}, diff={}",
                        (e - o).abs());
                }
            }
        }
    }
}
