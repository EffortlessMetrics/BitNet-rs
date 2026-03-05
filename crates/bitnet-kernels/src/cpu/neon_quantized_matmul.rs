//! NEON-optimized quantized matrix multiplication for I2_S ternary weights on Apple Silicon.
//!
//! Provides fused dequant+multiply kernels that operate directly on 2-bit packed I2_S
//! weight tensors, avoiding a separate dequantisation pass. All public functions require
//! the `neon` target feature and are gated behind `#[cfg(target_arch = "aarch64")]`.
//!
//! I2_S encoding (2 bits per value, 4 values per byte, LSB-first):
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1
//! - `0b10` → unused (treated as 0)

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::missing_safety_doc,
    clippy::float_cmp,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::let_and_return,
    clippy::derivable_impls,
    clippy::excessive_precision,
    clippy::manual_is_multiple_of
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Lookup table shared by all kernels ─────────────────────────────────

/// LUT: 2-bit code → f32 value.  Index by `(byte >> (shift)) & 0x03`.
const I2S_LUT: [f32; 4] = [0.0, 1.0, 0.0, -1.0];

// ── Helpers ────────────────────────────────────────────────────────────

/// Unpack a single byte into 4 ternary `i8` values (LSB-first).
///
/// Encoding: `0b00`→0, `0b01`→+1, `0b11`→−1, `0b10`→0 (unused).
#[inline(always)]
pub fn unpack_i2s_byte(byte: u8) -> [i8; 4] {
    let mut out = [0i8; 4];
    for i in 0..4 {
        let bits = (byte >> (i * 2)) & 0x03;
        out[i] = match bits {
            0b01 => 1,
            0b11 => -1,
            _ => 0,
        };
    }
    out
}

/// Decode a single 2-bit I2_S code to `f32`.
#[inline(always)]
fn decode_i2s(bits: u8) -> f32 {
    I2S_LUT[(bits & 0x03) as usize]
}

/// Number of packed bytes required to store `k` ternary values.
#[inline(always)]
fn packed_len(k: usize) -> usize {
    (k + 3) / 4
}

// ── Scalar reference (used for tail handling and tests) ────────────────

/// Scalar dot product of one packed weight row and an f32 input vector.
#[allow(dead_code)]
#[cfg(any(target_arch = "aarch64", test))]
fn scalar_row_dot(packed_row: &[u8], input: &[f32], k: usize) -> f32 {
    let mut sum = 0.0f32;
    for col in 0..k {
        let byte_idx = col / 4;
        let bit_off = (col % 4) * 2;
        let bits = (packed_row[byte_idx] >> bit_off) & 0x03;
        sum += decode_i2s(bits) * input[col];
    }
    sum
}

// ── NEON accumulation helper ───────────────────────────────────────────

/// Compute the dot product of `len` packed I2_S ternary values against `input` floats.
///
/// `packed` must contain at least `ceil(len/4)` bytes.
/// `input` must contain at least `len` floats.
///
/// Uses NEON FMA for the bulk of the work and a scalar tail for remainder.
///
/// # Safety
///
/// Requires the `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_accumulate_i2s_chunk(packed: &[u8], input: &[f32], len: usize) -> f32 {
    let full_bytes = len / 4;
    let remainder = len % 4;
    let in_ptr = input.as_ptr();

    let mut acc = vdupq_n_f32(0.0);

    for b in 0..full_bytes {
        let byte = packed[b];
        let c0 = (byte & 0x03) as usize;
        let c1 = ((byte >> 2) & 0x03) as usize;
        let c2 = ((byte >> 4) & 0x03) as usize;
        let c3 = ((byte >> 6) & 0x03) as usize;

        let w_arr = [I2S_LUT[c0], I2S_LUT[c1], I2S_LUT[c2], I2S_LUT[c3]];
        let vw = vld1q_f32(w_arr.as_ptr());
        let va = vld1q_f32(in_ptr.add(b * 4));
        acc = vfmaq_f32(acc, vw, va);
    }

    let mut sum = vaddvq_f32(acc);

    // Scalar tail
    if remainder > 0 {
        let byte = packed[full_bytes];
        let tail_start = full_bytes * 4;
        for j in 0..remainder {
            let bits = (byte >> (j * 2)) & 0x03;
            sum += decode_i2s(bits) * input[tail_start + j];
        }
    }

    sum
}

// ── Kernel: matrix-vector multiply ─────────────────────────────────────

/// Matrix-vector multiply with I2_S packed weights.
///
/// `weights_packed`: row-major packed weights, `m` rows × `k` columns (unpacked).
/// `input`: f32 vector of length `k`.
/// Returns f32 vector of length `m`.
///
/// # Safety
///
/// Requires the `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_matvec(
    weights_packed: &[u8],
    input: &[f32],
    m: usize,
    k: usize,
) -> Vec<f32> {
    let pk = packed_len(k);
    assert!(weights_packed.len() >= m * pk);
    assert!(input.len() >= k);

    let mut output = vec![0.0f32; m];
    for row in 0..m {
        let row_start = row * pk;
        let row_slice = &weights_packed[row_start..row_start + pk];
        output[row] = neon_accumulate_i2s_chunk(row_slice, input, k);
    }
    output
}

// ── Kernel: general matmul ─────────────────────────────────────────────

/// General matrix multiply: C = A_packed × B, where A is I2_S packed.
///
/// A: `m × k` (packed), B: `k × n` (f32 row-major), C: `m × n` (f32 row-major).
///
/// # Safety
///
/// Requires the `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_matmul(
    a_packed: &[u8],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let pk = packed_len(k);
    assert!(a_packed.len() >= m * pk);
    assert!(b.len() >= k * n);

    let mut output = vec![0.0f32; m * n];
    // Pre-allocate column gather buffer once, reused across all (row, col) iterations.
    let mut col_buf = vec![0.0f32; k];

    // For each output row, compute dot products with each column of B.
    // We iterate row-of-A × col-of-B to produce output[row, col].
    for row in 0..m {
        let a_row_start = row * pk;
        let a_row = &a_packed[a_row_start..a_row_start + pk];
        for col in 0..n {
            // Gather column `col` of B into the reused buffer for NEON.
            for r in 0..k {
                col_buf[r] = b[r * n + col];
            }
            output[row * n + col] = neon_accumulate_i2s_chunk(a_row, &col_buf, k);
        }
    }
    output
}

// ── Kernel: scaled matvec ──────────────────────────────────────────────

/// Matrix-vector multiply with per-row scales: `output[i] = scales[i] * (row_i · input)`.
///
/// # Safety
///
/// Requires the `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_matvec_scaled(
    weights_packed: &[u8],
    input: &[f32],
    scales: &[f32],
    m: usize,
    k: usize,
) -> Vec<f32> {
    let pk = packed_len(k);
    assert!(weights_packed.len() >= m * pk);
    assert!(input.len() >= k);
    assert!(scales.len() >= m);

    let mut output = vec![0.0f32; m];
    for row in 0..m {
        let row_start = row * pk;
        let row_slice = &weights_packed[row_start..row_start + pk];
        let dot = neon_accumulate_i2s_chunk(row_slice, input, k);
        output[row] = scales[row] * dot;
    }
    output
}

// ── Kernel: matvec with bias ───────────────────────────────────────────

/// Matrix-vector multiply with bias addition: `output[i] = (row_i · input) + bias[i]`.
///
/// # Safety
///
/// Requires the `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_matvec_bias(
    weights_packed: &[u8],
    input: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
) -> Vec<f32> {
    let pk = packed_len(k);
    assert!(weights_packed.len() >= m * pk);
    assert!(input.len() >= k);
    assert!(bias.len() >= m);

    let mut output = vec![0.0f32; m];
    for row in 0..m {
        let row_start = row * pk;
        let row_slice = &weights_packed[row_start..row_start + pk];
        let dot = neon_accumulate_i2s_chunk(row_slice, input, k);
        output[row] = dot + bias[row];
    }
    output
}

// ── Kernel: fused matvec + ReLU ────────────────────────────────────────

/// Matrix-vector multiply fused with ReLU: `output[i] = max(0, row_i · input)`.
///
/// # Safety
///
/// Requires the `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_matvec_fused_relu(
    weights_packed: &[u8],
    input: &[f32],
    m: usize,
    k: usize,
) -> Vec<f32> {
    let pk = packed_len(k);
    assert!(weights_packed.len() >= m * pk);
    assert!(input.len() >= k);

    let mut output = vec![0.0f32; m];
    for row in 0..m {
        let row_start = row * pk;
        let row_slice = &weights_packed[row_start..row_start + pk];
        let dot = neon_accumulate_i2s_chunk(row_slice, input, k);
        output[row] = if dot > 0.0 { dot } else { 0.0 };
    }
    output
}

// ── Kernel: batched matvec ─────────────────────────────────────────────

/// Batched matrix-vector multiply.
///
/// `inputs`: `batch × k` row-major f32 input vectors.
/// Returns: `batch × m` row-major f32 outputs.
/// Each batch element performs `W · x[b]` independently.
///
/// # Safety
///
/// Requires the `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_i2s_batch_matvec(
    weights_packed: &[u8],
    inputs: &[f32],
    batch: usize,
    m: usize,
    k: usize,
) -> Vec<f32> {
    let pk = packed_len(k);
    assert!(weights_packed.len() >= m * pk);
    assert!(inputs.len() >= batch * k);

    let mut output = vec![0.0f32; batch * m];
    for b in 0..batch {
        let in_start = b * k;
        let in_vec = &inputs[in_start..in_start + k];
        for row in 0..m {
            let row_start = row * pk;
            let row_slice = &weights_packed[row_start..row_start + pk];
            let dot = neon_accumulate_i2s_chunk(row_slice, in_vec, k);
            output[b * m + row] = dot;
        }
    }
    output
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Scalar reference helpers ───────────────────────────────────────

    /// Pack a row-major i8 ternary matrix into I2_S bytes.
    fn pack_row_major(vals: &[i8], rows: usize, cols: usize) -> Vec<u8> {
        let pk = packed_len(cols);
        let mut packed = vec![0u8; rows * pk];
        for row in 0..rows {
            for col in 0..cols {
                let v = vals[row * cols + col];
                let code: u8 = match v {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                let byte_idx = row * pk + col / 4;
                let bit_off = (col % 4) * 2;
                packed[byte_idx] |= code << bit_off;
            }
        }
        packed
    }

    /// Scalar matvec reference.
    fn scalar_matvec(packed: &[u8], input: &[f32], m: usize, k: usize) -> Vec<f32> {
        let pk = packed_len(k);
        (0..m).map(|row| scalar_row_dot(&packed[row * pk..row * pk + pk], input, k)).collect()
    }

    /// Scalar matmul reference.  A: m×k (packed), B: k×n (f32).
    fn scalar_matmul(a: &[u8], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let pk = packed_len(k);
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            let a_row = &a[row * pk..row * pk + pk];
            for col in 0..n {
                let mut sum = 0.0f32;
                for r in 0..k {
                    let byte_idx = r / 4;
                    let bit_off = (r % 4) * 2;
                    let bits = (a_row[byte_idx] >> bit_off) & 0x03;
                    sum += decode_i2s(bits) * b[r * n + col];
                }
                out[row * n + col] = sum;
            }
        }
        out
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (tol {tol})");
        }
    }

    /// Generate a deterministic f32 vector.
    fn gen_f32(len: usize, seed: usize) -> Vec<f32> {
        (0..len).map(|i| ((i + seed) as f32 * 0.37).sin()).collect()
    }

    /// Generate a deterministic ternary vector.
    fn gen_ternary(len: usize, seed: usize) -> Vec<i8> {
        (0..len).map(|i| [-1, 0, 1][(i + seed) % 3]).collect()
    }

    // ====================================================================
    // unpack_i2s_byte tests (1-10)
    // ====================================================================

    #[test]
    fn test_unpack_all_zero() {
        assert_eq!(unpack_i2s_byte(0x00), [0, 0, 0, 0]);
    }

    #[test]
    fn test_unpack_all_plus_one() {
        // 0b01_01_01_01 = 0x55
        assert_eq!(unpack_i2s_byte(0x55), [1, 1, 1, 1]);
    }

    #[test]
    fn test_unpack_all_minus_one() {
        // 0b11_11_11_11 = 0xFF
        assert_eq!(unpack_i2s_byte(0xFF), [-1, -1, -1, -1]);
    }

    #[test]
    fn test_unpack_mixed_pattern_a() {
        // byte: 0b01_00_11_01 → [+1, −1, 0, +1]
        assert_eq!(unpack_i2s_byte(0b01_00_11_01), [1, -1, 0, 1]);
    }

    #[test]
    fn test_unpack_mixed_pattern_b() {
        // byte: 0b11_01_00_11 → [−1, 0, +1, −1]
        assert_eq!(unpack_i2s_byte(0b11_01_00_11), [-1, 0, 1, -1]);
    }

    #[test]
    fn test_unpack_unused_code() {
        // 0b10 = unused → 0
        assert_eq!(unpack_i2s_byte(0b10_10_10_10), [0, 0, 0, 0]);
    }

    #[test]
    fn test_unpack_single_plus_one_slot0() {
        assert_eq!(unpack_i2s_byte(0b00_00_00_01), [1, 0, 0, 0]);
    }

    #[test]
    fn test_unpack_single_minus_one_slot3() {
        assert_eq!(unpack_i2s_byte(0b11_00_00_00), [0, 0, 0, -1]);
    }

    #[test]
    fn test_unpack_alternating_plus_minus() {
        // +1, −1, +1, −1 → 0b11_01_11_01
        assert_eq!(unpack_i2s_byte(0b11_01_11_01), [1, -1, 1, -1]);
    }

    #[test]
    fn test_unpack_alternating_minus_plus() {
        // −1, +1, −1, +1 → 0b01_11_01_11
        assert_eq!(unpack_i2s_byte(0b01_11_01_11), [-1, 1, -1, 1]);
    }

    // ====================================================================
    // neon_accumulate_i2s_chunk tests (11-20)
    // ====================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_accum_all_zeros_weights() {
        let packed = [0x00u8; 2];
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = unsafe { neon_accumulate_i2s_chunk(&packed, &input, 8) };
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_accum_all_plus_one_weights() {
        // All +1 → sum of inputs
        let packed = [0x55u8; 2]; // 0b01_01_01_01
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = unsafe { neon_accumulate_i2s_chunk(&packed, &input, 8) };
        assert!((result - 36.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_accum_all_minus_one_weights() {
        let packed = [0xFFu8; 2];
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = unsafe { neon_accumulate_i2s_chunk(&packed, &input, 8) };
        assert!((result - (-36.0)).abs() < 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_accum_single_element() {
        let packed = [0b01u8]; // +1
        let input = [42.0f32];
        let result = unsafe { neon_accumulate_i2s_chunk(&packed, &input, 1) };
        assert!((result - 42.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_accum_non_aligned_len() {
        // 5 values
        let vals: Vec<i8> = vec![1, -1, 1, 0, -1];
        let packed = pack_row_major(&vals, 1, 5);
        let input = [2.0f32, 3.0, 4.0, 5.0, 6.0];
        let result = unsafe { neon_accumulate_i2s_chunk(&packed, &input, 5) };
        // 1*2 + (-1)*3 + 1*4 + 0*5 + (-1)*6 = 2 - 3 + 4 + 0 - 6 = -3
        assert!((result - (-3.0)).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_accum_matches_scalar() {
        let vals = gen_ternary(33, 0);
        let packed = pack_row_major(&vals, 1, 33);
        let input = gen_f32(33, 7);
        let neon_result = unsafe { neon_accumulate_i2s_chunk(&packed, &input, 33) };
        let scalar_result = scalar_row_dot(&packed, &input, 33);
        assert!((neon_result - scalar_result).abs() < 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_accum_exactly_4_elements() {
        let vals: Vec<i8> = vec![1, -1, 1, -1];
        let packed = pack_row_major(&vals, 1, 4);
        let input = [1.0f32, 1.0, 1.0, 1.0];
        let result = unsafe { neon_accumulate_i2s_chunk(&packed, &input, 4) };
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_accum_exactly_8_elements() {
        let vals: Vec<i8> = vec![1; 8];
        let packed = pack_row_major(&vals, 1, 8);
        let input = [1.0f32; 8];
        let result = unsafe { neon_accumulate_i2s_chunk(&packed, &input, 8) };
        assert!((result - 8.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_accum_large_k_64() {
        let vals = gen_ternary(64, 42);
        let packed = pack_row_major(&vals, 1, 64);
        let input = gen_f32(64, 99);
        let neon_result = unsafe { neon_accumulate_i2s_chunk(&packed, &input, 64) };
        let scalar_result = scalar_row_dot(&packed, &input, 64);
        assert!((neon_result - scalar_result).abs() < 1e-3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_accum_large_k_257() {
        let vals = gen_ternary(257, 13);
        let packed = pack_row_major(&vals, 1, 257);
        let input = gen_f32(257, 21);
        let neon_result = unsafe { neon_accumulate_i2s_chunk(&packed, &input, 257) };
        let scalar_result = scalar_row_dot(&packed, &input, 257);
        assert!((neon_result - scalar_result).abs() < 1e-2);
    }

    // ====================================================================
    // neon_i2s_matvec tests (21-40)
    // ====================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_1x1() {
        let packed = pack_row_major(&[1i8], 1, 1);
        let input = [7.0f32];
        let out = unsafe { neon_i2s_matvec(&packed, &input, 1, 1) };
        assert_close(&out, &[7.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_1x1_minus() {
        let packed = pack_row_major(&[-1i8], 1, 1);
        let input = [5.0f32];
        let out = unsafe { neon_i2s_matvec(&packed, &input, 1, 1) };
        assert_close(&out, &[-5.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_1x1_zero() {
        let packed = pack_row_major(&[0i8], 1, 1);
        let input = [99.0f32];
        let out = unsafe { neon_i2s_matvec(&packed, &input, 1, 1) };
        assert_close(&out, &[0.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_identity_2x2() {
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let packed = pack_row_major(&w, 2, 2);
        let input = [3.0f32, 7.0];
        let out = unsafe { neon_i2s_matvec(&packed, &input, 2, 2) };
        assert_close(&out, &[3.0, 7.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_identity_4x4() {
        let mut w = vec![0i8; 16];
        for i in 0..4 {
            w[i * 4 + i] = 1;
        }
        let packed = pack_row_major(&w, 4, 4);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let out = unsafe { neon_i2s_matvec(&packed, &input, 4, 4) };
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_all_ones_4x8() {
        let w = vec![1i8; 4 * 8];
        let packed = pack_row_major(&w, 4, 8);
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let out = unsafe { neon_i2s_matvec(&packed, &input, 4, 8) };
        assert_close(&out, &[36.0, 36.0, 36.0, 36.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_all_minus_ones() {
        let w = vec![-1i8; 3 * 4];
        let packed = pack_row_major(&w, 3, 4);
        let input = [1.0f32, 1.0, 1.0, 1.0];
        let out = unsafe { neon_i2s_matvec(&packed, &input, 3, 4) };
        assert_close(&out, &[-4.0, -4.0, -4.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_all_zero_weights() {
        let w = vec![0i8; 4 * 4];
        let packed = pack_row_major(&w, 4, 4);
        let input = [42.0f32; 4];
        let out = unsafe { neon_i2s_matvec(&packed, &input, 4, 4) };
        assert_close(&out, &[0.0; 4], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_non_aligned_3x7() {
        let w = gen_ternary(3 * 7, 0);
        let packed = pack_row_major(&w, 3, 7);
        let input = gen_f32(7, 1);
        let expected = scalar_matvec(&packed, &input, 3, 7);
        let out = unsafe { neon_i2s_matvec(&packed, &input, 3, 7) };
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_non_aligned_5x13() {
        let w = gen_ternary(5 * 13, 5);
        let packed = pack_row_major(&w, 5, 13);
        let input = gen_f32(13, 3);
        let expected = scalar_matvec(&packed, &input, 5, 13);
        let out = unsafe { neon_i2s_matvec(&packed, &input, 5, 13) };
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_32x32() {
        let w = gen_ternary(32 * 32, 10);
        let packed = pack_row_major(&w, 32, 32);
        let input = gen_f32(32, 20);
        let expected = scalar_matvec(&packed, &input, 32, 32);
        let out = unsafe { neon_i2s_matvec(&packed, &input, 32, 32) };
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_128x512() {
        let m = 128;
        let k = 512;
        let w = gen_ternary(m * k, 42);
        let packed = pack_row_major(&w, m, k);
        let input = gen_f32(k, 77);
        let expected = scalar_matvec(&packed, &input, m, k);
        let out = unsafe { neon_i2s_matvec(&packed, &input, m, k) };
        assert_close(&out, &expected, 5e-2);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_2048x768() {
        let m = 2048;
        let k = 768;
        let w = gen_ternary(m * k, 99);
        let packed = pack_row_major(&w, m, k);
        let input = gen_f32(k, 33);
        let expected = scalar_matvec(&packed, &input, m, k);
        let out = unsafe { neon_i2s_matvec(&packed, &input, m, k) };
        // Larger dimensions accumulate more FP error
        for (i, (&e, &o)) in expected.iter().zip(out.iter()).enumerate() {
            assert!((e - o).abs() < 0.5, "row {i}: expected={e}, got={o}, diff={}", (e - o).abs());
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_single_row_large() {
        let k = 1024;
        let w = gen_ternary(k, 7);
        let packed = pack_row_major(&w, 1, k);
        let input = gen_f32(k, 11);
        let expected = scalar_matvec(&packed, &input, 1, k);
        let out = unsafe { neon_i2s_matvec(&packed, &input, 1, k) };
        assert!((out[0] - expected[0]).abs() < 0.1);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_mixed_pattern_16x16() {
        let w = gen_ternary(16 * 16, 55);
        let packed = pack_row_major(&w, 16, 16);
        let input = gen_f32(16, 8);
        let expected = scalar_matvec(&packed, &input, 16, 16);
        let out = unsafe { neon_i2s_matvec(&packed, &input, 16, 16) };
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_unit_input() {
        // All-1s input: each output = sum of ternary row
        let w: Vec<i8> = vec![1, 1, -1, 0, -1, 1, 1, 1]; // row0: sum=2, row1: sum=2
        let packed = pack_row_major(&w, 2, 4);
        let input = [1.0f32; 4];
        let out = unsafe { neon_i2s_matvec(&packed, &input, 2, 4) };
        // row0: 1+1+(-1)+0 = 1, row1: -1+1+1+1 = 2
        let expected = scalar_matvec(&packed, &input, 2, 4);
        assert_close(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_negative_inputs() {
        let w = vec![1i8; 2 * 4];
        let packed = pack_row_major(&w, 2, 4);
        let input = [-1.0f32, -2.0, -3.0, -4.0];
        let out = unsafe { neon_i2s_matvec(&packed, &input, 2, 4) };
        assert_close(&out, &[-10.0, -10.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_k_equals_1() {
        let w: Vec<i8> = vec![1, -1, 0];
        let packed = pack_row_major(&w, 3, 1);
        let input = [5.0f32];
        let out = unsafe { neon_i2s_matvec(&packed, &input, 3, 1) };
        assert_close(&out, &[5.0, -5.0, 0.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_k_equals_3() {
        let w: Vec<i8> = vec![1, -1, 1, -1, 1, -1];
        let packed = pack_row_major(&w, 2, 3);
        let input = [10.0f32, 20.0, 30.0];
        let out = unsafe { neon_i2s_matvec(&packed, &input, 2, 3) };
        // row0: 10 - 20 + 30 = 20, row1: -10 + 20 - 30 = -20
        assert_close(&out, &[20.0, -20.0], 1e-5);
    }

    // ====================================================================
    // neon_i2s_matmul tests (41-55)
    // ====================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_1x1x1() {
        let a_packed = pack_row_major(&[1i8], 1, 1);
        let b = [3.0f32];
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, 1, 1, 1) };
        assert_close(&out, &[3.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_identity_2x2() {
        let a: Vec<i8> = vec![1, 0, 0, 1];
        let a_packed = pack_row_major(&a, 2, 2);
        let b = [1.0f32, 2.0, 3.0, 4.0]; // 2×2 row-major
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, 2, 2, 2) };
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_2x3x2() {
        // A: 2×3 all +1, B: 3×2
        let a = vec![1i8; 6];
        let a_packed = pack_row_major(&a, 2, 3);
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3×2
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, 2, 2, 3) };
        // Each row of A sums columns of B: [1+3+5, 2+4+6] = [9, 12]
        assert_close(&out, &[9.0, 12.0, 9.0, 12.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_all_zero_a() {
        let a = vec![0i8; 4 * 4];
        let a_packed = pack_row_major(&a, 4, 4);
        let b = vec![42.0f32; 4 * 4];
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, 4, 4, 4) };
        assert_close(&out, &vec![0.0f32; 16], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_4x4x4() {
        let a = gen_ternary(16, 0);
        let a_packed = pack_row_major(&a, 4, 4);
        let b = gen_f32(16, 5);
        let expected = scalar_matmul(&a_packed, &b, 4, 4, 4);
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, 4, 4, 4) };
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_non_square_3x5x7() {
        let a = gen_ternary(3 * 7, 1);
        let a_packed = pack_row_major(&a, 3, 7);
        let b = gen_f32(7 * 5, 2);
        let expected = scalar_matmul(&a_packed, &b, 3, 5, 7);
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, 3, 5, 7) };
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_m1_n4_k8() {
        let a = gen_ternary(8, 10);
        let a_packed = pack_row_major(&a, 1, 8);
        let b = gen_f32(8 * 4, 20);
        let expected = scalar_matmul(&a_packed, &b, 1, 4, 8);
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, 1, 4, 8) };
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_all_minus_one_a() {
        let a = vec![-1i8; 2 * 3];
        let a_packed = pack_row_major(&a, 2, 3);
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3×2
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, 2, 2, 3) };
        // Each row: -(1+3+5), -(2+4+6) = [-9, -12]
        assert_close(&out, &[-9.0, -12.0, -9.0, -12.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_single_column_b() {
        let a = gen_ternary(4 * 4, 7);
        let a_packed = pack_row_major(&a, 4, 4);
        let b = gen_f32(4, 11); // 4×1
        let expected = scalar_matmul(&a_packed, &b, 4, 1, 4);
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, 4, 1, 4) };
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_16x8x16() {
        let m = 16;
        let k = 8;
        let n = 16;
        let a = gen_ternary(m * k, 50);
        let a_packed = pack_row_major(&a, m, k);
        let b = gen_f32(k * n, 60);
        let expected = scalar_matmul(&a_packed, &b, m, n, k);
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, m, n, k) };
        assert_close(&out, &expected, 1e-2);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_vs_matvec() {
        // matmul with n=1 should match matvec
        let m = 8;
        let k = 16;
        let w = gen_ternary(m * k, 3);
        let packed = pack_row_major(&w, m, k);
        let input = gen_f32(k, 5);
        let matvec_out = unsafe { neon_i2s_matvec(&packed, &input, m, k) };
        let matmul_out = unsafe { neon_i2s_matmul(&packed, &input, m, 1, k) };
        assert_close(&matvec_out, &matmul_out, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_32x32_medium() {
        let m = 32;
        let k = 32;
        let n = 32;
        let a = gen_ternary(m * k, 100);
        let a_packed = pack_row_major(&a, m, k);
        let b = gen_f32(k * n, 200);
        let expected = scalar_matmul(&a_packed, &b, m, n, k);
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, m, n, k) };
        assert_close(&out, &expected, 5e-2);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_negative_b_values() {
        let a = vec![1i8; 2 * 2];
        let a_packed = pack_row_major(&a, 2, 2);
        let b = [-1.0f32, -2.0, -3.0, -4.0];
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, 2, 2, 2) };
        // row0: (-1)+(-3)=-4, (-2)+(-4)=-6
        assert_close(&out, &[-4.0, -6.0, -4.0, -6.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_non_aligned_5x3x7() {
        let a = gen_ternary(5 * 7, 0);
        let a_packed = pack_row_major(&a, 5, 7);
        let b = gen_f32(7 * 3, 1);
        let expected = scalar_matmul(&a_packed, &b, 5, 3, 7);
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, 5, 3, 7) };
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matmul_k1() {
        // k=1: each output is a[row]*b[col]
        let a: Vec<i8> = vec![1, -1, 0];
        let a_packed = pack_row_major(&a, 3, 1);
        let b = [2.0f32, 3.0]; // 1×2
        let out = unsafe { neon_i2s_matmul(&a_packed, &b, 3, 2, 1) };
        assert_close(&out, &[2.0, 3.0, -2.0, -3.0, 0.0, 0.0], 1e-5);
    }

    // ====================================================================
    // neon_i2s_matvec_scaled tests (56-65)
    // ====================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scaled_uniform_scale() {
        let w = vec![1i8; 2 * 4];
        let packed = pack_row_major(&w, 2, 4);
        let input = [1.0f32; 4];
        let scales = [2.0f32, 2.0];
        let out = unsafe { neon_i2s_matvec_scaled(&packed, &input, &scales, 2, 4) };
        // dot=4 each, scale=2 → 8
        assert_close(&out, &[8.0, 8.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scaled_varying_scales() {
        let w = vec![1i8; 3 * 4];
        let packed = pack_row_major(&w, 3, 4);
        let input = [1.0f32; 4];
        let scales = [1.0f32, 0.5, -1.0];
        let out = unsafe { neon_i2s_matvec_scaled(&packed, &input, &scales, 3, 4) };
        assert_close(&out, &[4.0, 2.0, -4.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scaled_zero_scale() {
        let w = vec![1i8; 2 * 4];
        let packed = pack_row_major(&w, 2, 4);
        let input = [100.0f32; 4];
        let scales = [0.0f32, 0.0];
        let out = unsafe { neon_i2s_matvec_scaled(&packed, &input, &scales, 2, 4) };
        assert_close(&out, &[0.0, 0.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scaled_identity_scale() {
        let w = gen_ternary(4 * 8, 0);
        let packed = pack_row_major(&w, 4, 8);
        let input = gen_f32(8, 1);
        let scales = [1.0f32; 4];
        let scaled_out = unsafe { neon_i2s_matvec_scaled(&packed, &input, &scales, 4, 8) };
        let plain_out = unsafe { neon_i2s_matvec(&packed, &input, 4, 8) };
        assert_close(&scaled_out, &plain_out, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scaled_non_aligned_k() {
        let w = gen_ternary(3 * 5, 2);
        let packed = pack_row_major(&w, 3, 5);
        let input = gen_f32(5, 3);
        let scales = [0.5f32, 1.5, 2.0];
        let plain = unsafe { neon_i2s_matvec(&packed, &input, 3, 5) };
        let out = unsafe { neon_i2s_matvec_scaled(&packed, &input, &scales, 3, 5) };
        let expected: Vec<f32> = plain.iter().zip(scales.iter()).map(|(d, s)| d * s).collect();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scaled_large_32x32() {
        let m = 32;
        let k = 32;
        let w = gen_ternary(m * k, 5);
        let packed = pack_row_major(&w, m, k);
        let input = gen_f32(k, 7);
        let scales: Vec<f32> = (0..m).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let plain = unsafe { neon_i2s_matvec(&packed, &input, m, k) };
        let out = unsafe { neon_i2s_matvec_scaled(&packed, &input, &scales, m, k) };
        let expected: Vec<f32> = plain.iter().zip(scales.iter()).map(|(d, s)| d * s).collect();
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scaled_negative_scale() {
        let w = vec![1i8; 1 * 4];
        let packed = pack_row_major(&w, 1, 4);
        let input = [1.0f32; 4];
        let scales = [-3.0f32];
        let out = unsafe { neon_i2s_matvec_scaled(&packed, &input, &scales, 1, 4) };
        assert_close(&out, &[-12.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scaled_fractional() {
        let w = vec![1i8; 2 * 8];
        let packed = pack_row_major(&w, 2, 8);
        let input = [1.0f32; 8];
        let scales = [0.125f32, 0.25];
        let out = unsafe { neon_i2s_matvec_scaled(&packed, &input, &scales, 2, 8) };
        // dot=8, scaled: 1.0, 2.0
        assert_close(&out, &[1.0, 2.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scaled_single_row() {
        let w = gen_ternary(128, 0);
        let packed = pack_row_major(&w, 1, 128);
        let input = gen_f32(128, 1);
        let scales = [0.01f32];
        let plain = unsafe { neon_i2s_matvec(&packed, &input, 1, 128) };
        let out = unsafe { neon_i2s_matvec_scaled(&packed, &input, &scales, 1, 128) };
        assert!((out[0] - plain[0] * 0.01).abs() < 1e-3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scaled_all_zero_weights() {
        let w = vec![0i8; 3 * 4];
        let packed = pack_row_major(&w, 3, 4);
        let input = [99.0f32; 4];
        let scales = [100.0f32, 200.0, 300.0];
        let out = unsafe { neon_i2s_matvec_scaled(&packed, &input, &scales, 3, 4) };
        assert_close(&out, &[0.0, 0.0, 0.0], 1e-6);
    }

    // ====================================================================
    // neon_i2s_matvec_bias tests (66-75)
    // ====================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_bias_basic() {
        let w = vec![1i8; 2 * 4];
        let packed = pack_row_major(&w, 2, 4);
        let input = [1.0f32; 4];
        let bias = [10.0f32, 20.0];
        let out = unsafe { neon_i2s_matvec_bias(&packed, &input, &bias, 2, 4) };
        assert_close(&out, &[14.0, 24.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_bias_zero_bias() {
        let w = gen_ternary(3 * 8, 0);
        let packed = pack_row_major(&w, 3, 8);
        let input = gen_f32(8, 1);
        let bias = [0.0f32; 3];
        let biased = unsafe { neon_i2s_matvec_bias(&packed, &input, &bias, 3, 8) };
        let plain = unsafe { neon_i2s_matvec(&packed, &input, 3, 8) };
        assert_close(&biased, &plain, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_bias_negative() {
        let w = vec![1i8; 1 * 4];
        let packed = pack_row_major(&w, 1, 4);
        let input = [1.0f32; 4];
        let bias = [-100.0f32];
        let out = unsafe { neon_i2s_matvec_bias(&packed, &input, &bias, 1, 4) };
        assert_close(&out, &[-96.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_bias_zero_weights() {
        let w = vec![0i8; 2 * 4];
        let packed = pack_row_major(&w, 2, 4);
        let input = [1.0f32; 4];
        let bias = [5.0f32, 7.0];
        let out = unsafe { neon_i2s_matvec_bias(&packed, &input, &bias, 2, 4) };
        // dot=0, so output = bias
        assert_close(&out, &[5.0, 7.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_bias_non_aligned_k() {
        let w = gen_ternary(3 * 5, 0);
        let packed = pack_row_major(&w, 3, 5);
        let input = gen_f32(5, 1);
        let bias = [1.0f32, 2.0, 3.0];
        let plain = unsafe { neon_i2s_matvec(&packed, &input, 3, 5) };
        let out = unsafe { neon_i2s_matvec_bias(&packed, &input, &bias, 3, 5) };
        let expected: Vec<f32> = plain.iter().zip(bias.iter()).map(|(d, b)| d + b).collect();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_bias_large_32x32() {
        let m = 32;
        let k = 32;
        let w = gen_ternary(m * k, 5);
        let packed = pack_row_major(&w, m, k);
        let input = gen_f32(k, 7);
        let bias: Vec<f32> = (0..m).map(|i| i as f32 * 0.5).collect();
        let plain = unsafe { neon_i2s_matvec(&packed, &input, m, k) };
        let out = unsafe { neon_i2s_matvec_bias(&packed, &input, &bias, m, k) };
        let expected: Vec<f32> = plain.iter().zip(bias.iter()).map(|(d, b)| d + b).collect();
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_bias_single_element() {
        let packed = pack_row_major(&[1i8], 1, 1);
        let input = [3.0f32];
        let bias = [10.0f32];
        let out = unsafe { neon_i2s_matvec_bias(&packed, &input, &bias, 1, 1) };
        assert_close(&out, &[13.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_bias_large_bias_values() {
        let w = vec![1i8; 2 * 4];
        let packed = pack_row_major(&w, 2, 4);
        let input = [0.001f32; 4];
        let bias = [1e6f32, -1e6];
        let out = unsafe { neon_i2s_matvec_bias(&packed, &input, &bias, 2, 4) };
        assert!((out[0] - (0.004 + 1e6)).abs() < 1.0);
        assert!((out[1] - (0.004 - 1e6)).abs() < 1.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_bias_cancellation() {
        // bias exactly cancels the dot product
        let w = vec![1i8; 1 * 4];
        let packed = pack_row_major(&w, 1, 4);
        let input = [1.0f32; 4];
        let bias = [-4.0f32];
        let out = unsafe { neon_i2s_matvec_bias(&packed, &input, &bias, 1, 4) };
        assert_close(&out, &[0.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_bias_matches_manual() {
        let m = 8;
        let k = 16;
        let w = gen_ternary(m * k, 9);
        let packed = pack_row_major(&w, m, k);
        let input = gen_f32(k, 11);
        let bias = gen_f32(m, 13);
        let plain = unsafe { neon_i2s_matvec(&packed, &input, m, k) };
        let out = unsafe { neon_i2s_matvec_bias(&packed, &input, &bias, m, k) };
        let expected: Vec<f32> = plain.iter().zip(bias.iter()).map(|(d, b)| d + b).collect();
        assert_close(&out, &expected, 1e-4);
    }

    // ====================================================================
    // neon_i2s_matvec_fused_relu tests (76-85)
    // ====================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_relu_positive_pass_through() {
        let w = vec![1i8; 2 * 4];
        let packed = pack_row_major(&w, 2, 4);
        let input = [1.0f32; 4];
        let out = unsafe { neon_i2s_matvec_fused_relu(&packed, &input, 2, 4) };
        // dot=4 > 0 → 4
        assert_close(&out, &[4.0, 4.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_relu_negative_clamp() {
        let w = vec![-1i8; 2 * 4];
        let packed = pack_row_major(&w, 2, 4);
        let input = [1.0f32; 4];
        let out = unsafe { neon_i2s_matvec_fused_relu(&packed, &input, 2, 4) };
        // dot=-4 → clamped to 0
        assert_close(&out, &[0.0, 0.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_relu_zero_through() {
        let w = vec![0i8; 2 * 4];
        let packed = pack_row_major(&w, 2, 4);
        let input = [1.0f32; 4];
        let out = unsafe { neon_i2s_matvec_fused_relu(&packed, &input, 2, 4) };
        assert_close(&out, &[0.0, 0.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_relu_mixed_signs() {
        // row0: +1,+1,+1,+1 → dot=4 → 4
        // row1: -1,-1,-1,-1 → dot=-4 → 0
        let w: Vec<i8> = vec![1, 1, 1, 1, -1, -1, -1, -1];
        let packed = pack_row_major(&w, 2, 4);
        let input = [1.0f32; 4];
        let out = unsafe { neon_i2s_matvec_fused_relu(&packed, &input, 2, 4) };
        assert_close(&out, &[4.0, 0.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_relu_non_aligned() {
        let w = gen_ternary(3 * 5, 0);
        let packed = pack_row_major(&w, 3, 5);
        let input = gen_f32(5, 1);
        let plain = unsafe { neon_i2s_matvec(&packed, &input, 3, 5) };
        let out = unsafe { neon_i2s_matvec_fused_relu(&packed, &input, 3, 5) };
        let expected: Vec<f32> = plain.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_relu_vs_manual() {
        let m = 16;
        let k = 32;
        let w = gen_ternary(m * k, 7);
        let packed = pack_row_major(&w, m, k);
        let input = gen_f32(k, 3);
        let plain = unsafe { neon_i2s_matvec(&packed, &input, m, k) };
        let out = unsafe { neon_i2s_matvec_fused_relu(&packed, &input, m, k) };
        let expected: Vec<f32> = plain.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect();
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_relu_single_element_positive() {
        let packed = pack_row_major(&[1i8], 1, 1);
        let input = [5.0f32];
        let out = unsafe { neon_i2s_matvec_fused_relu(&packed, &input, 1, 1) };
        assert_close(&out, &[5.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_relu_single_element_negative() {
        let packed = pack_row_major(&[-1i8], 1, 1);
        let input = [5.0f32];
        let out = unsafe { neon_i2s_matvec_fused_relu(&packed, &input, 1, 1) };
        assert_close(&out, &[0.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_relu_large_32x32() {
        let m = 32;
        let k = 32;
        let w = gen_ternary(m * k, 50);
        let packed = pack_row_major(&w, m, k);
        let input = gen_f32(k, 60);
        let plain = unsafe { neon_i2s_matvec(&packed, &input, m, k) };
        let out = unsafe { neon_i2s_matvec_fused_relu(&packed, &input, m, k) };
        for (i, (&p, &o)) in plain.iter().zip(out.iter()).enumerate() {
            let expected = if p > 0.0 { p } else { 0.0 };
            assert!((expected - o).abs() < 1e-3, "row {i}: expected={expected}, got={o}");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_relu_all_output_non_negative() {
        let m = 64;
        let k = 64;
        let w = gen_ternary(m * k, 0);
        let packed = pack_row_major(&w, m, k);
        let input = gen_f32(k, 1);
        let out = unsafe { neon_i2s_matvec_fused_relu(&packed, &input, m, k) };
        for (i, &v) in out.iter().enumerate() {
            assert!(v >= 0.0, "row {i}: got negative {v}");
        }
    }

    // ====================================================================
    // neon_i2s_batch_matvec tests (86-95)
    // ====================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batch_single_batch() {
        let w = vec![1i8; 2 * 4];
        let packed = pack_row_major(&w, 2, 4);
        let input = [1.0f32; 4];
        let out = unsafe { neon_i2s_batch_matvec(&packed, &input, 1, 2, 4) };
        let expected = unsafe { neon_i2s_matvec(&packed, &input, 2, 4) };
        assert_close(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batch_two_batches() {
        let w = vec![1i8; 2 * 4];
        let packed = pack_row_major(&w, 2, 4);
        let inputs = [1.0f32, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]; // batch=2, k=4
        let out = unsafe { neon_i2s_batch_matvec(&packed, &inputs, 2, 2, 4) };
        // batch0: each row dot=4, batch1: each row dot=8
        assert_close(&out, &[4.0, 4.0, 8.0, 8.0], 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batch_matches_individual() {
        let m = 4;
        let k = 8;
        let batch = 3;
        let w = gen_ternary(m * k, 0);
        let packed = pack_row_major(&w, m, k);
        let inputs = gen_f32(batch * k, 1);
        let batched = unsafe { neon_i2s_batch_matvec(&packed, &inputs, batch, m, k) };
        for b in 0..batch {
            let single = unsafe { neon_i2s_matvec(&packed, &inputs[b * k..(b + 1) * k], m, k) };
            assert_close(&batched[b * m..(b + 1) * m], &single, 1e-5);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batch_zero_weights() {
        let w = vec![0i8; 3 * 4];
        let packed = pack_row_major(&w, 3, 4);
        let inputs = [99.0f32; 2 * 4]; // batch=2
        let out = unsafe { neon_i2s_batch_matvec(&packed, &inputs, 2, 3, 4) };
        assert_close(&out, &[0.0; 6], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batch_non_aligned_k() {
        let m = 3;
        let k = 5;
        let batch = 2;
        let w = gen_ternary(m * k, 7);
        let packed = pack_row_major(&w, m, k);
        let inputs = gen_f32(batch * k, 3);
        let batched = unsafe { neon_i2s_batch_matvec(&packed, &inputs, batch, m, k) };
        for b in 0..batch {
            let single = unsafe { neon_i2s_matvec(&packed, &inputs[b * k..(b + 1) * k], m, k) };
            assert_close(&batched[b * m..(b + 1) * m], &single, 1e-5);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batch_large() {
        let m = 16;
        let k = 32;
        let batch = 8;
        let w = gen_ternary(m * k, 99);
        let packed = pack_row_major(&w, m, k);
        let inputs = gen_f32(batch * k, 42);
        let batched = unsafe { neon_i2s_batch_matvec(&packed, &inputs, batch, m, k) };
        assert_eq!(batched.len(), batch * m);
        for b in 0..batch {
            let single = unsafe { neon_i2s_matvec(&packed, &inputs[b * k..(b + 1) * k], m, k) };
            assert_close(&batched[b * m..(b + 1) * m], &single, 1e-3);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batch_1x1() {
        let packed = pack_row_major(&[1i8], 1, 1);
        let inputs = [3.0f32, 5.0, 7.0]; // batch=3, k=1
        let out = unsafe { neon_i2s_batch_matvec(&packed, &inputs, 3, 1, 1) };
        assert_close(&out, &[3.0, 5.0, 7.0], 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batch_identical_inputs() {
        let m = 4;
        let k = 8;
        let w = gen_ternary(m * k, 0);
        let packed = pack_row_major(&w, m, k);
        let single_input = gen_f32(k, 1);
        let mut inputs = Vec::new();
        for _ in 0..4 {
            inputs.extend_from_slice(&single_input);
        }
        let out = unsafe { neon_i2s_batch_matvec(&packed, &inputs, 4, m, k) };
        // All batch results should be identical
        let first_batch = &out[0..m];
        for b in 1..4 {
            assert_close(&out[b * m..(b + 1) * m], first_batch, 1e-6);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batch_output_length() {
        let m = 5;
        let k = 10;
        let batch = 7;
        let w = gen_ternary(m * k, 0);
        let packed = pack_row_major(&w, m, k);
        let inputs = gen_f32(batch * k, 0);
        let out = unsafe { neon_i2s_batch_matvec(&packed, &inputs, batch, m, k) };
        assert_eq!(out.len(), batch * m);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_batch_all_minus_one_weights() {
        let m = 2;
        let k = 4;
        let batch = 3;
        let w = vec![-1i8; m * k];
        let packed = pack_row_major(&w, m, k);
        let inputs: Vec<f32> = (0..batch * k).map(|i| (i + 1) as f32).collect();
        let out = unsafe { neon_i2s_batch_matvec(&packed, &inputs, batch, m, k) };
        // batch0: input=[1,2,3,4], each row dot = -(1+2+3+4) = -10
        // batch1: input=[5,6,7,8], each row dot = -(5+6+7+8) = -26
        // batch2: input=[9,10,11,12], each row dot = -(9+10+11+12) = -42
        assert_close(&out, &[-10.0, -10.0, -26.0, -26.0, -42.0, -42.0], 1e-4);
    }

    // ====================================================================
    // Additional precision / edge-case tests (96-100)
    // ====================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_precision_small_values() {
        let w = vec![1i8; 4];
        let packed = pack_row_major(&w, 1, 4);
        let input = [1e-7f32; 4];
        let out = unsafe { neon_i2s_matvec(&packed, &input, 1, 4) };
        assert!((out[0] - 4e-7).abs() < 1e-12);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_precision_large_values() {
        let w = vec![1i8; 4];
        let packed = pack_row_major(&w, 1, 4);
        let input = [1e6f32; 4];
        let out = unsafe { neon_i2s_matvec(&packed, &input, 1, 4) };
        assert!((out[0] - 4e6).abs() < 1.0);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_matvec_matches_matmul_n1() {
        // Verify matvec and matmul with n=1 produce identical results for various sizes
        for (m, k) in [(1, 1), (4, 4), (8, 16), (3, 7), (16, 33)] {
            let w = gen_ternary(m * k, m + k);
            let packed = pack_row_major(&w, m, k);
            let input = gen_f32(k, m * k);
            let mv = unsafe { neon_i2s_matvec(&packed, &input, m, k) };
            let mm = unsafe { neon_i2s_matmul(&packed, &input, m, 1, k) };
            assert_close(&mv, &mm, 1e-5);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scaled_bias_equivalence() {
        // scale=1, bias=b should equal plain + b
        let m = 8;
        let k = 16;
        let w = gen_ternary(m * k, 0);
        let packed = pack_row_major(&w, m, k);
        let input = gen_f32(k, 1);
        let bias = gen_f32(m, 2);
        let biased = unsafe { neon_i2s_matvec_bias(&packed, &input, &bias, m, k) };
        let scales = vec![1.0f32; m];
        let scaled = unsafe { neon_i2s_matvec_scaled(&packed, &input, &scales, m, k) };
        let expected: Vec<f32> = scaled.iter().zip(bias.iter()).map(|(d, b)| d + b).collect();
        assert_close(&biased, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_relu_idempotent_on_positive() {
        // If all dots are positive, ReLU should match plain
        let w = vec![1i8; 4 * 4];
        let packed = pack_row_major(&w, 4, 4);
        let input = [1.0f32; 4]; // all dots = 4 > 0
        let plain = unsafe { neon_i2s_matvec(&packed, &input, 4, 4) };
        let relu = unsafe { neon_i2s_matvec_fused_relu(&packed, &input, 4, 4) };
        assert_close(&plain, &relu, 1e-6);
    }
}
