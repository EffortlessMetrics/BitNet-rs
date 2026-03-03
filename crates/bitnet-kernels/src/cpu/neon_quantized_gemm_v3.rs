//! ARM NEON quantized GEMM v3 kernels for Apple Silicon.
//!
//! Tiled quantized matrix multiplication for 1-bit/2-bit weight inference.
//! Provides GEMM, tiled GEMM, GEMV, and block dequantization with NEON
//! SIMD acceleration and scalar fallbacks for non-NEON platforms.
//!
//! I2_S encoding (2 bits per value, 4 values per byte, LSB-first):
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1
//! - `0b10` → unused (treated as 0)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Constants ──────────────────────────────────────────────────────

/// Default tile size for M dimension.
const DEFAULT_TILE_M: usize = 4;
/// Default tile size for N dimension.
const DEFAULT_TILE_N: usize = 4;
/// Number of weights packed per byte (2 bits each).
const WEIGHTS_PER_BYTE: usize = 4;
/// NEON f32 lane width.
const NEON_F32_LANES: usize = 4;

// ── I2_S decode helpers ────────────────────────────────────────────

/// I2_S f32 LUT: index by 2-bit code → f32 value.
const I2S_LUT: [f32; 4] = [0.0, 1.0, 0.0, -1.0];

/// Decode a single 2-bit I2_S code to its signed float value.
#[inline(always)]
fn decode_i2s(bits: u8) -> f32 {
    I2S_LUT[(bits & 0x03) as usize]
}

/// Unpack one packed byte into 4 f32 values via the LUT.
#[inline(always)]
fn unpack_byte_f32(byte: u8) -> [f32; 4] {
    [
        I2S_LUT[(byte & 0x03) as usize],
        I2S_LUT[((byte >> 2) & 0x03) as usize],
        I2S_LUT[((byte >> 4) & 0x03) as usize],
        I2S_LUT[((byte >> 6) & 0x03) as usize],
    ]
}

// ── Block dequantization ───────────────────────────────────────────

/// Dequantize a block of I2_S packed values to f32 using NEON.
///
/// Each byte of `quant` encodes 4 ternary values (2 bits each, LSB-first).
/// Output values are multiplied by `scale`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dequantize_i2_block_neon(quant: &[u8], scale: f32, output: &mut [f32], count: usize) {
    debug_assert!(output.len() >= count);
    let full_bytes = count / WEIGHTS_PER_BYTE;
    let remainder = count % WEIGHTS_PER_BYTE;
    debug_assert!(quant.len() >= full_bytes + usize::from(remainder > 0));

    let scale_v = unsafe { vdupq_n_f32(scale) };

    let mut out_idx = 0usize;
    for i in 0..full_bytes {
        let byte = quant[i];
        let vals = unpack_byte_f32(byte);
        let v = unsafe { vld1q_f32(vals.as_ptr()) };
        let scaled = unsafe { vmulq_f32(v, scale_v) };
        if out_idx + NEON_F32_LANES <= count {
            unsafe { vst1q_f32(output.as_mut_ptr().add(out_idx), scaled) };
        }
        out_idx += NEON_F32_LANES;
    }

    // Scalar remainder
    if remainder > 0 && full_bytes < quant.len() {
        let byte = quant[full_bytes];
        for j in 0..remainder {
            let bits = (byte >> (j * 2)) & 0x03;
            output[out_idx] = decode_i2s(bits) * scale;
            out_idx += 1;
        }
    }
}

/// Scalar fallback for block dequantization.
pub fn dequantize_i2_block_scalar(quant: &[u8], scale: f32, output: &mut [f32], count: usize) {
    debug_assert!(output.len() >= count);
    let mut out_idx = 0usize;
    for &byte in quant.iter() {
        for j in 0..WEIGHTS_PER_BYTE {
            if out_idx >= count {
                return;
            }
            let bits = (byte >> (j * 2)) & 0x03;
            output[out_idx] = decode_i2s(bits) * scale;
            out_idx += 1;
        }
    }
}

// ── I2_S quantized GEMM with NEON ─────────────────────────────────

/// NEON I2_S quantized GEMM: `output[m×n] = a[m×k] × dequant(b_quant)[k×n]`.
///
/// `b_quant` is column-major I2_S packed: each column of length `k`
/// is packed into `ceil(k/4)` bytes. `scales` has one entry per column
/// of `b_quant` (length `n`). `a` is row-major `[m×k]`, `output` is
/// row-major `[m×n]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn quantized_gemm_i2_neon(
    a: &[f32],
    b_quant: &[u8],
    scales: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
    debug_assert!(a.len() >= m * k);
    debug_assert!(b_quant.len() >= n * packed_k);
    debug_assert!(scales.len() >= n);
    debug_assert!(output.len() >= m * n);

    // Temp buffer for dequantized column
    let mut dequant_col = vec![0.0f32; k];

    for j in 0..n {
        let col_packed = &b_quant[j * packed_k..(j + 1) * packed_k];
        let scale = scales[j];

        // Dequantize this column
        unsafe { dequantize_i2_block_neon_inner(col_packed, scale, &mut dequant_col, k) };

        // Dot product each row of a with the dequantized column
        for i in 0..m {
            let row_a = &a[i * k..(i + 1) * k];
            output[i * n + j] = unsafe { neon_dot_f32(row_a, &dequant_col, k) };
        }
    }
}

/// Internal helper: dequantize with NEON (avoids nested target_feature).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dequantize_i2_block_neon_inner(
    quant: &[u8],
    scale: f32,
    output: &mut [f32],
    count: usize,
) {
    let scale_v = unsafe { vdupq_n_f32(scale) };
    let full_bytes = count / WEIGHTS_PER_BYTE;
    let remainder = count % WEIGHTS_PER_BYTE;

    let mut out_idx = 0usize;
    for i in 0..full_bytes {
        let byte = quant[i];
        let vals = unpack_byte_f32(byte);
        let v = unsafe { vld1q_f32(vals.as_ptr()) };
        let scaled = unsafe { vmulq_f32(v, scale_v) };
        unsafe { vst1q_f32(output.as_mut_ptr().add(out_idx), scaled) };
        out_idx += NEON_F32_LANES;
    }

    if remainder > 0 && full_bytes < quant.len() {
        let byte = quant[full_bytes];
        for j in 0..remainder {
            let bits = (byte >> (j * 2)) & 0x03;
            output[out_idx] = decode_i2s(bits) * scale;
            out_idx += 1;
        }
    }
}

/// NEON f32 dot product of two slices.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut acc = unsafe { vdupq_n_f32(0.0) };
    let chunks = len / NEON_F32_LANES;
    let remainder = len % NEON_F32_LANES;

    for i in 0..chunks {
        let offset = i * NEON_F32_LANES;
        let va = unsafe { vld1q_f32(a.as_ptr().add(offset)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(offset)) };
        acc = unsafe { vfmaq_f32(acc, va, vb) };
    }

    let mut sum = unsafe { vaddvq_f32(acc) };

    // Scalar tail
    let tail_start = chunks * NEON_F32_LANES;
    for i in 0..remainder {
        sum += a[tail_start + i] * b[tail_start + i];
    }
    sum
}

/// Scalar fallback for I2_S quantized GEMM.
pub fn quantized_gemm_i2_scalar(
    a: &[f32],
    b_quant: &[u8],
    scales: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
    debug_assert!(a.len() >= m * k);
    debug_assert!(b_quant.len() >= n * packed_k);
    debug_assert!(scales.len() >= n);
    debug_assert!(output.len() >= m * n);

    let mut dequant_col = vec![0.0f32; k];

    for j in 0..n {
        let col_packed = &b_quant[j * packed_k..(j + 1) * packed_k];
        let scale = scales[j];

        dequantize_i2_block_scalar(col_packed, scale, &mut dequant_col, k);

        for i in 0..m {
            let row_a = &a[i * k..(i + 1) * k];
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += row_a[l] * dequant_col[l];
            }
            output[i * n + j] = sum;
        }
    }
}

// ── Tiled GEMM ─────────────────────────────────────────────────────

/// Tiled NEON I2_S quantized GEMM for cache efficiency.
///
/// Processes `tile_m × tile_n` output tiles to maximize L1/L2 cache
/// utilization. Falls back to default tile sizes if 0 is passed.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn quantized_gemm_tiled_neon(
    a: &[f32],
    b_quant: &[u8],
    scales: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    tile_m: usize,
    tile_n: usize,
) {
    let tm = if tile_m == 0 { DEFAULT_TILE_M } else { tile_m };
    let tn = if tile_n == 0 { DEFAULT_TILE_N } else { tile_n };
    let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);

    debug_assert!(a.len() >= m * k);
    debug_assert!(b_quant.len() >= n * packed_k);
    debug_assert!(scales.len() >= n);
    debug_assert!(output.len() >= m * n);

    // Zero output
    for v in output[..m * n].iter_mut() {
        *v = 0.0;
    }

    // Dequantized column buffer (reused across tiles)
    let mut dequant_col = vec![0.0f32; k];

    // Tile over output
    let mut j0 = 0;
    while j0 < n {
        let j_end = (j0 + tn).min(n);

        for j in j0..j_end {
            let col_packed = &b_quant[j * packed_k..(j + 1) * packed_k];
            let scale = scales[j];
            unsafe { dequantize_i2_block_neon_inner(col_packed, scale, &mut dequant_col, k) };

            let mut i0 = 0;
            while i0 < m {
                let i_end = (i0 + tm).min(m);
                for i in i0..i_end {
                    let row_a = &a[i * k..(i + 1) * k];
                    output[i * n + j] = unsafe { neon_dot_f32(row_a, &dequant_col, k) };
                }
                i0 += tm;
            }
        }
        j0 += tn;
    }
}

/// Scalar fallback for tiled GEMM.
pub fn quantized_gemm_tiled_scalar(
    a: &[f32],
    b_quant: &[u8],
    scales: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    tile_m: usize,
    tile_n: usize,
) {
    let tm = if tile_m == 0 { DEFAULT_TILE_M } else { tile_m };
    let tn = if tile_n == 0 { DEFAULT_TILE_N } else { tile_n };
    let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);

    debug_assert!(a.len() >= m * k);
    debug_assert!(b_quant.len() >= n * packed_k);
    debug_assert!(scales.len() >= n);
    debug_assert!(output.len() >= m * n);

    for v in output[..m * n].iter_mut() {
        *v = 0.0;
    }

    let mut dequant_col = vec![0.0f32; k];

    let mut j0 = 0;
    while j0 < n {
        let j_end = (j0 + tn).min(n);
        for j in j0..j_end {
            let col_packed = &b_quant[j * packed_k..(j + 1) * packed_k];
            let scale = scales[j];
            dequantize_i2_block_scalar(col_packed, scale, &mut dequant_col, k);

            let mut i0 = 0;
            while i0 < m {
                let i_end = (i0 + tm).min(m);
                for i in i0..i_end {
                    let row_a = &a[i * k..(i + 1) * k];
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        sum += row_a[l] * dequant_col[l];
                    }
                    output[i * n + j] = sum;
                }
                i0 += tm;
            }
        }
        j0 += tn;
    }
}

// ── GEMV (matrix-vector) ──────────────────────────────────────────

/// NEON I2_S quantized GEMV: `output[n] = dequant(weight_quant)[n×k] · input[k]`.
///
/// Specialized single-token inference path. `weight_quant` is row-major
/// I2_S packed: each row of length `k` is packed into `ceil(k/4)` bytes.
/// `scales` has one entry per output row (length `n`).
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn quantized_gemv_i2_neon(
    input: &[f32],
    weight_quant: &[u8],
    scales: &[f32],
    output: &mut [f32],
    n: usize,
    k: usize,
) {
    let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
    debug_assert!(input.len() >= k);
    debug_assert!(weight_quant.len() >= n * packed_k);
    debug_assert!(scales.len() >= n);
    debug_assert!(output.len() >= n);

    let mut dequant_row = vec![0.0f32; k];

    for i in 0..n {
        let row_packed = &weight_quant[i * packed_k..(i + 1) * packed_k];
        let scale = scales[i];

        unsafe { dequantize_i2_block_neon_inner(row_packed, scale, &mut dequant_row, k) };
        output[i] = unsafe { neon_dot_f32(&dequant_row, input, k) };
    }
}

/// Scalar fallback for GEMV.
pub fn quantized_gemv_i2_scalar(
    input: &[f32],
    weight_quant: &[u8],
    scales: &[f32],
    output: &mut [f32],
    n: usize,
    k: usize,
) {
    let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
    debug_assert!(input.len() >= k);
    debug_assert!(weight_quant.len() >= n * packed_k);
    debug_assert!(scales.len() >= n);
    debug_assert!(output.len() >= n);

    let mut dequant_row = vec![0.0f32; k];

    for i in 0..n {
        let row_packed = &weight_quant[i * packed_k..(i + 1) * packed_k];
        let scale = scales[i];

        dequantize_i2_block_scalar(row_packed, scale, &mut dequant_row, k);

        let mut sum = 0.0f32;
        for l in 0..k {
            sum += dequant_row[l] * input[l];
        }
        output[i] = sum;
    }
}

// ── Naive reference (for testing) ──────────────────────────────────

/// Naive reference GEMM for correctness validation.
///
/// `b_quant` is column-major I2_S packed, `a` is row-major `[m×k]`.
#[cfg(test)]
fn naive_gemm_i2(
    a: &[f32],
    b_quant: &[u8],
    scales: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            let col_packed = &b_quant[j * packed_k..(j + 1) * packed_k];
            let scale = scales[j];
            for l in 0..k {
                let byte_idx = l / WEIGHTS_PER_BYTE;
                let bit_idx = l % WEIGHTS_PER_BYTE;
                let bits = (col_packed[byte_idx] >> (bit_idx * 2)) & 0x03;
                let w = decode_i2s(bits) * scale;
                sum += a[i * k + l] * w;
            }
            output[i * n + j] = sum;
        }
    }
}

/// Naive reference GEMV for correctness validation.
///
/// `weight_quant` is row-major I2_S packed.
#[cfg(test)]
fn naive_gemv_i2(
    input: &[f32],
    weight_quant: &[u8],
    scales: &[f32],
    output: &mut [f32],
    n: usize,
    k: usize,
) {
    let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
    for i in 0..n {
        let mut sum = 0.0f32;
        let row_packed = &weight_quant[i * packed_k..(i + 1) * packed_k];
        let scale = scales[i];
        for l in 0..k {
            let byte_idx = l / WEIGHTS_PER_BYTE;
            let bit_idx = l % WEIGHTS_PER_BYTE;
            let bits = (row_packed[byte_idx] >> (bit_idx * 2)) & 0x03;
            let w = decode_i2s(bits) * scale;
            sum += input[l] * w;
        }
        output[i] = sum;
    }
}

// ── Test helpers ───────────────────────────────────────────────────

/// Pack a slice of ternary values (-1, 0, +1) as f32 into I2_S bytes.
#[cfg(test)]
fn pack_i2s(values: &[f32]) -> Vec<u8> {
    let num_bytes = values.len().div_ceil(WEIGHTS_PER_BYTE);
    let mut packed = vec![0u8; num_bytes];
    for (i, &v) in values.iter().enumerate() {
        let byte_idx = i / WEIGHTS_PER_BYTE;
        let bit_idx = i % WEIGHTS_PER_BYTE;
        let code: u8 = if v > 0.5 {
            0b01 // +1
        } else if v < -0.5 {
            0b11 // -1
        } else {
            0b00 // 0
        };
        packed[byte_idx] |= code << (bit_idx * 2);
    }
    packed
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TOLERANCE
    }

    fn approx_eq_slice(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| approx_eq(*x, *y))
    }

    // ── I2_S encoding/decoding tests ───────────────────────────────

    #[test]
    fn test_decode_i2s_zero() {
        assert_eq!(decode_i2s(0b00), 0.0);
    }

    #[test]
    fn test_decode_i2s_plus_one() {
        assert_eq!(decode_i2s(0b01), 1.0);
    }

    #[test]
    fn test_decode_i2s_minus_one() {
        assert_eq!(decode_i2s(0b11), -1.0);
    }

    #[test]
    fn test_decode_i2s_unused() {
        assert_eq!(decode_i2s(0b10), 0.0);
    }

    #[test]
    fn test_decode_i2s_masks_high_bits() {
        // Only lower 2 bits matter
        assert_eq!(decode_i2s(0xFF), -1.0); // 0xFF & 0x03 = 0b11
        assert_eq!(decode_i2s(0xFC), 0.0); // 0xFC & 0x03 = 0b00
    }

    #[test]
    fn test_unpack_byte_all_zeros() {
        let result = unpack_byte_f32(0x00);
        assert_eq!(result, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_unpack_byte_all_plus_ones() {
        // 0b01_01_01_01 = 0x55
        let result = unpack_byte_f32(0x55);
        assert_eq!(result, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_unpack_byte_all_minus_ones() {
        // 0b11_11_11_11 = 0xFF
        let result = unpack_byte_f32(0xFF);
        assert_eq!(result, [-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_unpack_byte_mixed() {
        // byte = 0b11_00_01_00 = LSB-first: [0, +1, 0, -1]
        // bits: [00, 01, 00, 11] → [0.0, 1.0, 0.0, -1.0]
        let byte = 0b11_00_01_00;
        let result = unpack_byte_f32(byte);
        assert_eq!(result, [0.0, 1.0, 0.0, -1.0]);
    }

    #[test]
    fn test_pack_i2s_round_trip() {
        let values = [1.0f32, -1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0];
        let packed = pack_i2s(&values);
        let mut unpacked = vec![0.0f32; values.len()];
        dequantize_i2_block_scalar(&packed, 1.0, &mut unpacked, values.len());
        assert!(approx_eq_slice(&values, &unpacked));
    }

    #[test]
    fn test_pack_i2s_non_aligned() {
        // 5 values = 2 bytes (last byte has 1 valid value)
        let values = [1.0f32, -1.0, 0.0, 1.0, -1.0];
        let packed = pack_i2s(&values);
        assert_eq!(packed.len(), 2);
        let mut unpacked = vec![0.0f32; values.len()];
        dequantize_i2_block_scalar(&packed, 1.0, &mut unpacked, values.len());
        assert!(approx_eq_slice(&values, &unpacked));
    }

    #[test]
    fn test_pack_i2s_single_value() {
        for &v in &[1.0f32, -1.0, 0.0] {
            let packed = pack_i2s(&[v]);
            let mut unpacked = [0.0f32];
            dequantize_i2_block_scalar(&packed, 1.0, &mut unpacked, 1);
            assert!(approx_eq(v, unpacked[0]));
        }
    }

    // ── Scalar dequantize tests ────────────────────────────────────

    #[test]
    fn test_dequantize_scalar_scale_factor() {
        let values = [1.0f32, -1.0, 0.0, 1.0];
        let packed = pack_i2s(&values);
        let mut output = vec![0.0f32; 4];
        dequantize_i2_block_scalar(&packed, 2.5, &mut output, 4);
        let expected = [2.5, -2.5, 0.0, 2.5];
        assert!(approx_eq_slice(&output, &expected));
    }

    #[test]
    fn test_dequantize_scalar_zero_scale() {
        let values = [1.0f32, -1.0, 0.0, 1.0];
        let packed = pack_i2s(&values);
        let mut output = vec![0.0f32; 4];
        dequantize_i2_block_scalar(&packed, 0.0, &mut output, 4);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequantize_scalar_negative_scale() {
        let values = [1.0f32, -1.0, 0.0, 1.0];
        let packed = pack_i2s(&values);
        let mut output = vec![0.0f32; 4];
        dequantize_i2_block_scalar(&packed, -1.0, &mut output, 4);
        let expected = [-1.0, 1.0, 0.0, -1.0];
        assert!(approx_eq_slice(&output, &expected));
    }

    #[test]
    fn test_dequantize_scalar_large_block() {
        let k: usize = 64;
        let values: Vec<f32> = (0..k)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        let packed = pack_i2s(&values);
        let mut output = vec![0.0f32; k];
        dequantize_i2_block_scalar(&packed, 1.0, &mut output, k);
        assert!(approx_eq_slice(&values, &output));
    }

    // ── NEON dequantize tests ──────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_dequantize_neon_basic() {
        let values = [1.0f32, -1.0, 0.0, 1.0];
        let packed = pack_i2s(&values);
        let mut output = vec![0.0f32; 4];
        unsafe { dequantize_i2_block_neon(&packed, 1.0, &mut output, 4) };
        assert!(approx_eq_slice(&values, &output));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_dequantize_neon_with_scale() {
        let values = [1.0f32, -1.0, 0.0, 1.0];
        let packed = pack_i2s(&values);
        let mut output = vec![0.0f32; 4];
        unsafe { dequantize_i2_block_neon(&packed, 3.0, &mut output, 4) };
        let expected = [3.0, -3.0, 0.0, 3.0];
        assert!(approx_eq_slice(&output, &expected));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_dequantize_neon_vs_scalar_parity() {
        let k: usize = 37; // Non-aligned count
        let values: Vec<f32> = (0..k)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        let packed = pack_i2s(&values);
        let mut neon_out = vec![0.0f32; k];
        let mut scalar_out = vec![0.0f32; k];
        unsafe { dequantize_i2_block_neon(&packed, 2.0, &mut neon_out, k) };
        dequantize_i2_block_scalar(&packed, 2.0, &mut scalar_out, k);
        assert!(approx_eq_slice(&neon_out, &scalar_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_dequantize_neon_non_aligned_count() {
        // 5 elements = 1 full byte + 1 partial byte
        let values = [1.0f32, -1.0, 0.0, 1.0, -1.0];
        let packed = pack_i2s(&values);
        let mut output = vec![0.0f32; 5];
        unsafe { dequantize_i2_block_neon(&packed, 1.0, &mut output, 5) };
        assert!(approx_eq_slice(&values, &output));
    }

    // ── Scalar GEMM tests ──────────────────────────────────────────

    #[test]
    fn test_gemm_scalar_identity_weights() {
        // 2x2 identity-like: weight col 0 = [1,0], col 1 = [0,1]
        let m: usize = 2;
        let n: usize = 2;
        let k: usize = 2;
        let a = [1.0f32, 0.0, 0.0, 1.0]; // identity
        let w_col0 = [1.0f32, 0.0];
        let w_col1 = [0.0f32, 1.0];
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let p0 = pack_i2s(&w_col0);
        let p1 = pack_i2s(&w_col1);
        b_quant[..packed_k].copy_from_slice(&p0);
        b_quant[packed_k..2 * packed_k].copy_from_slice(&p1);
        let scales = [1.0f32, 1.0];
        let mut output = vec![0.0f32; m * n];
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut output, m, n, k);
        let mut expected = vec![0.0f32; m * n];
        naive_gemm_i2(&a, &b_quant, &scales, &mut expected, m, n, k);
        assert!(approx_eq_slice(&output, &expected));
    }

    #[test]
    fn test_gemm_scalar_1x1() {
        let a = [3.0f32];
        let w = [1.0f32];
        let packed = pack_i2s(&w);
        let scales = [2.0f32];
        let mut output = [0.0f32];
        quantized_gemm_i2_scalar(&a, &packed, &scales, &mut output, 1, 1, 1);
        // 3.0 * (1.0 * 2.0) = 6.0
        assert!(approx_eq(output[0], 6.0));
    }

    #[test]
    fn test_gemm_scalar_single_row() {
        let m: usize = 1;
        let n: usize = 4;
        let k: usize = 8;
        let a: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales: Vec<f32> = (0..n).map(|j| 1.0 + j as f32 * 0.5).collect();
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut output = vec![0.0f32; m * n];
        let mut expected = vec![0.0f32; m * n];
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut output, m, n, k);
        naive_gemm_i2(&a, &b_quant, &scales, &mut expected, m, n, k);
        assert!(approx_eq_slice(&output, &expected));
    }

    #[test]
    fn test_gemm_scalar_single_column() {
        let m: usize = 4;
        let n: usize = 1;
        let k: usize = 8;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let w: Vec<f32> = (0..k)
            .map(|l| match l % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        let packed = pack_i2s(&w);
        let scales = [1.5f32];
        let mut output = vec![0.0f32; m * n];
        let mut expected = vec![0.0f32; m * n];
        quantized_gemm_i2_scalar(&a, &packed, &scales, &mut output, m, n, k);
        naive_gemm_i2(&a, &packed, &scales, &mut expected, m, n, k);
        assert!(approx_eq_slice(&output, &expected));
    }

    #[test]
    fn test_gemm_scalar_4x4() {
        let m: usize = 4;
        let n: usize = 4;
        let k: usize = 16;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j * 3 + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = [1.0f32, 0.5, 2.0, 1.5];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut output = vec![0.0f32; m * n];
        let mut expected = vec![0.0f32; m * n];
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut output, m, n, k);
        naive_gemm_i2(&a, &b_quant, &scales, &mut expected, m, n, k);
        assert!(approx_eq_slice(&output, &expected));
    }

    #[test]
    fn test_gemm_scalar_non_power_of_2() {
        let m: usize = 3;
        let n: usize = 5;
        let k: usize = 7;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.05).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales: Vec<f32> = vec![1.0; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut output = vec![0.0f32; m * n];
        let mut expected = vec![0.0f32; m * n];
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut output, m, n, k);
        naive_gemm_i2(&a, &b_quant, &scales, &mut expected, m, n, k);
        assert!(approx_eq_slice(&output, &expected));
    }

    #[test]
    fn test_gemm_scalar_all_zero_weights() {
        let m: usize = 2;
        let n: usize = 2;
        let k: usize = 8;
        let a: Vec<f32> = vec![1.0; m * k];
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let b_quant = vec![0u8; n * packed_k]; // all zeros
        let scales = vec![1.0f32; n];
        let mut output = vec![999.0f32; m * n];
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut output, m, n, k);
        assert!(output.iter().all(|&v| approx_eq(v, 0.0)));
    }

    #[test]
    fn test_gemm_scalar_all_plus_one_weights() {
        let m: usize = 1;
        let n: usize = 1;
        let k: usize = 8;
        let a: Vec<f32> = vec![1.0; k];
        let w = vec![1.0f32; k];
        let packed = pack_i2s(&w);
        let scales = [1.0f32];
        let mut output = [0.0f32];
        quantized_gemm_i2_scalar(&a, &packed, &scales, &mut output, m, n, k);
        assert!(approx_eq(output[0], 8.0));
    }

    #[test]
    fn test_gemm_scalar_all_minus_one_weights() {
        let m: usize = 1;
        let n: usize = 1;
        let k: usize = 4;
        let a: Vec<f32> = vec![2.0; k];
        let w = vec![-1.0f32; k];
        let packed = pack_i2s(&w);
        let scales = [1.0f32];
        let mut output = [0.0f32];
        quantized_gemm_i2_scalar(&a, &packed, &scales, &mut output, m, n, k);
        // 4 * (2.0 * -1.0) = -8.0
        assert!(approx_eq(output[0], -8.0));
    }

    // ── NEON GEMM tests ────────────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_neon_1x1() {
        let a = [3.0f32];
        let w = [1.0f32];
        let packed = pack_i2s(&w);
        let scales = [2.0f32];
        let mut output = [0.0f32];
        unsafe {
            quantized_gemm_i2_neon(&a, &packed, &scales, &mut output, 1, 1, 1);
        }
        assert!(approx_eq(output[0], 6.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_neon_vs_scalar_4x4() {
        let m: usize = 4;
        let n: usize = 4;
        let k: usize = 16;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = [1.0f32, 2.0, 0.5, 1.5];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; m * n];
        let mut scalar_out = vec![0.0f32; m * n];
        unsafe {
            quantized_gemm_i2_neon(&a, &b_quant, &scales, &mut neon_out, m, n, k);
        }
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut scalar_out, m, n, k);
        assert!(approx_eq_slice(&neon_out, &scalar_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_neon_vs_scalar_non_pow2() {
        let m: usize = 3;
        let n: usize = 5;
        let k: usize = 7;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j * 2 + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; m * n];
        let mut scalar_out = vec![0.0f32; m * n];
        unsafe {
            quantized_gemm_i2_neon(&a, &b_quant, &scales, &mut neon_out, m, n, k);
        }
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut scalar_out, m, n, k);
        assert!(approx_eq_slice(&neon_out, &scalar_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_neon_vs_naive_medium() {
        let m: usize = 8;
        let n: usize = 8;
        let k: usize = 32;
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 11) as f32 * 0.1 - 0.5).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l * 2) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales: Vec<f32> = (0..n).map(|j| 0.5 + j as f32 * 0.25).collect();
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; m * n];
        let mut naive_out = vec![0.0f32; m * n];
        unsafe {
            quantized_gemm_i2_neon(&a, &b_quant, &scales, &mut neon_out, m, n, k);
        }
        naive_gemm_i2(&a, &b_quant, &scales, &mut naive_out, m, n, k);
        assert!(approx_eq_slice(&neon_out, &naive_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_neon_single_row() {
        let m: usize = 1;
        let n: usize = 8;
        let k: usize = 16;
        let a: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; m * n];
        let mut scalar_out = vec![0.0f32; m * n];
        unsafe {
            quantized_gemm_i2_neon(&a, &b_quant, &scales, &mut neon_out, m, n, k);
        }
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut scalar_out, m, n, k);
        assert!(approx_eq_slice(&neon_out, &scalar_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_neon_single_column() {
        let m: usize = 8;
        let n: usize = 1;
        let k: usize = 16;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.05).collect();
        let w: Vec<f32> = (0..k)
            .map(|l| match l % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        let packed = pack_i2s(&w);
        let scales = [1.0f32];
        let mut neon_out = vec![0.0f32; m * n];
        let mut scalar_out = vec![0.0f32; m * n];
        unsafe {
            quantized_gemm_i2_neon(&a, &packed, &scales, &mut neon_out, m, n, k);
        }
        quantized_gemm_i2_scalar(&a, &packed, &scales, &mut scalar_out, m, n, k);
        assert!(approx_eq_slice(&neon_out, &scalar_out));
    }

    // ── Tiled GEMM tests ──────────────────────────────────────────

    #[test]
    fn test_tiled_gemm_scalar_vs_naive() {
        let m: usize = 4;
        let n: usize = 4;
        let k: usize = 16;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut tiled_out = vec![0.0f32; m * n];
        let mut naive_out = vec![0.0f32; m * n];
        quantized_gemm_tiled_scalar(&a, &b_quant, &scales, &mut tiled_out, m, n, k, 2, 2);
        naive_gemm_i2(&a, &b_quant, &scales, &mut naive_out, m, n, k);
        assert!(approx_eq_slice(&tiled_out, &naive_out));
    }

    #[test]
    fn test_tiled_gemm_scalar_tile_1x1() {
        let m: usize = 3;
        let n: usize = 3;
        let k: usize = 8;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 2 {
                        0 => 1.0,
                        _ => -1.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut tiled_out = vec![0.0f32; m * n];
        let mut naive_out = vec![0.0f32; m * n];
        quantized_gemm_tiled_scalar(&a, &b_quant, &scales, &mut tiled_out, m, n, k, 1, 1);
        naive_gemm_i2(&a, &b_quant, &scales, &mut naive_out, m, n, k);
        assert!(approx_eq_slice(&tiled_out, &naive_out));
    }

    #[test]
    fn test_tiled_gemm_scalar_default_tiles() {
        let m: usize = 6;
        let n: usize = 6;
        let k: usize = 12;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.05).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut tiled_out = vec![0.0f32; m * n];
        let mut naive_out = vec![0.0f32; m * n];
        // tile_m=0, tile_n=0 → defaults
        quantized_gemm_tiled_scalar(&a, &b_quant, &scales, &mut tiled_out, m, n, k, 0, 0);
        naive_gemm_i2(&a, &b_quant, &scales, &mut naive_out, m, n, k);
        assert!(approx_eq_slice(&tiled_out, &naive_out));
    }

    #[test]
    fn test_tiled_gemm_scalar_large_tile() {
        // Tile larger than matrix
        let m: usize = 2;
        let n: usize = 2;
        let k: usize = 8;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut tiled_out = vec![0.0f32; m * n];
        let mut naive_out = vec![0.0f32; m * n];
        quantized_gemm_tiled_scalar(&a, &b_quant, &scales, &mut tiled_out, m, n, k, 16, 16);
        naive_gemm_i2(&a, &b_quant, &scales, &mut naive_out, m, n, k);
        assert!(approx_eq_slice(&tiled_out, &naive_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_tiled_gemm_neon_vs_scalar() {
        let m: usize = 4;
        let n: usize = 4;
        let k: usize = 16;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; m * n];
        let mut scalar_out = vec![0.0f32; m * n];
        unsafe {
            quantized_gemm_tiled_neon(&a, &b_quant, &scales, &mut neon_out, m, n, k, 2, 2);
        }
        quantized_gemm_tiled_scalar(&a, &b_quant, &scales, &mut scalar_out, m, n, k, 2, 2);
        assert!(approx_eq_slice(&neon_out, &scalar_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_tiled_gemm_neon_non_pow2() {
        let m: usize = 5;
        let n: usize = 7;
        let k: usize = 11;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.05).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; m * n];
        let mut naive_out = vec![0.0f32; m * n];
        unsafe {
            quantized_gemm_tiled_neon(&a, &b_quant, &scales, &mut neon_out, m, n, k, 3, 3);
        }
        naive_gemm_i2(&a, &b_quant, &scales, &mut naive_out, m, n, k);
        assert!(approx_eq_slice(&neon_out, &naive_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_tiled_gemm_neon_default_tiles() {
        let m: usize = 8;
        let n: usize = 8;
        let k: usize = 32;
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 13) % 17) as f32 * 0.1 - 0.8).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales: Vec<f32> = (0..n).map(|j| 0.5 + j as f32 * 0.1).collect();
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; m * n];
        let mut naive_out = vec![0.0f32; m * n];
        unsafe {
            quantized_gemm_tiled_neon(&a, &b_quant, &scales, &mut neon_out, m, n, k, 0, 0);
        }
        naive_gemm_i2(&a, &b_quant, &scales, &mut naive_out, m, n, k);
        assert!(approx_eq_slice(&neon_out, &naive_out));
    }

    // ── GEMV tests ─────────────────────────────────────────────────

    #[test]
    fn test_gemv_scalar_basic() {
        let n: usize = 4;
        let k: usize = 8;
        let input: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|l| match (i + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut w_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for i in 0..n {
            let p = pack_i2s(&weights[i]);
            w_quant[i * packed_k..(i + 1) * packed_k].copy_from_slice(&p);
        }
        let mut output = vec![0.0f32; n];
        let mut expected = vec![0.0f32; n];
        quantized_gemv_i2_scalar(&input, &w_quant, &scales, &mut output, n, k);
        naive_gemv_i2(&input, &w_quant, &scales, &mut expected, n, k);
        assert!(approx_eq_slice(&output, &expected));
    }

    #[test]
    fn test_gemv_scalar_single_output() {
        let n: usize = 1;
        let k: usize = 4;
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32, -1.0, 1.0, -1.0];
        let packed = pack_i2s(&w);
        let scales = [1.0f32];
        let mut output = [0.0f32];
        quantized_gemv_i2_scalar(&input, &packed, &scales, &mut output, n, k);
        // 1*1 + 2*(-1) + 3*1 + 4*(-1) = 1 - 2 + 3 - 4 = -2
        assert!(approx_eq(output[0], -2.0));
    }

    #[test]
    fn test_gemv_scalar_with_scale() {
        let n: usize = 1;
        let k: usize = 4;
        let input = [1.0f32, 1.0, 1.0, 1.0];
        let w = [1.0f32, 1.0, 1.0, 1.0];
        let packed = pack_i2s(&w);
        let scales = [3.0f32];
        let mut output = [0.0f32];
        quantized_gemv_i2_scalar(&input, &packed, &scales, &mut output, n, k);
        // 4 * (1.0 * 3.0) = 12.0
        assert!(approx_eq(output[0], 12.0));
    }

    #[test]
    fn test_gemv_scalar_non_aligned_k() {
        let n: usize = 2;
        let k: usize = 5;
        let input: Vec<f32> = (0..k).map(|i| i as f32).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|l| match (i + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut w_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for i in 0..n {
            let p = pack_i2s(&weights[i]);
            w_quant[i * packed_k..(i + 1) * packed_k].copy_from_slice(&p);
        }
        let mut output = vec![0.0f32; n];
        let mut expected = vec![0.0f32; n];
        quantized_gemv_i2_scalar(&input, &w_quant, &scales, &mut output, n, k);
        naive_gemv_i2(&input, &w_quant, &scales, &mut expected, n, k);
        assert!(approx_eq_slice(&output, &expected));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemv_neon_vs_scalar() {
        let n: usize = 8;
        let k: usize = 32;
        let input: Vec<f32> = (0..k).map(|i| i as f32 * 0.05 - 0.5).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|l| match (i * 3 + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut w_quant = vec![0u8; n * packed_k];
        let scales: Vec<f32> = (0..n).map(|i| 0.5 + i as f32 * 0.1).collect();
        for i in 0..n {
            let p = pack_i2s(&weights[i]);
            w_quant[i * packed_k..(i + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; n];
        let mut scalar_out = vec![0.0f32; n];
        unsafe {
            quantized_gemv_i2_neon(&input, &w_quant, &scales, &mut neon_out, n, k);
        }
        quantized_gemv_i2_scalar(&input, &w_quant, &scales, &mut scalar_out, n, k);
        assert!(approx_eq_slice(&neon_out, &scalar_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemv_neon_vs_naive() {
        let n: usize = 4;
        let k: usize = 16;
        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.2).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|l| match (i + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut w_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for i in 0..n {
            let p = pack_i2s(&weights[i]);
            w_quant[i * packed_k..(i + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; n];
        let mut naive_out = vec![0.0f32; n];
        unsafe {
            quantized_gemv_i2_neon(&input, &w_quant, &scales, &mut neon_out, n, k);
        }
        naive_gemv_i2(&input, &w_quant, &scales, &mut naive_out, n, k);
        assert!(approx_eq_slice(&neon_out, &naive_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemv_neon_single_output() {
        let n: usize = 1;
        let k: usize = 4;
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32, -1.0, 1.0, -1.0];
        let packed = pack_i2s(&w);
        let scales = [1.0f32];
        let mut output = [0.0f32];
        unsafe {
            quantized_gemv_i2_neon(&input, &packed, &scales, &mut output, n, k);
        }
        assert!(approx_eq(output[0], -2.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemv_neon_non_aligned_k() {
        let n: usize = 3;
        let k: usize = 13;
        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|l| match (i + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut w_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for i in 0..n {
            let p = pack_i2s(&weights[i]);
            w_quant[i * packed_k..(i + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; n];
        let mut scalar_out = vec![0.0f32; n];
        unsafe {
            quantized_gemv_i2_neon(&input, &w_quant, &scales, &mut neon_out, n, k);
        }
        quantized_gemv_i2_scalar(&input, &w_quant, &scales, &mut scalar_out, n, k);
        assert!(approx_eq_slice(&neon_out, &scalar_out));
    }

    // ── Edge case tests ────────────────────────────────────────────

    #[test]
    fn test_gemm_scalar_k_equals_1() {
        let m: usize = 2;
        let n: usize = 3;
        let k: usize = 1;
        let a = [1.0f32, 2.0];
        let weights: Vec<Vec<f32>> = vec![vec![1.0], vec![-1.0], vec![0.0]];
        let packed_k = 1;
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut output = vec![0.0f32; m * n];
        let mut expected = vec![0.0f32; m * n];
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut output, m, n, k);
        naive_gemm_i2(&a, &b_quant, &scales, &mut expected, m, n, k);
        assert!(approx_eq_slice(&output, &expected));
    }

    #[test]
    fn test_gemm_scalar_k_equals_3() {
        // k=3 → 1 byte (with 1 unused slot)
        let m: usize = 2;
        let n: usize = 2;
        let k: usize = 3;
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights: Vec<Vec<f32>> = vec![vec![1.0, -1.0, 1.0], vec![-1.0, 0.0, 1.0]];
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut output = vec![0.0f32; m * n];
        let mut expected = vec![0.0f32; m * n];
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut output, m, n, k);
        naive_gemm_i2(&a, &b_quant, &scales, &mut expected, m, n, k);
        assert!(approx_eq_slice(&output, &expected));
    }

    #[test]
    fn test_gemm_scalar_large_k_256() {
        let m: usize = 2;
        let n: usize = 2;
        let k: usize = 256;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 10) as f32 * 0.1).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut output = vec![0.0f32; m * n];
        let mut expected = vec![0.0f32; m * n];
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut output, m, n, k);
        naive_gemm_i2(&a, &b_quant, &scales, &mut expected, m, n, k);
        assert!(approx_eq_slice(&output, &expected));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_neon_k_equals_1() {
        let m: usize = 2;
        let n: usize = 2;
        let k: usize = 1;
        let a = [1.0f32, 2.0];
        let weights: Vec<Vec<f32>> = vec![vec![1.0], vec![-1.0]];
        let packed_k = 1;
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; m * n];
        let mut scalar_out = vec![0.0f32; m * n];
        unsafe {
            quantized_gemm_i2_neon(&a, &b_quant, &scales, &mut neon_out, m, n, k);
        }
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut scalar_out, m, n, k);
        assert!(approx_eq_slice(&neon_out, &scalar_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_neon_large_k_256() {
        let m: usize = 2;
        let n: usize = 2;
        let k: usize = 256;
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 13) as f32 * 0.1 - 0.5).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; m * n];
        let mut scalar_out = vec![0.0f32; m * n];
        unsafe {
            quantized_gemm_i2_neon(&a, &b_quant, &scales, &mut neon_out, m, n, k);
        }
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut scalar_out, m, n, k);
        assert!(approx_eq_slice(&neon_out, &scalar_out));
    }

    #[test]
    fn test_gemv_scalar_all_zero_weights() {
        let n: usize = 4;
        let k: usize = 8;
        let input = vec![1.0f32; k];
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let w_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        let mut output = vec![999.0f32; n];
        quantized_gemv_i2_scalar(&input, &w_quant, &scales, &mut output, n, k);
        assert!(output.iter().all(|&v| approx_eq(v, 0.0)));
    }

    #[test]
    fn test_gemv_scalar_large_k_128() {
        let n: usize = 4;
        let k: usize = 128;
        let input: Vec<f32> = (0..k).map(|i| (i % 5) as f32 * 0.2).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|l| match (i + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut w_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for i in 0..n {
            let p = pack_i2s(&weights[i]);
            w_quant[i * packed_k..(i + 1) * packed_k].copy_from_slice(&p);
        }
        let mut output = vec![0.0f32; n];
        let mut expected = vec![0.0f32; n];
        quantized_gemv_i2_scalar(&input, &w_quant, &scales, &mut output, n, k);
        naive_gemv_i2(&input, &w_quant, &scales, &mut expected, n, k);
        assert!(approx_eq_slice(&output, &expected));
    }

    // ── Cross-function consistency tests ───────────────────────────

    #[test]
    fn test_gemm_m1_equals_gemv_scalar() {
        // GEMM with m=1 should produce the same result as GEMV
        let n: usize = 4;
        let k: usize = 16;
        let input: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();

        // For GEMV: weight_quant is row-major (each row is a weight vector)
        // For GEMM: b_quant is column-major (each column is a weight vector)
        // These are different layouts, so we need separate packed data.
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|l| match (i + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();

        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);

        // Pack for GEMV (row-major: row i is weights[i])
        let mut w_quant_gemv = vec![0u8; n * packed_k];
        for i in 0..n {
            let p = pack_i2s(&weights[i]);
            w_quant_gemv[i * packed_k..(i + 1) * packed_k].copy_from_slice(&p);
        }
        let scales = vec![1.0f32; n];

        let mut gemv_out = vec![0.0f32; n];
        quantized_gemv_i2_scalar(&input, &w_quant_gemv, &scales, &mut gemv_out, n, k);

        // Pack for GEMM (column-major: column j is weights[j])
        let mut b_quant_gemm = vec![0u8; n * packed_k];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant_gemm[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }

        let mut gemm_out = vec![0.0f32; n];
        quantized_gemm_i2_scalar(&input, &b_quant_gemm, &scales, &mut gemm_out, 1, n, k);

        assert!(approx_eq_slice(&gemm_out, &gemv_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_m1_equals_gemv_neon() {
        let n: usize = 4;
        let k: usize = 16;
        let input: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|l| match (i + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut w_quant_gemv = vec![0u8; n * packed_k];
        for i in 0..n {
            let p = pack_i2s(&weights[i]);
            w_quant_gemv[i * packed_k..(i + 1) * packed_k].copy_from_slice(&p);
        }
        let scales = vec![1.0f32; n];
        let mut gemv_out = vec![0.0f32; n];
        unsafe {
            quantized_gemv_i2_neon(&input, &w_quant_gemv, &scales, &mut gemv_out, n, k);
        }

        let mut b_quant_gemm = vec![0u8; n * packed_k];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant_gemm[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut gemm_out = vec![0.0f32; n];
        unsafe {
            quantized_gemm_i2_neon(&input, &b_quant_gemm, &scales, &mut gemm_out, 1, n, k);
        }
        assert!(approx_eq_slice(&gemm_out, &gemv_out));
    }

    #[test]
    fn test_tiled_vs_untiled_scalar() {
        let m: usize = 6;
        let n: usize = 5;
        let k: usize = 20;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03 - 1.0).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut untiled_out = vec![0.0f32; m * n];
        let mut tiled_out = vec![0.0f32; m * n];
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut untiled_out, m, n, k);
        quantized_gemm_tiled_scalar(&a, &b_quant, &scales, &mut tiled_out, m, n, k, 3, 2);
        assert!(approx_eq_slice(&untiled_out, &tiled_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_tiled_vs_untiled_neon() {
        let m: usize = 6;
        let n: usize = 5;
        let k: usize = 20;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03 - 1.0).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut untiled_out = vec![0.0f32; m * n];
        let mut tiled_out = vec![0.0f32; m * n];
        unsafe {
            quantized_gemm_i2_neon(&a, &b_quant, &scales, &mut untiled_out, m, n, k);
            quantized_gemm_tiled_neon(&a, &b_quant, &scales, &mut tiled_out, m, n, k, 3, 2);
        }
        assert!(approx_eq_slice(&untiled_out, &tiled_out));
    }

    // ── Various tile size tests ────────────────────────────────────

    #[test]
    fn test_tiled_scalar_various_tile_sizes() {
        let m: usize = 8;
        let n: usize = 8;
        let k: usize = 16;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.05).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut expected = vec![0.0f32; m * n];
        naive_gemm_i2(&a, &b_quant, &scales, &mut expected, m, n, k);

        for &(tm, tn) in &[(1, 1), (2, 2), (4, 4), (8, 8), (1, 8), (8, 1), (3, 5)] {
            let mut output = vec![0.0f32; m * n];
            quantized_gemm_tiled_scalar(&a, &b_quant, &scales, &mut output, m, n, k, tm, tn);
            assert!(approx_eq_slice(&output, &expected), "Tile ({tm}, {tn}) mismatch");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_tiled_neon_various_tile_sizes() {
        let m: usize = 8;
        let n: usize = 8;
        let k: usize = 16;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.05).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| match (j + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut expected = vec![0.0f32; m * n];
        naive_gemm_i2(&a, &b_quant, &scales, &mut expected, m, n, k);

        for &(tm, tn) in &[(1, 1), (2, 2), (4, 4), (8, 8), (1, 8), (8, 1), (3, 5)] {
            let mut output = vec![0.0f32; m * n];
            unsafe {
                quantized_gemm_tiled_neon(&a, &b_quant, &scales, &mut output, m, n, k, tm, tn);
            }
            assert!(approx_eq_slice(&output, &expected), "Tile ({tm}, {tn}) mismatch");
        }
    }

    // ── Scale variation tests ──────────────────────────────────────

    #[test]
    fn test_gemm_scalar_varying_scales() {
        let m: usize = 2;
        let n: usize = 4;
        let k: usize = 8;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..k)
                    .map(|l| match l % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = [0.1f32, 1.0, 10.0, 100.0];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut output = vec![0.0f32; m * n];
        let mut expected = vec![0.0f32; m * n];
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut output, m, n, k);
        naive_gemm_i2(&a, &b_quant, &scales, &mut expected, m, n, k);
        assert!(approx_eq_slice(&output, &expected));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemm_neon_varying_scales() {
        let m: usize = 2;
        let n: usize = 4;
        let k: usize = 8;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..k)
                    .map(|l| match l % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut b_quant = vec![0u8; n * packed_k];
        let scales = [0.1f32, 1.0, 10.0, 100.0];
        for j in 0..n {
            let p = pack_i2s(&weights[j]);
            b_quant[j * packed_k..(j + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; m * n];
        let mut scalar_out = vec![0.0f32; m * n];
        unsafe {
            quantized_gemm_i2_neon(&a, &b_quant, &scales, &mut neon_out, m, n, k);
        }
        quantized_gemm_i2_scalar(&a, &b_quant, &scales, &mut scalar_out, m, n, k);
        assert!(approx_eq_slice(&neon_out, &scalar_out));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gemv_neon_large_n() {
        let n: usize = 64;
        let k: usize = 32;
        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.05).collect();
        let weights: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|l| match (i + l) % 3 {
                        0 => 1.0,
                        1 => -1.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let packed_k = k.div_ceil(WEIGHTS_PER_BYTE);
        let mut w_quant = vec![0u8; n * packed_k];
        let scales = vec![1.0f32; n];
        for i in 0..n {
            let p = pack_i2s(&weights[i]);
            w_quant[i * packed_k..(i + 1) * packed_k].copy_from_slice(&p);
        }
        let mut neon_out = vec![0.0f32; n];
        let mut naive_out = vec![0.0f32; n];
        unsafe {
            quantized_gemv_i2_neon(&input, &w_quant, &scales, &mut neon_out, n, k);
        }
        naive_gemv_i2(&input, &w_quant, &scales, &mut naive_out, n, k);
        assert!(approx_eq_slice(&neon_out, &naive_out));
    }
}
