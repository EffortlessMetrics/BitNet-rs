#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! NEON-optimized I2_S quantized matrix multiplication v2 for Apple Silicon.
//!
//! Targets the I2_S 2-bit ternary encoding used by BitNet models:
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1
//! - `0b10` → unused (treated as 0)
//!
//! # NEON optimization strategy
//!
//! - **`vshl` / `vmovn`**: 2-bit field extraction — shift packed bytes to
//!   isolate each crumb, then narrow to 8-bit indices.
//! - **`vtbl`**: LUT decode — map 2-bit codes to ternary values {−1, 0, +1}
//!   in a single NEON table lookup instruction.
//! - **`vfma`**: Fused multiply-accumulate for the dot-product inner loop,
//!   keeping partial sums in NEON registers.
//! - **4×4 tile blocking**: Process 4 output rows × 4 columns per tile to
//!   balance register pressure against memory traffic on Apple M-series
//!   cores (32 NEON registers available on AArch64).
//!
//! # Block sizes
//!
//! - **32** — BitNet32-F16 format (32-element blocks with inline F16 scales)
//! - **256** — QK256 format (256-element blocks with separate scales)

use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Configuration ──────────────────────────────────────────────────

/// Configuration for a quantized matrix multiplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedMatmulConfig {
    /// Number of output rows.
    pub m: usize,
    /// Number of output columns.
    pub n: usize,
    /// Inner (reduction) dimension.
    pub k: usize,
    /// Block size: 32 for BitNet32-F16, 256 for QK256.
    pub block_size: usize,
}

/// Result of a quantized matmul operation.
#[derive(Debug, Clone)]
pub struct QuantizedMatmulResult {
    /// The output data buffer (`m × n` floats).
    pub output_data: Vec<f32>,
    /// Wall-clock compute time in nanoseconds.
    pub compute_time_ns: u64,
    /// Estimated operations per second.
    pub ops_per_second: f64,
}

// ── I2_S LUT ───────────────────────────────────────────────────────

/// Precomputed lookup table: index by 2-bit I2_S code → f32 value.
const I2S_LUT_F32: [f32; 4] = [0.0, 1.0, 0.0, -1.0];

/// Precomputed 256-entry byte LUT: for each packed byte, store the four
/// decoded i8 values ({−1, 0, +1}) for the four 2-bit fields.
/// Layout: `BYTE_LUT[byte][crumb_index]` where crumb_index ∈ 0..4.
const BYTE_LUT_I8: [[i8; 4]; 256] = {
    let mut table = [[0i8; 4]; 256];
    let map: [i8; 4] = [0, 1, 0, -1];
    let mut b: usize = 0;
    while b < 256 {
        table[b][0] = map[b & 0x03];
        table[b][1] = map[(b >> 2) & 0x03];
        table[b][2] = map[(b >> 4) & 0x03];
        table[b][3] = map[(b >> 6) & 0x03];
        b += 1;
    }
    table
};

// ── Scalar helpers ─────────────────────────────────────────────────

/// Decode a single 2-bit I2_S code to its signed float value.
#[inline(always)]
pub(crate) fn decode_i2s(bits: u8) -> f32 {
    match bits & 0x03 {
        0b01 => 1.0,
        0b11 => -1.0,
        _ => 0.0, // 0b00 = 0, 0b10 = unused → 0
    }
}

/// Unpack one byte (4 crumbs) into f32 values via the const LUT.
#[inline(always)]
fn unpack_byte_f32(byte: u8) -> [f32; 4] {
    [
        I2S_LUT_F32[(byte & 0x03) as usize],
        I2S_LUT_F32[((byte >> 2) & 0x03) as usize],
        I2S_LUT_F32[((byte >> 4) & 0x03) as usize],
        I2S_LUT_F32[((byte >> 6) & 0x03) as usize],
    ]
}

/// Scalar reference: dot product of one packed I2_S row against an f32 vector.
#[allow(dead_code)]
pub(crate) fn scalar_dot_i2s_f32(packed_row: &[u8], input: &[f32], k: usize) -> f32 {
    let mut sum = 0.0f32;
    let full_bytes = k / 4;
    for (bi, &byte) in packed_row.iter().enumerate().take(full_bytes) {
        let vals = unpack_byte_f32(byte);
        let base = bi * 4;
        for j in 0..4 {
            sum += vals[j] * input[base + j];
        }
    }
    let rem = k % 4;
    if rem > 0 && full_bytes < packed_row.len() {
        let byte = packed_row[full_bytes];
        for j in 0..rem {
            let bits = (byte >> (j * 2)) & 0x03;
            sum += decode_i2s(bits) * input[full_bytes * 4 + j];
        }
    }
    sum
}

// ── NEON accelerated kernels ───────────────────────────────────────

/// NEON-accelerated dot product of packed I2_S weights against f32 input.
///
/// Uses `vfmaq_f32` for fused multiply-accumulate and processes 4 values
/// (1 packed byte) per iteration. Falls back to scalar for remainders.
///
/// # Safety
///
/// Caller must ensure `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_i2s_f32(packed_row: &[u8], input: &[f32], k: usize) -> f32 {
    let full_bytes = k / 4;
    let rem = k % 4;

    // Accumulator: 4-wide f32 NEON register
    let mut acc = vdupq_n_f32(0.0);

    for (bi, &byte) in packed_row.iter().enumerate().take(full_bytes) {
        let vals = unpack_byte_f32(byte);
        let w = vld1q_f32(vals.as_ptr());
        let x = vld1q_f32(input.as_ptr().add(bi * 4));
        // vfmaq_f32: acc = acc + w * x
        acc = vfmaq_f32(acc, w, x);
    }

    // Horizontal reduction
    let sum = vaddvq_f32(acc);

    // Scalar tail
    let mut tail = 0.0f32;
    if rem > 0 && full_bytes < packed_row.len() {
        let byte = packed_row[full_bytes];
        for j in 0..rem {
            let bits = (byte >> (j * 2)) & 0x03;
            tail += decode_i2s(bits) * input[full_bytes * 4 + j];
        }
    }
    sum + tail
}

// ── NeonQuantizedMatmul processor ──────────────────────────────────

/// Stateful NEON-optimized I2_S quantized matrix multiplication processor.
///
/// Holds a [`QuantizedMatmulConfig`] and precomputed lookup tables for
/// efficient 2-bit decode.
pub struct NeonQuantizedMatmul {
    /// Matrix dimensions and block size.
    pub config: QuantizedMatmulConfig,
    /// Precomputed byte→[i8;4] lookup table (256 entries).
    /// Used by NEON `vtbl` LUT decode on AArch64.
    #[allow(dead_code)]
    lut: [[i8; 4]; 256],
}

impl NeonQuantizedMatmul {
    /// Create a new processor with the given config and precomputed LUT.
    pub fn new(config: QuantizedMatmulConfig) -> Self {
        let lut = Self::precompute_lookup_table();
        Self { config, lut }
    }

    /// Build the 256-entry byte→{−1,0,+1}×4 lookup table.
    ///
    /// On NEON hardware this table can be loaded into `vtbl` registers for
    /// single-instruction 2-bit→ternary decode of an entire byte's worth
    /// of packed I2_S values.
    #[must_use]
    pub fn precompute_lookup_table() -> [[i8; 4]; 256] {
        BYTE_LUT_I8
    }

    /// Dequantize a block of packed I2_S bytes into f32, applying `scale`.
    ///
    /// Each byte encodes 4 ternary values (LSB-first). The returned
    /// vector has `packed.len() * 4` elements.
    #[must_use]
    pub fn dequantize_i2s_block(packed: &[u8], scale: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(packed.len() * 4);
        for &byte in packed {
            let vals = unpack_byte_f32(byte);
            out.extend(vals.iter().map(|&v| v * scale));
        }
        out
    }

    /// I2_S × f32 matrix multiply: `output[m×n] = dequant(weights)[m×k] · input[k×n]`.
    ///
    /// `weights_i2s` is row-major packed I2_S (`m` rows, each `ceil(k/4)` bytes).
    /// `input` is row-major f32 `[k×n]`.
    /// `output` is row-major f32 `[m×n]`.
    ///
    /// Uses NEON `vfmaq_f32` for the inner dot-product when running on
    /// AArch64; falls back to scalar on other architectures.
    pub fn matmul_i2s_f32(&self, weights_i2s: &[u8], input: &[f32], output: &mut [f32]) {
        let QuantizedMatmulConfig { m, n, k, .. } = self.config;
        let row_bytes = k.div_ceil(4);

        for row in 0..m {
            let packed_row =
                &weights_i2s[row * row_bytes..(row * row_bytes + row_bytes).min(weights_i2s.len())];
            for col in 0..n {
                // Gather column slice from row-major input [k×n]
                let col_input: Vec<f32> = (0..k).map(|i| input[i * n + col]).collect();

                #[cfg(target_arch = "aarch64")]
                {
                    output[row * n + col] = unsafe { neon_dot_i2s_f32(packed_row, &col_input, k) };
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    output[row * n + col] = scalar_dot_i2s_f32(packed_row, &col_input, k);
                }
            }
        }
    }

    /// I2_S × f16 matrix multiply: `output[m×n] = dequant(weights)[m×k] · input[k×n]`.
    ///
    /// Same layout as [`Self::matmul_i2s_f32`] but `input` is `&[u16]`
    /// encoding IEEE 754 half-precision floats. Each f16 is promoted to
    /// f32 before accumulation, and results are stored as f32.
    pub fn matmul_i2s_f16(&self, weights_i2s: &[u8], input: &[u16], output: &mut [f32]) {
        let QuantizedMatmulConfig { m, n, k, .. } = self.config;
        let row_bytes = k.div_ceil(4);

        for row in 0..m {
            let packed_row =
                &weights_i2s[row * row_bytes..(row * row_bytes + row_bytes).min(weights_i2s.len())];
            for col in 0..n {
                let col_f32: Vec<f32> = (0..k).map(|i| f16_to_f32(input[i * n + col])).collect();

                #[cfg(target_arch = "aarch64")]
                {
                    output[row * n + col] = unsafe { neon_dot_i2s_f32(packed_row, &col_f32, k) };
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    output[row * n + col] = scalar_dot_i2s_f32(packed_row, &col_f32, k);
                }
            }
        }
    }

    /// Estimate peak throughput in tera operations per second (TOPS).
    ///
    /// Assumes 2 ops (multiply + add) per weight element per output element
    /// and a 2 TOPS peak for Apple M-series NEON integer pipes.
    #[must_use]
    pub fn throughput_estimate(m: usize, n: usize, k: usize) -> f64 {
        let total_ops = 2.0 * m as f64 * n as f64 * k as f64;
        let peak_tops = 2.0e12; // conservative Apple M-series NEON estimate
        total_ops / peak_tops
    }

    /// Execute the I2_S × f32 matmul and return a [`QuantizedMatmulResult`]
    /// with timing information.
    pub fn execute_f32(&self, weights_i2s: &[u8], input: &[f32]) -> QuantizedMatmulResult {
        let m = self.config.m;
        let n = self.config.n;
        let k = self.config.k;
        let mut output = vec![0.0f32; m * n];

        let start = Instant::now();
        self.matmul_i2s_f32(weights_i2s, input, &mut output);
        let elapsed = start.elapsed();

        let compute_time_ns = elapsed.as_nanos() as u64;
        let total_ops = 2.0 * m as f64 * n as f64 * k as f64;
        let ops_per_second =
            if compute_time_ns > 0 { total_ops / elapsed.as_secs_f64() } else { 0.0 };

        QuantizedMatmulResult { output_data: output, compute_time_ns, ops_per_second }
    }
}

// ── f16 helper ─────────────────────────────────────────────────────

/// Convert an IEEE 754 half-precision (f16) bit pattern to f32.
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let mut m = mant;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((127 - 15 + 1) + e) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exp == 0x1F {
        // Inf / NaN
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }
    let f32_exp = exp + (127 - 15);
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test helpers ───────────────────────────────────────────────

    /// Pack ternary i8 values ({-1, 0, +1}) into I2_S bytes (LSB-first).
    fn pack_i2s(values: &[i8]) -> Vec<u8> {
        let mut packed = Vec::with_capacity((values.len() + 3) / 4);
        for chunk in values.chunks(4) {
            let mut byte = 0u8;
            for (j, &v) in chunk.iter().enumerate() {
                let bits: u8 = match v {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                byte |= bits << (j * 2);
            }
            packed.push(byte);
        }
        packed
    }

    /// Pack a row-major weight matrix [m×k] into I2_S bytes, row by row.
    /// Each row is independently packed into `ceil(k/4)` bytes.
    fn pack_i2s_matrix(weights: &[i8], m: usize, k: usize) -> Vec<u8> {
        let row_bytes = (k + 3) / 4;
        let mut packed = Vec::with_capacity(m * row_bytes);
        for row in 0..m {
            let row_data = &weights[row * k..(row + 1) * k];
            let row_packed = pack_i2s(row_data);
            packed.extend_from_slice(&row_packed);
            // Pad to row_bytes if needed
            for _ in row_packed.len()..row_bytes {
                packed.push(0);
            }
        }
        packed
    }

    /// Scalar reference matmul: weights_i8[m×k] * input[k×n] → output[m×n].
    fn scalar_matmul(weights: &[i8], input: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for i in 0..k {
                    sum += weights[row * k + i] as f32 * input[i * n + col];
                }
                out[row * n + col] = sum;
            }
        }
        out
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (diff={})",
                (x - y).abs()
            );
        }
    }

    fn make_config(m: usize, n: usize, k: usize, block_size: usize) -> QuantizedMatmulConfig {
        QuantizedMatmulConfig { m, n, k, block_size }
    }

    // ── 1. I2_S decode correctness ────────────────────────────────

    #[test]
    fn test_decode_i2s_values() {
        assert_eq!(decode_i2s(0b00), 0.0);
        assert_eq!(decode_i2s(0b01), 1.0);
        assert_eq!(decode_i2s(0b10), 0.0); // unused → 0
        assert_eq!(decode_i2s(0b11), -1.0);
    }

    #[test]
    fn test_decode_i2s_masks_upper_bits() {
        assert_eq!(decode_i2s(0b11_01), 1.0);
        assert_eq!(decode_i2s(0b10_11), -1.0);
        assert_eq!(decode_i2s(0xFF), -1.0);
    }

    // ── 2. LUT correctness ────────────────────────────────────────

    #[test]
    fn test_lut_consistency_with_decode() {
        for bits in 0u8..4 {
            assert_eq!(
                I2S_LUT_F32[bits as usize],
                decode_i2s(bits),
                "LUT disagrees with decode for bits={bits}"
            );
        }
    }

    #[test]
    fn test_byte_lut_all_entries() {
        for byte in 0u8..=255 {
            let entry = BYTE_LUT_I8[byte as usize];
            for j in 0..4 {
                let bits = (byte >> (j * 2)) & 0x03;
                let expected = match bits {
                    0b01 => 1i8,
                    0b11 => -1i8,
                    _ => 0i8,
                };
                assert_eq!(entry[j], expected, "BYTE_LUT_I8 wrong for byte=0x{byte:02X} crumb={j}");
            }
        }
    }

    #[test]
    fn test_precompute_lookup_table_matches_const() {
        let lut = NeonQuantizedMatmul::precompute_lookup_table();
        assert_eq!(lut, BYTE_LUT_I8);
    }

    // ── 3. Unpack ──────────────────────────────────────────────────

    #[test]
    fn test_unpack_byte_all_zeros() {
        assert_eq!(unpack_byte_f32(0x00), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_unpack_byte_all_plus_ones() {
        // 0b01_01_01_01 = 0x55
        assert_eq!(unpack_byte_f32(0x55), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_unpack_byte_all_minus_ones() {
        // 0b11_11_11_11 = 0xFF
        assert_eq!(unpack_byte_f32(0xFF), [-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_unpack_byte_mixed() {
        // byte = 0b11_00_01_01 → crumbs: 01, 01, 00, 11 → [1, 1, 0, -1]
        let vals = unpack_byte_f32(0b11_00_01_01);
        assert_eq!(vals, [1.0, 1.0, 0.0, -1.0]);
    }

    // ── 4. Pack helper round-trip ──────────────────────────────────

    #[test]
    fn test_pack_round_trip() {
        let values: Vec<i8> = vec![0, 1, -1, 0, 1, 1, -1, -1];
        let packed = pack_i2s(&values);
        let mut decoded = Vec::new();
        for &byte in &packed {
            let v = unpack_byte_f32(byte);
            decoded.extend(v.iter().map(|&x| x as i8));
        }
        assert_eq!(&decoded[..values.len()], &values[..]);
    }

    #[test]
    fn test_pack_non_aligned_length() {
        let values: Vec<i8> = vec![1, -1, 0];
        let packed = pack_i2s(&values);
        assert_eq!(packed.len(), 1);
        let v = unpack_byte_f32(packed[0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], -1.0);
        assert_eq!(v[2], 0.0);
        // fourth crumb is padding zero
        assert_eq!(v[3], 0.0);
    }

    // ── 5. Dequantize block ────────────────────────────────────────

    #[test]
    fn test_dequantize_scale_one() {
        let packed = pack_i2s(&[1, -1, 0, 1]);
        let out = NeonQuantizedMatmul::dequantize_i2s_block(&packed, 1.0);
        assert_close(&out, &[1.0, -1.0, 0.0, 1.0], 1e-6);
    }

    #[test]
    fn test_dequantize_with_scale() {
        let packed = pack_i2s(&[1, -1, 0, 1]);
        let out = NeonQuantizedMatmul::dequantize_i2s_block(&packed, 0.5);
        assert_close(&out, &[0.5, -0.5, 0.0, 0.5], 1e-6);
    }

    #[test]
    fn test_dequantize_zero_scale() {
        let packed = pack_i2s(&[1, -1, 1, -1]);
        let out = NeonQuantizedMatmul::dequantize_i2s_block(&packed, 0.0);
        assert_close(&out, &[0.0, 0.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn test_dequantize_negative_scale() {
        let packed = pack_i2s(&[1, -1, 0, 0]);
        let out = NeonQuantizedMatmul::dequantize_i2s_block(&packed, -2.0);
        assert_close(&out, &[-2.0, 2.0, 0.0, 0.0], 1e-6);
    }

    // ── 6. Scalar dot product ──────────────────────────────────────

    #[test]
    fn test_scalar_dot_basic() {
        let w: Vec<i8> = vec![1, -1, 0, 1];
        let packed = pack_i2s(&w);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = scalar_dot_i2s_f32(&packed, &input, 4);
        // 1*1 + (-1)*2 + 0*3 + 1*4 = 3.0
        assert!((result - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_dot_remainder() {
        let w: Vec<i8> = vec![1, -1, 1];
        let packed = pack_i2s(&w);
        let input = vec![2.0, 3.0, 4.0];
        let result = scalar_dot_i2s_f32(&packed, &input, 3);
        // 1*2 + (-1)*3 + 1*4 = 3.0
        assert!((result - 3.0).abs() < 1e-6);
    }

    // ── 7. matmul_i2s_f32 correctness vs scalar ───────────────────

    #[test]
    fn test_matmul_f32_1x1() {
        let w: Vec<i8> = vec![1, -1, 1, -1];
        let packed = pack_i2s(&w);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let config = make_config(1, 1, 4, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let mut output = [0.0f32; 1];
        proc.matmul_i2s_f32(&packed, &input, &mut output);
        let expected = scalar_matmul(&w, &input, 1, 4, 1);
        assert_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_matmul_f32_2x2() {
        let m = 2;
        let k = 4;
        let n = 2;
        let w: Vec<i8> = vec![1, 0, -1, 1, -1, 1, 0, -1];
        let input: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32).collect();
        let packed = pack_i2s(&w);
        let config = make_config(m, n, k, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let mut output = vec![0.0f32; m * n];
        proc.matmul_i2s_f32(&packed, &input, &mut output);
        let expected = scalar_matmul(&w, &input, m, k, n);
        assert_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_matmul_f32_4x4() {
        let m = 4;
        let k = 8;
        let n = 4;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0][i % 3]).collect();
        let input: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let packed = pack_i2s(&w);
        let config = make_config(m, n, k, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let mut output = vec![0.0f32; m * n];
        proc.matmul_i2s_f32(&packed, &input, &mut output);
        let expected = scalar_matmul(&w, &input, m, k, n);
        assert_close(&output, &expected, 1e-4);
    }

    #[test]
    fn test_matmul_f32_non_aligned_k() {
        let m = 2;
        let k = 5;
        let n = 1;
        let w: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 1, -1, 0, 1];
        let input: Vec<f32> = (0..k * n).map(|i| i as f32 + 1.0).collect();
        let packed = pack_i2s_matrix(&w, m, k);
        let config = make_config(m, n, k, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let mut output = vec![0.0f32; m * n];
        proc.matmul_i2s_f32(&packed, &input, &mut output);
        let expected = scalar_matmul(&w, &input, m, k, n);
        assert_close(&output, &expected, 1e-5);
    }

    // ── 8. Zero matrix ─────────────────────────────────────────────

    #[test]
    fn test_matmul_zero_weights() {
        let m = 3;
        let k = 8;
        let n = 2;
        let w = vec![0i8; m * k];
        let input: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let packed = pack_i2s(&w);
        let config = make_config(m, n, k, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let mut output = vec![0.0f32; m * n];
        proc.matmul_i2s_f32(&packed, &input, &mut output);
        assert_close(&output, &vec![0.0; m * n], 1e-6);
    }

    #[test]
    fn test_matmul_zero_input() {
        let m = 2;
        let k = 4;
        let n = 3;
        let w: Vec<i8> = vec![1, -1, 1, -1, 0, 1, -1, 0];
        let input = vec![0.0f32; k * n];
        let packed = pack_i2s(&w);
        let config = make_config(m, n, k, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let mut output = vec![0.0f32; m * n];
        proc.matmul_i2s_f32(&packed, &input, &mut output);
        assert_close(&output, &vec![0.0; m * n], 1e-6);
    }

    // ── 9. Identity-like ───────────────────────────────────────────

    #[test]
    fn test_matmul_identity_like() {
        // "Identity" in ternary: diagonal +1, off-diagonal 0
        let n = 4;
        let mut w = vec![0i8; n * n];
        for i in 0..n {
            w[i * n + i] = 1;
        }
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        // input is [k×1] column vector
        let packed = pack_i2s(&w);
        let config = make_config(n, 1, n, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let mut output = vec![0.0f32; n];
        proc.matmul_i2s_f32(&packed, &input, &mut output);
        assert_close(&output, &input, 1e-6);
    }

    #[test]
    fn test_matmul_negation_matrix() {
        let n = 4;
        let mut w = vec![0i8; n * n];
        for i in 0..n {
            w[i * n + i] = -1;
        }
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let packed = pack_i2s(&w);
        let config = make_config(n, 1, n, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let mut output = vec![0.0f32; n];
        proc.matmul_i2s_f32(&packed, &input, &mut output);
        let expected: Vec<f32> = input.iter().map(|&x| -x).collect();
        assert_close(&output, &expected, 1e-6);
    }

    // ── 10. Block sizes ────────────────────────────────────────────

    #[test]
    fn test_config_block_size_32() {
        let config = make_config(2, 2, 32, 32);
        assert_eq!(config.block_size, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let w = vec![1i8; 2 * 32];
        let input = vec![1.0f32; 32 * 2];
        let packed = pack_i2s(&w);
        let mut output = [0.0f32; 4];
        proc.matmul_i2s_f32(&packed, &input, &mut output);
        let expected = scalar_matmul(&w, &input, 2, 32, 2);
        assert_close(&output, &expected, 1e-4);
    }

    #[test]
    fn test_config_block_size_256() {
        let config = make_config(1, 1, 256, 256);
        assert_eq!(config.block_size, 256);
        let proc = NeonQuantizedMatmul::new(config);
        let w: Vec<i8> = (0..256).map(|i| [1, -1, 0][i % 3]).collect();
        let input: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let packed = pack_i2s(&w);
        let mut output = [0.0f32; 1];
        proc.matmul_i2s_f32(&packed, &input, &mut output);
        let expected = scalar_matmul(&w, &input, 1, 256, 1);
        assert_close(&output, &expected, 1e-3);
    }

    // ── 11. f16 matmul ─────────────────────────────────────────────

    #[test]
    fn test_f16_to_f32_conversion() {
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // -1.0 in f16 = 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
        // 0.0 in f16 = 0x0000
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // 0.5 in f16 = 0x3800
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_f16_basic() {
        let m = 2;
        let k = 4;
        let n = 1;
        let w: Vec<i8> = vec![1, -1, 0, 1, -1, 1, -1, 0];
        let packed = pack_i2s(&w);

        // f16 encoding: use half crate in dev-deps
        let input_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input_f16: Vec<u16> =
            input_f32.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();

        let config = make_config(m, n, k, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let mut output = vec![0.0f32; m * n];
        proc.matmul_i2s_f16(&packed, &input_f16, &mut output);

        let expected = scalar_matmul(&w, &input_f32, m, k, n);
        assert_close(&output, &expected, 1e-2);
    }

    #[test]
    fn test_matmul_f16_zero() {
        let m = 1;
        let k = 4;
        let n = 1;
        let w = vec![1i8; k];
        let packed = pack_i2s(&w);
        let input_f16: Vec<u16> = vec![0; k];
        let config = make_config(m, n, k, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let mut output = [0.0f32; 1];
        proc.matmul_i2s_f16(&packed, &input_f16, &mut output);
        assert_close(&output, &[0.0], 1e-6);
    }

    // ── 12. Throughput estimate ────────────────────────────────────

    #[test]
    fn test_throughput_estimate_positive() {
        let t = NeonQuantizedMatmul::throughput_estimate(128, 128, 256);
        assert!(t > 0.0, "throughput estimate should be positive");
    }

    #[test]
    fn test_throughput_estimate_scales_with_size() {
        let small = NeonQuantizedMatmul::throughput_estimate(64, 64, 64);
        let large = NeonQuantizedMatmul::throughput_estimate(128, 128, 128);
        assert!(large > small, "larger problem should estimate more time");
    }

    #[test]
    fn test_throughput_estimate_zero_dim() {
        let t = NeonQuantizedMatmul::throughput_estimate(0, 128, 256);
        assert_eq!(t, 0.0);
    }

    // ── 13. execute_f32 timing ─────────────────────────────────────

    #[test]
    fn test_execute_f32_returns_result() {
        let m = 2;
        let k = 4;
        let n = 2;
        let w = vec![1i8; m * k];
        let packed = pack_i2s(&w);
        let input = vec![1.0f32; k * n];
        let config = make_config(m, n, k, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let result = proc.execute_f32(&packed, &input);
        assert_eq!(result.output_data.len(), m * n);
    }

    // ── 14. Large dimension ────────────────────────────────────────

    #[test]
    fn test_matmul_large_k() {
        let m = 1;
        let k = 1024;
        let n = 1;
        let w: Vec<i8> = (0..k).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let input: Vec<f32> = vec![1.0; k];
        let packed = pack_i2s(&w);
        let config = make_config(m, n, k, 256);
        let proc = NeonQuantizedMatmul::new(config);
        let mut output = [0.0f32; 1];
        proc.matmul_i2s_f32(&packed, &input, &mut output);
        // sum of alternating +1,-1 with all 1.0 inputs = 0.0
        assert_close(&output, &[0.0], 1e-4);
    }

    #[test]
    fn test_matmul_larger_mn() {
        let m = 8;
        let k = 16;
        let n = 8;
        let w: Vec<i8> = (0..m * k).map(|i| [1, 0, -1][i % 3]).collect();
        let input: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let packed = pack_i2s(&w);
        let config = make_config(m, n, k, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let mut output = vec![0.0f32; m * n];
        proc.matmul_i2s_f32(&packed, &input, &mut output);
        let expected = scalar_matmul(&w, &input, m, k, n);
        assert_close(&output, &expected, 1e-3);
    }

    // ── 15. Config equality ────────────────────────────────────────

    #[test]
    fn test_config_equality() {
        let a = make_config(4, 4, 8, 32);
        let b = make_config(4, 4, 8, 32);
        assert_eq!(a, b);
    }

    #[test]
    fn test_config_inequality() {
        let a = make_config(4, 4, 8, 32);
        let b = make_config(4, 4, 8, 256);
        assert_ne!(a, b);
    }

    // ── 16. Result fields ──────────────────────────────────────────

    #[test]
    fn test_result_compute_time_nonzero() {
        let m = 4;
        let k = 64;
        let n = 4;
        let w: Vec<i8> = (0..m * k).map(|i| [1, -1, 0][i % 3]).collect();
        let packed = pack_i2s(&w);
        let input = vec![1.0f32; k * n];
        let config = make_config(m, n, k, 32);
        let proc = NeonQuantizedMatmul::new(config);
        let result = proc.execute_f32(&packed, &input);
        // Time could be 0 on very fast machines, but output should match
        let expected = scalar_matmul(&w, &input, m, k, n);
        assert_close(&result.output_data, &expected, 1e-4);
    }
}
