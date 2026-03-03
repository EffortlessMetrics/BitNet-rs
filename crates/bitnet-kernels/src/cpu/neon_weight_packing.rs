//! NEON-accelerated weight packing/unpacking for ternary and 2-bit quantized models.
//!
//! Provides high-throughput packing of ternary {-1, 0, +1} weights into 2-bit
//! encoding and binary {-1, +1} weights into single-bit encoding, using ARM
//! NEON SIMD intrinsics on Apple Silicon (aarch64).
//!
//! ## Ternary encoding (2 bits per value, 4 values per byte, LSB-first)
//!
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1
//!
//! ## Binary encoding (1 bit per value, 8 values per byte, LSB-first)
//!
//! - `0` → −1
//! - `1` → +1

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc, clippy::float_cmp, clippy::manual_div_ceil, clippy::unnecessary_cast, clippy::needless_range_loop, clippy::too_many_arguments, clippy::collapsible_if, clippy::let_and_return, clippy::derivable_impls, clippy::excessive_precision, clippy::manual_is_multiple_of)]
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Scalar helpers ─────────────────────────────────────────────────────

/// Encode a single ternary value into its 2-bit representation.
#[inline(always)]
fn encode_ternary(v: i8) -> u8 {
    match v {
        1 => 0b01,
        -1 => 0b11,
        _ => 0b00, // 0 or anything else maps to 0
    }
}

/// Decode a single 2-bit code to a ternary i8 value.
#[inline(always)]
fn decode_ternary(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

// ── Ternary packing ───────────────────────────────────────────────────

/// Pack ternary {-1, 0, +1} weights into 2-bit encoding.
///
/// Each output byte stores 4 ternary values (2 bits each, LSB-first).
/// `packed` must have length `ceil(weights.len() / 4)`.
///
/// Uses NEON SIMD to process 16 weights (4 output bytes) at a time with
/// scalar tail processing for the remainder.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn pack_ternary_weights(weights: &[i8], packed: &mut [u8]) {
    let n = weights.len();
    let required = (n + 3) / 4;
    assert!(
        packed.len() >= required,
        "packed buffer too small: need {required}, got {}",
        packed.len()
    );

    let mut wi = 0; // weight index
    let mut pi = 0; // packed index

    // NEON path: process 16 weights → 4 packed bytes at a time
    let ones = vdupq_n_s8(1);
    let neg_ones = vdupq_n_s8(-1);

    while wi + 16 <= n {
        let v = vld1q_s8(weights.as_ptr().add(wi));
        // bit0: set where value == 1 or value == -1 (i.e. non-zero)
        let is_pos = vceqq_s8(v, ones);
        let is_neg = vceqq_s8(v, neg_ones);
        // Ternary: 0b01 for +1, 0b11 for -1, 0b00 for 0
        // bit0 = is_pos | is_neg (non-zero indicator)
        // bit1 = is_neg
        let bit0 = vorrq_u8(is_pos, is_neg);
        let bit1 = is_neg;

        // Extract results to scalar and pack 4 values per byte
        let mut b0 = [0u8; 16];
        let mut b1 = [0u8; 16];
        vst1q_u8(b0.as_mut_ptr(), bit0);
        vst1q_u8(b1.as_mut_ptr(), bit1);

        for chunk in 0..4 {
            let base = chunk * 4;
            let mut byte = 0u8;
            for j in 0..4 {
                let lo = b0[base + j] & 1;
                let hi = b1[base + j] & 1;
                byte |= (lo | (hi << 1)) << (j * 2);
            }
            packed[pi] = byte;
            pi += 1;
        }
        wi += 16;
    }

    // Scalar tail: pack remaining weights 4 at a time
    while wi < n {
        let mut byte = 0u8;
        for j in 0..4 {
            if wi + j < n {
                byte |= encode_ternary(weights[wi + j]) << (j * 2);
            }
        }
        packed[pi] = byte;
        pi += 1;
        wi += 4;
    }
}

/// Unpack 2-bit ternary encoded data back to i8 weights.
///
/// Each byte of `packed` contains 4 ternary values (2 bits, LSB-first).
/// `weights` must have length ≥ `packed.len() * 4` (or the caller may
/// provide a shorter buffer; only `weights.len()` values are written).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn unpack_ternary_weights(packed: &[u8], weights: &mut [i8]) {
    let n = weights.len();
    let mut wi = 0;
    let mut pi = 0;

    // NEON path: unpack 4 bytes → 16 weights at a time
    while wi + 16 <= n && pi + 4 <= packed.len() {
        let mut buf = [0i8; 16];
        for b in 0..4 {
            let byte = packed[pi + b];
            for j in 0..4 {
                buf[b * 4 + j] = decode_ternary((byte >> (j * 2)) as u8);
            }
        }
        let v = vld1q_s8(buf.as_ptr());
        vst1q_s8(weights.as_mut_ptr().add(wi), v);
        wi += 16;
        pi += 4;
    }

    // Scalar tail
    while wi < n && pi < packed.len() {
        let byte = packed[pi];
        for j in 0..4 {
            if wi < n {
                weights[wi] = decode_ternary((byte >> (j * 2)) as u8);
                wi += 1;
            }
        }
        pi += 1;
    }
}

// ── Binary packing ────────────────────────────────────────────────────

/// Pack binary {-1, +1} weights into bits (1 bit per value, LSB-first).
///
/// `packed` must have length ≥ `ceil(weights.len() / 8)`.
/// +1 encodes as bit 1, -1 encodes as bit 0.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn pack_binary_weights(weights: &[i8], packed: &mut [u8]) {
    let n = weights.len();
    let required = (n + 7) / 8;
    assert!(
        packed.len() >= required,
        "packed buffer too small: need {required}, got {}",
        packed.len()
    );

    let mut wi = 0;
    let mut pi = 0;

    // NEON path: compare 16 weights at once, extract bit mask for 2 bytes
    let ones = vdupq_n_s8(1);

    while wi + 16 <= n {
        let v = vld1q_s8(weights.as_ptr().add(wi));
        let mask = vceqq_s8(v, ones); // 0xFF where +1, 0x00 where -1

        // Extract per-byte sign bits into two packed bytes
        let mut m = [0u8; 16];
        vst1q_u8(m.as_mut_ptr(), mask);

        for byte_idx in 0..2 {
            let base = byte_idx * 8;
            let mut byte = 0u8;
            for bit in 0..8 {
                if m[base + bit] != 0 {
                    byte |= 1 << bit;
                }
            }
            packed[pi] = byte;
            pi += 1;
        }
        wi += 16;
    }

    // Scalar tail
    while wi < n {
        let mut byte = 0u8;
        for bit in 0..8 {
            if wi + bit < n && weights[wi + bit] == 1 {
                byte |= 1 << bit;
            }
        }
        packed[pi] = byte;
        pi += 1;
        wi += 8;
    }
}

/// Unpack binary bits to i8 {-1, +1}.
///
/// `count` specifies the exact number of weights to unpack.
/// Bit 1 → +1, bit 0 → -1.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn unpack_binary_weights(packed: &[u8], weights: &mut [i8], count: usize) {
    assert!(
        weights.len() >= count,
        "weights buffer too small: need {count}, got {}",
        weights.len()
    );
    let required_bytes = (count + 7) / 8;
    assert!(
        packed.len() >= required_bytes,
        "packed buffer too small: need {required_bytes} bytes, got {}",
        packed.len()
    );

    let mut wi = 0;
    let mut pi = 0;

    // NEON path: unpack 2 bytes → 16 weights at a time
    while wi + 16 <= count && pi + 2 <= packed.len() {
        let mut buf = [0i8; 16];
        for byte_idx in 0..2 {
            let byte = packed[pi + byte_idx];
            for bit in 0..8 {
                buf[byte_idx * 8 + bit] = if (byte >> bit) & 1 == 1 { 1 } else { -1 };
            }
        }
        let v = vld1q_s8(buf.as_ptr());
        vst1q_s8(weights.as_mut_ptr().add(wi), v);
        wi += 16;
        pi += 2;
    }

    // Scalar tail
    while wi < count && pi < packed.len() {
        let byte = packed[pi];
        for bit in 0..8 {
            if wi < count {
                weights[wi] = if (byte >> bit) & 1 == 1 { 1 } else { -1 };
                wi += 1;
            }
        }
        pi += 1;
    }
}

// ── Ternary matmul ────────────────────────────────────────────────────

/// Matrix multiply with packed ternary weights.
///
/// Computes `output[r] = sum_c(weight[r][c] * input[c])` where weights are
/// packed in 2-bit ternary format (row-major, 4 values per byte).
///
/// - `packed_weights`: row-major packed ternary, `rows * ceil(cols/4)` bytes.
/// - `input`: f32 vector of length `cols`.
/// - `output`: f32 vector of length `rows` (accumulated, not zeroed).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn ternary_matmul_packed(
    packed_weights: &[u8],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    let row_bytes = (cols + 3) / 4;
    assert!(
        packed_weights.len() >= rows * row_bytes,
        "packed_weights too small: need {}, got {}",
        rows * row_bytes,
        packed_weights.len()
    );
    assert!(input.len() >= cols, "input too small: need {cols}, got {}", input.len());
    assert!(output.len() >= rows, "output too small: need {rows}, got {}", output.len());

    for r in 0..rows {
        let row_start = r * row_bytes;
        let mut acc = vdupq_n_f32(0.0);
        let mut ci = 0; // column index
        let mut bi = 0; // byte index within row

        // NEON: process 4 columns (1 packed byte) at a time using f32x4
        while ci + 4 <= cols && bi < row_bytes {
            let byte = packed_weights[row_start + bi];
            let mut w = [0.0f32; 4];
            for j in 0..4 {
                w[j] = match decode_ternary((byte >> (j * 2)) as u8) {
                    1 => 1.0,
                    -1 => -1.0,
                    _ => 0.0,
                };
            }
            let wv = vld1q_f32(w.as_ptr());
            let iv = vld1q_f32(input.as_ptr().add(ci));
            acc = vfmaq_f32(acc, wv, iv);
            ci += 4;
            bi += 1;
        }

        // Horizontal sum of NEON accumulator
        let sum = vaddvq_f32(acc);
        let mut scalar_sum = sum;

        // Scalar tail for remaining columns
        if ci < cols && bi < row_bytes {
            let byte = packed_weights[row_start + bi];
            for j in 0..4 {
                if ci < cols {
                    let w = match decode_ternary((byte >> (j * 2)) as u8) {
                        1 => 1.0f32,
                        -1 => -1.0,
                        _ => 0.0,
                    };
                    scalar_sum += w * input[ci];
                    ci += 1;
                }
            }
        }

        output[r] += scalar_sum;
    }
}

// ── I2_S dequantize ───────────────────────────────────────────────────

/// Dequantize an I2_S block: unpack 2-bit ternary values and multiply by scale.
///
/// Each byte of `packed` contains 4 ternary values. Output receives
/// `packed.len() * 4` floats (capped by `output.len()`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dequantize_i2s_block(packed: &[u8], scale: f32, output: &mut [f32]) {
    let n = output.len().min(packed.len() * 4);
    let scale_v = vdupq_n_f32(scale);
    let mut oi = 0;
    let mut pi = 0;

    // NEON: process 4 packed bytes → 16 floats at a time
    while oi + 16 <= n && pi + 4 <= packed.len() {
        for b in 0..4 {
            let byte = packed[pi + b];
            let mut w = [0.0f32; 4];
            for j in 0..4 {
                w[j] = match decode_ternary((byte >> (j * 2)) as u8) {
                    1 => 1.0,
                    -1 => -1.0,
                    _ => 0.0,
                };
            }
            let wv = vld1q_f32(w.as_ptr());
            let scaled = vmulq_f32(wv, scale_v);
            vst1q_f32(output.as_mut_ptr().add(oi), scaled);
            oi += 4;
        }
        pi += 4;
    }

    // Scalar tail
    while oi < n && pi < packed.len() {
        let byte = packed[pi];
        for j in 0..4 {
            if oi < n {
                let val = match decode_ternary((byte >> (j * 2)) as u8) {
                    1 => scale,
                    -1 => -scale,
                    _ => 0.0,
                };
                output[oi] = val;
                oi += 1;
            }
        }
        pi += 1;
    }
}

// ── Quantize to ternary ───────────────────────────────────────────────

/// Quantize f32 values to ternary {-1, 0, +1} based on a threshold.
///
/// - Values > `threshold` become +1
/// - Values < `-threshold` become -1
/// - Otherwise 0
///
/// Uses NEON to compare 4 floats at a time.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn quantize_to_ternary(values: &[f32], threshold: f32) -> Vec<i8> {
    let n = values.len();
    let mut result = vec![0i8; n];

    let pos_thresh = vdupq_n_f32(threshold);
    let neg_thresh = vdupq_n_f32(-threshold);
    let mut i = 0;

    // NEON: compare 4 floats at a time
    while i + 4 <= n {
        let v = vld1q_f32(values.as_ptr().add(i));
        let gt_pos = vcgtq_f32(v, pos_thresh); // > threshold
        let lt_neg = vcltq_f32(v, neg_thresh); // < -threshold

        let mut gt = [0u32; 4];
        let mut lt = [0u32; 4];
        vst1q_u32(gt.as_mut_ptr(), gt_pos);
        vst1q_u32(lt.as_mut_ptr(), lt_neg);

        for j in 0..4 {
            if gt[j] != 0 {
                result[i + j] = 1;
            } else if lt[j] != 0 {
                result[i + j] = -1;
            }
            // else stays 0
        }
        i += 4;
    }

    // Scalar tail
    while i < n {
        if values[i] > threshold {
            result[i] = 1;
        } else if values[i] < -threshold {
            result[i] = -1;
        }
        i += 1;
    }

    result
}

// ── Popcount ──────────────────────────────────────────────────────────

/// Vectorized popcount over a byte slice using NEON `cnt` instruction.
///
/// Returns the total number of set bits across all bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_popcount_u8x16(data: &[u8]) -> u32 {
    let n = data.len();
    let mut total: u64 = 0;
    let mut i = 0;

    // NEON: process 16 bytes at a time using vcntq_u8
    let mut acc = vdupq_n_u8(0);

    while i + 16 <= n {
        let v = vld1q_u8(data.as_ptr().add(i));
        let bits = vcntq_u8(v); // per-byte popcount
        acc = vaddq_u8(acc, bits);
        i += 16;

        // Flush accumulator every 255 iterations to avoid u8 overflow
        if (i / 16) % 255 == 0 {
            total += vaddlvq_u8(acc) as u64;
            acc = vdupq_n_u8(0);
        }
    }

    // Flush remaining NEON accumulator
    total += vaddlvq_u8(acc) as u64;

    // Scalar tail
    while i < n {
        total += data[i].count_ones() as u64;
        i += 1;
    }

    total as u32
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    // ── Ternary pack/unpack roundtrip ──────────────────────────────────

    #[test]
    fn test_ternary_roundtrip_basic() {
        let weights: Vec<i8> = vec![0, 1, -1, 0];
        let mut packed = vec![0u8; 1];
        let mut unpacked = vec![0i8; 4];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
            unpack_ternary_weights(&packed, &mut unpacked);
        }
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_ternary_roundtrip_16_elements() {
        let weights: Vec<i8> = vec![1, -1, 0, 1, -1, -1, 0, 0, 1, 1, 1, -1, 0, -1, 1, 0];
        let mut packed = vec![0u8; 4];
        let mut unpacked = vec![0i8; 16];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
            unpack_ternary_weights(&packed, &mut unpacked);
        }
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_ternary_roundtrip_17_elements() {
        let weights: Vec<i8> = vec![1, -1, 0, 1, -1, -1, 0, 0, 1, 1, 1, -1, 0, -1, 1, 0, -1];
        let mut packed = vec![0u8; 5]; // ceil(17/4) = 5
        let mut unpacked = vec![0i8; 17];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
            unpack_ternary_weights(&packed, &mut unpacked);
        }
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_ternary_roundtrip_all_zeros() {
        let weights = vec![0i8; 32];
        let mut packed = vec![0u8; 8];
        let mut unpacked = vec![99i8; 32];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
            unpack_ternary_weights(&packed, &mut unpacked);
        }
        assert_eq!(unpacked, weights);
        assert!(packed.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_ternary_roundtrip_all_ones() {
        let weights = vec![1i8; 20];
        let mut packed = vec![0u8; 5];
        let mut unpacked = vec![0i8; 20];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
            unpack_ternary_weights(&packed, &mut unpacked);
        }
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_ternary_roundtrip_all_neg_ones() {
        let weights = vec![-1i8; 20];
        let mut packed = vec![0u8; 5];
        let mut unpacked = vec![0i8; 20];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
            unpack_ternary_weights(&packed, &mut unpacked);
        }
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_ternary_roundtrip_single_element() {
        for &val in &[-1i8, 0, 1] {
            let weights = vec![val];
            let mut packed = vec![0u8; 1];
            let mut unpacked = vec![0i8; 1];
            unsafe {
                pack_ternary_weights(&weights, &mut packed);
                unpack_ternary_weights(&packed, &mut unpacked);
            }
            assert_eq!(unpacked[0], val, "roundtrip failed for {val}");
        }
    }

    #[test]
    fn test_ternary_roundtrip_empty() {
        let weights: Vec<i8> = vec![];
        let mut packed: Vec<u8> = vec![];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
        }
        // Nothing to unpack
    }

    #[test]
    fn test_ternary_pack_known_values() {
        // [0, 1, -1, 0] → byte: 0b00_11_01_00 = 0x0C + 0x04 = bits: 00|11|01|00 = 0b00110100
        // LSB-first: val0 at bits[1:0], val1 at bits[3:2], val2 at bits[5:4], val3 at bits[7:6]
        // 0→0b00, 1→0b01, -1→0b11, 0→0b00 = 0b00_11_01_00 = 0x34
        let weights: Vec<i8> = vec![0, 1, -1, 0];
        let mut packed = vec![0u8; 1];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
        }
        assert_eq!(packed[0], 0b00_11_01_00);
    }

    #[test]
    fn test_ternary_pack_all_neg_one_known() {
        // [-1, -1, -1, -1] → 0b11_11_11_11 = 0xFF
        let weights: Vec<i8> = vec![-1, -1, -1, -1];
        let mut packed = vec![0u8; 1];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
        }
        assert_eq!(packed[0], 0xFF);
    }

    #[test]
    fn test_ternary_pack_all_pos_one_known() {
        // [1, 1, 1, 1] → 0b01_01_01_01 = 0x55
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let mut packed = vec![0u8; 1];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
        }
        assert_eq!(packed[0], 0x55);
    }

    #[test]
    fn test_ternary_roundtrip_large() {
        let n = 1000;
        let weights: Vec<i8> = (0..n).map(|i| ((i % 3) as i8) - 1).collect();
        let packed_len = (n + 3) / 4;
        let mut packed = vec![0u8; packed_len];
        let mut unpacked = vec![0i8; n];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
            unpack_ternary_weights(&packed, &mut unpacked);
        }
        assert_eq!(unpacked, weights);
    }

    // ── Binary pack/unpack roundtrip ──────────────────────────────────

    #[test]
    fn test_binary_roundtrip_basic() {
        let weights: Vec<i8> = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let mut packed = vec![0u8; 1];
        let mut unpacked = vec![0i8; 8];
        unsafe {
            pack_binary_weights(&weights, &mut packed);
            unpack_binary_weights(&packed, &mut unpacked, 8);
        }
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_binary_roundtrip_16_elements() {
        let weights: Vec<i8> = vec![1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1];
        let mut packed = vec![0u8; 2];
        let mut unpacked = vec![0i8; 16];
        unsafe {
            pack_binary_weights(&weights, &mut packed);
            unpack_binary_weights(&packed, &mut unpacked, 16);
        }
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_binary_roundtrip_17_elements() {
        let weights: Vec<i8> = vec![1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1];
        let mut packed = vec![0u8; 3]; // ceil(17/8)=3
        let mut unpacked = vec![0i8; 17];
        unsafe {
            pack_binary_weights(&weights, &mut packed);
            unpack_binary_weights(&packed, &mut unpacked, 17);
        }
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_binary_roundtrip_all_positive() {
        let weights = vec![1i8; 24];
        let mut packed = vec![0u8; 3];
        let mut unpacked = vec![0i8; 24];
        unsafe {
            pack_binary_weights(&weights, &mut packed);
            unpack_binary_weights(&packed, &mut unpacked, 24);
        }
        assert_eq!(unpacked, weights);
        assert!(packed.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_binary_roundtrip_all_negative() {
        let weights = vec![-1i8; 24];
        let mut packed = vec![0u8; 3];
        let mut unpacked = vec![0i8; 24];
        unsafe {
            pack_binary_weights(&weights, &mut packed);
            unpack_binary_weights(&packed, &mut unpacked, 24);
        }
        assert_eq!(unpacked, weights);
        assert!(packed.iter().all(|&b| b == 0x00));
    }

    #[test]
    fn test_binary_roundtrip_single() {
        for &val in &[-1i8, 1] {
            let weights = vec![val];
            let mut packed = vec![0u8; 1];
            let mut unpacked = vec![0i8; 1];
            unsafe {
                pack_binary_weights(&weights, &mut packed);
                unpack_binary_weights(&packed, &mut unpacked, 1);
            }
            assert_eq!(unpacked[0], val, "roundtrip failed for {val}");
        }
    }

    #[test]
    fn test_binary_roundtrip_empty() {
        let weights: Vec<i8> = vec![];
        let mut packed: Vec<u8> = vec![];
        unsafe {
            pack_binary_weights(&weights, &mut packed);
        }
    }

    #[test]
    fn test_binary_pack_known() {
        // [1, -1, 1, 1, -1, -1, 1, -1] → bits: 1,0,1,1,0,0,1,0 = 0b01001101 = 0x4D
        let weights: Vec<i8> = vec![1, -1, 1, 1, -1, -1, 1, -1];
        let mut packed = vec![0u8; 1];
        unsafe {
            pack_binary_weights(&weights, &mut packed);
        }
        assert_eq!(packed[0], 0b01_00_11_01);
    }

    #[test]
    fn test_binary_roundtrip_large() {
        let n = 1000;
        let weights: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let packed_len = (n + 7) / 8;
        let mut packed = vec![0u8; packed_len];
        let mut unpacked = vec![0i8; n];
        unsafe {
            pack_binary_weights(&weights, &mut packed);
            unpack_binary_weights(&packed, &mut unpacked, n);
        }
        assert_eq!(unpacked, weights);
    }

    // ── Ternary matmul ────────────────────────────────────────────────

    #[test]
    fn test_matmul_identity_like() {
        // 2x2 identity-like with ternary weights [1,0,0,1]
        let weights: Vec<i8> = vec![1, 0, 0, 1];
        let mut packed = vec![0u8; 2]; // 2 rows × 1 byte each (2 cols → ceil(2/4)=1)
        unsafe {
            pack_ternary_weights(&weights[0..2], &mut packed[0..1]);
            pack_ternary_weights(&weights[2..4], &mut packed[1..2]);
        }
        let input = vec![3.0f32, 7.0];
        let mut output = vec![0.0f32; 2];
        unsafe {
            ternary_matmul_packed(&packed, &input, &mut output, 2, 2);
        }
        assert!((output[0] - 3.0).abs() < 1e-6);
        assert!((output[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_negation() {
        // 1x4 row of all -1s
        let weights: Vec<i8> = vec![-1, -1, -1, -1];
        let mut packed = vec![0u8; 1];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
        }
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 1];
        unsafe {
            ternary_matmul_packed(&packed, &input, &mut output, 1, 4);
        }
        assert!((output[0] - (-10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_zeros() {
        let weights: Vec<i8> = vec![0, 0, 0, 0];
        let mut packed = vec![0u8; 1];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
        }
        let input = vec![100.0f32, 200.0, 300.0, 400.0];
        let mut output = vec![0.0f32; 1];
        unsafe {
            ternary_matmul_packed(&packed, &input, &mut output, 1, 4);
        }
        assert!((output[0]).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_accumulates() {
        // output should accumulate, not overwrite
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let mut packed = vec![0u8; 1];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
        }
        let input = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut output = vec![5.0f32; 1]; // start with 5.0
        unsafe {
            ternary_matmul_packed(&packed, &input, &mut output, 1, 4);
        }
        assert!((output[0] - 9.0).abs() < 1e-6); // 5.0 + 4.0
    }

    #[test]
    fn test_matmul_non_aligned_cols() {
        // 1 row, 5 cols (not multiple of 4)
        let weights: Vec<i8> = vec![1, -1, 1, 0, -1];
        let row_bytes = (5 + 3) / 4; // 2 bytes
        let mut packed = vec![0u8; row_bytes];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
        }
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 1];
        unsafe {
            ternary_matmul_packed(&packed, &input, &mut output, 1, 5);
        }
        // 1*1 + (-1)*2 + 1*3 + 0*4 + (-1)*5 = 1 - 2 + 3 + 0 - 5 = -3
        assert!((output[0] - (-3.0)).abs() < 1e-6);
    }

    // ── Dequantize I2_S ───────────────────────────────────────────────

    #[test]
    fn test_dequantize_scale_1() {
        let weights: Vec<i8> = vec![0, 1, -1, 0];
        let mut packed = vec![0u8; 1];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
        }
        let mut output = vec![0.0f32; 4];
        unsafe {
            dequantize_i2s_block(&packed, 1.0, &mut output);
        }
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
        assert!((output[2] - (-1.0)).abs() < 1e-6);
        assert!((output[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_scale_half() {
        let weights: Vec<i8> = vec![1, -1, 1, -1];
        let mut packed = vec![0u8; 1];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
        }
        let mut output = vec![0.0f32; 4];
        unsafe {
            dequantize_i2s_block(&packed, 0.5, &mut output);
        }
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - (-0.5)).abs() < 1e-6);
        assert!((output[2] - 0.5).abs() < 1e-6);
        assert!((output[3] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_scale_negative() {
        let weights: Vec<i8> = vec![1, -1, 0, 1];
        let mut packed = vec![0u8; 1];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
        }
        let mut output = vec![0.0f32; 4];
        unsafe {
            dequantize_i2s_block(&packed, -1.0, &mut output);
        }
        assert!((output[0] - (-1.0)).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
        assert!((output[2] - 0.0).abs() < 1e-6);
        assert!((output[3] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_16_elements() {
        let weights: Vec<i8> = vec![1, -1, 0, 1, -1, -1, 0, 0, 1, 1, 1, -1, 0, -1, 1, 0];
        let mut packed = vec![0u8; 4];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
        }
        let mut output = vec![0.0f32; 16];
        unsafe {
            dequantize_i2s_block(&packed, 2.0, &mut output);
        }
        for (i, &w) in weights.iter().enumerate() {
            let expected = w as f32 * 2.0;
            assert!(
                (output[i] - expected).abs() < 1e-6,
                "mismatch at {i}: expected {expected}, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn test_dequantize_non_aligned() {
        // 5 values → 2 packed bytes, but output only 5 floats
        let weights: Vec<i8> = vec![1, -1, 0, 1, -1];
        let mut packed = vec![0u8; 2];
        unsafe {
            pack_ternary_weights(&weights, &mut packed);
        }
        let mut output = vec![0.0f32; 5];
        unsafe {
            dequantize_i2s_block(&packed, 1.0, &mut output);
        }
        for (i, &w) in weights.iter().enumerate() {
            assert!(
                (output[i] - w as f32).abs() < 1e-6,
                "mismatch at {i}: expected {}, got {}",
                w as f32,
                output[i]
            );
        }
    }

    #[test]
    fn test_dequantize_empty() {
        let packed: Vec<u8> = vec![];
        let mut output: Vec<f32> = vec![];
        unsafe {
            dequantize_i2s_block(&packed, 1.0, &mut output);
        }
    }

    // ── Quantize to ternary ───────────────────────────────────────────

    #[test]
    fn test_quantize_zero_threshold() {
        let values = vec![0.5f32, -0.5, 0.0, 1.0, -1.0];
        let result = unsafe { quantize_to_ternary(&values, 0.0) };
        assert_eq!(result, vec![1, -1, 0, 1, -1]);
    }

    #[test]
    fn test_quantize_large_threshold() {
        let values = vec![0.5f32, -0.5, 0.0, 1.0, -1.0];
        let result = unsafe { quantize_to_ternary(&values, 10.0) };
        assert_eq!(result, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_quantize_mid_threshold() {
        let values = vec![0.6f32, -0.6, 0.4, -0.4, 0.0];
        let result = unsafe { quantize_to_ternary(&values, 0.5) };
        assert_eq!(result, vec![1, -1, 0, 0, 0]);
    }

    #[test]
    fn test_quantize_empty() {
        let result = unsafe { quantize_to_ternary(&[], 0.5) };
        assert!(result.is_empty());
    }

    #[test]
    fn test_quantize_single() {
        assert_eq!(unsafe { quantize_to_ternary(&[1.0], 0.5) }, vec![1]);
        assert_eq!(unsafe { quantize_to_ternary(&[-1.0], 0.5) }, vec![-1]);
        assert_eq!(unsafe { quantize_to_ternary(&[0.0], 0.5) }, vec![0]);
    }

    #[test]
    fn test_quantize_at_threshold_boundary() {
        // At exactly the threshold, should be 0 (strict >)
        let values = vec![0.5f32, -0.5];
        let result = unsafe { quantize_to_ternary(&values, 0.5) };
        assert_eq!(result, vec![0, 0]);
    }

    #[test]
    fn test_quantize_roundtrip() {
        // quantize then pack then unpack should preserve ternary structure
        let values = vec![1.5f32, -2.0, 0.1, 0.0, -0.8, 3.0, -0.01, 0.6];
        let ternary = unsafe { quantize_to_ternary(&values, 0.5) };
        let mut packed = vec![0u8; (ternary.len() + 3) / 4];
        let mut unpacked = vec![0i8; ternary.len()];
        unsafe {
            pack_ternary_weights(&ternary, &mut packed);
            unpack_ternary_weights(&packed, &mut unpacked);
        }
        assert_eq!(unpacked, ternary);
    }

    // ── Popcount ──────────────────────────────────────────────────────

    #[test]
    fn test_popcount_all_zeros() {
        let data = vec![0u8; 32];
        let count = unsafe { neon_popcount_u8x16(&data) };
        assert_eq!(count, 0);
    }

    #[test]
    fn test_popcount_all_ones() {
        let data = vec![0xFFu8; 16];
        let count = unsafe { neon_popcount_u8x16(&data) };
        assert_eq!(count, 128); // 16 bytes × 8 bits
    }

    #[test]
    fn test_popcount_known_pattern() {
        // 0x55 = 0b01010101 → 4 bits set per byte
        let data = vec![0x55u8; 8];
        let count = unsafe { neon_popcount_u8x16(&data) };
        assert_eq!(count, 32); // 8 bytes × 4 bits
    }

    #[test]
    fn test_popcount_single_byte() {
        let data = vec![0b10110011u8]; // 5 bits set
        let count = unsafe { neon_popcount_u8x16(&data) };
        assert_eq!(count, 5);
    }

    #[test]
    fn test_popcount_empty() {
        let count = unsafe { neon_popcount_u8x16(&[]) };
        assert_eq!(count, 0);
    }

    #[test]
    fn test_popcount_17_bytes() {
        // 16 bytes of 0xFF + 1 byte of 0x0F
        let mut data = vec![0xFFu8; 16];
        data.push(0x0F); // 4 bits
        let count = unsafe { neon_popcount_u8x16(&data) };
        assert_eq!(count, 128 + 4);
    }

    #[test]
    fn test_popcount_large() {
        let data = vec![0xAAu8; 1000]; // 0xAA = 4 bits per byte
        let count = unsafe { neon_popcount_u8x16(&data) };
        assert_eq!(count, 4000);
    }

    // ── 4-element (NEON lane width) tests ─────────────────────────────

    #[test]
    fn test_ternary_4_element_neon_lane() {
        for pattern in
            [vec![1i8, 1, 1, 1], vec![-1, -1, -1, -1], vec![0, 0, 0, 0], vec![1, -1, 0, 1]]
        {
            let mut packed = vec![0u8; 1];
            let mut unpacked = vec![0i8; 4];
            unsafe {
                pack_ternary_weights(&pattern, &mut packed);
                unpack_ternary_weights(&packed, &mut unpacked);
            }
            assert_eq!(unpacked, pattern, "4-elem roundtrip failed for {pattern:?}");
        }
    }

    #[test]
    fn test_binary_8_element_neon_lane() {
        let pattern = vec![1i8, -1, 1, -1, 1, -1, 1, -1];
        let mut packed = vec![0u8; 1];
        let mut unpacked = vec![0i8; 8];
        unsafe {
            pack_binary_weights(&pattern, &mut packed);
            unpack_binary_weights(&packed, &mut unpacked, 8);
        }
        assert_eq!(unpacked, pattern);
    }

    // ── Mixed size stress ─────────────────────────────────────────────

    #[test]
    fn test_ternary_roundtrip_sizes_1_to_33() {
        for n in 1..=33 {
            let weights: Vec<i8> = (0..n).map(|i| ((i % 3) as i8) - 1).collect();
            let packed_len = (n + 3) / 4;
            let mut packed = vec![0u8; packed_len];
            let mut unpacked = vec![0i8; n];
            unsafe {
                pack_ternary_weights(&weights, &mut packed);
                unpack_ternary_weights(&packed, &mut unpacked);
            }
            assert_eq!(unpacked, weights, "ternary roundtrip failed for n={n}");
        }
    }

    #[test]
    fn test_binary_roundtrip_sizes_1_to_33() {
        for n in 1..=33 {
            let weights: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
            let packed_len = (n + 7) / 8;
            let mut packed = vec![0u8; packed_len];
            let mut unpacked = vec![0i8; n];
            unsafe {
                pack_binary_weights(&weights, &mut packed);
                unpack_binary_weights(&packed, &mut unpacked, n);
            }
            assert_eq!(unpacked, weights, "binary roundtrip failed for n={n}");
        }
    }
}
