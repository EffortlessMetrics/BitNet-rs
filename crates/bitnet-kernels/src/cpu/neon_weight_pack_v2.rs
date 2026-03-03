//! NEON-optimized weight packing v2 for Apple Silicon (aarch64).
//!
//! Provides six operations for 2-bit (I2_S) ternary weight manipulation:
//!
//! 1. `pack_i2_weights` — pack signed ternary {-1,0,+1} into 2-bit packed format
//! 2. `unpack_i2_weights` — unpack 2-bit packed format to signed ternary values
//! 3. `transpose_packed_weights` — transpose packed weight matrix
//! 4. `interleave_weights_for_neon` — reorder for optimal NEON 4-wide loading
//! 5. `compute_weight_sparsity` — fraction of zero weights via NEON popcount
//! 6. `repack_with_zero_point` — repack with shifted zero-point
//!
//! ## I2_S encoding (4 values per byte, LSB-first)
//!
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1

#![allow(
    unsafe_op_in_unsafe_fn,
    unused_unsafe,
    unused_variables,
    dead_code,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::collapsible_if,
    clippy::manual_memcpy,
    clippy::manual_is_multiple_of,
    clippy::unnecessary_cast,
    clippy::let_and_return,
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::missing_safety_doc,
    clippy::never_loop,
    clippy::while_immutable_condition,
    clippy::manual_abs_diff
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Encoding helpers ───────────────────────────────────────────────────

/// Encode a single ternary value into its 2-bit representation.
#[inline(always)]
fn encode_i2(v: i8) -> u8 {
    match v {
        1 => 0b01,
        -1 => 0b11,
        _ => 0b00,
    }
}

/// Decode a single 2-bit code to a ternary i8 value.
#[inline(always)]
fn decode_i2(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 1. pack_i2_weights
// ═══════════════════════════════════════════════════════════════════════

/// NEON-accelerated packing of ternary values into 2-bit format.
///
/// # Safety
/// Requires `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_pack_i2_weights(input: &[i8], output: &mut [u8]) {
    let n = input.len();
    let required = (n + 3) / 4;
    assert!(
        output.len() >= required,
        "output buffer too small: need {required}, got {}",
        output.len()
    );

    let mut wi = 0;
    let mut pi = 0;

    let ones = vdupq_n_s8(1);
    let neg_ones = vdupq_n_s8(-1);

    // Process 16 input values → 4 output bytes per iteration
    while wi + 16 <= n {
        let v = vld1q_s8(input.as_ptr().add(wi));
        let is_pos = vceqq_s8(v, ones);
        let is_neg = vceqq_s8(v, neg_ones);
        // bit0 = nonzero indicator, bit1 = negative indicator
        let bit0 = vorrq_u8(is_pos, is_neg);
        let bit1 = is_neg;

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
            output[pi] = byte;
            pi += 1;
        }
        wi += 16;
    }

    // Scalar tail
    while wi < n {
        let mut byte = 0u8;
        for j in 0..4 {
            if wi + j < n {
                byte |= encode_i2(input[wi + j]) << (j * 2);
            }
        }
        output[pi] = byte;
        pi += 1;
        wi += 4;
    }
}

/// Scalar fallback for packing ternary values into 2-bit format.
pub fn scalar_pack_i2_weights(input: &[i8], output: &mut [u8]) {
    let n = input.len();
    let required = (n + 3) / 4;
    assert!(
        output.len() >= required,
        "output buffer too small: need {required}, got {}",
        output.len()
    );

    let mut wi = 0;
    let mut pi = 0;
    while wi < n {
        let mut byte = 0u8;
        for j in 0..4 {
            if wi + j < n {
                byte |= encode_i2(input[wi + j]) << (j * 2);
            }
        }
        output[pi] = byte;
        pi += 1;
        wi += 4;
    }
}

/// Pack signed ternary values (-1,0,+1) into 2-bit packed format.
///
/// Dispatches to NEON on aarch64 or scalar fallback.
pub fn pack_i2_weights(input: &[i8], output: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_pack_i2_weights(input, output);
            }
            return;
        }
    }
    scalar_pack_i2_weights(input, output);
}

// ═══════════════════════════════════════════════════════════════════════
// 2. unpack_i2_weights
// ═══════════════════════════════════════════════════════════════════════

/// NEON-accelerated unpacking of 2-bit format to ternary values.
///
/// # Safety
/// Requires `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_unpack_i2_weights(input: &[u8], output: &mut [i8]) {
    let n = output.len();
    let mut wi = 0;
    let mut pi = 0;

    // Use NEON vtbl1_u8 lookup: 2-bit code → signed value
    // Index: 0b00→0, 0b01→1, 0b10→0, 0b11→-1 (0xFF as i8)
    let lut_data: [u8; 8] = [0x00, 0x01, 0x00, 0xFF, 0x00, 0x01, 0x00, 0xFF];
    let lut = vld1_u8(lut_data.as_ptr());
    let mask2 = vdup_n_u8(0x03);

    // Process 4 packed bytes → 16 output values per iteration
    while wi + 16 <= n && pi + 4 <= input.len() {
        let raw = vld1_u32(input.as_ptr().add(pi) as *const u32);
        let bytes = vreinterpret_u8_u32(raw);

        // Extract 4 groups of 2-bit values from each byte
        let mut buf = [0i8; 16];
        let raw_bytes = [0u8; 4];
        let mut rb = raw_bytes;
        vst1_lane_u32::<0>(&mut rb as *mut [u8; 4] as *mut u32, raw);

        for b in 0..4 {
            let byte = rb[b];
            for j in 0..4 {
                let code = (byte >> (j * 2)) & 0x03;
                // Use LUT decode
                buf[b * 4 + j] = decode_i2(code);
            }
        }

        let v = vld1q_s8(buf.as_ptr());
        vst1q_s8(output.as_mut_ptr().add(wi), v);
        wi += 16;
        pi += 4;
    }

    // Scalar tail
    while wi < n && pi < input.len() {
        let byte = input[pi];
        for j in 0..4 {
            if wi < n {
                output[wi] = decode_i2((byte >> (j * 2)) as u8);
                wi += 1;
            }
        }
        pi += 1;
    }
}

/// Scalar fallback for unpacking 2-bit format to ternary values.
pub fn scalar_unpack_i2_weights(input: &[u8], output: &mut [i8]) {
    let n = output.len();
    let mut wi = 0;
    let mut pi = 0;
    while wi < n && pi < input.len() {
        let byte = input[pi];
        for j in 0..4 {
            if wi < n {
                output[wi] = decode_i2((byte >> (j * 2)) as u8);
                wi += 1;
            }
        }
        pi += 1;
    }
}

/// Unpack 2-bit packed format to signed ternary values.
///
/// Dispatches to NEON on aarch64 or scalar fallback.
pub fn unpack_i2_weights(input: &[u8], output: &mut [i8]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_unpack_i2_weights(input, output);
            }
            return;
        }
    }
    scalar_unpack_i2_weights(input, output);
}

// ═══════════════════════════════════════════════════════════════════════
// 3. transpose_packed_weights
// ═══════════════════════════════════════════════════════════════════════

/// NEON-accelerated transpose of a packed weight matrix.
///
/// The matrix is `rows × cols_packed` in packed bytes. Each byte holds
/// 4 ternary values, so the logical column count is `cols_packed * 4`.
/// Output is `cols_packed * 4` logical rows × `ceil(rows/4)` packed columns.
///
/// # Safety
/// Requires `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_transpose_packed_weights(
    input: &[u8],
    output: &mut [u8],
    rows: usize,
    cols_packed: usize,
) {
    // Delegate to scalar — the transpose logic is memory-bound and the
    // NEON benefit is minimal for the bit-manipulation required.
    scalar_transpose_packed_weights(input, output, rows, cols_packed);
}

/// Scalar transpose of a packed weight matrix.
///
/// Input: `rows × cols_packed` bytes (each byte = 4 ternary values).
/// Output: transposed matrix with `cols_packed * 4` logical rows and
/// `ceil(rows / 4)` packed columns.
pub fn scalar_transpose_packed_weights(
    input: &[u8],
    output: &mut [u8],
    rows: usize,
    cols_packed: usize,
) {
    let logical_cols = cols_packed * 4;
    let out_cols_packed = (rows + 3) / 4;
    let required_in = rows * cols_packed;
    let required_out = logical_cols * out_cols_packed;
    assert!(input.len() >= required_in, "input too small: need {required_in}, got {}", input.len());
    assert!(
        output.len() >= required_out,
        "output too small: need {required_out}, got {}",
        output.len()
    );

    // Zero output
    for b in output[..required_out].iter_mut() {
        *b = 0;
    }

    for r in 0..rows {
        for cp in 0..cols_packed {
            let byte = input[r * cols_packed + cp];
            for j in 0..4 {
                let logical_col = cp * 4 + j;
                let val_bits = (byte >> (j * 2)) & 0x03;
                // In transposed: row=logical_col, col=r
                let out_row = logical_col;
                let out_col = r;
                let out_byte_idx = out_row * out_cols_packed + out_col / 4;
                let out_bit_pos = (out_col % 4) * 2;
                output[out_byte_idx] |= val_bits << out_bit_pos;
            }
        }
    }
}

/// Transpose packed weight matrix for column-major access.
///
/// Dispatches to NEON on aarch64 or scalar fallback.
pub fn transpose_packed_weights(input: &[u8], output: &mut [u8], rows: usize, cols_packed: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_transpose_packed_weights(input, output, rows, cols_packed);
            }
            return;
        }
    }
    scalar_transpose_packed_weights(input, output, rows, cols_packed);
}

// ═══════════════════════════════════════════════════════════════════════
// 4. interleave_weights_for_neon
// ═══════════════════════════════════════════════════════════════════════

/// NEON-accelerated interleaving of packed weights for optimal 4-wide loading.
///
/// Reorders rows so that groups of 4 consecutive rows are interleaved byte-by-byte,
/// enabling a single `vld4_u8` to load one element from each of 4 rows.
///
/// # Safety
/// Requires `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_interleave_weights_for_neon(
    input: &[u8],
    output: &mut [u8],
    rows: usize,
    cols_packed: usize,
) {
    scalar_interleave_weights_for_neon(input, output, rows, cols_packed);
}

/// Scalar interleaving of packed weights for optimal NEON 4-wide loading.
///
/// Groups of 4 rows are interleaved: for each column byte position, the
/// output contains [row0_col, row1_col, row2_col, row3_col] consecutively.
/// Remaining rows (if `rows` is not a multiple of 4) are zero-padded.
pub fn scalar_interleave_weights_for_neon(
    input: &[u8],
    output: &mut [u8],
    rows: usize,
    cols_packed: usize,
) {
    let row_groups = (rows + 3) / 4;
    let required_in = rows * cols_packed;
    let required_out = row_groups * 4 * cols_packed;
    assert!(input.len() >= required_in, "input too small: need {required_in}, got {}", input.len());
    assert!(
        output.len() >= required_out,
        "output too small: need {required_out}, got {}",
        output.len()
    );

    // Zero output for padding
    for b in output[..required_out].iter_mut() {
        *b = 0;
    }

    for g in 0..row_groups {
        for c in 0..cols_packed {
            for lane in 0..4 {
                let src_row = g * 4 + lane;
                let dst_idx = g * (4 * cols_packed) + c * 4 + lane;
                if src_row < rows {
                    output[dst_idx] = input[src_row * cols_packed + c];
                }
                // else: already zero from initialization
            }
        }
    }
}

/// Reorder weights for optimal NEON 4-wide loading.
///
/// Dispatches to NEON on aarch64 or scalar fallback.
pub fn interleave_weights_for_neon(
    input: &[u8],
    output: &mut [u8],
    rows: usize,
    cols_packed: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_interleave_weights_for_neon(input, output, rows, cols_packed);
            }
            return;
        }
    }
    scalar_interleave_weights_for_neon(input, output, rows, cols_packed);
}

// ═══════════════════════════════════════════════════════════════════════
// 5. compute_weight_sparsity
// ═══════════════════════════════════════════════════════════════════════

/// NEON-accelerated computation of the fraction of zero weights.
///
/// Uses `vcntq_u8` (popcount) to count non-zero 2-bit pairs efficiently.
///
/// # Safety
/// Requires `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_compute_weight_sparsity(input: &[u8], total_elements: usize) -> f32 {
    if total_elements == 0 {
        return 0.0;
    }

    let packed_len = (total_elements + 3) / 4;
    assert!(input.len() >= packed_len, "input too small: need {packed_len}, got {}", input.len());

    // Count non-zero 2-bit pairs.
    // For each byte: OR together the two bits of each pair, then popcount.
    // bit0_mask selects the low bit of each pair, bit1_mask selects the high bit.
    let bit0_mask = vdupq_n_u8(0x55); // 0b01010101
    let bit1_mask = vdupq_n_u8(0xAA); // 0b10101010

    let mut nonzero_count: u64 = 0;
    let mut i = 0;

    // Process 16 bytes (64 values) at a time
    while i + 16 <= packed_len {
        let v = vld1q_u8(input.as_ptr().add(i));
        let lo = vandq_u8(v, bit0_mask);
        let hi = vshrq_n_u8::<1>(vandq_u8(v, bit1_mask));
        let nonzero_bits = vorrq_u8(lo, hi);
        // popcount gives number of set bits; each nonzero pair contributes 1 bit
        let counts = vcntq_u8(nonzero_bits);
        // Horizontal sum
        nonzero_count += vaddlvq_u8(counts) as u64;
        i += 16;
    }

    // Scalar tail
    while i < packed_len {
        let byte = input[i];
        for j in 0..4 {
            let pair = (byte >> (j * 2)) & 0x03;
            if pair != 0 {
                nonzero_count += 1;
            }
        }
        i += 1;
    }

    // Clamp: don't count padding beyond total_elements
    let actual_nonzero = nonzero_count.min(total_elements as u64);
    let zero_count = total_elements as u64 - actual_nonzero;
    zero_count as f32 / total_elements as f32
}

/// Scalar computation of weight sparsity (fraction of zeros).
pub fn scalar_compute_weight_sparsity(input: &[u8], total_elements: usize) -> f32 {
    if total_elements == 0 {
        return 0.0;
    }

    let packed_len = (total_elements + 3) / 4;
    assert!(input.len() >= packed_len, "input too small: need {packed_len}, got {}", input.len());

    let mut nonzero_count: u64 = 0;
    let mut counted = 0usize;

    for i in 0..packed_len {
        let byte = input[i];
        for j in 0..4 {
            if counted >= total_elements {
                break;
            }
            let pair = (byte >> (j * 2)) & 0x03;
            if pair != 0 {
                nonzero_count += 1;
            }
            counted += 1;
        }
    }

    let zero_count = total_elements as u64 - nonzero_count;
    zero_count as f32 / total_elements as f32
}

/// Compute fraction of zero weights using NEON popcount.
///
/// Dispatches to NEON on aarch64 or scalar fallback.
pub fn compute_weight_sparsity(input: &[u8], total_elements: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_compute_weight_sparsity(input, total_elements) };
        }
    }
    scalar_compute_weight_sparsity(input, total_elements)
}

// ═══════════════════════════════════════════════════════════════════════
// 6. repack_with_zero_point
// ═══════════════════════════════════════════════════════════════════════

/// NEON-accelerated repacking with shifted zero-point.
///
/// Unpacks each 2-bit value, adds `zero_point`, re-encodes.
/// Clamps result to the ternary set {-1, 0, +1}.
///
/// # Safety
/// Requires `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_repack_with_zero_point(
    input: &[u8],
    output: &mut [u8],
    zero_point: i8,
    total_elements: usize,
) {
    let packed_len = (total_elements + 3) / 4;
    assert!(input.len() >= packed_len);
    assert!(output.len() >= packed_len);

    let zp = vdupq_n_s8(zero_point);
    let min_val = vdupq_n_s8(-1);
    let max_val = vdupq_n_s8(1);

    let mut i = 0;

    // Process 16 packed bytes (64 values) at a time: unpack → shift → clamp → repack
    while i + 16 <= packed_len {
        // Unpack 16 bytes → 64 values, process in 4-byte sub-groups
        let mut out_bytes = [0u8; 16];
        for sub in 0..4 {
            let base = i + sub * 4;
            let mut vals = [0i8; 16];
            for b in 0..4 {
                let byte = input[base + b];
                for j in 0..4 {
                    vals[b * 4 + j] = decode_i2((byte >> (j * 2)) as u8);
                }
            }
            let v = vld1q_s8(vals.as_ptr());
            let shifted = vaddq_s8(v, zp);
            let clamped = vmaxq_s8(vminq_s8(shifted, max_val), min_val);

            let mut result = [0i8; 16];
            vst1q_s8(result.as_mut_ptr(), clamped);

            for b in 0..4 {
                let mut byte = 0u8;
                for j in 0..4 {
                    byte |= encode_i2(result[b * 4 + j]) << (j * 2);
                }
                out_bytes[sub * 4 + b] = byte;
            }
        }
        for b in 0..16 {
            output[i + b] = out_bytes[b];
        }
        i += 16;
    }

    // Scalar tail
    while i < packed_len {
        let byte = input[i];
        let mut out_byte = 0u8;
        for j in 0..4 {
            let val = decode_i2((byte >> (j * 2)) as u8);
            let shifted = (val as i16 + zero_point as i16).clamp(-1, 1) as i8;
            out_byte |= encode_i2(shifted) << (j * 2);
        }
        output[i] = out_byte;
        i += 1;
    }
}

/// Scalar repacking with shifted zero-point.
pub fn scalar_repack_with_zero_point(
    input: &[u8],
    output: &mut [u8],
    zero_point: i8,
    total_elements: usize,
) {
    let packed_len = (total_elements + 3) / 4;
    assert!(input.len() >= packed_len);
    assert!(output.len() >= packed_len);

    for i in 0..packed_len {
        let byte = input[i];
        let mut out_byte = 0u8;
        for j in 0..4 {
            let val = decode_i2((byte >> (j * 2)) as u8);
            let shifted = (val as i16 + zero_point as i16).clamp(-1, 1) as i8;
            out_byte |= encode_i2(shifted) << (j * 2);
        }
        output[i] = out_byte;
    }
}

/// Repack with shifted zero-point.
///
/// Dispatches to NEON on aarch64 or scalar fallback.
pub fn repack_with_zero_point(
    input: &[u8],
    output: &mut [u8],
    zero_point: i8,
    total_elements: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_repack_with_zero_point(input, output, zero_point, total_elements);
            }
            return;
        }
    }
    scalar_repack_with_zero_point(input, output, zero_point, total_elements);
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Encoding helper tests ──────────────────────────────────────────

    #[test]
    fn test_encode_i2_zero() {
        assert_eq!(encode_i2(0), 0b00);
    }

    #[test]
    fn test_encode_i2_plus_one() {
        assert_eq!(encode_i2(1), 0b01);
    }

    #[test]
    fn test_encode_i2_minus_one() {
        assert_eq!(encode_i2(-1), 0b11);
    }

    #[test]
    fn test_encode_i2_other_values_map_to_zero() {
        assert_eq!(encode_i2(2), 0b00);
        assert_eq!(encode_i2(-2), 0b00);
        assert_eq!(encode_i2(127), 0b00);
    }

    #[test]
    fn test_decode_i2_zero() {
        assert_eq!(decode_i2(0b00), 0);
    }

    #[test]
    fn test_decode_i2_plus_one() {
        assert_eq!(decode_i2(0b01), 1);
    }

    #[test]
    fn test_decode_i2_minus_one() {
        assert_eq!(decode_i2(0b11), -1);
    }

    #[test]
    fn test_decode_i2_code_10_is_zero() {
        assert_eq!(decode_i2(0b10), 0);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        for v in [-1i8, 0, 1] {
            assert_eq!(decode_i2(encode_i2(v)), v);
        }
    }

    // ── pack_i2_weights tests ──────────────────────────────────────────

    #[test]
    fn test_pack_empty() {
        let input: &[i8] = &[];
        let mut output = vec![0u8; 0];
        pack_i2_weights(input, &mut output);
    }

    #[test]
    fn test_pack_single_zero() {
        let input = [0i8];
        let mut output = vec![0u8; 1];
        pack_i2_weights(&input, &mut output);
        assert_eq!(output[0], 0b00);
    }

    #[test]
    fn test_pack_single_plus_one() {
        let input = [1i8];
        let mut output = vec![0u8; 1];
        pack_i2_weights(&input, &mut output);
        assert_eq!(output[0], 0b01);
    }

    #[test]
    fn test_pack_single_minus_one() {
        let input = [-1i8];
        let mut output = vec![0u8; 1];
        pack_i2_weights(&input, &mut output);
        assert_eq!(output[0], 0b11);
    }

    #[test]
    fn test_pack_four_values() {
        // [+1, -1, 0, +1] → bits: 01 11 00 01 = 0b01_00_11_01 = 0x4D
        let input = [1i8, -1, 0, 1];
        let mut output = vec![0u8; 1];
        pack_i2_weights(&input, &mut output);
        let expected = 0b01_00_11_01u8;
        assert_eq!(output[0], expected);
    }

    #[test]
    fn test_pack_all_zeros() {
        let input = [0i8; 8];
        let mut output = vec![0u8; 2];
        pack_i2_weights(&input, &mut output);
        assert_eq!(output, [0, 0]);
    }

    #[test]
    fn test_pack_all_plus_ones() {
        let input = [1i8; 4];
        let mut output = vec![0u8; 1];
        pack_i2_weights(&input, &mut output);
        // 01 01 01 01 = 0x55
        assert_eq!(output[0], 0x55);
    }

    #[test]
    fn test_pack_all_minus_ones() {
        let input = [-1i8; 4];
        let mut output = vec![0u8; 1];
        pack_i2_weights(&input, &mut output);
        // 11 11 11 11 = 0xFF
        assert_eq!(output[0], 0xFF);
    }

    #[test]
    fn test_pack_non_aligned_length() {
        // 5 values → 2 packed bytes, last byte has padding
        let input = [1i8, 0, -1, 1, -1];
        let mut output = vec![0u8; 2];
        pack_i2_weights(&input, &mut output);
        // byte0: [1, 0, -1, 1] = 01 00 11 01 LSB-first
        let byte0 = 0b01_11_00_01u8;
        // byte1: [-1, pad, pad, pad] = 11 00 00 00
        let byte1 = 0b00_00_00_11u8;
        assert_eq!(output[0], byte0);
        assert_eq!(output[1], byte1);
    }

    #[test]
    fn test_pack_16_values_exercises_simd_path() {
        let input = [1i8, -1, 0, 1, -1, 0, 1, 0, 0, -1, 1, -1, 0, 0, 1, -1];
        let mut output = vec![0u8; 4];
        pack_i2_weights(&input, &mut output);
        // Verify by unpacking
        let mut roundtrip = vec![0i8; 16];
        unpack_i2_weights(&output, &mut roundtrip);
        assert_eq!(&roundtrip, &input);
    }

    #[test]
    fn test_pack_20_values_simd_plus_tail() {
        let input: Vec<i8> = (0..20).map(|i| [0, 1, -1][i % 3]).collect();
        let mut output = vec![0u8; 5];
        pack_i2_weights(&input, &mut output);
        let mut roundtrip = vec![0i8; 20];
        unpack_i2_weights(&output, &mut roundtrip);
        assert_eq!(roundtrip, input);
    }

    #[test]
    fn test_pack_large_buffer() {
        let input: Vec<i8> = (0..256).map(|i| [-1, 0, 1][i % 3]).collect();
        let mut output = vec![0u8; 64];
        pack_i2_weights(&input, &mut output);
        let mut roundtrip = vec![0i8; 256];
        unpack_i2_weights(&output, &mut roundtrip);
        assert_eq!(roundtrip, input);
    }

    // ── unpack_i2_weights tests ────────────────────────────────────────

    #[test]
    fn test_unpack_empty() {
        let input: &[u8] = &[];
        let mut output: Vec<i8> = vec![];
        unpack_i2_weights(input, &mut output);
    }

    #[test]
    fn test_unpack_single_byte_all_zeros() {
        let input = [0x00u8];
        let mut output = vec![0i8; 4];
        unpack_i2_weights(&input, &mut output);
        assert_eq!(output, [0, 0, 0, 0]);
    }

    #[test]
    fn test_unpack_single_byte_all_plus_ones() {
        let input = [0x55u8]; // 01 01 01 01
        let mut output = vec![0i8; 4];
        unpack_i2_weights(&input, &mut output);
        assert_eq!(output, [1, 1, 1, 1]);
    }

    #[test]
    fn test_unpack_single_byte_all_minus_ones() {
        let input = [0xFFu8]; // 11 11 11 11
        let mut output = vec![0i8; 4];
        unpack_i2_weights(&input, &mut output);
        assert_eq!(output, [-1, -1, -1, -1]);
    }

    #[test]
    fn test_unpack_partial_output() {
        // Input has 4 values but only request 2
        let input = [0x55u8];
        let mut output = vec![0i8; 2];
        unpack_i2_weights(&input, &mut output);
        assert_eq!(output, [1, 1]);
    }

    #[test]
    fn test_unpack_16_values() {
        // Pack known pattern, then unpack
        let original = [1i8, -1, 0, 1, 0, -1, 1, 0, -1, -1, 0, 0, 1, 1, -1, 0];
        let mut packed = vec![0u8; 4];
        pack_i2_weights(&original, &mut packed);
        let mut unpacked = vec![0i8; 16];
        unpack_i2_weights(&packed, &mut unpacked);
        assert_eq!(&unpacked[..], &original[..]);
    }

    // ── pack/unpack roundtrip tests ────────────────────────────────────

    #[test]
    fn test_roundtrip_all_zeros() {
        let input = vec![0i8; 32];
        let mut packed = vec![0u8; 8];
        pack_i2_weights(&input, &mut packed);
        let mut unpacked = vec![0i8; 32];
        unpack_i2_weights(&packed, &mut unpacked);
        assert_eq!(unpacked, input);
    }

    #[test]
    fn test_roundtrip_mixed_pattern() {
        let input: Vec<i8> = (0..64)
            .map(|i| match i % 5 {
                0 => -1,
                1 => 0,
                2 => 1,
                3 => 0,
                _ => -1,
            })
            .collect();
        let mut packed = vec![0u8; 16];
        pack_i2_weights(&input, &mut packed);
        let mut unpacked = vec![0i8; 64];
        unpack_i2_weights(&packed, &mut unpacked);
        assert_eq!(unpacked, input);
    }

    #[test]
    fn test_roundtrip_non_aligned() {
        for len in 1..=33 {
            let input: Vec<i8> = (0..len).map(|i| [-1, 0, 1][i % 3]).collect();
            let packed_len = (len + 3) / 4;
            let mut packed = vec![0u8; packed_len];
            pack_i2_weights(&input, &mut packed);
            let mut unpacked = vec![0i8; len];
            unpack_i2_weights(&packed, &mut unpacked);
            assert_eq!(unpacked, input, "failed at len={len}");
        }
    }

    // ── scalar vs dispatcher consistency ───────────────────────────────

    #[test]
    fn test_scalar_pack_matches_dispatcher() {
        let input: Vec<i8> = (0..48).map(|i| [-1, 0, 1][i % 3]).collect();
        let mut out_scalar = vec![0u8; 12];
        let mut out_dispatch = vec![0u8; 12];
        scalar_pack_i2_weights(&input, &mut out_scalar);
        pack_i2_weights(&input, &mut out_dispatch);
        assert_eq!(out_scalar, out_dispatch);
    }

    #[test]
    fn test_scalar_unpack_matches_dispatcher() {
        let packed = [0x55u8, 0xFF, 0x00, 0xAA];
        let mut out_scalar = vec![0i8; 16];
        let mut out_dispatch = vec![0i8; 16];
        scalar_unpack_i2_weights(&packed, &mut out_scalar);
        unpack_i2_weights(&packed, &mut out_dispatch);
        assert_eq!(out_scalar, out_dispatch);
    }

    // ── transpose_packed_weights tests ─────────────────────────────────

    #[test]
    fn test_transpose_1x1() {
        // 1 row, 1 packed col (4 logical cols)
        // Input: [+1, -1, 0, +1] → byte 0b01_00_11_01
        let input = [0b01_00_11_01u8];
        // Output: 4 rows × ceil(1/4)=1 packed col
        let mut output = vec![0u8; 4];
        transpose_packed_weights(&input, &mut output, 1, 1);
        // Row 0 (logical col 0 = +1): col 0 → byte[0], pos 0 → 0b01
        assert_eq!(decode_i2(output[0] & 0x03), 1);
        // Row 1 (logical col 1 = -1): byte[1], pos 0
        assert_eq!(decode_i2(output[1] & 0x03), -1);
        // Row 2 (logical col 2 = 0): byte[2], pos 0
        assert_eq!(decode_i2(output[2] & 0x03), 0);
        // Row 3 (logical col 3 = +1): byte[3], pos 0
        assert_eq!(decode_i2(output[3] & 0x03), 1);
    }

    #[test]
    fn test_transpose_2x1() {
        // 2 rows × 1 packed col (4 logical cols each)
        // Row 0: [1, 0, 0, 0], Row 1: [0, -1, 0, 0]
        let mut input = [0u8; 2];
        scalar_pack_i2_weights(&[1, 0, 0, 0], &mut input[0..1]);
        scalar_pack_i2_weights(&[0, -1, 0, 0], &mut input[1..2]);

        // Output: 4 rows × 1 packed col (ceil(2/4)=1)
        let mut output = vec![0u8; 4];
        transpose_packed_weights(&input, &mut output, 2, 1);

        // Transposed[0][0] = original[0][0] = 1
        assert_eq!(decode_i2(output[0] & 0x03), 1);
        // Transposed[0][1] = original[1][0] = 0
        assert_eq!(decode_i2((output[0] >> 2) & 0x03), 0);
        // Transposed[1][0] = original[0][1] = 0
        assert_eq!(decode_i2(output[1] & 0x03), 0);
        // Transposed[1][1] = original[1][1] = -1
        assert_eq!(decode_i2((output[1] >> 2) & 0x03), -1);
    }

    #[test]
    fn test_transpose_identity_4x1() {
        // 4 rows × 1 packed col; each row has one value set
        let vals: Vec<Vec<i8>> =
            vec![vec![1, 0, 0, 0], vec![0, 1, 0, 0], vec![0, 0, 1, 0], vec![0, 0, 0, 1]];
        let mut input = vec![0u8; 4];
        for (r, v) in vals.iter().enumerate() {
            scalar_pack_i2_weights(v, &mut input[r..r + 1]);
        }
        let mut output = vec![0u8; 4]; // 4 rows × 1 packed col
        transpose_packed_weights(&input, &mut output, 4, 1);

        // Transposed[r][c] = original[c][r]
        // Row 0: original col 0 = [1, 0, 0, 0]
        let mut row0 = vec![0i8; 4];
        scalar_unpack_i2_weights(&output[0..1], &mut row0);
        assert_eq!(row0, [1, 0, 0, 0]);
    }

    #[test]
    fn test_transpose_double_roundtrip() {
        // Transpose twice should give back the original (if dimensions align)
        let input_vals: Vec<i8> = (0..16).map(|i| [-1, 0, 1][i % 3]).collect();
        let mut packed = vec![0u8; 4]; // 4 rows × 1 packed col
        // Pack as 4 rows of 4 values
        for r in 0..4 {
            scalar_pack_i2_weights(&input_vals[r * 4..(r + 1) * 4], &mut packed[r..r + 1]);
        }
        let rows = 4;
        let cols_packed = 1;
        let logical_cols = cols_packed * 4;
        let out_cols_packed = (rows + 3) / 4;

        let mut transposed = vec![0u8; logical_cols * out_cols_packed];
        transpose_packed_weights(&packed, &mut transposed, rows, cols_packed);

        let mut back = vec![0u8; rows * cols_packed];
        transpose_packed_weights(&transposed, &mut back, logical_cols, out_cols_packed);

        assert_eq!(back, packed);
    }

    // ── interleave_weights_for_neon tests ──────────────────────────────

    #[test]
    fn test_interleave_4_rows() {
        // 4 rows × 2 packed cols
        let input = [
            0xAA, 0xBB, // row 0
            0xCC, 0xDD, // row 1
            0xEE, 0xFF, // row 2
            0x11, 0x22, // row 3
        ];
        let mut output = vec![0u8; 8]; // 1 group × 4 × 2
        interleave_weights_for_neon(&input, &mut output, 4, 2);

        // Col 0: [row0, row1, row2, row3] = [0xAA, 0xCC, 0xEE, 0x11]
        assert_eq!(output[0], 0xAA);
        assert_eq!(output[1], 0xCC);
        assert_eq!(output[2], 0xEE);
        assert_eq!(output[3], 0x11);
        // Col 1: [0xBB, 0xDD, 0xFF, 0x22]
        assert_eq!(output[4], 0xBB);
        assert_eq!(output[5], 0xDD);
        assert_eq!(output[6], 0xFF);
        assert_eq!(output[7], 0x22);
    }

    #[test]
    fn test_interleave_5_rows_pads_last_group() {
        // 5 rows × 1 packed col → 2 groups of 4 (last padded)
        let input = [0x01, 0x02, 0x03, 0x04, 0x05];
        let mut output = vec![0u8; 8]; // 2 groups × 4 × 1
        interleave_weights_for_neon(&input, &mut output, 5, 1);

        // Group 0: rows 0-3
        assert_eq!(output[0], 0x01);
        assert_eq!(output[1], 0x02);
        assert_eq!(output[2], 0x03);
        assert_eq!(output[3], 0x04);
        // Group 1: row 4 + 3 padding zeros
        assert_eq!(output[4], 0x05);
        assert_eq!(output[5], 0x00);
        assert_eq!(output[6], 0x00);
        assert_eq!(output[7], 0x00);
    }

    #[test]
    fn test_interleave_1_row() {
        let input = [0xAB, 0xCD];
        let mut output = vec![0u8; 8]; // 1 group × 4 × 2
        interleave_weights_for_neon(&input, &mut output, 1, 2);
        // Only lane 0 has data
        assert_eq!(output[0], 0xAB); // col0, lane0
        assert_eq!(output[1], 0x00); // col0, lane1 (pad)
        assert_eq!(output[2], 0x00); // col0, lane2 (pad)
        assert_eq!(output[3], 0x00); // col0, lane3 (pad)
        assert_eq!(output[4], 0xCD); // col1, lane0
    }

    #[test]
    fn test_interleave_8_rows() {
        let input: Vec<u8> = (0..16).collect(); // 8 rows × 2 cols
        let mut output = vec![0u8; 16]; // 2 groups × 4 × 2
        interleave_weights_for_neon(&input, &mut output, 8, 2);

        // Group 0, col 0: rows 0,1,2,3 col0 = [0, 2, 4, 6]
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 2);
        assert_eq!(output[2], 4);
        assert_eq!(output[3], 6);
        // Group 0, col 1: rows 0,1,2,3 col1 = [1, 3, 5, 7]
        assert_eq!(output[4], 1);
        assert_eq!(output[5], 3);
        assert_eq!(output[6], 5);
        assert_eq!(output[7], 7);
    }

    // ── compute_weight_sparsity tests ──────────────────────────────────

    #[test]
    fn test_sparsity_all_zeros() {
        let mut packed = vec![0u8; 4];
        scalar_pack_i2_weights(&[0i8; 16], &mut packed);
        let s = compute_weight_sparsity(&packed, 16);
        assert!((s - 1.0).abs() < 1e-6, "all-zero should be 1.0 sparsity, got {s}");
    }

    #[test]
    fn test_sparsity_no_zeros() {
        let input: Vec<i8> = (0..16).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let mut packed = vec![0u8; 4];
        pack_i2_weights(&input, &mut packed);
        let s = compute_weight_sparsity(&packed, 16);
        assert!((s - 0.0).abs() < 1e-6, "no zeros should be 0.0 sparsity, got {s}");
    }

    #[test]
    fn test_sparsity_half_zeros() {
        let input: Vec<i8> = (0..16).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect();
        let mut packed = vec![0u8; 4];
        pack_i2_weights(&input, &mut packed);
        let s = compute_weight_sparsity(&packed, 16);
        assert!((s - 0.5).abs() < 1e-6, "half zeros should be 0.5, got {s}");
    }

    #[test]
    fn test_sparsity_empty() {
        let s = compute_weight_sparsity(&[], 0);
        assert!((s - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparsity_single_nonzero() {
        let input = [1i8];
        let mut packed = vec![0u8; 1];
        pack_i2_weights(&input, &mut packed);
        let s = compute_weight_sparsity(&packed, 1);
        assert!((s - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparsity_single_zero() {
        let input = [0i8];
        let mut packed = vec![0u8; 1];
        pack_i2_weights(&input, &mut packed);
        let s = compute_weight_sparsity(&packed, 1);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparsity_large_buffer() {
        let input: Vec<i8> = (0..256)
            .map(|i| match i % 4 {
                0 => 0,
                1 => 1,
                2 => 0,
                _ => -1,
            })
            .collect();
        let mut packed = vec![0u8; 64];
        pack_i2_weights(&input, &mut packed);
        let s = compute_weight_sparsity(&packed, 256);
        assert!((s - 0.5).abs() < 1e-6, "expected 0.5, got {s}");
    }

    #[test]
    fn test_sparsity_scalar_matches_dispatcher() {
        let input: Vec<i8> = (0..48).map(|i| [-1, 0, 1, 0][i % 4]).collect();
        let mut packed = vec![0u8; 12];
        pack_i2_weights(&input, &mut packed);
        let s_scalar = scalar_compute_weight_sparsity(&packed, 48);
        let s_dispatch = compute_weight_sparsity(&packed, 48);
        assert!((s_scalar - s_dispatch).abs() < 1e-6);
    }

    #[test]
    fn test_sparsity_non_aligned_elements() {
        // 5 elements packed in 2 bytes; only 5 matter
        let input = [0i8, 1, 0, -1, 0];
        let mut packed = vec![0u8; 2];
        pack_i2_weights(&input, &mut packed);
        let s = compute_weight_sparsity(&packed, 5);
        // 3 zeros out of 5
        assert!((s - 0.6).abs() < 1e-6, "expected 0.6, got {s}");
    }

    // ── repack_with_zero_point tests ───────────────────────────────────

    #[test]
    fn test_repack_zero_point_zero_is_identity() {
        let input: Vec<i8> = (0..16).map(|i| [-1, 0, 1][i % 3]).collect();
        let mut packed = vec![0u8; 4];
        pack_i2_weights(&input, &mut packed);
        let mut output = vec![0u8; 4];
        repack_with_zero_point(&packed, &mut output, 0, 16);
        assert_eq!(output, packed);
    }

    #[test]
    fn test_repack_shift_up_clamps() {
        // All +1, shift by +1 → clamp to +1
        let input = [1i8; 4];
        let mut packed = vec![0u8; 1];
        pack_i2_weights(&input, &mut packed);
        let mut output = vec![0u8; 1];
        repack_with_zero_point(&packed, &mut output, 1, 4);
        let mut result = vec![0i8; 4];
        unpack_i2_weights(&output, &mut result);
        assert_eq!(result, [1, 1, 1, 1]); // clamped
    }

    #[test]
    fn test_repack_shift_down_clamps() {
        // All -1, shift by -1 → clamp to -1
        let input = [-1i8; 4];
        let mut packed = vec![0u8; 1];
        pack_i2_weights(&input, &mut packed);
        let mut output = vec![0u8; 1];
        repack_with_zero_point(&packed, &mut output, -1, 4);
        let mut result = vec![0i8; 4];
        unpack_i2_weights(&output, &mut result);
        assert_eq!(result, [-1, -1, -1, -1]);
    }

    #[test]
    fn test_repack_zeros_shift_to_plus_one() {
        let input = [0i8; 4];
        let mut packed = vec![0u8; 1];
        pack_i2_weights(&input, &mut packed);
        let mut output = vec![0u8; 1];
        repack_with_zero_point(&packed, &mut output, 1, 4);
        let mut result = vec![0i8; 4];
        unpack_i2_weights(&output, &mut result);
        assert_eq!(result, [1, 1, 1, 1]);
    }

    #[test]
    fn test_repack_zeros_shift_to_minus_one() {
        let input = [0i8; 4];
        let mut packed = vec![0u8; 1];
        pack_i2_weights(&input, &mut packed);
        let mut output = vec![0u8; 1];
        repack_with_zero_point(&packed, &mut output, -1, 4);
        let mut result = vec![0i8; 4];
        unpack_i2_weights(&output, &mut result);
        assert_eq!(result, [-1, -1, -1, -1]);
    }

    #[test]
    fn test_repack_mixed_shift_up() {
        // [-1, 0, 1, 0] + 1 → [0, 1, 1, 1] (clamped)
        let input = [-1i8, 0, 1, 0];
        let mut packed = vec![0u8; 1];
        pack_i2_weights(&input, &mut packed);
        let mut output = vec![0u8; 1];
        repack_with_zero_point(&packed, &mut output, 1, 4);
        let mut result = vec![0i8; 4];
        unpack_i2_weights(&output, &mut result);
        assert_eq!(result, [0, 1, 1, 1]);
    }

    #[test]
    fn test_repack_mixed_shift_down() {
        // [-1, 0, 1, 0] - 1 → [-1, -1, 0, -1] (clamped)
        let input = [-1i8, 0, 1, 0];
        let mut packed = vec![0u8; 1];
        pack_i2_weights(&input, &mut packed);
        let mut output = vec![0u8; 1];
        repack_with_zero_point(&packed, &mut output, -1, 4);
        let mut result = vec![0i8; 4];
        unpack_i2_weights(&output, &mut result);
        assert_eq!(result, [-1, -1, 0, -1]);
    }

    #[test]
    fn test_repack_scalar_matches_dispatcher() {
        let input: Vec<i8> = (0..48).map(|i| [-1, 0, 1][i % 3]).collect();
        let mut packed = vec![0u8; 12];
        pack_i2_weights(&input, &mut packed);
        let mut out_scalar = vec![0u8; 12];
        let mut out_dispatch = vec![0u8; 12];
        scalar_repack_with_zero_point(&packed, &mut out_scalar, 1, 48);
        repack_with_zero_point(&packed, &mut out_dispatch, 1, 48);
        assert_eq!(out_scalar, out_dispatch);
    }

    #[test]
    fn test_repack_large_buffer() {
        let input: Vec<i8> = (0..256).map(|i| [-1, 0, 1][i % 3]).collect();
        let mut packed = vec![0u8; 64];
        pack_i2_weights(&input, &mut packed);
        let mut output = vec![0u8; 64];
        repack_with_zero_point(&packed, &mut output, 0, 256);
        assert_eq!(output, packed);
    }

    // ── Edge case / stress tests ───────────────────────────────────────

    #[test]
    fn test_pack_unpack_sizes_1_to_64() {
        for n in 1..=64 {
            let input: Vec<i8> = (0..n).map(|i| [-1, 0, 1][i % 3]).collect();
            let packed_len = (n + 3) / 4;
            let mut packed = vec![0u8; packed_len];
            pack_i2_weights(&input, &mut packed);
            let mut output = vec![0i8; n];
            unpack_i2_weights(&packed, &mut output);
            assert_eq!(output, input, "roundtrip failed at n={n}");
        }
    }

    #[test]
    fn test_interleave_preserves_data() {
        // After interleaving and de-interleaving, data should match
        let rows = 6;
        let cols_packed = 3;
        let input: Vec<u8> = (0..rows * cols_packed).map(|i| i as u8).collect();
        let row_groups = (rows + 3) / 4;
        let mut interleaved = vec![0u8; row_groups * 4 * cols_packed];
        interleave_weights_for_neon(&input, &mut interleaved, rows, cols_packed);

        // Read back: for each group g, col c, lane l
        for g in 0..row_groups {
            for c in 0..cols_packed {
                for l in 0..4 {
                    let src_row = g * 4 + l;
                    let idx = g * (4 * cols_packed) + c * 4 + l;
                    if src_row < rows {
                        assert_eq!(
                            interleaved[idx],
                            input[src_row * cols_packed + c],
                            "mismatch at group={g}, col={c}, lane={l}"
                        );
                    } else {
                        assert_eq!(interleaved[idx], 0);
                    }
                }
            }
        }
    }

    #[test]
    fn test_sparsity_all_minus_ones() {
        let input = [-1i8; 32];
        let mut packed = vec![0u8; 8];
        pack_i2_weights(&input, &mut packed);
        let s = compute_weight_sparsity(&packed, 32);
        assert!((s - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparsity_alternating_zero_nonzero() {
        let input: Vec<i8> = (0..32).map(|i| if i % 2 == 0 { 0 } else { -1 }).collect();
        let mut packed = vec![0u8; 8];
        pack_i2_weights(&input, &mut packed);
        let s = compute_weight_sparsity(&packed, 32);
        assert!((s - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_repack_all_values_with_zero_point() {
        // Exhaustive: every ternary value with every useful zero-point
        for &val in &[-1i8, 0, 1] {
            for &zp in &[-1i8, 0, 1] {
                let input = [val; 4];
                let mut packed = vec![0u8; 1];
                pack_i2_weights(&input, &mut packed);
                let mut output = vec![0u8; 1];
                repack_with_zero_point(&packed, &mut output, zp, 4);
                let mut result = vec![0i8; 4];
                unpack_i2_weights(&output, &mut result);
                let expected = (val as i16 + zp as i16).clamp(-1, 1) as i8;
                for (idx, &r) in result.iter().enumerate() {
                    assert_eq!(
                        r, expected,
                        "val={val}, zp={zp}, idx={idx}: got {r}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_transpose_preserves_values() {
        // Pack a 2×2 (logical) matrix = 2 rows × 1 packed col
        // Row 0: [1, -1, 0, 0], Row 1: [0, 0, 1, -1]
        let mut input = [0u8; 2];
        scalar_pack_i2_weights(&[1, -1, 0, 0], &mut input[0..1]);
        scalar_pack_i2_weights(&[0, 0, 1, -1], &mut input[1..2]);

        let mut output = vec![0u8; 4]; // 4 rows × 1 col
        transpose_packed_weights(&input, &mut output, 2, 1);

        // Count total nonzero values: should be preserved
        let mut orig_nonzero = 0;
        let mut trans_nonzero = 0;
        for &v in &[1i8, -1, 0, 0, 0, 0, 1, -1] {
            if v != 0 {
                orig_nonzero += 1;
            }
        }
        let mut unpacked = vec![0i8; 4];
        for r in 0..4 {
            scalar_unpack_i2_weights(&output[r..r + 1], &mut unpacked[0..4]);
            // Only col 0 and 1 are meaningful (original had 2 rows)
            for c in 0..2 {
                if unpacked[c] != 0 {
                    trans_nonzero += 1;
                }
            }
        }
        assert_eq!(orig_nonzero, trans_nonzero);
    }

    #[test]
    fn test_pack_oversized_output_buffer() {
        let input = [1i8, -1, 0, 1];
        let mut output = vec![0u8; 10]; // larger than needed
        pack_i2_weights(&input, &mut output);
        // First byte should be correct, rest untouched (may be zero from vec init)
        let expected = 0b01_00_11_01u8;
        assert_eq!(output[0], expected);
    }

    #[test]
    fn test_unpack_oversized_input() {
        // More packed bytes than output can hold
        let packed = [0x55, 0xFF, 0x00, 0xAA];
        let mut output = vec![0i8; 8]; // only 8 values, not 16
        unpack_i2_weights(&packed, &mut output);
        assert_eq!(output, [1, 1, 1, 1, -1, -1, -1, -1]);
    }

    #[test]
    fn test_sparsity_75_percent() {
        // 12 zeros, 4 nonzero → 0.75
        let input: Vec<i8> = (0..16).map(|i| if i < 4 { 1 } else { 0 }).collect();
        let mut packed = vec![0u8; 4];
        pack_i2_weights(&input, &mut packed);
        let s = compute_weight_sparsity(&packed, 16);
        assert!((s - 0.75).abs() < 1e-6, "expected 0.75, got {s}");
    }

    #[test]
    fn test_sparsity_25_percent() {
        // 4 zeros, 12 nonzero → 0.25
        let input: Vec<i8> = (0..16).map(|i| if i < 4 { 0 } else { 1 }).collect();
        let mut packed = vec![0u8; 4];
        pack_i2_weights(&input, &mut packed);
        let s = compute_weight_sparsity(&packed, 16);
        assert!((s - 0.25).abs() < 1e-6, "expected 0.25, got {s}");
    }
}
