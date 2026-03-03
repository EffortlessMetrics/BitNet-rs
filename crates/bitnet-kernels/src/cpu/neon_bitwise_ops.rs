//! NEON-optimized bitwise operations for ternary weight processing.
//!
//! Provides element-wise AND, OR, XOR, population count, and ternary
//! 2-bit pack/unpack using ARM NEON SIMD intrinsics on aarch64, with
//! pure-Rust scalar fallbacks for other architectures.
//!
//! ## Ternary 2-bit encoding (4 values per byte, LSB-first)
//!
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1
//! - `0b10` → 0 (unused, treated as 0)

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::excessive_precision, clippy::let_and_return)]
#![allow(
    clippy::missing_safety_doc,
    clippy::float_cmp,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    dead_code
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Bitwise AND ────────────────────────────────────────────────────────

/// NEON-accelerated element-wise bitwise AND of two `i8` slices.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_bitwise_and_i8(a: &[i8], b: &[i8], output: &mut [i8]) {
    let n = a.len();
    let chunks = n / 16;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 16;
        unsafe {
            let va = vld1q_s8(a_ptr.add(offset));
            let vb = vld1q_s8(b_ptr.add(offset));
            let vr = vandq_s8(va, vb);
            vst1q_s8(o_ptr.add(offset), vr);
        }
    }

    for i in (chunks * 16)..n {
        output[i] = a[i] & b[i];
    }
}

/// Scalar fallback for element-wise bitwise AND.
#[inline(always)]
fn scalar_bitwise_and_i8(a: &[i8], b: &[i8], output: &mut [i8]) {
    for i in 0..a.len() {
        output[i] = a[i] & b[i];
    }
}

/// Element-wise bitwise AND of two `i8` slices.
///
/// Dispatches to NEON on aarch64, scalar fallback otherwise.
pub fn bitwise_and_i8(a: &[i8], b: &[i8], output: &mut [i8]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    if a.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: feature detection guarantees neon is available.
            unsafe {
                neon_bitwise_and_i8(a, b, output);
            }
            return;
        }
    }

    scalar_bitwise_and_i8(a, b, output);
}

// ── Bitwise OR ─────────────────────────────────────────────────────────

/// NEON-accelerated element-wise bitwise OR of two `i8` slices.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_bitwise_or_i8(a: &[i8], b: &[i8], output: &mut [i8]) {
    let n = a.len();
    let chunks = n / 16;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 16;
        unsafe {
            let va = vld1q_s8(a_ptr.add(offset));
            let vb = vld1q_s8(b_ptr.add(offset));
            let vr = vorrq_s8(va, vb);
            vst1q_s8(o_ptr.add(offset), vr);
        }
    }

    for i in (chunks * 16)..n {
        output[i] = a[i] | b[i];
    }
}

/// Scalar fallback for element-wise bitwise OR.
#[inline(always)]
fn scalar_bitwise_or_i8(a: &[i8], b: &[i8], output: &mut [i8]) {
    for i in 0..a.len() {
        output[i] = a[i] | b[i];
    }
}

/// Element-wise bitwise OR of two `i8` slices.
///
/// Dispatches to NEON on aarch64, scalar fallback otherwise.
pub fn bitwise_or_i8(a: &[i8], b: &[i8], output: &mut [i8]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    if a.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_bitwise_or_i8(a, b, output);
            }
            return;
        }
    }

    scalar_bitwise_or_i8(a, b, output);
}

// ── Bitwise XOR ────────────────────────────────────────────────────────

/// NEON-accelerated element-wise bitwise XOR of two `i8` slices.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_bitwise_xor_i8(a: &[i8], b: &[i8], output: &mut [i8]) {
    let n = a.len();
    let chunks = n / 16;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 16;
        unsafe {
            let va = vld1q_s8(a_ptr.add(offset));
            let vb = vld1q_s8(b_ptr.add(offset));
            let vr = veorq_s8(va, vb);
            vst1q_s8(o_ptr.add(offset), vr);
        }
    }

    for i in (chunks * 16)..n {
        output[i] = a[i] ^ b[i];
    }
}

/// Scalar fallback for element-wise bitwise XOR.
#[inline(always)]
fn scalar_bitwise_xor_i8(a: &[i8], b: &[i8], output: &mut [i8]) {
    for i in 0..a.len() {
        output[i] = a[i] ^ b[i];
    }
}

/// Element-wise bitwise XOR of two `i8` slices.
///
/// Dispatches to NEON on aarch64, scalar fallback otherwise.
pub fn bitwise_xor_i8(a: &[i8], b: &[i8], output: &mut [i8]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    if a.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_bitwise_xor_i8(a, b, output);
            }
            return;
        }
    }

    scalar_bitwise_xor_i8(a, b, output);
}

// ── Population count ───────────────────────────────────────────────────

/// NEON-accelerated per-element population count on `i8` → `u8`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_popcount_i8(input: &[i8], output: &mut [u8]) {
    let n = input.len();
    let chunks = n / 16;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 16;
        unsafe {
            let v = vld1q_s8(in_ptr.add(offset));
            let cnt = vcntq_s8(v);
            // vcntq_s8 returns int8x16_t; reinterpret as u8 for output.
            vst1q_u8(out_ptr.add(offset), vreinterpretq_u8_s8(cnt));
        }
    }

    for i in (chunks * 16)..n {
        output[i] = (input[i] as u8).count_ones() as u8;
    }
}

/// Scalar fallback for per-element population count.
#[inline(always)]
fn scalar_popcount_i8(input: &[i8], output: &mut [u8]) {
    for i in 0..input.len() {
        output[i] = (input[i] as u8).count_ones() as u8;
    }
}

/// Per-element population count: counts set bits in each `i8`,
/// writing the count as `u8`.
///
/// Dispatches to NEON on aarch64, scalar fallback otherwise.
pub fn popcount_i8(input: &[i8], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());
    if input.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_popcount_i8(input, output);
            }
            return;
        }
    }

    scalar_popcount_i8(input, output);
}

// ── Ternary 2-bit packing ─────────────────────────────────────────────

/// Encode a single ternary value into its 2-bit representation.
#[inline(always)]
fn encode_ternary(v: i8) -> u8 {
    match v {
        1 => 0b01,
        -1 => 0b11,
        _ => 0b00,
    }
}

/// Decode a single 2-bit code to a ternary `i8` value.
#[inline(always)]
fn decode_ternary(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

/// NEON-accelerated ternary packing: 4 values per byte, LSB-first.
///
/// Uses NEON comparisons to build bit-planes, then extracts and packs
/// 4 values per output byte. Processes 16 values (4 bytes) per iteration.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_pack_ternary_2bit(input: &[i8], output: &mut [u8]) {
    let n = input.len();
    let mut wi = 0;
    let mut pi = 0;

    let ones = vdupq_n_s8(1);
    let neg_ones = vdupq_n_s8(-1);

    // Process 16 values → 4 packed bytes at a time.
    while wi + 16 <= n {
        unsafe {
            let v = vld1q_s8(input.as_ptr().add(wi));
            // bit0 mask: 0xFF where |val|==1 (nonzero ternary)
            let is_pos = vceqq_s8(v, ones);
            let is_neg = vceqq_s8(v, neg_ones);
            let bit0_mask = vorrq_u8(is_pos, is_neg);
            // bit1 mask: 0xFF where val==-1
            let bit1_mask = is_neg;

            // Store masks to stack, then pack groups of 4 lanes.
            let mut b0 = [0u8; 16];
            let mut b1 = [0u8; 16];
            vst1q_u8(b0.as_mut_ptr(), bit0_mask);
            vst1q_u8(b1.as_mut_ptr(), bit1_mask);

            for g in 0..4usize {
                let base = g * 4;
                let mut byte: u8 = 0;
                for j in 0..4usize {
                    let lo = b0[base + j] & 1;
                    let hi = b1[base + j] & 1;
                    byte |= (lo | (hi << 1)) << (j * 2);
                }
                output[pi] = byte;
                pi += 1;
            }
        }
        wi += 16;
    }

    // Scalar tail.
    while wi < n {
        let remaining = (n - wi).min(4);
        let mut byte: u8 = 0;
        for j in 0..remaining {
            byte |= encode_ternary(input[wi + j]) << (j * 2);
        }
        output[pi] = byte;
        pi += 1;
        wi += remaining;
    }
}

/// Scalar fallback for ternary 2-bit packing.
#[inline(always)]
fn scalar_pack_ternary_2bit(input: &[i8], output: &mut [u8]) {
    let n = input.len();
    let mut wi = 0;
    let mut pi = 0;

    while wi < n {
        let remaining = (n - wi).min(4);
        let mut byte: u8 = 0;
        for j in 0..remaining {
            byte |= encode_ternary(input[wi + j]) << (j * 2);
        }
        output[pi] = byte;
        pi += 1;
        wi += remaining;
    }
}

/// Pack ternary `{-1, 0, 1}` values into 2-bit encoding (4 values per byte).
///
/// Encoding: `-1` → `0b11`, `0` → `0b00`, `1` → `0b01`.
/// `output` must have length `≥ ceil(input.len() / 4)`.
///
/// Dispatches to NEON on aarch64, scalar fallback otherwise.
pub fn pack_ternary_2bit(input: &[i8], output: &mut [u8]) {
    let required = (input.len() + 3) / 4;
    assert!(
        output.len() >= required,
        "output buffer too small: need {required}, got {}",
        output.len()
    );
    if input.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_pack_ternary_2bit(input, output);
            }
            return;
        }
    }

    scalar_pack_ternary_2bit(input, output);
}

// ── Ternary 2-bit unpacking ───────────────────────────────────────────

/// NEON-accelerated ternary unpacking: 4 values per input byte.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_unpack_ternary_2bit(input: &[u8], count: usize, output: &mut [i8]) {
    // For unpacking, the bottleneck is decode logic, not loads. Use scalar
    // with NEON reserved for future LUT-based decode if needed.
    scalar_unpack_ternary_2bit(input, count, output);
}

/// Scalar fallback for ternary 2-bit unpacking.
#[inline(always)]
fn scalar_unpack_ternary_2bit(input: &[u8], count: usize, output: &mut [i8]) {
    let mut oi = 0;
    let mut pi = 0;

    while oi < count {
        let byte = input[pi];
        let remaining = (count - oi).min(4);
        for j in 0..remaining {
            let bits = (byte >> (j * 2)) & 0x03;
            output[oi + j] = decode_ternary(bits);
        }
        oi += remaining;
        pi += 1;
    }
}

/// Unpack 2-bit ternary encoding back to `i8` values.
///
/// Decoding: `0b00` → `0`, `0b01` → `1`, `0b11` → `-1`, `0b10` → `0`.
/// `count` is the number of values to unpack (not bytes).
/// `output` must have length `≥ count`.
///
/// Dispatches to NEON on aarch64, scalar fallback otherwise.
pub fn unpack_ternary_2bit(input: &[u8], count: usize, output: &mut [i8]) {
    let required_bytes = (count + 3) / 4;
    assert!(
        input.len() >= required_bytes,
        "input buffer too small: need {required_bytes} bytes, got {}",
        input.len()
    );
    assert!(output.len() >= count, "output buffer too small: need {count}, got {}", output.len());
    if count == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_unpack_ternary_2bit(input, count, output);
            }
            return;
        }
    }

    scalar_unpack_ternary_2bit(input, count, output);
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AND tests ──────────────────────────────────────────────────────

    #[test]
    fn test_and_basic() {
        let a: Vec<i8> = vec![0x0F, 0x33, 0x55, -1];
        let b: Vec<i8> = vec![-16, 0x33, -86i8, 0x00];
        let mut out = vec![0i8; 4];
        bitwise_and_i8(&a, &b, &mut out);
        for i in 0..4 {
            assert_eq!(out[i], a[i] & b[i], "mismatch at index {i}");
        }
    }

    #[test]
    fn test_and_empty() {
        let a: Vec<i8> = vec![];
        let b: Vec<i8> = vec![];
        let mut out: Vec<i8> = vec![];
        bitwise_and_i8(&a, &b, &mut out);
    }

    #[test]
    fn test_and_single() {
        let a = vec![0x7Fi8];
        let b = vec![-1i8];
        let mut out = vec![0i8; 1];
        bitwise_and_i8(&a, &b, &mut out);
        assert_eq!(out[0], 0x7F);
    }

    #[test]
    fn test_and_len_15() {
        let a: Vec<i8> = (0..15).map(|i| i as i8).collect();
        let b: Vec<i8> = (0..15).map(|i| (14 - i) as i8).collect();
        let mut out = vec![0i8; 15];
        bitwise_and_i8(&a, &b, &mut out);
        for i in 0..15 {
            assert_eq!(out[i], a[i] & b[i]);
        }
    }

    #[test]
    fn test_and_len_16() {
        let a: Vec<i8> = (0..16).map(|i| i as i8).collect();
        let b: Vec<i8> = vec![-1i8; 16];
        let mut out = vec![0i8; 16];
        bitwise_and_i8(&a, &b, &mut out);
        for i in 0..16 {
            assert_eq!(out[i], a[i]);
        }
    }

    #[test]
    fn test_and_len_17() {
        let a: Vec<i8> = (0..17).map(|i| (i * 3) as i8).collect();
        let b: Vec<i8> = (0..17).map(|i| (i * 7) as i8).collect();
        let mut out = vec![0i8; 17];
        bitwise_and_i8(&a, &b, &mut out);
        for i in 0..17 {
            assert_eq!(out[i], a[i] & b[i]);
        }
    }

    #[test]
    fn test_and_len_31() {
        let a: Vec<i8> = (0..31).map(|i| i as i8).collect();
        let b: Vec<i8> = (0..31).map(|_| -1i8).collect();
        let mut out = vec![0i8; 31];
        bitwise_and_i8(&a, &b, &mut out);
        for i in 0..31 {
            assert_eq!(out[i], a[i]);
        }
    }

    #[test]
    fn test_and_len_32() {
        let a: Vec<i8> = (0..32).map(|i| i as i8).collect();
        let b: Vec<i8> = (0..32).map(|i| (i ^ 0xFF) as i8).collect();
        let mut out = vec![0i8; 32];
        bitwise_and_i8(&a, &b, &mut out);
        for i in 0..32 {
            assert_eq!(out[i], a[i] & b[i]);
        }
    }

    #[test]
    fn test_and_len_33() {
        let a: Vec<i8> = (0..33).map(|i| (i * 5) as i8).collect();
        let b: Vec<i8> = (0..33).map(|i| (i * 11) as i8).collect();
        let mut out = vec![0i8; 33];
        bitwise_and_i8(&a, &b, &mut out);
        for i in 0..33 {
            assert_eq!(out[i], a[i] & b[i]);
        }
    }

    #[test]
    fn test_and_all_zeros() {
        let a = vec![0i8; 32];
        let b = vec![-1i8; 32];
        let mut out = vec![-1i8; 32];
        bitwise_and_i8(&a, &b, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_and_all_ones() {
        let a = vec![-1i8; 32];
        let b = vec![-1i8; 32];
        let mut out = vec![0i8; 32];
        bitwise_and_i8(&a, &b, &mut out);
        assert!(out.iter().all(|&v| v == -1));
    }

    // ── OR tests ───────────────────────────────────────────────────────

    #[test]
    fn test_or_basic() {
        let a: Vec<i8> = vec![0x0F, 0x33, 0x00, 0x55];
        let b: Vec<i8> = vec![-16, 0x33, -1, -86i8];
        let mut out = vec![0i8; 4];
        bitwise_or_i8(&a, &b, &mut out);
        for i in 0..4 {
            assert_eq!(out[i], a[i] | b[i], "mismatch at index {i}");
        }
    }

    #[test]
    fn test_or_empty() {
        bitwise_or_i8(&[], &[], &mut []);
    }

    #[test]
    fn test_or_single() {
        let mut out = vec![0i8; 1];
        bitwise_or_i8(&[0x10], &[0x01], &mut out);
        assert_eq!(out[0], 0x11);
    }

    #[test]
    fn test_or_len_16() {
        let a = vec![0i8; 16];
        let b: Vec<i8> = (0..16).map(|i| i as i8).collect();
        let mut out = vec![0i8; 16];
        bitwise_or_i8(&a, &b, &mut out);
        for i in 0..16 {
            assert_eq!(out[i], b[i]);
        }
    }

    #[test]
    fn test_or_len_17() {
        let a: Vec<i8> = (0..17).map(|i| (i * 3) as i8).collect();
        let b: Vec<i8> = (0..17).map(|i| (i * 7) as i8).collect();
        let mut out = vec![0i8; 17];
        bitwise_or_i8(&a, &b, &mut out);
        for i in 0..17 {
            assert_eq!(out[i], a[i] | b[i]);
        }
    }

    #[test]
    fn test_or_len_32() {
        let a: Vec<i8> = (0..32).map(|i| i as i8).collect();
        let b: Vec<i8> = (0..32).map(|i| (!i) as i8).collect();
        let mut out = vec![0i8; 32];
        bitwise_or_i8(&a, &b, &mut out);
        assert!(out.iter().all(|&v| v == -1));
    }

    #[test]
    fn test_or_all_zeros() {
        let a = vec![0i8; 32];
        let b = vec![0i8; 32];
        let mut out = vec![-1i8; 32];
        bitwise_or_i8(&a, &b, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }

    // ── XOR tests ──────────────────────────────────────────────────────

    #[test]
    fn test_xor_basic() {
        let a: Vec<i8> = vec![0x0F, 0x33, -1, 0x00];
        let b: Vec<i8> = vec![0x0F, 0x55, -1, -1];
        let mut out = vec![0i8; 4];
        bitwise_xor_i8(&a, &b, &mut out);
        assert_eq!(out[0], 0x00); // same → 0
        assert_eq!(out[1], 0x33 ^ 0x55);
        assert_eq!(out[2], 0x00); // -1 ^ -1 = 0
        assert_eq!(out[3], -1); // 0 ^ -1 = -1
    }

    #[test]
    fn test_xor_empty() {
        bitwise_xor_i8(&[], &[], &mut []);
    }

    #[test]
    fn test_xor_single() {
        let mut out = vec![0i8; 1];
        bitwise_xor_i8(&[-1], &[0x55], &mut out);
        assert_eq!(out[0], !0x55);
    }

    #[test]
    fn test_xor_len_15() {
        let a: Vec<i8> = (0..15).map(|i| i as i8).collect();
        let b: Vec<i8> = (0..15).map(|i| (i * 2) as i8).collect();
        let mut out = vec![0i8; 15];
        bitwise_xor_i8(&a, &b, &mut out);
        for i in 0..15 {
            assert_eq!(out[i], a[i] ^ b[i]);
        }
    }

    #[test]
    fn test_xor_len_16() {
        let a: Vec<i8> = (0..16).map(|i| i as i8).collect();
        let b = a.clone();
        let mut out = vec![-1i8; 16];
        bitwise_xor_i8(&a, &b, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_xor_len_33() {
        let a: Vec<i8> = (0..33).map(|i| (i * 13) as i8).collect();
        let b: Vec<i8> = (0..33).map(|i| (i * 17) as i8).collect();
        let mut out = vec![0i8; 33];
        bitwise_xor_i8(&a, &b, &mut out);
        for i in 0..33 {
            assert_eq!(out[i], a[i] ^ b[i]);
        }
    }

    #[test]
    fn test_xor_self_is_zero() {
        let a: Vec<i8> = (0..64).map(|i| (i * 7) as i8).collect();
        let mut out = vec![-1i8; 64];
        bitwise_xor_i8(&a, &a, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_xor_all_neg1() {
        let a = vec![-1i8; 32];
        let b = vec![-1i8; 32];
        let mut out = vec![-1i8; 32];
        bitwise_xor_i8(&a, &b, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }

    // ── Popcount tests ─────────────────────────────────────────────────

    #[test]
    fn test_popcount_basic() {
        let input: Vec<i8> = vec![0, 1, 3, 7, 15, 31, 63, 127];
        let mut out = vec![0u8; 8];
        popcount_i8(&input, &mut out);
        assert_eq!(out, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_popcount_empty() {
        popcount_i8(&[], &mut []);
    }

    #[test]
    fn test_popcount_single() {
        let mut out = vec![0u8; 1];
        popcount_i8(&[-1], &mut out);
        assert_eq!(out[0], 8); // -1 = 0xFF → 8 bits
    }

    #[test]
    fn test_popcount_zero() {
        let mut out = vec![99u8; 1];
        popcount_i8(&[0i8], &mut out);
        assert_eq!(out[0], 0);
    }

    #[test]
    fn test_popcount_neg1() {
        let mut out = vec![0u8; 1];
        popcount_i8(&[-1i8], &mut out);
        assert_eq!(out[0], 8);
    }

    #[test]
    fn test_popcount_127() {
        let mut out = vec![0u8; 1];
        popcount_i8(&[127i8], &mut out);
        assert_eq!(out[0], 7); // 0x7F → 7 bits
    }

    #[test]
    fn test_popcount_neg128() {
        let mut out = vec![0u8; 1];
        popcount_i8(&[-128i8], &mut out);
        assert_eq!(out[0], 1); // 0x80 → 1 bit
    }

    #[test]
    fn test_popcount_len_16() {
        let input: Vec<i8> = (0..16).map(|i| i as i8).collect();
        let mut out = vec![0u8; 16];
        popcount_i8(&input, &mut out);
        for i in 0..16 {
            assert_eq!(out[i], (input[i] as u8).count_ones() as u8, "mismatch at {i}");
        }
    }

    #[test]
    fn test_popcount_len_17() {
        let input: Vec<i8> = (0..17).map(|i| (i * 5) as i8).collect();
        let mut out = vec![0u8; 17];
        popcount_i8(&input, &mut out);
        for i in 0..17 {
            assert_eq!(out[i], (input[i] as u8).count_ones() as u8);
        }
    }

    #[test]
    fn test_popcount_len_33() {
        let input: Vec<i8> = (0..33).map(|i| (i * 11) as i8).collect();
        let mut out = vec![0u8; 33];
        popcount_i8(&input, &mut out);
        for i in 0..33 {
            assert_eq!(out[i], (input[i] as u8).count_ones() as u8);
        }
    }

    #[test]
    fn test_popcount_all_zeros() {
        let input = vec![0i8; 32];
        let mut out = vec![99u8; 32];
        popcount_i8(&input, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_popcount_all_neg1() {
        let input = vec![-1i8; 32];
        let mut out = vec![0u8; 32];
        popcount_i8(&input, &mut out);
        assert!(out.iter().all(|&v| v == 8));
    }

    // ── Pack ternary tests ─────────────────────────────────────────────

    #[test]
    fn test_pack_basic_four() {
        // [-1, 0, 1, 0] → bits: 11_00_01_00 = 0b00_01_00_11 = 0x43 (LSB-first)
        let input: Vec<i8> = vec![-1, 0, 1, 0];
        let mut out = vec![0u8; 1];
        pack_ternary_2bit(&input, &mut out);
        let expected: u8 = encode_ternary(-1)
            | (encode_ternary(0) << 2)
            | (encode_ternary(1) << 4)
            | (encode_ternary(0) << 6);
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_pack_empty() {
        pack_ternary_2bit(&[], &mut []);
    }

    #[test]
    fn test_pack_single() {
        let mut out = vec![0u8; 1];
        pack_ternary_2bit(&[1], &mut out);
        assert_eq!(out[0], 0b01);
    }

    #[test]
    fn test_pack_two() {
        let mut out = vec![0u8; 1];
        pack_ternary_2bit(&[-1, 1], &mut out);
        assert_eq!(out[0], 0b01_11); // LSB: -1=11, then 1=01
    }

    #[test]
    fn test_pack_three() {
        let mut out = vec![0u8; 1];
        pack_ternary_2bit(&[0, 1, -1], &mut out);
        let expected = encode_ternary(0) | (encode_ternary(1) << 2) | (encode_ternary(-1) << 4);
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_pack_five() {
        // 5 values → 2 bytes
        let input: Vec<i8> = vec![1, -1, 0, 1, -1];
        let mut out = vec![0u8; 2];
        pack_ternary_2bit(&input, &mut out);
        let byte0 = encode_ternary(1)
            | (encode_ternary(-1) << 2)
            | (encode_ternary(0) << 4)
            | (encode_ternary(1) << 6);
        let byte1 = encode_ternary(-1);
        assert_eq!(out[0], byte0);
        assert_eq!(out[1], byte1);
    }

    #[test]
    fn test_pack_all_zeros() {
        let input = vec![0i8; 16];
        let mut out = vec![0xFFu8; 4];
        pack_ternary_2bit(&input, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_pack_all_ones() {
        let input = vec![1i8; 16];
        let mut out = vec![0u8; 4];
        pack_ternary_2bit(&input, &mut out);
        // Each byte: 01_01_01_01 = 0x55
        assert!(out.iter().all(|&v| v == 0x55));
    }

    #[test]
    fn test_pack_all_neg1() {
        let input = vec![-1i8; 16];
        let mut out = vec![0u8; 4];
        pack_ternary_2bit(&input, &mut out);
        // Each byte: 11_11_11_11 = 0xFF
        assert!(out.iter().all(|&v| v == 0xFF));
    }

    #[test]
    fn test_pack_len_15() {
        let input: Vec<i8> = (0..15).map(|i| [-1, 0, 1][i % 3]).collect();
        let mut out = vec![0u8; 4]; // ceil(15/4)
        pack_ternary_2bit(&input, &mut out);
        // Verify via unpack round-trip.
        let mut recovered = vec![0i8; 15];
        unpack_ternary_2bit(&out, 15, &mut recovered);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_pack_len_17() {
        let input: Vec<i8> = (0..17).map(|i| [1, 0, -1][i % 3]).collect();
        let mut out = vec![0u8; 5]; // ceil(17/4)
        pack_ternary_2bit(&input, &mut out);
        let mut recovered = vec![0i8; 17];
        unpack_ternary_2bit(&out, 17, &mut recovered);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_pack_len_32() {
        let input: Vec<i8> = (0..32).map(|i| [-1, 0, 1, 0][i % 4]).collect();
        let mut out = vec![0u8; 8];
        pack_ternary_2bit(&input, &mut out);
        let mut recovered = vec![0i8; 32];
        unpack_ternary_2bit(&out, 32, &mut recovered);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_pack_len_33() {
        let input: Vec<i8> = (0..33).map(|i| [0, 1, -1][i % 3]).collect();
        let mut out = vec![0u8; 9]; // ceil(33/4)
        pack_ternary_2bit(&input, &mut out);
        let mut recovered = vec![0i8; 33];
        unpack_ternary_2bit(&out, 33, &mut recovered);
        assert_eq!(recovered, input);
    }

    // ── Unpack ternary tests ───────────────────────────────────────────

    #[test]
    fn test_unpack_basic_four() {
        let packed = vec![
            encode_ternary(-1)
                | (encode_ternary(0) << 2)
                | (encode_ternary(1) << 4)
                | (encode_ternary(0) << 6),
        ];
        let mut out = vec![0i8; 4];
        unpack_ternary_2bit(&packed, 4, &mut out);
        assert_eq!(out, vec![-1, 0, 1, 0]);
    }

    #[test]
    fn test_unpack_empty() {
        unpack_ternary_2bit(&[], 0, &mut []);
    }

    #[test]
    fn test_unpack_single() {
        let packed = vec![0b01]; // 1
        let mut out = vec![0i8; 1];
        unpack_ternary_2bit(&packed, 1, &mut out);
        assert_eq!(out[0], 1);
    }

    #[test]
    fn test_unpack_code_10_is_zero() {
        // 0b10 is an unused code; should decode as 0.
        let packed = vec![0b10];
        let mut out = vec![99i8; 1];
        unpack_ternary_2bit(&packed, 1, &mut out);
        assert_eq!(out[0], 0);
    }

    #[test]
    fn test_unpack_all_zeros() {
        let packed = vec![0u8; 4];
        let mut out = vec![99i8; 16];
        unpack_ternary_2bit(&packed, 16, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_unpack_all_ones() {
        let packed = vec![0x55u8; 4]; // 01_01_01_01 per byte
        let mut out = vec![0i8; 16];
        unpack_ternary_2bit(&packed, 16, &mut out);
        assert!(out.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_unpack_all_neg1() {
        let packed = vec![0xFFu8; 4]; // 11_11_11_11 per byte
        let mut out = vec![0i8; 16];
        unpack_ternary_2bit(&packed, 16, &mut out);
        assert!(out.iter().all(|&v| v == -1));
    }

    // ── Round-trip tests ───────────────────────────────────────────────

    #[test]
    fn test_round_trip_basic() {
        let input: Vec<i8> = vec![-1, 0, 1, 0, 1, -1, -1, 1];
        let packed_len = (input.len() + 3) / 4;
        let mut packed = vec![0u8; packed_len];
        pack_ternary_2bit(&input, &mut packed);
        let mut recovered = vec![0i8; input.len()];
        unpack_ternary_2bit(&packed, input.len(), &mut recovered);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_round_trip_len_1() {
        for &val in &[-1i8, 0, 1] {
            let input = vec![val];
            let mut packed = vec![0u8; 1];
            pack_ternary_2bit(&input, &mut packed);
            let mut recovered = vec![0i8; 1];
            unpack_ternary_2bit(&packed, 1, &mut recovered);
            assert_eq!(recovered[0], val);
        }
    }

    #[test]
    fn test_round_trip_len_15() {
        let input: Vec<i8> = (0..15).map(|i| [-1, 0, 1][i % 3]).collect();
        let mut packed = vec![0u8; 4];
        pack_ternary_2bit(&input, &mut packed);
        let mut recovered = vec![0i8; 15];
        unpack_ternary_2bit(&packed, 15, &mut recovered);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_round_trip_len_16() {
        let input: Vec<i8> = (0..16).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let mut packed = vec![0u8; 4];
        pack_ternary_2bit(&input, &mut packed);
        let mut recovered = vec![0i8; 16];
        unpack_ternary_2bit(&packed, 16, &mut recovered);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_round_trip_len_31() {
        let input: Vec<i8> = (0..31).map(|i| [-1, 1, 0][i % 3]).collect();
        let mut packed = vec![0u8; 8];
        pack_ternary_2bit(&input, &mut packed);
        let mut recovered = vec![0i8; 31];
        unpack_ternary_2bit(&packed, 31, &mut recovered);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_round_trip_len_32() {
        let input: Vec<i8> = (0..32).map(|i| [0, -1, 1, -1][i % 4]).collect();
        let mut packed = vec![0u8; 8];
        pack_ternary_2bit(&input, &mut packed);
        let mut recovered = vec![0i8; 32];
        unpack_ternary_2bit(&packed, 32, &mut recovered);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_round_trip_len_33() {
        let input: Vec<i8> = (0..33).map(|i| [1, 0, -1][i % 3]).collect();
        let mut packed = vec![0u8; 9];
        pack_ternary_2bit(&input, &mut packed);
        let mut recovered = vec![0i8; 33];
        unpack_ternary_2bit(&packed, 33, &mut recovered);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_round_trip_all_neg1() {
        let input = vec![-1i8; 64];
        let mut packed = vec![0u8; 16];
        pack_ternary_2bit(&input, &mut packed);
        let mut recovered = vec![0i8; 64];
        unpack_ternary_2bit(&packed, 64, &mut recovered);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_round_trip_all_zeros() {
        let input = vec![0i8; 64];
        let mut packed = vec![0u8; 16];
        pack_ternary_2bit(&input, &mut packed);
        let mut recovered = vec![0i8; 64];
        unpack_ternary_2bit(&packed, 64, &mut recovered);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_round_trip_all_ones() {
        let input = vec![1i8; 64];
        let mut packed = vec![0u8; 16];
        pack_ternary_2bit(&input, &mut packed);
        let mut recovered = vec![0i8; 64];
        unpack_ternary_2bit(&packed, 64, &mut recovered);
        assert_eq!(recovered, input);
    }

    // ── NEON vs scalar consistency ─────────────────────────────────────

    #[test]
    fn test_scalar_and_matches_dispatch() {
        let a: Vec<i8> = (0..64).map(|i| (i * 7) as i8).collect();
        let b: Vec<i8> = (0..64).map(|i| (i * 13) as i8).collect();
        let mut out_dispatch = vec![0i8; 64];
        let mut out_scalar = vec![0i8; 64];
        bitwise_and_i8(&a, &b, &mut out_dispatch);
        scalar_bitwise_and_i8(&a, &b, &mut out_scalar);
        assert_eq!(out_dispatch, out_scalar);
    }

    #[test]
    fn test_scalar_or_matches_dispatch() {
        let a: Vec<i8> = (0..64).map(|i| (i * 7) as i8).collect();
        let b: Vec<i8> = (0..64).map(|i| (i * 13) as i8).collect();
        let mut out_dispatch = vec![0i8; 64];
        let mut out_scalar = vec![0i8; 64];
        bitwise_or_i8(&a, &b, &mut out_dispatch);
        scalar_bitwise_or_i8(&a, &b, &mut out_scalar);
        assert_eq!(out_dispatch, out_scalar);
    }

    #[test]
    fn test_scalar_xor_matches_dispatch() {
        let a: Vec<i8> = (0..64).map(|i| (i * 7) as i8).collect();
        let b: Vec<i8> = (0..64).map(|i| (i * 13) as i8).collect();
        let mut out_dispatch = vec![0i8; 64];
        let mut out_scalar = vec![0i8; 64];
        bitwise_xor_i8(&a, &b, &mut out_dispatch);
        scalar_bitwise_xor_i8(&a, &b, &mut out_scalar);
        assert_eq!(out_dispatch, out_scalar);
    }

    #[test]
    fn test_scalar_popcount_matches_dispatch() {
        let input: Vec<i8> = (0..64).map(|i| (i * 11) as i8).collect();
        let mut out_dispatch = vec![0u8; 64];
        let mut out_scalar = vec![0u8; 64];
        popcount_i8(&input, &mut out_dispatch);
        scalar_popcount_i8(&input, &mut out_scalar);
        assert_eq!(out_dispatch, out_scalar);
    }

    #[test]
    fn test_scalar_pack_matches_dispatch() {
        let input: Vec<i8> = (0..64).map(|i| [-1, 0, 1][i % 3]).collect();
        let mut out_dispatch = vec![0u8; 16];
        let mut out_scalar = vec![0u8; 16];
        pack_ternary_2bit(&input, &mut out_dispatch);
        scalar_pack_ternary_2bit(&input, &mut out_scalar);
        assert_eq!(out_dispatch, out_scalar);
    }

    #[test]
    fn test_scalar_unpack_matches_dispatch() {
        let packed: Vec<u8> = (0..16).map(|i| (i * 17) as u8).collect();
        let mut out_dispatch = vec![0i8; 64];
        let mut out_scalar = vec![0i8; 64];
        unpack_ternary_2bit(&packed, 64, &mut out_dispatch);
        scalar_unpack_ternary_2bit(&packed, 64, &mut out_scalar);
        assert_eq!(out_dispatch, out_scalar);
    }

    // ── Known-value verification ───────────────────────────────────────

    #[test]
    fn test_and_known_values() {
        // 0xFF & 0x0F = 0x0F, 0xAA & 0x55 = 0x00
        let a = vec![-1i8, -86i8]; // 0xFF, 0xAA
        let b = vec![0x0Fi8, 0x55i8];
        let mut out = vec![0i8; 2];
        bitwise_and_i8(&a, &b, &mut out);
        assert_eq!(out[0], 0x0F);
        assert_eq!(out[1], 0x00);
    }

    #[test]
    fn test_or_known_values() {
        let a = vec![0x0Fi8, 0x00i8];
        let b = vec![-16i8, -1i8]; // 0xF0, 0xFF
        let mut out = vec![0i8; 2];
        bitwise_or_i8(&a, &b, &mut out);
        assert_eq!(out[0], -1); // 0xFF
        assert_eq!(out[1], -1);
    }

    #[test]
    fn test_xor_known_values() {
        let a = vec![-1i8, 0x55i8];
        let b = vec![-1i8, -86i8]; // 0xFF, 0xAA
        let mut out = vec![0i8; 2];
        bitwise_xor_i8(&a, &b, &mut out);
        assert_eq!(out[0], 0); // FF ^ FF
        assert_eq!(out[1], -1); // 55 ^ AA = FF
    }

    #[test]
    fn test_popcount_known_values() {
        // 0x00=0, 0x01=1, 0x03=2, 0x0F=4, 0xFF=8, 0x80=1
        let input: Vec<i8> = vec![0, 1, 3, 15, -1, -128];
        let mut out = vec![0u8; 6];
        popcount_i8(&input, &mut out);
        assert_eq!(out, vec![0, 1, 2, 4, 8, 1]);
    }

    #[test]
    fn test_encode_decode_ternary() {
        assert_eq!(encode_ternary(-1), 0b11);
        assert_eq!(encode_ternary(0), 0b00);
        assert_eq!(encode_ternary(1), 0b01);
        assert_eq!(decode_ternary(0b00), 0);
        assert_eq!(decode_ternary(0b01), 1);
        assert_eq!(decode_ternary(0b11), -1);
        assert_eq!(decode_ternary(0b10), 0); // unused code
    }
}
