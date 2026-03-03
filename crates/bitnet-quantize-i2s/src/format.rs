//! Block format definitions for `I2_S` quantization.
//!
//! Two formats are supported:
//! - [`BitNet32Block`]: 32-element blocks with an inline F16 scale (10 bytes).
//! - [`Qk256Block`]: 256-element blocks, no per-block scale (64 bytes).

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Block format enum
// ---------------------------------------------------------------------------

/// Selects which `I2_S` block layout to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockFormat {
    /// 32-element blocks with inline F16 scale.
    BitNet32F16,
    /// 256-element blocks, no per-block scale (GGML QK256).
    Qk256,
}

impl BlockFormat {
    /// Number of elements per block.
    #[must_use]
    pub const fn block_size(self) -> usize {
        match self {
            Self::BitNet32F16 => 32,
            Self::Qk256 => 256,
        }
    }

    /// Total bytes per block (packed data + optional scale).
    #[must_use]
    pub const fn bytes_per_block(self) -> usize {
        match self {
            Self::BitNet32F16 => 10, // 8 data + 2 f16 scale
            Self::Qk256 => 64,       // 64 data, no scale
        }
    }

    /// Bytes used for packed 2-bit data in one block.
    #[must_use]
    pub const fn data_bytes(self) -> usize {
        match self {
            Self::BitNet32F16 => 8, // 32 * 2 / 8
            Self::Qk256 => 64,      // 256 * 2 / 8
        }
    }

    /// Detect the most likely format from element count and packed byte length.
    ///
    /// Returns `None` when neither format matches.
    #[must_use]
    pub const fn detect(num_elements: usize, packed_bytes: usize) -> Option<Self> {
        // Try QK256 first (higher priority per project convention)
        if num_elements.is_multiple_of(256) {
            let expected = (num_elements / 256) * 64;
            if packed_bytes == expected {
                return Some(Self::Qk256);
            }
        }
        // Then BitNet32-F16
        if num_elements.is_multiple_of(32) {
            let expected = (num_elements / 32) * 10;
            if packed_bytes == expected {
                return Some(Self::BitNet32F16);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// QK256 block (plain-old-data, 64 bytes)
// ---------------------------------------------------------------------------

/// A single 256-element QK256 block (no per-block scale).
///
/// 64 packed bytes encode 256 two-bit values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct Qk256Block {
    /// Packed 2-bit quantised weights (4 values per byte, LSB-first).
    pub data: [u8; 64],
}

// SAFETY: Qk256Block is #[repr(C)] with only a fixed-size u8 array.
unsafe impl Zeroable for Qk256Block {}
unsafe impl Pod for Qk256Block {}

impl Qk256Block {
    /// Number of elements encoded in one block.
    pub const ELEMS: usize = 256;

    /// Create a zeroed block.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self { data: [0u8; 64] }
    }
}

// ---------------------------------------------------------------------------
// BitNet32-F16 block (10 bytes: 8 data + 2 scale)
// ---------------------------------------------------------------------------

/// A single 32-element `BitNet32`-F16 block with an inline F16 scale.
///
/// Layout: `[8 bytes packed data][2 bytes f16 scale]`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BitNet32Block {
    /// Packed 2-bit quantised weights (4 values per byte, LSB-first).
    pub data: [u8; 8],
    /// Per-block scale stored as IEEE 754 binary16 (little-endian).
    pub scale_f16: [u8; 2],
}

impl BitNet32Block {
    /// Number of elements encoded in one block.
    pub const ELEMS: usize = 32;

    /// Create a zeroed block.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self { data: [0u8; 8], scale_f16: [0u8; 2] }
    }

    /// Read the scale as `f32`.
    #[must_use]
    pub fn scale(&self) -> f32 {
        f16_to_f32(u16::from_le_bytes(self.scale_f16))
    }

    /// Write an `f32` scale (converted to f16).
    pub fn set_scale(&mut self, s: f32) {
        self.scale_f16 = f32_to_f16(s).to_le_bytes();
    }
}

impl PartialEq for BitNet32Block {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.scale_f16 == other.scale_f16
    }
}
impl Eq for BitNet32Block {}

// ---------------------------------------------------------------------------
// Minimal f16 <-> f32 helpers (no external `half` dependency)
// ---------------------------------------------------------------------------

/// Convert an IEEE 754 binary16 bit pattern to `f32`.
#[allow(clippy::cast_lossless, clippy::cast_precision_loss, clippy::cast_possible_wrap)]
#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1);
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);

    if exp == 0 {
        // Subnormal or zero
        let val = (mant as f32) * (1.0 / 16_777_216.0); // 2^-24
        if sign == 1 { -val } else { val }
    } else if exp == 31 {
        // Inf / NaN
        if mant == 0 { if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY } } else { f32::NAN }
    } else {
        let f32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        f32::from_bits(f32_bits)
    }
}

/// Convert `f32` to IEEE 754 binary16 bit pattern (round-to-nearest-even).
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]
#[must_use]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 255 {
        // Inf / NaN
        let h_mant = u32::from(mant != 0) * 0x200;
        return ((sign << 15) | (0x1F << 10) | h_mant) as u16;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        // Overflow -> Inf
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    if unbiased < -24 {
        // Underflow -> zero
        return (sign << 15) as u16;
    }
    if unbiased < -14 {
        // Subnormal
        let shift = (-14 - unbiased) as u32;
        let subnorm = (0x400 | (mant >> 13)) >> shift;
        return ((sign << 15) | subnorm) as u16;
    }

    let h_exp = ((unbiased + 15) as u32) << 10;
    let h_mant = mant >> 13;
    // Round-to-nearest-even
    let round_bit = (mant >> 12) & 1;
    let sticky = mant & 0xFFF;
    let round_up = if sticky != 0 { round_bit } else { round_bit & h_mant };
    ((sign << 15) | h_exp | (h_mant + round_up)) as u16
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_format_sizes() {
        assert_eq!(BlockFormat::BitNet32F16.block_size(), 32);
        assert_eq!(BlockFormat::BitNet32F16.bytes_per_block(), 10);
        assert_eq!(BlockFormat::BitNet32F16.data_bytes(), 8);

        assert_eq!(BlockFormat::Qk256.block_size(), 256);
        assert_eq!(BlockFormat::Qk256.bytes_per_block(), 64);
        assert_eq!(BlockFormat::Qk256.data_bytes(), 64);
    }

    #[test]
    fn detect_qk256() {
        assert_eq!(BlockFormat::detect(256, 64), Some(BlockFormat::Qk256));
        assert_eq!(BlockFormat::detect(512, 128), Some(BlockFormat::Qk256));
    }

    #[test]
    fn detect_bitnet32() {
        assert_eq!(BlockFormat::detect(32, 10), Some(BlockFormat::BitNet32F16));
        assert_eq!(BlockFormat::detect(64, 20), Some(BlockFormat::BitNet32F16));
    }

    #[test]
    fn detect_prefers_qk256_when_ambiguous() {
        // 256 is divisible by both 32 and 256, but QK256 is checked first
        assert_eq!(BlockFormat::detect(256, 64), Some(BlockFormat::Qk256));
    }

    #[test]
    fn detect_none_on_mismatch() {
        assert_eq!(BlockFormat::detect(100, 50), None);
        assert_eq!(BlockFormat::detect(256, 65), None);
    }

    #[test]
    fn f16_roundtrip_normal() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, -0.5, 65504.0] {
            let bits = f32_to_f16(v);
            let back = f16_to_f32(bits);
            assert!(
                (back - v).abs() <= v.abs().mul_add(1e-3, 1e-7),
                "f16 roundtrip failed for {v}: got {back}",
            );
        }
    }

    #[test]
    fn f16_special_values() {
        assert!(f16_to_f32(f32_to_f16(f32::INFINITY)).is_infinite());
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        assert!((f16_to_f32(f32_to_f16(0.0))).abs() < f32::EPSILON);
    }

    #[test]
    fn bitnet32_block_scale_roundtrip() {
        let mut block = BitNet32Block::zeroed();
        block.set_scale(1.5);
        let s = block.scale();
        assert!((s - 1.5).abs() < 0.01);
    }

    #[test]
    fn qk256_block_zeroed() {
        let b = Qk256Block::zeroed();
        assert!(b.data.iter().all(|&x| x == 0));
    }
}
