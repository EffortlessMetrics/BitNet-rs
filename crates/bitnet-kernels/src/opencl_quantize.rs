//! OpenCL I2\_S quantization / dequantization for A770 GPU inference.
//!
//! BitNet uses 1.58-bit quantization (ternary: -1, 0, +1). The I2\_S format
//! packs ternary values into 2-bit fields:
//!
//! | Bits   | Value |
//! |--------|-------|
//! | `0b00` |   0   |
//! | `0b01` |  +1   |
//! | `0b10` |  -1   |
//! | `0b11` | unused|
//!
//! This module provides:
//! - CPU reference implementations for validation
//! - Embedded OpenCL kernel sources for GPU dispatch
//! - Round-trip quantize ↔ dequantize support

use std::fmt;

use bitnet_common::{BitNetError, KernelError};

// ---------------------------------------------------------------------------
// QuantFormat
// ---------------------------------------------------------------------------

/// Quantization formats supported by the I2\_S family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// 2-bit ternary: 00=0, 01=+1, 10=-1, 11=unused.
    I2S,
    /// 32-element blocks with F16 per-block scales.
    BitNet32F16,
    /// 256-element blocks (GGML QK256 layout).
    QK256,
}

impl QuantFormat {
    /// Canonical block size for this format.
    pub fn block_size(self) -> usize {
        match self {
            Self::I2S => 1, // no block structure beyond byte packing
            Self::BitNet32F16 => 32,
            Self::QK256 => 256,
        }
    }

    /// Number of ternary elements stored per byte.
    pub fn elements_per_byte(self) -> usize {
        // All I2S-family formats pack 4 ternary values per byte (2 bits each).
        4
    }
}

impl fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I2S => write!(f, "I2S"),
            Self::BitNet32F16 => write!(f, "BitNet32-F16"),
            Self::QK256 => write!(f, "QK256"),
        }
    }
}

// ---------------------------------------------------------------------------
// DequantConfig
// ---------------------------------------------------------------------------

/// Configuration for a dequantization pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DequantConfig {
    /// Quantization format.
    pub format: QuantFormat,
    /// Number of elements per quantization block.
    pub block_size: usize,
    /// Scale factor applied during dequantization.
    pub scale: f32,
}

impl DequantConfig {
    /// Create a config for I2S format with the given scale.
    pub fn i2s(scale: f32) -> Self {
        Self { format: QuantFormat::I2S, block_size: 1, scale }
    }

    /// Create a config for BitNet32-F16 format.
    pub fn bitnet32_f16(scale: f32) -> Self {
        Self { format: QuantFormat::BitNet32F16, block_size: 32, scale }
    }

    /// Create a config for QK256 format.
    pub fn qk256(scale: f32) -> Self {
        Self { format: QuantFormat::QK256, block_size: 256, scale }
    }

    /// Bytes of packed ternary data per block (not counting scales).
    pub fn packed_bytes_per_block(&self) -> usize {
        self.block_size / self.format.elements_per_byte()
    }

    /// Number of ternary values stored in each byte.
    pub fn elements_per_byte(&self) -> usize {
        self.format.elements_per_byte()
    }
}

// ---------------------------------------------------------------------------
// Dequantizer – CPU reference implementations
// ---------------------------------------------------------------------------

/// CPU-side I2\_S dequantizer / quantizer used for validation and fallback.
#[derive(Debug, Clone, Copy)]
pub struct Dequantizer;

impl Dequantizer {
    /// Dequantize I2S packed bytes into f32 using a uniform `scale`.
    ///
    /// Each byte yields 4 output values (bits 0-1, 2-3, 4-5, 6-7).
    /// Mapping: `0b00 → 0.0`, `0b01 → scale`, `0b10 → -scale`.
    /// The unused pattern `0b11` maps to `0.0`.
    pub fn dequantize_i2s(packed: &[u8], scale: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(packed.len() * 4);
        for &byte in packed {
            for shift in (0..8).step_by(2) {
                let bits = (byte >> shift) & 0b11;
                let value = match bits {
                    0b01 => scale,
                    0b10 => -scale,
                    _ => 0.0, // 0b00 and 0b11
                };
                out.push(value);
            }
        }
        out
    }

    /// Dequantize into a caller-provided buffer. Returns `Err` if `output`
    /// is too small.
    pub fn dequantize_i2s_into(
        packed: &[u8],
        scale: f32,
        output: &mut [f32],
    ) -> Result<(), BitNetError> {
        let required = packed.len() * 4;
        if output.len() < required {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "output buffer too small: need {} elements, got {}",
                    required,
                    output.len()
                ),
            }));
        }
        for (i, &byte) in packed.iter().enumerate() {
            let base = i * 4;
            for j in 0..4 {
                let bits = (byte >> (j * 2)) & 0b11;
                output[base + j] = match bits {
                    0b01 => scale,
                    0b10 => -scale,
                    _ => 0.0,
                };
            }
        }
        Ok(())
    }

    /// Quantize f32 values to I2S packed bytes.
    ///
    /// Each value is rounded to the nearest ternary {-1, 0, +1} and 4 values
    /// are packed per byte. Values outside \[-1, 1\] are clipped.
    /// Input length is padded to a multiple of 4 (extra slots → 0).
    pub fn quantize_to_i2s(values: &[f32], scale: f32) -> Vec<u8> {
        let packed_len = values.len().div_ceil(4);
        let mut packed = vec![0u8; packed_len];

        for (i, &v) in values.iter().enumerate() {
            let normalised = if scale == 0.0 { 0.0 } else { v / scale };
            let clamped = normalised.clamp(-1.0, 1.0);
            let ternary = clamped.round() as i32; // -1, 0, or +1
            let bits: u8 = match ternary {
                1 => 0b01,
                -1 => 0b10,
                _ => 0b00,
            };
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            packed[byte_idx] |= bits << bit_offset;
        }
        packed
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C kernel source for I2S dequantization / quantization.
///
/// # Kernels
///
/// * **`dequantize_i2s`** – each work-item processes one packed byte → 4 f32
///   outputs. Global size = number of packed bytes.
///
/// * **`dequantize_i2s_scaled`** – per-block scale factors stored in a
///   separate `__global const float* scales` buffer. The block index is
///   derived from the global id and `block_size`.
///
/// * **`quantize_to_i2s`** – reverse operation. Each work-item reads 4 f32
///   inputs and writes one packed byte.
///
/// ## A770 optimisation notes
///
/// * **Coalesced access**: consecutive work-items read consecutive bytes /
///   write consecutive floats, which maps well to the A770 memory subsystem.
/// * **Subgroup operations**: the kernels are written so that 16-wide
///   subgroups (matching Xe-HPG EU width) process contiguous memory regions.
/// * **Local memory**: not required – the kernels are pure streaming.
pub const QUANTIZE_I2S_SRC: &str = r#"
// ── I2S Dequantize ─────────────────────────────────────────────────
// Encoding: 0b00 → 0, 0b01 → +1, 0b10 → -1, 0b11 → 0 (unused)
//
// A770 notes:
//   - Each work-item loads one byte (coalesced) and writes 4 floats.
//   - Use global_size = num_packed_bytes for full utilisation.
//   - Subgroup size 16 gives 16 × 4 = 64 contiguous float writes per
//     subgroup, aligning well with 256-byte cache lines on Xe-HPG.

__kernel void dequantize_i2s(
    __global const uchar* packed,   // input: packed I2S bytes
    __global float* output,         // output: dequantised values
    const float scale,              // uniform scale factor
    const uint num_packed_bytes     // number of packed bytes
) {
    uint gid = get_global_id(0);
    if (gid >= num_packed_bytes) return;

    uchar byte_val = packed[gid];
    uint out_base = gid * 4;

    // Unpack 4 ternary values from 2-bit fields
    for (int j = 0; j < 4; j++) {
        uchar bits = (byte_val >> (j * 2)) & 0x3;
        float val = (bits == 1) ? scale : ((bits == 2) ? -scale : 0.0f);
        output[out_base + j] = val;
    }
}

// ── I2S Dequantize with per-block scales ───────────────────────────
// Each block of `block_size` elements shares a single scale factor.
// The `scales` buffer has one entry per block.

__kernel void dequantize_i2s_scaled(
    __global const uchar* packed,   // input: packed I2S bytes
    __global float* output,         // output: dequantised values
    __global const float* scales,   // per-block scale factors
    const uint block_size,          // elements per block
    const uint num_packed_bytes     // total packed bytes
) {
    uint gid = get_global_id(0);
    if (gid >= num_packed_bytes) return;

    uint elem_base = gid * 4;
    uchar byte_val = packed[gid];

    for (int j = 0; j < 4; j++) {
        uint elem_idx = elem_base + j;
        // Determine which block this element belongs to
        uint block_idx = elem_idx / block_size;
        float s = scales[block_idx];

        uchar bits = (byte_val >> (j * 2)) & 0x3;
        float val = (bits == 1) ? s : ((bits == 2) ? -s : 0.0f);
        output[elem_idx] = val;
    }
}

// ── I2S Quantize ───────────────────────────────────────────────────
// Each work-item reads 4 float values, rounds to nearest ternary,
// and packs into one byte.
//
// A770 notes:
//   - Reads 4 consecutive floats (16 bytes) per work-item – good
//     for coalesced 128-bit loads.
//   - Writes one byte per work-item (coalesced across subgroup).

__kernel void quantize_to_i2s(
    __global const float* input,    // input: float values
    __global uchar* packed,         // output: packed I2S bytes
    const float scale,              // quantization scale
    const uint num_packed_bytes     // total output bytes
) {
    uint gid = get_global_id(0);
    if (gid >= num_packed_bytes) return;

    uint in_base = gid * 4;
    uchar byte_val = 0;

    float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

    for (int j = 0; j < 4; j++) {
        float v = input[in_base + j];
        float normalised = v * inv_scale;
        // Clamp to [-1, 1] and round to nearest integer
        normalised = clamp(normalised, -1.0f, 1.0f);
        int ternary = convert_int_rte(normalised); // round-to-even
        uchar bits = (ternary == 1) ? (uchar)1 :
                     (ternary == -1) ? (uchar)2 : (uchar)0;
        byte_val |= (bits << (j * 2));
    }

    packed[gid] = byte_val;
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── QuantFormat ────────────────────────────────────────────────

    #[test]
    fn quant_format_block_sizes() {
        assert_eq!(QuantFormat::I2S.block_size(), 1);
        assert_eq!(QuantFormat::BitNet32F16.block_size(), 32);
        assert_eq!(QuantFormat::QK256.block_size(), 256);
    }

    #[test]
    fn quant_format_elements_per_byte() {
        assert_eq!(QuantFormat::I2S.elements_per_byte(), 4);
        assert_eq!(QuantFormat::BitNet32F16.elements_per_byte(), 4);
        assert_eq!(QuantFormat::QK256.elements_per_byte(), 4);
    }

    #[test]
    fn quant_format_display() {
        assert_eq!(format!("{}", QuantFormat::I2S), "I2S");
        assert_eq!(format!("{}", QuantFormat::BitNet32F16), "BitNet32-F16");
        assert_eq!(format!("{}", QuantFormat::QK256), "QK256");
    }

    #[test]
    fn quant_format_equality() {
        assert_eq!(QuantFormat::I2S, QuantFormat::I2S);
        assert_ne!(QuantFormat::I2S, QuantFormat::QK256);
    }

    // ── DequantConfig ─────────────────────────────────────────────

    #[test]
    fn config_i2s_defaults() {
        let cfg = DequantConfig::i2s(1.0);
        assert_eq!(cfg.format, QuantFormat::I2S);
        assert_eq!(cfg.block_size, 1);
        assert_eq!(cfg.scale, 1.0);
    }

    #[test]
    fn config_bitnet32_defaults() {
        let cfg = DequantConfig::bitnet32_f16(0.5);
        assert_eq!(cfg.format, QuantFormat::BitNet32F16);
        assert_eq!(cfg.block_size, 32);
        assert_eq!(cfg.packed_bytes_per_block(), 8);
    }

    #[test]
    fn config_qk256_defaults() {
        let cfg = DequantConfig::qk256(2.0);
        assert_eq!(cfg.format, QuantFormat::QK256);
        assert_eq!(cfg.block_size, 256);
        assert_eq!(cfg.packed_bytes_per_block(), 64);
    }

    #[test]
    fn config_elements_per_byte() {
        let cfg = DequantConfig::i2s(1.0);
        assert_eq!(cfg.elements_per_byte(), 4);
    }

    #[test]
    fn config_packed_bytes_per_block_i2s() {
        // block_size=1, elements_per_byte=4 → 1/4 = 0 (integer division)
        let cfg = DequantConfig::i2s(1.0);
        assert_eq!(cfg.packed_bytes_per_block(), 0);
    }

    // ── Dequantize: single-byte patterns ──────────────────────────

    #[test]
    fn dequant_all_zeros() {
        // 0b00_00_00_00 → [0, 0, 0, 0]
        let out = Dequantizer::dequantize_i2s(&[0b0000_0000], 1.0);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn dequant_all_plus_ones() {
        // 0b01_01_01_01 = 0x55 → [+1, +1, +1, +1]
        let out = Dequantizer::dequantize_i2s(&[0b0101_0101], 1.0);
        assert_eq!(out, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn dequant_all_minus_ones() {
        // 0b10_10_10_10 = 0xAA → [-1, -1, -1, -1]
        let out = Dequantizer::dequantize_i2s(&[0b1010_1010], 1.0);
        assert_eq!(out, vec![-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn dequant_all_unused_pattern() {
        // 0b11_11_11_11 = 0xFF → [0, 0, 0, 0]  (unused maps to 0)
        let out = Dequantizer::dequantize_i2s(&[0xFF], 1.0);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn dequant_mixed_pattern() {
        // byte: bits[1:0]=01(+1), bits[3:2]=10(-1), bits[5:4]=00(0), bits[7:6]=01(+1)
        // 0b01_00_10_01 = 0x49
        let byte = 0b01_00_10_01u8;
        let out = Dequantizer::dequantize_i2s(&[byte], 1.0);
        assert_eq!(out, vec![1.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn dequant_single_plus_one_in_slot0() {
        let out = Dequantizer::dequantize_i2s(&[0b00_00_00_01], 1.0);
        assert_eq!(out, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn dequant_single_minus_one_in_slot3() {
        // bits[7:6] = 10 → -1 in slot 3
        let out = Dequantizer::dequantize_i2s(&[0b10_00_00_00], 1.0);
        assert_eq!(out, vec![0.0, 0.0, 0.0, -1.0]);
    }

    // ── Dequantize: scale factors ─────────────────────────────────

    #[test]
    fn dequant_scale_factor() {
        let out = Dequantizer::dequantize_i2s(&[0b0101_0101], 2.5);
        assert_eq!(out, vec![2.5, 2.5, 2.5, 2.5]);
    }

    #[test]
    fn dequant_negative_scale() {
        // negative scale: +1 pattern → negative, -1 pattern → positive
        let out = Dequantizer::dequantize_i2s(&[0b0101_0101], -3.0);
        assert_eq!(out, vec![-3.0, -3.0, -3.0, -3.0]);
    }

    #[test]
    fn dequant_zero_scale() {
        let out = Dequantizer::dequantize_i2s(&[0b0101_0101], 0.0);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn dequant_fractional_scale() {
        let out = Dequantizer::dequantize_i2s(&[0b1010_1010], 0.125);
        for &v in &out {
            assert!((v - (-0.125)).abs() < f32::EPSILON);
        }
    }

    // ── Dequantize: multi-byte ────────────────────────────────────

    #[test]
    fn dequant_two_bytes() {
        let packed = [0b0101_0101, 0b1010_1010]; // +1×4, -1×4
        let out = Dequantizer::dequantize_i2s(&packed, 1.0);
        assert_eq!(out.len(), 8);
        assert_eq!(&out[..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&out[4..], &[-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn dequant_large_input() {
        let packed = vec![0b0101_0101; 256]; // 1024 elements
        let out = Dequantizer::dequantize_i2s(&packed, 1.0);
        assert_eq!(out.len(), 1024);
        assert!(out.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn dequant_output_length_property() {
        for n in 0..=64 {
            let packed = vec![0u8; n];
            let out = Dequantizer::dequantize_i2s(&packed, 1.0);
            assert_eq!(out.len(), n * 4, "failed for n={n}");
        }
    }

    // ── Dequantize: empty input ───────────────────────────────────

    #[test]
    fn dequant_empty() {
        let out = Dequantizer::dequantize_i2s(&[], 1.0);
        assert!(out.is_empty());
    }

    // ── Dequantize into buffer ────────────────────────────────────

    #[test]
    fn dequant_into_success() {
        let packed = [0b0101_0101u8];
        let mut buf = [0.0f32; 4];
        Dequantizer::dequantize_i2s_into(&packed, 1.0, &mut buf).unwrap();
        assert_eq!(buf, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn dequant_into_oversized_buffer() {
        let packed = [0b0101_0101u8];
        let mut buf = [0.0f32; 8]; // bigger than needed is OK
        Dequantizer::dequantize_i2s_into(&packed, 1.0, &mut buf).unwrap();
        assert_eq!(&buf[..4], &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn dequant_into_buffer_too_small() {
        let packed = [0b0101_0101u8];
        let mut buf = [0.0f32; 3]; // need 4
        let err = Dequantizer::dequantize_i2s_into(&packed, 1.0, &mut buf);
        assert!(err.is_err());
    }

    #[test]
    fn dequant_into_empty() {
        let mut buf = [0.0f32; 0];
        Dequantizer::dequantize_i2s_into(&[], 1.0, &mut buf).unwrap();
    }

    // ── Quantize ──────────────────────────────────────────────────

    #[test]
    fn quantize_basic_ternary() {
        let values = [1.0f32, -1.0, 0.0, 1.0];
        let packed = Dequantizer::quantize_to_i2s(&values, 1.0);
        assert_eq!(packed.len(), 1);
        // slot0=+1→01, slot1=-1→10, slot2=0→00, slot3=+1→01
        assert_eq!(packed[0], 0b01_00_10_01);
    }

    #[test]
    fn quantize_all_zeros() {
        let values = [0.0f32; 4];
        let packed = Dequantizer::quantize_to_i2s(&values, 1.0);
        assert_eq!(packed[0], 0b0000_0000);
    }

    #[test]
    fn quantize_all_plus_ones() {
        let values = [1.0f32; 4];
        let packed = Dequantizer::quantize_to_i2s(&values, 1.0);
        assert_eq!(packed[0], 0b0101_0101);
    }

    #[test]
    fn quantize_all_minus_ones() {
        let values = [-1.0f32; 4];
        let packed = Dequantizer::quantize_to_i2s(&values, 1.0);
        assert_eq!(packed[0], 0b1010_1010);
    }

    #[test]
    fn quantize_clips_large_values() {
        let values = [5.0f32, -10.0, 100.0, -0.3];
        let packed = Dequantizer::quantize_to_i2s(&values, 1.0);
        // 5→clip→+1→01, -10→clip→-1→10, 100→clip→+1→01, -0.3→round→0→00
        assert_eq!(packed[0], 0b00_01_10_01);
    }

    #[test]
    fn quantize_with_scale() {
        // values = [2.0, -2.0, 0.0, 2.0] with scale=2.0
        // normalised = [1.0, -1.0, 0.0, 1.0]
        let values = [2.0f32, -2.0, 0.0, 2.0];
        let packed = Dequantizer::quantize_to_i2s(&values, 2.0);
        assert_eq!(packed[0], 0b01_00_10_01);
    }

    #[test]
    fn quantize_zero_scale() {
        // zero scale → everything becomes 0
        let values = [1.0f32, -1.0, 0.5, -0.5];
        let packed = Dequantizer::quantize_to_i2s(&values, 0.0);
        assert_eq!(packed[0], 0b0000_0000);
    }

    #[test]
    fn quantize_empty() {
        let packed = Dequantizer::quantize_to_i2s(&[], 1.0);
        assert!(packed.is_empty());
    }

    #[test]
    fn quantize_non_multiple_of_four() {
        // 5 values → 2 packed bytes, last 3 slots of byte 2 are zero
        let values = [1.0f32, -1.0, 0.0, 1.0, -1.0];
        let packed = Dequantizer::quantize_to_i2s(&values, 1.0);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], 0b01_00_10_01);
        assert_eq!(packed[1], 0b00_00_00_10); // -1, then zeros
    }

    // ── Round-trip ────────────────────────────────────────────────

    #[test]
    fn round_trip_all_ternary() {
        let original = [1.0f32, -1.0, 0.0, 1.0, 0.0, -1.0, -1.0, 1.0];
        let packed = Dequantizer::quantize_to_i2s(&original, 1.0);
        let recovered = Dequantizer::dequantize_i2s(&packed, 1.0);
        assert_eq!(&recovered[..original.len()], &original);
    }

    #[test]
    fn round_trip_with_scale() {
        let scale = 0.75;
        let ternary = [scale, -scale, 0.0, scale];
        let packed = Dequantizer::quantize_to_i2s(&ternary, scale);
        let recovered = Dequantizer::dequantize_i2s(&packed, scale);
        for (a, b) in recovered.iter().zip(ternary.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn round_trip_large() {
        let n = 1024;
        let original: Vec<f32> =
            (0..n).map(|i| [1.0, -1.0, 0.0, 1.0][i % 4]).collect();
        let packed = Dequantizer::quantize_to_i2s(&original, 1.0);
        let recovered = Dequantizer::dequantize_i2s(&packed, 1.0);
        assert_eq!(recovered, original);
    }

    #[test]
    fn round_trip_negative_scale() {
        let scale = -2.0;
        let values = [scale, -scale, 0.0, scale]; // [-2, 2, 0, -2]
        let packed = Dequantizer::quantize_to_i2s(&values, scale);
        let recovered = Dequantizer::dequantize_i2s(&packed, scale);
        for (a, b) in recovered.iter().zip(values.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    // ── Kernel source validation ──────────────────────────────────

    #[test]
    fn kernel_source_contains_dequantize() {
        assert!(QUANTIZE_I2S_SRC.contains("__kernel void dequantize_i2s("));
    }

    #[test]
    fn kernel_source_contains_dequantize_scaled() {
        assert!(QUANTIZE_I2S_SRC.contains("__kernel void dequantize_i2s_scaled("));
    }

    #[test]
    fn kernel_source_contains_quantize() {
        assert!(QUANTIZE_I2S_SRC.contains("__kernel void quantize_to_i2s("));
    }

    #[test]
    fn kernel_source_not_empty() {
        assert!(!QUANTIZE_I2S_SRC.is_empty());
        assert!(QUANTIZE_I2S_SRC.len() > 100);
    }

    #[test]
    fn kernel_source_mentions_a770() {
        assert!(QUANTIZE_I2S_SRC.contains("A770"));
    }

    #[test]
    fn kernel_source_mentions_coalesced() {
        assert!(QUANTIZE_I2S_SRC.contains("coalesced"));
    }

    #[test]
    fn kernel_source_mentions_subgroup() {
        assert!(QUANTIZE_I2S_SRC.contains("subgroup") || QUANTIZE_I2S_SRC.contains("Subgroup"));
    }

    // ── Edge cases ────────────────────────────────────────────────

    #[test]
    fn dequant_interleaved_pattern() {
        // +1, -1, +1, -1 → bits: 01 10 01 10 = 0b10_01_10_01 = 0x99
        let byte = 0b10_01_10_01u8;
        let out = Dequantizer::dequantize_i2s(&[byte], 1.0);
        assert_eq!(out, vec![1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn quantize_rounding_half() {
        // 0.5 rounds to +1, -0.5 rounds to -1 (round-to-nearest)
        let values = [0.5f32, -0.5, 0.4, -0.4];
        let packed = Dequantizer::quantize_to_i2s(&values, 1.0);
        let recovered = Dequantizer::dequantize_i2s(&packed, 1.0);
        // 0.5→round→1, -0.5→round→-1 (Rust rounds 0.5 to 1.0 via .round() banker's rounding: actually round() rounds away from zero on tie for f32)
        // 0.4→round→0, -0.4→round→0
        assert_eq!(recovered[0], 1.0); // 0.5 rounds to 1
        assert_eq!(recovered[2], 0.0); // 0.4 rounds to 0
        assert_eq!(recovered[3], 0.0); // -0.4 rounds to 0
    }

    #[test]
    fn quantize_exact_boundaries() {
        let values = [1.0f32, -1.0, 0.0, 0.0];
        let packed = Dequantizer::quantize_to_i2s(&values, 1.0);
        let recovered = Dequantizer::dequantize_i2s(&packed, 1.0);
        assert_eq!(recovered, vec![1.0, -1.0, 0.0, 0.0]);
    }
}
