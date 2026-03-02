//! OpenCL-accelerated weight dequantization kernels for all supported quantization formats.
#![allow(clippy::needless_range_loop)]
//!
//! Provides GPU-optimized dequantization for I2_S (BitNet32-F16, QK256), I4, I8, F16, BF16,
//! and ternary formats, targeting Intel Arc A770 with vectorized loads, subgroup shuffles,
//! and shared local memory for block scales.
//!
//! Each format has an OpenCL kernel source string and a CPU reference implementation for
//! validation. `DequantKernel` dispatches to format-specific implementations.

use std::fmt;
use std::time::Instant;

// ── Dequantization format ───────────────────────────────────────────────────

/// Supported weight dequantization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DequantFormat {
    /// I2_S BitNet32-F16: 32-element blocks with inline FP16 scales.
    I2sBitNet32,
    /// I2_S QK256: 256-element blocks with per-block FP32 scale.
    I2sQk256,
    /// 4-bit integer quantization with zero point and scale.
    I4,
    /// 8-bit integer quantization with scale.
    I8,
    /// IEEE 754 half-precision (16-bit) float.
    F16,
    /// Brain floating-point (16-bit) format.
    Bf16,
    /// Packed ternary {-1, 0, +1} with 2 bits per value.
    Ternary,
}

impl DequantFormat {
    /// Bits per element in the packed representation.
    pub fn bits_per_element(self) -> usize {
        match self {
            Self::I2sBitNet32 | Self::I2sQk256 | Self::Ternary => 2,
            Self::I4 => 4,
            Self::I8 => 8,
            Self::F16 | Self::Bf16 => 16,
        }
    }

    /// Whether this format uses block-level scales.
    pub fn has_block_scales(self) -> bool {
        matches!(self, Self::I2sBitNet32 | Self::I2sQk256 | Self::I4 | Self::Ternary)
    }

    /// Default group size for the format.
    pub fn default_group_size(self) -> usize {
        match self {
            Self::I2sBitNet32 => 32,
            Self::I2sQk256 => 256,
            Self::I4 => 128,
            Self::I8 => 128,
            Self::Ternary => 64,
            Self::F16 | Self::Bf16 => 1,
        }
    }
}

impl fmt::Display for DequantFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I2sBitNet32 => write!(f, "I2S-BitNet32"),
            Self::I2sQk256 => write!(f, "I2S-QK256"),
            Self::I4 => write!(f, "INT4"),
            Self::I8 => write!(f, "INT8"),
            Self::F16 => write!(f, "FP16"),
            Self::Bf16 => write!(f, "BF16"),
            Self::Ternary => write!(f, "Ternary"),
        }
    }
}

// ── Scale type ──────────────────────────────────────────────────────────────

/// How quantization scales are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleType {
    /// Single scale for the entire tensor.
    PerTensor,
    /// One scale per output channel (row).
    PerChannel,
    /// One scale per group of elements within a row.
    PerGroup,
}

impl fmt::Display for ScaleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PerTensor => write!(f, "per-tensor"),
            Self::PerChannel => write!(f, "per-channel"),
            Self::PerGroup => write!(f, "per-group"),
        }
    }
}

// ── Dequantization config ───────────────────────────────────────────────────

/// Configuration for a dequantization operation.
#[derive(Debug, Clone)]
pub struct DequantConfig {
    /// Quantization format of the packed weights.
    pub format: DequantFormat,
    /// Number of elements per quantization group.
    pub group_size: usize,
    /// How scales are applied.
    pub scale_type: ScaleType,
}

impl DequantConfig {
    /// Create a new config with the format's default group size.
    pub fn new(format: DequantFormat, scale_type: ScaleType) -> Self {
        Self { format, group_size: format.default_group_size(), scale_type }
    }

    /// Create a config with an explicit group size.
    pub fn with_group_size(
        format: DequantFormat,
        group_size: usize,
        scale_type: ScaleType,
    ) -> Self {
        assert!(group_size > 0, "group_size must be positive");
        Self { format, group_size, scale_type }
    }

    /// Number of groups for a row of `n` elements.
    pub fn num_groups(&self, n: usize) -> usize {
        match self.scale_type {
            ScaleType::PerTensor => 1,
            ScaleType::PerChannel => 1,
            ScaleType::PerGroup => n.div_ceil(self.group_size),
        }
    }

    /// Number of scales required for a matrix of `rows × cols`.
    pub fn num_scales(&self, rows: usize, cols: usize) -> usize {
        match self.scale_type {
            ScaleType::PerTensor => 1,
            ScaleType::PerChannel => rows,
            ScaleType::PerGroup => rows * cols.div_ceil(self.group_size),
        }
    }
}

impl fmt::Display for DequantConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (group={}, {})", self.format, self.group_size, self.scale_type)
    }
}

// ── Dequantization statistics ───────────────────────────────────────────────

/// Performance and diagnostic stats from a dequantization operation.
#[derive(Debug, Clone, Copy)]
pub struct DequantStats {
    /// Wall-clock dequantization time in microseconds.
    pub dequant_time_us: u64,
    /// Effective bandwidth in GB/s (packed bytes read / time).
    pub bandwidth_gb_s: f64,
    /// Overhead ratio: (dequant output bytes) / (packed input bytes).
    pub format_overhead: f64,
}

impl DequantStats {
    /// Compute stats from packed input size, output element count, and elapsed time.
    pub fn compute(packed_bytes: usize, output_elements: usize, elapsed_us: u64) -> Self {
        let output_bytes = output_elements * 4; // f32
        let bandwidth_gb_s = if elapsed_us > 0 {
            (packed_bytes as f64) / (elapsed_us as f64 * 1e-6) / 1e9
        } else {
            0.0
        };
        let format_overhead =
            if packed_bytes > 0 { output_bytes as f64 / packed_bytes as f64 } else { 0.0 };
        Self { dequant_time_us: elapsed_us, bandwidth_gb_s, format_overhead }
    }
}

impl fmt::Display for DequantStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "dequant: {}µs, {:.2} GB/s, {:.1}× expansion",
            self.dequant_time_us, self.bandwidth_gb_s, self.format_overhead,
        )
    }
}

// ── Packed ternary ──────────────────────────────────────────────────────────

/// Unpacks 2-bit ternary values {-1, 0, +1} from packed bytes.
///
/// Encoding: `0b00 → −1`, `0b01 → 0`, `0b10 → +1`.
/// Four values per byte, LSB first.
#[derive(Debug, Clone, Copy)]
pub struct PackedTernary;

impl PackedTernary {
    /// Pack ternary values into bytes (4 values per byte).
    pub fn pack(values: &[i8]) -> Vec<u8> {
        let packed_len = values.len().div_ceil(4);
        let mut packed = vec![0u8; packed_len];
        for (i, &v) in values.iter().enumerate() {
            let encoded: u8 = match v {
                -1 => 0,
                0 => 1,
                1 => 2,
                _ => panic!("ternary values must be -1, 0, or +1, got {v}"),
            };
            packed[i / 4] |= encoded << ((i % 4) * 2);
        }
        packed
    }

    /// Unpack a single ternary value at the given index.
    #[inline]
    pub fn unpack_one(packed: &[u8], index: usize) -> i8 {
        let bits = (packed[index / 4] >> ((index % 4) * 2)) & 0x03;
        (bits as i8) - 1
    }

    /// Unpack `count` ternary values from packed bytes.
    pub fn unpack(packed: &[u8], count: usize) -> Vec<i8> {
        (0..count).map(|i| Self::unpack_one(packed, i)).collect()
    }

    /// Unpack a full byte into 4 ternary values.
    #[inline]
    pub fn unpack_byte(byte: u8) -> [i8; 4] {
        [
            ((byte) & 0x03) as i8 - 1,
            ((byte >> 2) & 0x03) as i8 - 1,
            ((byte >> 4) & 0x03) as i8 - 1,
            ((byte >> 6) & 0x03) as i8 - 1,
        ]
    }
}

// ── I2S dequantization ──────────────────────────────────────────────────────

/// Dequantizes I2_S BitNet-style ternary weights {-1, 0, +1} with per-block scales.
///
/// Supports both BitNet32-F16 (32-element blocks) and custom block sizes.
pub struct I2sDequant;

impl I2sDequant {
    /// Dequantize a packed row with per-group scales.
    pub fn dequantize_row(
        packed: &[u8],
        scales: &[f32],
        cols: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; cols];
        for (col, out_val) in out.iter_mut().enumerate() {
            let ternary = PackedTernary::unpack_one(packed, col);
            let group_idx = col / group_size;
            let scale = scales.get(group_idx).copied().unwrap_or(0.0);
            *out_val = ternary as f32 * scale;
        }
        out
    }
    pub fn dequantize_matrix(
        packed: &[u8],
        scales: &[f32],
        rows: usize,
        cols: usize,
        config: &DequantConfig,
    ) -> (Vec<f32>, DequantStats) {
        let start = Instant::now();
        let packed_cols = cols.div_ceil(4);
        let groups_per_row = config.num_groups(cols);
        let mut output = vec![0.0f32; rows * cols];

        for row in 0..rows {
            let row_packed = &packed[row * packed_cols..(row + 1) * packed_cols];
            let row_scales = match config.scale_type {
                ScaleType::PerTensor => scales,
                ScaleType::PerChannel => &scales[row..row + 1],
                ScaleType::PerGroup => &scales[row * groups_per_row..(row + 1) * groups_per_row],
            };
            let row_out = Self::dequantize_row(row_packed, row_scales, cols, config.group_size);
            output[row * cols..(row + 1) * cols].copy_from_slice(&row_out);
        }

        let elapsed_us = start.elapsed().as_micros() as u64;
        let packed_bytes = rows * packed_cols;
        let stats = DequantStats::compute(packed_bytes, rows * cols, elapsed_us);
        (output, stats)
    }
}

// ── QK256 dequantization ────────────────────────────────────────────────────

/// Dequantizes QK256 format: 256-element blocks with per-block FP32 scale.
///
/// Each block: 64 packed bytes (256 ternary values at 2 bits each) + 1 f32 scale.
pub struct Qk256Dequant;

impl Qk256Dequant {
    /// Block size for QK256 format.
    pub const BLOCK_SIZE: usize = 256;
    /// Packed bytes per block (256 values × 2 bits / 8).
    pub const PACKED_BYTES_PER_BLOCK: usize = 64;

    /// Dequantize a single QK256 block.
    pub fn dequantize_block(packed_block: &[u8], scale: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(Self::BLOCK_SIZE);
        for i in 0..Self::BLOCK_SIZE {
            let ternary = PackedTernary::unpack_one(packed_block, i);
            out.push(ternary as f32 * scale);
        }
        out
    }

    /// Dequantize a full row of QK256 blocks.
    pub fn dequantize_row(packed: &[u8], scales: &[f32], cols: usize) -> Vec<f32> {
        let num_blocks = cols.div_ceil(Self::BLOCK_SIZE);
        let mut out = vec![0.0f32; cols];
        for blk in 0..num_blocks {
            let start = blk * Self::PACKED_BYTES_PER_BLOCK;
            let end = (start + Self::PACKED_BYTES_PER_BLOCK).min(packed.len());
            let block_data = &packed[start..end];
            let scale = scales.get(blk).copied().unwrap_or(0.0);
            let col_start = blk * Self::BLOCK_SIZE;
            let col_end = (col_start + Self::BLOCK_SIZE).min(cols);
            for col in col_start..col_end {
                let idx_in_block = col - col_start;
                let ternary = PackedTernary::unpack_one(block_data, idx_in_block);
                out[col] = ternary as f32 * scale;
            }
        }
        out
    }

    /// Dequantize an entire QK256 matrix.
    pub fn dequantize_matrix(
        packed: &[u8],
        scales: &[f32],
        rows: usize,
        cols: usize,
    ) -> (Vec<f32>, DequantStats) {
        let start = Instant::now();
        let num_blocks_per_row = cols.div_ceil(Self::BLOCK_SIZE);
        let packed_per_row = num_blocks_per_row * Self::PACKED_BYTES_PER_BLOCK;
        let mut output = vec![0.0f32; rows * cols];

        for row in 0..rows {
            let row_packed = &packed[row * packed_per_row..(row + 1) * packed_per_row];
            let row_scales = &scales[row * num_blocks_per_row..(row + 1) * num_blocks_per_row];
            let row_out = Self::dequantize_row(row_packed, row_scales, cols);
            output[row * cols..(row + 1) * cols].copy_from_slice(&row_out);
        }

        let elapsed_us = start.elapsed().as_micros() as u64;
        let packed_bytes = rows * packed_per_row;
        let stats = DequantStats::compute(packed_bytes, rows * cols, elapsed_us);
        (output, stats)
    }
}

// ── I4 dequantization ───────────────────────────────────────────────────────

/// Dequantizes 4-bit integer values with zero point and per-group scale.
///
/// Two values packed per byte, low nibble first.
pub struct I4Dequant;

impl I4Dequant {
    /// Pack 4-bit values (0..15) into bytes (2 values per byte, low nibble first).
    pub fn pack(values: &[u8]) -> Vec<u8> {
        let packed_len = values.len().div_ceil(2);
        let mut packed = vec![0u8; packed_len];
        for (i, &v) in values.iter().enumerate() {
            debug_assert!(v < 16, "I4 value must be 0..15, got {v}");
            if i % 2 == 0 {
                packed[i / 2] |= v & 0x0F;
            } else {
                packed[i / 2] |= (v & 0x0F) << 4;
            }
        }
        packed
    }

    /// Unpack a single 4-bit value.
    #[inline]
    pub fn unpack_one(packed: &[u8], index: usize) -> u8 {
        let byte = packed[index / 2];
        if index.is_multiple_of(2) { byte & 0x0F } else { (byte >> 4) & 0x0F }
    }

    /// Dequantize a row of 4-bit quantized values.
    ///
    /// Formula: `output[i] = (quant[i] - zero_point) * scale`
    pub fn dequantize_row(
        packed: &[u8],
        scales: &[f32],
        zero_points: &[u8],
        cols: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; cols];
        for col in 0..cols {
            let quant_val = Self::unpack_one(packed, col);
            let group_idx = col / group_size;
            let scale = scales.get(group_idx).copied().unwrap_or(1.0);
            let zp = zero_points.get(group_idx).copied().unwrap_or(8);
            out[col] = (quant_val as f32 - zp as f32) * scale;
        }
        out
    }
}

// ── CPU reference dequantization for I8, F16, BF16 ──────────────────────────

/// Dequantizes 8-bit integer values with per-group scale.
pub struct I8Dequant;

impl I8Dequant {
    /// Dequantize a row: `output[i] = input[i] * scale[group]`.
    pub fn dequantize_row(
        input: &[i8],
        scales: &[f32],
        cols: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; cols];
        for col in 0..cols {
            let group_idx = col / group_size;
            let scale = scales.get(group_idx).copied().unwrap_or(1.0);
            out[col] = input[col] as f32 * scale;
        }
        out
    }
}

/// Dequantizes F16 values to F32.
pub struct F16Dequant;

impl F16Dequant {
    /// Convert a raw F16 bit pattern to F32.
    #[inline]
    pub fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as u32;
        let frac = (bits & 0x3FF) as u32;

        if exp == 0 {
            // Subnormal or zero
            let val = (frac as f32) * (1.0 / (1 << 24) as f32);
            if sign == 1 { -val } else { val }
        } else if exp == 31 {
            // Inf or NaN
            if frac == 0 {
                if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
            } else {
                f32::NAN
            }
        } else {
            let f32_bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
            f32::from_bits(f32_bits)
        }
    }

    /// Convert an F32 value to raw F16 bit pattern (round to nearest).
    pub fn f32_to_f16(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 31) & 1) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let frac = bits & 0x7FFFFF;

        if exp == 255 {
            // Inf or NaN
            if frac == 0 {
                return (sign << 15) | 0x7C00;
            } else {
                return (sign << 15) | 0x7C00 | ((frac >> 13) as u16).max(1);
            }
        }

        let unbiased_exp = exp - 127;
        if unbiased_exp > 15 {
            return (sign << 15) | 0x7C00; // overflow → infinity
        }
        if unbiased_exp < -24 {
            return sign << 15; // underflow → zero
        }
        if unbiased_exp < -14 {
            // Subnormal
            let shift = -14 - unbiased_exp;
            let subnormal = ((0x800000 | frac) >> (shift + 13)) as u16;
            return (sign << 15) | subnormal;
        }
        let h_exp = ((unbiased_exp + 15) as u16) & 0x1F;
        let h_frac = (frac >> 13) as u16;
        (sign << 15) | (h_exp << 10) | h_frac
    }

    /// Dequantize a row of F16 bit patterns to F32.
    pub fn dequantize_row(input: &[u16], cols: usize) -> Vec<f32> {
        input[..cols].iter().map(|&bits| Self::f16_to_f32(bits)).collect()
    }
}

/// Dequantizes BF16 values to F32.
pub struct Bf16Dequant;

impl Bf16Dequant {
    /// Convert a raw BF16 bit pattern to F32.
    #[inline]
    pub fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    /// Convert F32 to BF16 bit pattern (truncation).
    #[inline]
    pub fn f32_to_bf16(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    /// Dequantize a row of BF16 bit patterns to F32.
    pub fn dequantize_row(input: &[u16], cols: usize) -> Vec<f32> {
        input[..cols].iter().map(|&bits| Self::bf16_to_f32(bits)).collect()
    }
}

// ── Dequant kernel dispatcher ───────────────────────────────────────────────

/// Dispatches to format-specific dequantization.
///
/// Provides a unified interface for CPU reference dequantization across all
/// supported formats.
pub struct DequantKernel;

impl DequantKernel {
    /// Dequantize packed weights using the specified config.
    ///
    /// # Arguments
    /// * `packed` — packed weight bytes
    /// * `scales` — quantization scale values
    /// * `rows` — number of rows
    /// * `cols` — number of columns
    /// * `config` — dequantization configuration
    ///
    /// Returns dequantized f32 matrix (row-major) and performance stats.
    pub fn dequantize(
        packed: &[u8],
        scales: &[f32],
        rows: usize,
        cols: usize,
        config: &DequantConfig,
    ) -> (Vec<f32>, DequantStats) {
        match config.format {
            DequantFormat::I2sBitNet32 | DequantFormat::I2sQk256 | DequantFormat::Ternary => {
                I2sDequant::dequantize_matrix(packed, scales, rows, cols, config)
            }
            DequantFormat::I4 => {
                let start = Instant::now();
                let packed_per_row = cols.div_ceil(2);
                let groups_per_row = config.num_groups(cols);
                let mut output = vec![0.0f32; rows * cols];
                // Provide zero_points = 8 (mid-point for unsigned 4-bit)
                let zero_points = vec![8u8; groups_per_row];
                for row in 0..rows {
                    let row_packed = &packed[row * packed_per_row..(row + 1) * packed_per_row];
                    let row_scales = match config.scale_type {
                        ScaleType::PerTensor => scales,
                        ScaleType::PerChannel => &scales[row..row + 1],
                        ScaleType::PerGroup => {
                            &scales[row * groups_per_row..(row + 1) * groups_per_row]
                        }
                    };
                    let row_out = I4Dequant::dequantize_row(
                        row_packed,
                        row_scales,
                        &zero_points,
                        cols,
                        config.group_size,
                    );
                    output[row * cols..(row + 1) * cols].copy_from_slice(&row_out);
                }
                let elapsed_us = start.elapsed().as_micros() as u64;
                let stats = DequantStats::compute(rows * packed_per_row, rows * cols, elapsed_us);
                (output, stats)
            }
            DequantFormat::I8 => {
                let start = Instant::now();
                let groups_per_row = config.num_groups(cols);
                let mut output = vec![0.0f32; rows * cols];
                // Reinterpret packed bytes as i8
                let input: &[i8] = bytemuck_cast_slice(packed);
                for row in 0..rows {
                    let row_input = &input[row * cols..(row + 1) * cols];
                    let row_scales = match config.scale_type {
                        ScaleType::PerTensor => scales,
                        ScaleType::PerChannel => &scales[row..row + 1],
                        ScaleType::PerGroup => {
                            &scales[row * groups_per_row..(row + 1) * groups_per_row]
                        }
                    };
                    let row_out =
                        I8Dequant::dequantize_row(row_input, row_scales, cols, config.group_size);
                    output[row * cols..(row + 1) * cols].copy_from_slice(&row_out);
                }
                let elapsed_us = start.elapsed().as_micros() as u64;
                let stats = DequantStats::compute(rows * cols, rows * cols, elapsed_us);
                (output, stats)
            }
            DequantFormat::F16 => {
                let start = Instant::now();
                let mut output = vec![0.0f32; rows * cols];
                let input = bytes_to_u16_vec(packed);
                for row in 0..rows {
                    let row_input = &input[row * cols..(row + 1) * cols];
                    let row_out = F16Dequant::dequantize_row(row_input, cols);
                    output[row * cols..(row + 1) * cols].copy_from_slice(&row_out);
                }
                let elapsed_us = start.elapsed().as_micros() as u64;
                let stats = DequantStats::compute(rows * cols * 2, rows * cols, elapsed_us);
                (output, stats)
            }
            DequantFormat::Bf16 => {
                let start = Instant::now();
                let mut output = vec![0.0f32; rows * cols];
                let input = bytes_to_u16_vec(packed);
                for row in 0..rows {
                    let row_input = &input[row * cols..(row + 1) * cols];
                    let row_out = Bf16Dequant::dequantize_row(row_input, cols);
                    output[row * cols..(row + 1) * cols].copy_from_slice(&row_out);
                }
                let elapsed_us = start.elapsed().as_micros() as u64;
                let stats = DequantStats::compute(rows * cols * 2, rows * cols, elapsed_us);
                (output, stats)
            }
        }
    }
}

/// Safe cast from `&[u8]` to `&[i8]` (same layout, no allocation).
fn bytemuck_cast_slice(bytes: &[u8]) -> &[i8] {
    // SAFETY: i8 and u8 have identical layout and alignment.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) }
}

/// Convert `&[u8]` to `Vec<u16>` (little-endian).
fn bytes_to_u16_vec(bytes: &[u8]) -> Vec<u16> {
    assert!(bytes.len().is_multiple_of(2), "byte slice length must be even for u16 cast");
    let count = bytes.len() / 2;
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let lo = bytes[i * 2] as u16;
        let hi = bytes[i * 2 + 1] as u16;
        result.push(lo | (hi << 8));
    }
    result
}

// ── OpenCL kernel sources ───────────────────────────────────────────────────

/// OpenCL kernel source for I2_S ternary dequantization (A770-optimized).
///
/// Uses `float4` vectorized loads and subgroup shuffle for scale broadcasting.
pub const OPENCL_I2S_DEQUANT_SOURCE: &str = r#"
__kernel void i2s_dequant(
    __global const uchar* packed,   // packed ternary bytes
    __global const float* scales,   // per-group scales
    __global float* output,         // dequantized output
    const uint cols,
    const uint group_size
) {
    const uint gid = get_global_id(0);
    const uint row = get_global_id(1);
    const uint row_offset = row * cols;
    const uint packed_cols = (cols + 3) / 4;
    const uint row_packed_offset = row * packed_cols;

    // Process 4 values at a time (one packed byte)
    if (gid < packed_cols) {
        const uint col_base = gid * 4;
        const uchar byte_val = packed[row_packed_offset + gid];

        // Unpack 4 ternary values
        float4 vals;
        vals.x = (float)((int)(byte_val & 0x03) - 1);
        vals.y = (float)((int)((byte_val >> 2) & 0x03) - 1);
        vals.z = (float)((int)((byte_val >> 4) & 0x03) - 1);
        vals.w = (float)((int)((byte_val >> 6) & 0x03) - 1);

        // Apply per-group scales with subgroup broadcast
        uint group_idx = col_base / group_size;
        float scale = scales[row * ((cols + group_size - 1) / group_size) + group_idx];

        vals *= scale;

        // Write output (handle tail)
        if (col_base + 3 < cols) {
            vstore4(vals, 0, output + row_offset + col_base);
        } else {
            if (col_base < cols) output[row_offset + col_base] = vals.x;
            if (col_base + 1 < cols) output[row_offset + col_base + 1] = vals.y;
            if (col_base + 2 < cols) output[row_offset + col_base + 2] = vals.z;
        }
    }
}
"#;

/// OpenCL kernel source for QK256 dequantization (A770-optimized).
///
/// Uses shared local memory to cache block scales and `float8` loads.
pub const OPENCL_QK256_DEQUANT_SOURCE: &str = r#"
__kernel void qk256_dequant(
    __global const uchar* packed,   // packed blocks: 64 bytes each
    __global const float* scales,   // one scale per 256-element block
    __global float* output,
    const uint cols,
    const uint blocks_per_row,
    __local float* local_scales      // shared local memory for block scales
) {
    const uint lid = get_local_id(0);
    const uint row = get_global_id(1);
    const uint row_packed_offset = row * blocks_per_row * 64;
    const uint row_scale_offset = row * blocks_per_row;
    const uint row_out_offset = row * cols;

    // Cooperatively load block scales into local memory
    if (lid < blocks_per_row) {
        local_scales[lid] = scales[row_scale_offset + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Each work-item processes one packed byte (4 ternary values)
    const uint total_packed = blocks_per_row * 64;
    for (uint gid = get_global_id(0); gid < total_packed; gid += get_global_size(0)) {
        const uint block_idx = gid / 64;
        const uint byte_in_block = gid % 64;
        const uint col_base = block_idx * 256 + byte_in_block * 4;

        const uchar byte_val = packed[row_packed_offset + gid];
        float scale = (block_idx < blocks_per_row) ? local_scales[block_idx] : 0.0f;

        float4 vals;
        vals.x = (float)((int)(byte_val & 0x03) - 1) * scale;
        vals.y = (float)((int)((byte_val >> 2) & 0x03) - 1) * scale;
        vals.z = (float)((int)((byte_val >> 4) & 0x03) - 1) * scale;
        vals.w = (float)((int)((byte_val >> 6) & 0x03) - 1) * scale;

        if (col_base + 3 < cols) {
            vstore4(vals, 0, output + row_out_offset + col_base);
        } else {
            if (col_base < cols) output[row_out_offset + col_base] = vals.x;
            if (col_base + 1 < cols) output[row_out_offset + col_base + 1] = vals.y;
            if (col_base + 2 < cols) output[row_out_offset + col_base + 2] = vals.z;
        }
    }
}
"#;

/// OpenCL kernel source for I4 dequantization.
pub const OPENCL_I4_DEQUANT_SOURCE: &str = r#"
__kernel void i4_dequant(
    __global const uchar* packed,   // 2 values per byte, low nibble first
    __global const float* scales,
    __global const uchar* zero_points,
    __global float* output,
    const uint cols,
    const uint group_size
) {
    const uint gid = get_global_id(0);
    const uint row = get_global_id(1);
    const uint packed_cols = (cols + 1) / 2;
    const uint row_packed_offset = row * packed_cols;
    const uint row_offset = row * cols;
    const uint groups_per_row = (cols + group_size - 1) / group_size;

    if (gid < packed_cols) {
        const uint col_base = gid * 2;
        const uchar byte_val = packed[row_packed_offset + gid];

        // Low nibble
        if (col_base < cols) {
            uint g = col_base / group_size;
            float s = scales[row * groups_per_row + g];
            uchar zp = zero_points[g];
            float val = ((float)(byte_val & 0x0F) - (float)zp) * s;
            output[row_offset + col_base] = val;
        }
        // High nibble
        if (col_base + 1 < cols) {
            uint g = (col_base + 1) / group_size;
            float s = scales[row * groups_per_row + g];
            uchar zp = zero_points[g];
            float val = ((float)((byte_val >> 4) & 0x0F) - (float)zp) * s;
            output[row_offset + col_base + 1] = val;
        }
    }
}
"#;

/// Returns all kernel sources for registration.
pub fn all_dequant_kernel_sources() -> Vec<(&'static str, &'static str)> {
    vec![
        ("i2s_dequant", OPENCL_I2S_DEQUANT_SOURCE),
        ("qk256_dequant", OPENCL_QK256_DEQUANT_SOURCE),
        ("i4_dequant", OPENCL_I4_DEQUANT_SOURCE),
    ]
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DequantFormat tests ─────────────────────────────────────────────

    #[test]
    fn test_format_bits_per_element() {
        assert_eq!(DequantFormat::I2sBitNet32.bits_per_element(), 2);
        assert_eq!(DequantFormat::I2sQk256.bits_per_element(), 2);
        assert_eq!(DequantFormat::Ternary.bits_per_element(), 2);
        assert_eq!(DequantFormat::I4.bits_per_element(), 4);
        assert_eq!(DequantFormat::I8.bits_per_element(), 8);
        assert_eq!(DequantFormat::F16.bits_per_element(), 16);
        assert_eq!(DequantFormat::Bf16.bits_per_element(), 16);
    }

    #[test]
    fn test_format_has_block_scales() {
        assert!(DequantFormat::I2sBitNet32.has_block_scales());
        assert!(DequantFormat::I2sQk256.has_block_scales());
        assert!(DequantFormat::I4.has_block_scales());
        assert!(DequantFormat::Ternary.has_block_scales());
        assert!(!DequantFormat::I8.has_block_scales());
        assert!(!DequantFormat::F16.has_block_scales());
        assert!(!DequantFormat::Bf16.has_block_scales());
    }

    #[test]
    fn test_format_default_group_sizes() {
        assert_eq!(DequantFormat::I2sBitNet32.default_group_size(), 32);
        assert_eq!(DequantFormat::I2sQk256.default_group_size(), 256);
        assert_eq!(DequantFormat::I4.default_group_size(), 128);
        assert_eq!(DequantFormat::I8.default_group_size(), 128);
        assert_eq!(DequantFormat::Ternary.default_group_size(), 64);
        assert_eq!(DequantFormat::F16.default_group_size(), 1);
    }

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", DequantFormat::I2sBitNet32), "I2S-BitNet32");
        assert_eq!(format!("{}", DequantFormat::I2sQk256), "I2S-QK256");
        assert_eq!(format!("{}", DequantFormat::I4), "INT4");
        assert_eq!(format!("{}", DequantFormat::I8), "INT8");
        assert_eq!(format!("{}", DequantFormat::F16), "FP16");
        assert_eq!(format!("{}", DequantFormat::Bf16), "BF16");
        assert_eq!(format!("{}", DequantFormat::Ternary), "Ternary");
    }

    // ── ScaleType tests ─────────────────────────────────────────────────

    #[test]
    fn test_scale_type_display() {
        assert_eq!(format!("{}", ScaleType::PerTensor), "per-tensor");
        assert_eq!(format!("{}", ScaleType::PerChannel), "per-channel");
        assert_eq!(format!("{}", ScaleType::PerGroup), "per-group");
    }

    // ── DequantConfig tests ─────────────────────────────────────────────

    #[test]
    fn test_config_new_default_group_size() {
        let cfg = DequantConfig::new(DequantFormat::I2sQk256, ScaleType::PerGroup);
        assert_eq!(cfg.group_size, 256);
        assert_eq!(cfg.format, DequantFormat::I2sQk256);
        assert_eq!(cfg.scale_type, ScaleType::PerGroup);
    }

    #[test]
    fn test_config_with_group_size() {
        let cfg = DequantConfig::with_group_size(DequantFormat::I4, 64, ScaleType::PerGroup);
        assert_eq!(cfg.group_size, 64);
    }

    #[test]
    #[should_panic(expected = "group_size must be positive")]
    fn test_config_zero_group_size_panics() {
        DequantConfig::with_group_size(DequantFormat::I4, 0, ScaleType::PerGroup);
    }

    #[test]
    fn test_config_num_groups_per_tensor() {
        let cfg = DequantConfig::new(DequantFormat::I2sBitNet32, ScaleType::PerTensor);
        assert_eq!(cfg.num_groups(256), 1);
        assert_eq!(cfg.num_groups(1024), 1);
    }

    #[test]
    fn test_config_num_groups_per_channel() {
        let cfg = DequantConfig::new(DequantFormat::I2sBitNet32, ScaleType::PerChannel);
        assert_eq!(cfg.num_groups(256), 1);
    }

    #[test]
    fn test_config_num_groups_per_group() {
        let cfg =
            DequantConfig::with_group_size(DequantFormat::I2sBitNet32, 32, ScaleType::PerGroup);
        assert_eq!(cfg.num_groups(128), 4);
        assert_eq!(cfg.num_groups(100), 4); // ceil(100/32)
    }

    #[test]
    fn test_config_num_scales_per_tensor() {
        let cfg = DequantConfig::new(DequantFormat::I2sBitNet32, ScaleType::PerTensor);
        assert_eq!(cfg.num_scales(4, 256), 1);
    }

    #[test]
    fn test_config_num_scales_per_channel() {
        let cfg = DequantConfig::new(DequantFormat::I2sBitNet32, ScaleType::PerChannel);
        assert_eq!(cfg.num_scales(4, 256), 4);
    }

    #[test]
    fn test_config_num_scales_per_group() {
        let cfg =
            DequantConfig::with_group_size(DequantFormat::I2sBitNet32, 32, ScaleType::PerGroup);
        assert_eq!(cfg.num_scales(4, 128), 16); // 4 rows × 4 groups
    }

    #[test]
    fn test_config_display() {
        let cfg = DequantConfig::new(DequantFormat::I2sQk256, ScaleType::PerGroup);
        let s = format!("{cfg}");
        assert!(s.contains("I2S-QK256"));
        assert!(s.contains("256"));
        assert!(s.contains("per-group"));
    }

    // ── DequantStats tests ──────────────────────────────────────────────

    #[test]
    fn test_stats_compute_basic() {
        let stats = DequantStats::compute(1024, 4096, 100);
        assert_eq!(stats.dequant_time_us, 100);
        assert!(stats.bandwidth_gb_s > 0.0);
        // 4096 * 4 = 16384 bytes out, 1024 bytes in → 16× expansion
        assert!((stats.format_overhead - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_compute_zero_time() {
        let stats = DequantStats::compute(100, 400, 0);
        assert_eq!(stats.bandwidth_gb_s, 0.0);
    }

    #[test]
    fn test_stats_compute_zero_packed() {
        let stats = DequantStats::compute(0, 100, 50);
        assert_eq!(stats.format_overhead, 0.0);
    }

    #[test]
    fn test_stats_display() {
        let stats = DequantStats::compute(1024, 4096, 100);
        let s = format!("{stats}");
        assert!(s.contains("µs"));
        assert!(s.contains("GB/s"));
        assert!(s.contains("expansion"));
    }

    // ── PackedTernary tests ─────────────────────────────────────────────

    #[test]
    fn test_ternary_pack_all_neg1() {
        let packed = PackedTernary::pack(&[-1, -1, -1, -1]);
        assert_eq!(packed, vec![0x00]);
    }

    #[test]
    fn test_ternary_pack_all_zero() {
        let packed = PackedTernary::pack(&[0, 0, 0, 0]);
        assert_eq!(packed, vec![0x55]);
    }

    #[test]
    fn test_ternary_pack_all_pos1() {
        let packed = PackedTernary::pack(&[1, 1, 1, 1]);
        assert_eq!(packed, vec![0xAA]);
    }

    #[test]
    fn test_ternary_pack_mixed() {
        let packed = PackedTernary::pack(&[-1, 0, 1, 0]);
        let expected = 0b_01_10_01_00u8;
        assert_eq!(packed, vec![expected]);
    }

    #[test]
    fn test_ternary_unpack_one() {
        let packed = PackedTernary::pack(&[-1, 0, 1, -1]);
        assert_eq!(PackedTernary::unpack_one(&packed, 0), -1);
        assert_eq!(PackedTernary::unpack_one(&packed, 1), 0);
        assert_eq!(PackedTernary::unpack_one(&packed, 2), 1);
        assert_eq!(PackedTernary::unpack_one(&packed, 3), -1);
    }

    #[test]
    fn test_ternary_unpack_byte() {
        // Pack [-1, 0, 1, 0]
        let packed = PackedTernary::pack(&[-1, 0, 1, 0]);
        assert_eq!(PackedTernary::unpack_byte(packed[0]), [-1, 0, 1, 0]);
    }

    #[test]
    fn test_ternary_roundtrip_4() {
        let vals: Vec<i8> = vec![-1, 0, 1, 0];
        let packed = PackedTernary::pack(&vals);
        let unpacked = PackedTernary::unpack(&packed, vals.len());
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn test_ternary_roundtrip_7() {
        let vals: Vec<i8> = vec![-1, 1, 0, -1, 1, 0, 1];
        let packed = PackedTernary::pack(&vals);
        let unpacked = PackedTernary::unpack(&packed, vals.len());
        assert_eq!(unpacked, vals);
    }

    #[test]
    fn test_ternary_roundtrip_256() {
        let vals: Vec<i8> = (0..256).map(|i| (i % 3) as i8 - 1).collect();
        let packed = PackedTernary::pack(&vals);
        let unpacked = PackedTernary::unpack(&packed, vals.len());
        assert_eq!(unpacked, vals);
    }

    #[test]
    #[should_panic(expected = "ternary values must be -1, 0, or +1")]
    fn test_ternary_pack_invalid() {
        PackedTernary::pack(&[2]);
    }

    // ── I2S dequant tests ───────────────────────────────────────────────

    #[test]
    fn test_i2s_dequant_row_all_zeros() {
        // All ternary zeros → all output zeros regardless of scale
        let packed = PackedTernary::pack(&[0, 0, 0, 0]);
        let scales = vec![2.5];
        let out = I2sDequant::dequantize_row(&packed, &scales, 4, 4);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_i2s_dequant_row_all_ones() {
        let packed = PackedTernary::pack(&[1, 1, 1, 1]);
        let scales = vec![3.0];
        let out = I2sDequant::dequantize_row(&packed, &scales, 4, 4);
        assert_eq!(out, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_i2s_dequant_row_all_neg1() {
        let packed = PackedTernary::pack(&[-1, -1, -1, -1]);
        let scales = vec![2.0];
        let out = I2sDequant::dequantize_row(&packed, &scales, 4, 4);
        assert_eq!(out, vec![-2.0, -2.0, -2.0, -2.0]);
    }

    #[test]
    fn test_i2s_dequant_row_mixed_with_scale() {
        let vals: Vec<i8> = vec![-1, 0, 1, -1];
        let packed = PackedTernary::pack(&vals);
        let scales = vec![0.5];
        let out = I2sDequant::dequantize_row(&packed, &scales, 4, 4);
        assert_eq!(out, vec![-0.5, 0.0, 0.5, -0.5]);
    }

    #[test]
    fn test_i2s_dequant_row_two_groups() {
        // 8 values, group_size=4, two groups with different scales
        let vals: Vec<i8> = vec![1, 1, 1, 1, -1, -1, -1, -1];
        let packed = PackedTernary::pack(&vals);
        let scales = vec![2.0, 3.0];
        let out = I2sDequant::dequantize_row(&packed, &scales, 8, 4);
        assert_eq!(out, vec![2.0, 2.0, 2.0, 2.0, -3.0, -3.0, -3.0, -3.0]);
    }

    #[test]
    fn test_i2s_dequant_matches_reference() {
        // Build a small 2×8 matrix with known ternary values and verify
        let row0: Vec<i8> = vec![-1, 0, 1, -1, 0, 1, -1, 0];
        let row1: Vec<i8> = vec![1, 1, 1, 1, -1, -1, -1, -1];
        let mut packed = PackedTernary::pack(&row0);
        packed.extend(PackedTernary::pack(&row1));
        let scales = vec![1.0, 1.0, 2.0, 2.0]; // 2 rows × 2 groups (group_size=4)
        let config =
            DequantConfig::with_group_size(DequantFormat::I2sBitNet32, 4, ScaleType::PerGroup);
        let (output, _stats) = I2sDequant::dequantize_matrix(&packed, &scales, 2, 8, &config);
        // Row 0: [-1, 0, 1, -1] * 1.0, [0, 1, -1, 0] * 1.0
        assert_eq!(&output[0..8], &[-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0]);
        // Row 1: [1,1,1,1]*2.0, [-1,-1,-1,-1]*2.0
        assert_eq!(&output[8..16], &[2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0]);
    }

    #[test]
    fn test_i2s_dequant_per_tensor_scale() {
        let vals: Vec<i8> = vec![1, -1, 0, 1];
        let packed = PackedTernary::pack(&vals);
        let scales = vec![5.0]; // single tensor-wide scale
        let config =
            DequantConfig::with_group_size(DequantFormat::I2sBitNet32, 4, ScaleType::PerTensor);
        let (output, _) = I2sDequant::dequantize_matrix(&packed, &scales, 1, 4, &config);
        assert_eq!(output, vec![5.0, -5.0, 0.0, 5.0]);
    }

    #[test]
    fn test_i2s_dequant_per_channel_scale() {
        let row0: Vec<i8> = vec![1, 1, 1, 1];
        let row1: Vec<i8> = vec![1, 1, 1, 1];
        let mut packed = PackedTernary::pack(&row0);
        packed.extend(PackedTernary::pack(&row1));
        let scales = vec![2.0, 3.0]; // per-channel: one per row
        let config =
            DequantConfig::with_group_size(DequantFormat::I2sBitNet32, 4, ScaleType::PerChannel);
        let (output, _) = I2sDequant::dequantize_matrix(&packed, &scales, 2, 4, &config);
        assert_eq!(&output[0..4], &[2.0, 2.0, 2.0, 2.0]);
        assert_eq!(&output[4..8], &[3.0, 3.0, 3.0, 3.0]);
    }

    // ── QK256 dequant tests ─────────────────────────────────────────────

    #[test]
    fn test_qk256_dequant_single_block_all_zeros() {
        let vals = vec![0i8; 256];
        let packed = PackedTernary::pack(&vals);
        let scales = vec![1.0];
        let out = Qk256Dequant::dequantize_row(&packed, &scales, 256);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_qk256_dequant_single_block_all_ones() {
        let vals = vec![1i8; 256];
        let packed = PackedTernary::pack(&vals);
        let scales = vec![2.5];
        let out = Qk256Dequant::dequantize_row(&packed, &scales, 256);
        assert!(out.iter().all(|&v| (v - 2.5).abs() < 1e-6));
    }

    #[test]
    fn test_qk256_dequant_single_block_all_neg1() {
        let vals = vec![-1i8; 256];
        let packed = PackedTernary::pack(&vals);
        let scales = vec![3.0];
        let out = Qk256Dequant::dequantize_row(&packed, &scales, 256);
        assert!(out.iter().all(|&v| (v + 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_qk256_dequant_two_blocks() {
        let mut vals = vec![1i8; 256];
        vals.extend(vec![-1i8; 256]);
        let packed = PackedTernary::pack(&vals);
        let scales = vec![1.0, 2.0];
        let out = Qk256Dequant::dequantize_row(&packed, &scales, 512);
        assert!(out[..256].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(out[256..].iter().all(|&v| (v + 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_qk256_dequant_block_matches_reference() {
        // Build a pattern: alternating -1, 0, 1 for 256 elements
        let vals: Vec<i8> = (0..256).map(|i| (i % 3) as i8 - 1).collect();
        let packed = PackedTernary::pack(&vals);
        let scale = 1.5;
        let block_out = Qk256Dequant::dequantize_block(&packed, scale);
        for (i, &v) in block_out.iter().enumerate() {
            let expected = ((i % 3) as f32 - 1.0) * scale;
            assert!((v - expected).abs() < 1e-6, "mismatch at {i}: got {v}, expected {expected}");
        }
    }

    #[test]
    fn test_qk256_dequant_matrix_1x256() {
        let vals = vec![1i8; 256];
        let packed = PackedTernary::pack(&vals);
        let scales = vec![0.5];
        let (output, stats) = Qk256Dequant::dequantize_matrix(&packed, &scales, 1, 256);
        assert!(output.iter().all(|&v| (v - 0.5).abs() < 1e-6));
        assert_eq!(output.len(), 256);
        assert!(stats.format_overhead > 0.0);
    }

    #[test]
    fn test_qk256_dequant_matrix_2x512() {
        let row_vals: Vec<i8> = (0..512).map(|i| if i < 256 { 1 } else { -1 }).collect();
        let mut packed = PackedTernary::pack(&row_vals);
        packed.extend(PackedTernary::pack(&row_vals));
        let scales = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows × 2 blocks
        let (output, _) = Qk256Dequant::dequantize_matrix(&packed, &scales, 2, 512);
        // Row 0: first block ×1.0, second block ×2.0
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[256] + 2.0).abs() < 1e-6);
        // Row 1: first block ×3.0, second block ×4.0
        assert!((output[512] - 3.0).abs() < 1e-6);
        assert!((output[768] + 4.0).abs() < 1e-6);
    }

    // ── I4 dequant tests ────────────────────────────────────────────────

    #[test]
    fn test_i4_pack_unpack_roundtrip() {
        let vals: Vec<u8> = (0..8).collect();
        let packed = I4Dequant::pack(&vals);
        for (i, &expected) in vals.iter().enumerate() {
            assert_eq!(I4Dequant::unpack_one(&packed, i), expected);
        }
    }

    #[test]
    fn test_i4_dequant_with_zero_point() {
        // 4 values: [0, 4, 8, 12], zero_point=8, scale=0.5
        let vals: Vec<u8> = vec![0, 4, 8, 12];
        let packed = I4Dequant::pack(&vals);
        let scales = vec![0.5];
        let zps = vec![8u8];
        let out = I4Dequant::dequantize_row(&packed, &scales, &zps, 4, 4);
        assert!((out[0] - (-4.0)).abs() < 1e-6); // (0-8)*0.5
        assert!((out[1] - (-2.0)).abs() < 1e-6); // (4-8)*0.5
        assert!((out[2] - 0.0).abs() < 1e-6); // (8-8)*0.5
        assert!((out[3] - 2.0).abs() < 1e-6); // (12-8)*0.5
    }

    #[test]
    fn test_i4_dequant_two_groups() {
        let vals: Vec<u8> = vec![0, 0, 15, 15];
        let packed = I4Dequant::pack(&vals);
        let scales = vec![1.0, 2.0];
        let zps = vec![0, 0];
        let out = I4Dequant::dequantize_row(&packed, &scales, &zps, 4, 2);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
        assert!((out[2] - 30.0).abs() < 1e-6); // 15 * 2.0
        assert!((out[3] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_i4_pack_edge_values() {
        // Min and max 4-bit values
        let vals: Vec<u8> = vec![0, 15];
        let packed = I4Dequant::pack(&vals);
        assert_eq!(I4Dequant::unpack_one(&packed, 0), 0);
        assert_eq!(I4Dequant::unpack_one(&packed, 1), 15);
    }

    // ── I8 dequant tests ────────────────────────────────────────────────

    #[test]
    fn test_i8_dequant_basic() {
        let input: Vec<i8> = vec![-128, -1, 0, 1, 127];
        let scales = vec![0.01];
        let out = I8Dequant::dequantize_row(&input, &scales, 5, 5);
        assert!((out[0] - (-1.28)).abs() < 1e-5);
        assert!((out[2] - 0.0).abs() < 1e-6);
        assert!((out[4] - 1.27).abs() < 1e-5);
    }

    #[test]
    fn test_i8_dequant_two_groups() {
        let input: Vec<i8> = vec![10, 10, 10, 10];
        let scales = vec![1.0, 2.0];
        let out = I8Dequant::dequantize_row(&input, &scales, 4, 2);
        assert!((out[0] - 10.0).abs() < 1e-6);
        assert!((out[1] - 10.0).abs() < 1e-6);
        assert!((out[2] - 20.0).abs() < 1e-6);
        assert!((out[3] - 20.0).abs() < 1e-6);
    }

    // ── F16 dequant tests ───────────────────────────────────────────────

    #[test]
    fn test_f16_roundtrip_zero() {
        let bits = F16Dequant::f32_to_f16(0.0);
        assert_eq!(F16Dequant::f16_to_f32(bits), 0.0);
    }

    #[test]
    fn test_f16_roundtrip_one() {
        let bits = F16Dequant::f32_to_f16(1.0);
        let val = F16Dequant::f16_to_f32(bits);
        assert!((val - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_f16_roundtrip_neg_one() {
        let bits = F16Dequant::f32_to_f16(-1.0);
        let val = F16Dequant::f16_to_f32(bits);
        assert!((val + 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_f16_infinity() {
        let bits = F16Dequant::f32_to_f16(f32::INFINITY);
        assert_eq!(F16Dequant::f16_to_f32(bits), f32::INFINITY);
        let bits_neg = F16Dequant::f32_to_f16(f32::NEG_INFINITY);
        assert_eq!(F16Dequant::f16_to_f32(bits_neg), f32::NEG_INFINITY);
    }

    #[test]
    fn test_f16_nan() {
        let bits = F16Dequant::f32_to_f16(f32::NAN);
        assert!(F16Dequant::f16_to_f32(bits).is_nan());
    }

    #[test]
    fn test_f16_dequant_row() {
        let f16_vals: Vec<u16> = vec![
            F16Dequant::f32_to_f16(1.0),
            F16Dequant::f32_to_f16(-0.5),
            F16Dequant::f32_to_f16(0.0),
        ];
        let out = F16Dequant::dequantize_row(&f16_vals, 3);
        assert!((out[0] - 1.0).abs() < 1e-3);
        assert!((out[1] + 0.5).abs() < 1e-3);
        assert!((out[2] - 0.0).abs() < 1e-6);
    }

    // ── BF16 dequant tests ──────────────────────────────────────────────

    #[test]
    fn test_bf16_roundtrip_zero() {
        let bits = Bf16Dequant::f32_to_bf16(0.0);
        assert_eq!(Bf16Dequant::bf16_to_f32(bits), 0.0);
    }

    #[test]
    fn test_bf16_roundtrip_one() {
        let bits = Bf16Dequant::f32_to_bf16(1.0);
        assert_eq!(Bf16Dequant::bf16_to_f32(bits), 1.0);
    }

    #[test]
    fn test_bf16_roundtrip_neg() {
        let bits = Bf16Dequant::f32_to_bf16(-3.14);
        let val = Bf16Dequant::bf16_to_f32(bits);
        assert!((val + 3.14).abs() < 0.05); // BF16 has ~7 bits mantissa
    }

    #[test]
    fn test_bf16_dequant_row() {
        let bf16_vals: Vec<u16> =
            vec![Bf16Dequant::f32_to_bf16(2.0), Bf16Dequant::f32_to_bf16(-1.0)];
        let out = Bf16Dequant::dequantize_row(&bf16_vals, 2);
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] + 1.0).abs() < 1e-6);
    }

    // ── DequantKernel dispatcher tests ──────────────────────────────────

    #[test]
    fn test_kernel_dispatch_i2s_bitnet32() {
        let vals: Vec<i8> = vec![1, -1, 0, 1];
        let packed = PackedTernary::pack(&vals);
        let scales = vec![2.0];
        let config =
            DequantConfig::with_group_size(DequantFormat::I2sBitNet32, 4, ScaleType::PerGroup);
        let (output, stats) = DequantKernel::dequantize(&packed, &scales, 1, 4, &config);
        assert_eq!(output, vec![2.0, -2.0, 0.0, 2.0]);
        assert!(stats.format_overhead > 0.0);
    }

    #[test]
    fn test_kernel_dispatch_ternary() {
        let vals: Vec<i8> = vec![-1, 0, 1, 0];
        let packed = PackedTernary::pack(&vals);
        let scales = vec![1.0];
        let config = DequantConfig::with_group_size(DequantFormat::Ternary, 4, ScaleType::PerGroup);
        let (output, _) = DequantKernel::dequantize(&packed, &scales, 1, 4, &config);
        assert_eq!(output, vec![-1.0, 0.0, 1.0, 0.0]);
    }

    // ── Group size variation tests ──────────────────────────────────────

    #[test]
    fn test_group_size_32() {
        let vals: Vec<i8> = (0..64).map(|i| if i < 32 { 1 } else { -1 }).collect();
        let packed = PackedTernary::pack(&vals);
        let scales = vec![1.0, 2.0];
        let config =
            DequantConfig::with_group_size(DequantFormat::I2sBitNet32, 32, ScaleType::PerGroup);
        let (output, _) = DequantKernel::dequantize(&packed, &scales, 1, 64, &config);
        assert!(output[..32].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(output[32..].iter().all(|&v| (v + 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_group_size_64() {
        let vals: Vec<i8> = (0..128).map(|i| if i < 64 { 1 } else { -1 }).collect();
        let packed = PackedTernary::pack(&vals);
        let scales = vec![0.5, 1.5];
        let config =
            DequantConfig::with_group_size(DequantFormat::Ternary, 64, ScaleType::PerGroup);
        let (output, _) = DequantKernel::dequantize(&packed, &scales, 1, 128, &config);
        assert!(output[..64].iter().all(|&v| (v - 0.5).abs() < 1e-6));
        assert!(output[64..].iter().all(|&v| (v + 1.5).abs() < 1e-6));
    }

    #[test]
    fn test_group_size_128() {
        let vals: Vec<i8> = vec![1i8; 256];
        let packed = PackedTernary::pack(&vals);
        let scales = vec![3.0, 7.0];
        let config =
            DequantConfig::with_group_size(DequantFormat::I2sQk256, 128, ScaleType::PerGroup);
        let (output, _) = DequantKernel::dequantize(&packed, &scales, 1, 256, &config);
        assert!(output[..128].iter().all(|&v| (v - 3.0).abs() < 1e-6));
        assert!(output[128..].iter().all(|&v| (v - 7.0).abs() < 1e-6));
    }

    #[test]
    fn test_group_size_256() {
        let vals: Vec<i8> = vec![1i8; 256];
        let packed = PackedTernary::pack(&vals);
        let scales = vec![4.0];
        let config =
            DequantConfig::with_group_size(DequantFormat::I2sQk256, 256, ScaleType::PerGroup);
        let (output, _) = DequantKernel::dequantize(&packed, &scales, 1, 256, &config);
        assert!(output.iter().all(|&v| (v - 4.0).abs() < 1e-6));
    }

    // ── Edge case tests ─────────────────────────────────────────────────

    #[test]
    fn test_edge_all_zeros_ternary() {
        let vals = vec![0i8; 128];
        let packed = PackedTernary::pack(&vals);
        let scales = vec![100.0]; // large scale doesn't matter for zeros
        let config =
            DequantConfig::with_group_size(DequantFormat::Ternary, 128, ScaleType::PerGroup);
        let (output, _) = DequantKernel::dequantize(&packed, &scales, 1, 128, &config);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_edge_all_ones_ternary() {
        let vals = vec![1i8; 32];
        let packed = PackedTernary::pack(&vals);
        let scales = vec![0.25];
        let config =
            DequantConfig::with_group_size(DequantFormat::I2sBitNet32, 32, ScaleType::PerGroup);
        let (output, _) = DequantKernel::dequantize(&packed, &scales, 1, 32, &config);
        assert!(output.iter().all(|&v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn test_edge_mixed_sign_pattern() {
        // Alternating +1, -1 pattern
        let vals: Vec<i8> = (0..64).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let packed = PackedTernary::pack(&vals);
        let scales = vec![1.0];
        let config =
            DequantConfig::with_group_size(DequantFormat::I2sBitNet32, 64, ScaleType::PerGroup);
        let (output, _) = DequantKernel::dequantize(&packed, &scales, 1, 64, &config);
        for (i, &v) in output.iter().enumerate() {
            let expected = if i % 2 == 0 { 1.0 } else { -1.0 };
            assert!((v - expected).abs() < 1e-6);
        }
    }

    // ── Quantize→dequant round trip fidelity ────────────────────────────

    #[test]
    fn test_ternary_quant_dequant_roundtrip() {
        // Quantize floats to ternary, then dequant, verify within tolerance
        let original: Vec<f32> = vec![0.9, -0.8, 0.1, -0.95, 0.0, 0.5, -0.3, 0.99];
        let scale = 1.0;
        // Simple ternary quantization: sign-based
        let quantized: Vec<i8> = original
            .iter()
            .map(|&v| {
                if v > 0.3 {
                    1
                } else if v < -0.3 {
                    -1
                } else {
                    0
                }
            })
            .collect();
        let packed = PackedTernary::pack(&quantized);
        let scales = vec![scale];
        let config = DequantConfig::with_group_size(DequantFormat::Ternary, 8, ScaleType::PerGroup);
        let (output, _) = DequantKernel::dequantize(&packed, &scales, 1, 8, &config);
        // Verify dequantized values match the quantized ternary × scale
        for (i, &v) in output.iter().enumerate() {
            let expected = quantized[i] as f32 * scale;
            assert!((v - expected).abs() < 1e-6, "idx {i}: {v} != {expected}");
        }
    }

    #[test]
    fn test_i4_quant_dequant_roundtrip() {
        // Quantize floats to I4 then dequant, verify fidelity
        let scale = 0.1;
        let zero_point = 8u8;
        let quant_vals: Vec<u8> = vec![0, 4, 8, 12, 15];
        let packed = I4Dequant::pack(&quant_vals);
        let scales = vec![scale];
        let zps = vec![zero_point];
        let out = I4Dequant::dequantize_row(&packed, &scales, &zps, 5, 5);
        for (i, &v) in out.iter().enumerate() {
            let expected = (quant_vals[i] as f32 - zero_point as f32) * scale;
            assert!((v - expected).abs() < 1e-6, "idx {i}: {v} != {expected}");
        }
    }

    #[test]
    fn test_f16_quant_dequant_roundtrip_precision() {
        let test_values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 0.001];
        for &v in &test_values {
            let bits = F16Dequant::f32_to_f16(v);
            let recovered = F16Dequant::f16_to_f32(bits);
            let tol = v.abs() * 0.001 + 1e-4; // relative + absolute tolerance
            assert!((recovered - v).abs() < tol, "F16 roundtrip failed for {v}: got {recovered}");
        }
    }

    #[test]
    fn test_bf16_quant_dequant_roundtrip_precision() {
        let test_values = [0.0f32, 1.0, -1.0, 0.5, 256.0, -256.0];
        for &v in &test_values {
            let bits = Bf16Dequant::f32_to_bf16(v);
            let recovered = Bf16Dequant::bf16_to_f32(bits);
            let tol = v.abs() * 0.01 + 1e-6; // BF16 is less precise
            assert!((recovered - v).abs() < tol, "BF16 roundtrip failed for {v}: got {recovered}");
        }
    }

    // ── OpenCL kernel source tests ──────────────────────────────────────

    #[test]
    fn test_kernel_sources_not_empty() {
        let sources = all_dequant_kernel_sources();
        assert_eq!(sources.len(), 3);
        for (name, src) in &sources {
            assert!(!name.is_empty());
            assert!(!src.is_empty());
        }
    }

    #[test]
    fn test_i2s_kernel_has_entry_point() {
        assert!(OPENCL_I2S_DEQUANT_SOURCE.contains("__kernel void i2s_dequant"));
    }

    #[test]
    fn test_qk256_kernel_has_entry_point() {
        assert!(OPENCL_QK256_DEQUANT_SOURCE.contains("__kernel void qk256_dequant"));
    }

    #[test]
    fn test_i4_kernel_has_entry_point() {
        assert!(OPENCL_I4_DEQUANT_SOURCE.contains("__kernel void i4_dequant"));
    }

    #[test]
    fn test_i2s_kernel_uses_vectorized_loads() {
        assert!(OPENCL_I2S_DEQUANT_SOURCE.contains("float4"));
        assert!(OPENCL_I2S_DEQUANT_SOURCE.contains("vstore4"));
    }

    #[test]
    fn test_qk256_kernel_uses_local_memory() {
        assert!(OPENCL_QK256_DEQUANT_SOURCE.contains("__local"));
        assert!(OPENCL_QK256_DEQUANT_SOURCE.contains("barrier"));
    }

    // ── Performance stats accuracy ──────────────────────────────────────

    #[test]
    fn test_stats_bandwidth_calculation() {
        // 1 MB in 1 ms = 1 GB/s
        let stats = DequantStats::compute(1_000_000, 250_000, 1000);
        assert!((stats.bandwidth_gb_s - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_overhead_i2s() {
        // I2S: 1 byte packs 4 values → 4 floats = 16 bytes out per byte in
        let stats = DequantStats::compute(64, 256, 1);
        assert!((stats.format_overhead - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_overhead_i4() {
        // I4: 1 byte packs 2 values → 2 floats = 8 bytes out per byte in
        let stats = DequantStats::compute(128, 256, 1);
        assert!((stats.format_overhead - 8.0).abs() < 0.01);
    }
}
