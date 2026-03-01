//! OpenCL-optimized I2_S quantized matrix-vector multiply for BitNet 1-bit inference.
#![allow(clippy::needless_range_loop)]
//!
//! This module provides CPU reference implementations and an embedded OpenCL kernel
//! for ternary-weight (I2_S packed) matrix-vector products, the core operation in
//! BitNet inference on Intel GPUs.
//!
//! # I2_S Encoding
//!
//! Four ternary values are packed into a single byte (2 bits each):
//!
//! | Bits | Value | Ternary |
//! |------|-------|---------|
//! | 0b00 |   0   |   −1   |
//! | 0b01 |   1   |    0   |
//! | 0b10 |   2   |   +1   |

use std::fmt;
use std::time::Instant;

// ── I2_S packed format ──────────────────────────────────────────────────────

/// Describes the I2_S packed format: 4 ternary values per byte, 2 bits each.
///
/// Encoding: bits `0b00 → −1`, `0b01 → 0`, `0b10 → +1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I2sPackedFormat {
    /// Number of ternary values packed per byte.
    pub values_per_byte: u8,
    /// Number of bits per ternary value.
    pub bits_per_value: u8,
}

impl Default for I2sPackedFormat {
    fn default() -> Self {
        Self { values_per_byte: 4, bits_per_value: 2 }
    }
}

impl I2sPackedFormat {
    /// Number of bytes required to store `n` ternary values.
    pub fn packed_len(n: usize) -> usize {
        n.div_ceil(4)
    }

    /// Pack a slice of ternary values (−1, 0, +1) into I2_S bytes.
    pub fn pack(values: &[i8]) -> Vec<u8> {
        let packed_len = Self::packed_len(values.len());
        let mut packed = vec![0u8; packed_len];
        for (i, &v) in values.iter().enumerate() {
            let encoded: u8 = match v {
                -1 => 0,
                0 => 1,
                1 => 2,
                _ => panic!("I2_S values must be -1, 0, or +1, got {v}"),
            };
            let byte_idx = i / 4;
            let bit_pos = (i % 4) * 2;
            packed[byte_idx] |= encoded << bit_pos;
        }
        packed
    }

    /// Unpack a single ternary value from packed bytes.
    #[inline]
    pub fn unpack_one(packed: &[u8], index: usize) -> i8 {
        let byte_idx = index / 4;
        let bit_pos = (index % 4) * 2;
        let val = (packed[byte_idx] >> bit_pos) & 0x03;
        (val as i8) - 1
    }

    /// Unpack all ternary values from packed bytes.
    pub fn unpack(packed: &[u8], count: usize) -> Vec<i8> {
        (0..count).map(|i| Self::unpack_one(packed, i)).collect()
    }
}

impl fmt::Display for I2sPackedFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I2_S({} vals/byte, {} bits/val)", self.values_per_byte, self.bits_per_value)
    }
}

// ── Scale format ────────────────────────────────────────────────────────────

/// Describes the scale format used for per-block quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum I2sScaleFormat {
    /// 32-bit floating-point scales (standard).
    Fp32,
    /// 16-bit floating-point scales (compact, used in some QK256 variants).
    Fp16,
}

impl I2sScaleFormat {
    /// Size in bytes of a single scale value.
    pub fn byte_size(self) -> usize {
        match self {
            Self::Fp32 => 4,
            Self::Fp16 => 2,
        }
    }
}

impl fmt::Display for I2sScaleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fp32 => write!(f, "FP32"),
            Self::Fp16 => write!(f, "FP16"),
        }
    }
}

// ── Block layout ────────────────────────────────────────────────────────────

/// Describes the block layout for I2_S quantization.
///
/// BitNet uses two primary flavors:
/// - **BitNet32-F16**: 32-element blocks with inline FP16 scales.
/// - **QK256**: 256-element blocks (GGML standard).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum I2sBlockLayout {
    /// BitNet32-F16: 32 elements per block, FP16 scales.
    BitNet32F16,
    /// QK256: 256 elements per block (GGML standard).
    Qk256,
    /// Custom block size.
    Custom(usize),
}

impl I2sBlockLayout {
    /// The number of elements per block.
    pub fn block_size(self) -> usize {
        match self {
            Self::BitNet32F16 => 32,
            Self::Qk256 => 256,
            Self::Custom(n) => n,
        }
    }

    /// Scale format associated with this layout.
    pub fn scale_format(self) -> I2sScaleFormat {
        match self {
            Self::BitNet32F16 => I2sScaleFormat::Fp16,
            Self::Qk256 => I2sScaleFormat::Fp32,
            Self::Custom(_) => I2sScaleFormat::Fp32,
        }
    }

    /// Number of blocks per row given column count.
    pub fn blocks_per_row(self, cols: usize) -> usize {
        cols.div_ceil(self.block_size())
    }
}

impl fmt::Display for I2sBlockLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitNet32F16 => write!(f, "BitNet32-F16 (block=32)"),
            Self::Qk256 => write!(f, "QK256 (block=256)"),
            Self::Custom(n) => write!(f, "Custom (block={n})"),
        }
    }
}

// ── Kernel variant ──────────────────────────────────────────────────────────

/// Enum of OpenCL kernel implementation variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedKernelVariant {
    /// Scalar loop per row — baseline, always correct.
    Scalar,
    /// Uses OpenCL vector types (float4/float8) for inner loop.
    Vectorized,
    /// 2D tiled dispatch, shared local memory for input tile.
    Tiled,
    /// Intel sub-group shuffle reduction (SIMD16 / SIMD32).
    SubGroup,
}

impl QuantizedKernelVariant {
    /// Preferred local work-group size for this variant.
    pub fn preferred_local_size(self) -> usize {
        match self {
            Self::Scalar => 64,
            Self::Vectorized => 128,
            Self::Tiled => 256,
            Self::SubGroup => 16, // one subgroup
        }
    }
}

impl fmt::Display for QuantizedKernelVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar => write!(f, "Scalar"),
            Self::Vectorized => write!(f, "Vectorized"),
            Self::Tiled => write!(f, "Tiled"),
            Self::SubGroup => write!(f, "SubGroup"),
        }
    }
}

// ── Quantized matvec config ─────────────────────────────────────────────────

/// Configuration for a quantized matrix-vector multiply operation.
#[derive(Debug, Clone)]
pub struct QuantizedMatVecConfig {
    /// Number of rows in the weight matrix.
    pub rows: usize,
    /// Number of columns in the weight matrix.
    pub cols: usize,
    /// Block layout (determines block_size).
    pub layout: I2sBlockLayout,
    /// Kernel variant to select.
    pub variant: QuantizedKernelVariant,
}

impl QuantizedMatVecConfig {
    /// Create a new config.
    pub fn new(rows: usize, cols: usize, layout: I2sBlockLayout) -> Self {
        Self { rows, cols, layout, variant: QuantizedKernelVariant::Scalar }
    }

    /// Number of packed bytes per row of weights.
    pub fn packed_cols(&self) -> usize {
        I2sPackedFormat::packed_len(self.cols)
    }

    /// Number of scale values per row.
    pub fn scales_per_row(&self) -> usize {
        self.layout.blocks_per_row(self.cols)
    }

    /// Total packed weight bytes.
    pub fn total_weight_bytes(&self) -> usize {
        self.rows * self.packed_cols()
    }

    /// Total scale values.
    pub fn total_scales(&self) -> usize {
        self.rows * self.scales_per_row()
    }

    /// Block size from layout.
    pub fn block_size(&self) -> usize {
        self.layout.block_size()
    }
}

// ── Optimal block config ────────────────────────────────────────────────────

/// Computes optimal block size for given matrix dimensions.
#[derive(Debug, Clone, Copy)]
pub struct OptimalBlockConfig {
    /// Recommended block size.
    pub block_size: usize,
    /// Recommended layout.
    pub layout: I2sBlockLayout,
    /// Waste ratio: fraction of elements in padding.
    pub waste_ratio: f32,
}

impl OptimalBlockConfig {
    /// Select the optimal layout for given column count.
    ///
    /// Prefers QK256 when columns are divisible by 256. Falls back to
    /// BitNet32-F16 when columns are small or indivisible by 256.
    pub fn for_cols(cols: usize) -> Self {
        if cols >= 256 && cols.is_multiple_of(256) {
            Self { block_size: 256, layout: I2sBlockLayout::Qk256, waste_ratio: 0.0 }
        } else if cols >= 32 && cols.is_multiple_of(32) {
            Self { block_size: 32, layout: I2sBlockLayout::BitNet32F16, waste_ratio: 0.0 }
        } else if cols >= 256 {
            let blocks = cols.div_ceil(256);
            let total = blocks * 256;
            let waste = (total - cols) as f32 / total as f32;
            Self { block_size: 256, layout: I2sBlockLayout::Qk256, waste_ratio: waste }
        } else {
            let blocks = cols.div_ceil(32);
            let total = blocks * 32;
            let waste = (total - cols) as f32 / total as f32;
            Self { block_size: 32, layout: I2sBlockLayout::BitNet32F16, waste_ratio: waste }
        }
    }
}

// ── Quantized matvec result ─────────────────────────────────────────────────

/// Result with timing and correctness info.
#[derive(Debug, Clone)]
pub struct QuantizedMatVecResult {
    /// Output vector.
    pub output: Vec<f32>,
    /// Wall-clock time in microseconds.
    pub elapsed_us: u64,
    /// Max absolute difference from reference (if checked).
    pub max_abs_error: Option<f32>,
}

// ── Bench reference ─────────────────────────────────────────────────────────

/// Reference timings for known configurations (for perf regression tracking).
#[derive(Debug, Clone, Copy)]
pub struct QuantizedBenchReference {
    /// Matrix dimensions.
    pub rows: usize,
    pub cols: usize,
    /// Block size.
    pub block_size: usize,
    /// Expected microseconds on Intel Arc A770 (scalar kernel).
    pub expected_us_arc_a770: u64,
}

impl QuantizedBenchReference {
    /// Standard BitNet 2B model layer: 2048 × 2048, QK256.
    pub const BITNET_2B_QK256: Self =
        Self { rows: 2048, cols: 2048, block_size: 256, expected_us_arc_a770: 250 };

    /// Standard BitNet 2B model layer: 2048 × 2048, BitNet32-F16.
    pub const BITNET_2B_B32F16: Self =
        Self { rows: 2048, cols: 2048, block_size: 32, expected_us_arc_a770: 280 };
}

// ── I2_S dequantizer ────────────────────────────────────────────────────────

/// CPU reference dequantization for validation.
///
/// Unpacks I2_S weights and scales into a full f32 matrix.
pub struct I2sDequantizer;

impl I2sDequantizer {
    /// Dequantize a packed row into f32.
    pub fn dequantize_row(
        packed_row: &[u8],
        scales: &[f32],
        cols: usize,
        block_size: usize,
    ) -> Vec<f32> {
        let blocks = cols.div_ceil(block_size);
        let mut out = vec![0.0f32; cols];
        for col in 0..cols {
            let ternary = I2sPackedFormat::unpack_one(packed_row, col);
            let block_idx = col / block_size;
            let scale = if block_idx < blocks { scales[block_idx] } else { 0.0 };
            out[col] = ternary as f32 * scale;
        }
        out
    }

    /// Dequantize the full weight matrix into f32.
    pub fn dequantize_matrix(
        weight_packed: &[u8],
        scales: &[f32],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) -> Vec<f32> {
        let packed_cols = I2sPackedFormat::packed_len(cols);
        let blocks_per_row = cols.div_ceil(block_size);
        let mut out = vec![0.0f32; rows * cols];
        for row in 0..rows {
            let row_packed = &weight_packed[row * packed_cols..(row + 1) * packed_cols];
            let row_scales = &scales[row * blocks_per_row..(row + 1) * blocks_per_row];
            let row_out = Self::dequantize_row(row_packed, row_scales, cols, block_size);
            out[row * cols..(row + 1) * cols].copy_from_slice(&row_out);
        }
        out
    }
}

// ── CPU reference: quantized matvec ─────────────────────────────────────────

/// CPU reference implementations for quantized matrix-vector multiply.
pub struct QuantizedMatVecCpu;

impl QuantizedMatVecCpu {
    /// Naive reference: I2_S quantized matrix-vector multiply.
    ///
    /// `weight_packed`: row-major packed I2_S weights `[rows × ceil(cols/4)]`
    /// `scales`: per-block scales `[rows × ceil(cols/block_size)]`
    /// `x`: input vector `[cols]`
    /// `y`: output vector `[rows]`
    pub fn matvec_ref(
        weight_packed: &[u8],
        scales: &[f32],
        x: &[f32],
        y: &mut [f32],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) {
        let packed_cols = cols.div_ceil(4);
        let blocks_per_row = cols.div_ceil(block_size);

        for row in 0..rows {
            let mut sum = 0.0f32;
            for col in 0..cols {
                let packed_idx = row * packed_cols + col / 4;
                let bit_pos = (col % 4) * 2;
                let val = (weight_packed[packed_idx] >> bit_pos) & 0x03;
                // I2_S encoding: 0→-1, 1→0, 2→+1
                let ternary = (val as i32) - 1;
                let block_idx = row * blocks_per_row + col / block_size;
                let scale = scales[block_idx];
                sum += (ternary as f32) * scale * x[col];
            }
            y[row] = sum;
        }
    }

    /// Block-optimized: accumulate using integer dot products, scale once per block.
    pub fn matvec_block_opt(
        weight_packed: &[u8],
        scales: &[f32],
        x: &[f32],
        y: &mut [f32],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) {
        let packed_cols = cols.div_ceil(4);
        let blocks_per_row = cols.div_ceil(block_size);

        for row in 0..rows {
            let mut total = 0.0f32;
            for block in 0..blocks_per_row {
                let block_start = block * block_size;
                let block_end = (block_start + block_size).min(cols);
                let scale = scales[row * blocks_per_row + block];
                let mut block_sum = 0.0f32;
                for col in block_start..block_end {
                    let packed_idx = row * packed_cols + col / 4;
                    let bit_pos = (col % 4) * 2;
                    let val = (weight_packed[packed_idx] >> bit_pos) & 0x03;
                    let ternary = (val as i32) - 1;
                    block_sum += (ternary as f32) * x[col];
                }
                total += scale * block_sum;
            }
            y[row] = total;
        }
    }

    /// Timed wrapper around `matvec_ref`.
    pub fn matvec_ref_timed(
        weight_packed: &[u8],
        scales: &[f32],
        x: &[f32],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) -> QuantizedMatVecResult {
        let mut y = vec![0.0f32; rows];
        let start = Instant::now();
        Self::matvec_ref(weight_packed, scales, x, &mut y, rows, cols, block_size);
        let elapsed_us = start.elapsed().as_micros() as u64;
        QuantizedMatVecResult { output: y, elapsed_us, max_abs_error: None }
    }

    /// Timed wrapper around `matvec_block_opt`.
    pub fn matvec_block_opt_timed(
        weight_packed: &[u8],
        scales: &[f32],
        x: &[f32],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) -> QuantizedMatVecResult {
        let mut y = vec![0.0f32; rows];
        let start = Instant::now();
        Self::matvec_block_opt(weight_packed, scales, x, &mut y, rows, cols, block_size);
        let elapsed_us = start.elapsed().as_micros() as u64;
        QuantizedMatVecResult { output: y, elapsed_us, max_abs_error: None }
    }

    /// Compare ref and block_opt outputs, returning max absolute error.
    pub fn compare_implementations(
        weight_packed: &[u8],
        scales: &[f32],
        x: &[f32],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) -> f32 {
        let ref_result = Self::matvec_ref_timed(weight_packed, scales, x, rows, cols, block_size);
        let opt_result =
            Self::matvec_block_opt_timed(weight_packed, scales, x, rows, cols, block_size);
        ref_result
            .output
            .iter()
            .zip(opt_result.output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }
}

// ── OpenCL kernel source ────────────────────────────────────────────────────

/// OpenCL kernel source for I2_S quantized matrix-vector multiply.
///
/// Scalar baseline kernel: one work-item per row. Each work-item iterates
/// over all blocks in its row, accumulates the ternary-scaled dot product,
/// and writes a single output element.
pub const QUANTIZED_MATVEC_CL: &str = r#"
__kernel void quantized_matvec_i2s(
    __global const uchar* weights,  // packed I2_S weights
    __global const float* scales,   // per-block scales
    __global const float* x,        // input vector
    __global float* y,              // output vector
    const int rows,
    const int cols,
    const int block_size)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    int packed_cols = (cols + 3) / 4;
    int blocks_per_row = cols.div_ceil(block_size);

    float sum = 0.0f;
    for (int block = 0; block < blocks_per_row; block++) {
        int block_start = block * block_size;
        int block_end = min(block_start + block_size, cols);
        float scale = scales[row * blocks_per_row + block];
        float block_sum = 0.0f;
        for (int col = block_start; col < block_end; col++) {
            int packed_idx = row * packed_cols + col / 4;
            int bit_pos = (col % 4) * 2;
            int val = (weights[packed_idx] >> bit_pos) & 0x03;
            float ternary = (float)(val - 1);
            block_sum += ternary * x[col];
        }
        sum += scale * block_sum;
    }
    y[row] = sum;
}
"#;

/// OpenCL kernel source for sub-group optimized I2_S matvec.
///
/// Uses Intel sub-group operations to reduce across SIMD lanes.
pub const QUANTIZED_MATVEC_SUBGROUP_CL: &str = r#"
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

__kernel void quantized_matvec_i2s_subgroup(
    __global const uchar* weights,
    __global const float* scales,
    __global const float* x,
    __global float* y,
    const int rows,
    const int cols,
    const int block_size)
{
    int row = get_group_id(0);
    if (row >= rows) return;

    int lid = get_sub_group_local_id();
    int sg_size = get_sub_group_size();
    int packed_cols = (cols + 3) / 4;
    int blocks_per_row = cols.div_ceil(block_size);

    float partial = 0.0f;
    for (int block = 0; block < blocks_per_row; block++) {
        int block_start = block * block_size;
        int block_end = min(block_start + block_size, cols);
        float scale = scales[row * blocks_per_row + block];
        float block_partial = 0.0f;
        for (int col = block_start + lid; col < block_end; col += sg_size) {
            int packed_idx = row * packed_cols + col / 4;
            int bit_pos = (col % 4) * 2;
            int val = (weights[packed_idx] >> bit_pos) & 0x03;
            float ternary = (float)(val - 1);
            block_partial += ternary * x[col];
        }
        partial += scale * block_partial;
    }

    float reduced = sub_group_reduce_add(partial);
    if (lid == 0) {
        y[row] = reduced;
    }
}
"#;

// ── Helper: build test data ─────────────────────────────────────────────────

/// Build packed weights and scales from a ternary weight matrix.
///
/// `weights_ternary` is row-major `[rows × cols]` with values −1, 0, +1.
/// Returns `(packed_weights, scales)`.
pub fn build_test_data(
    weights_ternary: &[i8],
    rows: usize,
    cols: usize,
    block_size: usize,
    scale_value: f32,
) -> (Vec<u8>, Vec<f32>) {
    let packed_cols = I2sPackedFormat::packed_len(cols);
    let blocks_per_row = cols.div_ceil(block_size);

    let mut packed = vec![0u8; rows * packed_cols];
    for row in 0..rows {
        for col in 0..cols {
            let v = weights_ternary[row * cols + col];
            let encoded: u8 = match v {
                -1 => 0,
                0 => 1,
                1 => 2,
                _ => panic!("I2_S values must be -1, 0, or +1"),
            };
            let byte_idx = row * packed_cols + col / 4;
            let bit_pos = (col % 4) * 2;
            packed[byte_idx] |= encoded << bit_pos;
        }
    }

    let scales = vec![scale_value; rows * blocks_per_row];
    (packed, scales)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Packing format tests ────────────────────────────────────────────

    #[test]
    fn test_packed_format_defaults() {
        let fmt = I2sPackedFormat::default();
        assert_eq!(fmt.values_per_byte, 4);
        assert_eq!(fmt.bits_per_value, 2);
    }

    #[test]
    fn test_packed_len() {
        assert_eq!(I2sPackedFormat::packed_len(1), 1);
        assert_eq!(I2sPackedFormat::packed_len(4), 1);
        assert_eq!(I2sPackedFormat::packed_len(5), 2);
        assert_eq!(I2sPackedFormat::packed_len(8), 2);
        assert_eq!(I2sPackedFormat::packed_len(256), 64);
    }

    #[test]
    fn test_pack_known_pattern_all_neg1() {
        // -1, -1, -1, -1 → encoded as 0,0,0,0 → byte = 0x00
        let packed = I2sPackedFormat::pack(&[-1, -1, -1, -1]);
        assert_eq!(packed, vec![0x00]);
    }

    #[test]
    fn test_pack_known_pattern_all_zero() {
        // 0, 0, 0, 0 → encoded as 1,1,1,1 → byte = 0b01_01_01_01 = 0x55
        let packed = I2sPackedFormat::pack(&[0, 0, 0, 0]);
        assert_eq!(packed, vec![0x55]);
    }

    #[test]
    fn test_pack_known_pattern_all_pos1() {
        // +1, +1, +1, +1 → encoded as 2,2,2,2 → byte = 0b10_10_10_10 = 0xAA
        let packed = I2sPackedFormat::pack(&[1, 1, 1, 1]);
        assert_eq!(packed, vec![0xAA]);
    }

    #[test]
    fn test_pack_mixed_pattern() {
        // -1, 0, +1, 0 → encoded as 0, 1, 2, 1 → 0b01_10_01_00 = 0x64
        let packed = I2sPackedFormat::pack(&[-1, 0, 1, 0]);
        let expected = 0b_01_10_01_00u8; // bit positions: [1:0]=00, [3:2]=01, [5:4]=10, [7:6]=01
        assert_eq!(packed, vec![expected]);
    }

    #[test]
    fn test_pack_unpack_roundtrip_4() {
        let values: Vec<i8> = vec![-1, 0, 1, 0];
        let packed = I2sPackedFormat::pack(&values);
        let unpacked = I2sPackedFormat::unpack(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_pack_unpack_roundtrip_7() {
        // Non-multiple of 4
        let values: Vec<i8> = vec![-1, 1, 0, -1, 1, 0, 1];
        let packed = I2sPackedFormat::pack(&values);
        let unpacked = I2sPackedFormat::unpack(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_pack_unpack_roundtrip_large() {
        let values: Vec<i8> = (0..256).map(|i| (i % 3) as i8 - 1).collect();
        let packed = I2sPackedFormat::pack(&values);
        let unpacked = I2sPackedFormat::unpack(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_unpack_one_positions() {
        // Pack [-1, 0, +1, -1] → unpack individual positions
        let packed = I2sPackedFormat::pack(&[-1, 0, 1, -1]);
        assert_eq!(I2sPackedFormat::unpack_one(&packed, 0), -1);
        assert_eq!(I2sPackedFormat::unpack_one(&packed, 1), 0);
        assert_eq!(I2sPackedFormat::unpack_one(&packed, 2), 1);
        assert_eq!(I2sPackedFormat::unpack_one(&packed, 3), -1);
    }

    #[test]
    #[should_panic(expected = "I2_S values must be -1, 0, or +1")]
    fn test_pack_invalid_value() {
        I2sPackedFormat::pack(&[2]);
    }

    #[test]
    fn test_packed_format_display() {
        let fmt = I2sPackedFormat::default();
        let s = fmt.to_string();
        assert!(s.contains("I2_S"));
        assert!(s.contains("4 vals/byte"));
    }

    // ── Scale format tests ──────────────────────────────────────────────

    #[test]
    fn test_scale_format_byte_sizes() {
        assert_eq!(I2sScaleFormat::Fp32.byte_size(), 4);
        assert_eq!(I2sScaleFormat::Fp16.byte_size(), 2);
    }

    #[test]
    fn test_scale_format_display() {
        assert_eq!(I2sScaleFormat::Fp32.to_string(), "FP32");
        assert_eq!(I2sScaleFormat::Fp16.to_string(), "FP16");
    }

    // ── Block layout tests ──────────────────────────────────────────────

    #[test]
    fn test_block_layout_sizes() {
        assert_eq!(I2sBlockLayout::BitNet32F16.block_size(), 32);
        assert_eq!(I2sBlockLayout::Qk256.block_size(), 256);
        assert_eq!(I2sBlockLayout::Custom(128).block_size(), 128);
    }

    #[test]
    fn test_block_layout_scale_format() {
        assert_eq!(I2sBlockLayout::BitNet32F16.scale_format(), I2sScaleFormat::Fp16);
        assert_eq!(I2sBlockLayout::Qk256.scale_format(), I2sScaleFormat::Fp32);
    }

    #[test]
    fn test_blocks_per_row_exact() {
        assert_eq!(I2sBlockLayout::Qk256.blocks_per_row(256), 1);
        assert_eq!(I2sBlockLayout::Qk256.blocks_per_row(512), 2);
        assert_eq!(I2sBlockLayout::BitNet32F16.blocks_per_row(32), 1);
        assert_eq!(I2sBlockLayout::BitNet32F16.blocks_per_row(64), 2);
    }

    #[test]
    fn test_blocks_per_row_partial() {
        assert_eq!(I2sBlockLayout::Qk256.blocks_per_row(257), 2);
        assert_eq!(I2sBlockLayout::BitNet32F16.blocks_per_row(33), 2);
    }

    #[test]
    fn test_block_layout_display() {
        assert!(I2sBlockLayout::Qk256.to_string().contains("QK256"));
        assert!(I2sBlockLayout::BitNet32F16.to_string().contains("BitNet32"));
    }

    // ── Kernel variant tests ────────────────────────────────────────────

    #[test]
    fn test_kernel_variant_local_sizes() {
        assert_eq!(QuantizedKernelVariant::Scalar.preferred_local_size(), 64);
        assert_eq!(QuantizedKernelVariant::SubGroup.preferred_local_size(), 16);
    }

    #[test]
    fn test_kernel_variant_display() {
        assert_eq!(QuantizedKernelVariant::Scalar.to_string(), "Scalar");
        assert_eq!(QuantizedKernelVariant::Tiled.to_string(), "Tiled");
    }

    // ── Config tests ────────────────────────────────────────────────────

    #[test]
    fn test_config_packed_cols() {
        let cfg = QuantizedMatVecConfig::new(4, 8, I2sBlockLayout::Qk256);
        assert_eq!(cfg.packed_cols(), 2);
    }

    #[test]
    fn test_config_scales_per_row() {
        let cfg = QuantizedMatVecConfig::new(4, 256, I2sBlockLayout::Qk256);
        assert_eq!(cfg.scales_per_row(), 1);
        let cfg2 = QuantizedMatVecConfig::new(4, 256, I2sBlockLayout::BitNet32F16);
        assert_eq!(cfg2.scales_per_row(), 8);
    }

    #[test]
    fn test_config_total_weight_bytes() {
        let cfg = QuantizedMatVecConfig::new(4, 8, I2sBlockLayout::Qk256);
        assert_eq!(cfg.total_weight_bytes(), 4 * 2); // 4 rows × 2 bytes each
    }

    #[test]
    fn test_config_total_scales() {
        let cfg = QuantizedMatVecConfig::new(4, 256, I2sBlockLayout::Qk256);
        assert_eq!(cfg.total_scales(), 4); // 4 rows × 1 block each
    }

    // ── Optimal block config tests ──────────────────────────────────────

    #[test]
    fn test_optimal_qk256_aligned() {
        let opt = OptimalBlockConfig::for_cols(2048);
        assert_eq!(opt.block_size, 256);
        assert_eq!(opt.layout, I2sBlockLayout::Qk256);
        assert_eq!(opt.waste_ratio, 0.0);
    }

    #[test]
    fn test_optimal_b32_aligned() {
        let opt = OptimalBlockConfig::for_cols(96);
        assert_eq!(opt.block_size, 32);
        assert_eq!(opt.layout, I2sBlockLayout::BitNet32F16);
        assert_eq!(opt.waste_ratio, 0.0);
    }

    #[test]
    fn test_optimal_qk256_unaligned() {
        let opt = OptimalBlockConfig::for_cols(300);
        assert_eq!(opt.block_size, 256);
        assert!(opt.waste_ratio > 0.0);
        assert!(opt.waste_ratio < 1.0);
    }

    #[test]
    fn test_optimal_small_cols() {
        let opt = OptimalBlockConfig::for_cols(10);
        assert_eq!(opt.block_size, 32);
        assert!(opt.waste_ratio > 0.0);
    }

    // ── Dequantizer tests ───────────────────────────────────────────────

    #[test]
    fn test_dequant_all_zero_weights() {
        // All-zero weights (encoded as 1) → dequantized to 0.0
        let packed = I2sPackedFormat::pack(&[0, 0, 0, 0]);
        let scales = vec![1.0f32];
        let row = I2sDequantizer::dequantize_row(&packed, &scales, 4, 32);
        assert_eq!(row, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dequant_all_pos1_unit_scale() {
        let packed = I2sPackedFormat::pack(&[1, 1, 1, 1]);
        let scales = vec![1.0f32];
        let row = I2sDequantizer::dequantize_row(&packed, &scales, 4, 32);
        assert_eq!(row, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_dequant_all_neg1_unit_scale() {
        let packed = I2sPackedFormat::pack(&[-1, -1, -1, -1]);
        let scales = vec![1.0f32];
        let row = I2sDequantizer::dequantize_row(&packed, &scales, 4, 32);
        assert_eq!(row, vec![-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_dequant_with_scale() {
        let packed = I2sPackedFormat::pack(&[-1, 0, 1, -1]);
        let scales = vec![2.5f32];
        let row = I2sDequantizer::dequantize_row(&packed, &scales, 4, 32);
        assert_eq!(row, vec![-2.5, 0.0, 2.5, -2.5]);
    }

    #[test]
    fn test_dequant_matrix() {
        // 2×4 matrix: row0 = [1,0,-1,0], row1 = [-1,1,0,1]
        let weights: Vec<i8> = vec![1, 0, -1, 0, -1, 1, 0, 1];
        let (packed, scales) = build_test_data(&weights, 2, 4, 32, 1.0);
        let mat = I2sDequantizer::dequantize_matrix(&packed, &scales, 2, 4, 32);
        assert_eq!(mat[0..4], [1.0, 0.0, -1.0, 0.0]);
        assert_eq!(mat[4..8], [-1.0, 1.0, 0.0, 1.0]);
    }

    // ── MatVec: small known matrices ────────────────────────────────────

    #[test]
    fn test_matvec_2x2_identity_like() {
        // W = [[1, 0], [0, 1]], x = [3.0, 5.0] → y = [3.0, 5.0]
        let weights: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = build_test_data(&weights, 2, 2, 32, 1.0);
        let x = vec![3.0f32, 5.0];
        let mut y = vec![0.0f32; 2];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 2, 2, 32);
        assert!((y[0] - 3.0).abs() < 1e-6);
        assert!((y[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_2x2_all_ones() {
        // W = [[1, 1], [1, 1]], x = [2.0, 3.0] → y = [5.0, 5.0]
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let (packed, scales) = build_test_data(&weights, 2, 2, 32, 1.0);
        let x = vec![2.0f32, 3.0];
        let mut y = vec![0.0f32; 2];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 2, 2, 32);
        assert!((y[0] - 5.0).abs() < 1e-6);
        assert!((y[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_2x2_neg() {
        // W = [[-1, -1], [-1, -1]], x = [2.0, 3.0] → y = [-5.0, -5.0]
        let weights: Vec<i8> = vec![-1, -1, -1, -1];
        let (packed, scales) = build_test_data(&weights, 2, 2, 32, 1.0);
        let x = vec![2.0f32, 3.0];
        let mut y = vec![0.0f32; 2];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 2, 2, 32);
        assert!((y[0] + 5.0).abs() < 1e-6);
        assert!((y[1] + 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_4x4_known() {
        // 4×4 identity
        #[rustfmt::skip]
        let weights: Vec<i8> = vec![
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ];
        let (packed, scales) = build_test_data(&weights, 4, 4, 32, 1.0);
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 4];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 4, 4, 32);
        for i in 0..4 {
            assert!((y[i] - x[i]).abs() < 1e-6, "y[{i}] = {} expected {}", y[i], x[i]);
        }
    }

    #[test]
    fn test_matvec_all_ones_vector_gives_row_sums() {
        // W = [[1, -1, 0], [-1, 1, 1]], x = [1,1,1] → y = [0, 1]
        let weights: Vec<i8> = vec![1, -1, 0, -1, 1, 1];
        let (packed, scales) = build_test_data(&weights, 2, 3, 32, 1.0);
        let x = vec![1.0f32; 3];
        let mut y = vec![0.0f32; 2];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 2, 3, 32);
        assert!((y[0] - 0.0).abs() < 1e-6);
        assert!((y[1] - 1.0).abs() < 1e-6);
    }

    // ── Ref vs block-opt agreement ──────────────────────────────────────

    #[test]
    fn test_ref_vs_opt_small() {
        let weights: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 1, -1];
        let (packed, scales) = build_test_data(&weights, 2, 4, 32, 1.5);
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let max_err = QuantizedMatVecCpu::compare_implementations(&packed, &scales, &x, 2, 4, 32);
        assert!(max_err < 1e-5, "max error = {max_err}");
    }

    #[test]
    fn test_ref_vs_opt_medium() {
        let rows = 16;
        let cols = 64;
        let weights: Vec<i8> = (0..rows * cols).map(|i| (i % 3) as i8 - 1).collect();
        let (packed, scales) = build_test_data(&weights, rows, cols, 32, 0.5);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1).collect();
        let max_err =
            QuantizedMatVecCpu::compare_implementations(&packed, &scales, &x, rows, cols, 32);
        assert!(max_err < 1e-4, "max error = {max_err}");
    }

    #[test]
    fn test_ref_vs_opt_block_size_32() {
        let rows = 8;
        let cols = 64;
        let weights: Vec<i8> = (0..rows * cols).map(|i| (i % 3) as i8 - 1).collect();
        let (packed, scales) = build_test_data(&weights, rows, cols, 32, 1.0);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();
        let max_err =
            QuantizedMatVecCpu::compare_implementations(&packed, &scales, &x, rows, cols, 32);
        assert!(max_err < 1e-4, "max error = {max_err}");
    }

    #[test]
    fn test_ref_vs_opt_block_size_256() {
        let rows = 4;
        let cols = 256;
        let weights: Vec<i8> = (0..rows * cols).map(|i| (i % 3) as i8 - 1).collect();
        let (packed, scales) = build_test_data(&weights, rows, cols, 256, 2.0);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();
        let max_err =
            QuantizedMatVecCpu::compare_implementations(&packed, &scales, &x, rows, cols, 256);
        assert!(max_err < 1e-3, "max error = {max_err}");
    }

    // ── Block size 32 vs 256 give same results (for uniform scale) ──────

    #[test]
    fn test_block_32_vs_256_uniform_scale() {
        let rows = 4;
        let cols = 256;
        let weights: Vec<i8> = (0..rows * cols).map(|i| (i % 3) as i8 - 1).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();

        let (packed32, scales32) = build_test_data(&weights, rows, cols, 32, 1.0);
        let (packed256, scales256) = build_test_data(&weights, rows, cols, 256, 1.0);

        let mut y32 = vec![0.0f32; rows];
        let mut y256 = vec![0.0f32; rows];
        QuantizedMatVecCpu::matvec_ref(&packed32, &scales32, &x, &mut y32, rows, cols, 32);
        QuantizedMatVecCpu::matvec_ref(&packed256, &scales256, &x, &mut y256, rows, cols, 256);

        for i in 0..rows {
            assert!((y32[i] - y256[i]).abs() < 1e-3, "row {i}: y32={}, y256={}", y32[i], y256[i]);
        }
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_matvec_single_row() {
        let weights: Vec<i8> = vec![1, -1, 1, -1];
        let (packed, scales) = build_test_data(&weights, 1, 4, 32, 1.0);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 1];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 1, 4, 32);
        // 1*1 + (-1)*2 + 1*3 + (-1)*4 = 1 - 2 + 3 - 4 = -2
        assert!((y[0] + 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_single_col() {
        let weights: Vec<i8> = vec![1, -1, 0];
        let (packed, scales) = build_test_data(&weights, 3, 1, 32, 1.0);
        let x = vec![5.0f32];
        let mut y = vec![0.0f32; 3];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 3, 1, 32);
        assert!((y[0] - 5.0).abs() < 1e-6);
        assert!((y[1] + 5.0).abs() < 1e-6);
        assert!((y[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_cols_not_div_by_4() {
        // 2×3 matrix
        let weights: Vec<i8> = vec![1, 0, -1, -1, 1, 0];
        let (packed, scales) = build_test_data(&weights, 2, 3, 32, 1.0);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0f32; 2];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 2, 3, 32);
        // row0: 1*1 + 0*2 + (-1)*3 = -2
        // row1: (-1)*1 + 1*2 + 0*3 = 1
        assert!((y[0] + 2.0).abs() < 1e-6);
        assert!((y[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_cols_5() {
        let weights: Vec<i8> = vec![1, 1, 1, 1, 1];
        let (packed, scales) = build_test_data(&weights, 1, 5, 32, 1.0);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y = vec![0.0f32; 1];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 1, 5, 32);
        assert!((y[0] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_cols_1() {
        let weights: Vec<i8> = vec![1];
        let (packed, scales) = build_test_data(&weights, 1, 1, 32, 2.0);
        let x = vec![7.0f32];
        let mut y = vec![0.0f32; 1];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 1, 1, 32);
        assert!((y[0] - 14.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_zero_weights() {
        let weights: Vec<i8> = vec![0; 16];
        let (packed, scales) = build_test_data(&weights, 4, 4, 32, 1.0);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 4];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 4, 4, 32);
        for val in &y {
            assert!((val - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matvec_zero_input() {
        let weights: Vec<i8> = vec![1, -1, 0, 1];
        let (packed, scales) = build_test_data(&weights, 1, 4, 32, 1.0);
        let x = vec![0.0f32; 4];
        let mut y = vec![0.0f32; 1];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 1, 4, 32);
        assert!((y[0]).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_zero_scale() {
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let packed = I2sPackedFormat::pack(&weights);
        let scales = vec![0.0f32];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 1];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 1, 4, 32);
        assert!((y[0]).abs() < 1e-6);
    }

    // ── Large matrix sanity ─────────────────────────────────────────────

    #[test]
    fn test_matvec_large_2048x2048_finite() {
        let rows = 2048;
        let cols = 2048;
        let weights: Vec<i8> = (0..rows * cols).map(|i| (i % 3) as i8 - 1).collect();
        let (packed, scales) = build_test_data(&weights, rows, cols, 256, 0.01);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.001).collect();
        let mut y = vec![0.0f32; rows];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, rows, cols, 256);
        for (i, val) in y.iter().enumerate() {
            assert!(val.is_finite(), "y[{i}] = {val} is not finite");
        }
    }

    #[test]
    fn test_matvec_large_ref_vs_opt_agreement() {
        let rows = 512;
        let cols = 512;
        let weights: Vec<i8> = (0..rows * cols).map(|i| (i % 3) as i8 - 1).collect();
        let (packed, scales) = build_test_data(&weights, rows, cols, 256, 1.0);
        let x: Vec<f32> = (0..cols).map(|i| ((i * 7 + 13) % 100) as f32 * 0.01).collect();
        let max_err =
            QuantizedMatVecCpu::compare_implementations(&packed, &scales, &x, rows, cols, 256);
        assert!(max_err < 1e-2, "max error = {max_err}");
    }

    // ── QK256-specific tests ────────────────────────────────────────────

    #[test]
    fn test_qk256_single_block_row() {
        let cols = 256;
        let weights: Vec<i8> = (0..cols).map(|i| (i % 3) as i8 - 1).collect();
        let (packed, scales) = build_test_data(&weights, 1, cols, 256, 1.0);
        let x = vec![1.0f32; cols];
        let mut y_ref = vec![0.0f32; 1];
        let mut y_opt = vec![0.0f32; 1];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y_ref, 1, cols, 256);
        QuantizedMatVecCpu::matvec_block_opt(&packed, &scales, &x, &mut y_opt, 1, cols, 256);
        assert!((y_ref[0] - y_opt[0]).abs() < 1e-4);
    }

    #[test]
    fn test_qk256_multi_block_row() {
        let cols = 512;
        let weights: Vec<i8> = (0..cols).map(|i| (i % 3) as i8 - 1).collect();
        let (packed, scales) = build_test_data(&weights, 1, cols, 256, 0.5);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();
        let mut y_ref = vec![0.0f32; 1];
        let mut y_opt = vec![0.0f32; 1];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y_ref, 1, cols, 256);
        QuantizedMatVecCpu::matvec_block_opt(&packed, &scales, &x, &mut y_opt, 1, cols, 256);
        assert!((y_ref[0] - y_opt[0]).abs() < 1e-3);
    }

    #[test]
    fn test_qk256_layout_properties() {
        let layout = I2sBlockLayout::Qk256;
        assert_eq!(layout.block_size(), 256);
        assert_eq!(layout.scale_format(), I2sScaleFormat::Fp32);
        assert_eq!(layout.blocks_per_row(2048), 8);
    }

    // ── BitNet32-F16 specific tests ─────────────────────────────────────

    #[test]
    fn test_bitnet32_single_block() {
        let cols = 32;
        let weights: Vec<i8> = (0..cols).map(|i| (i % 3) as i8 - 1).collect();
        let (packed, scales) = build_test_data(&weights, 1, cols, 32, 1.0);
        let x = vec![1.0f32; cols];
        let mut y = vec![0.0f32; 1];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 1, cols, 32);
        // Known sum: 11×(-1) + 10×0 + 11×1 = 0
        // Indices 0..32 mod 3: pattern -1,0,1 repeats ⌊32/3⌋=10 full + 2 remain(-1,0)
        // So: 11×(-1) + 11×0 + 10×(+1) = -1
        assert!((y[0] + 1.0).abs() < 1e-6, "y[0] = {}", y[0]);
    }

    #[test]
    fn test_bitnet32_multi_block() {
        let cols = 64;
        let weights: Vec<i8> = vec![1; cols];
        let (packed, scales) = build_test_data(&weights, 1, cols, 32, 0.5);
        let x = vec![1.0f32; cols];
        let mut y = vec![0.0f32; 1];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y, 1, cols, 32);
        // All +1 × scale 0.5 × input 1.0 → 64 × 0.5 = 32.0
        assert!((y[0] - 32.0).abs() < 1e-4, "y[0] = {}", y[0]);
    }

    #[test]
    fn test_bitnet32_layout_properties() {
        let layout = I2sBlockLayout::BitNet32F16;
        assert_eq!(layout.block_size(), 32);
        assert_eq!(layout.scale_format(), I2sScaleFormat::Fp16);
        assert_eq!(layout.blocks_per_row(2048), 64);
    }

    // ── Timed wrappers ──────────────────────────────────────────────────

    #[test]
    fn test_timed_ref_returns_valid_result() {
        let weights: Vec<i8> = vec![1, 0, -1, 0];
        let (packed, scales) = build_test_data(&weights, 1, 4, 32, 1.0);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let result = QuantizedMatVecCpu::matvec_ref_timed(&packed, &scales, &x, 1, 4, 32);
        assert_eq!(result.output.len(), 1);
        assert!(result.max_abs_error.is_none());
    }

    #[test]
    fn test_timed_opt_returns_valid_result() {
        let weights: Vec<i8> = vec![1, 0, -1, 0];
        let (packed, scales) = build_test_data(&weights, 1, 4, 32, 1.0);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let result = QuantizedMatVecCpu::matvec_block_opt_timed(&packed, &scales, &x, 1, 4, 32);
        assert_eq!(result.output.len(), 1);
        assert!(result.max_abs_error.is_none());
    }

    // ── OpenCL kernel source tests ──────────────────────────────────────

    #[test]
    fn test_cl_source_contains_kernel_name() {
        assert!(QUANTIZED_MATVEC_CL.contains("quantized_matvec_i2s"));
    }

    #[test]
    fn test_cl_source_contains_kernel_args() {
        assert!(QUANTIZED_MATVEC_CL.contains("__global const uchar* weights"));
        assert!(QUANTIZED_MATVEC_CL.contains("__global const float* scales"));
        assert!(QUANTIZED_MATVEC_CL.contains("__global const float* x"));
        assert!(QUANTIZED_MATVEC_CL.contains("__global float* y"));
    }

    #[test]
    fn test_cl_source_contains_ternary_decode() {
        assert!(QUANTIZED_MATVEC_CL.contains("val - 1"));
    }

    #[test]
    fn test_cl_source_contains_block_loop() {
        assert!(QUANTIZED_MATVEC_CL.contains("blocks_per_row"));
        assert!(QUANTIZED_MATVEC_CL.contains("block_size"));
    }

    #[test]
    fn test_cl_subgroup_source_contains_kernel_name() {
        assert!(QUANTIZED_MATVEC_SUBGROUP_CL.contains("quantized_matvec_i2s_subgroup"));
    }

    #[test]
    fn test_cl_subgroup_source_contains_extensions() {
        assert!(QUANTIZED_MATVEC_SUBGROUP_CL.contains("cl_intel_subgroups"));
        assert!(QUANTIZED_MATVEC_SUBGROUP_CL.contains("sub_group_reduce_add"));
    }

    // ── Build test data helper ──────────────────────────────────────────

    #[test]
    fn test_build_test_data_dimensions() {
        let weights: Vec<i8> = vec![1, 0, -1, 0, 1, -1, 0, 1];
        let (packed, scales) = build_test_data(&weights, 2, 4, 32, 1.0);
        assert_eq!(packed.len(), 2); // 2 rows × 1 byte/row
        assert_eq!(scales.len(), 2); // 2 rows × 1 block/row
    }

    #[test]
    fn test_build_test_data_scale_value() {
        let weights: Vec<i8> = vec![0; 4];
        let (_, scales) = build_test_data(&weights, 1, 4, 32, 3.14);
        assert!((scales[0] - 3.14).abs() < 1e-6);
    }

    // ── Bench reference ─────────────────────────────────────────────────

    #[test]
    fn test_bench_reference_constants() {
        assert_eq!(QuantizedBenchReference::BITNET_2B_QK256.rows, 2048);
        assert_eq!(QuantizedBenchReference::BITNET_2B_QK256.cols, 2048);
        assert_eq!(QuantizedBenchReference::BITNET_2B_QK256.block_size, 256);
        assert_eq!(QuantizedBenchReference::BITNET_2B_B32F16.block_size, 32);
    }

    // ── Scale affects output ────────────────────────────────────────────

    #[test]
    fn test_scale_doubles_output() {
        let weights: Vec<i8> = vec![1, 1, 1, 1];
        let x = vec![1.0f32; 4];

        let (packed1, scales1) = build_test_data(&weights, 1, 4, 32, 1.0);
        let (packed2, scales2) = build_test_data(&weights, 1, 4, 32, 2.0);

        let mut y1 = vec![0.0f32; 1];
        let mut y2 = vec![0.0f32; 1];
        QuantizedMatVecCpu::matvec_ref(&packed1, &scales1, &x, &mut y1, 1, 4, 32);
        QuantizedMatVecCpu::matvec_ref(&packed2, &scales2, &x, &mut y2, 1, 4, 32);
        assert!((y2[0] - 2.0 * y1[0]).abs() < 1e-6);
    }

    // ── Opt on edge cols ────────────────────────────────────────────────

    #[test]
    fn test_block_opt_cols_not_div_by_4() {
        let weights: Vec<i8> = vec![1, 0, -1, -1, 1, 0];
        let (packed, scales) = build_test_data(&weights, 2, 3, 32, 1.0);
        let x = vec![1.0, 2.0, 3.0];
        let mut y_ref = vec![0.0f32; 2];
        let mut y_opt = vec![0.0f32; 2];
        QuantizedMatVecCpu::matvec_ref(&packed, &scales, &x, &mut y_ref, 2, 3, 32);
        QuantizedMatVecCpu::matvec_block_opt(&packed, &scales, &x, &mut y_opt, 2, 3, 32);
        for i in 0..2 {
            assert!(
                (y_ref[i] - y_opt[i]).abs() < 1e-6,
                "row {i}: ref={} opt={}",
                y_ref[i],
                y_opt[i]
            );
        }
    }

    #[test]
    fn test_block_opt_single_col() {
        let weights: Vec<i8> = vec![1];
        let (packed, scales) = build_test_data(&weights, 1, 1, 32, 3.0);
        let x = vec![2.0f32];
        let mut y = vec![0.0f32; 1];
        QuantizedMatVecCpu::matvec_block_opt(&packed, &scales, &x, &mut y, 1, 1, 32);
        assert!((y[0] - 6.0).abs() < 1e-6);
    }
}
