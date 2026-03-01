//! Weight preprocessing pipeline for GPU-optimized packing (Intel Arc A770 / OpenCL).
//!
//! Converts model weights (I2_S ternary or QK256) into GPU-friendly packed formats
//! for efficient OpenCL kernel consumption.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Supported weight storage formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    RawF32,
    TernaryI2S,
    QK256Packed,
    Int8Packed,
    GpuTernaryPacked,
    GpuInt4Packed,
}

impl fmt::Display for WeightFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RawF32 => write!(f, "Raw_F32"),
            Self::TernaryI2S => write!(f, "Ternary_I2S"),
            Self::QK256Packed => write!(f, "QK256_Packed"),
            Self::Int8Packed => write!(f, "Int8_Packed"),
            Self::GpuTernaryPacked => write!(f, "Gpu_Ternary_Packed"),
            Self::GpuInt4Packed => write!(f, "Gpu_Int4_Packed"),
        }
    }
}

/// Configuration for weight packing.
#[derive(Debug, Clone)]
pub struct PackingConfig {
    pub target_format: WeightFormat,
    pub tile_size: usize,
    pub alignment: usize,
    pub pack_transpose: bool,
}

impl Default for PackingConfig {
    fn default() -> Self {
        Self { target_format: WeightFormat::GpuTernaryPacked, tile_size: 16, alignment: 64, pack_transpose: false }
    }
}

/// An unprocessed weight tensor.
#[derive(Debug, Clone)]
pub struct WeightTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub format: WeightFormat,
    pub byte_stride: usize,
}

/// Packed weight tensor ready for GPU upload.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedWeights {
    pub packed_data: Vec<u8>,
    pub original_shape: Vec<usize>,
    pub packed_shape: Vec<usize>,
    pub format: WeightFormat,
    pub scale_factors: Option<Vec<f32>>,
    pub metadata: WeightMetadata,
}

/// Metadata about a weight packing operation.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightMetadata {
    pub original_format: WeightFormat,
    pub packed_format: WeightFormat,
    pub compression_ratio: f32,
    pub pack_time_us: u64,
    pub num_elements: usize,
}

/// A single stage in the preprocessing pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreprocessStage {
    Transpose,
    Quantize,
    Pack,
    PadAlignment(usize),
    TileReorder(usize, usize),
}

/// A complete preprocessing pipeline.
#[derive(Debug, Clone)]
pub struct PreprocessPipeline {
    pub stages: Vec<PreprocessStage>,
    pub config: PackingConfig,
}

/// Errors that can occur during weight preprocessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreprocessError {
    InvalidFormat,
    ShapeMismatch,
    PackingFailed(String),
    UnsupportedConversion { from: WeightFormat, to: WeightFormat },
}

impl fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat => write!(f, "invalid weight format"),
            Self::ShapeMismatch => write!(f, "weight shape mismatch"),
            Self::PackingFailed(msg) => write!(f, "packing failed: {msg}"),
            Self::UnsupportedConversion { from, to } => {
                write!(f, "unsupported conversion: {from} -> {to}")
            }
        }
    }
}

impl std::error::Error for PreprocessError {}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Pack ternary weights ({-1,0,1} as `i8`) into 2-bit representation.
///
/// Encoding per value: -1→0b10, 0→0b00, 1→0b01.
/// Four values per byte, LSB-first.
pub fn cpu_pack_ternary_weights(weights: &[i8], rows: usize, cols: usize) -> PackedWeights {
    let start = Instant::now();
    let num_elements = rows * cols;
    assert!(weights.len() >= num_elements, "weights slice too short");

    let packed_len = num_elements.div_ceil(4);
    let mut packed = vec![0u8; packed_len];

    for (i, &w) in weights[..num_elements].iter().enumerate() {
        let bits: u8 = match w {
            -1 => 0b10,
            0 => 0b00,
            1 => 0b01,
            _ => 0b00, // clamp unexpected values
        };
        let byte_idx = i / 4;
        let shift = (i % 4) * 2;
        packed[byte_idx] |= bits << shift;
    }

    let elapsed = start.elapsed().as_micros() as u64;
    let original_bytes = num_elements; // 1 byte per i8
    let compression = if packed_len > 0 { original_bytes as f32 / packed_len as f32 } else { 1.0 };

    PackedWeights {
        packed_data: packed,
        original_shape: vec![rows, cols],
        packed_shape: vec![rows, cols.div_ceil(4)],
        format: WeightFormat::GpuTernaryPacked,
        scale_factors: None,
        metadata: WeightMetadata {
            original_format: WeightFormat::TernaryI2S,
            packed_format: WeightFormat::GpuTernaryPacked,
            compression_ratio: compression,
            pack_time_us: elapsed,
            num_elements,
        },
    }
}

/// Unpack 2-bit ternary weights back to `i8` {-1, 0, 1}.
pub fn cpu_unpack_ternary_weights(packed: &PackedWeights) -> Vec<i8> {
    let num_elements = packed.metadata.num_elements;
    let mut out = Vec::with_capacity(num_elements);

    for i in 0..num_elements {
        let byte_idx = i / 4;
        let shift = (i % 4) * 2;
        let bits = (packed.packed_data[byte_idx] >> shift) & 0b11;
        let val = match bits {
            0b10 => -1,
            0b01 => 1,
            _ => 0,
        };
        out.push(val);
    }
    out
}

/// Pack int8 weights in tile-ordered layout.
pub fn cpu_pack_int8_weights(
    weights: &[i8],
    rows: usize,
    cols: usize,
    tile_size: usize,
) -> PackedWeights {
    let start = Instant::now();
    let num_elements = rows * cols;
    assert!(weights.len() >= num_elements, "weights slice too short");

    let tile_rows = rows.div_ceil(tile_size);
    let tile_cols = cols.div_ceil(tile_size);
    let padded_rows = tile_rows * tile_size;
    let padded_cols = tile_cols * tile_size;
    let packed_len = padded_rows * padded_cols;
    let mut packed = vec![0u8; packed_len];

    for tr in 0..tile_rows {
        for tc in 0..tile_cols {
            for lr in 0..tile_size {
                for lc in 0..tile_size {
                    let r = tr * tile_size + lr;
                    let c = tc * tile_size + lc;
                    let val = if r < rows && c < cols { weights[r * cols + c] } else { 0 };
                    let tile_idx = tr * tile_cols + tc;
                    let in_tile = lr * tile_size + lc;
                    let dst = tile_idx * (tile_size * tile_size) + in_tile;
                    packed[dst] = val as u8;
                }
            }
        }
    }

    let elapsed = start.elapsed().as_micros() as u64;

    PackedWeights {
        packed_data: packed,
        original_shape: vec![rows, cols],
        packed_shape: vec![padded_rows, padded_cols],
        format: WeightFormat::Int8Packed,
        scale_factors: None,
        metadata: WeightMetadata {
            original_format: WeightFormat::Int8Packed,
            packed_format: WeightFormat::Int8Packed,
            compression_ratio: num_elements as f32 / packed_len as f32,
            pack_time_us: elapsed,
            num_elements,
        },
    }
}

/// Unpack tile-ordered int8 weights back to row-major.
pub fn cpu_unpack_int8_weights(packed: &PackedWeights, tile_size: usize) -> Vec<i8> {
    let rows = packed.original_shape[0];
    let cols = packed.original_shape[1];
    let tile_rows = rows.div_ceil(tile_size);
    let tile_cols = cols.div_ceil(tile_size);

    let mut out = vec![0i8; rows * cols];

    for tr in 0..tile_rows {
        for tc in 0..tile_cols {
            for lr in 0..tile_size {
                for lc in 0..tile_size {
                    let r = tr * tile_size + lr;
                    let c = tc * tile_size + lc;
                    if r < rows && c < cols {
                        let tile_idx = tr * tile_cols + tc;
                        let in_tile = lr * tile_size + lc;
                        let src = tile_idx * (tile_size * tile_size) + in_tile;
                        out[r * cols + c] = packed.packed_data[src] as i8;
                    }
                }
            }
        }
    }
    out
}

/// Transpose a row-major f32 matrix.
pub fn cpu_transpose_weights(weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert!(weights.len() >= rows * cols, "weights slice too short");
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = weights[r * cols + c];
        }
    }
    out
}

/// Pad rows and cols up to the given alignment. Returns (padded_data, new_rows, new_cols).
pub fn cpu_pad_to_alignment(
    weights: &[f32],
    rows: usize,
    cols: usize,
    alignment: usize,
) -> (Vec<f32>, usize, usize) {
    assert!(alignment > 0, "alignment must be > 0");
    let new_rows = rows.div_ceil(alignment) * alignment;
    let new_cols = cols.div_ceil(alignment) * alignment;
    let mut out = vec![0.0f32; new_rows * new_cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * new_cols + c] = weights[r * cols + c];
        }
    }
    (out, new_rows, new_cols)
}

/// Quantize f32 weights to ternary {-1, 0, 1}.
///
/// Values with `|v| <= threshold` map to 0; positive to 1; negative to -1.
pub fn cpu_quantize_f32_to_ternary(weights: &[f32], threshold: f32) -> Vec<i8> {
    weights
        .iter()
        .map(|&v| {
            if v.abs() <= threshold {
                0i8
            } else if v > 0.0 {
                1
            } else {
                -1
            }
        })
        .collect()
}

/// Compute per-block absolute-max scale factors.
pub fn cpu_compute_scale_factors(weights: &[f32], block_size: usize) -> Vec<f32> {
    assert!(block_size > 0, "block_size must be > 0");
    weights
        .chunks(block_size)
        .map(|block| block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs())))
        .collect()
}

/// Dequantize: multiply each element by its block's scale factor.
pub fn cpu_apply_scale_factors(weights: &[f32], scales: &[f32], block_size: usize) -> Vec<f32> {
    assert!(block_size > 0, "block_size must be > 0");
    weights
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let block_idx = i / block_size;
            let scale = scales.get(block_idx).copied().unwrap_or(1.0);
            v * scale
        })
        .collect()
}

/// Build a preprocessing pipeline from a [`PackingConfig`].
pub fn create_preprocess_pipeline(config: PackingConfig) -> PreprocessPipeline {
    let mut stages = Vec::new();

    if config.pack_transpose {
        stages.push(PreprocessStage::Transpose);
    }
    if config.alignment > 1 {
        stages.push(PreprocessStage::PadAlignment(config.alignment));
    }
    if config.tile_size > 1 {
        stages.push(PreprocessStage::TileReorder(config.tile_size, config.tile_size));
    }
    match config.target_format {
        WeightFormat::GpuTernaryPacked | WeightFormat::TernaryI2S => {
            stages.push(PreprocessStage::Quantize);
            stages.push(PreprocessStage::Pack);
        }
        WeightFormat::Int8Packed | WeightFormat::GpuInt4Packed | WeightFormat::QK256Packed => {
            stages.push(PreprocessStage::Pack);
        }
        _ => {}
    }

    PreprocessPipeline { stages, config }
}

/// Execute the preprocessing pipeline on a [`WeightTensor`].
pub fn cpu_run_pipeline(
    pipeline: &PreprocessPipeline,
    input: WeightTensor,
) -> Result<PackedWeights, PreprocessError> {
    if input.shape.len() != 2 {
        return Err(PreprocessError::ShapeMismatch);
    }

    let start = Instant::now();
    let mut rows = input.shape[0];
    let mut cols = input.shape[1];
    let num_elements = rows * cols;

    // Interpret input bytes as f32 for RawF32 or as i8 for ternary/int8.
    let mut f32_data: Vec<f32> = match input.format {
        WeightFormat::RawF32 => {
            if input.data.len() < num_elements * 4 {
                return Err(PreprocessError::ShapeMismatch);
            }
            input.data.chunks_exact(4).take(num_elements).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
        }
        WeightFormat::TernaryI2S | WeightFormat::Int8Packed => {
            if input.data.len() < num_elements {
                return Err(PreprocessError::ShapeMismatch);
            }
            input.data[..num_elements].iter().map(|&b| (b as i8) as f32).collect()
        }
        _ => {
            return Err(PreprocessError::UnsupportedConversion {
                from: input.format,
                to: pipeline.config.target_format,
            });
        }
    };

    for stage in &pipeline.stages {
        match stage {
            PreprocessStage::Transpose => {
                f32_data = cpu_transpose_weights(&f32_data, rows, cols);
                std::mem::swap(&mut rows, &mut cols);
            }
            PreprocessStage::PadAlignment(align) => {
                let (padded, nr, nc) = cpu_pad_to_alignment(&f32_data, rows, cols, *align);
                f32_data = padded;
                rows = nr;
                cols = nc;
            }
            PreprocessStage::TileReorder(_tr, _tc) => {
                // Tile reorder is handled during the final pack step for int8.
                // For f32 intermediate, this is a no-op marker.
            }
            PreprocessStage::Quantize => {
                // Quantize in-place (threshold = 0.5 by default for pipeline).
                let ternary = cpu_quantize_f32_to_ternary(&f32_data, 0.5);
                f32_data = ternary.iter().map(|&v| v as f32).collect();
            }
            PreprocessStage::Pack => {
                // Terminal stage — produce the final packed output.
                let elapsed = start.elapsed().as_micros() as u64;
                let final_num = rows * cols;

                match pipeline.config.target_format {
                    WeightFormat::GpuTernaryPacked | WeightFormat::TernaryI2S => {
                        let i8_data: Vec<i8> = f32_data.iter().map(|&v| v as i8).collect();
                        let mut packed = cpu_pack_ternary_weights(&i8_data, rows, cols);
                        packed.metadata.pack_time_us = elapsed;
                        packed.metadata.num_elements = final_num;
                        return Ok(packed);
                    }
                    WeightFormat::Int8Packed => {
                        let i8_data: Vec<i8> = f32_data.iter().map(|&v| v as i8).collect();
                        let mut packed =
                            cpu_pack_int8_weights(&i8_data, rows, cols, pipeline.config.tile_size);
                        packed.metadata.pack_time_us = elapsed;
                        packed.metadata.num_elements = final_num;
                        return Ok(packed);
                    }
                    other => {
                        return Err(PreprocessError::UnsupportedConversion {
                            from: input.format,
                            to: other,
                        });
                    }
                }
            }
        }
    }

    // If there was no Pack stage, wrap f32 data as-is.
    let elapsed = start.elapsed().as_micros() as u64;
    let final_num = rows * cols;
    let packed_bytes: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let original_bytes = num_elements * 4;
    let packed_bytes_len = packed_bytes.len();

    Ok(PackedWeights {
        packed_data: packed_bytes,
        original_shape: vec![input.shape[0], input.shape[1]],
        packed_shape: vec![rows, cols],
        format: pipeline.config.target_format,
        scale_factors: None,
        metadata: WeightMetadata {
            original_format: input.format,
            packed_format: pipeline.config.target_format,
            compression_ratio: original_bytes as f32 / packed_bytes_len as f32,
            pack_time_us: elapsed,
            num_elements: final_num,
        },
    })
}

/// Verify round-trip correctness: quantize f32 → ternary → pack → unpack → compare.
pub fn cpu_validate_packing(original: &[f32], packed: &PackedWeights) -> bool {
    let unpacked = cpu_unpack_ternary_weights(packed);
    if unpacked.len() != original.len() {
        return false;
    }
    // Compare against the quantized form (threshold = 0.5).
    let expected = cpu_quantize_f32_to_ternary(original, 0.5);
    unpacked == expected
}

/// Format metadata as a human-readable string.
pub fn format_weight_metadata(meta: &WeightMetadata) -> String {
    format!(
        "WeightMetadata {{ original: {}, packed: {}, ratio: {:.2}x, time: {}µs, elements: {} }}",
        meta.original_format, meta.packed_format, meta.compression_ratio, meta.pack_time_us, meta.num_elements,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Ternary pack / unpack ------------------------------------------

    #[test]
    fn test_ternary_pack_unpack_basic() {
        let weights: Vec<i8> = vec![-1, 0, 1, 0, 1, -1, 0, 1];
        let packed = cpu_pack_ternary_weights(&weights, 2, 4);
        let unpacked = cpu_unpack_ternary_weights(&packed);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_ternary_pack_all_neg1() {
        let weights = vec![-1i8; 16];
        let packed = cpu_pack_ternary_weights(&weights, 4, 4);
        let unpacked = cpu_unpack_ternary_weights(&packed);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_ternary_pack_all_zeros() {
        let weights = vec![0i8; 12];
        let packed = cpu_pack_ternary_weights(&weights, 3, 4);
        let unpacked = cpu_unpack_ternary_weights(&packed);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_ternary_pack_all_ones() {
        let weights = vec![1i8; 8];
        let packed = cpu_pack_ternary_weights(&weights, 2, 4);
        let unpacked = cpu_unpack_ternary_weights(&packed);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_ternary_pack_non_multiple_of_4() {
        let weights: Vec<i8> = vec![1, -1, 0, 1, -1];
        let packed = cpu_pack_ternary_weights(&weights, 1, 5);
        let unpacked = cpu_unpack_ternary_weights(&packed);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_ternary_pack_single_element() {
        for &val in &[-1i8, 0, 1] {
            let packed = cpu_pack_ternary_weights(&[val], 1, 1);
            let unpacked = cpu_unpack_ternary_weights(&packed);
            assert_eq!(unpacked, vec![val]);
        }
    }

    #[test]
    fn test_ternary_pack_large_matrix() {
        let rows = 128;
        let cols = 256;
        let weights: Vec<i8> =
            (0..rows * cols).map(|i| [(-1i8), 0, 1][i % 3]).collect();
        let packed = cpu_pack_ternary_weights(&weights, rows, cols);
        let unpacked = cpu_unpack_ternary_weights(&packed);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_ternary_compression_ratio_gt_1() {
        let weights = vec![0i8; 1024];
        let packed = cpu_pack_ternary_weights(&weights, 32, 32);
        assert!(packed.metadata.compression_ratio > 1.0);
    }

    #[test]
    fn test_ternary_packed_shape() {
        let packed = cpu_pack_ternary_weights(&vec![0i8; 20], 4, 5);
        assert_eq!(packed.original_shape, vec![4, 5]);
        assert_eq!(packed.packed_shape, vec![4, 2]); // ceil(5/4) = 2
    }

    #[test]
    fn test_ternary_pack_metadata_format() {
        let packed = cpu_pack_ternary_weights(&vec![0i8; 8], 2, 4);
        assert_eq!(packed.metadata.original_format, WeightFormat::TernaryI2S);
        assert_eq!(packed.metadata.packed_format, WeightFormat::GpuTernaryPacked);
        assert_eq!(packed.metadata.num_elements, 8);
    }

    // ---- Int8 pack / unpack ---------------------------------------------

    #[test]
    fn test_int8_pack_unpack_basic() {
        let weights: Vec<i8> = (0..16).map(|i| i as i8).collect();
        let packed = cpu_pack_int8_weights(&weights, 4, 4, 2);
        let unpacked = cpu_unpack_int8_weights(&packed, 2);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_int8_pack_unpack_non_tile_aligned() {
        let weights: Vec<i8> = (0..15).map(|i| (i + 1) as i8).collect();
        let packed = cpu_pack_int8_weights(&weights, 3, 5, 4);
        let unpacked = cpu_unpack_int8_weights(&packed, 4);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_int8_pack_single_tile() {
        let weights: Vec<i8> = vec![1, 2, 3, 4];
        let packed = cpu_pack_int8_weights(&weights, 2, 2, 2);
        let unpacked = cpu_unpack_int8_weights(&packed, 2);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_int8_pack_1x1() {
        let weights: Vec<i8> = vec![42];
        let packed = cpu_pack_int8_weights(&weights, 1, 1, 1);
        let unpacked = cpu_unpack_int8_weights(&packed, 1);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_int8_pack_negative_values() {
        let weights: Vec<i8> = vec![-128, -1, 0, 1, 127, -50, 50, 100, -100];
        let packed = cpu_pack_int8_weights(&weights, 3, 3, 2);
        let unpacked = cpu_unpack_int8_weights(&packed, 2);
        assert_eq!(unpacked, weights);
    }

    // ---- Transpose ------------------------------------------------------

    #[test]
    fn test_transpose_2x3() {
        let m = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = cpu_transpose_weights(&m, 2, 3);
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_1x1() {
        let t = cpu_transpose_weights(&[42.0], 1, 1);
        assert_eq!(t, vec![42.0]);
    }

    #[test]
    fn test_transpose_square() {
        let m = vec![1.0, 2.0, 3.0, 4.0];
        let t = cpu_transpose_weights(&m, 2, 2);
        assert_eq!(t, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_double_is_identity() {
        let m: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t1 = cpu_transpose_weights(&m, 3, 4);
        let t2 = cpu_transpose_weights(&t1, 4, 3);
        assert_eq!(t2, m);
    }

    // ---- Pad to alignment -----------------------------------------------

    #[test]
    fn test_pad_already_aligned() {
        let m = vec![1.0; 16];
        let (padded, nr, nc) = cpu_pad_to_alignment(&m, 4, 4, 4);
        assert_eq!(nr, 4);
        assert_eq!(nc, 4);
        assert_eq!(padded.len(), 16);
    }

    #[test]
    fn test_pad_needs_padding() {
        let m = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (padded, nr, nc) = cpu_pad_to_alignment(&m, 2, 3, 4);
        assert_eq!(nr, 4);
        assert_eq!(nc, 4);
        assert_eq!(padded.len(), 16);
        // Original values preserved.
        assert_eq!(padded[0], 1.0);
        assert_eq!(padded[1], 2.0);
        assert_eq!(padded[2], 3.0);
        assert_eq!(padded[3], 0.0); // padding
        assert_eq!(padded[4], 4.0);
        assert_eq!(padded[5], 5.0);
        assert_eq!(padded[6], 6.0);
    }

    #[test]
    fn test_pad_1x1_to_8() {
        let (padded, nr, nc) = cpu_pad_to_alignment(&[7.0], 1, 1, 8);
        assert_eq!(nr, 8);
        assert_eq!(nc, 8);
        assert_eq!(padded[0], 7.0);
        assert_eq!(padded.iter().filter(|&&v| v == 0.0).count(), 63);
    }

    #[test]
    fn test_pad_preserves_all_values() {
        let m: Vec<f32> = (1..=6).map(|i| i as f32).collect();
        let (padded, _nr, nc) = cpu_pad_to_alignment(&m, 2, 3, 4);
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(padded[r * nc + c], m[r * 3 + c]);
            }
        }
    }

    // ---- Quantize f32 → ternary -----------------------------------------

    #[test]
    fn test_quantize_basic() {
        let w = vec![1.0, -1.0, 0.0, 0.3, -0.3, 0.6, -0.6];
        let q = cpu_quantize_f32_to_ternary(&w, 0.5);
        assert_eq!(q, vec![1, -1, 0, 0, 0, 1, -1]);
    }

    #[test]
    fn test_quantize_threshold_boundary() {
        let w = vec![0.5, -0.5, 0.500001, -0.500001];
        let q = cpu_quantize_f32_to_ternary(&w, 0.5);
        // |0.5| <= 0.5 → 0
        assert_eq!(q[0], 0);
        assert_eq!(q[1], 0);
        assert_eq!(q[2], 1);
        assert_eq!(q[3], -1);
    }

    #[test]
    fn test_quantize_all_zeros() {
        let q = cpu_quantize_f32_to_ternary(&vec![0.0; 10], 0.1);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_quantize_zero_threshold() {
        let w = vec![0.0, 0.001, -0.001];
        let q = cpu_quantize_f32_to_ternary(&w, 0.0);
        assert_eq!(q, vec![0, 1, -1]);
    }

    // ---- Scale factors --------------------------------------------------

    #[test]
    fn test_scale_factors_basic() {
        let w = vec![1.0, -2.0, 3.0, -0.5, 0.1, 4.0];
        let s = cpu_compute_scale_factors(&w, 3);
        assert_eq!(s, vec![3.0, 4.0]);
    }

    #[test]
    fn test_scale_factors_single_block() {
        let w = vec![-7.0, 3.0, 5.0];
        let s = cpu_compute_scale_factors(&w, 10);
        assert_eq!(s, vec![7.0]);
    }

    #[test]
    fn test_scale_factors_all_zeros() {
        let s = cpu_compute_scale_factors(&vec![0.0; 8], 4);
        assert_eq!(s, vec![0.0, 0.0]);
    }

    #[test]
    fn test_apply_scale_factors() {
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let s = vec![2.0, 0.5];
        let result = cpu_apply_scale_factors(&w, &s, 2);
        assert_eq!(result, vec![2.0, 4.0, 1.5, 2.0]);
    }

    #[test]
    fn test_apply_scale_factors_identity() {
        let w = vec![1.0, 2.0, 3.0];
        let s = vec![1.0];
        let result = cpu_apply_scale_factors(&w, &s, 4);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    // ---- Pipeline -------------------------------------------------------

    #[test]
    fn test_pipeline_single_stage_pack() {
        let config = PackingConfig {
            target_format: WeightFormat::GpuTernaryPacked,
            tile_size: 1,
            alignment: 1,
            pack_transpose: false,
        };
        let pipeline = create_preprocess_pipeline(config);
        assert!(pipeline.stages.contains(&PreprocessStage::Quantize));
        assert!(pipeline.stages.contains(&PreprocessStage::Pack));
    }

    #[test]
    fn test_pipeline_with_transpose() {
        let config = PackingConfig {
            target_format: WeightFormat::GpuTernaryPacked,
            tile_size: 1,
            alignment: 1,
            pack_transpose: true,
        };
        let pipeline = create_preprocess_pipeline(config);
        assert_eq!(pipeline.stages[0], PreprocessStage::Transpose);
    }

    #[test]
    fn test_pipeline_with_alignment() {
        let config = PackingConfig {
            target_format: WeightFormat::Int8Packed,
            tile_size: 1,
            alignment: 16,
            pack_transpose: false,
        };
        let pipeline = create_preprocess_pipeline(config);
        assert!(pipeline.stages.contains(&PreprocessStage::PadAlignment(16)));
    }

    #[test]
    fn test_pipeline_multi_stage() {
        let config = PackingConfig {
            target_format: WeightFormat::GpuTernaryPacked,
            tile_size: 8,
            alignment: 16,
            pack_transpose: true,
        };
        let pipeline = create_preprocess_pipeline(config);
        // Transpose, PadAlignment, TileReorder, Quantize, Pack
        assert!(pipeline.stages.len() >= 4);
        assert_eq!(pipeline.stages[0], PreprocessStage::Transpose);
    }

    #[test]
    fn test_run_pipeline_ternary() {
        let config = PackingConfig {
            target_format: WeightFormat::GpuTernaryPacked,
            tile_size: 1,
            alignment: 1,
            pack_transpose: false,
        };
        let pipeline = create_preprocess_pipeline(config);

        let data: Vec<u8> = vec![1.0f32, -1.0, 0.0, 0.8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input = WeightTensor {
            data,
            shape: vec![2, 2],
            format: WeightFormat::RawF32,
            byte_stride: 8,
        };

        let result = cpu_run_pipeline(&pipeline, input);
        assert!(result.is_ok());
        let packed = result.unwrap();
        assert_eq!(packed.format, WeightFormat::GpuTernaryPacked);
        assert_eq!(packed.original_shape, vec![2, 2]);
    }

    #[test]
    fn test_run_pipeline_int8() {
        let config = PackingConfig {
            target_format: WeightFormat::Int8Packed,
            tile_size: 2,
            alignment: 1,
            pack_transpose: false,
        };
        let pipeline = create_preprocess_pipeline(config);

        let data: Vec<u8> = vec![10i8, 20, -10, -20].iter().map(|&v| v as u8).collect();
        let input = WeightTensor {
            data,
            shape: vec![2, 2],
            format: WeightFormat::Int8Packed,
            byte_stride: 2,
        };

        let result = cpu_run_pipeline(&pipeline, input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_pipeline_shape_mismatch() {
        let config = PackingConfig::default();
        let pipeline = create_preprocess_pipeline(config);
        let input = WeightTensor {
            data: vec![0; 4],
            shape: vec![2, 2, 2], // 3D — not supported
            format: WeightFormat::RawF32,
            byte_stride: 0,
        };
        assert_eq!(cpu_run_pipeline(&pipeline, input), Err(PreprocessError::ShapeMismatch));
    }

    #[test]
    fn test_run_pipeline_unsupported_format() {
        let config = PackingConfig {
            target_format: WeightFormat::GpuTernaryPacked,
            ..PackingConfig::default()
        };
        let pipeline = create_preprocess_pipeline(config);
        let input = WeightTensor {
            data: vec![0; 4],
            shape: vec![2, 2],
            format: WeightFormat::QK256Packed, // not handled as input
            byte_stride: 0,
        };
        assert!(cpu_run_pipeline(&pipeline, input).is_err());
    }

    // ---- Validate packing -----------------------------------------------

    #[test]
    fn test_validate_packing_roundtrip() {
        let original = vec![1.0, -1.0, 0.0, 0.8, -0.8, 0.3];
        let ternary = cpu_quantize_f32_to_ternary(&original, 0.5);
        let packed = cpu_pack_ternary_weights(&ternary, 2, 3);
        assert!(cpu_validate_packing(&original, &packed));
    }

    #[test]
    fn test_validate_packing_fails_on_mismatch() {
        let original = vec![1.0, -1.0, 0.0, 0.8];
        let fake_ternary = vec![0i8; 4]; // all-zero — doesn't match original quantized
        let packed = cpu_pack_ternary_weights(&fake_ternary, 2, 2);
        assert!(!cpu_validate_packing(&original, &packed));
    }

    // ---- Metadata formatting --------------------------------------------

    #[test]
    fn test_format_metadata() {
        let meta = WeightMetadata {
            original_format: WeightFormat::RawF32,
            packed_format: WeightFormat::GpuTernaryPacked,
            compression_ratio: 4.0,
            pack_time_us: 123,
            num_elements: 1024,
        };
        let s = format_weight_metadata(&meta);
        assert!(s.contains("4.00x"));
        assert!(s.contains("123µs"));
        assert!(s.contains("1024"));
    }

    #[test]
    fn test_weight_format_display() {
        assert_eq!(format!("{}", WeightFormat::RawF32), "Raw_F32");
        assert_eq!(format!("{}", WeightFormat::GpuTernaryPacked), "Gpu_Ternary_Packed");
    }

    // ---- Edge cases -----------------------------------------------------

    #[test]
    fn test_edge_1x1_ternary() {
        let packed = cpu_pack_ternary_weights(&[1], 1, 1);
        assert_eq!(cpu_unpack_ternary_weights(&packed), vec![1]);
    }

    #[test]
    fn test_edge_non_power_of_2_dims() {
        let rows = 7;
        let cols = 13;
        let weights: Vec<i8> = (0..rows * cols).map(|i| (i % 3) as i8 - 1).collect();
        let packed = cpu_pack_ternary_weights(&weights, rows, cols);
        let unpacked = cpu_unpack_ternary_weights(&packed);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_edge_all_same_value_ternary() {
        for &val in &[-1i8, 0, 1] {
            let weights = vec![val; 100];
            let packed = cpu_pack_ternary_weights(&weights, 10, 10);
            let unpacked = cpu_unpack_ternary_weights(&packed);
            assert_eq!(unpacked, weights);
        }
    }

    #[test]
    fn test_property_pack_unpack_identity() {
        // Property: for all valid ternary inputs, pack→unpack is identity.
        let patterns: Vec<Vec<i8>> = vec![
            vec![-1, 0, 1, -1, 0, 1],
            vec![1; 33],
            vec![0; 1],
            (0..256).map(|i| (i % 3) as i8 - 1).collect(),
        ];
        for w in &patterns {
            let n = w.len();
            let packed = cpu_pack_ternary_weights(w, 1, n);
            let unpacked = cpu_unpack_ternary_weights(&packed);
            assert_eq!(&unpacked, w, "round-trip failed for len={n}");
        }
    }

    #[test]
    fn test_compression_ratio_ternary() {
        let w = vec![1i8; 1024];
        let packed = cpu_pack_ternary_weights(&w, 32, 32);
        // 1024 i8 bytes → 256 packed bytes → ratio = 4.0
        assert!((packed.metadata.compression_ratio - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_preprocess_error_display() {
        assert_eq!(format!("{}", PreprocessError::InvalidFormat), "invalid weight format");
        assert_eq!(format!("{}", PreprocessError::ShapeMismatch), "weight shape mismatch");
        let e = PreprocessError::UnsupportedConversion {
            from: WeightFormat::RawF32,
            to: WeightFormat::QK256Packed,
        };
        assert!(format!("{e}").contains("Raw_F32"));
    }

    #[test]
    fn test_packing_config_default() {
        let cfg = PackingConfig::default();
        assert_eq!(cfg.target_format, WeightFormat::GpuTernaryPacked);
        assert_eq!(cfg.tile_size, 16);
        assert_eq!(cfg.alignment, 64);
        assert!(!cfg.pack_transpose);
    }

    #[test]
    fn test_pipeline_rawf32_passthrough() {
        // A pipeline targeting RawF32 should pass data through without packing.
        let config = PackingConfig {
            target_format: WeightFormat::RawF32,
            tile_size: 1,
            alignment: 1,
            pack_transpose: false,
        };
        let pipeline = create_preprocess_pipeline(config);

        let vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input = WeightTensor {
            data,
            shape: vec![2, 2],
            format: WeightFormat::RawF32,
            byte_stride: 8,
        };

        let result = cpu_run_pipeline(&pipeline, input).unwrap();
        assert_eq!(result.format, WeightFormat::RawF32);
        let round: Vec<f32> = result
            .packed_data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(round, vals);
    }

    #[test]
    fn test_scale_factors_partial_last_block() {
        let w = vec![1.0, -2.0, 3.0, -4.0, 5.0]; // block_size=3 → [1,-2,3] and [-4,5]
        let s = cpu_compute_scale_factors(&w, 3);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], 3.0);
        assert_eq!(s[1], 5.0);
    }

    #[test]
    fn test_transpose_wide_matrix() {
        let m: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let t = cpu_transpose_weights(&m, 2, 4);
        // 2×4 → 4×2
        assert_eq!(t, vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]);
    }
}
