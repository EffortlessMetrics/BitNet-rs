//! High-level quantization / dequantization of `f32` slices.
//!
//! Supports both [`BlockFormat::BitNet32F16`] and [`BlockFormat::Qk256`].

use crate::error::I2SError;
use crate::format::{BlockFormat, f16_to_f32, f32_to_f16};
use crate::pack::{pack_i2s, unpack_i2s};

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Options controlling quantization behaviour.
#[derive(Debug, Clone)]
pub struct QuantizeOpts {
    /// Block format to use.
    pub format: BlockFormat,
}

impl Default for QuantizeOpts {
    fn default() -> Self {
        Self { format: BlockFormat::Qk256 }
    }
}

impl QuantizeOpts {
    /// Create options for QK256 format.
    #[must_use]
    pub const fn qk256() -> Self {
        Self { format: BlockFormat::Qk256 }
    }

    /// Create options for `BitNet32`-F16 format.
    #[must_use]
    pub const fn bitnet32() -> Self {
        Self { format: BlockFormat::BitNet32F16 }
    }
}

// ---------------------------------------------------------------------------
// Ternary snap: f32 -> {-1, 0, +1}
// ---------------------------------------------------------------------------

/// Snap a float to the nearest ternary value.
///
/// - `v <= -0.5` maps to -1
/// - `-0.5 < v < 0.5` maps to 0
/// - `v >= 0.5` maps to +1
#[allow(clippy::bool_to_int_with_if)]
#[inline]
fn snap_ternary(v: f32) -> i8 {
    if v <= -0.5 {
        -1
    } else if v >= 0.5 {
        1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Per-block scale helpers
// ---------------------------------------------------------------------------

/// Compute the absmax scale for a block of `f32` values.
fn block_absmax(block: &[f32]) -> f32 {
    block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()))
}

// ---------------------------------------------------------------------------
// Quantize f32 -> I2_S packed bytes
// ---------------------------------------------------------------------------

/// Quantize an `f32` slice into packed `I2_S` bytes.
///
/// For **QK256**: values are snapped directly to ternary (no per-block scale).
/// For **BitNet32-F16**: an absmax scale is computed per 32-element block and
/// values are normalised before snapping. Scales are appended as little-endian
/// f16 after each block's 8 packed bytes.
///
/// # Errors
///
/// - [`I2SError::EmptyInput`] if `data` is empty.
/// - [`I2SError::BlockAlignment`] if `data.len()` is not a multiple of the
///   block size.
pub fn quantize_f32(data: &[f32], opts: &QuantizeOpts) -> Result<Vec<u8>, I2SError> {
    if data.is_empty() {
        return Err(I2SError::EmptyInput);
    }
    let bs = opts.format.block_size();
    if !data.len().is_multiple_of(bs) {
        return Err(I2SError::BlockAlignment { len: data.len(), block_size: bs });
    }

    match opts.format {
        BlockFormat::Qk256 => quantize_qk256(data),
        BlockFormat::BitNet32F16 => quantize_bitnet32(data),
    }
}

fn quantize_qk256(data: &[f32]) -> Result<Vec<u8>, I2SError> {
    let ternary: Vec<i8> = data.iter().map(|&v| snap_ternary(v)).collect();
    pack_i2s(&ternary)
}

fn quantize_bitnet32(data: &[f32]) -> Result<Vec<u8>, I2SError> {
    let num_blocks = data.len() / 32;
    // 10 bytes per block: 8 packed + 2 f16 scale
    let mut out = Vec::with_capacity(num_blocks * 10);

    for block in data.chunks_exact(32) {
        let scale = block_absmax(block);
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let ternary: Vec<i8> = block.iter().map(|&v| snap_ternary(v * inv_scale)).collect();
        let packed = pack_i2s(&ternary)?;
        out.extend_from_slice(&packed);

        // Append f16 scale (little-endian)
        let scale_f16 = f32_to_f16(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Dequantize I2_S packed bytes -> f32
// ---------------------------------------------------------------------------

/// Dequantize packed `I2_S` bytes back to `f32` values.
///
/// `num_elements` is the number of output `f32` values.
///
/// For **QK256**: codes map directly to `{-1.0, 0.0, 1.0}`.
/// For **BitNet32-F16**: each 10-byte block is unpacked and multiplied by its
/// per-block f16 scale.
///
/// # Errors
///
/// - [`I2SError::EmptyInput`] if `packed` is empty or `num_elements` is 0.
/// - [`I2SError::PackedLengthMismatch`] if `packed` is too short.
/// - [`I2SError::BlockAlignment`] if `num_elements` is not a multiple of the
///   block size.
pub fn dequantize_f32(
    packed: &[u8],
    num_elements: usize,
    opts: &QuantizeOpts,
) -> Result<Vec<f32>, I2SError> {
    if packed.is_empty() || num_elements == 0 {
        return Err(I2SError::EmptyInput);
    }
    let bs = opts.format.block_size();
    if !num_elements.is_multiple_of(bs) {
        return Err(I2SError::BlockAlignment { len: num_elements, block_size: bs });
    }

    match opts.format {
        BlockFormat::Qk256 => dequantize_qk256(packed, num_elements),
        BlockFormat::BitNet32F16 => dequantize_bitnet32(packed, num_elements),
    }
}

fn dequantize_qk256(packed: &[u8], num_elements: usize) -> Result<Vec<f32>, I2SError> {
    let ternary = unpack_i2s(packed, num_elements)?;
    Ok(ternary.iter().map(|&v| f32::from(v)).collect())
}

fn dequantize_bitnet32(packed: &[u8], num_elements: usize) -> Result<Vec<f32>, I2SError> {
    let num_blocks = num_elements / 32;
    let expected_bytes = num_blocks * 10;
    if packed.len() < expected_bytes {
        return Err(I2SError::PackedLengthMismatch {
            actual: packed.len(),
            expected: expected_bytes,
        });
    }

    let mut out = Vec::with_capacity(num_elements);
    for b in 0..num_blocks {
        let block_start = b * 10;
        let data_bytes = &packed[block_start..block_start + 8];
        let scale_bytes: [u8; 2] = packed[block_start + 8..block_start + 10].try_into().unwrap();
        let scale = f16_to_f32(u16::from_le_bytes(scale_bytes));

        let ternary = unpack_i2s(data_bytes, 32)?;
        for &v in &ternary {
            out.push(f32::from(v) * scale);
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Batch quantization for model conversion
// ---------------------------------------------------------------------------

/// Result of quantizing one tensor in a batch.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Packed `I2_S` bytes.
    pub data: Vec<u8>,
    /// Number of original `f32` elements.
    pub num_elements: usize,
    /// Block format used.
    pub format: BlockFormat,
}

/// Quantize a batch of `f32` tensors for model conversion.
///
/// Each tensor is quantized independently using the specified options. Tensors
/// that are not block-aligned are returned as `Err` in the result vec.
#[must_use]
pub fn quantize_batch(
    tensors: &[&[f32]],
    opts: &QuantizeOpts,
) -> Vec<Result<QuantizedTensor, I2SError>> {
    tensors
        .iter()
        .map(|data| {
            let packed = quantize_f32(data, opts)?;
            Ok(QuantizedTensor { data: packed, num_elements: data.len(), format: opts.format })
        })
        .collect()
}

/// Dequantize a batch of [`QuantizedTensor`]s back to `f32` vecs.
///
/// Each tensor is dequantized independently. Per-tensor errors are captured.
#[must_use]
pub fn dequantize_batch(tensors: &[QuantizedTensor]) -> Vec<Result<Vec<f32>, I2SError>> {
    tensors
        .iter()
        .map(|qt| {
            let opts = QuantizeOpts { format: qt.format };
            dequantize_f32(&qt.data, qt.num_elements, &opts)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // -- QK256 quantize/dequantize ------------------------------------------

    #[test]
    fn qk256_roundtrip_ternary() {
        let opts = QuantizeOpts::qk256();
        let data: Vec<f32> = (0..256).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 256, &opts).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn qk256_all_zeros() {
        let opts = QuantizeOpts::qk256();
        let data = vec![0.0f32; 256];
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 256, &opts).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn qk256_all_ones() {
        let opts = QuantizeOpts::qk256();
        let data = vec![1.0f32; 256];
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 256, &opts).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn qk256_all_neg_ones() {
        let opts = QuantizeOpts::qk256();
        let data = vec![-1.0f32; 256];
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 256, &opts).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn qk256_snapping() {
        let opts = QuantizeOpts::qk256();
        let data: Vec<f32> = {
            let mut v = vec![0.0f32; 256];
            v[0] = 0.49; // -> 0
            v[1] = 0.51; // -> 1
            v[2] = -0.49; // -> 0
            v[3] = -0.51; // -> -1
            v[4] = 0.5; // -> 1
            v[5] = -0.5; // -> -1
            v
        };
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 256, &opts).unwrap();
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], -1.0);
        assert_eq!(out[4], 1.0);
        assert_eq!(out[5], -1.0);
    }

    #[test]
    fn qk256_multiple_blocks() {
        let opts = QuantizeOpts::qk256();
        let data: Vec<f32> = (0..512).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let packed = quantize_f32(&data, &opts).unwrap();
        assert_eq!(packed.len(), 128);
        let out = dequantize_f32(&packed, 512, &opts).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn qk256_block_alignment_error() {
        let opts = QuantizeOpts::qk256();
        let data = vec![0.0f32; 100];
        let err = quantize_f32(&data, &opts).unwrap_err();
        assert!(matches!(err, I2SError::BlockAlignment { .. }));
    }

    // -- BitNet32-F16 quantize/dequantize -----------------------------------

    #[test]
    fn bitnet32_roundtrip_ternary() {
        let opts = QuantizeOpts::bitnet32();
        let data: Vec<f32> = (0..32).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let packed = quantize_f32(&data, &opts).unwrap();
        assert_eq!(packed.len(), 10);
        let out = dequantize_f32(&packed, 32, &opts).unwrap();
        for (a, b) in out.iter().zip(data.iter()) {
            assert!((a - b).abs() < 0.01, "bitnet32 roundtrip: {a} vs {b}",);
        }
    }

    #[test]
    fn bitnet32_all_zeros() {
        let opts = QuantizeOpts::bitnet32();
        let data = vec![0.0f32; 32];
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 32, &opts).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn bitnet32_scale_preserved() {
        let opts = QuantizeOpts::bitnet32();
        let data: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 5.0 } else { -5.0 }).collect();
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 32, &opts).unwrap();
        for (a, b) in out.iter().zip(data.iter()) {
            assert!((a - b).abs() < 0.1, "bitnet32 scale: {a} vs {b}");
        }
    }

    #[test]
    fn bitnet32_multiple_blocks() {
        let opts = QuantizeOpts::bitnet32();
        let data: Vec<f32> = (0..64).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let packed = quantize_f32(&data, &opts).unwrap();
        assert_eq!(packed.len(), 20);
        let out = dequantize_f32(&packed, 64, &opts).unwrap();
        for (a, b) in out.iter().zip(data.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn bitnet32_block_alignment_error() {
        let opts = QuantizeOpts::bitnet32();
        let data = vec![0.0f32; 33];
        let err = quantize_f32(&data, &opts).unwrap_err();
        assert!(matches!(err, I2SError::BlockAlignment { .. }));
    }

    // -- Edge values --------------------------------------------------------

    #[test]
    fn edge_values_minus_one() {
        let opts = QuantizeOpts::qk256();
        let data = vec![-1.0f32; 256];
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 256, &opts).unwrap();
        assert!(out.iter().all(|&v| v == -1.0));
    }

    #[test]
    fn edge_values_zero() {
        let opts = QuantizeOpts::qk256();
        let data = vec![0.0f32; 256];
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 256, &opts).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn edge_values_plus_one() {
        let opts = QuantizeOpts::qk256();
        let data = vec![1.0f32; 256];
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 256, &opts).unwrap();
        assert!(out.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn edge_values_mixed_pattern() {
        let opts = QuantizeOpts::qk256();
        let mut data = vec![0.0f32; 256];
        for (i, val) in data.iter_mut().enumerate() {
            *val = match i % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            };
        }
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 256, &opts).unwrap();
        assert_eq!(out, data);
    }

    // -- Empty / error cases ------------------------------------------------

    #[test]
    fn quantize_empty_errors() {
        let opts = QuantizeOpts::qk256();
        assert!(quantize_f32(&[], &opts).is_err());
    }

    #[test]
    fn dequantize_empty_errors() {
        let opts = QuantizeOpts::qk256();
        assert!(dequantize_f32(&[], 256, &opts).is_err());
        assert!(dequantize_f32(&[0u8; 64], 0, &opts).is_err());
    }

    #[test]
    fn dequantize_packed_too_short() {
        let opts = QuantizeOpts::bitnet32();
        let err = dequantize_f32(&[0u8; 5], 32, &opts).unwrap_err();
        assert!(matches!(err, I2SError::PackedLengthMismatch { .. }));
    }

    // -- Batch API ----------------------------------------------------------

    #[test]
    fn batch_quantize_basic() {
        let opts = QuantizeOpts::qk256();
        let t1: Vec<f32> = (0..256).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let t2: Vec<f32> = vec![0.0; 256];
        let results = quantize_batch(&[&t1, &t2], &opts);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
    }

    #[test]
    fn batch_quantize_mixed_errors() {
        let opts = QuantizeOpts::qk256();
        let good: Vec<f32> = vec![0.0; 256];
        let bad: Vec<f32> = vec![0.0; 100];
        let results = quantize_batch(&[&good, &bad], &opts);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
    }

    #[test]
    fn batch_roundtrip() {
        let opts = QuantizeOpts::qk256();
        let t1: Vec<f32> = vec![-1.0; 256];
        let t2: Vec<f32> = vec![1.0; 512];
        let quantized: Vec<QuantizedTensor> =
            quantize_batch(&[&t1, &t2], &opts).into_iter().filter_map(Result::ok).collect();
        let deq = dequantize_batch(&quantized);
        assert_eq!(deq.len(), 2);
        assert_eq!(deq[0].as_ref().unwrap(), &t1);
        assert_eq!(deq[1].as_ref().unwrap(), &t2);
    }

    #[test]
    fn batch_empty() {
        let opts = QuantizeOpts::qk256();
        let results = quantize_batch(&[], &opts);
        assert!(results.is_empty());
    }

    // -- Large tensor sizes -------------------------------------------------

    #[test]
    fn large_tensor_1k_qk256() {
        let opts = QuantizeOpts::qk256();
        let data: Vec<f32> = (0..1024).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 1024, &opts).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn large_tensor_4k_bitnet32() {
        let opts = QuantizeOpts::bitnet32();
        let data: Vec<f32> = (0..4096).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 4096, &opts).unwrap();
        for (a, b) in out.iter().zip(data.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn large_tensor_8k_qk256() {
        let opts = QuantizeOpts::qk256();
        let data: Vec<f32> = (0..8192).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let packed = quantize_f32(&data, &opts).unwrap();
        let out = dequantize_f32(&packed, 8192, &opts).unwrap();
        assert_eq!(out, data);
    }

    // -- Format detection ---------------------------------------------------

    #[test]
    fn detect_format_from_tensor_qk256() {
        let opts = QuantizeOpts::qk256();
        let data = vec![1.0f32; 256];
        let packed = quantize_f32(&data, &opts).unwrap();
        let detected = BlockFormat::detect(256, packed.len());
        assert_eq!(detected, Some(BlockFormat::Qk256));
    }

    #[test]
    fn detect_format_from_tensor_bitnet32() {
        let opts = QuantizeOpts::bitnet32();
        let data = vec![1.0f32; 32];
        let packed = quantize_f32(&data, &opts).unwrap();
        let detected = BlockFormat::detect(32, packed.len());
        assert_eq!(detected, Some(BlockFormat::BitNet32F16));
    }

    // -- f16 helpers --------------------------------------------------------

    #[test]
    fn f16_simple_roundtrip() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 5.0, 100.0] {
            let bits = f32_to_f16(v);
            let back = f16_to_f32(bits);
            assert!(
                (back - v).abs() <= v.abs().mul_add(1e-3, 1e-7),
                "f16 roundtrip failed for {v}: got {back}",
            );
        }
    }

    #[test]
    fn snap_ternary_values() {
        assert_eq!(snap_ternary(-2.0), -1);
        assert_eq!(snap_ternary(-0.5), -1);
        assert_eq!(snap_ternary(-0.49), 0);
        assert_eq!(snap_ternary(0.0), 0);
        assert_eq!(snap_ternary(0.49), 0);
        assert_eq!(snap_ternary(0.5), 1);
        assert_eq!(snap_ternary(2.0), 1);
    }
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn qk256_roundtrip_arbitrary(
            values in proptest::collection::vec(-1.0f32..=1.0, 256)
        ) {
            let opts = QuantizeOpts::qk256();
            let packed = quantize_f32(&values, &opts).unwrap();
            let out = dequantize_f32(&packed, 256, &opts).unwrap();
            for &v in &out {
                prop_assert!(v == -1.0 || v == 0.0 || v == 1.0);
            }
        }

        #[test]
        fn bitnet32_roundtrip_arbitrary(
            values in proptest::collection::vec(-10.0f32..=10.0, 32)
        ) {
            let opts = QuantizeOpts::bitnet32();
            let packed = quantize_f32(&values, &opts).unwrap();
            let out = dequantize_f32(&packed, 32, &opts).unwrap();
            let scale_bits = u16::from_le_bytes([packed[8], packed[9]]);
            let scale = f16_to_f32(scale_bits);
            for &v in &out {
                let normalised = if scale > 0.0 { v / scale } else { 0.0 };
                prop_assert!(
                    (normalised - (-1.0)).abs() < 0.01
                    || normalised.abs() < 0.01
                    || (normalised - 1.0).abs() < 0.01,
                    "unexpected dequantized value {v} (normalised {normalised}, scale {scale})"
                );
            }
        }

        #[test]
        fn pack_unpack_roundtrip(
            values in proptest::collection::vec(-1i8..=1, 1..=1024)
        ) {
            let packed = crate::pack::pack_i2s(&values).unwrap();
            let unpacked = crate::pack::unpack_i2s(&packed, values.len()).unwrap();
            prop_assert_eq!(unpacked, values);
        }

        #[test]
        fn ternary_snap_idempotent(v in -10.0f32..=10.0) {
            let snapped = snap_ternary(v);
            prop_assert!((-1..=1).contains(&snapped));
            let resnapped = snap_ternary(f32::from(snapped));
            prop_assert_eq!(snapped, resnapped);
        }

        #[test]
        fn quantize_output_length_qk256(
            n_blocks in 1usize..=16
        ) {
            let opts = QuantizeOpts::qk256();
            let data = vec![0.0f32; n_blocks * 256];
            let packed = quantize_f32(&data, &opts).unwrap();
            prop_assert_eq!(packed.len(), n_blocks * 64);
        }

        #[test]
        fn quantize_output_length_bitnet32(
            n_blocks in 1usize..=128
        ) {
            let opts = QuantizeOpts::bitnet32();
            let data = vec![0.0f32; n_blocks * 32];
            let packed = quantize_f32(&data, &opts).unwrap();
            prop_assert_eq!(packed.len(), n_blocks * 10);
        }

        #[test]
        fn batch_all_succeed_when_aligned(
            n_tensors in 1usize..=8,
            n_blocks in 1usize..=4
        ) {
            let opts = QuantizeOpts::qk256();
            let data = vec![0.0f32; n_blocks * 256];
            let tensors: Vec<&[f32]> = (0..n_tensors).map(|_| data.as_slice()).collect();
            let results = quantize_batch(&tensors, &opts);
            for r in &results {
                prop_assert!(r.is_ok());
            }
        }
    }
}
