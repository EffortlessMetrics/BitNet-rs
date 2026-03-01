//! GPU-accelerated quantization kernels for `BitNet` formats.
//!
//! Provides `OpenCL` kernel sources and CPU reference implementations for:
//! - **`I2_S`**: 2-bit integer symmetric pack/unpack (16 ternary values per
//!   32-bit word)
//! - **QK256**: 256-element block dequantization with f16 per-block scale
//! - **TL1**: Ternary lookup with 3-entry LUT
//! - **TL2**: Paired vectorized ternary lookup

/// `OpenCL` kernel source for `I2_S` pack/unpack.
pub const KERNEL_I2S: &str =
    include_str!("kernels/quantize_i2s.cl");

/// `OpenCL` kernel source for QK256 quantize/dequantize.
pub const KERNEL_QK256: &str =
    include_str!("kernels/quantize_qk256.cl");

/// `OpenCL` kernel source for TL1 dequantize.
pub const KERNEL_TL1: &str =
    include_str!("kernels/quantize_tl1.cl");

/// `OpenCL` kernel source for TL2 dequantize.
pub const KERNEL_TL2: &str =
    include_str!("kernels/quantize_tl2.cl");

// ── CPU reference implementations ──────────────────────────────────────

/// GPU quantization context.
///
/// Holds compiled kernel sources and provides methods that mirror the
/// `OpenCL` kernels but execute on the CPU for verification purposes.
pub struct GpuQuantizer {
    _private: (),
}

/// Errors produced by quantization operations.
#[derive(Debug, thiserror::Error)]
pub enum QuantError {
    #[error("input length {len} is not a multiple of {alignment}")]
    Alignment { len: usize, alignment: usize },

    #[error("value {value} is not a valid ternary (expected -1, 0, or 1)")]
    InvalidTernary { value: i32 },

    #[error("block count mismatch: expected {expected}, got {actual}")]
    BlockCount { expected: usize, actual: usize },

    #[error("lookup table must have exactly {expected} entries, got {actual}")]
    LutSize { expected: usize, actual: usize },
}

impl GpuQuantizer {
    /// Create a new [`GpuQuantizer`].
    pub const fn new() -> Self {
        Self { _private: () }
    }

    // ── I2_S ───────────────────────────────────────────────────────────

    /// Pack ternary values into 32-bit words (CPU reference).
    ///
    /// `values.len()` must be a multiple of 16.
    pub fn pack_i2s(&self, values: &[i32]) -> Result<Vec<u32>, QuantError> {
        if !values.len().is_multiple_of(16) {
            return Err(QuantError::Alignment {
                len: values.len(),
                alignment: 16,
            });
        }
        let mut packed = Vec::with_capacity(values.len() / 16);
        for chunk in values.chunks_exact(16) {
            let mut word: u32 = 0;
            for (i, &v) in chunk.iter().enumerate() {
                let bits: u32 = match v {
                    -1 => 2,
                    0 => 0,
                    1 => 1,
                    _ => {
                        return Err(QuantError::InvalidTernary {
                            value: v,
                        });
                    }
                };
                word |= (bits & 0x3) << (i * 2);
            }
            packed.push(word);
        }
        Ok(packed)
    }

    /// Unpack `I2_S` packed data to float (CPU reference).
    pub fn unpack_i2s(
        &self,
        packed: &[u32],
        scale: f32,
        count: usize,
    ) -> Vec<f32> {
        let mut output = Vec::with_capacity(count);
        for gid in 0..count {
            let word_idx = gid / 16;
            let bit_idx = (gid % 16) * 2;
            let bits = (packed[word_idx] >> bit_idx) & 0x3;
            #[allow(clippy::cast_precision_loss)]
            let val: f32 =
                if bits == 2 { -1.0 } else { bits as f32 };
            output.push(val * scale);
        }
        output
    }

    // ── QK256 ──────────────────────────────────────────────────────────

    /// Dequantize QK256 blocks to float (CPU reference).
    pub fn dequant_qk256(
        &self,
        packed_data: &[u32],
        scales: &[f32],
        num_blocks: usize,
    ) -> Result<Vec<f32>, QuantError> {
        if packed_data.len() != num_blocks * 16 {
            return Err(QuantError::BlockCount {
                expected: num_blocks * 16,
                actual: packed_data.len(),
            });
        }
        let mut output = vec![0.0_f32; num_blocks * 256];
        for (block_id, &scale) in scales[..num_blocks].iter().enumerate()
        {
            for local_id in 0..16_usize {
                let word_offset = block_id * 16 + local_id;
                let val_offset = block_id * 256 + local_id * 16;
                let word = packed_data[word_offset];
                for i in 0..16 {
                    let bits = (word >> (i * 2)) & 0x3;
                    #[allow(clippy::cast_precision_loss)]
                    let val: f32 =
                        if bits == 2 { -1.0 } else { bits as f32 };
                    output[val_offset + i] = val * scale;
                }
            }
        }
        Ok(output)
    }

    /// Quantize float data to QK256 format (CPU reference).
    ///
    /// Returns `(packed_words, scales)`. Input length must be a multiple
    /// of 256.
    pub fn quant_qk256(
        &self,
        input: &[f32],
    ) -> Result<(Vec<u32>, Vec<f32>), QuantError> {
        if !input.len().is_multiple_of(256) {
            return Err(QuantError::Alignment {
                len: input.len(),
                alignment: 256,
            });
        }
        let num_blocks = input.len() / 256;
        let mut packed = Vec::with_capacity(num_blocks * 16);
        let mut scales = Vec::with_capacity(num_blocks);

        for block_id in 0..num_blocks {
            let base = block_id * 256;
            let block = &input[base..base + 256];

            let scale = block
                .iter()
                .fold(0.0_f32, |acc, &v| acc.max(v.abs()));
            scales.push(scale);

            let inv_scale =
                if scale > 0.0 { 1.0 / scale } else { 0.0 };

            for chunk in block.chunks_exact(16) {
                let mut word: u32 = 0;
                for (i, &v) in chunk.iter().enumerate() {
                    let normalized = v * inv_scale;
                    let q = if normalized > 0.5 {
                        1
                    } else if normalized < -0.5 {
                        -1
                    } else {
                        0
                    };
                    #[allow(clippy::cast_sign_loss)]
                    let bits: u32 = if q == -1 { 2 } else { q as u32 };
                    word |= (bits & 0x3) << (i * 2);
                }
                packed.push(word);
            }
        }
        Ok((packed, scales))
    }

    // ── TL1 ────────────────────────────────────────────────────────────

    /// Dequantize TL1 packed data using a 3-entry LUT (CPU reference).
    pub fn dequant_tl1(
        &self,
        packed: &[u8],
        lut: &[f32; 3],
        count: usize,
    ) -> Vec<f32> {
        let mut output = Vec::with_capacity(count);
        for gid in 0..count {
            let byte_idx = gid / 4;
            let shift = (gid % 4) * 2;
            let bits =
                ((packed[byte_idx] >> shift) & 0x3) as usize;
            let idx = bits.min(2);
            output.push(lut[idx]);
        }
        output
    }

    // ── TL2 ────────────────────────────────────────────────────────────

    /// Dequantize TL2 packed data using paired LUT (CPU reference).
    ///
    /// `lut_pairs` holds `(x, y)` for each of the 3 codewords.
    pub fn dequant_tl2(
        &self,
        packed: &[u8],
        lut_pairs: &[[f32; 2]; 3],
        count: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; count];
        let num_pairs = count.div_ceil(2);
        for gid in 0..num_pairs {
            let byte_idx = gid / 2;
            let shift = (gid % 2) * 4;
            let bits0 =
                ((packed[byte_idx] >> shift) & 0x3) as usize;
            let bits1 =
                ((packed[byte_idx] >> (shift + 2)) & 0x3) as usize;
            let idx0 = bits0.min(2);
            let idx1 = bits1.min(2);
            output[gid * 2] = lut_pairs[idx0][0];
            if gid * 2 + 1 < count {
                output[gid * 2 + 1] = lut_pairs[idx1][1];
            }
        }
        output
    }
}

impl Default for GpuQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sources_are_non_empty() {
        assert!(!KERNEL_I2S.is_empty());
        assert!(!KERNEL_QK256.is_empty());
        assert!(!KERNEL_TL1.is_empty());
        assert!(!KERNEL_TL2.is_empty());
    }

    #[test]
    fn kernel_i2s_contains_entry_points() {
        assert!(KERNEL_I2S.contains("pack_i2s"));
        assert!(KERNEL_I2S.contains("unpack_i2s"));
    }

    #[test]
    fn kernel_qk256_contains_entry_points() {
        assert!(KERNEL_QK256.contains("dequant_qk256"));
        assert!(KERNEL_QK256.contains("quant_qk256"));
    }
}
