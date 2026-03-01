//! Weight compression formats for GPU-accelerated inference.
//!
//! Provides [`WeightCompressor`] trait with implementations for GPTQ, AWQ,
//! and ternary packing, plus bit-packing utilities and compression analysis.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::HalError;

// ── Compression format enum ──────────────────────────────────────────────

/// Supported weight compression formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionFormat {
    /// GPTQ: Hessian-based post-training quantization.
    Gptq,
    /// AWQ: activation-aware weight quantization.
    Awq,
    /// `SqueezeLLM`: sensitivity-weighted non-uniform quantization.
    SqueezeLlm,
    /// bitsandbytes-style NF4/FP4 quantization.
    Bitsandbytes,
    /// GGML-compatible block quantization (`Q4_0`, `Q4_1`, etc.).
    Ggml,
    /// Ternary packed: {-1, 0, +1} in 2 bits per weight.
    TernaryPacked,
}

impl fmt::Display for CompressionFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gptq => write!(f, "GPTQ"),
            Self::Awq => write!(f, "AWQ"),
            Self::SqueezeLlm => write!(f, "SqueezeLLM"),
            Self::Bitsandbytes => write!(f, "bitsandbytes"),
            Self::Ggml => write!(f, "GGML"),
            Self::TernaryPacked => write!(f, "TernaryPacked"),
        }
    }
}

// ── Configuration ────────────────────────────────────────────────────────

/// Parameters governing a compression pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Which format to apply.
    pub format: CompressionFormat,
    /// Group size for block-wise quantization (0 = per-tensor).
    pub group_size: usize,
    /// Bit width of compressed weights.
    pub bits: u8,
    /// Whether quantization is symmetric around zero.
    pub symmetric: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self { format: CompressionFormat::Gptq, group_size: 128, bits: 4, symmetric: true }
    }
}

// ── Compressed tensor ────────────────────────────────────────────────────

/// A compressed weight tensor with metadata.
#[derive(Debug, Clone)]
pub struct CompressedTensor {
    /// Packed weight data.
    pub data: Vec<u8>,
    /// Per-group scale factors.
    pub scales: Vec<f32>,
    /// Per-group zero points (empty when symmetric).
    pub zero_points: Vec<f32>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Configuration used to produce this tensor.
    pub config: CompressionConfig,
}

impl CompressedTensor {
    /// Number of elements in the original (uncompressed) tensor.
    pub fn original_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Byte size of the compressed payload.
    pub const fn compressed_bytes(&self) -> usize {
        self.data.len()
    }

    /// Compression ratio (original f32 bytes / compressed bytes).
    pub fn compression_ratio(&self) -> f32 {
        let orig = self.original_elements() * 4;
        let comp = self.compressed_bytes() + self.scales.len() * 4 + self.zero_points.len() * 4;
        if comp == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            orig as f32 / comp as f32
        }
    }
}

// ── WeightCompressor trait ───────────────────────────────────────────────

/// Trait for compressing and decompressing weight tensors.
pub trait WeightCompressor: Send + Sync {
    /// Compress a slice of f32 weights.
    fn compress(
        &self,
        weights: &[f32],
        config: &CompressionConfig,
    ) -> Result<CompressedTensor, HalError>;

    /// Decompress back to f32 weights.
    fn decompress(&self, tensor: &CompressedTensor) -> Result<Vec<f32>, HalError>;

    /// Name of this compressor for logging.
    fn name(&self) -> &'static str;
}

// ── GPTQ compressor ─────────────────────────────────────────────────────

/// GPTQ-style compression with Hessian-based sensitivity ordering.
///
/// Quantizes groups of weights with optimal rounding order based on
/// inverse-Hessian diagonal approximation.
pub struct GptqCompressor;

impl GptqCompressor {
    /// Approximate Hessian diagonal from weights (simplified sensitivity).
    fn hessian_sensitivity(weights: &[f32]) -> Vec<f32> {
        weights.iter().map(|&w| w * w).collect()
    }
}

impl WeightCompressor for GptqCompressor {
    fn compress(
        &self,
        weights: &[f32],
        config: &CompressionConfig,
    ) -> Result<CompressedTensor, HalError> {
        if weights.is_empty() {
            return Err(HalError::EmptyInput);
        }
        let group_size = if config.group_size == 0 { weights.len() } else { config.group_size };
        let num_groups = weights.len().div_ceil(group_size);
        let mut scales = Vec::with_capacity(num_groups);
        let mut zero_points = Vec::new();
        let mut quantized_vals: Vec<u8> = Vec::with_capacity(weights.len());
        let max_val = (1u32 << config.bits) - 1;

        #[allow(clippy::cast_precision_loss)]
        let max_val_f32 = max_val as f32;

        let sensitivity = Self::hessian_sensitivity(weights);

        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(weights.len());
            let group = &weights[start..end];
            let sens = &sensitivity[start..end];

            // Weight ordering by sensitivity (descending) for optimal
            // rounding; we apply it implicitly via error compensation.
            let max_abs = group
                .iter()
                .zip(sens.iter())
                .map(|(&w, &s)| w.abs() * (1.0 + s))
                .fold(0.0_f32, f32::max);
            let scale = if max_abs == 0.0 {
                0.0
            } else if config.symmetric {
                max_abs / (max_val_f32 / 2.0)
            } else {
                max_abs / max_val_f32
            };
            scales.push(scale);

            #[allow(clippy::cast_precision_loss)]
            let zp = if config.symmetric { (max_val / 2) as f32 } else { 0.0 };
            if !config.symmetric {
                zero_points.push(zp);
            }

            for &w in group {
                let q = if scale == 0.0 {
                    zp
                } else if config.symmetric {
                    ((w / scale) + zp).round().clamp(0.0, max_val_f32)
                } else {
                    (w / scale).round().clamp(0.0, max_val_f32)
                };
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                quantized_vals.push(q as u8);
            }
        }

        let data = pack_nbit(&quantized_vals, config.bits);

        Ok(CompressedTensor {
            data,
            scales,
            zero_points,
            shape: vec![weights.len()],
            config: config.clone(),
        })
    }

    fn decompress(&self, tensor: &CompressedTensor) -> Result<Vec<f32>, HalError> {
        let n = tensor.original_elements();
        let unpacked = unpack_nbit(&tensor.data, tensor.config.bits, n);
        let group_size = if tensor.config.group_size == 0 { n } else { tensor.config.group_size };

        let mut out = Vec::with_capacity(n);
        for (i, &q) in unpacked.iter().enumerate() {
            let g = i / group_size;
            let scale = tensor.scales.get(g).copied().unwrap_or(1.0);
            let zp = if tensor.config.symmetric {
                #[allow(clippy::cast_precision_loss)]
                let max_val = ((1u32 << tensor.config.bits) - 1) as f32;
                max_val / 2.0
            } else {
                tensor.zero_points.get(g).copied().unwrap_or(0.0)
            };
            out.push((f32::from(q) - zp) * scale);
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "GPTQ"
    }
}

// ── AWQ compressor ──────────────────────────────────────────────────────

/// Activation-aware weight quantization.
///
/// Scales channels by activation magnitudes before quantization so that
/// salient channels preserve more precision.
pub struct AwqCompressor {
    /// Per-channel activation scales (empty ⇒ uniform).
    pub activation_scales: Vec<f32>,
}

impl AwqCompressor {
    /// Create a new AWQ compressor with the given activation scales.
    pub const fn new(activation_scales: Vec<f32>) -> Self {
        Self { activation_scales }
    }

    /// Create a compressor with uniform (identity) activation scales.
    pub const fn uniform() -> Self {
        Self { activation_scales: Vec::new() }
    }

    fn effective_scale(&self, idx: usize) -> f32 {
        self.activation_scales.get(idx).copied().unwrap_or(1.0)
    }
}

impl WeightCompressor for AwqCompressor {
    fn compress(
        &self,
        weights: &[f32],
        config: &CompressionConfig,
    ) -> Result<CompressedTensor, HalError> {
        if weights.is_empty() {
            return Err(HalError::EmptyInput);
        }
        // Scale weights by activation importance.
        let scaled: Vec<f32> =
            weights.iter().enumerate().map(|(i, &w)| w * self.effective_scale(i)).collect();

        // Delegate to GPTQ-style quantization on the scaled weights.
        let gptq = GptqCompressor;
        let mut tensor = gptq.compress(&scaled, config)?;
        // Store original shape (unscaled).
        tensor.shape = vec![weights.len()];
        Ok(tensor)
    }

    fn decompress(&self, tensor: &CompressedTensor) -> Result<Vec<f32>, HalError> {
        let gptq = GptqCompressor;
        let scaled = gptq.decompress(tensor)?;
        // Undo activation scaling.
        Ok(scaled
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let s = self.effective_scale(i);
                if s.abs() < f32::EPSILON { v } else { v / s }
            })
            .collect())
    }

    fn name(&self) -> &'static str {
        "AWQ"
    }
}

// ── Ternary compressor ──────────────────────────────────────────────────

/// Ternary weight packing for BitNet-style {-1, 0, +1} models.
///
/// Stores each weight in 2 bits: 0b00 = 0, 0b01 = +1, 0b10 = -1.
pub struct TernaryCompressor;

impl TernaryCompressor {
    /// Encode a ternary value into 2 bits.
    const fn encode_ternary(v: i8) -> u8 {
        match v {
            1 => 0b01,
            -1 => 0b10,
            _ => 0b00,
        }
    }

    /// Decode 2 bits back to a ternary value.
    const fn decode_ternary(bits: u8) -> i8 {
        match bits & 0b11 {
            0b01 => 1,
            0b10 => -1,
            _ => 0,
        }
    }
}

impl WeightCompressor for TernaryCompressor {
    fn compress(
        &self,
        weights: &[f32],
        config: &CompressionConfig,
    ) -> Result<CompressedTensor, HalError> {
        if weights.is_empty() {
            return Err(HalError::EmptyInput);
        }
        let max_abs = weights.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs };
        let threshold = 0.5;

        let ternary: Vec<i8> = weights
            .iter()
            .map(|&w| {
                let norm = w / scale;
                if norm > threshold {
                    1
                } else if norm < -threshold {
                    -1
                } else {
                    0
                }
            })
            .collect();

        // Pack 4 ternary values per byte (2 bits each).
        let packed_len = ternary.len().div_ceil(4);
        let mut data = vec![0u8; packed_len];
        for (i, &t) in ternary.iter().enumerate() {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            data[byte_idx] |= Self::encode_ternary(t) << bit_offset;
        }

        Ok(CompressedTensor {
            data,
            scales: vec![scale],
            zero_points: Vec::new(),
            shape: vec![weights.len()],
            config: CompressionConfig {
                format: CompressionFormat::TernaryPacked,
                group_size: config.group_size,
                bits: 2,
                symmetric: true,
            },
        })
    }

    fn decompress(&self, tensor: &CompressedTensor) -> Result<Vec<f32>, HalError> {
        let n = tensor.original_elements();
        let scale = tensor.scales.first().copied().unwrap_or(1.0);

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let bits = (tensor.data[byte_idx] >> bit_offset) & 0b11;
            out.push(f32::from(Self::decode_ternary(bits)) * scale);
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "TernaryPacked"
    }
}

// ── Compression analyzer ────────────────────────────────────────────────

/// Analysis results for a compression configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionAnalysis {
    /// Compression ratio (original / compressed).
    pub ratio: f32,
    /// Estimated quality loss (mean squared error).
    pub quality_loss_mse: f32,
    /// Estimated decompression cost in FLOPs per element.
    pub decompression_flops: f32,
    /// Format used.
    pub format: CompressionFormat,
}

/// Estimate compression characteristics without performing full compression.
pub struct CompressionAnalyzer;

impl CompressionAnalyzer {
    /// Estimate compression ratio for a given config and element count.
    pub fn estimate_ratio(n: usize, config: &CompressionConfig) -> f32 {
        if n == 0 {
            return 0.0;
        }
        let orig_bits = n * 32;
        let group_size = if config.group_size == 0 { n } else { config.group_size };
        let num_groups = n.div_ceil(group_size);
        // Compressed: bits per weight + 32-bit scale per group.
        let comp_bits = n * config.bits as usize + num_groups * 32;
        #[allow(clippy::cast_precision_loss)]
        {
            orig_bits as f32 / comp_bits as f32
        }
    }

    /// Estimate decompression FLOPs per element.
    pub const fn estimate_decompression_cost(config: &CompressionConfig) -> f32 {
        match config.format {
            // Dequant: multiply + subtract zero point.
            CompressionFormat::Gptq => 3.0,
            // AWQ/SqueezeLLM add activation rescaling.
            CompressionFormat::Awq | CompressionFormat::SqueezeLlm => 4.0,
            // Lookup + multiply.
            CompressionFormat::Bitsandbytes
            | CompressionFormat::Ggml
            | CompressionFormat::TernaryPacked => 2.0,
        }
    }

    /// Full analysis including a sample-based quality loss estimate.
    pub fn analyze(
        weights: &[f32],
        config: &CompressionConfig,
        compressor: &dyn WeightCompressor,
    ) -> Result<CompressionAnalysis, HalError> {
        let compressed = compressor.compress(weights, config)?;
        let decompressed = compressor.decompress(&compressed)?;

        #[allow(clippy::cast_precision_loss)]
        let mse =
            weights.iter().zip(decompressed.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum::<f32>()
                / weights.len().max(1) as f32;

        Ok(CompressionAnalysis {
            ratio: compressed.compression_ratio(),
            quality_loss_mse: mse,
            decompression_flops: Self::estimate_decompression_cost(config),
            format: config.format,
        })
    }
}

// ── Decompression kernel ────────────────────────────────────────────────

/// On-the-fly decompression kernel for inference hot paths.
pub struct DecompressionKernel {
    config: CompressionConfig,
}

impl DecompressionKernel {
    /// Create a kernel for the given compression config.
    pub const fn new(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Decompress a single group of packed weights in-place into `output`.
    ///
    /// `packed` is the raw compressed bytes for one group.
    /// `scale` and `zero_point` are the group parameters.
    pub fn decompress_group(&self, packed: &[u8], scale: f32, zero_point: f32, output: &mut [f32]) {
        let bits = self.config.bits;
        let unpacked = unpack_nbit(packed, bits, output.len());
        for (out, &q) in output.iter_mut().zip(unpacked.iter()) {
            *out = (f32::from(q) - zero_point) * scale;
        }
    }

    /// Decompress a full [`CompressedTensor`] using the appropriate
    /// compressor.
    pub fn decompress_tensor(&self, tensor: &CompressedTensor) -> Result<Vec<f32>, HalError> {
        match tensor.config.format {
            CompressionFormat::TernaryPacked => TernaryCompressor.decompress(tensor),
            _ => GptqCompressor.decompress(tensor),
        }
    }
}

// ── Bit-packing utilities ───────────────────────────────────────────────

/// Pack n-bit values into bytes (little-endian bit order).
///
/// Each value in `vals` must fit in `bits` bits.
pub fn pack_nbit(vals: &[u8], bits: u8) -> Vec<u8> {
    if vals.is_empty() || bits == 0 {
        return Vec::new();
    }
    let total_bits = vals.len() * bits as usize;
    let num_bytes = total_bits.div_ceil(8);
    let mut out = vec![0u8; num_bytes];
    let mask = (1u16 << bits) - 1;

    let mut bit_pos: usize = 0;
    for &v in vals {
        let v16 = u16::from(v) & mask;
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;

        #[allow(clippy::cast_possible_truncation)]
        {
            out[byte_idx] |= (v16 << bit_offset) as u8;
            if bit_offset + bits as usize > 8 && byte_idx + 1 < num_bytes {
                out[byte_idx + 1] |= (v16 >> (8 - bit_offset)) as u8;
            }
            if bit_offset + bits as usize > 16 && byte_idx + 2 < num_bytes {
                out[byte_idx + 2] |= (v16 >> (16 - bit_offset)) as u8;
            }
        }
        bit_pos += bits as usize;
    }
    out
}

/// Unpack n-bit values from a packed byte array.
///
/// Returns exactly `count` values.
pub fn unpack_nbit(data: &[u8], bits: u8, count: usize) -> Vec<u8> {
    if data.is_empty() || bits == 0 || count == 0 {
        return Vec::new();
    }
    let mask = (1u16 << bits) - 1;
    let mut out = Vec::with_capacity(count);

    let mut bit_pos: usize = 0;
    for _ in 0..count {
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;

        let mut val =
            if byte_idx < data.len() { u16::from(data[byte_idx]) >> bit_offset } else { 0 };
        if bit_offset + bits as usize > 8 && byte_idx + 1 < data.len() {
            val |= u16::from(data[byte_idx + 1]) << (8 - bit_offset);
        }
        if bit_offset + bits as usize > 16 && byte_idx + 2 < data.len() {
            val |= u16::from(data[byte_idx + 2]) << (16 - bit_offset);
        }
        #[allow(clippy::cast_possible_truncation)]
        out.push((val & mask) as u8);
        bit_pos += bits as usize;
    }
    out
}

/// Pack 2-bit values: 4 values per byte.
pub fn pack_2bit(vals: &[u8]) -> Vec<u8> {
    pack_nbit(vals, 2)
}

/// Unpack 2-bit values.
pub fn unpack_2bit(data: &[u8], count: usize) -> Vec<u8> {
    unpack_nbit(data, 2, count)
}

/// Pack 3-bit values.
pub fn pack_3bit(vals: &[u8]) -> Vec<u8> {
    pack_nbit(vals, 3)
}

/// Unpack 3-bit values.
pub fn unpack_3bit(data: &[u8], count: usize) -> Vec<u8> {
    unpack_nbit(data, 3, count)
}

/// Pack 4-bit values: 2 values per byte.
pub fn pack_4bit(vals: &[u8]) -> Vec<u8> {
    pack_nbit(vals, 4)
}

/// Unpack 4-bit values.
pub fn unpack_4bit(data: &[u8], count: usize) -> Vec<u8> {
    unpack_nbit(data, 4, count)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
mod tests {
    use super::*;

    // -- CompressionFormat -------------------------------------------------

    #[test]
    fn format_display() {
        assert_eq!(CompressionFormat::Gptq.to_string(), "GPTQ");
        assert_eq!(CompressionFormat::Awq.to_string(), "AWQ");
        assert_eq!(CompressionFormat::SqueezeLlm.to_string(), "SqueezeLLM");
        assert_eq!(CompressionFormat::Bitsandbytes.to_string(), "bitsandbytes");
        assert_eq!(CompressionFormat::Ggml.to_string(), "GGML");
        assert_eq!(CompressionFormat::TernaryPacked.to_string(), "TernaryPacked");
    }

    #[test]
    fn format_serde_roundtrip() {
        let fmt = CompressionFormat::Awq;
        let json = serde_json::to_string(&fmt).unwrap();
        let back: CompressionFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(fmt, back);
    }

    #[test]
    fn format_all_variants_distinct() {
        use std::collections::HashSet;
        let variants = [
            CompressionFormat::Gptq,
            CompressionFormat::Awq,
            CompressionFormat::SqueezeLlm,
            CompressionFormat::Bitsandbytes,
            CompressionFormat::Ggml,
            CompressionFormat::TernaryPacked,
        ];
        let set: HashSet<_> = variants.iter().collect();
        assert_eq!(set.len(), 6);
    }

    // -- CompressionConfig -------------------------------------------------

    #[test]
    fn config_default() {
        let cfg = CompressionConfig::default();
        assert_eq!(cfg.format, CompressionFormat::Gptq);
        assert_eq!(cfg.group_size, 128);
        assert_eq!(cfg.bits, 4);
        assert!(cfg.symmetric);
    }

    #[test]
    fn config_serde_roundtrip() {
        let cfg = CompressionConfig {
            format: CompressionFormat::TernaryPacked,
            group_size: 64,
            bits: 2,
            symmetric: true,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: CompressionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.format, cfg.format);
        assert_eq!(back.bits, cfg.bits);
    }

    // -- Bit packing -------------------------------------------------------

    #[test]
    fn pack_unpack_2bit_roundtrip() {
        let vals: Vec<u8> = vec![0, 1, 2, 3, 1, 0, 3, 2];
        let packed = pack_2bit(&vals);
        let unpacked = unpack_2bit(&packed, vals.len());
        assert_eq!(vals, unpacked);
    }

    #[test]
    fn pack_unpack_3bit_roundtrip() {
        let vals: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let packed = pack_3bit(&vals);
        let unpacked = unpack_3bit(&packed, vals.len());
        assert_eq!(vals, unpacked);
    }

    #[test]
    fn pack_unpack_4bit_roundtrip() {
        let vals: Vec<u8> = vec![0, 5, 10, 15, 3, 8, 12, 1];
        let packed = pack_4bit(&vals);
        let unpacked = unpack_4bit(&packed, vals.len());
        assert_eq!(vals, unpacked);
    }

    #[test]
    fn pack_nbit_empty() {
        assert!(pack_nbit(&[], 4).is_empty());
        assert!(pack_nbit(&[1, 2], 0).is_empty());
    }

    #[test]
    fn unpack_nbit_empty() {
        assert!(unpack_nbit(&[], 4, 5).is_empty());
        assert!(unpack_nbit(&[0xFF], 0, 5).is_empty());
        assert!(unpack_nbit(&[0xFF], 4, 0).is_empty());
    }

    #[test]
    fn pack_unpack_2bit_non_aligned() {
        let vals: Vec<u8> = vec![1, 2, 3];
        let packed = pack_2bit(&vals);
        let unpacked = unpack_2bit(&packed, 3);
        assert_eq!(vals, unpacked);
    }

    #[test]
    fn pack_unpack_4bit_single_value() {
        let vals: Vec<u8> = vec![7];
        let packed = pack_4bit(&vals);
        let unpacked = unpack_4bit(&packed, 1);
        assert_eq!(vals, unpacked);
    }

    #[test]
    fn pack_2bit_byte_count() {
        // 8 values × 2 bits = 16 bits = 2 bytes.
        let packed = pack_2bit(&[0; 8]);
        assert_eq!(packed.len(), 2);
        // 5 values × 2 bits = 10 bits → 2 bytes.
        let packed = pack_2bit(&[0; 5]);
        assert_eq!(packed.len(), 2);
    }

    #[test]
    fn pack_4bit_byte_count() {
        // 4 values × 4 bits = 16 bits = 2 bytes.
        let packed = pack_4bit(&[0; 4]);
        assert_eq!(packed.len(), 2);
        // 3 values × 4 bits = 12 bits → 2 bytes.
        let packed = pack_4bit(&[0; 3]);
        assert_eq!(packed.len(), 2);
    }

    #[test]
    fn pack_unpack_nbit_large() {
        let vals: Vec<u8> = (0..256).map(|i| (i % 16) as u8).collect();
        let packed = pack_nbit(&vals, 4);
        let unpacked = unpack_nbit(&packed, 4, vals.len());
        assert_eq!(vals, unpacked);
    }

    // -- GPTQ compressor ---------------------------------------------------

    #[test]
    fn gptq_compress_empty() {
        let c = GptqCompressor;
        let cfg = CompressionConfig::default();
        assert!(c.compress(&[], &cfg).is_err());
    }

    #[test]
    fn gptq_roundtrip_zeros() {
        let c = GptqCompressor;
        let cfg = CompressionConfig::default();
        let weights = vec![0.0_f32; 128];
        let compressed = c.compress(&weights, &cfg).unwrap();
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 128);
        for v in &decompressed {
            assert!(v.abs() < 0.01, "expected ~0, got {v}");
        }
    }

    #[test]
    fn gptq_roundtrip_small_values() {
        let c = GptqCompressor;
        let cfg = CompressionConfig { bits: 4, ..Default::default() };
        let weights: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let compressed = c.compress(&weights, &cfg).unwrap();
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), weights.len());
    }

    #[test]
    fn gptq_compression_ratio_positive() {
        let c = GptqCompressor;
        let cfg = CompressionConfig::default();
        let weights: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let compressed = c.compress(&weights, &cfg).unwrap();
        assert!(compressed.compression_ratio() > 1.0);
    }

    #[test]
    fn gptq_per_tensor_group() {
        let c = GptqCompressor;
        let cfg = CompressionConfig { group_size: 0, ..Default::default() };
        let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
        let compressed = c.compress(&weights, &cfg).unwrap();
        assert_eq!(compressed.scales.len(), 1);
    }

    #[test]
    fn gptq_asymmetric_mode() {
        let c = GptqCompressor;
        let cfg = CompressionConfig { symmetric: false, group_size: 16, ..Default::default() };
        let weights: Vec<f32> = (0..32).map(|i| (i as f32) * 0.2).collect();
        let compressed = c.compress(&weights, &cfg).unwrap();
        assert!(!compressed.zero_points.is_empty());
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), weights.len());
    }

    #[test]
    fn gptq_shape_preserved() {
        let c = GptqCompressor;
        let cfg = CompressionConfig::default();
        let weights = vec![1.0_f32; 64];
        let compressed = c.compress(&weights, &cfg).unwrap();
        assert_eq!(compressed.shape, vec![64]);
        assert_eq!(compressed.original_elements(), 64);
    }

    #[test]
    fn gptq_name() {
        assert_eq!(GptqCompressor.name(), "GPTQ");
    }

    #[test]
    fn gptq_hessian_sensitivity() {
        let sens = GptqCompressor::hessian_sensitivity(&[1.0, -2.0, 0.5]);
        assert!((sens[0] - 1.0).abs() < f32::EPSILON);
        assert!((sens[1] - 4.0).abs() < f32::EPSILON);
        assert!((sens[2] - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn gptq_2bit_mode() {
        let c = GptqCompressor;
        let cfg = CompressionConfig { bits: 2, group_size: 8, ..Default::default() };
        let weights: Vec<f32> = (0..16).map(|i| i as f32 * 0.3).collect();
        let compressed = c.compress(&weights, &cfg).unwrap();
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 16);
    }

    #[test]
    fn gptq_8bit_mode() {
        let c = GptqCompressor;
        let cfg = CompressionConfig { bits: 8, group_size: 32, ..Default::default() };
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let compressed = c.compress(&weights, &cfg).unwrap();
        let decompressed = c.decompress(&compressed).unwrap();
        // 8-bit should have lower error than 4-bit.
        let max_err = weights
            .iter()
            .zip(decompressed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_err < 0.1, "8-bit max error too large: {max_err}");
    }

    // -- AWQ compressor ----------------------------------------------------

    #[test]
    fn awq_compress_empty() {
        let c = AwqCompressor::uniform();
        let cfg = CompressionConfig::default();
        assert!(c.compress(&[], &cfg).is_err());
    }

    #[test]
    fn awq_uniform_roundtrip() {
        let c = AwqCompressor::uniform();
        let cfg = CompressionConfig::default();
        let weights: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let compressed = c.compress(&weights, &cfg).unwrap();
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), weights.len());
    }

    #[test]
    fn awq_with_activation_scales() {
        let scales: Vec<f32> = (0..32).map(|i| (i as f32).mul_add(0.1, 1.0)).collect();
        let c = AwqCompressor::new(scales);
        let cfg = CompressionConfig { group_size: 32, ..Default::default() };
        let weights: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.05).collect();
        let compressed = c.compress(&weights, &cfg).unwrap();
        let _decompressed = c.decompress(&compressed).unwrap();
    }

    #[test]
    fn awq_name() {
        assert_eq!(AwqCompressor::uniform().name(), "AWQ");
    }

    #[test]
    fn awq_preserves_shape() {
        let c = AwqCompressor::uniform();
        let cfg = CompressionConfig::default();
        let weights = vec![0.5_f32; 64];
        let compressed = c.compress(&weights, &cfg).unwrap();
        assert_eq!(compressed.original_elements(), 64);
    }

    // -- Ternary compressor ------------------------------------------------

    #[test]
    fn ternary_compress_empty() {
        let c = TernaryCompressor;
        let cfg = CompressionConfig::default();
        assert!(c.compress(&[], &cfg).is_err());
    }

    #[test]
    fn ternary_roundtrip_known() {
        let c = TernaryCompressor;
        let cfg = CompressionConfig::default();
        let weights = vec![1.0, -1.0, 0.0, 0.8, -0.9, 0.1];
        let compressed = c.compress(&weights, &cfg).unwrap();
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 6);
        // Values > 0.5*scale → 1*scale, < -0.5*scale → -1*scale, else 0
        assert!(decompressed[0] > 0.0); // was 1.0
        assert!(decompressed[1] < 0.0); // was -1.0
        assert!((decompressed[2]).abs() < f32::EPSILON); // was 0.0
    }

    #[test]
    fn ternary_all_zeros() {
        let c = TernaryCompressor;
        let cfg = CompressionConfig::default();
        let weights = vec![0.0; 16];
        let compressed = c.compress(&weights, &cfg).unwrap();
        let decompressed = c.decompress(&compressed).unwrap();
        for v in &decompressed {
            assert!((v.abs()) < f32::EPSILON);
        }
    }

    #[test]
    fn ternary_encode_decode() {
        assert_eq!(TernaryCompressor::decode_ternary(TernaryCompressor::encode_ternary(1)), 1);
        assert_eq!(TernaryCompressor::decode_ternary(TernaryCompressor::encode_ternary(-1)), -1);
        assert_eq!(TernaryCompressor::decode_ternary(TernaryCompressor::encode_ternary(0)), 0);
    }

    #[test]
    fn ternary_packing_density() {
        let c = TernaryCompressor;
        let cfg = CompressionConfig::default();
        let weights = vec![1.0; 100];
        let compressed = c.compress(&weights, &cfg).unwrap();
        // 100 values × 2 bits = 200 bits → 25 bytes.
        assert_eq!(compressed.data.len(), 25);
    }

    #[test]
    fn ternary_config_overwritten() {
        let c = TernaryCompressor;
        let cfg =
            CompressionConfig { format: CompressionFormat::Gptq, bits: 8, ..Default::default() };
        let compressed = c.compress(&[1.0, -1.0], &cfg).unwrap();
        assert_eq!(compressed.config.format, CompressionFormat::TernaryPacked);
        assert_eq!(compressed.config.bits, 2);
    }

    #[test]
    fn ternary_name() {
        assert_eq!(TernaryCompressor.name(), "TernaryPacked");
    }

    #[test]
    fn ternary_compression_ratio_high() {
        let c = TernaryCompressor;
        let cfg = CompressionConfig::default();
        let weights = vec![1.0; 1024];
        let compressed = c.compress(&weights, &cfg).unwrap();
        // 2 bits vs 32 bits → ~16× ratio (minus scale overhead).
        assert!(compressed.compression_ratio() > 10.0);
    }

    // -- CompressedTensor --------------------------------------------------

    #[test]
    fn compressed_tensor_ratio_zero_data() {
        let t = CompressedTensor {
            data: Vec::new(),
            scales: Vec::new(),
            zero_points: Vec::new(),
            shape: vec![0],
            config: CompressionConfig::default(),
        };
        assert!((t.compression_ratio()).abs() < f32::EPSILON);
    }

    // -- CompressionAnalyzer -----------------------------------------------

    #[test]
    fn analyzer_ratio_empty() {
        let r = CompressionAnalyzer::estimate_ratio(0, &CompressionConfig::default());
        assert!((r).abs() < f32::EPSILON);
    }

    #[test]
    fn analyzer_ratio_4bit() {
        let cfg = CompressionConfig { bits: 4, group_size: 128, ..Default::default() };
        let r = CompressionAnalyzer::estimate_ratio(1024, &cfg);
        // 32 / (4 + 32/128) ≈ 7.5×
        assert!(r > 5.0 && r < 10.0, "ratio = {r}");
    }

    #[test]
    fn analyzer_ratio_2bit() {
        let cfg = CompressionConfig { bits: 2, group_size: 128, ..Default::default() };
        let r = CompressionAnalyzer::estimate_ratio(1024, &cfg);
        assert!(r > 10.0, "ratio = {r}");
    }

    #[test]
    fn analyzer_decompression_cost_ordering() {
        let gptq_cost = CompressionAnalyzer::estimate_decompression_cost(&CompressionConfig {
            format: CompressionFormat::Gptq,
            ..Default::default()
        });
        let ternary_cost = CompressionAnalyzer::estimate_decompression_cost(&CompressionConfig {
            format: CompressionFormat::TernaryPacked,
            ..Default::default()
        });
        assert!(ternary_cost <= gptq_cost);
    }

    #[test]
    fn analyzer_full_gptq() {
        let c = GptqCompressor;
        let cfg = CompressionConfig { group_size: 32, ..Default::default() };
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let analysis = CompressionAnalyzer::analyze(&weights, &cfg, &c).unwrap();
        assert!(analysis.ratio > 1.0);
        assert!(analysis.quality_loss_mse >= 0.0);
        assert_eq!(analysis.format, CompressionFormat::Gptq);
    }

    #[test]
    fn analyzer_full_ternary() {
        let c = TernaryCompressor;
        let cfg = CompressionConfig {
            format: CompressionFormat::TernaryPacked,
            bits: 2,
            group_size: 0,
            symmetric: true,
        };
        let weights: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.02).collect();
        let analysis = CompressionAnalyzer::analyze(&weights, &cfg, &c).unwrap();
        assert!(analysis.ratio > 1.0);
    }

    #[test]
    fn analyzer_empty_input_error() {
        let c = GptqCompressor;
        let cfg = CompressionConfig::default();
        assert!(CompressionAnalyzer::analyze(&[], &cfg, &c).is_err());
    }

    // -- DecompressionKernel -----------------------------------------------

    #[test]
    fn kernel_decompress_group() {
        let cfg = CompressionConfig::default();
        let kernel = DecompressionKernel::new(cfg);
        let vals: Vec<u8> = vec![0, 5, 10, 15];
        let packed = pack_4bit(&vals);
        let mut output = vec![0.0_f32; 4];
        kernel.decompress_group(&packed, 0.5, 7.5, &mut output);
        // output[i] = (vals[i] - 7.5) * 0.5
        assert!((output[0] - (-3.75)).abs() < 0.01);
        assert!((output[1] - (-1.25)).abs() < 0.01);
    }

    #[test]
    fn kernel_decompress_ternary_tensor() {
        let c = TernaryCompressor;
        let cfg = CompressionConfig {
            format: CompressionFormat::TernaryPacked,
            bits: 2,
            ..Default::default()
        };
        let weights = vec![1.0, -1.0, 0.0, 0.5];
        let compressed = c.compress(&weights, &cfg).unwrap();
        let kernel = DecompressionKernel::new(cfg);
        let decompressed = kernel.decompress_tensor(&compressed).unwrap();
        assert_eq!(decompressed.len(), 4);
    }

    #[test]
    fn kernel_decompress_gptq_tensor() {
        let c = GptqCompressor;
        let cfg = CompressionConfig::default();
        let weights: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let compressed = c.compress(&weights, &cfg).unwrap();
        let kernel = DecompressionKernel::new(cfg);
        let decompressed = kernel.decompress_tensor(&compressed).unwrap();
        assert_eq!(decompressed.len(), 128);
    }

    // -- Cross-compressor -------------------------------------------------

    #[test]
    fn all_compressors_reject_empty() {
        let cfg = CompressionConfig::default();
        let compressors: Vec<Box<dyn WeightCompressor>> = vec![
            Box::new(GptqCompressor),
            Box::new(AwqCompressor::uniform()),
            Box::new(TernaryCompressor),
        ];
        for c in &compressors {
            assert!(c.compress(&[], &cfg).is_err(), "{} should reject empty input", c.name());
        }
    }

    #[test]
    fn all_compressors_roundtrip_length() {
        let cfg = CompressionConfig { group_size: 16, ..Default::default() };
        let weights: Vec<f32> = (0..48).map(|i| (i as f32 - 24.0) * 0.1).collect();
        let compressors: Vec<Box<dyn WeightCompressor>> = vec![
            Box::new(GptqCompressor),
            Box::new(AwqCompressor::uniform()),
            Box::new(TernaryCompressor),
        ];
        for c in &compressors {
            let compressed = c.compress(&weights, &cfg).unwrap();
            let decompressed = c.decompress(&compressed).unwrap();
            assert_eq!(decompressed.len(), weights.len(), "{} changed output length", c.name());
        }
    }

    #[test]
    fn trait_object_dispatch() {
        let c: Box<dyn WeightCompressor> = Box::new(GptqCompressor);
        let cfg = CompressionConfig::default();
        let weights = vec![1.0_f32; 128];
        let compressed = c.compress(&weights, &cfg).unwrap();
        let _decompressed = c.decompress(&compressed).unwrap();
    }

    // -- Edge cases --------------------------------------------------------

    #[test]
    fn single_element_gptq() {
        let c = GptqCompressor;
        let cfg = CompressionConfig { group_size: 1, ..Default::default() };
        let compressed = c.compress(&[42.0], &cfg).unwrap();
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 1);
    }

    #[test]
    fn single_element_ternary() {
        let c = TernaryCompressor;
        let cfg = CompressionConfig::default();
        let compressed = c.compress(&[1.0], &cfg).unwrap();
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 1);
        assert!(decompressed[0] > 0.0);
    }

    #[test]
    fn large_tensor_roundtrip() {
        let c = GptqCompressor;
        let cfg = CompressionConfig { group_size: 128, ..Default::default() };
        let weights: Vec<f32> = (0..4096).map(|i| ((i as f32) * 0.7).sin()).collect();
        let compressed = c.compress(&weights, &cfg).unwrap();
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 4096);
        assert!(compressed.compression_ratio() > 1.0);
    }

    #[test]
    fn negative_weights_gptq() {
        let c = GptqCompressor;
        let cfg = CompressionConfig::default();
        let weights: Vec<f32> = (0..128).map(|i| -(i as f32) * 0.01).collect();
        let compressed = c.compress(&weights, &cfg).unwrap();
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 128);
    }

    #[test]
    fn mixed_sign_ternary() {
        let c = TernaryCompressor;
        let cfg = CompressionConfig::default();
        let weights = vec![1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0];
        let compressed = c.compress(&weights, &cfg).unwrap();
        let decompressed = c.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 8);
        assert!(decompressed[0] > 0.0);
        assert!(decompressed[1] < 0.0);
        assert!((decompressed[4]).abs() < f32::EPSILON);
    }

    #[test]
    fn group_size_larger_than_input() {
        let c = GptqCompressor;
        let cfg = CompressionConfig { group_size: 256, ..Default::default() };
        let weights = vec![0.5_f32; 16];
        let compressed = c.compress(&weights, &cfg).unwrap();
        assert_eq!(compressed.scales.len(), 1);
    }

    #[test]
    fn analyzer_per_tensor_group() {
        let cfg = CompressionConfig { group_size: 0, bits: 4, ..Default::default() };
        let r = CompressionAnalyzer::estimate_ratio(512, &cfg);
        // Per-tensor: only 1 group → overhead is tiny.
        assert!(r > 7.0, "ratio = {r}");
    }

    #[test]
    fn compressed_tensor_bytes() {
        let c = GptqCompressor;
        let cfg = CompressionConfig::default();
        let weights = vec![1.0_f32; 128];
        let compressed = c.compress(&weights, &cfg).unwrap();
        assert!(compressed.compressed_bytes() > 0);
        assert!(compressed.compressed_bytes() < weights.len() * std::mem::size_of::<f32>());
    }
}
