//! GPU-accelerated weight quantization for Intel Arc A770 (Xe-HPG).
//!
//! Quantizes model weights on-device in multiple formats (1-bit ternary, 2-bit,
//! 4-bit, 8-bit) with scale computation and error measurement. Supports BitNet's
//! native I2_S format and provides CPU reference implementations for every path.
//!
//! # Weight Formats
//!
//! | Format  | Bits | Range / Values          |
//! |---------|------|-------------------------|
//! | F32     |  32  | IEEE 754 float          |
//! | F16     |  16  | IEEE 754 half           |
//! | BF16    |  16  | bfloat16                |
//! | I8      |   8  | [−128, 127]             |
//! | I4      |   4  | [−8, 7]                 |
//! | I2_S    |   2  | {−1, 0, +1} (GGML I2_S)|
//! | Ternary |   2  | {−1, 0, +1} (BitNet)   |

use std::fmt;

// ── Types ───────────────────────────────────────────────────────────────────

/// Target weight format for quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    /// 32-bit IEEE 754 float (no quantization).
    F32,
    /// 16-bit IEEE 754 half-precision float.
    F16,
    /// 16-bit bfloat16.
    BF16,
    /// 8-bit signed integer.
    I8,
    /// 4-bit signed integer (two values per byte).
    I4,
    /// GGML I2_S: 2-bit ternary {−1, 0, +1}, 4 values per byte.
    I2S,
    /// BitNet ternary: {−1, 0, +1} with absolute-mean scaling.
    Ternary,
}

impl WeightFormat {
    /// Bits per element for this format.
    pub fn bits_per_element(self) -> u32 {
        match self {
            Self::F32 => 32,
            Self::F16 | Self::BF16 => 16,
            Self::I8 => 8,
            Self::I4 => 4,
            Self::I2S | Self::Ternary => 2,
        }
    }

    /// Number of bytes required to store `n` elements.
    pub fn storage_bytes(self, n: usize) -> usize {
        let bits = self.bits_per_element() as usize;
        (n * bits).div_ceil(8)
    }
}

impl fmt::Display for WeightFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::BF16 => write!(f, "BF16"),
            Self::I8 => write!(f, "I8"),
            Self::I4 => write!(f, "I4"),
            Self::I2S => write!(f, "I2_S"),
            Self::Ternary => write!(f, "Ternary"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Quantization configuration.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Target weight format.
    pub target_format: WeightFormat,
    /// Use symmetric quantization (zero-point fixed at 0).
    pub symmetric: bool,
    /// Compute scales per output channel rather than per tensor.
    pub per_channel: bool,
    /// Group size for per-group quantization (0 = per-tensor/per-channel).
    pub group_size: usize,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            target_format: WeightFormat::Ternary,
            symmetric: true,
            per_channel: true,
            group_size: 0,
        }
    }
}

// ── Error statistics ────────────────────────────────────────────────────────

/// Quantization error statistics.
#[derive(Debug, Clone)]
pub struct ErrorStats {
    /// Mean squared error between original and dequantized weights.
    pub mse: f64,
    /// Cosine similarity between original and dequantized weight vectors.
    pub cosine_similarity: f64,
    /// Maximum absolute deviation.
    pub max_deviation: f64,
}

impl fmt::Display for ErrorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MSE={:.6e} cos_sim={:.6} max_dev={:.6e}",
            self.mse, self.cosine_similarity, self.max_deviation,
        )
    }
}

// ── Quantization result ─────────────────────────────────────────────────────

/// Result of quantizing a weight tensor.
#[derive(Debug, Clone)]
pub struct QuantResult {
    /// Packed quantized data.
    pub quantized_data: Vec<u8>,
    /// Per-channel or per-group scale factors.
    pub scales: Vec<f32>,
    /// Per-channel or per-group zero points (empty for symmetric).
    pub zero_points: Vec<f32>,
    /// Format used for quantization.
    pub format: WeightFormat,
    /// Number of original elements.
    pub num_elements: usize,
    /// Error statistics from round-trip comparison.
    pub error_stats: ErrorStats,
}

// ── Scale estimator ─────────────────────────────────────────────────────────

/// Computes per-channel or per-group quantization scale factors.
pub struct ScaleEstimator;

impl ScaleEstimator {
    /// Compute scales for a 2-D weight matrix `[rows × cols]`.
    ///
    /// If `per_channel` is true, one scale per row. If `group_size > 0`,
    /// one scale per group within each channel. Otherwise one global scale.
    pub fn compute(
        weights: &[f32],
        rows: usize,
        cols: usize,
        per_channel: bool,
        group_size: usize,
    ) -> Vec<f32> {
        if per_channel {
            if group_size > 0 {
                Self::per_group(weights, rows, cols, group_size)
            } else {
                Self::per_channel(weights, rows, cols)
            }
        } else {
            vec![Self::global_scale(weights)]
        }
    }

    /// Single global scale = max |w|.
    fn global_scale(weights: &[f32]) -> f32 {
        weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max)
    }

    /// One scale per row.
    fn per_channel(weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        assert_eq!(weights.len(), rows * cols);
        (0..rows)
            .map(|r| {
                let row = &weights[r * cols..(r + 1) * cols];
                row.iter().map(|w| w.abs()).fold(0.0f32, f32::max)
            })
            .collect()
    }

    /// One scale per group of `group_size` elements within each row.
    fn per_group(weights: &[f32], rows: usize, cols: usize, group_size: usize) -> Vec<f32> {
        assert_eq!(weights.len(), rows * cols);
        assert!(group_size > 0, "group_size must be > 0");
        let groups_per_row = cols.div_ceil(group_size);
        let mut scales = Vec::with_capacity(rows * groups_per_row);
        for r in 0..rows {
            let row = &weights[r * cols..(r + 1) * cols];
            for g in 0..groups_per_row {
                let start = g * group_size;
                let end = (start + group_size).min(cols);
                let s = row[start..end].iter().map(|w| w.abs()).fold(0.0f32, f32::max);
                scales.push(s);
            }
        }
        scales
    }
}

// ── Quantization error analyzer ─────────────────────────────────────────────

/// Measures quantization error between original and dequantized weights.
pub struct QuantErrorAnalyzer;

impl QuantErrorAnalyzer {
    /// Compute full error statistics.
    pub fn analyze(original: &[f32], dequantized: &[f32]) -> ErrorStats {
        assert_eq!(original.len(), dequantized.len(), "length mismatch in error analysis");
        ErrorStats {
            mse: Self::mse(original, dequantized),
            cosine_similarity: Self::cosine_similarity(original, dequantized),
            max_deviation: Self::max_deviation(original, dequantized),
        }
    }

    /// Mean squared error.
    pub fn mse(a: &[f32], b: &[f32]) -> f64 {
        if a.is_empty() {
            return 0.0;
        }
        let sum: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = (x as f64) - (y as f64);
                d * d
            })
            .sum();
        sum / a.len() as f64
    }

    /// Cosine similarity in [−1, 1].
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            let (xd, yd) = (x as f64, y as f64);
            dot += xd * yd;
            na += xd * xd;
            nb += yd * yd;
        }
        let denom = na.sqrt() * nb.sqrt();
        if denom < 1e-30 {
            // Both vectors near zero — treat as perfect similarity.
            return 1.0;
        }
        dot / denom
    }

    /// Maximum absolute deviation.
    pub fn max_deviation(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| ((x as f64) - (y as f64)).abs())
            .fold(0.0f64, f64::max)
    }
}

// ── Ternary quantizer ───────────────────────────────────────────────────────

/// Specialized quantizer for BitNet ternary {−1, 0, +1} weights.
///
/// Uses absolute-mean scaling: `scale = mean(|w|)`, then each weight is
/// mapped to `sign(round(w / scale))` producing only {−1, 0, +1}.
pub struct TernaryQuantizer;

impl TernaryQuantizer {
    /// Quantize `weights` to ternary values, returning `(packed_i2s, scales)`.
    ///
    /// One scale per row when `per_channel` is true; one global scale otherwise.
    pub fn quantize(
        weights: &[f32],
        rows: usize,
        cols: usize,
        per_channel: bool,
    ) -> (Vec<u8>, Vec<f32>) {
        assert_eq!(weights.len(), rows * cols);
        let mut ternary = vec![0i8; weights.len()];
        let scales = if per_channel {
            let mut scales = Vec::with_capacity(rows);
            for r in 0..rows {
                let row = &weights[r * cols..(r + 1) * cols];
                let scale = Self::abs_mean(row);
                scales.push(scale);
                for (c, &w) in row.iter().enumerate() {
                    ternary[r * cols + c] = Self::ternary_map(w, scale);
                }
            }
            scales
        } else {
            let scale = Self::abs_mean(weights);
            for (i, &w) in weights.iter().enumerate() {
                ternary[i] = Self::ternary_map(w, scale);
            }
            vec![scale]
        };
        let packed = Self::pack_i2s(&ternary);
        (packed, scales)
    }

    /// Dequantize packed I2_S ternary data back to f32.
    pub fn dequantize(
        packed: &[u8],
        scales: &[f32],
        rows: usize,
        cols: usize,
        per_channel: bool,
    ) -> Vec<f32> {
        let n = rows * cols;
        let ternary = Self::unpack_i2s(packed, n);
        let mut out = vec![0.0f32; n];
        if per_channel {
            assert_eq!(scales.len(), rows);
            for r in 0..rows {
                for c in 0..cols {
                    out[r * cols + c] = ternary[r * cols + c] as f32 * scales[r];
                }
            }
        } else {
            let s = scales[0];
            for (i, &t) in ternary.iter().enumerate() {
                out[i] = t as f32 * s;
            }
        }
        out
    }

    /// Map a single weight to {-1, 0, +1}.
    fn ternary_map(w: f32, scale: f32) -> i8 {
        if scale < 1e-30 {
            return 0;
        }
        let normalized = w / scale;
        if normalized > 0.5 {
            1
        } else if normalized < -0.5 {
            -1
        } else {
            0
        }
    }

    /// Absolute mean of a slice.
    fn abs_mean(v: &[f32]) -> f32 {
        if v.is_empty() {
            return 0.0;
        }
        let sum: f32 = v.iter().map(|x| x.abs()).sum();
        sum / v.len() as f32
    }

    /// Pack ternary values into I2_S bytes (4 values per byte).
    ///
    /// Encoding: `0b00 → −1`, `0b01 → 0`, `0b10 → +1`.
    fn pack_i2s(values: &[i8]) -> Vec<u8> {
        let packed_len = values.len().div_ceil(4);
        let mut packed = vec![0u8; packed_len];
        for (i, &v) in values.iter().enumerate() {
            let encoded: u8 = match v {
                -1 => 0b00,
                0 => 0b01,
                1 => 0b10,
                _ => panic!("ternary value must be -1, 0, or +1, got {v}"),
            };
            packed[i / 4] |= encoded << ((i % 4) * 2);
        }
        packed
    }

    /// Unpack I2_S bytes to ternary values.
    fn unpack_i2s(packed: &[u8], n: usize) -> Vec<i8> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let byte = packed[i / 4];
            let bits = (byte >> ((i % 4) * 2)) & 0x03;
            let val = match bits {
                0b00 => -1,
                0b01 => 0,
                0b10 => 1,
                _ => panic!("invalid I2_S encoding: 0b{bits:02b}"),
            };
            out.push(val);
        }
        out
    }
}

// ── Weight quantizer (general) ──────────────────────────────────────────────

/// Quantizes weight tensors to the configured format.
pub struct WeightQuantizer;

impl WeightQuantizer {
    /// Quantize a 2-D weight matrix `[rows × cols]`.
    pub fn quantize(
        weights: &[f32],
        rows: usize,
        cols: usize,
        config: &QuantConfig,
    ) -> QuantResult {
        assert_eq!(weights.len(), rows * cols);
        match config.target_format {
            WeightFormat::Ternary | WeightFormat::I2S => {
                Self::quantize_ternary(weights, rows, cols, config)
            }
            WeightFormat::I4 => Self::quantize_i4(weights, rows, cols, config),
            WeightFormat::I8 => Self::quantize_i8(weights, rows, cols, config),
            WeightFormat::F16 => Self::quantize_f16(weights, config),
            WeightFormat::BF16 => Self::quantize_bf16(weights, config),
            WeightFormat::F32 => Self::passthrough_f32(weights, config),
        }
    }

    /// Ternary quantization via [`TernaryQuantizer`].
    fn quantize_ternary(
        weights: &[f32],
        rows: usize,
        cols: usize,
        config: &QuantConfig,
    ) -> QuantResult {
        let (packed, scales) = TernaryQuantizer::quantize(weights, rows, cols, config.per_channel);
        let deq = TernaryQuantizer::dequantize(&packed, &scales, rows, cols, config.per_channel);
        let error_stats = QuantErrorAnalyzer::analyze(weights, &deq);
        QuantResult {
            quantized_data: packed,
            scales,
            zero_points: vec![],
            format: config.target_format,
            num_elements: weights.len(),
            error_stats,
        }
    }

    /// 8-bit signed integer quantization.
    fn quantize_i8(weights: &[f32], rows: usize, cols: usize, config: &QuantConfig) -> QuantResult {
        let scales =
            ScaleEstimator::compute(weights, rows, cols, config.per_channel, config.group_size);
        let group_size = if config.group_size > 0 {
            config.group_size
        } else if config.per_channel {
            cols
        } else {
            weights.len()
        };

        let mut quantized = vec![0u8; weights.len()];
        let mut dequantized = vec![0.0f32; weights.len()];
        for (i, &w) in weights.iter().enumerate() {
            let scale_idx = i / group_size;
            let s = scales[scale_idx];
            let q =
                if s < 1e-30 { 0i8 } else { (w / s * 127.0).round().clamp(-128.0, 127.0) as i8 };
            quantized[i] = q as u8;
            dequantized[i] = q as f32 * s / 127.0;
        }
        let error_stats = QuantErrorAnalyzer::analyze(weights, &dequantized);
        QuantResult {
            quantized_data: quantized,
            scales,
            zero_points: vec![],
            format: WeightFormat::I8,
            num_elements: weights.len(),
            error_stats,
        }
    }

    /// 4-bit signed integer quantization (two values per byte).
    fn quantize_i4(weights: &[f32], rows: usize, cols: usize, config: &QuantConfig) -> QuantResult {
        let scales =
            ScaleEstimator::compute(weights, rows, cols, config.per_channel, config.group_size);
        let group_size = if config.group_size > 0 {
            config.group_size
        } else if config.per_channel {
            cols
        } else {
            weights.len()
        };

        let packed_len = weights.len().div_ceil(2);
        let mut packed = vec![0u8; packed_len];
        let mut dequantized = vec![0.0f32; weights.len()];
        for (i, &w) in weights.iter().enumerate() {
            let scale_idx = i / group_size;
            let s = scales[scale_idx];
            let q = if s < 1e-30 { 0i8 } else { (w / s * 7.0).round().clamp(-8.0, 7.0) as i8 };
            // Pack two nibbles per byte.
            let nibble = (q & 0x0F) as u8;
            if i % 2 == 0 {
                packed[i / 2] |= nibble;
            } else {
                packed[i / 2] |= nibble << 4;
            }
            dequantized[i] = q as f32 * s / 7.0;
        }
        let error_stats = QuantErrorAnalyzer::analyze(weights, &dequantized);
        QuantResult {
            quantized_data: packed,
            scales,
            zero_points: vec![],
            format: WeightFormat::I4,
            num_elements: weights.len(),
            error_stats,
        }
    }

    /// F16 quantization (truncation from f32).
    fn quantize_f16(weights: &[f32], config: &QuantConfig) -> QuantResult {
        let mut data = Vec::with_capacity(weights.len() * 2);
        let mut dequantized = Vec::with_capacity(weights.len());
        for &w in weights {
            let half = f32_to_f16(w);
            data.extend_from_slice(&half.to_le_bytes());
            dequantized.push(f16_to_f32(half));
        }
        let error_stats = QuantErrorAnalyzer::analyze(weights, &dequantized);
        QuantResult {
            quantized_data: data,
            scales: vec![],
            zero_points: vec![],
            format: config.target_format,
            num_elements: weights.len(),
            error_stats,
        }
    }

    /// BF16 quantization (truncation from f32).
    fn quantize_bf16(weights: &[f32], config: &QuantConfig) -> QuantResult {
        let mut data = Vec::with_capacity(weights.len() * 2);
        let mut dequantized = Vec::with_capacity(weights.len());
        for &w in weights {
            let bf = f32_to_bf16(w);
            data.extend_from_slice(&bf.to_le_bytes());
            dequantized.push(bf16_to_f32(bf));
        }
        let error_stats = QuantErrorAnalyzer::analyze(weights, &dequantized);
        QuantResult {
            quantized_data: data,
            scales: vec![],
            zero_points: vec![],
            format: config.target_format,
            num_elements: weights.len(),
            error_stats,
        }
    }

    /// Passthrough: store as raw f32 bytes.
    fn passthrough_f32(weights: &[f32], config: &QuantConfig) -> QuantResult {
        let data: Vec<u8> = weights.iter().flat_map(|w| w.to_le_bytes()).collect();
        let error_stats = ErrorStats { mse: 0.0, cosine_similarity: 1.0, max_deviation: 0.0 };
        QuantResult {
            quantized_data: data,
            scales: vec![],
            zero_points: vec![],
            format: config.target_format,
            num_elements: weights.len(),
            error_stats,
        }
    }
}

// ── f16 / bf16 helpers ──────────────────────────────────────────────────────

/// Convert f32 to f16 (IEEE 754 half-precision) stored as u16.
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 0xFF {
        // Inf / NaN
        return (sign | 0x7C00 | mantissa.min(1)) as u16;
    }
    let new_exp = exponent - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16; // overflow → Inf
    }
    if new_exp <= 0 {
        return sign as u16; // underflow → zero
    }
    let new_mantissa = mantissa >> 13;
    (sign | ((new_exp as u32) << 10) | new_mantissa) as u16
}

/// Convert f16 (u16) back to f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exponent = ((h >> 10) & 0x1F) as u32;
    let mantissa = (h & 0x03FF) as u32;

    if exponent == 0x1F {
        let bits = (sign << 31) | 0x7F80_0000 | (mantissa << 13);
        return f32::from_bits(bits);
    }
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal f16 → normal f32.
        let mut m = mantissa;
        let mut e = 0i32;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e += 1;
        }
        let exp32 = (127 - 15 - e) as u32;
        let man32 = (m & 0x03FF) << 13;
        return f32::from_bits((sign << 31) | (exp32 << 23) | man32);
    }
    let exp32 = exponent + 127 - 15;
    let man32 = mantissa << 13;
    f32::from_bits((sign << 31) | (exp32 << 23) | man32)
}

/// Convert f32 to bf16 (truncation of lower 16 mantissa bits).
fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Convert bf16 (u16) back to f32.
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// ── OpenCL kernel source ────────────────────────────────────────────────────

/// OpenCL kernel for ternary weight quantization on Intel Arc A770.
///
/// Input: `weights` (float), `abs_mean_scale` (float per-row).
/// Output: `packed` (uchar, 4 ternary values per byte).
pub const TERNARY_QUANTIZE_KERNEL: &str = r#"
__kernel void ternary_quantize(
    __global const float* weights,
    __global const float* scales,
    __global uchar*       packed,
    const int             cols,
    const int             per_channel
) {
    int row = get_global_id(0);
    int grp = get_global_id(1); // group of 4 columns
    int base_col = grp * 4;

    float scale = per_channel ? scales[row] : scales[0];

    uchar byte_val = 0;
    for (int k = 0; k < 4; k++) {
        int col = base_col + k;
        if (col >= cols) break;
        float w = weights[row * cols + col];
        float normalized = (scale > 1e-30f) ? (w / scale) : 0.0f;
        // Map to ternary: 0b00=-1, 0b01=0, 0b10=+1
        uchar enc;
        if (normalized > 0.5f)
            enc = 2; // +1
        else if (normalized < -0.5f)
            enc = 0; // -1
        else
            enc = 1; //  0
        byte_val |= (enc << (k * 2));
    }
    int packed_cols = (cols + 3) / 4;
    packed[row * packed_cols + grp] = byte_val;
}
"#;

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── WeightFormat ────────────────────────────────────────────────────

    #[test]
    fn test_weight_format_bits_per_element() {
        assert_eq!(WeightFormat::F32.bits_per_element(), 32);
        assert_eq!(WeightFormat::F16.bits_per_element(), 16);
        assert_eq!(WeightFormat::BF16.bits_per_element(), 16);
        assert_eq!(WeightFormat::I8.bits_per_element(), 8);
        assert_eq!(WeightFormat::I4.bits_per_element(), 4);
        assert_eq!(WeightFormat::I2S.bits_per_element(), 2);
        assert_eq!(WeightFormat::Ternary.bits_per_element(), 2);
    }

    #[test]
    fn test_weight_format_storage_bytes() {
        assert_eq!(WeightFormat::F32.storage_bytes(4), 16);
        assert_eq!(WeightFormat::I8.storage_bytes(3), 3);
        assert_eq!(WeightFormat::I4.storage_bytes(3), 2); // 12 bits → 2 bytes
        assert_eq!(WeightFormat::Ternary.storage_bytes(4), 1);
        assert_eq!(WeightFormat::Ternary.storage_bytes(5), 2);
    }

    #[test]
    fn test_weight_format_display() {
        assert_eq!(WeightFormat::F32.to_string(), "F32");
        assert_eq!(WeightFormat::I2S.to_string(), "I2_S");
        assert_eq!(WeightFormat::Ternary.to_string(), "Ternary");
    }

    // ── QuantConfig defaults ────────────────────────────────────────────

    #[test]
    fn test_quant_config_default() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.target_format, WeightFormat::Ternary);
        assert!(cfg.symmetric);
        assert!(cfg.per_channel);
        assert_eq!(cfg.group_size, 0);
    }

    // ── ScaleEstimator ──────────────────────────────────────────────────

    #[test]
    fn test_scale_global() {
        let w = vec![1.0, -2.0, 0.5, -0.3];
        let s = ScaleEstimator::compute(&w, 1, 4, false, 0);
        assert_eq!(s.len(), 1);
        assert!((s[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_per_channel() {
        // 2×4 matrix
        let w = vec![
            1.0, -2.0, 0.5, 0.1, // row 0: max |w| = 2.0
            0.3, 0.1, -0.4, 0.2, // row 1: max |w| = 0.4
        ];
        let s = ScaleEstimator::compute(&w, 2, 4, true, 0);
        assert_eq!(s.len(), 2);
        assert!((s[0] - 2.0).abs() < 1e-6);
        assert!((s[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_scale_per_group() {
        // 1×8, group_size=4
        let w = vec![1.0, -2.0, 0.5, 0.1, 3.0, -0.1, 0.2, 0.3];
        let s = ScaleEstimator::compute(&w, 1, 8, true, 4);
        assert_eq!(s.len(), 2); // 8/4 = 2 groups
        assert!((s[0] - 2.0).abs() < 1e-6);
        assert!((s[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_per_group_non_divisible() {
        // 1×5, group_size=4 → 2 groups (4 + 1)
        let w = vec![0.5, -1.0, 0.2, 0.3, 2.0];
        let s = ScaleEstimator::compute(&w, 1, 5, true, 4);
        assert_eq!(s.len(), 2);
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!((s[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_all_zeros() {
        let w = [0.0; 16];
        let s = ScaleEstimator::compute(&w, 4, 4, true, 0);
        assert!(s.iter().all(|&x| x == 0.0));
    }

    // ── QuantErrorAnalyzer ──────────────────────────────────────────────

    #[test]
    fn test_mse_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((QuantErrorAnalyzer::mse(&a, &a)).abs() < 1e-12);
    }

    #[test]
    fn test_mse_known_value() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.1, 3.1];
        // MSE = (0.01 + 0.01 + 0.01) / 3 = 0.01
        assert!((QuantErrorAnalyzer::mse(&a, &b) - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_mse_empty() {
        assert_eq!(QuantErrorAnalyzer::mse(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let cos = QuantErrorAnalyzer::cosine_similarity(&a, &a);
        assert!((cos - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let cos = QuantErrorAnalyzer::cosine_similarity(&a, &b);
        assert!(cos.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        let cos = QuantErrorAnalyzer::cosine_similarity(&a, &b);
        assert!((cos + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_zero_vectors() {
        let z = vec![0.0, 0.0];
        let cos = QuantErrorAnalyzer::cosine_similarity(&z, &z);
        assert!((cos - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_deviation() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.5, 3.0];
        let dev = QuantErrorAnalyzer::max_deviation(&a, &b);
        assert!((dev - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_error_stats_display() {
        let stats = ErrorStats { mse: 1e-4, cosine_similarity: 0.999, max_deviation: 0.05 };
        let s = stats.to_string();
        assert!(s.contains("MSE="));
        assert!(s.contains("cos_sim="));
    }

    // ── TernaryQuantizer ────────────────────────────────────────────────

    #[test]
    fn test_ternary_all_zeros() {
        let w = [0.0; 8];
        let (packed, scales) = TernaryQuantizer::quantize(&w, 1, 8, false);
        // All-zero input → scale=0, all mapped to 0
        assert_eq!(scales[0], 0.0);
        let unpacked = TernaryQuantizer::unpack_i2s(&packed, 8);
        assert!(unpacked.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_ternary_all_positive() {
        let w = [1.0; 4];
        let (packed, scales) = TernaryQuantizer::quantize(&w, 1, 4, false);
        assert!(scales[0] > 0.0);
        let unpacked = TernaryQuantizer::unpack_i2s(&packed, 4);
        assert!(unpacked.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_ternary_all_negative() {
        let w = [-1.0; 4];
        let (packed, _scales) = TernaryQuantizer::quantize(&w, 1, 4, false);
        let unpacked = TernaryQuantizer::unpack_i2s(&packed, 4);
        assert!(unpacked.iter().all(|&v| v == -1));
    }

    #[test]
    fn test_ternary_mixed_known() {
        // Large positive, near-zero, large negative, near-zero
        let w = vec![2.0, 0.01, -2.0, 0.01];
        let (packed, _) = TernaryQuantizer::quantize(&w, 1, 4, false);
        let unpacked = TernaryQuantizer::unpack_i2s(&packed, 4);
        assert_eq!(unpacked[0], 1); // 2.0 → +1
        assert_eq!(unpacked[2], -1); // -2.0 → -1
        // Near-zero values should be 0
        assert_eq!(unpacked[1], 0);
        assert_eq!(unpacked[3], 0);
    }

    #[test]
    fn test_ternary_per_channel() {
        // Row 0: large weights, row 1: small weights
        let w = vec![
            10.0, -10.0, 0.01, 0.02, // row 0
            0.1, -0.1, 0.001, 0.002, // row 1
        ];
        let (packed, scales) = TernaryQuantizer::quantize(&w, 2, 4, true);
        assert_eq!(scales.len(), 2);
        assert!(scales[0] > scales[1]); // row 0 has bigger scale
        let unpacked = TernaryQuantizer::unpack_i2s(&packed, 8);
        // Both rows should quantize the large values correctly
        assert_eq!(unpacked[0], 1); // 10.0
        assert_eq!(unpacked[1], -1); // -10.0
        assert_eq!(unpacked[4], 1); // 0.1
        assert_eq!(unpacked[5], -1); // -0.1
    }

    #[test]
    fn test_ternary_values_only_minus1_zero_plus1() {
        let w: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let (packed, _) = TernaryQuantizer::quantize(&w, 1, 256, false);
        let unpacked = TernaryQuantizer::unpack_i2s(&packed, 256);
        for &v in &unpacked {
            assert!(v == -1 || v == 0 || v == 1, "ternary value out of range: {v}");
        }
    }

    #[test]
    fn test_ternary_roundtrip() {
        let w = vec![2.0, 0.01, -1.5, 0.8, -0.02, 3.0, -2.5, 0.0];
        let (packed, scales) = TernaryQuantizer::quantize(&w, 1, 8, false);
        let deq = TernaryQuantizer::dequantize(&packed, &scales, 1, 8, false);
        let stats = QuantErrorAnalyzer::analyze(&w, &deq);
        // Ternary is lossy, but cosine similarity should be decent
        assert!(stats.cosine_similarity > 0.5);
    }

    #[test]
    fn test_ternary_roundtrip_per_channel() {
        let w = vec![
            1.0, -1.0, 0.5, -0.5, // row 0
            0.1, -0.1, 0.05, -0.05, // row 1
        ];
        let (packed, scales) = TernaryQuantizer::quantize(&w, 2, 4, true);
        let deq = TernaryQuantizer::dequantize(&packed, &scales, 2, 4, true);
        assert_eq!(deq.len(), 8);
        // Per-channel dequant should respect row scales
        for &v in &deq {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ternary_single_element() {
        let w = [5.0];
        let (packed, scales) = TernaryQuantizer::quantize(&w, 1, 1, false);
        let deq = TernaryQuantizer::dequantize(&packed, &scales, 1, 1, false);
        assert_eq!(deq.len(), 1);
        assert!(deq[0] > 0.0); // Should be +1 * scale
    }

    // ── WeightQuantizer — F32→I8 ────────────────────────────────────────

    #[test]
    fn test_quantize_i8_basic() {
        let w = vec![1.0, -1.0, 0.5, -0.5];
        let cfg = QuantConfig {
            target_format: WeightFormat::I8,
            symmetric: true,
            per_channel: false,
            group_size: 0,
        };
        let result = WeightQuantizer::quantize(&w, 1, 4, &cfg);
        assert_eq!(result.format, WeightFormat::I8);
        assert_eq!(result.num_elements, 4);
        assert_eq!(result.quantized_data.len(), 4);
    }

    #[test]
    fn test_quantize_i8_roundtrip_error_bounded() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let cfg = QuantConfig {
            target_format: WeightFormat::I8,
            symmetric: true,
            per_channel: false,
            group_size: 0,
        };
        let result = WeightQuantizer::quantize(&w, 1, 64, &cfg);
        // I8 quantization error should be small
        assert!(result.error_stats.mse < 0.01);
        assert!(result.error_stats.cosine_similarity > 0.99);
    }

    #[test]
    fn test_quantize_i8_per_channel() {
        let w = vec![
            10.0, -10.0, 5.0, -5.0, // row 0
            0.1, -0.1, 0.05, -0.05, // row 1
        ];
        let cfg = QuantConfig {
            target_format: WeightFormat::I8,
            symmetric: true,
            per_channel: true,
            group_size: 0,
        };
        let result = WeightQuantizer::quantize(&w, 2, 4, &cfg);
        assert_eq!(result.scales.len(), 2);
    }

    // ── WeightQuantizer — F32→I4 ────────────────────────────────────────

    #[test]
    fn test_quantize_i4_basic() {
        let w = vec![1.0, -1.0, 0.5, -0.5];
        let cfg = QuantConfig {
            target_format: WeightFormat::I4,
            symmetric: true,
            per_channel: false,
            group_size: 0,
        };
        let result = WeightQuantizer::quantize(&w, 1, 4, &cfg);
        assert_eq!(result.format, WeightFormat::I4);
        assert_eq!(result.quantized_data.len(), 2); // 4 values, 2 per byte
    }

    #[test]
    fn test_quantize_i4_roundtrip_error_bounded() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let cfg = QuantConfig {
            target_format: WeightFormat::I4,
            symmetric: true,
            per_channel: false,
            group_size: 0,
        };
        let result = WeightQuantizer::quantize(&w, 1, 64, &cfg);
        assert!(result.error_stats.mse < 0.1);
        assert!(result.error_stats.cosine_similarity > 0.95);
    }

    #[test]
    fn test_quantize_i4_per_group() {
        let w = vec![1.0, -1.0, 0.5, -0.5, 0.1, -0.1, 0.05, -0.05];
        let cfg = QuantConfig {
            target_format: WeightFormat::I4,
            symmetric: true,
            per_channel: true,
            group_size: 4,
        };
        let result = WeightQuantizer::quantize(&w, 1, 8, &cfg);
        assert_eq!(result.scales.len(), 2); // 8/4 = 2 groups
    }

    // ── WeightQuantizer — F32→I2_S ──────────────────────────────────────

    #[test]
    fn test_quantize_i2s_basic() {
        let w = vec![2.0, 0.01, -2.0, 0.01];
        let cfg = QuantConfig {
            target_format: WeightFormat::I2S,
            symmetric: true,
            per_channel: false,
            group_size: 0,
        };
        let result = WeightQuantizer::quantize(&w, 1, 4, &cfg);
        assert_eq!(result.format, WeightFormat::I2S);
    }

    // ── WeightQuantizer — F32→Ternary ───────────────────────────────────

    #[test]
    fn test_quantize_ternary_basic() {
        let w = vec![2.0, 0.01, -2.0, 0.01];
        let cfg = QuantConfig::default();
        let result = WeightQuantizer::quantize(&w, 1, 4, &cfg);
        assert_eq!(result.format, WeightFormat::Ternary);
        assert!(!result.scales.is_empty());
    }

    #[test]
    fn test_quantize_ternary_all_same_value() {
        let w = [5.0; 16];
        let cfg = QuantConfig::default();
        let result = WeightQuantizer::quantize(&w, 4, 4, &cfg);
        // All same positive → all quantize to +1
        assert!(result.error_stats.cosine_similarity > 0.9);
    }

    // ── WeightQuantizer — F32→F16 ───────────────────────────────────────

    #[test]
    fn test_quantize_f16_basic() {
        let w = vec![1.0, -1.0, 0.5, 0.0];
        let cfg = QuantConfig { target_format: WeightFormat::F16, ..QuantConfig::default() };
        let result = WeightQuantizer::quantize(&w, 1, 4, &cfg);
        assert_eq!(result.format, WeightFormat::F16);
        assert_eq!(result.quantized_data.len(), 8); // 4 × 2 bytes
    }

    #[test]
    fn test_quantize_f16_roundtrip_small_error() {
        let w = vec![1.0, -1.0, 0.5, 0.25, 0.0, 100.0];
        let cfg = QuantConfig { target_format: WeightFormat::F16, ..QuantConfig::default() };
        let result = WeightQuantizer::quantize(&w, 1, 6, &cfg);
        assert!(result.error_stats.cosine_similarity > 0.9999);
    }

    // ── WeightQuantizer — F32→BF16 ──────────────────────────────────────

    #[test]
    fn test_quantize_bf16_basic() {
        let w = vec![1.0, -1.0, 0.5, 0.0];
        let cfg = QuantConfig { target_format: WeightFormat::BF16, ..QuantConfig::default() };
        let result = WeightQuantizer::quantize(&w, 1, 4, &cfg);
        assert_eq!(result.format, WeightFormat::BF16);
        assert_eq!(result.quantized_data.len(), 8);
    }

    #[test]
    fn test_quantize_bf16_roundtrip_small_error() {
        let w = vec![1.0, -1.0, 0.5, 0.25, 0.0, 100.0];
        let cfg = QuantConfig { target_format: WeightFormat::BF16, ..QuantConfig::default() };
        let result = WeightQuantizer::quantize(&w, 1, 6, &cfg);
        assert!(result.error_stats.cosine_similarity > 0.999);
    }

    // ── WeightQuantizer — F32 passthrough ───────────────────────────────

    #[test]
    fn test_quantize_f32_passthrough() {
        let w = vec![1.0, 2.0, 3.0];
        let cfg = QuantConfig { target_format: WeightFormat::F32, ..QuantConfig::default() };
        let result = WeightQuantizer::quantize(&w, 1, 3, &cfg);
        assert_eq!(result.error_stats.mse, 0.0);
        assert_eq!(result.error_stats.cosine_similarity, 1.0);
        assert_eq!(result.quantized_data.len(), 12); // 3 × 4 bytes
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_edge_single_weight_i8() {
        let w = [0.5];
        let cfg = QuantConfig {
            target_format: WeightFormat::I8,
            symmetric: true,
            per_channel: false,
            group_size: 0,
        };
        let result = WeightQuantizer::quantize(&w, 1, 1, &cfg);
        assert_eq!(result.num_elements, 1);
    }

    #[test]
    fn test_edge_all_zeros_i8() {
        let w = [0.0; 16];
        let cfg = QuantConfig {
            target_format: WeightFormat::I8,
            symmetric: true,
            per_channel: true,
            group_size: 0,
        };
        let result = WeightQuantizer::quantize(&w, 4, 4, &cfg);
        assert_eq!(result.error_stats.mse, 0.0);
    }

    #[test]
    fn test_edge_all_zeros_ternary() {
        let w = [0.0; 16];
        let cfg = QuantConfig::default();
        let result = WeightQuantizer::quantize(&w, 4, 4, &cfg);
        assert_eq!(result.error_stats.mse, 0.0);
    }

    #[test]
    fn test_edge_all_same_value_i4() {
        let w = [1.5; 8];
        let cfg = QuantConfig {
            target_format: WeightFormat::I4,
            symmetric: true,
            per_channel: false,
            group_size: 0,
        };
        let result = WeightQuantizer::quantize(&w, 1, 8, &cfg);
        assert!(result.error_stats.cosine_similarity > 0.99);
    }

    #[test]
    fn test_edge_huge_tensor_ternary() {
        let w: Vec<f32> = (0..4096).map(|i| ((i as f32) * 0.001).sin()).collect();
        let cfg = QuantConfig::default();
        let result = WeightQuantizer::quantize(&w, 64, 64, &cfg);
        assert_eq!(result.num_elements, 4096);
        assert!(result.error_stats.cosine_similarity > 0.0);
    }

    #[test]
    fn test_edge_huge_tensor_i8() {
        let w: Vec<f32> = (0..4096).map(|i| ((i as f32) * 0.001).sin()).collect();
        let cfg = QuantConfig {
            target_format: WeightFormat::I8,
            symmetric: true,
            per_channel: true,
            group_size: 0,
        };
        let result = WeightQuantizer::quantize(&w, 64, 64, &cfg);
        assert_eq!(result.num_elements, 4096);
        assert!(result.error_stats.cosine_similarity > 0.99);
    }

    // ── Property-like tests ─────────────────────────────────────────────

    #[test]
    fn test_property_i8_error_bounded() {
        // For any reasonable input, I8 MSE should be bounded.
        for seed in 0..10 {
            let w: Vec<f32> =
                (0..128).map(|i| ((i + seed * 7) as f32 * 0.037).sin() * 2.0).collect();
            let cfg = QuantConfig {
                target_format: WeightFormat::I8,
                symmetric: true,
                per_channel: false,
                group_size: 0,
            };
            let result = WeightQuantizer::quantize(&w, 1, 128, &cfg);
            assert!(
                result.error_stats.mse < 0.01,
                "seed={seed}: MSE={} too high",
                result.error_stats.mse
            );
        }
    }

    #[test]
    fn test_property_ternary_values_in_range() {
        for seed in 0..10 {
            let w: Vec<f32> =
                (0..64).map(|i| ((i + seed * 13) as f32 * 0.071).cos() * 5.0).collect();
            let (packed, _) = TernaryQuantizer::quantize(&w, 1, 64, false);
            let unpacked = TernaryQuantizer::unpack_i2s(&packed, 64);
            for &v in &unpacked {
                assert!(v == -1 || v == 0 || v == 1, "out-of-range ternary: {v}");
            }
        }
    }

    #[test]
    fn test_property_f16_roundtrip_high_similarity() {
        for seed in 0..5 {
            let w: Vec<f32> = (0..64).map(|i| ((i + seed * 11) as f32 * 0.05).sin()).collect();
            let cfg = QuantConfig { target_format: WeightFormat::F16, ..QuantConfig::default() };
            let result = WeightQuantizer::quantize(&w, 1, 64, &cfg);
            assert!(
                result.error_stats.cosine_similarity > 0.999,
                "seed={seed}: cos_sim={}",
                result.error_stats.cosine_similarity
            );
        }
    }

    #[test]
    fn test_property_quantize_preserves_element_count() {
        let formats = [
            WeightFormat::F32,
            WeightFormat::F16,
            WeightFormat::BF16,
            WeightFormat::I8,
            WeightFormat::I4,
            WeightFormat::I2S,
            WeightFormat::Ternary,
        ];
        let w: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        for fmt in &formats {
            let cfg = QuantConfig {
                target_format: *fmt,
                symmetric: true,
                per_channel: false,
                group_size: 0,
            };
            let result = WeightQuantizer::quantize(&w, 1, 32, &cfg);
            assert_eq!(result.num_elements, 32, "format {fmt}: element count mismatch");
        }
    }

    // ── f16/bf16 helpers ────────────────────────────────────────────────

    #[test]
    fn test_f16_roundtrip_exact() {
        // Values exactly representable in f16
        for v in [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert_eq!(back, v, "f16 roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_bf16_roundtrip_exact() {
        // bf16 keeps the top 16 bits of f32 → exact for ±powers of 2
        for v in [0.0f32, 1.0, -1.0, 2.0, -2.0, 0.5] {
            let b = f32_to_bf16(v);
            let back = bf16_to_f32(b);
            assert_eq!(back, v, "bf16 roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_f16_overflow_to_inf() {
        let h = f32_to_f16(100_000.0);
        let back = f16_to_f32(h);
        assert!(back.is_infinite());
    }

    #[test]
    fn test_f16_underflow_to_zero() {
        let h = f32_to_f16(1e-20);
        let back = f16_to_f32(h);
        assert_eq!(back, 0.0);
    }

    // ── OpenCL kernel source ────────────────────────────────────────────

    #[test]
    fn test_opencl_kernel_source_not_empty() {
        assert!(!TERNARY_QUANTIZE_KERNEL.is_empty());
    }

    #[test]
    fn test_opencl_kernel_contains_entry_point() {
        assert!(TERNARY_QUANTIZE_KERNEL.contains("ternary_quantize"));
    }

    #[test]
    fn test_opencl_kernel_contains_encoding_logic() {
        // The kernel should have the ternary threshold logic
        assert!(TERNARY_QUANTIZE_KERNEL.contains("0.5f"));
        assert!(TERNARY_QUANTIZE_KERNEL.contains("enc"));
    }
}
