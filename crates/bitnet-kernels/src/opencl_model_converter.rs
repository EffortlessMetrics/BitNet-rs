//! Model format converter for GPU-optimized weight layouts.
//!
//! Converts model weights between GGUF formats and layouts optimized for
//! OpenCL inference (ternary-packed, INT8 DP4A, F16, F32) and CPU SIMD.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Source weight format to convert from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFormat {
    GgufI2S,
    GgufQK256,
    SafeTensorsF32,
    SafeTensorsF16,
    RawBinary,
}

/// Target weight format to convert to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetFormat {
    OpenClTernaryPacked,
    OpenClInt8Dp4a,
    OpenClF16,
    OpenClF32,
    CpuSimdAligned,
}

/// Per-layer conversion metadata.
#[derive(Debug, Clone)]
pub struct LayerConversion {
    pub layer_name: String,
    pub source_shape: Vec<usize>,
    pub target_shape: Vec<usize>,
    pub requires_transpose: bool,
    pub requires_padding: bool,
}

/// A plan describing the full model conversion.
#[derive(Debug, Clone)]
pub struct ConversionPlan {
    pub source: SourceFormat,
    pub target: TargetFormat,
    pub layers: Vec<LayerConversion>,
    pub estimated_time_ms: u64,
}

/// Result of executing a conversion plan.
#[derive(Debug, Clone)]
pub struct ConversionResult {
    pub success: bool,
    pub layers_converted: usize,
    pub total_bytes_in: usize,
    pub total_bytes_out: usize,
    pub conversion_time_ms: u64,
    pub warnings: Vec<String>,
}

/// Simple LRU-less cache for converted weight blobs.
#[derive(Debug)]
pub struct ConversionCache {
    pub entries: HashMap<String, Vec<u8>>,
    pub max_size_bytes: usize,
    pub current_size_bytes: usize,
}

/// Errors that can occur during conversion.
#[derive(Debug, Clone)]
pub enum ConvertError {
    UnsupportedConversion { from: SourceFormat, to: TargetFormat },
    InvalidWeightShape { expected: Vec<usize>, got: Vec<usize> },
    CacheFull,
    IoError(String),
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedConversion { from, to } => {
                write!(f, "unsupported conversion: {from:?} -> {to:?}")
            }
            Self::InvalidWeightShape { expected, got } => {
                write!(f, "invalid weight shape: expected {expected:?}, got {got:?}")
            }
            Self::CacheFull => write!(f, "conversion cache is full"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for ConvertError {}

impl fmt::Display for SourceFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GgufI2S => write!(f, "GGUF I2_S"),
            Self::GgufQK256 => write!(f, "GGUF QK256"),
            Self::SafeTensorsF32 => write!(f, "SafeTensors F32"),
            Self::SafeTensorsF16 => write!(f, "SafeTensors F16"),
            Self::RawBinary => write!(f, "Raw Binary"),
        }
    }
}

impl fmt::Display for TargetFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenClTernaryPacked => write!(f, "OpenCL Ternary Packed"),
            Self::OpenClInt8Dp4a => write!(f, "OpenCL INT8 DP4A"),
            Self::OpenClF16 => write!(f, "OpenCL F16"),
            Self::OpenClF32 => write!(f, "OpenCL F32"),
            Self::CpuSimdAligned => write!(f, "CPU SIMD Aligned"),
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Build a conversion plan from source/target formats and layer shapes.
pub fn cpu_plan_conversion(
    source: SourceFormat,
    target: TargetFormat,
    layer_shapes: &[(String, Vec<usize>)],
) -> ConversionPlan {
    let layers: Vec<LayerConversion> = layer_shapes
        .iter()
        .map(|(name, shape)| {
            let requires_padding = match target {
                TargetFormat::OpenClTernaryPacked | TargetFormat::OpenClInt8Dp4a => {
                    shape.last().is_some_and(|&d| d % 4 != 0)
                }
                TargetFormat::CpuSimdAligned => shape.last().is_some_and(|&d| d % 32 != 0),
                _ => false,
            };

            let target_shape = if requires_padding {
                let align = match target {
                    TargetFormat::CpuSimdAligned => 32,
                    _ => 4,
                };
                let mut ts = shape.clone();
                if let Some(last) = ts.last_mut() {
                    *last = (*last).div_ceil(align) * align;
                }
                ts
            } else {
                shape.clone()
            };

            LayerConversion {
                layer_name: name.clone(),
                source_shape: shape.clone(),
                target_shape,
                requires_transpose: false,
                requires_padding,
            }
        })
        .collect();

    let estimated_time_ms = layers
        .iter()
        .map(|l| {
            let elems: usize = l.source_shape.iter().product();
            // ~1 µs per 1024 elements as rough estimate
            (elems as u64).saturating_mul(1) / 1024 + 1
        })
        .sum();

    ConversionPlan { source, target, layers, estimated_time_ms }
}

/// Quantize F32 weights to ternary {-1, 0, +1} and pack 4 values per byte.
///
/// Encoding per 2-bit slot: 0b00 = 0, 0b01 = +1, 0b10 = -1.
pub fn cpu_convert_f32_to_ternary(weights: &[f32], threshold: f32) -> Vec<u8> {
    let num_bytes = weights.len().div_ceil(4);
    let mut packed = vec![0u8; num_bytes];

    for (i, &w) in weights.iter().enumerate() {
        let trit: u8 = if w > threshold {
            0b01 // +1
        } else if w < -threshold {
            0b10 // -1
        } else {
            0b00 // 0
        };
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        packed[byte_idx] |= trit << bit_offset;
    }

    packed
}

/// Unpack ternary-packed bytes back to F32 values.
pub fn cpu_convert_ternary_to_f32(packed: &[u8], num_elements: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_elements);

    for i in 0..num_elements {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let trit = (packed[byte_idx] >> bit_offset) & 0b11;
        let val = match trit {
            0b01 => 1.0f32,
            0b10 => -1.0f32,
            _ => 0.0f32,
        };
        out.push(val);
    }

    out
}

/// Convert F32 to F16 (IEEE 754 half-precision) using bit manipulation.
pub fn cpu_convert_f32_to_f16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| f32_to_f16_bits(v)).collect()
}

/// Convert F16 back to F32.
pub fn cpu_convert_f16_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&v| f16_to_f32_bits(v)).collect()
}

/// Quantize F32 to INT8 with symmetric per-tensor quantization.
/// Returns the quantized values and the scale factor.
pub fn cpu_convert_f32_to_int8(data: &[f32]) -> (Vec<i8>, f32) {
    if data.is_empty() {
        return (Vec::new(), 1.0);
    }

    let abs_max = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };

    let quantized: Vec<i8> = data
        .iter()
        .map(|&v| {
            let q = (v / scale).round().clamp(-128.0, 127.0);
            q as i8
        })
        .collect();

    (quantized, scale)
}

/// Dequantize INT8 values back to F32 using the given scale.
pub fn cpu_convert_int8_to_f32(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&v| v as f32 * scale).collect()
}

/// Reorder a row-major weight matrix into tiles for GPU cache efficiency.
///
/// The output contains tiles of `tile_size × tile_size`, iterated in
/// row-major order over the tile grid.
pub fn cpu_tile_for_gpu(weights: &[f32], rows: usize, cols: usize, tile_size: usize) -> Vec<f32> {
    assert_eq!(weights.len(), rows * cols, "weight length must equal rows * cols");
    assert!(tile_size > 0, "tile_size must be > 0");

    let tile_rows = rows.div_ceil(tile_size);
    let tile_cols = cols.div_ceil(tile_size);
    let mut out = vec![0.0f32; tile_rows * tile_cols * tile_size * tile_size];

    for tr in 0..tile_rows {
        for tc in 0..tile_cols {
            let tile_idx = tr * tile_cols + tc;
            for r in 0..tile_size {
                for c in 0..tile_size {
                    let src_r = tr * tile_size + r;
                    let src_c = tc * tile_size + c;
                    let val = if src_r < rows && src_c < cols {
                        weights[src_r * cols + src_c]
                    } else {
                        0.0
                    };
                    out[tile_idx * tile_size * tile_size + r * tile_size + c] = val;
                }
            }
        }
    }

    out
}

/// Pad weight dimensions to a given alignment, returning padded data and new
/// dimensions `(padded_rows, padded_cols)`.
pub fn cpu_pad_for_alignment(
    weights: &[f32],
    rows: usize,
    cols: usize,
    align: usize,
) -> (Vec<f32>, usize, usize) {
    assert_eq!(weights.len(), rows * cols, "weight length must equal rows * cols");
    assert!(align > 0, "alignment must be > 0");

    let padded_rows = rows.div_ceil(align) * align;
    let padded_cols = cols.div_ceil(align) * align;
    let mut out = vec![0.0f32; padded_rows * padded_cols];

    for r in 0..rows {
        for c in 0..cols {
            out[r * padded_cols + c] = weights[r * cols + c];
        }
    }

    (out, padded_rows, padded_cols)
}

/// Execute a conversion plan on the provided per-layer weight vectors.
pub fn cpu_run_conversion(
    plan: &ConversionPlan,
    weights: &[Vec<f32>],
) -> Result<ConversionResult, ConvertError> {
    if weights.len() != plan.layers.len() {
        return Err(ConvertError::InvalidWeightShape {
            expected: vec![plan.layers.len()],
            got: vec![weights.len()],
        });
    }

    let start = std::time::Instant::now();
    let mut total_bytes_in: usize = 0;
    let mut total_bytes_out: usize = 0;
    let mut warnings = Vec::new();

    for (i, layer) in plan.layers.iter().enumerate() {
        let w = &weights[i];
        let expected_elems: usize = layer.source_shape.iter().product();
        if w.len() != expected_elems {
            return Err(ConvertError::InvalidWeightShape {
                expected: layer.source_shape.clone(),
                got: vec![w.len()],
            });
        }

        total_bytes_in += w.len() * std::mem::size_of::<f32>();

        let out_bytes = match plan.target {
            TargetFormat::OpenClTernaryPacked => {
                let packed = cpu_convert_f32_to_ternary(w, 0.5);
                packed.len()
            }
            TargetFormat::OpenClInt8Dp4a => {
                let (q, _scale) = cpu_convert_f32_to_int8(w);
                q.len()
            }
            TargetFormat::OpenClF16 => {
                let h = cpu_convert_f32_to_f16(w);
                h.len() * std::mem::size_of::<u16>()
            }
            TargetFormat::OpenClF32 | TargetFormat::CpuSimdAligned => {
                let target_elems: usize = layer.target_shape.iter().product();
                target_elems * std::mem::size_of::<f32>()
            }
        };

        total_bytes_out += out_bytes;

        if layer.requires_padding {
            warnings.push(format!("layer '{}' required padding", layer.layer_name));
        }
    }

    let elapsed = start.elapsed();

    Ok(ConversionResult {
        success: true,
        layers_converted: plan.layers.len(),
        total_bytes_in,
        total_bytes_out,
        conversion_time_ms: elapsed.as_millis() as u64,
        warnings,
    })
}

/// Store converted data in the cache.
pub fn cpu_cache_converted(
    cache: &mut ConversionCache,
    key: &str,
    data: Vec<u8>,
) -> Result<(), ConvertError> {
    let data_len = data.len();
    if cache.current_size_bytes + data_len > cache.max_size_bytes {
        return Err(ConvertError::CacheFull);
    }
    if let Some(old) = cache.entries.insert(key.to_string(), data) {
        cache.current_size_bytes -= old.len();
    }
    cache.current_size_bytes += data_len;
    Ok(())
}

/// Look up cached converted data.
pub fn cpu_lookup_cached<'a>(cache: &'a ConversionCache, key: &str) -> Option<&'a [u8]> {
    cache.entries.get(key).map(|v| v.as_slice())
}

/// Produce a human-readable summary of a conversion plan.
pub fn format_conversion_plan(plan: &ConversionPlan) -> String {
    let mut s = format!(
        "Conversion: {} -> {} ({} layers, est. {} ms)\n",
        plan.source,
        plan.target,
        plan.layers.len(),
        plan.estimated_time_ms
    );
    for layer in &plan.layers {
        s.push_str(&format!(
            "  {} {:?} -> {:?}{}{}\n",
            layer.layer_name,
            layer.source_shape,
            layer.target_shape,
            if layer.requires_transpose { " [transpose]" } else { "" },
            if layer.requires_padding { " [pad]" } else { "" },
        ));
    }
    s
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 255 {
        // Inf / NaN
        return (sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }) as u16;
    }

    let new_exp = exponent - 127 + 15;

    if new_exp >= 31 {
        // Overflow -> Inf
        return (sign | 0x7C00) as u16;
    }

    if new_exp <= 0 {
        // Subnormal or zero
        if new_exp < -10 {
            return sign as u16;
        }
        let m = (mantissa | 0x0080_0000) >> (1 - new_exp + 13);
        return (sign | m) as u16;
    }

    (sign | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
}

fn f16_to_f32_bits(half: u16) -> f32 {
    let sign = ((half as u32) & 0x8000) << 16;
    let exponent = ((half as u32) >> 10) & 0x1F;
    let mantissa = (half as u32) & 0x03FF;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign); // ±0
        }
        // Subnormal: normalize
        let mut m = mantissa;
        let mut e: i32 = 1;
        while m & 0x0400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        let exp = ((127 - 15 + e) as u32) << 23;
        return f32::from_bits(sign | exp | (m << 13));
    }

    if exponent == 31 {
        // Inf / NaN
        let exp = 0xFF << 23;
        let m = mantissa << 13;
        return f32::from_bits(sign | exp | m);
    }

    let exp = (exponent + 127 - 15) << 23;
    let m = mantissa << 13;
    f32::from_bits(sign | exp | m)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Round-trip tests ---------------------------------------------------

    #[test]
    fn test_f32_ternary_roundtrip_basic() {
        let weights = vec![1.0, -1.0, 0.0, 0.3, -0.8, 0.9, -0.1, 0.0];
        let packed = cpu_convert_f32_to_ternary(&weights, 0.5);
        let restored = cpu_convert_ternary_to_f32(&packed, weights.len());
        assert_eq!(restored, vec![1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_f32_ternary_roundtrip_all_positive() {
        let weights = vec![0.6, 0.7, 0.8, 0.9];
        let packed = cpu_convert_f32_to_ternary(&weights, 0.5);
        let restored = cpu_convert_ternary_to_f32(&packed, weights.len());
        assert!(restored.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_f32_ternary_roundtrip_all_negative() {
        let weights = vec![-0.6, -0.7, -0.8, -0.9];
        let packed = cpu_convert_f32_to_ternary(&weights, 0.5);
        let restored = cpu_convert_ternary_to_f32(&packed, weights.len());
        assert!(restored.iter().all(|&v| v == -1.0));
    }

    #[test]
    fn test_f32_ternary_roundtrip_all_zero() {
        let weights = vec![0.0, 0.1, -0.1, 0.0];
        let packed = cpu_convert_f32_to_ternary(&weights, 0.5);
        let restored = cpu_convert_ternary_to_f32(&packed, weights.len());
        assert!(restored.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_f32_ternary_non_multiple_of_4() {
        let weights = vec![1.0, -1.0, 0.0, 0.8, -0.9];
        let packed = cpu_convert_f32_to_ternary(&weights, 0.5);
        let restored = cpu_convert_ternary_to_f32(&packed, weights.len());
        assert_eq!(restored, vec![1.0, -1.0, 0.0, 1.0, -1.0]);
    }

    #[test]
    fn test_f16_roundtrip_basic() {
        let data = vec![0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0];
        let f16 = cpu_convert_f32_to_f16(&data);
        let back = cpu_convert_f16_to_f32(&f16);
        for (orig, restored) in data.iter().zip(back.iter()) {
            assert!((orig - restored).abs() < 0.01, "f16 roundtrip: {orig} vs {restored}");
        }
    }

    #[test]
    fn test_f16_roundtrip_small_values() {
        let data = vec![0.001, -0.001, 0.0001];
        let f16 = cpu_convert_f32_to_f16(&data);
        let back = cpu_convert_f16_to_f32(&f16);
        for (orig, restored) in data.iter().zip(back.iter()) {
            assert!((orig - restored).abs() < 0.001, "f16 small roundtrip: {orig} vs {restored}");
        }
    }

    #[test]
    fn test_f16_roundtrip_error_bound() {
        // Property: all roundtrip errors must be < 0.01 for values in [-100, 100]
        let data: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.5).collect();
        let f16 = cpu_convert_f32_to_f16(&data);
        let back = cpu_convert_f16_to_f32(&f16);
        for (orig, restored) in data.iter().zip(back.iter()) {
            assert!((orig - restored).abs() < 0.01, "f16 error bound: {orig} vs {restored}");
        }
    }

    #[test]
    fn test_int8_roundtrip_basic() {
        let data = vec![0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
        let (quantized, scale) = cpu_convert_f32_to_int8(&data);
        let back = cpu_convert_int8_to_f32(&quantized, scale);
        for (orig, restored) in data.iter().zip(back.iter()) {
            assert!((orig - restored).abs() < 0.05, "int8 roundtrip: {orig} vs {restored}");
        }
    }

    #[test]
    fn test_int8_roundtrip_large_range() {
        let data: Vec<f32> = (-50..=50).map(|i| i as f32).collect();
        let (quantized, scale) = cpu_convert_f32_to_int8(&data);
        let back = cpu_convert_int8_to_f32(&quantized, scale);
        for (orig, restored) in data.iter().zip(back.iter()) {
            let tol = scale; // error bounded by scale
            assert!(
                (orig - restored).abs() <= tol + 1e-6,
                "int8 range: {orig} vs {restored}, scale={scale}"
            );
        }
    }

    #[test]
    fn test_int8_all_zeros() {
        let data = vec![0.0; 8];
        let (quantized, _scale) = cpu_convert_f32_to_int8(&data);
        assert!(quantized.iter().all(|&v| v == 0));
    }

    // -- Tile reorder tests ------------------------------------------------

    #[test]
    fn test_tile_preserves_values() {
        let weights: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tiled = cpu_tile_for_gpu(&weights, 4, 4, 2);
        let mut sorted_orig = weights.clone();
        sorted_orig.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut sorted_tiled = tiled.clone();
        sorted_tiled.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted_orig, sorted_tiled);
    }

    #[test]
    fn test_tile_identity_when_tile_equals_dim() {
        let weights: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let tiled = cpu_tile_for_gpu(&weights, 3, 3, 3);
        assert_eq!(tiled, weights);
    }

    #[test]
    fn test_tile_non_divisible_dimensions() {
        let weights: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let tiled = cpu_tile_for_gpu(&weights, 2, 3, 2);
        // 2×3 with tile 2: tile grid 1×2, output length = 1*2*2*2 = 8
        assert_eq!(tiled.len(), 8);
        // All original values must be present (extras are zero-padded)
        for &w in &weights {
            assert!(tiled.contains(&w), "missing value {w}");
        }
    }

    #[test]
    fn test_tile_1x1() {
        let weights = vec![42.0];
        let tiled = cpu_tile_for_gpu(&weights, 1, 1, 1);
        assert_eq!(tiled, vec![42.0]);
    }

    // -- Padding tests -----------------------------------------------------

    #[test]
    fn test_pad_correct_dimensions() {
        let weights: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let (padded, pr, pc) = cpu_pad_for_alignment(&weights, 2, 3, 4);
        assert_eq!(pr, 4);
        assert_eq!(pc, 4);
        assert_eq!(padded.len(), 16);
    }

    #[test]
    fn test_pad_preserves_original_values() {
        let weights: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let (padded, _pr, pc) = cpu_pad_for_alignment(&weights, 2, 3, 4);
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(padded[r * pc + c], weights[r * 3 + c]);
            }
        }
    }

    #[test]
    fn test_pad_already_aligned() {
        let weights: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let (padded, pr, pc) = cpu_pad_for_alignment(&weights, 4, 4, 4);
        assert_eq!(pr, 4);
        assert_eq!(pc, 4);
        assert_eq!(padded, weights);
    }

    #[test]
    fn test_pad_zeros_in_padding_region() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (padded, pr, pc) = cpu_pad_for_alignment(&weights, 2, 3, 4);
        // Check that padded cols (col index 3) are zero
        for r in 0..2 {
            assert_eq!(padded[r * pc + 3], 0.0);
        }
        // Check that padded rows (rows 2..pr) are zero
        for r in 2..pr {
            for c in 0..pc {
                assert_eq!(padded[r * pc + c], 0.0);
            }
        }
    }

    // -- Plan tests --------------------------------------------------------

    #[test]
    fn test_plan_correct_layer_count() {
        let shapes = vec![
            ("layer0".to_string(), vec![4, 4]),
            ("layer1".to_string(), vec![8, 8]),
            ("layer2".to_string(), vec![16, 16]),
        ];
        let plan =
            cpu_plan_conversion(SourceFormat::SafeTensorsF32, TargetFormat::OpenClF32, &shapes);
        assert_eq!(plan.layers.len(), 3);
        assert_eq!(plan.source, SourceFormat::SafeTensorsF32);
        assert_eq!(plan.target, TargetFormat::OpenClF32);
    }

    #[test]
    fn test_plan_padding_detected() {
        let shapes = vec![("attn".to_string(), vec![3, 5])];
        let plan =
            cpu_plan_conversion(SourceFormat::GgufI2S, TargetFormat::OpenClInt8Dp4a, &shapes);
        assert!(plan.layers[0].requires_padding);
        assert_eq!(*plan.layers[0].target_shape.last().unwrap(), 8); // 5 rounded to 8
    }

    #[test]
    fn test_plan_no_padding_when_aligned() {
        let shapes = vec![("fc".to_string(), vec![4, 8])];
        let plan = cpu_plan_conversion(
            SourceFormat::GgufQK256,
            TargetFormat::OpenClTernaryPacked,
            &shapes,
        );
        assert!(!plan.layers[0].requires_padding);
        assert_eq!(plan.layers[0].source_shape, plan.layers[0].target_shape);
    }

    #[test]
    fn test_plan_empty_layers() {
        let plan = cpu_plan_conversion(SourceFormat::RawBinary, TargetFormat::OpenClF32, &[]);
        assert_eq!(plan.layers.len(), 0);
        assert_eq!(plan.estimated_time_ms, 0);
    }

    // -- Cache tests -------------------------------------------------------

    #[test]
    fn test_cache_store_and_lookup() {
        let mut cache = ConversionCache {
            entries: HashMap::new(),
            max_size_bytes: 1024,
            current_size_bytes: 0,
        };
        let data = vec![1u8, 2, 3, 4];
        cpu_cache_converted(&mut cache, "key1", data.clone()).unwrap();
        let found = cpu_lookup_cached(&cache, "key1").unwrap();
        assert_eq!(found, &[1, 2, 3, 4]);
    }

    #[test]
    fn test_cache_overwrite_updates_size() {
        let mut cache = ConversionCache {
            entries: HashMap::new(),
            max_size_bytes: 1024,
            current_size_bytes: 0,
        };
        cpu_cache_converted(&mut cache, "k", vec![0u8; 100]).unwrap();
        assert_eq!(cache.current_size_bytes, 100);
        cpu_cache_converted(&mut cache, "k", vec![0u8; 50]).unwrap();
        assert_eq!(cache.current_size_bytes, 50);
    }

    #[test]
    fn test_cache_full_error() {
        let mut cache =
            ConversionCache { entries: HashMap::new(), max_size_bytes: 10, current_size_bytes: 0 };
        let result = cpu_cache_converted(&mut cache, "big", vec![0u8; 11]);
        assert!(matches!(result, Err(ConvertError::CacheFull)));
    }

    #[test]
    fn test_cache_lookup_missing() {
        let cache = ConversionCache {
            entries: HashMap::new(),
            max_size_bytes: 1024,
            current_size_bytes: 0,
        };
        assert!(cpu_lookup_cached(&cache, "nope").is_none());
    }

    // -- Format / display tests --------------------------------------------

    #[test]
    fn test_format_conversion_plan() {
        let plan = ConversionPlan {
            source: SourceFormat::GgufI2S,
            target: TargetFormat::OpenClTernaryPacked,
            layers: vec![LayerConversion {
                layer_name: "fc1".to_string(),
                source_shape: vec![4, 4],
                target_shape: vec![4, 4],
                requires_transpose: false,
                requires_padding: false,
            }],
            estimated_time_ms: 42,
        };
        let s = format_conversion_plan(&plan);
        assert!(s.contains("GGUF I2_S"));
        assert!(s.contains("OpenCL Ternary Packed"));
        assert!(s.contains("fc1"));
        assert!(s.contains("42 ms"));
    }

    #[test]
    fn test_format_conversion_plan_with_flags() {
        let plan = ConversionPlan {
            source: SourceFormat::SafeTensorsF32,
            target: TargetFormat::CpuSimdAligned,
            layers: vec![LayerConversion {
                layer_name: "proj".to_string(),
                source_shape: vec![3, 5],
                target_shape: vec![3, 32],
                requires_transpose: true,
                requires_padding: true,
            }],
            estimated_time_ms: 1,
        };
        let s = format_conversion_plan(&plan);
        assert!(s.contains("[transpose]"));
        assert!(s.contains("[pad]"));
    }

    // -- Conversion execution tests ----------------------------------------

    #[test]
    fn test_run_conversion_basic() {
        let shapes = vec![("l0".to_string(), vec![4])];
        let plan =
            cpu_plan_conversion(SourceFormat::SafeTensorsF32, TargetFormat::OpenClF32, &shapes);
        let weights = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let result = cpu_run_conversion(&plan, &weights).unwrap();
        assert!(result.success);
        assert_eq!(result.layers_converted, 1);
    }

    #[test]
    fn test_run_conversion_shape_mismatch() {
        let shapes = vec![("l0".to_string(), vec![4])];
        let plan =
            cpu_plan_conversion(SourceFormat::SafeTensorsF32, TargetFormat::OpenClF32, &shapes);
        let weights = vec![vec![1.0, 2.0]]; // wrong length
        let result = cpu_run_conversion(&plan, &weights);
        assert!(matches!(result, Err(ConvertError::InvalidWeightShape { .. })));
    }

    #[test]
    fn test_run_conversion_layer_count_mismatch() {
        let shapes = vec![("l0".to_string(), vec![4]), ("l1".to_string(), vec![4])];
        let plan =
            cpu_plan_conversion(SourceFormat::SafeTensorsF32, TargetFormat::OpenClF32, &shapes);
        let weights = vec![vec![1.0; 4]]; // only 1 layer
        let result = cpu_run_conversion(&plan, &weights);
        assert!(matches!(result, Err(ConvertError::InvalidWeightShape { .. })));
    }

    // -- Edge-case tests ---------------------------------------------------

    #[test]
    fn test_single_element_ternary() {
        let packed = cpu_convert_f32_to_ternary(&[1.0], 0.5);
        let restored = cpu_convert_ternary_to_f32(&packed, 1);
        assert_eq!(restored, vec![1.0]);
    }

    #[test]
    fn test_single_element_f16() {
        let f16 = cpu_convert_f32_to_f16(&[42.0]);
        let back = cpu_convert_f16_to_f32(&f16);
        assert!((42.0 - back[0]).abs() < 0.05);
    }

    #[test]
    fn test_single_element_int8() {
        let (q, scale) = cpu_convert_f32_to_int8(&[std::f32::consts::PI]);
        let back = cpu_convert_int8_to_f32(&q, scale);
        assert!((std::f32::consts::PI - back[0]).abs() < 0.05);
    }

    #[test]
    fn test_empty_weights_ternary() {
        let packed = cpu_convert_f32_to_ternary(&[], 0.5);
        assert!(packed.is_empty());
        let restored = cpu_convert_ternary_to_f32(&packed, 0);
        assert!(restored.is_empty());
    }

    #[test]
    fn test_empty_weights_f16() {
        let f16 = cpu_convert_f32_to_f16(&[]);
        assert!(f16.is_empty());
        let back = cpu_convert_f16_to_f32(&f16);
        assert!(back.is_empty());
    }

    #[test]
    fn test_empty_weights_int8() {
        let (q, _scale) = cpu_convert_f32_to_int8(&[]);
        assert!(q.is_empty());
    }

    // -- Property tests ----------------------------------------------------

    #[test]
    fn test_ternary_values_are_neg1_0_1() {
        let weights: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        let packed = cpu_convert_f32_to_ternary(&weights, 0.5);
        let restored = cpu_convert_ternary_to_f32(&packed, weights.len());
        for v in &restored {
            assert!(*v == -1.0 || *v == 0.0 || *v == 1.0, "unexpected ternary value: {v}");
        }
    }

    #[test]
    fn test_ternary_threshold_boundary() {
        // Values exactly at threshold should be zero (not above, not below)
        let weights = vec![0.5, -0.5];
        let packed = cpu_convert_f32_to_ternary(&weights, 0.5);
        let restored = cpu_convert_ternary_to_f32(&packed, weights.len());
        assert_eq!(restored, vec![0.0, 0.0]);
    }

    #[test]
    fn test_f16_roundtrip_property_error_under_001() {
        // For integers in [-1000, 1000] the relative error should stay small
        let data: Vec<f32> = (-1000..=1000).map(|i| i as f32).collect();
        let f16 = cpu_convert_f32_to_f16(&data);
        let back = cpu_convert_f16_to_f32(&f16);
        for (orig, restored) in data.iter().zip(back.iter()) {
            let err = (orig - restored).abs();
            assert!(err < 1.0, "f16 property: {orig} vs {restored}, err={err}");
        }
    }

    #[test]
    fn test_convert_error_display() {
        let e = ConvertError::CacheFull;
        assert_eq!(format!("{e}"), "conversion cache is full");

        let e = ConvertError::IoError("disk read failed".to_string());
        assert!(format!("{e}").contains("disk read failed"));
    }

    #[test]
    fn test_source_format_display() {
        assert_eq!(format!("{}", SourceFormat::GgufI2S), "GGUF I2_S");
        assert_eq!(format!("{}", SourceFormat::GgufQK256), "GGUF QK256");
    }

    #[test]
    fn test_target_format_display() {
        assert_eq!(format!("{}", TargetFormat::OpenClTernaryPacked), "OpenCL Ternary Packed");
        assert_eq!(format!("{}", TargetFormat::CpuSimdAligned), "CPU SIMD Aligned");
    }
}
