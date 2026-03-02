//! HIP quantization kernel stubs for INT2/INT4 with CPU fallback.
//!
//! Mirrors the CUDA quantization interface in [`crate::cuda::quantize`]
//! but targets AMD GPUs via HIP. Provides ternary ({-1, 0, 1}) and
//! INT2/INT4 quantization with configurable per-block scale strategies.
//!
//! # Encoding format
//!
//! INT2: 4 values packed per byte, LSB-first.
//! INT4: 2 values packed per byte, LSB-first.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Configuration ────────────────────────────────────────────────────

/// Quantization bit width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipQuantBits {
    /// 2-bit quantization (ternary: -1, 0, +1).
    Int2,
    /// 4-bit quantization (-8..+7).
    Int4,
}

/// Scale calibration method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipQuantMethod {
    /// Scale = max(|x|) in the block.
    AbsMax,
    /// Scale from symmetric range.
    Symmetric,
}

/// Configuration for a HIP quantization pass.
#[derive(Debug, Clone)]
pub struct HipQuantizeConfig {
    /// Number of elements per quantization block.
    pub block_size: usize,
    /// Bit width.
    pub bits: HipQuantBits,
    /// Calibration method.
    pub method: HipQuantMethod,
}

impl Default for HipQuantizeConfig {
    fn default() -> Self {
        Self { block_size: 32, bits: HipQuantBits::Int2, method: HipQuantMethod::AbsMax }
    }
}

impl HipQuantizeConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.block_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "block_size must be non-zero".into(),
            }
            .into());
        }
        Ok(())
    }

    /// Number of values packed per byte for this bit width.
    pub fn values_per_byte(&self) -> usize {
        match self.bits {
            HipQuantBits::Int2 => 4,
            HipQuantBits::Int4 => 2,
        }
    }

    /// Output byte count for a given number of input elements.
    pub fn output_bytes(&self, num_elements: usize) -> usize {
        num_elements.div_ceil(self.values_per_byte())
    }

    /// Number of scale values for a given number of input elements.
    pub fn num_scales(&self, num_elements: usize) -> usize {
        num_elements.div_ceil(self.block_size)
    }
}

// ── HIP kernel source (stub) ────────────────────────────────────────

/// HIP C source for quantization kernels.
#[cfg(feature = "rocm")]
pub const HIP_QUANTIZE_INT2_KERNEL_SRC: &str = r#"
// TODO: HIP INT2 quantization kernel
extern "C" __global__ void quantize_int2(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    float* __restrict__ scales,
    int num_elements, int block_size)
{
    // Stub — to be implemented
}
"#;

/// HIP C source for dequantization kernels.
#[cfg(feature = "rocm")]
pub const HIP_DEQUANTIZE_INT2_KERNEL_SRC: &str = r#"
// TODO: HIP INT2 dequantization kernel
extern "C" __global__ void dequantize_int2(
    const unsigned char* __restrict__ input,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int num_elements, int block_size)
{
    // Stub — to be implemented
}
"#;

// ── CPU fallback: scale calibration ──────────────────────────────────

/// Compute per-block scale factors.
pub fn hip_calibrate_scales(input: &[f32], config: &HipQuantizeConfig) -> Result<Vec<f32>> {
    config.validate()?;
    let num_blocks = input.len().div_ceil(config.block_size);
    let mut scales = Vec::with_capacity(num_blocks);

    for blk in 0..num_blocks {
        let start = blk * config.block_size;
        let end = (start + config.block_size).min(input.len());
        let block = &input[start..end];
        let scale = match config.method {
            HipQuantMethod::AbsMax | HipQuantMethod::Symmetric => {
                block.iter().fold(0.0_f32, |m, &v| m.max(v.abs()))
            }
        };
        scales.push(scale);
    }
    Ok(scales)
}

// ── CPU fallback: INT2 quantize/dequantize ───────────────────────────

/// Quantize f32 values to INT2 (ternary: -1, 0, +1), packing 4 per byte.
pub fn hip_quantize_int2_cpu(
    input: &[f32],
    output: &mut [u8],
    scales: &mut [f32],
    config: &HipQuantizeConfig,
) -> Result<()> {
    config.validate()?;
    let needed_bytes = config.output_bytes(input.len());
    if output.len() < needed_bytes {
        return Err(KernelError::InvalidArguments {
            reason: format!("output buffer too small: {} < {}", output.len(), needed_bytes),
        }
        .into());
    }
    let needed_scales = config.num_scales(input.len());
    if scales.len() < needed_scales {
        return Err(KernelError::InvalidArguments {
            reason: format!("scales buffer too small: {} < {}", scales.len(), needed_scales),
        }
        .into());
    }

    // Compute scales
    let computed_scales = hip_calibrate_scales(input, config)?;
    scales[..needed_scales].copy_from_slice(&computed_scales);

    // Quantize: map to -1, 0, +1
    for (i, &val) in input.iter().enumerate() {
        let block_idx = i / config.block_size;
        let scale = scales[block_idx];
        let q = if scale == 0.0 {
            0i8
        } else {
            let normalized = val / scale;
            if normalized > 0.5 {
                1
            } else if normalized < -0.5 {
                -1
            } else {
                0
            }
        };
        // Pack: 4 values per byte, 2 bits each, LSB-first
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let encoded = (q & 0x03) as u8;
        if bit_offset == 0 {
            output[byte_idx] = encoded;
        } else {
            output[byte_idx] |= encoded << bit_offset;
        }
    }
    Ok(())
}

/// Dequantize INT2 packed data back to f32.
pub fn hip_dequantize_int2_cpu(
    input: &[u8],
    scales: &[f32],
    output: &mut [f32],
    num_elements: usize,
    config: &HipQuantizeConfig,
) -> Result<()> {
    config.validate()?;
    if output.len() < num_elements {
        return Err(
            KernelError::InvalidArguments { reason: "output buffer too small".into() }.into()
        );
    }

    for i in 0..num_elements {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let raw = (input[byte_idx] >> bit_offset) & 0x03;
        // Decode 2-bit signed: 0b01=+1, 0b11=-1, else=0
        let val = match raw {
            0b01 => 1.0f32,
            0b11 => -1.0f32,
            _ => 0.0f32,
        };
        let block_idx = i / config.block_size;
        let scale = scales.get(block_idx).copied().unwrap_or(1.0);
        output[i] = val * scale;
    }
    Ok(())
}

// ── CPU fallback: INT4 quantize/dequantize ───────────────────────────

/// Quantize f32 values to INT4 (-8..+7), packing 2 per byte.
pub fn hip_quantize_int4_cpu(
    input: &[f32],
    output: &mut [u8],
    scales: &mut [f32],
    config: &HipQuantizeConfig,
) -> Result<()> {
    config.validate()?;
    let needed_bytes = config.output_bytes(input.len());
    if output.len() < needed_bytes {
        return Err(KernelError::InvalidArguments {
            reason: format!("output buffer too small: {} < {}", output.len(), needed_bytes),
        }
        .into());
    }

    let computed_scales = hip_calibrate_scales(input, config)?;
    let needed_scales = config.num_scales(input.len());
    scales[..needed_scales].copy_from_slice(&computed_scales);

    for (i, &val) in input.iter().enumerate() {
        let block_idx = i / config.block_size;
        let scale = scales[block_idx];
        let q = if scale == 0.0 { 0i8 } else { (val / scale * 7.0).round().clamp(-8.0, 7.0) as i8 };
        let byte_idx = i / 2;
        let nibble = (q & 0x0F) as u8;
        if i % 2 == 0 {
            output[byte_idx] = nibble;
        } else {
            output[byte_idx] |= nibble << 4;
        }
    }
    Ok(())
}

/// Dequantize INT4 packed data back to f32.
pub fn hip_dequantize_int4_cpu(
    input: &[u8],
    scales: &[f32],
    output: &mut [f32],
    num_elements: usize,
    config: &HipQuantizeConfig,
) -> Result<()> {
    config.validate()?;
    if output.len() < num_elements {
        return Err(
            KernelError::InvalidArguments { reason: "output buffer too small".into() }.into()
        );
    }

    for i in 0..num_elements {
        let byte_idx = i / 2;
        let raw = if i % 2 == 0 { input[byte_idx] & 0x0F } else { (input[byte_idx] >> 4) & 0x0F };
        // Sign-extend from 4-bit
        let val = if raw & 0x08 != 0 { (raw as i8 | !0x0F) as f32 } else { raw as f32 };
        let block_idx = i / config.block_size;
        let scale = scales.get(block_idx).copied().unwrap_or(1.0);
        output[i] = val * scale / 7.0;
    }
    Ok(())
}

/// Launch the HIP quantization kernel on the GPU.
///
/// Stub — returns error until HIP runtime integration.
#[cfg(feature = "rocm")]
pub fn launch_hip_quantize(
    _input: &[f32],
    _output: &mut [u8],
    _scales: &mut [f32],
    _config: &HipQuantizeConfig,
) -> Result<()> {
    Err(BitNetError::Kernel(KernelError::ExecutionFailed {
        reason: "HIP quantization kernel is not yet implemented".into(),
    }))
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default() {
        let cfg = HipQuantizeConfig::default();
        assert_eq!(cfg.block_size, 32);
        assert_eq!(cfg.bits, HipQuantBits::Int2);
        assert_eq!(cfg.method, HipQuantMethod::AbsMax);
    }

    #[test]
    fn config_validate_ok() {
        assert!(HipQuantizeConfig::default().validate().is_ok());
    }

    #[test]
    fn config_validate_zero_block() {
        let cfg = HipQuantizeConfig { block_size: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn values_per_byte_int2() {
        let cfg = HipQuantizeConfig { bits: HipQuantBits::Int2, ..Default::default() };
        assert_eq!(cfg.values_per_byte(), 4);
    }

    #[test]
    fn values_per_byte_int4() {
        let cfg = HipQuantizeConfig { bits: HipQuantBits::Int4, ..Default::default() };
        assert_eq!(cfg.values_per_byte(), 2);
    }

    #[test]
    fn output_bytes_calculation() {
        let cfg = HipQuantizeConfig { bits: HipQuantBits::Int2, ..Default::default() };
        assert_eq!(cfg.output_bytes(8), 2);
        assert_eq!(cfg.output_bytes(9), 3);
    }

    #[test]
    fn num_scales_calculation() {
        let cfg = HipQuantizeConfig { block_size: 4, ..Default::default() };
        assert_eq!(cfg.num_scales(8), 2);
        assert_eq!(cfg.num_scales(9), 3);
    }

    #[test]
    fn calibrate_scales_absmax() {
        let input = vec![1.0, -2.0, 0.5, -0.5, 3.0, 0.0, -1.0, 0.0];
        let cfg = HipQuantizeConfig { block_size: 4, ..Default::default() };
        let scales = hip_calibrate_scales(&input, &cfg).unwrap();
        assert_eq!(scales.len(), 2);
        assert!((scales[0] - 2.0).abs() < 1e-6);
        assert!((scales[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn quantize_int2_basic() {
        let input = vec![1.0, -1.0, 0.0, 0.5];
        let mut output = vec![0u8; 1];
        let mut scales = vec![0.0f32; 1];
        let cfg =
            HipQuantizeConfig { block_size: 4, bits: HipQuantBits::Int2, ..Default::default() };
        hip_quantize_int2_cpu(&input, &mut output, &mut scales, &cfg).unwrap();
        assert!(scales[0] > 0.0);
    }

    #[test]
    fn quantize_int2_output_too_small() {
        let input = vec![1.0; 8];
        let mut output = vec![0u8; 1]; // need 2
        let mut scales = vec![0.0f32; 1];
        let cfg =
            HipQuantizeConfig { block_size: 8, bits: HipQuantBits::Int2, ..Default::default() };
        assert!(hip_quantize_int2_cpu(&input, &mut output, &mut scales, &cfg,).is_err());
    }

    #[test]
    fn quantize_int2_scales_too_small() {
        let input = vec![1.0; 8];
        let mut output = vec![0u8; 2];
        let mut scales = vec![0.0f32; 0];
        let cfg =
            HipQuantizeConfig { block_size: 4, bits: HipQuantBits::Int2, ..Default::default() };
        assert!(hip_quantize_int2_cpu(&input, &mut output, &mut scales, &cfg,).is_err());
    }

    #[test]
    fn dequantize_int2_basic() {
        // Pack: +1 at position 0, -1 at position 1
        // pos0=01(+1), pos1=11(-1), pos2=00(0), pos3=00(0)
        let packed = vec![0b00_00_11_01u8];
        let scales = vec![2.0f32];
        let mut output = vec![0.0f32; 4];
        let cfg =
            HipQuantizeConfig { block_size: 4, bits: HipQuantBits::Int2, ..Default::default() };
        hip_dequantize_int2_cpu(&packed, &scales, &mut output, 4, &cfg).unwrap();
        assert!((output[0] - 2.0).abs() < 1e-6); // +1 * 2.0
        assert!((output[1] - (-2.0)).abs() < 1e-6); // -1 * 2.0
        assert!((output[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn quantize_int4_basic() {
        let input = vec![0.5, -0.5, 1.0, -1.0];
        let mut output = vec![0u8; 2];
        let mut scales = vec![0.0f32; 1];
        let cfg = HipQuantizeConfig {
            block_size: 4,
            bits: HipQuantBits::Int4,
            method: HipQuantMethod::AbsMax,
        };
        hip_quantize_int4_cpu(&input, &mut output, &mut scales, &cfg).unwrap();
        assert!(scales[0] > 0.0);
    }

    #[test]
    fn quantize_int4_output_too_small() {
        let input = vec![1.0; 4];
        let mut output = vec![0u8; 1]; // need 2
        let mut scales = vec![0.0f32; 1];
        let cfg =
            HipQuantizeConfig { block_size: 4, bits: HipQuantBits::Int4, ..Default::default() };
        assert!(hip_quantize_int4_cpu(&input, &mut output, &mut scales, &cfg,).is_err());
    }

    #[test]
    fn dequantize_int4_basic() {
        // Pack: +3 at position 0, -3 at position 1
        // low nibble=0011(+3), high=1101(-3 in 4-bit signed)
        let packed = vec![0b1101_0011u8];
        let scales = vec![7.0f32];
        let mut output = vec![0.0f32; 2];
        let cfg =
            HipQuantizeConfig { block_size: 4, bits: HipQuantBits::Int4, ..Default::default() };
        hip_dequantize_int4_cpu(&packed, &scales, &mut output, 2, &cfg).unwrap();
        // +3 * 7.0 / 7.0 = 3.0
        assert!((output[0] - 3.0).abs() < 1e-5);
        // -3 * 7.0 / 7.0 = -3.0
        assert!((output[1] - (-3.0)).abs() < 1e-5);
    }

    #[test]
    fn quantize_zero_input() {
        let input = vec![0.0; 4];
        let mut output = vec![0u8; 1];
        let mut scales = vec![0.0f32; 1];
        let cfg =
            HipQuantizeConfig { block_size: 4, bits: HipQuantBits::Int2, ..Default::default() };
        hip_quantize_int2_cpu(&input, &mut output, &mut scales, &cfg).unwrap();
        assert_eq!(scales[0], 0.0);
        assert_eq!(output[0], 0); // all zeros
    }

    #[test]
    fn bits_enum_equality() {
        assert_eq!(HipQuantBits::Int2, HipQuantBits::Int2);
        assert_ne!(HipQuantBits::Int2, HipQuantBits::Int4);
    }

    #[test]
    fn method_enum_equality() {
        assert_eq!(HipQuantMethod::AbsMax, HipQuantMethod::AbsMax);
        assert_ne!(HipQuantMethod::AbsMax, HipQuantMethod::Symmetric);
    }
}
