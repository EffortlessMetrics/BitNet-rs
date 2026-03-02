//! CUDA quantized operations: quantization, dequantization, and fused
//! quantized matmul for I2_S and TL1 formats.
//!
//! # Kernel strategy
//!
//! This module provides GPU-accelerated quantization and dequantization
//! for BitNet 2-bit signed (I2_S) and table-lookup (TL1) formats, plus
//! fused dequantize-and-matmul operations that avoid materialising full
//! FP32 weight matrices in global memory.
//!
//! - **I2_S**: 2-bit signed ternary {-1, 0, +1}, packed 4 values per byte.
//! - **TL1**: 2-bit table-lookup with a 4-entry codebook per block.
//! - **Quantized matmul**: INT8 accumulator dot-product with per-block
//!   scale post-multiplication.
//! - **Fused dequant+matmul**: Dequantize in shared memory and multiply
//!   in a single kernel launch, eliminating an intermediate buffer.
//!
//! # CPU fallback
//!
//! All public functions have pure-Rust implementations that work on any
//! platform.  GPU-specific CUDA kernel source strings are gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use std::fmt;

use bitnet_common::KernelError;
#[cfg(any(feature = "gpu", feature = "cuda"))]
use bitnet_common::Result;

// ── Configuration ─────────────────────────────────────────────────────

/// Type of zero-point adjustment applied during asymmetric quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroPointType {
    /// No zero-point offset (symmetric quantization).
    None,
    /// Per-block zero-point stored as a separate i8 value.
    PerBlock,
    /// Per-channel zero-point (one i8 per output column).
    PerChannel,
}

/// Configuration for GPU-side quantization / dequantization operations.
#[derive(Debug, Clone)]
pub struct QuantizedOpsConfig {
    /// Number of elements per quantization block.
    pub block_size: usize,
    /// Bit-width of the quantized representation.
    pub num_bits: u8,
    /// Whether the quantization range is symmetric around zero.
    pub symmetric: bool,
    /// Type of zero-point adjustment.
    pub zero_point_type: ZeroPointType,
}

impl Default for QuantizedOpsConfig {
    fn default() -> Self {
        Self { block_size: 32, num_bits: 2, symmetric: true, zero_point_type: ZeroPointType::None }
    }
}

// ── Error type ────────────────────────────────────────────────────────

/// Errors specific to quantized GPU operations.
#[derive(Debug)]
pub enum QuantizedOpsError {
    /// A configuration parameter is invalid.
    InvalidConfig(String),
    /// Input dimensions are incompatible with the operation.
    DimensionMismatch { expected: usize, got: usize },
    /// A numerical result is not finite (NaN or Inf).
    NumericalInstability(String),
    /// An underlying kernel error.
    Kernel(KernelError),
}

impl fmt::Display for QuantizedOpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid quantized-ops config: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::NumericalInstability(msg) => {
                write!(f, "numerical instability: {msg}")
            }
            Self::Kernel(e) => write!(f, "kernel error: {e}"),
        }
    }
}

impl std::error::Error for QuantizedOpsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Kernel(e) => Some(e),
            _ => None,
        }
    }
}

impl From<KernelError> for QuantizedOpsError {
    fn from(e: KernelError) -> Self {
        Self::Kernel(e)
    }
}

// ── Validation helpers ────────────────────────────────────────────────

/// Validate a `QuantizedOpsConfig`, returning an error on invalid settings.
fn validate_config(config: &QuantizedOpsConfig) -> std::result::Result<(), QuantizedOpsError> {
    if config.block_size == 0 {
        return Err(QuantizedOpsError::InvalidConfig("block_size must be > 0".into()));
    }
    if config.num_bits == 0 || config.num_bits > 8 {
        return Err(QuantizedOpsError::InvalidConfig("num_bits must be in [1, 8]".into()));
    }
    if !config.symmetric && config.zero_point_type == ZeroPointType::None {
        return Err(QuantizedOpsError::InvalidConfig(
            "asymmetric quantization requires a zero-point type other than None".into(),
        ));
    }
    Ok(())
}

// ── I2_S encoding helpers ─────────────────────────────────────────────

/// Encode a ternary value to its 2-bit I2_S code.
#[inline(always)]
fn encode_i2s(v: i8) -> u8 {
    match v {
        1 => 0b01,
        -1 => 0b11,
        _ => 0b00,
    }
}

/// Decode a 2-bit I2_S code to its signed integer value.
#[inline(always)]
fn decode_i2s(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

// ── I2_S quantization / dequantization ────────────────────────────────

/// Quantize an f32 tensor to I2_S packed bytes (CPU fallback).
///
/// Four 2-bit ternary values are packed per byte (LSB-first).  Returns
/// `(packed_bytes, per_block_scales)`.
///
/// # Errors
///
/// Returns [`QuantizedOpsError`] if `config` is invalid or the input
/// contains non-finite values.
pub fn quantize_tensor_i2s(
    input: &[f32],
    config: &QuantizedOpsConfig,
) -> std::result::Result<(Vec<u8>, Vec<f32>), QuantizedOpsError> {
    validate_config(config)?;

    if input.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    if let Some(pos) = input.iter().position(|v| !v.is_finite()) {
        return Err(QuantizedOpsError::NumericalInstability(format!(
            "non-finite value at index {pos}: {}",
            input[pos]
        )));
    }

    let block_size = config.block_size;
    let num_blocks = input.len().div_ceil(block_size);
    let packed_len = input.len().div_ceil(4);
    let mut packed = vec![0u8; packed_len];
    let mut scales = Vec::with_capacity(num_blocks);

    for blk in 0..num_blocks {
        let start = blk * block_size;
        let end = (start + block_size).min(input.len());
        let block = &input[start..end];

        let abs_max = block.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        scales.push(abs_max);

        let threshold = abs_max * 0.5;
        for (i, &v) in block.iter().enumerate() {
            let global_idx = start + i;
            let ternary = if abs_max == 0.0 {
                0_i8
            } else if v > threshold {
                1_i8
            } else if v < -threshold {
                -1_i8
            } else {
                0_i8
            };
            let byte_idx = global_idx / 4;
            let bit_off = (global_idx % 4) * 2;
            packed[byte_idx] |= encode_i2s(ternary) << bit_off;
        }
    }
    Ok((packed, scales))
}

/// Dequantize I2_S packed bytes back to f32 values (CPU fallback).
///
/// # Errors
///
/// Returns [`QuantizedOpsError`] if `config` is invalid or buffers are
/// too short for `output_len`.
pub fn dequantize_tensor_i2s(
    packed: &[u8],
    scales: &[f32],
    output_len: usize,
    config: &QuantizedOpsConfig,
) -> std::result::Result<Vec<f32>, QuantizedOpsError> {
    validate_config(config)?;

    if output_len == 0 {
        return Ok(Vec::new());
    }

    let block_size = config.block_size;
    let required_bytes = output_len.div_ceil(4);
    if packed.len() < required_bytes {
        return Err(QuantizedOpsError::DimensionMismatch {
            expected: required_bytes,
            got: packed.len(),
        });
    }
    let num_blocks = output_len.div_ceil(block_size);
    if scales.len() < num_blocks {
        return Err(QuantizedOpsError::DimensionMismatch {
            expected: num_blocks,
            got: scales.len(),
        });
    }

    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_off) & 0x03;
        let val = decode_i2s(bits);
        let blk = i / block_size;
        output.push(val as f32 * scales[blk]);
    }
    Ok(output)
}

// ── TL1 quantization / dequantization ─────────────────────────────────

/// Quantize an f32 tensor to TL1 format (CPU fallback).
///
/// TL1 uses a 4-entry codebook per block.  The codebook is derived from
/// the block's statistics: `{-scale, -scale/3, scale/3, scale}` where
/// `scale = max(|x|)`.  Each value is mapped to its nearest codebook
/// entry (2 bits), packed 4 values per byte.
///
/// Returns `(packed_bytes, per_block_scales)`.
///
/// # Errors
///
/// Returns [`QuantizedOpsError`] if `config` is invalid.
pub fn quantize_tensor_tl1(
    input: &[f32],
    config: &QuantizedOpsConfig,
) -> std::result::Result<(Vec<u8>, Vec<f32>), QuantizedOpsError> {
    validate_config(config)?;

    if input.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    if let Some(pos) = input.iter().position(|v| !v.is_finite()) {
        return Err(QuantizedOpsError::NumericalInstability(format!(
            "non-finite value at index {pos}: {}",
            input[pos]
        )));
    }

    let block_size = config.block_size;
    let num_blocks = input.len().div_ceil(block_size);
    let packed_len = input.len().div_ceil(4);
    let mut packed = vec![0u8; packed_len];
    let mut scales = Vec::with_capacity(num_blocks);

    for blk in 0..num_blocks {
        let start = blk * block_size;
        let end = (start + block_size).min(input.len());
        let block = &input[start..end];

        let abs_max = block.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        scales.push(abs_max);

        // TL1 codebook: {-scale, -scale/3, scale/3, scale}
        // Codes:         0b11     0b10       0b00     0b01
        for (i, &v) in block.iter().enumerate() {
            let code = if abs_max == 0.0 {
                0b00_u8
            } else {
                let normalised = v / abs_max;
                if normalised > 2.0 / 3.0 {
                    0b01 // +scale
                } else if normalised > 0.0 {
                    0b00 // +scale/3
                } else if normalised > -2.0 / 3.0 {
                    0b10 // -scale/3
                } else {
                    0b11 // -scale
                }
            };
            let global_idx = start + i;
            let byte_idx = global_idx / 4;
            let bit_off = (global_idx % 4) * 2;
            packed[byte_idx] |= code << bit_off;
        }
    }
    Ok((packed, scales))
}

/// Decode a 2-bit TL1 code to its codebook multiplier.
#[inline(always)]
fn decode_tl1(bits: u8) -> f32 {
    match bits & 0x03 {
        0b01 => 1.0,        // +scale
        0b00 => 1.0 / 3.0,  // +scale/3
        0b10 => -1.0 / 3.0, // -scale/3
        0b11 => -1.0,       // -scale
        _ => unreachable!(),
    }
}

/// Dequantize TL1 packed bytes back to f32 values (CPU fallback).
///
/// # Errors
///
/// Returns [`QuantizedOpsError`] if `config` is invalid or buffers are
/// too short.
pub fn dequantize_tensor_tl1(
    packed: &[u8],
    scales: &[f32],
    output_len: usize,
    config: &QuantizedOpsConfig,
) -> std::result::Result<Vec<f32>, QuantizedOpsError> {
    validate_config(config)?;

    if output_len == 0 {
        return Ok(Vec::new());
    }

    let block_size = config.block_size;
    let required_bytes = output_len.div_ceil(4);
    if packed.len() < required_bytes {
        return Err(QuantizedOpsError::DimensionMismatch {
            expected: required_bytes,
            got: packed.len(),
        });
    }
    let num_blocks = output_len.div_ceil(block_size);
    if scales.len() < num_blocks {
        return Err(QuantizedOpsError::DimensionMismatch {
            expected: num_blocks,
            got: scales.len(),
        });
    }

    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_off) & 0x03;
        let multiplier = decode_tl1(bits);
        let blk = i / block_size;
        output.push(multiplier * scales[blk]);
    }
    Ok(output)
}

// ── Quantized matrix multiply ─────────────────────────────────────────

/// Perform quantized matrix multiplication (CPU fallback).
///
/// Computes `C = A_quant × B_quant` where both matrices are I2_S packed.
/// Accumulation uses INT8 intermediate values before scaling by the
/// combined per-block scale factors.
///
/// # Arguments
///
/// - `a_packed`: Packed I2_S bytes for matrix A (row-major, `m × k`).
/// - `a_scales`: Per-block scales for A.
/// - `b_packed`: Packed I2_S bytes for matrix B (row-major, `k × n`).
/// - `b_scales`: Per-block scales for B.
/// - `m`, `n`, `k`: Dimensions of the matmul.
/// - `config`: Quantization configuration.
///
/// Returns `m × n` output in row-major order.
///
/// # Errors
///
/// Returns [`QuantizedOpsError`] on dimension or configuration errors.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul(
    a_packed: &[u8],
    a_scales: &[f32],
    b_packed: &[u8],
    b_scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
    config: &QuantizedOpsConfig,
) -> std::result::Result<Vec<f32>, QuantizedOpsError> {
    validate_config(config)?;

    if m == 0 || n == 0 || k == 0 {
        return Ok(vec![0.0; m * n]);
    }

    let block_size = config.block_size;

    // Validate packed buffer sizes
    let a_elements = m * k;
    let b_elements = k * n;
    let a_required = a_elements.div_ceil(4);
    let b_required = b_elements.div_ceil(4);
    if a_packed.len() < a_required {
        return Err(QuantizedOpsError::DimensionMismatch {
            expected: a_required,
            got: a_packed.len(),
        });
    }
    if b_packed.len() < b_required {
        return Err(QuantizedOpsError::DimensionMismatch {
            expected: b_required,
            got: b_packed.len(),
        });
    }

    // Validate scale counts
    let a_blocks = a_elements.div_ceil(block_size);
    let b_blocks = b_elements.div_ceil(block_size);
    if a_scales.len() < a_blocks {
        return Err(QuantizedOpsError::DimensionMismatch {
            expected: a_blocks,
            got: a_scales.len(),
        });
    }
    if b_scales.len() < b_blocks {
        return Err(QuantizedOpsError::DimensionMismatch {
            expected: b_blocks,
            got: b_scales.len(),
        });
    }

    // Unpack A and B to i8, then accumulate with scale post-multiply
    let a_vals = unpack_i2s(a_packed, a_elements);
    let b_vals = unpack_i2s(b_packed, b_elements);

    let mut output = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0_i32;
            for t in 0..k {
                let a_idx = row * k + t;
                let b_idx = t * n + col;
                acc += a_vals[a_idx] as i32 * b_vals[b_idx] as i32;
            }

            // Combined scale: product of per-element block scales
            let a_blk = (row * k) / block_size;
            let b_blk = col / block_size;
            let a_scale = if a_blk < a_scales.len() { a_scales[a_blk] } else { 1.0 };
            let b_scale = if b_blk < b_scales.len() { b_scales[b_blk] } else { 1.0 };

            output[row * n + col] = acc as f32 * a_scale * b_scale;
        }
    }
    Ok(output)
}

/// Unpack I2_S packed bytes into a Vec of signed ternary values.
fn unpack_i2s(packed: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        let bits = if byte_idx < packed.len() { (packed[byte_idx] >> bit_off) & 0x03 } else { 0 };
        out.push(decode_i2s(bits));
    }
    out
}

// ── Fused dequant + matmul ────────────────────────────────────────────

/// Fused dequantization and matrix multiplication (CPU fallback).
///
/// Dequantizes I2_S-packed weight matrix B on-the-fly and multiplies
/// with dense f32 activations A.  Equivalent to dequantizing B to f32
/// and then computing `C = A × B`, but avoids allocating the full
/// dequantized B.
///
/// # Arguments
///
/// - `a`: Dense f32 activations, row-major `m × k`.
/// - `b_packed`: Packed I2_S bytes for weight matrix B (row-major, `k × n`).
/// - `b_scales`: Per-block scales for B.
/// - `m`, `n`, `k`: Dimensions of the matmul.
/// - `config`: Quantization configuration.
///
/// Returns `m × n` output in row-major order.
///
/// # Errors
///
/// Returns [`QuantizedOpsError`] on dimension or configuration errors.
#[allow(clippy::too_many_arguments)]
pub fn fused_dequant_matmul(
    a: &[f32],
    b_packed: &[u8],
    b_scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
    config: &QuantizedOpsConfig,
) -> std::result::Result<Vec<f32>, QuantizedOpsError> {
    validate_config(config)?;

    if m == 0 || n == 0 || k == 0 {
        return Ok(vec![0.0; m * n]);
    }

    let block_size = config.block_size;

    // Validate activation dimensions
    if a.len() < m * k {
        return Err(QuantizedOpsError::DimensionMismatch { expected: m * k, got: a.len() });
    }

    // Validate packed weight buffer
    let b_elements = k * n;
    let b_required = b_elements.div_ceil(4);
    if b_packed.len() < b_required {
        return Err(QuantizedOpsError::DimensionMismatch {
            expected: b_required,
            got: b_packed.len(),
        });
    }

    let b_num_blocks = b_elements.div_ceil(block_size);
    if b_scales.len() < b_num_blocks {
        return Err(QuantizedOpsError::DimensionMismatch {
            expected: b_num_blocks,
            got: b_scales.len(),
        });
    }

    let mut output = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0_f32;
            for t in 0..k {
                let b_idx = t * n + col;
                let byte_idx = b_idx / 4;
                let bit_off = (b_idx % 4) * 2;
                let bits = (b_packed[byte_idx] >> bit_off) & 0x03;
                let b_val = decode_i2s(bits);
                let b_blk = b_idx / block_size;
                let b_dequant = b_val as f32 * b_scales[b_blk];
                acc += a[row * k + t] * b_dequant;
            }
            output[row * n + col] = acc;
        }
    }
    Ok(output)
}

// ── GPU launch stubs ──────────────────────────────────────────────────

/// CUDA C source for quantized ops kernels.
///
/// Contains I2_S quantize/dequantize and TL1 quantize/dequantize
/// kernels, plus a fused dequant+matmul kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const QUANTIZED_OPS_KERNEL_SRC: &str = r#"
// I2_S quantize kernel — maps f32 to 2-bit ternary, packed 4-per-byte.
extern "C" __global__ void quantized_ops_i2s_quantize(
    const float* __restrict__ input,
    unsigned char* __restrict__ packed,
    const float* __restrict__ block_scales,
    int n,
    int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int blk = i / block_size;
        float abs_max = block_scales[blk];
        float threshold = abs_max * 0.5f;
        float v = input[i];
        signed char q;
        if (abs_max == 0.0f) q = 0;
        else if (v > threshold) q = 1;
        else if (v < -threshold) q = -1;
        else q = 0;
        unsigned char code;
        if (q == 1) code = 0x01;
        else if (q == -1) code = 0x03;
        else code = 0x00;
        int byte_idx = i / 4;
        int bit_off  = (i % 4) * 2;
        atomicOr(&packed[byte_idx], code << bit_off);
    }
}

// I2_S dequantize kernel — unpack 2-bit codes to f32.
extern "C" __global__ void quantized_ops_i2s_dequantize(
    const unsigned char* __restrict__ packed,
    const float* __restrict__ block_scales,
    float* __restrict__ output,
    int n,
    int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int byte_idx = i / 4;
        int bit_off  = (i % 4) * 2;
        unsigned char bits = (packed[byte_idx] >> bit_off) & 0x03;
        signed char val;
        if (bits == 0x01) val = 1;
        else if (bits == 0x03) val = -1;
        else val = 0;
        int blk = i / block_size;
        output[i] = (float)val * block_scales[blk];
    }
}

// Fused dequant + matmul kernel.
// Grid: (ceil(n/TILE), ceil(m/TILE))  Block: (TILE, 1)
extern "C" __global__ void quantized_ops_fused_dequant_matmul(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b_packed,
    const float* __restrict__ b_scales,
    float* __restrict__ c,
    int m, int n, int k,
    int block_size)
{
    int row = blockIdx.y * blockDim.x + threadIdx.x;
    int col = blockIdx.x;
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (int t = 0; t < k; t++) {
        int b_idx = t * n + col;
        int byte_idx = b_idx / 4;
        int bit_off  = (b_idx % 4) * 2;
        unsigned char bits = (b_packed[byte_idx] >> bit_off) & 0x03;
        signed char bv;
        if (bits == 0x01) bv = 1;
        else if (bits == 0x03) bv = -1;
        else bv = 0;
        int blk = b_idx / block_size;
        float b_val = (float)bv * b_scales[blk];
        acc += a[row * k + t] * b_val;
    }
    c[row * n + col] = acc;
}
"#;

/// Launch the quantized-ops GPU kernel (stub — returns error without CUDA runtime).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_quantized_ops_i2s(
    _input: &[f32],
    _config: &QuantizedOpsConfig,
) -> Result<(Vec<u8>, Vec<f32>)> {
    Err(KernelError::GpuError {
        reason: "quantized_ops_i2s: CUDA runtime dispatch not yet implemented".into(),
    }
    .into())
}

/// Launch the fused dequant+matmul GPU kernel (stub).
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn launch_fused_dequant_matmul(
    _a: &[f32],
    _b_packed: &[u8],
    _b_scales: &[f32],
    _m: usize,
    _n: usize,
    _k: usize,
    _config: &QuantizedOpsConfig,
) -> Result<Vec<f32>> {
    Err(KernelError::GpuError {
        reason: "fused_dequant_matmul: CUDA runtime dispatch not yet implemented".into(),
    }
    .into())
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> QuantizedOpsConfig {
        QuantizedOpsConfig::default()
    }

    fn config_with_block(block_size: usize) -> QuantizedOpsConfig {
        QuantizedOpsConfig { block_size, ..default_config() }
    }

    #[allow(clippy::float_cmp)]
    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (tol={tol})");
        }
    }

    // ── QuantizedOpsConfig ────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = default_config();
        assert_eq!(cfg.block_size, 32);
        assert_eq!(cfg.num_bits, 2);
        assert!(cfg.symmetric);
        assert_eq!(cfg.zero_point_type, ZeroPointType::None);
    }

    #[test]
    fn config_validates_zero_block_size() {
        let cfg = QuantizedOpsConfig { block_size: 0, ..default_config() };
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn config_validates_zero_num_bits() {
        let cfg = QuantizedOpsConfig { num_bits: 0, ..default_config() };
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn config_validates_num_bits_overflow() {
        let cfg = QuantizedOpsConfig { num_bits: 9, ..default_config() };
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn config_validates_asymmetric_requires_zero_point() {
        let cfg = QuantizedOpsConfig {
            symmetric: false,
            zero_point_type: ZeroPointType::None,
            ..default_config()
        };
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn config_asymmetric_per_block_ok() {
        let cfg = QuantizedOpsConfig {
            symmetric: false,
            zero_point_type: ZeroPointType::PerBlock,
            ..default_config()
        };
        assert!(validate_config(&cfg).is_ok());
    }

    // ── QuantizedOpsError Display ─────────────────────────────────

    #[test]
    fn error_display_invalid_config() {
        let e = QuantizedOpsError::InvalidConfig("test".into());
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn error_display_dimension_mismatch() {
        let e = QuantizedOpsError::DimensionMismatch { expected: 10, got: 5 };
        let msg = e.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn error_display_numerical() {
        let e = QuantizedOpsError::NumericalInstability("nan detected".into());
        assert!(e.to_string().contains("nan detected"));
    }

    #[test]
    fn error_source_kernel_variant() {
        let inner = KernelError::InvalidArguments { reason: "bad".into() };
        let e = QuantizedOpsError::Kernel(inner);
        assert!(std::error::Error::source(&e).is_some());
    }

    #[test]
    fn error_source_non_kernel_variant() {
        let e = QuantizedOpsError::InvalidConfig("x".into());
        assert!(std::error::Error::source(&e).is_none());
    }

    // ── I2_S encode / decode roundtrip ────────────────────────────

    #[test]
    fn i2s_encode_decode_roundtrip() {
        for val in [-1_i8, 0, 1] {
            let code = encode_i2s(val);
            let decoded = decode_i2s(code);
            assert_eq!(val, decoded, "roundtrip failed for {val}");
        }
    }

    #[test]
    fn i2s_decode_unused_code() {
        assert_eq!(decode_i2s(0b10), 0);
    }

    // ── quantize_tensor_i2s ───────────────────────────────────────

    #[test]
    fn i2s_quantize_empty_input() {
        let (packed, scales) = quantize_tensor_i2s(&[], &default_config()).unwrap();
        assert!(packed.is_empty());
        assert!(scales.is_empty());
    }

    #[test]
    fn i2s_quantize_basic_values() {
        let input = vec![1.0, -1.0, 0.0, 1.0];
        let cfg = config_with_block(4);
        let (packed, scales) = quantize_tensor_i2s(&input, &cfg).unwrap();
        assert_eq!(packed.len(), 1);
        assert_eq!(scales.len(), 1);
        assert!((scales[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn i2s_quantize_rejects_nan() {
        let input = vec![1.0, f32::NAN, 0.0];
        let result = quantize_tensor_i2s(&input, &default_config());
        assert!(result.is_err());
    }

    #[test]
    fn i2s_quantize_rejects_inf() {
        let input = vec![f32::INFINITY, 0.0];
        let result = quantize_tensor_i2s(&input, &default_config());
        assert!(result.is_err());
    }

    #[test]
    fn i2s_quantize_rejects_zero_block_size() {
        let cfg = QuantizedOpsConfig { block_size: 0, ..default_config() };
        assert!(quantize_tensor_i2s(&[1.0], &cfg).is_err());
    }

    #[test]
    fn i2s_quantize_all_zeros() {
        let input = vec![0.0; 16];
        let cfg = config_with_block(8);
        let (packed, scales) = quantize_tensor_i2s(&input, &cfg).unwrap();
        assert!(packed.iter().all(|&b| b == 0));
        assert!(scales.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn i2s_quantize_multiple_blocks() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0) - 1.0).collect();
        let cfg = config_with_block(32);
        let (_, scales) = quantize_tensor_i2s(&input, &cfg).unwrap();
        assert_eq!(scales.len(), 2);
    }

    // ── dequantize_tensor_i2s ─────────────────────────────────────

    #[test]
    fn i2s_dequant_empty() {
        let out = dequantize_tensor_i2s(&[], &[], 0, &default_config()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn i2s_dequant_rejects_short_packed() {
        let cfg = config_with_block(4);
        let result = dequantize_tensor_i2s(&[0], &[1.0, 1.0], 5, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn i2s_dequant_rejects_short_scales() {
        let cfg = config_with_block(4);
        let result = dequantize_tensor_i2s(&[0, 0], &[1.0], 5, &cfg);
        assert!(result.is_err());
    }

    // ── I2_S roundtrip ────────────────────────────────────────────

    #[test]
    fn i2s_roundtrip_sign_preservation() {
        let input = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let cfg = config_with_block(8);
        let (packed, scales) = quantize_tensor_i2s(&input, &cfg).unwrap();
        let output = dequantize_tensor_i2s(&packed, &scales, input.len(), &cfg).unwrap();
        for (i, (&orig, &deq)) in input.iter().zip(output.iter()).enumerate() {
            if orig > 0.0 {
                assert!(deq > 0.0, "positive sign lost at {i}");
            } else if orig < 0.0 {
                assert!(deq < 0.0, "negative sign lost at {i}");
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn i2s_roundtrip_all_zeros() {
        let input = vec![0.0; 32];
        let cfg = config_with_block(32);
        let (packed, scales) = quantize_tensor_i2s(&input, &cfg).unwrap();
        let output = dequantize_tensor_i2s(&packed, &scales, input.len(), &cfg).unwrap();
        assert!(output.iter().all(|&v| v == 0.0));
    }

    // ── quantize_tensor_tl1 ──────────────────────────────────────

    #[test]
    fn tl1_quantize_empty() {
        let (packed, scales) = quantize_tensor_tl1(&[], &default_config()).unwrap();
        assert!(packed.is_empty());
        assert!(scales.is_empty());
    }

    #[test]
    fn tl1_quantize_basic() {
        let input = vec![1.0, -1.0, 0.2, -0.2];
        let cfg = config_with_block(4);
        let (packed, scales) = quantize_tensor_tl1(&input, &cfg).unwrap();
        assert_eq!(packed.len(), 1);
        assert_eq!(scales.len(), 1);
        assert!((scales[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tl1_quantize_rejects_nan() {
        let result = quantize_tensor_tl1(&[f32::NAN], &default_config());
        assert!(result.is_err());
    }

    // ── dequantize_tensor_tl1 ─────────────────────────────────────

    #[test]
    fn tl1_dequant_empty() {
        let out = dequantize_tensor_tl1(&[], &[], 0, &default_config()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn tl1_dequant_rejects_short_packed() {
        let cfg = config_with_block(4);
        let result = dequantize_tensor_tl1(&[0], &[1.0, 1.0], 5, &cfg);
        assert!(result.is_err());
    }

    // ── TL1 roundtrip ─────────────────────────────────────────────

    #[test]
    fn tl1_roundtrip_sign_preservation() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.1, -0.1, 0.9, -0.9];
        let cfg = config_with_block(8);
        let (packed, scales) = quantize_tensor_tl1(&input, &cfg).unwrap();
        let output = dequantize_tensor_tl1(&packed, &scales, input.len(), &cfg).unwrap();
        for (i, (&orig, &deq)) in input.iter().zip(output.iter()).enumerate() {
            if orig > 0.0 {
                assert!(deq > 0.0, "positive sign lost at {i}");
            } else if orig < 0.0 {
                assert!(deq < 0.0, "negative sign lost at {i}");
            }
        }
    }

    #[test]
    fn tl1_dequant_codebook_values() {
        let scale = 3.0;
        // Pack: code 0b01 (+scale), 0b00 (+scale/3), 0b10 (-scale/3), 0b11 (-scale)
        let packed = vec![(0b01) | (0b00 << 2) | (0b10 << 4) | (0b11 << 6)];
        let scales = vec![scale];
        let cfg = config_with_block(4);
        let output = dequantize_tensor_tl1(&packed, &scales, 4, &cfg).unwrap();
        assert_close(&output, &[3.0, 1.0, -1.0, -3.0], 0.01);
    }

    // ── quantized_matmul ──────────────────────────────────────────

    #[test]
    fn qmatmul_zero_dimensions() {
        let cfg = default_config();
        let out = quantized_matmul(&[], &[], &[], &[], 0, 4, 4, &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn qmatmul_identity_like() {
        // 1×2 @ 2×1 with all weights = +1
        let a_packed = vec![(0b01) | (0b01 << 2)]; // two +1 values
        let b_packed = vec![(0b01) | (0b01 << 2)]; // two +1 values
        let a_scales = vec![1.0];
        let b_scales = vec![1.0];
        let cfg = config_with_block(4);
        let out =
            quantized_matmul(&a_packed, &a_scales, &b_packed, &b_scales, 1, 1, 2, &cfg).unwrap();
        // 1*1 + 1*1 = 2, scaled by 1.0 * 1.0 = 2.0
        assert_eq!(out.len(), 1);
        assert!((out[0] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn qmatmul_rejects_short_a_packed() {
        let cfg = config_with_block(4);
        let result = quantized_matmul(&[], &[1.0], &[0], &[1.0], 2, 2, 2, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn qmatmul_rejects_short_b_packed() {
        let cfg = config_with_block(4);
        let result = quantized_matmul(&[0], &[1.0], &[], &[1.0], 1, 2, 2, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn qmatmul_rejects_short_a_scales() {
        let cfg = config_with_block(4);
        let result = quantized_matmul(&[0], &[], &[0], &[1.0], 1, 1, 2, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn qmatmul_rejects_short_b_scales() {
        let cfg = config_with_block(4);
        let result = quantized_matmul(&[0], &[1.0], &[0], &[], 1, 1, 2, &cfg);
        assert!(result.is_err());
    }

    // ── fused_dequant_matmul ──────────────────────────────────────

    #[test]
    fn fused_dequant_zero_dimensions() {
        let cfg = default_config();
        let out = fused_dequant_matmul(&[], &[], &[], 0, 4, 4, &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn fused_dequant_basic() {
        // A = [[1.0, 2.0]], B weights all +1, scale=1.0 → output = [3.0]
        let a = vec![1.0, 2.0]; // 1×2
        let b_packed = vec![(0b01) | (0b01 << 2)]; // two +1 values, k=2 n=1
        let b_scales = vec![1.0];
        let cfg = config_with_block(4);
        let out = fused_dequant_matmul(&a, &b_packed, &b_scales, 1, 1, 2, &cfg).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn fused_dequant_rejects_short_a() {
        let cfg = config_with_block(4);
        let result = fused_dequant_matmul(&[1.0], &[0], &[1.0], 2, 1, 2, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn fused_dequant_rejects_short_b() {
        let cfg = config_with_block(4);
        let result = fused_dequant_matmul(&[1.0, 2.0], &[], &[1.0], 1, 1, 2, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn fused_dequant_rejects_short_scales() {
        let cfg = config_with_block(4);
        let b_packed = vec![(0b01) | (0b01 << 2)];
        let result = fused_dequant_matmul(&[1.0, 2.0], &b_packed, &[], 1, 1, 2, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn fused_dequant_matches_separate_ops() {
        let a = vec![1.0, -0.5, 0.3, 0.7]; // 2×2
        let b_input = vec![0.8, -0.6, 0.4, -0.9]; // 2×2 weights
        let cfg = config_with_block(4);

        // Quantize B
        let (b_packed, b_scales) = quantize_tensor_i2s(&b_input, &cfg).unwrap();

        // Fused path
        let fused_out = fused_dequant_matmul(&a, &b_packed, &b_scales, 2, 2, 2, &cfg).unwrap();

        // Separate: dequantize B, then dense matmul
        let b_deq = dequantize_tensor_i2s(&b_packed, &b_scales, 4, &cfg).unwrap();
        let mut separate_out = vec![0.0_f32; 4];
        for row in 0..2 {
            for col in 0..2 {
                let mut acc = 0.0_f32;
                for t in 0..2 {
                    acc += a[row * 2 + t] * b_deq[t * 2 + col];
                }
                separate_out[row * 2 + col] = acc;
            }
        }

        assert_close(&fused_out, &separate_out, 1e-5);
    }

    #[test]
    fn fused_dequant_negative_weights() {
        // A = [[2.0]], B = [[-1]] (packed as -1 with scale 1.0)
        let a = vec![2.0];
        let b_packed = vec![0b11]; // -1 in I2_S
        let b_scales = vec![1.0];
        let cfg = config_with_block(4);
        let out = fused_dequant_matmul(&a, &b_packed, &b_scales, 1, 1, 1, &cfg).unwrap();
        assert!((out[0] - (-2.0)).abs() < 1e-4);
    }

    // ── Large batch integration ───────────────────────────────────

    #[test]
    fn i2s_large_batch_roundtrip() {
        let input: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.1).sin()).collect();
        let cfg = config_with_block(32);
        let (packed, scales) = quantize_tensor_i2s(&input, &cfg).unwrap();
        assert_eq!(scales.len(), 32);
        let output = dequantize_tensor_i2s(&packed, &scales, 1024, &cfg).unwrap();
        assert_eq!(output.len(), 1024);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tl1_large_batch_roundtrip() {
        let input: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.05).cos()).collect();
        let cfg = config_with_block(64);
        let (packed, scales) = quantize_tensor_tl1(&input, &cfg).unwrap();
        assert_eq!(scales.len(), 16);
        let output = dequantize_tensor_tl1(&packed, &scales, 1024, &cfg).unwrap();
        assert_eq!(output.len(), 1024);
        assert!(output.iter().all(|v| v.is_finite()));
    }
}
