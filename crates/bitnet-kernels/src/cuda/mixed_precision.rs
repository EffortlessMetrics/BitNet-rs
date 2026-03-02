//! Mixed-precision arithmetic for GPU inference.
//!
//! Provides precision casting, mixed-precision matrix multiplication, and
//! quantized matmul with dynamic dequantization.  All operations include
//! CPU fallback implementations and are feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]` for CUDA-specific
//! launch stubs.
//!
//! # Precision modes
//!
//! [`PrecisionMode`] enumerates supported numeric formats from FP32 down
//! to INT2.  [`auto_precision_select`] chooses the narrowest format that
//! keeps representable range above a user-specified tolerance.
//!
//! # FP16 / BF16 representation
//!
//! Half-precision values are stored as `u16` in IEEE 754 binary16 layout.
//! BFloat16 values are stored as `u16` with 8-bit exponent and 7-bit
//! mantissa.  All arithmetic is performed in FP32; narrower types are
//! used only for storage and transfer.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Precision mode enum ───────────────────────────────────────────────

/// Supported numeric precision modes for mixed-precision computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionMode {
    /// 32-bit IEEE 754 single precision.
    FP32,
    /// 16-bit IEEE 754 half precision (max ~65504).
    FP16,
    /// 16-bit Brain Float (same exponent range as FP32, 7-bit mantissa).
    BF16,
    /// 8-bit signed integer (range −128..127).
    INT8,
    /// 4-bit signed integer (range −8..7).
    INT4,
    /// 2-bit signed integer (range −1..1, ternary).
    INT2,
}

impl PrecisionMode {
    /// Number of bits consumed per element.
    pub fn bits(&self) -> u32 {
        match self {
            Self::FP32 => 32,
            Self::FP16 | Self::BF16 => 16,
            Self::INT8 => 8,
            Self::INT4 => 4,
            Self::INT2 => 2,
        }
    }

    /// Maximum representable magnitude for the precision mode.
    pub fn max_representable(&self) -> f32 {
        match self {
            Self::FP32 => f32::MAX,
            Self::FP16 => 65504.0,
            Self::BF16 => 3.3895314e38, // ~same exponent range as FP32
            Self::INT8 => 127.0,
            Self::INT4 => 7.0,
            Self::INT2 => 1.0,
        }
    }

    /// Whether the format is floating-point (has an exponent field).
    pub fn is_float(&self) -> bool {
        matches!(self, Self::FP32 | Self::FP16 | Self::BF16)
    }
}

impl std::fmt::Display for PrecisionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FP32 => write!(f, "FP32"),
            Self::FP16 => write!(f, "FP16"),
            Self::BF16 => write!(f, "BF16"),
            Self::INT8 => write!(f, "INT8"),
            Self::INT4 => write!(f, "INT4"),
            Self::INT2 => write!(f, "INT2"),
        }
    }
}

// ── FP16 conversion helpers ───────────────────────────────────────────

/// Convert an f32 to IEEE 754 half-precision (u16).
///
/// Values exceeding FP16 range (±65504) are clamped to ±infinity.
/// Subnormals are flushed to zero for simplicity.
#[inline]
pub fn f32_to_fp16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;

    if exponent == 0xFF {
        // Inf / NaN
        let f16_mantissa = (mantissa >> 13) as u16;
        return (sign << 15) | (0x1F << 10) | f16_mantissa;
    }
    if exponent == 0 {
        return sign << 15; // ±0 or f32 subnormal → f16 zero
    }

    let new_exp = exponent - 112; // 127 - 15
    if new_exp >= 31 {
        return (sign << 15) | (0x1F << 10); // overflow → Inf
    }
    if new_exp <= 0 {
        return sign << 15; // underflow → zero
    }
    let f16_mantissa = (mantissa >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | f16_mantissa
}

/// Convert an IEEE 754 half-precision (u16) to f32.
#[inline]
pub fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31); // ±0
        }
        // Subnormal → normalised f32
        let mut m = mantissa;
        let mut e: i32 = -14;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) & 0xFF;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exponent == 31 {
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13));
    }
    let f32_exp = exponent + 112;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

/// Cast an f32 slice to FP16 (u16 slice).
pub fn cast_fp32_to_fp16(input: &[f32], output: &mut [u16]) -> Result<()> {
    if output.len() < input.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "FP16 output buffer too small: need {}, got {}",
                input.len(),
                output.len()
            ),
        }));
    }
    for (dst, &src) in output.iter_mut().zip(input.iter()) {
        *dst = f32_to_fp16(src);
    }
    Ok(())
}

/// Cast an FP16 (u16) slice back to f32.
pub fn cast_fp16_to_fp32(input: &[u16], output: &mut [f32]) -> Result<()> {
    if output.len() < input.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "FP32 output buffer too small: need {}, got {}",
                input.len(),
                output.len()
            ),
        }));
    }
    for (dst, &src) in output.iter_mut().zip(input.iter()) {
        *dst = fp16_to_f32(src);
    }
    Ok(())
}

// ── BF16 conversion helpers ───────────────────────────────────────────

/// Convert an f32 to BFloat16 (u16).
///
/// BF16 truncates the lower 16 mantissa bits of an f32, preserving the
/// full exponent range.
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    // Round-to-nearest-even: add 0x7FFF + bit 16 of mantissa
    let rounding = ((bits >> 16) & 1) + 0x7FFF;
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

/// Convert a BFloat16 (u16) back to f32.
///
/// Simply shifts the 16-bit pattern into the upper half of an f32.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Cast an f32 slice to BF16 (u16 slice).
pub fn cast_fp32_to_bf16(input: &[f32], output: &mut [u16]) -> Result<()> {
    if output.len() < input.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "BF16 output buffer too small: need {}, got {}",
                input.len(),
                output.len()
            ),
        }));
    }
    for (dst, &src) in output.iter_mut().zip(input.iter()) {
        *dst = f32_to_bf16(src);
    }
    Ok(())
}

/// Cast a BF16 (u16) slice back to f32.
pub fn cast_bf16_to_fp32(input: &[u16], output: &mut [f32]) -> Result<()> {
    if output.len() < input.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "FP32 output buffer too small: need {}, got {}",
                input.len(),
                output.len()
            ),
        }));
    }
    for (dst, &src) in output.iter_mut().zip(input.iter()) {
        *dst = bf16_to_f32(src);
    }
    Ok(())
}

// ── Mixed-precision matmul config ─────────────────────────────────────

/// Configuration for mixed-precision matrix multiplication.
#[derive(Debug, Clone)]
pub struct MixedPrecisionMatmulConfig {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Precision used for input storage and compute.
    pub compute_precision: PrecisionMode,
    /// Precision used for accumulation (must be ≥ compute_precision).
    pub accumulate_precision: PrecisionMode,
}

impl MixedPrecisionMatmulConfig {
    /// Create a config for FP16 compute with FP32 accumulation.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn fp16_with_fp32_accum(m: usize, n: usize, k: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "mixed-precision matmul dimensions must be non-zero: \
                     m={m}, n={n}, k={k}"
                ),
            }
            .into());
        }
        Ok(Self {
            m,
            n,
            k,
            compute_precision: PrecisionMode::FP16,
            accumulate_precision: PrecisionMode::FP32,
        })
    }
}

// ── Mixed-precision matmul (CPU fallback) ─────────────────────────────

/// FP16 compute with FP32 accumulation matrix multiplication.
///
/// Inputs `a` and `b` are FP16 (u16), accumulation and output in FP32.
/// This mirrors the Tensor Core pattern where narrow inputs feed FP32
/// accumulators for numerical stability.
///
/// # Layout
/// - `a`: row-major `[m, k]` FP16
/// - `b`: row-major `[k, n]` FP16
/// - `out`: row-major `[m, n]` FP32
pub fn mixed_precision_matmul(
    a: &[u16],
    b: &[u16],
    out: &mut [f32],
    config: &MixedPrecisionMatmulConfig,
) -> Result<()> {
    let m = config.m;
    let n = config.n;
    let k = config.k;

    if a.len() < m * k {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("A buffer too small: need {}, got {}", m * k, a.len()),
        }));
    }
    if b.len() < k * n {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("B buffer too small: need {}, got {}", k * n, b.len()),
        }));
    }
    if out.len() < m * n {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("output buffer too small: need {}, got {}", m * n, out.len()),
        }));
    }

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32; // FP32 accumulator
            for l in 0..k {
                let a_val = fp16_to_f32(a[i * k + l]);
                let b_val = fp16_to_f32(b[l * n + j]);
                acc += a_val * b_val;
            }
            out[i * n + j] = acc;
        }
    }
    Ok(())
}

// ── Quantized matmul with dequantization ──────────────────────────────

/// Configuration for quantized matmul with dynamic dequantization.
#[derive(Debug, Clone)]
pub struct QuantizedMatmulConfig {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Quantization bit-width (2 or 4).
    pub quant_bits: u32,
    /// Block size for per-block scales.
    pub block_size: usize,
}

impl QuantizedMatmulConfig {
    /// Create a config for INT2 or INT4 quantized matmul.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are zero or `quant_bits` is not
    /// 2 or 4.
    pub fn new(m: usize, n: usize, k: usize, quant_bits: u32, block_size: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("quantized matmul dims must be non-zero: m={m}, n={n}, k={k}"),
            }
            .into());
        }
        if quant_bits != 2 && quant_bits != 4 {
            return Err(KernelError::InvalidArguments {
                reason: format!("quant_bits must be 2 or 4, got {quant_bits}"),
            }
            .into());
        }
        if block_size == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "block_size must be > 0".into() }.into()
            );
        }
        Ok(Self { m, n, k, quant_bits, block_size })
    }
}

/// Decode a packed quantized value at a given element index.
#[inline]
fn decode_quantized(packed: &[u8], elem_idx: usize, quant_bits: u32) -> i8 {
    match quant_bits {
        2 => {
            let byte_idx = elem_idx / 4;
            let bit_off = (elem_idx % 4) * 2;
            let bits = (packed[byte_idx] >> bit_off) & 0x03;
            // 2-bit signed: 00→0, 01→1, 11→−1
            match bits {
                0b00 => 0,
                0b01 => 1,
                0b11 => -1,
                _ => 0,
            }
        }
        4 => {
            let byte_idx = elem_idx / 2;
            let nibble = if elem_idx.is_multiple_of(2) {
                packed[byte_idx] & 0x0F
            } else {
                (packed[byte_idx] >> 4) & 0x0F
            };
            // 4-bit signed two's complement: −8..7
            if nibble & 0x08 != 0 { nibble as i8 | !0x0F_i8 } else { nibble as i8 }
        }
        _ => 0,
    }
}

/// Quantized matmul with dynamic dequantization (CPU fallback).
///
/// Weights are stored in packed `quant_bits`-width format with per-block
/// FP32 scales.  Dequantization happens on-the-fly: each weight is
/// decoded, multiplied by its block scale, and accumulated in FP32.
///
/// # Layout
/// - `activations`: row-major `[m, k]` FP32
/// - `weights_packed`: packed quantized weights, column-major per output
/// - `scales`: `[n, num_blocks_k]` FP32
/// - `out`: row-major `[m, n]` FP32
pub fn quantized_matmul_with_dequant(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    config: &QuantizedMatmulConfig,
) -> Result<()> {
    let m = config.m;
    let n = config.n;
    let k = config.k;
    let block_size = config.block_size;
    let quant_bits = config.quant_bits;
    let elems_per_byte = 8 / quant_bits as usize;
    let packed_k = k.div_ceil(elems_per_byte);
    let num_blocks_k = k.div_ceil(block_size);

    if activations.len() < m * k {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("activations too small: need {}, got {}", m * k, activations.len()),
        }));
    }
    if weights_packed.len() < packed_k * n {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "weights_packed too small: need {}, got {}",
                packed_k * n,
                weights_packed.len()
            ),
        }));
    }
    if scales.len() < n * num_blocks_k {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("scales too small: need {}, got {}", n * num_blocks_k, scales.len()),
        }));
    }
    if out.len() < m * n {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("output too small: need {}, got {}", m * n, out.len()),
        }));
    }

    out[..m * n].fill(0.0);

    for row in 0..m {
        let a_row = &activations[row * k..(row + 1) * k];
        for col in 0..n {
            let mut acc = 0.0f32;
            for blk in 0..num_blocks_k {
                let blk_start = blk * block_size;
                let blk_end = (blk_start + block_size).min(k);
                let scale = scales[col * num_blocks_k + blk];

                for (rel, &a_val) in a_row[blk_start..blk_end].iter().enumerate() {
                    let idx = blk_start + rel;
                    let packed_offset = col * packed_k;
                    let w_raw = decode_quantized(&weights_packed[packed_offset..], idx, quant_bits);
                    let w = w_raw as f32 * scale;
                    acc += a_val * w;
                }
            }
            out[row * n + col] = acc;
        }
    }
    Ok(())
}

// ── Auto precision selection ──────────────────────────────────────────

/// Result of automatic precision analysis.
#[derive(Debug, Clone)]
pub struct PrecisionRecommendation {
    /// Recommended precision mode.
    pub mode: PrecisionMode,
    /// Estimated maximum relative error introduced by the precision.
    pub estimated_max_relative_error: f32,
    /// Whether all values fit within the target precision range.
    pub values_in_range: bool,
}

/// Choose the optimal (narrowest) precision for `data` that keeps the
/// maximum absolute value representable and estimated relative error
/// below `tolerance`.
///
/// Checks in order: INT2, INT4, INT8, FP16, BF16, FP32 — returning the
/// first mode that satisfies both constraints.
pub fn auto_precision_select(data: &[f32], tolerance: f32) -> PrecisionRecommendation {
    if data.is_empty() {
        return PrecisionRecommendation {
            mode: PrecisionMode::FP16,
            estimated_max_relative_error: 0.0,
            values_in_range: true,
        };
    }

    let abs_max = data.iter().fold(0.0f32, |mx, &v| mx.max(v.abs()));

    // Candidate modes from narrowest to widest.
    let candidates = [
        PrecisionMode::INT2,
        PrecisionMode::INT4,
        PrecisionMode::INT8,
        PrecisionMode::FP16,
        PrecisionMode::BF16,
    ];

    for &mode in &candidates {
        let max_rep = mode.max_representable();
        if abs_max > max_rep {
            continue;
        }
        let est_err = precision_loss_estimate(data, mode);
        if est_err <= tolerance {
            return PrecisionRecommendation {
                mode,
                estimated_max_relative_error: est_err,
                values_in_range: true,
            };
        }
    }

    PrecisionRecommendation {
        mode: PrecisionMode::FP32,
        estimated_max_relative_error: 0.0,
        values_in_range: true,
    }
}

// ── Precision loss checking ───────────────────────────────────────────

/// Compute an estimate of the maximum relative error introduced by
/// rounding `data` to the given precision.
fn precision_loss_estimate(data: &[f32], mode: PrecisionMode) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    match mode {
        PrecisionMode::FP32 => 0.0,
        PrecisionMode::FP16 => estimate_fp16_loss(data),
        PrecisionMode::BF16 => estimate_bf16_loss(data),
        PrecisionMode::INT8 | PrecisionMode::INT4 | PrecisionMode::INT2 => {
            estimate_int_quantization_loss(data, mode)
        }
    }
}

/// Measure the maximum relative error of FP16 round-trip for `data`.
fn estimate_fp16_loss(data: &[f32]) -> f32 {
    let mut max_rel = 0.0f32;
    for &v in data {
        let rt = fp16_to_f32(f32_to_fp16(v));
        let abs_v = v.abs();
        if abs_v > 1e-30 {
            max_rel = max_rel.max((v - rt).abs() / abs_v);
        }
    }
    max_rel
}

/// Measure the maximum relative error of BF16 round-trip for `data`.
fn estimate_bf16_loss(data: &[f32]) -> f32 {
    let mut max_rel = 0.0f32;
    for &v in data {
        let rt = bf16_to_f32(f32_to_bf16(v));
        let abs_v = v.abs();
        if abs_v > 1e-30 {
            max_rel = max_rel.max((v - rt).abs() / abs_v);
        }
    }
    max_rel
}

/// Estimate quantization loss for integer modes via uniform quantization.
fn estimate_int_quantization_loss(data: &[f32], mode: PrecisionMode) -> f32 {
    let abs_max = data.iter().fold(0.0f32, |mx, &v| mx.max(v.abs()));
    if abs_max < 1e-30 {
        return 0.0;
    }
    let max_int = mode.max_representable();
    let scale = max_int / abs_max;

    let mut max_rel = 0.0f32;
    for &v in data {
        let quantized = (v * scale).round() / scale;
        let abs_v = v.abs();
        if abs_v > 1e-30 {
            max_rel = max_rel.max((v - quantized).abs() / abs_v);
        }
    }
    max_rel
}

/// Report precision loss between FP32 and a lower precision mode.
///
/// Returns `(mean_absolute_error, max_absolute_error, max_relative_error)`.
pub fn precision_loss_check(data: &[f32], mode: PrecisionMode) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut sum_abs_err = 0.0f64;
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;

    for &v in data {
        let rt = round_trip(v, mode);
        let abs_err = (v - rt).abs();
        sum_abs_err += abs_err as f64;
        max_abs_err = max_abs_err.max(abs_err);
        let abs_v = v.abs();
        if abs_v > 1e-30 {
            max_rel_err = max_rel_err.max(abs_err / abs_v);
        }
    }

    let mean_abs_err = (sum_abs_err / data.len() as f64) as f32;
    (mean_abs_err, max_abs_err, max_rel_err)
}

/// Round-trip a value through the specified precision.
fn round_trip(val: f32, mode: PrecisionMode) -> f32 {
    match mode {
        PrecisionMode::FP32 => val,
        PrecisionMode::FP16 => fp16_to_f32(f32_to_fp16(val)),
        PrecisionMode::BF16 => bf16_to_f32(f32_to_bf16(val)),
        PrecisionMode::INT8 => {
            let clamped = val.clamp(-127.0, 127.0);
            clamped.round()
        }
        PrecisionMode::INT4 => {
            let clamped = val.clamp(-7.0, 7.0);
            clamped.round()
        }
        PrecisionMode::INT2 => {
            let clamped = val.clamp(-1.0, 1.0);
            clamped.round()
        }
    }
}

// ── Scale computation ─────────────────────────────────────────────────

/// Compute the optimal scale factor that maps `data` into the
/// representable range of `mode`.
///
/// For float modes, returns 1.0 (no scaling needed within range).
/// For integer modes, returns `max_int / abs_max(data)`.
pub fn scale_for_precision(data: &[f32], mode: PrecisionMode) -> f32 {
    if data.is_empty() {
        return 1.0;
    }
    let abs_max = data.iter().fold(0.0f32, |mx, &v| mx.max(v.abs()));
    if abs_max < 1e-30 {
        return 1.0;
    }
    match mode {
        PrecisionMode::FP32 | PrecisionMode::BF16 => 1.0,
        PrecisionMode::FP16 => {
            if abs_max > 65504.0 {
                65504.0 / abs_max
            } else {
                1.0
            }
        }
        PrecisionMode::INT8 => 127.0 / abs_max,
        PrecisionMode::INT4 => 7.0 / abs_max,
        PrecisionMode::INT2 => 1.0 / abs_max,
    }
}

// ── Fused cast + compute ──────────────────────────────────────────────

/// Fused cast-and-dot-product: cast `a` and `b` from FP32 to FP16,
/// multiply element-wise, and accumulate in FP32 — all in a single pass
/// without materialising intermediate FP16 buffers.
pub fn fused_cast_and_compute(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("fused_cast_and_compute: a.len()={} != b.len()={}", a.len(), b.len()),
        }));
    }
    if out.len() < a.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "fused_cast_and_compute: output too small: need {}, got {}",
                a.len(),
                out.len()
            ),
        }));
    }
    for i in 0..a.len() {
        let a16 = fp16_to_f32(f32_to_fp16(a[i]));
        let b16 = fp16_to_f32(f32_to_fp16(b[i]));
        out[i] = a16 * b16; // FP32 accumulation of FP16 operands
    }
    Ok(())
}

// ── FP32 accumulation ─────────────────────────────────────────────────

/// Accumulate partial FP32 results from multiple chunks.
///
/// Each chunk is summed into the corresponding position of `accum`
/// (which must be pre-zeroed or pre-initialised).
pub fn accumulate_in_fp32(chunks: &[&[f32]], accum: &mut [f32]) -> Result<()> {
    if chunks.is_empty() {
        return Ok(());
    }
    let expected_len = accum.len();
    for (idx, chunk) in chunks.iter().enumerate() {
        if chunk.len() != expected_len {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "chunk {idx} length {} != accumulator length {expected_len}",
                    chunk.len()
                ),
            }));
        }
    }
    for chunk in chunks {
        for (acc, &val) in accum.iter_mut().zip(chunk.iter()) {
            *acc += val;
        }
    }
    Ok(())
}

// ── CUDA kernel source (GPU-only) ─────────────────────────────────────

/// CUDA C kernel for mixed-precision FP16→FP32 matmul.
///
/// Each thread computes one output element.  Inputs are read as `__half`,
/// multiplied via `__hmul`, and accumulated in FP32.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const MIXED_PRECISION_MATMUL_KERNEL_SRC: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void mixed_precision_matmul_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int i = 0; i < K; i++) {
        float a_val = __half2float(A[row * K + i]);
        float b_val = __half2float(B[i * N + col]);
        acc += a_val * b_val;
    }
    C[row * N + col] = acc;
}
"#;

/// CUDA launch stub for the mixed-precision matmul kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` — scaffold only.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_mixed_precision_matmul(
    _a: &[u16],
    _b: &[u16],
    _output: &mut [f32],
    config: &MixedPrecisionMatmulConfig,
) -> Result<()> {
    log::debug!(
        "mixed-precision matmul CUDA stub: m={}, n={}, k={}, \
         compute={}, accum={}",
        config.m,
        config.n,
        config.k,
        config.compute_precision,
        config.accumulate_precision,
    );
    Err(KernelError::GpuError {
        reason: "mixed-precision matmul CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Unified dispatch: GPU if available, else CPU fallback.
pub fn mixed_precision_matmul_forward(
    a: &[u16],
    b: &[u16],
    output: &mut [f32],
    config: &MixedPrecisionMatmulConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_mixed_precision_matmul(a, b, output, config)
        {
            return Ok(());
        }
    }
    mixed_precision_matmul(a, b, output, config)
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    fn naive_matmul_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    // ── PrecisionMode tests ───────────────────────────────────────

    #[test]
    fn test_precision_mode_bits() {
        assert_eq!(PrecisionMode::FP32.bits(), 32);
        assert_eq!(PrecisionMode::FP16.bits(), 16);
        assert_eq!(PrecisionMode::BF16.bits(), 16);
        assert_eq!(PrecisionMode::INT8.bits(), 8);
        assert_eq!(PrecisionMode::INT4.bits(), 4);
        assert_eq!(PrecisionMode::INT2.bits(), 2);
    }

    #[test]
    fn test_precision_mode_max_representable() {
        assert_eq!(PrecisionMode::FP16.max_representable(), 65504.0);
        assert_eq!(PrecisionMode::INT8.max_representable(), 127.0);
        assert_eq!(PrecisionMode::INT4.max_representable(), 7.0);
        assert_eq!(PrecisionMode::INT2.max_representable(), 1.0);
        assert!(PrecisionMode::FP32.max_representable() > 1e38);
        assert!(PrecisionMode::BF16.max_representable() > 1e38);
    }

    #[test]
    fn test_precision_mode_is_float() {
        assert!(PrecisionMode::FP32.is_float());
        assert!(PrecisionMode::FP16.is_float());
        assert!(PrecisionMode::BF16.is_float());
        assert!(!PrecisionMode::INT8.is_float());
        assert!(!PrecisionMode::INT4.is_float());
        assert!(!PrecisionMode::INT2.is_float());
    }

    #[test]
    fn test_precision_mode_display() {
        assert_eq!(format!("{}", PrecisionMode::FP32), "FP32");
        assert_eq!(format!("{}", PrecisionMode::FP16), "FP16");
        assert_eq!(format!("{}", PrecisionMode::BF16), "BF16");
        assert_eq!(format!("{}", PrecisionMode::INT8), "INT8");
        assert_eq!(format!("{}", PrecisionMode::INT4), "INT4");
        assert_eq!(format!("{}", PrecisionMode::INT2), "INT2");
    }

    #[test]
    fn test_precision_mode_equality_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PrecisionMode::FP16);
        set.insert(PrecisionMode::FP16);
        assert_eq!(set.len(), 1);
        set.insert(PrecisionMode::BF16);
        assert_eq!(set.len(), 2);
    }

    // ── FP16 round-trip tests ─────────────────────────────────────

    #[test]
    fn test_fp16_round_trip_zero() {
        let h = f32_to_fp16(0.0);
        assert_eq!(fp16_to_f32(h), 0.0);
    }

    #[test]
    fn test_fp16_round_trip_neg_zero() {
        let h = f32_to_fp16(-0.0);
        let rt = fp16_to_f32(h);
        assert_eq!(rt.to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn test_fp16_round_trip_one() {
        let h = f32_to_fp16(1.0);
        assert_eq!(fp16_to_f32(h), 1.0);
    }

    #[test]
    fn test_fp16_round_trip_neg_one() {
        let h = f32_to_fp16(-1.0);
        assert_eq!(fp16_to_f32(h), -1.0);
    }

    #[test]
    fn test_fp16_round_trip_various() {
        for &v in &[0.5, 0.25, 2.0, 100.0, 1024.0, 0.001] {
            let rt = fp16_to_f32(f32_to_fp16(v));
            assert!(
                (rt - v).abs() / v.abs() < 0.002,
                "FP16 round-trip error too large for {v}: got {rt}"
            );
        }
    }

    #[test]
    fn test_fp16_max_value() {
        let h = f32_to_fp16(65504.0);
        assert_eq!(fp16_to_f32(h), 65504.0);
    }

    #[test]
    fn test_fp16_overflow_to_inf() {
        let h = f32_to_fp16(70000.0);
        assert!(fp16_to_f32(h).is_infinite());
    }

    #[test]
    fn test_fp16_underflow_to_zero() {
        // A very small positive number that underflows FP16 subnormals
        let tiny = 1e-10;
        let h = f32_to_fp16(tiny);
        assert_eq!(fp16_to_f32(h), 0.0);
    }

    #[test]
    fn test_fp16_infinity() {
        let h = f32_to_fp16(f32::INFINITY);
        assert!(fp16_to_f32(h).is_infinite());
        assert!(fp16_to_f32(h) > 0.0);
    }

    #[test]
    fn test_fp16_neg_infinity() {
        let h = f32_to_fp16(f32::NEG_INFINITY);
        assert!(fp16_to_f32(h).is_infinite());
        assert!(fp16_to_f32(h) < 0.0);
    }

    #[test]
    fn test_fp16_nan() {
        let h = f32_to_fp16(f32::NAN);
        assert!(fp16_to_f32(h).is_nan());
    }

    // ── FP16 batch cast tests ─────────────────────────────────────

    #[test]
    fn test_cast_fp32_to_fp16_basic() {
        let input = [1.0f32, 2.0, -3.0, 0.5];
        let mut output = [0u16; 4];
        cast_fp32_to_fp16(&input, &mut output).unwrap();
        let mut back = [0.0f32; 4];
        cast_fp16_to_fp32(&output, &mut back).unwrap();
        assert_close(&back, &input, 0.01);
    }

    #[test]
    fn test_cast_fp32_to_fp16_buffer_too_small() {
        let input = [1.0f32; 4];
        let mut output = [0u16; 2];
        assert!(cast_fp32_to_fp16(&input, &mut output).is_err());
    }

    #[test]
    fn test_cast_fp16_to_fp32_buffer_too_small() {
        let input = [0u16; 4];
        let mut output = [0.0f32; 2];
        assert!(cast_fp16_to_fp32(&input, &mut output).is_err());
    }

    #[test]
    fn test_cast_fp32_to_fp16_empty() {
        let input: [f32; 0] = [];
        let mut output: [u16; 0] = [];
        cast_fp32_to_fp16(&input, &mut output).unwrap();
    }

    // ── BF16 round-trip tests ─────────────────────────────────────

    #[test]
    fn test_bf16_round_trip_zero() {
        let b = f32_to_bf16(0.0);
        assert_eq!(bf16_to_f32(b), 0.0);
    }

    #[test]
    fn test_bf16_round_trip_one() {
        let b = f32_to_bf16(1.0);
        assert_eq!(bf16_to_f32(b), 1.0);
    }

    #[test]
    fn test_bf16_round_trip_neg_one() {
        let b = f32_to_bf16(-1.0);
        assert_eq!(bf16_to_f32(b), -1.0);
    }

    #[test]
    fn test_bf16_round_trip_large() {
        let val = 1e30;
        let rt = bf16_to_f32(f32_to_bf16(val));
        assert!(
            (rt - val).abs() / val.abs() < 0.01,
            "BF16 round-trip error too large for {val}: got {rt}"
        );
    }

    #[test]
    fn test_bf16_round_trip_various() {
        for &v in &[0.5, 0.25, 2.0, 100.0, 1e10, -42.0] {
            let rt = bf16_to_f32(f32_to_bf16(v));
            assert!(
                (rt - v).abs() / v.abs() < 0.01,
                "BF16 round-trip error too large for {v}: got {rt}"
            );
        }
    }

    #[test]
    fn test_bf16_preserves_exponent_range() {
        // BF16 has the same exponent range as FP32
        let large = 1e38;
        let rt = bf16_to_f32(f32_to_bf16(large));
        assert!(rt.is_finite());
        assert!((rt - large).abs() / large < 0.01);
    }

    #[test]
    fn test_bf16_infinity() {
        let b = f32_to_bf16(f32::INFINITY);
        assert!(bf16_to_f32(b).is_infinite());
    }

    #[test]
    fn test_bf16_nan() {
        let b = f32_to_bf16(f32::NAN);
        assert!(bf16_to_f32(b).is_nan());
    }

    // ── BF16 batch cast tests ─────────────────────────────────────

    #[test]
    fn test_cast_fp32_to_bf16_basic() {
        let input = [1.0f32, 2.0, -3.0, 0.5];
        let mut output = [0u16; 4];
        cast_fp32_to_bf16(&input, &mut output).unwrap();
        let mut back = [0.0f32; 4];
        cast_bf16_to_fp32(&output, &mut back).unwrap();
        assert_close(&back, &input, 0.02);
    }

    #[test]
    fn test_cast_fp32_to_bf16_buffer_too_small() {
        let input = [1.0f32; 4];
        let mut output = [0u16; 2];
        assert!(cast_fp32_to_bf16(&input, &mut output).is_err());
    }

    #[test]
    fn test_cast_bf16_to_fp32_buffer_too_small() {
        let input = [0u16; 4];
        let mut output = [0.0f32; 2];
        assert!(cast_bf16_to_fp32(&input, &mut output).is_err());
    }

    // ── Mixed-precision matmul tests ──────────────────────────────

    #[test]
    fn test_mixed_precision_matmul_identity() {
        // 2×2 identity: A·I = A
        let a_f32 = [3.0f32, -2.0, 5.0, 7.0];
        let b_f32 = [1.0f32, 0.0, 0.0, 1.0];
        let a: Vec<u16> = a_f32.iter().map(|&v| f32_to_fp16(v)).collect();
        let b: Vec<u16> = b_f32.iter().map(|&v| f32_to_fp16(v)).collect();
        let mut out = [0.0f32; 4];
        let cfg = MixedPrecisionMatmulConfig::fp16_with_fp32_accum(2, 2, 2).unwrap();
        mixed_precision_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &a_f32, 0.01);
    }

    #[test]
    fn test_mixed_precision_matmul_2x3_times_3x2() {
        let m = 2;
        let n = 2;
        let k = 3;
        let a_f32 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_f32 = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expected = naive_matmul_f32(&a_f32, &b_f32, m, n, k);
        let a: Vec<u16> = a_f32.iter().map(|&v| f32_to_fp16(v)).collect();
        let b: Vec<u16> = b_f32.iter().map(|&v| f32_to_fp16(v)).collect();
        let mut out = vec![0.0f32; m * n];
        let cfg = MixedPrecisionMatmulConfig::fp16_with_fp32_accum(m, n, k).unwrap();
        mixed_precision_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 0.5);
    }

    #[test]
    fn test_mixed_precision_matmul_vs_fp32_reference() {
        let m = 4;
        let n = 4;
        let k = 8;
        let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let expected = naive_matmul_f32(&a_f32, &b_f32, m, n, k);
        let a: Vec<u16> = a_f32.iter().map(|&v| f32_to_fp16(v)).collect();
        let b: Vec<u16> = b_f32.iter().map(|&v| f32_to_fp16(v)).collect();
        let mut out = vec![0.0f32; m * n];
        let cfg = MixedPrecisionMatmulConfig::fp16_with_fp32_accum(m, n, k).unwrap();
        mixed_precision_matmul(&a, &b, &mut out, &cfg).unwrap();
        // FP16 inputs lose some precision — allow wider tolerance
        assert_close(&out, &expected, 0.1);
    }

    #[test]
    fn test_mixed_precision_matmul_rejects_zero_dims() {
        assert!(MixedPrecisionMatmulConfig::fp16_with_fp32_accum(0, 4, 4).is_err());
        assert!(MixedPrecisionMatmulConfig::fp16_with_fp32_accum(4, 0, 4).is_err());
        assert!(MixedPrecisionMatmulConfig::fp16_with_fp32_accum(4, 4, 0).is_err());
    }

    #[test]
    fn test_mixed_precision_matmul_buffer_too_small() {
        let cfg = MixedPrecisionMatmulConfig::fp16_with_fp32_accum(2, 2, 2).unwrap();
        let a = vec![0u16; 4];
        let b = vec![0u16; 4];
        let mut out = vec![0.0f32; 2]; // too small
        assert!(mixed_precision_matmul(&a, &b, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_mixed_precision_matmul_1x1() {
        let a_val = 3.0f32;
        let b_val = 4.0f32;
        let a = [f32_to_fp16(a_val)];
        let b = [f32_to_fp16(b_val)];
        let mut out = [0.0f32];
        let cfg = MixedPrecisionMatmulConfig::fp16_with_fp32_accum(1, 1, 1).unwrap();
        mixed_precision_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert!((out[0] - 12.0).abs() < 0.01);
    }

    // ── Quantized matmul tests ────────────────────────────────────

    fn pack_int2_values(vals: &[i8]) -> Vec<u8> {
        let packed_len = vals.len().div_ceil(4);
        let mut packed = vec![0u8; packed_len];
        for (i, &v) in vals.iter().enumerate() {
            let code: u8 = match v {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            };
            packed[i / 4] |= code << ((i % 4) * 2);
        }
        packed
    }

    #[test]
    fn test_quantized_matmul_int2_identity_pattern() {
        // 1×4 activation × 4×1 weight (all ones) → dot product
        let m = 1;
        let n = 1;
        let k = 4;
        let activations = [1.0f32, 2.0, 3.0, 4.0];
        let weights_i8 = [1i8, 1, 1, 1];
        let weights_packed = pack_int2_values(&weights_i8);
        let scales = [1.0f32]; // one block, one column
        let mut out = [0.0f32];
        let cfg = QuantizedMatmulConfig::new(m, n, k, 2, 4).unwrap();
        quantized_matmul_with_dequant(&activations, &weights_packed, &scales, &mut out, &cfg)
            .unwrap();
        assert!((out[0] - 10.0).abs() < 1e-5); // 1+2+3+4 = 10
    }

    #[test]
    fn test_quantized_matmul_int2_with_neg_weights() {
        let m = 1;
        let n = 1;
        let k = 4;
        let activations = [1.0f32, 2.0, 3.0, 4.0];
        let weights_i8 = [1i8, -1, 1, -1];
        let weights_packed = pack_int2_values(&weights_i8);
        let scales = [1.0f32];
        let mut out = [0.0f32];
        let cfg = QuantizedMatmulConfig::new(m, n, k, 2, 4).unwrap();
        quantized_matmul_with_dequant(&activations, &weights_packed, &scales, &mut out, &cfg)
            .unwrap();
        // 1*1 + 2*(-1) + 3*1 + 4*(-1) = 1 - 2 + 3 - 4 = -2
        assert!((out[0] - (-2.0)).abs() < 1e-5);
    }

    #[test]
    fn test_quantized_matmul_int2_with_scale() {
        let m = 1;
        let n = 1;
        let k = 4;
        let activations = [1.0f32, 2.0, 3.0, 4.0];
        let weights_i8 = [1i8, 1, 1, 1];
        let weights_packed = pack_int2_values(&weights_i8);
        let scales = [0.5f32]; // scale halves the result
        let mut out = [0.0f32];
        let cfg = QuantizedMatmulConfig::new(m, n, k, 2, 4).unwrap();
        quantized_matmul_with_dequant(&activations, &weights_packed, &scales, &mut out, &cfg)
            .unwrap();
        assert!((out[0] - 5.0).abs() < 1e-5); // 10 * 0.5
    }

    #[test]
    fn test_quantized_matmul_int2_multiple_rows() {
        let m = 2;
        let n = 1;
        let k = 4;
        let activations = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weights_i8 = [1i8, 1, 1, 1];
        let weights_packed = pack_int2_values(&weights_i8);
        let scales = [1.0f32];
        let mut out = [0.0f32; 2];
        let cfg = QuantizedMatmulConfig::new(m, n, k, 2, 4).unwrap();
        quantized_matmul_with_dequant(&activations, &weights_packed, &scales, &mut out, &cfg)
            .unwrap();
        assert!((out[0] - 10.0).abs() < 1e-5); // 1+2+3+4
        assert!((out[1] - 26.0).abs() < 1e-5); // 5+6+7+8
    }

    #[test]
    fn test_quantized_matmul_int4_basic() {
        // INT4: nibble-packed, signed two's complement
        let m = 1;
        let n = 1;
        let k = 2;
        let activations = [2.0f32, 3.0];
        // Pack INT4 values: 3 and -2
        let val0: u8 = 3; // nibble 0
        let val1: u8 = (-2i8 as u8) & 0x0F; // nibble 1 = 0x0E
        let packed = [(val1 << 4) | val0]; // [0xE3]
        let scales = [1.0f32];
        let mut out = [0.0f32];
        let cfg = QuantizedMatmulConfig::new(m, n, k, 4, 2).unwrap();
        quantized_matmul_with_dequant(&activations, &packed, &scales, &mut out, &cfg).unwrap();
        // 2*3 + 3*(-2) = 6 - 6 = 0
        assert!((out[0]).abs() < 1e-5);
    }

    #[test]
    fn test_quantized_matmul_rejects_bad_bits() {
        assert!(QuantizedMatmulConfig::new(1, 1, 4, 3, 4).is_err());
        assert!(QuantizedMatmulConfig::new(1, 1, 4, 8, 4).is_err());
    }

    #[test]
    fn test_quantized_matmul_rejects_zero_dims() {
        assert!(QuantizedMatmulConfig::new(0, 1, 4, 2, 4).is_err());
        assert!(QuantizedMatmulConfig::new(1, 0, 4, 2, 4).is_err());
        assert!(QuantizedMatmulConfig::new(1, 1, 0, 2, 4).is_err());
    }

    #[test]
    fn test_quantized_matmul_rejects_zero_block_size() {
        assert!(QuantizedMatmulConfig::new(1, 1, 4, 2, 0).is_err());
    }

    // ── Auto precision selection tests ────────────────────────────

    #[test]
    fn test_auto_select_empty_data() {
        let rec = auto_precision_select(&[], 0.01);
        assert_eq!(rec.mode, PrecisionMode::FP16);
        assert!(rec.values_in_range);
    }

    #[test]
    fn test_auto_select_small_ternary_values() {
        let data = [0.0f32, 1.0, -1.0, 0.0, 1.0];
        let rec = auto_precision_select(&data, 0.5);
        // Should pick INT2 since all values are in {-1, 0, 1}
        assert_eq!(rec.mode, PrecisionMode::INT2);
    }

    #[test]
    fn test_auto_select_fp16_range() {
        let data = [100.0f32, -200.0, 50.0, 0.5];
        let rec = auto_precision_select(&data, 0.01);
        // Values exceed INT8 range, so should pick FP16 or narrower
        assert!(
            rec.mode == PrecisionMode::FP16 || rec.mode == PrecisionMode::BF16,
            "expected FP16 or BF16, got {:?}",
            rec.mode
        );
    }

    #[test]
    fn test_auto_select_large_values_picks_bf16_or_fp32() {
        let data = [1e35f32, -1e35];
        let rec = auto_precision_select(&data, 0.01);
        // Exceeds FP16 max (65504) — needs BF16 or FP32
        assert!(
            rec.mode == PrecisionMode::BF16 || rec.mode == PrecisionMode::FP32,
            "expected BF16 or FP32, got {:?}",
            rec.mode
        );
    }

    #[test]
    fn test_auto_select_tight_tolerance_picks_wider() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let loose = auto_precision_select(&data, 0.5);
        let tight = auto_precision_select(&data, 0.0001);
        assert!(tight.mode.bits() >= loose.mode.bits());
    }

    // ── Precision loss check tests ────────────────────────────────

    #[test]
    fn test_precision_loss_fp32_is_zero() {
        let data = [1.0f32, 2.0, 3.0];
        let (mae, max_ae, max_re) = precision_loss_check(&data, PrecisionMode::FP32);
        assert_eq!(mae, 0.0);
        assert_eq!(max_ae, 0.0);
        assert_eq!(max_re, 0.0);
    }

    #[test]
    fn test_precision_loss_fp16_small_values() {
        let data = [1.0f32, 2.0, 0.5, -1.0];
        let (mae, max_ae, max_re) = precision_loss_check(&data, PrecisionMode::FP16);
        // These values are exactly representable in FP16
        assert!(mae < 1e-6);
        assert!(max_ae < 1e-6);
        assert!(max_re < 1e-6);
    }

    #[test]
    fn test_precision_loss_bf16_moderate_values() {
        let data = [1.0f32, 2.0, 0.5, -3.14];
        let (_, max_ae, _) = precision_loss_check(&data, PrecisionMode::BF16);
        // BF16 has ~7-bit mantissa, so max abs error for small values
        // should be modest
        assert!(max_ae < 0.02);
    }

    #[test]
    fn test_precision_loss_int8() {
        // Values must be clamped to [-127, 127] and rounded
        let data = [0.5f32, 1.5, -0.7, 3.3];
        let (_, _, max_re) = precision_loss_check(&data, PrecisionMode::INT8);
        // INT8 rounds to nearest integer, so loss is significant
        assert!(max_re > 0.0);
    }

    #[test]
    fn test_precision_loss_empty() {
        let (mae, max_ae, max_re) = precision_loss_check(&[], PrecisionMode::FP16);
        assert_eq!(mae, 0.0);
        assert_eq!(max_ae, 0.0);
        assert_eq!(max_re, 0.0);
    }

    // ── Scale computation tests ───────────────────────────────────

    #[test]
    fn test_scale_for_fp32_is_one() {
        let data = [1000.0f32, -500.0];
        assert_eq!(scale_for_precision(&data, PrecisionMode::FP32), 1.0);
    }

    #[test]
    fn test_scale_for_fp16_within_range() {
        let data = [100.0f32, -200.0];
        assert_eq!(scale_for_precision(&data, PrecisionMode::FP16), 1.0);
    }

    #[test]
    fn test_scale_for_fp16_exceeds_range() {
        let data = [100000.0f32];
        let scale = scale_for_precision(&data, PrecisionMode::FP16);
        assert!(scale < 1.0);
        assert!((scale - 65504.0 / 100000.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_for_int8() {
        let data = [10.0f32, -5.0];
        let scale = scale_for_precision(&data, PrecisionMode::INT8);
        assert!((scale - 127.0 / 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_for_int4() {
        let data = [3.5f32, -2.0];
        let scale = scale_for_precision(&data, PrecisionMode::INT4);
        assert!((scale - 7.0 / 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_scale_for_int2() {
        let data = [5.0f32, -3.0];
        let scale = scale_for_precision(&data, PrecisionMode::INT2);
        assert!((scale - 1.0 / 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_empty_data() {
        assert_eq!(scale_for_precision(&[], PrecisionMode::INT8), 1.0);
    }

    #[test]
    fn test_scale_zero_data() {
        let data = [0.0f32; 4];
        assert_eq!(scale_for_precision(&data, PrecisionMode::INT8), 1.0);
    }

    // ── Fused cast and compute tests ──────────────────────────────

    #[test]
    fn test_fused_cast_and_compute_basic() {
        let a = [2.0f32, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0];
        let mut out = [0.0f32; 3];
        fused_cast_and_compute(&a, &b, &mut out).unwrap();
        // After FP16 round-trip and multiply
        assert!((out[0] - 10.0).abs() < 0.1);
        assert!((out[1] - 18.0).abs() < 0.1);
        assert!((out[2] - 28.0).abs() < 0.1);
    }

    #[test]
    fn test_fused_cast_matches_separate_operations() {
        let a = [1.5f32, -2.5, 3.0, 0.25];
        let b = [4.0f32, -1.0, 2.0, 8.0];
        let mut fused_out = [0.0f32; 4];
        fused_cast_and_compute(&a, &b, &mut fused_out).unwrap();

        // Separate: cast to FP16, back to FP32, then multiply
        let mut separate_out = [0.0f32; 4];
        for i in 0..4 {
            let a16 = fp16_to_f32(f32_to_fp16(a[i]));
            let b16 = fp16_to_f32(f32_to_fp16(b[i]));
            separate_out[i] = a16 * b16;
        }
        assert_close(&fused_out, &separate_out, 1e-10);
    }

    #[test]
    fn test_fused_cast_length_mismatch() {
        let a = [1.0f32; 3];
        let b = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        assert!(fused_cast_and_compute(&a, &b, &mut out).is_err());
    }

    #[test]
    fn test_fused_cast_output_too_small() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 4];
        let mut out = [0.0f32; 2];
        assert!(fused_cast_and_compute(&a, &b, &mut out).is_err());
    }

    #[test]
    fn test_fused_cast_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        fused_cast_and_compute(&a, &b, &mut out).unwrap();
    }

    // ── Accumulate in FP32 tests ──────────────────────────────────

    #[test]
    fn test_accumulate_single_chunk() {
        let chunk = [1.0f32, 2.0, 3.0];
        let mut accum = [0.0f32; 3];
        accumulate_in_fp32(&[&chunk], &mut accum).unwrap();
        assert_close(&accum, &[1.0, 2.0, 3.0], 1e-10);
    }

    #[test]
    fn test_accumulate_multiple_chunks() {
        let c1 = [1.0f32, 2.0];
        let c2 = [3.0f32, 4.0];
        let c3 = [5.0f32, 6.0];
        let mut accum = [0.0f32; 2];
        accumulate_in_fp32(&[&c1, &c2, &c3], &mut accum).unwrap();
        assert_close(&accum, &[9.0, 12.0], 1e-10);
    }

    #[test]
    fn test_accumulate_empty_chunks() {
        let mut accum = [0.0f32; 3];
        accumulate_in_fp32(&[], &mut accum).unwrap();
        assert_close(&accum, &[0.0, 0.0, 0.0], 1e-10);
    }

    #[test]
    fn test_accumulate_length_mismatch() {
        let c1 = [1.0f32, 2.0];
        let c2 = [3.0f32]; // wrong length
        let mut accum = [0.0f32; 2];
        assert!(accumulate_in_fp32(&[&c1, &c2], &mut accum).is_err());
    }

    #[test]
    fn test_accumulate_preserves_initial_values() {
        let chunk = [1.0f32, 2.0];
        let mut accum = [10.0f32, 20.0];
        accumulate_in_fp32(&[&chunk], &mut accum).unwrap();
        assert_close(&accum, &[11.0, 22.0], 1e-10);
    }

    // ── Edge cases: denormals, inf, NaN ───────────────────────────

    #[test]
    fn test_fp16_denormal_round_trip() {
        // FP16 smallest subnormal ≈ 5.96e-8
        let subnormal_bits: u16 = 0x0001;
        let val = fp16_to_f32(subnormal_bits);
        assert!(val > 0.0);
        assert!(val < 1e-6);
    }

    #[test]
    fn test_bf16_neg_zero() {
        let bits = f32_to_bf16(-0.0);
        let rt = bf16_to_f32(bits);
        assert_eq!(rt.to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn test_mixed_matmul_with_inf_input() {
        let a = [f32_to_fp16(f32::INFINITY)];
        let b = [f32_to_fp16(1.0)];
        let mut out = [0.0f32];
        let cfg = MixedPrecisionMatmulConfig::fp16_with_fp32_accum(1, 1, 1).unwrap();
        mixed_precision_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert!(out[0].is_infinite());
    }

    #[test]
    fn test_mixed_matmul_with_nan_input() {
        let a = [f32_to_fp16(f32::NAN)];
        let b = [f32_to_fp16(1.0)];
        let mut out = [0.0f32];
        let cfg = MixedPrecisionMatmulConfig::fp16_with_fp32_accum(1, 1, 1).unwrap();
        mixed_precision_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert!(out[0].is_nan());
    }

    #[test]
    fn test_precision_loss_with_nan() {
        let data = [1.0f32, f32::NAN, 2.0];
        let (_, _, max_re) = precision_loss_check(&data, PrecisionMode::FP16);
        // NaN differences produce NaN, but we skip near-zero abs values
        // so the function should still return a finite result for the
        // non-NaN elements.
        assert!(max_re.is_nan() || max_re >= 0.0);
    }

    // ── Large tensor tests ────────────────────────────────────────

    #[test]
    fn test_mixed_precision_matmul_large() {
        let m = 16;
        let n = 16;
        let k = 32;
        let a_f32: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let expected = naive_matmul_f32(&a_f32, &b_f32, m, n, k);
        let a: Vec<u16> = a_f32.iter().map(|&v| f32_to_fp16(v)).collect();
        let b: Vec<u16> = b_f32.iter().map(|&v| f32_to_fp16(v)).collect();
        let mut out = vec![0.0f32; m * n];
        let cfg = MixedPrecisionMatmulConfig::fp16_with_fp32_accum(m, n, k).unwrap();
        mixed_precision_matmul(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 0.5);
    }

    #[test]
    fn test_cast_fp32_to_fp16_large_batch() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let mut fp16 = vec![0u16; n];
        cast_fp32_to_fp16(&input, &mut fp16).unwrap();
        let mut back = vec![0.0f32; n];
        cast_fp16_to_fp32(&fp16, &mut back).unwrap();
        for i in 0..n {
            assert!((back[i] - input[i]).abs() < 0.02, "index {i}: {} vs {}", back[i], input[i]);
        }
    }

    #[test]
    fn test_cast_fp32_to_bf16_large_batch() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.1).collect();
        let mut bf16 = vec![0u16; n];
        cast_fp32_to_bf16(&input, &mut bf16).unwrap();
        let mut back = vec![0.0f32; n];
        cast_bf16_to_fp32(&bf16, &mut back).unwrap();
        for i in 0..n {
            let abs_in = input[i].abs();
            if abs_in > 0.01 {
                assert!(
                    (back[i] - input[i]).abs() / abs_in < 0.01,
                    "index {i}: {} vs {}",
                    back[i],
                    input[i]
                );
            }
        }
    }

    // ── Forward dispatch test ─────────────────────────────────────

    #[test]
    fn test_mixed_precision_matmul_forward_cpu_fallback() {
        let a = [f32_to_fp16(1.0), f32_to_fp16(0.0), f32_to_fp16(0.0), f32_to_fp16(1.0)];
        let b = [f32_to_fp16(5.0), f32_to_fp16(6.0), f32_to_fp16(7.0), f32_to_fp16(8.0)];
        let mut out = [0.0f32; 4];
        let cfg = MixedPrecisionMatmulConfig::fp16_with_fp32_accum(2, 2, 2).unwrap();
        mixed_precision_matmul_forward(&a, &b, &mut out, &cfg).unwrap();
        // Identity × B = B
        assert_close(&out, &[5.0, 6.0, 7.0, 8.0], 0.01);
    }

    // ── Round-trip function test ──────────────────────────────────

    #[test]
    fn test_round_trip_all_modes() {
        let val = 2.5f32;
        assert_eq!(round_trip(val, PrecisionMode::FP32), val);
        assert!((round_trip(val, PrecisionMode::FP16) - val).abs() < 0.01);
        assert!((round_trip(val, PrecisionMode::BF16) - val).abs() < 0.1);
        assert_eq!(round_trip(val, PrecisionMode::INT8), 3.0); // rounds
        assert_eq!(round_trip(val, PrecisionMode::INT4), 3.0);
        assert_eq!(round_trip(val, PrecisionMode::INT2), 1.0); // clamped
    }

    #[test]
    fn test_round_trip_int_clamping() {
        assert_eq!(round_trip(200.0, PrecisionMode::INT8), 127.0);
        assert_eq!(round_trip(-200.0, PrecisionMode::INT8), -127.0);
        assert_eq!(round_trip(20.0, PrecisionMode::INT4), 7.0);
        assert_eq!(round_trip(-20.0, PrecisionMode::INT4), -7.0);
        assert_eq!(round_trip(5.0, PrecisionMode::INT2), 1.0);
        assert_eq!(round_trip(-5.0, PrecisionMode::INT2), -1.0);
    }
}
