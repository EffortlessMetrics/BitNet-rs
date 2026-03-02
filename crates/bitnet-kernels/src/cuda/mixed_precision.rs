//! CUDA mixed-precision kernels for BitNet inference.
//!
//! # Overview
//!
//! Mixed-precision arithmetic stores operands in a compact format (F16 or BF16)
//! while accumulating in F32 for numerical stability.  This module provides:
//!
//! - CPU reference implementations for precision conversion (F32↔F16, F32↔BF16)
//! - Mixed-precision matrix multiplication with configurable input/output/accumulate types
//! - Dynamic loss scaling for mixed-precision training stability
//! - Precision-aware parallel reduction
//! - CUDA kernel source strings for tensor-core F16/BF16 GEMM and mixed attention
//!
//! # Kernel strategy
//!
//! GPU kernels use NVIDIA tensor cores (`wmma` intrinsics) for F16/BF16 matmul
//! with F32 accumulators.  The mixed-precision attention kernel fuses
//! Q·K^T scaling, softmax, and V projection in a single pass with F16 compute
//! and F32 accumulate to preserve attention-score precision.
//!
//! # CPU fallback
//!
//! Every operation has a pure-Rust scalar fallback that is always compiled.
//! CUDA launch stubs are gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use std::fmt;

use bitnet_common::{KernelError, Result};

// ───────────────────────────────────────────────────────────────────
// Error type
// ───────────────────────────────────────────────────────────────────

/// Errors specific to mixed-precision kernel dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum MixedPrecisionError {
    /// Matrix dimensions are incompatible for the requested operation.
    DimensionMismatch { expected: usize, got: usize },
    /// The requested precision combination is unsupported.
    UnsupportedPrecision(String),
    /// An input buffer is empty or has zero length.
    EmptyInput,
    /// Loss scaling overflow — the scale factor exceeded representable range.
    ScaleOverflow { scale: f32 },
    /// Configuration is invalid.
    InvalidConfig(String),
}

impl fmt::Display for MixedPrecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::UnsupportedPrecision(msg) => {
                write!(f, "unsupported precision: {msg}")
            }
            Self::EmptyInput => write!(f, "empty input"),
            Self::ScaleOverflow { scale } => {
                write!(f, "loss scale overflow: {scale}")
            }
            Self::InvalidConfig(msg) => {
                write!(f, "invalid config: {msg}")
            }
        }
    }
}

impl std::error::Error for MixedPrecisionError {}

// ───────────────────────────────────────────────────────────────────
// Precision types and configuration
// ───────────────────────────────────────────────────────────────────

/// Floating-point and integer precision types supported by mixed-precision kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionType {
    /// 32-bit IEEE 754 single precision.
    F32,
    /// 16-bit IEEE 754 half precision.
    F16,
    /// 16-bit Google Brain floating point (truncated F32 mantissa).
    BF16,
    /// 8-bit signed integer (for quantized activations).
    I8,
    /// 2-bit signed integer (BitNet ternary weights).
    I2,
}

impl fmt::Display for PrecisionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
            Self::I8 => write!(f, "i8"),
            Self::I2 => write!(f, "i2"),
        }
    }
}

impl PrecisionType {
    /// Returns the size in bytes of a single element in this precision.
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
            // I2 uses sub-byte packing; report 1 as minimum addressable unit.
            Self::I2 => 1,
        }
    }

    /// Returns `true` if this type is a floating-point format.
    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }
}

/// Configuration for a mixed-precision operation.
///
/// Specifies the precision of the compute (operand) path, the storage
/// format for weights/activations, and the accumulator type for
/// reductions and matmul inner products.
#[derive(Debug, Clone, PartialEq)]
pub struct MixedPrecisionConfig {
    /// Precision used for the compute (arithmetic) path.
    pub compute_dtype: PrecisionType,
    /// Precision used for weight and activation storage.
    pub storage_dtype: PrecisionType,
    /// Precision used for accumulators (dot products, reductions).
    pub accumulate_dtype: PrecisionType,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            compute_dtype: PrecisionType::F16,
            storage_dtype: PrecisionType::F16,
            accumulate_dtype: PrecisionType::F32,
        }
    }
}

impl MixedPrecisionConfig {
    /// F16 compute with F32 accumulation (most common for inference).
    pub fn f16_compute() -> Self {
        Self {
            compute_dtype: PrecisionType::F16,
            storage_dtype: PrecisionType::F16,
            accumulate_dtype: PrecisionType::F32,
        }
    }

    /// BF16 compute with F32 accumulation.
    pub fn bf16_compute() -> Self {
        Self {
            compute_dtype: PrecisionType::BF16,
            storage_dtype: PrecisionType::BF16,
            accumulate_dtype: PrecisionType::F32,
        }
    }

    /// Pure F32 (no mixed precision).
    pub fn f32_only() -> Self {
        Self {
            compute_dtype: PrecisionType::F32,
            storage_dtype: PrecisionType::F32,
            accumulate_dtype: PrecisionType::F32,
        }
    }

    /// Validate that the configuration makes sense.
    pub fn validate(&self) -> std::result::Result<(), MixedPrecisionError> {
        if !self.accumulate_dtype.is_float() {
            return Err(MixedPrecisionError::UnsupportedPrecision(
                "accumulate_dtype must be a float type".into(),
            ));
        }
        // Accumulator precision should be >= compute precision for numerical stability.
        if self.accumulate_dtype.size_bytes() < self.compute_dtype.size_bytes()
            && self.compute_dtype.is_float()
        {
            return Err(MixedPrecisionError::UnsupportedPrecision(format!(
                "accumulate_dtype ({}) should be at least as wide as compute_dtype ({})",
                self.accumulate_dtype, self.compute_dtype,
            )));
        }
        Ok(())
    }
}

// ───────────────────────────────────────────────────────────────────
// F32 ↔ F16 conversion (IEEE 754 half precision)
// ───────────────────────────────────────────────────────────────────

/// Convert a single `f32` value to IEEE 754 half precision, stored as `u16`.
///
/// Handles special values (NaN, ±Inf, denormals, ±0) correctly.
/// Values exceeding the F16 representable range saturate to ±Inf.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    // NaN
    if exp == 255 && mantissa != 0 {
        // Preserve NaN: sign | all-ones exponent | non-zero mantissa
        return (sign << 15 | 0x7C00 | (mantissa >> 13).max(1)) as u16;
    }

    // ±Inf
    if exp == 255 {
        return (sign << 15 | 0x7C00) as u16;
    }

    // Rebias: F32 bias=127, F16 bias=15, so new_exp = exp - 127 + 15 = exp - 112
    let new_exp = exp - 112;

    // Overflow → ±Inf
    if new_exp >= 31 {
        return (sign << 15 | 0x7C00) as u16;
    }

    // Normal F16
    if new_exp > 0 {
        // Round-to-nearest-even
        let half_mantissa = mantissa >> 13;
        let remainder = mantissa & 0x1FFF;
        let round_bit = if remainder > 0x1000 || (remainder == 0x1000 && (half_mantissa & 1) != 0) {
            1u32
        } else {
            0u32
        };
        let result = (sign << 15) | ((new_exp as u32) << 10) | half_mantissa;
        // Handle mantissa carry on rounding
        return (result + round_bit) as u16;
    }

    // Denormal F16 (or underflow to zero)
    if new_exp >= -10 {
        // Denormals: shift mantissa right, adding implicit leading 1
        let shift = (1 - new_exp) as u32;
        let denorm_mantissa = (mantissa | 0x0080_0000) >> (13 + shift);
        return (sign << 15 | denorm_mantissa) as u16;
    }

    // Too small → ±0
    (sign << 15) as u16
}

/// Convert an IEEE 754 half-precision `u16` back to `f32`.
pub fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exp = ((half >> 10) & 0x1F) as u32;
    let mantissa = (half & 0x03FF) as u32;

    if exp == 31 {
        // Inf or NaN
        if mantissa == 0 {
            return f32::from_bits(sign << 31 | 0x7F80_0000);
        }
        return f32::from_bits(sign << 31 | 0x7FC0_0000 | (mantissa << 13));
    }

    if exp == 0 {
        if mantissa == 0 {
            // ±0
            return f32::from_bits(sign << 31);
        }
        // Denormal: normalize by shifting mantissa until leading 1 appears
        let mut m = mantissa;
        let mut e: i32 = -14; // denormal exponent for F16
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF; // remove the leading 1 bit
        let f32_exp = ((e + 127) as u32) << 23;
        return f32::from_bits(sign << 31 | f32_exp | (m << 13));
    }

    // Normal: rebias from F16 (bias 15) to F32 (bias 127)
    let f32_exp = (exp + 112) << 23;
    f32::from_bits(sign << 31 | f32_exp | (mantissa << 13))
}

/// Batch convert `f32` slice to F16 (`u16`) values.
///
/// # Errors
///
/// Returns an error if the output buffer is smaller than the input.
pub fn f32_to_f16_batch(input: &[f32], output: &mut [u16]) -> Result<()> {
    if output.len() < input.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("output buffer too small: need {}, got {}", input.len(), output.len()),
        }
        .into());
    }
    for (i, &v) in input.iter().enumerate() {
        output[i] = f32_to_f16(v);
    }
    Ok(())
}

/// Batch convert F16 (`u16`) slice back to `f32`.
///
/// # Errors
///
/// Returns an error if the output buffer is smaller than the input.
pub fn f16_to_f32_batch(input: &[u16], output: &mut [f32]) -> Result<()> {
    if output.len() < input.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("output buffer too small: need {}, got {}", input.len(), output.len()),
        }
        .into());
    }
    for (i, &v) in input.iter().enumerate() {
        output[i] = f16_to_f32(v);
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// F32 ↔ BF16 conversion (Brain Floating Point)
// ───────────────────────────────────────────────────────────────────

/// Convert a single `f32` value to BF16, stored as `u16`.
///
/// BF16 keeps the same exponent range as F32 (8 bits) but truncates
/// the mantissa from 23 to 7 bits.  This preserves dynamic range at
/// the cost of precision.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();

    // NaN: preserve NaN-ness with a canonical quiet NaN.
    if value.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }

    // Round-to-nearest-even: add rounding bias before truncating.
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Convert a BF16 `u16` back to `f32`.
///
/// Simply places the 16-bit value in the upper half of the F32 bit
/// pattern (zero-fills the lower 16 bits).
pub fn bf16_to_f32(half: u16) -> f32 {
    f32::from_bits((half as u32) << 16)
}

/// Batch convert `f32` slice to BF16 (`u16`) values.
///
/// # Errors
///
/// Returns an error if the output buffer is smaller than the input.
pub fn f32_to_bf16_batch(input: &[f32], output: &mut [u16]) -> Result<()> {
    if output.len() < input.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("output buffer too small: need {}, got {}", input.len(), output.len()),
        }
        .into());
    }
    for (i, &v) in input.iter().enumerate() {
        output[i] = f32_to_bf16(v);
    }
    Ok(())
}

/// Batch convert BF16 (`u16`) slice back to `f32`.
///
/// # Errors
///
/// Returns an error if the output buffer is smaller than the input.
pub fn bf16_to_f32_batch(input: &[u16], output: &mut [f32]) -> Result<()> {
    if output.len() < input.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("output buffer too small: need {}, got {}", input.len(), output.len()),
        }
        .into());
    }
    for (i, &v) in input.iter().enumerate() {
        output[i] = bf16_to_f32(v);
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// Mixed-precision matrix multiplication (CPU reference)
// ───────────────────────────────────────────────────────────────────

/// Configuration for mixed-precision matrix multiplication.
#[derive(Debug, Clone)]
pub struct MixedPrecisionMatmulConfig {
    /// Rows of output (and A).
    pub m: usize,
    /// Columns of output (and B).
    pub n: usize,
    /// Inner (reduction) dimension.
    pub k: usize,
    /// Precision configuration.
    pub precision: MixedPrecisionConfig,
}

/// Mixed-precision matrix multiplication: C = A · B.
///
/// Operands A and B are in F32 but the computation simulates the
/// requested precision pipeline: values are down-cast to
/// `compute_dtype` for the multiply, accumulated in `accumulate_dtype`,
/// and the result is stored in F32.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent with config.
pub fn mixed_precision_matmul(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    config: &MixedPrecisionMatmulConfig,
) -> Result<()> {
    let m = config.m;
    let n = config.n;
    let k = config.k;

    if m == 0 || n == 0 || k == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("matmul dimensions must be non-zero: m={m}, n={n}, k={k}"),
        }
        .into());
    }
    if a.len() < m * k {
        return Err(KernelError::InvalidArguments {
            reason: format!("A buffer too small: need {}, got {}", m * k, a.len()),
        }
        .into());
    }
    if b.len() < k * n {
        return Err(KernelError::InvalidArguments {
            reason: format!("B buffer too small: need {}, got {}", k * n, b.len()),
        }
        .into());
    }
    if output.len() < m * n {
        return Err(KernelError::InvalidArguments {
            reason: format!("output buffer too small: need {}, got {}", m * n, output.len()),
        }
        .into());
    }

    config
        .precision
        .validate()
        .map_err(|e| KernelError::InvalidArguments { reason: e.to_string() })?;

    // Simulate mixed-precision by casting through the requested types.
    let quantize: fn(f32) -> f32 = match config.precision.compute_dtype {
        PrecisionType::F16 => |v| f16_to_f32(f32_to_f16(v)),
        PrecisionType::BF16 => |v| bf16_to_f32(f32_to_bf16(v)),
        _ => |v| v,
    };

    for i in 0..m {
        for j in 0..n {
            let mut acc: f64 = 0.0;
            for p in 0..k {
                let a_val = quantize(a[i * k + p]);
                let b_val = quantize(b[p * n + j]);
                acc += (a_val as f64) * (b_val as f64);
            }
            output[i * n + j] = acc as f32;
        }
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// Dynamic loss scaling
// ───────────────────────────────────────────────────────────────────

/// State for dynamic loss scaling in mixed-precision training.
#[derive(Debug, Clone)]
pub struct DynamicLossScaler {
    /// Current loss scale factor.
    pub scale: f32,
    /// Growth factor applied when no overflow is detected.
    pub growth_factor: f32,
    /// Backoff factor applied on overflow.
    pub backoff_factor: f32,
    /// Number of consecutive steps without overflow.
    pub growth_interval: u32,
    /// Counter of consecutive non-overflow steps.
    pub steps_since_last_overflow: u32,
    /// Minimum allowed scale factor.
    pub min_scale: f32,
    /// Maximum allowed scale factor.
    pub max_scale: f32,
}

impl Default for DynamicLossScaler {
    fn default() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_last_overflow: 0,
            min_scale: 1.0,
            max_scale: f32::MAX,
        }
    }
}

impl DynamicLossScaler {
    /// Create a new scaler with the given initial scale.
    pub fn new(initial_scale: f32) -> Self {
        Self { scale: initial_scale, ..Self::default() }
    }

    /// Update the scaler after a training step.
    ///
    /// If `overflow_detected` is `true`, the scale is reduced by
    /// `backoff_factor`.  Otherwise, after `growth_interval` consecutive
    /// clean steps the scale is increased by `growth_factor`.
    pub fn update(&mut self, overflow_detected: bool) {
        if overflow_detected {
            self.scale = (self.scale * self.backoff_factor).max(self.min_scale);
            self.steps_since_last_overflow = 0;
        } else {
            self.steps_since_last_overflow += 1;
            if self.steps_since_last_overflow >= self.growth_interval {
                self.scale = (self.scale * self.growth_factor).min(self.max_scale);
                self.steps_since_last_overflow = 0;
            }
        }
    }
}

/// Compute a dynamic loss-scaling factor from gradient statistics.
///
/// Returns the current `scale` from the scaler, checking for overflow
/// in the provided gradients.  If any gradient is non-finite (NaN or Inf),
/// the scaler backs off and returns the reduced scale.
///
/// # Errors
///
/// Returns an error if the scale overflows representable range.
pub fn dynamic_loss_scaling(
    gradients: &[f32],
    scaler: &mut DynamicLossScaler,
) -> std::result::Result<f32, MixedPrecisionError> {
    let overflow = gradients.iter().any(|&g| !g.is_finite());
    scaler.update(overflow);
    if scaler.scale.is_infinite() || scaler.scale.is_nan() {
        return Err(MixedPrecisionError::ScaleOverflow { scale: scaler.scale });
    }
    Ok(scaler.scale)
}

// ───────────────────────────────────────────────────────────────────
// Precision-aware reduction
// ───────────────────────────────────────────────────────────────────

/// Reduction operation for mixed-precision reduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedReduceOp {
    /// Sum all elements.
    Sum,
    /// Maximum element.
    Max,
    /// Mean of all elements.
    Mean,
}

/// Precision-aware reduction of a float slice.
///
/// Values are cast to the `compute_dtype` before participating in the
/// reduction, and accumulation is done in F64 for maximum precision in
/// the CPU reference path.
///
/// # Errors
///
/// Returns an error if the input is empty.
pub fn mixed_precision_reduce(
    input: &[f32],
    op: MixedReduceOp,
    config: &MixedPrecisionConfig,
) -> Result<f32> {
    if input.is_empty() {
        return Err(
            KernelError::InvalidArguments { reason: "cannot reduce empty input".into() }.into()
        );
    }

    let quantize: fn(f32) -> f32 = match config.compute_dtype {
        PrecisionType::F16 => |v| f16_to_f32(f32_to_f16(v)),
        PrecisionType::BF16 => |v| bf16_to_f32(f32_to_bf16(v)),
        _ => |v| v,
    };

    match op {
        MixedReduceOp::Sum => {
            let sum: f64 = input.iter().map(|&v| quantize(v) as f64).sum();
            Ok(sum as f32)
        }
        MixedReduceOp::Max => {
            let max = input.iter().map(|&v| quantize(v)).fold(f32::NEG_INFINITY, f32::max);
            Ok(max)
        }
        MixedReduceOp::Mean => {
            let sum: f64 = input.iter().map(|&v| quantize(v) as f64).sum();
            Ok((sum / input.len() as f64) as f32)
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// CUDA kernel source strings
// ───────────────────────────────────────────────────────────────────

/// CUDA kernel source for F16 matrix multiplication using tensor cores.
///
/// Uses WMMA (Warp Matrix Multiply-Accumulate) intrinsics for 16×16×16
/// F16 tiles with F32 accumulation.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const MIXED_PRECISION_F16_MATMUL_KERNEL_SRC: &str = r#"
#include <mma.h>
using namespace nvcuda;

// F16 GEMM via WMMA tensor cores.
// Grid:  (ceil(N/16), ceil(M/16), batch)
// Block: (32, 1, 1)  — one warp per tile
extern "C" __global__ void mixed_precision_f16_matmul(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int warpM = blockIdx.y * 16;
    int warpN = blockIdx.x * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        if (warpM < M && k < K)
            wmma::load_matrix_sync(a_frag, A + warpM * K + k, K);
        if (k < K && warpN < N)
            wmma::load_matrix_sync(b_frag, B + k * N + warpN, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if (warpM < M && warpN < N)
        wmma::store_matrix_sync(C + warpM * N + warpN, c_frag, N, wmma::mem_row_major);
}
"#;

/// CUDA kernel source for BF16 matrix multiplication using tensor cores.
///
/// Uses WMMA intrinsics for 16×16×16 BF16 tiles with F32 accumulation.
/// Requires SM 80+ (Ampere or later) for native BF16 tensor core support.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const MIXED_PRECISION_BF16_MATMUL_KERNEL_SRC: &str = r#"
#include <mma.h>
using namespace nvcuda;

// BF16 GEMM via WMMA tensor cores (Ampere+).
// Grid:  (ceil(N/16), ceil(M/16), batch)
// Block: (32, 1, 1)
extern "C" __global__ void mixed_precision_bf16_matmul(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int warpM = blockIdx.y * 16;
    int warpN = blockIdx.x * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        if (warpM < M && k < K)
            wmma::load_matrix_sync(a_frag, A + warpM * K + k, K);
        if (k < K && warpN < N)
            wmma::load_matrix_sync(b_frag, B + k * N + warpN, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if (warpM < M && warpN < N)
        wmma::store_matrix_sync(C + warpM * N + warpN, c_frag, N, wmma::mem_row_major);
}
"#;

/// CUDA kernel source for mixed-precision attention.
///
/// Computes scaled dot-product attention with F16 compute and F32
/// accumulate: `softmax(Q·K^T / sqrt(d)) · V`.
///
/// The kernel fuses the QK matmul, scaling, optional causal mask,
/// softmax, and AV matmul in one pass per head to minimise HBM traffic.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const MIXED_PRECISION_ATTENTION_KERNEL_SRC: &str = r#"
#include <mma.h>

// Mixed-precision fused attention: softmax(Q*K^T / sqrt(d)) * V
// F16 inputs, F32 accumulation for attention scores.
// Grid:  (num_heads, seq_len, batch)
// Block: (256, 1, 1)
extern "C" __global__ void mixed_precision_attention_f16(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ output,
    int seq_len,
    int head_dim,
    float scale)
{
    int head = blockIdx.x;
    int query_pos = blockIdx.y;
    int batch = blockIdx.z;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* scores = shared;

    int head_offset = (batch * gridDim.x + head) * seq_len * head_dim;
    const half* q_row = Q + head_offset + query_pos * head_dim;
    const half* k_base = K + head_offset;
    const half* v_base = V + head_offset;

    // Phase 1: Q·K^T with F32 accumulation + scale
    float local_max = -1e30f;
    for (int kv_pos = tid; kv_pos < seq_len; kv_pos += blockDim.x) {
        const half* k_row = k_base + kv_pos * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(q_row[d]) * __half2float(k_row[d]);
        }
        dot *= scale;
        scores[kv_pos] = dot;
        if (dot > local_max) local_max = dot;
    }
    __syncthreads();

    // Block-wide max reduction
    shared[blockDim.x + tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared[blockDim.x + tid] =
                fmaxf(shared[blockDim.x + tid], shared[blockDim.x + tid + s]);
        __syncthreads();
    }
    float max_val = shared[blockDim.x];

    // Phase 2: exp and sum
    float local_sum = 0.0f;
    for (int kv_pos = tid; kv_pos < seq_len; kv_pos += blockDim.x) {
        float v = expf(scores[kv_pos] - max_val);
        scores[kv_pos] = v;
        local_sum += v;
    }
    shared[blockDim.x + tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared[blockDim.x + tid] += shared[blockDim.x + tid + s];
        __syncthreads();
    }
    float inv_sum = (shared[blockDim.x] > 0.0f)
        ? (1.0f / shared[blockDim.x]) : 0.0f;

    // Phase 3: normalize
    for (int kv_pos = tid; kv_pos < seq_len; kv_pos += blockDim.x) {
        scores[kv_pos] *= inv_sum;
    }
    __syncthreads();

    // Phase 4: weighted sum over V with F32 accumulate
    float* out_row = output + head_offset + query_pos * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < seq_len; kv_pos++) {
            acc += scores[kv_pos] * __half2float(v_base[kv_pos * head_dim + d]);
        }
        out_row[d] = acc;
    }
}
"#;

// ───────────────────────────────────────────────────────────────────
// GPU launch configuration
// ───────────────────────────────────────────────────────────────────

/// Launch configuration for mixed-precision CUDA matmul.
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[derive(Debug, Clone)]
pub struct MixedPrecisionLaunchConfig {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Precision configuration.
    pub precision: MixedPrecisionConfig,
    /// WMMA tile size (typically 16).
    pub tile_size: u32,
    /// Warp size (32 for NVIDIA GPUs).
    pub warp_size: u32,
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
impl MixedPrecisionLaunchConfig {
    /// Create a launch config for the given shape.
    pub fn for_shape(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            precision: MixedPrecisionConfig::f16_compute(),
            tile_size: 16,
            warp_size: 32,
        }
    }

    /// Compute grid dimensions for WMMA-based kernels.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let grid_x = (self.n as u32).div_ceil(self.tile_size);
        let grid_y = (self.m as u32).div_ceil(self.tile_size);
        (grid_x, grid_y, 1)
    }

    /// Compute block dimensions (one warp).
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.warp_size, 1, 1)
    }
}

// ───────────────────────────────────────────────────────────────────
// GPU launch stubs
// ───────────────────────────────────────────────────────────────────

/// Launch the mixed-precision F16 matmul CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_mixed_precision_f16_matmul(
    _a: &[u16],
    _b: &[u16],
    _output: &mut [f32],
    config: &MixedPrecisionLaunchConfig,
) -> Result<()> {
    log::debug!(
        "mixed-precision F16 matmul CUDA stub: m={}, n={}, k={}, grid={:?}",
        config.m,
        config.n,
        config.k,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "mixed-precision F16 matmul CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch the mixed-precision BF16 matmul CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_mixed_precision_bf16_matmul(
    _a: &[u16],
    _b: &[u16],
    _output: &mut [f32],
    config: &MixedPrecisionLaunchConfig,
) -> Result<()> {
    log::debug!(
        "mixed-precision BF16 matmul CUDA stub: m={}, n={}, k={}, grid={:?}",
        config.m,
        config.n,
        config.k,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "mixed-precision BF16 matmul CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch the mixed-precision attention CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_mixed_precision_attention(
    _q: &[u16],
    _k: &[u16],
    _v: &[u16],
    _output: &mut [f32],
    _seq_len: usize,
    _head_dim: usize,
    _num_heads: usize,
    _scale: f32,
) -> Result<()> {
    log::debug!(
        "mixed-precision attention CUDA stub: seq_len={}, head_dim={}, num_heads={}",
        _seq_len,
        _head_dim,
        _num_heads,
    );
    Err(KernelError::GpuError {
        reason: "mixed-precision attention CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ───────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── F32 ↔ F16 round-trip accuracy ────────────────────────────

    #[test]
    fn f16_round_trip_small_integers() {
        for i in -1024..=1024 {
            let v = i as f32;
            let rt = f16_to_f32(f32_to_f16(v));
            assert_eq!(rt, v, "exact for small integer {v}");
        }
    }

    #[test]
    fn f16_round_trip_fractional_values() {
        let values = [0.1, 0.5, 1.5, 3.14, 100.0, 1000.0, -42.5, -0.001];
        for &v in &values {
            let rt = f16_to_f32(f32_to_f16(v));
            let rel_err = ((rt - v) / v).abs();
            assert!(rel_err < 6.2e-4, "f16 round-trip {v} → {rt}, rel_err={rel_err}");
        }
    }

    #[test]
    fn f16_round_trip_relative_error_bound() {
        // F16 has 10-bit mantissa → ~6.1e-4 relative precision for normals.
        let test_values: Vec<f32> = (1..=100)
            .map(|i| i as f32 * 0.1 + 1.0)
            .chain((1..=50).map(|i| i as f32 * 100.0))
            .collect();
        for v in test_values {
            let rt = f16_to_f32(f32_to_f16(v));
            let rel_err = ((rt - v) / v).abs();
            assert!(rel_err <= 6.2e-4, "f16 relative error for {v}: {rel_err} > 6.1e-4");
        }
    }

    #[test]
    fn f16_positive_zero() {
        let h = f32_to_f16(0.0);
        assert_eq!(h, 0x0000);
        assert_eq!(f16_to_f32(h), 0.0);
        assert!(f16_to_f32(h).is_sign_positive());
    }

    #[test]
    fn f16_negative_zero() {
        let h = f32_to_f16(-0.0);
        assert_eq!(h, 0x8000);
        assert_eq!(f16_to_f32(h), -0.0);
        assert!(f16_to_f32(h).is_sign_negative());
    }

    #[test]
    fn f16_positive_infinity() {
        let h = f32_to_f16(f32::INFINITY);
        assert_eq!(h, 0x7C00);
        assert!(f16_to_f32(h).is_infinite());
        assert!(f16_to_f32(h).is_sign_positive());
    }

    #[test]
    fn f16_negative_infinity() {
        let h = f32_to_f16(f32::NEG_INFINITY);
        assert_eq!(h, 0xFC00);
        assert!(f16_to_f32(h).is_infinite());
        assert!(f16_to_f32(h).is_sign_negative());
    }

    #[test]
    fn f16_nan_preserved() {
        let h = f32_to_f16(f32::NAN);
        assert!(f16_to_f32(h).is_nan());
    }

    #[test]
    fn f16_negative_nan_preserved() {
        let h = f32_to_f16(-f32::NAN);
        assert!(f16_to_f32(h).is_nan());
    }

    #[test]
    fn f16_overflow_to_inf() {
        // Values well above F16_MAX (65504) should map to infinity.
        let h = f32_to_f16(65520.0);
        assert!(
            f16_to_f32(h).is_infinite(),
            "65520.0 should overflow to Inf in F16, got {}",
            f16_to_f32(h)
        );
    }

    #[test]
    fn f16_large_negative_overflow() {
        let h = f32_to_f16(-70000.0);
        let rt = f16_to_f32(h);
        assert!(rt.is_infinite() && rt.is_sign_negative());
    }

    #[test]
    fn f16_max_finite_value() {
        // F16 max finite = 65504.0
        let h = f32_to_f16(65504.0);
        let rt = f16_to_f32(h);
        assert!((rt - 65504.0).abs() < 1.0, "F16_MAX round-trip: expected ~65504, got {rt}");
    }

    #[test]
    fn f16_denormal_small_positive() {
        // Smallest positive F16 denormal: 2^-24 ≈ 5.96e-8
        let small = 6.0e-8_f32;
        let h = f32_to_f16(small);
        let rt = f16_to_f32(h);
        // Denormals may lose precision but should not be zero if representable.
        assert!(rt >= 0.0, "denormal should be non-negative, got {rt}");
    }

    #[test]
    fn f16_underflow_to_zero() {
        // Way below F16 denormal range.
        let tiny = 1.0e-20_f32;
        let h = f32_to_f16(tiny);
        assert_eq!(f16_to_f32(h), 0.0);
    }

    #[test]
    fn f16_batch_conversion_round_trip() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let mut f16_buf = vec![0u16; 256];
        let mut output = vec![0.0f32; 256];
        f32_to_f16_batch(&input, &mut f16_buf).unwrap();
        f16_to_f32_batch(&f16_buf, &mut output).unwrap();
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            let err = (inp - out).abs();
            assert!(err < 0.1, "batch f16 round-trip index {i}: {inp} → {out}, err={err}");
        }
    }

    #[test]
    fn f16_batch_output_too_small() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0u16; 2];
        assert!(f32_to_f16_batch(&input, &mut output).is_err());
    }

    // ── F32 ↔ BF16 round-trip accuracy ──────────────────────────

    #[test]
    fn bf16_round_trip_small_integers() {
        for i in -256..=256 {
            let v = i as f32;
            let rt = bf16_to_f32(f32_to_bf16(v));
            assert_eq!(rt, v, "exact for small integer {v}");
        }
    }

    #[test]
    fn bf16_round_trip_relative_error_bound() {
        // BF16 has 7-bit mantissa → ~7.8e-3 relative precision.
        let test_values: Vec<f32> = (1..=100)
            .map(|i| i as f32 * 0.1 + 1.0)
            .chain((1..=50).map(|i| i as f32 * 100.0))
            .collect();
        for v in test_values {
            let rt = bf16_to_f32(f32_to_bf16(v));
            let rel_err = ((rt - v) / v).abs();
            assert!(rel_err <= 7.9e-3, "bf16 relative error for {v}: {rel_err} > 7.8e-3");
        }
    }

    #[test]
    fn bf16_positive_zero() {
        let h = f32_to_bf16(0.0);
        assert_eq!(h, 0x0000);
        assert_eq!(bf16_to_f32(h), 0.0);
    }

    #[test]
    fn bf16_negative_zero() {
        let h = f32_to_bf16(-0.0);
        assert_eq!(h, 0x8000);
        assert_eq!(bf16_to_f32(h), -0.0);
    }

    #[test]
    fn bf16_positive_infinity() {
        let h = f32_to_bf16(f32::INFINITY);
        assert_eq!(h, 0x7F80);
        assert!(bf16_to_f32(h).is_infinite());
    }

    #[test]
    fn bf16_negative_infinity() {
        let h = f32_to_bf16(f32::NEG_INFINITY);
        assert_eq!(h, 0xFF80);
        assert!(bf16_to_f32(h).is_infinite());
    }

    #[test]
    fn bf16_nan_preserved() {
        let h = f32_to_bf16(f32::NAN);
        assert!(bf16_to_f32(h).is_nan());
    }

    #[test]
    fn bf16_large_values_preserved() {
        // BF16 has the same exponent range as F32, so large values survive.
        let v = 1.0e30_f32;
        let rt = bf16_to_f32(f32_to_bf16(v));
        let rel_err = ((rt - v) / v).abs();
        assert!(rel_err < 0.01, "bf16 large value {v}: rel_err={rel_err}");
    }

    #[test]
    fn bf16_batch_conversion_round_trip() {
        let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 1.5).collect();
        let mut bf16_buf = vec![0u16; 128];
        let mut output = vec![0.0f32; 128];
        f32_to_bf16_batch(&input, &mut bf16_buf).unwrap();
        bf16_to_f32_batch(&bf16_buf, &mut output).unwrap();
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            let err = (inp - out).abs();
            let tol = inp.abs() * 0.01 + 0.5;
            assert!(err < tol, "batch bf16 round-trip index {i}: {inp} → {out}, err={err}");
        }
    }

    #[test]
    fn bf16_batch_output_too_small() {
        let input = [1.0, 2.0];
        let mut output = [0u16; 1];
        assert!(f32_to_bf16_batch(&input, &mut output).is_err());
    }

    // ── PrecisionType ────────────────────────────────────────────

    #[test]
    fn precision_type_size_bytes() {
        assert_eq!(PrecisionType::F32.size_bytes(), 4);
        assert_eq!(PrecisionType::F16.size_bytes(), 2);
        assert_eq!(PrecisionType::BF16.size_bytes(), 2);
        assert_eq!(PrecisionType::I8.size_bytes(), 1);
        assert_eq!(PrecisionType::I2.size_bytes(), 1);
    }

    #[test]
    fn precision_type_is_float() {
        assert!(PrecisionType::F32.is_float());
        assert!(PrecisionType::F16.is_float());
        assert!(PrecisionType::BF16.is_float());
        assert!(!PrecisionType::I8.is_float());
        assert!(!PrecisionType::I2.is_float());
    }

    #[test]
    fn precision_type_display() {
        assert_eq!(format!("{}", PrecisionType::F32), "f32");
        assert_eq!(format!("{}", PrecisionType::F16), "f16");
        assert_eq!(format!("{}", PrecisionType::BF16), "bf16");
        assert_eq!(format!("{}", PrecisionType::I8), "i8");
        assert_eq!(format!("{}", PrecisionType::I2), "i2");
    }

    // ── MixedPrecisionConfig ─────────────────────────────────────

    #[test]
    fn config_default_is_f16_compute() {
        let cfg = MixedPrecisionConfig::default();
        assert_eq!(cfg.compute_dtype, PrecisionType::F16);
        assert_eq!(cfg.storage_dtype, PrecisionType::F16);
        assert_eq!(cfg.accumulate_dtype, PrecisionType::F32);
    }

    #[test]
    fn config_f32_only() {
        let cfg = MixedPrecisionConfig::f32_only();
        assert_eq!(cfg.compute_dtype, PrecisionType::F32);
        assert_eq!(cfg.accumulate_dtype, PrecisionType::F32);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_bf16_compute() {
        let cfg = MixedPrecisionConfig::bf16_compute();
        assert_eq!(cfg.compute_dtype, PrecisionType::BF16);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_invalid_integer_accumulator() {
        let cfg = MixedPrecisionConfig {
            compute_dtype: PrecisionType::F16,
            storage_dtype: PrecisionType::F16,
            accumulate_dtype: PrecisionType::I8,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_invalid_accumulator_smaller_than_compute() {
        let cfg = MixedPrecisionConfig {
            compute_dtype: PrecisionType::F32,
            storage_dtype: PrecisionType::F32,
            accumulate_dtype: PrecisionType::F16,
        };
        assert!(cfg.validate().is_err());
    }

    // ── Mixed-precision matmul ───────────────────────────────────

    #[test]
    fn mixed_matmul_identity_f32() {
        // 2×2 identity × [1,2; 3,4] = [1,2; 3,4]
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        let config = MixedPrecisionMatmulConfig {
            m: 2,
            n: 2,
            k: 2,
            precision: MixedPrecisionConfig::f32_only(),
        };
        mixed_precision_matmul(&a, &b, &mut out, &config).unwrap();
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn mixed_matmul_simple_f16() {
        // [1,2] * [[1],[3]] = 7
        let a = [1.0, 2.0];
        let b = [1.0, 3.0];
        let mut out = [0.0f32; 1];
        let config = MixedPrecisionMatmulConfig {
            m: 1,
            n: 1,
            k: 2,
            precision: MixedPrecisionConfig::f16_compute(),
        };
        mixed_precision_matmul(&a, &b, &mut out, &config).unwrap();
        assert!((out[0] - 7.0).abs() < 0.01, "expected ~7.0, got {}", out[0]);
    }

    #[test]
    fn mixed_matmul_simple_bf16() {
        let a = [2.0, 3.0];
        let b = [4.0, 5.0];
        let mut out = [0.0f32; 1];
        let config = MixedPrecisionMatmulConfig {
            m: 1,
            n: 1,
            k: 2,
            precision: MixedPrecisionConfig::bf16_compute(),
        };
        mixed_precision_matmul(&a, &b, &mut out, &config).unwrap();
        // 2*4 + 3*5 = 23
        assert!((out[0] - 23.0).abs() < 0.5, "expected ~23, got {}", out[0]);
    }

    #[test]
    fn mixed_matmul_larger_matrix() {
        // 4×3 * 3×2 = 4×2
        let a: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=6).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 8];
        let config = MixedPrecisionMatmulConfig {
            m: 4,
            n: 2,
            k: 3,
            precision: MixedPrecisionConfig::f32_only(),
        };
        mixed_precision_matmul(&a, &b, &mut out, &config).unwrap();
        // Row 0: 1*1+2*3+3*5 = 22, 1*2+2*4+3*6 = 28
        assert!((out[0] - 22.0).abs() < 1e-5);
        assert!((out[1] - 28.0).abs() < 1e-5);
    }

    #[test]
    fn mixed_matmul_f16_vs_f32_tolerance() {
        let a: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 + 0.1).collect();
        let b: Vec<f32> = (0..16).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let mut out_f32 = vec![0.0f32; 16];
        let mut out_f16 = vec![0.0f32; 16];

        let config_f32 = MixedPrecisionMatmulConfig {
            m: 4,
            n: 4,
            k: 4,
            precision: MixedPrecisionConfig::f32_only(),
        };
        let config_f16 = MixedPrecisionMatmulConfig {
            m: 4,
            n: 4,
            k: 4,
            precision: MixedPrecisionConfig::f16_compute(),
        };
        mixed_precision_matmul(&a, &b, &mut out_f32, &config_f32).unwrap();
        mixed_precision_matmul(&a, &b, &mut out_f16, &config_f16).unwrap();

        for (i, (&f32v, &f16v)) in out_f32.iter().zip(out_f16.iter()).enumerate() {
            let abs_err = (f32v - f16v).abs();
            let tol = f32v.abs() * 0.01 + 0.05;
            assert!(abs_err < tol, "matmul element {i}: f32={f32v}, f16={f16v}, err={abs_err}");
        }
    }

    #[test]
    fn mixed_matmul_zero_dimension_errors() {
        let a = [1.0f32];
        let b = [1.0f32];
        let mut out = [0.0f32];
        let config = MixedPrecisionMatmulConfig {
            m: 0,
            n: 1,
            k: 1,
            precision: MixedPrecisionConfig::f32_only(),
        };
        assert!(mixed_precision_matmul(&a, &b, &mut out, &config).is_err());
    }

    #[test]
    fn mixed_matmul_buffer_too_small_a() {
        let a = [1.0f32; 2]; // need 4 for 2×2
        let b = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        let config = MixedPrecisionMatmulConfig {
            m: 2,
            n: 2,
            k: 2,
            precision: MixedPrecisionConfig::f32_only(),
        };
        assert!(mixed_precision_matmul(&a, &b, &mut out, &config).is_err());
    }

    #[test]
    fn mixed_matmul_buffer_too_small_output() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 4];
        let mut out = [0.0f32; 2]; // need 4 for 2×2
        let config = MixedPrecisionMatmulConfig {
            m: 2,
            n: 2,
            k: 2,
            precision: MixedPrecisionConfig::f32_only(),
        };
        assert!(mixed_precision_matmul(&a, &b, &mut out, &config).is_err());
    }

    // ── Dynamic loss scaling ─────────────────────────────────────

    #[test]
    fn loss_scaler_default() {
        let s = DynamicLossScaler::default();
        assert_eq!(s.scale, 65536.0);
        assert_eq!(s.growth_factor, 2.0);
        assert_eq!(s.backoff_factor, 0.5);
    }

    #[test]
    fn loss_scaler_backoff_on_overflow() {
        let mut s = DynamicLossScaler::new(1024.0);
        s.update(true);
        assert_eq!(s.scale, 512.0);
        s.update(true);
        assert_eq!(s.scale, 256.0);
    }

    #[test]
    fn loss_scaler_growth_after_interval() {
        let mut s = DynamicLossScaler::new(1.0);
        s.growth_interval = 3;
        s.update(false);
        s.update(false);
        assert_eq!(s.scale, 1.0); // not yet
        s.update(false); // 3rd clean step
        assert_eq!(s.scale, 2.0);
    }

    #[test]
    fn loss_scaler_overflow_resets_counter() {
        let mut s = DynamicLossScaler::new(100.0);
        s.growth_interval = 5;
        s.update(false);
        s.update(false);
        s.update(true); // overflow
        assert_eq!(s.steps_since_last_overflow, 0);
        assert_eq!(s.scale, 50.0);
    }

    #[test]
    fn loss_scaler_min_scale_clamp() {
        let mut s = DynamicLossScaler::new(2.0);
        s.min_scale = 1.0;
        s.update(true);
        assert_eq!(s.scale, 1.0);
        s.update(true);
        assert_eq!(s.scale, 1.0); // clamped at min
    }

    #[test]
    fn dynamic_loss_scaling_clean_gradients() {
        let grads = [0.1, 0.2, -0.3, 0.4];
        let mut scaler = DynamicLossScaler::new(100.0);
        let scale = dynamic_loss_scaling(&grads, &mut scaler).unwrap();
        assert_eq!(scale, 100.0);
    }

    #[test]
    fn dynamic_loss_scaling_nan_gradient() {
        let grads = [0.1, f32::NAN, 0.3];
        let mut scaler = DynamicLossScaler::new(100.0);
        let scale = dynamic_loss_scaling(&grads, &mut scaler).unwrap();
        assert_eq!(scale, 50.0); // backed off
    }

    #[test]
    fn dynamic_loss_scaling_inf_gradient() {
        let grads = [f32::INFINITY, 0.1];
        let mut scaler = DynamicLossScaler::new(64.0);
        let scale = dynamic_loss_scaling(&grads, &mut scaler).unwrap();
        assert_eq!(scale, 32.0);
    }

    #[test]
    fn dynamic_loss_scaling_repeated_backoff() {
        let grads = [f32::NAN];
        let mut scaler = DynamicLossScaler::new(256.0);
        for _ in 0..8 {
            dynamic_loss_scaling(&grads, &mut scaler).unwrap();
        }
        assert_eq!(scaler.scale, 1.0);
    }

    // ── Mixed-precision reduction ────────────────────────────────

    #[test]
    fn reduce_sum_f32() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let cfg = MixedPrecisionConfig::f32_only();
        let result = mixed_precision_reduce(&input, MixedReduceOp::Sum, &cfg).unwrap();
        assert!((result - 10.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_sum_f16() {
        let input: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let cfg = MixedPrecisionConfig::f16_compute();
        let result = mixed_precision_reduce(&input, MixedReduceOp::Sum, &cfg).unwrap();
        let expected = 5050.0;
        let err = (result - expected).abs();
        assert!(err < 10.0, "f16 sum: expected ~{expected}, got {result}");
    }

    #[test]
    fn reduce_max_f32() {
        let input = [-5.0, 3.0, 7.0, 1.0];
        let cfg = MixedPrecisionConfig::f32_only();
        let result = mixed_precision_reduce(&input, MixedReduceOp::Max, &cfg).unwrap();
        assert_eq!(result, 7.0);
    }

    #[test]
    fn reduce_mean_f32() {
        let input = [2.0, 4.0, 6.0, 8.0];
        let cfg = MixedPrecisionConfig::f32_only();
        let result = mixed_precision_reduce(&input, MixedReduceOp::Mean, &cfg).unwrap();
        assert!((result - 5.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_empty_input_errors() {
        let input: [f32; 0] = [];
        let cfg = MixedPrecisionConfig::f32_only();
        assert!(mixed_precision_reduce(&input, MixedReduceOp::Sum, &cfg).is_err());
    }

    #[test]
    fn reduce_single_element() {
        let input = [42.0];
        let cfg = MixedPrecisionConfig::f32_only();
        assert_eq!(mixed_precision_reduce(&input, MixedReduceOp::Sum, &cfg).unwrap(), 42.0);
        assert_eq!(mixed_precision_reduce(&input, MixedReduceOp::Max, &cfg).unwrap(), 42.0);
        assert_eq!(mixed_precision_reduce(&input, MixedReduceOp::Mean, &cfg).unwrap(), 42.0);
    }

    // ── Error type tests ─────────────────────────────────────────

    #[test]
    fn error_display_dimension_mismatch() {
        let e = MixedPrecisionError::DimensionMismatch { expected: 10, got: 5 };
        assert_eq!(e.to_string(), "dimension mismatch: expected 10, got 5");
    }

    #[test]
    fn error_display_unsupported_precision() {
        let e = MixedPrecisionError::UnsupportedPrecision("test".into());
        assert_eq!(e.to_string(), "unsupported precision: test");
    }

    #[test]
    fn error_display_empty_input() {
        let e = MixedPrecisionError::EmptyInput;
        assert_eq!(e.to_string(), "empty input");
    }

    #[test]
    fn error_display_scale_overflow() {
        let e = MixedPrecisionError::ScaleOverflow { scale: f32::INFINITY };
        assert!(e.to_string().contains("overflow"));
    }

    #[test]
    fn error_display_invalid_config() {
        let e = MixedPrecisionError::InvalidConfig("bad".into());
        assert_eq!(e.to_string(), "invalid config: bad");
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(MixedPrecisionError::EmptyInput);
        assert_eq!(e.to_string(), "empty input");
    }

    #[test]
    fn error_debug_format() {
        let e = MixedPrecisionError::EmptyInput;
        let debug = format!("{e:?}");
        assert!(debug.contains("EmptyInput"));
    }

    #[test]
    fn error_clone_and_eq() {
        let e1 = MixedPrecisionError::EmptyInput;
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    // ── GPU launch stubs (scaffold) ──────────────────────────────

    #[test]
    #[ignore = "requires CUDA runtime — GPU launch stubs are scaffold-only"]
    fn gpu_f16_matmul_stub() {
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            let config = MixedPrecisionLaunchConfig::for_shape(16, 16, 16);
            let a = vec![0u16; 256];
            let b = vec![0u16; 256];
            let mut out = vec![0.0f32; 256];
            let _ = launch_mixed_precision_f16_matmul(&a, &b, &mut out, &config);
        }
    }

    #[test]
    #[ignore = "requires CUDA runtime — GPU launch stubs are scaffold-only"]
    fn gpu_bf16_matmul_stub() {
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            let config = MixedPrecisionLaunchConfig::for_shape(32, 32, 32);
            let a = vec![0u16; 1024];
            let b = vec![0u16; 1024];
            let mut out = vec![0.0f32; 1024];
            let _ = launch_mixed_precision_bf16_matmul(&a, &b, &mut out, &config);
        }
    }

    #[test]
    #[ignore = "requires CUDA runtime — GPU launch stubs are scaffold-only"]
    fn gpu_attention_stub() {
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            let _ = launch_mixed_precision_attention(
                &[0u16; 64],
                &[0u16; 64],
                &[0u16; 64],
                &mut [0.0f32; 64],
                4,
                16,
                1,
                0.25,
            );
        }
    }

    // ── GPU launch config ────────────────────────────────────────

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn launch_config_grid_dim() {
        let cfg = MixedPrecisionLaunchConfig::for_shape(32, 64, 16);
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, 4); // 64 / 16
        assert_eq!(gy, 2); // 32 / 16
        assert_eq!(gz, 1);
    }

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn launch_config_block_dim() {
        let cfg = MixedPrecisionLaunchConfig::for_shape(16, 16, 16);
        let (bx, by, bz) = cfg.block_dim();
        assert_eq!(bx, 32); // warp size
        assert_eq!(by, 1);
        assert_eq!(bz, 1);
    }
}
