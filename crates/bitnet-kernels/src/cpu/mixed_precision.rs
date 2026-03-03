//! CPU mixed-precision inference kernels.
//!
//! Provides software-emulated F16 and BF16 conversions, mixed-precision
//! matrix multiplication, dynamic loss scaling, and precision diagnostics.
//! All half-precision representations use pure bit-manipulation — no
//! hardware F16/BF16 instructions are required.
//!
//! # IEEE 754 layouts
//!
//! | Format | Sign | Exponent | Mantissa | Bias |
//! |--------|------|----------|----------|------|
//! | F16    | 1    | 5        | 10       | 15   |
//! | BF16   | 1    | 8        | 7        | 127  |
//! | F32    | 1    | 8        | 23       | 127  |
//!
//! AVX2 fast-paths are provided for batch conversions when the target is
//! x86_64 and the feature is detected at runtime.
#![allow(unsafe_op_in_unsafe_fn)]

use bitnet_common::{BitNetError, KernelError, Result};

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

// ── Helpers ────────────────────────────────────────────────────────────

fn invalid_args(reason: impl Into<String>) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.into() })
}

// ── Data types ─────────────────────────────────────────────────────────

/// Floating-point data types for mixed-precision inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 16-bit IEEE 754 half-precision.
    F16,
    /// 16-bit brain floating-point (same exponent range as F32).
    BF16,
    /// 32-bit IEEE 754 single precision.
    F32,
    /// 64-bit IEEE 754 double precision.
    F64,
}

impl DType {
    /// Size in bytes of a single element.
    #[inline]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Human-readable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ── Configuration ──────────────────────────────────────────────────────

/// Describes the precision strategy for a mixed-precision forward pass.
#[derive(Debug, Clone, Copy)]
pub struct PrecisionConfig {
    /// Data type used for compute (matmul inner loop).
    pub compute_dtype: DType,
    /// Data type used for weight / activation storage.
    pub storage_dtype: DType,
    /// Data type used for accumulation registers.
    pub accumulate_dtype: DType,
    /// Initial loss scale for dynamic loss scaling (must be > 0).
    pub loss_scale: f32,
}

impl PrecisionConfig {
    /// Sensible default: compute and accumulate in F32, store in F16.
    pub const DEFAULT: Self = Self {
        compute_dtype: DType::F32,
        storage_dtype: DType::F16,
        accumulate_dtype: DType::F32,
        loss_scale: 1024.0,
    };

    /// BF16 storage with F32 compute/accumulate.
    pub const BF16_MIXED: Self = Self {
        compute_dtype: DType::F32,
        storage_dtype: DType::BF16,
        accumulate_dtype: DType::F32,
        loss_scale: 1024.0,
    };

    /// Validate that the configuration is self-consistent.
    pub fn validate(&self) -> Result<()> {
        if self.loss_scale <= 0.0 || !self.loss_scale.is_finite() {
            return Err(invalid_args("loss_scale must be a positive finite number"));
        }
        if self.accumulate_dtype.size_bytes() < self.compute_dtype.size_bytes() {
            return Err(invalid_args("accumulate_dtype must be at least as wide as compute_dtype"));
        }
        Ok(())
    }
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ── Software F16 conversion (IEEE 754 half-precision) ──────────────────

/// Convert an `f32` value to its IEEE 754 half-precision (F16) `u16`
/// representation using pure bit manipulation.
///
/// Handles ±0, ±Inf, NaN, denormals, and rounding (round-to-nearest-even).
#[inline]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 0xFF {
        // Inf / NaN
        if mantissa == 0 {
            return (sign | 0x7C00) as u16; // ±Inf
        }
        // NaN — preserve some payload bits
        return (sign | 0x7C00 | (mantissa >> 13).max(1)) as u16;
    }

    // Re-bias: F32 bias=127, F16 bias=15 → subtract 112
    let new_exp = exponent - 112;

    if new_exp >= 31 {
        // Overflow → ±Inf
        return (sign | 0x7C00) as u16;
    }

    if new_exp <= 0 {
        // Denormal or zero in F16
        if new_exp < -10 {
            return sign as u16; // too small → ±0
        }
        // Denormalise: shift mantissa right, adding the implicit leading 1
        let m = (mantissa | 0x0080_0000) >> (1 - new_exp + 13);
        // Round-to-nearest-even
        let round_bit = 1u32 << (-new_exp + 12);
        let result = if (mantissa | 0x0080_0000) & (round_bit - 1) != 0 || (m & 1) != 0 {
            sign | ((m + ((mantissa | 0x0080_0000) >> (1 - new_exp + 13 - 1))) & 1)
        } else {
            sign | m
        };
        return result as u16;
    }

    // Normal case: round-to-nearest-even on the 13 truncated bits
    let round_bit = 1u32 << 12;
    let truncated = mantissa & ((1u32 << 13) - 1);
    let half_way = round_bit;
    let m13 = mantissa >> 13;

    let rounded = if truncated > half_way || (truncated == half_way && (m13 & 1) != 0) {
        m13 + 1
    } else {
        m13
    };

    // Mantissa carry may bump exponent
    if rounded >= 0x0400 {
        let ne = new_exp + 1;
        if ne >= 31 {
            return (sign | 0x7C00) as u16; // overflow
        }
        return (sign | ((ne as u32) << 10)) as u16;
    }

    (sign | ((new_exp as u32) << 10) | rounded) as u16
}

/// Convert an IEEE 754 half-precision (F16) `u16` back to `f32`.
#[inline]
pub fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half as u32) & 0x8000) << 16;
    let exponent = ((half >> 10) & 0x1F) as u32;
    let mantissa = (half & 0x03FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign); // ±0
        }
        // Denormal → normalise
        let mut m = mantissa;
        let mut e: i32 = -14 + 127; // F16 denorm exp → F32 exp
        while m & 0x0400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF; // remove implicit bit
        let bits = sign | ((e as u32) << 23) | (m << 13);
        return f32::from_bits(bits);
    }

    if exponent == 31 {
        if mantissa == 0 {
            return f32::from_bits(sign | 0x7F80_0000); // ±Inf
        }
        return f32::from_bits(sign | 0x7FC0_0000 | (mantissa << 13)); // NaN
    }

    // Normal: re-bias 15 → 127 (add 112)
    let bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    f32::from_bits(bits)
}

// ── Software BF16 conversion ───────────────────────────────────────────

/// Convert an `f32` value to BF16 `u16` using round-to-nearest-even.
///
/// BF16 is simply the upper 16 bits of F32 with rounding applied to the
/// truncated lower 16 bits.
#[inline]
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();

    // NaN: ensure the quiet-NaN bit is set so the result stays NaN
    if value.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }

    // Round-to-nearest-even on bit 15 (the MSB being truncated)
    let round_bit = 1u32 << 15;
    let truncated = bits & 0xFFFF;
    let upper = bits >> 16;

    if truncated > round_bit || (truncated == round_bit && (upper & 1) != 0) {
        (upper + 1) as u16
    } else {
        upper as u16
    }
}

/// Convert a BF16 `u16` back to `f32`.
#[inline]
pub fn bf16_to_f32(bf: u16) -> f32 {
    f32::from_bits((bf as u32) << 16)
}

// ── Batch conversions ──────────────────────────────────────────────────

/// Convert a slice of `f32` values to F16 in bulk.
pub fn f32_slice_to_f16(src: &[f32], dst: &mut [u16]) -> Result<()> {
    if src.len() != dst.len() {
        return Err(invalid_args(format!(
            "f32_slice_to_f16: length mismatch ({} vs {})",
            src.len(),
            dst.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && src.len() >= 8 {
            // SAFETY: AVX2 detected, pointers valid, lengths checked.
            unsafe { f32_slice_to_f16_avx2(src, dst) };
            return Ok(());
        }
    }

    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f32_to_f16(*s);
    }
    Ok(())
}

/// Convert a slice of F16 values to `f32` in bulk.
pub fn f16_slice_to_f32(src: &[u16], dst: &mut [f32]) -> Result<()> {
    if src.len() != dst.len() {
        return Err(invalid_args(format!(
            "f16_slice_to_f32: length mismatch ({} vs {})",
            src.len(),
            dst.len()
        )));
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f16_to_f32(*s);
    }
    Ok(())
}

/// Convert a slice of `f32` values to BF16 in bulk.
pub fn f32_slice_to_bf16(src: &[f32], dst: &mut [u16]) -> Result<()> {
    if src.len() != dst.len() {
        return Err(invalid_args(format!(
            "f32_slice_to_bf16: length mismatch ({} vs {})",
            src.len(),
            dst.len()
        )));
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f32_to_bf16(*s);
    }
    Ok(())
}

/// Convert a slice of BF16 values to `f32` in bulk.
pub fn bf16_slice_to_f32(src: &[u16], dst: &mut [f32]) -> Result<()> {
    if src.len() != dst.len() {
        return Err(invalid_args(format!(
            "bf16_slice_to_f32: length mismatch ({} vs {})",
            src.len(),
            dst.len()
        )));
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = bf16_to_f32(*s);
    }
    Ok(())
}

// ── AVX2 batch F32→F16 conversion ─────────────────────────────────────

/// AVX2-accelerated F32→F16 batch conversion.
///
/// Processes 8 floats at a time using integer SIMD for the bit-field
/// extraction / rounding, with a scalar tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn f32_slice_to_f16_avx2(src: &[f32], dst: &mut [u16]) {
    let n = src.len();
    let chunks = n / 8;

    let bias_sub = _mm256_set1_epi32(112);
    let inf_exp = _mm256_set1_epi32(31);
    let zero = _mm256_setzero_si256();
    let max_mantissa = _mm256_set1_epi32(0x03FF);

    for i in 0..chunks {
        let base = i * 8;
        let raw = _mm256_loadu_ps(src.as_ptr().add(base));
        let bits = _mm256_castps_si256(raw);

        let sign = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(0x8000));
        let exp32 = _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(0xFF));
        let man32 = _mm256_and_si256(bits, _mm256_set1_epi32(0x007F_FFFF));

        // Re-bias exponent
        let new_exp = _mm256_sub_epi32(exp32, bias_sub);

        // Clamp exponent and compute mantissa (simplified: truncation)
        let clamped_exp = _mm256_min_epi32(_mm256_max_epi32(new_exp, zero), inf_exp);
        let man16 = _mm256_and_si256(_mm256_srli_epi32(man32, 13), max_mantissa);

        // Detect overflow (exp >= 31) → force Inf
        let is_ovf = _mm256_cmpeq_epi32(clamped_exp, inf_exp);
        let is_zero_or_denorm = _mm256_cmpeq_epi32(_mm256_max_epi32(new_exp, zero), zero);

        // Assemble: sign | (exp << 10) | mantissa
        let shifted_exp = _mm256_slli_epi32(clamped_exp, 10);
        let mut result = _mm256_or_si256(sign, _mm256_or_si256(shifted_exp, man16));

        // Overflow → 0x7C00 | sign
        let inf_val = _mm256_or_si256(sign, _mm256_set1_epi32(0x7C00));
        result = _mm256_blendv_epi8(result, inf_val, is_ovf);

        // Underflow → sign only (±0)
        result = _mm256_blendv_epi8(result, sign, is_zero_or_denorm);

        // Pack 32-bit lanes to 16-bit and store
        let packed_lo = _mm256_and_si256(result, _mm256_set1_epi32(0xFFFF));
        let arr: [i32; 8] = std::mem::transmute(packed_lo);
        for (j, &val) in arr.iter().enumerate() {
            *dst.get_unchecked_mut(base + j) = val as u16;
        }
    }

    // Scalar tail
    for i in (chunks * 8)..n {
        *dst.get_unchecked_mut(i) = f32_to_f16(*src.get_unchecked(i));
    }
}

// ── Mixed-precision matrix multiplication ──────────────────────────────

/// Mixed-precision matmul: `C[m×n] = A[m×k] · B[k×n]`.
///
/// Weights in `b_f16` are stored as F16 `u16` values; activations and
/// output are f32.  Accumulation is in f32 regardless of `config`.
/// This simulates the common AMP pattern: store weights in low precision,
/// compute in high precision.
pub fn mixed_matmul(
    a: &[f32],
    b_f16: &[u16],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &PrecisionConfig,
) -> Result<()> {
    config.validate()?;
    if a.len() < m * k {
        return Err(invalid_args(format!("mixed_matmul: A too small ({} < {})", a.len(), m * k)));
    }
    if b_f16.len() < k * n {
        return Err(invalid_args(format!(
            "mixed_matmul: B too small ({} < {})",
            b_f16.len(),
            k * n
        )));
    }
    if out.len() < m * n {
        return Err(invalid_args(format!(
            "mixed_matmul: out too small ({} < {})",
            out.len(),
            m * n
        )));
    }

    // Row-major A[m,k] × row-major B[k,n] → C[m,n]
    for i in 0..m {
        for j in 0..n {
            let mut acc: f64 = 0.0;
            for p in 0..k {
                let a_val = a[i * k + p] as f64;
                let b_val = f16_to_f32(b_f16[p * n + j]) as f64;
                acc += a_val * b_val;
            }
            out[i * n + j] = acc as f32;
        }
    }
    Ok(())
}

/// Mixed-precision matmul with BF16 weights.
pub fn mixed_matmul_bf16(
    a: &[f32],
    b_bf16: &[u16],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &PrecisionConfig,
) -> Result<()> {
    config.validate()?;
    if a.len() < m * k {
        return Err(invalid_args(format!(
            "mixed_matmul_bf16: A too small ({} < {})",
            a.len(),
            m * k
        )));
    }
    if b_bf16.len() < k * n {
        return Err(invalid_args(format!(
            "mixed_matmul_bf16: B too small ({} < {})",
            b_bf16.len(),
            k * n
        )));
    }
    if out.len() < m * n {
        return Err(invalid_args(format!(
            "mixed_matmul_bf16: out too small ({} < {})",
            out.len(),
            m * n
        )));
    }

    for i in 0..m {
        for j in 0..n {
            let mut acc: f64 = 0.0;
            for p in 0..k {
                let a_val = a[i * k + p] as f64;
                let b_val = bf16_to_f32(b_bf16[p * n + j]) as f64;
                acc += a_val * b_val;
            }
            out[i * n + j] = acc as f32;
        }
    }
    Ok(())
}

// ── Dynamic loss scaling ───────────────────────────────────────────────

/// State for adaptive loss scaling used in mixed-precision training.
#[derive(Debug, Clone)]
pub struct DynamicLossScaler {
    /// Current scale factor.
    pub scale: f32,
    /// Growth factor applied when no overflow is detected.
    pub growth_factor: f32,
    /// Backoff factor applied when overflow is detected.
    pub backoff_factor: f32,
    /// Number of consecutive steps without overflow before growing.
    pub growth_interval: u32,
    /// Counter of consecutive clean steps.
    consecutive_ok: u32,
}

impl DynamicLossScaler {
    /// Create a new scaler with the given initial scale.
    pub fn new(initial_scale: f32) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            consecutive_ok: 0,
        }
    }

    /// Report the outcome of a training step and update the scale.
    ///
    /// Returns the current scale *after* the update.
    pub fn update(&mut self, overflow_detected: bool) -> f32 {
        if overflow_detected {
            self.scale *= self.backoff_factor;
            if self.scale < 1.0 {
                self.scale = 1.0;
            }
            self.consecutive_ok = 0;
        } else {
            self.consecutive_ok += 1;
            if self.consecutive_ok >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.consecutive_ok = 0;
            }
        }
        self.scale
    }
}

/// Apply dynamic loss scaling: multiply each gradient by `scale`.
pub fn dynamic_loss_scaling(gradients: &[f32], scaled: &mut [f32], scale: f32) -> Result<()> {
    if gradients.len() != scaled.len() {
        return Err(invalid_args(format!(
            "dynamic_loss_scaling: length mismatch ({} vs {})",
            gradients.len(),
            scaled.len()
        )));
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(invalid_args("dynamic_loss_scaling: scale must be positive and finite"));
    }
    for (g, s) in gradients.iter().zip(scaled.iter_mut()) {
        *s = *g * scale;
    }
    Ok(())
}

// ── Gradient scaling (unscale after loss scaling) ──────────────────────

/// Unscale gradients by dividing by `scale` (inverse of loss scaling).
///
/// Returns `true` if any element is non-finite after unscaling (overflow).
pub fn gradient_scaling(gradients: &mut [f32], scale: f32) -> Result<bool> {
    if !scale.is_finite() || scale == 0.0 {
        return Err(invalid_args("gradient_scaling: scale must be finite and non-zero"));
    }
    let inv = 1.0 / scale;
    let mut has_overflow = false;
    for g in gradients.iter_mut() {
        *g *= inv;
        if !g.is_finite() {
            has_overflow = true;
        }
    }
    Ok(has_overflow)
}

// ── Overflow detection ─────────────────────────────────────────────────

/// Check a tensor for non-finite values (±Inf or NaN).
///
/// Returns `(inf_count, nan_count)`.
pub fn overflow_check(data: &[f32]) -> (usize, usize) {
    let mut infs = 0usize;
    let mut nans = 0usize;
    for &v in data {
        if v.is_nan() {
            nans += 1;
        } else if v.is_infinite() {
            infs += 1;
        }
    }
    (infs, nans)
}

// ── Mixed-precision forward pass ───────────────────────────────────────

/// Simulate a mixed-precision forward pass through a single linear layer.
///
/// 1. Cast `weights_f32` to F16 (simulating stored weights).
/// 2. Compute `output = input × weights^T` using F16 weights with F32
///    accumulation.
/// 3. Add `bias` (if provided) in F32.
///
/// `input` is `[batch, in_features]`, `weights_f32` is `[out_features,
/// in_features]`, `bias` is `[out_features]`, `output` is `[batch,
/// out_features]`.
pub fn mixed_precision_forward(
    input: &[f32],
    weights_f32: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
    config: &PrecisionConfig,
) -> Result<()> {
    config.validate()?;
    if input.len() < batch * in_features {
        return Err(invalid_args("mixed_precision_forward: input too small"));
    }
    if weights_f32.len() < out_features * in_features {
        return Err(invalid_args("mixed_precision_forward: weights too small"));
    }
    if output.len() < batch * out_features {
        return Err(invalid_args("mixed_precision_forward: output too small"));
    }
    if let Some(b) = bias
        && b.len() < out_features
    {
        return Err(invalid_args("mixed_precision_forward: bias too small"));
    }

    // Step 1: quantise weights to storage dtype (F16 or BF16)
    let total_weights = out_features * in_features;
    let mut w_half = vec![0u16; total_weights];
    match config.storage_dtype {
        DType::BF16 => {
            for (i, &w) in weights_f32[..total_weights].iter().enumerate() {
                w_half[i] = f32_to_bf16(w);
            }
        }
        _ => {
            // Default to F16
            for (i, &w) in weights_f32[..total_weights].iter().enumerate() {
                w_half[i] = f32_to_f16(w);
            }
        }
    }

    // Step 2: matmul with F32 accumulation
    // weights are [out_features, in_features] stored row-major
    // We compute input[b, :] · weights[o, :]^T for each (b, o).
    for b_idx in 0..batch {
        for o in 0..out_features {
            let mut acc: f64 = 0.0;
            for f in 0..in_features {
                let x = input[b_idx * in_features + f] as f64;
                let w = match config.storage_dtype {
                    DType::BF16 => bf16_to_f32(w_half[o * in_features + f]) as f64,
                    _ => f16_to_f32(w_half[o * in_features + f]) as f64,
                };
                acc += x * w;
            }
            output[b_idx * out_features + o] = acc as f32;
        }
    }

    // Step 3: add bias
    if let Some(b) = bias {
        for b_idx in 0..batch {
            for o in 0..out_features {
                output[b_idx * out_features + o] += b[o];
            }
        }
    }

    Ok(())
}

// ── Auto-cast ──────────────────────────────────────────────────────────

/// Auto-cast an f32 slice to the target dtype, storing the result in
/// `dst` as packed `u16` values (F16 or BF16).
///
/// For F32→F32 or F64, this is a no-op error because the destination is
/// `u16`.
pub fn auto_cast(src: &[f32], dst: &mut [u16], target: DType) -> Result<()> {
    match target {
        DType::F16 => f32_slice_to_f16(src, dst),
        DType::BF16 => f32_slice_to_bf16(src, dst),
        other => Err(invalid_args(format!(
            "auto_cast: cannot cast f32 to u16 for dtype {other}; \
             only F16 and BF16 targets are supported"
        ))),
    }
}

// ── Precision statistics ───────────────────────────────────────────────

/// Statistics about precision loss when converting f32 data to a lower
/// precision format.
#[derive(Debug, Clone, Copy)]
pub struct PrecisionStats {
    /// Maximum absolute error across all elements.
    pub max_abs_error: f64,
    /// Mean absolute error across all elements.
    pub mean_abs_error: f64,
    /// Root-mean-square error.
    pub rmse: f64,
    /// Number of elements that became ±Inf after conversion.
    pub overflow_count: usize,
    /// Number of non-zero elements that became zero.
    pub underflow_count: usize,
    /// Total number of elements examined.
    pub total: usize,
}

/// Measure precision loss of an F32→F16→F32 round-trip.
pub fn precision_stats_f16(data: &[f32]) -> PrecisionStats {
    precision_stats_impl(data, f32_to_f16, f16_to_f32)
}

/// Measure precision loss of an F32→BF16→F32 round-trip.
pub fn precision_stats_bf16(data: &[f32]) -> PrecisionStats {
    precision_stats_impl(data, f32_to_bf16, bf16_to_f32)
}

fn precision_stats_impl(
    data: &[f32],
    to_half: fn(f32) -> u16,
    from_half: fn(u16) -> f32,
) -> PrecisionStats {
    let mut max_abs: f64 = 0.0;
    let mut sum_abs: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    let mut overflow_count = 0usize;
    let mut underflow_count = 0usize;

    for &v in data {
        if !v.is_finite() {
            continue;
        }
        let rt = from_half(to_half(v));
        if rt.is_infinite() {
            overflow_count += 1;
            continue;
        }
        if v != 0.0 && rt == 0.0 {
            underflow_count += 1;
        }
        let err = (v as f64 - rt as f64).abs();
        if err > max_abs {
            max_abs = err;
        }
        sum_abs += err;
        sum_sq += err * err;
    }

    let n = data.iter().filter(|v| v.is_finite()).count();
    let nf = n.max(1) as f64;

    PrecisionStats {
        max_abs_error: max_abs,
        mean_abs_error: sum_abs / nf,
        rmse: (sum_sq / nf).sqrt(),
        overflow_count,
        underflow_count,
        total: data.len(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── DType basic tests ──────────────────────────────────────────────

    #[test]
    fn test_dtype_size_bytes() {
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F64.size_bytes(), 8);
    }

    #[test]
    fn test_dtype_name() {
        assert_eq!(DType::F16.name(), "f16");
        assert_eq!(DType::BF16.name(), "bf16");
        assert_eq!(DType::F32.name(), "f32");
        assert_eq!(DType::F64.name(), "f64");
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", DType::F16), "f16");
        assert_eq!(format!("{}", DType::F64), "f64");
    }

    #[test]
    fn test_dtype_eq_and_hash() {
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(DType::F16);
        s.insert(DType::BF16);
        s.insert(DType::F32);
        s.insert(DType::F64);
        assert_eq!(s.len(), 4);
        assert!(s.contains(&DType::F16));
    }

    // ── PrecisionConfig tests ──────────────────────────────────────────

    #[test]
    fn test_precision_config_default() {
        let cfg = PrecisionConfig::default();
        assert_eq!(cfg.compute_dtype, DType::F32);
        assert_eq!(cfg.storage_dtype, DType::F16);
        assert_eq!(cfg.accumulate_dtype, DType::F32);
        assert_eq!(cfg.loss_scale, 1024.0);
    }

    #[test]
    fn test_precision_config_bf16_mixed() {
        let cfg = PrecisionConfig::BF16_MIXED;
        assert_eq!(cfg.storage_dtype, DType::BF16);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_precision_config_validate_ok() {
        assert!(PrecisionConfig::DEFAULT.validate().is_ok());
    }

    #[test]
    fn test_precision_config_validate_bad_scale_zero() {
        let mut cfg = PrecisionConfig::DEFAULT;
        cfg.loss_scale = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_precision_config_validate_bad_scale_negative() {
        let mut cfg = PrecisionConfig::DEFAULT;
        cfg.loss_scale = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_precision_config_validate_bad_scale_inf() {
        let mut cfg = PrecisionConfig::DEFAULT;
        cfg.loss_scale = f32::INFINITY;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_precision_config_validate_bad_scale_nan() {
        let mut cfg = PrecisionConfig::DEFAULT;
        cfg.loss_scale = f32::NAN;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_precision_config_validate_narrow_accumulate() {
        let cfg = PrecisionConfig {
            compute_dtype: DType::F32,
            storage_dtype: DType::F16,
            accumulate_dtype: DType::F16, // narrower than compute
            loss_scale: 1.0,
        };
        assert!(cfg.validate().is_err());
    }

    // ── F16 conversion: basic values ───────────────────────────────────

    #[test]
    fn test_f16_zero() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_f16_negative_zero() {
        let h = f32_to_f16(-0.0);
        assert_eq!(h, 0x8000);
        let rt = f16_to_f32(h);
        assert!(rt == 0.0 && rt.is_sign_negative());
    }

    #[test]
    fn test_f16_one() {
        let h = f32_to_f16(1.0);
        assert_eq!(h, 0x3C00); // IEEE F16 encoding of 1.0
        assert!((f16_to_f32(h) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_minus_one() {
        let h = f32_to_f16(-1.0);
        assert_eq!(h, 0xBC00);
        assert!((f16_to_f32(h) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_half() {
        let h = f32_to_f16(0.5);
        assert!((f16_to_f32(h) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_f16_two() {
        let h = f32_to_f16(2.0);
        assert!((f16_to_f32(h) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_pi() {
        let h = f32_to_f16(std::f32::consts::PI);
        let rt = f16_to_f32(h);
        assert!((rt - std::f32::consts::PI).abs() < 0.002);
    }

    #[test]
    fn test_f16_inf_positive() {
        let h = f32_to_f16(f32::INFINITY);
        assert_eq!(h, 0x7C00);
        assert_eq!(f16_to_f32(h), f32::INFINITY);
    }

    #[test]
    fn test_f16_inf_negative() {
        let h = f32_to_f16(f32::NEG_INFINITY);
        assert_eq!(h, 0xFC00);
        assert_eq!(f16_to_f32(h), f32::NEG_INFINITY);
    }

    #[test]
    fn test_f16_nan() {
        let h = f32_to_f16(f32::NAN);
        assert!(f16_to_f32(h).is_nan());
    }

    #[test]
    fn test_f16_overflow_to_inf() {
        // F16 max normal ≈ 65504; values above should become Inf
        let h = f32_to_f16(100_000.0);
        assert_eq!(f16_to_f32(h), f32::INFINITY);
    }

    #[test]
    fn test_f16_underflow_to_zero() {
        // F16 min denormal ≈ 5.96e-8; very small values flush to zero
        let h = f32_to_f16(1e-10);
        assert_eq!(f16_to_f32(h), 0.0);
    }

    #[test]
    fn test_f16_max_normal() {
        // F16 max normal = 65504.0
        let h = f32_to_f16(65504.0);
        let rt = f16_to_f32(h);
        assert!((rt - 65504.0).abs() < 1.0);
    }

    #[test]
    fn test_f16_small_positive() {
        let h = f32_to_f16(0.001);
        let rt = f16_to_f32(h);
        assert!((rt - 0.001).abs() < 1e-4);
    }

    #[test]
    fn test_f16_denormal_region() {
        // F16 min denormal ≈ 5.96e-8
        let val = 6.0e-8_f32;
        let h = f32_to_f16(val);
        let rt = f16_to_f32(h);
        // Should be close or flush to zero
        assert!(rt.abs() < val * 2.0 || rt == 0.0);
    }

    #[test]
    fn test_f16_round_trip_accuracy_normal_range() {
        // For values in F16 normal range, round-trip error < 0.1%
        let values = [0.1, 0.5, 1.0, 10.0, 100.0, 1000.0, 60000.0];
        for &v in &values {
            let rt = f16_to_f32(f32_to_f16(v));
            let rel = ((rt - v) / v).abs();
            assert!(rel < 0.001, "f16 round-trip: {v} → {rt}, rel err = {rel}");
        }
    }

    #[test]
    fn test_f16_round_trip_negative_values() {
        let values = [-0.5, -1.0, -100.0, -65504.0];
        for &v in &values {
            let rt = f16_to_f32(f32_to_f16(v));
            let rel = ((rt - v) / v).abs();
            assert!(rel < 0.001, "f16 negative round-trip: {v} → {rt}, rel err = {rel}");
        }
    }

    // ── BF16 conversion: basic values ──────────────────────────────────

    #[test]
    fn test_bf16_zero() {
        assert_eq!(f32_to_bf16(0.0), 0x0000);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_bf16_negative_zero() {
        let b = f32_to_bf16(-0.0);
        assert_eq!(b, 0x8000);
        let rt = bf16_to_f32(b);
        assert!(rt == 0.0 && rt.is_sign_negative());
    }

    #[test]
    fn test_bf16_one() {
        let b = f32_to_bf16(1.0);
        assert_eq!(b, 0x3F80); // BF16 encoding of 1.0
        assert!((bf16_to_f32(b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_minus_one() {
        let b = f32_to_bf16(-1.0);
        assert_eq!(b, 0xBF80);
        assert!((bf16_to_f32(b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_pi() {
        let b = f32_to_bf16(std::f32::consts::PI);
        let rt = bf16_to_f32(b);
        assert!((rt - std::f32::consts::PI).abs() < 0.02);
    }

    #[test]
    fn test_bf16_inf_positive() {
        let b = f32_to_bf16(f32::INFINITY);
        assert_eq!(bf16_to_f32(b), f32::INFINITY);
    }

    #[test]
    fn test_bf16_inf_negative() {
        let b = f32_to_bf16(f32::NEG_INFINITY);
        assert_eq!(bf16_to_f32(b), f32::NEG_INFINITY);
    }

    #[test]
    fn test_bf16_nan() {
        let b = f32_to_bf16(f32::NAN);
        assert!(bf16_to_f32(b).is_nan());
    }

    #[test]
    fn test_bf16_large_value() {
        // BF16 can represent much larger values than F16 (same range as F32)
        let v = 100_000.0_f32;
        let rt = bf16_to_f32(f32_to_bf16(v));
        let rel = ((rt - v) / v).abs();
        assert!(rel < 0.01, "bf16 large: {v} → {rt}");
    }

    #[test]
    fn test_bf16_round_trip_accuracy() {
        let values = [0.1, 0.5, 1.0, 10.0, 100.0, 1e6, 1e10];
        for &v in &values {
            let rt = bf16_to_f32(f32_to_bf16(v));
            let rel = ((rt - v) / v).abs();
            assert!(rel < 0.01, "bf16 round-trip: {v} → {rt}, rel err = {rel}");
        }
    }

    #[test]
    fn test_bf16_small_values() {
        let v = 1e-20_f32;
        let rt = bf16_to_f32(f32_to_bf16(v));
        let rel = ((rt - v) / v).abs();
        assert!(rel < 0.01);
    }

    #[test]
    fn test_bf16_vs_f16_range_difference() {
        // BF16 can handle 100000 without overflow, F16 cannot
        let v = 100_000.0_f32;
        let f16_rt = f16_to_f32(f32_to_f16(v));
        let bf16_rt = bf16_to_f32(f32_to_bf16(v));
        assert!(f16_rt.is_infinite(), "f16 should overflow to inf");
        assert!(bf16_rt.is_finite(), "bf16 should NOT overflow");
    }

    // ── Batch conversion tests ─────────────────────────────────────────

    #[test]
    fn test_batch_f16_roundtrip() {
        let src = vec![0.0, 1.0, -1.0, 0.5, 100.0, 65504.0];
        let mut half = vec![0u16; src.len()];
        let mut dst = vec![0.0f32; src.len()];
        f32_slice_to_f16(&src, &mut half).unwrap();
        f16_slice_to_f32(&half, &mut dst).unwrap();
        for (s, d) in src.iter().zip(dst.iter()) {
            let err = (s - d).abs();
            assert!(err < s.abs() * 0.001 + 1e-7, "{s} → {d}");
        }
    }

    #[test]
    fn test_batch_bf16_roundtrip() {
        let src = vec![0.0, 1.0, -1.0, 0.5, 1e6, -1e10];
        let mut half = vec![0u16; src.len()];
        let mut dst = vec![0.0f32; src.len()];
        f32_slice_to_bf16(&src, &mut half).unwrap();
        bf16_slice_to_f32(&half, &mut dst).unwrap();
        for (s, d) in src.iter().zip(dst.iter()) {
            if *s == 0.0 {
                assert_eq!(*d, 0.0);
            } else {
                let rel = ((s - d) / s).abs();
                assert!(rel < 0.01, "{s} → {d}");
            }
        }
    }

    #[test]
    fn test_batch_f16_length_mismatch() {
        let src = vec![1.0; 4];
        let mut dst = vec![0u16; 3];
        assert!(f32_slice_to_f16(&src, &mut dst).is_err());
    }

    #[test]
    fn test_batch_bf16_length_mismatch() {
        let src = vec![1.0; 4];
        let mut dst = vec![0u16; 5];
        assert!(f32_slice_to_bf16(&src, &mut dst).is_err());
    }

    #[test]
    fn test_batch_f16_to_f32_length_mismatch() {
        let src = vec![0u16; 3];
        let mut dst = vec![0.0f32; 2];
        assert!(f16_slice_to_f32(&src, &mut dst).is_err());
    }

    #[test]
    fn test_batch_bf16_to_f32_length_mismatch() {
        let src = vec![0u16; 3];
        let mut dst = vec![0.0f32; 4];
        assert!(bf16_slice_to_f32(&src, &mut dst).is_err());
    }

    #[test]
    fn test_batch_f16_empty() {
        let src: Vec<f32> = vec![];
        let mut dst: Vec<u16> = vec![];
        assert!(f32_slice_to_f16(&src, &mut dst).is_ok());
    }

    #[test]
    fn test_batch_bf16_empty() {
        let src: Vec<f32> = vec![];
        let mut dst: Vec<u16> = vec![];
        assert!(f32_slice_to_bf16(&src, &mut dst).is_ok());
    }

    // ── AVX2 batch tests (run on x86_64 with AVX2) ────────────────────

    #[test]
    fn test_batch_f16_large_avx2_path() {
        // Exercises the AVX2 path (>= 8 elements) on supported hardware
        let n = 64;
        let src: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let mut half = vec![0u16; n];
        let mut dst = vec![0.0f32; n];
        f32_slice_to_f16(&src, &mut half).unwrap();
        f16_slice_to_f32(&half, &mut dst).unwrap();
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!((s - d).abs() < s.abs() * 0.002 + 1e-6, "AVX2 path: {s} → {d}");
        }
    }

    #[test]
    fn test_batch_f16_avx2_tail_handling() {
        // 11 elements: 8 via AVX2 + 3 scalar tail
        let n = 11;
        let src: Vec<f32> = (0..n).map(|i| (i as f32) - 5.0).collect();
        let mut half = vec![0u16; n];
        let mut dst = vec![0.0f32; n];
        f32_slice_to_f16(&src, &mut half).unwrap();
        f16_slice_to_f32(&half, &mut dst).unwrap();
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!((s - d).abs() < s.abs() * 0.002 + 1e-4, "tail: {s} → {d}");
        }
    }

    // ── Mixed-matmul tests ─────────────────────────────────────────────

    #[test]
    fn test_mixed_matmul_identity() {
        // 2×2 identity in F16 weights
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b_f32 = vec![1.0, 0.0, 0.0, 1.0]; // identity [2, 2]
        let mut b_f16 = vec![0u16; 4];
        for (i, &v) in b_f32.iter().enumerate() {
            b_f16[i] = f32_to_f16(v);
        }
        let mut out = vec![0.0f32; 4];
        mixed_matmul(&a, &b_f16, &mut out, 2, 2, 2, &PrecisionConfig::DEFAULT).unwrap();
        for (o, e) in out.iter().zip(a.iter()) {
            assert!((o - e).abs() < 1e-3, "identity: got {o}, expected {e}");
        }
    }

    #[test]
    fn test_mixed_matmul_1x1() {
        let a = [3.0f32];
        let b = [f32_to_f16(4.0)];
        let mut out = [0.0f32];
        mixed_matmul(&a, &b, &mut out, 1, 1, 1, &PrecisionConfig::DEFAULT).unwrap();
        assert!((out[0] - 12.0).abs() < 0.1);
    }

    #[test]
    fn test_mixed_matmul_a_too_small() {
        let a = [1.0f32; 3]; // need 4
        let b = [f32_to_f16(1.0); 4];
        let mut out = [0.0f32; 4];
        assert!(mixed_matmul(&a, &b, &mut out, 2, 2, 2, &PrecisionConfig::DEFAULT).is_err());
    }

    #[test]
    fn test_mixed_matmul_b_too_small() {
        let a = [1.0f32; 4];
        let b = [f32_to_f16(1.0); 3]; // need 4
        let mut out = [0.0f32; 4];
        assert!(mixed_matmul(&a, &b, &mut out, 2, 2, 2, &PrecisionConfig::DEFAULT).is_err());
    }

    #[test]
    fn test_mixed_matmul_out_too_small() {
        let a = [1.0f32; 4];
        let b = [f32_to_f16(1.0); 4];
        let mut out = [0.0f32; 3]; // need 4
        assert!(mixed_matmul(&a, &b, &mut out, 2, 2, 2, &PrecisionConfig::DEFAULT).is_err());
    }

    #[test]
    fn test_mixed_matmul_bf16() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b_f32 = vec![1.0, 0.0, 0.0, 1.0];
        let mut b_bf16 = vec![0u16; 4];
        for (i, &v) in b_f32.iter().enumerate() {
            b_bf16[i] = f32_to_bf16(v);
        }
        let mut out = vec![0.0f32; 4];
        mixed_matmul_bf16(&a, &b_bf16, &mut out, 2, 2, 2, &PrecisionConfig::BF16_MIXED).unwrap();
        for (o, e) in out.iter().zip(a.iter()) {
            assert!((o - e).abs() < 1e-2, "bf16 identity: {o} vs {e}");
        }
    }

    #[test]
    fn test_mixed_matmul_bf16_errors() {
        let a = [1.0f32; 2];
        let b = [f32_to_bf16(1.0); 4];
        let mut out = [0.0f32; 4];
        assert!(
            mixed_matmul_bf16(&a, &b, &mut out, 2, 2, 2, &PrecisionConfig::BF16_MIXED).is_err()
        );
    }

    // ── Dynamic loss scaling tests ─────────────────────────────────────

    #[test]
    fn test_dynamic_loss_scaling_basic() {
        let grads = vec![1.0, 2.0, -3.0];
        let mut scaled = vec![0.0; 3];
        dynamic_loss_scaling(&grads, &mut scaled, 2.0).unwrap();
        assert_eq!(scaled, vec![2.0, 4.0, -6.0]);
    }

    #[test]
    fn test_dynamic_loss_scaling_length_mismatch() {
        let grads = vec![1.0; 3];
        let mut scaled = vec![0.0; 2];
        assert!(dynamic_loss_scaling(&grads, &mut scaled, 1.0).is_err());
    }

    #[test]
    fn test_dynamic_loss_scaling_invalid_scale_zero() {
        let grads = vec![1.0];
        let mut scaled = vec![0.0];
        assert!(dynamic_loss_scaling(&grads, &mut scaled, 0.0).is_err());
    }

    #[test]
    fn test_dynamic_loss_scaling_invalid_scale_negative() {
        let grads = vec![1.0];
        let mut scaled = vec![0.0];
        assert!(dynamic_loss_scaling(&grads, &mut scaled, -1.0).is_err());
    }

    #[test]
    fn test_dynamic_loss_scaling_invalid_scale_nan() {
        let grads = vec![1.0];
        let mut scaled = vec![0.0];
        assert!(dynamic_loss_scaling(&grads, &mut scaled, f32::NAN).is_err());
    }

    // ── DynamicLossScaler state machine ────────────────────────────────

    #[test]
    fn test_scaler_new() {
        let s = DynamicLossScaler::new(1024.0);
        assert_eq!(s.scale, 1024.0);
        assert_eq!(s.growth_factor, 2.0);
        assert_eq!(s.backoff_factor, 0.5);
    }

    #[test]
    fn test_scaler_overflow_backs_off() {
        let mut s = DynamicLossScaler::new(1024.0);
        s.update(true);
        assert_eq!(s.scale, 512.0);
    }

    #[test]
    fn test_scaler_overflow_clamps_to_one() {
        let mut s = DynamicLossScaler::new(1.0);
        s.update(true);
        assert_eq!(s.scale, 1.0); // clamped
    }

    #[test]
    fn test_scaler_growth_after_interval() {
        let mut s = DynamicLossScaler::new(1024.0);
        s.growth_interval = 3;
        s.update(false);
        s.update(false);
        assert_eq!(s.scale, 1024.0); // not yet
        s.update(false);
        assert_eq!(s.scale, 2048.0); // now
    }

    #[test]
    fn test_scaler_overflow_resets_counter() {
        let mut s = DynamicLossScaler::new(1024.0);
        s.growth_interval = 3;
        s.update(false);
        s.update(false);
        s.update(true); // resets counter
        s.update(false);
        s.update(false);
        assert_eq!(s.scale, 512.0); // backed off, not grown
    }

    // ── Gradient scaling tests ─────────────────────────────────────────

    #[test]
    fn test_gradient_scaling_basic() {
        let mut grads = vec![2.0, 4.0, -6.0];
        let overflow = gradient_scaling(&mut grads, 2.0).unwrap();
        assert!(!overflow);
        assert_eq!(grads, vec![1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_gradient_scaling_detects_inf() {
        let mut grads = vec![f32::MAX, 1.0];
        let overflow = gradient_scaling(&mut grads, 1e-30).unwrap();
        assert!(overflow);
    }

    #[test]
    fn test_gradient_scaling_zero_scale_error() {
        let mut grads = vec![1.0];
        assert!(gradient_scaling(&mut grads, 0.0).is_err());
    }

    #[test]
    fn test_gradient_scaling_nan_scale_error() {
        let mut grads = vec![1.0];
        assert!(gradient_scaling(&mut grads, f32::NAN).is_err());
    }

    // ── Overflow check tests ───────────────────────────────────────────

    #[test]
    fn test_overflow_check_clean() {
        let data = vec![1.0, 2.0, -3.0, 0.0];
        assert_eq!(overflow_check(&data), (0, 0));
    }

    #[test]
    fn test_overflow_check_infs() {
        let data = vec![1.0, f32::INFINITY, f32::NEG_INFINITY];
        assert_eq!(overflow_check(&data), (2, 0));
    }

    #[test]
    fn test_overflow_check_nans() {
        let data = vec![f32::NAN, 1.0, f32::NAN];
        assert_eq!(overflow_check(&data), (0, 2));
    }

    #[test]
    fn test_overflow_check_mixed() {
        let data = vec![f32::NAN, f32::INFINITY, 1.0];
        assert_eq!(overflow_check(&data), (1, 1));
    }

    #[test]
    fn test_overflow_check_empty() {
        assert_eq!(overflow_check(&[]), (0, 0));
    }

    // ── mixed_precision_forward tests ──────────────────────────────────

    #[test]
    fn test_forward_identity_no_bias() {
        let input = vec![1.0, 2.0]; // [1, 2]
        let weights = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2] identity
        let mut output = vec![0.0; 2]; // [1, 2]
        mixed_precision_forward(
            &input,
            &weights,
            None,
            &mut output,
            1,
            2,
            2,
            &PrecisionConfig::DEFAULT,
        )
        .unwrap();
        assert!((output[0] - 1.0).abs() < 1e-3);
        assert!((output[1] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_forward_with_bias() {
        let input = vec![1.0, 0.0]; // [1, 2]
        let weights = vec![2.0, 0.0, 0.0, 3.0]; // [2, 2]
        let bias = vec![10.0, 20.0]; // [2]
        let mut output = vec![0.0; 2];
        mixed_precision_forward(
            &input,
            &weights,
            Some(&bias),
            &mut output,
            1,
            2,
            2,
            &PrecisionConfig::DEFAULT,
        )
        .unwrap();
        assert!((output[0] - 12.0).abs() < 0.1); // 2*1 + 0*0 + 10
        assert!((output[1] - 20.0).abs() < 0.1); // 0*1 + 3*0 + 20
    }

    #[test]
    fn test_forward_batched() {
        let input = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2]
        let weights = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2]
        let mut output = vec![0.0; 4]; // [2, 2]
        mixed_precision_forward(
            &input,
            &weights,
            None,
            &mut output,
            2,
            2,
            2,
            &PrecisionConfig::DEFAULT,
        )
        .unwrap();
        // Should approximate identity
        assert!((output[0] - 1.0).abs() < 1e-3);
        assert!((output[3] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_forward_bf16_storage() {
        let input = vec![1.0, 2.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0];
        let mut output = vec![0.0; 2];
        mixed_precision_forward(
            &input,
            &weights,
            None,
            &mut output,
            1,
            2,
            2,
            &PrecisionConfig::BF16_MIXED,
        )
        .unwrap();
        assert!((output[0] - 1.0).abs() < 0.1);
        assert!((output[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_forward_input_too_small() {
        let input = vec![1.0]; // need 2
        let weights = vec![1.0; 4];
        let mut output = vec![0.0; 2];
        assert!(
            mixed_precision_forward(
                &input,
                &weights,
                None,
                &mut output,
                1,
                2,
                2,
                &PrecisionConfig::DEFAULT
            )
            .is_err()
        );
    }

    #[test]
    fn test_forward_weights_too_small() {
        let input = vec![1.0; 2];
        let weights = vec![1.0; 3]; // need 4
        let mut output = vec![0.0; 2];
        assert!(
            mixed_precision_forward(
                &input,
                &weights,
                None,
                &mut output,
                1,
                2,
                2,
                &PrecisionConfig::DEFAULT
            )
            .is_err()
        );
    }

    #[test]
    fn test_forward_output_too_small() {
        let input = vec![1.0; 2];
        let weights = vec![1.0; 4];
        let mut output = vec![0.0; 1]; // need 2
        assert!(
            mixed_precision_forward(
                &input,
                &weights,
                None,
                &mut output,
                1,
                2,
                2,
                &PrecisionConfig::DEFAULT
            )
            .is_err()
        );
    }

    #[test]
    fn test_forward_bias_too_small() {
        let input = vec![1.0; 2];
        let weights = vec![1.0; 4];
        let mut output = vec![0.0; 2];
        let bias = vec![1.0]; // need 2
        assert!(
            mixed_precision_forward(
                &input,
                &weights,
                Some(&bias),
                &mut output,
                1,
                2,
                2,
                &PrecisionConfig::DEFAULT
            )
            .is_err()
        );
    }

    // ── Auto-cast tests ────────────────────────────────────────────────

    #[test]
    fn test_auto_cast_f16() {
        let src = vec![1.0, -1.0, 0.5];
        let mut dst = vec![0u16; 3];
        auto_cast(&src, &mut dst, DType::F16).unwrap();
        assert_eq!(dst[0], f32_to_f16(1.0));
    }

    #[test]
    fn test_auto_cast_bf16() {
        let src = vec![1.0, -1.0];
        let mut dst = vec![0u16; 2];
        auto_cast(&src, &mut dst, DType::BF16).unwrap();
        assert_eq!(dst[0], f32_to_bf16(1.0));
    }

    #[test]
    fn test_auto_cast_f32_errors() {
        let src = vec![1.0];
        let mut dst = vec![0u16; 1];
        assert!(auto_cast(&src, &mut dst, DType::F32).is_err());
    }

    #[test]
    fn test_auto_cast_f64_errors() {
        let src = vec![1.0];
        let mut dst = vec![0u16; 1];
        assert!(auto_cast(&src, &mut dst, DType::F64).is_err());
    }

    // ── Precision stats tests ──────────────────────────────────────────

    #[test]
    fn test_precision_stats_f16_zeros() {
        let data = vec![0.0; 10];
        let s = precision_stats_f16(&data);
        assert_eq!(s.max_abs_error, 0.0);
        assert_eq!(s.overflow_count, 0);
        assert_eq!(s.underflow_count, 0);
        assert_eq!(s.total, 10);
    }

    #[test]
    fn test_precision_stats_f16_normal() {
        let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let s = precision_stats_f16(&data);
        assert!(s.max_abs_error < 0.1);
        assert!(s.mean_abs_error < 0.05);
        assert!(s.rmse < 0.1);
        assert_eq!(s.overflow_count, 0);
        assert_eq!(s.total, 100);
    }

    #[test]
    fn test_precision_stats_f16_overflow_detection() {
        let data = vec![1.0, 100_000.0, 200_000.0];
        let s = precision_stats_f16(&data);
        assert_eq!(s.overflow_count, 2);
    }

    #[test]
    fn test_precision_stats_f16_underflow_detection() {
        let data = vec![1e-10, 1e-12];
        let s = precision_stats_f16(&data);
        assert_eq!(s.underflow_count, 2);
    }

    #[test]
    fn test_precision_stats_bf16_normal() {
        let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let s = precision_stats_bf16(&data);
        assert!(s.max_abs_error < 1.0);
        assert_eq!(s.overflow_count, 0);
        assert_eq!(s.total, 100);
    }

    #[test]
    fn test_precision_stats_bf16_no_overflow_at_100k() {
        let data = vec![100_000.0];
        let s = precision_stats_bf16(&data);
        assert_eq!(s.overflow_count, 0); // BF16 handles this
    }

    #[test]
    fn test_precision_stats_skips_non_finite_input() {
        let data = vec![1.0, f32::NAN, f32::INFINITY, 2.0];
        let s = precision_stats_f16(&data);
        assert_eq!(s.total, 4);
        // Only the two finite values contribute to error stats
        assert!(s.max_abs_error < 0.1);
    }

    #[test]
    fn test_precision_stats_empty() {
        let s = precision_stats_f16(&[]);
        assert_eq!(s.total, 0);
        assert_eq!(s.max_abs_error, 0.0);
    }
}
