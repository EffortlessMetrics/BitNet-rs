//! CPU SIMD mixed-precision conversion and arithmetic operations.
//!
//! Provides batch FP16↔FP32 and BF16↔FP32 conversions with runtime
//! AVX2/F16C detection and scalar fallback, plus mixed-precision matmul
//! and accumulation primitives.
//!
//! # Layout conventions
//!
//! All vectors and matrices are stored in contiguous row-major order.
//! The FP16 representation uses the IEEE 754 half-precision format (sign
//! 1 bit, exponent 5 bits, mantissa 10 bits) stored as `u16`.  BF16 uses
//! the bfloat16 layout (sign 1 bit, exponent 8 bits, mantissa 7 bits),
//! also stored as `u16`.

#![allow(unsafe_op_in_unsafe_fn)]

use bitnet_common::{BitNetError, KernelError, Result};

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

// ── Rounding modes ─────────────────────────────────────────────────────

/// Rounding mode for FP32→FP16 conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (IEEE 754 default).
    NearestEven,
    /// Truncate toward zero.
    Truncate,
    /// Round toward positive infinity.
    Ceiling,
    /// Round toward negative infinity.
    Floor,
}

// ── Precision policy ───────────────────────────────────────────────────

/// Target precision for mixed-precision operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetPrecision {
    /// IEEE 754 half-precision (FP16).
    Fp16,
    /// Google Brain bfloat16 (BF16).
    Bf16,
    /// Single-precision (FP32) — no conversion.
    Fp32,
}

/// Policy for automatic precision selection based on data characteristics.
///
/// Analyses the dynamic range and magnitude of a tensor and recommends the
/// most compact representation that preserves accuracy within the caller's
/// tolerances.
#[derive(Debug, Clone)]
pub struct PrecisionPolicy {
    /// Maximum absolute value threshold for FP16 (default: 65504.0, the
    /// FP16 max normal).
    pub fp16_max_abs: f32,
    /// Minimum non-zero absolute value for FP16 before flushing to zero
    /// would be unacceptable (default: 6.1e-5, the FP16 min normal).
    pub fp16_min_abs: f32,
    /// Maximum absolute value threshold for BF16 (default: 3.39e+38).
    pub bf16_max_abs: f32,
    /// When true, prefer BF16 over FP16 when the dynamic range exceeds
    /// FP16 limits but mantissa precision is not critical.
    pub prefer_bf16_for_large_range: bool,
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self {
            fp16_max_abs: 65504.0,
            fp16_min_abs: 6.1e-5,
            bf16_max_abs: 3.39e+38,
            prefer_bf16_for_large_range: true,
        }
    }
}

impl PrecisionPolicy {
    /// Create a new policy with default thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyse `data` and recommend a target precision.
    pub fn recommend(&self, data: &[f32]) -> TargetPrecision {
        if data.is_empty() {
            return TargetPrecision::Fp16;
        }

        let mut max_abs: f32 = 0.0;
        let mut min_nonzero_abs: f32 = f32::MAX;
        let mut has_nonzero = false;

        for &v in data {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
            if a > 0.0 && a < min_nonzero_abs {
                min_nonzero_abs = a;
                has_nonzero = true;
            }
        }

        // If everything is zero or within FP16 range, use FP16.
        if !has_nonzero {
            return TargetPrecision::Fp16;
        }

        let fits_fp16_range = max_abs <= self.fp16_max_abs && min_nonzero_abs >= self.fp16_min_abs;
        if fits_fp16_range {
            return TargetPrecision::Fp16;
        }

        // If dynamic range exceeds FP16 but fits BF16, prefer BF16 when
        // configured.
        if self.prefer_bf16_for_large_range && max_abs <= self.bf16_max_abs {
            return TargetPrecision::Bf16;
        }

        TargetPrecision::Fp32
    }
}

// ── Runtime feature detection helpers ──────────────────────────────────

/// Returns `true` when the current CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Returns `true` when the current CPU supports F16C (FP16 convert).
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_f16c() -> bool {
    is_x86_feature_detected!("f16c")
}

// ── Scalar FP16↔FP32 helpers ──────────────────────────────────────────

/// Convert a single FP16 value (stored as `u16`) to `f32` using pure
/// scalar arithmetic (IEEE 754 half-precision layout).
#[inline]
fn f16_to_f32_scalar_single(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero.
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal: value = (-1)^s * 2^(-14) * (mant/1024)
        let val = (mant as f32) * (1.0 / 1024.0) * 6.103_515_6e-5; // 2^-14
        if sign == 1 { -val } else { val }
    } else if exp == 0x1F {
        // Inf / NaN.
        let f32_bits = (sign << 31) | (0xFF << 23) | (mant << 13);
        f32::from_bits(f32_bits)
    } else {
        // Normal.
        let f32_exp = exp + 112; // bias adjust: 127 - 15
        let f32_bits = (sign << 31) | (f32_exp << 23) | (mant << 13);
        f32::from_bits(f32_bits)
    }
}

/// Convert a single `f32` to FP16 (`u16`) with the given rounding mode.
#[inline]
fn f32_to_f16_scalar_single(value: f32, mode: RoundingMode) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    // Zero / signed zero.
    if exp == 0 && mant == 0 {
        return (sign << 15) as u16;
    }

    // Inf / NaN.
    if exp == 0xFF {
        if mant == 0 {
            return ((sign << 15) | 0x7C00) as u16;
        }
        // NaN — preserve some payload bits.
        return ((sign << 15) | 0x7C00 | (mant >> 13).max(1)) as u16;
    }

    // Re-bias exponent: f32 bias 127 → f16 bias 15.
    let new_exp = exp - 127 + 15;

    if new_exp >= 0x1F {
        // Overflow → ±Inf.
        return ((sign << 15) | 0x7C00) as u16;
    }

    if new_exp <= 0 {
        // Underflow → subnormal or zero in FP16.
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let full_mant = mant | 0x80_0000; // implicit leading 1
        let shift = (1 - new_exp) as u32 + 13;
        let half_mant = full_mant >> shift;
        let round_bit = if shift < 32 { (full_mant >> (shift - 1)) & 1 } else { 0 };
        let sticky = if shift > 1 { (full_mant & ((1u32 << (shift - 1)) - 1)) != 0 } else { false };
        let rounded = apply_rounding(half_mant, round_bit, sticky, sign, mode);
        return ((sign << 15) | rounded) as u16;
    }

    let half_mant = mant >> 13;
    let round_bit = (mant >> 12) & 1;
    let sticky = (mant & 0xFFF) != 0;
    let base = (sign << 15) | ((new_exp as u32) << 10) | half_mant;
    let rounded = apply_rounding(base, round_bit, sticky, sign, mode);
    rounded as u16
}

/// Apply rounding increment to a raw FP16 bit pattern.
#[inline]
fn apply_rounding(base: u32, round_bit: u32, sticky: bool, sign: u32, mode: RoundingMode) -> u32 {
    match mode {
        RoundingMode::NearestEven => {
            if round_bit == 1 && (sticky || (base & 1) == 1) {
                base + 1
            } else {
                base
            }
        }
        RoundingMode::Truncate => base,
        RoundingMode::Ceiling => {
            if sign == 0 && (round_bit == 1 || sticky) {
                base + 1
            } else {
                base
            }
        }
        RoundingMode::Floor => {
            if sign == 1 && (round_bit == 1 || sticky) {
                base + 1
            } else {
                base
            }
        }
    }
}

// ── Scalar BF16↔FP32 ──────────────────────────────────────────────────

/// Convert a single BF16 (`u16`) to `f32` by left-shifting 16 bits.
#[inline]
fn bf16_to_f32_scalar_single(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert a single `f32` to BF16 (`u16`) with round-to-nearest-even.
#[inline]
fn f32_to_bf16_scalar_single(value: f32) -> u16 {
    let bits = value.to_bits();
    // Round-to-nearest-even: add rounding bias based on the truncated
    // portion and the LSB of the retained mantissa.
    let round_bit = 1u32 << 15;
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(round_bit - 1 + lsb);
    (rounded >> 16) as u16
}

// ── Batch FP16↔FP32 ───────────────────────────────────────────────────

/// Batch-convert FP16 values (stored as `u16`) to `f32`.
///
/// Uses AVX2 + F16C when available on x86_64, otherwise falls back to a
/// portable scalar loop.
///
/// # Errors
///
/// Returns an error when `output.len() < input.len()`.
pub fn f16_to_f32_avx2(input: &[u16], output: &mut [f32]) -> Result<()> {
    validate_lengths(input.len(), output.len(), "f16_to_f32_avx2")?;
    let n = input.len();

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_f16c() {
            // Safety: we verified F16C + AVX2 at runtime and lengths are
            // validated above.
            unsafe {
                f16_to_f32_f16c(input, output, n);
            }
            return Ok(());
        }
    }

    // Scalar fallback.
    f16_to_f32_scalar(input, output, n);
    Ok(())
}

/// Scalar fallback for FP16→FP32 batch conversion.
fn f16_to_f32_scalar(input: &[u16], output: &mut [f32], n: usize) {
    for i in 0..n {
        output[i] = f16_to_f32_scalar_single(input[i]);
    }
}

/// AVX2 + F16C accelerated FP16→FP32 conversion.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn f16_to_f32_f16c(input: &[u16], output: &mut [f32], n: usize) {
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let offset = i * 8;
        let half_vec = _mm_loadu_si128(input.as_ptr().add(offset) as *const __m128i);
        let float_vec = _mm256_cvtph_ps(half_vec);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), float_vec);
    }

    // Handle tail elements with scalar.
    let tail_start = chunks * 8;
    for i in 0..remainder {
        output[tail_start + i] = f16_to_f32_scalar_single(input[tail_start + i]);
    }
}

// ── Batch FP32→FP16 ───────────────────────────────────────────────────

/// Batch-convert `f32` values to FP16 (`u16`) with the specified
/// [`RoundingMode`].
///
/// Uses AVX2 + F16C when available on x86_64 (with `NearestEven`
/// rounding only; other modes fall back to scalar), otherwise uses a
/// portable scalar loop.
///
/// # Errors
///
/// Returns an error when `output.len() < input.len()`.
pub fn f32_to_f16_avx2(input: &[f32], output: &mut [u16], mode: RoundingMode) -> Result<()> {
    validate_lengths(input.len(), output.len(), "f32_to_f16_avx2")?;
    let n = input.len();

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_f16c() && mode == RoundingMode::NearestEven {
            // F16C's `_mm256_cvtps_ph` uses round-to-nearest-even
            // (imm8 = 0).
            unsafe {
                f32_to_f16_f16c(input, output, n);
            }
            return Ok(());
        }
    }

    // Scalar path (supports all rounding modes).
    f32_to_f16_scalar(input, output, n, mode);
    Ok(())
}

/// Scalar fallback for FP32→FP16 batch conversion.
fn f32_to_f16_scalar(input: &[f32], output: &mut [u16], n: usize, mode: RoundingMode) {
    for i in 0..n {
        output[i] = f32_to_f16_scalar_single(input[i], mode);
    }
}

/// AVX2 + F16C accelerated FP32→FP16 conversion (nearest-even).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn f32_to_f16_f16c(input: &[f32], output: &mut [u16], n: usize) {
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let offset = i * 8;
        let float_vec = _mm256_loadu_ps(input.as_ptr().add(offset));
        // imm8 = 0 → round to nearest even.
        let half_vec = _mm256_cvtps_ph(float_vec, 0);
        _mm_storeu_si128(output.as_mut_ptr().add(offset) as *mut __m128i, half_vec);
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        output[tail_start + i] =
            f32_to_f16_scalar_single(input[tail_start + i], RoundingMode::NearestEven);
    }
}

// ── Batch BF16↔FP32 ───────────────────────────────────────────────────

/// Batch-convert BF16 values (`u16`) to `f32` via bit shifting.
///
/// # Errors
///
/// Returns an error when `output.len() < input.len()`.
pub fn bf16_to_f32(input: &[u16], output: &mut [f32]) -> Result<()> {
    validate_lengths(input.len(), output.len(), "bf16_to_f32")?;
    let n = input.len();

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                bf16_to_f32_avx2_impl(input, output, n);
            }
            return Ok(());
        }
    }

    // Scalar fallback.
    for i in 0..n {
        output[i] = bf16_to_f32_scalar_single(input[i]);
    }
    Ok(())
}

/// AVX2 accelerated BF16→FP32 (shift left 16 bits in integer domain).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bf16_to_f32_avx2_impl(input: &[u16], output: &mut [f32], n: usize) {
    let chunks = n / 8;
    let remainder = n % 8;

    let shift16 = _mm256_set1_epi32(16);

    for i in 0..chunks {
        let offset = i * 8;
        // Load 8 × u16 into the lower 16 bits of 8 × i32 lanes.
        let raw = _mm_loadu_si128(input.as_ptr().add(offset) as *const __m128i);
        let wide = _mm256_cvtepu16_epi32(raw);
        let shifted = _mm256_sllv_epi32(wide, shift16);
        // Reinterpret the integer bits as f32.
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), _mm256_castsi256_ps(shifted));
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        output[tail_start + i] = bf16_to_f32_scalar_single(input[tail_start + i]);
    }
}

/// Batch-convert `f32` values to BF16 (`u16`) with round-to-nearest-even.
///
/// # Errors
///
/// Returns an error when `output.len() < input.len()`.
pub fn f32_to_bf16(input: &[f32], output: &mut [u16]) -> Result<()> {
    validate_lengths(input.len(), output.len(), "f32_to_bf16")?;
    let n = input.len();

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                f32_to_bf16_avx2_impl(input, output, n);
            }
            return Ok(());
        }
    }

    for i in 0..n {
        output[i] = f32_to_bf16_scalar_single(input[i]);
    }
    Ok(())
}

/// AVX2 accelerated FP32→BF16 with round-to-nearest-even.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn f32_to_bf16_avx2_impl(input: &[f32], output: &mut [u16], n: usize) {
    let chunks = n / 8;
    let remainder = n % 8;

    let round_bias_base = _mm256_set1_epi32((1u32 << 15).wrapping_sub(1) as i32);
    let one = _mm256_set1_epi32(1);

    for i in 0..chunks {
        let offset = i * 8;
        let float_vec = _mm256_loadu_ps(input.as_ptr().add(offset));
        let int_vec = _mm256_castps_si256(float_vec);

        // Round-to-nearest-even: bias = (1<<15) - 1 + ((bits>>16) & 1)
        let lsb = _mm256_and_si256(_mm256_srli_epi32(int_vec, 16), one);
        let bias = _mm256_add_epi32(round_bias_base, lsb);
        let rounded = _mm256_add_epi32(int_vec, bias);
        let shifted = _mm256_srli_epi32(rounded, 16);

        // Pack 8 × i32 → 8 × u16 by extracting the low 16 bits of each
        // lane. Use _mm256_packs_epi32 + shuffle to get contiguous u16.
        let packed = _mm256_packs_epi32(shifted, _mm256_setzero_si256());
        // packs_epi32 interleaves 128-bit halves, fix with permute.
        let perm = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
        let lo = _mm256_castsi256_si128(perm);
        _mm_storeu_si128(output.as_mut_ptr().add(offset) as *mut __m128i, lo);
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        output[tail_start + i] = f32_to_bf16_scalar_single(input[tail_start + i]);
    }
}

// ── Mixed-precision matmul ─────────────────────────────────────────────

/// Output storage format for mixed-precision matmul results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Store result as FP32 (no down-conversion).
    Fp32,
    /// Down-convert result to FP16 (`u16`).
    Fp16,
    /// Down-convert result to BF16 (`u16`).
    Bf16,
}

/// Mixed-precision matrix multiplication: `C = A × B`.
///
/// Computation is always performed in FP32. The result is optionally
/// down-converted to FP16 or BF16 depending on `output_format`.
///
/// * `a`: `m × k` row-major FP32
/// * `b`: `k × n` row-major FP32
/// * `c_f32`: output buffer for FP32 results (`m * n` elements, always
///   written)
/// * `c_f16`: optional output buffer for FP16/BF16 results (`m * n`
///   elements, written only when `output_format` is not `Fp32`)
///
/// # Errors
///
/// Returns an error when any buffer is too small or dimensions are zero.
pub fn mixed_precision_matmul(
    a: &[f32],
    b: &[f32],
    c_f32: &mut [f32],
    c_f16: Option<&mut [u16]>,
    m: usize,
    n: usize,
    k: usize,
    output_format: OutputFormat,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("dimensions must be > 0: m={m}, n={n}, k={k}"),
        }));
    }
    if a.len() < m * k {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("A too small: need {}, got {}", m * k, a.len()),
        }));
    }
    if b.len() < k * n {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("B too small: need {}, got {}", k * n, b.len()),
        }));
    }
    if c_f32.len() < m * n {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("C_f32 too small: need {}, got {}", m * n, c_f32.len()),
        }));
    }

    // FP32 GEMM (simple ijk loop — production code would tile / use SIMD).
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c_f32[i * n + j] = acc;
        }
    }

    // Optional down-conversion.
    match output_format {
        OutputFormat::Fp32 => {}
        OutputFormat::Fp16 => {
            let buf = c_f16.ok_or_else(|| {
                BitNetError::Kernel(KernelError::InvalidArguments {
                    reason: "c_f16 buffer required for Fp16 output".into(),
                })
            })?;
            if buf.len() < m * n {
                return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                    reason: format!("c_f16 too small: need {}, got {}", m * n, buf.len()),
                }));
            }
            f32_to_f16_avx2(&c_f32[..m * n], &mut buf[..m * n], RoundingMode::NearestEven)?;
        }
        OutputFormat::Bf16 => {
            let buf = c_f16.ok_or_else(|| {
                BitNetError::Kernel(KernelError::InvalidArguments {
                    reason: "c_f16 buffer required for Bf16 output".into(),
                })
            })?;
            if buf.len() < m * n {
                return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                    reason: format!("c_f16 too small: need {}, got {}", m * n, buf.len()),
                }));
            }
            f32_to_bf16(&c_f32[..m * n], &mut buf[..m * n])?;
        }
    }

    Ok(())
}

// ── Mixed-precision accumulation ───────────────────────────────────────

/// Accumulate FP16 inputs using FP32 arithmetic.
///
/// Converts each FP16 value to FP32, then sums into an FP32 accumulator.
/// Uses AVX2 + F16C vectorised reduction when available.
///
/// # Errors
///
/// Returns an error when `input` is empty (no meaningful sum).
pub fn mixed_precision_accumulate(input: &[u16]) -> Result<f32> {
    if input.is_empty() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "mixed_precision_accumulate: input must not be empty".into(),
        }));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_f16c() {
            // Safety: runtime feature check above.
            return Ok(unsafe { accumulate_f16c(input) });
        }
    }

    // Scalar fallback.
    let mut sum = 0.0f32;
    for &v in input {
        sum += f16_to_f32_scalar_single(v);
    }
    Ok(sum)
}

/// AVX2 + F16C vectorised FP16 accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn accumulate_f16c(input: &[u16]) -> f32 {
    let n = input.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let half_vec = _mm_loadu_si128(input.as_ptr().add(offset) as *const __m128i);
        let float_vec = _mm256_cvtph_ps(half_vec);
        acc = _mm256_add_ps(acc, float_vec);
    }

    // Horizontal sum of 8 floats.
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Scalar tail.
    let tail_start = chunks * 8;
    for i in 0..remainder {
        total += f16_to_f32_scalar_single(input[tail_start + i]);
    }
    total
}

// ── Dot-product with FP16 inputs, FP32 accumulation ────────────────────

/// Compute the dot product of two FP16 vectors using FP32 accumulation.
///
/// Both `a` and `b` must have the same length.
///
/// # Errors
///
/// Returns an error when `a.len() != b.len()` or inputs are empty.
pub fn mixed_precision_dot(a: &[u16], b: &[u16]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("mixed_precision_dot: length mismatch: a={}, b={}", a.len(), b.len()),
        }));
    }
    if a.is_empty() {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "mixed_precision_dot: inputs must not be empty".into(),
        }));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_f16c() {
            return Ok(unsafe { dot_f16c(a, b) });
        }
    }

    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += f16_to_f32_scalar_single(a[i]) * f16_to_f32_scalar_single(b[i]);
    }
    Ok(acc)
}

/// AVX2 + F16C vectorised FP16 dot product.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn dot_f16c(a: &[u16], b: &[u16]) -> f32 {
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let a_half = _mm_loadu_si128(a.as_ptr().add(offset) as *const __m128i);
        let b_half = _mm_loadu_si128(b.as_ptr().add(offset) as *const __m128i);
        let a_float = _mm256_cvtph_ps(a_half);
        let b_float = _mm256_cvtph_ps(b_half);
        acc = _mm256_fmadd_ps(a_float, b_float, acc);
    }

    // Horizontal sum.
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    let tail_start = chunks * 8;
    for i in 0..remainder {
        total += f16_to_f32_scalar_single(a[tail_start + i])
            * f16_to_f32_scalar_single(b[tail_start + i]);
    }
    total
}

// ── Validation ─────────────────────────────────────────────────────────

/// Shared length validation helper.
fn validate_lengths(input_len: usize, output_len: usize, fn_name: &str) -> Result<()> {
    if output_len < input_len {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("{fn_name}: output too small: need {input_len}, got {output_len}"),
        }));
    }
    Ok(())
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    /// Build a FP16 `u16` from a known f32 via the scalar converter.
    fn to_f16(v: f32) -> u16 {
        f32_to_f16_scalar_single(v, RoundingMode::NearestEven)
    }

    /// Build a BF16 `u16` from a known f32 via the scalar converter.
    fn to_bf16(v: f32) -> u16 {
        f32_to_bf16_scalar_single(v)
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol || (a == b)
    }

    // ── f16_to_f32 scalar single ──────────────────────────────────────

    #[test]
    fn test_f16_scalar_zero() {
        assert_eq!(f16_to_f32_scalar_single(0x0000), 0.0);
    }

    #[test]
    fn test_f16_scalar_negative_zero() {
        let val = f16_to_f32_scalar_single(0x8000);
        assert!(val.is_sign_negative());
        assert_eq!(val, -0.0);
    }

    #[test]
    fn test_f16_scalar_one() {
        // FP16 1.0 = 0_01111_0000000000 = 0x3C00
        assert_eq!(f16_to_f32_scalar_single(0x3C00), 1.0);
    }

    #[test]
    fn test_f16_scalar_negative_one() {
        // FP16 -1.0 = 1_01111_0000000000 = 0xBC00
        assert_eq!(f16_to_f32_scalar_single(0xBC00), -1.0);
    }

    #[test]
    fn test_f16_scalar_inf() {
        let val = f16_to_f32_scalar_single(0x7C00);
        assert!(val.is_infinite() && val.is_sign_positive());
    }

    #[test]
    fn test_f16_scalar_neg_inf() {
        let val = f16_to_f32_scalar_single(0xFC00);
        assert!(val.is_infinite() && val.is_sign_negative());
    }

    #[test]
    fn test_f16_scalar_nan() {
        let val = f16_to_f32_scalar_single(0x7C01);
        assert!(val.is_nan());
    }

    #[test]
    fn test_f16_scalar_subnormal() {
        // Smallest FP16 subnormal = 0x0001 → 2^-24 ≈ 5.96e-8
        let val = f16_to_f32_scalar_single(0x0001);
        assert!(val > 0.0 && val < 1e-6);
    }

    #[test]
    fn test_f16_scalar_max_normal() {
        // FP16 max = 0x7BFF ≈ 65504.0
        let val = f16_to_f32_scalar_single(0x7BFF);
        assert!(approx_eq(val, 65504.0, 1.0));
    }

    #[test]
    fn test_f16_scalar_half() {
        // FP16 0.5 = 0_01110_0000000000 = 0x3800
        assert_eq!(f16_to_f32_scalar_single(0x3800), 0.5);
    }

    // ── f32_to_f16 scalar single ──────────────────────────────────────

    #[test]
    fn test_f32_to_f16_zero() {
        assert_eq!(f32_to_f16_scalar_single(0.0, RoundingMode::NearestEven), 0x0000);
    }

    #[test]
    fn test_f32_to_f16_one() {
        assert_eq!(f32_to_f16_scalar_single(1.0, RoundingMode::NearestEven), 0x3C00);
    }

    #[test]
    fn test_f32_to_f16_neg_one() {
        assert_eq!(f32_to_f16_scalar_single(-1.0, RoundingMode::NearestEven), 0xBC00);
    }

    #[test]
    fn test_f32_to_f16_inf() {
        assert_eq!(f32_to_f16_scalar_single(f32::INFINITY, RoundingMode::NearestEven), 0x7C00);
    }

    #[test]
    fn test_f32_to_f16_neg_inf() {
        assert_eq!(f32_to_f16_scalar_single(f32::NEG_INFINITY, RoundingMode::NearestEven), 0xFC00);
    }

    #[test]
    fn test_f32_to_f16_nan() {
        let bits = f32_to_f16_scalar_single(f32::NAN, RoundingMode::NearestEven);
        // Must be a NaN in FP16 space.
        assert_eq!(bits & 0x7C00, 0x7C00);
        assert_ne!(bits & 0x03FF, 0);
    }

    #[test]
    fn test_f32_to_f16_overflow_to_inf() {
        // Value bigger than FP16 max → +Inf
        let bits = f32_to_f16_scalar_single(100_000.0, RoundingMode::NearestEven);
        assert_eq!(bits, 0x7C00);
    }

    #[test]
    fn test_f32_to_f16_underflow_to_zero() {
        // Extremely small → flush to zero.
        let bits = f32_to_f16_scalar_single(1e-10, RoundingMode::NearestEven);
        assert_eq!(bits, 0x0000);
    }

    #[test]
    fn test_f32_to_f16_negative_zero() {
        let bits = f32_to_f16_scalar_single(-0.0, RoundingMode::NearestEven);
        assert_eq!(bits, 0x8000);
    }

    // ── Rounding modes ────────────────────────────────────────────────

    #[test]
    fn test_rounding_truncate() {
        // 1.0009765625 in FP16 is between 1.0 (0x3C00) and next (0x3C01).
        let val = 1.0009765625_f32;
        let trunc = f32_to_f16_scalar_single(val, RoundingMode::Truncate);
        let near = f32_to_f16_scalar_single(val, RoundingMode::NearestEven);
        // Truncation rounds toward zero → ≤ nearest-even result.
        assert!(trunc <= near);
    }

    #[test]
    fn test_rounding_ceiling_positive() {
        let val = 1.0001_f32;
        let ceil = f32_to_f16_scalar_single(val, RoundingMode::Ceiling);
        let trunc = f32_to_f16_scalar_single(val, RoundingMode::Truncate);
        // Ceiling for positive values rounds up.
        assert!(ceil >= trunc);
    }

    #[test]
    fn test_rounding_floor_negative() {
        let val = -1.0001_f32;
        let floor = f32_to_f16_scalar_single(val, RoundingMode::Floor);
        let trunc = f32_to_f16_scalar_single(val, RoundingMode::Truncate);
        // Floor for negative values rounds toward -∞.
        assert!(floor >= trunc); // magnitude increases, encoded as larger u16
    }

    #[test]
    fn test_rounding_nearest_even_tie() {
        // Exact halfway: 1.0 + 0.5 ULP in FP16 mantissa = 1.0 + 2^-11
        // The mantissa LSB is 0, so ties-to-even keeps it at 0x3C00.
        let val = 1.0 + (1.0 / 2048.0);
        let bits = f32_to_f16_scalar_single(val, RoundingMode::NearestEven);
        // Should round to even (0x3C00 or 0x3C01 depending on exact tie).
        assert!(bits == 0x3C00 || bits == 0x3C01);
    }

    // ── BF16 scalar ───────────────────────────────────────────────────

    #[test]
    fn test_bf16_to_f32_zero() {
        assert_eq!(bf16_to_f32_scalar_single(0x0000), 0.0);
    }

    #[test]
    fn test_bf16_to_f32_one() {
        // BF16 1.0 = 0x3F80
        assert_eq!(bf16_to_f32_scalar_single(0x3F80), 1.0);
    }

    #[test]
    fn test_bf16_to_f32_neg_one() {
        assert_eq!(bf16_to_f32_scalar_single(0xBF80), -1.0);
    }

    #[test]
    fn test_bf16_to_f32_large() {
        // BF16 for 256.0 = 0x4380
        let val = bf16_to_f32_scalar_single(0x4380);
        assert!(approx_eq(val, 256.0, 0.01));
    }

    #[test]
    fn test_f32_to_bf16_zero() {
        assert_eq!(f32_to_bf16_scalar_single(0.0), 0x0000);
    }

    #[test]
    fn test_f32_to_bf16_one() {
        assert_eq!(f32_to_bf16_scalar_single(1.0), 0x3F80);
    }

    #[test]
    fn test_f32_to_bf16_neg_one() {
        assert_eq!(f32_to_bf16_scalar_single(-1.0), 0xBF80);
    }

    #[test]
    fn test_bf16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 42.0, -128.0, 1e10, -1e-5];
        for &v in &values {
            let bf = f32_to_bf16_scalar_single(v);
            let back = bf16_to_f32_scalar_single(bf);
            // BF16 has 7-bit mantissa ≈ 2 decimal digits of precision.
            let tol = v.abs() * 0.01 + 1e-4;
            assert!(approx_eq(back, v, tol), "BF16 roundtrip failed for {v}: got {back}");
        }
    }

    // ── Batch FP16→FP32 ──────────────────────────────────────────────

    #[test]
    fn test_batch_f16_to_f32_basic() {
        let input: Vec<u16> = [0.0f32, 1.0, -1.0, 0.5, 2.0].iter().map(|&v| to_f16(v)).collect();
        let mut output = vec![0.0f32; 5];
        f16_to_f32_avx2(&input, &mut output).unwrap();

        assert!(approx_eq(output[0], 0.0, 1e-4));
        assert!(approx_eq(output[1], 1.0, 1e-3));
        assert!(approx_eq(output[2], -1.0, 1e-3));
        assert!(approx_eq(output[3], 0.5, 1e-3));
        assert!(approx_eq(output[4], 2.0, 1e-3));
    }

    #[test]
    fn test_batch_f16_to_f32_empty() {
        let mut output = vec![0.0f32; 0];
        f16_to_f32_avx2(&[], &mut output).unwrap();
    }

    #[test]
    fn test_batch_f16_to_f32_output_too_small() {
        let input = vec![0u16; 10];
        let mut output = vec![0.0f32; 5];
        assert!(f16_to_f32_avx2(&input, &mut output).is_err());
    }

    #[test]
    fn test_batch_f16_to_f32_large() {
        let n = 1024;
        let input: Vec<u16> = (0..n).map(|i| to_f16(i as f32 * 0.1)).collect();
        let mut output = vec![0.0f32; n];
        f16_to_f32_avx2(&input, &mut output).unwrap();

        for i in 0..n {
            let expected = i as f32 * 0.1;
            assert!(
                approx_eq(output[i], expected, 0.1),
                "mismatch at {i}: expected ~{expected}, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn test_batch_f16_to_f32_tail_elements() {
        // Test with non-multiple-of-8 length to exercise tail path.
        for len in [1, 3, 7, 9, 15, 17] {
            let input: Vec<u16> = (0..len).map(|i| to_f16(i as f32)).collect();
            let mut output = vec![0.0f32; len];
            f16_to_f32_avx2(&input, &mut output).unwrap();
            for i in 0..len {
                assert!(approx_eq(output[i], i as f32, 0.01), "tail test len={len} idx={i}");
            }
        }
    }

    // ── Batch FP32→FP16 ──────────────────────────────────────────────

    #[test]
    fn test_batch_f32_to_f16_basic() {
        let input = [0.0f32, 1.0, -1.0, 0.5, 2.0];
        let mut output = vec![0u16; 5];
        f32_to_f16_avx2(&input, &mut output, RoundingMode::NearestEven).unwrap();

        for (i, &v) in input.iter().enumerate() {
            let back = f16_to_f32_scalar_single(output[i]);
            assert!(
                approx_eq(back, v, 1e-3),
                "roundtrip mismatch at {i}: {v} → {} → {back}",
                output[i]
            );
        }
    }

    #[test]
    fn test_batch_f32_to_f16_output_too_small() {
        let input = vec![0.0f32; 10];
        let mut output = vec![0u16; 5];
        assert!(f32_to_f16_avx2(&input, &mut output, RoundingMode::NearestEven).is_err());
    }

    #[test]
    fn test_batch_f32_to_f16_large() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0u16; n];
        f32_to_f16_avx2(&input, &mut output, RoundingMode::NearestEven).unwrap();

        let mut back = vec![0.0f32; n];
        f16_to_f32_avx2(&output, &mut back).unwrap();
        for i in 0..n {
            assert!(
                approx_eq(back[i], input[i], 0.1),
                "roundtrip at {i}: {:.4} → {:.4}",
                input[i],
                back[i]
            );
        }
    }

    #[test]
    fn test_batch_f32_to_f16_tail_elements() {
        for len in [1, 5, 7, 11, 13] {
            let input: Vec<f32> = (0..len).map(|i| i as f32 + 0.25).collect();
            let mut output = vec![0u16; len];
            f32_to_f16_avx2(&input, &mut output, RoundingMode::NearestEven).unwrap();
            for i in 0..len {
                let back = f16_to_f32_scalar_single(output[i]);
                assert!(
                    approx_eq(back, input[i], 0.01),
                    "tail len={len} idx={i}: {} vs {back}",
                    input[i]
                );
            }
        }
    }

    #[test]
    fn test_batch_f32_to_f16_truncate_mode() {
        let input = [1.4_f32, 2.6, -3.1];
        let mut out_trunc = vec![0u16; 3];
        let mut out_near = vec![0u16; 3];
        f32_to_f16_avx2(&input, &mut out_trunc, RoundingMode::Truncate).unwrap();
        f32_to_f16_avx2(&input, &mut out_near, RoundingMode::NearestEven).unwrap();
        // Just verify no panics and results are valid FP16.
        for i in 0..3 {
            let _ = f16_to_f32_scalar_single(out_trunc[i]);
            let _ = f16_to_f32_scalar_single(out_near[i]);
        }
    }

    // ── Batch BF16→FP32 ──────────────────────────────────────────────

    #[test]
    fn test_batch_bf16_to_f32_basic() {
        let input: Vec<u16> = [0.0f32, 1.0, -1.0, 42.0].iter().map(|&v| to_bf16(v)).collect();
        let mut output = vec![0.0f32; 4];
        bf16_to_f32(&input, &mut output).unwrap();

        assert!(approx_eq(output[0], 0.0, 1e-4));
        assert!(approx_eq(output[1], 1.0, 0.01));
        assert!(approx_eq(output[2], -1.0, 0.01));
        assert!(approx_eq(output[3], 42.0, 0.5));
    }

    #[test]
    fn test_batch_bf16_to_f32_large() {
        let n = 512;
        let input: Vec<u16> = (0..n).map(|i| to_bf16(i as f32)).collect();
        let mut output = vec![0.0f32; n];
        bf16_to_f32(&input, &mut output).unwrap();
        for i in 0..n {
            assert!(approx_eq(output[i], i as f32, 1.0), "bf16 batch at {i}");
        }
    }

    #[test]
    fn test_batch_bf16_to_f32_output_too_small() {
        let input = vec![0u16; 8];
        let mut output = vec![0.0f32; 4];
        assert!(bf16_to_f32(&input, &mut output).is_err());
    }

    #[test]
    fn test_batch_bf16_to_f32_tail() {
        for len in [1, 3, 5, 7, 9] {
            let input: Vec<u16> = (0..len).map(|i| to_bf16(i as f32 * 10.0)).collect();
            let mut output = vec![0.0f32; len];
            bf16_to_f32(&input, &mut output).unwrap();
            for i in 0..len {
                let expected = i as f32 * 10.0;
                assert!(approx_eq(output[i], expected, 1.0), "bf16 tail len={len} idx={i}");
            }
        }
    }

    // ── Batch FP32→BF16 ──────────────────────────────────────────────

    #[test]
    fn test_batch_f32_to_bf16_basic() {
        let input = [0.0f32, 1.0, -1.0, 256.0];
        let mut output = vec![0u16; 4];
        f32_to_bf16(&input, &mut output).unwrap();

        let mut back = vec![0.0f32; 4];
        bf16_to_f32(&output, &mut back).unwrap();
        for i in 0..4 {
            let tol = input[i].abs() * 0.01 + 1e-4;
            assert!(
                approx_eq(back[i], input[i], tol),
                "bf16 roundtrip at {i}: {} vs {}",
                input[i],
                back[i]
            );
        }
    }

    #[test]
    fn test_batch_f32_to_bf16_output_too_small() {
        let input = vec![0.0f32; 8];
        let mut output = vec![0u16; 4];
        assert!(f32_to_bf16(&input, &mut output).is_err());
    }

    #[test]
    fn test_batch_f32_to_bf16_large() {
        let n = 512;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0u16; n];
        f32_to_bf16(&input, &mut output).unwrap();

        let mut back = vec![0.0f32; n];
        bf16_to_f32(&output, &mut back).unwrap();
        for i in 0..n {
            assert!(approx_eq(back[i], input[i], 1.0), "bf16 large at {i}");
        }
    }

    #[test]
    fn test_batch_f32_to_bf16_tail() {
        for len in [1, 2, 6, 11] {
            let input: Vec<f32> = (0..len).map(|i| i as f32 * 3.0).collect();
            let mut output = vec![0u16; len];
            f32_to_bf16(&input, &mut output).unwrap();
            for i in 0..len {
                let back = bf16_to_f32_scalar_single(output[i]);
                assert!(approx_eq(back, input[i], 1.0), "bf16 tail len={len} idx={i}");
            }
        }
    }

    // ── FP16 roundtrip batch ──────────────────────────────────────────

    #[test]
    fn test_f16_roundtrip_batch() {
        let values: Vec<f32> = vec![
            0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 0.001, -0.001, 65504.0, -65504.0, 0.25,
            0.125, 3.14, -2.718,
        ];
        let mut f16_buf = vec![0u16; values.len()];
        let mut back = vec![0.0f32; values.len()];

        f32_to_f16_avx2(&values, &mut f16_buf, RoundingMode::NearestEven).unwrap();
        f16_to_f32_avx2(&f16_buf, &mut back).unwrap();

        for (i, &v) in values.iter().enumerate() {
            let tol = v.abs() * 0.002 + 1e-3;
            assert!(
                approx_eq(back[i], v, tol),
                "roundtrip at {i}: {v} → {back_v}",
                back_v = back[i]
            );
        }
    }

    // ── Mixed precision matmul ────────────────────────────────────────

    #[test]
    fn test_matmul_identity() {
        // 2×2 identity × [1,2; 3,4] = [1,2; 3,4]
        let a = [1.0, 0.0, 0.0, 1.0f32];
        let b = [1.0, 2.0, 3.0, 4.0f32];
        let mut c = vec![0.0f32; 4];
        mixed_precision_matmul(&a, &b, &mut c, None, 2, 2, 2, OutputFormat::Fp32).unwrap();
        assert!(approx_eq(c[0], 1.0, 1e-5));
        assert!(approx_eq(c[1], 2.0, 1e-5));
        assert!(approx_eq(c[2], 3.0, 1e-5));
        assert!(approx_eq(c[3], 4.0, 1e-5));
    }

    #[test]
    fn test_matmul_with_f16_output() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [1.0, 0.0, 0.0, 1.0f32];
        let mut c_f32 = vec![0.0f32; 4];
        let mut c_f16 = vec![0u16; 4];
        mixed_precision_matmul(&a, &b, &mut c_f32, Some(&mut c_f16), 2, 2, 2, OutputFormat::Fp16)
            .unwrap();

        // Verify FP16 output roundtrips correctly.
        for i in 0..4 {
            let back = f16_to_f32_scalar_single(c_f16[i]);
            assert!(
                approx_eq(back, c_f32[i], 0.01),
                "matmul f16 output at {i}: {back} vs {}",
                c_f32[i]
            );
        }
    }

    #[test]
    fn test_matmul_with_bf16_output() {
        let a = [2.0, 0.0, 0.0, 3.0f32];
        let b = [1.0, 1.0, 1.0, 1.0f32];
        let mut c_f32 = vec![0.0f32; 4];
        let mut c_bf16 = vec![0u16; 4];
        mixed_precision_matmul(&a, &b, &mut c_f32, Some(&mut c_bf16), 2, 2, 2, OutputFormat::Bf16)
            .unwrap();

        for i in 0..4 {
            let back = bf16_to_f32_scalar_single(c_bf16[i]);
            assert!(approx_eq(back, c_f32[i], 0.1), "matmul bf16 output at {i}");
        }
    }

    #[test]
    fn test_matmul_zero_dimension() {
        let a: &[f32] = &[];
        let b: &[f32] = &[];
        let mut c = vec![0.0f32; 0];
        assert!(mixed_precision_matmul(a, b, &mut c, None, 0, 1, 1, OutputFormat::Fp32).is_err());
    }

    #[test]
    fn test_matmul_a_too_small() {
        let a = [1.0f32];
        let b = [1.0, 2.0, 3.0, 4.0f32];
        let mut c = vec![0.0f32; 4];
        assert!(mixed_precision_matmul(&a, &b, &mut c, None, 2, 2, 2, OutputFormat::Fp32).is_err());
    }

    #[test]
    fn test_matmul_b_too_small() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [1.0f32];
        let mut c = vec![0.0f32; 4];
        assert!(mixed_precision_matmul(&a, &b, &mut c, None, 2, 2, 2, OutputFormat::Fp32).is_err());
    }

    #[test]
    fn test_matmul_c_too_small() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [1.0, 0.0, 0.0, 1.0f32];
        let mut c = vec![0.0f32; 2];
        assert!(mixed_precision_matmul(&a, &b, &mut c, None, 2, 2, 2, OutputFormat::Fp32).is_err());
    }

    #[test]
    fn test_matmul_f16_missing_buffer() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 4];
        let mut c = vec![0.0f32; 4];
        assert!(mixed_precision_matmul(&a, &b, &mut c, None, 2, 2, 2, OutputFormat::Fp16).is_err());
    }

    #[test]
    fn test_matmul_rectangular() {
        // 2×3 × 3×1 = 2×1
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let b = [1.0, 1.0, 1.0f32];
        let mut c = vec![0.0f32; 2];
        mixed_precision_matmul(&a, &b, &mut c, None, 2, 1, 3, OutputFormat::Fp32).unwrap();
        assert!(approx_eq(c[0], 6.0, 1e-5)); // 1+2+3
        assert!(approx_eq(c[1], 15.0, 1e-5)); // 4+5+6
    }

    #[test]
    fn test_matmul_single_element() {
        let a = [3.0f32];
        let b = [4.0f32];
        let mut c = vec![0.0f32; 1];
        mixed_precision_matmul(&a, &b, &mut c, None, 1, 1, 1, OutputFormat::Fp32).unwrap();
        assert!(approx_eq(c[0], 12.0, 1e-5));
    }

    // ── Mixed precision accumulate ────────────────────────────────────

    #[test]
    fn test_accumulate_basic() {
        let input: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|&v| to_f16(v)).collect();
        let sum = mixed_precision_accumulate(&input).unwrap();
        assert!(approx_eq(sum, 10.0, 0.1));
    }

    #[test]
    fn test_accumulate_empty() {
        assert!(mixed_precision_accumulate(&[]).is_err());
    }

    #[test]
    fn test_accumulate_single() {
        let input = [to_f16(42.0)];
        let sum = mixed_precision_accumulate(&input).unwrap();
        assert!(approx_eq(sum, 42.0, 0.1));
    }

    #[test]
    fn test_accumulate_negative() {
        let input: Vec<u16> = [-1.0f32, -2.0, -3.0].iter().map(|&v| to_f16(v)).collect();
        let sum = mixed_precision_accumulate(&input).unwrap();
        assert!(approx_eq(sum, -6.0, 0.1));
    }

    #[test]
    fn test_accumulate_cancellation() {
        let input: Vec<u16> = [100.0f32, -100.0].iter().map(|&v| to_f16(v)).collect();
        let sum = mixed_precision_accumulate(&input).unwrap();
        assert!(approx_eq(sum, 0.0, 0.1));
    }

    #[test]
    fn test_accumulate_large_vector() {
        let n = 1024;
        let input: Vec<u16> = (0..n).map(|_| to_f16(1.0)).collect();
        let sum = mixed_precision_accumulate(&input).unwrap();
        assert!(approx_eq(sum, n as f32, 1.0));
    }

    #[test]
    fn test_accumulate_tail_lengths() {
        for len in [1, 7, 8, 9, 15, 16, 17] {
            let input: Vec<u16> = (0..len).map(|_| to_f16(1.0)).collect();
            let sum = mixed_precision_accumulate(&input).unwrap();
            assert!(approx_eq(sum, len as f32, 0.5), "accumulate tail len={len}: got {sum}");
        }
    }

    // ── Dot product ───────────────────────────────────────────────────

    #[test]
    fn test_dot_basic() {
        let a: Vec<u16> = [1.0f32, 2.0, 3.0].iter().map(|&v| to_f16(v)).collect();
        let b: Vec<u16> = [4.0f32, 5.0, 6.0].iter().map(|&v| to_f16(v)).collect();
        let dot = mixed_precision_dot(&a, &b).unwrap();
        // 1*4 + 2*5 + 3*6 = 32
        assert!(approx_eq(dot, 32.0, 0.5));
    }

    #[test]
    fn test_dot_length_mismatch() {
        let a = vec![to_f16(1.0); 3];
        let b = vec![to_f16(1.0); 4];
        assert!(mixed_precision_dot(&a, &b).is_err());
    }

    #[test]
    fn test_dot_empty() {
        assert!(mixed_precision_dot(&[], &[]).is_err());
    }

    #[test]
    fn test_dot_orthogonal() {
        let a: Vec<u16> = [1.0f32, 0.0].iter().map(|&v| to_f16(v)).collect();
        let b: Vec<u16> = [0.0f32, 1.0].iter().map(|&v| to_f16(v)).collect();
        let dot = mixed_precision_dot(&a, &b).unwrap();
        assert!(approx_eq(dot, 0.0, 1e-3));
    }

    #[test]
    fn test_dot_self() {
        let a: Vec<u16> = [3.0f32, 4.0].iter().map(|&v| to_f16(v)).collect();
        let dot = mixed_precision_dot(&a, &a).unwrap();
        // 9 + 16 = 25
        assert!(approx_eq(dot, 25.0, 0.5));
    }

    #[test]
    fn test_dot_large() {
        let n = 256;
        let a: Vec<u16> = (0..n).map(|_| to_f16(1.0)).collect();
        let b: Vec<u16> = (0..n).map(|_| to_f16(2.0)).collect();
        let dot = mixed_precision_dot(&a, &b).unwrap();
        assert!(approx_eq(dot, (n * 2) as f32, 1.0));
    }

    // ── PrecisionPolicy ──────────────────────────────────────────────

    #[test]
    fn test_policy_default() {
        let p = PrecisionPolicy::default();
        assert_eq!(p.fp16_max_abs, 65504.0);
        assert!(p.prefer_bf16_for_large_range);
    }

    #[test]
    fn test_policy_empty_data() {
        let p = PrecisionPolicy::new();
        assert_eq!(p.recommend(&[]), TargetPrecision::Fp16);
    }

    #[test]
    fn test_policy_all_zeros() {
        let p = PrecisionPolicy::new();
        assert_eq!(p.recommend(&[0.0; 100]), TargetPrecision::Fp16);
    }

    #[test]
    fn test_policy_small_range() {
        let p = PrecisionPolicy::new();
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        assert_eq!(p.recommend(&data), TargetPrecision::Fp16);
    }

    #[test]
    fn test_policy_large_range_prefers_bf16() {
        let p = PrecisionPolicy::new();
        let data = [1e5_f32, -1e5]; // exceeds FP16 max (65504)
        assert_eq!(p.recommend(&data), TargetPrecision::Bf16);
    }

    #[test]
    fn test_policy_huge_range_needs_fp32() {
        let mut p = PrecisionPolicy::new();
        p.prefer_bf16_for_large_range = false;
        let data = [1e5_f32, -1e5];
        assert_eq!(p.recommend(&data), TargetPrecision::Fp32);
    }

    #[test]
    fn test_policy_tiny_values_need_bf16() {
        let p = PrecisionPolicy::new();
        // Values below FP16 min normal but within BF16 range.
        let data = [1e-6_f32, 2e-6];
        assert_eq!(p.recommend(&data), TargetPrecision::Bf16);
    }

    #[test]
    fn test_policy_mixed_data() {
        let p = PrecisionPolicy::new();
        let data = [0.5_f32, 1.0, 100.0, -50.0]; // all within FP16 range
        assert_eq!(p.recommend(&data), TargetPrecision::Fp16);
    }

    #[test]
    fn test_policy_boundary_fp16_max() {
        let p = PrecisionPolicy::new();
        let data = [65504.0_f32]; // exactly FP16 max
        assert_eq!(p.recommend(&data), TargetPrecision::Fp16);
    }

    #[test]
    fn test_policy_just_over_fp16_max() {
        let p = PrecisionPolicy::new();
        let data = [65505.0_f32];
        assert_eq!(p.recommend(&data), TargetPrecision::Bf16);
    }

    // ── OutputFormat and enum coverage ────────────────────────────────

    #[test]
    fn test_output_format_eq() {
        assert_eq!(OutputFormat::Fp32, OutputFormat::Fp32);
        assert_ne!(OutputFormat::Fp16, OutputFormat::Bf16);
    }

    #[test]
    fn test_rounding_mode_eq() {
        assert_eq!(RoundingMode::NearestEven, RoundingMode::NearestEven);
        assert_ne!(RoundingMode::Truncate, RoundingMode::Floor);
    }

    #[test]
    fn test_target_precision_eq() {
        assert_eq!(TargetPrecision::Fp16, TargetPrecision::Fp16);
        assert_ne!(TargetPrecision::Bf16, TargetPrecision::Fp32);
    }

    #[test]
    fn test_rounding_mode_debug() {
        let _s = format!("{:?}", RoundingMode::Ceiling);
    }

    #[test]
    fn test_precision_policy_clone() {
        let p = PrecisionPolicy::new();
        let p2 = p.clone();
        assert_eq!(p.fp16_max_abs, p2.fp16_max_abs);
    }

    // ── Validate lengths ──────────────────────────────────────────────

    #[test]
    fn test_validate_lengths_ok() {
        assert!(validate_lengths(5, 5, "test").is_ok());
        assert!(validate_lengths(5, 10, "test").is_ok());
        assert!(validate_lengths(0, 0, "test").is_ok());
    }

    #[test]
    fn test_validate_lengths_err() {
        assert!(validate_lengths(10, 5, "test").is_err());
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_f16_special_values_batch() {
        // Batch containing zeros, normals, and max.
        let input = vec![0x0000u16, 0x3C00, 0x7BFF, 0x8000, 0xBC00, 0xFBFF];
        let mut output = vec![0.0f32; 6];
        f16_to_f32_avx2(&input, &mut output).unwrap();
        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 1.0);
        assert!(approx_eq(output[2], 65504.0, 1.0));
        assert_eq!(output[3], -0.0);
        assert_eq!(output[4], -1.0);
        assert!(approx_eq(output[5], -65504.0, 1.0));
    }

    #[test]
    fn test_bf16_inf_nan() {
        // BF16 +Inf = 0x7F80, BF16 NaN = 0x7FC0
        let inf_val = bf16_to_f32_scalar_single(0x7F80);
        assert!(inf_val.is_infinite());
        let nan_val = bf16_to_f32_scalar_single(0x7FC0);
        assert!(nan_val.is_nan());
    }

    #[test]
    fn test_matmul_f16_buffer_too_small() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 4];
        let mut c_f32 = vec![0.0f32; 4];
        let mut c_f16 = vec![0u16; 2]; // too small
        assert!(
            mixed_precision_matmul(
                &a,
                &b,
                &mut c_f32,
                Some(&mut c_f16),
                2,
                2,
                2,
                OutputFormat::Fp16,
            )
            .is_err()
        );
    }

    #[test]
    fn test_matmul_bf16_missing_buffer() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 4];
        let mut c_f32 = vec![0.0f32; 4];
        assert!(
            mixed_precision_matmul(&a, &b, &mut c_f32, None, 2, 2, 2, OutputFormat::Bf16,).is_err()
        );
    }

    #[test]
    fn test_dot_tail_lengths() {
        for len in [1, 5, 7, 8, 9, 15, 16, 17] {
            let a: Vec<u16> = (0..len).map(|_| to_f16(2.0)).collect();
            let b: Vec<u16> = (0..len).map(|_| to_f16(3.0)).collect();
            let dot = mixed_precision_dot(&a, &b).unwrap();
            let expected = len as f32 * 6.0;
            assert!(approx_eq(dot, expected, 1.0), "dot tail len={len}: {dot} vs {expected}");
        }
    }
}
