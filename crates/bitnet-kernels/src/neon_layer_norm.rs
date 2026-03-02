//! NEON-optimized LayerNorm and RMSNorm kernels with Welford's online algorithm.
//!
//! Provides:
//! - **LayerNorm**: `y = γ(x − μ) / √(σ² + ε) + β`
//! - **RMSNorm**: `y = γx / √(mean(x²) + ε)` (LLaMA-style)
//! - **Welford's algorithm**: Numerically stable single-pass mean/variance
//! - **Batch processing**: Normalizes multiple rows in one call
//! - **f16 precision**: Half-precision I/O with f32 internal compute
//!
//! On `aarch64` targets, NEON SIMD processes 4×f32 lanes at a time with
//! scalar tail handling. On other architectures, a scalar fallback provides
//! identical results.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Half-precision (f16) support ──────────────────────────────────

/// IEEE 754 half-precision float represented as raw bits.
///
/// Provides platform-independent f16 ↔ f32 conversion without external
/// dependencies. All arithmetic is performed in f32; this type is used
/// only for storage and I/O.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct F16(pub u16);

impl F16 {
    /// Convert an `f32` value to half-precision bits (truncation, no rounding).
    pub fn from_f32(val: f32) -> Self {
        Self(f32_to_f16_bits(val))
    }

    /// Widen to `f32` with exact conversion.
    pub fn to_f32(self) -> f32 {
        f16_bits_to_f32(self.0)
    }

    /// Positive zero.
    pub const ZERO: Self = Self(0);

    /// One (`1.0` in f16 = `0x3C00`).
    pub const ONE: Self = Self(0x3C00);
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x007F_FFFF;

    if exp == 255 {
        // Inf / NaN
        let nan_bit = if man != 0 { 0x0200 } else { 0 };
        return (sign | 0x7C00 | nan_bit) as u16;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        return (sign | 0x7C00) as u16; // overflow → ±Inf
    }
    if unbiased < -24 {
        return sign as u16; // underflow → ±0
    }
    if unbiased < -14 {
        // Subnormal f16
        let shift = -1 - unbiased;
        let man16 = ((man | 0x0080_0000) >> (shift + 13)) as u32;
        return (sign | man16) as u16;
    }

    let exp16 = ((unbiased + 15) as u32) << 10;
    let man16 = man >> 13;
    (sign | exp16 | man16) as u16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let man = (bits & 0x03FF) as u32;

    if exp == 0 {
        if man == 0 {
            return f32::from_bits(sign); // ±0
        }
        // Subnormal: normalize
        let mut e = 1u32;
        let mut m = man;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e += 1;
        }
        let exp32 = (127 - 15 + 1 - e) << 23;
        let man32 = (m & 0x03FF) << 13;
        return f32::from_bits(sign | exp32 | man32);
    }
    if exp == 31 {
        let nan_bits = if man != 0 { 0x007F_FFFF } else { 0 };
        return f32::from_bits(sign | 0x7F80_0000 | nan_bits);
    }

    let exp32 = (exp + 127 - 15) << 23;
    let man32 = man << 13;
    f32::from_bits(sign | exp32 | man32)
}

// ── Welford online accumulator ────────────────────────────────────

/// Welford's online accumulator for numerically stable mean/variance.
///
/// Operates in `f64` to minimise rounding error during accumulation.
/// Use [`WelfordAccumulator::merge`] to combine partial results from
/// parallel lanes (e.g. NEON).
#[derive(Clone, Copy, Debug)]
pub struct WelfordAccumulator {
    /// Number of samples observed.
    pub count: usize,
    /// Running mean.
    pub mean: f64,
    /// Sum of squared deviations from the running mean (M2).
    pub m2: f64,
}

impl WelfordAccumulator {
    /// Create an empty accumulator.
    pub fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0 }
    }

    /// Incorporate a single observation.
    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Merge two accumulators (parallel Welford combine).
    pub fn merge(a: &Self, b: &Self) -> Self {
        if a.count == 0 {
            return *b;
        }
        if b.count == 0 {
            return *a;
        }
        let count = a.count + b.count;
        let delta = b.mean - a.mean;
        let mean = a.mean + delta * (b.count as f64 / count as f64);
        let m2 = a.m2 + b.m2 + delta * delta * (a.count as f64 * b.count as f64 / count as f64);
        Self { count, mean, m2 }
    }

    /// Population variance (`M2 / count`).
    pub fn variance(&self) -> f64 {
        if self.count < 1 {
            return 0.0;
        }
        self.m2 / self.count as f64
    }

    /// Mean downcast to `f32`.
    pub fn mean_f32(&self) -> f32 {
        self.mean as f32
    }

    /// Population variance downcast to `f32`.
    pub fn variance_f32(&self) -> f32 {
        self.variance() as f32
    }
}

impl Default for WelfordAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute Welford mean/variance statistics for `data`.
pub fn welford_stats(data: &[f32]) -> WelfordAccumulator {
    let mut acc = WelfordAccumulator::new();
    for &x in data {
        acc.update(x as f64);
    }
    acc
}

// ── Public API: f32 ───────────────────────────────────────────────

/// LayerNorm: `y = γ(x − μ) / √(σ² + ε) + β`.
///
/// Dispatches to NEON on `aarch64`, scalar otherwise.
///
/// # Panics
///
/// Panics if slice lengths are inconsistent or `input` is empty.
pub fn layer_norm_f32(input: &[f32], output: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = input.len();
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");
    if n == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on AArch64 (ARMv8-A).
        unsafe {
            neon_layer_norm_impl(input, output, gamma, beta, eps);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mean: f32 = input.iter().sum::<f32>() / n as f32;
        let var: f32 = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..n {
            output[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
        }
    }
}

/// RMSNorm: `y = γx / √(mean(x²) + ε)` (LLaMA-style).
///
/// Dispatches to NEON on `aarch64`, scalar otherwise.
pub fn rms_norm_f32(input: &[f32], output: &mut [f32], gamma: &[f32], eps: f32) {
    let n = input.len();
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    if n == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe {
            neon_rms_norm_impl(input, output, gamma, eps);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mean_sq: f32 = input.iter().map(|&x| x * x).sum::<f32>() / n as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..n {
            output[i] = gamma[i] * input[i] * inv_rms;
        }
    }
}

/// LayerNorm using Welford's online algorithm for numerical stability.
///
/// Statistics are computed in `f64`; the normalize step dispatches to
/// NEON on `aarch64`.
pub fn layer_norm_welford_f32(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");
    if n == 0 {
        return;
    }

    let acc = welford_stats(input);
    let mean = acc.mean_f32();
    let inv_std = 1.0 / (acc.variance_f32() + eps).sqrt();

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe {
            neon_affine_transform(input, output, gamma, beta, mean, inv_std);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            output[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
        }
    }
}

/// RMSNorm using Welford's online algorithm for numerical stability.
pub fn rms_norm_welford_f32(input: &[f32], output: &mut [f32], gamma: &[f32], eps: f32) {
    let n = input.len();
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    if n == 0 {
        return;
    }

    let mut acc = WelfordAccumulator::new();
    for &x in input {
        acc.update((x * x) as f64);
    }
    let mean_sq = acc.mean_f32();
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe {
            neon_scale_transform(input, output, gamma, inv_rms);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            output[i] = gamma[i] * input[i] * inv_rms;
        }
    }
}

// ── Public API: batch ─────────────────────────────────────────────

/// Batch LayerNorm over a flat buffer of `batch_size × hidden_dim` elements.
pub fn batch_layer_norm_f32(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    hidden_dim: usize,
    eps: f32,
) {
    assert_eq!(input.len(), output.len(), "I/O length mismatch");
    assert!(hidden_dim > 0, "hidden_dim must be > 0");
    assert_eq!(input.len() % hidden_dim, 0, "input length not divisible by hidden_dim");
    assert_eq!(gamma.len(), hidden_dim, "gamma length mismatch");
    assert_eq!(beta.len(), hidden_dim, "beta length mismatch");

    let batch_size = input.len() / hidden_dim;
    for b in 0..batch_size {
        let s = b * hidden_dim;
        let e = s + hidden_dim;
        layer_norm_f32(&input[s..e], &mut output[s..e], gamma, beta, eps);
    }
}

/// Batch RMSNorm over a flat buffer of `batch_size × hidden_dim` elements.
pub fn batch_rms_norm_f32(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    hidden_dim: usize,
    eps: f32,
) {
    assert_eq!(input.len(), output.len(), "I/O length mismatch");
    assert!(hidden_dim > 0, "hidden_dim must be > 0");
    assert_eq!(input.len() % hidden_dim, 0, "input length not divisible by hidden_dim");
    assert_eq!(gamma.len(), hidden_dim, "gamma length mismatch");

    let batch_size = input.len() / hidden_dim;
    for b in 0..batch_size {
        let s = b * hidden_dim;
        let e = s + hidden_dim;
        rms_norm_f32(&input[s..e], &mut output[s..e], gamma, eps);
    }
}

// ── Public API: f16 ───────────────────────────────────────────────

/// LayerNorm for f16 input/output (computation in f32 internally).
pub fn layer_norm_f16(input: &[F16], output: &mut [F16], gamma: &[F16], beta: &[F16], eps: f32) {
    let in_f32: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();
    let gam_f32: Vec<f32> = gamma.iter().map(|x| x.to_f32()).collect();
    let bet_f32: Vec<f32> = beta.iter().map(|x| x.to_f32()).collect();
    let mut out_f32 = vec![0.0f32; input.len()];

    layer_norm_f32(&in_f32, &mut out_f32, &gam_f32, &bet_f32, eps);

    for (dst, &v) in output.iter_mut().zip(out_f32.iter()) {
        *dst = F16::from_f32(v);
    }
}

/// RMSNorm for f16 input/output (computation in f32 internally).
pub fn rms_norm_f16(input: &[F16], output: &mut [F16], gamma: &[F16], eps: f32) {
    let in_f32: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();
    let gam_f32: Vec<f32> = gamma.iter().map(|x| x.to_f32()).collect();
    let mut out_f32 = vec![0.0f32; input.len()];

    rms_norm_f32(&in_f32, &mut out_f32, &gam_f32, eps);

    for (dst, &v) in output.iter_mut().zip(out_f32.iter()) {
        *dst = F16::from_f32(v);
    }
}

/// Batch LayerNorm for f16.
pub fn batch_layer_norm_f16(
    input: &[F16],
    output: &mut [F16],
    gamma: &[F16],
    beta: &[F16],
    hidden_dim: usize,
    eps: f32,
) {
    let in_f32: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();
    let gam_f32: Vec<f32> = gamma.iter().map(|x| x.to_f32()).collect();
    let bet_f32: Vec<f32> = beta.iter().map(|x| x.to_f32()).collect();
    let mut out_f32 = vec![0.0f32; input.len()];

    batch_layer_norm_f32(&in_f32, &mut out_f32, &gam_f32, &bet_f32, hidden_dim, eps);

    for (dst, &v) in output.iter_mut().zip(out_f32.iter()) {
        *dst = F16::from_f32(v);
    }
}

/// Batch RMSNorm for f16.
pub fn batch_rms_norm_f16(
    input: &[F16],
    output: &mut [F16],
    gamma: &[F16],
    hidden_dim: usize,
    eps: f32,
) {
    let in_f32: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();
    let gam_f32: Vec<f32> = gamma.iter().map(|x| x.to_f32()).collect();
    let mut out_f32 = vec![0.0f32; input.len()];

    batch_rms_norm_f32(&in_f32, &mut out_f32, &gam_f32, hidden_dim, eps);

    for (dst, &v) in output.iter_mut().zip(out_f32.iter()) {
        *dst = F16::from_f32(v);
    }
}

// ── NEON implementation (aarch64 only) ────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_layer_norm_impl(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    // SAFETY: caller guarantees aarch64 + NEON; all intrinsics valid.
    unsafe {
        let n = input.len();
        let mean = neon_sum(input) / n as f32;
        let var = neon_sum_sq_diff(input, mean) / n as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        neon_affine_transform(input, output, gamma, beta, mean, inv_std);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_rms_norm_impl(input: &[f32], output: &mut [f32], gamma: &[f32], eps: f32) {
    // SAFETY: caller guarantees aarch64 + NEON; all intrinsics valid.
    unsafe {
        let n = input.len();
        let mean_sq = neon_sum_of_squares(input) / n as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        neon_scale_transform(input, output, gamma, inv_rms);
    }
}

/// NEON horizontal sum of `data`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum(data: &[f32]) -> f32 {
    // SAFETY: all NEON intrinsics valid under target_feature guarantee.
    unsafe {
        let n = data.len();
        let chunks = n / 4;
        let remainder = n % 4;
        let ptr = data.as_ptr();

        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            acc = vaddq_f32(acc, v);
        }
        let mut sum: f32 = vaddvq_f32(acc);

        let tail = chunks * 4;
        for j in 0..remainder {
            sum += data[tail + j];
        }
        sum
    }
}

/// NEON sum of `(x - mean)²`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum_sq_diff(data: &[f32], mean: f32) -> f32 {
    // SAFETY: all NEON intrinsics valid under target_feature guarantee.
    unsafe {
        let n = data.len();
        let chunks = n / 4;
        let remainder = n % 4;
        let ptr = data.as_ptr();
        let mean_vec = vdupq_n_f32(mean);

        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            let diff = vsubq_f32(v, mean_vec);
            acc = vfmaq_f32(acc, diff, diff);
        }
        let mut sum: f32 = vaddvq_f32(acc);

        let tail = chunks * 4;
        for j in 0..remainder {
            let d = data[tail + j] - mean;
            sum += d * d;
        }
        sum
    }
}

/// NEON sum of `x²`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum_of_squares(data: &[f32]) -> f32 {
    // SAFETY: all NEON intrinsics valid under target_feature guarantee.
    unsafe {
        let n = data.len();
        let chunks = n / 4;
        let remainder = n % 4;
        let ptr = data.as_ptr();

        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            acc = vfmaq_f32(acc, v, v);
        }
        let mut sum: f32 = vaddvq_f32(acc);

        let tail = chunks * 4;
        for j in 0..remainder {
            let x = data[tail + j];
            sum += x * x;
        }
        sum
    }
}

/// NEON: `output = γ * (input − mean) * inv_std + β`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_affine_transform(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    mean: f32,
    inv_std: f32,
) {
    // SAFETY: all NEON intrinsics valid under target_feature guarantee.
    unsafe {
        let n = input.len();
        let chunks = n / 4;
        let remainder = n % 4;

        let mean_v = vdupq_n_f32(mean);
        let inv_v = vdupq_n_f32(inv_std);
        let ip = input.as_ptr();
        let gp = gamma.as_ptr();
        let bp = beta.as_ptr();
        let op = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(ip.add(off));
            let g = vld1q_f32(gp.add(off));
            let b = vld1q_f32(bp.add(off));
            let centered = vsubq_f32(x, mean_v);
            let normed = vmulq_f32(centered, inv_v);
            let scaled = vfmaq_f32(b, g, normed); // b + g*normed
            vst1q_f32(op.add(off), scaled);
        }

        let tail = chunks * 4;
        for j in 0..remainder {
            let idx = tail + j;
            let normed = (input[idx] - mean) * inv_std;
            output[idx] = gamma[idx] * normed + beta[idx];
        }
    }
}

/// NEON: `output = γ * input * inv_rms`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_scale_transform(input: &[f32], output: &mut [f32], gamma: &[f32], inv_rms: f32) {
    // SAFETY: all NEON intrinsics valid under target_feature guarantee.
    unsafe {
        let n = input.len();
        let chunks = n / 4;
        let remainder = n % 4;

        let inv_v = vdupq_n_f32(inv_rms);
        let ip = input.as_ptr();
        let gp = gamma.as_ptr();
        let op = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(ip.add(off));
            let g = vld1q_f32(gp.add(off));
            let normed = vmulq_f32(x, inv_v);
            let scaled = vmulq_f32(g, normed);
            vst1q_f32(op.add(off), scaled);
        }

        let tail = chunks * 4;
        for j in 0..remainder {
            let idx = tail + j;
            output[idx] = gamma[idx] * input[idx] * inv_rms;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-4;
    const F16_TOL: f32 = 5e-2; // f16 has ~3 decimal digits of precision

    // ── scalar references for parity checks ──────────────────────

    fn ref_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        let mean: f32 = input.iter().sum::<f32>() / n as f32;
        let var: f32 = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + eps).sqrt();
        input.iter().enumerate().map(|(i, &x)| gamma[i] * (x - mean) * inv + beta[i]).collect()
    }

    fn ref_rms_norm(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        let ms: f32 = input.iter().map(|&x| x * x).sum::<f32>() / n as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        input.iter().enumerate().map(|(i, &x)| gamma[i] * x * inv).collect()
    }

    fn assert_approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at [{i}]: {x} vs {y} (diff={}, tol={tol})",
                (x - y).abs()
            );
        }
    }

    // ── LayerNorm f32 ────────────────────────────────────────────

    #[test]
    fn test_layer_norm_basic_aligned() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 8];
        layer_norm_f32(&input, &mut out, &gamma, &beta, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_with_affine_params() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let beta = vec![0.1, -0.1, 0.0, 0.5, -0.5];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 5];
        layer_norm_f32(&input, &mut out, &gamma, &beta, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_zero_variance() {
        let input = vec![3.0; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let mut out = vec![0.0; 8];
        layer_norm_f32(&input, &mut out, &gamma, &beta, EPS);
        for &v in &out {
            assert!(v.abs() < TOL, "expected ~0 for constant input, got {v}");
        }
    }

    #[test]
    fn test_layer_norm_single_element() {
        let input = vec![42.0];
        let gamma = vec![2.0];
        let beta = vec![1.0];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 1];
        layer_norm_f32(&input, &mut out, &gamma, &beta, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_large_1024() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let gamma = vec![1.0; n];
        let beta = vec![0.0; n];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; n];
        layer_norm_f32(&input, &mut out, &gamma, &beta, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_non_aligned_13() {
        let input: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let gamma = vec![1.0; 13];
        let beta = vec![0.0; 13];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 13];
        layer_norm_f32(&input, &mut out, &gamma, &beta, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_negative_values() {
        let input = vec![-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 8];
        layer_norm_f32(&input, &mut out, &gamma, &beta, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_identity_transform() {
        // gamma=1, beta=0 → should produce standard normalization
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        layer_norm_f32(&input, &mut out, &gamma, &beta, EPS);
        // Mean = 5, Var = 5, check the normalized result sums to ~0
        let sum: f32 = out.iter().sum();
        assert!(sum.abs() < TOL, "normalized mean should be ~0, got {sum}");
    }

    #[test]
    fn test_layer_norm_empty_input() {
        let mut out: Vec<f32> = vec![];
        layer_norm_f32(&[], &mut out, &[], &[], EPS); // should not panic
    }

    // ── RMSNorm f32 ─────────────────────────────────────────────

    #[test]
    fn test_rms_norm_basic_aligned() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let expected = ref_rms_norm(&input, &gamma, EPS);
        let mut out = vec![0.0; 8];
        rms_norm_f32(&input, &mut out, &gamma, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_with_scale() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let expected = ref_rms_norm(&input, &gamma, EPS);
        let mut out = vec![0.0; 5];
        rms_norm_f32(&input, &mut out, &gamma, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_single_element() {
        let input = vec![42.0];
        let gamma = vec![2.0];
        let expected = ref_rms_norm(&input, &gamma, EPS);
        let mut out = vec![0.0; 1];
        rms_norm_f32(&input, &mut out, &gamma, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_large_1024() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let gamma = vec![1.0; n];
        let expected = ref_rms_norm(&input, &gamma, EPS);
        let mut out = vec![0.0; n];
        rms_norm_f32(&input, &mut out, &gamma, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_non_aligned_13() {
        let input: Vec<f32> = (0..13).map(|i| (i as f32) + 1.0).collect();
        let gamma = vec![1.0; 13];
        let expected = ref_rms_norm(&input, &gamma, EPS);
        let mut out = vec![0.0; 13];
        rms_norm_f32(&input, &mut out, &gamma, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_all_ones() {
        let input = vec![1.0; 8];
        let gamma = vec![1.0; 8];
        let mut out = vec![0.0; 8];
        rms_norm_f32(&input, &mut out, &gamma, EPS);
        // rms(1,1,...) ≈ 1 → output ≈ 1
        for &v in &out {
            assert!((v - 1.0).abs() < TOL, "expected ~1.0, got {v}");
        }
    }

    // ── Welford accumulator ─────────────────────────────────────

    #[test]
    fn test_welford_accumulator_basic() {
        let data = [2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let acc = welford_stats(&data);
        let expected_mean = 5.0f32;
        let expected_var = 4.0f32;
        assert!((acc.mean_f32() - expected_mean).abs() < TOL);
        assert!((acc.variance_f32() - expected_var).abs() < TOL);
    }

    #[test]
    fn test_welford_accumulator_merge() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let full = welford_stats(&data);

        let a = welford_stats(&data[..4]);
        let b = welford_stats(&data[4..]);
        let merged = WelfordAccumulator::merge(&a, &b);

        assert!((merged.mean_f32() - full.mean_f32()).abs() < TOL);
        assert!((merged.variance_f32() - full.variance_f32()).abs() < TOL);
        assert_eq!(merged.count, full.count);
    }

    #[test]
    fn test_welford_accumulator_empty() {
        let acc = WelfordAccumulator::new();
        assert_eq!(acc.count, 0);
        assert_eq!(acc.variance(), 0.0);

        // merge with empty
        let data = welford_stats(&[1.0, 2.0, 3.0]);
        let merged = WelfordAccumulator::merge(&acc, &data);
        assert_eq!(merged.count, data.count);
        assert!((merged.mean_f32() - data.mean_f32()).abs() < TOL);
    }

    #[test]
    fn test_welford_large_values_stability() {
        // Large offset: naive f32 sum would lose precision, Welford in f64 stays accurate.
        let offset = 1e7_f64;
        let data: Vec<f32> = (0..256).map(|i| (offset + (i as f64) * 0.001) as f32).collect();
        let acc = welford_stats(&data);

        // f64 reference for ground truth
        let n = data.len() as f64;
        let mean_f64: f64 = data.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var_f64: f64 = data
            .iter()
            .map(|&x| {
                let d = x as f64 - mean_f64;
                d * d
            })
            .sum::<f64>()
            / n;

        assert!((acc.mean - mean_f64).abs() < 1.0, "mean: {} vs {}", acc.mean, mean_f64);
        assert!(
            (acc.variance() - var_f64).abs() / (var_f64 + 1e-12) < 0.01,
            "variance: {} vs {}",
            acc.variance(),
            var_f64
        );
    }

    // ── Welford LayerNorm / RMSNorm ──────────────────────────────

    #[test]
    fn test_layer_norm_welford_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 8];
        layer_norm_welford_f32(&input, &mut out, &gamma, &beta, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_welford_matches_standard() {
        let n = 137;
        let input: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
        let beta: Vec<f32> = (0..n).map(|i| -0.3 + (i % 3) as f32 * 0.1).collect();

        let mut standard = vec![0.0; n];
        let mut welford = vec![0.0; n];
        layer_norm_f32(&input, &mut standard, &gamma, &beta, EPS);
        layer_norm_welford_f32(&input, &mut welford, &gamma, &beta, EPS);
        assert_approx(&standard, &welford, TOL);
    }

    #[test]
    fn test_rms_norm_welford_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let expected = ref_rms_norm(&input, &gamma, EPS);
        let mut out = vec![0.0; 8];
        rms_norm_welford_f32(&input, &mut out, &gamma, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_welford_matches_standard() {
        let n = 137;
        let input: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();

        let mut standard = vec![0.0; n];
        let mut welford = vec![0.0; n];
        rms_norm_f32(&input, &mut standard, &gamma, EPS);
        rms_norm_welford_f32(&input, &mut welford, &gamma, EPS);
        assert_approx(&standard, &welford, TOL);
    }

    // ── Batch processing ─────────────────────────────────────────

    #[test]
    fn test_batch_layer_norm_single_row() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 4];
        batch_layer_norm_f32(&input, &mut out, &gamma, &beta, 4, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_batch_layer_norm_multi_row() {
        let hidden = 4;
        let rows = 3;
        let input: Vec<f32> = (0..hidden * rows).map(|i| i as f32).collect();
        let gamma = vec![1.0; hidden];
        let beta = vec![0.0; hidden];
        let mut out = vec![0.0; hidden * rows];
        batch_layer_norm_f32(&input, &mut out, &gamma, &beta, hidden, EPS);

        // Each row should match independent layer_norm
        for r in 0..rows {
            let s = r * hidden;
            let e = s + hidden;
            let expected = ref_layer_norm(&input[s..e], &gamma, &beta, EPS);
            assert_approx(&out[s..e], &expected, TOL);
        }
    }

    #[test]
    fn test_batch_layer_norm_row_independence() {
        let hidden = 8;
        let gamma = vec![1.0; hidden];
        let beta = vec![0.0; hidden];

        // Row 0 is constant, row 1 is varied
        let mut input = vec![5.0; hidden * 2];
        for i in 0..hidden {
            input[hidden + i] = i as f32;
        }

        let mut out = vec![0.0; hidden * 2];
        batch_layer_norm_f32(&input, &mut out, &gamma, &beta, hidden, EPS);

        // Row 0: constant → ~0 output
        for &v in &out[..hidden] {
            assert!(v.abs() < TOL, "constant row should give ~0, got {v}");
        }
        // Row 1: should match individual layer_norm
        let expected = ref_layer_norm(&input[hidden..], &gamma, &beta, EPS);
        assert_approx(&out[hidden..], &expected, TOL);
    }

    #[test]
    fn test_batch_rms_norm_single_row() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let expected = ref_rms_norm(&input, &gamma, EPS);
        let mut out = vec![0.0; 4];
        batch_rms_norm_f32(&input, &mut out, &gamma, 4, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_batch_rms_norm_multi_row() {
        let hidden = 4;
        let rows = 3;
        let input: Vec<f32> = (1..=hidden * rows).map(|i| i as f32).collect();
        let gamma = vec![1.0; hidden];
        let mut out = vec![0.0; hidden * rows];
        batch_rms_norm_f32(&input, &mut out, &gamma, hidden, EPS);

        for r in 0..rows {
            let s = r * hidden;
            let e = s + hidden;
            let expected = ref_rms_norm(&input[s..e], &gamma, EPS);
            assert_approx(&out[s..e], &expected, TOL);
        }
    }

    // ── f16 support ──────────────────────────────────────────────

    #[test]
    fn test_f16_roundtrip_normal() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0, 0.001];
        for &v in &values {
            let h = F16::from_f32(v);
            let back = h.to_f32();
            assert!(
                (back - v).abs() <= v.abs() * 0.01 + 1e-4,
                "f16 roundtrip failed for {v}: got {back}"
            );
        }
    }

    #[test]
    fn test_f16_special_values() {
        // Zero
        assert_eq!(F16::from_f32(0.0).to_f32(), 0.0);
        // Infinity
        assert!(F16::from_f32(f32::INFINITY).to_f32().is_infinite());
        assert!(F16::from_f32(f32::NEG_INFINITY).to_f32().is_infinite());
        // NaN
        assert!(F16::from_f32(f32::NAN).to_f32().is_nan());
        // Constants
        assert_eq!(F16::ZERO.to_f32(), 0.0);
        assert!((F16::ONE.to_f32() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_layer_norm_f16_basic() {
        let input: Vec<F16> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|&v| F16::from_f32(v)).collect();
        let gamma: Vec<F16> = vec![F16::ONE; 4];
        let beta: Vec<F16> = vec![F16::ZERO; 4];
        let mut out = vec![F16::ZERO; 4];

        layer_norm_f16(&input, &mut out, &gamma, &beta, EPS);

        let in_f32: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();
        let expected = ref_layer_norm(&in_f32, &[1.0; 4], &[0.0; 4], EPS);
        let out_f32: Vec<f32> = out.iter().map(|x| x.to_f32()).collect();
        assert_approx(&out_f32, &expected, F16_TOL);
    }

    #[test]
    fn test_rms_norm_f16_basic() {
        let input: Vec<F16> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|&v| F16::from_f32(v)).collect();
        let gamma: Vec<F16> = vec![F16::ONE; 4];
        let mut out = vec![F16::ZERO; 4];

        rms_norm_f16(&input, &mut out, &gamma, EPS);

        let in_f32: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();
        let expected = ref_rms_norm(&in_f32, &[1.0; 4], EPS);
        let out_f32: Vec<f32> = out.iter().map(|x| x.to_f32()).collect();
        assert_approx(&out_f32, &expected, F16_TOL);
    }

    #[test]
    fn test_batch_layer_norm_f16() {
        let hidden = 4;
        let vals: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let input: Vec<F16> = vals.iter().map(|&v| F16::from_f32(v)).collect();
        let gamma: Vec<F16> = vec![F16::ONE; hidden];
        let beta: Vec<F16> = vec![F16::ZERO; hidden];
        let mut out = vec![F16::ZERO; 8];

        batch_layer_norm_f16(&input, &mut out, &gamma, &beta, hidden, EPS);

        let in_f32: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();
        let mut expected = vec![0.0f32; 8];
        batch_layer_norm_f32(&in_f32, &mut expected, &[1.0; 4], &[0.0; 4], hidden, EPS);
        let out_f32: Vec<f32> = out.iter().map(|x| x.to_f32()).collect();
        assert_approx(&out_f32, &expected, F16_TOL);
    }

    #[test]
    fn test_batch_rms_norm_f16() {
        let hidden = 4;
        let vals: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let input: Vec<F16> = vals.iter().map(|&v| F16::from_f32(v)).collect();
        let gamma: Vec<F16> = vec![F16::ONE; hidden];
        let mut out = vec![F16::ZERO; 8];

        batch_rms_norm_f16(&input, &mut out, &gamma, hidden, EPS);

        let in_f32: Vec<f32> = input.iter().map(|x| x.to_f32()).collect();
        let mut expected = vec![0.0f32; 8];
        batch_rms_norm_f32(&in_f32, &mut expected, &[1.0; 4], hidden, EPS);
        let out_f32: Vec<f32> = out.iter().map(|x| x.to_f32()).collect();
        assert_approx(&out_f32, &expected, F16_TOL);
    }

    // ── Parity: non-power-of-two with non-trivial params ─────────

    #[test]
    fn test_layer_norm_parity_large_non_aligned() {
        let n = 137;
        let input: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
        let beta: Vec<f32> = (0..n).map(|i| -0.3 + (i % 3) as f32 * 0.1).collect();
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; n];
        layer_norm_f32(&input, &mut out, &gamma, &beta, EPS);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_parity_large_non_aligned() {
        let n = 137;
        let input: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
        let expected = ref_rms_norm(&input, &gamma, EPS);
        let mut out = vec![0.0; n];
        rms_norm_f32(&input, &mut out, &gamma, EPS);
        assert_approx(&out, &expected, TOL);
    }
}
