//! SIMD-optimized batch normalization kernels.
//!
//! Provides batch normalization for 1D (NxC) and 2D (NxCxHxW) inputs,
//! group normalization, instance normalization, and fused BN+ReLU /
//! BN+residual-add operations.  Statistics are computed via f64
//! accumulation for numerical stability, with AVX2-accelerated inner
//! loops on x86_64.

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

use bitnet_common::{BitNetError, KernelError, Result};

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Configuration & state ──────────────────────────────────────────

/// Configuration for SIMD-optimized batch normalization.
#[derive(Debug, Clone)]
pub struct SimdBatchNormConfig {
    /// Small constant added to variance for numerical stability.
    pub epsilon: f32,
    /// Momentum for running mean/variance update (new = (1-momentum)*old + momentum*batch).
    pub momentum: f32,
    /// Whether to apply learnable affine parameters (gamma/beta).
    pub affine: bool,
    /// Whether to maintain running statistics.
    pub track_running_stats: bool,
}

impl SimdBatchNormConfig {
    /// Construct with default epsilon (1e-5), momentum (0.1), affine on,
    /// tracking on.
    pub fn new() -> Self {
        Self { epsilon: 1e-5, momentum: 0.1, affine: true, track_running_stats: true }
    }
}

impl Default for SimdBatchNormConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Running mean/variance state for batch normalization.
#[derive(Debug, Clone)]
pub struct SimdBatchNormState {
    /// Per-channel running mean.
    pub running_mean: Vec<f32>,
    /// Per-channel running variance.
    pub running_var: Vec<f32>,
    /// Number of batches tracked so far.
    pub num_batches_tracked: u64,
}

impl SimdBatchNormState {
    /// Create initial state for `num_features` channels.
    pub fn new(num_features: usize) -> Self {
        Self {
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            num_batches_tracked: 0,
        }
    }
}

// ── SIMD helpers ───────────────────────────────────────────────────

/// Horizontal sum of an AVX2 register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let sum4 = _mm_add_ps(hi, lo);
    let hi2 = _mm_movehl_ps(sum4, sum4);
    let sum2 = _mm_add_ps(sum4, hi2);
    let hi1 = _mm_shuffle_ps::<0x01>(sum2, sum2);
    _mm_cvtss_f32(_mm_add_ss(sum2, hi1))
}

// ── Compute primitives ─────────────────────────────────────────────

/// Compute the mean of `data` using SIMD when available.
pub fn compute_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { compute_mean_avx2(data) };
        }
    }
    compute_mean_scalar(data)
}

fn compute_mean_scalar(data: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    for &x in data {
        sum += x as f64;
    }
    (sum / data.len() as f64) as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_mean_avx2(data: &[f32]) -> f32 {
    unsafe {
        let n = data.len();
        let chunks = n / 8;
        let mut acc = _mm256_setzero_ps();
        let ptr = data.as_ptr();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(ptr.add(i * 8));
            acc = _mm256_add_ps(acc, v);
        }
        let mut sum = hsum_avx2(acc) as f64;
        for i in (chunks * 8)..n {
            sum += *ptr.add(i) as f64;
        }
        (sum / n as f64) as f32
    }
}

/// Compute the variance of `data` given its mean using SIMD when available.
pub fn compute_variance(data: &[f32], mean: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { compute_variance_avx2(data, mean) };
        }
    }
    compute_variance_scalar(data, mean)
}

fn compute_variance_scalar(data: &[f32], mean: f32) -> f32 {
    let mean_d = mean as f64;
    let mut sum = 0.0f64;
    for &x in data {
        let d = x as f64 - mean_d;
        sum += d * d;
    }
    (sum / data.len() as f64) as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_variance_avx2(data: &[f32], mean: f32) -> f32 {
    unsafe {
        let n = data.len();
        let chunks = n / 8;
        let mean_v = _mm256_set1_ps(mean);
        let mut acc = _mm256_setzero_ps();
        let ptr = data.as_ptr();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(ptr.add(i * 8));
            let d = _mm256_sub_ps(v, mean_v);
            acc = _mm256_fmadd_ps(d, d, acc);
        }
        let mut sum = hsum_avx2(acc) as f64;
        let mean_d = mean as f64;
        for i in (chunks * 8)..n {
            let d = *ptr.add(i) as f64 - mean_d;
            sum += d * d;
        }
        (sum / n as f64) as f32
    }
}

/// Normalize `data` and optionally apply affine transform (gamma * x_hat + beta).
pub fn normalize_and_scale(
    data: &[f32],
    mean: f32,
    inv_std: f32,
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    output: &mut [f32],
) {
    debug_assert_eq!(data.len(), output.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                normalize_and_scale_avx2(data, mean, inv_std, gamma, beta, output);
            }
            return;
        }
    }
    normalize_and_scale_scalar(data, mean, inv_std, gamma, beta, output);
}

fn normalize_and_scale_scalar(
    data: &[f32],
    mean: f32,
    inv_std: f32,
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    output: &mut [f32],
) {
    match (gamma, beta) {
        (Some(g), Some(b)) => {
            for i in 0..data.len() {
                output[i] = (data[i] - mean) * inv_std * g[i] + b[i];
            }
        }
        (Some(g), None) => {
            for i in 0..data.len() {
                output[i] = (data[i] - mean) * inv_std * g[i];
            }
        }
        _ => {
            for i in 0..data.len() {
                output[i] = (data[i] - mean) * inv_std;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_and_scale_avx2(
    data: &[f32],
    mean: f32,
    inv_std: f32,
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    output: &mut [f32],
) {
    unsafe {
        let n = data.len();
        let chunks = n / 8;
        let mean_v = _mm256_set1_ps(mean);
        let inv_v = _mm256_set1_ps(inv_std);
        let dptr = data.as_ptr();
        let optr = output.as_mut_ptr();

        match (gamma, beta) {
            (Some(g), Some(b)) => {
                let gptr = g.as_ptr();
                let bptr = b.as_ptr();
                for i in 0..chunks {
                    let off = i * 8;
                    let v = _mm256_loadu_ps(dptr.add(off));
                    let gv = _mm256_loadu_ps(gptr.add(off));
                    let bv = _mm256_loadu_ps(bptr.add(off));
                    let d = _mm256_sub_ps(v, mean_v);
                    let normed = _mm256_mul_ps(d, inv_v);
                    let scaled = _mm256_fmadd_ps(normed, gv, bv);
                    _mm256_storeu_ps(optr.add(off), scaled);
                }
                for i in (chunks * 8)..n {
                    *optr.add(i) = (*dptr.add(i) - mean) * inv_std * *gptr.add(i) + *bptr.add(i);
                }
            }
            (Some(g), None) => {
                let gptr = g.as_ptr();
                for i in 0..chunks {
                    let off = i * 8;
                    let v = _mm256_loadu_ps(dptr.add(off));
                    let gv = _mm256_loadu_ps(gptr.add(off));
                    let d = _mm256_sub_ps(v, mean_v);
                    let normed = _mm256_mul_ps(d, inv_v);
                    let scaled = _mm256_mul_ps(normed, gv);
                    _mm256_storeu_ps(optr.add(off), scaled);
                }
                for i in (chunks * 8)..n {
                    *optr.add(i) = (*dptr.add(i) - mean) * inv_std * *gptr.add(i);
                }
            }
            _ => {
                for i in 0..chunks {
                    let off = i * 8;
                    let v = _mm256_loadu_ps(dptr.add(off));
                    let d = _mm256_sub_ps(v, mean_v);
                    let normed = _mm256_mul_ps(d, inv_v);
                    _mm256_storeu_ps(optr.add(off), normed);
                }
                for i in (chunks * 8)..n {
                    *optr.add(i) = (*dptr.add(i) - mean) * inv_std;
                }
            }
        }
    }
}

// ── Core batch normalization ───────────────────────────────────────

/// Batch normalization forward pass (training mode — computes batch
/// statistics and updates running state).
///
/// Input layout: `[N, C]` where `N = input.len() / num_features`.
///
/// Returns the normalized output.
pub fn batch_norm_forward(
    input: &[f32],
    num_features: usize,
    gamma: &[f32],
    beta: &[f32],
    state: &mut SimdBatchNormState,
    config: &SimdBatchNormConfig,
) -> Result<Vec<f32>> {
    validate_bn_args(input, num_features, gamma, beta, config)?;
    if state.running_mean.len() != num_features || state.running_var.len() != num_features {
        return Err(invalid_args("state dimensions must match num_features"));
    }
    let batch_size = input.len() / num_features;

    // Per-channel batch statistics.
    let mut batch_mean = vec![0.0f64; num_features];
    let mut batch_var = vec![0.0f64; num_features];
    let count = batch_size as f64;

    for n in 0..batch_size {
        let row = &input[n * num_features..(n + 1) * num_features];
        for (ch, val) in row.iter().enumerate() {
            batch_mean[ch] += *val as f64;
        }
    }
    for m in &mut batch_mean {
        *m /= count;
    }
    for n in 0..batch_size {
        let row = &input[n * num_features..(n + 1) * num_features];
        for (ch, val) in row.iter().enumerate() {
            let d = *val as f64 - batch_mean[ch];
            batch_var[ch] += d * d;
        }
    }
    for v in &mut batch_var {
        *v /= count;
    }

    // Normalize.
    let mut output = vec![0.0f32; input.len()];
    let eps = config.epsilon as f64;
    for ch in 0..num_features {
        let inv_std = 1.0 / (batch_var[ch] + eps).sqrt();
        let mean_f = batch_mean[ch] as f32;
        let inv_f = inv_std as f32;
        // Collect per-channel slices for SIMD.
        let channel_data: Vec<f32> =
            (0..batch_size).map(|n| input[n * num_features + ch]).collect();
        let mut channel_out = vec![0.0f32; batch_size];
        if config.affine {
            let g = vec![gamma[ch]; batch_size];
            let b = vec![beta[ch]; batch_size];
            normalize_and_scale(&channel_data, mean_f, inv_f, Some(&g), Some(&b), &mut channel_out);
        } else {
            normalize_and_scale(&channel_data, mean_f, inv_f, None, None, &mut channel_out);
        }
        for n in 0..batch_size {
            output[n * num_features + ch] = channel_out[n];
        }
    }

    // Update running stats.
    if config.track_running_stats {
        let mom = config.momentum as f64;
        for ch in 0..num_features {
            state.running_mean[ch] =
                ((1.0 - mom) * state.running_mean[ch] as f64 + mom * batch_mean[ch]) as f32;
            state.running_var[ch] =
                ((1.0 - mom) * state.running_var[ch] as f64 + mom * batch_var[ch]) as f32;
        }
        state.num_batches_tracked += 1;
    }

    Ok(output)
}

/// Batch normalization inference pass (uses running statistics, no
/// state update).
pub fn batch_norm_inference(
    input: &[f32],
    num_features: usize,
    gamma: &[f32],
    beta: &[f32],
    state: &SimdBatchNormState,
    config: &SimdBatchNormConfig,
) -> Result<Vec<f32>> {
    validate_bn_args(input, num_features, gamma, beta, config)?;
    if state.running_mean.len() != num_features || state.running_var.len() != num_features {
        return Err(invalid_args("state dimensions must match num_features"));
    }
    let batch_size = input.len() / num_features;
    let eps = config.epsilon as f64;
    let mut output = vec![0.0f32; input.len()];

    for ch in 0..num_features {
        let inv_std = (1.0 / (state.running_var[ch] as f64 + eps).sqrt()) as f32;
        let mean_f = state.running_mean[ch];
        let channel_data: Vec<f32> =
            (0..batch_size).map(|n| input[n * num_features + ch]).collect();
        let mut channel_out = vec![0.0f32; batch_size];
        if config.affine {
            let g = vec![gamma[ch]; batch_size];
            let b = vec![beta[ch]; batch_size];
            normalize_and_scale(
                &channel_data,
                mean_f,
                inv_std,
                Some(&g),
                Some(&b),
                &mut channel_out,
            );
        } else {
            normalize_and_scale(&channel_data, mean_f, inv_std, None, None, &mut channel_out);
        }
        for n in 0..batch_size {
            output[n * num_features + ch] = channel_out[n];
        }
    }

    Ok(output)
}

// ── 1D / 2D convenience wrappers ──────────────────────────────────

/// Batch normalization for 1D input layout `[N, C]`.
///
/// This is a thin wrapper around [`batch_norm_inference`] that validates
/// the shape.
pub fn batch_norm_1d(
    input: &[f32],
    num_features: usize,
    gamma: &[f32],
    beta: &[f32],
    state: &SimdBatchNormState,
    config: &SimdBatchNormConfig,
) -> Result<Vec<f32>> {
    if !input.len().is_multiple_of(num_features) {
        return Err(invalid_args("input length must be a multiple of num_features for 1D BN"));
    }
    batch_norm_inference(input, num_features, gamma, beta, state, config)
}

/// Batch normalization for 2D input layout `[N, C, H, W]`.
///
/// Reshapes to `[N*H*W, C]`, applies BN, and reshapes back.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_2d(
    input: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    gamma: &[f32],
    beta: &[f32],
    state: &SimdBatchNormState,
    config: &SimdBatchNormConfig,
) -> Result<Vec<f32>> {
    let expected = n * c * h * w;
    if input.len() != expected {
        return Err(invalid_args(&format!("input length {} != N*C*H*W = {expected}", input.len())));
    }
    // Transpose from NCHW → (N*H*W, C) for per-channel normalization.
    let spatial = h * w;
    let total = n * spatial;
    let mut transposed = vec![0.0f32; expected];
    for ni in 0..n {
        for ci in 0..c {
            for si in 0..spatial {
                let src = ni * c * spatial + ci * spatial + si;
                let dst = (ni * spatial + si) * c + ci;
                transposed[dst] = input[src];
            }
        }
    }

    let normed = batch_norm_inference(&transposed, c, gamma, beta, state, config)?;

    // Transpose back (N*H*W, C) → NCHW.
    let mut output = vec![0.0f32; expected];
    for ni in 0..n {
        for ci in 0..c {
            for si in 0..spatial {
                let src = (ni * spatial + si) * c + ci;
                let dst = ni * c * spatial + ci * spatial + si;
                output[dst] = normed[src];
            }
        }
    }
    let _ = total; // suppress unused warning
    Ok(output)
}

// ── Fused operations ───────────────────────────────────────────────

/// Fused batch normalization + ReLU activation.
///
/// Applies BN in inference mode followed by `max(0, x)`.
pub fn fused_bn_relu(
    input: &[f32],
    num_features: usize,
    gamma: &[f32],
    beta: &[f32],
    state: &SimdBatchNormState,
    config: &SimdBatchNormConfig,
) -> Result<Vec<f32>> {
    let mut out = batch_norm_inference(input, num_features, gamma, beta, state, config)?;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { relu_inplace_avx2(&mut out) };
            return Ok(out);
        }
    }
    for v in &mut out {
        *v = v.max(0.0);
    }
    Ok(out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_inplace_avx2(data: &mut [f32]) {
    unsafe {
        let n = data.len();
        let chunks = n / 8;
        let zero = _mm256_setzero_ps();
        let ptr = data.as_mut_ptr();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(ptr.add(i * 8));
            _mm256_storeu_ps(ptr.add(i * 8), _mm256_max_ps(v, zero));
        }
        for i in (chunks * 8)..n {
            let p = ptr.add(i);
            if *p < 0.0 {
                *p = 0.0;
            }
        }
    }
}

/// Fused batch normalization + residual addition.
///
/// Computes `BN(input) + residual`.
pub fn fused_bn_add(
    input: &[f32],
    residual: &[f32],
    num_features: usize,
    gamma: &[f32],
    beta: &[f32],
    state: &SimdBatchNormState,
    config: &SimdBatchNormConfig,
) -> Result<Vec<f32>> {
    if input.len() != residual.len() {
        return Err(invalid_args("input and residual must have the same length"));
    }
    let mut out = batch_norm_inference(input, num_features, gamma, beta, state, config)?;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { vec_add_inplace_avx2(&mut out, residual) };
            return Ok(out);
        }
    }
    for (o, &r) in out.iter_mut().zip(residual.iter()) {
        *o += r;
    }
    Ok(out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_add_inplace_avx2(a: &mut [f32], b: &[f32]) {
    unsafe {
        let n = a.len();
        let chunks = n / 8;
        let ap = a.as_mut_ptr();
        let bp = b.as_ptr();
        for i in 0..chunks {
            let off = i * 8;
            let va = _mm256_loadu_ps(ap.add(off));
            let vb = _mm256_loadu_ps(bp.add(off));
            _mm256_storeu_ps(ap.add(off), _mm256_add_ps(va, vb));
        }
        for i in (chunks * 8)..n {
            *ap.add(i) += *bp.add(i);
        }
    }
}

// ── Group normalization ────────────────────────────────────────────

/// Group normalization over `[N, C, spatial…]` inputs.
///
/// Channels are divided into `num_groups` equal groups and each group is
/// independently normalized.
pub fn group_norm(
    input: &[f32],
    num_groups: usize,
    num_channels: usize,
    spatial_size: usize,
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    epsilon: f32,
) -> Result<Vec<f32>> {
    if num_channels == 0 || num_groups == 0 || spatial_size == 0 {
        return Err(invalid_args("num_channels, num_groups, and spatial_size must be > 0"));
    }
    if !num_channels.is_multiple_of(num_groups) {
        return Err(invalid_args("num_channels must be divisible by num_groups"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    let total_per_sample = num_channels * spatial_size;
    if !input.len().is_multiple_of(total_per_sample) {
        return Err(invalid_args("input length must be a multiple of num_channels * spatial_size"));
    }
    if let Some(g) = gamma
        && g.len() != num_channels
    {
        return Err(invalid_args("gamma length must equal num_channels"));
    }
    if let Some(b) = beta
        && b.len() != num_channels
    {
        return Err(invalid_args("beta length must equal num_channels"));
    }
    if epsilon <= 0.0 || !epsilon.is_finite() {
        return Err(invalid_args("epsilon must be positive and finite"));
    }

    let batch_size = input.len() / total_per_sample;
    let cpg = num_channels / num_groups;
    let group_size = cpg * spatial_size;
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch_size {
        for g in 0..num_groups {
            // Gather the group elements into a contiguous slice for SIMD.
            let mut group_data = Vec::with_capacity(group_size);
            for c in (g * cpg)..((g + 1) * cpg) {
                let off = b * total_per_sample + c * spatial_size;
                group_data.extend_from_slice(&input[off..off + spatial_size]);
            }

            let mean = compute_mean(&group_data);
            let var = compute_variance(&group_data, mean);
            let inv_std = 1.0 / (var + epsilon).sqrt();

            for (ci, c) in ((g * cpg)..((g + 1) * cpg)).enumerate() {
                let off = b * total_per_sample + c * spatial_size;
                let src = &input[off..off + spatial_size];
                let dst = &mut output[off..off + spatial_size];
                match (gamma, beta) {
                    (Some(gam), Some(bet)) => {
                        let gv = vec![gam[c]; spatial_size];
                        let bv = vec![bet[c]; spatial_size];
                        normalize_and_scale(src, mean, inv_std, Some(&gv), Some(&bv), dst);
                    }
                    (Some(gam), None) => {
                        let gv = vec![gam[c]; spatial_size];
                        normalize_and_scale(src, mean, inv_std, Some(&gv), None, dst);
                    }
                    _ => {
                        normalize_and_scale(src, mean, inv_std, None, None, dst);
                    }
                }
                let _ = ci;
            }
        }
    }

    Ok(output)
}

// ── Instance normalization ─────────────────────────────────────────

/// Instance normalization — equivalent to group normalization with
/// `num_groups == num_channels`.
pub fn instance_norm(
    input: &[f32],
    num_channels: usize,
    spatial_size: usize,
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    epsilon: f32,
) -> Result<Vec<f32>> {
    group_norm(input, num_channels, num_channels, spatial_size, gamma, beta, epsilon)
}

// ── Validation helpers ─────────────────────────────────────────────

fn validate_bn_args(
    input: &[f32],
    num_features: usize,
    gamma: &[f32],
    beta: &[f32],
    config: &SimdBatchNormConfig,
) -> Result<()> {
    if num_features == 0 {
        return Err(invalid_args("num_features must be > 0"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if config.epsilon <= 0.0 || !config.epsilon.is_finite() {
        return Err(invalid_args("epsilon must be positive and finite"));
    }
    if !config.momentum.is_finite() || config.momentum < 0.0 || config.momentum > 1.0 {
        return Err(invalid_args("momentum must be in [0, 1] and finite"));
    }
    if config.affine && gamma.len() != num_features {
        return Err(invalid_args(&format!(
            "gamma length {} != num_features {num_features}",
            gamma.len()
        )));
    }
    if config.affine && beta.len() != num_features {
        return Err(invalid_args(&format!(
            "beta length {} != num_features {num_features}",
            beta.len()
        )));
    }
    if !input.len().is_multiple_of(num_features) {
        return Err(invalid_args("input length must be a multiple of num_features"));
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    fn default_state(c: usize) -> SimdBatchNormState {
        SimdBatchNormState::new(c)
    }

    fn default_config() -> SimdBatchNormConfig {
        SimdBatchNormConfig::new()
    }

    // ── SimdBatchNormConfig ───────────────────────────────────────

    #[test]
    fn config_defaults() {
        let c = SimdBatchNormConfig::new();
        assert!((c.epsilon - 1e-5).abs() < 1e-10);
        assert!((c.momentum - 0.1).abs() < 1e-10);
        assert!(c.affine);
        assert!(c.track_running_stats);
    }

    #[test]
    fn config_default_trait() {
        let c = SimdBatchNormConfig::default();
        assert!(c.affine);
        assert!(c.track_running_stats);
    }

    #[test]
    fn config_custom() {
        let c = SimdBatchNormConfig {
            epsilon: 1e-3,
            momentum: 0.5,
            affine: false,
            track_running_stats: false,
        };
        assert!((c.epsilon - 1e-3).abs() < 1e-10);
        assert!(!c.affine);
    }

    // ── SimdBatchNormState ────────────────────────────────────────

    #[test]
    fn state_initial_values() {
        let s = SimdBatchNormState::new(4);
        assert_eq!(s.running_mean, vec![0.0; 4]);
        assert_eq!(s.running_var, vec![1.0; 4]);
        assert_eq!(s.num_batches_tracked, 0);
    }

    // ── Forward pass correctness ──────────────────────────────

    #[test]
    fn forward_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // N=3, C=2
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let mut state = default_state(2);
        let cfg = default_config();
        let out = batch_norm_forward(&input, 2, &gamma, &beta, &mut state, &cfg).unwrap();
        // Per-channel mean should be ~0 after normalization.
        let ch0_mean: f32 = (0..3).map(|n| out[n * 2]).sum::<f32>() / 3.0;
        let ch1_mean: f32 = (0..3).map(|n| out[n * 2 + 1]).sum::<f32>() / 3.0;
        assert!(ch0_mean.abs() < TOL);
        assert!(ch1_mean.abs() < TOL);
    }

    #[test]
    fn forward_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // N=2, C=2
        let gamma = vec![2.0, 0.5];
        let beta = vec![1.0, -1.0];
        let mut state = default_state(2);
        let cfg = default_config();
        let out = batch_norm_forward(&input, 2, &gamma, &beta, &mut state, &cfg).unwrap();
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_no_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut state = default_state(2);
        let cfg = SimdBatchNormConfig { affine: false, ..default_config() };
        let out = batch_norm_forward(&input, 2, &[], &[], &mut state, &cfg).unwrap();
        // Without affine: output = (x - mean) / std
        assert_eq!(out.len(), 4);
        let ch0_mean: f32 = (0..2).map(|n| out[n * 2]).sum::<f32>() / 2.0;
        assert!(ch0_mean.abs() < TOL);
    }

    #[test]
    fn forward_uniform_input() {
        let input = vec![5.0; 8]; // N=4, C=2
        let mut state = default_state(2);
        let cfg = default_config();
        let out = batch_norm_forward(&input, 2, &[1.0; 2], &[0.0; 2], &mut state, &cfg).unwrap();
        for &v in &out {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn forward_single_sample() {
        let input = vec![3.0, 7.0]; // N=1, C=2
        let mut state = default_state(2);
        let cfg = default_config();
        let out = batch_norm_forward(&input, 2, &[1.0; 2], &[0.0; 2], &mut state, &cfg).unwrap();
        // Single sample: mean=value, var=0 → output ~0
        assert!(out[0].abs() < TOL);
        assert!(out[1].abs() < TOL);
    }

    #[test]
    fn forward_known_values() {
        // N=4, C=1, input=[1,2,3,4] → mean=2.5, var=1.25
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut state = default_state(1);
        let cfg = default_config();
        let out = batch_norm_forward(&input, 1, &[1.0], &[0.0], &mut state, &cfg).unwrap();
        let mean = 2.5_f64;
        let var = 1.25_f64;
        let inv_std = 1.0 / (var + 1e-5_f64).sqrt();
        for (i, &x) in input.iter().enumerate() {
            let expected = ((x as f64 - mean) * inv_std) as f32;
            assert!(
                (out[i] - expected).abs() < TOL,
                "idx {i}: got {}, expected {expected}",
                out[i]
            );
        }
    }

    // ── Inference mode ────────────────────────────────────────

    #[test]
    fn inference_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut state = default_state(2);
        state.running_mean = vec![2.0, 3.0];
        state.running_var = vec![1.0, 1.0];
        let cfg = default_config();
        let out = batch_norm_inference(&input, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        // Expected: (x - running_mean) / sqrt(running_var + eps)
        let eps = 1e-5_f64;
        let inv0 = 1.0 / (1.0 + eps).sqrt();
        let inv1 = 1.0 / (1.0 + eps).sqrt();
        let expected = vec![
            ((1.0 - 2.0) * inv0) as f32,
            ((2.0 - 3.0) * inv1) as f32,
            ((3.0 - 2.0) * inv0) as f32,
            ((4.0 - 3.0) * inv1) as f32,
        ];
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn inference_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![2.0, 0.5];
        let beta = vec![1.0, -1.0];
        let mut state = default_state(2);
        state.running_mean = vec![3.0, 4.0];
        state.running_var = vec![4.0, 9.0];
        let cfg = default_config();
        let out = batch_norm_inference(&input, 2, &gamma, &beta, &state, &cfg).unwrap();
        let eps = 1e-5_f64;
        let inv0 = 1.0 / (4.0_f64 + eps).sqrt();
        let inv1 = 1.0 / (9.0_f64 + eps).sqrt();
        for n in 0..3 {
            let x0 = input[n * 2] as f64;
            let x1 = input[n * 2 + 1] as f64;
            let e0 = ((x0 - 3.0) * inv0 * 2.0 + 1.0) as f32;
            let e1 = ((x1 - 4.0) * inv1 * 0.5 + (-1.0)) as f32;
            assert!((out[n * 2] - e0).abs() < TOL, "n={n} ch0: {} vs {e0}", out[n * 2]);
            assert!((out[n * 2 + 1] - e1).abs() < TOL, "n={n} ch1: {} vs {e1}", out[n * 2 + 1]);
        }
    }

    #[test]
    fn inference_identity() {
        // running_mean=0, running_var=1, gamma=1, beta=0 → identity-ish
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let state = default_state(2);
        let cfg = default_config();
        let out = batch_norm_inference(&input, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        assert!(approx_eq(&out, &input, 1e-3));
    }

    #[test]
    fn inference_no_state_update() {
        let input = vec![100.0, 200.0, 300.0, 400.0];
        let state = default_state(2);
        let cfg = default_config();
        let _ = batch_norm_inference(&input, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        // State must remain unchanged.
        assert_eq!(state.running_mean, vec![0.0; 2]);
        assert_eq!(state.running_var, vec![1.0; 2]);
    }

    // ── Training vs inference differ ──────────────────────────

    #[test]
    fn training_and_inference_differ() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut state = default_state(2);
        state.running_mean = vec![10.0, 20.0];
        state.running_var = vec![5.0, 5.0];
        let cfg = default_config();
        let i_out = batch_norm_inference(&input, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        let mut state2 = state.clone();
        let t_out = batch_norm_forward(&input, 2, &[1.0; 2], &[0.0; 2], &mut state2, &cfg).unwrap();
        assert!(!approx_eq(&t_out, &i_out, TOL));
    }

    // ── Running stats update ──────────────────────────────────

    #[test]
    fn running_mean_update() {
        let input = vec![2.0, 4.0, 6.0, 8.0]; // N=2, C=2, ch0=[2,6]→mean=4, ch1=[4,8]→mean=6
        let mut state = default_state(2);
        let cfg = default_config();
        let _ = batch_norm_forward(&input, 2, &[1.0; 2], &[0.0; 2], &mut state, &cfg).unwrap();
        // updated_mean = (1-0.1)*0 + 0.1*batch_mean
        assert!((state.running_mean[0] - 0.4).abs() < TOL);
        assert!((state.running_mean[1] - 0.6).abs() < TOL);
    }

    #[test]
    fn running_var_update() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let mut state = default_state(2);
        let cfg = default_config();
        let _ = batch_norm_forward(&input, 2, &[1.0; 2], &[0.0; 2], &mut state, &cfg).unwrap();
        // ch0 var = ((2-4)^2 + (6-4)^2) / 2 = 4
        // updated_var = 0.9*1.0 + 0.1*4.0 = 1.3
        assert!((state.running_var[0] - 1.3).abs() < TOL);
        assert!((state.running_var[1] - 1.3).abs() < TOL);
    }

    #[test]
    fn running_stats_zero_momentum() {
        let mut state = SimdBatchNormState {
            running_mean: vec![5.0, 15.0],
            running_var: vec![2.0, 3.0],
            num_batches_tracked: 0,
        };
        let cfg = SimdBatchNormConfig { momentum: 0.0, ..default_config() };
        let _ = batch_norm_forward(
            &[10.0, 20.0, 30.0, 40.0],
            2,
            &[1.0; 2],
            &[0.0; 2],
            &mut state,
            &cfg,
        )
        .unwrap();
        assert!(approx_eq(&state.running_mean, &[5.0, 15.0], TOL));
        assert!(approx_eq(&state.running_var, &[2.0, 3.0], TOL));
    }

    #[test]
    fn running_stats_full_momentum() {
        let mut state = SimdBatchNormState {
            running_mean: vec![100.0, 200.0],
            running_var: vec![50.0, 60.0],
            num_batches_tracked: 0,
        };
        let cfg = SimdBatchNormConfig { momentum: 1.0, ..default_config() };
        let _ =
            batch_norm_forward(&[2.0, 4.0, 6.0, 8.0], 2, &[1.0; 2], &[0.0; 2], &mut state, &cfg)
                .unwrap();
        // batch_mean = [4.0, 6.0], batch_var = [4.0, 4.0]
        assert!((state.running_mean[0] - 4.0).abs() < TOL);
        assert!((state.running_mean[1] - 6.0).abs() < TOL);
        assert!((state.running_var[0] - 4.0).abs() < TOL);
        assert!((state.running_var[1] - 4.0).abs() < TOL);
    }

    #[test]
    fn num_batches_tracked_increments() {
        let mut state = default_state(1);
        let cfg = default_config();
        let _ = batch_norm_forward(&[1.0, 2.0], 1, &[1.0], &[0.0], &mut state, &cfg).unwrap();
        assert_eq!(state.num_batches_tracked, 1);
        let _ = batch_norm_forward(&[3.0, 4.0], 1, &[1.0], &[0.0], &mut state, &cfg).unwrap();
        assert_eq!(state.num_batches_tracked, 2);
    }

    #[test]
    fn no_tracking_when_disabled() {
        let mut state = default_state(1);
        let cfg = SimdBatchNormConfig { track_running_stats: false, ..default_config() };
        let _ = batch_norm_forward(&[100.0, 200.0], 1, &[1.0], &[0.0], &mut state, &cfg).unwrap();
        assert_eq!(state.running_mean, vec![0.0]);
        assert_eq!(state.running_var, vec![1.0]);
        assert_eq!(state.num_batches_tracked, 0);
    }

    // ── 1D inputs ─────────────────────────────────────────────

    #[test]
    fn batch_norm_1d_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // N=3, C=2
        let state = default_state(2);
        let cfg = default_config();
        let out = batch_norm_1d(&input, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        assert_eq!(out.len(), 6);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn batch_norm_1d_matches_inference() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let state = default_state(2);
        let cfg = default_config();
        let out_1d = batch_norm_1d(&input, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        let out_inf = batch_norm_inference(&input, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        assert!(approx_eq(&out_1d, &out_inf, TOL));
    }

    // ── 2D inputs ─────────────────────────────────────────────

    #[test]
    fn batch_norm_2d_basic() {
        // N=1, C=2, H=2, W=2 → 8 elements
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let state = default_state(2);
        let cfg = default_config();
        let out = batch_norm_2d(&input, 1, 2, 2, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        assert_eq!(out.len(), 8);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn batch_norm_2d_identity() {
        // With running_mean=0, running_var=1 → approximate identity
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let state = default_state(2);
        let cfg = default_config();
        let out = batch_norm_2d(&input, 1, 2, 2, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        assert!(approx_eq(&out, &input, 1e-3));
    }

    #[test]
    fn batch_norm_2d_multi_batch() {
        // N=2, C=2, H=1, W=1 → 4 elements
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut state = default_state(2);
        state.running_mean = vec![2.0, 3.0];
        state.running_var = vec![1.0, 1.0];
        let cfg = default_config();
        let out = batch_norm_2d(&input, 2, 2, 1, 1, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        assert_eq!(out.len(), 4);
        // For H=W=1, should behave like 1D BN.
        let out_1d = batch_norm_1d(&input, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        assert!(approx_eq(&out, &out_1d, TOL));
    }

    #[test]
    fn batch_norm_2d_wrong_size() {
        let state = default_state(2);
        let cfg = default_config();
        assert!(batch_norm_2d(&[1.0; 7], 1, 2, 2, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).is_err());
    }

    // ── Fused operations ──────────────────────────────────────

    #[test]
    fn fused_bn_relu_positive_values() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let state = default_state(2);
        let cfg = default_config();
        let out = fused_bn_relu(&input, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        // All positive inputs with identity-like BN → positive outputs
        for &v in &out {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn fused_bn_relu_clips_negatives() {
        // Use large negative beta to force negative BN output
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let state = default_state(2);
        let cfg = default_config();
        let out = fused_bn_relu(&input, 2, &[1.0; 2], &[-10.0; 2], &state, &cfg).unwrap();
        for &v in &out {
            assert!((v - 0.0).abs() < TOL, "expected 0 after ReLU, got {v}");
        }
    }

    #[test]
    fn fused_bn_relu_matches_manual() {
        let input = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        let state = default_state(2);
        let cfg = default_config();
        let fused = fused_bn_relu(&input, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        let manual = batch_norm_inference(&input, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        let manual_relu: Vec<f32> = manual.iter().map(|&v| v.max(0.0)).collect();
        assert!(approx_eq(&fused, &manual_relu, TOL));
    }

    #[test]
    fn fused_bn_add_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![10.0, 20.0, 30.0, 40.0];
        let state = default_state(2);
        let cfg = default_config();
        let out = fused_bn_add(&input, &residual, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        let bn = batch_norm_inference(&input, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        let expected: Vec<f32> = bn.iter().zip(residual.iter()).map(|(b, r)| b + r).collect();
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn fused_bn_add_length_mismatch() {
        let state = default_state(2);
        let cfg = default_config();
        assert!(fused_bn_add(&[1.0; 4], &[1.0; 6], 2, &[1.0; 2], &[0.0; 2], &state, &cfg).is_err());
    }

    #[test]
    fn fused_bn_add_zero_residual() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.0; 4];
        let state = default_state(2);
        let cfg = default_config();
        let out = fused_bn_add(&input, &residual, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        let bn = batch_norm_inference(&input, 2, &[1.0; 2], &[0.0; 2], &state, &cfg).unwrap();
        assert!(approx_eq(&out, &bn, TOL));
    }

    // ── Group normalization ───────────────────────────────────

    #[test]
    fn group_norm_basic() {
        // N=1, C=4, spatial=2, groups=2
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let out = group_norm(&input, 2, 4, 2, Some(&gamma), Some(&beta), 1e-5).unwrap();
        assert_eq!(out.len(), 8);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn group_norm_no_affine() {
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let out = group_norm(&input, 2, 4, 2, None, None, 1e-5).unwrap();
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn group_norm_single_group() {
        // Single group = layer norm style
        let input = vec![1.0, 2.0, 3.0, 4.0]; // N=1, C=2, sp=2
        let out = group_norm(&input, 1, 2, 2, None, None, 1e-5).unwrap();
        let mean = compute_mean(&input);
        let var = compute_variance(&input, mean);
        let inv = 1.0 / (var + 1e-5).sqrt();
        let expected: Vec<f32> = input.iter().map(|&x| (x - mean) * inv).collect();
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn group_norm_per_group_mean_zero() {
        // N=1, C=4, sp=1, groups=2 → group0=[1,2], group1=[3,4]
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = group_norm(&input, 2, 4, 1, None, None, 1e-5).unwrap();
        // Each group's output mean should be ~0
        let g0_mean = (out[0] + out[1]) / 2.0;
        let g1_mean = (out[2] + out[3]) / 2.0;
        assert!(g0_mean.abs() < TOL);
        assert!(g1_mean.abs() < TOL);
    }

    #[test]
    fn group_norm_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // N=1, C=4, sp=1, groups=2
        let gamma = vec![2.0, 2.0, 0.5, 0.5];
        let beta = vec![1.0, 1.0, -1.0, -1.0];
        let out = group_norm(&input, 2, 4, 1, Some(&gamma), Some(&beta), 1e-5).unwrap();
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn group_norm_channels_not_divisible() {
        assert!(group_norm(&[1.0; 6], 3, 5, 1, None, None, 1e-5).is_err());
    }

    #[test]
    fn group_norm_zero_groups() {
        assert!(group_norm(&[1.0; 4], 0, 4, 1, None, None, 1e-5).is_err());
    }

    #[test]
    fn group_norm_multi_batch() {
        // N=2, C=4, sp=1, groups=2
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = group_norm(&input, 2, 4, 1, None, None, 1e-5).unwrap();
        assert_eq!(out.len(), 8);
        // Each batch-group pair should have mean ~0
        let g0_b0 = (out[0] + out[1]) / 2.0;
        let g0_b1 = (out[4] + out[5]) / 2.0;
        assert!(g0_b0.abs() < TOL);
        assert!(g0_b1.abs() < TOL);
    }

    // ── Instance normalization ────────────────────────────────

    #[test]
    fn instance_norm_basic() {
        // N=1, C=2, sp=3
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = instance_norm(&input, 2, 3, None, None, 1e-5).unwrap();
        assert_eq!(out.len(), 6);
        // Each channel is independently normalized.
        let ch0_mean = (out[0] + out[1] + out[2]) / 3.0;
        let ch1_mean = (out[3] + out[4] + out[5]) / 3.0;
        assert!(ch0_mean.abs() < TOL);
        assert!(ch1_mean.abs() < TOL);
    }

    #[test]
    fn instance_norm_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![2.0, 0.5];
        let beta = vec![1.0, -1.0];
        let out = instance_norm(&input, 2, 3, Some(&gamma), Some(&beta), 1e-5).unwrap();
        assert_eq!(out.len(), 6);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn instance_norm_is_group_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gn = group_norm(&input, 2, 2, 3, None, None, 1e-5).unwrap();
        let in_ = instance_norm(&input, 2, 3, None, None, 1e-5).unwrap();
        assert!(approx_eq(&gn, &in_, TOL));
    }

    // ── Edge cases ────────────────────────────────────────────

    #[test]
    fn forward_all_zeros() {
        let mut state = default_state(2);
        let cfg = default_config();
        let out = batch_norm_forward(&[0.0; 8], 2, &[1.0; 2], &[0.0; 2], &mut state, &cfg).unwrap();
        assert!(out.iter().all(|&v| v.abs() < TOL));
    }

    #[test]
    fn forward_single_element() {
        let mut state = default_state(1);
        let cfg = default_config();
        let out = batch_norm_forward(&[42.0], 1, &[1.0], &[0.0], &mut state, &cfg).unwrap();
        // Single element: mean=42, var=0 → ~0
        assert!(out[0].abs() < 1e-2);
    }

    #[test]
    fn inference_zero_variance() {
        // running_var=0 → rely on epsilon
        let mut state = default_state(1);
        state.running_var = vec![0.0];
        let cfg = default_config();
        let out = batch_norm_inference(&[5.0, 10.0], 1, &[1.0], &[0.0], &state, &cfg).unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_large_values() {
        let mut state = default_state(1);
        let cfg = default_config();
        let input = vec![1e30, 1e30 + 1.0, 1e30 + 2.0, 1e30 + 3.0];
        let out = batch_norm_forward(&input, 1, &[1.0], &[0.0], &mut state, &cfg).unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_small_values() {
        let mut state = default_state(1);
        let cfg = default_config();
        let input = vec![1e-30, 2e-30, 3e-30, 4e-30];
        let out = batch_norm_forward(&input, 1, &[1.0], &[0.0], &mut state, &cfg).unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_subnormal() {
        let mut state = default_state(1);
        let cfg = default_config();
        let tiny = f32::MIN_POSITIVE / 2.0;
        let input = vec![tiny, tiny * 2.0, tiny * 3.0, tiny * 4.0];
        let out = batch_norm_forward(&input, 1, &[1.0], &[0.0], &mut state, &cfg).unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    // ── SIMD path correctness ─────────────────────────────────

    #[test]
    fn compute_mean_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!((compute_mean(&data) - 2.5).abs() < TOL);
    }

    #[test]
    fn compute_mean_empty() {
        assert_eq!(compute_mean(&[]), 0.0);
    }

    #[test]
    fn compute_mean_single() {
        assert!((compute_mean(&[42.0]) - 42.0).abs() < TOL);
    }

    #[test]
    fn compute_mean_large_vector() {
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let expected = 511.5;
        assert!((compute_mean(&data) - expected).abs() < 0.01);
    }

    #[test]
    fn compute_mean_non_aligned() {
        // Length not a multiple of 8 → tests scalar tail
        let data: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let expected = 6.0;
        assert!((compute_mean(&data) - expected).abs() < TOL);
    }

    #[test]
    fn compute_variance_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mean = compute_mean(&data);
        let var = compute_variance(&data, mean);
        assert!((var - 1.25).abs() < TOL);
    }

    #[test]
    fn compute_variance_empty() {
        assert_eq!(compute_variance(&[], 0.0), 0.0);
    }

    #[test]
    fn compute_variance_constant() {
        let data = vec![5.0; 16];
        let mean = compute_mean(&data);
        let var = compute_variance(&data, mean);
        assert!(var.abs() < TOL);
    }

    #[test]
    fn compute_variance_large_vector() {
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let mean = compute_mean(&data);
        let var = compute_variance(&data, mean);
        // Var of 0..1023 = (1023*1024*(2*1023+1)/6)/1024 - mean^2
        let n = 1024.0_f64;
        let expected = (n - 1.0) * (n + 1.0) / 12.0; // ~87381.25
        assert!((var as f64 - expected).abs() / expected < 1e-4);
    }

    #[test]
    fn normalize_and_scale_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        normalize_and_scale(&data, 2.5, 1.0, None, None, &mut out);
        let expected = vec![-1.5, -0.5, 0.5, 1.5];
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn normalize_and_scale_with_affine() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0; 4];
        let beta = vec![1.0; 4];
        let mut out = vec![0.0; 4];
        normalize_and_scale(&data, 2.5, 1.0, Some(&gamma), Some(&beta), &mut out);
        let expected: Vec<f32> = data.iter().map(|&x| (x - 2.5) * 2.0 + 1.0).collect();
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn normalize_and_scale_large_vector() {
        let n = 1025; // Not a multiple of 8
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mean = compute_mean(&data);
        let var = compute_variance(&data, mean);
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        let mut out = vec![0.0f32; n];
        normalize_and_scale(&data, mean, inv_std, None, None, &mut out);
        // Output mean should be ~0
        let out_mean: f32 = out.iter().sum::<f32>() / n as f32;
        assert!(out_mean.abs() < 0.01);
    }

    // ── Error cases ───────────────────────────────────────────

    #[test]
    fn forward_empty_input() {
        let mut state = default_state(2);
        let cfg = default_config();
        assert!(batch_norm_forward(&[], 2, &[1.0; 2], &[0.0; 2], &mut state, &cfg).is_err());
    }

    #[test]
    fn forward_zero_features() {
        let mut state = default_state(0);
        let cfg = default_config();
        assert!(batch_norm_forward(&[1.0], 0, &[], &[], &mut state, &cfg).is_err());
    }

    #[test]
    fn forward_gamma_mismatch() {
        let mut state = default_state(2);
        let cfg = default_config();
        assert!(batch_norm_forward(&[1.0, 2.0], 2, &[1.0], &[0.0; 2], &mut state, &cfg).is_err());
    }

    #[test]
    fn forward_beta_mismatch() {
        let mut state = default_state(2);
        let cfg = default_config();
        assert!(batch_norm_forward(&[1.0, 2.0], 2, &[1.0; 2], &[0.0], &mut state, &cfg).is_err());
    }

    #[test]
    fn forward_input_not_multiple() {
        let mut state = default_state(3);
        let cfg = default_config();
        assert!(
            batch_norm_forward(&[1.0, 2.0], 3, &[1.0; 3], &[0.0; 3], &mut state, &cfg).is_err()
        );
    }

    #[test]
    fn forward_zero_eps() {
        let mut state = default_state(1);
        let cfg = SimdBatchNormConfig { epsilon: 0.0, ..default_config() };
        assert!(batch_norm_forward(&[1.0], 1, &[1.0], &[0.0], &mut state, &cfg).is_err());
    }

    #[test]
    fn forward_negative_eps() {
        let mut state = default_state(1);
        let cfg = SimdBatchNormConfig { epsilon: -1e-5, ..default_config() };
        assert!(batch_norm_forward(&[1.0], 1, &[1.0], &[0.0], &mut state, &cfg).is_err());
    }

    #[test]
    fn forward_invalid_momentum() {
        let mut state = default_state(1);
        let cfg = SimdBatchNormConfig { momentum: 1.5, ..default_config() };
        assert!(batch_norm_forward(&[1.0], 1, &[1.0], &[0.0], &mut state, &cfg).is_err());
    }

    #[test]
    fn forward_state_dimension_mismatch() {
        let mut state = default_state(3); // 3 != 2
        let cfg = default_config();
        assert!(
            batch_norm_forward(&[1.0, 2.0], 2, &[1.0; 2], &[0.0; 2], &mut state, &cfg).is_err()
        );
    }

    #[test]
    fn inference_empty_input() {
        let state = default_state(1);
        let cfg = default_config();
        assert!(batch_norm_inference(&[], 1, &[1.0], &[0.0], &state, &cfg).is_err());
    }

    #[test]
    fn group_norm_empty_input() {
        assert!(group_norm(&[], 2, 4, 1, None, None, 1e-5).is_err());
    }

    #[test]
    fn group_norm_gamma_mismatch() {
        assert!(group_norm(&[1.0; 4], 2, 4, 1, Some(&[1.0; 2]), None, 1e-5).is_err());
    }

    #[test]
    fn group_norm_zero_eps() {
        assert!(group_norm(&[1.0; 4], 2, 4, 1, None, None, 0.0).is_err());
    }

    // ── Larger scale ──────────────────────────────────────────

    #[test]
    fn forward_large_batch() {
        let c = 4;
        let n = 256;
        let mut state = default_state(c);
        let cfg = default_config();
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1).collect();
        let out =
            batch_norm_forward(&input, c, &vec![1.0; c], &vec![0.0; c], &mut state, &cfg).unwrap();
        assert_eq!(out.len(), n * c);
        for ch in 0..c {
            let ch_mean: f32 = (0..n).map(|i| out[i * c + ch]).sum::<f32>() / n as f32;
            assert!(ch_mean.abs() < 1e-3, "ch {ch}: mean={ch_mean}");
        }
    }

    #[test]
    fn inference_large_batch() {
        let c = 4;
        let n = 256;
        let mut state = default_state(c);
        state.running_mean = vec![1.0, 2.0, 3.0, 4.0];
        state.running_var = vec![2.0; c];
        let cfg = default_config();
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.01).collect();
        let out =
            batch_norm_inference(&input, c, &vec![1.0; c], &vec![0.0; c], &state, &cfg).unwrap();
        assert_eq!(out.len(), n * c);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn group_norm_large_spatial() {
        // N=1, C=8, sp=128, groups=4
        let n_elem = 8 * 128;
        let input: Vec<f32> = (0..n_elem).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let out = group_norm(&input, 4, 8, 128, None, None, 1e-5).unwrap();
        assert_eq!(out.len(), n_elem);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    // ── Momentum convergence ──────────────────────────────────

    #[test]
    fn momentum_convergence() {
        let mut state = default_state(1);
        let cfg = default_config();
        let input = vec![2.0, 4.0, 6.0, 8.0]; // mean=5, var=5
        for _ in 0..100 {
            let _ = batch_norm_forward(&input, 1, &[1.0], &[0.0], &mut state, &cfg).unwrap();
        }
        assert!((state.running_mean[0] - 5.0).abs() < 1e-3);
        assert!((state.running_var[0] - 5.0).abs() < 0.1);
    }

    // ── Property tests ────────────────────────────────────────

    mod prop {
        use super::*;
        use proptest::prelude::*;

        fn bn_scenario() -> impl Strategy<Value = (usize, usize, Vec<f32>)> {
            (1..=64_usize, 1..=16_usize).prop_flat_map(|(batch, features)| {
                let len = batch * features;
                (Just(batch), Just(features), proptest::collection::vec(-100.0_f32..100.0, len))
            })
        }

        proptest! {
            #[test]
            fn prop_forward_output_mean_near_zero(
                (batch, features, input) in bn_scenario()
            ) {
                let mut state = SimdBatchNormState::new(features);
                let cfg = SimdBatchNormConfig::new();
                let out = batch_norm_forward(
                    &input, features, &vec![1.0; features], &vec![0.0; features],
                    &mut state, &cfg,
                ).unwrap();
                for ch in 0..features {
                    let ch_mean: f32 = (0..batch).map(|b| out[b * features + ch]).sum::<f32>() / batch as f32;
                    prop_assert!(ch_mean.abs() < 1e-3, "ch {ch}: mean={ch_mean}");
                }
            }

            #[test]
            fn prop_output_finite(
                (_batch, features, input) in bn_scenario()
            ) {
                let mut state = SimdBatchNormState::new(features);
                let cfg = SimdBatchNormConfig::new();
                let out = batch_norm_forward(
                    &input, features, &vec![1.0; features], &vec![0.0; features],
                    &mut state, &cfg,
                ).unwrap();
                prop_assert!(out.iter().all(|v| v.is_finite()));
                prop_assert!(state.running_mean.iter().all(|v| v.is_finite()));
                prop_assert!(state.running_var.iter().all(|v| v.is_finite()));
            }

            #[test]
            fn prop_running_var_non_negative(
                (_batch, features, input) in bn_scenario()
            ) {
                let mut state = SimdBatchNormState::new(features);
                let cfg = SimdBatchNormConfig::new();
                let _ = batch_norm_forward(
                    &input, features, &vec![1.0; features], &vec![0.0; features],
                    &mut state, &cfg,
                ).unwrap();
                for ch in 0..features {
                    prop_assert!(state.running_var[ch] >= 0.0);
                }
            }

            #[test]
            fn prop_inference_deterministic(
                (_batch, features, input) in bn_scenario()
            ) {
                let state = SimdBatchNormState::new(features);
                let cfg = SimdBatchNormConfig::new();
                let g = vec![1.0f32; features];
                let b = vec![0.0f32; features];
                let out1 = batch_norm_inference(&input, features, &g, &b, &state, &cfg).unwrap();
                let out2 = batch_norm_inference(&input, features, &g, &b, &state, &cfg).unwrap();
                prop_assert_eq!(&out1, &out2);
            }

            #[test]
            fn prop_compute_mean_matches_scalar(
                data in proptest::collection::vec(-1000.0_f32..1000.0, 1..=512)
            ) {
                let simd_result = compute_mean(&data);
                let scalar_result = compute_mean_scalar(&data);
                prop_assert!((simd_result - scalar_result).abs() < 1e-3,
                    "simd={simd_result} scalar={scalar_result}");
            }

            #[test]
            fn prop_compute_variance_matches_scalar(
                data in proptest::collection::vec(-1000.0_f32..1000.0, 2..=512)
            ) {
                let mean = compute_mean_scalar(&data);
                let simd_result = compute_variance(&data, mean);
                let scalar_result = compute_variance_scalar(&data, mean);
                let tol = scalar_result.abs() * 1e-3 + 1e-3;
                prop_assert!((simd_result - scalar_result).abs() < tol,
                    "simd={simd_result} scalar={scalar_result}");
            }
        }
    }
}
