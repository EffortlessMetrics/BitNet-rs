//! ARM NEON batch normalization v2 kernels for Apple Silicon.
//!
//! Extends the original NEON batch norm with:
//! - Running mean/variance update (training vs inference modes)
//! - Fused batch norm + activation (ReLU, SiLU)
//! - Group normalization variant
//! - Instance normalization variant
//!
//! Input layout is `[N, C, ...]` (batch × channels × spatial).
//! Processes 4 × f32 NEON lanes with scalar tail fallback.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc, clippy::float_cmp, clippy::manual_div_ceil, clippy::unnecessary_cast, clippy::needless_range_loop, clippy::too_many_arguments, clippy::collapsible_if, clippy::let_and_return, clippy::derivable_impls, clippy::excessive_precision, clippy::manual_is_multiple_of)]
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Configuration ─────────────────────────────────────────────────

/// Fused activation applied after normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormActivation {
    /// No activation — identity pass-through.
    None,
    /// Rectified linear unit: max(0, x).
    Relu,
    /// Sigmoid linear unit: x · σ(x).
    Silu,
}

/// Statistics mode for normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatsMode {
    /// Training: compute batch statistics and update running stats.
    Training,
    /// Inference: use pre-computed running mean/variance.
    Inference,
}

/// Configuration for batch normalization v2.
#[derive(Debug, Clone)]
pub struct BatchNormV2Config {
    /// Number of feature channels.
    pub num_features: usize,
    /// Small constant for numerical stability (default 1e-5).
    pub eps: f32,
    /// Exponential moving average momentum (default 0.1).
    pub momentum: f32,
    /// Statistics mode: training or inference.
    pub stats_mode: StatsMode,
    /// Optional fused activation.
    pub activation: NormActivation,
}

impl BatchNormV2Config {
    /// Create config for the given number of features with defaults.
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            stats_mode: StatsMode::Inference,
            activation: NormActivation::None,
        }
    }

    /// Builder: set statistics mode.
    #[must_use]
    pub fn with_stats_mode(mut self, mode: StatsMode) -> Self {
        self.stats_mode = mode;
        self
    }

    /// Builder: set fused activation.
    #[must_use]
    pub fn with_activation(mut self, act: NormActivation) -> Self {
        self.activation = act;
        self
    }
}

/// Output from [`neon_batch_norm_v2`].
#[derive(Debug)]
pub struct BatchNormV2Output {
    /// Normalized (and optionally activated) output.
    pub output: Vec<f32>,
    /// Updated running mean (only modified in training mode).
    pub running_mean: Vec<f32>,
    /// Updated running variance (only modified in training mode).
    pub running_var: Vec<f32>,
}

/// Configuration for group normalization.
#[derive(Debug, Clone)]
pub struct GroupNormConfig {
    /// Number of feature channels.
    pub num_channels: usize,
    /// Number of groups (must divide `num_channels`).
    pub num_groups: usize,
    /// Numerical stability constant (default 1e-5).
    pub eps: f32,
    /// Optional fused activation.
    pub activation: NormActivation,
}

impl GroupNormConfig {
    pub fn new(num_channels: usize, num_groups: usize) -> Self {
        Self { num_channels, num_groups, eps: 1e-5, activation: NormActivation::None }
    }
}

/// Configuration for instance normalization v2.
#[derive(Debug, Clone)]
pub struct InstanceNormV2Config {
    /// Number of feature channels.
    pub num_channels: usize,
    /// Numerical stability constant (default 1e-5).
    pub eps: f32,
    /// Statistics mode: training or inference.
    pub stats_mode: StatsMode,
    /// EMA momentum for running stats (default 0.1).
    pub momentum: f32,
    /// Optional fused activation.
    pub activation: NormActivation,
}

impl InstanceNormV2Config {
    pub fn new(num_channels: usize) -> Self {
        Self {
            num_channels,
            eps: 1e-5,
            stats_mode: StatsMode::Inference,
            momentum: 0.1,
            activation: NormActivation::None,
        }
    }
}

// ── Batch Normalization v2 ────────────────────────────────────────

/// NEON-accelerated batch normalization with running statistics.
///
/// Input shape: flat `[N, C]` where `N = input.len() / num_features`.
///
/// In **training** mode the batch mean/variance are computed from the
/// input and the running statistics are updated via exponential moving
/// average.  In **inference** mode the supplied running statistics are
/// used directly.
///
/// Returns the normalized output together with (possibly updated)
/// running mean and variance.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics on length mismatches or if `input.len()` is not divisible by
/// `num_features`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_batch_norm_v2(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    config: &BatchNormV2Config,
) -> BatchNormV2Output {
    let c = config.num_features;
    assert!(c > 0, "num_features must be > 0");
    assert_eq!(gamma.len(), c, "gamma length mismatch");
    assert_eq!(beta.len(), c, "beta length mismatch");
    assert_eq!(running_mean.len(), c, "running_mean length mismatch");
    assert_eq!(running_var.len(), c, "running_var length mismatch");
    assert_eq!(input.len() % c, 0, "input length not divisible by num_features");

    let batch_size = input.len() / c;
    let mut output = vec![0.0f32; input.len()];

    let (mean, var, new_rmean, new_rvar) = match config.stats_mode {
        StatsMode::Training => {
            let (bm, bv) = compute_batch_stats(input, batch_size, c);
            let new_rm = ema_update(running_mean, &bm, config.momentum);
            let new_rv = ema_update(running_var, &bv, config.momentum);
            (bm, bv, new_rm, new_rv)
        }
        StatsMode::Inference => {
            let m = running_mean.to_vec();
            let v = running_var.to_vec();
            let rm = running_mean.to_vec();
            let rv = running_var.to_vec();
            (m, v, rm, rv)
        }
    };

    // SAFETY: inside target_feature(enable = "neon") function.
    unsafe {
        neon_bn_v2_apply(
            input,
            &mut output,
            gamma,
            beta,
            &mean,
            &var,
            config.eps,
            config.activation,
            batch_size,
            c,
        );
    }

    BatchNormV2Output { output, running_mean: new_rmean, running_var: new_rvar }
}

// ── Group Normalization ───────────────────────────────────────────

/// NEON-accelerated group normalization.
///
/// Input shape: flat `[N, C, spatial...]` where
/// `spatial = input.len() / (batch_size * num_channels)`.
///
/// Channels are split into `num_groups` equal groups.  Each group is
/// normalised independently over its channels and spatial dims.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics on length mismatches or if `num_channels` is not divisible by
/// `num_groups`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_group_norm(
    input: &[f32],
    batch_size: usize,
    gamma: &[f32],
    beta: &[f32],
    config: &GroupNormConfig,
) -> Vec<f32> {
    let c = config.num_channels;
    let g = config.num_groups;
    assert!(g > 0, "num_groups must be > 0");
    assert_eq!(c % g, 0, "num_channels must be divisible by num_groups");
    assert_eq!(gamma.len(), c, "gamma length mismatch");
    assert_eq!(beta.len(), c, "beta length mismatch");
    let total = input.len();
    assert_eq!(total % (batch_size * c), 0, "input not divisible by batch*channels");

    let spatial = total / (batch_size * c);
    let channels_per_group = c / g;
    let group_size = channels_per_group * spatial;
    let mut output = vec![0.0f32; total];

    // SAFETY: inside target_feature(enable = "neon") function.
    unsafe {
        for n in 0..batch_size {
            let sample_off = n * c * spatial;
            for grp in 0..g {
                let grp_off = sample_off + grp * group_size;
                let grp_data = &input[grp_off..grp_off + group_size];

                let mean = neon_reduce_sum(grp_data) / group_size as f32;
                let var = neon_reduce_sum_sq_diff(grp_data, mean) / group_size as f32;
                let inv_std = 1.0 / (var + config.eps).sqrt();

                for ch_local in 0..channels_per_group {
                    let ch_global = grp * channels_per_group + ch_local;
                    let ch_off = grp_off + ch_local * spatial;
                    let src = &input[ch_off..ch_off + spatial];
                    let dst = &mut output[ch_off..ch_off + spatial];
                    neon_normalize_affine_bcast(
                        src,
                        dst,
                        gamma[ch_global],
                        beta[ch_global],
                        mean,
                        inv_std,
                        config.activation,
                    );
                }
            }
        }
    }

    output
}

// ── Instance Normalization v2 ─────────────────────────────────────

/// NEON-accelerated instance normalization with running statistics.
///
/// Input shape: flat `[N, C, H, W]`.  Each `(n, c)` slice is normalised
/// independently over the spatial dimensions.
///
/// In **training** mode the per-channel running stats are updated from
/// the spatial mean/variance averaged over the batch.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics on length mismatches or zero spatial size.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_instance_norm_v2(
    input: &[f32],
    batch_size: usize,
    height: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    config: &InstanceNormV2Config,
) -> BatchNormV2Output {
    let c = config.num_channels;
    let spatial = height * width;
    assert!(spatial > 0, "spatial dimensions must be non-zero");
    assert_eq!(input.len(), batch_size * c * spatial, "input length mismatch");
    assert_eq!(gamma.len(), c, "gamma length mismatch");
    assert_eq!(beta.len(), c, "beta length mismatch");
    assert_eq!(running_mean.len(), c, "running_mean length mismatch");
    assert_eq!(running_var.len(), c, "running_var length mismatch");

    let mut output = vec![0.0f32; input.len()];

    let (new_rmean, new_rvar) = match config.stats_mode {
        StatsMode::Training => {
            // Accumulate per-channel mean/var across the batch.
            let mut ch_mean_acc = vec![0.0f32; c];
            let mut ch_var_acc = vec![0.0f32; c];

            // SAFETY: inside target_feature(enable = "neon") function.
            unsafe {
                for n in 0..batch_size {
                    for ch in 0..c {
                        let off = (n * c + ch) * spatial;
                        let src = &input[off..off + spatial];
                        let dst = &mut output[off..off + spatial];

                        let m = neon_reduce_sum(src) / spatial as f32;
                        let v = neon_reduce_sum_sq_diff(src, m) / spatial as f32;
                        ch_mean_acc[ch] += m;
                        ch_var_acc[ch] += v;
                        let inv_std = 1.0 / (v + config.eps).sqrt();
                        neon_normalize_affine_bcast(
                            src,
                            dst,
                            gamma[ch],
                            beta[ch],
                            m,
                            inv_std,
                            config.activation,
                        );
                    }
                }
            }

            // Average across batch.
            let bs = batch_size as f32;
            for v in &mut ch_mean_acc {
                *v /= bs;
            }
            for v in &mut ch_var_acc {
                *v /= bs;
            }

            let new_rm = ema_update(running_mean, &ch_mean_acc, config.momentum);
            let new_rv = ema_update(running_var, &ch_var_acc, config.momentum);
            (new_rm, new_rv)
        }
        StatsMode::Inference => {
            // SAFETY: inside target_feature(enable = "neon") function.
            unsafe {
                for n in 0..batch_size {
                    for ch in 0..c {
                        let off = (n * c + ch) * spatial;
                        let src = &input[off..off + spatial];
                        let dst = &mut output[off..off + spatial];
                        let inv_std = 1.0 / (running_var[ch] + config.eps).sqrt();
                        neon_normalize_affine_bcast(
                            src,
                            dst,
                            gamma[ch],
                            beta[ch],
                            running_mean[ch],
                            inv_std,
                            config.activation,
                        );
                    }
                }
            }

            (running_mean.to_vec(), running_var.to_vec())
        }
    };

    BatchNormV2Output { output, running_mean: new_rmean, running_var: new_rvar }
}

// ── NEON helpers ──────────────────────────────────────────────────

/// Compute per-channel batch mean and variance from `[N, C]` flat input.
fn compute_batch_stats(
    input: &[f32],
    batch_size: usize,
    num_features: usize,
) -> (Vec<f32>, Vec<f32>) {
    let c = num_features;
    let mut mean = vec![0.0f64; c];
    let mut var = vec![0.0f64; c];
    let count = batch_size as f64;

    for n in 0..batch_size {
        for ch in 0..c {
            mean[ch] += input[n * c + ch] as f64;
        }
    }
    for m in &mut mean {
        *m /= count;
    }

    for n in 0..batch_size {
        for ch in 0..c {
            let d = input[n * c + ch] as f64 - mean[ch];
            var[ch] += d * d;
        }
    }
    for v in &mut var {
        *v /= count;
    }

    (mean.iter().map(|&v| v as f32).collect(), var.iter().map(|&v| v as f32).collect())
}

/// Exponential moving average: `new = (1 - momentum) * old + momentum * batch`.
fn ema_update(running: &[f32], batch: &[f32], momentum: f32) -> Vec<f32> {
    running.iter().zip(batch.iter()).map(|(&r, &b)| (1.0 - momentum) * r + momentum * b).collect()
}

/// Apply batch norm + optional activation across `[N, C]` layout via NEON.
///
/// # Safety
///
/// Must be called from a NEON-enabled context.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_bn_v2_apply(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
    activation: NormActivation,
    batch_size: usize,
    num_features: usize,
) {
    let c = num_features;
    // Pre-compute inv_std per channel.
    let inv_std: Vec<f32> = var.iter().map(|&v| 1.0 / (v + eps).sqrt()).collect();

    // SAFETY: inside target_feature(enable = "neon") function.
    unsafe {
        for n in 0..batch_size {
            let row_off = n * c;
            let chunks = c / 4;
            let remainder = c % 4;

            for i in 0..chunks {
                let off = row_off + i * 4;
                let x = vld1q_f32(input.as_ptr().add(off));
                let g = vld1q_f32(gamma.as_ptr().add(i * 4));
                let b = vld1q_f32(beta.as_ptr().add(i * 4));
                let m = vld1q_f32(mean.as_ptr().add(i * 4));
                let is = vld1q_f32(inv_std.as_ptr().add(i * 4));

                let centered = vsubq_f32(x, m);
                let normed = vmulq_f32(centered, is);
                let scaled = vaddq_f32(vmulq_f32(g, normed), b);
                let activated = neon_apply_activation(scaled, activation);
                vst1q_f32(output.as_mut_ptr().add(off), activated);
            }

            let tail = row_off + chunks * 4;
            for i in 0..remainder {
                let idx = tail + i;
                let ch = chunks * 4 + i;
                let v = gamma[ch] * (input[idx] - mean[ch]) * inv_std[ch] + beta[ch];
                output[idx] = scalar_activate(v, activation);
            }
        }
    }
}

/// Apply fused activation to a NEON f32x4 register.
///
/// # Safety
///
/// Must be called from a NEON-enabled context.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_apply_activation(x: float32x4_t, activation: NormActivation) -> float32x4_t {
    match activation {
        NormActivation::None => x,
        NormActivation::Relu => {
            // SAFETY: inside NEON context; vdupq/vmaxq require neon.
            unsafe {
                let zero = vdupq_n_f32(0.0);
                vmaxq_f32(x, zero)
            }
        }
        NormActivation::Silu => {
            // SiLU(x) = x * sigmoid(x) ≈ x * 1/(1 + exp(-x))
            // Use scalar fallback extracted per lane for exp().
            // SAFETY: inside NEON context; lane extraction is safe.
            unsafe {
                let mut buf = [0.0f32; 4];
                vst1q_f32(buf.as_mut_ptr(), x);
                for v in &mut buf {
                    let sig = 1.0 / (1.0 + (-*v).exp());
                    *v *= sig;
                }
                vld1q_f32(buf.as_ptr())
            }
        }
    }
}

/// Scalar activation (for remainder elements).
#[inline(always)]
fn scalar_activate(x: f32, activation: NormActivation) -> f32 {
    match activation {
        NormActivation::None => x,
        NormActivation::Relu => x.max(0.0),
        NormActivation::Silu => {
            let sig = 1.0 / (1.0 + (-x).exp());
            x * sig
        }
    }
}

/// NEON-accelerated horizontal sum.
///
/// # Safety
///
/// Must be called from a NEON-enabled context.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_reduce_sum(data: &[f32]) -> f32 {
    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: inside target_feature(enable = "neon") function.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            acc = vaddq_f32(acc, v);
        }
        let mut sum: f32 = vaddvq_f32(acc);
        let tail = chunks * 4;
        for i in 0..remainder {
            sum += data[tail + i];
        }
        sum
    }
}

/// NEON-accelerated sum of squared differences from `center`.
///
/// # Safety
///
/// Must be called from a NEON-enabled context.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_reduce_sum_sq_diff(data: &[f32], center: f32) -> f32 {
    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: inside target_feature(enable = "neon") function.
    unsafe {
        let c_vec = vdupq_n_f32(center);
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            let diff = vsubq_f32(v, c_vec);
            acc = vfmaq_f32(acc, diff, diff);
        }
        let mut total: f32 = vaddvq_f32(acc);
        let tail = chunks * 4;
        for i in 0..remainder {
            let d = data[tail + i] - center;
            total += d * d;
        }
        total
    }
}

/// Normalise with broadcast scalar affine + optional fused activation.
///
/// `output[i] = act(gamma * (input[i] - mean) * inv_std + beta)`
///
/// # Safety
///
/// Must be called from a NEON-enabled context.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_normalize_affine_bcast(
    input: &[f32],
    output: &mut [f32],
    gamma: f32,
    beta: f32,
    mean: f32,
    inv_std: f32,
    activation: NormActivation,
) {
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: inside target_feature(enable = "neon") function.
    unsafe {
        let m_vec = vdupq_n_f32(mean);
        let is_vec = vdupq_n_f32(inv_std);
        let g_vec = vdupq_n_f32(gamma);
        let b_vec = vdupq_n_f32(beta);

        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(input.as_ptr().add(off));
            let centered = vsubq_f32(x, m_vec);
            let normed = vmulq_f32(centered, is_vec);
            let scaled = vaddq_f32(vmulq_f32(g_vec, normed), b_vec);
            let activated = neon_apply_activation(scaled, activation);
            vst1q_f32(output.as_mut_ptr().add(off), activated);
        }
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        let idx = tail + i;
        let v = gamma * (input[idx] - mean) * inv_std + beta;
        output[idx] = scalar_activate(v, activation);
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-4;

    fn assert_approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "idx {i}: {x} vs {y} (diff {})", (x - y).abs());
        }
    }

    /// Scalar reference batch norm for verification.
    fn scalar_bn(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mean: &[f32],
        var: &[f32],
        eps: f32,
        activation: NormActivation,
        batch_size: usize,
        num_features: usize,
    ) -> Vec<f32> {
        let c = num_features;
        let mut out = vec![0.0f32; input.len()];
        for n in 0..batch_size {
            for ch in 0..c {
                let idx = n * c + ch;
                let inv_std = 1.0 / (var[ch] + eps).sqrt();
                let v = gamma[ch] * (input[idx] - mean[ch]) * inv_std + beta[ch];
                out[idx] = scalar_activate(v, activation);
            }
        }
        out
    }

    /// Scalar reference group norm for verification.
    fn scalar_group_norm(
        input: &[f32],
        batch_size: usize,
        gamma: &[f32],
        beta: &[f32],
        num_channels: usize,
        num_groups: usize,
        spatial: usize,
        eps: f32,
        activation: NormActivation,
    ) -> Vec<f32> {
        let cpg = num_channels / num_groups;
        let gs = cpg * spatial;
        let mut out = vec![0.0f32; input.len()];
        for n in 0..batch_size {
            let s_off = n * num_channels * spatial;
            for g in 0..num_groups {
                let g_off = s_off + g * gs;
                let grp = &input[g_off..g_off + gs];
                let mean: f32 = grp.iter().sum::<f32>() / gs as f32;
                let var: f32 = grp.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / gs as f32;
                let inv_std = 1.0 / (var + eps).sqrt();
                for cl in 0..cpg {
                    let cg = g * cpg + cl;
                    let ch_off = g_off + cl * spatial;
                    for s in 0..spatial {
                        let v = gamma[cg] * (input[ch_off + s] - mean) * inv_std + beta[cg];
                        out[ch_off + s] = scalar_activate(v, activation);
                    }
                }
            }
        }
        out
    }

    // ── Batch Norm v2 tests ───────────────────────────────────────

    #[test]
    fn test_bn_v2_inference_identity() {
        let c = 4;
        let n = 2;
        let input: Vec<f32> = (0..n * c).map(|i| i as f32).collect();
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let mean = vec![0.0; c];
        let var = vec![1.0; c];
        let config = BatchNormV2Config::new(c);

        let result = unsafe { neon_batch_norm_v2(&input, &gamma, &beta, &mean, &var, &config) };

        let expected =
            scalar_bn(&input, &gamma, &beta, &mean, &var, EPS, NormActivation::None, n, c);
        assert_approx(&result.output, &expected, TOL);
    }

    #[test]
    fn test_bn_v2_inference_affine() {
        let c = 6;
        let n = 3;
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let gamma: Vec<f32> = (0..c).map(|i| 0.5 + i as f32 * 0.2).collect();
        let beta: Vec<f32> = (0..c).map(|i| -0.1 + i as f32 * 0.05).collect();
        let mean: Vec<f32> = (0..c).map(|i| i as f32 * 0.3 - 1.0).collect();
        let var: Vec<f32> = (0..c).map(|i| 0.5 + i as f32 * 0.4).collect();
        let config = BatchNormV2Config::new(c);

        let result = unsafe { neon_batch_norm_v2(&input, &gamma, &beta, &mean, &var, &config) };

        let expected =
            scalar_bn(&input, &gamma, &beta, &mean, &var, EPS, NormActivation::None, n, c);
        assert_approx(&result.output, &expected, TOL);
    }

    #[test]
    fn test_bn_v2_training_updates_running_stats() {
        let c = 4;
        let n = 8;
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let rmean = vec![0.0; c];
        let rvar = vec![1.0; c];
        let config = BatchNormV2Config::new(c).with_stats_mode(StatsMode::Training);

        let result = unsafe { neon_batch_norm_v2(&input, &gamma, &beta, &rmean, &rvar, &config) };

        // Running stats should have moved towards batch stats.
        assert_ne!(result.running_mean, rmean);
        assert_ne!(result.running_var, rvar);
    }

    #[test]
    fn test_bn_v2_inference_preserves_running_stats() {
        let c = 4;
        let n = 2;
        let input: Vec<f32> = (0..n * c).map(|i| i as f32).collect();
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let rmean = vec![1.0, 2.0, 3.0, 4.0];
        let rvar = vec![0.5, 1.0, 1.5, 2.0];
        let config = BatchNormV2Config::new(c);

        let result = unsafe { neon_batch_norm_v2(&input, &gamma, &beta, &rmean, &rvar, &config) };

        assert_eq!(result.running_mean, rmean);
        assert_eq!(result.running_var, rvar);
    }

    // ── Fused activation tests ────────────────────────────────────

    #[test]
    fn test_bn_v2_fused_relu() {
        let c = 8;
        let n = 2;
        // Mix of values that will be positive and negative after norm.
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) - 7.0).collect();
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let mean = vec![0.0; c];
        let var = vec![1.0; c];
        let config = BatchNormV2Config::new(c).with_activation(NormActivation::Relu);

        let result = unsafe { neon_batch_norm_v2(&input, &gamma, &beta, &mean, &var, &config) };

        let expected =
            scalar_bn(&input, &gamma, &beta, &mean, &var, EPS, NormActivation::Relu, n, c);
        assert_approx(&result.output, &expected, TOL);
        // Verify all outputs are non-negative (ReLU property).
        for &v in &result.output {
            assert!(v >= 0.0, "ReLU output must be >= 0, got {v}");
        }
    }

    #[test]
    fn test_bn_v2_fused_silu() {
        let c = 8;
        let n = 2;
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let mean = vec![0.0; c];
        let var = vec![1.0; c];
        let config = BatchNormV2Config::new(c).with_activation(NormActivation::Silu);

        let result = unsafe { neon_batch_norm_v2(&input, &gamma, &beta, &mean, &var, &config) };

        let expected =
            scalar_bn(&input, &gamma, &beta, &mean, &var, EPS, NormActivation::Silu, n, c);
        assert_approx(&result.output, &expected, TOL);
    }

    // ── Group normalization tests ─────────────────────────────────

    #[test]
    fn test_group_norm_single_group() {
        // Single group = LayerNorm-like behaviour.
        let c = 4;
        let bs = 2;
        let spatial = 3;
        let input: Vec<f32> = (0..bs * c * spatial).map(|i| i as f32 * 0.1).collect();
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let config = GroupNormConfig::new(c, 1);

        let result = unsafe { neon_group_norm(&input, bs, &gamma, &beta, &config) };

        let expected =
            scalar_group_norm(&input, bs, &gamma, &beta, c, 1, spatial, EPS, NormActivation::None);
        assert_approx(&result, &expected, TOL);
    }

    #[test]
    fn test_group_norm_multiple_groups() {
        let c = 8;
        let g = 2;
        let bs = 2;
        let spatial = 4;
        let input: Vec<f32> = (0..bs * c * spatial).map(|i| (i as f32) * 0.05 - 1.5).collect();
        let gamma: Vec<f32> = (0..c).map(|i| 0.8 + i as f32 * 0.05).collect();
        let beta: Vec<f32> = (0..c).map(|i| -0.1 + i as f32 * 0.02).collect();
        let config = GroupNormConfig::new(c, g);

        let result = unsafe { neon_group_norm(&input, bs, &gamma, &beta, &config) };

        let expected =
            scalar_group_norm(&input, bs, &gamma, &beta, c, g, spatial, EPS, NormActivation::None);
        assert_approx(&result, &expected, TOL);
    }

    #[test]
    fn test_group_norm_with_relu() {
        let c = 4;
        let g = 2;
        let bs = 1;
        let spatial = 6;
        let input: Vec<f32> = (0..bs * c * spatial).map(|i| (i as f32) - 12.0).collect();
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let config = GroupNormConfig {
            num_channels: c,
            num_groups: g,
            eps: EPS,
            activation: NormActivation::Relu,
        };

        let result = unsafe { neon_group_norm(&input, bs, &gamma, &beta, &config) };

        let expected =
            scalar_group_norm(&input, bs, &gamma, &beta, c, g, spatial, EPS, NormActivation::Relu);
        assert_approx(&result, &expected, TOL);
        for &v in &result {
            assert!(v >= 0.0, "ReLU output must be >= 0, got {v}");
        }
    }

    // ── Instance normalization v2 tests ───────────────────────────

    #[test]
    fn test_instance_norm_v2_inference() {
        let c = 2;
        let bs = 2;
        let (h, w) = (3, 3);
        let spatial = h * w;
        let input: Vec<f32> = (0..bs * c * spatial).map(|i| i as f32 * 0.1).collect();
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let rmean = vec![1.0, 2.0];
        let rvar = vec![0.5, 1.5];
        let config = InstanceNormV2Config::new(c);

        let result = unsafe {
            neon_instance_norm_v2(&input, bs, h, w, &gamma, &beta, &rmean, &rvar, &config)
        };

        // Verify shape.
        assert_eq!(result.output.len(), input.len());
        // Running stats unchanged in inference mode.
        assert_eq!(result.running_mean, rmean);
        assert_eq!(result.running_var, rvar);
    }

    #[test]
    fn test_instance_norm_v2_training() {
        let c = 2;
        let bs = 4;
        let (h, w) = (2, 2);
        let spatial = h * w;
        let input: Vec<f32> = (0..bs * c * spatial).map(|i| (i as f32) * 0.2 - 3.0).collect();
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let rmean = vec![0.0; c];
        let rvar = vec![1.0; c];
        let mut config = InstanceNormV2Config::new(c);
        config.stats_mode = StatsMode::Training;

        let result = unsafe {
            neon_instance_norm_v2(&input, bs, h, w, &gamma, &beta, &rmean, &rvar, &config)
        };

        assert_eq!(result.output.len(), input.len());
        // Running stats should have been updated.
        assert_ne!(result.running_mean, rmean);
    }

    #[test]
    fn test_instance_norm_v2_with_silu() {
        let c = 2;
        let bs = 1;
        let (h, w) = (2, 3);
        let spatial = h * w;
        let input: Vec<f32> = (0..bs * c * spatial).map(|i| i as f32 * 0.5 - 2.0).collect();
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let rmean = vec![0.5, 1.5];
        let rvar = vec![1.0, 2.0];
        let config = InstanceNormV2Config {
            num_channels: c,
            eps: EPS,
            stats_mode: StatsMode::Inference,
            momentum: 0.1,
            activation: NormActivation::Silu,
        };

        let result = unsafe {
            neon_instance_norm_v2(&input, bs, h, w, &gamma, &beta, &rmean, &rvar, &config)
        };

        assert_eq!(result.output.len(), input.len());
    }

    // ── Non-aligned / edge-case tests ─────────────────────────────

    #[test]
    fn test_bn_v2_non_aligned_channels() {
        // 7 channels → 1 NEON chunk + 3 scalar remainder.
        let c = 7;
        let n = 3;
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.3 - 2.5).collect();
        let gamma: Vec<f32> = (0..c).map(|i| 0.5 + i as f32 * 0.1).collect();
        let beta: Vec<f32> = (0..c).map(|i| -0.2 + i as f32 * 0.05).collect();
        let mean: Vec<f32> = (0..c).map(|i| i as f32 * 0.2 - 0.5).collect();
        let var: Vec<f32> = (0..c).map(|i| 0.3 + i as f32 * 0.3).collect();
        let config = BatchNormV2Config::new(c);

        let result = unsafe { neon_batch_norm_v2(&input, &gamma, &beta, &mean, &var, &config) };

        let expected =
            scalar_bn(&input, &gamma, &beta, &mean, &var, EPS, NormActivation::None, n, c);
        assert_approx(&result.output, &expected, TOL);
    }

    #[test]
    fn test_bn_v2_single_element_batch() {
        let c = 4;
        let n = 1;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let mean = vec![0.0; c];
        let var = vec![1.0; c];
        let config = BatchNormV2Config::new(c);

        let result = unsafe { neon_batch_norm_v2(&input, &gamma, &beta, &mean, &var, &config) };

        let expected =
            scalar_bn(&input, &gamma, &beta, &mean, &var, EPS, NormActivation::None, n, c);
        assert_approx(&result.output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_non_aligned_spatial() {
        // 5 spatial elements → 1 NEON chunk + 1 scalar remainder.
        let c = 4;
        let g = 2;
        let bs = 1;
        let spatial = 5;
        let input: Vec<f32> = (0..bs * c * spatial).map(|i| i as f32 * 0.2 - 1.0).collect();
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let config = GroupNormConfig::new(c, g);

        let result = unsafe { neon_group_norm(&input, bs, &gamma, &beta, &config) };

        let expected =
            scalar_group_norm(&input, bs, &gamma, &beta, c, g, spatial, EPS, NormActivation::None);
        assert_approx(&result, &expected, TOL);
    }

    // ── EMA update tests ──────────────────────────────────────────

    #[test]
    fn test_ema_update_basic() {
        let running = vec![0.0, 0.0, 0.0];
        let batch = vec![1.0, 2.0, 3.0];
        let result = ema_update(&running, &batch, 0.1);
        assert_approx(&result, &[0.1, 0.2, 0.3], 1e-6);
    }

    #[test]
    fn test_ema_update_convergence() {
        let mut running = vec![0.0];
        let batch = vec![10.0];
        for _ in 0..100 {
            running = ema_update(&running, &batch, 0.1);
        }
        // Should converge to batch value.
        assert!((running[0] - 10.0).abs() < 0.01);
    }

    // ── Activation scalar tests ───────────────────────────────────

    #[test]
    fn test_scalar_relu() {
        assert_eq!(scalar_activate(5.0, NormActivation::Relu), 5.0);
        assert_eq!(scalar_activate(-3.0, NormActivation::Relu), 0.0);
        assert_eq!(scalar_activate(0.0, NormActivation::Relu), 0.0);
    }

    #[test]
    fn test_scalar_silu() {
        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!((scalar_activate(0.0, NormActivation::Silu)).abs() < 1e-6);
        // SiLU(x) > 0 for x > 0
        assert!(scalar_activate(2.0, NormActivation::Silu) > 0.0);
        // SiLU(x) < 0 for some x < 0
        assert!(scalar_activate(-1.0, NormActivation::Silu) < 0.0);
    }

    #[test]
    fn test_scalar_none() {
        assert_eq!(scalar_activate(42.0, NormActivation::None), 42.0);
    }

    // ── Ignored tests (require hardware benchmarking) ─────────────

    #[test]
    #[ignore = "benchmark: requires large tensor allocation for perf measurement"]
    fn test_bn_v2_large_tensor_perf() {
        let c = 512;
        let n = 256;
        let input = vec![1.0f32; n * c];
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let mean = vec![0.0; c];
        let var = vec![1.0; c];
        let config = BatchNormV2Config::new(c).with_activation(NormActivation::Relu);

        let _result = unsafe { neon_batch_norm_v2(&input, &gamma, &beta, &mean, &var, &config) };
    }

    #[test]
    #[ignore = "benchmark: group norm with many groups for perf profiling"]
    fn test_group_norm_many_groups_perf() {
        let c = 256;
        let g = 32;
        let bs = 16;
        let spatial = 64;
        let input = vec![0.5f32; bs * c * spatial];
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let config = GroupNormConfig::new(c, g);

        let _result = unsafe { neon_group_norm(&input, bs, &gamma, &beta, &config) };
    }
}
