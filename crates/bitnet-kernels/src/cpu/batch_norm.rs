//! CPU batch normalization kernel.
//!
//! Provides batch normalization for 1D, 2D (NCHW), and 3D inputs on
//! contiguous `f32` slices with optional learnable affine parameters
//! (gamma/beta), running statistics tracking, and backward pass for
//! gradient computation.

use bitnet_common::{BitNetError, KernelError, Result};

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Configuration ──────────────────────────────────────────────────

/// Configuration for batch normalization.
#[derive(Debug, Clone)]
pub struct BatchNormConfig {
    /// Number of features (channels).
    pub num_features: usize,
    /// Small constant added to variance for numerical stability.
    pub eps: f32,
    /// Momentum for running mean/variance update
    /// (`new = (1-momentum)*old + momentum*batch`).
    pub momentum: f32,
    /// Whether to apply learnable affine parameters (gamma/beta).
    pub affine: bool,
    /// Whether to track running mean/variance across batches.
    pub track_running_stats: bool,
}

impl BatchNormConfig {
    /// Convenience constructor with default eps (1e-5), momentum (0.1),
    /// affine enabled, and running stats tracking enabled.
    #[must_use]
    pub fn new(num_features: usize) -> Self {
        Self { num_features, eps: 1e-5, momentum: 0.1, affine: true, track_running_stats: true }
    }
}

impl Default for BatchNormConfig {
    fn default() -> Self {
        Self { num_features: 1, eps: 1e-5, momentum: 0.1, affine: true, track_running_stats: true }
    }
}

// ── Forward ────────────────────────────────────────────────────────

/// Compute batch normalization forward pass.
///
/// Returns `(output, updated_running_mean, updated_running_var)`.
///
/// When `training` is true, normalizes using batch statistics and updates
/// the running mean/variance (if `config.track_running_stats` is true and
/// running stats are provided). When `training` is false, normalizes using
/// the provided running statistics.
///
/// When `config.affine` is false, gamma/beta are ignored and the identity
/// transform (gamma=1, beta=0) is used.
///
/// Input is a flat buffer in `[N, C]` channel-interleaved order where
/// `N = input.len() / num_features`. For 2D/3D data (NCHW), pre-flatten
/// spatial dims into the batch dimension.
pub fn batch_norm_forward(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: Option<&[f32]>,
    running_var: Option<&[f32]>,
    config: &BatchNormConfig,
    training: bool,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let c = config.num_features;
    validate_common_args(input, c, config.eps)?;
    validate_momentum(config.momentum)?;
    if config.affine {
        validate_param_len(gamma, c, "gamma")?;
        validate_param_len(beta, c, "beta")?;
    }

    let batch_size = input.len() / c;

    let ones = vec![1.0f32; c];
    let zeros = vec![0.0f32; c];
    let eff_gamma = if config.affine { gamma } else { &ones };
    let eff_beta = if config.affine { beta } else { &zeros };

    if training {
        // Compute batch statistics.
        let (batch_mean, batch_var) = compute_batch_stats(input, c)?;

        // Normalize using batch statistics.
        let output = normalize(
            input,
            eff_gamma,
            eff_beta,
            &batch_mean,
            &batch_var,
            config.eps,
            c,
            batch_size,
        );

        // Update running stats.
        let (updated_mean, updated_var) = if config.track_running_stats {
            update_running_stats(
                &batch_mean,
                &batch_var,
                running_mean,
                running_var,
                config.momentum,
                c,
            )?
        } else {
            (batch_mean, batch_var)
        };

        Ok((output, updated_mean, updated_var))
    } else {
        // Inference mode — require running stats.
        let rm = running_mean.ok_or_else(|| invalid_args("running_mean required in eval mode"))?;
        let rv = running_var.ok_or_else(|| invalid_args("running_var required in eval mode"))?;
        validate_param_len(rm, c, "running_mean")?;
        validate_param_len(rv, c, "running_var")?;

        let rm_f32: Vec<f32> = rm.to_vec();
        let rv_f32: Vec<f32> = rv.to_vec();

        let output =
            normalize(input, eff_gamma, eff_beta, &rm_f32, &rv_f32, config.eps, c, batch_size);

        Ok((output, rm_f32, rv_f32))
    }
}

// ── Backward ───────────────────────────────────────────────────────

/// Compute batch normalization backward pass.
///
/// Returns `(grad_input, grad_gamma, grad_beta)`.
///
/// `mean` and `var` are the per-channel batch statistics from the forward
/// pass (as returned by [`compute_batch_stats`]).
pub fn batch_norm_backward(
    grad_output: &[f32],
    input: &[f32],
    gamma: &[f32],
    mean: &[f32],
    var: &[f32],
    config: &BatchNormConfig,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let c = config.num_features;
    validate_common_args(input, c, config.eps)?;
    if grad_output.len() != input.len() {
        return Err(invalid_args(&format!(
            "grad_output length {} != input length {}",
            grad_output.len(),
            input.len()
        )));
    }
    validate_param_len(gamma, c, "gamma")?;
    validate_param_len(mean, c, "mean")?;
    validate_param_len(var, c, "var")?;

    let batch_size = input.len() / c;
    let n_f64 = batch_size as f64;

    let mut grad_gamma = vec![0.0f32; c];
    let mut grad_beta = vec![0.0f32; c];
    let mut grad_input = vec![0.0f32; input.len()];

    // Pre-compute inv_std per channel.
    let inv_std: Vec<f64> =
        (0..c).map(|ch| 1.0 / (var[ch] as f64 + config.eps as f64).sqrt()).collect();

    // Accumulate grad_gamma, grad_beta, and intermediate sums.
    let mut sum_dy = vec![0.0f64; c];
    let mut sum_dy_xhat = vec![0.0f64; c];

    for n in 0..batch_size {
        for ch in 0..c {
            let idx = n * c + ch;
            let x_hat = (input[idx] as f64 - mean[ch] as f64) * inv_std[ch];
            let dy = grad_output[idx] as f64;

            grad_beta[ch] += grad_output[idx];
            grad_gamma[ch] += (dy * x_hat) as f32;

            sum_dy[ch] += dy;
            sum_dy_xhat[ch] += dy * x_hat;
        }
    }

    // Compute grad_input.
    for n in 0..batch_size {
        for ch in 0..c {
            let idx = n * c + ch;
            let x_hat = (input[idx] as f64 - mean[ch] as f64) * inv_std[ch];
            let dy = grad_output[idx] as f64;

            let di = gamma[ch] as f64 * inv_std[ch] / n_f64
                * (n_f64 * dy - sum_dy[ch] - x_hat * sum_dy_xhat[ch]);
            grad_input[idx] = di as f32;
        }
    }

    Ok((grad_input, grad_gamma, grad_beta))
}

// ── Batch statistics ───────────────────────────────────────────────

/// Compute per-channel mean and variance from a batch.
///
/// Input is in `[N, C]` layout. Returns `(mean, variance)` vectors each
/// of length `num_features`.
pub fn compute_batch_stats(input: &[f32], num_features: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    if num_features == 0 {
        return Err(invalid_args("num_features must be > 0"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if !input.len().is_multiple_of(num_features) {
        return Err(invalid_args("input length must be a multiple of num_features"));
    }

    let batch_size = input.len() / num_features;
    let count = batch_size as f64;

    let mut mean = vec![0.0f64; num_features];
    for n in 0..batch_size {
        for ch in 0..num_features {
            mean[ch] += input[n * num_features + ch] as f64;
        }
    }
    for m in &mut mean {
        *m /= count;
    }

    let mut var = vec![0.0f64; num_features];
    for n in 0..batch_size {
        for ch in 0..num_features {
            let d = input[n * num_features + ch] as f64 - mean[ch];
            var[ch] += d * d;
        }
    }
    for v in &mut var {
        *v /= count;
    }

    let mean_f32: Vec<f32> = mean.iter().map(|&m| m as f32).collect();
    let var_f32: Vec<f32> = var.iter().map(|&v| v as f32).collect();
    Ok((mean_f32, var_f32))
}

// ── Internal helpers ───────────────────────────────────────────────

fn validate_common_args(input: &[f32], c: usize, eps: f32) -> Result<()> {
    if c == 0 {
        return Err(invalid_args("num_features must be > 0"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if eps <= 0.0 || !eps.is_finite() {
        return Err(invalid_args("eps must be positive and finite"));
    }
    if !input.len().is_multiple_of(c) {
        return Err(invalid_args("input length must be a multiple of num_features"));
    }
    Ok(())
}

fn validate_momentum(momentum: f32) -> Result<()> {
    if !momentum.is_finite() || !(0.0..=1.0).contains(&momentum) {
        return Err(invalid_args("momentum must be in [0, 1] and finite"));
    }
    Ok(())
}

fn validate_param_len(param: &[f32], expected: usize, name: &str) -> Result<()> {
    if param.len() != expected {
        return Err(invalid_args(&format!(
            "{name} length {} != num_features {expected}",
            param.len()
        )));
    }
    Ok(())
}

fn normalize(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
    c: usize,
    batch_size: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for n in 0..batch_size {
        for ch in 0..c {
            let inv_std = 1.0 / (var[ch] as f64 + eps as f64).sqrt();
            let x_hat = (input[n * c + ch] as f64 - mean[ch] as f64) * inv_std;
            output[n * c + ch] = (gamma[ch] as f64 * x_hat + beta[ch] as f64) as f32;
        }
    }
    output
}

fn update_running_stats(
    batch_mean: &[f32],
    batch_var: &[f32],
    running_mean: Option<&[f32]>,
    running_var: Option<&[f32]>,
    momentum: f32,
    c: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mom = momentum as f64;
    match (running_mean, running_var) {
        (Some(rm), Some(rv)) => {
            validate_param_len(rm, c, "running_mean")?;
            validate_param_len(rv, c, "running_var")?;
            let mut um = vec![0.0f32; c];
            let mut uv = vec![0.0f32; c];
            for ch in 0..c {
                um[ch] = ((1.0 - mom) * rm[ch] as f64 + mom * batch_mean[ch] as f64) as f32;
                uv[ch] = ((1.0 - mom) * rv[ch] as f64 + mom * batch_var[ch] as f64) as f32;
            }
            Ok((um, uv))
        }
        // No prior running stats — initialise from batch.
        _ => Ok((batch_mean.to_vec(), batch_var.to_vec())),
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    fn reference_bn(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mean: &[f32],
        var: &[f32],
        eps: f64,
    ) -> Vec<f32> {
        let c = gamma.len();
        let n = input.len() / c;
        let mut output = vec![0.0f32; input.len()];
        for i in 0..n {
            for ch in 0..c {
                let inv_std = 1.0 / (var[ch] as f64 + eps).sqrt();
                let x_hat = (input[i * c + ch] as f64 - mean[ch] as f64) * inv_std;
                output[i * c + ch] = (gamma[ch] as f64 * x_hat + beta[ch] as f64) as f32;
            }
        }
        output
    }

    fn default_cfg(c: usize) -> BatchNormConfig {
        BatchNormConfig::new(c)
    }

    // ── Config ─────────────────────────────────────────────

    #[test]
    fn config_default() {
        let c = BatchNormConfig::default();
        assert_eq!(c.num_features, 1);
        assert!((c.eps - 1e-5).abs() < 1e-10);
        assert!((c.momentum - 0.1).abs() < 1e-10);
        assert!(c.affine);
        assert!(c.track_running_stats);
    }

    #[test]
    fn config_new() {
        let c = BatchNormConfig::new(64);
        assert_eq!(c.num_features, 64);
        assert!(c.affine);
        assert!(c.track_running_stats);
    }

    // ── Forward (training) ─────────────────────────────────

    #[test]
    fn forward_basic_1d() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cfg = default_cfg(2);
        let (out, _, _) = batch_norm_forward(
            &input,
            &[1.0; 2],
            &[0.0; 2],
            Some(&[0.0; 2]),
            Some(&[1.0; 2]),
            &cfg,
            true,
        )
        .unwrap();
        let ch0_mean: f32 = (0..3).map(|n| out[n * 2]).sum::<f32>() / 3.0;
        let ch1_mean: f32 = (0..3).map(|n| out[n * 2 + 1]).sum::<f32>() / 3.0;
        assert!(ch0_mean.abs() < TOL);
        assert!(ch1_mean.abs() < TOL);
    }

    #[test]
    fn forward_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = default_cfg(2);
        let (out, _, _) = batch_norm_forward(
            &input,
            &[2.0, 0.5],
            &[1.0, -1.0],
            Some(&[0.0; 2]),
            Some(&[1.0; 2]),
            &cfg,
            true,
        )
        .unwrap();
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_uniform_input() {
        let input = vec![5.0; 8];
        let cfg = default_cfg(2);
        let (out, _, _) = batch_norm_forward(
            &input,
            &[1.0; 2],
            &[0.0; 2],
            Some(&[0.0; 2]),
            Some(&[1.0; 2]),
            &cfg,
            true,
        )
        .unwrap();
        for &v in &out {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn forward_single_sample() {
        let input = vec![3.0, 7.0];
        let cfg = default_cfg(2);
        let (out, _, _) = batch_norm_forward(
            &input,
            &[1.0; 2],
            &[0.0; 2],
            Some(&[0.0; 2]),
            Some(&[1.0; 2]),
            &cfg,
            true,
        )
        .unwrap();
        assert!(out[0].abs() < TOL);
        assert!(out[1].abs() < TOL);
    }

    // ── Running stats ──────────────────────────────────────

    #[test]
    fn forward_running_mean_update() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = default_cfg(2);
        let (_, um, _) = batch_norm_forward(
            &input,
            &[1.0; 2],
            &[0.0; 2],
            Some(&[0.0; 2]),
            Some(&[1.0; 2]),
            &cfg,
            true,
        )
        .unwrap();
        assert!((um[0] - 0.4).abs() < TOL);
        assert!((um[1] - 0.6).abs() < TOL);
    }

    #[test]
    fn forward_running_var_update() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = default_cfg(2);
        let (_, _, uv) = batch_norm_forward(
            &input,
            &[1.0; 2],
            &[0.0; 2],
            Some(&[0.0; 2]),
            Some(&[1.0; 2]),
            &cfg,
            true,
        )
        .unwrap();
        // batch var for each channel: ((2-4)^2+(6-4)^2)/2 = 4
        // running_var update: (1-0.1)*1.0 + 0.1*4.0 = 1.3
        assert!((uv[0] - 1.3).abs() < TOL);
        assert!((uv[1] - 1.3).abs() < TOL);
    }

    #[test]
    fn forward_zero_momentum_preserves_stats() {
        let rm = vec![5.0, 15.0];
        let rv = vec![2.0, 3.0];
        let mut cfg = default_cfg(2);
        cfg.momentum = 0.0;
        let (_, um, uv) = batch_norm_forward(
            &[10.0, 20.0, 30.0, 40.0],
            &[1.0; 2],
            &[0.0; 2],
            Some(&rm),
            Some(&rv),
            &cfg,
            true,
        )
        .unwrap();
        assert!(approx_eq(&um, &rm, TOL));
        assert!(approx_eq(&uv, &rv, TOL));
    }

    #[test]
    fn forward_full_momentum_uses_batch_stats() {
        let mut cfg = default_cfg(2);
        cfg.momentum = 1.0;
        let (_, um, uv) = batch_norm_forward(
            &[2.0, 4.0, 6.0, 8.0],
            &[1.0; 2],
            &[0.0; 2],
            Some(&[100.0, 200.0]),
            Some(&[50.0, 60.0]),
            &cfg,
            true,
        )
        .unwrap();
        // batch_mean = [4.0, 6.0], batch_var = [4.0, 4.0]
        assert!((um[0] - 4.0).abs() < TOL);
        assert!((um[1] - 6.0).abs() < TOL);
        assert!((uv[0] - 4.0).abs() < TOL);
        assert!((uv[1] - 4.0).abs() < TOL);
    }

    #[test]
    fn forward_no_running_stats_training() {
        let cfg = default_cfg(2);
        let (out, bm, bv) =
            batch_norm_forward(&[1.0, 2.0, 3.0, 4.0], &[1.0; 2], &[0.0; 2], None, None, &cfg, true)
                .unwrap();
        // Should succeed — batch stats initialise from batch.
        assert_eq!(out.len(), 4);
        assert_eq!(bm.len(), 2);
        assert_eq!(bv.len(), 2);
    }

    #[test]
    fn forward_track_running_stats_false_skips_update() {
        let mut cfg = default_cfg(2);
        cfg.track_running_stats = false;
        let (_, bm, _bv) = batch_norm_forward(
            &[2.0, 4.0, 6.0, 8.0],
            &[1.0; 2],
            &[0.0; 2],
            Some(&[0.0; 2]),
            Some(&[1.0; 2]),
            &cfg,
            true,
        )
        .unwrap();
        // Returns raw batch stats, not the momentum-blended update.
        assert!((bm[0] - 4.0).abs() < TOL);
        assert!((bm[1] - 6.0).abs() < TOL);
    }

    // ── Inference (eval mode) ──────────────────────────────

    #[test]
    fn inference_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let rm = vec![2.0, 3.0];
        let rv = vec![1.0, 1.0];
        let cfg = default_cfg(2);
        let (out, _, _) =
            batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], Some(&rm), Some(&rv), &cfg, false)
                .unwrap();
        let exp = reference_bn(&input, &[1.0; 2], &[0.0; 2], &rm, &rv, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn inference_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![2.0, 0.5];
        let beta = vec![1.0, -1.0];
        let rm = vec![3.0, 4.0];
        let rv = vec![4.0, 9.0];
        let cfg = default_cfg(2);
        let (out, _, _) =
            batch_norm_forward(&input, &gamma, &beta, Some(&rm), Some(&rv), &cfg, false).unwrap();
        let exp = reference_bn(&input, &gamma, &beta, &rm, &rv, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn inference_identity_transform() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = default_cfg(2);
        let (out, _, _) = batch_norm_forward(
            &input,
            &[1.0; 2],
            &[0.0; 2],
            Some(&[0.0; 2]),
            Some(&[1.0; 2]),
            &cfg,
            false,
        )
        .unwrap();
        assert!(approx_eq(&out, &input, 1e-3));
    }

    #[test]
    fn inference_requires_running_stats() {
        let cfg = default_cfg(2);
        let res = batch_norm_forward(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], None, None, &cfg, false);
        assert!(res.is_err());
    }

    // ── Training vs inference ──────────────────────────────

    #[test]
    fn training_and_inference_differ() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rm = vec![10.0, 20.0];
        let rv = vec![5.0, 5.0];
        let cfg = default_cfg(2);
        let (t_out, _, _) =
            batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], Some(&rm), Some(&rv), &cfg, true)
                .unwrap();
        let (i_out, _, _) =
            batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], Some(&rm), Some(&rv), &cfg, false)
                .unwrap();
        assert!(!approx_eq(&t_out, &i_out, TOL));
    }

    // ── Affine disabled ────────────────────────────────────

    #[test]
    fn forward_affine_disabled() {
        let mut cfg = default_cfg(2);
        cfg.affine = false;
        let (out, _, _) = batch_norm_forward(
            &[1.0, 2.0, 3.0, 4.0],
            &[],
            &[], // ignored
            Some(&[0.0; 2]),
            Some(&[1.0; 2]),
            &cfg,
            true,
        )
        .unwrap();
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    // ── Numerical stability ────────────────────────────────

    #[test]
    fn forward_large_values() {
        let input = vec![1e6, 1e6 + 1.0, 1e6, 1e6 + 1.0];
        let cfg = default_cfg(2);
        let (out, _, _) = batch_norm_forward(
            &input,
            &[1.0; 2],
            &[0.0; 2],
            Some(&[0.0; 2]),
            Some(&[1.0; 2]),
            &cfg,
            true,
        )
        .unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_no_nan_or_inf() {
        let input = vec![1e10, -1e10, 0.0, 1e-10, 1e10, -1e10, 0.0, 1e-10];
        let cfg = default_cfg(4);
        let (out, um, uv) = batch_norm_forward(
            &input,
            &[1.0; 4],
            &[0.0; 4],
            Some(&[0.0; 4]),
            Some(&[1.0; 4]),
            &cfg,
            true,
        )
        .unwrap();
        for &v in out.iter().chain(um.iter()).chain(uv.iter()) {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn inference_no_nan_or_inf() {
        let input = vec![1e10, -1e10, 0.0, 1e-10];
        let cfg = default_cfg(2);
        let (out, _, _) = batch_norm_forward(
            &input,
            &[1.0; 2],
            &[0.0; 2],
            Some(&[0.0; 2]),
            Some(&[1.0; 2]),
            &cfg,
            false,
        )
        .unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    // ── 2D (NCHW) input ───────────────────────────────────

    #[test]
    fn forward_2d_nchw() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cfg = default_cfg(2);
        let (out, _, _) = batch_norm_forward(
            &input,
            &[1.0; 2],
            &[0.0; 2],
            Some(&[0.0; 2]),
            Some(&[1.0; 2]),
            &cfg,
            true,
        )
        .unwrap();
        assert_eq!(out.len(), 8);
        let ch0_mean: f32 = (0..4).map(|n| out[n * 2]).sum::<f32>() / 4.0;
        let ch1_mean: f32 = (0..4).map(|n| out[n * 2 + 1]).sum::<f32>() / 4.0;
        assert!(ch0_mean.abs() < TOL);
        assert!(ch1_mean.abs() < TOL);
    }

    #[test]
    fn forward_3d_input() {
        let input: Vec<f32> = (1..=18).map(|i| i as f32).collect();
        let cfg = default_cfg(3);
        let (out, _, _) = batch_norm_forward(
            &input,
            &[1.0; 3],
            &[0.0; 3],
            Some(&[0.0; 3]),
            Some(&[1.0; 3]),
            &cfg,
            true,
        )
        .unwrap();
        assert_eq!(out.len(), 18);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    // ── Backward pass ──────────────────────────────────────

    #[test]
    fn backward_grad_beta_equals_sum_grad_output() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let grad_output = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let (mean, var) = compute_batch_stats(&input, 2).unwrap();
        let cfg = default_cfg(2);
        let (_, _, grad_beta) =
            batch_norm_backward(&grad_output, &input, &[1.0; 2], &mean, &var, &cfg).unwrap();
        // grad_beta[ch] = sum of grad_output for that channel.
        let expected_gb0 = 0.1 + 0.3 + 0.5;
        let expected_gb1 = 0.2 + 0.4 + 0.6;
        assert!((grad_beta[0] - expected_gb0).abs() < TOL);
        assert!((grad_beta[1] - expected_gb1).abs() < TOL);
    }

    #[test]
    fn backward_zero_grad_output_gives_zero_grads() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (mean, var) = compute_batch_stats(&input, 2).unwrap();
        let cfg = default_cfg(2);
        let (gi, gg, gb) =
            batch_norm_backward(&[0.0; 4], &input, &[1.0; 2], &mean, &var, &cfg).unwrap();
        for &v in gi.iter().chain(gg.iter()).chain(gb.iter()) {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn backward_numerical_gradient_check() {
        // Finite-difference gradient check for a small batch.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.5, 0.8];
        let c = 2;
        let cfg = default_cfg(c);
        let (mean, var) = compute_batch_stats(&input, c).unwrap();

        // Forward pass to get output.
        let (output, _, _) = batch_norm_forward(
            &input,
            &gamma,
            &[0.0; 2],
            Some(&[0.0; 2]),
            Some(&[1.0; 2]),
            &cfg,
            true,
        )
        .unwrap();

        // Use output as grad_output for chain-rule simplicity.
        let grad_output = vec![1.0f32; output.len()];
        let (grad_input, _, _) =
            batch_norm_backward(&grad_output, &input, &gamma, &mean, &var, &cfg).unwrap();

        // Numerical gradient via forward finite difference.
        let eps = 1e-3;
        for i in 0..input.len() {
            let mut inp_plus = input.clone();
            let mut inp_minus = input.clone();
            inp_plus[i] += eps;
            inp_minus[i] -= eps;

            let (out_plus, _, _) = batch_norm_forward(
                &inp_plus,
                &gamma,
                &[0.0; 2],
                Some(&[0.0; 2]),
                Some(&[1.0; 2]),
                &cfg,
                true,
            )
            .unwrap();
            let (out_minus, _, _) = batch_norm_forward(
                &inp_minus,
                &gamma,
                &[0.0; 2],
                Some(&[0.0; 2]),
                Some(&[1.0; 2]),
                &cfg,
                true,
            )
            .unwrap();

            let numerical: f32 =
                out_plus.iter().zip(out_minus.iter()).map(|(p, m)| (p - m) / (2.0 * eps)).sum();
            assert!(
                (grad_input[i] - numerical).abs() < 0.01,
                "grad_input[{i}]: analytic={}, numerical={numerical}",
                grad_input[i]
            );
        }
    }

    #[test]
    fn backward_grad_output_length_mismatch() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (mean, var) = compute_batch_stats(&input, 2).unwrap();
        let cfg = default_cfg(2);
        assert!(batch_norm_backward(&[0.0; 6], &input, &[1.0; 2], &mean, &var, &cfg).is_err());
    }

    #[test]
    fn backward_output_shapes() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let grad_output = vec![1.0; 6];
        let (mean, var) = compute_batch_stats(&input, 3).unwrap();
        let cfg = default_cfg(3);
        let (gi, gg, gb) =
            batch_norm_backward(&grad_output, &input, &[1.0; 3], &mean, &var, &cfg).unwrap();
        assert_eq!(gi.len(), 6);
        assert_eq!(gg.len(), 3);
        assert_eq!(gb.len(), 3);
    }

    // ── compute_batch_stats ────────────────────────────────

    #[test]
    fn batch_stats_basic() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let (mean, var) = compute_batch_stats(&input, 2).unwrap();
        assert!((mean[0] - 4.0).abs() < TOL);
        assert!((mean[1] - 6.0).abs() < TOL);
        assert!((var[0] - 4.0).abs() < TOL);
        assert!((var[1] - 4.0).abs() < TOL);
    }

    #[test]
    fn batch_stats_single_sample() {
        let (mean, var) = compute_batch_stats(&[3.0, 7.0], 2).unwrap();
        assert!((mean[0] - 3.0).abs() < TOL);
        assert!((mean[1] - 7.0).abs() < TOL);
        assert!(var[0].abs() < TOL);
        assert!(var[1].abs() < TOL);
    }

    #[test]
    fn batch_stats_uniform_has_zero_var() {
        let input = vec![5.0; 6];
        let (_, var) = compute_batch_stats(&input, 3).unwrap();
        for &v in &var {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn batch_stats_zero_features_error() {
        assert!(compute_batch_stats(&[1.0], 0).is_err());
    }

    #[test]
    fn batch_stats_empty_input_error() {
        assert!(compute_batch_stats(&[], 2).is_err());
    }

    #[test]
    fn batch_stats_not_multiple_error() {
        assert!(compute_batch_stats(&[1.0, 2.0, 3.0], 2).is_err());
    }

    // ── Error cases ────────────────────────────────────────

    #[test]
    fn forward_empty_input_returns_error() {
        let cfg = default_cfg(2);
        assert!(
            batch_norm_forward(
                &[],
                &[1.0; 2],
                &[0.0; 2],
                Some(&[0.0; 2]),
                Some(&[1.0; 2]),
                &cfg,
                true,
            )
            .is_err()
        );
    }

    #[test]
    fn forward_zero_features_returns_error() {
        let cfg = default_cfg(0);
        assert!(batch_norm_forward(&[1.0], &[], &[], None, None, &cfg, true,).is_err());
    }

    #[test]
    fn forward_gamma_length_mismatch() {
        let cfg = default_cfg(2);
        assert!(
            batch_norm_forward(
                &[1.0, 2.0],
                &[1.0],
                &[0.0; 2],
                Some(&[0.0; 2]),
                Some(&[1.0; 2]),
                &cfg,
                true,
            )
            .is_err()
        );
    }

    #[test]
    fn forward_beta_length_mismatch() {
        let cfg = default_cfg(2);
        assert!(
            batch_norm_forward(
                &[1.0, 2.0],
                &[1.0; 2],
                &[0.0],
                Some(&[0.0; 2]),
                Some(&[1.0; 2]),
                &cfg,
                true,
            )
            .is_err()
        );
    }

    #[test]
    fn forward_zero_eps_returns_error() {
        let mut cfg = default_cfg(2);
        cfg.eps = 0.0;
        assert!(
            batch_norm_forward(
                &[1.0, 2.0],
                &[1.0; 2],
                &[0.0; 2],
                Some(&[0.0; 2]),
                Some(&[1.0; 2]),
                &cfg,
                true,
            )
            .is_err()
        );
    }

    #[test]
    fn forward_negative_eps_returns_error() {
        let mut cfg = default_cfg(2);
        cfg.eps = -1e-5;
        assert!(
            batch_norm_forward(
                &[1.0, 2.0],
                &[1.0; 2],
                &[0.0; 2],
                Some(&[0.0; 2]),
                Some(&[1.0; 2]),
                &cfg,
                true,
            )
            .is_err()
        );
    }

    #[test]
    fn forward_invalid_momentum_returns_error() {
        let mut cfg = default_cfg(2);
        cfg.momentum = 1.5;
        assert!(
            batch_norm_forward(
                &[1.0, 2.0],
                &[1.0; 2],
                &[0.0; 2],
                Some(&[0.0; 2]),
                Some(&[1.0; 2]),
                &cfg,
                true,
            )
            .is_err()
        );
    }

    #[test]
    fn forward_input_not_multiple_of_features() {
        let cfg = default_cfg(3);
        assert!(
            batch_norm_forward(
                &[1.0, 2.0],
                &[1.0; 3],
                &[0.0; 3],
                Some(&[0.0; 3]),
                Some(&[1.0; 3]),
                &cfg,
                true,
            )
            .is_err()
        );
    }

    // ── Larger batch ───────────────────────────────────────

    #[test]
    fn forward_larger_batch() {
        let c = 4;
        let n = 16;
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1).collect();
        let cfg = default_cfg(c);
        let (out, _, _) = batch_norm_forward(
            &input,
            &vec![1.0; c],
            &vec![0.0; c],
            Some(&vec![0.0; c]),
            Some(&vec![1.0; c]),
            &cfg,
            true,
        )
        .unwrap();
        assert_eq!(out.len(), n * c);
        for ch in 0..c {
            let ch_mean: f32 = (0..n).map(|i| out[i * c + ch]).sum::<f32>() / n as f32;
            assert!(ch_mean.abs() < TOL, "ch{ch} mean should be ~0, got {ch_mean}");
        }
    }

    #[test]
    fn inference_larger_batch() {
        let c = 4;
        let n = 16;
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1).collect();
        let rm: Vec<f32> = (0..c).map(|ch| (ch as f32) * 0.8).collect();
        let rv = vec![2.0; c];
        let cfg = default_cfg(c);
        let (out, _, _) = batch_norm_forward(
            &input,
            &vec![1.0; c],
            &vec![0.0; c],
            Some(&rm),
            Some(&rv),
            &cfg,
            false,
        )
        .unwrap();
        let exp = reference_bn(&input, &vec![1.0; c], &vec![0.0; c], &rm, &rv, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }
}
