//! CPU batch normalization kernel.
//!
//! Provides batch normalization for 1D, 2D (NCHW), and 3D inputs on
//! contiguous `f32` slices with learnable affine parameters (gamma/beta)
//! and running statistics tracking.

use bitnet_common::{BitNetError, KernelError, Result};

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

/// Configuration for batch normalization.
#[derive(Debug, Clone)]
pub struct BatchNormConfig {
    /// Number of features (channels).
    pub num_features: usize,
    /// Small constant added to variance for numerical stability.
    pub eps: f32,
    /// Momentum for running mean/variance update (new = (1-momentum)*old + momentum*batch).
    pub momentum: f32,
    /// Whether we are in training mode (updates running stats).
    pub training: bool,
}

impl BatchNormConfig {
    /// Convenience constructor with default eps (1e-5), momentum (0.1), training off.
    pub fn new(num_features: usize) -> Self {
        Self { num_features, eps: 1e-5, momentum: 0.1, training: false }
    }
}

impl Default for BatchNormConfig {
    fn default() -> Self {
        Self { num_features: 1, eps: 1e-5, momentum: 0.1, training: false }
    }
}

/// Compute batch normalization in training mode.
///
/// Returns `(output, updated_running_mean, updated_running_var)`.
///
/// Input is a flat buffer in `[N, C]` channel-interleaved order where
/// `N = input.len() / num_features`. For 2D/3D data (NCHW), pre-flatten
/// spatial dims into the batch dimension.
pub fn batch_norm_forward(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    config: &BatchNormConfig,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let batch_size = validate_forward_args(input, gamma, beta, running_mean, running_var, config)?;
    let c = config.num_features;
    let count = batch_size as f64;

    let mut batch_mean = vec![0.0f64; c];
    let mut batch_var = vec![0.0f64; c];

    for n in 0..batch_size {
        for ch in 0..c {
            batch_mean[ch] += input[n * c + ch] as f64;
        }
    }
    for m in &mut batch_mean {
        *m /= count;
    }

    for n in 0..batch_size {
        for ch in 0..c {
            let d = input[n * c + ch] as f64 - batch_mean[ch];
            batch_var[ch] += d * d;
        }
    }
    for v in &mut batch_var {
        *v /= count;
    }

    let mut output = vec![0.0f32; input.len()];
    for n in 0..batch_size {
        for ch in 0..c {
            let inv_std = 1.0 / (batch_var[ch] + config.eps as f64).sqrt();
            let x_hat = (input[n * c + ch] as f64 - batch_mean[ch]) * inv_std;
            output[n * c + ch] = (gamma[ch] as f64 * x_hat + beta[ch] as f64) as f32;
        }
    }

    let mom = config.momentum as f64;
    let mut updated_mean = vec![0.0f32; c];
    let mut updated_var = vec![0.0f32; c];
    for ch in 0..c {
        updated_mean[ch] = ((1.0 - mom) * running_mean[ch] as f64 + mom * batch_mean[ch]) as f32;
        updated_var[ch] = ((1.0 - mom) * running_var[ch] as f64 + mom * batch_var[ch]) as f32;
    }

    Ok((output, updated_mean, updated_var))
}

/// Compute batch normalization in inference mode.
///
/// Uses pre-computed running mean/variance. No statistics update.
pub fn batch_norm_inference(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    let c = gamma.len();
    validate_inference_args(input, gamma, beta, running_mean, running_var, c, eps)?;
    let batch_size = input.len() / c;

    let mut output = vec![0.0f32; input.len()];
    for n in 0..batch_size {
        for ch in 0..c {
            let inv_std = 1.0 / (running_var[ch] as f64 + eps as f64).sqrt();
            let x_hat = (input[n * c + ch] as f64 - running_mean[ch] as f64) * inv_std;
            output[n * c + ch] = (gamma[ch] as f64 * x_hat + beta[ch] as f64) as f32;
        }
    }

    Ok(output)
}

fn validate_forward_args(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    config: &BatchNormConfig,
) -> Result<usize> {
    let c = config.num_features;
    if c == 0 {
        return Err(invalid_args("num_features must be > 0"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if config.eps <= 0.0 || !config.eps.is_finite() {
        return Err(invalid_args("eps must be positive and finite"));
    }
    if !config.momentum.is_finite() || config.momentum < 0.0 || config.momentum > 1.0 {
        return Err(invalid_args("momentum must be in [0, 1] and finite"));
    }
    if gamma.len() != c {
        return Err(invalid_args(&format!("gamma length {} != num_features {c}", gamma.len())));
    }
    if beta.len() != c {
        return Err(invalid_args(&format!("beta length {} != num_features {c}", beta.len())));
    }
    if running_mean.len() != c {
        return Err(invalid_args(&format!(
            "running_mean length {} != num_features {c}",
            running_mean.len()
        )));
    }
    if running_var.len() != c {
        return Err(invalid_args(&format!(
            "running_var length {} != num_features {c}",
            running_var.len()
        )));
    }
    if !input.len().is_multiple_of(c) {
        return Err(invalid_args("input length must be a multiple of num_features"));
    }
    Ok(input.len() / c)
}

fn validate_inference_args(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    num_features: usize,
    eps: f32,
) -> Result<()> {
    if num_features == 0 {
        return Err(invalid_args("num_features must be > 0"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if eps <= 0.0 || !eps.is_finite() {
        return Err(invalid_args("eps must be positive and finite"));
    }
    if gamma.len() != num_features {
        return Err(invalid_args(&format!(
            "gamma length {} != num_features {num_features}",
            gamma.len()
        )));
    }
    if beta.len() != num_features {
        return Err(invalid_args(&format!(
            "beta length {} != num_features {num_features}",
            beta.len()
        )));
    }
    if running_mean.len() != num_features {
        return Err(invalid_args(&format!(
            "running_mean length {} != num_features {num_features}",
            running_mean.len()
        )));
    }
    if running_var.len() != num_features {
        return Err(invalid_args(&format!(
            "running_var length {} != num_features {num_features}",
            running_var.len()
        )));
    }
    if !input.len().is_multiple_of(num_features) {
        return Err(invalid_args("input length must be a multiple of num_features"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    fn reference_bn_inference(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        running_mean: &[f32],
        running_var: &[f32],
        eps: f64,
    ) -> Vec<f32> {
        let c = gamma.len();
        let n = input.len() / c;
        let mut output = vec![0.0f32; input.len()];
        for i in 0..n {
            for ch in 0..c {
                let inv_std = 1.0 / (running_var[ch] as f64 + eps).sqrt();
                let x_hat = (input[i * c + ch] as f64 - running_mean[ch] as f64) * inv_std;
                output[i * c + ch] = (gamma[ch] as f64 * x_hat + beta[ch] as f64) as f32;
            }
        }
        output
    }

    // ── Config ─────────────────────────────────────────────

    #[test]
    fn config_default() {
        let c = BatchNormConfig::default();
        assert_eq!(c.num_features, 1);
        assert!((c.eps - 1e-5).abs() < 1e-10);
        assert!((c.momentum - 0.1).abs() < 1e-10);
        assert!(!c.training);
    }

    #[test]
    fn config_new() {
        let c = BatchNormConfig::new(64);
        assert_eq!(c.num_features, 64);
        assert!(!c.training);
    }

    // ── Forward correctness ────────────────────────────────

    #[test]
    fn forward_basic_1d() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let rm = vec![0.0, 0.0];
        let rv = vec![1.0, 1.0];
        let cfg = BatchNormConfig { num_features: 2, eps: 1e-5, momentum: 0.1, training: true };
        let (out, _, _) = batch_norm_forward(&input, &gamma, &beta, &rm, &rv, &cfg).unwrap();
        let ch0_mean: f32 = (0..3).map(|n| out[n * 2]).sum::<f32>() / 3.0;
        let ch1_mean: f32 = (0..3).map(|n| out[n * 2 + 1]).sum::<f32>() / 3.0;
        assert!(ch0_mean.abs() < TOL);
        assert!(ch1_mean.abs() < TOL);
    }

    #[test]
    fn forward_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0, 0.5];
        let beta = vec![1.0, -1.0];
        let cfg = BatchNormConfig { num_features: 2, eps: 1e-5, momentum: 0.1, training: true };
        let (out, _, _) =
            batch_norm_forward(&input, &gamma, &beta, &[0.0; 2], &[1.0; 2], &cfg).unwrap();
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_uniform_input() {
        let input = vec![5.0; 8];
        let cfg = BatchNormConfig { num_features: 2, eps: 1e-5, momentum: 0.1, training: true };
        let (out, _, _) =
            batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], &cfg).unwrap();
        for &v in &out {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn forward_single_sample() {
        let input = vec![3.0, 7.0];
        let cfg = BatchNormConfig { num_features: 2, eps: 1e-5, momentum: 0.1, training: true };
        let (out, _, _) =
            batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], &cfg).unwrap();
        assert!(out[0].abs() < TOL);
        assert!(out[1].abs() < TOL);
    }

    // ── Running stats ──────────────────────────────────────

    #[test]
    fn forward_running_mean_update() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = BatchNormConfig { num_features: 2, eps: 1e-5, momentum: 0.1, training: true };
        let (_, um, _) =
            batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], &cfg).unwrap();
        assert!((um[0] - 0.4).abs() < TOL);
        assert!((um[1] - 0.6).abs() < TOL);
    }

    #[test]
    fn forward_running_var_update() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = BatchNormConfig { num_features: 2, eps: 1e-5, momentum: 0.1, training: true };
        let (_, _, uv) =
            batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], &cfg).unwrap();
        // batch var for each channel: ((2-4)^2+(6-4)^2)/2 = 4 for ch0
        // running_var update: (1-0.1)*1.0 + 0.1*4.0 = 0.9 + 0.4 = 1.3
        assert!((uv[0] - 1.3).abs() < TOL);
        assert!((uv[1] - 1.3).abs() < TOL);
    }

    #[test]
    fn forward_zero_momentum_preserves_stats() {
        let rm = vec![5.0, 15.0];
        let rv = vec![2.0, 3.0];
        let cfg = BatchNormConfig { num_features: 2, eps: 1e-5, momentum: 0.0, training: true };
        let (_, um, uv) =
            batch_norm_forward(&[10.0, 20.0, 30.0, 40.0], &[1.0; 2], &[0.0; 2], &rm, &rv, &cfg)
                .unwrap();
        assert!(approx_eq(&um, &rm, TOL));
        assert!(approx_eq(&uv, &rv, TOL));
    }

    #[test]
    fn forward_full_momentum_uses_batch_stats() {
        let cfg = BatchNormConfig { num_features: 2, eps: 1e-5, momentum: 1.0, training: true };
        let (_, um, uv) = batch_norm_forward(
            &[2.0, 4.0, 6.0, 8.0],
            &[1.0; 2],
            &[0.0; 2],
            &[100.0, 200.0],
            &[50.0, 60.0],
            &cfg,
        )
        .unwrap();
        // batch_mean = [4.0, 6.0], batch_var = [4.0, 4.0]
        assert!((um[0] - 4.0).abs() < TOL);
        assert!((um[1] - 6.0).abs() < TOL);
        assert!((uv[0] - 4.0).abs() < TOL);
        assert!((uv[1] - 4.0).abs() < TOL);
    }

    // ── Inference ──────────────────────────────────────────

    #[test]
    fn inference_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let rm = vec![2.0, 3.0];
        let rv = vec![1.0, 1.0];
        let out = batch_norm_inference(&input, &[1.0; 2], &[0.0; 2], &rm, &rv, 1e-5).unwrap();
        let exp = reference_bn_inference(&input, &[1.0; 2], &[0.0; 2], &rm, &rv, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn inference_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![2.0, 0.5];
        let beta = vec![1.0, -1.0];
        let rm = vec![3.0, 4.0];
        let rv = vec![4.0, 9.0];
        let out = batch_norm_inference(&input, &gamma, &beta, &rm, &rv, 1e-5).unwrap();
        let exp = reference_bn_inference(&input, &gamma, &beta, &rm, &rv, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn inference_identity_transform() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out =
            batch_norm_inference(&input, &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], 1e-5).unwrap();
        assert!(approx_eq(&out, &input, 1e-3));
    }

    // ── Training vs inference ──────────────────────────────

    #[test]
    fn training_and_inference_differ() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rm = vec![10.0, 20.0];
        let rv = vec![5.0, 5.0];
        let cfg = BatchNormConfig { num_features: 2, eps: 1e-5, momentum: 0.1, training: true };
        let (t_out, _, _) =
            batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], &rm, &rv, &cfg).unwrap();
        let i_out = batch_norm_inference(&input, &[1.0; 2], &[0.0; 2], &rm, &rv, 1e-5).unwrap();
        assert!(!approx_eq(&t_out, &i_out, TOL));
    }

    // ── Numerical stability ────────────────────────────────

    #[test]
    fn forward_large_values() {
        let input = vec![1e6, 1e6 + 1.0, 1e6, 1e6 + 1.0];
        let cfg = BatchNormConfig { num_features: 2, eps: 1e-5, momentum: 0.1, training: true };
        let (out, _, _) =
            batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], &cfg).unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_no_nan_or_inf() {
        let input = vec![1e10, -1e10, 0.0, 1e-10, 1e10, -1e10, 0.0, 1e-10];
        let cfg = BatchNormConfig { num_features: 4, eps: 1e-5, momentum: 0.1, training: true };
        let (out, um, uv) =
            batch_norm_forward(&input, &[1.0; 4], &[0.0; 4], &[0.0; 4], &[1.0; 4], &cfg).unwrap();
        for &v in out.iter().chain(um.iter()).chain(uv.iter()) {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn inference_no_nan_or_inf() {
        let input = vec![1e10, -1e10, 0.0, 1e-10];
        let out =
            batch_norm_inference(&input, &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], 1e-5).unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    // ── 2D (NCHW) input ───────────────────────────────────

    #[test]
    fn forward_2d_nchw() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cfg = BatchNormConfig { num_features: 2, eps: 1e-5, momentum: 0.1, training: true };
        let (out, _, _) =
            batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], &cfg).unwrap();
        assert_eq!(out.len(), 8);
        let ch0_mean: f32 = (0..4).map(|n| out[n * 2]).sum::<f32>() / 4.0;
        let ch1_mean: f32 = (0..4).map(|n| out[n * 2 + 1]).sum::<f32>() / 4.0;
        assert!(ch0_mean.abs() < TOL);
        assert!(ch1_mean.abs() < TOL);
    }

    #[test]
    fn inference_2d_nchw() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rm = vec![3.0, 5.0];
        let rv = vec![4.0, 4.0];
        let out = batch_norm_inference(&input, &[1.0; 2], &[0.0; 2], &rm, &rv, 1e-5).unwrap();
        let exp = reference_bn_inference(&input, &[1.0; 2], &[0.0; 2], &rm, &rv, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    // ── 3D input ───────────────────────────────────────────

    #[test]
    fn forward_3d_input() {
        let input: Vec<f32> = (1..=18).map(|i| i as f32).collect();
        let cfg = BatchNormConfig { num_features: 3, eps: 1e-5, momentum: 0.1, training: true };
        let (out, _, _) =
            batch_norm_forward(&input, &[1.0; 3], &[0.0; 3], &[0.0; 3], &[1.0; 3], &cfg).unwrap();
        assert_eq!(out.len(), 18);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn inference_3d_input() {
        let input: Vec<f32> = (1..=18).map(|i| i as f32).collect();
        let rm = vec![5.0, 10.0, 15.0];
        let rv = vec![10.0, 10.0, 10.0];
        let out = batch_norm_inference(&input, &[1.0; 3], &[0.0; 3], &rm, &rv, 1e-5).unwrap();
        let exp = reference_bn_inference(&input, &[1.0; 3], &[0.0; 3], &rm, &rv, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    // ── Error cases ────────────────────────────────────────

    #[test]
    fn forward_empty_input_returns_error() {
        let cfg = BatchNormConfig::new(2);
        assert!(
            batch_norm_forward(&[], &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], &cfg,).is_err()
        );
    }

    #[test]
    fn forward_zero_features_returns_error() {
        let cfg = BatchNormConfig::new(0);
        assert!(batch_norm_forward(&[1.0], &[], &[], &[], &[], &cfg).is_err());
    }

    #[test]
    fn forward_gamma_length_mismatch() {
        let cfg = BatchNormConfig::new(2);
        assert!(
            batch_norm_forward(&[1.0, 2.0], &[1.0], &[0.0; 2], &[0.0; 2], &[1.0; 2], &cfg,)
                .is_err()
        );
    }

    #[test]
    fn forward_beta_length_mismatch() {
        let cfg = BatchNormConfig::new(2);
        assert!(
            batch_norm_forward(&[1.0, 2.0], &[1.0; 2], &[0.0], &[0.0; 2], &[1.0; 2], &cfg,)
                .is_err()
        );
    }

    #[test]
    fn forward_running_mean_length_mismatch() {
        let cfg = BatchNormConfig::new(2);
        assert!(
            batch_norm_forward(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], &[0.0], &[1.0; 2], &cfg,)
                .is_err()
        );
    }

    #[test]
    fn forward_running_var_length_mismatch() {
        let cfg = BatchNormConfig::new(2);
        assert!(
            batch_norm_forward(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0], &cfg,)
                .is_err()
        );
    }

    #[test]
    fn forward_zero_eps_returns_error() {
        let mut cfg = BatchNormConfig::new(2);
        cfg.eps = 0.0;
        assert!(
            batch_norm_forward(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], &cfg,)
                .is_err()
        );
    }

    #[test]
    fn forward_negative_eps_returns_error() {
        let mut cfg = BatchNormConfig::new(2);
        cfg.eps = -1e-5;
        assert!(
            batch_norm_forward(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], &cfg,)
                .is_err()
        );
    }

    #[test]
    fn forward_invalid_momentum_returns_error() {
        let mut cfg = BatchNormConfig::new(2);
        cfg.momentum = 1.5;
        assert!(
            batch_norm_forward(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], &cfg,)
                .is_err()
        );
    }

    #[test]
    fn forward_input_not_multiple_of_features() {
        let cfg = BatchNormConfig::new(3);
        assert!(
            batch_norm_forward(&[1.0, 2.0], &[1.0; 3], &[0.0; 3], &[0.0; 3], &[1.0; 3], &cfg,)
                .is_err()
        );
    }

    #[test]
    fn inference_empty_input_returns_error() {
        assert!(batch_norm_inference(&[], &[1.0], &[0.0], &[0.0], &[1.0], 1e-5).is_err());
    }

    #[test]
    fn inference_gamma_length_mismatch() {
        assert!(
            batch_norm_inference(&[1.0, 2.0], &[1.0; 3], &[0.0; 2], &[0.0; 2], &[1.0; 2], 1e-5,)
                .is_err()
        );
    }

    #[test]
    fn inference_zero_eps_returns_error() {
        assert!(
            batch_norm_inference(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], 0.0,)
                .is_err()
        );
    }

    // ── Larger batch ───────────────────────────────────────

    #[test]
    fn forward_larger_batch() {
        let c = 4;
        let n = 16;
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1).collect();
        let cfg = BatchNormConfig { num_features: c, eps: 1e-5, momentum: 0.1, training: true };
        let (out, _, _) = batch_norm_forward(
            &input,
            &vec![1.0; c],
            &vec![0.0; c],
            &vec![0.0; c],
            &vec![1.0; c],
            &cfg,
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
        let out =
            batch_norm_inference(&input, &vec![1.0; c], &vec![0.0; c], &rm, &rv, 1e-5).unwrap();
        let exp = reference_bn_inference(&input, &vec![1.0; c], &vec![0.0; c], &rm, &rv, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }
}
