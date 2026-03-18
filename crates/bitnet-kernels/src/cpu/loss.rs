//! CPU loss function kernels.
//!
//! Provides common loss functions for training and evaluation:
//! cross-entropy, binary cross-entropy, MSE, L1, smooth L1,
//! KL divergence, cosine similarity, contrastive loss, focal loss,
//! label-smoothing cross-entropy, and perplexity.
//!
//! Also provides gradient utility functions: accumulation, norm
//! clipping, and value clipping.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Helpers ────────────────────────────────────────────────────────

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

fn validate_same_len(a: &[f32], b: &[f32], name: &str) -> Result<()> {
    if a.is_empty() {
        return Err(invalid_args(&format!("{name}: inputs must not be empty")));
    }
    if a.len() != b.len() {
        return Err(invalid_args(&format!("{name}: length mismatch ({} vs {})", a.len(), b.len())));
    }
    Ok(())
}

/// Numerical stability clamp for log arguments.
const EPS: f32 = 1e-7;

// ── Types ──────────────────────────────────────────────────────────

/// How to reduce per-element losses into a scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LossReduction {
    /// Return the raw per-element sum (no normalisation).
    None,
    /// Arithmetic mean of per-element losses.
    Mean,
    /// Sum of per-element losses.
    Sum,
}

fn reduce(values: &[f32], reduction: LossReduction) -> f32 {
    match reduction {
        LossReduction::None => values.iter().sum(),
        LossReduction::Sum => values.iter().sum(),
        LossReduction::Mean => values.iter().sum::<f32>() / values.len() as f32,
    }
}

// ── Loss Functions ─────────────────────────────────────────────────

/// Cross-entropy loss over a batch of logits and integer class targets.
///
/// `logits` is `[batch_size, num_classes]` in row-major order.
/// `targets` contains the correct class index for each sample.
///
/// Returns `(scalar_loss, per_sample_losses)` where the scalar is
/// reduced according to `reduction`.
pub fn cross_entropy_loss(
    logits: &[f32],
    targets: &[usize],
    num_classes: usize,
    reduction: LossReduction,
) -> Result<(f32, Vec<f32>)> {
    if targets.is_empty() {
        return Err(invalid_args("cross_entropy_loss: targets must not be empty"));
    }
    let batch_size = targets.len();
    if num_classes == 0 {
        return Err(invalid_args("cross_entropy_loss: num_classes must be > 0"));
    }
    if logits.len() != batch_size * num_classes {
        return Err(invalid_args("cross_entropy_loss: logits length mismatch"));
    }
    for (i, &t) in targets.iter().enumerate() {
        if t >= num_classes {
            return Err(invalid_args(&format!(
                "cross_entropy_loss: target[{i}]={t} >= num_classes={num_classes}"
            )));
        }
    }

    let mut per_sample = Vec::with_capacity(batch_size);
    for (i, &target) in targets.iter().enumerate() {
        let row = &logits[i * num_classes..(i + 1) * num_classes];
        // log-sum-exp for numerical stability
        let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&x| (x - max_logit).exp()).sum();
        let log_sum_exp = max_logit + sum_exp.ln();
        let loss = log_sum_exp - row[target];
        per_sample.push(loss);
    }

    let scalar = reduce(&per_sample, reduction);
    Ok((scalar, per_sample))
}

/// Binary cross-entropy loss.
///
/// `predictions` should be probabilities in `(0, 1)`. Values are
/// clamped to `[EPS, 1-EPS]` for numerical stability.
pub fn binary_cross_entropy(
    predictions: &[f32],
    targets: &[f32],
    reduction: LossReduction,
) -> Result<f32> {
    validate_same_len(predictions, targets, "binary_cross_entropy")?;
    let losses: Vec<f32> = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| {
            let p = p.clamp(EPS, 1.0 - EPS);
            -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
        })
        .collect();
    Ok(reduce(&losses, reduction))
}

/// Mean squared error loss.
pub fn mse_loss(predictions: &[f32], targets: &[f32], reduction: LossReduction) -> Result<f32> {
    validate_same_len(predictions, targets, "mse_loss")?;
    let losses: Vec<f32> =
        predictions.iter().zip(targets.iter()).map(|(&p, &t)| (p - t).powi(2)).collect();
    Ok(reduce(&losses, reduction))
}

/// L1 (mean absolute error) loss.
pub fn l1_loss(predictions: &[f32], targets: &[f32], reduction: LossReduction) -> Result<f32> {
    validate_same_len(predictions, targets, "l1_loss")?;
    let losses: Vec<f32> =
        predictions.iter().zip(targets.iter()).map(|(&p, &t)| (p - t).abs()).collect();
    Ok(reduce(&losses, reduction))
}

/// Smooth L1 (Huber) loss.
///
/// Uses the quadratic regime when `|d| < beta`, and linear otherwise.
pub fn smooth_l1_loss(
    predictions: &[f32],
    targets: &[f32],
    beta: f32,
    reduction: LossReduction,
) -> Result<f32> {
    validate_same_len(predictions, targets, "smooth_l1_loss")?;
    if beta <= 0.0 {
        return Err(invalid_args("smooth_l1_loss: beta must be > 0"));
    }
    let losses: Vec<f32> = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| {
            let d = (p - t).abs();
            if d < beta { 0.5 * d * d / beta } else { d - 0.5 * beta }
        })
        .collect();
    Ok(reduce(&losses, reduction))
}

/// KL divergence: `D_KL(target || exp(log_probs))`.
///
/// `log_probs` are **log-probabilities** (e.g. after log-softmax).
/// `targets` are a probability distribution (should sum to 1).
pub fn kl_divergence(log_probs: &[f32], targets: &[f32], reduction: LossReduction) -> Result<f32> {
    validate_same_len(log_probs, targets, "kl_divergence")?;
    let losses: Vec<f32> = log_probs
        .iter()
        .zip(targets.iter())
        .map(|(&lp, &t)| if t <= 0.0 { 0.0 } else { t * (t.ln() - lp) })
        .collect();
    Ok(reduce(&losses, reduction))
}

/// Cosine similarity loss: `1 - cos(a, b)`.
///
/// Returns a value in `[0, 2]`. Zero when vectors are identical in
/// direction.
pub fn cosine_similarity_loss(a: &[f32], b: &[f32]) -> Result<f32> {
    validate_same_len(a, b, "cosine_similarity_loss")?;
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm_a * norm_b;
    if denom < EPS {
        return Ok(1.0); // undefined → treat as orthogonal
    }
    Ok(1.0 - dot / denom)
}

/// Contrastive loss (Siamese networks).
///
/// `label` = 1.0 for a positive pair (same class), 0.0 for negative.
/// `margin` is the minimum distance required for negative pairs.
pub fn contrastive_loss(a: &[f32], b: &[f32], label: f32, margin: f32) -> Result<f32> {
    validate_same_len(a, b, "contrastive_loss")?;
    let dist_sq: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum();
    let dist = dist_sq.sqrt();
    let pos = label * dist_sq;
    let neg = (1.0 - label) * (margin - dist).max(0.0).powi(2);
    Ok(0.5 * (pos + neg))
}

/// Cross-entropy loss from raw logits and a soft target distribution.
///
/// `logits` is `[num_classes]` (single sample). `targets` is a
/// probability distribution over classes (should sum to 1).
/// Applies log-softmax internally for numerical stability.
pub fn cross_entropy_with_logits(
    logits: &[f32],
    targets: &[f32],
    reduction: LossReduction,
) -> Result<f32> {
    validate_same_len(logits, targets, "cross_entropy_with_logits")?;
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    let log_sum_exp = max_logit + sum_exp.ln();
    // per-element: -target_i * (logit_i - log_sum_exp)
    let losses: Vec<f32> = logits
        .iter()
        .zip(targets.iter())
        .map(|(&l, &t)| if t <= 0.0 { 0.0 } else { -t * (l - log_sum_exp) })
        .collect();
    Ok(reduce(&losses, reduction))
}

/// Mean absolute error (L1) loss. Alias for [`l1_loss`].
pub fn mae_loss(predictions: &[f32], targets: &[f32], reduction: LossReduction) -> Result<f32> {
    l1_loss(predictions, targets, reduction)
}

/// Huber loss (smooth L1). Alias for [`smooth_l1_loss`] with `delta` as the transition point.
pub fn huber_loss(
    predictions: &[f32],
    targets: &[f32],
    delta: f32,
    reduction: LossReduction,
) -> Result<f32> {
    smooth_l1_loss(predictions, targets, delta, reduction)
}

/// Focal loss for addressing class imbalance.
///
/// `probabilities` are predicted class probabilities in `(0, 1)`.
/// `targets` are ground-truth labels (0.0 or 1.0).
/// `gamma` is the focusing parameter (0 = standard CE, higher = more focus on hard examples).
/// `alpha` is the class-balancing weight in `(0, 1)`.
pub fn focal_loss(
    probabilities: &[f32],
    targets: &[f32],
    gamma: f32,
    alpha: f32,
    reduction: LossReduction,
) -> Result<f32> {
    validate_same_len(probabilities, targets, "focal_loss")?;
    if gamma < 0.0 {
        return Err(invalid_args("focal_loss: gamma must be >= 0"));
    }
    let losses: Vec<f32> = probabilities
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| {
            let p = p.clamp(EPS, 1.0 - EPS);
            let p_t = if t >= 0.5 { p } else { 1.0 - p };
            let alpha_t = if t >= 0.5 { alpha } else { 1.0 - alpha };
            -alpha_t * (1.0 - p_t).powf(gamma) * p_t.ln()
        })
        .collect();
    Ok(reduce(&losses, reduction))
}

/// Cross-entropy loss with label smoothing.
///
/// Smoothed targets: `(1 - alpha) * one_hot + alpha / num_classes`.
/// `logits` is `[batch_size, num_classes]` in row-major order.
pub fn label_smoothing_ce(
    logits: &[f32],
    targets: &[usize],
    num_classes: usize,
    alpha: f32,
    reduction: LossReduction,
) -> Result<(f32, Vec<f32>)> {
    if targets.is_empty() {
        return Err(invalid_args("label_smoothing_ce: targets must not be empty"));
    }
    if num_classes == 0 {
        return Err(invalid_args("label_smoothing_ce: num_classes must be > 0"));
    }
    let batch_size = targets.len();
    if logits.len() != batch_size * num_classes {
        return Err(invalid_args("label_smoothing_ce: logits length mismatch"));
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(invalid_args("label_smoothing_ce: alpha must be in [0, 1]"));
    }
    for (i, &t) in targets.iter().enumerate() {
        if t >= num_classes {
            return Err(invalid_args(&format!(
                "label_smoothing_ce: target[{i}]={t} >= num_classes={num_classes}"
            )));
        }
    }

    let uniform = alpha / num_classes as f32;
    let confidence = 1.0 - alpha;

    let mut per_sample = Vec::with_capacity(batch_size);
    for (i, &target) in targets.iter().enumerate() {
        let row = &logits[i * num_classes..(i + 1) * num_classes];
        let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&x| (x - max_logit).exp()).sum();
        let log_sum_exp = max_logit + sum_exp.ln();
        // log_softmax values
        let mut loss = 0.0_f32;
        for (j, &logit) in row.iter().enumerate() {
            let log_prob = logit - log_sum_exp;
            let smooth_target = if j == target { confidence + uniform } else { uniform };
            loss -= smooth_target * log_prob;
        }
        per_sample.push(loss);
    }

    let scalar = reduce(&per_sample, reduction);
    Ok((scalar, per_sample))
}

/// Compute perplexity from a mean cross-entropy loss value.
///
/// `perplexity = exp(ce_loss)`.
pub fn perplexity(ce_loss: f32) -> f32 {
    ce_loss.exp()
}

// ── Gradient Utilities ────────────────────────────────────────────

/// Accumulate gradients from `source` into `accumulator` (element-wise addition).
///
/// Used for gradient accumulation across micro-batches.
pub fn gradient_accumulate(accumulator: &mut [f32], source: &[f32]) -> Result<()> {
    if accumulator.len() != source.len() {
        return Err(invalid_args(&format!(
            "gradient_accumulate: length mismatch ({} vs {})",
            accumulator.len(),
            source.len()
        )));
    }
    for (acc, &src) in accumulator.iter_mut().zip(source.iter()) {
        *acc += src;
    }
    Ok(())
}

/// Clip gradients by global L2 norm.
///
/// If the global norm exceeds `max_norm`, all gradients are scaled
/// down proportionally. Returns the original global norm.
pub fn gradient_clip_norm(gradients: &mut [f32], max_norm: f32) -> Result<f32> {
    if max_norm <= 0.0 {
        return Err(invalid_args("gradient_clip_norm: max_norm must be > 0"));
    }
    let global_norm: f32 = gradients.iter().map(|&g| g * g).sum::<f32>().sqrt();
    if global_norm > max_norm {
        let scale = max_norm / global_norm;
        for g in gradients.iter_mut() {
            *g *= scale;
        }
    }
    Ok(global_norm)
}

/// Clip gradients by value, clamping each element to `[-max_value, max_value]`.
///
/// Returns the number of elements that were clipped.
pub fn gradient_clip_value(gradients: &mut [f32], max_value: f32) -> Result<usize> {
    if max_value <= 0.0 {
        return Err(invalid_args("gradient_clip_value: max_value must be > 0"));
    }
    let mut clipped = 0;
    for g in gradients.iter_mut() {
        if *g > max_value {
            *g = max_value;
            clipped += 1;
        } else if *g < -max_value {
            *g = -max_value;
            clipped += 1;
        }
    }
    Ok(clipped)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    // ── Cross-Entropy ──────────────────────────────────────────

    #[test]
    fn cross_entropy_basic() {
        let logits = [1.0, 2.0, 0.5];
        let (loss, per) = cross_entropy_loss(&logits, &[1], 3, LossReduction::Mean).unwrap();
        let max_l = 2.0_f32;
        let lse = max_l + ((1.0 - max_l).exp() + 0.0_f32.exp() + (0.5 - max_l).exp()).ln();
        let expected = lse - 2.0;
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
        assert_eq!(per.len(), 1);
    }

    #[test]
    fn cross_entropy_batch() {
        let logits = [1.0, 0.0, 0.0, 1.0];
        let (loss_mean, per) =
            cross_entropy_loss(&logits, &[0, 1], 2, LossReduction::Mean).unwrap();
        assert_eq!(per.len(), 2);
        assert!(approx(per[0], per[1]));
        assert!(approx(loss_mean, per[0]));
    }

    #[test]
    fn cross_entropy_sum_reduction() {
        let logits = [1.0, 0.0, 0.0, 1.0];
        let (loss_sum, per) = cross_entropy_loss(&logits, &[0, 1], 2, LossReduction::Sum).unwrap();
        let expected_sum: f32 = per.iter().sum();
        assert!(approx(loss_sum, expected_sum));
    }

    #[test]
    fn cross_entropy_target_out_of_range() {
        let logits = [1.0, 2.0, 3.0];
        assert!(cross_entropy_loss(&logits, &[3], 3, LossReduction::Mean).is_err());
    }

    #[test]
    fn cross_entropy_empty_targets() {
        assert!(cross_entropy_loss(&[], &[], 3, LossReduction::Mean).is_err());
    }

    #[test]
    fn cross_entropy_length_mismatch() {
        let logits = [1.0, 2.0];
        assert!(cross_entropy_loss(&logits, &[0, 1], 3, LossReduction::Mean).is_err());
    }

    #[test]
    fn cross_entropy_single_class() {
        let logits = [5.0];
        let (loss, _) = cross_entropy_loss(&logits, &[0], 1, LossReduction::Mean).unwrap();
        assert!(approx(loss, 0.0), "single class should be 0, got {loss}");
    }

    #[test]
    fn cross_entropy_perfect_logits() {
        // Very high logit on the correct class
        let logits = [100.0, -100.0, -100.0];
        let (loss, _) = cross_entropy_loss(&logits, &[0], 3, LossReduction::Mean).unwrap();
        assert!(loss < 1e-6, "near-perfect prediction, got {loss}");
    }

    #[test]
    fn cross_entropy_large_logits() {
        let logits = [1000.0, 0.0, 0.0];
        let (loss, _) = cross_entropy_loss(&logits, &[0], 3, LossReduction::Mean).unwrap();
        assert!(loss.is_finite(), "got {loss}");
        assert!(loss >= 0.0, "loss should be non-negative, got {loss}");
    }

    #[test]
    fn cross_entropy_negative_logits() {
        let logits = [-1000.0, 0.0, 0.0];
        let (loss, _) = cross_entropy_loss(&logits, &[0], 3, LossReduction::Mean).unwrap();
        assert!(loss.is_finite(), "got {loss}");
    }

    #[test]
    fn cross_entropy_uniform_logits() {
        // All logits equal → loss = ln(num_classes)
        let logits = [0.0, 0.0, 0.0, 0.0];
        let (loss, _) = cross_entropy_loss(&logits, &[0], 4, LossReduction::Mean).unwrap();
        let expected = (4.0_f32).ln();
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn cross_entropy_batch_vs_individual() {
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (_, batch_per) = cross_entropy_loss(&logits, &[0, 2], 3, LossReduction::Mean).unwrap();
        let (_, per0) = cross_entropy_loss(&logits[0..3], &[0], 3, LossReduction::Mean).unwrap();
        let (_, per1) = cross_entropy_loss(&logits[3..6], &[2], 3, LossReduction::Mean).unwrap();
        assert!(approx(batch_per[0], per0[0]));
        assert!(approx(batch_per[1], per1[0]));
    }

    // ── Cross-Entropy with Logits (soft targets) ───────────────

    #[test]
    fn ce_with_logits_one_hot_matches_hard_ce() {
        let logits = [1.0, 2.0, 0.5];
        let targets = [0.0, 1.0, 0.0]; // one-hot for class 1
        let soft = cross_entropy_with_logits(&logits, &targets, LossReduction::Sum).unwrap();
        let (hard, _) = cross_entropy_loss(&logits, &[1], 3, LossReduction::Sum).unwrap();
        assert!(approx(soft, hard), "soft={soft}, hard={hard}");
    }

    #[test]
    fn ce_with_logits_uniform_targets() {
        let logits = [0.0, 0.0];
        let targets = [0.5, 0.5];
        let loss = cross_entropy_with_logits(&logits, &targets, LossReduction::Sum).unwrap();
        // -0.5*(0 - ln(2)) - 0.5*(0 - ln(2)) = ln(2)
        let expected = (2.0_f32).ln();
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn ce_with_logits_numerical_stability() {
        let logits = [1000.0, -1000.0];
        let targets = [1.0, 0.0];
        let loss = cross_entropy_with_logits(&logits, &targets, LossReduction::Sum).unwrap();
        assert!(loss.is_finite(), "got {loss}");
        assert!(loss >= 0.0);
    }

    #[test]
    fn ce_with_logits_empty_rejected() {
        assert!(cross_entropy_with_logits(&[], &[], LossReduction::Mean).is_err());
    }

    #[test]
    fn ce_with_logits_length_mismatch() {
        assert!(cross_entropy_with_logits(&[1.0, 2.0], &[1.0], LossReduction::Mean).is_err());
    }

    // ── Binary Cross-Entropy ───────────────────────────────────

    #[test]
    fn bce_perfect_prediction() {
        let loss = binary_cross_entropy(&[1.0, 0.0], &[1.0, 0.0], LossReduction::Mean).unwrap();
        assert!(loss < 0.01, "got {loss}");
    }

    #[test]
    fn bce_worst_prediction() {
        let loss = binary_cross_entropy(&[0.0, 1.0], &[1.0, 0.0], LossReduction::Mean).unwrap();
        assert!(loss > 10.0, "got {loss}");
    }

    #[test]
    fn bce_half_probability() {
        let loss = binary_cross_entropy(&[0.5], &[1.0], LossReduction::Mean).unwrap();
        let expected = -(0.5_f32.ln());
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn bce_empty_rejected() {
        assert!(binary_cross_entropy(&[], &[], LossReduction::Mean).is_err());
    }

    #[test]
    fn bce_length_mismatch() {
        assert!(binary_cross_entropy(&[0.5], &[1.0, 0.0], LossReduction::Mean).is_err());
    }

    #[test]
    fn bce_numerical_stability() {
        let loss = binary_cross_entropy(&[0.0, 1.0], &[0.0, 1.0], LossReduction::Mean).unwrap();
        assert!(loss.is_finite(), "got {loss}");
    }

    #[test]
    fn bce_single_element() {
        let loss = binary_cross_entropy(&[0.7], &[1.0], LossReduction::Mean).unwrap();
        let expected = -(0.7_f32.ln());
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn bce_sum_vs_mean() {
        let sum = binary_cross_entropy(&[0.5, 0.5], &[1.0, 0.0], LossReduction::Sum).unwrap();
        let mean = binary_cross_entropy(&[0.5, 0.5], &[1.0, 0.0], LossReduction::Mean).unwrap();
        assert!(approx(sum, mean * 2.0), "sum={sum}, mean={mean}");
    }

    // ── MSE ────────────────────────────────────────────────────

    #[test]
    fn mse_zero_error() {
        let loss = mse_loss(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], LossReduction::Mean).unwrap();
        assert!(approx(loss, 0.0));
    }

    #[test]
    fn mse_known_value() {
        let loss = mse_loss(&[1.0, 2.0], &[3.0, 4.0], LossReduction::Mean).unwrap();
        assert!(approx(loss, 4.0), "got {loss}");
    }

    #[test]
    fn mse_sum_reduction() {
        let loss = mse_loss(&[1.0, 2.0], &[3.0, 4.0], LossReduction::Sum).unwrap();
        assert!(approx(loss, 8.0), "got {loss}");
    }

    #[test]
    fn mse_empty_rejected() {
        assert!(mse_loss(&[], &[], LossReduction::Mean).is_err());
    }

    #[test]
    fn mse_large_values() {
        let loss = mse_loss(&[1e6], &[-1e6], LossReduction::Mean).unwrap();
        assert!(loss.is_finite(), "got {loss}");
        assert!(loss > 0.0);
    }

    #[test]
    fn mse_single_element() {
        let loss = mse_loss(&[3.0], &[1.0], LossReduction::Mean).unwrap();
        assert!(approx(loss, 4.0), "got {loss}");
    }

    #[test]
    fn mse_symmetric() {
        let l1 = mse_loss(&[1.0, 2.0], &[3.0, 4.0], LossReduction::Mean).unwrap();
        let l2 = mse_loss(&[3.0, 4.0], &[1.0, 2.0], LossReduction::Mean).unwrap();
        assert!(approx(l1, l2));
    }

    // ── MAE / L1 ──────────────────────────────────────────────

    #[test]
    fn l1_zero_error() {
        let loss = l1_loss(&[1.0, 2.0], &[1.0, 2.0], LossReduction::Mean).unwrap();
        assert!(approx(loss, 0.0));
    }

    #[test]
    fn l1_known_value() {
        let loss = l1_loss(&[1.0, 2.0], &[3.0, 4.0], LossReduction::Mean).unwrap();
        assert!(approx(loss, 2.0), "got {loss}");
    }

    #[test]
    fn l1_sum_reduction() {
        let loss = l1_loss(&[1.0, 2.0], &[3.0, 4.0], LossReduction::Sum).unwrap();
        assert!(approx(loss, 4.0), "got {loss}");
    }

    #[test]
    fn l1_negative_values() {
        let loss = l1_loss(&[-1.0, -2.0], &[1.0, 2.0], LossReduction::Mean).unwrap();
        assert!(approx(loss, 3.0), "got {loss}");
    }

    #[test]
    fn mae_alias_matches_l1() {
        let l1 = l1_loss(&[1.0, 2.0], &[4.0, 6.0], LossReduction::Mean).unwrap();
        let mae = mae_loss(&[1.0, 2.0], &[4.0, 6.0], LossReduction::Mean).unwrap();
        assert!(approx(l1, mae));
    }

    #[test]
    fn mae_single_element() {
        let loss = mae_loss(&[5.0], &[2.0], LossReduction::Mean).unwrap();
        assert!(approx(loss, 3.0), "got {loss}");
    }

    // ── Smooth L1 / Huber ──────────────────────────────────────

    #[test]
    fn smooth_l1_quadratic_regime() {
        let loss = smooth_l1_loss(&[1.0], &[1.5], 1.0, LossReduction::Mean).unwrap();
        assert!(approx(loss, 0.125), "got {loss}");
    }

    #[test]
    fn smooth_l1_linear_regime() {
        let loss = smooth_l1_loss(&[1.0], &[3.0], 1.0, LossReduction::Mean).unwrap();
        assert!(approx(loss, 1.5), "got {loss}");
    }

    #[test]
    fn smooth_l1_zero_error() {
        let loss = smooth_l1_loss(&[2.0, 3.0], &[2.0, 3.0], 1.0, LossReduction::Mean).unwrap();
        assert!(approx(loss, 0.0));
    }

    #[test]
    fn smooth_l1_invalid_beta() {
        assert!(smooth_l1_loss(&[1.0], &[2.0], 0.0, LossReduction::Mean).is_err());
        assert!(smooth_l1_loss(&[1.0], &[2.0], -1.0, LossReduction::Mean).is_err());
    }

    #[test]
    fn smooth_l1_at_boundary() {
        let loss = smooth_l1_loss(&[0.0], &[1.0], 1.0, LossReduction::Mean).unwrap();
        assert!(approx(loss, 0.5), "got {loss}");
    }

    #[test]
    fn huber_alias_matches_smooth_l1() {
        let s = smooth_l1_loss(&[1.0, 5.0], &[2.0, 0.0], 1.5, LossReduction::Mean).unwrap();
        let h = huber_loss(&[1.0, 5.0], &[2.0, 0.0], 1.5, LossReduction::Mean).unwrap();
        assert!(approx(s, h));
    }

    #[test]
    fn huber_transition_between_l1_l2() {
        let delta = 1.0;
        // Small error (quadratic): |d|=0.3 < 1 → 0.5 * 0.09 / 1.0 = 0.045
        let small = huber_loss(&[0.0], &[0.3], delta, LossReduction::Mean).unwrap();
        assert!(approx(small, 0.045), "got {small}");
        // Large error (linear): |d|=3.0 >= 1 → 3.0 - 0.5 = 2.5
        let large = huber_loss(&[0.0], &[3.0], delta, LossReduction::Mean).unwrap();
        assert!(approx(large, 2.5), "got {large}");
    }

    #[test]
    fn huber_small_delta() {
        let delta = 0.1;
        // |d|=1.0 >> delta → linear: 1.0 - 0.05 = 0.95
        let loss = huber_loss(&[0.0], &[1.0], delta, LossReduction::Mean).unwrap();
        assert!(approx(loss, 0.95), "got {loss}");
    }

    // ── KL Divergence ──────────────────────────────────────────

    #[test]
    fn kl_identical_distributions() {
        let probs: [f32; 4] = [0.25, 0.25, 0.25, 0.25];
        let log_probs: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let loss = kl_divergence(&log_probs, &probs, LossReduction::Sum).unwrap();
        assert!(approx(loss, 0.0), "got {loss}");
    }

    #[test]
    fn kl_different_distributions() {
        let targets = [0.9, 0.1];
        let log_probs = [0.5_f32.ln(), 0.5_f32.ln()];
        let loss = kl_divergence(&log_probs, &targets, LossReduction::Sum).unwrap();
        let expected = 0.9 * (0.9_f32.ln() - 0.5_f32.ln()) + 0.1 * (0.1_f32.ln() - 0.5_f32.ln());
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn kl_zero_target_ignored() {
        let log_probs = [0.5_f32.ln(), 0.5_f32.ln()];
        let targets = [0.0, 1.0];
        let loss = kl_divergence(&log_probs, &targets, LossReduction::Sum).unwrap();
        let expected = 2.0_f32.ln();
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn kl_empty_rejected() {
        assert!(kl_divergence(&[], &[], LossReduction::Mean).is_err());
    }

    #[test]
    fn kl_mean_reduction() {
        let targets: [f32; 2] = [0.5, 0.5];
        let log_probs: Vec<f32> = targets.iter().map(|p| p.ln()).collect();
        let loss = kl_divergence(&log_probs, &targets, LossReduction::Mean).unwrap();
        assert!(approx(loss, 0.0), "got {loss}");
    }

    #[test]
    fn kl_non_negative() {
        // KL divergence is always non-negative (Gibbs' inequality)
        let targets = [0.3, 0.7];
        let log_probs = [0.6_f32.ln(), 0.4_f32.ln()];
        let loss = kl_divergence(&log_probs, &targets, LossReduction::Sum).unwrap();
        assert!(loss >= -TOL, "KL should be non-negative, got {loss}");
    }

    #[test]
    fn kl_single_element() {
        let targets = [1.0];
        let log_probs = [0.5_f32.ln()];
        let loss = kl_divergence(&log_probs, &targets, LossReduction::Sum).unwrap();
        // 1.0 * (ln(1.0) - ln(0.5)) = ln(2)
        assert!(approx(loss, 2.0_f32.ln()), "got {loss}");
    }

    // ── Cosine Similarity Loss ─────────────────────────────────

    #[test]
    fn cosine_identical_vectors() {
        let loss = cosine_similarity_loss(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap();
        assert!(approx(loss, 0.0), "got {loss}");
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let loss = cosine_similarity_loss(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!(approx(loss, 1.0), "got {loss}");
    }

    #[test]
    fn cosine_opposite_vectors() {
        let loss = cosine_similarity_loss(&[1.0, 0.0], &[-1.0, 0.0]).unwrap();
        assert!(approx(loss, 2.0), "got {loss}");
    }

    #[test]
    fn cosine_zero_vector() {
        let loss = cosine_similarity_loss(&[0.0, 0.0], &[1.0, 2.0]).unwrap();
        assert!(approx(loss, 1.0), "got {loss}");
    }

    #[test]
    fn cosine_empty_rejected() {
        assert!(cosine_similarity_loss(&[], &[]).is_err());
    }

    #[test]
    fn cosine_scaled_vectors() {
        let loss_a = cosine_similarity_loss(&[1.0, 2.0], &[2.0, 4.0]).unwrap();
        assert!(approx(loss_a, 0.0), "got {loss_a}");
    }

    #[test]
    fn cosine_single_dimension() {
        let loss = cosine_similarity_loss(&[3.0], &[7.0]).unwrap();
        assert!(approx(loss, 0.0), "parallel 1D vectors, got {loss}");
    }

    // ── Contrastive Loss ───────────────────────────────────────

    #[test]
    fn contrastive_positive_pair_same() {
        let loss = contrastive_loss(&[1.0, 2.0], &[1.0, 2.0], 1.0, 1.0).unwrap();
        assert!(approx(loss, 0.0), "got {loss}");
    }

    #[test]
    fn contrastive_positive_pair_different() {
        let loss = contrastive_loss(&[1.0, 2.0], &[3.0, 4.0], 1.0, 1.0).unwrap();
        assert!(approx(loss, 4.0), "got {loss}");
    }

    #[test]
    fn contrastive_negative_pair_within_margin() {
        let loss = contrastive_loss(&[1.0, 0.0], &[0.0, 1.0], 0.0, 5.0).unwrap();
        let dist = 2.0_f32.sqrt();
        let expected = 0.5 * (5.0 - dist).powi(2);
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn contrastive_negative_pair_beyond_margin() {
        let loss = contrastive_loss(&[1.0, 2.0], &[3.0, 4.0], 0.0, 1.0).unwrap();
        assert!(approx(loss, 0.0), "got {loss}");
    }

    #[test]
    fn contrastive_empty_rejected() {
        assert!(contrastive_loss(&[], &[], 1.0, 1.0).is_err());
    }

    // ── Focal Loss ─────────────────────────────────────────────

    #[test]
    fn focal_gamma_zero_equals_weighted_ce() {
        // gamma=0 makes (1-p_t)^0 = 1, so focal = -alpha_t * ln(p_t)
        let fl = focal_loss(&[0.8, 0.3], &[1.0, 0.0], 0.0, 0.5, LossReduction::Sum).unwrap();
        // target=1: -0.5 * ln(0.8), target=0: -0.5 * ln(0.7)
        let expected = -0.5 * 0.8_f32.ln() - 0.5 * 0.7_f32.ln();
        assert!(approx(fl, expected), "got {fl}, expected {expected}");
    }

    #[test]
    fn focal_high_gamma_suppresses_easy() {
        // With high gamma, loss on confident (easy) examples is suppressed
        let easy_low_gamma = focal_loss(&[0.95], &[1.0], 0.0, 1.0, LossReduction::Mean).unwrap();
        let easy_high_gamma = focal_loss(&[0.95], &[1.0], 5.0, 1.0, LossReduction::Mean).unwrap();
        assert!(
            easy_high_gamma < easy_low_gamma,
            "high gamma should reduce easy loss: {easy_high_gamma} >= {easy_low_gamma}"
        );
    }

    #[test]
    fn focal_gamma2_known_value() {
        // p=0.8, target=1, gamma=2, alpha=1
        // FL = -1 * (1-0.8)^2 * ln(0.8) = -0.04 * ln(0.8)
        let fl = focal_loss(&[0.8], &[1.0], 2.0, 1.0, LossReduction::Mean).unwrap();
        let expected = -(0.2_f32).powi(2) * 0.8_f32.ln();
        assert!(approx(fl, expected), "got {fl}, expected {expected}");
    }

    #[test]
    fn focal_hard_example_high_loss() {
        // Hard example (p close to 0) should have higher loss
        let hard = focal_loss(&[0.1], &[1.0], 2.0, 1.0, LossReduction::Mean).unwrap();
        let easy = focal_loss(&[0.9], &[1.0], 2.0, 1.0, LossReduction::Mean).unwrap();
        assert!(hard > easy, "hard={hard} should be > easy={easy}");
    }

    #[test]
    fn focal_negative_target() {
        // target=0: p_t = 1-p, alpha_t = 1-alpha
        let fl = focal_loss(&[0.3], &[0.0], 2.0, 0.75, LossReduction::Mean).unwrap();
        // p_t=0.7, alpha_t=0.25, loss = -0.25 * (0.3)^2 * ln(0.7)
        let expected = -0.25 * (0.3_f32).powi(2) * 0.7_f32.ln();
        assert!(approx(fl, expected), "got {fl}, expected {expected}");
    }

    #[test]
    fn focal_empty_rejected() {
        assert!(focal_loss(&[], &[], 2.0, 0.5, LossReduction::Mean).is_err());
    }

    #[test]
    fn focal_negative_gamma_rejected() {
        assert!(focal_loss(&[0.5], &[1.0], -1.0, 0.5, LossReduction::Mean).is_err());
    }

    #[test]
    fn focal_numerical_stability() {
        let fl = focal_loss(&[0.0, 1.0], &[1.0, 0.0], 2.0, 0.5, LossReduction::Mean).unwrap();
        assert!(fl.is_finite(), "got {fl}");
    }

    #[test]
    fn focal_batch_sum() {
        let batch =
            focal_loss(&[0.8, 0.2, 0.9], &[1.0, 0.0, 1.0], 2.0, 0.5, LossReduction::Sum).unwrap();
        let individual: f32 = [0.8_f32, 0.2, 0.9]
            .iter()
            .zip([1.0_f32, 0.0, 1.0].iter())
            .map(|(&p, &t)| focal_loss(&[p], &[t], 2.0, 0.5, LossReduction::Sum).unwrap())
            .sum();
        assert!(approx(batch, individual), "batch={batch}, individual={individual}");
    }

    // ── Label Smoothing CE ─────────────────────────────────────

    #[test]
    fn label_smoothing_alpha_zero_matches_ce() {
        let logits = [1.0, 2.0, 0.5];
        let (smooth, _) = label_smoothing_ce(&logits, &[1], 3, 0.0, LossReduction::Mean).unwrap();
        let (hard, _) = cross_entropy_loss(&logits, &[1], 3, LossReduction::Mean).unwrap();
        assert!(approx(smooth, hard), "smooth={smooth}, hard={hard}");
    }

    #[test]
    fn label_smoothing_alpha_01() {
        let logits = [2.0, 1.0, 0.0];
        let (loss, _) = label_smoothing_ce(&logits, &[0], 3, 0.1, LossReduction::Mean).unwrap();
        assert!(loss.is_finite() && loss > 0.0, "got {loss}");
        // Smoothed should be slightly higher than hard CE (more entropy)
        let (hard, _) = cross_entropy_loss(&logits, &[0], 3, LossReduction::Mean).unwrap();
        assert!(loss > hard, "smoothed={loss} should be > hard={hard}");
    }

    #[test]
    fn label_smoothing_alpha_05() {
        let logits = [1.0, 1.0, 1.0];
        let (l05, _) = label_smoothing_ce(&logits, &[0], 3, 0.5, LossReduction::Mean).unwrap();
        let (l00, _) = label_smoothing_ce(&logits, &[0], 3, 0.0, LossReduction::Mean).unwrap();
        // With equal logits, smoothing doesn't change loss
        assert!(approx(l05, l00), "l05={l05}, l00={l00}");
    }

    #[test]
    fn label_smoothing_batch() {
        let logits = [2.0, 1.0, 1.0, 2.0];
        let (_, per) = label_smoothing_ce(&logits, &[0, 1], 2, 0.1, LossReduction::Mean).unwrap();
        assert_eq!(per.len(), 2);
        // Symmetric inputs → identical per-sample losses
        assert!(approx(per[0], per[1]), "per[0]={}, per[1]={}", per[0], per[1]);
    }

    #[test]
    fn label_smoothing_invalid_alpha() {
        assert!(label_smoothing_ce(&[1.0, 2.0], &[0], 2, -0.1, LossReduction::Mean).is_err());
        assert!(label_smoothing_ce(&[1.0, 2.0], &[0], 2, 1.1, LossReduction::Mean).is_err());
    }

    #[test]
    fn label_smoothing_empty_rejected() {
        assert!(label_smoothing_ce(&[], &[], 3, 0.1, LossReduction::Mean).is_err());
    }

    #[test]
    fn label_smoothing_target_out_of_range() {
        assert!(label_smoothing_ce(&[1.0, 2.0], &[2], 2, 0.1, LossReduction::Mean).is_err());
    }

    // ── Perplexity ─────────────────────────────────────────────

    #[test]
    fn perplexity_from_zero_ce() {
        assert!(approx(perplexity(0.0), 1.0));
    }

    #[test]
    fn perplexity_from_ln2() {
        assert!(approx(perplexity(2.0_f32.ln()), 2.0));
    }

    #[test]
    fn perplexity_from_known_ce() {
        let (ce, _) =
            cross_entropy_loss(&[0.0, 0.0, 0.0, 0.0], &[0], 4, LossReduction::Mean).unwrap();
        // uniform → CE = ln(4), perplexity = 4
        let ppl = perplexity(ce);
        assert!(approx(ppl, 4.0), "got {ppl}");
    }

    #[test]
    fn perplexity_monotonically_increasing() {
        let p1 = perplexity(1.0);
        let p2 = perplexity(2.0);
        let p3 = perplexity(3.0);
        assert!(p1 < p2 && p2 < p3, "p1={p1}, p2={p2}, p3={p3}");
    }

    // ── Gradient Accumulation ──────────────────────────────────

    #[test]
    fn grad_accumulate_basic() {
        let mut acc = vec![1.0, 2.0, 3.0];
        gradient_accumulate(&mut acc, &[0.1, 0.2, 0.3]).unwrap();
        assert!(approx(acc[0], 1.1));
        assert!(approx(acc[1], 2.2));
        assert!(approx(acc[2], 3.3));
    }

    #[test]
    fn grad_accumulate_zero_source() {
        let mut acc = vec![1.0, 2.0];
        gradient_accumulate(&mut acc, &[0.0, 0.0]).unwrap();
        assert!(approx(acc[0], 1.0));
        assert!(approx(acc[1], 2.0));
    }

    #[test]
    fn grad_accumulate_negative() {
        let mut acc = vec![1.0, -1.0];
        gradient_accumulate(&mut acc, &[-0.5, 0.5]).unwrap();
        assert!(approx(acc[0], 0.5));
        assert!(approx(acc[1], -0.5));
    }

    #[test]
    fn grad_accumulate_multiple_steps() {
        let mut acc = [0.0; 3];
        for _ in 0..4 {
            gradient_accumulate(&mut acc, &[1.0, 2.0, 3.0]).unwrap();
        }
        assert!(approx(acc[0], 4.0));
        assert!(approx(acc[1], 8.0));
        assert!(approx(acc[2], 12.0));
    }

    #[test]
    fn grad_accumulate_length_mismatch() {
        let mut acc = vec![1.0, 2.0];
        assert!(gradient_accumulate(&mut acc, &[1.0]).is_err());
    }

    #[test]
    fn grad_accumulate_empty() {
        let mut acc: Vec<f32> = vec![];
        // Empty is technically OK (0-length match)
        gradient_accumulate(&mut acc, &[]).unwrap();
    }

    // ── Gradient Clip Norm ─────────────────────────────────────

    #[test]
    fn grad_clip_norm_below_threshold() {
        let mut grads = vec![1.0, 2.0, 3.0];
        let orig = grads.clone();
        let norm = gradient_clip_norm(&mut grads, 100.0).unwrap();
        // norm = sqrt(14) ≈ 3.74, below 100 → no change
        assert!(approx(norm, 14.0_f32.sqrt()));
        for (a, b) in grads.iter().zip(orig.iter()) {
            assert!(approx(*a, *b));
        }
    }

    #[test]
    fn grad_clip_norm_above_threshold() {
        let mut grads = vec![3.0, 4.0]; // norm = 5
        let norm = gradient_clip_norm(&mut grads, 2.5).unwrap();
        assert!(approx(norm, 5.0));
        let new_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!(approx(new_norm, 2.5), "clipped norm={new_norm}");
    }

    #[test]
    fn grad_clip_norm_preserves_direction() {
        let mut grads = vec![3.0, 4.0];
        gradient_clip_norm(&mut grads, 1.0).unwrap();
        // ratio should be preserved: 3/4
        let ratio = grads[0] / grads[1];
        assert!(approx(ratio, 0.75), "direction changed: ratio={ratio}");
    }

    #[test]
    fn grad_clip_norm_exactly_at_threshold() {
        let mut grads = vec![3.0, 4.0]; // norm = 5
        gradient_clip_norm(&mut grads, 5.0).unwrap();
        // Exactly at threshold → no change
        assert!(approx(grads[0], 3.0));
        assert!(approx(grads[1], 4.0));
    }

    #[test]
    fn grad_clip_norm_zero_grads() {
        let mut grads = vec![0.0, 0.0];
        let norm = gradient_clip_norm(&mut grads, 1.0).unwrap();
        assert!(approx(norm, 0.0));
        assert!(approx(grads[0], 0.0));
    }

    #[test]
    fn grad_clip_norm_invalid_max() {
        assert!(gradient_clip_norm(&mut [1.0], 0.0).is_err());
        assert!(gradient_clip_norm(&mut [1.0], -1.0).is_err());
    }

    #[test]
    fn grad_clip_norm_single_element() {
        let mut grads = [10.0];
        gradient_clip_norm(&mut grads, 3.0).unwrap();
        assert!(approx(grads[0], 3.0), "got {}", grads[0]);
    }

    // ── Gradient Clip Value ────────────────────────────────────

    #[test]
    fn grad_clip_value_below_threshold() {
        let mut grads = vec![0.5, -0.3, 0.8];
        let clipped = gradient_clip_value(&mut grads, 1.0).unwrap();
        assert_eq!(clipped, 0);
        assert!(approx(grads[0], 0.5));
    }

    #[test]
    fn grad_clip_value_above_threshold() {
        let mut grads = vec![5.0, -3.0, 0.5];
        let clipped = gradient_clip_value(&mut grads, 1.0).unwrap();
        assert_eq!(clipped, 2);
        assert!(approx(grads[0], 1.0));
        assert!(approx(grads[1], -1.0));
        assert!(approx(grads[2], 0.5));
    }

    #[test]
    fn grad_clip_value_all_clipped() {
        let mut grads = vec![10.0, -10.0, 20.0, -20.0];
        let clipped = gradient_clip_value(&mut grads, 0.1).unwrap();
        assert_eq!(clipped, 4);
        for &g in &grads {
            assert!(g.abs() <= 0.1 + TOL);
        }
    }

    #[test]
    fn grad_clip_value_at_boundary() {
        let mut grads = vec![1.0, -1.0];
        let clipped = gradient_clip_value(&mut grads, 1.0).unwrap();
        assert_eq!(clipped, 0);
    }

    #[test]
    fn grad_clip_value_invalid_max() {
        assert!(gradient_clip_value(&mut [1.0], 0.0).is_err());
        assert!(gradient_clip_value(&mut [1.0], -1.0).is_err());
    }

    #[test]
    fn grad_clip_value_empty() {
        let mut grads: Vec<f32> = vec![];
        let clipped = gradient_clip_value(&mut grads, 1.0).unwrap();
        assert_eq!(clipped, 0);
    }

    // ── Reduction & Edge Cases ─────────────────────────────────

    #[test]
    fn reduction_none_equals_sum() {
        let loss_none = mse_loss(&[1.0, 2.0], &[3.0, 4.0], LossReduction::None).unwrap();
        let loss_sum = mse_loss(&[1.0, 2.0], &[3.0, 4.0], LossReduction::Sum).unwrap();
        assert!(approx(loss_none, loss_sum));
    }

    #[test]
    fn all_loss_reductions_consistent() {
        let preds = [1.0, 2.0, 3.0, 4.0];
        let tgts = [1.5, 2.5, 3.5, 4.5];
        let mean = mse_loss(&preds, &tgts, LossReduction::Mean).unwrap();
        let sum = mse_loss(&preds, &tgts, LossReduction::Sum).unwrap();
        assert!(approx(mean * 4.0, sum), "mean*4={}, sum={sum}", mean * 4.0);
    }

    #[test]
    fn losses_non_negative_on_random_inputs() {
        let p = [0.3, 0.7, 0.5, 0.1];
        let t = [0.5, 0.2, 0.8, 0.9];
        assert!(mse_loss(&p, &t, LossReduction::Mean).unwrap() >= 0.0);
        assert!(l1_loss(&p, &t, LossReduction::Mean).unwrap() >= 0.0);
        assert!(smooth_l1_loss(&p, &t, 1.0, LossReduction::Mean).unwrap() >= 0.0);
        assert!(binary_cross_entropy(&p, &t, LossReduction::Mean).unwrap() >= 0.0);
    }
}
