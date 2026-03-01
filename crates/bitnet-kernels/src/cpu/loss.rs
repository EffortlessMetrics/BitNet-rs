//! CPU loss function kernels.
//!
//! Provides common loss functions for training and evaluation:
//! cross-entropy, binary cross-entropy, MSE, L1, smooth L1,
//! KL divergence, cosine similarity, and contrastive loss.

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
        // Single sample, 3 classes, target=1
        let logits = [1.0, 2.0, 0.5];
        let (loss, per) = cross_entropy_loss(&logits, &[1], 3, LossReduction::Mean).unwrap();
        // -log(softmax(2.0)) among [1, 2, 0.5]
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
        // Both have same structure: correct class logit=1.0
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

    // ── Binary Cross-Entropy ───────────────────────────────────

    #[test]
    fn bce_perfect_prediction() {
        let loss = binary_cross_entropy(&[1.0, 0.0], &[1.0, 0.0], LossReduction::Mean).unwrap();
        // With clamping, not exactly 0 but very small
        assert!(loss < 0.01, "got {loss}");
    }

    #[test]
    fn bce_worst_prediction() {
        let loss = binary_cross_entropy(&[0.0, 1.0], &[1.0, 0.0], LossReduction::Mean).unwrap();
        // Should be very large (clamped avoids infinity)
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

    // ── MSE ────────────────────────────────────────────────────

    #[test]
    fn mse_zero_error() {
        let loss = mse_loss(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], LossReduction::Mean).unwrap();
        assert!(approx(loss, 0.0));
    }

    #[test]
    fn mse_known_value() {
        // (1-3)^2 + (2-4)^2 = 4 + 4 = 8; mean = 4
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

    // ── L1 ─────────────────────────────────────────────────────

    #[test]
    fn l1_zero_error() {
        let loss = l1_loss(&[1.0, 2.0], &[1.0, 2.0], LossReduction::Mean).unwrap();
        assert!(approx(loss, 0.0));
    }

    #[test]
    fn l1_known_value() {
        // |1-3| + |2-4| = 2 + 2 = 4; mean = 2
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

    // ── Smooth L1 ──────────────────────────────────────────────

    #[test]
    fn smooth_l1_quadratic_regime() {
        // |d| = 0.5 < beta=1.0 → 0.5 * 0.25 / 1.0 = 0.125
        let loss = smooth_l1_loss(&[1.0], &[1.5], 1.0, LossReduction::Mean).unwrap();
        assert!(approx(loss, 0.125), "got {loss}");
    }

    #[test]
    fn smooth_l1_linear_regime() {
        // |d| = 2.0 >= beta=1.0 → 2.0 - 0.5 = 1.5
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
        // 0.9 * (ln(0.9) - ln(0.5)) + 0.1 * (ln(0.1) - ln(0.5))
        let expected = 0.9 * (0.9_f32.ln() - 0.5_f32.ln()) + 0.1 * (0.1_f32.ln() - 0.5_f32.ln());
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn kl_zero_target_ignored() {
        let log_probs = [0.5_f32.ln(), 0.5_f32.ln()];
        let targets = [0.0, 1.0];
        let loss = kl_divergence(&log_probs, &targets, LossReduction::Sum).unwrap();
        // Only second term: 1.0 * (ln(1.0) - ln(0.5)) = ln(2)
        let expected = 2.0_f32.ln();
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn kl_empty_rejected() {
        assert!(kl_divergence(&[], &[], LossReduction::Mean).is_err());
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
        assert!(approx(loss, 1.0), "got {loss}"); // treat as orthogonal
    }

    #[test]
    fn cosine_empty_rejected() {
        assert!(cosine_similarity_loss(&[], &[]).is_err());
    }

    // ── Contrastive Loss ───────────────────────────────────────

    #[test]
    fn contrastive_positive_pair_same() {
        // Same vectors, positive pair → 0.5 * 1.0 * 0 = 0
        let loss = contrastive_loss(&[1.0, 2.0], &[1.0, 2.0], 1.0, 1.0).unwrap();
        assert!(approx(loss, 0.0), "got {loss}");
    }

    #[test]
    fn contrastive_positive_pair_different() {
        // dist_sq = (1-3)^2 + (2-4)^2 = 8; loss = 0.5 * 1.0 * 8 = 4
        let loss = contrastive_loss(&[1.0, 2.0], &[3.0, 4.0], 1.0, 1.0).unwrap();
        assert!(approx(loss, 4.0), "got {loss}");
    }

    #[test]
    fn contrastive_negative_pair_within_margin() {
        // dist = sqrt(2) ≈ 1.414, margin = 5.0
        // loss = 0.5 * 1.0 * (5.0 - 1.414)^2 ≈ 0.5 * 12.858 ≈ 6.429
        let loss = contrastive_loss(&[1.0, 0.0], &[0.0, 1.0], 0.0, 5.0).unwrap();
        let dist = 2.0_f32.sqrt();
        let expected = 0.5 * (5.0 - dist).powi(2);
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn contrastive_negative_pair_beyond_margin() {
        // dist = sqrt(8) ≈ 2.83, margin = 1.0 → max(0, 1-2.83)^2 = 0
        let loss = contrastive_loss(&[1.0, 2.0], &[3.0, 4.0], 0.0, 1.0).unwrap();
        assert!(approx(loss, 0.0), "got {loss}");
    }

    #[test]
    fn contrastive_empty_rejected() {
        assert!(contrastive_loss(&[], &[], 1.0, 1.0).is_err());
    }

    // ── Numerical Stability & Edge Cases ───────────────────────

    #[test]
    fn cross_entropy_large_logits() {
        // Very large logits should not overflow with log-sum-exp
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
    fn bce_numerical_stability() {
        // Predictions at boundaries should not produce NaN/Inf
        let loss = binary_cross_entropy(&[0.0, 1.0], &[0.0, 1.0], LossReduction::Mean).unwrap();
        assert!(loss.is_finite(), "got {loss}");
    }

    #[test]
    fn mse_large_values() {
        let loss = mse_loss(&[1e6], &[-1e6], LossReduction::Mean).unwrap();
        assert!(loss.is_finite(), "got {loss}");
        assert!(loss > 0.0);
    }

    #[test]
    fn smooth_l1_at_boundary() {
        // |d| exactly at beta boundary
        let loss = smooth_l1_loss(&[0.0], &[1.0], 1.0, LossReduction::Mean).unwrap();
        // |d|=1.0 is not < beta=1.0, so linear: 1.0 - 0.5 = 0.5
        assert!(approx(loss, 0.5), "got {loss}");
    }

    #[test]
    fn kl_mean_reduction() {
        let targets: [f32; 2] = [0.5, 0.5];
        let log_probs: Vec<f32> = targets.iter().map(|p| p.ln()).collect();
        let loss = kl_divergence(&log_probs, &targets, LossReduction::Mean).unwrap();
        assert!(approx(loss, 0.0), "got {loss}");
    }

    #[test]
    fn cosine_scaled_vectors() {
        // Scaling shouldn't affect cosine similarity
        let loss_a = cosine_similarity_loss(&[1.0, 2.0], &[2.0, 4.0]).unwrap();
        assert!(approx(loss_a, 0.0), "got {loss_a}");
    }

    #[test]
    fn reduction_none_equals_sum() {
        // LossReduction::None behaves like Sum
        let loss_none = mse_loss(&[1.0, 2.0], &[3.0, 4.0], LossReduction::None).unwrap();
        let loss_sum = mse_loss(&[1.0, 2.0], &[3.0, 4.0], LossReduction::Sum).unwrap();
        assert!(approx(loss_none, loss_sum));
    }
}
