//! Extended CPU loss function kernels for inference evaluation.
//!
//! Provides forward-only loss computations useful for model evaluation,
//! distillation scoring, and perplexity measurement during inference:
//! cross-entropy with label smoothing, binary cross-entropy, MSE, MAE,
//! Huber, KL divergence, cosine embedding, focal loss, and perplexity.

use core::fmt;

// ── Error Type ─────────────────────────────────────────────────────

/// Errors produced by loss function kernels.
#[derive(Debug, Clone, PartialEq)]
pub enum LossError {
    /// Input slices have incompatible lengths.
    DimensionMismatch { expected: usize, actual: usize },
    /// An input slice was empty when non-empty input is required.
    EmptyInput,
    /// A numeric parameter is outside its valid range.
    InvalidParameter { name: &'static str, reason: String },
    /// Probability values are outside `[0, 1]`.
    InvalidProbability { index: usize, value: f32 },
}

impl fmt::Display for LossError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::EmptyInput => write!(f, "input must not be empty"),
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::InvalidProbability { index, value } => {
                write!(f, "invalid probability at index {index}: {value} not in [0, 1]")
            }
        }
    }
}

impl std::error::Error for LossError {}

// ── Reduction Mode ─────────────────────────────────────────────────

/// How to reduce per-element losses into a scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionMode {
    /// Arithmetic mean of per-element losses.
    Mean,
    /// Sum of per-element losses.
    Sum,
    /// No reduction — return the sum as a convenience scalar, but the
    /// caller should use the per-element vector when available.
    None,
}

// ── Helpers ────────────────────────────────────────────────────────

/// Numerical stability clamp for log arguments.
const EPS: f32 = 1e-7;

fn validate_same_len(a: &[f32], b: &[f32]) -> Result<(), LossError> {
    if a.is_empty() {
        return Err(LossError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(LossError::DimensionMismatch { expected: a.len(), actual: b.len() });
    }
    Ok(())
}

fn reduce(values: &[f32], mode: ReductionMode) -> f32 {
    match mode {
        ReductionMode::Mean => values.iter().sum::<f32>() / values.len() as f32,
        ReductionMode::Sum | ReductionMode::None => values.iter().sum(),
    }
}

/// Numerically-stable log-softmax for a single row of logits.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum_exp = max + sum_exp.ln();
    logits.iter().map(|&x| x - log_sum_exp).collect()
}

// ── Loss Functions ─────────────────────────────────────────────────

/// Cross-entropy loss over a batch of logits and integer class targets,
/// with optional label smoothing.
///
/// `logits` is `[batch_size, num_classes]` in row-major order.
/// `targets` contains the correct class index for each sample.
/// `label_smoothing` in `[0, 1)` mixes uniform distribution into targets.
///
/// Returns `(scalar_loss, per_sample_losses)`.
pub fn cross_entropy_loss(
    logits: &[f32],
    targets: &[usize],
    num_classes: usize,
    label_smoothing: f32,
    reduction: ReductionMode,
) -> Result<(f32, Vec<f32>), LossError> {
    if targets.is_empty() {
        return Err(LossError::EmptyInput);
    }
    if num_classes == 0 {
        return Err(LossError::InvalidParameter {
            name: "num_classes",
            reason: "must be > 0".to_string(),
        });
    }
    let batch_size = targets.len();
    if logits.len() != batch_size * num_classes {
        return Err(LossError::DimensionMismatch {
            expected: batch_size * num_classes,
            actual: logits.len(),
        });
    }
    if !(0.0..1.0).contains(&label_smoothing) {
        return Err(LossError::InvalidParameter {
            name: "label_smoothing",
            reason: format!("must be in [0, 1), got {label_smoothing}"),
        });
    }
    for (i, &t) in targets.iter().enumerate() {
        if t >= num_classes {
            return Err(LossError::InvalidParameter {
                name: "targets",
                reason: format!("target[{i}]={t} >= num_classes={num_classes}"),
            });
        }
    }

    let smooth = label_smoothing;
    let confidence = 1.0 - smooth;
    let uniform = smooth / num_classes as f32;

    let mut per_sample = Vec::with_capacity(batch_size);
    for (i, &target) in targets.iter().enumerate() {
        let row = &logits[i * num_classes..(i + 1) * num_classes];
        let lsp = log_softmax(row);

        let loss = if smooth == 0.0 {
            -lsp[target]
        } else {
            let mut l = 0.0_f32;
            for (c, &lp) in lsp.iter().enumerate() {
                let q = if c == target { confidence + uniform } else { uniform };
                l -= q * lp;
            }
            l
        };
        per_sample.push(loss);
    }

    let scalar = reduce(&per_sample, reduction);
    Ok((scalar, per_sample))
}

/// Binary cross-entropy loss.
///
/// `predictions` should be probabilities in `(0, 1)`. Values are clamped
/// to `[EPS, 1-EPS]` for numerical stability.
pub fn binary_cross_entropy(
    predictions: &[f32],
    targets: &[f32],
    reduction: ReductionMode,
) -> Result<f32, LossError> {
    validate_same_len(predictions, targets)?;
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

/// Mean squared error (L2) loss.
pub fn mse_loss(
    predictions: &[f32],
    targets: &[f32],
    reduction: ReductionMode,
) -> Result<f32, LossError> {
    validate_same_len(predictions, targets)?;
    let losses: Vec<f32> =
        predictions.iter().zip(targets.iter()).map(|(&p, &t)| (p - t).powi(2)).collect();
    Ok(reduce(&losses, reduction))
}

/// Mean absolute error (L1) loss.
pub fn mae_loss(
    predictions: &[f32],
    targets: &[f32],
    reduction: ReductionMode,
) -> Result<f32, LossError> {
    validate_same_len(predictions, targets)?;
    let losses: Vec<f32> =
        predictions.iter().zip(targets.iter()).map(|(&p, &t)| (p - t).abs()).collect();
    Ok(reduce(&losses, reduction))
}

/// Huber (smooth L1) loss with configurable `delta` transition point.
///
/// Quadratic when `|error| <= delta`, linear when `|error| > delta`.
pub fn huber_loss(
    predictions: &[f32],
    targets: &[f32],
    delta: f32,
    reduction: ReductionMode,
) -> Result<f32, LossError> {
    validate_same_len(predictions, targets)?;
    if delta <= 0.0 {
        return Err(LossError::InvalidParameter {
            name: "delta",
            reason: format!("must be > 0, got {delta}"),
        });
    }
    let losses: Vec<f32> = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| {
            let d = (p - t).abs();
            if d <= delta { 0.5 * d * d } else { delta * (d - 0.5 * delta) }
        })
        .collect();
    Ok(reduce(&losses, reduction))
}

/// Kullback-Leibler divergence: `D_KL(target || exp(log_probs))`.
///
/// `log_probs` are **log-probabilities** (e.g. after log-softmax).
/// `targets` are a probability distribution (should sum to ≈1).
pub fn kl_divergence(
    log_probs: &[f32],
    targets: &[f32],
    reduction: ReductionMode,
) -> Result<f32, LossError> {
    validate_same_len(log_probs, targets)?;
    let losses: Vec<f32> = log_probs
        .iter()
        .zip(targets.iter())
        .map(|(&lp, &t)| if t <= 0.0 { 0.0 } else { t * (t.ln() - lp) })
        .collect();
    Ok(reduce(&losses, reduction))
}

/// Cosine embedding loss for pairs of vectors.
///
/// Returns `1 - cos(a, b)` when `target = 1` (similar pair), and
/// `max(0, cos(a, b) - margin)` when `target = -1` (dissimilar pair).
///
/// `target` should be `1` or `-1`.
pub fn cosine_embedding_loss(
    a: &[f32],
    b: &[f32],
    target: i32,
    margin: f32,
) -> Result<f32, LossError> {
    validate_same_len(a, b)?;
    if target != 1 && target != -1 {
        return Err(LossError::InvalidParameter {
            name: "target",
            reason: format!("must be 1 or -1, got {target}"),
        });
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm_a * norm_b;
    let cos_sim = if denom < EPS { 0.0 } else { dot / denom };

    if target == 1 { Ok(1.0 - cos_sim) } else { Ok((cos_sim - margin).max(0.0)) }
}

/// Focal loss for addressing class imbalance.
///
/// `probs` are predicted probabilities for the **true** class.
/// `gamma` controls the focusing: higher gamma down-weights easy examples.
/// When `gamma = 0`, this reduces to standard cross-entropy.
///
/// `alpha` is an optional per-sample weighting factor (pass `None` for 1.0).
pub fn focal_loss(
    probs: &[f32],
    gamma: f32,
    alpha: Option<&[f32]>,
    reduction: ReductionMode,
) -> Result<f32, LossError> {
    if probs.is_empty() {
        return Err(LossError::EmptyInput);
    }
    if gamma < 0.0 {
        return Err(LossError::InvalidParameter {
            name: "gamma",
            reason: format!("must be >= 0, got {gamma}"),
        });
    }
    if let Some(a) = alpha
        && a.len() != probs.len()
    {
        return Err(LossError::DimensionMismatch { expected: probs.len(), actual: a.len() });
    }
    for (i, &p) in probs.iter().enumerate() {
        if !(0.0..=1.0).contains(&p) {
            return Err(LossError::InvalidProbability { index: i, value: p });
        }
    }

    let losses: Vec<f32> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let p_clamped = p.clamp(EPS, 1.0 - EPS);
            let focal_weight = (1.0 - p_clamped).powf(gamma);
            let ce = -p_clamped.ln();
            let a = alpha.map_or(1.0, |a| a[i]);
            a * focal_weight * ce
        })
        .collect();
    Ok(reduce(&losses, reduction))
}

/// Perplexity: `exp(cross_entropy)`.
///
/// A standard evaluation metric for language models. Lower is better.
/// Accepts the same arguments as [`cross_entropy_loss`] and returns the
/// mean perplexity across the batch.
pub fn perplexity(logits: &[f32], targets: &[usize], num_classes: usize) -> Result<f32, LossError> {
    let (mean_ce, _) = cross_entropy_loss(logits, targets, num_classes, 0.0, ReductionMode::Mean)?;
    Ok(mean_ce.exp())
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::too_many_lines)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    // ── Cross-Entropy ──────────────────────────────────────────

    #[test]
    fn ce_known_value() {
        let logits = [1.0, 2.0, 0.5];
        let (loss, per) = cross_entropy_loss(&logits, &[1], 3, 0.0, ReductionMode::Mean).unwrap();
        let max_l = 2.0_f32;
        let lse = max_l + ((1.0 - max_l).exp() + 0.0_f32.exp() + (0.5 - max_l).exp()).ln();
        let expected = lse - 2.0;
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
        assert_eq!(per.len(), 1);
    }

    #[test]
    fn ce_batch() {
        let logits = [1.0, 0.0, 0.0, 1.0];
        let (loss, per) =
            cross_entropy_loss(&logits, &[0, 1], 2, 0.0, ReductionMode::Mean).unwrap();
        assert_eq!(per.len(), 2);
        assert!(approx(per[0], per[1]));
        assert!(approx(loss, per[0]));
    }

    #[test]
    fn ce_sum_reduction() {
        let logits = [1.0, 0.0, 0.0, 1.0];
        let (loss, per) = cross_entropy_loss(&logits, &[0, 1], 2, 0.0, ReductionMode::Sum).unwrap();
        let expected_sum: f32 = per.iter().sum();
        assert!(approx(loss, expected_sum));
    }

    #[test]
    fn ce_label_smoothing_increases_loss() {
        let logits = [3.0, 0.0, 0.0];
        let (no_smooth, _) =
            cross_entropy_loss(&logits, &[0], 3, 0.0, ReductionMode::Mean).unwrap();
        let (smooth, _) = cross_entropy_loss(&logits, &[0], 3, 0.1, ReductionMode::Mean).unwrap();
        assert!(
            smooth > no_smooth,
            "label smoothing should increase loss: smooth={smooth}, no_smooth={no_smooth}"
        );
    }

    #[test]
    fn ce_label_smoothing_uniform_distribution() {
        // With smoothing=0 and uniform logits, loss equals ln(num_classes)
        let logits = [0.0, 0.0, 0.0, 0.0];
        let (loss, _) = cross_entropy_loss(&logits, &[0], 4, 0.0, ReductionMode::Mean).unwrap();
        let expected = (4.0_f32).ln();
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn ce_target_out_of_range() {
        let logits = [1.0, 2.0, 3.0];
        assert!(cross_entropy_loss(&logits, &[3], 3, 0.0, ReductionMode::Mean).is_err());
    }

    #[test]
    fn ce_empty_targets() {
        assert!(cross_entropy_loss(&[], &[], 3, 0.0, ReductionMode::Mean).is_err());
    }

    #[test]
    fn ce_length_mismatch() {
        let logits = [1.0, 2.0];
        assert!(cross_entropy_loss(&logits, &[0, 1], 3, 0.0, ReductionMode::Mean).is_err());
    }

    #[test]
    fn ce_invalid_label_smoothing() {
        let logits = [1.0, 0.0];
        assert!(cross_entropy_loss(&logits, &[0], 2, 1.0, ReductionMode::Mean).is_err());
        assert!(cross_entropy_loss(&logits, &[0], 2, -0.1, ReductionMode::Mean).is_err());
    }

    #[test]
    fn ce_large_logits_stable() {
        let logits = [1000.0, 0.0, 0.0];
        let (loss, _) = cross_entropy_loss(&logits, &[0], 3, 0.0, ReductionMode::Mean).unwrap();
        assert!(loss.is_finite(), "got {loss}");
        assert!(loss >= 0.0);
    }

    // ── Binary Cross-Entropy ───────────────────────────────────

    #[test]
    fn bce_perfect_prediction() {
        let loss = binary_cross_entropy(&[1.0, 0.0], &[1.0, 0.0], ReductionMode::Mean).unwrap();
        assert!(loss < 0.01, "got {loss}");
    }

    #[test]
    fn bce_worst_prediction() {
        let loss = binary_cross_entropy(&[0.0, 1.0], &[1.0, 0.0], ReductionMode::Mean).unwrap();
        assert!(loss > 10.0, "got {loss}");
    }

    #[test]
    fn bce_half_probability() {
        let loss = binary_cross_entropy(&[0.5], &[1.0], ReductionMode::Mean).unwrap();
        let expected = -(0.5_f32.ln());
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn bce_boundary_values_stable() {
        let loss = binary_cross_entropy(&[0.0, 1.0], &[0.0, 1.0], ReductionMode::Mean).unwrap();
        assert!(loss.is_finite(), "got {loss}");
    }

    #[test]
    fn bce_empty_rejected() {
        assert!(binary_cross_entropy(&[], &[], ReductionMode::Mean).is_err());
    }

    #[test]
    fn bce_length_mismatch() {
        assert!(binary_cross_entropy(&[0.5], &[1.0, 0.0], ReductionMode::Mean).is_err());
    }

    // ── MSE ────────────────────────────────────────────────────

    #[test]
    fn mse_zero_error() {
        let loss = mse_loss(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], ReductionMode::Mean).unwrap();
        assert!(approx(loss, 0.0));
    }

    #[test]
    fn mse_known_value() {
        let loss = mse_loss(&[1.0, 2.0], &[3.0, 4.0], ReductionMode::Mean).unwrap();
        assert!(approx(loss, 4.0), "got {loss}");
    }

    #[test]
    fn mse_sum_reduction() {
        let loss = mse_loss(&[1.0, 2.0], &[3.0, 4.0], ReductionMode::Sum).unwrap();
        assert!(approx(loss, 8.0), "got {loss}");
    }

    #[test]
    fn mse_none_reduction() {
        let loss = mse_loss(&[1.0, 2.0], &[3.0, 4.0], ReductionMode::None).unwrap();
        assert!(approx(loss, 8.0), "got {loss}");
    }

    #[test]
    fn mse_empty_rejected() {
        assert!(mse_loss(&[], &[], ReductionMode::Mean).is_err());
    }

    // ── MAE ────────────────────────────────────────────────────

    #[test]
    fn mae_zero_error() {
        let loss = mae_loss(&[1.0, 2.0], &[1.0, 2.0], ReductionMode::Mean).unwrap();
        assert!(approx(loss, 0.0));
    }

    #[test]
    fn mae_known_value() {
        let loss = mae_loss(&[1.0, 2.0], &[3.0, 4.0], ReductionMode::Mean).unwrap();
        assert!(approx(loss, 2.0), "got {loss}");
    }

    #[test]
    fn mae_sum_reduction() {
        let loss = mae_loss(&[1.0, 2.0], &[3.0, 4.0], ReductionMode::Sum).unwrap();
        assert!(approx(loss, 4.0), "got {loss}");
    }

    #[test]
    fn mae_negative_values() {
        let loss = mae_loss(&[-1.0, -2.0], &[1.0, 2.0], ReductionMode::Mean).unwrap();
        assert!(approx(loss, 3.0), "got {loss}");
    }

    #[test]
    fn mae_empty_rejected() {
        assert!(mae_loss(&[], &[], ReductionMode::Mean).is_err());
    }

    // ── Huber ──────────────────────────────────────────────────

    #[test]
    fn huber_quadratic_regime() {
        // |d|=0.5 <= delta=1.0 → 0.5 * 0.25 = 0.125
        let loss = huber_loss(&[1.0], &[1.5], 1.0, ReductionMode::Mean).unwrap();
        assert!(approx(loss, 0.125), "got {loss}");
    }

    #[test]
    fn huber_linear_regime() {
        // |d|=2.0 > delta=1.0 → 1.0 * (2.0 - 0.5) = 1.5
        let loss = huber_loss(&[1.0], &[3.0], 1.0, ReductionMode::Mean).unwrap();
        assert!(approx(loss, 1.5), "got {loss}");
    }

    #[test]
    fn huber_at_delta_boundary() {
        // |d|=1.0 == delta=1.0 → quadratic: 0.5 * 1.0 = 0.5
        let loss = huber_loss(&[0.0], &[1.0], 1.0, ReductionMode::Mean).unwrap();
        assert!(approx(loss, 0.5), "got {loss}");
    }

    #[test]
    fn huber_equivalent_to_mse_small_errors() {
        // For |d| <= delta, huber = 0.5 * d^2, mse = d^2.
        // So huber = 0.5 * mse for a single element.
        let delta = 10.0;
        let pred = &[1.0, 2.0];
        let tgt = &[1.5, 2.5];
        let hub = huber_loss(pred, tgt, delta, ReductionMode::Sum).unwrap();
        let mse = mse_loss(pred, tgt, ReductionMode::Sum).unwrap();
        assert!(approx(hub, 0.5 * mse), "huber={hub}, 0.5*mse={}", 0.5 * mse);
    }

    #[test]
    fn huber_equivalent_to_mae_large_errors() {
        // For |d| >> delta, huber ≈ delta * |d| - 0.5 * delta^2.
        // With delta very small, huber approaches delta * mae.
        let delta = 0.001;
        let pred = &[0.0];
        let tgt = &[100.0];
        let hub = huber_loss(pred, tgt, delta, ReductionMode::Sum).unwrap();
        let mae = mae_loss(pred, tgt, ReductionMode::Sum).unwrap();
        // huber = delta * (mae - 0.5 * delta)
        let expected = delta * (mae - 0.5 * delta);
        assert!(approx(hub, expected), "huber={hub}, expected={expected}");
    }

    #[test]
    fn huber_zero_error() {
        let loss = huber_loss(&[2.0, 3.0], &[2.0, 3.0], 1.0, ReductionMode::Mean).unwrap();
        assert!(approx(loss, 0.0));
    }

    #[test]
    fn huber_invalid_delta() {
        assert!(huber_loss(&[1.0], &[2.0], 0.0, ReductionMode::Mean).is_err());
        assert!(huber_loss(&[1.0], &[2.0], -1.0, ReductionMode::Mean).is_err());
    }

    #[test]
    fn huber_empty_rejected() {
        assert!(huber_loss(&[], &[], 1.0, ReductionMode::Mean).is_err());
    }

    // ── KL Divergence ──────────────────────────────────────────

    #[test]
    fn kl_identical_distributions() {
        let probs: [f32; 4] = [0.25, 0.25, 0.25, 0.25];
        let log_probs: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let loss = kl_divergence(&log_probs, &probs, ReductionMode::Sum).unwrap();
        assert!(approx(loss, 0.0), "got {loss}");
    }

    #[test]
    fn kl_different_distributions() {
        let targets = [0.9, 0.1];
        let log_probs = [0.5_f32.ln(), 0.5_f32.ln()];
        let loss = kl_divergence(&log_probs, &targets, ReductionMode::Sum).unwrap();
        let expected = 0.9 * (0.9_f32.ln() - 0.5_f32.ln()) + 0.1 * (0.1_f32.ln() - 0.5_f32.ln());
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn kl_non_negative() {
        // KL divergence is always >= 0 (Gibbs' inequality).
        let targets = [0.7, 0.2, 0.1];
        let log_probs = [0.3_f32.ln(), 0.5_f32.ln(), 0.2_f32.ln()];
        let loss = kl_divergence(&log_probs, &targets, ReductionMode::Sum).unwrap();
        assert!(loss >= -TOL, "KL should be non-negative, got {loss}");
    }

    #[test]
    fn kl_zero_target_ignored() {
        let log_probs = [0.5_f32.ln(), 0.5_f32.ln()];
        let targets = [0.0, 1.0];
        let loss = kl_divergence(&log_probs, &targets, ReductionMode::Sum).unwrap();
        let expected = 2.0_f32.ln();
        assert!(approx(loss, expected), "got {loss}, expected {expected}");
    }

    #[test]
    fn kl_empty_rejected() {
        assert!(kl_divergence(&[], &[], ReductionMode::Mean).is_err());
    }

    #[test]
    fn kl_mean_reduction() {
        let targets: [f32; 2] = [0.5, 0.5];
        let log_probs: Vec<f32> = targets.iter().map(|p| p.ln()).collect();
        let loss = kl_divergence(&log_probs, &targets, ReductionMode::Mean).unwrap();
        assert!(approx(loss, 0.0), "got {loss}");
    }

    // ── Cosine Embedding Loss ──────────────────────────────────

    #[test]
    fn cosine_emb_identical_vectors() {
        let loss = cosine_embedding_loss(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1, 0.0).unwrap();
        assert!(approx(loss, 0.0), "got {loss}");
    }

    #[test]
    fn cosine_emb_orthogonal_vectors() {
        let loss = cosine_embedding_loss(&[1.0, 0.0], &[0.0, 1.0], 1, 0.0).unwrap();
        assert!(approx(loss, 1.0), "got {loss}");
    }

    #[test]
    fn cosine_emb_opposite_vectors() {
        let loss = cosine_embedding_loss(&[1.0, 0.0], &[-1.0, 0.0], 1, 0.0).unwrap();
        assert!(approx(loss, 2.0), "got {loss}");
    }

    #[test]
    fn cosine_emb_dissimilar_pair_within_margin() {
        // target=-1, cos_sim=1.0, margin=0.5 → max(0, 1.0 - 0.5) = 0.5
        let loss = cosine_embedding_loss(&[1.0, 0.0], &[2.0, 0.0], -1, 0.5).unwrap();
        assert!(approx(loss, 0.5), "got {loss}");
    }

    #[test]
    fn cosine_emb_dissimilar_pair_beyond_margin() {
        // target=-1, cos_sim=-1.0, margin=0.0 → max(0, -1.0 - 0.0) = 0
        let loss = cosine_embedding_loss(&[1.0, 0.0], &[-1.0, 0.0], -1, 0.0).unwrap();
        assert!(approx(loss, 0.0), "got {loss}");
    }

    #[test]
    fn cosine_emb_invalid_target() {
        assert!(cosine_embedding_loss(&[1.0], &[1.0], 0, 0.0).is_err());
    }

    #[test]
    fn cosine_emb_empty_rejected() {
        assert!(cosine_embedding_loss(&[], &[], 1, 0.0).is_err());
    }

    #[test]
    fn cosine_emb_zero_vector() {
        let loss = cosine_embedding_loss(&[0.0, 0.0], &[1.0, 2.0], 1, 0.0).unwrap();
        // Zero vector → cos_sim=0 → loss=1.0
        assert!(approx(loss, 1.0), "got {loss}");
    }

    // ── Focal Loss ─────────────────────────────────────────────

    #[test]
    fn focal_loss_gamma_zero_is_ce() {
        // When gamma=0, focal loss = -log(p), same as cross-entropy.
        let probs = [0.8, 0.2, 0.5];
        let focal = focal_loss(&probs, 0.0, None, ReductionMode::Sum).unwrap();
        let ce: f32 = probs.iter().map(|&p| -p.clamp(EPS, 1.0 - EPS).ln()).sum();
        assert!(approx(focal, ce), "focal={focal}, ce={ce}");
    }

    #[test]
    fn focal_loss_high_gamma_downweights_easy() {
        // High confidence (easy) example should have lower focal loss
        let easy = [0.95];
        let hard = [0.2];
        let gamma = 2.0;
        let loss_easy = focal_loss(&easy, gamma, None, ReductionMode::Mean).unwrap();
        let loss_hard = focal_loss(&hard, gamma, None, ReductionMode::Mean).unwrap();
        assert!(loss_easy < loss_hard, "easy={loss_easy} should be < hard={loss_hard}");
    }

    #[test]
    fn focal_loss_with_alpha() {
        let probs = [0.8, 0.5];
        let alpha = [0.25, 0.75];
        let loss = focal_loss(&probs, 2.0, Some(&alpha), ReductionMode::Mean).unwrap();
        assert!(loss.is_finite() && loss >= 0.0, "got {loss}");
    }

    #[test]
    fn focal_loss_alpha_length_mismatch() {
        let probs = [0.5, 0.5];
        let alpha = [1.0];
        assert!(focal_loss(&probs, 2.0, Some(&alpha), ReductionMode::Mean).is_err());
    }

    #[test]
    fn focal_loss_negative_gamma_rejected() {
        assert!(focal_loss(&[0.5], -1.0, None, ReductionMode::Mean).is_err());
    }

    #[test]
    fn focal_loss_negative_probability_rejected() {
        assert!(focal_loss(&[-0.1], 2.0, None, ReductionMode::Mean).is_err());
    }

    #[test]
    fn focal_loss_probability_above_one_rejected() {
        assert!(focal_loss(&[1.1], 2.0, None, ReductionMode::Mean).is_err());
    }

    #[test]
    fn focal_loss_empty_rejected() {
        assert!(focal_loss(&[], 2.0, None, ReductionMode::Mean).is_err());
    }

    // ── Perplexity ─────────────────────────────────────────────

    #[test]
    fn perplexity_relationship_to_ce() {
        let logits = [1.0, 2.0, 0.5];
        let (ce, _) = cross_entropy_loss(&logits, &[1], 3, 0.0, ReductionMode::Mean).unwrap();
        let ppl = perplexity(&logits, &[1], 3).unwrap();
        assert!(approx(ppl, ce.exp()), "ppl={ppl}, exp(ce)={}", ce.exp());
    }

    #[test]
    fn perplexity_perfect_prediction() {
        // Very confident correct prediction → perplexity close to 1.
        let logits = [100.0, 0.0, 0.0];
        let ppl = perplexity(&logits, &[0], 3).unwrap();
        assert!(ppl < 1.01, "perfect prediction perplexity should be ~1, got {ppl}");
    }

    #[test]
    fn perplexity_uniform_distribution() {
        // Uniform logits → perplexity = num_classes.
        let logits = [0.0, 0.0, 0.0, 0.0];
        let ppl = perplexity(&logits, &[0], 4).unwrap();
        assert!(approx(ppl, 4.0), "uniform ppl should be 4, got {ppl}");
    }

    #[test]
    fn perplexity_empty_rejected() {
        assert!(perplexity(&[], &[], 3).is_err());
    }

    // ── Error Type ─────────────────────────────────────────────

    #[test]
    fn error_display_dimension_mismatch() {
        let e = LossError::DimensionMismatch { expected: 4, actual: 2 };
        let msg = format!("{e}");
        assert!(msg.contains("4") && msg.contains("2"));
    }

    #[test]
    fn error_display_empty_input() {
        let e = LossError::EmptyInput;
        let msg = format!("{e}");
        assert!(msg.contains("empty"));
    }

    #[test]
    fn error_display_invalid_param() {
        let e = LossError::InvalidParameter { name: "delta", reason: "must be > 0".into() };
        let msg = format!("{e}");
        assert!(msg.contains("delta"));
    }

    #[test]
    fn error_display_invalid_probability() {
        let e = LossError::InvalidProbability { index: 3, value: -0.5 };
        let msg = format!("{e}");
        assert!(msg.contains("3") && msg.contains("-0.5"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(LossError::EmptyInput);
        assert!(!e.to_string().is_empty());
    }
}
