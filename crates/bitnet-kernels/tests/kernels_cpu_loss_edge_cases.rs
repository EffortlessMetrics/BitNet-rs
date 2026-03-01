//! Edge-case integration tests for `bitnet_kernels::cpu::loss` module.
//!
//! Covers:
//! - cross_entropy_loss: batch, reductions, large logits, uniform distribution
//! - binary_cross_entropy: perfect/worst predictions, boundary clamping
//! - mse_loss: zero error, symmetric, large values
//! - l1_loss: zero error, symmetric, single element
//! - smooth_l1_loss: quadratic/linear regimes, beta edge cases
//! - kl_divergence: identical distributions, zero targets, asymmetry
//! - cosine_similarity_loss: identical, orthogonal, opposite, zero vectors
//! - contrastive_loss: positive/negative pairs, margin effects
//! - LossReduction: None vs Mean vs Sum consistency
//! - Error paths: empty inputs, length mismatches, invalid arguments

use bitnet_kernels::cpu::loss::{
    LossReduction, binary_cross_entropy, contrastive_loss, cosine_similarity_loss,
    cross_entropy_loss, kl_divergence, l1_loss, mse_loss, smooth_l1_loss,
};

const TOL: f32 = 1e-4;

// =========================================================================
// cross_entropy_loss
// =========================================================================

#[test]
fn cross_entropy_perfect_prediction() {
    // Very high logit for correct class → loss should be near 0
    let logits = vec![100.0, -100.0, -100.0];
    let (loss, per) = cross_entropy_loss(&logits, &[0], 3, LossReduction::Mean).unwrap();
    assert!(loss < 1e-5, "perfect prediction loss should be ~0, got {loss}");
    assert_eq!(per.len(), 1);
}

#[test]
fn cross_entropy_worst_prediction() {
    // Very low logit for correct class → high loss
    let logits = vec![-100.0, 100.0, 100.0];
    let (loss, _per) = cross_entropy_loss(&logits, &[0], 3, LossReduction::Mean).unwrap();
    assert!(loss > 50.0, "worst prediction should have high loss, got {loss}");
}

#[test]
fn cross_entropy_uniform_logits() {
    // Uniform logits → loss = ln(num_classes)
    let num_classes = 4;
    let logits = vec![0.0; num_classes];
    let (loss, _) = cross_entropy_loss(&logits, &[0], num_classes, LossReduction::Mean).unwrap();
    let expected = (num_classes as f32).ln();
    assert!(
        (loss - expected).abs() < TOL,
        "uniform logits loss should be ln(C), got {loss}, expected {expected}"
    );
}

#[test]
fn cross_entropy_multi_batch() {
    let logits = vec![
        1.0, 0.0, 0.0, // sample 0
        0.0, 1.0, 0.0, // sample 1
        0.0, 0.0, 1.0, // sample 2
    ];
    let (loss_mean, per) = cross_entropy_loss(&logits, &[0, 1, 2], 3, LossReduction::Mean).unwrap();
    assert_eq!(per.len(), 3);
    // All three have the same structure → same per-sample loss
    assert!((per[0] - per[1]).abs() < TOL);
    assert!((per[1] - per[2]).abs() < TOL);
    assert!((loss_mean - per[0]).abs() < TOL);
}

#[test]
fn cross_entropy_reduction_none_vs_sum() {
    let logits = vec![1.0, 0.0, 0.0, 1.0];
    let (loss_none, _) = cross_entropy_loss(&logits, &[0, 1], 2, LossReduction::None).unwrap();
    let (loss_sum, _) = cross_entropy_loss(&logits, &[0, 1], 2, LossReduction::Sum).unwrap();
    // LossReduction::None and Sum both compute sum
    assert!((loss_none - loss_sum).abs() < TOL);
}

#[test]
fn cross_entropy_large_logits_stable() {
    // Very large logits should not produce NaN/Inf due to log-sum-exp trick
    let logits = vec![1000.0, 999.0, 998.0];
    let (loss, _) = cross_entropy_loss(&logits, &[0], 3, LossReduction::Mean).unwrap();
    assert!(loss.is_finite(), "large logits should be stable, got {loss}");
    assert!(loss >= 0.0, "cross-entropy should be non-negative");
}

#[test]
fn cross_entropy_many_classes() {
    let num_classes = 1000;
    let logits: Vec<f32> = (0..num_classes).map(|i| i as f32 * 0.01).collect();
    let (loss, _) = cross_entropy_loss(&logits, &[500], num_classes, LossReduction::Mean).unwrap();
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

// cross_entropy error paths

#[test]
fn cross_entropy_empty_targets_error() {
    assert!(cross_entropy_loss(&[], &[], 3, LossReduction::Mean).is_err());
}

#[test]
fn cross_entropy_zero_classes_error() {
    assert!(cross_entropy_loss(&[], &[0], 0, LossReduction::Mean).is_err());
}

#[test]
fn cross_entropy_logits_length_mismatch_error() {
    assert!(cross_entropy_loss(&[1.0, 2.0], &[0], 3, LossReduction::Mean).is_err());
}

#[test]
fn cross_entropy_target_out_of_range_error() {
    assert!(cross_entropy_loss(&[1.0, 2.0, 3.0], &[3], 3, LossReduction::Mean).is_err());
}

// =========================================================================
// binary_cross_entropy
// =========================================================================

#[test]
fn bce_perfect_prediction_low_loss() {
    // pred ≈ target → low loss
    let preds = vec![0.99, 0.01, 0.99];
    let targets = vec![1.0, 0.0, 1.0];
    let loss = binary_cross_entropy(&preds, &targets, LossReduction::Mean).unwrap();
    assert!(loss < 0.05, "near-perfect BCE should be low, got {loss}");
}

#[test]
fn bce_worst_prediction_high_loss() {
    let preds = vec![0.01, 0.99];
    let targets = vec![1.0, 0.0];
    let loss = binary_cross_entropy(&preds, &targets, LossReduction::Mean).unwrap();
    assert!(loss > 2.0, "worst BCE should be high, got {loss}");
}

#[test]
fn bce_boundary_clamping() {
    // Predictions at exactly 0.0 and 1.0 should be clamped, not produce NaN
    let preds = vec![0.0, 1.0];
    let targets = vec![0.0, 1.0];
    let loss = binary_cross_entropy(&preds, &targets, LossReduction::Mean).unwrap();
    assert!(loss.is_finite(), "boundary values should be clamped, got {loss}");
}

#[test]
fn bce_all_reductions_consistent() {
    let preds = vec![0.7, 0.3, 0.9];
    let targets = vec![1.0, 0.0, 1.0];
    let loss_sum = binary_cross_entropy(&preds, &targets, LossReduction::Sum).unwrap();
    let loss_mean = binary_cross_entropy(&preds, &targets, LossReduction::Mean).unwrap();
    assert!((loss_mean - loss_sum / 3.0).abs() < TOL);
}

#[test]
fn bce_empty_input_error() {
    assert!(binary_cross_entropy(&[], &[], LossReduction::Mean).is_err());
}

#[test]
fn bce_length_mismatch_error() {
    assert!(binary_cross_entropy(&[0.5], &[0.0, 1.0], LossReduction::Mean).is_err());
}

// =========================================================================
// mse_loss
// =========================================================================

#[test]
fn mse_zero_error() {
    let v = vec![1.0, 2.0, 3.0];
    let loss = mse_loss(&v, &v, LossReduction::Mean).unwrap();
    assert!(loss.abs() < TOL, "identical inputs should give 0 MSE, got {loss}");
}

#[test]
fn mse_known_value() {
    // (1-3)² + (2-4)² = 4 + 4 = 8, mean = 4
    let preds = vec![1.0, 2.0];
    let targets = vec![3.0, 4.0];
    let loss = mse_loss(&preds, &targets, LossReduction::Mean).unwrap();
    assert!((loss - 4.0).abs() < TOL, "expected 4.0, got {loss}");
}

#[test]
fn mse_symmetric() {
    let a = vec![1.0, 5.0, 9.0];
    let b = vec![3.0, 7.0, 2.0];
    let loss_ab = mse_loss(&a, &b, LossReduction::Mean).unwrap();
    let loss_ba = mse_loss(&b, &a, LossReduction::Mean).unwrap();
    assert!((loss_ab - loss_ba).abs() < TOL, "MSE should be symmetric");
}

#[test]
fn mse_single_element() {
    let loss = mse_loss(&[5.0], &[3.0], LossReduction::Mean).unwrap();
    assert!((loss - 4.0).abs() < TOL);
}

#[test]
fn mse_large_values() {
    let preds = vec![1e6, -1e6];
    let targets = vec![1e6 + 1.0, -1e6 + 1.0];
    let loss = mse_loss(&preds, &targets, LossReduction::Mean).unwrap();
    assert!(loss.is_finite());
    assert!((loss - 1.0).abs() < TOL);
}

#[test]
fn mse_empty_error() {
    assert!(mse_loss(&[], &[], LossReduction::Mean).is_err());
}

#[test]
fn mse_length_mismatch_error() {
    assert!(mse_loss(&[1.0], &[2.0, 3.0], LossReduction::Mean).is_err());
}

// =========================================================================
// l1_loss
// =========================================================================

#[test]
fn l1_zero_error() {
    let v = vec![1.0, 2.0, 3.0];
    let loss = l1_loss(&v, &v, LossReduction::Mean).unwrap();
    assert!(loss.abs() < TOL);
}

#[test]
fn l1_known_value() {
    // |1-3| + |2-4| = 2 + 2 = 4, mean = 2
    let loss = l1_loss(&[1.0, 2.0], &[3.0, 4.0], LossReduction::Mean).unwrap();
    assert!((loss - 2.0).abs() < TOL);
}

#[test]
fn l1_symmetric() {
    let a = vec![1.0, 5.0];
    let b = vec![3.0, 7.0];
    let ab = l1_loss(&a, &b, LossReduction::Sum).unwrap();
    let ba = l1_loss(&b, &a, LossReduction::Sum).unwrap();
    assert!((ab - ba).abs() < TOL);
}

#[test]
fn l1_always_non_negative() {
    let a = vec![-10.0, 5.0, 0.0];
    let b = vec![10.0, -5.0, 0.0];
    let loss = l1_loss(&a, &b, LossReduction::Mean).unwrap();
    assert!(loss >= 0.0);
}

#[test]
fn l1_empty_error() {
    assert!(l1_loss(&[], &[], LossReduction::Mean).is_err());
}

// =========================================================================
// smooth_l1_loss
// =========================================================================

#[test]
fn smooth_l1_zero_error() {
    let v = vec![1.0, 2.0, 3.0];
    let loss = smooth_l1_loss(&v, &v, 1.0, LossReduction::Mean).unwrap();
    assert!(loss.abs() < TOL);
}

#[test]
fn smooth_l1_quadratic_regime() {
    // |d| < beta → loss = 0.5 * d² / beta
    // d = 0.5, beta = 1.0 → loss = 0.5 * 0.25 / 1.0 = 0.125
    let loss = smooth_l1_loss(&[0.0], &[0.5], 1.0, LossReduction::Sum).unwrap();
    assert!((loss - 0.125).abs() < TOL, "quadratic regime: got {loss}");
}

#[test]
fn smooth_l1_linear_regime() {
    // |d| >= beta → loss = |d| - 0.5 * beta
    // d = 2.0, beta = 1.0 → loss = 2.0 - 0.5 = 1.5
    let loss = smooth_l1_loss(&[0.0], &[2.0], 1.0, LossReduction::Sum).unwrap();
    assert!((loss - 1.5).abs() < TOL, "linear regime: got {loss}");
}

#[test]
fn smooth_l1_at_beta_boundary() {
    // d = beta exactly → quadratic: 0.5 * beta² / beta = 0.5 * beta
    // d = 1.0, beta = 1.0 → 0.5
    let loss = smooth_l1_loss(&[0.0], &[1.0], 1.0, LossReduction::Sum).unwrap();
    assert!((loss - 0.5).abs() < TOL, "at boundary: got {loss}");
}

#[test]
fn smooth_l1_small_beta() {
    let loss = smooth_l1_loss(&[0.0], &[1.0], 0.01, LossReduction::Sum).unwrap();
    assert!(loss.is_finite());
    // d=1.0 >> beta=0.01 → linear: 1.0 - 0.005 = 0.995
    assert!((loss - 0.995).abs() < TOL);
}

#[test]
fn smooth_l1_zero_beta_error() {
    assert!(smooth_l1_loss(&[0.0], &[1.0], 0.0, LossReduction::Sum).is_err());
}

#[test]
fn smooth_l1_negative_beta_error() {
    assert!(smooth_l1_loss(&[0.0], &[1.0], -1.0, LossReduction::Sum).is_err());
}

// =========================================================================
// kl_divergence
// =========================================================================

#[test]
fn kl_identical_distributions_zero() {
    // KL(p || p) = 0
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let log_p: Vec<f32> = p.iter().map(|x| (*x as f32).ln()).collect();
    let loss = kl_divergence(&log_p, &p, LossReduction::Sum).unwrap();
    assert!(loss.abs() < TOL, "KL(p||p) should be 0, got {loss}");
}

#[test]
fn kl_zero_targets_contribute_nothing() {
    // When target=0, that term contributes 0 to the sum
    let log_probs = vec![-1.0, -2.0, -3.0];
    let targets = vec![0.0, 0.0, 0.0];
    let loss = kl_divergence(&log_probs, &targets, LossReduction::Sum).unwrap();
    assert!(loss.abs() < TOL, "all-zero targets should give 0, got {loss}");
}

#[test]
fn kl_non_negative() {
    // KL divergence should always be >= 0 (Gibbs inequality)
    let log_probs = vec![(-0.3_f32).ln(), (-0.7_f32).ln()]; // invalid but test the math
    let targets = vec![0.5, 0.5];
    let _loss = kl_divergence(&log_probs, &targets, LossReduction::Sum).unwrap();
    // With valid probability inputs, KL >= 0
    // log_probs must be actual log probabilities for this to hold
    let log_probs = vec![0.3_f32.ln(), 0.7_f32.ln()];
    let loss = kl_divergence(&log_probs, &targets, LossReduction::Sum).unwrap();
    assert!(loss >= -TOL, "KL divergence should be >= 0, got {loss}");
}

#[test]
fn kl_empty_error() {
    assert!(kl_divergence(&[], &[], LossReduction::Mean).is_err());
}

// =========================================================================
// cosine_similarity_loss
// =========================================================================

#[test]
fn cosine_identical_vectors_zero_loss() {
    let a = vec![1.0, 2.0, 3.0];
    let loss = cosine_similarity_loss(&a, &a).unwrap();
    assert!(loss.abs() < TOL, "identical vectors → loss=0, got {loss}");
}

#[test]
fn cosine_opposite_vectors_loss_two() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![-1.0, 0.0, 0.0];
    let loss = cosine_similarity_loss(&a, &b).unwrap();
    assert!((loss - 2.0).abs() < TOL, "opposite vectors → loss=2, got {loss}");
}

#[test]
fn cosine_orthogonal_vectors_loss_one() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let loss = cosine_similarity_loss(&a, &b).unwrap();
    assert!((loss - 1.0).abs() < TOL, "orthogonal vectors → loss=1, got {loss}");
}

#[test]
fn cosine_zero_vector_returns_one() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 2.0, 3.0];
    let loss = cosine_similarity_loss(&a, &b).unwrap();
    assert!((loss - 1.0).abs() < TOL, "zero vector → loss=1, got {loss}");
}

#[test]
fn cosine_scaled_vectors_same_loss() {
    // cos similarity is scale-invariant
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![2.0, 4.0, 6.0]; // 2*a
    let loss = cosine_similarity_loss(&a, &b).unwrap();
    assert!(loss.abs() < TOL, "scaled vectors → loss=0, got {loss}");
}

#[test]
fn cosine_empty_error() {
    assert!(cosine_similarity_loss(&[], &[]).is_err());
}

#[test]
fn cosine_length_mismatch_error() {
    assert!(cosine_similarity_loss(&[1.0], &[1.0, 2.0]).is_err());
}

// =========================================================================
// contrastive_loss
// =========================================================================

#[test]
fn contrastive_positive_pair_identical() {
    // Identical vectors, positive pair → dist=0 → loss=0
    let a = vec![1.0, 2.0, 3.0];
    let loss = contrastive_loss(&a, &a, 1.0, 1.0).unwrap();
    assert!(loss.abs() < TOL, "identical positive pair → loss=0, got {loss}");
}

#[test]
fn contrastive_positive_pair_distant() {
    // Positive pair, far apart → high loss
    let a = vec![0.0, 0.0];
    let b = vec![10.0, 0.0];
    let loss = contrastive_loss(&a, &b, 1.0, 1.0).unwrap();
    // loss = 0.5 * label * dist² = 0.5 * 1.0 * 100 = 50
    assert!((loss - 50.0).abs() < TOL, "positive far pair: got {loss}");
}

#[test]
fn contrastive_negative_pair_close() {
    // Negative pair, within margin → loss > 0
    let a = vec![0.0, 0.0];
    let b = vec![0.5, 0.0]; // dist = 0.5
    let margin = 2.0;
    let loss = contrastive_loss(&a, &b, 0.0, margin).unwrap();
    // loss = 0.5 * (1-0) * (margin - dist)² = 0.5 * (2.0 - 0.5)² = 0.5 * 2.25 = 1.125
    assert!((loss - 1.125).abs() < TOL, "negative close pair: got {loss}");
}

#[test]
fn contrastive_negative_pair_beyond_margin() {
    // Negative pair, beyond margin → loss = 0
    let a = vec![0.0, 0.0];
    let b = vec![5.0, 0.0]; // dist = 5
    let margin = 2.0;
    let loss = contrastive_loss(&a, &b, 0.0, margin).unwrap();
    assert!(loss.abs() < TOL, "negative pair beyond margin → loss=0, got {loss}");
}

#[test]
fn contrastive_empty_error() {
    assert!(contrastive_loss(&[], &[], 1.0, 1.0).is_err());
}

// =========================================================================
// LossReduction consistency across functions
// =========================================================================

#[test]
fn reduction_mean_is_sum_div_n() {
    let preds = vec![1.0, 3.0, 5.0];
    let targets = vec![2.0, 4.0, 6.0];
    let sum = mse_loss(&preds, &targets, LossReduction::Sum).unwrap();
    let mean = mse_loss(&preds, &targets, LossReduction::Mean).unwrap();
    assert!((mean - sum / 3.0).abs() < TOL, "mean should be sum/n: {mean} vs {sum}/3");
}

#[test]
fn reduction_sum_equals_none_for_mse() {
    let preds = vec![1.0, 3.0];
    let targets = vec![2.0, 4.0];
    let sum = mse_loss(&preds, &targets, LossReduction::Sum).unwrap();
    let none = mse_loss(&preds, &targets, LossReduction::None).unwrap();
    assert!((sum - none).abs() < TOL, "Sum and None should be identical");
}

// =========================================================================
// Numerical stability
// =========================================================================

#[test]
fn bce_extreme_probabilities_stable() {
    let preds = vec![1e-10, 1.0 - 1e-10, 0.5];
    let targets = vec![0.0, 1.0, 0.5];
    let loss = binary_cross_entropy(&preds, &targets, LossReduction::Mean).unwrap();
    assert!(loss.is_finite(), "extreme BCE should be stable, got {loss}");
}

#[test]
fn cross_entropy_negative_logits_stable() {
    let logits = vec![-100.0, -99.0, -101.0];
    let (loss, _) = cross_entropy_loss(&logits, &[1], 3, LossReduction::Mean).unwrap();
    assert!(loss.is_finite(), "negative logits should be stable, got {loss}");
}
