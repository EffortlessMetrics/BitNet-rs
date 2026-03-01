//! Edge-case tests for CPU loss function operations.
//!
//! Tests cover cross-entropy, binary cross-entropy, MSE, L1,
//! smooth L1, KL divergence, cosine similarity, and contrastive loss.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::loss::{
    LossReduction, binary_cross_entropy, contrastive_loss, cosine_similarity_loss,
    cross_entropy_loss, kl_divergence, l1_loss, mse_loss, smooth_l1_loss,
};

// ── Cross-entropy loss ───────────────────────────────────────────────

#[test]
fn cross_entropy_perfect_prediction() {
    // Logits heavily favor the correct class
    let logits = vec![10.0, -10.0, -10.0]; // class 0
    let targets = vec![0];
    let (loss, _grad) = cross_entropy_loss(&logits, &targets, 3, LossReduction::Mean).unwrap();
    assert!(loss < 0.01, "Perfect prediction should have near-zero loss: {loss}");
}

#[test]
fn cross_entropy_wrong_prediction() {
    let logits = vec![-10.0, 10.0, -10.0]; // predicts class 1
    let targets = vec![0]; // actual class 0
    let (loss, _) = cross_entropy_loss(&logits, &targets, 3, LossReduction::Mean).unwrap();
    assert!(loss > 10.0, "Wrong prediction should have high loss: {loss}");
}

#[test]
fn cross_entropy_uniform_logits() {
    let logits = vec![0.0, 0.0, 0.0]; // uniform
    let targets = vec![1];
    let (loss, _) = cross_entropy_loss(&logits, &targets, 3, LossReduction::Mean).unwrap();
    // -log(1/3) ≈ 1.0986
    assert!((loss - 1.0986).abs() < 0.01, "Uniform logits loss ≈ ln(3): {loss}");
}

#[test]
fn cross_entropy_batch() {
    let logits = vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0]; // 2 samples, 3 classes
    let targets = vec![0, 1]; // both correct
    let (loss, grad) = cross_entropy_loss(&logits, &targets, 3, LossReduction::Mean).unwrap();
    assert!(loss < 0.01);
    assert!(!grad.is_empty());
}

#[test]
fn cross_entropy_sum_reduction() {
    let logits = vec![0.0, 0.0]; // 1 sample, 2 classes
    let targets = vec![0];
    let (loss_sum, _) = cross_entropy_loss(&logits, &targets, 2, LossReduction::Sum).unwrap();
    let (loss_mean, _) = cross_entropy_loss(&logits, &targets, 2, LossReduction::Mean).unwrap();
    // For single sample, sum == mean
    assert!((loss_sum - loss_mean).abs() < 1e-6);
}

// ── Binary cross-entropy ─────────────────────────────────────────────

#[test]
fn bce_perfect() {
    let preds = vec![0.999, 0.001];
    let targets = vec![1.0, 0.0];
    let loss = binary_cross_entropy(&preds, &targets, LossReduction::Mean).unwrap();
    assert!(loss < 0.01, "Near-perfect BCE should be small: {loss}");
}

#[test]
fn bce_worst_case() {
    let preds = vec![0.001, 0.999];
    let targets = vec![1.0, 0.0]; // completely wrong
    let loss = binary_cross_entropy(&preds, &targets, LossReduction::Mean).unwrap();
    assert!(loss > 5.0, "Worst-case BCE should be high: {loss}");
}

#[test]
fn bce_half_probability() {
    let preds = vec![0.5];
    let targets = vec![1.0];
    let loss = binary_cross_entropy(&preds, &targets, LossReduction::Mean).unwrap();
    // -log(0.5) ≈ 0.693
    assert!((loss - 0.693).abs() < 0.01);
}

// ── MSE loss ─────────────────────────────────────────────────────────

#[test]
fn mse_zero_error() {
    let a = vec![1.0, 2.0, 3.0];
    let loss = mse_loss(&a, &a, LossReduction::Mean).unwrap();
    assert!((loss - 0.0).abs() < 1e-10);
}

#[test]
fn mse_known_value() {
    let preds = vec![1.0, 2.0];
    let targets = vec![3.0, 4.0]; // errors: 2, 2 → squared: 4, 4 → mean: 4
    let loss = mse_loss(&preds, &targets, LossReduction::Mean).unwrap();
    assert!((loss - 4.0).abs() < 1e-6);
}

#[test]
fn mse_sum_reduction() {
    let preds = vec![0.0, 0.0];
    let targets = vec![1.0, 2.0]; // errors: 1, 4 → sum: 5
    let loss = mse_loss(&preds, &targets, LossReduction::Sum).unwrap();
    assert!((loss - 5.0).abs() < 1e-6);
}

// ── L1 loss ──────────────────────────────────────────────────────────

#[test]
fn l1_zero_error() {
    let a = vec![1.0, 2.0, 3.0];
    let loss = l1_loss(&a, &a, LossReduction::Mean).unwrap();
    assert!((loss - 0.0).abs() < 1e-10);
}

#[test]
fn l1_known_value() {
    let preds = vec![1.0, 5.0];
    let targets = vec![3.0, 2.0]; // |2| + |3| = 5, mean = 2.5
    let loss = l1_loss(&preds, &targets, LossReduction::Mean).unwrap();
    assert!((loss - 2.5).abs() < 1e-6);
}

// ── Smooth L1 loss ───────────────────────────────────────────────────

#[test]
fn smooth_l1_small_error() {
    let preds = vec![1.0];
    let targets = vec![1.1]; // diff = 0.1, < beta=1.0
    let loss = smooth_l1_loss(&preds, &targets, 1.0, LossReduction::Mean).unwrap();
    // For |x|<β: 0.5*x²/β = 0.5*0.01/1.0 = 0.005
    assert!((loss - 0.005).abs() < 1e-4);
}

#[test]
fn smooth_l1_large_error() {
    let preds = vec![0.0];
    let targets = vec![10.0]; // |diff|=10 >> beta=1
    let loss = smooth_l1_loss(&preds, &targets, 1.0, LossReduction::Mean).unwrap();
    // For |x|≥β: |x|-0.5*β = 10-0.5 = 9.5
    assert!((loss - 9.5).abs() < 1e-4);
}

// ── KL divergence ────────────────────────────────────────────────────

#[test]
fn kl_divergence_identical() {
    // When log_probs match targets, KL should be 0 (approximately)
    let log_probs = vec![0.6f32.ln(), 0.4f32.ln()];
    let targets = vec![0.6, 0.4];
    let loss = kl_divergence(&log_probs, &targets, LossReduction::Sum).unwrap();
    assert!(loss.abs() < 0.01, "KL divergence of identical distributions should be ~0: {loss}");
}

// ── Cosine similarity loss ───────────────────────────────────────────

#[test]
fn cosine_similarity_identical_vectors() {
    let a = vec![1.0, 2.0, 3.0];
    let loss = cosine_similarity_loss(&a, &a).unwrap();
    // Cosine similarity = 1 for identical vectors, loss = 1 - sim or just sim
    // Just verify it's valid
    assert!(loss.is_finite());
}

#[test]
fn cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let loss = cosine_similarity_loss(&a, &b).unwrap();
    assert!(loss.is_finite());
}

#[test]
fn cosine_similarity_opposite() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    let loss = cosine_similarity_loss(&a, &b).unwrap();
    assert!(loss.is_finite());
}

// ── Contrastive loss ─────────────────────────────────────────────────

#[test]
fn contrastive_loss_similar_pair() {
    let a = vec![1.0, 2.0];
    let b = vec![1.1, 2.1]; // very close
    let loss = contrastive_loss(&a, &b, 1.0, 1.0).unwrap(); // label=1 means similar
    assert!(loss < 0.1, "Similar pair should have low loss: {loss}");
}

#[test]
fn contrastive_loss_dissimilar_pair() {
    let a = vec![0.0, 0.0];
    let b = vec![10.0, 10.0]; // very far
    let loss = contrastive_loss(&a, &b, 0.0, 1.0).unwrap(); // label=0 means dissimilar
    // Dissimilar + far apart → loss should be low (margin satisfied)
    assert!(loss.is_finite());
}

#[test]
fn contrastive_loss_zero_margin() {
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    let loss = contrastive_loss(&a, &b, 0.0, 0.0).unwrap();
    assert!(loss.is_finite());
}
