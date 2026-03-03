#![allow(clippy::all, clippy::pedantic, clippy::nursery)]
//! Wave 14 property tests: CPU fallback kernel invariants for conv2d,
//! loss functions, and batch normalization.
//!
//! Key invariants tested (20 properties):
//! - Conv2d: output length matches formula, output is finite, identity 1×1 kernel
//!   preserves input, zero-weight kernel produces zero/bias output,
//!   compute_output_size with padding=same reproduces input size,
//!   stride ≥ 2 reduces spatial dims, dilation increases receptive field
//! - Loss: cross_entropy always non-negative, MSE always non-negative,
//!   L1 always non-negative, binary_cross_entropy finite and non-negative,
//!   MSE of identical inputs is zero, L1 symmetric, smooth_l1 ≤ L1,
//!   cosine_similarity_loss in [0,2], contrastive_loss non-negative,
//!   KL divergence of identical distributions is zero
//! - BatchNorm: output length equals input length, inference output is finite,
//!   forward with identity affine has zero mean per channel,
//!   running stats update stays finite

use bitnet_kernels::cpu::batch_norm::{BatchNormConfig, batch_norm_forward, batch_norm_inference};
use bitnet_kernels::cpu::conv2d::{Conv2dConfig, compute_output_size, conv2d};
use bitnet_kernels::cpu::loss::{
    LossReduction, binary_cross_entropy, contrastive_loss, cosine_similarity_loss,
    cross_entropy_loss, kl_divergence, l1_loss, mse_loss, smooth_l1_loss,
};
use proptest::prelude::*;

// ===================================================================
// Strategy helpers
// ===================================================================

fn finite_f32_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0f32, min_len..=max_len)
}

// ===================================================================
// 1. Conv2d properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Conv2d output length matches the expected formula.
    #[test]
    fn prop_conv2d_output_length_matches_formula(
        batch in 1usize..=3,
        in_c in 1usize..=4,
        out_c in 1usize..=4,
        in_h in 3usize..=8,
        in_w in 3usize..=8,
        k in 1usize..=3,
    ) {
        let config = Conv2dConfig::new(in_c, out_c, k);
        let out_h = compute_output_size(in_h, k, 1, 0, 1);
        let out_w = compute_output_size(in_w, k, 1, 0, 1);
        if out_h == 0 || out_w == 0 {
            return Ok(());
        }
        let input = vec![1.0f32; batch * in_c * in_h * in_w];
        let weight = vec![0.1f32; out_c * in_c * k * k];
        let result = conv2d(&input, &weight, None, &config, batch, in_h, in_w);
        prop_assert!(result.is_ok());
        let output = result.unwrap();
        prop_assert_eq!(output.len(), batch * out_c * out_h * out_w);
    }

    /// Conv2d output values are always finite for finite inputs.
    #[test]
    fn prop_conv2d_output_finite(
        in_h in 3usize..=6,
        in_w in 3usize..=6,
    ) {
        let config = Conv2dConfig::new(1, 1, 3);
        let input: Vec<f32> = (0..(in_h * in_w)).map(|i| (i as f32) * 0.01).collect();
        let weight = vec![0.1f32; 9];
        if let Ok(output) = conv2d(&input, &weight, None, &config, 1, in_h, in_w) {
            for &v in &output {
                prop_assert!(v.is_finite(), "output contains non-finite value: {v}");
            }
        }
    }

    /// Identity 1×1 convolution with weight=1 preserves input (single channel).
    #[test]
    fn prop_conv2d_identity_1x1_preserves_input(
        h in 1usize..=8,
        w in 1usize..=8,
    ) {
        let config = Conv2dConfig::new(1, 1, 1);
        let input: Vec<f32> = (0..(h * w)).map(|i| i as f32).collect();
        let weight = vec![1.0f32];
        let output = conv2d(&input, &weight, None, &config, 1, h, w).unwrap();
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            prop_assert!(
                (inp - out).abs() < 1e-5,
                "1x1 identity mismatch at {i}: {inp} != {out}"
            );
        }
    }

    /// Zero-weight kernel with bias produces constant output equal to bias.
    #[test]
    fn prop_conv2d_zero_weight_equals_bias(
        bias_val in -5.0f32..5.0f32,
        h in 3usize..=6,
        w in 3usize..=6,
    ) {
        let config = Conv2dConfig::new(1, 1, 3);
        let input = vec![1.0f32; h * w];
        let weight = vec![0.0f32; 9];
        let bias = vec![bias_val];
        let out_h = compute_output_size(h, 3, 1, 0, 1);
        let out_w = compute_output_size(w, 3, 1, 0, 1);
        if out_h == 0 || out_w == 0 {
            return Ok(());
        }
        let output = conv2d(&input, &weight, Some(&bias), &config, 1, h, w).unwrap();
        for &v in &output {
            prop_assert!(
                (v - bias_val).abs() < 1e-5,
                "expected {bias_val}, got {v}"
            );
        }
    }

    /// compute_output_size is monotonically non-decreasing with padding.
    #[test]
    fn prop_compute_output_size_increases_with_padding(
        in_size in 4usize..=16,
        kernel in 1usize..=3,
        pad_a in 0usize..=2,
        pad_b in 0usize..=4,
    ) {
        let small_pad = pad_a.min(pad_b);
        let large_pad = pad_a.max(pad_b);
        let out_small = compute_output_size(in_size, kernel, 1, small_pad, 1);
        let out_large = compute_output_size(in_size, kernel, 1, large_pad, 1);
        prop_assert!(out_large >= out_small, "more padding should not reduce output size");
    }

    /// Stride ≥ 2 produces output strictly smaller than input (when input is large enough).
    #[test]
    fn prop_conv2d_stride_reduces_output(
        in_size in 6usize..=16,
        stride in 2usize..=4,
    ) {
        let out = compute_output_size(in_size, 1, stride, 0, 1);
        prop_assert!(out < in_size, "stride {stride} should reduce {in_size} -> {out}");
    }

    /// Dilation increases effective kernel size, reducing output.
    #[test]
    fn prop_conv2d_dilation_reduces_output(
        in_size in 8usize..=16,
        kernel in 2usize..=3,
    ) {
        let out_dil1 = compute_output_size(in_size, kernel, 1, 0, 1);
        let out_dil2 = compute_output_size(in_size, kernel, 1, 0, 2);
        prop_assert!(out_dil2 <= out_dil1, "dilation should not increase output size");
    }
}

// ===================================================================
// 2. Loss function properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Cross-entropy loss is always non-negative for valid inputs.
    #[test]
    fn prop_cross_entropy_non_negative(
        batch in 1usize..=8,
        n_classes in 2usize..=10,
    ) {
        let logits: Vec<f32> = (0..(batch * n_classes)).map(|i| ((i as f32) - 5.0) * 0.5).collect();
        let targets: Vec<usize> = (0..batch).map(|i| i % n_classes).collect();
        let (loss, per) = cross_entropy_loss(&logits, &targets, n_classes, LossReduction::Mean).unwrap();
        prop_assert!(loss >= 0.0, "cross_entropy loss must be non-negative, got {loss}");
        for (i, &l) in per.iter().enumerate() {
            prop_assert!(l >= 0.0, "per-sample loss[{i}] must be non-negative, got {l}");
        }
    }

    /// Cross-entropy scalar loss is finite for finite logits.
    #[test]
    fn prop_cross_entropy_finite(
        logits in prop::collection::vec(-50.0f32..50.0f32, 4..=20),
    ) {
        let n_classes = 2.max(logits.len().min(10));
        let batch = logits.len() / n_classes;
        if batch == 0 {
            return Ok(());
        }
        let truncated = &logits[..batch * n_classes];
        let targets: Vec<usize> = (0..batch).map(|i| i % n_classes).collect();
        let (loss, _) = cross_entropy_loss(truncated, &targets, n_classes, LossReduction::Mean).unwrap();
        prop_assert!(loss.is_finite(), "cross_entropy should be finite, got {loss}");
    }

    /// MSE loss is always non-negative.
    #[test]
    fn prop_mse_non_negative(
        preds in finite_f32_vec(1, 32),
    ) {
        let targets: Vec<f32> = preds.iter().map(|&x| x + 0.5).collect();
        let loss = mse_loss(&preds, &targets, LossReduction::Mean).unwrap();
        prop_assert!(loss >= 0.0, "MSE must be non-negative, got {loss}");
    }

    /// MSE of identical inputs is zero.
    #[test]
    fn prop_mse_identical_is_zero(
        data in finite_f32_vec(1, 32),
    ) {
        let loss = mse_loss(&data, &data, LossReduction::Mean).unwrap();
        prop_assert!(loss.abs() < 1e-6, "MSE of identical inputs should be ~0, got {loss}");
    }

    /// L1 loss is always non-negative.
    #[test]
    fn prop_l1_non_negative(
        preds in finite_f32_vec(1, 32),
    ) {
        let targets: Vec<f32> = preds.iter().map(|&x| x + 1.0).collect();
        let loss = l1_loss(&preds, &targets, LossReduction::Mean).unwrap();
        prop_assert!(loss >= 0.0, "L1 must be non-negative, got {loss}");
    }

    /// L1 loss is symmetric: L1(a, b) == L1(b, a).
    #[test]
    fn prop_l1_symmetric(
        a in finite_f32_vec(1, 16),
    ) {
        let b: Vec<f32> = a.iter().map(|&x| x + 0.3).collect();
        let loss_ab = l1_loss(&a, &b, LossReduction::Mean).unwrap();
        let loss_ba = l1_loss(&b, &a, LossReduction::Mean).unwrap();
        prop_assert!(
            (loss_ab - loss_ba).abs() < 1e-5,
            "L1 should be symmetric: {loss_ab} vs {loss_ba}"
        );
    }

    /// Binary cross-entropy is finite and non-negative for valid probabilities.
    #[test]
    fn prop_bce_finite_non_negative(
        n in 1usize..=16,
    ) {
        let preds: Vec<f32> = (0..n).map(|i| 0.1 + 0.8 * (i as f32) / n.max(1) as f32).collect();
        let targets: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let loss = binary_cross_entropy(&preds, &targets, LossReduction::Mean).unwrap();
        prop_assert!(loss.is_finite(), "BCE should be finite, got {loss}");
        prop_assert!(loss >= 0.0, "BCE should be non-negative, got {loss}");
    }

    /// Smooth L1 loss ≤ L1 loss for same inputs (smooth L1 is always ≤ linear).
    #[test]
    fn prop_smooth_l1_leq_l1(
        a in finite_f32_vec(1, 16),
    ) {
        let b: Vec<f32> = a.iter().map(|&x| x + 0.5).collect();
        let l1 = l1_loss(&a, &b, LossReduction::Sum).unwrap();
        let sl1 = smooth_l1_loss(&a, &b, 1.0, LossReduction::Sum).unwrap();
        prop_assert!(sl1 <= l1 + 1e-5, "smooth_l1 ({sl1}) should be ≤ l1 ({l1})");
    }

    /// Cosine similarity loss is in [0, 2].
    #[test]
    fn prop_cosine_loss_in_range(
        a in finite_f32_vec(2, 16),
    ) {
        let b: Vec<f32> = a.iter().map(|&x| x * 0.5 + 1.0).collect();
        let loss = cosine_similarity_loss(&a, &b).unwrap();
        prop_assert!(loss >= -1e-5 && loss <= 2.0 + 1e-5, "cosine loss should be in [0,2], got {loss}");
    }

    /// Contrastive loss is always non-negative.
    #[test]
    fn prop_contrastive_non_negative(
        a in finite_f32_vec(2, 8),
        label in 0.0f32..=1.0f32,
        margin in 0.1f32..5.0f32,
    ) {
        let b: Vec<f32> = a.iter().map(|&x| x + 0.1).collect();
        let loss = contrastive_loss(&a, &b, label, margin).unwrap();
        prop_assert!(loss >= -1e-6, "contrastive loss must be non-negative, got {loss}");
    }

    /// KL divergence of identical distributions is zero.
    #[test]
    fn prop_kl_identical_is_zero(
        n in 2usize..=8,
    ) {
        let p: Vec<f32> = {
            let raw: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
            let sum: f32 = raw.iter().sum();
            raw.iter().map(|x| x / sum).collect()
        };
        let log_p: Vec<f32> = p.iter().map(|x| x.ln()).collect();
        let loss = kl_divergence(&log_p, &p, LossReduction::Sum).unwrap();
        prop_assert!(loss.abs() < 1e-4, "KL of identical distributions should be ~0, got {loss}");
    }

    /// MSE is symmetric: MSE(a, b) == MSE(b, a).
    #[test]
    fn prop_mse_symmetric(
        a in finite_f32_vec(1, 16),
    ) {
        let b: Vec<f32> = a.iter().map(|&x| x + 0.7).collect();
        let loss_ab = mse_loss(&a, &b, LossReduction::Mean).unwrap();
        let loss_ba = mse_loss(&b, &a, LossReduction::Mean).unwrap();
        prop_assert!(
            (loss_ab - loss_ba).abs() < 1e-5,
            "MSE should be symmetric: {loss_ab} vs {loss_ba}"
        );
    }
}

// ===================================================================
// 3. Batch normalization properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Batch norm output length always equals input length.
    #[test]
    fn prop_batch_norm_output_length_equals_input(
        c in 1usize..=8,
        n in 1usize..=8,
    ) {
        let input: Vec<f32> = (0..(n * c)).map(|i| i as f32 * 0.1).collect();
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let rm = vec![0.0f32; c];
        let rv = vec![1.0f32; c];
        let config = BatchNormConfig { num_features: c, eps: 1e-5, momentum: 0.1, training: true };
        let (out, _, _) = batch_norm_forward(&input, &gamma, &beta, &rm, &rv, &config).unwrap();
        prop_assert_eq!(out.len(), input.len());
    }

    /// Batch norm inference output is always finite for finite inputs.
    #[test]
    fn prop_batch_norm_inference_finite(
        c in 1usize..=8,
        n in 1usize..=8,
    ) {
        let input: Vec<f32> = (0..(n * c)).map(|i| (i as f32 - 5.0) * 0.3).collect();
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let rm = vec![0.0f32; c];
        let rv = vec![1.0f32; c];
        let out = batch_norm_inference(&input, &gamma, &beta, &rm, &rv, 1e-5).unwrap();
        for (i, &v) in out.iter().enumerate() {
            prop_assert!(v.is_finite(), "inference output[{i}] is not finite: {v}");
        }
    }

    /// Forward with identity affine (gamma=1, beta=0) has zero mean per channel.
    #[test]
    fn prop_batch_norm_forward_zero_mean(
        c in 1usize..=4,
        n in 2usize..=8,
    ) {
        let input: Vec<f32> = (0..(n * c)).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let rm = vec![0.0f32; c];
        let rv = vec![1.0f32; c];
        let config = BatchNormConfig { num_features: c, eps: 1e-5, momentum: 0.1, training: true };
        let (out, _, _) = batch_norm_forward(&input, &gamma, &beta, &rm, &rv, &config).unwrap();
        for ch in 0..c {
            let mean: f32 = (0..n).map(|i| out[i * c + ch]).sum::<f32>() / n as f32;
            prop_assert!(
                mean.abs() < 1e-4,
                "channel {ch} mean should be ~0, got {mean}"
            );
        }
    }

    /// Running stats update stays finite after forward pass.
    #[test]
    fn prop_batch_norm_running_stats_finite(
        c in 1usize..=8,
        n in 2usize..=8,
    ) {
        let input: Vec<f32> = (0..(n * c)).map(|i| (i as f32) * 0.2).collect();
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let rm = vec![0.0f32; c];
        let rv = vec![1.0f32; c];
        let config = BatchNormConfig { num_features: c, eps: 1e-5, momentum: 0.1, training: true };
        let (_, um, uv) = batch_norm_forward(&input, &gamma, &beta, &rm, &rv, &config).unwrap();
        for (i, &v) in um.iter().chain(uv.iter()).enumerate() {
            prop_assert!(v.is_finite(), "running stat[{i}] is not finite: {v}");
        }
    }

    /// Batch norm inference with identity params (mean=0, var=1, gamma=1, beta=0) is near-identity.
    #[test]
    fn prop_batch_norm_identity_params_near_identity(
        c in 1usize..=4,
        n in 1usize..=4,
    ) {
        let input: Vec<f32> = (0..(n * c)).map(|i| i as f32 * 0.1).collect();
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let rm = vec![0.0f32; c];
        let rv = vec![1.0f32; c];
        let out = batch_norm_inference(&input, &gamma, &beta, &rm, &rv, 1e-5).unwrap();
        for (i, (&inp, &o)) in input.iter().zip(out.iter()).enumerate() {
            prop_assert!(
                (inp - o).abs() < 1e-3,
                "identity BN mismatch at {i}: {inp} vs {o}"
            );
        }
    }

    /// Forward output with scaled gamma matches expected scaling.
    #[test]
    fn prop_batch_norm_gamma_scales_output(
        c in 1usize..=4,
        n in 2usize..=6,
        scale in 0.5f32..3.0f32,
    ) {
        let input: Vec<f32> = (0..(n * c)).map(|i| (i as f32) - (n * c) as f32 / 2.0).collect();
        let gamma_1 = vec![1.0f32; c];
        let gamma_s = vec![scale; c];
        let beta = vec![0.0f32; c];
        let rm = vec![0.0f32; c];
        let rv = vec![1.0f32; c];
        let config = BatchNormConfig { num_features: c, eps: 1e-5, momentum: 0.1, training: true };
        let (out_1, _, _) = batch_norm_forward(&input, &gamma_1, &beta, &rm, &rv, &config).unwrap();
        let (out_s, _, _) = batch_norm_forward(&input, &gamma_s, &beta, &rm, &rv, &config).unwrap();
        for (i, (&a, &b)) in out_1.iter().zip(out_s.iter()).enumerate() {
            prop_assert!(
                (b - a * scale).abs() < 1e-3,
                "gamma scaling mismatch at {i}: {b} vs {a} * {scale}"
            );
        }
    }
}
