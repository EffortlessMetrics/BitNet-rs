#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::loss::{
    LossReduction, binary_cross_entropy, contrastive_loss, cosine_similarity_loss,
    cross_entropy_loss, kl_divergence, l1_loss, mse_loss, smooth_l1_loss,
};
use libfuzzer_sys::fuzz_target;

/// Fuzz all loss functions in `bitnet_kernels::cpu::loss` with
/// arbitrary predictions, targets, and configuration.
#[derive(Arbitrary, Debug)]
struct LossFuzzInput {
    /// Predictions / first vector.
    predictions: Vec<f32>,
    /// Targets / second vector.
    targets: Vec<f32>,
    /// Integer targets for cross-entropy.
    class_targets: Vec<u8>,
    /// Which loss function to fuzz (mod 8).
    op: u8,
    /// Reduction mode selector (mod 3).
    reduction: u8,
    /// Smooth-L1 beta selector.
    beta_selector: u8,
    /// Contrastive loss label (0 or 1).
    contrastive_label: bool,
    /// Contrastive margin selector.
    margin_selector: u8,
    /// Number of classes for cross-entropy.
    num_classes: u8,
}

fn select_reduction(selector: u8) -> LossReduction {
    match selector % 3 {
        0 => LossReduction::Mean,
        1 => LossReduction::Sum,
        _ => LossReduction::None,
    }
}

fuzz_target!(|input: LossFuzzInput| {
    let reduction = select_reduction(input.reduction);

    // Cap vector lengths to avoid slow iterations.
    let preds: Vec<f32> = input.predictions.iter().copied().take(256).collect();
    let tgts: Vec<f32> = input.targets.iter().copied().take(256).collect();

    match input.op % 8 {
        0 => {
            // --- cross_entropy_loss ---
            let num_classes = (input.num_classes as usize % 16) + 2;
            let class_targets: Vec<usize> =
                input.class_targets.iter().take(64).map(|&t| t as usize % num_classes).collect();
            if class_targets.is_empty() {
                return;
            }
            let batch_size = class_targets.len();
            let needed = batch_size * num_classes;

            // Build logits from predictions, pad if needed.
            let logits: Vec<f32> = if preds.len() >= needed {
                preds[..needed].to_vec()
            } else {
                let mut v = preds.clone();
                v.resize(needed, 0.0);
                v
            };

            // Skip non-finite logits.
            if logits.iter().any(|x| !x.is_finite()) {
                let _ = cross_entropy_loss(&logits, &class_targets, num_classes, reduction);
                return;
            }

            match cross_entropy_loss(&logits, &class_targets, num_classes, reduction) {
                Ok((scalar, per_sample)) => {
                    assert_eq!(per_sample.len(), batch_size);
                    // Cross-entropy per-sample losses should be non-negative.
                    for (i, &l) in per_sample.iter().enumerate() {
                        assert!(l >= -1e-5, "CE per-sample[{i}] negative: {l}");
                        assert!(l.is_finite(), "CE per-sample[{i}] non-finite: {l}");
                    }
                    assert!(scalar.is_finite(), "CE scalar non-finite: {scalar}");
                }
                Err(_) => {}
            }
        }
        1 => {
            // --- mse_loss ---
            let len = preds.len().min(tgts.len());
            if len == 0 {
                assert!(mse_loss(&[], &[], reduction).is_err());
                return;
            }
            let p = &preds[..len];
            let t = &tgts[..len];
            if p.iter().chain(t.iter()).any(|x| !x.is_finite()) {
                let _ = mse_loss(p, t, reduction);
                return;
            }
            match mse_loss(p, t, reduction) {
                Ok(loss) => {
                    assert!(loss >= 0.0, "MSE negative: {loss}");
                    assert!(loss.is_finite(), "MSE non-finite: {loss}");
                }
                Err(_) => {}
            }
        }
        2 => {
            // --- l1_loss (MAE) ---
            let len = preds.len().min(tgts.len());
            if len == 0 {
                assert!(l1_loss(&[], &[], reduction).is_err());
                return;
            }
            let p = &preds[..len];
            let t = &tgts[..len];
            if p.iter().chain(t.iter()).any(|x| !x.is_finite()) {
                let _ = l1_loss(p, t, reduction);
                return;
            }
            match l1_loss(p, t, reduction) {
                Ok(loss) => {
                    assert!(loss >= 0.0, "L1 negative: {loss}");
                    assert!(loss.is_finite(), "L1 non-finite: {loss}");
                }
                Err(_) => {}
            }
        }
        3 => {
            // --- binary_cross_entropy ---
            let len = preds.len().min(tgts.len());
            if len == 0 {
                assert!(binary_cross_entropy(&[], &[], reduction).is_err());
                return;
            }
            let p = &preds[..len];
            let t = &tgts[..len];
            if p.iter().chain(t.iter()).any(|x| !x.is_finite()) {
                let _ = binary_cross_entropy(p, t, reduction);
                return;
            }
            match binary_cross_entropy(p, t, reduction) {
                Ok(loss) => {
                    assert!(loss.is_finite(), "BCE non-finite: {loss}");
                }
                Err(_) => {}
            }
        }
        4 => {
            // --- smooth_l1_loss ---
            let len = preds.len().min(tgts.len());
            if len == 0 {
                return;
            }
            let p = &preds[..len];
            let t = &tgts[..len];
            let beta = match input.beta_selector % 3 {
                0 => 0.5,
                1 => 1.0,
                _ => 2.0,
            };
            if p.iter().chain(t.iter()).any(|x| !x.is_finite()) {
                let _ = smooth_l1_loss(p, t, beta, reduction);
                return;
            }
            match smooth_l1_loss(p, t, beta, reduction) {
                Ok(loss) => {
                    assert!(loss >= 0.0, "Smooth L1 negative: {loss}");
                    assert!(loss.is_finite(), "Smooth L1 non-finite: {loss}");
                }
                Err(_) => {}
            }
        }
        5 => {
            // --- kl_divergence ---
            let len = preds.len().min(tgts.len());
            if len == 0 {
                assert!(kl_divergence(&[], &[], reduction).is_err());
                return;
            }
            let log_probs = &preds[..len];
            let t = &tgts[..len];
            if log_probs.iter().chain(t.iter()).any(|x| !x.is_finite()) {
                let _ = kl_divergence(log_probs, t, reduction);
                return;
            }
            // KL divergence can return any finite value — just check no panic.
            let _ = kl_divergence(log_probs, t, reduction);
        }
        6 => {
            // --- cosine_similarity_loss ---
            let len = preds.len().min(tgts.len());
            if len == 0 {
                assert!(cosine_similarity_loss(&[], &[]).is_err());
                return;
            }
            let a = &preds[..len];
            let b = &tgts[..len];
            if a.iter().chain(b.iter()).any(|x| !x.is_finite()) {
                let _ = cosine_similarity_loss(a, b);
                return;
            }
            match cosine_similarity_loss(a, b) {
                Ok(loss) => {
                    assert!(loss.is_finite(), "cosine loss non-finite: {loss}");
                    assert!(
                        loss >= -1e-5 && loss <= 2.0 + 1e-5,
                        "cosine loss out of [0,2]: {loss}"
                    );
                }
                Err(_) => {}
            }
        }
        _ => {
            // --- contrastive_loss ---
            let len = preds.len().min(tgts.len());
            if len == 0 {
                assert!(contrastive_loss(&[], &[], 1.0, 1.0).is_err());
                return;
            }
            let a = &preds[..len];
            let b = &tgts[..len];
            let label = if input.contrastive_label { 1.0 } else { 0.0 };
            let margin = match input.margin_selector % 3 {
                0 => 0.5,
                1 => 1.0,
                _ => 5.0,
            };
            if a.iter().chain(b.iter()).any(|x| !x.is_finite()) {
                let _ = contrastive_loss(a, b, label, margin);
                return;
            }
            match contrastive_loss(a, b, label, margin) {
                Ok(loss) => {
                    assert!(loss >= 0.0, "contrastive loss negative: {loss}");
                    assert!(loss.is_finite(), "contrastive loss non-finite: {loss}");
                }
                Err(_) => {}
            }
        }
    }
});
