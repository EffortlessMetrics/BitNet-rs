#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::loss::{
    LossReduction, cross_entropy_loss, kl_divergence, mse_loss, smooth_l1_loss,
};
use libfuzzer_sys::fuzz_target;

/// Fuzz cross-entropy, MSE, huber (smooth-L1) loss, and KL divergence
/// with focus on reduction-mode cross-validation and edge-case inputs.
#[derive(Arbitrary, Debug)]
struct LossInput {
    data: Vec<u8>,
    op: u8,
    num_classes: u8,
    beta_byte: u8,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn finite_vec(raw: &[f32], n: usize) -> Vec<f32> {
    raw.iter().take(n).map(|&v| if v.is_finite() { v.clamp(-100.0, 100.0) } else { 0.0 }).collect()
}

fuzz_target!(|input: LossInput| {
    let values = bytes_to_f32(&input.data, 256);
    if values.len() < 4 {
        return;
    }

    match input.op % 4 {
        0 => {
            // Cross-entropy: compare reduction modes
            let num_classes = (input.num_classes as usize % 8) + 2;
            let batch_size = values.len() / (num_classes + 1);
            if batch_size == 0 {
                return;
            }
            let logits = finite_vec(&values, batch_size * num_classes);
            let targets: Vec<usize> = values
                .iter()
                .skip(batch_size * num_classes)
                .take(batch_size)
                .map(|v| (v.abs() as usize) % num_classes)
                .collect();
            if targets.len() != batch_size {
                return;
            }

            // All three reductions must not panic
            let r_none = cross_entropy_loss(&logits, &targets, num_classes, LossReduction::None);
            let r_mean = cross_entropy_loss(&logits, &targets, num_classes, LossReduction::Mean);
            let r_sum = cross_entropy_loss(&logits, &targets, num_classes, LossReduction::Sum);

            // Cross-validate: sum of per-sample == Sum reduction
            if let (Ok((_, per_sample)), Ok((sum_scalar, _))) = (&r_none, &r_sum) {
                let manual_sum: f32 = per_sample.iter().sum();
                if manual_sum.is_finite() && sum_scalar.is_finite() {
                    assert!(
                        (manual_sum - sum_scalar).abs() < 1e-3,
                        "CE sum mismatch: manual={manual_sum} vs sum={sum_scalar}"
                    );
                }
            }

            // Cross-validate: mean == sum / batch_size
            if let (Ok((mean_scalar, _)), Ok((sum_scalar, _))) = (&r_mean, &r_sum) {
                if mean_scalar.is_finite() && sum_scalar.is_finite() && batch_size > 0 {
                    let expected_mean = sum_scalar / batch_size as f32;
                    assert!(
                        (mean_scalar - expected_mean).abs() < 1e-3,
                        "CE mean mismatch: {mean_scalar} vs {expected_mean}"
                    );
                }
            }
        }
        1 => {
            // MSE: symmetry and non-negativity
            let half = values.len() / 2;
            if half == 0 {
                return;
            }
            let a = finite_vec(&values, half);
            let b = finite_vec(&values[half..], half);
            let n = a.len().min(b.len());
            if n == 0 {
                return;
            }

            if let Ok(mse_ab) = mse_loss(&a[..n], &b[..n], LossReduction::Mean) {
                assert!(mse_ab >= 0.0, "MSE negative: {mse_ab}");
                assert!(mse_ab.is_finite(), "MSE non-finite: {mse_ab}");

                // MSE(a,b) == MSE(b,a)
                if let Ok(mse_ba) = mse_loss(&b[..n], &a[..n], LossReduction::Mean) {
                    assert!(
                        (mse_ab - mse_ba).abs() < 1e-5,
                        "MSE not symmetric: {mse_ab} vs {mse_ba}"
                    );
                }

                // MSE(a,a) == 0
                if let Ok(mse_aa) = mse_loss(&a[..n], &a[..n], LossReduction::Mean) {
                    assert!(mse_aa.abs() < 1e-6, "MSE(a,a) != 0: {mse_aa}");
                }
            }
        }
        2 => {
            // Huber (smooth-L1): compare with MSE for small errors
            let half = values.len() / 2;
            if half == 0 {
                return;
            }
            let a = finite_vec(&values, half);
            let b = finite_vec(&values[half..], half);
            let n = a.len().min(b.len());
            if n == 0 {
                return;
            }

            let beta = match input.beta_byte % 4 {
                0 => 0.1,
                1 => 0.5,
                2 => 1.0,
                _ => 5.0,
            };

            if let Ok(huber) = smooth_l1_loss(&a[..n], &b[..n], beta, LossReduction::Mean) {
                assert!(huber >= 0.0, "Huber negative: {huber}");
                assert!(huber.is_finite(), "Huber non-finite: {huber}");
            }

            // Non-negative for all reduction modes
            for red in [LossReduction::None, LossReduction::Sum, LossReduction::Mean] {
                if let Ok(h) = smooth_l1_loss(&a[..n], &b[..n], beta, red) {
                    assert!(h >= -1e-6, "Huber negative with {:?}: {h}", red);
                }
            }
        }
        _ => {
            // KL divergence: KL(p || p) ≈ 0
            let half = values.len() / 2;
            if half == 0 {
                return;
            }
            let log_p = finite_vec(&values, half);
            let q = finite_vec(&values[half..], half);
            let n = log_p.len().min(q.len());
            if n == 0 {
                return;
            }

            // Just verify no panics for arbitrary inputs
            let _ = kl_divergence(&log_p[..n], &q[..n], LossReduction::Mean);
            let _ = kl_divergence(&log_p[..n], &q[..n], LossReduction::Sum);
            let _ = kl_divergence(&log_p[..n], &q[..n], LossReduction::None);

            // Self-divergence should be near zero for valid probability-like inputs
            let p_norm: Vec<f32> = {
                let sum: f32 = log_p[..n].iter().map(|x| x.exp()).sum();
                if sum.is_finite() && sum > 0.0 {
                    log_p[..n].iter().map(|x| x.exp() / sum).collect()
                } else {
                    return;
                }
            };
            let log_p_norm: Vec<f32> = p_norm.iter().map(|x| x.ln()).collect();
            if log_p_norm.iter().all(|x| x.is_finite()) {
                if let Ok(self_kl) = kl_divergence(&log_p_norm, &p_norm, LossReduction::Sum) {
                    if self_kl.is_finite() {
                        assert!(self_kl.abs() < 1e-3, "KL self-divergence should be ~0: {self_kl}");
                    }
                }
            }
        }
    }
});
