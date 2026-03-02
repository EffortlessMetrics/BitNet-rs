#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct LayerNormInput {
    batch_size: u8,
    hidden_dim: u16,
    eps_byte: u8,
    use_beta: bool,
    use_rms: bool,
    input_data: Vec<u8>,
    gamma_data: Vec<u8>,
    beta_data: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: LayerNormInput| {
    let batch_size = (input.batch_size as usize % 8) + 1;
    let hidden_dim = (input.hidden_dim as usize % 512) + 1;
    let total = batch_size * hidden_dim;

    // Map eps to a small positive value in [1e-12, 1e-1].
    let eps = 10.0f32.powf(-12.0 + (input.eps_byte as f32 / 255.0) * 11.0);

    let config =
        LayerNormConfig { normalized_shape: vec![hidden_dim], eps, elementwise_affine: true };

    let raw_input = bytes_to_f32(&input.input_data, total);
    if raw_input.len() < total {
        return;
    }
    let input_data: Vec<f32> = raw_input[..total]
        .iter()
        .map(|&v| if v.is_finite() { v.clamp(-1e6, 1e6) } else { 0.0 })
        .collect();

    let raw_gamma = bytes_to_f32(&input.gamma_data, hidden_dim);
    if raw_gamma.len() < hidden_dim {
        return;
    }
    let gamma: Vec<f32> = raw_gamma[..hidden_dim]
        .iter()
        .map(|&v| if v.is_finite() { v.clamp(-1e3, 1e3) } else { 1.0 })
        .collect();

    if input.use_rms {
        // --- RMS norm ---
        if let Ok(out) = rms_norm(&input_data, &gamma, &config) {
            assert_eq!(out.len(), total, "rms_norm output length mismatch");
            for (i, &val) in out.iter().enumerate() {
                assert!(val.is_finite(), "rms_norm output non-finite at {i}: {val}");
            }
        }
    } else {
        // --- Layer norm ---
        let beta_opt = if input.use_beta {
            let raw_beta = bytes_to_f32(&input.beta_data, hidden_dim);
            if raw_beta.len() < hidden_dim {
                return;
            }
            let beta: Vec<f32> = raw_beta[..hidden_dim]
                .iter()
                .map(|&v| if v.is_finite() { v.clamp(-1e3, 1e3) } else { 0.0 })
                .collect();
            Some(beta)
        } else {
            None
        };

        if let Ok(out) = layer_norm(&input_data, &gamma, beta_opt.as_deref(), &config) {
            assert_eq!(out.len(), total, "layer_norm output length mismatch");
            for (i, &val) in out.iter().enumerate() {
                assert!(val.is_finite(), "layer_norm output non-finite at {i}: {val}");
            }

            // Invariant: if gamma is all-ones and beta is all-zeros, output should
            // be approximately zero-mean per sample (within floating-point tolerance).
            let all_ones = gamma.iter().all(|&g| (g - 1.0).abs() < 1e-6);
            let all_zero_beta =
                beta_opt.as_ref().map(|b| b.iter().all(|&v| v.abs() < 1e-6)).unwrap_or(true);
            if all_ones && all_zero_beta && !input.use_beta {
                for b in 0..batch_size {
                    let start = b * hidden_dim;
                    let end = start + hidden_dim;
                    let mean: f32 = out[start..end].iter().sum::<f32>() / hidden_dim as f32;
                    assert!(
                        mean.abs() < 0.1,
                        "layer_norm output mean {mean} for batch {b} (expected ~0)"
                    );
                }
            }
        }
    }
});
