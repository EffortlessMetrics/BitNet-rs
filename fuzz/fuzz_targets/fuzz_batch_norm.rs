#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::batch_norm::{BatchNormConfig, batch_norm_forward, batch_norm_inference};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct BatchNormFuzzInput {
    channels: u8,
    batch_size: u8,
    data: Vec<u8>,
    gamma_data: Vec<u8>,
    beta_data: Vec<u8>,
    mean_data: Vec<u8>,
    var_data: Vec<u8>,
    eps: u8,
    momentum: u8,
    training: bool,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: BatchNormFuzzInput| {
    let channels = (input.channels as usize % 16) + 1;
    let batch_size = (input.batch_size as usize % 8) + 1;
    let total = batch_size * channels;
    let eps = 1e-5 * (1.0 + input.eps as f32);
    let momentum = (input.momentum as f32 % 100.0) / 100.0;

    let data = bytes_to_f32(&input.data, total);
    let gamma = bytes_to_f32(&input.gamma_data, channels);
    let beta = bytes_to_f32(&input.beta_data, channels);
    let mean_raw = bytes_to_f32(&input.mean_data, channels);
    let var_raw = bytes_to_f32(&input.var_data, channels);

    if data.len() < total
        || gamma.len() < channels
        || beta.len() < channels
        || mean_raw.len() < channels
        || var_raw.len() < channels
    {
        return;
    }

    let data = &data[..total];
    let gamma = &gamma[..channels];
    let beta = &beta[..channels];
    let running_mean = &mean_raw[..channels];
    // Variance must be non-negative.
    let running_var: Vec<f32> = var_raw[..channels].iter().map(|&v| v.abs()).collect();

    // Skip non-finite inputs.
    if data
        .iter()
        .chain(gamma.iter())
        .chain(beta.iter())
        .chain(running_mean.iter())
        .chain(running_var.iter())
        .any(|x| !x.is_finite())
    {
        return;
    }

    let config =
        BatchNormConfig { num_features: channels, eps, momentum, training: input.training };

    // Test batch_norm_forward (training path).
    if let Ok((output, updated_mean, updated_var)) =
        batch_norm_forward(data, gamma, beta, running_mean, &running_var, &config)
    {
        assert_eq!(output.len(), total, "forward output length mismatch");
        assert_eq!(updated_mean.len(), channels, "updated mean length mismatch");
        assert_eq!(updated_var.len(), channels, "updated var length mismatch");

        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "forward non-finite at {i}: {val}");
        }
        for (i, &val) in updated_mean.iter().enumerate() {
            assert!(val.is_finite(), "updated mean non-finite at {i}: {val}");
        }
        for (i, &val) in updated_var.iter().enumerate() {
            assert!(val.is_finite(), "updated var non-finite at {i}: {val}");
        }
    }

    // Test batch_norm_inference.
    if let Ok(output) = batch_norm_inference(data, gamma, beta, running_mean, &running_var, eps) {
        assert_eq!(output.len(), total, "inference output length mismatch");
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "inference non-finite at {i}: {val}");
        }
    }

    // Invariant: input == mean per channel → output ≈ beta (with gamma=1).
    let ones = vec![1.0f32; channels];
    let mut at_mean = Vec::with_capacity(total);
    for _ in 0..batch_size {
        at_mean.extend_from_slice(running_mean);
    }
    if let Ok(out) = batch_norm_inference(&at_mean, &ones, beta, running_mean, &running_var, eps) {
        for (i, &val) in out.iter().enumerate() {
            let ch = i % channels;
            assert!(
                (val - beta[ch]).abs() < 1e-2,
                "at-mean output should ≈ beta[{ch}]={}, got {val}",
                beta[ch]
            );
        }
    }
});
