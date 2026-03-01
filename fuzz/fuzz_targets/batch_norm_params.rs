#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct BatchNormInput {
    /// Channel count (clamped to small range).
    channels: u8,
    /// Batch size (clamped to small range).
    batch_size: u8,
    /// Raw input data (f32 bytes).
    data: Vec<u8>,
    /// Running mean bytes.
    mean_data: Vec<u8>,
    /// Running variance bytes.
    var_data: Vec<u8>,
    /// Gamma (scale) bytes.
    gamma_data: Vec<u8>,
    /// Beta (shift) bytes.
    beta_data: Vec<u8>,
    /// Epsilon selector.
    eps: u8,
    /// Momentum selector for running stats update.
    momentum: u8,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Batch normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta
fn batch_norm(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    eps: f32,
) -> Vec<f32> {
    input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let c = i % channels;
            let inv_std = 1.0 / (var[c] + eps).sqrt();
            gamma[c] * (x - mean[c]) * inv_std + beta[c]
        })
        .collect()
}

/// Update running statistics with exponential moving average.
fn update_running_stats(running: &mut [f32], batch_stat: &[f32], momentum: f32) {
    for (r, &b) in running.iter_mut().zip(batch_stat.iter()) {
        *r = (1.0 - momentum) * *r + momentum * b;
    }
}

fuzz_target!(|input: BatchNormInput| {
    let channels = (input.channels as usize % 32) + 1;
    let batch_size = (input.batch_size as usize % 8) + 1;
    let total = batch_size * channels;
    let eps = 1e-5 * (1.0 + input.eps as f32);
    let momentum = (input.momentum as f32 % 100.0) / 100.0;

    let data = bytes_to_f32(&input.data, total);
    let mean_raw = bytes_to_f32(&input.mean_data, channels);
    let var_raw = bytes_to_f32(&input.var_data, channels);
    let gamma_raw = bytes_to_f32(&input.gamma_data, channels);
    let beta_raw = bytes_to_f32(&input.beta_data, channels);

    if data.len() < total
        || mean_raw.len() < channels
        || var_raw.len() < channels
        || gamma_raw.len() < channels
        || beta_raw.len() < channels
    {
        return;
    }

    let mean = &mean_raw[..channels];
    let var_vals: Vec<f32> = var_raw[..channels].iter().map(|&v| v.abs()).collect();
    let gamma = &gamma_raw[..channels];
    let beta = &beta_raw[..channels];

    // Skip non-finite inputs.
    if data[..total]
        .iter()
        .chain(mean.iter())
        .chain(var_vals.iter())
        .chain(gamma.iter())
        .chain(beta.iter())
        .any(|x| !x.is_finite())
    {
        return;
    }

    // --- Invariant 1: Output dimension matches input ---
    let output = batch_norm(&data[..total], mean, &var_vals, gamma, beta, channels, eps);
    assert_eq!(output.len(), total, "output dimension mismatch");

    // --- Invariant 2: Output contains no NaN/Inf ---
    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "non-finite at idx {i}: {val}");
    }

    // --- Invariant 3: With gamma=1, beta=0, constant input per channel produces zero ---
    let ones = vec![1.0f32; channels];
    let zeros = vec![0.0f32; channels];
    let mut const_input = Vec::with_capacity(total);
    for _ in 0..batch_size {
        for c in 0..channels {
            const_input.push(mean[c]); // input == mean → normalized to 0
        }
    }
    let const_out = batch_norm(&const_input, mean, &var_vals, &ones, &zeros, channels, eps);
    for (i, &val) in const_out.iter().enumerate() {
        assert!(val.abs() < 1e-3, "constant input at mean should yield ~0, got {val} at idx {i}");
    }

    // --- Invariant 4: Running stats update preserves vector length ---
    let mut running_mean = mean.to_vec();
    let batch_mean: Vec<f32> = (0..channels)
        .map(|c| data[..total].iter().skip(c).step_by(channels).sum::<f32>() / batch_size as f32)
        .collect();
    update_running_stats(&mut running_mean, &batch_mean, momentum);
    assert_eq!(running_mean.len(), channels, "running mean length changed");
    for &val in &running_mean {
        assert!(val.is_finite(), "running mean became non-finite: {val}");
    }

    // --- Invariant 5: Momentum=0 leaves running stats unchanged ---
    let mut frozen_mean = mean.to_vec();
    update_running_stats(&mut frozen_mean, &batch_mean, 0.0);
    for (i, (&orig, &updated)) in mean.iter().zip(frozen_mean.iter()).enumerate() {
        assert!(
            (orig - updated).abs() < 1e-6,
            "momentum=0 should not change stats, idx {i}: {orig} vs {updated}"
        );
    }
});
