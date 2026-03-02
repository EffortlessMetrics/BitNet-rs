#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::batch_norm::{BatchNormConfig, batch_norm_forward, batch_norm_inference};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct BatchNormNumericalInput {
    channels: u8,
    batch_size: u8,
    data: Vec<u8>,
    gamma_data: Vec<u8>,
    beta_data: Vec<u8>,
    mean_data: Vec<u8>,
    var_data: Vec<u8>,
    eps_byte: u8,
    momentum_byte: u8,
    inject_extreme: bool,
    extreme_positions: Vec<u8>,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: BatchNormNumericalInput| {
    let channels = (input.channels as usize % 32) + 1;
    let batch_size = (input.batch_size as usize % 8) + 1;
    let total = batch_size * channels;
    let eps = 1e-7 + (input.eps_byte as f32 / 255.0) * 1e-3;
    let momentum = (input.momentum_byte as f32 / 255.0).clamp(0.01, 0.99);

    let mut data = bytes_to_f32(&input.data, total);
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

    // Inject extreme but finite values to stress numerical stability.
    if input.inject_extreme {
        for (i, &pos) in input.extreme_positions.iter().take(8).enumerate() {
            let idx = pos as usize % total;
            match i % 6 {
                0 => data[idx] = 1e30,
                1 => data[idx] = -1e30,
                2 => data[idx] = 1e-30,
                3 => data[idx] = -1e-30,
                4 => data[idx] = f32::MIN_POSITIVE,
                5 => data[idx] = -f32::MIN_POSITIVE,
                _ => {}
            }
        }
    }

    // Filter out non-finite inputs.
    if data[..total]
        .iter()
        .chain(gamma[..channels].iter())
        .chain(beta[..channels].iter())
        .chain(mean_raw[..channels].iter())
        .chain(var_raw[..channels].iter())
        .any(|x| !x.is_finite())
    {
        return;
    }

    let running_var: Vec<f32> = var_raw[..channels].iter().map(|&v| v.abs()).collect();

    let config = BatchNormConfig { num_features: channels, eps, momentum, training: true };

    // Test forward (training) path.
    if let Ok((output, updated_mean, updated_var)) = batch_norm_forward(
        &data[..total],
        &gamma[..channels],
        &beta[..channels],
        &mean_raw[..channels],
        &running_var,
        &config,
    ) {
        assert_eq!(output.len(), total);
        for (i, &val) in output.iter().enumerate() {
            assert!(!val.is_nan(), "forward NaN at {i}");
            assert!(!val.is_infinite(), "forward Inf at {i}");
        }
        for (i, &val) in updated_mean.iter().enumerate() {
            assert!(!val.is_nan(), "updated_mean NaN at {i}");
            assert!(!val.is_infinite(), "updated_mean Inf at {i}");
        }
        for (i, &val) in updated_var.iter().enumerate() {
            assert!(!val.is_nan(), "updated_var NaN at {i}");
            assert!(val >= 0.0 || !val.is_finite(), "updated_var negative at {i}: {val}");
        }
    }

    // Test inference path.
    let infer_config = BatchNormConfig { num_features: channels, eps, momentum, training: false };
    if let Ok(output) = batch_norm_inference(
        &data[..total],
        &gamma[..channels],
        &beta[..channels],
        &mean_raw[..channels],
        &running_var,
        eps,
    ) {
        assert_eq!(output.len(), total);
        for (i, &val) in output.iter().enumerate() {
            assert!(!val.is_nan(), "inference NaN at {i}");
            assert!(!val.is_infinite(), "inference Inf at {i}");
        }
    }

    // Invariant: zero variance with non-zero eps should not produce NaN.
    let zero_var = vec![0.0f32; channels];
    if let Ok(output) = batch_norm_inference(
        &data[..total],
        &gamma[..channels],
        &beta[..channels],
        &mean_raw[..channels],
        &zero_var,
        eps,
    ) {
        for (i, &val) in output.iter().enumerate() {
            assert!(!val.is_nan(), "zero-var inference NaN at {i}");
        }
    }

    // Invariant: identical inputs across batch should produce identical outputs.
    let single_sample: Vec<f32> = data[..channels].to_vec();
    let repeated: Vec<f32> = single_sample.iter().copied().cycle().take(total).collect();
    if let Ok(output) = batch_norm_inference(
        &repeated,
        &gamma[..channels],
        &beta[..channels],
        &mean_raw[..channels],
        &running_var,
        eps,
    ) {
        for bi in 1..batch_size {
            for c in 0..channels {
                let first = output[c];
                let other = output[bi * channels + c];
                if first.is_finite() && other.is_finite() {
                    assert!(
                        (first - other).abs() < 1e-5,
                        "batch consistency violation ch={c} b0={first} b{bi}={other}"
                    );
                }
            }
        }
    }

    // Suppress unused variable warning.
    let _ = infer_config;
});
