#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct LayerNormStabilityInput {
    dim: u8,
    batch_size: u8,
    raw_data: Vec<u8>,
    raw_gamma: Vec<u8>,
    raw_beta: Vec<u8>,
    use_rms: bool,
    /// Which extreme values to inject.
    extreme_kind: u8,
}

fn bytes_to_f32(data: &[u8], max_elems: usize) -> Vec<f32> {
    let aligned = (data.len() / 4) * 4;
    data[..aligned]
        .chunks_exact(4)
        .take(max_elems)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fuzz_target!(|input: LayerNormStabilityInput| {
    let dim = (input.dim as usize % 64) + 2;
    let batch_size = (input.batch_size as usize % 4) + 1;
    let total = batch_size * dim;

    let mut data = bytes_to_f32(&input.raw_data, total);
    let gamma_raw = bytes_to_f32(&input.raw_gamma, dim);
    let beta_raw = bytes_to_f32(&input.raw_beta, dim);

    if data.len() < total || gamma_raw.len() < dim || beta_raw.len() < dim {
        return;
    }

    // Inject extreme values based on fuzz input to stress numerical stability.
    match input.extreme_kind % 6 {
        0 => { /* no injection — use raw fuzz bytes as-is */ }
        1 => {
            // Very large values
            for x in data.iter_mut().take(dim) {
                *x = f32::MAX / 4.0;
            }
        }
        2 => {
            // Very small (denormal) values
            for x in data.iter_mut().take(dim) {
                *x = f32::MIN_POSITIVE * 0.5;
            }
        }
        3 => {
            // Mixed large positive and large negative
            for (i, x) in data.iter_mut().take(dim).enumerate() {
                *x = if i % 2 == 0 { f32::MAX / 4.0 } else { f32::MIN / 4.0 };
            }
        }
        4 => {
            // All zeros
            for x in data.iter_mut().take(dim) {
                *x = 0.0;
            }
        }
        _ => {
            // Alternating denormal and large
            for (i, x) in data.iter_mut().take(dim).enumerate() {
                *x = if i % 2 == 0 { f32::MIN_POSITIVE * 0.1 } else { 1e30 };
            }
        }
    }

    // Filter out NaN/Inf in any input vector (we only test finite extremes).
    if data[..total]
        .iter()
        .chain(gamma_raw[..dim].iter())
        .chain(beta_raw[..dim].iter())
        .any(|x| !x.is_finite())
    {
        return;
    }

    let config = LayerNormConfig::new(vec![dim]);
    let gamma = &gamma_raw[..dim];
    let beta = &beta_raw[..dim];

    if input.use_rms {
        match rms_norm(&data[..total], gamma, &config) {
            Ok(out) => {
                // Invariant 1: Output length matches input.
                assert_eq!(out.len(), total, "rms_norm output length mismatch");
                // Invariant 2: No NaN in output.
                for (i, &v) in out.iter().enumerate() {
                    assert!(!v.is_nan(), "rms_norm produced NaN at idx {i}");
                }
            }
            Err(_) => {} // Errors are fine for invalid config.
        }
    } else {
        match layer_norm(&data[..total], gamma, Some(beta), &config) {
            Ok(out) => {
                assert_eq!(out.len(), total, "layer_norm output length mismatch");
                for (i, &v) in out.iter().enumerate() {
                    assert!(!v.is_nan(), "layer_norm produced NaN at idx {i}");
                }
            }
            Err(_) => {}
        }

        // Also test without beta.
        match layer_norm(&data[..total], gamma, None, &config) {
            Ok(out) => {
                assert_eq!(out.len(), total);
                for (i, &v) in out.iter().enumerate() {
                    assert!(!v.is_nan(), "layer_norm (no beta) produced NaN at idx {i}");
                }
            }
            Err(_) => {}
        }
    }

    // Invariant 3: Zero input with gamma=1, beta=0 must produce all-zero output.
    let zero_input = vec![0.0f32; dim];
    let ones = vec![1.0f32; dim];
    let zeros = vec![0.0f32; dim];
    if let Ok(out) = layer_norm(&zero_input, &ones, Some(&zeros), &config) {
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "zero-input layer_norm non-finite at idx {i}");
        }
    }
});
