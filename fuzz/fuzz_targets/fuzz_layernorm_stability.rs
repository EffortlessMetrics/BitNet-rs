#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct LayerNormStabilityInput {
    dim: u8,
    batch_size: u8,
    data: Vec<u8>,
    gamma_data: Vec<u8>,
    beta_data: Vec<u8>,
    inject_extreme: bool,
    extreme_positions: Vec<u8>,
    use_rms: bool,
    use_beta: bool,
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

    let mut data = bytes_to_f32(&input.data, total);
    let gamma = bytes_to_f32(&input.gamma_data, dim);
    let beta = bytes_to_f32(&input.beta_data, dim);

    if data.len() < total || gamma.len() < dim || beta.len() < dim {
        return;
    }

    // Inject extreme but finite values at fuzz-selected positions.
    if input.inject_extreme {
        for (i, &pos) in input.extreme_positions.iter().take(8).enumerate() {
            let idx = pos as usize % total;
            match i % 4 {
                0 => data[idx] = f32::MAX / 2.0,
                1 => data[idx] = f32::MIN / 2.0,
                2 => data[idx] = f32::MIN_POSITIVE,
                3 => data[idx] = -f32::MIN_POSITIVE,
                _ => {}
            }
        }
    }

    // Skip non-finite inputs.
    if data[..total]
        .iter()
        .chain(gamma[..dim].iter())
        .chain(beta[..dim].iter())
        .any(|x| !x.is_finite())
    {
        return;
    }

    let config = LayerNormConfig::new(vec![dim]);

    if input.use_rms {
        // RMS norm path.
        if let Ok(out) = rms_norm(&data[..total], &gamma[..dim], &config) {
            assert_eq!(out.len(), total, "rms_norm output length mismatch");
            for (i, &val) in out.iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "rms_norm produced NaN at index {i} (dim={dim}, batch={batch_size})"
                );
            }
        }
    } else {
        // LayerNorm path.
        let beta_ref = if input.use_beta { Some(&beta[..dim]) } else { None };
        if let Ok(out) = layer_norm(&data[..total], &gamma[..dim], beta_ref, &config) {
            assert_eq!(out.len(), total, "layer_norm output length mismatch");
            for (i, &val) in out.iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "layer_norm produced NaN at index {i} (dim={dim}, batch={batch_size})"
                );
            }
        }
    }

    // Invariant: constant input → constant output (LayerNorm with gamma=1, no beta).
    let constant_val = data[0]; // Use first element as constant
    if constant_val.is_finite() && constant_val.abs() < 1e6 {
        let const_input = vec![constant_val; dim];
        let ones = vec![1.0f32; dim];
        let unity_config = LayerNormConfig::new(vec![dim]);

        if let Ok(out) = layer_norm(&const_input, &ones, None, &unity_config) {
            // All elements should be identical after normalizing constant input.
            let first = out[0];
            for (i, &val) in out.iter().enumerate() {
                assert!(
                    (val - first).abs() < 1e-4 || !val.is_finite(),
                    "constant input diverged at idx {i}: {first} vs {val}"
                );
            }
        }
    }
});
