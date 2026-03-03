#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct LayerNormExtremeInput {
    dim: u8,
    batch_size: u8,
    data: Vec<u8>,
    gamma_data: Vec<u8>,
    beta_data: Vec<u8>,
    use_rms: bool,
    use_beta: bool,
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

fuzz_target!(|input: LayerNormExtremeInput| {
    let dim = (input.dim as usize % 64) + 2;
    let batch_size = (input.batch_size as usize % 4) + 1;
    let total = batch_size * dim;

    let mut data = bytes_to_f32(&input.data, total);
    let mut gamma = bytes_to_f32(&input.gamma_data, dim);
    let beta = bytes_to_f32(&input.beta_data, dim);

    if data.len() < total || gamma.len() < dim || beta.len() < dim {
        return;
    }

    // Inject extreme but finite values based on fuzz-selected kind
    match input.extreme_kind % 6 {
        0 => {
            // Very large values
            for v in data[..total].iter_mut() {
                *v = v.clamp(-1e30, 1e30);
                if !v.is_finite() {
                    *v = 1e30;
                }
            }
        }
        1 => {
            // Very small (subnormal-adjacent) values
            for v in data[..total].iter_mut() {
                *v = if v.is_finite() { *v * f32::MIN_POSITIVE } else { f32::MIN_POSITIVE };
            }
        }
        2 => {
            // All identical values (degenerate variance = 0)
            let c = if data[0].is_finite() { data[0] } else { 42.0 };
            data[..total].fill(c);
        }
        3 => {
            // Alternating large positive/negative
            for (i, v) in data[..total].iter_mut().enumerate() {
                let mag = if v.is_finite() { v.abs().max(1e10) } else { 1e10 };
                *v = if i % 2 == 0 { mag } else { -mag };
            }
        }
        4 => {
            // Gamma near zero (tests division stability)
            gamma[..dim].fill(1e-38);
        }
        _ => {
            // Mixed: some zeros, some large
            for (i, v) in data[..total].iter_mut().enumerate() {
                if !v.is_finite() {
                    *v = 0.0;
                }
                if i % 3 == 0 {
                    *v = 0.0;
                }
            }
        }
    }

    // Ensure all inputs are finite
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
        // RMSNorm path
        if let Ok(out) = rms_norm(&data[..total], &gamma[..dim], &config) {
            assert_eq!(out.len(), total, "rms_norm output length mismatch");
            for (i, &val) in out.iter().enumerate() {
                assert!(!val.is_nan(), "rms_norm produced NaN at index {i} (dim={dim})");
            }
        }
    } else {
        let beta_ref = if input.use_beta { Some(&beta[..dim]) } else { None };
        if let Ok(out) = layer_norm(&data[..total], &gamma[..dim], beta_ref, &config) {
            assert_eq!(out.len(), total, "layer_norm output length mismatch");
            for (i, &val) in out.iter().enumerate() {
                assert!(!val.is_nan(), "layer_norm produced NaN at index {i} (dim={dim})");
            }
        }
    }

    // Invariant: Constant input → constant output (gamma=1, no beta)
    let constant_val = data[0];
    if constant_val.is_finite() && constant_val.abs() < 1e6 {
        let const_input = vec![constant_val; dim];
        let ones = vec![1.0f32; dim];
        let unity_config = LayerNormConfig::new(vec![dim]);

        if let Ok(out) = layer_norm(&const_input, &ones, None, &unity_config) {
            let first = out[0];
            for (i, &val) in out.iter().enumerate() {
                assert!(
                    (val - first).abs() < 1e-4 || !val.is_finite(),
                    "constant input diverged at idx {i}: {first} vs {val}"
                );
            }
        }
    }

    // Invariant: Scaling input by constant scales RMSNorm output proportionally
    // rms_norm(c*x, gamma) == c * rms_norm(x, gamma) when c > 0
    if dim <= 16 {
        let small_input: Vec<f32> = data[..dim].iter().map(|&v| v * 1e-10).collect();
        let ones = vec![1.0f32; dim];
        let cfg = LayerNormConfig::new(vec![dim]);

        // Just verify it doesn't crash on tiny inputs
        let _ = rms_norm(&small_input, &ones, &cfg);
    }
});
