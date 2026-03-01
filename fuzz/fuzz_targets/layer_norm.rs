#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use libfuzzer_sys::fuzz_target;

/// Fuzz the real `bitnet_kernels::cpu::layer_norm` API with arbitrary
/// inputs, gamma, beta, eps, and normalized shapes.
#[derive(Arbitrary, Debug)]
struct LayerNormFuzzInput {
    /// Raw f32 data bytes for the input tensor.
    data: Vec<f32>,
    /// Raw f32 data bytes for gamma (scale).
    gamma: Vec<f32>,
    /// Raw f32 data bytes for beta (shift).
    beta: Vec<f32>,
    /// Normalized shape dimensions (each clamped to small range).
    shape_dims: Vec<u8>,
    /// Epsilon selector (mapped to a positive value).
    eps_selector: u8,
    /// Whether to include beta.
    include_beta: bool,
    /// Whether to enable elementwise affine.
    elementwise_affine: bool,
    /// Whether to also test rms_norm on the same input.
    test_rms: bool,
}

fuzz_target!(|input: LayerNormFuzzInput| {
    // Build normalized_shape from arbitrary dims, capped at 3 dims.
    let shape_dims: Vec<usize> =
        input.shape_dims.iter().take(3).map(|&d| (d as usize % 32) + 1).collect();
    if shape_dims.is_empty() {
        return;
    }
    let norm_size: usize = shape_dims.iter().product();
    if norm_size == 0 || norm_size > 256 {
        return;
    }

    // Pick a positive, finite eps.
    let eps = match input.eps_selector % 4 {
        0 => 1e-12,
        1 => 1e-5,
        2 => 1e-3,
        _ => 1.0,
    };

    let config = LayerNormConfig {
        normalized_shape: shape_dims,
        eps,
        elementwise_affine: input.elementwise_affine,
    };

    // Need enough data for at least one batch.
    let data: Vec<f32> = input.data.iter().copied().take(256).collect();
    if data.len() < norm_size || data.len() % norm_size != 0 {
        return;
    }

    // Skip non-finite inputs — we want to fuzz logic, not float weirdness.
    if data.iter().any(|x| !x.is_finite()) {
        return;
    }

    let gamma: Vec<f32> = input.gamma.iter().copied().take(norm_size).collect();
    if gamma.len() != norm_size || gamma.iter().any(|x| !x.is_finite()) {
        return;
    }

    let beta_vec: Vec<f32> = input.beta.iter().copied().take(norm_size).collect();
    let beta = if input.include_beta && beta_vec.len() == norm_size {
        if beta_vec.iter().any(|x| !x.is_finite()) {
            return;
        }
        Some(beta_vec)
    } else {
        None
    };

    // --- Fuzz layer_norm ---
    let result = layer_norm(&data, &gamma, beta.as_deref(), &config);
    match result {
        Ok(output) => {
            // Invariant 1: output length == input length
            assert_eq!(output.len(), data.len());

            // Invariant 2: output is finite for finite inputs
            for (i, &v) in output.iter().enumerate() {
                assert!(v.is_finite(), "layer_norm output non-finite at {i}: {v}");
            }
        }
        Err(_) => {
            // Validation errors are acceptable (dimension mismatches, etc.)
        }
    }

    // --- Fuzz rms_norm ---
    if input.test_rms {
        let rms_result = rms_norm(&data, &gamma, &config);
        match rms_result {
            Ok(output) => {
                assert_eq!(output.len(), data.len());
                for (i, &v) in output.iter().enumerate() {
                    assert!(v.is_finite(), "rms_norm output non-finite at {i}: {v}");
                }
            }
            Err(_) => {}
        }
    }
});
