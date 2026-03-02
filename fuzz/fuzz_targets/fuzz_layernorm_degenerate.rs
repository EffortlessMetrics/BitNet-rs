#![no_main]

//! Fuzz layer normalization with degenerate and edge-case inputs.
//!
//! Exercises `bitnet_kernels::cpu::layer_norm` with adversarial patterns:
//! all-zero, single-element, huge batch, tiny eps, constant input, and
//! alternating sign patterns.

use arbitrary::Arbitrary;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug, Clone, Copy)]
enum DegeneratePattern {
    AllZero,
    AllSame { value: f32 },
    AlternatingSign { magnitude: f32 },
    SingleLargeSpike { position_frac: u8, magnitude: f32 },
    GradualRamp,
    TinyValues,
}

#[derive(Arbitrary, Debug)]
struct DegenerateInput {
    /// Normalization dimension (clamped to [1, 256]).
    norm_dim: u8,
    /// Batch count (clamped to [1, 8]).
    batch: u8,
    /// Epsilon exponent: eps = 10^(-exp), clamped to [1, 12].
    eps_exp: u8,
    /// Pattern to generate.
    pattern: DegeneratePattern,
    /// Whether to test RMS norm.
    use_rms: bool,
    /// Whether to include beta.
    use_beta: bool,
    /// Gamma scaling factor.
    gamma_scale: f32,
}

const MAX_DIM: usize = 256;
const MAX_BATCH: usize = 8;

fuzz_target!(|input: DegenerateInput| {
    let dim = (input.norm_dim as usize % MAX_DIM) + 1;
    let batch = (input.batch as usize % MAX_BATCH) + 1;
    let total = batch * dim;
    let eps_exp = (input.eps_exp % 12).max(1);
    let eps = 10.0f32.powi(-(eps_exp as i32));

    // Generate input based on degenerate pattern.
    let data: Vec<f32> = match input.pattern {
        DegeneratePattern::AllZero => vec![0.0; total],
        DegeneratePattern::AllSame { value } => {
            let v = if value.is_finite() { value.clamp(-1e6, 1e6) } else { 1.0 };
            vec![v; total]
        }
        DegeneratePattern::AlternatingSign { magnitude } => {
            let m = if magnitude.is_finite() { magnitude.abs().clamp(1e-10, 1e6) } else { 1.0 };
            (0..total).map(|i| if i % 2 == 0 { m } else { -m }).collect()
        }
        DegeneratePattern::SingleLargeSpike { position_frac, magnitude } => {
            let m = if magnitude.is_finite() { magnitude.clamp(-1e6, 1e6) } else { 1e5 };
            let mut v = vec![0.0f32; total];
            for b in 0..batch {
                let pos = (position_frac as usize) % dim;
                v[b * dim + pos] = m;
            }
            v
        }
        DegeneratePattern::GradualRamp => (0..total).map(|i| (i as f32) / (total as f32)).collect(),
        DegeneratePattern::TinyValues => {
            (0..total).map(|i| f32::MIN_POSITIVE * (1.0 + (i as f32))).collect()
        }
    };

    // Gamma: scaled ones (clamped to finite).
    let gs =
        if input.gamma_scale.is_finite() { input.gamma_scale.clamp(-100.0, 100.0) } else { 1.0 };
    let gamma = vec![gs; dim];
    let beta_vec = vec![0.0f32; dim];
    let beta = if input.use_beta { Some(beta_vec.as_slice()) } else { None };

    let config = LayerNormConfig { normalized_shape: vec![dim], eps, elementwise_affine: true };

    if input.use_rms {
        match rms_norm(&data, &gamma, &config) {
            Ok(out) => {
                assert_eq!(out.len(), total, "rms_norm length mismatch");
            }
            Err(_) => {}
        }
    } else {
        match layer_norm(&data, &gamma, beta, &config) {
            Ok(out) => {
                assert_eq!(out.len(), total, "layer_norm length mismatch");
            }
            Err(_) => {}
        }
    }

    // Cross-check: both layer_norm and rms_norm must not panic.
    let _ = layer_norm(&data, &gamma, beta, &config);
    let _ = rms_norm(&data, &gamma, &config);
});
