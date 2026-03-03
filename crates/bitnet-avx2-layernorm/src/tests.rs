//! Comprehensive tests for AVX2 `LayerNorm`, `RMSNorm`, and `BatchNorm`.

#![allow(
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    clippy::approx_constant,
    clippy::single_char_pattern
)]

use crate::scalar;
use crate::{BatchNorm, BatchNormParams, LayerNorm, NormError, RmsNorm};

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff <= tol, "{label}[{i}]: {x} vs {y}, diff={diff} > tol={tol}");
    }
}

const EPS: f32 = 1e-5;
const TOL: f32 = 1e-4;

// Sizes that exercise AVX2 chunks (8-wide) and scalar remainder
const TEST_SIZES: &[usize] = &[1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 255, 256, 257];

fn make_input(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.1) - (n as f32 * 0.05)).collect()
}
fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}
fn zeros(n: usize) -> Vec<f32> {
    vec![0.0; n]
}

// =========================================================================
// LayerNorm tests
// =========================================================================

#[test]
fn layer_norm_identity_gamma_zero_beta() {
    for &n in TEST_SIZES {
        let input = make_input(n);
        let ln = LayerNorm::new(ones(n), zeros(n), EPS).unwrap();
        let out = ln.forward_alloc(&input).unwrap();
        let mut ref_out = vec![0.0; n];
        scalar::layer_norm(&input, &ones(n), &zeros(n), EPS, &mut ref_out);
        assert_close(&out, &ref_out, TOL, &format!("ln_identity n={n}"));
    }
}

#[test]
fn layer_norm_with_gamma_and_beta() {
    for &n in TEST_SIZES {
        let input = make_input(n);
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let beta: Vec<f32> = (0..n).map(|i| -0.1 + (i as f32) * 0.005).collect();
        let ln = LayerNorm::new(gamma.clone(), beta.clone(), EPS).unwrap();
        let out = ln.forward_alloc(&input).unwrap();
        let mut ref_out = vec![0.0; n];
        scalar::layer_norm(&input, &gamma, &beta, EPS, &mut ref_out);
        assert_close(&out, &ref_out, TOL, &format!("ln_gamma_beta n={n}"));
    }
}

#[test]
fn layer_norm_constant_input() {
    let n = 64;
    let input = vec![3.14; n];
    let ln = LayerNorm::new(ones(n), zeros(n), EPS).unwrap();
    let out = ln.forward_alloc(&input).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(v.abs() < 0.01, "constant input should normalize near zero, got {v} at {i}");
    }
}

#[test]
fn layer_norm_single_element() {
    let ln = LayerNorm::new(vec![2.0], vec![0.5], EPS).unwrap();
    let out = ln.forward_alloc(&[7.0]).unwrap();
    // (7 - 7) / sqrt(0 + eps) * 2 + 0.5 = 0.5
    assert!((out[0] - 0.5).abs() < TOL, "single elem: {}", out[0]);
}

#[test]
fn layer_norm_negative_values() {
    let n = 32;
    let input: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.3).collect();
    let ln = LayerNorm::new(ones(n), zeros(n), EPS).unwrap();
    let out = ln.forward_alloc(&input).unwrap();
    let mut ref_out = vec![0.0; n];
    scalar::layer_norm(&input, &ones(n), &zeros(n), EPS, &mut ref_out);
    assert_close(&out, &ref_out, TOL, "ln_negative");
}

#[test]
fn layer_norm_large_values() {
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 1000.0).collect();
    let ln = LayerNorm::new(ones(n), zeros(n), EPS).unwrap();
    let out = ln.forward_alloc(&input).unwrap();
    let mut ref_out = vec![0.0; n];
    scalar::layer_norm(&input, &ones(n), &zeros(n), EPS, &mut ref_out);
    assert_close(&out, &ref_out, TOL, "ln_large");
}

#[test]
fn layer_norm_tiny_epsilon() {
    let n = 32;
    let input = make_input(n);
    let tiny_eps = 1e-12;
    let ln = LayerNorm::new(ones(n), zeros(n), tiny_eps).unwrap();
    let out = ln.forward_alloc(&input).unwrap();
    let mut ref_out = vec![0.0; n];
    scalar::layer_norm(&input, &ones(n), &zeros(n), tiny_eps, &mut ref_out);
    assert_close(&out, &ref_out, TOL, "ln_tiny_eps");
}

#[test]
fn layer_norm_large_epsilon() {
    let n = 32;
    let input = make_input(n);
    let large_eps = 1.0;
    let ln = LayerNorm::new(ones(n), zeros(n), large_eps).unwrap();
    let out = ln.forward_alloc(&input).unwrap();
    let mut ref_out = vec![0.0; n];
    scalar::layer_norm(&input, &ones(n), &zeros(n), large_eps, &mut ref_out);
    assert_close(&out, &ref_out, TOL, "ln_large_eps");
}

#[test]
fn layer_norm_zero_input() {
    let n = 16;
    let input = zeros(n);
    let ln = LayerNorm::new(ones(n), zeros(n), EPS).unwrap();
    let out = ln.forward_alloc(&input).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(v.abs() < TOL, "zero input should stay ~0, got {v} at {i}");
    }
}

#[test]
fn layer_norm_inplace_output() {
    let n = 32;
    let input = make_input(n);
    let ln = LayerNorm::new(ones(n), zeros(n), EPS).unwrap();
    let mut out = vec![999.0; n];
    ln.forward(&input, &mut out).unwrap();
    assert!(out.iter().all(|&v| (v - 999.0).abs() > f32::EPSILON), "output should be overwritten");
}

// =========================================================================
// RMSNorm tests
// =========================================================================

#[test]
fn rms_norm_identity_gamma() {
    for &n in TEST_SIZES {
        let input = make_input(n);
        let rn = RmsNorm::new(ones(n), EPS).unwrap();
        let out = rn.forward_alloc(&input).unwrap();
        let mut ref_out = vec![0.0; n];
        scalar::rms_norm(&input, &ones(n), EPS, &mut ref_out);
        assert_close(&out, &ref_out, TOL, &format!("rms_identity n={n}"));
    }
}

#[test]
fn rms_norm_with_gamma() {
    for &n in TEST_SIZES {
        let input = make_input(n);
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.02).collect();
        let rn = RmsNorm::new(gamma.clone(), EPS).unwrap();
        let out = rn.forward_alloc(&input).unwrap();
        let mut ref_out = vec![0.0; n];
        scalar::rms_norm(&input, &gamma, EPS, &mut ref_out);
        assert_close(&out, &ref_out, TOL, &format!("rms_gamma n={n}"));
    }
}

#[test]
fn rms_norm_constant_input() {
    let n = 64;
    let input = vec![2.0; n];
    let rn = RmsNorm::new(ones(n), EPS).unwrap();
    let out = rn.forward_alloc(&input).unwrap();
    // RMS of constant c = c, so result = c / c = 1.0
    for (i, &v) in out.iter().enumerate() {
        assert!((v - 1.0).abs() < TOL, "rms constant: got {v} at {i}");
    }
}

#[test]
fn rms_norm_single_element() {
    let rn = RmsNorm::new(vec![3.0], EPS).unwrap();
    let out = rn.forward_alloc(&[5.0]).unwrap();
    let expected = 3.0 * 5.0 / (25.0_f32 + EPS).sqrt();
    assert!((out[0] - expected).abs() < TOL, "rms single: {} vs {expected}", out[0]);
}

#[test]
fn rms_norm_negative_values() {
    let n = 32;
    let input: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.3).collect();
    let rn = RmsNorm::new(ones(n), EPS).unwrap();
    let out = rn.forward_alloc(&input).unwrap();
    let mut ref_out = vec![0.0; n];
    scalar::rms_norm(&input, &ones(n), EPS, &mut ref_out);
    assert_close(&out, &ref_out, TOL, "rms_negative");
}

#[test]
fn rms_norm_large_values() {
    let n = 16;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 500.0).collect();
    let rn = RmsNorm::new(ones(n), EPS).unwrap();
    let out = rn.forward_alloc(&input).unwrap();
    let mut ref_out = vec![0.0; n];
    scalar::rms_norm(&input, &ones(n), EPS, &mut ref_out);
    assert_close(&out, &ref_out, TOL, "rms_large");
}

#[test]
fn rms_norm_preserves_sign() {
    let n = 8;
    let input = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let rn = RmsNorm::new(ones(n), EPS).unwrap();
    let out = rn.forward_alloc(&input).unwrap();
    for (i, (&inp, &o)) in input.iter().zip(out.iter()).enumerate() {
        if inp > 0.0 {
            assert!(o > 0.0, "rms sign pos at {i}");
        } else if inp < 0.0 {
            assert!(o < 0.0, "rms sign neg at {i}");
        }
    }
}

// =========================================================================
// BatchNorm tests
// =========================================================================

fn make_bn(n: usize) -> BatchNorm {
    let gamma: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32) * 0.01).collect();
    let beta: Vec<f32> = (0..n).map(|i| 0.1 - (i as f32) * 0.005).collect();
    let running_mean: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
    let running_var: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.01).collect();
    BatchNorm::new(BatchNormParams { gamma, beta, running_mean, running_var, epsilon: EPS })
        .unwrap()
}

#[test]
fn batch_norm_various_sizes() {
    for &n in TEST_SIZES {
        let input = make_input(n);
        let bn = make_bn(n);
        let out = bn.forward_alloc(&input).unwrap();

        let gamma: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32) * 0.01).collect();
        let beta: Vec<f32> = (0..n).map(|i| 0.1 - (i as f32) * 0.005).collect();
        let rm: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let rv: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let mut ref_out = vec![0.0; n];
        scalar::batch_norm(&input, &gamma, &beta, &rm, &rv, EPS, &mut ref_out);
        assert_close(&out, &ref_out, TOL, &format!("bn n={n}"));
    }
}

#[test]
fn batch_norm_identity_stats() {
    let n = 32;
    let input = make_input(n);
    let bn = BatchNorm::new(BatchNormParams {
        gamma: ones(n),
        beta: zeros(n),
        running_mean: zeros(n),
        running_var: ones(n),
        epsilon: EPS,
    })
    .unwrap();
    let out = bn.forward_alloc(&input).unwrap();
    // With mean=0, var=1, gamma=1, beta=0 => output ≈ input / sqrt(1+eps) ≈ input
    for (i, (&inp, &o)) in input.iter().zip(out.iter()).enumerate() {
        assert!((inp - o).abs() < TOL, "bn identity: {inp} vs {o} at {i}");
    }
}

#[test]
fn batch_norm_zero_mean_shift() {
    let n = 16;
    let input = vec![5.0; n];
    let bn = BatchNorm::new(BatchNormParams {
        gamma: ones(n),
        beta: zeros(n),
        running_mean: vec![5.0; n],
        running_var: ones(n),
        epsilon: EPS,
    })
    .unwrap();
    let out = bn.forward_alloc(&input).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(v.abs() < TOL, "bn zero shift: got {v} at {i}");
    }
}

// =========================================================================
// Error handling tests
// =========================================================================

#[test]
fn layer_norm_dimension_mismatch_gamma_beta() {
    let result = LayerNorm::new(vec![1.0; 4], vec![1.0; 5], EPS);
    assert!(matches!(result, Err(NormError::DimensionMismatch { .. })));
}

#[test]
fn layer_norm_dimension_mismatch_input() {
    let ln = LayerNorm::new(vec![1.0; 4], vec![0.0; 4], EPS).unwrap();
    let result = ln.forward_alloc(&[1.0, 2.0, 3.0]);
    assert!(matches!(result, Err(NormError::DimensionMismatch { .. })));
}

#[test]
fn layer_norm_dimension_mismatch_output() {
    let ln = LayerNorm::new(vec![1.0; 4], vec![0.0; 4], EPS).unwrap();
    let mut out = vec![0.0; 3];
    let result = ln.forward(&[1.0, 2.0, 3.0, 4.0], &mut out);
    assert!(matches!(result, Err(NormError::DimensionMismatch { .. })));
}

#[test]
fn layer_norm_invalid_epsilon_zero() {
    let result = LayerNorm::new(vec![1.0], vec![0.0], 0.0);
    assert!(matches!(result, Err(NormError::InvalidEpsilon(_))));
}

#[test]
fn layer_norm_invalid_epsilon_negative() {
    let result = LayerNorm::new(vec![1.0], vec![0.0], -1.0);
    assert!(matches!(result, Err(NormError::InvalidEpsilon(_))));
}

#[test]
fn rms_norm_dimension_mismatch() {
    let rn = RmsNorm::new(vec![1.0; 4], EPS).unwrap();
    let result = rn.forward_alloc(&[1.0, 2.0]);
    assert!(matches!(result, Err(NormError::DimensionMismatch { .. })));
}

#[test]
fn rms_norm_invalid_epsilon() {
    let result = RmsNorm::new(vec![1.0], 0.0);
    assert!(matches!(result, Err(NormError::InvalidEpsilon(_))));
}

#[test]
fn batch_norm_dimension_mismatch_beta() {
    let result = BatchNorm::new(BatchNormParams {
        gamma: vec![1.0; 4],
        beta: vec![0.0; 3],
        running_mean: vec![0.0; 4],
        running_var: vec![1.0; 4],
        epsilon: EPS,
    });
    assert!(matches!(result, Err(NormError::DimensionMismatch { .. })));
}

#[test]
fn batch_norm_dimension_mismatch_running_mean() {
    let result = BatchNorm::new(BatchNormParams {
        gamma: vec![1.0; 4],
        beta: vec![0.0; 4],
        running_mean: vec![0.0; 2],
        running_var: vec![1.0; 4],
        epsilon: EPS,
    });
    assert!(matches!(result, Err(NormError::DimensionMismatch { .. })));
}

#[test]
fn batch_norm_dimension_mismatch_running_var() {
    let result = BatchNorm::new(BatchNormParams {
        gamma: vec![1.0; 4],
        beta: vec![0.0; 4],
        running_mean: vec![0.0; 4],
        running_var: vec![1.0; 5],
        epsilon: EPS,
    });
    assert!(matches!(result, Err(NormError::DimensionMismatch { .. })));
}

#[test]
fn batch_norm_dimension_mismatch_input() {
    let bn = make_bn(8);
    let result = bn.forward_alloc(&[1.0; 4]);
    assert!(matches!(result, Err(NormError::DimensionMismatch { .. })));
}

#[test]
fn batch_norm_invalid_epsilon() {
    let result = BatchNorm::new(BatchNormParams {
        gamma: vec![1.0],
        beta: vec![0.0],
        running_mean: vec![0.0],
        running_var: vec![1.0],
        epsilon: -0.001,
    });
    assert!(matches!(result, Err(NormError::InvalidEpsilon(_))));
}

// =========================================================================
// Accessor / #[must_use] tests
// =========================================================================

#[test]
fn layer_norm_accessors() {
    let ln = LayerNorm::new(vec![1.0, 2.0], vec![3.0, 4.0], 0.01).unwrap();
    assert_eq!(ln.dim(), 2);
    assert!((ln.epsilon() - 0.01).abs() < f32::EPSILON);
    assert_eq!(ln.gamma(), &[1.0, 2.0]);
    assert_eq!(ln.beta(), &[3.0, 4.0]);
}

#[test]
fn rms_norm_accessors() {
    let rn = RmsNorm::new(vec![1.0, 2.0, 3.0], 0.001).unwrap();
    assert_eq!(rn.dim(), 3);
    assert!((rn.epsilon() - 0.001).abs() < f32::EPSILON);
    assert_eq!(rn.gamma(), &[1.0, 2.0, 3.0]);
}

#[test]
fn batch_norm_accessors() {
    let bn = make_bn(4);
    assert_eq!(bn.dim(), 4);
    assert!((bn.epsilon() - EPS).abs() < f32::EPSILON);
}

// =========================================================================
// NormError Display tests
// =========================================================================

#[test]
fn norm_error_display_dimension_mismatch() {
    let e = NormError::DimensionMismatch { input_len: 4, param_len: 5, param_name: "beta" };
    let s = e.to_string();
    assert!(s.contains("dimension mismatch"));
    assert!(s.contains("4"));
    assert!(s.contains("5"));
}

#[test]
fn norm_error_display_invalid_epsilon() {
    let e = NormError::InvalidEpsilon(-1.0);
    assert!(e.to_string().contains("invalid epsilon"));
}

#[test]
fn norm_error_display_empty_input() {
    let e = NormError::EmptyInput;
    assert!(e.to_string().contains("empty"));
}

// =========================================================================
// Scalar-only unit tests (mean, variance, mean_of_squares)
// =========================================================================

#[test]
fn scalar_mean_basic() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let m = scalar::mean(&data);
    assert!((m - 2.5).abs() < TOL);
}

#[test]
fn scalar_mean_empty() {
    assert!((scalar::mean(&[]) - 0.0).abs() < f32::EPSILON);
}

#[test]
fn scalar_variance_basic() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let m = scalar::mean(&data);
    let v = scalar::variance(&data, m);
    // var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4 = 1.25
    assert!((v - 1.25).abs() < TOL);
}

#[test]
fn scalar_mean_of_squares_basic() {
    let data = [1.0, 2.0, 3.0];
    let ms = scalar::mean_of_squares(&data);
    // (1 + 4 + 9) / 3 = 14/3 ≈ 4.6667
    assert!((ms - 14.0 / 3.0).abs() < TOL);
}

// =========================================================================
// AVX2 runtime detection test
// =========================================================================

#[test]
fn avx2_available_returns_bool() {
    // Just verify it doesn't panic and returns a bool.
    let _available = crate::avx2_available();
}

// =========================================================================
// Edge cases: non-8-aligned sizes
// =========================================================================

#[test]
fn layer_norm_size_1_to_9() {
    for n in 1..=9 {
        let input = make_input(n);
        let ln = LayerNorm::new(ones(n), zeros(n), EPS).unwrap();
        let out = ln.forward_alloc(&input).unwrap();
        let mut ref_out = vec![0.0; n];
        scalar::layer_norm(&input, &ones(n), &zeros(n), EPS, &mut ref_out);
        assert_close(&out, &ref_out, TOL, &format!("ln_edge n={n}"));
    }
}

#[test]
fn rms_norm_size_1_to_9() {
    for n in 1..=9 {
        let input = make_input(n);
        let rn = RmsNorm::new(ones(n), EPS).unwrap();
        let out = rn.forward_alloc(&input).unwrap();
        let mut ref_out = vec![0.0; n];
        scalar::rms_norm(&input, &ones(n), EPS, &mut ref_out);
        assert_close(&out, &ref_out, TOL, &format!("rms_edge n={n}"));
    }
}

#[test]
fn batch_norm_size_1_to_9() {
    for n in 1..=9 {
        let input = make_input(n);
        let bn = make_bn(n);
        let out = bn.forward_alloc(&input).unwrap();

        let gamma: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32) * 0.01).collect();
        let beta: Vec<f32> = (0..n).map(|i| 0.1 - (i as f32) * 0.005).collect();
        let rm: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let rv: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let mut ref_out = vec![0.0; n];
        scalar::batch_norm(&input, &gamma, &beta, &rm, &rv, EPS, &mut ref_out);
        assert_close(&out, &ref_out, TOL, &format!("bn_edge n={n}"));
    }
}

// =========================================================================
// Large dimension stress test
// =========================================================================

#[test]
fn layer_norm_large_dim() {
    let n = 4096;
    let input: Vec<f32> = (0..n).map(|i| ((i * 17 + 3) % 100) as f32 * 0.1 - 5.0).collect();
    let gamma: Vec<f32> = (0..n).map(|i| 0.9 + (i % 10) as f32 * 0.02).collect();
    let beta: Vec<f32> = (0..n).map(|i| -0.5 + (i % 10) as f32 * 0.1).collect();
    let ln = LayerNorm::new(gamma.clone(), beta.clone(), EPS).unwrap();
    let out = ln.forward_alloc(&input).unwrap();
    let mut ref_out = vec![0.0; n];
    scalar::layer_norm(&input, &gamma, &beta, EPS, &mut ref_out);
    assert_close(&out, &ref_out, TOL, "ln_large_dim");
}

#[test]
fn rms_norm_large_dim() {
    let n = 4096;
    let input: Vec<f32> = (0..n).map(|i| ((i * 17 + 3) % 100) as f32 * 0.1 - 5.0).collect();
    let gamma: Vec<f32> = (0..n).map(|i| 0.9 + (i % 10) as f32 * 0.02).collect();
    let rn = RmsNorm::new(gamma.clone(), EPS).unwrap();
    let out = rn.forward_alloc(&input).unwrap();
    let mut ref_out = vec![0.0; n];
    scalar::rms_norm(&input, &gamma, EPS, &mut ref_out);
    assert_close(&out, &ref_out, TOL, "rms_large_dim");
}

// =========================================================================
// LayerNorm output is zero-mean, unit-variance (with identity params)
// =========================================================================

#[test]
fn layer_norm_output_zero_mean_unit_var() {
    let n = 256;
    let input = make_input(n);
    let ln = LayerNorm::new(ones(n), zeros(n), EPS).unwrap();
    let out = ln.forward_alloc(&input).unwrap();
    let m = scalar::mean(&out);
    let v = scalar::variance(&out, m);
    assert!(m.abs() < 0.01, "output mean should be ~0, got {m}");
    assert!((v - 1.0).abs() < 0.01, "output var should be ~1, got {v}");
}

// =========================================================================
// RMSNorm: unit-RMS output with identity gamma
// =========================================================================

#[test]
fn rms_norm_output_unit_rms() {
    let n = 256;
    let input = make_input(n);
    let rn = RmsNorm::new(ones(n), EPS).unwrap();
    let out = rn.forward_alloc(&input).unwrap();
    let ms = scalar::mean_of_squares(&out);
    assert!((ms - 1.0).abs() < 0.01, "output RMS should be ~1, got sqrt({ms})");
}

// =========================================================================
// Mixed positive/negative gamma
// =========================================================================

#[test]
fn layer_norm_negative_gamma() {
    let n = 16;
    let input = make_input(n);
    let gamma: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let beta = zeros(n);
    let ln = LayerNorm::new(gamma.clone(), beta.clone(), EPS).unwrap();
    let out = ln.forward_alloc(&input).unwrap();
    let mut ref_out = vec![0.0; n];
    scalar::layer_norm(&input, &gamma, &beta, EPS, &mut ref_out);
    assert_close(&out, &ref_out, TOL, "ln_neg_gamma");
}

// =========================================================================
// BatchNorm with large running variance
// =========================================================================

#[test]
fn batch_norm_large_running_var() {
    let n = 32;
    let input = make_input(n);
    let bn = BatchNorm::new(BatchNormParams {
        gamma: ones(n),
        beta: zeros(n),
        running_mean: zeros(n),
        running_var: vec![1e6; n],
        epsilon: EPS,
    })
    .unwrap();
    let out = bn.forward_alloc(&input).unwrap();
    // Large var means small normalization effect
    for (i, &v) in out.iter().enumerate() {
        assert!(v.abs() < 0.01, "bn large var: got {v} at {i}");
    }
}

// =========================================================================
// Proptest: AVX2 vs scalar parity
// =========================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn vec_f32(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(-100.0_f32..100.0, min_len..=max_len)
    }

    fn positive_f32() -> impl Strategy<Value = f32> {
        (1e-10_f32..1.0).prop_map(|v| v)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_layer_norm_avx2_matches_scalar(
            input in vec_f32(1, 512),
            eps in positive_f32(),
        ) {
            let n = input.len();
            let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 % 10.0) * 0.1).collect();
            let beta: Vec<f32> = (0..n).map(|i| -0.3 + (i as f32 % 10.0) * 0.05).collect();

            let ln = LayerNorm::new(gamma.clone(), beta.clone(), eps).unwrap();
            let out = ln.forward_alloc(&input).unwrap();

            let mut ref_out = vec![0.0; n];
            scalar::layer_norm(&input, &gamma, &beta, eps, &mut ref_out);
            assert_close(&out, &ref_out, 1e-3, "prop_ln");
        }

        #[test]
        fn prop_rms_norm_avx2_matches_scalar(
            input in vec_f32(1, 512),
            eps in positive_f32(),
        ) {
            let n = input.len();
            let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 % 10.0) * 0.1).collect();

            let rn = RmsNorm::new(gamma.clone(), eps).unwrap();
            let out = rn.forward_alloc(&input).unwrap();

            let mut ref_out = vec![0.0; n];
            scalar::rms_norm(&input, &gamma, eps, &mut ref_out);
            assert_close(&out, &ref_out, 1e-3, "prop_rms");
        }

        #[test]
        fn prop_batch_norm_avx2_matches_scalar(
            input in vec_f32(1, 256),
            eps in positive_f32(),
        ) {
            let n = input.len();
            let gamma: Vec<f32> = (0..n).map(|i| 0.8 + (i as f32 % 5.0) * 0.04).collect();
            let beta: Vec<f32> = (0..n).map(|i| (i as f32 % 5.0) * 0.02).collect();
            let rm: Vec<f32> = (0..n).map(|i| (i as f32 % 10.0) * 0.1).collect();
            let rv: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 % 10.0) * 0.05).collect();

            let bn = BatchNorm::new(BatchNormParams {
                gamma: gamma.clone(),
                beta: beta.clone(),
                running_mean: rm.clone(),
                running_var: rv.clone(),
                epsilon: eps,
            }).unwrap();
            let out = bn.forward_alloc(&input).unwrap();

            let mut ref_out = vec![0.0; n];
            scalar::batch_norm(&input, &gamma, &beta, &rm, &rv, eps, &mut ref_out);
            assert_close(&out, &ref_out, 1e-3, "prop_bn");
        }

        #[test]
        fn prop_layer_norm_output_zero_mean(
            input in vec_f32(2, 512),
        ) {
            let n = input.len();
            let ln = LayerNorm::new(ones(n), zeros(n), EPS).unwrap();
            let out = ln.forward_alloc(&input).unwrap();
            let m = scalar::mean(&out);
            prop_assert!((m.abs()) < 0.1, "mean={m} too far from 0");
        }

        #[test]
        fn prop_rms_norm_preserves_sign(
            input in vec_f32(1, 256),
        ) {
            let n = input.len();
            let rn = RmsNorm::new(ones(n), EPS).unwrap();
            let out = rn.forward_alloc(&input).unwrap();
            for (i, (&inp, &o)) in input.iter().zip(out.iter()).enumerate() {
                if inp > 1e-6 {
                    prop_assert!(o > -1e-6, "sign mismatch at {i}: input={inp}, output={o}");
                } else if inp < -1e-6 {
                    prop_assert!(o < 1e-6, "sign mismatch at {i}: input={inp}, output={o}");
                }
            }
        }

        #[test]
        fn prop_layer_norm_scale_equivariance(
            input in vec_f32(4, 128),
            scale in 0.1_f32..10.0,
        ) {
            // LayerNorm(c*x) with gamma=1,beta=0 should equal LayerNorm(x)
            let n = input.len();
            let scaled: Vec<f32> = input.iter().map(|&x| x * scale).collect();
            let ln = LayerNorm::new(ones(n), zeros(n), EPS).unwrap();
            let out_orig = ln.forward_alloc(&input).unwrap();
            let out_scaled = ln.forward_alloc(&scaled).unwrap();
            assert_close(&out_orig, &out_scaled, 0.05, "prop_equivariance");
        }
    }
}
