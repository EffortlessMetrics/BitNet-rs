#![cfg(feature = "cpu")]
#![allow(clippy::needless_range_loop)]
//! CPU kernel scalar-vs-SIMD parity tests.
//!
//! Exercises key CPU kernels with known inputs across multiple sizes
//! (4, 8, 16, 32 elements) to cover both scalar and potential SIMD paths.
//! All tests are pure-math — no model files required.

use bitnet_kernels::cpu::activations::{
    ActivationType, apply_activation, gelu_approx_vec, gelu_inplace, gelu_vec, relu_inplace,
    silu_inplace, silu_vec,
};
use bitnet_kernels::cpu::embedding::embedding_lookup;
use bitnet_kernels::cpu::gating::{GatingType, apply_gating, geglu, reglu, swiglu};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::linear::{LinearConfig, linear_cpu};
use bitnet_kernels::cpu::reduction::ReductionKernel;
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};

// ── Tolerance helpers ──────────────────────────────────────────────

const ABS_TOL: f32 = 1e-7;
const REL_TOL: f32 = 1e-5;

fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= ABS_TOL || diff <= REL_TOL * b.abs().max(a.abs())
}

fn assert_close(actual: f32, expected: f32, ctx: &str) {
    assert!(
        close(actual, expected),
        "{ctx}: expected {expected}, got {actual} (diff={})",
        (actual - expected).abs()
    );
}

fn assert_vec_close(actual: &[f32], expected: &[f32], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(close(a, e), "{ctx}[{i}]: expected {e}, got {a} (diff={})", (a - e).abs());
    }
}

// ── Reference scalar implementations ───────────────────────────────

fn ref_relu(x: f32) -> f32 {
    x.max(0.0)
}

fn ref_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn ref_gelu(x: f32) -> f32 {
    // erf-based GELU using f64 for precision, then truncating back.
    let xd = x as f64;
    let cdf = 0.5 * (1.0 + libm::erf(xd / std::f64::consts::SQRT_2));
    (xd * cdf) as f32
}

fn ref_gelu_approx(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    0.5 * x * (1.0 + inner.tanh())
}

// ── Test input generators ──────────────────────────────────────────

/// Returns a vector of `n` linearly spaced values in `[lo, hi]`.
fn linspace(lo: f32, hi: f32, n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![lo];
    }
    (0..n).map(|i| lo + (hi - lo) * (i as f32 / (n - 1) as f32)).collect()
}

/// Mixed edge-case inputs: zeros, negatives, large, small.
fn edge_case_inputs(n: usize) -> Vec<f32> {
    let mut v = vec![0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1e-8, -1e-8, 100.0, -100.0, 1e-30, -1e-30];
    v.truncate(n);
    while v.len() < n {
        v.push(v.len() as f32 * 0.1 - 0.5);
    }
    v
}

// ===========================================================================
// Activation tests
// ===========================================================================

mod activations {
    use super::*;

    // ── ReLU ───────────────────────────────────────────────────────

    #[test]
    fn relu_inplace_small_sizes() {
        for &n in &[4, 8, 16, 32] {
            let input = linspace(-2.0, 2.0, n);
            let mut buf = input.clone();
            relu_inplace(&mut buf);
            let expected: Vec<f32> = input.iter().map(|&x| ref_relu(x)).collect();
            assert_vec_close(&buf, &expected, &format!("relu_inplace n={n}"));
        }
    }

    #[test]
    fn relu_zeros() {
        let mut buf = vec![0.0; 8];
        relu_inplace(&mut buf);
        assert!(buf.iter().all(|&v| v == 0.0), "ReLU(0) should be 0");
    }

    #[test]
    fn relu_all_negative() {
        let mut buf = vec![-1.0, -5.0, -100.0, -1e-6];
        relu_inplace(&mut buf);
        assert!(buf.iter().all(|&v| v == 0.0), "ReLU of negatives should be 0");
    }

    #[test]
    fn relu_edge_cases() {
        for &n in &[4, 8, 16] {
            let input = edge_case_inputs(n);
            let mut buf = input.clone();
            relu_inplace(&mut buf);
            for (i, (&got, &x)) in buf.iter().zip(input.iter()).enumerate() {
                assert_close(got, ref_relu(x), &format!("relu edge n={n} [{i}]"));
            }
        }
    }

    // ── SiLU ───────────────────────────────────────────────────────

    #[test]
    fn silu_inplace_small_sizes() {
        for &n in &[4, 8, 16, 32] {
            let input = linspace(-3.0, 3.0, n);
            let mut buf = input.clone();
            silu_inplace(&mut buf);
            let expected: Vec<f32> = input.iter().map(|&x| ref_silu(x)).collect();
            assert_vec_close(&buf, &expected, &format!("silu_inplace n={n}"));
        }
    }

    #[test]
    fn silu_vec_small_sizes() {
        for &n in &[4, 8, 16, 32] {
            let input = linspace(-3.0, 3.0, n);
            let result = silu_vec(&input);
            let expected: Vec<f32> = input.iter().map(|&x| ref_silu(x)).collect();
            assert_vec_close(&result, &expected, &format!("silu_vec n={n}"));
        }
    }

    #[test]
    fn silu_zero_is_zero() {
        let result = silu_vec(&[0.0]);
        assert_close(result[0], 0.0, "silu(0)");
    }

    #[test]
    fn silu_edge_cases() {
        let input = edge_case_inputs(16);
        let result = silu_vec(&input);
        let expected: Vec<f32> = input.iter().map(|&x| ref_silu(x)).collect();
        assert_vec_close(&result, &expected, "silu_vec edge");
    }

    // ── GELU ───────────────────────────────────────────────────────

    #[test]
    fn gelu_vec_small_sizes() {
        for &n in &[4, 8, 16, 32] {
            let input = linspace(-3.0, 3.0, n);
            let result = gelu_vec(&input);
            let expected: Vec<f32> = input.iter().map(|&x| ref_gelu(x)).collect();
            assert_vec_close(&result, &expected, &format!("gelu_vec n={n}"));
        }
    }

    #[test]
    fn gelu_inplace_matches_vec() {
        let input = linspace(-2.0, 2.0, 16);
        let vec_result = gelu_vec(&input);
        let mut inplace_buf = input.clone();
        gelu_inplace(&mut inplace_buf);
        assert_vec_close(&inplace_buf, &vec_result, "gelu inplace vs vec");
    }

    #[test]
    fn gelu_zero_is_zero() {
        let result = gelu_vec(&[0.0]);
        assert_close(result[0], 0.0, "gelu(0)");
    }

    #[test]
    fn gelu_symmetry() {
        // GELU is NOT symmetric: gelu(-x) != -gelu(x), but gelu(x) > 0 for
        // large x and gelu(x) ≈ 0 for large negative x.
        let result = gelu_vec(&[5.0, -5.0]);
        assert!(result[0] > 4.9, "gelu(5) should be close to 5");
        assert!(result[1].abs() < 0.01, "gelu(-5) should be close to 0");
    }

    // ── GELU approx ───────────────────────────────────────────────

    #[test]
    fn gelu_approx_vec_small_sizes() {
        for &n in &[4, 8, 16, 32] {
            let input = linspace(-3.0, 3.0, n);
            let result = gelu_approx_vec(&input);
            let expected: Vec<f32> = input.iter().map(|&x| ref_gelu_approx(x)).collect();
            assert_vec_close(&result, &expected, &format!("gelu_approx n={n}"));
        }
    }

    #[test]
    fn gelu_approx_close_to_exact() {
        // The tanh approximation should be close to exact GELU for moderate
        // inputs (within ~0.01 absolute tolerance).
        let input = linspace(-2.0, 2.0, 32);
        let exact = gelu_vec(&input);
        let approx = gelu_approx_vec(&input);
        for (i, (&e, &a)) in exact.iter().zip(approx.iter()).enumerate() {
            let diff = (e - a).abs();
            assert!(diff < 0.02, "gelu exact vs approx [{i}]: exact={e}, approx={a}, diff={diff}");
        }
    }

    // ── apply_activation dispatch ──────────────────────────────────

    #[test]
    fn apply_activation_relu() {
        let input = linspace(-2.0, 2.0, 8);
        let result = apply_activation(&input, ActivationType::ReLU);
        let expected: Vec<f32> = input.iter().map(|&x| ref_relu(x)).collect();
        assert_vec_close(&result, &expected, "apply_activation ReLU");
    }

    #[test]
    fn apply_activation_silu() {
        let input = linspace(-2.0, 2.0, 8);
        let result = apply_activation(&input, ActivationType::SiLU);
        let expected: Vec<f32> = input.iter().map(|&x| ref_silu(x)).collect();
        assert_vec_close(&result, &expected, "apply_activation SiLU");
    }

    #[test]
    fn apply_activation_gelu() {
        let input = linspace(-2.0, 2.0, 8);
        let result = apply_activation(&input, ActivationType::GELU);
        let expected: Vec<f32> = input.iter().map(|&x| ref_gelu(x)).collect();
        assert_vec_close(&result, &expected, "apply_activation GELU");
    }

    #[test]
    fn apply_activation_gelu_tanh() {
        let input = linspace(-2.0, 2.0, 8);
        let result = apply_activation(&input, ActivationType::GELUTanh);
        let expected: Vec<f32> = input.iter().map(|&x| ref_gelu_approx(x)).collect();
        assert_vec_close(&result, &expected, "apply_activation GELUTanh");
    }

    // ── Large values ───────────────────────────────────────────────

    #[test]
    fn relu_large_positive() {
        let mut buf = vec![1e30];
        relu_inplace(&mut buf);
        assert_eq!(buf[0], 1e30);
    }

    #[test]
    fn silu_large_positive() {
        // silu(x) → x for large positive x
        let result = silu_vec(&[50.0]);
        assert_close(result[0], 50.0, "silu(50)");
    }

    #[test]
    fn silu_large_negative() {
        // silu(x) → 0 for large negative x
        let result = silu_vec(&[-50.0]);
        assert!(result[0].abs() < 1e-10, "silu(-50) should be ~0");
    }
}

// ===========================================================================
// Layer normalization tests
// ===========================================================================

mod layer_norm_tests {
    use super::*;

    fn make_config(norm_size: usize) -> LayerNormConfig {
        LayerNormConfig { normalized_shape: vec![norm_size], eps: 1e-5, elementwise_affine: true }
    }

    #[test]
    fn layer_norm_ones_gamma_zero_beta() {
        for &n in &[4, 8, 16, 32] {
            let input = linspace(-1.0, 1.0, n);
            let gamma = vec![1.0; n];
            let beta = vec![0.0; n];
            let config = make_config(n);
            let result = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();

            // After layer norm with gamma=1, beta=0, output should have
            // mean ≈ 0 and variance ≈ 1.
            let mean: f32 = result.iter().sum::<f32>() / n as f32;
            let var: f32 = result.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;

            assert_close(mean, 0.0, &format!("layer_norm mean n={n}"));
            assert!((var - 1.0).abs() < 0.05, "layer_norm var n={n}: expected ~1.0, got {var}");
        }
    }

    #[test]
    fn layer_norm_constant_input() {
        // All-same values → output should be all zeros (after centering).
        let input = vec![3.0; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let config = make_config(8);
        let result = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        for (i, &v) in result.iter().enumerate() {
            assert_close(v, 0.0, &format!("layer_norm constant [{i}]"));
        }
    }

    #[test]
    fn layer_norm_with_gamma_and_beta() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0; 4];
        let beta = vec![1.0; 4];
        let config = make_config(4);
        let result = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();

        // Verify the output is shifted and scaled.
        let mean: f32 = result.iter().sum::<f32>() / 4.0;
        assert_close(mean, 1.0, "layer_norm shifted mean");
    }

    #[test]
    fn layer_norm_no_beta() {
        let input = linspace(-1.0, 1.0, 8);
        let gamma = vec![1.0; 8];
        let config = make_config(8);
        let result = layer_norm(&input, &gamma, None, &config).unwrap();
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn layer_norm_batch() {
        // Two sequences batched together.
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let config = make_config(4);
        let result = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        assert_eq!(result.len(), 8);

        // Each batch element should independently have mean ≈ 0.
        let mean1: f32 = result[0..4].iter().sum::<f32>() / 4.0;
        let mean2: f32 = result[4..8].iter().sum::<f32>() / 4.0;
        assert_close(mean1, 0.0, "batch[0] mean");
        assert_close(mean2, 0.0, "batch[1] mean");
    }

    // ── RMS norm ───────────────────────────────────────────────────

    #[test]
    fn rms_norm_basic() {
        for &n in &[4, 8, 16] {
            let input = linspace(0.5, 2.0, n);
            let gamma = vec![1.0; n];
            let config = make_config(n);
            let result = rms_norm(&input, &gamma, &config).unwrap();
            assert_eq!(result.len(), n);

            // RMS norm: output = x / sqrt(mean(x^2) + eps) * gamma
            let rms_sq: f32 = input.iter().map(|&x| x * x).sum::<f32>() / n as f32;
            let inv_rms = 1.0 / (rms_sq + 1e-5_f32).sqrt();
            let expected: Vec<f32> = input.iter().map(|&x| x * inv_rms).collect();
            assert_vec_close(&result, &expected, &format!("rms_norm n={n}"));
        }
    }

    #[test]
    fn rms_norm_with_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0];
        let config = make_config(4);
        let result = rms_norm(&input, &gamma, &config).unwrap();

        let rms_sq: f32 = input.iter().map(|&x| x * x).sum::<f32>() / 4.0;
        let inv_rms = 1.0 / (rms_sq + 1e-5_f32).sqrt();
        let expected: Vec<f32> =
            input.iter().zip(gamma.iter()).map(|(&x, &g)| x * inv_rms * g).collect();
        assert_vec_close(&result, &expected, "rms_norm gamma");
    }

    // ── Error cases ────────────────────────────────────────────────

    #[test]
    fn layer_norm_empty_input_errors() {
        let config = make_config(4);
        assert!(layer_norm(&[], &[1.0; 4], None, &config).is_err());
    }

    #[test]
    fn layer_norm_mismatched_gamma_errors() {
        let config = make_config(4);
        // gamma length (2) doesn't match norm_size (4).
        assert!(layer_norm(&[1.0; 4], &[1.0; 2], None, &config).is_err());
    }
}

// ===========================================================================
// Embedding lookup tests
// ===========================================================================

mod embedding_tests {
    use super::*;

    #[test]
    fn embedding_lookup_basic() {
        // 3 embeddings of dim 4.
        let table = vec![
            1.0, 2.0, 3.0, 4.0, // idx 0
            5.0, 6.0, 7.0, 8.0, // idx 1
            9.0, 10.0, 11.0, 12.0, // idx 2
        ];
        let indices = vec![2, 0, 1];
        let result = embedding_lookup(&table, &indices, 4).unwrap();
        let expected = vec![
            9.0, 10.0, 11.0, 12.0, // idx 2
            1.0, 2.0, 3.0, 4.0, // idx 0
            5.0, 6.0, 7.0, 8.0, // idx 1
        ];
        assert_vec_close(&result, &expected, "embedding basic");
    }

    #[test]
    fn embedding_lookup_single() {
        let table = vec![10.0, 20.0];
        let indices = vec![0];
        let result = embedding_lookup(&table, &indices, 2).unwrap();
        assert_vec_close(&result, &[10.0, 20.0], "embedding single");
    }

    #[test]
    fn embedding_lookup_repeated_index() {
        let table = vec![1.0, 2.0, 3.0, 4.0]; // 2 embeddings of dim 2
        let indices = vec![0, 0, 1, 1];
        let result = embedding_lookup(&table, &indices, 2).unwrap();
        let expected = vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0];
        assert_vec_close(&result, &expected, "embedding repeated");
    }

    #[test]
    fn embedding_lookup_out_of_bounds_errors() {
        let table = vec![1.0, 2.0]; // 1 embedding of dim 2
        let indices = vec![5]; // out of bounds
        assert!(embedding_lookup(&table, &indices, 2).is_err());
    }
}

// ===========================================================================
// Linear projection tests
// ===========================================================================

mod linear_tests {
    use super::*;

    #[test]
    fn linear_identity() {
        // y = x * I (identity weight matrix, no bias).
        let config = LinearConfig::new(1, 4, 4).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        #[rustfmt::skip]
        let weight = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let mut output = vec![0.0; 4];
        linear_cpu(&x, &weight, None, &mut output, &config).unwrap();
        assert_vec_close(&output, &x, "linear identity");
    }

    #[test]
    fn linear_with_bias() {
        let config = LinearConfig::new(1, 2, 2).unwrap().with_bias(true);
        let x = vec![1.0, 2.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let bias = vec![10.0, 20.0];
        let mut output = vec![0.0; 2];
        linear_cpu(&x, &weight, Some(&bias), &mut output, &config).unwrap();
        assert_vec_close(&output, &[11.0, 22.0], "linear with bias");
    }

    #[test]
    fn linear_batch() {
        let config = LinearConfig::new(2, 3, 2).unwrap();
        let x = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // batch=2
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let mut output = vec![0.0; 4]; // 2x2
        linear_cpu(&x, &weight, None, &mut output, &config).unwrap();
        // row0: [1,0,0] · [1,2,3] = 1;  [1,0,0] · [4,5,6] = 4
        // row1: [0,1,0] · [1,2,3] = 2;  [0,1,0] · [4,5,6] = 5
        assert_vec_close(&output, &[1.0, 4.0, 2.0, 5.0], "linear batch");
    }

    #[test]
    fn linear_zeros_input() {
        let config = LinearConfig::new(1, 4, 4).unwrap();
        let x = vec![0.0; 4];
        let weight = vec![1.0; 16];
        let mut output = vec![0.0; 4];
        linear_cpu(&x, &weight, None, &mut output, &config).unwrap();
        assert!(output.iter().all(|&v| v == 0.0), "zero input → zero output");
    }

    #[test]
    fn linear_dimension_mismatch_errors() {
        let config = LinearConfig::new(1, 4, 4).unwrap();
        let x = vec![1.0; 2]; // too short
        let weight = vec![1.0; 16];
        let mut output = vec![0.0; 4];
        assert!(linear_cpu(&x, &weight, None, &mut output, &config).is_err());
    }
}

// ===========================================================================
// Reduction operation tests
// ===========================================================================

mod reduction_tests {
    use super::*;

    // ── Sum ────────────────────────────────────────────────────────

    #[test]
    fn sum_basic_sizes() {
        for &n in &[4, 8, 16, 32] {
            let data: Vec<f32> = (1..=n as u32).map(|x| x as f32).collect();
            let expected = (n * (n + 1)) as f32 / 2.0;
            let result = ReductionKernel::sum(&data).unwrap();
            assert_close(result, expected, &format!("sum n={n}"));
        }
    }

    #[test]
    fn sum_negative_values() {
        let data = vec![-1.0, -2.0, -3.0, -4.0];
        let result = ReductionKernel::sum(&data).unwrap();
        assert_close(result, -10.0, "sum negative");
    }

    #[test]
    fn sum_mixed_cancellation() {
        let data = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
        let result = ReductionKernel::sum(&data).unwrap();
        assert_close(result, 0.0, "sum cancellation");
    }

    // ── Mean ───────────────────────────────────────────────────────

    #[test]
    fn mean_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = ReductionKernel::mean(&data).unwrap();
        assert_close(result, 2.5, "mean basic");
    }

    #[test]
    fn mean_uniform() {
        let data = vec![5.0; 16];
        let result = ReductionKernel::mean(&data).unwrap();
        assert_close(result, 5.0, "mean uniform");
    }

    // ── Max ────────────────────────────────────────────────────────

    #[test]
    fn max_basic() {
        let data = vec![1.0, 4.0, 2.0, 3.0];
        let result = ReductionKernel::max(&data).unwrap();
        assert_close(result.value, 4.0, "max value");
        assert_eq!(result.index, 1, "max index");
    }

    #[test]
    fn max_all_negative() {
        let data = vec![-10.0, -5.0, -20.0, -1.0];
        let result = ReductionKernel::max(&data).unwrap();
        assert_close(result.value, -1.0, "max negative");
        assert_eq!(result.index, 3);
    }

    #[test]
    fn max_single_element() {
        let result = ReductionKernel::max(&[42.0]).unwrap();
        assert_close(result.value, 42.0, "max single");
        assert_eq!(result.index, 0);
    }

    // ── Min ────────────────────────────────────────────────────────

    #[test]
    fn min_basic() {
        let data = vec![3.0, 1.0, 4.0, 2.0];
        let result = ReductionKernel::min(&data).unwrap();
        assert_close(result.value, 1.0, "min value");
        assert_eq!(result.index, 1, "min index");
    }

    // ── L1 norm ────────────────────────────────────────────────────

    #[test]
    fn l1_norm_basic() {
        let data = vec![1.0, -2.0, 3.0, -4.0];
        let result = ReductionKernel::l1_norm(&data).unwrap();
        assert_close(result, 10.0, "l1_norm");
    }

    // ── L2 norm ────────────────────────────────────────────────────

    #[test]
    fn l2_norm_basic() {
        let data = vec![3.0, 4.0];
        let result = ReductionKernel::l2_norm(&data).unwrap();
        assert_close(result, 5.0, "l2_norm");
    }

    #[test]
    fn l2_norm_unit_vector() {
        let data = vec![1.0, 0.0, 0.0, 0.0];
        let result = ReductionKernel::l2_norm(&data).unwrap();
        assert_close(result, 1.0, "l2_norm unit");
    }

    // ── Product ────────────────────────────────────────────────────

    #[test]
    fn product_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = ReductionKernel::product(&data).unwrap();
        assert_close(result, 24.0, "product");
    }

    #[test]
    fn product_with_zero() {
        let data = vec![1.0, 0.0, 3.0, 4.0];
        let result = ReductionKernel::product(&data).unwrap();
        assert_close(result, 0.0, "product with zero");
    }

    // ── Error cases ────────────────────────────────────────────────

    #[test]
    fn reduction_empty_errors() {
        assert!(ReductionKernel::sum(&[]).is_err());
        assert!(ReductionKernel::mean(&[]).is_err());
        assert!(ReductionKernel::max(&[]).is_err());
        assert!(ReductionKernel::min(&[]).is_err());
        assert!(ReductionKernel::l1_norm(&[]).is_err());
        assert!(ReductionKernel::l2_norm(&[]).is_err());
        assert!(ReductionKernel::product(&[]).is_err());
    }
}

// ===========================================================================
// Residual connection tests
// ===========================================================================

mod residual_tests {
    use super::*;

    #[test]
    fn add_residual_basic() {
        for &n in &[4, 8, 16, 32] {
            let mut output: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let residual: Vec<f32> = (0..n).map(|i| (i * 10) as f32).collect();
            add_residual(&mut output, &residual).unwrap();
            for i in 0..n {
                assert_close(output[i], (i + i * 10) as f32, &format!("residual n={n} [{i}]"));
            }
        }
    }

    #[test]
    fn add_residual_zeros() {
        let mut output = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.0; 4];
        add_residual(&mut output, &residual).unwrap();
        assert_vec_close(&output, &[1.0, 2.0, 3.0, 4.0], "residual zeros");
    }

    #[test]
    fn add_residual_scaled_basic() {
        let mut output = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![10.0, 20.0, 30.0, 40.0];
        add_residual_scaled(&mut output, &residual, 0.5).unwrap();
        assert_vec_close(&output, &[6.0, 12.0, 18.0, 24.0], "residual scaled");
    }

    #[test]
    fn add_residual_length_mismatch_errors() {
        let mut output = vec![1.0, 2.0];
        let residual = vec![1.0, 2.0, 3.0];
        assert!(add_residual(&mut output, &residual).is_err());
    }
}

// ===========================================================================
// Gating tests
// ===========================================================================

mod gating_tests {
    use super::*;

    fn ref_silu_local(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn ref_gelu_approx_local(x: f32) -> f32 {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        const COEFF: f32 = 0.044_715;
        let x3 = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        0.5 * x * (1.0 + inner.tanh())
    }

    #[test]
    fn swiglu_basic() {
        let gate = vec![1.0, -1.0, 0.0, 2.0];
        let up = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];
        swiglu(&gate, &up, &mut output).unwrap();
        let expected: Vec<f32> = gate.iter().map(|&g| ref_silu_local(g)).collect();
        assert_vec_close(&output, &expected, "swiglu");
    }

    #[test]
    fn geglu_basic() {
        let gate = vec![1.0, -1.0, 0.0, 2.0];
        let up = vec![2.0, 2.0, 2.0, 2.0];
        let mut output = vec![0.0; 4];
        geglu(&gate, &up, &mut output).unwrap();
        let expected: Vec<f32> = gate.iter().map(|&g| ref_gelu_approx_local(g) * 2.0).collect();
        assert_vec_close(&output, &expected, "geglu");
    }

    #[test]
    fn reglu_basic() {
        let gate = vec![1.0, -1.0, 0.0, 2.0];
        let up = vec![3.0, 3.0, 3.0, 3.0];
        let mut output = vec![0.0; 4];
        reglu(&gate, &up, &mut output).unwrap();
        let expected: Vec<f32> = gate.iter().map(|&g| g.max(0.0) * 3.0).collect();
        assert_vec_close(&output, &expected, "reglu");
    }

    #[test]
    fn apply_gating_dispatch() {
        let gate = vec![1.0, 2.0];
        let up = vec![1.0, 1.0];
        let mut out = vec![0.0; 2];
        apply_gating(GatingType::SwiGLU, &gate, &up, &mut out).unwrap();
        assert!(out[0] > 0.0 && out[1] > 0.0);

        out.fill(0.0);
        apply_gating(GatingType::GeGLU, &gate, &up, &mut out).unwrap();
        assert!(out[0] > 0.0 && out[1] > 0.0);

        out.fill(0.0);
        apply_gating(GatingType::ReGLU, &gate, &up, &mut out).unwrap();
        assert!(out[0] > 0.0 && out[1] > 0.0);
    }
}

// ===========================================================================
// Cross-size parity: same math at every width
// ===========================================================================

mod cross_size_parity {
    use super::*;

    /// For each activation, verify that applying it to the first 4 elements of
    /// a longer buffer produces the same result as applying it to a 4-element
    /// buffer directly.
    #[test]
    fn activation_prefix_parity() {
        let full = linspace(-3.0, 3.0, 32);
        let prefix = &full[..4];

        let silu_full = silu_vec(&full);
        let silu_short = silu_vec(prefix);
        assert_vec_close(&silu_full[..4], &silu_short, "silu prefix parity");

        let gelu_full = gelu_vec(&full);
        let gelu_short = gelu_vec(prefix);
        assert_vec_close(&gelu_full[..4], &gelu_short, "gelu prefix parity");

        let gelu_a_full = gelu_approx_vec(&full);
        let gelu_a_short = gelu_approx_vec(prefix);
        assert_vec_close(&gelu_a_full[..4], &gelu_a_short, "gelu_approx prefix parity");
    }

    /// Verify reductions give same answer regardless of evaluation order.
    #[test]
    fn reduction_order_independence() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let mut reversed = data.clone();
        reversed.reverse();

        let sum_fwd = ReductionKernel::sum(&data).unwrap();
        let sum_rev = ReductionKernel::sum(&reversed).unwrap();
        assert_close(sum_fwd, sum_rev, "sum order independence");

        let mean_fwd = ReductionKernel::mean(&data).unwrap();
        let mean_rev = ReductionKernel::mean(&reversed).unwrap();
        assert_close(mean_fwd, mean_rev, "mean order independence");
    }
}
