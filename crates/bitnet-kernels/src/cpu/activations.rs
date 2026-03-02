//! CPU activation function kernels.
//!
//! Re-exports the full scalar/vector/dispatch API from `bitnet-cpu-activations`,
//! and adds validated two-buffer kernel variants with `ActivationError` handling.
//!
//! The two-buffer functions use the `_to` suffix (e.g. `relu_to`) to write into
//! a caller-provided output slice, avoiding allocation and enabling pre-allocated
//! buffer reuse in hot inference loops.

pub use bitnet_cpu_activations::*;

use std::fmt;

// ── Error type ──────────────────────────────────────────────────────

/// Errors from activation kernel operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationError {
    /// Input slice was empty.
    EmptyInput,
    /// Output slice length does not match input.
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for ActivationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty input"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for ActivationError {}

// ── Validation helpers ──────────────────────────────────────────────

fn validate_buffers(input: &[f32], output: &[f32]) -> Result<(), ActivationError> {
    if input.is_empty() {
        return Err(ActivationError::EmptyInput);
    }
    if output.len() != input.len() {
        return Err(ActivationError::DimensionMismatch {
            expected: input.len(),
            got: output.len(),
        });
    }
    Ok(())
}

fn validate_inplace(data: &[f32]) -> Result<(), ActivationError> {
    if data.is_empty() {
        return Err(ActivationError::EmptyInput);
    }
    Ok(())
}

// ── Two-buffer kernel functions ─────────────────────────────────────

/// ReLU: max(0, x), writing into `output`.
pub fn relu_to(input: &[f32], output: &mut [f32]) -> Result<(), ActivationError> {
    validate_buffers(input, output)?;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = bitnet_cpu_activations::relu(x);
    }
    Ok(())
}

/// Exact GELU (erf-based), writing into `output`.
pub fn gelu_to(input: &[f32], output: &mut [f32]) -> Result<(), ActivationError> {
    validate_buffers(input, output)?;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = bitnet_cpu_activations::gelu(x);
    }
    Ok(())
}

/// Fast GELU approximation (tanh-based), writing into `output`.
pub fn gelu_fast_to(input: &[f32], output: &mut [f32]) -> Result<(), ActivationError> {
    validate_buffers(input, output)?;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = bitnet_cpu_activations::gelu_tanh(x);
    }
    Ok(())
}

/// SiLU (Swish-1): x * sigmoid(x), writing into `output`.
pub fn silu_to(input: &[f32], output: &mut [f32]) -> Result<(), ActivationError> {
    validate_buffers(input, output)?;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = bitnet_cpu_activations::silu(x);
    }
    Ok(())
}

/// Sigmoid: 1/(1+exp(-x)), writing into `output`.
pub fn sigmoid_to(input: &[f32], output: &mut [f32]) -> Result<(), ActivationError> {
    validate_buffers(input, output)?;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = bitnet_cpu_activations::sigmoid(x);
    }
    Ok(())
}

/// Tanh activation, writing into `output`.
pub fn tanh_activation_to(input: &[f32], output: &mut [f32]) -> Result<(), ActivationError> {
    validate_buffers(input, output)?;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = bitnet_cpu_activations::tanh_act(x);
    }
    Ok(())
}

/// Softplus: ln(1+exp(x)), writing into `output`.
pub fn softplus_to(input: &[f32], output: &mut [f32]) -> Result<(), ActivationError> {
    validate_buffers(input, output)?;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = bitnet_cpu_activations::softplus(x);
    }
    Ok(())
}

/// Mish: x * tanh(softplus(x)), writing into `output`.
pub fn mish_to(input: &[f32], output: &mut [f32]) -> Result<(), ActivationError> {
    validate_buffers(input, output)?;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = bitnet_cpu_activations::mish(x);
    }
    Ok(())
}

/// Quick GELU: x * sigmoid(1.702 * x), writing into `output`.
pub fn quick_gelu_to(input: &[f32], output: &mut [f32]) -> Result<(), ActivationError> {
    validate_buffers(input, output)?;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = bitnet_cpu_activations::quick_gelu(x);
    }
    Ok(())
}

// ── Two-buffer dispatch ─────────────────────────────────────────────

/// Apply an activation by type, reading from `input` and writing to `output`.
pub fn apply_activation_to(
    input: &[f32],
    output: &mut [f32],
    activation: ActivationType,
) -> Result<(), ActivationError> {
    validate_buffers(input, output)?;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = match activation {
            ActivationType::ReLU => bitnet_cpu_activations::relu(x),
            ActivationType::LeakyReLU(a) => bitnet_cpu_activations::leaky_relu(x, a),
            ActivationType::GELU => bitnet_cpu_activations::gelu(x),
            ActivationType::GELUTanh => bitnet_cpu_activations::gelu_tanh(x),
            ActivationType::SiLU => bitnet_cpu_activations::silu(x),
            ActivationType::Swish(b) => bitnet_cpu_activations::swish(x, b),
            ActivationType::Sigmoid => bitnet_cpu_activations::sigmoid(x),
            ActivationType::Tanh => bitnet_cpu_activations::tanh_act(x),
            ActivationType::HardSigmoid => bitnet_cpu_activations::hard_sigmoid(x),
            ActivationType::HardSwish => bitnet_cpu_activations::hard_swish(x),
            ActivationType::Mish => bitnet_cpu_activations::mish(x),
            ActivationType::Softplus => bitnet_cpu_activations::softplus(x),
            ActivationType::ELU(a) => bitnet_cpu_activations::elu(x, a),
            ActivationType::SELU => bitnet_cpu_activations::selu(x),
            ActivationType::QuickGELU => bitnet_cpu_activations::quick_gelu(x),
        };
    }
    Ok(())
}

/// Apply an activation in-place with validation.
pub fn apply_activation_inplace(
    data: &mut [f32],
    activation: ActivationType,
) -> Result<(), ActivationError> {
    validate_inplace(data)?;
    for x in data.iter_mut() {
        *x = match activation {
            ActivationType::ReLU => bitnet_cpu_activations::relu(*x),
            ActivationType::LeakyReLU(a) => bitnet_cpu_activations::leaky_relu(*x, a),
            ActivationType::GELU => bitnet_cpu_activations::gelu(*x),
            ActivationType::GELUTanh => bitnet_cpu_activations::gelu_tanh(*x),
            ActivationType::SiLU => bitnet_cpu_activations::silu(*x),
            ActivationType::Swish(b) => bitnet_cpu_activations::swish(*x, b),
            ActivationType::Sigmoid => bitnet_cpu_activations::sigmoid(*x),
            ActivationType::Tanh => bitnet_cpu_activations::tanh_act(*x),
            ActivationType::HardSigmoid => bitnet_cpu_activations::hard_sigmoid(*x),
            ActivationType::HardSwish => bitnet_cpu_activations::hard_swish(*x),
            ActivationType::Mish => bitnet_cpu_activations::mish(*x),
            ActivationType::Softplus => bitnet_cpu_activations::softplus(*x),
            ActivationType::ELU(a) => bitnet_cpu_activations::elu(*x, a),
            ActivationType::SELU => bitnet_cpu_activations::selu(*x),
            ActivationType::QuickGELU => bitnet_cpu_activations::quick_gelu(*x),
        };
    }
    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod kernel_tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    // ── relu_to ──

    #[test]
    fn relu_to_known_values() {
        let input = [-2.0_f32, -0.5, 0.0, 0.5, 2.0];
        let mut out = [0.0; 5];
        relu_to(&input, &mut out).unwrap();
        assert_eq!(out, [0.0, 0.0, 0.0, 0.5, 2.0]);
    }

    #[test]
    fn relu_to_large_positive() {
        let input = [1e6_f32];
        let mut out = [0.0; 1];
        relu_to(&input, &mut out).unwrap();
        assert_eq!(out[0], 1e6);
    }

    #[test]
    fn relu_to_large_negative() {
        let input = [-1e6_f32];
        let mut out = [0.0; 1];
        relu_to(&input, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
    }

    // ── gelu_to ──

    #[test]
    fn gelu_to_at_zero() {
        let input = [0.0_f32];
        let mut out = [999.0; 1];
        gelu_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 0.0));
    }

    #[test]
    fn gelu_to_positive_approx_identity() {
        let input = [5.0_f32];
        let mut out = [0.0; 1];
        gelu_to(&input, &mut out).unwrap();
        assert!(out[0] > 4.99);
    }

    #[test]
    fn gelu_to_negative_region() {
        let input = [-0.5_f32];
        let mut out = [0.0; 1];
        gelu_to(&input, &mut out).unwrap();
        assert!(out[0] < 0.0);
    }

    // ── gelu_fast_to ──

    #[test]
    fn gelu_fast_to_close_to_exact() {
        let input = [-2.0_f32, -1.0, 0.0, 1.0, 2.0];
        let mut exact = [0.0; 5];
        let mut fast = [0.0; 5];
        gelu_to(&input, &mut exact).unwrap();
        gelu_fast_to(&input, &mut fast).unwrap();
        for (e, f) in exact.iter().zip(fast.iter()) {
            assert!((e - f).abs() < 0.02, "exact={e}, fast={f}");
        }
    }

    // ── silu_to ──

    #[test]
    fn silu_to_at_zero() {
        let input = [0.0_f32];
        let mut out = [999.0; 1];
        silu_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 0.0));
    }

    #[test]
    fn silu_to_positive_approx_identity() {
        let input = [10.0_f32];
        let mut out = [0.0; 1];
        silu_to(&input, &mut out).unwrap();
        assert!(out[0] > 9.99);
    }

    // ── sigmoid_to ──

    #[test]
    fn sigmoid_to_at_zero_is_half() {
        let input = [0.0_f32];
        let mut out = [0.0; 1];
        sigmoid_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 0.5));
    }

    #[test]
    fn sigmoid_to_bounds() {
        let input = [-100.0_f32, 100.0];
        let mut out = [0.0; 2];
        sigmoid_to(&input, &mut out).unwrap();
        assert!(out[0] >= 0.0 && out[0] <= 1.0);
        assert!(out[1] >= 0.0 && out[1] <= 1.0);
        assert!(out[0] < 0.001);
        assert!(out[1] > 0.999);
    }

    #[test]
    fn sigmoid_to_range_always_01() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.2).collect();
        let mut out = vec![0.0; input.len()];
        sigmoid_to(&input, &mut out).unwrap();
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "sigmoid out of [0,1]: {v}");
        }
    }

    // ── tanh_activation_to ──

    #[test]
    fn tanh_activation_to_at_zero() {
        let input = [0.0_f32];
        let mut out = [999.0; 1];
        tanh_activation_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 0.0));
    }

    #[test]
    fn tanh_activation_to_bounds() {
        let input = [-100.0_f32, 100.0];
        let mut out = [0.0; 2];
        tanh_activation_to(&input, &mut out).unwrap();
        assert!(out[0] >= -1.0 && out[0] <= 1.0);
        assert!(out[1] >= -1.0 && out[1] <= 1.0);
        assert!(out[0] < -0.999);
        assert!(out[1] > 0.999);
    }

    #[test]
    fn tanh_activation_to_range_always_neg1_1() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.2).collect();
        let mut out = vec![0.0; input.len()];
        tanh_activation_to(&input, &mut out).unwrap();
        for &v in &out {
            assert!((-1.0..=1.0).contains(&v), "tanh out of [-1,1]: {v}");
        }
    }

    // ── softplus_to ──

    #[test]
    fn softplus_to_at_zero_is_ln2() {
        let input = [0.0_f32];
        let mut out = [0.0; 1];
        softplus_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 2.0_f32.ln()));
    }

    #[test]
    fn softplus_to_large_is_identity() {
        let input = [50.0_f32];
        let mut out = [0.0; 1];
        softplus_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 50.0));
    }

    #[test]
    fn softplus_to_always_positive() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.2).collect();
        let mut out = vec![0.0; input.len()];
        softplus_to(&input, &mut out).unwrap();
        for &v in &out {
            assert!(v >= 0.0, "softplus should be non-negative, got {v}");
        }
    }

    // ── mish_to ──

    #[test]
    fn mish_to_at_zero() {
        let input = [0.0_f32];
        let mut out = [999.0; 1];
        mish_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 0.0));
    }

    #[test]
    fn mish_to_positive_approx_identity() {
        let input = [10.0_f32];
        let mut out = [0.0; 1];
        mish_to(&input, &mut out).unwrap();
        assert!(out[0] > 9.99);
    }

    // ── quick_gelu_to ──

    #[test]
    fn quick_gelu_to_at_zero() {
        let input = [0.0_f32];
        let mut out = [999.0; 1];
        quick_gelu_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 0.0));
    }

    #[test]
    fn quick_gelu_to_close_to_gelu() {
        let input = [-1.0_f32, 0.0, 1.0, 2.0];
        let mut qg = [0.0; 4];
        let mut ge = [0.0; 4];
        quick_gelu_to(&input, &mut qg).unwrap();
        gelu_to(&input, &mut ge).unwrap();
        for (q, g) in qg.iter().zip(ge.iter()) {
            assert!((q - g).abs() < 0.05, "quick_gelu={q}, gelu={g}");
        }
    }

    // ── Error cases ──

    #[test]
    fn error_empty_input() {
        let input: &[f32] = &[];
        let mut out: Vec<f32> = vec![];
        assert!(matches!(relu_to(input, &mut out), Err(ActivationError::EmptyInput)));
    }

    #[test]
    fn error_dimension_mismatch_too_short() {
        let input = [1.0_f32, 2.0, 3.0];
        let mut out = [0.0; 2];
        let err = relu_to(&input, &mut out).unwrap_err();
        assert!(matches!(err, ActivationError::DimensionMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn error_dimension_mismatch_too_long() {
        let input = [1.0_f32, 2.0];
        let mut out = [0.0; 5];
        let err = sigmoid_to(&input, &mut out).unwrap_err();
        assert!(matches!(err, ActivationError::DimensionMismatch { expected: 2, got: 5 }));
    }

    #[test]
    fn error_empty_inplace() {
        let mut data: Vec<f32> = vec![];
        assert!(matches!(
            apply_activation_inplace(&mut data, ActivationType::ReLU),
            Err(ActivationError::EmptyInput)
        ));
    }

    #[test]
    fn error_display_empty() {
        let e = ActivationError::EmptyInput;
        assert_eq!(e.to_string(), "empty input");
    }

    #[test]
    fn error_display_mismatch() {
        let e = ActivationError::DimensionMismatch { expected: 10, got: 5 };
        assert_eq!(e.to_string(), "dimension mismatch: expected 10, got 5");
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(ActivationError::EmptyInput);
        assert_eq!(e.to_string(), "empty input");
    }

    // ── Monotonicity ──

    #[test]
    fn monotonicity_relu_to() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let mut out = vec![0.0; input.len()];
        relu_to(&input, &mut out).unwrap();
        for i in 1..out.len() {
            assert!(out[i] >= out[i - 1] - TOL, "relu not monotonic at i={i}");
        }
    }

    #[test]
    fn monotonicity_sigmoid_to() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let mut out = vec![0.0; input.len()];
        sigmoid_to(&input, &mut out).unwrap();
        for i in 1..out.len() {
            assert!(out[i] >= out[i - 1] - TOL, "sigmoid not monotonic at i={i}");
        }
    }

    #[test]
    fn monotonicity_softplus_to() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let mut out = vec![0.0; input.len()];
        softplus_to(&input, &mut out).unwrap();
        for i in 1..out.len() {
            assert!(out[i] >= out[i - 1] - TOL, "softplus not monotonic at i={i}");
        }
    }

    #[test]
    fn monotonicity_tanh_to() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let mut out = vec![0.0; input.len()];
        tanh_activation_to(&input, &mut out).unwrap();
        for i in 1..out.len() {
            assert!(out[i] >= out[i - 1] - TOL, "tanh not monotonic at i={i}");
        }
    }

    // ── apply_activation_to dispatch ──

    #[test]
    fn dispatch_relu() {
        let input = [-1.0_f32, 0.0, 1.0];
        let mut out = [0.0; 3];
        let mut expected = [0.0; 3];
        relu_to(&input, &mut expected).unwrap();
        apply_activation_to(&input, &mut out, ActivationType::ReLU).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn dispatch_all_types_match_scalar() {
        let input = [-3.0_f32, -1.5, 0.0, 1.5, 3.0];
        let types = [
            ActivationType::ReLU,
            ActivationType::GELU,
            ActivationType::SiLU,
            ActivationType::Sigmoid,
            ActivationType::Tanh,
            ActivationType::Softplus,
            ActivationType::Mish,
            ActivationType::QuickGELU,
            ActivationType::HardSigmoid,
            ActivationType::HardSwish,
            ActivationType::SELU,
            ActivationType::ELU(1.0),
            ActivationType::LeakyReLU(0.01),
            ActivationType::Swish(1.0),
            ActivationType::GELUTanh,
        ];
        for act in types {
            let mut out = [0.0; 5];
            apply_activation_to(&input, &mut out, act).unwrap();
            let scalar = activate(&input, act);
            for (i, (&o, &s)) in out.iter().zip(scalar.iter()).enumerate() {
                assert!(approx(o, s), "{act:?} mismatch at {i}: kernel={o}, scalar={s}");
            }
        }
    }

    #[test]
    fn dispatch_error_propagation() {
        let input: &[f32] = &[];
        let mut out: Vec<f32> = vec![];
        assert!(matches!(
            apply_activation_to(input, &mut out, ActivationType::ReLU),
            Err(ActivationError::EmptyInput)
        ));
    }

    // ── apply_activation_inplace ──

    #[test]
    fn inplace_matches_two_buffer() {
        let input = [-2.0_f32, -1.0, 0.0, 1.0, 2.0];
        let types = [
            ActivationType::ReLU,
            ActivationType::GELU,
            ActivationType::SiLU,
            ActivationType::Sigmoid,
            ActivationType::Tanh,
            ActivationType::Mish,
            ActivationType::QuickGELU,
        ];
        for act in types {
            let mut buf = input;
            let mut expected = [0.0; 5];
            apply_activation_to(&input, &mut expected, act).unwrap();
            apply_activation_inplace(&mut buf, act).unwrap();
            for (i, (&b, &e)) in buf.iter().zip(expected.iter()).enumerate() {
                assert!(approx(b, e), "{act:?} inplace mismatch at {i}: {b} vs {e}");
            }
        }
    }

    // ── Edge-case: very large / very negative ──

    #[test]
    fn large_input_no_nan() {
        let input = [1e6_f32, -1e6, 1e3, -1e3];
        let mut out = [0.0; 4];
        for f in [
            relu_to as fn(&[f32], &mut [f32]) -> Result<(), ActivationError>,
            gelu_to,
            silu_to,
            sigmoid_to,
            tanh_activation_to,
            softplus_to,
            mish_to,
            quick_gelu_to,
            gelu_fast_to,
        ] {
            f(&input, &mut out).unwrap();
            for (i, &v) in out.iter().enumerate() {
                assert!(!v.is_nan(), "NaN at index {i} for input {}", input[i]);
            }
        }
    }

    // ── Zero input ──

    #[test]
    fn all_activations_at_zero() {
        let input = [0.0_f32];
        let mut out = [999.0; 1];

        relu_to(&input, &mut out).unwrap();
        assert_eq!(out[0], 0.0);

        silu_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 0.0));

        sigmoid_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 0.5));

        tanh_activation_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 0.0));

        softplus_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 2.0_f32.ln()));

        mish_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 0.0));

        quick_gelu_to(&input, &mut out).unwrap();
        assert!(approx(out[0], 0.0));
    }
}
