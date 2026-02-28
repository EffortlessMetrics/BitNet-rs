//! Activation function kernels for GPU HAL.
//!
//! Provides a trait-based abstraction over common activation functions used in
//! neural-network inference (GELU, `SiLU`, `ReLU`, Swish, sigmoid, tanh,
//! `LeakyReLU`, ELU) together with a numerically-stable CPU reference
//! implementation, batch/in-place helpers, and fused activation+bias patterns.

use std::f32::consts::{FRAC_2_SQRT_PI, SQRT_2};

use crate::HalError;

// ── Configuration ─────────────────────────────────────────────────────────

/// Tunable parameters for activation functions.
#[derive(Debug, Clone, Copy)]
pub struct ActivationConfig {
    /// Slope for the negative region of `LeakyReLU` (default 0.01).
    pub leaky_relu_alpha: f32,
    /// Alpha for ELU (default 1.0).
    pub elu_alpha: f32,
    /// Epsilon added to denominators for numerical stability.
    pub epsilon: f32,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self { leaky_relu_alpha: 0.01, elu_alpha: 1.0, epsilon: 1e-8 }
    }
}

// ── Trait ──────────────────────────────────────────────────────────────────

/// Device-agnostic activation function kernel interface.
///
/// Implementations operate element-wise on `f32` slices. Every method
/// writes results into `output` and returns the number of elements
/// written.
pub trait ActivationKernel {
    /// GELU (Gaussian Error Linear Unit) — the tanh approximation:
    /// `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
    fn gelu(&self, input: &[f32], output: &mut [f32]) -> Result<usize, HalError>;

    /// `SiLU` / Swish-1: `x * σ(x)`.
    fn silu(&self, input: &[f32], output: &mut [f32]) -> Result<usize, HalError>;

    /// `ReLU`: `max(0, x)`.
    fn relu(&self, input: &[f32], output: &mut [f32]) -> Result<usize, HalError>;

    /// Swish: `x * σ(β·x)` with configurable β.
    fn swish(
        &self,
        input: &[f32],
        output: &mut [f32],
        beta: f32,
    ) -> Result<usize, HalError>;

    /// Logistic sigmoid: `1 / (1 + exp(-x))`.
    fn sigmoid(&self, input: &[f32], output: &mut [f32]) -> Result<usize, HalError>;

    /// Hyperbolic tangent activation.
    fn tanh_activation(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<usize, HalError>;

    /// Leaky `ReLU`: `x` if `x ≥ 0`, else `alpha·x`.
    fn leaky_relu(
        &self,
        input: &[f32],
        output: &mut [f32],
        alpha: f32,
    ) -> Result<usize, HalError>;

    /// ELU: `x` if `x ≥ 0`, else `alpha·(exp(x) − 1)`.
    fn elu(
        &self,
        input: &[f32],
        output: &mut [f32],
        alpha: f32,
    ) -> Result<usize, HalError>;
}

// ── CPU reference implementation ──────────────────────────────────────────

/// Numerically-stable CPU reference kernels.
pub struct CpuActivationKernels {
    pub config: ActivationConfig,
}

impl CpuActivationKernels {
    pub const fn new(config: ActivationConfig) -> Self {
        Self { config }
    }
}

impl Default for CpuActivationKernels {
    fn default() -> Self {
        Self::new(ActivationConfig::default())
    }
}

/// Clamp `x` into a range safe for `exp()`.
#[inline]
fn safe_exp(x: f32) -> f32 {
    x.clamp(-88.0, 88.0).exp()
}

/// Numerically-stable sigmoid.
#[inline]
fn stable_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = safe_exp(-x);
        1.0 / (1.0 + e)
    } else {
        let e = safe_exp(x);
        e / (1.0 + e)
    }
}

/// Validate that `output` is at least as large as `input`.
#[inline]
const fn check_len(input: &[f32], output: &[f32]) -> Result<usize, HalError> {
    let n = input.len();
    if output.len() < n {
        return Err(HalError::ShapeMismatch {
            expected: n,
            actual: output.len(),
        });
    }
    Ok(n)
}

impl ActivationKernel for CpuActivationKernels {
    fn gelu(&self, input: &[f32], output: &mut [f32]) -> Result<usize, HalError> {
        let n = check_len(input, output)?;
        let coeff = (FRAC_2_SQRT_PI / SQRT_2).sqrt();
        for i in 0..n {
            let x = input[i];
            let inner = coeff * x.mul_add(x.mul_add(0.044_715 * x, 0.0), x);
            output[i] = 0.5 * x * (1.0 + inner.tanh());
        }
        Ok(n)
    }

    fn silu(&self, input: &[f32], output: &mut [f32]) -> Result<usize, HalError> {
        let n = check_len(input, output)?;
        for i in 0..n {
            output[i] = input[i] * stable_sigmoid(input[i]);
        }
        Ok(n)
    }

    fn relu(&self, input: &[f32], output: &mut [f32]) -> Result<usize, HalError> {
        let n = check_len(input, output)?;
        for i in 0..n {
            output[i] = input[i].max(0.0);
        }
        Ok(n)
    }

    fn swish(
        &self,
        input: &[f32],
        output: &mut [f32],
        beta: f32,
    ) -> Result<usize, HalError> {
        let n = check_len(input, output)?;
        for i in 0..n {
            output[i] = input[i] * stable_sigmoid(beta * input[i]);
        }
        Ok(n)
    }

    fn sigmoid(&self, input: &[f32], output: &mut [f32]) -> Result<usize, HalError> {
        let n = check_len(input, output)?;
        for i in 0..n {
            output[i] = stable_sigmoid(input[i]);
        }
        Ok(n)
    }

    fn tanh_activation(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<usize, HalError> {
        let n = check_len(input, output)?;
        for i in 0..n {
            output[i] = input[i].tanh();
        }
        Ok(n)
    }

    fn leaky_relu(
        &self,
        input: &[f32],
        output: &mut [f32],
        alpha: f32,
    ) -> Result<usize, HalError> {
        let n = check_len(input, output)?;
        for i in 0..n {
            output[i] = if input[i] >= 0.0 { input[i] } else { alpha * input[i] };
        }
        Ok(n)
    }

    fn elu(
        &self,
        input: &[f32],
        output: &mut [f32],
        alpha: f32,
    ) -> Result<usize, HalError> {
        let n = check_len(input, output)?;
        for i in 0..n {
            output[i] = if input[i] >= 0.0 {
                input[i]
            } else {
                alpha * (safe_exp(input[i]) - 1.0)
            };
        }
        Ok(n)
    }
}

// ── In-place helpers ──────────────────────────────────────────────────────

/// Apply an activation function in-place.
pub fn activate_inplace(
    kernel: &dyn ActivationKernel,
    kind: ActivationKind,
    data: &mut [f32],
    config: &ActivationConfig,
) -> Result<usize, HalError> {
    // Use a temporary copy so the trait can read from input while writing
    // to the same buffer.
    let tmp = data.to_vec();
    match kind {
        ActivationKind::Gelu => kernel.gelu(&tmp, data),
        ActivationKind::Silu => kernel.silu(&tmp, data),
        ActivationKind::Relu => kernel.relu(&tmp, data),
        ActivationKind::Swish => kernel.swish(&tmp, data, 1.0),
        ActivationKind::Sigmoid => kernel.sigmoid(&tmp, data),
        ActivationKind::Tanh => kernel.tanh_activation(&tmp, data),
        ActivationKind::LeakyRelu => {
            kernel.leaky_relu(&tmp, data, config.leaky_relu_alpha)
        }
        ActivationKind::Elu => kernel.elu(&tmp, data, config.elu_alpha),
    }
}

/// Selector enum for dynamic activation dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    Gelu,
    Silu,
    Relu,
    Swish,
    Sigmoid,
    Tanh,
    LeakyRelu,
    Elu,
}

// ── Batch processing ──────────────────────────────────────────────────────

/// Apply an activation to `input` in fixed-size chunks, writing results to
/// `output`. Returns total elements written.
pub fn activate_batched(
    kernel: &dyn ActivationKernel,
    kind: ActivationKind,
    input: &[f32],
    output: &mut [f32],
    config: &ActivationConfig,
    batch_size: usize,
) -> Result<usize, HalError> {
    if input.is_empty() {
        return Ok(0);
    }
    if batch_size == 0 {
        return Err(HalError::ShapeMismatch { expected: 1, actual: 0 });
    }
    let n = check_len(input, output)?;
    let mut written = 0;
    for start in (0..n).step_by(batch_size) {
        let end = (start + batch_size).min(n);
        let chunk_in = &input[start..end];
        let chunk_out = &mut output[start..end];
        let w = match kind {
            ActivationKind::Gelu => kernel.gelu(chunk_in, chunk_out)?,
            ActivationKind::Silu => kernel.silu(chunk_in, chunk_out)?,
            ActivationKind::Relu => kernel.relu(chunk_in, chunk_out)?,
            ActivationKind::Swish => kernel.swish(chunk_in, chunk_out, 1.0)?,
            ActivationKind::Sigmoid => kernel.sigmoid(chunk_in, chunk_out)?,
            ActivationKind::Tanh => {
                kernel.tanh_activation(chunk_in, chunk_out)?
            }
            ActivationKind::LeakyRelu => {
                kernel.leaky_relu(chunk_in, chunk_out, config.leaky_relu_alpha)?
            }
            ActivationKind::Elu => {
                kernel.elu(chunk_in, chunk_out, config.elu_alpha)?
            }
        };
        written += w;
    }
    Ok(written)
}

// ── Fused activation + bias ───────────────────────────────────────────────

/// Compute `activation(input + bias)` in one pass, writing to `output`.
///
/// `bias` is broadcast: if `bias.len() < input.len()` it is tiled. The
/// bias length must evenly divide the input length.
pub fn fused_activation_bias(
    kernel: &dyn ActivationKernel,
    kind: ActivationKind,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
    config: &ActivationConfig,
) -> Result<usize, HalError> {
    if bias.is_empty() {
        return Err(HalError::EmptyInput);
    }
    let n = check_len(input, output)?;
    if n % bias.len() != 0 {
        return Err(HalError::ShapeMismatch {
            expected: n,
            actual: bias.len(),
        });
    }
    // Fuse: add bias first, then activate.
    let biased: Vec<f32> =
        input.iter().enumerate().map(|(i, &v)| v + bias[i % bias.len()]).collect();
    match kind {
        ActivationKind::Gelu => kernel.gelu(&biased, output),
        ActivationKind::Silu => kernel.silu(&biased, output),
        ActivationKind::Relu => kernel.relu(&biased, output),
        ActivationKind::Swish => kernel.swish(&biased, output, 1.0),
        ActivationKind::Sigmoid => kernel.sigmoid(&biased, output),
        ActivationKind::Tanh => kernel.tanh_activation(&biased, output),
        ActivationKind::LeakyRelu => {
            kernel.leaky_relu(&biased, output, config.leaky_relu_alpha)
        }
        ActivationKind::Elu => kernel.elu(&biased, output, config.elu_alpha),
    }
}

// ── Benchmark helper ──────────────────────────────────────────────────────

/// Lightweight benchmark record for comparing activation throughput.
#[derive(Debug, Clone)]
pub struct ActivationBenchmark {
    pub kind: ActivationKind,
    pub elements: usize,
    pub iterations: u32,
    pub total_ns: u128,
}

impl ActivationBenchmark {
    /// Elements processed per second.
    pub fn throughput(&self) -> f64 {
        if self.total_ns == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let total_elems = self.elements as f64 * f64::from(self.iterations);
        #[allow(clippy::cast_precision_loss)]
        let ns = self.total_ns as f64;
        total_elems / (ns / 1e9)
    }

    /// Run a trivial benchmark of `kind` over `elements` for `iters`
    /// iterations using the supplied kernel.
    pub fn run(
        kernel: &dyn ActivationKernel,
        kind: ActivationKind,
        elements: usize,
        iters: u32,
        config: &ActivationConfig,
    ) -> Self {
        let input = vec![0.5_f32; elements];
        let mut output = vec![0.0_f32; elements];
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = match kind {
                ActivationKind::Gelu => kernel.gelu(&input, &mut output),
                ActivationKind::Silu => kernel.silu(&input, &mut output),
                ActivationKind::Relu => kernel.relu(&input, &mut output),
                ActivationKind::Swish => {
                    kernel.swish(&input, &mut output, 1.0)
                }
                ActivationKind::Sigmoid => {
                    kernel.sigmoid(&input, &mut output)
                }
                ActivationKind::Tanh => {
                    kernel.tanh_activation(&input, &mut output)
                }
                ActivationKind::LeakyRelu => {
                    kernel.leaky_relu(
                        &input,
                        &mut output,
                        config.leaky_relu_alpha,
                    )
                }
                ActivationKind::Elu => {
                    kernel.elu(&input, &mut output, config.elu_alpha)
                }
            };
        }
        let total_ns = start.elapsed().as_nanos();
        Self { kind, elements, iterations: iters, total_ns }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    fn kern() -> CpuActivationKernels {
        CpuActivationKernels::default()
    }

    fn cfg() -> ActivationConfig {
        ActivationConfig::default()
    }

    // -- GELU unit tests ------------------------------------------------

    #[test]
    fn gelu_zero() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.gelu(&[0.0], &mut out).unwrap();
        assert!((out[0]).abs() < 1e-6, "GELU(0) ≈ 0");
    }

    #[test]
    fn gelu_positive() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.gelu(&[2.0], &mut out).unwrap();
        assert!(out[0] > 1.9, "GELU(2) ≈ 2");
    }

    #[test]
    fn gelu_negative() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.gelu(&[-3.0], &mut out).unwrap();
        assert!(out[0] < 0.0 && out[0] > -0.1, "GELU(-3) ≈ small negative");
    }

    #[test]
    fn gelu_batch() {
        let k = kern();
        let input = vec![0.0, 1.0, -1.0, 2.0];
        let mut out = vec![0.0; 4];
        let n = k.gelu(&input, &mut out).unwrap();
        assert_eq!(n, 4);
    }

    // -- SiLU unit tests ------------------------------------------------

    #[test]
    fn silu_zero() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.silu(&[0.0], &mut out).unwrap();
        assert!((out[0]).abs() < 1e-6, "SiLU(0) = 0");
    }

    #[test]
    fn silu_positive() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.silu(&[5.0], &mut out).unwrap();
        assert!(out[0] > 4.9, "SiLU(large positive) ≈ x");
    }

    #[test]
    fn silu_negative() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.silu(&[-5.0], &mut out).unwrap();
        assert!(out[0] < 0.0 && out[0] > -0.05);
    }

    // -- ReLU unit tests ------------------------------------------------

    #[test]
    fn relu_positive() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.relu(&[3.5], &mut out).unwrap();
        assert!((out[0] - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn relu_zero() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.relu(&[0.0], &mut out).unwrap();
        assert!((out[0]).abs() < f32::EPSILON);
    }

    #[test]
    fn relu_negative() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.relu(&[-2.0], &mut out).unwrap();
        assert!((out[0]).abs() < f32::EPSILON);
    }

    #[test]
    fn relu_batch() {
        let k = kern();
        let input = [-1.0, 0.0, 1.0, -0.5, 2.0];
        let mut out = [0.0_f32; 5];
        k.relu(&input, &mut out).unwrap();
        assert_eq!(out, [0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    // -- Swish unit tests -----------------------------------------------

    #[test]
    fn swish_beta_one_equals_silu() {
        let k = kern();
        let input = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut silu_out = [0.0_f32; 5];
        let mut swish_out = [0.0_f32; 5];
        k.silu(&input, &mut silu_out).unwrap();
        k.swish(&input, &mut swish_out, 1.0).unwrap();
        for (a, b) in silu_out.iter().zip(&swish_out) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn swish_beta_zero_is_half_x() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.swish(&[4.0], &mut out, 0.0).unwrap();
        // σ(0) = 0.5, so swish(x,0) = x * 0.5
        assert!((out[0] - 2.0).abs() < 1e-6);
    }

    // -- Sigmoid unit tests ---------------------------------------------

    #[test]
    fn sigmoid_zero() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.sigmoid(&[0.0], &mut out).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_large_positive() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.sigmoid(&[100.0], &mut out).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_large_negative() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.sigmoid(&[-100.0], &mut out).unwrap();
        assert!(out[0].abs() < 1e-6);
    }

    #[test]
    fn sigmoid_symmetry() {
        let k = kern();
        let mut out_pos = [0.0_f32; 1];
        let mut out_neg = [0.0_f32; 1];
        k.sigmoid(&[3.0], &mut out_pos).unwrap();
        k.sigmoid(&[-3.0], &mut out_neg).unwrap();
        assert!((out_pos[0] + out_neg[0] - 1.0).abs() < 1e-6);
    }

    // -- Tanh unit tests ------------------------------------------------

    #[test]
    fn tanh_zero() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.tanh_activation(&[0.0], &mut out).unwrap();
        assert!(out[0].abs() < 1e-6);
    }

    #[test]
    fn tanh_saturation_positive() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.tanh_activation(&[50.0], &mut out).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tanh_saturation_negative() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.tanh_activation(&[-50.0], &mut out).unwrap();
        assert!((out[0] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn tanh_odd_symmetry() {
        let k = kern();
        let mut out_pos = [0.0_f32; 1];
        let mut out_neg = [0.0_f32; 1];
        k.tanh_activation(&[1.5], &mut out_pos).unwrap();
        k.tanh_activation(&[-1.5], &mut out_neg).unwrap();
        assert!((out_pos[0] + out_neg[0]).abs() < 1e-6);
    }

    // -- LeakyReLU unit tests -------------------------------------------

    #[test]
    fn leaky_relu_positive() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.leaky_relu(&[2.0], &mut out, 0.01).unwrap();
        assert!((out[0] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn leaky_relu_negative() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.leaky_relu(&[-2.0], &mut out, 0.01).unwrap();
        assert!((out[0] - (-0.02)).abs() < 1e-6);
    }

    #[test]
    fn leaky_relu_zero_alpha_is_relu() {
        let k = kern();
        let input = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut relu_out = [0.0_f32; 5];
        let mut lrelu_out = [0.0_f32; 5];
        k.relu(&input, &mut relu_out).unwrap();
        k.leaky_relu(&input, &mut lrelu_out, 0.0).unwrap();
        assert_eq!(relu_out, lrelu_out);
    }

    // -- ELU unit tests -------------------------------------------------

    #[test]
    fn elu_positive() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.elu(&[3.0], &mut out, 1.0).unwrap();
        assert!((out[0] - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn elu_zero() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.elu(&[0.0], &mut out, 1.0).unwrap();
        assert!(out[0].abs() < f32::EPSILON);
    }

    #[test]
    fn elu_negative() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.elu(&[-1.0], &mut out, 1.0).unwrap();
        let expected = (-1.0_f32).exp_m1() * 1.0;
        assert!((out[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn elu_negative_custom_alpha() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.elu(&[-1.0], &mut out, 2.5).unwrap();
        let expected = 2.5 * (-1.0_f32).exp_m1();
        assert!((out[0] - expected).abs() < 1e-5);
    }

    // -- Error handling -------------------------------------------------

    #[test]
    fn output_too_small_returns_error() {
        let k = kern();
        let input = [1.0, 2.0, 3.0];
        let mut out = [0.0_f32; 2];
        assert!(k.gelu(&input, &mut out).is_err());
    }

    #[test]
    fn empty_input_returns_zero() {
        let k = kern();
        let mut out = [0.0_f32; 0];
        let n = k.relu(&[], &mut out).unwrap();
        assert_eq!(n, 0);
    }

    // -- NaN / Inf handling ---------------------------------------------

    #[test]
    fn relu_nan_returns_zero() {
        // f32::max(NaN, 0.0) returns 0.0 per Rust semantics
        let k = kern();
        let mut out = [f32::NAN; 1];
        k.relu(&[f32::NAN], &mut out).unwrap();
        assert!((out[0] - 0.0).abs() < f32::EPSILON || out[0].is_nan());
    }

    #[test]
    fn sigmoid_inf_saturates() {
        let k = kern();
        let mut out = [0.0_f32; 2];
        k.sigmoid(&[f32::INFINITY, f32::NEG_INFINITY], &mut out).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
    }

    #[test]
    fn tanh_inf_saturates() {
        let k = kern();
        let mut out = [0.0_f32; 2];
        k.tanh_activation(
            &[f32::INFINITY, f32::NEG_INFINITY],
            &mut out,
        )
        .unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn elu_neg_inf_clamps() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.elu(&[f32::NEG_INFINITY], &mut out, 1.0).unwrap();
        // safe_exp clamps input to -88, so result ≈ -1.0
        assert!((out[0] + 1.0).abs() < 1e-3);
    }

    #[test]
    fn gelu_nan_propagates() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.gelu(&[f32::NAN], &mut out).unwrap();
        assert!(out[0].is_nan());
    }

    // -- In-place helpers -----------------------------------------------

    #[test]
    fn inplace_relu() {
        let k = kern();
        let c = cfg();
        let mut data = vec![-1.0, 0.0, 1.0, -0.5, 2.0];
        activate_inplace(&k, ActivationKind::Relu, &mut data, &c).unwrap();
        assert_eq!(data, [0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn inplace_sigmoid() {
        let k = kern();
        let c = cfg();
        let mut data = vec![0.0];
        activate_inplace(&k, ActivationKind::Sigmoid, &mut data, &c).unwrap();
        assert!((data[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn inplace_gelu() {
        let k = kern();
        let c = cfg();
        let mut data = vec![0.0, 1.0];
        activate_inplace(&k, ActivationKind::Gelu, &mut data, &c).unwrap();
        assert!(data[0].abs() < 1e-6);
        assert!(data[1] > 0.8);
    }

    #[test]
    fn inplace_all_kinds_run() {
        let k = kern();
        let c = cfg();
        let kinds = [
            ActivationKind::Gelu,
            ActivationKind::Silu,
            ActivationKind::Relu,
            ActivationKind::Swish,
            ActivationKind::Sigmoid,
            ActivationKind::Tanh,
            ActivationKind::LeakyRelu,
            ActivationKind::Elu,
        ];
        for kind in kinds {
            let mut data = vec![0.5, -0.5];
            activate_inplace(&k, kind, &mut data, &c).unwrap();
        }
    }

    // -- Batch processing -----------------------------------------------

    #[test]
    fn batched_relu_full() {
        let k = kern();
        let c = cfg();
        let input = vec![-1.0, 2.0, -3.0, 4.0];
        let mut out = vec![0.0; 4];
        let n = activate_batched(
            &k,
            ActivationKind::Relu,
            &input,
            &mut out,
            &c,
            2,
        )
        .unwrap();
        assert_eq!(n, 4);
        assert_eq!(out, [0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn batched_with_remainder() {
        let k = kern();
        let c = cfg();
        let input = vec![1.0, -1.0, 2.0];
        let mut out = vec![0.0; 3];
        let n = activate_batched(
            &k,
            ActivationKind::Relu,
            &input,
            &mut out,
            &c,
            2,
        )
        .unwrap();
        assert_eq!(n, 3);
        assert_eq!(out, [1.0, 0.0, 2.0]);
    }

    #[test]
    fn batched_empty_input() {
        let k = kern();
        let c = cfg();
        let mut out = [];
        let n = activate_batched(
            &k,
            ActivationKind::Relu,
            &[],
            &mut out,
            &c,
            64,
        )
        .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn batched_zero_batch_size_errors() {
        let k = kern();
        let c = cfg();
        let input = vec![1.0];
        let mut out = vec![0.0; 1];
        assert!(activate_batched(
            &k,
            ActivationKind::Relu,
            &input,
            &mut out,
            &c,
            0,
        )
        .is_err());
    }

    #[test]
    fn batched_all_kinds() {
        let k = kern();
        let c = cfg();
        let kinds = [
            ActivationKind::Gelu,
            ActivationKind::Silu,
            ActivationKind::Relu,
            ActivationKind::Swish,
            ActivationKind::Sigmoid,
            ActivationKind::Tanh,
            ActivationKind::LeakyRelu,
            ActivationKind::Elu,
        ];
        let input = vec![0.5, -0.5, 1.0, -1.0];
        for kind in kinds {
            let mut out = vec![0.0; 4];
            activate_batched(&k, kind, &input, &mut out, &c, 2).unwrap();
        }
    }

    // -- Fused activation + bias ----------------------------------------

    #[test]
    fn fused_relu_bias() {
        let k = kern();
        let c = cfg();
        let input = [-2.0, 0.0, 1.0, -1.0];
        let bias = [1.0];
        let mut out = [0.0_f32; 4];
        fused_activation_bias(
            &k,
            ActivationKind::Relu,
            &input,
            &bias,
            &mut out,
            &c,
        )
        .unwrap();
        // input + 1 => [-1, 1, 2, 0], relu => [0, 1, 2, 0]
        assert_eq!(out, [0.0, 1.0, 2.0, 0.0]);
    }

    #[test]
    fn fused_sigmoid_bias_broadcast() {
        let k = kern();
        let c = cfg();
        let input = [0.0, 0.0, 0.0, 0.0];
        let bias = [0.0, 100.0];
        let mut out = [0.0_f32; 4];
        fused_activation_bias(
            &k,
            ActivationKind::Sigmoid,
            &input,
            &bias,
            &mut out,
            &c,
        )
        .unwrap();
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - 0.5).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fused_empty_bias_errors() {
        let k = kern();
        let c = cfg();
        let input = [1.0];
        let mut out = [0.0_f32; 1];
        assert!(fused_activation_bias(
            &k,
            ActivationKind::Relu,
            &input,
            &[],
            &mut out,
            &c,
        )
        .is_err());
    }

    #[test]
    fn fused_mismatched_bias_errors() {
        let k = kern();
        let c = cfg();
        let input = [1.0, 2.0, 3.0];
        let bias = [0.5, 0.5];
        let mut out = [0.0_f32; 3];
        // 3 % 2 != 0 → error
        assert!(fused_activation_bias(
            &k,
            ActivationKind::Relu,
            &input,
            &bias,
            &mut out,
            &c,
        )
        .is_err());
    }

    #[test]
    fn fused_all_kinds() {
        let k = kern();
        let c = cfg();
        let kinds = [
            ActivationKind::Gelu,
            ActivationKind::Silu,
            ActivationKind::Relu,
            ActivationKind::Swish,
            ActivationKind::Sigmoid,
            ActivationKind::Tanh,
            ActivationKind::LeakyRelu,
            ActivationKind::Elu,
        ];
        let input = [0.5, -0.5];
        let bias = [0.1];
        for kind in kinds {
            let mut out = [0.0_f32; 2];
            fused_activation_bias(
                &k,
                kind,
                &input,
                &bias,
                &mut out,
                &c,
            )
            .unwrap();
        }
    }

    // -- Benchmark helper -----------------------------------------------

    #[test]
    fn benchmark_basic() {
        let k = kern();
        let c = cfg();
        let b = ActivationBenchmark::run(
            &k,
            ActivationKind::Relu,
            1024,
            10,
            &c,
        );
        assert!(b.throughput() > 0.0);
        assert_eq!(b.elements, 1024);
        assert_eq!(b.iterations, 10);
    }

    #[test]
    fn benchmark_all_activations() {
        let k = kern();
        let c = cfg();
        let kinds = [
            ActivationKind::Gelu,
            ActivationKind::Silu,
            ActivationKind::Relu,
            ActivationKind::Swish,
            ActivationKind::Sigmoid,
            ActivationKind::Tanh,
            ActivationKind::LeakyRelu,
            ActivationKind::Elu,
        ];
        for kind in kinds {
            let b = ActivationBenchmark::run(&k, kind, 256, 5, &c);
            assert!(b.throughput() > 0.0);
        }
    }

    // -- Config defaults ------------------------------------------------

    #[test]
    fn default_config_values() {
        let c = ActivationConfig::default();
        assert!((c.leaky_relu_alpha - 0.01).abs() < f32::EPSILON);
        assert!((c.elu_alpha - 1.0).abs() < f32::EPSILON);
        assert!((c.epsilon - 1e-8).abs() < 1e-12);
    }

    // -- Numerical stability edge cases ---------------------------------

    #[test]
    fn sigmoid_very_large_values() {
        let k = kern();
        let mut out = [0.0_f32; 4];
        k.sigmoid(&[500.0, -500.0, 1000.0, -1000.0], &mut out).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
        assert!(out[3].abs() < 1e-6);
    }

    #[test]
    fn silu_large_negative_near_zero() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.silu(&[-100.0], &mut out).unwrap();
        assert!(out[0].abs() < 1e-3, "SiLU(very negative) ≈ 0");
    }

    #[test]
    fn elu_very_large_negative() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.elu(&[-500.0], &mut out, 1.0).unwrap();
        // Clamped exp, so result ≈ -1.0
        assert!((out[0] + 1.0).abs() < 1e-3);
        assert!(out[0].is_finite());
    }

    #[test]
    fn gelu_large_positive() {
        let k = kern();
        let mut out = [0.0_f32; 1];
        k.gelu(&[100.0], &mut out).unwrap();
        assert!((out[0] - 100.0).abs() < 1e-2, "GELU(large) ≈ x");
    }

    #[test]
    fn swish_large_beta() {
        let k = kern();
        let mut out = [0.0_f32; 2];
        k.swish(&[5.0, -5.0], &mut out, 100.0).unwrap();
        // σ(500) ≈ 1 → swish ≈ 5; σ(-500) ≈ 0 → swish ≈ 0
        assert!((out[0] - 5.0).abs() < 1e-3);
        assert!(out[1].abs() < 1e-3);
    }

    // -- Monotonicity / derivative properties ---------------------------

    #[test]
    fn relu_monotonic() {
        let k = kern();
        let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0.0_f32; input.len()];
        k.relu(&input, &mut out).unwrap();
        for w in out.windows(2) {
            assert!(w[1] >= w[0], "ReLU must be non-decreasing");
        }
    }

    #[test]
    fn sigmoid_monotonic() {
        let k = kern();
        let input: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0.0_f32; input.len()];
        k.sigmoid(&input, &mut out).unwrap();
        for w in out.windows(2) {
            assert!(w[1] >= w[0], "Sigmoid must be non-decreasing");
        }
    }

    #[test]
    fn sigmoid_bounded() {
        let k = kern();
        let input: Vec<f32> = (-50..=50).map(|i| i as f32).collect();
        let mut out = vec![0.0_f32; input.len()];
        k.sigmoid(&input, &mut out).unwrap();
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "Sigmoid in [0,1]");
        }
    }

    #[test]
    fn tanh_bounded() {
        let k = kern();
        let input: Vec<f32> = (-50..=50).map(|i| i as f32).collect();
        let mut out = vec![0.0_f32; input.len()];
        k.tanh_activation(&input, &mut out).unwrap();
        for &v in &out {
            assert!(
                (-1.0..=1.0).contains(&v),
                "tanh in [-1,1], got {v}"
            );
        }
    }

    // -- ActivationKind enum coverage -----------------------------------

    #[test]
    fn activation_kind_debug() {
        let k = ActivationKind::Gelu;
        let s = format!("{k:?}");
        assert_eq!(s, "Gelu");
    }

    #[test]
    fn activation_kind_eq() {
        assert_eq!(ActivationKind::Relu, ActivationKind::Relu);
        assert_ne!(ActivationKind::Relu, ActivationKind::Elu);
    }

    #[test]
    fn activation_kind_clone() {
        let k = ActivationKind::Sigmoid;
        let k2 = k;
        assert_eq!(k, k2);
    }

    // -- Large-vector stress test ---------------------------------------

    #[test]
    fn large_vector_all_activations() {
        let k = kern();
        let c = cfg();
        let n = 10_000;
        let input: Vec<f32> =
            (0..n).map(|i| (i as f32 - 5000.0) * 0.01).collect();
        let kinds = [
            ActivationKind::Gelu,
            ActivationKind::Silu,
            ActivationKind::Relu,
            ActivationKind::Swish,
            ActivationKind::Sigmoid,
            ActivationKind::Tanh,
            ActivationKind::LeakyRelu,
            ActivationKind::Elu,
        ];
        for kind in kinds {
            let mut out = vec![0.0_f32; n];
            activate_batched(&k, kind, &input, &mut out, &c, 512)
                .unwrap();
            // Verify all outputs are finite (no NaN/Inf from stable math)
            for &v in &out {
                assert!(
                    v.is_finite(),
                    "{kind:?} produced non-finite value"
                );
            }
        }
    }
}

// -- Property-based tests (proptest) ------------------------------------

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn kern() -> CpuActivationKernels {
        CpuActivationKernels::default()
    }

    proptest! {
        #[test]
        fn relu_non_negative(x in -100.0_f32..100.0) {
            let k = kern();
            let mut out = [0.0_f32; 1];
            k.relu(&[x], &mut out).unwrap();
            prop_assert!(out[0] >= 0.0);
        }

        #[test]
        fn sigmoid_in_unit_interval(x in -100.0_f32..100.0) {
            let k = kern();
            let mut out = [0.0_f32; 1];
            k.sigmoid(&[x], &mut out).unwrap();
            prop_assert!(out[0] >= 0.0 && out[0] <= 1.0);
        }

        #[test]
        fn tanh_in_range(x in -100.0_f32..100.0) {
            let k = kern();
            let mut out = [0.0_f32; 1];
            k.tanh_activation(&[x], &mut out).unwrap();
            prop_assert!(out[0] >= -1.0 && out[0] <= 1.0);
        }

        #[test]
        fn gelu_finite(x in -50.0_f32..50.0) {
            let k = kern();
            let mut out = [0.0_f32; 1];
            k.gelu(&[x], &mut out).unwrap();
            prop_assert!(out[0].is_finite());
        }

        #[test]
        fn silu_finite(x in -100.0_f32..100.0) {
            let k = kern();
            let mut out = [0.0_f32; 1];
            k.silu(&[x], &mut out).unwrap();
            prop_assert!(out[0].is_finite());
        }

        #[test]
        fn elu_continuous_at_zero(
            x in -0.01_f32..0.01
        ) {
            let k = kern();
            let mut out = [0.0_f32; 1];
            k.elu(&[x], &mut out, 1.0).unwrap();
            // Near 0, ELU ≈ x (within tolerance)
            prop_assert!((out[0] - x).abs() < 0.02);
        }

        #[test]
        fn leaky_relu_nonzero_for_negative(
            x in -100.0_f32..-0.001
        ) {
            let k = kern();
            let mut out = [0.0_f32; 1];
            k.leaky_relu(&[x], &mut out, 0.01).unwrap();
            prop_assert!(out[0] < 0.0, "LeakyReLU(neg) < 0");
        }

        #[test]
        fn swish_beta1_matches_silu(x in -50.0_f32..50.0) {
            let k = kern();
            let mut silu_out = [0.0_f32; 1];
            let mut swish_out = [0.0_f32; 1];
            k.silu(&[x], &mut silu_out).unwrap();
            k.swish(&[x], &mut swish_out, 1.0).unwrap();
            prop_assert!((silu_out[0] - swish_out[0]).abs() < 1e-5);
        }
    }
}
