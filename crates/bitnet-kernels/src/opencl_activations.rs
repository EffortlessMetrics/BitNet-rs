//! Comprehensive activation function library with OpenCL kernel sources.
//!
//! Provides 14 activation functions as CPU reference implementations plus
//! embedded OpenCL C source for GPU dispatch. Each activation has:
//!
//! - **Exact CPU scalar** — `ActivationKernel::apply_ref`
//! - **Derivative** — `ActivationDerivative::derivative_ref` (training support)
//! - **Fast approximate** — `ApproximateActivation::apply_approx` (polynomial)
//! - **Fused linear + activation** — `FusedActivation::apply_fused_ref`
//! - **Statistics** — `ActivationStats` tracking throughput and numerical health
//!
//! # OpenCL kernel
//!
//! [`ACTIVATIONS_CL`] contains OpenCL C source implementing all 14 activations,
//! element-wise application, fused linear+activation, and derivative kernels.

use std::time::Instant;

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// ActivationKind
// ---------------------------------------------------------------------------

/// All supported activation functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationKind {
    /// Rectified Linear Unit: `max(0, x)`.
    ReLU,
    /// Gaussian Error Linear Unit (exact via erf).
    GELU,
    /// GELU with tanh approximation.
    GELUTanh,
    /// Sigmoid Linear Unit: `x * sigmoid(x)`.
    SiLU,
    /// Swish: `x * sigmoid(beta * x)` with beta=1 (equivalent to SiLU).
    Swish,
    /// Logistic sigmoid: `1 / (1 + exp(-x))`.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Leaky ReLU: `x if x >= 0, alpha*x otherwise`.
    LeakyReLU(f32),
    /// Exponential Linear Unit: `x if x >= 0, alpha*(exp(x)-1) otherwise`.
    ELU(f32),
    /// Softplus: `(1/beta) * ln(1 + exp(beta*x))`.
    Softplus(f32),
    /// Mish: `x * tanh(softplus(x))` where softplus uses beta=1.
    Mish,
    /// Hard Swish: `x * clamp((x+3)/6, 0, 1)`.
    HardSwish,
    /// Hard Sigmoid: `clamp((x+3)/6, 0, 1)`.
    HardSigmoid,
    /// Quick GELU: `x * sigmoid(1.702 * x)`.
    QuickGELU,
}

impl ActivationKind {
    /// Apply the activation function to a single scalar value.
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::ReLU => x.max(0.0),
            Self::GELU => x * 0.5 * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2)),
            Self::GELUTanh => {
                let c = (2.0_f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
            }
            Self::SiLU | Self::Swish => x * sigmoid(x),
            Self::Sigmoid => sigmoid(x),
            Self::Tanh => x.tanh(),
            Self::LeakyReLU(alpha) => {
                if x >= 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            Self::ELU(alpha) => {
                if x >= 0.0 {
                    x
                } else {
                    alpha * (x.exp() - 1.0)
                }
            }
            Self::Softplus(beta) => {
                // Numerically stable: for large beta*x use x directly
                let bx = beta * x;
                if bx > 20.0 {
                    x
                } else if bx < -20.0 {
                    0.0
                } else {
                    (1.0 + bx.exp()).ln() / beta
                }
            }
            Self::Mish => {
                let sp = softplus_f32(x);
                x * sp.tanh()
            }
            Self::HardSwish => x * ((x + 3.0) / 6.0).clamp(0.0, 1.0),
            Self::HardSigmoid => ((x + 3.0) / 6.0).clamp(0.0, 1.0),
            Self::QuickGELU => x * sigmoid(1.702 * x),
        }
    }

    /// Compute the derivative of the activation at `x`.
    #[inline]
    pub fn derivative(self, x: f32) -> f32 {
        match self {
            Self::ReLU => {
                if x >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::GELU => {
                let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
                let cdf = 0.5 * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2));
                let pdf = sqrt_2_over_pi * 0.5 * (-0.5 * x * x).exp();
                cdf + x * pdf
            }
            Self::GELUTanh => {
                let c = (2.0_f32 / std::f32::consts::PI).sqrt();
                let inner = c * (x + 0.044715 * x * x * x);
                let t = inner.tanh();
                let sech2 = 1.0 - t * t;
                let d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
                0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
            }
            Self::SiLU | Self::Swish => {
                let s = sigmoid(x);
                s + x * s * (1.0 - s)
            }
            Self::Sigmoid => {
                let s = sigmoid(x);
                s * (1.0 - s)
            }
            Self::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Self::LeakyReLU(alpha) => {
                if x >= 0.0 {
                    1.0
                } else {
                    alpha
                }
            }
            Self::ELU(alpha) => {
                if x >= 0.0 {
                    1.0
                } else {
                    alpha * x.exp()
                }
            }
            Self::Softplus(beta) => {
                let bx = beta * x;
                if bx > 20.0 {
                    1.0
                } else if bx < -20.0 {
                    0.0
                } else {
                    sigmoid(bx)
                }
            }
            Self::Mish => {
                let sp = softplus_f32(x);
                let t = sp.tanh();
                let s = sigmoid(x);
                t + x * (1.0 - t * t) * s
            }
            Self::HardSwish => {
                if x <= -3.0 {
                    0.0
                } else if x >= 3.0 {
                    1.0
                } else {
                    (2.0 * x + 3.0) / 6.0
                }
            }
            Self::HardSigmoid => {
                if !(-3.0..=3.0).contains(&x) {
                    0.0
                } else {
                    1.0 / 6.0
                }
            }
            Self::QuickGELU => {
                let bx = 1.702 * x;
                let s = sigmoid(bx);
                s + x * 1.702 * s * (1.0 - s)
            }
        }
    }

    /// Return the name suitable for OpenCL kernel dispatch.
    pub fn kernel_name(self) -> &'static str {
        match self {
            Self::ReLU => "relu",
            Self::GELU => "gelu",
            Self::GELUTanh => "gelu_tanh",
            Self::SiLU => "silu",
            Self::Swish => "swish",
            Self::Sigmoid => "sigmoid_act",
            Self::Tanh => "tanh_act",
            Self::LeakyReLU(_) => "leaky_relu",
            Self::ELU(_) => "elu",
            Self::Softplus(_) => "softplus",
            Self::Mish => "mish",
            Self::HardSwish => "hard_swish",
            Self::HardSigmoid => "hard_sigmoid",
            Self::QuickGELU => "quick_gelu",
        }
    }

    /// Whether this activation requires an extra scalar parameter.
    pub fn has_parameter(self) -> bool {
        matches!(self, Self::LeakyReLU(_) | Self::ELU(_) | Self::Softplus(_))
    }

    /// Return the activation parameter, if any.
    pub fn parameter(self) -> Option<f32> {
        match self {
            Self::LeakyReLU(a) | Self::ELU(a) | Self::Softplus(a) => Some(a),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

/// Standard sigmoid: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softplus with beta=1: `ln(1 + exp(x))`.
#[inline]
fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Approximate erf via Abramowitz & Stegun (max error ~1.5e-7).
#[inline]
fn erf_approx(x: f32) -> f32 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ---------------------------------------------------------------------------
// ActivationKernel — element-wise application
// ---------------------------------------------------------------------------

/// Applies an activation function element-wise over a buffer.
#[derive(Debug, Clone)]
pub struct ActivationKernel {
    /// Which activation to apply.
    pub kind: ActivationKind,
}

impl ActivationKernel {
    /// Create a new kernel for the given activation.
    pub fn new(kind: ActivationKind) -> Self {
        Self { kind }
    }

    /// Apply activation element-wise (CPU reference, in-place).
    pub fn apply_inplace(&self, data: &mut [f32]) {
        for v in data.iter_mut() {
            *v = self.kind.apply(*v);
        }
    }

    /// Apply activation element-wise (CPU reference, out-of-place).
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `output.len() < input.len()`.
    pub fn apply_ref(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if output.len() < input.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!("output length {} < input length {}", output.len(), input.len()),
            }
            .into());
        }
        for (o, &x) in output.iter_mut().zip(input.iter()) {
            *o = self.kind.apply(x);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FusedActivation — fused linear + activation
// ---------------------------------------------------------------------------

/// Fused linear transformation + activation: `activation(x * weight + bias)`.
///
/// Avoids an extra memory pass compared to applying the linear and activation
/// separately.
#[derive(Debug, Clone)]
pub struct FusedActivation {
    /// Which activation to apply after the linear transform.
    pub kind: ActivationKind,
}

impl FusedActivation {
    pub fn new(kind: ActivationKind) -> Self {
        Self { kind }
    }

    /// Fused `activation(input @ weight^T + bias)` (CPU reference).
    ///
    /// - `input`:  `[batch, in_features]`
    /// - `weight`: `[out_features, in_features]` (row-major, each row = output neuron)
    /// - `bias`:   `[out_features]` (or empty for no bias)
    /// - `output`: `[batch, out_features]`
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] on dimension mismatch.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_fused_ref(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        output: &mut [f32],
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<()> {
        if batch == 0 || in_features == 0 || out_features == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "dimensions must be non-zero".into(),
            }
            .into());
        }
        if input.len() < batch * in_features {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "input length {} < batch*in_features {}",
                    input.len(),
                    batch * in_features,
                ),
            }
            .into());
        }
        if weight.len() < out_features * in_features {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "weight length {} < out*in {}",
                    weight.len(),
                    out_features * in_features,
                ),
            }
            .into());
        }
        if !bias.is_empty() && bias.len() < out_features {
            return Err(KernelError::InvalidArguments {
                reason: format!("bias length {} < out_features {}", bias.len(), out_features),
            }
            .into());
        }
        if output.len() < batch * out_features {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output length {} < batch*out_features {}",
                    output.len(),
                    batch * out_features,
                ),
            }
            .into());
        }

        for b in 0..batch {
            for j in 0..out_features {
                let mut acc = 0.0_f32;
                for k in 0..in_features {
                    acc += input[b * in_features + k] * weight[j * in_features + k];
                }
                if !bias.is_empty() {
                    acc += bias[j];
                }
                output[b * out_features + j] = self.kind.apply(acc);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ActivationDerivative
// ---------------------------------------------------------------------------

/// Computes the derivative of an activation function element-wise.
#[derive(Debug, Clone)]
pub struct ActivationDerivative {
    pub kind: ActivationKind,
}

impl ActivationDerivative {
    pub fn new(kind: ActivationKind) -> Self {
        Self { kind }
    }

    /// Compute derivative element-wise (CPU reference).
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `output.len() < input.len()`.
    pub fn derivative_ref(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if output.len() < input.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!("output length {} < input length {}", output.len(), input.len()),
            }
            .into());
        }
        for (o, &x) in output.iter_mut().zip(input.iter()) {
            *o = self.kind.derivative(x);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ApproximateActivation
// ---------------------------------------------------------------------------

/// Fast approximate versions of activation functions using polynomials or
/// piece-wise linear approximations. Trades accuracy for throughput.
#[derive(Debug, Clone)]
pub struct ApproximateActivation {
    pub kind: ActivationKind,
}

impl ApproximateActivation {
    pub fn new(kind: ActivationKind) -> Self {
        Self { kind }
    }

    /// Apply the fast approximate version of the activation.
    #[inline]
    pub fn apply_approx(x: f32, kind: ActivationKind) -> f32 {
        match kind {
            ActivationKind::GELU => {
                // Tanh approximation (same as GELUTanh)
                let c = (2.0_f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + fast_tanh(c * (x + 0.044715 * x * x * x)))
            }
            ActivationKind::GELUTanh => {
                // Same tanh form, but use fast_tanh
                let c = (2.0_f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + fast_tanh(c * (x + 0.044715 * x * x * x)))
            }
            ActivationKind::Sigmoid => fast_sigmoid(x),
            ActivationKind::Tanh => fast_tanh(x),
            ActivationKind::SiLU | ActivationKind::Swish => x * fast_sigmoid(x),
            ActivationKind::Mish => {
                let sp = fast_softplus(x);
                x * fast_tanh(sp)
            }
            ActivationKind::Softplus(beta) => {
                let bx = beta * x;
                fast_softplus(bx) / beta
            }
            ActivationKind::QuickGELU => x * fast_sigmoid(1.702 * x),
            // For simple activations the approximate == exact
            other => other.apply(x),
        }
    }

    /// Apply the fast approximate activation element-wise.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `output.len() < input.len()`.
    pub fn apply_approx_ref(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if output.len() < input.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!("output length {} < input length {}", output.len(), input.len()),
            }
            .into());
        }
        for (o, &x) in output.iter_mut().zip(input.iter()) {
            *o = Self::apply_approx(x, self.kind);
        }
        Ok(())
    }
}

/// Fast sigmoid using `1/(1+exp(-x))` — same as exact for f32, but the
/// *callers* (fast_tanh, etc.) provide the speed-up. Kept as a separate
/// function so the approximate pipeline stays explicit.
#[inline]
fn fast_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Fast tanh: `2*sigmoid(2x) - 1` using the fast sigmoid.
#[inline]
fn fast_tanh(x: f32) -> f32 {
    // Clamp to avoid divergence far from origin
    let x = x.clamp(-5.0, 5.0);
    let x2 = x * x;
    // Padé[3/3] approximant
    let num = x * (135135.0 + x2 * (17325.0 + x2 * 378.0));
    let den = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2));
    num / den
}

/// Fast softplus: `log(1 + exp(x))` with branch for large |x|.
#[inline]
fn fast_softplus(x: f32) -> f32 {
    if x > 10.0 {
        x
    } else if x < -10.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

// ---------------------------------------------------------------------------
// ActivationStats
// ---------------------------------------------------------------------------

/// Statistics gathered from an activation pass.
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Activation kind applied.
    pub kind: ActivationKind,
    /// Number of elements processed.
    pub element_count: usize,
    /// Minimum output value.
    pub min_value: f32,
    /// Maximum output value.
    pub max_value: f32,
    /// Mean output value.
    pub mean_value: f32,
    /// Number of NaN values in the output.
    pub nan_count: usize,
    /// Number of ±Inf values in the output.
    pub inf_count: usize,
    /// Number of exactly-zero outputs.
    pub zero_count: usize,
    /// Wall-clock time in microseconds.
    pub elapsed_us: u64,
}

impl ActivationStats {
    /// Apply activation to `input`, write to `output`, and collect stats.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `output.len() < input.len()`.
    pub fn apply_and_collect(
        kind: ActivationKind,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<Self> {
        if output.len() < input.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!("output length {} < input length {}", output.len(), input.len()),
            }
            .into());
        }

        let start = Instant::now();
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut sum = 0.0_f64;
        let mut nan_count = 0_usize;
        let mut inf_count = 0_usize;
        let mut zero_count = 0_usize;

        for (o, &x) in output.iter_mut().zip(input.iter()) {
            let y = kind.apply(x);
            *o = y;
            if y.is_nan() {
                nan_count += 1;
            } else if y.is_infinite() {
                inf_count += 1;
            } else {
                if y < min_val {
                    min_val = y;
                }
                if y > max_val {
                    max_val = y;
                }
                sum += y as f64;
            }
            if y == 0.0 {
                zero_count += 1;
            }
        }

        let n = input.len();
        let elapsed_us = start.elapsed().as_micros() as u64;
        let mean_value = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };
        if min_val == f32::INFINITY {
            min_val = 0.0;
        }
        if max_val == f32::NEG_INFINITY {
            max_val = 0.0;
        }

        Ok(Self {
            kind,
            element_count: n,
            min_value: min_val,
            max_value: max_val,
            mean_value,
            nan_count,
            inf_count,
            zero_count,
            elapsed_us,
        })
    }

    /// Throughput in millions of elements per second.
    pub fn throughput_meps(&self) -> f64 {
        if self.elapsed_us == 0 {
            return 0.0;
        }
        self.element_count as f64 / self.elapsed_us as f64
    }

    /// Whether any NaN or Inf values were detected.
    pub fn has_numerical_issues(&self) -> bool {
        self.nan_count > 0 || self.inf_count > 0
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C source covering all 14 activation functions, element-wise
/// application, fused linear+activation, and derivative computation.
pub const ACTIVATIONS_CL: &str = r#"
// ── scalar activations ──────────────────────────────────────────

inline float act_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float act_relu(float x) {
    return fmax(x, 0.0f);
}

inline float act_gelu(float x) {
    // Exact: x * 0.5 * (1 + erf(x / sqrt(2)))
    return x * 0.5f * (1.0f + erf(x * 0.7071067811865475f));
}

inline float act_gelu_tanh(float x) {
    float c = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}

inline float act_silu(float x) {
    return x * act_sigmoid(x);
}

inline float act_swish(float x) {
    return x * act_sigmoid(x);
}

inline float act_tanh(float x) {
    return tanh(x);
}

inline float act_leaky_relu(float x, float alpha) {
    return x >= 0.0f ? x : alpha * x;
}

inline float act_elu(float x, float alpha) {
    return x >= 0.0f ? x : alpha * (exp(x) - 1.0f);
}

inline float act_softplus(float x, float beta) {
    float bx = beta * x;
    if (bx > 20.0f) return x;
    if (bx < -20.0f) return 0.0f;
    return log(1.0f + exp(bx)) / beta;
}

inline float act_mish(float x) {
    float sp = log(1.0f + exp(x));
    return x * tanh(sp);
}

inline float act_hard_swish(float x) {
    return x * clamp((x + 3.0f) / 6.0f, 0.0f, 1.0f);
}

inline float act_hard_sigmoid(float x) {
    return clamp((x + 3.0f) / 6.0f, 0.0f, 1.0f);
}

inline float act_quick_gelu(float x) {
    return x * act_sigmoid(1.702f * x);
}

// ── element-wise kernels (one per activation) ───────────────────

__kernel void relu(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_relu(in[i]);
}

__kernel void gelu(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_gelu(in[i]);
}

__kernel void gelu_tanh(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_gelu_tanh(in[i]);
}

__kernel void silu(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_silu(in[i]);
}

__kernel void swish(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_swish(in[i]);
}

__kernel void sigmoid_act(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_sigmoid(in[i]);
}

__kernel void tanh_act(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_tanh(in[i]);
}

__kernel void leaky_relu(
    __global const float* in, __global float* out, int n, float alpha
) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_leaky_relu(in[i], alpha);
}

__kernel void elu(
    __global const float* in, __global float* out, int n, float alpha
) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_elu(in[i], alpha);
}

__kernel void softplus(
    __global const float* in, __global float* out, int n, float beta
) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_softplus(in[i], beta);
}

__kernel void mish(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_mish(in[i]);
}

__kernel void hard_swish(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_hard_swish(in[i]);
}

__kernel void hard_sigmoid(
    __global const float* in, __global float* out, int n
) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_hard_sigmoid(in[i]);
}

__kernel void quick_gelu(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = act_quick_gelu(in[i]);
}

// ── fused linear + activation ───────────────────────────────────
// output[b * out_f + j] = activation(sum_k input[b*in_f+k]*weight[j*in_f+k] + bias[j])
// Dispatch: global = (out_features, batch), local = (min(out_features,256), 1)

__kernel void fused_linear_relu(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    int batch, int in_f, int out_f, int has_bias
) {
    int j = get_global_id(0);
    int b = get_global_id(1);
    if (b >= batch || j >= out_f) return;
    float acc = 0.0f;
    for (int k = 0; k < in_f; k++)
        acc += input[b * in_f + k] * weight[j * in_f + k];
    if (has_bias) acc += bias[j];
    output[b * out_f + j] = act_relu(acc);
}

__kernel void fused_linear_silu(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    int batch, int in_f, int out_f, int has_bias
) {
    int j = get_global_id(0);
    int b = get_global_id(1);
    if (b >= batch || j >= out_f) return;
    float acc = 0.0f;
    for (int k = 0; k < in_f; k++)
        acc += input[b * in_f + k] * weight[j * in_f + k];
    if (has_bias) acc += bias[j];
    output[b * out_f + j] = act_silu(acc);
}

__kernel void fused_linear_gelu(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    int batch, int in_f, int out_f, int has_bias
) {
    int j = get_global_id(0);
    int b = get_global_id(1);
    if (b >= batch || j >= out_f) return;
    float acc = 0.0f;
    for (int k = 0; k < in_f; k++)
        acc += input[b * in_f + k] * weight[j * in_f + k];
    if (has_bias) acc += bias[j];
    output[b * out_f + j] = act_gelu(acc);
}

// ── derivative kernels ──────────────────────────────────────────

inline float deriv_sigmoid(float x) {
    float s = act_sigmoid(x);
    return s * (1.0f - s);
}

inline float deriv_relu(float x) {
    return x >= 0.0f ? 1.0f : 0.0f;
}

inline float deriv_silu(float x) {
    float s = act_sigmoid(x);
    return s + x * s * (1.0f - s);
}

inline float deriv_tanh(float x) {
    float t = tanh(x);
    return 1.0f - t * t;
}

__kernel void relu_deriv(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = deriv_relu(in[i]);
}

__kernel void sigmoid_deriv(
    __global const float* in, __global float* out, int n
) {
    int i = get_global_id(0);
    if (i < n) out[i] = deriv_sigmoid(in[i]);
}

__kernel void silu_deriv(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = deriv_silu(in[i]);
}

__kernel void tanh_deriv(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = deriv_tanh(in[i]);
}
"#;

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ----------------------------------------------------------

    fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
        assert!((a - b).abs() <= tol, "{msg}: {a} vs {b} (diff {})", (a - b).abs());
    }

    /// Numerical derivative via central difference.
    fn numerical_derivative(f: impl Fn(f32) -> f32, x: f32, h: f32) -> f32 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    /// All non-parameterized kinds for iteration.
    fn basic_kinds() -> Vec<ActivationKind> {
        vec![
            ActivationKind::ReLU,
            ActivationKind::GELU,
            ActivationKind::GELUTanh,
            ActivationKind::SiLU,
            ActivationKind::Swish,
            ActivationKind::Sigmoid,
            ActivationKind::Tanh,
            ActivationKind::Mish,
            ActivationKind::HardSwish,
            ActivationKind::HardSigmoid,
            ActivationKind::QuickGELU,
        ]
    }

    fn all_kinds() -> Vec<ActivationKind> {
        let mut v = basic_kinds();
        v.push(ActivationKind::LeakyReLU(0.01));
        v.push(ActivationKind::ELU(1.0));
        v.push(ActivationKind::Softplus(1.0));
        v
    }

    // =====================================================================
    // 1. Known-point tests for each activation
    // =====================================================================

    #[test]
    fn test_relu_known_points() {
        let k = ActivationKind::ReLU;
        assert_eq!(k.apply(0.0), 0.0);
        assert_eq!(k.apply(1.0), 1.0);
        assert_eq!(k.apply(-1.0), 0.0);
        assert_eq!(k.apply(5.0), 5.0);
        assert_eq!(k.apply(-100.0), 0.0);
    }

    #[test]
    fn test_gelu_known_points() {
        let k = ActivationKind::GELU;
        assert_close(k.apply(0.0), 0.0, 1e-6, "GELU(0)");
        assert_close(k.apply(1.0), 0.8413, 1e-3, "GELU(1)");
        assert_close(k.apply(-1.0), -0.1587, 1e-3, "GELU(-1)");
    }

    #[test]
    fn test_gelu_tanh_known_points() {
        let k = ActivationKind::GELUTanh;
        assert_close(k.apply(0.0), 0.0, 1e-6, "GELUTanh(0)");
        assert_close(k.apply(1.0), 0.8412, 1e-3, "GELUTanh(1)");
        assert_close(k.apply(-1.0), -0.1588, 1e-3, "GELUTanh(-1)");
    }

    #[test]
    fn test_silu_known_points() {
        let k = ActivationKind::SiLU;
        assert_close(k.apply(0.0), 0.0, 1e-6, "SiLU(0)");
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.7311
        assert_close(k.apply(1.0), 0.7311, 1e-3, "SiLU(1)");
    }

    #[test]
    fn test_swish_known_points() {
        let k = ActivationKind::Swish;
        // Swish = SiLU when beta=1
        assert_close(k.apply(0.0), 0.0, 1e-6, "Swish(0)");
        assert_close(k.apply(1.0), 0.7311, 1e-3, "Swish(1)");
    }

    #[test]
    fn test_sigmoid_known_points() {
        let k = ActivationKind::Sigmoid;
        assert_close(k.apply(0.0), 0.5, 1e-6, "Sigmoid(0)");
        assert!(k.apply(10.0) > 0.999);
        assert!(k.apply(-10.0) < 0.001);
    }

    #[test]
    fn test_tanh_known_points() {
        let k = ActivationKind::Tanh;
        assert_close(k.apply(0.0), 0.0, 1e-6, "Tanh(0)");
        assert_close(k.apply(1.0), 1.0_f32.tanh(), 1e-6, "Tanh(1)");
    }

    #[test]
    fn test_leaky_relu_known_points() {
        let k = ActivationKind::LeakyReLU(0.01);
        assert_eq!(k.apply(0.0), 0.0);
        assert_eq!(k.apply(1.0), 1.0);
        assert_close(k.apply(-1.0), -0.01, 1e-6, "LeakyReLU(-1)");
        assert_close(k.apply(-100.0), -1.0, 1e-6, "LeakyReLU(-100)");
    }

    #[test]
    fn test_elu_known_points() {
        let k = ActivationKind::ELU(1.0);
        assert_eq!(k.apply(0.0), 0.0);
        assert_eq!(k.apply(1.0), 1.0);
        // ELU(-1, alpha=1) = exp(-1) - 1 ≈ -0.6321
        assert_close(k.apply(-1.0), -0.6321, 1e-3, "ELU(-1)");
    }

    #[test]
    fn test_softplus_known_points() {
        let k = ActivationKind::Softplus(1.0);
        // softplus(0) = ln(2) ≈ 0.6931
        assert_close(k.apply(0.0), 0.6931, 1e-3, "Softplus(0)");
        // Large x → x
        assert_close(k.apply(100.0), 100.0, 0.1, "Softplus(100)");
    }

    #[test]
    fn test_mish_known_points() {
        let k = ActivationKind::Mish;
        assert_close(k.apply(0.0), 0.0, 1e-5, "Mish(0)");
        // Mish(1) = 1 * tanh(ln(1+e^1)) ≈ 0.8651
        assert_close(k.apply(1.0), 0.8651, 1e-3, "Mish(1)");
    }

    #[test]
    fn test_hard_swish_known_points() {
        let k = ActivationKind::HardSwish;
        assert_close(k.apply(0.0), 0.0, 1e-6, "HardSwish(0)");
        // HardSwish(3) = 3 * clamp(6/6, 0, 1) = 3
        assert_close(k.apply(3.0), 3.0, 1e-6, "HardSwish(3)");
        // HardSwish(-3) = -3 * clamp(0/6, 0, 1) = 0
        assert_close(k.apply(-3.0), 0.0, 1e-6, "HardSwish(-3)");
        // HardSwish(-4) = -4 * 0 = 0
        assert_close(k.apply(-4.0), 0.0, 1e-6, "HardSwish(-4)");
    }

    #[test]
    fn test_hard_sigmoid_known_points() {
        let k = ActivationKind::HardSigmoid;
        assert_close(k.apply(0.0), 0.5, 1e-6, "HardSig(0)");
        assert_close(k.apply(3.0), 1.0, 1e-6, "HardSig(3)");
        assert_close(k.apply(-3.0), 0.0, 1e-6, "HardSig(-3)");
        assert_close(k.apply(10.0), 1.0, 1e-6, "HardSig(10)");
    }

    #[test]
    fn test_quick_gelu_known_points() {
        let k = ActivationKind::QuickGELU;
        assert_close(k.apply(0.0), 0.0, 1e-6, "QuickGELU(0)");
        // QuickGELU(1) = sigmoid(1.702) ≈ 0.8455
        assert_close(k.apply(1.0), 0.8455, 1e-3, "QuickGELU(1)");
    }

    // =====================================================================
    // 2. GELU vs GELUTanh approximation closeness
    // =====================================================================

    #[test]
    fn test_gelu_vs_gelu_tanh_close() {
        let gelu = ActivationKind::GELU;
        let gelu_t = ActivationKind::GELUTanh;
        for &x in &[-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0] {
            let exact = gelu.apply(x);
            let approx = gelu_t.apply(x);
            assert_close(exact, approx, 0.02, &format!("GELU≈GELUTanh x={x}"));
        }
    }

    #[test]
    fn test_gelu_vs_gelu_tanh_relative_error() {
        let gelu = ActivationKind::GELU;
        let gelu_t = ActivationKind::GELUTanh;
        for i in -50..=50 {
            let x = i as f32 * 0.1;
            let exact = gelu.apply(x);
            let approx = gelu_t.apply(x);
            if exact.abs() > 0.05 {
                let rel = ((exact - approx) / exact).abs();
                assert!(rel < 0.05, "GELU vs GELUTanh rel error {rel} at x={x}");
            }
        }
    }

    // =====================================================================
    // 3. SiLU == x * sigmoid(x) identity
    // =====================================================================

    #[test]
    fn test_silu_equals_x_times_sigmoid() {
        let silu = ActivationKind::SiLU;
        let sig = ActivationKind::Sigmoid;
        for i in -100..=100 {
            let x = i as f32 * 0.1;
            let expected = x * sig.apply(x);
            assert_close(silu.apply(x), expected, 1e-6, &format!("SiLU=x*σ x={x}"));
        }
    }

    #[test]
    fn test_swish_equals_silu() {
        let silu = ActivationKind::SiLU;
        let swish = ActivationKind::Swish;
        for i in -50..=50 {
            let x = i as f32 * 0.2;
            assert_eq!(silu.apply(x), swish.apply(x), "SiLU==Swish at x={x}");
        }
    }

    // =====================================================================
    // 4. Sigmoid output in [0,1], Tanh in [-1,1]
    // =====================================================================

    #[test]
    fn test_sigmoid_range() {
        let k = ActivationKind::Sigmoid;
        for i in -500..=500 {
            let x = i as f32 * 0.1;
            let y = k.apply(x);
            assert!((0.0..=1.0).contains(&y), "Sigmoid({x}) = {y} out of [0,1]");
        }
    }

    #[test]
    fn test_tanh_range() {
        let k = ActivationKind::Tanh;
        for i in -500..=500 {
            let x = i as f32 * 0.1;
            let y = k.apply(x);
            assert!((-1.0..=1.0).contains(&y), "Tanh({x}) = {y} out of [-1,1]");
        }
    }

    #[test]
    fn test_hard_sigmoid_range() {
        let k = ActivationKind::HardSigmoid;
        for i in -200..=200 {
            let x = i as f32 * 0.1;
            let y = k.apply(x);
            assert!((0.0..=1.0).contains(&y), "HardSigmoid({x}) = {y} out of [0,1]");
        }
    }

    // =====================================================================
    // 5. ReLU(negative) = 0
    // =====================================================================

    #[test]
    fn test_relu_negative_zero() {
        let k = ActivationKind::ReLU;
        for i in 1..=100 {
            let x = -(i as f32);
            assert_eq!(k.apply(x), 0.0, "ReLU({x}) should be 0");
        }
    }

    #[test]
    fn test_relu_positive_passthrough() {
        let k = ActivationKind::ReLU;
        for i in 0..=100 {
            let x = i as f32 * 0.5;
            assert_eq!(k.apply(x), x, "ReLU({x}) should be {x}");
        }
    }

    // =====================================================================
    // 6. Derivative correctness for each activation
    // =====================================================================

    #[test]
    fn test_relu_derivative() {
        let k = ActivationKind::ReLU;
        assert_eq!(k.derivative(1.0), 1.0);
        assert_eq!(k.derivative(-1.0), 0.0);
        assert_eq!(k.derivative(5.0), 1.0);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let k = ActivationKind::Sigmoid;
        // σ'(0) = σ(0)*(1-σ(0)) = 0.5*0.5 = 0.25
        assert_close(k.derivative(0.0), 0.25, 1e-6, "σ'(0)");
    }

    #[test]
    fn test_tanh_derivative() {
        let k = ActivationKind::Tanh;
        // tanh'(0) = 1 - tanh²(0) = 1
        assert_close(k.derivative(0.0), 1.0, 1e-6, "tanh'(0)");
    }

    #[test]
    fn test_silu_derivative() {
        let k = ActivationKind::SiLU;
        // SiLU'(0) = σ(0) + 0*σ(0)*(1-σ(0)) = 0.5
        assert_close(k.derivative(0.0), 0.5, 1e-6, "SiLU'(0)");
    }

    #[test]
    fn test_leaky_relu_derivative() {
        let k = ActivationKind::LeakyReLU(0.01);
        assert_eq!(k.derivative(1.0), 1.0);
        assert_close(k.derivative(-1.0), 0.01, 1e-6, "LeakyReLU'(-1)");
    }

    #[test]
    fn test_elu_derivative() {
        let k = ActivationKind::ELU(1.0);
        assert_eq!(k.derivative(1.0), 1.0);
        // ELU'(-1, alpha=1) = alpha * exp(-1) ≈ 0.3679
        assert_close(k.derivative(-1.0), 0.3679, 1e-3, "ELU'(-1)");
    }

    #[test]
    fn test_softplus_derivative() {
        let k = ActivationKind::Softplus(1.0);
        // softplus'(x, beta=1) = sigmoid(x)
        assert_close(k.derivative(0.0), 0.5, 1e-6, "Softplus'(0)");
    }

    #[test]
    fn test_hard_swish_derivative() {
        let k = ActivationKind::HardSwish;
        // In linear region (|x| < 3): (2x+3)/6
        // At x=0: 3/6 = 0.5
        assert_close(k.derivative(0.0), 0.5, 1e-6, "HardSwish'(0)");
        // At x=3: saturated → 1
        assert_close(k.derivative(3.0), 1.0, 1e-6, "HardSwish'(3)");
        // At x=-3: saturated → 0
        assert_close(k.derivative(-3.0), 0.0, 1e-6, "HardSwish'(-3)");
        assert_close(k.derivative(-4.0), 0.0, 1e-6, "HardSwish'(-4)");
    }

    #[test]
    fn test_hard_sigmoid_derivative() {
        let k = ActivationKind::HardSigmoid;
        assert_close(k.derivative(0.0), 1.0 / 6.0, 1e-6, "HardSig'(0)");
        assert_eq!(k.derivative(5.0), 0.0);
        assert_eq!(k.derivative(-5.0), 0.0);
    }

    #[test]
    fn test_quick_gelu_derivative() {
        let k = ActivationKind::QuickGELU;
        // QuickGELU'(0) = σ(0) + 0 * ... = 0.5
        assert_close(k.derivative(0.0), 0.5, 1e-6, "QuickGELU'(0)");
    }

    #[test]
    fn test_mish_derivative() {
        let k = ActivationKind::Mish;
        // Mish'(0) = tanh(ln2) + 0 * ... = tanh(ln2)
        let expected = (2.0_f32.ln()).tanh();
        assert_close(k.derivative(0.0), expected, 1e-5, "Mish'(0)");
    }

    #[test]
    fn test_gelu_derivative_at_zero() {
        let k = ActivationKind::GELU;
        // GELU'(0) = 0.5 + 0 = 0.5
        assert_close(k.derivative(0.0), 0.5, 1e-3, "GELU'(0)");
    }

    #[test]
    fn test_gelu_tanh_derivative_at_zero() {
        let k = ActivationKind::GELUTanh;
        assert_close(k.derivative(0.0), 0.5, 1e-3, "GELUTanh'(0)");
    }

    // Numerical derivative check for all activations
    #[test]
    fn test_all_derivatives_numerical() {
        let h = 1e-4;
        let tol = 5e-2; // generous tolerance for float32
        for kind in all_kinds() {
            for &x in &[-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0] {
                let analytical = kind.derivative(x);
                let numerical = numerical_derivative(|v| kind.apply(v), x, h);
                assert_close(analytical, numerical, tol, &format!("{kind:?}' at x={x}"));
            }
        }
    }

    // =====================================================================
    // 7. Fused linear + activation matches separate
    // =====================================================================

    #[test]
    fn test_fused_matches_separate_relu() {
        test_fused_matches_separate(ActivationKind::ReLU);
    }

    #[test]
    fn test_fused_matches_separate_silu() {
        test_fused_matches_separate(ActivationKind::SiLU);
    }

    #[test]
    fn test_fused_matches_separate_gelu() {
        test_fused_matches_separate(ActivationKind::GELU);
    }

    #[test]
    fn test_fused_matches_separate_sigmoid() {
        test_fused_matches_separate(ActivationKind::Sigmoid);
    }

    #[test]
    fn test_fused_matches_separate_tanh() {
        test_fused_matches_separate(ActivationKind::Tanh);
    }

    fn test_fused_matches_separate(kind: ActivationKind) {
        let batch = 2;
        let in_f = 3;
        let out_f = 4;
        // Simple weight and bias
        let input: Vec<f32> = (0..batch * in_f).map(|i| i as f32 * 0.1).collect();
        let weight: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32 - 5.0) * 0.1).collect();
        let bias: Vec<f32> = (0..out_f).map(|i| i as f32 * 0.05).collect();

        // Separate: linear then activation
        let mut linear_out = vec![0.0_f32; batch * out_f];
        for b in 0..batch {
            for j in 0..out_f {
                let mut acc = 0.0_f32;
                for k in 0..in_f {
                    acc += input[b * in_f + k] * weight[j * in_f + k];
                }
                acc += bias[j];
                linear_out[b * out_f + j] = acc;
            }
        }
        let mut separate_out = vec![0.0_f32; batch * out_f];
        let kernel = ActivationKernel::new(kind);
        kernel.apply_ref(&linear_out, &mut separate_out).unwrap();

        // Fused
        let fused = FusedActivation::new(kind);
        let mut fused_out = vec![0.0_f32; batch * out_f];
        fused.apply_fused_ref(&input, &weight, &bias, &mut fused_out, batch, in_f, out_f).unwrap();

        for i in 0..batch * out_f {
            assert_close(
                fused_out[i],
                separate_out[i],
                1e-5,
                &format!("{kind:?} fused vs separate [{i}]"),
            );
        }
    }

    #[test]
    fn test_fused_no_bias() {
        let fused = FusedActivation::new(ActivationKind::ReLU);
        let input = vec![1.0, 2.0, 3.0];
        // Identity-like weight [1, 3] → one output per batch
        let weight = vec![1.0, 0.0, 0.0];
        let mut output = vec![0.0_f32; 1];
        fused.apply_fused_ref(&input, &weight, &[], &mut output, 1, 3, 1).unwrap();
        assert_close(output[0], 1.0, 1e-6, "fused no bias");
    }

    #[test]
    fn test_fused_error_on_zero_dims() {
        let fused = FusedActivation::new(ActivationKind::ReLU);
        let result = fused.apply_fused_ref(&[], &[], &[], &mut [], 0, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_error_on_short_output() {
        let fused = FusedActivation::new(ActivationKind::ReLU);
        let input = vec![1.0; 4];
        let weight = vec![1.0; 4];
        let mut output = vec![0.0_f32; 1]; // too short for batch=2, out=2
        let result = fused.apply_fused_ref(&input, &weight, &[], &mut output, 2, 2, 2);
        assert!(result.is_err());
    }

    // =====================================================================
    // 8. Approximate vs exact closeness
    // =====================================================================

    #[test]
    fn test_approx_gelu_closeness() {
        let exact = ActivationKind::GELU;
        for i in -20..=20 {
            let x = i as f32 * 0.1;
            let e = exact.apply(x);
            let a = ApproximateActivation::apply_approx(x, exact);
            if e.abs() > 0.05 {
                let rel = ((e - a) / e).abs();
                assert!(rel < 0.05, "approx GELU rel error {rel} at x={x}");
            }
        }
    }

    #[test]
    fn test_approx_sigmoid_closeness() {
        let kind = ActivationKind::Sigmoid;
        // The piece-wise fast_sigmoid is accurate for moderate range
        for i in -30..=30 {
            let x = i as f32 * 0.1;
            let exact = kind.apply(x);
            let approx = ApproximateActivation::apply_approx(x, kind);
            assert_close(exact, approx, 0.05, &format!("approx sigmoid x={x}"));
        }
    }

    #[test]
    fn test_approx_tanh_closeness() {
        let kind = ActivationKind::Tanh;
        for i in -30..=30 {
            let x = i as f32 * 0.1;
            let exact = kind.apply(x);
            let approx = ApproximateActivation::apply_approx(x, kind);
            assert_close(exact, approx, 0.02, &format!("approx tanh x={x}"));
        }
    }

    #[test]
    fn test_approx_silu_closeness() {
        let kind = ActivationKind::SiLU;
        for i in -30..=30 {
            let x = i as f32 * 0.1;
            let exact = kind.apply(x);
            let approx = ApproximateActivation::apply_approx(x, kind);
            assert_close(exact, approx, 0.06, &format!("approx SiLU x={x}"));
        }
    }

    #[test]
    fn test_approx_mish_closeness() {
        let kind = ActivationKind::Mish;
        for i in -20..=20 {
            let x = i as f32 * 0.1;
            let exact = kind.apply(x);
            let approx = ApproximateActivation::apply_approx(x, kind);
            assert_close(exact, approx, 0.06, &format!("approx Mish x={x}"));
        }
    }

    #[test]
    fn test_approx_relu_is_exact() {
        let kind = ActivationKind::ReLU;
        for i in -50..=50 {
            let x = i as f32 * 0.1;
            assert_eq!(
                kind.apply(x),
                ApproximateActivation::apply_approx(x, kind),
                "approx ReLU should be exact"
            );
        }
    }

    #[test]
    fn test_approx_ref_output() {
        let approx = ApproximateActivation::new(ActivationKind::GELU);
        let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0_f32; input.len()];
        approx.apply_approx_ref(&input, &mut output).unwrap();
        for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
            let expected = ApproximateActivation::apply_approx(x, ActivationKind::GELU);
            assert_eq!(y, expected, "approx ref mismatch at [{i}]");
        }
    }

    // =====================================================================
    // 9. Numerical stability (very large/small inputs)
    // =====================================================================

    #[test]
    fn test_sigmoid_large_inputs() {
        let k = ActivationKind::Sigmoid;
        assert!(!k.apply(100.0).is_nan());
        assert!(!k.apply(-100.0).is_nan());
        assert!(!k.apply(1000.0).is_nan());
        assert!(!k.apply(-1000.0).is_nan());
        assert_close(k.apply(100.0), 1.0, 1e-6, "sigmoid(100)");
        assert_close(k.apply(-100.0), 0.0, 1e-6, "sigmoid(-100)");
    }

    #[test]
    fn test_softplus_large_inputs() {
        let k = ActivationKind::Softplus(1.0);
        assert!(!k.apply(100.0).is_nan());
        assert!(!k.apply(-100.0).is_nan());
        assert!(!k.apply(100.0).is_infinite());
        assert_close(k.apply(100.0), 100.0, 0.1, "softplus(100)");
        assert_close(k.apply(-100.0), 0.0, 1e-6, "softplus(-100)");
    }

    #[test]
    fn test_gelu_extreme_inputs() {
        let k = ActivationKind::GELU;
        assert!(!k.apply(50.0).is_nan());
        assert!(!k.apply(-50.0).is_nan());
        assert!(!k.apply(50.0).is_infinite());
        // GELU(large) ≈ x, GELU(very negative) ≈ 0
        assert_close(k.apply(50.0), 50.0, 0.01, "GELU(50)");
        assert_close(k.apply(-50.0), 0.0, 0.01, "GELU(-50)");
    }

    #[test]
    fn test_elu_large_negative() {
        let k = ActivationKind::ELU(1.0);
        // ELU(very negative) = alpha * (exp(x)-1) ≈ -alpha
        assert!(!k.apply(-100.0).is_nan());
        assert_close(k.apply(-100.0), -1.0, 1e-5, "ELU(-100)");
    }

    #[test]
    fn test_mish_extreme() {
        let k = ActivationKind::Mish;
        assert!(!k.apply(50.0).is_nan());
        assert!(!k.apply(-50.0).is_nan());
        // Mish(large positive) ≈ x, Mish(large negative) ≈ 0
        assert_close(k.apply(50.0), 50.0, 0.1, "Mish(50)");
        assert_close(k.apply(-50.0), 0.0, 0.01, "Mish(-50)");
    }

    #[test]
    fn test_all_activations_no_nan_moderate_range() {
        for kind in all_kinds() {
            for i in -100..=100 {
                let x = i as f32 * 0.1;
                let y = kind.apply(x);
                assert!(!y.is_nan(), "{kind:?}({x}) = NaN");
            }
        }
    }

    #[test]
    fn test_all_activations_no_nan_extreme_range() {
        for kind in all_kinds() {
            for &x in &[-1e6, -1e3, -100.0, 100.0, 1e3, 1e6] {
                let y = kind.apply(x);
                assert!(!y.is_nan(), "{kind:?}({x}) = NaN");
            }
        }
    }

    #[test]
    fn test_derivatives_no_nan_moderate_range() {
        for kind in all_kinds() {
            for i in -100..=100 {
                let x = i as f32 * 0.1;
                let d = kind.derivative(x);
                assert!(!d.is_nan(), "{kind:?}'({x}) = NaN");
            }
        }
    }

    // =====================================================================
    // 10. Property tests
    // =====================================================================

    #[test]
    fn test_sigmoid_monotonic() {
        let k = ActivationKind::Sigmoid;
        let mut prev = k.apply(-50.0);
        for i in -499..=500 {
            let x = i as f32 * 0.1;
            let y = k.apply(x);
            assert!(y >= prev, "Sigmoid not monotonic at x={x}: {prev} > {y}");
            prev = y;
        }
    }

    #[test]
    fn test_tanh_monotonic() {
        let k = ActivationKind::Tanh;
        let mut prev = k.apply(-50.0);
        for i in -499..=500 {
            let x = i as f32 * 0.1;
            let y = k.apply(x);
            assert!(y >= prev, "Tanh not monotonic at x={x}: {prev} > {y}");
            prev = y;
        }
    }

    #[test]
    fn test_hard_sigmoid_monotonic() {
        let k = ActivationKind::HardSigmoid;
        let mut prev = k.apply(-10.0);
        for i in -99..=100 {
            let x = i as f32 * 0.1;
            let y = k.apply(x);
            assert!(y >= prev, "HardSigmoid not monotonic at x={x}: {prev} > {y}");
            prev = y;
        }
    }

    #[test]
    fn test_relu_idempotent() {
        let k = ActivationKind::ReLU;
        for i in -100..=100 {
            let x = i as f32 * 0.1;
            let once = k.apply(x);
            let twice = k.apply(once);
            assert_eq!(once, twice, "ReLU not idempotent at x={x}");
        }
    }

    #[test]
    fn test_sigmoid_idempotent_limit() {
        // sigmoid is NOT idempotent, but sigmoid(sigmoid(x)) ∈ (0.5, ~0.73)
        let k = ActivationKind::Sigmoid;
        for i in -100..=100 {
            let x = i as f32 * 0.1;
            let y = k.apply(k.apply(x));
            assert!((0.0..=1.0).contains(&y), "σ(σ(x)) out of range at x={x}");
        }
    }

    #[test]
    fn test_relu_scale_equivariance() {
        // ReLU(a*x) = a*ReLU(x) for a > 0
        let k = ActivationKind::ReLU;
        let a = 2.5_f32;
        for i in -50..=50 {
            let x = i as f32 * 0.1;
            let lhs = k.apply(a * x);
            let rhs = a * k.apply(x);
            assert_close(lhs, rhs, 1e-6, &format!("ReLU scale equiv x={x}"));
        }
    }

    #[test]
    fn test_softplus_positive() {
        // softplus always > 0
        let k = ActivationKind::Softplus(1.0);
        for i in -100..=100 {
            let x = i as f32 * 0.1;
            assert!(k.apply(x) >= 0.0, "Softplus({x}) should be >= 0, got {}", k.apply(x));
        }
    }

    #[test]
    fn test_elu_continuity_at_zero() {
        let k = ActivationKind::ELU(1.0);
        let eps = 1e-5_f32;
        let left = k.apply(-eps);
        let right = k.apply(eps);
        let center = k.apply(0.0);
        assert_close(left, center, 0.01, "ELU continuous at 0 (left)");
        assert_close(right, center, 0.01, "ELU continuous at 0 (right)");
    }

    // =====================================================================
    // 11. ActivationKernel API tests
    // =====================================================================

    #[test]
    fn test_kernel_apply_ref() {
        let kernel = ActivationKernel::new(ActivationKind::ReLU);
        let input = vec![-1.0, 0.0, 1.0, -0.5, 2.0];
        let mut output = vec![0.0_f32; 5];
        kernel.apply_ref(&input, &mut output).unwrap();
        assert_eq!(output, vec![0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_kernel_apply_inplace() {
        let kernel = ActivationKernel::new(ActivationKind::ReLU);
        let mut data = vec![-1.0, 0.0, 1.0, -0.5, 2.0];
        kernel.apply_inplace(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_kernel_apply_ref_error_short_output() {
        let kernel = ActivationKernel::new(ActivationKind::ReLU);
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0_f32; 2]; // too short
        assert!(kernel.apply_ref(&input, &mut output).is_err());
    }

    #[test]
    fn test_kernel_apply_empty() {
        let kernel = ActivationKernel::new(ActivationKind::GELU);
        let mut output: Vec<f32> = vec![];
        kernel.apply_ref(&[], &mut output).unwrap();
    }

    // =====================================================================
    // 12. ActivationDerivative API tests
    // =====================================================================

    #[test]
    fn test_derivative_ref_api() {
        let d = ActivationDerivative::new(ActivationKind::ReLU);
        let input = vec![-1.0, 0.0, 1.0];
        let mut output = vec![0.0_f32; 3];
        d.derivative_ref(&input, &mut output).unwrap();
        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 1.0);
        assert_eq!(output[2], 1.0);
    }

    #[test]
    fn test_derivative_ref_error_short() {
        let d = ActivationDerivative::new(ActivationKind::Sigmoid);
        let input = vec![1.0; 5];
        let mut output = vec![0.0_f32; 3];
        assert!(d.derivative_ref(&input, &mut output).is_err());
    }

    // =====================================================================
    // 13. ActivationStats tests
    // =====================================================================

    #[test]
    fn test_stats_basic() {
        let input: Vec<f32> = (-10..=10).map(|i| i as f32).collect();
        let mut output = vec![0.0_f32; input.len()];
        let stats =
            ActivationStats::apply_and_collect(ActivationKind::ReLU, &input, &mut output).unwrap();
        assert_eq!(stats.element_count, 21);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);
        assert_close(stats.min_value, 0.0, 1e-6, "stats min");
        assert_close(stats.max_value, 10.0, 1e-6, "stats max");
        // 11 zeros (from ReLU clamping negatives) + 1 for 0 input
        assert_eq!(stats.zero_count, 11);
    }

    #[test]
    fn test_stats_nan_detection() {
        let input = vec![f32::NAN, 1.0, f32::NEG_INFINITY];
        let mut output = vec![0.0_f32; 3];
        let stats =
            ActivationStats::apply_and_collect(ActivationKind::SiLU, &input, &mut output).unwrap();
        assert!(stats.nan_count >= 1, "should detect NaN");
    }

    #[test]
    fn test_stats_throughput() {
        let input: Vec<f32> = (0..10000).map(|i| i as f32 * 0.001).collect();
        let mut output = vec![0.0_f32; input.len()];
        let stats =
            ActivationStats::apply_and_collect(ActivationKind::GELU, &input, &mut output).unwrap();
        assert_eq!(stats.element_count, 10000);
        // Just check it doesn't panic; actual throughput is variable
        let _tp = stats.throughput_meps();
    }

    #[test]
    fn test_stats_has_numerical_issues() {
        let input = vec![0.0, 1.0, 2.0];
        let mut output = vec![0.0_f32; 3];
        let stats =
            ActivationStats::apply_and_collect(ActivationKind::ReLU, &input, &mut output).unwrap();
        assert!(!stats.has_numerical_issues());
    }

    #[test]
    fn test_stats_empty_input() {
        let mut output: Vec<f32> = vec![];
        let stats =
            ActivationStats::apply_and_collect(ActivationKind::ReLU, &[], &mut output).unwrap();
        assert_eq!(stats.element_count, 0);
        assert_eq!(stats.nan_count, 0);
    }

    #[test]
    fn test_stats_error_short_output() {
        let input = vec![1.0; 5];
        let mut output = vec![0.0_f32; 2];
        assert!(
            ActivationStats::apply_and_collect(ActivationKind::ReLU, &input, &mut output,).is_err()
        );
    }

    // =====================================================================
    // 14. OpenCL source sanity
    // =====================================================================

    #[test]
    fn test_opencl_source_contains_all_kernels() {
        let src = ACTIVATIONS_CL;
        for name in &[
            "relu",
            "gelu",
            "gelu_tanh",
            "silu",
            "swish",
            "sigmoid_act",
            "tanh_act",
            "leaky_relu",
            "elu",
            "softplus",
            "mish",
            "hard_swish",
            "hard_sigmoid",
            "quick_gelu",
        ] {
            assert!(
                src.contains(&format!("__kernel void {name}")),
                "ACTIVATIONS_CL missing __kernel void {name}"
            );
        }
    }

    #[test]
    fn test_opencl_source_contains_derivatives() {
        let src = ACTIVATIONS_CL;
        for name in &["relu_deriv", "sigmoid_deriv", "silu_deriv", "tanh_deriv"] {
            assert!(
                src.contains(&format!("__kernel void {name}")),
                "ACTIVATIONS_CL missing derivative kernel {name}"
            );
        }
    }

    #[test]
    fn test_opencl_source_contains_fused() {
        let src = ACTIVATIONS_CL;
        for name in &["fused_linear_relu", "fused_linear_silu", "fused_linear_gelu"] {
            assert!(
                src.contains(&format!("__kernel void {name}")),
                "ACTIVATIONS_CL missing fused kernel {name}"
            );
        }
    }

    // =====================================================================
    // 15. ActivationKind metadata
    // =====================================================================

    #[test]
    fn test_kernel_names_unique() {
        let kinds = all_kinds();
        let names: Vec<&str> = kinds.iter().map(|k| k.kernel_name()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(names.len(), sorted.len(), "kernel names not unique: {names:?}");
    }

    #[test]
    fn test_has_parameter() {
        assert!(!ActivationKind::ReLU.has_parameter());
        assert!(!ActivationKind::GELU.has_parameter());
        assert!(ActivationKind::LeakyReLU(0.01).has_parameter());
        assert!(ActivationKind::ELU(1.0).has_parameter());
        assert!(ActivationKind::Softplus(1.0).has_parameter());
    }

    #[test]
    fn test_parameter_value() {
        assert_eq!(ActivationKind::LeakyReLU(0.01).parameter(), Some(0.01));
        assert_eq!(ActivationKind::ELU(2.0).parameter(), Some(2.0));
        assert_eq!(ActivationKind::Softplus(0.5).parameter(), Some(0.5));
        assert_eq!(ActivationKind::ReLU.parameter(), None);
    }

    // =====================================================================
    // 16. Cross-activation comparisons
    // =====================================================================

    #[test]
    fn test_quick_gelu_vs_gelu_approximate() {
        // QuickGELU ≈ GELU, moderate closeness
        let gelu = ActivationKind::GELU;
        let qgelu = ActivationKind::QuickGELU;
        for i in -20..=20 {
            let x = i as f32 * 0.2;
            let g = gelu.apply(x);
            let q = qgelu.apply(x);
            assert_close(g, q, 0.05, &format!("QuickGELU≈GELU x={x}"));
        }
    }

    #[test]
    fn test_hard_swish_vs_swish() {
        // HardSwish approximates Swish/SiLU
        let silu = ActivationKind::SiLU;
        let hs = ActivationKind::HardSwish;
        for i in -20..=20 {
            let x = i as f32 * 0.2;
            let s = silu.apply(x);
            let h = hs.apply(x);
            assert_close(s, h, 0.25, &format!("HardSwish≈SiLU x={x}"));
        }
    }

    #[test]
    fn test_hard_sigmoid_vs_sigmoid() {
        let sig = ActivationKind::Sigmoid;
        let hsig = ActivationKind::HardSigmoid;
        for i in -20..=20 {
            let x = i as f32 * 0.2;
            let s = sig.apply(x);
            let h = hsig.apply(x);
            assert_close(s, h, 0.25, &format!("HardSig≈Sig x={x}"));
        }
    }

    // =====================================================================
    // 17. Softplus with different betas
    // =====================================================================

    #[test]
    fn test_softplus_beta_scaling() {
        let sp1 = ActivationKind::Softplus(1.0);
        let sp2 = ActivationKind::Softplus(2.0);
        // softplus(0, beta=1) = ln2 ≈ 0.6931
        // softplus(0, beta=2) = ln2/2 ≈ 0.3466
        assert_close(sp1.apply(0.0), 2.0_f32.ln(), 1e-4, "sp(0,β=1)");
        assert_close(sp2.apply(0.0), 2.0_f32.ln() / 2.0, 1e-4, "sp(0,β=2)");
    }

    #[test]
    fn test_softplus_approaches_relu() {
        // As beta→∞, softplus→ReLU. With beta=100, should be close.
        let sp = ActivationKind::Softplus(100.0);
        let relu = ActivationKind::ReLU;
        for &x in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
            assert_close(
                sp.apply(x),
                relu.apply(x),
                0.02,
                &format!("Softplus(β=100) ≈ ReLU at x={x}"),
            );
        }
    }

    // =====================================================================
    // 18. LeakyReLU with various alphas
    // =====================================================================

    #[test]
    fn test_leaky_relu_alpha_zero_is_relu() {
        let lrelu = ActivationKind::LeakyReLU(0.0);
        let relu = ActivationKind::ReLU;
        for i in -50..=50 {
            let x = i as f32 * 0.1;
            assert_eq!(lrelu.apply(x), relu.apply(x), "LeakyReLU(α=0) != ReLU at x={x}");
        }
    }

    #[test]
    fn test_leaky_relu_alpha_one_is_identity() {
        let k = ActivationKind::LeakyReLU(1.0);
        for i in -50..=50 {
            let x = i as f32 * 0.1;
            assert_close(k.apply(x), x, 1e-6, &format!("LeakyReLU(α=1) x={x}"));
        }
    }

    // =====================================================================
    // 19. ELU alpha variations
    // =====================================================================

    #[test]
    fn test_elu_alpha_zero_is_relu() {
        let elu = ActivationKind::ELU(0.0);
        let relu = ActivationKind::ReLU;
        for i in -50..=50 {
            let x = i as f32 * 0.1;
            assert_close(elu.apply(x), relu.apply(x), 1e-6, &format!("ELU(α=0) vs ReLU x={x}"));
        }
    }

    #[test]
    fn test_elu_negative_saturation() {
        let k = ActivationKind::ELU(2.0);
        // ELU(very negative, alpha=2) → -2
        assert_close(k.apply(-100.0), -2.0, 1e-4, "ELU sat");
    }

    // =====================================================================
    // 20. Miscellaneous
    // =====================================================================

    #[test]
    fn test_all_activations_at_zero() {
        // Document the value at x=0 for each activation
        let expected_at_zero: Vec<(ActivationKind, f32)> = vec![
            (ActivationKind::ReLU, 0.0),
            (ActivationKind::GELU, 0.0),
            (ActivationKind::GELUTanh, 0.0),
            (ActivationKind::SiLU, 0.0),
            (ActivationKind::Swish, 0.0),
            (ActivationKind::Sigmoid, 0.5),
            (ActivationKind::Tanh, 0.0),
            (ActivationKind::LeakyReLU(0.01), 0.0),
            (ActivationKind::ELU(1.0), 0.0),
            (ActivationKind::Softplus(1.0), 2.0_f32.ln()),
            (ActivationKind::Mish, 0.0),
            (ActivationKind::HardSwish, 0.0),
            (ActivationKind::HardSigmoid, 0.5),
            (ActivationKind::QuickGELU, 0.0),
        ];
        for (kind, expected) in expected_at_zero {
            assert_close(kind.apply(0.0), expected, 1e-5, &format!("{kind:?}(0)"));
        }
    }

    #[test]
    fn test_mish_vs_silu_close() {
        // Mish ≈ SiLU for small x
        let mish = ActivationKind::Mish;
        let silu = ActivationKind::SiLU;
        for i in -10..=10 {
            let x = i as f32 * 0.1;
            let m = mish.apply(x);
            let s = silu.apply(x);
            assert_close(m, s, 0.15, &format!("Mish≈SiLU x={x}"));
        }
    }

    #[test]
    fn test_gelu_symmetry() {
        // GELU is NOT symmetric, but GELU(x) + GELU(-x) ≈ x for small x?
        // Actually: GELU(-x) ≈ -x*(1-Φ(x)). Test anti-symmetry at 0.
        let k = ActivationKind::GELU;
        assert_close(k.apply(0.0), -k.apply(-0.0), 1e-6, "GELU(0)=-GELU(-0)");
    }

    #[test]
    fn test_activation_kernel_all_kinds() {
        for kind in all_kinds() {
            let kernel = ActivationKernel::new(kind);
            let input = vec![-1.0, 0.0, 1.0];
            let mut output = vec![0.0_f32; 3];
            kernel.apply_ref(&input, &mut output).unwrap();
            for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
                assert_eq!(y, kind.apply(x), "{kind:?} kernel mismatch at [{i}]");
            }
        }
    }

    #[test]
    fn test_derivative_kernel_all_kinds() {
        for kind in all_kinds() {
            let d = ActivationDerivative::new(kind);
            let input = vec![-1.0, 0.0, 1.0];
            let mut output = vec![0.0_f32; 3];
            d.derivative_ref(&input, &mut output).unwrap();
            for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
                assert_eq!(y, kind.derivative(x), "{kind:?} derivative mismatch at [{i}]");
            }
        }
    }

    #[test]
    fn test_approx_error_short_output() {
        let approx = ApproximateActivation::new(ActivationKind::GELU);
        let input = vec![1.0; 5];
        let mut output = vec![0.0_f32; 3];
        assert!(approx.apply_approx_ref(&input, &mut output).is_err());
    }
}
