//! Comprehensive activation function library for Intel Arc A770 GPU.
//!
//! Provides 10 activation functions with CPU reference implementations and
//! OpenCL kernel sources for GPU dispatch. Supports fused activation patterns
//! (e.g. `SiLU(x) * y` for gated FFN) and activation statistics tracking.
//!
//! # Activation functions
//!
//! | Function    | Formula                                             | Range        |
//! |-------------|-----------------------------------------------------|--------------|
//! | ReLU        | `max(0, x)`                                         | `[0, ∞)`     |
//! | GELU        | `x · Φ(x)` (exact erf)                              | `≈ [−0.17, ∞)` |
//! | GELUTanh    | `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`           | `≈ [−0.17, ∞)` |
//! | SiLU        | `x · σ(x)`                                          | `≈ [−0.278, ∞)` |
//! | Swish       | `x · σ(βx)` (parameterized)                         | depends on β |
//! | Sigmoid     | `1 / (1 + exp(−x))`                                 | `(0, 1)`     |
//! | Tanh        | `tanh(x)`                                           | `(−1, 1)`    |
//! | Mish        | `x · tanh(softplus(x))`                             | `≈ [−0.31, ∞)` |
//! | HardSwish   | `x · clamp((x+3)/6, 0, 1)`                          | `[−ε, ∞)`    |
//! | LeakyReLU   | `x if x≥0, αx otherwise`                            | `(−∞, ∞)`    |
//!
//! # CPU reference
//!
//! All functions are available as scalar CPU implementations via
//! [`ActivationType::apply`] and the standalone `apply_*` functions.
//!
//! # OpenCL kernels
//!
//! [`ActivationKernels`] provides OpenCL C source strings for the key
//! activations (SiLU, GELU, ReLU) ready for GPU dispatch.

use std::fmt;

// ---------------------------------------------------------------------------
// Activation type enum
// ---------------------------------------------------------------------------

/// Supported activation functions for transformer inference.
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationType {
    /// Rectified Linear Unit: `max(0, x)`.
    ReLU,
    /// Gaussian Error Linear Unit (exact erf).
    GELU,
    /// GELU with fast tanh approximation.
    GELUTanh,
    /// Sigmoid Linear Unit: `x · σ(x)`.
    SiLU,
    /// Swish: `x · σ(βx)` with configurable beta (β=1 is SiLU).
    Swish,
    /// Sigmoid: `1 / (1 + exp(−x))`.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Mish: `x · tanh(softplus(x))`.
    Mish,
    /// HardSwish: `x · clamp((x+3)/6, 0, 1)`.
    HardSwish,
    /// Leaky ReLU: `max(αx, x)` with configurable negative slope.
    LeakyReLU(f32),
}

impl fmt::Display for ActivationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReLU => write!(f, "ReLU"),
            Self::GELU => write!(f, "GELU"),
            Self::GELUTanh => write!(f, "GELUTanh"),
            Self::SiLU => write!(f, "SiLU"),
            Self::Swish => write!(f, "Swish"),
            Self::Sigmoid => write!(f, "Sigmoid"),
            Self::Tanh => write!(f, "Tanh"),
            Self::Mish => write!(f, "Mish"),
            Self::HardSwish => write!(f, "HardSwish"),
            Self::LeakyReLU(alpha) => write!(f, "LeakyReLU(α={alpha})"),
        }
    }
}

impl ActivationType {
    /// Apply the activation function to a single f32 value.
    #[inline]
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Self::ReLU => apply_relu(x),
            Self::GELU => apply_gelu(x),
            Self::GELUTanh => apply_gelu_tanh(x),
            Self::SiLU => apply_silu(x),
            Self::Swish => apply_silu(x), // Swish with β=1 is SiLU
            Self::Sigmoid => apply_sigmoid(x),
            Self::Tanh => apply_tanh(x),
            Self::Mish => apply_mish(x),
            Self::HardSwish => apply_hard_swish(x),
            Self::LeakyReLU(alpha) => apply_leaky_relu(x, *alpha),
        }
    }

    /// Compute the derivative of the activation function at `x`.
    #[inline]
    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            Self::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::GELU => derivative_gelu(x),
            Self::GELUTanh => derivative_gelu_tanh(x),
            Self::SiLU | Self::Swish => derivative_silu(x),
            Self::Sigmoid => {
                let s = apply_sigmoid(x);
                s * (1.0 - s)
            }
            Self::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Self::Mish => derivative_mish(x),
            Self::HardSwish => {
                if x <= -3.0 {
                    0.0
                } else if x >= 3.0 {
                    1.0
                } else {
                    (2.0 * x + 3.0) / 6.0
                }
            }
            Self::LeakyReLU(alpha) => {
                if x >= 0.0 {
                    1.0
                } else {
                    *alpha
                }
            }
        }
    }

    /// Apply the activation function to a slice, writing results to `output`.
    pub fn apply_slice(&self, input: &[f32], output: &mut [f32]) {
        let n = input.len().min(output.len());
        for i in 0..n {
            output[i] = self.apply(input[i]);
        }
    }

    /// Apply the activation function in-place on a mutable slice.
    pub fn apply_in_place(&self, data: &mut [f32]) {
        for x in data.iter_mut() {
            *x = self.apply(*x);
        }
    }

    /// Returns true if this activation is monotonically non-decreasing.
    pub fn is_monotonic(&self) -> bool {
        matches!(self, Self::ReLU | Self::Sigmoid | Self::Tanh | Self::LeakyReLU(_))
    }
}

// ---------------------------------------------------------------------------
// Scalar CPU reference implementations
// ---------------------------------------------------------------------------

/// Sigmoid: `1 / (1 + exp(−x))`.
#[inline]
pub fn apply_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// ReLU: `max(0, x)`.
#[inline]
pub fn apply_relu(x: f32) -> f32 {
    x.max(0.0)
}

/// GELU (exact): `x · 0.5 · (1 + erf(x / √2))`.
#[inline]
pub fn apply_gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// GELU with tanh approximation: `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`.
#[inline]
pub fn apply_gelu_tanh(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// SiLU (Swish with β=1): `x · σ(x)`.
#[inline]
pub fn apply_silu(x: f32) -> f32 {
    x * apply_sigmoid(x)
}

/// Tanh: `tanh(x)`.
#[inline]
pub fn apply_tanh(x: f32) -> f32 {
    x.tanh()
}

/// Mish: `x · tanh(softplus(x))` where `softplus(x) = ln(1 + exp(x))`.
#[inline]
pub fn apply_mish(x: f32) -> f32 {
    let sp = softplus(x);
    x * sp.tanh()
}

/// HardSwish: `x · clamp((x + 3) / 6, 0, 1)`.
#[inline]
pub fn apply_hard_swish(x: f32) -> f32 {
    x * ((x + 3.0) / 6.0).clamp(0.0, 1.0)
}

/// Leaky ReLU: `max(αx, x)`.
#[inline]
pub fn apply_leaky_relu(x: f32, alpha: f32) -> f32 {
    if x >= 0.0 { x } else { alpha * x }
}

/// Softplus: `ln(1 + exp(x))` with numerical stability.
#[inline]
fn softplus(x: f32) -> f32 {
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
// Derivative helpers
// ---------------------------------------------------------------------------

/// GELU derivative: `0.5(1 + erf(x/√2)) + x · exp(−x²/2) / √(2π)`.
#[inline]
fn derivative_gelu(x: f32) -> f32 {
    let cdf = 0.5 * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f32::consts::PI).sqrt();
    cdf + x * pdf
}

/// GELUTanh derivative (approximation).
#[inline]
fn derivative_gelu_tanh(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    let tanh_val = inner.tanh();
    let sech2 = 1.0 - tanh_val * tanh_val;
    0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x * x)
}

/// SiLU derivative: `σ(x) + x · σ(x) · (1 − σ(x)) = σ(x)(1 + x(1 − σ(x)))`.
#[inline]
fn derivative_silu(x: f32) -> f32 {
    let s = apply_sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

/// Mish derivative.
#[inline]
fn derivative_mish(x: f32) -> f32 {
    let sp = softplus(x);
    let tanh_sp = sp.tanh();
    let sig = apply_sigmoid(x);
    let sech2 = 1.0 - tanh_sp * tanh_sp;
    tanh_sp + x * sig * sech2
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for activation kernel dispatch.
#[derive(Debug, Clone)]
pub struct ActivationConfig {
    /// The activation function to apply.
    pub activation_type: ActivationType,
    /// Whether to apply the activation in-place (overwriting input).
    pub in_place: bool,
    /// Whether to use FP16 (half-precision) for GPU computation.
    pub use_fp16: bool,
    /// Whether activation is fused with an element-wise multiply.
    pub fused_with_mul: bool,
}

impl ActivationConfig {
    /// Create a new configuration for the given activation type.
    pub fn new(activation_type: ActivationType) -> Self {
        Self { activation_type, in_place: false, use_fp16: false, fused_with_mul: false }
    }

    /// Enable in-place activation.
    #[must_use]
    pub fn with_in_place(mut self, in_place: bool) -> Self {
        self.in_place = in_place;
        self
    }

    /// Enable FP16 mode.
    #[must_use]
    pub fn with_fp16(mut self, use_fp16: bool) -> Self {
        self.use_fp16 = use_fp16;
        self
    }

    /// Enable fused activation-multiply.
    #[must_use]
    pub fn with_fused_mul(mut self, fused: bool) -> Self {
        self.fused_with_mul = fused;
        self
    }
}

// ---------------------------------------------------------------------------
// Activation kernels (OpenCL source)
// ---------------------------------------------------------------------------

/// Container for OpenCL kernel source strings per activation function.
#[derive(Debug, Clone)]
pub struct ActivationKernels {
    /// SiLU activation kernel source.
    pub silu: &'static str,
    /// GELU activation kernel source (tanh approximation).
    pub gelu: &'static str,
    /// ReLU activation kernel source.
    pub relu: &'static str,
    /// Fused SiLU-multiply kernel source (for gated FFN).
    pub fused_silu_mul: &'static str,
}

impl ActivationKernels {
    /// Return the default set of OpenCL kernel sources.
    pub fn new() -> Self {
        Self { silu: SILU_CL, gelu: GELU_CL, relu: RELU_CL, fused_silu_mul: FUSED_SILU_MUL_CL }
    }

    /// Get the kernel source for the given activation type.
    ///
    /// Returns `None` for activations without a dedicated OpenCL kernel.
    pub fn source_for(&self, act: &ActivationType) -> Option<&'static str> {
        match act {
            ActivationType::SiLU | ActivationType::Swish => Some(self.silu),
            ActivationType::GELU | ActivationType::GELUTanh => Some(self.gelu),
            ActivationType::ReLU => Some(self.relu),
            _ => None,
        }
    }
}

impl Default for ActivationKernels {
    fn default() -> Self {
        Self::new()
    }
}

/// OpenCL C source for the SiLU activation kernel.
pub const SILU_CL: &str = r#"
__kernel void silu_activation(
    __global const float* input,
    __global float* output,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid < count) {
        float x = input[gid];
        float sigmoid_x = 1.0f / (1.0f + exp(-x));
        output[gid] = x * sigmoid_x;
    }
}

__kernel void silu_activation_inplace(
    __global float* data,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid < count) {
        float x = data[gid];
        float sigmoid_x = 1.0f / (1.0f + exp(-x));
        data[gid] = x * sigmoid_x;
    }
}
"#;

/// OpenCL C source for the GELU activation kernel (tanh approximation).
pub const GELU_CL: &str = r#"
__kernel void gelu_activation(
    __global const float* input,
    __global float* output,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid < count) {
        float x = input[gid];
        float c = 0.7978845608f;  // sqrt(2/pi)
        float inner = c * (x + 0.044715f * x * x * x);
        output[gid] = 0.5f * x * (1.0f + tanh(inner));
    }
}
"#;

/// OpenCL C source for the ReLU activation kernel.
pub const RELU_CL: &str = r#"
__kernel void relu_activation(
    __global const float* input,
    __global float* output,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid < count) {
        output[gid] = fmax(0.0f, input[gid]);
    }
}
"#;

/// OpenCL C source for fused SiLU-multiply (gated FFN pattern).
pub const FUSED_SILU_MUL_CL: &str = r#"
__kernel void fused_silu_mul(
    __global const float* gate,
    __global const float* up,
    __global float* output,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid < count) {
        float g = gate[gid];
        float sigmoid_g = 1.0f / (1.0f + exp(-g));
        output[gid] = g * sigmoid_g * up[gid];
    }
}
"#;

// ---------------------------------------------------------------------------
// Activation statistics
// ---------------------------------------------------------------------------

/// Statistics about the output of an activation function.
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Minimum output value.
    pub min_output: f32,
    /// Maximum output value.
    pub max_output: f32,
    /// Mean output value.
    pub mean: f32,
    /// Fraction of outputs that are exactly zero (sparsity %).
    pub sparsity: f32,
    /// Number of outputs that are saturated (at extreme bounds).
    pub saturated_count: usize,
}

impl ActivationStats {
    /// Compute activation statistics for a slice of values.
    pub fn compute(data: &[f32], saturation_threshold: f32) -> Self {
        if data.is_empty() {
            return Self {
                min_output: 0.0,
                max_output: 0.0,
                mean: 0.0,
                sparsity: 0.0,
                saturated_count: 0,
            };
        }

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut sum = 0.0_f64;
        let mut zero_count = 0_usize;
        let mut saturated = 0_usize;

        for &v in data {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
            sum += v as f64;
            if v == 0.0 {
                zero_count += 1;
            }
            if v.abs() >= saturation_threshold {
                saturated += 1;
            }
        }

        let n = data.len() as f64;
        Self {
            min_output: min_val,
            max_output: max_val,
            mean: (sum / n) as f32,
            sparsity: (zero_count as f32 / data.len() as f32) * 100.0,
            saturated_count: saturated,
        }
    }
}

impl fmt::Display for ActivationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "min={:.4} max={:.4} mean={:.4} sparsity={:.1}% saturated={}",
            self.min_output, self.max_output, self.mean, self.sparsity, self.saturated_count
        )
    }
}

// ---------------------------------------------------------------------------
// Fused activation
// ---------------------------------------------------------------------------

/// Fused activation with element-wise multiply: `activation(gate) * up`.
///
/// Used in gated FFN blocks (SwiGLU pattern): `SiLU(gate_proj(x)) * up_proj(x)`.
#[derive(Debug, Clone)]
pub struct FusedActivation {
    /// The activation function to apply to the gate input.
    pub activation: ActivationType,
}

impl FusedActivation {
    /// Create a new fused activation.
    pub fn new(activation: ActivationType) -> Self {
        Self { activation }
    }

    /// Apply the fused activation: `activation(gate[i]) * up[i]` for each element.
    ///
    /// The `gate` and `up` slices must have the same length. Results are written
    /// to `output`.
    pub fn apply(&self, gate: &[f32], up: &[f32], output: &mut [f32]) {
        let n = gate.len().min(up.len()).min(output.len());
        for i in 0..n {
            output[i] = self.activation.apply(gate[i]) * up[i];
        }
    }

    /// Apply the fused activation in-place, writing results into `gate`.
    pub fn apply_in_place(&self, gate: &mut [f32], up: &[f32]) {
        let n = gate.len().min(up.len());
        for i in 0..n {
            gate[i] = self.activation.apply(gate[i]) * up[i];
        }
    }
}

// ---------------------------------------------------------------------------
// FP16 simulation helpers
// ---------------------------------------------------------------------------

/// Simulate FP16 precision by rounding to half-precision range.
///
/// This truncates mantissa bits to approximate FP16 behaviour on CPU
/// for accuracy comparison testing.
#[inline]
pub fn simulate_fp16(x: f32) -> f32 {
    // fp16: 1 sign + 5 exponent + 10 mantissa bits
    // Truncate the lower 13 mantissa bits of f32
    let bits = x.to_bits();
    let truncated = bits & 0xFFFF_E000;
    f32::from_bits(truncated)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const FP16_TOL: f32 = 1e-2;

    fn assert_near(a: f32, b: f32, tol: f32, msg: &str) {
        assert!((a - b).abs() < tol, "{msg}: got {a}, expected {b} (tol={tol})");
    }

    // =======================================================================
    // ReLU tests
    // =======================================================================

    #[test]
    fn test_relu_zero() {
        assert_eq!(apply_relu(0.0), 0.0);
    }

    #[test]
    fn test_relu_positive() {
        assert_eq!(apply_relu(3.5), 3.5);
    }

    #[test]
    fn test_relu_negative() {
        assert_eq!(apply_relu(-2.0), 0.0);
    }

    #[test]
    fn test_relu_large_positive() {
        assert_eq!(apply_relu(1e6), 1e6);
    }

    // =======================================================================
    // GELU tests
    // =======================================================================

    #[test]
    fn test_gelu_zero() {
        assert_near(apply_gelu(0.0), 0.0, EPS, "GELU(0)");
    }

    #[test]
    fn test_gelu_positive() {
        // GELU(1) ≈ 0.8413
        assert_near(apply_gelu(1.0), 0.8413, 1e-3, "GELU(1)");
    }

    #[test]
    fn test_gelu_negative() {
        // GELU(-1) ≈ -0.1587
        assert_near(apply_gelu(-1.0), -0.1587, 1e-3, "GELU(-1)");
    }

    // =======================================================================
    // GELUTanh tests
    // =======================================================================

    #[test]
    fn test_gelu_tanh_zero() {
        assert_near(apply_gelu_tanh(0.0), 0.0, EPS, "GELUTanh(0)");
    }

    #[test]
    fn test_gelu_tanh_positive() {
        assert_near(apply_gelu_tanh(1.0), 0.8412, 1e-3, "GELUTanh(1)");
    }

    #[test]
    fn test_gelu_tanh_vs_exact() {
        // The tanh approximation should be close to exact GELU
        for &x in &[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let exact = apply_gelu(x);
            let approx = apply_gelu_tanh(x);
            assert_near(approx, exact, 5e-3, &format!("GELUTanh vs GELU at x={x}"));
        }
    }

    // =======================================================================
    // SiLU tests
    // =======================================================================

    #[test]
    fn test_silu_zero() {
        assert_near(apply_silu(0.0), 0.0, EPS, "SiLU(0)");
    }

    #[test]
    fn test_silu_positive() {
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.7311
        assert_near(apply_silu(1.0), 0.7311, 1e-3, "SiLU(1)");
    }

    #[test]
    fn test_silu_negative() {
        // SiLU(-1) = -1 * sigmoid(-1) ≈ -0.2689
        assert_near(apply_silu(-1.0), -0.2689, 1e-3, "SiLU(-1)");
    }

    #[test]
    fn test_silu_large() {
        assert_near(apply_silu(10.0), 10.0, 1e-3, "SiLU(10)");
    }

    // =======================================================================
    // Sigmoid tests
    // =======================================================================

    #[test]
    fn test_sigmoid_zero() {
        assert_near(apply_sigmoid(0.0), 0.5, EPS, "sigmoid(0)");
    }

    #[test]
    fn test_sigmoid_positive() {
        assert!(apply_sigmoid(5.0) > 0.99);
    }

    #[test]
    fn test_sigmoid_negative() {
        assert!(apply_sigmoid(-5.0) < 0.01);
    }

    #[test]
    fn test_sigmoid_bounds() {
        // Sigmoid output must always be in [0, 1]
        for &x in &[-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let y = apply_sigmoid(x);
            assert!(y >= 0.0 && y <= 1.0, "sigmoid({x}) = {y} not in [0,1]");
        }
        // Moderate values are strictly inside (0, 1)
        for &x in &[-10.0, -1.0, 0.0, 1.0, 10.0] {
            let y = apply_sigmoid(x);
            assert!(y > 0.0 && y < 1.0, "sigmoid({x}) = {y} not in (0,1)");
        }
    }

    // =======================================================================
    // Tanh tests
    // =======================================================================

    #[test]
    fn test_tanh_zero() {
        assert_near(apply_tanh(0.0), 0.0, EPS, "tanh(0)");
    }

    #[test]
    fn test_tanh_bounds() {
        // Tanh output must always be in [-1, 1]
        for &x in &[-100.0, -5.0, -1.0, 0.0, 1.0, 5.0, 100.0] {
            let y = apply_tanh(x);
            assert!(y >= -1.0 && y <= 1.0, "tanh({x}) = {y} not in [-1,1]");
        }
        // Moderate values are strictly inside (-1, 1)
        for &x in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
            let y = apply_tanh(x);
            assert!(y > -1.0 && y < 1.0, "tanh({x}) = {y} not in (-1,1)");
        }
    }

    #[test]
    fn test_tanh_symmetry() {
        assert_near(apply_tanh(1.0), -apply_tanh(-1.0), EPS, "tanh symmetry");
    }

    // =======================================================================
    // Mish tests
    // =======================================================================

    #[test]
    fn test_mish_zero() {
        assert_near(apply_mish(0.0), 0.0, EPS, "Mish(0)");
    }

    #[test]
    fn test_mish_positive() {
        // Mish(1) = 1 * tanh(softplus(1)) = tanh(ln(1+e)) ≈ 0.8651
        assert_near(apply_mish(1.0), 0.8651, 1e-3, "Mish(1)");
    }

    #[test]
    fn test_mish_negative() {
        // Mish(-1) ≈ -0.3034
        assert_near(apply_mish(-1.0), -0.3034, 1e-3, "Mish(-1)");
    }

    // =======================================================================
    // HardSwish tests
    // =======================================================================

    #[test]
    fn test_hard_swish_zero() {
        assert_near(apply_hard_swish(0.0), 0.0, EPS, "HardSwish(0)");
    }

    #[test]
    fn test_hard_swish_below_minus3() {
        assert_eq!(apply_hard_swish(-4.0), 0.0);
    }

    #[test]
    fn test_hard_swish_above_3() {
        assert_eq!(apply_hard_swish(5.0), 5.0);
    }

    #[test]
    fn test_hard_swish_midrange() {
        // HardSwish(1) = 1 * clamp((1+3)/6, 0, 1) = 1 * 4/6 ≈ 0.6667
        assert_near(apply_hard_swish(1.0), 1.0 * (4.0 / 6.0), EPS, "HardSwish(1)");
    }

    // =======================================================================
    // LeakyReLU tests
    // =======================================================================

    #[test]
    fn test_leaky_relu_positive() {
        assert_eq!(apply_leaky_relu(3.0, 0.01), 3.0);
    }

    #[test]
    fn test_leaky_relu_negative() {
        assert_near(apply_leaky_relu(-2.0, 0.01), -0.02, EPS, "LeakyReLU(-2)");
    }

    #[test]
    fn test_leaky_relu_zero() {
        assert_eq!(apply_leaky_relu(0.0, 0.01), 0.0);
    }

    #[test]
    fn test_leaky_relu_custom_alpha() {
        assert_near(apply_leaky_relu(-1.0, 0.2), -0.2, EPS, "LeakyReLU(-1, α=0.2)");
    }

    // =======================================================================
    // Derivative tests
    // =======================================================================

    #[test]
    fn test_relu_derivative() {
        let act = ActivationType::ReLU;
        assert_eq!(act.derivative(1.0), 1.0);
        assert_eq!(act.derivative(-1.0), 0.0);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let act = ActivationType::Sigmoid;
        // d/dx sigmoid(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        assert_near(act.derivative(0.0), 0.25, EPS, "sigmoid'(0)");
    }

    #[test]
    fn test_tanh_derivative() {
        let act = ActivationType::Tanh;
        // d/dx tanh(0) = 1 - tanh²(0) = 1
        assert_near(act.derivative(0.0), 1.0, EPS, "tanh'(0)");
    }

    #[test]
    fn test_silu_derivative_at_zero() {
        let act = ActivationType::SiLU;
        // SiLU'(0) = σ(0)(1 + 0·(1 − σ(0))) = 0.5
        assert_near(act.derivative(0.0), 0.5, EPS, "SiLU'(0)");
    }

    #[test]
    fn test_gelu_derivative_at_zero() {
        let act = ActivationType::GELU;
        // GELU'(0) = 0.5 + 0 = 0.5
        assert_near(act.derivative(0.0), 0.5, EPS, "GELU'(0)");
    }

    #[test]
    fn test_gelu_tanh_derivative_at_zero() {
        let act = ActivationType::GELUTanh;
        assert_near(act.derivative(0.0), 0.5, EPS, "GELUTanh'(0)");
    }

    #[test]
    fn test_hard_swish_derivative_regions() {
        let act = ActivationType::HardSwish;
        assert_eq!(act.derivative(-4.0), 0.0);
        assert_eq!(act.derivative(4.0), 1.0);
        // At x=0: (2*0 + 3)/6 = 0.5
        assert_near(act.derivative(0.0), 0.5, EPS, "HardSwish'(0)");
    }

    #[test]
    fn test_leaky_relu_derivative() {
        let act = ActivationType::LeakyReLU(0.01);
        assert_eq!(act.derivative(1.0), 1.0);
        assert_near(act.derivative(-1.0), 0.01, EPS, "LeakyReLU'(-1)");
    }

    #[test]
    fn test_mish_derivative_at_zero() {
        let act = ActivationType::Mish;
        // Mish'(0) = tanh(ln2) + 0 * ... = tanh(0.6931) ≈ 0.6
        let d = act.derivative(0.0);
        assert!(d > 0.5 && d < 0.7, "Mish'(0)={d}");
    }

    // =======================================================================
    // Fused activation tests
    // =======================================================================

    #[test]
    fn test_fused_silu_mul() {
        let fused = FusedActivation::new(ActivationType::SiLU);
        let gate = [0.0, 1.0, -1.0, 2.0];
        let up = [1.0, 2.0, 3.0, 0.5];
        let mut output = [0.0_f32; 4];
        fused.apply(&gate, &up, &mut output);

        // output[i] = SiLU(gate[i]) * up[i]
        for i in 0..4 {
            let expected = apply_silu(gate[i]) * up[i];
            assert_near(output[i], expected, EPS, &format!("fused[{i}]"));
        }
    }

    #[test]
    fn test_fused_gelu_mul() {
        let fused = FusedActivation::new(ActivationType::GELU);
        let gate = [1.0, -0.5, 0.0];
        let up = [2.0, 1.0, 3.0];
        let mut output = [0.0_f32; 3];
        fused.apply(&gate, &up, &mut output);

        for i in 0..3 {
            let expected = apply_gelu(gate[i]) * up[i];
            assert_near(output[i], expected, EPS, &format!("fused_gelu[{i}]"));
        }
    }

    #[test]
    fn test_fused_in_place() {
        let fused = FusedActivation::new(ActivationType::SiLU);
        let mut gate = [1.0, 2.0, -1.0];
        let up = [0.5, 1.0, 2.0];
        let expected: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| apply_silu(g) * u).collect();
        fused.apply_in_place(&mut gate, &up);
        for i in 0..3 {
            assert_near(gate[i], expected[i], EPS, &format!("in_place[{i}]"));
        }
    }

    // =======================================================================
    // In-place vs out-of-place equivalence
    // =======================================================================

    #[test]
    fn test_in_place_vs_out_of_place() {
        let input = [-2.0, -1.0, 0.0, 0.5, 1.0, 3.0];
        let act = ActivationType::SiLU;

        let mut out = vec![0.0_f32; input.len()];
        act.apply_slice(&input, &mut out);

        let mut in_place = input.to_vec();
        act.apply_in_place(&mut in_place);

        for i in 0..input.len() {
            assert_near(out[i], in_place[i], EPS, &format!("in_place_eq[{i}]"));
        }
    }

    #[test]
    fn test_in_place_vs_out_of_place_all_activations() {
        let input = [-2.0, -1.0, 0.0, 0.5, 1.0, 3.0];
        let activations = vec![
            ActivationType::ReLU,
            ActivationType::GELU,
            ActivationType::GELUTanh,
            ActivationType::SiLU,
            ActivationType::Swish,
            ActivationType::Sigmoid,
            ActivationType::Tanh,
            ActivationType::Mish,
            ActivationType::HardSwish,
            ActivationType::LeakyReLU(0.01),
        ];

        for act in &activations {
            let mut out = vec![0.0_f32; input.len()];
            act.apply_slice(&input, &mut out);

            let mut in_place = input.to_vec();
            act.apply_in_place(&mut in_place);

            for i in 0..input.len() {
                assert_near(out[i], in_place[i], EPS, &format!("{act} in_place_eq[{i}]"));
            }
        }
    }

    // =======================================================================
    // FP16 vs FP32 accuracy bounds
    // =======================================================================

    #[test]
    fn test_fp16_silu_accuracy() {
        for &x in &[-2.0, -1.0, 0.0, 0.5, 1.0, 3.0] {
            let fp32 = apply_silu(x);
            let fp16_x = simulate_fp16(x);
            let fp16_result = apply_silu(fp16_x);
            assert_near(fp16_result, fp32, FP16_TOL, &format!("FP16 SiLU({x})"));
        }
    }

    #[test]
    fn test_fp16_gelu_accuracy() {
        for &x in &[-2.0, -1.0, 0.0, 0.5, 1.0, 3.0] {
            let fp32 = apply_gelu(x);
            let fp16_x = simulate_fp16(x);
            let fp16_result = apply_gelu(fp16_x);
            assert_near(fp16_result, fp32, FP16_TOL, &format!("FP16 GELU({x})"));
        }
    }

    #[test]
    fn test_fp16_sigmoid_accuracy() {
        for &x in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
            let fp32 = apply_sigmoid(x);
            let fp16_x = simulate_fp16(x);
            let fp16_result = apply_sigmoid(fp16_x);
            assert_near(fp16_result, fp32, FP16_TOL, &format!("FP16 sigmoid({x})"));
        }
    }

    // =======================================================================
    // Activation statistics
    // =======================================================================

    #[test]
    fn test_stats_basic() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0];
        let stats = ActivationStats::compute(&data, 10.0);
        assert_eq!(stats.min_output, 0.0);
        assert_eq!(stats.max_output, 4.0);
        assert_near(stats.mean, 2.0, EPS, "stats mean");
        assert_near(stats.sparsity, 20.0, EPS, "stats sparsity");
        assert_eq!(stats.saturated_count, 0);
    }

    #[test]
    fn test_stats_all_zeros() {
        let data = [0.0; 5];
        let stats = ActivationStats::compute(&data, 1.0);
        assert_near(stats.sparsity, 100.0, EPS, "sparsity");
        assert_near(stats.mean, 0.0, EPS, "mean");
    }

    #[test]
    fn test_stats_saturation() {
        let data = [0.0, 0.5, 0.99, 1.0, 1.5];
        let stats = ActivationStats::compute(&data, 1.0);
        assert_eq!(stats.saturated_count, 2); // 1.0 and 1.5
    }

    #[test]
    fn test_stats_empty() {
        let data: [f32; 0] = [];
        let stats = ActivationStats::compute(&data, 1.0);
        assert_eq!(stats.sparsity, 0.0);
        assert_eq!(stats.saturated_count, 0);
    }

    #[test]
    fn test_stats_after_relu() {
        let input = [-3.0, -1.0, 0.0, 1.0, 2.0];
        let output: Vec<f32> = input.iter().map(|&x| apply_relu(x)).collect();
        let stats = ActivationStats::compute(&output, 10.0);
        // 3 zeros: relu(-3), relu(-1), relu(0)
        assert_near(stats.sparsity, 60.0, EPS, "relu sparsity");
        assert_eq!(stats.min_output, 0.0);
        assert_eq!(stats.max_output, 2.0);
    }

    #[test]
    fn test_stats_display() {
        let stats = ActivationStats {
            min_output: -0.5,
            max_output: 3.0,
            mean: 1.25,
            sparsity: 10.0,
            saturated_count: 2,
        };
        let s = format!("{stats}");
        assert!(s.contains("min="));
        assert!(s.contains("max="));
        assert!(s.contains("sparsity="));
    }

    // =======================================================================
    // Special values (NaN, Inf, zero, large negative)
    // =======================================================================

    #[test]
    fn test_relu_nan_returns_zero_or_nan() {
        // f32::max(0, NaN) behaviour is platform-dependent.
        // On x86 it may return 0.0; we accept either.
        let result = apply_relu(f32::NAN);
        assert!(result.is_nan() || result == 0.0, "ReLU(NaN) should be NaN or 0, got {result}");
    }

    #[test]
    fn test_sigmoid_inf() {
        assert_near(apply_sigmoid(f32::INFINITY), 1.0, EPS, "sigmoid(+inf)");
        assert_near(apply_sigmoid(f32::NEG_INFINITY), 0.0, EPS, "sigmoid(-inf)");
    }

    #[test]
    fn test_tanh_inf() {
        assert_near(apply_tanh(f32::INFINITY), 1.0, EPS, "tanh(+inf)");
        assert_near(apply_tanh(f32::NEG_INFINITY), -1.0, EPS, "tanh(-inf)");
    }

    #[test]
    fn test_silu_large_negative() {
        // SiLU(-100) ≈ -100 * sigmoid(-100) ≈ 0
        let result = apply_silu(-100.0);
        assert!(result.abs() < 1e-10, "SiLU(-100) should be ~0, got {result}");
    }

    #[test]
    fn test_gelu_large_negative() {
        let result = apply_gelu(-10.0);
        assert!(result.abs() < 1e-3, "GELU(-10) should be ~0, got {result}");
    }

    #[test]
    fn test_hard_swish_at_minus3() {
        assert_near(apply_hard_swish(-3.0), 0.0, EPS, "HardSwish(-3)");
    }

    #[test]
    fn test_hard_swish_at_3() {
        assert_near(apply_hard_swish(3.0), 3.0, EPS, "HardSwish(3)");
    }

    // =======================================================================
    // Monotonicity property tests
    // =======================================================================

    #[test]
    fn test_relu_monotonic() {
        let values: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        for w in values.windows(2) {
            assert!(
                apply_relu(w[1]) >= apply_relu(w[0]),
                "ReLU not monotonic at {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_sigmoid_monotonic() {
        let values: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        for w in values.windows(2) {
            assert!(
                apply_sigmoid(w[1]) >= apply_sigmoid(w[0]),
                "sigmoid not monotonic at {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_tanh_monotonic() {
        let values: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        for w in values.windows(2) {
            assert!(
                apply_tanh(w[1]) >= apply_tanh(w[0]),
                "tanh not monotonic at {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_leaky_relu_monotonic() {
        let values: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        for w in values.windows(2) {
            assert!(
                apply_leaky_relu(w[1], 0.01) >= apply_leaky_relu(w[0], 0.01),
                "LeakyReLU not monotonic at {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    // =======================================================================
    // Output range bounds (property tests)
    // =======================================================================

    #[test]
    fn test_sigmoid_output_range() {
        let test_values = [-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0];
        for &x in &test_values {
            let y = apply_sigmoid(x);
            assert!(y >= 0.0 && y <= 1.0, "sigmoid({x}) = {y} outside [0,1]");
        }
    }

    #[test]
    fn test_tanh_output_range() {
        let test_values = [-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0];
        for &x in &test_values {
            let y = apply_tanh(x);
            assert!(y >= -1.0 && y <= 1.0, "tanh({x}) = {y} outside [-1,1]");
        }
    }

    #[test]
    fn test_relu_output_non_negative() {
        let test_values = [-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0];
        for &x in &test_values {
            let y = apply_relu(x);
            assert!(y >= 0.0, "ReLU({x}) = {y} is negative");
        }
    }

    #[test]
    fn test_hard_swish_non_negative_for_positive() {
        for i in 0..100 {
            let x = i as f32 * 0.1;
            let y = apply_hard_swish(x);
            assert!(y >= 0.0, "HardSwish({x}) = {y} is negative");
        }
    }

    // =======================================================================
    // ActivationType enum tests
    // =======================================================================

    #[test]
    fn test_activation_type_apply() {
        // Verify the enum dispatch matches the standalone functions
        let x = 1.5_f32;
        assert_eq!(ActivationType::ReLU.apply(x), apply_relu(x));
        assert_eq!(ActivationType::GELU.apply(x), apply_gelu(x));
        assert_eq!(ActivationType::GELUTanh.apply(x), apply_gelu_tanh(x));
        assert_eq!(ActivationType::SiLU.apply(x), apply_silu(x));
        assert_eq!(ActivationType::Swish.apply(x), apply_silu(x));
        assert_eq!(ActivationType::Sigmoid.apply(x), apply_sigmoid(x));
        assert_eq!(ActivationType::Tanh.apply(x), apply_tanh(x));
        assert_eq!(ActivationType::Mish.apply(x), apply_mish(x));
        assert_eq!(ActivationType::HardSwish.apply(x), apply_hard_swish(x));
        assert_eq!(ActivationType::LeakyReLU(0.01).apply(x), apply_leaky_relu(x, 0.01));
    }

    #[test]
    fn test_activation_type_display() {
        assert_eq!(format!("{}", ActivationType::ReLU), "ReLU");
        assert_eq!(format!("{}", ActivationType::SiLU), "SiLU");
        assert_eq!(format!("{}", ActivationType::LeakyReLU(0.01)), "LeakyReLU(α=0.01)");
    }

    #[test]
    fn test_is_monotonic() {
        assert!(ActivationType::ReLU.is_monotonic());
        assert!(ActivationType::Sigmoid.is_monotonic());
        assert!(ActivationType::Tanh.is_monotonic());
        assert!(ActivationType::LeakyReLU(0.01).is_monotonic());
        assert!(!ActivationType::SiLU.is_monotonic());
        assert!(!ActivationType::GELU.is_monotonic());
        assert!(!ActivationType::Mish.is_monotonic());
    }

    // =======================================================================
    // Configuration tests
    // =======================================================================

    #[test]
    fn test_activation_config_defaults() {
        let config = ActivationConfig::new(ActivationType::SiLU);
        assert!(!config.in_place);
        assert!(!config.use_fp16);
        assert!(!config.fused_with_mul);
    }

    #[test]
    fn test_activation_config_builder() {
        let config = ActivationConfig::new(ActivationType::GELU)
            .with_in_place(true)
            .with_fp16(true)
            .with_fused_mul(true);
        assert!(config.in_place);
        assert!(config.use_fp16);
        assert!(config.fused_with_mul);
    }

    // =======================================================================
    // Kernel source tests
    // =======================================================================

    #[test]
    fn test_kernel_sources_non_empty() {
        let kernels = ActivationKernels::new();
        assert!(!kernels.silu.is_empty());
        assert!(!kernels.gelu.is_empty());
        assert!(!kernels.relu.is_empty());
        assert!(!kernels.fused_silu_mul.is_empty());
    }

    #[test]
    fn test_kernel_source_for() {
        let kernels = ActivationKernels::new();
        assert!(kernels.source_for(&ActivationType::SiLU).is_some());
        assert!(kernels.source_for(&ActivationType::GELU).is_some());
        assert!(kernels.source_for(&ActivationType::ReLU).is_some());
        assert!(kernels.source_for(&ActivationType::Swish).is_some());
        assert!(kernels.source_for(&ActivationType::GELUTanh).is_some());
        // No dedicated kernel for these
        assert!(kernels.source_for(&ActivationType::Tanh).is_none());
        assert!(kernels.source_for(&ActivationType::Mish).is_none());
    }

    #[test]
    fn test_silu_kernel_contains_function() {
        assert!(SILU_CL.contains("silu_activation"));
        assert!(SILU_CL.contains("sigmoid"));
    }

    #[test]
    fn test_gelu_kernel_contains_function() {
        assert!(GELU_CL.contains("gelu_activation"));
        assert!(GELU_CL.contains("tanh"));
    }

    #[test]
    fn test_relu_kernel_contains_function() {
        assert!(RELU_CL.contains("relu_activation"));
        assert!(RELU_CL.contains("fmax"));
    }

    #[test]
    fn test_fused_kernel_contains_function() {
        assert!(FUSED_SILU_MUL_CL.contains("fused_silu_mul"));
        assert!(FUSED_SILU_MUL_CL.contains("gate"));
    }

    // =======================================================================
    // FP16 simulation
    // =======================================================================

    #[test]
    fn test_fp16_roundtrip() {
        let x = 1.0_f32;
        assert_eq!(simulate_fp16(x), 1.0);
    }

    #[test]
    fn test_fp16_zero() {
        assert_eq!(simulate_fp16(0.0), 0.0);
    }

    #[test]
    fn test_fp16_precision_loss() {
        // FP16 has ~3 decimal digits of precision
        let x = 1.0009765625_f32; // 1 + 2^-10, exactly representable in fp16
        let fp16 = simulate_fp16(x);
        assert!((fp16 - x).abs() < 1e-3, "FP16({x}) = {fp16}");
    }

    // =======================================================================
    // Numerical accuracy: fast approximation vs reference
    // =======================================================================

    #[test]
    fn test_gelu_tanh_max_error() {
        let mut max_err = 0.0_f32;
        for i in -1000..=1000 {
            let x = i as f32 * 0.01;
            let exact = apply_gelu(x);
            let approx = apply_gelu_tanh(x);
            let err = (exact - approx).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 0.02, "Max GELU tanh approximation error {max_err} exceeds 0.02");
    }

    #[test]
    fn test_hard_swish_vs_silu_similarity() {
        // HardSwish is a linear approximation of SiLU — they should be
        // reasonably close in the [-3, 3] range
        let mut max_err = 0.0_f32;
        for i in -30..=30 {
            let x = i as f32 * 0.1;
            let hs = apply_hard_swish(x);
            let silu = apply_silu(x);
            let err = (hs - silu).abs();
            if err > max_err {
                max_err = err;
            }
        }
        // They're not identical, but should be within reasonable bounds
        assert!(max_err < 0.2, "Max HardSwish vs SiLU error {max_err} exceeds 0.2");
    }

    // =======================================================================
    // apply_slice tests
    // =======================================================================

    #[test]
    fn test_apply_slice_relu() {
        let input = [-1.0, 0.0, 1.0, 2.0];
        let mut output = [0.0_f32; 4];
        ActivationType::ReLU.apply_slice(&input, &mut output);
        assert_eq!(output, [0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_apply_slice_partial() {
        // Output shorter than input — only fills up to min(len)
        let input = [-1.0, 0.0, 1.0, 2.0];
        let mut output = [0.0_f32; 2];
        ActivationType::ReLU.apply_slice(&input, &mut output);
        assert_eq!(output, [0.0, 0.0]);
    }
}
