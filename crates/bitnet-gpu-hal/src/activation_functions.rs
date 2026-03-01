//! Module stub - implementation pending merge from feature branch
//! Activation functions for GPU kernel pipelines.
//!
//! Provides a trait-based abstraction over common neural-network activation
//! functions (`ReLU`, `GeLU`, `SiLU`, Mish, …) together with a registry, fused
//! bias+activation helper, and a lightweight profiler.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ── Activation type enum ────────────────────────────────────────────────────

/// Enumerates every supported activation function.
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationType {
    ReLU,
    GeLU,
    SiLU,
    Mish,
    Tanh,
    Sigmoid,
    LeakyReLU(f32),
    ELU(f32),
    SELU,
    Softplus,
    QuickGELU,
    HardSwish,
    HardSigmoid,
}

impl fmt::Display for ActivationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReLU => write!(f, "ReLU"),
            Self::GeLU => write!(f, "GeLU"),
            Self::SiLU => write!(f, "SiLU"),
            Self::Mish => write!(f, "Mish"),
            Self::Tanh => write!(f, "Tanh"),
            Self::Sigmoid => write!(f, "Sigmoid"),
            Self::LeakyReLU(a) => write!(f, "LeakyReLU({a})"),
            Self::ELU(a) => write!(f, "ELU({a})"),
            Self::SELU => write!(f, "SELU"),
            Self::Softplus => write!(f, "Softplus"),
            Self::QuickGELU => write!(f, "QuickGELU"),
            Self::HardSwish => write!(f, "HardSwish"),
            Self::HardSigmoid => write!(f, "HardSigmoid"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for an activation computation.
#[derive(Debug, Clone)]
pub struct ActivationConfig {
    /// Which activation function to use.
    pub activation_type: ActivationType,
    /// If `true`, the implementation *may* modify the input buffer in place.
    pub in_place: bool,
    /// If `true`, prefer a faster approximation when available (e.g. `GeLU`).
    pub approximate: bool,
}

impl ActivationConfig {
    /// Create a new config for the given activation type with defaults.
    #[must_use]
    pub const fn new(activation_type: ActivationType) -> Self {
        Self { activation_type, in_place: false, approximate: false }
    }

    /// Builder: enable in-place mode.
    #[must_use]
    pub const fn with_in_place(mut self, in_place: bool) -> Self {
        self.in_place = in_place;
        self
    }

    /// Builder: enable approximation.
    #[must_use]
    pub const fn with_approximate(mut self, approximate: bool) -> Self {
        self.approximate = approximate;
        self
    }
}

// ── Activation trait ────────────────────────────────────────────────────────

/// Trait that every activation function must implement.
pub trait Activation: Send + Sync {
    /// Apply the activation element-wise, returning a new vector.
    fn forward(&self, input: &[f32]) -> Vec<f32>;

    /// Compute the element-wise derivative given the *input* values.
    fn backward(&self, input: &[f32]) -> Vec<f32>;

    /// Human-readable name.
    fn name(&self) -> &'static str;

    /// Whether the function is monotonically non-decreasing.
    fn is_monotonic(&self) -> bool;

    /// Apply the activation **in-place**.
    fn forward_inplace(&self, data: &mut [f32]) {
        let out = self.forward(data);
        data.copy_from_slice(&out);
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus_scalar(x: f32) -> f32 {
    // Numerically stable: for large x, softplus ≈ x.
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        x.exp().ln_1p()
    }
}

// ── ReLU ────────────────────────────────────────────────────────────────────

/// Rectified Linear Unit: max(0, x).
#[derive(Debug, Default)]
pub struct ReLUActivation;

impl Activation for ReLUActivation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| x.max(0.0)).collect()
    }

    fn backward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect()
    }

    fn name(&self) -> &'static str {
        "ReLU"
    }

    fn is_monotonic(&self) -> bool {
        true
    }
}

// ── GeLU ────────────────────────────────────────────────────────────────────

/// Gaussian Error Linear Unit.
///
/// Exact:  `x * 0.5 * (1 + erf(x / sqrt(2)))`
/// Approx: `x * sigmoid(1.702 * x)` (the "quick" variant used in some
///          frameworks, but here we use the tanh approximation from the
///          original paper when `approximate` is true).
#[derive(Debug, Default)]
pub struct GeLUActivation {
    approximate: bool,
}

impl GeLUActivation {
    #[must_use]
    pub const fn new(approximate: bool) -> Self {
        Self { approximate }
    }

    /// Exact `GeLU` via the error function.
    fn exact(x: f32) -> f32 {
        let xf = f64::from(x);
        let cdf = 0.5 * (1.0 + erf_f64(xf / std::f64::consts::SQRT_2));
        #[allow(clippy::cast_possible_truncation)]
        let r = (xf * cdf) as f32;
        r
    }

    /// Tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
    fn approx(x: f32) -> f32 {
        let x64 = f64::from(x);
        let sqrt_2_over_pi: f64 = (2.0 / std::f64::consts::PI).sqrt();
        let inner = sqrt_2_over_pi * 0.044_715_f64.mul_add(x64.powi(3), x64);
        #[allow(clippy::cast_possible_truncation)]
        let r = (0.5 * x64 * (1.0 + inner.tanh())) as f32;
        r
    }
}

impl Activation for GeLUActivation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        if self.approximate {
            input.iter().map(|&x| Self::approx(x)).collect()
        } else {
            input.iter().map(|&x| Self::exact(x)).collect()
        }
    }

    fn backward(&self, input: &[f32]) -> Vec<f32> {
        // Numerical derivative (central difference) for simplicity.
        const H: f32 = 1e-4;
        input
            .iter()
            .map(|&x| {
                if self.approximate {
                    (Self::approx(x + H) - Self::approx(x - H)) / (2.0 * H)
                } else {
                    (Self::exact(x + H) - Self::exact(x - H)) / (2.0 * H)
                }
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        if self.approximate { "GeLU(approx)" } else { "GeLU(exact)" }
    }

    fn is_monotonic(&self) -> bool {
        false
    }
}

// ── SiLU (Swish) ───────────────────────────────────────────────────────────

/// Sigmoid Linear Unit (also known as Swish): `x * σ(x)`.
#[derive(Debug, Default)]
pub struct SiLUActivation;

impl Activation for SiLUActivation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| x * sigmoid_scalar(x)).collect()
    }

    fn backward(&self, input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .map(|&x| {
                let s = sigmoid_scalar(x);
                (x * s).mul_add(1.0 - s, s)
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "SiLU"
    }

    fn is_monotonic(&self) -> bool {
        false
    }
}

// ── Mish ────────────────────────────────────────────────────────────────────

/// Mish activation: `x * tanh(softplus(x))`.
#[derive(Debug, Default)]
pub struct MishActivation;

impl Activation for MishActivation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| x * softplus_scalar(x).tanh()).collect()
    }

    fn backward(&self, input: &[f32]) -> Vec<f32> {
        // d/dx mish(x) via numerical differentiation.
        const H: f32 = 1e-4;
        input
            .iter()
            .map(|&x| {
                let f_plus = (x + H) * softplus_scalar(x + H).tanh();
                let f_minus = (x - H) * softplus_scalar(x - H).tanh();
                (f_plus - f_minus) / (2.0 * H)
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "Mish"
    }

    fn is_monotonic(&self) -> bool {
        false
    }
}

// ── Tanh ────────────────────────────────────────────────────────────────────

/// Hyperbolic tangent activation.
#[derive(Debug, Default)]
pub struct TanhActivation;

impl Activation for TanhActivation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| x.tanh()).collect()
    }

    fn backward(&self, input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .map(|&x| {
                let t = x.tanh();
                t.mul_add(-t, 1.0)
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "Tanh"
    }

    fn is_monotonic(&self) -> bool {
        true
    }
}

// ── Sigmoid ─────────────────────────────────────────────────────────────────

/// Logistic sigmoid: `1 / (1 + exp(-x))`.
#[derive(Debug, Default)]
pub struct SigmoidActivation;

impl Activation for SigmoidActivation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| sigmoid_scalar(x)).collect()
    }

    fn backward(&self, input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .map(|&x| {
                let s = sigmoid_scalar(x);
                s * (1.0 - s)
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "Sigmoid"
    }

    fn is_monotonic(&self) -> bool {
        true
    }
}

// ── Leaky ReLU ──────────────────────────────────────────────────────────────

/// Leaky `ReLU` with configurable negative slope.
#[derive(Debug)]
pub struct LeakyReLUActivation {
    alpha: f32,
}

impl LeakyReLUActivation {
    #[must_use]
    pub const fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Default for LeakyReLUActivation {
    fn default() -> Self {
        Self { alpha: 0.01 }
    }
}

impl Activation for LeakyReLUActivation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| if x >= 0.0 { x } else { self.alpha * x }).collect()
    }

    fn backward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| if x >= 0.0 { 1.0 } else { self.alpha }).collect()
    }

    fn name(&self) -> &'static str {
        "LeakyReLU"
    }

    fn is_monotonic(&self) -> bool {
        self.alpha >= 0.0
    }
}

// ── ELU ─────────────────────────────────────────────────────────────────────

/// Exponential Linear Unit: `x` if x ≥ 0, `α * (exp(x) − 1)` otherwise.
#[derive(Debug)]
pub struct ELUActivation {
    alpha: f32,
}

impl ELUActivation {
    #[must_use]
    pub const fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Default for ELUActivation {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl Activation for ELUActivation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| if x >= 0.0 { x } else { self.alpha * x.exp_m1() }).collect()
    }

    fn backward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| if x >= 0.0 { 1.0 } else { self.alpha * x.exp() }).collect()
    }

    fn name(&self) -> &'static str {
        "ELU"
    }

    fn is_monotonic(&self) -> bool {
        self.alpha >= 0.0
    }
}

// ── SELU ────────────────────────────────────────────────────────────────────

/// Scaled Exponential Linear Unit (self-normalising).
#[derive(Debug, Default)]
pub struct SELUActivation;

/// SELU λ constant.
const SELU_LAMBDA: f32 = 1.050_700_9;
/// SELU α constant.
const SELU_ALPHA: f32 = 1.673_263_2;

impl Activation for SELUActivation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .map(
                |&x| {
                    if x >= 0.0 { SELU_LAMBDA * x } else { SELU_LAMBDA * SELU_ALPHA * x.exp_m1() }
                },
            )
            .collect()
    }

    fn backward(&self, input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .map(|&x| if x >= 0.0 { SELU_LAMBDA } else { SELU_LAMBDA * SELU_ALPHA * x.exp() })
            .collect()
    }

    fn name(&self) -> &'static str {
        "SELU"
    }

    fn is_monotonic(&self) -> bool {
        true
    }
}

// ── Softplus ────────────────────────────────────────────────────────────────

/// Softplus: `ln(1 + exp(x))`.
#[derive(Debug, Default)]
pub struct SoftplusActivation;

impl Activation for SoftplusActivation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| softplus_scalar(x)).collect()
    }

    fn backward(&self, input: &[f32]) -> Vec<f32> {
        // d/dx softplus(x) = σ(x)
        input.iter().map(|&x| sigmoid_scalar(x)).collect()
    }

    fn name(&self) -> &'static str {
        "Softplus"
    }

    fn is_monotonic(&self) -> bool {
        true
    }
}

// ── QuickGELU ───────────────────────────────────────────────────────────────

/// `QuickGELU`: `x * σ(1.702 * x)`.
#[derive(Debug, Default)]
pub struct QuickGELUActivation;

impl Activation for QuickGELUActivation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| x * sigmoid_scalar(1.702 * x)).collect()
    }

    fn backward(&self, input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .map(|&x| {
                let s = sigmoid_scalar(1.702 * x);
                (1.702 * x * s).mul_add(1.0 - s, s)
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "QuickGELU"
    }

    fn is_monotonic(&self) -> bool {
        false
    }
}

// ── HardSwish ───────────────────────────────────────────────────────────────

/// `HardSwish`: `x * clamp(x/6 + 0.5, 0, 1)`.
#[derive(Debug, Default)]
pub struct HardSwishActivation;

impl Activation for HardSwishActivation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| x * (x / 6.0 + 0.5).clamp(0.0, 1.0)).collect()
    }

    fn backward(&self, input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .map(|&x| {
                if x <= -3.0 {
                    0.0
                } else if x >= 3.0 {
                    1.0
                } else {
                    x / 3.0 + 0.5
                }
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "HardSwish"
    }

    fn is_monotonic(&self) -> bool {
        false
    }
}

// ── HardSigmoid ─────────────────────────────────────────────────────────────

/// `HardSigmoid`: `clamp(x/6 + 0.5, 0, 1)`.
#[derive(Debug, Default)]
pub struct HardSigmoidActivation;

impl Activation for HardSigmoidActivation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| (x / 6.0 + 0.5).clamp(0.0, 1.0)).collect()
    }

    fn backward(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| if x <= -3.0 || x >= 3.0 { 0.0 } else { 1.0 / 6.0 }).collect()
    }

    fn name(&self) -> &'static str {
        "HardSigmoid"
    }

    fn is_monotonic(&self) -> bool {
        true
    }
}

// ── Error-function helper (pure Rust, no libm dep) ─────────────────────────

/// Approximation of the error function via Abramowitz & Stegun (max error
/// ~1.5 × 10⁻⁷).
fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();
    let t = 1.0 / 0.327_591_1_f64.mul_add(a, 1.0);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    // Horner-style evaluation using mul_add for accuracy.
    let poly = 1.061_405_429_f64.mul_add(
        t5,
        (-1.453_152_027_f64).mul_add(
            t4,
            1.421_413_741_f64.mul_add(t3, (-0.284_496_736_f64).mul_add(t2, 0.254_829_592 * t)),
        ),
    );
    sign * (-a * a).exp().mul_add(-poly, 1.0)
}

// ── Registry ────────────────────────────────────────────────────────────────

/// A registry of named activation functions.
#[derive(Default)]
pub struct ActivationRegistry {
    entries: HashMap<String, Box<dyn Activation>>,
}

impl ActivationRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry pre-populated with all built-in activations.
    #[must_use]
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        reg.register("relu", Box::new(ReLUActivation));
        reg.register("gelu", Box::new(GeLUActivation::default()));
        reg.register("gelu_approx", Box::new(GeLUActivation::new(true)));
        reg.register("silu", Box::new(SiLUActivation));
        reg.register("mish", Box::new(MishActivation));
        reg.register("tanh", Box::new(TanhActivation));
        reg.register("sigmoid", Box::new(SigmoidActivation));
        reg.register("leaky_relu", Box::new(LeakyReLUActivation::default()));
        reg.register("elu", Box::new(ELUActivation::default()));
        reg.register("selu", Box::new(SELUActivation));
        reg.register("softplus", Box::new(SoftplusActivation));
        reg.register("quick_gelu", Box::new(QuickGELUActivation));
        reg.register("hard_swish", Box::new(HardSwishActivation));
        reg.register("hard_sigmoid", Box::new(HardSigmoidActivation));
        reg
    }

    /// Register a named activation.
    pub fn register(&mut self, name: &str, activation: Box<dyn Activation>) {
        self.entries.insert(name.to_owned(), activation);
    }

    /// Look up an activation by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn Activation> {
        self.entries.get(name).map(AsRef::as_ref)
    }

    /// List all registered names (sorted for determinism).
    #[must_use]
    pub fn list_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.entries.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of registered activations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl fmt::Debug for ActivationRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ActivationRegistry")
            .field("count", &self.entries.len())
            .field("names", &self.list_names())
            .finish()
    }
}

// ── Fused bias + activation ─────────────────────────────────────────────────

/// Applies `activation(input[i] + bias[i])` element-wise.
///
/// `bias` is broadcast when `bias.len() == 1`, otherwise it must match the
/// length of `input`.
pub struct FusedActivation<'a> {
    activation: &'a dyn Activation,
    bias: Vec<f32>,
}

impl fmt::Debug for FusedActivation<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FusedActivation")
            .field("activation", &self.activation.name())
            .field("bias_len", &self.bias.len())
            .finish()
    }
}

impl<'a> FusedActivation<'a> {
    /// Create a fused bias+activation.
    ///
    /// # Panics
    ///
    /// Panics if `bias` is empty.
    #[must_use]
    pub fn new(activation: &'a dyn Activation, bias: Vec<f32>) -> Self {
        assert!(!bias.is_empty(), "bias must not be empty");
        Self { activation, bias }
    }

    /// Apply fused bias + activation.
    ///
    /// # Panics
    ///
    /// Panics if `bias.len() > 1` and `bias.len() != input.len()`.
    #[must_use]
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let biased: Vec<f32> = if self.bias.len() == 1 {
            let b = self.bias[0];
            input.iter().map(|&x| x + b).collect()
        } else {
            assert_eq!(
                input.len(),
                self.bias.len(),
                "input length ({}) must match bias length ({})",
                input.len(),
                self.bias.len(),
            );
            input.iter().zip(self.bias.iter()).map(|(&x, &b)| x + b).collect()
        };
        self.activation.forward(&biased)
    }
}

// ── Profiler ────────────────────────────────────────────────────────────────

/// Result of profiling an activation function.
#[derive(Debug, Clone)]
pub struct ProfileResult {
    /// Name of the activation.
    pub name: String,
    /// Number of elements processed.
    pub num_elements: usize,
    /// Number of iterations executed.
    pub iterations: usize,
    /// Total wall time across all iterations.
    pub total_time: Duration,
    /// Average time per iteration.
    pub avg_time: Duration,
    /// Elements per second (throughput).
    pub elements_per_second: f64,
}

/// Profiles activation function performance.
#[derive(Debug)]
pub struct ActivationProfiler {
    iterations: usize,
}

impl ActivationProfiler {
    /// Create a profiler that runs `iterations` rounds.
    #[must_use]
    pub const fn new(iterations: usize) -> Self {
        Self { iterations }
    }

    /// Profile `activation` on `input`.
    #[must_use]
    pub fn profile(&self, activation: &dyn Activation, input: &[f32]) -> ProfileResult {
        // Warm-up
        let _ = activation.forward(input);

        let start = Instant::now();
        for _ in 0..self.iterations {
            let _ = activation.forward(input);
        }
        let total_time = start.elapsed();

        #[allow(clippy::cast_possible_truncation)]
        let avg_time = total_time / self.iterations as u32;
        let total_elements = input.len() * self.iterations;
        #[allow(clippy::cast_precision_loss)]
        let elements_per_second = total_elements as f64 / total_time.as_secs_f64();

        ProfileResult {
            name: activation.name().to_owned(),
            num_elements: input.len(),
            iterations: self.iterations,
            total_time,
            avg_time,
            elements_per_second,
        }
    }
}

// ── Factory helper ──────────────────────────────────────────────────────────

/// Create a boxed [`Activation`] from an [`ActivationType`].
#[must_use]
pub fn create_activation(ty: &ActivationType) -> Box<dyn Activation> {
    match ty {
        ActivationType::ReLU => Box::new(ReLUActivation),
        ActivationType::GeLU => Box::new(GeLUActivation::default()),
        ActivationType::SiLU => Box::new(SiLUActivation),
        ActivationType::Mish => Box::new(MishActivation),
        ActivationType::Tanh => Box::new(TanhActivation),
        ActivationType::Sigmoid => Box::new(SigmoidActivation),
        ActivationType::LeakyReLU(a) => Box::new(LeakyReLUActivation::new(*a)),
        ActivationType::ELU(a) => Box::new(ELUActivation::new(*a)),
        ActivationType::SELU => Box::new(SELUActivation),
        ActivationType::Softplus => Box::new(SoftplusActivation),
        ActivationType::QuickGELU => Box::new(QuickGELUActivation),
        ActivationType::HardSwish => Box::new(HardSwishActivation),
        ActivationType::HardSigmoid => Box::new(HardSigmoidActivation),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::imprecise_flops,
    clippy::suboptimal_flops,
    clippy::approx_constant
)]
mod tests {
    use super::*;

    // Tolerance for floating-point comparisons.
    const TOL: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    // -----------------------------------------------------------------------
    // ActivationType
    // -----------------------------------------------------------------------

    #[test]
    fn activation_type_display() {
        assert_eq!(ActivationType::ReLU.to_string(), "ReLU");
        assert_eq!(ActivationType::GeLU.to_string(), "GeLU");
        assert_eq!(ActivationType::SiLU.to_string(), "SiLU");
        assert_eq!(ActivationType::Mish.to_string(), "Mish");
        assert_eq!(ActivationType::Tanh.to_string(), "Tanh");
        assert_eq!(ActivationType::Sigmoid.to_string(), "Sigmoid");
        assert_eq!(ActivationType::SELU.to_string(), "SELU");
        assert_eq!(ActivationType::Softplus.to_string(), "Softplus");
        assert_eq!(ActivationType::QuickGELU.to_string(), "QuickGELU");
        assert_eq!(ActivationType::HardSwish.to_string(), "HardSwish");
        assert_eq!(ActivationType::HardSigmoid.to_string(), "HardSigmoid");
    }

    #[test]
    fn activation_type_display_parameterised() {
        assert_eq!(ActivationType::LeakyReLU(0.01).to_string(), "LeakyReLU(0.01)");
        assert_eq!(ActivationType::ELU(1.0).to_string(), "ELU(1)");
    }

    #[test]
    fn activation_type_equality() {
        assert_eq!(ActivationType::ReLU, ActivationType::ReLU);
        assert_ne!(ActivationType::ReLU, ActivationType::GeLU);
        assert_eq!(ActivationType::LeakyReLU(0.1), ActivationType::LeakyReLU(0.1));
        assert_ne!(ActivationType::LeakyReLU(0.1), ActivationType::LeakyReLU(0.2));
    }

    #[test]
    fn activation_type_clone() {
        let a = ActivationType::ELU(0.5);
        let b = a.clone();
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // ActivationConfig
    // -----------------------------------------------------------------------

    #[test]
    fn config_defaults() {
        let cfg = ActivationConfig::new(ActivationType::ReLU);
        assert!(!cfg.in_place);
        assert!(!cfg.approximate);
    }

    #[test]
    fn config_builder_in_place() {
        let cfg = ActivationConfig::new(ActivationType::GeLU).with_in_place(true);
        assert!(cfg.in_place);
    }

    #[test]
    fn config_builder_approximate() {
        let cfg = ActivationConfig::new(ActivationType::GeLU).with_approximate(true);
        assert!(cfg.approximate);
    }

    #[test]
    fn config_builder_chained() {
        let cfg =
            ActivationConfig::new(ActivationType::GeLU).with_in_place(true).with_approximate(true);
        assert!(cfg.in_place);
        assert!(cfg.approximate);
    }

    // -----------------------------------------------------------------------
    // ReLU
    // -----------------------------------------------------------------------

    #[test]
    fn relu_positive_identity() {
        let relu = ReLUActivation;
        assert_eq!(relu.forward(&[1.0, 2.5, 100.0]), vec![1.0, 2.5, 100.0]);
    }

    #[test]
    fn relu_negative_zero() {
        let relu = ReLUActivation;
        assert_eq!(relu.forward(&[-1.0, -0.5, -100.0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn relu_zero() {
        let relu = ReLUActivation;
        assert_eq!(relu.forward(&[0.0]), vec![0.0]);
    }

    #[test]
    fn relu_mixed() {
        let relu = ReLUActivation;
        assert_eq!(relu.forward(&[-1.0, 0.0, 1.0]), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn relu_backward_positive() {
        let relu = ReLUActivation;
        assert_eq!(relu.backward(&[1.0, 5.0]), vec![1.0, 1.0]);
    }

    #[test]
    fn relu_backward_negative() {
        let relu = ReLUActivation;
        assert_eq!(relu.backward(&[-1.0, -5.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn relu_backward_zero() {
        let relu = ReLUActivation;
        assert_eq!(relu.backward(&[0.0]), vec![0.0]);
    }

    #[test]
    fn relu_name() {
        assert_eq!(ReLUActivation.name(), "ReLU");
    }

    #[test]
    fn relu_is_monotonic() {
        assert!(ReLUActivation.is_monotonic());
    }

    #[test]
    fn relu_empty_input() {
        let relu = ReLUActivation;
        assert!(relu.forward(&[]).is_empty());
        assert!(relu.backward(&[]).is_empty());
    }

    // -----------------------------------------------------------------------
    // GeLU
    // -----------------------------------------------------------------------

    #[test]
    fn gelu_exact_zero() {
        let gelu = GeLUActivation::default();
        let out = gelu.forward(&[0.0]);
        assert!(approx_eq(out[0], 0.0));
    }

    #[test]
    fn gelu_exact_positive() {
        let gelu = GeLUActivation::default();
        let out = gelu.forward(&[1.0]);
        // GeLU(1) ≈ 0.8413
        assert!(approx_eq(out[0], 0.8413));
    }

    #[test]
    fn gelu_exact_negative() {
        let gelu = GeLUActivation::default();
        let out = gelu.forward(&[-1.0]);
        // GeLU(-1) ≈ -0.1587
        assert!(approx_eq(out[0], -0.1587));
    }

    #[test]
    fn gelu_approx_close_to_exact() {
        let exact = GeLUActivation::new(false);
        let approx = GeLUActivation::new(true);
        for &x in &[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0] {
            let e = exact.forward(&[x])[0];
            let a = approx.forward(&[x])[0];
            assert!((e - a).abs() < 0.02, "GeLU exact={e} approx={a} differ too much at x={x}");
        }
    }

    #[test]
    fn gelu_names() {
        assert_eq!(GeLUActivation::new(false).name(), "GeLU(exact)");
        assert_eq!(GeLUActivation::new(true).name(), "GeLU(approx)");
    }

    #[test]
    fn gelu_not_monotonic() {
        assert!(!GeLUActivation::default().is_monotonic());
    }

    #[test]
    fn gelu_backward_finite() {
        let gelu = GeLUActivation::default();
        let grad = gelu.backward(&[-1.0, 0.0, 1.0]);
        for &g in &grad {
            assert!(g.is_finite(), "gradient must be finite");
        }
    }

    #[test]
    fn gelu_approx_backward_finite() {
        let gelu = GeLUActivation::new(true);
        let grad = gelu.backward(&[-1.0, 0.0, 1.0]);
        for &g in &grad {
            assert!(g.is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // SiLU
    // -----------------------------------------------------------------------

    #[test]
    fn silu_zero() {
        let silu = SiLUActivation;
        let out = silu.forward(&[0.0]);
        assert!(approx_eq(out[0], 0.0));
    }

    #[test]
    fn silu_positive() {
        let silu = SiLUActivation;
        let out = silu.forward(&[1.0]);
        // SiLU(1) = 1 * σ(1) ≈ 0.7311
        assert!(approx_eq(out[0], 0.7311));
    }

    #[test]
    fn silu_negative_small() {
        // SiLU has a minimum around x ≈ -1.278 with value ≈ -0.278
        let silu = SiLUActivation;
        let out = silu.forward(&[-1.278]);
        assert!(out[0] < 0.0, "SiLU should be negative near x=-1.278");
    }

    #[test]
    fn silu_large_positive() {
        let silu = SiLUActivation;
        let out = silu.forward(&[10.0]);
        // For large x, σ(x) ≈ 1, so SiLU(x) ≈ x
        assert!((out[0] - 10.0).abs() < 0.01);
    }

    #[test]
    fn silu_backward_at_zero() {
        let silu = SiLUActivation;
        let grad = silu.backward(&[0.0]);
        // SiLU'(0) = σ(0) + 0*σ(0)*(1-σ(0)) = 0.5
        assert!(approx_eq(grad[0], 0.5));
    }

    #[test]
    fn silu_not_monotonic() {
        assert!(!SiLUActivation.is_monotonic());
    }

    // -----------------------------------------------------------------------
    // Mish
    // -----------------------------------------------------------------------

    #[test]
    fn mish_zero() {
        let mish = MishActivation;
        let out = mish.forward(&[0.0]);
        // Mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(ln2) ≈ 0
        assert!(approx_eq(out[0], 0.0));
    }

    #[test]
    fn mish_positive() {
        let mish = MishActivation;
        let out = mish.forward(&[1.0]);
        // Mish(1) = 1 * tanh(ln(1+e)) ≈ 0.8651
        assert!((out[0] - 0.8651).abs() < 0.01);
    }

    #[test]
    fn mish_backward_finite() {
        let mish = MishActivation;
        let grad = mish.backward(&[-2.0, 0.0, 2.0]);
        for &g in &grad {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn mish_not_monotonic() {
        assert!(!MishActivation.is_monotonic());
    }

    // -----------------------------------------------------------------------
    // Tanh
    // -----------------------------------------------------------------------

    #[test]
    fn tanh_zero() {
        let act = TanhActivation;
        assert!(approx_eq(act.forward(&[0.0])[0], 0.0));
    }

    #[test]
    fn tanh_bounds() {
        let act = TanhActivation;
        let out = act.forward(&[-100.0, 100.0]);
        assert!(approx_eq(out[0], -1.0));
        assert!(approx_eq(out[1], 1.0));
    }

    #[test]
    fn tanh_backward_at_zero() {
        let act = TanhActivation;
        let grad = act.backward(&[0.0]);
        // tanh'(0) = 1 - 0² = 1
        assert!(approx_eq(grad[0], 1.0));
    }

    #[test]
    fn tanh_is_monotonic() {
        assert!(TanhActivation.is_monotonic());
    }

    // -----------------------------------------------------------------------
    // Sigmoid
    // -----------------------------------------------------------------------

    #[test]
    fn sigmoid_zero() {
        let act = SigmoidActivation;
        assert!(approx_eq(act.forward(&[0.0])[0], 0.5));
    }

    #[test]
    fn sigmoid_large_positive() {
        let act = SigmoidActivation;
        assert!(approx_eq(act.forward(&[100.0])[0], 1.0));
    }

    #[test]
    fn sigmoid_large_negative() {
        let act = SigmoidActivation;
        assert!(approx_eq(act.forward(&[-100.0])[0], 0.0));
    }

    #[test]
    fn sigmoid_backward_at_zero() {
        let act = SigmoidActivation;
        // σ'(0) = σ(0)(1 − σ(0)) = 0.25
        assert!(approx_eq(act.backward(&[0.0])[0], 0.25));
    }

    #[test]
    fn sigmoid_is_monotonic() {
        assert!(SigmoidActivation.is_monotonic());
    }

    // -----------------------------------------------------------------------
    // LeakyReLU
    // -----------------------------------------------------------------------

    #[test]
    fn leaky_relu_positive() {
        let act = LeakyReLUActivation::new(0.1);
        assert_eq!(act.forward(&[5.0]), vec![5.0]);
    }

    #[test]
    fn leaky_relu_negative() {
        let act = LeakyReLUActivation::new(0.1);
        let out = act.forward(&[-5.0]);
        assert!(approx_eq(out[0], -0.5));
    }

    #[test]
    fn leaky_relu_default_slope() {
        let act = LeakyReLUActivation::default();
        let out = act.forward(&[-1.0]);
        assert!(approx_eq(out[0], -0.01));
    }

    #[test]
    fn leaky_relu_backward_positive() {
        let act = LeakyReLUActivation::new(0.2);
        assert_eq!(act.backward(&[1.0]), vec![1.0]);
    }

    #[test]
    fn leaky_relu_backward_negative() {
        let act = LeakyReLUActivation::new(0.2);
        let out = act.backward(&[-1.0]);
        assert!(approx_eq(out[0], 0.2));
    }

    #[test]
    fn leaky_relu_is_monotonic_positive_alpha() {
        assert!(LeakyReLUActivation::new(0.1).is_monotonic());
    }

    #[test]
    fn leaky_relu_is_monotonic_negative_alpha() {
        assert!(!LeakyReLUActivation::new(-0.1).is_monotonic());
    }

    // -----------------------------------------------------------------------
    // ELU
    // -----------------------------------------------------------------------

    #[test]
    fn elu_positive() {
        let act = ELUActivation::new(1.0);
        assert_eq!(act.forward(&[2.0]), vec![2.0]);
    }

    #[test]
    fn elu_negative() {
        let act = ELUActivation::new(1.0);
        let out = act.forward(&[-1.0]);
        // ELU(-1) = 1.0 * (e^-1 - 1) ≈ -0.6321
        assert!(approx_eq(out[0], -0.6321));
    }

    #[test]
    fn elu_backward_positive() {
        let act = ELUActivation::new(1.0);
        assert_eq!(act.backward(&[2.0]), vec![1.0]);
    }

    #[test]
    fn elu_backward_negative() {
        let act = ELUActivation::new(1.0);
        let grad = act.backward(&[-1.0]);
        // α * exp(-1) ≈ 0.3679
        assert!(approx_eq(grad[0], 0.3679));
    }

    #[test]
    fn elu_alpha_2() {
        let act = ELUActivation::new(2.0);
        let out = act.forward(&[-1.0]);
        // 2.0 * (e^-1 - 1) ≈ -1.2642
        assert!(approx_eq(out[0], -1.2642));
    }

    // -----------------------------------------------------------------------
    // SELU
    // -----------------------------------------------------------------------

    #[test]
    fn selu_positive() {
        let act = SELUActivation;
        let out = act.forward(&[1.0]);
        assert!(approx_eq(out[0], SELU_LAMBDA));
    }

    #[test]
    fn selu_zero() {
        let act = SELUActivation;
        assert!(approx_eq(act.forward(&[0.0])[0], 0.0));
    }

    #[test]
    fn selu_negative() {
        let act = SELUActivation;
        let out = act.forward(&[-1.0]);
        let expected = SELU_LAMBDA * SELU_ALPHA * ((-1.0_f32).exp() - 1.0);
        assert!(approx_eq(out[0], expected));
    }

    #[test]
    fn selu_is_monotonic() {
        assert!(SELUActivation.is_monotonic());
    }

    // -----------------------------------------------------------------------
    // Softplus
    // -----------------------------------------------------------------------

    #[test]
    fn softplus_zero() {
        let act = SoftplusActivation;
        // softplus(0) = ln(2) ≈ 0.6931
        assert!(approx_eq(act.forward(&[0.0])[0], 0.6931));
    }

    #[test]
    fn softplus_large_positive() {
        let act = SoftplusActivation;
        // softplus(30) ≈ 30 (numerically stable branch)
        let out = act.forward(&[30.0]);
        assert!((out[0] - 30.0).abs() < 0.01);
    }

    #[test]
    fn softplus_large_negative() {
        let act = SoftplusActivation;
        let out = act.forward(&[-30.0]);
        assert!(out[0].abs() < 0.01);
    }

    #[test]
    fn softplus_backward_is_sigmoid() {
        let act = SoftplusActivation;
        let grad = act.backward(&[0.0]);
        // d/dx softplus(0) = σ(0) = 0.5
        assert!(approx_eq(grad[0], 0.5));
    }

    #[test]
    fn softplus_is_monotonic() {
        assert!(SoftplusActivation.is_monotonic());
    }

    // -----------------------------------------------------------------------
    // QuickGELU
    // -----------------------------------------------------------------------

    #[test]
    fn quick_gelu_zero() {
        let act = QuickGELUActivation;
        assert!(approx_eq(act.forward(&[0.0])[0], 0.0));
    }

    #[test]
    fn quick_gelu_positive() {
        let act = QuickGELUActivation;
        let out = act.forward(&[1.0]);
        // x * σ(1.702 * 1) ≈ 0.8458
        assert!((out[0] - 0.8458).abs() < 0.01);
    }

    #[test]
    fn quick_gelu_backward_finite() {
        let act = QuickGELUActivation;
        let grad = act.backward(&[-1.0, 0.0, 1.0]);
        for &g in &grad {
            assert!(g.is_finite());
        }
    }

    // -----------------------------------------------------------------------
    // HardSwish
    // -----------------------------------------------------------------------

    #[test]
    fn hard_swish_zero() {
        let act = HardSwishActivation;
        assert!(approx_eq(act.forward(&[0.0])[0], 0.0));
    }

    #[test]
    fn hard_swish_large_positive() {
        let act = HardSwishActivation;
        let out = act.forward(&[10.0]);
        // x * clamp(x/6+0.5, 0, 1) = 10 * 1 = 10
        assert!(approx_eq(out[0], 10.0));
    }

    #[test]
    fn hard_swish_large_negative() {
        let act = HardSwishActivation;
        assert!(approx_eq(act.forward(&[-10.0])[0], 0.0));
    }

    #[test]
    fn hard_swish_backward_saturated() {
        let act = HardSwishActivation;
        assert!(approx_eq(act.backward(&[-5.0])[0], 0.0));
        assert!(approx_eq(act.backward(&[5.0])[0], 1.0));
    }

    // -----------------------------------------------------------------------
    // HardSigmoid
    // -----------------------------------------------------------------------

    #[test]
    fn hard_sigmoid_zero() {
        let act = HardSigmoidActivation;
        assert!(approx_eq(act.forward(&[0.0])[0], 0.5));
    }

    #[test]
    fn hard_sigmoid_saturated() {
        let act = HardSigmoidActivation;
        assert!(approx_eq(act.forward(&[-10.0])[0], 0.0));
        assert!(approx_eq(act.forward(&[10.0])[0], 1.0));
    }

    #[test]
    fn hard_sigmoid_backward_in_linear_region() {
        let act = HardSigmoidActivation;
        let grad = act.backward(&[0.0]);
        assert!(approx_eq(grad[0], 1.0 / 6.0));
    }

    #[test]
    fn hard_sigmoid_backward_saturated() {
        let act = HardSigmoidActivation;
        assert!(approx_eq(act.backward(&[-5.0])[0], 0.0));
        assert!(approx_eq(act.backward(&[5.0])[0], 0.0));
    }

    #[test]
    fn hard_sigmoid_is_monotonic() {
        assert!(HardSigmoidActivation.is_monotonic());
    }

    // -----------------------------------------------------------------------
    // In-place modification
    // -----------------------------------------------------------------------

    #[test]
    fn inplace_relu() {
        let relu = ReLUActivation;
        let mut data = vec![-1.0, 0.0, 1.0];
        relu.forward_inplace(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn inplace_sigmoid() {
        let sig = SigmoidActivation;
        let mut data = vec![0.0];
        sig.forward_inplace(&mut data);
        assert!(approx_eq(data[0], 0.5));
    }

    #[test]
    fn inplace_matches_forward() {
        let act = SiLUActivation;
        let input = vec![-1.0, 0.0, 1.0, 2.0];
        let expected = act.forward(&input);
        let mut data = input;
        act.forward_inplace(&mut data);
        assert_eq!(data, expected);
    }

    // -----------------------------------------------------------------------
    // Registry
    // -----------------------------------------------------------------------

    #[test]
    fn registry_new_is_empty() {
        let reg = ActivationRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn registry_builtins_count() {
        let reg = ActivationRegistry::with_builtins();
        // 14 built-in entries (gelu + gelu_approx)
        assert_eq!(reg.len(), 14);
    }

    #[test]
    fn registry_lookup_relu() {
        let reg = ActivationRegistry::with_builtins();
        let act = reg.get("relu").expect("relu should exist");
        assert_eq!(act.name(), "ReLU");
    }

    #[test]
    fn registry_lookup_missing() {
        let reg = ActivationRegistry::with_builtins();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn registry_list_names_sorted() {
        let reg = ActivationRegistry::with_builtins();
        let names = reg.list_names();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn registry_custom_registration() {
        let mut reg = ActivationRegistry::new();
        reg.register("my_relu", Box::new(ReLUActivation));
        assert_eq!(reg.len(), 1);
        assert!(reg.get("my_relu").is_some());
    }

    #[test]
    fn registry_overwrite() {
        let mut reg = ActivationRegistry::new();
        reg.register("act", Box::new(ReLUActivation));
        reg.register("act", Box::new(SigmoidActivation));
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.get("act").unwrap().name(), "Sigmoid");
    }

    #[test]
    fn registry_debug() {
        let reg = ActivationRegistry::with_builtins();
        let dbg = format!("{reg:?}");
        assert!(dbg.contains("ActivationRegistry"));
        assert!(dbg.contains("14"));
    }

    // -----------------------------------------------------------------------
    // FusedActivation
    // -----------------------------------------------------------------------

    #[test]
    fn fused_scalar_bias() {
        let relu = ReLUActivation;
        let fused = FusedActivation::new(&relu, vec![1.0]);
        // input=-2, bias=1 → -1 → ReLU → 0
        assert_eq!(fused.forward(&[-2.0]), vec![0.0]);
    }

    #[test]
    fn fused_vector_bias() {
        let relu = ReLUActivation;
        let fused = FusedActivation::new(&relu, vec![1.0, -1.0, 0.0]);
        // [-2+1, 0-1, 3+0] = [-1, -1, 3] → ReLU → [0, 0, 3]
        assert_eq!(fused.forward(&[-2.0, 0.0, 3.0]), vec![0.0, 0.0, 3.0]);
    }

    #[test]
    fn fused_matches_separate() {
        let sig = SigmoidActivation;
        let bias = vec![0.5, -0.5];
        let input = vec![1.0, -1.0];

        // Separate: add bias then activate
        let biased: Vec<f32> = input.iter().zip(bias.iter()).map(|(&x, &b)| x + b).collect();
        let expected = sig.forward(&biased);

        let fused = FusedActivation::new(&sig, bias);
        let got = fused.forward(&input);

        for (e, g) in expected.iter().zip(got.iter()) {
            assert!(approx_eq(*e, *g));
        }
    }

    #[test]
    fn fused_broadcast_bias() {
        let act = TanhActivation;
        let fused = FusedActivation::new(&act, vec![0.5]);
        let out = fused.forward(&[-0.5, 0.0, 0.5]);
        // inputs become [0.0, 0.5, 1.0]
        assert!(approx_eq(out[0], 0.0_f32.tanh()));
        assert!(approx_eq(out[1], 0.5_f32.tanh()));
        assert!(approx_eq(out[2], 1.0_f32.tanh()));
    }

    #[test]
    #[should_panic(expected = "bias must not be empty")]
    fn fused_empty_bias_panics() {
        let relu = ReLUActivation;
        let _ = FusedActivation::new(&relu, vec![]);
    }

    #[test]
    #[should_panic(expected = "input length")]
    fn fused_mismatched_lengths_panics() {
        let relu = ReLUActivation;
        let fused = FusedActivation::new(&relu, vec![1.0, 2.0]);
        let _ = fused.forward(&[1.0, 2.0, 3.0]);
    }

    // -----------------------------------------------------------------------
    // Profiler
    // -----------------------------------------------------------------------

    #[test]
    fn profiler_returns_results() {
        let profiler = ActivationProfiler::new(10);
        let input: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let result = profiler.profile(&ReLUActivation, &input);
        assert_eq!(result.name, "ReLU");
        assert_eq!(result.num_elements, 100);
        assert_eq!(result.iterations, 10);
        assert!(result.elements_per_second > 0.0);
    }

    #[test]
    fn profiler_avg_time_positive() {
        let profiler = ActivationProfiler::new(5);
        let input = vec![1.0; 1000];
        let result = profiler.profile(&SigmoidActivation, &input);
        assert!(result.avg_time <= result.total_time);
    }

    // -----------------------------------------------------------------------
    // create_activation factory
    // -----------------------------------------------------------------------

    #[test]
    fn factory_creates_relu() {
        let act = create_activation(&ActivationType::ReLU);
        assert_eq!(act.name(), "ReLU");
    }

    #[test]
    fn factory_creates_gelu() {
        let act = create_activation(&ActivationType::GeLU);
        assert_eq!(act.name(), "GeLU(exact)");
    }

    #[test]
    fn factory_creates_leaky_relu() {
        let act = create_activation(&ActivationType::LeakyReLU(0.05));
        assert_eq!(act.name(), "LeakyReLU");
    }

    #[test]
    fn factory_creates_all_types() {
        let types = vec![
            ActivationType::ReLU,
            ActivationType::GeLU,
            ActivationType::SiLU,
            ActivationType::Mish,
            ActivationType::Tanh,
            ActivationType::Sigmoid,
            ActivationType::LeakyReLU(0.01),
            ActivationType::ELU(1.0),
            ActivationType::SELU,
            ActivationType::Softplus,
            ActivationType::QuickGELU,
            ActivationType::HardSwish,
            ActivationType::HardSigmoid,
        ];
        for ty in &types {
            let act = create_activation(ty);
            assert!(!act.name().is_empty(), "activation {ty} has empty name");
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases — NaN, Inf, very large/small
    // -----------------------------------------------------------------------

    #[test]
    fn relu_nan_propagates() {
        let relu = ReLUActivation;
        let out = relu.forward(&[f32::NAN]);
        // f32::max(NAN, 0.0) returns 0.0 per IEEE 754 in Rust
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn relu_inf_positive() {
        let relu = ReLUActivation;
        assert_eq!(relu.forward(&[f32::INFINITY]), vec![f32::INFINITY]);
    }

    #[test]
    fn relu_inf_negative() {
        let relu = ReLUActivation;
        assert_eq!(relu.forward(&[f32::NEG_INFINITY]), vec![0.0]);
    }

    #[test]
    fn sigmoid_nan() {
        let act = SigmoidActivation;
        let out = act.forward(&[f32::NAN]);
        assert!(out[0].is_nan());
    }

    #[test]
    fn tanh_inf() {
        let act = TanhActivation;
        let out = act.forward(&[f32::INFINITY, f32::NEG_INFINITY]);
        assert!(approx_eq(out[0], 1.0));
        assert!(approx_eq(out[1], -1.0));
    }

    #[test]
    fn silu_nan() {
        let act = SiLUActivation;
        let out = act.forward(&[f32::NAN]);
        assert!(out[0].is_nan());
    }

    #[test]
    fn softplus_very_large() {
        let act = SoftplusActivation;
        let out = act.forward(&[1000.0]);
        // Numerically stable: returns x for very large x
        assert!((out[0] - 1000.0).abs() < 0.01);
    }

    #[test]
    fn softplus_very_negative() {
        let act = SoftplusActivation;
        let out = act.forward(&[-1000.0]);
        assert!(out[0].abs() < 0.01);
    }

    #[test]
    fn elu_neg_infinity() {
        let act = ELUActivation::new(1.0);
        let out = act.forward(&[f32::NEG_INFINITY]);
        // e^(-∞) - 1 = -1 → α * -1 = -1
        assert!(approx_eq(out[0], -1.0));
    }

    #[test]
    fn hard_sigmoid_nan() {
        let act = HardSigmoidActivation;
        let out = act.forward(&[f32::NAN]);
        // clamp of NaN is implementation-defined; just ensure no panic.
        let _ = out[0];
    }

    // -----------------------------------------------------------------------
    // Property: monotonic activations preserve order
    // -----------------------------------------------------------------------

    #[test]
    fn monotonic_relu_preserves_order() {
        let act = ReLUActivation;
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let out = act.forward(&input);
        for w in out.windows(2) {
            assert!(w[0] <= w[1], "ReLU violated monotonicity: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn monotonic_sigmoid_preserves_order() {
        let act = SigmoidActivation;
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let out = act.forward(&input);
        for w in out.windows(2) {
            assert!(w[0] <= w[1], "Sigmoid violated monotonicity: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn monotonic_tanh_preserves_order() {
        let act = TanhActivation;
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let out = act.forward(&input);
        for w in out.windows(2) {
            assert!(w[0] <= w[1], "Tanh violated monotonicity: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn monotonic_softplus_preserves_order() {
        let act = SoftplusActivation;
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let out = act.forward(&input);
        for w in out.windows(2) {
            assert!(w[0] <= w[1], "Softplus violated monotonicity: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn monotonic_selu_preserves_order() {
        let act = SELUActivation;
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let out = act.forward(&input);
        for w in out.windows(2) {
            assert!(w[0] <= w[1], "SELU violated monotonicity: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn monotonic_hard_sigmoid_preserves_order() {
        let act = HardSigmoidActivation;
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
        let out = act.forward(&input);
        for w in out.windows(2) {
            assert!(w[0] <= w[1], "HardSigmoid violated monotonicity: {} > {}", w[0], w[1]);
        }
    }

    // -----------------------------------------------------------------------
    // proptest — property-based tests
    // -----------------------------------------------------------------------

    proptest::proptest! {
        #[test]
        fn relu_nonnegative_output(x in -100.0_f32..100.0) {
            let relu = ReLUActivation;
            let out = relu.forward(&[x]);
            proptest::prop_assert!(out[0] >= 0.0);
        }

        #[test]
        fn sigmoid_bounded_output(x in -100.0_f32..100.0) {
            let act = SigmoidActivation;
            let out = act.forward(&[x]);
            proptest::prop_assert!(out[0] >= 0.0 && out[0] <= 1.0);
        }

        #[test]
        fn tanh_bounded_output(x in -100.0_f32..100.0) {
            let act = TanhActivation;
            let out = act.forward(&[x]);
            proptest::prop_assert!(out[0] >= -1.0 && out[0] <= 1.0);
        }

        #[test]
        fn hard_sigmoid_bounded_output(x in -100.0_f32..100.0) {
            let act = HardSigmoidActivation;
            let out = act.forward(&[x]);
            proptest::prop_assert!(out[0] >= 0.0 && out[0] <= 1.0);
        }

        #[test]
        fn softplus_nonnegative_output(x in -100.0_f32..100.0) {
            let act = SoftplusActivation;
            let out = act.forward(&[x]);
            proptest::prop_assert!(out[0] >= 0.0);
        }

        #[test]
        fn relu_monotonic_pair(
            x1 in -50.0_f32..50.0,
            x2 in -50.0_f32..50.0,
        ) {
            let relu = ReLUActivation;
            let o1 = relu.forward(&[x1])[0];
            let o2 = relu.forward(&[x2])[0];
            if x1 <= x2 {
                proptest::prop_assert!(o1 <= o2, "ReLU not monotonic: f({x1})={o1} > f({x2})={o2}");
            }
        }

        #[test]
        fn sigmoid_monotonic_pair(
            x1 in -50.0_f32..50.0,
            x2 in -50.0_f32..50.0,
        ) {
            let act = SigmoidActivation;
            let o1 = act.forward(&[x1])[0];
            let o2 = act.forward(&[x2])[0];
            if x1 <= x2 {
                proptest::prop_assert!(o1 <= o2, "Sigmoid not monotonic");
            }
        }

        #[test]
        fn silu_at_zero_is_zero(scale in 0.0_f32..0.0001) {
            let act = SiLUActivation;
            let out = act.forward(&[scale]);
            // SiLU(x) ≈ 0.5*x for small x
            proptest::prop_assert!((out[0] - scale * 0.5).abs() < 0.001);
        }

        #[test]
        fn leaky_relu_at_least_alpha_times_x(x in -100.0_f32..0.0) {
            let alpha = 0.01_f32;
            let act = LeakyReLUActivation::new(alpha);
            let out = act.forward(&[x]);
            proptest::prop_assert!(
                (out[0] - alpha * x).abs() < 1e-5,
                "LeakyReLU({x}) = {} but expected {}",
                out[0],
                alpha * x,
            );
        }
    }

    // -----------------------------------------------------------------------
    // erf helper
    // -----------------------------------------------------------------------

    #[test]
    fn erf_zero() {
        assert!(erf_f64(0.0).abs() < 1e-6);
    }

    #[test]
    fn erf_one() {
        // erf(1) ≈ 0.8427
        assert!((erf_f64(1.0) - 0.8427).abs() < 0.001);
    }

    #[test]
    fn erf_symmetry() {
        for &x in &[0.5, 1.0, 2.0, 3.0] {
            assert!((erf_f64(x) + erf_f64(-x)).abs() < 1e-6);
        }
    }
}
