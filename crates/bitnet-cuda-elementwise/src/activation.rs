//! Element-wise activation functions.
//!
//! CPU reference implementations are always available.  When compiled with
//! the `gpu` or `cuda` feature, CUDA kernel stubs are included for future
//! device dispatch.

use std::f32::consts::PI;

// ── Activation enum ────────────────────────────────────────────────

/// Supported activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x).
    ReLU,
    /// Gaussian Error Linear Unit (tanh approximation).
    GELU,
    /// Sigmoid Linear Unit / `SwiGLU`: x · σ(x).
    SiLU,
    /// Logistic sigmoid: 1 / (1 + exp(−x)).
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
}

impl Activation {
    /// Evaluate the activation on a single scalar.
    #[must_use]
    #[inline]
    pub fn apply_scalar(self, x: f32) -> f32 {
        match self {
            Self::ReLU => relu(x),
            Self::GELU => gelu(x),
            Self::SiLU => silu(x),
            Self::Sigmoid => sigmoid(x),
            Self::Tanh => x.tanh(),
        }
    }

    /// Return every supported variant (useful for exhaustive testing).
    #[must_use]
    pub const fn all_variants() -> &'static [Self] {
        &[Self::ReLU, Self::GELU, Self::SiLU, Self::Sigmoid, Self::Tanh]
    }
}

// ── scalar kernels ─────────────────────────────────────────────────

#[inline]
fn relu(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > 0.0 { x } else { 0.0 }
}

#[inline]
fn gelu(x: f32) -> f32 {
    // Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let c = (2.0_f32 / PI).sqrt();
    0.5 * x * (1.0 + (c * (0.044_715 * x * x).mul_add(x, x)).tanh())
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── vectorised entry points ────────────────────────────────────────

/// Apply `act` element-wise, returning a new vector.
#[must_use = "returns a new vector; use apply_activation_inplace to mutate"]
pub fn apply_activation(act: Activation, data: &[f32]) -> Vec<f32> {
    data.iter().map(|&x| act.apply_scalar(x)).collect()
}

/// Apply `act` element-wise **in-place**.
pub fn apply_activation_inplace(act: Activation, data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = act.apply_scalar(*v);
    }
}

// ── CUDA stubs ─────────────────────────────────────────────────────

#[cfg(any(feature = "gpu", feature = "cuda"))]
mod cuda {
    use super::Activation;

    /// Placeholder: launch an activation kernel on the GPU.
    pub fn launch_activation(_act: Activation, _data: &[f32], _out: &mut [f32]) {
        unimplemented!("CUDA activation kernel not yet wired");
    }
}
