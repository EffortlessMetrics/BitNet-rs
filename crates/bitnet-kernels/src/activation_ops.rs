//! Standard activation function implementations.
//!
//! CPU-optimized activation functions for SLM inference:
//! SiLU, GELU, ReLU², Mish, Swish variants, with in-place and batched ops.

use std::f32::consts::PI;

/// SiLU (Sigmoid Linear Unit) / Swish: x * sigmoid(x).
/// Used by Phi-4, LLaMA-3, Mistral.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SiLU applied to a slice (in-place).
pub fn silu_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = silu(*v);
    }
}

/// SiLU producing a new vector.
pub fn silu_vec(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&x| silu(x)).collect()
}

/// GELU (Gaussian Error Linear Unit) — approximate (tanh version).
/// Used by GPT-2, BERT, many models.
#[inline]
pub fn gelu_approx(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// GELU approximate applied to a slice.
pub fn gelu_approx_vec(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&x| gelu_approx(x)).collect()
}

/// GELU exact using error function.
#[inline]
pub fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_approx(x / 2.0_f32.sqrt()))
}

/// Approximate error function (Abramowitz & Stegun).
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

/// ReLU² (squared ReLU): max(0, x)². Used by BitNet.
#[inline]
pub fn relu_squared(x: f32) -> f32 {
    let r = x.max(0.0);
    r * r
}

/// ReLU² applied to a slice.
pub fn relu_squared_vec(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&x| relu_squared(x)).collect()
}

/// Standard ReLU: max(0, x).
#[inline]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Mish: x * tanh(softplus(x)).
#[inline]
pub fn mish(x: f32) -> f32 {
    x * ((1.0 + x.exp()).ln()).tanh()
}

/// Sigmoid: 1 / (1 + exp(-x)).
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU-gated FFN: silu(gate) * up.
/// Common pattern in LLaMA/Mistral/Phi FFN blocks.
pub fn silu_gate_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up.iter()).map(|(&g, &u)| silu(g) * u).collect()
}

/// In-place gated activation: gate\[i] = silu(gate\[i]) * up\[i].
pub fn silu_gate_mul_inplace(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g = silu(*g) * u;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_zero() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu_positive() {
        let v = silu(1.0);
        // silu(1) = 1 / (1 + e^-1) ≈ 0.7311
        assert!((v - 0.7311).abs() < 0.001);
    }

    #[test]
    fn test_silu_vec() {
        let data = vec![0.0, 1.0, -1.0];
        let result = silu_vec(&data);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!(result[1] > 0.0);
        assert!(result[2] < 0.0);
    }

    #[test]
    fn test_silu_inplace() {
        let mut data = vec![0.0, 1.0];
        silu_inplace(&mut data);
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - silu(1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_approx() {
        assert!((gelu_approx(0.0) - 0.0).abs() < 1e-6);
        assert!(gelu_approx(1.0) > 0.0);
        assert!(gelu_approx(-3.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu_exact_near_approx() {
        let x = 1.0;
        let approx = gelu_approx(x);
        let exact = gelu_exact(x);
        assert!((approx - exact).abs() < 0.01);
    }

    #[test]
    fn test_relu_squared() {
        assert_eq!(relu_squared(-1.0), 0.0);
        assert!((relu_squared(2.0) - 4.0).abs() < 1e-6);
        assert!((relu_squared(0.5) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(-5.0), 0.0);
        assert_eq!(relu(3.0), 3.0);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_mish() {
        assert!((mish(0.0) - 0.0).abs() < 1e-5);
        assert!(mish(1.0) > 0.0);
    }

    #[test]
    fn test_silu_gate_mul() {
        let gate = vec![1.0, 2.0];
        let up = vec![3.0, 4.0];
        let result = silu_gate_mul(&gate, &up);
        assert!((result[0] - silu(1.0) * 3.0).abs() < 1e-6);
        assert!((result[1] - silu(2.0) * 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu_gate_mul_inplace() {
        let mut gate = vec![1.0, 2.0];
        let up = vec![3.0, 4.0];
        silu_gate_mul_inplace(&mut gate, &up);
        assert!((gate[0] - silu(1.0) * 3.0).abs() < 1e-6);
    }
}
