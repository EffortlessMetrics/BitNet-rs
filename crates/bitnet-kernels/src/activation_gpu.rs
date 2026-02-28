//! GPU-accelerated activation functions for transformer inference.
//! Provides CPU reference implementations matching the OpenCL silu_gate.cl kernels.

/// Apply SiLU (Sigmoid Linear Unit) element-wise: x * sigmoid(x).
pub fn silu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            let sig = 1.0 / (1.0 + (-x).exp());
            x * sig
        })
        .collect()
}

/// Fused SiLU-gate: silu(gate) * value.
/// Common in LLaMA-style FFN where the up-projection splits into gate and value halves.
///
/// # Panics
/// Panics if gate and value have different lengths.
pub fn silu_gate_fused(gate: &[f32], value: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), value.len(), "gate/value length mismatch");
    gate.iter()
        .zip(value.iter())
        .map(|(&g, &v)| {
            let sig = 1.0 / (1.0 + (-g).exp());
            (g * sig) * v
        })
        .collect()
}

/// GELU (Gaussian Error Linear Unit) with tanh approximation.
pub fn gelu(input: &[f32]) -> Vec<f32> {
    let c: f32 = (2.0f32 / std::f32::consts::PI).sqrt(); // sqrt(2/pi)
    input
        .iter()
        .map(|&x| {
            let inner = c * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "index {i}: {x} vs {y} (diff {})", (x - y).abs());
        }
    }

    #[test]
    fn silu_zero() {
        let out = silu(&[0.0]);
        assert_eq!(out, vec![0.0]);
    }

    #[test]
    fn silu_positive() {
        let out = silu(&[1.0]);
        // silu(1) = 1 * sigmoid(1) = 1 / (1 + e^-1) ~ 0.7311
        approx_eq(&out, &[0.7311], 0.001);
    }

    #[test]
    fn silu_negative() {
        let out = silu(&[-1.0]);
        // silu(-1) = -1 * sigmoid(-1) = -1 / (1 + e) ~ -0.2689
        approx_eq(&out, &[-0.2689], 0.001);
    }

    #[test]
    fn silu_batch() {
        let out = silu(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert_eq!(out.len(), 5);
        assert!(out[2] == 0.0); // silu(0) = 0
        assert!(out[3] > 0.0); // silu(1) > 0
        assert!(out[0] < 0.0); // silu(-2) < 0
    }

    #[test]
    fn silu_gate_fused_basic() {
        let gate = vec![1.0, 0.0, -1.0];
        let value = vec![2.0, 3.0, 4.0];
        let out = silu_gate_fused(&gate, &value);
        // silu(1)*2 ~ 1.4623, silu(0)*3 = 0, silu(-1)*4 ~ -1.0756
        approx_eq(&out, &[1.4623, 0.0, -1.0756], 0.001);
    }

    #[test]
    #[should_panic(expected = "gate/value length mismatch")]
    fn silu_gate_length_mismatch() {
        silu_gate_fused(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn gelu_zero() {
        let out = gelu(&[0.0]);
        approx_eq(&out, &[0.0], 0.001);
    }

    #[test]
    fn gelu_positive() {
        let out = gelu(&[1.0]);
        // GELU(1) ~ 0.8412
        approx_eq(&out, &[0.8412], 0.001);
    }

    #[test]
    fn gelu_negative() {
        let out = gelu(&[-1.0]);
        // GELU(-1) ~ -0.1588
        approx_eq(&out, &[-0.1588], 0.001);
    }

    #[test]
    fn gelu_symmetry() {
        // GELU is not symmetric, but GELU(x) + GELU(-x) ~ x for small x
        let out_pos = gelu(&[0.5]);
        let out_neg = gelu(&[-0.5]);
        // GELU(0.5) ~ 0.3457, GELU(-0.5) ~ -0.1543
        assert!(out_pos[0] > 0.0);
        assert!(out_neg[0] < 0.0);
        assert!(out_pos[0].abs() > out_neg[0].abs());
    }

    #[test]
    fn silu_large_input() {
        let out = silu(&[10.0]);
        // sigmoid(10) ~ 1.0, so silu(10) ~ 10.0
        approx_eq(&out, &[10.0], 0.001);
    }

    #[test]
    fn empty_inputs() {
        assert!(silu(&[]).is_empty());
        assert!(gelu(&[]).is_empty());
        assert!(silu_gate_fused(&[], &[]).is_empty());
    }
}
