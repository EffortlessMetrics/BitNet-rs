//! CPU feed-forward network (FFN) kernels for transformer layers.
//!
//! Provides standard and gated FFN forward passes used as the second
//! major component of each transformer block (after attention).
//!
//! # Supported variants
//!
//! - **Standard FFN**: `activation(input · W_up^T) · W_down^T`
//! - **Gated FFN** (SwiGLU / GeGLU / ReGLU):
//!   `(activation(input · W_gate^T) ⊙ (input · W_up^T)) · W_down^T`
//! - **Batched FFN**: standard FFN over multiple independent rows.

use bitnet_common::{KernelError, Result};

// ── Activation enum (FFN-local) ─────────────────────────────────────

/// Activation function used inside the FFN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    /// GELU (tanh approximation).
    GeLU,
    /// SiLU (a.k.a. Swish-1): `x · σ(x)`.
    SiLU,
    /// ReLU: `max(0, x)`.
    ReLU,
}

// ── Scalar activation helpers ───────────────────────────────────────

#[inline]
fn activate(x: f32, act: FfnActivation) -> f32 {
    match act {
        FfnActivation::GeLU => {
            const SQRT_2_OVER_PI: f32 = 0.797_884_6;
            const COEFF: f32 = 0.044_715;
            let x3 = x * x * x;
            let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            0.5 * x * (1.0 + inner.tanh())
        }
        FfnActivation::SiLU => x / (1.0 + (-x).exp()),
        FfnActivation::ReLU => x.max(0.0),
    }
}

// ── Configuration ───────────────────────────────────────────────────

/// Configuration for a feed-forward network layer.
///
/// Describes the shapes used in the two-projection FFN:
///
/// - `W_up`:   `[intermediate_dim, hidden_dim]`
/// - `W_down`: `[hidden_dim, intermediate_dim]`
/// - optional `W_gate`: same shape as `W_up` (for gated variants)
#[derive(Debug, Clone)]
pub struct FfnConfig {
    /// Model hidden dimension (input/output width).
    pub hidden_dim: usize,
    /// Intermediate (expanded) dimension.
    pub intermediate_dim: usize,
    /// Activation applied after the up-projection.
    pub activation: FfnActivation,
}

impl FfnConfig {
    /// Create a new FFN configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if either dimension is zero.
    pub fn new(
        hidden_dim: usize,
        intermediate_dim: usize,
        activation: FfnActivation,
    ) -> Result<Self> {
        if hidden_dim == 0 || intermediate_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "FFN dimensions must be non-zero: \
                     hidden_dim={hidden_dim}, intermediate_dim={intermediate_dim}"
                ),
            }
            .into());
        }
        Ok(Self { hidden_dim, intermediate_dim, activation })
    }
}

// ── Validation helpers ──────────────────────────────────────────────

/// Validate buffer sizes for a standard (non-gated) FFN forward pass.
fn validate_ffn_buffers(
    input: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    config: &FfnConfig,
    batch: usize,
) -> Result<()> {
    if batch == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "batch size must be non-zero".into() }.into()
        );
    }

    let h = config.hidden_dim;
    let inter = config.intermediate_dim;

    let input_expected = batch * h;
    if input.len() < input_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "input buffer too small: expected >= {input_expected}, got {}",
                input.len()
            ),
        }
        .into());
    }

    let w_up_expected = inter * h;
    if w_up.len() < w_up_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "w_up buffer too small: expected >= {w_up_expected}, got {}",
                w_up.len()
            ),
        }
        .into());
    }

    let w_down_expected = h * inter;
    if w_down.len() < w_down_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "w_down buffer too small: expected >= {w_down_expected}, got {}",
                w_down.len()
            ),
        }
        .into());
    }

    Ok(())
}

/// Validate buffer sizes for a gated FFN forward pass (adds `w_gate`).
fn validate_gated_ffn_buffers(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    config: &FfnConfig,
    batch: usize,
) -> Result<()> {
    validate_ffn_buffers(input, w_up, w_down, config, batch)?;

    let w_gate_expected = config.intermediate_dim * config.hidden_dim;
    if w_gate.len() < w_gate_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "w_gate buffer too small: expected >= {w_gate_expected}, got {}",
                w_gate.len()
            ),
        }
        .into());
    }

    Ok(())
}

// ── Matrix-vector helper ────────────────────────────────────────────

/// Compute `y = x · W^T` for a single row.
///
/// - `x`:  `[in_dim]`
/// - `w`:  row-major `[out_dim, in_dim]`
/// - `y`:  `[out_dim]`
#[inline]
fn matvec(x: &[f32], w: &[f32], y: &mut [f32], in_dim: usize, out_dim: usize) {
    for (j, y_j) in y.iter_mut().enumerate().take(out_dim) {
        let mut acc = 0.0f32;
        let w_off = j * in_dim;
        for k in 0..in_dim {
            acc += x[k] * w[w_off + k];
        }
        *y_j = acc;
    }
}

// ── Standard FFN ────────────────────────────────────────────────────

/// Standard (non-gated) FFN forward pass for a single input vector.
///
/// Computes `output = W_down · activation(W_up · input)` where:
///
/// - `input`:  `[hidden_dim]`
/// - `w_up`:   row-major `[intermediate_dim, hidden_dim]`
/// - `w_down`: row-major `[hidden_dim, intermediate_dim]`
/// - returns:  `[hidden_dim]`
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on dimension mismatch.
pub fn ffn_forward(
    input: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    config: &FfnConfig,
) -> Result<Vec<f32>> {
    validate_ffn_buffers(input, w_up, w_down, config, 1)?;

    let h = config.hidden_dim;
    let inter = config.intermediate_dim;

    // Up-project: hidden → intermediate
    let mut hidden = vec![0.0f32; inter];
    matvec(input, w_up, &mut hidden, h, inter);

    // Activation
    for v in &mut hidden {
        *v = activate(*v, config.activation);
    }

    // Down-project: intermediate → hidden
    let mut output = vec![0.0f32; h];
    matvec(&hidden, w_down, &mut output, inter, h);

    Ok(output)
}

// ── Gated FFN ───────────────────────────────────────────────────────

/// Gated FFN forward pass (SwiGLU / GeGLU / ReGLU style).
///
/// Computes `output = W_down · (activation(W_gate · input) ⊙ (W_up · input))`
///
/// - `input`:  `[hidden_dim]`
/// - `w_gate`: row-major `[intermediate_dim, hidden_dim]`
/// - `w_up`:   row-major `[intermediate_dim, hidden_dim]`
/// - `w_down`: row-major `[hidden_dim, intermediate_dim]`
/// - returns:  `[hidden_dim]`
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on dimension mismatch.
pub fn gated_ffn_forward(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    config: &FfnConfig,
) -> Result<Vec<f32>> {
    validate_gated_ffn_buffers(input, w_gate, w_up, w_down, config, 1)?;

    let h = config.hidden_dim;
    let inter = config.intermediate_dim;

    // Gate projection
    let mut gate = vec![0.0f32; inter];
    matvec(input, w_gate, &mut gate, h, inter);

    // Up projection
    let mut up = vec![0.0f32; inter];
    matvec(input, w_up, &mut up, h, inter);

    // activation(gate) ⊙ up
    for i in 0..inter {
        gate[i] = activate(gate[i], config.activation) * up[i];
    }

    // Down-project
    let mut output = vec![0.0f32; h];
    matvec(&gate, w_down, &mut output, inter, h);

    Ok(output)
}

// ── Batched FFN ─────────────────────────────────────────────────────

/// Batched standard FFN forward pass.
///
/// Applies [`ffn_forward`] independently to each of the `batch` rows
/// packed contiguously in `input`.
///
/// - `input`:  `[batch * hidden_dim]` (row-major)
/// - returns:  `[batch * hidden_dim]`
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on dimension mismatch or
/// zero batch size.
pub fn ffn_forward_batched(
    input: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    config: &FfnConfig,
    batch: usize,
) -> Result<Vec<f32>> {
    validate_ffn_buffers(input, w_up, w_down, config, batch)?;

    let h = config.hidden_dim;
    let inter = config.intermediate_dim;
    let mut output = vec![0.0f32; batch * h];

    for b in 0..batch {
        let x = &input[b * h..(b + 1) * h];

        // Up-project
        let mut hidden = vec![0.0f32; inter];
        matvec(x, w_up, &mut hidden, h, inter);

        // Activation
        for v in &mut hidden {
            *v = activate(*v, config.activation);
        }

        // Down-project
        let out_row = &mut output[b * h..(b + 1) * h];
        matvec(&hidden, w_down, out_row, inter, h);
    }

    Ok(output)
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    // ── Config tests ────────────────────────────────────────────────

    #[test]
    fn test_config_new() {
        let cfg = FfnConfig::new(4, 8, FfnActivation::SiLU).unwrap();
        assert_eq!(cfg.hidden_dim, 4);
        assert_eq!(cfg.intermediate_dim, 8);
        assert_eq!(cfg.activation, FfnActivation::SiLU);
    }

    #[test]
    fn test_config_rejects_zero_hidden() {
        assert!(FfnConfig::new(0, 8, FfnActivation::ReLU).is_err());
    }

    #[test]
    fn test_config_rejects_zero_intermediate() {
        assert!(FfnConfig::new(4, 0, FfnActivation::GeLU).is_err());
    }

    // ── Known-value: identity-like with ReLU ────────────────────────

    #[test]
    fn test_ffn_identity_relu() {
        // hidden_dim=2, intermediate_dim=2
        // W_up = I₂, W_down = I₂, ReLU
        // input = [1, 2] → up = [1, 2] → relu = [1, 2] → down = [1, 2]
        let cfg = FfnConfig::new(2, 2, FfnActivation::ReLU).unwrap();
        #[rustfmt::skip]
        let w_up = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        #[rustfmt::skip]
        let w_down = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        let input = vec![1.0, 2.0];
        let out = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();
        assert_close(&out, &[1.0, 2.0], 1e-6);
    }

    // ── Known-value: small hand-computed result ─────────────────────

    #[test]
    fn test_ffn_known_small_relu() {
        // hidden=2, inter=3, ReLU
        // W_up = [[1, 0], [0, 1], [1, 1]]  (3×2)
        // input = [2, 3]
        // up = [2, 3, 5] → relu = [2, 3, 5]
        // W_down = [[1, 1, 1], [0, 1, 0]]  (2×3)
        // out = [2+3+5, 0+3+0] = [10, 3]
        let cfg = FfnConfig::new(2, 3, FfnActivation::ReLU).unwrap();
        #[rustfmt::skip]
        let w_up = vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ];
        #[rustfmt::skip]
        let w_down = vec![
            1.0, 1.0, 1.0,
            0.0, 1.0, 0.0,
        ];
        let input = vec![2.0, 3.0];
        let out = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();
        assert_close(&out, &[10.0, 3.0], 1e-6);
    }

    #[test]
    fn test_ffn_known_negative_input_relu() {
        // ReLU clamps negatives to zero.
        // hidden=2, inter=2, W_up=I, W_down=I
        // input = [-1, 2] → up = [-1, 2] → relu = [0, 2] → down = [0, 2]
        let cfg = FfnConfig::new(2, 2, FfnActivation::ReLU).unwrap();
        #[rustfmt::skip]
        let eye = vec![1.0, 0.0, 0.0, 1.0];
        let input = vec![-1.0, 2.0];
        let out = ffn_forward(&input, &eye, &eye, &cfg).unwrap();
        assert_close(&out, &[0.0, 2.0], 1e-6);
    }

    // ── Gated FFN known values ──────────────────────────────────────

    #[test]
    fn test_gated_ffn_relu_known() {
        // hidden=2, inter=2, ReLU
        // W_gate = I₂, W_up = I₂, W_down = I₂
        // input = [3, -1]
        // gate = [3, -1] → relu(gate) = [3, 0]
        // up   = [3, -1]
        // hidden = [3*3, 0*(-1)] = [9, 0]
        // out = [9, 0]
        let cfg = FfnConfig::new(2, 2, FfnActivation::ReLU).unwrap();
        #[rustfmt::skip]
        let eye = vec![1.0, 0.0, 0.0, 1.0];
        let input = vec![3.0, -1.0];
        let out = gated_ffn_forward(&input, &eye, &eye, &eye, &cfg).unwrap();
        assert_close(&out, &[9.0, 0.0], 1e-6);
    }

    #[test]
    fn test_gated_ffn_silu_known() {
        // hidden=1, inter=1, SiLU
        // All weight matrices are [[1.0]] (scalars)
        // input = [2.0]
        // gate = [2.0], up = [2.0]
        // silu(2) * 2 → (2/(1+exp(-2))) * 2
        let cfg = FfnConfig::new(1, 1, FfnActivation::SiLU).unwrap();
        let w = vec![1.0];
        let input = vec![2.0];
        let out = gated_ffn_forward(&input, &w, &w, &w, &cfg).unwrap();
        let silu_2 = 2.0f32 / (1.0 + (-2.0f32).exp());
        let expected = silu_2 * 2.0; // activation(gate) * up
        // then down-project: expected * 1.0
        assert!((out[0] - expected).abs() < 1e-5, "got {}, expected {}", out[0], expected);
    }

    #[test]
    fn test_gated_ffn_gelu_known() {
        // hidden=1, inter=1, GeLU, w=[[1]]
        // input = [1.0]
        // gate=1, up=1 → gelu(1)*1 → down → gelu(1)
        let cfg = FfnConfig::new(1, 1, FfnActivation::GeLU).unwrap();
        let w = vec![1.0];
        let input = vec![1.0];
        let out = gated_ffn_forward(&input, &w, &w, &w, &cfg).unwrap();
        let gelu_1 = activate(1.0, FfnActivation::GeLU);
        assert!((out[0] - gelu_1).abs() < 1e-5, "got {}, expected {}", out[0], gelu_1);
    }

    // ── Activation function differences ─────────────────────────────

    #[test]
    fn test_activations_produce_different_results() {
        let cfg_gelu = FfnConfig::new(2, 2, FfnActivation::GeLU).unwrap();
        let cfg_silu = FfnConfig::new(2, 2, FfnActivation::SiLU).unwrap();
        let cfg_relu = FfnConfig::new(2, 2, FfnActivation::ReLU).unwrap();

        #[rustfmt::skip]
        let eye = vec![1.0, 0.0, 0.0, 1.0];
        let input = vec![1.0, -0.5];

        let out_gelu = ffn_forward(&input, &eye, &eye, &cfg_gelu).unwrap();
        let out_silu = ffn_forward(&input, &eye, &eye, &cfg_silu).unwrap();
        let out_relu = ffn_forward(&input, &eye, &eye, &cfg_relu).unwrap();

        // All three must differ for at least one element.
        assert_ne!(out_gelu, out_silu);
        assert_ne!(out_gelu, out_relu);
        assert_ne!(out_silu, out_relu);
    }

    #[test]
    fn test_relu_zeros_negative_gelu_silu_dont() {
        // With identity weights and input = [-1], ReLU output = 0
        // but GeLU and SiLU produce small negative values.
        let cfg_relu = FfnConfig::new(1, 1, FfnActivation::ReLU).unwrap();
        let cfg_gelu = FfnConfig::new(1, 1, FfnActivation::GeLU).unwrap();
        let cfg_silu = FfnConfig::new(1, 1, FfnActivation::SiLU).unwrap();

        let w = vec![1.0];
        let input = vec![-1.0];

        let out_relu = ffn_forward(&input, &w, &w, &cfg_relu).unwrap();
        let out_gelu = ffn_forward(&input, &w, &w, &cfg_gelu).unwrap();
        let out_silu = ffn_forward(&input, &w, &w, &cfg_silu).unwrap();

        assert_eq!(out_relu[0], 0.0);
        assert!(out_gelu[0] < 0.0, "gelu(-1) should be negative: {}", out_gelu[0]);
        assert!(out_silu[0] < 0.0, "silu(-1) should be negative: {}", out_silu[0]);
    }

    // ── Batched FFN ─────────────────────────────────────────────────

    #[test]
    fn test_batched_matches_individual() {
        let cfg = FfnConfig::new(2, 3, FfnActivation::SiLU).unwrap();
        #[rustfmt::skip]
        let w_up = vec![
            0.5, -0.3,
            0.2,  0.7,
            -0.1, 0.4,
        ];
        #[rustfmt::skip]
        let w_down = vec![
            0.3,  0.1, -0.2,
            -0.5, 0.6,  0.8,
        ];
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows of [hidden_dim=2]

        let batched = ffn_forward_batched(&input, &w_up, &w_down, &cfg, 2).unwrap();

        let row0 = ffn_forward(&input[0..2], &w_up, &w_down, &cfg).unwrap();
        let row1 = ffn_forward(&input[2..4], &w_up, &w_down, &cfg).unwrap();

        assert_close(&batched[0..2], &row0, 1e-6);
        assert_close(&batched[2..4], &row1, 1e-6);
    }

    #[test]
    fn test_batched_single_row_matches_unbatched() {
        let cfg = FfnConfig::new(3, 4, FfnActivation::GeLU).unwrap();
        let w_up: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let w_down: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let input = vec![1.0, -0.5, 0.3];

        let single = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();
        let batched = ffn_forward_batched(&input, &w_up, &w_down, &cfg, 1).unwrap();

        assert_close(&single, &batched, 1e-6);
    }

    // ── Property: output shape ──────────────────────────────────────

    #[test]
    fn test_output_shape_standard() {
        for (h, inter) in [(1, 1), (2, 4), (4, 8), (8, 16)] {
            let cfg = FfnConfig::new(h, inter, FfnActivation::ReLU).unwrap();
            let input = vec![1.0f32; h];
            let w_up = vec![0.1f32; inter * h];
            let w_down = vec![0.1f32; h * inter];
            let out = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();
            assert_eq!(out.len(), h, "output should be [hidden_dim]");
        }
    }

    #[test]
    fn test_output_shape_gated() {
        for (h, inter) in [(1, 1), (3, 6), (8, 16)] {
            let cfg = FfnConfig::new(h, inter, FfnActivation::SiLU).unwrap();
            let input = vec![1.0f32; h];
            let w_gate = vec![0.1f32; inter * h];
            let w_up = vec![0.1f32; inter * h];
            let w_down = vec![0.1f32; h * inter];
            let out = gated_ffn_forward(&input, &w_gate, &w_up, &w_down, &cfg).unwrap();
            assert_eq!(out.len(), h);
        }
    }

    #[test]
    fn test_output_shape_batched() {
        for batch in [1, 2, 5, 16] {
            let cfg = FfnConfig::new(4, 8, FfnActivation::GeLU).unwrap();
            let input = vec![0.5f32; batch * 4];
            let w_up = vec![0.1f32; 8 * 4];
            let w_down = vec![0.1f32; 4 * 8];
            let out = ffn_forward_batched(&input, &w_up, &w_down, &cfg, batch).unwrap();
            assert_eq!(out.len(), batch * 4);
        }
    }

    // ── Property: batch consistency ─────────────────────────────────

    #[test]
    fn test_batch_rows_independent() {
        // Changing one row should not affect the other.
        let cfg = FfnConfig::new(2, 3, FfnActivation::ReLU).unwrap();
        let w_up = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let w_down = vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0];

        let input_a = vec![1.0, 2.0, 3.0, 4.0];
        let input_b = vec![1.0, 2.0, 99.0, 99.0]; // row 1 changed

        let out_a = ffn_forward_batched(&input_a, &w_up, &w_down, &cfg, 2).unwrap();
        let out_b = ffn_forward_batched(&input_b, &w_up, &w_down, &cfg, 2).unwrap();

        // Row 0 unchanged.
        assert_close(&out_a[0..2], &out_b[0..2], 1e-6);
        // Row 1 should differ.
        assert_ne!(&out_a[2..4], &out_b[2..4]);
    }

    // ── Error handling: dimension mismatches ────────────────────────

    #[test]
    fn test_ffn_input_too_small() {
        let cfg = FfnConfig::new(4, 8, FfnActivation::ReLU).unwrap();
        let input = vec![1.0f32; 2]; // needs 4
        let w_up = vec![0.1f32; 32];
        let w_down = vec![0.1f32; 32];
        assert!(ffn_forward(&input, &w_up, &w_down, &cfg).is_err());
    }

    #[test]
    fn test_ffn_w_up_too_small() {
        let cfg = FfnConfig::new(2, 4, FfnActivation::SiLU).unwrap();
        let input = vec![1.0f32; 2];
        let w_up = vec![0.1f32; 4]; // needs 8
        let w_down = vec![0.1f32; 8];
        assert!(ffn_forward(&input, &w_up, &w_down, &cfg).is_err());
    }

    #[test]
    fn test_ffn_w_down_too_small() {
        let cfg = FfnConfig::new(2, 4, FfnActivation::GeLU).unwrap();
        let input = vec![1.0f32; 2];
        let w_up = vec![0.1f32; 8];
        let w_down = vec![0.1f32; 4]; // needs 8
        assert!(ffn_forward(&input, &w_up, &w_down, &cfg).is_err());
    }

    #[test]
    fn test_gated_ffn_w_gate_too_small() {
        let cfg = FfnConfig::new(2, 4, FfnActivation::SiLU).unwrap();
        let input = vec![1.0f32; 2];
        let w_gate = vec![0.1f32; 4]; // needs 8
        let w_up = vec![0.1f32; 8];
        let w_down = vec![0.1f32; 8];
        assert!(gated_ffn_forward(&input, &w_gate, &w_up, &w_down, &cfg).is_err());
    }

    #[test]
    fn test_batched_zero_batch_rejected() {
        let cfg = FfnConfig::new(2, 4, FfnActivation::ReLU).unwrap();
        let input = vec![1.0f32; 2];
        let w_up = vec![0.1f32; 8];
        let w_down = vec![0.1f32; 8];
        assert!(ffn_forward_batched(&input, &w_up, &w_down, &cfg, 0).is_err());
    }

    #[test]
    fn test_batched_input_too_small_for_batch() {
        let cfg = FfnConfig::new(2, 4, FfnActivation::ReLU).unwrap();
        let input = vec![1.0f32; 2]; // needs 4 for batch=2
        let w_up = vec![0.1f32; 8];
        let w_down = vec![0.1f32; 8];
        assert!(ffn_forward_batched(&input, &w_up, &w_down, &cfg, 2).is_err());
    }

    // ── Property: zero input → zero output (ReLU + identity) ────────

    #[test]
    fn test_zero_input_zero_output() {
        let cfg = FfnConfig::new(3, 4, FfnActivation::ReLU).unwrap();
        let input = vec![0.0f32; 3];
        let w_up = vec![0.5f32; 12];
        // For ReLU: act(0) = 0, so down-project on zeros → zeros
        let w_down = vec![0.5f32; 12];
        let out = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();
        assert_close(&out, &[0.0, 0.0, 0.0], 1e-7);
    }

    // ── Property: gated FFN with zero gate → zero output ────────────

    #[test]
    fn test_gated_zero_gate_weights_zero_output() {
        // If W_gate is all zeros, gate projection = 0, activation(0) varies
        // but for ReLU: relu(0)=0, so output = 0 regardless of W_up
        let cfg = FfnConfig::new(2, 3, FfnActivation::ReLU).unwrap();
        let input = vec![5.0, 10.0];
        let w_gate = vec![0.0f32; 6]; // zero gate
        let w_up = vec![1.0f32; 6];
        let w_down = vec![1.0f32; 6];
        let out = gated_ffn_forward(&input, &w_gate, &w_up, &w_down, &cfg).unwrap();
        assert_close(&out, &[0.0, 0.0], 1e-7);
    }

    // ── Larger regression test ──────────────────────────────────────

    #[test]
    fn test_ffn_larger_manual_check() {
        // hidden=3, inter=4, ReLU, verify row-by-row against naive matmul
        let cfg = FfnConfig::new(3, 4, FfnActivation::ReLU).unwrap();
        let input = vec![1.0, -2.0, 0.5];
        #[rustfmt::skip]
        let w_up = vec![
            0.1,  0.2, 0.3,
           -0.1,  0.4, 0.5,
            0.3, -0.2, 0.1,
            0.2,  0.1, -0.3,
        ];
        #[rustfmt::skip]
        let w_down = vec![
            0.1, -0.2, 0.3, 0.4,
            0.5,  0.1, 0.2, -0.1,
           -0.3,  0.4, 0.1, 0.2,
        ];

        let out = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();

        // Manual: up = W_up · input
        // up[0] = 0.1*1 + 0.2*(-2) + 0.3*0.5 = 0.1-0.4+0.15 = -0.15
        // up[1] = -0.1 + 0.4*(-2) + 0.5*0.5 = -0.1-0.8+0.25 = -0.65
        // up[2] = 0.3 + (-0.2)*(-2) + 0.1*0.5 = 0.3+0.4+0.05 = 0.75
        // up[3] = 0.2 + 0.1*(-2) + (-0.3)*0.5 = 0.2-0.2-0.15 = -0.15
        // relu: [0, 0, 0.75, 0]
        // out = W_down · relu
        // out[0] = 0.1*0 + (-0.2)*0 + 0.3*0.75 + 0.4*0 = 0.225
        // out[1] = 0.5*0 + 0.1*0 + 0.2*0.75 + (-0.1)*0 = 0.15
        // out[2] = (-0.3)*0 + 0.4*0 + 0.1*0.75 + 0.2*0 = 0.075
        assert_close(&out, &[0.225, 0.15, 0.075], 1e-5);
    }

    // ── Scalar activation unit tests ────────────────────────────────

    #[test]
    fn test_activate_relu_values() {
        assert_eq!(activate(0.0, FfnActivation::ReLU), 0.0);
        assert_eq!(activate(1.0, FfnActivation::ReLU), 1.0);
        assert_eq!(activate(-1.0, FfnActivation::ReLU), 0.0);
        assert_eq!(activate(100.0, FfnActivation::ReLU), 100.0);
    }

    #[test]
    fn test_activate_silu_values() {
        // SiLU(0) = 0
        assert!(activate(0.0, FfnActivation::SiLU).abs() < 1e-7);
        // SiLU(large positive) ≈ x
        let val = activate(100.0, FfnActivation::SiLU);
        assert!((val - 100.0).abs() < 0.01);
        // SiLU(large negative) ≈ 0
        let val = activate(-100.0, FfnActivation::SiLU);
        assert!(val.abs() < 0.01);
    }

    #[test]
    fn test_activate_gelu_values() {
        // GeLU(0) = 0
        assert!(activate(0.0, FfnActivation::GeLU).abs() < 1e-7);
        // GeLU(1) ≈ 0.8412
        let val = activate(1.0, FfnActivation::GeLU);
        assert!((val - 0.8412).abs() < 1e-3, "gelu(1)={val}");
    }
}
