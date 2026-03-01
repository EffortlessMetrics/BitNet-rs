//! OpenCL FFN (Feed-Forward Network) block for transformer layers.
//!
//! # Architecture
//!
//! Transformer FFN blocks come in two variants:
//!
//! - **Standard FFN**: `output = down_proj(activation(up_proj(x)))`
//! - **Gated FFN** (Llama/BitNet): `output = down_proj(activation(gate_proj(x)) * up_proj(x))`
//!
//! The gated variant uses an element-wise product between the activated gate
//! projection and the up projection, which improves training dynamics for
//! SwiGLU-style architectures.
//!
//! # CPU reference
//!
//! [`ffn_forward_ref`] and [`gated_ffn_forward_ref`] provide scalar CPU
//! implementations for correctness testing and non-GPU environments.
//!
//! # OpenCL kernel
//!
//! [`FFN_CL`] contains fused OpenCL C source for the gated FFN, performing
//! gate+up projection, SiLU activation, element-wise multiply, and down
//! projection in a single dispatch.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Activation types
// ---------------------------------------------------------------------------

/// Activation function used between projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// Sigmoid Linear Unit: `x * sigmoid(x)`.
    SiLU,
    /// Gaussian Error Linear Unit (exact).
    GELU,
    /// GELU with fast tanh approximation.
    GELUApprox,
    /// Rectified Linear Unit: `max(0, x)`.
    ReLU,
    /// Swish with beta=1 (equivalent to SiLU).
    Swish,
}

impl ActivationType {
    /// Apply the activation function to a scalar value.
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::SiLU | Self::Swish => x * sigmoid(x),
            Self::GELU => {
                // Exact: x * 0.5 * (1 + erf(x / sqrt(2)))
                x * 0.5 * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2))
            }
            Self::GELUApprox => {
                // Tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
                let c = (2.0_f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
            }
            Self::ReLU => x.max(0.0),
        }
    }
}

/// Standard sigmoid: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for an FFN block.
#[derive(Debug, Clone)]
pub struct FfnConfig {
    /// Input/output dimension (model hidden size).
    pub hidden_size: usize,
    /// Inner dimension of the FFN (typically 4× or ~2.75× hidden_size).
    pub intermediate_size: usize,
    /// Activation function between projections.
    pub activation: ActivationType,
}

impl FfnConfig {
    /// Create a new FFN configuration.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if any dimension is zero.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        activation: ActivationType,
    ) -> Result<Self> {
        if hidden_size == 0 || intermediate_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "FFN dimensions must be non-zero: \
                     hidden_size={hidden_size}, intermediate_size={intermediate_size}"
                ),
            }
            .into());
        }
        Ok(Self { hidden_size, intermediate_size, activation })
    }

    /// BitNet-compatible configuration: hidden=2048, intermediate=5632, SiLU.
    pub fn bitnet_2b() -> Self {
        Self { hidden_size: 2048, intermediate_size: 5632, activation: ActivationType::SiLU }
    }
}

// ---------------------------------------------------------------------------
// CPU reference: standard FFN
// ---------------------------------------------------------------------------

/// Standard FFN forward pass: `down_proj(activation(up_proj(x)))`.
///
/// # Layout
///
/// - `x`:      `[seq_len, hidden_size]` — input activations
/// - `w_up`:   `[hidden_size, intermediate_size]` — up-projection weights
/// - `w_down`: `[intermediate_size, hidden_size]` — down-projection weights
/// - `output`: `[seq_len, hidden_size]` — result
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on dimension mismatch.
pub fn ffn_forward_ref(
    x: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &mut [f32],
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
    activation: ActivationType,
) -> Result<()> {
    validate_ffn_args(x, w_up, w_down, output, seq_len, hidden_size, intermediate_size)?;

    let mut intermediate = vec![0.0_f32; seq_len * intermediate_size];

    // Up projection: intermediate = x @ w_up  [seq_len, intermediate_size]
    matmul_ref(x, w_up, &mut intermediate, seq_len, hidden_size, intermediate_size);

    // Activation in-place
    for v in &mut intermediate {
        *v = activation.apply(*v);
    }

    // Down projection: output = intermediate @ w_down  [seq_len, hidden_size]
    matmul_ref(&intermediate, w_down, output, seq_len, intermediate_size, hidden_size);

    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference: gated FFN
// ---------------------------------------------------------------------------

/// Gated FFN forward pass: `down_proj(activation(gate_proj(x)) * up_proj(x))`.
///
/// Used in Llama, BitNet, and other SwiGLU-style architectures.
///
/// # Layout
///
/// - `x`:      `[seq_len, hidden_size]`
/// - `w_gate`: `[hidden_size, intermediate_size]` — gate projection weights
/// - `w_up`:   `[hidden_size, intermediate_size]` — up projection weights
/// - `w_down`: `[intermediate_size, hidden_size]` — down projection weights
/// - `output`: `[seq_len, hidden_size]`
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on dimension mismatch.
#[allow(clippy::too_many_arguments)]
pub fn gated_ffn_forward_ref(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &mut [f32],
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
    activation: ActivationType,
) -> Result<()> {
    validate_gated_ffn_args(
        x,
        w_gate,
        w_up,
        w_down,
        output,
        seq_len,
        hidden_size,
        intermediate_size,
    )?;

    let inter_len = seq_len * intermediate_size;
    let mut gate = vec![0.0_f32; inter_len];
    let mut up = vec![0.0_f32; inter_len];

    // Gate projection: gate = x @ w_gate
    matmul_ref(x, w_gate, &mut gate, seq_len, hidden_size, intermediate_size);

    // Up projection: up = x @ w_up
    matmul_ref(x, w_up, &mut up, seq_len, hidden_size, intermediate_size);

    // Element-wise: hidden = activation(gate) * up
    let mut hidden = vec![0.0_f32; inter_len];
    for i in 0..inter_len {
        hidden[i] = activation.apply(gate[i]) * up[i];
    }

    // Down projection: output = hidden @ w_down
    matmul_ref(&hidden, w_down, output, seq_len, intermediate_size, hidden_size);

    Ok(())
}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C source for fused gated FFN.
///
/// Performs gate+up projection, SiLU activation, element-wise multiply,
/// and down projection. Temporary buffers `temp_gate` and `temp_up` must
/// be pre-allocated as `[seq_len * intermediate_size]`.
pub const FFN_CL: &str = r#"
// SiLU activation: x * sigmoid(x)
float silu(float x) {
    return x / (1.0f + exp(-x));
}

// Gated FFN: down_proj(silu(gate_proj(x)) * up_proj(x))
//
// Work-item layout: global_id(0) = intermediate column, global_id(1) = seq row.
// Phase 1: compute gate and up projections.
// Phase 2: apply SiLU to gate, multiply with up, store in temp_gate.
// Phase 3: down projection from temp_gate to output.
__kernel void gated_ffn(
    __global const float* x,
    __global const float* w_gate,
    __global const float* w_up,
    __global const float* w_down,
    __global float* output,
    __global float* temp_gate,
    __global float* temp_up,
    const int seq_len,
    const int hidden_size,
    const int intermediate_size)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= seq_len || col >= intermediate_size) return;

    // Gate projection: temp_gate[row, col] = sum_k x[row,k] * w_gate[k, col]
    float gate_val = 0.0f;
    float up_val = 0.0f;
    for (int k = 0; k < hidden_size; k++) {
        float xk = x[row * hidden_size + k];
        gate_val += xk * w_gate[k * intermediate_size + col];
        up_val   += xk * w_up[k * intermediate_size + col];
    }

    // SiLU activation on gate, element-wise multiply with up
    float activated = silu(gate_val) * up_val;
    temp_gate[row * intermediate_size + col] = activated;

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Down projection: each thread computes one output element
    if (col < hidden_size) {
        float sum = 0.0f;
        for (int k = 0; k < intermediate_size; k++) {
            sum += temp_gate[row * intermediate_size + k]
                 * w_down[k * hidden_size + col];
        }
        output[row * hidden_size + col] = sum;
    }
}
"#;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive row-major matmul: `C[M,N] = A[M,K] @ B[K,N]`.
fn matmul_ref(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

/// Build an identity-like weight matrix `[rows, cols]` where diagonal = 1.
#[cfg(test)]
fn identity_weight(rows: usize, cols: usize) -> Vec<f32> {
    let mut w = vec![0.0_f32; rows * cols];
    let diag = rows.min(cols);
    for i in 0..diag {
        w[i * cols + i] = 1.0;
    }
    w
}

fn validate_ffn_args(
    x: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &[f32],
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<()> {
    if seq_len == 0 || hidden_size == 0 || intermediate_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "FFN dimensions must be non-zero: \
                 seq_len={seq_len}, hidden={hidden_size}, inter={intermediate_size}"
            ),
        }
        .into());
    }
    let x_expected = seq_len * hidden_size;
    let up_expected = hidden_size * intermediate_size;
    let down_expected = intermediate_size * hidden_size;
    let out_expected = seq_len * hidden_size;
    if x.len() < x_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("FFN: x length {} < expected {x_expected}", x.len()),
        }
        .into());
    }
    if w_up.len() < up_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("FFN: w_up length {} < expected {up_expected}", w_up.len()),
        }
        .into());
    }
    if w_down.len() < down_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("FFN: w_down length {} < expected {down_expected}", w_down.len()),
        }
        .into());
    }
    if output.len() < out_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("FFN: output length {} < expected {out_expected}", output.len()),
        }
        .into());
    }
    Ok(())
}

fn validate_gated_ffn_args(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &[f32],
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<()> {
    validate_ffn_args(x, w_up, w_down, output, seq_len, hidden_size, intermediate_size)?;
    let gate_expected = hidden_size * intermediate_size;
    if w_gate.len() < gate_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("FFN: w_gate length {} < expected {gate_expected}", w_gate.len()),
        }
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn assert_near(a: f32, b: f32, tol: f32, msg: &str) {
        assert!((a - b).abs() < tol, "{msg}: {a} vs {b} (tol={tol})");
    }

    // ===================================================================
    // Activation function tests
    // ===================================================================

    #[test]
    fn test_silu_at_zero() {
        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert_near(ActivationType::SiLU.apply(0.0), 0.0, EPS, "SiLU(0)");
    }

    #[test]
    fn test_silu_positive() {
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.7311
        let result = ActivationType::SiLU.apply(1.0);
        assert_near(result, 0.7311, 1e-3, "SiLU(1)");
    }

    #[test]
    fn test_silu_negative() {
        // SiLU(-1) = -1 * sigmoid(-1) ≈ -0.2689
        let result = ActivationType::SiLU.apply(-1.0);
        assert_near(result, -0.2689, 1e-3, "SiLU(-1)");
    }

    #[test]
    fn test_silu_large_positive() {
        // SiLU(10) ≈ 10 * 1.0 ≈ 10.0
        let result = ActivationType::SiLU.apply(10.0);
        assert_near(result, 10.0, 1e-3, "SiLU(10)");
    }

    #[test]
    fn test_gelu_at_zero() {
        // GELU(0) = 0
        assert_near(ActivationType::GELU.apply(0.0), 0.0, EPS, "GELU(0)");
    }

    #[test]
    fn test_gelu_positive() {
        // GELU(1) ≈ 0.8413
        let result = ActivationType::GELU.apply(1.0);
        assert_near(result, 0.8413, 1e-3, "GELU(1)");
    }

    #[test]
    fn test_gelu_negative() {
        // GELU(-1) ≈ -0.1587
        let result = ActivationType::GELU.apply(-1.0);
        assert_near(result, -0.1587, 1e-3, "GELU(-1)");
    }

    #[test]
    fn test_gelu_approx_matches_gelu() {
        for &x in &[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let exact = ActivationType::GELU.apply(x);
            let approx = ActivationType::GELUApprox.apply(x);
            assert_near(exact, approx, 0.02, &format!("GELU vs GELUApprox at {x}"));
        }
    }

    #[test]
    fn test_relu_positive() {
        assert_near(ActivationType::ReLU.apply(3.0), 3.0, EPS, "ReLU(3)");
    }

    #[test]
    fn test_relu_negative() {
        assert_near(ActivationType::ReLU.apply(-3.0), 0.0, EPS, "ReLU(-3)");
    }

    #[test]
    fn test_relu_zero() {
        assert_near(ActivationType::ReLU.apply(0.0), 0.0, EPS, "ReLU(0)");
    }

    #[test]
    fn test_swish_equals_silu() {
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let silu = ActivationType::SiLU.apply(x);
            let swish = ActivationType::Swish.apply(x);
            assert_near(silu, swish, EPS, &format!("SiLU vs Swish at {x}"));
        }
    }

    // ===================================================================
    // FfnConfig tests
    // ===================================================================

    #[test]
    fn test_config_new_valid() {
        let cfg = FfnConfig::new(256, 512, ActivationType::SiLU).unwrap();
        assert_eq!(cfg.hidden_size, 256);
        assert_eq!(cfg.intermediate_size, 512);
        assert_eq!(cfg.activation, ActivationType::SiLU);
    }

    #[test]
    fn test_config_rejects_zero_hidden() {
        assert!(FfnConfig::new(0, 512, ActivationType::SiLU).is_err());
    }

    #[test]
    fn test_config_rejects_zero_intermediate() {
        assert!(FfnConfig::new(256, 0, ActivationType::SiLU).is_err());
    }

    #[test]
    fn test_config_bitnet_2b() {
        let cfg = FfnConfig::bitnet_2b();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.intermediate_size, 5632);
        assert_eq!(cfg.activation, ActivationType::SiLU);
    }

    // ===================================================================
    // Standard FFN tests
    // ===================================================================

    #[test]
    fn test_ffn_zero_input_gives_zero_output() {
        let h = 4;
        let inter = 8;
        let seq = 2;
        let x = vec![0.0_f32; seq * h];
        let w_up = vec![1.0_f32; h * inter];
        let w_down = vec![1.0_f32; inter * h];
        let mut output = vec![99.0_f32; seq * h];
        ffn_forward_ref(&x, &w_up, &w_down, &mut output, seq, h, inter, ActivationType::SiLU)
            .unwrap();
        // SiLU(0) = 0, so output should be all zeros
        for &v in &output {
            assert_near(v, 0.0, EPS, "zero input → zero output");
        }
    }

    #[test]
    fn test_ffn_relu_zero_input() {
        let h = 4;
        let inter = 6;
        let seq = 1;
        let x = vec![0.0_f32; seq * h];
        let w_up = vec![1.0_f32; h * inter];
        let w_down = vec![1.0_f32; inter * h];
        let mut output = vec![0.0_f32; seq * h];
        ffn_forward_ref(&x, &w_up, &w_down, &mut output, seq, h, inter, ActivationType::ReLU)
            .unwrap();
        for &v in &output {
            assert_near(v, 0.0, EPS, "ReLU(0) → zero output");
        }
    }

    #[test]
    fn test_ffn_identity_weights_relu() {
        // With identity weights and ReLU on positive input, output ≈ input
        let h = 4;
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w_up = identity_weight(h, h);
        let w_down = identity_weight(h, h);
        let mut output = vec![0.0_f32; h];
        ffn_forward_ref(&x, &w_up, &w_down, &mut output, 1, h, h, ActivationType::ReLU).unwrap();
        for i in 0..h {
            assert_near(output[i], x[i], EPS, &format!("identity ReLU [{i}]"));
        }
    }

    #[test]
    fn test_ffn_seq_len_1() {
        let h = 2;
        let inter = 3;
        let x = vec![1.0, 0.5];
        let w_up = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2,3]
        let w_down = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // [3,2]
        let mut output = vec![0.0_f32; 2];
        ffn_forward_ref(&x, &w_up, &w_down, &mut output, 1, h, inter, ActivationType::ReLU)
            .unwrap();
        // up = [1.0, 0.5, 0.0], relu → [1.0, 0.5, 0.0]
        // down = [1.0*1+0.5*0+0*0, 1.0*0+0.5*1+0*0] = [1.0, 0.5]
        assert_near(output[0], 1.0, EPS, "seq1[0]");
        assert_near(output[1], 0.5, EPS, "seq1[1]");
    }

    #[test]
    fn test_ffn_multiple_sequences() {
        let h = 2;
        let inter = 2;
        let seq = 3;
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let w_up = identity_weight(h, inter);
        let w_down = identity_weight(inter, h);
        let mut output = vec![0.0_f32; seq * h];
        ffn_forward_ref(&x, &w_up, &w_down, &mut output, seq, h, inter, ActivationType::ReLU)
            .unwrap();
        // Identity weights + ReLU on non-negative → pass-through
        assert_near(output[0], 1.0, EPS, "seq0[0]");
        assert_near(output[1], 0.0, EPS, "seq0[1]");
        assert_near(output[2], 0.0, EPS, "seq1[0]");
        assert_near(output[3], 1.0, EPS, "seq1[1]");
    }

    #[test]
    fn test_ffn_negative_input_relu() {
        let h = 2;
        let inter = 2;
        let x = vec![-1.0, -2.0];
        let w_up = identity_weight(h, inter);
        let w_down = identity_weight(inter, h);
        let mut output = vec![0.0_f32; h];
        ffn_forward_ref(&x, &w_up, &w_down, &mut output, 1, h, inter, ActivationType::ReLU)
            .unwrap();
        // ReLU clips negatives → output = 0
        for &v in &output {
            assert_near(v, 0.0, EPS, "negative input ReLU");
        }
    }

    #[test]
    fn test_ffn_rejects_short_x() {
        let w_up = vec![0.0_f32; 4];
        let w_down = vec![0.0_f32; 4];
        let mut output = vec![0.0_f32; 2];
        let result =
            ffn_forward_ref(&[0.0], &w_up, &w_down, &mut output, 1, 2, 2, ActivationType::ReLU);
        assert!(result.is_err());
    }

    #[test]
    fn test_ffn_rejects_short_w_up() {
        let x = vec![0.0_f32; 4];
        let w_down = vec![0.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        let result =
            ffn_forward_ref(&x, &[0.0], &w_down, &mut output, 2, 2, 2, ActivationType::ReLU);
        assert!(result.is_err());
    }

    #[test]
    fn test_ffn_rejects_short_w_down() {
        let x = vec![0.0_f32; 4];
        let w_up = vec![0.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        let result = ffn_forward_ref(&x, &w_up, &[0.0], &mut output, 2, 2, 2, ActivationType::ReLU);
        assert!(result.is_err());
    }

    #[test]
    fn test_ffn_rejects_short_output() {
        let x = vec![0.0_f32; 4];
        let w_up = vec![0.0_f32; 4];
        let w_down = vec![0.0_f32; 4];
        let result = ffn_forward_ref(&x, &w_up, &w_down, &mut [0.0], 2, 2, 2, ActivationType::ReLU);
        assert!(result.is_err());
    }

    #[test]
    fn test_ffn_rejects_zero_seq_len() {
        let result =
            ffn_forward_ref(&[], &[0.0; 4], &[0.0; 4], &mut [], 0, 2, 2, ActivationType::ReLU);
        assert!(result.is_err());
    }

    // ===================================================================
    // Gated FFN tests
    // ===================================================================

    #[test]
    fn test_gated_ffn_zero_gate_gives_zero_output() {
        // If gate weights are all zero, gate_proj(x)=0, SiLU(0)=0, so output=0
        let h = 4;
        let inter = 6;
        let seq = 2;
        let x = vec![1.0_f32; seq * h];
        let w_gate = vec![0.0_f32; h * inter];
        let w_up = vec![1.0_f32; h * inter];
        let w_down = vec![1.0_f32; inter * h];
        let mut output = vec![99.0_f32; seq * h];
        gated_ffn_forward_ref(
            &x,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            seq,
            h,
            inter,
            ActivationType::SiLU,
        )
        .unwrap();
        for &v in &output {
            assert_near(v, 0.0, EPS, "zero gate → zero output");
        }
    }

    #[test]
    fn test_gated_ffn_zero_up_gives_zero_output() {
        // If up weights are all zero, up_proj(x)=0, gate*0=0, so output=0
        let h = 4;
        let inter = 6;
        let seq = 1;
        let x = vec![1.0_f32; seq * h];
        let w_gate = vec![1.0_f32; h * inter];
        let w_up = vec![0.0_f32; h * inter];
        let w_down = vec![1.0_f32; inter * h];
        let mut output = vec![99.0_f32; seq * h];
        gated_ffn_forward_ref(
            &x,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            seq,
            h,
            inter,
            ActivationType::SiLU,
        )
        .unwrap();
        for &v in &output {
            assert_near(v, 0.0, EPS, "zero up → zero output");
        }
    }

    #[test]
    fn test_gated_ffn_relu_gate_ones_equivalent() {
        // With ReLU, gate=large positive → ReLU(gate)≈gate,
        // so gated FFN ≈ standard FFN when gate activations are large.
        // Test: identity gate (large values), identity up, identity down
        let h = 3;
        let inter = 3;
        let x = vec![1.0, 2.0, 3.0];

        // Gate weights produce large positive values so ReLU is identity-like
        let w_gate = identity_weight(h, inter);
        let w_up = identity_weight(h, inter);
        let w_down = identity_weight(inter, h);

        let mut gated_out = vec![0.0_f32; h];
        gated_ffn_forward_ref(
            &x,
            &w_gate,
            &w_up,
            &w_down,
            &mut gated_out,
            1,
            h,
            inter,
            ActivationType::ReLU,
        )
        .unwrap();

        // Expected: gate=x, ReLU(gate)=x, up=x, hidden=x*x, down=x*x
        for i in 0..h {
            let expected = x[i] * x[i]; // ReLU(x_i) * x_i = x_i^2
            assert_near(gated_out[i], expected, EPS, &format!("gated ReLU [{i}]"));
        }
    }

    #[test]
    fn test_gated_ffn_zero_input() {
        let h = 4;
        let inter = 8;
        let x = vec![0.0_f32; h];
        let w_gate = vec![1.0_f32; h * inter];
        let w_up = vec![1.0_f32; h * inter];
        let w_down = vec![1.0_f32; inter * h];
        let mut output = vec![99.0_f32; h];
        gated_ffn_forward_ref(
            &x,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            1,
            h,
            inter,
            ActivationType::SiLU,
        )
        .unwrap();
        for &v in &output {
            assert_near(v, 0.0, EPS, "zero input gated → zero output");
        }
    }

    #[test]
    fn test_gated_ffn_seq_len_1() {
        let h = 2;
        let inter = 2;
        let x = vec![1.0, 1.0];
        let w_gate = identity_weight(h, inter);
        let w_up = identity_weight(h, inter);
        let w_down = identity_weight(inter, h);
        let mut output = vec![0.0_f32; h];
        gated_ffn_forward_ref(
            &x,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            1,
            h,
            inter,
            ActivationType::SiLU,
        )
        .unwrap();
        // gate = [1,1], SiLU(1)≈0.7311, up = [1,1]
        // hidden = [0.7311, 0.7311], down = [0.7311, 0.7311]
        let expected = ActivationType::SiLU.apply(1.0) * 1.0;
        for i in 0..h {
            assert_near(output[i], expected, 1e-3, &format!("gated seq1 [{i}]"));
        }
    }

    #[test]
    fn test_gated_ffn_rejects_short_gate() {
        let x = vec![0.0_f32; 4];
        let w_up = vec![0.0_f32; 4];
        let w_down = vec![0.0_f32; 4];
        let mut output = vec![0.0_f32; 4];
        let result = gated_ffn_forward_ref(
            &x,
            &[0.0],
            &w_up,
            &w_down,
            &mut output,
            2,
            2,
            2,
            ActivationType::SiLU,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gated_ffn_multiple_sequences() {
        let h = 2;
        let inter = 2;
        let seq = 2;
        let x = vec![1.0, 0.0, 0.0, 1.0];
        let w_gate = identity_weight(h, inter);
        let w_up = identity_weight(h, inter);
        let w_down = identity_weight(inter, h);
        let mut output = vec![0.0_f32; seq * h];
        gated_ffn_forward_ref(
            &x,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            seq,
            h,
            inter,
            ActivationType::ReLU,
        )
        .unwrap();
        // seq0: x=[1,0], gate=[1,0], ReLU→[1,0], up=[1,0], hidden=[1,0]
        assert_near(output[0], 1.0, EPS, "gated multi seq0[0]");
        assert_near(output[1], 0.0, EPS, "gated multi seq0[1]");
        // seq1: x=[0,1], gate=[0,1], ReLU→[0,1], up=[0,1], hidden=[0,1]
        assert_near(output[2], 0.0, EPS, "gated multi seq1[0]");
        assert_near(output[3], 1.0, EPS, "gated multi seq1[1]");
    }

    // ===================================================================
    // Dimension ratio tests
    // ===================================================================

    #[test]
    fn test_ffn_hidden_gt_intermediate() {
        // hidden_size > intermediate_size (down-expansion)
        let h = 8;
        let inter = 4;
        let x = vec![0.5_f32; h];
        let w_up = vec![0.1_f32; h * inter];
        let w_down = vec![0.1_f32; inter * h];
        let mut output = vec![0.0_f32; h];
        ffn_forward_ref(&x, &w_up, &w_down, &mut output, 1, h, inter, ActivationType::ReLU)
            .unwrap();
        // Just check it runs without error and produces finite output
        assert!(output.iter().all(|v| v.is_finite()), "output should be finite");
    }

    #[test]
    fn test_ffn_4x_expansion() {
        // Standard 4× expansion ratio
        let h = 8;
        let inter = 32;
        let x = vec![0.1_f32; h];
        let w_up = vec![0.01_f32; h * inter];
        let w_down = vec![0.01_f32; inter * h];
        let mut output = vec![0.0_f32; h];
        ffn_forward_ref(&x, &w_up, &w_down, &mut output, 1, h, inter, ActivationType::SiLU)
            .unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gated_ffn_non_square_ratio() {
        // intermediate = 2.75 * hidden (common in Llama)
        let h = 4;
        let inter = 11;
        let x = vec![0.1_f32; h];
        let w_gate = vec![0.01_f32; h * inter];
        let w_up = vec![0.01_f32; h * inter];
        let w_down = vec![0.01_f32; inter * h];
        let mut output = vec![0.0_f32; h];
        gated_ffn_forward_ref(
            &x,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            1,
            h,
            inter,
            ActivationType::SiLU,
        )
        .unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // ===================================================================
    // BitNet configuration tests
    // ===================================================================

    #[test]
    fn test_ffn_bitnet_config_runs() {
        // Scaled-down test with BitNet ratio: 2048/5632 ≈ 0.364
        // Use small dims with same ratio: 8/22
        let h = 8;
        let inter = 22;
        let seq = 2;
        let x = vec![0.01_f32; seq * h];
        let w_gate = vec![0.01_f32; h * inter];
        let w_up = vec![0.01_f32; h * inter];
        let w_down = vec![0.01_f32; inter * h];
        let mut output = vec![0.0_f32; seq * h];
        gated_ffn_forward_ref(
            &x,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            seq,
            h,
            inter,
            ActivationType::SiLU,
        )
        .unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ffn_bitnet_2b_config_object() {
        let cfg = FfnConfig::bitnet_2b();
        // Verify it constructs without panic
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.intermediate_size, 5632);

        // Run a minimal forward with these dimensions (seq=1, small weights)
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let x = vec![0.001_f32; h];
        let w_gate = vec![0.001_f32; h * inter];
        let w_up = vec![0.001_f32; h * inter];
        let w_down = vec![0.001_f32; inter * h];
        let mut output = vec![0.0_f32; h];
        gated_ffn_forward_ref(
            &x,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            1,
            h,
            inter,
            ActivationType::SiLU,
        )
        .unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // ===================================================================
    // Numerical stability
    // ===================================================================

    #[test]
    fn test_ffn_large_weights_no_overflow() {
        let h = 4;
        let inter = 4;
        let x = vec![1.0_f32; h];
        let w_up = vec![10.0_f32; h * inter];
        let w_down = vec![0.01_f32; inter * h];
        let mut output = vec![0.0_f32; h];
        ffn_forward_ref(&x, &w_up, &w_down, &mut output, 1, h, inter, ActivationType::SiLU)
            .unwrap();
        assert!(output.iter().all(|v| v.is_finite()), "large weights should not overflow");
    }

    #[test]
    fn test_gated_ffn_large_weights_no_overflow() {
        let h = 4;
        let inter = 4;
        let x = vec![1.0_f32; h];
        let w_gate = vec![10.0_f32; h * inter];
        let w_up = vec![10.0_f32; h * inter];
        let w_down = vec![0.001_f32; inter * h];
        let mut output = vec![0.0_f32; h];
        gated_ffn_forward_ref(
            &x,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            1,
            h,
            inter,
            ActivationType::SiLU,
        )
        .unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_activation_large_negative_silu() {
        // SiLU(-100) ≈ -100 * sigmoid(-100) ≈ -100 * 0 ≈ 0
        let result = ActivationType::SiLU.apply(-100.0);
        assert!(result.is_finite());
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_activation_large_positive_gelu() {
        // GELU(100) ≈ 100
        let result = ActivationType::GELU.apply(100.0);
        assert_near(result, 100.0, 0.1, "GELU(100)");
    }

    // ===================================================================
    // Edge cases
    // ===================================================================

    #[test]
    fn test_ffn_single_element() {
        // hidden=1, intermediate=1, seq=1
        let x = vec![2.0_f32];
        let w_up = vec![1.0_f32];
        let w_down = vec![1.0_f32];
        let mut output = vec![0.0_f32];
        ffn_forward_ref(&x, &w_up, &w_down, &mut output, 1, 1, 1, ActivationType::ReLU).unwrap();
        // up = 2.0, ReLU(2.0) = 2.0, down = 2.0
        assert_near(output[0], 2.0, EPS, "single element FFN");
    }

    #[test]
    fn test_gated_ffn_single_element() {
        let x = vec![2.0_f32];
        let w_gate = vec![1.0_f32];
        let w_up = vec![1.0_f32];
        let w_down = vec![1.0_f32];
        let mut output = vec![0.0_f32];
        gated_ffn_forward_ref(
            &x,
            &w_gate,
            &w_up,
            &w_down,
            &mut output,
            1,
            1,
            1,
            ActivationType::ReLU,
        )
        .unwrap();
        // gate=2.0, ReLU(2.0)=2.0, up=2.0, hidden=4.0, down=4.0
        assert_near(output[0], 4.0, EPS, "single element gated FFN");
    }

    #[test]
    fn test_ffn_all_negative_weights_silu() {
        let h = 2;
        let inter = 2;
        let x = vec![1.0, 1.0];
        let w_up = vec![-1.0_f32; h * inter];
        let w_down = vec![1.0_f32; inter * h];
        let mut output = vec![0.0_f32; h];
        ffn_forward_ref(&x, &w_up, &w_down, &mut output, 1, h, inter, ActivationType::SiLU)
            .unwrap();
        // up_proj yields negative values, SiLU of negative ≈ small negative
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // ===================================================================
    // OpenCL kernel source tests
    // ===================================================================

    #[test]
    fn test_opencl_kernel_source_not_empty() {
        assert!(!FFN_CL.is_empty());
    }

    #[test]
    fn test_opencl_kernel_contains_entry_point() {
        assert!(FFN_CL.contains("__kernel void gated_ffn"));
    }

    #[test]
    fn test_opencl_kernel_contains_silu() {
        assert!(FFN_CL.contains("silu"));
    }

    #[test]
    fn test_opencl_kernel_contains_barrier() {
        assert!(FFN_CL.contains("barrier(CLK_GLOBAL_MEM_FENCE)"));
    }

    // ===================================================================
    // Cross-variant consistency
    // ===================================================================

    #[test]
    fn test_gated_vs_standard_ffn_with_identity_gate() {
        // When gate weights = identity and activation = ReLU on positive input,
        // gated FFN with gate producing x means hidden = x * x.
        let h = 3;
        let inter = 3;
        let x = vec![1.0, 2.0, 3.0];
        let w_id = identity_weight(h, inter);

        let mut std_out = vec![0.0_f32; h];
        ffn_forward_ref(&x, &w_id, &w_id, &mut std_out, 1, h, inter, ActivationType::ReLU).unwrap();
        // Standard: up=x, ReLU(x)=x, down=x → output=x

        let mut gated_out = vec![0.0_f32; h];
        gated_ffn_forward_ref(
            &x,
            &w_id,
            &w_id,
            &w_id,
            &mut gated_out,
            1,
            h,
            inter,
            ActivationType::ReLU,
        )
        .unwrap();
        // Gated: gate=x, ReLU(x)=x, up=x, hidden=x*x, down=x*x

        // Gated output = x^2, standard output = x
        for i in 0..h {
            assert_near(std_out[i], x[i], EPS, &format!("standard [{i}]"));
            assert_near(gated_out[i], x[i] * x[i], EPS, &format!("gated [{i}]"));
        }
    }

    #[test]
    fn test_ffn_activation_sweep() {
        // Verify all activation types work with standard FFN
        let h = 4;
        let inter = 4;
        let x = vec![0.5_f32; h];
        let w_up = identity_weight(h, inter);
        let w_down = identity_weight(inter, h);
        for act in [
            ActivationType::SiLU,
            ActivationType::GELU,
            ActivationType::GELUApprox,
            ActivationType::ReLU,
            ActivationType::Swish,
        ] {
            let mut output = vec![0.0_f32; h];
            ffn_forward_ref(&x, &w_up, &w_down, &mut output, 1, h, inter, act).unwrap();
            assert!(
                output.iter().all(|v| v.is_finite()),
                "activation {act:?} produced non-finite output"
            );
            // All activations on 0.5 should give positive result
            assert!(
                output.iter().all(|&v| v > 0.0),
                "activation {act:?} on positive input should give positive output"
            );
        }
    }

    #[test]
    fn test_gated_ffn_activation_sweep() {
        let h = 4;
        let inter = 4;
        let x = vec![0.5_f32; h];
        let w = identity_weight(h, inter);
        for act in [
            ActivationType::SiLU,
            ActivationType::GELU,
            ActivationType::GELUApprox,
            ActivationType::ReLU,
            ActivationType::Swish,
        ] {
            let mut output = vec![0.0_f32; h];
            gated_ffn_forward_ref(&x, &w, &w, &w, &mut output, 1, h, inter, act).unwrap();
            assert!(
                output.iter().all(|v| v.is_finite()),
                "gated activation {act:?} produced non-finite output"
            );
        }
    }

    // ===================================================================
    // Matmul helper test
    // ===================================================================

    #[test]
    fn test_matmul_ref_2x2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0_f32; 4];
        matmul_ref(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_identity_weight_square() {
        let w = identity_weight(3, 3);
        #[rustfmt::skip]
        assert_eq!(w, vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
    }

    #[test]
    fn test_identity_weight_rectangular() {
        let w = identity_weight(2, 4);
        #[rustfmt::skip]
        assert_eq!(w, vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ]);
    }
}
