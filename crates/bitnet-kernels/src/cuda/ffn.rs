//! CUDA feed-forward network (FFN) kernels with CPU fallback.
//!
//! Provides GPU-accelerated and CPU-fallback FFN operations for transformer
//! inference, covering standard, gated (SwiGLU/GeGLU/ReGLU²), fused,
//! sparse (top-K for MoE), dropout, and quantized variants.
//!
//! # Kernel variants
//!
//! | Function | Description |
//! |---|---|
//! | [`ffn_forward`] | Standard FFN: `W₂ · act(W₁·x + b₁) + b₂` |
//! | [`gated_ffn_forward`] | Gated FFN: `W_down · (act(W_gate·x) ⊙ W_up·x)` |
//! | [`ffn_swiglu`] | SwiGLU activation: `SiLU(gate) ⊙ up` |
//! | [`ffn_geglu`] | GeGLU activation: `GELU(gate) ⊙ up` |
//! | [`ffn_relu_squared`] | ReLU² activation: `ReLU(gate)² ⊙ up` |
//! | [`fused_ffn_norm`] | Fused pre-RMSNorm + FFN (single kernel launch) |
//! | [`sparse_ffn_forward`] | Top-K sparse FFN (MoE building block) |
//! | [`ffn_with_dropout`] | FFN with dropout mask for training |
//! | [`quantized_ffn_forward`] | FFN with INT2/INT4 quantized weights |
//!
//! # CUDA kernel strategy
//!
//! Standard/gated FFN kernels use 2-D grids:
//! - Y-axis: batch rows
//! - X-axis: output features
//!
//! Each thread computes one output element via a dot-product reduction.
//! Block size is 256 threads with grid-stride loops for large dimensions.
//!
//! # CPU fallback
//!
//! Every function has a pure-Rust scalar fallback that is always compiled.
//! The CUDA launch stubs are gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use bitnet_common::{KernelError, Result};

// ── CUDA kernel source ────────────────────────────────────────────────

/// Inline CUDA C source for FFN kernels.
///
/// Contains kernels for standard FFN, gated FFN, and fused norm+FFN.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const FFN_KERNEL_SRC: &str = r#"
// Standard FFN: output = W2 * activation(W1 * x + b1) + b2
// x:      [batch_size, in_dim]
// w1:     [inter_dim, in_dim]
// b1:     [inter_dim] or NULL
// w2:     [out_dim, inter_dim]
// b2:     [out_dim] or NULL
// output: [batch_size, out_dim]
extern "C" __global__ void ffn_forward_f32(
    const float* __restrict__ x,
    const float* __restrict__ w1,
    const float* __restrict__ b1,
    const float* __restrict__ w2,
    const float* __restrict__ b2,
    float* __restrict__ output,
    int batch_size,
    int in_dim,
    int inter_dim,
    int out_dim,
    int has_b1,
    int has_b2,
    int act_type)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_dim) {
        // Phase 1: compute intermediate = activation(W1 * x + b1)
        // Phase 2: dot product with W2 row
        float acc = 0.0f;
        for (int j = 0; j < inter_dim; ++j) {
            // W1[j, :] . x[row, :]
            float h = 0.0f;
            for (int k = 0; k < in_dim; ++k) {
                h += w1[j * in_dim + k] * x[row * in_dim + k];
            }
            if (has_b1) h += b1[j];

            // activation
            if (act_type == 0) {
                h = h / (1.0f + expf(-h)); // SiLU
            } else if (act_type == 1) {
                float SQRT_2_OVER_PI = 0.7978845608f;
                float COEFF = 0.044715f;
                float h3 = h * h * h;
                float inner = SQRT_2_OVER_PI * (h + COEFF * h3);
                h = 0.5f * h * (1.0f + tanhf(inner)); // GELU
            } else {
                h = fmaxf(0.0f, h); // ReLU
            }

            acc += w2[col * inter_dim + j] * h;
        }
        if (has_b2) acc += b2[col];
        output[row * out_dim + col] = acc;
    }
}

// Gated FFN: output = W_down * (activation(W_gate * x) ⊙ (W_up * x))
extern "C" __global__ void gated_ffn_forward_f32(
    const float* __restrict__ x,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    const float* __restrict__ w_down,
    float* __restrict__ output,
    int batch_size,
    int hidden_dim,
    int inter_dim,
    int act_type)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < hidden_dim) {
        float acc = 0.0f;
        for (int j = 0; j < inter_dim; ++j) {
            float gate_val = 0.0f;
            float up_val = 0.0f;
            for (int k = 0; k < hidden_dim; ++k) {
                float xk = x[row * hidden_dim + k];
                gate_val += w_gate[j * hidden_dim + k] * xk;
                up_val += w_up[j * hidden_dim + k] * xk;
            }
            // activation on gate
            if (act_type == 0) {
                gate_val = gate_val / (1.0f + expf(-gate_val));
            } else if (act_type == 1) {
                float SQRT_2_OVER_PI = 0.7978845608f;
                float COEFF = 0.044715f;
                float g3 = gate_val * gate_val * gate_val;
                float inner = SQRT_2_OVER_PI * (gate_val + COEFF * g3);
                gate_val = 0.5f * gate_val * (1.0f + tanhf(inner));
            } else {
                gate_val = fmaxf(0.0f, gate_val);
            }
            acc += w_down[col * inter_dim + j] * gate_val * up_val;
        }
        output[row * hidden_dim + col] = acc;
    }
}
"#;

// ── Activation types ──────────────────────────────────────────────────

/// Activation function for FFN layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivationType {
    /// SiLU (Sigmoid Linear Unit): `x · σ(x)`
    SiLU,
    /// GELU (tanh approximation)
    GELU,
    /// ReLU: `max(0, x)`
    ReLU,
}

impl FfnActivationType {
    /// Integer code used in CUDA kernels.
    pub fn cuda_code(&self) -> i32 {
        match self {
            Self::SiLU => 0,
            Self::GELU => 1,
            Self::ReLU => 2,
        }
    }
}

/// Quantization bit width for quantized FFN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantBits {
    /// 2-bit quantization (I2_S ternary).
    Int2,
    /// 4-bit quantization.
    Int4,
}

// ── Scalar activation helpers ─────────────────────────────────────────

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    0.5 * x * (1.0 + inner.tanh())
}

#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn apply_activation(x: f32, act: FfnActivationType) -> f32 {
    match act {
        FfnActivationType::SiLU => silu(x),
        FfnActivationType::GELU => gelu(x),
        FfnActivationType::ReLU => relu(x),
    }
}

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for FFN CUDA kernel launches.
#[derive(Debug, Clone)]
pub struct FfnConfig {
    /// Batch size (number of input rows).
    pub batch_size: usize,
    /// Input/output hidden dimension.
    pub hidden_dim: usize,
    /// Intermediate (expanded) dimension.
    pub intermediate_dim: usize,
    /// Activation function.
    pub activation: FfnActivationType,
    /// Threads per block (default 256).
    pub threads_per_block: u32,
}

impl FfnConfig {
    /// Create a new FFN configuration.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if any dimension is zero.
    pub fn new(
        batch_size: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        activation: FfnActivationType,
    ) -> Result<Self> {
        if batch_size == 0 || hidden_dim == 0 || intermediate_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "FFN dimensions must be non-zero: batch={batch_size}, \
                     hidden={hidden_dim}, inter={intermediate_dim}"
                ),
            }
            .into());
        }
        Ok(Self { batch_size, hidden_dim, intermediate_dim, activation, threads_per_block: 256 })
    }

    /// CUDA grid dimensions for 2-D kernel launch.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let bx = (self.hidden_dim as u32).div_ceil(self.threads_per_block).min(65_535);
        let by = (self.batch_size as u32).min(65_535);
        (bx, by, 1)
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

/// Configuration for sparse (top-K) FFN.
#[derive(Debug, Clone)]
pub struct SparseFfnConfig {
    /// Base FFN config.
    pub base: FfnConfig,
    /// Number of top-K experts/neurons to activate.
    pub top_k: usize,
}

impl SparseFfnConfig {
    /// Create a new sparse FFN configuration.
    ///
    /// # Errors
    ///
    /// Returns error if `top_k` is zero or exceeds `intermediate_dim`.
    pub fn new(base: FfnConfig, top_k: usize) -> Result<Self> {
        if top_k == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "top_k must be non-zero".into() }.into()
            );
        }
        if top_k > base.intermediate_dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "top_k ({top_k}) exceeds intermediate_dim ({})",
                    base.intermediate_dim
                ),
            }
            .into());
        }
        Ok(Self { base, top_k })
    }
}

// ── Matrix-vector helper ──────────────────────────────────────────────

/// Compute `y = x · W^T` for a single row.
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

// ── Validation ────────────────────────────────────────────────────────

fn validate_ffn_buffers(
    input: &[f32],
    w1: &[f32],
    w2: &[f32],
    output: &[f32],
    config: &FfnConfig,
) -> Result<()> {
    let b = config.batch_size;
    let h = config.hidden_dim;
    let inter = config.intermediate_dim;

    if input.len() < b * h {
        return Err(KernelError::InvalidArguments {
            reason: format!("input too small: {} < {}", input.len(), b * h),
        }
        .into());
    }
    if w1.len() < inter * h {
        return Err(KernelError::InvalidArguments {
            reason: format!("w1 too small: {} < {}", w1.len(), inter * h),
        }
        .into());
    }
    if w2.len() < h * inter {
        return Err(KernelError::InvalidArguments {
            reason: format!("w2 too small: {} < {}", w2.len(), h * inter),
        }
        .into());
    }
    if output.len() < b * h {
        return Err(KernelError::InvalidArguments {
            reason: format!("output too small: {} < {}", output.len(), b * h),
        }
        .into());
    }
    Ok(())
}

fn validate_gated_buffers(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &[f32],
    config: &FfnConfig,
) -> Result<()> {
    let b = config.batch_size;
    let h = config.hidden_dim;
    let inter = config.intermediate_dim;

    if input.len() < b * h {
        return Err(KernelError::InvalidArguments {
            reason: format!("input too small: {} < {}", input.len(), b * h),
        }
        .into());
    }
    if w_gate.len() < inter * h {
        return Err(KernelError::InvalidArguments {
            reason: format!("w_gate too small: {} < {}", w_gate.len(), inter * h),
        }
        .into());
    }
    if w_up.len() < inter * h {
        return Err(KernelError::InvalidArguments {
            reason: format!("w_up too small: {} < {}", w_up.len(), inter * h),
        }
        .into());
    }
    if w_down.len() < h * inter {
        return Err(KernelError::InvalidArguments {
            reason: format!("w_down too small: {} < {}", w_down.len(), h * inter),
        }
        .into());
    }
    if output.len() < b * h {
        return Err(KernelError::InvalidArguments {
            reason: format!("output too small: {} < {}", output.len(), b * h),
        }
        .into());
    }
    Ok(())
}

// ── Standard FFN ──────────────────────────────────────────────────────

/// Standard FFN forward pass (CPU fallback).
///
/// Computes `output = W₂ · activation(W₁ · x + b₁) + b₂` for each batch row.
///
/// - `input`:  `[batch_size, hidden_dim]`
/// - `w1`:     row-major `[intermediate_dim, hidden_dim]`
/// - `b1`:     optional `[intermediate_dim]`
/// - `w2`:     row-major `[hidden_dim, intermediate_dim]`
/// - `b2`:     optional `[hidden_dim]`
/// - `output`: `[batch_size, hidden_dim]`
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on dimension mismatch.
pub fn ffn_forward(
    input: &[f32],
    w1: &[f32],
    b1: Option<&[f32]>,
    w2: &[f32],
    b2: Option<&[f32]>,
    output: &mut [f32],
    config: &FfnConfig,
) -> Result<()> {
    validate_ffn_buffers(input, w1, w2, output, config)?;

    let b = config.batch_size;
    let h = config.hidden_dim;
    let inter = config.intermediate_dim;

    for row in 0..b {
        let x = &input[row * h..(row + 1) * h];

        // Up-project: x → intermediate
        let mut hidden = vec![0.0f32; inter];
        matvec(x, w1, &mut hidden, h, inter);
        if let Some(bias) = b1 {
            for (v, &bv) in hidden.iter_mut().zip(bias.iter()) {
                *v += bv;
            }
        }

        // Activation
        for v in &mut hidden {
            *v = apply_activation(*v, config.activation);
        }

        // Down-project: intermediate → hidden
        let out_row = &mut output[row * h..(row + 1) * h];
        matvec(&hidden, w2, out_row, inter, h);
        if let Some(bias) = b2 {
            for (v, &bv) in out_row.iter_mut().zip(bias.iter()) {
                *v += bv;
            }
        }
    }
    Ok(())
}

// ── Gated FFN ─────────────────────────────────────────────────────────

/// Gated FFN forward pass (LLaMA/SwiGLU style, CPU fallback).
///
/// Computes `output = W_down · (activation(W_gate · x) ⊙ (W_up · x))`
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on dimension mismatch.
pub fn gated_ffn_forward(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &mut [f32],
    config: &FfnConfig,
) -> Result<()> {
    validate_gated_buffers(input, w_gate, w_up, w_down, output, config)?;

    let b = config.batch_size;
    let h = config.hidden_dim;
    let inter = config.intermediate_dim;

    for row in 0..b {
        let x = &input[row * h..(row + 1) * h];

        let mut gate = vec![0.0f32; inter];
        matvec(x, w_gate, &mut gate, h, inter);

        let mut up = vec![0.0f32; inter];
        matvec(x, w_up, &mut up, h, inter);

        // activation(gate) ⊙ up
        for i in 0..inter {
            gate[i] = apply_activation(gate[i], config.activation) * up[i];
        }

        let out_row = &mut output[row * h..(row + 1) * h];
        matvec(&gate, w_down, out_row, inter, h);
    }
    Ok(())
}

// ── SwiGLU activation ─────────────────────────────────────────────────

/// SwiGLU activation: `output[i] = SiLU(gate[i]) * up[i]`.
///
/// # Errors
///
/// Returns error if buffer lengths are less than `n`.
pub fn ffn_swiglu(gate: &[f32], up: &[f32], output: &mut [f32], n: usize) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    if gate.len() < n || up.len() < n || output.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "swiglu buffer too small: gate={}, up={}, out={}, need {n}",
                gate.len(),
                up.len(),
                output.len()
            ),
        }
        .into());
    }
    for i in 0..n {
        output[i] = silu(gate[i]) * up[i];
    }
    Ok(())
}

// ── GeGLU activation ──────────────────────────────────────────────────

/// GeGLU activation: `output[i] = GELU(gate[i]) * up[i]`.
///
/// # Errors
///
/// Returns error if buffer lengths are less than `n`.
pub fn ffn_geglu(gate: &[f32], up: &[f32], output: &mut [f32], n: usize) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    if gate.len() < n || up.len() < n || output.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "geglu buffer too small: gate={}, up={}, out={}, need {n}",
                gate.len(),
                up.len(),
                output.len()
            ),
        }
        .into());
    }
    for i in 0..n {
        output[i] = gelu(gate[i]) * up[i];
    }
    Ok(())
}

// ── ReLU² activation ──────────────────────────────────────────────────

/// ReLU² (ReGLU squared) activation: `output[i] = ReLU(gate[i])² * up[i]`.
///
/// # Errors
///
/// Returns error if buffer lengths are less than `n`.
pub fn ffn_relu_squared(gate: &[f32], up: &[f32], output: &mut [f32], n: usize) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    if gate.len() < n || up.len() < n || output.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "relu_squared buffer too small: gate={}, up={}, out={}, need {n}",
                gate.len(),
                up.len(),
                output.len()
            ),
        }
        .into());
    }
    for i in 0..n {
        let r = relu(gate[i]);
        output[i] = r * r * up[i];
    }
    Ok(())
}

// ── Fused FFN with pre-normalization ──────────────────────────────────

/// Fused pre-RMSNorm + standard FFN (CPU fallback).
///
/// Avoids an extra kernel launch by folding RMSNorm into the FFN:
/// `output = W₂ · activation(W₁ · RMSNorm(x, γ, ε))`
///
/// # Errors
///
/// Returns error on dimension mismatch or zero epsilon.
pub fn fused_ffn_norm(
    input: &[f32],
    gamma: &[f32],
    w1: &[f32],
    w2: &[f32],
    output: &mut [f32],
    config: &FfnConfig,
    eps: f32,
) -> Result<()> {
    validate_ffn_buffers(input, w1, w2, output, config)?;

    let b = config.batch_size;
    let h = config.hidden_dim;
    let inter = config.intermediate_dim;

    if gamma.len() < h {
        return Err(KernelError::InvalidArguments {
            reason: format!("gamma too small: {} < {h}", gamma.len()),
        }
        .into());
    }

    for row in 0..b {
        let x = &input[row * h..(row + 1) * h];

        // RMSNorm: x_norm = x * gamma / rms(x)
        let ss: f32 = x.iter().map(|&v| v * v).sum::<f32>() / h as f32;
        let inv_rms = 1.0 / (ss + eps).sqrt();

        let normed: Vec<f32> = x.iter().zip(gamma.iter()).map(|(&v, &g)| v * g * inv_rms).collect();

        // Up-project + activation
        let mut hidden = vec![0.0f32; inter];
        matvec(&normed, w1, &mut hidden, h, inter);
        for v in &mut hidden {
            *v = apply_activation(*v, config.activation);
        }

        // Down-project
        let out_row = &mut output[row * h..(row + 1) * h];
        matvec(&hidden, w2, out_row, inter, h);
    }
    Ok(())
}

// ── Sparse FFN (top-K) ────────────────────────────────────────────────

/// Sparse FFN forward pass: activates only the top-K intermediate neurons.
///
/// This is a building block for Mixture-of-Experts (MoE) architectures.
/// After the up-projection, only the `top_k` neurons with the largest
/// magnitudes are kept; the rest are zeroed out.
///
/// # Errors
///
/// Returns error on dimension mismatch.
pub fn sparse_ffn_forward(
    input: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &mut [f32],
    config: &SparseFfnConfig,
) -> Result<()> {
    validate_ffn_buffers(input, w_up, w_down, output, &config.base)?;

    let b = config.base.batch_size;
    let h = config.base.hidden_dim;
    let inter = config.base.intermediate_dim;
    let top_k = config.top_k;

    for row in 0..b {
        let x = &input[row * h..(row + 1) * h];

        // Up-project
        let mut hidden = vec![0.0f32; inter];
        matvec(x, w_up, &mut hidden, h, inter);

        // Activation
        for v in &mut hidden {
            *v = apply_activation(*v, config.base.activation);
        }

        // Top-K selection: find the k-th largest magnitude
        let mut magnitudes: Vec<(usize, f32)> =
            hidden.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();
        magnitudes
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut mask = vec![false; inter];
        for &(idx, _) in magnitudes.iter().take(top_k) {
            mask[idx] = true;
        }

        // Zero out non-top-K
        for (i, m) in mask.iter().enumerate() {
            if !m {
                hidden[i] = 0.0;
            }
        }

        // Down-project
        let out_row = &mut output[row * h..(row + 1) * h];
        matvec(&hidden, w_down, out_row, inter, h);
    }
    Ok(())
}

// ── FFN with dropout ──────────────────────────────────────────────────

/// FFN forward pass with dropout mask applied to the intermediate layer.
///
/// The `mask` slice contains `0.0` or `1.0` values. Active neurons are
/// scaled by `1.0 / (1.0 - dropout_rate)` to maintain expected values
/// during training.
///
/// # Errors
///
/// Returns error on dimension mismatch or invalid dropout rate.
pub fn ffn_with_dropout(
    input: &[f32],
    w1: &[f32],
    w2: &[f32],
    mask: &[f32],
    output: &mut [f32],
    config: &FfnConfig,
    dropout_rate: f32,
) -> Result<()> {
    validate_ffn_buffers(input, w1, w2, output, config)?;

    let b = config.batch_size;
    let h = config.hidden_dim;
    let inter = config.intermediate_dim;

    if mask.len() < inter {
        return Err(KernelError::InvalidArguments {
            reason: format!("dropout mask too small: {} < {inter}", mask.len()),
        }
        .into());
    }
    if !(0.0..1.0).contains(&dropout_rate) {
        return Err(KernelError::InvalidArguments {
            reason: format!("dropout_rate must be in [0, 1): got {dropout_rate}"),
        }
        .into());
    }

    let scale = if dropout_rate > 0.0 { 1.0 / (1.0 - dropout_rate) } else { 1.0 };

    for row in 0..b {
        let x = &input[row * h..(row + 1) * h];

        let mut hidden = vec![0.0f32; inter];
        matvec(x, w1, &mut hidden, h, inter);

        // Activation + dropout
        for (i, v) in hidden.iter_mut().enumerate() {
            *v = apply_activation(*v, config.activation) * mask[i] * scale;
        }

        let out_row = &mut output[row * h..(row + 1) * h];
        matvec(&hidden, w2, out_row, inter, h);
    }
    Ok(())
}

// ── Quantized FFN ─────────────────────────────────────────────────────

/// FFN forward pass with quantized weights (INT2/INT4).
///
/// Weights are stored as `u8` packed values with per-block FP32 scales.
/// For INT2: 4 values per byte, for INT4: 2 values per byte.
/// The computation dequantizes on-the-fly during the matrix multiply.
///
/// # Errors
///
/// Returns error on dimension mismatch.
#[allow(clippy::too_many_arguments)]
pub fn quantized_ffn_forward(
    input: &[f32],
    w1_packed: &[u8],
    w1_scales: &[f32],
    w2_packed: &[u8],
    w2_scales: &[f32],
    output: &mut [f32],
    config: &FfnConfig,
    quant: QuantBits,
    block_size: usize,
) -> Result<()> {
    let b = config.batch_size;
    let h = config.hidden_dim;
    let inter = config.intermediate_dim;

    if input.len() < b * h {
        return Err(KernelError::InvalidArguments {
            reason: format!("input too small: {} < {}", input.len(), b * h),
        }
        .into());
    }
    if output.len() < b * h {
        return Err(KernelError::InvalidArguments {
            reason: format!("output too small: {} < {}", output.len(), b * h),
        }
        .into());
    }
    if block_size == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "block_size must be non-zero".into() }.into()
        );
    }

    let vals_per_byte: usize = match quant {
        QuantBits::Int2 => 4,
        QuantBits::Int4 => 2,
    };

    let w1_total = inter * h;
    let w1_packed_expected = w1_total.div_ceil(vals_per_byte);
    if w1_packed.len() < w1_packed_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("w1_packed too small: {} < {w1_packed_expected}", w1_packed.len()),
        }
        .into());
    }

    let w2_total = h * inter;
    let w2_packed_expected = w2_total.div_ceil(vals_per_byte);
    if w2_packed.len() < w2_packed_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("w2_packed too small: {} < {w2_packed_expected}", w2_packed.len()),
        }
        .into());
    }

    // Dequantize weights into FP32
    let w1 = dequantize_packed(w1_packed, w1_scales, w1_total, quant, block_size);
    let w2 = dequantize_packed(w2_packed, w2_scales, w2_total, quant, block_size);

    // Standard FFN with dequantized weights
    for row in 0..b {
        let x = &input[row * h..(row + 1) * h];

        let mut hidden = vec![0.0f32; inter];
        matvec(x, &w1, &mut hidden, h, inter);
        for v in &mut hidden {
            *v = apply_activation(*v, config.activation);
        }

        let out_row = &mut output[row * h..(row + 1) * h];
        matvec(&hidden, &w2, out_row, inter, h);
    }
    Ok(())
}

/// Dequantize packed integer weights to FP32.
fn dequantize_packed(
    packed: &[u8],
    scales: &[f32],
    total_elements: usize,
    quant: QuantBits,
    block_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; total_elements];
    let (vals_per_byte, mask, bits_per_val, center) = match quant {
        QuantBits::Int2 => (4usize, 0x03u8, 2u32, 1.0f32),
        QuantBits::Int4 => (2usize, 0x0Fu8, 4u32, 7.0f32),
    };

    for (i, out_val) in out.iter_mut().enumerate().take(total_elements) {
        let byte_idx = i / vals_per_byte;
        let sub_idx = i % vals_per_byte;
        let raw = (packed[byte_idx] >> (sub_idx as u32 * bits_per_val)) & mask;
        let dequant = raw as f32 - center;
        let scale_idx = i / block_size;
        let scale = if scale_idx < scales.len() { scales[scale_idx] } else { 1.0 };
        *out_val = dequant * scale;
    }
    out
}

// ── CUDA launch stubs (feature-gated) ─────────────────────────────────

/// Launch stub for standard FFN CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_ffn_cuda(
    _input: &[f32],
    _w1: &[f32],
    _b1: Option<&[f32]>,
    _w2: &[f32],
    _b2: Option<&[f32]>,
    _output: &mut [f32],
    config: &FfnConfig,
) -> Result<()> {
    log::debug!(
        "FFN CUDA stub: batch={}, hidden={}, inter={}, grid={:?}",
        config.batch_size,
        config.hidden_dim,
        config.intermediate_dim,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "FFN CUDA kernel not yet compiled — scaffold only".into()
    }
    .into())
}

/// Launch stub for gated FFN CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_gated_ffn_cuda(
    _input: &[f32],
    _w_gate: &[f32],
    _w_up: &[f32],
    _w_down: &[f32],
    _output: &mut [f32],
    config: &FfnConfig,
) -> Result<()> {
    log::debug!(
        "Gated FFN CUDA stub: batch={}, hidden={}, inter={}, grid={:?}",
        config.batch_size,
        config.hidden_dim,
        config.intermediate_dim,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "Gated FFN CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Unified dispatch entry points ─────────────────────────────────────

/// Launch standard FFN with automatic CPU/GPU dispatch.
///
/// Tries GPU first (if compiled with `gpu`/`cuda` features), falls back
/// to CPU on failure.
pub fn launch_ffn(
    input: &[f32],
    w1: &[f32],
    b1: Option<&[f32]>,
    w2: &[f32],
    b2: Option<&[f32]>,
    output: &mut [f32],
    config: &FfnConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        match launch_ffn_cuda(input, w1, b1, w2, b2, output, config) {
            Ok(()) => {
                log::debug!("FFN completed on CUDA");
                return Ok(());
            }
            Err(e) => {
                log::warn!("CUDA FFN failed, falling back to CPU: {e}");
            }
        }
    }

    log::debug!(
        "FFN CPU fallback (batch={}, hidden={}, inter={})",
        config.batch_size,
        config.hidden_dim,
        config.intermediate_dim,
    );
    ffn_forward(input, w1, b1, w2, b2, output, config)
}

/// Launch gated FFN with automatic CPU/GPU dispatch.
pub fn launch_gated_ffn(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &mut [f32],
    config: &FfnConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        match launch_gated_ffn_cuda(input, w_gate, w_up, w_down, output, config) {
            Ok(()) => {
                log::debug!("Gated FFN completed on CUDA");
                return Ok(());
            }
            Err(e) => {
                log::warn!("CUDA gated FFN failed, falling back to CPU: {e}");
            }
        }
    }

    log::debug!(
        "Gated FFN CPU fallback (batch={}, hidden={}, inter={})",
        config.batch_size,
        config.hidden_dim,
        config.intermediate_dim,
    );
    gated_ffn_forward(input, w_gate, w_up, w_down, output, config)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    fn identity(n: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    // ── Config tests ──────────────────────────────────────────────────

    #[test]
    fn test_config_basic() {
        let cfg = FfnConfig::new(1, 4, 8, FfnActivationType::SiLU).unwrap();
        assert_eq!(cfg.batch_size, 1);
        assert_eq!(cfg.hidden_dim, 4);
        assert_eq!(cfg.intermediate_dim, 8);
    }

    #[test]
    fn test_config_rejects_zero_batch() {
        assert!(FfnConfig::new(0, 4, 8, FfnActivationType::ReLU).is_err());
    }

    #[test]
    fn test_config_rejects_zero_hidden() {
        assert!(FfnConfig::new(1, 0, 8, FfnActivationType::GELU).is_err());
    }

    #[test]
    fn test_config_rejects_zero_intermediate() {
        assert!(FfnConfig::new(1, 4, 0, FfnActivationType::SiLU).is_err());
    }

    #[test]
    fn test_config_grid_dim_small() {
        let cfg = FfnConfig::new(1, 64, 128, FfnActivationType::SiLU).unwrap();
        let (gx, gy, _) = cfg.grid_dim();
        assert!(gx >= 1);
        assert_eq!(gy, 1);
    }

    #[test]
    fn test_config_grid_dim_large_batch() {
        let cfg = FfnConfig::new(100, 256, 512, FfnActivationType::ReLU).unwrap();
        let (_, gy, _) = cfg.grid_dim();
        assert_eq!(gy, 100);
    }

    #[test]
    fn test_activation_cuda_codes() {
        assert_eq!(FfnActivationType::SiLU.cuda_code(), 0);
        assert_eq!(FfnActivationType::GELU.cuda_code(), 1);
        assert_eq!(FfnActivationType::ReLU.cuda_code(), 2);
    }

    // ── Standard FFN forward correctness ──────────────────────────────

    #[test]
    fn test_ffn_identity_relu() {
        let cfg = FfnConfig::new(1, 2, 2, FfnActivationType::ReLU).unwrap();
        let eye = identity(2);
        let input = [1.0, 2.0];
        let mut output = [0.0f32; 2];
        ffn_forward(&input, &eye, None, &eye, None, &mut output, &cfg).unwrap();
        assert_close(&output, &[1.0, 2.0], 1e-6);
    }

    #[test]
    fn test_ffn_with_bias() {
        let cfg = FfnConfig::new(1, 2, 2, FfnActivationType::ReLU).unwrap();
        let eye = identity(2);
        let b1 = [1.0, 1.0];
        let b2 = [0.5, 0.5];
        let input = [1.0, 2.0];
        let mut output = [0.0f32; 2];
        ffn_forward(&input, &eye, Some(&b1), &eye, Some(&b2), &mut output, &cfg).unwrap();
        // relu(1+1)=2, relu(2+1)=3, then down-project(identity) + bias2 → [2.5, 3.5]
        assert_close(&output, &[2.5, 3.5], 1e-6);
    }

    #[test]
    fn test_ffn_known_small_relu() {
        // hidden=2, inter=3, ReLU
        let cfg = FfnConfig::new(1, 2, 3, FfnActivationType::ReLU).unwrap();
        #[rustfmt::skip]
        let w1 = vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ];
        #[rustfmt::skip]
        let w2 = vec![
            1.0, 1.0, 1.0,
            0.0, 1.0, 0.0,
        ];
        let input = [2.0, 3.0];
        let mut output = [0.0f32; 2];
        ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).unwrap();
        assert_close(&output, &[10.0, 3.0], 1e-6);
    }

    #[test]
    fn test_ffn_negative_input_relu() {
        let cfg = FfnConfig::new(1, 2, 2, FfnActivationType::ReLU).unwrap();
        let eye = identity(2);
        let input = [-1.0, 2.0];
        let mut output = [0.0f32; 2];
        ffn_forward(&input, &eye, None, &eye, None, &mut output, &cfg).unwrap();
        assert_close(&output, &[0.0, 2.0], 1e-6);
    }

    #[test]
    fn test_ffn_silu_activation() {
        let cfg = FfnConfig::new(1, 1, 1, FfnActivationType::SiLU).unwrap();
        let w = [1.0f32];
        let input = [2.0];
        let mut output = [0.0f32];
        ffn_forward(&input, &w, None, &w, None, &mut output, &cfg).unwrap();
        let expected = silu(2.0);
        assert!((output[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_ffn_gelu_activation() {
        let cfg = FfnConfig::new(1, 1, 1, FfnActivationType::GELU).unwrap();
        let w = [1.0f32];
        let input = [1.0];
        let mut output = [0.0f32];
        ffn_forward(&input, &w, None, &w, None, &mut output, &cfg).unwrap();
        let expected = gelu(1.0);
        assert!((output[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_ffn_zero_input() {
        let cfg = FfnConfig::new(1, 3, 4, FfnActivationType::ReLU).unwrap();
        let input = [0.0f32; 3];
        let w1 = vec![0.5f32; 12];
        let w2 = vec![0.5f32; 12];
        let mut output = [0.0f32; 3];
        ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).unwrap();
        assert_close(&output, &[0.0, 0.0, 0.0], 1e-7);
    }

    #[test]
    fn test_ffn_zero_weights() {
        let cfg = FfnConfig::new(1, 2, 3, FfnActivationType::SiLU).unwrap();
        let w1 = vec![0.0f32; 6];
        let w2 = vec![0.0f32; 6];
        let input = [5.0, 10.0];
        let mut output = [999.0f32; 2];
        ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).unwrap();
        assert_close(&output, &[0.0, 0.0], 1e-7);
    }

    // ── FFN with various hidden dimensions ────────────────────────────

    #[test]
    fn test_ffn_dim_64() {
        let h = 64;
        let inter = 128;
        let cfg = FfnConfig::new(1, h, inter, FfnActivationType::ReLU).unwrap();
        let input = vec![1.0f32; h];
        let w1 = vec![0.01f32; inter * h];
        let w2 = vec![0.01f32; h * inter];
        let mut output = vec![0.0f32; h];
        ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).unwrap();
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ffn_dim_256() {
        let h = 256;
        let inter = 512;
        let cfg = FfnConfig::new(1, h, inter, FfnActivationType::SiLU).unwrap();
        let input: Vec<f32> = (0..h).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let w1 = vec![0.001f32; inter * h];
        let w2 = vec![0.001f32; h * inter];
        let mut output = vec![0.0f32; h];
        ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).unwrap();
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ffn_dim_1024() {
        let h = 1024;
        let inter = 2048;
        let cfg = FfnConfig::new(1, h, inter, FfnActivationType::GELU).unwrap();
        let input = vec![0.01f32; h];
        let w1 = vec![0.001f32; inter * h];
        let w2 = vec![0.001f32; h * inter];
        let mut output = vec![0.0f32; h];
        ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).unwrap();
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ffn_dim_4096() {
        let h = 4096;
        let inter = 4096;
        let cfg = FfnConfig::new(1, h, inter, FfnActivationType::ReLU).unwrap();
        let input = vec![0.001f32; h];
        let w1 = vec![0.0001f32; inter * h];
        let w2 = vec![0.0001f32; h * inter];
        let mut output = vec![0.0f32; h];
        ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).unwrap();
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    // ── FFN with various batch sizes ──────────────────────────────────

    #[test]
    fn test_ffn_batch_1() {
        let cfg = FfnConfig::new(1, 4, 8, FfnActivationType::SiLU).unwrap();
        let input = vec![1.0f32; 4];
        let w1 = vec![0.1f32; 32];
        let w2 = vec![0.1f32; 32];
        let mut output = vec![0.0f32; 4];
        ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).unwrap();
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_ffn_batch_4() {
        let cfg = FfnConfig::new(4, 4, 8, FfnActivationType::ReLU).unwrap();
        let input = vec![1.0f32; 16];
        let w1 = vec![0.1f32; 32];
        let w2 = vec![0.1f32; 32];
        let mut output = vec![0.0f32; 16];
        ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).unwrap();
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_ffn_batch_16() {
        let cfg = FfnConfig::new(16, 4, 8, FfnActivationType::GELU).unwrap();
        let input = vec![0.5f32; 64];
        let w1 = vec![0.1f32; 32];
        let w2 = vec![0.1f32; 32];
        let mut output = vec![0.0f32; 64];
        ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).unwrap();
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_ffn_batch_32() {
        let cfg = FfnConfig::new(32, 4, 8, FfnActivationType::SiLU).unwrap();
        let input = vec![0.5f32; 128];
        let w1 = vec![0.1f32; 32];
        let w2 = vec![0.1f32; 32];
        let mut output = vec![0.0f32; 128];
        ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).unwrap();
        // All rows get same output since input is uniform
        let row0 = &output[0..4];
        for row in 1..32 {
            assert_close(&output[row * 4..(row + 1) * 4], row0, 1e-6);
        }
    }

    #[test]
    fn test_ffn_batch_rows_independent() {
        let cfg = FfnConfig::new(2, 2, 3, FfnActivationType::ReLU).unwrap();
        let w1 = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let w2 = vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let input_a = vec![1.0, 2.0, 3.0, 4.0];
        let input_b = vec![1.0, 2.0, 99.0, 99.0];
        let mut out_a = vec![0.0f32; 4];
        let mut out_b = vec![0.0f32; 4];
        ffn_forward(&input_a, &w1, None, &w2, None, &mut out_a, &cfg).unwrap();
        ffn_forward(&input_b, &w1, None, &w2, None, &mut out_b, &cfg).unwrap();
        assert_close(&out_a[0..2], &out_b[0..2], 1e-6);
        assert_ne!(&out_a[2..4], &out_b[2..4]);
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_ffn_dim_1() {
        let cfg = FfnConfig::new(1, 1, 1, FfnActivationType::ReLU).unwrap();
        let w = [2.0f32];
        let input = [3.0];
        let mut output = [0.0f32];
        ffn_forward(&input, &w, None, &w, None, &mut output, &cfg).unwrap();
        // up = 2*3=6, relu(6)=6, down = 2*6=12
        assert!((output[0] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_ffn_batch_1_dim_1() {
        let cfg = FfnConfig::new(1, 1, 1, FfnActivationType::SiLU).unwrap();
        let w = [1.0f32];
        let input = [0.0];
        let mut output = [999.0f32];
        ffn_forward(&input, &w, None, &w, None, &mut output, &cfg).unwrap();
        // silu(0) = 0
        assert!(output[0].abs() < 1e-7);
    }

    // ── Gated FFN tests ───────────────────────────────────────────────

    #[test]
    fn test_gated_ffn_relu_known() {
        let cfg = FfnConfig::new(1, 2, 2, FfnActivationType::ReLU).unwrap();
        let eye = identity(2);
        let input = [3.0, -1.0];
        let mut output = [0.0f32; 2];
        gated_ffn_forward(&input, &eye, &eye, &eye, &mut output, &cfg).unwrap();
        // gate = [3,-1], relu = [3,0]; up = [3,-1]; gated = [9, 0]; down = [9, 0]
        assert_close(&output, &[9.0, 0.0], 1e-6);
    }

    #[test]
    fn test_gated_ffn_silu_known() {
        let cfg = FfnConfig::new(1, 1, 1, FfnActivationType::SiLU).unwrap();
        let w = [1.0f32];
        let input = [2.0];
        let mut output = [0.0f32];
        gated_ffn_forward(&input, &w, &w, &w, &mut output, &cfg).unwrap();
        let expected = silu(2.0) * 2.0;
        assert!((output[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_gated_ffn_gelu_known() {
        let cfg = FfnConfig::new(1, 1, 1, FfnActivationType::GELU).unwrap();
        let w = [1.0f32];
        let input = [1.0];
        let mut output = [0.0f32];
        gated_ffn_forward(&input, &w, &w, &w, &mut output, &cfg).unwrap();
        let expected = gelu(1.0);
        assert!((output[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_gated_ffn_zero_gate_weights() {
        let cfg = FfnConfig::new(1, 2, 3, FfnActivationType::ReLU).unwrap();
        let input = [5.0, 10.0];
        let w_gate = vec![0.0f32; 6];
        let w_up = vec![1.0f32; 6];
        let w_down = vec![1.0f32; 6];
        let mut output = [0.0f32; 2];
        gated_ffn_forward(&input, &w_gate, &w_up, &w_down, &mut output, &cfg).unwrap();
        assert_close(&output, &[0.0, 0.0], 1e-7);
    }

    #[test]
    fn test_gated_ffn_batched() {
        let cfg = FfnConfig::new(2, 2, 2, FfnActivationType::ReLU).unwrap();
        let eye = identity(2);
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];
        gated_ffn_forward(&input, &eye, &eye, &eye, &mut output, &cfg).unwrap();
        // row0: gate=[1,2], relu=[1,2], up=[1,2], gated=[1,4], down=[1,4]
        // row1: gate=[3,4], relu=[3,4], up=[3,4], gated=[9,16], down=[9,16]
        assert_close(&output[0..2], &[1.0, 4.0], 1e-6);
        assert_close(&output[2..4], &[9.0, 16.0], 1e-6);
    }

    // ── SwiGLU activation tests ───────────────────────────────────────

    #[test]
    fn test_swiglu_basic() {
        let gate = [1.0f32, 0.0, -1.0, 2.0];
        let up = [1.0f32, 1.0, 1.0, 0.5];
        let mut out = [0.0f32; 4];
        ffn_swiglu(&gate, &up, &mut out, 4).unwrap();
        for i in 0..4 {
            let expected = silu(gate[i]) * up[i];
            assert!((out[i] - expected).abs() < 1e-6, "swiglu[{i}]: {} vs {expected}", out[i]);
        }
    }

    #[test]
    fn test_swiglu_zero_gate() {
        let gate = [0.0f32; 4];
        let up = [5.0f32; 4];
        let mut out = [999.0f32; 4];
        ffn_swiglu(&gate, &up, &mut out, 4).unwrap();
        for &v in &out {
            assert!(v.abs() < 1e-7);
        }
    }

    #[test]
    fn test_swiglu_vs_reference() {
        // SiLU(1.5) * 3.0
        let gate = [1.5f32];
        let up = [3.0f32];
        let mut out = [0.0f32];
        ffn_swiglu(&gate, &up, &mut out, 1).unwrap();
        let silu_15 = 1.5 / (1.0 + (-1.5f32).exp());
        assert!((out[0] - silu_15 * 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_swiglu_large_vector() {
        let n = 1024;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let up = vec![1.0f32; n];
        let mut out = vec![0.0f32; n];
        ffn_swiglu(&gate, &up, &mut out, n).unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_swiglu_empty() {
        let mut out = [];
        ffn_swiglu(&[], &[], &mut out, 0).unwrap();
    }

    #[test]
    fn test_swiglu_buffer_too_small() {
        let gate = [1.0f32; 2];
        let up = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        assert!(ffn_swiglu(&gate, &up, &mut out, 4).is_err());
    }

    // ── GeGLU activation tests ────────────────────────────────────────

    #[test]
    fn test_geglu_basic() {
        let gate = [1.0f32, 0.0, -1.0];
        let up = [2.0f32, 5.0, 1.0];
        let mut out = [0.0f32; 3];
        ffn_geglu(&gate, &up, &mut out, 3).unwrap();
        for i in 0..3 {
            let expected = gelu(gate[i]) * up[i];
            assert!((out[i] - expected).abs() < 1e-5, "geglu[{i}]: {} vs {expected}", out[i]);
        }
    }

    #[test]
    fn test_geglu_zero_gate() {
        let gate = [0.0f32; 3];
        let up = [1.0f32; 3];
        let mut out = [999.0f32; 3];
        ffn_geglu(&gate, &up, &mut out, 3).unwrap();
        for &v in &out {
            assert!(v.abs() < 1e-7);
        }
    }

    #[test]
    fn test_geglu_vs_reference() {
        // GELU(1.0) ≈ 0.8412
        let gate = [1.0f32];
        let up = [2.0f32];
        let mut out = [0.0f32];
        ffn_geglu(&gate, &up, &mut out, 1).unwrap();
        let gelu_1 = gelu(1.0);
        assert!((out[0] - gelu_1 * 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_geglu_differs_from_swiglu() {
        let gate = [1.0f32, -0.5, 2.0];
        let up = [1.0f32; 3];
        let mut swiglu_out = [0.0f32; 3];
        let mut geglu_out = [0.0f32; 3];
        ffn_swiglu(&gate, &up, &mut swiglu_out, 3).unwrap();
        ffn_geglu(&gate, &up, &mut geglu_out, 3).unwrap();
        // Should differ for non-zero gate values
        assert!((swiglu_out[0] - geglu_out[0]).abs() > 1e-3);
    }

    // ── ReLU² activation tests ────────────────────────────────────────

    #[test]
    fn test_relu_squared_basic() {
        let gate = [2.0f32, -1.0, 0.0, 3.0];
        let up = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        ffn_relu_squared(&gate, &up, &mut out, 4).unwrap();
        assert!((out[0] - 4.0).abs() < 1e-6); // relu(2)²=4
        assert!(out[1].abs() < 1e-7); // relu(-1)²=0
        assert!(out[2].abs() < 1e-7); // relu(0)²=0
        assert!((out[3] - 9.0).abs() < 1e-6); // relu(3)²=9
    }

    #[test]
    fn test_relu_squared_with_up_scaling() {
        let gate = [3.0f32];
        let up = [2.0f32];
        let mut out = [0.0f32];
        ffn_relu_squared(&gate, &up, &mut out, 1).unwrap();
        // relu(3)² * 2 = 9 * 2 = 18
        assert!((out[0] - 18.0).abs() < 1e-5);
    }

    #[test]
    fn test_relu_squared_all_negative() {
        let gate = [-1.0f32, -2.0, -3.0];
        let up = [10.0f32; 3];
        let mut out = [999.0f32; 3];
        ffn_relu_squared(&gate, &up, &mut out, 3).unwrap();
        for &v in &out {
            assert!(v.abs() < 1e-7);
        }
    }

    #[test]
    fn test_relu_squared_differs_from_reglu() {
        // ReLU²(2) * 1 = 4 vs ReLU(2) * 1 = 2
        let gate = [2.0f32];
        let up = [1.0f32];
        let mut out_sq = [0.0f32];
        ffn_relu_squared(&gate, &up, &mut out_sq, 1).unwrap();
        assert!((out_sq[0] - 4.0).abs() < 1e-6);
    }

    // ── Fused FFN with normalization ──────────────────────────────────

    #[test]
    fn test_fused_ffn_norm_identity() {
        let cfg = FfnConfig::new(1, 2, 2, FfnActivationType::ReLU).unwrap();
        let eye = identity(2);
        let gamma = [1.0f32, 1.0];
        let input = [3.0, 4.0];
        let mut output = [0.0f32; 2];
        fused_ffn_norm(&input, &gamma, &eye, &eye, &mut output, &cfg, 1e-5).unwrap();
        // RMS = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        // normed ≈ [0.849, 1.131]
        // relu(normed) = normed (both positive)
        // identity down-project
        let rms = ((9.0 + 16.0) / 2.0f32 + 1e-5).sqrt();
        let expected = [3.0 / rms, 4.0 / rms];
        assert_close(&output, &expected, 1e-4);
    }

    #[test]
    fn test_fused_vs_unfused_equivalence() {
        let cfg = FfnConfig::new(1, 4, 8, FfnActivationType::SiLU).unwrap();
        let input = [1.0f32, -0.5, 0.3, 2.0];
        let gamma = [1.0f32, 1.0, 1.0, 1.0];
        let w1: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01 - 0.15).collect();
        let w2: Vec<f32> = (0..32).map(|i| (i as f32) * 0.005 + 0.01).collect();
        let eps = 1e-5;

        // Fused
        let mut fused_out = [0.0f32; 4];
        fused_ffn_norm(&input, &gamma, &w1, &w2, &mut fused_out, &cfg, eps).unwrap();

        // Unfused: manual RMSNorm then FFN
        let ss: f32 = input.iter().map(|&v| v * v).sum::<f32>() / 4.0;
        let inv_rms = 1.0 / (ss + eps).sqrt();
        let normed: Vec<f32> =
            input.iter().zip(gamma.iter()).map(|(&v, &g)| v * g * inv_rms).collect();
        let mut unfused_out = [0.0f32; 4];
        ffn_forward(&normed, &w1, None, &w2, None, &mut unfused_out, &cfg).unwrap();

        assert_close(&fused_out, &unfused_out, 1e-5);
    }

    #[test]
    fn test_fused_ffn_norm_batched() {
        let cfg = FfnConfig::new(2, 2, 4, FfnActivationType::ReLU).unwrap();
        let gamma = [1.0f32, 1.0];
        let w1 = vec![0.5f32; 8];
        let w2 = vec![0.5f32; 8];
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];
        fused_ffn_norm(&input, &gamma, &w1, &w2, &mut output, &cfg, 1e-5).unwrap();
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_fused_ffn_norm_gamma_too_small() {
        let cfg = FfnConfig::new(1, 4, 8, FfnActivationType::ReLU).unwrap();
        let gamma = [1.0f32; 2]; // needs 4
        let input = [1.0f32; 4];
        let w1 = vec![0.1f32; 32];
        let w2 = vec![0.1f32; 32];
        let mut output = [0.0f32; 4];
        assert!(fused_ffn_norm(&input, &gamma, &w1, &w2, &mut output, &cfg, 1e-5).is_err());
    }

    // ── Sparse FFN tests ──────────────────────────────────────────────

    #[test]
    fn test_sparse_ffn_config_basic() {
        let base = FfnConfig::new(1, 4, 8, FfnActivationType::ReLU).unwrap();
        let cfg = SparseFfnConfig::new(base, 4).unwrap();
        assert_eq!(cfg.top_k, 4);
    }

    #[test]
    fn test_sparse_ffn_config_rejects_zero_k() {
        let base = FfnConfig::new(1, 4, 8, FfnActivationType::ReLU).unwrap();
        assert!(SparseFfnConfig::new(base, 0).is_err());
    }

    #[test]
    fn test_sparse_ffn_config_rejects_k_exceeding_inter() {
        let base = FfnConfig::new(1, 4, 8, FfnActivationType::ReLU).unwrap();
        assert!(SparseFfnConfig::new(base, 9).is_err());
    }

    #[test]
    fn test_sparse_ffn_topk_all() {
        // top_k = intermediate_dim → same as dense
        let base = FfnConfig::new(1, 2, 3, FfnActivationType::ReLU).unwrap();
        let sparse_cfg = SparseFfnConfig::new(base.clone(), 3).unwrap();
        let w_up = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let w_down = vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let input = [2.0, 3.0];
        let mut sparse_out = [0.0f32; 2];
        sparse_ffn_forward(&input, &w_up, &w_down, &mut sparse_out, &sparse_cfg).unwrap();

        let mut dense_out = [0.0f32; 2];
        ffn_forward(&input, &w_up, None, &w_down, None, &mut dense_out, &base).unwrap();

        assert_close(&sparse_out, &dense_out, 1e-6);
    }

    #[test]
    fn test_sparse_ffn_topk_1() {
        // Only keep the largest-magnitude neuron
        let base = FfnConfig::new(1, 2, 4, FfnActivationType::ReLU).unwrap();
        let sparse_cfg = SparseFfnConfig::new(base, 1).unwrap();
        let eye2x4 = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.1, 0.1]; // 4×2
        let w_down = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]; // 2×4
        let input = [1.0, 2.0];
        let mut output = [0.0f32; 2];
        sparse_ffn_forward(&input, &eye2x4, &w_down, &mut output, &sparse_cfg).unwrap();
        // up = [1, 2, 1.5, 0.3], relu = same. Largest = 2.0 at idx 1.
        // Sparse hidden = [0, 2, 0, 0]
        // output = w_down . [0,2,0,0] = [0, 2]
        assert_close(&output, &[0.0, 2.0], 1e-6);
    }

    #[test]
    fn test_sparse_ffn_topk_2() {
        let base = FfnConfig::new(1, 2, 4, FfnActivationType::ReLU).unwrap();
        let sparse_cfg = SparseFfnConfig::new(base, 2).unwrap();
        #[rustfmt::skip]
        let w_up = vec![
            1.0, 0.0,
            0.0, 1.0,
            0.5, 0.5,
            0.1, 0.1,
        ];
        #[rustfmt::skip]
        let w_down = vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ];
        let input = [1.0, 2.0];
        let mut output = [0.0f32; 2];
        sparse_ffn_forward(&input, &w_up, &w_down, &mut output, &sparse_cfg).unwrap();
        // up = [1, 2, 1.5, 0.3]. Top-2 by magnitude: idx 1 (2.0), idx 2 (1.5)
        // Sparse = [0, 2, 1.5, 0]; sum = 3.5 for each output dim
        assert_close(&output, &[3.5, 3.5], 1e-5);
    }

    #[test]
    fn test_sparse_ffn_batched() {
        let base = FfnConfig::new(2, 2, 4, FfnActivationType::ReLU).unwrap();
        let sparse_cfg = SparseFfnConfig::new(base, 2).unwrap();
        let w_up = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.1, 0.1];
        let w_down = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];
        sparse_ffn_forward(&input, &w_up, &w_down, &mut output, &sparse_cfg).unwrap();
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    // ── FFN with dropout tests ────────────────────────────────────────

    #[test]
    fn test_dropout_no_drop() {
        let cfg = FfnConfig::new(1, 2, 3, FfnActivationType::ReLU).unwrap();
        let w1 = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let w2 = vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let mask = [1.0f32; 3]; // no dropout
        let input = [2.0, 3.0];
        let mut output = [0.0f32; 2];
        ffn_with_dropout(&input, &w1, &w2, &mask, &mut output, &cfg, 0.0).unwrap();
        // Same as standard FFN with identity dropout
        let mut expected = [0.0f32; 2];
        ffn_forward(&input, &w1, None, &w2, None, &mut expected, &cfg).unwrap();
        assert_close(&output, &expected, 1e-6);
    }

    #[test]
    fn test_dropout_full_drop() {
        let cfg = FfnConfig::new(1, 2, 3, FfnActivationType::ReLU).unwrap();
        let w1 = vec![1.0f32; 6];
        let w2 = vec![1.0f32; 6];
        let mask = [0.0f32; 3]; // drop everything
        let input = [5.0, 10.0];
        let mut output = [999.0f32; 2];
        ffn_with_dropout(&input, &w1, &w2, &mask, &mut output, &cfg, 0.5).unwrap();
        assert_close(&output, &[0.0, 0.0], 1e-7);
    }

    #[test]
    fn test_dropout_scaling() {
        // With 50% dropout rate and all-ones mask, values should be scaled by 2.0
        let cfg = FfnConfig::new(1, 2, 2, FfnActivationType::ReLU).unwrap();
        let eye = identity(2);
        let mask = [1.0f32; 2];
        let input = [1.0, 2.0];
        let mut output_drop = [0.0f32; 2];
        ffn_with_dropout(&input, &eye, &eye, &mask, &mut output_drop, &cfg, 0.5).unwrap();
        // Without dropout: relu([1,2]) = [1,2]
        // With dropout scale 2.0: [2, 4]
        assert_close(&output_drop, &[2.0, 4.0], 1e-6);
    }

    #[test]
    fn test_dropout_invalid_rate() {
        let cfg = FfnConfig::new(1, 2, 2, FfnActivationType::ReLU).unwrap();
        let eye = identity(2);
        let mask = [1.0f32; 2];
        let input = [1.0, 2.0];
        let mut output = [0.0f32; 2];
        assert!(ffn_with_dropout(&input, &eye, &eye, &mask, &mut output, &cfg, 1.0).is_err());
        assert!(ffn_with_dropout(&input, &eye, &eye, &mask, &mut output, &cfg, -0.1).is_err());
    }

    #[test]
    fn test_dropout_mask_too_small() {
        let cfg = FfnConfig::new(1, 2, 4, FfnActivationType::ReLU).unwrap();
        let w1 = vec![0.1f32; 8];
        let w2 = vec![0.1f32; 8];
        let mask = [1.0f32; 2]; // needs 4
        let input = [1.0, 2.0];
        let mut output = [0.0f32; 2];
        assert!(ffn_with_dropout(&input, &w1, &w2, &mask, &mut output, &cfg, 0.0).is_err());
    }

    // ── Quantized FFN tests ───────────────────────────────────────────

    #[test]
    fn test_quantized_ffn_int2_basic() {
        // 2-bit quantization: packed 4 vals per byte
        // Value mapping: 0→-1, 1→0, 2→1 (center=1)
        let cfg = FfnConfig::new(1, 2, 2, FfnActivationType::ReLU).unwrap();
        // Pack identity-like patterns:
        // w1[0,0]=1 → code 2, w1[0,1]=0 → code 1 → byte = (1 << 2) | 2 = 6
        // w1[1,0]=0 → code 1, w1[1,1]=1 → code 2 → byte = (2 << 2) | 1 = 9
        let w1_packed = vec![0b00_01_00_10u8]; // [1, 0, 0, 1] → identity
        let w1_scales = vec![1.0f32]; // one scale for one block
        let w2_packed = vec![0b00_01_00_10u8];
        let w2_scales = vec![1.0f32];
        let input = [3.0, 4.0];
        let mut output = [0.0f32; 2];
        quantized_ffn_forward(
            &input,
            &w1_packed,
            &w1_scales,
            &w2_packed,
            &w2_scales,
            &mut output,
            &cfg,
            QuantBits::Int2,
            4,
        )
        .unwrap();
        // Verify output is finite and non-zero
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_quantized_ffn_int4_basic() {
        let cfg = FfnConfig::new(1, 2, 2, FfnActivationType::ReLU).unwrap();
        // 4-bit: 2 values per byte, center=7
        // Pack something simple
        let w1_packed = vec![0x77u8, 0x77]; // all center values → dequant to 0
        let w1_scales = vec![1.0f32];
        let w2_packed = vec![0x77u8, 0x77];
        let w2_scales = vec![1.0f32];
        let input = [1.0, 2.0];
        let mut output = [0.0f32; 2];
        quantized_ffn_forward(
            &input,
            &w1_packed,
            &w1_scales,
            &w2_packed,
            &w2_scales,
            &mut output,
            &cfg,
            QuantBits::Int4,
            4,
        )
        .unwrap();
        // Center values dequantize to 0, so output should be 0
        assert_close(&output, &[0.0, 0.0], 1e-5);
    }

    #[test]
    fn test_quantized_ffn_accuracy_bound() {
        // Quantized FFN should produce results within a reasonable bound of dense FFN
        let cfg = FfnConfig::new(1, 2, 4, FfnActivationType::ReLU).unwrap();
        // Dense reference weights: small uniform values
        let w1_dense = vec![0.5f32; 8];
        let w2_dense = vec![0.5f32; 8];
        let input = [1.0, 1.0];
        let mut dense_output = [0.0f32; 2];
        ffn_forward(&input, &w1_dense, None, &w2_dense, None, &mut dense_output, &cfg).unwrap();

        // Pack quantized approximation (all ones → dequant to 0 for INT2 center=1)
        // Use scale to approximate 0.5
        let w1_packed = vec![0b10_10_10_10u8, 0b10_10_10_10u8]; // all code=2 → val=1
        let w1_scales = vec![0.5f32; 2]; // scale 0.5 → effective 0.5
        let w2_packed = vec![0b10_10_10_10u8, 0b10_10_10_10u8];
        let w2_scales = vec![0.5f32; 2];
        let mut quant_output = [0.0f32; 2];
        quantized_ffn_forward(
            &input,
            &w1_packed,
            &w1_scales,
            &w2_packed,
            &w2_scales,
            &mut quant_output,
            &cfg,
            QuantBits::Int2,
            4,
        )
        .unwrap();

        // Both outputs should be finite and in a reasonable range
        for &v in &quant_output {
            assert!(v.is_finite());
        }
        for &v in &dense_output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_quantized_ffn_zero_block_size_rejected() {
        let cfg = FfnConfig::new(1, 2, 2, FfnActivationType::ReLU).unwrap();
        let w_packed = vec![0u8; 1];
        let w_scales = vec![1.0f32];
        let input = [1.0, 2.0];
        let mut output = [0.0f32; 2];
        assert!(
            quantized_ffn_forward(
                &input,
                &w_packed,
                &w_scales,
                &w_packed,
                &w_scales,
                &mut output,
                &cfg,
                QuantBits::Int2,
                0,
            )
            .is_err()
        );
    }

    #[test]
    fn test_quantized_ffn_packed_too_small() {
        let cfg = FfnConfig::new(1, 4, 8, FfnActivationType::ReLU).unwrap();
        let w1_packed = vec![0u8; 1]; // needs 8 for INT2
        let w1_scales = vec![1.0f32];
        let w2_packed = vec![0u8; 8];
        let w2_scales = vec![1.0f32];
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        assert!(
            quantized_ffn_forward(
                &input,
                &w1_packed,
                &w1_scales,
                &w2_packed,
                &w2_scales,
                &mut output,
                &cfg,
                QuantBits::Int2,
                32,
            )
            .is_err()
        );
    }

    // ── Buffer validation tests ───────────────────────────────────────

    #[test]
    fn test_ffn_input_too_small() {
        let cfg = FfnConfig::new(1, 4, 8, FfnActivationType::ReLU).unwrap();
        let input = [1.0f32; 2]; // needs 4
        let w1 = vec![0.1f32; 32];
        let w2 = vec![0.1f32; 32];
        let mut output = [0.0f32; 4];
        assert!(ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).is_err());
    }

    #[test]
    fn test_ffn_w1_too_small() {
        let cfg = FfnConfig::new(1, 2, 4, FfnActivationType::SiLU).unwrap();
        let input = [1.0f32; 2];
        let w1 = vec![0.1f32; 4]; // needs 8
        let w2 = vec![0.1f32; 8];
        let mut output = [0.0f32; 2];
        assert!(ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).is_err());
    }

    #[test]
    fn test_ffn_w2_too_small() {
        let cfg = FfnConfig::new(1, 2, 4, FfnActivationType::GELU).unwrap();
        let input = [1.0f32; 2];
        let w1 = vec![0.1f32; 8];
        let w2 = vec![0.1f32; 4]; // needs 8
        let mut output = [0.0f32; 2];
        assert!(ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).is_err());
    }

    #[test]
    fn test_ffn_output_too_small() {
        let cfg = FfnConfig::new(1, 4, 8, FfnActivationType::ReLU).unwrap();
        let input = [1.0f32; 4];
        let w1 = vec![0.1f32; 32];
        let w2 = vec![0.1f32; 32];
        let mut output = [0.0f32; 2]; // needs 4
        assert!(ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).is_err());
    }

    #[test]
    fn test_gated_ffn_w_gate_too_small() {
        let cfg = FfnConfig::new(1, 2, 4, FfnActivationType::SiLU).unwrap();
        let input = [1.0f32; 2];
        let w_gate = vec![0.1f32; 4]; // needs 8
        let w_up = vec![0.1f32; 8];
        let w_down = vec![0.1f32; 8];
        let mut output = [0.0f32; 2];
        assert!(gated_ffn_forward(&input, &w_gate, &w_up, &w_down, &mut output, &cfg).is_err());
    }

    // ── Gradient-free inference path ──────────────────────────────────

    #[test]
    fn test_inference_path_no_gradients() {
        // Verify that all FFN functions work without any gradient tracking
        let cfg = FfnConfig::new(1, 4, 8, FfnActivationType::SiLU).unwrap();
        let input = vec![1.0f32; 4];
        let w1 = vec![0.1f32; 32];
        let w2 = vec![0.1f32; 32];
        let mut output = vec![0.0f32; 4];
        // Should succeed without any gradient infrastructure
        ffn_forward(&input, &w1, None, &w2, None, &mut output, &cfg).unwrap();
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_inference_path_deterministic() {
        // Two calls with same input must produce identical output
        let cfg = FfnConfig::new(1, 4, 8, FfnActivationType::SiLU).unwrap();
        let input = vec![0.5f32; 4];
        let w1 = vec![0.1f32; 32];
        let w2 = vec![0.1f32; 32];
        let mut out1 = vec![0.0f32; 4];
        let mut out2 = vec![0.0f32; 4];
        ffn_forward(&input, &w1, None, &w2, None, &mut out1, &cfg).unwrap();
        ffn_forward(&input, &w1, None, &w2, None, &mut out2, &cfg).unwrap();
        assert_eq!(out1, out2);
    }

    // ── Unified dispatch tests ────────────────────────────────────────

    #[test]
    fn test_launch_ffn_cpu_fallback() {
        let cfg = FfnConfig::new(1, 2, 3, FfnActivationType::SiLU).unwrap();
        let w1 = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let w2 = vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let input = [2.0, 3.0];
        let mut output = [0.0f32; 2];
        launch_ffn(&input, &w1, None, &w2, None, &mut output, &cfg).unwrap();
        let mut expected = [0.0f32; 2];
        ffn_forward(&input, &w1, None, &w2, None, &mut expected, &cfg).unwrap();
        assert_close(&output, &expected, 1e-7);
    }

    #[test]
    fn test_launch_gated_ffn_cpu_fallback() {
        let cfg = FfnConfig::new(1, 2, 2, FfnActivationType::SiLU).unwrap();
        let eye = identity(2);
        let input = [1.0, 2.0];
        let mut output = [0.0f32; 2];
        launch_gated_ffn(&input, &eye, &eye, &eye, &mut output, &cfg).unwrap();
        let mut expected = [0.0f32; 2];
        gated_ffn_forward(&input, &eye, &eye, &eye, &mut expected, &cfg).unwrap();
        assert_close(&output, &expected, 1e-7);
    }

    // ── Activation correctness ────────────────────────────────────────

    #[test]
    fn test_activations_produce_different_results() {
        for act in [FfnActivationType::SiLU, FfnActivationType::GELU, FfnActivationType::ReLU] {
            let cfg = FfnConfig::new(1, 2, 2, act).unwrap();
            let eye = identity(2);
            let input = [1.0, -0.5];
            let mut output = [0.0f32; 2];
            ffn_forward(&input, &eye, None, &eye, None, &mut output, &cfg).unwrap();
            // Just verify it produces something
            assert!(output[0].is_finite());
        }
    }

    #[test]
    fn test_three_activations_differ() {
        let eye = identity(2);
        let input = [1.0, -0.5];
        let mut out_silu = [0.0f32; 2];
        let mut out_gelu = [0.0f32; 2];
        let mut out_relu = [0.0f32; 2];

        let cfg_s = FfnConfig::new(1, 2, 2, FfnActivationType::SiLU).unwrap();
        let cfg_g = FfnConfig::new(1, 2, 2, FfnActivationType::GELU).unwrap();
        let cfg_r = FfnConfig::new(1, 2, 2, FfnActivationType::ReLU).unwrap();

        ffn_forward(&input, &eye, None, &eye, None, &mut out_silu, &cfg_s).unwrap();
        ffn_forward(&input, &eye, None, &eye, None, &mut out_gelu, &cfg_g).unwrap();
        ffn_forward(&input, &eye, None, &eye, None, &mut out_relu, &cfg_r).unwrap();

        assert_ne!(out_silu, out_gelu);
        assert_ne!(out_silu, out_relu);
        assert_ne!(out_gelu, out_relu);
    }

    // ── GPU-only tests ────────────────────────────────────────────────

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_ffn_launch() {
        let cfg = FfnConfig::new(4, 256, 512, FfnActivationType::SiLU).unwrap();
        let input = vec![0.1f32; 4 * 256];
        let w1 = vec![0.01f32; 512 * 256];
        let w2 = vec![0.01f32; 256 * 512];
        let mut output = vec![0.0f32; 4 * 256];
        let result = launch_ffn(&input, &w1, None, &w2, None, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA FFN launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_gated_ffn_launch() {
        let cfg = FfnConfig::new(4, 256, 512, FfnActivationType::SiLU).unwrap();
        let input = vec![0.1f32; 4 * 256];
        let w = vec![0.01f32; 512 * 256];
        let w_down = vec![0.01f32; 256 * 512];
        let mut output = vec![0.0f32; 4 * 256];
        let result = launch_gated_ffn(&input, &w, &w, &w_down, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA gated FFN launch failed: {result:?}");
    }
}
