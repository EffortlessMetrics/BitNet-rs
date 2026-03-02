//! OpenCL-accelerated SwiGLU (Swish-Gated Linear Unit) feed-forward network.
//!
//! # Architecture
//!
//! SwiGLU is used in LLaMA-style transformer models:
//!
//! ```text
//! output = W_down × (activation(W_gate × x) ⊙ W_up × x)
//! ```
//!
//! The gating mechanism multiplies the activated gate projection element-wise
//! with the up projection before the down projection, improving training
//! dynamics compared to standard FFN.
//!
//! # Components
//!
//! - [`SwiGluConfig`] — layer configuration (dimensions, bias, activation).
//! - [`ActivationType`] — SiLU, GELU, ReLU, Swish(β) activation selector.
//! - [`GateProjection`] — computes gate and up projections from input.
//! - [`GateActivation`] — applies activation with optional vectorized paths.
//! - [`SwiGluLayer`] — full unfused SwiGLU forward pass.
//! - [`FusedSwiGlu`] — fused gate+up projection in a single pass.
//! - [`SwiGluStats`] — FLOPs, activation range, and sparsity statistics.
//! - [`MoERouter`] — Mixture-of-Experts top-k gating router.
//!
//! # OpenCL kernels
//!
//! [`SWIGLU_CL`] contains fused OpenCL C source for the SwiGLU FFN, and
//! [`ELEMENTWISE_CL`] provides element-wise activation kernels.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Activation types
// ---------------------------------------------------------------------------

/// Activation function used in the SwiGLU gate path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    /// Sigmoid Linear Unit: `x × σ(x)`.
    SiLU,
    /// Gaussian Error Linear Unit (exact).
    GELU,
    /// Rectified Linear Unit: `max(0, x)`.
    ReLU,
    /// Swish with configurable β: `x × σ(β·x)`.
    Swish(f32),
}

impl ActivationType {
    /// Apply the activation function to a scalar value.
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::SiLU => x * sigmoid(x),
            Self::GELU => {
                x * 0.5 * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2))
            }
            Self::ReLU => x.max(0.0),
            Self::Swish(beta) => x * sigmoid(beta * x),
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
            + t * (-0.284_496_74
                + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a SwiGLU feed-forward layer.
#[derive(Debug, Clone)]
pub struct SwiGluConfig {
    /// Model hidden dimension (input/output size).
    pub hidden_dim: usize,
    /// Intermediate (expanded) dimension.
    pub intermediate_dim: usize,
    /// Whether bias terms are added after projections.
    pub use_bias: bool,
    /// Activation function applied to the gate projection.
    pub activation_type: ActivationType,
}

impl SwiGluConfig {
    /// Create a new SwiGLU configuration.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if any dimension is zero.
    pub fn new(
        hidden_dim: usize,
        intermediate_dim: usize,
        use_bias: bool,
        activation_type: ActivationType,
    ) -> Result<Self> {
        if hidden_dim == 0 || intermediate_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "SwiGLU dimensions must be non-zero: \
                     hidden_dim={hidden_dim}, intermediate_dim={intermediate_dim}"
                ),
            }
            .into());
        }
        Ok(Self { hidden_dim, intermediate_dim, use_bias, activation_type })
    }

    /// LLaMA-2 7B compatible configuration.
    pub fn llama_7b() -> Self {
        Self {
            hidden_dim: 4096,
            intermediate_dim: 11008,
            use_bias: false,
            activation_type: ActivationType::SiLU,
        }
    }

    /// BitNet 2B compatible configuration.
    pub fn bitnet_2b() -> Self {
        Self {
            hidden_dim: 2048,
            intermediate_dim: 5632,
            use_bias: false,
            activation_type: ActivationType::SiLU,
        }
    }
}

// ---------------------------------------------------------------------------
// GateProjection
// ---------------------------------------------------------------------------

/// Result of computing gate and up projections: gate = W_gate × x, up = W_up × x.
#[derive(Debug, Clone)]
pub struct GateProjection {
    /// Projected gate values `[seq_len × intermediate_dim]`.
    pub gate: Vec<f32>,
    /// Projected up values `[seq_len × intermediate_dim]`.
    pub up: Vec<f32>,
    /// Sequence length used in this projection.
    pub seq_len: usize,
    /// Intermediate dimension.
    pub intermediate_dim: usize,
}

impl GateProjection {
    /// Compute gate and up projections from input.
    ///
    /// # Layout
    ///
    /// - `x`:      `[seq_len, hidden_dim]`
    /// - `w_gate`: `[hidden_dim, intermediate_dim]`
    /// - `w_up`:   `[hidden_dim, intermediate_dim]`
    ///
    /// # Errors
    ///
    /// Returns an error on dimension mismatch.
    pub fn compute(
        x: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<Self> {
        validate_projection_args(x, w_gate, w_up, seq_len, hidden_dim, intermediate_dim)?;

        let inter_len = seq_len * intermediate_dim;
        let mut gate = vec![0.0_f32; inter_len];
        let mut up = vec![0.0_f32; inter_len];

        matmul_ref(x, w_gate, &mut gate, seq_len, hidden_dim, intermediate_dim);
        matmul_ref(x, w_up, &mut up, seq_len, hidden_dim, intermediate_dim);

        Ok(Self { gate, up, seq_len, intermediate_dim })
    }

    /// Compute gate and up projections with bias.
    ///
    /// - `b_gate`: `[intermediate_dim]`
    /// - `b_up`:   `[intermediate_dim]`
    #[allow(clippy::too_many_arguments)]
    pub fn compute_with_bias(
        x: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        b_gate: &[f32],
        b_up: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<Self> {
        let mut proj = Self::compute(x, w_gate, w_up, seq_len, hidden_dim, intermediate_dim)?;

        if b_gate.len() < intermediate_dim || b_up.len() < intermediate_dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "Bias vectors too short: b_gate={}, b_up={}, expected {intermediate_dim}",
                    b_gate.len(),
                    b_up.len(),
                ),
            }
            .into());
        }
        for s in 0..seq_len {
            for j in 0..intermediate_dim {
                let idx = s * intermediate_dim + j;
                proj.gate[idx] += b_gate[j];
                proj.up[idx] += b_up[j];
            }
        }
        Ok(proj)
    }
}

// ---------------------------------------------------------------------------
// GateActivation
// ---------------------------------------------------------------------------

/// Applies activation to gate values with optional vectorized fast-paths.
pub struct GateActivation;

impl GateActivation {
    /// Apply activation element-wise: `out[i] = activation(gate[i]) × up[i]`.
    pub fn apply(
        gate: &[f32],
        up: &[f32],
        output: &mut [f32],
        activation: ActivationType,
    ) -> Result<()> {
        let n = gate.len();
        if up.len() < n || output.len() < n {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "GateActivation size mismatch: gate={n}, up={}, out={}",
                    up.len(),
                    output.len()
                ),
            }
            .into());
        }

        // Vectorized path: process 4 elements at a time
        let chunks = n / 4;
        for c in 0..chunks {
            let base = c * 4;
            output[base] = activation.apply(gate[base]) * up[base];
            output[base + 1] = activation.apply(gate[base + 1]) * up[base + 1];
            output[base + 2] = activation.apply(gate[base + 2]) * up[base + 2];
            output[base + 3] = activation.apply(gate[base + 3]) * up[base + 3];
        }
        // Scalar tail
        for i in (chunks * 4)..n {
            output[i] = activation.apply(gate[i]) * up[i];
        }
        Ok(())
    }

    /// Apply activation in-place on gate, then element-wise multiply with up.
    pub fn apply_inplace(gate: &mut [f32], up: &[f32], activation: ActivationType) -> Result<()> {
        let n = gate.len();
        if up.len() < n {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "GateActivation inplace size mismatch: gate={n}, up={}",
                    up.len()
                ),
            }
            .into());
        }
        for i in 0..n {
            gate[i] = activation.apply(gate[i]) * up[i];
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SwiGluLayer — unfused forward pass
// ---------------------------------------------------------------------------

/// Full SwiGLU layer: `output = W_down × (activation(W_gate × x) ⊙ W_up × x)`.
pub struct SwiGluLayer;

impl SwiGluLayer {
    /// SwiGLU forward pass (unfused, step-by-step).
    ///
    /// # Layout
    ///
    /// - `x`:      `[seq_len, hidden_dim]`
    /// - `w_gate`: `[hidden_dim, intermediate_dim]`
    /// - `w_up`:   `[hidden_dim, intermediate_dim]`
    /// - `w_down`: `[intermediate_dim, hidden_dim]`
    /// - `output`: `[seq_len, hidden_dim]`
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        x: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        output: &mut [f32],
        config: &SwiGluConfig,
        seq_len: usize,
    ) -> Result<()> {
        let h = config.hidden_dim;
        let inter = config.intermediate_dim;

        validate_swiglu_args(x, w_gate, w_up, w_down, output, seq_len, h, inter)?;

        // Step 1: Gate and up projections
        let proj = GateProjection::compute(x, w_gate, w_up, seq_len, h, inter)?;

        // Step 2: activation(gate) ⊙ up
        let inter_len = seq_len * inter;
        let mut hidden = vec![0.0_f32; inter_len];
        GateActivation::apply(&proj.gate, &proj.up, &mut hidden, config.activation_type)?;

        // Step 3: Down projection
        matmul_ref(&hidden, w_down, output, seq_len, inter, h);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FusedSwiGlu — single-pass fused implementation
// ---------------------------------------------------------------------------

/// Fused SwiGLU that computes gate+up projection, activation, and
/// element-wise multiply in a single pass per sequence position.
pub struct FusedSwiGlu;

impl FusedSwiGlu {
    /// Fused SwiGLU forward pass.
    ///
    /// Combines gate and up projection into a single loop over `hidden_dim`,
    /// then applies activation and element-wise multiply before the down
    /// projection. This reduces memory traffic compared to the unfused path.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        x: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        output: &mut [f32],
        config: &SwiGluConfig,
        seq_len: usize,
    ) -> Result<()> {
        let h = config.hidden_dim;
        let inter = config.intermediate_dim;

        validate_swiglu_args(x, w_gate, w_up, w_down, output, seq_len, h, inter)?;

        let mut hidden = vec![0.0_f32; seq_len * inter];

        // Fused gate+up projection with activation and element-wise multiply
        for s in 0..seq_len {
            for j in 0..inter {
                let mut gate_val = 0.0_f32;
                let mut up_val = 0.0_f32;
                for k in 0..h {
                    let xk = x[s * h + k];
                    gate_val += xk * w_gate[k * inter + j];
                    up_val += xk * w_up[k * inter + j];
                }
                hidden[s * inter + j] =
                    config.activation_type.apply(gate_val) * up_val;
            }
        }

        // Down projection
        matmul_ref(&hidden, w_down, output, seq_len, inter, h);

        Ok(())
    }

    /// Fused forward with bias terms.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_bias(
        x: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        b_gate: &[f32],
        b_up: &[f32],
        b_down: &[f32],
        output: &mut [f32],
        config: &SwiGluConfig,
        seq_len: usize,
    ) -> Result<()> {
        let h = config.hidden_dim;
        let inter = config.intermediate_dim;

        validate_swiglu_args(x, w_gate, w_up, w_down, output, seq_len, h, inter)?;

        if b_gate.len() < inter || b_up.len() < inter || b_down.len() < h {
            return Err(KernelError::InvalidArguments {
                reason: "Bias vector length mismatch".into(),
            }
            .into());
        }

        let mut hidden = vec![0.0_f32; seq_len * inter];

        for s in 0..seq_len {
            for j in 0..inter {
                let mut gate_val = b_gate[j];
                let mut up_val = b_up[j];
                for k in 0..h {
                    let xk = x[s * h + k];
                    gate_val += xk * w_gate[k * inter + j];
                    up_val += xk * w_up[k * inter + j];
                }
                hidden[s * inter + j] =
                    config.activation_type.apply(gate_val) * up_val;
            }
        }

        // Down projection + bias
        matmul_ref(&hidden, w_down, output, seq_len, inter, h);
        for s in 0..seq_len {
            for j in 0..h {
                output[s * h + j] += b_down[j];
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SwiGluStats
// ---------------------------------------------------------------------------

/// Performance and activation statistics for a SwiGLU forward pass.
#[derive(Debug, Clone)]
pub struct SwiGluStats {
    /// Total floating-point operations (gate + up + activation + elementwise + down).
    pub flops: u64,
    /// Range of activated values `(min, max)`.
    pub activation_range: (f32, f32),
    /// Fraction of activated gate values that are near-zero (|v| < threshold).
    pub sparsity_ratio: f32,
}

impl SwiGluStats {
    /// Compute statistics for a SwiGLU forward pass.
    ///
    /// `hidden_values` are the post-activation, pre-down-projection values.
    pub fn compute(
        hidden_values: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        sparsity_threshold: f32,
    ) -> Self {
        // FLOPs: gate matmul + up matmul + activation + elementwise + down matmul
        // Each matmul [M,K]×[K,N] = 2*M*K*N FLOPs
        let gate_flops = 2 * seq_len as u64 * hidden_dim as u64 * intermediate_dim as u64;
        let up_flops = gate_flops;
        let activation_flops = seq_len as u64 * intermediate_dim as u64 * 5; // ~5 ops for SiLU
        let elementwise_flops = seq_len as u64 * intermediate_dim as u64;
        let down_flops = 2 * seq_len as u64 * intermediate_dim as u64 * hidden_dim as u64;
        let flops = gate_flops + up_flops + activation_flops + elementwise_flops + down_flops;

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut near_zero_count = 0u64;

        for &v in hidden_values {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
            if v.abs() < sparsity_threshold {
                near_zero_count += 1;
            }
        }

        let sparsity_ratio = if hidden_values.is_empty() {
            0.0
        } else {
            near_zero_count as f32 / hidden_values.len() as f32
        };

        Self {
            flops,
            activation_range: (min_val, max_val),
            sparsity_ratio,
        }
    }

    /// Compute expected FLOPs for given dimensions (without running forward pass).
    pub fn estimate_flops(
        seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> u64 {
        let s = seq_len as u64;
        let h = hidden_dim as u64;
        let i = intermediate_dim as u64;
        // gate + up + activation + elementwise + down
        2 * s * h * i + 2 * s * h * i + s * i * 5 + s * i + 2 * s * i * h
    }
}

// ---------------------------------------------------------------------------
// MoERouter
// ---------------------------------------------------------------------------

/// Mixture-of-Experts router that selects top-k experts per token.
///
/// Given router logits `[seq_len, num_experts]`, produces expert assignments
/// and gating weights for each token.
#[derive(Debug, Clone)]
pub struct MoERouter {
    /// Number of available experts.
    pub num_experts: usize,
    /// Number of experts to activate per token.
    pub top_k: usize,
}

/// Result of MoE routing for a single token.
#[derive(Debug, Clone)]
pub struct MoEAssignment {
    /// Indices of selected experts, length = top_k.
    pub expert_indices: Vec<usize>,
    /// Normalized gating weights, length = top_k, sum = 1.0.
    pub weights: Vec<f32>,
}

impl MoERouter {
    /// Create a new MoE router.
    ///
    /// # Errors
    ///
    /// Returns an error if `top_k` > `num_experts` or either is zero.
    pub fn new(num_experts: usize, top_k: usize) -> Result<Self> {
        if num_experts == 0 || top_k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "MoE: num_experts and top_k must be > 0: \
                     num_experts={num_experts}, top_k={top_k}"
                ),
            }
            .into());
        }
        if top_k > num_experts {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "MoE: top_k ({top_k}) must be <= num_experts ({num_experts})"
                ),
            }
            .into());
        }
        Ok(Self { num_experts, top_k })
    }

    /// Route a batch of tokens to experts.
    ///
    /// `logits`: `[seq_len, num_experts]` — raw router scores.
    ///
    /// Returns one [`MoEAssignment`] per token with top-k expert selections
    /// and softmax-normalized weights.
    pub fn route(&self, logits: &[f32], seq_len: usize) -> Result<Vec<MoEAssignment>> {
        if logits.len() < seq_len * self.num_experts {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "MoE logits length {} < expected {}",
                    logits.len(),
                    seq_len * self.num_experts,
                ),
            }
            .into());
        }

        let mut assignments = Vec::with_capacity(seq_len);
        for s in 0..seq_len {
            let row = &logits[s * self.num_experts..(s + 1) * self.num_experts];
            assignments.push(self.route_single(row));
        }
        Ok(assignments)
    }

    /// Route a single token.
    fn route_single(&self, logits: &[f32]) -> MoEAssignment {
        // Find top-k indices by score
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top = &indexed[..self.top_k];
        let expert_indices: Vec<usize> = top.iter().map(|&(i, _)| i).collect();

        // Softmax over selected logits
        let max_logit = top.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = top.iter().map(|&(_, v)| (v - max_logit).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let weights: Vec<f32> = if sum > 0.0 {
            exp_vals.iter().map(|&e| e / sum).collect()
        } else {
            vec![1.0 / self.top_k as f32; self.top_k]
        };

        MoEAssignment { expert_indices, weights }
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel sources
// ---------------------------------------------------------------------------

/// Fused SwiGLU OpenCL C kernel.
///
/// Performs gate+up projection, configurable activation, element-wise multiply,
/// and down projection. Temporary buffer `temp_hidden` must be pre-allocated
/// as `[seq_len × intermediate_dim]`.
pub const SWIGLU_CL: &str = r#"
// SiLU activation: x * sigmoid(x)
float silu(float x) {
    return x / (1.0f + exp(-x));
}

// Swish with beta: x * sigmoid(beta * x)
float swish_beta(float x, float beta) {
    return x / (1.0f + exp(-beta * x));
}

// GELU approximation using tanh
float gelu_approx(float x) {
    float c = 0.7978845608f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}

// Fused SwiGLU: W_down × (activation(W_gate × x) ⊙ W_up × x)
//
// Phase 1: compute gate and up projections, apply activation and multiply.
// Phase 2: down projection.
//
// Work-item layout: global_id(0) = column, global_id(1) = sequence row.
__kernel void fused_swiglu(
    __global const float* x,
    __global const float* w_gate,
    __global const float* w_up,
    __global const float* w_down,
    __global float* output,
    __global float* temp_hidden,
    const int seq_len,
    const int hidden_dim,
    const int intermediate_dim,
    const int activation_type)  // 0=SiLU, 1=GELU, 2=ReLU, 3=Swish
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= seq_len || col >= intermediate_dim) return;

    // Fused gate + up projection
    float gate_val = 0.0f;
    float up_val = 0.0f;
    for (int k = 0; k < hidden_dim; k++) {
        float xk = x[row * hidden_dim + k];
        gate_val += xk * w_gate[k * intermediate_dim + col];
        up_val   += xk * w_up[k * intermediate_dim + col];
    }

    // Apply activation to gate
    float activated;
    if (activation_type == 0) {
        activated = silu(gate_val);
    } else if (activation_type == 1) {
        activated = gelu_approx(gate_val);
    } else if (activation_type == 2) {
        activated = fmax(0.0f, gate_val);
    } else {
        activated = silu(gate_val);  // default to SiLU
    }

    temp_hidden[row * intermediate_dim + col] = activated * up_val;

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Down projection: each thread computes one output element
    if (col < hidden_dim) {
        float sum = 0.0f;
        for (int k = 0; k < intermediate_dim; k++) {
            sum += temp_hidden[row * intermediate_dim + k]
                 * w_down[k * hidden_dim + col];
        }
        output[row * hidden_dim + col] = sum;
    }
}
"#;

/// Element-wise OpenCL kernels for activation functions.
pub const ELEMENTWISE_CL: &str = r#"
// Element-wise SiLU: out[i] = x[i] * sigmoid(x[i])
__kernel void ewise_silu(
    __global const float* input,
    __global float* output,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float x = input[i];
    output[i] = x / (1.0f + exp(-x));
}

// Element-wise Swish(beta): out[i] = x[i] * sigmoid(beta * x[i])
__kernel void ewise_swish(
    __global const float* input,
    __global float* output,
    const int n,
    const float beta)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float x = input[i];
    output[i] = x / (1.0f + exp(-beta * x));
}

// Element-wise gated multiply: out[i] = act(gate[i]) * up[i]
__kernel void ewise_gate_mul(
    __global const float* gate,
    __global const float* up,
    __global float* output,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float g = gate[i];
    float activated = g / (1.0f + exp(-g));  // SiLU
    output[i] = activated * up[i];
}
"#;

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// CPU reference for the full SwiGLU forward pass.
///
/// `output = W_down × (activation(W_gate × x) ⊙ W_up × x)`
#[allow(clippy::too_many_arguments)]
pub fn swiglu_forward_ref(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    activation: ActivationType,
) -> Result<()> {
    validate_swiglu_args(x, w_gate, w_up, w_down, output, seq_len, hidden_dim, intermediate_dim)?;

    let inter_len = seq_len * intermediate_dim;
    let mut gate = vec![0.0_f32; inter_len];
    let mut up = vec![0.0_f32; inter_len];

    matmul_ref(x, w_gate, &mut gate, seq_len, hidden_dim, intermediate_dim);
    matmul_ref(x, w_up, &mut up, seq_len, hidden_dim, intermediate_dim);

    let mut hidden = vec![0.0_f32; inter_len];
    for i in 0..inter_len {
        hidden[i] = activation.apply(gate[i]) * up[i];
    }

    matmul_ref(&hidden, w_down, output, seq_len, intermediate_dim, hidden_dim);

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive row-major matmul: `C[M,N] = A[M,K] × B[K,N]`.
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

fn validate_projection_args(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Result<()> {
    if seq_len == 0 || hidden_dim == 0 || intermediate_dim == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "Projection dims must be non-zero: seq={seq_len}, h={hidden_dim}, inter={intermediate_dim}"
            ),
        }
        .into());
    }
    let x_expected = seq_len * hidden_dim;
    let w_expected = hidden_dim * intermediate_dim;
    if x.len() < x_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("x length {} < expected {x_expected}", x.len()),
        }
        .into());
    }
    if w_gate.len() < w_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("w_gate length {} < expected {w_expected}", w_gate.len()),
        }
        .into());
    }
    if w_up.len() < w_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("w_up length {} < expected {w_expected}", w_up.len()),
        }
        .into());
    }
    Ok(())
}

fn validate_swiglu_args(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    output: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Result<()> {
    validate_projection_args(x, w_gate, w_up, seq_len, hidden_dim, intermediate_dim)?;
    let down_expected = intermediate_dim * hidden_dim;
    let out_expected = seq_len * hidden_dim;
    if w_down.len() < down_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("w_down length {} < expected {down_expected}", w_down.len()),
        }
        .into());
    }
    if output.len() < out_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < expected {out_expected}", output.len()),
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
    const TOL: f32 = 1e-3;

    fn assert_near(a: f32, b: f32, tol: f32, msg: &str) {
        assert!((a - b).abs() < tol, "{msg}: {a} vs {b} (tol={tol})");
    }

    // =======================================================================
    // Activation tests
    // =======================================================================

    #[test]
    fn test_silu_at_zero() {
        assert_near(ActivationType::SiLU.apply(0.0), 0.0, EPS, "SiLU(0)");
    }

    #[test]
    fn test_silu_positive() {
        assert_near(ActivationType::SiLU.apply(1.0), 0.7311, TOL, "SiLU(1)");
    }

    #[test]
    fn test_silu_negative() {
        assert_near(ActivationType::SiLU.apply(-1.0), -0.2689, TOL, "SiLU(-1)");
    }

    #[test]
    fn test_silu_large_positive() {
        let result = ActivationType::SiLU.apply(10.0);
        assert_near(result, 10.0, TOL, "SiLU(10)");
    }

    #[test]
    fn test_silu_large_negative_near_zero() {
        let result = ActivationType::SiLU.apply(-100.0);
        assert!(result.is_finite());
        assert!(result.abs() < 1e-10, "SiLU(-100) should be ~0");
    }

    #[test]
    fn test_gelu_at_zero() {
        assert_near(ActivationType::GELU.apply(0.0), 0.0, EPS, "GELU(0)");
    }

    #[test]
    fn test_gelu_positive() {
        assert_near(ActivationType::GELU.apply(1.0), 0.8413, TOL, "GELU(1)");
    }

    #[test]
    fn test_gelu_negative() {
        assert_near(ActivationType::GELU.apply(-1.0), -0.1587, TOL, "GELU(-1)");
    }

    #[test]
    fn test_gelu_large_positive() {
        assert_near(ActivationType::GELU.apply(100.0), 100.0, 0.1, "GELU(100)");
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
    fn test_swish_beta1_equals_silu() {
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let silu = ActivationType::SiLU.apply(x);
            let swish = ActivationType::Swish(1.0).apply(x);
            assert_near(silu, swish, EPS, &format!("SiLU vs Swish(1) at {x}"));
        }
    }

    #[test]
    fn test_swish_beta0_is_half_x() {
        // Swish(β=0): x * σ(0) = x * 0.5
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let result = ActivationType::Swish(0.0).apply(x);
            assert_near(result, x * 0.5, EPS, &format!("Swish(0) at {x}"));
        }
    }

    #[test]
    fn test_swish_large_beta_approaches_relu() {
        // Swish(β→∞) approaches ReLU for positive x
        let x = 1.0;
        let result = ActivationType::Swish(100.0).apply(x);
        assert_near(result, x, TOL, "Swish(100)(1) ≈ ReLU(1)");

        let neg = ActivationType::Swish(100.0).apply(-1.0);
        assert!(neg.abs() < 0.01, "Swish(100)(-1) ≈ 0");
    }

    // =======================================================================
    // SwiGluConfig tests
    // =======================================================================

    #[test]
    fn test_config_new_valid() {
        let cfg = SwiGluConfig::new(256, 512, false, ActivationType::SiLU).unwrap();
        assert_eq!(cfg.hidden_dim, 256);
        assert_eq!(cfg.intermediate_dim, 512);
        assert!(!cfg.use_bias);
    }

    #[test]
    fn test_config_with_bias() {
        let cfg = SwiGluConfig::new(128, 256, true, ActivationType::GELU).unwrap();
        assert!(cfg.use_bias);
        assert_eq!(cfg.activation_type, ActivationType::GELU);
    }

    #[test]
    fn test_config_rejects_zero_hidden() {
        assert!(SwiGluConfig::new(0, 512, false, ActivationType::SiLU).is_err());
    }

    #[test]
    fn test_config_rejects_zero_intermediate() {
        assert!(SwiGluConfig::new(256, 0, false, ActivationType::SiLU).is_err());
    }

    #[test]
    fn test_config_llama_7b() {
        let cfg = SwiGluConfig::llama_7b();
        assert_eq!(cfg.hidden_dim, 4096);
        assert_eq!(cfg.intermediate_dim, 11008);
        assert!(!cfg.use_bias);
    }

    #[test]
    fn test_config_bitnet_2b() {
        let cfg = SwiGluConfig::bitnet_2b();
        assert_eq!(cfg.hidden_dim, 2048);
        assert_eq!(cfg.intermediate_dim, 5632);
    }

    // =======================================================================
    // GateProjection tests
    // =======================================================================

    #[test]
    fn test_gate_projection_identity_weights() {
        let h = 3;
        let inter = 3;
        let x = vec![1.0, 2.0, 3.0];
        let w = identity_weight(h, inter);
        let proj = GateProjection::compute(&x, &w, &w, 1, h, inter).unwrap();
        for i in 0..h {
            assert_near(proj.gate[i], x[i], EPS, &format!("gate[{i}]"));
            assert_near(proj.up[i], x[i], EPS, &format!("up[{i}]"));
        }
    }

    #[test]
    fn test_gate_projection_zero_weights() {
        let h = 4;
        let inter = 4;
        let x = vec![1.0_f32; h];
        let w_zero = vec![0.0_f32; h * inter];
        let proj = GateProjection::compute(&x, &w_zero, &w_zero, 1, h, inter).unwrap();
        for &v in &proj.gate {
            assert_near(v, 0.0, EPS, "zero gate");
        }
        for &v in &proj.up {
            assert_near(v, 0.0, EPS, "zero up");
        }
    }

    #[test]
    fn test_gate_projection_with_bias() {
        let h = 2;
        let inter = 2;
        let x = vec![1.0, 0.0];
        let w = identity_weight(h, inter);
        let b_gate = vec![0.5, 0.5];
        let b_up = vec![-0.5, -0.5];
        let proj = GateProjection::compute_with_bias(
            &x, &w, &w, &b_gate, &b_up, 1, h, inter,
        )
        .unwrap();
        assert_near(proj.gate[0], 1.5, EPS, "gate[0]+bias");
        assert_near(proj.gate[1], 0.5, EPS, "gate[1]+bias");
        assert_near(proj.up[0], 0.5, EPS, "up[0]+bias");
        assert_near(proj.up[1], -0.5, EPS, "up[1]+bias");
    }

    #[test]
    fn test_gate_projection_rejects_bad_dims() {
        assert!(GateProjection::compute(&[1.0], &[1.0], &[1.0], 1, 2, 2).is_err());
    }

    // =======================================================================
    // GateActivation tests
    // =======================================================================

    #[test]
    fn test_gate_activation_silu() {
        let gate = vec![0.0, 1.0, -1.0, 2.0];
        let up = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0_f32; 4];
        GateActivation::apply(&gate, &up, &mut out, ActivationType::SiLU).unwrap();
        assert_near(out[0], 0.0, EPS, "act(0)*1");
        assert_near(out[1], 0.7311, TOL, "act(1)*1");
    }

    #[test]
    fn test_gate_activation_relu() {
        let gate = vec![-1.0, 0.0, 1.0, 2.0];
        let up = vec![2.0, 2.0, 2.0, 2.0];
        let mut out = vec![0.0_f32; 4];
        GateActivation::apply(&gate, &up, &mut out, ActivationType::ReLU).unwrap();
        assert_near(out[0], 0.0, EPS, "ReLU(-1)*2");
        assert_near(out[1], 0.0, EPS, "ReLU(0)*2");
        assert_near(out[2], 2.0, EPS, "ReLU(1)*2");
        assert_near(out[3], 4.0, EPS, "ReLU(2)*2");
    }

    #[test]
    fn test_gate_activation_inplace() {
        let mut gate = vec![1.0, 2.0, -1.0, 0.0];
        let up = vec![1.0, 1.0, 1.0, 1.0];
        GateActivation::apply_inplace(&mut gate, &up, ActivationType::ReLU).unwrap();
        assert_near(gate[0], 1.0, EPS, "inplace[0]");
        assert_near(gate[1], 2.0, EPS, "inplace[1]");
        assert_near(gate[2], 0.0, EPS, "inplace[2]");
        assert_near(gate[3], 0.0, EPS, "inplace[3]");
    }

    #[test]
    fn test_gate_activation_size_mismatch() {
        let gate = vec![1.0, 2.0];
        let up = vec![1.0];
        let mut out = vec![0.0_f32; 2];
        assert!(GateActivation::apply(&gate, &up, &mut out, ActivationType::SiLU).is_err());
    }

    #[test]
    fn test_gate_activation_vectorized_path() {
        // Input size > 4 to exercise the vectorized (4-wide) path
        let n = 9;
        let gate: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let up = vec![1.0_f32; n];
        let mut out = vec![0.0_f32; n];
        GateActivation::apply(&gate, &up, &mut out, ActivationType::SiLU).unwrap();
        for i in 0..n {
            let expected = ActivationType::SiLU.apply(gate[i]) * up[i];
            assert_near(out[i], expected, EPS, &format!("vec[{i}]"));
        }
    }

    // =======================================================================
    // SwiGluLayer (unfused) tests
    // =======================================================================

    #[test]
    fn test_swiglu_layer_zero_input() {
        let cfg = SwiGluConfig::new(4, 8, false, ActivationType::SiLU).unwrap();
        let x = vec![0.0_f32; 4];
        let w_gate = vec![1.0_f32; 4 * 8];
        let w_up = vec![1.0_f32; 4 * 8];
        let w_down = vec![1.0_f32; 8 * 4];
        let mut out = vec![99.0_f32; 4];
        SwiGluLayer::forward(&x, &w_gate, &w_up, &w_down, &mut out, &cfg, 1).unwrap();
        for &v in &out {
            assert_near(v, 0.0, EPS, "zero input → zero output");
        }
    }

    #[test]
    fn test_swiglu_layer_identity_relu() {
        let h = 3;
        let cfg = SwiGluConfig::new(h, h, false, ActivationType::ReLU).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let w = identity_weight(h, h);
        let mut out = vec![0.0_f32; h];
        SwiGluLayer::forward(&x, &w, &w, &w, &mut out, &cfg, 1).unwrap();
        // gate=x, ReLU(x)=x, up=x, hidden=x*x, down=x*x
        for i in 0..h {
            assert_near(out[i], x[i] * x[i], EPS, &format!("identity ReLU [{i}]"));
        }
    }

    #[test]
    fn test_swiglu_layer_silu_seq1() {
        let h = 2;
        let inter = 2;
        let cfg = SwiGluConfig::new(h, inter, false, ActivationType::SiLU).unwrap();
        let x = vec![1.0, 1.0];
        let w = identity_weight(h, inter);
        let mut out = vec![0.0_f32; h];
        SwiGluLayer::forward(&x, &w, &w, &w, &mut out, &cfg, 1).unwrap();
        let expected = ActivationType::SiLU.apply(1.0) * 1.0;
        for i in 0..h {
            assert_near(out[i], expected, TOL, &format!("silu seq1 [{i}]"));
        }
    }

    #[test]
    fn test_swiglu_layer_multiple_seqs() {
        let h = 2;
        let inter = 2;
        let seq = 2;
        let cfg = SwiGluConfig::new(h, inter, false, ActivationType::ReLU).unwrap();
        let x = vec![1.0, 0.0, 0.0, 1.0];
        let w = identity_weight(h, inter);
        let mut out = vec![0.0_f32; seq * h];
        SwiGluLayer::forward(&x, &w, &w, &w, &mut out, &cfg, seq).unwrap();
        assert_near(out[0], 1.0, EPS, "seq0[0]");
        assert_near(out[1], 0.0, EPS, "seq0[1]");
        assert_near(out[2], 0.0, EPS, "seq1[0]");
        assert_near(out[3], 1.0, EPS, "seq1[1]");
    }

    // =======================================================================
    // FusedSwiGlu tests
    // =======================================================================

    #[test]
    fn test_fused_matches_unfused_silu() {
        let h = 4;
        let inter = 6;
        let seq = 2;
        let cfg = SwiGluConfig::new(h, inter, false, ActivationType::SiLU).unwrap();
        let x: Vec<f32> = (0..seq * h).map(|i| (i as f32) * 0.1).collect();
        let w_gate: Vec<f32> = (0..h * inter).map(|i| (i as f32) * 0.01).collect();
        let w_up: Vec<f32> = (0..h * inter).map(|i| (i as f32) * -0.01 + 0.5).collect();
        let w_down: Vec<f32> = (0..inter * h).map(|i| (i as f32) * 0.02 - 0.3).collect();

        let mut unfused_out = vec![0.0_f32; seq * h];
        SwiGluLayer::forward(&x, &w_gate, &w_up, &w_down, &mut unfused_out, &cfg, seq).unwrap();

        let mut fused_out = vec![0.0_f32; seq * h];
        FusedSwiGlu::forward(&x, &w_gate, &w_up, &w_down, &mut fused_out, &cfg, seq).unwrap();

        for i in 0..seq * h {
            assert_near(
                fused_out[i],
                unfused_out[i],
                EPS,
                &format!("fused vs unfused [{i}]"),
            );
        }
    }

    #[test]
    fn test_fused_matches_unfused_relu() {
        let h = 3;
        let inter = 5;
        let cfg = SwiGluConfig::new(h, inter, false, ActivationType::ReLU).unwrap();
        let x: Vec<f32> = vec![0.5, -0.3, 0.8];
        let w_gate: Vec<f32> = (0..h * inter).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let w_up: Vec<f32> = (0..h * inter).map(|i| (i as f32) * -0.015 + 0.2).collect();
        let w_down: Vec<f32> = (0..inter * h).map(|i| (i as f32) * 0.01).collect();

        let mut unfused = vec![0.0_f32; h];
        SwiGluLayer::forward(&x, &w_gate, &w_up, &w_down, &mut unfused, &cfg, 1).unwrap();

        let mut fused = vec![0.0_f32; h];
        FusedSwiGlu::forward(&x, &w_gate, &w_up, &w_down, &mut fused, &cfg, 1).unwrap();

        for i in 0..h {
            assert_near(fused[i], unfused[i], EPS, &format!("fused ReLU [{i}]"));
        }
    }

    #[test]
    fn test_fused_matches_unfused_gelu() {
        let h = 4;
        let inter = 4;
        let cfg = SwiGluConfig::new(h, inter, false, ActivationType::GELU).unwrap();
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let w = identity_weight(h, inter);

        let mut unfused = vec![0.0_f32; h];
        SwiGluLayer::forward(&x, &w, &w, &w, &mut unfused, &cfg, 1).unwrap();

        let mut fused = vec![0.0_f32; h];
        FusedSwiGlu::forward(&x, &w, &w, &w, &mut fused, &cfg, 1).unwrap();

        for i in 0..h {
            assert_near(fused[i], unfused[i], EPS, &format!("fused GELU [{i}]"));
        }
    }

    #[test]
    fn test_fused_matches_unfused_swish() {
        let h = 3;
        let inter = 3;
        let cfg = SwiGluConfig::new(h, inter, false, ActivationType::Swish(2.0)).unwrap();
        let x = vec![0.5, -0.5, 1.0];
        let w = identity_weight(h, inter);

        let mut unfused = vec![0.0_f32; h];
        SwiGluLayer::forward(&x, &w, &w, &w, &mut unfused, &cfg, 1).unwrap();

        let mut fused = vec![0.0_f32; h];
        FusedSwiGlu::forward(&x, &w, &w, &w, &mut fused, &cfg, 1).unwrap();

        for i in 0..h {
            assert_near(fused[i], unfused[i], EPS, &format!("fused Swish [{i}]"));
        }
    }

    #[test]
    fn test_fused_with_bias() {
        let h = 2;
        let inter = 2;
        let cfg = SwiGluConfig::new(h, inter, true, ActivationType::ReLU).unwrap();
        let x = vec![1.0, 0.0];
        let w = identity_weight(h, inter);
        let b_gate = vec![0.0; inter];
        let b_up = vec![0.0; inter];
        let b_down = vec![1.0; h]; // adds 1.0 to output

        let mut out = vec![0.0_f32; h];
        FusedSwiGlu::forward_with_bias(
            &x, &w, &w, &w, &b_gate, &b_up, &b_down, &mut out, &cfg, 1,
        )
        .unwrap();

        // Without bias: gate=[1,0], ReLU→[1,0], up=[1,0], hidden=[1,0], down=[1,0]
        // With b_down=[1,1]: [2,1]
        assert_near(out[0], 2.0, EPS, "bias out[0]");
        assert_near(out[1], 1.0, EPS, "bias out[1]");
    }

    #[test]
    fn test_fused_rejects_bad_output_size() {
        let cfg = SwiGluConfig::new(4, 4, false, ActivationType::SiLU).unwrap();
        let x = vec![0.0_f32; 4];
        let w = vec![0.0_f32; 16];
        let mut out = vec![0.0_f32; 2]; // too small
        assert!(FusedSwiGlu::forward(&x, &w, &w, &w, &mut out, &cfg, 1).is_err());
    }

    // =======================================================================
    // CPU reference tests
    // =======================================================================

    #[test]
    fn test_ref_matches_layer_forward() {
        let h = 4;
        let inter = 6;
        let cfg = SwiGluConfig::new(h, inter, false, ActivationType::SiLU).unwrap();
        let x: Vec<f32> = (0..h).map(|i| i as f32 * 0.3).collect();
        let w_gate: Vec<f32> = (0..h * inter).map(|i| i as f32 * 0.01).collect();
        let w_up: Vec<f32> = (0..h * inter).map(|i| i as f32 * -0.01 + 0.1).collect();
        let w_down: Vec<f32> = (0..inter * h).map(|i| i as f32 * 0.02).collect();

        let mut layer_out = vec![0.0_f32; h];
        SwiGluLayer::forward(&x, &w_gate, &w_up, &w_down, &mut layer_out, &cfg, 1).unwrap();

        let mut ref_out = vec![0.0_f32; h];
        swiglu_forward_ref(&x, &w_gate, &w_up, &w_down, &mut ref_out, 1, h, inter, ActivationType::SiLU).unwrap();

        for i in 0..h {
            assert_near(layer_out[i], ref_out[i], EPS, &format!("layer vs ref [{i}]"));
        }
    }

    #[test]
    fn test_ref_zero_gate_gives_zero() {
        let h = 4;
        let inter = 6;
        let x = vec![1.0_f32; h];
        let w_gate = vec![0.0_f32; h * inter];
        let w_up = vec![1.0_f32; h * inter];
        let w_down = vec![1.0_f32; inter * h];
        let mut out = vec![99.0_f32; h];
        swiglu_forward_ref(&x, &w_gate, &w_up, &w_down, &mut out, 1, h, inter, ActivationType::SiLU).unwrap();
        for &v in &out {
            assert_near(v, 0.0, EPS, "zero gate → zero");
        }
    }

    #[test]
    fn test_ref_zero_up_gives_zero() {
        let h = 4;
        let inter = 4;
        let x = vec![1.0_f32; h];
        let w_gate = vec![1.0_f32; h * inter];
        let w_up = vec![0.0_f32; h * inter];
        let w_down = vec![1.0_f32; inter * h];
        let mut out = vec![99.0_f32; h];
        swiglu_forward_ref(&x, &w_gate, &w_up, &w_down, &mut out, 1, h, inter, ActivationType::SiLU).unwrap();
        for &v in &out {
            assert_near(v, 0.0, EPS, "zero up → zero");
        }
    }

    // =======================================================================
    // SwiGluStats tests
    // =======================================================================

    #[test]
    fn test_stats_compute_basic() {
        let hidden = vec![0.0, 1.0, -1.0, 0.5, -0.5, 0.001];
        let stats = SwiGluStats::compute(&hidden, 1, 4, 6, 0.01);
        assert_eq!(stats.activation_range.0, -1.0);
        assert_eq!(stats.activation_range.1, 1.0);
        assert!(stats.sparsity_ratio > 0.0, "some near-zero values");
    }

    #[test]
    fn test_stats_empty_hidden() {
        let stats = SwiGluStats::compute(&[], 0, 4, 6, 0.01);
        assert_eq!(stats.sparsity_ratio, 0.0);
    }

    #[test]
    fn test_stats_all_sparse() {
        let hidden = vec![0.001, -0.001, 0.0, 0.005];
        let stats = SwiGluStats::compute(&hidden, 1, 4, 4, 0.01);
        assert_near(stats.sparsity_ratio, 1.0, EPS, "all sparse");
    }

    #[test]
    fn test_stats_none_sparse() {
        let hidden = vec![1.0, 2.0, 3.0, 4.0];
        let stats = SwiGluStats::compute(&hidden, 1, 4, 4, 0.01);
        assert_near(stats.sparsity_ratio, 0.0, EPS, "none sparse");
    }

    #[test]
    fn test_stats_flops_estimate() {
        let flops = SwiGluStats::estimate_flops(1, 2048, 5632);
        assert!(flops > 0, "FLOPs should be positive");
        // gate + up + down matmuls dominate
        let expected_matmul = 2 * 1 * 2048u64 * 5632 * 3;
        assert!(flops > expected_matmul, "FLOPs should include activation costs");
    }

    // =======================================================================
    // MoERouter tests
    // =======================================================================

    #[test]
    fn test_moe_router_top1() {
        let router = MoERouter::new(4, 1).unwrap();
        let logits = vec![0.1, 0.5, 0.2, 0.3];
        let assignments = router.route(&logits, 1).unwrap();
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].expert_indices, vec![1]); // highest logit
        assert_near(assignments[0].weights[0], 1.0, EPS, "top-1 weight");
    }

    #[test]
    fn test_moe_router_top2() {
        let router = MoERouter::new(4, 2).unwrap();
        let logits = vec![0.1, 0.9, 0.2, 0.8];
        let assignments = router.route(&logits, 1).unwrap();
        assert_eq!(assignments[0].expert_indices.len(), 2);
        assert_eq!(assignments[0].expert_indices[0], 1); // highest
        assert_eq!(assignments[0].expert_indices[1], 3); // second highest
        // Weights should sum to 1.0
        let wsum: f32 = assignments[0].weights.iter().sum();
        assert_near(wsum, 1.0, EPS, "weights sum");
    }

    #[test]
    fn test_moe_router_multiple_tokens() {
        let router = MoERouter::new(3, 1).unwrap();
        let logits = vec![
            0.1, 0.5, 0.2, // token 0 → expert 1
            0.8, 0.1, 0.3, // token 1 → expert 0
        ];
        let assignments = router.route(&logits, 2).unwrap();
        assert_eq!(assignments[0].expert_indices[0], 1);
        assert_eq!(assignments[1].expert_indices[0], 0);
    }

    #[test]
    fn test_moe_router_equal_logits() {
        let router = MoERouter::new(3, 2).unwrap();
        let logits = vec![1.0, 1.0, 1.0];
        let assignments = router.route(&logits, 1).unwrap();
        assert_eq!(assignments[0].expert_indices.len(), 2);
        // All equal → top-2 picked by stable sort order
        let wsum: f32 = assignments[0].weights.iter().sum();
        assert_near(wsum, 1.0, EPS, "equal logits weight sum");
    }

    #[test]
    fn test_moe_router_rejects_topk_gt_experts() {
        assert!(MoERouter::new(3, 5).is_err());
    }

    #[test]
    fn test_moe_router_rejects_zero_experts() {
        assert!(MoERouter::new(0, 1).is_err());
    }

    #[test]
    fn test_moe_router_rejects_zero_topk() {
        assert!(MoERouter::new(4, 0).is_err());
    }

    #[test]
    fn test_moe_router_rejects_short_logits() {
        let router = MoERouter::new(4, 1).unwrap();
        assert!(router.route(&[0.1, 0.2], 1).is_err());
    }

    #[test]
    fn test_moe_router_topk_equals_experts() {
        let router = MoERouter::new(3, 3).unwrap();
        let logits = vec![0.3, 0.1, 0.5];
        let assignments = router.route(&logits, 1).unwrap();
        assert_eq!(assignments[0].expert_indices.len(), 3);
        let wsum: f32 = assignments[0].weights.iter().sum();
        assert_near(wsum, 1.0, EPS, "full selection weight sum");
    }

    #[test]
    fn test_moe_router_negative_logits() {
        let router = MoERouter::new(4, 1).unwrap();
        let logits = vec![-10.0, -5.0, -20.0, -1.0];
        let assignments = router.route(&logits, 1).unwrap();
        assert_eq!(assignments[0].expert_indices[0], 3); // least negative
    }

    // =======================================================================
    // OpenCL kernel source tests
    // =======================================================================

    #[test]
    fn test_swiglu_cl_not_empty() {
        assert!(!SWIGLU_CL.is_empty());
    }

    #[test]
    fn test_swiglu_cl_contains_entry_point() {
        assert!(SWIGLU_CL.contains("__kernel void fused_swiglu"));
    }

    #[test]
    fn test_swiglu_cl_contains_silu() {
        assert!(SWIGLU_CL.contains("silu"));
    }

    #[test]
    fn test_swiglu_cl_contains_barrier() {
        assert!(SWIGLU_CL.contains("barrier(CLK_GLOBAL_MEM_FENCE)"));
    }

    #[test]
    fn test_swiglu_cl_contains_activation_dispatch() {
        assert!(SWIGLU_CL.contains("activation_type"));
    }

    #[test]
    fn test_elementwise_cl_not_empty() {
        assert!(!ELEMENTWISE_CL.is_empty());
    }

    #[test]
    fn test_elementwise_cl_has_silu_kernel() {
        assert!(ELEMENTWISE_CL.contains("__kernel void ewise_silu"));
    }

    #[test]
    fn test_elementwise_cl_has_swish_kernel() {
        assert!(ELEMENTWISE_CL.contains("__kernel void ewise_swish"));
    }

    #[test]
    fn test_elementwise_cl_has_gate_mul_kernel() {
        assert!(ELEMENTWISE_CL.contains("__kernel void ewise_gate_mul"));
    }

    // =======================================================================
    // Dimension ratio tests
    // =======================================================================

    #[test]
    fn test_swiglu_hidden_gt_intermediate() {
        let h = 8;
        let inter = 4;
        let cfg = SwiGluConfig::new(h, inter, false, ActivationType::SiLU).unwrap();
        let x = vec![0.1_f32; h];
        let w_gate = vec![0.01_f32; h * inter];
        let w_up = vec![0.01_f32; h * inter];
        let w_down = vec![0.01_f32; inter * h];
        let mut out = vec![0.0_f32; h];
        FusedSwiGlu::forward(&x, &w_gate, &w_up, &w_down, &mut out, &cfg, 1).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_swiglu_4x_expansion() {
        let h = 8;
        let inter = 32;
        let cfg = SwiGluConfig::new(h, inter, false, ActivationType::SiLU).unwrap();
        let x = vec![0.1_f32; h];
        let w_gate = vec![0.01_f32; h * inter];
        let w_up = vec![0.01_f32; h * inter];
        let w_down = vec![0.01_f32; inter * h];
        let mut out = vec![0.0_f32; h];
        FusedSwiGlu::forward(&x, &w_gate, &w_up, &w_down, &mut out, &cfg, 1).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_swiglu_llama_ratio() {
        // intermediate = 2.6875 * hidden (LLaMA-2 ratio: 11008/4096)
        let h = 4;
        let inter = 11;
        let cfg = SwiGluConfig::new(h, inter, false, ActivationType::SiLU).unwrap();
        let x = vec![0.1_f32; h];
        let w_gate = vec![0.01_f32; h * inter];
        let w_up = vec![0.01_f32; h * inter];
        let w_down = vec![0.01_f32; inter * h];
        let mut out = vec![0.0_f32; h];
        FusedSwiGlu::forward(&x, &w_gate, &w_up, &w_down, &mut out, &cfg, 1).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // =======================================================================
    // Edge cases
    // =======================================================================

    #[test]
    fn test_swiglu_single_element() {
        let cfg = SwiGluConfig::new(1, 1, false, ActivationType::ReLU).unwrap();
        let x = vec![2.0_f32];
        let w = vec![1.0_f32];
        let mut out = vec![0.0_f32];
        FusedSwiGlu::forward(&x, &w, &w, &w, &mut out, &cfg, 1).unwrap();
        // gate=2, ReLU(2)=2, up=2, hidden=4, down=4
        assert_near(out[0], 4.0, EPS, "single element");
    }

    #[test]
    fn test_swiglu_zero_weights() {
        let h = 4;
        let inter = 4;
        let cfg = SwiGluConfig::new(h, inter, false, ActivationType::SiLU).unwrap();
        let x = vec![1.0_f32; h];
        let w_zero = vec![0.0_f32; h * inter];
        let mut out = vec![99.0_f32; h];
        FusedSwiGlu::forward(&x, &w_zero, &w_zero, &w_zero, &mut out, &cfg, 1).unwrap();
        for &v in &out {
            assert_near(v, 0.0, EPS, "zero weights → zero output");
        }
    }

    // =======================================================================
    // Numerical stability
    // =======================================================================

    #[test]
    fn test_swiglu_large_weights_finite() {
        let h = 4;
        let inter = 4;
        let cfg = SwiGluConfig::new(h, inter, false, ActivationType::SiLU).unwrap();
        let x = vec![1.0_f32; h];
        let w_gate = vec![10.0_f32; h * inter];
        let w_up = vec![10.0_f32; h * inter];
        let w_down = vec![0.001_f32; inter * h];
        let mut out = vec![0.0_f32; h];
        FusedSwiGlu::forward(&x, &w_gate, &w_up, &w_down, &mut out, &cfg, 1).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "large weights should stay finite");
    }

    #[test]
    fn test_activation_sweep_all_types() {
        let activations = [
            ActivationType::SiLU,
            ActivationType::GELU,
            ActivationType::ReLU,
            ActivationType::Swish(0.5),
            ActivationType::Swish(1.0),
            ActivationType::Swish(2.0),
        ];
        for act in activations {
            let h = 4;
            let inter = 4;
            let cfg = SwiGluConfig::new(h, inter, false, act).unwrap();
            let x = vec![0.5_f32; h];
            let w = identity_weight(h, inter);
            let mut out = vec![0.0_f32; h];
            FusedSwiGlu::forward(&x, &w, &w, &w, &mut out, &cfg, 1).unwrap();
            assert!(
                out.iter().all(|v| v.is_finite()),
                "activation {act:?} produced non-finite output"
            );
        }
    }

    // =======================================================================
    // Property tests
    // =======================================================================

    #[test]
    fn test_output_bounded_for_bounded_input() {
        // With small weights and bounded input, output should be bounded.
        let h = 4;
        let inter = 4;
        let cfg = SwiGluConfig::new(h, inter, false, ActivationType::SiLU).unwrap();
        let x = vec![1.0_f32; h];
        let w = vec![0.1_f32; h * inter];
        let mut out = vec![0.0_f32; h];
        FusedSwiGlu::forward(&x, &w, &w, &w, &mut out, &cfg, 1).unwrap();
        // SiLU(x) <= x for x>0, so output should be bounded
        let max_abs = out.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(
            max_abs < 100.0,
            "output magnitude {max_abs} should be bounded for small weights"
        );
    }

    #[test]
    fn test_silu_output_range_property() {
        // SiLU(x) ∈ [~-0.278, ∞) and SiLU(x) <= x for x > 0
        for i in -100..=100 {
            let x = i as f32 * 0.1;
            let y = ActivationType::SiLU.apply(x);
            assert!(y.is_finite(), "SiLU({x}) must be finite");
            if x >= 0.0 {
                assert!(y <= x + EPS, "SiLU({x})={y} should be <= {x}");
                assert!(y >= -0.28, "SiLU({x})={y} should be >= -0.28");
            }
        }
    }

    #[test]
    fn test_relu_output_range_property() {
        for i in -100..=100 {
            let x = i as f32 * 0.1;
            let y = ActivationType::ReLU.apply(x);
            assert!(y >= 0.0, "ReLU({x}) must be >= 0");
            if x > 0.0 {
                assert_near(y, x, EPS, &format!("ReLU({x})"));
            }
        }
    }

    // =======================================================================
    // Matmul helper
    // =======================================================================

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
}
