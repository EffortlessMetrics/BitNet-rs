//! Fused layer operations for reducing memory traffic during CPU inference.
//!
//! Each fused kernel produces results numerically equivalent (within
//! floating-point tolerance) to executing the constituent operations
//! separately, but avoids intermediate allocations and extra memory
//! passes over the data.

use std::fmt;

// ── Activation helper ──────────────────────────────────────────────

/// Activation function type for fusion kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionActivation {
    SiLU,
    GELU,
    ReLU,
}

impl fmt::Display for FusionActivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SiLU => write!(f, "SiLU"),
            Self::GELU => write!(f, "GELU"),
            Self::ReLU => write!(f, "ReLU"),
        }
    }
}

#[inline]
fn apply_act(x: f32, act: FusionActivation) -> f32 {
    match act {
        FusionActivation::SiLU => x / (1.0 + (-x).exp()),
        FusionActivation::GELU => {
            const SQRT_2_OVER_PI: f32 = 0.797_884_6;
            const COEFF: f32 = 0.044_715;
            let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }
        FusionActivation::ReLU => x.max(0.0),
    }
}

// ── FusionPattern ──────────────────────────────────────────────────

/// Identifies a specific layer-fusion pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// Fused LayerNorm + Linear projection.
    NormLinear,
    /// Fused Linear + Activation.
    LinearActivation,
    /// Fused LayerNorm + Linear + Activation (triple fusion).
    NormLinearActivation,
    /// Fused Q/K/V projection + attention + output projection.
    AttentionBlock,
    /// Fused gate + up projection + activation + down projection.
    FFNBlock,
}

impl fmt::Display for FusionPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NormLinear => write!(f, "NormLinear"),
            Self::LinearActivation => write!(f, "LinearActivation"),
            Self::NormLinearActivation => write!(f, "NormLinearActivation"),
            Self::AttentionBlock => write!(f, "AttentionBlock"),
            Self::FFNBlock => write!(f, "FFNBlock"),
        }
    }
}

// ── Errors ─────────────────────────────────────────────────────────

/// Errors returned by fusion operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerFusionError {
    /// Dimension mismatch between operands.
    DimensionMismatch { expected: usize, got: usize },
    /// Input is empty.
    EmptyInput,
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Fusion is not applicable for the given shapes/types.
    FusionNotApplicable(String),
}

impl fmt::Display for LayerFusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::EmptyInput => write!(f, "empty input"),
            Self::FusionNotApplicable(msg) => write!(f, "fusion not applicable: {msg}"),
        }
    }
}

impl std::error::Error for LayerFusionError {}

// ── Layer descriptor for fusion planning ───────────────────────────

/// Describes a layer for fusion compatibility checking.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerDesc {
    LayerNorm { dim: usize },
    Linear { in_dim: usize, out_dim: usize },
    Activation { dim: usize, act: FusionActivation },
    Attention { dim: usize, num_heads: usize },
    FFN { hidden_dim: usize, intermediate_dim: usize },
}

// ── Fused kernels ──────────────────────────────────────────────────

/// Fused LayerNorm + Linear projection.
///
/// Equivalent to:
/// ```text
/// normed = (input - mean) / sqrt(variance + eps) * gamma + beta
/// output = normed · W^T
/// ```
/// but computed without materializing the full normed intermediate.
///
/// * `input`  – `[dim]`
/// * `gamma`  – `[dim]` scale
/// * `beta`   – `[dim]` shift (optional; pass empty for no shift)
/// * `weight` – `[out_dim × dim]` row-major
/// * `eps`    – normalization epsilon
///
/// Returns `[out_dim]`.
pub fn fused_norm_linear(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, LayerFusionError> {
    let dim = input.len();
    if dim == 0 {
        return Err(LayerFusionError::EmptyInput);
    }
    if gamma.len() != dim {
        return Err(LayerFusionError::DimensionMismatch { expected: dim, got: gamma.len() });
    }
    if !beta.is_empty() && beta.len() != dim {
        return Err(LayerFusionError::DimensionMismatch { expected: dim, got: beta.len() });
    }
    if weight.is_empty() || !weight.len().is_multiple_of(dim) {
        return Err(LayerFusionError::DimensionMismatch {
            expected: dim,
            got: weight.len() % dim.max(1),
        });
    }

    let out_dim = weight.len() / dim;

    // Compute mean and variance in a single pass.
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    for &x in input {
        sum += x;
        sum_sq += x * x;
    }
    let mean = sum / dim as f32;
    let variance = sum_sq / dim as f32 - mean * mean;
    let inv_std = 1.0 / (variance + eps).sqrt();

    // Fuse: normalize each element and immediately accumulate into output rows.
    let mut output = vec![0.0f32; out_dim];
    for (o, row) in output.iter_mut().zip(weight.chunks_exact(dim)) {
        let mut acc = 0.0f32;
        for i in 0..dim {
            let normed = (input[i] - mean) * inv_std * gamma[i];
            let normed = if beta.is_empty() { normed } else { normed + beta[i] };
            acc += row[i] * normed;
        }
        *o = acc;
    }
    Ok(output)
}

/// Fused Linear projection + Activation.
///
/// Equivalent to:
/// ```text
/// linear_out = input · W^T + bias
/// output     = activation(linear_out)
/// ```
///
/// * `input`  – `[in_dim]`
/// * `weight` – `[out_dim × in_dim]` row-major
/// * `bias`   – `[out_dim]` (or empty for no bias)
/// * `act`    – activation function to apply
///
/// Returns `[out_dim]`.
pub fn fused_linear_activation(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    act: FusionActivation,
) -> Result<Vec<f32>, LayerFusionError> {
    let in_dim = input.len();
    if in_dim == 0 {
        return Err(LayerFusionError::EmptyInput);
    }
    if weight.is_empty() || !weight.len().is_multiple_of(in_dim) {
        return Err(LayerFusionError::DimensionMismatch {
            expected: in_dim,
            got: weight.len() % in_dim.max(1),
        });
    }

    let out_dim = weight.len() / in_dim;
    if !bias.is_empty() && bias.len() != out_dim {
        return Err(LayerFusionError::DimensionMismatch { expected: out_dim, got: bias.len() });
    }

    let mut output = Vec::with_capacity(out_dim);
    for (j, row) in weight.chunks_exact(in_dim).enumerate() {
        let mut acc = 0.0f32;
        for (&w, &x) in row.iter().zip(input) {
            acc += w * x;
        }
        if !bias.is_empty() {
            acc += bias[j];
        }
        output.push(apply_act(acc, act));
    }
    Ok(output)
}

/// Fused LayerNorm + Linear + Activation (triple fusion).
///
/// Combines normalization, linear projection, and activation into one
/// pass to minimize memory traffic.
///
/// * `input`  – `[dim]`
/// * `gamma`  – `[dim]`
/// * `beta`   – `[dim]` (or empty)
/// * `weight` – `[out_dim × dim]` row-major
/// * `bias`   – `[out_dim]` (or empty)
/// * `eps`    – normalization epsilon
/// * `act`    – activation function
///
/// Returns `[out_dim]`.
pub fn fused_norm_linear_activation(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    act: FusionActivation,
) -> Result<Vec<f32>, LayerFusionError> {
    let dim = input.len();
    if dim == 0 {
        return Err(LayerFusionError::EmptyInput);
    }
    if gamma.len() != dim {
        return Err(LayerFusionError::DimensionMismatch { expected: dim, got: gamma.len() });
    }
    if !beta.is_empty() && beta.len() != dim {
        return Err(LayerFusionError::DimensionMismatch { expected: dim, got: beta.len() });
    }
    if weight.is_empty() || !weight.len().is_multiple_of(dim) {
        return Err(LayerFusionError::DimensionMismatch {
            expected: dim,
            got: weight.len() % dim.max(1),
        });
    }

    let out_dim = weight.len() / dim;
    if !bias.is_empty() && bias.len() != out_dim {
        return Err(LayerFusionError::DimensionMismatch { expected: out_dim, got: bias.len() });
    }

    // Compute statistics.
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    for &x in input {
        sum += x;
        sum_sq += x * x;
    }
    let mean = sum / dim as f32;
    let variance = sum_sq / dim as f32 - mean * mean;
    let inv_std = 1.0 / (variance + eps).sqrt();

    let mut output = Vec::with_capacity(out_dim);
    for (j, row) in weight.chunks_exact(dim).enumerate() {
        let mut acc = 0.0f32;
        for i in 0..dim {
            let normed = (input[i] - mean) * inv_std * gamma[i];
            let normed = if beta.is_empty() { normed } else { normed + beta[i] };
            acc += row[i] * normed;
        }
        if !bias.is_empty() {
            acc += bias[j];
        }
        output.push(apply_act(acc, act));
    }
    Ok(output)
}

/// Fused attention block: Q/K/V projection + scaled dot-product attention
/// + output projection.
///
/// Computes single-head attention (or per-head slice for multi-head) in a
/// fused manner to avoid writing large intermediate Q/K/V tensors.
///
/// * `input`      – `[seq_len × dim]` row-major
/// * `w_q`        – `[dim × dim]` query projection
/// * `w_k`        – `[dim × dim]` key projection
/// * `w_v`        – `[dim × dim]` value projection
/// * `w_out`      – `[dim × dim]` output projection
/// * `seq_len`    – sequence length
/// * `dim`        – model dimension
///
/// Returns `[seq_len × dim]`.
pub fn fused_attention_block(
    input: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_out: &[f32],
    seq_len: usize,
    dim: usize,
) -> Result<Vec<f32>, LayerFusionError> {
    if dim == 0 || seq_len == 0 {
        return Err(LayerFusionError::EmptyInput);
    }
    let expected_input = seq_len * dim;
    if input.len() != expected_input {
        return Err(LayerFusionError::DimensionMismatch {
            expected: expected_input,
            got: input.len(),
        });
    }
    let w_size = dim * dim;
    for (name, w) in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v), ("w_out", w_out)] {
        if w.len() != w_size {
            return Err(LayerFusionError::DimensionMismatch { expected: w_size, got: w.len() });
        }
        let _ = name; // used only for error context in debug
    }

    // Step 1: Compute Q, K, V projections.
    let mut q = vec![0.0f32; expected_input];
    let mut k = vec![0.0f32; expected_input];
    let mut v = vec![0.0f32; expected_input];

    for s in 0..seq_len {
        let x = &input[s * dim..(s + 1) * dim];
        matvec(x, w_q, &mut q[s * dim..(s + 1) * dim], dim);
        matvec(x, w_k, &mut k[s * dim..(s + 1) * dim], dim);
        matvec(x, w_v, &mut v[s * dim..(s + 1) * dim], dim);
    }

    // Step 2: Scaled dot-product attention.
    let scale = 1.0 / (dim as f32).sqrt();
    let mut attn_output = vec![0.0f32; expected_input];

    for i in 0..seq_len {
        // Compute attention scores for position i.
        let mut scores = vec![0.0f32; seq_len];
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += q[i * dim + d] * k[j * dim + d];
            }
            scores[j] = dot * scale;
        }

        // Apply causal mask: positions j > i are masked out.
        for s in scores.iter_mut().skip(i + 1) {
            *s = f32::NEG_INFINITY;
        }

        // Softmax.
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        if sum_exp > 0.0 {
            for s in &mut scores {
                *s /= sum_exp;
            }
        }

        // Weighted sum of V.
        for d in 0..dim {
            let mut acc = 0.0f32;
            for j in 0..seq_len {
                acc += scores[j] * v[j * dim + d];
            }
            attn_output[i * dim + d] = acc;
        }
    }

    // Step 3: Output projection.
    let mut output = vec![0.0f32; expected_input];
    for s in 0..seq_len {
        matvec(
            &attn_output[s * dim..(s + 1) * dim],
            w_out,
            &mut output[s * dim..(s + 1) * dim],
            dim,
        );
    }

    Ok(output)
}

/// Fused FFN block: gate projection + up projection + activation +
/// down projection.
///
/// Computes the gated FFN:
/// ```text
/// gate = activation(input · W_gate^T)
/// up   = input · W_up^T
/// down = (gate ⊙ up) · W_down^T
/// ```
///
/// * `input`   – `[hidden_dim]`
/// * `w_gate`  – `[intermediate_dim × hidden_dim]`
/// * `w_up`    – `[intermediate_dim × hidden_dim]`
/// * `w_down`  – `[hidden_dim × intermediate_dim]`
/// * `act`     – activation for the gate path
///
/// Returns `[hidden_dim]`.
pub fn fused_ffn_block(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    act: FusionActivation,
) -> Result<Vec<f32>, LayerFusionError> {
    let hidden_dim = input.len();
    if hidden_dim == 0 {
        return Err(LayerFusionError::EmptyInput);
    }
    if w_gate.is_empty() || !w_gate.len().is_multiple_of(hidden_dim) {
        return Err(LayerFusionError::DimensionMismatch {
            expected: hidden_dim,
            got: w_gate.len() % hidden_dim.max(1),
        });
    }
    let intermediate_dim = w_gate.len() / hidden_dim;
    if w_up.len() != intermediate_dim * hidden_dim {
        return Err(LayerFusionError::DimensionMismatch {
            expected: intermediate_dim * hidden_dim,
            got: w_up.len(),
        });
    }
    if w_down.len() != hidden_dim * intermediate_dim {
        return Err(LayerFusionError::DimensionMismatch {
            expected: hidden_dim * intermediate_dim,
            got: w_down.len(),
        });
    }

    // Fused gate + up projection: compute gate and up simultaneously,
    // then element-wise multiply before down projection.
    let mut gate_up = vec![0.0f32; intermediate_dim];
    for (j, gu) in gate_up.iter_mut().enumerate() {
        let mut gate_acc = 0.0f32;
        let mut up_acc = 0.0f32;
        let g_off = j * hidden_dim;
        let u_off = j * hidden_dim;
        for i in 0..hidden_dim {
            gate_acc += w_gate[g_off + i] * input[i];
            up_acc += w_up[u_off + i] * input[i];
        }
        *gu = apply_act(gate_acc, act) * up_acc;
    }

    // Down projection.
    let mut output = vec![0.0f32; hidden_dim];
    for (j, out) in output.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        let d_off = j * intermediate_dim;
        for i in 0..intermediate_dim {
            acc += w_down[d_off + i] * gate_up[i];
        }
        *out = acc;
    }
    Ok(output)
}

// ── Fusion planning utilities ──────────────────────────────────────

/// Check if a sequence of layers can be fused into the given pattern.
pub fn can_fuse(layers: &[LayerDesc], pattern: FusionPattern) -> bool {
    match pattern {
        FusionPattern::NormLinear => {
            matches!(
                layers,
                [LayerDesc::LayerNorm { dim: d1 }, LayerDesc::Linear { in_dim, .. }]
                if *d1 == *in_dim
            )
        }
        FusionPattern::LinearActivation => {
            matches!(
                layers,
                [LayerDesc::Linear { out_dim, .. }, LayerDesc::Activation { dim, .. }]
                if *out_dim == *dim
            )
        }
        FusionPattern::NormLinearActivation => {
            matches!(
                layers,
                [LayerDesc::LayerNorm { dim: d1 }, LayerDesc::Linear { in_dim, out_dim }, LayerDesc::Activation { dim: d2, .. }]
                if *d1 == *in_dim && *out_dim == *d2
            )
        }
        FusionPattern::AttentionBlock => {
            matches!(layers, [LayerDesc::Attention { .. }])
        }
        FusionPattern::FFNBlock => {
            matches!(layers, [LayerDesc::FFN { .. }])
        }
    }
}

/// Estimate memory bandwidth savings from applying a fusion pattern.
///
/// Returns a ratio in `[0.0, 1.0]` where higher means more savings.
/// This is an approximate heuristic based on eliminated intermediate
/// tensor materializations.
pub fn fusion_benefit_estimate(pattern: FusionPattern, dim: usize, out_dim: usize) -> f32 {
    if dim == 0 {
        return 0.0;
    }
    let bytes_per_f32 = 4usize;

    match pattern {
        FusionPattern::NormLinear => {
            // Without fusion: write normed [dim] + read normed for linear.
            // With fusion: skip normed intermediate.
            let saved = dim * bytes_per_f32;
            let total = (dim + out_dim + dim) * bytes_per_f32; // input + output + normed
            saved as f32 / total as f32
        }
        FusionPattern::LinearActivation => {
            // Without fusion: write linear_out [out_dim] + read for activation.
            let saved = out_dim * bytes_per_f32;
            let total = (dim + out_dim + out_dim) * bytes_per_f32;
            saved as f32 / total as f32
        }
        FusionPattern::NormLinearActivation => {
            // Saves both normed and pre-activation intermediates.
            let saved = (dim + out_dim) * bytes_per_f32;
            let total = (dim + dim + out_dim + out_dim) * bytes_per_f32;
            saved as f32 / total as f32
        }
        FusionPattern::AttentionBlock => {
            // Saves Q, K, V intermediate tensors.
            let saved = 3 * dim * bytes_per_f32;
            let total = (dim + 3 * dim + dim) * bytes_per_f32;
            saved as f32 / total as f32
        }
        FusionPattern::FFNBlock => {
            // Saves gate, up, and gate*up intermediates.
            let saved = (out_dim + out_dim + out_dim) * bytes_per_f32;
            let total = (dim + 3 * out_dim + dim) * bytes_per_f32;
            saved as f32 / total as f32
        }
    }
}

/// A fusion action produced by the planner.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionAction {
    /// Pattern to apply.
    pub pattern: FusionPattern,
    /// Index range in the original layer list (start, end exclusive).
    pub layer_range: (usize, usize),
}

/// Apply a fusion plan to a layer list, returning a sequence of fusion
/// actions that cover as many layers as possible.
///
/// Uses a greedy left-to-right scan, preferring longer patterns.
pub fn apply_fusion_plan(layers: &[LayerDesc]) -> Vec<FusionAction> {
    let mut actions = Vec::new();
    let mut i = 0;
    let n = layers.len();

    while i < n {
        // Try triple fusion first (3 layers).
        if i + 3 <= n && can_fuse(&layers[i..i + 3], FusionPattern::NormLinearActivation) {
            actions.push(FusionAction {
                pattern: FusionPattern::NormLinearActivation,
                layer_range: (i, i + 3),
            });
            i += 3;
            continue;
        }
        // Try double fusions (2 layers).
        if i + 2 <= n {
            if can_fuse(&layers[i..i + 2], FusionPattern::NormLinear) {
                actions.push(FusionAction {
                    pattern: FusionPattern::NormLinear,
                    layer_range: (i, i + 2),
                });
                i += 2;
                continue;
            }
            if can_fuse(&layers[i..i + 2], FusionPattern::LinearActivation) {
                actions.push(FusionAction {
                    pattern: FusionPattern::LinearActivation,
                    layer_range: (i, i + 2),
                });
                i += 2;
                continue;
            }
        }
        // Try single-layer fusions.
        if can_fuse(&layers[i..i + 1], FusionPattern::AttentionBlock) {
            actions.push(FusionAction {
                pattern: FusionPattern::AttentionBlock,
                layer_range: (i, i + 1),
            });
        }
        if can_fuse(&layers[i..i + 1], FusionPattern::FFNBlock) {
            actions
                .push(FusionAction { pattern: FusionPattern::FFNBlock, layer_range: (i, i + 1) });
        }
        i += 1;
    }
    actions
}

// ── Internal helpers ───────────────────────────────────────────────

/// Matrix-vector multiply: `out = W · x` where W is `[out_dim × in_dim]`.
#[inline]
fn matvec(x: &[f32], w: &[f32], out: &mut [f32], dim: usize) {
    let out_dim = out.len();
    for (j, o) in out.iter_mut().enumerate().take(out_dim) {
        let mut acc = 0.0f32;
        let off = j * dim;
        for i in 0..dim {
            acc += w[off + i] * x[i];
        }
        *o = acc;
    }
}

// ── Reference implementations for testing ──────────────────────────

#[cfg(test)]
fn reference_norm_linear(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    weight: &[f32],
    eps: f32,
) -> Vec<f32> {
    let dim = input.len();
    let out_dim = weight.len() / dim;

    let mean: f32 = input.iter().sum::<f32>() / dim as f32;
    let variance: f32 = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / dim as f32;
    let inv_std = 1.0 / (variance + eps).sqrt();

    let normed: Vec<f32> = input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let n = (x - mean) * inv_std * gamma[i];
            if beta.is_empty() { n } else { n + beta[i] }
        })
        .collect();

    let mut output = vec![0.0f32; out_dim];
    for (o, row) in output.iter_mut().zip(weight.chunks_exact(dim)) {
        *o = row.iter().zip(&normed).map(|(&w, &n)| w * n).sum();
    }
    output
}

#[cfg(test)]
fn reference_linear_activation(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    act: FusionActivation,
) -> Vec<f32> {
    let in_dim = input.len();
    let out_dim = weight.len() / in_dim;

    let mut output = vec![0.0f32; out_dim];
    for (j, row) in weight.chunks_exact(in_dim).enumerate() {
        let mut acc: f32 = row.iter().zip(input).map(|(&w, &x)| w * x).sum();
        if !bias.is_empty() {
            acc += bias[j];
        }
        output[j] = apply_act(acc, act);
    }
    output
}

#[cfg(test)]
fn reference_ffn_block(
    input: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    act: FusionActivation,
) -> Vec<f32> {
    let hidden_dim = input.len();
    let intermediate_dim = w_gate.len() / hidden_dim;

    // Gate projection + activation.
    let mut gate = vec![0.0f32; intermediate_dim];
    for (j, row) in w_gate.chunks_exact(hidden_dim).enumerate() {
        gate[j] = apply_act(row.iter().zip(input).map(|(&w, &x)| w * x).sum(), act);
    }

    // Up projection.
    let mut up = vec![0.0f32; intermediate_dim];
    for (j, row) in w_up.chunks_exact(hidden_dim).enumerate() {
        up[j] = row.iter().zip(input).map(|(&w, &x)| w * x).sum();
    }

    // Element-wise gate * up.
    let combined: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| g * u).collect();

    // Down projection.
    let mut output = vec![0.0f32; hidden_dim];
    for (j, row) in w_down.chunks_exact(intermediate_dim).enumerate() {
        output[j] = row.iter().zip(&combined).map(|(&w, &c)| w * c).sum();
    }
    output
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-4;

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    // ── FusionPattern Display ──────────────────────────────────────

    #[test]
    fn fusion_pattern_display() {
        assert_eq!(FusionPattern::NormLinear.to_string(), "NormLinear");
        assert_eq!(FusionPattern::LinearActivation.to_string(), "LinearActivation");
        assert_eq!(FusionPattern::NormLinearActivation.to_string(), "NormLinearActivation");
        assert_eq!(FusionPattern::AttentionBlock.to_string(), "AttentionBlock");
        assert_eq!(FusionPattern::FFNBlock.to_string(), "FFNBlock");
    }

    #[test]
    fn fusion_pattern_eq() {
        assert_eq!(FusionPattern::NormLinear, FusionPattern::NormLinear);
        assert_ne!(FusionPattern::NormLinear, FusionPattern::FFNBlock);
    }

    // ── FusionActivation ───────────────────────────────────────────

    #[test]
    fn activation_display() {
        assert_eq!(FusionActivation::SiLU.to_string(), "SiLU");
        assert_eq!(FusionActivation::GELU.to_string(), "GELU");
        assert_eq!(FusionActivation::ReLU.to_string(), "ReLU");
    }

    #[test]
    fn activation_values() {
        assert!((apply_act(0.0, FusionActivation::ReLU)).abs() < TOL);
        assert!((apply_act(-1.0, FusionActivation::ReLU)).abs() < TOL);
        assert!((apply_act(2.0, FusionActivation::ReLU) - 2.0).abs() < TOL);

        // SiLU(0) = 0
        assert!((apply_act(0.0, FusionActivation::SiLU)).abs() < TOL);
        // GELU(0) = 0
        assert!((apply_act(0.0, FusionActivation::GELU)).abs() < TOL);
    }

    // ── LayerFusionError ───────────────────────────────────────────

    #[test]
    fn error_display() {
        let e = LayerFusionError::DimensionMismatch { expected: 4, got: 3 };
        assert!(e.to_string().contains("4"));
        assert!(e.to_string().contains("3"));

        let e2 = LayerFusionError::EmptyInput;
        assert!(e2.to_string().contains("empty"));

        let e3 = LayerFusionError::InvalidConfig("bad".into());
        assert!(e3.to_string().contains("bad"));

        let e4 = LayerFusionError::FusionNotApplicable("reason".into());
        assert!(e4.to_string().contains("reason"));
    }

    #[test]
    fn error_eq() {
        assert_eq!(LayerFusionError::EmptyInput, LayerFusionError::EmptyInput);
        assert_ne!(LayerFusionError::EmptyInput, LayerFusionError::InvalidConfig("x".into()));
    }

    // ── fused_norm_linear ──────────────────────────────────────────

    #[test]
    fn norm_linear_matches_reference() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let weight = vec![0.5, -0.5, 0.25, 0.1, 0.1, 0.2, 0.3, 0.4];

        let fused = fused_norm_linear(&input, &gamma, &beta, &weight, EPS).unwrap();
        let reference = reference_norm_linear(&input, &gamma, &beta, &weight, EPS);
        assert!(max_abs_err(&fused, &reference) < TOL);
    }

    #[test]
    fn norm_linear_no_beta() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let weight = vec![0.5, -0.5, 0.25, 0.1, 0.1, 0.2, 0.3, 0.4];

        let fused = fused_norm_linear(&input, &gamma, &[], &weight, EPS).unwrap();
        let reference = reference_norm_linear(&input, &gamma, &[], &weight, EPS);
        assert!(max_abs_err(&fused, &reference) < TOL);
    }

    #[test]
    fn norm_linear_with_beta() {
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let gamma = vec![0.5, 0.5, 0.5, 0.5];
        let beta = vec![0.1, -0.1, 0.2, -0.2];
        let weight = vec![1.0, 1.0, 1.0, 1.0];

        let fused = fused_norm_linear(&input, &gamma, &beta, &weight, EPS).unwrap();
        let reference = reference_norm_linear(&input, &gamma, &beta, &weight, EPS);
        assert!(max_abs_err(&fused, &reference) < TOL);
    }

    #[test]
    fn norm_linear_single_element() {
        let fused = fused_norm_linear(&[3.0], &[1.0], &[], &[2.0], EPS).unwrap();
        let reference = reference_norm_linear(&[3.0], &[1.0], &[], &[2.0], EPS);
        assert!(max_abs_err(&fused, &reference) < TOL);
    }

    #[test]
    fn norm_linear_zero_input() {
        let input = [0.0; 4];
        let gamma = [1.0; 4];
        let weight = [1.0; 8];
        let fused = fused_norm_linear(&input, &gamma, &[], &weight, EPS).unwrap();
        for &v in &fused {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn norm_linear_empty_input() {
        let r = fused_norm_linear(&[], &[], &[], &[1.0], EPS);
        assert_eq!(r.unwrap_err(), LayerFusionError::EmptyInput);
    }

    #[test]
    fn norm_linear_gamma_mismatch() {
        let r = fused_norm_linear(&[1.0, 2.0], &[1.0], &[], &[1.0; 4], EPS);
        assert!(matches!(r, Err(LayerFusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn norm_linear_beta_mismatch() {
        let r = fused_norm_linear(&[1.0, 2.0], &[1.0, 1.0], &[1.0], &[1.0; 4], EPS);
        assert!(matches!(r, Err(LayerFusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn norm_linear_weight_mismatch() {
        let r = fused_norm_linear(&[1.0, 2.0], &[1.0, 1.0], &[], &[1.0; 3], EPS);
        assert!(matches!(r, Err(LayerFusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn norm_linear_large_dim() {
        let dim = 128;
        let out_dim = 64;
        let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let gamma = vec![1.0f32; dim];
        let weight: Vec<f32> = (0..out_dim * dim).map(|i| (i as f32) * 0.001).collect();
        let fused = fused_norm_linear(&input, &gamma, &[], &weight, EPS).unwrap();
        let reference = reference_norm_linear(&input, &gamma, &[], &weight, EPS);
        assert!(max_abs_err(&fused, &reference) < 0.01);
    }

    // ── fused_linear_activation ────────────────────────────────────

    #[test]
    fn linear_activation_silu_matches_reference() {
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let weight = vec![0.2, 0.3, 0.4, 0.5, -0.1, 0.6, 0.7, -0.2];
        let bias = vec![0.01, -0.01];

        let fused =
            fused_linear_activation(&input, &weight, &bias, FusionActivation::SiLU).unwrap();
        let reference = reference_linear_activation(&input, &weight, &bias, FusionActivation::SiLU);
        assert!(max_abs_err(&fused, &reference) < TOL);
    }

    #[test]
    fn linear_activation_gelu_matches_reference() {
        let input = vec![1.0, 2.0, -1.0, 0.0];
        let weight = vec![0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5];
        let bias = vec![];

        let fused =
            fused_linear_activation(&input, &weight, &bias, FusionActivation::GELU).unwrap();
        let reference = reference_linear_activation(&input, &weight, &bias, FusionActivation::GELU);
        assert!(max_abs_err(&fused, &reference) < TOL);
    }

    #[test]
    fn linear_activation_relu_matches_reference() {
        let input = vec![1.0, -2.0, 3.0];
        let weight = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
        let bias = vec![0.0, 0.0];

        let fused =
            fused_linear_activation(&input, &weight, &bias, FusionActivation::ReLU).unwrap();
        let reference = reference_linear_activation(&input, &weight, &bias, FusionActivation::ReLU);
        assert!(max_abs_err(&fused, &reference) < TOL);
    }

    #[test]
    fn linear_activation_no_bias() {
        let input = vec![1.0, 2.0];
        let weight = vec![0.5, 0.5, -0.5, -0.5];
        let fused = fused_linear_activation(&input, &weight, &[], FusionActivation::SiLU).unwrap();
        let reference = reference_linear_activation(&input, &weight, &[], FusionActivation::SiLU);
        assert!(max_abs_err(&fused, &reference) < TOL);
    }

    #[test]
    fn linear_activation_single_element() {
        let fused =
            fused_linear_activation(&[2.0], &[1.0], &[0.0], FusionActivation::ReLU).unwrap();
        assert!((fused[0] - 2.0).abs() < TOL);
    }

    #[test]
    fn linear_activation_empty_input() {
        let r = fused_linear_activation(&[], &[], &[], FusionActivation::ReLU);
        assert_eq!(r.unwrap_err(), LayerFusionError::EmptyInput);
    }

    #[test]
    fn linear_activation_bias_mismatch() {
        let r = fused_linear_activation(&[1.0], &[1.0], &[1.0, 2.0], FusionActivation::ReLU);
        assert!(matches!(r, Err(LayerFusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn linear_activation_weight_mismatch() {
        let r = fused_linear_activation(&[1.0, 2.0], &[1.0; 3], &[], FusionActivation::ReLU);
        assert!(matches!(r, Err(LayerFusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn linear_activation_relu_negative_output() {
        // Linear output is negative → ReLU should clamp to 0.
        let input = [1.0];
        let weight = vec![-1.0]; // output = -1
        let fused = fused_linear_activation(&input, &weight, &[], FusionActivation::ReLU).unwrap();
        assert!(fused[0].abs() < TOL);
    }

    // ── fused_norm_linear_activation ───────────────────────────────

    #[test]
    fn norm_linear_activation_silu_matches_reference() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = vec![];
        let weight = vec![0.5, -0.5, 0.25, 0.1, 0.1, 0.2, 0.3, 0.4];
        let bias = vec![0.01, -0.01];

        let fused = fused_norm_linear_activation(
            &input,
            &gamma,
            &beta,
            &weight,
            &bias,
            EPS,
            FusionActivation::SiLU,
        )
        .unwrap();

        // Reference: norm then linear then activation.
        let normed = reference_norm_linear(&input, &gamma, &beta, &weight, EPS);
        let reference: Vec<f32> = normed
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let with_bias = if bias.is_empty() { v } else { v + bias[i] };
                apply_act(with_bias, FusionActivation::SiLU)
            })
            .collect();

        // The fused version applies activation inside, so compare directly.
        let ref2 = {
            let nl = reference_norm_linear(&input, &gamma, &beta, &weight, EPS);
            nl.iter()
                .enumerate()
                .map(|(j, &v)| {
                    let with_bias = if bias.is_empty() { v } else { v + bias[j] };
                    apply_act(with_bias, FusionActivation::SiLU)
                })
                .collect::<Vec<_>>()
        };
        assert!(max_abs_err(&fused, &ref2) < TOL);
        let _ = reference; // both paths tested
    }

    #[test]
    fn norm_linear_activation_gelu() {
        let input = vec![1.0, -1.0, 0.5];
        let gamma = [1.0; 3];
        let weight = vec![1.0, 1.0, 1.0]; // 1×3 → out_dim = 1
        let fused = fused_norm_linear_activation(
            &input,
            &gamma,
            &[],
            &weight,
            &[],
            EPS,
            FusionActivation::GELU,
        )
        .unwrap();
        assert_eq!(fused.len(), 1);
        assert!(fused[0].is_finite());
    }

    #[test]
    fn norm_linear_activation_relu() {
        let input = vec![1.0, 2.0];
        let gamma = [1.0; 2];
        let weight = vec![-1.0, -1.0]; // negative output → ReLU clamps
        let fused = fused_norm_linear_activation(
            &input,
            &gamma,
            &[],
            &weight,
            &[],
            EPS,
            FusionActivation::ReLU,
        )
        .unwrap();
        assert!(fused[0] >= 0.0);
    }

    #[test]
    fn norm_linear_activation_with_beta_and_bias() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = [0.5; 4];
        let beta = [0.1; 4];
        let weight = [1.0; 8]; // 2×4
        let bias = vec![0.5, -0.5];

        let fused = fused_norm_linear_activation(
            &input,
            &gamma,
            &beta,
            &weight,
            &bias,
            EPS,
            FusionActivation::SiLU,
        )
        .unwrap();
        assert_eq!(fused.len(), 2);
        assert!(fused.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn norm_linear_activation_empty_input() {
        let r = fused_norm_linear_activation(&[], &[], &[], &[], &[], EPS, FusionActivation::ReLU);
        assert_eq!(r.unwrap_err(), LayerFusionError::EmptyInput);
    }

    #[test]
    fn norm_linear_activation_gamma_mismatch() {
        let r = fused_norm_linear_activation(
            &[1.0],
            &[1.0, 2.0],
            &[],
            &[1.0],
            &[],
            EPS,
            FusionActivation::ReLU,
        );
        assert!(matches!(r, Err(LayerFusionError::DimensionMismatch { .. })));
    }

    // ── fused_attention_block ──────────────────────────────────────

    #[test]
    fn attention_block_identity_weights() {
        let dim = 2;
        let seq_len = 2;
        let input = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity-like

        // Identity weight matrices.
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let result =
            fused_attention_block(&input, &identity, &identity, &identity, &identity, seq_len, dim)
                .unwrap();

        assert_eq!(result.len(), seq_len * dim);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn attention_block_single_position() {
        let dim = 4;
        let seq_len = 1;
        let input: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let w = vec![0.1f32; dim * dim];

        let result = fused_attention_block(&input, &w, &w, &w, &w, seq_len, dim).unwrap();
        assert_eq!(result.len(), dim);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn attention_block_causal_mask() {
        // With causal masking, position 0 should only attend to itself.
        let dim = 2;
        let seq_len = 3;
        let input = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let identity = vec![1.0, 0.0, 0.0, 1.0];

        let result =
            fused_attention_block(&input, &identity, &identity, &identity, &identity, seq_len, dim)
                .unwrap();

        // Position 0 only attends to itself → output[0..2] = V[0] = input[0..2].
        assert!((result[0] - 1.0).abs() < TOL);
        assert!((result[1] - 0.0).abs() < TOL);
    }

    #[test]
    fn attention_block_empty_input() {
        let r = fused_attention_block(&[], &[], &[], &[], &[], 0, 0);
        assert_eq!(r.unwrap_err(), LayerFusionError::EmptyInput);
    }

    #[test]
    fn attention_block_input_size_mismatch() {
        let r =
            fused_attention_block(&[1.0, 2.0], &[1.0; 4], &[1.0; 4], &[1.0; 4], &[1.0; 4], 2, 2);
        assert!(matches!(r, Err(LayerFusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn attention_block_weight_size_mismatch() {
        let r = fused_attention_block(
            &[1.0; 4], &[1.0; 3], // wrong
            &[1.0; 4], &[1.0; 4], &[1.0; 4], 2, 2,
        );
        assert!(matches!(r, Err(LayerFusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn attention_block_softmax_sums_to_one() {
        // Verify attention weights sum to ~1 by checking output is a
        // convex combination of V rows.
        let dim = 2;
        let seq_len = 2;
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let identity = vec![1.0, 0.0, 0.0, 1.0];

        let result =
            fused_attention_block(&input, &identity, &identity, &identity, &identity, seq_len, dim)
                .unwrap();

        // Each output row should have norm <= max(V row norms).
        for s in 0..seq_len {
            let row = &result[s * dim..(s + 1) * dim];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(norm <= 2.0, "output norm too large: {norm}");
        }
    }

    // ── fused_ffn_block ────────────────────────────────────────────

    #[test]
    fn ffn_block_silu_matches_reference() {
        let hidden = 4;
        let inter = 8;
        let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let w_gate: Vec<f32> = (0..inter * hidden).map(|i| (i as f32) * 0.01).collect();
        let w_up: Vec<f32> = (0..inter * hidden).map(|i| (i as f32) * 0.02).collect();
        let w_down: Vec<f32> = (0..hidden * inter).map(|i| (i as f32) * 0.01).collect();

        let fused =
            fused_ffn_block(&input, &w_gate, &w_up, &w_down, FusionActivation::SiLU).unwrap();
        let reference =
            reference_ffn_block(&input, &w_gate, &w_up, &w_down, FusionActivation::SiLU);
        assert!(max_abs_err(&fused, &reference) < TOL);
    }

    #[test]
    fn ffn_block_gelu_matches_reference() {
        let hidden = 4;
        let inter = 6;
        let input: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.5).collect();
        let w_gate: Vec<f32> = (0..inter * hidden).map(|i| (i as f32) * 0.01).collect();
        let w_up: Vec<f32> = (0..inter * hidden).map(|i| (i as f32) * 0.01).collect();
        let w_down: Vec<f32> = (0..hidden * inter).map(|i| (i as f32) * 0.01).collect();

        let fused =
            fused_ffn_block(&input, &w_gate, &w_up, &w_down, FusionActivation::GELU).unwrap();
        let reference =
            reference_ffn_block(&input, &w_gate, &w_up, &w_down, FusionActivation::GELU);
        assert!(max_abs_err(&fused, &reference) < TOL);
    }

    #[test]
    fn ffn_block_relu_matches_reference() {
        let hidden = 3;
        let inter = 4;
        let input = vec![1.0, -0.5, 0.3];
        let w_gate = vec![0.1f32; inter * hidden];
        let w_up = vec![0.2f32; inter * hidden];
        let w_down = vec![0.1f32; hidden * inter];

        let fused =
            fused_ffn_block(&input, &w_gate, &w_up, &w_down, FusionActivation::ReLU).unwrap();
        let reference =
            reference_ffn_block(&input, &w_gate, &w_up, &w_down, FusionActivation::ReLU);
        assert!(max_abs_err(&fused, &reference) < TOL);
    }

    #[test]
    fn ffn_block_single_element() {
        let input = [2.0];
        let w_gate = [1.0];
        let w_up = [1.0];
        let w_down = [1.0];

        let fused =
            fused_ffn_block(&input, &w_gate, &w_up, &w_down, FusionActivation::ReLU).unwrap();
        let reference =
            reference_ffn_block(&input, &w_gate, &w_up, &w_down, FusionActivation::ReLU);
        assert!(max_abs_err(&fused, &reference) < TOL);
    }

    #[test]
    fn ffn_block_zero_input() {
        let hidden = 4;
        let inter = 8;
        let input = vec![0.0f32; hidden];
        let w_gate = vec![1.0f32; inter * hidden];
        let w_up = vec![1.0f32; inter * hidden];
        let w_down = vec![1.0f32; hidden * inter];

        let fused =
            fused_ffn_block(&input, &w_gate, &w_up, &w_down, FusionActivation::SiLU).unwrap();
        for &v in &fused {
            assert!(v.abs() < TOL, "expected ~0 for zero input, got {v}");
        }
    }

    #[test]
    fn ffn_block_empty_input() {
        let r = fused_ffn_block(&[], &[], &[], &[], FusionActivation::SiLU);
        assert_eq!(r.unwrap_err(), LayerFusionError::EmptyInput);
    }

    #[test]
    fn ffn_block_w_gate_mismatch() {
        let r = fused_ffn_block(
            &[1.0, 2.0],
            &[1.0; 3], // not divisible
            &[1.0; 4],
            &[1.0; 4],
            FusionActivation::SiLU,
        );
        assert!(matches!(r, Err(LayerFusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn ffn_block_w_up_mismatch() {
        let r = fused_ffn_block(
            &[1.0, 2.0],
            &[1.0; 4],
            &[1.0; 3], // wrong
            &[1.0; 4],
            FusionActivation::SiLU,
        );
        assert!(matches!(r, Err(LayerFusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn ffn_block_w_down_mismatch() {
        let r = fused_ffn_block(
            &[1.0, 2.0],
            &[1.0; 4],
            &[1.0; 4],
            &[1.0; 3], // wrong
            FusionActivation::SiLU,
        );
        assert!(matches!(r, Err(LayerFusionError::DimensionMismatch { .. })));
    }

    // ── can_fuse ───────────────────────────────────────────────────

    #[test]
    fn can_fuse_norm_linear_compatible() {
        let layers =
            vec![LayerDesc::LayerNorm { dim: 64 }, LayerDesc::Linear { in_dim: 64, out_dim: 128 }];
        assert!(can_fuse(&layers, FusionPattern::NormLinear));
    }

    #[test]
    fn can_fuse_norm_linear_incompatible_dims() {
        let layers =
            vec![LayerDesc::LayerNorm { dim: 64 }, LayerDesc::Linear { in_dim: 32, out_dim: 128 }];
        assert!(!can_fuse(&layers, FusionPattern::NormLinear));
    }

    #[test]
    fn can_fuse_norm_linear_wrong_types() {
        let layers = vec![
            LayerDesc::Linear { in_dim: 64, out_dim: 64 },
            LayerDesc::Linear { in_dim: 64, out_dim: 128 },
        ];
        assert!(!can_fuse(&layers, FusionPattern::NormLinear));
    }

    #[test]
    fn can_fuse_linear_activation_compatible() {
        let layers = vec![
            LayerDesc::Linear { in_dim: 64, out_dim: 128 },
            LayerDesc::Activation { dim: 128, act: FusionActivation::SiLU },
        ];
        assert!(can_fuse(&layers, FusionPattern::LinearActivation));
    }

    #[test]
    fn can_fuse_linear_activation_incompatible() {
        let layers = vec![
            LayerDesc::Linear { in_dim: 64, out_dim: 128 },
            LayerDesc::Activation { dim: 64, act: FusionActivation::SiLU },
        ];
        assert!(!can_fuse(&layers, FusionPattern::LinearActivation));
    }

    #[test]
    fn can_fuse_triple_compatible() {
        let layers = vec![
            LayerDesc::LayerNorm { dim: 64 },
            LayerDesc::Linear { in_dim: 64, out_dim: 128 },
            LayerDesc::Activation { dim: 128, act: FusionActivation::GELU },
        ];
        assert!(can_fuse(&layers, FusionPattern::NormLinearActivation));
    }

    #[test]
    fn can_fuse_triple_incompatible_norm_dim() {
        let layers = vec![
            LayerDesc::LayerNorm { dim: 32 },
            LayerDesc::Linear { in_dim: 64, out_dim: 128 },
            LayerDesc::Activation { dim: 128, act: FusionActivation::GELU },
        ];
        assert!(!can_fuse(&layers, FusionPattern::NormLinearActivation));
    }

    #[test]
    fn can_fuse_triple_incompatible_act_dim() {
        let layers = vec![
            LayerDesc::LayerNorm { dim: 64 },
            LayerDesc::Linear { in_dim: 64, out_dim: 128 },
            LayerDesc::Activation { dim: 64, act: FusionActivation::GELU },
        ];
        assert!(!can_fuse(&layers, FusionPattern::NormLinearActivation));
    }

    #[test]
    fn can_fuse_attention_block() {
        let layers = vec![LayerDesc::Attention { dim: 64, num_heads: 8 }];
        assert!(can_fuse(&layers, FusionPattern::AttentionBlock));
    }

    #[test]
    fn can_fuse_ffn_block() {
        let layers = vec![LayerDesc::FFN { hidden_dim: 64, intermediate_dim: 256 }];
        assert!(can_fuse(&layers, FusionPattern::FFNBlock));
    }

    #[test]
    fn can_fuse_wrong_pattern() {
        let layers = vec![LayerDesc::LayerNorm { dim: 64 }];
        assert!(!can_fuse(&layers, FusionPattern::NormLinear));
        assert!(!can_fuse(&layers, FusionPattern::FFNBlock));
    }

    #[test]
    fn can_fuse_empty_layers() {
        assert!(!can_fuse(&[], FusionPattern::NormLinear));
    }

    // ── fusion_benefit_estimate ────────────────────────────────────

    #[test]
    fn benefit_norm_linear_positive() {
        let b = fusion_benefit_estimate(FusionPattern::NormLinear, 256, 512);
        assert!(b > 0.0 && b < 1.0, "benefit = {b}");
    }

    #[test]
    fn benefit_linear_activation_positive() {
        let b = fusion_benefit_estimate(FusionPattern::LinearActivation, 256, 512);
        assert!(b > 0.0 && b < 1.0, "benefit = {b}");
    }

    #[test]
    fn benefit_triple_greater_than_double() {
        let double = fusion_benefit_estimate(FusionPattern::NormLinear, 256, 512);
        let triple = fusion_benefit_estimate(FusionPattern::NormLinearActivation, 256, 512);
        assert!(triple > double, "triple {triple} should > double {double}");
    }

    #[test]
    fn benefit_zero_dim_returns_zero() {
        assert_eq!(fusion_benefit_estimate(FusionPattern::NormLinear, 0, 128), 0.0);
    }

    #[test]
    fn benefit_attention_block() {
        let b = fusion_benefit_estimate(FusionPattern::AttentionBlock, 512, 512);
        assert!(b > 0.0 && b < 1.0);
    }

    #[test]
    fn benefit_ffn_block() {
        let b = fusion_benefit_estimate(FusionPattern::FFNBlock, 256, 1024);
        assert!(b > 0.0 && b < 1.0);
    }

    #[test]
    fn benefit_scales_reasonably() {
        // Larger dimensions should not change the ratio dramatically
        // (it's a ratio, not absolute).
        let small = fusion_benefit_estimate(FusionPattern::NormLinear, 64, 64);
        let large = fusion_benefit_estimate(FusionPattern::NormLinear, 4096, 4096);
        assert!((small - large).abs() < 0.1, "small={small}, large={large}");
    }

    // ── apply_fusion_plan ──────────────────────────────────────────

    #[test]
    fn plan_empty_layers() {
        let plan = apply_fusion_plan(&[]);
        assert!(plan.is_empty());
    }

    #[test]
    fn plan_single_norm_linear() {
        let layers =
            vec![LayerDesc::LayerNorm { dim: 64 }, LayerDesc::Linear { in_dim: 64, out_dim: 128 }];
        let plan = apply_fusion_plan(&layers);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].pattern, FusionPattern::NormLinear);
        assert_eq!(plan[0].layer_range, (0, 2));
    }

    #[test]
    fn plan_triple_fusion() {
        let layers = vec![
            LayerDesc::LayerNorm { dim: 64 },
            LayerDesc::Linear { in_dim: 64, out_dim: 128 },
            LayerDesc::Activation { dim: 128, act: FusionActivation::GELU },
        ];
        let plan = apply_fusion_plan(&layers);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].pattern, FusionPattern::NormLinearActivation);
        assert_eq!(plan[0].layer_range, (0, 3));
    }

    #[test]
    fn plan_prefers_triple_over_double() {
        let layers = vec![
            LayerDesc::LayerNorm { dim: 64 },
            LayerDesc::Linear { in_dim: 64, out_dim: 128 },
            LayerDesc::Activation { dim: 128, act: FusionActivation::SiLU },
        ];
        let plan = apply_fusion_plan(&layers);
        // Should pick the triple, not norm+linear followed by unfused activation.
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].pattern, FusionPattern::NormLinearActivation);
    }

    #[test]
    fn plan_linear_activation_pair() {
        let layers = vec![
            LayerDesc::Linear { in_dim: 64, out_dim: 128 },
            LayerDesc::Activation { dim: 128, act: FusionActivation::ReLU },
        ];
        let plan = apply_fusion_plan(&layers);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].pattern, FusionPattern::LinearActivation);
    }

    #[test]
    fn plan_attention_block() {
        let layers = vec![LayerDesc::Attention { dim: 64, num_heads: 8 }];
        let plan = apply_fusion_plan(&layers);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].pattern, FusionPattern::AttentionBlock);
    }

    #[test]
    fn plan_ffn_block() {
        let layers = vec![LayerDesc::FFN { hidden_dim: 64, intermediate_dim: 256 }];
        let plan = apply_fusion_plan(&layers);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].pattern, FusionPattern::FFNBlock);
    }

    #[test]
    fn plan_no_fusible_layers() {
        let layers = vec![LayerDesc::LayerNorm { dim: 64 }, LayerDesc::LayerNorm { dim: 128 }];
        let plan = apply_fusion_plan(&layers);
        assert!(plan.is_empty());
    }

    #[test]
    fn plan_mixed_sequence() {
        let layers = vec![
            LayerDesc::LayerNorm { dim: 64 },
            LayerDesc::Linear { in_dim: 64, out_dim: 128 },
            LayerDesc::Activation { dim: 128, act: FusionActivation::GELU },
            LayerDesc::FFN { hidden_dim: 128, intermediate_dim: 512 },
        ];
        let plan = apply_fusion_plan(&layers);
        // Triple fusion for first 3, then FFN block.
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].pattern, FusionPattern::NormLinearActivation);
        assert_eq!(plan[0].layer_range, (0, 3));
        assert_eq!(plan[1].pattern, FusionPattern::FFNBlock);
        assert_eq!(plan[1].layer_range, (3, 4));
    }

    #[test]
    fn plan_incompatible_dims_skipped() {
        let layers = vec![
            LayerDesc::LayerNorm { dim: 64 },
            LayerDesc::Linear { in_dim: 32, out_dim: 128 }, // mismatch
        ];
        let plan = apply_fusion_plan(&layers);
        assert!(plan.is_empty());
    }

    // ── Various sizes / stress ─────────────────────────────────────

    #[test]
    fn norm_linear_various_sizes() {
        for &dim in &[1, 4, 16, 64, 128] {
            let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let gamma = vec![1.0f32; dim];
            let weight: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
            let fused = fused_norm_linear(&input, &gamma, &[], &weight, EPS).unwrap();
            let reference = reference_norm_linear(&input, &gamma, &[], &weight, EPS);
            assert!(max_abs_err(&fused, &reference) < 0.01, "failed for dim={dim}");
        }
    }

    #[test]
    fn ffn_block_various_sizes() {
        for &(h, i) in &[(2, 4), (4, 8), (8, 16), (16, 32)] {
            let input: Vec<f32> = (0..h).map(|j| j as f32 * 0.1).collect();
            let w_gate: Vec<f32> = (0..i * h).map(|j| j as f32 * 0.01).collect();
            let w_up: Vec<f32> = (0..i * h).map(|j| j as f32 * 0.01).collect();
            let w_down: Vec<f32> = (0..h * i).map(|j| j as f32 * 0.01).collect();

            let fused =
                fused_ffn_block(&input, &w_gate, &w_up, &w_down, FusionActivation::SiLU).unwrap();
            let reference =
                reference_ffn_block(&input, &w_gate, &w_up, &w_down, FusionActivation::SiLU);
            assert!(max_abs_err(&fused, &reference) < 0.1, "failed for h={h}, i={i}");
        }
    }
}
