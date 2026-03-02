//! Complete transformer layer executor for OpenCL (Intel Arc A770) inference.
//!
//! Chains attention, FFN, normalization, and residual operations into a full
//! forward pass. CPU reference implementations are provided for validation;
//! OpenCL-accelerated paths will be layered on top.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Activation function used in the FFN sublayer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    SiLU,
    GELU,
    ReLU,
    SwiGLU,
}

/// Core transformer geometry.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub norm_eps: f32,
    pub activation: ActivationType,
    pub num_layers: usize,
}

/// Input to a single transformer layer.
#[derive(Debug, Clone)]
pub struct LayerInput {
    pub hidden_states: Vec<f32>,
    pub attention_mask: Option<Vec<f32>>,
    pub position_ids: Vec<usize>,
    pub past_key: Option<Vec<f32>>,
    pub past_value: Option<Vec<f32>>,
}

/// Output from a single transformer layer.
#[derive(Debug, Clone)]
pub struct LayerOutput {
    pub hidden_states: Vec<f32>,
    pub present_key: Vec<f32>,
    pub present_value: Vec<f32>,
    pub attention_weights: Option<Vec<f32>>,
}

/// Full set of weights for one transformer layer.
#[derive(Debug, Clone)]
pub struct LayerWeightSet {
    pub q_weight: Vec<f32>,
    pub k_weight: Vec<f32>,
    pub v_weight: Vec<f32>,
    pub o_weight: Vec<f32>,
    pub gate_weight: Vec<f32>,
    pub up_weight: Vec<f32>,
    pub down_weight: Vec<f32>,
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
}

/// Runtime knobs for the executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub transformer: TransformerConfig,
    pub use_flash_attention: bool,
    pub use_kv_cache: bool,
    pub compute_attention_weights: bool,
}

/// Cumulative statistics gathered during execution.
#[derive(Debug, Clone, Default)]
pub struct ExecutorStats {
    pub layers_executed: u64,
    pub total_time_us: u64,
    pub attention_time_us: u64,
    pub ffn_time_us: u64,
    pub norm_time_us: u64,
}

/// The layer executor.
#[derive(Debug)]
pub struct LayerExecutor {
    pub config: ExecutorConfig,
    pub stats: ExecutorStats,
}

/// Errors that can occur during layer execution.
#[derive(Debug, Clone)]
pub enum ExecutorError {
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    WeightsNotLoaded,
    NumericalError(String),
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected:?}, got {got:?}")
            }
            Self::WeightsNotLoaded => write!(f, "weights not loaded"),
            Self::NumericalError(msg) => write!(f, "numerical error: {msg}"),
        }
    }
}

impl std::error::Error for ExecutorError {}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Create a new [`LayerExecutor`] with the given configuration.
pub fn create_layer_executor(config: ExecutorConfig) -> LayerExecutor {
    LayerExecutor { config, stats: ExecutorStats::default() }
}

// ---------------------------------------------------------------------------
// Primitive CPU helpers
// ---------------------------------------------------------------------------

/// RMS-norm: `y_i = x_i / rms(x) * weight_i` where `rms(x) = sqrt(mean(x^2) + eps)`.
pub fn cpu_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    assert_eq!(n, weight.len(), "rmsnorm: x and weight length mismatch");
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rms = (mean_sq + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(xi, wi)| xi / rms * wi).collect()
}

/// Element-wise residual addition.
pub fn cpu_residual_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "residual_add: length mismatch");
    a.iter().zip(b.iter()).map(|(ai, bi)| ai + bi).collect()
}

/// Apply an activation function element-wise.
pub fn cpu_apply_activation(x: &[f32], activation: &ActivationType) -> Vec<f32> {
    match activation {
        ActivationType::SiLU => x.iter().map(|&v| v * sigmoid(v)).collect(),
        ActivationType::GELU => x.iter().map(|&v| gelu(v)).collect(),
        ActivationType::ReLU => x.iter().map(|&v| v.max(0.0)).collect(),
        ActivationType::SwiGLU => {
            // SwiGLU splits the input in half: silu(first_half) * second_half
            let half = x.len() / 2;
            let gate = &x[..half];
            let up = &x[half..];
            gate.iter()
                .zip(up.iter())
                .map(|(&g, &u)| (g * sigmoid(g)) * u)
                .collect()
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn gelu(x: f32) -> f32 {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
}

/// Dense linear projection: `output = input * weight^T`.
///
/// `weight` is stored row-major with shape `[out_features, in_features]`.
pub fn cpu_linear(
    input: &[f32],
    weight: &[f32],
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    assert_eq!(
        weight.len(),
        in_features * out_features,
        "linear: weight size mismatch"
    );
    let seq_len = input.len() / in_features;
    let mut output = vec![0.0f32; seq_len * out_features];
    for s in 0..seq_len {
        let inp = &input[s * in_features..(s + 1) * in_features];
        for o in 0..out_features {
            let w_row = &weight[o * in_features..(o + 1) * in_features];
            output[s * out_features + o] =
                inp.iter().zip(w_row.iter()).map(|(a, b)| a * b).sum();
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Sublayers
// ---------------------------------------------------------------------------

/// Multi-head self-attention sublayer (CPU reference).
///
/// Returns `(output, new_keys, new_values)`.
pub fn cpu_attention_sublayer(
    hidden: &[f32],
    weights: &LayerWeightSet,
    config: &TransformerConfig,
    mask: Option<&[f32]>,
    seq_len: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let h = config.hidden_size;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;

    // Project Q, K, V
    let q = cpu_linear(hidden, &weights.q_weight, h, h);
    let k = cpu_linear(hidden, &weights.k_weight, h, h);
    let v = cpu_linear(hidden, &weights.v_weight, h, h);

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Per-head scaled dot-product attention
    let mut attn_out = vec![0.0f32; seq_len * h];
    for head in 0..num_heads {
        for qi in 0..seq_len {
            // Compute attention scores for this query position
            let mut scores = Vec::with_capacity(seq_len);
            for ki in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = qi * h + head * head_dim + d;
                    let k_idx = ki * h + head * head_dim + d;
                    dot += q[q_idx] * k[k_idx];
                }
                let mut s = dot * scale;
                if let Some(m) = mask {
                    s += m[qi * seq_len + ki];
                }
                scores.push(s);
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

            // Weighted sum of values
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (ki, prob) in probs.iter().enumerate() {
                    let v_idx = ki * h + head * head_dim + d;
                    acc += prob * v[v_idx];
                }
                attn_out[qi * h + head * head_dim + d] = acc;
            }
        }
    }

    // Output projection
    let output = cpu_linear(&attn_out, &weights.o_weight, h, h);
    (output, k, v)
}

/// Feed-forward network sublayer (CPU reference).
pub fn cpu_ffn_sublayer(
    hidden: &[f32],
    weights: &LayerWeightSet,
    config: &TransformerConfig,
) -> Vec<f32> {
    let h = config.hidden_size;
    let inter = config.intermediate_size;

    match config.activation {
        ActivationType::SwiGLU => {
            let gate = cpu_linear(hidden, &weights.gate_weight, h, inter);
            let up = cpu_linear(hidden, &weights.up_weight, h, inter);
            // SiLU(gate) * up
            let activated: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(&g, &u)| (g * sigmoid(g)) * u)
                .collect();
            cpu_linear(&activated, &weights.down_weight, inter, h)
        }
        _ => {
            let up = cpu_linear(hidden, &weights.up_weight, h, inter);
            let activated = cpu_apply_activation(&up, &config.activation);
            cpu_linear(&activated, &weights.down_weight, inter, h)
        }
    }
}

// ---------------------------------------------------------------------------
// Full layer execution
// ---------------------------------------------------------------------------

/// Execute a single transformer layer on CPU.
pub fn cpu_execute_layer(
    executor: &mut LayerExecutor,
    input: &LayerInput,
    weights: &LayerWeightSet,
) -> Result<LayerOutput, ExecutorError> {
    let cfg = &executor.config.transformer;
    let h = cfg.hidden_size;
    let seq_len = input.hidden_states.len() / h;

    if !input.hidden_states.len().is_multiple_of(h) {
        return Err(ExecutorError::ShapeMismatch {
            expected: vec![seq_len, h],
            got: vec![input.hidden_states.len()],
        });
    }
    if weights.q_weight.is_empty() {
        return Err(ExecutorError::WeightsNotLoaded);
    }

    let layer_start = Instant::now();

    // 1. Input RMSNorm
    let norm_start = Instant::now();
    let mut normed = Vec::with_capacity(seq_len * h);
    for s in 0..seq_len {
        let slice = &input.hidden_states[s * h..(s + 1) * h];
        normed.extend(cpu_rmsnorm(slice, &weights.input_norm, cfg.norm_eps));
    }
    let norm_elapsed = norm_start.elapsed().as_micros() as u64;

    // 2. Attention sublayer
    let attn_start = Instant::now();
    let mask = input.attention_mask.as_deref();
    let (attn_out, new_keys, new_values) =
        cpu_attention_sublayer(&normed, weights, cfg, mask, seq_len);
    let attn_elapsed = attn_start.elapsed().as_micros() as u64;

    // 3. Residual
    let hidden_after_attn = cpu_residual_add(&input.hidden_states, &attn_out);

    // 4. Post-attention RMSNorm
    let norm2_start = Instant::now();
    let mut normed2 = Vec::with_capacity(seq_len * h);
    for s in 0..seq_len {
        let slice = &hidden_after_attn[s * h..(s + 1) * h];
        normed2.extend(cpu_rmsnorm(slice, &weights.post_norm, cfg.norm_eps));
    }
    let norm2_elapsed = norm2_start.elapsed().as_micros() as u64;

    // 5. FFN sublayer
    let ffn_start = Instant::now();
    let ffn_out = cpu_ffn_sublayer(&normed2, weights, cfg);
    let ffn_elapsed = ffn_start.elapsed().as_micros() as u64;

    // 6. Residual
    let final_hidden = cpu_residual_add(&hidden_after_attn, &ffn_out);

    // Check for NaN/Inf
    if final_hidden.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return Err(ExecutorError::NumericalError(
            "NaN or Inf in layer output".to_string(),
        ));
    }

    // Update stats
    let total_elapsed = layer_start.elapsed().as_micros() as u64;
    executor.stats.layers_executed += 1;
    executor.stats.total_time_us += total_elapsed;
    executor.stats.attention_time_us += attn_elapsed;
    executor.stats.ffn_time_us += ffn_elapsed;
    executor.stats.norm_time_us += norm_elapsed + norm2_elapsed;

    Ok(LayerOutput {
        hidden_states: final_hidden,
        present_key: new_keys,
        present_value: new_values,
        attention_weights: None,
    })
}

/// Execute multiple transformer layers sequentially.
pub fn cpu_execute_multi_layer(
    executor: &mut LayerExecutor,
    input: LayerInput,
    all_weights: &[LayerWeightSet],
) -> Result<Vec<f32>, ExecutorError> {
    let mut current = input;
    for weights in all_weights {
        let output = cpu_execute_layer(executor, &current, weights)?;
        current = LayerInput {
            hidden_states: output.hidden_states,
            attention_mask: current.attention_mask.clone(),
            position_ids: current.position_ids.clone(),
            past_key: Some(output.present_key),
            past_value: Some(output.present_value),
        };
    }
    Ok(current.hidden_states)
}

/// Retrieve a snapshot of the executor statistics.
pub fn cpu_get_stats(executor: &LayerExecutor) -> ExecutorStats {
    executor.stats.clone()
}

/// Human-readable summary of executor statistics.
pub fn format_executor_stats(stats: &ExecutorStats) -> String {
    format!(
        "layers={} total={}µs attn={}µs ffn={}µs norm={}µs",
        stats.layers_executed,
        stats.total_time_us,
        stats.attention_time_us,
        stats.ffn_time_us,
        stats.norm_time_us,
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers ----------------------------------------------------------

    fn tiny_config() -> TransformerConfig {
        TransformerConfig {
            hidden_size: 8,
            num_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            norm_eps: 1e-5,
            activation: ActivationType::SiLU,
            num_layers: 1,
        }
    }

    fn tiny_executor_config() -> ExecutorConfig {
        ExecutorConfig {
            transformer: tiny_config(),
            use_flash_attention: false,
            use_kv_cache: true,
            compute_attention_weights: false,
        }
    }

    /// Build deterministic weights for the tiny config.
    fn tiny_weights() -> LayerWeightSet {
        let h = 8;
        let inter = 16;
        let make = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect()
        };
        LayerWeightSet {
            q_weight: make(h, h),
            k_weight: make(h, h),
            v_weight: make(h, h),
            o_weight: make(h, h),
            gate_weight: make(inter, h),
            up_weight: make(inter, h),
            down_weight: make(h, inter),
            input_norm: vec![1.0; h],
            post_norm: vec![1.0; h],
        }
    }

    fn tiny_input(seq_len: usize) -> LayerInput {
        let h = 8;
        LayerInput {
            hidden_states: (0..seq_len * h).map(|i| (i as f32) * 0.01).collect(),
            attention_mask: None,
            position_ids: (0..seq_len).collect(),
            past_key: None,
            past_value: None,
        }
    }

    // ---- 1. create executor -----------------------------------------------

    #[test]
    fn test_create_executor() {
        let exec = create_layer_executor(tiny_executor_config());
        assert_eq!(exec.stats.layers_executed, 0);
        assert_eq!(exec.config.transformer.hidden_size, 8);
    }

    #[test]
    fn test_create_executor_default_stats() {
        let exec = create_layer_executor(tiny_executor_config());
        assert_eq!(exec.stats.total_time_us, 0);
        assert_eq!(exec.stats.attention_time_us, 0);
        assert_eq!(exec.stats.ffn_time_us, 0);
        assert_eq!(exec.stats.norm_time_us, 0);
    }

    // ---- 2. RMSNorm -------------------------------------------------------

    #[test]
    fn test_rmsnorm_unit_norm() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let w = vec![1.0; 4];
        let y = cpu_rmsnorm(&x, &w, 1e-5);
        // rms ≈ 1.0, so output ≈ input
        for v in &y {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_rmsnorm_zero_input() {
        let x = vec![0.0; 4];
        let w = vec![1.0; 4];
        let y = cpu_rmsnorm(&x, &w, 1e-5);
        for v in &y {
            assert!(v.abs() < 1e-2, "expected ~0, got {v}");
        }
    }

    #[test]
    fn test_rmsnorm_with_nontrivial_weight() {
        let x = vec![2.0, 2.0, 2.0, 2.0];
        let w = vec![0.5, 0.5, 0.5, 0.5];
        let y = cpu_rmsnorm(&x, &w, 1e-5);
        // rms ≈ 2.0, normalised ≈ 1.0, then *0.5 ≈ 0.5
        for v in &y {
            assert!((v - 0.5).abs() < 1e-3, "expected ~0.5, got {v}");
        }
    }

    // ---- 3. Residual add ---------------------------------------------------

    #[test]
    fn test_residual_add_correct_sum() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = cpu_residual_add(&a, &b);
        assert_eq!(c, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_residual_add_identity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0; 3];
        assert_eq!(cpu_residual_add(&a, &b), a);
    }

    #[test]
    fn test_residual_preserves_dimension() {
        let a = vec![0.0; 32];
        let b = vec![1.0; 32];
        let c = cpu_residual_add(&a, &b);
        assert_eq!(c.len(), 32);
    }

    // ---- 4. Activations ----------------------------------------------------

    #[test]
    fn test_activation_silu() {
        let x = vec![0.0, 1.0, -1.0];
        let y = cpu_apply_activation(&x, &ActivationType::SiLU);
        assert!((y[0]).abs() < 1e-5); // silu(0) = 0
        assert!(y[1] > 0.5); // silu(1) ≈ 0.731
        assert!(y[2] < 0.0); // silu(-1) ≈ -0.269
    }

    #[test]
    fn test_activation_gelu() {
        let x = vec![0.0, 1.0, -1.0];
        let y = cpu_apply_activation(&x, &ActivationType::GELU);
        assert!((y[0]).abs() < 1e-5); // gelu(0) = 0
        assert!(y[1] > 0.8); // gelu(1) ≈ 0.841
        assert!(y[2] < 0.0); // gelu(-1) ≈ -0.159
    }

    #[test]
    fn test_activation_relu() {
        let x = vec![-2.0, 0.0, 3.0];
        let y = cpu_apply_activation(&x, &ActivationType::ReLU);
        assert_eq!(y, vec![0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_activation_swiglu() {
        let x = vec![1.0, 2.0, 0.5, 0.5]; // gate=[1,2], up=[0.5,0.5]
        let y = cpu_apply_activation(&x, &ActivationType::SwiGLU);
        assert_eq!(y.len(), 2);
        assert!(y[0] > 0.0);
        assert!(y[1] > 0.0);
    }

    #[test]
    fn test_activation_relu_zeros() {
        let x = vec![-1.0, -0.5, 0.0];
        let y = cpu_apply_activation(&x, &ActivationType::ReLU);
        assert!(y.iter().all(|v| *v == 0.0));
    }

    // ---- 5. Linear ---------------------------------------------------------

    #[test]
    fn test_linear_correct_output_shape() {
        let input = vec![1.0; 12]; // 3 tokens, 4 features
        let weight = vec![0.1; 4 * 6]; // 6 outputs, 4 inputs
        let out = cpu_linear(&input, &weight, 4, 6);
        assert_eq!(out.len(), 3 * 6);
    }

    #[test]
    fn test_linear_identity_like() {
        // 1×2 input, 2×2 identity-ish weight
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let out = cpu_linear(&input, &weight, 2, 2);
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_zero_weight() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![0.0; 3 * 4];
        let out = cpu_linear(&input, &weight, 3, 4);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    // ---- 6. Attention sublayer ---------------------------------------------

    #[test]
    fn test_attention_sublayer_output_shape() {
        let cfg = tiny_config();
        let w = tiny_weights();
        let input = vec![0.1f32; 2 * 8]; // seq_len=2
        let (out, keys, vals) = cpu_attention_sublayer(&input, &w, &cfg, None, 2);
        assert_eq!(out.len(), 2 * 8);
        assert_eq!(keys.len(), 2 * 8);
        assert_eq!(vals.len(), 2 * 8);
    }

    #[test]
    fn test_attention_sublayer_single_token() {
        let cfg = tiny_config();
        let w = tiny_weights();
        let input = vec![0.5f32; 8]; // seq_len=1
        let (out, _, _) = cpu_attention_sublayer(&input, &w, &cfg, None, 1);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_attention_with_mask() {
        let cfg = tiny_config();
        let w = tiny_weights();
        let seq_len = 2;
        let input = vec![0.1f32; seq_len * 8];
        // Causal mask: block future positions
        let mask = vec![0.0, -1e9, 0.0, 0.0];
        let (out, _, _) = cpu_attention_sublayer(&input, &w, &cfg, Some(&mask), seq_len);
        assert_eq!(out.len(), seq_len * 8);
    }

    // ---- 7. FFN sublayer ---------------------------------------------------

    #[test]
    fn test_ffn_sublayer_output_shape() {
        let cfg = tiny_config();
        let w = tiny_weights();
        let input = vec![0.1f32; 2 * 8]; // seq_len=2
        let out = cpu_ffn_sublayer(&input, &w, &cfg);
        assert_eq!(out.len(), 2 * 8);
    }

    #[test]
    fn test_ffn_sublayer_single_token() {
        let cfg = tiny_config();
        let w = tiny_weights();
        let input = vec![0.1f32; 8];
        let out = cpu_ffn_sublayer(&input, &w, &cfg);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_ffn_with_gelu() {
        let mut cfg = tiny_config();
        cfg.activation = ActivationType::GELU;
        let w = tiny_weights();
        let input = vec![0.1f32; 8];
        let out = cpu_ffn_sublayer(&input, &w, &cfg);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_ffn_with_relu() {
        let mut cfg = tiny_config();
        cfg.activation = ActivationType::ReLU;
        let w = tiny_weights();
        let input = vec![0.1f32; 8];
        let out = cpu_ffn_sublayer(&input, &w, &cfg);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_ffn_with_swiglu() {
        let mut cfg = tiny_config();
        cfg.activation = ActivationType::SwiGLU;
        let w = tiny_weights();
        let input = vec![0.1f32; 8];
        let out = cpu_ffn_sublayer(&input, &w, &cfg);
        assert_eq!(out.len(), 8);
    }

    // ---- 8. Full layer execution -------------------------------------------

    #[test]
    fn test_execute_layer_correct_shapes() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let input = tiny_input(2);
        let w = tiny_weights();
        let out = cpu_execute_layer(&mut exec, &input, &w).unwrap();
        assert_eq!(out.hidden_states.len(), 2 * 8);
        assert_eq!(out.present_key.len(), 2 * 8);
        assert_eq!(out.present_value.len(), 2 * 8);
    }

    #[test]
    fn test_execute_layer_output_shape_matches_input() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let input = tiny_input(3);
        let w = tiny_weights();
        let out = cpu_execute_layer(&mut exec, &input, &w).unwrap();
        assert_eq!(out.hidden_states.len(), input.hidden_states.len());
    }

    #[test]
    fn test_execute_layer_updates_stats() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let input = tiny_input(1);
        let w = tiny_weights();
        cpu_execute_layer(&mut exec, &input, &w).unwrap();
        assert_eq!(exec.stats.layers_executed, 1);
    }

    #[test]
    fn test_execute_layer_stats_accumulate() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let input = tiny_input(1);
        let w = tiny_weights();
        cpu_execute_layer(&mut exec, &input, &w).unwrap();
        cpu_execute_layer(&mut exec, &input, &w).unwrap();
        assert_eq!(exec.stats.layers_executed, 2);
    }

    #[test]
    fn test_execute_layer_shape_mismatch() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let mut input = tiny_input(1);
        input.hidden_states.push(0.0); // break alignment
        let w = tiny_weights();
        let err = cpu_execute_layer(&mut exec, &input, &w).unwrap_err();
        assert!(matches!(err, ExecutorError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_execute_layer_weights_not_loaded() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let input = tiny_input(1);
        let mut w = tiny_weights();
        w.q_weight.clear();
        let err = cpu_execute_layer(&mut exec, &input, &w).unwrap_err();
        assert!(matches!(err, ExecutorError::WeightsNotLoaded));
    }

    // ---- 9. Multi-layer execution ------------------------------------------

    #[test]
    fn test_multi_layer_output_shape() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let input = tiny_input(2);
        let weights = vec![tiny_weights(), tiny_weights()];
        let out = cpu_execute_multi_layer(&mut exec, input, &weights).unwrap();
        assert_eq!(out.len(), 2 * 8);
    }

    #[test]
    fn test_multi_layer_stats() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let input = tiny_input(1);
        let weights = vec![tiny_weights(), tiny_weights(), tiny_weights()];
        cpu_execute_multi_layer(&mut exec, input, &weights).unwrap();
        assert_eq!(exec.stats.layers_executed, 3);
    }

    #[test]
    fn test_multi_layer_single_layer() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let input = tiny_input(1);
        let weights = vec![tiny_weights()];
        let out = cpu_execute_multi_layer(&mut exec, input, &weights).unwrap();
        assert_eq!(out.len(), 8);
    }

    // ---- 10. KV cache ------------------------------------------------------

    #[test]
    fn test_kv_cache_returns_present() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let input = tiny_input(2);
        let w = tiny_weights();
        let out = cpu_execute_layer(&mut exec, &input, &w).unwrap();
        assert!(!out.present_key.is_empty());
        assert!(!out.present_value.is_empty());
    }

    #[test]
    fn test_kv_cache_present_key_shape() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let input = tiny_input(3);
        let w = tiny_weights();
        let out = cpu_execute_layer(&mut exec, &input, &w).unwrap();
        // Keys are full projections: seq_len * hidden_size
        assert_eq!(out.present_key.len(), 3 * 8);
    }

    // ---- 11. Edge cases ----------------------------------------------------

    #[test]
    fn test_edge_hidden4_heads1() {
        let cfg = ExecutorConfig {
            transformer: TransformerConfig {
                hidden_size: 4,
                num_heads: 1,
                head_dim: 4,
                intermediate_size: 8,
                norm_eps: 1e-5,
                activation: ActivationType::SiLU,
                num_layers: 1,
            },
            use_flash_attention: false,
            use_kv_cache: true,
            compute_attention_weights: false,
        };
        let mut exec = create_layer_executor(cfg);
        let h = 4;
        let input = LayerInput {
            hidden_states: vec![0.1; h],
            attention_mask: None,
            position_ids: vec![0],
            past_key: None,
            past_value: None,
        };
        let w = LayerWeightSet {
            q_weight: vec![0.1; h * h],
            k_weight: vec![0.1; h * h],
            v_weight: vec![0.1; h * h],
            o_weight: vec![0.1; h * h],
            gate_weight: vec![0.1; 8 * h],
            up_weight: vec![0.1; 8 * h],
            down_weight: vec![0.1; h * 8],
            input_norm: vec![1.0; h],
            post_norm: vec![1.0; h],
        };
        let out = cpu_execute_layer(&mut exec, &input, &w).unwrap();
        assert_eq!(out.hidden_states.len(), h);
    }

    #[test]
    fn test_edge_single_token_single_layer() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let input = tiny_input(1);
        let w = tiny_weights();
        let out = cpu_execute_layer(&mut exec, &input, &w).unwrap();
        assert_eq!(out.hidden_states.len(), 8);
    }

    // ---- 12. Properties ----------------------------------------------------

    #[test]
    fn test_property_attention_output_bounded() {
        let cfg = tiny_config();
        let w = tiny_weights();
        let input = vec![0.1f32; 2 * 8];
        let (out, _, _) = cpu_attention_sublayer(&input, &w, &cfg, None, 2);
        // Attention output should be finite and reasonably bounded
        for v in &out {
            assert!(v.is_finite(), "attention output must be finite");
            assert!(v.abs() < 1e6, "attention output unexpectedly large: {v}");
        }
    }

    #[test]
    fn test_property_ffn_output_finite() {
        let cfg = tiny_config();
        let w = tiny_weights();
        let input = vec![0.1f32; 8];
        let out = cpu_ffn_sublayer(&input, &w, &cfg);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_property_layer_output_finite() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let input = tiny_input(2);
        let w = tiny_weights();
        let out = cpu_execute_layer(&mut exec, &input, &w).unwrap();
        assert!(out.hidden_states.iter().all(|v| v.is_finite()));
    }

    // ---- 13. Stats & formatting --------------------------------------------

    #[test]
    fn test_get_stats_snapshot() {
        let mut exec = create_layer_executor(tiny_executor_config());
        let input = tiny_input(1);
        let w = tiny_weights();
        cpu_execute_layer(&mut exec, &input, &w).unwrap();
        let stats = cpu_get_stats(&exec);
        assert_eq!(stats.layers_executed, 1);
    }

    #[test]
    fn test_format_stats() {
        let stats = ExecutorStats {
            layers_executed: 2,
            total_time_us: 1000,
            attention_time_us: 400,
            ffn_time_us: 500,
            norm_time_us: 100,
        };
        let s = format_executor_stats(&stats);
        assert!(s.contains("layers=2"));
        assert!(s.contains("total=1000"));
    }

    // ---- 14. Error display -------------------------------------------------

    #[test]
    fn test_error_display_shape_mismatch() {
        let e = ExecutorError::ShapeMismatch {
            expected: vec![2, 8],
            got: vec![17],
        };
        let msg = format!("{e}");
        assert!(msg.contains("shape mismatch"));
    }

    #[test]
    fn test_error_display_numerical() {
        let e = ExecutorError::NumericalError("NaN detected".into());
        assert!(format!("{e}").contains("NaN detected"));
    }

    #[test]
    fn test_error_display_weights_not_loaded() {
        let e = ExecutorError::WeightsNotLoaded;
        assert!(format!("{e}").contains("weights not loaded"));
    }

    // ---- 15. Activation type enum ------------------------------------------

    #[test]
    fn test_activation_type_clone_eq() {
        let a = ActivationType::SiLU;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_activation_type_debug() {
        let s = format!("{:?}", ActivationType::GELU);
        assert!(s.contains("GELU"));
    }
}
