//! Transformer layer composition for Intel A770 BitNet inference.
//!
//! Composes individual OpenCL kernel modules (attention, FFN, normalization)
//! into complete transformer layer abstractions for 1-bit BitNet models.
//!
//! # Architecture
//!
//! Each transformer block follows the pre-norm pattern:
//!
//! ```text
//!   normed  = rms_norm(x, attn_norm_weight, eps)
//!   attn_out = multi_head_attention(normed, kv_cache)
//!   x        = x + attn_out          // residual
//!   normed  = rms_norm(x, ffn_norm_weight, eps)
//!   ffn_out = down_proj(silu(gate_proj(normed)) * up_proj(normed))
//!   x        = x + ffn_out           // residual
//! ```
//!
//! # CPU reference
//!
//! All public functions have pure-CPU scalar implementations so results
//! are deterministic and easy to validate against the OpenCL GPU path.

use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a single transformer layer.
#[derive(Debug, Clone)]
pub struct LayerConfig {
    /// Model hidden dimension (e.g. 2048).
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// FFN intermediate dimension.
    pub intermediate_dim: usize,
    /// Epsilon for RMS normalization.
    pub epsilon: f32,
    /// Whether to use flash-attention style fused kernel.
    pub use_flash_attention: bool,
    /// Whether to use fused ops (norm + projection in one kernel).
    pub use_fused_ops: bool,
}

impl LayerConfig {
    /// Validate that the configuration is internally consistent.
    pub fn validate(&self) -> Result<(), LayerComposeError> {
        if self.hidden_dim == 0 {
            return Err(LayerComposeError::InvalidConfig(
                "hidden_dim must be non-zero".into(),
            ));
        }
        if self.num_heads == 0 {
            return Err(LayerComposeError::InvalidConfig(
                "num_heads must be non-zero".into(),
            ));
        }
        if self.head_dim == 0 {
            return Err(LayerComposeError::InvalidConfig(
                "head_dim must be non-zero".into(),
            ));
        }
        if self.intermediate_dim == 0 {
            return Err(LayerComposeError::InvalidConfig(
                "intermediate_dim must be non-zero".into(),
            ));
        }
        if self.epsilon <= 0.0 {
            return Err(LayerComposeError::InvalidConfig(
                "epsilon must be positive".into(),
            ));
        }
        if self.num_heads * self.head_dim != self.hidden_dim {
            return Err(LayerComposeError::DimensionMismatch {
                expected: self.hidden_dim,
                got: self.num_heads * self.head_dim,
                context: "num_heads * head_dim must equal hidden_dim".into(),
            });
        }
        Ok(())
    }
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 2048,
            num_heads: 32,
            head_dim: 64,
            intermediate_dim: 5632,
            epsilon: 1e-5,
            use_flash_attention: false,
            use_fused_ops: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / output types
// ---------------------------------------------------------------------------

/// Input to a transformer layer.
pub struct LayerInput<'a> {
    /// Hidden states `[seq_len * hidden_dim]`.
    pub hidden_states: &'a [f32],
    /// Optional attention mask `[seq_len * kv_len]`.
    pub attention_mask: Option<&'a [f32]>,
    /// Position IDs `[seq_len]`.
    pub position_ids: &'a [u32],
    /// Optional KV cache from previous steps.
    pub kv_cache: Option<KVSlice<'a>>,
}

/// A borrowed slice into an existing KV cache.
pub struct KVSlice<'a> {
    /// Cached key vectors `[cache_len * hidden_dim]`.
    pub key_cache: &'a [f32],
    /// Cached value vectors `[cache_len * hidden_dim]`.
    pub value_cache: &'a [f32],
    /// Number of positions currently cached.
    pub cache_len: usize,
}

/// Output from a transformer layer.
#[derive(Debug, Clone)]
pub struct LayerOutput {
    /// Output hidden states `[seq_len * hidden_dim]`.
    pub hidden_states: Vec<f32>,
    /// New key vectors produced by this layer `[seq_len * hidden_dim]`.
    pub new_key: Vec<f32>,
    /// New value vectors produced by this layer `[seq_len * hidden_dim]`.
    pub new_value: Vec<f32>,
    /// Optional attention weights `[num_heads * seq_len * kv_len]`.
    pub attention_weights: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// All weight tensors for one transformer block.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Layer configuration.
    pub config: LayerConfig,
    /// Attention RMS-norm weight `[hidden_dim]`.
    pub attention_norm_weight: Vec<f32>,
    /// FFN RMS-norm weight `[hidden_dim]`.
    pub ffn_norm_weight: Vec<f32>,
    /// Query projection `[hidden_dim * hidden_dim]`.
    pub q_proj: Vec<f32>,
    /// Key projection `[hidden_dim * hidden_dim]`.
    pub k_proj: Vec<f32>,
    /// Value projection `[hidden_dim * hidden_dim]`.
    pub v_proj: Vec<f32>,
    /// Output projection `[hidden_dim * hidden_dim]`.
    pub o_proj: Vec<f32>,
    /// Gate projection `[hidden_dim * intermediate_dim]`.
    pub gate_proj: Vec<f32>,
    /// Up projection `[hidden_dim * intermediate_dim]`.
    pub up_proj: Vec<f32>,
    /// Down projection `[intermediate_dim * hidden_dim]`.
    pub down_proj: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by layer composition operations.
#[derive(Debug, Clone)]
pub enum LayerComposeError {
    /// Tensor dimensions do not match expectations.
    DimensionMismatch {
        expected: usize,
        got: usize,
        context: String,
    },
    /// Layer configuration is invalid.
    InvalidConfig(String),
    /// A compute operation failed.
    ComputeError(String),
}

impl fmt::Display for LayerComposeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got, context } => {
                write!(
                    f,
                    "dimension mismatch: expected {expected}, got {got} ({context})"
                )
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::ComputeError(msg) => write!(f, "compute error: {msg}"),
        }
    }
}

impl std::error::Error for LayerComposeError {}

// ---------------------------------------------------------------------------
// CPU reference: RMS normalization
// ---------------------------------------------------------------------------

/// RMS normalization: `y[i] = x[i] * weight[i] / rms(x)`.
///
/// Used as the pre-attention and pre-FFN normalization step.
pub fn cpu_rmsnorm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    debug_assert_eq!(n, weight.len());
    let sq_sum: f32 = input.iter().map(|&v| v * v).sum();
    let rms = (sq_sum / n as f32 + eps).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * wi / rms)
        .collect()
}

// ---------------------------------------------------------------------------
// CPU reference: matrix multiply
// ---------------------------------------------------------------------------

/// Row-major matrix multiply: `C[m,n] = A[m,k] @ B[k,n]`.
pub fn cpu_linear(
    input: &[f32],
    weight: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), m * k);
    debug_assert_eq!(weight.len(), k * n);
    let mut output = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += input[i * k + p] * weight[p * n + j];
            }
            output[i * n + j] = acc;
        }
    }
    output
}

// ---------------------------------------------------------------------------
// CPU reference: activations
// ---------------------------------------------------------------------------

/// SiLU activation: `silu(x) = x * sigmoid(x)`.
#[inline]
pub fn cpu_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softmax over a mutable slice (in-place).
fn softmax_inplace(x: &mut [f32]) {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference: attention sublayer
// ---------------------------------------------------------------------------

/// Full attention sublayer: norm → QKV projection → multi-head attention →
/// output projection → residual add.
///
/// Processes `seq_len` tokens. The input hidden states have shape
/// `[seq_len, hidden_dim]` stored in row-major order.
pub fn cpu_attention_layer(
    input: &LayerInput<'_>,
    block: &TransformerBlock,
) -> Result<LayerOutput, LayerComposeError> {
    let cfg = &block.config;
    cfg.validate()?;

    let h = cfg.hidden_dim;
    let hd = cfg.head_dim;
    let nh = cfg.num_heads;
    let seq_len = input.position_ids.len();

    if input.hidden_states.len() != seq_len * h {
        return Err(LayerComposeError::DimensionMismatch {
            expected: seq_len * h,
            got: input.hidden_states.len(),
            context: "hidden_states length".into(),
        });
    }

    // 1. Pre-attention RMS norm (per token)
    let mut normed = Vec::with_capacity(seq_len * h);
    for t in 0..seq_len {
        let tok = &input.hidden_states[t * h..(t + 1) * h];
        normed.extend(cpu_rmsnorm(tok, &block.attention_norm_weight, cfg.epsilon));
    }

    // 2. QKV projections: [seq_len, h] @ [h, h] → [seq_len, h]
    let q = cpu_linear(&normed, &block.q_proj, seq_len, h, h);
    let k_new = cpu_linear(&normed, &block.k_proj, seq_len, h, h);
    let v_new = cpu_linear(&normed, &block.v_proj, seq_len, h, h);

    // 3. Build full K/V including cache
    let (full_k, full_v, kv_len) = if let Some(ref cache) = input.kv_cache {
        let kv_len = cache.cache_len + seq_len;
        let mut fk = Vec::with_capacity(kv_len * h);
        fk.extend_from_slice(&cache.key_cache[..cache.cache_len * h]);
        fk.extend_from_slice(&k_new);
        let mut fv = Vec::with_capacity(kv_len * h);
        fv.extend_from_slice(&cache.value_cache[..cache.cache_len * h]);
        fv.extend_from_slice(&v_new);
        (fk, fv, kv_len)
    } else {
        (k_new.clone(), v_new.clone(), seq_len)
    };

    // 4. Multi-head attention
    let scale = 1.0 / (hd as f32).sqrt();
    let mut attn_out = vec![0.0_f32; seq_len * h];
    let mut all_weights = Vec::with_capacity(nh * seq_len * kv_len);

    for head in 0..nh {
        for t in 0..seq_len {
            // Extract query for this head/token
            let q_off = t * h + head * hd;
            let q_slice = &q[q_off..q_off + hd];

            // Compute attention scores against all KV positions
            let mut scores = vec![0.0_f32; kv_len];
            for (kv_pos, score) in scores.iter_mut().enumerate() {
                let k_off = kv_pos * h + head * hd;
                let mut dot = 0.0_f32;
                for d in 0..hd {
                    dot += q_slice[d] * full_k[k_off + d];
                }
                *score = dot * scale;
            }

            // Apply attention mask if provided
            if let Some(mask) = input.attention_mask {
                let mask_row = &mask[t * kv_len..(t * kv_len + kv_len).min(mask.len())];
                for (s, &m) in scores.iter_mut().zip(mask_row.iter()) {
                    if m <= 0.0 {
                        *s = f32::NEG_INFINITY;
                    }
                }
            }

            softmax_inplace(&mut scores);

            // Weighted sum of values
            let out_off = t * h + head * hd;
            for (kv_pos, &score) in scores.iter().enumerate() {
                let v_off = kv_pos * h + head * hd;
                for d in 0..hd {
                    attn_out[out_off + d] += score * full_v[v_off + d];
                }
            }

            all_weights.extend_from_slice(&scores);
        }
    }

    // 5. Output projection: [seq_len, h] @ [h, h]
    let projected = cpu_linear(&attn_out, &block.o_proj, seq_len, h, h);

    // 6. Residual connection
    let hidden_out: Vec<f32> = input
        .hidden_states
        .iter()
        .zip(projected.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    Ok(LayerOutput {
        hidden_states: hidden_out,
        new_key: k_new,
        new_value: v_new,
        attention_weights: Some(all_weights),
    })
}

// ---------------------------------------------------------------------------
// CPU reference: FFN sublayer
// ---------------------------------------------------------------------------

/// Full FFN sublayer: norm → gate_proj/up_proj → SwiGLU → down_proj →
/// residual add.
///
/// `hidden` has shape `[seq_len * hidden_dim]` where `seq_len` is inferred
/// from the slice length.
pub fn cpu_ffn_layer(
    hidden: &[f32],
    block: &TransformerBlock,
) -> Result<Vec<f32>, LayerComposeError> {
    let cfg = &block.config;
    cfg.validate()?;

    let h = cfg.hidden_dim;
    let inter = cfg.intermediate_dim;

    if !hidden.len().is_multiple_of(h) {
        return Err(LayerComposeError::DimensionMismatch {
            expected: h,
            got: hidden.len() % h,
            context: "hidden length must be a multiple of hidden_dim".into(),
        });
    }
    let seq_len = hidden.len() / h;

    // 1. Pre-FFN RMS norm (per token)
    let mut normed = Vec::with_capacity(seq_len * h);
    for t in 0..seq_len {
        let tok = &hidden[t * h..(t + 1) * h];
        normed.extend(cpu_rmsnorm(tok, &block.ffn_norm_weight, cfg.epsilon));
    }

    // 2. Gate and up projections: [seq_len, h] @ [h, inter]
    let gate = cpu_linear(&normed, &block.gate_proj, seq_len, h, inter);
    let up = cpu_linear(&normed, &block.up_proj, seq_len, h, inter);

    // 3. SwiGLU: silu(gate) * up
    let activated: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| cpu_silu(g) * u)
        .collect();

    // 4. Down projection: [seq_len, inter] @ [inter, h]
    let ffn_out = cpu_linear(&activated, &block.down_proj, seq_len, inter, h);

    // 5. Residual connection
    let result: Vec<f32> = hidden
        .iter()
        .zip(ffn_out.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    Ok(result)
}

// ---------------------------------------------------------------------------
// CPU reference: full transformer block
// ---------------------------------------------------------------------------

/// Complete transformer block: attention sublayer → FFN sublayer.
pub fn cpu_transformer_block(
    input: &LayerInput<'_>,
    block: &TransformerBlock,
) -> Result<LayerOutput, LayerComposeError> {
    // Attention sublayer (includes residual)
    let attn_output = cpu_attention_layer(input, block)?;

    // FFN sublayer (includes residual)
    let ffn_output = cpu_ffn_layer(&attn_output.hidden_states, block)?;

    Ok(LayerOutput {
        hidden_states: ffn_output,
        new_key: attn_output.new_key,
        new_value: attn_output.new_value,
        attention_weights: attn_output.attention_weights,
    })
}

// ---------------------------------------------------------------------------
// CPU reference: multi-block forward
// ---------------------------------------------------------------------------

/// N-layer forward pass through a stack of transformer blocks.
///
/// Each block's new KV vectors are passed to the next via an accumulated
/// cache, allowing autoregressive decoding across all layers.
pub fn cpu_multi_block_forward(
    input: &LayerInput<'_>,
    blocks: &[TransformerBlock],
) -> Result<LayerOutput, LayerComposeError> {
    if blocks.is_empty() {
        return Err(LayerComposeError::InvalidConfig(
            "blocks must not be empty".into(),
        ));
    }

    let seq_len = input.position_ids.len();

    // First block uses the original input
    let mut current = cpu_transformer_block(input, &blocks[0])?;

    // Accumulated KV for passing between layers
    let mut key_accum = current.new_key.clone();
    let mut val_accum = current.new_value.clone();

    for block in &blocks[1..] {
        let cache = KVSlice {
            key_cache: &key_accum,
            value_cache: &val_accum,
            cache_len: seq_len,
        };
        let layer_input = LayerInput {
            hidden_states: &current.hidden_states,
            attention_mask: input.attention_mask,
            position_ids: input.position_ids,
            kv_cache: Some(cache),
        };
        current = cpu_transformer_block(&layer_input, block)?;

        // Extend accumulated KV
        key_accum.extend_from_slice(&current.new_key);
        val_accum.extend_from_slice(&current.new_value);
    }

    Ok(current)
}

// ---------------------------------------------------------------------------
// Test block creation
// ---------------------------------------------------------------------------

/// Create a deterministic test block with pseudo-random weights seeded by
/// `seed`. Uses a simple xorshift PRNG to produce values in `[-0.1, 0.1]`.
pub fn cpu_create_test_block(config: &LayerConfig, seed: u64) -> TransformerBlock {
    let mut rng = SimpleRng::new(seed);

    let h = config.hidden_dim;
    let inter = config.intermediate_dim;

    let make_vec = |len: usize, rng: &mut SimpleRng| -> Vec<f32> {
        (0..len).map(|_| rng.next_f32() * 0.2 - 0.1).collect()
    };

    TransformerBlock {
        config: config.clone(),
        attention_norm_weight: vec![1.0; h], // norm weights = 1 for stability
        ffn_norm_weight: vec![1.0; h],
        q_proj: make_vec(h * h, &mut rng),
        k_proj: make_vec(h * h, &mut rng),
        v_proj: make_vec(h * h, &mut rng),
        o_proj: make_vec(h * h, &mut rng),
        gate_proj: make_vec(h * inter, &mut rng),
        up_proj: make_vec(h * inter, &mut rng),
        down_proj: make_vec(inter * h, &mut rng),
    }
}

/// Minimal xorshift64 PRNG for deterministic test weight generation.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) } // avoid zero state
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a value in `[0.0, 1.0)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0x00FF_FFFF) as f32 / 16_777_216.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ---------------------------------------------------------------

    fn tiny_config() -> LayerConfig {
        LayerConfig {
            hidden_dim: 8,
            num_heads: 2,
            head_dim: 4,
            intermediate_dim: 16,
            epsilon: 1e-5,
            use_flash_attention: false,
            use_fused_ops: false,
        }
    }

    fn make_input<'a>(
        hidden: &'a [f32],
        positions: &'a [u32],
        mask: Option<&'a [f32]>,
        cache: Option<KVSlice<'a>>,
    ) -> LayerInput<'a> {
        LayerInput {
            hidden_states: hidden,
            attention_mask: mask,
            position_ids: positions,
            kv_cache: cache,
        }
    }

    /// Assert all values are finite.
    fn assert_finite(v: &[f32], label: &str) {
        for (i, &val) in v.iter().enumerate() {
            assert!(val.is_finite(), "{label}[{i}] = {val} is not finite");
        }
    }

    // =====================================================================
    // RMS norm tests
    // =====================================================================

    #[test]
    fn test_rmsnorm_unit_weight() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let out = cpu_rmsnorm(&x, &w, 1e-5);
        assert_eq!(out.len(), 4);
        // After norm with unit weight, values should be scaled uniformly
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / 4.0 + 1e-5).sqrt();
        for (i, &val) in out.iter().enumerate() {
            let expected = x[i] / rms;
            assert!(
                (val - expected).abs() < 1e-5,
                "rmsnorm[{i}]: expected {expected}, got {val}"
            );
        }
    }

    #[test]
    fn test_rmsnorm_output_length() {
        let x = vec![0.5; 8];
        let w = vec![1.0; 8];
        let out = cpu_rmsnorm(&x, &w, 1e-5);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_rmsnorm_scaled_weight() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let w = vec![2.0, 2.0, 2.0, 2.0];
        let out = cpu_rmsnorm(&x, &w, 1e-5);
        // rms ≈ 1.0, so output ≈ 2.0 each
        for &val in &out {
            assert!((val - 2.0).abs() < 1e-4, "expected ~2.0, got {val}");
        }
    }

    // =====================================================================
    // Linear (matmul) tests
    // =====================================================================

    #[test]
    fn test_linear_identity() {
        #[rustfmt::skip]
        let w = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        let x = vec![3.0, 7.0];
        let out = cpu_linear(&x, &w, 1, 2, 2);
        assert_eq!(out, vec![3.0, 7.0]);
    }

    #[test]
    fn test_linear_known_result() {
        #[rustfmt::skip]
        let w = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        let x = vec![1.0, 1.0];
        let out = cpu_linear(&x, &w, 1, 2, 2);
        // [1,1] @ [[1,2],[3,4]] = [4, 6]
        assert_eq!(out, vec![4.0, 6.0]);
    }

    #[test]
    fn test_linear_output_shape() {
        let w = vec![0.1; 6]; // 2x3
        let x = vec![1.0, 1.0]; // 1x2
        let out = cpu_linear(&x, &w, 1, 2, 3);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_linear_batch() {
        // 2 tokens, each dim=2, weight 2x2 identity
        #[rustfmt::skip]
        let w = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        let x = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens
        let out = cpu_linear(&x, &w, 2, 2, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // =====================================================================
    // SiLU tests
    // =====================================================================

    #[test]
    fn test_silu_zero() {
        assert!((cpu_silu(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu_positive() {
        // silu(1.0) = 1.0 / (1.0 + exp(-1.0)) ≈ 0.7311
        let val = cpu_silu(1.0);
        assert!((val - 0.7311).abs() < 0.001, "silu(1.0) = {val}");
    }

    #[test]
    fn test_silu_negative() {
        // silu(-1.0) = -1.0 / (1.0 + exp(1.0)) ≈ -0.2689
        let val = cpu_silu(-1.0);
        assert!((val - (-0.2689)).abs() < 0.001, "silu(-1.0) = {val}");
    }

    // =====================================================================
    // Config validation tests
    // =====================================================================

    #[test]
    fn test_config_valid() {
        let cfg = tiny_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_head_dim_mismatch() {
        let cfg = LayerConfig {
            hidden_dim: 8,
            num_heads: 3, // 3 * 4 = 12 ≠ 8
            head_dim: 4,
            ..tiny_config()
        };
        assert!(matches!(
            cfg.validate(),
            Err(LayerComposeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_config_zero_hidden() {
        let cfg = LayerConfig { hidden_dim: 0, ..tiny_config() };
        assert!(matches!(
            cfg.validate(),
            Err(LayerComposeError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_zero_heads() {
        let cfg = LayerConfig {
            hidden_dim: 4,
            num_heads: 0,
            head_dim: 4,
            ..tiny_config()
        };
        assert!(matches!(
            cfg.validate(),
            Err(LayerComposeError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_negative_epsilon() {
        let cfg = LayerConfig { epsilon: -1.0, ..tiny_config() };
        assert!(matches!(
            cfg.validate(),
            Err(LayerComposeError::InvalidConfig(_))
        ));
    }

    // =====================================================================
    // Attention layer tests
    // =====================================================================

    #[test]
    fn test_attention_output_shape() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![0.1_f32; cfg.hidden_dim];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_attention_layer(&input, &block).unwrap();
        assert_eq!(out.hidden_states.len(), cfg.hidden_dim);
        assert_eq!(out.new_key.len(), cfg.hidden_dim);
        assert_eq!(out.new_value.len(), cfg.hidden_dim);
    }

    #[test]
    fn test_attention_multi_token() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let seq_len = 3;
        let hidden = vec![0.1_f32; seq_len * cfg.hidden_dim];
        let positions = vec![0, 1, 2];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_attention_layer(&input, &block).unwrap();
        assert_eq!(out.hidden_states.len(), seq_len * cfg.hidden_dim);
    }

    #[test]
    fn test_attention_with_kv_cache() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let h = cfg.hidden_dim;

        // First pass without cache
        let hidden1 = vec![0.1_f32; h];
        let pos1 = vec![0_u32];
        let input1 = make_input(&hidden1, &pos1, None, None);
        let out1 = cpu_attention_layer(&input1, &block).unwrap();

        // Second pass with cache from first
        let hidden2 = vec![0.2_f32; h];
        let pos2 = vec![1_u32];
        let cache = KVSlice {
            key_cache: &out1.new_key,
            value_cache: &out1.new_value,
            cache_len: 1,
        };
        let input2 = make_input(&hidden2, &pos2, None, Some(cache));
        let out2 = cpu_attention_layer(&input2, &block).unwrap();
        assert_eq!(out2.hidden_states.len(), h);
        assert_finite(&out2.hidden_states, "attn_with_cache");
    }

    #[test]
    fn test_attention_with_mask() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let h = cfg.hidden_dim;
        let hidden = vec![0.1_f32; h];
        let positions = vec![0_u32];
        // Mask allows attending to position 0
        let mask = vec![1.0_f32];
        let input = make_input(&hidden, &positions, Some(&mask), None);
        let out = cpu_attention_layer(&input, &block).unwrap();
        assert_eq!(out.hidden_states.len(), h);
        assert_finite(&out.hidden_states, "attn_masked");
    }

    #[test]
    fn test_attention_dimension_mismatch() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![0.1_f32; cfg.hidden_dim + 1]; // wrong size
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        assert!(cpu_attention_layer(&input, &block).is_err());
    }

    #[test]
    fn test_attention_returns_weights() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![0.1_f32; cfg.hidden_dim];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_attention_layer(&input, &block).unwrap();
        assert!(out.attention_weights.is_some());
        let weights = out.attention_weights.unwrap();
        // num_heads * seq_len * kv_len = 2 * 1 * 1 = 2
        assert_eq!(weights.len(), cfg.num_heads);
    }

    // =====================================================================
    // FFN layer tests
    // =====================================================================

    #[test]
    fn test_ffn_output_shape() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![0.1_f32; cfg.hidden_dim];
        let out = cpu_ffn_layer(&hidden, &block).unwrap();
        assert_eq!(out.len(), cfg.hidden_dim);
    }

    #[test]
    fn test_ffn_multi_token() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let seq_len = 4;
        let hidden = vec![0.1_f32; seq_len * cfg.hidden_dim];
        let out = cpu_ffn_layer(&hidden, &block).unwrap();
        assert_eq!(out.len(), seq_len * cfg.hidden_dim);
    }

    #[test]
    fn test_ffn_swiglu_activation() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![1.0_f32; cfg.hidden_dim];
        let out = cpu_ffn_layer(&hidden, &block).unwrap();
        // Output should differ from input (residual + FFN contribution)
        assert_ne!(out, hidden, "FFN should modify hidden states");
    }

    #[test]
    fn test_ffn_dimension_mismatch() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![0.1_f32; cfg.hidden_dim + 3]; // not a multiple
        assert!(cpu_ffn_layer(&hidden, &block).is_err());
    }

    // =====================================================================
    // Residual connection tests
    // =====================================================================

    #[test]
    fn test_residual_attention() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![1.0_f32; cfg.hidden_dim];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_attention_layer(&input, &block).unwrap();
        // Residual means output != 0 even if sublayer output is small
        let has_nonzero = out.hidden_states.iter().any(|&v| v.abs() > 1e-10);
        assert!(has_nonzero, "residual should preserve input signal");
    }

    #[test]
    fn test_residual_ffn() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![5.0_f32; cfg.hidden_dim];
        let out = cpu_ffn_layer(&hidden, &block).unwrap();
        // With residual, output should contain the original signal
        let has_nonzero = out.iter().any(|&v| v.abs() > 1.0);
        assert!(has_nonzero, "residual should preserve input signal");
    }

    // =====================================================================
    // Full transformer block tests
    // =====================================================================

    #[test]
    fn test_transformer_block_output_shape() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![0.1_f32; cfg.hidden_dim];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_transformer_block(&input, &block).unwrap();
        assert_eq!(out.hidden_states.len(), cfg.hidden_dim);
    }

    #[test]
    fn test_transformer_block_multi_token() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let seq_len = 3;
        let hidden = vec![0.1_f32; seq_len * cfg.hidden_dim];
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_transformer_block(&input, &block).unwrap();
        assert_eq!(out.hidden_states.len(), seq_len * cfg.hidden_dim);
    }

    #[test]
    fn test_transformer_block_deterministic() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![0.1_f32; cfg.hidden_dim];
        let positions = vec![0_u32];

        let input1 = make_input(&hidden, &positions, None, None);
        let out1 = cpu_transformer_block(&input1, &block).unwrap();

        let input2 = make_input(&hidden, &positions, None, None);
        let out2 = cpu_transformer_block(&input2, &block).unwrap();

        assert_eq!(
            out1.hidden_states, out2.hidden_states,
            "same input must produce identical output"
        );
    }

    // =====================================================================
    // Multi-block tests
    // =====================================================================

    #[test]
    fn test_multi_block_2_layers() {
        let cfg = tiny_config();
        let blocks: Vec<TransformerBlock> =
            (0..2).map(|i| cpu_create_test_block(&cfg, i)).collect();
        let hidden = vec![0.1_f32; cfg.hidden_dim];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_multi_block_forward(&input, &blocks).unwrap();
        assert_eq!(out.hidden_states.len(), cfg.hidden_dim);
        assert_finite(&out.hidden_states, "2-layer");
    }

    #[test]
    fn test_multi_block_4_layers() {
        let cfg = tiny_config();
        let blocks: Vec<TransformerBlock> =
            (0..4).map(|i| cpu_create_test_block(&cfg, i)).collect();
        let hidden = vec![0.1_f32; cfg.hidden_dim];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_multi_block_forward(&input, &blocks).unwrap();
        assert_eq!(out.hidden_states.len(), cfg.hidden_dim);
        assert_finite(&out.hidden_states, "4-layer");
    }

    #[test]
    fn test_multi_block_empty_blocks() {
        let hidden = vec![0.1_f32; 8];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let blocks: Vec<TransformerBlock> = vec![];
        assert!(cpu_multi_block_forward(&input, &blocks).is_err());
    }

    // =====================================================================
    // Numerical stability tests
    // =====================================================================

    #[test]
    fn test_numerical_stability_multi_layer() {
        let cfg = tiny_config();
        let blocks: Vec<TransformerBlock> =
            (0..4).map(|i| cpu_create_test_block(&cfg, i + 100)).collect();
        let hidden = vec![0.1_f32; cfg.hidden_dim];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_multi_block_forward(&input, &blocks).unwrap();
        assert!(
            !out.hidden_states.iter().any(|v| v.is_nan()),
            "NaN in multi-layer output"
        );
        assert!(
            !out.hidden_states.iter().any(|v| v.is_infinite()),
            "Inf in multi-layer output"
        );
    }

    #[test]
    fn test_numerical_stability_large_input() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![100.0_f32; cfg.hidden_dim];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_transformer_block(&input, &block).unwrap();
        assert_finite(&out.hidden_states, "large_input");
    }

    // =====================================================================
    // Edge case tests
    // =====================================================================

    #[test]
    fn test_seq_len_1() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![0.5_f32; cfg.hidden_dim];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_transformer_block(&input, &block).unwrap();
        assert_eq!(out.hidden_states.len(), cfg.hidden_dim);
    }

    #[test]
    fn test_hidden_dim_1() {
        let cfg = LayerConfig {
            hidden_dim: 1,
            num_heads: 1,
            head_dim: 1,
            intermediate_dim: 2,
            epsilon: 1e-5,
            use_flash_attention: false,
            use_fused_ops: false,
        };
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![1.0_f32];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_transformer_block(&input, &block).unwrap();
        assert_eq!(out.hidden_states.len(), 1);
        assert_finite(&out.hidden_states, "hidden_dim_1");
    }

    #[test]
    fn test_single_head() {
        let cfg = LayerConfig {
            hidden_dim: 4,
            num_heads: 1,
            head_dim: 4,
            intermediate_dim: 8,
            epsilon: 1e-5,
            use_flash_attention: false,
            use_fused_ops: false,
        };
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![0.1_f32; 4];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_transformer_block(&input, &block).unwrap();
        assert_eq!(out.hidden_states.len(), 4);
        assert_finite(&out.hidden_states, "single_head");
    }

    // =====================================================================
    // Test block creation
    // =====================================================================

    #[test]
    fn test_create_block_deterministic() {
        let cfg = tiny_config();
        let b1 = cpu_create_test_block(&cfg, 42);
        let b2 = cpu_create_test_block(&cfg, 42);
        assert_eq!(b1.q_proj, b2.q_proj);
        assert_eq!(b1.gate_proj, b2.gate_proj);
    }

    #[test]
    fn test_create_block_different_seeds() {
        let cfg = tiny_config();
        let b1 = cpu_create_test_block(&cfg, 1);
        let b2 = cpu_create_test_block(&cfg, 2);
        assert_ne!(b1.q_proj, b2.q_proj, "different seeds → different weights");
    }

    #[test]
    fn test_create_block_weight_sizes() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let h = cfg.hidden_dim;
        let inter = cfg.intermediate_dim;
        assert_eq!(block.attention_norm_weight.len(), h);
        assert_eq!(block.ffn_norm_weight.len(), h);
        assert_eq!(block.q_proj.len(), h * h);
        assert_eq!(block.k_proj.len(), h * h);
        assert_eq!(block.v_proj.len(), h * h);
        assert_eq!(block.o_proj.len(), h * h);
        assert_eq!(block.gate_proj.len(), h * inter);
        assert_eq!(block.up_proj.len(), h * inter);
        assert_eq!(block.down_proj.len(), inter * h);
    }

    // =====================================================================
    // Property: output shape matches input shape
    // =====================================================================

    #[test]
    fn test_property_shape_preserved() {
        for hidden_dim in [4, 8, 16] {
            let cfg = LayerConfig {
                hidden_dim,
                num_heads: hidden_dim / 4,
                head_dim: 4,
                intermediate_dim: hidden_dim * 2,
                ..tiny_config()
            };
            let block = cpu_create_test_block(&cfg, 99);
            let hidden = vec![0.1_f32; hidden_dim];
            let positions = vec![0_u32];
            let input = make_input(&hidden, &positions, None, None);
            let out = cpu_transformer_block(&input, &block).unwrap();
            assert_eq!(
                out.hidden_states.len(),
                hidden_dim,
                "shape preserved for hidden_dim={hidden_dim}"
            );
        }
    }

    // =====================================================================
    // Without KV cache
    // =====================================================================

    #[test]
    fn test_no_kv_cache() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![0.1_f32; cfg.hidden_dim];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_transformer_block(&input, &block).unwrap();
        assert_eq!(out.hidden_states.len(), cfg.hidden_dim);
        assert!(!out.new_key.is_empty());
        assert!(!out.new_value.is_empty());
    }

    // =====================================================================
    // Without attention mask
    // =====================================================================

    #[test]
    fn test_no_attention_mask() {
        let cfg = tiny_config();
        let block = cpu_create_test_block(&cfg, 42);
        let hidden = vec![0.1_f32; cfg.hidden_dim];
        let positions = vec![0_u32];
        let input = make_input(&hidden, &positions, None, None);
        let out = cpu_attention_layer(&input, &block).unwrap();
        assert_finite(&out.hidden_states, "no_mask");
    }
}
