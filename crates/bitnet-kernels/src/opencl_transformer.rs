//! OpenCL transformer layer composition for end-to-end GPU inference.
//!
//! # Architecture
//!
//! Composes individual kernels (attention, FFN, normalization) into a complete
//! pre-norm transformer layer suitable for BitNet models:
//!
//! ```text
//!   hidden = rms_norm(x)
//!   attn_out = multi_head_attention(hidden, kv_cache)
//!   x = x + attn_out
//!   hidden = rms_norm(x)
//!   ffn_out = gated_ffn(hidden)
//!   x = x + ffn_out
//! ```
//!
//! # CPU reference
//!
//! [`TransformerLayerRef`] provides a pure-CPU reference implementation for
//! correctness testing and non-GPU environments. All operations use scalar f32
//! arithmetic so results are deterministic and easy to validate.
//!
//! # GPU path (future)
//!
//! When the `oneapi` feature is enabled, the same layer graph will dispatch to
//! OpenCL kernels for attention, FFN, and normalization on Intel GPUs.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Full configuration for one transformer layer.
#[derive(Debug, Clone)]
pub struct TransformerLayerConfig {
    /// Model hidden dimension (e.g. 2048 for BitNet-2B).
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key/value heads (GQA when < num_heads).
    pub num_kv_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// FFN intermediate (up-projection) dimension.
    pub intermediate_size: usize,
    /// Epsilon for RMS normalization.
    pub rms_norm_eps: f32,
    /// Base frequency for rotary positional embeddings.
    pub rope_theta: f32,
    /// Maximum supported sequence length.
    pub max_seq_len: usize,
    /// Whether to use gated FFN (SiLU gate; true for LLaMA-style models).
    pub use_gated_ffn: bool,
}

impl Default for TransformerLayerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 64,
            intermediate_size: 5632,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 4096,
            use_gated_ffn: true,
        }
    }
}

impl TransformerLayerConfig {
    /// Validate that the configuration is internally consistent.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "hidden_size must be non-zero".into(),
            }
            .into());
        }
        if self.num_heads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_heads must be non-zero".into(),
            }
            .into());
        }
        if self.num_kv_heads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_kv_heads must be non-zero".into(),
            }
            .into());
        }
        if self.head_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "head_dim must be non-zero".into(),
            }
            .into());
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "num_heads ({}) must be divisible by num_kv_heads ({})",
                    self.num_heads, self.num_kv_heads
                ),
            }
            .into());
        }
        if self.num_heads * self.head_dim != self.hidden_size {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "num_heads * head_dim ({} * {}) must equal hidden_size ({})",
                    self.num_heads, self.head_dim, self.hidden_size
                ),
            }
            .into());
        }
        if self.intermediate_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "intermediate_size must be non-zero".into(),
            }
            .into());
        }
        if self.rms_norm_eps <= 0.0 {
            return Err(KernelError::InvalidArguments {
                reason: "rms_norm_eps must be positive".into(),
            }
            .into());
        }
        if self.max_seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "max_seq_len must be non-zero".into(),
            }
            .into());
        }
        Ok(())
    }

    /// Number of query heads per KV head (GQA ratio).
    pub fn gqa_ratio(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Total KV projection size (num_kv_heads * head_dim).
    pub fn kv_size(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
}

/// BitNet-2B (2-billion parameter) model configuration.
pub fn bitnet_2b_config() -> TransformerLayerConfig {
    TransformerLayerConfig {
        hidden_size: 2048,
        num_heads: 32,
        num_kv_heads: 8, // GQA
        head_dim: 64,
        intermediate_size: 5632,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 4096,
        use_gated_ffn: true,
    }
}

// ---------------------------------------------------------------------------
// Layer weights
// ---------------------------------------------------------------------------

/// Holds all weight tensors for a single transformer layer.
///
/// Weight shapes are documented in terms of the [`TransformerLayerConfig`]
/// fields: `H` = hidden_size, `D` = head_dim, `Nh` = num_heads,
/// `Nkv` = num_kv_heads, `I` = intermediate_size.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    // Attention
    /// Query projection weight `[H, H]` (H = Nh * D).
    pub wq: Vec<f32>,
    /// Key projection weight `[H, Nkv * D]`.
    pub wk: Vec<f32>,
    /// Value projection weight `[H, Nkv * D]`.
    pub wv: Vec<f32>,
    /// Output projection weight `[H, H]`.
    pub wo: Vec<f32>,

    // FFN
    /// Gate projection weight `[H, I]` (gated FFN).
    pub w_gate: Vec<f32>,
    /// Up projection weight `[H, I]`.
    pub w_up: Vec<f32>,
    /// Down projection weight `[I, H]`.
    pub w_down: Vec<f32>,

    // Norms
    /// RMS norm weight for attention sub-layer `[H]`.
    pub attn_norm: Vec<f32>,
    /// RMS norm weight for FFN sub-layer `[H]`.
    pub ffn_norm: Vec<f32>,
}

impl LayerWeights {
    /// Create zero-initialized weights matching the given config.
    pub fn zeros(config: &TransformerLayerConfig) -> Self {
        let h = config.hidden_size;
        let kv = config.kv_size();
        let i = config.intermediate_size;
        Self {
            wq: vec![0.0; h * h],
            wk: vec![0.0; h * kv],
            wv: vec![0.0; h * kv],
            wo: vec![0.0; h * h],
            w_gate: vec![0.0; h * i],
            w_up: vec![0.0; h * i],
            w_down: vec![0.0; i * h],
            attn_norm: vec![0.0; h],
            ffn_norm: vec![0.0; h],
        }
    }

    /// Create weights filled with ones (useful for testing norm pass-through).
    pub fn ones(config: &TransformerLayerConfig) -> Self {
        let h = config.hidden_size;
        let kv = config.kv_size();
        let i = config.intermediate_size;
        Self {
            wq: vec![1.0; h * h],
            wk: vec![1.0; h * kv],
            wv: vec![1.0; h * kv],
            wo: vec![1.0; h * h],
            w_gate: vec![1.0; h * i],
            w_up: vec![1.0; h * i],
            w_down: vec![1.0; i * h],
            attn_norm: vec![1.0; h],
            ffn_norm: vec![1.0; h],
        }
    }

    /// Validate that weight dimensions match the config.
    pub fn validate(&self, config: &TransformerLayerConfig) -> Result<()> {
        let h = config.hidden_size;
        let kv = config.kv_size();
        let i = config.intermediate_size;

        let checks = [
            (&self.wq, h * h, "wq"),
            (&self.wk, h * kv, "wk"),
            (&self.wv, h * kv, "wv"),
            (&self.wo, h * h, "wo"),
            (&self.w_gate, h * i, "w_gate"),
            (&self.w_up, h * i, "w_up"),
            (&self.w_down, i * h, "w_down"),
            (&self.attn_norm, h, "attn_norm"),
            (&self.ffn_norm, h, "ffn_norm"),
        ];

        for (tensor, expected, name) in checks {
            if tensor.len() != expected {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "weight {name}: expected length {expected}, got {}",
                        tensor.len()
                    ),
                }
                .into());
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// KV cache state
// ---------------------------------------------------------------------------

/// Key-value cache for incremental (autoregressive) decoding.
///
/// Stores projected K and V vectors for all positions seen so far.
/// Shape per cache: `[cached_len, num_kv_heads * head_dim]`.
#[derive(Debug, Clone)]
pub struct KvCacheState {
    /// Cached key vectors.
    pub k_cache: Vec<f32>,
    /// Cached value vectors.
    pub v_cache: Vec<f32>,
    /// Number of positions currently cached.
    pub cached_len: usize,
    /// KV projection size (num_kv_heads * head_dim).
    kv_size: usize,
    /// Maximum sequence length the cache can hold.
    max_len: usize,
}

impl KvCacheState {
    /// Create a new empty KV cache.
    pub fn new(config: &TransformerLayerConfig) -> Self {
        let kv_size = config.kv_size();
        Self {
            k_cache: vec![0.0; config.max_seq_len * kv_size],
            v_cache: vec![0.0; config.max_seq_len * kv_size],
            cached_len: 0,
            kv_size,
            max_len: config.max_seq_len,
        }
    }

    /// Append new K/V vectors for `seq_len` positions.
    pub fn append(&mut self, k: &[f32], v: &[f32], seq_len: usize) -> Result<()> {
        if self.cached_len + seq_len > self.max_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "KV cache overflow: {} + {} > {}",
                    self.cached_len, seq_len, self.max_len
                ),
            }
            .into());
        }
        let expected = seq_len * self.kv_size;
        if k.len() != expected || v.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "KV append size mismatch: expected {expected}, got k={} v={}",
                    k.len(),
                    v.len()
                ),
            }
            .into());
        }
        let offset = self.cached_len * self.kv_size;
        self.k_cache[offset..offset + expected].copy_from_slice(k);
        self.v_cache[offset..offset + expected].copy_from_slice(v);
        self.cached_len += seq_len;
        Ok(())
    }

    /// Get all cached keys up to `cached_len`.
    pub fn keys(&self) -> &[f32] {
        &self.k_cache[..self.cached_len * self.kv_size]
    }

    /// Get all cached values up to `cached_len`.
    pub fn values(&self) -> &[f32] {
        &self.v_cache[..self.cached_len * self.kv_size]
    }

    /// Reset the cache (clear all cached positions).
    pub fn reset(&mut self) {
        self.cached_len = 0;
    }
}

// ---------------------------------------------------------------------------
// Inference state
// ---------------------------------------------------------------------------

/// Tracks mutable state during autoregressive generation.
pub struct InferenceState {
    /// Current hidden state `[seq_len, hidden_size]`.
    pub hidden: Vec<f32>,
    /// KV caches, one per layer.
    pub kv_caches: Vec<KvCacheState>,
    /// Current position in the sequence.
    pub position: usize,
    /// Configuration.
    pub config: TransformerLayerConfig,
}

impl InferenceState {
    /// Create a new inference state for `num_layers` transformer layers.
    pub fn new(config: TransformerLayerConfig, num_layers: usize) -> Result<Self> {
        config.validate()?;
        let kv_caches = (0..num_layers).map(|_| KvCacheState::new(&config)).collect();
        Ok(Self { hidden: Vec::new(), kv_caches, position: 0, config })
    }

    /// Set the initial hidden state for a new sequence.
    pub fn set_input(&mut self, hidden: Vec<f32>, seq_len: usize) -> Result<()> {
        let expected = seq_len * self.config.hidden_size;
        if hidden.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "hidden state size mismatch: expected {expected}, got {}",
                    hidden.len()
                ),
            }
            .into());
        }
        self.hidden = hidden;
        Ok(())
    }

    /// Advance the position counter after processing tokens.
    pub fn advance(&mut self, num_tokens: usize) {
        self.position += num_tokens;
    }

    /// Reset to initial state (for a new sequence).
    pub fn reset(&mut self) {
        self.hidden.clear();
        self.position = 0;
        for cache in &mut self.kv_caches {
            cache.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference primitives
// ---------------------------------------------------------------------------

/// RMS normalization: `y[i] = x[i] * weight[i] / rms(x)`.
fn rms_norm_ref(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    debug_assert_eq!(n, weight.len());
    let sq_sum: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sq_sum / n as f32 + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(&xi, &wi)| xi * wi / rms).collect()
}

/// Matrix-vector multiply: `y = W * x` where W is `[rows, cols]` (row-major).
fn matvec_ref(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    (0..rows)
        .map(|r| {
            let row_start = r * cols;
            w[row_start..row_start + cols].iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()
        })
        .collect()
}

/// SiLU activation: `silu(x) = x * sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softmax over a slice (in-place).
fn softmax_ref(x: &mut [f32]) {
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

/// Simplified RoPE: rotate pairs of elements by position-dependent angles.
fn apply_rope_ref(x: &mut [f32], head_dim: usize, position: usize, theta: f32) {
    for pair_idx in 0..head_dim / 2 {
        let freq = 1.0 / theta.powf(2.0 * pair_idx as f32 / head_dim as f32);
        let angle = position as f32 * freq;
        let (cos_a, sin_a) = (angle.cos(), angle.sin());
        let i = pair_idx * 2;
        let j = i + 1;
        if j < x.len() {
            let (xi, xj) = (x[i], x[j]);
            x[i] = xi * cos_a - xj * sin_a;
            x[j] = xi * sin_a + xj * cos_a;
        }
    }
}

// ---------------------------------------------------------------------------
// Residual connection
// ---------------------------------------------------------------------------

/// Adds a residual connection: `output = x + sublayer(x)`.
pub struct ResidualConnection;

impl ResidualConnection {
    /// Compute `x + sublayer_output` element-wise.
    pub fn forward(x: &[f32], sublayer_output: &[f32]) -> Result<Vec<f32>> {
        if x.len() != sublayer_output.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "residual dimension mismatch: {} vs {}",
                    x.len(),
                    sublayer_output.len()
                ),
            }
            .into());
        }
        Ok(x.iter().zip(sublayer_output.iter()).map(|(&a, &b)| a + b).collect())
    }
}

// ---------------------------------------------------------------------------
// Pre-norm block
// ---------------------------------------------------------------------------

/// Pre-normalization block: normalize → sublayer → residual.
///
/// Used in LLaMA/BitNet-style architectures where the norm is applied
/// *before* the sublayer (attention or FFN), not after.
pub struct PreNormBlock;

impl PreNormBlock {
    /// Apply pre-norm: `output = x + sublayer(rms_norm(x, weight))`.
    ///
    /// The `sublayer_fn` receives the normalized hidden state and returns the
    /// sublayer output (same dimension as input).
    pub fn forward(
        x: &[f32],
        norm_weight: &[f32],
        eps: f32,
        sublayer_fn: impl FnOnce(&[f32]) -> Result<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let normalized = rms_norm_ref(x, norm_weight, eps);
        let sublayer_out = sublayer_fn(&normalized)?;
        ResidualConnection::forward(x, &sublayer_out)
    }
}

// ---------------------------------------------------------------------------
// Transformer layer reference (CPU)
// ---------------------------------------------------------------------------

/// CPU reference implementation of a single transformer layer.
///
/// Implements the pre-norm architecture used by LLaMA / BitNet models.
/// Processes one token at a time for simplicity (seq_len = 1 in generation).
pub struct TransformerLayerRef;

impl TransformerLayerRef {
    /// Single-head attention for one query token against the full KV cache.
    ///
    /// `q`, `k_cache`, `v_cache` are in per-head space (`head_dim`-wide).
    fn single_head_attention(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        head_dim: usize,
        cache_len: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores: Vec<f32> = (0..cache_len)
            .map(|t| {
                let k_off = t * head_dim;
                q.iter()
                    .zip(k_cache[k_off..k_off + head_dim].iter())
                    .map(|(&qi, &ki)| qi * ki)
                    .sum::<f32>()
                    * scale
            })
            .collect();
        softmax_ref(&mut scores);
        // Weighted sum of values
        let mut out = vec![0.0_f32; head_dim];
        for (t, &score) in scores.iter().enumerate() {
            let v_off = t * head_dim;
            for d in 0..head_dim {
                out[d] += score * v_cache[v_off + d];
            }
        }
        out
    }

    /// Multi-head attention with GQA support.
    ///
    /// `hidden` is the normalized hidden state `[hidden_size]`.
    /// Projects Q/K/V, applies RoPE, updates KV cache, computes attention,
    /// then projects back to hidden_size.
    fn attention_ref(
        hidden: &[f32],
        weights: &LayerWeights,
        kv_cache: &mut KvCacheState,
        position: usize,
        config: &TransformerLayerConfig,
    ) -> Result<Vec<f32>> {
        let h = config.hidden_size;
        let hd = config.head_dim;
        let nkv = config.num_kv_heads;
        let kv_dim = config.kv_size();

        // Q/K/V projections
        let mut q = matvec_ref(&weights.wq, hidden, h, h);
        let mut k = matvec_ref(&weights.wk, hidden, kv_dim, h);
        let v = matvec_ref(&weights.wv, hidden, kv_dim, h);

        // Apply RoPE to each Q head
        for head in 0..config.num_heads {
            let start = head * hd;
            apply_rope_ref(&mut q[start..start + hd], hd, position, config.rope_theta);
        }
        // Apply RoPE to each KV head
        for head in 0..nkv {
            let start = head * hd;
            apply_rope_ref(&mut k[start..start + hd], hd, position, config.rope_theta);
        }

        // Update KV cache
        kv_cache.append(&k, &v, 1)?;

        // Multi-head attention with GQA
        let gqa_ratio = config.gqa_ratio();
        let mut attn_output = vec![0.0_f32; h];

        for head in 0..config.num_heads {
            let kv_head = head / gqa_ratio;
            let q_slice = &q[head * hd..(head + 1) * hd];

            // Extract this KV head's cache (strided access)
            let cache_len = kv_cache.cached_len;
            let mut head_k = vec![0.0_f32; cache_len * hd];
            let mut head_v = vec![0.0_f32; cache_len * hd];
            for t in 0..cache_len {
                let src_off = t * kv_dim + kv_head * hd;
                let dst_off = t * hd;
                head_k[dst_off..dst_off + hd]
                    .copy_from_slice(&kv_cache.keys()[src_off..src_off + hd]);
                head_v[dst_off..dst_off + hd]
                    .copy_from_slice(&kv_cache.values()[src_off..src_off + hd]);
            }

            let head_out = Self::single_head_attention(q_slice, &head_k, &head_v, hd, cache_len);
            attn_output[head * hd..(head + 1) * hd].copy_from_slice(&head_out);
        }

        // Output projection
        Ok(matvec_ref(&weights.wo, &attn_output, h, h))
    }

    /// Gated FFN: `down(silu(gate(x)) * up(x))`.
    fn ffn_ref(
        hidden: &[f32],
        weights: &LayerWeights,
        config: &TransformerLayerConfig,
    ) -> Result<Vec<f32>> {
        let h = config.hidden_size;
        let i = config.intermediate_size;

        let gate = matvec_ref(&weights.w_gate, hidden, i, h);
        let up = matvec_ref(&weights.w_up, hidden, i, h);

        let activated: Vec<f32> = if config.use_gated_ffn {
            gate.iter().zip(up.iter()).map(|(&g, &u)| silu(g) * u).collect()
        } else {
            gate.iter().map(|&g| silu(g)).collect()
        };

        Ok(matvec_ref(&weights.w_down, &activated, h, i))
    }
}

/// Single transformer layer forward pass (pre-norm architecture):
///   hidden = rms_norm(x)
///   attn_out = multi_head_attention(hidden, kv_cache)
///   x = x + attn_out
///   hidden = rms_norm(x)
///   ffn_out = gated_ffn(hidden)
///   x = x + ffn_out
pub fn transformer_layer_forward_ref(
    x: &[f32],
    weights: &LayerWeights,
    kv_cache: &mut KvCacheState,
    position: usize,
    config: &TransformerLayerConfig,
) -> Result<Vec<f32>> {
    config.validate()?;
    weights.validate(config)?;

    let h = config.hidden_size;
    if x.len() != h {
        return Err(KernelError::InvalidArguments {
            reason: format!("input size mismatch: expected {h}, got {}", x.len()),
        }
        .into());
    }

    // Pre-norm → attention → residual
    let after_attn = PreNormBlock::forward(x, &weights.attn_norm, config.rms_norm_eps, |normed| {
        TransformerLayerRef::attention_ref(normed, weights, kv_cache, position, config)
    })?;

    // Pre-norm → FFN → residual
    let after_ffn =
        PreNormBlock::forward(&after_attn, &weights.ffn_norm, config.rms_norm_eps, |normed| {
            TransformerLayerRef::ffn_ref(normed, weights, config)
        })?;

    Ok(after_ffn)
}

// ---------------------------------------------------------------------------
// Transformer block (manages one block with its weights)
// ---------------------------------------------------------------------------

/// Manages a single transformer block with its own weights and KV cache.
pub struct TransformerBlock {
    /// Layer weights.
    pub weights: LayerWeights,
    /// KV cache for incremental decoding.
    pub kv_cache: KvCacheState,
    /// Layer index (for debugging/logging).
    pub layer_idx: usize,
}

impl TransformerBlock {
    /// Create a new transformer block with the given weights.
    pub fn new(
        weights: LayerWeights,
        config: &TransformerLayerConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        weights.validate(config)?;
        Ok(Self { weights, kv_cache: KvCacheState::new(config), layer_idx })
    }

    /// Forward pass through this block.
    pub fn forward(
        &mut self,
        x: &[f32],
        position: usize,
        config: &TransformerLayerConfig,
    ) -> Result<Vec<f32>> {
        transformer_layer_forward_ref(x, &self.weights, &mut self.kv_cache, position, config)
    }

    /// Reset the KV cache for a new sequence.
    pub fn reset_cache(&mut self) {
        self.kv_cache.reset();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers -----------------------------------------------------------

    /// Small config for fast tests (4 heads, hidden=8, head_dim=2).
    fn tiny_config() -> TransformerLayerConfig {
        TransformerLayerConfig {
            hidden_size: 8,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 2,
            intermediate_size: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 32,
            use_gated_ffn: true,
        }
    }

    /// Small GQA config (4 heads, 2 KV heads).
    fn tiny_gqa_config() -> TransformerLayerConfig {
        TransformerLayerConfig {
            hidden_size: 8,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 2,
            intermediate_size: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 32,
            use_gated_ffn: true,
        }
    }

    /// Create small random-ish weights (deterministic from seed).
    fn seeded_weights(config: &TransformerLayerConfig, seed: u64) -> LayerWeights {
        let mut w = LayerWeights::ones(config);
        // Simple LCG-based pseudo-random to avoid pulling in rand
        let mut state = seed;
        let mut next_f32 = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map to [-0.1, 0.1] for numerical stability
            ((state >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.2
        };
        for v in w.wq.iter_mut() {
            *v = next_f32();
        }
        for v in w.wk.iter_mut() {
            *v = next_f32();
        }
        for v in w.wv.iter_mut() {
            *v = next_f32();
        }
        for v in w.wo.iter_mut() {
            *v = next_f32();
        }
        for v in w.w_gate.iter_mut() {
            *v = next_f32();
        }
        for v in w.w_up.iter_mut() {
            *v = next_f32();
        }
        for v in w.w_down.iter_mut() {
            *v = next_f32();
        }
        // Norm weights stay as 1.0 (identity scaling)
        w
    }

    // =====================================================================
    // Config tests
    // =====================================================================

    #[test]
    fn test_bitnet_2b_config_validates() {
        let cfg = bitnet_2b_config();
        cfg.validate().unwrap();
    }

    #[test]
    fn test_default_config_validates() {
        let cfg = TransformerLayerConfig::default();
        cfg.validate().unwrap();
    }

    #[test]
    fn test_tiny_config_validates() {
        tiny_config().validate().unwrap();
    }

    #[test]
    fn test_tiny_gqa_config_validates() {
        tiny_gqa_config().validate().unwrap();
    }

    #[test]
    fn test_config_rejects_zero_hidden_size() {
        let mut cfg = tiny_config();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_rejects_zero_num_heads() {
        let mut cfg = tiny_config();
        cfg.num_heads = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_rejects_zero_num_kv_heads() {
        let mut cfg = tiny_config();
        cfg.num_kv_heads = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_rejects_zero_head_dim() {
        let mut cfg = tiny_config();
        cfg.head_dim = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_rejects_zero_intermediate_size() {
        let mut cfg = tiny_config();
        cfg.intermediate_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_rejects_zero_max_seq_len() {
        let mut cfg = tiny_config();
        cfg.max_seq_len = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_rejects_non_divisible_kv_heads() {
        let mut cfg = tiny_config();
        cfg.num_kv_heads = 3; // 4 % 3 != 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_rejects_inconsistent_dims() {
        let mut cfg = tiny_config();
        cfg.head_dim = 3; // 4 * 3 != 8
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_rejects_zero_eps() {
        let mut cfg = tiny_config();
        cfg.rms_norm_eps = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_rejects_negative_eps() {
        let mut cfg = tiny_config();
        cfg.rms_norm_eps = -1e-5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_gqa_ratio() {
        assert_eq!(bitnet_2b_config().gqa_ratio(), 4); // 32 / 8
        assert_eq!(tiny_config().gqa_ratio(), 1); // 4 / 4 (MHA)
        assert_eq!(tiny_gqa_config().gqa_ratio(), 2); // 4 / 2
    }

    #[test]
    fn test_config_kv_size() {
        assert_eq!(bitnet_2b_config().kv_size(), 8 * 64); // 512
        assert_eq!(tiny_config().kv_size(), 4 * 2); // 8
    }

    // =====================================================================
    // LayerWeights tests
    // =====================================================================

    #[test]
    fn test_weights_zeros_validates() {
        let cfg = tiny_config();
        let w = LayerWeights::zeros(&cfg);
        w.validate(&cfg).unwrap();
    }

    #[test]
    fn test_weights_ones_validates() {
        let cfg = tiny_config();
        let w = LayerWeights::ones(&cfg);
        w.validate(&cfg).unwrap();
    }

    #[test]
    fn test_weights_wrong_wq_size_rejected() {
        let cfg = tiny_config();
        let mut w = LayerWeights::zeros(&cfg);
        w.wq.push(0.0);
        assert!(w.validate(&cfg).is_err());
    }

    #[test]
    fn test_weights_wrong_norm_size_rejected() {
        let cfg = tiny_config();
        let mut w = LayerWeights::zeros(&cfg);
        w.attn_norm.pop();
        assert!(w.validate(&cfg).is_err());
    }

    #[test]
    fn test_weights_gqa_kv_sizes() {
        let cfg = tiny_gqa_config();
        let w = LayerWeights::zeros(&cfg);
        assert_eq!(w.wk.len(), 8 * 4); // hidden * kv_size = 8 * (2*2)
        assert_eq!(w.wv.len(), 8 * 4);
        w.validate(&cfg).unwrap();
    }

    // =====================================================================
    // KV cache tests
    // =====================================================================

    #[test]
    fn test_kv_cache_new_empty() {
        let cfg = tiny_config();
        let cache = KvCacheState::new(&cfg);
        assert_eq!(cache.cached_len, 0);
        assert!(cache.keys().is_empty());
        assert!(cache.values().is_empty());
    }

    #[test]
    fn test_kv_cache_append_one_token() {
        let cfg = tiny_config();
        let mut cache = KvCacheState::new(&cfg);
        let kv_size = cfg.kv_size();
        let k = vec![1.0_f32; kv_size];
        let v = vec![2.0_f32; kv_size];
        cache.append(&k, &v, 1).unwrap();
        assert_eq!(cache.cached_len, 1);
        assert_eq!(cache.keys().len(), kv_size);
        assert_eq!(cache.values().len(), kv_size);
        assert!(cache.keys().iter().all(|&x| (x - 1.0).abs() < f32::EPSILON));
        assert!(cache.values().iter().all(|&x| (x - 2.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_kv_cache_append_multiple() {
        let cfg = tiny_config();
        let mut cache = KvCacheState::new(&cfg);
        let kv_size = cfg.kv_size();
        for i in 0..5 {
            let k = vec![i as f32; kv_size];
            let v = vec![(i as f32) * 10.0; kv_size];
            cache.append(&k, &v, 1).unwrap();
        }
        assert_eq!(cache.cached_len, 5);
        assert_eq!(cache.keys().len(), 5 * kv_size);
    }

    #[test]
    fn test_kv_cache_overflow_rejected() {
        let mut cfg = tiny_config();
        cfg.max_seq_len = 2;
        let mut cache = KvCacheState::new(&cfg);
        let kv_size = cfg.kv_size();
        let k = vec![1.0; kv_size];
        let v = vec![1.0; kv_size];
        cache.append(&k, &v, 1).unwrap();
        cache.append(&k, &v, 1).unwrap();
        assert!(cache.append(&k, &v, 1).is_err());
    }

    #[test]
    fn test_kv_cache_size_mismatch_rejected() {
        let cfg = tiny_config();
        let mut cache = KvCacheState::new(&cfg);
        let k = vec![1.0; 3]; // wrong size
        let v = vec![1.0; cfg.kv_size()];
        assert!(cache.append(&k, &v, 1).is_err());
    }

    #[test]
    fn test_kv_cache_reset() {
        let cfg = tiny_config();
        let mut cache = KvCacheState::new(&cfg);
        let kv_size = cfg.kv_size();
        cache.append(&vec![1.0; kv_size], &vec![1.0; kv_size], 1).unwrap();
        assert_eq!(cache.cached_len, 1);
        cache.reset();
        assert_eq!(cache.cached_len, 0);
        assert!(cache.keys().is_empty());
    }

    // =====================================================================
    // Inference state tests
    // =====================================================================

    #[test]
    fn test_inference_state_new() {
        let cfg = tiny_config();
        let state = InferenceState::new(cfg, 4).unwrap();
        assert_eq!(state.kv_caches.len(), 4);
        assert_eq!(state.position, 0);
    }

    #[test]
    fn test_inference_state_set_input() {
        let cfg = tiny_config();
        let mut state = InferenceState::new(cfg, 1).unwrap();
        let hidden = vec![1.0_f32; 8];
        state.set_input(hidden, 1).unwrap();
        assert_eq!(state.hidden.len(), 8);
    }

    #[test]
    fn test_inference_state_set_input_wrong_size() {
        let cfg = tiny_config();
        let mut state = InferenceState::new(cfg, 1).unwrap();
        let hidden = vec![1.0_f32; 5]; // wrong
        assert!(state.set_input(hidden, 1).is_err());
    }

    #[test]
    fn test_inference_state_advance() {
        let cfg = tiny_config();
        let mut state = InferenceState::new(cfg, 1).unwrap();
        state.advance(3);
        assert_eq!(state.position, 3);
        state.advance(2);
        assert_eq!(state.position, 5);
    }

    #[test]
    fn test_inference_state_reset() {
        let cfg = tiny_config();
        let mut state = InferenceState::new(cfg, 2).unwrap();
        state.set_input(vec![1.0; 8], 1).unwrap();
        state.advance(5);
        state.reset();
        assert_eq!(state.position, 0);
        assert!(state.hidden.is_empty());
    }

    // =====================================================================
    // RMS norm reference tests
    // =====================================================================

    #[test]
    fn test_rms_norm_unit_weights() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let y = rms_norm_ref(&x, &w, 1e-5);
        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (30.0_f32 / 4.0 + 1e-5).sqrt();
        for (i, &yi) in y.iter().enumerate() {
            let expected = x[i] / rms;
            assert!((yi - expected).abs() < 1e-5, "rms_norm mismatch at {i}");
        }
    }

    #[test]
    fn test_rms_norm_preserves_unit_rms() {
        let x = vec![3.0, 4.0];
        let w = vec![1.0; 2];
        let y = rms_norm_ref(&x, &w, 1e-5);
        let y_rms: f32 = (y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32).sqrt();
        assert!((y_rms - 1.0).abs() < 1e-4, "output RMS should be ~1.0, got {y_rms}");
    }

    #[test]
    fn test_rms_norm_with_weights() {
        let x = vec![1.0, 2.0];
        let w = vec![2.0, 0.5];
        let y = rms_norm_ref(&x, &w, 1e-5);
        let rms = (5.0_f32 / 2.0 + 1e-5).sqrt();
        assert!((y[0] - 2.0 / rms).abs() < 1e-5);
        assert!((y[1] - 1.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_all_zeros() {
        let x = vec![0.0; 4];
        let w = vec![1.0; 4];
        let y = rms_norm_ref(&x, &w, 1e-5);
        // All zeros, RMS ≈ sqrt(eps), output ≈ 0
        for &yi in &y {
            assert!(yi.abs() < 0.1, "near-zero input should give near-zero output");
        }
    }

    // =====================================================================
    // Residual connection tests
    // =====================================================================

    #[test]
    fn test_residual_identity_sublayer() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let zeros = vec![0.0; 4];
        let out = ResidualConnection::forward(&x, &zeros).unwrap();
        assert_eq!(out, x, "x + 0 should equal x");
    }

    #[test]
    fn test_residual_adds_correctly() {
        let x = vec![1.0, 2.0, 3.0];
        let sub = vec![10.0, 20.0, 30.0];
        let out = ResidualConnection::forward(&x, &sub).unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_residual_dimension_mismatch() {
        let x = vec![1.0, 2.0];
        let sub = vec![1.0, 2.0, 3.0];
        assert!(ResidualConnection::forward(&x, &sub).is_err());
    }

    #[test]
    fn test_residual_negative_values() {
        let x = vec![1.0, -2.0, 3.0];
        let sub = vec![-1.0, 2.0, -3.0];
        let out = ResidualConnection::forward(&x, &sub).unwrap();
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    // =====================================================================
    // PreNorm block tests
    // =====================================================================

    #[test]
    fn test_prenorm_identity_sublayer() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let out = PreNormBlock::forward(&x, &w, 1e-5, |_normed| Ok(vec![0.0; 4])).unwrap();
        // sublayer returns 0 → residual should be x
        assert_eq!(out, x);
    }

    #[test]
    fn test_prenorm_sublayer_receives_normalized_input() {
        let x = vec![3.0, 4.0];
        let w = vec![1.0; 2];
        let mut received = Vec::new();
        let _ = PreNormBlock::forward(&x, &w, 1e-5, |normed| {
            received = normed.to_vec();
            Ok(vec![0.0; 2])
        });
        // Check that the sublayer received normalized input
        let rms: f32 = (received.iter().map(|v| v * v).sum::<f32>() / 2.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-4, "sublayer input should have unit RMS");
    }

    #[test]
    fn test_prenorm_propagates_sublayer_error() {
        let x = vec![1.0, 2.0];
        let w = vec![1.0; 2];
        let result = PreNormBlock::forward(&x, &w, 1e-5, |_| {
            Err(KernelError::InvalidArguments { reason: "test error".into() }.into())
        });
        assert!(result.is_err());
    }

    // =====================================================================
    // Softmax reference tests
    // =====================================================================

    #[test]
    fn test_softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        softmax_ref(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_monotonic() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_ref(&mut x);
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);
    }

    #[test]
    fn test_softmax_single_element() {
        let mut x = vec![42.0];
        softmax_ref(&mut x);
        assert!((x[0] - 1.0).abs() < 1e-6);
    }

    // =====================================================================
    // SiLU tests
    // =====================================================================

    #[test]
    fn test_silu_zero() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu_positive() {
        let y = silu(2.0);
        assert!(y > 0.0);
        assert!(y < 2.0);
    }

    // =====================================================================
    // RoPE tests
    // =====================================================================

    #[test]
    fn test_rope_position_zero_is_identity() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let orig = x.clone();
        apply_rope_ref(&mut x, 4, 0, 10000.0);
        // At position 0, angle=0, cos=1, sin=0 → identity
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>();
        apply_rope_ref(&mut x, 4, 5, 10000.0);
        let norm_after: f32 = x.iter().map(|v| v * v).sum::<f32>();
        assert!((norm_before - norm_after).abs() < 1e-3, "RoPE should preserve vector norm");
    }

    // =====================================================================
    // Layer forward pass tests
    // =====================================================================

    #[test]
    fn test_layer_forward_output_dimensions() {
        let cfg = tiny_config();
        let w = seeded_weights(&cfg, 42);
        let mut kv = KvCacheState::new(&cfg);
        let x = vec![0.1_f32; cfg.hidden_size];
        let out = transformer_layer_forward_ref(&x, &w, &mut kv, 0, &cfg).unwrap();
        assert_eq!(out.len(), cfg.hidden_size);
    }

    #[test]
    fn test_layer_forward_gqa_output_dimensions() {
        let cfg = tiny_gqa_config();
        let w = seeded_weights(&cfg, 42);
        let mut kv = KvCacheState::new(&cfg);
        let x = vec![0.1_f32; cfg.hidden_size];
        let out = transformer_layer_forward_ref(&x, &w, &mut kv, 0, &cfg).unwrap();
        assert_eq!(out.len(), cfg.hidden_size);
    }

    #[test]
    fn test_layer_forward_wrong_input_size() {
        let cfg = tiny_config();
        let w = seeded_weights(&cfg, 42);
        let mut kv = KvCacheState::new(&cfg);
        let x = vec![0.1_f32; 3]; // wrong size
        assert!(transformer_layer_forward_ref(&x, &w, &mut kv, 0, &cfg).is_err());
    }

    #[test]
    fn test_layer_forward_no_nan() {
        let cfg = tiny_config();
        let w = seeded_weights(&cfg, 42);
        let mut kv = KvCacheState::new(&cfg);
        let x = vec![0.1_f32; cfg.hidden_size];
        let out = transformer_layer_forward_ref(&x, &w, &mut kv, 0, &cfg).unwrap();
        assert!(!out.iter().any(|v| v.is_nan()), "output must not contain NaN");
    }

    #[test]
    fn test_layer_forward_no_inf() {
        let cfg = tiny_config();
        let w = seeded_weights(&cfg, 42);
        let mut kv = KvCacheState::new(&cfg);
        let x = vec![0.1_f32; cfg.hidden_size];
        let out = transformer_layer_forward_ref(&x, &w, &mut kv, 0, &cfg).unwrap();
        assert!(!out.iter().any(|v| v.is_infinite()), "output must not contain Inf");
    }

    #[test]
    fn test_layer_forward_updates_kv_cache() {
        let cfg = tiny_config();
        let w = seeded_weights(&cfg, 42);
        let mut kv = KvCacheState::new(&cfg);
        let x = vec![0.1_f32; cfg.hidden_size];
        transformer_layer_forward_ref(&x, &w, &mut kv, 0, &cfg).unwrap();
        assert_eq!(kv.cached_len, 1);
    }

    // =====================================================================
    // KV cache incremental decoding tests
    // =====================================================================

    #[test]
    fn test_incremental_decoding_three_steps() {
        let cfg = tiny_config();
        let w = seeded_weights(&cfg, 42);
        let mut kv = KvCacheState::new(&cfg);
        let x = vec![0.1_f32; cfg.hidden_size];
        for pos in 0..3 {
            let out = transformer_layer_forward_ref(&x, &w, &mut kv, pos, &cfg).unwrap();
            assert_eq!(out.len(), cfg.hidden_size);
            assert_eq!(kv.cached_len, pos + 1);
        }
    }

    #[test]
    fn test_incremental_decoding_kv_cache_grows() {
        let cfg = tiny_config();
        let w = seeded_weights(&cfg, 42);
        let mut kv = KvCacheState::new(&cfg);
        let x = vec![0.05_f32; cfg.hidden_size];
        for pos in 0..10 {
            transformer_layer_forward_ref(&x, &w, &mut kv, pos, &cfg).unwrap();
        }
        assert_eq!(kv.cached_len, 10);
        assert_eq!(kv.keys().len(), 10 * cfg.kv_size());
    }

    // =====================================================================
    // Multiple layers (stacked) tests
    // =====================================================================

    #[test]
    fn test_stacked_layers_no_overflow() {
        let cfg = tiny_config();
        let num_layers = 4;
        let weights: Vec<_> = (0..num_layers).map(|i| seeded_weights(&cfg, i as u64)).collect();
        let mut kv_caches: Vec<_> = (0..num_layers).map(|_| KvCacheState::new(&cfg)).collect();
        let mut x = vec![0.1_f32; cfg.hidden_size];
        for pos in 0..3 {
            for layer in 0..num_layers {
                x = transformer_layer_forward_ref(
                    &x,
                    &weights[layer],
                    &mut kv_caches[layer],
                    pos,
                    &cfg,
                )
                .unwrap();
            }
            assert!(!x.iter().any(|v| v.is_nan()), "NaN at position {pos}");
            assert!(!x.iter().any(|v| v.is_infinite()), "Inf at position {pos}");
        }
    }

    #[test]
    fn test_stacked_layers_dimensions_preserved() {
        let cfg = tiny_config();
        let num_layers = 3;
        let weights: Vec<_> =
            (0..num_layers).map(|i| seeded_weights(&cfg, i as u64 + 100)).collect();
        let mut kv_caches: Vec<_> = (0..num_layers).map(|_| KvCacheState::new(&cfg)).collect();
        let mut x = vec![0.05_f32; cfg.hidden_size];
        for layer in 0..num_layers {
            x = transformer_layer_forward_ref(&x, &weights[layer], &mut kv_caches[layer], 0, &cfg)
                .unwrap();
            assert_eq!(x.len(), cfg.hidden_size);
        }
    }

    // =====================================================================
    // Edge cases
    // =====================================================================

    #[test]
    fn test_seq_len_1_single_token_generation() {
        let cfg = tiny_config();
        let w = seeded_weights(&cfg, 42);
        let mut kv = KvCacheState::new(&cfg);
        // Process a single token at position 0
        let x = vec![0.1_f32; cfg.hidden_size];
        let out = transformer_layer_forward_ref(&x, &w, &mut kv, 0, &cfg).unwrap();
        assert_eq!(out.len(), cfg.hidden_size);
        assert_eq!(kv.cached_len, 1);
    }

    #[test]
    fn test_deterministic_output() {
        let cfg = tiny_config();
        let x = vec![0.1_f32; cfg.hidden_size];
        let w = seeded_weights(&cfg, 42);

        let mut kv1 = KvCacheState::new(&cfg);
        let out1 = transformer_layer_forward_ref(&x, &w, &mut kv1, 0, &cfg).unwrap();

        let mut kv2 = KvCacheState::new(&cfg);
        let out2 = transformer_layer_forward_ref(&x, &w, &mut kv2, 0, &cfg).unwrap();

        assert_eq!(out1, out2, "same inputs must produce identical outputs");
    }

    #[test]
    fn test_different_positions_different_outputs() {
        let cfg = tiny_config();
        let w = seeded_weights(&cfg, 42);
        let x = vec![0.1_f32; cfg.hidden_size];

        // Prefill a shared context so attention has multiple KV entries
        let mut kv1 = KvCacheState::new(&cfg);
        for pos in 0..3 {
            transformer_layer_forward_ref(&x, &w, &mut kv1, pos, &cfg).unwrap();
        }
        let mut kv2 = kv1.clone();

        // Now decode at two different positions from the same KV cache state
        let out3 = transformer_layer_forward_ref(&x, &w, &mut kv1, 3, &cfg).unwrap();
        let out7 = transformer_layer_forward_ref(&x, &w, &mut kv2, 7, &cfg).unwrap();

        // RoPE at different positions should produce different outputs
        assert_ne!(out3, out7, "different positions should give different outputs");
    }

    // =====================================================================
    // TransformerBlock tests
    // =====================================================================

    #[test]
    fn test_transformer_block_creation() {
        let cfg = tiny_config();
        let w = seeded_weights(&cfg, 42);
        let block = TransformerBlock::new(w, &cfg, 0).unwrap();
        assert_eq!(block.layer_idx, 0);
        assert_eq!(block.kv_cache.cached_len, 0);
    }

    #[test]
    fn test_transformer_block_forward() {
        let cfg = tiny_config();
        let w = seeded_weights(&cfg, 42);
        let mut block = TransformerBlock::new(w, &cfg, 0).unwrap();
        let x = vec![0.1_f32; cfg.hidden_size];
        let out = block.forward(&x, 0, &cfg).unwrap();
        assert_eq!(out.len(), cfg.hidden_size);
        assert_eq!(block.kv_cache.cached_len, 1);
    }

    #[test]
    fn test_transformer_block_reset_cache() {
        let cfg = tiny_config();
        let w = seeded_weights(&cfg, 42);
        let mut block = TransformerBlock::new(w, &cfg, 0).unwrap();
        let x = vec![0.1_f32; cfg.hidden_size];
        block.forward(&x, 0, &cfg).unwrap();
        assert_eq!(block.kv_cache.cached_len, 1);
        block.reset_cache();
        assert_eq!(block.kv_cache.cached_len, 0);
    }

    #[test]
    fn test_transformer_block_invalid_weights() {
        let cfg = tiny_config();
        let mut w = LayerWeights::zeros(&cfg);
        w.wq.push(0.0); // corrupt
        assert!(TransformerBlock::new(w, &cfg, 0).is_err());
    }

    // =====================================================================
    // Matvec reference tests
    // =====================================================================

    #[test]
    fn test_matvec_identity() {
        #[rustfmt::skip]
        let w = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        let x = vec![3.0, 7.0];
        let y = matvec_ref(&w, &x, 2, 2);
        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn test_matvec_known_result() {
        #[rustfmt::skip]
        let w = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        let x = vec![1.0, 1.0];
        let y = matvec_ref(&w, &x, 2, 2);
        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn test_matvec_rectangular() {
        // 3x2 matrix
        #[rustfmt::skip]
        let w = vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ];
        let x = vec![2.0, 3.0];
        let y = matvec_ref(&w, &x, 3, 2);
        assert_eq!(y, vec![2.0, 3.0, 5.0]);
    }
}
