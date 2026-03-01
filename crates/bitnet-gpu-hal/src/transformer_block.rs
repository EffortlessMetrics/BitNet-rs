//! Module stub - implementation pending merge from feature branch
//! Transformer decoder block and full model architecture.
//!
//! Provides [`TransformerConfig`], [`TransformerBlock`], [`TransformerStack`],
//! [`EmbeddingLayer`], [`OutputHead`], and [`TransformerModel`] for assembling
//! a complete decoder-only language model. [`ModelBuilder`] offers a builder
//! pattern for ergonomic construction and [`ForwardPass`] encapsulates the
//! execution context (KV cache, causal mask, position IDs).

use std::fmt;
use std::time::Duration;

// ── Activation & Norm variants ──────────────────────────────────────────────

/// Activation function used in the feed-forward network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Gaussian Error Linear Unit.
    GeLU,
    /// Sigmoid Linear Unit (SwiGLU-style).
    SiLU,
    /// Rectified Linear Unit.
    ReLU,
}

/// Layer normalisation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// Standard Layer Norm.
    LayerNorm,
    /// RMS Norm (no mean subtraction).
    RMSNorm,
}

// ── TransformerConfig ───────────────────────────────────────────────────────

/// Complete configuration for a transformer decoder model.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Number of decoder layers.
    pub num_layers: usize,
    /// Hidden (model) dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Intermediate dimension of the feed-forward network.
    pub intermediate_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum supported sequence length.
    pub max_seq_len: usize,
    /// Activation function for the FFN.
    pub activation: Activation,
    /// Normalisation type.
    pub norm_type: NormType,
    /// Base frequency for Rotary Position Embeddings.
    pub rope_base: f32,
}

impl TransformerConfig {
    /// Validate the configuration, returning an error message on failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_layers == 0 {
            return Err("num_layers must be > 0".into());
        }
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be > 0".into());
        }
        if self.num_heads == 0 {
            return Err("num_heads must be > 0".into());
        }
        if self.head_dim == 0 {
            return Err("head_dim must be > 0".into());
        }
        if self.intermediate_dim == 0 {
            return Err("intermediate_dim must be > 0".into());
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".into());
        }
        if self.max_seq_len == 0 {
            return Err("max_seq_len must be > 0".into());
        }
        if self.hidden_dim != self.num_heads * self.head_dim {
            return Err(format!(
                "hidden_dim ({}) must equal num_heads ({}) * head_dim ({})",
                self.hidden_dim, self.num_heads, self.head_dim,
            ));
        }
        Ok(())
    }

    /// Total number of parameters (rough estimate) across all layers.
    #[allow(clippy::cast_precision_loss)]
    pub const fn estimated_param_count(&self) -> u64 {
        let h = self.hidden_dim as u64;
        let i = self.intermediate_dim as u64;
        let v = self.vocab_size as u64;
        let n = self.num_layers as u64;
        // attention: Q,K,V,O projections per layer
        let attn_per_layer = 4 * h * h;
        // ffn: up + down projections per layer
        let ffn_per_layer = 2 * h * i;
        // embedding + output head
        let embed = v * h;
        n * (attn_per_layer + ffn_per_layer) + 2 * embed
    }
}

/// Create a small config suitable for unit tests.
#[cfg(test)]
const fn test_config() -> TransformerConfig {
    TransformerConfig {
        num_layers: 2,
        hidden_dim: 64,
        num_heads: 4,
        head_dim: 16,
        intermediate_dim: 128,
        vocab_size: 256,
        max_seq_len: 128,
        activation: Activation::SiLU,
        norm_type: NormType::RMSNorm,
        rope_base: 10000.0,
    }
}

// ── Tensor representation ───────────────────────────────────────────────────

/// Lightweight tensor for testing and prototyping.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Shape dimensions, e.g. `[batch, seq_len, hidden_dim]`.
    pub shape: Vec<usize>,
    /// Flat row-major f32 data.
    pub data: Vec<f32>,
}

impl Tensor {
    /// Create a zero-filled tensor with the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self { shape: shape.to_vec(), data: vec![0.0; numel] }
    }

    /// Create a tensor filled with the given value.
    pub fn full(shape: &[usize], value: f32) -> Self {
        let numel: usize = shape.iter().product();
        Self { shape: shape.to_vec(), data: vec![value; numel] }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Element-wise addition of two tensors with matching shapes.
    pub fn add(&self, other: &Self) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err(format!("shape mismatch: {:?} vs {:?}", self.shape, other.shape));
        }
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Ok(Self { shape: self.shape.clone(), data })
    }
}

// ── ResidualConnection ──────────────────────────────────────────────────────

/// Pre-norm residual connection: `output = input + sublayer(norm(input))`.
#[derive(Debug)]
pub struct ResidualConnection {
    /// Normalisation type applied before the sublayer.
    pub norm_type: NormType,
    /// Hidden dimension (for norm weights).
    pub hidden_dim: usize,
}

impl ResidualConnection {
    /// Create a new residual connection.
    pub const fn new(norm_type: NormType, hidden_dim: usize) -> Self {
        Self { norm_type, hidden_dim }
    }

    /// Apply the residual connection.
    ///
    /// `sublayer_fn` receives the normalised input and returns the sublayer
    /// output. The original `input` is then added as the residual.
    pub fn forward<F>(&self, input: &Tensor, sublayer_fn: F) -> Result<Tensor, String>
    where
        F: FnOnce(&Tensor) -> Tensor,
    {
        let normed = self.apply_norm(input);
        let sublayer_out = sublayer_fn(&normed);
        input.add(&sublayer_out)
    }

    /// Stub normalisation — applies identity (real impl normalises in-place).
    #[allow(clippy::unused_self)]
    fn apply_norm(&self, input: &Tensor) -> Tensor {
        // Identity for now; real implementation normalises in-place.
        input.clone()
    }
}

// ── TransformerBlock ────────────────────────────────────────────────────────

/// Single transformer decoder block.
///
/// Architecture: `norm1 → attention → residual → norm2 → ffn → residual`
#[derive(Debug)]
pub struct TransformerBlock {
    /// Layer index within the stack.
    pub layer_idx: usize,
    /// Block configuration.
    pub config: TransformerConfig,
    /// Residual around the attention sublayer.
    pub attn_residual: ResidualConnection,
    /// Residual around the feed-forward sublayer.
    pub ffn_residual: ResidualConnection,
}

impl TransformerBlock {
    /// Create a new block for the given layer index.
    pub const fn new(layer_idx: usize, config: TransformerConfig) -> Self {
        let norm_type = config.norm_type;
        let hidden_dim = config.hidden_dim;
        Self {
            layer_idx,
            config,
            attn_residual: ResidualConnection::new(norm_type, hidden_dim),
            ffn_residual: ResidualConnection::new(norm_type, hidden_dim),
        }
    }

    /// Forward pass through this block.
    ///
    /// `kv_cache` is mutated to append the new key/value entries.
    /// `causal_mask` restricts attention to prior positions.
    #[allow(clippy::needless_pass_by_value)]
    pub fn forward(
        &self,
        input: &Tensor,
        kv_cache: Option<&mut KVCache>,
        causal_mask: Option<&CausalMask>,
    ) -> Result<Tensor, String> {
        // attention sublayer
        let after_attn = self.attn_residual.forward(input, |normed| {
            self.mock_attention(normed, kv_cache.is_some(), causal_mask)
        })?;
        // ffn sublayer
        let after_ffn = self.ffn_residual.forward(&after_attn, |normed| self.mock_ffn(normed))?;
        Ok(after_ffn)
    }

    /// Stub multi-head attention.
    #[allow(clippy::unused_self)]
    fn mock_attention(
        &self,
        input: &Tensor,
        _has_cache: bool,
        _mask: Option<&CausalMask>,
    ) -> Tensor {
        // Returns input scaled by 0.5 to simulate attention output.
        let data: Vec<f32> = input.data.iter().map(|v| v * 0.5).collect();
        Tensor { shape: input.shape.clone(), data }
    }

    /// Stub feed-forward network.
    fn mock_ffn(&self, input: &Tensor) -> Tensor {
        let data: Vec<f32> =
            input.data.iter().map(|v| apply_activation(self.config.activation, *v)).collect();
        Tensor { shape: input.shape.clone(), data }
    }
}

/// Apply the selected activation function element-wise.
fn apply_activation(act: Activation, x: f32) -> f32 {
    match act {
        Activation::GeLU => {
            // Approximate GeLU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            let c = 0.797_884_6; // sqrt(2/π)
            x * 0.5 * (1.0 + (c * (0.044_715 * x * x).mul_add(x, x)).tanh())
        }
        Activation::SiLU => x / (1.0 + (-x).exp()),
        Activation::ReLU => x.max(0.0),
    }
}

// ── KV Cache ────────────────────────────────────────────────────────────────

/// Per-layer key/value cache for autoregressive decoding.
#[derive(Debug, Clone)]
pub struct KVCacheLayer {
    /// Cached key tensor — `[batch, cached_len, head_dim]`.
    pub keys: Tensor,
    /// Cached value tensor — `[batch, cached_len, head_dim]`.
    pub values: Tensor,
    /// Current number of cached positions.
    pub len: usize,
}

/// Full KV cache spanning all layers.
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Per-layer caches.
    pub layers: Vec<KVCacheLayer>,
    /// Maximum sequence length supported.
    pub max_seq_len: usize,
}

impl KVCache {
    /// Allocate a new empty cache for `num_layers` layers.
    pub fn new(num_layers: usize, batch_size: usize, max_seq_len: usize, head_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| KVCacheLayer {
                keys: Tensor::zeros(&[batch_size, 0, head_dim]),
                values: Tensor::zeros(&[batch_size, 0, head_dim]),
                len: 0,
            })
            .collect();
        Self { layers, max_seq_len }
    }

    /// Number of layers in the cache.
    pub const fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Current cached sequence length (from layer 0).
    pub fn cached_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.len)
    }

    /// Append new entries to a specific layer's cache.
    pub fn append(&mut self, layer_idx: usize, new_len: usize) -> Result<(), String> {
        let num = self.layers.len();
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| format!("layer index {layer_idx} out of bounds ({num})"))?;
        let total = layer.len + new_len;
        if total > self.max_seq_len {
            return Err(format!("KV cache overflow: {total} > max_seq_len {}", self.max_seq_len));
        }
        layer.len = total;
        Ok(())
    }

    /// Reset all layers to empty.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.len = 0;
        }
    }
}

// ── CausalMask ──────────────────────────────────────────────────────────────

/// Causal (autoregressive) attention mask.
#[derive(Debug, Clone)]
pub struct CausalMask {
    /// Sequence length this mask covers.
    pub seq_len: usize,
}

impl CausalMask {
    /// Create a causal mask for the given sequence length.
    pub const fn new(seq_len: usize) -> Self {
        Self { seq_len }
    }

    /// Check whether position `query_pos` can attend to `key_pos`.
    pub const fn can_attend(&self, query_pos: usize, key_pos: usize) -> bool {
        key_pos <= query_pos && query_pos < self.seq_len && key_pos < self.seq_len
    }
}

// ── TransformerStack ────────────────────────────────────────────────────────

/// Stack of N transformer blocks with shared config.
#[derive(Debug)]
pub struct TransformerStack {
    /// Ordered decoder blocks.
    pub blocks: Vec<TransformerBlock>,
    /// Shared configuration.
    pub config: TransformerConfig,
}

impl TransformerStack {
    /// Build a stack of `config.num_layers` blocks.
    pub fn new(config: TransformerConfig) -> Result<Self, String> {
        config.validate()?;
        let blocks =
            (0..config.num_layers).map(|i| TransformerBlock::new(i, config.clone())).collect();
        Ok(Self { blocks, config })
    }

    /// Forward pass through all blocks sequentially.
    pub fn forward(
        &self,
        mut hidden: Tensor,
        mut kv_cache: Option<&mut KVCache>,
        causal_mask: Option<&CausalMask>,
    ) -> Result<Tensor, String> {
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, None, causal_mask)?;
            if let Some(ref mut cache) = kv_cache {
                // Record that this layer processed the current sequence step.
                let seq_len = if hidden.shape.len() >= 2 { hidden.shape[1] } else { 1 };
                // Only append on the first pass (cache is empty).
                if cache.layers[i].len == 0 {
                    cache.append(i, seq_len)?;
                }
            }
        }
        Ok(hidden)
    }

    /// Number of blocks in the stack.
    pub const fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

// ── EmbeddingLayer ──────────────────────────────────────────────────────────

/// Token and optional position embedding layer.
#[derive(Debug)]
pub struct EmbeddingLayer {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Maximum sequence length (for position embeddings).
    pub max_seq_len: usize,
    /// Whether to add learned position embeddings.
    pub use_position_embedding: bool,
}

impl EmbeddingLayer {
    /// Create a new embedding layer.
    pub const fn new(
        vocab_size: usize,
        embed_dim: usize,
        max_seq_len: usize,
        use_position_embedding: bool,
    ) -> Self {
        Self { vocab_size, embed_dim, max_seq_len, use_position_embedding }
    }

    /// Look up embeddings for a batch of token IDs.
    ///
    /// `token_ids` shape: `[batch_size, seq_len]` (flattened).
    /// Returns shape `[batch_size, seq_len, embed_dim]`.
    pub fn forward(
        &self,
        token_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor, String> {
        if token_ids.len() != batch_size * seq_len {
            return Err(format!(
                "expected {} token IDs, got {}",
                batch_size * seq_len,
                token_ids.len()
            ));
        }
        for &id in token_ids {
            if (id as usize) >= self.vocab_size {
                return Err(format!(
                    "token ID {id} out of vocabulary range [0, {})",
                    self.vocab_size
                ));
            }
        }
        if seq_len > self.max_seq_len {
            return Err(format!("seq_len {seq_len} exceeds max_seq_len {}", self.max_seq_len));
        }
        // Stub: deterministic embedding based on token ID.
        let numel = batch_size * seq_len * self.embed_dim;
        let mut data = Vec::with_capacity(numel);
        for (pos, &id) in token_ids.iter().enumerate() {
            let seq_pos = pos % seq_len;
            for d in 0..self.embed_dim {
                #[allow(clippy::cast_precision_loss)]
                let token_embed = ((id as f32) + 1.0).mul_add(0.01, (d as f32) * 0.001);
                #[allow(clippy::cast_precision_loss)]
                let pos_embed =
                    if self.use_position_embedding { (seq_pos as f32) * 0.001 } else { 0.0 };
                data.push(token_embed + pos_embed);
            }
        }
        Ok(Tensor { shape: vec![batch_size, seq_len, self.embed_dim], data })
    }
}

// ── OutputHead ──────────────────────────────────────────────────────────────

/// Language model output head: norm → linear → logits.
#[derive(Debug)]
pub struct OutputHead {
    /// Hidden dimension (input).
    pub hidden_dim: usize,
    /// Vocabulary size (output).
    pub vocab_size: usize,
    /// Normalisation type applied before projection.
    pub norm_type: NormType,
}

impl OutputHead {
    /// Create a new output head.
    pub const fn new(hidden_dim: usize, vocab_size: usize, norm_type: NormType) -> Self {
        Self { hidden_dim, vocab_size, norm_type }
    }

    /// Compute logits from the last hidden state.
    ///
    /// Input shape: `[batch_size, seq_len, hidden_dim]`.
    /// Output shape: `[batch_size, seq_len, vocab_size]`.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor, String> {
        if hidden.shape.len() != 3 {
            return Err(format!("expected 3D input [batch, seq, hidden], got {:?}", hidden.shape));
        }
        if hidden.shape[2] != self.hidden_dim {
            return Err(format!(
                "hidden dim mismatch: expected {}, got {}",
                self.hidden_dim, hidden.shape[2]
            ));
        }
        let batch = hidden.shape[0];
        let seq = hidden.shape[1];
        let numel = batch * seq * self.vocab_size;
        let mut data = Vec::with_capacity(numel);
        // Stub projection: each vocab logit is a weighted sum of hidden dims.
        for b in 0..batch {
            for s in 0..seq {
                let offset = (b * seq + s) * self.hidden_dim;
                let hidden_slice = &hidden.data[offset..offset + self.hidden_dim];
                #[allow(clippy::cast_precision_loss)]
                let mean: f32 = hidden_slice.iter().sum::<f32>() / self.hidden_dim as f32;
                for v in 0..self.vocab_size {
                    #[allow(clippy::cast_precision_loss)]
                    let logit = (v as f32).mul_add(0.001, mean);
                    data.push(logit);
                }
            }
        }
        Ok(Tensor { shape: vec![batch, seq, self.vocab_size], data })
    }
}

// ── TransformerModel ────────────────────────────────────────────────────────

/// Complete transformer decoder model.
///
/// `embedding → blocks → output_head`
#[derive(Debug)]
pub struct TransformerModel {
    /// Configuration.
    pub config: TransformerConfig,
    /// Token (+ position) embedding.
    pub embedding: EmbeddingLayer,
    /// Decoder block stack.
    pub stack: TransformerStack,
    /// Output projection head.
    pub output_head: OutputHead,
}

impl TransformerModel {
    /// Forward pass: token IDs → logits.
    pub fn forward(
        &self,
        token_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
        kv_cache: Option<&mut KVCache>,
        causal_mask: Option<&CausalMask>,
    ) -> Result<Tensor, String> {
        let hidden = self.embedding.forward(token_ids, batch_size, seq_len)?;
        let transformed = self.stack.forward(hidden, kv_cache, causal_mask)?;
        self.output_head.forward(&transformed)
    }
}

impl fmt::Display for TransformerModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TransformerModel(layers={}, hidden={}, heads={}, vocab={}, params≈{})",
            self.config.num_layers,
            self.config.hidden_dim,
            self.config.num_heads,
            self.config.vocab_size,
            self.config.estimated_param_count(),
        )
    }
}

// ── ModelBuilder ────────────────────────────────────────────────────────────

/// Builder pattern for constructing a [`TransformerModel`] from config.
#[derive(Debug)]
pub struct ModelBuilder {
    config: TransformerConfig,
    use_position_embedding: bool,
}

impl ModelBuilder {
    /// Start building with the given config.
    pub const fn new(config: TransformerConfig) -> Self {
        Self { config, use_position_embedding: false }
    }

    /// Enable learned position embeddings.
    #[must_use]
    pub const fn with_position_embedding(mut self) -> Self {
        self.use_position_embedding = true;
        self
    }

    /// Build the model, validating the config.
    pub fn build(self) -> Result<TransformerModel, String> {
        self.config.validate()?;
        let embedding = EmbeddingLayer::new(
            self.config.vocab_size,
            self.config.hidden_dim,
            self.config.max_seq_len,
            self.use_position_embedding,
        );
        let stack = TransformerStack::new(self.config.clone())?;
        let output_head =
            OutputHead::new(self.config.hidden_dim, self.config.vocab_size, self.config.norm_type);
        Ok(TransformerModel { config: self.config, embedding, stack, output_head })
    }
}

// ── ForwardPass ─────────────────────────────────────────────────────────────

/// Forward pass execution context.
///
/// Bundles the runtime state required for a single forward pass:
/// KV cache, causal mask, and position IDs.
#[derive(Debug)]
pub struct ForwardPass {
    /// Optional KV cache for autoregressive decoding.
    pub kv_cache: Option<KVCache>,
    /// Optional causal mask.
    pub causal_mask: Option<CausalMask>,
    /// Current position IDs per batch element (for `RoPE`).
    pub position_ids: Vec<usize>,
}

impl ForwardPass {
    /// Create a prefill forward pass context (no cache).
    pub fn prefill(seq_len: usize, batch_size: usize) -> Self {
        let position_ids = (0..batch_size).flat_map(|_| 0..seq_len).collect();
        Self { kv_cache: None, causal_mask: Some(CausalMask::new(seq_len)), position_ids }
    }

    /// Create a decode-step context with an existing KV cache.
    pub fn decode_step(kv_cache: KVCache, next_pos: usize, batch_size: usize) -> Self {
        let total_len = kv_cache.cached_len() + 1;
        Self {
            kv_cache: Some(kv_cache),
            causal_mask: Some(CausalMask::new(total_len)),
            position_ids: vec![next_pos; batch_size],
        }
    }

    /// Execute the forward pass on a model.
    pub fn execute(
        &mut self,
        model: &TransformerModel,
        token_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor, String> {
        model.forward(
            token_ids,
            batch_size,
            seq_len,
            self.kv_cache.as_mut(),
            self.causal_mask.as_ref(),
        )
    }
}

// ── TransformerMetrics ──────────────────────────────────────────────────────

/// Per-layer and aggregate timing metrics for a forward pass.
#[derive(Debug, Clone)]
pub struct TransformerMetrics {
    /// Time spent in each layer.
    pub layer_times: Vec<Duration>,
    /// Total forward pass wall-clock time.
    pub total_forward_time: Duration,
    /// Estimated memory consumption per layer in bytes.
    pub memory_per_layer: Vec<u64>,
}

impl TransformerMetrics {
    /// Create empty metrics for `num_layers` layers.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layer_times: vec![Duration::ZERO; num_layers],
            total_forward_time: Duration::ZERO,
            memory_per_layer: vec![0; num_layers],
        }
    }

    /// Record the time for a specific layer.
    pub fn record_layer(&mut self, layer_idx: usize, elapsed: Duration) {
        if layer_idx < self.layer_times.len() {
            self.layer_times[layer_idx] = elapsed;
        }
    }

    /// Finalize by computing the total from individual layer times.
    pub fn finalize(&mut self) {
        self.total_forward_time = self.layer_times.iter().sum();
    }

    /// Average layer time across all layers.
    pub fn avg_layer_time(&self) -> Duration {
        if self.layer_times.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.layer_times.iter().sum();
        #[allow(clippy::cast_possible_truncation)]
        let divisor = self.layer_times.len() as u32;
        total / divisor
    }

    /// Maximum layer time.
    pub fn max_layer_time(&self) -> Duration {
        self.layer_times.iter().copied().max().unwrap_or(Duration::ZERO)
    }

    /// Record memory for a specific layer.
    pub fn record_memory(&mut self, layer_idx: usize, bytes: u64) {
        if layer_idx < self.memory_per_layer.len() {
            self.memory_per_layer[layer_idx] = bytes;
        }
    }

    /// Total memory across all layers.
    pub fn total_memory(&self) -> u64 {
        self.memory_per_layer.iter().sum()
    }
}

impl fmt::Display for TransformerMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Metrics(layers={}, total={:?}, avg={:?}, max={:?}, mem={}B)",
            self.layer_times.len(),
            self.total_forward_time,
            self.avg_layer_time(),
            self.max_layer_time(),
            self.total_memory(),
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config tests ────────────────────────────────────────────────────

    #[test]
    fn config_valid() {
        assert!(test_config().validate().is_ok());
    }

    #[test]
    fn config_zero_layers_invalid() {
        let mut c = test_config();
        c.num_layers = 0;
        assert!(c.validate().unwrap_err().contains("num_layers"));
    }

    #[test]
    fn config_zero_hidden_dim_invalid() {
        let mut c = test_config();
        c.hidden_dim = 0;
        assert!(c.validate().unwrap_err().contains("hidden_dim"));
    }

    #[test]
    fn config_zero_num_heads_invalid() {
        let mut c = test_config();
        c.num_heads = 0;
        assert!(c.validate().unwrap_err().contains("num_heads"));
    }

    #[test]
    fn config_zero_head_dim_invalid() {
        let mut c = test_config();
        c.head_dim = 0;
        assert!(c.validate().unwrap_err().contains("head_dim"));
    }

    #[test]
    fn config_zero_intermediate_dim_invalid() {
        let mut c = test_config();
        c.intermediate_dim = 0;
        assert!(c.validate().unwrap_err().contains("intermediate_dim"));
    }

    #[test]
    fn config_zero_vocab_size_invalid() {
        let mut c = test_config();
        c.vocab_size = 0;
        assert!(c.validate().unwrap_err().contains("vocab_size"));
    }

    #[test]
    fn config_zero_max_seq_len_invalid() {
        let mut c = test_config();
        c.max_seq_len = 0;
        assert!(c.validate().unwrap_err().contains("max_seq_len"));
    }

    #[test]
    fn config_head_dim_mismatch_invalid() {
        let mut c = test_config();
        c.head_dim = 32; // 4 * 32 = 128 ≠ 64
        assert!(c.validate().unwrap_err().contains("hidden_dim"));
    }

    #[test]
    fn config_estimated_param_count() {
        let c = test_config();
        assert!(c.estimated_param_count() > 0);
    }

    #[test]
    fn config_param_count_scales_with_layers() {
        let mut c1 = test_config();
        c1.num_layers = 2;
        let mut c2 = test_config();
        c2.num_layers = 4;
        assert!(c2.estimated_param_count() > c1.estimated_param_count());
    }

    #[test]
    fn config_clone() {
        let c = test_config();
        let c2 = c.clone();
        assert_eq!(c.num_layers, c2.num_layers);
        assert_eq!(c.hidden_dim, c2.hidden_dim);
    }

    #[test]
    fn config_debug() {
        let c = test_config();
        let dbg = format!("{c:?}");
        assert!(dbg.contains("TransformerConfig"));
    }

    // ── Activation tests ────────────────────────────────────────────────

    #[test]
    fn activation_relu_positive() {
        assert!((apply_activation(Activation::ReLU, 1.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn activation_relu_negative() {
        assert!((apply_activation(Activation::ReLU, -1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn activation_relu_zero() {
        assert!((apply_activation(Activation::ReLU, 0.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn activation_silu_zero() {
        // SiLU(0) = 0 / (1 + 1) = 0
        assert!((apply_activation(Activation::SiLU, 0.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn activation_silu_positive() {
        let y = apply_activation(Activation::SiLU, 2.0);
        assert!(y > 0.0);
        assert!(y < 2.0); // SiLU(x) < x for positive x
    }

    #[test]
    fn activation_gelu_zero() {
        assert!((apply_activation(Activation::GeLU, 0.0)).abs() < 0.01);
    }

    #[test]
    fn activation_gelu_positive() {
        let y = apply_activation(Activation::GeLU, 2.0);
        assert!(y > 0.0);
    }

    // ── Tensor tests ────────────────────────────────────────────────────

    #[test]
    fn tensor_zeros_shape() {
        let t = Tensor::zeros(&[2, 3, 4]);
        assert_eq!(t.shape, vec![2, 3, 4]);
        assert_eq!(t.numel(), 24);
        assert!(t.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn tensor_full() {
        let t = Tensor::full(&[2, 2], 1.234);
        assert_eq!(t.numel(), 4);
        assert!(t.data.iter().all(|&v| (v - 1.234).abs() < f32::EPSILON));
    }

    #[test]
    fn tensor_add_ok() {
        let a = Tensor::full(&[2, 3], 1.0);
        let b = Tensor::full(&[2, 3], 2.0);
        let c = a.add(&b).unwrap();
        assert!(c.data.iter().all(|&v| (v - 3.0).abs() < f32::EPSILON));
    }

    #[test]
    fn tensor_add_shape_mismatch() {
        let a = Tensor::zeros(&[2, 3]);
        let b = Tensor::zeros(&[3, 2]);
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn tensor_empty() {
        let t = Tensor::zeros(&[0]);
        assert_eq!(t.numel(), 0);
        assert!(t.data.is_empty());
    }

    // ── ResidualConnection tests ────────────────────────────────────────

    #[test]
    fn residual_identity_sublayer() {
        let res = ResidualConnection::new(NormType::RMSNorm, 4);
        let input = Tensor::full(&[1, 1, 4], 1.0);
        let out = res.forward(&input, |_normed| Tensor::zeros(&[1, 1, 4])).unwrap();
        // output = input + 0 = input
        assert!(out.data.iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
    }

    #[test]
    fn residual_additive() {
        let res = ResidualConnection::new(NormType::LayerNorm, 4);
        let input = Tensor::full(&[1, 1, 4], 2.0);
        let out = res.forward(&input, |_| Tensor::full(&[1, 1, 4], 3.0)).unwrap();
        // output = 2.0 + 3.0 = 5.0
        assert!(out.data.iter().all(|&v| (v - 5.0).abs() < f32::EPSILON));
    }

    #[test]
    fn residual_shape_preserved() {
        let res = ResidualConnection::new(NormType::RMSNorm, 8);
        let input = Tensor::zeros(&[2, 4, 8]);
        let out = res.forward(&input, std::clone::Clone::clone).unwrap();
        assert_eq!(out.shape, vec![2, 4, 8]);
    }

    // ── TransformerBlock tests ──────────────────────────────────────────

    #[test]
    fn block_forward_shape() {
        let cfg = test_config();
        let block = TransformerBlock::new(0, cfg.clone());
        let input = Tensor::full(&[1, 4, cfg.hidden_dim], 1.0);
        let out = block.forward(&input, None, None).unwrap();
        assert_eq!(out.shape, input.shape);
    }

    #[test]
    fn block_forward_nonzero_output() {
        let cfg = test_config();
        let block = TransformerBlock::new(0, cfg.clone());
        let input = Tensor::full(&[1, 2, cfg.hidden_dim], 1.0);
        let out = block.forward(&input, None, None).unwrap();
        assert!(out.data.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn block_forward_with_mask() {
        let cfg = test_config();
        let block = TransformerBlock::new(0, cfg.clone());
        let input = Tensor::full(&[1, 4, cfg.hidden_dim], 1.0);
        let mask = CausalMask::new(4);
        let out = block.forward(&input, None, Some(&mask)).unwrap();
        assert_eq!(out.shape, input.shape);
    }

    #[test]
    fn block_layer_idx() {
        let cfg = test_config();
        let block = TransformerBlock::new(5, cfg);
        assert_eq!(block.layer_idx, 5);
    }

    #[test]
    fn block_different_activations() {
        for act in [Activation::GeLU, Activation::SiLU, Activation::ReLU] {
            let mut cfg = test_config();
            cfg.activation = act;
            let block = TransformerBlock::new(0, cfg.clone());
            let input = Tensor::full(&[1, 1, cfg.hidden_dim], 1.0);
            assert!(block.forward(&input, None, None).is_ok());
        }
    }

    // ── KVCache tests ───────────────────────────────────────────────────

    #[test]
    fn kv_cache_new() {
        let cache = KVCache::new(4, 1, 128, 16);
        assert_eq!(cache.num_layers(), 4);
        assert_eq!(cache.cached_len(), 0);
    }

    #[test]
    fn kv_cache_append() {
        let mut cache = KVCache::new(2, 1, 128, 16);
        cache.append(0, 10).unwrap();
        assert_eq!(cache.layers[0].len, 10);
        assert_eq!(cache.layers[1].len, 0);
    }

    #[test]
    fn kv_cache_append_overflow() {
        let mut cache = KVCache::new(1, 1, 10, 16);
        assert!(cache.append(0, 11).is_err());
    }

    #[test]
    fn kv_cache_append_out_of_bounds() {
        let mut cache = KVCache::new(2, 1, 128, 16);
        assert!(cache.append(5, 1).is_err());
    }

    #[test]
    fn kv_cache_clear() {
        let mut cache = KVCache::new(2, 1, 128, 16);
        cache.append(0, 5).unwrap();
        cache.append(1, 3).unwrap();
        cache.clear();
        assert_eq!(cache.cached_len(), 0);
        assert_eq!(cache.layers[1].len, 0);
    }

    #[test]
    fn kv_cache_incremental_append() {
        let mut cache = KVCache::new(1, 1, 100, 16);
        cache.append(0, 30).unwrap();
        cache.append(0, 30).unwrap();
        assert_eq!(cache.layers[0].len, 60);
    }

    #[test]
    fn kv_cache_incremental_overflow() {
        let mut cache = KVCache::new(1, 1, 50, 16);
        cache.append(0, 30).unwrap();
        assert!(cache.append(0, 30).is_err());
    }

    // ── CausalMask tests ────────────────────────────────────────────────

    #[test]
    fn causal_mask_self_attend() {
        let mask = CausalMask::new(4);
        assert!(mask.can_attend(0, 0));
        assert!(mask.can_attend(3, 3));
    }

    #[test]
    fn causal_mask_attend_past() {
        let mask = CausalMask::new(4);
        assert!(mask.can_attend(2, 0));
        assert!(mask.can_attend(3, 1));
    }

    #[test]
    fn causal_mask_no_future() {
        let mask = CausalMask::new(4);
        assert!(!mask.can_attend(0, 1));
        assert!(!mask.can_attend(1, 3));
    }

    #[test]
    fn causal_mask_out_of_bounds() {
        let mask = CausalMask::new(4);
        assert!(!mask.can_attend(4, 0));
        assert!(!mask.can_attend(0, 4));
    }

    #[test]
    fn causal_mask_single_pos() {
        let mask = CausalMask::new(1);
        assert!(mask.can_attend(0, 0));
        assert!(!mask.can_attend(1, 0));
    }

    // ── TransformerStack tests ──────────────────────────────────────────

    #[test]
    fn stack_num_blocks() {
        let stack = TransformerStack::new(test_config()).unwrap();
        assert_eq!(stack.num_blocks(), 2);
    }

    #[test]
    fn stack_forward_shape() {
        let cfg = test_config();
        let stack = TransformerStack::new(cfg.clone()).unwrap();
        let input = Tensor::full(&[1, 4, cfg.hidden_dim], 1.0);
        let out = stack.forward(input, None, None).unwrap();
        assert_eq!(out.shape, vec![1, 4, cfg.hidden_dim]);
    }

    #[test]
    fn stack_forward_with_cache() {
        let cfg = test_config();
        let stack = TransformerStack::new(cfg.clone()).unwrap();
        let input = Tensor::full(&[1, 4, cfg.hidden_dim], 1.0);
        let mut cache = KVCache::new(cfg.num_layers, 1, cfg.max_seq_len, cfg.head_dim);
        let out = stack.forward(input, Some(&mut cache), None).unwrap();
        assert_eq!(out.shape, vec![1, 4, cfg.hidden_dim]);
        // Cache should have been populated.
        assert_eq!(cache.layers[0].len, 4);
    }

    #[test]
    fn stack_forward_with_mask_and_cache() {
        let cfg = test_config();
        let stack = TransformerStack::new(cfg.clone()).unwrap();
        let input = Tensor::full(&[1, 4, cfg.hidden_dim], 1.0);
        let mut cache = KVCache::new(cfg.num_layers, 1, cfg.max_seq_len, cfg.head_dim);
        let mask = CausalMask::new(4);
        let out = stack.forward(input, Some(&mut cache), Some(&mask)).unwrap();
        assert_eq!(out.shape, vec![1, 4, cfg.hidden_dim]);
    }

    #[test]
    fn stack_invalid_config() {
        let mut c = test_config();
        c.num_layers = 0;
        assert!(TransformerStack::new(c).is_err());
    }

    #[test]
    fn stack_single_layer() {
        let mut cfg = test_config();
        cfg.num_layers = 1;
        let stack = TransformerStack::new(cfg.clone()).unwrap();
        assert_eq!(stack.num_blocks(), 1);
        let input = Tensor::full(&[1, 2, cfg.hidden_dim], 1.0);
        assert!(stack.forward(input, None, None).is_ok());
    }

    // ── EmbeddingLayer tests ────────────────────────────────────────────

    #[test]
    fn embedding_output_shape() {
        let emb = EmbeddingLayer::new(256, 64, 128, false);
        let ids = vec![1, 2, 3, 4];
        let out = emb.forward(&ids, 1, 4).unwrap();
        assert_eq!(out.shape, vec![1, 4, 64]);
    }

    #[test]
    fn embedding_batch() {
        let emb = EmbeddingLayer::new(256, 64, 128, false);
        let ids = vec![1, 2, 3, 4, 5, 6];
        let out = emb.forward(&ids, 2, 3).unwrap();
        assert_eq!(out.shape, vec![2, 3, 64]);
    }

    #[test]
    fn embedding_position_adds_offset() {
        let emb_no = EmbeddingLayer::new(256, 4, 128, false);
        let emb_yes = EmbeddingLayer::new(256, 4, 128, true);
        let ids = vec![1, 2];
        let out_no = emb_no.forward(&ids, 1, 2).unwrap();
        let out_yes = emb_yes.forward(&ids, 1, 2).unwrap();
        // Position 0 should be the same; position 1 should differ.
        assert!((out_no.data[0] - out_yes.data[0]).abs() < f32::EPSILON);
        // Second position has pos_embed added.
        assert!((out_no.data[4] - out_yes.data[4]).abs() > f32::EPSILON);
    }

    #[test]
    fn embedding_oov_rejected() {
        let emb = EmbeddingLayer::new(10, 4, 128, false);
        let ids = vec![10]; // out of vocab
        assert!(emb.forward(&ids, 1, 1).is_err());
    }

    #[test]
    fn embedding_wrong_count() {
        let emb = EmbeddingLayer::new(256, 64, 128, false);
        let ids = vec![1, 2, 3]; // 3 ids but batch*seq = 4
        assert!(emb.forward(&ids, 2, 2).is_err());
    }

    #[test]
    fn embedding_seq_too_long() {
        let emb = EmbeddingLayer::new(256, 64, 8, false);
        let ids = vec![0; 10];
        assert!(emb.forward(&ids, 1, 10).is_err());
    }

    #[test]
    fn embedding_single_token() {
        let emb = EmbeddingLayer::new(256, 64, 128, false);
        let ids = vec![42];
        let out = emb.forward(&ids, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 64]);
        assert!(out.data.iter().any(|&v| v != 0.0));
    }

    // ── OutputHead tests ────────────────────────────────────────────────

    #[test]
    fn output_head_shape() {
        let head = OutputHead::new(64, 256, NormType::RMSNorm);
        let hidden = Tensor::full(&[1, 4, 64], 1.0);
        let logits = head.forward(&hidden).unwrap();
        assert_eq!(logits.shape, vec![1, 4, 256]);
    }

    #[test]
    fn output_head_batch() {
        let head = OutputHead::new(64, 256, NormType::RMSNorm);
        let hidden = Tensor::full(&[2, 3, 64], 1.0);
        let logits = head.forward(&hidden).unwrap();
        assert_eq!(logits.shape, vec![2, 3, 256]);
    }

    #[test]
    fn output_head_wrong_rank() {
        let head = OutputHead::new(64, 256, NormType::RMSNorm);
        let hidden = Tensor::full(&[64], 1.0);
        assert!(head.forward(&hidden).is_err());
    }

    #[test]
    fn output_head_wrong_hidden_dim() {
        let head = OutputHead::new(64, 256, NormType::RMSNorm);
        let hidden = Tensor::full(&[1, 4, 32], 1.0);
        assert!(head.forward(&hidden).is_err());
    }

    #[test]
    fn output_head_single_position() {
        let head = OutputHead::new(64, 256, NormType::RMSNorm);
        let hidden = Tensor::full(&[1, 1, 64], 0.5);
        let logits = head.forward(&hidden).unwrap();
        assert_eq!(logits.shape, vec![1, 1, 256]);
        // Logits should be monotonically increasing (mean + v*0.001).
        assert!(logits.data[0] < logits.data[255]);
    }

    // ── Full model tests ────────────────────────────────────────────────

    #[test]
    fn model_forward() {
        let model = ModelBuilder::new(test_config()).build().unwrap();
        let ids = vec![1, 2, 3, 4];
        let out = model.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out.shape, vec![1, 4, 256]);
    }

    #[test]
    fn model_forward_batch() {
        let model = ModelBuilder::new(test_config()).build().unwrap();
        let ids = vec![1, 2, 3, 4, 5, 6];
        let out = model.forward(&ids, 2, 3, None, None).unwrap();
        assert_eq!(out.shape, vec![2, 3, 256]);
    }

    #[test]
    fn model_forward_with_kv_cache() {
        let cfg = test_config();
        let model = ModelBuilder::new(cfg.clone()).build().unwrap();
        let ids = vec![1, 2, 3, 4];
        let mut cache = KVCache::new(cfg.num_layers, 1, cfg.max_seq_len, cfg.head_dim);
        let out = model.forward(&ids, 1, 4, Some(&mut cache), None).unwrap();
        assert_eq!(out.shape, vec![1, 4, 256]);
    }

    #[test]
    fn model_forward_with_mask() {
        let model = ModelBuilder::new(test_config()).build().unwrap();
        let ids = vec![1, 2, 3, 4];
        let mask = CausalMask::new(4);
        let out = model.forward(&ids, 1, 4, None, Some(&mask)).unwrap();
        assert_eq!(out.shape, vec![1, 4, 256]);
    }

    #[test]
    fn model_display() {
        let model = ModelBuilder::new(test_config()).build().unwrap();
        let display = format!("{model}");
        assert!(display.contains("TransformerModel"));
        assert!(display.contains("layers=2"));
    }

    #[test]
    fn model_debug() {
        let model = ModelBuilder::new(test_config()).build().unwrap();
        let dbg = format!("{model:?}");
        assert!(dbg.contains("TransformerModel"));
    }

    // ── ModelBuilder tests ──────────────────────────────────────────────

    #[test]
    fn builder_default_no_position_embed() {
        let model = ModelBuilder::new(test_config()).build().unwrap();
        assert!(!model.embedding.use_position_embedding);
    }

    #[test]
    fn builder_with_position_embed() {
        let model = ModelBuilder::new(test_config()).with_position_embedding().build().unwrap();
        assert!(model.embedding.use_position_embedding);
    }

    #[test]
    fn builder_invalid_config() {
        let mut c = test_config();
        c.num_layers = 0;
        assert!(ModelBuilder::new(c).build().is_err());
    }

    #[test]
    fn builder_debug() {
        let b = ModelBuilder::new(test_config());
        let dbg = format!("{b:?}");
        assert!(dbg.contains("ModelBuilder"));
    }

    // ── ForwardPass tests ───────────────────────────────────────────────

    #[test]
    fn forward_pass_prefill() {
        let fp = ForwardPass::prefill(8, 1);
        assert!(fp.kv_cache.is_none());
        assert!(fp.causal_mask.is_some());
        assert_eq!(fp.position_ids.len(), 8);
    }

    #[test]
    fn forward_pass_prefill_batch() {
        let fp = ForwardPass::prefill(4, 2);
        assert_eq!(fp.position_ids.len(), 8);
        // Positions: [0,1,2,3, 0,1,2,3]
        assert_eq!(fp.position_ids[0], 0);
        assert_eq!(fp.position_ids[4], 0);
    }

    #[test]
    fn forward_pass_decode_step() {
        let cache = KVCache::new(2, 1, 128, 16);
        let fp = ForwardPass::decode_step(cache, 5, 1);
        assert!(fp.kv_cache.is_some());
        assert_eq!(fp.position_ids, vec![5]);
    }

    #[test]
    fn forward_pass_execute() {
        let model = ModelBuilder::new(test_config()).build().unwrap();
        let mut fp = ForwardPass::prefill(4, 1);
        let ids = vec![1, 2, 3, 4];
        let out = fp.execute(&model, &ids, 1, 4).unwrap();
        assert_eq!(out.shape, vec![1, 4, 256]);
    }

    #[test]
    fn forward_pass_decode_execute() {
        let cfg = test_config();
        let model = ModelBuilder::new(cfg.clone()).build().unwrap();
        let cache = KVCache::new(cfg.num_layers, 1, cfg.max_seq_len, cfg.head_dim);
        let mut fp = ForwardPass::decode_step(cache, 0, 1);
        let ids = vec![42];
        let out = fp.execute(&model, &ids, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 256]);
    }

    // ── TransformerMetrics tests ────────────────────────────────────────

    #[test]
    fn metrics_new() {
        let m = TransformerMetrics::new(4);
        assert_eq!(m.layer_times.len(), 4);
        assert_eq!(m.total_forward_time, Duration::ZERO);
    }

    #[test]
    fn metrics_record_layer() {
        let mut m = TransformerMetrics::new(2);
        m.record_layer(0, Duration::from_millis(10));
        m.record_layer(1, Duration::from_millis(20));
        assert_eq!(m.layer_times[0], Duration::from_millis(10));
        assert_eq!(m.layer_times[1], Duration::from_millis(20));
    }

    #[test]
    fn metrics_record_out_of_bounds() {
        let mut m = TransformerMetrics::new(1);
        m.record_layer(5, Duration::from_millis(10)); // should be a no-op
        assert_eq!(m.layer_times[0], Duration::ZERO);
    }

    #[test]
    fn metrics_finalize() {
        let mut m = TransformerMetrics::new(2);
        m.record_layer(0, Duration::from_millis(10));
        m.record_layer(1, Duration::from_millis(20));
        m.finalize();
        assert_eq!(m.total_forward_time, Duration::from_millis(30));
    }

    #[test]
    fn metrics_avg_layer_time() {
        let mut m = TransformerMetrics::new(2);
        m.record_layer(0, Duration::from_millis(10));
        m.record_layer(1, Duration::from_millis(30));
        assert_eq!(m.avg_layer_time(), Duration::from_millis(20));
    }

    #[test]
    fn metrics_avg_empty() {
        let m = TransformerMetrics::new(0);
        assert_eq!(m.avg_layer_time(), Duration::ZERO);
    }

    #[test]
    fn metrics_max_layer_time() {
        let mut m = TransformerMetrics::new(3);
        m.record_layer(0, Duration::from_millis(5));
        m.record_layer(1, Duration::from_millis(50));
        m.record_layer(2, Duration::from_millis(20));
        assert_eq!(m.max_layer_time(), Duration::from_millis(50));
    }

    #[test]
    fn metrics_max_empty() {
        let m = TransformerMetrics::new(0);
        assert_eq!(m.max_layer_time(), Duration::ZERO);
    }

    #[test]
    fn metrics_memory() {
        let mut m = TransformerMetrics::new(2);
        m.record_memory(0, 1024);
        m.record_memory(1, 2048);
        assert_eq!(m.total_memory(), 3072);
    }

    #[test]
    fn metrics_memory_out_of_bounds() {
        let mut m = TransformerMetrics::new(1);
        m.record_memory(5, 999); // no-op
        assert_eq!(m.total_memory(), 0);
    }

    #[test]
    fn metrics_display() {
        let m = TransformerMetrics::new(2);
        let display = format!("{m}");
        assert!(display.contains("Metrics"));
        assert!(display.contains("layers=2"));
    }

    #[test]
    fn metrics_clone() {
        let mut m = TransformerMetrics::new(2);
        m.record_layer(0, Duration::from_millis(42));
        let m2 = m.clone();
        assert_eq!(m.layer_times[0], m2.layer_times[0]);
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn single_layer_model() {
        let mut cfg = test_config();
        cfg.num_layers = 1;
        let model = ModelBuilder::new(cfg).build().unwrap();
        let ids = vec![0];
        let out = model.forward(&ids, 1, 1, None, None).unwrap();
        assert_eq!(out.shape, vec![1, 1, 256]);
    }

    #[test]
    fn large_batch() {
        let cfg = test_config();
        let model = ModelBuilder::new(cfg).build().unwrap();
        let ids = vec![0; 8 * 4]; // batch=8, seq=4
        let out = model.forward(&ids, 8, 4, None, None).unwrap();
        assert_eq!(out.shape, vec![8, 4, 256]);
    }

    #[test]
    fn kv_cache_multi_step() {
        let cfg = test_config();
        let mut cache = KVCache::new(cfg.num_layers, 1, cfg.max_seq_len, cfg.head_dim);
        // Simulate multiple decode steps.
        for i in 0..10 {
            for layer in 0..cfg.num_layers {
                cache.append(layer, 1).unwrap();
            }
            assert_eq!(cache.cached_len(), i + 1);
        }
    }

    #[test]
    fn causal_mask_full_sequence() {
        let mask = CausalMask::new(16);
        // Every position can attend to itself and all prior positions.
        for q in 0..16 {
            for k in 0..16 {
                assert_eq!(mask.can_attend(q, k), k <= q);
            }
        }
    }

    #[test]
    fn different_norm_types() {
        for norm in [NormType::LayerNorm, NormType::RMSNorm] {
            let mut cfg = test_config();
            cfg.norm_type = norm;
            let model = ModelBuilder::new(cfg).build().unwrap();
            let ids = vec![1, 2];
            assert!(model.forward(&ids, 1, 2, None, None).is_ok());
        }
    }

    #[test]
    fn embedding_boundary_token_id() {
        let emb = EmbeddingLayer::new(256, 4, 128, false);
        let ids = vec![0, 255]; // min and max valid IDs
        assert!(emb.forward(&ids, 1, 2).is_ok());
    }

    #[test]
    fn deterministic_forward() {
        let model = ModelBuilder::new(test_config()).build().unwrap();
        let ids = vec![1, 2, 3, 4];
        let out1 = model.forward(&ids, 1, 4, None, None).unwrap();
        let out2 = model.forward(&ids, 1, 4, None, None).unwrap();
        assert_eq!(out1.data, out2.data);
    }

    #[test]
    fn forward_pass_position_ids_sequential() {
        let fp = ForwardPass::prefill(5, 1);
        assert_eq!(fp.position_ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn norm_type_equality() {
        assert_eq!(NormType::LayerNorm, NormType::LayerNorm);
        assert_ne!(NormType::LayerNorm, NormType::RMSNorm);
    }

    #[test]
    fn activation_equality() {
        assert_eq!(Activation::GeLU, Activation::GeLU);
        assert_ne!(Activation::GeLU, Activation::ReLU);
    }
}
