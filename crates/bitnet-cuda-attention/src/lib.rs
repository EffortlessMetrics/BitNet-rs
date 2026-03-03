//! CUDA multi-head attention kernels for `BitNet` LLM inference.
//!
//! This crate provides scaffold types and interfaces for GPU-accelerated attention
//! computation, including standard multi-head attention, flash attention v2, and
//! masked attention with key-value cache integration.
//!
//! # Overview
//!
//! - [`AttentionConfig`] — per-layer attention parameters (heads, dims, scaling).
//! - [`AttentionMask`] — causal, full, or custom mask patterns.
//! - [`KvCacheEntry`] / [`KvCacheConfig`] — key-value cache types for incremental
//!   decoding.
//! - [`FlashAttentionConfig`] — flash attention v2 tile/block parameters.
//! - [`AttentionScore`] — raw and softmax-normalised attention scores.
//! - [`MultiHeadAttentionKernel`] / [`FlashAttentionV2Kernel`] — kernel launch
//!   descriptors.
//! - [`AttentionOutput`] — result container for attention forward passes.

use std::fmt;

// ---------------------------------------------------------------------------
// Attention configuration
// ---------------------------------------------------------------------------

/// Configuration for a single multi-head attention layer.
///
/// Holds the static geometry that does not change between forward passes.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimensionality of each head (`head_dim` = `model_dim` / `num_heads`).
    pub head_dim: usize,
    /// Total model / hidden dimension.
    pub model_dim: usize,
    /// Number of key-value heads (for grouped-query attention).
    /// When equal to `num_heads` this is standard MHA.
    pub num_kv_heads: usize,
    /// Softmax temperature scaling factor (typically 1 / sqrt(`head_dim`)).
    pub scale: f32,
    /// Whether to apply causal masking by default.
    pub causal: bool,
    /// Optional dropout probability (0.0 means disabled).
    pub dropout_prob: f32,
}

impl AttentionConfig {
    /// Create a new configuration with the given head geometry.
    ///
    /// `scale` is derived as 1 / sqrt(`head_dim`) when not overridden.
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        #[expect(clippy::cast_precision_loss, reason = "head_dim is small enough for f32")]
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            num_heads,
            head_dim,
            model_dim: num_heads * head_dim,
            num_kv_heads: num_heads,
            scale,
            causal: true,
            dropout_prob: 0.0,
        }
    }

    /// Create a grouped-query attention configuration.
    #[must_use]
    pub const fn with_gqa(mut self, num_kv_heads: usize) -> Self {
        self.num_kv_heads = num_kv_heads;
        self
    }

    /// Override the softmax scaling factor.
    #[must_use]
    pub const fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set the causal masking flag.
    #[must_use]
    pub const fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set dropout probability.
    #[must_use]
    pub const fn with_dropout(mut self, prob: f32) -> Self {
        self.dropout_prob = prob;
        self
    }

    /// Validate that the configuration is internally consistent.
    pub fn validate(&self) -> Result<(), AttentionError> {
        if self.num_heads == 0 {
            return Err(AttentionError::InvalidConfig("num_heads must be > 0".into()));
        }
        if self.head_dim == 0 {
            return Err(AttentionError::InvalidConfig("head_dim must be > 0".into()));
        }
        if self.model_dim != self.num_heads * self.head_dim {
            return Err(AttentionError::InvalidConfig(
                "model_dim must equal num_heads * head_dim".into(),
            ));
        }
        if self.num_kv_heads == 0 {
            return Err(AttentionError::InvalidConfig("num_kv_heads must be > 0".into()));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(AttentionError::InvalidConfig(
                "num_heads must be divisible by num_kv_heads".into(),
            ));
        }
        if self.scale <= 0.0 || !self.scale.is_finite() {
            return Err(AttentionError::InvalidConfig("scale must be positive and finite".into()));
        }
        if !(0.0..=1.0).contains(&self.dropout_prob) {
            return Err(AttentionError::InvalidConfig("dropout_prob must be in [0.0, 1.0]".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Attention masks
// ---------------------------------------------------------------------------

/// Attention mask pattern applied during score computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionMask {
    /// No masking — all positions attend to all positions.
    None,
    /// Lower-triangular causal mask (autoregressive).
    Causal,
    /// Explicit boolean mask: `true` = attend, `false` = masked.
    /// Shape: `[seq_len_q, seq_len_kv]`.
    Custom(Vec<Vec<bool>>),
    /// Sliding-window local attention with a fixed window size.
    SlidingWindow {
        /// Maximum number of past tokens each position can attend to.
        window_size: usize,
    },
    /// Prefix mask: first `prefix_len` tokens attend to everything;
    /// remaining tokens use causal masking.
    Prefix {
        /// Number of prefix tokens with full bidirectional attention.
        prefix_len: usize,
    },
}

impl AttentionMask {
    /// Returns `true` if position `q` may attend to position `kv` under this mask.
    pub fn allows(&self, q: usize, kv: usize, seq_len_q: usize) -> bool {
        match self {
            Self::None => true,
            Self::Causal => kv <= q,
            Self::Custom(mask) => mask.get(q).and_then(|row| row.get(kv)).copied().unwrap_or(false),
            Self::SlidingWindow { window_size } => kv <= q && (q - kv) < *window_size,
            Self::Prefix { prefix_len } => {
                let _ = seq_len_q;
                if q < *prefix_len { true } else { kv <= q }
            }
        }
    }

    /// Return the number of allowed (non-masked) positions for query position `q`.
    pub fn count_allowed(&self, q: usize, seq_len_kv: usize, seq_len_q: usize) -> usize {
        (0..seq_len_kv).filter(|&kv| self.allows(q, kv, seq_len_q)).count()
    }
}

// ---------------------------------------------------------------------------
// Key-value cache
// ---------------------------------------------------------------------------

/// Precision for cached key/value tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvPrecision {
    /// Full 32-bit float.
    F32,
    /// Half-precision 16-bit float.
    F16,
    /// Brain float 16.
    Bf16,
    /// 8-bit quantised cache entries.
    Int8,
}

impl fmt::Display for KvPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::Bf16 => write!(f, "bf16"),
            Self::Int8 => write!(f, "int8"),
        }
    }
}

/// Configuration for the key-value cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheConfig {
    /// Maximum sequence length the cache can hold.
    pub max_seq_len: usize,
    /// Number of layers in the model.
    pub num_layers: usize,
    /// Number of KV heads per layer.
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Storage precision.
    pub precision: KvPrecision,
    /// Whether the cache uses paged (block) allocation.
    pub paged: bool,
    /// Block size for paged allocation (only meaningful when `paged` is true).
    pub block_size: usize,
}

impl KvCacheConfig {
    /// Create a simple (non-paged) cache configuration.
    pub const fn new(
        max_seq_len: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            max_seq_len,
            num_layers,
            num_kv_heads,
            head_dim,
            precision: KvPrecision::F16,
            paged: false,
            block_size: 64,
        }
    }

    /// Compute the memory footprint of a **single layer** in bytes.
    pub const fn layer_bytes(&self) -> usize {
        let elem_bytes = match self.precision {
            KvPrecision::F32 => 4,
            KvPrecision::F16 | KvPrecision::Bf16 => 2,
            KvPrecision::Int8 => 1,
        };
        // key + value, each: [max_seq_len, num_kv_heads, head_dim]
        2 * self.max_seq_len * self.num_kv_heads * self.head_dim * elem_bytes
    }

    /// Total memory across all layers.
    pub const fn total_bytes(&self) -> usize {
        self.num_layers * self.layer_bytes()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), AttentionError> {
        if self.max_seq_len == 0 {
            return Err(AttentionError::InvalidConfig("max_seq_len must be > 0".into()));
        }
        if self.num_layers == 0 {
            return Err(AttentionError::InvalidConfig("num_layers must be > 0".into()));
        }
        if self.num_kv_heads == 0 {
            return Err(AttentionError::InvalidConfig("num_kv_heads must be > 0".into()));
        }
        if self.head_dim == 0 {
            return Err(AttentionError::InvalidConfig("head_dim must be > 0".into()));
        }
        if self.paged && self.block_size == 0 {
            return Err(AttentionError::InvalidConfig("block_size must be > 0 when paged".into()));
        }
        Ok(())
    }
}

/// A single key-value cache entry for one layer at a given position.
#[derive(Debug, Clone, PartialEq)]
pub struct KvCacheEntry {
    /// Layer index this entry belongs to.
    pub layer: usize,
    /// Current sequence length stored in the cache.
    pub seq_len: usize,
    /// Key tensor data (flattened: `[seq_len, num_kv_heads, head_dim]`).
    pub keys: Vec<f32>,
    /// Value tensor data (flattened: same layout as keys).
    pub values: Vec<f32>,
}

impl KvCacheEntry {
    /// Create an empty cache entry for the given layer.
    pub const fn new(layer: usize) -> Self {
        Self { layer, seq_len: 0, keys: Vec::new(), values: Vec::new() }
    }

    /// Number of elements currently stored.
    pub const fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the entry is empty.
    pub const fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Reset the cache entry, freeing stored data.
    pub fn clear(&mut self) {
        self.seq_len = 0;
        self.keys.clear();
        self.values.clear();
    }
}

// ---------------------------------------------------------------------------
// Attention scores
// ---------------------------------------------------------------------------

/// Raw and normalised attention score container.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionScore {
    /// Raw (pre-softmax) logits. Shape: `[num_heads, seq_len_q, seq_len_kv]`.
    pub raw_scores: Vec<f32>,
    /// Post-softmax attention weights.
    pub weights: Vec<f32>,
    /// Number of heads.
    pub num_heads: usize,
    /// Query sequence length.
    pub seq_len_q: usize,
    /// Key/value sequence length.
    pub seq_len_kv: usize,
}

impl AttentionScore {
    /// Create an empty score container with the given dimensions.
    pub fn new(num_heads: usize, seq_len_q: usize, seq_len_kv: usize) -> Self {
        let n = num_heads * seq_len_q * seq_len_kv;
        Self { raw_scores: vec![0.0; n], weights: vec![0.0; n], num_heads, seq_len_q, seq_len_kv }
    }

    /// Total number of elements per head-slice.
    pub const fn head_slice_len(&self) -> usize {
        self.seq_len_q * self.seq_len_kv
    }

    /// Apply softmax normalisation along the key dimension for every query position
    /// and head, writing results into `self.weights`.
    pub fn apply_softmax(&mut self) {
        let kv = self.seq_len_kv;
        for row_start in (0..self.raw_scores.len()).step_by(kv) {
            let row_end = row_start + kv;
            let row = &self.raw_scores[row_start..row_end];
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let inv_sum = if sum == 0.0 { 0.0 } else { 1.0 / sum };
            for (i, e) in exps.iter().enumerate() {
                self.weights[row_start + i] = e * inv_sum;
            }
        }
    }

    /// Scale raw scores by a factor (typically `1 / sqrt(d_k)`).
    pub fn apply_scale(&mut self, scale: f32) {
        for v in &mut self.raw_scores {
            *v *= scale;
        }
    }

    /// Apply an [`AttentionMask`], setting masked positions to negative infinity.
    pub fn apply_mask(&mut self, mask: &AttentionMask) {
        for h in 0..self.num_heads {
            for q in 0..self.seq_len_q {
                for kv in 0..self.seq_len_kv {
                    if !mask.allows(q, kv, self.seq_len_q) {
                        let idx = h * self.head_slice_len() + q * self.seq_len_kv + kv;
                        self.raw_scores[idx] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flash attention v2 config
// ---------------------------------------------------------------------------

/// Configuration for flash attention v2 tiled computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashAttentionConfig {
    /// Block/tile size along the query dimension.
    pub block_q: usize,
    /// Block/tile size along the key dimension.
    pub block_kv: usize,
    /// Number of warps per thread block.
    pub num_warps: usize,
    /// Number of pipeline stages for software pipelining.
    pub num_stages: usize,
    /// Use causal masking within tiles.
    pub causal: bool,
    /// Maximum sequence length supported by this configuration.
    pub max_seq_len: usize,
}

impl FlashAttentionConfig {
    /// Create a default flash attention v2 config.
    pub const fn new() -> Self {
        Self {
            block_q: 128,
            block_kv: 128,
            num_warps: 4,
            num_stages: 2,
            causal: true,
            max_seq_len: 8192,
        }
    }

    /// Override block sizes.
    #[must_use]
    pub const fn with_block_sizes(mut self, block_q: usize, block_kv: usize) -> Self {
        self.block_q = block_q;
        self.block_kv = block_kv;
        self
    }

    /// Set the number of warps per thread block.
    #[must_use]
    pub const fn with_warps(mut self, warps: usize) -> Self {
        self.num_warps = warps;
        self
    }

    /// Set the number of pipeline stages.
    #[must_use]
    pub const fn with_stages(mut self, stages: usize) -> Self {
        self.num_stages = stages;
        self
    }

    /// The number of query tiles needed for a sequence of length `seq_len`.
    pub const fn num_q_tiles(&self, seq_len: usize) -> usize {
        seq_len.div_ceil(self.block_q)
    }

    /// The number of key/value tiles needed for a sequence of length `seq_len`.
    pub const fn num_kv_tiles(&self, seq_len: usize) -> usize {
        seq_len.div_ceil(self.block_kv)
    }

    /// Validate the flash attention configuration.
    pub fn validate(&self) -> Result<(), AttentionError> {
        if self.block_q == 0 || self.block_kv == 0 {
            return Err(AttentionError::InvalidConfig("block sizes must be > 0".into()));
        }
        if !self.block_q.is_power_of_two() {
            return Err(AttentionError::InvalidConfig("block_q must be a power of two".into()));
        }
        if !self.block_kv.is_power_of_two() {
            return Err(AttentionError::InvalidConfig("block_kv must be a power of two".into()));
        }
        if self.num_warps == 0 {
            return Err(AttentionError::InvalidConfig("num_warps must be > 0".into()));
        }
        if self.num_stages == 0 {
            return Err(AttentionError::InvalidConfig("num_stages must be > 0".into()));
        }
        if self.max_seq_len == 0 {
            return Err(AttentionError::InvalidConfig("max_seq_len must be > 0".into()));
        }
        Ok(())
    }
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Kernel descriptors
// ---------------------------------------------------------------------------

/// Launch descriptor for a standard multi-head attention kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiHeadAttentionKernel {
    /// Attention layer configuration.
    pub config: AttentionConfig,
    /// Mask to apply.
    pub mask: AttentionMask,
    /// Batch size.
    pub batch_size: usize,
    /// Query sequence length.
    pub seq_len_q: usize,
    /// Key/value sequence length.
    pub seq_len_kv: usize,
}

impl MultiHeadAttentionKernel {
    /// Create a new kernel descriptor.
    pub const fn new(config: AttentionConfig, batch_size: usize, seq_len: usize) -> Self {
        let mask = if config.causal { AttentionMask::Causal } else { AttentionMask::None };
        Self { config, mask, batch_size, seq_len_q: seq_len, seq_len_kv: seq_len }
    }

    /// Set a custom mask.
    #[must_use]
    pub fn with_mask(mut self, mask: AttentionMask) -> Self {
        self.mask = mask;
        self
    }

    /// Set different Q and KV sequence lengths (cross-attention).
    #[must_use]
    pub const fn with_kv_len(mut self, seq_len_kv: usize) -> Self {
        self.seq_len_kv = seq_len_kv;
        self
    }

    /// Total FLOPs estimate for the QK^T and softmax·V matmuls.
    pub const fn estimated_flops(&self) -> u64 {
        let b = self.batch_size as u64;
        let h = self.config.num_heads as u64;
        let sq = self.seq_len_q as u64;
        let skv = self.seq_len_kv as u64;
        let d = self.config.head_dim as u64;
        // QK^T: 2*b*h*sq*skv*d  +  softmax*V: 2*b*h*sq*d*skv
        2 * b * h * sq * skv * d + 2 * b * h * sq * d * skv
    }

    /// Validate the kernel descriptor before launch.
    pub fn validate(&self) -> Result<(), AttentionError> {
        self.config.validate()?;
        if self.batch_size == 0 {
            return Err(AttentionError::InvalidConfig("batch_size must be > 0".into()));
        }
        if self.seq_len_q == 0 || self.seq_len_kv == 0 {
            return Err(AttentionError::InvalidConfig("seq_len must be > 0".into()));
        }
        Ok(())
    }
}

/// Launch descriptor for flash attention v2 kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct FlashAttentionV2Kernel {
    /// Per-layer attention geometry.
    pub attention_config: AttentionConfig,
    /// Flash-specific tile/block parameters.
    pub flash_config: FlashAttentionConfig,
    /// Batch size.
    pub batch_size: usize,
    /// Query sequence length.
    pub seq_len_q: usize,
    /// Key/value sequence length.
    pub seq_len_kv: usize,
}

impl FlashAttentionV2Kernel {
    /// Create a new flash attention v2 kernel descriptor.
    pub const fn new(
        attention_config: AttentionConfig,
        flash_config: FlashAttentionConfig,
        batch_size: usize,
        seq_len: usize,
    ) -> Self {
        Self { attention_config, flash_config, batch_size, seq_len_q: seq_len, seq_len_kv: seq_len }
    }

    /// Total number of thread blocks required.
    pub const fn grid_size(&self) -> (usize, usize, usize) {
        let q_tiles = self.flash_config.num_q_tiles(self.seq_len_q);
        (q_tiles, self.attention_config.num_heads, self.batch_size)
    }

    /// Shared memory required per thread block in bytes (approximation).
    pub const fn shared_mem_bytes(&self) -> usize {
        let d = self.attention_config.head_dim;
        let bq = self.flash_config.block_q;
        let bkv = self.flash_config.block_kv;
        // Q tile + K tile + V tile + score tile, all in f32
        (bq * d + bkv * d + bkv * d + bq * bkv) * 4
    }

    /// Validate both attention and flash configs.
    pub fn validate(&self) -> Result<(), AttentionError> {
        self.attention_config.validate()?;
        self.flash_config.validate()?;
        if self.batch_size == 0 {
            return Err(AttentionError::InvalidConfig("batch_size must be > 0".into()));
        }
        if self.seq_len_q == 0 || self.seq_len_kv == 0 {
            return Err(AttentionError::InvalidConfig("seq_len must be > 0".into()));
        }
        if self.seq_len_q > self.flash_config.max_seq_len
            || self.seq_len_kv > self.flash_config.max_seq_len
        {
            return Err(AttentionError::InvalidConfig(
                "seq_len exceeds flash attention max_seq_len".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Attention output
// ---------------------------------------------------------------------------

/// Output container from an attention forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionOutput {
    /// The attention output tensor, flattened: `[batch, seq_len_q, model_dim]`.
    pub data: Vec<f32>,
    /// Batch size.
    pub batch_size: usize,
    /// Query sequence length.
    pub seq_len: usize,
    /// Model dimension.
    pub model_dim: usize,
}

impl AttentionOutput {
    /// Create an output filled with zeros.
    pub fn zeros(batch_size: usize, seq_len: usize, model_dim: usize) -> Self {
        Self { data: vec![0.0; batch_size * seq_len * model_dim], batch_size, seq_len, model_dim }
    }

    /// Total number of elements.
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the output is empty.
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Validate that the stored data matches the declared dimensions.
    pub fn validate(&self) -> Result<(), AttentionError> {
        let expected = self.batch_size * self.seq_len * self.model_dim;
        if self.data.len() != expected {
            return Err(AttentionError::ShapeMismatch {
                expected: vec![self.batch_size, self.seq_len, self.model_dim],
                actual: vec![self.data.len()],
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur in attention kernel configuration or execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionError {
    /// A configuration parameter is invalid.
    InvalidConfig(String),
    /// Tensor shapes do not match expectations.
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        actual: Vec<usize>,
    },
    /// The requested CUDA device is not available.
    DeviceUnavailable(String),
    /// A kernel launch failed.
    KernelLaunchFailed(String),
    /// Out of GPU memory.
    OutOfMemory {
        /// Bytes requested.
        requested: usize,
        /// Bytes available.
        available: usize,
    },
}

impl fmt::Display for AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid attention config: {msg}"),
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected:?}, got {actual:?}")
            }
            Self::DeviceUnavailable(msg) => write!(f, "CUDA device unavailable: {msg}"),
            Self::KernelLaunchFailed(msg) => write!(f, "kernel launch failed: {msg}"),
            Self::OutOfMemory { requested, available } => {
                write!(f, "out of GPU memory: requested {requested} bytes, {available} available")
            }
        }
    }
}

impl std::error::Error for AttentionError {}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- AttentionConfig tests ------------------------------------------------

    #[test]
    fn config_new_defaults() {
        let c = AttentionConfig::new(8, 64);
        assert_eq!(c.num_heads, 8);
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.model_dim, 512);
        assert_eq!(c.num_kv_heads, 8);
        assert!(c.causal);
        assert_eq!(c.dropout_prob, 0.0);
    }

    #[test]
    fn config_scale_default() {
        let c = AttentionConfig::new(1, 64);
        let expected = 1.0 / (64.0_f32).sqrt();
        assert!((c.scale - expected).abs() < 1e-6);
    }

    #[test]
    fn config_with_gqa() {
        let c = AttentionConfig::new(32, 128).with_gqa(8);
        assert_eq!(c.num_kv_heads, 8);
    }

    #[test]
    fn config_with_scale_override() {
        let c = AttentionConfig::new(8, 64).with_scale(0.5);
        assert!((c.scale - 0.5).abs() < 1e-6);
    }

    #[test]
    fn config_with_causal_false() {
        let c = AttentionConfig::new(8, 64).with_causal(false);
        assert!(!c.causal);
    }

    #[test]
    fn config_with_dropout() {
        let c = AttentionConfig::new(8, 64).with_dropout(0.1);
        assert!((c.dropout_prob - 0.1).abs() < 1e-6);
    }

    #[test]
    fn config_validate_ok() {
        assert!(AttentionConfig::new(8, 64).validate().is_ok());
    }

    #[test]
    fn config_validate_zero_heads() {
        let mut c = AttentionConfig::new(8, 64);
        c.num_heads = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_zero_head_dim() {
        let mut c = AttentionConfig::new(8, 64);
        c.head_dim = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_model_dim_mismatch() {
        let mut c = AttentionConfig::new(8, 64);
        c.model_dim = 999;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_kv_heads_zero() {
        let mut c = AttentionConfig::new(8, 64);
        c.num_kv_heads = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_kv_heads_not_divisor() {
        let mut c = AttentionConfig::new(8, 64);
        c.num_kv_heads = 3;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_negative_scale() {
        let c = AttentionConfig::new(8, 64).with_scale(-1.0);
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_nan_scale() {
        let c = AttentionConfig::new(8, 64).with_scale(f32::NAN);
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_inf_scale() {
        let c = AttentionConfig::new(8, 64).with_scale(f32::INFINITY);
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_dropout_too_high() {
        let c = AttentionConfig::new(8, 64).with_dropout(1.5);
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_dropout_negative() {
        let c = AttentionConfig::new(8, 64).with_dropout(-0.1);
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_dropout_boundary_one() {
        let c = AttentionConfig::new(8, 64).with_dropout(1.0);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn config_gqa_divisor_ok() {
        let c = AttentionConfig::new(32, 64).with_gqa(8);
        assert!(c.validate().is_ok());
    }

    // -- AttentionMask tests --------------------------------------------------

    #[test]
    fn mask_none_allows_all() {
        let m = AttentionMask::None;
        assert!(m.allows(0, 5, 6));
        assert!(m.allows(3, 0, 6));
    }

    #[test]
    fn mask_causal_lower_triangle() {
        let m = AttentionMask::Causal;
        assert!(m.allows(2, 0, 4));
        assert!(m.allows(2, 2, 4));
        assert!(!m.allows(2, 3, 4));
    }

    #[test]
    fn mask_causal_first_position() {
        let m = AttentionMask::Causal;
        assert!(m.allows(0, 0, 4));
        assert!(!m.allows(0, 1, 4));
    }

    #[test]
    fn mask_custom_basic() {
        let custom = vec![vec![true, false], vec![true, true]];
        let m = AttentionMask::Custom(custom);
        assert!(m.allows(0, 0, 2));
        assert!(!m.allows(0, 1, 2));
        assert!(m.allows(1, 0, 2));
        assert!(m.allows(1, 1, 2));
    }

    #[test]
    fn mask_custom_out_of_bounds() {
        let custom = vec![vec![true]];
        let m = AttentionMask::Custom(custom);
        assert!(!m.allows(0, 5, 1));
        assert!(!m.allows(5, 0, 1));
    }

    #[test]
    fn mask_sliding_window() {
        let m = AttentionMask::SlidingWindow { window_size: 3 };
        assert!(m.allows(5, 5, 8));
        assert!(m.allows(5, 4, 8));
        assert!(m.allows(5, 3, 8));
        assert!(!m.allows(5, 2, 8));
        assert!(!m.allows(5, 6, 8));
    }

    #[test]
    fn mask_sliding_window_size_one() {
        let m = AttentionMask::SlidingWindow { window_size: 1 };
        assert!(m.allows(3, 3, 4));
        assert!(!m.allows(3, 2, 4));
    }

    #[test]
    fn mask_prefix_bidirectional() {
        let m = AttentionMask::Prefix { prefix_len: 3 };
        // Prefix positions attend everywhere
        assert!(m.allows(0, 5, 8));
        assert!(m.allows(2, 7, 8));
        // Non-prefix positions are causal
        assert!(m.allows(4, 3, 8));
        assert!(!m.allows(4, 5, 8));
    }

    #[test]
    fn mask_count_allowed_none() {
        let m = AttentionMask::None;
        assert_eq!(m.count_allowed(0, 10, 10), 10);
    }

    #[test]
    fn mask_count_allowed_causal() {
        let m = AttentionMask::Causal;
        assert_eq!(m.count_allowed(0, 5, 5), 1);
        assert_eq!(m.count_allowed(4, 5, 5), 5);
    }

    // -- KvPrecision tests ----------------------------------------------------

    #[test]
    fn kv_precision_display() {
        assert_eq!(KvPrecision::F32.to_string(), "f32");
        assert_eq!(KvPrecision::F16.to_string(), "f16");
        assert_eq!(KvPrecision::Bf16.to_string(), "bf16");
        assert_eq!(KvPrecision::Int8.to_string(), "int8");
    }

    #[test]
    fn kv_precision_eq() {
        assert_eq!(KvPrecision::F32, KvPrecision::F32);
        assert_ne!(KvPrecision::F16, KvPrecision::Bf16);
    }

    // -- KvCacheConfig tests --------------------------------------------------

    #[test]
    fn kv_cache_config_new() {
        let c = KvCacheConfig::new(2048, 32, 8, 64);
        assert_eq!(c.max_seq_len, 2048);
        assert_eq!(c.num_layers, 32);
        assert_eq!(c.num_kv_heads, 8);
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.precision, KvPrecision::F16);
        assert!(!c.paged);
    }

    #[test]
    fn kv_cache_layer_bytes_f16() {
        let c = KvCacheConfig::new(1024, 1, 8, 64);
        // 2 (k+v) * 1024 * 8 * 64 * 2 bytes
        assert_eq!(c.layer_bytes(), 2 * 1024 * 8 * 64 * 2);
    }

    #[test]
    fn kv_cache_layer_bytes_f32() {
        let mut c = KvCacheConfig::new(512, 1, 4, 32);
        c.precision = KvPrecision::F32;
        assert_eq!(c.layer_bytes(), 2 * 512 * 4 * 32 * 4);
    }

    #[test]
    fn kv_cache_layer_bytes_int8() {
        let mut c = KvCacheConfig::new(256, 1, 2, 16);
        c.precision = KvPrecision::Int8;
        assert_eq!(c.layer_bytes(), 2 * 256 * 2 * 16 * 1);
    }

    #[test]
    fn kv_cache_total_bytes() {
        let c = KvCacheConfig::new(1024, 32, 8, 64);
        assert_eq!(c.total_bytes(), 32 * c.layer_bytes());
    }

    #[test]
    fn kv_cache_validate_ok() {
        assert!(KvCacheConfig::new(2048, 32, 8, 64).validate().is_ok());
    }

    #[test]
    fn kv_cache_validate_zero_seq_len() {
        let c = KvCacheConfig::new(0, 32, 8, 64);
        assert!(c.validate().is_err());
    }

    #[test]
    fn kv_cache_validate_zero_layers() {
        let c = KvCacheConfig::new(2048, 0, 8, 64);
        assert!(c.validate().is_err());
    }

    #[test]
    fn kv_cache_validate_paged_zero_block() {
        let mut c = KvCacheConfig::new(2048, 32, 8, 64);
        c.paged = true;
        c.block_size = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn kv_cache_validate_paged_ok() {
        let mut c = KvCacheConfig::new(2048, 32, 8, 64);
        c.paged = true;
        c.block_size = 128;
        assert!(c.validate().is_ok());
    }

    // -- KvCacheEntry tests ---------------------------------------------------

    #[test]
    fn kv_entry_new_empty() {
        let e = KvCacheEntry::new(5);
        assert_eq!(e.layer, 5);
        assert_eq!(e.seq_len, 0);
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn kv_entry_with_data() {
        let mut e = KvCacheEntry::new(0);
        e.keys = vec![1.0, 2.0, 3.0];
        e.values = vec![4.0, 5.0, 6.0];
        e.seq_len = 1;
        assert_eq!(e.len(), 3);
        assert!(!e.is_empty());
    }

    #[test]
    fn kv_entry_clear() {
        let mut e = KvCacheEntry::new(0);
        e.keys = vec![1.0];
        e.values = vec![2.0];
        e.seq_len = 1;
        e.clear();
        assert!(e.is_empty());
        assert_eq!(e.seq_len, 0);
    }

    // -- AttentionScore tests -------------------------------------------------

    #[test]
    fn score_new_dimensions() {
        let s = AttentionScore::new(4, 8, 16);
        assert_eq!(s.raw_scores.len(), 4 * 8 * 16);
        assert_eq!(s.weights.len(), 4 * 8 * 16);
        assert_eq!(s.num_heads, 4);
        assert_eq!(s.seq_len_q, 8);
        assert_eq!(s.seq_len_kv, 16);
    }

    #[test]
    fn score_head_slice_len() {
        let s = AttentionScore::new(2, 3, 5);
        assert_eq!(s.head_slice_len(), 15);
    }

    #[test]
    fn score_apply_scale() {
        let mut s = AttentionScore::new(1, 1, 3);
        s.raw_scores = vec![2.0, 4.0, 6.0];
        s.apply_scale(0.5);
        assert_eq!(s.raw_scores, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn score_softmax_uniform() {
        let mut s = AttentionScore::new(1, 1, 3);
        s.raw_scores = vec![0.0, 0.0, 0.0];
        s.apply_softmax();
        for &w in &s.weights {
            assert!((w - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn score_softmax_sum_to_one() {
        let mut s = AttentionScore::new(1, 1, 4);
        s.raw_scores = vec![1.0, 2.0, 3.0, 4.0];
        s.apply_softmax();
        let sum: f32 = s.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn score_softmax_monotonic() {
        let mut s = AttentionScore::new(1, 1, 3);
        s.raw_scores = vec![1.0, 2.0, 3.0];
        s.apply_softmax();
        assert!(s.weights[0] < s.weights[1]);
        assert!(s.weights[1] < s.weights[2]);
    }

    #[test]
    fn score_softmax_multi_row() {
        let mut s = AttentionScore::new(1, 2, 2);
        s.raw_scores = vec![0.0, 0.0, 1.0, 0.0];
        s.apply_softmax();
        // First row: uniform
        assert!((s.weights[0] - 0.5).abs() < 1e-5);
        assert!((s.weights[1] - 0.5).abs() < 1e-5);
        // Second row: not uniform
        assert!(s.weights[2] > s.weights[3]);
        let sum: f32 = s.weights[2..4].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn score_softmax_large_values() {
        let mut s = AttentionScore::new(1, 1, 2);
        s.raw_scores = vec![1000.0, 1001.0];
        s.apply_softmax();
        let sum: f32 = s.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn score_apply_mask_causal() {
        let mut s = AttentionScore::new(1, 3, 3);
        s.raw_scores = vec![1.0; 9];
        s.apply_mask(&AttentionMask::Causal);
        // Position (0,1), (0,2), (1,2) should be -inf
        assert!(s.raw_scores[1].is_infinite());
        assert!(s.raw_scores[2].is_infinite());
        assert!(s.raw_scores[5].is_infinite());
        // Diagonal and below should remain 1.0
        assert_eq!(s.raw_scores[0], 1.0);
        assert_eq!(s.raw_scores[3], 1.0);
        assert_eq!(s.raw_scores[4], 1.0);
    }

    #[test]
    fn score_apply_mask_none() {
        let mut s = AttentionScore::new(1, 2, 2);
        s.raw_scores = vec![1.0; 4];
        s.apply_mask(&AttentionMask::None);
        assert!(s.raw_scores.iter().all(|&v| v == 1.0));
    }

    // -- FlashAttentionConfig tests -------------------------------------------

    #[test]
    fn flash_config_defaults() {
        let f = FlashAttentionConfig::new();
        assert_eq!(f.block_q, 128);
        assert_eq!(f.block_kv, 128);
        assert_eq!(f.num_warps, 4);
        assert_eq!(f.num_stages, 2);
        assert!(f.causal);
        assert_eq!(f.max_seq_len, 8192);
    }

    #[test]
    fn flash_config_default_trait() {
        let f = FlashAttentionConfig::default();
        assert_eq!(f, FlashAttentionConfig::new());
    }

    #[test]
    fn flash_config_with_block_sizes() {
        let f = FlashAttentionConfig::new().with_block_sizes(64, 256);
        assert_eq!(f.block_q, 64);
        assert_eq!(f.block_kv, 256);
    }

    #[test]
    fn flash_config_with_warps() {
        let f = FlashAttentionConfig::new().with_warps(8);
        assert_eq!(f.num_warps, 8);
    }

    #[test]
    fn flash_config_with_stages() {
        let f = FlashAttentionConfig::new().with_stages(4);
        assert_eq!(f.num_stages, 4);
    }

    #[test]
    fn flash_num_q_tiles_exact() {
        let f = FlashAttentionConfig::new().with_block_sizes(64, 64);
        assert_eq!(f.num_q_tiles(256), 4);
    }

    #[test]
    fn flash_num_q_tiles_remainder() {
        let f = FlashAttentionConfig::new().with_block_sizes(64, 64);
        assert_eq!(f.num_q_tiles(100), 2);
    }

    #[test]
    fn flash_num_kv_tiles() {
        let f = FlashAttentionConfig::new().with_block_sizes(64, 128);
        assert_eq!(f.num_kv_tiles(512), 4);
        assert_eq!(f.num_kv_tiles(129), 2);
    }

    #[test]
    fn flash_validate_ok() {
        assert!(FlashAttentionConfig::new().validate().is_ok());
    }

    #[test]
    fn flash_validate_zero_block_q() {
        let mut f = FlashAttentionConfig::new();
        f.block_q = 0;
        assert!(f.validate().is_err());
    }

    #[test]
    fn flash_validate_non_power_of_two_block_q() {
        let mut f = FlashAttentionConfig::new();
        f.block_q = 100;
        assert!(f.validate().is_err());
    }

    #[test]
    fn flash_validate_non_power_of_two_block_kv() {
        let mut f = FlashAttentionConfig::new();
        f.block_kv = 65;
        assert!(f.validate().is_err());
    }

    #[test]
    fn flash_validate_zero_warps() {
        let mut f = FlashAttentionConfig::new();
        f.num_warps = 0;
        assert!(f.validate().is_err());
    }

    #[test]
    fn flash_validate_zero_stages() {
        let mut f = FlashAttentionConfig::new();
        f.num_stages = 0;
        assert!(f.validate().is_err());
    }

    #[test]
    fn flash_validate_zero_max_seq() {
        let mut f = FlashAttentionConfig::new();
        f.max_seq_len = 0;
        assert!(f.validate().is_err());
    }

    // -- MultiHeadAttentionKernel tests ---------------------------------------

    #[test]
    fn mha_kernel_new_causal() {
        let c = AttentionConfig::new(8, 64);
        let k = MultiHeadAttentionKernel::new(c, 4, 128);
        assert_eq!(k.batch_size, 4);
        assert_eq!(k.seq_len_q, 128);
        assert_eq!(k.seq_len_kv, 128);
        assert_eq!(k.mask, AttentionMask::Causal);
    }

    #[test]
    fn mha_kernel_non_causal() {
        let c = AttentionConfig::new(8, 64).with_causal(false);
        let k = MultiHeadAttentionKernel::new(c, 1, 64);
        assert_eq!(k.mask, AttentionMask::None);
    }

    #[test]
    fn mha_kernel_custom_mask() {
        let c = AttentionConfig::new(1, 64);
        let k = MultiHeadAttentionKernel::new(c, 1, 2)
            .with_mask(AttentionMask::SlidingWindow { window_size: 4 });
        assert_eq!(k.mask, AttentionMask::SlidingWindow { window_size: 4 });
    }

    #[test]
    fn mha_kernel_cross_attention() {
        let c = AttentionConfig::new(8, 64);
        let k = MultiHeadAttentionKernel::new(c, 1, 32).with_kv_len(128);
        assert_eq!(k.seq_len_q, 32);
        assert_eq!(k.seq_len_kv, 128);
    }

    #[test]
    fn mha_kernel_estimated_flops() {
        let c = AttentionConfig::new(1, 64);
        let k = MultiHeadAttentionKernel::new(c, 1, 10);
        // 2*1*1*10*10*64 + 2*1*1*10*64*10 = 12800 + 12800 = 25600
        assert_eq!(k.estimated_flops(), 25600);
    }

    #[test]
    fn mha_kernel_validate_ok() {
        let c = AttentionConfig::new(8, 64);
        let k = MultiHeadAttentionKernel::new(c, 1, 32);
        assert!(k.validate().is_ok());
    }

    #[test]
    fn mha_kernel_validate_zero_batch() {
        let c = AttentionConfig::new(8, 64);
        let mut k = MultiHeadAttentionKernel::new(c, 1, 32);
        k.batch_size = 0;
        assert!(k.validate().is_err());
    }

    #[test]
    fn mha_kernel_validate_zero_seq() {
        let c = AttentionConfig::new(8, 64);
        let mut k = MultiHeadAttentionKernel::new(c, 1, 32);
        k.seq_len_q = 0;
        assert!(k.validate().is_err());
    }

    // -- FlashAttentionV2Kernel tests -----------------------------------------

    #[test]
    fn flash_kernel_new() {
        let ac = AttentionConfig::new(8, 64);
        let fc = FlashAttentionConfig::new();
        let k = FlashAttentionV2Kernel::new(ac, fc, 2, 256);
        assert_eq!(k.batch_size, 2);
        assert_eq!(k.seq_len_q, 256);
        assert_eq!(k.seq_len_kv, 256);
    }

    #[test]
    fn flash_kernel_grid_size() {
        let ac = AttentionConfig::new(8, 64);
        let fc = FlashAttentionConfig::new().with_block_sizes(64, 64);
        let k = FlashAttentionV2Kernel::new(ac, fc, 2, 256);
        assert_eq!(k.grid_size(), (4, 8, 2));
    }

    #[test]
    fn flash_kernel_shared_mem() {
        let ac = AttentionConfig::new(1, 64);
        let fc = FlashAttentionConfig::new().with_block_sizes(64, 64);
        let k = FlashAttentionV2Kernel::new(ac, fc, 1, 64);
        // (64*64 + 64*64 + 64*64 + 64*64) * 4 = 4*4096*4 = 65536
        assert_eq!(k.shared_mem_bytes(), 65536);
    }

    #[test]
    fn flash_kernel_validate_ok() {
        let ac = AttentionConfig::new(8, 64);
        let fc = FlashAttentionConfig::new();
        let k = FlashAttentionV2Kernel::new(ac, fc, 1, 1024);
        assert!(k.validate().is_ok());
    }

    #[test]
    fn flash_kernel_validate_seq_exceeds_max() {
        let ac = AttentionConfig::new(8, 64);
        let mut fc = FlashAttentionConfig::new();
        fc.max_seq_len = 512;
        let k = FlashAttentionV2Kernel::new(ac, fc, 1, 1024);
        assert!(k.validate().is_err());
    }

    #[test]
    fn flash_kernel_validate_zero_batch() {
        let ac = AttentionConfig::new(8, 64);
        let fc = FlashAttentionConfig::new();
        let mut k = FlashAttentionV2Kernel::new(ac, fc, 1, 64);
        k.batch_size = 0;
        assert!(k.validate().is_err());
    }

    // -- AttentionOutput tests ------------------------------------------------

    #[test]
    fn output_zeros() {
        let o = AttentionOutput::zeros(2, 16, 512);
        assert_eq!(o.len(), 2 * 16 * 512);
        assert!(!o.is_empty());
        assert!(o.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn output_validate_ok() {
        let o = AttentionOutput::zeros(1, 4, 64);
        assert!(o.validate().is_ok());
    }

    #[test]
    fn output_validate_mismatch() {
        let mut o = AttentionOutput::zeros(1, 4, 64);
        o.data.push(1.0);
        assert!(o.validate().is_err());
    }

    #[test]
    fn output_empty_zero_dim() {
        let o = AttentionOutput::zeros(0, 0, 0);
        assert!(o.is_empty());
    }

    // -- AttentionError tests -------------------------------------------------

    #[test]
    fn error_display_invalid_config() {
        let e = AttentionError::InvalidConfig("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn error_display_shape_mismatch() {
        let e = AttentionError::ShapeMismatch { expected: vec![1, 2], actual: vec![3] };
        let s = e.to_string();
        assert!(s.contains("[1, 2]"));
        assert!(s.contains("[3]"));
    }

    #[test]
    fn error_display_device_unavailable() {
        let e = AttentionError::DeviceUnavailable("gpu0".into());
        assert!(e.to_string().contains("gpu0"));
    }

    #[test]
    fn error_display_kernel_launch() {
        let e = AttentionError::KernelLaunchFailed("timeout".into());
        assert!(e.to_string().contains("timeout"));
    }

    #[test]
    fn error_display_oom() {
        let e = AttentionError::OutOfMemory { requested: 1024, available: 512 };
        let s = e.to_string();
        assert!(s.contains("1024"));
        assert!(s.contains("512"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(AttentionError::InvalidConfig("test".into()));
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn error_eq() {
        let a = AttentionError::InvalidConfig("x".into());
        let b = AttentionError::InvalidConfig("x".into());
        assert_eq!(a, b);
    }

    #[test]
    fn error_ne() {
        let a = AttentionError::InvalidConfig("x".into());
        let b = AttentionError::InvalidConfig("y".into());
        assert_ne!(a, b);
    }
}
